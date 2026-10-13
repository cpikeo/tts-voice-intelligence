#!/usr/bin/env python3
"""Reproducible model-free microbenchmarks for hot paths (not TTS quality/RTF evidence).

Examples:
  python tests/benchmark_hotpaths.py --mode pipeline --segments 512
  python tests/benchmark_hotpaths.py --mode qa --qa-seconds 150
  python tests/benchmark_hotpaths.py --mode asset --asset-mib 32 --segments 64
  python tests/benchmark_hotpaths.py --mode io

For a before/after comparison, run the same script against each checkout with --scripts-dir.
"""
import argparse
import importlib.metadata
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def _median_seconds(fn, repeats):
    values = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values), values, result


def _rss_kib():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value / 1024) if sys.platform == "darwin" else int(value)
    except (ImportError, AttributeError, ValueError):
        return None


def _versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__}
    for name in ("soundfile", "scipy"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _bench_pipeline(tts, segments, repeats):
    sr = 22050
    t = np.arange(sr, dtype=np.float32) / sr
    chunk = (0.12 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    class FixedEngine(tts.Engine):
        name = "benchmark_fixture"
        sample_rate = sr
        language = "en"

        def synthesize(self, _text):
            return chunk.copy()

    text = "".join(f"Sentence {i} ends here. " for i in range(segments))
    engine = FixedEngine()
    pauses = {"para": 700, "sentence": 400, "clause": 200, "none": 0}

    def run():
        return tts.synthesize_text(engine, text, normalize=False, language="en",
                                  processing="minimal", pauses=pauses)

    median, values, (audio, _rate, report) = _median_seconds(run, repeats)
    return {
        "mode": "pipeline_fixture",
        "segments": report["segment_count"],
        "output_seconds": report["duration_seconds"],
        "output_bytes": int(audio.nbytes),
        "output_dtype": str(audio.dtype),
        "wall_seconds": [round(v, 6) for v in values],
        "median_wall_seconds": round(median, 6),
        "max_rss_kib": _rss_kib(),
        "boundary_repairs": report["boundary_repair"]["repairs"],
        "segment_rms_std_db": report["segment_rms_std_db"],
        "real_engine": False,
        "synthesis_rtf": None,
    }


def _bench_qa(qa, duration_seconds, repeats):
    sr = 22050
    t = np.arange(sr, dtype=np.float32) / sr
    tone = (0.12 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    silence = np.zeros(sr // 4, dtype=np.float32)
    blocks = []
    cycle_seconds = 1.0 + len(silence) / sr
    for _ in range(max(1, round(duration_seconds / cycle_seconds))):
        blocks.extend((tone, silence))
    audio = np.concatenate(blocks)
    info = type("Info", (), {"channels": 1, "subtype": "FLOAT"})()
    median, values, checks = _median_seconds(lambda: qa.run_signal_checks(audio, sr, info), repeats)
    return {
        "mode": "qa_fixture",
        "input_seconds": round(len(audio) / sr, 6),
        "input_bytes": int(audio.nbytes),
        "wall_seconds": [round(v, 6) for v in values],
        "median_wall_seconds": round(median, 6),
        "max_rss_kib": _rss_kib(),
        "fail_count": sum(c["status"] == "fail" for c in checks),
        "warn_count": sum(c["status"] == "warn" for c in checks),
        "skip_count": sum(c["status"] == "skip" for c in checks),
        "real_audio": False,
    }


def _bench_io(qa):
    import contextlib
    import io
    import soundfile as sf
    sr = 22050
    t = np.arange(sr * 2, dtype=np.float32) / sr
    mono = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="tts-decoder-bench-") as td:
        mono_path = Path(td) / "mono.wav"
        stereo_path = Path(td) / "stereo.wav"
        sf.write(mono_path, mono, sr, subtype="PCM_16")
        sf.write(stereo_path, np.stack((mono, mono), axis=1), sr, subtype="PCM_16")
        original = sf.SoundFile
        counts = {"opens": 0}

        class CountedSoundFile(original):
            def __init__(self, *args, **kwargs):
                counts["opens"] += 1
                super().__init__(*args, **kwargs)

        sf.SoundFile = CountedSoundFile
        try:
            counts["opens"] = 0
            qa.load_mono(mono_path)
            mono_load_opens = counts["opens"]
            counts["opens"] = 0
            with contextlib.redirect_stdout(io.StringIO()):
                qa.main([str(stereo_path)])
            stereo_qa_opens = counts["opens"]
        finally:
            sf.SoundFile = original
    return {
        "mode": "decoder_open_fixture",
        "mono_load_soundfile_opens": mono_load_opens,
        "stereo_qa_soundfile_opens": stereo_qa_opens,
        "report_hash_pass_included": False,
        "real_audio": False,
    }


def _bench_asset(engines, size_mib, segment_count, repeats):
    block = b"0123456789abcdef" * (1 << 16)  # 1 MiB deterministic test data
    with tempfile.TemporaryDirectory(prefix="tts-asset-bench-") as td:
        asset = Path(td) / "bound-asset.bin"
        with asset.open("wb") as f:
            for _ in range(size_mib):
                f.write(block)
        _raw, fingerprint = engines._read_and_fingerprint(asset)

        def verify_n(n):
            def run():
                for _ in range(n):
                    engines.verify_bound_asset(fingerprint)
            return run

        per_segment_median, per_segment_values, _ = _median_seconds(
            verify_n(segment_count), repeats)
        batch_median, batch_values, _ = _median_seconds(verify_n(2), repeats)
    return {
        "mode": "asset_hash_fixture",
        "asset_mib": size_mib,
        "full_sha_checks_before": segment_count,
        "full_sha_checks_per_batch_after": 2,
        "before_seconds": round(per_segment_median, 6),
        "before_samples_seconds": [round(v, 6) for v in per_segment_values],
        "after_seconds": round(batch_median, 6),
        "after_samples_seconds": [round(v, 6) for v in batch_values],
        "bytes_hashed_mib_before": size_mib * segment_count,
        "bytes_hashed_mib_after": size_mib * 2,
        "real_model": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("pipeline", "qa", "asset", "io"), required=True)
    parser.add_argument("--scripts-dir", default=str(Path(__file__).resolve().parents[1] / "scripts"),
                        help="scripts directory to benchmark; use a different checkout for baseline")
    parser.add_argument("--segments", type=int, default=64)
    parser.add_argument("--qa-seconds", type=int, default=150)
    parser.add_argument("--asset-mib", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if min(args.segments, args.qa_seconds, args.asset_mib, args.repeats) <= 0:
        parser.error("all workload sizes and repeats must be positive")

    sys.path.insert(0, str(Path(args.scripts_dir).resolve()))
    import engines
    import qa
    import tts

    if args.mode == "pipeline":
        result = _bench_pipeline(tts, args.segments, args.repeats)
    elif args.mode == "qa":
        result = _bench_qa(qa, args.qa_seconds, args.repeats)
    elif args.mode == "asset":
        result = _bench_asset(engines, args.asset_mib, args.segments, args.repeats)
    else:
        result = _bench_io(qa)
    result["environment"] = _versions()
    result["scripts_dir"] = str(Path(args.scripts_dir).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

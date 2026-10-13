#!/usr/bin/env python3
"""Controlled A/B for post-processing: same real engine segments, different processing only.

Engine output is non-deterministic for Piper (noise is sampled inside the ONNX graph), so
pipeline comparisons must reuse ONE captured set of raw segments. This tool:

  capture  - run the real engine once per case/repeat and store raw per-segment arrays + hashes
  compare  - replay those raw segments through several processing variants (synthesize_text with a
             replay engine), write each WAV, and compute objective metrics (no human scores)

Variants (only processing parameters differ):
  A_no_pause_no_repair : previous CLI default (minimal, pauses={}, no boundary repair)
  B_pauses_only        : minimal + default punctuation pauses, repair disabled
  C_pauses_repair      : current CLI default (pauses + conditional boundary repair)
  D_legacy_reference   : legacy (per-segment RMS alignment, 12 ms fades) for reference only

Objective metrics are signal/timing proxies. They do NOT measure naturalness; human listening
must be performed separately (see blindtest.py) and is recorded as not_performed here.
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qa  # noqa: E402
import tts  # noqa: E402
from engines import EspeakEngine, PiperEngine  # noqa: E402

DEFAULT_PAUSES = {"para": 700, "sentence": 400, "clause": 200, "none": 0}
VARIANTS = {
    "A_no_pause_no_repair": dict(processing="minimal", pauses={}, boundary_repair_enabled=False),
    "B_pauses_only": dict(processing="minimal", pauses=dict(DEFAULT_PAUSES), boundary_repair_enabled=False),
    "C_pauses_repair": dict(processing="minimal", pauses=dict(DEFAULT_PAUSES), boundary_repair_enabled=True),
    "D_legacy_reference": dict(processing="legacy", pauses=dict(DEFAULT_PAUSES), boundary_repair_enabled=True),
}


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------ capture

def cmd_capture(args):
    cases = {c["id"]: c for c in (json.loads(l) for l in (ROOT / "tests/cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    ids = [i for i in args.ids.split(",") if i]
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print("capture 输出目录必须为空（保护既有原始证据）", file=sys.stderr)
        return 4
    out.mkdir(parents=True)
    model_sha = None
    if args.engine == "piper" and not args.model:
        print("piper 捕获需要 --model", file=sys.stderr)
        return 2
    if args.engine == "piper":
        model_sha = tts.model_provenance(args.model)["sha256"]
    engines_by_lang = {}

    def engine_for(language):
        if language not in engines_by_lang:
            if args.engine == "piper":
                engines_by_lang[language] = PiperEngine(model=args.model, language=language, timeout=args.timeout)
            else:
                engines_by_lang[language] = EspeakEngine(language=language, timeout=args.timeout)
        return engines_by_lang[language]

    meta = []
    for r in range(args.repeat):
        for cid in ids:
            c = cases[cid]
            language = args.language if args.engine == "piper" else c.get("language", args.language)
            eng = engine_for(language)
            norm = tts.normalize_text(c["text"], eng.language, None)
            segs = tts.segment(norm, 90)
            raws = [np.asarray(eng.synthesize(s.text), dtype=np.float32) for s in segs]
            name = f"{args.engine}.{cid}.r{r}"
            np.savez(out / f"{name}.npz", **{f"seg{i:03d}": x for i, x in enumerate(raws)})
            item = dict(name=name, engine=args.engine, case=cid, repeat=r, language=eng.language,
                        sr=int(eng.sample_rate), text=c["text"], normalized=norm,
                        seg_texts=[s.text for s in segs], seg_pause=[s.pause_after for s in segs],
                        raw_sha256=[_sha_bytes(x.tobytes()) for x in raws], model_sha256=model_sha)
            meta.append(item)
            print(f"captured {name}: {len(segs)} segments", flush=True)
    (out / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


# ------------------------------------------------------------------ compare

class ReplayEngine:
    """Returns previously captured raw segments in order; asserts the text sequence matches."""

    name = "replay"
    capabilities = {"rate": False}

    def __init__(self, item, raws):
        self.sample_rate = item["sr"]
        self.language = item["language"]
        self._texts = item["seg_texts"]
        self._raws = raws
        self._i = 0

    def available(self):
        return True, "replay of captured real engine segments"

    def synthesize(self, text):
        if self._i >= len(self._texts) or text != self._texts[self._i]:
            raise AssertionError(f"segment mismatch at {self._i}: {text!r}")
        out = self._raws[self._i]
        self._i += 1
        return out


def _load_raws(path, n):
    with np.load(path) as z:
        return [z[f"seg{i:03d}"] for i in range(n)]


def _join_gaps_ms(x, starts, sr, thr_db=-50.0):
    thr = 10 ** (thr_db / 20)
    out = []
    for s in starts:
        if s <= 0 or s >= len(x):
            continue
        i, left = s - 1, 0
        while i >= 0 and abs(x[i]) <= thr:
            i, left = i - 1, left + 1
        j, right = s, 0
        while j < len(x) and abs(x[j]) <= thr:
            j, right = j + 1, right + 1
        out.append((left + right) * 1000.0 / sr)
    return out


def _metrics(path, audio, sr, rep, language, text_expected, asr_model):
    info = type("I", (), {"channels": 1, "subtype": "PCM_24"})()
    checks = {c["name"]: c for c in qa.run_signal_checks(audio, sr, info)}
    x = audio.astype(np.float64)
    starts = [s["start_sample"] for s in rep["segments"]]
    jg = _join_gaps_ms(x, starts, sr)
    kinds = [s["pause_after"] for s in rep["segments"][:-1]]
    jumps = qa.boundary_jumps(x, sr)
    m = dict(
        seconds=round(len(x) / sr, 3),
        peak_dbfs=round(float(20 * np.log10(np.max(np.abs(x)) + 1e-12)), 2),
        true_peak_dbtp=checks["true_peak_dbtp"]["value"],
        lufs=checks["integrated_loudness_lufs"]["value"],
        loudness_rms_dbfs=checks["loudness_rms"]["value"],
        noise_floor_dbfs=checks.get("noise_floor_dbfs", {}).get("value"),
        dc=checks["dc_offset"]["value"],
        clicks_in_silence=checks["clicks_in_silence"]["value"],
        boundary_jumps_over_m40=sum(1 for _, v in jumps if qa._db(v) > -40),
        boundary_max_step_dbfs=round(qa._db(max(v for _, v in jumps)), 2) if jumps else None,
        join_gap_median_ms=round(float(np.median(jg)), 1) if jg else None,
        join_gap_min_ms=round(float(np.min(jg)), 1) if jg else None,
        join_gap_sentence_median_ms=(round(float(np.median([g for g, k in zip(jg, kinds) if k == "sentence"])), 1)
                                     if any(k == "sentence" for k in kinds) else None),
        segment_rms_std_db=rep["segment_rms_std_db"],
        signal_statuses={k: v["status"] for k, v in checks.items()},
        repairs=(rep.get("boundary_repair") or {}).get("repairs", 0),
    )
    if asr_model:
        asr = qa.run_asr_check(path, text_expected, asr_model, qa.PROFILES["default"],
                               None, language, spoken=True)
        cer = next((c for c in asr if c["name"] == "asr_cer"), None)
        m["asr_cer"] = cer["value"] if cer else None
        m["asr_status"] = cer["status"] if cer else "missing"
        m["asr_hyp"] = (cer or {}).get("hypothesis_text", "")[:120] if cer else ""
    return m


def cmd_compare(args):
    raw_dir = Path(args.raw)
    meta = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print("compare 输出目录必须为空", file=sys.stderr)
        return 4
    out.mkdir(parents=True)
    variants = args.variants.split(",")
    rows = []
    for item in meta:
        raws = _load_raws(raw_dir / f"{item['name']}.npz", len(item["seg_texts"]))
        for vname in variants:
            cfg = dict(VARIANTS[vname])
            eng = ReplayEngine(item, raws)
            # Replay yields captured arrays only; processing is the sole variable.
            audio, sr, rep = tts.synthesize_text(
                eng, item["text"], language=item["language"], lexicon=None, normalize=True,
                max_chars=90, **cfg)
            wav = out / vname / f"{item['name']}.wav"
            tts.write_audio(wav, audio, sr)
            m = _metrics(wav, audio, sr, rep, item["language"], item["normalized"], args.asr)
            m.update(name=item["name"], engine=item["engine"], case=item["case"], repeat=item["repeat"],
                     variant=vname, wav=str(wav.relative_to(out)), wav_sha256=tts.sha256_file(wav),
                     raw_sha256_match=True)
            rows.append(m)
            print(f"{vname:22s} {item['name']:32s} {m['seconds']:8.2f}s join_med={m['join_gap_median_ms']} "
                  f"bnd>-40={m['boundary_jumps_over_m40']} lufs={m['lufs']}", flush=True)
    (out / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(_summary(rows, variants, args.asr), encoding="utf-8")
    (out / "source_sha256.json").write_text(json.dumps({
        str(p.relative_to(ROOT)): tts.sha256_file(p)
        for p in sorted((ROOT / "scripts").glob("*.py"))} | {"tests/ab_processing.py": tts.sha256_file(Path(__file__))},
        indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def _mean(vals):
    v = [x for x in vals if x is not None]
    return round(float(np.mean(v)), 3) if v else None


def _summary(rows, variants, asr):
    grp = collections.defaultdict(list)
    for r in rows:
        grp[(r["engine"], r["variant"])].append(r)
    cols = ["n", "seconds", "join_gap_median_ms", "join_gap_sentence_median_ms", "boundary_jumps_over_m40",
            "boundary_max_step_dbfs", "true_peak_dbtp", "lufs", "loudness_rms_dbfs", "noise_floor_dbfs",
            "segment_rms_std_db", "clicks_in_silence"] + (["asr_cer"] if asr else [])
    lines = ["# A/B processing summary (objective proxies; no human listening)", "",
             "Each row averages all cases and repeats for that engine/variant.", "",
             "| engine | variant | " + " | ".join(cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for (eng, var) in sorted(grp, key=lambda k: (k[0], variants.index(k[1]) if k[1] in variants else 99)):
        rs = grp[(eng, var)]
        vals = []
        for c in cols:
            if c == "n":
                vals.append(str(len(rs)))
            elif c in ("clicks_in_silence", "boundary_jumps_over_m40"):
                vals.append(str(sum(r[c] or 0 for r in rs)))
            else:
                vals.append(str(_mean([r.get(c) for r in rs])))
        lines.append(f"| {eng} | {var} | " + " | ".join(vals) + " |")
    lines += ["", "Columns: join_gap = inactive (<= -50 dBFS) duration around each segment join (ms); "
              "boundary_jumps_over_m40 = digital-silence boundaries with step > -40 dBFS (summed); "
              "clicks_in_silence summed. LUFS/TP/RMS are signal measurements, not loudness targets."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--engine", choices=["piper", "espeak"], required=True)
    c.add_argument("--model")
    c.add_argument("--language", default="zh-CN", help="Piper locale; eSpeak uses each case's language field")
    c.add_argument("--ids", required=True)
    c.add_argument("--repeat", type=int, default=1)
    c.add_argument("--timeout", type=float, default=60.0)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_capture)
    k = sub.add_parser("compare")
    k.add_argument("--raw", required=True)
    k.add_argument("--variants", default=",".join(VARIANTS))
    k.add_argument("--asr", help="e.g. base (faster-whisper; diagnostic only)")
    k.add_argument("--out", required=True)
    k.set_defaults(func=cmd_compare)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Local real-engine evaluation. No human scores, no automatic quality conclusion."""
import argparse
import gc
import json
import platform
import random
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qa
import tts


def _stats(values):
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"mean": round(statistics.fmean(v), 4), "min": min(v), "max": max(v),
            "n": len(v), "std": statistics.stdev(v) if len(v) > 1 else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=["piper", "espeak", "xtts"], default="piper")
    ap.add_argument("--model")
    ap.add_argument("--config")
    ap.add_argument("--language", default="auto")
    ap.add_argument("--reference-audio")
    ap.add_argument("--authorization")
    ap.add_argument("--purpose")
    ap.add_argument("--out", required=True)
    ap.add_argument("--asr", help="optional ASR model (may download)")
    ap.add_argument("--lexicon")
    ap.add_argument("--ids")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    if not 1 <= args.repeat <= 10:
        ap.error("repeat must be 1..10")
    engine_args = SimpleNamespace(**vars(args), speaker=None, length_scale=1.,
                                  noise_scale=.667, noise_w=.8, cuda=False)
    try:
        engine = tts.build_engine(engine_args)
    except (tts.EngineError, ValueError, OSError) as e:
        ap.exit(2, f"engine unavailable: {e}\n")
    lex = tts.load_lexicon(args.lexicon) if args.lexicon else None
    language = engine.language
    capabilities = engine.capabilities
    config_path = getattr(engine, "config_path", args.config)
    cases = [json.loads(l) for l in (ROOT / "tests/cases.jsonl").read_text().splitlines() if l.strip()]
    if args.ids:
        cases = [c for c in cases if c["id"] in args.ids.split(",")]
    compatible, skipped = [], []
    for c in cases:
        tag = c.get("language", "zh-CN").lower()
        selected = "zh-cn" if language == "cmn" else language
        if c.get("requires_clone") and not capabilities["clone"]:
            skipped.append({"id": c["id"], "reason": "engine has no clone capability"})
        elif tag != selected and tag != selected.split("-")[0]:
            skipped.append({"id": c["id"], "reason": f"locale {tag} != {selected}"})
        else:
            compatible.append(c)
    if not compatible:
        ap.exit(2, "no compatible cases for selected locale/capability\n")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        ap.error("output directory must be empty (protect prior evidence)")
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    rng = random.Random(args.seed)
    for c in compatible:
        for repeat in range(args.repeat):
            arms = [True, False]
            rng.shuffle(arms)
            for norm in arms:
                tag = "norm" if norm else "raw"
                run = {"id": c["id"], "category": c["category"], "arm": tag, "repeat": repeat,
                       "source_text": c["text"], "language": language}
                wav = out / f"{c['id']}.{tag}.r{repeat}.wav"
                try:
                    audio, sr, rep = tts.synthesize_text(engine, c["text"], normalize=norm,
                                                       language=language, lexicon=lex,
                                                       processing="minimal", pauses={})
                    tts.write_audio(wav, audio, sr)
                    x, sr, info = qa.load_mono(wav)
                    run.update(wav=str(wav), sha256=tts.sha256_file(wav), synthesis=rep,
                               signal=qa.run_signal_checks(x, sr, info), rtf=rep["rtf"])
                except (tts.EngineError, ValueError, RuntimeError) as e:
                    run["error"] = str(e)
                runs.append(run)
                # Preserve partial evidence if interrupted later.
                (out / "runs.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2))
    del engine
    gc.collect()  # not a guarantee that native allocators release all memory
    for run in runs:
        if args.asr and "wav" in run:
            # Same intended reading reference and ASR treatment in both arms.
            run["asr"] = qa.run_asr_check(run["wav"], run["source_text"], args.asr,
                                           qa.PROFILES["default"], lex, language)
    rows = []
    for c in compatible:
        row = {"id": c["id"], "category": c["category"]}
        for tag in ("norm", "raw"):
            arm = [r for r in runs if r["id"] == c["id"] and r["arm"] == tag]
            metrics = [{m["name"]: m for m in r.get("asr", [])} for r in arm]
            row[tag] = {"rtf": _stats([r.get("rtf") for r in arm]),
                        "cer": _stats([m.get("asr_cer", {}).get("value") for m in metrics]),
                        "errors": [r["error"] for r in arm if "error" in r],
                        "asr_status": [m.get("asr_cer", {}).get("status", "not_requested") for m in metrics]}
        rows.append(row)
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss / (1024 ** 2 if sys.platform == "darwin" else 1024)
    except ImportError:
        rss_mb = None
    summary = {"engine": args.engine, "capabilities": capabilities,
               "model": tts.model_provenance(args.model), "config": tts.model_provenance(config_path),
               "python": sys.version, "platform": platform.platform(), "asr_model": args.asr,
               "repeat": args.repeat, "seed": args.seed, "skill_version": tts.VERSION,
               "peak_rss_mb": rss_mb, "human_listening": "not_performed", "skipped": skipped,
               "rows": rows, "runs": runs, "evidence_type": "current_objective_run",
               "source_sha256": {str(p.relative_to(ROOT)): tts.sha256_file(p) for p in
                    sorted((ROOT / "scripts").glob("*.py")) + [ROOT / "tests/cases.jsonl", Path(__file__)]}}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (out / "runs.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2))
    lines = ["# Objective evaluation (not human quality)",
             "| case | norm CER | raw CER | norm RTF |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['id']} | {row['norm']['cer']} | {row['raw']['cer']} | {row['norm']['rtf']} |")
    lines.append(f"\nSkipped: {len(skipped)}. Human listening: not performed. No significance claim.")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if any("error" in r or any(m["status"] == "fail" for m in r.get("signal", [])) for r in runs) else 0


if __name__ == "__main__":
    sys.exit(main())

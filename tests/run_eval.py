#!/usr/bin/env python3
"""真实引擎评测：逐条合成 -> 信号验收 -> 回听 CER，并对比“开/关文本规范化”。

用法：python tests/run_eval.py --model MODEL.onnx --out DIR [--asr base]
输出：DIR/summary.md 与 DIR/summary.json（所有数值均来自本次运行，不做人工填写）。
需要 piper-tts 与 faster-whisper；缺失时对应指标标记为 skip。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import qa  # noqa: E402
import tts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--asr", default="small")
    ap.add_argument("--lexicon", help="发音词典 TSV")
    ap.add_argument("--ids", help="逗号分隔的用例 id 子集")
    args = ap.parse_args()
    lex = tts.load_lexicon(args.lexicon) if args.lexicon else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    engine = tts.PiperEngine(model=args.model, config=args.config)
    cases = [json.loads(l) for l in (ROOT / "tests" / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.ids:
        cases = [c for c in cases if c["id"] in args.ids.split(",")]
    rows = []
    for c in cases:
        row = {"id": c["id"], "category": c["category"], "text": c["text"]}
        for norm in (True, False):
            tag = "norm" if norm else "raw"
            audio, sr, rep = tts.synthesize_text(engine, c["text"], normalize=norm, lexicon=lex)
            wav = out / f"{c['id']}.{tag}.wav"
            tts.write_audio(wav, audio, sr)
            x, sr_w, info = qa.load_mono(wav)
            checks = qa.run_signal_checks(x, sr_w, info, c["text"])
            fails = [k["name"] for k in checks if k["status"] == "fail"]
            warns = [k["name"] for k in checks if k["status"] == "warn"]
            row[f"{tag}_seconds"] = rep["duration_seconds"]
            row[f"{tag}_rtf"] = rep["rtf"]
            row[f"{tag}_signal_fail"] = fails
            row[f"{tag}_signal_warn"] = warns
            if args.asr:
                asr = qa.run_asr_check(wav, c["text"], args.asr, qa.PROFILES["default"], lex)[0]
                row[f"{tag}_cer"] = asr["value"]
                row[f"{tag}_asr_text"] = asr["detail"].split("ASR 文本：")[-1]
        rows.append(row)
        print(json.dumps({k: row[k] for k in row if k.endswith("cer") or k.endswith("rtf")}, ensure_ascii=False), c["id"], flush=True)

    (out / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["| 用例 | 类别 | CER（规范化） | CER（原文直送） | RTF | 信号层 fail/warn |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['category']} | {r.get('norm_cer')} | {r.get('raw_cer')} | "
                     f"{r['norm_rtf']} | {r['norm_signal_fail'] or '-'} / {r['norm_signal_warn'] or '-'} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

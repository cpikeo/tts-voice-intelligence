#!/usr/bin/env python3
"""真实引擎评测：逐条合成 -> 信号验收 -> 回听 CER，并对比“开/关文本规范化”。

用法：python tests/run_eval.py --model MODEL.onnx --out DIR [--asr small] [--repeat 3]
输出：DIR/summary.md 与 DIR/summary.json（所有数值均来自本次运行，不做人工填写）。
需要 piper-tts 与 faster-whisper；缺失时对应指标标记为 skip。

设计要点（避免对照实验的混杂因素）：
- 规范化开/关两只手臂的**回听后处理完全相同**（繁转简 + 去标点），参考文本都用
  “正确朗读形式”，因此两只手臂可比；差异只来自“送进引擎的文本”。
- 每条用例重复 --repeat 次，报告均值与极差。单次运行的 CER 波动不可忽略，
  不重复就不该下结论。
- 报告 asr_cer（整体）、asr_cer_text（遮蔽数词）、asr_num_cer（仅数词）三个量，
  不把它们混成一个数。
"""
import argparse
import gc
import json
import resource
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import qa  # noqa: E402
import tts  # noqa: E402


def _stats(values):
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"mean": round(statistics.fmean(v), 4), "min": round(min(v), 4),
            "max": round(max(v), 4), "n": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--asr", default="small")
    ap.add_argument("--lexicon", help="发音词典 TSV")
    ap.add_argument("--ids", help="逗号分隔的用例 id 子集")
    ap.add_argument("--repeat", type=int, default=3, help="每条用例重复次数（用于估计波动）")
    args = ap.parse_args()
    lex = tts.load_lexicon(args.lexicon) if args.lexicon else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    engine = tts.PiperEngine(model=args.model, config=args.config)
    cases = [json.loads(l) for l in (ROOT / "tests" / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.ids:
        cases = [c for c in cases if c["id"] in args.ids.split(",")]
    # 阶段一：合成 + 信号检查（此阶段只驻留引擎）
    rows = []
    for c in cases:
        row = {"id": c["id"], "category": c["category"], "text": c["text"], "arms": {}}
        for norm in (True, False):
            tag = "norm" if norm else "raw"
            arm = {"wavs": [], "rtfs": [], "fails": set(), "warns": set()}
            for r in range(args.repeat):
                audio, sr, rep = tts.synthesize_text(engine, c["text"], normalize=norm, lexicon=lex)
                wav = out / f"{c['id']}.{tag}.r{r}.wav"
                tts.write_audio(wav, audio, sr)
                x, sr_w, info = qa.load_mono(wav)
                checks = qa.run_signal_checks(x, sr_w, info)
                arm["fails"].update(k["name"] for k in checks if k["status"] == "fail")
                arm["warns"].update(k["name"] for k in checks if k["status"] == "warn")
                arm["rtfs"].append(rep["rtf"])
                arm["wavs"].append(str(wav))
                arm.setdefault("dur", rep["duration_seconds"])
            row["arms"][tag] = arm
        rows.append(row)

    # 阶段二：释放引擎后再做回听。引擎（尤其带 g2pW 前端的中文模型）与 whisper
    # 同时在内存时，2 GB 环境会被 OOM 杀掉（实测 xiao_ya + whisper-small → exit 137）。
    del engine
    gc.collect()
    if args.asr:
        for row in rows:
            for tag, arm in row["arms"].items():
                for r, wav in enumerate(arm["wavs"]):
                    res = {k["name"]: k for k in
                           qa.run_asr_check(wav, row["text"], args.asr, qa.PROFILES["default"], lex)}
                    arm.setdefault("cers", []).append(res["asr_cer"]["value"])
                    arm.setdefault("cer_texts", []).append(res["asr_cer_text"]["value"])
                    arm.setdefault("num_cers", []).append(res["asr_num_cer"]["value"])
                    if r == 0:
                        arm["asr_text"] = res["asr_cer"]["detail"].split("ASR 文本：")[-1]
            print(f"{row['id']:12} norm={_stats(row['arms']['norm'].get('cers'))} "
                  f"raw={_stats(row['arms']['raw'].get('cers'))}", flush=True)

    for row in rows:
        for tag, arm in row["arms"].items():
            row[f"{tag}_rtf"] = round(statistics.fmean(arm["rtfs"]), 3)
            row[f"{tag}_signal_fail"] = sorted(arm["fails"])
            row[f"{tag}_signal_warn"] = sorted(arm["warns"])
            row[f"{tag}_cer"] = _stats(arm.get("cers"))
            row[f"{tag}_cer_text"] = _stats(arm.get("cer_texts"))
            row[f"{tag}_num_cer"] = _stats(arm.get("num_cers"))
            row[f"{tag}_asr_text"] = arm.get("asr_text")
        row.pop("arms")

    summary = {"model": tts.model_provenance(args.model), "asr_model": args.asr,
               "repeat": args.repeat, "lexicon": args.lexicon,
               "skill_version": tts.VERSION,
               "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
               "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def _cell(s, key="mean"):
        if not s:
            return "-"
        return f"{s[key]:.3f}" if key in s else "-"

    lines = ["| 用例 | 类别 | CER 均值（规范） | CER 均值（原文） | 遮蔽数词 CER（规范） | "
             "数词 CER（规范） | 极差（规范） | RTF | 信号 fail/warn |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['category']} | {_cell(r['norm_cer'])} | {_cell(r['raw_cer'])} | "
            f"{_cell(r['norm_cer_text'])} | {_cell(r['norm_num_cer'])} | "
            f"{_cell(r['norm_cer'], 'max')}−{_cell(r['norm_cer'], 'min')} | {r['norm_rtf']} | "
            f"{r['norm_signal_fail'] or '-'} / {r['norm_signal_warn'] or '-'} |")
    nc = [r["norm_cer"]["mean"] for r in rows if r["norm_cer"]]
    rc = [r["raw_cer"]["mean"] for r in rows if r["raw_cer"]]
    tc = [r["norm_cer_text"]["mean"] for r in rows if r["norm_cer_text"]]
    if nc:
        lines.append("")
        lines.append(f"- 均值：规范化 {statistics.fmean(nc):.4f}（n={len(nc)}），"
                     f"原文直送 {statistics.fmean(rc):.4f}，遮蔽数词后 {statistics.fmean(tc):.4f}")
        lines.append(f"- 单条最大极差（重复 {args.repeat} 次）："
                     f"{max(r['norm_cer']['max'] - r['norm_cer']['min'] for r in rows if r['norm_cer']):.4f}")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

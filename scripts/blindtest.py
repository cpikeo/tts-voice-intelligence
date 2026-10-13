#!/usr/bin/env python3
"""轻量盲听评测（人类听评）——客观信号检查与 ASR 不能替代人耳。

三步：
  prepare  把音频复制成匿名样本，生成记分表（记分表里不出现原始文件名）
  score    汇总已填写的记分表
  unblind  评完之后再揭示样本与原始文件的对应关系

用法：
  python scripts/blindtest.py prepare --audio a.wav b.wav --out pkg --seed 7
  # 人工试听 pkg/listen/*.wav 并填写 pkg/ratings.csv（1–5 分，5 最好）
  python scripts/blindtest.py score   --pkg pkg
  python scripts/blindtest.py unblind --pkg pkg

设计要点：
- 可重复：同一 --seed 必得同一套匿名编号与试听顺序，结果可跨轮次比较。
- 真盲：记分表与试听文件名只含匿名标签；对应关系只在 blind_map.json 里，
  评完再 unblind，避免“知道是哪个引擎”带来的期望偏差。
- 只做记录与汇总，不做任何主观判断的自动化替代。

评分维度（1–5 分）：
  accuracy 发音准确度 | naturalness 自然度 | prosody 韵律 | expressiveness 表现力
  timbre_consistency 音色一致性 | continuity 段间连续性

依赖：仅标准库。
"""
import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

DIMS = ["accuracy", "naturalness", "prosody", "expressiveness",
        "timbre_consistency", "continuity"]
DIM_ZH = {"accuracy": "发音准确度", "naturalness": "自然度", "prosody": "韵律",
          "expressiveness": "表现力", "timbre_consistency": "音色一致性",
          "continuity": "段间连续性"}
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
AUDIO_EXT = {".wav", ".flac", ".ogg"}


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def collect_audio(inputs):
    """展开为音频文件列表（文件、目录、glob 均可）。"""
    files, seen = [], set()
    for raw in inputs:
        p = Path(raw)
        cands = sorted(p.glob("**/*")) if p.is_dir() else [p]
        for c in cands:
            if c.is_file() and c.suffix.lower() in AUDIO_EXT and c.resolve() not in seen:
                seen.add(c.resolve())
                files.append(c)
    if not files:
        raise SystemExit(f"没有找到音频文件（支持 {'/'.join(sorted(AUDIO_EXT))}）")
    return files


def assign_labels(files, seed):
    """按种子随机化试听顺序并分配匿名标签；同种子必得同结果。"""
    order = list(files)
    random.Random(seed).shuffle(order)
    items = []
    for i, src in enumerate(order):
        label = LABELS[i] if i < len(LABELS) else f"S{i + 1}"
        items.append({"label": label, "order": i + 1, "source": str(src),
                      "sha256": _sha256(src)})
    return items


def cmd_prepare(args):
    files = collect_audio(args.audio)
    if len(files) > 30:
        raise SystemExit(f"样本过多（{len(files)} 个）。盲听听评一轮建议不超过 30 个，"
                         f"否则评分质量会下降；请分批。")
    pkg = Path(args.out)
    listen = pkg / "listen"
    listen.mkdir(parents=True, exist_ok=True)
    items = assign_labels(files, args.seed)
    for it in items:
        dst = listen / f"{it['label']}{Path(it['source']).suffix.lower()}"
        shutil.copy2(it["source"], dst)
        it["listen_file"] = dst.name
    (pkg / "blind_map.json").write_text(
        json.dumps({"seed": args.seed, "items": items}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    sheet = pkg / "ratings.csv"
    with sheet.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample"] + DIMS + ["comment"])
        for it in items:
            w.writerow([it["label"]] + [""] * len(DIMS) + [""])
    (pkg / "README.txt").write_text(
        "盲听听评说明\n"
        "1. 按 blind_map.json 里的 order 顺序试听 listen/ 下的匿名样本。\n"
        "2. 在 ratings.csv 中为每个样本打 1–5 分（5 最好），不要修改 sample 列。\n"
        "   " + " | ".join(f"{d}={DIM_ZH[d]}" for d in DIMS) + "\n"
        "3. 音色一致性与段间连续性请听完整个样本再评；样本应是完整的段落序列。\n"
        "4. 评完前不要打开 blind_map.json；评完后运行 unblind 揭示对应关系。\n",
        encoding="utf-8")
    print(json.dumps({"package": str(pkg), "samples": len(items), "seed": args.seed,
                      "listen_dir": str(listen), "sheet": str(sheet)},
                     ensure_ascii=False, indent=2))
    return 0


def _read_sheet(pkg):
    sheet = Path(pkg) / "ratings.csv"
    if not sheet.is_file():
        raise SystemExit(f"找不到记分表：{sheet}（先运行 prepare）")
    rows = []
    with sheet.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not (r.get("sample") or "").strip():
                continue
            scores, missing = {}, []
            for d in DIMS:
                v = (r.get(d) or "").strip()
                if v == "":
                    missing.append(d)
                    continue
                try:
                    fv = float(v)
                except ValueError:
                    raise SystemExit(f"样本 {r['sample']} 的 {d} 不是数字：{v!r}")
                if not 1 <= fv <= 5:
                    raise SystemExit(f"样本 {r['sample']} 的 {d}={fv} 超出 1–5 范围")
                scores[d] = fv
            rows.append({"sample": r["sample"].strip(), "scores": scores,
                         "missing": missing, "comment": (r.get("comment") or "").strip()})
    return rows


def cmd_score(args):
    rows = _read_sheet(args.pkg)
    scored = [r for r in rows if r["scores"]]
    if not scored:
        raise SystemExit("记分表还没有任何评分")
    per_sample = []
    for r in scored:
        vals = list(r["scores"].values())
        per_sample.append({"sample": r["sample"], **r["scores"],
                           "mean": round(sum(vals) / len(vals), 3),
                           "missing": r["missing"], "comment": r["comment"]})
    dim_mean = {d: round(sum(r["scores"][d] for r in scored if d in r["scores"])
                         / max(1, sum(1 for r in scored if d in r["scores"])), 3) for d in DIMS}
    overall = round(sum(r["mean"] for r in per_sample) / len(per_sample), 3)
    out = {"package": str(args.pkg), "n_samples": len(rows), "n_scored": len(scored),
           "per_sample": sorted(per_sample, key=lambda r: r["sample"]),
           "dimension_mean": dim_mean, "overall_mean": overall}
    (Path(args.pkg) / "scores.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    lines = ["| 样本 | " + " | ".join(DIM_ZH[d] for d in DIMS) + " | 均值 | 备注 |",
             "|---|" + "---|" * (len(DIMS) + 2)]
    for r in out["per_sample"]:
        cells = [f"{r[d]:.1f}" if d in r else "-" for d in DIMS]
        lines.append(f"| {r['sample']} | " + " | ".join(cells) +
                     f" | {r['mean']:.2f} | {r['comment']} |")
    lines.append("")
    lines.append("维度均值：" + "，".join(f"{DIM_ZH[d]} {dim_mean[d]:.2f}" for d in DIMS))
    lines.append(f"总均值 {overall:.2f}（已评分 {len(scored)}/{len(rows)} 个样本）")
    (Path(args.pkg) / "scores.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def cmd_unblind(args):
    mp = Path(args.pkg) / "blind_map.json"
    if not mp.is_file():
        raise SystemExit(f"找不到映射文件：{mp}（先运行 prepare）")
    mapping = json.loads(mp.read_text(encoding="utf-8"))
    scores = {}
    sp = Path(args.pkg) / "scores.json"
    if sp.is_file():
        scores = {r["sample"]: r for r in json.loads(sp.read_text(encoding="utf-8"))["per_sample"]}
    lines = ["| 试听序 | 匿名 | 原始文件 | sha256 前 8 位 | 均值 |", "|---|---|---|---|---|"]
    for it in sorted(mapping["items"], key=lambda x: x["order"]):
        s = scores.get(it["label"], {})
        lines.append(f"| {it['order']} | {it['label']} | {it['source']} | "
                     f"{it['sha256'][:8]} | {s.get('mean', '-')} |")
    print("\n".join(lines))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="盲听听评（人类主观评价）")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare", help="生成匿名试听包与记分表")
    a.add_argument("--audio", nargs="+", required=True, help="音频文件或目录")
    a.add_argument("--out", required=True, help="输出目录")
    a.add_argument("--seed", type=int, default=7, help="随机种子（同种子可复现）")
    a.set_defaults(func=cmd_prepare)
    b = sub.add_parser("score", help="汇总记分表")
    b.add_argument("--pkg", required=True)
    b.set_defaults(func=cmd_score)
    c = sub.add_parser("unblind", help="揭示匿名与原始文件的对应")
    c.add_argument("--pkg", required=True)
    c.set_defaults(func=cmd_unblind)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

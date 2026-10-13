#!/usr/bin/env python3
"""Anonymous human-rating packages; never generates subjective scores."""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

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


def _atomic_write_text(path, content, mode=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        Path(tmp).write_text(content, encoding="utf-8")
        if mode is not None:
            Path(tmp).chmod(mode)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


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


def _resolve_ratings_dir(pkg, ratings_dir=None):
    pkg = Path(pkg)
    if ratings_dir:
        return Path(ratings_dir)
    if (pkg / "ratings.csv").is_file():
        return pkg
    if (pkg / "ratings" / "ratings.csv").is_file():
        return pkg / "ratings"
    return pkg


def cmd_prepare(args):
    import soundfile as sf
    files = collect_audio(args.audio)
    if len(files) > 30:
        raise SystemExit(f"样本过多（{len(files)} 个）。盲听听评一轮建议不超过 30 个，"
                         f"否则评分质量会下降；请分批。")
    pkg = Path(args.out)
    if pkg.exists() and any(pkg.iterdir()):
        raise SystemExit("包目录非空；拒绝覆盖评分与旧试听样本")
    listen = pkg / "listen"
    ratings_dir = Path(args.ratings_dir) if getattr(args, "ratings_dir", None) else (pkg / "ratings" if getattr(args, "isolate", False) else pkg)
    map_path = Path(args.map_file) if getattr(args, "map_file", None) else (pkg / "blind_map.json")

    if getattr(args, "map_file", None) or getattr(args, "isolate", False):
        if not getattr(args, "map_file", None):
            raise SystemExit("启用 --isolate 时必须通过 --map-file 指定包外私有映射路径")
        try:
            map_res = map_path.resolve()
            if map_res.is_relative_to(pkg.resolve()) or map_res.is_relative_to(ratings_dir.resolve()):
                raise SystemExit("隔离模式下 --map-file 不能位于试听包或评分目录内部")
        except AttributeError:
            if str(map_path.resolve()).startswith(str(pkg.resolve())):
                raise SystemExit("隔离模式下 --map-file 不能位于试听包内部")

    listen.mkdir(parents=True, exist_ok=True)
    ratings_dir.mkdir(parents=True, exist_ok=True)
    items = assign_labels(files, args.seed)
    public_manifest = []
    for it in items:
        dst = listen / f"{it['label']}{Path(it['source']).suffix.lower()}"
        # Re-encode to strip identifying container metadata.
        audio, sr = sf.read(it["source"], dtype="float32", always_2d=True)
        sf.write(dst, audio, sr)
        # Immediate playability & anonymity verification on the generated listening file
        verify_audio, verify_sr = sf.read(dst, dtype="float32", always_2d=True)
        if verify_sr != sr or verify_audio.size == 0 or not np.isfinite(verify_audio).all():
            raise SystemExit(f"试听样本校验失败（不可播放或非有限值）：{dst}")
        raw_bytes = dst.read_bytes()
        src_stem = Path(it["source"]).stem.encode("utf-8")
        if len(src_stem) >= 4 and src_stem in raw_bytes:
            raise SystemExit(f"试听样本容器仍残留原始文件名元数据：{dst}")
        listen_sha = _sha256(dst)
        dur = round(float(verify_audio.shape[0]) / float(verify_sr), 3)
        it["listen_file"] = dst.name
        it["listen_sha256"] = listen_sha
        it["duration_seconds"] = dur
        it["sample_rate"] = int(verify_sr)
        public_manifest.append({
            "label": it["label"],
            "order": it["order"],
            "listen_file": dst.name,
            "listen_sha256": listen_sha,
            "duration_seconds": dur,
            "sample_rate": int(verify_sr),
            "playable": True,
        })

    _atomic_write_text(
        map_path,
        json.dumps({"seed": args.seed, "package": str(pkg), "items": items}, ensure_ascii=False, indent=2),
        mode=0o600,
    )
    _atomic_write_text(listen / "order.txt", "\n".join(it["listen_file"] for it in items) + "\n")
    _atomic_write_text(
        listen / "manifest.json",
        json.dumps({"seed": args.seed, "samples": len(public_manifest), "items": public_manifest},
                   ensure_ascii=False, indent=2),
    )
    sheet = ratings_dir / "ratings.csv"
    with sheet.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample"] + DIMS + ["comment"])
        for it in items:
            w.writerow([it["label"]] + [""] * len(DIMS) + [""])
    _atomic_write_text(
        pkg / "README.txt",
        "盲听听评说明\n"
        "1. 按 listen/order.txt 试听；只向评员发 listen/ 与 ratings.csv，不发映射。\n"
        "2. 在 ratings.csv 中为每个样本打 1–5 分（5 最好），不要修改 sample 列。\n"
        "   " + " | ".join(f"{d}={DIM_ZH[d]}" for d in DIMS) + "\n"
        "3. 音色一致性与段间连续性请听完整个样本再评；样本应是完整的段落序列。\n"
        "4. 评完前不要打开 blind_map.json；评完后运行 unblind 揭示对应关系。\n",
    )
    print(json.dumps({"package": str(pkg), "samples": len(items), "seed": args.seed,
                      "listen_dir": str(listen), "sheet": str(sheet),
                      "isolated_map": bool(getattr(args, "map_file", None))},
                     ensure_ascii=False, indent=2))
    return 0


def _expected_labels(pkg, map_file=None):
    pkg = Path(pkg)
    manifest_p = pkg / "listen" / "manifest.json"
    if manifest_p.is_file():
        manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
        for it in manifest.get("items", []):
            lf = pkg / "listen" / it["listen_file"]
            if not lf.is_file() or _sha256(lf) != it.get("listen_sha256"):
                raise SystemExit(f"试听音频缺失或哈希与 prepare 清单不符：{lf}")
        return {it["label"] for it in manifest["items"]}
    mp = Path(map_file) if map_file else (pkg / "blind_map.json")
    if mp.is_file():
        mapping = json.loads(mp.read_text(encoding="utf-8"))
        return {it["label"] for it in mapping["items"]}
    order_p = pkg / "listen" / "order.txt"
    if order_p.is_file():
        return {Path(line.strip()).stem for line in order_p.read_text(encoding="utf-8").splitlines() if line.strip()}
    raise SystemExit(f"找不到试听清单或映射文件：{pkg}")


def _read_sheet(pkg, ratings_dir=None, map_file=None):
    rdir = _resolve_ratings_dir(pkg, ratings_dir)
    sheet = rdir / "ratings.csv"
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
    labels = [r["sample"] for r in rows]
    expected = _expected_labels(pkg, map_file=map_file)
    if len(labels) != len(set(labels)) or set(labels) != expected:
        raise SystemExit("评分标签重复、遗漏或未知")
    return rows, sheet, rdir


def cmd_score(args):
    rows, sheet, rdir = _read_sheet(
        args.pkg,
        ratings_dir=getattr(args, "ratings_dir", None),
        map_file=getattr(args, "map_file", None),
    )
    scored = [r for r in rows if r["scores"]]
    if not scored:
        raise SystemExit("记分表还没有任何评分")
    per_sample = []
    for r in scored:
        vals = list(r["scores"].values())
        per_sample.append({"sample": r["sample"], **r["scores"],
                           "mean": round(sum(vals) / len(vals), 3),
                           "missing": r["missing"], "comment": r["comment"]})
    counts = {d: sum(d in r["scores"] for r in scored) for d in DIMS}
    dim_mean = {d: round(sum(r["scores"][d] for r in scored if d in r["scores"]) / counts[d], 3)
                if counts[d] else None for d in DIMS}
    overall = round(sum(r["mean"] for r in per_sample) / len(per_sample), 3)
    out = {"package": str(args.pkg), "ratings_sheet": str(sheet),
           "ratings_sha256": _sha256(sheet), "n_samples": len(rows), "n_scored": len(scored),
           "per_sample": sorted(per_sample, key=lambda r: r["sample"]),
           "dimension_mean": dim_mean, "dimension_n": counts, "overall_mean": overall,
           "complete": all(not r["missing"] for r in rows),
           "evidence_type": "human_entered_not_automatically_verified"}
    scores_json = json.dumps(out, ensure_ascii=False, indent=2)
    _atomic_write_text(rdir / "scores.json", scores_json)
    if rdir.resolve() != Path(args.pkg).resolve():
        _atomic_write_text(Path(args.pkg) / "scores.json", scores_json)
    lines = ["| 样本 | " + " | ".join(DIM_ZH[d] for d in DIMS) + " | 均值 | 备注 |",
             "|---|" + "---|" * (len(DIMS) + 2)]
    for r in out["per_sample"]:
        cells = [f"{r[d]:.1f}" if d in r else "-" for d in DIMS]
        lines.append(f"| {r['sample']} | " + " | ".join(cells) +
                     f" | {r['mean']:.2f} | {r['comment']} |")
    lines.append("")
    lines.append("维度均值：" + "，".join(f"{DIM_ZH[d]} {dim_mean[d] if dim_mean[d] is not None else '未评分'}" for d in DIMS))
    lines.append(f"总均值 {overall:.2f}（已评分 {len(scored)}/{len(rows)} 个样本）")
    scores_md = "\n".join(lines) + "\n"
    _atomic_write_text(rdir / "scores.md", scores_md)
    if rdir.resolve() != Path(args.pkg).resolve():
        _atomic_write_text(Path(args.pkg) / "scores.md", scores_md)
    print("\n".join(lines))
    return 0


def cmd_unblind(args):
    pkg = Path(args.pkg)
    mp = Path(args.map_file) if getattr(args, "map_file", None) else (pkg / "blind_map.json")
    if not mp.is_file():
        raise SystemExit(f"找不到映射文件：{mp}（先运行 prepare 或通过 --map-file 指定私有映射）")
    rows, sheet, rdir = _read_sheet(
        args.pkg,
        ratings_dir=getattr(args, "ratings_dir", None),
        map_file=str(mp),
    )
    if not rows or any(r["missing"] for r in rows):
        raise SystemExit("评分未完整，拒绝揭盲")
    mapping = json.loads(mp.read_text(encoding="utf-8"))
    for it in mapping.get("items", []):
        if "listen_file" in it and "listen_sha256" in it:
            lf = pkg / "listen" / it["listen_file"]
            if not lf.is_file() or _sha256(lf) != it["listen_sha256"]:
                raise SystemExit(f"拒绝揭盲：试听文件缺失或已被篡改：{lf}")
    scores = {}
    sp = rdir / "scores.json"
    if not sp.is_file():
        sp = pkg / "scores.json"
    if sp.is_file():
        scores = {r["sample"]: r for r in json.loads(sp.read_text(encoding="utf-8"))["per_sample"]}
    else:
        for r in rows:
            vals = list(r["scores"].values())
            scores[r["sample"]] = {"sample": r["sample"], **r["scores"],
                                   "mean": round(sum(vals) / len(vals), 3), "comment": r["comment"]}
    lines = ["| 试听序 | 匿名 | 原始文件 | sha256 前 8 位 | 均值 |", "|---|---|---|---|---|"]
    unblinded_items = []
    for it in sorted(mapping["items"], key=lambda x: x["order"]):
        s = scores.get(it["label"], {})
        lines.append(f"| {it['order']} | {it['label']} | {it['source']} | "
                     f"{it['sha256'][:8]} | {s.get('mean', '-')} |")
        unblinded_items.append({
            "order": it["order"],
            "label": it["label"],
            "source": it["source"],
            "source_sha256": it["sha256"],
            "listen_file": it.get("listen_file"),
            "listen_sha256": it.get("listen_sha256"),
            "scores": {d: s.get(d) for d in DIMS if d in s},
            "mean": s.get("mean"),
            "comment": s.get("comment", ""),
        })
    unblinded_report = {
        "package": str(pkg),
        "seed": mapping.get("seed"),
        "ratings_sheet": str(sheet),
        "ratings_sha256": _sha256(sheet),
        "map_sha256": _sha256(mp),
        "items": unblinded_items,
        "evidence_type": "unblinded_after_complete_ratings",
    }
    out_json = Path(args.out) if getattr(args, "out", None) else (rdir / "unblinded.json")
    _atomic_write_text(out_json, json.dumps(unblinded_report, ensure_ascii=False, indent=2))
    _atomic_write_text(out_json.with_suffix(".md"), "\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="盲听听评（人类主观评价）")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare", help="生成匿名试听包与记分表")
    a.add_argument("--audio", nargs="+", required=True, help="音频文件或目录")
    a.add_argument("--out", required=True, help="输出目录")
    a.add_argument("--seed", type=int, default=7, help="随机种子（同种子可复现）")
    a.add_argument("--map-file", help="将 blind_map.json 隔离保存到包外私有路径")
    a.add_argument("--ratings-dir", help="将 ratings.csv 隔离保存到独立评分目录")
    a.add_argument("--isolate", action="store_true", help="强制隔离音频(listen/)、评分(ratings/)与私有映射(--map-file)")
    a.set_defaults(func=cmd_prepare)
    b = sub.add_parser("score", help="汇总记分表")
    b.add_argument("--pkg", required=True)
    b.add_argument("--ratings-dir", help="独立评分目录（若与 pkg 分离）")
    b.add_argument("--map-file", help="可选私有映射路径")
    b.set_defaults(func=cmd_score)
    c = sub.add_parser("unblind", help="揭示匿名与原始文件的对应")
    c.add_argument("--pkg", required=True)
    c.add_argument("--map-file", help="包外私有 blind_map.json 路径")
    c.add_argument("--ratings-dir", help="独立评分目录（若与 pkg 分离）")
    c.add_argument("--out", help="揭盲 JSON 报告输出路径")
    c.set_defaults(func=cmd_unblind)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

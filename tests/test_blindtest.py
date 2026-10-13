"""盲听工具的自测：只验证机制（可重复、真匿名、汇总正确），不替代人工评分。"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import blindtest  # noqa: E402
import tts  # noqa: E402

SR = 8000


def _make(pkg, n=3, seed=7):
    """造 n 个可解码的音频并生成盲听包。"""
    src = pkg / "src"
    src.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(n):
        p = src / f"system_{i}.wav"
        tts.write_audio(p, (0.1 * np.sin(np.arange(SR) / (i + 3))).astype(np.float32), SR)
        files.append(str(p))
    blindtest.main(["prepare", "--audio", *files, "--out", str(pkg / "pkg"), "--seed", str(seed)])
    return pkg / "pkg"


def test_prepare_is_reproducible_with_same_seed(tmp_path):
    a = _make(tmp_path / "a", seed=7)
    b = _make(tmp_path / "b", seed=7)
    ma = json.loads((a / "blind_map.json").read_text(encoding="utf-8"))
    mb = json.loads((b / "blind_map.json").read_text(encoding="utf-8"))
    # 同一批输入 + 同一种子 -> 同一套匿名编号与试听顺序（内容顺序可比，绝对路径不可比）
    assert [i["label"] for i in ma["items"]] == [i["label"] for i in mb["items"]]
    assert [Path(i["source"]).name for i in ma["items"]] == [Path(i["source"]).name for i in mb["items"]]


def test_prepare_is_actually_blind(tmp_path):
    """记分表与试听文件名都不得泄露原始文件名（否则不是盲测）。"""
    pkg = _make(tmp_path / "p", seed=3)
    sheet = (pkg / "ratings.csv").read_text(encoding="utf-8")
    assert "system_" not in sheet
    assert "system_" not in "".join(p.name for p in (pkg / "listen").iterdir())
    mapping = json.loads((pkg / "blind_map.json").read_text(encoding="utf-8"))
    assert {i["label"] for i in mapping["items"]} == {"A", "B", "C"}
    assert all(i["sha256"] for i in mapping["items"])


def test_prepare_copy_is_playable(tmp_path):
    pkg = _make(tmp_path / "p")
    for f in (pkg / "listen").iterdir():
        assert f.stat().st_size > 0


def _fill(pkg, scores):
    sheet = pkg / "ratings.csv"
    rows = list(csv.reader(sheet.read_text(encoding="utf-8").splitlines()))
    header, body = rows[0], rows[1:]
    for row, vals in zip(body, scores):
        row[1:1 + len(vals)] = [str(v) for v in vals]
    sheet.write_text("\n".join(",".join(r) for r in [header] + body) + "\n", encoding="utf-8")


def test_score_aggregates(tmp_path):
    pkg = _make(tmp_path / "p")
    _fill(pkg, [[5, 4, 3, 2, 1, 5], [1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3]])
    blindtest.main(["score", "--pkg", str(pkg)])
    data = json.loads((pkg / "scores.json").read_text(encoding="utf-8"))
    assert data["n_scored"] == 3
    assert data["overall_mean"] == pytest.approx((20 / 6 + 1 + 3) / 3, abs=1e-3)
    assert data["dimension_mean"]["accuracy"] == pytest.approx(3.0)
    assert (pkg / "scores.md").is_file()


def test_score_rejects_out_of_range(tmp_path):
    pkg = _make(tmp_path / "p")
    _fill(pkg, [[9, 4, 3, 2, 1, 5], [1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3]])
    with pytest.raises(SystemExit, match="超出 1–5 范围"):
        blindtest.main(["score", "--pkg", str(pkg)])


def test_unblind_reveals_mapping(tmp_path, capsys):
    pkg = _make(tmp_path / "p", seed=7)
    _fill(pkg, [[5, 4, 3, 2, 1, 5], [1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3]])
    blindtest.main(["score", "--pkg", str(pkg)])
    capsys.readouterr()
    blindtest.main(["unblind", "--pkg", str(pkg)])
    table = capsys.readouterr().out
    assert "system_" in table  # 评完才揭示
    for i in range(3):
        assert f"system_{i}.wav" in table

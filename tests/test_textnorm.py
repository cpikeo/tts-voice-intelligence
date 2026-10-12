import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest  # noqa: E402
from textnorm import int_zh, normalize_zh  # noqa: E402


@pytest.mark.parametrize("n,expected", [
    (0, "零"), (10, "十"), (12, "十二"), (20, "二十"), (105, "一百零五"),
    (1005, "一千零五"), (1050, "一千零五十"), (10000, "一万"), (10086, "一万零八十六"),
    (120000, "十二万"), (100000, "十万"),
])
def test_int_zh(n, expected):
    assert int_zh(n) == expected


@pytest.mark.parametrize("src,expected", [
    ("2026年", "二零二六年"),
    ("今天是2026年10月12日。", "今天是二零二六年十月十二日。"),
    ("会议10:30开始", "会议十点三十分开始"),
    ("10:00 开始，10:05 也行", "十点整 开始，十点零五分 也行"),
    ("增长3.5%", "增长百分之三点五"),
    ("Python 3.13 很快", "Python 三点一三 很快"),
    ("ＧＰＵ１２３", "GPU一百二十三"),
    ("GPU & CPU", "GPU 和 CPU"),
    ("等待……", "等待……"),
    ("**重点**", "重点"),
])
def test_normalize(src, expected):
    assert normalize_zh(src) == expected


def test_normalize_keeps_plain_chinese():
    s = "这是一段没有数字的普通中文。"
    assert normalize_zh(s) == s

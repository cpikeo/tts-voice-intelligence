"""中文为主的朗读文本规范化（引擎无关）。

只处理引擎前端容易读错的形态：年份、时刻、百分比、小数、整数、部分符号。
不做语义改写；调用方应同时保留原文与规范化文本（见 tts.py 的 report）。
"""
import re
import unicodedata

DIGITS = "零一二三四五六七八九"


def _lt10000(n: int) -> str:
    """1..9999 -> 中文数词（如 1005 -> 一千零五，10 -> 十）。"""
    out, pending_zero = "", False
    for val, unit in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        d, n = divmod(n, val)
        if d == 0:
            pending_zero = bool(out)
            continue
        if pending_zero:
            out += "零"
            pending_zero = False
        out += ("" if (val == 10 and d == 1 and not out) else DIGITS[d]) + unit
    return out


def int_zh(n: int) -> str:
    """非负整数 -> 中文读法。超过亿级时逐位读。"""
    if n == 0:
        return "零"
    if n >= 10**8:
        return digits_zh(str(n))
    wan, rest = divmod(n, 10000)
    out = _lt10000(wan) + "万" if wan else ""
    if rest:
        if wan and rest < 1000:
            out += "零"
        out += _lt10000(rest)
    return out


def digits_zh(s: str) -> str:
    """逐位读（年份、编号）：2026 -> 二零二六。"""
    return "".join(DIGITS[int(c)] for c in s)


def _num(s: str) -> str:
    """'3.5' -> 三点五；'1005' -> 一千零五。"""
    if "." in s:
        a, b = s.split(".", 1)
        return int_zh(int(a)) + "点" + digits_zh(b)
    return int_zh(int(s))


def load_lexicon(path):
    """发音词典：TSV，每行 `原词<TAB>朗读形式`，# 开头为注释。长词优先匹配。"""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            src, _, dst = line.partition("\t")
            if src and dst:
                pairs.append((src.strip(), dst.strip()))
    return sorted(pairs, key=lambda p: -len(p[0]))


def apply_lexicon(text: str, pairs) -> str:
    for src, dst in pairs:
        if src.isascii():  # 英文术语：忽略大小写，且不匹配单词内部
            pat = r"(?<![A-Za-z0-9])" + re.escape(src) + r"(?![A-Za-z0-9])"
            text = re.sub(pat, dst, text, flags=re.IGNORECASE)
        else:
            text = text.replace(src, dst)
    return text


def normalize_zh(text: str, lexicon=None) -> str:
    if lexicon:
        text = apply_lexicon(text, lexicon)
    # 仅把全角数字/拉丁字母/％ 转半角；全角标点（，。！？）必须保留，NFKC 会破坏其韵律语义
    t = "".join(
        unicodedata.normalize("NFKC", ch)
        if ("\uff10" <= ch <= "\uff19" or "\uff21" <= ch <= "\uff3a"
            or "\uff41" <= ch <= "\uff5a" or ch == "\uff05") else ch
        for ch in text
    )
    t = re.sub(r"[*#`]+", "", t)              # 去除常见 Markdown 残留
    t = t.replace("&", "和").replace("+", "加")
    t = re.sub(r"[ \t]+", " ", t)

    # 1. 年份（1900–2099，4 位）逐位读
    t = re.sub(r"(?<![\d.])((?:19|20)\d{2})(?![\d.])", lambda m: digits_zh(m.group(1)), t)
    # 2. 时刻 HH:MM
    def _time(m):
        h, mi = int(m.group(1)), int(m.group(2))
        if mi == 0:
            return f"{int_zh(h)}点整"
        prefix = "零" if mi < 10 else ""
        return f"{int_zh(h)}点{prefix}{int_zh(mi)}分"
    t = re.sub(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)", _time, t)
    # 3. 百分比
    t = re.sub(r"(\d+(?:\.\d+)?)\s*%", lambda m: "百分之" + _num(m.group(1)), t)
    # 4. 其余数字（小数 / 整数）
    t = re.sub(r"\d+(?:\.\d+)?", lambda m: _num(m.group(0)), t)
    # 5. 省略号与空白规整
    t = re.sub(r"\.{3,}|…+", "……", t)
    t = re.sub(r" *\n *", "\n", t).strip()
    return t

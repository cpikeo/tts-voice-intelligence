"""中文为主的朗读文本规范化（引擎无关）。

只处理引擎前端容易读错的形态：日期、年份、时刻、百分比、小数、整数、编号与部分符号。
不做语义改写；调用方应同时保留原文与规范化文本（见 tts.py 的 report）。

设计约定：
- 词典条目先于一切规则替换，并用哨兵占位保护，因此**词典给出的朗读形式是最终形式**，
  不会被后续的数字规则二次改写。
- 规则顺序不可随意调换（日期 -> 年份 -> 时刻 -> 编号 -> 亿/万 -> 百分比 -> 负号 -> 其余数字）。
"""
import re
import unicodedata

DIGITS = "零一二三四五六七八九"

# 词典替换值的占位符：用私用区字符，不含数字、不受 NFKC 影响，因此不会被任何数字规则误伤
def _guard(i: int) -> str:
    return "\x01" + chr(0xE000 + i) + "\x01"

# 数词遮蔽（用于 qa 的 cer_text：两侧用同一规则去掉数词，只比较其余发音）。
# 必须避免把普通词里的数字字符也去掉——例如“一起”的“一”不是数词，
# 若一并遮蔽，cer_text 就不再只衡量“数词以外”的发音。
_MASK_NUM = re.compile(
    r"\d+(?:\.\d+)?%?"                                  # 阿拉伯数字（含百分号）
    r"|百分之[零一二三四五六七八九十百千万亿两]*(?:点[零一二三四五六七八九]+)?"  # 百分之X（对应 X%）
    r"|[零一二三四五六七八九十百千万亿两]{2,}"              # 两个及以上中文数字连用
    r"|[零一二三四五六七八九]*[十百千万亿][零一二三四五六七八九]*"   # 含量级单位
    r"|[零一二三四五六七八九]点[零一二三四五六七八九]"          # 小数点两侧都是单个数字
    r"|(?<=[点])[零一二三四五六七八九]"                      # 小数点后的单个数字（不吞掉“点”本身，
    r"|[零一二三四五六七八九](?=[点])",                      # 两侧才能完全对称地遮蔽）
)


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


def _lt1e8(n: int) -> str:
    """1..99999999 -> 中文数词，按“万”分节。"""
    wan, rest = divmod(n, 10000)
    out = _lt10000(wan) + "万" if wan else ""
    if rest:
        if wan and rest < 1000:
            out += "零"
        out += _lt10000(rest)
    return out


def int_zh(n: int) -> str:
    """非负整数 -> 中文读法。超过万亿（10**12）时逐位读。"""
    if n == 0:
        return "零"
    if n >= 10**12:
        return digits_zh(str(n))
    yi, rest = divmod(n, 10**8)
    out = _lt1e8(yi) + "亿" if yi else ""
    if rest:
        # 亿级与后续部分之间：若后续不足千万（如 1_0500_0000）需补“零”
        if yi and rest < 10**7:
            out += "零"
        out += _lt1e8(rest)
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


def mask_numbers(text: str) -> str:
    """去掉中英文数词（两侧用同一规则，供 qa 计算不含数字的 CER）。"""
    return _MASK_NUM.sub("", text)


def normalize_zh(text: str, lexicon=None) -> str:
    t = text
    # 0. 发音词典：用哨兵占位，保证朗读形式不被后续规则改写
    guards = []
    if lexicon:
        for i, (src, dst) in enumerate(lexicon):
            tok = _guard(i)
            if src.isascii():  # 英文术语：忽略大小写，且不匹配单词内部
                t, n = re.subn(r"(?<![A-Za-z0-9])" + re.escape(src) + r"(?![A-Za-z0-9])",
                               tok, t, flags=re.IGNORECASE)
            else:
                n = t.count(src)
                t = t.replace(src, tok)
            if n:
                guards.append((tok, dst))
    # 仅把全角数字/拉丁字母/％ 转半角；全角标点（，。！？）必须保留，NFKC 会破坏其韵律语义
    t = "".join(
        unicodedata.normalize("NFKC", ch)
        if ("\uff10" <= ch <= "\uff19" or "\uff21" <= ch <= "\uff3a"
            or "\uff41" <= ch <= "\uff5a" or ch == "\uff05") else ch
        for ch in t
    )
    t = re.sub(r"[*#`]+", "", t)              # 去除常见 Markdown 残留
    t = t.replace("&", "和").replace("+", "加")
    t = re.sub(r"[ \t]+", " ", t)

    # 1. 日期 2026/10/12、2026-10-12（必须早于年份与负号规则，否则只剩分隔符）
    def _date(m):
        y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
        return f"{digits_zh(y)}年{int_zh(mo)}月{int_zh(d)}日"
    t = re.sub(r"(?<!\d)(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})(?!\d)", _date, t)
    # 2. 带“亿/万”单位的数量（1亿2000万 -> 一亿二千万）
    #    必须早于年份规则，否则“2000万”会被当成年份 2000 逐位读成“二零零零万”
    t = re.sub(r"(\d+(?:\.\d+)?)\s*([亿万])",
               lambda m: _num(m.group(1)) + m.group(2), t)
    # 3. 年份（1900–2099，4 位）逐位读
    t = re.sub(r"(?<![\d.])((?:19|20)\d{2})(?![\d.万亿])", lambda m: digits_zh(m.group(1)), t)
    # 4. 时刻 HH:MM
    def _time(m):
        h, mi = int(m.group(1)), int(m.group(2))
        if mi == 0:
            return f"{int_zh(h)}点整"
        prefix = "零" if mi < 10 else ""
        return f"{int_zh(h)}点{prefix}{int_zh(mi)}分"
    t = re.sub(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)", _time, t)
    # 4. 编号/型号（A-1024、GPT-4）：数字逐位读，不做数值读法
    t = re.sub(r"(?<![A-Za-z0-9])([A-Za-z]{1,10})-(\d+)(?![A-Za-z0-9])",
               lambda m: f"{m.group(1)}-{digits_zh(m.group(2))}", t)
    # 5. 带“亿/万”单位的数量（1亿2000万 -> 一亿二千万）
    t = re.sub(r"(\d+(?:\.\d+)?)\s*([亿万])",
               lambda m: _num(m.group(1)) + m.group(2), t)
    # 6. 百分比
    t = re.sub(r"(\d+(?:\.\d+)?)\s*%", lambda m: "百分之" + _num(m.group(1)), t)
    # 7. 负号（不触碰范围 10-20 与已处理完的日期）
    t = re.sub(r"(?<![\d\-])-(?=\d)", "负", t)
    # 8. 其余数字（小数 / 整数）
    t = re.sub(r"\d+(?:\.\d+)?", lambda m: _num(m.group(0)), t)
    # 9. 省略号与空白规整
    t = re.sub(r"\.{3,}|…+", "……", t)
    t = re.sub(r" *\n *", "\n", t)
    for tok, dst in guards:  # 最后还原词典朗读形式
        t = t.replace(tok, dst)
    return t.strip()

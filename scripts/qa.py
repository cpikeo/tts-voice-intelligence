#!/usr/bin/env python3
"""音频交付验收（独立于引擎，可对任意 WAV/FLAC/OGG 运行）。

客观检查：文件可解码、时长合理、无削波/NaN/直流、响度在范围内、无异常长静音、
无静音区的爆点。可选 --asr：用 faster-whisper 回听识别，与规范化原文比较字符错误率（CER）。

退出码：0 = 无 fail；1 = 存在 fail；2 = 文件无法读取。
阈值为经验启发值，不是行业标准；阈值依据、重复性与适用范围见 references/eval-results.md，
可用 --profile 调整。

回听指标分为三个互不混淆的量（详见 references/eval-results.md 第 2 节）：
  asr_cer        整体 CER（回听文本的阿拉伯数字串会先读成中文再比对）
  asr_cer_text   遮蔽数词后的 CER（只衡量非数字部分的发音，不受规范化链路影响）
  asr_num_cer    仅数词部分的 CER（只能验证“听到的数字串”，不能验证朗读风格）
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from textnorm import _MASK_NUM, load_lexicon, mask_numbers, normalize_zh  # noqa: E402

FRAME_MS = 10


def _db(x):
    return 20 * np.log10(max(float(x), 1e-9))


def _check(name, status, value=None, detail=""):
    return {"name": name, "status": status, "value": value, "detail": detail}


def load_mono(path):
    import soundfile as sf
    info = sf.info(str(path))
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x.mean(axis=1), sr, info


def run_signal_checks(x, sr, info):
    checks = []
    dur = len(x) / sr
    checks.append(_check("format", "pass" if sr >= 16000 else "warn",
                         {"sr": sr, "channels": info.channels, "subtype": info.subtype},
                         "" if sr >= 16000 else "采样率低于 16 kHz"))
    if not np.all(np.isfinite(x)):
        checks.append(_check("finite", "fail", None, "存在 NaN/Inf 采样"))
        return checks
    checks.append(_check("duration", "pass" if dur >= 0.3 else "fail", round(dur, 3)))

    clip_ratio = float(np.mean(np.abs(x) >= 0.999))
    checks.append(_check("clipping", "fail" if clip_ratio > 5e-4 else ("warn" if clip_ratio > 0 else "pass"),
                         clip_ratio, "削波采样比例"))

    peak = float(np.max(np.abs(x)))
    checks.append(_check("peak", "warn" if peak > 0.99 or peak < 10 ** (-18 / 20) else "pass",
                         round(_db(peak), 2), "峰值 dBFS"))

    dc = float(abs(np.mean(x)))
    checks.append(_check("dc_offset", "warn" if dc > 0.01 else "pass", round(dc, 5)))

    # 帧级分析（10 ms，向量化；长文本下逐帧 Python 循环会成为瓶颈）
    hop = int(sr * FRAME_MS / 1000)
    n = len(x) // hop
    frames = x[: n * hop].reshape(n, hop)
    frame_db = 20 * np.log10(np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-12)
    voiced = frame_db > -50
    if not voiced.any():
        checks.append(_check("speech_present", "fail", None, "整段音频近似静音"))
        return checks
    # 帧掩码映射回采样级（尾部不足一帧的采样沿用最后一帧状态）
    sample_voiced = np.repeat(voiced, hop)
    sample_voiced = np.concatenate([sample_voiced, np.full(len(x) - len(sample_voiced), voiced[-1])])
    speech_db = _db(np.sqrt(np.mean(x[sample_voiced].astype(np.float64) ** 2)))
    checks.append(_check("loudness_rms", "pass" if -26 <= speech_db <= -14 else "warn",
                         round(speech_db, 2), "有声段 RMS dBFS，目标约 -20"))

    # 内部长静音（首尾除外）
    first, last = np.argmax(voiced), len(voiced) - np.argmax(voiced[::-1])
    longest, run, run_start, where = 0, 0, 0, None
    for i in range(first, last):
        if not voiced[i]:
            if run == 0:
                run_start = i
            run += 1
            if run > longest:
                longest, where = run, run_start
        else:
            run = 0
    longest_s = longest * FRAME_MS / 1000
    checks.append(_check("internal_silence", "warn" if longest_s > 2.5 else "pass", round(longest_s, 2),
                         f"最长内部静音（秒），起点 {where * FRAME_MS / 1000 if where is not None else 0:.2f}s"))

    # 静音区爆点：静音帧内出现的孤立尖峰
    # 候选：采样与前后邻点的偏离 > 0.1；仅当其 ±20 ms 邻域（排除自身 ±2 采样）整体低于 -40 dBFS 时才计为静音区爆点。
    # 注意：不能用“帧是否静音”做门限，因为爆点本身会把所在帧的能量抬高，导致永远判不出（已用注入爆点验证并修正）。
    d2 = np.abs(x[1:-1] - 0.5 * (x[:-2] + x[2:]))
    cand = np.where(d2 > 0.1)[0] + 1
    clicks = 0
    if cand.size:
        cs = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
        w = int(sr * 0.02)
        lo, hi = np.maximum(0, cand - w), np.minimum(len(x), cand + w + 1)
        clo, chi = np.maximum(0, cand - 2), np.minimum(len(x), cand + 3)
        cnt = (hi - lo) - (chi - clo)
        power = (cs[hi] - cs[lo]) - (cs[chi] - cs[clo])
        quiet = (cnt > 0) & (np.sqrt(np.maximum(power, 0.0) / np.maximum(cnt, 1))
                             < 10 ** (-40 / 20))
        cand = cand[quiet]
        if cand.size:  # 相邻采样合并为同一事件
            clicks = int(1 + np.sum(np.diff(cand) > 3))
    checks.append(_check("clicks_in_silence", "warn" if clicks > 0 else "pass", clicks, "静音区孤立尖峰数"))
    return checks


def _cer(ref, hyp):
    """字符错误率（编辑距离 / 参考长度），参考为空时返回 None。"""
    if not ref:
        return None
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


_PUNCT = re.compile(r"[\s，。！？、；：,.!?;:“”‘’\"'()（）【】《》…\-—]+")


_asr_models = {}


def _to_simplified(s):
    """whisper 中文常输出繁体，比对前统一为简体（需 opencc-python-reimplemented；缺失则原样返回）。"""
    try:
        from opencc import OpenCC
    except ImportError:
        return s
    return OpenCC("t2s").convert(s)


def reference_text(text, lexicon=None):
    """朗读形式参考文本（规范化 + 词典 + 去标点）。

    与 speech_rate 共用同一口径，避免两处字符数不一致。
    """
    return _PUNCT.sub("", normalize_zh(text, lexicon))


def run_asr_check(path, expected_text, model_size, profile, lexicon=None):
    def _skip(reason):
        return [_check("asr_cer", "skip", None, reason)]
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return _skip("未安装 faster-whisper，跳过回听识别")
    try:
        from scipy.signal import resample_poly
    except ImportError:
        return _skip("未安装 scipy（回听需重采样到 16 kHz）：pip install scipy")
    if model_size not in _asr_models:  # 同进程内复用模型，避免批量验收时反复加载
        _asr_models[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    model = _asr_models[model_size]
    # 自行解码并重采样到 16 kHz，绕开 faster-whisper 对 PyAV 的版本耦合（不需要 ffmpeg）
    x, sr, _ = load_mono(path)
    if sr != 16000:
        from math import gcd
        g = gcd(16000, sr)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    segs, _ = model.transcribe(x, language="zh", beam_size=5, vad_filter=False,
                               initial_prompt="以下是普通话的句子。")  # 提示简体输出
    hyp_raw = "".join(s.text for s in segs)
    ref = reference_text(expected_text, lexicon)
    # 回听文本只做繁转简 + 去标点；数字串保留 ASR 原样
    hyp = _PUNCT.sub("", _to_simplified(hyp_raw))
    # whisper 会把听到的中文数字写成阿拉伯数字，必须读回中文才能与朗读形式参考文本对齐；
    # 这一步使 asr_cer 无法区分“读法对”与“数字串对”，故另给两个不受该影响的量。
    hyp_canon = _PUNCT.sub("", normalize_zh(_to_simplified(hyp_raw)))
    cer = _cer(ref, hyp_canon)
    cer_text = _cer(mask_numbers(ref), mask_numbers(hyp))
    ref_num = "".join(_MASK_NUM.findall(ref))
    hyp_num = "".join(_MASK_NUM.findall(hyp_canon))
    num_cer = _cer(ref_num, hyp_num) if ref_num else None
    status = "pass" if cer <= profile["cer_warn"] else ("warn" if cer <= profile["cer_fail"] else "fail")
    checks = [_check("asr_cer", status, round(cer, 4),
                     f"回听识别 CER（模型 {model_size}）；ASR 文本：{hyp_raw[:80]}")]
    # 以下两项为诊断量，不参与退出码判定
    checks.append(_check("asr_cer_text", "info", round(cer_text, 4),
                         "遮蔽数词后的 CER（衡量非数字部分的发音，不受规范化链路影响）"))
    checks.append(_check("asr_num_cer", "info", None if num_cer is None else round(num_cer, 4),
                         f"仅数词部分的 CER（参考数词 {len(ref_num)} 字；"
                         f"只能验证听到的数字串，不能验证朗读风格）"))
    return checks


# 阈值依据：default 的 0.10/0.25 是在 9 条用例、2 个中文模型、whisper-small 回听下，
# 与“人工可接受”粗略对齐后选定的经验值；实测单次运行的逐条波动可达 0.08
# （见 references/eval-results.md 第 6 节），因此 strict 的 0.05 低于噪声水平，
# 仅在多次重复取均值后使用。
PROFILES = {
    "default": {"cer_warn": 0.10, "cer_fail": 0.25},
    "strict": {"cer_warn": 0.05, "cer_fail": 0.15},
}


def main(argv=None):
    p = argparse.ArgumentParser(description="TTS 音频验收")
    p.add_argument("audio", help="待检查音频")
    p.add_argument("--text", help="期望朗读的原文（用于字数/时长与回听比对）")
    p.add_argument("--text-file", help="期望朗读的原文文件")
    p.add_argument("--asr", metavar="MODEL", help="启用回听识别，如 base / small（需 faster-whisper）")
    p.add_argument("--lexicon", help="与合成时相同的发音词典，使参考文本与朗读形式一致")
    p.add_argument("--profile", choices=list(PROFILES), default="default",
                   help="default: warn 0.10 / fail 0.25；strict: 0.05 / 0.15（低于单次运行噪声，"
                        "需多次重复取均值后使用）")
    p.add_argument("--report", help="JSON 报告输出路径")
    args = p.parse_args(argv)

    path = Path(args.audio)
    if not path.is_file():
        print(json.dumps({"ok": False, "error": f"文件不存在：{path}"}, ensure_ascii=False))
        return 2
    try:
        x, sr, info = load_mono(path)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"无法解码：{e}"}, ensure_ascii=False))
        return 2

    expected = args.text
    if args.text_file:
        expected = Path(args.text_file).read_text(encoding="utf-8")
    checks = run_signal_checks(x, sr, info)

    if expected:
        ref_chars = len(reference_text(expected, load_lexicon(args.lexicon) if args.lexicon else None))
        rate = ref_chars / (len(x) / sr)
        checks.append(_check("speech_rate_chars_per_s", "pass" if 1.5 <= rate <= 8 else "warn",
                             round(rate, 2), f"规范化后字符数 {ref_chars}；合理范围约 1.5–8 字/秒"))
    if args.asr and expected:
        checks += run_asr_check(path, expected, args.asr, PROFILES[args.profile],
                                load_lexicon(args.lexicon) if args.lexicon else None)
    elif args.asr:
        checks.append(_check("asr_cer", "skip", None, "回听需要 --text 或 --text-file"))

    fails = [c for c in checks if c["status"] == "fail"]
    report = {"file": str(path), "ok": not fails, "profile": args.profile, "checks": checks}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(text, encoding="utf-8")
    print(text)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

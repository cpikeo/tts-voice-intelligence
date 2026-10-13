#!/usr/bin/env python3
"""Signal checks and optional ASR; heuristics are not perceptual quality certification."""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from textnorm import _MASK_NUM, load_lexicon, mask_numbers, normalize_zh, normalize_text, detect_language  # noqa: E402

FRAME_MS = 10


def active_frames(x, sr):
    """10ms RMS > -50dBFS. Signal activity proxy, not phonetic VAD."""
    hop = max(1, int(sr * FRAME_MS / 1000))
    n = (len(x) + hop - 1) // hop
    if n == 0:
        return np.zeros(0, dtype=bool)
    padded = np.pad(x, (0, n * hop - len(x)))
    power = np.sum(padded.reshape(n, hop).astype(np.float64) ** 2, axis=1)
    counts = np.full(n, hop)
    counts[-1] = len(x) - (n - 1) * hop
    return np.sqrt(power / counts) > 10 ** (-50 / 20)


def active_seconds(x, sr):
    if sr <= 0 or not len(x) or not np.isfinite(x).all():
        return 0.0
    hop = max(1, int(sr * FRAME_MS / 1000))
    mask = np.repeat(active_frames(x, sr), hop)[:len(x)]
    return float(np.sum(mask)) / sr



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
    if sr <= 0 or not len(x):
        return [_check("duration", "fail", 0, "空音频/无效采样率")]
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

    hop = max(1, int(sr * FRAME_MS / 1000))
    voiced = active_frames(x, sr)
    if not voiced.any():
        checks.append(_check("speech_present", "fail", None, "整段音频近似静音"))
        return checks
    # 帧掩码映射回采样级（尾部不足一帧的采样沿用最后一帧状态）
    sample_voiced = np.repeat(voiced, hop)[:len(x)]
    speech_db = _db(np.sqrt(np.mean(x[sample_voiced].astype(np.float64) ** 2)))
    checks.append(_check("active_signal_seconds", "info", round(active_seconds(x, sr), 4),
                         "10ms RMS>-50dBFS；不是语音识别/VAD"))
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


def reference_text(text, lexicon=None, language="zh", spoken=False):
    """朗读形式参考文本（规范化 + 词典 + 去标点）。

    与 speech_rate 共用同一口径，避免两处字符数不一致。
    """
    return _PUNCT.sub("", text if spoken else normalize_text(text, language, lexicon)).casefold()


def _run_asr_check(path, expected_text, model_size, profile, lexicon=None, language="zh", spoken=False):
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
    lang = language.replace("_", "-").split("-")[0]
    if lang == "cmn":
        lang = "zh"
    if lang == "auto":
        lang = None
    segs, _ = model.transcribe(x, language=lang, beam_size=5, vad_filter=False)
    hyp_raw = "".join(s.text for s in segs)
    ref = reference_text(expected_text, lexicon, language, spoken)
    if not ref:
        return _skip("无可比较参考字符")
    if len(ref) > 5000 or len(hyp_raw) > 10000:
        return _skip("CER编辑距离输入超过本实现上限；请分章节验收")
    # 回听文本只做繁转简 + 去标点；数字串保留 ASR 原样
    decoded = _to_simplified(hyp_raw) if lang == "zh" else hyp_raw
    hyp = _PUNCT.sub("", decoded).casefold()
    # whisper 会把听到的中文数字写成阿拉伯数字，必须读回中文才能与朗读形式参考文本对齐；
    # 这一步使 asr_cer 无法区分“读法对”与“数字串对”，故另给两个不受该影响的量。
    hyp_canon = reference_text(decoded, language=language)
    cer = _cer(ref, hyp_canon)
    cer_text = _cer(mask_numbers(ref), mask_numbers(hyp))
    ref_num = "".join(_MASK_NUM.findall(ref))
    hyp_num = "".join(_MASK_NUM.findall(hyp_canon))
    num_cer = _cer(ref_num, hyp_num) if ref_num else None
    status = "pass" if cer <= profile["cer_warn"] else ("warn" if cer <= profile["cer_fail"] else "fail")
    checks = [_check("asr_cer", status, round(cer, 4),
                     f"回听识别 CER（模型 {model_size}）；ASR 文本：{hyp_raw[:80]}")]
    checks[0].update(reference_text=ref, hypothesis_text=hyp_raw, canonical_hypothesis=hyp_canon,
                     language=lang, threshold_basis="uncalibrated_heuristic")
    if lang == "zh":
        import importlib.util
        checks[0]["opencc_available"] = importlib.util.find_spec("opencc") is not None
    if lang != "zh":
        return checks  # Chinese number masking is not a multilingual metric
    # 以下两项为诊断量，不参与退出码判定
    checks.append(_check("asr_cer_text", "info", None if cer_text is None else round(cer_text, 4),
                         "中文启发式遮蔽数词CER；存在误遮蔽/残留，非独立发音真值"))
    checks.append(_check("asr_num_cer", "info", None if num_cer is None else round(num_cer, 4),
                         f"仅数词部分的 CER（参考数词 {len(ref_num)} 字；"
                         f"只能验证听到的数字串，不能验证朗读风格）"))
    return checks


def run_asr_check(path, expected_text, model_size, profile, lexicon=None, language="zh", spoken=False):
    try:
        return _run_asr_check(path, expected_text, model_size, profile, lexicon, language, spoken)
    except Exception as e:
        return [_check("asr_cer", "skip", None, f"ASR 不可用：{type(e).__name__}: {e}")]


# Uncalibrated engineering heuristics, not human acceptance or significance bounds.
PROFILES = {
    "default": {"cer_warn": 0.10, "cer_fail": 0.25},
    "strict": {"cer_warn": 0.05, "cer_fail": 0.15},
}


def _main(argv=None):
    p = argparse.ArgumentParser(description="TTS 音频验收")
    p.add_argument("audio", help="待检查音频")
    p.add_argument("--text", help="期望朗读的原文（用于字数/时长与回听比对）")
    p.add_argument("--text-file", help="期望朗读的原文文件")
    p.add_argument("--asr", metavar="MODEL", help="启用回听识别，如 base / small（需 faster-whisper）")
    p.add_argument("--lexicon", help="与合成时相同的发音词典，使参考文本与朗读形式一致")
    p.add_argument("--profile", choices=list(PROFILES), default="default",
                   help="default: warn 0.10 / fail 0.25；strict: 0.05 / 0.15；未校准启发式，非自然度或显著性标准")
    p.add_argument("--language", default="auto", help="期望语言，auto 为脚本启发式")
    p.add_argument("--synthesis-report", help="用本次合成报告的精确 normalized_text 与语言验收")
    p.add_argument("--report", help="JSON 报告输出路径")
    args = p.parse_args(argv)

    path = Path(args.audio)
    if args.report and any(Path(args.report).resolve() == Path(src).resolve() for src in
                           (args.audio, args.text_file, args.lexicon, args.synthesis_report) if src):
        raise ValueError("QA报告不能覆盖音频或其他输入")
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
    spoken = False
    language = args.language
    synthesis = None
    if args.synthesis_report:
        synthesis = json.loads(Path(args.synthesis_report).read_text(encoding="utf-8"))
        import hashlib
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if synthesis.get("output_sha256") != digest:
            raise ValueError("合成报告的输出哈希与待验收音频不符")
        expected = synthesis["normalized_text"]
        language = synthesis["language"]
        spoken = True
    if language == "auto" and expected:
        language = detect_language(expected)
    lex = load_lexicon(args.lexicon) if args.lexicon else None
    checks = run_signal_checks(x, sr, info)
    if synthesis and synthesis.get("direction") is not None:
        dv = synthesis.get("direction_validation") or {}
        beats = dv.get("beats", [])
        all_ok = bool(beats) and all(b.get("signal_execution_verified") for b in beats)
        checks.append(_check(
            "direction_execution",
            "pass" if all_ok else "warn",
            {
                "beat_count": len(beats),
                "supported_controls": dv.get("supported_controls", []),
                "unsupported_or_unverified": dv.get("unsupported_or_unverified", {}),
            },
            "语境→意图→可听变化→引擎支持控制（语速/段间停顿）已核验；情绪/重音不支持，自然度需人评",
        ))
    if info.channels > 1:
        import soundfile as sf
        raw, _ = sf.read(str(path), dtype="float32", always_2d=True)
        ratio = float(np.mean(np.abs(raw) >= .999))
        checks.append(_check("channel_clipping", "fail" if ratio > 5e-4 else "warn" if ratio else "pass", ratio))
        checks.append(_check("multichannel", "warn", info.channels, "单声道混合可能掩盖反相；需逐通道复核"))

    if expected:
        ref_chars = len(reference_text(expected, lex, language, spoken))
        active = active_seconds(x, sr)
        rate = ref_chars / active if active > 0 else None
        checks.append(_check("speech_rate_chars_per_s", "info", None if rate is None else round(rate, 2),
                             f"{ref_chars}规范化字符/有效信号秒；不含静音，非语音VAD，无通用语言阈值"))
        checks.append(_check("delivery_rate_chars_per_s", "info", round(ref_chars / (len(x) / sr), 2) if len(x) else None,
                             "字符/成品总秒数，包含静音，不等同发声速度"))
    if args.asr and expected:
        checks += run_asr_check(path, expected, args.asr, PROFILES[args.profile],
                                lex, language, spoken)
    elif args.asr:
        checks.append(_check("asr_cer", "skip", None, "回听需要 --text 或 --text-file"))

    fails = [c for c in checks if c["status"] == "fail"]
    report = {"file": str(path), "ok": not fails, "profile": args.profile, "language": language, "checks": checks,
              "validation": {"human_listening": "not_performed", "asr": "not_requested" if not args.asr else
                  "skipped" if any(c["name"] == "asr_cer" and c["status"] == "skip" for c in checks) else "performed"},
              "threshold_basis": "uncalibrated_heuristic"}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(text, encoding="utf-8")
    print(text)
    return 1 if fails else 0


def main(argv=None):
    try:
        return _main(argv)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as e:
        print(json.dumps({"ok": False, "error": f"验收输入错误：{e}"}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())

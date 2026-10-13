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


def frame_activity(x, sr):
    """Return (frame RMS, active-frame mask, active seconds, active RMS) in one pass.

    Activity is 10ms RMS > -50dBFS, a signal proxy rather than phonetic VAD. The final partial
    frame is weighted by its real sample count; no sample-sized repeated mask is allocated.
    """
    x = np.asarray(x)
    if sr <= 0 or x.ndim != 1 or not x.size:
        empty = np.zeros(0, dtype=np.float64)
        return empty, np.zeros(0, dtype=bool), 0.0, None

    hop = max(1, int(sr * FRAME_MS / 1000))
    full_frames, tail_samples = divmod(len(x), hop)
    n_frames = full_frames + bool(tail_samples)
    power = np.empty(n_frames, dtype=np.float64)
    if full_frames:
        frames = x[:full_frames * hop].reshape(full_frames, hop)
        power[:full_frames] = np.einsum("ij,ij->i", frames, frames, dtype=np.float64, optimize=False)
    if tail_samples:
        tail = np.asarray(x[full_frames * hop:], dtype=np.float64)
        power[-1] = np.dot(tail, tail)

    frame_rms = np.sqrt(power / hop)
    if tail_samples:
        frame_rms[-1] = np.sqrt(power[-1] / tail_samples)
    active = frame_rms > 10 ** (-50 / 20)
    active_samples = int(np.count_nonzero(active)) * hop
    if tail_samples and active[-1]:
        active_samples -= hop - tail_samples
    active_s = active_samples / sr
    active_rms = (float(np.sqrt(np.sum(power[active], dtype=np.float64) / active_samples))
                  if active_samples else None)
    return frame_rms, active, active_s, active_rms


def active_frames(x, sr):
    """10ms RMS > -50dBFS. Signal activity proxy, not phonetic VAD."""
    return frame_activity(x, sr)[1]


def active_seconds(x, sr):
    x = np.asarray(x)
    if sr <= 0 or not x.size or not np.isfinite(x).all():
        return 0.0
    return frame_activity(x, sr)[2]



def _db(x):
    return 20 * np.log10(max(float(x), 1e-9))


def _check(name, status, value=None, detail=""):
    return {"name": name, "status": status, "value": value, "detail": detail}


def digital_silence_runs(x, sr, min_ms=2.0):
    """Runs (start, end) of exact-zero samples lasting at least min_ms.

    Shorter runs (single zero crossings) are speech, not silence.
    """
    x = np.asarray(x)
    if x.size == 0:
        return []
    min_n = max(1, int(round(sr * min_ms / 1000)))
    z = np.empty(x.size + 2, dtype=np.int8)
    z[0] = z[-1] = 0
    np.equal(x, 0, out=z[1:-1])  # write transitions directly; avoid a full-size bool + int8 copy
    d = np.diff(z)
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (e - s) >= min_n
    return [(int(a), int(b)) for a, b in zip(s[keep], e[keep])]


def boundary_jumps(x, sr, min_ms=2.0):
    """(index, |sample| adjacent to digital silence) at each speech<->silence boundary.

    Digital silence starts from exactly 0, so this value is the sample-to-silence step.
    """
    out = []
    for a, b in digital_silence_runs(x, sr, min_ms):
        if a > 0:
            out.append((a, float(abs(x[a - 1]))))
        if b < len(x):
            out.append((b, float(abs(x[b]))))
    return out


def true_peak_dbtp(x, sr, chunk_s=30.0):
    """Inter-sample peak via 4x oversampling (scipy), computed in chunks to bound memory.

    Each chunk carries 0.5 s of context on both sides so the filter transient is excluded.
    None if scipy is unavailable.
    """
    try:
        from scipy.signal import resample_poly
    except ImportError:
        return None
    n = len(x)
    if n < 8:
        return None
    ov, cs = int(sr * 0.5), max(1, int(sr * chunk_s))
    peak = 0.0
    for s0 in range(0, n, cs):
        s1 = min(n, s0 + cs)
        a, b = max(0, s0 - ov), min(n, s1 + ov)
        up = resample_poly(np.asarray(x[a:b], dtype=np.float64), 4, 1)
        peak = max(peak, float(np.max(np.abs(up[(s0 - a) * 4:(s1 - a) * 4]))))
    return _db(peak)


def integrated_lufs(x, sr):
    """BS.1770 integrated loudness via pyloudnorm. None if unavailable or too short/silent."""
    try:
        import pyloudnorm as pyln
    except ImportError:
        return None
    if len(x) < sr * 0.4:
        return None
    val = float(pyln.Meter(sr).integrated_loudness(np.asarray(x, dtype=np.float64)))
    return val if np.isfinite(val) else None


def load_mono(path):
    """Decode once and retain the original channel metadata for signal checks."""
    from types import SimpleNamespace
    import soundfile as sf
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        info = SimpleNamespace(
            name=str(f.name), samplerate=sr, channels=f.channels, frames=f.frames,
            duration=float(f.frames) / sr, format=f.format, subtype=f.subtype,
            endian=f.endian, format_info=f.format_info, subtype_info=f.subtype_info,
            sections=f.sections, extra_info=f.extra_info,
        )
        raw = f.read(dtype="float32", always_2d=True)
    if info.channels > 1:
        clipped = (np.count_nonzero(raw >= 0.999) + np.count_nonzero(raw <= -0.999))
        info.channel_clipping_ratio = float(clipped / raw.size) if raw.size else 0.0
    return raw.mean(axis=1), sr, info



def _runs(mask):
    """(start, end) of each True run in a boolean array."""
    m = np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def hum_peak_ratio_db(x, sr, inactive_samples):
    """Narrowband 50/60 Hz peak vs neighbouring spectrum, from CONTIGUOUS non-voiced runs >= 1 s.

    Concatenating fragments creates broadband low-frequency leakage, so runs are analysed separately
    and their Welch PSDs averaged. Returns the worst (max) peak-to-neighbourhood ratio in dB, or None.
    """
    try:
        from scipy.signal import welch
    except ImportError:
        return None
    nper = 8192
    runs = [(a, b) for a, b in _runs(inactive_samples) if b - a >= max(nper, sr)]
    if not runs:
        return None
    psd = None
    for a, b in runs:
        f, P = welch(np.asarray(x[a:b], dtype=np.float64), fs=sr, nperseg=nper)
        psd = P if psd is None else psd + P
    worst = None
    for f0 in (50.0, 60.0):
        peak = psd[np.abs(f - f0) <= 1.5]
        neigh = psd[(np.abs(f - f0) >= 4) & (np.abs(f - f0) <= 12)]
        if peak.size == 0 or neigh.size == 0 or np.median(neigh) <= 1e-30:
            continue
        r = 10 * np.log10(np.max(peak) / np.median(neigh) + 1e-30)
        worst = r if worst is None else max(worst, r)
    return None if worst is None else float(worst)


def _hum_check(x, sr, inactive_samples_mask, hop, floor_db):
    mask = np.repeat(inactive_samples_mask, hop)[:len(x)]
    ratio = hum_peak_ratio_db(x, sr, mask)
    if ratio is None:
        return _check("hum_50_60hz_peak_db", "skip", None,
                      "非有声连续段不足 1 s 或未安装 scipy，无法判定 50/60 Hz 嗡声")
    status = "warn" if (ratio > 15 and floor_db > -60) else "pass"
    return _check("hum_50_60hz_peak_db", status, round(ratio, 2),
                  "非有声连续段 50/60 Hz 窄带峰相对邻近频带（±4–12 Hz 中位数）的 dB 值；>15 dB 为嗡声告警")


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

    clip_count = np.count_nonzero(x >= 0.999) + np.count_nonzero(x <= -0.999)
    clip_ratio = float(clip_count / len(x))
    checks.append(_check("clipping", "fail" if clip_ratio > 5e-4 else ("warn" if clip_ratio > 0 else "pass"),
                         clip_ratio, "削波采样比例"))

    peak = max(float(np.max(x)), float(-np.min(x)))
    checks.append(_check("peak", "warn" if peak > 0.99 or peak < 10 ** (-18 / 20) else "pass",
                         round(_db(peak), 2), "峰值 dBFS"))

    dc = float(abs(np.mean(x)))
    # 0.002 ≈ -54 dBFS；原 0.01（-40 dBFS）过宽，无法发现可闻前的直流偏移
    checks.append(_check("dc_offset", "warn" if dc > 0.002 else "pass", round(dc, 5),
                         "整段均值绝对值（阈值 0.002 启发式）"))

    tp = true_peak_dbtp(x, sr)
    if tp is None:
        checks.append(_check("true_peak_dbtp", "skip", None, "未安装 scipy，无法做 4x 过采样真峰值"))
    else:
        checks.append(_check("true_peak_dbtp", "fail" if tp > 0 else ("warn" if tp > -1.0 else "pass"),
                             round(tp, 2), "4x 过采样真峰值 dBTP；>0 表示重建后可能削波"))
    lufs = integrated_lufs(x, sr)
    if lufs is None:
        checks.append(_check("integrated_loudness_lufs", "skip", None,
                             "未安装 pyloudnorm 或音频过短，无法计算 BS.1770 综合响度"))
    else:
        checks.append(_check("integrated_loudness_lufs",
                             "warn" if (lufs < -30 or lufs > -10) else "info", round(lufs, 2),
                             "BS.1770 综合响度；仅极端值告警，目标需由交付场景决定"))

    # 数字静音与语音交界的采样跳变（拼接点/人为停顿点的点击来源）
    jumps = boundary_jumps(x, sr)
    if jumps:
        worst = max(v for _, v in jumps)
        n_over = sum(1 for _, v in jumps if _db(v) > -40)
        checks.append(_check("boundary_discontinuity", "warn" if n_over else "pass",
                             round(_db(worst), 2),
                             f"数字静音交界处的跳变：超过 -40 dBFS 的 {n_over}/{len(jumps)} 处"
                             "（启发式；未校准的可闻阈值）"))
    edge = max(abs(float(x[0])), abs(float(x[-1])))
    checks.append(_check("file_edge_step", "warn" if _db(edge) > -40 else "pass", round(_db(edge), 2),
                         "文件首/末采样幅度（非零起止会产生开头/结尾咔哒声）"))

    frame_rms, voiced, active_s, speech_rms = frame_activity(x, sr)
    if not voiced.any():
        checks.append(_check("speech_present", "fail", None, "整段音频近似静音"))
        return checks
    speech_db = _db(speech_rms)
    checks.append(_check("active_signal_seconds", "info", round(active_s, 4),
                         "10ms RMS>-50dBFS；不是语音识别/VAD"))
    checks.append(_check("loudness_rms", "pass" if -26 <= speech_db <= -14 else "warn",
                         round(speech_db, 2), "有声段 RMS dBFS，目标约 -20"))

    # 底噪与嗡声：复用同一批 10 ms 帧 RMS；仅告警，不据此降噪。
    if frame_rms.size >= 50:
        hop = max(1, int(sr * FRAME_MS / 1000))
        # Preserve the noise/hum estimator's prior zero-padded final-frame convention; the
        # activity estimator above intentionally uses the true tail length for VAD proxy math.
        noise_frame_rms = frame_rms
        tail_samples = len(x) % hop
        if tail_samples:
            noise_frame_rms = frame_rms.copy()
            noise_frame_rms[-1] *= np.sqrt(tail_samples / hop)
        p10 = float(np.percentile(noise_frame_rms, 10))
        floor_db = _db(p10)
        checks.append(_check("noise_floor_dbfs", "warn" if floor_db > -45 else "pass", round(floor_db, 2),
                             "10 ms 帧 RMS 第 10 百分位 dBFS（底噪代理）；> -45 为可闻底噪的启发式阈值"))
        checks.append(_hum_check(x, sr, noise_frame_rms <= 10 ** (-50 / 20), hop, floor_db))

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
    d2 = np.empty(max(0, len(x) - 2), dtype=np.result_type(x.dtype, np.float32))
    np.add(x[:-2], x[2:], out=d2)
    d2 *= 0.5
    np.subtract(x[1:-1], d2, out=d2)
    np.abs(d2, out=d2)
    cand = np.flatnonzero(d2 > 0.1) + 1
    clicks = 0
    if cand.size:
        cs = np.empty(len(x) + 1, dtype=np.float64)
        cs[0] = 0.0
        np.multiply(x, x, out=cs[1:])
        np.cumsum(cs[1:], out=cs[1:])
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
        ratio = info.channel_clipping_ratio
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

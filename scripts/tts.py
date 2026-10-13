#!/usr/bin/env python3
"""Deterministic TTS execution. See SKILL.md for decisions and README for CLI."""
import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from textnorm import load_lexicon, normalize_zh, normalize_text, detect_language, text_warnings  # noqa: E402

VERSION = "0.4.0"

from engines import (  # noqa: E402
    DEFAULT_TIMEOUT,
    Engine,
    EngineError,
    EngineTimeoutError,
    EspeakEngine,
    PiperEngine,
    XTTSEngine,
    run_with_timeout,
    validate_authorization,
)


def build_engine(args):
    timeout = getattr(args, "timeout", DEFAULT_TIMEOUT)
    if args.engine == "piper":
        if args.reference_audio or args.authorization:
            raise EngineError("Piper 不支持克隆")
        return PiperEngine(model=args.model, config=args.config, speaker=args.speaker,
                           length_scale=args.length_scale, noise_scale=args.noise_scale,
                           noise_w_scale=args.noise_w, use_cuda=args.cuda,
                           language=args.language, timeout=timeout)
    if args.engine != "piper" and (args.noise_scale != .667 or args.noise_w != .8):
        raise EngineError("该适配器不支持 Piper 噪声参数")
    if args.engine == "espeak":
        if args.reference_audio or args.authorization or args.speaker is not None or args.cuda or args.model or args.config:
            raise EngineError("eSpeak 不接受克隆、speaker id、CUDA 或模型/config参数")
        return EspeakEngine(language=args.language, length_scale=args.length_scale, timeout=timeout)
    if args.engine == "xtts":
        if args.speaker is not None or args.length_scale != 1.0:
            raise EngineError("XTTS 适配器不暴露 speaker id / 语速参数")
        return XTTSEngine(
            args.model, args.config, args.language, args.reference_audio,
            args.authorization, args.purpose, args.cuda, timeout=timeout,
            signing_key=getattr(args, "auth_signing_key", None),
            require_signature=getattr(args, "require_signed_auth", False),
        )
    raise EngineError(f"未知引擎：{args.engine}（piper/espeak/xtts）")


# ---------------------------------------------------------------- 分段

@dataclass
class Segment:
    text: str
    pause_after: str  # "para" | "sentence" | "clause" | "none"


_PUNCT_ONLY = re.compile(r"^[\s，。！？!?…、；;：:,.~—\-]+$")


def _has_content(s):
    """是否含可朗读内容。纯标点片段（如“！”“…”）不能单独成段，
    否则会被引擎当成一个独立句子合成，并额外插入句间停顿。"""
    return bool(s) and not _PUNCT_ONLY.match(s)


def _split_long(sentence, max_chars):
    if len(sentence) <= max_chars:
        return [sentence]
    parts = [p for p in re.split(r"(?<=[，,；;、：:])", sentence) if p]
    pieces, buf = [], ""
    for p in parts:
        while len(p) > max_chars:  # 无标点的超长片段硬切
            if buf:
                pieces.append(buf)
                buf = ""
            # Prefer whitespace boundaries for Latin words; hard split only when unavoidable.
            cut = p.rfind(" ", 0, max_chars)
            cut = cut + 1 if cut > 0 else max_chars
            pieces.append(p[:cut])
            p = p[cut:].lstrip()
        if len(buf) + len(p) <= max_chars:
            buf += p
        else:
            pieces.append(buf)
            buf = p
    if buf:
        pieces.append(buf)
    return pieces


def segment(text, max_chars=90):
    """按段落、句子、子句切分；每段记录其后应插入的停顿类型。

    纯标点片段（如“真的吗？！”里的“！”）并入上一段，不单独成段。
    """
    if max_chars <= 0:
        raise ValueError("max_chars 必须大于零")
    segs = []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    for para in paragraphs:
        sents = [s.strip() for s in re.split(r"(?<=[。！？!?…])\s*|(?<=[.?!])\s+", para) if s.strip()]
        items = []
        for si, sent in enumerate(sents):
            pieces = _split_long(sent, max_chars)
            for k, piece in enumerate(pieces):
                end_of_sent = k == len(pieces) - 1
                items.append((piece, end_of_sent, end_of_sent and si == len(sents) - 1))
        for piece, sent_end, para_end in items:
            if not _has_content(piece):
                # 并入上一段：否则引擎会把它当成一个独立句子合成，并额外插入句间停顿
                if not segs:
                    continue
                segs[-1].text += piece
                if para_end:
                    segs[-1].pause_after = "para"
                elif sent_end and segs[-1].pause_after == "clause":
                    segs[-1].pause_after = "sentence"
                continue
            segs.append(Segment(piece, "para" if para_end else ("sentence" if sent_end else "clause")))
    if segs:
        segs[-1].pause_after = "none"
    return segs


# ---------------------------------------------------------------- 音频处理

def _db(x):
    return 20 * np.log10(max(x, 1e-9))


def _voiced(x, thr_db=-50.0):
    return np.abs(x) > 10 ** (thr_db / 20)


def trim_silence(x, sr, pad_ms=30):
    m = _voiced(x)
    if not m.any():
        return x
    i0, i1 = np.argmax(m), len(m) - np.argmax(m[::-1])
    pad = int(sr * pad_ms / 1000)
    return x[max(0, i0 - pad): min(len(x), i1 + pad)]


def voiced_rms_db(x, sr=22050):
    """有声部分 RMS（dBFS）；整段无声时返回 None。"""
    from qa import active_frames
    v = x[np.repeat(active_frames(x, sr), max(1, int(sr * 0.01)))[:len(x)]]
    if v.size == 0:
        return None
    return _db(np.sqrt(np.mean(v.astype(np.float64) ** 2)))


def level_to(x, target_db, sr=22050):
    """段内响度对齐到 target_db（基于有声帧 RMS），增益限制在 ±12 dB。"""
    rms_db = voiced_rms_db(x, sr)
    if rms_db is None:
        return x
    gain = np.clip(10 ** ((target_db - rms_db) / 20), 10 ** (-12 / 20), 10 ** (12 / 20))
    return (x * gain).astype(np.float32)


def fade(x, sr, ms=12):
    n = min(int(sr * ms / 1000), len(x) // 2)
    if n <= 0:
        return x
    ramp = np.linspace(0, 1, n, dtype=np.float32)
    x = x.copy()
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


_ALLOWED_DIRECTION_TOP_KEYS = {
    "intent", "audience", "character", "context", "audible_goal",
    "unsupported_capabilities", "beats",
}
_ALLOWED_BEAT_KEYS = {
    "index", "text", "context", "intent", "rationale",
    "audible_change", "length_scale", "pause_after_ms", "verification",
}


def _validate_direction_contract(direction, segs, engine):
    if direction is None:
        return {}
    if not isinstance(direction, dict) or any(not direction.get(k) for k in ("intent", "audience", "character")):
        raise ValueError("direction 需 intent/audience/character")
    if set(direction) - _ALLOWED_DIRECTION_TOP_KEYS:
        raise ValueError("不支持的声音控制；不能伪装情绪/重音/音色能力")
    beats = {}
    for beat in direction.get("beats", []):
        if not isinstance(beat, dict) or (set(beat) - _ALLOWED_BEAT_KEYS):
            raise ValueError("不支持的声音控制；不能伪装情绪/重音能力")
        index = beat.get("index")
        if not isinstance(index, int) or not 0 <= index < len(segs) or index in beats:
            raise ValueError("beat 索引无效或重复")
        if beat.get("text") != segs[index].text or not beat.get("intent") or not beat.get("rationale"):
            raise ValueError("beat 文本不匹配或缺少语义理由")
        if "length_scale" in beat and (not getattr(engine, "capabilities", {}).get("rate") or
                not np.isfinite(beat["length_scale"]) or not .5 <= beat["length_scale"] <= 2):
            raise ValueError("引擎不支持该语速或倍率超界")
        if "pause_after_ms" in beat:
            pm = beat["pause_after_ms"]
            if not isinstance(pm, (int, float)) or isinstance(pm, bool) or not np.isfinite(pm) or not 0 <= pm <= 5000:
                raise ValueError("pause_after_ms 必须在 0–5000 ms 之间")
        beats[index] = beat
    return beats


def _plan_digest(engine, language, normalized_text, max_chars, processing, target_db, pauses, peak_db, direction):
    bound = getattr(engine, "bound_assets", None)
    asset_hashes = {}
    if isinstance(bound, dict):
        for k, v in bound.items():
            if isinstance(v, dict) and "sha256" in v:
                asset_hashes[k] = v["sha256"]
    payload = {
        "engine": getattr(engine, "name", "unknown"),
        "sample_rate": int(engine.sample_rate),
        "language": language,
        "normalized_text": normalized_text,
        "max_chars": max_chars,
        "processing": processing,
        "target_db": target_db if processing == "legacy" else None,
        "pauses": pauses,
        "peak_db": peak_db,
        "direction": direction,
        "assets": asset_hashes,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(fd)
    try:
        Path(tmp).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def synthesize_text(engine, text, *, normalize=True, max_chars=90, target_db=-20.0,
                    pauses=None, peak_db=-1.0, lexicon=None, language=None, processing="legacy",
                    max_output_seconds=1800, direction=None, segment_timeout=None,
                    cancel_event=None, checkpoint_dir=None):
    """完整流水线，返回 (audio float32, sr, report dict)。"""
    pauses = pauses if pauses is not None else {"para": 700, "sentence": 400, "clause": 200, "none": 0}
    sr = engine.sample_rate
    if not isinstance(sr, int) or sr < 1000 or max_chars <= 0:
        raise ValueError("无效采样率 / max_chars")
    if processing not in ("minimal", "legacy") or any(
            not np.isfinite(v) or v < 0 for v in pauses.values()):
        raise ValueError("无效处理模式或停顿参数")
    if not np.isfinite(target_db) or not -60 <= target_db <= -3 or not np.isfinite(peak_db) or not -12 <= peak_db <= 0:
        raise ValueError("无效响度/峰值参数")
    if not np.isfinite(max_output_seconds) or not 1 <= max_output_seconds <= 1800:
        raise ValueError("max_output_seconds 需为 1–1800 秒（内存边界，不支持无限长）")
    eff_timeout = segment_timeout if segment_timeout is not None else getattr(engine, "timeout", None)
    if eff_timeout is not None and (not isinstance(eff_timeout, (int, float)) or not np.isfinite(eff_timeout) or eff_timeout <= 0):
        raise ValueError("segment_timeout 必须为大于 0 的有限秒数")
    original = text
    if len(text) > 100000:
        raise ValueError("单次输入超过100000字符；请分章节")
    language = language or getattr(engine, "language", "zh")
    if language == "auto":
        language = detect_language(text)
    text = normalize_text(text, language, lexicon) if normalize else text
    segs = segment(text, max_chars=max_chars)
    if not segs:
        raise ValueError("输入文本为空（去除空白后无可朗读内容）")

    beats = _validate_direction_contract(direction, segs, engine)
    plan_sha256 = _plan_digest(engine, language, text, max_chars, processing, target_db, pauses, peak_db, direction)
    ckpt_path = Path(checkpoint_dir) if checkpoint_dir else None
    ckpt_state = {"plan_sha256": plan_sha256, "sample_rate": sr, "segment_count": len(segs), "segments": {}}
    if ckpt_path is not None:
        ckpt_path.mkdir(parents=True, exist_ok=True)
        state_file = ckpt_path / "state.json"
        if state_file.is_file():
            prior = json.loads(state_file.read_text(encoding="utf-8"))
            if prior.get("plan_sha256") != plan_sha256 or prior.get("sample_rate") != sr or prior.get("segment_count") != len(segs):
                raise ValueError("断点续合成状态与当前文本/模型/参数不匹配；拒绝混用旧分段")
            ckpt_state = prior

    parts, seg_reports, warnings = [], [], text_warnings(original, language)
    output_samples = 0
    t_total = 0.0
    resumed_count = 0
    synthesized_count = 0

    for i, seg in enumerate(segs):
        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            raise EngineError(f"第 {i + 1} 段合成前收到取消信号")
        beat = beats.get(i, {})
        raw = None
        dt = 0.0
        attempt = 0
        resumed = False

        if ckpt_path is not None:
            saved_meta = ckpt_state.get("segments", {}).get(str(i))
            seg_wav = ckpt_path / f"seg_{i:04d}.wav"
            if isinstance(saved_meta, dict) and seg_wav.is_file() and sha256_file(seg_wav) == saved_meta.get("sha256"):
                if getattr(engine, "name", "") == "xtts":
                    validate_authorization(
                        engine.authorization_path, engine.reference, engine.purpose,
                        signing_key=getattr(engine, "signing_key", None),
                        require_signature=getattr(engine, "require_signature", False),
                    )
                import soundfile as sf
                loaded, loaded_sr = sf.read(str(seg_wav), dtype="float32")
                if loaded_sr == sr and loaded.ndim == 1 and loaded.size > 0 and np.isfinite(loaded).all():
                    raw = loaded
                    dt = float(saved_meta.get("synth_seconds", 0.0))
                    attempt = int(saved_meta.get("attempts", 1))
                    resumed = True
                    resumed_count += 1

        if raw is None:
            saved_rate = None
            if "length_scale" in beat:
                if engine.name == "piper":
                    saved_rate = engine.syn.length_scale
                    engine.syn.length_scale = beat["length_scale"]
                elif engine.name == "espeak":
                    saved_rate = engine.speed
                    engine.speed = round(175 / beat["length_scale"])
                else:
                    raise ValueError("该适配器未实现逐段语速")
            t0 = time.perf_counter()
            last_err = None
            try:
                for attempt in (1, 2):  # 单段可恢复故障最多重试一次；超时与取消绝不重试
                    if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                        raise EngineError(f"第 {i + 1} 段合成已取消")
                    try:
                        call_out = (
                            run_with_timeout(engine.synthesize, eff_timeout, seg.text)
                            if eff_timeout is not None else engine.synthesize(seg.text)
                        )
                        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                            raise EngineError(f"第 {i + 1} 段合成完成后收到取消信号")
                        raw = np.asarray(call_out)
                        if raw.ndim != 1 or raw.size == 0 or not np.isfinite(raw).all():
                            raise EngineError("引擎输出为空、非有限或非单声道")
                        if np.max(np.abs(raw)) > 1.0001:
                            raise EngineError("引擎输出超出 [-1,1]；拒绝掩盖异常")
                        break
                    except EngineTimeoutError as e:
                        if ckpt_path is not None:
                            ckpt_state["failed_segment"] = i
                            _atomic_write_json(ckpt_path / "state.json", ckpt_state)
                        raise EngineTimeoutError(f"第 {i + 1} 段合成超时（{seg.text[:30]}…）：{e}") from e
                    except EngineError as e:
                        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                            raise
                        last_err = e
                else:
                    if ckpt_path is not None:
                        ckpt_state["failed_segment"] = i
                        _atomic_write_json(ckpt_path / "state.json", ckpt_state)
                    raise EngineError(f"第 {i + 1} 段合成失败（{seg.text[:30]}…）：{last_err}") from last_err
            finally:
                if saved_rate is not None:
                    if engine.name == "piper":
                        engine.syn.length_scale = saved_rate
                    else:
                        engine.speed = saved_rate
            dt = time.perf_counter() - t0
            synthesized_count += 1
            if ckpt_path is not None:
                seg_wav = ckpt_path / f"seg_{i:04d}.wav"
                write_audio(seg_wav, raw.astype(np.float32), sr, "wav", subtype="FLOAT")
                ckpt_state.setdefault("segments", {})[str(i)] = {
                    "index": i,
                    "text": seg.text,
                    "sha256": sha256_file(seg_wav),
                    "synth_seconds": round(dt, 4),
                    "attempts": attempt,
                }
                ckpt_state.pop("failed_segment", None)
                _atomic_write_json(ckpt_path / "state.json", ckpt_state)

        t_total += dt
        x = raw.astype(np.float32)
        if processing == "legacy":
            x = trim_silence(x, sr)
        if x.size < sr * 0.05:
            warnings.append(f"第 {i + 1} 段输出过短，可能发音缺失：{seg.text[:30]}")
        if processing == "legacy":
            x = fade(level_to(x, target_db, sr), sr)
        dur = len(x) / sr
        rms_db = voiced_rms_db(x, sr)
        if rms_db is None:
            warnings.append(f"第 {i + 1} 段几乎无声（有声帧低于 -50 dBFS），未做响度对齐：{seg.text[:30]}")
        elif processing == "legacy" and abs(rms_db - target_db) > 1.0:
            warnings.append(f"第 {i + 1} 段响度 {rms_db:.1f} dBFS 偏离目标 {target_db:.1f} dBFS"
                            f"（已触发 ±12 dB 增益限制）：{seg.text[:30]}")
        gap = beat.get("pause_after_ms", pauses.get(seg.pause_after, 0))
        seg_reports.append({
            "index": i, "text": seg.text, "seconds": round(dur, 3),
            "synth_seconds": round(dt, 3), "pause_after": seg.pause_after,
            "pause_after_ms": round(float(gap), 1),
            "rms_dbfs": None if rms_db is None else round(rms_db, 2),
            "attempts": attempt, "resumed_from_checkpoint": resumed,
            "direction": beat, "start_sample": output_samples, "samples": len(x),
        })
        parts.append(x)
        output_samples += len(x)
        if gap:
            gap_samples = int(sr * gap / 1000)
            parts.append(np.zeros(gap_samples, dtype=np.float32))
            output_samples += gap_samples
        if output_samples > sr * max_output_seconds:
            raise EngineError("输出超过时长上限；请按章节分批交付")

    audio = np.concatenate(parts)
    peak = float(np.max(np.abs(audio)))
    limit = 10 ** (peak_db / 20)
    if peak > limit:
        audio = audio * (limit / peak)
        warnings.append(f"峰值 {_db(peak):.1f} dBFS 超过上限，已整体衰减到 {peak_db} dBFS")

    from qa import active_seconds, reference_text
    for srep in seg_reports:
        final = audio[srep["start_sample"]:srep["start_sample"] + srep["samples"]]
        rms = voiced_rms_db(final, sr)
        srep["rms_dbfs"] = None if rms is None else round(rms, 2)
        act_s = active_seconds(final, sr)
        ref_len = len(reference_text(srep["text"], language=language, spoken=True))
        srep["active_seconds"] = round(act_s, 3)
        srep["speech_rate_chars_per_s"] = round(ref_len / act_s, 2) if act_s > 0 else None

    direction_validation = None
    if direction is not None:
        undirected_rates = [
            s["speech_rate_chars_per_s"] for s in seg_reports
            if not s["direction"].get("length_scale") and s["speech_rate_chars_per_s"] is not None
        ]
        baseline_rate = float(np.mean(undirected_rates)) if undirected_rates else None
        beat_verifications = []
        for srep in seg_reports:
            b = srep["direction"]
            if not b:
                continue
            ls = b.get("length_scale")
            pm = b.get("pause_after_ms")
            applied = {}
            if ls is not None:
                applied["length_scale"] = ls
            if pm is not None:
                applied["pause_after_ms"] = pm
            rate_val = srep["speech_rate_chars_per_s"]
            rate_effect_ok = (srep["active_seconds"] > 0) if ls is not None else True
            pause_effect_ok = (abs(srep["pause_after_ms"] - float(pm)) < 1.0) if pm is not None else True
            ver = {
                "index": srep["index"],
                "context": b.get("context") or direction.get("context", "unspecified"),
                "intent": b["intent"],
                "rationale": b["rationale"],
                "audible_change": b.get("audible_change", "rate_or_pause_adjustment"),
                "applied_controls": applied,
                "unsupported_controls": {
                    "emotion": "unsupported",
                    "emphasis": "unsupported",
                    "timbre_shift": "unsupported",
                    "cross_segment_hidden_state": "unsupported",
                },
                "measured": {
                    "duration_seconds": srep["seconds"],
                    "active_seconds": srep["active_seconds"],
                    "speech_rate_chars_per_s": rate_val,
                    "undirected_mean_rate_chars_per_s": round(baseline_rate, 2) if baseline_rate is not None else None,
                    "pause_after_ms": srep["pause_after_ms"],
                },
                "signal_execution_verified": bool(applied) and rate_effect_ok and pause_effect_ok,
                "perceptual_verification": "unverified_requires_human_listening",
            }
            srep["direction_verification"] = ver
            beat_verifications.append(ver)
        direction_validation = {
            "context": direction.get("context", "unspecified"),
            "intent": direction["intent"],
            "audience": direction["audience"],
            "character": direction["character"],
            "audible_goal": direction.get("audible_goal", "controlled_tempo_and_boundary_pacing"),
            "supported_controls": [k for k in ("length_scale", "pause_after_ms") if k == "pause_after_ms" or getattr(engine, "capabilities", {}).get("rate")],
            "unsupported_or_unverified": {
                "emotion": "unsupported",
                "emphasis": "unsupported",
                "timbre_modulation": "unsupported",
                "cross_segment_prosody_continuity": "unverified",
                "human_perceptual_effect": "unverified_requires_human_listening",
            },
            "beats": beat_verifications,
        }

    total = len(audio) / sr
    rms_list = [s["rms_dbfs"] for s in seg_reports if s["rms_dbfs"] is not None]
    rms_std = round(float(np.std(rms_list)), 3) if len(rms_list) > 1 else 0.0
    if rms_std > 1.0:
        warnings.append(f"段间响度标准差 {rms_std:.2f} dB 偏大（>1 dB），跨段一致性可能不足")
    report = {
        "ok": True,  # synthesis success only; not QA acceptance
        "direction": direction,
        "direction_validation": direction_validation,
        "engine": engine.name,
        "engine_info": engine.available()[1],
        "sample_rate": sr,
        "normalized": normalize,
        "original_text": original,
        "normalized_text": text,
        "segments": seg_reports,
        "segment_count": len(segs),
        "duration_seconds": round(total, 3),
        "speech_seconds": round(active_seconds(audio, sr), 3),
        "segment_audio_seconds": round(sum(s["seconds"] for s in seg_reports), 3),
        "language": language,
        "capabilities": getattr(engine, "capabilities", {}),
        "authorization": getattr(engine, "authorization", None),
        "bound_assets": getattr(engine, "bound_assets", None),
        "checkpoint_recovery": {
            "enabled": ckpt_path is not None,
            "checkpoint_dir": str(ckpt_path) if ckpt_path is not None else None,
            "plan_sha256": plan_sha256,
            "resumed_segments": resumed_count,
            "synthesized_segments": synthesized_count,
        },
        "validation": {"human_listening": "not_performed", "asr": "not_performed"},
        "synth_seconds": round(t_total, 3),
        "rtf": round(t_total / total, 3) if total > 0 else None,
        "peak_dbfs": round(_db(float(np.max(np.abs(audio)))), 2),
        # 段间一致性：各段对齐后的有声 RMS 与其标准差（客观代理，音色一致性仍需人工听评）
        "segment_rms_std_db": rms_std,
        "params": {"normalize": normalize, "max_chars": max_chars, "target_db": target_db if processing == "legacy" else None,
                   "pause_ms": pauses, "peak_limit_dbfs": peak_db,
                   "lexicon_entries": len(lexicon) if lexicon else 0, "processing": processing,
                   "max_output_seconds": max_output_seconds,
                   "segment_timeout": eff_timeout},
        "warnings": warnings,
    }
    return audio.astype(np.float32), sr, report


SUPPORTED_FORMATS = ("wav", "flac", "ogg")


def _stage_audio_file(path, audio, sr, fmt=None, subtype=None):
    import soundfile as sf
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = (fmt or path.suffix.lstrip(".")).lower()
    resolved_subtype = subtype or {"wav": "PCM_24", "flac": "PCM_16", "ogg": "VORBIS"}.get(ext)
    if resolved_subtype is None:
        raise ValueError(f"不支持的输出格式：.{ext}（支持 {'/'.join(SUPPORTED_FORMATS)}；"
                         f"MP3 需要额外编码器，本技能不支持）")
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".staged." + ext)
    os.close(fd)
    try:
        sf.write(tmp, audio, sr, subtype=resolved_subtype, format=ext.upper())
        return Path(tmp)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_audio(path, audio, sr, fmt=None, subtype=None):
    path = Path(path)
    tmp = _stage_audio_file(path, audio, sr, fmt=fmt, subtype=subtype)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def transactional_write_bundle(master_path, audio, sr, derived_formats=None, report=None, report_path=None):
    """All-or-nothing transactional commit for master WAV + derived audio + JSON report.

    Stages all files to temporary paths first, computes hashes on staged artifacts, backs up any
    existing destination files, and rolls back the entire group if any stage or replace fails.
    """
    master_path = Path(master_path)
    staged_pairs = []  # list of (tmp_path, final_path)
    backups = []       # list of (final_path, backup_path, existed)
    committed = []
    try:
        tmp_master = _stage_audio_file(master_path, audio, sr, "wav")
        staged_pairs.append((tmp_master, master_path))
        derived_paths = []
        derived_hashes = {}
        for fmt in derived_formats or []:
            dst = master_path.with_suffix("." + fmt)
            tmp_drv = _stage_audio_file(dst, audio, sr, fmt)
            staged_pairs.append((tmp_drv, dst))
            derived_paths.append(str(dst))
            derived_hashes[str(dst)] = sha256_file(tmp_drv)

        master_sha = sha256_file(tmp_master)
        if report is not None:
            report["output"] = str(master_path)
            report["output_sha256"] = master_sha
            report["derived"] = derived_paths
            report["derived_sha256"] = derived_hashes
            report["delivery_transaction"] = "atomic_group_committed"
        if report_path is not None and report is not None:
            rp = Path(report_path)
            rp.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_rep = tempfile.mkstemp(dir=rp.parent, suffix=".staged.json")
            os.close(fd)
            tmp_rep_path = Path(tmp_rep)
            staged_pairs.append((tmp_rep_path, rp))
            tmp_rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        # Prepare rollback backups for any pre-existing targets
        for _tmp, dst in staged_pairs:
            if dst.exists():
                fd, bkp = tempfile.mkstemp(dir=dst.parent, suffix=".rollback.bak")
                os.close(fd)
                bkp_path = Path(bkp)
                bkp_path.write_bytes(dst.read_bytes())
                backups.append((dst, bkp_path, True))
            else:
                backups.append((dst, None, False))

        # Commit all staged files
        for (tmp_p, dst), _b in zip(staged_pairs, backups):
            os.replace(tmp_p, dst)
            committed.append(dst)

        # Clean up rollback backups on complete success
        for _dst, bkp_path, _existed in backups:
            if bkp_path is not None:
                bkp_path.unlink(missing_ok=True)
        return master_path, derived_paths
    except BaseException:
        # Clean up any remaining staged files
        for tmp_p, _dst in staged_pairs:
            tmp_p.unlink(missing_ok=True)
        # Roll back any already-committed targets in reverse order
        for dst, bkp_path, existed in reversed(backups):
            try:
                if existed and bkp_path is not None and bkp_path.exists():
                    os.replace(bkp_path, dst)
                elif not existed and dst in committed:
                    dst.unlink(missing_ok=True)
            except OSError:
                pass
            finally:
                if bkp_path is not None:
                    bkp_path.unlink(missing_ok=True)
        raise


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def model_provenance(path):
    """模型可追溯信息：路径、大小、sha256。用于跨任务复现与许可核对。"""
    if not path:
        return {"path": None, "exists": False}
    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "exists": False}
    return {"path": str(p), "name": p.name, "bytes": p.stat().st_size,
            "sha256": sha256_file(p), "exists": True}


# ---------------------------------------------------------------- CLI

def cmd_check(args):
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "skill_version": VERSION, "engines": {}}
    try:
        import soundfile as sf
        info["soundfile"] = sf.__libsndfile_version__
    except ImportError:
        info["soundfile"] = None
        info["hint"] = "pip install soundfile numpy"
    try:
        e = build_engine(args)
        info["engines"][args.engine] = {"available": True, "info": e.available()[1],
                                        "capabilities": e.capabilities}
        if getattr(args, "model", None):
            info["engines"][args.engine]["model"] = model_provenance(args.model)
    except Exception as e:  # noqa: BLE001 - check 必须给出诊断而不是抛栈
        info["engines"][args.engine] = {"available": False,
                                        "reason": f"{type(e).__name__}: {e}"}
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0 if all(v["available"] for v in info["engines"].values()) else 2


def _validate_outputs(out, formats):
    """提前校验输出路径，避免合成完成后才失败（长文本浪费一次完整合成）。"""
    out = Path(out)
    if out.suffix.lower() != ".wav":
        raise ValueError(f"母版必须是 WAV：不支持的输出格式 {out.suffix}"
                         f"（支持 {'/'.join(SUPPORTED_FORMATS)}；MP3 需额外编码器，本技能不支持）")
    for fmt in formats or []:
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(f"不支持的派生格式：{fmt}")
    return out


def _run_isolated_subprocess_synth(args, raw_argv):
    """Execute synthesis in an isolated child process group with offline env & hard timeout."""
    filtered_argv = [a for a in raw_argv if a != "--isolate-model"]
    # Resolve all path arguments to absolute paths because child runs in an isolated empty temp cwd
    path_flags = {"--model", "--config", "--input", "--out", "--report", "--lexicon",
                  "--reference-audio", "--authorization", "--direction", "--checkpoint-dir"}
    resolved_argv = []
    i = 0
    while i < len(filtered_argv):
        tok = filtered_argv[i]
        resolved_argv.append(tok)
        if tok in path_flags and i + 1 < len(filtered_argv):
            resolved_argv.append(str(Path(filtered_argv[i + 1]).resolve()))
            i += 2
            continue
        i += 1

    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    for k in ("PYTHONPATH", "VIRTUAL_ENV", "LD_LIBRARY_PATH", "TTS_AUTH_SIGNING_KEY", "TTS_REQUIRE_SIGNED_AUTH"):
        if k in os.environ:
            env[k] = os.environ[k]

    total_timeout = max(float(args.timeout or DEFAULT_TIMEOUT) * 4.0, 120.0)
    with tempfile.TemporaryDirectory(prefix="tts_isolated_") as iso_cwd:
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *resolved_argv],
            cwd=iso_cwd, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=total_timeout)
        except subprocess.TimeoutExpired as e:
            if hasattr(os, "killpg"):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
            proc.kill()
            proc.wait(timeout=2)
            raise EngineTimeoutError(f"隔离子进程合成超时（>{total_timeout:.1f}s）") from e
        except BaseException:
            if hasattr(os, "killpg"):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
            proc.kill()
            proc.wait(timeout=2)
            raise
    if stderr:
        sys.stderr.write(stderr.decode("utf-8", errors="replace"))
    if stdout:
        sys.stdout.write(stdout.decode("utf-8", errors="replace"))
    return proc.returncode


def cmd_synth(args):
    if getattr(args, "isolate_model", False):
        return _run_isolated_subprocess_synth(args, args._raw_argv)
    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    else:
        text = args.text or ""
    if not text.strip():
        print("错误：没有输入文本（--text 或 --input）", file=sys.stderr)
        return 1
    try:
        out = _validate_outputs(args.out, args.format)
        destinations = [out] + [out.with_suffix("." + fmt) for fmt in args.format or []]
        if args.report:
            destinations.append(Path(args.report))
        sources = [Path(v) for v in (args.input, args.model, args.config, args.lexicon,
                                    args.reference_audio, args.authorization, args.direction) if v]
        if args.engine == "piper" and args.model and not args.config:
            sources.append(Path(str(args.model) + ".json"))
        if len({p.resolve() for p in destinations}) != len(destinations) or any(
                d.resolve() == src.resolve() for d in destinations for src in sources):
            raise ValueError("输出/报告不能互相覆盖或覆盖输入/私有资源")
        if getattr(args, "checkpoint_dir", None):
            ckpt_res = Path(args.checkpoint_dir).resolve()
            if any(ckpt_res == p.resolve() for p in destinations + sources):
                raise ValueError("checkpoint-dir 不能与输入或输出路径冲突")
        if args.processing == "minimal" and ((args.pause_para, args.pause_sentence, args.pause_clause) != (700, 400, 200) or args.target_db != -20):
            raise ValueError("停顿/目标RMS参数仅在 --processing legacy 下生效")
        engine = build_engine(args)
    except EngineError as e:
        print(f"引擎不可用：{e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"输出参数错误：{e}", file=sys.stderr)
        return 4
    try:
        direction = json.loads(Path(args.direction).read_text(encoding="utf-8")) if args.direction else None
        audio, sr, report = synthesize_text(
            engine, text, normalize=not args.no_normalize, max_chars=args.max_chars,
            target_db=args.target_db, language=getattr(engine, "language", args.language),
            processing=args.processing, max_output_seconds=args.max_seconds, direction=direction,
            lexicon=load_lexicon(args.lexicon) if args.lexicon else None,
            pauses={"para": args.pause_para, "sentence": args.pause_sentence,
                    "clause": args.pause_clause, "none": 0} if args.processing == "legacy" else {},
            segment_timeout=args.timeout,
            checkpoint_dir=getattr(args, "checkpoint_dir", None),
        )
    except EngineError as e:
        print(f"合成失败：{e}", file=sys.stderr)
        return 3
    except ValueError as e:
        print(f"输入文本无效：{e}", file=sys.stderr)
        return 1
    report["created"] = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    report["skill_version"] = VERSION
    report["model"] = model_provenance(getattr(args, "model", None))
    report["config"] = model_provenance(getattr(engine, "config_path", args.config))
    report["lexicon"] = model_provenance(args.lexicon) if args.lexicon else None
    report["runtime"] = {"python": sys.version.split()[0], "platform": platform.platform()}
    report["engine_params"] = ({"length_scale": args.length_scale, "noise_scale": args.noise_scale,
                               "noise_w": args.noise_w, "speaker": args.speaker, "cuda": args.cuda,
                               "timeout": args.timeout}
                              if args.engine == "piper" else {"length_scale": args.length_scale, "speed_wpm": engine.speed, "timeout": args.timeout}
                              if args.engine == "espeak" else {"cuda": args.cuda, "timeout": args.timeout})
    transactional_write_bundle(
        out, audio, sr, derived_formats=args.format or [],
        report=report, report_path=args.report,
    )
    print(json.dumps({k: report[k] for k in
                      ("output", "derived", "duration_seconds", "segment_count", "rtf",
                       "segment_rms_std_db", "warnings")},
                     ensure_ascii=False, indent=2))
    return 0


def _failure_report(args, code, reason):
    # Never overwrite audio or protected inputs while reporting an invalid request.
    if not getattr(args, "report", None):
        return
    rp = Path(args.report)
    protected = [args.out, args.input, args.model, args.config, args.lexicon,
                 args.reference_audio, args.authorization, args.direction]
    if args.engine == "piper" and args.model and not args.config:
        protected.append(str(args.model) + ".json")
    if rp.suffix.lower() != ".json" or any(rp.resolve() == Path(p).resolve() for p in protected if p):
        return
    try:
        _atomic_write_json(rp, {
            "ok": False, "exit_code": code, "reason": reason,
            "created": _dt.datetime.now().astimezone().isoformat(),
            "skill_version": VERSION,
        })
    except OSError:
        pass  # Original error is already visible; do not mask it with report I/O.


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description="TTS 合成流水线")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_engine_args(sp):
        sp.add_argument("--engine", default="piper", help="piper / espeak / xtts")
        sp.add_argument("--language", default="auto", help="BCP-47；Piper 默认取模型，其他引擎显式指定")
        sp.add_argument("--reference-audio", help="授权克隆本地参考音频")
        sp.add_argument("--authorization", help="本地授权记录 JSON")
        sp.add_argument("--purpose", help="需与授权用途完全一致")
        sp.add_argument("--require-signed-auth", action="store_true", help="强制要求授权记录附带有效 HMAC-SHA256 签名")
        sp.add_argument("--model", help="引擎模型路径（piper: .onnx）")
        sp.add_argument("--config", help="模型配置（默认同名 .onnx.json）")
        sp.add_argument("--speaker", type=int, default=None, help="多说话人模型的 speaker id")
        sp.add_argument("--length-scale", type=float, default=1.0, help="语速倍率，>1 变慢")
        sp.add_argument("--noise-scale", type=float, default=0.667, help="发音随机度")
        sp.add_argument("--noise-w", type=float, default=0.8, help="时长随机度")
        sp.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="单段与模型加载硬超时秒数（默认 60）")
        sp.add_argument("--cuda", action="store_true", help="使用 GPU（需 onnxruntime-gpu）")

    c = sub.add_parser("check", help="检查环境与引擎")
    add_engine_args(c)
    c.set_defaults(func=cmd_check)

    s = sub.add_parser("synth", help="合成音频")
    add_engine_args(s)
    src = s.add_mutually_exclusive_group()
    src.add_argument("--text", help="直接输入文本")
    src.add_argument("--input", help="UTF-8 文本文件")
    s.add_argument("--out", required=True, help="输出 WAV 母版路径")
    s.add_argument("--format", nargs="*", choices=["flac", "ogg"], help="额外导出的派生格式")
    s.add_argument("--report", help="JSON 报告路径")
    s.add_argument("--no-normalize", action="store_true", help="关闭中文文本规范化")
    s.add_argument("--lexicon", help="发音词典 TSV（原词<TAB>朗读形式）")
    s.add_argument("--max-chars", type=int, default=90, help="单段最大字符数")
    s.add_argument("--direction", help="语义声音策略 JSON，按规范化后的段落文本匹配 beats")
    s.add_argument("--processing", choices=["minimal", "legacy"], default="minimal", help="minimal 保留原动态；legacy 逐段对齐并加静音")
    s.add_argument("--max-seconds", type=float, default=1800, help="输出时长上限，最大1800秒")
    s.add_argument("--checkpoint-dir", help="分段断点保存与失败恢复目录（校验文本/模型/参数哈希一致后恢复）")
    s.add_argument("--isolate-model", action="store_true", help="在独立隔离子进程中加载模型并执行合成")
    s.add_argument("--target-db", type=float, default=-20.0, help="段内目标 RMS（dBFS）")
    s.add_argument("--pause-para", type=int, default=700, help="段落间停顿 ms")
    s.add_argument("--pause-sentence", type=int, default=400, help="句间停顿 ms")
    s.add_argument("--pause-clause", type=int, default=200, help="子句间停顿 ms")
    s.set_defaults(func=cmd_synth)

    args = p.parse_args(raw_argv)
    args._raw_argv = raw_argv
    try:
        code = args.func(args)
        if code != 0 and args.cmd == "synth":
            _failure_report(args, code, "见 CLI stderr：本次未成功，不可将旧产物作为本次交付")
        return code
    except EngineError as e:
        print(f"引擎错误：{e}", file=sys.stderr)
        code = 2 if args.cmd == "check" else 3
        if args.cmd == "synth":
            _failure_report(args, code, str(e))
        return code
    except (OSError, ValueError, RuntimeError, TypeError) as e:
        print(f"输入/配置/写入错误：{e}", file=sys.stderr)
        if args.cmd == "synth":
            _failure_report(args, 4, f"{type(e).__name__}: {e}")
        return 4


if __name__ == "__main__":
    sys.exit(main())

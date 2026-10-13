#!/usr/bin/env python3
"""Deterministic TTS execution. See SKILL.md for decisions and README for CLI."""
import argparse
import datetime as _dt
import hashlib
import json
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from textnorm import load_lexicon, normalize_zh, normalize_text, detect_language, text_warnings  # noqa: E402

VERSION = "0.3.0"


from engines import Engine, EngineError, PiperEngine, EspeakEngine, XTTSEngine


def build_engine(args):
    if args.engine == "piper":
        if args.reference_audio or args.authorization:
            raise EngineError("Piper 不支持克隆")
        return PiperEngine(model=args.model, config=args.config, speaker=args.speaker,
                           length_scale=args.length_scale, noise_scale=args.noise_scale,
                           noise_w_scale=args.noise_w, use_cuda=args.cuda, language=args.language)
    if args.engine != "piper" and (args.noise_scale != .667 or args.noise_w != .8):
        raise EngineError("该适配器不支持 Piper 噪声参数")
    if args.engine == "espeak":
        if args.reference_audio or args.authorization or args.speaker is not None or args.cuda or args.model or args.config:
            raise EngineError("eSpeak 不接受克隆、speaker id、CUDA 或模型/config参数")
        return EspeakEngine(language=args.language, length_scale=args.length_scale)
    if args.engine == "xtts":
        if args.speaker is not None or args.length_scale != 1.0:
            raise EngineError("XTTS 适配器不暴露 speaker id / 语速参数")
        return XTTSEngine(args.model, args.config, args.language, args.reference_audio,
                          args.authorization, args.purpose, args.cuda)
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
    # Same frame activity definition as QA; sr supplied via parameter below.
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


def synthesize_text(engine, text, *, normalize=True, max_chars=90, target_db=-20.0,
                    pauses=None, peak_db=-1.0, lexicon=None, language=None, processing="legacy",
                    max_output_seconds=1800, direction=None):
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

    beats = {}
    if direction is not None:
        if not isinstance(direction, dict) or any(not direction.get(k) for k in ("intent", "audience", "character")):
            raise ValueError("direction 需 intent/audience/character")
        for beat in direction.get("beats", []):
            if set(beat) - {"index", "text", "intent", "rationale", "length_scale"}:
                raise ValueError("不支持的声音控制；不能伪装情绪/重音能力")
            index = beat.get("index")
            if not isinstance(index, int) or not 0 <= index < len(segs) or index in beats:
                raise ValueError("beat 索引无效或重复")
            if beat.get("text") != segs[index].text or not beat.get("intent") or not beat.get("rationale"):
                raise ValueError("beat 文本不匹配或缺少语义理由")
            if "length_scale" in beat and (not getattr(engine, "capabilities", {}).get("rate") or
                    not np.isfinite(beat["length_scale"]) or not .5 <= beat["length_scale"] <= 2):
                raise ValueError("引擎不支持该语速或倍率超界")
            beats[index] = beat
    parts, seg_reports, warnings = [], [], text_warnings(original, language)
    output_samples = 0
    t_total = 0.0
    for i, seg in enumerate(segs):
        beat = beats.get(i, {})
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
            for attempt in (1, 2):  # 单段失败重试一次
                try:
                    raw = np.asarray(engine.synthesize(seg.text))
                    if raw.ndim != 1 or raw.size == 0 or not np.isfinite(raw).all():
                        raise EngineError("引擎输出为空、非有限或非单声道")
                    if np.max(np.abs(raw)) > 1.0001:
                        raise EngineError("引擎输出超出 [-1,1]；拒绝掩盖异常")
                    break
                except EngineError as e:  # Only adapter-declared failures are retried
                    last_err = e
            else:
                raise EngineError(f"第 {i + 1} 段合成失败（{seg.text[:30]}…）：{last_err}") from last_err
        finally:
            if saved_rate is not None:
                if engine.name == "piper":
                    engine.syn.length_scale = saved_rate
                else:
                    engine.speed = saved_rate
        dt = time.perf_counter() - t0
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
        seg_reports.append({"index": i, "text": seg.text, "seconds": round(dur, 3),
                            "synth_seconds": round(dt, 3), "pause_after": seg.pause_after,
                            "rms_dbfs": None if rms_db is None else round(rms_db, 2),
                            "attempts": attempt, "direction": beat, "start_sample": output_samples, "samples": len(x)})
        parts.append(x)
        output_samples += len(x)
        gap = pauses.get(seg.pause_after, 0)
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

    for srep in seg_reports:
        final = audio[srep["start_sample"]:srep["start_sample"] + srep["samples"]]
        rms = voiced_rms_db(final, sr)
        srep["rms_dbfs"] = None if rms is None else round(rms, 2)
    from qa import active_seconds
    total = len(audio) / sr
    rms_list = [s["rms_dbfs"] for s in seg_reports if s["rms_dbfs"] is not None]
    rms_std = round(float(np.std(rms_list)), 3) if len(rms_list) > 1 else 0.0
    if rms_std > 1.0:
        warnings.append(f"段间响度标准差 {rms_std:.2f} dB 偏大（>1 dB），跨段一致性可能不足")
    report = {
        "ok": True,  # synthesis success only; not QA acceptance
        "direction": direction,
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
        "validation": {"human_listening": "not_performed", "asr": "not_performed"},
        "synth_seconds": round(t_total, 3),
        "rtf": round(t_total / total, 3) if total > 0 else None,
        "peak_dbfs": round(_db(float(np.max(np.abs(audio)))), 2),
        # 段间一致性：各段对齐后的有声 RMS 与其标准差（客观代理，音色一致性仍需人工听评）
        "segment_rms_std_db": rms_std,
        "params": {"normalize": normalize, "max_chars": max_chars, "target_db": target_db if processing == "legacy" else None,
                   "pause_ms": pauses, "peak_limit_dbfs": peak_db,
                   "lexicon_entries": len(lexicon) if lexicon else 0, "processing": processing,
                   "max_output_seconds": max_output_seconds},
        "warnings": warnings,
    }
    return audio.astype(np.float32), sr, report


SUPPORTED_FORMATS = ("wav", "flac", "ogg")


def write_audio(path, audio, sr, fmt=None):
    import soundfile as sf
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = (fmt or path.suffix.lstrip(".")).lower()
    subtype = {"wav": "PCM_24", "flac": "PCM_16", "ogg": "VORBIS"}.get(ext)
    if subtype is None:
        raise ValueError(f"不支持的输出格式：.{ext}（支持 {'/'.join(SUPPORTED_FORMATS)}；"
                         f"MP3 需要额外编码器，本技能不支持）")
    import os
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix="." + ext)
    os.close(fd)
    try:
        sf.write(tmp, audio, sr, subtype=subtype, format=ext.upper())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return path


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


def cmd_synth(args):
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
            processing=args.processing, max_output_seconds=args.max_seconds, direction=direction, lexicon=load_lexicon(args.lexicon) if args.lexicon else None,
            pauses={"para": args.pause_para, "sentence": args.pause_sentence,
                    "clause": args.pause_clause, "none": 0} if args.processing == "legacy" else {},
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
                               "noise_w": args.noise_w, "speaker": args.speaker, "cuda": args.cuda}
                              if args.engine == "piper" else {"length_scale": args.length_scale, "speed_wpm": engine.speed}
                              if args.engine == "espeak" else {"cuda": args.cuda})
    write_audio(out, audio, sr, "wav")  # 母版始终为 WAV
    derived = []
    for fmt in args.format or []:
        derived.append(str(write_audio(out.with_suffix("." + fmt), audio, sr, fmt)))
    report["output"] = str(out)
    report["output_sha256"] = sha256_file(out)
    report["derived"] = derived
    if args.report:
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("output", "derived", "duration_seconds", "segment_count", "rtf",
                       "segment_rms_std_db", "warnings")},
                     ensure_ascii=False, indent=2))
    return 0


def _failure_report(args, code, reason):
    # Never overwrite audio or protected inputs while reporting an invalid request.
    if not args.report:
        return
    rp = Path(args.report)
    protected = [args.out, args.input, args.model, args.config, args.lexicon,
                 args.reference_audio, args.authorization, args.direction]
    if args.engine == "piper" and args.model and not args.config:
        protected.append(str(args.model) + ".json")
    if rp.suffix.lower() != ".json" or any(rp.resolve() == Path(p).resolve() for p in protected if p):
        return
    try:
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps({"ok": False, "exit_code": code, "reason": reason,
                                  "created": _dt.datetime.now().astimezone().isoformat(),
                                  "skill_version": VERSION}, ensure_ascii=False, indent=2))
    except OSError:
        pass  # Original error is already visible; do not mask it with report I/O.


def main(argv=None):
    p = argparse.ArgumentParser(description="TTS 合成流水线")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_engine_args(sp):
        sp.add_argument("--engine", default="piper", help="piper / espeak / xtts")
        sp.add_argument("--language", default="auto", help="BCP-47；Piper 默认取模型，其他引擎显式指定")
        sp.add_argument("--reference-audio", help="授权克隆本地参考音频")
        sp.add_argument("--authorization", help="本地授权记录 JSON")
        sp.add_argument("--purpose", help="需与授权用途完全一致")
        sp.add_argument("--model", help="引擎模型路径（piper: .onnx）")
        sp.add_argument("--config", help="模型配置（默认同名 .onnx.json）")
        sp.add_argument("--speaker", type=int, default=None, help="多说话人模型的 speaker id")
        sp.add_argument("--length-scale", type=float, default=1.0, help="语速倍率，>1 变慢")
        sp.add_argument("--noise-scale", type=float, default=0.667, help="发音随机度")
        sp.add_argument("--noise-w", type=float, default=0.8, help="时长随机度")
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
    s.add_argument("--target-db", type=float, default=-20.0, help="段内目标 RMS（dBFS）")
    s.add_argument("--pause-para", type=int, default=700, help="段落间停顿 ms")
    s.add_argument("--pause-sentence", type=int, default=400, help="句间停顿 ms")
    s.add_argument("--pause-clause", type=int, default=200, help="子句间停顿 ms")
    s.set_defaults(func=cmd_synth)

    args = p.parse_args(argv)
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

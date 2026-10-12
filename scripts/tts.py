#!/usr/bin/env python3
"""TTS 合成流水线（引擎可插拔）。

流程：规范化 -> 分句/分段 -> 引擎逐段合成 -> 段内响度对齐 -> 边界淡变 -> 插入停顿 -> 峰值限制 -> 写母版 + 报告

子命令：
  check   检查环境与引擎可用性（JSON 输出，不合成）
  synth   合成文本为音频，输出 WAV 母版（可选 FLAC/OGG 派生件）与 JSON 报告

引擎接口（新增引擎只需实现 Engine）：
  available() -> (bool, str)        可用性与原因
  synthesize(text) -> np.ndarray    float32 单声道 [-1, 1]
  sample_rate                       输出采样率

依赖：numpy, soundfile；piper 引擎另需 piper-tts 包与 .onnx 模型（见 references/engines.md）。
"""
import argparse
import json
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from textnorm import load_lexicon, normalize_zh  # noqa: E402


class EngineError(RuntimeError):
    """引擎不可用或合成失败。消息需可直接给用户诊断。"""


# ---------------------------------------------------------------- 引擎层

class Engine:
    name = "base"
    sample_rate = 0

    def available(self):
        return True, "ok"

    def synthesize(self, text):
        raise NotImplementedError


class PiperEngine(Engine):
    """Piper（ONNX，CPU 可跑）。注意：piper-tts 包为 GPL-3.0-or-later。"""
    name = "piper"

    def __init__(self, model, config=None, speaker=None, length_scale=1.0,
                 noise_scale=0.667, noise_w_scale=0.8, use_cuda=False, download_dir=None):
        try:
            from piper import PiperVoice
            from piper.config import SynthesisConfig
        except ImportError as e:
            raise EngineError(
                "未安装 piper-tts。安装：pip install piper-tts（GPL-3.0，见 references/engines.md）"
            ) from e
        if not model:
            raise EngineError("piper 引擎需要 --model 指定 .onnx 模型路径（见 references/engines.md）")
        model = Path(model)
        if not model.is_file():
            raise EngineError(f"找不到 Piper 模型文件：{model}")
        cfg = Path(config) if config else Path(str(model) + ".json")
        if not cfg.is_file():
            raise EngineError(f"找不到模型配置文件：{cfg}（需与 .onnx 同名的 .onnx.json）")
        # 中文模型的 g2pW 资源默认会下载到 CWD；显式指定为模型同目录，避免污染工作区
        self.voice = PiperVoice.load(str(model), str(cfg), use_cuda=use_cuda,
                                     download_dir=str(download_dir or model.parent))
        self.sample_rate = self.voice.config.sample_rate
        self.syn = SynthesisConfig(
            speaker_id=speaker,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
            normalize_audio=False,  # 关闭逐句峰值归一化，响度由流水线统一控制
        )
        self.model_path = str(model)

    def available(self):
        return True, f"piper model={self.model_path} sr={self.sample_rate}"

    def synthesize(self, text):
        chunks = list(self.voice.synthesize(text, syn_config=self.syn))
        if not chunks:
            raise EngineError("Piper 返回空结果")
        return np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)


def build_engine(args):
    if args.engine == "piper":
        return PiperEngine(
            model=args.model, config=args.config, speaker=args.speaker,
            length_scale=args.length_scale, noise_scale=args.noise_scale,
            noise_w_scale=args.noise_w, use_cuda=args.cuda,
        )
    raise EngineError(f"未知引擎：{args.engine}（当前支持：piper）")


# ---------------------------------------------------------------- 分段

@dataclass
class Segment:
    text: str
    pause_after: str  # "para" | "sentence" | "clause" | "none"


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
            pieces.append(p[:max_chars])
            p = p[max_chars:]
        if len(buf) + len(p) <= max_chars:
            buf += p
        else:
            pieces.append(buf)
            buf = p
    if buf:
        pieces.append(buf)
    return pieces


def segment(text, max_chars=90):
    """按段落、句子、子句切分；每段记录其后应插入的停顿类型。"""
    segs = []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    for para in paragraphs:
        sents = [s.strip() for s in re.split(r"(?<=[。！？!?…])\s*|(?<=[.?!])\s+", para) if s.strip()]
        for si, sent in enumerate(sents):
            pieces = _split_long(sent, max_chars)
            for k, piece in enumerate(pieces):
                last_in_para = si == len(sents) - 1 and k == len(pieces) - 1
                if last_in_para:
                    kind = "para"
                elif k == len(pieces) - 1:
                    kind = "sentence"
                else:
                    kind = "clause"
                segs.append(Segment(piece, kind))
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


def level_to(x, target_db):
    """段内响度对齐到 target_db（基于有声帧 RMS），增益限制在 ±12 dB。"""
    v = x[_voiced(x)]
    if v.size == 0:
        return x
    rms = np.sqrt(np.mean(v.astype(np.float64) ** 2))
    gain = np.clip(10 ** (target_db / 20) / max(rms, 1e-9), 10 ** (-12 / 20), 10 ** (12 / 20))
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
                    pauses=None, peak_db=-1.0, lexicon=None):
    """完整流水线，返回 (audio float32, sr, report dict)。"""
    pauses = pauses or {"para": 700, "sentence": 400, "clause": 200, "none": 0}
    sr = engine.sample_rate
    original = text
    text = normalize_zh(text, lexicon) if normalize else text
    segs = segment(text, max_chars=max_chars)
    if not segs:
        raise ValueError("输入文本为空（去除空白后无可朗读内容）")

    parts, seg_reports, warnings = [], [], []
    t_total = 0.0
    for i, seg in enumerate(segs):
        t0 = time.perf_counter()
        last_err = None
        for attempt in (1, 2):  # 单段失败重试一次
            try:
                raw = engine.synthesize(seg.text)
                break
            except Exception as e:  # noqa: BLE001 - 引擎异常统一包装
                last_err = e
        else:
            raise EngineError(f"第 {i + 1} 段合成失败（{seg.text[:30]}…）：{last_err}") from last_err
        dt = time.perf_counter() - t0
        t_total += dt
        x = trim_silence(raw.astype(np.float32), sr)
        if x.size < sr * 0.05:
            warnings.append(f"第 {i + 1} 段输出过短，可能发音缺失：{seg.text[:30]}")
        x = level_to(x, target_db)
        x = fade(x, sr)
        dur = len(x) / sr
        seg_reports.append({"index": i, "text": seg.text, "seconds": round(dur, 3),
                            "synth_seconds": round(dt, 3), "pause_after": seg.pause_after})
        parts.append(x)
        gap = pauses.get(seg.pause_after, 0)
        if gap:
            parts.append(np.zeros(int(sr * gap / 1000), dtype=np.float32))

    audio = np.concatenate(parts)
    peak = float(np.max(np.abs(audio)))
    limit = 10 ** (peak_db / 20)
    if peak > limit:
        audio = audio * (limit / peak)
        warnings.append(f"峰值 {_db(peak):.1f} dBFS 超过上限，已整体衰减到 {peak_db} dBFS")

    total = len(audio) / sr
    report = {
        "engine": engine.name,
        "engine_info": engine.available()[1],
        "sample_rate": sr,
        "normalized": normalize,
        "original_text": original,
        "normalized_text": text,
        "segments": seg_reports,
        "segment_count": len(segs),
        "duration_seconds": round(total, 3),
        "synth_seconds": round(t_total, 3),
        "rtf": round(t_total / total, 3) if total > 0 else None,
        "peak_dbfs": round(_db(float(np.max(np.abs(audio)))), 2),
        "warnings": warnings,
    }
    return audio.astype(np.float32), sr, report


def write_audio(path, audio, sr, fmt=None):
    import soundfile as sf
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subtype = {"wav": "PCM_24", "flac": "PCM_16", "ogg": "VORBIS"}.get(
        (fmt or path.suffix.lstrip(".")).lower())
    if subtype is None:
        raise ValueError(f"不支持的输出格式：{path.suffix}（支持 wav/flac/ogg）")
    sf.write(str(path), audio, sr, subtype=subtype)
    return path


# ---------------------------------------------------------------- CLI

def cmd_check(args):
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "engines": {}}
    try:
        import soundfile as sf
        info["soundfile"] = sf.__libsndfile_version__
    except ImportError:
        info["soundfile"] = None
        info["hint"] = "pip install soundfile numpy"
    try:
        e = build_engine(args)
        info["engines"][args.engine] = {"available": True, "info": e.available()[1]}
    except EngineError as e:
        info["engines"][args.engine] = {"available": False, "reason": str(e)}
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0 if all(v["available"] for v in info["engines"].values()) else 2


def cmd_synth(args):
    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    else:
        text = args.text or ""
    if not text.strip():
        print("错误：没有输入文本（--text 或 --input）", file=sys.stderr)
        return 1
    try:
        engine = build_engine(args)
    except EngineError as e:
        print(f"引擎不可用：{e}", file=sys.stderr)
        return 2
    try:
        audio, sr, report = synthesize_text(
            engine, text, normalize=not args.no_normalize, max_chars=args.max_chars,
            target_db=args.target_db, lexicon=load_lexicon(args.lexicon) if args.lexicon else None,
            pauses={"para": args.pause_para, "sentence": args.pause_sentence,
                    "clause": args.pause_clause, "none": 0},
        )
    except EngineError as e:
        print(f"合成失败：{e}", file=sys.stderr)
        return 3
    out = Path(args.out)
    write_audio(out, audio, sr, "wav")  # 母版始终为 WAV
    derived = []
    for fmt in args.format or []:
        derived.append(str(write_audio(out.with_suffix("." + fmt), audio, sr, fmt)))
    report["output"] = str(out)
    report["derived"] = derived
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("output", "derived", "duration_seconds", "segment_count", "rtf", "warnings")},
                     ensure_ascii=False, indent=2))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="TTS 合成流水线")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_engine_args(sp):
        sp.add_argument("--engine", default="piper", help="引擎名（当前：piper）")
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
    s.add_argument("--target-db", type=float, default=-20.0, help="段内目标 RMS（dBFS）")
    s.add_argument("--pause-para", type=int, default=700, help="段落间停顿 ms")
    s.add_argument("--pause-sentence", type=int, default=400, help="句间停顿 ms")
    s.add_argument("--pause-clause", type=int, default=200, help="子句间停顿 ms")
    s.set_defaults(func=cmd_synth)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

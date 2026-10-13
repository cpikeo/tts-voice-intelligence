"""真实 Piper 引擎集成测试。需设置环境变量 TTS_PIPER_MODEL=/path/to/model.onnx，否则跳过。

运行：TTS_PIPER_MODEL=... python -m pytest -q tests/test_piper_integration.py
未设置环境变量时全部跳过（不伪造引擎、不伪造指标）。
"""
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import qa  # noqa: E402
import tts  # noqa: E402

MODEL = os.environ.get("TTS_PIPER_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="未设置 TTS_PIPER_MODEL，跳过真实引擎测试")

TEXT = "会议定在2026年10月12日上午10:30召开。我们用Python 3.13处理3.5%的数据。"

# 长文本：约 1200 字，用于验证分段数量、内存与耗时（不验证音质）
LONG_TEXT = ("清晨的城市还没有完全醒来，街道上只有零星的车灯。"
             "早餐店的蒸汽慢慢升起，老人牵着狗走过公园，孩子们背着书包跑过路口。"
             "这是一座忙碌的城市，但在这一刻，它显得格外安静。") * 12


@pytest.fixture(scope="module")
def engine():
    return tts.PiperEngine(model=MODEL)


def test_end_to_end_wav_passes_qa(engine, tmp_path):
    audio, sr, rep = tts.synthesize_text(engine, TEXT)
    assert rep["segment_count"] == 2
    assert rep["normalized_text"].startswith("会议定在二零二六年十月十二日上午十点三十分召开")
    out = tts.write_audio(tmp_path / "e2e.wav", audio, sr)
    x, sr2, info = qa.load_mono(out)
    assert sr2 == engine.sample_rate
    checks = qa.run_signal_checks(x, sr2, info)
    assert not [c for c in checks if c["status"] == "fail"], checks


def test_loudness_target_held_for_multi_segment(engine, tmp_path):
    """多段合成后整体有声 RMS 应接近目标 -20 dBFS（误差 ±3 dB），峰值不超过 -1 dBFS。"""
    audio, sr, rep = tts.synthesize_text(engine, "第一句话比较短。这是第二句话，它稍微长一些，用来比较响度。")
    out = tts.write_audio(tmp_path / "loud.wav", audio, sr)
    x, sr2, info = qa.load_mono(out)
    checks = {c["name"]: c for c in qa.run_signal_checks(x, sr2, info)}
    assert abs(checks["loudness_rms"]["value"] - (-20)) <= 3
    assert rep["peak_dbfs"] <= -0.9


def test_length_scale_changes_duration(engine):
    fast = tts.PiperEngine(model=MODEL, length_scale=0.8)
    slow = tts.PiperEngine(model=MODEL, length_scale=1.3)
    _, _, rf = tts.synthesize_text(fast, "这是一个用于比较语速的测试句子。")
    _, _, rs = tts.synthesize_text(slow, "这是一个用于比较语速的测试句子。")
    assert rs["duration_seconds"] > rf["duration_seconds"]


def test_report_carries_model_provenance(engine):
    """跨任务复现要求：报告必须能回答“用了哪个模型文件”。"""
    _, _, rep = tts.synthesize_text(engine, "很短的一句。")
    info = tts.model_provenance(MODEL)
    assert info["exists"] and len(info["sha256"]) == 64 and info["bytes"] > 0
    assert rep["params"]["target_db"] == -20.0
    assert rep["speech_seconds"] <= rep["duration_seconds"]


def test_long_text_stability_and_consistency(engine, tmp_path):
    """长文本：必须能跑完，段间响度标准差 ≤1 dB，信号层无 fail，并记录资源消耗。"""
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    audio, sr, rep = tts.synthesize_text(engine, LONG_TEXT)
    wall = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    assert rep["segment_count"] >= 12
    assert rep["segment_rms_std_db"] <= 1.0, rep["segment_rms_std_db"]
    assert rep["peak_dbfs"] <= -0.9
    assert np.all(np.isfinite(audio))
    out = tts.write_audio(tmp_path / "long.wav", audio, sr)
    x, sr2, info = qa.load_mono(out)
    checks = qa.run_signal_checks(x, sr2, info)
    assert not [c for c in checks if c["status"] == "fail"], checks
    print(f"\n[long text] chars={len(LONG_TEXT)} segments={rep['segment_count']} "
          f"dur={rep['duration_seconds']}s synth={rep['synth_seconds']}s rtf={rep['rtf']} "
          f"wall={wall:.1f}s rms_std={rep['segment_rms_std_db']}dB "
          f"peakRSS={rss_mb:.0f}MB (+{rss_mb - rss0 / 1024:.0f}MB) "
          f"audio_MB={audio.nbytes / 1e6:.1f}", file=sys.stderr)


def test_synth_cli_end_to_end(tmp_path):
    """端到端 CLI：退出码 0、报告落盘、派生格式可解码。"""
    script = tmp_path / "in.txt"
    script.write_text(TEXT, encoding="utf-8")
    report = tmp_path / "report.json"
    assert tts.main(["synth", "--model", MODEL, "--input", str(script),
                     "--out", str(tmp_path / "o.wav"), "--format", "flac",
                     "--report", str(report),
                     "--lexicon", str(ROOT / "assets" / "lexicon.example.tsv")]) == 0
    data = report.read_text(encoding="utf-8")
    assert "sha256" in data and "created" in data
    import json as _json
    rep = _json.loads(data)
    assert rep["params"]["lexicon_entries"] == 5
    assert rep["normalized_text"].startswith("会议定在二零二六年")
    x, sr, info = qa.load_mono(tmp_path / "o.flac")
    assert len(x) > 0


def test_synth_cli_rejects_mp3_before_synthesizing(tmp_path):
    assert tts.main(["synth", "--model", MODEL, "--text", TEXT,
                     "--out", str(tmp_path / "o.mp3")]) == 4


def test_synth_cli_empty_after_normalize_returns_1(tmp_path):
    script = tmp_path / "in.txt"
    script.write_text("***", encoding="utf-8")
    assert tts.main(["synth", "--model", MODEL, "--input", str(script),
                     "--out", str(tmp_path / "o.wav")]) == 1

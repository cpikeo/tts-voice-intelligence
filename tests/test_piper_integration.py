"""真实 Piper 引擎集成测试。需设置环境变量 TTS_PIPER_MODEL=/path/to/model.onnx，否则跳过。

运行：TTS_PIPER_MODEL=... python -m pytest -q tests/test_piper_integration.py
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import qa  # noqa: E402
import tts  # noqa: E402

MODEL = os.environ.get("TTS_PIPER_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="未设置 TTS_PIPER_MODEL，跳过真实引擎测试")

TEXT = "会议定在2026年10月12日上午10:30召开。我们用Python 3.13处理3.5%的数据。"


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
    checks = qa.run_signal_checks(x, sr2, info, TEXT)
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

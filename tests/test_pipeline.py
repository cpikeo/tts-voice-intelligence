import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import tts  # noqa: E402
import qa  # noqa: E402

SR = 22050


class ToneEngine(tts.Engine):
    """确定性假引擎：每个字 0.18s 正弦音，用于验证流水线逻辑（不代表音质）。"""
    name = "tone"
    sample_rate = SR

    def __init__(self, fail_on=None, empty_on=None):
        self.fail_on = fail_on
        self.empty_on = empty_on
        self.calls = []

    def synthesize(self, text):
        self.calls.append(text)
        if self.fail_on and self.fail_on in text:
            raise tts.EngineError("模拟故障")
        if self.empty_on and self.empty_on in text:
            return np.zeros(10, dtype=np.float32)
        n = int(SR * 0.18 * max(1, len(text)))
        t = np.arange(n) / SR
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_segment_pause_kinds_and_split():
    segs = tts.segment("第一句，带逗号。第二句！\n第三段很短", max_chars=90)
    assert [s.text for s in segs] == ["第一句，带逗号。", "第二句！", "第三段很短"]
    assert [s.pause_after for s in segs] == ["sentence", "para", "none"]


def test_long_sentence_clause_pauses():
    segs = tts.segment("甲甲甲甲，乙乙乙乙，丙丙丙丙。", max_chars=6)
    assert [s.text for s in segs] == ["甲甲甲甲，", "乙乙乙乙，", "丙丙丙丙。"]
    assert [s.pause_after for s in segs] == ["clause", "clause", "none"]


def test_long_sentence_split_respects_max_chars():
    long = "这是一个很长的句子" * 3 + "，" + "后半句" * 20
    segs = tts.segment(long, max_chars=40)
    assert all(len(s.text) <= 40 for s in segs)
    assert len(segs) > 2


def test_pipeline_outputs_consistent_level_and_pauses():
    eng = ToneEngine()
    audio, sr, rep = tts.synthesize_text(eng, "今天天气很好。我们去公园散步吧。", target_db=-20)
    assert sr == SR
    assert rep["segment_count"] == 2
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) <= 10 ** (-1 / 20) + 1e-6  # 峰值限制
    # 段间停顿：400ms 句间停顿应存在于两段之间
    assert rep["duration_seconds"] > sum(s["seconds"] for s in rep["segments"]) + 0.35
    assert rep["normalized_text"] == "今天天气很好。我们去公园散步吧。"


def test_normalization_applied_before_engine():
    eng = ToneEngine()
    _, _, rep = tts.synthesize_text(eng, "2026年10月12日")
    assert eng.calls == ["二零二六年十月十二日"]
    assert rep["original_text"] == "2026年10月12日"


def test_retry_then_success_and_hard_failure():
    class Flaky(ToneEngine):
        def __init__(self):
            super().__init__()
            self.n = 0

        def synthesize(self, text):
            self.n += 1
            if self.n == 1:
                raise tts.EngineError("瞬时错误")
            return super().synthesize(text)

    audio, _, _ = tts.synthesize_text(Flaky(), "测试一下。")
    assert len(audio) > 0
    with pytest.raises(tts.EngineError, match="第 1 段合成失败"):
        tts.synthesize_text(ToneEngine(fail_on="坏"), "这是坏句子。")


def test_empty_output_warns():
    _, _, rep = tts.synthesize_text(ToneEngine(empty_on="空"), "空的句子。")
    assert any("过短" in w for w in rep["warnings"])


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        tts.synthesize_text(ToneEngine(), "   \n  ")


def test_write_formats(tmp_path):
    audio = np.zeros(SR, dtype=np.float32)
    for ext in ("wav", "flac", "ogg"):
        p = tts.write_audio(tmp_path / f"a.{ext}", audio, SR)
        assert p.stat().st_size > 0
    with pytest.raises(ValueError):
        tts.write_audio(tmp_path / "a.mp3", audio, SR)


def test_missing_piper_model_gives_diagnosable_error(tmp_path):
    with pytest.raises(tts.EngineError, match="找不到 Piper 模型"):
        tts.PiperEngine(model=tmp_path / "none.onnx")


# ---------------- QA

def _write(tmp_path, x, name="a.wav"):
    import soundfile as sf
    p = tmp_path / name
    sf.write(str(p), x, SR, subtype="PCM_16")
    return p


def test_qa_passes_reasonable_audio(tmp_path):
    eng = ToneEngine()
    audio, sr, _ = tts.synthesize_text(eng, "这是一段用于验收的测试音频，时长足够。")
    p = _write(tmp_path, audio * 0.1)  # 把音量降到 -20 dBFS 附近
    assert qa.main([str(p), "--text", "这是一段用于验收的测试音频，时长足够。"]) == 0


def test_qa_detects_clipping(tmp_path):
    x = np.clip(np.sin(np.linspace(0, 200, SR * 2)) * 3, -1, 1).astype(np.float32)
    p = _write(tmp_path, x)
    rep = tmp_path / "r.json"
    assert qa.main([str(p), "--report", str(rep)]) == 1
    data = json.loads(rep.read_text(encoding="utf-8"))
    assert any(c["name"] == "clipping" and c["status"] == "fail" for c in data["checks"])


def test_qa_detects_silence_and_long_gap(tmp_path):
    x = np.zeros(SR * 6, dtype=np.float32)
    x[: SR] = 0.1 * np.sin(np.arange(SR) / 5)
    x[-SR:] = 0.1 * np.sin(np.arange(SR) / 5)  # 中间 4 秒静音
    p = _write(tmp_path, x)
    rep = tmp_path / "r.json"
    qa.main([str(p), "--report", str(rep)])
    data = json.loads(rep.read_text(encoding="utf-8"))
    gap = next(c for c in data["checks"] if c["name"] == "internal_silence")
    assert gap["status"] == "warn" and gap["value"] > 2.5


def test_qa_missing_file_exit_code(tmp_path):
    assert qa.main([str(tmp_path / "nope.wav")]) == 2


def test_cer():
    assert qa._cer("今天天气好", "今天天气好") == 0
    assert qa._cer("今天天气好", "今天天汽好") == pytest.approx(0.2)
    assert qa._cer("", "x") is None


def _clicks(x):
    info = type("I", (), {"channels": 1, "subtype": "PCM_16"})()
    checks = {c["name"]: c for c in qa.run_signal_checks(x, SR, info)}
    return checks["clicks_in_silence"]["value"]


def test_qa_detects_injected_click_in_silence():
    """回归测试：此前检测器因门限设计错误而永远输出 0（假阴性）。"""
    x = np.zeros(SR * 2, dtype=np.float32)
    x[SR] = 0.5
    assert _clicks(x) == 1


def test_qa_no_click_on_clean_tone():
    t = np.arange(SR * 2) / SR
    assert _clicks((0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)) == 0

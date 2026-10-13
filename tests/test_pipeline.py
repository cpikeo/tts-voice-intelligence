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


def test_punctuation_only_fragment_is_not_a_segment():
    """回归：此前“！”/“…”会单独成段被引擎合成，并额外插入句间停顿。"""
    segs = tts.segment("真的吗？！接下来。", max_chars=90)
    assert [s.text for s in segs] == ["真的吗？！", "接下来。"]
    segs = tts.segment("请稍等……然后再试一次。", max_chars=90)
    assert [s.text for s in segs] == ["请稍等……", "然后再试一次。"]


def test_punctuation_only_input_is_dropped_not_crashed():
    with pytest.raises(ValueError):
        tts.synthesize_text(ToneEngine(), "！！！")


def test_long_sentence_clause_pauses():
    segs = tts.segment("甲甲甲甲，乙乙乙乙，丙丙丙丙。", max_chars=6)
    assert [s.text for s in segs] == ["甲甲甲甲，", "乙乙乙乙，", "丙丙丙丙。"]
    assert [s.pause_after for s in segs] == ["clause", "clause", "none"]


def test_long_sentence_split_respects_max_chars():
    long = "这是一个很长的句子" * 3 + "，" + "后半句" * 20
    segs = tts.segment(long, max_chars=40)
    assert all(len(s.text) <= 40 for s in segs)
    assert len(segs) > 2


def test_peak_attenuation_does_not_relabel_pre_gain_segment_rms_warning():
    class ImpulseEngine(ToneEngine):
        def synthesize(self, _text):
            audio = np.zeros(SR, dtype=np.float32)
            audio[SR // 2] = 0.99
            return audio

    audio, _, report = tts.synthesize_text(
        ImpulseEngine(), "检查增益。", processing="legacy", peak_db=-12.0,
        boundary_repair_enabled=False)
    assert audio.size
    assert any("整体衰减" in warning for warning in report["warnings"])
    # This warning was based on per-segment RMS before master attenuation in the baseline;
    # final reported RMS may differ, but the later gain must not invent a segment warning.
    assert not any("偏离目标" in warning for warning in report["warnings"])
    assert report["segments"][0]["rms_dbfs"] < -30


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


def test_piper_batch_checks_full_integrity_before_and_after_loaded_session():
    class BoundPiper(ToneEngine):
        name = "piper"
        def __init__(self):
            super().__init__()
            self.integrity_checks = 0
        def verify_integrity(self):
            self.integrity_checks += 1
        def _synthesize_loaded(self, text):
            return super().synthesize(text)

    eng = BoundPiper()
    audio, _, rep = tts.synthesize_text(eng, "第一句。第二句。", processing="minimal")
    assert audio.size and rep["segment_count"] == 2
    assert eng.integrity_checks == 2  # once before, once after the complete batch


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
    """必须给出可诊断的 EngineError。

    两种环境都得通过：装了 piper 但模型不存在 -> “找不到 Piper 模型”；
    没装 piper -> “未安装 piper-tts”。只匹配前一种会让干净 CI 误报失败。
    """
    with pytest.raises(tts.EngineError, match="找不到 Piper 模型|未安装 piper-tts") as ei:
        tts.PiperEngine(model=tmp_path / "none.onnx")
    assert str(ei.value)


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


def test_model_provenance(tmp_path):
    p = tmp_path / "m.onnx"
    p.write_bytes(b"abc")
    info = tts.model_provenance(p)
    assert info["exists"] and info["bytes"] == 3 and len(info["sha256"]) == 64
    assert tts.model_provenance(tmp_path / "nope.onnx")["exists"] is False


def test_cli_rejects_unsupported_output_format(tmp_path):
    """回归：此前 .mp3 输出会抛出未捕获异常而不是给出可诊断的退出码。"""
    assert tts.main(["synth", "--model", str(tmp_path / "x.onnx"), "--text", "测试",
                     "--out", str(tmp_path / "a.mp3")]) == 4


def test_cli_engine_unavailable_returns_2(tmp_path):
    assert tts.main(["synth", "--model", str(tmp_path / "x.onnx"), "--text", "测试",
                     "--out", str(tmp_path / "a.wav")]) == 2


def test_cli_empty_text_returns_1(tmp_path):
    assert tts.main(["synth", "--model", str(tmp_path / "x.onnx"),
                     "--out", str(tmp_path / "a.wav")]) == 1


def test_check_reports_unavailable_instead_of_traceback(tmp_path, capsys):
    assert tts.main(["check", "--model", str(tmp_path / "x.onnx")]) == 2
    data = json.loads(capsys.readouterr().out)
    assert data["engines"]["piper"]["available"] is False
    assert data["engines"]["piper"]["reason"]


def test_qa_creates_missing_report_dir(tmp_path):
    x = np.zeros(SR, dtype=np.float32)
    p = _write(tmp_path, x)
    assert qa.main([str(p), "--report", str(tmp_path / "deep" / "r.json")]) in (0, 1)
    assert (tmp_path / "deep" / "r.json").is_file()


def test_segment_loudness_consistency_is_measured():
    """level_to 把每段对齐到同一 RMS，因此段间标准差应接近 0（客观一致性代理）。"""
    class VarEngine(ToneEngine):
        def synthesize(self, text):
            self.calls.append(text)
            n = int(SR * 0.2 * max(1, len(text)))
            amp = 0.3 if len(self.calls) % 2 else 0.05
            t = np.arange(n) / SR
            return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    _, _, rep = tts.synthesize_text(VarEngine(), "第一句话。这是第二句话，稍微长一些。", target_db=-20)
    assert all("rms_dbfs" in s for s in rep["segments"])
    assert rep["segment_rms_std_db"] < 1.0
    assert rep["speech_seconds"] < rep["duration_seconds"]  # 停顿计入成品时长


def test_loudness_clamp_is_warned():
    """增益超出 ±12 dB 限制时必须在报告里说明，而不是静默偏差。"""
    class QuietEngine(ToneEngine):
        def synthesize(self, text):
            self.calls.append(text)
            n = int(SR * 0.2 * max(1, len(text)))
            t = np.arange(n) / SR
            return (0.01 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    _, _, rep = tts.synthesize_text(QuietEngine(), "很轻的一段话。", target_db=-20)
    assert any("偏离目标" in w for w in rep["warnings"])


def test_near_silent_segment_is_warned():
    class SilentEngine(ToneEngine):
        def synthesize(self, text):
            self.calls.append(text)
            n = int(SR * 0.2 * max(1, len(text)))
            t = np.arange(n) / SR
            return (0.00001 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    _, _, rep = tts.synthesize_text(SilentEngine(), "极轻的一段话。", target_db=-20)
    assert any("几乎无声" in w for w in rep["warnings"])


def test_qa_counts_multiple_clicks_separately():
    x = np.zeros(SR * 3, dtype=np.float32)
    for pos in (SR // 2, SR, SR + SR // 2, SR * 2):  # 相隔很远，应为 4 次独立事件
        x[pos] = 0.5
    assert _clicks(x) == 4


def test_qa_merges_adjacent_click_samples():
    x = np.zeros(SR * 2, dtype=np.float32)
    x[SR: SR + 3] = 0.5  # 相邻采样：1 次事件
    assert _clicks(x) == 1

"""Audio-quality regression tests: boundary pauses, conditional click repair, and QA detectors.

Deterministic fixtures only (no engine). Real-engine behaviour is covered in test_piper_integration.py
and test_intelligence.py. These tests verify mechanisms; they do NOT certify perceived naturalness.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import engines  # noqa: E402
import qa  # noqa: E402
import tts  # noqa: E402

SR = 22050


def _tone(seconds, amp=0.2, freq=220.0):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class FixedEngine(tts.Engine):
    """Returns preset arrays in order; segment text is ignored except for call counting."""
    name = "fixed"
    sample_rate = SR
    capabilities = {"rate": False}

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def available(self):
        return True, "fixture"

    def synthesize(self, text):
        out = self.outputs[self.calls % len(self.outputs)]
        self.calls += 1
        return out.copy()


# ------------------------------------------------------------------ trim_margins

def test_trim_margins_removes_only_subthreshold_margins():
    t = np.arange(int(SR * 0.4)) / SR
    speech = (0.2 * np.sin(2 * np.pi * 220 * t + 0.5)).astype(np.float32)  # first sample above threshold
    x = np.concatenate([np.zeros(SR // 2, np.float32), speech, np.zeros(SR // 2, np.float32)])
    y = tts.trim_margins(x, SR, pad_ms=30)
    # tone sample 0 is exactly 0 (below threshold), so the trimmed head may start 1 sample later
    assert abs(len(y) - (len(speech) + 2 * int(SR * 0.03))) <= 2
    assert np.array_equal(y[int(SR * 0.03): int(SR * 0.03) + len(speech)], speech)


def test_trim_margins_keeps_speech_edges_intact():
    x = _tone(0.3)
    assert np.array_equal(tts.trim_margins(x, SR), x)


def test_trim_margins_head_and_tail_flags_are_independent():
    x = np.concatenate([np.zeros(SR // 2, np.float32), _tone(0.2), np.zeros(SR // 2, np.float32)])
    head_only = tts.trim_margins(x, SR, head=True, tail=False)
    assert abs(len(head_only) - (len(x) - SR // 2 + int(SR * 0.03))) <= 2


# ------------------------------------------------------------------ boundary_repair

def _step_fixture(step_amp=0.1):
    left = _tone(0.3, amp=0.2)
    mid = np.zeros(SR // 5, np.float32)
    right = _tone(0.3, amp=0.2, freq=330.0)
    # Make both sides start / end on a clearly non-zero sample so the step is real.
    left[-1] = step_amp
    right[0] = step_amp
    return np.concatenate([left, mid, right]), len(left), len(left) + len(mid)


def test_boundary_repair_removes_steps_and_reports_them():
    x, a, b = _step_fixture()
    before = [v for _, v in qa.boundary_jumps(x, SR)]
    assert max(before) == pytest.approx(0.1, abs=1e-6)
    y, stats = tts.boundary_repair(x, SR, thr_db=-40.0, ms=2.0)
    assert stats["repairs"] == 2
    assert stats["max_step_before_dbfs"] == pytest.approx(-20.0, abs=0.05)
    assert stats["max_step_after_dbfs"] < -40.0
    assert len(y) == len(x)


def test_boundary_repair_only_touches_first_two_ms_of_speech():
    x, a, b = _step_fixture()
    y, _ = tts.boundary_repair(x, SR, ms=2.0)
    n = int(SR * 0.002)
    changed = np.flatnonzero(y != x)
    assert changed.size > 0
    # every changed sample lies within the fade window adjacent to a silence boundary
    near_left = (changed >= a - n) & (changed < a)
    near_right = (changed >= b) & (changed < b + n)
    assert np.all(near_left | near_right)


def test_boundary_repair_is_noop_on_clean_boundaries():
    left = _tone(0.2)
    left[-1], left[-2] = 0.0, 1e-5          # speech ends near zero: step well below -40 dBFS
    right = _tone(0.2, freq=330.0)
    right[0] = 1e-5                          # speech starts near zero
    x = np.concatenate([left, np.zeros(SR // 10, np.float32), right])
    y, stats = tts.boundary_repair(x, SR)
    assert stats["repairs"] == 0
    assert np.array_equal(x, y)


# ------------------------------------------------------------------ synthesize_text minimal

def test_minimal_default_pauses_are_exact_between_sentences():
    eng = FixedEngine([_tone(0.5)])
    x, sr, rep = tts.synthesize_text(eng, "第一句。第二句。", processing="minimal")
    gap = int(SR * 0.4)
    assert rep["duration_seconds"] == pytest.approx(1.0 + 0.4, abs=0.01)
    assert rep["segments"][0]["pause_after_ms"] == 400.0
    assert np.all(x[int(SR * 0.5): int(SR * 0.5) + gap] == 0)


def test_minimal_pauses_replace_engine_margin_exactly():
    padded = np.concatenate([_tone(0.5), np.zeros(SR // 2, np.float32)])  # engine adds 0.5 s tail
    eng = FixedEngine([padded])
    _, _, rep = tts.synthesize_text(eng, "第一句。第二句。", processing="minimal")
    # seg0 gets a sentence pause -> its 0.5 s engine tail is trimmed to 30 ms, then exact 400 ms.
    # seg1 is last (no pause) -> its tail is left exactly as the engine produced it (0.5 s).
    assert rep["segments"][0]["seconds"] == pytest.approx(0.5 + 0.03, abs=0.005)
    assert rep["segments"][1]["seconds"] == pytest.approx(1.0, abs=0.005)
    assert rep["duration_seconds"] == pytest.approx(0.53 + 0.4 + 1.0, abs=0.01)


def test_minimal_without_pauses_keeps_engine_output_unchanged():
    """pauses={} performs no trimming; with repair disabled the output equals raw concatenation."""
    padded = np.concatenate([_tone(0.5), np.zeros(SR // 2, np.float32)])
    eng = FixedEngine([padded])
    x, _, rep = tts.synthesize_text(eng, "第一句。第二句。", processing="minimal", pauses={},
                                    boundary_repair_enabled=False)
    assert len(x) == 2 * len(padded)
    assert np.array_equal(x, np.concatenate([padded, padded]))


def test_minimal_without_pauses_repair_only_touches_boundaries():
    padded = np.concatenate([_tone(0.5), np.zeros(SR // 2, np.float32)])
    eng = FixedEngine([padded])
    x, _, rep = tts.synthesize_text(eng, "第一句。第二句。", processing="minimal", pauses={})
    raw = np.concatenate([padded, padded])
    changed = np.flatnonzero(x != raw)
    assert rep["boundary_repair"]["repairs"] > 0
    n = int(SR * 0.002)
    # every change lies within 2 ms of a speech <-> digital-silence boundary
    boundaries = [i for i in range(len(raw)) if (raw[i] == 0) != (i > 0 and raw[i - 1] == 0)]
    near = np.zeros(len(raw), bool)
    for b in boundaries:
        near[max(0, b - n): b + n] = True
    assert np.all(near[changed])


def test_repair_can_be_disabled_for_ab_control():
    x, a, b = _step_fixture()
    eng = FixedEngine([x[:a], x[b:]])
    _, _, rep_on = tts.synthesize_text(eng, "第一句。第二句。", processing="minimal", pauses={})
    eng2 = FixedEngine([x[:a], x[b:]])
    _, _, rep_off = tts.synthesize_text(eng2, "第一句。第二句。", processing="minimal", pauses={},
                                        boundary_repair_enabled=False)
    assert rep_off["boundary_repair"] is None
    assert rep_on["boundary_repair"] is not None


def test_cli_minimal_pause_arguments_now_apply(tmp_path, monkeypatch):
    """回归：旧版 CLI 在 minimal 下忽略停顿参数；新版必须生效。"""
    calls = {}
    real = tts.synthesize_text

    def spy(engine, text, **kw):
        calls.update(kw)
        return real(engine, text, **kw)

    monkeypatch.setattr(tts, "synthesize_text", spy)
    monkeypatch.setattr(tts, "build_engine", lambda args: FixedEngine([_tone(0.3)]))
    out = tmp_path / "a.wav"
    code = tts.main(["synth", "--engine", "piper", "--model", str(tmp_path / "missing.onnx"),
                     "--text", "第一句。第二句。", "--out", str(out), "--pause-sentence", "250"])
    assert code == 0
    assert calls["pauses"]["sentence"] == 250
    assert calls["boundary_repair_enabled"] is True


# ------------------------------------------------------------------ QA detectors

def _info():
    return type("I", (), {"channels": 1, "subtype": "PCM_16"})()


def _checks(x):
    return {c["name"]: c for c in qa.run_signal_checks(np.asarray(x, np.float32), SR, _info())}


def test_qa_flags_step_at_digital_silence_boundary():
    x = np.concatenate([_tone(0.3, amp=0.2), np.zeros(SR // 4, np.float32), _tone(0.3, amp=0.2)])
    x[int(SR * 0.3) - 1] = 0.1  # 0.1 step into silence -> -20 dBFS
    c = _checks(x)["boundary_discontinuity"]
    assert c["status"] == "warn"
    assert c["value"] == pytest.approx(-20.0, abs=0.1)


def test_qa_clean_zero_pause_is_not_flagged():
    left = _tone(0.3, amp=0.2)
    left[-1], left[-2] = 0.0, 1e-5
    right = _tone(0.3, amp=0.2)
    right[0], right[1] = 1e-5, 1e-5
    x = np.concatenate([left, np.zeros(SR // 4, np.float32), right])
    assert _checks(x)["boundary_discontinuity"]["status"] == "pass"


def test_qa_dc_threshold_tightened():
    t = np.arange(SR * 2) / SR
    x = (0.005 + 0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    assert _checks(x)["dc_offset"]["status"] == "warn"  # 0.005 > 0.002 (old 0.01 would pass)


def test_qa_noise_floor_flags_audible_hiss_in_pauses():
    rng = np.random.default_rng(0)
    x = np.concatenate([_tone(0.5, amp=0.2), (0.01 * rng.standard_normal(SR)).astype(np.float32)])
    c = _checks(x)["noise_floor_dbfs"]
    assert c["status"] == "warn" and c["value"] > -45


def test_qa_exact_digital_silence_floor_passes():
    x = np.concatenate([_tone(0.5, amp=0.2), np.zeros(SR, np.float32)])
    assert _checks(x)["noise_floor_dbfs"]["status"] == "pass"


def test_qa_hum_detected_in_quiet_frames():
    t = np.arange(SR * 6) / SR  # quietest 10% of frames must span >= 0.5 s for the spectral check
    hum = (0.003 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
    x = np.concatenate([_tone(0.5, amp=0.2), hum])
    c = _checks(x)["hum_50_60hz_peak_db"]
    assert c["status"] == "warn"


def test_true_peak_detects_intersample_overshoot():
    n = np.arange(SR)
    x = np.sin(2 * np.pi * (SR / 4) * n / SR + np.pi / 4).astype(np.float64) * 0.9
    sample_peak = qa._db(np.max(np.abs(x)))
    tp = qa.true_peak_dbtp(x, SR)
    if tp is None:
        pytest.skip("scipy not installed")
    assert sample_peak == pytest.approx(qa._db(0.9 * np.sqrt(0.5)), abs=0.05)
    assert tp > sample_peak + 2.5  # 4x oversampling reveals the crest between samples


def test_integrated_loudness_matches_reference_formula():
    pyln = pytest.importorskip("pyloudnorm")
    t = np.arange(SR * 3) / SR
    x = (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float64)
    ref = pyln.Meter(SR).integrated_loudness(x)
    assert qa.integrated_lufs(x, SR) == pytest.approx(ref, abs=1e-9)
    assert -25 < ref < -21


def test_qa_reports_skip_not_fail_without_optional_deps(monkeypatch):
    monkeypatch.setattr(qa, "true_peak_dbtp", lambda x, sr: None)
    monkeypatch.setattr(qa, "integrated_lufs", lambda x, sr: None)
    c = _checks(_tone(1.0))
    assert c["true_peak_dbtp"]["status"] == "skip"
    assert c["integrated_loudness_lufs"]["status"] == "skip"


# ------------------------------------------------------------------ eSpeak error path

def test_espeak_sample_rate_mismatch_is_engine_error(tmp_path):
    """回归：异常顺序错误使 ValueError 原样外泄；现在必须为 EngineError（exit 码语义正确）。"""
    import stat
    wav = tmp_path / "out16k.wav"
    sf.write(str(wav), np.zeros(1600, np.float32), 16000)
    fake = tmp_path / "fake-espeak"
    fake.write_text(f"#!/bin/sh\ncat '{wav}'\n", encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    eng = object.__new__(engines.EspeakEngine)
    eng.binary = str(fake)
    eng.language = "en-gb"
    eng.speed = 175
    eng.sample_rate = 22050
    eng.timeout = 5.0
    with pytest.raises(engines.EngineError, match="采样率"):
        eng.synthesize("hello")


def test_true_peak_chunking_matches_whole_signal():
    rng = np.random.default_rng(3)
    x = (0.3 * rng.standard_normal(SR * 70)).astype(np.float64)
    whole = qa.true_peak_dbtp(x, SR, chunk_s=1e9)
    chunked = qa.true_peak_dbtp(x, SR, chunk_s=7.0)
    if whole is None:
        pytest.skip("scipy not installed")
    assert chunked == pytest.approx(whole, abs=1e-9)

"""Regression, engine-contract and safety tests. Synthetic audio is NOT clone evidence."""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import blindtest
import engines
import qa
import textnorm
import tts

SR = 16000


class Signal(tts.Engine):
    sample_rate = SR
    def synthesize(self, text):
        return (.1 * np.sin(np.arange(SR) * .08)).astype(np.float32)


@pytest.mark.parametrize("maximum", [0, -1])
def test_invalid_segment_length(maximum):
    with pytest.raises(ValueError):
        tts.segment("cannot hang", maximum)


@pytest.mark.parametrize("raw", [np.array([]), np.array([np.nan]), np.array([np.inf]),
                                 np.zeros((5, 2)), np.array([1.1])])
def test_engine_output_contract_rejected(raw):
    class Broken(Signal):
        def synthesize(self, text):
            return raw
    with pytest.raises(tts.EngineError):
        tts.synthesize_text(Broken(), "测试")


@pytest.mark.parametrize("kwargs", [{"max_chars": 0}, {"target_db": float("nan")},
                                    {"pauses": {"none": -1}}, {"peak_db": 2},
                                    {"max_output_seconds": 0}])
def test_invalid_pipeline_parameters(kwargs):
    with pytest.raises(ValueError):
        tts.synthesize_text(Signal(), "测试", **kwargs)


def test_output_limit():
    with pytest.raises(tts.EngineError, match="上限"):
        tts.synthesize_text(Signal(), "第一句。第二句。", max_output_seconds=1)


def test_minimal_preserves_audio_and_no_extra_silence():
    engine = Signal()
    x, _, rep = tts.synthesize_text(engine, "第一句。第二句。", processing="minimal", pauses={})
    assert np.array_equal(x, np.concatenate([engine.synthesize("x")] * 2))
    assert rep["duration_seconds"] == 2
    assert rep["validation"]["human_listening"] == "not_performed"


def test_rate_ignores_added_silence(tmp_path):
    x = np.concatenate([Signal().synthesize("x"), np.zeros(SR * 4), Signal().synthesize("x")])
    wav = tts.write_audio(tmp_path / "a.wav", x, SR)
    report = tmp_path / "q.json"
    assert qa.main([str(wav), "--text", "十个字用于这个测试啊", "--report", str(report)]) == 0
    checks = {c["name"]: c for c in json.loads(report.read_text())["checks"]}
    assert checks["active_signal_seconds"]["value"] == pytest.approx(2, abs=.02)
    assert checks["speech_rate_chars_per_s"]["value"] == pytest.approx(5, abs=.05)
    assert checks["delivery_rate_chars_per_s"]["value"] == pytest.approx(10 / 6, abs=.01)


def test_active_tail_frame_and_empty_signal():
    assert qa.active_seconds(np.full(161, .1), SR) == pytest.approx(161 / SR)
    info = SimpleNamespace(channels=1, subtype="FLOAT")
    assert qa.run_signal_checks(np.array([]), SR, info)[0]["status"] == "fail"
    assert qa.active_seconds(np.array([np.nan]), SR) == 0


def test_stereo_clipping_not_hidden_by_downmix(tmp_path):
    import soundfile as sf
    x = np.tile(np.array([[1., -1.], [-1., 1.]]), (SR, 1))
    sf.write(tmp_path / "a.wav", x, SR, subtype="FLOAT")
    assert qa.main([str(tmp_path / "a.wav")]) == 1


@pytest.mark.parametrize("locale,text", [("en-US", "Meet on 2026-10-13 at 10:30, 3.5%."),
                                          ("de", "Wert 3,5%"), ("ja", "2026年10月13日")])
def test_non_chinese_never_uses_chinese_number_rules(locale, text):
    assert textnorm.normalize_text(text, locale) == text


@pytest.mark.parametrize("text", ["2026-02-30", "2026/13/01", "25:30", "10:60"])
def test_invalid_date_time_rejected(text):
    with pytest.raises(ValueError):
        textnorm.normalize_text(text, "zh-CN")


def test_foreign_lexicon_reading_final():
    assert textnorm.normalize_text("OpenAI and GPU", "en", [("OpenAI", "GPU"), ("GPU", "gee pee you")]) == "GPU and gee pee you"


def test_explicit_language_for_asr_reference():
    assert qa.reference_text("Version 3.13", language="en-US") == "version313"
    assert qa.reference_text("三", spoken=True, language="en-US") == "三"


def test_asr_failure_is_skip(monkeypatch):
    def fail(*args):
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(qa, "_run_asr_check", fail)
    assert qa.run_asr_check("x", "x", "small", qa.PROFILES["default"])[0]["status"] == "skip"


def grant(tmp_path):
    wav = tts.write_audio(tmp_path / "reference.wav", np.tile(Signal().synthesize("x"), 4), SR)
    model = tmp_path / "model.pth"
    model.write_bytes(b"contract fixture, not a neural model")
    data = {"record_id": "test-only", "subject_id": "fixture", "consent_evidence": "fixture-not-legal-evidence",
            "consent": True, "revoked": False, "deceptive_use": False, "reference_reviewed": True,
            "allowed_purposes": ["contract-test"], "allowed_languages": ["en", "zh-cn"],
            "expires_at": "2099-01-01T00:00:00+00:00", "model_license": "test-only",
            "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(wav.read_bytes()).hexdigest()}
    auth = tmp_path / "authorization.json"
    auth.write_text(json.dumps(data))
    return wav, model, auth, data


@pytest.mark.parametrize("key,value", [("consent", False), ("revoked", True), ("deceptive_use", True),
                                       ("reference_reviewed", False), ("expires_at", "2000-01-01T00:00:00+00:00"),
                                       ("expires_at", "2099-01-01T00:00:00"), ("reference_sha256", "bad"),
                                       ("allowed_purposes", [])])
def test_clone_fail_closed(tmp_path, key, value):
    wav, model, auth, data = grant(tmp_path)
    data[key] = value
    auth.write_text(json.dumps(data))
    with pytest.raises(tts.EngineError):
        engines.XTTSEngine(model, "missing", "en", wav, auth, "contract-test")


def test_no_authorization_no_clone_model_load():
    with pytest.raises(tts.EngineError, match="克隆需"):
        engines.XTTSEngine(None, None, "en", None, None, None)


def test_xtts_real_api_contract_with_test_double(tmp_path, monkeypatch):
    wav, model, auth, data = grant(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": "xtts", "languages": ["en", "zh-cn"]}))
    calls = []
    class FakeAPI:
        def __init__(self, **kw):
            assert kw["model_path"] == str(model) and kw["config_path"] == str(config)
            self.synthesizer = SimpleNamespace(output_sample_rate=SR)
        def tts(self, **kw):
            calls.append(kw)
            return Signal().synthesize("x")
    module = ModuleType("TTS.api")
    module.TTS = FakeAPI
    monkeypatch.setitem(sys.modules, "TTS", ModuleType("TTS"))
    monkeypatch.setitem(sys.modules, "TTS.api", module)
    engine = engines.XTTSEngine(model, config, "en", wav, auth, "contract-test")
    x, sr, rep = tts.synthesize_text(engine, "A different sentence.", processing="minimal", pauses={})
    assert sr == SR and x.size
    assert calls[0]["speaker_wav"] == str(wav) and calls[0]["split_sentences"] is False
    assert "subject_id" not in rep["authorization"] and "consent_evidence" not in rep["authorization"]
    data["revoked"] = True
    auth.write_text(json.dumps(data))
    with pytest.raises(tts.EngineError):
        engine.synthesize("stop")
    assert len(calls) == 1


def test_xtts_model_binding(tmp_path):
    wav, model, auth, data = grant(tmp_path)
    config = tmp_path / "config.json"
    config.write_text('{"model":"xtts","languages":["en"]}')
    model.write_bytes(b"modified")
    with pytest.raises(tts.EngineError, match="model_sha256"):
        engines.XTTSEngine(model, config, "en", wav, auth, "contract-test")


@pytest.mark.skipif(not shutil.which("espeak-ng"), reason="espeak-ng executable missing")
@pytest.mark.parametrize("language,text", [("en-US", "The date is October 13, 2026."),
                                            ("zh-CN", "今天是二零二六年十月十三日。"),
                                            ("de", "Guten Morgen."), ("ja", "こんにちは。")])
def test_real_espeak_multilingual(language, text, tmp_path):
    engine = engines.EspeakEngine(language)
    x, sr, rep = tts.synthesize_text(engine, text, processing="minimal", pauses={})
    wav = tts.write_audio(tmp_path / "o.wav", x, sr)
    y, rate, info = qa.load_mono(wav)
    assert rate == sr
    assert not [c for c in qa.run_signal_checks(y, rate, info) if c["status"] == "fail"]
    assert rep["capabilities"]["clone"] is False


@pytest.mark.skipif(not shutil.which("espeak-ng"), reason="espeak-ng missing")
def test_real_direction_rate_and_restore(tmp_path):
    engine = engines.EspeakEngine("en-US")
    text = "This discovery changes everything."
    plan = {"intent": "reveal", "audience": "learners", "character": "narrator",
            "beats": [{"index": 0, "text": text, "intent": "let the consequence land",
                       "rationale": "new key idea, slow without simulated emotion", "length_scale": 1.3}]}
    x, _, rep = tts.synthesize_text(engine, text, direction=plan, processing="minimal", pauses={})
    y, _, _ = tts.synthesize_text(engine, text, processing="minimal", pauses={})
    assert len(x) > len(y) and engine.speed == 175
    assert rep["segments"][0]["direction"]["length_scale"] == 1.3
    plan["beats"][0]["emotion"] = "happy"
    with pytest.raises(ValueError, match="控制"):
        tts.synthesize_text(engine, text, direction=plan)


def test_direction_text_mismatch_preflight():
    plan = {"intent": "explain", "audience": "all", "character": "narrator",
            "beats": [{"index": 0, "text": "wrong", "intent": "clarify", "rationale": "reason"}]}
    with pytest.raises(ValueError, match="文本"):
        tts.synthesize_text(Signal(), "test", direction=plan)


def test_cli_does_not_overwrite_source_or_report_audio(tmp_path):
    source = tmp_path / "source.wav"
    source.write_text("保护")
    assert tts.main(["synth", "--input", str(source), "--out", str(source)]) == 4
    assert source.read_text() == "保护"
    assert tts.main(["synth", "--text", "x", "--out", str(source), "--report", str(source)]) == 4
    assert source.read_text() == "保护"


def test_missing_input_has_failure_report(tmp_path):
    report = tmp_path / "report.json"
    assert tts.main(["synth", "--input", str(tmp_path / "missing"), "--out", str(tmp_path / "o.wav"),
                     "--report", str(report)]) == 4
    assert json.loads(report.read_text())["ok"] is False


def test_wav_master_only(tmp_path):
    with pytest.raises(ValueError):
        tts._validate_outputs(tmp_path / "o.flac", [])


def test_blind_order_without_identity_and_partial_dimension(tmp_path):
    wav = tts.write_audio(tmp_path / "engine-identity.wav", Signal().synthesize("x"), SR)
    pkg = tmp_path / "pkg"
    blindtest.main(["prepare", "--audio", str(wav), "--out", str(pkg)])
    assert "engine-identity" not in (pkg / "listen/order.txt").read_text()
    rows = list(csv.reader((pkg / "ratings.csv").read_text().splitlines()))
    rows[1][1] = "5"
    with (pkg / "ratings.csv").open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    blindtest.main(["score", "--pkg", str(pkg)])
    scores = json.loads((pkg / "scores.json").read_text())
    assert scores["dimension_mean"]["naturalness"] is None and not scores["complete"]
    with pytest.raises(SystemExit, match="未完整"):
        blindtest.main(["unblind", "--pkg", str(pkg)])
    rows.append(rows[1])
    with (pkg / "ratings.csv").open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    with pytest.raises(SystemExit, match="标签"):
        blindtest.main(["score", "--pkg", str(pkg)])


def test_qa_rejects_unmatched_synthesis_report(tmp_path):
    wav = tts.write_audio(tmp_path / "a.wav", Signal().synthesize("x"), SR)
    report = tmp_path / "s.json"
    report.write_text(json.dumps({"normalized_text": "test", "language": "en", "output_sha256": "wrong"}))
    assert qa.main([str(wav), "--synthesis-report", str(report)]) == 2


def test_missing_qa_text_file_is_diagnosable(tmp_path):
    wav = tts.write_audio(tmp_path / "a.wav", Signal().synthesize("x"), SR)
    assert qa.main([str(wav), "--text-file", str(tmp_path / "missing")]) == 2


def test_direction_failure_restores_engine_rate():
    class Broken(Signal):
        name = "espeak"
        speed = 175
        capabilities = {"rate": True}
        def synthesize(self, text):
            raise tts.EngineError("transient")
    engine = Broken()
    plan = {"intent": "explain", "audience": "all", "character": "guide",
            "beats": [{"index": 0, "text": "test", "intent": "clarify", "rationale": "new idea", "length_scale": 1.3}]}
    with pytest.raises(tts.EngineError):
        tts.synthesize_text(engine, "test", direction=plan)
    assert engine.speed == 175


@pytest.mark.skipif(not shutil.which("espeak-ng"), reason="espeak-ng missing")
def test_eval_asr_skip_has_raw_evidence_and_no_keyerror(tmp_path, monkeypatch):
    import importlib.util
    path = Path(__file__).resolve().parent / "run_eval.py"
    spec = importlib.util.spec_from_file_location("evaluation", path)
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    monkeypatch.setattr(qa, "run_asr_check", lambda *a, **kw: [{"name": "asr_cer", "status": "skip", "value": None, "detail": "fixture missing ASR"}])
    out = tmp_path / "eval"
    assert evaluation.main(["--engine", "espeak", "--language", "en-US", "--ids", "en_us_digits",
                            "--repeat", "1", "--asr", "missing", "--out", str(out)]) == 0
    result = json.loads((out / "summary.json").read_text())
    assert result["rows"][0]["norm"]["cer"] is None
    assert result["runs"][0]["asr"][0]["status"] == "skip"
    assert result["runs"][0]["sha256"]


@pytest.mark.skipif(not shutil.which("espeak-ng"), reason="espeak-ng missing")
def test_unsupported_locale_not_silently_downgraded():
    with pytest.raises(tts.EngineError, match="未安装语言"):
        engines.EspeakEngine("en-JP")


def test_atomic_write_does_not_damage_prior_audio(tmp_path, monkeypatch):
    import soundfile as sf
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"prior artifact")
    def fail(*a, **kw):
        raise RuntimeError("disk failure fixture")
    monkeypatch.setattr(sf, "write", fail)
    with pytest.raises(RuntimeError):
        tts.write_audio(wav, Signal().synthesize("x"), SR)
    assert wav.read_bytes() == b"prior artifact"
    assert list(tmp_path.iterdir()) == [wav]


def test_qa_report_cannot_overwrite_audio(tmp_path):
    wav = tts.write_audio(tmp_path / "a.wav", Signal().synthesize("x"), SR)
    original = wav.read_bytes()
    assert qa.main([str(wav), "--report", str(wav)]) == 2
    assert wav.read_bytes() == original


def test_implicit_piper_config_protected(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fixture")
    config = Path(str(model) + ".json")
    config.write_text("protected")
    assert tts.main(["synth", "--model", str(model), "--text", "test", "--out", str(tmp_path / "o.wav"),
                     "--report", str(config)]) == 4
    assert config.read_text() == "protected"


def test_authorization_purpose_must_be_list(tmp_path):
    wav, model, auth, data = grant(tmp_path)
    data["allowed_purposes"] = "contract-test"
    auth.write_text(json.dumps(data))
    with pytest.raises(tts.EngineError, match="字符串列表"):
        engines.validate_authorization(auth, wav, "contract")


def test_foreign_word_split_obeys_character_budget():
    segs = tts.segment("abcdefghij remainder morewords", max_chars=10)
    assert all(len(s.text) <= 10 for s in segs)
    assert "".join(s.text for s in segs).replace(" ", "") == "abcdefghijremaindermorewords"


def test_piper_wraps_native_failure():
    engine = object.__new__(engines.PiperEngine)
    def broken(*a, **kw):
        raise RuntimeError("native failure fixture")
    engine.voice = SimpleNamespace(synthesize=broken)
    engine.syn = None
    with pytest.raises(tts.EngineError, match="Piper 合成失败"):
        engine.synthesize("test")

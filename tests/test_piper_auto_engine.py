import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import engines  # noqa: E402
import tts  # noqa: E402


class _FakeVoice:
    loads = []

    @classmethod
    def load(cls, model_path, config_path, **kwargs):
        cls.loads.append((model_path, config_path, kwargs))
        return SimpleNamespace(config=SimpleNamespace(sample_rate=22050))


class _FakeSynthesisConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _install_fake_piper(monkeypatch):
    _FakeVoice.loads.clear()
    piper = types.ModuleType("piper")
    piper.__path__ = []
    piper.PiperVoice = _FakeVoice
    config = types.ModuleType("piper.config")
    config.SynthesisConfig = _FakeSynthesisConfig
    chinese = types.ModuleType("piper.phonemize_chinese")
    chinese.ChinesePhonemizer = object
    monkeypatch.setitem(sys.modules, "piper", piper)
    monkeypatch.setitem(sys.modules, "piper.config", config)
    monkeypatch.setitem(sys.modules, "piper.phonemize_chinese", chinese)


def _write_frontend_bundle(tmp_path, *, corrupt_g2pw=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = tmp_path / "voice.onnx"
    config = tmp_path / "voice.onnx.json"
    model.write_bytes(b"pinned fake onnx")
    config.write_text(json.dumps({
        "language": {"code": "zh_CN"}, "num_speakers": 1,
        "phoneme_type": "pinyin", "audio": {"sample_rate": 22050},
    }), encoding="utf-8")
    required = {
        "g2pw_model", "g2pw_config", "g2pw_polyphonic", "g2pw_monophonic",
        "g2pw_bopomofo_to_pinyin", "g2pw_char_bopomofo",
        "g2pw_s2t_dictionary", "bert_vocab", "bert_tokenizer_config",
    }
    paths, expected = {}, {
        "model": hashlib.sha256(model.read_bytes()).hexdigest(),
        "config": hashlib.sha256(config.read_bytes()).hexdigest(),
    }
    for key in sorted(required):
        path = tmp_path / key
        content = (b"wrong" if corrupt_g2pw and key == "g2pw_model" else f"asset:{key}".encode())
        path.write_bytes(content)
        paths[key] = path
        expected[key] = hashlib.sha256(f"asset:{key}".encode()).hexdigest()
    return SimpleNamespace(
        model_path=model,
        config_path=config,
        voice_dir=tmp_path,
        additional_asset_paths=paths,
        expected_sha256=expected,
        metadata={"license": {"repo_license_metadata": "MIT"}, "voice_id": "fake-zh"},
        downloaded=False,
    )


def _args(**overrides):
    values = dict(
        engine="piper", reference_audio=None, authorization=None,
        piper_voice="fake-zh", model=None, config=None,
        piper_cache_dir=None, accept_model_license=True, download_timeout=1200.0,
        speaker=None, length_scale=1.0, noise_scale=.667, noise_w=.8,
        cuda=False, language="auto", timeout=60.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_unmanaged_pinyin_model_cannot_trigger_mutable_runtime_download(tmp_path, monkeypatch):
    _install_fake_piper(monkeypatch)
    model = tmp_path / "local.onnx"
    config = tmp_path / "local.onnx.json"
    model.write_bytes(b"local model")
    config.write_text(json.dumps({"language": {"code": "zh_CN"}, "phoneme_type": "pinyin"}),
                      encoding="utf-8")
    with pytest.raises(engines.EngineError, match="固定校验的完整 g2pW/BERT"):
        engines.PiperEngine(model=model, config=config)
    assert _FakeVoice.loads == []


def test_managed_piper_voice_repairs_one_hash_failure_without_engine_fallback(tmp_path, monkeypatch):
    _install_fake_piper(monkeypatch)
    damaged = _write_frontend_bundle(tmp_path / "damaged", corrupt_g2pw=True)
    repaired = _write_frontend_bundle(tmp_path / "repaired", corrupt_g2pw=False)
    calls = []

    def ensure(_voice_id, **kwargs):
        calls.append(kwargs["force_download"])
        return damaged if not kwargs["force_download"] else repaired

    monkeypatch.setattr(tts, "ensure_piper_voice", ensure)
    engine = tts.build_engine(_args())
    assert calls == [False, True]
    assert len(_FakeVoice.loads) == 1  # first, tampered bundle was rejected before model load
    assert engine.model_path == str(repaired.model_path)
    assert engine.voice_manifest["voice_id"] == "fake-zh"
    assert engine.capabilities["model_license"]["repo_license_metadata"] == "MIT"
    for key in repaired.additional_asset_paths:
        assert key in engine.bound_assets

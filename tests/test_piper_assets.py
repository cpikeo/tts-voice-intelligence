import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import piper_assets  # noqa: E402
from engines import ModelAssetIntegrityError, _verify_expected_asset  # noqa: E402


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _fake_catalog(tmp_path, monkeypatch):
    revision = "a" * 40
    g2pw_revision = "b" * 40
    lookup_revision = "c" * 40
    tokenizer_revision = "d" * 40

    model = b"fake-onnx-model"
    config = b'{"language":{"code":"zh_CN"},"phoneme_type":"pinyin"}'
    archive_config = b"manual_seed = 1313\nmodel_source = 'bert-base-chinese'\n"
    patched_config = (
        b'import os\nmanual_seed = 1313\n'
        b'model_source = os.path.join(os.path.dirname(__file__), "bert-base-chinese")\n'
    )
    members = {
        "g2pw.onnx": b"fake-g2pw-model",
        "config.py": archive_config,
        "POLYPHONIC_CHARS.txt": b"poly\n",
        "MONOPHONIC_CHARS.txt": b"mono\n",
        "version": b"test\n",
    }
    tar_bytes_io = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes_io, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    archive_bytes = tar_bytes_io.getvalue()

    lookup = [
        ("g2pw_bopomofo_to_pinyin", "bopomofo_to_pinyin_wo_tune_dict.json", b"{}"),
        ("g2pw_char_bopomofo", "char_bopomofo_dict.json", b"[]"),
        ("g2pw_s2t_dictionary", "bert-base-chinese_s2t_dict.txt", b"a\tb\n"),
    ]
    tokenizer = [
        ("bert_vocab", "vocab.txt", b"[PAD]\n[UNK]\n"),
        ("bert_tokenizer_config", "tokenizer_config.json", b'{"do_lower_case":false}'),
    ]
    voice = {
        "repo_id": "rhasspy/piper-voices",
        "revision": revision,
        "language": "zh-CN",
        "speaker_count": 1,
        "model": {"path": "a/model.onnx", "filename": "model.onnx",
                  "size": len(model), "sha256": _sha(model)},
        "config": {"path": "a/model.onnx.json", "filename": "model.onnx.json",
                   "size": len(config), "sha256": _sha(config)},
        "license": {"policy_id": "fake-policy-v1", "weight_license_note": "fake license",
                    "license_sources": [{"name": "test license", "repo_id": "rhasspy/piper-voices",
                                         "revision": revision, "filename": "licenses/README.md",
                                         "url": f"https://huggingface.co/rhasspy/piper-voices/resolve/{revision}/README.md",
                                         "size": len(b"license proof"), "sha256": _sha(b"license proof")}],
                    "repo_license_metadata": "MIT"},
        "frontend": {
            "g2pw_archive": {"repo_id": "rhasspy/piper-checkpoints", "revision": revision,
                              "path": "g2pw.tar.gz", "size": len(archive_bytes),
                              "sha256": _sha(archive_bytes)},
            "g2pw_version": {"archive_member": "version", "filename": "g2pw.version",
                              "size": len(members["version"]), "sha256": _sha(members["version"])},
            "g2pw_files": [
                {"name": "g2pw_model", "archive_member": "g2pw.onnx", "filename": "g2pw.onnx",
                 "size": len(members["g2pw.onnx"]), "sha256": _sha(members["g2pw.onnx"])},
                {"name": "g2pw_config", "archive_member": "config.py", "filename": "config.py",
                 "size": len(patched_config), "sha256": _sha(patched_config),
                 "source_size": len(archive_config), "source_sha256": _sha(archive_config)},
                {"name": "g2pw_polyphonic", "archive_member": "POLYPHONIC_CHARS.txt",
                 "filename": "POLYPHONIC_CHARS.txt", "size": len(members["POLYPHONIC_CHARS.txt"]),
                 "sha256": _sha(members["POLYPHONIC_CHARS.txt"])},
                {"name": "g2pw_monophonic", "archive_member": "MONOPHONIC_CHARS.txt",
                 "filename": "MONOPHONIC_CHARS.txt", "size": len(members["MONOPHONIC_CHARS.txt"]),
                 "sha256": _sha(members["MONOPHONIC_CHARS.txt"])},
            ],
            "g2pw_lookup_tables": {
                "repo_id": "GitYCC/g2pW", "revision": lookup_revision,
                "files": [
                    {"name": name, "path": f"g2pw/{filename}", "filename": filename,
                     "size": len(data), "sha256": _sha(data)}
                    for name, filename, data in lookup
                ],
            },
            "bert_tokenizer": {
                "repo_id": "google-bert/bert-base-chinese", "revision": tokenizer_revision,
                "files": [
                    {"name": name, "path": filename, "filename": f"bert-base-chinese/{filename}",
                     "size": len(data), "sha256": _sha(data)}
                    for name, filename, data in tokenizer
                ],
            },
        },
    }
    registry_path = tmp_path / "catalog.json"
    registry_path.write_text(json.dumps({"schema_version": 1, "voices": {"test-voice": voice}}),
                             encoding="utf-8")
    monkeypatch.setattr(piper_assets, "_REGISTRY", registry_path)

    downloads = {}
    downloads[f"https://huggingface.co/rhasspy/piper-voices/resolve/{revision}/a/model.onnx"] = model
    downloads[f"https://huggingface.co/rhasspy/piper-voices/resolve/{revision}/a/model.onnx.json"] = config
    downloads[f"https://huggingface.co/rhasspy/piper-voices/resolve/{revision}/README.md"] = b"license proof"
    downloads[f"https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/{revision}/g2pw.tar.gz"] = archive_bytes
    for name, filename, data in lookup:
        downloads[f"https://raw.githubusercontent.com/GitYCC/g2pW/{lookup_revision}/g2pw/{filename}"] = data
    for _, filename, data in tokenizer:
        downloads[f"https://huggingface.co/google-bert/bert-base-chinese/resolve/{tokenizer_revision}/{filename}"] = data
    return voice, downloads, patched_config, members, model, config


class _Response(io.BytesIO):
    def __init__(self, content):
        super().__init__(content)
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def _mock_urlopen(monkeypatch, downloads):
    calls = []

    def open_url(request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        calls.append((url, timeout))
        if url not in downloads:
            raise OSError(f"unexpected URL: {url}")
        return _Response(downloads[url])

    monkeypatch.setattr(piper_assets.urllib.request, "urlopen", open_url)
    return calls


def test_pinned_piper_bundle_download_cache_and_license_gate(tmp_path, monkeypatch):
    _voice, downloads, patched_config, members, model, config = _fake_catalog(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    calls = _mock_urlopen(monkeypatch, downloads)

    with pytest.raises(piper_assets.PiperAssetError, match="首次使用.*需审阅"):
        piper_assets.ensure_piper_voice("test-voice", cache_dir=cache)
    assert calls == []

    result = piper_assets.ensure_piper_voice(
        "test-voice", cache_dir=cache, accept_model_license=True)
    assert result.downloaded is True
    assert result.model_path.read_bytes() == model
    assert result.config_path.read_bytes() == config
    assert (result.voice_dir / "licenses" / "README.md").read_bytes() == b"license proof"
    assert (result.voice_dir / "config.py").read_bytes() == patched_config
    assert (result.voice_dir / "g2pw.onnx").read_bytes() == members["g2pw.onnx"]
    assert result.metadata["model_sha256"] == _sha(model)
    assert result.metadata["config_sha256"] == _sha(config)
    assert result.metadata["accepted_at"]
    provenance = json.loads((result.voice_dir / "provisioning.json").read_text(encoding="utf-8"))
    assert provenance["voice_revision"] == _voice["revision"]
    assert provenance["license_sources_sha256"] == {
        item["filename"]: item["sha256"] for item in _voice["license"]["license_sources"]
    }
    assert len(calls) == len(downloads)

    # Reuse must not contact the network; content authenticity is bound by PiperEngine's
    # expected-hash pass rather than duplicated in this path resolver.
    def no_network(*_args, **_kwargs):
        raise AssertionError("cache hit attempted network access")
    monkeypatch.setattr(piper_assets.urllib.request, "urlopen", no_network)
    cached = piper_assets.ensure_piper_voice("test-voice", cache_dir=cache)
    assert cached.downloaded is False
    assert cached.model_path == result.model_path

    provenance_path = cached.voice_dir / "provisioning.json"
    provenance_path.write_text("{}", encoding="utf-8")
    refreshed = piper_assets.ensure_piper_voice("test-voice", cache_dir=cache)
    assert refreshed.downloaded is False
    assert json.loads(provenance_path.read_text(encoding="utf-8")) == (
        piper_assets._provisioning_record("test-voice", _voice)
    )


def test_cache_tampering_is_detected_by_streaming_engine_hash(tmp_path, monkeypatch):
    _voice, downloads, _patched, _members, _model, _config = _fake_catalog(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    _mock_urlopen(monkeypatch, downloads)
    result = piper_assets.ensure_piper_voice(
        "test-voice", cache_dir=cache, accept_model_license=True)

    target = result.additional_asset_paths["g2pw_model"]
    target.write_bytes(b"X" * target.stat().st_size)  # same size: stat-only resolver must not trust it
    with pytest.raises(ModelAssetIntegrityError, match="SHA-256"):
        _verify_expected_asset(target, result.expected_sha256["g2pw_model"], label="g2pw_model")

    # An explicit integrity failure triggers one complete fixed-version repair, not a fallback.
    repaired = piper_assets.ensure_piper_voice(
        "test-voice", cache_dir=cache, force_download=True)
    assert repaired.downloaded is True
    assert target.read_bytes() == _members["g2pw.onnx"]


def test_tampered_cached_license_snapshot_is_repaired_from_fixed_source(tmp_path, monkeypatch):
    voice, downloads, _patched, _members, _model, _config = _fake_catalog(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    _mock_urlopen(monkeypatch, downloads)
    result = piper_assets.ensure_piper_voice(
        "test-voice", cache_dir=cache, accept_model_license=True)
    source = voice["license"]["license_sources"][0]
    target = result.voice_dir.joinpath(*Path(source["filename"]).parts)
    target.write_bytes(b"X" * source["size"])

    repaired = piper_assets.ensure_piper_voice("test-voice", cache_dir=cache)
    assert repaired.downloaded is True
    assert target.read_bytes() == downloads[source["url"]]
    assert _sha(target.read_bytes()) == source["sha256"]


def test_bad_download_does_not_replace_previous_cache(tmp_path, monkeypatch):
    _voice, downloads, _patched, _members, model, _config = _fake_catalog(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    _mock_urlopen(monkeypatch, downloads)
    result = piper_assets.ensure_piper_voice(
        "test-voice", cache_dir=cache, accept_model_license=True)

    prior = result.model_path.read_bytes()
    result.model_path.write_bytes(b"tampered-old-cache")
    prior_bad_cache = result.model_path.read_bytes()
    bad_downloads = dict(downloads)
    model_url = next(url for url in downloads if url.endswith("/a/model.onnx"))
    bad_downloads[model_url] = b"X" * len(model)
    _mock_urlopen(monkeypatch, bad_downloads)

    with pytest.raises(piper_assets.PiperAssetError, match="SHA-256"):
        piper_assets.ensure_piper_voice("test-voice", cache_dir=cache, force_download=True)
    assert result.model_path.read_bytes() == prior_bad_cache
    assert not list(result.voice_dir.parent.glob("*.staging-*"))
    assert prior == model


def test_offline_missing_cache_and_unregistered_voice_fail_clearly(tmp_path, monkeypatch):
    _fake_catalog(tmp_path, monkeypatch)
    monkeypatch.setenv("TTS_PIPER_OFFLINE", "1")
    with pytest.raises(piper_assets.PiperAssetError, match="离线模式"):
        piper_assets.ensure_piper_voice(
            "test-voice", cache_dir=tmp_path / "empty-cache", accept_model_license=True)
    with pytest.raises(piper_assets.PiperAssetError, match="未审核/未登记"):
        piper_assets.ensure_piper_voice("arbitrary-voice", cache_dir=tmp_path / "cache")

"""Pinned, integrity-checked first-use provisioning for Piper voice assets.

The package never consults a mutable voice index and never accepts arbitrary remote URLs.
It downloads only entries in assets/piper_voices.json, verifies SHA-256 while streaming, and
installs the complete voice/frontend bundle by a directory-level replace. Actual use still binds
and rechecks every active asset in PiperEngine; this module deliberately uses size-only checks
for an already-present cache so a valid 160+ MB g2pW ONNX file is not hashed twice per load.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


_REGISTRY = Path(__file__).resolve().parents[1] / "assets" / "piper_voices.json"
_HEX = set("0123456789abcdef")
_BLOCK = 1024 * 1024


class PiperAssetError(RuntimeError):
    """Pinned asset lookup, licensing, download, or verification failure."""


@dataclass(frozen=True)
class PiperVoiceAssets:
    voice_id: str
    model_path: Path
    config_path: Path
    additional_asset_paths: dict
    expected_sha256: dict
    metadata: dict
    downloaded: bool
    cache_root: Path
    voice_dir: Path


def _valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and set(value.lower()) <= _HEX


def _load_catalog():
    try:
        catalog = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PiperAssetError(f"无法读取固定 Piper 资产目录 {_REGISTRY}: {exc}") from exc
    if catalog.get("schema_version") != 1 or not isinstance(catalog.get("voices"), dict):
        raise PiperAssetError("Piper 资产目录版本无效；拒绝使用未审核目录")
    return catalog


def _voice_spec(voice_id):
    if not isinstance(voice_id, str) or not voice_id:
        raise PiperAssetError("请通过 --piper-voice 明确选择一个已审核的音色 ID")
    catalog = _load_catalog()
    spec = catalog["voices"].get(voice_id)
    if spec is None:
        known = ", ".join(sorted(catalog["voices"])) or "（目录为空）"
        raise PiperAssetError(
            f"未审核/未登记的 Piper 音色：{voice_id}。当前固定目录：{known}。"
            "不会从动态索引或任意仓库下载；请提供本地 --model，或先审计后更新目录。"
        )
    if (len(PurePosixPath(voice_id).parts) != 1
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in voice_id)):
        raise PiperAssetError("固定 Piper 音色 ID 含有不安全路径字符")
    revision = spec.get("revision", "")
    if len(revision) != 40 or any(c not in _HEX for c in revision.lower()):
        raise PiperAssetError(f"音色目录没有固定的 40 位提交版本：{voice_id}")
    _validate_asset_spec(spec)
    return spec


def _validate_asset_spec(spec):
    items = [("model", spec.get("model")), ("config", spec.get("config"))]
    frontend = spec.get("frontend", {})
    items.extend((f"g2pw:{x.get('name')}", x) for x in frontend.get("g2pw_files", []))
    items.extend((f"g2pw_lookup:{x.get('name')}", x)
                 for x in frontend.get("g2pw_lookup_tables", {}).get("files", []))
    items.extend((f"bert_tokenizer:{x.get('name')}", x)
                 for x in frontend.get("bert_tokenizer", {}).get("files", []))
    revision = spec.get("revision", "")
    license_sources = spec.get("license", {}).get("license_sources", [])
    if not isinstance(license_sources, list) or not license_sources:
        raise PiperAssetError("Piper 音色没有固定的许可来源快照")
    for source in license_sources:
        if not isinstance(source, dict):
            raise PiperAssetError("Piper 许可来源记录格式无效")
        filename = source.get("filename")
        url = source.get("url")
        source_revision = source.get("revision")
        source_repo = source.get("repo_id")
        size = source.get("size")
        if (not isinstance(filename, str) or not isinstance(url, str)
                or not isinstance(source_revision, str) or not isinstance(source_repo, str)
                or not source_repo or len(source_revision) != 40
                or any(c not in _HEX for c in source_revision.lower())
                or not _valid_sha(source.get("sha256"))
                or not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= 2_000_000):
            raise PiperAssetError("Piper 许可来源的固定版本/大小/SHA-256 无效")
        rel = PurePosixPath(filename)
        source_prefix = (f"https://huggingface.co/datasets/{source_repo}/resolve/{source_revision}/"
                         if source_repo == "rhasspy/piper-checkpoints"
                         else f"https://huggingface.co/{source_repo}/resolve/{source_revision}/")
        source_path = url[len(source_prefix):] if url.startswith(source_prefix) else ""
        decoded_path = urllib.parse.unquote(source_path)
        url_parts = urllib.parse.urlsplit(url)
        if (rel.is_absolute() or not rel.parts or any(p in ("", ".", "..") for p in rel.parts)
                or not source_path or url_parts.query or url_parts.fragment
                or url != source_prefix + urllib.parse.quote(decoded_path, safe="/")
                or PurePosixPath(decoded_path).is_absolute()
                or any(p in ("", ".", "..") for p in PurePosixPath(decoded_path).parts)):
            raise PiperAssetError("Piper 许可来源的固定 URL/相对路径无效")
    seen_names = set()
    for label, item in items:
        if not isinstance(item, dict) or not _valid_sha(item.get("sha256")):
            raise PiperAssetError(f"资产目录的 SHA-256 缺失/格式错误：{label}")
        size = item.get("size")
        if not isinstance(size, int) or size <= 0:
            raise PiperAssetError(f"资产目录的文件大小无效：{label}")
        filename = item.get("filename")
        if not isinstance(filename, str):
            raise PiperAssetError(f"资产目录缺少文件名：{label}")
        rel = PurePosixPath(filename)
        if rel.is_absolute() or not rel.parts or any(p in ("", ".", "..") for p in rel.parts):
            raise PiperAssetError(f"资产目录包含不安全的相对路径：{label}")
        key = rel.as_posix().lower()
        if key in seen_names:
            raise PiperAssetError(f"资产目录文件名重复：{filename}")
        seen_names.add(key)
    archive = frontend.get("g2pw_archive", {})
    if not _valid_sha(archive.get("sha256")) or not isinstance(archive.get("size"), int):
        raise PiperAssetError("g2pW 固定归档的 SHA-256/大小无效")
    if archive.get("repo_id") != "rhasspy/piper-checkpoints":
        raise PiperAssetError("g2pW 下载来源不在审核的固定仓库中")
    archive_revision = archive.get("revision", "")
    if len(archive_revision) != 40 or any(c not in _HEX for c in archive_revision.lower()):
        raise PiperAssetError("g2pW archive 没有固定提交版本")
    version_item = frontend.get("g2pw_version", {})
    version_filename = PurePosixPath(version_item.get("filename", ""))
    if (not _valid_sha(version_item.get("sha256")) or not isinstance(version_item.get("size"), int)
            or version_item.get("archive_member") != "version"
            or version_filename.is_absolute() or version_filename.as_posix() != "g2pw.version"):
        raise PiperAssetError("g2pW archive version 文件的固定清单无效")
    for section in (frontend.get("g2pw_lookup_tables", {}), frontend.get("bert_tokenizer", {})):
        revision = section.get("revision", "")
        if len(revision) != 40 or any(c not in _HEX for c in revision.lower()):
            raise PiperAssetError("Piper 中文前端资产没有固定提交版本")


def default_cache_dir():
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return base / "arena-tts" / "piper"


def _cache_root(cache_dir=None):
    path = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    try:
        return path.resolve()
    except OSError as exc:
        raise PiperAssetError(f"无法解析 Piper 缓存目录 {path}: {exc}") from exc


def _asset_records(spec, voice_dir):
    records = {}
    for key in ("model", "config"):
        item = spec[key]
        records[key] = {
            "path": voice_dir / item["filename"],
            "size": item["size"], "sha256": item["sha256"], "kind": key,
        }
    frontend = spec["frontend"]
    for section_name, section in (("g2pw", frontend["g2pw_files"]),
                                  ("g2pw_lookup", frontend["g2pw_lookup_tables"]["files"]),
                                  ("bert_tokenizer", frontend["bert_tokenizer"]["files"])):
        for item in section:
            records[item["name"]] = {
                "path": voice_dir.joinpath(*PurePosixPath(item["filename"]).parts),
                "size": item["size"], "sha256": item["sha256"], "kind": section_name,
            }
    version_item = frontend["g2pw_version"]
    records["g2pw_version"] = {
        "path": voice_dir / version_item["filename"],
        "size": version_item["size"], "sha256": version_item["sha256"], "kind": "g2pw",
    }
    return records


def expected_voice_paths(voice_id, cache_dir=None):
    """Return fixed cache locations without creating directories or touching the network."""
    spec = _voice_spec(voice_id)
    root = _cache_root(cache_dir)
    voice_dir = root / "voices" / voice_id / spec["revision"]
    records = _asset_records(spec, voice_dir)
    return {
        "cache_root": root,
        "voice_dir": voice_dir,
        "model_path": records["model"]["path"],
        "config_path": records["config"]["path"],
        "additional_asset_paths": {k: v["path"] for k, v in records.items()
                                    if k not in ("model", "config")},
    }


def _source_url(repo_id, revision, path, *, dataset=False, raw_github=False):
    path = PurePosixPath(path)
    if path.is_absolute() or any(p in ("", ".", "..") for p in path.parts):
        raise PiperAssetError(f"不安全的固定资产 URL 路径：{path}")
    quoted_path = urllib.parse.quote(path.as_posix(), safe="/")
    if raw_github:
        if "/" not in repo_id:
            raise PiperAssetError("GitHub 仓库 ID 格式错误")
        return f"https://raw.githubusercontent.com/{repo_id}/{revision}/{quoted_path}"
    if dataset:
        return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{quoted_path}"
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{quoted_path}"


def _download_file(url, destination, expected_size, expected_sha256, *, timeout=45.0,
                   max_download_seconds=1200.0, deadline=None):
    """Stream one pinned file to a new staged path; never trusts Content-Length alone."""
    if not url.startswith("https://"):
        raise PiperAssetError("模型资产只允许 HTTPS 来源")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    deadline = deadline if deadline is not None else time.monotonic() + max_download_seconds
    request = urllib.request.Request(url, headers={"User-Agent": "arena-tts-piper-assets/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            final_url = response.geturl() if hasattr(response, "geturl") else url
            if urllib.parse.urlsplit(final_url).scheme.lower() != "https":
                raise PiperAssetError(f"固定 Piper 资产下载发生 HTTPS 降级重定向：{final_url}")
            header_size = response.headers.get("Content-Length")
            if header_size is not None:
                try:
                    if int(header_size) != expected_size:
                        raise PiperAssetError(
                            f"下载响应大小声明不匹配：{destination.name}，"
                            f"期望 {expected_size}，响应 {header_size}"
                        )
                except ValueError as exc:
                    raise PiperAssetError(f"下载响应 Content-Length 无效：{header_size}") from exc
            with destination.open("xb") as out:
                while True:
                    if time.monotonic() > deadline:
                        raise PiperAssetError(
                            f"下载超过软总时限 {max_download_seconds:g}s：{url}"
                        )
                    block = response.read(_BLOCK)
                    if not block:
                        break
                    total += len(block)
                    if total > expected_size:
                        raise PiperAssetError(f"下载超过清单文件大小上限：{destination.name}")
                    digest.update(block)
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
        if total != expected_size:
            raise PiperAssetError(
                f"下载大小不匹配：{destination.name}，期望 {expected_size}，实际 {total} bytes"
            )
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise PiperAssetError(
                f"下载 SHA-256 校验失败：{destination.name}，期望 {expected_sha256}，实际 {actual}"
            )
        try:
            os.chmod(destination, 0o600)
        except OSError:
            pass
    except BaseException:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _stream_tar_member(tar, member, destination, expected_size, expected_sha256,
                       *, patch_config=False, source_size=None, source_sha256=None):
    if not member.isfile() or member.name != PurePosixPath(member.name).name:
        raise PiperAssetError(f"g2pW 归档包含非普通文件或非扁平成员：{member.name!r}")
    if member.size != (source_size if patch_config else expected_size):
        raise PiperAssetError(f"g2pW 归档成员大小不匹配：{member.name}")
    source = tar.extractfile(member)
    if source is None:
        raise PiperAssetError(f"无法读取 g2pW 归档成员：{member.name}")
    h = hashlib.sha256()
    total = 0
    if patch_config:
        raw = source.read(source_size + 1)
        if len(raw) != source_size or hashlib.sha256(raw).hexdigest() != source_sha256:
            raise PiperAssetError("g2pW config.py 原始文件 SHA-256 不匹配")
        needle = b"model_source = 'bert-base-chinese'"
        replacement = b'model_source = os.path.join(os.path.dirname(__file__), "bert-base-chinese")'
        if raw.count(needle) != 1:
            raise PiperAssetError("g2pW config.py 结构变化；拒绝应用未审核补丁")
        data = b"import os\n" + raw.replace(needle, replacement, 1)
        h.update(data)
        total = len(data)
        with destination.open("xb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
    else:
        with destination.open("xb") as out:
            while True:
                block = source.read(_BLOCK)
                if not block:
                    break
                total += len(block)
                if total > expected_size:
                    raise PiperAssetError(f"g2pW 成员超出清单大小：{member.name}")
                h.update(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
    if total != expected_size or h.hexdigest() != expected_sha256:
        destination.unlink(missing_ok=True)
        raise PiperAssetError(f"g2pW 解包成员 SHA-256/大小不匹配：{member.name}")
    try:
        os.chmod(destination, 0o600)
    except OSError:
        pass


def _extract_g2pw_archive(archive_path, stage, frontend):
    files = frontend["g2pw_files"]
    expected_members = {item["archive_member"] for item in files}
    version_record = frontend.get("g2pw_version")
    if not version_record:
        raise PiperAssetError("目录缺少 g2pW archive version 文件的固定 SHA-256")
    expected_members.add(version_record["archive_member"])
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            names = [m.name for m in members]
            if len(names) != len(set(names)) or set(names) != expected_members:
                raise PiperAssetError(
                    f"g2pW 归档成员集合与固定清单不符：{sorted(names)}"
                )
            by_name = {m.name: m for m in members}
            for item in files:
                _stream_tar_member(
                    archive, by_name[item["archive_member"]], stage / item["filename"],
                    item["size"], item["sha256"],
                    patch_config=(item["name"] == "g2pw_config"),
                    source_size=item.get("source_size"), source_sha256=item.get("source_sha256"),
                )
            _stream_tar_member(
                archive, by_name[version_record["archive_member"]],
                stage / version_record["filename"], version_record["size"], version_record["sha256"],
            )
    except (tarfile.TarError, OSError) as exc:
        if isinstance(exc, PiperAssetError):
            raise
        raise PiperAssetError(f"安全读取固定 g2pW 归档失败：{exc}") from exc


def _copy_direct_asset(repo_id, revision, item, stage, *, dataset=False, raw_github=False,
                       timeout=45.0, max_download_seconds=1200.0, deadline=None):
    url = _source_url(repo_id, revision, item["path"], dataset=dataset, raw_github=raw_github)
    destination = stage.joinpath(*PurePosixPath(item["filename"]).parts)
    _download_file(url, destination, item["size"], item["sha256"],
                   timeout=timeout, max_download_seconds=max_download_seconds, deadline=deadline)


def _fsync_dir(path):
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _read_local_json(path, max_bytes=1024 * 1024):
    """Read bounded, non-symlink JSON owned by the local cache."""
    path = Path(path)
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            return None
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as source:
            opened = os.fstat(source.fileno())
            if (not stat.S_ISREG(opened.st_mode) or opened.st_size > max_bytes
                    or (before.st_dev, before.st_ino, before.st_size)
                    != (opened.st_dev, opened.st_ino, opened.st_size)):
                return None
            raw = source.read(max_bytes + 1)
            if len(raw) > max_bytes:
                return None
        after = path.lstat()
        if (after.st_dev, after.st_ino, after.st_size) != (
                opened.st_dev, opened.st_ino, opened.st_size):
            return None
        value = json.loads(raw.decode("utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _write_json_atomic(path, data):
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    except OSError as exc:
        raise PiperAssetError(f"无法准备 Piper 本地记录文件 {path}: {exc}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(data, out, ensure_ascii=False, indent=2, sort_keys=True)
            out.flush()
            os.fsync(out.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except OSError as exc:
        raise PiperAssetError(f"无法原子写入 Piper 本地记录文件 {path}: {exc}") from exc
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass


def _ensure_child_directories(root, *parts):
    """Create/check cache-owned directory components without following symlinks."""
    root = Path(root)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PiperAssetError(f"无法创建 Piper 缓存目录 {root}: {exc}") from exc
    if root.is_symlink() or not root.is_dir():
        raise PiperAssetError(f"Piper 缓存根目录不是普通目录：{root}")
    current = root
    for part in parts:
        if not isinstance(part, str) or part in ("", ".", "..") or "/" in part or "\\" in part:
            raise PiperAssetError(f"Piper 缓存目录组件无效：{part!r}")
        current = current / part
        if current.is_symlink():
            raise PiperAssetError(f"Piper 缓存目录组件不能是符号链接：{current}")
        try:
            current.mkdir(exist_ok=True)
        except OSError as exc:
            raise PiperAssetError(f"无法创建 Piper 缓存子目录 {current}: {exc}") from exc
        if current.is_symlink() or not current.is_dir():
            raise PiperAssetError(f"Piper 缓存目录组件不是普通目录：{current}")
    return current


@contextmanager
def _cache_lock(path, timeout_seconds=120.0):
    """Cross-process advisory lock; lock file remains harmlessly after unlock."""
    path = Path(path)
    _ensure_child_directories(path.parent.parent, path.parent.name)
    if path.is_symlink():
        raise PiperAssetError(f"缓存锁不能是符号链接：{path}")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise PiperAssetError(f"无法安全打开 Piper 缓存锁 {path}: {exc}") from exc
    handle = os.fdopen(fd, "r+b")
    deadline = time.monotonic() + timeout_seconds
    locked = False
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            while time.monotonic() <= deadline:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    locked = True
                    break
                except OSError:
                    time.sleep(0.1)
        else:
            import fcntl
            while time.monotonic() <= deadline:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                    break
                except BlockingIOError:
                    time.sleep(0.1)
        if not locked:
            raise PiperAssetError(f"等待 Piper 缓存锁超时（>{timeout_seconds:g}s）：{path}")
        yield
    finally:
        if locked:
            try:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


def _regular_cached_file(root, path, expected_size):
    """Check a cached file without accepting symlinked directory components."""
    try:
        relative = Path(path).relative_to(root)
        root_stat = Path(root).lstat()
        if not stat.S_ISDIR(root_stat.st_mode):
            return False
        current = Path(root)
        for part in relative.parts[:-1]:
            current = current / part
            if not stat.S_ISDIR(current.lstat().st_mode):
                return False
        st = (current / relative.parts[-1]).lstat()
    except (OSError, ValueError):
        return False
    return stat.S_ISREG(st.st_mode) and st.st_size == expected_size


def _license_snapshots_intact(voice_dir, sources):
    """SHA-256-check the tiny cached license evidence on every cache reuse."""
    for item in sources:
        path = Path(voice_dir).joinpath(*PurePosixPath(item["filename"]).parts)
        if not _regular_cached_file(voice_dir, path, item["size"]):
            return False
        digest = hashlib.sha256()
        total = 0
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
            with os.fdopen(fd, "rb") as source:
                opened = os.fstat(source.fileno())
                if not stat.S_ISREG(opened.st_mode) or opened.st_size != item["size"]:
                    return False
                while True:
                    block = source.read(_BLOCK)
                    if not block:
                        break
                    total += len(block)
                    if total > item["size"]:
                        return False
                    digest.update(block)
            if (total != item["size"] or digest.hexdigest() != item["sha256"]
                    or not _regular_cached_file(voice_dir, path, item["size"])):
                return False
        except OSError:
            return False
    return True


def _is_expected_size_cache(voice_dir, records):
    if voice_dir.is_symlink() or not voice_dir.is_dir():
        return False
    for record in records.values():
        if not _regular_cached_file(voice_dir, record["path"], record["size"]):
            return False
    return True


def _recover_interrupted_install(voice_id, spec, voice_dir):
    """Recover a killed install and remove abandoned staging data under the voice lock."""
    parent = voice_dir.parent
    def _backup_mtime(path):
        try:
            return path.lstat().st_mtime_ns
        except OSError:
            return 0
    backups = sorted(parent.glob(f".{voice_dir.name}.previous-*"),
                     key=_backup_mtime, reverse=True)
    if voice_dir.is_symlink():
        raise PiperAssetError(f"Piper 语音缓存目录不能是符号链接：{voice_dir}")
    if not voice_dir.exists():
        for backup in backups:
            if backup.is_symlink() or not backup.is_dir():
                continue
            backup_records = _asset_records(spec, backup)
            if _is_expected_size_cache(backup, backup_records):
                os.replace(backup, voice_dir)
                _fsync_dir(parent)
                break
    for backup in backups:
        if backup.exists() and backup != voice_dir:
            if backup.is_symlink():
                backup.unlink(missing_ok=True)
            elif backup.is_dir():
                shutil.rmtree(backup, ignore_errors=True)
    for stage in parent.glob(f".{voice_id}.staging-*"):
        if stage.is_symlink():
            stage.unlink(missing_ok=True)
        elif stage.is_dir():
            shutil.rmtree(stage, ignore_errors=True)


def _acceptance_record(root, voice_id, spec):
    policy_id = spec["license"]["policy_id"]
    safe_policy = "".join(c for c in policy_id if c.isalnum() or c in "-_")
    path = root / "acceptances" / f"{voice_id}-{spec['revision']}-{safe_policy}.json"
    data = _read_local_json(path)
    if data is None:
        return path, None
    expected = {
        "voice_id": voice_id,
        "revision": spec["revision"],
        "policy_id": policy_id,
        "model_sha256": spec["model"]["sha256"],
        "config_sha256": spec["config"]["sha256"],
        "license_sources_sha256": {x["filename"]: x["sha256"]
                                   for x in spec["license"]["license_sources"]},
    }
    if all(data.get(k) == v for k, v in expected.items()) and isinstance(data.get("accepted_at"), str):
        return path, data
    return path, None


def _provisioning_record(voice_id, spec):
    return {
        "schema_version": 1,
        "voice_id": voice_id,
        "voice_repo": spec["repo_id"],
        "voice_revision": spec["revision"],
        "model": {"path": spec["model"]["path"], "size": spec["model"]["size"],
                  "sha256": spec["model"]["sha256"]},
        "config": {"path": spec["config"]["path"], "size": spec["config"]["size"],
                   "sha256": spec["config"]["sha256"]},
        "license": spec["license"],
        "license_sources_sha256": {x["filename"]: x["sha256"]
                                    for x in spec["license"]["license_sources"]},
        "frontend_asset_sha256": {
            item["name"]: item["sha256"]
            for item in (spec["frontend"]["g2pw_files"]
                         + spec["frontend"]["g2pw_lookup_tables"]["files"]
                         + spec["frontend"]["bert_tokenizer"]["files"])
        } | {"g2pw_version": spec["frontend"]["g2pw_version"]["sha256"]},
    }


def _build_metadata(voice_id, spec, voice_dir, acceptance, downloaded):
    return {
        "voice_id": voice_id,
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "language": spec["language"],
        "speaker_count": spec["speaker_count"],
        "model_sha256": spec["model"]["sha256"],
        "config_sha256": spec["config"]["sha256"],
        "frontend_sha256": {
            item["name"]: item["sha256"]
            for item in (spec["frontend"]["g2pw_files"]
                         + spec["frontend"]["g2pw_lookup_tables"]["files"]
                         + spec["frontend"]["bert_tokenizer"]["files"])
        } | {"g2pw_version": spec["frontend"]["g2pw_version"]["sha256"]},
        "license_sources_sha256": {x["filename"]: x["sha256"]
                                   for x in spec["license"]["license_sources"]},
        "license": spec["license"],
        "accepted_at": acceptance.get("accepted_at") if acceptance else None,
        "cache_directory": str(voice_dir),
        "downloaded_this_call": bool(downloaded),
    }


def _download_and_install(voice_id, spec, root, voice_dir, *, timeout, max_download_seconds,
                          progress=None):
    _ensure_child_directories(root, "voices", voice_id)
    if progress is not None:
        progress(
            f"首次获取 {voice_id} 固定资产：ONNX {spec['model']['size']:,} bytes，"
            f"g2pW 压缩包 {spec['frontend']['g2pw_archive']['size']:,} bytes，"
            "另含中文查词表/tokenizer；文件会校验 SHA-256 后原子缓存。"
        )
    if voice_dir.is_symlink():
        raise PiperAssetError(f"Piper 语音缓存目录不能是符号链接：{voice_dir}")
    try:
        stage = Path(tempfile.mkdtemp(prefix=f".{voice_id}.staging-", dir=voice_dir.parent))
    except OSError as exc:
        raise PiperAssetError(f"无法创建 Piper 原子安装暂存目录：{exc}") from exc
    archive_path = stage / ".g2pw.tar.gz.part"
    download_deadline = time.monotonic() + max_download_seconds
    try:
        base = f"https://huggingface.co/{spec['repo_id']}/resolve/{spec['revision']}/"
        for key in ("model", "config"):
            item = spec[key]
            _download_file(base + urllib.parse.quote(item["path"], safe="/"),
                           stage / item["filename"], item["size"], item["sha256"],
                           timeout=timeout, max_download_seconds=max_download_seconds,
                           deadline=download_deadline)
        for source in spec["license"]["license_sources"]:
            destination = stage.joinpath(*PurePosixPath(source["filename"]).parts)
            _download_file(source["url"], destination, source["size"], source["sha256"],
                           timeout=timeout, max_download_seconds=max_download_seconds,
                           deadline=download_deadline)

        frontend = spec["frontend"]
        archive = frontend["g2pw_archive"]
        archive_url = _source_url(archive["repo_id"], archive["revision"], archive["path"], dataset=True)
        _download_file(archive_url, archive_path, archive["size"], archive["sha256"],
                       timeout=timeout, max_download_seconds=max_download_seconds,
                       deadline=download_deadline)
        _extract_g2pw_archive(archive_path, stage, frontend)
        archive_path.unlink(missing_ok=True)

        lookup = frontend["g2pw_lookup_tables"]
        for item in lookup["files"]:
            _copy_direct_asset(lookup["repo_id"], lookup["revision"], item, stage,
                               raw_github=True, timeout=timeout,
                               max_download_seconds=max_download_seconds, deadline=download_deadline)

        tokenizer = frontend["bert_tokenizer"]
        for item in tokenizer["files"]:
            _copy_direct_asset(tokenizer["repo_id"], tokenizer["revision"], item, stage,
                               timeout=timeout, max_download_seconds=max_download_seconds,
                               deadline=download_deadline)

        _write_json_atomic(stage / "provisioning.json", _provisioning_record(voice_id, spec))
        _fsync_dir(stage)

        backup = None
        if voice_dir.exists():
            if not voice_dir.is_dir() or voice_dir.is_symlink():
                raise PiperAssetError(f"缓存目标不是普通目录：{voice_dir}")
            backup = voice_dir.parent / f".{voice_dir.name}.previous-{uuid.uuid4().hex}"
            os.replace(voice_dir, backup)
        try:
            os.replace(stage, voice_dir)
            _fsync_dir(voice_dir.parent)
        except BaseException:
            if backup is not None and backup.exists() and not voice_dir.exists():
                os.replace(backup, voice_dir)
                _fsync_dir(voice_dir.parent)
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
        return True
    except PiperAssetError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError, tarfile.TarError) as exc:
        raise PiperAssetError(
            f"Piper 固定资产下载/安装失败：{type(exc).__name__}: {exc}。"
            "缓存中的旧版本未被覆盖；不会跳过段落，也不会静默切换引擎。"
        ) from exc
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def ensure_piper_voice(voice_id, *, cache_dir=None, accept_model_license=False,
                       force_download=False, timeout=45.0, max_download_seconds=1200.0,
                       lock_timeout=120.0, progress=None):
    """Ensure a pinned voice bundle exists; return expected hashes for PiperEngine to bind.

    The first use of each license policy requires an explicit caller acknowledgement. The
    acceptance marker is local, version/license-bound, and never sent over the network.
    """
    for label, value in (("network timeout", timeout),
                         ("download total timeout", max_download_seconds),
                         ("lock timeout", lock_timeout)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise PiperAssetError(f"{label} 必须是大于 0 的有限秒数")
    spec = _voice_spec(voice_id)
    root = _cache_root(cache_dir)
    voice_dir = root / "voices" / voice_id / spec["revision"]
    if voice_dir.is_symlink():
        raise PiperAssetError(f"Piper 语音缓存目录不能是符号链接：{voice_dir}")
    acceptance_path, acceptance = _acceptance_record(root, voice_id, spec)
    if acceptance is None:
        if not accept_model_license:
            license_note = spec["license"].get("weight_license_note", "请先审阅固定模型来源的许可信息。")
            sources = "\n".join(f"  - {x['name']}: {x['url']}" for x in spec["license"].get("license_sources", []))
            raise PiperAssetError(
                f"首次使用 Piper 音色 {voice_id} 前需审阅并显式确认许可：\n{license_note}\n"
                f"来源：\n{sources}\n确认后仅需首次添加 --accept-model-license。"
            )
        _ensure_child_directories(root, "acceptances")
        from datetime import datetime, timezone
        acceptance = {
            "voice_id": voice_id,
            "revision": spec["revision"],
            "policy_id": spec["license"]["policy_id"],
            "model_sha256": spec["model"]["sha256"],
            "config_sha256": spec["config"]["sha256"],
            "license_sources_sha256": {x["filename"]: x["sha256"]
                                       for x in spec["license"]["license_sources"]},
            "accepted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "license_summary": spec["license"],
        }
        _write_json_atomic(acceptance_path, acceptance)

    _ensure_child_directories(root, "voices", voice_id)
    records = _asset_records(spec, voice_dir)
    lock_path = root / ".locks" / f"{voice_id}-{spec['revision']}.lock"
    with _cache_lock(lock_path, timeout_seconds=lock_timeout):
        try:
            _recover_interrupted_install(voice_id, spec, voice_dir)
        except OSError as exc:
            raise PiperAssetError(f"无法恢复/清理 Piper 缓存暂存目录：{exc}") from exc
        records = _asset_records(spec, voice_dir)
        downloaded = False
        cache_ok = (_is_expected_size_cache(voice_dir, records)
                    and _license_snapshots_intact(
                        voice_dir, spec["license"]["license_sources"]))
        if force_download or not cache_ok:
            if os.environ.get("TTS_PIPER_OFFLINE", "").strip().lower() in ("1", "true", "yes"):
                raise PiperAssetError(
                    f"Piper 音色 {voice_id} 的模型/前端资产或许可快照不在有效缓存中；已启用离线模式，"
                    "没有尝试其他引擎。联网后重试，或清楚地安装受支持的固定资产。"
                )
            downloaded = _download_and_install(
                voice_id, spec, root, voice_dir, timeout=timeout,
                max_download_seconds=max_download_seconds, progress=progress,
            )
            records = _asset_records(spec, voice_dir)
            if (not _is_expected_size_cache(voice_dir, records)
                    or not _license_snapshots_intact(
                        voice_dir, spec["license"]["license_sources"])):
                raise PiperAssetError("原子安装后的 Piper 资产或许可快照缺失/完整性不符")

        metadata = _build_metadata(voice_id, spec, voice_dir, acceptance, downloaded)
        # The registry remains authoritative; cache provenance is a regenerable audit note.
        provenance_path = voice_dir / "provisioning.json"
        expected_provenance = _provisioning_record(voice_id, spec)
        actual_provenance = _read_local_json(provenance_path)
        if actual_provenance != expected_provenance:
            _write_json_atomic(provenance_path, expected_provenance)

        additional = {k: v["path"] for k, v in records.items() if k not in ("model", "config")}
        expected = {k: v["sha256"] for k, v in records.items()}
        return PiperVoiceAssets(
            voice_id=voice_id,
            model_path=records["model"]["path"],
            config_path=records["config"]["path"],
            additional_asset_paths=additional,
            expected_sha256=expected,
            metadata=metadata,
            downloaded=downloaded,
            cache_root=root,
            voice_dir=voice_dir,
        )

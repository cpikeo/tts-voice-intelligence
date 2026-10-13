"""Thin local engine adapters. No model distribution or automatic clone downloads."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import hashlib
import hmac
import io
import json
import os
import pickletools
from pathlib import Path
import re
import signal
import stat
import threading
import zipfile

import numpy as np


class EngineError(RuntimeError):
    """引擎不可用或合成失败。消息需可直接给用户诊断。"""


class EngineTimeoutError(EngineError):
    """引擎超时；是否能硬终止由适配器和原生调用边界决定。"""


DEFAULT_TIMEOUT = 60.0

_FORBIDDEN_PICKLE_MODULES = {
    "os", "posix", "nt", "subprocess", "sys", "builtins", "__builtin__",
    "socket", "shutil", "importlib", "runpy", "ctypes", "pty", "code",
    "marshal", "urllib", "http", "ftplib", "telnetlib", "webbrowser", "pip",
}
_FORBIDDEN_PICKLE_NAMES = {
    "system", "popen", "spawn", "exec", "eval", "compile", "open",
    "getattr", "setattr", "delattr", "__import__", "Popen", "call", "check_output",
}
_TEMPLATE_PLACEHOLDERS = {
    "REPLACE-WITH-PRIVATE-RECORD-ID",
    "PSEUDONYMOUS-VOICE-OWNER-ID",
    "PRIVATE-VERIFIABLE-CONSENT-REFERENCE",
    "REPLACE-AFTER-REVIEW",
    "REPLACE-AFTER-LICENSE-REVIEW",
    "REPLACE-WITH-VERIFIED-LICENSE-AND-SCOPE",
}


def language_tag(tag):
    return tag.replace("_", "-").lower()


def _validate_timeout(timeout):
    if timeout is None:
        return DEFAULT_TIMEOUT
    if not isinstance(timeout, (int, float)) or not np.isfinite(timeout) or timeout <= 0:
        raise EngineError("timeout 必须为大于 0 的有限秒数")
    return float(timeout)


def run_with_timeout(func, timeout, *args, **kwargs):
    """Apply a wall-clock deadline; native code may defer signal handling and outlive it."""
    limit = _validate_timeout(timeout)
    if hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread():
        def _alarm_handler(_signum, _frame):
            raise EngineTimeoutError(f"引擎合成超时（>{limit:.2f}s）")

        old_handler = signal.getsignal(signal.SIGALRM)
        old_timer = signal.getitimer(signal.ITIMER_REAL)
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, limit)
        try:
            return func(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)
            if old_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])

    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(func, *args, **kwargs)
    try:
        return fut.result(timeout=limit)
    except FutureTimeoutError as e:
        fut.cancel()
        raise EngineTimeoutError(f"引擎合成超时（>{limit:.2f}s）") from e
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


_STAT_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")


def _stat_signature(st):
    return tuple(int(getattr(st, name)) for name in _STAT_FIELDS)


def _fingerprint_stat(st):
    return {
        "dev": int(st.st_dev),
        "ino": int(st.st_ino),
        "bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "ctime_ns": int(st.st_ctime_ns),
    }


def _open_bound_file(path):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    return os.open(path, flags)


def _path_stat(path):
    try:
        st = os.stat(path, follow_symlinks=False)
    except OSError as e:
        raise EngineError(f"绑定资产已丢失或无法 stat：{path}: {e}") from e
    if not stat.S_ISREG(st.st_mode):
        raise EngineError(f"绑定资产不是普通文件或变成符号链接：{path}")
    return st


def _read_and_fingerprint(path, max_bytes=2 * 1024 * 1024 * 1024):
    """Read once from a single descriptor, hash those exact bytes, and bind its stat identity."""
    p = Path(path)
    if not p.is_file():
        raise EngineError(f"资产文件不存在：{p}")
    try:
        fd = _open_bound_file(p)
    except OSError as e:
        raise EngineError(f"打开资产文件失败：{p}: {e}") from e
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or _stat_signature(st) != _stat_signature(_path_stat(p)):
            raise EngineError(f"资产路径不是普通文件或在读取期间被替换：{p}")
        if st.st_size > max_bytes:
            raise EngineError(f"资产文件超过大小上限（{st.st_size} > {max_bytes} bytes）：{p}")
        chunks = []
        h = hashlib.sha256()
        total = 0
        while True:
            buf = os.read(fd, 1 << 20)
            if not buf:
                break
            total += len(buf)
            if total > max_bytes:
                raise EngineError(f"资产文件读取中超出大小上限：{p}")
            chunks.append(buf)
            h.update(buf)
        after = os.fstat(fd)
        path_after = _path_stat(p)
        if (_stat_signature(st) != _stat_signature(after)
                or _stat_signature(after) != _stat_signature(path_after)
                or total != after.st_size):
            raise EngineError(f"资产文件在读取期间被变更或替换：{p}")
        raw = b"".join(chunks)
        fp = {"path": str(p), **_fingerprint_stat(after), "sha256": h.hexdigest()}
        return raw, fp
    finally:
        os.close(fd)


def verify_bound_asset(fp):
    """Recompute SHA256 and verify the path still names the same stable file during the read."""
    p = Path(fp["path"])
    path_before = _path_stat(p)  # Reject FIFOs/devices before open; do not risk a blocking read.
    try:
        fd = _open_bound_file(p)
    except OSError as e:
        raise EngineError(f"绑定资产已丢失或无法打开：{p}: {e}") from e
    try:
        before = os.fstat(fd)
        if (_stat_signature(before) != _stat_signature(path_before)
                or _stat_signature(before) != _stat_signature(_path_stat(p))):
            raise EngineError(f"资产路径在校验期间被替换：{p}")
        h = hashlib.sha256()
        total = 0
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            total += len(block)
            h.update(block)
        after = os.fstat(fd)
        path_after = _path_stat(p)
        if (_stat_signature(before) != _stat_signature(after)
                or _stat_signature(after) != _stat_signature(path_after)):
            raise EngineError(f"资产在校验期间被变更或替换：{p}")
        if total != fp["bytes"] or h.hexdigest() != fp["sha256"]:
            raise EngineError(f"资产绑定校验失败（加载或合成期间文件被篡改/替换）：{p}")
    finally:
        os.close(fd)


def _scan_pickle_bytes(data, source_label):
    try:
        stack = []
        for op, arg, _pos in pickletools.genops(io.BytesIO(data)):
            if op.name in ("EXT1", "EXT2", "EXT4"):
                raise EngineError(f"拒绝不可信模型（含危险 pickle 扩展指令 {op.name}）：{source_label}")
            if op.name == "GLOBAL" and isinstance(arg, str):
                parts = arg.split(" ", 1)
                mod = parts[0].split(".")[0]
                name = parts[1] if len(parts) > 1 else ""
                if mod in _FORBIDDEN_PICKLE_MODULES or name in _FORBIDDEN_PICKLE_NAMES:
                    raise EngineError(f"拒绝不可信模型（检测到高危全局调用 {mod}.{name}）：{source_label}")
            elif op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE") and isinstance(arg, str):
                stack.append(arg)
                if len(stack) > 8:
                    stack.pop(0)
            elif op.name == "STACK_GLOBAL":
                if len(stack) >= 2:
                    mod_full, name = stack[-2], stack[-1]
                    mod = mod_full.split(".")[0]
                    if mod in _FORBIDDEN_PICKLE_MODULES or name in _FORBIDDEN_PICKLE_NAMES:
                        raise EngineError(f"拒绝不可信模型（检测到高危 STACK_GLOBAL {mod_full}.{name}）：{source_label}")
    except EngineError:
        raise
    except Exception as e:
        raise EngineError(f"不可信模型 pickle 结构异常，拒绝加载：{source_label}: {e}") from e


def inspect_model_safety(model_path, config_path=None, engine="piper", model_bytes=None, config_data=None):
    """Static safety gate before loading model/config into runtime."""
    if config_data is not None and isinstance(config_data, dict):
        for k, v in config_data.items():
            if isinstance(v, str) and any(tok in k.lower() for tok in ("path", "file", "vocab", "checkpoint", "model")):
                if ".." in Path(v).parts or v.startswith(("http://", "https://", "ftp://")):
                    raise EngineError(f"拒绝不安全配置路径 {k}={v!r}（禁止路径穿越或远程拉取）")
    if not model_path:
        return {"inspected": False}
    p = Path(model_path)
    raw = model_bytes if model_bytes is not None else _read_and_fingerprint(p)[0]
    if engine == "piper":
        if re.search(rb"location[\x00-\x7f]{1,32}\.\.[/\\]", raw):
            raise EngineError(f"拒绝包含外部路径穿越引用的 ONNX 模型：{p}")
        return {"inspected": True, "format": "onnx", "status": "safe"}
    if engine == "xtts":
        if zipfile.is_zipfile(io.BytesIO(raw)):
            with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
                for info in zf.infolist():
                    if ".." in Path(info.filename).parts or info.filename.startswith(("/", "\\")):
                        raise EngineError(f"拒绝包含路径穿越条目的模型归档：{info.filename}")
                    if info.filename.endswith((".pkl", ".pickle")):
                        _scan_pickle_bytes(zf.read(info.filename), f"{p}:{info.filename}")
            return {"inspected": True, "format": "torch_zip", "status": "safe"}
        if re.match(rb"^(?:\x80[\x01-\x05]|\(?c[A-Za-z0-9_.]+\r?\n[A-Za-z0-9_.]+\r?\n)", raw):
            _scan_pickle_bytes(raw, str(p))
            return {"inspected": True, "format": "pickle", "status": "safe"}
        return {"inspected": True, "format": "opaque_fixture", "status": "safe"}
    return {"inspected": True, "format": "unknown", "status": "safe"}


def canonical_authorization_payload(data):
    """Canonical JSON bytes for signing/verifying authorization authenticity."""
    fields = (
        "record_id", "subject_id", "consent_evidence", "consent", "revoked",
        "deceptive_use", "reference_reviewed", "allowed_purposes", "allowed_languages",
        "expires_at", "reference_sha256", "model_sha256", "model_license",
    )
    payload = {k: data.get(k) for k in fields if k in data}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign_authorization(data, key, key_id="default"):
    """Create an HMAC-SHA256 signature block for an authorization dictionary."""
    if isinstance(key, str):
        key = key.encode("utf-8")
    if not key:
        raise ValueError("签名密钥不能为空")
    digest = hmac.new(key, canonical_authorization_payload(data), hashlib.sha256).hexdigest()
    return {"algorithm": "hmac-sha256", "key_id": str(key_id), "value": digest}


def validate_authorization(path, reference, purpose, *, signing_key=None, require_signature=False):
    """Fail closed BEFORE loading a clone model. Binds reference audio in memory and checks signature if present/required."""
    import datetime as dt
    import soundfile as sf
    if not path or not reference or not purpose:
        raise EngineError("克隆需 --authorization、--reference-audio 与 --purpose")
    try:
        auth_bytes, auth_fp = _read_and_fingerprint(path, max_bytes=5 * 1024 * 1024)
        data = json.loads(auth_bytes.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("授权记录需为 JSON object")
        if not isinstance(data.get("allowed_purposes"), list) or not all(isinstance(v, str) and v for v in data["allowed_purposes"]):
            raise ValueError("allowed_purposes 需为用途字符串列表")
        for key in ("record_id", "subject_id", "consent_evidence", "model_license"):
            val = data.get(key)
            if not isinstance(val, str) or not val.strip() or val.strip() in _TEMPLATE_PLACEHOLDERS:
                raise ValueError(f"缺少有效 {key}（不得使用未替换模板占位符）")
        if data.get("reference_reviewed") is not True:
            raise ValueError("需人工确认参考音频为已授权单说话人干净语音")
        if data.get("consent") is not True or data.get("revoked") is not False:
            raise ValueError("未授权或已撤销")
        if data.get("deceptive_use") is not False:
            raise ValueError("不得用于欺骗性身份冒用")
        if purpose not in data.get("allowed_purposes", []):
            raise ValueError("用途不在授权范围")
        until = dt.datetime.fromisoformat(data["expires_at"])
        if until.tzinfo is None or until <= dt.datetime.now(dt.timezone.utc):
            raise ValueError("授权已过期或时区缺失")

        env_key = os.environ.get("TTS_AUTH_SIGNING_KEY")
        eff_key = signing_key if signing_key is not None else env_key
        must_verify = bool(
            require_signature
            or os.environ.get("TTS_REQUIRE_SIGNED_AUTH") == "1"
            or eff_key is not None
            or "signature" in data
        )
        verification_mode = "local_assertion_requires_human_review"
        sig_key_id = None
        if must_verify:
            sig = data.get("signature")
            if not isinstance(sig, dict) or sig.get("algorithm") != "hmac-sha256" or not isinstance(sig.get("value"), str):
                raise ValueError("缺少有效 hmac-sha256 授权签名")
            if not eff_key:
                raise ValueError("授权含签名或要求强制验签，但未提供签名密钥（TTS_AUTH_SIGNING_KEY）")
            expected_sig = sign_authorization(data, eff_key, sig.get("key_id", "default"))["value"]
            if not hmac.compare_digest(sig["value"].lower(), expected_sig.lower()):
                raise ValueError("授权签名校验失败（记录可能被篡改或伪造）")
            verification_mode = "hmac_sha256_verified"
            sig_key_id = str(sig.get("key_id", "default"))

        # Single-read reference audio validation from memory buffer (no TOCTOU between hash and decode)
        ref_bytes, ref_fp = _read_and_fingerprint(reference, max_bytes=100 * 1024 * 1024)
        digest = ref_fp["sha256"]
        if digest != data.get("reference_sha256"):
            raise ValueError("参考音频与授权哈希不符")
        info = sf.info(io.BytesIO(ref_bytes))
        if info.channels != 1 or not 3 <= info.duration <= 30 or info.samplerate < 16000:
            raise ValueError("参考音频需单声道、16kHz以上、3–30秒（启发式收录门槛）")
        x, sr = sf.read(io.BytesIO(ref_bytes), dtype="float32")
        if not np.isfinite(x).all() or np.max(np.abs(x)) >= .999:
            raise ValueError("参考音频非有限或存在削波")
        from qa import active_seconds
        active = active_seconds(x, sr)
        if active < 2 or active / info.duration < .5:
            raise ValueError("参考音频有效信号不足；该门槛不是语音/单说话人识别")
        # No identifying evidence/path in synthesis reports.
        summary = {
            "record_id": data["record_id"],
            "reference_sha256": digest,
            "authorization_sha256": auth_fp["sha256"],
            "purpose": purpose,
            "expires_at": data["expires_at"],
            "model_license": data["model_license"],
            "reference_seconds": info.duration,
            "verification": verification_mode,
        }
        if sig_key_id is not None:
            summary["signature_key_id"] = sig_key_id
        return summary
    except EngineError:
        raise
    except (OSError, KeyError, ValueError, TypeError, RuntimeError) as e:
        raise EngineError(f"克隆授权/参考检查失败：{e}") from e


# ---------------------------------------------------------------- 引擎层

class Engine:
    name = "base"
    sample_rate = 0
    timeout = DEFAULT_TIMEOUT

    def available(self):
        return True, "ok"

    def synthesize(self, text):
        raise NotImplementedError


class PiperEngine(Engine):
    """Piper（ONNX，CPU 可跑）。注意：piper-tts 包为 GPL-3.0-or-later。"""
    name = "piper"

    def __init__(self, model, config=None, speaker=None, length_scale=1.0,
                 noise_scale=0.667, noise_w_scale=0.8, use_cuda=False, download_dir=None,
                 language="auto", timeout=DEFAULT_TIMEOUT):
        self.timeout = _validate_timeout(timeout)
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
        cfg_bytes, cfg_fp = _read_and_fingerprint(cfg, max_bytes=10 * 1024 * 1024)
        model_bytes, model_fp = _read_and_fingerprint(model)
        data = json.loads(cfg_bytes.decode("utf-8"))
        safety = inspect_model_safety(model, cfg, engine="piper", model_bytes=model_bytes, config_data=data)
        locale = language_tag(data.get("language", {}).get("code", "unknown"))
        requested = language_tag(language)
        if requested != "auto" and requested != locale and requested != locale.split("-")[0]:
            raise EngineError(f"模型语言 {locale} 不支持请求地区/语言 {language}")
        self.language = locale
        count = data.get("num_speakers", 1)
        if speaker is not None and not 0 <= speaker < count:
            raise EngineError(f"speaker id 超出 0..{count - 1}")
        if not np.isfinite(length_scale) or length_scale <= 0 or any(
                not np.isfinite(n) or n < 0 for n in (noise_scale, noise_w_scale)):
            raise EngineError("无效 Piper 合成参数")
        self.bound_assets = {"model": model_fp, "config": cfg_fp, "safety": safety}
        self.capabilities = {
            "languages": [locale], "selected_language": locale,
            "voice": "model fixed", "speakers": count, "rate": True,
            "pause_override": True, "emotion": False, "emphasis": False, "timbre_shift": False,
            "clone": False, "hardware": "CPU/CUDA (GPU unverified)",
            "timeout_seconds": self.timeout, "asset_binding": True, "model_safety_check": True,
            "dependencies": ["piper-tts", "onnxruntime"],
            "software_license": "GPL-3.0-or-later", "model_license": "user must verify model card",
            "resources": "model/config required; zh frontend may fetch resources",
        }
        self.config_path = str(cfg)
        # 中文模型的 g2pW 资源默认会下载到 CWD；显式指定为模型同目录，避免污染工作区
        self.voice = run_with_timeout(
            PiperVoice.load, self.timeout, str(model), str(cfg),
            use_cuda=use_cuda, download_dir=str(download_dir or model.parent),
        )
        verify_bound_asset(model_fp)
        verify_bound_asset(cfg_fp)
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

    def verify_integrity(self):
        """Full SHA256 checkpoint used after synthesis (and after direct capture batches)."""
        if hasattr(self, "bound_assets"):
            verify_bound_asset(self.bound_assets["model"])
            verify_bound_asset(self.bound_assets["config"])

    def _synthesize_loaded(self, text):
        """One segment from the already loaded, immutable in-memory Piper session.

        The caller must verify the bound model/config before and after the complete batch. This
        avoids re-reading large model files for every sentence without weakening batch integrity.
        """
        try:
            def _do_synth():
                chunks = list(self.voice.synthesize(text, syn_config=self.syn))
                if not chunks:
                    raise EngineError("Piper 返回空结果")
                return np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32, copy=False)

            return run_with_timeout(_do_synth, getattr(self, "timeout", DEFAULT_TIMEOUT))
        except EngineError:
            raise
        except Exception as e:
            raise EngineError(f"Piper 合成失败：{e}") from e

    def synthesize(self, text):
        """Direct one-segment API retains full pre/post asset SHA256 checks."""
        self.verify_integrity()
        audio = self._synthesize_loaded(text)
        self.verify_integrity()
        return audio


class EspeakEngine(Engine):
    """Native executable; explicit locale only, no shell, process-group hard timeout."""
    name = "espeak"
    sample_rate = 22050

    def __init__(self, language="en", length_scale=1., timeout=DEFAULT_TIMEOUT):
        import shutil
        import subprocess
        self.timeout = _validate_timeout(timeout)
        self.binary = shutil.which("espeak-ng")
        if not self.binary:
            raise EngineError("未安装 espeak-ng（系统包，不是 pip 包）")
        if language == "auto":
            raise EngineError("eSpeak 需显式 --language（例如 en-us / cmn / de）")
        voices = subprocess.run([self.binary, "--voices"], check=True, capture_output=True,
                                text=True, timeout=min(15.0, self.timeout)).stdout
        languages = [line.split()[1] for line in voices.splitlines()[1:] if len(line.split()) >= 4]
        aliases = {"zh": "cmn", "zh-cn": "cmn", "en": "en-gb"}
        self.language = aliases.get(language_tag(language), language_tag(language))
        if self.language not in languages:
            raise EngineError(f"eSpeak 未安装语言/地区 {language}；用 espeak-ng --voices 查询")
        if not np.isfinite(length_scale) or not .5 <= length_scale <= 2:
            raise EngineError("length_scale 必须在 0.5–2")
        self.speed = round(175 / length_scale)
        self.capabilities = {
            "languages": sorted(set(languages)), "selected_language": self.language,
            "voice": "locale default", "speakers": False, "rate": True,
            "pause_override": True, "emotion": False, "emphasis": False, "timbre_shift": False,
            "clone": False, "hardware": "CPU", "timeout_seconds": self.timeout,
            "dependencies": ["espeak-ng"], "software_license": "GPL-3.0-or-later",
            "model_license": "bundled voice data: verify distribution",
            "resources": f"no neural weights; executable hard timeout {self.timeout:g}s",
        }

    def synthesize(self, text):
        import subprocess
        import soundfile as sf
        limit = _validate_timeout(getattr(self, "timeout", DEFAULT_TIMEOUT))
        proc = subprocess.Popen(
            [self.binary, "--stdout", "--stdin", "-v", self.language, "-s", str(self.speed)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(input=text.encode("utf-8"), timeout=limit)
            if proc.returncode != 0:
                err_msg = stderr.decode("utf-8", errors="replace").strip()
                raise EngineError(f"eSpeak 进程退出码 {proc.returncode}: {err_msg}")
            x, sr = sf.read(io.BytesIO(stdout), dtype="float32")
            if sr != self.sample_rate:
                raise EngineError(f"eSpeak 输出采样率 {sr} 与期望 {self.sample_rate} 不符")
            return x
        except subprocess.TimeoutExpired as e:
            self._terminate_proc(proc)
            raise EngineTimeoutError(f"eSpeak 合成超时（>{limit:.2f}s）") from e
        except EngineError:
            self._terminate_proc(proc)
            raise
        except (subprocess.SubprocessError, ValueError, RuntimeError, OSError) as e:
            self._terminate_proc(proc)
            raise EngineError(f"eSpeak 合成失败：{e}") from e
        except BaseException:  # 取消/中断：仍须清理整个进程组
            self._terminate_proc(proc)
            raise

    @staticmethod
    def _terminate_proc(proc):
        if proc.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                pass


class XTTSEngine(Engine):
    """Optional coqui-tts API, local XTTS checkpoint only. Never download weights."""
    name = "xtts"

    def __init__(self, model, config, language, reference, authorization, purpose, cuda=False,
                 timeout=DEFAULT_TIMEOUT, signing_key=None, require_signature=False):
        self.timeout = _validate_timeout(timeout)
        self.signing_key = signing_key
        self.require_signature = require_signature
        self.authorization = validate_authorization(
            authorization, reference, purpose,
            signing_key=signing_key, require_signature=require_signature,
        )
        if not model or not config or not Path(model).is_file() or not Path(config).is_file():
            raise EngineError("XTTS 需本地 checkpoint 与 --config；不自动下载模型")
        auth_bytes, auth_fp = _read_and_fingerprint(authorization, max_bytes=5 * 1024 * 1024)
        grant = json.loads(auth_bytes.decode("utf-8"))
        model_bytes, model_fp = _read_and_fingerprint(model)
        digest = model_fp["sha256"]
        if digest != grant.get("model_sha256"):
            raise EngineError("XTTS 模型不匹配授权 model_sha256")
        self.authorization["model_sha256"] = digest
        cfg_bytes, cfg_fp = _read_and_fingerprint(config, max_bytes=10 * 1024 * 1024)
        data = json.loads(cfg_bytes.decode("utf-8"))
        if data.get("model") != "xtts":
            raise EngineError("只接受 XTTS 配置")
        safety = inspect_model_safety(model, config, engine="xtts", model_bytes=model_bytes, config_data=data)
        _, ref_fp = _read_and_fingerprint(reference, max_bytes=100 * 1024 * 1024)
        self.language = language_tag(language)
        languages = data.get("languages", [])
        if not isinstance(languages, list) or not isinstance(grant.get("allowed_languages"), list):
            raise EngineError("配置和授权语言需为列表")
        # No regional fallback: zh-cn is a genuine XTTS locale; en-GB is not en.
        if self.language not in languages or self.language not in grant.get("allowed_languages", []):
            raise EngineError(f"XTTS 配置不支持地区/语言 {language}：{languages}")
        try:
            from TTS.api import TTS
            self.api = run_with_timeout(
                TTS, self.timeout, model_path=str(model), config_path=str(config),
                gpu=cuda, progress_bar=False,
            )
            self.sample_rate = int(self.api.synthesizer.output_sample_rate)
        except EngineError:
            raise
        except Exception as e:
            raise EngineError(f"XTTS 加载失败；需 coqui-tts、torch 及完整本地资源：{e}") from e
        # Verify no TOCTOU mutation occurred during model loading
        verify_bound_asset(model_fp)
        verify_bound_asset(cfg_fp)
        verify_bound_asset(ref_fp)
        self.bound_assets = {
            "model": model_fp, "config": cfg_fp,
            "reference": ref_fp, "authorization": auth_fp,
            "safety": safety,
        }
        self.reference = str(reference)
        self.authorization_path = str(authorization)
        self.purpose = purpose
        self.config_path = str(config)
        self.capabilities = {
            "languages": sorted(set(languages)), "selected_language": self.language,
            "voice": "authorized reference", "speakers": False, "rate": False,
            "pause_override": True, "emotion": False, "emphasis": False, "timbre_shift": False,
            "clone": True, "hardware": "CPU/CUDA (unverified)",
            "timeout_seconds": self.timeout, "asset_binding": True, "model_safety_check": True,
            "dependencies": ["coqui-tts", "torch"], "software_license": "MPL-2.0 (verify package)",
            "model_license": self.authorization["model_license"],
            "resources": "user-supplied local assets; memory/runtime unverified",
        }

    def synthesize(self, text):
        # Recheck revocation, expiry, signature, and file binding for each segment.
        validate_authorization(
            self.authorization_path, self.reference, self.purpose,
            signing_key=getattr(self, "signing_key", None),
            require_signature=getattr(self, "require_signature", False),
        )
        grant = json.loads(Path(self.authorization_path).read_text(encoding="utf-8"))
        if self.language not in grant.get("allowed_languages", []) or grant.get("model_sha256") != self.authorization["model_sha256"]:
            raise EngineError("克隆模型/语言授权已变更")
        if hasattr(self, "bound_assets"):
            verify_bound_asset(self.bound_assets["model"])
            verify_bound_asset(self.bound_assets["config"])
            verify_bound_asset(self.bound_assets["reference"])
        try:
            out = run_with_timeout(
                self.api.tts, getattr(self, "timeout", DEFAULT_TIMEOUT),
                text=text, language=self.language, speaker_wav=self.reference, split_sentences=False,
            )
            if hasattr(self, "bound_assets"):
                verify_bound_asset(self.bound_assets["reference"])
            return np.asarray(out, dtype=np.float32)
        except EngineError:
            raise
        except Exception as e:
            raise EngineError(f"XTTS 合成失败：{e}") from e

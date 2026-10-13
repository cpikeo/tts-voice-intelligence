"""Thin local engine adapters. No model distribution or automatic clone downloads."""
import json
from pathlib import Path
import numpy as np

class EngineError(RuntimeError):
    """引擎不可用或合成失败。消息需可直接给用户诊断。"""


# ---------------------------------------------------------------- 引擎层

class Engine:
    name = "base"
    sample_rate = 0

    def available(self):
        return True, "ok"

    def synthesize(self, text):
        raise NotImplementedError


class PiperEngine(Engine):
    """Piper（ONNX，CPU 可跑）。注意：piper-tts 包为 GPL-3.0-or-later。"""
    name = "piper"

    def __init__(self, model, config=None, speaker=None, length_scale=1.0,
                 noise_scale=0.667, noise_w_scale=0.8, use_cuda=False, download_dir=None, language="auto"):
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
        data = json.loads(cfg.read_text(encoding="utf-8"))
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
        self.capabilities = {"languages": [locale], "selected_language": locale,
                             "voice": "model fixed", "speakers": count, "rate": True,
                             "emotion": False, "clone": False, "hardware": "CPU/CUDA (GPU unverified)",
                             "dependencies": ["piper-tts", "onnxruntime"],
                             "software_license": "GPL-3.0-or-later", "model_license": "user must verify model card",
                             "resources": "model/config required; zh frontend may fetch resources"}
        self.config_path = str(cfg)
        # 中文模型的 g2pW 资源默认会下载到 CWD；显式指定为模型同目录，避免污染工作区
        self.voice = PiperVoice.load(str(model), str(cfg), use_cuda=use_cuda,
                                     download_dir=str(download_dir or model.parent))
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

    def synthesize(self, text):
        try:
            chunks = list(self.voice.synthesize(text, syn_config=self.syn))
            if not chunks:
                raise EngineError("Piper 返回空结果")
            return np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        except EngineError:
            raise
        except Exception as e:
            raise EngineError(f"Piper 合成失败：{e}") from e



def language_tag(tag):
    return tag.replace('_', '-').lower()


def validate_authorization(path, reference, purpose):
    """Fail closed BEFORE loading a clone model. Local assertion, not legal attestation."""
    import datetime as dt
    import hashlib
    import soundfile as sf
    if not path or not reference or not purpose:
        raise EngineError("克隆需 --authorization、--reference-audio 与 --purpose")
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("授权记录需为 JSON object")
        if not isinstance(data.get("allowed_purposes"), list) or not all(isinstance(v, str) and v for v in data["allowed_purposes"]):
            raise ValueError("allowed_purposes 需为用途字符串列表")
        for key in ("record_id", "subject_id", "consent_evidence", "model_license"):
            if not isinstance(data.get(key), str) or not data[key].strip():
                raise ValueError(f"缺少 {key}")
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
        ref = Path(reference)
        if ref.stat().st_size > 100 * 1024 * 1024:
            raise ValueError("参考文件超过 100 MiB")
        digest = hashlib.sha256(ref.read_bytes()).hexdigest()
        if digest != data.get("reference_sha256"):
            raise ValueError("参考音频与授权哈希不符")
        info = sf.info(str(ref))
        if info.channels != 1 or not 3 <= info.duration <= 30 or info.samplerate < 16000:
            raise ValueError("参考音频需单声道、16kHz以上、3–30秒（启发式收录门槛）")
        x, sr = sf.read(str(ref), dtype="float32")
        if not np.isfinite(x).all() or np.max(np.abs(x)) >= .999:
            raise ValueError("参考音频非有限或存在削波")
        from qa import active_seconds
        active = active_seconds(x, sr)
        if active < 2 or active / info.duration < .5:
            raise ValueError("参考音频有效信号不足；该门槛不是语音/单说话人识别")
        # No identifying evidence/path in synthesis reports.
        return {"record_id": data["record_id"], "reference_sha256": digest,
                "purpose": purpose, "expires_at": data["expires_at"],
                "model_license": data["model_license"], "reference_seconds": info.duration,
                "verification": "local_assertion_requires_human_review"}
    except (OSError, KeyError, ValueError, TypeError, RuntimeError) as e:
        raise EngineError(f"克隆授权/参考检查失败：{e}") from e


class EspeakEngine(Engine):
    """Native executable; explicit locale only, no shell and no network."""
    name = "espeak"
    sample_rate = 22050

    def __init__(self, language="en", length_scale=1.):
        import shutil
        import subprocess
        self.binary = shutil.which("espeak-ng")
        if not self.binary:
            raise EngineError("未安装 espeak-ng（系统包，不是 pip 包）")
        if language == "auto":
            raise EngineError("eSpeak 需显式 --language（例如 en-us / cmn / de）")
        voices = subprocess.run([self.binary, "--voices"], check=True, capture_output=True,
                                text=True, timeout=15).stdout
        languages = [line.split()[1] for line in voices.splitlines()[1:] if len(line.split()) >= 4]
        aliases = {"zh": "cmn", "zh-cn": "cmn", "en": "en-gb"}
        self.language = aliases.get(language_tag(language), language_tag(language))
        if self.language not in languages:
            raise EngineError(f"eSpeak 未安装语言/地区 {language}；用 espeak-ng --voices 查询")
        if not np.isfinite(length_scale) or not .5 <= length_scale <= 2:
            raise EngineError("length_scale 必须在 0.5–2")
        self.speed = round(175 / length_scale)
        self.capabilities = {"languages": sorted(set(languages)), "selected_language": self.language,
                             "voice": "locale default", "speakers": False, "rate": True,
                             "emotion": False, "clone": False, "hardware": "CPU",
                             "dependencies": ["espeak-ng"], "software_license": "GPL-3.0-or-later",
                             "model_license": "bundled voice data: verify distribution",
                             "resources": "no neural weights; executable timeout 60s"}

    def synthesize(self, text):
        import io
        import subprocess
        import soundfile as sf
        try:
            result = subprocess.run([self.binary, "--stdout", "--stdin", "-v", self.language,
                                     "-s", str(self.speed)], input=text.encode("utf-8"),
                                    capture_output=True, check=True, timeout=60)
            x, sr = sf.read(io.BytesIO(result.stdout), dtype="float32")
            if sr != self.sample_rate:
                raise ValueError(f"意外采样率 {sr}")
            return x
        except (subprocess.SubprocessError, ValueError, RuntimeError) as e:
            raise EngineError(f"eSpeak 合成失败：{e}") from e


class XTTSEngine(Engine):
    """Optional coqui-tts API, local XTTS checkpoint only. Never download weights."""
    name = "xtts"

    def __init__(self, model, config, language, reference, authorization, purpose, cuda=False):
        self.authorization = validate_authorization(authorization, reference, purpose)
        if not model or not config or not Path(model).is_file() or not Path(config).is_file():
            raise EngineError("XTTS 需本地 checkpoint 与 --config；不自动下载模型")
        import hashlib
        grant = json.loads(Path(authorization).read_text(encoding="utf-8"))
        with open(model, "rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest != grant.get("model_sha256"):
            raise EngineError("XTTS 模型不匹配授权 model_sha256")
        self.authorization["model_sha256"] = digest
        data = json.loads(Path(config).read_text(encoding="utf-8"))
        if data.get("model") != "xtts":
            raise EngineError("只接受 XTTS 配置")
        self.language = language_tag(language)
        languages = data.get("languages", [])
        if not isinstance(languages, list) or not isinstance(grant.get("allowed_languages"), list):
            raise EngineError("配置和授权语言需为列表")
        # No regional fallback: zh-cn is a genuine XTTS locale; en-GB is not en.
        if self.language not in languages or self.language not in grant.get("allowed_languages", []):
            raise EngineError(f"XTTS 配置不支持地区/语言 {language}：{languages}")
        try:
            from TTS.api import TTS
            self.api = TTS(model_path=str(model), config_path=str(config), gpu=cuda,
                           progress_bar=False)
            self.sample_rate = int(self.api.synthesizer.output_sample_rate)
        except Exception as e:
            raise EngineError(f"XTTS 加载失败；需 coqui-tts、torch 及完整本地资源：{e}") from e
        self.reference = str(reference)
        self.authorization_path = str(authorization)
        self.purpose = purpose
        self.config_path = str(config)
        self.capabilities = {"languages": sorted(set(languages)), "selected_language": self.language,
                             "voice": "authorized reference", "speakers": False, "rate": False,
                             "emotion": False, "clone": True, "hardware": "CPU/CUDA (unverified)",
                             "dependencies": ["coqui-tts", "torch"], "software_license": "MPL-2.0 (verify package)",
                             "model_license": self.authorization["model_license"],
                             "resources": "user-supplied local assets; memory/runtime unverified"}

    def synthesize(self, text):
        # Recheck revocation, expiry, file binding for each segment.
        validate_authorization(self.authorization_path, self.reference, self.purpose)
        grant = json.loads(Path(self.authorization_path).read_text(encoding="utf-8"))
        if self.language not in grant.get("allowed_languages", []) or grant.get("model_sha256") != self.authorization["model_sha256"]:
            raise EngineError("克隆模型/语言授权已变更")
        try:
            return np.asarray(self.api.tts(text=text, language=self.language,
                                          speaker_wav=self.reference, split_sentences=False), dtype=np.float32)
        except Exception as e:
            raise EngineError(f"XTTS 合成失败：{e}") from e

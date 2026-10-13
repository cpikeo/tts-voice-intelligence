# 引擎、依赖、安全隔离与能力边界

决策以 SKILL.md 为准。适配器在 `scripts/engines.py`；实例 JSON 能力比此矩阵更具体。

| 引擎 | 实现/验证 | 语言与地区 | 音色/说话人 | 语速/停顿/情绪 | 克隆 | 硬超时/绑定/隔离 | 许可 |
|---|---|---|---|---|---|---|---|
| Piper | 真实实现；本次 huayan 中文 7 类场景实测 | 模型配置 locale；裸语言匹配同根地区，不跨地区静默回退 | 固定模型；配置内 speaker id | 逐段 `length_scale` 与 `pause_after_ms`；无情绪/重音/音色接口 | 不支持 | 默认 60s 硬超时；加载与合成前后校验模型/配置哈希；ONNX 路径穿越筛查；可选 `--isolate-model` | piper-tts GPL-3.0-or-later；每模型数据/权重许可单独确认 |
| eSpeak NG | 真实子进程实现；中/英/德/日实测 | 从本机 `--voices` 读取；显式地区不回退 | 地区默认 voice | 逐段 `length_scale`（映射 wpm）与 `pause_after_ms`；无情绪接口 | 不支持 | 独立进程组（`start_new_session`）+ 默认 60s 硬超时；超时或取消时 `SIGKILL` 清理进程组 | 软件 GPL-3.0-or-later；随系统语音数据许可另核对 |
| XTTS | 本地 coqui-tts API 实现；替身契约、验签与门禁实测，真实克隆缺资产未验证 | 配置 `languages` 与授权 `allowed_languages` 交集；不降级地区 | 已授权参考音色 | 仅支持管线 `pause_after_ms`；不暴露语速、情绪或重音 | 支持调用，目标音色保持未实测 | 默认 60s 硬超时；单次内存读入消除参考音频 TOCTOU；HMAC-SHA256 授权验签；pickle/ZIP 静态安全拦截；可选 `--isolate-model` | 包通常 MPL-2.0；XTTS 权重常为 CPML，不可假定商用 |

没有默认模型排名；历史无原始产物的结论已撤回。eSpeak 的多语言实现不是专业自然度背书。仓库本身未附 LICENSE：不要擅自假设源码可按某许可证分发。

## 最小依赖与可选安装

```bash
pip install -r requirements.txt
pip install piper-tts                    # 本次 1.4.2 / onnxruntime 1.24.4
sudo apt-get install espeak-ng           # 本次 1.52.0
pip install coqui-tts                    # 本次未安装/未验证；不是基础依赖
pip install faster-whisper scipy opencc-python-reimplemented
```

Piper 用户自备 `.onnx` + 同名 `.onnx.json`。本次非商业技术测试 `zh_CN-huayan-medium` 模型卡为 dataset License: Unknown，不能据此商业交付。不要将模型权重放入仓库。

## 资产绑定、授权验签与不可信模型隔离

1. **单次读取与哈希绑定（防 TOCTOU）**：`_read_and_fingerprint` 从同一文件描述符单次读入内存并计算 SHA256，配置解析与参考音频 `soundfile` 解码直接作用于该内存缓冲；模型加载前后与每次 `synthesize` 前后通过 `verify_bound_asset` 复核磁盘文件大小与 SHA256，发生变更立即拒绝。
2. **授权真实性校验**：`validate_authorization` 拒绝模板占位符（如 `REPLACE-WITH-...`）。当传入 `--require-signed-auth`、设置 `TTS_REQUIRE_SIGNED_AUTH=1`、提供 `TTS_AUTH_SIGNING_KEY` 或授权 JSON 含 `signature` 字段时，使用 `hmac.compare_digest` 校验规范化载荷的 `hmac-sha256` 签名；未配置外部公钥基础设施时，本地签名仅防未持密钥篡改，不等同法律身份公证。
3. **不可信模型静态筛查与子进程隔离**：`inspect_model_safety` 在加载前扫描配置路径穿越/远程 URL、ONNX 外部路径穿越、ZIP 归档路径穿越（ZIP Slip），并用 `pickletools.genops` 静态反汇编排查 `.pth`/`.pt` 中的高危全局调用（`os`、`posix`、`subprocess`、`builtins.eval/exec/getattr` 等）与 `EXT*` 指令；`tts.py synth --isolate-model` 可将模型加载与合成放入独立离线子进程组（`HF_HUB_OFFLINE=1`、独立临时工作目录、超时进程组强杀）。

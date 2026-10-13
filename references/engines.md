# 引擎适配与依赖

本 Skill 的引擎层只负责：统一接口、参数转换、能力声明与结果返回。决策（选哪个引擎、用什么参数）在 `SKILL.md`。

## 1. 引擎矩阵

| 引擎 | 状态 | 运行位置 | 中文 | 许可（软件 / 模型数据） | 备注 |
|---|---|---|---|---|---|
| `piper` | **已实现并实测** | 本地 CPU（ONNX） | 有 zh_CN 模型 | 软件 **GPL-3.0-or-later**；模型见下表 | 本 Skill 唯一内置适配器 |
| 其他（如 GPT-SoVITS、云端 API） | **未实现** | — | — | — | 新增方式见第 5 节；不得在未实现时声称可用 |

GPT-SoVITS 仅作为质量对标参考，本 Skill 不依赖其仓库、代码、权重或目录结构。

## 2. Piper 安装（实测环境：Debian 13，Python 3.13，2 vCPU，无 GPU）

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install piper-tts numpy soundfile
```

- `piper-tts` 1.8.0 已验证可用（`PiperVoice.load` / `synthesize`）。
- **许可提示**：`piper-tts` 为 GPL-3.0-or-later。若你的分发场景与 GPL 不兼容，需要替换为其他引擎适配器，本 Skill 的其余部分（规范化、分段、响度、QA）不依赖 piper。
- 不需要 ffmpeg。WAV/FLAC/OGG 由 `soundfile`（libsndfile 1.2.2）写出；**不支持 MP3**（需额外安装 ffmpeg 或编码器）。

### 中文模型的额外依赖（以 `zh_CN-xiao_ya-medium` 为例）

`xiao_ya` 使用 g2pW 中文多音字前端，需要以下依赖，缺一即报错：

```bash
pip install unicode-rbnf sentence-stream transformers tokenizers
pip install --no-deps g2pw        # 仅取得查找表；--no-deps 避免拉入 torch
```

- 首次使用时 piper 会把 g2pW 资源（约 150 MB）下载到**模型所在目录**（本 Skill 已显式设置 `download_dir`，不会写入 CWD）。需要一次网络访问。
- 未装 `transformers` 会报 `No module named 'transformers'`；未装 `sentence_stream` 会报 `No module named 'sentence_stream'`。
- 仅 `transformers` 无 PyTorch 时会打印 `PyTorch was not found` 提示，属正常现象。

`huayan` 模型未使用上述前端，安装最小依赖即可运行（已验证）。

### 回听验收依赖（仅 `qa.py --asr` 需要）

```bash
pip install faster-whisper scipy opencc-python-reimplemented
```

- `scipy` 用于把音频重采样到 16 kHz；本 Skill 自行解码，**不依赖 ffmpeg 或 PyAV**（实测 faster-whisper 1.2.1 与当前 PyAV 版本不兼容，故绕开）。
- `opencc-python-reimplemented` 用于把 whisper 输出的繁体转为简体再比对；缺失时会直接比较，CER 可能虚高。
- 首次使用 `--asr small` 会从 Hugging Face 下载模型（实测缓存约 464 MB；`base` 约 142 MB），以 int8 在 CPU 上运行。

## 3. 模型获取

官方仓库：`https://huggingface.co/rhasspy/piper-voices`（中文路径 `zh/zh_CN/...`）。每个模型需要两个文件：`*.onnx` 与同名 `*.onnx.json`。

| 模型 | 采样率 | 模型数据许可（以模型卡为准） | 实测结论（见 eval-results.md） |
|---|---|---|---|
| `zh_CN-huayan-medium` | 22050 Hz | 模型卡标注 dataset License: **Unknown** | 可用；中文 CER 明显高于 xiao_ya；英文/专名差 |
| `zh_CN-xiao_ya-medium` | 22050 Hz | 模型卡标注 **Non-commercial use**（BZNSYP 数据集） | 中文 CER 显著更低；需额外依赖 |

**许可结论**：两个中文模型的数据许可都**不适合直接用于商业交付**，且 huayan 许可不明。使用前必须让用户确认授权，并在交付说明中写明。本 Skill 不内置、不分发任何模型权重。

模型放置建议：`~/.cache/tts-voice/piper/`（不进入工作区快照）。

## 4. 参数与资源

| 参数 | 默认 | 作用 | 说明 |
|---|---|---|---|
| `--length-scale` | 1.0 | 语速倍率（>1 变慢） | 已验证时长随其单调变化（集成测试） |
| `--noise-scale` | 0.667 | 发音随机度 | 未做听感对比，保持默认 |
| `--noise-w` | 0.8 | 时长随机度 | 同上 |
| `--speaker` | None | 多说话人模型的说话人 id | 单说话人模型无需设置 |
| `--cuda` | 关 | GPU 推理 | 需要 onnxruntime-gpu；**本环境无 GPU，未测试** |

`tts.py check` 会额外输出模型路径、大小与 sha256，用于跨任务复现与许可核对。

**随机性与复现（已验证）**：Piper 的采样噪声位于 ONNX 图内部，Python 层无法固定种子。默认参数下，同一文本三次合成的音频字节哈希各不相同、时长相差约 ±1.7%。若需要逐字节可复现（如回归测试、A/B 对比），使用 `--noise-scale 0 --noise-w 0`：实测三次输出完全一致，但韵律可能变平，**其听感未经人工评估，不建议直接作为成品默认值**。

运行资源（实测，2 vCPU / 2 GB，无 GPU）：RTF 约 0.06–0.15（合成耗时 / 成品音频时长）。首次调用会加载模型与 g2pW，耗时更长（实测一次 RTF 1.26–1.46，属冷启动）。

内存（实测，同一环境）：

| 场景 | 峰值 RSS | 说明 |
|---|---|---|
| huayan 合成 972 字（36 段，成品 187 s） | 551 MB | 含模型与全部音频驻留内存 |
| huayan + whisper-small 回听 | 约 1.0 GB | 顺序执行，可接受 |
| xiao_ya（g2pW 前端）+ whisper-small **同时**驻留 | **被 OOM 杀掉（exit 137）** | 2 GB 环境不够 |

因此 `tests/run_eval.py` 采用两阶段：先合成全部并落盘，再 `del engine` 释放引擎后才加载 whisper 回听。
若你的环境内存同样紧张，用 `--asr base`（约 142 MB）替代 `small`（约 464 MB）。

## 5. 新增引擎适配器（约定）

在 `scripts/tts.py` 中继承 `Engine`，实现：

- `sample_rate`：输出采样率
- `available() -> (bool, str)`：可用性与原因
- `synthesize(text) -> np.ndarray`：float32、单声道、范围 [-1, 1]；失败抛 `EngineError`，消息必须可诊断

然后在 `build_engine()` 中注册。流水线的规范化、分段、响度、停顿、QA 与引擎无关，无需改动。

## 6. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `piper 引擎需要 --model` | 未指定模型 | 传入 `--model xxx.onnx` |
| `找不到 Piper 模型文件` | 路径错误 | 检查路径 |
| `找不到模型配置文件` | 缺少 `.onnx.json` | 与 `.onnx` 同名下载 |
| `未安装 piper-tts` | 缺依赖 | `pip install piper-tts` |
| `Chinese lookup tables not found ... Install the [zh] extra` | xiao_ya 缺 g2pw | `pip install --no-deps g2pw` |
| `No module named 'unicode_rbnf' / 'sentence_stream' / 'transformers'` | xiao_ya 缺中文前端依赖 | 见第 2 节 |
| 合成结果 `warnings` 含“输出过短” | 某段文本被引擎吞掉（如纯符号、乱码） | 检查 `normalized_text`，修改该段文本 |
| 长时间无响应 | g2pW 首次下载 / 冷启动 | 等待；或预先下载到模型目录 |

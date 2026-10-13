---
name: tts-voice-intelligence
description: 独立多引擎语音智能 Skill。用于朗读、配音、旁白、多语言 TTS、授权声音克隆及音频验收。先理解表达意图，再按真实引擎能力制定声音策略，执行规范化、分段、合成、必要处理、分层验收与针对性修复。Piper 与 eSpeak NG 可本地运行；XTTS 仅用户本地模型与有效授权，不保证情绪、跨语言音色稳定或专业听感。
compatibility: Python 3.13 已验证；numpy、soundfile 必需。引擎与 ASR 均为可选依赖；模型由用户提供，CPU 可用，GPU 和真实 XTTS 克隆未验证。
metadata:
  version: "0.4.0"
---

# TTS Voice Intelligence

本文件是唯一决策基准。`scripts/` 做确定性执行，`references/` 只提供按需依据，`tests/` 验证行为；不再设场景路由或重复规则表。

## 理解 → 判断 → 编排 → 生成 → 听评 → 修复

### 1. 理解表达目的

读完整文本，确定谁说、对谁说、为什么说、叙事转折、关键事实、目标语言/地区、时长、交付格式及商业用途。只有会改变结果的缺口才询问：尤其专名读法、口音、模型许可、克隆授权。
不要默认中文。自动语言识别只是脚本启发式：汉字不能可靠区分中日，拉丁文字不能区分英语/德语等，不能识别口音；优先用户明确语言，Piper 默认使用模型配置。混合语言标注风险，不默默换音色或音译所有英文。

### 2. 先判断，再映射能力（声音导演五步链）

按 **语境理解 (`context`) → 表达目标 (`intent`/`rationale`) → 可听变化 (`audible_change`) → 引擎支持的控制 (`length_scale`/`pause_after_ms`) → 结果验证 (`direction_validation` + QA `direction_execution` + 人评)** 制定策略。不要机械套用场景参数表；一条事实播报与一句迟疑反问即使同场景也有不同目的。

| 层级 | 当前可用手段 | 不得混淆 |
|---|---|---|
| 直接 | Piper 模型音色/合法 speaker id/逐段 `length_scale`；eSpeak 地区 voice/逐段 `length_scale`；逐段边界 `pause_after_ms`；XTTS 授权参考音色与配置内语言 | 不能从统一 API 推断所有引擎都有克隆/情绪/语速 |
| 间接 | 用户确认的朗读稿、标点、语义分段、必要整体峰值衰减 | 标点不保证重音；插入静音不是自然停顿 |
| 不支持 / 未验证 | `emotion`（不支持）、`emphasis` 重音（不支持）、`timbre` 音色调制（不支持）、跨段隐状态连续（不支持）、专业自然度（未人评验证） | 信号时长/语速变化已核验不等于主观听感自然；低 CER 不是自然度高 |

默认保持原文含义与原音动态。必要改写须保留原文并由用户确认；不可为了低 CER 丢词、改事实。可用 `--direction` 保存语境、意图、可听变化预期与逐段 `length_scale` / `pause_after_ms`；beat 必须匹配规范化后的实际段文本，传入 `emotion`/`emphasis`/`timbre` 等未实现控制立即拒绝，并在报告 `direction_validation` 中显式标记不支持与未验证项。

### 3. 检查引擎、许可、授权与模型隔离

```bash
python scripts/tts.py check --model MODEL.onnx
python scripts/tts.py check --engine espeak --language en-US
```

读取实例 `capabilities` 与可用性，不从项目支持语言列表推断某个模型支持全部语言。地区不匹配时拒绝，不能静默降级；具体依赖与矩阵按需读 `references/engines.md`。软件、模型、数据许可分别核对；许可未知不可默认可商用。

**克隆先授权、验签、绑定资产，再安全加载**：
- 仅本人或有效授权；收集用途范围、语言范围、有效期、撤销状态、授权证据标识、参考和模型哈希及模型许可，拒绝未替换的模板占位符。
- 支持 HMAC-SHA256 授权签名核验（`--require-signed-auth` 或 `TTS_AUTH_SIGNING_KEY`）；无外部公钥/CA 体系时，本地 JSON 或对称签名仍不等同法律公证。
- 参考音频单次读入内存完成 SHA256 校验与信号检查（单声道、≥16kHz、3–30s、无削波、有效信号≥2s且≥50%），消除检查与解码间 TOCTOU；但信号门槛不自动证明单说话人或身份，人工确认才设置 `reference_reviewed`。
- 模型与配置加载前执行静态安全筛查（拦截危险 pickle 全局调用、ZIP 路径穿越、配置远程拉取与 `..` 穿越，及 ONNX 外部路径穿越），加载前后及每段合成前后复核资产哈希绑定（`bound_assets`）；可选 `--isolate-model` 在独立离线子进程组中隔离执行。

### 4. 规范化与上下文编排

中文采用可测试规则：全角字母数字先转半角再匹配词典；校验 `YYYY-MM-DD`、`YYYY/MM/DD`、`YYYY年MM月DD日/号` 及 `MM月DD日/号` 合法日历日，无效日期/时刻拒绝；前导零编号（如 `001024`）与型号（`A-1024`）逐位读，不吞前导零。范围、长编号、电话、多音字、专名、混合语言仍需人工核对。英文和其他语言保留数字交实际引擎前端，不冒称语言特定读法均已实现。词典读法是最终文本，不能靠 ASR 证明它符合业务（同音字可能导致高 CER）。

### 5. 生成、硬超时、断点恢复与整组事务交付

```bash
python scripts/tts.py synth --model MODEL.onnx --input script.txt \
  --out output/narration.wav --format flac --report output/report.json \
  --checkpoint-dir output/ckpt --timeout 60
python scripts/qa.py output/narration.wav --synthesis-report output/report.json \
  --asr base --report output/qa.json
```

CLI 默认 `minimal`（保留原始动态与静音，仅超峰时整体衰减）；需旧行为时显式 `--processing legacy`。低层 `synthesize_text` 保留 `legacy` 默认以兼容旧调用。
- **硬超时与异常取消**：引擎加载与每段合成均受 `--timeout`（默认 60s）硬超时保护，子进程引擎超时或取消时终止整个进程组；超时与 `cancel_event` 取消绝不重试，可诊断瞬时故障最多重试一次，参数在 `finally` 中恢复。
- **失败恢复**：可选 `--checkpoint-dir` 按计划指纹（`plan_sha256`，绑定引擎、模型哈希、文本、参数与 direction）逐段原子保存音频与状态；中断后重跑自动校验哈希并从失败段续合成，文本或参数变更时拒绝混用。
- **整组交付事务**：母版 WAV、派生 FLAC/OGG 与 JSON 报告先写入临时文件并校验哈希，再带备份原子替换；任一环节失败则整组回滚，绝不留下半套新旧混合产物。时长上限 30 分钟、输入上限 10 万字符，全音频驻留内存，不声称流式或无限长。

### 6. 分层验收、隔离盲测与可追溯交付

| 层 | 实际执行 / 判定边界 | 修复优先级 |
|---|---|---|
| 文本 | 原文/朗读稿/报告逐项核对数字、前导零、专名、多音字、语言切换 | 只改错误条目，重新生成受影响内容 |
| 音频 | 解码、采样率、时长、有限值、削波、DC、活动RMS、长静音、静音区爆点、`direction_execution` | 先查引擎/拼接，必要处理，不能压低异常后声称无问题 |
| 识别（诊断） | `--asr base`；CER 含同音字与转写偏差，仅作诊断代理，不等同听感认证；skip 不是通过 | 对照原稿和实际听感，不靠重试碰运气或删减原文难点 |
| 盲听与克隆 | `--isolate` 物理隔离 `listen/`、`ratings/` 与包外私有 `--map-file`；prepare 校验可播性与元数据剥离 | 仅由真实人类评员打分，完整评分且试听音频哈希未变才可 `unblind`；不得编造人评、克隆结果或统计显著性 |

交付母版、派生件、合成/QA报告与真实评分（如有），包含模型/配置/词典/输出哈希、资产绑定、direction 验证、重试、警告与未验证边界。

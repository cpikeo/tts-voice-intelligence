---
name: tts-voice-intelligence
description: 文本转语音（TTS）端到端技能。理解朗读场景并制定声音策略（语速、停顿、表达节奏），对中文文本做数字、年份、时刻、百分比与专有名词规范化，分段合成并统一响度，输出 WAV 母版（可选 FLAC/OGG），再做音频验收（解码、削波、静音、回听识别字符错误率）。适用于配音、旁白、语音播报、有声内容、朗读稿生成与合成音频质量检查。当用户提到 text-to-speech、TTS、语音合成、朗读、配音、旁白、voice over、把文字转成语音、检查合成音频质量时使用。不做声音克隆。
compatibility: 已在 Python 3.13 上测试（更低版本未验证）；需要 numpy 与 soundfile。piper 引擎需要 piper-tts（GPL-3.0）及用户自备的 Piper .onnx 模型；中文模型还需要 g2pW 相关依赖。CPU 可运行，不需要 GPU 或 ffmpeg。
metadata:
  version: "0.2.0"
---

# TTS Voice Intelligence

把文本变成可交付的语音文件。本文件是唯一的决策基准；`scripts/` 只负责执行，`references/` 按需读取。

## 何时读哪个参考文件

- 选引擎、安装依赖、排查报错：读 `references/engines.md`
- 需要具体的声音参数、停顿、专名处理的判断依据：读 `references/voice-direction.md`
- 需要判断质量阈值、比较引擎表现、解释指标：读 `references/eval-results.md`

## 工作流程

按顺序执行。每一步的结论要能在报告或交付说明中找到依据。

### 1. 理解任务

明确以下信息。用户未说明的，按默认值执行并在交付说明中写明假设：

- 场景与听众（播报、讲解、故事、提示音）
- 语言（默认中文；中英混合时需要专名清单）
- 时长量级（短于 10 秒、几分钟、更长）
- 输出格式（默认 WAV 母版；需要时加 FLAC/OGG）
- 用途是否商业（决定可用模型，见第 3 步）

### 2. 制定声音策略

依据 `references/voice-direction.md` 第 2 节给出参数起点，例如语速 `--length-scale`、段落与句间停顿。这些参数都是起点，未经人工听评校准。

- 声音只能是所选模型的固定音色。不能承诺“换成某个人的声音”或“高兴地说”。
- 不做声音克隆。用户要求克隆他人声音时，说明本技能不支持，并要求其提供所有者授权与模型许可；不要借此推进合成。

### 3. 选择可用引擎

先检查，再决定：

```bash
python scripts/tts.py check --model MODEL.onnx
```

- `engines.piper.available` 为 `true`：继续。
- 为 `false`：如实报告 `engines.piper.reason`，并依据 `references/engines.md` 给出安装或获取模型的步骤。**不得在引擎不可用时生成音频、不得编造合成结果或指标。**
- 模型许可必须确认。`zh_CN-huayan-medium` 的数据许可为 Unknown，`zh_CN-xiao_ya-medium` 为非商用。若用户的用途是商业交付，且未确认授权，应先询问用户，不要默默使用。
- 同为中文时**优先 `xiao_ya`**：回听 CER 明显更低（0.044 vs 0.153，见 `references/eval-results.md`），
  且已有 1 名听者在**有效盲听 A/B**中偏好它（听匿名样本、评分前未揭示对应关系）。
  代价是它需 g2pW 前端依赖、更慢（RTF 约 0.09 vs 0.07）、内存占用更高（1.3 GB vs 0.5 GB），
  且**它是非商用许可**——商用交付两者都需先确认授权，不能因为音质好就默认可用。

### 4. 文本规范化

规范化由流水线自动完成（`scripts/textnorm.py`）：日期分隔符（`2026/10/12`、`2026-10-12`→二零二六年十月十二日）、年份逐位读（2026→二零二六）、时刻（10:30→十点三十分）、百分比（3.5%→百分之三点五）、亿/万 单位（1亿2000万→一亿二千万）、亿级大数（100000000→一亿）、负号（-5→负五）、编号型号（A-1024、GPT-4→逐位读）、小数与整数，以及全角字符统一。

**规范化覆盖不到、必须先人工改写**的形式：电话号码、价格区间与范围（10-20 只转成“十-二十”）、比值（3:2）、分数、复杂公式、表格、代码、列表符号。

需要人工判断的部分：

- **专有名词与英文缩写**：中文模型对英文读音不稳定。用 `--lexicon` 指定朗读形式（格式见 `assets/lexicon.example.tsv`）。词典读音必须由用户或业务方确认，不要替用户臆造品牌读音。词典给出的朗读形式是最终形式，不会被后续数字规则改写。
- **多音字**：没有引擎侧的消歧能力。发现误读时，用词典或改写词语搭配消歧（如“重庆”写成“重庆市”），不要靠反复重合成碰运气。
- 合成前检查 `report.json` 中的 `normalized_text`，确认关键数字与专名读法正确。

### 5. 合成

```bash
python scripts/tts.py synth --model MODEL.onnx --input script.txt \
  --lexicon assets/lexicon.example.tsv \
  --out out/narration.wav --format flac --report out/report.json
```

流水线做的事：规范化 → 按段落、句子、子句切分（单段默认不超过 90 字）→ 逐段合成 → 段内 RMS 对齐到 -20 dBFS（增益限制 ±12 dB）→ 段边界 12 ms 淡变 → 按标点插入停顿 → 峰值限制到 -1 dBFS → 写出母版。

- 引擎关闭了逐块峰值归一化（否则每段会被拉到峰值 1.0，段间响度不一致），响度完全由流水线控制。
- 单段失败会自动重试一次，仍失败则报错并给出段落位置。
- 报告含 `segment_rms_std_db`（段间响度标准差）与每段 `rms_dbfs`；超过 1 dB 会在 `warnings` 中提示。
  这是**客观一致性代理**，不代表音色一致性，仍需人工听评。
- 退出码：`0` 成功；`1` 输入为空（含规范化后为空）；`2` 引擎不可用；`3` 合成失败；`4` 输出参数错误（如 MP3）。

### 6. 验收

```bash
python scripts/qa.py out/narration.wav --text-file script.txt \
  --lexicon assets/lexicon.example.tsv --asr small --report out/qa.json
```

- 信号层（默认开启）：可解码、时长、削波、峰值、直流、有声 RMS、内部长静音（>2.5 秒为 warn）、静音区爆点。
- 回听层（`--asr small`，需安装 faster-whisper）：识别结果与规范化原文比较，输出**三个互不混淆的量**：
  - `asr_cer`（参与判定）：整体 CER。注意 whisper 会把听到的中文数字写成阿拉伯数字，
    比对时必须读回中文，因此它无法区分“读法对”与“数字串对”。
  - `asr_cer_text`（诊断）：按同一规则遮蔽数词后的 CER，只衡量非数字部分的发音，
    不受规范化链路影响。
  - `asr_num_cer`（诊断）：仅数词部分。只能验证“听到的数字串对不对”，**不能验证朗读风格**。
- 阈值见 `--profile`（default：warn 0.10、fail 0.25）。实测同一输入重复合成的 CER 波动
  可达 ±0.1，**单条用例的 CER 差异小于 0.1 不具有统计意义**；`strict`（0.05）低于该噪声水平，
  只在多次重复取均值后使用。
- 退出码：`0` 无 fail；`1` 存在 fail；`2` 文件不可读。
- 若未启用回听，必须在交付说明中写明“未做回听验证”。

客观指标只能作为代理。自然度、表现力、音色一致性需要人工抽听，本技能不能替代。

### 7. 人工听评（盲听）

自然度、表现力、韵律、音色一致性与段间连续性**只能由人判断**。交付前至少做一轮：

```bash
python scripts/blindtest.py prepare --audio out/*.wav --out pkg --seed 7
# 按 pkg/blind_map.json 的 order 试听 pkg/listen/ 下的匿名样本，填写 pkg/ratings.csv（1–5 分）
python scripts/blindtest.py score   --pkg pkg     # 汇总
python scripts/blindtest.py unblind --pkg pkg     # 评完再揭示样本与文件的对应关系
```

- 六个维度：发音准确度、自然度、韵律、表现力、音色一致性、段间连续性。
- 同一 `--seed` 可复现同一套匿名编号与试听顺序，便于跨轮次比较。
- 记分表里不出现原始文件名；**评完前不要打开 `blind_map.json`**，否则就不是盲测。
- 本技能只记录与汇总评分，**不能代替人打分**。未做人工听评时，交付说明必须写明。

### 8. 针对性修复

只修复验收中实际出现的问题，用最小改动：

| 现象 | 优先检查 | 处理 |
|---|---|---|
| 某句 CER 高，其余正常 | `normalized_text` | 增加词典条目；改写难读的词；拆短该句 |
| 内部静音过长 | `report.json` 的 `segments` 与 `pause_after` | 检查该段是否有符号或空内容；必要时调整停顿参数 |
| 削波或峰值告警 | `report.json` 的 `warnings` 与 `peak_dbfs` | 流水线已把峰值限制在 -1 dBFS，若仍告警，说明引擎输出异常；复核引擎与模型，不要靠调参掩盖 |
| 整体偏响或偏轻 | `loudness_rms` | 调整 `--target-db` |
| 警告“输出过短” | 该段文本 | 检查是否为纯符号或乱码，修改后重跑 |
| 语速太快或太慢 | 听感 | 调整 `--length-scale` 一次，再重跑 |
| 引擎报错 | 报错信息 | 按 `references/engines.md` 第 6 节排查 |
| 听感平淡/生硬（客观指标正常） | 人工听评 | 这是模型的表达能力上限，先确认无文本与分段问题；不要靠调参掩盖，必要时说明边界 |

修复后只重跑受影响的环节。例如只改了词典，就重新合成并重新验收，不必重复评测整个用例集。

### 9. 交付

交付前逐项确认：

- [ ] 母版 WAV 存在，`qa.py` 退出码为 0，或已说明未通过的项与原因
- [ ] 时长与文本量匹配（`qa.py` 的 `speech_rate_chars_per_s`，合理范围约 1.5–8 字/秒）
- [ ] 派生格式（如 FLAC/OGG）可解码
- [ ] `report.json` 与 `qa.json` 已保存；`report.json` 含模型路径与 sha256、参数、词典条目数、时间戳
- [ ] 已做人工听评并保留 `scores.json`，或明确写明“未做人工听评”
- [ ] 交付说明包含：使用的引擎与模型、模型许可、参数、词典、已做与未做的验证、已知限制

交付说明要如实写出：哪些是实测数字，哪些是启发式判断，哪些未验证（如人工听评、GPU、MP3）。

## 边界与禁止

- 不得在无合成引擎时声称已生成音频。
- 不得编造 CER、RTF、听感评价或验收结果。报告中的数字必须来自本次运行。
- 不得声称支持情绪控制、声音克隆、MP3 输出或 GPU 加速，除非已经实测。
- 不得把 GPT-SoVITS 或其他项目设为必需依赖。它们只能作为质量对标参考。
- 不内置、不分发模型权重。模型路径由用户提供。
- 中文模型的许可可能限制商用。商业交付前必须向用户确认。

## 脚本一览

| 文件 | 作用 | 依赖 |
|---|---|---|
| `scripts/tts.py` | 合成流水线与引擎适配（`check` / `synth`） | numpy、soundfile；piper 引擎需 piper-tts |
| `scripts/textnorm.py` | 中文文本规范化与发音词典（被 tts.py、qa.py 引用） | 标准库 |
| `scripts/qa.py` | 音频验收，可对任意 WAV/FLAC/OGG 运行 | numpy、soundfile；`--asr` 需 faster-whisper + scipy；繁简转换需 opencc-python-reimplemented |
| `scripts/blindtest.py` | 盲听听评：生成匿名样本与记分表、汇总评分、揭示对应 | 标准库 |
| `tests/` | 单元测试（无需模型）、真实引擎集成测试（需 `TTS_PIPER_MODEL`）、评测脚本 `run_eval.py` | pytest |

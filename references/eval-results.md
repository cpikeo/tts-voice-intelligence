# 评测口径与当前真实证据

核心决策只在 SKILL.md。以下数字全部来自当前仓库 `validation/` 中的真实音频与原始 JSON 产物，不含任何编造的人工评分、克隆结果或统计显著性声称。

## 指标定义与边界（代码、报告与文档统一）

| 指标 | 定义与边界 |
|---|---|
| `active_signal_seconds` / `speech_seconds` | 10ms 帧 RMS > -50 dBFS 的有效信号秒，末帧按真实长度；包含非语音声响，非语音学 VAD |
| `speech_rate_chars_per_s` | 规范化去标点、casefold 字符数 / 有效信号秒；不含静音；无跨语言统一阈值 |
| `delivery_rate_chars_per_s` | 同一字符数 / 成品总秒数（含静音）；不等同发声速度 |
| `segment_audio_seconds` | 各段音频总秒数（含模型自身尾部静音） |
| `loudness_rms` / 段 `rms_dbfs` | 活动帧掩码下信号 RMS dBFS（整体峰值衰减后测量）；不是 LUFS 或音色一致性 |
| `segment_rms_std_db` | 各段最终活动 RMS 标准差（>1 dB 提示属信号统计代理，不等于听感不连贯） |
| `asr_cer` | 字符编辑距离 / 参考字符数（可 >1）；仅作回听诊断代理，受同音字、数字转写与专名偏差影响，不等同听感认证 |
| `asr_cer_text` / `asr_num_cer` | 仅中文启发式遮蔽/提取数词后的诊断量；用于分离数词读法与普通文本偏差，非独立发音真值 |

## 本次 Piper 中文 7 类重点场景重复评测

模型：`zh_CN-huayan-medium.onnx`（SHA256 `9929917bf8cabb26fd528ea44d3a6699c11e87317a14765312420be230be0f3d`）+ `faster-whisper base`（int8 CPU）+ `assets/lexicon.example.tsv`。
覆盖 7 类核心场景 × 2 手臂（`norm` 规范化+词典 vs `raw` 原文直送）× 3 重复（`seed=7`），共 **42 个真实 WAV 音频**，全部保留在 `validation/piper-repeat/`，完整原始记录见 `validation/piper-repeat/summary.json`。

| 用例 ID | 类别 | `norm` CER 均值 (范围) | `raw` CER 均值 (范围) | `norm` 文本/数词 CER | 声学与诊断解读 |
|---|---|---:|---:|---:|---|
| `zh_plain` | 中文准确性 | 0.1146 (0.0938–0.1562) | 0.1458 (0.1250–0.1562) | text 0.1146 / num None | 信号全通过；ASR 差异主要为同音字（“再→在”“合→核”） |
| `digits_date` | 数字日期 | 0.0938 (0.0625–0.1250) | 0.2708 (0.2500–0.2812) | text 0.1755 / **num 0.0000** | `norm` 数词 CER 为 0；`raw` 将 2026 读作二千二十六、漏读冒号与小数点 |
| `mixed_en` | 中英混合 | 0.3763 (0.3226–0.4516) | 0.4301 (0.4194–0.4516) | text 0.4286 / num 0.3333 | 两手臂均触发 ASR fail：单语中文模型直读英文失真，音译词（吉皮优/西皮优）被 ASR 写成同音字 |
| `proper` | 专有名词 | 0.2933 (0.2800–0.3200) | 0.3067 (0.2400–0.4000) | text 0.2933 / num None | `raw` 将 Piper/OpenAI 读成乱码；`norm` 读出“派珀/欧朋艾艾”但被 ASR 记为同音字“拍扑/欧彭爱爱” |
| `polyphone` | 多音字 | 0.0667 (0.0000–0.1000) | 0.0500 (0.0000–0.1500) | text 0.0667 / num None | “银行行长/重庆重要”在 `r0` 达 0.0 CER；复杂语境多音字仍须人耳核对 |
| `zh_emotional_turn` | 叙事韵律 | 0.0595 (0.0357–0.0714) | 0.0714 (0.0714–0.0714) | text 0.0595 / num None | 低 CER 仅反映字面可懂度（“约定→月定”同音），不证明叙事韵律或情绪感染力 |
| `zh_long` | 长文本段落 | 0.1867 (0.1733–0.2000) | 0.2044 (0.1600–0.2400) | text 0.1867 / num None | 4 句叙事长段信号无异常；ASR 同音字累积（“街道→借道”“安静→暗静”） |

42 个音频信号层 **0 fail / 0 warn**；ASR 诊断层共 **15 pass / 14 warn / 13 fail**（主要集中在 `mixed_en` 与 `proper`）。3 次重复与单模型样本不作统计显著性声称。

## 多语言 eSpeak、声音导演与长文本连续性实测

- **eSpeak NG 四语言重复评测**：`en-US`、`en-GB`、`de`、`ja` 各 1 用例 × 2 手臂 × 3 重复 = **24 个真实 WAV**（均保留在 `validation/espeak-*/`），信号层全部 0 fail / 0 warn。`en-US` 启用 `whisper-base` 回听，6 次 CER 均为 `0.125`（warn，因英文数字/日期书写与转写格式差异）。
- **声音导演契约实测 (`validation/direction.json` / `direction-qa.json`)**：使用 `assets/direction.example.json` 执行完整五步链（`context` → `intent` → `audible_change` → `length_scale=1.3, pause_after_ms=450` → `direction_validation`），QA `direction_execution` 检查通过（`pass`），报告明确标注 `emotion`/`emphasis`/`timbre_modulation` 为 `unsupported`。
- **长文本连续性 (`validation/long.wav` / `long.json` / `long-qa.json`)**：`zh_long_production`（1072 字符、64 段），成品总长 **201.805 秒**，有效信号 **177.766 秒**，合成耗时 **21.116 秒**（wall **23.225 秒**，RTF **0.105**），峰值 RSS **338.988 MiB**，段间活动 RMS 标准差 **1.043 dB**。信号层 0 fail / 0 warn；本次新增 `whisper-base` 全文回听诊断，`asr_cer = 0.2775`（因 16 段重复文本中“街道/蒸汽/牵着狗/安静”等同音字在长音频转写中累积）。

## 隔离盲测包与未验证边界

- `validation/listening-unrated/listen/` 保存 **9 个真实匿名试听 WAV（`A.wav`–`I.wav`，`seed=17`）**，覆盖上述 7 类核心中文场景与 norm/raw 对照；`validation/listening-unrated/ratings/ratings.csv` 为待人类评员填写的空表；密钥映射隔离在 `private/listening-blind-map.json`（`0600`，`.gitignore` 排除，不入库）。
- 试听、评分、汇总与揭盲完整机制已通过隔离工作流实测（见 `validation/blindtest-workflow-check.json`）。真实人类主观听评、真实授权 XTTS 模型克隆与 GPU 加速仍属未验证项（`not_performed`）。

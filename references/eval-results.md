# 评测口径与当前证据

核心决策只在 SKILL.md。以下数字来自 validation/ 原始产物，不是人工听评。

## 历史结论纠正

原版本的 CER/资源表未附原始 summary、音频或评员评分；同时声称“无人评”和“有效盲听偏好 xiao_ya”，互相矛盾。现撤回模型优先推荐、通用噪声下限/统计显著性与“人工可接受阈值”的确定性描述。历史文件可由基线 git 查阅，不把它们当本次证据。

## 指标（统一代码、报告与文档）

| 指标 | 定义与边界 |
|---|---|
| active_signal_seconds / speech_seconds | 按实际采样率的约10ms帧RMS>-50dBFS的有效信号秒，末帧按真实长度；包括非语音声响，非语音VAD |
| speech_rate_chars_per_s | 规范化后去标点、casefold字符数 / 有效信号秒；排除静音；无跨语言统一速度阈值 |
| delivery_rate_chars_per_s | 同一字符数 / 成品总秒数（包括静音）；不能叫发声速度 |
| segment_audio_seconds | 各段音频总秒数（包含模型内部静音），与旧speech_seconds的含义相同 |
| loudness_rms / segment rms_dbfs | 同一帧活动掩码下信号RMS dBFS；不是LUFS或音色；各段值在最终整体衰减后测量 |
| segment_rms_std_db | 最终各段活动RMS标准差，响度统计代理，不是音色/连续性验收；>1dB提示属启发式 |
| asr_cer | 编辑距离 / 参考字符数，可>1；去标点与casefold。中文假定数词规范化、可选繁简转换；外语不套中文数词规则 |
| asr_cer_text / asr_num_cer | 仅中文启发式遮蔽/提取数词后的诊断量，可能误遮蔽普通词/残留单字；不是独立真值或目标音色指标 |
| RTF | 逐段合成耗时 / 成品总时长（含静音）；不是全流程冷启动耗时，不能代表音质 |

ASR 会写错专名、归一化数字、受语言设置影响。中文CER不能区分读法风格；外语文本保留数字时，读音正确但ASR拼成英文数词也可增大CER。warn/fail是未人评校准的工程阈值：default 0.10/0.25、strict 0.05/0.15，无统一显著性界限。strict也不是自然度标准。未启用ASR、依赖/模型不可用均须区别记录，skip不能视为通过。

## 本次重复评测

Piper huayan + whisper-base int8 CPU，3文本×2手臂×3重复，共18音频。手臂顺序按seed=7随机；norm含中文规范化与词典，raw直送原文，因此是联合干预，不是孤立词典或单条规则消融。每次原始信号/ASR文本与指标、输出哈希、模型/配置、代码哈希保留于 validation/piper-repeat/summary.json。

| 用例 | norm CER均值 | raw CER均值 | norm范围 |
|---|---|---|---|
| digits_date | 0.0834 | 0.2604 | 0.0625–0.0938 |
| mixed_en | 0.3011 | 0.4839 | 0.2903–0.3226 |
| zh_emotional_turn | 0.0714 | 0.0714 | 0.0714–0.0714 |

18个文件信号层无fail/warn；ASR整体有7 fail与2 warn，其中mixed_en两手臂全部fail。数字日期样本的代理指标方向有利，但3重复、1模型、单ASR、未人评不支持总体效果/自然度结论。叙事转折低CER不代表情绪表达已实现。

eSpeak en-US/en-GB/de/ja各1文本×2手臂×3重复，共24文件信号层无fail/warn；en-US启用ASR，6次CER均0.125（全部warn），数字书写/读法不一致是可能原因，需人耳核对；其余3语言未做ASR。单元集成另有cmn真实信号检查。外语norm通常不改文本，两手臂作为零效应技术对照，不能宣称规范化收益。

长文本单次：1072字符（含空白标点）、64段、203.976秒，有效信号177.886秒；整体RTF 0.109、wall 25.76秒、峰值RSS 406.05MiB。QA无fail/warn；合成报告RMS标准差 1.033dB触发响度统计提示，不等于听感不连贯。没有ASR或人评；未覆盖10分钟以上真实长篇。

## 未验证与人评

真实XTTS、目标音色保持、跨语言克隆漂移、GPU、专业自然度/表现力/连续性均未验证。validation/listening-unrated 有匿名实际样本与空记分表，无scores.json，不伪造评分。测试fixture中的数值只验证汇总逻辑，不是人评证据。

复现命令见 README 与 validation/reproduce.sh；环境/依赖与ASR资产哈希见 validation/environment.json。部分工程测试与评测并行，RTF/RSS是环境样本，不能当严格基准。模型随机噪声意味着重新生成哈希与指标可能不同；哈希仅追踪本次文件。

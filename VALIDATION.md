# 全仓实施与验证交付报告

**结论：已完成按优先级的全部代码修复、安全门禁强化、真实多引擎合成、7 类中文重点场景重复 ASR 评测及隔离盲测工作流验证；因缺少真实人类评员主观打分与合法授权的本地 XTTS 声纹/权重资产，“主观听感认证”与“真实授权克隆音色保持”两项维持显式未完成（`not_performed` / `skipped`），不作虚假宣称。**

## 1. 基线与审计范围

- **远端基线**：`https://github.com/cpikeo/tts-voice-intelligence.git` @ `14e09dcadccfdd5ab9212b93d96d6f395a7423f4`（前序基线 `132d5f938fb0f7797eb4280a7d3047665f3b1500`）。
- **版本升级**：`0.3.0` → `0.4.0`。
- **修复的核心基线缺陷**（详见 [AUDIT.md](AUDIT.md)）：
  1. **证据与 Git 状态不一致**：`14e09dc` 中 `.gitignore` 忽略了 `validation/**/*.wav`，导致 `artifact-manifest.json` 声称存在的 48 个音频在仓库中全部缺失，同时 `validation/listening-unrated/blind_map.json` 被公开提交至 Git，且 `scripts/qa.py` 在评测运行后被修改造成 `source-versions.json` 哈希漂移。
  2. **盲测未隔离**：原 `blindtest.py` 将 `blind_map.json` 默认写入试听包同目录，且 `score` 强依赖读取 `blind_map.json`。
  3. **真实质量覆盖不足与文本规范化漏洞**：原 Piper 重复评测仅跑 3 用例，漏测普通中文、专名、多音字、长段及长文本全文 ASR；`normalize_zh` 全角转半角顺序晚于词典匹配、未校验 `YYYY年MM月DD日` 日历合法性、前导零编号 `001024` 丢失前导零。
  4. **声音导演闭环缺失**：原 `direction` 缺少 `context`（语境）、`audible_change`（可听变化）、`pause_after_ms`（逐段边界停顿）及合成报告 `direction_validation` / QA `direction_execution` 自动验证与不支持能力结构化标注。
  5. **生产可靠性与安全缺口**：原实现缺少引擎硬超时、异常取消清理、整组交付原子事务与回滚、分段断点恢复（`--checkpoint-dir`）、参考音频单次读取防 TOCTOU 与加载前后哈希绑定、授权 HMAC-SHA256 签名验真及不可信模型静态拦截/子进程隔离（`--isolate-model`）。

## 2. 按优先级的实施与变更清单

| 优先级与模块 | 修改文件 | 具体实施内容 | 验证方式 |
|---|---|---|---|
| **1. 证据与盲测** | `.gitignore`、`scripts/blindtest.py`、`validation/reproduce.sh` | 豁免 `!validation/**/*.wav` 并忽略 `**/blind_map.json`；`blindtest.py` 新增 `--isolate`、`--map-file`、`--ratings-dir`，将试听音频（`listen/`）、评分表（`ratings/`）与私有映射（`private/listening-blind-map.json`，`0600`）物理隔离；`prepare` 自动解码核验每个试听音频可播性与源文件名零残留并生成匿名 `listen/manifest.json`；`score` 无需映射即可汇总；`unblind` 校验全量评分完整性与试听音频哈希防篡改后生成 `unblinded.json` | `tests/test_blindtest.py`、`validation/blindtest-workflow-check.json` |
| **2. 真实质量与规范化** | `scripts/textnorm.py`、`tests/run_eval.py`、`tests/test_textnorm.py` | 全角字母数字转半角提前至词典匹配前（修复 `ＧＰＵ` 匹配）；新增 `YYYY年MM月DD日/号` 与 `MM月DD日/号` 合法日历校验；前导零编号（`0\d+` 如 `001024`）按逐位数字读法保留前导零；`run_eval.py` 汇总 `cer`、`cer_text`、`num_cer`；真实执行 Piper 7 类中文核心场景 × 2 手臂 × 3 重复 + 长文本全文 ASR | `tests/test_textnorm.py`、`validation/piper-repeat/summary.json`、`validation/long-qa.json` |
| **3. 声音导演** | `scripts/tts.py`、`scripts/qa.py`、`assets/direction.example.json` | 扩展 `direction` 契约支持 `context`、`audible_goal`、beat `audible_change` 与 `pause_after_ms`（0–5000ms）；严格拒绝 `emotion`/`emphasis`/`timbre` 等未实现控制；合成报告输出 `direction_validation`（含每段实测 `active_seconds`、`speech_rate_chars_per_s`、`pause_after_ms` 及 `unsupported_or_unverified` 显式标记）；`qa.py` 新增 `direction_execution` 检查项 | `tests/test_intelligence.py`、`validation/direction.json`、`validation/direction-qa.json` |
| **4. 生产可靠性与安全** | `scripts/engines.py`、`scripts/tts.py`、`tests/test_intelligence.py` | ① `run_with_timeout` 与 `EngineTimeoutError` 实现默认 60s 硬超时，eSpeak 使用 `start_new_session` + `killpg` 清理进程组，超时与 `cancel_event` 取消绝不重试；② `transactional_write_bundle` 实现母版 WAV + 派生 FLAC/OGG + 报告整组暂存、哈希校验与失败全量回滚；③ `--checkpoint-dir` 实现按 `plan_sha256` 校验的分段断点保存与失败恢复；④ `_read_and_fingerprint` + `verify_bound_asset` 消除参考音频 TOCTOU 并绑定模型加载前后哈希；⑤ `sign_authorization` / `validate_authorization` 支持 HMAC-SHA256 授权签名核验并拦截模板占位符；⑥ `inspect_model_safety` 静态拦截恶意 pickle 全局调用、ZIP Slip、配置路径穿越及 ONNX 外部路径穿越，支持 `--isolate-model` 离线子进程隔离 | `tests/test_intelligence.py`（新增 8 组专项验收测试） |
| **5. 可维护性** | `SKILL.md`、`README.md`、`references/*.md` | `SKILL.md` 保持唯一决策基准（3933 字符，较初始基线 5903 字符精简 33.4%）；五篇核心文档总计 15135 字符（较初始基线 19560 字符减少 22.6%）；所有安全门禁、测试与证据完整保留 | `validation/complexity.json` |

## 3. 测试与真实评测证据区分

| 证据类别 | 执行命令 / 产物位置 | 实测结果 | 边界说明 |
|---|---|---|---|
| **回归与安全契约测试（含测试替身）** | `python -m pytest -q`（`validation/pytest-no-piper.log`） | **150 passed, 9 skipped**（0.93s） | 无 Piper 模型时自动跳过 8 项 Piper 与 1 项 XTTS；验证规范化、分段、硬超时、取消、事务回滚、断点恢复、HMAC 验签、恶意 pickle/ZIP 拦截、盲测隔离与 FakeAPI 契约 |
| **真实引擎集成测试（Piper + eSpeak）** | `TTS_PIPER_MODEL=/home/user/.cache/tts-models/zh_CN-huayan-medium.onnx python -m pytest -q`（`validation/pytest-full.log`） | **158 passed, 1 skipped**（29.74s） | Piper 8 项真实中文集成与 eSpeak 中/英/德/日及声音导演真实集成全部通过；唯一 skip 为 `test_real_authorized_clone_cross_text`（缺真实 XTTS 资产） |
| **真实中文 7 类场景重复合成 + ASR 诊断** | `validation/piper-repeat/`（42 个真实 `.wav` + `summary.json` + `runs.json`） | 42 个音频信号层 **0 fail / 0 warn**；`whisper-base` 诊断层 **15 pass / 14 warn / 13 fail** | 覆盖 `zh_plain`、`digits_date`、`mixed_en`、`proper`、`polyphone`、`zh_emotional_turn`、`zh_long`；`source-versions.json` 确认源码零漂移 |
| **真实多语言 eSpeak 重复评测** | `validation/espeak-en-us/`、`espeak-en-GB/`、`espeak-de/`、`espeak-ja/`（24 个真实 `.wav`） | 24 个音频信号层 **0 fail / 0 warn**；`en-US` 6 次 ASR CER 均 `0.125`（warn） | 外语保留数字交引擎前端；`en-GB`/`de`/`ja` 仅做信号验收，未做 ASR 或人评 |
| **真实长文本连续性与 ASR 诊断** | `validation/long.wav`、`long.json`、`long-qa.json`、`long-resources.json` | 1072 字符、64 段、201.805s（有效信号 177.766s），RTF 0.105，wall 23.225s，峰值 RSS 338.988 MiB；信号 0 fail/0 warn，全文 ASR CER `0.2775` | 段间活动 RMS 标准差 1.043 dB（minimal 模式保留自然动态）；长音频同音字累积使 `whisper-base` CER 超出 0.25 工程阈值 |
| **真实声音导演执行与 QA 核验** | `validation/direction.wav`、`direction.json`、`direction-qa.json` | 退出码 0，`direction_execution` 状态 `pass` | 实测 `length_scale=1.3` 与 `pause_after_ms=450` 生效，显式标记 `emotion`/`emphasis`/`timbre_modulation` 为 `unsupported` |
| **物理隔离盲测包与工作流实测** | `validation/listening-unrated/`、`private/listening-blind-map.json`、`validation/blindtest-workflow-check.json` | 9 个匿名真实音频 `A.wav`–`I.wav`（`seed=17`）全部通过解码与元数据剥离校验；隔离汇总与揭盲流程验证通过 | `validation/listening-unrated/ratings/ratings.csv` 保持空白未评分（`human_listening: not_performed`）；不编造人类评分 |
| **真实授权 XTTS 克隆** | `validation/xtts-blocked.json`、`tests/test_xtts_integration.py` | 门禁阻断返回码 `2`；真实跨文本集成测试 `skipped` | 未安装 `coqui-tts`，无合法真人参考录音与 XTTS 权重；绝不伪造克隆样本或相似度指标 |

### Piper 7 类中文重点场景 CER 诊断汇总（3 重复 × 2 手臂）

| 用例 ID | 类别 | `norm` CER 均值 (min–max) | `raw` CER 均值 (min–max) | `norm` `cer_text` / `num_cer` | 关键诊断结论 |
|---|---|---:|---:|---:|---|
| `zh_plain` | 普通中文 | 0.1146 (0.0938–0.1562) | 0.1458 (0.1250–0.1562) | 0.1146 / None | 基础发音清晰，ASR 误差源于“再→在”“合→核”同音转写 |
| `digits_date` | 数字日期 | 0.0938 (0.0625–0.1250) | 0.2708 (0.2500–0.2812) | 0.1755 / **0.0000** | 规范化消除年份基数误读（二千二十六）及冒号/小数点漏读，`norm` 数词 CER 降为 0 |
| `mixed_en` | 中英混合 | 0.3763 (0.3226–0.4516) | 0.4301 (0.4194–0.4516) | 0.4286 / 0.3333 | 单语中文模型直读英文失真；音译词（吉皮优/西皮优）被 Whisper 写成同音字（极皮幼/细皮幼），不可当作成品交付 |
| `proper` | 专有名词 | 0.2933 (0.2800–0.3200) | 0.3067 (0.2400–0.4000) | 0.2933 / None | 词典使发音变为预定音译（派珀/欧朋艾艾），但 ASR 转写为同音异形字（拍扑/欧彭爱爱），证明词典读法必须靠人耳验收而非字面 CER |
| `polyphone` | 多音字 | 0.0667 (0.0000–0.1000) | 0.0500 (0.0000–0.1500) | 0.0667 / None | 常见词组（银行行长/重庆重要）发音正确，生僻或歧义语境多音字仍需人工核对 |
| `zh_emotional_turn` | 叙事韵律 | 0.0595 (0.0357–0.0714) | 0.0714 (0.0714–0.0714) | 0.0595 / None | 字面可懂度高，但叙事转折的停顿与语气表现力需盲听评估 |
| `zh_long` | 长文本段落 | 0.1867 (0.1733–0.2000) | 0.2044 (0.1600–0.2400) | 0.1867 / None | 信号稳定，ASR 存在多处同音字替换（借道/灵性/正期/暗静） |

## 4. 关键资产哈希与复现方式

- **测试用本地模型（未入库）**：
  - `zh_CN-huayan-medium.onnx`：63,201,294 bytes，SHA256 `9929917bf8cabb26fd528ea44d3a6699c11e87317a14765312420be230be0f3d`
  - `zh_CN-huayan-medium.onnx.json`：4,822 bytes，SHA256 `d521dc45504a8ccc99e325822b35946dd701840bfb07e3dbb31a40929ed6a82b`
- **核心脚本最终 SHA256**（与 `validation/source-versions.json` 及所有 `summary.json` 内 `source_sha256` 完全一致）：
  - `scripts/blindtest.py`：`0e1d36409614b9a8414e577920cb50750a5f06e61a3998f8d65e33c6ec0d87ae`
  - `scripts/engines.py`：`3de29a2eb1b6c5bf6d4f79825c20bb8f3923e39bd591383179ce9db415303ec0`
  - `scripts/qa.py`：`64b049aa954434da3bf24148542a79ff9a0a672e03ab39c255cf007184129c46`
  - `scripts/textnorm.py`：`e4c929a8a8e31db5b2e076c42dbca962ab896b76bfceef7fd23c05318efbb69e`
  - `scripts/tts.py`：`4f119cfcf88eb963db57d8ab7ab8f92e6478585b1bd68ff8be6d055b7867eb9c`
  - `tests/cases.jsonl`：`266e60ef8504c64b1f141204e33a94e15dd1ba7d0b669a9dc2186ceeb215501c`
  - `tests/run_eval.py`：`c19f27206c6e0a236b329ff942ba41a4be0d6699603caed6d402685b8fd68971`
- **完整复现命令**：
  ```bash
  bash validation/reproduce.sh /path/to/zh_CN-huayan-medium.onnx new-validation private-out
  ```
- 全仓 100+ 交付文件与全部 77 个真实 `.wav` 音频的字节大小与 SHA256 清单见 `validation/artifact-manifest.json`。

## 5. 剩余风险与分阶段验收结论

### 剩余风险
1. **中英混合与专名发音局限**：单语中文 Piper 模型（`huayan`）直读英文严重失真，词典音译虽可固定读法但听感生硬且易触发 ASR 同音异形字误报；高标准中英混读需多语言原生模型并经人工听审。
2. **主观听感与真实克隆待外部条件闭环**：`validation/listening-unrated` 已备好 9 个真实试听音频与隔离映射，仍待具备目标语言母语能力的独立评员完成盲听打分；真实 XTTS 克隆需在具备合法授权参考录音与本地模型权重的环境中执行。
3. **长文本内存边界与授权法律属性**：当前合成仍采用全量音频驻留内存（上限 30 分钟/10 万字符），超长内容需按章节分批并配合 `--checkpoint-dir`；HMAC-SHA256 签名保障授权文件防篡改，但签署人真实身份与法律效力仍依赖外部合规流程。

### 分阶段验收结论
- **阶段一（代码架构、声音导演契约、生产可靠性与安全隔离）**：**通过（Passed）**。158 项自动化回归与真实引擎集成测试全部通过，硬超时、取消、整组事务回滚、断点恢复、资产哈希绑定、HMAC 授权验签、恶意模型拦截与盲测三方隔离均已闭环。
- **阶段二（真实引擎客观评测与可追溯证据交付）**：**通过（Passed）**。77 个真实合成与盲听 `.wav` 音频、ASR 原始输出、源码哈希与 `artifact-manifest.json` 完全一致且受控入库，消除了基线仓库的证据缺失与映射泄露。
- **阶段三（专业级人类主观盲听认证与真实授权克隆终验）**：**挂起待外部资产/人评（Pending External Human Raters & Authorized Clone Assets）**。在补齐真人盲评与合法克隆模型/参考录音前，不宣称专业级主观听感或真实克隆终验完成。

# 全仓与真实调用路径深度审计报告

审计基线：`https://github.com/cpikeo/tts-voice-intelligence.git` @ `14e09dcadccfdd5ab9212b93d96d6f395a7423f4`（及其前序基线 `132d5f938fb0f7797eb4280a7d3047665f3b1500`）。
审读范围：全仓 60 个受控文件与真实调用路径（CLI → `build_engine` → `inspect_model_safety` / `validate_authorization` → `normalize_text` → `segment` → `synthesize_text` → `transactional_write_bundle` → `qa.py` → `blindtest.py` → `run_eval.py`）。

## 1. 当前基线（`14e09dc`）深度审计发现清单

| 优先级 | 范畴 | 位置 | 审计发现的具体问题与可验证风险 | 状态 |
|---|---|---|---|---|
| **P0** | 证据与盲测 | `.gitignore`、`validation/`、`VALIDATION.md` | **验证报告与 Git 状态不一致**：`artifact-manifest.json` 与 `VALIDATION.md` 声称包含 48 个评测与盲听 `.wav` 音频，但 `.gitignore` 全局忽略 `*.wav`（仅豁免 `tests/`），导致克隆仓库后 `validation/` 下 **0 个音频文件**；同时 `VALIDATION.md` 声称“未推送 GitHub”，与已在远端 `14e09dc` 的事实矛盾 | **已修复** |
| **P0** | 证据与盲测 | `validation/listening-unrated/`、`scripts/blindtest.py` | **公开映射泄露且试听材料缺失**：`validation/listening-unrated/blind_map.json` 直接提交至公开 Git 仓库并与 `ratings.csv` 同目录，而 `listen/A.wav..D.wav` 完全缺失；`blindtest.py score` 强依赖读取同目录下的 `blind_map.json`，无法做到真实音频、评分表与私有映射物理隔离 | **已修复** |
| **P0** | 证据与盲测 | `validation/source-versions.json` | **源码哈希漂移**：数值评测运行后又修改了 `scripts/qa.py`，导致 `validation/*/summary.json` 内记录的 `source_sha256` 与仓库最终 `scripts/qa.py` 哈希不一致 | **已修复** |
| **P1** | 真实质量 | `validation/piper-repeat/`、`textnorm.py` | **核心中文场景评测残缺与规范化漏洞**：仅跑了 3 个用例，漏测 `zh_plain`（中文准确性）、`proper`（专名）、`polyphone`（多音字）、`zh_long`（长段）及 `long.wav` 的 ASR 诊断；`normalize_zh` 全角转半角晚于词典匹配（`ＧＰＵ` 漏匹配）、未校验 `2026年2月30日` 等中文年月日格式、前导零编号 `001024` 被当基数词吞掉前导零读成“一千零二十四” | **已修复** |
| **P1** | 声音导演 | `scripts/tts.py`、`scripts/qa.py` | **导演契约缺少完整验证闭环**：`direction` 仅有 `intent` 与 `length_scale`，缺少 `context`（语境理解）、`audible_change`（可听变化目标）、`pause_after_ms`（逐段边界停顿控制）以及合成后信号层 `direction_validation` 与 QA `direction_execution` 自动核验，未结构化标记不支持的情绪/重音/音色能力 | **已修复** |
| **P1** | 生产可靠性 | `scripts/engines.py`、`scripts/tts.py` | **无硬超时与异常取消清理**：`PiperEngine`、`XTTSEngine` 与 `synthesize_text` 无硬超时；`EspeakEngine` 超时或收到取消信号时未按进程组（`start_new_session` + `killpg`）清理子进程；缺乏 `cancel_event` 即时中断机制 | **已修复** |
| **P1** | 生产可靠性 | `scripts/tts.py` | **非事务交付与无失败恢复**：母版 WAV、派生 FLAC/OGG 与 JSON 报告逐个直接覆盖，中途失败会留下新旧混杂的损坏交付组；多段长文本无分段 checkpoint 保存与哈希校验续合成能力 | **已修复** |
| **P1** | 安全与隔离 | `scripts/engines.py` | **资产检查与加载脱节（TOCTOU）、授权无签名验真、不可信模型无拦截**：参考音频先读字节算哈希再二次按路径打开解码；模型加载前后未复核文件指纹；授权 JSON 仅靠布尔字段自证且未拦截模板占位符；未静态拦截恶意 PyTorch `.pth` pickle 代码执行/ZIP 穿越，无 `--isolate-model` 子进程隔离 | **已修复** |

## 2. 实施准则与架构约束

1. **不扩大体系、不堆叠规则**：保持 `SKILL.md` 为唯一决策基准，`references/` 按需读取；不新增场景参数路由表或虚构情绪/重音参数。
2. **向后兼容**：保留 `Engine`、`PiperEngine`、`EspeakEngine`、`XTTSEngine`、`build_engine`、`segment`、`synthesize_text`、`write_audio`、`qa.main`、`blindtest.main` 原有签名与默认行为；新增能力均为可选参数或安全强化。
3. **证据诚实**：严格区分测试替身（正弦波/FakeAPI）、真实引擎合成（Piper/eSpeak）、ASR 诊断（`faster-whisper base`）、人工盲评（仅模板与机制验证，不伪造人评分数）与授权克隆（缺真实合法资产，保持 `skipped`/未验证）。

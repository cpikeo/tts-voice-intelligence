# 声音导演契约与隔离盲听协议（按需读取）

核心决策只在 SKILL.md；此处不再重复场景参数表或规范化规则。

## 声音导演五步链（语境 → 意图 → 可听变化 → 引擎控制 → 结果验证）

1. **语境理解 (`context`)**：明确当前句在全篇的位置、人物关系、信息新旧与转折关系（如“前文铺垫后的颠覆性结论”）。
2. **表达目标 (`intent` / `rationale`)**：明确要让听众获得什么认知或感受（如“给关键结论留出消化时间，避免夸张播报”）。
3. **可听变化 (`audible_change`)**：将抽象意图转化为可听的声学目标（如“语速放慢至约 0.77 倍，句尾留 450ms 停顿”）。
4. **引擎支持的控制 (`length_scale` / `pause_after_ms`) 与不支持项标记**：
   - 已实现直接控制：Piper/eSpeak 的逐段 `length_scale`（`0.5..2.0`）与管线逐段边界停顿 `pause_after_ms`（`0..5000` ms，即使在 `minimal` 模式下也可由 beat 显式指定）。
   - 未实现控制拒绝伪装：在 `direction` 或 `beats` 中传入 `emotion`、`emphasis`、`timbre`、`pitch`、`style` 等参数立即抛出 `ValueError`。
   - 合成报告 `direction_validation` 显式标记：`emotion: "unsupported"`、`emphasis: "unsupported"`、`timbre_modulation: "unsupported"`、`cross_segment_prosody_continuity: "unverified"`、`human_perceptual_effect: "unverified_requires_human_listening"`。
5. **结果验证 (`direction_validation` + QA `direction_execution`)**：
   - 信号层核验每段实际 `duration_seconds`、`active_seconds`、`speech_rate_chars_per_s` 与 `pause_after_ms` 是否按契约生效；`qa.py --synthesis-report` 自动输出 `direction_execution` 检查项。
   - 信号层控制生效不等同主观表现力自然，最终韵律与情绪传染力须经盲听人评确认。

## 停顿与处理的代价

CLI 默认 `minimal` 保留模型原始动态，不做逐段 RMS 对齐或固定淡变；但默认停顿策略仍会插入段落/句子/子句间 700/400/200 ms 停顿，`--pause-* 0` 可关闭，direction beat 的 `pause_after_ms` 覆盖对应边界。插入策略停顿时仅裁掉引擎自带边界静音、保留约 30 ms 余量，再插入目标长度；数字静音交界仅在检测到超过 -40 dBFS 的跳变时对语音侧做 2 ms 修复，超峰时才整体衰减。`--processing legacy` 额外启用逐段 RMS 对齐（±12 dB）和 12ms 淡变。模型发音错误先修朗读稿/词典，不能用音频处理掩盖。

## 物理隔离的盲听协议

1. **三方隔离 (`--isolate`)**：
   ```bash
   python scripts/blindtest.py prepare --audio sample1.wav sample2.wav \
     --out listening/unrated \
     --ratings-dir listening/unrated/ratings \
     --map-file private/listening-blind-map.json --isolate --seed 17
   ```
   - `listen/`：仅含重编码剥离容器元数据的匿名音频（`A.wav..I.wav`）、`order.txt` 与不含源路径的 `manifest.json`（prepare 阶段自动解码验证可播放性、有限值与源文件名零残留）。
   - `ratings/`：仅含发给人类评员的 `ratings.csv`（评完后生成 `scores.json` / `scores.md`）。
   - `private/listening-blind-map.json`：操作者私有保存（`chmod 0600`），位于试听包目录之外且被 `.gitignore` 忽略，防止 Git 提交泄露映射。
2. **无映射汇总与防篡改揭盲**：
   - `blindtest.py score` 直接依据 `listen/manifest.json` 校验样本标签与试听文件哈希，无需访问私有 `blind_map.json`。
   - `blindtest.py unblind` 强制核对所有样本全部 6 个维度已完整填写，并复核 `listen/*.wav` 的 SHA256 与 prepare 时一致（防试听音频被替换），通过后才输出揭盲表并写入 `unblinded.json` / `unblinded.md`。
   - 严禁由 Agent 编造人工评分；测试中的固定分值仅作工作流回归 fixture，必须明确标注。

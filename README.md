# tts-voice-intelligence

独立、标准 Skill 结构的本地语音系统：**意图优先，能力约束，证据交付**。
`SKILL.md` 是唯一决策基准；无平台专属路由。包含 Piper、eSpeak NG 与本地授权 XTTS 薄适配器。

## 状态（不要把实现等同听感认证）

- **Piper**：真实中文集成与 7 类重点场景（普通中文、数字日期、中英混合、专名、多音字、叙事转折、长文本）重复 ASR 评测；模型固定音色，支持逐段语速（`length_scale`）与段间停顿（`pause_after_ms`），无克隆/显式情绪。
- **eSpeak NG**：中、英（US/GB）、德、日真实合成与信号检查；进程组硬超时与取消清理已实测，专业自然度未人评。
- **XTTS**：真实本地 API 调用实现，含单次读取参考绑定、模型/配置加载前后 SHA256 绑定、HMAC-SHA256 授权验签、pickle/ZIP 静态安全筛查与替身契约测试；**缺本地真实模型与合法授权参考语音，真实克隆效果未验证**。
- **生产可靠性**：引擎硬超时（`--timeout`）、取消即停、整组交付原子事务（`transactional_write_bundle`，含母版 WAV + 派生 FLAC/OGG + JSON 报告回滚）、分段断点恢复（`--checkpoint-dir`）、子进程离线隔离（`--isolate-model`）。
- 完整审计与验证证据见 [AUDIT.md](AUDIT.md)、[VALIDATION.md](VALIDATION.md)。源码原仓库未附 LICENSE，软件/数据/模型许可须分别核对。

## 安装与运行

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install piper-tts                      # 可选，用户自备 .onnx/.onnx.json
# 或系统安装 espeak-ng：sudo apt-get install espeak-ng
python scripts/tts.py check --model /path/to/model.onnx
python scripts/tts.py synth --model /path/to/model.onnx --input script.txt \
  --out output/narration.wav --format flac --report output/report.json \
  --checkpoint-dir output/ckpt --timeout 60
python scripts/qa.py output/narration.wav --synthesis-report output/report.json \
  --asr base --report output/qa.json
# 声音导演五步链契约（语境→意图→可听变化→支持控制→结果验证）：
python scripts/tts.py synth --engine espeak --language en-US \
  --text 'This discovery changes everything.' --direction assets/direction.example.json \
  --out output/english.wav --report output/english.json
```

CLI 默认 **minimal**：保留原始动态与静音，超峰时整体衰减；`--direction` 中的 `pause_after_ms` 按节拍精确插入停顿。需旧版逐段 RMS 对齐与固定停顿时显式传 `--processing legacy`。低层 `synthesize_text` 保留 `legacy` 默认以兼容旧调用。
输入最多 10 万字符、输出最多 30 分钟；母版 WAV、派生 FLAC/OGG 与 JSON 报告采用整组事务提交，任一文件失败即全组回滚。

合成退出码：`0` 生成成功（非质量认证）/ `1` 空或无效文本 / `2` 引擎不可用 / `3` 合成或超时失败 / `4` 输入、配置、输出错误。
QA 退出码：`0` 无技术 fail / `1` 存在 fail / `2` 不可读音频或验收输入错误。`warn` 与 `skip` 必须解释，不等于完整通过。

## 回听诊断与物理隔离盲测

```bash
pip install faster-whisper scipy opencc-python-reimplemented  # 可选
python scripts/qa.py output/narration.wav --synthesis-report output/report.json --asr base
python scripts/blindtest.py prepare --audio output/narration.wav output/english.wav \
  --out listening --ratings-dir listening/ratings --map-file private/blind_map.json \
  --isolate --seed 17
# 仅向人类评员分发 listening/listen/ 与 listening/ratings/ratings.csv，绝不分发 private/blind_map.json
python scripts/blindtest.py score --pkg listening --ratings-dir listening/ratings
python scripts/blindtest.py unblind --pkg listening --ratings-dir listening/ratings \
  --map-file private/blind_map.json --out listening/ratings/unblinded.json
```

ASR/CER 仅作诊断代理，受同音字与数字转写影响，不等同听感认证；不得由 Agent 编造人工评分或克隆相似度。

## 测试与复现

```bash
python -m pytest -q                         # 无 Piper 模型时明确 skip 8 项 Piper 与 1 项 XTTS
TTS_PIPER_MODEL=/path/to/model.onnx python -m pytest -q
bash validation/reproduce.sh /path/to/model.onnx new-validation private-out
```

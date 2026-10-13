# 评测口径与当前仓库可复现性

**当前 checkout 不包含 `validation/` 目录、历史 WAV、历史 JSON 报告或任何神经模型资产。** 因而较早文档中引用的 Piper/eSpeak CER、RTF、真实音频数量、盲听包和长文本 RSS 等数字，不能由此 checkout 复核；这里不把它们当作当前证据，也不继续引用。当前可复现入口是 `tests/` 与 `tests/cases.jsonl`。

## 当前环境检查（2026-10-13）

- Python 3.13.16、NumPy、SoundFile、SciPy 与 pytest 已安装。
- `piper-tts`、`espeak-ng`、`faster-whisper`、OpenCC、`pyloudnorm` 未安装；没有设置 `TTS_PIPER_MODEL` 或真实 XTTS 授权资产变量。
- 因此当前环境只能验证无模型单元/契约/合成 fixture 测试；不得把 fixture 当作真实引擎音质证据。真实 Piper/eSpeak/XTTS 合成、ASR 诊断、克隆效果和人工盲听均未执行。

## 可复现命令

```bash
# 无模型回归测试；skip 是未执行，不是通过
python3 -m pytest -q -rs

# Piper 本地模型集成测试（保留兼容路径）
TTS_PIPER_MODEL=/path/to/model.onnx python3 -m pytest -q tests/test_piper_integration.py
python3 tests/run_eval.py --engine piper --model /path/to/model.onnx \
  --out validation/piper-run --repeat 3

# 固定中文语音首次资产供给（需要安装 requirements-piper.txt、审阅许可并联网）
python3 scripts/tts.py check --piper-voice zh_CN-chaowen-medium --accept-model-license

# eSpeak NG：需要先安装系统包；地区与用例须匹配
python3 tests/run_eval.py --engine espeak --language en-US \
  --out validation/espeak-run --repeat 3

# XTTS：六项都由用户提供；授权与模型许可需先由真人核对，测试不证明克隆相似度
TTS_XTTS_MODEL=/local/model.pth TTS_XTTS_CONFIG=/local/config.json \
TTS_XTTS_LANGUAGE=en TTS_XTTS_REFERENCE=/local/authorized-reference.wav \
TTS_XTTS_AUTHORIZATION=/local/authorization.json TTS_XTTS_PURPOSE=approved-purpose \
python3 -m pytest -q tests/test_xtts_integration.py
```

`tests/run_eval.py` 要求输出目录不存在或为空；它生成的客观信号/ASR 代理不等于主观自然度、发音真值或统计显著性。`--asr` 可能需要下载模型；没有安装或未完成时必须记录为 `skip`，不能记为通过。人类听评使用 `scripts/blindtest.py`，由真实评员评分，不得由脚本/Agent 代填。

## 指标边界

- `active_signal_seconds`：10ms 帧 RMS > -50 dBFS 的信号活动代理，不是语音学 VAD。
- `speech_rate_chars_per_s`：参考字符数/活动信号秒；`delivery_rate_chars_per_s` 包含静音。
- `asr_cer`：识别诊断代理，受同音字、数字格式、专名与 ASR 模型影响；不证明听感或发音真值。
- QA 的阈值属于未听评校准的工程启发式；`skip`、`warn` 均不等于完整通过。
- 情绪、重音、克隆相似度、跨段自然度和专业旁白听感必须分别标为不支持或未验证，直到真实证据取得。

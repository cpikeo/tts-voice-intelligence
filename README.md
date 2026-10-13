# tts-voice-intelligence

独立、标准 Skill 结构的本地语音系统：**意图优先，能力约束，证据交付**。
`SKILL.md` 是唯一决策基准；无平台专属路由。保留 Piper，新增薄 eSpeak NG 与本地授权 XTTS 适配器。

## 状态（不要把实现等同听感认证）

- Piper：本次真实中文集成与重复ASR评测；模型固定音色，无克隆/显式情绪。
- eSpeak NG：本次中英德日真实合成和信号检查；语言依本机 voices，专业自然度未人评。
- XTTS：真实本地 API 调用实现，授权/参考/模型绑定与契约测试；**缺本地模型、有效授权语音，真实克隆未验证**。
- WAV母版 + 可选FLAC/OGG，无MP3；GPU、目标音色相似性、专业自然度未验证。
- 完整问题与证据见 [AUDIT.md](AUDIT.md)、[VALIDATION.md](VALIDATION.md)。源码原仓库没有 LICENSE，不擅自赋予分发许可；软件/数据/模型许可分别核对。

## 安装与运行

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install piper-tts                      # 可选，用户自备 .onnx/.onnx.json
# 或系统安装 espeak-ng：sudo apt-get install espeak-ng
python scripts/tts.py check --model /path/to/model.onnx
python scripts/tts.py synth --model /path/to/model.onnx --input script.txt \
  --out output/narration.wav --report output/report.json
python scripts/qa.py output/narration.wav --synthesis-report output/report.json \
  --report output/qa.json
# 无模型的另一真实引擎：
python scripts/tts.py synth --engine espeak --language en-US \
  --text 'This discovery changes everything.' --direction assets/direction.example.json \
  --out output/english.wav --report output/english.json
```

CLI新默认 **minimal**：保留原始动态/静音，必要时整体衰减峰值，不逐段归一化。旧处理可用 `--processing legacy`，此时 `--target-db/--pause-*` 才实际用于处理。低层 `synthesize_text` 保留 legacy 默认；新增 language、processing、direction 为可选关键词，旧调用仍可用。
地区语言不得静默回退；Piper默认取模型。非中文保留数字给引擎前端，歧义由用户确认朗读稿，不承诺自动多语言消歧。
输入最多10万字符、输出最多30分钟，音频仍驻留内存，更长请分章节；无断点续合成保证。
母版必须 .wav；每文件原子写入，整组派生输出不保证事务。报告失败不得用旧产物冒充本次结果。

合成退出码：0生成成功（非质量认证）/1空或无效文本/2引擎不可用/3合成失败/4输入、配置、输出错误。
QA：0无技术fail/1存在fail/2不可读音频或验收输入错误。warn与skip必须解释，不等于完整通过。

## 回听与人工评价

```bash
pip install faster-whisper scipy opencc-python-reimplemented  # 可选，首次ASR可联网下载
python scripts/qa.py output/narration.wav --synthesis-report output/report.json --asr base
python scripts/blindtest.py prepare --audio output/narration.wav output/english.wav --out listening --seed 7
# 只发 listen/ 与 ratings.csv 给评员，不发 blind_map.json；按 listen/order.txt 试听
python scripts/blindtest.py score --pkg listening
python scripts/blindtest.py unblind --pkg listening           # 完整人评后
```

CER不等于自然度，RMS不等于音色/感知响度；字符速度基于有效信号秒，不含静音，非语音VAD。
未ASR/未人评必须明示，指标口径见 `references/eval-results.md`。
授权克隆安装、门禁模板、隐私与许可边界见 `references/engines.md`；不能给Piper/eSpeak传参考音频冒充克隆。

## 测试与评测

```bash
pip install pytest
python -m pytest -q                         # 无Piper模型时明确skip；eSpeak存在时运行真实测试
TTS_PIPER_MODEL=/path/to/model.onnx python -m pytest -q
python tests/run_eval.py --model /path/to/model.onnx --out eval-piper \
  --ids digits_date,mixed_en --repeat 3 --asr base --lexicon assets/lexicon.example.tsv
python tests/run_eval.py --engine espeak --language de --out eval-de --repeat 3
```

评测输出目录须为空，防止混淆旧证据；按实际locale/克隆能力过滤用例并记录skip。
每次原始指标、哈希、错误、警告保存在 summary.json/runs.json；ASR缺失不崩溃也不虚报0。
基础依赖只有numpy/soundfile；可选组件版本快照在 validation/environment.json，不是所有平台兼容承诺。

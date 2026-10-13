# tts-voice-intelligence

把文本变成可交付的语音文件。`SKILL.md` 是唯一的决策基准；`scripts/` 只负责执行，
`references/` 按需读取。

## 最短安装

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # numpy + soundfile，必装
pip install piper-tts                    # 引擎（GPL-3.0-or-later）
```

引擎需要一个自备的 Piper 模型（`*.onnx` 与同名 `*.onnx.json`，见 `references/engines.md` 第 3 节）。
中文 `xiao_ya` 模型还需要 g2pW 前端依赖，见 `requirements.txt` 注释。

不需要 GPU，不需要 ffmpeg。**不支持 MP3**（需额外编码器）。

## 最短运行

```bash
python scripts/tts.py check --model /path/to/model.onnx      # 先确认引擎可用
python scripts/tts.py synth --model /path/to/model.onnx \
  --input script.txt --lexicon assets/lexicon.example.tsv \
  --out out/narration.wav --report out/report.json
python scripts/qa.py out/narration.wav --text-file script.txt \
  --lexicon assets/lexicon.example.tsv --report out/qa.json
```

退出码：`tts.py` 0 成功 / 1 输入为空 / 2 引擎不可用 / 3 合成失败 / 4 输出参数错误；
`qa.py` 0 无 fail / 1 存在 fail / 2 文件不可读。

## 回听识别（可选）

```bash
pip install faster-whisper scipy opencc-python-reimplemented
python scripts/qa.py out/narration.wav --text-file script.txt --asr small
```

未启用回听时，交付说明必须写明“未做回听验证”。

## 人工听评（盲听，可选但推荐）

客观指标与 ASR 都不能替代人耳：

```bash
python scripts/blindtest.py prepare --audio out/*.wav --out pkg --seed 7
# 试听 pkg/listen/ 下的匿名样本，填写 pkg/ratings.csv（1–5 分）
python scripts/blindtest.py score   --pkg pkg     # 汇总
python scripts/blindtest.py unblind --pkg pkg     # 评完再揭示对应关系
```

## 测试

```bash
pip install pytest
python -m pytest -q                                            # 单元测试，无需模型
TTS_PIPER_MODEL=/path/to/model.onnx python -m pytest -q        # 含真实引擎集成测试

python tests/run_eval.py --model /path/to/model.onnx --out eval_out \
  --asr small --lexicon assets/lexicon.example.tsv --repeat 3  # 真实引擎评测
```

## 许可

软件与模型数据的许可不同，需分别确认：`piper-tts` 为 GPL-3.0-or-later；
常见中文 Piper 模型的数据许可为「非商用」或「未知」，商用前必须确认授权。
本仓库不内置、不分发任何模型权重。详见 `references/engines.md` 第 1、3 节。

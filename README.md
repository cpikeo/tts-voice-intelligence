# tts-voice-intelligence

本地多引擎 TTS 与音频验收工具：**先理解意图，再遵守引擎能力，最后以证据交付**。`SKILL.md` 是唯一决策基准；`scripts/` 实现确定性执行，`references/` 记录边界与复现方法。

## 能力与当前验证边界

- 代码包含 Piper、eSpeak NG 与本地授权 XTTS 适配器，支持中文规范化/词典、语义分段、声音策略、音频 QA、断点恢复和交付事务。
- 仓库不内置模型权重、授权参考音频或静态 `validation/` 评测产物。Piper 支持从仓库内固定目录首次联网获取已审核模型与中文前端资产，并按 SHA-256 校验、缓存；目前目录只含一个单说话人候选，许可与试听边界见 [references/engines.md](references/engines.md)。其他 Piper 模型可走本地路径；XTTS 仍需本地权重与授权。真实引擎效果、实际生成 RTF、ASR 回听与人工盲听必须在准备好相应模型/工具后单独运行，不能由单元测试代替。
- Piper 固定模型音色；Piper/eSpeak 支持实现范围内的逐段语速和边界停顿；XTTS 调用仍受模型、语言和授权限制。代码不宣称未实现或未验证的情绪、重音、自然度或克隆相似度。
- 当前性能审计、可复现的 fixture 基准、测试结果及限制见 [OPTIMIZATION.md](OPTIMIZATION.md)；当前仓库评测入口与资产边界见 [references/eval-results.md](references/eval-results.md)。
- 软件、模型、数据许可须分开核对；源码仓库未附 `LICENSE`，不要假定可再分发或商用。

## 安装

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install pytest                         # 可选：运行测试
pip install -r requirements-piper.txt       # 可选：Piper + 中文 pinyin 前端依赖；模型首次由固定目录自动下载
sudo apt-get install espeak-ng             # 可选：eSpeak NG 系统程序
# XTTS、ASR 及额外检查为可选组件，见 references/engines.md
```

## 合成与验收

```bash
# 本地模型（兼容路径）
python scripts/tts.py check --model /path/to/model.onnx
python scripts/tts.py synth --model /path/to/model.onnx --input script.txt \
  --out output/narration.wav --format flac --report output/report.json \
  --checkpoint-dir output/ckpt --timeout 60

# 固定审核目录中的 Piper 中文音色：首用先审阅许可，再显式确认；自动下载并缓存
python scripts/tts.py check --piper-voice zh_CN-chaowen-medium --accept-model-license
python scripts/tts.py synth --piper-voice zh_CN-chaowen-medium --text '试听用句。' \
  --out output/chaowen_audition.wav --report output/chaowen_audition.json
python scripts/qa.py output/narration.wav --synthesis-report output/report.json \
  --asr base --report output/qa.json

# eSpeak 必须明确指定地区，不静默降级
python scripts/tts.py synth --engine espeak --language en-US \
  --text 'This discovery changes everything.' --direction assets/direction.example.json \
  --out output/english.wav --report output/english.json
```

自动下载只认 `assets/piper_voices.json` 中的固定提交与逐文件 SHA-256；模型、g2pW ONNX/配置/查词表及本地 BERT tokenizer 完整验证后才原子落盘。首次获取约 178 MB 网络资产、最终缓存约 225 MB；解包暂存峰值约 338 MB（十进制估算，另计已有旧缓存）。默认缓存为 `~/.cache/arena-tts/piper`，可用 `--piper-cache-dir` 指定。无网络且缓存缺失/损坏时明确失败，不跳段、不切换引擎。当前仅登记单说话人 `zh_CN-chaowen-medium`，不是 EP001 双角色的已确认方案；即使两个角色都用它，音色也相同，须先试听并由用户确认。

CLI 默认 `minimal`：保留引擎段落动态、不逐段响度归一；默认段间停顿为段落/句子/子句 700/400/200 ms，`--pause-* 0` 可关闭，direction beat 可覆盖指定边界。只在点击风险超过阈值时修复边界，只在超峰时整体衰减。`--processing legacy` 才启用旧式逐段 RMS 对齐与淡变。低层 `synthesize_text` 的 `legacy` 默认仅为兼容旧调用。

`--timeout` 对模型加载与每段调用生效，但进程内 Piper/XTTS 的原生推理不保证能被立即中断；eSpeak 子进程与 `--isolate-model` 子进程组实现硬终止。输入最多 100,000 字符、输出最多 30 分钟，全音频驻留内存，不声称流式或无限长。

合成退出码：`0` 生成成功（不等于质量认证）、`1` 空/无效文本、`2` 引擎不可用、`3` 合成/超时失败、`4` 输入/配置/输出错误。QA 退出码：`0` 无技术 fail、`1` 存在 fail、`2` 音频或验收输入错误；`warn` 和 `skip` 需解释，不等于完整通过。

## 人工盲听

```bash
python scripts/blindtest.py prepare --audio output/narration.wav output/english.wav \
  --out listening/pkg --ratings-dir listening/pkg/ratings \
  --map-file private/blind_map.json --isolate --seed 17
# 只把 listening/pkg/listen/ 与 ratings.csv 发给评员；不要分发私有映射
python scripts/blindtest.py score --pkg listening/pkg --ratings-dir listening/pkg/ratings
python scripts/blindtest.py unblind --pkg listening/pkg --ratings-dir listening/pkg/ratings \
  --map-file private/blind_map.json --out listening/pkg/ratings/unblinded.json
```

ASR/CER 只是诊断代理，不等同听感认证；不得由 Agent 编造人工评分或克隆相似度。

## 测试与真实引擎复现

```bash
python3 -m pytest -q -rs
TTS_PIPER_MODEL=/path/to/model.onnx python3 -m pytest -q tests/test_piper_integration.py
python3 tests/run_eval.py --engine piper --model /path/to/model.onnx \
  --out validation/piper-run --repeat 3
```

Piper/eSpeak/XTTS 的完整复现条件、当前缺失资产和 skip 边界见 [references/eval-results.md](references/eval-results.md)。

# 全仓性能审计与优化记录

## 审计范围、基线与限制

- 基线是本次 clone 的干净 `main`：`5613c33`（2026-10-13）。优化只针对当前树中的热点和失效文档；没有加载其他分支代码或模型资产。
- 环境：Python 3.13.16、NumPy 2.3.5、SoundFile 0.13.1、SciPy 1.17.1、Linux x86_64。基线与优化版使用同一环境和同一 `tests/benchmark_hotpaths.py`，通过 `--scripts-dir` 切换待测源码。
- 本机没有 `piper-tts`、Piper 模型、`espeak-ng`、XTTS 本地模型/有效授权、`faster-whisper`、OpenCC 或 `pyloudnorm`。**真实模型加载耗时、真实合成 RTF、ASR、授权克隆效果与人工盲听没有执行**；fixture 数字不能当作真实 TTS 速度或音质结论。
- 优化前实际全套测试为 `164 passed, 20 skipped`；仓库原 README 写的 `173 passed / 11 skipped` 与当前测试树不符。

## 问题清单与排序

| 优先级 | 基线问题 / 证据 | 风险/收益判断 |
|---|---|---|
| P1 | Piper 流水线每句都对磁盘上的整个模型和配置做完整 SHA256；64 段即重复 64 次模型哈希。推理实际使用已加载的内存模型。 | 对大模型/长稿造成与语音计算无关的读盘与哈希耗时；可按已加载模型生命周期减少重复，但必须保留加载前后与批次边界的完整校验。 |
| P1 | 512 段基线 `cProfile`：`active_frames` 1,537 次、`voiced_rms_db` 1,024 次、`active_seconds` 513 次；同一段重复计算 RMS/活动帧。 | 统计定义不需要改变，活动帧、活动秒与有声 RMS 可从同一组帧能量得到。优化后为 `frame_activity` 1,025 次、全局 `active_seconds` 1 次；帧分析调用数减少 33.3%。 |
| P1 | `boundary_repair` 在无修复时仍复制整段音频，重复扫描静音边界；最终拼接后保留片段列表，峰值衰减与 `astype(float32)` 又复制完整母版。 | 长稿峰值内存主要来自可避免的全长副本，不应通过缩短/丢段解决。 |
| P2 | QA 对活动帧与帧 RMS 重复扫描；点击判定构建多份全长临时数组；`load_mono` 的 `sf.info` + `sf.read` 打开两次，立体声 QA 又完整解码一次。 | 可复用同批帧能量、原位复用短时分析缓冲区、一次解码保留原通道元数据。必要真峰值与底噪检查不跳过。 |
| P2 | README 指向已删除的 `AUDIT.md` / `VALIDATION.md` 与 `validation/reproduce.sh`；评测文档声称引用的 WAV/JSON 已不在当前树；默认停顿/超时/验证描述互相矛盾。 | 文档不具备当前 checkout 可复现性，会导致错误验收或无效调用。 |

## 实际改动

### 1. Piper 资产完整性：按加载生命周期减少重复哈希

- `PiperEngine` 加载后仍执行模型与配置完整 SHA256；`synthesize_text` 对整个合成批次开始、结束各执行一次完整 SHA256。批次内句段走私有 `_synthesize_loaded`，只调用已经加载到内存的 Piper 会话，不为每句重读磁盘模型。
- 直接单段 `PiperEngine.synthesize()` 保留完整前/后校验；`tests/ab_processing.py` 的裸段捕获也显式做批次前/后完整复核。
- `verify_bound_asset` **始终重新计算 SHA256**，不使用仅基于 mtime/ctime 的缓存；同尺寸原地改写仍会被测试检测。`_read_and_fingerprint` 改用递增字节计数，避免每个 1 MiB 块重新 `sum()` 已读块长度，并核对打开文件、路径和读前后状态。
- 已加载模型内容是本批次实际推理输入；批次结束完整复核失败则不返回可交付音频。失败/取消不跳过安全检查后静默交付。

### 2. 合并重复音频分析与减少全长副本

- 新增 `qa.frame_activity()`，一次帧能量分析返回帧 RMS、活动掩码、活动秒和活动 RMS；使用实际尾帧长度，不再构造采样级 `repeat(mask)`。
- `synthesize_text` 将最终段 RMS 与活动秒合并为同一次帧分析；保留整体主峰处理之前用于告警的段级 RMS 扫描，以维持旧告警语义。方向验证与信号阈值未放宽。
- `boundary_repair` 先扫描边界再决定是否复制；无修复时不做不必要的全长副本。合成流水线只对自有的 `np.concatenate` 缓冲区使用 `copy=False`，该函数默认 API 仍为非原位，回归测试校验两种模式逐样本相同。
- 拼接后释放分段数组引用；峰值由 min/max 计算；整体衰减原位执行；已知为 float32 的成品用 `astype(copy=False)` 返回。
- QA 点击判定在一个复用数组中计算邻点差与平方前缀和；边界/峰值/削波统计不再创建不必要的全长 `abs` 音频副本。

### 3. QA I/O 与报告修正

- `load_mono` 用一次 `SoundFile` 打开并解码，保留通道/子类型等元数据；多声道削波率直接在同一解码缓冲上测，不再在 `qa.main` 二次读取立体声文件。
- 单独的 `--synthesis-report` 输出哈希读取仍保留：它是报告与待验收音频绑定的必要复核，不因解码复用而省略。
- README、`SKILL.md`、引擎/评测/导演参考已同步 Piper 批次 SHA 边界、eSpeak 当前未安装、Piper 原生推理超时不保证立即中断、默认停顿与 `minimal` 行为；撤回当前树无法复核的历史数字与失效链接。

### 4. 新增可复现基准与回归

- `tests/benchmark_hotpaths.py` 提供无模型 pipeline、QA、SHA256 文件、SoundFile 打开次数四种 fixture 基准。它明确标注 `real_engine: false` / `synthesis_rtf: null`，不伪装真实模型性能。
- 新增帧活动尾帧等价、资产同尺寸篡改检测、边界原位/非原位逐样本等价、Piper 批次前后完整校验及主峰衰减警告语义回归测试。

## 每项改动的收益、质量影响与回滚映射

| 优化项 | 预期收益与实测结果 | 质量/可靠性影响 | 单项回滚范围 |
|---|---|---|---|
| Piper 批次完整 SHA256 | 消除句间重复的大文件扫描；对单个 32 MiB 资产，fixture 从 64 次/2,048 MiB 降至每批 2 次/64 MiB，哈希耗时少 96.9%（模型与配置各自仍前后校验）。 | 加载前/后、批次前/后、直接单段前/后仍做完整 SHA256；同尺寸篡改测试拒绝通过；fixture 波形/报告逐项一致。 | 回滚 `scripts/engines.py`、`scripts/tts.py`、`tests/ab_processing.py` 与对应 Piper 回归测试。 |
| 合成分析与整段缓冲区 | 64 段耗时少 80.0%；512 段少 75.2%，peak RSS 少 66.7%；profile 帧分析调用数少 33.3%。 | 信号门限、分段和主峰规则不变；基线/当前 fixture bitwise-identical，含 126 次边界修复及同报告。 | 回滚 `scripts/qa.py`、`scripts/tts.py` 及帧/边界/峰值回归测试。 |
| QA 解码与分析临时量 | 约 150 秒 fixture QA 少 16.5%；SoundFile 打开数单声道 2→1、立体声全检 3→1。 | 同一 QA fixture 各项检查名称、状态和值完全一致；不跳过多声道削波、真峰值、底噪或静音点击检查。 | 回滚 `scripts/qa.py`；解码次数由 `tests/benchmark_hotpaths.py --mode io` 复核。 |
| 运行文档与 harness | 修复失效链接/陈旧验收说明；一次命令复测无模型热点。 | 文档明确 fixture 不代表真实 RTF/自然度，skip 不算通过。 | 回滚 `README.md`、`SKILL.md`、`references/*.md`、`OPTIMIZATION.md` 与 `tests/benchmark_hotpaths.py`。 |

## 优化前后测量

所有 wall/RSS 数字均来自同一 fixture、同一环境；pipeline 中的引擎每段只返回 1 秒固定正弦，不代表推理速度。每次耗时仅包括代码/QA 路径，**不是 TTS 合成 RTF**。

| 基准 | 基线（`5613c33`） | 优化后 | 变化 |
|---|---:|---:|---:|
| 64 段 pipeline，输出 89.2 秒；7 次中位 wall | 104.388 ms | 20.854 ms | 少 80.0%，约 5.01× 快 |
| 512 段 pipeline，输出 716.4 秒；单次 wall | 984.937 ms | 244.106 ms | 少 75.2%，约 4.03× 快 |
| 同一 512 段、60.26 MiB float32 输出；进程 peak RSS | 486,232 KiB（474.8 MiB） | 162,108 KiB（158.3 MiB） | 少 324,124 KiB（316.5 MiB，66.7%） |
| 约 150 秒 / 13.23 MB 合成正弦+静音 QA；6 次中位 wall | 371.457 ms | 310.305 ms | 少 16.5%；两边均 0 fail / 1 warn / 2 skip |
| 同一 32 MiB 文件的 SHA256 验证；64 次逐段检查 vs 2 次批次检查 | 2.189065 s，2,048 MiB 重复读取/哈希 | 0.067071 s，64 MiB | 少 96.9% 检查耗时与哈希字节 |
| SoundFile API 打开数（不含合成报告哈希读取） | `load_mono` 单/立体声各 2 次；立体声整体验收 3 次 | 各 1 次；立体声整体验收 1 次 | 解码器打开数减少 50% / 66.7% |

QA fixture 的 2 个 `skip` 是未安装 `pyloudnorm` 的 LUFS 检查与连续静音不足以判定嗡声；`warn` 来自此固定信号的启发式峰值阈值，不能当作真实语音缺陷。

### 明确不可测/未引入的指标

- **缓存命中率：不适用**——没有内容哈希缓存，也没有基于时间戳的“命中”；每个显式 `verify_bound_asset()` 调用都会重新读文件并做完整 SHA256。优化的是调用次数，不是跳过校验。
- **失败恢复耗时：未作性能归因**——基线与当前测试合同均要求故障后保留已提交段、重跑时恢复 1 段、只重新合成另外 2 段，并拒绝计划不匹配；没有真实引擎失败，计时由小型 fixture/文件系统主导，因此不报告 TTS 恢复秒数或改善比例。
- **模型加载 / 真实 RTF / 生产磁盘 I/O / ASR / 盲听：未执行**——缺真实模型、eSpeak 程序、ASR 依赖及授权音频。报告的 RSS 和耗时只适用于上述无模型 workload。

### 可复现命令

```bash
# 优化后当前源码
python3 tests/benchmark_hotpaths.py --mode pipeline --segments 64 --repeats 7
python3 tests/benchmark_hotpaths.py --mode pipeline --segments 512 --repeats 1
python3 tests/benchmark_hotpaths.py --mode qa --qa-seconds 150 --repeats 6
python3 tests/benchmark_hotpaths.py --mode asset --asset-mib 32 --segments 64 --repeats 3
python3 tests/benchmark_hotpaths.py --mode io
python3 -m cProfile -o /tmp/tts-current.prof tests/benchmark_hotpaths.py \
  --mode pipeline --segments 512 --repeats 1

# 对同一 harness 使用基线源码（从当前 Git 历史抽取）
mkdir -p /tmp/tts-baseline && git archive 5613c33 | tar -x -C /tmp/tts-baseline
python3 tests/benchmark_hotpaths.py --mode pipeline \
  --scripts-dir /tmp/tts-baseline/scripts --segments 64 --repeats 7
python3 tests/benchmark_hotpaths.py --mode pipeline \
  --scripts-dir /tmp/tts-baseline/scripts --segments 512 --repeats 1
python3 tests/benchmark_hotpaths.py --mode qa \
  --scripts-dir /tmp/tts-baseline/scripts --qa-seconds 150 --repeats 6
python3 tests/benchmark_hotpaths.py --mode io --scripts-dir /tmp/tts-baseline/scripts
python3 -m cProfile -o /tmp/tts-baseline.prof tests/benchmark_hotpaths.py \
  --mode pipeline --scripts-dir /tmp/tts-baseline/scripts --segments 512 --repeats 1
```

SHA256 fixture 模式在同一实现上计时 64 次与 2 次完整验证的工作量差；它不是 Piper 模型加载基准。表中 MiB 是程序级累计读取/哈希字节，文件可能已在 OS page cache 中，不等同物理磁盘 I/O。I/O 打开次数由 SoundFile 构造计数得出，不是块设备/内核级 `strace` 统计。当前没有真实模型，不能报告模型加载、缓存命中率或真实合成 RTF；项目没有引入内容哈希缓存。

## 质量、安全、失败恢复与验收结果

- 基线与优化版在同一 64 段确定性 fixture 上生成 **逐样本 bitwise-identical** 音频；输出 1,966,860 个 float32 采样、89.2 秒、126 次相同边界修复。`duration_seconds`、活动秒、峰值、段 RMS 标准差、修复前/后跳变、warnings 全部相同。
- 同一 WAV 的 QA check 名称、状态和值均与基线完全相同；这证明实现路径与阈值未改变 fixture 结果，不证明真实语音自然度。
- 真实主观听评、ASR、Piper/eSpeak/XTTS 集成因依赖/模型/授权资产缺失未执行；`skip` 不记作通过。
- 断点恢复与参数不匹配拒绝、超时不重试、取消、整组事务回滚、授权/资产安全、数字/多音字/方向契约均仍由现有回归测试覆盖。没有降低 QA 阈值，没有跳段、缩短音频或放宽授权/哈希检查。

## 当前测试结果（本次优化树）

`python3 -m pytest -q -rs`：**176 passed, 20 skipped**。跳过原因均为可选资产/依赖：10 个 Piper 模型集成用例未设置 `TTS_PIPER_MODEL`；XTTS 真实模型/授权测试缺少六个本地环境变量；eSpeak NG 集成缺系统程序；可选 LUFS 测试缺 `pyloudnorm`。未执行项不是通过。

## 未解决问题与回滚

- 未取得授权 Piper 权重、eSpeak NG 和 XTTS 模型/授权；真实端到端速度、启动/加载时间、ASR、盲听自然度、GPU 和 XTTS 相似度均未验证。
- 当前 Piper 批次级哈希减少的是同一已加载内存模型每句重复读盘；若有新的模型热加载、lazy-load 或磁盘外部资源依赖，应重新审查批次边界，不应直接套用本复用条件。
- 未做内核块 I/O、硬件计数器或跨机器基准；fixture 用于比较代码热点，绝不等同生产吞吐。
- 当前未提交变更。此 checkout 原本干净；如需回滚，先保留 tracked diff 与两个新增文件，再还原下列修改文件：

```bash
git diff --binary > /tmp/tts-voice-optimization.patch
cp OPTIMIZATION.md /tmp/tts-voice-OPTIMIZATION.md
cp tests/benchmark_hotpaths.py /tmp/tts-benchmark_hotpaths.py
git checkout -- README.md SKILL.md references/engines.md references/eval-results.md \
  references/voice-direction.md scripts/engines.py scripts/qa.py scripts/tts.py \
  tests/ab_processing.py tests/test_audio_quality.py tests/test_intelligence.py tests/test_pipeline.py
rm -f OPTIMIZATION.md tests/benchmark_hotpaths.py
```

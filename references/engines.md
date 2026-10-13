# 引擎、依赖、安全隔离与能力边界

决策以 `SKILL.md` 为准；适配器在 `scripts/engines.py`。**实现存在不等于本机模型效果通过验收**：当前 checkout 不附带 Piper/XTTS 权重、授权参考音频或 `validation/` 评测产物；eSpeak NG 也不是基础依赖。

| 引擎 | 代码与可复现验证 | 语言/地区 | 可用控制与能力边界 | 超时与资产复核 | 许可 |
|---|---|---|---|---|---|
| Piper | ONNX 适配器已实现；新增首次资产供给只认 `assets/piper_voices.json` 固定目录。真实集成测试由本地模型环境变量门控；当前环境缺 `piper-tts` 和模型，未执行 | 读取模型配置 locale；不跨地区静默回退 | 固定模型音色/配置内 speaker id；逐段 `length_scale` 与管线停顿；无情绪/重音/音色调制或克隆接口 | 托管音色的 ONNX/JSON/g2pW/tokenizer 资产均先 SHA256 校验；加载前/后及合成批次开始/结束完整复核；段内复用已加载会话。Pinyin 模型缺固定前端资产时拒绝加载，不触发动态下载 | `piper-tts` GPL-3.0-or-later；托管 Chaowen 源仓库元数据 MIT、声卡训练数据 CC0，但权重许可未在 voice card 单独写明； opt-in、先人工审阅 |
| eSpeak NG | 子进程适配器已实现；集成用例仅在检测到 `espeak-ng` 时运行，当前环境未安装，未执行 | 从本机 `--voices` 查询；显式地区不回退 | 地区默认 voice；`length_scale` 映射为 WPM；无神经音色/情绪/重音控制、无克隆 | 单段子进程组超时/取消时强制终止；当前环境未实测系统级行为 | 程序 GPL-3.0-or-later；随系统语音数据许可另核对 |
| XTTS | 本地 Coqui API 适配器；合同/授权测试使用替身，真实模型与克隆效果未验证 | 必须同时匹配 XTTS 配置与授权允许语言；不静默地区降级 | 授权参考音色；只提供管线停顿，不暴露语速、情绪或重音控制 | 当前进程内超时不可保证立即中断；每段开始重验授权并完整校验模型/配置/参考，结束复核参考。真实超时、资源占用与克隆效果未验证 | 包与权重许可须分别确认；XTTS 权重可能有非商业限制，不能默认商用 |

## 最小依赖与可选组件

```bash
pip install -r requirements.txt
pip install -r requirements-piper.txt   # 可选：piper-tts[zh] 固定为 1.8.0
sudo apt-get install espeak-ng           # 可选系统程序
pip install coqui-tts                    # 可选独立环境；需要本地 XTTS 资产与有效授权
pip install faster-whisper scipy opencc-python-reimplemented  # 可选 QA/ASR
```

模型权重不应放入仓库；软件、数据与每份模型/语音许可须分别核对。本仓库未附 `LICENSE`，不要推断源码的再分发许可。

## Piper 首次联网自动供给（不需要用户手动给 ONNX/JSON）

```bash
# requirements-piper.txt 安装完成后，首次运行前逐项审阅许可提示
python scripts/tts.py check --piper-voice zh_CN-chaowen-medium --accept-model-license
# 试听用途示例；是否用于成片须在真实试听和用户确认后决定
python scripts/tts.py synth --piper-voice zh_CN-chaowen-medium \
  --text '这是一段用于试听的短句。' --out output/chaowen_audition.wav \
  --report output/chaowen_audition.json
```

供给目录为仓库内 `assets/piper_voices.json`，运行时代码不请求可变 `voices.json`、任意 Hub ID 或 `main` 分支。

- 唯一当前登记项 `zh_CN-chaowen-medium` 固定到 `rhasspy/piper-voices` commit `10eb5c756ae21b759c8344d54aef86f9399ae92d`；ONNX SHA-256 `820d64ac16048fbcf38dd0823d37fab5f5e0c2bd71b01ca5a50f553fac19e746`（63,221,984 bytes），sidecar SHA-256 `5a8dda00084d3cdbfe4fcfefac5e78935f79af16452c35c70c337bdcef7cc755`（2,699 bytes）。逐资产版本、来源、大小、摘要与许可证据均在清单中；缓存内保留经哈希校验的许可快照和可由清单重建的 `provisioning.json`。缓存复用时重新校验小型许可快照，registry 始终是来源事实。
- 该 Pinyin voice 还依赖中文前端。清单同时固定 `rhasspy/piper-checkpoints` 的 g2pW archive commit/SHA（112,991,734 bytes），固定每个解压成员 SHA；只接受严格白名单普通文件，不使用 `extractall`。g2pW 的 `model_source` 以固定小补丁转成本地 BERT tokenizer 路径；tokenizer 与 g2pW 查词/简繁资源也固定到 commit、逐文件验证。加载前后和每个合成批次边界会校验所有活动模型/前端资产，防止 Piper 自己从可变地址补下载。
- 首次模型与前端下载约 178 MB 网络流量；最终缓存约 225 MB，原子解包期间暂存峰值约 338 MB（十进制估算，另计 pip 依赖与可能保留的旧缓存）。默认放在用户缓存 `~/.cache/arena-tts/piper`，`--piper-cache-dir` 可覆盖；模型不进入仓库。
- 下载必须是 HTTPS、SHA-256 和字节数都匹配后才进缓存；暂存目录与正式目录同一文件系统，整包目录原子替换，失败时旧缓存不动。已有缓存不做重复预哈希，而由 `PiperEngine` 在同一次必要资产绑定读取中校验；同尺寸篡改触发一次同版本整包修复。`--download-timeout` 默认 1200 秒，是整包下载软总时限；单次阻塞网络读取超时 45 秒。`--isolate-model` 的子进程硬超时会为该下载预算增加相应余量；被硬杀留下的 staging 目录会在下次同 voice 获取缓存锁后清理/恢复。若无网络且无可验证缓存，明确失败，不静默用其他引擎。
- 模型许可：固定仓库 README 的 metadata 标 MIT，固定 voice card 的训练数据标 CC0；voice card 没有另外写明权重专属许可。这个不等于法律审查或商用保证，所以没有默认音色，必须明确 `--piper-voice`，首用再 `--accept-model-license`，并由使用者审阅固定来源。许可确认记录只留在本机缓存，绑定 voice/revision/policy id。Xiao Ya（数据集标非商业）及 Huayan（Unknown）不在自动下载目录。
- 当前模型是单说话人；没有满足该项目商业/可能商业边界、且已试听确认的第二个中文音色。不得把一个模型生成的两个角色称为不同音色；EP001 双角色成片仍须等待第二个音色许可核实、试听及用户确认。

## 资产绑定、授权与模型安全检查

1. **同一描述符读取**：`_read_and_fingerprint` 从单个文件描述符读取内容并计算 SHA256，记录设备、inode、大小与时间戳；读取期间若文件/路径状态变化即拒绝。模型安全扫描使用这次读入的字节，避免校验内容与加载前扫描内容脱节。
2. **SHA256 不以文件时间戳替代**：`verify_bound_asset` 每次调用都重新读取并计算完整 SHA256；不使用仅靠 `mtime`/`ctime` 的内容缓存。Piper 优化基于加载生命周期：模型和配置在加载后驻留于内存，因此流水线在批次开始、结束完整复核，不对每个句段重复读取不再参与推理的磁盘文件；结束复核失败时不交付。直接单段适配器调用仍做前后完整复核。
3. **授权真实性校验**：模板占位符拒绝；当要求签名或授权记录含签名时，使用 `hmac.compare_digest` 验证 HMAC-SHA256。没有外部公钥/CA 体系时，本地对称签名只能证明持密钥者签署了载荷，不等同法律身份公证。
4. **不可信模型静态筛查**：加载前扫描配置远程/穿越路径、ONNX 外部路径穿越、ZIP 路径穿越及 `.pth`/`.pt` pickle 高危全局调用。静态扫描不能证明任意第三方模型绝对安全；`--isolate-model` 通过离线环境中的独立子进程组隔离并支持硬终止，**不等同操作系统容器或完整沙箱**。

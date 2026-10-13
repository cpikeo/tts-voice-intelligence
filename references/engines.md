# 引擎、依赖与能力边界

决策以 SKILL.md 为准。适配器在 `scripts/engines.py`；实例 JSON 能力比此矩阵更具体。

| 引擎 | 实现/验证 | 语言与地区 | 音色/说话人 | 语速/情绪 | 克隆 | 硬件/资源 | 许可 |
|---|---|---|---|---|---|---|---|
| Piper | 保留真实实现；本次 huayan 中文集成实测 | 只加载模型配置的 locale；裸语言可匹配同根地区，不能替代另一地区 | 固定模型；配置范围内 speaker id | length_scale；无情绪/精细重音接口 | 不支持 | CPU ONNX；CUDA参数保留但未验证；内存依模型 | piper-tts GPL-3.0-or-later；每模型数据/权重许可单独确认 |
| eSpeak NG | 真实子进程实现；中/英/德/日信号层实测 | 从本机 `--voices` 读取；en 默认 en-gb，zh/zh-CN 映射 cmn；显式地区不回退 | 语言默认 voice；本适配器不暴露角色 speaker | words/min 由 length_scale 映射；无情绪接口 | 不支持 | CPU；无额外神经权重；每段60秒超时 | 软件 GPL-3.0-or-later；随系统语音数据许可另核对 |
| XTTS | 本地 coqui-tts API 实现；仅替身契约和授权门禁测试，真实克隆阻塞 | 用户配置 languages 与授权 allowed_languages 的交集；不把 en-GB 降级 en | 已授权参考音色；当前不暴露内置 speaker | 不暴露语速、情绪或精细重音 | 支持调用，但目标音色保持未实测 | 本地 checkpoint/config/vocab 等资源；torch；CPU/CUDA均未实测 | 包通常 MPL-2.0需核实；XTTS权重许可独立（常见 CPML），不可假定商用 |

没有默认模型排名；旧 huayan/xiao_ya 数字无法从仓库原始产物复核，已撤回确定性结论。eSpeak 的多语言实现不是专业自然度背书。仓库本身未附 LICENSE：不要擅自假设源码可按某许可证分发。

## 最小依赖与可选安装

```bash
pip install -r requirements.txt
pip install piper-tts                    # 本次 1.8.0 / onnxruntime 1.31.0
# eSpeak 是系统包（例 Debian）：
sudo apt-get install espeak-ng           # 本次 1.52.0
# XTTS：在独立、资源充足且版本兼容的环境按该包要求安装
pip install coqui-tts                    # 本次未安装/未验证；不是基础依赖
# ASR 可选，可能首次下载权重：
pip install faster-whisper scipy opencc-python-reimplemented
```

Piper 用户自备 `.onnx` + 同名 `.onnx.json`。本次非商业技术测试 huayan 模型卡仍为 dataset License: Unknown，不能据此商业交付。可从官方 `rhasspy/piper-voices` 查看模型卡；不要将下载权重放入仓库。本次 SHA/配置见 validation/piper-repeat/summary.json。
xiao_ya 前端可能需 `unicode-rbnf sentence-stream transformers tokenizers` 与 `pip install --no-deps g2pw`，Piper 将额外资源下载到模型目录；本次未测试该模型。离线或敏感部署请预置并核实资源，注意安装/首次前端初始化可能联网。XTTS API仅传本地 model_path/config_path，不指定 model_name，不主动下载克隆权重；仍需完整本地辅助资源，勿加载不可信 torch checkpoint。

## 扩展约定（单层）

继承 Engine：`sample_rate`（int），`language`，实例 `capabilities`，`available()`，`synthesize(text)` 返回非空有限 float32 单声道 [-1,1]；可诊断执行故障抛 EngineError。在 `tts.build_engine` 注册即可，不新增路由层。能力须描述语言/voice/speaker/速率/情绪/克隆/硬件/依赖/许可/资源；未实现的控制拒绝，不保留静默无效参数。新引擎自己包装可恢复异常；程序错误不能无限重试。

## XTTS 授权与本地资产

模板 `assets/authorization.example.json` 默认拒绝，必须人工核实并填写。本地授权不是电子签名认证，不能自动验证签署人身份或法律效力；不应伪造证据。参考检查收录门槛单声道、≥16kHz、3–30秒、无削波、活动≥2秒且占比≥50%是启发式，不能证明单说话人或净语音；人工确认才设置 reviewed。

```bash
python scripts/tts.py synth --engine xtts --model /private/model.pth \
  --config /private/config.json --language zh-CN \
  --reference-audio /private/reference.wav --authorization /private/grant.json \
  --purpose approved-narration --input script.txt --out output/clone.wav \
  --report output/clone.json
```

每段重新检查授权/撤销/有效期/参考绑定；模型哈希在加载前绑定，语言同时检查配置与授权。固定参考是跨段一致性的必要条件，不是充分条件。参考、授权、声纹嵌入不入库；报告只存 record_id/哈希/用途/有效期等最小摘要，报告仍可能含业务文本与本地模型路径，应私有保存。私有目录权限、磁盘加密、删除与访问控制由部署方负责，非本脚本已自动实现的保障。

## 故障与资源

`check` 不可用返回2与真实原因。缺依赖/文件先补资源，不伪造成功。Piper Python调用目前无硬超时隔离；XTTS也无进程沙箱。CLI输入上限10万字符、输出30分钟不是内存恒定保证；音频全量驻留，保守分章节。评测先释放 TTS 再加载 ASR，但 gc 不保证 native allocator 归还内存；资源紧张时用不同进程。资源实测是当前环境样本，不是通用容量承诺。

# 官方日包与下载脚本

本流程吸收 download_install_torch.sh 的解释器选择、CPU torch 获取、日期包解析和下载缓存规则。脚本是可选入口，不随技能复制；用户提供时读取实际版本并传入明确参数，没有脚本时按本文件规则直接获取产物。

## 锁定产物

1. 根据目标解释器确认 Python ABI、系统和架构；NPU 场景显式选择 npu，通常使用匹配的 CPU torch wheel 加 TorchNPU，不因下载主机没有 npu-smi 而落到 GPU/普通 torch 分支。跨机下载按目标 ABI/架构选包，不能用 Windows 下载机的自动检测替代 Linux NPU 目标。
2. 源码最新请求继续走 [源码构建](build.md)。官方日包请求区分“最新已发布包”“指定日期最高构建号”和“固定 URL/构建号”；不把官方包日期推断为源码 SHA。
3. 从用户/项目确认的官方索引列出实际存在的包。脚本使用 PTA_URL_BASE 下的 cache/v<version>/ 与 pta/Daily/v<version>/ 两类目录；后者按 YYYYMMDD.N 排序，先按日期、再按数字 N 降序选择匹配 Python 标签的产物，并做有界可达性检查。不要猜连续构建号；索引分页未读完时说明枚举边界，不能据此断言不存在。
4. 固定 URL 优先，但必须与用户同时给出的日期、系列和 ABI 一致；冲突先解决，不让脚本优先级静默覆盖用户目标。指定日期找不到匹配包时报告可用构建/ABI，不自动跨日期或更换系列。
5. 未指定日期时，原脚本优先 cache 的可变 latest URL，不等价于查 Daily 目录最高日期。需要固定“最新已发布”时先解析到具体构建号/URL；只能得到可变 URL 时记录下载时间与包内真实版本，并保留首次下载文件供续建使用。

torch 与 TorchNPU 分别核对版本和配套。脚本遇到已安装且可导入的 torch 会直接复用，后续还可能按这个版本选 PTA 目录；调用前必须核对请求版本。错误版本应在本轮候选内按计划纠正，不能覆盖正式环境。脚本的 PTA_TORCH_VERSION 不能单独保证已安装包被替换。

## 调用已有脚本

解释器选择优先显式 PYTHON，其次 VIRTUAL_ENV/UV_PROJECT_ENVIRONMENT，最后才是 PATH 中 python3。日构建始终显式传 PYTHON；同时确认其为本轮隔离环境。以下变量由本轮已核实的值填写：

| 输入 | 用法 |
|---|---|
| PYTHON、UV_EXECUTABLE | 目标 venv 解释器与可用 uv 的绝对路径 |
| TORCH_ACCELERATOR_TYPE | NPU 包任务设为 npu，避免 auto 依赖本机设备检测 |
| PTA_WORKDIR | 个人下载目录；不采用脚本内另一用户的默认目录 |
| PTA_TORCH_VERSION、PTA_DAILY_VERSION | 显式匹配的 torch 版本与 Daily 目录版本，并核对已安装 torch |
| PTA_DATE | 显式赋值会选择日期包；只想调整归档目录时不要误设它 |
| PTA_URL | 已解析的固定包 URL；固定它后续建不再解析 latest |
| PTA_DAILY_MAX_PROBES | 原脚本默认最多探测 10 个候选；有界调整并保留失败信息 |
| --download-only / PTA_DOWNLOAD_ONLY=1 | 仅获取 torch wheel 和 PTA 归档，不安装或解包 |
| PTA_DOWNLOAD_DEPS=1 | 额外下载脚本声明的依赖；不代表完整离线依赖集合 |

调用示例使用已解析的固定 URL，避免环境遗留参数改变版本选择；执行前还需检查该脚本的下载证书选项和缓存是否符合下文要求：

```bash
# SCRIPT_PATH 为已审阅脚本，PYTHON/UV 及下列目标值已核实。
PYTHON="${PYTHON}" UV_EXECUTABLE="${UV}" TORCH_ACCELERATOR_TYPE=npu \
PTA_WORKDIR="${DOWNLOAD_ROOT}" PTA_TORCH_VERSION="${TORCH_VERSION}" \
PTA_DAILY_VERSION="${DAILY_VERSION}" PTA_URL="${RESOLVED_PTA_URL}" \
PTA_DOWNLOAD_ONLY=1 PTA_DOWNLOAD_DEPS=0 bash "${SCRIPT_PATH}" --download-only
```

**仅下载的实际边界：** 原脚本的下载功能仍使用 pip download；其版本推断函数也可能导入已经安装的 torch。若要求严格不导入/不执行目标包，使用固定 URL 的直接下载方式，不调用这些探测函数，也不安装缺失的 uv/pip。下载不要求存在 NPU，更不等待空闲卡。

## 缓存、解包与安装

- torch 缓存按版本、Python ABI、架构区分；PTA 日期归档必须加入完整 YYYYMMDD.N 构建号，同名压缩包不能互相替代。可变 latest URL 使用本轮独立目录，不因旧文件非空就认定最新或完整。
- 采用有界网络超时、重试和断点续传。写入 .part 等临时文件，验证归档/ZIP 完整后再认定下载成功；原脚本仅检查 PTA 文件非空的复用规则不足以证明完整，不能直接复用中断下载。保留失败日志，不清理无关缓存。
- 若代理环境的 HEAD 探测超时，可改用有界 Range GET（bytes=0-0）；区分网络故障与确实不存在，不能把一次 HEAD 失败直接当成缺包并切换目标。
- 保持 HTTPS 证书验证；脚本含跳过验证选项时使用已核实的安全下载入口或任务副本修正下载参数。现有官方来源仅支持 HTTP 时如实记录传输方式，有官方摘要时校验，不把 HTTP 下载说成受 TLS 保护。
- 解包到候选暂存目录，核对 wheel 的 Python/平台/架构标签、包名与元数据，只安装请求范围中的匹配 wheel。不要把 tar 内所有同架构 wheel 一股脑装入环境。
- 按 [uv 安装规则](uv.md) 先解决配套依赖，再安装明确 wheel，保留当前有效个人补丁。--no-deps 不代表依赖已满足；下载成功不代表安装或导入通过。
- 原脚本主要处理 torch/TorchNPU，不构建 TA/NPU IR。完整日构建请求继续核对剩余组件和实际编译器入口；仅下载请求交付路径、来源及后续安装方法，到此结束。

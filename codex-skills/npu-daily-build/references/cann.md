# CANN 版本适配与官网获取

通常沿用项目已配套的 CANN。出现缺失组件、接口/符号不兼容或明确的版本问题时，先记录实际加载的 CANN、底层 Bisheng、驱动/固件、芯片与主机架构及首条实质错误，排除旧 set_env.sh、PATH/LD_LIBRARY_PATH 或 CMake 缓存串入另一版本。不能把所有构建失败都归为 CANN 版本问题。

## 选择官网版本

- 从 [CANN 官网下载入口](https://www.hiascend.com/cann/download) 或 [用户提供的筛选入口](https://www.hiascend.com/cann/download?versionId=799&ids=d806%2Ch0501%2Ch0601%2Ch0703&currentTab=1) 开始，结合该版本发布说明、配套关系和安装文档选包。筛选参数不是固定的“最新版本”承诺，每轮读取页面实际版本与文件名。
- 可获取官网最新发布包，也可选择更适配目标 TorchNPU/TA/NPU IR、芯片及现有驱动/固件的早期版本。优先兼容性和问题对应的修复说明，不能只按版本号大小升级；若要求最新，则核实官方当前发布信息并说明与目标栈是否配套。
- 按远程 Linux 的架构、芯片和使用场景选择 Toolkit 及该版本需要的算子/运行库等包；组件命名与拆分随版本变化，以目标版本文档为准，不能混用不同发布批次。Windows 本机只作下载中转，不下载 Windows 包代替远程 Linux 安装包。
- 固定所选完整版本、官方来源页、文件名及可用的摘要/签名信息；不把 Weekly、beta、RC 与正式版视为可互换。不为解决某个版本问题修改 CANN 源码、安装包内容或二进制。

官方操作参考：[CANN 文档入口](https://www.hiascend.com/cann/document)、[软件包准备示例](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/softwareinst/instg/instg_0003.html)、[离线安装与上传示例](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0005.html)。示例属于其标注版本，实际安装参数及配套以本轮选定版本文档为准。

## 远程直下与本机中转

先用官网实际提供的下载地址在 SSH 远程的个人目录获取，采用有界超时/重试与临时文件，保留错误。不要猜测包地址或用反复无限重试等待网络恢复。

远程因网络、代理或登录态无法下载时，可从本机通过官网或已登录的浏览器下载**同一个目标包**，再用 scp/SFTP 上传到 SSH 远程个人暂存目录；无需重新选择版本。使用明确的 SSH 主机与绝对目标路径，上传完成后比较两端文件大小和 SHA256，并按官网提供的方式核验摘要/数字签名。下载到本机或 scp 退出 0 都不能代替远端文件完整性核对。

登录、账号权限或必须由用户完成的协议操作确实阻塞获取时，保留已解析的版本与文件信息并说明阻塞点；不绕过访问限制。只传安装包及必要的公开校验文件，不把浏览器会话、cookie 或临时下载凭据写进共享目录和报告。

## 候选安装与重新核验

环境搭建范围需要安装时，使用该版本官方支持的非 root/自定义安装路径，在个人目录并行保留新旧 CANN；先核对安装器帮助及组件安装顺序。不套用另一版本的 .run 参数，也不更改公共 CANN、系统 latest 链接、驱动、固件或其他任务的环境。

为本轮候选生成独立激活入口，明确引用选定版本 set_env.sh，保留旧入口作为恢复方式；继续遵循 [环境检查](validation.md) 的新进程、nounset 及环境变量隔离规则。重新核对实际 CANN/Bisheng、Python 包导入、TA 选择的编译器路径及版本，避免 PATH 看似已切换而实际仍用旧包内编译器。

CANN/Bisheng 改变后重新评估 CMake 缓存与 [共享包](shared-packages.md) 的配套；同 Git SHA 不自动代表旧 CANN 构建的 wheel 可复用。为不兼容项使用新的候选构建目录，仅续建受影响组件，保留旧环境与产物。完成环境检查不代表算子、精度或性能通过，不为 CANN 获取流程额外启动设备测试。

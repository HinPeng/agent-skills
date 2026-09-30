# 配套、构建与候选安装

## 确定输入与复用边界

新轮次记录各组件的来源、ref/精确 SHA（包则记录真实构建标识）、Python ABI、目标架构、CANN 与有效构建选项。日志日期和 wheel 日期不自动等于源码日期或 SHA。已有官方二进制来源不完整时说明，不伪造对应提交。

固定输入后、启动任一组件编译前，按 [共享包复用](shared-packages.md) 查找 /data/share 下同提交且配套一致的产物；命中则直接进入候选安装与检查，只构建缺失组件。CANN 配套有问题时先按 [官网获取与适配](cann.md) 确定目标版本，再按新配套重查缓存；不为沿用旧包忽略 CANN 差异。

官方源码入口：

- TorchNPU：[Ascend/pytorch](https://gitcode.com/Ascend/pytorch)；它与上游 PyTorch 不同，按该提交构建说明选择匹配的 torch wheel。
- Triton-Ascend：[triton-lang/triton-ascend](https://github.com/triton-lang/triton-ascend)；从目标分支的依赖和构建说明选择社区 Triton 版本。GitHub 与 GitCode 镜像的 ref 不视为天然同步。
- NPU IR：[Ascend/AscendNPU-IR](https://gitcode.com/Ascend/AscendNPU-IR)；LLVM 等子模块使用父提交记录的 gitlink。
- TA nightly：[官方镜像目录](https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly/triton-ascend/)；使用时核实可达性和实际 wheel 内容，不假设普通 nightly 包已含完整 IR。

先核实已存在构建/锁的所属与活动状态，避免同时改同一轮次；不要盲删锁文件。新组合使用隔离的源码 checkout、构建目录和候选 venv；同一轮中断续建复用原目录与有效产物，不重新 clone 或清理构建树。保留状态、日志、原 wheel 及当前入口快照，检查进程和产物真实状态，不能仅相信日志最后的 PASS 或状态文件的阶段名。

来源、ABI、CANN、LLVM、构建配置匹配时可复用已验证产物；切换其中任一项时重新评估缓存。哈希只用于有价值的二进制同源/完整性核验，普通日志与源码检查无需例行生成全目录哈希清单。

## TorchNPU 与 TA

按 [uv 管理](uv.md) 在候选环境或专用构建 venv 安装构建依赖，显式指定解释器和 wheel 文件路径。检查目标提交的构建入口、版本覆盖选项和子模块，不把旧提交的命令原样套到新版。源码最新与官方最新日包是不同输入；包获取按 [官方包参考](packages.md) 处理。

若目标版本使用 ci/build.sh，核对其 Python/torch 系列选择参数及依赖解析结果，不能默认沿用历史参数。例如请求 2.13 时，必须确认依赖解析没有升级到其他系列。检查基础 PyTorch wheel ABI 与 TorchNPU 扩展编译 ABI，不能靠改 wheel 文件名解决不匹配。TA 按目标提交实际入口编包；旧 setup_ascend.py 在部分版本已改为根目录 setup.py，入口不存在时先核对源码，仅修正本轮编排，不修改外部产品源码。

TorchNPU、TA 和 NPU IR 共用 [并发与续建策略](concurrency.md)。逐阶段确认实际 -j，不能只调高顶层变量而留下 runner 内的 min(16, jobs) 或子构建硬编码上限。

下载或 Git 获取遇到连接问题时使用有界超时；允许改用可信镜像或本机获取再传输，但保持所选版本和来源可追溯。实际权限或来源缺失无法解决时保留已有工作，不猜下载地址。

## NPU IR 构建与打包

加载目标 CANN 和构建 venv，使用个人持久化 TMPDIR。按所选源码的脚本/帮助确认 CMake、Ninja、Clang、LLD、构建选项及模板库目标。按并发参考统一各层 -j 与环境变量，单独限制 LLVM 链接并发。

同时核对 CMake 缓存中的 BISHENG_COMPILER_PATH、BISHENG_COMPILER_EXECUTABLE 实际属于选定 CANN。只改 PATH 无法纠正旧缓存。复用缓存时避免仅变安装前缀触发无关的 LLVM 全量重编；向新目录收集安装产物，不覆盖旧安装。

失败处理保留第一条实质错误。只有确实出现 HACCBaseDialect.h.inc 等生成头缺失、且该版本存在相应构建目标时，才对实际 build_hivmc 目录执行 mlir-headers 后续建。不要将这一历史恢复步骤作为每次必跑项，也不要用源码修改掩盖编译器缺陷。

需要自合包时：

- 从同一次配套构建收集 bishengir-compile、bishengir-opt、所需 HIVMC 工具及完整 .bc 模板库；按当前消费方接口判断是否需要 hivmc-a5/hivmc 兼容入口。
- 按实际安装清单定位产物；部分版本将 HIVMC 放到 build/install/bin/hivmc，不再沿用旧 build_hivmc 布局。收集路径失效先修正本轮打包入口，复用已成功产物，不因旧路径不存在重编全部组件。
- 独立 IR wheel 和 TA wheel 使用同源 payload。TA 内常见位置为 triton/backends/ascend/bishengir/bin 与 lib，先确认目标版本实际布局。
- 仅在独立解包目录替换 payload，保留原 TA Python 实现、原 wheel 和必要执行权限；标准 wheel 重打包更新 RECORD，不能只改压缩包文件名或版本号。
- 核对实际编译器版本、配套依赖和原 TA 非 payload 文件保持一致。用构建标识与简短来源记录说明这是本地合包，不称官方发行包。

不需要更换 IR 的任务可复用兼容的现有组件；不因技能覆盖三种组件就每次全量重建。

## 编包后共享归档

完成 wheel 打包和文件完整性检查后，按 [共享归档](shared-packages.md) 将本轮产物复制到 /data/share/<工号>/ 的独立构建目录，附精确源码/配套信息及当前验证状态。候选安装检查结束后更新状态；编包成功不等于环境或设备验收通过。归档失败保留原产物并报告，不因此重编。

## 安装与个人补丁

依次安装匹配的 torch/TorchNPU、社区 Triton、对应 TA 及请求的 IR 产物。社区 Triton 与 TA 可能写入同一 triton 命名空间，重装社区 Triton 后需按配套流程重装 TA 并验证实际导入。wheel 安装后的路径和依赖才是证据，不能仅凭安装退出 0 宣布完成。

从当前个人环境确认仍有效的 TorchNPU 补丁，比较新源码是否已等价包含，避免重复应用或复活撤回方案。内容已由其他任务修改或补丁冲突时保留差异；解决授权范围内的迁移问题后继续，不能静默丢弃用户代码。接下来按 [验收与启用](validation.md) 验证候选。

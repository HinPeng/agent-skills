# TorchNPU 架构分析技能使用指南

在 Codex 中加载此技能后，把 `$torch-npu-architecture` 放在提问开头。下面的示例可以直接复制。

| 功能 | 提问示例 |
| --- | --- |
| 画软件分层图，解释初始化顺序 | `$torch-npu-architecture` 画出从 `import torch_npu` 到昇腾设备的软件分层，标出 Dispatcher、op-plugin、CANN 和首次设备初始化。 |
| 追踪算子调用链 | `$torch-npu-architecture` 对比 `aten::copy_` 在 CPU→NPU、NPU→CPU 和同设备 NPU→NPU 的调用路径，标出分支条件和源码入口。 |
| 分析异步执行和内存生命周期 | `$torch-npu-architecture` 一个张量在 Stream A 创建、在 Stream B 使用。画出 `wait_event`、`record_stream` 和内存安全复用的时序。 |
| 分析 ACL Graph 录制与回放 | `$torch-npu-architecture` 画出 `NPUGraph` 从 `capture_begin` 到 `replay` 的时序，标出图内存池的使用和释放。 |
| 排查 HCCL 集合通信 | `$torch-npu-architecture` 画出一次 `dist.all_reduce` 从计算流到 HCCL 流的时序，列出通信卡住时应按 rank 核对的信息。 |
| 解释编译链路和定位性能问题 | `$torch-npu-architecture` 同一个模型用 Inductor 比 eager 慢。画出从 FX 到 kernel launch 的路径，指出应查看的日志、生成代码和 profiler 数据。 |
| 解释 AI Core 数据通路和 kernel 硬件映射 | `$torch-npu-architecture` 画出 351x 的矩阵与向量数据通路，说明如何确认某个 kernel 用到了哪些计算单元。 |
| 评估代码改动并设计验证 | `$torch-npu-architecture` 我要优化 `copy_(non_blocking=True)` 的 dtype 转换。定位修改入口，说明需要保持的语义，并列出最小验证用例。 |

解释基线架构时，可直接使用技能附带的资料。分析具体版本或运行问题时，可在提问里附上源码分支或提交、芯片型号和 CANN 版本，以及已有的日志、生成代码或 profiler 记录。

更多细节见 [技能说明](SKILL.md) 以及 [软件架构](references/software.md)、[运行时](references/runtime.md)、[编译架构](references/compiler.md) 和 [硬件架构](references/hardware.md) 参考页。

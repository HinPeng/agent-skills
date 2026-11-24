---
name: torch-npu-architecture
description: 使用已验证的 TorchNPU 架构知识库解释或绘制软件分层、算子调用、设备运行时、torch.compile、HCCL 与昇腾 AI Core，并为相关开发和问题定位提供入口。适用于 TorchNPU 架构设计、调用链、硬件映射、性能归因和改动影响分析。
---

# TorchNPU 架构分析

以本技能的参考资料直接回答架构问题。基线为 TorchNPU 提交 `8c651a8418042502cd007bda76bfddd796065d19`，分析自 `D:/codes/pytorch`。基线可离线使用。解释机制、画图、做设计讨论时优先复用已有模型和时序。涉及新版本差异、具体算子分支、实际修改或部署行为时，按问题核验对应锚点或运行证据。

## 按问题加载

- 软件总览、初始化、Dispatcher、算子接入、构建及 `copy_` 示例 → [software.md](references/software.md)
- device/stream/event、队列、内存生命周期、ACL Graph、HCCL 和 profiler → [runtime.md](references/runtime.md)
- Dynamo、TorchAir、Inductor、Triton、CATLASS、AOTI 和性能诊断 → [compiler.md](references/compiler.md)
- Ascend AI Core、AIC/AIV、片上存储、搬运及型号证据 → [hardware.md](references/hardware.md)

## 交付方式

1. 识别用户问的是组件边界、执行时序、内存生命周期、编译选择还是物理数据流。只加载相关参考页。涉及软件到硬件时，同时加载硬件页和实际软件入口所在页。
2. 复用参考图。组件依赖和数据流用 Mermaid flowchart，调用、跨 stream 或 capture/replay 用 sequenceDiagram。画综合架构时从用户 API 到设备侧保持层次，并标注条件路径、主机与设备边界、同步及所有权。物理图单独呈现产品或架构版本。
3. 在图下给出关键机制、变更入口与影响，引用参考页中的仓库相对路径及符号。标注属于提交源码、官方硬件资料、运行观测还是推断。未初始化子模块可确认 gitlink、配置、主仓接口，内部路径需由匹配提交验证。
4. 变更设计先说明需要保持的语义，再定位修改入口和最小验证矩阵。以运行材料判定实际生效的 branch/kernel/stream。输入缺少芯片型号、runtime 版本或已生成 kernel 时可先给分支图，并写明哪条边仍待核实。

## 使用示例速查

在提问前加上 `$torch-npu-architecture`。各参考页的「使用示例」提供可直接复制的提问和预期交付。

| 要做的事 | 示例所在页 |
|---|---|
| 画软件分层与初始化、追踪单个算子、评估改动与设计验证 | [软件架构](references/software.md) |
| 分析异步执行与内存、画 ACL Graph 时序、分析 HCCL 通信 | [运行时](references/runtime.md) |
| 画编译链路并定位性能问题 | [编译架构](references/compiler.md) |
| 画 AI Core 数据通路并核验 kernel 硬件映射 | [硬件架构](references/hardware.md) |

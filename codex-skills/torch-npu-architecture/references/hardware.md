# 昇腾硬件视图与软件映射

AI Core 编程模型和相应产品规格来自昇腾官方资料。TorchNPU 基线源码提供软件边界、能力查询和分支条件。具体算子的执行单元和实际瓶颈需要结合编译产物及 profiling 观察。

## 分离模式概念图

~~~mermaid
flowchart LR
  subgraph Host["主机"]
    P["PyTorch / TorchNPU"] --> A["CANN runtime / driver"]
  end
  subgraph Chip["Ascend 设备 · AI Core 分离模式示意"]
    G["Global Memory"] <--> L["L2 Cache"]
    L <-->|"搬运和写回"| B["AIC · 本地缓存/矩阵操作数"]
    B --> C["Cube 矩阵计算"]
    C --> S["矩阵结果缓冲"]
    S -->|"写回通路"| L
    L <-->|"搬运和写回"| U["AIV · Unified Buffer / 本地存储"]
    U <--> V["Vector 向量计算"]
    I["各核 Scalar 控制"]
    I -.-> C
    I -.-> V
  end
  A -->|"stream 上的任务提交"| Chip
  A <-->|"显式 host/device 拷贝"| G
~~~

- 在分离模式，AIC 负责矩阵计算，AIV 负责向量计算，两者有各自的 Scalar 控制。MTE/DMA 和写回单元连接存储层，具体通路依架构而变。GM 代表设备全局地址空间层。HBM 容量、带宽和物理封装使用型号规格。参见 [昇腾官方 AI Core 架构](https://www.hiascend.com/doc_center/source/zh/canncommercial/80RC2/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html)。
- 为 `351x` 作图时可展开官方图中的 GM/L2、AIC 的 L1/操作数与输出缓冲、AIV 的 UB/向量寄存器以及 SSBuffer 核间通信。在图标题中写明 `351x`。用已确认的产品与架构对照关联指定型号。参见 [CANN 9.0.0-beta2 Ascend C 指南 `2.6.2.4](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/CANN%E7%A4%BE%E5%8C%BA%E7%89%88%209.0.0-beta2%20Ascend%20C%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E6%8C%87%E5%8D%97%2001.pdf)。
- 推导某款 950、A2 或 A3 的物理结构时记录产品全名、SKU、对应 `__NPU_ARCH__` 和资料版本。若只知道产品族，给出概念图并标明未确认信息。

## 如何把软件行为映射到硬件

| 软件证据 | 可以确认或进一步核验的结论 |
|---|---|
| `torch_npu/npu/__init__.py::get_device_properties`、`torch_npu/csrc/npu/Module.cpp` | 运行时暴露名称/内存/核数等。`aclGetDeviceCapability` 分别查询 Cube、Vector 核数。兼容用 `multi_processor_count` 优先采用 Vector 核数，不能当作 Cube 核数 |
| `torch_npu/_inductor/config.py::_obtain_and_limit_cube_vector_core_num`、wrapper launch 参数 | 编译器的核数预算与计划网格。实际核参与情况结合 kernel 类型与运行记录 |
| `aten::matmul`、`aclnn*`、`CATLASS` 或 Triton 名称 | 只能定位语义或执行候选。确认 Cube/Vector、融合与搬运需匹配实现、生成 IR 或 profiler 数据 |
| `aclrtMemcpy*`、`copy_` 的分支、内部格式转换 | 区分 host/device、D2D 和计算核内搬运。某些 copy 包含 cast 或布局变换，检查实际任务 |
| `ProcessGroupHCCL` 与 communication stream | 确认提交、依赖与 completion。设备互联拓扑与带宽由实际机器和网络配置决定 |

性能假设按形状和数据流提出。例如，逐元素融合可能减少中间 tensor 的全局写回，矩阵分块可能改变局部缓存复用和搬运开销。用 kernel 计时、可用的 Cube/Vector 指标、访存或吞吐指标对照，运行数据不足时保留为假设。

## 使用示例

### 昇腾硬件与数据通路

> $torch-npu-architecture 画出 351x 的矩阵和向量数据路径，标出 GM、L2、Cube、Vector 和 UB，再说明怎么验证某个 kernel 用了哪些计算单元。

交付一张注明 `351x` 的硬件图及官方资料出处。列出生成 IR、launch 参数、设备属性和 profiler 指标分别能验证的结论。指定某款 Ascend 950 时，补充确认完整产品型号与架构版本。

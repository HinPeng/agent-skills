# 运行时、内存、图与通信

基线 `8c651a8418042502cd007bda76bfddd796065d19`。下文简写路径 `core/npu/`、`distributed/` 和 `framework/` 相对于 `torch_npu/csrc/`。

## 提交和同步

~~~mermaid
sequenceDiagram
  participant P as Python/ATen 调用线程
  participant O as TorchNPU 执行包装
  participant Q as 可选 PTA host task queue
  participant S as CANN runtime stream
  participant D as Ascend 设备
  P->>O: dispatch / prepare / 选择 device 和 stream
  alt 启用并满足队列路径
    O->>Q: enCurrentNPUStream
    Q->>S: worker 提交 ACL 或算子任务
  else 直接提交
    O->>S: 直接调用 runtime/算子接口
  end
  S->>D: 按 stream 顺序执行 device tasks
  Note over P,D: host 返回、host 队列清空、设备执行完毕是不同的时刻
~~~

- `core/npu/NPUFunctions.cpp` 与 `core/npu/impl/NPUGuardImpl.cpp` 处理 device/guard。`core/npu/NPUStream.cpp` 保存线程局部的 per-device current stream，管理默认流、池及 `enCurrentNPUStream`。流 API 的 Python 入口见 `torch_npu/npu/streams.py`。
- `framework/OpCommand.cpp::RunOpApiV2`、`RunOpApiV3` 展示可选队列与直接调用。`core/npu/NPUQueue.cpp` 承接 host 任务及异步执行错误。区别 host 队列与 ACL stream，可解释错误延迟暴露、profile 中提交与执行错位。
- `core/npu/NPUEvent.cpp` 实现 record/wait/query，对应 `AclrtRecordEventWithFlag` 和 `AclrtStreamWaitEventWithFlag` 等。生产 stream record、消费 stream wait 建立执行依赖。allocator `recordStream` 延长存储对消费流的可用期。跨流传递同时检查这两项。
- `core/npu/interface/AclInterface.cpp` 与 `framework/interface/AclInterface.cpp` 承接 ACL 封装，`core/npu/register/FunctionLoader.cpp` 支撑动态符号解析。排查 API 存在性和行为时确认已加载 CANN 库版本、符号及生效开关。
- `TASK_QUEUE_ENABLE` 与 `ASCEND_LAUNCH_BLOCKING` 改变观察条件。复现报告应记录它们，性能比较使用目标配置。

## 内存、异步生命周期与 capture

| 关注点 | 机制与锚点 |
|---|---|
| device 内存 | `core/npu/NPUCachingAllocator.cpp::DeviceCachingAllocator` 管理每设备 block、缓存池和事件。常规路径可复用缓存块，需向 runtime 申请时可见 `AclrtMallocAlign32`，另有配置相关分支 |
| 跨流回收 | allocator 的 `recordStream` 记录额外使用流，释放时以 `insert_events` / `process_events` 跟踪可安全复用的时机。对比 Python 引用结束、allocated、reserved 与设备工作完成 |
| host 与 workspace | `core/npu/CachingHostAllocator.cpp` 对应 host pinned 缓冲，`core/npu/NPUWorkspaceAllocator.cpp` 对应 workspace。调试异步 copy 需追踪 host 缓冲及 device 临时张量，含异步 cast 结果 |
| 图 capture | `core/npu/NPUGraph.cpp::capture_begin` 先建立 allocator 池路由，并处理主机队列，再调用 `AclmdlRICaptureBegin`。`capture_end` 完成 runtime capture、撤销池路由并维护图的资源 |
| 图 replay | `NPUGraph::replay` 更新登记的 RNG 状态并调用 `AclmdlRIExecuteAsync`。图跨次执行依赖 capture 的地址/资源约束。`reset`/析构负责释放池及模型资源 |
| capture 与 event | allocator 在 capture 期间避免不适当的事件查询，使用延期处理路径。修改需要覆盖成功、失败清理、重放、共享池和跨流使用 |

验证入口 `test/test_npu.py`、`test/test_npu_allocator.py`、`test/test_copy_.py`、`test/dynamo/test_npugraphs.py`。ACL Graph 录制和回放 runtime 任务，Dynamo 等上层编译处理 FX 图，两种机制可组合。

## HCCL 集合通信

~~~mermaid
sequenceDiagram
  participant C as 计算 stream
  participant P as ProcessGroupHCCL
  participant H as HCCL stream
  participant W as WorkHCCL / 消费者
  P->>P: getHCCLComm 获取或建立 communicator
  C->>H: syncStreams 通过 event 建立依赖
  P->>H: collective 提交 hcclAllReduce
  P->>P: 按配置维持 tensor storage 生命周期
  H->>W: 完成事件及 Work 的同步/完成语义
~~~

- `torch_npu/_init/registry/distributed.py` 在 PyTorch 注册 `hccl`、`lccl`，相应 C++ 实现各自位于 `distributed/ProcessGroupHCCL.cpp`、`distributed/ProcessGroupLCCL.cpp`。插件名、communicator 和物理互联是不同层。
- `ProcessGroupHCCL::allreduce` → `collective` → `getHCCLComm`、`syncStreams`、HCCL stream、`hcclAllReduce`，随后由 `WorkHCCL` 管理完成及异常。生命周期策略包括 `NPUCachingAllocator::recordStream` 等配置相关路径。
- 分析卡死收集各 rank 的 collective 顺序、group、shape/dtype、设备映射、等待位置、超时和首个报错。性能分析用时间线核对计算流是否与通信流重叠。链路类型或拓扑从目标部署资料、HCCL 日志和设备工具获取。

## Profiler 与归因

`torch_npu/profiler/profiler.py::profile` / `analyse` → `torch_npu/profiler/profiler_interface.py::_ProfInterface` → `torch_npu._C._profiler`、`torch_npu/csrc/profiler/` → Python `torch_npu/profiler/analysis/` 处理输出。用 host op、runtime 和 device kernel 事件关联，检查初始化、host dispatch/等待、编译、device 执行和通信分别占用的时间。比较 eager 与编译结果时，统一设备、输入、同步策略与采集窗口。

## 使用示例

### 异步执行和内存生命周期

> $torch-npu-architecture 一个 NPU tensor 在 Stream A 上创建、在 Stream B 上消费。画出 wait_event 和 record_stream 各自的作用，以及 Python 引用释放后内存何时能复用。

交付一张跨流时序图，分别标出生产完成依赖、消费开始与 allocator 安全复用时机。

### ACL Graph 录制与回放

> $torch-npu-architecture 画出 NPUGraph 从 capture_begin 到 replay 的时序，标出 Host Task Queue、图内存池以及成功与失败时的清理。

交付一张 capture/replay 时序图，标出池路由注册与撤销、runtime capture、重放和最终资源释放。

### HCCL 集合通信

> $torch-npu-architecture 画出一次 dist.all_reduce 从计算流到 HCCL 流的时序，并标出遇到 collective 卡住时该从哪里看。

交付一张包含 communicator、计算到通信的 event 依赖及 `WorkHCCL` 完成语义的图。按 rank 核对 collective 顺序、group、shape/dtype、设备映射和首个超时或异常。

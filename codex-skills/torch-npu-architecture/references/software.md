# TorchNPU 软件结构与算子接入

本页以提交 `8c651a8418042502cd007bda76bfddd796065d19` 为基线。锚点是该提交的仓库相对路径及符号。上游 PyTorch、CANN 安装与各 gitlink 是独立的版本变量。

~~~mermaid
flowchart TB
  U["用户 API · torch / torch.npu / torch_npu"] --> AT["ATen schema 与 Dispatcher"]
  B["torch_npu 初始化与注册"] -.-> AT
  AT -->|"PrivateUse1 / AutogradPrivateUse1 等"| REG["生成注册与手写绑定"]
  REG --> N["主仓 ATen 核心实现"]
  REG --> OP["op-plugin 算子适配"]
  N --> RT["分配器 · Stream / Event · ACL Runtime 封装"]
  OP --> CMD["aclnn / OpCommand 等执行入口"]
  N --> CMD
  CMD --> RT
  RT --> C["CANN Runtime / 驱动边界"]
  C --> H["Ascend 设备"]
~~~

## 导入与真正的设备初始化

- 打包入口点 `torch.backends: torch_npu = torch_npu:_autoload` 可由 PyTorch 导入。`torch_npu/__init__.py::_initialize` 顺序为加载核心模块、注册组件、打补丁、完成扩展生命周期、开启可选功能。导入中暂时关闭 autoload 避免循环，`_autoload` 恢复原设置。
- `torch_npu/_init/registry/backend.py::register_privateuse1_backend` 将通用后端槽改名为 `npu`，注册 `torch_npu.npu` 为设备模块并生成 tensor/module/storage 方法。改名影响 Python 设备名，C++ dispatch key 仍使用 `PrivateUse1`。
- `torch_npu/_init/core/runtime_lifecycle.py::_initialize_runtime_lifecycle` 调用 `_C._initExtension` 并注册退出清理。首次需要设备的路径由 `torch_npu/npu/__init__.py::_lazy_init` 调用 `_C._npu_init`，对应 `torch_npu/csrc/npu/Module.cpp::THNPModule_initExtension` → `torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp::NpuSysCtrl::Initialize`，包含系统运行时初始化和 NPU 分配器初始化。导入失败和第一次使用设备报错对应不同诊断入口。
- 两个易混淆的注册表是 ATen Dispatcher 的 `PrivateUse1` 以及 Dynamo 的 `npu` 编译后端。后者见 [compiler.md](compiler.md)。

## 生成链及职责边界

| 入口 | 职责 |
|---|---|
| `pyproject.toml`、`setup.py`、`CMakeLists.txt` | wheel 入口点、编译/打包、C++ 模块与可选依赖接入 |
| `generate_code.sh` | 按 PyTorch 主次版本计算 `vNrM`，调用 op-plugin 生成器和本仓生成器。脚本含对 `third_party/acl_src` 的删除步骤，阅读架构不需要执行它 |
| `torch_npu/csrc/aten/npu_native_functions.yaml` | NPU 适配列表、支持和自定义项以及部分路径控制，如 `op_api` 和 `internal_format_opapi` |
| `torchnpugen/gen_backend_stubs.py` | 合并版本相关 op-plugin 配置，构建 backend index，生成 `TORCH_LIBRARY_IMPL(aten, …)`、op-plugin wrapper 和自定义注册。`NPU` YAML 名映射到 `PrivateUse1`，autograd 可生成 `AutogradPrivateUse1` |
| `torchnpugen/autograd/gen_autograd.py`、`torchnpugen/templates/` | autograd 及 schema、redispatch、functionalization 等生成代码的维护入口 |
| `torch_npu/csrc/aten/common/`、`torch_npu/csrc/aten/ops/` | tensor factory、copy、view/resize、format 等核心操作及部分本地包装 |
| `torch_npu/csrc/framework/OpCommand.cpp`、`OpCmdHelper.cpp`（同目录） | 算子参数、命令提交。`RunOpApi*` 可按条件进入 host 队列，或直接运行传入 callable；旧命令路径可走 ACL 算子编译/执行接口 |
| `torch_npu/csrc/core/NPUTensorImpl.cpp`、`torch_npu/csrc/core/NPUStorageImpl.cpp`、`torch_npu/csrc/framework/StorageDescHelper.cpp` | Tensor/Storage 元数据与内部格式。逻辑 shape/stride/offset 与物理存储描述都影响实际拷贝和格式转换 |

排查算子按 schema 和 overload 查起。先检查 dtype/device 及 dispatch key、是否由上游复合实现分解，再检查 YAML、生成 wrapper、主仓 C++ 实现或 op-plugin 绑定。兼容实现切换、复合分解、显式 CPU fallback 和 Inductor extern fallback 各有独立触发条件。源码锚点 `torch_npu/csrc/framework/utils/CpuFallbackUtils.h`、`torch_npu/csrc/aten/ops/SparseCsrReductionFallbackKernelNpu.cpp` 可用于解释该提交的显式 CPU fallback 控制及具体示例。

## 已追踪样例 · aten::copy_

- `npu_native_functions.yaml` 将 `copy_` 同时标为 `op_api` 和 `internal_format_opapi`。实现入口为 `torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp::NPUNativeOpApiFunctions::copy_` 及 `torch_npu/csrc/aten/common/CopyKernel.cpp::NPUNativeFunctions::copy_`。实际 dispatcher wrapper 的选择条件仍随版本和开关变化。
- op-api 实现入场先处理部分特殊元数据及 `DO_COMPATIBILITY(aclnnInplaceCopy, ...)`。同设备 D2D 可落到 `copy_d2d_by_memcpy` 或 `EXEC_NPU_CMD(aclnnInplaceCopy, ...)`，也可进入 native 路径。内部格式、SoC 与 dtype 会影响可选路径。
- H2D/D2H 先处理布局与 dtype，满足条件时以临时张量过渡，`copy_between_host_and_device_opapi` 再选择异步任务或带同步的拷贝。异步路径还要追踪 pinned host 缓冲及中间结果的生命周期。
- 修改时保留 in-place 返回、广播、目标 stride、格式、空 tensor 和同步语义。验证入口 `test/test_copy_.py`，重点看 mixed dtype、pinned memory、strided layout、临时量生命周期和 blocking 对照。跨设备需额外覆盖 P2P 可用性及两侧 stream 依赖。

## 基线提交的外部代码边界

| gitlink | 固定提交 | 用途 |
|---|---|---|
| `third_party/op-plugin` | `c07087dab2e0dfec3e91af5b624945e29e29307c` | 批量算子实现、版本配置、schema/绑定生成输入 |
| `third_party/torchair/torchair` | `e26223489c337050bcd4874cf4c3d06a03cf573d` | 图编译和对应的可选 Python 后端包 |
| `third_party/acl_src/runtime` | `c71314c12cefba3011823bd1786012858f310c1d` | Runtime 源码/构建所需 ACL 头来源 |
| `third_party/acl_src/ge` | `a74342b574f03ffe45e059c1a0fa74e50ccf5733` | GE 源码/构建所需 ACL 头来源 |
| `third_party/dvm/dvm` | `94174b797cbe0e4068c8eb60db2e3bf04db6fafb` | 可选 DVM 依赖 |

以上 gitlink 在基线分析时均未展开。运行实际加载的 CANN 库由安装环境及适配层符号解析决定，参见 [runtime.md](runtime.md)。

## 使用示例

### 软件分层与初始化

> $torch-npu-architecture 画一张从 import torch_npu 到 Ascend 设备的软件架构图。标出 PyTorch Dispatcher、PrivateUse1、op-plugin、CANN，以及导入注册和首次设备初始化分别发生在哪里。

交付一张 Mermaid 分层图，说明组件职责和运行时边界。用 `_initialize`、`register_privateuse1_backend` 与 `_lazy_init` 锚定初始化顺序。

### 单个算子的执行路径

> $torch-npu-architecture 画出 aten::copy_ 在 CPU→NPU、NPU→CPU 和同设备 NPU→NPU 时的执行路径。标出 memcpy、aclnnInplaceCopy、dtype 转换和内部格式分支。

交付一张按方向展开的流程图，附分支条件和源码入口。涉及异步提交、临时量和跨流生命周期时，结合 [运行时参考](runtime.md)。

### 改动影响和验证矩阵

> $torch-npu-architecture 我要优化 copy_(non_blocking=True) 的 dtype 转换。帮我找修改入口，说明需要保持的语义，给出最小验证矩阵。

定位 OpAPI 和 native 中相关入口，说明对 inplace、目标 stride 和异步生命周期的影响。用 `test/test_copy_.py` 组织验证，覆盖 pinned memory、非连续布局、临时张量生命周期及 blocking 对照。

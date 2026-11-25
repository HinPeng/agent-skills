# TorchNPU 编译路径与性能定位

基线 `8c651a8418042502cd007bda76bfddd796065d19`。编译后端、生成的 kernel 和 ACL Graph 的 replay 是不同层次的概念。

~~~mermaid
flowchart TD
  P["torch.compile"] --> D["Dynamo · FX 子图与 guards"]
  D -->|"backend=npu"| T["TorchAir 可用时走图后端"]
  D -->|"backend=npu 且缺少构建产物"| E["eager FX 路径"]
  D -->|"backend=inductor"| A["AOTAutograd 等编译阶段"]
  D -->|"backend=npugraph_ex"| X["独立注册的图后端"]
  A --> F["NPU lowering / FX passes"]
  F --> S["scheduling / wrapper / 选定 codegen"]
  F -->|"extern fallback"| L["ATen 或外部算子调用"]
  S --> K["编译产物加载与 kernel launch"]
  K --> R["CANN runtime stream"]
  L --> R
~~~

## 两层选择

- `pyproject.toml`、`torch_npu/_init/registry/dynamo.py`、`torch_npu/dynamo/__init__.py` 决定 Dynamo 入口。`_get_default_backend` 在 TorchAir 包可用时安装 lazy loader，`_get_global_npu_backend` 用 `torchair.get_npu_backend`。缺少该包时存在 eager FX 路径。TorchAir 内部 lowering 与 GE 转换细节需要匹配的子模块提交。
- `torch_npu/dynamo/__init__.py` 的 Inductor wrapper 与 `torch_npu/utils/_dynamo.py` 负责 NPU device interface/注册和 lazy setup。`_resolve_npu_backend` 按显式选择、全局 compile 选项、环境变量解析设置。定位同进程切换、配置优先级和缓存污染时，从这里进入。
- `torch_npu/_inductor/__init__.py::_load_backend` 在选择 loader 前恢复 Inductor baseline，加载后清除相关 decomposition 缓存。查看 `_BACKEND_LOADERS` 确认 default、Triton、Triton experimental、MLIR、DVM、Ascend C 的别名和依赖。
- Triton 路径在 `_load_triton_backend` 注册 `NPUCombinedScheduling`、`NPUPythonWrapperCodeGen`、`CppWrapperNpu` 并加载 lowering/外部库选择。CATLASS 可参与 GEMM 模板与 epilogue 调度，具体选择取决于 operator、配置和候选。`torch_npu/_inductor/codegen/catlass/` 是源码入口。

## 改动入口

| 场景 | 优先入口与检查 |
|---|---|
| graph break 或反复编译 | FX 子图、guards、动态 shape、graph breaks、compile 次数。先查 Dynamo trace rules、`torch_npu/utils/_dynamo.py` |
| decomposition、融合、fallback | `torch_npu/_inductor/lowering.py`、`lowering_fallback_list.py`、`lowering_override_list.py`、`fx_passes/`。串起原始 FX、变换后 FX、lowering 及生成代码。extern fallback 可调用 NPU 算子 |
| schedule、layout、tile | `torch_npu/_inductor/codegen/npu_combined_scheduling.py`、`codegen/scheduling.py`、`codegen/catlass/`。核对 shape/stride、归约轴、数据类型和候选实现 |
| 编译与 launch | `torch_npu/_inductor/async_compile.py`、`codegen/wrapper.py`、`codegen/cpp_wrapper_npu.py`。实验 Triton 有 `triton_experimental/static_launcher/` 及 `torch_npu/csrc/inductor/static_launcher/` |
| C++/AOTI | `torch_npu/csrc/inductor/aoti_torch/`（shim）、`aoti_runtime/`、`aoti_runner/`（C++ 容器与运行），以及 `torchnpugen/gen_npu_c_shim.py`。核对 PyTorch 版本、符号、wrapper 与运行安装包 |
| 设备参数 | `torch_npu/_inductor/config.py::_obtain_and_limit_cube_vector_core_num` 读取并对编译可用核数施加限制。将编译预算、硬件属性和实际 launch 网格分别记录 |

## 从现象走到代码

1. 记录 Torch/PyTorch、TorchNPU、CANN、Triton 或外部后端版本，SoC、输入、dtype/stride、编译后端及选项。分开首次编译和稳态运行。
2. 用 FX、guards 和生成 wrapper 确定实际分支。定位 eager、extern、融合 Triton、CATLASS 或其他候选，记录源 FX 节点及 kernel 名。相同 Python API 可能对应零次、一次或多次设备 launch。
3. 从 wrapper 拿到 launch 网格、stream、常量和中间缓冲区，再用 profiler 验证设备时长、kernel 数、空隙和同步。对动态大小补充边界 shape 与重复运行，对精度补充 eager/参考实现对比。硬件归因见 [hardware.md](hardware.md)。
4. 针对变更选择 `test/_inductor/test_inductor_dynamic_shapes.py`、`test/_inductor/test_inductor_fallback_list_triton.py`、`test/_inductor/test_npu_stream_fusion_regression.py` 或 `test/dynamo/test_npu_backend.py` 对应的行为矩阵。

## 使用示例

### 编译链路与性能定位

> $torch-npu-architecture 同一个模型用 torch.compile(backend='inductor') 比 eager 慢。帮我画出从 FX 到 kernel 的路径，并指出需要哪些日志或 profiling 数据来定位。

交付一张标出 lowering、extern fallback、生成 kernel 和 launch 的分支图。列出 FX、guards、生成 wrapper 和 profiler 的观察点，区分首次编译、主机提交和稳态设备执行。提交与设备同步的解释见 [运行时参考](runtime.md)。

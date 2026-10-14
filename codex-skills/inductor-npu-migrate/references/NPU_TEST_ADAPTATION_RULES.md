# PyTorch Inductor 测试用例 NPU 适配规则报告

> 用途：供开发者或代码模型阅读后，对 PyTorch Inductor 测试进行第一轮、可审查的 NPU 适配。
>
> 目标任务：以用户指定的 PyTorch 2.10 测试为目标，参考 Ascend PyTorch 2.7.1 的测试 diff，按测试语义迁移修改；没有对应 diff 的文件也按适用的通用规则处理。
>
> 核对日期：2026-10-14。本文是修改规则与验证指南，不是全量用例已在 NPU 上通过的证明。

## 1. 阅读与执行方式

如果把本文交给另一个模型，请同时提供原始测试目录、参考 diff 目录、实际目标 PyTorch/torch_npu 源码或版本信息，以及独立输出目录。模型应先阅读第 2～6 节确认边界，再根据第 7～17 节逐文件修改，最后按第 18～22 节验证和输出报告。

本文中的代码示例分为三种：

- **修改模板**：说明如何改某一种已经确认适用的结构，不表示所有文件都要插入这些代码。
- **诊断模板**：在匹配的 NPU 环境中检查设备、版本、导入顺序、框架状态；不能用诊断结果替代原用例。
- **反例**：展示禁止机械照搬的操作。

本报告以“保留原测试意图”为首要原则。适配的目标是让同一个测试在 NPU 路径上检查同一种性质，而不是通过删除断言、修改预期结果、禁用编译或扩大 skip 获得绿色结果。

## 2. 当前工作区、证据等级与版本边界

### 2.1 本次实际核对的文件

| 项目 | 当前路径或情况 | 可以据此得出的结论 |
|---|---|---|
| 原始待处理用例 | `D:\TEST213\test-inductor` | 可作为当前文件内容和测试意图的输入；目录名本身不能证明具体上游版本 |
| 已初步改写用例 | `D:\TEST213\test-inductor_npu` | 可观察之前的修改方式，不应直接当作正确答案 |
| 历史批量脚本 | `D:\TEST213\port_inductor_npu.js` | 可审查此前自动化规则及其缺陷 |
| 历史结果表 | `D:\TEST213\inductor_npu_adaptation_report_all.csv` | 记录 92 个“适配成功”和 97 个“通用规则适配成功”，合计 189 个 Python 文件 |
| 当前 Ascend 源码 | `D:\TEST213\pytorch_ascend` | 可验证当前实现，但不能自动推断 2.10 分支具有完全相同的私有 API |
| 当前 PyTorch 源码 | `D:\TEST213\pytorch_213` | `version.txt` 为 `2.13.0a0`；这里只用于理解框架，不能作为 2.10 API 的最终依据 |
| 历史 2.7.1 diff 路径 | 原任务中的 `ascend_pytorch\test_upstream\test\inductor` | 当前目录已不存在；本文对这些 diff 的描述来自先前处理记录，实施时必须重新提供并核对原 diff |

当前 `pytorch_ascend/version.txt` 列出的构建版本为 2.15.0、2.14.0、2.13.0。因此，下文对 `transfer_to_npu`、NPU Inductor 和测试补丁的源码说明，是**当前工作区可核对的实现行为**，不是对历史 2.7.1 或目标 2.10 环境的无条件保证。

进行正式迁移时，必须记录目标测试的来源 tag/commit 和实际安装包版本。文件夹叫 `pytorch_213` 不代表用户已经改变目标，本文仍按用户指定的 2.10 描述迁移流程；如果目标文件实际来自其他版本，应在结果报告中明确真实来源。

### 2.2 证据等级

| 等级 | 证据 | 允许的表述 |
|---|---|---|
| E1 | 当前源码中明确存在并已检查的实现 | “当前源码实现如此；目标版本仍需核对” |
| E2 | 参考 diff 中明确的修改，且已定位到目标同一测试 | “已迁移该条 diff 的语义” |
| E3 | 从其他测试归纳的通用规则，条件已检查 | “已进行通用规则初步适配” |
| E4 | 实际 NPU 环境执行日志、设备与版本信息 | “该用例/该参数组实机通过” |
| 无证据 | 猜测私有接口、能力、精度或后端支持 | “待确认”，不能写成已经支持 |

### 2.3 本次静态检查事实

在当前目录中重新检查，`test-inductor_npu` 内有 189 个 Python 文件；189 个文件都能被 Python AST 解析，189 个都含有 `from torch_npu.contrib import transfer_to_npu`，177 个含有 `import torch_npu._inductor`。

这些事实只证明文件可解析、包含对应导入，不证明文件能够通过 Python 完整编译。进一步用 `compile(source, filename, "exec")` 做只读静态检查时，185 个文件通过，以下 4 个文件因新增头部位于 `from __future__` 导入之前而失败：

| 文件 | 报错位置 | 应采用的规则 |
|---|---|---|
| `test_cache.py` | 第 11 行 | R01：将初始化放在 future import之后 |
| `test_caching.py` | 第 12 行 | R01 |
| `test_user_streams.py` | 第 17 行 | R01 |
| `extension_backends/triton/device_interface.py` | 第 10 行 | R01、R36：先判断辅助接口是否真的需要迁移初始化 |

统一错误是 `from __future__ imports must occur at the beginning of the file`。这说明仅 `ast.parse()` 通过仍不足以报告完整语法检查通过；必须增加 Python 编译阶段的静态检查。本次只记录这些问题，没有改动测试文件。

即使静态编译通过，也不证明导入时不会报错，不证明测试被收集、没有全部跳过、实际运行在 NPU、触发了 Inductor，或输出与原测试语义一致。本次报告编写没有执行 NPU 实机测试。

## 3. “用 transfer_to_npu 适配”到底意味着什么

### 3.1 它是导入时执行补丁的模块

当前源码在模块末尾执行 `_init()`。因此：

```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu  # 导入触发进程内兼容补丁
```

这属于启用兼容迁移机制。`transfer_to_npu` 在这里是模块，不是一个把测试代码转换完毕的函数。不要写：

```python
# 反例：没有依据的调用方式
# transfer_to_npu(test_module)
# transfer_to_npu()
```

原文件中出现 `.cuda()` 或 `device="cuda"` 不一定说明迁移遗漏：这些调用可能经过兼容补丁被转到 NPU。但必须根据实际包版本检查它是否覆盖该 API，必要时验证结果张量的 `device.type`。

### 3.2 当前实现的主要覆盖范围

当前实现具有 torch 函数、Tensor 方法、Module 方法、CUDA 方法等白名单，并对部分调用参数执行转换。

| 对象 | 当前实现中的机制 | 修改判断 |
|---|---|---|
| `.cuda()` 等已映射方法 | 替换为对应 NPU 方法 | 可在兼容模式保留调用形式；不要据此推断所有 CUDA 方法都存在映射 |
| 部分 torch 张量创建函数 | 白名单函数包装 | 核对具体函数、参数传递方式与版本 |
| Tensor/Module 的 `.to()`、Module 的 `.to_empty()` | 包装并处理设备参数 | 保留 dtype、non_blocking、memory_format 等非设备参数 |
| `device`、`device_type`、`map_location`、`device_id` | 包装器识别这些关键字 | 只在被包装的 API 上保证执行转换 |
| 部分字符串和 `torch.device` 位置参数 | 替换包含 `cuda` 的设备表示 | 不代表任意业务字符串、任意容器都会被语义正确地转换 |
| 部分 `device_ids` 列表/元组 | 设备转换 | 多设备可用性和编号仍需验证 |
| 部分 CUDA stream/event/memory/AMP API | 对应补丁或包装 | 核对方法签名、对象属性和实际行为 |
| `torch.cuda.CUDAGraph` | 当前实现映射到 `torch.npu.NPUGraph` | Graph replay、内存池、RNG 等契约还需单独检查 |
| 部分分布式接口 | 补丁调整设备/通信行为 | 不等价于 NCCL 测试天然变成合法 HCCL 测试 |

包装器的字符串处理在某些分支中采用包含 `cuda` 的字符串替换，而不是完整语法解析。给被包装函数传入与设备无关、但含有该词的字符串也可能受到影响。修改时应核查参数实际含义，不能把这些包装器当作可任意调用的源代码转换器。

### 3.3 它不会自动解决的事情

以下内容不应假定能靠导入自动完成：

1. 新版本私有 API 的移动、签名变化、删除和测试重构。
2. CUDA C++、PTX、CUTLASS、cuBLAS、ROCm/CK 等特定平台实现。
3. 生成代码中的精确文本、FileCheck 模式、日志文案、编译器指令。
4. Python 的普通设备字符串比较、任意字典键、手写设备分支。
5. 已经缓存的 `HAS_GPU`、`HAS_TRITON`、`GPU_TYPE` 和已经求值的装饰器。
6. 测试框架是否实例化 NPU 测试类，以及环境变量是否过滤掉 CPU/NPU。
7. NPU 是否具有对应 dtype、布局、浮点格式、低层 kernel 特性。
8. CUDA 与 NPU 对随机数、错误信息、异步行为、内存统计的差异。
9. 编译后端安装完整性和硬件上的实际编译执行。

### 3.4 全局副作用与运行隔离

这些补丁影响当前 Python 进程。一个文件先导入迁移模块，可能改变后来导入的其他测试。不要在空的 `__init__.py` 中统一启用全部补丁，也不要依赖“先运行某文件，另一个文件才会通过”。

建议以独立子进程运行不同文件、不同迁移模式和不同后端；记录进程的初始化入口。使用一个进程收集大量文件时，必须验证导入顺序不会造成缓存状态不同或 CPU 测试被过滤。

## 4. 适配模式与文件分类

### 4.1 两种可选模式

| 模式 | 做法 | 适用范围 | 必须注意 |
|---|---|---|---|
| 兼容迁移模式 | 导入 `transfer_to_npu`，在必要位置补充 NPU 框架/后端初始化 | 参考 diff 采用该方式、CUDA 调用较多且测试意图可迁移的文件 | 全局补丁、导入顺序、JIT 变化、API 白名单 |
| 显式设备模式 | `import torch_npu`，明确使用 NPU device/API，按需要初始化 NPU Inductor | 新写 NPU 测试、需要清晰设备语义、兼容补丁破坏原测试意图的文件 | 逐个改 API 和设备分支；不能遗漏框架实例化与后端 |

本批历史修改主要使用兼容迁移模式。不能因此要求所有辅助模块或 CPU 专项测试也导入迁移模块。对某个文件选定模式后，保持内部一致，并在报告里记录原因。

如果同一文件含 CPU 与加速器两类测试，兼容补丁可能影响 CPU 路径，尤其测试初始化模块还可能通过环境变量过滤 CPU。应尽量复用已确认的上游/NPU框架方案；必要时用隔离入口区分两条路径，而不是把原 CPU 测试全部改成 NPU。

### 4.2 按测试意图分类

| 分类 | 典型内容 | 默认处理 |
|---|---|---|
| 普通可迁移加速器测试 | 张量算子、融合、动态形状、autograd、一般控制流 | 初始化、设备/门禁/框架适配，保留核心输入与断言 |
| 纯逻辑/CPU 工具测试 | SymPy、索引化简、字符串处理、Python 配置逻辑 | 通常保留；不要因目录名含 Inductor 就添加 NPU 依赖 |
| CPU 专项代码生成 | C++ CPU kernel、向量化、CPU 编译器选项 | 保留 CPU 测试；如要新增 NPU覆盖，应建立对应独立测试 |
| Triton 专项测试 | Triton kernel、启发式、autotune、源码断言 | 确认 NPU Triton 实现与依赖后适配；不能假定 CUDA Triton 特性全兼容 |
| Graph/异步/内存专项 | graph capture/replay、stream、event、内存池 | 使用专项规则；优先人工复核 |
| 厂商专用测试 | CUDA SM/TMA、CUTLASS、ROCm/CK、特定 PTX | 先判断是否有明确 NPU 等价能力；没有则保留原路径并标记不适用/待人工 |
| 辅助模块/扩展后端 | `__init__.py`、工具、mock device interface、fixture | 只修改实际设备相关位置，避免污染包导入和外部后端 |
| 快照/资源 | `.txt`、`.cpp`、模板、数据文件 | 默认原样复制；确需更新时说明对应生成器与语义 |

文件名只能辅助分类。比如 `test_indexing.py` 可能既包含纯逻辑测试，也包含生成 Triton 代码的测试，应细化到类/方法层面。

## 5. 基础初始化规则

### R01：导入位置必须满足 Python 语法与初始化顺序

**触发条件**：该测试采用兼容迁移模式，且需要在导入设备相关测试框架之前启用补丁。

插入位置必须在 shebang/编码声明、模块 docstring、所有 `from __future__ import ...` 之后。保留许可证、Owner 注释和模块说明，不要把新 import 放到 future import 前。

```python
"""Inductor arithmetic tests."""
from __future__ import annotations

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu  # noqa: F401

# 对实际需要 NPU Inductor 的文件初始化；目标版本必须提供此模块。
import torch_npu._inductor  # noqa: F401

# 原文件后续的设备探测、框架和测试依赖导入放在初始化之后。
```

这是最小结构模板，不是所有文件都必须使用的统一头。`torch_npu.testing`、JIT 恢复和 Triton 探测修补需按 R02～R04 判断。已存在的 import 要复用；重复执行适配应无额外变化。

**禁止**：直接字符串前置到整份文件；重复插入头部；重新排序所有 import 导致依赖失效；吞掉导入错误后继续测试。

**验证**：AST 解析；模块 docstring 仍被识别；future import 顺序合法；在新进程中导入并观察实际探测值。

### R02：`torch_npu._inductor` 用于编译路径，不能只导入迁移模块

**触发条件**：测试调用 `torch.compile(..., backend="inductor")`，或直接使用 NPU Inductor lowering、调度、wrapper、autotune 等能力。

根据匹配目标版本的实现，在设备相关测试框架探测前完成 NPU Inductor 初始化。当前源码中，该模块包含设备接口、后端注册和多种编译路径初始化逻辑；仅 `import torch_npu` 或仅导入 `transfer_to_npu` 不能作为后端已就绪的证据。

当前源码使用 `TORCHINDUCTOR_NPU_BACKEND` 选择 `default`、`mlir`、`dvm`、`ascendc`、`triton_experimental` 等路径。这是当前实现说明，不应把这些选项直接写进 2.10 测试而不核对。

`backend="inductor"` 通常继续保持。把普通 Inductor 测试全部改成 `backend="eager"`、`aot_eager`、某个其他编译后端或 `npugraphs`，会改变测试对象，禁止作为默认适配。

**例外**：纯逻辑测试、纯 CPU helper、不测试编译的工具模块通常不需要导入这个模块。

### R03：`torch_npu.testing` 不是无副作用的辅助 import

**触发条件**：文件依赖 Ascend 测试框架中的设备映射、OpInfo 行为、skip 管理或设备测试类初始化。

当前 `torch_npu.testing` 的导入会执行测试补丁，并设置 `PYTORCH_TESTING_DEVICE_FOR_CUSTOM='privateuse1'`、`PYTORCH_TESTING_DEVICE_EXCEPT_FOR='cuda,cpu'`，以及修改部分 `onlyCUDA`、dtype 和 skip 行为。因此：

- 需要它时，应采用对应版本已验证的初始化顺序。
- 需要保留 CPU 覆盖时，不能不加检查地统一导入它；应明确设备过滤策略，并验证 CPU 类仍被收集。
- 它会导入上游测试基础模块，可能使一些状态提前求值；后面再修补 `has_triton` 不一定能修复已经导入的常量。
- 私有测试框架通常需要额外依赖和匹配源码；普通 PyTorch wheel 能导入 `torch` 不代表这些测试模块可用。

**验证**：记录相关环境变量、生成的测试类名、设备类型、skip 数量；对比原来的 CPU 与加速器覆盖。

### R04：Triton 探测补丁必须检查顺序和后端

当前 `transfer_to_npu` 会把多处 `has_triton` 设为返回 False 的函数。当前 `_dynamo.patch_has_triton()` 则把 `torch.utils._triton.has_triton` 替换为 NPU 感知的实现；NPU Inductor 也对部分内部位置进行修补。这些不是相同的补丁。

**触发条件**：需要 NPU Triton 路径，且确认当前目标版本在迁移模块导入后存在错误的探测结果。

```python
# 修改模板：只用于已确认走 NPU Triton 的测试环境。
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
from torch_npu.utils import _dynamo

_dynamo.patch_has_triton()

# 如该文件需要 Ascend 测试补丁，在探测修补后再导入。
# 注意 R03 中的 CPU 过滤副作用。
import torch_npu.testing  # noqa: F401

from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
```

该模板仍需目标版本实测，不能保证内部已经提前导入的所有常量都得到刷新。若测试仍被错误跳过，应检查常量的来源、缓存与装饰器求值时机，而不是追加更多无依据的全局 patch。

**禁止**：`HAS_TRITON = True`；把探测函数改成恒真；对非 Triton 后端无条件套此规则；认为 `_dynamo.patch_has_triton()` 会修补所有内部模块。

**验证**：真实 NPU 可用、正确 Triton 包可用、实际使用 NPU 编译路径，并检查 `HAS_TRITON`、`GPU_TYPE`、`HAS_GPU`、类装饰器结果。任一个布尔值为 True 都不能单独证明 kernel 能编译运行。

### R05：JIT 行为需要单独恢复，不能把测试变成普通 Python 执行

当前迁移实现会补丁 `torch.jit.script` 和 `torch.jit.script_method`。默认 `_dynamo.use_jit_script=False` 时，补丁后的 `torch.jit.script` 直接返回输入对象；设为 True 后才调用保存的原始 script 实现。`script_method` 的处理又不同，不能假定该标志同时恢复它。

**触发条件**：测试真的依赖 TorchScript 编译或其与 Inductor/Dynamo 的交互。

```python
from torch_npu.utils import _dynamo

# 当前实现中，用于恢复 torch.jit.script 的实际调用。
# 必须在依赖 script 的测试定义/执行前设置，并确认目标版本支持。
_dynamo.use_jit_script = True
```

**选择原则**：普通 Inductor 测试不要因为历史头部有这个标志就全部添加；纯 TorchScript 语义测试优先判断是否应避开兼容迁移模式。需要 `script_method` 的测试应查目标实现，不能用这个开关伪装恢复。

**验证**：被测对象确实经过预期的编译流程；保留原来的编译失败/成功断言与异常类型。不能只确认调用没有抛异常。

## 6. 设备表达、框架常量和设备实例化

### R06：先识别设备字符串的角色，再决定是否修改

| 出现位置 | 处理建议 |
|---|---|
| 张量创建/迁移 API 的 device 参数 | 兼容模式核对包装覆盖；显式模式改为 `npu`，保留 index |
| 类属性 `device_type` | 该类确实是 NPU测试时使用 `npu` 或已验证的 NPU-aware 常量 |
| `if device.type == "cuda"` | 判断这是普通加速器分支还是 CUDA 专用能力分支，再改 |
| `only_for` / `except_for` | 检查目标版本的注册类型与过滤逻辑，不只是替换文本 |
| 生成代码片段/日志/错误消息 | 按实际 NPU实现更新，不能按设备词替换 |
| 配置键/计数器/模块路径 | 查真实对象，很多名称仍保留 `cuda`/`cudagraph` |
| 类名/方法名/测试 ID | 通常保持，以便定位 diff 和对比测试结果 |
| 外部 C++/PTX 字符串 | 不属于普通设备参数迁移，按专项规则处理 |

显式模式例子：

```python
# 原始
x = torch.randn(8, 16, device="cuda:1")
model = model.to("cuda:1")

# 修改模板：只有当第二张 NPU 可用且测试确需该编号时使用。
x = torch.randn(8, 16, device="npu:1")
model = model.to("npu:1")
```

不要把 `cuda:1` 默认改成 `npu:0` 来规避单卡环境；多卡测试可依据真实 device_count 有理由地跳过，但测试语义仍需保留。

### R07：保留 CPU、meta、fake 和测试构造设备的意图

`cpu` 可能是参考计算设备；`meta` 可能用于 shape inference；FakeTensor 的逻辑设备与真实执行设备不同。不要统一改成 `npu`。

示例：CPU 参考输出与 NPU compiled 输出的比较，应继续保留 CPU 参考路径。如果测试检查混合设备错误，也不能把两个设备合并为同一个 NPU。

对于 mock device interface 和扩展后端，它们可能刻意模拟 CUDA 或自定义设备；需要先查测试是否是在验证抽象接口，而不是执行真实 CUDA 运算。

### R08：`GPU_TYPE`、`HAS_GPU`、`RUN_GPU` 不一定天然适用于 NPU

当前上游测试框架会在导入阶段计算部分常量，`HAS_GPU` 的表达式可能只考虑 CUDA/XPU 与 Triton；`GPU_TYPE` 来源可能被 NPU 补丁改成 `npu`。这意味着名称相同的常量不一定表示同一种能力。

**检查步骤**：

1. 在目标版本中找到定义与 import 时间。
2. 确认迁移初始化是在这些计算之前还是之后。
3. 确认 `HAS_GPU` 表示“可用加速器”还是“特定 Triton 路径可用”。
4. 确认 `RUN_GPU` 还受测试类注册与环境变量过滤影响。
5. 对当前模块内的门禁做最小修正，不篡改全局框架以让所有测试恒真。

对于独立、显式 NPU 测试，可定义语义明确的变量：

```python
HAS_NPU = torch.npu.is_available()
NPU_TYPE = "npu"
```

但 `HAS_NPU` 只表示设备可用，不表示编译器、Triton、某种 dtype 或 graph 特性可用。不能把所有 `HAS_GPU`/`HAS_TRITON` 都替换为它。

### R09：设备测试类必须真正实例化

**触发条件**：文件使用 `instantiate_device_type_tests`、自定义 device-specific test base，或从 generic class 自动生成测试。

检查目标版本的函数签名、PrivateUse1/NPU注册机制、过滤环境变量和最终类名。当前核对的上游签名包含 `only_for`、`except_for`、`allow_mps`、`allow_xpu`，没有据此确认的通用 `allow_npu=True` 参数；禁止凭直觉添加这个参数。

```python
# 修改模板：先确认框架中的 base.device_type 确实为 npu。
instantiate_device_type_tests(TestExample, globals(), only_for=("npu",))
```

如果框架注册名仍为 `privateuse1`，或 Ascend 框架内部已映射类型，应按该版本真实逻辑处理，不能强行使用上述模板。原来同测 CPU 的文件不应无理由改成只测 NPU。

**验证**：收集结果存在实际 NPU 类和方法；参数化实例数合理；测试不是零项；CPU 覆盖如计划保留则确实存在。

## 7. 装饰器、skip 条件和入口门禁

### R10：区分“设备身份门禁”和“能力门禁”

**设备身份门禁**：仅因为上游没有 NPU 类型而禁止一般加速器测试，例如某些 `requires_gpu()`、`onlyCUDA`。

**能力门禁**：检测真实依赖或特性，例如 Triton 包、TMA、特定 SM、内存大小、多卡、操作系统、编译器、分布式服务、已知未实现算子。

前者可在确认测试可迁移后替换为 NPU门禁；后者必须保留或改成真实 NPU等价能力检测。不能仅凭函数名 `requires_gpu` 就认定它只检查设备身份，必须查看定义。

### R11：优先明确替换装饰器，而不是批量删除

```python
# 原始：该测试已确认只需要普通加速器计算。
@requires_gpu()
def test_pointwise(self):
    ...

# 修改模板：unittest 已导入，且初始化顺序正确。
@unittest.skipUnless(torch.npu.is_available(), "requires NPU")
def test_pointwise(self):
    ...
```

删除装饰器只适用于已经由类级别/设备框架保证相同前置条件的情况，且应记录该保证。参考 diff 删除装饰器，不表示其他新文件中的同名装饰器也都可删。

保留 `@parametrize`、`@dtypes`、`@slowTest`、配置 patch、平台/内存限制和 unrelated skip。多行装饰器应整块解析，不得只删除第一行。

### R12：组合条件按布尔语义迁移

```python
# 原始：CUDA 身份 + 独立环境限制
@unittest.skipIf(not HAS_GPU or IS_WINDOWS, "requires supported accelerator environment")
def test_example(self):
    ...

# 修改模板：确认这里的 HAS_GPU 仅代表一般设备可用后。
@unittest.skipIf(
    not torch.npu.is_available() or IS_WINDOWS,
    "requires supported NPU environment",
)
def test_example(self):
    ...
```

若 `HAS_GPU` 实际包含 Triton 要求，转换后还必须保留相应依赖条件。不得把组合条件整体删掉，或直接改成 `skipIf(False, ...)`。

类级 skip、setUp 中的 `SkipTest`、测试函数内条件、pytest mark 和入口条件都属于门禁；只改装饰器不等于覆盖完成。

### R13：多卡与显存/内存门禁使用真实 NPU 状态

检查 `torch.npu.device_count()` 与目标 API 返回的设备容量。不要拿 CUDA属性对象直接套用字段名；不能通过假设无限内存删除 OOM 防护。内存阈值如果改变，必须说明为什么 NPU测试仍覆盖相同性质。

单卡机器运行多卡测试时，可保留明确的“至少两张 NPU”跳过条件；它属于环境不足，不是适配失败，也不能记为运行通过。

### R14：`__main__` 入口只迁移确实不适用的设备条件

```python
# 原始
if __name__ == "__main__":
    if HAS_CPU or HAS_GPU:
        run_tests("sympy")

# 修改模板：纯逻辑测试依赖条件已在其他位置正确保留时。
if __name__ == "__main__":
    run_tests("sympy")
```

如果是 NPU-only 文件，也可以改为实际 NPU 可用条件；必须避免“无 NPU 时无声退出返回 0”被记录为通过。保留 `run_tests` 的所有参数、依赖名、初始化、环境限制和错误处理。

**禁止**：把所有 `if HAS_*` 改成 `if True`；用正则移除整块条件再盲目缩进；删除 OS/编译器/sanitizer 等限制；把 runner 名或依赖参数丢掉。

## 8. 硬件能力与 Triton 专项规则

### R15：不要把 NPU 伪装成某个 NVIDIA SM 作为通用规则

历史适配头部包含：

```python
# 历史兼容写法，不是推荐的通用规则。
torch.cuda.get_device_capability = lambda: (10, 0)
```

这会向调用方声称具有 NVIDIA 某种 SM 能力，可能开启 NPU 不具备或不同实现的路径。此外零参数 lambda 不能正确处理带 `device` 参数的调用。

**默认处理**：不新增这种全局伪造。如果旧输出存在，要逐个查使用位置。一般可迁移测试中的 NVIDIA专用能力判定，应替换为真实 NPU能力检查或使用 NPU测试路径。

**特殊例外**：被测对象本身是启发式逻辑，测试刻意模拟硬件属性，不运行对应硬件 kernel。这种场景可以在单个测试内用目标精确的 `mock.patch` 保留模拟输入，但它属于 mock 逻辑测试，不能当作真实 NPU硬件支持。

```python
# 修改模板：仅用于测试模拟属性的逻辑，不用于启用真实硬件能力。
with mock.patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)):
    result = evaluate_mocked_heuristic()
```

若确需旧 diff 的兼容绕过，应记录限定原因、作用域、接口签名和风险，单独列为待实机验证；不能推广到整个测试目录。

### R16：Triton 安装与 Triton 能力不是同一件事

有 `triton` 包不代表安装了匹配 NPU 的实现，也不代表 NPU kernel 后端可用。检查版本、后端注册、设备接口、一次最小编译执行，以及被测特性的支持范围。

迁移 `DeviceProperties.create(torch.device("cuda"))` 这样的属性构造时，应指向实际 NPU设备；后续使用的属性字段也必须是 NPU实现提供的，不能仅修改构造参数。

### R17：TMA、warp、线程组织等底层概念按实现判断

NVIDIA SM、TMA、warp、PTX 指令、共享内存限制不应机械对应到 Ascend 某个字段。有些 NPU Triton 实现可能提供同名或兼容功能，因此也不能仅看到 `TMA` 就断言永远不可迁移。应以**目标版本的实际能力实现与被测语义**为准。

无法确认时，保留能力门禁，标记“待能力验证”。没有等价实现的厂商专项测试应标记“原测试目标对 NPU 不适用”，不能删除测试后记成功。

### R18：厂商专用库/后端不能靠字符串替换迁移

CUTLASS、CK、ROCm、CUDA扩展、PTX、特定 halide/C++后端等，应判断是否存在对应 NPU实现。如果要新增 Ascend/CATLAS/其他后端的测试，这通常是新的后端实现工作，不能由通用规则凭空补齐。

保持原厂商覆盖与新增 NPU覆盖的关系清晰；明确“不适用”与“待人工适配”，避免通过全 skip 得到假成功。

## 9. 编译路径、断言与生成代码

### R19：保留编译器测试的核心契约

适配前先写下该测试检查什么，例如：输出一致、必须单图、特定融合、某个 pass 触发、无 graph break、错误应抛出、内存复用、代码特征。

以下内容默认保留：输入形状和布局、动态维度、requires_grad、alias/in-place 关系、`fullgraph`、`dynamic`、编译次数、计数器、graph break 断言、预期失败类型、数值参考。

禁止为了通过而修改 `fullgraph=True` 为 False、禁用优化、调用 eager 替代 compiled、catch 编译异常后返回、删除失败断言。确因 NPU行为需要不同配置时，应说明对原测试意图的影响并标记专项修改。

### R20：FileCheck、生成源码和日志断言需要真实 NPU 证据

先区分通用计算语义与特定代码形式。例如某个 fusion 行为可能在 NPU仍存在，但 kernel名、wrapper、指令、launch syntax 不同。

处理顺序：

1. 保留数学与图结构断言。
2. 查目标 NPU codegen 的对应实现。
3. 获取实际生成代码/日志，定位同一种性质的稳定特征。
4. 用最小必要的断言更新替换厂商特征。
5. 尚无输出证据时保留待审查项，不自行编造新的 expected 字符串。

不能把 `check("cuda")` 简单改成 `check("npu")`，也不能删除整个 `FileCheck` 链，或把精确断言退化成“非空字符串”。

### R21：内部模块路径、计数器和配置名称以实际实现为准

`torch._inductor.cudagraph_utils`、`triton.cudagraphs`、`cudagraph_skips` 等名称可能仍被 NPU路径复用。API名称带 CUDA不等于必须重命名。

修改之前查对象定义与调用点。使用一个并不存在的 `triton.npugraphs` 配置键比保留已有 CUDA名称更错误。不要对模块路径、方法名、计数器键做全局 `cuda→npu` 替换。

### R22：私有 API 的版本迁移按符号与语义完成

`torch._inductor.*`、`torch._dynamo.*`、`torch.testing._internal.*` 与 `torch_npu.utils._dynamo` 都可能变动。2.7.1 diff 中的 import 在 2.10 文件中可能已经移动，函数可能增加参数或变成其他 abstraction。

迁移步骤：找到 2.10对应对象 → 阅读定义/调用点 → 定位同一测试意图 → 使用 2.10等价接口 → 记录映射。没有找到对象时列为待人工，不通过模糊匹配随便选择相似名字。

删除原版本的测试不应该被旧 diff 重新复活；新版本新增的测试必须重新分类，不应假定旧规则已经覆盖它。

## 10. NPU Graph、流、事件与内存

### R23：只在 Graph 专项语境中迁移 `cudagraphs` 后端

```python
# 原始：确实是在比较专门的 Graph 后端。
compiled = torch.compile(fn, backend="cudagraphs")

# 修改模板：目标版本已注册 npugraphs，并支持本测试语义。
compiled = torch.compile(fn, backend="npugraphs")
```

普通 `backend="inductor"` 不改。Graph参数化列表中的特定 `"cudagraphs"` 分支可按语义改为 `"npugraphs"`，但所有断言与配置键都要单独核对。

`npugraphs` 是具体后端名称，不是“任何 NPU编译”的通用替代词。动态形状、输入 mutation、RNG、capture/replay、嵌套调用和重编译行为取决于目标实现，不能默认与 CUDA完全一致。

### R24：Graph 对象与上下文按真实接口迁移

显式模式下检查 `torch.npu.NPUGraph`、graph capture context、graph pool、replay、reset 等接口。兼容模式下即使类已映射，也要检查参数和方法语义。

Graph测试应保留对重复 replay 输出、输入地址/内容变化、生命周期、分配行为、随机状态的检查。不得以普通 eager计算代替 capture/replay。

### R25：Stream/Event 必须核对对象属性和同步契约

`torch.cuda.Stream/Event` 的映射不意味着对象上的 `cuda_stream` 属性可直接使用。当前 NPU代码中有 `npu_stream` 用法；但实际需要哪种句柄，应根据调用方和目标接口判断。

检查 stream context、wait_event、record_event、设备归属、跨流同步、Event计时单位与支持范围。不要添加全局 synchronize 来掩盖原本测试的异步行为；若只是避免读取尚未完成的输出，放在需要读取的准确位置。

### R26：内存统计接口和结构不能只换名称

Graph/内存测试中可存在 `torch.npu.memory_snapshot()` 等对应 API，但必须核对返回结构、segment/block字段、allocation语义、峰值与缓存行为。

历史 Graph适配出现了 CUDA snapshot 到 NPU snapshot 的迁移，这是专项规则，不能推断所有 allocator诊断 API都有同名 NPU版本。

只在目标实现确实提供时修改 `memory_allocated`、`max_memory_allocated`、`reset_peak_memory_stats`、`empty_cache` 等调用。字节级预期值、地址复用、计数器和 OOM错误信息必须有实现或日志依据。

## 11. dtype、数值、随机数、AMP 与布局

### R27：不默认改变 dtype

原测试的 float64、complex、float8、低位整数等可能正是被测性质。NPU不支持时，先查目标算子与后端支持范围，使用细粒度有依据的 skip/xfail或标记待适配，不要统一改成 float32。

当前迁移实现包含某些 CUDA double构造到 NPU float构造的兼容行为。这正说明“调用没报错”不保证原 dtype语义。dtype专项测试必须显式检查结果 dtype，必要时避开这种兼容转换。

dtype列表调整必须说明被排除的具体参数组、理由和覆盖影响。不能删掉整个 dtype装饰器后只测默认 dtype。

### R28：精度阈值变化需要测量依据

保留原 `rtol`、`atol`、NaN/Inf、signed zero、溢出和 rounding断言作为默认。发生误差时先检查 device、dtype、AMP、输入、eager与compiled路径、精度模式，再判断误差是否合理。

若确需 NPU专用阈值，应对特定算子/dtype限定，并记录最大误差、参考设备、种子、版本与芯片。不能把所有 `assertEqual` 替换成粗略 allclose，不能全局扩大容差。

### R29：随机种子与 Generator 需要保留测试目标

将需要 NPU执行的 Generator创建改到真实 NPU设备，检查 seed、state、graphsafe行为的接口支持。保留 CPU参考 RNG，与 NPU RNG相互隔离。

同一 seed下 CUDA、CPU、NPU不一定产生相同随机序列。如果原测试检查确定性，应比较同一 NPU路径重复运行；如果原测试本身检查随机算法精确序列，不能擅自换成统计性质。

记录 seed，验证 eager/compiled对 RNG消耗和状态的要求。随机差异不能用删除输出断言解决。

### R30：AMP、autocast 和 GradScaler 按真实 NPU API 检查

兼容模式可能包装部分 AMP调用。显式模式中，若目标版本支持，可采用设备类型明确的 `torch.autocast(device_type="npu", dtype=...)` 等形式；GradScaler入口与构造参数需要检查该版本接口。

保留 enabled开关、dtype、scaling、overflow检测和梯度断言。不要关闭 autocast使测试失去原目标。

### R31：布局、stride、alias 与 in-place 不得简化

contiguous、channels_last、非连续 stride、view、storage_offset、alias、in-place及输入 mutation通常是 Inductor测试关键。不要为了 NPU运行方便统一添加 `.contiguous()`、`.clone()` 或转成标准布局。

如果某格式确实不支持，应记录具体组合与能力限制；若使用 NPU特定格式接口，说明它与原被测布局的关系。

## 12. 错误、进程、分布式与环境

### R32：错误消息只修改设备相关且已验证的部分

保留异常类型和关键错误性质。设备名、allocator或编译器消息可能不同，只有获取实际 NPU错误或找到目标实现后，才能更新精确字符串/正则。

不要把 `assertRaisesRegex` 改为 catch所有 Exception；不要使用 `.*` 代替整条预期；不要因没有错误而删除负例测试。

### R33：子进程也需要初始化与版本一致性

minifier/repro、distributed、subprocess脚本或动态生成 `.py` 文件可能在新的 Python进程中运行，不会自动继承父进程的 monkey patch。

检查生成的复现程序、子进程环境、工作目录、包路径和初始化语句；只适配外层测试而不适配被生成的代码，可能导致复现仍走 CUDA或没有 NPU后端。

进程启动方式、初始化位置和多进程安全性按目标环境验证；不强行套用与 CUDA完全相同的进程方案。

### R34：NCCL 到 HCCL 属于通信语义迁移

如果测试的是一般分布式张量计算，可确认 HCCL支持后适配 backend与device。若测试的对象是 NCCL专属行为、错误信息或选项，则不能只换成 HCCL。

保留 world_size、rank、设备绑定、超时、collective与结果断言。检查对 rendezvous、网络、权限与多设备的实际依赖，失败原因按环境/实现分类。

### R35：平台和构建依赖限制默认保留

保留 Linux/Windows/macOS、C++编译器、sanitizer、缺失库、线程配置等有意义的限制。本地 Windows可做语法检查；匹配的 Ascend运行环境才可完成 NPU验证。不要把 Windows静态通过写成 NPU执行通过。

## 13. 辅助文件与目录复制规则

### R36：辅助模块默认最小修改

`__init__.py`、配置常量、工具、fixture、扩展后端注册、mock接口不是都要导入 torch_npu/迁移模块。尤其空的 `__init__.py` 不应变成全局设备初始化入口。

仅在该模块本身需要 NPU对象或必须承担明确初始化职责时添加依赖，并说明副作用。否则保留原内容，以免影响不相关测试、CPU-only运行、包可导入性和扩展后端验证。

### R37：输出保持目录结构与资源完整性

建立新的输出目录，保留相对路径、文件名、非 Python资源和包结构。不要覆盖原测试，不删除已有输出中的用户改动。复制/更新前确认输出路径与原目录不同，且存在的输出属于本任务。

生成代码、快照、模板和 fixture需要成套一致。只复制 `.py` 可能导致资源丢失；反过来对所有文本执行设备词替换也不合法。

相对 import和资源访问应在新目录结构中检查，不能仅用绝对 import偶然绕过包结构问题。

## 14. 2.7.1 diff 向 2.10 的语义迁移流程

### 14.1 文件映射

1. 使用精确文件名/相对路径匹配，如 `test_indexing.diff` → `test_indexing.py`。
2. 对同名不同目录文件检查 diff header和上下文，避免 basename冲突。
3. 没有精确匹配时，查历史重命名/拼写错误，建立人工审查的别名表。
4. 任何模糊匹配只作为候选，不自动采用。

历史记录中出现过 `test_loop_orderiing.diff` 与 `test_loop_ordering.py`、`test_minifer*.diff` 与 `test_minifier*.py`、`test_move_constructors_to_cuda.diff` 与 `test_move_constructors_to_gpu.py` 之类候选关系。它们需要重新检查 diff及目标内容，不能直接视为已确认映射。

### 14.2 逐 hunk 转换为修改意图

每个 hunk至少记录：原类名/函数名、修改类别、设备/能力前提、核心改动、附带改动、2.10对应位置、最终处理状态。

例如包含“增加迁移 import、恢复 JIT、替换设备常量、删除装饰器”的 hunk，必须拆成多个意图。完成第一条 import不能把整个 hunk记完成。

### 14.3 定位对应测试

优先依次使用：完整类名+函数名 → 方法体关键调用/断言 → 参数化结构 → 相邻上下文 → 版本历史。行号只能用于提示，不能作为跨版本可靠定位依据。

同名方法出现在不同类中时必须使用类作用域；重构为多个测试时，应分别检查旧改动的意图是否仍然存在。装饰器离函数较远或多行时，使用 AST/CST结构定位。

### 14.4 应用与状态

| 状态 | 含义 | 是否计为修改已解决 |
|---|---|---|
| applied_exact | 精确应用且检查完整改动 | 是 |
| applied_semantic | 已按目标版本等价实现迁移所有意图 | 是，需映射证据 |
| already_satisfied | 原文件已经满足该意图 | 不需重复修改，需证据 |
| obsolete_in_target | 对应测试/路径在目标版本已移除 | 不需要新增，明确记录 |
| not_applicable | 被测平台/特性对 NPU无等价目标 | 不当作适配通过 |
| unresolved | 目标不明、改动不完整或版本不匹配 | 否，保留未解决项 |

不得通过“只要文件包含 torch_npu就算 applied_semantic”或修改统计数字来消除未解决项。多个候选装饰器出现时，应停止该意图自动应用，不能删除第一个匹配项。

### 14.5 典型参考语义与可迁移范围

| 历史文件 | 历史记录中的主要意图 | 迁移注意 |
|---|---|---|
| `test_indexing` | 迁移初始化、NPU Inductor、主入口调整 | 文件包含纯逻辑测试；保留 `run_tests("sympy")` 等依赖参数 |
| `test_control_flow` | 对选定方法去除 GPU门禁 | 对每个方法判断；不扩展为全文件删除所有门禁 |
| `test_smoke` | 选定 GPU skip与入口调整 | 保留其他能力/依赖条件 |
| `test_triton_kernels` | 选定门禁修改，部分能力限制仍保留 | 特别注意 Triton/TMA专项能力 |
| `test_triton_heuristics` | NPU设备类型、设备属性构造 | 检查字段含义、常量缓存与实际 NPU后端 |
| `test_cudagraph_trees` | NPU Graph后端、设备调用、内存快照、特定消息 | 保留真实存在的 `cudagraph*`配置/计数器，禁止全局重命名 |

本表是定位指南。因为原 diff目前不在工作区，后续模型必须阅读重新提供的 diff，不能仅凭本表声称 hunk全覆盖。

## 15. 没有 diff 的文件如何初步适配

“没有 diff”是参考资料匹配状态，不是测试适配难度，也不是运行结果。没有 diff可能是新版新增测试、重命名、纯工具文件或旧参考未覆盖。

逐文件流程：

1. 分析测试意图与文件类型，不从文件名直接判断。
2. 检查是否需要 NPU真实执行；纯 CPU/纯逻辑文件可以保持原样并说明原因。
3. 有一般加速器语义时，选择兼容/显式模式，应用必要初始化。
4. 审查设备参数、比较分支、框架常量和设备类实例化。
5. 检查所有装饰器、类门禁、setUp、内部 skip与main入口。
6. 保留或改造真实能力门禁；Graph/Triton/厂商专项另行审查。
7. 检查 dtype、精度、RNG、布局、错误、代码生成断言。
8. 检查子进程/生成 repro和资源文件。
9. 做语法与结构检查，记录仍需版本/硬件确认的项。
10. 在 NPU环境中按验证等级逐步运行。

没有 diff时，不允许把“已插入统一 import头部”称为完整适配。可以称“通用规则初步适配，未实机验证”，并列出实际适用的规则编号。

以下四种结果都合理：已做通用规则初步适配；无需 NPU修改且保留原样；原测试目标对 NPU不适用；需要专项人工适配。不要强迫所有文件都显示同一个成功状态。

## 16. 必须纠正的批量修改反例

| 反例 | 实际问题 | 正确做法 |
|---|---|---|
| 全局 `cuda→npu` 文本替换 | 损坏模块路径、配置、厂商代码、错误预期与测试 ID | 根据表达式角色逐处判断 |
| 每个 `.py` 都加相同头 | 给纯 helper和包引入全局副作用；可能破坏future import | 按文件职责与模式初始化 |
| 全删 `requires_gpu/cuda/triton` | 真正依赖与能力门禁被清除 | 精确定位并保持有效前提 |
| 将 capability设为 `(10,0)` | 伪造硬件，可能进入不支持路径 | 真实能力检查或限定mock逻辑 |
| 一个 hunk只加了import就返回成功 | hunk的其他语义丢失 | 拆分意图逐项记录 |
| 找不到目标就删第一个同名装饰器 | 修改了不相关测试 | 类/函数唯一定位，否则待人工 |
| 把main条件统一改成 `if True` | 环境依赖和限制被删除 | 只处理经确认的设备身份条件 |
| 为通过改 eager/删除断言 | 不再测试原 Inductor性质 | 保留编译路径和核心断言 |
| 统一改 float32/放大容差 | 掩盖dtype与数值错误 | 有测量依据的细粒度调整 |
| 全部 skip然后记通过 | 没有实际覆盖 | 单独统计executed/skip/xfail |
| 语法通过就称NPU通过 | 没检查导入、设备和运行 | 按验证等级描述 |

## 17. 当前脚本与旧结果表的使用限制

当前 `port_inductor_npu.js` 不能作为本报告的可信自动执行器，原因包括：

1. 它仍引用旧目录 `inductor`、`ascend_pytorch/test_upstream/test/inductor` 和 `inductor_npu`，与当前路径不同。
2. 它会删除已有输出目录再复制。不能直接重跑来生成本报告或“验证”结果。
3. 某些 import补丁分支直接返回成功，未核查同一 hunk的其余修改。
4. 多个相同装饰器候选存在时可能删除第一个，而没有确认类/方法作用域。
5. 入口门禁与缩进通过字符串/正则处理，可能丢失额外条件。
6. 有 `if (applied !== hunks.length) applied = hunks.length;`，会将未完成 hunk强制计为已完成。
7. 未应用报告写入固定的“全部语义已迁移”断言，而不是实际未解决项。

所以旧 CSV中的 `hunks_applied` 和“适配成功”不能作为全部差异已正确处理的证据。它仍可用来了解此前文件名与 diff匹配记录，但匹配状态也应在原 diff重新可用后复核。

本次仅写规则报告，没有重跑该脚本、修改测试或重写旧 CSV。后续若要修复自动迁移流程，应按第 14节重新计算逐意图状态，保留真实未解决列表。

## 18. 验证等级与推荐顺序

| 等级 | 检查内容 | 可排除的问题 | 不能证明 |
|---|---|---|---|
| V0 | 版本/输入/输出路径、目录资源完整性 | 处理错版本、资源遗漏、路径覆盖 | 代码正确 |
| V1 | AST解析、Python静态编译、future/docstring/import结构、修改审查 | 语法错误、编译阶段语法限制、部分结构破坏 | 导入和运行可用 |
| V2 | 匹配环境中导入与测试收集 | 缺包、私有API不存在、类未生成 | 用例执行成功 |
| V3 | NPU可用、结果设备与最小编译smoke | 设备未启用、普通compile路径不通 | 所有测试特性支持 |
| V4 | 代表性用例/参数组执行 | 对应已执行路径的错误 | 未执行组合通过 |
| V5 | 全量执行并核对覆盖与skip | 全量已执行用例问题 | 未覆盖硬件/版本组合通过 |

V1通过后可交付“初步改写可解析”。只有有日志、环境和实际执行记录，才能报告相应范围的实机通过。发现环境阻塞应保留代码产物、日志和准确原因，不假造后续验证。

### 18.1 静态检查示例

以下脚本只读地解析并编译Python文件，不执行编译结果，不导入torch_npu；它不会证明语义正确。将 `ROOT` 改为输出目录。`compile()` 不能省略，因为 AST解析可能接受编译阶段才拒绝的结构。

```python
import ast
from pathlib import Path

ROOT = Path("/workspace/test-inductor_npu")
errors = []
for path in sorted(ROOT.rglob("*.py")):
    try:
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        compile(source, str(path), "exec")  # 只编译，不执行，不触发测试或NPU初始化
    except (SyntaxError, UnicodeError) as exc:
        errors.append((str(path), str(exc)))
        continue

    # AST可辅助查future导入位置。这里还应结合原文件比较docstring与import结构。
    seen_non_future = False
    for index, node in enumerate(tree.body):
        is_docstring = (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        if is_docstring:
            continue
        is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
        if is_future and seen_non_future:
            errors.append((str(path), f"future import位置错误: {node.lineno}"))
        if not is_future:
            seen_non_future = True

for item in errors:
    print(item)
if errors:
    raise SystemExit(1)
print("static syntax/structure checks passed")
```

AST和静态编译只能发现部分问题。建议用CST或保留格式的定位方式修改，并人工审查diff；AST直接重新生成整个文件会带来不必要的格式、注释损失。

### 18.2 NPU环境与最小compile诊断示例

下面应在匹配版本的Ascend运行环境中以新进程运行。它采用显式设备模式，验证普通Inductor路径；兼容迁移模式还需要按被测文件相同顺序做独立诊断。

```python
import torch
import torch_npu
import torch_npu._inductor  # noqa: F401

print("torch:", torch.__version__)
print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))
print("npu available:", torch.npu.is_available())
if not torch.npu.is_available():
    raise SystemExit("NPU unavailable; no runtime validation performed")
print("npu count:", torch.npu.device_count())
print("npu name:", torch.npu.get_device_name(0))

torch.manual_seed(1234)
x = torch.randn(64, device="npu", dtype=torch.float32)

def fn(t):
    return torch.sin(t) + t * t

expected = fn(x)
compiled = torch.compile(fn, backend="inductor", fullgraph=True)
actual = compiled(x)
torch.npu.synchronize()
assert actual.device.type == "npu", actual.device
assert actual.dtype == expected.dtype
torch.testing.assert_close(actual, expected)
print("minimal NPU Inductor execution passed")
```

这个诊断证明该函数在所列环境下可执行，不替代原用例。`fullgraph=True` 限制了graph break，但单靠此示例不能证明某个特定fusion、kernel来源或专项后端行为；测试这些性质时需保留对应断言或使用目标版本真实的编译诊断。

### 18.3 兼容模式的探测诊断示例

在被测文件相同初始化顺序之后执行以下观察，并保存结果：

```python
import os
from torch.testing._internal import inductor_utils

print("GPU_TYPE:", inductor_utils.GPU_TYPE)
print("HAS_GPU:", bool(inductor_utils.HAS_GPU))
print("HAS_TRITON:", bool(inductor_utils.HAS_TRITON))
print("RUN_GPU:", bool(inductor_utils.RUN_GPU))
print("custom devices:", os.getenv("PYTORCH_TESTING_DEVICE_FOR_CUSTOM"))
print("excluded devices:", os.getenv("PYTORCH_TESTING_DEVICE_EXCEPT_FOR"))
```

目标版本可能没有某个字段；先查定义再调整诊断，不把不存在的字段加到框架。发现false时查原因，不直接改成True。

### 18.4 收集与执行命令示例

命令必须按目标PyTorch测试runner调整。支持pytest且包结构正确时，可参考：

```bash
python -m pytest --collect-only -q /workspace/test-inductor_npu/test_example.py
python -m pytest -vv /workspace/test-inductor_npu/test_example.py -k 'specific_test'
python -m pytest -vv /workspace/test-inductor_npu/test_example.py
```

使用上游 `run_tests` 入口的文件，保留其官方执行方式，例如：

```bash
python /workspace/test-inductor_npu/test_example.py -v
```

以上是占位路径，不是声明当前存在这些文件。参数化后的真实node ID应从收集结果取得；不要凭类名猜测“运行了某方法”。

完整执行前先选覆盖不同规则的代表用例：普通pointwise、reduction、dynamic shape、autograd、alias/in-place、dtype、Triton、Graph、子进程。每种专项规则至少有对应验证，未执行的规则保持“待验证”。

### 18.5 运行结果必须记录

记录命令、时间、目标文件/类/方法/node ID、PyTorch/torch_npu/编译后端版本、硬件型号与设备数、环境变量、seed、实际执行数、pass/fail/skip/xfail/error、日志路径。

进程返回0但没有执行任何测试、全部被skip、入口条件没有进入，不属于实机通过。把“不支持该特性”“环境缺依赖”“适配错误”“上游已知失败”分开记录。

## 19. 推荐结果表结构

建议至少分成“文件适配表”和“修改意图表”。一个文件可能找到了diff，但只迁移了部分规则；一个文件没有diff，也可能成功跑过实际NPU测试。

### 19.1 文件适配表

| 字段 | 含义 |
|---|---|
| source_file / output_file | 原始与输出相对路径 |
| source_version / source_commit | 原测试来源，未知就写unknown |
| diff_status | exact / reviewed_alias / missing |
| diff_file / diff_version | 参考路径与版本 |
| file_category | 普通、CPU逻辑、Triton、Graph、厂商专项、helper等 |
| adaptation_mode | transfer / explicit / unchanged |
| rule_ids | 实际适用的规则编号 |
| edit_status | preliminary / partial / unchanged / not_applicable / manual_required |
| intent_total / intent_resolved / intent_unresolved | 真实语义项统计，不用虚构hunk完成数 |
| syntax_status | pass / fail / not_run |
| import_status / collection_status | pass / fail / not_run，附收集数量 |
| runtime_status | pass / fail / skipped / environment_blocked / not_run |
| executed / passed / failed / skipped / xfailed | 实际执行统计；未知留空或unknown |
| unresolved_items | 仍需检查的具体位置与原因 |
| evidence / logs | 符号定位、diff位置、验证日志 |
| notes | 版本差异、覆盖限制、CPU保留策略 |

### 19.2 状态示例

| 文件 | diff | 修改 | 语法 | NPU运行 | 解释 |
|---|---|---|---|---|---|
| `test_example.py` | exact | preliminary | pass | not_run | 已按参考语义初步修改，等待实机 |
| `test_new_case.py` | missing | preliminary | pass | not_run | 没有diff，应用适用通用规则 |
| `helper.py` | missing | unchanged | pass | not_run | 纯工具模块，无需设备修改 |
| `test_vendor_feature.py` | missing | manual_required | pass | not_run | 厂商专项能力未确认 |
| `test_graph_case.py` | exact | partial | pass | fail | Graph设备已改，专项断言尚需处理 |

这些是状态表达示例，不是对当前这些占位文件的实际测试结果。

### 19.3 修改意图表

每一行包含：文件、参考hunk/自定义规则、类/方法、修改前性质、目标位置、修改后性质、状态、证据、待验证项。

使用独立列表示“找到diff”和“适配状态”。即使通用规则已处理，仍保留 `diff_status=missing`，以便用户追踪哪些文件没有参考资料。

## 20. 可直接交给模型的任务指令

下面这段可与本报告一起交给代码模型。调用方需要替换方括号内容；没有实际输入时，模型应明确缺少什么，而不是编造版本与文件。

```text
你要对PyTorch Inductor测试做NPU初步适配。请完整阅读随附的
《PyTorch Inductor测试用例NPU适配规则报告》，按其中规则执行。

输入：
- 原始测试目录：[SOURCE_DIR]
- 目标PyTorch版本/tag/commit：[TARGET_VERSION_OR_COMMIT]
- 匹配torch_npu版本/源码：[TORCH_NPU_VERSION_OR_DIR]
- Ascend 2.7.1参考diff目录：[REFERENCE_DIFF_DIR]
- 独立输出目录：[OUTPUT_DIR]
- NPU运行环境与后端：[ENVIRONMENT，若没有则明确只有静态验证]

目标：
1. 保留原测试目录，在输出目录保持相对路径与所需资源。
2. 找到同名diff后逐hunk理解语义，将其移植到目标版本对应类/方法。
3. 没有diff的文件也按适用通用规则分析并初步处理；纯工具/CPU文件
   可以保留原样，厂商专项文件标记不适用或待人工，不能强行假成功。
4. 保留原测试意图、CPU参考、关键输入、断言、编译路径及有效能力门禁。
5. 生成逐文件结果表、逐修改意图明细和未解决项清单。

开始前：
- 确认实际目录、源版本、匹配运行包，不凭目录名猜版本。
- 检查已有用户改动与输出，不删除已有成果。
- 阅读目标版本transfer_to_npu、NPU Inductor和测试框架相关定义。
- 明确全局补丁、JIT、Triton探测、CPU过滤的导入副作用与顺序。

逐文件算法：
A. 读取完整文件，按测试意图分类到类/方法层级。
B. 精确匹配diff。重命名/拼写错误只用经内容验证的显式别名映射。
C. 将每个hunk拆成独立修改意图，记录对应目标符号与状态。
D. 选择兼容迁移或显式设备模式；只添加必要初始化。
E. 按R01-R37检查导入、设备、框架、门禁、能力、编译、Graph、dtype、
   精度、RNG、布局、错误、子进程、资源与辅助模块。
F. 使用结构化定位修改。含future/docstring、多行装饰器、组合条件的
   文件必须保留语法与布尔逻辑。同名方法必须限定类作用域。
G. 无法唯一定位、没有等价接口、缺能力证据时保留未解决项；继续处理
   其他独立规则，不随便删第一个匹配项。
H. 审查修改diff，确认无不相关变化。重复执行不得重复加import。
I. 按V0-V5在可用环境中验证，明确未执行等级。

禁止：
- 直接把2.7.1 patch机械应用并声称适配完整。
- 对全目录做cuda→npu替换或对所有文件添加统一补丁头。
- 全删requires_gpu/cuda/triton、skip或所有入口条件。
- 伪造SM capability、HAS_TRITON/HAS_GPU恒真作为通用规则。
- 删除关键断言、关闭fullgraph、改成eager、扩大所有容差、统一改dtype。
- 吞掉异常、全部skip后记为通过。
- 猜测allow_npu参数、npugraph配置键或不存在的私有API。
- 一条import匹配就把整个hunk记成功，或强制hunks_applied=hunks_total。
- 将语法通过、导入存在、测试收集成功当作NPU运行通过。

交付：
- 输出目录中的初步适配文件，保留原文件名与相对路径。
- 文件表：diff匹配状态、类别、模式、规则、修改状态、语法/导入/收集/
  运行状态、执行数量、未解决项、日志。
- 修改意图表：每个diff意图/通用规则的定位、处理与证据。
- 未解决清单：版本接口、硬件能力、生成代码/错误预期、专项实现。
- 总结：真实修改数、missing diff数、无需修改数、待人工数、验证范围。

没有NPU实机时，完成合理的静态初步改写并明确not_run。
“初步适配”与“实机通过”必须分别报告，不制造已经运行的结果。
```

## 21. 交付前检查清单

### 输入与路径

- [ ] 目标版本来源有记录；参考diff和当前源码版本不混淆。
- [ ] 原目录未修改，输出目录不重叠，已有用户改动已保留。
- [ ] 文件名、相对路径、资源和包结构保持完整。
- [ ] 同名diff与reviewed_alias映射有证据，missing状态仍保留。

### 代码结构与设备

- [ ] future import、docstring、许可证与注释位置合法。
- [ ] 初始化在相关设备探测前完成，重复import已控制。
- [ ] helper/空`__init__.py`没有无理由的全局补丁。
- [ ] NPU Inductor与Triton探测按目标版本检查。
- [ ] JIT测试仍真正执行原要求的script行为。
- [ ] CPU/meta/fake/参考设备与多设备索引保持测试意图。
- [ ] 设备测试类实际生成，环境过滤没有意外减少覆盖。

### 语义与专项

- [ ] 每个装饰器定位到正确类/方法，多行和组合条件完整。
- [ ] 真正能力、依赖、平台、多卡和内存门禁被保留。
- [ ] 无全局伪造capability或恒真探测。
- [ ] backend、fullgraph、dynamic和编译行为断言保持。
- [ ] 配置键、模块名、counter名称有真实实现依据。
- [ ] Graph、stream、event和snapshot接口及断言经专项审查。
- [ ] dtype、精度、AMP、RNG、布局与alias语义未被无依据放宽。
- [ ] 生成代码/日志/错误预期无凭空替换。
- [ ] 子进程与生成repro有必要初始化，资源路径正确。

### 证据与统计

- [ ] 每个参考hunk拆分出的所有意图都真实计数。
- [ ] unresolved、obsolete、not_applicable与applied分开。
- [ ] 语法、导入、收集、设备、运行分别有状态。
- [ ] 全skip、零收集、未进入入口不被记为运行通过。
- [ ] 实机结论包含版本、硬件、命令、执行数量与日志。
- [ ] 旧CSV统计未被当作完整语义迁移证明。

## 22. 当前源码定位索引

下列链接指向本次核对的本地源码。它们便于查阅当前实现，不替代目标2.10版本的核对。

| 查阅内容 | 本地位置 |
|---|---|
| 迁移API白名单与设备关键字 | [transfer_to_npu.py:30](/D:/TEST213/pytorch_ascend/torch_npu/contrib/transfer_to_npu.py:30) |
| CUDA设备参数包装与转换 | [transfer_to_npu.py:177](/D:/TEST213/pytorch_ascend/torch_npu/contrib/transfer_to_npu.py:177) |
| JIT实际行为 | [transfer_to_npu.py:318](/D:/TEST213/pytorch_ascend/torch_npu/contrib/transfer_to_npu.py:318) |
| 迁移模块中的has_triton禁用 | [transfer_to_npu.py:346](/D:/TEST213/pytorch_ascend/torch_npu/contrib/transfer_to_npu.py:346) |
| 导入初始化 | [transfer_to_npu.py:468](/D:/TEST213/pytorch_ascend/torch_npu/contrib/transfer_to_npu.py:468) |
| Graph与编译探测补丁 | [transfer_to_npu.py:575](/D:/TEST213/pytorch_ascend/torch_npu/contrib/transfer_to_npu.py:575) |
| JIT恢复标志默认值 | [_dynamo.py:19](/D:/TEST213/pytorch_ascend/torch_npu/utils/_dynamo.py:19) |
| NPU感知Triton探测与修补 | [_dynamo.py:589](/D:/TEST213/pytorch_ascend/torch_npu/utils/_dynamo.py:589) |
| NPU Inductor内部探测修补 | [utils.py:30](/D:/TEST213/pytorch_ascend/torch_npu/_inductor/utils.py:30) |
| NPU编译设备后端注册 | [Inductor initializer:207](/D:/TEST213/pytorch_ascend/torch_npu/_inductor/__init__.py:207) |
| 当前NPU Inductor后端选择 | [Inductor initializer:380](/D:/TEST213/pytorch_ascend/torch_npu/_inductor/__init__.py:380) |
| 当前testing的框架导入 | [testing initializer:8](/D:/TEST213/pytorch_ascend/torch_npu/testing/__init__.py:8) |
| 当前testing的设备过滤副作用 | [testing initializer:204](/D:/TEST213/pytorch_ascend/torch_npu/testing/__init__.py:204) |
| 上游Inductor框架常量，当前为2.13源码 | [inductor_utils.py:69](/D:/TEST213/pytorch_213/torch/testing/_internal/inductor_utils.py:69) |
| 上游设备测试实例化签名，当前为2.13源码 | [common_device_type.py:1010](/D:/TEST213/pytorch_213/torch/testing/_internal/common_device_type.py:1010) |
| 当前改写头部示例 | [test_indexing.py:1](/D:/TEST213/test-inductor_npu/test_indexing.py:1) |
| 历史脚本强制完成计数的问题 | [port_inductor_npu.js:265](/D:/TEST213/port_inductor_npu.js:265) |
| 历史总表 | [inductor_npu_adaptation_report_all.csv](/D:/TEST213/inductor_npu_adaptation_report_all.csv) |

后续模型可以据此先完成可审查的改写，但每条规则都应结合被测文件与目标版本判断，保留未解决项与实际验证边界。

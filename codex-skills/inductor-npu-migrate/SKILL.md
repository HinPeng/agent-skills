---
name: inductor-npu-migrate
description: "完成 PyTorch test/inductor 社区用例的 Ascend NPU 初步适配、Linux NPU 测试、日志汇总及 Excel 结果回填。适用于指定正式 release tag 的全量或选定 Inductor 测试迁移；Windows 环境只做源码改写和静态链路验证，不执行真实 NPU 测试。"
---

# Inductor 社区用例 NPU 迁移

优先编写、审查并执行脚本，完成来源确定 → 原始用例清单 → 全量逐文件分析和初步改造 → 测试 runner → 日志表 → 原清单回填。所有结论区分静态改写与实机执行，不能把插入 import、全部 skip 或零项收集称为 NPU 通过。

## 输入和默认约定

- 每次任务确定 PyTorch 正式 tag（如 `v2.13.0`）、工作目录、测试范围（默认 `test/inductor` 递归全部 `test_*.py`）、匹配的 torch_npu 版本或源码。没有指定版本时先查询官方 tags 并询问目标版本，不把示例或本地文件夹名当成版本。
- 来源是 `https://github.com/pytorch/pytorch` 的 `refs/tags/<tag>`，记录实际 commit。未找到 tag 就停止下载阶段并报告，不能改用 main、rc 或近似版本。
- NPU 环境另需确认运行目录、依赖/包版本、CANN 环境初始化、后端及设备信息。已有授权/参数直接复用。当前只有 Windows 或用户要求不运行时，完成 V0/V1 和模拟链路，后续标记 `NOT_RUN`。
- torch_npu 从 [GitCode Ascend/pytorch](https://gitcode.com/Ascend/pytorch.git) 查找与社区用例版本对齐的分支，按该分支实际文档编译 wheel 并安装；记录 commit、wheel hash 和安装版本。环境配置参考用户提供的 [torch-dev-env](https://github.com/HinPeng/agent-skills/tree/main/codex-skills/torch-dev-env)。准备实机环境时读取 [references/npu-environment.md](references/npu-environment.md)，不以其 current-only 公开 wheel 流程代替本任务的源码构建。
- 输出清单默认 `待测用例.xlsx`，明细默认 `pytest_results.xlsx`。用户指定 `pytest_result.xlsx` 时使用其路径，避免同时生成两份不同来源的结果表。
- Python 3.10+（源码归档需要 3.12+，或 3.10/3.11 带 `tarfile.data_filter` 安全补丁）、Git、`openpyxl>=3.1`；运行阶段另需 Linux Bash、pytest、版本匹配的 torch/torch_npu 与上游测试依赖。优先已有依赖，不安装 torch 来验证 Windows 链路。

脚本均在本 skill 的 `scripts/` 中；命令里的 `S` 指向该目录、`RUN` 是新的任务输出目录。先阅读 [references/workflow.md](references/workflow.md) 的具体命令与适配计划格式。

## 来源与清单

1. 用 `git ls-remote --tags --refs https://github.com/pytorch/pytorch.git 'v*'` 查询实际存在的 tag。已选定且本地官方仓库有精确 tag 时可直接验证。读取当地 `AGENTS.md` 等适用指令。
2. 运行 `prepare_source.py --tag TAG --output RUN`；可用 `--source-repo LOCAL_REPO` 复用已有仓库。脚本通过 `git archive` 读取 tag 对应提交，忽略当前 checkout 的未提交改动，并导出完整源树到 `RUN/adapted`、保存原始 Inductor 副本到 `RUN/original/test/inductor`。保留辅助模块及非 Python 资源，不覆盖输入和既有任务。
3. 脚本自动调用 `extract_testcases_name.py` 在原始目录生成 `待测用例.xlsx`。仅扫描 `test_*.py`；静态提取顶层函数及类方法，保留相对路径、类名和同名定义。动态生成、继承和参数化实例不凭空展开，它们在实测表中按前缀聚合；需要完整收集清单时另在 NPU 环境做 collect-only。清单必须在改造前生成，之后不以改造输出重建基准。

## 全量初步适配

完整阅读 [references/NPU_TEST_ADAPTATION_RULES.md](references/NPU_TEST_ADAPTATION_RULES.md)，以其中 R01–R37 为改造依据。文档的 2.7.1/2.10/2.13 历史描述、本地绝对链接和已处理数量属于历史上下文；本次以用户 tag、匹配实现和实际证据为准。用户提供更新规则时优先使用并记录其路径/hash。

- 逐文件读取测试意图，进一步审查类、方法、装饰器、门禁、设备实例化和子进程。CPU/纯逻辑、普通加速器、Triton、Graph/流/内存、厂商专项和 helper 分别判断。纯 CPU/helper 可保持原样，但必须有原因。找不到旧 diff 也继续分析；不得直接运行规则指出不可信的旧 `port_inductor_npu.js`。
- 查询目标版本 torch_npu 的 transfer 白名单、Inductor 注册、testing 副作用与私有 API。初始化须在设备探测/框架导入前、docstring/future 后；按需要恢复 JIT/Triton 探测，不能统一加入有 CPU 过滤副作用的 `torch_npu.testing`。
- 保留原编译后端、关键断言、fullgraph、dtype、精度、CPU/meta/fake 参考和设备索引。禁止全局 cuda→npu、删全部 skip、伪造硬件能力、改 eager、吞异常或凭空修改代码生成预期。
- 由代理根据证据生成 `adaptation_plan.json`，将精确上下文改动、规则编号、目标类/方法、证据和未解决意图写进去。优先用 AST/CST 定位，再生成唯一匹配的精确编辑；批量运行 `apply_adaptation.py`。脚本要求每个 test 文件都有分类，校验源 hash、唯一匹配、用例名/类不丢失、静态 compile，生成逐文件及逐意图 JSON 报告，并可用同计划/报告重复执行而不重复加头。
- 脚本不会自动推断全部 NPU 语义。代理必须审查 diff 与报告，迭代解决可处理项，分别保留 `preliminary/unchanged/not_applicable/manual_required`。缺版本接口/硬件证据时保留具体未解决项，继续独立工作；不强制统计全成功。文件级 `preliminary` 只表示初步修改，实际每条 intent 仍有自己的状态。
- 检查资源完整性、docstring/future/import 顺序、所有 Python 文件 AST + `compile()`，保留适配前后 diff。无实机时只报告 V0/V1，不宣称运行链路已经在 NPU 上证实。

## Runner 与实机测试

运行 `generate_runner.py --test-root RUN/adapted/test/inductor --output RUN/run_pytest.sh`。全量时 `TEST_FILES` 包含该目录下全部 `test_*.py` 的相对文件名（含无静态 def 的动态测试文件）；选跑用 `--files` 精确列出路径。runner 不依靠递归 basename 查找，不静默排除专项/未解决文件。生成文件列表另存 JSON。

Windows 用 `bash run_pytest.sh TEST_ROOT --dry-run` 检查列表和命令，可用 `self_test.py` 在临时目录验证模拟链路。没有 Bash 时仍验证脚本生成/静态脚本内容并明确 Bash 未检查；不因此启动真实测试。dry-run 不产生伪 pytest 日志。

真实运行只在已指定 Linux NPU 环境进行。按用户环境 source CANN 等初始化，在 `adapted/test/inductor` 原目录结构下执行 `npu_preflight.py --expected-torch VERSION --smoke` 和 runner，尽量保留文件相对依赖；预检查核对 torch/torch_npu 的基础 release 与目标一致（允许 +cpu、post 等构建后缀），记录实际安装包位置/版本、设备、后端、过滤变量和诊断结果。不要从含 `torch/` 的仓库根目录误加载未构建源码，验证 `torch.__file__` 与上游 test/helper 导入来自预期路径。保留完整源码与资源；框架若要求源码构建/安装按实际环境处理。

先 collect-only 和代表性用例，再按用户范围执行 `bash run_pytest.sh TEST_ROOT`。每个文件一个独立子进程，日志在当前工作目录的 `./logs2`。日志记录精确文件、命令、时间和 exit code；单文件失败仍继续下一文件，最终返回非零。现有日志不混入新轮次；使用新的工作目录或先显式归档。pytest 不支持的官方入口文件单独保留原 runner 并记录结果，不能当作 pytest 已通过。

## 汇总和回填

1. 无论 runner 是否返回非零，都继续执行 `summarize_pytest_logs.py --input LOGS --output RESULTS --format xlsx`，强制 XLSX，避免缺依赖时悄悄变 CSV。表含文件名、用例名、原始状态、报错信息及 node_id。
2. 执行 `fill_test_results.py --inventory 待测用例.xlsx --results RESULTS`，只更新或追加 **执行结果、失败原因** 两列；其他单元格保持。原地回填自动保存首次备份，不新增其他列/页。结果表路径始终显式传入。
3. 同文件下实测方法名以待测用例名开头即匹配，包括参数/device/dtype 后缀；同名不同文件不混配。同一方法跨生成类/参数实例聚合，原表中的类名用于溯源，不猜 generic 类到生成类的映射。已有同文件不同类同名方法也按用户方法名前缀规则聚合；若用户要区分类，再取得明确类映射。
4. 有任意 `FAILED` → `FAILED`；无 FAILED 但有 `SKIPPED` → `SKIPPED`；所有匹配实例 `PASSED` → `PASSED`。否则保留 `ERROR/NOT_RUN/XFAIL/XPASS/UNKNOWN`，不默认为通过。无匹配默认 `NOT_RUN`（可选 `--unmatched blank`），原因带文件级收集/环境错误（若有）。不完整日志中全 PASSED 记录也不能声称完整通过。
5. FAILED 合并去重失败原因，PASSED 清空旧失败原因，SKIPPED/特殊状态保留可取得原因；源日志不足时明确缺失。混合特殊状态仍按用户三态优先顺序处理，原始特殊状态留在明细表供追踪。

交付 skill 使用时生成的来源 manifest、原始/改造目录、修改计划和报告、runner/文件列表、实际日志及环境证据（若运行）、实测结果表、已回填清单和未解决项。说明真实验证等级和未运行范围。当前创建 skill 的验收只需要静态与模拟验证，不能把模拟日志交付成社区用例实测结果。

# 可运行命令与适配计划

所有脚本以 Python 3.10+ 运行，prepare_source 使用安全 tar 提取需要 Python 3.12+ 或带 data_filter 的已修补 3.10/3.11。使用调用环境自己的解释器；Windows Codex 可通过 workspace dependency loader 找到 Python。Excel 脚本使用用户原脚本已有的 openpyxl，不依赖桌面 Excel。

## Windows 准备示例

PowerShell：`$S` 指向本 skill 的 scripts，`$P` 是可用 Python 路径，`$R` 是新的输出目录。下面的 v2.13.0 只是示例，先确认用户目标。

```powershell
$S = 'C:/Users/admin/.codex/skills/inductor-npu-migrate/scripts'
$P = 'python'
$R = 'D:/work/inductor-v2.13.0-run1'
git ls-remote --tags --refs https://github.com/pytorch/pytorch.git 'v2.13.0'
& $P "$S/prepare_source.py" --tag v2.13.0 --output $R
# 已有官方仓库时追加 --source-repo PATH；读取 tag，不读取当前改动。
```

结构：`original/test/inductor` 是不可改造的基准；`adapted` 是完整 tag 源码树，保留 test helper、conftest、配置、资源；其 `test/inductor` 是改造目标。manifest 保存 tag/commit、规则 hash 和原始 Inductor 每个文件 hash。路径参数不会固化到 skill，本 skill 不读取历史目录作为新任务输入。

若本地 tag 缺失，显式拉取对应 tag 后重试，不能切换用户 checkout 或改用 HEAD。仓库所有权问题只对已确认可信的路径使用一次性的 `git -c safe.directory=ABS_PATH`，不要改全局信任范围；脚本内部 git 需要在具有合法所有权的环境运行。

单独提取已有目录：

```powershell
& $P "$S/extract_testcases_name.py" --input "$R/original/test/inductor" --output "$R/待测用例.xlsx"
```

此命令仅在尚未生成清单时运行。列为 `file_name,test_case,class_name,执行结果,失败原因`，初始结果为空。可输出 CSV，但标准链路使用 XLSX。静态清单不是运行收集结果：动态 setattr/继承用例只有实机收集才可展开。

## 适配计划协议

先完整阅读规则，逐文件分析，使用 Python 脚本提取 hash、AST 类/方法和精确上下文，再生成 JSON，而不是手工抄行号。以下展示一条已判断需要显式设备初始化的编辑；真实计划必须包括所有 `test_*.py`，可另含需要修改的 `.py` helper。

```json
{
  "files": [
    {
      "path": "test_example.py",
      "sha256": "实际原文件字节的 SHA256",
      "category": "general_accelerator",
      "mode": "explicit",
      "status": "preliminary",
      "reason": "原测试验证单算子编译结果，已查目标版本对应 NPU API",
      "initialization": "import torch\nimport torch_npu\nimport torch_npu._inductor  # noqa: F401",
      "operations": [
        {
          "old": "        x = torch.randn(8, device=\"cuda\")",
          "new": "        x = torch.randn(8, device=\"npu\")"
        }
      ],
      "intents": [
        {
          "symbol": "TestExample.test_compute",
          "rule_ids": ["R01", "R02", "R06", "R19"],
          "before": "CUDA 创建输入，Inductor 结果与参考比较",
          "after": "NPU 创建输入，保留编译路径和数值断言",
          "status": "applied_semantic",
          "evidence": "填写实际源码路径/符号/版本，不使用此示例当证据",
          "unresolved": "待 NPU 收集和执行"
        }
      ]
    }
  ]
}
```

`initialization` 可省略。脚本将其放在 docstring/future 后、现有其他代码前，保持 shebang/编码/前导注释；它不替你判断导入副作用。原文件已有相关 import 时通过唯一上下文替换复用/移动，避免重复；需要在特定现有 import 之后初始化时，把完整初始化作为精确 `operations` 插入，不再填写 initialization。旧上下文必须唯一，不提供全局替换选项。必须在单次计划中说明所有语义，不把机械编辑次数当完成意图数。

每文件 `status`：`preliminary`、`unchanged`、`not_applicable`、`manual_required`。后三者不得有 edits，必须提供 reason/intents；部分初步修改但有待解决项用 preliminary，intent 明确 unresolved。intent 状态用 `applied_exact/applied_semantic/already_satisfied/obsolete_in_target/not_applicable/unresolved`；只填写有证据的规则编号，计数以真实意图为准。脚本拒绝没有实际 symbol/status/evidence 的意图。

```powershell
& $P "$S/apply_adaptation.py" --root "$R/adapted/test/inductor" --plan "$R/adaptation_plan.json" --report "$R/adaptation_report.json"
& $P "$S/generate_runner.py" --test-root "$R/adapted/test/inductor" --output "$R/run_pytest.sh"
# 选跑时追加 --files test_a.py nested/test_b.py，不要填模糊 basename。
```

同一计划重复执行必须使用同一 report；输出 hash 匹配才跳过已应用编辑。新一轮计划应基于当前文件的实际 hash，保留前一轮报告，不能覆盖未审查改动。应用前全部文件预检查，任何唯一匹配、hash、名称或语法错误都会在写源码前失败；文件系统写入失败仍需检查实际落盘文件与 hash。

## Linux NPU 运行

传输 skill scripts、完整 adapted 树、计划/报告、manifest、清单与生成 runner 到目标环境。路径按目标机调整。安装版本必须匹配，用户已有的 wheel/源码和初始化命令优先；不自动升级 torch/torch_npu。

```bash
S=/path/to/inductor-npu-migrate/scripts
RUN=/path/to/run
# 先按该服务器实际要求 source 环境；不猜 CANN 安装位置。
cd "$RUN/adapted/test/inductor"
python3 "$S/npu_preflight.py" --expected-torch 2.13.0 --smoke --output "$RUN/runtime_environment.json"
# 上一步返回非零则先处理环境，避免全量测试污染结果。
python3 -m pytest --collect-only -q test_example.py
# test_example.py 替换为真实、已初步适配的代表文件。
bash "$RUN/run_pytest.sh" "$RUN/adapted/test/inductor" --dry-run
bash "$RUN/run_pytest.sh" "$RUN/adapted/test/inductor"
runner_rc=$?
# 即使 runner_rc 非零仍要提取日志；logs2 位于这里的当前工作目录。
python3 "$S/summarize_pytest_logs.py" --input ./logs2 --output "$RUN/pytest_results.xlsx" --format xlsx
python3 "$S/fill_test_results.py" --inventory "$RUN/待测用例.xlsx" --results "$RUN/pytest_results.xlsx"
```

runner 支持 `PYTHON_BIN` 选择同环境解释器，以及 `--` 后传 pytest 参数，例如 `-- -k test_compute`；测试范围筛选须报告，不能称全量执行。pytest 插件/xdist/输出格式需兼容 verbose 文本；当前链路串行逐文件，避免并发输出交叉。无 collect 标记/exit 标记、进程异常退出、超时和零收集保留为文件问题，不伪造单用例 FAILED。所有日志保留供追溯。

`npu_preflight` 的显式设备 smoke 不代表兼容补丁下所有探测和 JIT 状态正常；采用 transfer 模式的文件还需按其初始化顺序验证框架常量/类收集。

## 回填的精确规则

匹配键：相对文件路径 + 实测最后一段方法名 `startswith(待测 test_case)`；类名原样保留，仅用于溯源。故 `test_add` 按用户前缀规则也匹配 `test_addmm`，不能擅自改成仅 `_` 或 `[` 分隔匹配。跨文件永不按方法名混配，basename 不唯一直接报错。不同类同名定义按这一规则取得相同聚合值。

三态之外：无 FAILED 但有 SKIPPED 仍是 SKIPPED（即使同时 ERROR）；否则 ERROR 优先于 NOT_RUN/XFAIL/XPASS，只有所有匹配实例 PASSED 才是 PASSED。ERROR 不静默变 FAILED，XFAIL 不变 SKIPPED，XPASS 不变 PASSED。若用户要求其他映射，应显式记录后再变更。

原地写入自动保存 `待测用例.before_fill.xlsx` 首次基准备份；只有两列修改，成功 PASS 清空原失败原因。`--output NEW.xlsx` 可以另存不覆盖原表；`--unmatched blank` 可让未匹配结果留空并保留原因。结果表名强制通过 `--results` 指定，接受 singular/plural 两种文件名。

## 离线验收

```powershell
& $P "$S/self_test.py"
```

使用临时 Git 仓库、模拟测试源码和手写 pytest 日志，验证 tag 导出、资源、原始清单、精确适配、重复执行、runner 列表/可用时 Bash 语法及 dry-run、特殊状态/收集错误/中断日志与回填不改其他单元格。它不导入 torch、不运行真实 pytest、不访问网络、不验证 NPU。结果写在临时目录，默认结束自动清理；用 `--output-dir NEW_DIR` 保留模拟产物，目录名应明确 fixture，不能交付为实测表。

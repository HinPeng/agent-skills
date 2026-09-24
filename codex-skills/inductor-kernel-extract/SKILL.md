---
name: inductor-kernel-extract
description: 从 TorchNPU/Inductor 日志、生成代码或 IR 抽取并验证独立 kernel 复现。用于编译、运行、精度问题的精简复现交付。
---

# Inductor 单 Kernel 抽取

交付可独立运行的原 kernel/IR、必要输入与检查、最终入口的真实结果日志和运行命令。文件数量、命名及数据存放按实际问题和用户要求选择，说明通常放文件头。

## 边界与完成条件

- 用户当前指令优先。抽取包含创建复现脚本与验证，不自动扩展为生产修复、重建环境、修改外部包或对外上传；本地材料由用户交接。
- 保留原算法、编译元数据、输入布局/别名、调用配置、必要同步及比较标准。先复现再精简，不靠修改算法、容差或替换编译器制造 PASS。
- 交付包只含复现所需内容，不依赖包外测试仓、调查目录或原始缓存。不默认添加来源字典、环境快照、哈希清单或通用调试框架；用户要求或具体完整性问题需要时才增加相关校验。
- 完成标准是最终独立入口已在目标环境验证，并说明是否复现同一失败。静态抽取、语法通过、导入失败都不能当原问题复现；未复现时检查数据、配置及前驱依赖。设备不可用时明确未运行及缺口。

## 工作要点

从用户指定的日志、日构建、output code 或 kernel/IR 定位原始失败及第一条实质错误；`NoValidChoicesError` 或异步异常不一定是根因。核实实际解释器、模块 `__file__` 和编译器入口，不套用历史版本或卡号。

先在个人持久化任务目录保存相关原件和来源，再恢复目标编译块、初始化、真实输入及调用。以新进程执行最终入口，保留真实退出码；数值错误需实际输入、参考值与误差，编译错误无需构造无关 golden。

远程运行按项目当前环境与资源规则，自主选择实时空闲卡，默认单卡串行；仅使用本人任务资源。证据按项目约定留在 `project-notes/archive/` 或远程个人持久化目录，不以 `/tmp` 为唯一副本。需要留档的结果和索引按项目约定维护。

## 按需参考与工具

| 当前需要 | 读取或使用 |
|---|---|
| 从日构建或 wiki 找来源、核对工具条件 | [wiki-and-environment.md](references/wiki-and-environment.md) |
| 恢复编译块、输入、布局与调用依赖 | [extraction.md](references/extraction.md) |
| 核验编译、运行、精度或固定 IR 复现 | [validation.md](references/validation.md) |
| 多文件、多用例或对照模式的交付组织 | [package-layout.md](references/package-layout.md) |

可用 [inspect_output_code.py](scripts/inspect_output_code.py) 静态列出绑定和调用点，再以精确 `binding` 或 `compiled_name` 导出：

```bash
python <技能目录>/scripts/inspect_output_code.py /path/to/output_code.py
python <技能目录>/scripts/inspect_output_code.py /path/to/output_code.py --kernel <精确名称> --out /path/to/new-evidence
```

`--out` 必须不存在。脚本只用标准库，不导入 Torch 或执行 output code，导出的是证据片段；它不恢复输入。非字面量源码、动态/间接调用、同名歧义或不支持的格式需手工分析，不能把未匹配当作没有 kernel。实际执行使用完整脚本路径。

最终回复给交付链接、一条运行命令、实际复现状态及必要环境边界。原始长日志和试跑材料内部保留，不默认整包交付。

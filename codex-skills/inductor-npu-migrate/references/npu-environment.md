# NPU 环境与源码编译

用户指定 torch_npu 来源：[Ascend/pytorch (GitCode)](https://gitcode.com/Ascend/pytorch.git)。必须与本次社区用例 tag 对齐，并记录实际源码 branch/commit、安装 torch/torch_npu 版本及包来源。不能用仓库目录名或相同主版本号推断完全兼容；patch/post 版本和私有测试 API 需要核对。

环境组织参考：[HinPeng/agent-skills torch-dev-env](https://github.com/HinPeng/agent-skills/tree/main/codex-skills/torch-dev-env)，其 [SKILL.md](https://github.com/HinPeng/agent-skills/blob/main/codex-skills/torch-dev-env/SKILL.md) 在本 skill 创建时已读取。它提供本地/SSH uv 环境创建、CPU PyTorch + 当前匹配 TorchNPU wheel、verify 和依赖快照。它的公开 wheel 是 current-only，并不提供历史 wheel 日期选择或源码构建入口；本任务需要对应分支编译时不能把该 wheel 下载流程当成源码构建。

## 解析并锁定版本

1. 用 `git ls-remote --heads https://gitcode.com/Ascend/pytorch.git` 查真实分支。结合对应分支 README、version.txt、setup.py、依赖文件以及当前源码支持矩阵，找与目标 torch tag 对齐的分支；不猜分支名（例如不能直接假定一定存在 v2.13.0）。存在多个兼容候选时说明区别并向用户确认所需维护线。
2. 使用新的构建 checkout，记录远端、branch、commit。源码构建环境优先重用已匹配的环境；Python ABI、架构、驱动、CANN、编译器及 torch ABI 均以该分支文档为准。社区 tests 所用私有 API 与实际 torch 包也要一致。
3. 查询/复用用户实际 CANN、驱动、设备可见变量和 source 命令，不能在 Windows 上安装或编译 NPU 包。远程 host、用户名、容器、挂载、env-root 没有明确输入时先询问；仅创建 skill 时将其作为后续调用参数。

## 构建与安装

在对应分支中读取完整构建文档和脚本参数。常见入口是 `ci/build.sh` 或 `python setup.py build bdist_wheel`，但**只有当前分支已证实支持时才使用**；不同分支可接受的 `--python`、`--torch` 以及禁用模块参数不同，不能照搬其他版本。

优先先编写一份本次环境专用的 shell 构建脚本，明确解释器、源码 commit、依赖安装、环境初始化、构建命令、日志和 exit code；`set -e`/pipefail 确保构建失败就停。构建成功再枚举 `dist/` 中与版本、Python ABI、架构匹配的 wheel，唯一确定后计算 SHA256，并用运行测试所用解释器安装该文件。不要宽泛安装 `dist/*.whl`，不要擅自升级无关依赖。

使用 torch-dev-env 时优先已有环境的 `verify` 和 `snapshot-dependencies`。其 `create` 会安装当前公开 TorchNPU，若随后要求源码覆盖必须明确顺序，且最终取证须指向本次源码 wheel。不要在源码安装后再执行 `install-latest-torch-npu` 把本次构建覆盖掉。

该参考 skill 的命令示例（只有已经获取该 skill 且调用方路径/host 已确定时运行）：

```bash
"$TORCH_DEV_ENV/scripts/manage-torch-env.sh" --host "$SSH_HOST" --env-root "$ENV_ROOT" verify "$SELECTOR"
"$TORCH_DEV_ENV/scripts/manage-torch-env.sh" --host "$SSH_HOST" --env-root "$ENV_ROOT" snapshot-dependencies "$SNAPSHOT_DIR"
```

selector 在当前参考版本为 `2_7/2_10/2_13`，不可用这些有限值限制未来版本。`create` 的非交互 uv 配置策略须明确选择；默认 `--no-sync-uv-config` 可避免额外改动。已有会话授权优先，不因参考 skill 的一般提醒重复请求已授权的构建/安装。

## 在原结构执行

迁移结果保留完整 PyTorch 源树，在 `adapted/test/inductor` 原相对位置改造并从该目录执行 runner，尽量保留相对路径依赖。原始 tag 副本仍保持不变。不能仅把所有 test 文件平铺到一个独立目录。

完整源码根目录含 `torch/`；从错误 cwd 启动会遮蔽已安装包。先在原测试目录验证 `torch.__file__`、torch_npu 路径/版本和 test helper 导入。如某文件确需特殊 cwd，保留单独执行方式和证据，不全局改 import 来掩盖问题。

通过 `npu_preflight.py` 保存基础版本和 NPU 诊断；通过依赖 snapshot/构建 manifest 另保存 driver/CANN/编译器版本、源码分支/commit、wheel 路径/hash、安装记录及构建日志。预检查通过不替代目标文件的收集和运行。

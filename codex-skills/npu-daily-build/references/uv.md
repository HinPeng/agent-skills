# 用 uv 管理隔离环境

## 新建与接管

管理层默认使用 uv。先解析可用 uv 的绝对路径并验证 --version，优先用户指定路径、项目既有适配器及个人工具目录；不覆盖全局镜像配置。选定解释器后所有操作显式传 --python，不能依赖当前 shell 恰好激活了哪个环境。

新建候选使用已确认的基础 Python，不自动下载另一套解释器；先确认目标目录不存在。下面变量均由本轮目标填入，不能用占位路径直接运行：

```bash
# UV、BASE_PYTHON、CANDIDATE_ENV 均为已解析的绝对路径。
if [ -e "${CANDIDATE_ENV}" ] || [ -L "${CANDIDATE_ENV}" ]; then
    printf 'Candidate path already exists: %s\n' "${CANDIDATE_ENV}" >&2
    exit 1
fi
"${UV}" venv --python "${BASE_PYTHON}" --no-python-downloads "${CANDIDATE_ENV}"
PYTHON="${CANDIDATE_ENV}/bin/python"
"${UV}" pip install --python "${PYTHON}" pip
```

pip 模块保留在候选中是为了上游构建脚本内部调用或 pip download；日常管理继续使用 uv。按目标版本安装 setuptools、wheel 等构建依赖，不无条件升级全部包。

接管现有标准 venv 时直接使用其 bin/python 执行 uv pip 命令，先记录版本、导入路径和有效个人补丁；不重跑 uv venv、不删除目录、不清理包。单纯切换管理工具不需要重装 torch/TorchNPU、TA、CANN 或基础 Python。

## 常用命令

```bash
"${UV}" pip list --python "${PYTHON}"
"${UV}" pip freeze --python "${PYTHON}"
"${UV}" pip check --python "${PYTHON}"
# 配套依赖已安装并核实后，安装明确的本地 wheel。
"${UV}" pip install --python "${PYTHON}" --no-deps "${WHEEL_PATH}"
# 仅当需要替换已安装的同版本构建时重装该 wheel。
"${UV}" pip install --python "${PYTHON}" --reinstall --no-deps "${WHEEL_PATH}"
```

把 torch 系列及其他核心版本写入本轮约束或显式安装计划，检查解析结果，避免安装构建依赖时被动升级 torch/Triton。--no-deps 只适用于依赖已另行处理的本地产物，不用于忽略真实冲突。不要用 uv sync/pip sync 清理用户原有环境。

需要通用 Python wheel 构建且项目后端支持时使用：

```bash
# 先在此构建 venv 安装并核实构建依赖；在所选源码目录运行。
"${UV}" build --wheel --python "${PYTHON}" --no-python-downloads \
    --no-build-isolation --out-dir "${WHEEL_DIR}" "${SOURCE_DIR}"
```

--no-build-isolation 要求所需构建依赖已就绪，防止隔离构建解析另一套 torch；依赖缺失时补齐指定依赖，不静默改用另一解释器。已有官方 ci/build.sh、CMake/Ninja 或 TA 构建入口仍按该版本调用，不把 uv build 当作所有原生构建的替代。

## 下载与兼容边界

- 缓存使用个人持久化目录，例如本轮指定 UV_CACHE_DIR；超时/重试使用已安装 uv 版本支持的 UV_HTTP_TIMEOUT、UV_HTTP_RETRIES，不机械透传 pip 的 --timeout/--retries。保留完整错误与退出码。
- uv 不可用时先在个人工具环境安装或修复，不改基础 Python。确实无法使用时可对本轮候选采用明确记录的 python -m pip 兼容方案；已经迁移的共享 runner 不因此退回 pip。uv 的依赖解析或构建失败不等于 uv 不可用，不自动换管理器掩盖首错。
- 获取 wheel 可使用经过核实的直接下载链接，或在专用下载 venv 中执行 python -m pip download；这是下载兼容入口，不是回退安装管理器。仅下载不引导安装 uv/pip；工具缺失时报告缺口或使用已有下载工具。
- 企业证书问题优先使用受信任系统证书/组织 CA，按 uv 版本帮助选择系统证书设置；不复制脚本中的 -k、--no-check-certificate 或 --trusted-host 作为默认解决办法。

命令语义参考 [uv 环境管理](https://docs.astral.sh/uv/pip/environments/) 与 [uv CLI](https://docs.astral.sh/uv/reference/cli/)，实际参数以执行环境 uv --help 为准。

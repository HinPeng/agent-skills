# Reusable Workflow Reference

This reference contains command shapes, not fixed values. Resolve all placeholders from the current server, log, repository, or user request.

## Variables to resolve

| Variable | Meaning |
| --- | --- |
| `<host>` | SSH target or current host; keep out of commands executed inside Docker |
| `<test_root>` | Run-specific directory containing model data, scripts, and logs |
| `<model_path>` | Model repository or entrypoint path visible inside the container |
| `<model_name>` | Runner/model identifier expected by the test script |
| `<image>` | Compatible Ascend base image and tag |
| `<container>` | Unique test-container name |
| `<device_ids>` | NPU IDs assigned to this run, e.g. `0` or `0,1` |
| `<mounts>` | Required host:container mounts, reviewed for sensitivity |
| `<install_script>` | Project environment installer, if any |
| `<packages_dir>` | Directory containing approved dependency packages |
| `<runner>` | Model test script, commonly a project `common_run_cmd.sh` |
| `<runner_args>` | Model-specific arguments in the runner's documented order |
| `<build_container>` | Existing compatible build container, if used |
| `<source_repo>` / `<branch>` | torch/torch_npu source and exact branch or commit |
| `<wheel>` | Built wheel selected after compatibility checks |

## Discover commands from logs

Search logs for complete commands and surrounding context. Examples:

```bash
rg -n -C 2 'docker (run|exec|start)|install\.sh|build\.sh|common_run_cmd|torch_npu.*\.whl' <log-file>
```

Record whether each line is a host command or an in-container command. Log wrappers often prepend timestamps, log levels, or quoting that must be removed before execution. Confirm paths exist after mounts are applied.

## Test container

Inspect before mutating:

```bash
docker image inspect <image>
docker ps -a --filter "name=^<container>$"
docker run --help | head
```

A typical launch is assembled from the source workflow, for example:

```bash
docker run -itd --name <container> \
  --net=host --shm-size=<shm_size> \
  --ulimit memlock=<soft>:<hard> --privileged \
  -e ASCEND_RT_VISIBLE_DEVICES=<device_ids> \
  <mounts> <image> /bin/bash
docker exec -i <container> /bin/bash -lc \
  'source /etc/profile; source /root/.bashrc; exec bash'
```

Do not add `--privileged`, host networking, SSH mounts, or driver mounts merely because an example used them. They are appropriate only when required by the image/runtime and authorized for the host.

## Environment installation and verification

Use the project's installer, preserving its argument order. A common shape is:

```bash
docker exec -i <container> /bin/bash -lc '
  set -e
  source /etc/profile
  source /root/.bashrc
  export PACKAGES_DIR=<packages_dir>
  <installer-env> bash <install_script> <package_input_dir>
  python -V
  python -m pip show torch torch-npu torch_npu 2>/dev/null || true
  python -c "import torch; import torch_npu; print(torch.__version__)"
'
```

Package names differ by distribution (`torch-npu` vs `torch_npu`). Use the names recognized by the target package index and verify the actual import module.

## Optional source build and overlay

Run build commands in a separate compatible container when the test container is busy or immutable:

```bash
docker ps -a --filter 'name=<build_container>'
docker start <build_container>
docker exec -it <build_container> /bin/bash

cd <source_workspace>
git clone --branch <branch> --depth 1 <source_repo> <source_dir>
cd <source_dir>
# Configure the project's proxy/toolchain only when required.
bash ci/build.sh --python=<python_major.minor>
find dist -maxdepth 1 -type f -name '*.whl' -print
```

Before choosing `<wheel>`, check its Python tag (`cp311` etc.), platform/architecture tag, torch compatibility, torch_npu version, and build commit. Compute a checksum if artifacts are transferred:

```bash
sha256sum <wheel>
docker exec -i <container> /bin/bash -lc 'python -m pip install --force-reinstall <wheel> && python -c "import torch_npu; print(torch_npu.__file__)"'
```

Use `--force-reinstall` only when the intent is explicitly to overlay the existing package. Otherwise uninstall/install according to the project's supported procedure. Re-verify versions and import after installation.

## Model runs and comparison matrix

Invoke the project's runner with its documented positional arguments. Keep mode and feature flags explicit. A common shape is:

```bash
docker exec -i <container> /bin/bash -lc '
  set -o pipefail
  source /root/.bashrc
  export REPORT_ID=<report_id>       # optional
  export SAVE_PROFILING=TRUE         # optional
  <runner> <model_path> <model_name> <device_arg> <mode> <flag> <batch_or_length> 2>&1 |
    tee <log_dir>/<mode>-<precision_state>.log
'
```

Run only the modes requested, typically:

- performance/functional pass with `mode=eager`;
- performance/functional pass with `mode=inductor`;
- precision pass with `CHECK_PRECISION=TRUE`, independently for each requested mode.

The runner's positional parameters are project-specific. Confirm their meaning from `--help`, the script, or a known-good log before substituting values. Do not assume `True`, `128`, or any other example argument has the same meaning for another model.

## Failure triage

- Docker/image/mount failure: inspect `docker inspect`, mount paths, permissions, and container logs.
- Device/runtime failure: verify driver/CANN visibility, `ASCEND_RT_VISIBLE_DEVICES`, free NPU memory, and single-process ownership.
- Import/package failure: compare Python, PyTorch, torch_npu, ABI, architecture, and wheel tags in the same container.
- Build failure: retain compiler output; check branch/toolchain/CANN compatibility and available disk/RAM before retrying.
- Model/data failure: validate model path, checkpoints, tokenizer/data files, and runner arguments.
- Precision mismatch: preserve both reference and NPU outputs and report tolerance/configuration; do not label it as a performance failure.

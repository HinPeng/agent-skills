# torch_npu Compatibility Matrix

Use this checklist before building or overlaying a wheel. Every row should be known or experimentally verified; a matching package filename alone is insufficient.

| Dimension | Test container | Build container/source | Required check |
| --- | --- | --- | --- |
| Python | `python -V` and interpreter path | build interpreter and `--python` target | Wheel `cpXY` tag matches the test interpreter |
| PyTorch | `python -c "import torch; print(torch.__version__)"` | installed torch and source build config | Supported torch/torch_npu pairing for the selected branch |
| torch_npu | installed version/commit, if any | source branch/commit and build metadata | Intended replacement and ABI/API compatibility |
| CANN/Ascend runtime | toolkit, driver, firmware versions | same major/minor compatibility family | Runtime supports the compiled operators |
| OS/architecture | `uname -a`, `uname -m`, libc | builder platform and toolchain | Wheel platform tag and architecture match |
| Compiler/toolchain | compiler and linker versions | compiler, CMake, Ninja, headers | Meets the source branch's build requirements |
| Resources | disk, RAM, shared memory | disk, RAM, parallelism | Enough capacity for compilation and wheel staging |

## Recommended evidence commands

Run inside each relevant container and retain the output with the build/test record:

```bash
python -V
python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
python -c "import torch_npu; print(torch_npu.__file__)"  # after installation
uname -m
df -h
free -h
```

Use the runtime/vendor commands available on the host to record driver and CANN versions; do not assume a command name across images. Compare source documentation and release notes for the exact branch when versions differ.

## Stop conditions

Stop before installation when any of these is unresolved: Python or platform tag mismatch, unsupported torch/torch_npu pairing, incompatible CANN/driver family, missing required compiler/toolchain, or insufficient disk/RAM. Ask for the intended compatibility target or choose a documented compatible container; do not “try and see” by overwriting the test environment.

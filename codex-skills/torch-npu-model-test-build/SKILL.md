---
name: torch-npu-model-test-build
description: "Run PyTorch models with torch_npu in reproducible Docker environments, including log-driven command recovery, package installation, optional torch_npu source builds, and eager/inductor or precision comparisons. Use for Ascend NPU model testing and torch_npu wheel build/install workflows; do not use for generic CPU/GPU Docker or unrelated PyTorch packaging."
---

# Torch NPU Model Test and Build

Use this skill to turn a model-test run or CI artifact into a repeatable, parameterized workflow. Keep host operations (Docker discovery/start/creation) separate from commands executed inside a test or build container. Never assume the example server, image, model, branch, device ID, paths, container name, or package version are reusable; discover or ask for them from the current artifacts.

## Operating rules

- Start by identifying the target model repository, test runner, expected modes, NPU device(s), torch/torch_npu versions, and whether a replacement wheel is required.
- Prefer the project's existing scripts and logged commands. Extract complete commands from logs with a structured search (for example, `rg -n "docker (run|exec)|common_run_cmd|install\.sh|build\.sh" <log>`), then normalize only environment-specific values.
- Before running a command, show or record the resolved values for image, container, device visibility, host mounts, model path, runner arguments, and output/report locations. Preserve quoting for paths and environment values.
- Treat credentials, SSH mounts, proxy settings, privileged mode, host networking, and large shared-memory or memlock limits as sensitive or high-impact. Use them only when the source workflow requires them and the user has authorized the operation.
- Reuse an existing compatible container when possible. For a new container, make the name unique and verify it is not already in use. Do not remove containers, images, source trees, or wheels unless explicitly requested.
- Keep package installation and model execution in the same test container so the tested environment is unambiguous. Verify the installed versions after any overlay installation.
- Run a minimal smoke test before a full matrix. Capture stdout/stderr, exit codes, elapsed time, report IDs, profiling artifacts, and precision results.
- If a command fails, classify it as host/container setup, dependency/package, device/runtime, model/data, compiler, or test-script failure before changing parameters. Preserve the failing command and relevant logs.

## Workflow

1. **Discover and parameterize.** Read the supplied log, CI record, or repository scripts. Extract the container launch, environment installation, model runner, and (if applicable) build commands. Map them to the variables in [references/workflow.md](references/workflow.md).
2. **Prepare the test container.** Validate Docker availability, image availability, NPU driver visibility, mounts, and device allocation. Create or start the container, then use a non-interactive `docker exec` shell that sources the required profiles and records `python`, `torch`, `torch_npu`, and driver versions.
3. **Install the test environment.** Run the project's installer with the correct package directory and model/test data paths. Keep an installation log and stop on the first failed package or unresolved dependency.
4. **Optionally build torch_npu.** Select a build container whose Python, PyTorch, CANN/driver, architecture, and source branch match the test target. Clone the requested repository and branch into a writable workspace, configure proxy only if needed, run the repository's documented build entrypoint, and locate the resulting wheel under its `dist` (or documented output) directory. Read [references/build-matrix.md](references/build-matrix.md) when compatibility is uncertain.
5. **Overlay and verify.** Install the exact wheel into the test container. Check `pip show`, import success, and version/ABI compatibility before testing. Do not silently install a wheel built for a different Python, architecture, or PyTorch line.
6. **Run the test matrix.** Execute the selected runner for `eager` and/or `inductor`. Add profiling/report variables only when requested. Run precision checks as a separate, clearly labeled pass; do not conflate performance and precision failures.
7. **Summarize artifacts.** Report resolved inputs, commands (with secrets redacted), pass/fail per mode, precision status, profiling/report locations, wheel path and checksum if built, and any residual warnings or cleanup recommendations.

## Command construction

Use the templates and checklists in [references/workflow.md](references/workflow.md). Replace every placeholder in angle brackets; never copy example values such as a particular IP, image tag, model name, date directory, device number, or report ID without confirming they exist in the current environment. Prefer arrays or quoted shell variables when constructing commands programmatically, and avoid embedding untrusted log text directly into a shell command.

For a request that only needs model execution, skip the source-build path. For a request that asks to compile/install a new torch_npu, use the build path and compatibility checklist before running the model matrix.

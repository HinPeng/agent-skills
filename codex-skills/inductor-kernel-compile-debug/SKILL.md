---
name: inductor-kernel-compile-debug
description: Diagnose torch_npu/PyTorch Inductor failures tied to one generated Triton kernel and autotune config, including compiler exceptions and NPU prerun launch or synchronization RuntimeErrors. Capture transient source plus shared real inputs or tensor metadata, replay one config, and export a standalone compile-only or compile-and-launch reproducer. Do not use for numerical mismatches after successful execution.
---

# Inductor Kernel Compile Debug

Isolate one generated kernel and autotune config without recompiling or
serializing unrelated Inductor kernels. The packaged scripts target the
`torch_npu` `NPUCachingAutotuner` flow:

- `scripts/trace_compile.py` runs the original test and captures matching
  compilation attempts; its opt-in prerun mode also snapshots arguments before
  the first candidate launch.
- `scripts/repro_compile.py` replays one captured compile or prerun attempt and
  exports an independent reproducer.

An exported reproducer is a direct fixed-config Triton program. It strips the
outer Inductor heuristic decorator and calls `ASTSource`, `GPUTarget`,
`triton.compile`, and the compiled-kernel runner directly; it does not discover
an `NPUCachingAutotuner` or call `make_launcher`.

Run both scripts in the exact failing environment. Generated source, serialized
NPU tensors, compiler behavior, and device binaries are stack- and
hardware-specific; a reference machine is not proof of reproduction on the
target machine.

## Choose the failure mode

Use compile mode when the innermost traceback reaches
`NPUCachingAutotuner._precompile_config`, `triton.compile`, MLIR/LLVM lowering,
or the device compiler.

Use prerun mode when compilation succeeded and the failure is raised while the
autotuner executes its `PreRun` candidate path: `pre_hook`, argument cloning or
reset, launcher invocation, event recording, or NPU synchronization.
Prerun replay launches the captured kernel once; it is not compile-only.

Neither mode diagnoses a numerical mismatch after successful execution. A
failure that depends on surrounding graph state, RNG, concurrent streams, or a
previous asynchronously failing kernel may require a graph-level reproducer.

## Capture the target kernel

Choose a distinctive regular expression from the failing kernel name. The
pattern is mandatory so a typo cannot silently instrument every generated
kernel. Use a fresh durable capture directory for each attempt:

```bash
export TRITON_TRACE_CAPTURE_DIR="$PWD/inductor_kernel_capture"
export TRITON_TRACE_KERNEL_PATTERN='flex_attention_backward'
unset TRITON_TRACE_SERIALIZE
unset TRITON_TRACE_PRERUN

python /path/to/skill/scripts/trace_compile.py \
  /path/to/test_file.py \
  TestClass.test_method
```

These variables are read when `trace_compile.py` starts, before it imports and
runs the target. Export them in the shell (or place them on the same command
line); assigning them inside the target after the wrapper has started cannot
change the capture root.

Pytest-style selectors such as `TestClass::test_method` and
`test_file.py::TestClass::test_method` are normalized by the wrapper.

For a `PreRun [...] RuntimeError`, enable the launch-input snapshot explicitly:

```bash
export TRITON_TRACE_PRERUN=1
export TRITON_TRACE_PRERUN_INPUTS=values

python /path/to/skill/scripts/trace_compile.py \
  /path/to/test_file.py \
  TestClass.test_method
```

Choose one input mode:

- `values` or `1` (default): serialize one full-value input snapshot per kernel
  invocation and share it across all of that invocation's candidates. The
  shared container stores the stable launch/input prefix; candidate-specific
  runtime blocks stay in each journal record and are appended during replay.
- `metadata`: save only scalar arguments and each Tensor's shape, stride,
  dtype, device, storage offset, and `requires_grad`; replay creates random
  backing storage with the recorded layout. Capture still writes a `.pt`
  container for atomic in-process replay; metadata export loads that container
  once and embeds the metadata dictionary directly instead of base64-encoding
  the `.pt`.
- `none` or `0`: journal diagnostics only. A launch reproducer cannot be
  created without an input artifact.

The movable capture root contains:

```text
inductor_kernel_capture/
|-- journal.jsonl
|-- kernel_sources/
|   `-- <kernel>_<source-hash>.py
`-- prerun_inputs/
    `-- <kernel>_<pid>_<input-group>.shared.pt
```

The tracer archives generated source before compiler entry. In prerun mode it
uses one `_benchmark_candidate_entries` invocation as the candidate batch. It
snapshots the base arguments once; each candidate records its own config and
runtime blocks while referencing the same `input_group_id`. This shares inputs
across candidates without incorrectly reusing them for a later invocation of
the same kernel. All-success snapshots are removed when the batch returns. If
any candidate fails, the one shared snapshot is retained and may be referenced
by multiple failed records. Older flows without that batch method fall back to
tuner plus outer `args`/`kwargs` object identity and clean at the next batch or
normal process exit. A hard abort may leave a journaled
`.shared.pending.pt`, which is still a replay candidate if the file is
complete.

Full-value capture can be large, may contain sensitive tensor data, and may add
a device synchronization or otherwise perturb timing. Metadata mode avoids
reading tensor values, but it does not preserve values or aliasing between
Tensor arguments. Do not use it for data-dependent failures such as indirect
indexing, or for failures that depend on overlapping inputs. Treat `.pt`
captures and exported files as sensitive, and load only artifacts you trust.
Metadata replay uses deterministic random data with seed `0`; override it with
`TRITON_REPRO_SEED=<integer>` when testing other synthetic values.

Keep normal compile concurrency during discovery. Do not globally set
`TORCHINDUCTOR_COMPILE_THREADS=1`, `TRITON_DISABLE_CACHE=1`, or
`TRITON_ALWAYS_COMPILE=1`; those settings change unrelated compilation and
cache behavior.

## Identify the failed record

Read `journal.jsonl` as JSON Lines rather than assuming its last line is the
failure. Compile records use:

- `BEGIN`: persisted before compiler entry.
- `PASS`: the config compiled successfully.
- `PYTHON_EXCEPTION`: compiler entry raised; this ID is the primary compile
  repro candidate.

Prerun records use:

- `PRERUN_BEGIN`: candidate identity, input group, and planned shared input path
  were journaled.
- `PRERUN_READY`: the shared snapshot is available and launch is next.
- `PRERUN_PASS`: launch and synchronization succeeded; an all-success shared
  snapshot is cleaned after the candidate batch.
- `PRERUN_PYTHON_EXCEPTION`: prerun raised; this ID and retained input are the
  primary launch repro candidate.

Find ordinary Python failures with:

```bash
rg '"event": "(PYTHON_EXCEPTION|PRERUN_PYTHON_EXCEPTION)"' \
  "$TRITON_TRACE_CAPTURE_DIR/journal.jsonl"
```

If the process or compiler aborts before an exception record, find IDs whose
`BEGIN`/`PRERUN_READY` has no corresponding terminal event. Parallel compile
configs can leave several candidates. Rerun in a new capture directory with:

```bash
export TRITON_TRACE_SERIALIZE=1
```

This serializes only compilation configs for the matched kernel; unrelated
kernels keep their normal path and concurrency.

If no prerun records appear, verify that the regex matches `fn_name` or
`kernel_name`, that `TRITON_TRACE_PRERUN=1` is exported, and that the installed
`torch_npu` still routes candidate preruns through `_measure_prerun_ms`.

## Replay and export

Replay exactly one compile config:

```bash
python /path/to/skill/scripts/repro_compile.py \
  "$TRITON_TRACE_CAPTURE_DIR" \
  <COMPILE_OR_PRERUN_ID>
```

This calls only `_precompile_config(config)` and remains useful for proving
that a prerun failure is not a compiler failure.

Replay a failed prerun candidate, including one launch and synchronization:

```bash
python /path/to/skill/scripts/repro_compile.py \
  "$TRITON_TRACE_CAPTURE_DIR" \
  <PRERUN_ID> \
  --launch
```

If the captured device index is unavailable, remap real tensor storages or the
metadata-mode construction device:

```bash
python /path/to/skill/scripts/repro_compile.py \
  "$TRITON_TRACE_CAPTURE_DIR" \
  <PRERUN_ID> \
  --launch \
  --map-location npu:0
```

Export independent reproducers:

```bash
# Compile-only: embeds generated source and config.
python /path/to/skill/scripts/repro_compile.py \
  "$TRITON_TRACE_CAPTURE_DIR" \
  <RECORD_ID> \
  --export "$PWD/inductor_kernel_compile_repro.py"

# Compile-and-launch: embeds the captured input payload (metadata literal or
# full-value bytes, depending on the capture mode).
python /path/to/skill/scripts/repro_compile.py \
  "$TRITON_TRACE_CAPTURE_DIR" \
  <PRERUN_ID> \
  --export-launch "$PWD/inductor_kernel_prerun_repro.py"
```

The launch export is one Python file and no longer depends on the capture
directory, journal, archived source, input `.pt`, or original Inductor cache.
Metadata mode loads the metadata-only `.pt` once during export, embeds a Python
literal, and reconstructs deterministic synthetic storage at runtime; the `.pt`
bytes are not embedded. Values mode retains the full-value `.pt` as base64
because the values cannot be represented by metadata alone. Both forms use the
direct fixed-config Triton path and do not instantiate Inductor heuristics or
an Inductor launcher.

The direct path follows the shape of a hand-written Triton repro: the exported
kernel keeps `@triton.jit`, constructs `ASTSource` and `GPUTarget`, calls
`triton.compile` once with the selected candidate's options, and invokes the
compiled binary with the captured grid and arguments. It does not import or
execute `triton_heuristics`, construct `NPUCachingAutotuner`, call
`_precompile_config()`, or call `make_launcher()`. For `FixedGrid` and
`PrecomputedGrid` kernels, extra `_grid_*` launcher values are separated from
the binary's signature before the direct call.
The resulting file is intended to look and behave like a small hand-written
`output_code` reproducer such as
`/Users/hp/workspace/AgentSpace/FlexAttn/test_results/output_code/test_a_batch_dynamic[dtype0-trig].py`:
one Triton kernel, explicit inputs, one fixed launch configuration, and a
single direct launch, with no surrounding Inductor graph or autotune search.
To remap its inputs, set `TRITON_REPRO_MAP_LOCATION=npu:0` when running it. It
still requires compatible Python, PyTorch, `torch_npu`, Triton, compiler,
environment, and hardware. An already-exported legacy file is not rewritten in
place; regenerate it from the capture journal with the current exporter.

If a cache hit prevents the desired compiler path, isolate or bypass the cache
only for this single-kernel replay. Verify portability by copying the export
outside the capture directory and running the copy. Do not delete the original
capture unless the user asks.

## Report the result

Report:

- generated function and Inductor kernel names;
- journal ID, candidate ID, input group ID/mode, exact config, and runtime
  blocks;
- direct-export signature, constexpr constants, target architecture, and
  effective compile options (these are stored under `direct_compile` in new
  journal records);
- archived source, retained input, and exported reproducer paths;
- whether failure arose in compile, pre-hook/reset, launch, or synchronization;
- innermost error and whether direct replay and independent export reproduce;
- relevant stack versions and target hardware;
- whether input capture could have perturbed timing.

An old journal whose source or input path points only to an already-deleted
temporary file cannot be repaired from metadata. Capture it again.

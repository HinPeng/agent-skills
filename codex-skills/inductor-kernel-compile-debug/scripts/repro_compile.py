"""Replay exactly one captured Triton compile or prerun attempt.

Usage:
    python repro_compile.py CAPTURE CONFIG_ID
    python repro_compile.py CAPTURE CONFIG_ID --export compile_repro.py
    python repro_compile.py CAPTURE PRERUN_ID --launch
    python repro_compile.py CAPTURE PRERUN_ID --export-launch launch_repro.py

In-process replay invokes only the captured autotuner config. Prerun replay also
restores the captured launch arguments, creates its launcher, launches it once,
and synchronizes the NPU. Exported files use a direct fixed-config Triton path:
they strip the outer heuristic decorator and call ASTSource/GPUTarget/triton
compile directly, without an Inductor autotuner or generated launcher.
"""

import argparse
import ast
import base64
import importlib.util
import json
import os
import pprint
import sys
from collections.abc import Mapping
from pathlib import Path

import torch
from torch._inductor.runtime.triton_heuristics import (
    config_from_dict,
    config_to_dict,
)


# ``trace_compile.py`` tags non-string mapping keys (notably Triton attrs
# descriptors whose keys are tuples) before writing JSON. Keep the decoder
# local so replay/export does not need to import the tracer.
_JSON_MAPPING_ITEMS = "__inductor_trace_mapping_items__"
_JSON_TUPLE = "__inductor_trace_tuple__"
_JSON_REPR = "__inductor_trace_repr__"


def _restore_json_metadata(value):
    if isinstance(value, list):
        return [_restore_json_metadata(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {_JSON_TUPLE}:
        return tuple(_restore_json_metadata(item) for item in value[_JSON_TUPLE])
    if set(value) == {_JSON_REPR}:
        # This is diagnostic-only data; retaining the repr is safer than
        # guessing a Python object from an untrusted journal.
        return value[_JSON_REPR]
    if set(value) == {_JSON_MAPPING_ITEMS}:
        items = value[_JSON_MAPPING_ITEMS]
        if not isinstance(items, list):
            # A malformed journal should produce a useful diagnostic later,
            # rather than an obscure ``TypeError: cannot unpack`` here.
            return {
                _JSON_MAPPING_ITEMS: _restore_json_metadata(items),
            }
        restored = {}
        for pair in items:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                # Preserve malformed entries as a diagnostic value.  Journals
                # are user-controlled artifacts and must not make export
                # crash while being decoded.
                return {
                    _JSON_MAPPING_ITEMS: _restore_json_metadata(items),
                }
            key = _restore_json_metadata(pair[0])
            item = _restore_json_metadata(pair[1])
            try:
                hash(key)
            except TypeError:
                # JSON mapping keys are required to be hashable.  Retaining a
                # repr keeps the exporter deterministic for an old/corrupt
                # capture without evaluating arbitrary text.
                key = repr(key)
            restored[key] = item
        return restored
    return {
        key: _restore_json_metadata(item)
        for key, item in value.items()
    }


def normalized(value):
    """Return a stable representation matching the journal serialization."""
    return json.dumps(value, default=repr, sort_keys=True)


def load_record(journal_path, config_id):
    records = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matching = [record for record in records if record.get("id") == config_id]
    if not matching:
        raise RuntimeError(
            f"Cannot find record id={config_id} in {journal_path}"
        )

    # BEGIN is followed by a terminal event in a normal process. Choosing the
    # latest record also handles a reused journal whose numeric IDs restart.
    # If the process aborts, BEGIN or PRERUN_READY remains the latest record.
    return matching[-1]


def resolve_record_artifact(journal_path, recorded_path, artifact_dir):
    if not recorded_path:
        return None

    recorded = Path(recorded_path).expanduser()
    candidates = [recorded]
    if not recorded.is_absolute():
        candidates.append(journal_path.parent / recorded)
    candidates.append(journal_path.parent / artifact_dir / recorded.name)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return recorded.resolve()


def load_generated_module(source_path, config_id):
    module_name = f"captured_triton_kernel_{config_id}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generated module from {source_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def find_tuner(module, fn_name):
    tuners = [
        value
        for value in vars(module).values()
        if callable(getattr(value, "_precompile_config", None))
        and hasattr(value, "configs")
    ]
    exact = [
        tuner
        for tuner in tuners
        if callable(getattr(tuner, "get_fn_name", None))
        and tuner.get_fn_name() == fn_name
    ]
    if len(exact) == 1:
        return exact[0]
    if not exact and len(tuners) == 1:
        return tuners[0]

    found_names = [
        tuner.get_fn_name()
        if callable(getattr(tuner, "get_fn_name", None))
        else type(tuner).__name__
        for tuner in tuners
    ]
    raise RuntimeError(
        f"Expected one tuner for {fn_name!r}; found {found_names}"
    )


def find_config(tuner, wanted_config):
    matches = [
        config
        for config in tuner.configs
        if normalized(config_to_dict(config)) == normalized(wanted_config)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        # Runtime-block candidate planning can derive a compile config which is
        # not stored verbatim in the freshly imported tuner's default list.
        return config_from_dict(wanted_config)

    available = [config_to_dict(config) for config in tuner.configs]
    raise RuntimeError(
        f"Expected one matching config, found {len(matches)}. "
        f"Wanted: {wanted_config!r}. Available: {available!r}"
    )


def torch_load(path_or_buffer, map_location=None):
    kwargs = {}
    if map_location is not None:
        kwargs["map_location"] = map_location
    try:
        return torch.load(path_or_buffer, weights_only=False, **kwargs)
    except TypeError:
        # PyTorch versions predating weights_only still accept this payload.
        return torch.load(path_or_buffer, **kwargs)


def resolve_dtype(dtype_name):
    name = dtype_name.removeprefix("torch.")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"Cannot reconstruct dtype {dtype_name!r}")
    return dtype


def required_storage_size(shape, stride, storage_offset):
    if len(shape) != len(stride):
        raise RuntimeError(
            f"Captured shape/stride rank mismatch: {shape} vs {stride}"
        )
    if storage_offset < 0 or any(step < 0 for step in stride):
        raise RuntimeError(
            "Metadata replay does not support negative storage offsets or strides"
        )
    if any(size < 0 for size in shape):
        raise RuntimeError(f"Invalid captured shape: {shape}")
    if any(size == 0 for size in shape):
        return max(storage_offset + 1, 1)
    return storage_offset + sum(
        (size - 1) * step for size, step in zip(shape, stride)
    ) + 1


def random_storage(size, dtype, device):
    try:
        if dtype == torch.bool:
            return torch.randint(0, 2, (size,), device=device, dtype=dtype)
        if dtype.is_floating_point or dtype.is_complex:
            return torch.randn((size,), device=device, dtype=dtype)
        return torch.randint(0, 2, (size,), device=device, dtype=dtype)
    except (RuntimeError, TypeError):
        # Some uncommon dtypes do not implement random generation.
        return torch.zeros((size,), device=device, dtype=dtype)


def materialize_input_metadata(spec, map_location=None):
    kind = spec.get("kind")
    if kind == "literal":
        return spec.get("value")
    if kind == "tuple":
        return tuple(
            materialize_input_metadata(item, map_location)
            for item in spec["items"]
        )
    if kind == "list":
        return [
            materialize_input_metadata(item, map_location)
            for item in spec["items"]
        ]
    if kind == "dict":
        return {
            materialize_input_metadata(
                key,
                map_location,
            ): materialize_input_metadata(value, map_location)
            for key, value in spec["items"]
        }
    if kind != "tensor":
        raise RuntimeError(f"Unknown captured input metadata kind: {kind!r}")

    dtype = resolve_dtype(spec["dtype"])
    device = map_location or spec["device"]
    shape = tuple(spec["shape"])
    stride = tuple(spec["stride"])
    storage_offset = int(spec["storage_offset"])
    storage_size = required_storage_size(shape, stride, storage_offset)
    storage = random_storage(storage_size, dtype, device)
    tensor = torch.as_strided(
        storage,
        shape,
        stride,
        storage_offset,
    )
    if spec.get("requires_grad") and (
        dtype.is_floating_point or dtype.is_complex
    ):
        tensor.requires_grad_(True)
    return tensor


def _stable_launch_prefix(
    args,
    runtime_blocks,
    legacy_launch_args=None,
    launch_prefix=None,
):
    """Recover a candidate-independent launch prefix from old payloads.

    Older captures stored the complete launch tuple, including the first
    candidate's runtime blocks.  Reusing that tuple for another candidate is
    incorrect, so strip the recorded suffix before appending the selected
    record's runtime blocks.  New format-3 captures may store ``launch_prefix``
    explicitly when it differs from ``args``.
    """
    args = tuple(args)
    runtime_blocks = tuple(runtime_blocks)
    if launch_prefix is not None:
        return tuple(launch_prefix)
    if legacy_launch_args is None:
        return args
    legacy = tuple(legacy_launch_args)
    runtime_count = len(runtime_blocks)
    if runtime_count and len(legacy) >= len(args) + runtime_count:
        prefix = legacy[:-runtime_count]
        if len(prefix) >= len(args):
            return prefix
    # A legacy payload without a runtime suffix may still contain extra
    # launcher arguments.  Preserve its longer prefix when it is compatible.
    if len(legacy) >= len(args):
        return legacy
    return args


def build_launch_args(
    tuner,
    args,
    runtime_blocks,
    legacy_launch_args=None,
    launch_prefix=None,
):
    prefix = _stable_launch_prefix(
        args,
        runtime_blocks,
        legacy_launch_args=legacy_launch_args,
        launch_prefix=launch_prefix,
    )
    builder = getattr(tuner, "_build_runtime_launch_args", None)
    if callable(builder):
        return tuple(builder(prefix, runtime_blocks))
    return (*prefix, *tuple(runtime_blocks))


def load_prerun_payload(
    input_path,
    tuner,
    runtime_blocks,
    map_location=None,
):
    payload = torch_load(input_path, map_location=map_location)
    required = {"args", "kwargs"}
    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeError(
            f"Captured prerun payload {input_path} is missing {missing}"
        )

    capture_mode = payload.get("capture_mode", "legacy-values")
    synthetic_seed = None
    if capture_mode == "metadata":
        synthetic_seed = int(os.environ.get("TRITON_REPRO_SEED", "0"))
        torch.manual_seed(synthetic_seed)
        args = tuple(materialize_input_metadata(payload["args"], map_location))
        kwargs = dict(materialize_input_metadata(payload["kwargs"], map_location))
        launch_prefix = (
            materialize_input_metadata(payload["launch_prefix"], map_location)
            if payload.get("launch_prefix") is not None
            else None
        )
    else:
        args = tuple(payload["args"])
        kwargs = dict(payload["kwargs"])
        launch_prefix = payload.get("launch_prefix")
    runtime_blocks = tuple(runtime_blocks)
    launch_args = build_launch_args(
        tuner,
        args,
        runtime_blocks,
        legacy_launch_args=payload.get("launch_args"),
        launch_prefix=launch_prefix,
    )
    return {
        "args": args,
        "launch_args": launch_args,
        "kwargs": kwargs,
        "runtime_blocks": runtime_blocks,
        "capture_mode": capture_mode,
        "synthetic_seed": synthetic_seed,
    }


def run_captured_prerun(tuner, config, payload):
    from torch._dynamo.device_interface import DeviceGuard

    args = tuple(payload["args"])
    launch_args = tuple(payload["launch_args"])
    kwargs = dict(payload["kwargs"])
    device_interface = tuner.get_device_interface()
    device_index = tuner.triton_meta.get("device", 0)

    with DeviceGuard(device_interface, device_index):
        device_interface.synchronize(device_interface.current_device())
        launcher = tuner._precompile_config(config).make_launcher()
        stream = device_interface.get_raw_stream(
            device_interface.current_device()
        )

        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record()
        if launcher.config.pre_hook is not None:
            launcher.config.pre_hook(
                {
                    **dict(zip(tuner.arg_names, launch_args)),
                    **launcher.config.kwargs,
                }
            )
        cloned_args, cloned_kwargs = tuner.clone_args(*launch_args, **kwargs)
        tuner.reset_to_zero_args(*args, **kwargs)
        launcher(*cloned_args, **cloned_kwargs, stream=stream)
        end_event.record()
        torch.npu.synchronize()
        return start_event.elapsed_time(end_event)


# The direct exporter below intentionally replaces the legacy tuner-based
# standalone footer.  In-process --launch replay above remains unchanged.
# ---------------------------------------------------------------------------
# Direct standalone export
# ---------------------------------------------------------------------------
#
# The first implementation of --export-launch appended a copy of the
# generated module and then discovered its NPUCachingAutotuner at runtime.
# That is useful for an in-process replay, but it is not a single-kernel
# reproducer: importing the file still executes Inductor's heuristic decorator
# and the footer calls ``make_launcher``.  The exporter below deliberately
# keeps only the Triton JIT function and calls Triton's compiler/runtime API
# with the captured signature, constants, target, and candidate options.


def _ast_dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _ast_dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _ast_metadata_value(node):
    """Evaluate the literal subset used by generated Triton decorators."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in {"True", "False", "None"}:
            return {"True": True, "False": False, "None": None}[node.id]
        return node.id
    if isinstance(node, ast.Attribute):
        return _ast_dotted_name(node)
    if isinstance(node, ast.List):
        return [_ast_metadata_value(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        # Tuple keys are significant in Triton attrs descriptors, e.g.
        # ``{(0,): [['tt.divisibility', 16]]}``.
        return tuple(_ast_metadata_value(item) for item in node.elts)
    if isinstance(node, ast.Set):
        return {_ast_metadata_value(item) for item in node.elts}
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values):
            if key is None:  # dictionary unpacking; not expected in metadata
                continue
            key_value = _ast_metadata_value(key)
            try:
                hash(key_value)
            except TypeError:
                key_value = repr(key_value)
            result[key_value] = _ast_metadata_value(value)
        return result
    if isinstance(node, ast.Call):
        name = _ast_dotted_name(node.func) or "<call>"
        short_name = name.rsplit(".", 1)[-1]
        if short_name == "from_dict" and name.endswith("AttrsDescriptor.from_dict"):
            # Generated Triton modules spell the descriptor as
            # ``AttrsDescriptor.from_dict({...})``.  Keep the serializable
            # dictionary so the standalone footer can reconstruct the
            # version-appropriate descriptor class at runtime.
            if node.args:
                return _ast_metadata_value(node.args[0])
            for keyword in node.keywords:
                if keyword.arg in ("data", "attrs"):
                    return _ast_metadata_value(keyword.value)
            return {}
        if short_name in {"set", "frozenset"}:
            if node.args:
                return _ast_metadata_value(node.args[0])
            return []
        if short_name in {"tuple", "list"}:
            if node.args:
                value = _ast_metadata_value(node.args[0])
                return list(value) if isinstance(value, (list, tuple, set)) else [value]
            return []
        if short_name == "DeviceProperties":
            return {
                keyword.arg: _ast_metadata_value(keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            }
        if short_name == "AttrsDescriptor":
            return {
                "__attrs_descriptor__": {
                    keyword.arg: _ast_metadata_value(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
            }
        return {
            "__call__": name,
            "args": [_ast_metadata_value(arg) for arg in node.args],
            "kwargs": {
                keyword.arg: _ast_metadata_value(keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            },
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _ast_metadata_value(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    try:
        return ast.unparse(node)
    except Exception:
        return repr(node)


def _normalise_literal(value):
    """Make a metadata value safe to embed as a Python literal."""
    value = _restore_json_metadata(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            key = _normalise_literal(key)
            try:
                hash(key)
            except TypeError:
                key = repr(key)
            normalized[key] = _normalise_literal(item)
        return normalized
    if isinstance(value, tuple):
        return tuple(_normalise_literal(item) for item in value)
    if isinstance(value, (list, set, frozenset)):
        return [_normalise_literal(item) for item in value]
    return repr(value)


def _npu_compile_option_names():
    """Return the backend option names without importing Inductor.

    The direct export is intentionally independent of Inductor's heuristic
    layer.  Triton-Ascend itself exposes the authoritative option dataclass,
    so use it opportunistically while exporting an older source/journal pair;
    a small conservative fallback keeps source-only exports portable to builds
    that do not expose the dataclass.
    """
    try:
        from triton.backends.ascend.compiler import NPUOptions

        return set(getattr(NPUOptions, "__dataclass_fields__", {}))
    except Exception:
        return None


def _source_compile_options(raw_options, wanted_config, constexpr_names):
    raw_options = dict(raw_options) if isinstance(raw_options, Mapping) else {}
    wanted_config = (
        dict(wanted_config) if isinstance(wanted_config, Mapping) else {}
    )
    option_names = _npu_compile_option_names()
    standard_options = {
        "num_warps",
        "num_stages",
        "debug",
        "compile_mode",
    }
    if option_names:
        candidate_options = {
            key: value
            for key, value in wanted_config.items()
            if key in option_names or key in standard_options
        }
        captured_options = {
            key: value
            for key, value in raw_options.items()
            if key in option_names or key in standard_options
        }
    else:
        # These keys are consumed by Inductor's grid/config planner and are
        # never compiler options on the Ascend backend.  Keep unknown keys when
        # the backend schema is unavailable so a newer vendor option is not
        # silently lost.
        planner_only = {
            *constexpr_names,
            "split_axis",
            "split_blocks",
            "fixed_grid",
            "precomputed_grids",
            "runtime_block_arg_names",
            "runtime_block_append_order",
            "extra_launcher_args",
        }
        candidate_options = {
            key: value
            for key, value in wanted_config.items()
            if key not in planner_only
            and not (
                isinstance(key, str)
                and key.isupper()
                and ("BLOCK" in key or "SPLIT" in key)
            )
        }
        captured_options = raw_options
    options = {
        key: value for key, value in captured_options.items() if value is not None
    }
    options.update(
        {key: value for key, value in candidate_options.items() if value is not None}
    )
    return options


def _find_kernel_metadata_nodes(source_text, fn_name):
    tree = ast.parse(source_text)
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == fn_name
        ),
        None,
    )
    if function is None:
        raise RuntimeError(
            f"Generated source does not contain kernel function {fn_name!r}"
        )
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        keywords = {keyword.arg: keyword.value for keyword in decorator.keywords}
        if "triton_meta" in keywords or "inductor_meta" in keywords:
            return tree, function, keywords
    raise RuntimeError(
        f"Could not find triton_meta/inductor_meta for kernel {fn_name!r}. "
        "Capture with the current trace_compile.py or provide a generated "
        "source containing the Inductor decorator metadata."
    )


def direct_compile_metadata_from_source(source_text, fn_name, wanted_config):
    """Recover direct-compile metadata from an older journal/source pair."""
    try:
        _, function, keywords = _find_kernel_metadata_nodes(source_text, fn_name)
    except RuntimeError:
        # Custom Triton modules may already use only @triton.jit and keep the
        # compile metadata exclusively in the journal.  Infer the argument
        # names in that case; the caller can merge journal-provided metadata.
        tree = ast.parse(source_text)
        function = next(
            (
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == fn_name
            ),
            None,
        )
        if function is None:
            raise
        keywords = {}
    triton_meta = _ast_metadata_value(keywords.get("triton_meta", ast.Dict([], [])))
    inductor_meta = _ast_metadata_value(
        keywords.get("inductor_meta", ast.Dict([], []))
    )
    triton_meta = triton_meta if isinstance(triton_meta, dict) else {}
    inductor_meta = inductor_meta if isinstance(inductor_meta, dict) else {}

    args = [*function.args.posonlyargs, *function.args.args]
    arg_names = [arg.arg for arg in args]
    constexpr_names = []
    for arg in args:
        try:
            annotation = ast.unparse(arg.annotation) if arg.annotation else ""
        except Exception:
            annotation = ""
        if "constexpr" in annotation:
            constexpr_names.append(arg.arg)

    signature = triton_meta.get("signature") or {}
    signature = _normalise_literal(signature)
    constants = _normalise_literal(triton_meta.get("constants") or {})
    if not isinstance(constants, dict):
        constants = {}
    wanted_config = _normalise_literal(wanted_config or {})
    for name in constexpr_names:
        if name in wanted_config:
            constants[name] = wanted_config[name]

    device = triton_meta.get("device") or {}
    if not isinstance(device, dict):
        device = {}
    backend = device.get("type") or triton_meta.get("device_type") or "npu"
    arch = device.get("cc") or triton_meta.get("cc")
    warp_size = device.get("warp_size") or 32

    attrs = triton_meta.get("configs")
    if isinstance(attrs, (tuple, list)):
        attrs = attrs[0] if attrs else {}
    if isinstance(attrs, dict) and "__attrs_descriptor__" in attrs:
        descriptor = attrs.get("__attrs_descriptor__")
        attrs = descriptor if isinstance(descriptor, dict) else {}
    if not isinstance(attrs, dict):
        # The NPU compiler accepts an empty attrs mapping and derives the
        # descriptor from the signature.  Avoid importing a version-specific
        # AttrsDescriptor class into the standalone file.
        to_dict = getattr(attrs, "to_dict", None)
        if callable(to_dict):
            try:
                attrs = to_dict()
            except Exception:
                attrs = {}
        else:
            attrs = {}

    runtime_block_names = list(
        inductor_meta.get("runtime_block_append_order")
        or inductor_meta.get("runtime_block_arg_names")
        or []
    )
    launch_arg_names = [name for name in signature if name not in constexpr_names]
    if not launch_arg_names:
        launch_arg_names = [name for name in arg_names if name not in constexpr_names]
    for name in runtime_block_names:
        if name not in launch_arg_names:
            launch_arg_names.append(name)

    grid_keys = (
        "grid_type",
        "split_axis",
        "axis_names",
        "runtime_block_arg_names",
        "runtime_block_append_order",
        "split_blocks",
        "fixed_grid",
        "precomputed_grids",
        "group_enabled",
        "extra_launcher_args",
    )
    grid_meta = {
        key: _normalise_literal(inductor_meta[key])
        for key in grid_keys
        if key in inductor_meta
    }
    raw_npu_options = triton_meta.get("npu_compile_options") or {}
    option_names = _npu_compile_option_names()
    options = _source_compile_options(
        raw_npu_options,
        wanted_config,
        constexpr_names,
    )
    # Keep the same precedence as NPUCachingAutotuner._precompile_config.  A
    # default compile_mode is needed by older source files that did not print
    # it; an explicit candidate value wins over that default.
    options.setdefault("compile_mode", "simd_simt_template")
    for option_name in ("num_warps", "num_stages", "compile_mode"):
        if option_name in keywords:
            option_value = _ast_metadata_value(keywords[option_name])
            if option_value is not None and option_name not in wanted_config:
                options[option_name] = option_value
    if (
        inductor_meta.get("enable_auto_blockify", False)
        or inductor_meta.get("requires_no_linear_block_remap") is True
    ):
        # Match NPUCachingAutotuner.parse_triton_ascend_options: on versions
        # with an option schema, do not pass a planner-only key to Triton if
        # that backend has removed it.
        if option_names is None or "enable_auto_blockify" in option_names:
            options["enable_auto_blockify"] = True
    if options.get("compile_mode") == "simt_only":
        try:
            import torch_npu._inductor.config as npu_config

            stack_limit = getattr(npu_config, "simt_default_warp_stacksize", None)
        except Exception:
            stack_limit = None
        if stack_limit is not None:
            options["simt_stack_limit"] = stack_limit
    options.setdefault("debug", False)

    return {
        "version": 1,
        "signature": signature,
        "constants": constants,
        "attrs": _normalise_literal(attrs),
        "target": {
            "backend": _normalise_literal(backend),
            "arch": _normalise_literal(arch),
            "warp_size": _normalise_literal(warp_size),
        },
        "device_index": _normalise_literal(device.get("index", 0)),
        "options": _normalise_literal(options),
        "inductor_meta": grid_meta,
        "arg_names": _normalise_literal(arg_names),
        "constexpr_names": _normalise_literal(constexpr_names),
        "launch_arg_names": _normalise_literal(launch_arg_names),
        "runtime_block_arg_names": _normalise_literal(runtime_block_names),
        "base_launch_arg_names": _normalise_literal(
            [name for name in launch_arg_names if name not in runtime_block_names]
        ),
        "extra_launcher_arg_names": _normalise_literal(
            inductor_meta.get("extra_launcher_args", ()) or ()
        ),
    }


def _wrapper_decorator_name(node):
    if isinstance(node, ast.Call):
        return _ast_dotted_name(node.func) or ""
    return _ast_dotted_name(node) or ""


def _is_inductor_wrapper(node):
    name = _wrapper_decorator_name(node).lower()
    return any(
        token in name
        for token in (
            "heuristic",
            "autotune",
            "fixed_config",
            "user_autotune",
            "cached_autotune",
        )
    )


def _contains_inductor_wrapper(node):
    return any(
        isinstance(child, ast.Name)
        and child.id
        in {
            "triton_heuristics",
            "NPUCachingAutotuner",
            "CachingAutotuner",
            "async_compile",
            "AsyncCompile",
        }
        for child in ast.walk(node)
    )


_STANDALONE_TRITON_HELPERS = {
    # These helpers are the small subset emitted by current Inductor/NPU
    # Triton codegen.  They are copied as Triton JIT functions rather than
    # imported from torch._inductor so the exported file remains independent
    # of the heuristic/runtime layer.
    "promote_to_tensor",
    "minimum",
    "maximum",
    "min2",
    "max2",
    "minimum_with_index",
    "maximum_with_index",
    "min_with_index",
    "max_with_index",
}


class _StandaloneSourceTransformer(ast.NodeTransformer):
    """Remove Inductor's outer tuner while retaining a Triton JIT function."""

    def __init__(self):
        super().__init__()
        self.helper_names = set()

    def visit_FunctionDef(self, node):  # noqa: N802
        had_wrapper = any(_is_inductor_wrapper(item) for item in node.decorator_list)
        node.decorator_list = [
            item for item in node.decorator_list if not _is_inductor_wrapper(item)
        ]
        self.generic_visit(node)
        if had_wrapper and not any(
            _wrapper_decorator_name(item).endswith(".jit")
            for item in node.decorator_list
        ):
            node.decorator_list.insert(
                0,
                ast.Attribute(
                    value=ast.Name(id="triton", ctx=ast.Load()),
                    attr="jit",
                    ctx=ast.Load(),
                ),
            )
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Attribute(self, node):  # noqa: N802
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "triton_helpers"
        ):
            helper_name = node.attr
            if helper_name == "set_driver_to_gpu":
                # Driver selection is performed by Triton itself when the
                # direct compile starts; the Inductor helper is unnecessary.
                return ast.Name(
                    id="_standalone_triton_set_driver_to_gpu",
                    ctx=ast.Load(),
                )
            if helper_name not in _STANDALONE_TRITON_HELPERS:
                raise RuntimeError(
                    "Direct export does not have a Triton-native shim for "
                    f"triton_helpers.{helper_name}"
                )
            self.helper_names.add(helper_name)
            return ast.Name(
                id=f"_standalone_triton_{helper_name}",
                ctx=ast.Load(),
            )
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):  # noqa: N802
        module = node.module or ""
        if module.endswith(".triton_heuristics") or module.endswith(".hints"):
            return None
        if module.endswith(".triton_helpers"):
            replacements = []
            for alias in node.names:
                bound_name = alias.asname or alias.name
                if alias.name in {"libdevice", "extension"}:
                    replacements.append(
                        ast.Import(
                            names=[
                                ast.alias(
                                    name=f"triton.language.extra.cann.{alias.name}",
                                    asname=bound_name,
                                )
                            ]
                        )
                    )
                elif alias.name == "math":
                    replacements.append(
                        ast.Assign(
                            targets=[ast.Name(id=bound_name, ctx=ast.Store())],
                            value=ast.Attribute(
                                value=ast.Name(id="tl", ctx=ast.Load()),
                                attr="math",
                                ctx=ast.Load(),
                            ),
                        )
                    )
                else:
                    # A few generated kernels use helper reductions which do
                    # not have a Triton-native spelling.  Keep only that
                    # narrow helper import; the tuner/heuristics path is still
                    # completely absent from the export.
                    replacements.append(
                        ast.ImportFrom(
                            module=module,
                            names=[ast.alias(name=alias.name, asname=alias.asname)],
                            level=node.level,
                        )
                    )
            return replacements
        if module.startswith(("torch._inductor", "torch_npu._inductor")):
            # The outer runtime module contains only Inductor launcher and
            # heuristic helpers.  A direct Triton export must not import it;
            # helper functions that have a Triton-native spelling are handled
            # by the ``*.triton_helpers`` branch above.
            return None
        return node

    def visit_Import(self, node):  # noqa: N802
        names = [
            alias
            for alias in node.names
            if not (
                "triton_heuristics" in alias.name
                or alias.name.startswith("torch._inductor")
                or alias.name.startswith("torch_npu._inductor")
            )
        ]
        return node if names == node.names else (ast.Import(names=names) if names else None)

    def visit_Assign(self, node):  # noqa: N802
        if _contains_inductor_wrapper(node.value):
            return None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):  # noqa: N802
        if _contains_inductor_wrapper(node.value) if node.value is not None else False:
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):  # noqa: N802
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "triton_helpers"
            and node.value.func.attr == "set_driver_to_gpu"
        ):
            return None
        if _contains_inductor_wrapper(node.value):
            return None
        return self.generic_visit(node)


def _standalone_helper_source(helper_names):
    """Return self-contained Triton shims for generated helper calls."""
    if not helper_names:
        return ""

    # Keep the definitions in dependency order.  A few helpers are emitted
    # together because Triton's JIT resolves their globals when the enclosing
    # kernel is compiled, not when this module is imported.
    return r'''


# ---- Triton helper shims (copied into the standalone file) ----
@triton.jit
def _standalone_triton_promote_to_tensor(x):
    return x + tl.zeros((1,), tl.int1)


@triton.jit
def _standalone_triton_is_floating(x):
    return _standalone_triton_promote_to_tensor(x).dtype.is_floating()


@triton.jit
def _standalone_triton_minimum(a, b):
    mask = a < b
    if _standalone_triton_is_floating(a):
        mask |= a != a
    return tl.where(mask, a, b)


@triton.jit
def _standalone_triton_maximum(a, b):
    mask = a > b
    if _standalone_triton_is_floating(a):
        mask |= a != a
    return tl.where(mask, a, b)


@triton.jit
def _standalone_triton_min2(a, dim):
    return tl.reduce(a, dim, _standalone_triton_minimum)


@triton.jit
def _standalone_triton_max2(a, dim):
    return tl.reduce(a, dim, _standalone_triton_maximum)


@triton.jit
def _standalone_triton_minimum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value < b_value
    equal = a_value == b_value
    if _standalone_triton_is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan & (not b_isnan)
        equal |= a_isnan & b_isnan
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def _standalone_triton_maximum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value > b_value
    equal = a_value == b_value
    if _standalone_triton_is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan & (not b_isnan)
        equal |= a_isnan & b_isnan
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def _standalone_triton_restore_reduced_dim(
    reduced, dim: tl.constexpr, ndim: tl.constexpr
):
    if ndim == 1:
        return _standalone_triton_promote_to_tensor(reduced)
    return tl.expand_dims(reduced, dim)


@triton.jit
def _standalone_triton_extremum(value, dim: tl.constexpr, want_max: tl.constexpr):
    if value.dtype.is_floating():
        if want_max:
            return tl.max(value, dim)
        return tl.min(value, dim)
    if want_max:
        return tl.reduce(value, dim, _standalone_triton_maximum)
    return tl.reduce(value, dim, _standalone_triton_minimum)


@triton.jit
def _standalone_triton_max_with_index(value, index, dim: tl.constexpr):
    peak = _standalone_triton_extremum(value, dim, True)
    peak_bcast = _standalone_triton_restore_reduced_dim(
        peak, dim, len(value.shape)
    )
    filler = _standalone_triton_restore_reduced_dim(
        tl.max(index, dim), dim, len(value.shape)
    )
    return peak, tl.min(
        tl.where(value == peak_bcast, index, filler),
        dim,
    )


@triton.jit
def _standalone_triton_min_with_index(value, index, dim: tl.constexpr):
    valley = _standalone_triton_extremum(value, dim, False)
    valley_bcast = _standalone_triton_restore_reduced_dim(
        valley, dim, len(value.shape)
    )
    filler = _standalone_triton_restore_reduced_dim(
        tl.max(index, dim), dim, len(value.shape)
    )
    return valley, tl.min(
        tl.where(value == valley_bcast, index, filler),
        dim,
    )
'''


def _remove_unused_helper_imports(tree):
    loaded_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            node.names = [
                alias
                for alias in node.names
                if not (
                    (alias.asname or alias.name.split(".")[-1])
                    in {"triton_helpers", "triton_heuristics"}
                    and (alias.asname or alias.name.split(".")[-1])
                    not in loaded_names
                )
            ]
        elif isinstance(node, ast.ImportFrom):
            node.names = [
                alias
                for alias in node.names
                if not (
                    (alias.asname or alias.name) in {"triton_helpers", "triton_heuristics"}
                    and (alias.asname or alias.name) not in loaded_names
                )
                and not (
                    (alias.asname or alias.name)
                    in {
                        "AttrsDescriptor",
                        "AutotuneHint",
                        "ReductionHint",
                        "TileHint",
                        "DeviceProperties",
                    }
                    and (alias.asname or alias.name) not in loaded_names
                )
            ]
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom)) or node.names
    ]
    return tree


def standalone_source_without_inductor(source_text, fn_name):
    # Be tolerant when a user feeds an already-exported reproducer back into
    # the tool: discard the previous footer before applying the transform.
    for marker in (
        "# ---- standalone reproducer helpers ----",
        "# ---- direct standalone reproducer helpers ----",
    ):
        if marker in source_text:
            source_text = source_text.split(marker, 1)[0]
    try:
        tree, _, _ = _find_kernel_metadata_nodes(source_text, fn_name)
    except RuntimeError:
        tree = ast.parse(source_text)
        if not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == fn_name
            for node in tree.body
        ):
            raise
    transformer = _StandaloneSourceTransformer()
    tree = transformer.visit(tree)
    tree = _remove_unused_helper_imports(ast.fix_missing_locations(tree))
    try:
        standalone = ast.unparse(tree)
    except Exception as error:
        raise RuntimeError(
            f"Could not rewrite generated source for direct export: {error}"
        ) from error
    if "torch._inductor" in standalone or "torch_npu._inductor" in standalone:
        raise RuntimeError(
            "Direct export still depends on an Inductor runtime import; "
            "recapture the kernel or replace the helper with its Triton-native "
            "equivalent before exporting"
        )
    return standalone + _standalone_helper_source(transformer.helper_names)


def _standalone_materializer_template():
    return r'''


def _standalone_repro_resolve_dtype(dtype_name):
    import torch

    name = dtype_name.removeprefix("torch.")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"Cannot reconstruct dtype {dtype_name!r}")
    return dtype


def _standalone_repro_required_storage_size(shape, stride, storage_offset):
    if len(shape) != len(stride):
        raise RuntimeError(
            f"Captured shape/stride rank mismatch: {shape} vs {stride}"
        )
    if storage_offset < 0 or any(step < 0 for step in stride):
        raise RuntimeError(
            "Metadata replay does not support negative storage offsets or strides"
        )
    if any(size < 0 for size in shape):
        raise RuntimeError(f"Invalid captured shape: {shape}")
    if any(size == 0 for size in shape):
        return max(storage_offset + 1, 1)
    return storage_offset + sum(
        (size - 1) * step for size, step in zip(shape, stride)
    ) + 1


def _standalone_repro_random_storage(size, dtype, device):
    import torch

    try:
        if dtype == torch.bool:
            return torch.randint(0, 2, (size,), device=device, dtype=dtype)
        if dtype.is_floating_point or dtype.is_complex:
            return torch.randn((size,), device=device, dtype=dtype)
        return torch.randint(0, 2, (size,), device=device, dtype=dtype)
    except (RuntimeError, TypeError):
        return torch.zeros((size,), device=device, dtype=dtype)


def _standalone_repro_materialize(spec, map_location=None):
    import torch

    kind = spec.get("kind")
    if kind == "literal":
        return spec.get("value")
    if kind == "tuple":
        return tuple(
            _standalone_repro_materialize(item, map_location)
            for item in spec["items"]
        )
    if kind == "list":
        return [
            _standalone_repro_materialize(item, map_location)
            for item in spec["items"]
        ]
    if kind == "dict":
        return {
            _standalone_repro_materialize(key, map_location):
            _standalone_repro_materialize(value, map_location)
            for key, value in spec["items"]
        }
    if kind != "tensor":
        raise RuntimeError(f"Unknown captured input metadata kind: {kind!r}")

    dtype = _standalone_repro_resolve_dtype(spec["dtype"])
    device = map_location or spec["device"]
    shape = tuple(spec["shape"])
    stride = tuple(spec["stride"])
    storage_offset = int(spec["storage_offset"])
    storage_size = _standalone_repro_required_storage_size(
        shape,
        stride,
        storage_offset,
    )
    storage = _standalone_repro_random_storage(storage_size, dtype, device)
    tensor = torch.as_strided(storage, shape, stride, storage_offset)
    if spec.get("requires_grad") and (
        dtype.is_floating_point or dtype.is_complex
    ):
        tensor.requires_grad_(True)
    return tensor
'''


def _standalone_payload_section(input_bytes, input_payload):
    if input_payload is not None:
        literal = pprint.pformat(input_payload, sort_dicts=False, width=88)
        return (
            "_STANDALONE_INPUT_CAPTURE_MODE = 'metadata'\n"
            f"_STANDALONE_INPUT_METADATA = {literal}\n\n"
            "def _standalone_repro_load_payload():\n"
            "    return _STANDALONE_INPUT_METADATA\n"
        )

    encoded_inputs = base64.b64encode(input_bytes).decode("ascii")
    return r'''
_STANDALONE_INPUT_CAPTURE_MODE = 'values'
_STANDALONE_PRERUN_INPUTS_B64 = __INPUT_BYTES__


def _standalone_repro_load_payload():
    import base64
    import io
    import os
    import torch

    raw = base64.b64decode(_STANDALONE_PRERUN_INPUTS_B64)
    map_location = os.environ.get("TRITON_REPRO_MAP_LOCATION") or None
    kwargs = {"map_location": map_location} if map_location else {}
    try:
        return torch.load(io.BytesIO(raw), weights_only=False, **kwargs)
    except TypeError:
        return torch.load(io.BytesIO(raw), **kwargs)
'''.replace("__INPUT_BYTES__", repr(encoded_inputs))


def standalone_common_footer(fn_name, wanted_config, direct_metadata=None):
    direct_metadata = direct_metadata or {}
    template = r'''

# ---- direct standalone reproducer helpers ----
_STANDALONE_FN_NAME = __FN_NAME__
_STANDALONE_WANTED_CONFIG = __WANTED_CONFIG__
_STANDALONE_SIGNATURE = __SIGNATURE__
_STANDALONE_CONSTANTS = __CONSTANTS__
_STANDALONE_ATTRS = __ATTRS__
_STANDALONE_TARGET = __TARGET__
_STANDALONE_DEVICE_INDEX = __DEVICE_INDEX__
_STANDALONE_OPTIONS = __OPTIONS__
_STANDALONE_INDUCTOR_META = __INDUCTOR_META__
_STANDALONE_LAUNCH_ARG_NAMES = __LAUNCH_ARG_NAMES__
_STANDALONE_RUNTIME_BLOCK_NAMES = __RUNTIME_BLOCK_NAMES__
_STANDALONE_BASE_LAUNCH_ARG_NAMES = __BASE_LAUNCH_ARG_NAMES__
_STANDALONE_EXTRA_LAUNCH_ARG_NAMES = __EXTRA_LAUNCH_ARG_NAMES__


def _standalone_repro_compile():
    import triton
    try:
        import torch
        if hasattr(torch, "npu") and hasattr(torch.npu, "set_device"):
            torch.npu.set_device(_STANDALONE_DEVICE_INDEX)
    except (ImportError, RuntimeError):
        # Triton can still select its active device in environments where the
        # torch_npu convenience namespace is unavailable (for example during
        # compile-only diagnostics with a mocked driver).
        pass
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource

    kernel_fn = globals().get(_STANDALONE_FN_NAME)
    if kernel_fn is None:
        raise RuntimeError(
            f"Kernel function {_STANDALONE_FN_NAME!r} is missing from export"
        )
    target_info = _STANDALONE_TARGET
    if not target_info.get("arch"):
        raise RuntimeError(
            "Captured device architecture is unavailable; recapture with "
            "trace_compile.py so direct compile metadata is journaled"
        )
    target = GPUTarget(
        target_info.get("backend", "npu"),
        target_info["arch"],
        target_info.get("warp_size") or 32,
    )
    attrs = _STANDALONE_ATTRS or None
    if isinstance(attrs, dict):
        # Triton-Ascend 3.2 serializes an AttrsDescriptor as
        # ``{'arg_properties': ..., 'cls': ...}``; older releases use keyword
        # fields such as ``divisible_by_16`` and newer releases consume the
        # raw tuple-key mapping directly.  Rebuild only the descriptor shapes
        # that identify themselves, leaving the newest raw mapping untouched.
        descriptor_wrapper = attrs.get("__attrs_descriptor__")
        if isinstance(descriptor_wrapper, dict):
            attrs = descriptor_wrapper
        descriptor_data = attrs
        try:
            from triton.compiler.compiler import AttrsDescriptor
        except ImportError:
            try:
                from triton.backends.compiler import AttrsDescriptor
            except ImportError:
                AttrsDescriptor = None
        if AttrsDescriptor is not None:
            if {
                "arg_properties",
                "cls",
            }.issubset(descriptor_data) and hasattr(
                AttrsDescriptor,
                "from_dict",
            ):
                attrs = AttrsDescriptor.from_dict(descriptor_data)
            elif any(
                key in descriptor_data
                for key in (
                    "divisible_by_16",
                    "equal_to_1",
                    "pointer_range_32",
                )
            ):
                try:
                    attrs = AttrsDescriptor(**descriptor_data)
                except (TypeError, ValueError):
                    # A backend-specific descriptor may require a different
                    # constructor; passing its mapping through is preferable
                    # to turning a useful capture into an export-time crash.
                    attrs = descriptor_data
    if attrs is None:
        source = ASTSource(
            kernel_fn,
            _STANDALONE_SIGNATURE,
            _STANDALONE_CONSTANTS,
        )
    else:
        source = ASTSource(
            kernel_fn,
            _STANDALONE_SIGNATURE,
            _STANDALONE_CONSTANTS,
            attrs,
        )
    return triton.compile(
        source,
        target=target,
        options=dict(_STANDALONE_OPTIONS),
    )


def _standalone_repro_current_stream():
    """Return the backend's raw current stream without an Inductor import."""
    import torch

    npu = getattr(torch, "npu", None)
    current_stream = getattr(npu, "current_stream", None)
    if not callable(current_stream):
        return None
    stream = current_stream()
    # torch_npu streams expose the integer ABI handle as ``npu_stream``;
    # keeping the object as a fallback also works with lightweight test
    # doubles and Triton versions that accept a stream wrapper directly.
    return getattr(stream, "npu_stream", stream)


def _standalone_repro_runtime_names():
    meta = _STANDALONE_INDUCTOR_META or {}
    return tuple(
        _STANDALONE_RUNTIME_BLOCK_NAMES
        or meta.get("runtime_block_append_order")
        or meta.get("runtime_block_arg_names")
        or ()
    )


def _standalone_repro_extra_names():
    meta = _STANDALONE_INDUCTOR_META or {}
    return tuple(
        _STANDALONE_EXTRA_LAUNCH_ARG_NAMES
        or meta.get("extra_launcher_args")
        or ()
    )


def _standalone_repro_legacy_prefix(args, legacy_launch_args):
    """Strip candidate-specific runtime blocks from an old payload tuple."""
    args = tuple(args)
    if legacy_launch_args is None:
        return args
    legacy = tuple(legacy_launch_args)
    runtime_count = len(_STANDALONE_RUNTIME_BLOCKS)
    if runtime_count and len(legacy) >= len(args) + runtime_count:
        prefix = legacy[:-runtime_count]
        if len(prefix) >= len(args):
            return prefix
    return legacy if len(legacy) >= len(args) else args


def _standalone_repro_split_launch_args(raw_launch_args):
    """Split Inductor's ``kernel args + extra grid args + runtime blocks``."""
    raw_launch_args = tuple(raw_launch_args)
    launch_names = tuple(_STANDALONE_LAUNCH_ARG_NAMES)
    runtime_names = _standalone_repro_runtime_names()
    base_names = tuple(
        _STANDALONE_BASE_LAUNCH_ARG_NAMES
        or (name for name in launch_names if name not in runtime_names)
    )
    extra_names = _standalone_repro_extra_names()
    raw_names = (*base_names, *extra_names, *runtime_names)

    if len(raw_launch_args) == len(raw_names):
        raw_values = dict(zip(raw_names, raw_launch_args))
        kernel_args = tuple(raw_values[name] for name in launch_names)
        return kernel_args, raw_values
    if len(raw_launch_args) == len(launch_names):
        # Newer capture records may already have removed extra launcher args.
        return raw_launch_args, dict(zip(launch_names, raw_launch_args))

    raise RuntimeError(
        "Captured launch argument count does not match the direct signature: "
        f"got {len(raw_launch_args)}, expected {len(raw_names)} (raw) or "
        f"{len(launch_names)} (kernel)"
    )


def _standalone_repro_value(values, name):
    if name in values:
        return values[name]
    lowered = str(name).lower()
    for key, value in values.items():
        if str(key).lower() == lowered:
            return value
    return None


def _standalone_repro_numel(values, axis):
    candidates = (
        f"{axis}_numel",
        f"{axis}numel",
        f"{str(axis).upper()}_NUMEL",
        f"{str(axis).upper()}NUMEL",
    )
    for name in candidates:
        value = _standalone_repro_value(values, name)
        if value is not None:
            return int(value)
    raise RuntimeError(f"No numel argument found for axis {axis!r}")


def _standalone_repro_block(values, axis):
    candidates = (
        f"{str(axis).upper()}BLOCK",
        f"{axis}block",
        f"{axis}_block",
    )
    for name in candidates:
        value = _standalone_repro_value(values, name)
        if value is not None:
            return int(value)
    # A non-runtime block is normally a constexpr/config value.
    for name in candidates:
        value = _standalone_repro_value(_STANDALONE_CONSTANTS, name)
        if value is None:
            value = _standalone_repro_value(_STANDALONE_WANTED_CONFIG, name)
        if value is not None:
            return int(value)
    raise RuntimeError(f"No block value found for axis {axis!r}")


def _standalone_repro_grid_value(expression, values):
    if isinstance(expression, bool):
        return int(expression)
    if isinstance(expression, (int, float)):
        return int(expression)
    if not isinstance(expression, str):
        raise RuntimeError(f"Unsupported captured grid expression: {expression!r}")
    direct = _standalone_repro_value(values, expression)
    if direct is not None:
        return int(direct)
    try:
        return int(expression)
    except ValueError:
        pass
    # Fixed/precomputed grids emitted by Inductor generally contain only
    # symbols and arithmetic.  Evaluate that narrow expression against the
    # captured values, with Python builtins disabled.
    try:
        return int(eval(expression, {"__builtins__": {}}, dict(values)))
    except Exception as error:
        raise RuntimeError(
            f"Cannot evaluate captured grid expression {expression!r}: {error}"
        ) from error


def _standalone_repro_grid(raw_launch_args):
    _, values = _standalone_repro_split_launch_args(raw_launch_args)
    meta = _STANDALONE_INDUCTOR_META or {}

    fixed_grid = meta.get("fixed_grid")
    if fixed_grid is not None:
        if not isinstance(fixed_grid, (tuple, list)):
            raise RuntimeError(f"Unsupported captured fixed_grid: {fixed_grid!r}")
        result = tuple(
            _standalone_repro_grid_value(expression, values)
            for expression in fixed_grid
        )
        return (result + (1, 1, 1))[:3]

    precomputed = meta.get("precomputed_grids")
    if precomputed:
        selected = None
        for candidate in precomputed:
            candidate_config = candidate.get("config", {})
            if all(
                _STANDALONE_WANTED_CONFIG.get(name) == value
                for name, value in candidate_config.items()
            ):
                selected = candidate
                break
        if selected is None:
            raise RuntimeError(
                "No precomputed grid matches the captured fixed config: "
                f"{_STANDALONE_WANTED_CONFIG!r}"
            )
        expressions = selected.get("python") or selected.get("python_slow")
        if expressions is None:
            raise RuntimeError(f"Precomputed grid has no Python expression: {selected!r}")
        result = tuple(
            _standalone_repro_grid_value(expression, values)
            for expression in expressions
        )
        return (result + (1, 1, 1))[:3]

    axis_names = tuple(meta.get("axis_names") or ())
    explicit_runtime_names = _standalone_repro_runtime_names()
    runtime_names = explicit_runtime_names
    grid_type = str(meta.get("grid_type") or "")
    # If an older record has only ``split_blocks`` metadata, defer the
    # runtime-name inference below.  Inferring ``Z0BLOCK``/etc. too early
    # would make the split-block fallback unreachable and incorrectly demand
    # values that were never appended to the launch ABI.
    if not runtime_names and not (
        grid_type == "GridNpu" and meta.get("split_blocks") is not None
    ):
        # Older records did not preserve runtime-block names.  GridNpu uses
        # <AXIS>BLOCK names in split-axis order; infer those conservatively.
        split_axis = tuple(meta.get("split_axis") or range(min(3, len(axis_names))))
        runtime_names = tuple(
            f"{str(axis_names[index]).upper()}BLOCK"
            for index in split_axis[:3]
            if index < len(axis_names)
        )

    grids = []
    if runtime_names and axis_names:
        for block_name in runtime_names[:3]:
            axis = str(block_name).removesuffix("BLOCK").lower()
            if axis not in axis_names:
                raise RuntimeError(
                    f"Cannot map runtime block {block_name!r} to captured axes "
                    f"{axis_names!r}"
                )
            block = _standalone_repro_value(values, block_name)
            if block is None:
                raise RuntimeError(f"No runtime value found for {block_name!r}")
            block = int(block)
            if block <= 0:
                raise RuntimeError(f"Invalid runtime block {block_name}={block}")
            numel = _standalone_repro_numel(values, axis)
            grids.append((numel + block - 1) // block)
        return tuple((grids + [1, 1, 1])[:3])

    # A few older NPU captures did not expose runtime block argument names;
    # their GridNpu metadata instead records the split blocks directly.  Keep
    # this path independent of Inductor's GridExpr implementation.
    if grid_type == "GridNpu" and axis_names:
        split_axis = tuple(meta.get("split_axis") or range(len(axis_names)))
        split_blocks = meta.get("split_blocks")
        for position, axis_index in enumerate(split_axis[:3]):
            if not isinstance(axis_index, int) or axis_index >= len(axis_names):
                raise RuntimeError(
                    f"Invalid captured split axis {axis_index!r} for {axis_names!r}"
                )
            axis = str(axis_names[axis_index])
            numel = _standalone_repro_numel(values, axis)
            if isinstance(split_blocks, (tuple, list)) and position < len(split_blocks):
                block_expr = split_blocks[position]
                block = (
                    1
                    if block_expr is None
                    else _standalone_repro_grid_value(block_expr, values)
                )
            else:
                block = _standalone_repro_block(values, axis)
            if block <= 0:
                raise RuntimeError(f"Invalid split block for {axis}: {block}")
            grids.append(numel if block == 1 else (numel + block - 1) // block)
        return tuple((grids + [1, 1, 1])[:3])

    # Common upstream grid classes use X/Y/ZBLOCK constexprs and xnumel /
    # ynumel / znumel arguments.  This also makes old non-NPU Grid1D/2D/3D
    # captures useful without importing Inductor's grid classes.
    dimension_axes = {
        "Grid1D": ("x",),
        "Grid2D": ("x", "y"),
        "Grid3D": ("x", "y", "z"),
        "Grid2DWithYZOverflow": ("x", "y"),
    }.get(grid_type)
    if dimension_axes:
        for axis in dimension_axes:
            block = _standalone_repro_block(values, axis)
            numel = _standalone_repro_numel(values, axis)
            grids.append((numel + block - 1) // block)
        if grid_type == "Grid2DWithYZOverflow":
            grids.append(1)
        return tuple((grids + [1, 1, 1])[:3])

    return (1, 1, 1)
'''
    replacements = {
        "__FN_NAME__": repr(fn_name),
        "__WANTED_CONFIG__": repr(_normalise_literal(wanted_config)),
        "__SIGNATURE__": repr(_normalise_literal(direct_metadata.get("signature", {}))),
        "__CONSTANTS__": repr(_normalise_literal(direct_metadata.get("constants", {}))),
        "__ATTRS__": repr(_normalise_literal(direct_metadata.get("attrs", {}))),
        "__TARGET__": repr(_normalise_literal(direct_metadata.get("target", {}))),
        "__DEVICE_INDEX__": repr(
            _normalise_literal(direct_metadata.get("device_index", 0))
        ),
        "__OPTIONS__": repr(_normalise_literal(direct_metadata.get("options", {}))),
        "__INDUCTOR_META__": repr(
            _normalise_literal(direct_metadata.get("inductor_meta", {}))
        ),
        "__LAUNCH_ARG_NAMES__": repr(
            _normalise_literal(direct_metadata.get("launch_arg_names", []))
        ),
        "__RUNTIME_BLOCK_NAMES__": repr(
            _normalise_literal(
                direct_metadata.get("runtime_block_arg_names")
                or (direct_metadata.get("inductor_meta") or {}).get(
                    "runtime_block_append_order"
                )
                or (direct_metadata.get("inductor_meta") or {}).get(
                    "runtime_block_arg_names"
                )
                or ()
            )
        ),
        "__BASE_LAUNCH_ARG_NAMES__": repr(
            _normalise_literal(
                direct_metadata.get("base_launch_arg_names")
                or [
                    name
                    for name in direct_metadata.get("launch_arg_names", [])
                    if name
                    not in (
                        direct_metadata.get("runtime_block_arg_names")
                        or (direct_metadata.get("inductor_meta") or {}).get(
                            "runtime_block_append_order"
                        )
                        or (direct_metadata.get("inductor_meta") or {}).get(
                            "runtime_block_arg_names"
                        )
                        or ()
                    )
                ]
            )
        ),
        "__EXTRA_LAUNCH_ARG_NAMES__": repr(
            _normalise_literal(
                direct_metadata.get("extra_launcher_arg_names")
                or (direct_metadata.get("inductor_meta") or {}).get(
                    "extra_launcher_args"
                )
                or ()
            )
        ),
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    return template


def standalone_compile_footer():
    return r'''


def _standalone_repro_main():
    print(f"Compiling fixed config only: {_STANDALONE_FN_NAME}")
    print(f"Config: {_STANDALONE_WANTED_CONFIG}")
    _standalone_repro_compile()
    print("Kernel compiled successfully; the original compiler failure did not reproduce.")


if __name__ == "__main__":
    _standalone_repro_main()
'''


def standalone_launch_footer(
    input_bytes=None,
    runtime_blocks=(),
    *,
    input_payload=None,
    direct_metadata=None,
):
    if input_bytes is None and input_payload is None:
        raise RuntimeError("launch export requires a captured input payload")
    payload_section = _standalone_payload_section(input_bytes, input_payload)
    template = r'''

_STANDALONE_RUNTIME_BLOCKS = __RUNTIME_BLOCKS__
__PAYLOAD_SECTION__
__MATERIALIZER__


def _standalone_repro_prepare_inputs():
    import os
    import torch

    payload = _standalone_repro_load_payload()
    capture_mode = payload.get("capture_mode", "legacy-values")
    map_location = os.environ.get("TRITON_REPRO_MAP_LOCATION") or None
    synthetic_seed = None
    if capture_mode == "metadata":
        synthetic_seed = int(os.environ.get("TRITON_REPRO_SEED", "0"))
        torch.manual_seed(synthetic_seed)
        args = tuple(_standalone_repro_materialize(payload["args"], map_location))
        kwargs = dict(_standalone_repro_materialize(payload["kwargs"], map_location))
        encoded_prefix = payload.get("launch_prefix")
        if encoded_prefix is not None:
            launch_prefix = tuple(
                _standalone_repro_materialize(encoded_prefix, map_location)
            )
        else:
            encoded_launch_args = payload.get("launch_args")
            if encoded_launch_args is not None:
                legacy_launch_args = tuple(
                    _standalone_repro_materialize(
                        encoded_launch_args,
                        map_location,
                    )
                )
                launch_prefix = _standalone_repro_legacy_prefix(
                    args,
                    legacy_launch_args,
                )
            else:
                launch_prefix = args
    else:
        args = tuple(payload["args"])
        kwargs = dict(payload["kwargs"])
        launch_prefix = payload.get("launch_prefix")
        if launch_prefix is None:
            legacy_launch_args = payload.get("launch_args")
            launch_prefix = _standalone_repro_legacy_prefix(
                args,
                tuple(legacy_launch_args) if legacy_launch_args is not None else None,
            )
        else:
            launch_prefix = tuple(launch_prefix)
    launch_args = (*tuple(launch_prefix), *_STANDALONE_RUNTIME_BLOCKS)
    return args, launch_args, kwargs, capture_mode, synthetic_seed


def _standalone_repro_main():
    import time
    import torch

    if hasattr(torch, "npu") and hasattr(torch.npu, "set_device"):
        torch.npu.set_device(_STANDALONE_DEVICE_INDEX)
    _args, raw_launch_args, kwargs, capture_mode, synthetic_seed = (
        _standalone_repro_prepare_inputs()
    )
    print(f"Compiling fixed config and launching once: {_STANDALONE_FN_NAME}")
    print(f"Config: {_STANDALONE_WANTED_CONFIG}")
    print(f"Runtime blocks: {_STANDALONE_RUNTIME_BLOCKS}")
    print(f"Input capture mode: {capture_mode}")
    if synthetic_seed is not None:
        print(f"Synthetic input seed: {synthetic_seed}")

    binary = _standalone_repro_compile()
    kernel_args, _launch_values = _standalone_repro_split_launch_args(
        raw_launch_args
    )
    grid = _standalone_repro_grid(raw_launch_args)
    print(f"Grid: {grid}")
    torch.npu.synchronize()
    start = time.perf_counter()
    # ``CompiledKernel.__getitem__`` accepts positional kernel arguments and
    # an optional raw ``stream``.  Inductor's generated launcher normally has
    # already consumed any other kwargs; fail explicitly if an old capture
    # contains one instead of silently passing it to the wrong ABI.
    launch_kwargs = dict(kwargs)
    stream = launch_kwargs.pop("stream", None)
    if stream is None:
        stream = _standalone_repro_current_stream()
    if launch_kwargs:
        raise RuntimeError(
            "Direct Triton runner cannot consume captured launcher kwargs: "
            f"{sorted(launch_kwargs)}"
        )
    binary[grid](*kernel_args, stream=stream)
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"Kernel prerun succeeded in {elapsed_ms:.3f} ms")


if __name__ == "__main__":
    _standalone_repro_main()
'''
    return (
        template.replace("__RUNTIME_BLOCKS__", repr(tuple(runtime_blocks)))
        .replace("__PAYLOAD_SECTION__", payload_section)
        .replace("__MATERIALIZER__", _standalone_materializer_template())
    )


def render_standalone_reproducer(
    source_text,
    fn_name,
    wanted_config,
    input_bytes=None,
    runtime_blocks=(),
    *,
    input_payload=None,
    direct_metadata=None,
):
    mode = "compile and launch" if input_bytes is not None or input_payload is not None else "compile"
    header = f"""# Standalone single-kernel {mode} reproducer.
# Extracted by repro_compile.py for {fn_name}.
# It calls Triton's compiler/runtime directly with one fixed candidate config.
# It does not instantiate Inductor's autotuner or rerun Dynamo/AOTAutograd.

"""
    source = standalone_source_without_inductor(source_text, fn_name)
    footer = standalone_common_footer(fn_name, wanted_config, direct_metadata)
    if input_bytes is None and input_payload is None:
        footer += standalone_compile_footer()
    else:
        footer += standalone_launch_footer(
            input_bytes,
            runtime_blocks,
            input_payload=input_payload,
            direct_metadata=direct_metadata,
        )
    return header + source.rstrip() + footer


def _payload_for_export(input_path):
    payload = torch_load(input_path)
    mode = payload.get("capture_mode", "legacy-values")
    if mode == "metadata":
        if _payload_contains_tensor(payload):
            raise RuntimeError(
                f"Metadata capture {input_path} unexpectedly contains Tensor values"
            )
        return None, payload
    return input_path.read_bytes(), None


def _payload_contains_tensor(value):
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(
            _payload_contains_tensor(key) or _payload_contains_tensor(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_payload_contains_tensor(item) for item in value)
    return False


def _direct_metadata_usable(metadata, source_text, fn_name):
    if not isinstance(metadata, dict):
        return False
    signature = metadata.get("signature")
    target = metadata.get("target") or {}
    if not isinstance(signature, dict) or not signature:
        return False
    if not isinstance(target, dict) or not target.get("arch"):
        return False
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == fn_name
        for node in tree.body
    )


def export_standalone_reproducer(
    output_path,
    source_path,
    fn_name,
    wanted_config,
    input_path=None,
    runtime_blocks=(),
    record=None,
):
    output_path = output_path.expanduser().resolve()
    protected_paths = {source_path}
    if input_path is not None:
        protected_paths.add(input_path)
    if output_path in protected_paths:
        raise RuntimeError("Refusing to overwrite a captured artifact")

    source_text = source_path.read_text(encoding="utf-8")
    direct_metadata = (record or {}).get("direct_compile")
    if isinstance(direct_metadata, dict):
        # New captures use tagged JSON objects for tuple-key mappings (Triton
        # attrs commonly use ``(arg_index,)`` keys).  Decode the journal before
        # looking at nested fields so both the direct metadata and the
        # generated Python literal receive the original key types.
        direct_metadata = _restore_json_metadata(direct_metadata)
    if not isinstance(direct_metadata, dict):
        try:
            direct_metadata = direct_compile_metadata_from_source(
                source_text,
                fn_name,
                wanted_config,
            )
        except RuntimeError:
            direct_metadata = None

    if not _direct_metadata_usable(direct_metadata, source_text, fn_name):
        raise RuntimeError(
            f"Direct export lacks usable Triton metadata for {fn_name!r}; "
            "recapture with the current trace_compile.py or provide an archived "
            "Triton source containing the kernel function and device metadata"
        )

    input_bytes = None
    input_payload = None
    if input_path is not None:
        input_bytes, input_payload = _payload_for_export(input_path)

    standalone = render_standalone_reproducer(
        source_text,
        fn_name,
        wanted_config,
        input_bytes=input_bytes,
        input_payload=input_payload,
        runtime_blocks=runtime_blocks,
        direct_metadata=direct_metadata,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(standalone, encoding="utf-8")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay or export one captured Triton attempt."
    )
    parser.add_argument(
        "capture",
        type=Path,
        help="capture directory or its journal.jsonl file",
    )
    parser.add_argument("config_id", type=int)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--export",
        type=Path,
        metavar="OUTPUT.py",
        help="write a direct fixed-config compile-only reproducer",
    )
    action.add_argument(
        "--launch",
        action="store_true",
        help="compile and launch the captured prerun candidate once",
    )
    action.add_argument(
        "--export-launch",
        type=Path,
        metavar="OUTPUT.py",
        help="write a direct fixed-config compile-and-launch reproducer",
    )
    parser.add_argument(
        "--map-location",
        help="torch.load map_location for direct --launch replay",
    )
    return parser.parse_args()


def resolve_journal_path(capture_path):
    capture_path = capture_path.expanduser().resolve()
    if capture_path.is_dir():
        return capture_path / "journal.jsonl"
    return capture_path


def require_prerun_input(record, journal_path):
    if not str(record.get("event", "")).startswith("PRERUN_"):
        raise RuntimeError(
            "Launch replay requires a PRERUN_* journal ID, not a compile ID"
        )
    input_path = resolve_record_artifact(
        journal_path,
        record.get("input_artifact"),
        "prerun_inputs",
    )
    if input_path is None:
        detail = record.get("input_capture_error") or "no input artifact recorded"
        raise RuntimeError(
            f"Selected prerun record has no replayable inputs: {detail}. "
            "Capture it again with TRITON_TRACE_PRERUN=1 and "
            "TRITON_TRACE_PRERUN_INPUTS=values or metadata."
        )
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Captured prerun inputs no longer exist: {input_path}. "
            "Capture again and keep the entire capture directory."
        )
    return input_path


def main():
    args = parse_args()
    journal_path = resolve_journal_path(args.capture)
    config_id = args.config_id
    record = load_record(journal_path, config_id)

    source_path = resolve_record_artifact(
        journal_path,
        record.get("source"),
        "kernel_sources",
    )
    if source_path is None:
        raise RuntimeError(
            "The selected record has no generated source path. "
            "Capture it again with the updated trace_compile.py."
        )
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Generated kernel source no longer exists: {source_path}. "
            "Capture again and keep the entire capture directory."
        )

    fn_name = record.get("fn_name") or record.get("kernel")
    wanted_config = record.get("config")
    if wanted_config is None:
        raise RuntimeError(f"Record id={config_id} has no config")

    if args.export_launch is not None:
        input_path = require_prerun_input(record, journal_path)
        output_path = export_standalone_reproducer(
            args.export_launch,
            source_path,
            fn_name,
            wanted_config,
            input_path=input_path,
            runtime_blocks=record.get("runtime_blocks", ()),
            record=record,
        )
        print(f"Standalone launch reproducer written to: {output_path}")
        print(f"Run it with: python {output_path}")
        return

    if args.export is not None:
        output_path = export_standalone_reproducer(
            args.export,
            source_path,
            fn_name,
            wanted_config,
            record=record,
        )
        print(f"Standalone compile reproducer written to: {output_path}")
        print(f"Run it with: python {output_path}")
        return

    module = load_generated_module(source_path, config_id)
    tuner = find_tuner(module, fn_name)
    config = find_config(tuner, wanted_config)

    if args.launch:
        input_path = require_prerun_input(record, journal_path)
        payload = load_prerun_payload(
            input_path,
            tuner,
            record.get("runtime_blocks", ()),
            map_location=args.map_location,
        )
        print(f"Compiling and launching once: {fn_name}")
        print(f"Source: {source_path}")
        print(f"Config: {config_to_dict(config)}")
        print(f"Runtime blocks: {payload.get('runtime_blocks', ())}")
        print(f"Input capture mode: {payload.get('capture_mode')}")
        if payload.get("synthetic_seed") is not None:
            print(f"Synthetic input seed: {payload['synthetic_seed']}")
        elapsed_ms = run_captured_prerun(tuner, config, payload)
        print(f"Kernel prerun succeeded in {elapsed_ms:.3f} ms")
        return

    print(f"Compiling only: {fn_name}")
    print(f"Source: {source_path}")
    print(f"Config: {config_to_dict(config)}")
    tuner._precompile_config(config)
    print(
        "Kernel compiled successfully; "
        "the original compiler failure did not reproduce."
    )


if __name__ == "__main__":
    main()

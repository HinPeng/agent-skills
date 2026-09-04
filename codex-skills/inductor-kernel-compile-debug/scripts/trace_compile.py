"""Capture compile or prerun attempts for one torch_npu Inductor kernel family.

Set TRITON_TRACE_CAPTURE_DIR and TRITON_TRACE_KERNEL_PATTERN, then run:
    python trace_compile.py TARGET.py [target arguments ...]

Set TRITON_TRACE_PRERUN=1 to retain one input snapshot shared by the matched
prerun candidates from the same kernel invocation.
"""

import atexit
import hashlib
import itertools
import json
import os
import re
import runpy
import sys
import threading
import time
import traceback
from collections.abc import Mapping

import torch
from torch._inductor.runtime.triton_heuristics import config_to_dict
import torch_npu._inductor.runtime.triton_heuristics as npu_triton_heuristics
from torch_npu._inductor.runtime.triton_heuristics import NPUCachingAutotuner


# Only this root directory is configurable.  Keeping the journal and copied
# kernel sources together makes the capture portable and prevents the source
# paths from disappearing with Inductor's process-owned /tmp directory.
CAPTURE_DIR = os.path.abspath(
    os.path.expanduser(
        os.environ.get(
            "TRITON_TRACE_CAPTURE_DIR",
            "/tmp/triton_compile_capture",
        )
    )
)
JOURNAL = os.path.join(CAPTURE_DIR, "journal.jsonl")
ARTIFACT_DIR = os.path.join(CAPTURE_DIR, "kernel_sources")
PRERUN_INPUT_DIR = os.path.join(CAPTURE_DIR, "prerun_inputs")
KERNEL_PATTERN = os.environ.get("TRITON_TRACE_KERNEL_PATTERN", "").strip()
if not KERNEL_PATTERN:
    raise SystemExit(
        "TRITON_TRACE_KERNEL_PATTERN is required; set it to a distinctive "
        "regular expression matching the failing kernel"
    )
SERIALIZE_MATCHED_KERNEL = os.environ.get(
    "TRITON_TRACE_SERIALIZE",
    "0",
).strip().lower() in ("1", "true", "yes", "on")
TRACE_PRERUN = os.environ.get(
    "TRITON_TRACE_PRERUN",
    "0",
).strip().lower() in ("1", "true", "yes", "on")


def parse_prerun_input_mode(value):
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on", "values", "full", "shared"):
        return "values"
    if value in ("metadata", "meta", "spec"):
        return "metadata"
    if value in ("0", "false", "no", "off", "none"):
        return "none"
    raise SystemExit(
        "TRITON_TRACE_PRERUN_INPUTS must be one of: "
        "values/1, metadata, or none/0"
    )


PRERUN_INPUT_MODE = (
    parse_prerun_input_mode(
        os.environ.get("TRITON_TRACE_PRERUN_INPUTS", "values")
    )
    if TRACE_PRERUN
    else "none"
)

try:
    kernel_regex = re.compile(KERNEL_PATTERN) if KERNEL_PATTERN else None
except re.error as error:
    raise SystemExit(
        f"invalid TRITON_TRACE_KERNEL_PATTERN={KERNEL_PATTERN!r}: {error}"
    ) from error

counter = itertools.count()
input_group_counter = itertools.count()
matched_kernel_lock = threading.Lock()
source_archive_lock = threading.Lock()
journal_lock = threading.Lock()
input_group_lock = threading.Lock()
prerun_thread_state = threading.local()
active_input_groups = {}
archived_sources = {}

# JSON object keys must be strings, while Triton ``attrs`` descriptors use
# tuple keys such as ``(0,)``.  Keep those keys losslessly in the journal with
# a small tagged representation.  The exporter understands this representation
# and also accepts the plain dictionaries emitted by older captures.
JSON_MAPPING_ITEMS = "__inductor_trace_mapping_items__"
JSON_TUPLE = "__inductor_trace_tuple__"
JSON_REPR = "__inductor_trace_repr__"
original_compile = NPUCachingAutotuner._precompile_config
original_benchmark_candidates = getattr(
    NPUCachingAutotuner,
    "_benchmark_candidate_entries",
    None,
)
original_measure_prerun = getattr(
    npu_triton_heuristics,
    "_measure_prerun_ms",
    None,
)


def persist(record):
    record["time"] = time.time()
    with journal_lock:
        os.makedirs(os.path.dirname(os.path.abspath(JOURNAL)), exist_ok=True)
        with open(JOURNAL, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, default=repr) + "\n")
            file.flush()
            os.fsync(file.fileno())


def kernel_identity(self):
    fn_name = self.get_fn_name()
    inductor_meta = getattr(self, "inductor_meta", None) or {}
    kernel_name = inductor_meta.get("kernel_name", fn_name)
    return fn_name, kernel_name


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return value[:120] or "triton_kernel"


def should_trace_kernel(fn_name, kernel_name):
    if kernel_regex is None:
        return True
    return any(
        kernel_regex.search(candidate)
        for candidate in (fn_name, kernel_name)
        if candidate
    )


def archive_generated_source(source, fn_name):
    """Copy a generated cache module out of its process-owned temp directory."""
    if not source:
        return None, "generated source path is unavailable"

    source_path = os.path.abspath(os.fspath(source))
    with source_archive_lock:
        archived = archived_sources.get(source_path)
        if archived is not None:
            return archived, None

        if not os.path.isfile(source_path):
            return None, f"generated source does not exist: {source_path}"

        try:
            with open(source_path, "rb") as source_file:
                source_bytes = source_file.read()
            digest = hashlib.sha256(source_bytes).hexdigest()[:16]
            os.makedirs(ARTIFACT_DIR, exist_ok=True)
            archived = os.path.join(
                os.path.abspath(ARTIFACT_DIR),
                f"{safe_name(fn_name)}_{digest}.py",
            )
            if not os.path.exists(archived):
                temporary_copy = (
                    f"{archived}.tmp-{os.getpid()}-{threading.get_ident()}"
                )
                with open(temporary_copy, "wb") as output_file:
                    output_file.write(source_bytes)
                    output_file.flush()
                    os.fsync(output_file.fileno())
                os.replace(temporary_copy, archived)
            archived_sources[source_path] = archived
            return archived, None
        except Exception as error:
            # Source capture must never mask the original compiler failure.
            return None, f"{type(error).__name__}: {error}"


def describe_value(value):
    try:
        if isinstance(value, torch.Tensor):
            return {
                "type": "Tensor",
                "dtype": str(value.dtype),
                "device": str(value.device),
                "shape": tuple(value.shape),
                "stride": tuple(value.stride()),
                "storage_offset": value.storage_offset(),
                "requires_grad": value.requires_grad,
            }
        value_repr = repr(value)
    except Exception as error:
        value_repr = f"<repr failed: {type(error).__name__}: {error}>"
    return {
        "type": type(value).__name__,
        "repr": value_repr[:500],
    }


def describe_arguments(values):
    return [describe_value(value) for value in values]


def _json_safe_key(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return {
            JSON_TUPLE: [json_safe(item) for item in value],
        }
    return {JSON_REPR: repr(value)}


def json_safe(value):
    """Convert runtime metadata to a small JSON-compatible value.

    The generated Triton module contains a few objects which are useful while
    compiling (``DeviceProperties``, torch dtypes, sets, and named tuples) but
    cannot be reconstructed from the journal's generic ``repr`` fallback.  The
    direct standalone exporter only needs a stable subset of this information,
    so normalize it explicitly at capture time.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Mapping):
        items = list(value.items())
        if all(isinstance(key, str) for key, _ in items):
            return {key: json_safe(item) for key, item in items}
        return {
            JSON_MAPPING_ITEMS: [
                [_json_safe_key(key), json_safe(item)]
                for key, item in items
            ]
        }
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        values = [json_safe(item) for item in value]
        return sorted(values, key=repr)
    if hasattr(value, "_asdict") and callable(value._asdict):
        return json_safe(value._asdict())
    # DeviceProperties and a few vendor descriptors expose their state through
    # dataclass fields rather than ``_asdict``.
    fields = getattr(value, "__dataclass_fields__", None)
    if fields:
        return {
            name: json_safe(getattr(value, name, None))
            for name in fields
        }
    return repr(value)


def _device_property(device, name, default=None):
    if isinstance(device, Mapping):
        return device.get(name, default)
    return getattr(device, name, default)


def _effective_compile_options(
    tuner,
    config,
    config_dict,
    triton_meta,
    inductor_meta,
):
    """Reproduce the NPU autotuner's effective Triton option dictionary.

    ``Config.kwargs`` contains both backend options and values used only by
    Inductor's grid planner (for example ``split_axis``).  Passing the latter
    through to the standalone compiler is harmless for some Triton releases,
    but it makes the exported program differ from the config that actually
    failed.  Newer ``NPUCachingAutotuner`` versions expose the same filtering
    helper used by ``_precompile_config``; use it when available and retain a
    conservative best-effort fallback for older versions and test doubles.
    """
    cfg_kwargs = getattr(config, "kwargs", None)
    if not isinstance(cfg_kwargs, Mapping):
        cfg_kwargs = {}
    else:
        # ``triton.Config`` normally omits ``None`` values, but a few vendor
        # wrappers leave optional keys in ``kwargs``.  The autotuner's
        # ``or default`` handling treats those as absent; preserve that
        # behavior in the captured option snapshot.
        cfg_kwargs = {
            key: value for key, value in cfg_kwargs.items() if value is not None
        }

    compile_mode = (
        cfg_kwargs.get("compile_mode") or "simd_simt_template"
    )
    effective = {
        "num_warps": getattr(
            config,
            "num_warps",
            config_dict.get("num_warps", 4),
        ),
        "num_stages": getattr(
            config,
            "num_stages",
            config_dict.get("num_stages", 3),
        ),
        "debug": (
            os.environ.get("INDUCTOR_ASCEND_DEBUG", "false").lower()
            in ("true", "1")
            and inductor_meta.get("assert_indirect_indexing", True)
            and not inductor_meta.get("is_hip", False)
        ),
        "compile_mode": compile_mode,
    }

    raw_npu_options = triton_meta.get("npu_compile_options", {}) or {}
    if not isinstance(raw_npu_options, Mapping):
        raw_npu_options = {}
    else:
        raw_npu_options = {
            key: value
            for key, value in raw_npu_options.items()
            if value is not None
        }

    parse_options = getattr(tuner, "parse_triton_ascend_options", None)
    if callable(parse_options):
        try:
            parsed = parse_options(dict(raw_npu_options), effective)
            if isinstance(parsed, Mapping):
                effective = dict(parsed)
            parsed = parse_options(cfg_kwargs, effective)
            if isinstance(parsed, Mapping):
                effective = dict(parsed)
        except Exception:
            # Metadata capture must not mask the compiler/prerun failure.  A
            # best-effort unfiltered fallback remains useful on older vendor
            # builds where the helper is present but not importable in the
            # tracing process.
            effective.update(raw_npu_options)
            effective.update(cfg_kwargs)
    else:
        # Older torch_npu releases do not expose the filtering helper.  Keep
        # the old behavior; AscendBackend.parse_options filters unknown keys
        # again when the exported program calls triton.compile.
        effective.update(raw_npu_options)
        effective.update(cfg_kwargs)

    if (
        inductor_meta.get("enable_auto_blockify", False)
        or inductor_meta.get("requires_no_linear_block_remap") is True
    ):
        if callable(parse_options):
            try:
                parsed = parse_options({"enable_auto_blockify": True}, effective)
                if isinstance(parsed, Mapping):
                    effective = dict(parsed)
            except Exception:
                effective["enable_auto_blockify"] = True
        else:
            effective["enable_auto_blockify"] = True

    # The backend, rather than the candidate config, owns the pure-SIMT stack
    # limit.  Capture the same value that _precompile_config uses whenever the
    # vendor config module is available.
    compile_mode = effective.get("compile_mode", compile_mode)
    if compile_mode == "simt_only":
        stack_limit = None
        try:
            import torch_npu._inductor.config as npu_config

            stack_limit = getattr(npu_config, "simt_default_warp_stacksize", None)
        except Exception:
            stack_limit = getattr(tuner, "simt_stack_limit", None)
        if stack_limit is not None:
            effective["simt_stack_limit"] = stack_limit

    return effective


def direct_compile_metadata(tuner, config):
    """Snapshot the inputs needed by a direct Triton compile.

    This deliberately records compile metadata rather than the autotuner
    object.  An exported reproducer can therefore instantiate ``ASTSource``
    and call ``triton.compile`` without importing or constructing Inductor's
    heuristics/autotuner machinery.
    """
    triton_meta = getattr(tuner, "triton_meta", None) or {}
    inductor_meta = getattr(tuner, "inductor_meta", None) or {}
    if not isinstance(triton_meta, Mapping):
        triton_meta = {}
    if not isinstance(inductor_meta, Mapping):
        inductor_meta = {}
    config_dict = {}
    try:
        config_dict = config_to_dict(config)
    except Exception:
        config_dict = {
            **(getattr(config, "kwargs", {}) or {}),
            "num_warps": getattr(config, "num_warps", 4),
            "num_stages": getattr(config, "num_stages", 3),
        }

    fn = getattr(tuner, "fn", None)
    arg_names = list(
        getattr(fn, "arg_names", ()) or getattr(tuner, "arg_names", ()) or ()
    )
    raw_constexprs = tuple(getattr(fn, "constexprs", ()) or ())
    constexpr_names = []
    for constexpr in raw_constexprs:
        if isinstance(constexpr, int):
            if 0 <= constexpr < len(arg_names):
                constexpr_names.append(arg_names[constexpr])
        elif isinstance(constexpr, str) and constexpr in arg_names:
            constexpr_names.append(constexpr)

    raw_constants = triton_meta.get("constants", {}) or {}
    constants = dict(raw_constants) if isinstance(raw_constants, Mapping) else {}
    # A few older Triton builds journal constants by positional index.  The
    # direct ASTSource API accepts names (or index tuples), so prefer names when
    # the function signature is available.
    for key in tuple(constants):
        if isinstance(key, int) and 0 <= key < len(arg_names):
            constants[arg_names[key]] = constants.pop(key)
    cfg_kwargs = getattr(config, "kwargs", None)
    if not isinstance(cfg_kwargs, Mapping):
        cfg_kwargs = config_dict
    for name in constexpr_names:
        if name in cfg_kwargs:
            constants[name] = cfg_kwargs[name]
    # ``num_warps`` and ``num_stages`` can be represented as implicit
    # constexprs on older Triton versions.  Mirror the autotuner's fallback
    # insertion so ASTSource sees the same constants as the failed compile.
    for name in constexpr_names:
        if name not in constants and name in ("num_warps", "num_stages"):
            if name in config_dict:
                constants[name] = config_dict[name]

    # ``device_props`` is the object actually used by NPUCachingAutotuner;
    # fall back to the serialized triton_meta for older versions.
    device = getattr(tuner, "device_props", None) or triton_meta.get("device")
    device_type = (
        _device_property(device, "type")
        or triton_meta.get("device_type")
        or "npu"
    )
    device_index = _device_property(device, "index", triton_meta.get("device_index", 0))
    cc = _device_property(device, "cc", triton_meta.get("cc"))
    warp_size = _device_property(device, "warp_size", None) or 32

    # ``configs`` is the attrs descriptor used by newer Triton versions.  It
    # is commonly absent on the NPU path; an empty attrs dict is equivalent.
    attrs = triton_meta.get("configs")
    if isinstance(attrs, (tuple, list)):
        attrs = attrs[0] if attrs else {}
    elif attrs is None:
        attrs = {}
    if isinstance(attrs, Mapping) and not isinstance(attrs, dict):
        attrs = dict(attrs)
    if not isinstance(attrs, dict):
        # AttrsDescriptor is version-specific, but its ``to_dict`` spelling
        # is stable across Triton-Ascend releases and can be reconstructed by
        # the direct exporter.  Fall back to an empty mapping only when the
        # installed descriptor does not expose that method.
        to_dict = getattr(attrs, "to_dict", None)
        if callable(to_dict):
            try:
                attrs = to_dict()
            except Exception:
                attrs = {}
        else:
            # Some legacy Triton builds expose the descriptor as a namedtuple
            # rather than an object with ``to_dict``.
            asdict = getattr(attrs, "_asdict", None)
            if callable(asdict):
                try:
                    attrs = asdict()
                except Exception:
                    attrs = {}
            else:
                attrs = {}

    effective_options = _effective_compile_options(
        tuner,
        config,
        config_dict,
        triton_meta,
        inductor_meta,
    )

    # Grid construction only needs these fields.  Keeping the subset small
    # also avoids leaking transient graph/profiling paths into the export.
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
        key: inductor_meta[key]
        for key in grid_keys
        if key in inductor_meta
    }

    launch_arg_names = [
        name for name in arg_names if name not in constexpr_names
    ]
    runtime_block_names = tuple(
        grid_meta.get("runtime_block_append_order")
        or grid_meta.get("runtime_block_arg_names")
        or ()
    )
    for name in runtime_block_names:
        if name not in launch_arg_names:
            launch_arg_names.append(name)

    return {
        "version": 1,
        "signature": json_safe(triton_meta.get("signature", {})),
        "constants": json_safe(constants),
        "attrs": json_safe(attrs),
        "target": {
            "backend": json_safe(device_type),
            "arch": json_safe(cc),
            "warp_size": json_safe(warp_size),
        },
        "device_index": json_safe(device_index),
        "options": json_safe(effective_options),
        "inductor_meta": json_safe(grid_meta),
        "arg_names": json_safe(arg_names),
        "constexpr_names": json_safe(constexpr_names),
        "launch_arg_names": json_safe(launch_arg_names),
        "runtime_block_arg_names": json_safe(runtime_block_names),
        "base_launch_arg_names": json_safe(
            [name for name in launch_arg_names if name not in runtime_block_names]
        ),
        "extra_launcher_arg_names": json_safe(
            grid_meta.get("extra_launcher_args", ()) or ()
        ),
    }


def safe_direct_compile_metadata(tuner, config):
    """Best-effort metadata capture that can never mask the real failure."""
    try:
        return direct_compile_metadata(tuner, config), None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def closure_bindings(function):
    closure = function.__closure__ or ()
    return {
        name: cell.cell_contents
        for name, cell in zip(function.__code__.co_freevars, closure)
    }


def default_bindings(function):
    defaults = function.__defaults__ or ()
    if not defaults:
        return {}
    arg_count = function.__code__.co_argcount
    arg_names = function.__code__.co_varnames[:arg_count]
    return dict(zip(arg_names[-len(defaults):], defaults))


def extract_prerun_context(kernel_call_fn):
    """Read the stable closure contract of NPUCachingAutotuner's prerun."""
    try:
        closed = closure_bindings(kernel_call_fn)
        defaults = default_bindings(kernel_call_fn)
    except Exception:
        return None

    tuner = closed.get("self")
    launcher = defaults.get("launcher")
    launch_args = defaults.get("launch_args")
    args = closed.get("args")
    kwargs = closed.get("kwargs")
    if not all(
        (
            tuner is not None,
            launcher is not None,
            launch_args is not None,
            args is not None,
            kwargs is not None,
        )
    ):
        return None
    if not callable(getattr(tuner, "get_fn_name", None)):
        return None
    if not callable(launcher):
        return None

    args_object = args
    kwargs_object = kwargs
    args = tuple(args_object)
    launch_args = tuple(launch_args)
    runtime_blocks = launch_args[len(args):]
    return {
        "tuner": tuner,
        "launcher": launcher,
        "args_object": args_object,
        "kwargs_object": kwargs_object,
        "args": args,
        "launch_args": launch_args,
        "kwargs": dict(kwargs_object),
        "runtime_blocks": tuple(runtime_blocks),
    }


def find_candidate(tuner, launcher, runtime_blocks):
    for entry in getattr(tuner, "compiled_candidate_entries", ()) or ():
        if entry.get("launcher") is not launcher:
            continue
        candidate = entry.get("candidate") or {}
        candidate_runtime_blocks = tuple(
            value for _, value in candidate.get("runtime_blocks", ())
        )
        if candidate_runtime_blocks == runtime_blocks:
            return candidate
    return {}


def planned_prerun_input_path(fn_name, group_id):
    filename = (
        f".{safe_name(fn_name)}_{os.getpid()}_"
        f"{threading.get_ident()}_{group_id}.shared.pending.pt"
    )
    return os.path.join(os.path.abspath(PRERUN_INPUT_DIR), filename)


def encode_input_metadata(value):
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise TypeError(
                f"metadata capture only supports strided tensors, got {value.layout}"
            )
        return {
            "kind": "tensor",
            "dtype": str(value.dtype),
            "device": str(value.device),
            "shape": tuple(int(size) for size in value.shape),
            "stride": tuple(int(stride) for stride in value.stride()),
            "storage_offset": int(value.storage_offset()),
            "requires_grad": bool(value.requires_grad),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": "literal", "value": value}
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [encode_input_metadata(item) for item in value],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [encode_input_metadata(item) for item in value],
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": [
                (encode_input_metadata(key), encode_input_metadata(item))
                for key, item in value.items()
            ],
        }
    raise TypeError(
        f"metadata capture cannot encode input type {type(value).__name__}"
    )


def build_input_payload(context):
    # ``launch_args`` is candidate-specific: the autotuner appends a different
    # runtime-block tuple for each candidate.  Persist only the stable input
    # prefix in the shared container.  The journal keeps each candidate's
    # runtime blocks and the replay/export path appends the selected tuple.
    # If a backend inserts a non-argument prefix, retain that prefix explicitly
    # (it is still shared by the candidate batch).
    args = tuple(context["args"])
    launch_args = tuple(context["launch_args"])
    runtime_blocks = tuple(context.get("runtime_blocks", ()))
    prefix_length = len(launch_args) - len(runtime_blocks)
    if prefix_length < 0:
        prefix_length = len(launch_args)
    launch_prefix = launch_args[:prefix_length]
    has_distinct_prefix = len(launch_prefix) != len(args) or any(
        left is not right for left, right in zip(launch_prefix, args)
    )

    if PRERUN_INPUT_MODE == "values":
        payload = {
            "format_version": 3,
            "capture_mode": "values",
            "args": args,
            "kwargs": context["kwargs"],
        }
        if has_distinct_prefix:
            payload["launch_prefix"] = launch_prefix
        return payload
    if PRERUN_INPUT_MODE == "metadata":
        payload = {
            "format_version": 3,
            "capture_mode": "metadata",
            "args": encode_input_metadata(args),
            "kwargs": encode_input_metadata(context["kwargs"]),
        }
        if has_distinct_prefix:
            payload["launch_prefix"] = encode_input_metadata(launch_prefix)
        return payload
    raise AssertionError(f"unexpected input mode: {PRERUN_INPUT_MODE}")


def capture_prerun_inputs(path, context):
    """Persist one shared input snapshot for a candidate batch."""
    if PRERUN_INPUT_MODE == "none":
        return None, "input capture disabled by TRITON_TRACE_PRERUN_INPUTS"

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as output_file:
            torch.save(build_input_payload(context), output_file)
            output_file.flush()
            os.fsync(output_file.fileno())
        return path, None
    except Exception as error:
        try:
            os.unlink(path)
        except OSError:
            pass
        return None, f"{type(error).__name__}: {error}"


def discard_input_artifact(path):
    if not path:
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except Exception as error:
        print(
            f"[trace_compile] could not remove shared prerun snapshot "
            f"{path}: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )


def finish_input_group(group):
    if group is None:
        return
    if not group["failed"]:
        discard_input_artifact(group["path"])
    with input_group_lock:
        active_input_groups.pop(group["id"], None)
    group["tuner"] = None
    group["args_object"] = None
    group["kwargs_object"] = None


def cleanup_active_input_groups():
    with input_group_lock:
        groups = list(active_input_groups.values())
        active_input_groups.clear()
    for group in groups:
        if not group["failed"]:
            discard_input_artifact(group["path"])


atexit.register(cleanup_active_input_groups)


def get_input_group(context, fn_name):
    group = getattr(prerun_thread_state, "input_group", None)
    batch_token = getattr(prerun_thread_state, "batch_token", None)
    if batch_token is not None:
        same_group = (
            group is not None
            and group["batch_token"] is batch_token
            and group["tuner"] is context["tuner"]
        )
    else:
        # Compatibility fallback for older torch_npu versions without the
        # candidate-batch method wrapper.
        same_group = (
            group is not None
            and group["tuner"] is context["tuner"]
            and group["args_object"] is context["args_object"]
            and group["kwargs_object"] is context["kwargs_object"]
        )
    if same_group:
        return group, True

    finish_input_group(group)
    group_id = next(input_group_counter)
    group = {
        "id": group_id,
        "batch_token": batch_token,
        "tuner": context["tuner"],
        "args_object": context["args_object"],
        "kwargs_object": context["kwargs_object"],
        "path": (
            planned_prerun_input_path(fn_name, group_id)
            if PRERUN_INPUT_MODE != "none"
            else None
        ),
        "capture_error": None,
        "captured": False,
        "failed": False,
    }
    prerun_thread_state.input_group = group
    with input_group_lock:
        active_input_groups[group_id] = group
    return group, False


def ensure_input_group_captured(group, context):
    if group["captured"]:
        return
    captured_path, capture_error = capture_prerun_inputs(
        group["path"],
        context,
    )
    group["path"] = captured_path
    group["capture_error"] = capture_error
    group["captured"] = True


def retain_failed_input_group(group, fn_name):
    group["failed"] = True
    pending_path = group["path"]
    if not pending_path or not os.path.isfile(pending_path):
        return pending_path, None
    if not pending_path.endswith(".pending.pt"):
        return pending_path, None
    retained_path = os.path.join(
        os.path.abspath(PRERUN_INPUT_DIR),
        f"{safe_name(fn_name)}_{os.getpid()}_{group['id']}.shared.pt",
    )
    try:
        os.replace(pending_path, retained_path)
        group["path"] = retained_path
        return retained_path, None
    except Exception as error:
        return pending_path, f"{type(error).__name__}: {error}"


def benchmark_candidates_with_trace(self, *args, **kwargs):
    previous_token = getattr(prerun_thread_state, "batch_token", None)
    previous_group = getattr(prerun_thread_state, "input_group", None)
    batch_token = object()
    prerun_thread_state.batch_token = batch_token
    prerun_thread_state.input_group = None
    try:
        return original_benchmark_candidates(self, *args, **kwargs)
    finally:
        group = getattr(prerun_thread_state, "input_group", None)
        if group is not None and group["batch_token"] is batch_token:
            finish_input_group(group)
        prerun_thread_state.batch_token = previous_token
        prerun_thread_state.input_group = previous_group


def measure_prerun_with_trace(kernel_call_fn):
    context = extract_prerun_context(kernel_call_fn)
    if context is None:
        return original_measure_prerun(kernel_call_fn)

    tuner = context["tuner"]
    fn_name, kernel_name = kernel_identity(tuner)
    if not should_trace_kernel(fn_name, kernel_name):
        return original_measure_prerun(kernel_call_fn)

    config_id = next(counter)
    launcher = context["launcher"]
    candidate = find_candidate(
        tuner,
        launcher,
        context["runtime_blocks"],
    )
    source = getattr(tuner, "filename", None)
    if not source:
        python_fn = getattr(getattr(tuner, "fn", None), "fn", None)
        code = getattr(python_fn, "__code__", None)
        source = getattr(code, "co_filename", None)
    original_source = str(source) if source else None
    archived_source, source_archive_error = archive_generated_source(
        source,
        fn_name,
    )
    input_group, input_reused = get_input_group(context, fn_name)
    direct_metadata, direct_metadata_error = safe_direct_compile_metadata(
        tuner,
        launcher.config,
    )
    common = {
        "id": config_id,
        "fn_name": fn_name,
        "kernel": kernel_name,
        "source": archived_source or original_source,
        "original_source": original_source,
        "source_archive_error": source_archive_error,
        "config": config_to_dict(launcher.config),
        "direct_compile": direct_metadata,
        "direct_compile_error": direct_metadata_error,
        "candidate_id": candidate.get("candidate_id"),
        "variant_id": candidate.get("variant_id"),
        "full_config": candidate.get("full_config"),
        "runtime_blocks": context["runtime_blocks"],
        "args_summary": describe_arguments(context["args"]),
        "launch_args_summary": describe_arguments(context["launch_args"]),
        "kwargs_summary": {
            key: describe_value(value)
            for key, value in context["kwargs"].items()
        },
        "input_group_id": input_group["id"],
        "input_capture_mode": PRERUN_INPUT_MODE,
        "input_reused": input_reused,
        "input_artifact": input_group["path"],
        "input_capture_error": input_group["capture_error"],
    }

    # Journal the shared pending path before its first serialization or launch.
    # A hard abort can therefore still be associated with one candidate batch.
    persist({"event": "PRERUN_BEGIN", **common})
    ensure_input_group_captured(input_group, context)
    common["input_artifact"] = input_group["path"]
    common["input_capture_error"] = input_group["capture_error"]
    persist({"event": "PRERUN_READY", **common})

    try:
        result = original_measure_prerun(kernel_call_fn)
    except BaseException as error:
        retained_path, retain_error = retain_failed_input_group(
            input_group,
            fn_name,
        )
        common["input_artifact"] = retained_path
        common["input_retain_error"] = retain_error
        persist({
            "event": "PRERUN_PYTHON_EXCEPTION",
            **common,
            "error_type": type(error).__name__,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        })
        raise

    # Keep the shared snapshot for later candidates in this invocation. The
    # batch wrapper removes it when the batch returns unless a candidate failed;
    # the compatibility fallback cleans it at the next batch or process exit.
    common["input_artifact"] = None
    persist({"event": "PRERUN_PASS", **common})
    return result


def compile_with_trace(self, cfg, fn_name, kernel_name):
    config_id = next(counter)

    source = getattr(self, "filename", None)
    if not source:
        python_fn = getattr(getattr(self, "fn", None), "fn", None)
        code = getattr(python_fn, "__code__", None)
        source = getattr(code, "co_filename", None)

    original_source = str(source) if source else None
    archived_source, source_archive_error = archive_generated_source(
        source,
        fn_name,
    )

    direct_metadata, direct_metadata_error = safe_direct_compile_metadata(
        self,
        cfg,
    )
    common = {
        "id": config_id,
        "fn_name": fn_name,
        "kernel": kernel_name,
        "source": archived_source or original_source,
        "original_source": original_source,
        "source_archive_error": source_archive_error,
        "config": config_to_dict(cfg),
        "direct_compile": direct_metadata,
        "direct_compile_error": direct_metadata_error,
    }

    # Persist before entering triton.compile so a hard abort leaves evidence.
    persist({
        "event": "BEGIN",
        **common,
    })

    try:
        result = original_compile(self, cfg)
    except BaseException as error:
        # A failed compiler subprocess normally surfaces as a Python compiler
        # exception. A hard parent-process abort still leaves the BEGIN record.
        persist({
            "event": "PYTHON_EXCEPTION",
            **common,
            "error_type": type(error).__name__,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        })
        raise

    persist({
        "event": "PASS",
        **common,
    })
    return result


def traced_compile(self, cfg):
    fn_name, kernel_name = kernel_identity(self)
    if not should_trace_kernel(fn_name, kernel_name):
        # Preserve the original path completely for unrelated kernels: no
        # journal writes and no serialization lock.
        return original_compile(self, cfg)

    if SERIALIZE_MATCHED_KERNEL:
        with matched_kernel_lock:
            return compile_with_trace(
                self,
                cfg,
                fn_name,
                kernel_name,
            )

    # The normal discovery mode records the target kernel without changing its
    # original compilation concurrency.
    return compile_with_trace(self, cfg, fn_name, kernel_name)


NPUCachingAutotuner._precompile_config = traced_compile
if TRACE_PRERUN:
    if not callable(original_measure_prerun):
        raise SystemExit(
            "TRITON_TRACE_PRERUN=1 requires "
            "torch_npu._inductor.runtime.triton_heuristics._measure_prerun_ms"
        )
    npu_triton_heuristics._measure_prerun_ms = measure_prerun_with_trace
    if callable(original_benchmark_candidates):
        NPUCachingAutotuner._benchmark_candidate_entries = (
            benchmark_candidates_with_trace
        )


def normalize_test_selectors(target, args):
    """Convert pytest node IDs to selectors understood by unittest.run_tests."""
    normalized = []
    target_name = os.path.basename(target)
    target_stem = os.path.splitext(target_name)[0]

    for arg in args:
        if "::" not in arg or arg.startswith("-"):
            normalized.append(arg)
            continue

        parts = arg.split("::")
        first = parts[0]
        if (
            first == target
            or os.path.basename(first) == target_name
            or first == target_stem
        ):
            parts = parts[1:]

        if len(parts) >= 2:
            normalized.append(".".join(parts))
        else:
            normalized.append(arg)

    return normalized


if len(sys.argv) < 2:
    raise SystemExit(
        "usage: python trace_compile.py TARGET.py [target arguments ...]"
    )

target = os.path.abspath(sys.argv[1])
target_args = normalize_test_selectors(target, sys.argv[2:])
sys.argv = [target, *target_args]
os.makedirs(CAPTURE_DIR, exist_ok=True)
trace_mode = "serialized" if SERIALIZE_MATCHED_KERNEL else "original concurrency"
print(
    f"[trace_compile] kernel pattern: {KERNEL_PATTERN!r}; mode: {trace_mode}",
    file=sys.stderr,
    flush=True,
)
print(
    f"[trace_compile] capture directory: {CAPTURE_DIR}",
    file=sys.stderr,
    flush=True,
)
print(
    f"[trace_compile] journal: {JOURNAL}",
    file=sys.stderr,
    flush=True,
)
print(
    f"[trace_compile] source artifacts: {ARTIFACT_DIR}",
    file=sys.stderr,
    flush=True,
)
if TRACE_PRERUN:
    input_mode = {
        "values": "one shared full-value snapshot per candidate batch",
        "metadata": "one shared metadata-only snapshot per candidate batch",
        "none": "journal metadata only; launch replay disabled",
    }[PRERUN_INPUT_MODE]
    print(
        f"[trace_compile] prerun tracing enabled: {input_mode}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[trace_compile] failed prerun inputs: {PRERUN_INPUT_DIR}",
        file=sys.stderr,
        flush=True,
    )
if os.environ.get("TORCHINDUCTOR_COMPILE_THREADS") == "1":
    print(
        "[trace_compile] warning: TORCHINDUCTOR_COMPILE_THREADS=1 still "
        "serializes all Inductor compilation; unset it for the original flow",
        file=sys.stderr,
        flush=True,
    )
for cache_env in ("TRITON_DISABLE_CACHE", "TRITON_ALWAYS_COMPILE"):
    if os.environ.get(cache_env, "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        print(
            f"[trace_compile] warning: {cache_env} globally recompiles all "
            "Triton kernels; unset it during discovery and enable it only for "
            "the exported single-kernel reproducer",
            file=sys.stderr,
            flush=True,
        )
print(f"[trace_compile] target argv: {sys.argv!r}", file=sys.stderr, flush=True)
sys.path.insert(0, os.path.dirname(target))
runpy.run_path(target, run_name="__main__")

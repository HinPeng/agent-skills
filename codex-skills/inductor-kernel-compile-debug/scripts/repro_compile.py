"""Replay exactly one captured Triton compile or prerun attempt.

Usage:
    python repro_compile.py CAPTURE CONFIG_ID
    python repro_compile.py CAPTURE CONFIG_ID --export compile_repro.py
    python repro_compile.py CAPTURE PRERUN_ID --launch
    python repro_compile.py CAPTURE PRERUN_ID --export-launch launch_repro.py

Compile replay invokes only the captured autotuner config. Prerun replay also
restores the captured launch arguments, creates its launcher, launches it once,
and synchronizes the NPU. Neither mode reruns Dynamo, AOTAutograd, or the graph.
"""

import argparse
import base64
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
from torch._inductor.runtime.triton_heuristics import (
    config_from_dict,
    config_to_dict,
)


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


def build_launch_args(tuner, args, runtime_blocks, legacy_launch_args=None):
    builder = getattr(tuner, "_build_runtime_launch_args", None)
    if callable(builder):
        return tuple(builder(args, runtime_blocks))
    if legacy_launch_args is not None:
        return tuple(legacy_launch_args)
    return (*args, *runtime_blocks)


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
    else:
        args = tuple(payload["args"])
        kwargs = dict(payload["kwargs"])
    runtime_blocks = tuple(runtime_blocks)
    launch_args = build_launch_args(
        tuner,
        args,
        runtime_blocks,
        legacy_launch_args=payload.get("launch_args"),
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


def standalone_common_footer(fn_name, wanted_config):
    template = r'''

# ---- standalone reproducer helpers ----
def _standalone_repro_normalized(value):
    import json

    return json.dumps(value, default=repr, sort_keys=True)


def _standalone_repro_tuner_and_config():
    from torch._inductor.runtime.triton_heuristics import (
        config_from_dict,
        config_to_dict,
    )

    fn_name = __FN_NAME__
    wanted_config = __WANTED_CONFIG__
    tuners = [
        value
        for value in globals().values()
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
        tuner = exact[0]
    elif not exact and len(tuners) == 1:
        tuner = tuners[0]
    else:
        found_names = [
            tuner.get_fn_name()
            if callable(getattr(tuner, "get_fn_name", None))
            else type(tuner).__name__
            for tuner in tuners
        ]
        raise RuntimeError(
            f"Expected one tuner for {fn_name!r}; found {found_names}"
        )

    configs = [
        config
        for config in tuner.configs
        if _standalone_repro_normalized(config_to_dict(config))
        == _standalone_repro_normalized(wanted_config)
    ]
    if len(configs) == 1:
        config = configs[0]
    elif not configs:
        config = config_from_dict(wanted_config)
    else:
        raise RuntimeError(
            f"Expected one matching config, found {len(configs)}"
        )
    return tuner, config, fn_name, wanted_config
'''
    return template.replace("__FN_NAME__", repr(fn_name)).replace(
        "__WANTED_CONFIG__",
        repr(wanted_config),
    )


def standalone_compile_footer():
    return r'''

def _standalone_repro_main():
    tuner, config, fn_name, wanted_config = (
        _standalone_repro_tuner_and_config()
    )
    print(f"Compiling only: {fn_name}")
    print(f"Config: {wanted_config}")
    tuner._precompile_config(config)
    print(
        "Kernel compiled successfully; "
        "the original compiler failure did not reproduce."
    )


if __name__ == "__main__":
    _standalone_repro_main()
'''


def standalone_launch_footer(input_bytes, runtime_blocks):
    encoded_inputs = base64.b64encode(input_bytes).decode("ascii")
    template = r'''

_STANDALONE_PRERUN_INPUTS_B64 = __INPUT_BYTES__
_STANDALONE_RUNTIME_BLOCKS = __RUNTIME_BLOCKS__


def _standalone_repro_torch_load_inputs():
    import base64
    import io
    import os
    import torch

    raw = base64.b64decode(_STANDALONE_PRERUN_INPUTS_B64)
    map_location = os.environ.get("TRITON_REPRO_MAP_LOCATION") or None
    kwargs = {"map_location": map_location} if map_location else {}
    try:
        return torch.load(
            io.BytesIO(raw),
            weights_only=False,
            **kwargs,
        )
    except TypeError:
        return torch.load(io.BytesIO(raw), **kwargs)


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
            _standalone_repro_materialize(
                key,
                map_location,
            ): _standalone_repro_materialize(value, map_location)
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


def _standalone_repro_prepare_inputs(tuner):
    import os
    import torch

    payload = _standalone_repro_torch_load_inputs()
    map_location = os.environ.get("TRITON_REPRO_MAP_LOCATION") or None
    capture_mode = payload.get("capture_mode", "legacy-values")
    synthetic_seed = None
    if capture_mode == "metadata":
        synthetic_seed = int(os.environ.get("TRITON_REPRO_SEED", "0"))
        torch.manual_seed(synthetic_seed)
        args = tuple(
            _standalone_repro_materialize(payload["args"], map_location)
        )
        kwargs = dict(
            _standalone_repro_materialize(payload["kwargs"], map_location)
        )
    else:
        args = tuple(payload["args"])
        kwargs = dict(payload["kwargs"])

    runtime_blocks = tuple(_STANDALONE_RUNTIME_BLOCKS)
    builder = getattr(tuner, "_build_runtime_launch_args", None)
    if callable(builder):
        launch_args = tuple(builder(args, runtime_blocks))
    elif "launch_args" in payload:
        launch_args = tuple(payload["launch_args"])
    else:
        launch_args = (*args, *runtime_blocks)
    return (
        args,
        launch_args,
        kwargs,
        runtime_blocks,
        capture_mode,
        synthetic_seed,
    )


def _standalone_repro_main():
    import torch
    from torch._dynamo.device_interface import DeviceGuard

    tuner, config, fn_name, wanted_config = (
        _standalone_repro_tuner_and_config()
    )
    (
        args,
        launch_args,
        kwargs,
        runtime_blocks,
        capture_mode,
        synthetic_seed,
    ) = _standalone_repro_prepare_inputs(tuner)
    device_interface = tuner.get_device_interface()
    device_index = tuner.triton_meta.get("device", 0)

    print(f"Compiling and launching once: {fn_name}")
    print(f"Config: {wanted_config}")
    print(f"Runtime blocks: {runtime_blocks}")
    print(f"Input capture mode: {capture_mode}")
    if synthetic_seed is not None:
        print(f"Synthetic input seed: {synthetic_seed}")
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
        cloned_args, cloned_kwargs = tuner.clone_args(
            *launch_args,
            **kwargs,
        )
        tuner.reset_to_zero_args(*args, **kwargs)
        launcher(*cloned_args, **cloned_kwargs, stream=stream)
        end_event.record()
        torch.npu.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    print(f"Kernel prerun succeeded in {elapsed_ms:.3f} ms")


if __name__ == "__main__":
    _standalone_repro_main()
'''
    return template.replace("__INPUT_BYTES__", repr(encoded_inputs)).replace(
        "__RUNTIME_BLOCKS__",
        repr(tuple(runtime_blocks)),
    )


def render_standalone_reproducer(
    source_text,
    fn_name,
    wanted_config,
    input_bytes=None,
    runtime_blocks=(),
):
    mode = "compile and launch" if input_bytes is not None else "compile"
    header = f"""# Standalone single-kernel {mode} reproducer.
# Extracted by repro_compile.py for {fn_name}.
# It does not rerun Dynamo, AOTAutograd, or the surrounding graph.

"""
    footer = standalone_common_footer(fn_name, wanted_config)
    if input_bytes is None:
        footer += standalone_compile_footer()
    else:
        footer += standalone_launch_footer(input_bytes, runtime_blocks)
    return header + source_text.rstrip() + footer


def export_standalone_reproducer(
    output_path,
    source_path,
    fn_name,
    wanted_config,
    input_path=None,
    runtime_blocks=(),
):
    output_path = output_path.expanduser().resolve()
    protected_paths = {source_path}
    if input_path is not None:
        protected_paths.add(input_path)
    if output_path in protected_paths:
        raise RuntimeError("Refusing to overwrite a captured artifact")

    source_text = source_path.read_text(encoding="utf-8")
    input_bytes = input_path.read_bytes() if input_path is not None else None
    standalone = render_standalone_reproducer(
        source_text,
        fn_name,
        wanted_config,
        input_bytes=input_bytes,
        runtime_blocks=runtime_blocks,
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
        help="write a standalone compile-only reproducer",
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
        help="write a standalone compile-and-launch reproducer",
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

#!/usr/bin/env python3
"""Statically inspect AsyncCompile.triton bindings; never execute input code.

Exports evidence fragments, NOT a runnable reproducer. Python standard library only.
"""

import argparse
import ast
import io
import json
from pathlib import Path
import sys
import tokenize


def segment(text, node):
    return ast.get_source_segment(text, node) or ""


def binding_name(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    elif isinstance(node, ast.AnnAssign):
        target = node.target
    else:
        return None
    return target.id if isinstance(target, ast.Name) else None


def literal_string(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def call_argument(call, index, keyword):
    if len(call.args) > index:
        return call.args[index]
    return next((kw.value for kw in call.keywords if kw.arg == keyword), None)


def inspect(text, filename):
    tree = ast.parse(text, filename=filename)
    nodes = sorted(ast.walk(tree), key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)))
    bindings = []
    for node in nodes:
        name = binding_name(node)
        call = getattr(node, "value", None)
        if not name or not isinstance(call, ast.Call):
            continue
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "triton":
            continue
        if not isinstance(call.func.value, ast.Name):
            continue
        compiler = call.func.value.id
        bindings.append({
            "binding": name,
            "compiled_name": literal_string(call_argument(call, 0, "kernel_name")),
            "compiler_receiver": compiler,
            "line": node.lineno,
            "end_line": node.end_lineno,
            "compile_block": segment(text, node),
            "kernel_source": literal_string(call_argument(call, 1, "source_code")),
            "calls": [],
            "support_fragments": [],
        })
    for item in bindings:
        for node in nodes:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                receiver = node.func.value
                if isinstance(receiver, ast.Name) and receiver.id == item["binding"] and node.func.attr == "run":
                    referenced = set()
                    for value in [*node.args, *(kw.value for kw in node.keywords)]:
                        referenced.update(n.id for n in ast.walk(value) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
                    item["calls"].append({
                        "line": node.lineno,
                        "end_line": node.end_lineno,
                        "source": segment(text, node),
                        "referenced_names": sorted(referenced),
                    })
                if isinstance(receiver, ast.Name) and receiver.id == item["compiler_receiver"] and node.func.attr == "wait":
                    item["support_fragments"].append({"kind": "wait", "line": node.lineno, "source": segment(text, node)})
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                item["support_fragments"].append({"kind": "import", "line": node.lineno, "source": segment(text, node)})
            if binding_name(node) == item["compiler_receiver"]:
                item["support_fragments"].append({"kind": "compiler_assignment", "line": node.lineno, "source": segment(text, node)})
        item["support_fragments"].sort(key=lambda part: part["line"])
    return bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_code", type=Path)
    parser.add_argument("--kernel", help="Exact Python binding or literal compiled name")
    parser.add_argument("--out", type=Path, help="New directory for evidence; requires --kernel")
    args = parser.parse_args()
    if args.out and not args.kernel:
        parser.error("--out requires --kernel")
    try:
        source_path = args.output_code.resolve(strict=True)
        raw = source_path.read_bytes()
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        text = raw.decode(encoding)
        items = inspect(text, str(source_path))
        warnings = [
            "Static evidence only: input code was not imported or executed.",
            "Matches simple name = receiver.triton(...) bindings and direct name.run(...) calls only.",
            "Receiver identity, scope, control flow, dependencies, tensor data/layout/aliasing and compiler options require manual review.",
            "No runnable reproducer or NPU validation is produced.",
        ]
        result = {
            "schema_version": 1,
            "source_path": str(source_path),
            "source_encoding": encoding,
            "warnings": warnings,
        }
        if args.kernel:
            matches = [item for item in items if args.kernel in (item["binding"], item["compiled_name"])]
            if len(matches) != 1:
                raise ValueError(f"Expected one exact kernel match; found {len(matches)}. List bindings and resolve ambiguity manually.")
            item = matches[0]
            if not item["calls"]:
                warnings.append("No direct .run call found; compilation evidence may still be useful. Restore the launch manually.")
            if item["kernel_source"] is None:
                warnings.append("Kernel source is not a supported literal string; preserve its generation dependencies manually.")
            result["kernel"] = {key: val for key, val in item.items() if key not in ("compile_block", "kernel_source", "support_fragments")}
            result["support_fragments"] = item["support_fragments"]
            if args.out:
                out = args.out.absolute()
                if out.exists() or out.is_symlink():
                    raise FileExistsError(f"Refusing to overwrite existing output: {out}")
                out.mkdir(parents=True, exist_ok=False)
                files = {
                    "original_output_code.py": raw,
                    "compile_block.txt": item["compile_block"].encode("utf-8"),
                    "calls.json": (json.dumps(item["calls"], ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
                    "support_fragments.json": (json.dumps(item["support_fragments"], ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
                }
                if item["kernel_source"] is not None:
                    files["kernel_source.txt"] = item["kernel_source"].encode("utf-8")
                result["saved_to"] = str(out)
                for name, content in files.items():
                    (out / name).write_bytes(content)
                (out / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                result.pop("support_fragments")
        else:
            result["kernels"] = [
                {"binding": item["binding"], "compiled_name": item["compiled_name"], "line": item["line"], "direct_run_calls": len(item["calls"])}
                for item in items
            ]
            if not items:
                warnings.append("No supported compilation bindings found. Inspect the original file or IR manually.")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, SyntaxError, UnicodeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

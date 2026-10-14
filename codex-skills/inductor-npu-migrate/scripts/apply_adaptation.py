#!/usr/bin/env python3
"""Apply reviewed, hash-guarded exact edits; never infer CUDA/NPU equivalence."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import tokenize
import io
from common import cases, digest, safe_path, test_files, write_json


def render(data, entry, filename):
    before_hash = digest(data)
    if before_hash != entry['sha256']:
        raise ValueError(f'{filename}: source SHA256 differs; regenerate the plan against current content')
    encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
    source = data.decode(encoding)
    for operation in entry.get('operations', []):
        old, new = operation['old'], operation['new']
        if not old or old == new or source.count(old) != 1:
            raise ValueError(f'{filename}: edit must have exactly one nonempty, distinct match: {old[:100]!r}')
        source = source.replace(old, new, 1)
    tree = ast.parse(source, filename=filename)
    initialization = entry.get('initialization', '')
    if initialization:
        compile(initialization, filename + ':initialization', 'exec')
        # Insert before every existing dependency import, after docstring/futures.
        prefix_end = 0
        for i, node in enumerate(tree.body):
            is_doc = i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
            if is_doc or isinstance(node, ast.ImportFrom) and node.module == '__future__':
                prefix_end = node.end_lineno
            else:
                break
        lines = source.splitlines(keepends=True)
        if not prefix_end:
            # Keep shebang, encoding and leading comments ahead of new imports.
            while prefix_end < len(lines) and (not lines[prefix_end].strip() or lines[prefix_end].lstrip().startswith('#')):
                prefix_end += 1
        if prefix_end and not lines[prefix_end - 1].endswith(('\n', '\r')):
            lines[prefix_end - 1] += '\n'
        source = ''.join(lines[:prefix_end]) + '\n' + initialization.rstrip() + '\n\n' + ''.join(lines[prefix_end:])
    compile(source, filename, 'exec')
    return source.encode(encoding)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='adapted/test/inductor')
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    entries = json.loads(args.plan.read_text(encoding='utf-8'))['files']
    root = args.root.resolve()
    paths = [e['path'] for e in entries]
    if len(paths) != len(set(paths)):
        parser.error('Duplicate file entries')
    actual = {p.relative_to(root).as_posix() for p in test_files(root)}
    if actual - set(paths):
        parser.error(f'Plan does not cover test files: {sorted(actual - set(paths))}')
    previous = {}
    if args.report.exists():
        old_report = json.loads(args.report.read_text(encoding='utf-8'))
        if old_report.get('plan_sha256') == digest(args.plan.read_bytes()):
            previous = {e['path']: e for e in old_report['files']}
    pending, reports = [], []
    for entry in entries:
        path = safe_path(root, entry['path'])
        if path.suffix != '.py' or not path.is_file():
            raise ValueError(f'Not an existing Python file: {path}')
        status = entry['status']
        if status not in {'preliminary', 'unchanged', 'not_applicable', 'manual_required'}:
            raise ValueError(f'Invalid adaptation status: {status}')
        if not entry.get('reason') or not entry.get('intents'):
            raise ValueError(f'{path}: reason and per-intent evidence are required')
        for intent in entry['intents']:
            if not all(intent.get(k) for k in ('symbol', 'status', 'evidence')):
                raise ValueError(f'{path}: each intent requires symbol/status/evidence')
            if intent['status'] not in {'applied_exact', 'applied_semantic', 'already_satisfied',
                                        'obsolete_in_target', 'not_applicable', 'unresolved'}:
                raise ValueError(f'{path}: invalid intent status')
        if status != 'preliminary' and (entry.get('operations') or entry.get('initialization')):
            raise ValueError(f'{path}: changed files must be preliminary')
        data = path.read_bytes()
        old = previous.get(entry['path'])
        if old and digest(data) == old['output_sha256']:
            output = data
        else:
            output = render(data, entry, str(path))
        before_cases = Counter(cases(path))
        # Test definitions must not be removed or renamed as part of migration.
        parsed = ast.parse(output, filename=str(path))
        compile(parsed, str(path), 'exec')
        # Compare function/class identities without writing any provisional file.
        def identities(tree):
            result = []
            def walk(node, classes=()):
                if isinstance(node, ast.ClassDef):
                    classes += (node.name,)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith('test_'):
                        result.append((node.name, '.'.join(classes)))
                    return
                for child in ast.iter_child_nodes(node):
                    walk(child, classes)
            walk(tree)
            return Counter(result)
        if before_cases != identities(parsed):
            raise ValueError(f'{path}: test definitions changed; preserve original names and class scope')
        pending.append((path, output))
        resolved = sum(i['status'] in {'applied_exact', 'applied_semantic', 'already_satisfied',
                                       'obsolete_in_target'} for i in entry['intents'])
        reports.append(dict(entry, output_sha256=digest(output), syntax_status='pass',
                            intent_total=len(entry['intents']), intent_resolved=resolved,
                            intent_unresolved=sum(i['status'] == 'unresolved' for i in entry['intents']),
                            runtime_status='not_run', import_status='not_run', collection_status='not_run'))
    pending_map = dict(pending)
    for path in root.rglob('*.py'):
        compile(pending_map.get(path, path.read_bytes()), str(path), 'exec')
    # All edits and syntax checks succeed before mutating any source file.
    for path, data in pending:
        if path.read_bytes() != data:
            path.write_bytes(data)
    write_json(args.report, dict(plan_sha256=digest(args.plan.read_bytes()), files=reports,
                                validation='V1 static only; NOT NPU execution evidence'))
    print(f'V1 passed for {len(reports)} reviewed files; runtime NOT_RUN')


if __name__ == '__main__':
    main()

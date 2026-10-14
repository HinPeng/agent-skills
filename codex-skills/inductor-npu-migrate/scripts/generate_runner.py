#!/usr/bin/env python3
"""Fill TEST_FILES with all or selected relative test_*.py paths."""
import argparse
from pathlib import Path
import shlex
from common import safe_path, test_files, write_json


def generate(root, output, selected=None):
    root, output = Path(root).resolve(), Path(output)
    files = [p.relative_to(root).as_posix() for p in test_files(root)]
    if not files:
        raise ValueError(f'No test_*.py files in {root}')
    if selected:
        missing = set(selected) - set(files)
        if missing:
            raise ValueError(f'Unknown test paths: {sorted(missing)}')
        files = sorted(set(selected))
    for name in files:
        safe_path(root, name)
        if '\n' in name or '\r' in name:
            raise ValueError('Newline in filename is unsupported')
    template = Path(__file__).with_name('run_pytest.sh').read_text(encoding='utf-8')
    text = template.replace('# GENERATED_TEST_FILES', '\n'.join('    ' + shlex.quote(n) for n in files))
    if output.exists():
        raise ValueError(f'Runner exists; use a new path: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding='utf-8', newline='\n')
    write_json(output.with_suffix('.files.json'), dict(scope='selected' if selected else 'all', test_files=files))
    print(f'TEST_FILES: {len(files)} files -> {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--files', nargs='+', help='Exact paths relative to test/inductor; omit for all')
    args = parser.parse_args()
    generate(args.test_root, args.output, args.files)


if __name__ == '__main__':
    main()

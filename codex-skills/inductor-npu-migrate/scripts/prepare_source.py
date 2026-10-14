#!/usr/bin/env python3
"""Export an exact PyTorch release tag without modifying an existing checkout."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
from common import digest, test_files, write_json

UPSTREAM = 'https://github.com/pytorch/pytorch.git'


def git(repo, *args, capture=True):
    return subprocess.run(['git', '-C', str(repo), *args], check=True,
                          capture_output=capture, text=capture).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--source-repo', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'v\d+\.\d+\.\d+', args.tag):
        parser.error('Specify an exact stable release tag, e.g. v2.13.0; never silently use main/rc')
    output = args.output.resolve()
    if output.exists():
        parser.error('Use a new output directory; existing runs are never deleted or overwritten')
    if args.source_repo and (output == args.source_repo.resolve() or
                             output.is_relative_to(args.source_repo.resolve())):
        parser.error('Output must be outside the source repository')
    output.mkdir(parents=True)
    repo = args.source_repo.resolve() if args.source_repo else output / 'upstream'
    if not args.source_repo:
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', args.tag,
                        '--single-branch', UPSTREAM, str(repo)], check=True)
    origin = git(repo, 'remote', 'get-url', 'origin').strip()
    if origin.rstrip('/').removesuffix('.git') not in (
        'https://github.com/pytorch/pytorch', 'git@github.com:pytorch/pytorch'):
        raise ValueError(f'Expected official PyTorch origin, got {origin}')
    ref = f'refs/tags/{args.tag}'
    commit = git(repo, 'rev-parse', '--verify', f'{ref}^{{commit}}').strip()
    git(repo, 'cat-file', '-e', f'{commit}:test/inductor')
    archive = output / 'source.tar'
    git(repo, 'archive', '--format=tar', f'--output={archive}', commit, capture=False)
    adapted = output / 'adapted'
    adapted.mkdir()
    with tarfile.open(archive) as tar:
        # data filter rejects outside paths, unsafe links and special files.
        tar.extractall(adapted, filter='data')
    archive.unlink()
    test_root = adapted / 'test' / 'inductor'
    original = output / 'original' / 'test' / 'inductor'
    shutil.copytree(test_root, original, symlinks=True)
    files = [p.relative_to(test_root).as_posix() for p in test_files(test_root)]
    if not files:
        raise ValueError('Tag has no test/inductor/test_*.py files')
    rules = Path(__file__).resolve().parent.parent / 'references' / 'NPU_TEST_ADAPTATION_RULES.md'
    manifest = dict(repository=origin, tag=args.tag, commit=commit,
                    test_files=files, rules_sha256=digest(rules.read_bytes()),
                    source_hashes={p.relative_to(original).as_posix(): digest(p.read_bytes())
                                   for p in original.rglob('*') if p.is_file()},
                    validation='V0 exported; NPU runtime NOT_RUN')
    write_json(output / 'source_manifest.json', manifest)
    subprocess.run([__import__('sys').executable, str(Path(__file__).with_name('extract_testcases_name.py')),
                    '--input', str(original), '--output', str(output / '待测用例.xlsx')], check=True)
    print(json.dumps(dict(tag=args.tag, commit=commit, files=len(files), output=str(output))))


if __name__ == '__main__':
    main()

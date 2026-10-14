#!/usr/bin/env python3
"""Extract static test definitions from test_*.py, preserving class and path."""
import argparse
from pathlib import Path
import csv
from common import cases, test_files, style_sheet


def extract(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'Test directory does not exist: {root}')
    return [(p.relative_to(root).as_posix(), name, cls, '', '')
            for p in test_files(root) for name, cls in cases(p)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('待测用例.xlsx'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'Refusing to overwrite existing inventory: {args.output}')
    rows = extract(args.input)
    if not rows:
        parser.error('No test definitions found; inventory was not created')
    headers = ['file_name', 'test_case', 'class_name', '执行结果', '失败原因']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == '.csv':
        with args.output.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)
    elif args.output.suffix.lower() == '.xlsx':
        from openpyxl import Workbook
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = '待测用例'
        sheet.append(headers)
        for row in rows:
            sheet.append(row)
        style_sheet(sheet)
        workbook.save(args.output)
    else:
        parser.error('Output must be .xlsx or .csv')
    print(f'{len(rows)} static definitions in {len(test_files(args.input))} files: {args.output}')


if __name__ == '__main__':
    main()

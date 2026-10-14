#!/usr/bin/env python3
"""Run only on Linux NPU: check actual installed packages and optional compile."""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import re
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-torch', required=True, help='Community testcase release, e.g. 2.13.0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    info = dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                python=sys.version, platform=platform.platform(), cwd=str(Path.cwd()),
                environment={k: v for k, v in os.environ.items() if k in {
                    'TORCHINDUCTOR_NPU_BACKEND', 'ASCEND_RT_VISIBLE_DEVICES',
                    'ASCEND_DEVICE_ID', 'PYTORCH_TESTING_DEVICE_FOR_CUSTOM',
                    'PYTORCH_TESTING_DEVICE_EXCEPT_FOR', 'PYTHONPATH'}}, status='ERROR')
    rc = 1
    try:
        if platform.system() != 'Linux':
            raise RuntimeError('NPU preflight is for Linux; do not run this during Windows static validation')
        import torch
        import torch_npu
        import torch_npu._inductor  # noqa: F401
        info.update(torch_version=torch.__version__, torch_file=torch.__file__,
                    torch_npu_version=getattr(torch_npu, '__version__', 'unknown'),
                    torch_npu_file=torch_npu.__file__, npu_available=torch.npu.is_available())
        def release(version):
            match = re.match(r'^v?(\d+\.\d+\.\d+)', str(version))
            return match.group(1) if match else None
        expected = release(args.expected_torch)
        info['expected_torch'] = args.expected_torch
        if not expected or release(torch.__version__) != expected or release(info['torch_npu_version']) != expected:
            raise RuntimeError(f'Version mismatch: require torch and torch_npu release {args.expected_torch}')
        if not info['npu_available']:
            raise RuntimeError('NPU unavailable')
        info.update(npu_count=torch.npu.device_count(), npu_name=torch.npu.get_device_name(0))
        if args.smoke:
            torch.manual_seed(1234)
            x = torch.randn(64, device='npu', dtype=torch.float32)
            def fn(t):
                return torch.sin(t) + t * t
            expected = fn(x)
            actual = torch.compile(fn, backend='inductor', fullgraph=True)(x)
            torch.npu.synchronize()
            assert actual.device.type == 'npu'
            assert actual.dtype == expected.dtype
            torch.testing.assert_close(actual, expected)
            info['smoke'] = 'PASSED (only this diagnostic function)'
        info['status'] = 'OK'
        rc = 0
    except Exception as exc:
        info['error'] = f'{type(exc).__name__}: {exc}'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())

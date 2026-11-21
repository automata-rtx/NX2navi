#!/usr/bin/env python3
"""Check the mesh fallback's real allocation and readback owner with a mock Vulkan device."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    cc = os.environ.get('CC') or shutil.which('clang')
    if not cc:
        raise SystemExit('Clang is required (set CC)')
    clang = Path(shutil.which(cc) or cc).resolve()
    vma = Path(os.environ.get('VMA_INCLUDE', clang.parent.parent / 'include' / 'vma'))
    env = dict(os.environ, PATH=str(clang.parent) + os.pathsep + os.environ.get('PATH', ''))
    with tempfile.TemporaryDirectory(prefix='mesh-fallback-admission-') as directory:
        work = Path(directory)
        exe = work / ('check.exe' if os.name == 'nt' else 'check')
        # The mock does not execute the scan shader. Production uses the generated module.
        (work / 'meshfb_scan_spv.h').write_text('static const uint32_t meshfb_scan_spv[] = {0x07230203u};\n')
        subprocess.run([str(clang), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror',
                        '-fsanitize=undefined', '-fsanitize-trap=undefined',
                        '-I' + str(ROOT / 'runtime'), '-I' + str(vma), '-I' + str(work),
                        str(ROOT / 'tests/mesh_fallback_admission_check.c'), '-o', str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == '__main__':
    main()

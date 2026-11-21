"""Build and run tests/gpa_check.c: runtime/vk_gpa.c's device address table, with blocks given back and taken again.

Needs the Vulkan headers and vk_mem_alloc.h (docs/building.md), no GPU and no game files.

Usage: gpa_check.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MSYS_BIN = Path(r'C:\msys64\ucrt64\bin')


def compiler():
    """$CC, else MSYS2 UCRT64's clang, else clang on PATH (as tests/diff/common.py picks it, without its Unicorn)."""
    if os.environ.get('CC'):
        return os.environ['CC']
    if os.name == 'nt' and (MSYS_BIN / 'clang.exe').exists():
        return str(MSYS_BIN / 'clang.exe')
    return shutil.which('clang') or 'clang'


def vma_include():
    chosen = Path(compiler()).parent.parent / 'include' / 'vma'
    for d in (chosen, MSYS_BIN.parent / 'include' / 'vma', ROOT / 'toolchain' / 'vma', Path('/usr/include')):
        if (d / 'vk_mem_alloc.h').is_file():
            return d
    sys.exit('gpa_check: vk_mem_alloc.h not found (docs/building.md)')


def main():
    work = ROOT / 'build' / 'test' / 'gpa'
    work.mkdir(parents=True, exist_ok=True)
    exe = work / ('gpa_check.exe' if os.name == 'nt' else 'gpa_check')
    env = dict(os.environ)
    if os.name == 'nt':
        env['PATH'] = str(Path(compiler()).parent) + os.pathsep + env.get('PATH', '')
    subprocess.check_call([compiler(), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror',
                           '-fno-strict-aliasing', '-fsanitize=undefined', '-fsanitize-trap=all', '-DRC_BUDGET_TEST',
                           '-I', str(ROOT / 'runtime'), '-I', str(vma_include()), str(ROOT / 'tests' / 'gpa_check.c'),
                           str(ROOT / 'runtime' / 'vk_gpa.c'), str(ROOT / 'runtime' / 'resource_core_registry.c'),
                           str(ROOT / 'runtime' / 'resource_host_write.c'),
                           str(ROOT / 'runtime' / 'resource_core_vma.c'),
                           str(ROOT / 'runtime' / 'resource_core_budget.c'),
                           '-o', str(exe)], env=env)
    for mode in ('refuse', 'fail-table', 'fail-zero', 'fail-sink', 'no-table-map', 'no-zero-map',
                 'no-device-address', 'no-capability', 'diagnostic-disabled', 'retry-after-relief', 'retry-after-cancel'):
        subprocess.check_call([str(exe), mode], env=env)
    return subprocess.run([str(exe)], env=env).returncode


if __name__ == '__main__':
    sys.exit(main())

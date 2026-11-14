"""Check command-copy lifetimes and uniform snapshot leases without a game or GPU.

Uses the actual runtime implementations with host-only platform/Vulkan stubs.
Requires Clang and Vulkan/VMA headers. CC selects Clang; VMA_INCLUDE can specify
the directory holding vk_mem_alloc.h. --ubsan repeats with undefined-behavior traps.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ubsan', action='store_true')
    args = parser.parse_args()
    default = Path(r'C:\msys64\ucrt64\bin\clang.exe')
    cc = os.environ.get('CC') or (str(default) if default.exists() else shutil.which('clang'))
    if not cc:
        raise SystemExit('Clang not found; set CC to its path')
    cc = Path(cc).resolve()
    candidates = [Path(os.environ['VMA_INCLUDE'])] if os.environ.get('VMA_INCLUDE') else [
        cc.parent.parent / 'include' / 'vma', ROOT / 'toolchain' / 'vma', Path('/usr/include/vma')]
    vma = next((p for p in candidates if (p / 'vk_mem_alloc.h').is_file()), None)
    if not vma:
        raise SystemExit('Set VMA_INCLUDE to the directory holding vk_mem_alloc.h')
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get('PATH', ''))
    work = ROOT / 'build' / 'test' / 'resource-preparation'
    work.mkdir(parents=True, exist_ok=True)
    flags = ['-std=gnu11', '-O1', '-g', '-ffunction-sections', '-fdata-sections',
             '-fuse-ld=lld', '-Wl,--gc-sections', '-Wall', '-Wextra', '-Werror',
             '-I' + str(ROOT / 'runtime'), '-I' + str(ROOT / 'include'), '-I' + str(vma)]
    if args.ubsan:
        flags += ['-fsanitize=undefined', '-fsanitize-trap=undefined']
    for name in ('command_index_check', 'usnap_lease_check'):
        exe = work / (name + ('-ubsan' if args.ubsan else '') + ('.exe' if os.name == 'nt' else ''))
        subprocess.run([str(cc), *flags, str(ROOT / 'tests' / (name + '.c')), '-o', str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == '__main__':
    main()

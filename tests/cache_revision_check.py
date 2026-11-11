"""Build and run the cache revision and pipeline-feedback policy checks; no GPU required."""
import os
from pathlib import Path
import subprocess
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]
MSYS_BIN = Path('C:/msys64/ucrt64/bin')


def main():
    work = ROOT / 'build' / 'test' / 'cache_revision'
    work.mkdir(parents=True, exist_ok=True)
    exe = work / ('cache_revision_check.exe' if os.name == 'nt' else 'cache_revision_check')
    env = dict(os.environ)
    if os.name == 'nt' and MSYS_BIN.exists():
        env['PATH'] = str(MSYS_BIN) + os.pathsep + env.get('PATH', '')
    cc = os.environ.get('CC') or shutil.which('clang', path=env.get('PATH'))
    if not cc:
        raise RuntimeError('Clang is required; set CC to the UCRT64 Clang executable on Windows')
    subprocess.check_call([cc, '-O1', '-Wall', '-Wextra', '-Werror', '-I', str(ROOT / 'runtime'),
                           str(ROOT / 'tests' / 'cache_revision_check.c'), '-o', str(exe)], env=env)
    return subprocess.run([str(exe)], env=env).returncode


if __name__ == '__main__':
    sys.exit(main())

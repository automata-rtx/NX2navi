"""Check that mesh lookahead cannot cross commands that can change its inputs."""
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def function(s, signature):
    start = s.index(signature + ' {')
    return s[start:s.index('\n}', start) + 2]


def main():
    work = ROOT / 'build/test/mesh_ahead'
    work.mkdir(parents=True, exist_ok=True)
    s = (ROOT / 'runtime/vk_exec.c').read_text()
    (work / 'mesh_run.inc').write_text(
        function(s, 'static int op_linear(uint32_t op)') + '\n' +
        s[s.index('#define MESH_RUN_OPS'):s.index('// For the perf log: the ops the runs ended')] +
        function(s, 'int vk_exec_mesh_run(const void *cur, const void **out, int max)'))
    cc = os.environ.get('CC') or shutil.which('clang')
    if not cc:
        raise RuntimeError('Set CC to the UCRT64 Clang executable')
    exe = work / 'check.exe'
    subprocess.run([cc, '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror',
                    '-I', str(work), '-I', str(ROOT / 'runtime'),
                    str(ROOT / 'tests/mesh_ahead_check.c'), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=10)


if __name__ == '__main__':
    main()

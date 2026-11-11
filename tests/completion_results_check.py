"""Exercise the runtime's result queue and query batching without a GPU."""
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    work = ROOT / 'build/test/completion_results'
    work.mkdir(parents=True, exist_ok=True)
    sync = (ROOT / 'runtime/vk_sync.c').read_text()
    start = sync.index('static void pending_push(')
    end = sync.index('static int fence_wait_impl', start)
    (work / 'results.inc').write_text(sync[start:end])
    xfer = (ROOT / 'runtime/vk_exec_xfer.c').read_text()
    start = xfer.index('#define OQ_BATCH')
    end = xfer.index('static void oq_cb(', start)
    (work / 'queries.inc').write_text(xfer[start:end])
    cc = os.environ.get('CC') or shutil.which('clang')
    if not cc:
        raise RuntimeError('Set CC to the UCRT64 Clang executable')
    exe = work / 'check.exe'
    subprocess.run([cc, '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror',
                    '-I', str(work), str(ROOT / 'tests/completion_results_check.c'),
                    '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=20)


if __name__ == '__main__':
    main()

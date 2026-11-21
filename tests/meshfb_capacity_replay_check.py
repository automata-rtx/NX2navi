"""Test the required mesh capacity receipt on the actual replay and native-prefix path."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from ordered_dispatch_preparation_check import REPLAY, between, function


def main():
    cc = os.environ.get('CC') or shutil.which('clang')
    if not cc:
        raise SystemExit('Clang is required (set CC)')
    clang = Path(shutil.which(cc) or cc).resolve()
    vma = Path(os.environ.get('VMA_INCLUDE', clang.parent.parent / 'include' / 'vma'))
    env = dict(os.environ, PATH=str(clang.parent) + os.pathsep + os.environ.get('PATH', ''))
    source = REPLAY.read_text(encoding='utf-8')
    captures = [
        between(source, 'static _Thread_local VkExec *x_cur;',
                'static void selected_sampled_failure_log('),
        between(source, 'static void (*probe_pass_record)', 'static void resource_target_prepare('),
        *(function(source, name) for name in ('cancel_recording', 'vk_exec_graphics_pool_progress',
                                             'progress_adopt_tail', 'settle_mesh_capacity')),
    ]
    tail = between(source, '    RP_CALL(RP_GROUP_DRAIN, group_drain(ex, PW_END));', '    if (n) n_subs++;')
    with tempfile.TemporaryDirectory(prefix='meshfb-replay-capacity-') as directory:
        work = Path(directory)
        (work / 'functions.inc').write_text('\n\n'.join(captures), encoding='utf-8')
        (work / 'replay_tail.inc').write_text(tail, encoding='utf-8')
        exe = work / ('check.exe' if os.name == 'nt' else 'check')
        fixture = Path(__file__).with_suffix('.c')
        subprocess.run([str(clang), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror',
                        '-fsanitize=undefined', '-fsanitize-trap=undefined',
                        '-I' + str(work), '-I' + str(REPLAY.parent), '-I' + str(vma),
                        str(fixture), '-o', str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == '__main__':
    main()

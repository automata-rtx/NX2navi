"""Compile production state deltas and compare with the frozen scalar replay semantics."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get('EXEC_STATE_REFERENCE_ROOT', ROOT.parent / 'change-driven-source'))


def function(source, name):
    match = re.search(r'^(?:static )?[^\n]*\b' + name + r'\(', source, re.M)
    start = match.start()
    opening = source.index('{', match.end())
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def main():
    cc = os.environ.get('CC') or shutil.which('clang')
    if not cc: raise SystemExit('Set CC to the UCRT64 Clang executable')
    cc = Path(cc)
    work = ROOT / 'build/test'
    work.mkdir(parents=True, exist_ok=True)
    source = (BASE / 'runtime/vk_exec_draw.c').read_text()
    replay = (BASE / 'runtime/vk_exec.c').read_text()
    first = replay.index('    case OP_VIEWPORTS:', replay.index('static void op_one('))
    last = replay.index('    case OP_BARRIER:', first)
    scalar = 'static void reference_state(VkExec *ex, const op_hdr *h) {\nconst void *a = nvn_op_payload(h);\nswitch (h->op) {\n'
    scalar += replay[first:last] + 'default: op_draw_impl(ex, h->op, a); break;\n}}\n'
    stubs = '\n'.join('static void ' + name + '(VkExec *ex, const void *p) {(void)ex;(void)p;abort();}'
                      for name in ['op_draw','op_draw_idx','op_draw_idx_ind','op_draw_mesh','op_draw_mesh_ind',
                                   'op_dispatch','op_dispatch_ind','op_update_ub'])
    sys.path.insert(0, str(BASE / 'tools'))
    from check_content import GENERATED_MARKER
    marker = GENERATED_MARKER.decode() if isinstance(GENERATED_MARKER, bytes) else GENERATED_MARKER
    generated = '// ' + marker + '\n' + stubs + '\n' + function(source, 'bind_handles') + '\n'
    generated += function(source, 'op_draw_impl') + '\n' + scalar
    (work / 'scalar_state_oracle.h').write_text(generated)
    env = dict(os.environ)
    env['PATH'] = str(cc.parent) + os.pathsep + env.get('PATH', '')
    exe = work / 'exec_state_check.exe'
    command = [str(cc), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function',
               '-Wno-unused-variable', '-fsanitize=undefined', '-fsanitize-trap=all',
               '-I', str(ROOT / 'runtime'), '-I', str(work),
               '-I', str(cc.parent.parent / 'include/vma'), str(ROOT / 'tests/exec_state_check.c'), '-o', str(exe)]
    subprocess.check_call(command, env=env)
    subprocess.check_call([str(exe)], env=env)


if __name__ == '__main__': main()

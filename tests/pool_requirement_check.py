"""Check exact requirement selectors and the production ordered-leader scheduling helpers."""
from pathlib import Path
import os
import re
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ['CC'])
def function(source, name):
    mask = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"', lambda m: ' ' * len(m[0]), source)
    match = re.search(r'(?m)^(?:static\s+)?(?:[\w*]+\s+)+\*?' + name + r'\([^;]*?\)\s*\{', mask)
    assert match, name
    end = mask.index('{', match.start()) + 1
    depth = 1
    while depth:
        depth += (mask[end] == '{') - (mask[end] == '}')
        end += 1
    return source[match.start():end]
work = ROOT / 'build/test/pool-requirement'
work.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'tools'))
from check_content import GENERATED_MARKER
marker = GENERATED_MARKER.decode() if isinstance(GENERATED_MARKER, bytes) else GENERATED_MARKER
draw = (ROOT / 'runtime/vk_exec_draw.c').read_text()
replay = (ROOT / 'runtime/vk_exec.c').read_text()
(work / 'pool_requirement_constants.h').write_text('// ' + marker + '\n' +
    '\n'.join(re.findall(r'^#define NVN_STAGE_.*$', draw, re.M)) + '\n')
parts = [function(replay, 'vk_exec_pool_requirement_begin'), function(draw, 'vk_exec_ahead_op'), function(replay, 'ahead_set_rt'),
         function(replay, 'pool_leader_draw'), function(replay, 'pool_leader_run')]
(work / 'pool_leader_under_test.h').write_text('// ' + marker + '\n' + '\n\n'.join(parts))
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get('PATH', ''))
exe = work / 'pool_requirement_check.exe'
subprocess.run([str(CC), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function',
    '-Wno-missing-field-initializers', '-fsanitize=undefined', '-fsanitize-trap=all',
    '-I', str(ROOT / 'runtime'), '-I', str(work), '-I', str(CC.parent.parent / 'include/vma'),
    str(ROOT / 'tests/pool_requirement_check.c'), '-o', str(exe)], env=env, check=True)
subprocess.run([str(exe)], env=env, check=True)

"""Exercise actual depth/stencil clear decisions at the canonical leader boundary."""
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

work = ROOT / 'build/test/clear-order'
work.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'tools'))
from check_content import GENERATED_MARKER
marker = GENERATED_MARKER.decode() if isinstance(GENERATED_MARKER, bytes) else GENERATED_MARKER
source = (ROOT / 'runtime/vk_exec.c').read_text()
parts = [function(source, name) for name in ('vk_exec_scissor', 'whole_target', 'clear_attachment', 'clear_ds')]
(work / 'clear_under_test.h').write_text('// ' + marker + '\n' + '\n\n'.join(parts))
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get('PATH', ''))
exe = work / 'clear_order_check.exe'
subprocess.run([str(CC), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function',
    '-Wno-missing-field-initializers', '-fsanitize=undefined', '-fsanitize-trap=all',
    '-I', str(ROOT / 'runtime'), '-I', str(work), '-I', str(CC.parent.parent / 'include/vma'),
    str(ROOT / 'tests/clear_order_check.c'), '-o', str(exe)], env=env, check=True)
subprocess.run([str(exe)], env=env, check=True)

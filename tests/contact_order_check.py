"""Exercise production contact-shadow hooks across a canonical leader pass boundary."""
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

work = ROOT / 'build/test/contact-order'
work.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'tools'))
from check_content import GENERATED_MARKER
marker = GENERATED_MARKER.decode() if isinstance(GENERATED_MARKER, bytes) else GENERATED_MARKER
source = (ROOT / 'runtime/vk_contact.c').read_text()
state = source[source.index('static struct {'):source.index('} C;') + 4]
parts = [function(source, name) for name in ('is_gbuffer', 'mem_barrier', 'vk_contact_pending',
                                          'vk_contact_flush', 'is_composite', 'vk_contact_pass_begin')]
(work / 'contact_under_test.h').write_text('// ' + marker + '\n' + state + '\n' + '\n\n'.join(parts))
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get('PATH', ''))
exe = work / 'contact_order_check.exe'
subprocess.run([str(CC), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function',
    '-Wno-missing-field-initializers', '-fsanitize=undefined', '-fsanitize-trap=all',
    '-I', str(ROOT / 'runtime'), '-I', str(work), '-I', str(CC.parent.parent / 'include/vma'),
    str(ROOT / 'tests/contact_order_check.c'), '-o', str(exe)], env=env, check=True)
subprocess.run([str(exe)], env=env, check=True)

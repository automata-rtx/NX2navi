"""Exercise production protection transitions without guest memory or a GPU."""
from pathlib import Path
import os
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ['CC'])

def function(source, name):
    mask = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"', lambda m: ' ' * len(m[0]), source)
    m = re.search(r'(?m)^(?:static\s+)?(?:[\w*]+\s+)+' + name + r'\([^;]*?\)\s*\{', mask)
    assert m, name
    i = mask.index('{', m.start()) + 1
    depth = 1
    while depth:
        depth += (mask[i] == '{') - (mask[i] == '}')
        i += 1
    return source[m.start():i]

work = ROOT / 'build/test/memory-events'
work.mkdir(parents=True, exist_ok=True)
source = (ROOT / 'runtime/memory.c').read_text()
names = ('protection_publish', 'mem_protection_serial', 'mem_protection_region',
         'mem_protection_range_unchanged', 'mem_protection_events', 'mem_frozen_stable',
         'thaw_begin_locked', 'thaw_locked', 'thaw_end_locked', 'mem_freeze', 'heap_lazy_commit_locked')
sys.path.insert(0, str(ROOT / 'tools'))
from check_content import GENERATED_MARKER
marker = GENERATED_MARKER.decode() if isinstance(GENERATED_MARKER, bytes) else GENERATED_MARKER
(work / 'memory_events_under_test.h').write_text('// ' + marker + '\n' + '\n\n'.join(function(source, n) for n in names))
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get('PATH', ''))
exe = work / 'memory_events_check.exe'
subprocess.run([str(CC), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror',
                '-Wno-unused-function', '-Wno-missing-field-initializers', '-fsanitize=undefined', '-fsanitize-trap=all',
                '-I', str(ROOT / 'runtime'), '-I', str(work), str(ROOT / 'tests/memory_events_check.c'),
                '-o', str(exe)], env=env, check=True)
subprocess.run([str(exe)], env=env, check=True)

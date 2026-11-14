"""Check the production snapshot and texture-absence guards without a device or guest files."""
from pathlib import Path
import os
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ['CC'])

def function(source, name):
    mask = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"', lambda m: ' ' * len(m[0]), source)
    match = re.search(r'(?m)^(?:static\s+)?(?:[\w*]+\s+)+' + name + r'\([^;]*?\)\s*\{', mask)
    assert match, name
    end = mask.index('{', match.start()) + 1
    depth = 1
    while depth:
        depth += (mask[end] == '{') - (mask[end] == '}')
        end += 1
    return source[match.start():end]

work = ROOT / 'build/test/resource-consumers'
work.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'tools'))
from check_content import GENERATED_MARKER
marker = GENERATED_MARKER.decode() if isinstance(GENERATED_MARKER, bytes) else GENERATED_MARKER
memory = (ROOT / 'runtime/vk_memory.c').read_text()
texture = (ROOT / 'runtime/vk_texture.c').read_text()
names = ('snap_lease_valid', 'vk_mem_dependency_sleep_locked', 'snap_current', 'vk_mem_dependency_adopt_locked',
         'vk_mem_dependency_region_locked', 'snap_regions', 'vk_mem_unchanged_why_locked',
         'vk_mem_snap_range_locked')
selected = '\n\n'.join(function(memory, name) for name in names)
selected += '\n\n' + '\n\n'.join(function(texture, name) for name in ('slot_storage_readable', 'slot_absence_valid'))
selected += '\n\n' + '\n\n'.join(function(texture, name) for name in ('pool_deps_wake', 'pool_deps_adopt', 'pool_deps_observe'))
(work / 'resource_consumers_under_test.h').write_text('// ' + marker + '\n' + selected)
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get('PATH', ''))
for name in ('resource_consumers_check', 'fresh_deps_poll_check', 'fresh_deps_check', 'resource_events_check',
             'resource_events_threads_check'):
    exe = work / (name + '.exe')
    extra = [str(ROOT / 'runtime/fresh_deps.c')] if name in ('fresh_deps_check', 'resource_consumers_check') else []
    subprocess.run([str(CC), '-std=gnu11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                    '-Wno-unused-function', '-Wno-unused-variable', '-fsanitize=undefined', '-fsanitize-trap=all',
                    '-I', str(ROOT / 'runtime'), '-I', str(work), str(ROOT / ('tests/' + name + '.c')),
                    *extra, '-o', str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

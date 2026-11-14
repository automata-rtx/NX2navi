"""Exercise production query bookkeeping with deterministic GPU and completion boundaries."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--scheduler-source', type=Path)
parser.add_argument('--require-counter-prefix', action='store_true')
args = parser.parse_args()
source = args.source.resolve()
root = Path(__file__).resolve().parents[1]
work = root / 'build/test/query-order'
work.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(source / 'tools'))
from check_content import GENERATED_MARKER
marker = GENERATED_MARKER.decode() if isinstance(GENERATED_MARKER, bytes) else GENERATED_MARKER

def function(text, name):
    masked = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"', lambda m: ' ' * len(m[0]), text)
    match = re.search(r'(?m)^(?:static\s+)?(?:[\w*]+\s+)+\*?' + name + r'\([^;]*?\)\s*\{', masked)
    assert match, name
    end = masked.index('{', match.start()) + 1
    depth = 1
    while depth:
        depth += (masked[end] == '{') - (masked[end] == '}')
        end += 1
    return text[match.start():end]

xfer_path = source / 'runtime/vk_exec_xfer.c'
xfer_hash = hashlib.sha256(xfer_path.read_bytes()).hexdigest()
xfer = xfer_path.read_text()
# Keep original declarations and every implementation through query reset, including
# batching and pending-record publication. No query algorithm is reimplemented here.
prefix = xfer[:xfer.index('// ---- event order check')]
extra = ['report_counter', 'counter_word', 'last_type', 'cond_enabled',
         'vk_exec_xfer_front', 'xfer_worker', 'op_xfer_impl']
stubbed = ['copy_b2b', 'copy_buffer_image', 'copy_t2t', 'clear_tex', 'signal_event', 'clear_buf']
generated = '// ' + marker + '\n' + prefix + '\n'
generated += '\n'.join('#define ' + name + ' unexpected_transfer' for name in stubbed) + '\n'
generated += '\n\n'.join(function(xfer, name) for name in extra) + '\n'
generated += '\n'.join('#undef ' + name for name in stubbed) + '\n'
(work / 'query_under_test.h').write_text(generated)
replay_path = (args.scheduler_source.resolve() if args.scheduler_source else source) / 'runtime/vk_exec.c'
replay_hash = hashlib.sha256(replay_path.read_bytes()).hexdigest()
replay = replay_path.read_text()
chunk = re.search(r'^#define CHUNK_DRAWS\s+\d+.*$', replay, re.M)
assert chunk
(work / 'query_leader_under_test.h').write_text('// ' + marker + '\n' + chunk[0] + '\n' +
    function(replay, 'pool_leader_draw') + '\n' + function(replay, 'pool_leader_run'))
cc = Path(os.environ['CC'])
exe = work / 'query_order_check.exe'
command = [str(cc), '-std=gnu11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
           '-Wno-unused-function', '-Wno-unused-parameter', '-Wno-missing-field-initializers',
           '-fsanitize=undefined', '-fsanitize-trap=all', '-ffunction-sections', '-fdata-sections',
           '-I', str(source / 'runtime'), '-I', str(work), '-I', str(cc.parent.parent / 'include/vma'),
           str(root / 'tests/query_order_check.c'), '-Wl,--gc-sections', '-o', str(exe)]
if args.require_counter_prefix:
    command.insert(1, '-DREQUIRE_COUNTER_PREFIX=1')
env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get('PATH', ''))
subprocess.run(command, env=env, check=True)
result = subprocess.run([str(exe)], env=env, text=True, capture_output=True)
print(result.stdout, end='')
print(result.stderr, end='', file=sys.stderr)
assert hashlib.sha256(xfer_path.read_bytes()).hexdigest() == xfer_hash
assert hashlib.sha256(replay_path.read_bytes()).hexdigest() == replay_hash
report = dict(source=str(source), source_sha256=xfer_hash,
              splitter_source=str(replay_path), splitter_sha256=replay_hash,
              required_counter_prefix=args.require_counter_prefix,
              returncode=result.returncode, output=result.stdout + result.stderr,
              runtime_modified=False, native_game_launched=False)
(work / 'result.json').write_text(json.dumps(report, indent=2))
raise SystemExit(result.returncode)

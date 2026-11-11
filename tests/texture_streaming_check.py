"""Check actual texture-streaming functions with fake clocks and GPU feedback; no GPU or game files."""
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def function(source, signature):
    start = source.index(signature + ' {')
    return source[start:source.index('\n}', start) + 2]


def main():
    source = (ROOT / 'runtime/vk_texture.c').read_text()
    work = ROOT / 'build/test/texture_streaming'
    work.mkdir(parents=True, exist_ok=True)
    signatures = ['static void make_key(const tex_t *t, TexKey *k)',
                  'void vk_pool_marks_read(const uint32_t *marks, uint32_t n)',
                  'static int want_take(void)',
                  'void vk_texture_create_budget_locked(int n, uint64_t deadline)',
                  'static int create_allowed(TexSlot *s)',
                  'int vk_texture_pending_count_locked(const uint32_t *ids, uint32_t n)']
    (work / 'texture_functions.inc').write_text('\n\n'.join(function(source, f) for f in signatures))
    draw = (ROOT / 'runtime/vk_exec_draw.c').read_text()
    (work / 'pool_function.inc').write_text(
        function(draw, 'static void pool_pending_note(uint32_t id, uint64_t token, uint32_t mark)') + '\n' +
        function(draw, 'static void pool_unres(uint32_t id, int got)') + '\n' +
        function(draw, 'static int pool_refill(VkExec *ex, uint32_t id, uint32_t used, uint64_t t, int spent)') + '\n' +
        function(draw, 'static void pool_dirty_slot(int id, void *arg)') + '\n' +
        function(draw, 'static void pool_sync_entries(VkExec *ex)'))
    (work / 'dirty_function.inc').write_text(function(source,
        'int vk_texture_dirty_slots_locked(uint64_t *pos, void (*fn)(int id, void *arg), void (*gfn)(int g, void *arg), void *arg)'))
    cc = os.environ.get('CC') or shutil.which('clang')
    if not cc:
        raise RuntimeError('Clang is required; set CC to the UCRT64 Clang executable')
    for name in ('texture_streaming_check', 'texture_pool_retry_check', 'texture_dirty_check'):
        exe = work / (name + ('.exe' if os.name == 'nt' else ''))
        subprocess.run([cc, '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror', '-I', str(ROOT / 'runtime'),
                        '-I', str(work), str(ROOT / 'tests' / (name + '.c')), '-o', str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == '__main__':
    main()

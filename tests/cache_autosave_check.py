"""Exercise the runtime's autosave entry point without a Vulkan device or game data."""
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = (ROOT / 'runtime/vk_pipeline.c').read_text()

    def function(signature):
        start = source.index(signature + ' {')
        end = source.index('\n}', start) + 2
        return source[start:end]

    work = ROOT / 'build/test/cache_autosave'
    work.mkdir(parents=True, exist_ok=True)
    harness = work / 'cache_autosave_check.c'
    harness.write_text(r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
static struct { int ready; } P;
static struct { uint32_t pend_n; } K;
static uint64_t now;
static unsigned keys, builds, uses, answers, snapshots;
#define SAVE_WAIT_MS 5000
static uint64_t plat_now_ms(void) { return now; }
static void keys_flush(void) { keys++; K.pend_n = 0; }
static void pb_flush(void) { builds++; }
static void cu_flush(void) { uses++; }
static void gfx_shader_record_flush(int force) { assert(!force); answers++; }
static void save_with(uint32_t wait_ms) { assert(wait_ms == SAVE_WAIT_MS); snapshots++; }
''' + function('static void autosave_once(void)') + '\n' +
                       function('void vk_pipeline_save(void)') + r'''
int main(void) {
    now = 10000;
    autosave_once();
    assert(!keys && !builds && !uses && !answers && !snapshots);
    P.ready = 1;
    for (now = 10000; now <= 130000; now += 500) {
        K.pend_n = 1;
        autosave_once();
    }
    assert(keys == 61 && builds == 61 && uses == 61 && answers == 241);
    assert(snapshots == 0);
    vk_pipeline_save();
    assert(snapshots == 1);
    puts("Cache autosave checks passed: records persist during play, full snapshots require an explicit save");
    return 0;
}
''')
    cc = os.environ.get('CC') or shutil.which('clang')
    if not cc:
        raise RuntimeError('Clang is required; set CC to the UCRT64 Clang executable')
    exe = work / ('cache_autosave_check.exe' if os.name == 'nt' else 'cache_autosave_check')
    subprocess.run([cc, '-O1', '-Wall', '-Wextra', '-Werror', str(harness), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


if __name__ == '__main__':
    main()

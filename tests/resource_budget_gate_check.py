"""Exercise production pipeline admission with controlled usage samples; no Vulkan or game process."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ.get('CC', r'C:\msys64\ucrt64\bin\clang.exe'))
if not CC.is_file():
    CC = Path(r'C:\Users\Andrew\AppData\Local\Programs\2NX2Navi\toolchain\msys64\ucrt64\bin\clang.exe')
sys.path.insert(0, str(ROOT / 'tools'))
from check_content import GENERATED_MARKER

source = (ROOT / 'runtime/vk_pipeline.c').read_text()
begin = source.index('#define GATE_BIG_MB')
end = source.index('// The gate\'s numbers', begin)
production = source[begin:end]
fixture = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "resource_budget.h"
typedef int PlatMutex;
#define PLAT_MUTEX_INIT 0
static int locked, samples;
static double fake_now, actual_used = 29838, actual_limit = 58517;
static void plat_mutex_lock(PlatMutex *lock) { (void)lock; assert(!locked); locked = 1; }
static void plat_mutex_unlock(PlatMutex *lock) { (void)lock; assert(locked); locked = 0; }
static double now_ms(void) { return fake_now; }
static const char *cfg_value(const char *name) {
    return !strcmp(name, "NX2NAVI_BUILD_FREE_MB") ? "4096" : "64";
}
static const char *cfg_get(const char *name) { (void)name; return NULL; }
static int plat_commit_enforced(void) { return 1; }
static void plat_commit_info(double *proc, double *used, double *limit) {
    assert(locked); samples++; *proc = 0; *used = actual_used; *limit = actual_limit;
}
''' + production + r'''
int main(void) {
    double held;
    // The captured failure: thousands of cache-hit builds finish inside one 100ms interval. Their peak estimates
    // must trigger fresh observation, not terminate prewarming while actual usage remains comfortably below budget.
    for (int i = 0; i < 2000; i++) {
        assert(gate_enter(0, &held)); assert(held == 64); gate_leave(held);
    }
    assert(samples > 1 && samples < 10 && !GATE.refused && !GATE.g.active.reserved_mb);
    // In-flight reservations survive a denial refresh. Failed first admission cannot reserve twice.
    resource_budget_sample(&GATE.g);
    GATE.g.completed_mb = 22000;
    GATE.g.active.reserved_mb = 1152;
    int before = samples;
    assert(gate_enter(0, &held));
    assert(samples == before + 1 && !GATE.g.completed_mb && GATE.g.active.reserved_mb == 1216);
    gate_leave(held);
    // Genuine retained growth stays denied after exactly one fresh sample, preserving active reservations.
    actual_used = actual_limit - GATE.g.active.free_mb - GATE.g.active.reserved_mb;
    GATE.g.completed_mb = 22000;
    before = samples;
    assert(!gate_enter(0, &held));
    assert(samples == before + 1 && GATE.refused == 1 && GATE.g.active.reserved_mb == 1152);
    assert(!GATE.g.completed_mb);
    before = samples;
    assert(!gate_enter(0, &held)); assert(samples == before); // no repeated sample when settled debt is gone
    // A changed system limit recalculates the shared floor on the immediate fresh sample.
    actual_limit = 80000; actual_used = 70000; GATE.g.completed_mb = 1000;
    assert(gate_enter(0, &held)); assert(GATE.g.active.free_mb == 8250);
    gate_leave(held);
    // Unknown limits and explicit gate disable still make progress without unnecessary denial retries.
    fake_now += 100; actual_limit = -1; actual_used = 1e9;
    assert(gate_enter(0, &held)); gate_leave(held);
    fake_now += 100; actual_limit = 58517; GATE.off = 1;
    assert(gate_enter(0, &held)); gate_leave(held);
    assert(!locked);
    puts("resource budget production gate: passed");
    return 0;
}
'''
out = ROOT / 'build/test/resource-budget-gate'
out.mkdir(parents=True, exist_ok=True)
marker = GENERATED_MARKER.decode() if isinstance(GENERATED_MARKER, bytes) else GENERATED_MARKER
src = out / 'gate.c'
src.write_text('// ' + marker + '\n' + fixture)
exe = out / 'gate.exe'
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get('PATH', ''))
subprocess.run([str(CC), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=undefined', '-fsanitize-trap=all', '-I', str(ROOT / 'runtime'),
                str(src), '-o', str(exe)], env=env, check=True)
subprocess.run([str(exe)], env=env, check=True)

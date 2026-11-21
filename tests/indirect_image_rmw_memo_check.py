"""Check that indirect image eligibility uses one source proof per immutable shader."""
from pathlib import Path
import os
import subprocess
import tempfile

from bind_recipe_check import function

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('NX2_RMW_SOURCE', ROOT / 'runtime/gfx_shader.c'))
CC = Path(os.environ['CC'])

PREFIX = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

enum {GFX_STAGE_FS, GFX_STAGE_CS, RC_EFFECT_WRITES_IMAGE = 1, RC_EFFECT_WRITES_BUFFER = 2};
typedef struct GfxShader {
    int stage, write_inventory_complete;
    uint32_t guest_writes;
    const uint32_t *spirv;
    uint64_t words;
    uint16_t indirect_image_rmw;
} GfxShader;
typedef struct {unsigned count; struct {uint8_t binding;} entries[1];} RcNativeImageSeedBindings;
typedef int PlatMutex;
static PlatMutex cache_lock;
static unsigned lock_count, parses;
static void plat_mutex_lock(PlatMutex *lock) {assert(lock == &cache_lock && !*lock); *lock = 1; ++lock_count;}
static void plat_mutex_unlock(PlatMutex *lock) {assert(lock == &cache_lock && *lock); *lock = 0;}
static int rc_native_image_seed_rmw_bindings(const uint32_t *words, size_t count, RcNativeImageSeedBindings *proof) {
    assert(cache_lock && words && count == 1);
    ++parses;
    switch (words[0]) {
    case 0: return 0;
    case 1: *proof = (RcNativeImageSeedBindings){.count = 2}; return 1;
    default: *proof = (RcNativeImageSeedBindings){.count = 1, .entries = {{words[0] - 2}}}; return 1;
    }
}
'''

TEST = r'''
int main(void) {
    const uint32_t positive = 2, highest_binding = 257, refused = 0, ambiguous = 1;
    GfxShader ok = {.stage=GFX_STAGE_CS, .guest_writes=RC_EFFECT_WRITES_IMAGE, .write_inventory_complete=1,
                    .spirv=&positive, .words=1};
    GfxShader highest = ok, no = ok, multi = ok, fragment = ok, incomplete = ok, mixed = ok;
    highest.spirv = &highest_binding; no.spirv = &refused; multi.spirv = &ambiguous;
    fragment.stage = GFX_STAGE_FS; incomplete.write_inventory_complete = 0; mixed.guest_writes |= RC_EFFECT_WRITES_BUFFER;
    for (unsigned i = 0; i < 4096; ++i) {
        assert(gfx_shader_indirect_image_rmw_binding(&ok) == 0);
        assert(gfx_shader_indirect_image_rmw_binding(&no) == -1);
        assert(gfx_shader_indirect_image_rmw_binding(&highest) == 255);
        assert(gfx_shader_indirect_image_rmw_binding(&multi) == -1);
        assert(gfx_shader_indirect_image_rmw_binding(&fragment) == -1);
        assert(gfx_shader_indirect_image_rmw_binding(&incomplete) == -1);
        assert(gfx_shader_indirect_image_rmw_binding(&mixed) == -1);
        assert(gfx_shader_indirect_image_rmw_binding(NULL) == -1);
    }
    assert(parses == 4 && lock_count == 4 && !cache_lock);
    puts("Indirect storage-image entry: one parse and cold lock per eligible immutable shader");
}
'''

with tempfile.TemporaryDirectory(prefix='indirect-rmw-memo-') as tmp:
    c = Path(tmp) / 'check.c'
    exe = c.with_suffix('.exe')
    c.write_text(PREFIX + function(SOURCE.read_text(encoding='utf-8'), 'gfx_shader_indirect_image_rmw_binding') +
                 TEST, encoding='utf-8')
    env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get('PATH', ''))
    subprocess.run([str(CC), '-std=gnu11', '-Wall', '-Wextra', '-Werror', '-O1', '-fsanitize=undefined',
                    '-fsanitize-trap=all', str(c), '-o', str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)


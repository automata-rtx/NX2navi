// The runtime functions are compiled verbatim by texture_streaming_check.py.
// Only the image allocation boundary and feedback state are replaced here.
#include <assert.h>
#include <stdio.h>
#include "nvn_obj.h"
#include "nvn_hmap.h"

typedef struct { uint64_t addr; nvn_texdesc d; uint8_t has_layout, layout[8], pad[7]; } TexKey;
typedef struct { uint64_t samp_frame; int retiring; } Tex;
typedef struct {
    tex_t t;
    Tex *ce;
    uint32_t cgen;
    uint8_t used, cheld, cwant;
} TexSlot;
static TexSlot tslots[1024];
static int ntslots;
static uint32_t sinfo[1024], agen[65536];
static struct { uint64_t frames; } gfx_counters;
static nvn_hmap by_key = {.klen = sizeof(TexKey)}, holds = {.klen = sizeof(TexKey)};
static uint64_t n_disuse_back, marks_peak, clock_ns;
static uint32_t marks_reads, marks_live, bumps;
static int create_budget, create_deferred;
static uint64_t create_deadline;
#define WANT_PER_FRAME 32
static void lock(void) {}
static void unlock(void) {}
static void retire_cancel(Tex *e) { e->retiring = 0; }
static void tex_gen_bump(uint64_t address) { assert(address); bumps++; }
static uint64_t kern_now_ns(void) { return clock_ns; }
#include "texture_functions.inc"

int main(void) {
    ntslots = 1;
    tslots[0].used = 1;
    tslots[0].t.mem = (void *)(uintptr_t)0x10000;
    uint32_t marks[1024] = {1}, pending[1024] = {0};
    gfx_counters.frames = 10;
    vk_pool_marks_read(marks, 1);
    assert(tslots[0].cwant && bumps == 1 && n_disuse_back == 0);
    vk_pool_marks_read(marks, 1);
    assert(bumps == 1); // repeated feedback must not invalidate the table again

    TexKey key;
    make_key(&tslots[0].t, &key);
    nvn_hm_put(&holds, &key, (void *)(uintptr_t)1);
    tslots[0].cheld = 1;
    vk_pool_marks_read(marks, 1);
    assert(!holds.n && !tslots[0].cheld && tslots[0].cwant && n_disuse_back == 1);

    Tex live = {.retiring = 1};
    nvn_hm_put(&by_key, &key, &live);
    vk_pool_marks_read(marks, 1);
    assert(!live.retiring && live.samp_frame == 10);
    tslots[0].ce = &live;
    gfx_counters.frames = 11;
    vk_pool_marks_read(marks, 1);
    assert(live.samp_frame == 11);
    assert(marks_reads == 5 && marks_live == 0);

    // Hundreds of aliases must not cross the 512-image loading threshold.
    ntslots = 700;
    for (int i = 1; i < ntslots; i++) {
        tslots[i] = tslots[0];
        tslots[i].t.d.swizzle_r = (unsigned)i & 7;
        pending[i] = (uint32_t)i;
    }
    assert(vk_texture_pending_count_locked(pending, 700) == 1);
    tslots[1].t.d.width = 64;
    tslots[2].t.d.target = 5;
    tslots[3].t.d.format = 7;
    tslots[4].t.layout[0] = 1;
    tslots[5].t.mem = (void *)(uintptr_t)0x20000;
    assert(vk_texture_pending_count_locked(pending, 700) == 6);
    pending[1] = 1024; // absent or invalid IDs are ignored
    tslots[2].used = 0;
    assert(vk_texture_pending_count_locked(pending, 700) == 4);
    assert(vk_texture_pending_count_locked(pending, 0) == 0);

    // A single walk must stop ordinary creates at its deadline, even with an
    // effectively unlimited count budget. Cached lookups do not use this gate.
    TexSlot unseen = {0};
    vk_texture_create_budget_locked(1 << 20, 6000000);
    clock_ns = 5999999;
    assert(create_allowed(&unseen));
    clock_ns = 6000000;
    assert(!create_allowed(&unseen));
    vk_texture_create_budget_locked(0, 0);
    assert(!create_allowed(&unseen));
    TexSlot wanted = {.cwant = 1};
    for (int i = 0; i < WANT_PER_FRAME; i++) assert(create_allowed(&wanted));
    assert(!create_allowed(&wanted));
    gfx_counters.frames++;
    assert(create_allowed(&wanted));
    clock_ns += 2000000;
    assert(!create_allowed(&wanted)); // priority recovery is time-bounded too
    gfx_counters.frames++;
    assert(create_allowed(&wanted)); // a slow image cannot starve later frames
    vk_texture_create_budget_locked(-1, 1);
    assert(create_allowed(&unseen)); // direct resource use stays unrestricted
    vk_texture_create_budget_locked(16, 0);
    assert(create_allowed(&unseen)); // explicit time-budget disable
    puts("Texture streaming checks passed: first-use recovery, deduplicated backlog, per-create deadline");
    return 0;
}

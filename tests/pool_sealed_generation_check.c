// The actual pool_current, pool_take_set, publication and pool_sync functions are extracted verbatim. External
// resource refresh and GPU fence services are fakes; tests prove the no-op branch does not bypass those services.
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef uintptr_t VkDescriptorSet;
typedef struct { uint64_t fence; } VkExec;
#include "vk_pool_publication.h"
static struct {
    int cur, nsets, sealed, ready, retry, layout, publication_exhausted;
    uint32_t kinds, gen, epoch, sgen;
    uint64_t cur_fence, set_fence[2], set_synced[2], completed_generation, logn, frame, frame_ns, sync_ns, catch_ns, waits;
    VkDescriptorSet set, sets[2];
} PL;
static uint64_t pool_ready_generation, pool_takes, ticks;
static uint32_t vk_pool_kinds_used, gen, ep, sg;
static struct { uint64_t frames; } gfx_counters;
static unsigned refreshes, catches, fence_waits;
typedef struct { uint64_t (*fence_done)(void); void (*fence_wait)(uint64_t, uint64_t); } GfxBackend;
static uint64_t fence_done(void) { return 100; }
static void fence_wait(uint64_t f, uint64_t timeout) {
    assert(pool_ready_generation == 0 && f > 100 && timeout == UINT64_MAX);
    fence_waits++;
}
static const GfxBackend *gfx_backend(void) {
    static GfxBackend b = {fence_done, fence_wait};
    return &b;
}
static uint32_t vk_texture_slots_gen(void) { return gen; }
static uint32_t vk_texture_epoch(void) { return ep; }
static uint32_t vk_sampler_gen(void) { return sg; }
static void vk_pool_set_layout(void) { PL.layout = 1; }
static uint64_t kern_now_ns(void) { return ++ticks; }
static void vk_texture_lock(void) { assert(!pool_ready_generation); }
static void vk_texture_unlock(void) { assert(!pool_ready_generation); }
static void pool_catch_up(uint64_t from, uint64_t to) {
    (void)from; (void)to;
    assert(!pool_ready_generation);
    catches++;
}
static void pool_sync_entries(VkExec *ex) {
    (void)ex;
    assert(!pool_ready_generation && !PL.sealed);
    refreshes++;
    PL.ready = 1, PL.retry = 0, PL.kinds = vk_pool_kinds_used, PL.gen = gen, PL.epoch = ep, PL.sgen = sg;
}
#include "pool_sealed_under_test.h"

static void reset(void) {
    memset(&PL, 0, sizeof PL);
    PL.nsets = 2, PL.sealed = PL.ready = PL.layout = 1;
    PL.kinds = vk_pool_kinds_used = 1;
    PL.gen = gen = 2, PL.epoch = ep = 3, PL.sgen = sg = 4;
    PL.sets[0] = PL.set = 11, PL.sets[1] = 22;
    PL.cur_fence = PL.set_fence[0] = 8;
    PL.completed_generation = pool_ready_generation = 10;
    pool_takes = refreshes = catches = fence_waits = 0;
    gfx_counters.frames = 1;
}

int main(void) {
    VkExec ex = {.fence = 8};
    VkExecPoolView group = {0};
    reset();
    assert(vk_pool_view_publish(&group, PL.set, ex.fence, 10, 1, 0));
    VkExecPoolView before = group;
    for (int i = 0; i < 100; i++) {
        pool_sync(&ex);
        VkDescriptorSet set = 0;
        assert(vk_pool_view_get(&group, ex.fence, 1, vk_exec_pool_generation(), &set) && set == 11);
    }
    assert(PL.sealed && PL.cur == 0 && PL.completed_generation == 10 && PL.set_fence[0] == 8);
    assert(!refreshes && !catches && !fence_waits && !pool_takes && !memcmp(&group, &before, sizeof group));
    // Each production pool_current input invalidates the sealed snapshot; fence change also requires a new set.
    for (int invalidator = 0; invalidator < 7; invalidator++) {
        reset(); ex.fence = 8;
        switch (invalidator) {
        case 0: PL.ready = 0; break;
        case 1: PL.retry = 1; break;
        case 2: vk_pool_kinds_used = 3; break;
        case 3: gen++; break;
        case 4: ep++; break;
        case 5: sg++; break;
        case 6: ex.fence++; break;
        }
        PL.set_fence[1] = 101;
        pool_sync(&ex);
        assert(!PL.sealed && PL.cur == 1 && PL.set == 22 && PL.cur_fence == ex.fence);
        assert(refreshes == 1 && catches == 1 && fence_waits == 1 && vk_exec_pool_generation() == 11);
        VkDescriptorSet set = 0;
        assert(!vk_pool_view_get(&group, ex.fence, vk_pool_kinds_used, vk_exec_pool_generation(), &set));
    }
    // Ordinary unsealed no-work handling is deliberately unchanged and still advances the generation.
    reset(); ex.fence = 8; PL.sealed = 0;
    pool_sync(&ex);
    assert(refreshes == 1 && !catches && vk_exec_pool_generation() == 11);
    reset(); PL.nsets = 1; gen++;
    pool_sync(&ex);
    assert(!PL.sealed && refreshes == 1 && !catches && vk_exec_pool_generation() == 11);
    reset(); pool_ready_generation = 0;
    pool_sync(&ex); assert(vk_exec_pool_generation() == 11 && !refreshes);
    reset(); pool_ready_generation = 9;
    pool_sync(&ex); assert(vk_exec_pool_generation() == 11 && !refreshes);
    reset(); PL.completed_generation = pool_ready_generation = UINT64_MAX;
    pool_sync(&ex); assert(!vk_exec_pool_generation() && PL.publication_exhausted);
    reset(); PL.publication_exhausted = 1;
    pool_sync(&ex); assert(!vk_exec_pool_generation() && PL.publication_exhausted);
    puts("Sealed pool generation: unchanged publication preserved; all current-state invalidators, ring waits, unsealed path, zero/mismatch/max/exhaustion passed");
    return 0;
}

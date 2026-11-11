// Compile the real pool synchronization path with deterministic descriptor and
// allocation responses. No Vulkan device, allocations or guest data are needed.
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct { int unused; } VkExec;
typedef uintptr_t VkSampler;
#define VK_POOL_TEX 1024
#define VK_POOL_SAMP 1
#define VK_POOL_KINDS 16
#define VK_NULL_HANDLE 0
#define POOL_LOAD_END 64
enum { VK_TS_POOL, VK_TS_POOL_LOCK, VK_TS_POOL_RETRY };
static struct {
    uint32_t kinds, gen, epoch, sgen;
    uint64_t frame_ns, frame, dirty_pos, n_frame_stops, n_refill, n_by_reg, n_by_addr, n_fresh, n_slots, n_defer_skips;
    int ready, retry, all;
    int gmode, unres_more, nunres, cunres, *unres;
    uint32_t unres_mg;
    uint8_t unres_on[VK_POOL_TEX], unres_in[VK_POOL_TEX];
    uint64_t n_visit, n_walk[2], walk_visit[2], walk_refill[2], walk_ns[2];
    uint64_t n_ep_ent, n_unres_walks, n_ep, ep_ns, unres_ns, n_full, full_ns;
    uint64_t tok[VK_POOL_TEX];
    VkSampler samp[VK_POOL_SAMP];
} PL;
static struct { VkSampler samp; } Z;
static uint32_t pool_defer[VK_POOL_TEX], vk_pool_kinds_used;
static uint64_t pool_defer_tok[VK_POOL_TEX];
static uint32_t pool_pending[VK_POOL_TEX], pool_pending_pos[VK_POOL_TEX], pool_pending_n;
static int32_t pool_loading;
static uint64_t pool_load_made, pool_load_frame0;
typedef struct { VkExec *ex; uint32_t used; int n, retry, spent; } PoolDirty;
static int slot_count = 1, fill_ok, distinct_missing = 1;
static int dirty_ok, dirty_id = -1, token_reads, fills, fresh_checks, list_checks;
static int fast_mode = 1;
static uint64_t token = 42;
static uint64_t seen_deadline;
static uint32_t vk_texture_slots_gen(void) { return 1; }
static uint32_t vk_texture_epoch(void) { return 1; }
static uint32_t vk_sampler_gen(void) { return 1; }
static uint64_t pool_create_ns(void) { return 6000000; }
static int pool_create_budget(void) { return pool_loading ? 1 << 20 : 16; }
static uint64_t kern_now_ns(void) { return 1000000; }
static void vk_texture_lock(void) {}
static void vk_texture_unlock(void) {}
static void vk_texture_pool_ctx(int x) { (void)x; }
static int vk_texture_slot_count_locked(void) { return slot_count; }
static int vk_sampler_slot_count_locked(void) { return 0; }
static VkSampler vk_sampler_by_id_locked(int id) { (void)id; return 0; }
static uint64_t vk_texture_slot_token(int id) { (void)id; token_reads++; return token; }
static int pool_fill(VkExec *ex, uint32_t id, uint32_t used, uint64_t t) {
    (void)ex; (void)id; (void)used; (void)t; fills++; return fill_ok;
}
static void pool_flush(void) {}
static void pool_write(uint32_t b, uint32_t id, uintptr_t v, VkSampler s) { (void)b; (void)id; (void)v; (void)s; }
static void vk_texture_create_budget_locked(int n, uint64_t deadline) { if (n >= 0) seen_deadline = deadline; }
static int vk_texture_dirty_slots_locked(uint64_t *p, void (*fn)(int, void *), void (*gfn)(int, void *), void *arg) {
    (void)p;
    (void)gfn;
    if (dirty_ok && dirty_id >= 0) fn(dirty_id, arg);
    return dirty_ok;
}
static uint64_t vk_texture_dirty_pos_locked(void) { return 1; }
static void vk_texture_by_id_locked(int id, void *a, void *b) { (void)id; (void)a; (void)b; fresh_checks++; }
static int pool_load_backlog(void) { return 512; }
static int vk_texture_pending_count_locked(const uint32_t *p, uint32_t n) { (void)p; (void)n; return distinct_missing; }
static void rt_log(const char *s, ...) { (void)s; }
static void vk_texstat_time(int k, uint64_t t) { (void)k; (void)t; }
static void vk_texstat_add(int k, uint64_t n, uint64_t b) { (void)k; (void)n; (void)b; }
static int pool_group(void) { return 0; }
static int pool_fast(void) { return fast_mode; }
static void pool_dirty_group(int g, void *arg) { (void)g; (void)arg; }
static void vk_texture_groups_unstamp_locked(void) {}
static int vk_texture_pool_fresh_locked(void) { list_checks++; return 1; }
static uint32_t mem_map_gen(void) { return 1; }
static int vk_texture_create_deferred_locked(void) { return 0; }
static int vk_texture_slot_retry_locked(int id) { (void)id; return 0; }
static void rt_fatal(const char *s) { (void)s; abort(); }
#include "pool_function.inc"

int main(void) {
    VkExec ex = {0};
    PL.ready = 1;
    PL.gen = PL.epoch = PL.sgen = 1;
    PL.kinds = 1;
    PL.tok[0] = 42;
    PL.frame = 7;
    PL.frame_ns = 6000000;
    vk_pool_kinds_used = 3; // a newly used sampler kind, with the old slot token
    pool_sync_entries(&ex);
    assert(PL.retry && PL.tok[0] == 0 && pool_defer[0] == 8);
    PL.frame++;
    PL.frame_ns = 2000000;
    fill_ok = 1;
    pool_sync_entries(&ex);
    assert(!PL.retry && PL.tok[0] == 42 && pool_defer[0] == 0);
    assert(seen_deadline == 5000000); // remaining four ms, not a fresh six ms

    // Hundreds of aliased deferred slots must keep the streaming budget.
    slot_count = 700;
    fill_ok = 0;
    PL.gen = 0;
    memset(PL.tok, 0, sizeof PL.tok);
    pool_sync_entries(&ex);
    assert(PL.retry && !pool_loading);
    // A real backlog still enters load mode, and leaves at the distinct count.
    distinct_missing = 600;
    pool_sync_entries(&ex);
    assert(pool_loading && pool_load_made == 600);
    distinct_missing = 1;
    pool_sync_entries(&ex);
    assert(!pool_loading);

    // Resolve all but one pending slot, then change an unrelated slot in the
    // same submission. Retry must not force another 700-slot scan.
    fill_ok = 1;
    PL.frame++;
    pool_sync_entries(&ex);
    assert(pool_pending_n == 0);
    pool_pending_note(17, token, (uint32_t)PL.frame + 1);
    PL.tok[17] = 0;
    PL.retry = 1;
    dirty_ok = 1;
    dirty_id = 31;
    PL.tok[31] = 0;
    token_reads = fills = 0;
    pool_sync_entries(&ex);
    assert(token_reads == 2 && fills == 2 && !PL.retry);
    assert(PL.tok[17] == token && PL.tok[31] == token);

    // Exhaustion skips only the same deferred token. A new registration in
    // the same frame must replace old views even when creation is exhausted.
    fill_ok = 0;
    pool_refill(&ex, 17, 3, token, 1);
    int before = fills;
    pool_refill(&ex, 17, 3, token, 1);
    assert(fills == before);
    pool_refill(&ex, 17, 3, ++token, 1);
    assert(fills == before + 1 && pool_pending_n == 1);

    // A new submission checks resolved entries without scanning every slot.
    fill_ok = 1;
    for (int id = 0; id < slot_count; id++) PL.tok[id] = token;
    PL.epoch = 0;
    fresh_checks = token_reads = 0;
    pool_sync_entries(&ex);
    assert(token_reads == 2 && fresh_checks == 0 && list_checks == 1);
    assert(!PL.retry);

    // The fallback knob keeps the old complete freshness walk available.
    fast_mode = 0;
    PL.epoch = 0;
    token_reads = fresh_checks = 0;
    pool_sync_entries(&ex);
    assert(token_reads == slot_count && fresh_checks == slot_count);

    // Resolving and becoming unresolved before the next refresh cannot add
    // duplicates to the unresolved list, even after many repetitions.
    for (int i = 0; i < 1000; i++) { pool_unres(12, 0); pool_unres(12, 1); }
    pool_unres(12, 0);
    assert(PL.nunres == 1 && PL.unres_on[12] && PL.unres_in[12]);

    // Ring overflow still falls back to a full refresh, with no lost retry.
    dirty_ok = 0;
    PL.gen = 0;
    token_reads = 0;
    pool_sync_entries(&ex);
    assert(token_reads == slot_count);
    puts("Texture pool checks passed: sparse retries, generation safety, full freshness, overflow fallback");
    return 0;
}

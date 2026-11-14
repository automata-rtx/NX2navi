// The selected production functions are extracted verbatim by descriptor_reuse_check.py. Pool publication and
// synchronization services are controlled fakes: these tests cover path selection, not Vulkan or memory ordering.
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "replay_work_trace.h"

static unsigned comparisons, stages[RWT_STAGE_COUNT];
static unsigned current_stage;
RwtStageScope gfx_rwt_stage_begin(unsigned stage) {
    RwtStageScope s = {.previous = current_stage};
    current_stage = stage;
    stages[stage]++;
    return s;
}
void gfx_rwt_stage_end(RwtStageScope *scope) { current_stage = scope->previous; }
static int counted_compare(const void *a, const void *b, size_t n) {
    comparisons++;
    return memcmp(a, b, n);
}

typedef uint64_t VkDescSlot;
typedef struct { uint32_t bytes, reserved; VkDescSlot data[]; } DescriptorPacket;
typedef uint64_t VkDescriptorSet;
typedef struct { unsigned local, group, prepared, adopted; } PoolStats;
typedef struct { PoolStats pool_stats; } DrawCtx;
typedef struct { DrawCtx *dc; uint64_t fence; } VkExec;
typedef struct { uint64_t fence, generation, set; uint32_t kinds; } VkExecPoolView;
static DrawCtx context, *dc = &context;
static VkExecPoolView group_view, local_view;
static uint32_t vk_pool_kinds_used;
static struct { VkDescriptorSet set; uint64_t cur_fence, completed_generation; int ready; uint32_t kinds; } PL;
static struct { uint32_t count; } pool_retry;
static int pool_mt, shared, in_chunk, waited_ready, locked;
static unsigned waits, syncs, locks, unlocks, publishes, generation_calls, mutate_generation_at;
static uint64_t generation;
static int vk_exec_shared(void) { return shared; }
static int vk_exec_in_chunk(void) { return in_chunk; }
static VkExecPoolView *vk_exec_pool_view(void) { return &group_view; }
static VkExecPoolView *vk_exec_pool_local_view(void) { return &local_view; }
static uint64_t vk_exec_pool_generation(void) {
    if (++generation_calls == mutate_generation_at) generation++;
    return generation;
}
static int vk_pool_view_get(const VkExecPoolView *v, uint64_t fence, uint32_t kinds, uint64_t gen, VkDescriptorSet *out) {
    if (!v->set || v->fence != fence || v->generation != gen || (v->kinds & kinds) != kinds) return 0;
    *out = v->set;
    return 1;
}
static int vk_exec_pool_wait(uint64_t fence, uint32_t kinds, VkDescriptorSet *out) {
    (void)fence; (void)kinds;
    assert(!locked);
    waits++;
    if (waited_ready) *out = 77;
    return waited_ready;
}
static void plat_mutex_lock(int *m) { assert(m == &pool_mt && !locked); locked = 1; locks++; }
static void plat_mutex_unlock(int *m) { assert(m == &pool_mt && locked); locked = 0; unlocks++; }
static void pool_sync(VkExec *ex) {
    assert(!shared || locked);
    syncs++;
    PL.set = 99, PL.cur_fence = ex->fence, PL.completed_generation = generation;
    PL.ready = 1, PL.kinds = vk_pool_kinds_used;
}
static void vk_pool_view_reset(VkExecPoolView *v) { memset(v, 0, sizeof *v); }
static void vk_pool_view_publish(VkExecPoolView *v, VkDescriptorSet set, uint64_t fence,
                                 uint64_t gen, uint32_t kinds, uint32_t retries) {
    (void)retries;
    assert(locked);
    *v = (VkExecPoolView){.set = set, .fence = fence, .generation = gen, .kinds = kinds};
    publishes++;
}

#define memcmp counted_compare
#include "descriptor_under_test.h"
#if DESCRIPTOR_DIFFERENTIAL
typedef struct { const DescriptorPacket *push_packet[2]; int pushed[2]; } DescriptorState;
typedef struct { uint32_t pn[2]; VkDescSlot psig[2][4]; } DsSig;
static int push_skip;
static int push_skip_on(void) { return push_skip; }
#include "descriptor_decision_under_test.h"
#endif
#undef memcmp

static void push_decision_checks(void) {
#if DESCRIPTOR_DIFFERENTIAL
    struct { uint32_t bytes, reserved; VkDescSlot data[4]; } storage = {32, 0, {1, 2, 3, 4}};
    const DescriptorPacket *p = (const DescriptorPacket *)&storage;
    VkDescSlot data[4] = {1, 2, 3, 4};
    unsigned cases = 0, saved = 0;
    for (int bytes = 0; bytes <= 32; bytes += 8)
    for (int pbytes = 0; pbytes <= 32; pbytes += 8)
    for (int changed = 0; changed < 2; changed++)
    for (int packet_mode = 0; packet_mode < 3; packet_mode++)
    for (int gfx = 0; gfx < 2; gfx++)
    for (int prepared = 0; prepared < 2; prepared++)
    for (int pushed = 0; pushed < 2; pushed++)
    for (push_skip = 0; push_skip < 2; push_skip++)
    for (int known = 0; known < 2; known++) {
        storage.bytes = (uint32_t)pbytes;
        data[0] = changed ? 9 : 1;
        DescriptorState state = {.push_packet = {packet_mode ? p : NULL, NULL}, .pushed = {pushed, 0}};
        DsSig sig = {.pn = {(uint32_t)pbytes, 0}, .psig = {{1, 2, 3, 4}, {0}}};
        int exact = descriptor_packet_equal(state.push_packet[0], data, (uint32_t)bytes);
        // A retained packet can share the current push pointer only after validating exactly these bytes.
        const DescriptorPacket *packet = packet_mode == 2 && prepared && exact ? p : NULL;
        int memo = known ? exact : -1;
        comparisons = 0;
        int before = push_decision_baseline(&state, &sig, gfx, prepared, packet, data, bytes, memo);
        unsigned old_compares = comparisons;
        comparisons = 0;
        int after = push_decision_candidate(&state, &sig, gfx, prepared, packet, data, bytes, memo);
        assert(before == after && comparisons <= old_compares);
        saved += old_compares - comparisons;
        cases++;
    }
    assert(saved > 0);
    printf("Push decision differential passed: %u cases, %u comparisons avoided\n", cases, saved);
#endif
}

static void packet_checks(void) {
    struct { uint32_t bytes, reserved; VkDescSlot data[4]; } storage = {sizeof(VkDescSlot) * 4, 0, {1, 2, 3, 4}};
    const DescriptorPacket *p = (const DescriptorPacket *)&storage;
    VkDescSlot equal[] = {1, 2, 3, 4}, changed[] = {1, 2, 3, 5};
    int push = -1, last = -1;
    comparisons = 0;
    assert(descriptor_packet_match(p, equal, sizeof equal, &push));
    assert(descriptor_packet_match(p, equal, sizeof equal, &push));
    assert(comparisons == 1);
    assert(!descriptor_packet_match(p, changed, sizeof changed, &last));
    assert(!descriptor_packet_match(p, changed, sizeof changed, &last));
    assert(comparisons == 2); // independent operand pairs cannot share a cached answer
    int known = -1;
    assert(!descriptor_packet_match(NULL, equal, sizeof equal, &known));
    known = -1;
    assert(!descriptor_packet_match(p, equal, sizeof equal - 1, &known));
    assert(comparisons == 2);
    storage.bytes = 0;
    known = -1;
    assert(descriptor_packet_match(p, NULL, 0, &known));
    assert(comparisons == 2);
    storage.bytes = sizeof equal;
    // Each new bind resets the cache. A newly copied packet must not reuse a failed result for the old pointer.
    known = -1;
    assert(!descriptor_packet_match(p, changed, sizeof changed, &known));
    memcpy(storage.data, changed, sizeof changed);
    known = -1;
    assert(descriptor_packet_match(p, changed, sizeof changed, &known));
    assert(comparisons == 4);
}

static void reset_pool(void) {
    memset(&context, 0, sizeof context);
    memset(&PL, 0, sizeof PL);
    memset(&group_view, 0, sizeof group_view);
    memset(&local_view, 0, sizeof local_view);
    memset(stages, 0, sizeof stages);
    shared = in_chunk = 1;
    waited_ready = locked = 0;
    waits = syncs = locks = unlocks = publishes = generation_calls = mutate_generation_at = 0;
    generation = 4, vk_pool_kinds_used = 3;
}

static void pool_checks(void) {
    VkExec ex = {.dc = &context, .fence = 12};
    reset_pool(); shared = 0;
    assert(pool_sync_draw(&ex) == 99 && syncs == 1 && locks == 0 && waits == 0);
    reset_pool(); in_chunk = 0;
    assert(pool_sync_draw(&ex) == 99 && syncs == 1 && locks == 1 && unlocks == 1 && waits == 0);
    reset_pool(); local_view = (VkExecPoolView){.set = 10, .fence = 12, .generation = 4, .kinds = 3};
    assert(pool_sync_draw(&ex) == 10 && context.pool_stats.local == 1 && !locks && !waits && !syncs);
    reset_pool(); group_view = (VkExecPoolView){.set = 11, .fence = 12, .generation = 4, .kinds = 3};
    assert(pool_sync_draw(&ex) == 11 && context.pool_stats.group == 1 && !locks && !waits && !syncs);
    reset_pool(); waited_ready = 1;
    assert(pool_sync_draw(&ex) == 77 && context.pool_stats.group == 1 && waits == 1 && !locks && !syncs);
    reset_pool();
    assert(pool_sync_draw(&ex) == 99 && context.pool_stats.prepared == 1 && syncs == 1 && waits == 1);
    assert(locks == 1 && unlocks == 1 && !locked && publishes == 2 && local_view.set == 99);
#if NX2NAVI_REPLAY_WORK_DESCRIPTORS
    assert(stages[RWT_STAGE_POOL_READY] == 1 && stages[RWT_STAGE_POOL_WAIT] == 1);
    assert(stages[RWT_STAGE_POOL_LOCK] == 1 && stages[RWT_STAGE_POOL_REFRESH] == 1);
#else
    for (unsigned i = 0; i < RWT_STAGE_COUNT; i++) assert(stages[i] == 0);
#endif
    reset_pool(); PL.set = 42, PL.cur_fence = 12, PL.completed_generation = 4, PL.ready = 1, PL.kinds = 3;
    assert(pool_sync_draw(&ex) == 42 && context.pool_stats.adopted == 1 && syncs == 0 && publishes == 2);
    reset_pool(); local_view = (VkExecPoolView){.set = 10, .fence = 12, .generation = 4, .kinds = 3};
    mutate_generation_at = 2;
    assert(pool_sync_draw(&ex) == 99 && !context.pool_stats.local && syncs == 1);
    reset_pool(); group_view = (VkExecPoolView){.set = 11, .fence = 11, .generation = 4, .kinds = 3};
    assert(pool_sync_draw(&ex) == 99 && syncs == 1); // stale fence never adopted
    reset_pool(); group_view = (VkExecPoolView){.set = 11, .fence = 12, .generation = 4, .kinds = 1};
    assert(pool_sync_draw(&ex) == 99 && syncs == 1); // insufficient kinds never adopted
}

int main(void) {
    packet_checks();
    push_decision_checks();
    pool_checks();
    puts("Descriptor comparison and pool path checks passed");
    return 0;
}

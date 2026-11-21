// The actual pool_current, pool_take_set, publication and pool_sync functions are extracted verbatim. External
// resource refresh and GPU fence services are fakes; tests prove the no-op branch does not bypass those services.
#include <assert.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef uintptr_t VkDescriptorSet;
typedef int TextureImagePublication;
#define VK_POOL_KINDS 1
#define VK_POOL_TEX 1
typedef uintptr_t VkCommandBuffer;
typedef struct VkTextureRegistrationScope { uint64_t publication; } VkTextureRegistrationScope;
typedef struct {
    uint64_t fence, tex_pool;
    int tex_pool_count;
    VkTextureRegistrationScope *registration_scope;
    int tex_serial, recorded, rt_resolved, dirty;
    VkCommandBuffer cb;
} VkExec;
#include "vk_pool_publication.h"
#include "replay_work_trace.h"
static struct {
    int cur, nsets, sealed, ready, retry, layout, publication_exhausted, registration_bound, registration_refresh;
    uint32_t kinds, gen, epoch, sgen;
    uint64_t cur_fence, set_fence[2], set_synced[2], completed_generation, logn, frame, frame_ns, sync_ns, catch_ns, waits;
    VkDescriptorSet set, sets[2];
    TextureImagePublication *publication[2];
    int publication_dirty[2], set_rebuild[2], wanted_reconcile, image[1], publication_updates;
    uint64_t authority, registration_scope, registration_publication, registration_target, registration_pool, marks_owner;
    int registration_limit, registration_live, marks_owner_valid, marks_mixed;
    uint32_t registration_gone_count, set_registration_gone[2];
    uint8_t registration_gone[VK_POOL_TEX], set_registration_missing[2], set_prefix_complete[2];
    uint32_t set_prepare_refs[2];
    uint64_t set_generation[2];
} PL;
typedef struct { int target_set; } GraphicsPoolRestore;
static GraphicsPoolRestore *graphics_pool_restore;
static const TextureImagePublication *pool_transfer_images;
static const TextureImagePublication *pool_member_installed_images(void) { return PL.publication[PL.cur]; }
static uint64_t pool_ready_generation, pool_ready_authority, authority, pool_takes, ticks;
static void *rc_registry_runtime(void) {return NULL;}
static uint64_t rc_registry_authority_epoch(void *r) {(void)r;return authority;}
static TextureImagePublication *vk_texture_publication_create_indexed(size_t n) {
    assert(n == VK_POOL_KINDS * VK_POOL_TEX); static int p; return &p;
}
static jmp_buf failed_prefix;
static int prefix_will_fail, prefix_failures;
static _Noreturn void rt_fatal(const char *s, ...) {
    if (prefix_will_fail) longjmp(failed_prefix, 1);
    fputs(s, stderr);
    abort();
}
static uint32_t vk_pool_kinds_used, gen, ep, sg;
static struct { uint64_t frames; } gfx_counters;
static unsigned refreshes, catches, fence_waits;
static const TextureImagePublication *expected_transfer;
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
static uint64_t registration_scope;
static uint64_t selected_pool;
static uint64_t vk_exec_texture_pool(void) {return selected_pool;}
static uint64_t vk_texture_registration_scope_token(void) { return registration_scope; }
static uint64_t vk_texture_registration_publication(void) { return registration_scope ? registration_scope : 1; }
typedef struct { uint64_t scene,ordinal; } RcRegistryPosition;
static RcRegistryPosition position;
static uint64_t mapping,protection;
static struct {uint64_t scene,ordinal,next_scene,next_ordinal,authority,legacy_authority,mapping,protection;uint32_t slots,epoch,samplers,kinds;} pool_order;
static RcRegistryPosition vk_exec_resource_position(void) {return position;}
static int rc_registry_position_open(void *r,RcRegistryPosition p) {(void)r;return p.scene!=0;}
static int rc_registry_position_compare(RcRegistryPosition a,RcRegistryPosition b) {
    if(a.scene!=b.scene)return a.scene<b.scene?-1:1;
    return a.ordinal==b.ordinal?0:(a.ordinal<b.ordinal?-1:1);
}
static uint64_t mem_mapping_serial(void) {return mapping;}
static uint64_t mem_protection_serial(void) {return protection;}
static void pool_order_publish(void) {
    pool_order.scene=position.scene;pool_order.ordinal=position.ordinal;
    pool_order.next_scene=pool_order.next_ordinal=0;
    pool_order.authority=authority;pool_order.mapping=mapping;pool_order.protection=protection;
    pool_order.slots=gen;pool_order.epoch=ep;pool_order.samplers=sg;pool_order.kinds=vk_pool_kinds_used;
}

static void vk_pool_set_layout(void) { PL.layout = 1; }
static uint64_t kern_now_ns(void) { return ++ticks; }
static void vk_texture_lock(void) { assert(!pool_ready_generation); }
static void vk_texture_unlock(void) { assert(!pool_ready_generation); }
static void pool_catch_up(uint64_t from, uint64_t to) {
    (void)from; (void)to;
    assert(!pool_ready_generation);
    if (expected_transfer) assert(pool_transfer_images == expected_transfer);
    PL.set_rebuild[PL.cur] = 0;
    catches++;
}
static void pool_sync_entries(VkExec *ex) {
    (void)ex;
    assert(!pool_ready_generation && !PL.sealed);
    refreshes++;
    PL.wanted_reconcile = 0;
    PL.registration_refresh = 0;
    PL.authority = authority;
    PL.ready = 1, PL.retry = 0, PL.kinds = vk_pool_kinds_used, PL.gen = gen, PL.epoch = ep, PL.sgen = sg;
}
typedef int VkResult;
enum {VK_SUCCESS, PW_SPLIT, VK_EXEC_DIRTY_ALL = 15};
static int split_stage, split_draws, part_start, n_parts;
static int vk_exec_recording_failed(void) {return 0;}
static uint64_t part_t0;
static void group_drain(VkExec *ex,int why) {assert(why==PW_SPLIT&&ex->registration_scope&&split_stage++==0);}
static void vk_exec_pass_end(VkExec *ex) {(void)ex;assert(split_stage++==1);}
static void vk_exec_barrier(VkExec *ex) {(void)ex;assert(split_stage++==2);}
static VkCommandBuffer vk_sync_split(void) {assert(split_stage++==3);return 71;}
static VkResult vk_device_wait_idle(void) {
    assert(PL.set_fence[0]==8&&PL.set_fence[1]==8&&split_stage++==4);
    return prefix_will_fail ? -1 : VK_SUCCESS;
}
static void vk_device_lost_check(VkResult result,const char *where) {assert(result==-1&&where&&prefix_will_fail);prefix_failures++;}
static void vk_exec_draw_new_cb(void) {assert(split_stage++==5);}
#include "pool_sealed_under_test.h"

static void reset(void) {
    memset(&PL, 0, sizeof PL);
    registration_scope = 0;
    selected_pool = 0;
    PL.registration_publication = 1;
    PL.nsets = 2, PL.sealed = PL.ready = PL.layout = 1;
    PL.kinds = vk_pool_kinds_used = 1;
    PL.gen = gen = 2, PL.epoch = ep = 3, PL.sgen = sg = 4;
    PL.sets[0] = PL.set = 11, PL.sets[1] = 22;
    PL.cur_fence = PL.set_fence[0] = 8;
    PL.completed_generation = pool_ready_generation = 10;
    PL.authority = pool_ready_authority = authority = 1;
    PL.publication[0] = vk_texture_publication_create_indexed(VK_POOL_KINDS * VK_POOL_TEX);
    pool_takes = refreshes = catches = fence_waits = 0;
    position=(RcRegistryPosition){8,2};mapping=protection=1;pool_order_publish();
    gfx_counters.frames = 1;
}

int main(void) {
    VkExec ex = {.fence = 8};
    VkExecPoolView group = {0};
    reset();
    assert(vk_pool_view_publish(&group, PL.set, ex.fence, 10, 1, 0));
    VkExecPoolView before = group;
    // Future predeclared writes do not change the global epoch when later activated. The actual production
    // aggregate receipt permits only the consumers before its earliest producer boundary.
    pool_order.next_scene=8;pool_order.next_ordinal=7;
    position.ordinal=6;assert(vk_exec_pool_generation()==10);
    position.ordinal=8;assert(!vk_exec_pool_generation());
    position.ordinal=1;assert(!vk_exec_pool_generation());
    position.ordinal=2;mapping++;assert(!vk_exec_pool_generation());mapping--;
    protection++;assert(!vk_exec_pool_generation());protection--;
    pool_order_publish();
    position.scene=9;assert(!vk_exec_pool_generation());position.scene=8;
    for (int i = 0; i < 100; i++) {
        pool_sync(&ex);
        VkDescriptorSet set = 0;
        assert(vk_pool_view_get(&group, ex.fence, 1, vk_exec_pool_generation(), &set) && set == 11);
    }
    assert(PL.sealed && PL.cur == 0 && PL.completed_generation == 10 && PL.set_fence[0] == 8);
    assert(!refreshes && !catches && !fence_waits && !pool_takes && !memcmp(&group, &before, sizeof group));
    // Each production pool_current input invalidates the sealed snapshot; fence change also requires a new set.
    for (int invalidator = 0; invalidator < 11; invalidator++) {
        reset(); ex.fence = 8;
        switch (invalidator) {
        case 0: PL.ready = 0; break;
        case 1: PL.retry = 1; break;
        case 2: vk_pool_kinds_used = 3; break;
        case 3: gen++; break;
        case 4: ep++; break;
        case 5: sg++; break;
        case 6: ex.fence++; break;
        case 7: authority++; break;
        case 8: PL.publication[0] = NULL; break;
        case 9: PL.set_rebuild[0] = 1; break;
        case 10: PL.wanted_reconcile = 1; break;
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
    reset(); ex.fence = 9; PL.logn = PL.set_synced[0] = 1;
    expected_transfer = PL.publication[0];
    pool_sync(&ex);
    assert(PL.cur == 1 && catches == 1 && !pool_transfer_images);
    expected_transfer = NULL; ex.fence = 8;
    reset(); PL.sealed = 0; PL.registration_bound = 1; registration_scope = 77;
    TextureImagePublication *earlier = PL.publication[0];
    pool_sync(&ex);
    assert(PL.cur == 1 && PL.set == 22 && !PL.registration_bound && !fence_waits);
    assert(PL.registration_scope == 77 && PL.set_fence[0] == 8 && PL.publication[0] == earlier);
    PL.registration_bound = 1;
    assert(!vk_exec_pool_registration_needs_prefix(ex.fence)); // Equal semantic publication shares the set.
    registration_scope = 78;
    assert(vk_exec_pool_registration_needs_prefix(ex.fence));
    VkTextureRegistrationScope third = {78};
    ex.recorded = 9;
    prefix_will_fail = 1;
    VkExec failed = ex;
    if (!setjmp(failed_prefix)) {
        cont_registration_select(&failed, &third);
        assert(0);
    }
    assert(prefix_failures == 1 && PL.set_fence[0] == 8 && PL.set_fence[1] == 8 && !n_parts);
    prefix_will_fail = split_stage = 0;
    split_draws = 30;
    cont_registration_select(&ex, &third); // Third exact horizon with only two native sets.
    assert(split_stage == 6 && n_parts == 1 && ex.cb == 71 && ex.dirty == VK_EXEC_DIRTY_ALL && part_start == 9);
    assert(!split_draws && part_t0 != 0);
    assert(!PL.cur_fence && !PL.set_fence[0] && !PL.set_fence[1] && PL.publication[0] == earlier);
    pool_sync(&ex);
    assert(PL.cur == 0 && PL.set == 11 && PL.registration_publication == 78 && !fence_waits);
    PL.registration_bound = 1;
    ex.tex_pool = selected_pool = 42;
    assert(!vk_exec_pool_registration_needs_prefix(ex.fence)); // One spare set can publish the new owner.
    earlier = PL.publication[0];
    pool_sync(&ex);
    assert(PL.cur == 1 && PL.registration_pool == 42 && PL.publication[0] == earlier && PL.marks_mixed);
    ex.tex_pool = selected_pool = 0;
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

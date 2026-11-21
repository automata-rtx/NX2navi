#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "vk.h"
#include "vk_gpa.h"
#include "resource_core_vma.h"

enum { POOL_IMPORT, POOL_PHYS, MS_ON, MS_REFUSED, MS_PENDING, MS_NONE, MS_CREATING, MS_DORMANT };
typedef struct {
    int kind, mstate, unpublished;
    uint64_t lo, size, import_lo, import_hi, host_mapping_cutoff, publication, mirror_incarnation;
    uint32_t mirror_input_refs, mirror_checked_sub;
    uint64_t mirror_import_only_scene;
    uint64_t import_writer_scene, import_writer_fence;
    uint32_t import_writer_refs;
    uint64_t wb_fence;
    unsigned mirror_retire_completed, mirror_import_transitioned, wb_kept, gpa_rec, twin, acc_sub;
    VkBuffer buf, mbuf;
    VmaAllocation malloc_;
    void *smap;
    int gpa_host_coherent;
    MemBackingCapture *host_backing;
    unsigned refs;
} PoolRec;
typedef struct { PoolRec *buf; } pidx_ent;
struct MemBackingCapture { RcInterval interval; uint64_t generation; };

static pidx_ent pool_index;
int vk_mirror_on;
static int pool_lock, native_lock, alone = 1;
static uint32_t msub = 1;
static uint64_t generation = 9, remapped_at;
static uint64_t done, read_scene = 17;
static unsigned mappings;
static const uint64_t address = UINT64_C(0x8000);
static struct { PoolRec *r; } wbl[1], snl[1];
static struct { VkMemSpan sp; } lz_skip[1];
static int nwbl, nsnl, lz_nskip;
static VkCtx context;
static int blocked, lazy_available = 1, lazy_borrowed, lazy_locked, gpa_safe;
static unsigned forgotten, destroyed, remaps;
static void (*retire_fn)(void *);
static void (*retire_cancel)(void *, int);
static void *retire_owner;

static void plat_mutex_lock_shared(int *lock) {
    assert(lock == &pool_lock && !native_lock);
    native_lock = 1;
}
static void plat_mutex_unlock_shared(int *lock) {
    assert(lock == &pool_lock && native_lock);
    native_lock = 0;
}
static void plat_mutex_lock(int *lock) { plat_mutex_lock_shared(lock); }
static void plat_mutex_unlock(int *lock) { plat_mutex_unlock_shared(lock); }
static const pidx_ent *pidx_find(const pidx_ent *index, uint64_t addr, uint64_t size) {
    const PoolRec *pool = index->buf;
    return pool && addr >= pool->lo && addr - pool->lo < pool->size &&
        size <= pool->size - (addr - pool->lo) ? index : NULL;
}
static void pool_native_acquire(PoolRec *pool) { assert(pool->refs); pool->refs++; }
static int mirror_pool_entry(const PoolRec *pool) { return pool == pool_index.buf ? 0 : -1; }
static int mirror_pool_alone(int entry) { return entry == 0 && alone; }
static void mirror_forget(PoolRec *pool) { assert(pool && native_lock && pool->mstate == MS_ON); forgotten++; }
static void destroy_rec(VkCtx *ctx, PoolRec *pool) { assert(ctx == &context && pool->refs && !native_lock); pool->refs--; }
VkCtx *vk_ctx(void) { return &context; }
int vk_exec_shared(void) { return 0; }
uint64_t vk_sync_fence_done(void) { return done; }
RcRegistryPosition vk_exec_resource_position(void) { return (RcRegistryPosition){read_scene, 2}; }
RcRegistry *rc_registry_runtime(void) { return (RcRegistry *)(uintptr_t)1; }
int rc_registry_provisional_read_current(RcRegistry *r, RcRegistryPosition p) {
    return r && !rc_registry_position_compare(p, vk_exec_resource_position());
}
RcRegistryResult rc_registry_read_range_at(RcRegistry *r, RcSourceRange range, RcRegistryPosition p) {
    assert(r && range.size && !rc_registry_position_compare(p, vk_exec_resource_position()));
    return blocked ? RC_REGISTRY_BLOCKED : RC_REGISTRY_OK;
}
volatile int32_t vk_lost;
int vk_sync_mirror_publication_safe(uint64_t scene) { return scene == read_scene && gpa_safe; }
uint32_t vk_gpa_remap_exact(uint64_t lo, uint64_t hi, uint32_t old, VkBuffer buffer, uint64_t start,
                            int keep_twin, int visible, int *uses_twin) {
    assert(lazy_locked && native_lock && hi > lo && old && buffer && lo == start && !keep_twin && visible && !uses_twin);
    remaps++;
    return old + 1;
}
int mem_lazy_provider_transition_try(void) {
    assert(native_lock && !lazy_locked);
    return lazy_locked = lazy_available;
}
int mem_lazy_provider_borrowed(const void *provider, uint64_t buffer) {
    const PoolRec *pool = provider;
    assert(lazy_locked && pool && pool->mbuf == (VkBuffer)(uintptr_t)buffer);
    return lazy_borrowed;
}
void mem_lazy_provider_transition_end(void) { assert(lazy_locked); lazy_locked = 0; }
void vk_completion_add_owned(uint64_t fence, void (*fn)(void *), void (*cancel)(void *, int), void *owner) {
    assert(fence == read_scene && !native_lock && !lazy_locked && !retire_owner);
    retire_fn = fn; retire_cancel = cancel; retire_owner = owner;
}
void resource_core_vma_destroy_buffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation) {
    const PoolRec *owner = retire_owner;
    assert(!native_lock && allocator == context.vma && buffer && allocation);
    assert(owner && owner->mstate == MS_DORMANT && !owner->mbuf);
    destroyed++;
}
void rt_fatal(const char *message, ...) { fputs(message, stderr); abort(); }

int mem_mapping_range_unchanged(uint64_t addr, uint64_t size, uint64_t cutoff) {
    return addr == address && size == 64 && cutoff >= remapped_at;
}
int mem_backing_current_range(const MemBackingCapture *capture, uint64_t addr, uint64_t size) {
    return capture && capture->generation == generation && addr >= capture->interval.address &&
        addr - capture->interval.address <= capture->interval.range.size &&
        size <= capture->interval.range.size - (addr - capture->interval.address);
}
RcBackingResult mem_backing_capture_bounded(const MemBackingCapturePlan *plan, MemBackingCapture **out) {
    if (plan->latest_mapping != generation) return RC_BACKING_STALE;
    assert(!native_lock && plan->count == 1 && plan->bytes == sizeof(MemBackingCapture) && !*out);
    MemBackingCapture *capture = malloc(sizeof *capture);
    assert(capture);
    *capture = (MemBackingCapture){.interval={.range={.backing=7,.offset=4160,.size=plan->size},
        .address=plan->address,.mapping=11},.generation=generation};
    *out = capture;
    mappings++;
    return RC_BACKING_OK;
}
const RcInterval *mem_backing_intervals(const MemBackingCapture *capture, uint32_t *count) {
    *count = capture ? 1 : 0;
    return capture ? &capture->interval : NULL;
}
void mem_backing_release(MemBackingCapture **owner) {
    if (!*owner) return;
    assert(mappings);
    mappings--;
    free(*owner);
    *owner = NULL;
}
void rc_backing_capture_release(RcBackingCapture **capture) { assert(!*capture); }
#include "selected_provider.inc"

int main(void) {
    PoolRec pool = {.kind=POOL_IMPORT,.mstate=MS_NONE,.lo=address-2048,.size=8192,.import_lo=address-4096,
        .publication=15,.host_mapping_cutoff=9,.buf=(VkBuffer)(uintptr_t)1,.refs=1};
    pool_index.buf = &pool;
    VkMemSpan imported = {.buffer=pool.buf,.offset=4096,.avail=6144,.pool=&pool};
    MemBackingCapturePlan host = {.address=address,.size=64,.latest_mapping=generation,
        .bytes=sizeof(MemBackingCapture),.count=1};
    VkMemImportInputPlan plan;
    VkMemSpanCapture *hold = NULL;
    assert(vk_mem_span_import_input_plan(&imported,&host,&plan)==RC_REGISTRY_OK);
    assert(plan.selected.buffer == pool.buf && plan.selected.offset == 4096 && plan.selected.avail == 64);
    assert(plan.bytes == sizeof(VkMemSpanCapture) + sizeof(MemBackingCapture));
    assert(!mappings && pool.refs == 1);

    // Reusing the same native handle is insufficient when its current publication changed.
    pool.publication++;
    assert(vk_mem_span_import_input_capture(&plan,&host,&hold)==RC_REGISTRY_STALE);
    assert(!hold && !mappings && pool.refs == 1);
    assert(vk_mem_span_import_input_plan(&imported,&host,&plan)==RC_REGISTRY_OK);
    generation++;
    remapped_at = generation;
    assert(vk_mem_span_import_input_capture(&plan,&host,&hold)==RC_REGISTRY_STALE);
    host.latest_mapping = generation;
    assert(vk_mem_span_import_input_plan(&imported,&host,&plan)==RC_REGISTRY_STALE);
    assert(!hold && !mappings && pool.refs == 1);
    pool.host_mapping_cutoff = generation;

    vk_mirror_on = 1;
    assert(vk_mem_span_import_input_plan(&imported,&host,&plan)==RC_REGISTRY_UNSUPPORTED);
    pool.mstate = MS_ON;
    pool.mbuf = (VkBuffer)(uintptr_t)2;
    pool.mirror_incarnation = 3;
    assert(vk_mem_span_import_input_plan(&imported,&host,&plan)==RC_REGISTRY_OK);
    assert(plan.selected.buffer == pool.mbuf && plan.mirror_incarnation == 3 && !pool.mirror_input_refs);
    plan.selected.offset++;
    assert(vk_mem_span_import_input_capture(&plan,&host,&hold)==RC_REGISTRY_STALE);
    assert(!hold && !mappings && pool.refs == 1);
    plan.selected.offset--;
    assert(vk_mem_span_import_input_capture(&plan,&host,&hold)==RC_REGISTRY_OK);
    const RcInterval *ranges;
    assert(vk_mem_span_import_input_ranges(hold,&ranges)==1);
    assert(ranges->address == address && ranges->range.backing == 7 && ranges->range.size == 64);
    assert(pool.refs == 2 && pool.mirror_input_refs == 1);

    // An unissued capture releases its own hold. A separately recorded CbRes reference holds the
    // mirror through verified native lifetime and drops the same provider once at disposal.
    assert(!vk_mem_span_take_input_provider(&hold).pool);
    VkMemSpanCapture *use = vk_mem_span_capture_retain(hold);
    use->input_recorded = 1;
    VkMemInputProvider provider = vk_mem_span_take_input_provider(&use);
    assert(!use && provider.pool == &pool && pool.refs == 3 && pool.mirror_input_refs == 2);
    vk_mem_span_capture_release(&hold);
    assert(!mappings && pool.refs == 2 && pool.mirror_input_refs == 1);
    vk_mem_input_provider_release(provider);
    assert(pool.refs == 1 && !pool.mirror_input_refs);

    pool.mstate = MS_NONE;
    assert(!vk_mem_span_import_writer_candidate(address, &imported)); // previously installed, even after eviction
    pool.mirror_incarnation = 0; // a new never-activated native publication
    pool.publication++;
    pool.mbuf = VK_NULL_HANDLE;
    assert(vk_mem_span_import_writer_candidate(address, &imported));
    assert(vk_mem_span_import_writer_plan(&imported, &host, &plan) == RC_REGISTRY_UNSUPPORTED);
    pool.gpa_host_coherent = 1;
    alone = 0;
    assert(!vk_mem_span_import_writer_candidate(address, &imported));
    alone = 1;
    assert(vk_mem_span_import_writer_plan(&imported, &host, &plan) == RC_REGISTRY_OK);
    assert(plan.cold_writer && plan.selected.buffer == pool.buf && !mirror_import_scene(&pool));
    assert(vk_mem_span_import_input_capture(&plan, &host, &hold) == RC_REGISTRY_OK);
    VkMemSpanCapture *second = NULL;
    assert(vk_mem_span_import_input_capture(&plan, &host, &second) == RC_REGISTRY_OK);
    VkMemSpanCapture *batch[] = {hold, second};
    second->host->generation--;
    assert(vk_mem_span_import_writers_publish(batch, 2, vk_exec_resource_position()) == RC_REGISTRY_STALE);
    assert(!pool.import_writer_refs && !mirror_import_scene(&pool));
    second->host->generation++;
    assert(vk_mem_span_import_writers_publish(batch, 2, vk_exec_resource_position()) == RC_REGISTRY_OK);
    assert(pool.import_writer_refs == 2 && mirror_import_scene(&pool) == msub + 1 && !pool.import_writer_fence);
    vk_mem_span_capture_release(&hold);
    vk_mem_span_capture_release(&second);
    assert(!pool.import_writer_refs && !mirror_import_scene(&pool));

    assert(vk_mem_span_import_input_capture(&plan, &host, &hold) == RC_REGISTRY_OK);
    assert(vk_mem_span_import_writers_publish(&hold, 1, vk_exec_resource_position()) == RC_REGISTRY_OK);
    vk_mem_span_import_writer_issued(hold, 17);
    vk_mem_span_capture_release(&hold);
    assert(!pool.import_writer_refs && pool.import_writer_fence == 17 && mirror_import_scene(&pool) == msub + 1);
    msub++;
    assert(mirror_import_scene(&pool) == msub + 1); // Later optional promotion still waits on actual producer.
    done = 17;
    assert(!mirror_import_scene(&pool) && !mappings && pool.refs == 1);

    read_scene++;
    pool.mstate = MS_ON;
    pool.import_hi = pool.import_lo + 8192;
    pool.mbuf = (VkBuffer)(uintptr_t)3;
    pool.malloc_ = (VmaAllocation)(uintptr_t)4;
    pool.mirror_incarnation = 1;
    pool.gpa_rec = 20;
    MemBackingCapture whole = {.interval={.range={.backing=7,.offset=64,.size=8192},
        .address=pool.import_lo,.mapping=11},.generation=generation};
    pool.host_backing = &whole;
    assert(vk_mem_span_import_writer_candidate(address, &imported) == VK_MEM_IMPORT_WRITER_ACTIVE);
    assert(vk_mem_span_import_writer_plan(&imported, &host, &plan) == RC_REGISTRY_BLOCKED);
    gpa_safe = 1;
    blocked = 1;
    assert(vk_mem_span_import_writer_plan(&imported, &host, &plan) == RC_REGISTRY_BLOCKED);
    blocked = 0;
    nwbl = 1;
    wbl[0].r = &pool;
    assert(vk_mem_span_import_writer_plan(&imported, &host, &plan) == RC_REGISTRY_BLOCKED);
    assert(vk_mem_span_import_input_plan(&imported, &host, &plan) == RC_REGISTRY_OK);
    VkMemSpanCapture *older = NULL;
    assert(vk_mem_span_import_input_capture(&plan, &host, &older) == RC_REGISTRY_OK);
    assert(older->selected.buffer == pool.mbuf && older->mirror_incarnation == pool.mirror_incarnation);
    assert(nwbl == 1 && !remaps && !pool.import_writer_refs && pool.mstate == MS_ON);
    nwbl = 0;
    assert(vk_mem_span_import_writer_plan(&imported, &host, &plan) == RC_REGISTRY_OK);
    assert(plan.transition_mirror == 1 && plan.selected.buffer == pool.buf);
    assert(vk_mem_span_import_input_capture(&plan, &host, &hold) == RC_REGISTRY_OK);
    lazy_available = 0;
    assert(vk_mem_span_import_writers_publish(&hold, 1, vk_exec_resource_position()) == RC_REGISTRY_BLOCKED);
    lazy_available = lazy_borrowed = 1;
    assert(vk_mem_span_import_writers_publish(&hold, 1, vk_exec_resource_position()) == RC_REGISTRY_BLOCKED);
    assert(pool.mstate == MS_ON && !remaps && !retire_owner && !pool.import_writer_refs);
    lazy_borrowed = 0;
    assert(vk_mem_span_import_writers_publish(&hold, 1, vk_exec_resource_position()) == RC_REGISTRY_OK);
    assert(pool.mstate == MS_DORMANT && pool.gpa_rec == 21 && remaps == 1 && forgotten == 1 && !destroyed);
    assert(pool.mbuf && retire_owner == &pool && retire_cancel);
    vk_mem_span_capture_release(&hold); // The actual draw is known unissued.
    assert(!destroyed && pool.mirror_input_refs == 1); // Earlier pin still owns its actual native backing.
    done = read_scene;
    vk_mem_span_capture_release(&older);
    assert(!destroyed && pool.mstate == MS_DORMANT && !pool.mirror_input_refs);
    retire_fn(retire_owner);
    assert(destroyed == 1 && pool.mstate == MS_NONE && !pool.mbuf && !pool.mirror_input_refs &&
           !pool.import_writer_refs && !mappings && pool.refs == 1);
    puts("Selected SSBO provider: cold atomic choice, active history refusal, dormant mirror custody and exact retirement passed");
}

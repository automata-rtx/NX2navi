#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "resource_core_registry.h"

enum { POOL_IMPORT = 1, POOL_PHYS = 2 };
typedef struct PoolRec { int kind; uint8_t *smap; uint64_t import_lo, import_hi; } PoolRec;
typedef struct VkMemSpan { PoolRec *pool; uint64_t buffer, offset, avail; } VkMemSpan;
struct VkMemSpanCapture { unsigned refs; int current; };
typedef uint64_t VkBuffer;
typedef struct { VkBuffer mirror, import; RcRegistryBufferSpan native; uint64_t descriptor_size; RcRegistryPosition read; } SelectedMirrorWord;
struct VkMemNativeRmwBitmap { SelectedMirrorWord source; VkMemSpan descriptor; VkMemSpanCapture *provider; };
static RcRegistry *registry;
static unsigned copy_interval_calls, apply_calls, callback_count, bitmap_done_count;
static int fail_next_realloc, fail_next_malloc;
static RcSourceRange guest_range, canonical_range;
static VkMemSpanCapture capture = {.refs = 1, .current = 1};
static PoolRec pool = {.kind = POOL_IMPORT, .import_lo = 0x1000, .import_hi = 0x4000};
static VkMemSpan descriptor = {.pool = &pool};
static uint8_t actual_output[32];
typedef struct { void (*done)(void *); void (*cancel)(void *, int); void *arg; } Callback;
static Callback callbacks[8];

static void *controlled_realloc(void *p, size_t n) {
    if (fail_next_realloc) { fail_next_realloc = 0; return NULL; }
    return realloc(p, n);
}
static void *controlled_malloc(size_t n) {
    if (fail_next_malloc) { fail_next_malloc = 0; return NULL; }
    return malloc(n);
}
static RcRegistry *rc_registry_runtime(void) { return registry; }
static int nvn_gpu_only(uint64_t addr) { (void)addr; return 0; }
static uint32_t copy_pool_host_intervals(const PoolRec *p, uint64_t addr, uint64_t size, RcInterval **out) {
    (void)p; (void)addr; (void)size; copy_interval_calls++; *out = NULL; return 0;
}
static VkMemSpanCapture *vk_mem_span_capture_retain(VkMemSpanCapture *c) {
    if (c) { assert(c->refs); c->refs++; }
    return c;
}
static void vk_mem_span_capture_release(VkMemSpanCapture **owned) {
    if (owned && *owned) { assert((*owned)->refs > 1); (*owned)->refs--; *owned = NULL; }
}
static int vk_mem_span_capture_current(const VkMemSpanCapture *c, const VkMemSpan *d) {
    assert(c && d && d->pool == &pool); return c->current;
}
static void rt_fatal(const char *fmt, ...) { (void)fmt; abort(); }
static void vk_completion_add_owned(uint64_t fence, void (*done)(void *), void (*cancel)(void *, int), void *arg) {
    assert(fence == 6 && callback_count < 8); callbacks[callback_count++] = (Callback){done, cancel, arg};
}
#define realloc controlled_realloc
#define malloc controlled_malloc
// Only the actual writeback spans become current. Descriptor extents can include lazy or deliberately skipped tails.
typedef struct {
    RcSourceRange range;
    VkMemSpanCapture *capture;
    VkMemSpan descriptor;
} CpuVisibilityRange;
typedef struct {
    RcRegistryVisibility proof;
    CpuVisibilityRange *ranges;
    uint32_t count, capacity;
    int incomplete;
} CpuVisibility;

static int cpu_visibility_reserve(CpuVisibility *job, uint32_t add) {
    if (add > UINT32_MAX - job->count) return 0;
    const uint32_t needed = job->count + add;
    if (needed <= job->capacity) return 1;
    uint32_t capacity = job->capacity ? job->capacity : 16;
    while (capacity < needed) {
        if (capacity > UINT32_MAX / 2) { capacity = needed; break; }
        capacity *= 2;
    }
    CpuVisibilityRange *ranges = realloc(job->ranges, (size_t)capacity * sizeof *ranges);
    if (!ranges) return 0;
    job->ranges = ranges;
    job->capacity = capacity;
    return 1;
}

static void cpu_visibility_add_guarded(CpuVisibility *job, const PoolRec *pool, uint64_t off, uint64_t size,
                                       VkMemSpanCapture *capture, const VkMemSpan *descriptor) {
    if (!job || job->incomplete || !job->proof.horizon || !pool || pool->kind != POOL_IMPORT || pool->smap ||
        pool->import_hi <= pool->import_lo || nvn_gpu_only(pool->import_lo) || (!!capture != !!descriptor)) return;
    uint64_t limit = pool->import_hi - pool->import_lo;
    if (off >= limit || !size) return;
    if (size > limit - off) size = limit - off;
    uint64_t addr = pool->import_lo + off;
    RcInterval *intervals;
    uint32_t count = copy_pool_host_intervals(pool, addr, size, &intervals);
    if (count == UINT32_MAX || !cpu_visibility_reserve(job, count + 1)) {
        free(intervals);
        job->incomplete = 1;
        return;
    }
    const VkMemSpan selected = descriptor ? *descriptor : (VkMemSpan){0};
    VkMemSpanCapture *held = capture ? vk_mem_span_capture_retain(capture) : NULL;
    if (capture && !held) {
        free(intervals);
        job->incomplete = 1;
        return;
    }
    job->ranges[job->count++] = (CpuVisibilityRange){
        .range = {1, addr, size}, .capture = held, .descriptor = selected};
    for (uint32_t i = 0; i < count; ++i) {
        VkMemSpanCapture *part = capture ? vk_mem_span_capture_retain(capture) : NULL;
        if (capture && !part) {
            free(intervals);
            job->incomplete = 1;
            return;
        }
        job->ranges[job->count++] = (CpuVisibilityRange){.range = intervals[i].range,
            .capture = part, .descriptor = selected};
    }
    free(intervals);
}

// The bitmap already owns an exact retained canonical source interval. Pair that interval with its guest
// compatibility address directly, so guarded visibility needs no late host-interval allocation. If the
// range journal cannot represent the pair, omit this scene's visibility proof and retain foreign authority.
static void cpu_visibility_add_guarded_pair(CpuVisibility *job, RcSourceRange guest, RcSourceRange canonical,
                                            VkMemSpanCapture *capture, const VkMemSpan *descriptor) {
    if (!job || job->incomplete || !job->proof.horizon || !capture || !descriptor ||
        guest.backing != RC_GUEST_COMPATIBILITY_BACKING || canonical.backing <= RC_GUEST_COMPATIBILITY_BACKING ||
        !rc_range_valid(guest) || !rc_range_valid(canonical) || guest.size != canonical.size) {
        if (job) job->incomplete = 1;
        return;
    }
    if (!cpu_visibility_reserve(job, 2)) {
        job->incomplete = 1;
        return;
    }
    VkMemSpanCapture *guest_hold = vk_mem_span_capture_retain(capture);
    VkMemSpanCapture *canonical_hold = vk_mem_span_capture_retain(capture);
    if (!guest_hold || !canonical_hold) {
        vk_mem_span_capture_release(&guest_hold);
        vk_mem_span_capture_release(&canonical_hold);
        job->incomplete = 1;
        return;
    }
    const VkMemSpan selected = *descriptor;
    job->ranges[job->count++] = (CpuVisibilityRange){.range = guest, .capture = guest_hold, .descriptor = selected};
    job->ranges[job->count++] = (CpuVisibilityRange){.range = canonical, .capture = canonical_hold, .descriptor = selected};
}

static void cpu_visibility_add(CpuVisibility *job, const PoolRec *pool, uint64_t off, uint64_t size) {
    cpu_visibility_add_guarded(job, pool, off, size, NULL, NULL);
}

static void cpu_visibility_ranges_release(CpuVisibility *job) {
    for (uint32_t i = 0; i < job->count; ++i)
        vk_mem_span_capture_release(&job->ranges[i].capture);
}
static void cpu_visibility_discard(CpuVisibility *job) {
    cpu_visibility_ranges_release(job);
    free(job->ranges);
    job->ranges = NULL;
    job->count = job->capacity = 0;
}

static void cpu_visibility_done(void *arg) {
    CpuVisibility *job = arg;
    for (uint32_t i = 0; i < job->count; i++) {
        CpuVisibilityRange *range = &job->ranges[i];
        if (!range->capture || vk_mem_span_capture_current(range->capture, &range->descriptor))
            rc_registry_visibility_apply(rc_registry_runtime(), &job->proof, range->range);
    }
    cpu_visibility_ranges_release(job);
    free(job->ranges);
    free(job);
}
static void cpu_visibility_cancel(void *arg, int may_have_executed) {
    (void)may_have_executed;
    CpuVisibility *job = arg;
    cpu_visibility_ranges_release(job);
    free(job->ranges);
    free(job);
}
static void cpu_visibility_queue(CpuVisibility *job, uint64_t fence) {
    if (job->incomplete) { cpu_visibility_discard(job); return; }
    if (!job->count) { free(job->ranges); job->ranges = NULL; job->capacity = 0; return; }
    CpuVisibility *owned = malloc(sizeof *owned);
    if (!owned) { cpu_visibility_discard(job); return; }
    *owned = *job;
    vk_completion_add_owned(fence, cpu_visibility_done, cpu_visibility_cancel, owned);
}

static int native_rmw_bitmap_visibility_ranges(const VkMemNativeRmwBitmap *owner, const PoolRec *pool,
    uint64_t off, uint64_t size, RcSourceRange *guest, RcSourceRange *canonical) {
    if (!owner || !pool || !guest || !canonical || !size || pool != owner->descriptor.pool ||
        pool->kind != POOL_IMPORT || pool->smap || pool->import_hi <= pool->import_lo ||
        nvn_gpu_only(pool->import_lo)) return 0;
    const SelectedMirrorWord *source = &owner->source;
    const uint64_t source_offset = source->native.native.offset;
    if (off < source_offset) return 0;
    const uint64_t delta = off - source_offset;
    if (!rc_range_valid(source->native.interval.range) || delta > source->native.interval.range.size ||
        size > source->native.interval.range.size - delta || delta > UINT64_MAX - source->native.interval.address ||
        delta > UINT64_MAX - source->native.interval.range.offset || pool->import_lo > UINT64_MAX - off ||
        pool->import_lo + off != source->native.interval.address + delta ||
        source->native.interval.address + delta < pool->import_lo ||
        source->native.interval.address + delta >= pool->import_hi ||
        size > pool->import_hi - (source->native.interval.address + delta)) return 0;
    *guest = (RcSourceRange){RC_GUEST_COMPATIBILITY_BACKING, source->native.interval.address + delta, size};
    *canonical = (RcSourceRange){source->native.interval.range.backing,
        source->native.interval.range.offset + delta, size};
    return rc_range_valid(*guest) && rc_range_valid(*canonical);
}

#undef realloc
#undef malloc

static uint64_t serial = 1;
static int blocked(uint64_t backing_id, uint64_t offset) {
    RcRegistryHandle h; RcRegistryObservation o;
    assert(rc_registry_source(registry, serial++, (RcSourceRange){backing_id, offset, 1}, &h) == RC_REGISTRY_OK);
    return rc_registry_observe(registry, h, 1, &o) != RC_REGISTRY_OK;
}
static RcRegistryWrite begin_write(uint64_t backing_id, uint64_t offset) {
    RcRegistryWrite w;
    assert(rc_registry_write_begin(registry, (RcSourceRange){backing_id, offset, 8}, RC_REGISTRY_BUFFER, &w) == RC_REGISTRY_OK);
    return w;
}
static void mark_foreign(void *arg) {
    RcRegistryWrite *w = arg;
    for (unsigned i = 0; i < 2; i++) {
        assert(rc_registry_write_submit(registry, w[i], 6) == RC_REGISTRY_OK);
        assert(rc_registry_write_complete(registry, w[i], 0) == RC_REGISTRY_OK);
    }
    bitmap_done_count++;
}
static void add_bitmap_completion(RcRegistryWrite *w) {
    assert(callback_count < 8);
    callbacks[callback_count++] = (Callback){mark_foreign, NULL, w};
}
static void run_callbacks(void) {
    unsigned n = callback_count; callback_count = 0;
    for (unsigned i = 0; i < n; i++) { assert(callbacks[i].done); callbacks[i].done(callbacks[i].arg); }
}
static CpuVisibility visibility_job(void) {
    CpuVisibility j = {0};
    assert(rc_registry_visibility_begin(registry, RC_REGISTRY_VISIBILITY_ORDERED, &j.proof) == RC_REGISTRY_OK);
    j.proof.submission = 6;
    return j;
}
static void begin_registry(void) {
    registry = rc_registry_create((RcRegistryConfig){64, 64, 8});
    assert(registry);
    callback_count = apply_calls = bitmap_done_count = 0;
    capture.refs = 1; capture.current = 1;
    memset(actual_output, 0, sizeof actual_output);
}
static RcRegistryWrite *new_writers(void) {
    RcRegistryWrite *w = malloc(2 * sizeof *w); assert(w);
    w[0] = begin_write(RC_GUEST_COMPATIBILITY_BACKING, 0x1008);
    w[1] = begin_write(42, 0x208);
    return w;
}
static void release_registry(void) {
    assert(capture.refs == 1);
    rc_registry_destroy(registry); registry = NULL;
}

static void mapping_case(void) {
    struct VkMemNativeRmwBitmap owner = {0};
    owner.descriptor = descriptor; owner.provider = &capture;
    owner.source.native.interval.address = 0x1000;
    owner.source.native.interval.range = (RcSourceRange){42, 0x200, 64};
    owner.source.native.native.offset = 0;
    RcSourceRange guest, canonical;
    assert(native_rmw_bitmap_visibility_ranges(&owner, &pool, 8, 8, &guest, &canonical));
    assert(guest.backing == RC_GUEST_COMPATIBILITY_BACKING && guest.offset == 0x1008 && guest.size == 8);
    assert(canonical.backing == 42 && canonical.offset == 0x208 && canonical.size == 8);
    assert(!native_rmw_bitmap_visibility_ranges(&owner, &pool, 60, 8, &guest, &canonical));
    assert(copy_interval_calls == 0);
}

static void realloc_failure_case(void) {
    begin_registry();
    RcRegistryWrite *w = new_writers();
    CpuVisibility j = visibility_job();
    actual_output[8] = 0x81; // The already-recorded write remains; only visibility reconciliation is optional.
    RcSourceRange guest = {RC_GUEST_COMPATIBILITY_BACKING, 0x1008, 8};
    RcSourceRange canonical = {42, 0x208, 8};
    fail_next_realloc = 1;
    cpu_visibility_add_guarded_pair(&j, guest, canonical, &capture, &descriptor);
    assert(j.incomplete && !j.count && capture.refs == 1);
    add_bitmap_completion(w);
    cpu_visibility_queue(&j, 6);
    assert(callback_count == 1 && capture.refs == 1);
    run_callbacks();
    assert(bitmap_done_count == 1 && actual_output[8] == 0x81);
    assert(blocked(RC_GUEST_COMPATIBILITY_BACKING, 0x1008) && blocked(42, 0x208));
    free(w); release_registry();
}

static void wrapper_failure_case(void) {
    begin_registry();
    RcRegistryWrite *w = new_writers();
    CpuVisibility j = visibility_job();
    actual_output[8] = 0x92;
    cpu_visibility_add_guarded_pair(&j, (RcSourceRange){1,0x1008,8}, (RcSourceRange){42,0x208,8}, &capture, &descriptor);
    assert(!j.incomplete && j.count == 2 && capture.refs == 3);
    add_bitmap_completion(w);
    fail_next_malloc = 1;
    cpu_visibility_queue(&j, 6);
    assert(callback_count == 1 && capture.refs == 1);
    run_callbacks();
    assert(bitmap_done_count == 1 && actual_output[8] == 0x92);
    assert(blocked(1,0x1008) && blocked(42,0x208));
    free(w); release_registry();
}

static void ordered_success_case(void) {
    begin_registry();
    RcRegistryWrite *w = new_writers();
    CpuVisibility j = visibility_job();
    cpu_visibility_add_guarded_pair(&j, (RcSourceRange){1,0x1008,8}, (RcSourceRange){42,0x208,8}, &capture, &descriptor);
    assert(!j.incomplete && j.count == 2 && capture.refs == 3);
    add_bitmap_completion(w); // bitmap's exact foreign completion is registered before visibility.
    cpu_visibility_queue(&j, 6);
    assert(callback_count == 2);
    run_callbacks();
    assert(bitmap_done_count == 1 && capture.refs == 1);
    assert(!blocked(1,0x1008) && !blocked(42,0x208));
    free(w); release_registry();
}

static void stale_mapping_case(void) {
    begin_registry();
    RcRegistryWrite *w = new_writers();
    CpuVisibility j = visibility_job();
    cpu_visibility_add_guarded_pair(&j, (RcSourceRange){1,0x1008,8}, (RcSourceRange){42,0x208,8}, &capture, &descriptor);
    add_bitmap_completion(w); cpu_visibility_queue(&j, 6);
    assert(callback_count == 2 && capture.refs == 3);
    capture.current = 0;
    run_callbacks();
    assert(bitmap_done_count == 1 && capture.refs == 1);
    assert(blocked(1,0x1008) && blocked(42,0x208));
    free(w); release_registry();
}
int main(void) {
    mapping_case(); realloc_failure_case(); wrapper_failure_case(); ordered_success_case(); stale_mapping_case();
    puts("bitmap CPU visibility: exact retained pair, vector/wrapper allocation refusal, stale source, ordered completion and unknown authority passed");
    return 0;
}

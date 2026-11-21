"""Run the host mapping adapter and native mutation ordering with a mocked memory platform."""
from pathlib import Path
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ["CC"])
MEMORY = (ROOT / "runtime/memory.c").read_text()
HOST = (ROOT / "runtime/host.h").read_text()
VK = (ROOT / "runtime/vk.h").read_text()
VK_MEMORY = (ROOT / "runtime/vk_memory.c").read_text()

def section(first, last, body=MEMORY):
    start = body.index(first)
    return body[start:body.index(last, start + len(first))]

prefix = r"""
#include <assert.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "resource_core_backing.h"
#include "resource_core_registry.h"

typedef struct MemBackingCapture MemBackingCapture;
typedef int PlatMutex;
#define PLAT_MUTEX_INIT 0
static PlatMutex host_backing_topology_lock;
static unsigned topology_mutations;
typedef struct {uint64_t base, size;} RtRange;
static struct { RtRange heap, stack, alias; uint64_t pool; } L;
typedef struct {uint64_t dst, src, size;} StackMap;
enum { MAX_STACK_MAPS = 8, PAGE = 4096, K = 0x200000 };
#define HEAP_CAP (32ull * K)
#define RT_ADDRESS_LIMIT UINT64_MAX
#define PLAT_PROT_RW 1
#define PLAT_PROT_NONE 0
#define PERF_M_GMEM 1
#define R_OK 0u
#define R_INVALID_STATE 1u
#define R_INVALID_ADDRESS 2u
#define R_INVALID_REGION 3u
static uint64_t heap_size;
static uint32_t map_gen, nstack_maps;
static int heap_lazy;
static uint8_t *alias_pages;
static StackMap stack_maps[MAX_STACK_MAPS];
static int mapping, lock, frz_lock;
static int native_calls, fail_native, check_commit, check_decommit;
static int capacity_disabled;
static jmp_buf fatal_jump;
static uint64_t expected_addr, expected_size;
static MemBackingCapture *before_native;
static uint64_t retired_backing, changed_backing;
static RcSourceRange retired_ranges[128];
static RcSourceRange changed_ranges[128];
static RcSourceRange overwritten_range;
static uint32_t retired_count, changed_count, overwritten_count;
RcRegistry *rc_registry_runtime(void) { return NULL; }
RcRegistryResult rc_registry_range_gone(RcRegistry *registry, RcSourceRange range) {
    (void)registry;
    retired_backing = range.backing;
    assert(retired_count < sizeof retired_ranges / sizeof *retired_ranges);
    retired_ranges[retired_count++] = range;
    return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_cpu_changed(RcRegistry *registry, RcSourceRange range) {
    (void)registry;
    assert(lock && changed_count < sizeof changed_ranges / sizeof *changed_ranges);
    changed_backing = range.backing;
    changed_ranges[changed_count++] = range;
    return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_visibility_apply(RcRegistry *registry, const RcRegistryVisibility *proof, RcSourceRange range) {
    (void)registry;
    assert(lock == 2 && proof->mode == RC_REGISTRY_VISIBILITY_COMPLETED && proof->horizon == 42);
    assert(proof->writer_kinds == (1u << RC_REGISTRY_IMAGE));
    assert(changed_count && rc_range_equal(range, changed_ranges[changed_count - 1]));
    overwritten_range = range;
    overwritten_count++;
    return RC_REGISTRY_OK;
}
static void verify_native_order(uint64_t a, uint64_t n, int expectation);

static void plat_mutex_lock(int *m) {
    assert(!*m); *m = 1;
    if (m == &host_backing_topology_lock) topology_mutations++;
}
static void plat_mutex_unlock(int *m) {assert(*m == 1); *m = 0;}
static void plat_mutex_lock_shared(int *m) {assert(*m != 1); *m += 2;}
static void plat_mutex_unlock_shared(int *m) {assert(*m >= 2); *m -= 2;}
static void thaw_wait_locked(uint64_t a, uint64_t n) {(void)a; (void)n;}
static void thaw_range_locked(uint64_t a, uint64_t n, int why) {(void)a; (void)n; (void)why;}
static void rt_log(const char *fmt, ...) {(void)fmt;}
static _Noreturn void rt_fatal(const char *fmt, ...) {(void)fmt; assert(fail_native); longjmp(fatal_jump, 1);}
static int plat_mem_commit(uint64_t a, uint64_t n) {
    ++native_calls;
    if (check_commit) verify_native_order(a, n, 1);
    return !fail_native;
}
static int plat_mem_decommit(uint64_t a, uint64_t n) {
    ++native_calls;
    if (check_decommit) verify_native_order(a, n, 0);
    return !fail_native;
}
static int plat_mem_protect(uint64_t a, uint64_t n, int p) {(void)a; (void)n; (void)p; return 1;}
static uint64_t perf_mark_begin(void) {return 0;}
static void perf_mark_end(int m, uint64_t start, uint64_t bytes) {(void)m; (void)start; (void)bytes;}
static uint64_t image_bytes(void) {return 0;}
static void mem_mapping_publish(int *m, uint64_t a, uint64_t n, int busy) {
    (void)a; (void)n; assert(lock == 1); *m = busy;
}
static int inside(uint64_t a, uint64_t n, RtRange r) {
    return a >= r.base && n <= r.size && a - r.base <= r.size - n;
}
static int overlaps(uint64_t a, uint64_t n, uint64_t b, uint64_t m) {
    return n && m && a < b + m && b < a + n;
}
static int readable_locked(uint64_t a, uint64_t n) {
    for (uint32_t i = 0; i < nstack_maps; i++)
        if (overlaps(a, n, stack_maps[i].src, stack_maps[i].size)) return 0;
    return 1;
}
"""

capture_instrumentation = r"""
static int check_capture_locks, truncate_capture;
static RcBackingResult capture_intervals(RcBackingStore *store, RcBackingMapId map, uint64_t address,
                                          uint64_t size, RcInterval *out, uint32_t capacity, uint32_t *count) {
    if (check_capture_locks) assert(lock >= 2 && host_backing_topology_lock >= 2);
    if (truncate_capture == 1 && out && capacity > 32) capacity--;
    RcBackingResult result = rc_backing_map_intervals(store, map, address, size, out, capacity, count);
    if (truncate_capture == 2 && result == RC_BACKING_OK && out && capacity > 32 && *count)
        out[*count - 1].range.size--;
    return result;
}
#define rc_backing_map_intervals capture_intervals
"""

pool_receipt_types = r"""
#include "nvn_pool_index.h"
typedef uint64_t VkBuffer, VkDeviceSize;
typedef void *VmaAllocation;
typedef struct VkMemSpanCapture VkMemSpanCapture;
enum { POOL_IMPORT = 1 };
typedef struct {
    const void *key;
    uint64_t lo, size, import_lo, publication, host_mapping_cutoff;
    VkBuffer buf;
    MemBackingCapture *host_backing;
    unsigned refs;
    int unpublished, kind;
} PoolRec;
static pidx pool_index;
static int pool_lock;
static uint64_t pool_gen;
"""

pool_receipt_adapters = r"""
static void pool_native_acquire(PoolRec *pool) { assert(pool_lock == 2 && pool->refs); ++pool->refs; }
static int mem_mapping_range_unchanged(uint64_t address, uint64_t size, uint64_t cutoff) {
    return address && size && cutoff == 1;
}
static RcRegistryResult import_mapping_current(const PoolRec *pool, const MemBackingCapture *receipt,
                                               uint64_t address, uint64_t size) {
    assert(pool_lock == 2);
    const MemBackingCapture *expected = receipt ? receipt : pool->host_backing;
    return mem_backing_current_range(expected, address, size) ? RC_REGISTRY_OK : RC_REGISTRY_STALE;
}
static int span_capture_match(const VkMemSpanCapture *capture, const VkMemSpan *selected) {
    return capture->provider == selected->pool && capture->selected.buffer == selected->buffer &&
           import_mapping_current(capture->provider, capture->host, capture->address, capture->size) == RC_REGISTRY_OK;
}
"""

fixture = r"""
static void verify_native_order(uint64_t a, uint64_t n, int commit) {
    assert(a == expected_addr && n == expected_size && lock == 1 && frz_lock == 1);
    if (capacity_disabled) {
        assert(!commit && host_backing_status() == RC_BACKING_CAPACITY);
        return;
    }
    RcInterval current;
    uint32_t count;
    assert(rc_backing_map_intervals(host_backing_store, host_backing_map, a, n, &current, 1, &count) == RC_BACKING_OK);
    assert(count == 1 && !current.range.backing && current.range.size == n);
    if (before_native)
        assert(!host_backing_range_current_locked(before_native, before_native->address, before_native->size));
}

static void start(void) {
    assert(!host_backing_store);
    host_backing_init();
    assert(host_backing_status() == RC_BACKING_OK);
}
static void stop(void) {
    assert(rc_backing_map_close(host_backing_store, host_backing_map) == RC_BACKING_OK);
    assert(rc_backing_store_destroy(host_backing_store) == RC_BACKING_OK);
    host_backing_store = NULL;
    host_backing_map = (RcBackingMapId){0};
    host_backing_failure = RC_BACKING_OK;
    lock = frz_lock = native_calls = check_commit = check_decommit = fail_native = capacity_disabled = 0;
    mapping = heap_size = heap_lazy = map_gen = nstack_maps = 0;
    before_native = NULL;
    retired_count = changed_count = overwritten_count = 0;
}
static MemBackingCapture *capture(uint64_t a, uint64_t n) {
    MemBackingCapture *owned = NULL;
    assert(mem_backing_capture(a, n, &owned) == RC_BACKING_OK && owned);
    return owned;
}
static RcInterval first(const MemBackingCapture *owned) {
    uint32_t count = 0;
    const RcInterval *intervals = mem_backing_intervals(owned, &count);
    assert(count == 1 && intervals && intervals[0].mapping && intervals[0].range.backing > 1);
    return intervals[0];
}

static void heap_lifetimes(void) {
    start();
    L.heap = (RtRange){4 * K, HEAP_CAP};
    L.pool = HEAP_CAP;
    MemBackingCapture *unavailable;
    assert(mem_backing_capture(L.heap.base, PAGE, &unavailable) == RC_BACKING_ABSENT && !unavailable);
    check_commit = 1;
    expected_addr = L.heap.base;
    expected_size = 2 * K;
    assert(mem_set_heap_size(2 * K, NULL) == R_OK);
    check_commit = 0;
    MemBackingCapture *left = capture(L.heap.base, K), *right = capture(L.heap.base + K, K);
    MemBackingCapture *overlap = capture(L.heap.base + K + PAGE, PAGE);
    RcInterval old = first(right), same = first(overlap);
    assert(old.range.backing == same.range.backing && old.mapping == same.mapping);
    assert(same.range.offset == old.range.offset + PAGE);
    assert(mem_backing_read_hold(left) && lock == 2);
    mem_read_release();

    check_decommit = 1;
    expected_addr = L.heap.base + K;
    expected_size = K;
    before_native = right;
    assert(mem_set_heap_size(K, NULL) == R_OK);
    check_decommit = 0;
    before_native = NULL;
    assert(mem_backing_current(left) && !mem_backing_current(right));
    assert(!mem_backing_read_hold(overlap) && !lock);
    assert(first(right).range.backing == old.range.backing);
    assert(mem_backing_capture(L.heap.base + K, K, &unavailable) == RC_BACKING_ABSENT && !unavailable);

    check_commit = 1;
    assert(mem_set_heap_size(2 * K, NULL) == R_OK);
    check_commit = 0;
    MemBackingCapture *replacement = capture(L.heap.base + K, K);
    RcInterval now = first(replacement);
    assert(now.range.backing != old.range.backing && now.mapping != old.mapping && now.range.offset == 0);
    assert(!mem_backing_current(right) && mem_backing_current(replacement));
    assert(retired_backing == old.range.backing);
    mem_backing_release(&overlap);
    mem_backing_release(&right);
    mem_backing_release(&replacement);
    mem_backing_release(&left);
    stop();
}

static void failed_commit_is_not_published(void) {
    start();
    L.heap = (RtRange){4 * K, HEAP_CAP};
    L.pool = HEAP_CAP;
    check_commit = fail_native = 1;
    expected_addr = L.heap.base;
    expected_size = K;
    if (!setjmp(fatal_jump)) {
        mem_set_heap_size(K, NULL);
        assert(0 && "native commit failure returned");
    }
    lock = frz_lock = 0; // The production fatal path never returns to the guest.
    assert(heap_size == 0);
    MemBackingCapture *bad;
    assert(mem_backing_capture(L.heap.base, K, &bad) == RC_BACKING_ABSENT && !bad);
    stop();
}

static void invalidation_capacity_disables_normalization_first(void) {
    host_backing_store = rc_backing_store_create((RcBackingConfig){2, 1, 1, 2, 2});
    assert(host_backing_store);
    assert(rc_backing_storage_map_create(host_backing_store, HOST_STORAGE_NAMESPACE, 0, 0,
                                         RT_ADDRESS_LIMIT, &host_backing_map) == RC_BACKING_OK);
    uint8_t pages[3] = {0};
    alias_pages = pages;
    L.alias = (RtRange){8 * K, sizeof pages * PAGE};
    lock = 1;
    assert(alias_runs(0, 3, 1) == 3 * PAGE);
    lock = 0;
    MemBackingCapture *old = capture(L.alias.base, PAGE);
    check_decommit = capacity_disabled = 1;
    expected_addr = L.alias.base + PAGE;
    expected_size = PAGE;
    lock = 1;
    assert(alias_runs(1, 1, 0) == PAGE);
    lock = 0;
    assert(!pages[1] && !mem_backing_current(old));
    MemBackingCapture *bad;
    assert(mem_backing_capture(L.alias.base, PAGE, &bad) == RC_BACKING_CAPACITY && !bad);
    mem_backing_release(&old);
    stop();
}

static void stack_copy_and_late_publication(void) {
    start();
    static _Alignas(PAGE) uint8_t src[PAGE], dst[PAGE];
    L.heap = (RtRange){(uintptr_t)src, PAGE};
    L.stack = (RtRange){(uintptr_t)dst, PAGE};
    heap_size = PAGE;
    mem_backing_publish_module((uintptr_t)src, PAGE); // Model a stable region whose allocation has already succeeded.
    MemBackingCapture *old = capture((uintptr_t)src, PAGE);
    src[0] = 7;
    assert(stack_move(0, (uintptr_t)dst, (uintptr_t)src, PAGE) == R_OK);
    MemBackingCapture *moved = capture((uintptr_t)dst, PAGE);
    assert(first(moved).range.backing != first(old).range.backing && dst[0] == 7);
    assert(mem_backing_current(old) && !mem_backing_read_hold(old));
    dst[0] = 9;
    check_decommit = 1;
    expected_addr = (uintptr_t)dst;
    expected_size = PAGE;
    before_native = moved;
    assert(stack_move(1, (uintptr_t)dst, (uintptr_t)src, PAGE) == R_OK);
    assert(!mem_backing_current(moved) && mem_backing_current(old));
    assert(changed_backing == first(old).range.backing && retired_backing == first(moved).range.backing);
    assert(mem_backing_read_hold(old) && src[0] == 9);
    mem_read_release();
    mem_backing_release(&old);
    mem_backing_release(&moved);
    stop();
}

static void lazy_heap_is_logically_mapped(void) {
    start();
    L.heap = (RtRange){4 * K, HEAP_CAP};
    L.pool = HEAP_CAP;
    heap_lazy = 1;
    assert(mem_set_heap_size(K, NULL) == R_OK && native_calls == 0);
    MemBackingCapture *logical = capture(L.heap.base, PAGE);
    assert(mem_backing_current(logical) && first(logical).range.offset == 0);
    mem_backing_release(&logical);
    stop();
}

static void host_values_do_not_consume_core_capture_slots(void) {
    host_backing_store = rc_backing_store_create((RcBackingConfig){2, 1, 2, 1, 1});
    assert(host_backing_store);
    assert(rc_backing_storage_map_create(host_backing_store, HOST_STORAGE_NAMESPACE, 0, 0, RT_ADDRESS_LIMIT,
                                         &host_backing_map) == RC_BACKING_OK);
    L.heap = (RtRange){4 * K, HEAP_CAP};
    L.pool = HEAP_CAP;
    assert(mem_set_heap_size(2 * K, NULL) == R_OK);
    RcBackingCapture *provider = NULL, *excess = NULL;
    uint64_t address = L.heap.base + K;
    assert(rc_backing_capture(host_backing_store, host_backing_map, address, K, RC_BACKING_REQUIRE_MAPPED,
                              &provider) == RC_BACKING_OK);
    assert(rc_backing_capture(host_backing_store, host_backing_map, address, K, RC_BACKING_REQUIRE_MAPPED,
                              &excess) == RC_BACKING_CAPACITY);
    enum { NVALUES = 8193 };
    MemBackingCapture *owned[NVALUES];
    for (unsigned i = 0; i < NVALUES; i++) owned[i] = capture(address, PAGE);
    RcInterval predecessor = first(owned[0]);
    assert(rc_backing_store_usage(host_backing_store).captures == 1);
    check_decommit = 1;
    expected_addr = address;
    expected_size = K;
    before_native = owned[0];
    assert(mem_set_heap_size(K, NULL) == R_OK);
    check_decommit = 0;
    before_native = NULL;
    assert(retired_count == 1 && retired_ranges[0].backing == predecessor.range.backing);
    assert(!mem_backing_read_hold_at(owned[NVALUES - 1], address, PAGE));
    assert(host_backing_status() == RC_BACKING_OK);
    rc_backing_capture_release(&provider);
    assert(mem_set_heap_size(2 * K, NULL) == R_OK);
    MemBackingCapture *replacement = capture(address, PAGE);
    assert(first(replacement).range.backing != predecessor.range.backing);
    assert(first(replacement).mapping != predecessor.mapping);
    assert(!mem_backing_current(owned[0]) && mem_backing_read_hold(replacement));
    mem_read_release();
    mem_backing_release(&replacement);
    for (unsigned i = 0; i < NVALUES; i++) mem_backing_release(&owned[i]);
    stop();
}

static void notification_batches_cover_the_entire_sparse_range(void) {
    start();
    enum { MAPPED = 80 };
    uint64_t address = 12 * K;
    RcSourceRange expected[MAPPED];
    for (unsigned i = 0; i < MAPPED; i++) {
        uint64_t at = address + (2ull * i) * PAGE;
        mem_backing_publish_module(at, PAGE);
        MemBackingCapture *old = capture(at, PAGE);
        expected[i] = first(old).range;
        mem_backing_release(&old);
    }
    assert(mem_backing_capture(address, 2ull * MAPPED * PAGE, &before_native) == RC_BACKING_ABSENT);
    mem_backing_cpu_changed(address, 2ull * MAPPED * PAGE);
    assert(!lock && changed_count == MAPPED);
    for (unsigned i = 0; i < MAPPED; i++) assert(rc_range_equal(expected[i], changed_ranges[i]));
    MemBackingCapture *selected = capture(address, PAGE);
    assert(mem_backing_read_hold(selected));
    mem_backing_cpu_changed_held(selected, address + 8, 12, NULL);
    assert(lock == 2 && changed_count == MAPPED + 1);
    assert(rc_range_equal(changed_ranges[MAPPED], (RcSourceRange){expected[0].backing, expected[0].offset + 8, 12}));
    assert(overwritten_count == 0);
    const RcRegistryVisibility exact = {.horizon = 42, .mode = RC_REGISTRY_VISIBILITY_COMPLETED,
                                         .writer_kinds = 1u << RC_REGISTRY_IMAGE};
    mem_backing_cpu_changed_held(selected, address, 24, &exact);
    assert(overwritten_count == 1 && rc_range_equal(overwritten_range,
                                                   (RcSourceRange){expected[0].backing, expected[0].offset, 24}));
    mem_read_release();
    mem_backing_release(&selected);
    plat_mutex_lock(&lock);
    host_backing_invalidate(address, 2ull * MAPPED * PAGE);
    plat_mutex_unlock(&lock);
    assert(retired_count == MAPPED && host_backing_status() == RC_BACKING_OK);
    for (unsigned i = 0; i < MAPPED; i++) assert(rc_range_equal(expected[i], retired_ranges[i]));
    assert(!rc_backing_store_usage(host_backing_store).backings);
    stop();
}

static void bounded_receipts_cover_the_full_projection(void) {
    start();
    enum { PAGES = 80 };
    uint64_t address = 12 * K;
    unsigned mutations = topology_mutations;
    for (unsigned i = 0; i < PAGES; ++i)
        host_backing_publish(address + (uint64_t)i * PAGE, PAGE); // Kernel allocator can publish outside lock.
    assert(topology_mutations == mutations + PAGES && !host_backing_topology_lock && !lock);

    MemBackingCapturePlan planned;
    MemBackingCapture *value = NULL;
    check_capture_locks = 1;
    assert(mem_backing_capture_plan(address, PAGES * PAGE, &planned) == RC_BACKING_OK);
    assert(planned.count == PAGES && planned.bytes == sizeof(MemBackingCapture) + PAGES * sizeof(RcInterval));
    host_backing_publish(address + (PAGES + 1) * PAGE, PAGE);
    assert(mem_backing_capture_bounded(&planned, &value) == RC_BACKING_OK);
    mem_backing_release(&value);
    for (truncate_capture = 1; truncate_capture <= 2; ++truncate_capture) {
        assert(mem_backing_capture_bounded(&planned, &value) == RC_BACKING_STALE && !value);
        assert(!lock && !host_backing_topology_lock);
    }
    truncate_capture = 0;
    assert(mem_backing_capture_bounded(&planned, &value) == RC_BACKING_OK);
    uint32_t count;
    const RcInterval *intervals = mem_backing_intervals(value, &count);
    assert(count == PAGES && intervals[count - 1].address + intervals[count - 1].range.size == address + PAGES * PAGE);
    uint64_t old_backing = intervals[0].range.backing;
    mem_backing_release(&value);

    check_capture_locks = 0;
    plat_mutex_lock(&lock);
    host_backing_invalidate(address, PAGE);
    host_backing_publish(address, PAGE);
    plat_mutex_unlock(&lock);
    check_capture_locks = 1;
    assert(mem_backing_capture_bounded(&planned, &value) == RC_BACKING_STALE && !value);
    assert(mem_backing_capture_plan(address, PAGES * PAGE, &planned) == RC_BACKING_OK && planned.count == PAGES);
    assert(mem_backing_capture_bounded(&planned, &value) == RC_BACKING_OK);
    intervals = mem_backing_intervals(value, &count);
    assert(count == PAGES && intervals[0].range.backing != old_backing);
    mem_backing_release(&value);
    check_capture_locks = 0;
    stop();
}

static void pool_receipts_bind_the_indexed_incarnation(void) {
    start();
    uint64_t address = 12 * K;
    host_backing_publish(address, PAGE);
    PoolRec main_pool = {.lo = address, .size = PAGE, .import_lo = address, .buf = 1,
        .host_backing = capture(address, PAGE), .refs = 1, .kind = POOL_IMPORT};
    main_pool.key = &main_pool;
    plat_mutex_lock(&pool_lock);
    index_put(&main_pool);
    plat_mutex_unlock(&pool_lock);

    MemBackingCapturePlan host;
    VkMemSpanCapturePlan planned;
    VkMemSpanCapture *value = NULL;
    VkMemSpan selected = {.buffer = main_pool.buf, .avail = PAGE, .pool = &main_pool};
    assert(mem_backing_capture_plan(address, PAGE, &host) == RC_BACKING_OK);
    assert(vk_mem_span_capture_plan(&selected, &host, &planned) == RC_REGISTRY_OK);
    assert(planned.bytes == sizeof(VkMemSpanCapture) && !planned.nested_host);
    uint64_t observed = pool_gen;
    PoolRec other = {.lo = address + 2 * PAGE, .size = PAGE, .buf = 2, .kind = POOL_IMPORT};
    other.key = &other;
    plat_mutex_lock(&pool_lock);
    index_put(&other);
    plat_mutex_unlock(&pool_lock);
    assert(pool_gen != observed);
    assert(vk_mem_span_capture_bounded(&planned, &host, &value) == RC_REGISTRY_OK);
    assert(value && main_pool.refs == 2 && !value->host);
    free(value);
    value = NULL;
    --main_pool.refs;

    // Reusing the same pool address and the same native handle is still a new publication.
    plat_mutex_lock(&pool_lock);
    assert(pidx_remove(&pool_index, main_pool.key, NULL));
    index_put(&main_pool);
    plat_mutex_unlock(&pool_lock);
    assert(vk_mem_span_capture_bounded(&planned, &host, &value) == RC_REGISTRY_STALE && !value);

    mem_backing_release(&main_pool.host_backing);
    main_pool.host_mapping_cutoff = 1;
    assert(vk_mem_span_capture_plan(&selected, &host, &planned) == RC_REGISTRY_OK);
    assert(planned.bytes == sizeof(VkMemSpanCapture) + host.bytes && planned.nested_host);
    assert(vk_mem_span_capture_bounded(&planned, &host, &value) == RC_REGISTRY_OK);
    assert(value && value->host && main_pool.refs == 2);
    mem_backing_release(&value->host);
    free(value);
    free(pool_index.e);
    free(pool_index.maxhi);
    pool_index = (pidx){0};
    stop();
}

int main(void) {
    heap_lifetimes();
    failed_commit_is_not_published();
    invalidation_capacity_disables_normalization_first();
    stack_copy_and_late_publication();
    lazy_heap_is_logically_mapped();
    host_values_do_not_consume_core_capture_slots();
    notification_batches_cover_the_entire_sparse_range();
    bounded_receipts_cover_the_full_projection();
    pool_receipts_bind_the_indexed_incarnation();
    puts("host backing: ordering, aliases, retained values, full invalidation and bounded receipt projections passed");
    return 0;
}
"""

source = "\n".join((
    prefix,
    HOST[HOST.index("typedef struct MemBackingCapturePlan"):HOST.index("RcBackingResult mem_backing_capture_plan")],
    section("static void commit(", "// The process is re-run"),
    section("// A single sparse storage map", "void mem_init(void)"),
    section("uint32_t mem_set_heap_size(", "// ---- module data permissions"),
    section("static int alias_all_mapped(", "uint32_t mem_map(uint64_t dst"),
    section("static uint64_t alias_runs(", "static uint32_t alias_map("),
    capture_instrumentation,
    section("void mem_read_release(void)", "uint32_t mem_query("),
    pool_receipt_types,
    section("// GPU address -> backing", "// Traffic census", VK),
    section("struct MemBackingCapturePlan;", "// Uniform snapshots reserve", VK),
    section("struct VkMemSpanCapture {", "static uint64_t mirror_import_scene", VK_MEMORY),
    pool_receipt_adapters,
    section("static void index_put(", "static PoolRec *phys_find", VK_MEMORY),
    section("// pool_lock held. Indexed containment", "RcRegistryResult vk_mem_span_capture_bound_input", VK_MEMORY),
    fixture,
))
work = ROOT / "build/test"
work.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="host-backing-", dir=work) as directory:
    test = Path(directory) / "host_backing.c"
    test.write_text(source)
    exe = test.with_suffix(".exe")
    subprocess.run([str(CC), "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fsanitize-trap=all", "-DRC_BACKING_SINGLE_THREADED",
                    "-I", str(ROOT / "runtime"), str(test), str(ROOT / "runtime/resource_core_backing.c"),
                    "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

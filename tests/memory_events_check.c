#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "mem_mapping.h"

#define FRZ_SHIFT 21
#define FRZ_SIZE (1ull << FRZ_SHIFT)
#define RT_ADDRESS_LIMIT (8 * FRZ_SIZE)
#define PAGE 4096ull
#define FRZ_THAWING 2
#define FRZ_HOLD_MS 5000u
#define FRZ_QUICK_MS 60000u
#define FRZ_HOT_MIN_MS 4000u
#define FRZ_HOT_MAX_MS 600000u
#define PLAT_MEM_RESERVED -1
static volatile uint8_t frz[8];
static uint32_t thaw_ms[8], frz_ms[8], frz_hot_ms[8], frz_hot_until[8];
static uint8_t lzr[8], prb[8];
static int frz_lock;
static int32_t nfrz;
static int64_t frz_count[4], frz_hot_holds;
static ResourceEvents protection_events;
static uint64_t protection_regions[8];
static MemMapping protection = {protection_regions, 2, 8, FRZ_SHIFT, 0, &protection_events};
static struct { struct { uint64_t base, size; } heap; } L;
static uint64_t heap_size, heap_lazy_bytes;
static int freeze_succeeds = 1, pages_writable = 1, commits;
static uint64_t clock_ms = 100000;
static void thaw_end_locked(uint64_t r, int why);
static int mem_frozen_stable(uint64_t addr);
static void protection_publish(uint64_t region, int busy);
static uint64_t mem_protection_region(uint32_t region);
static uint64_t plat_now_ms(void) { return clock_ms; }
static void plat_mutex_lock(int *p) { assert(!*p); *p = 1; }
static void plat_mutex_unlock(int *p) { assert(*p); *p = 0; }
static void plat_inc32(int32_t *p) { ++*p; }
static void plat_dec32(int32_t *p) { --*p; }
static void plat_inc64(int64_t *p) { ++*p; }
static int watched(uint64_t a, uint64_t n) { return a < RT_ADDRESS_LIMIT && n <= RT_ADDRESS_LIMIT - a; }
static uint64_t align_up(uint64_t a, uint64_t n) { return (a + n - 1) & ~(n - 1); }
static void lazy_drop_region_locked(uint64_t r, int why) { (void)why; assert(!mem_frozen_stable(r << FRZ_SHIFT)); lzr[r] = 0; }
static void unprobe_locked(uint64_t r) { assert(!mem_frozen_stable(r << FRZ_SHIFT)); prb[r] = 0; }
static int receiving_region = -1;
static int region_receiving_file_data(uint32_t region) { return region == (uint32_t)receiving_region; }
static int freeze_locked(uint64_t r) {
    assert(frz_lock && !mem_protection_region((uint32_t)r));
    if (freeze_succeeds) pages_writable = 0;
    return freeze_succeeds;
}
static void thaw_pages(uint64_t r, int strict) {
    (void)strict;
    assert(!mem_protection_region((uint32_t)r));
    assert(!mem_frozen_stable(r << FRZ_SHIFT));
    pages_writable = 1;
}
static void thaw_range_locked(uint64_t a, uint64_t n, int why) { (void)a; (void)n; (void)why; }
static int plat_mem_state(uint64_t a, uint64_t end, uint64_t *next) { (void)a; *next = end; return PLAT_MEM_RESERVED; }
static int plat_mem_commit(uint64_t a, uint64_t n) {
    for (uint64_t r = a >> FRZ_SHIFT; r <= (a + n - 1) >> FRZ_SHIFT; r++) assert(!mem_protection_region((uint32_t)r));
    commits++; return 1;
}
#define rt_fatal(...) assert(!"Unexpected fatal path")
#include "memory_events_under_test.h"

static uint32_t notified;
static unsigned visits;
static void visit(uint32_t first, uint32_t last, void *unused) {
    (void)unused;
    assert(first <= last && last < 8);
    for (uint32_t i = first; i <= last; i++) notified |= 1u << i;
    visits++;
}

int main(void) {
    uint64_t cursor = 0, other = 0;
    assert(mem_freeze(0) && mem_frozen_stable(0) && !pages_writable);
    uint64_t before = mem_protection_serial();
    assert(mem_protection_events(&cursor, visit, NULL) && visits == 2 && notified == 1);
    assert(mem_protection_events(&other, visit, NULL) && other == cursor);
    assert(mem_protection_range_unchanged(0, FRZ_SIZE, before));
    // Synchronous thaw must revoke the proof before changing the pages' protection.
    plat_mutex_lock(&frz_lock);
    lzr[0] = prb[0] = 1;
    thaw_locked(0, 1);
    plat_mutex_unlock(&frz_lock);
    assert(pages_writable && !frz[0] && !mem_frozen_stable(0));
    assert(!mem_protection_range_unchanged(0, FRZ_SIZE, before));
    assert(mem_protection_events(&cursor, visit, NULL));
    clock_ms += 700000;
    assert(mem_freeze(0));
    before = mem_protection_serial();
    // The asynchronous path publishes busy while the old frozen byte is still visible.
    plat_mutex_lock(&frz_lock);
    assert(thaw_begin_locked(0, 1) && frz[0] == 1 && !mem_frozen_stable(0));
    __atomic_store_n(&frz[0], FRZ_THAWING, __ATOMIC_RELEASE);
    plat_mutex_unlock(&frz_lock);
    thaw_pages(0, 0);
    assert(!mem_protection_region(0) && !mem_frozen_stable(0));
    plat_mutex_lock(&frz_lock);
    thaw_end_locked(0, 1);
    uint64_t done = mem_protection_serial();
    thaw_end_locked(0, 1);
    assert(mem_protection_serial() == done); // another completion does not manufacture a new transition
    plat_mutex_unlock(&frz_lock);
    clock_ms += 700000;
    assert(mem_freeze(0) && !mem_protection_range_unchanged(0, FRZ_SIZE, before));
    receiving_region = 1;
    assert(!mem_freeze(FRZ_SIZE) && !mem_frozen_stable(FRZ_SIZE));
    receiving_region = -1;
    // Failed freeze cannot leave a busy stamp or claim a frozen region.
    freeze_succeeds = 0;
    assert(!mem_freeze(FRZ_SIZE) && mem_protection_region(1) && !mem_frozen_stable(FRZ_SIZE));
    freeze_succeeds = 1;
    // Lazy commits have no guest mapping publication, but revoke absence through protection events.
    L.heap.base = FRZ_SIZE; L.heap.size = heap_size = FRZ_SIZE;
    before = mem_protection_serial();
    assert(heap_lazy_commit_locked(FRZ_SIZE, PAGE) == PAGE && commits == 1 && heap_lazy_bytes == PAGE);
    assert(!mem_protection_range_unchanged(FRZ_SIZE, PAGE, before));
    assert(mem_protection_range_unchanged(0, PAGE, before));
    notified = 0;
    assert(mem_protection_events(&cursor, visit, NULL) && (notified & 2));
    // Lost history forces a rebuild, then independent cursors can resume normally.
    for (unsigned i = 0; i <= RESOURCE_EVENTS_CAPACITY; i++) protection_publish(2, 0);
    assert(!mem_protection_events(&cursor, visit, NULL));
    assert(mem_protection_events(&cursor, visit, NULL));
    before = mem_protection_serial();
    mem_mapping_publish(&protection, RT_ADDRESS_LIMIT, PAGE, 1);
    assert(!mem_protection_serial() && !mem_frozen_stable(0));
    assert(!mem_protection_range_unchanged(0, PAGE, before));
    assert(!mem_protection_events(&cursor, visit, NULL));
    puts("memory events: actual freeze/thaw transitions, strict readiness, lazy commits, independent readers and fail-closed history passed");
    return 0;
}

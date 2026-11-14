#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "mem_mapping.h"
#include "fresh_deps.h"

#define SNAP_SHIFT 21
#define SNAP_REGIONS 8
#define RT_ADDRESS_LIMIT (8ull << SNAP_SHIFT)
static uint64_t mapping_versions[8], protection_versions[8];
static ResourceEvents mapping_events, protection_events, content_events;
static MemMapping mapping = {mapping_versions, 2, 8, 21, 0, &mapping_events};
static MemMapping protection = {protection_versions, 2, 8, 21, 0, &protection_events};
static uint8_t frozen[8], written[8];
static uint32_t snap_at[8], reg_max[8], reg_last[8], reg_tried[8], gen_now, ww_ep;
static uint8_t snap_untracked[8];
static uint64_t snap_protection[8], snap_mapping[8], gen_pages[8], n_snap_calls;
static int dependency_watch = 1, freeze_idle, nfcand, cfcand;
static uint32_t *fcand;
static unsigned queries, notifications, race, untracked;
static int observer_race;
static void transition(unsigned r, unsigned value);
static uint64_t content_serial = 1;
static FreshDeps *pool_deps;
static uint32_t pool_deps_epoch, submit_epoch;
static uint64_t pool_deps_map, pool_deps_protection, pool_deps_serial;
static uint64_t pool_map_cursor, pool_protection_cursor, pool_content_cursor, n_presnap_q;
static struct {
    uint64_t regions, mapping_regions, mapping_notifications, notifications;
    uint64_t dormant_visits, history_fallbacks, absence_learned, absence_reused;
} PD;
static uint64_t mem_mapping_serial(void) { return mem_mapping_clock(&mapping); }
static uint64_t mem_protection_serial(void) { return mem_mapping_clock(&protection); }
static uint64_t mem_mapping_region(uint32_t r) { return mem_mapping_version(&mapping, r); }
static uint64_t mem_protection_region(uint32_t r) { return mem_mapping_version(&protection, r); }
static int mem_mapping_events(uint64_t *cursor, ResourceEventVisit visit, void *user) {
    return resource_events_read(&mapping_events, cursor, visit, user) && mem_mapping_serial();
}
static int mem_protection_events(uint64_t *cursor, ResourceEventVisit visit, void *user) {
    return resource_events_read(&protection_events, cursor, visit, user) && mem_protection_serial();
}
static int vk_mem_dependency_events_locked(uint64_t *cursor, ResourceEventVisit visit, void *user) {
    int complete = resource_events_read(&content_events, cursor, visit, user);
    if (observer_race) {
        transition(0, 0); written[0] = 1; transition(0, 1);
        observer_race = 0;
    }
    return complete;
}
static uint64_t vk_mem_dependency_serial(void) { return content_serial; }
static int mem_mapping_range_unchanged(uint64_t a, uint64_t n, uint64_t v) { return mem_mapping_unchanged(&mapping, a, n, v); }
static int mem_protection_range_unchanged(uint64_t a, uint64_t n, uint64_t v) { return mem_mapping_unchanged(&protection, a, n, v); }
static int mem_frozen_stable(uint64_t a) { return frozen[a >> 21] == 1 && mem_mapping_version(&protection, (uint32_t)(a >> 21)); }
static void dependency_changed(uint32_t r) { notifications++; content_serial++; resource_events_publish(&content_events, r, r); }
static void stamp_page(uint64_t a, uint32_t gen) { reg_max[a >> 21] = gen; dependency_changed((uint32_t)(a >> 21)); }
static void grow(void **p, int *capacity, size_t size) { (void)p; (void)capacity; (void)size; assert(!"unexpected allocation"); }
static void transition(unsigned r, unsigned value) {
    mem_mapping_publish(&protection, (uint64_t)r << 21, 1ull << 21, 1);
    frozen[r] = (uint8_t)value;
    mem_mapping_publish(&protection, (uint64_t)r << 21, 1ull << 21, 0);
}
static uint64_t ww_query(uint64_t a, uint64_t size) {
    queries++;
    if (untracked) return UINT64_MAX;
    uint64_t n = 0;
    for (unsigned r = (unsigned)(a >> 21); r < (a + size) >> 21; r++) if (written[r]) {
        gen_pages[n++] = (uint64_t)r << 21; written[r] = 0;
    }
    if (race == 1) transition((unsigned)(a >> 21), 0);
    if (race == 2) {
        mem_mapping_publish(&mapping, a, size, 1);
        mem_mapping_publish(&mapping, a, size, 0);
    }
    race = 0;
    return n;
}
typedef struct {
    void *ce;
    uint32_t cgen;
    uint8_t finalized;
    uint64_t absent_mapping, absent_protection;
    struct { void *mem; struct { uint64_t size; } L; } t;
} TexSlot;
static int readable, alias, by_addr, read_race;
static unsigned read_calls;
static void *nvn_hm_get(int *map, const uint64_t *key) { (void)map; (void)key; return alias ? &alias : NULL; }
static int mem_readable(uint64_t a, uint64_t size) {
    read_calls++;
    if (read_race) {
        // Lazy commitment changes backing without publishing in the guest mapping domain.
        mem_mapping_publish(&protection, a, size, 1);
        mem_mapping_publish(&protection, a, size, 0);
        read_race = 0;
    }
    return readable;
}
#include "resource_consumers_under_test.h"

static void observer_integration(void) {
    protection.disabled = 0;
    transition(0, 1);
    pool_deps = fresh_deps_create(8, 21); assert(pool_deps);
    uint32_t id = fresh_deps_add(pool_deps, 1, 0, 4096), n;
    assert(id);
    submit_epoch = snap_at[0] + 1;
    pool_deps_observe(1);
    assert(vk_mem_dependency_sleep_locked(0));
    assert(fresh_deps_dirty(pool_deps, id));
    assert(!vk_mem_unchanged_why_locked(0, 4096, gen_now, submit_epoch));
    fresh_deps_clean(pool_deps, id);
    // Consume any notifications generated by that query. Afterwards the ordinary observer does no regional scan
    // and the existing authoritative freshness predicate still accepts the protected snapshot in newer epochs.
    submit_epoch++; pool_deps_observe(1);
    fresh_deps_poll_regions(pool_deps, &n); assert(!n);
    uint64_t visits = PD.regions;
    unsigned calls = queries;
    for (unsigned i = 0; i < 1000; i++) {
        submit_epoch++; pool_deps_observe(1);
        assert(!fresh_deps_dirty(pool_deps, id));
        assert(!vk_mem_unchanged_why_locked(0, 4096, gen_now, submit_epoch));
    }
    assert(PD.regions == visits && queries == calls);
    // Revocation after the notification cutoff is after this submission's logical observation. Adopting the
    // dormant snapshot keeps the old content stable; repeated observations must retain the wake for next epoch.
    observer_race = 1;
    uint32_t prior = reg_max[0];
    submit_epoch++; pool_deps_observe(1);
    assert(queries == calls && reg_max[0] == prior && written[0]);
    pool_deps_observe(1);
    assert(queries == calls && !vk_mem_unchanged_why_locked(0, 4096, prior, submit_epoch));
    fresh_deps_poll_regions(pool_deps, &n); assert(n == 1);
    pool_deps_observe(1);
    fresh_deps_poll_regions(pool_deps, &n); assert(n == 1 && queries == calls);
    submit_epoch++; pool_deps_observe(1);
    assert(queries > calls && reg_max[0] > prior && !written[0] && fresh_deps_dirty(pool_deps, id));
    fresh_deps_clean(pool_deps, id);
    submit_epoch++; pool_deps_observe(1);
    visits = PD.regions; calls = queries;
    // Unrelated events do not wake the subscribed region. A full thaw/write/refreeze does, with content observed
    // before its dependent node is eligible to be cleaned again.
    transition(3, 0); submit_epoch++; pool_deps_observe(1);
    assert(PD.regions == visits && !fresh_deps_dirty(pool_deps, id));
    transition(0, 0); written[0] = 1; transition(0, 1);
    submit_epoch++; pool_deps_observe(1);
    assert(PD.regions > visits && queries > calls && fresh_deps_dirty(pool_deps, id));
    fresh_deps_clean(pool_deps, id);
    submit_epoch++; pool_deps_observe(1);
    // A journal overflow forces full observation, even if all surviving notifications concern other regions.
    for (unsigned i = 0; i < RESOURCE_EVENTS_CAPACITY + 1; i++) transition(3, 0);
    visits = PD.regions;
    submit_epoch++; pool_deps_observe(1);
    assert(PD.regions > visits && fresh_deps_dirty(pool_deps, id));
    fresh_deps_remove(pool_deps, id);
    fresh_deps_poll_regions(pool_deps, &n); assert(!n);
    id = fresh_deps_add(pool_deps, 2, 0, 4096); assert(id);
    fresh_deps_poll_regions(pool_deps, &n); assert(n == 1);
    submit_epoch++; pool_deps_observe(1);
    assert(fresh_deps_owner(pool_deps, id) == 2 && fresh_deps_dirty(pool_deps, id));
    assert(pool_deps_epoch == submit_epoch && pool_deps_map && pool_deps_protection && pool_deps_serial);
    assert(n_presnap_q);
    fresh_deps_destroy(pool_deps);
    puts("resource observer: sleeping scans avoided, relevant events wake, overflow rebuilds and re-registration passed");
}

int main(void) {
    uint32_t v, epoch = 1;
    written[0] = 1;
    assert(vk_mem_snap_range_locked(0, 4096, epoch) == 1);
    assert(vk_mem_dependency_region_locked(0, epoch, &v) && v);
    assert(!vk_mem_dependency_sleep_locked(0));
    // An active region remains polled even when it did not change. A frozen region's pre-freeze snapshot cannot
    // establish sleep: its pending write must first pass through the actual OS query.
    uint32_t since = gen_now;
    written[0] = 1; transition(0, 1);
    assert(!vk_mem_dependency_sleep_locked(0) && snap_current(0, epoch));
    assert(!vk_mem_snap_range_locked(0, 4096, epoch) && reg_max[0] == since && written[0]);
    assert(vk_mem_snap_range_locked(0, 4096, ++epoch) == 1 && reg_max[0] > since);
    assert(vk_mem_dependency_sleep_locked(0));
    unsigned before = queries;
    for (unsigned i = 0; i < 100000; i++) {
        epoch++;
        assert(!vk_mem_snap_range_locked(0, 4096, epoch));
        assert(!vk_mem_unchanged_why_locked(0, 4096, gen_now, epoch));
    }
    assert(queries == before);
    // A full thaw/write/refreeze revokes cross-epoch sleep. This epoch's content observation remains stable until
    // the next epoch, preserving the renderer's existing once-per-submission visibility contract.
    transition(0, 0); written[0] = 1; transition(0, 1);
    assert(!vk_mem_dependency_sleep_locked(0));
    since = gen_now;
    assert(!vk_mem_snap_range_locked(0, 4096, epoch) && reg_max[0] == since && written[0]);
    assert(vk_mem_snap_range_locked(0, 4096, ++epoch) == 1 && reg_max[0] > since);
    assert(vk_mem_dependency_sleep_locked(0));
    // Unrelated backing changes preserve the range proof; relevant changes during a query cannot become tracked.
    transition(1, 0);
    assert(vk_mem_dependency_sleep_locked(0));
    transition(0, 0); race = 1;
    vk_mem_snap_range_locked(0, 4096, ++epoch);
    assert(!vk_mem_dependency_region_locked(0, epoch, &v));
    assert(!vk_mem_snap_range_locked(0, 4096, epoch));
    vk_mem_snap_range_locked(0, 4096, ++epoch);
    assert(vk_mem_dependency_region_locked(0, epoch, &v));
    race = 2;
    vk_mem_snap_range_locked(0, 4096, ++epoch);
    assert(!vk_mem_dependency_region_locked(0, epoch, &v));
    assert(!vk_mem_snap_range_locked(0, 4096, epoch));
    vk_mem_snap_range_locked(0, 4096, ++epoch);
    assert(vk_mem_dependency_region_locked(0, epoch, &v));
    untracked = 1; transition(0, 1);
    vk_mem_snap_range_locked(0, 4096, ++epoch);
    assert(!vk_mem_dependency_sleep_locked(0) && !vk_mem_dependency_region_locked(0, epoch, &v));
    untracked = 0;
    vk_mem_snap_range_locked(0, 4096, ++epoch);
    assert(vk_mem_dependency_sleep_locked(0));
    // Busy thaw state and unavailable/reset event tracking fail closed.
    mem_mapping_publish(&protection, 0, 1ull << 21, 1); frozen[0] = 2;
    assert(!vk_mem_dependency_sleep_locked(0));
    frozen[0] = 0; mem_mapping_publish(&protection, 0, 1ull << 21, 0);

    TexSlot slot = {.cgen = 7, .t = {.mem = (void *)(uintptr_t)(1ull << 21), .L.size = 4096}};
    assert(!slot_storage_readable(&slot) && slot_absence_valid(&slot, 7));
    before = read_calls;
    for (unsigned i = 0; i < 100000; i++) assert(slot_absence_valid(&slot, 7));
    assert(read_calls == before);
    transition(2, 0); assert(slot_absence_valid(&slot, 7));
    assert(!slot_absence_valid(&slot, 8));
    // A backing-only event revokes absence. Creation/alias generation and descriptor lifetime do so independently.
    transition(1, 0); assert(!slot_absence_valid(&slot, 7));
    slot.absent_mapping = slot.absent_protection = 0; alias = 1;
    assert(!slot_storage_readable(&slot) && !slot_absence_valid(&slot, 7));
    alias = 0; read_race = 1;
    assert(!slot_storage_readable(&slot) && !slot_absence_valid(&slot, 7));
    readable = 1;
    assert(slot_storage_readable(&slot) && !slot_absence_valid(&slot, 7));
    readable = 0;
    assert(!slot_storage_readable(&slot) && slot_absence_valid(&slot, 7));
    slot.finalized = 1; assert(!slot_absence_valid(&slot, 7)); slot.finalized = 0;
    protection.disabled = 1;
    assert(!slot_absence_valid(&slot, 7) && !vk_mem_dependency_sleep_locked(0));
    assert(n_snap_calls == queries && notifications);
    observer_integration();
    puts("resource consumers: frozen snapshot reuse, same-epoch races, backing changes and guarded absence passed");
}

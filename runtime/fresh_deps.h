// Reverse dependencies for submission-scoped resource freshness. The caller owns synchronization.
#ifndef FRESH_DEPS_H
#define FRESH_DEPS_H
#include <stddef.h>
#include <stdint.h>

typedef struct FreshDeps FreshDeps;
FreshDeps *fresh_deps_create(uint32_t regions, unsigned shift);
void fresh_deps_destroy(FreshDeps *d);
// Zero means allocation failure. Unsupported ranges get a permanently dirty node without region links.
uint32_t fresh_deps_add(FreshDeps *d, uintptr_t owner, uint64_t addr, uint64_t size);
void fresh_deps_remove(FreshDeps *d, uint32_t node);
void fresh_deps_mark(FreshDeps *d, uint32_t node);
void fresh_deps_mark_all(FreshDeps *d);
// Only the authoritative freshness check may certify a node. Unsupported ranges cannot be certified.
void fresh_deps_clean(FreshDeps *d, uint32_t node);
int fresh_deps_dirty(const FreshDeps *d, uint32_t node);
uintptr_t fresh_deps_owner(const FreshDeps *d, uint32_t node);
uint32_t fresh_deps_first(const FreshDeps *d);
uint32_t fresh_deps_next(const FreshDeps *d, uint32_t node);
// Sorted unique region IDs; valid until membership changes. Asking for this list never clears dirty nodes.
const uint32_t *fresh_deps_regions(FreshDeps *d, uint32_t *count);
// New referenced regions enter this polling set. Only the consumer's established protected-memory proof may
// put a region to sleep. Producer notifications wake it again; these operations never certify any node.
const uint32_t *fresh_deps_poll_regions(FreshDeps *d, uint32_t *count);
void fresh_deps_wake_region(FreshDeps *d, uint32_t region);
void fresh_deps_wake_all(FreshDeps *d);
void fresh_deps_sleep_region(FreshDeps *d, uint32_t region);
// Visit sleeping memberships without polling producer state. The callback may wake regions, but may not add or
// remove subscribers. This establishes their logical submission observation after complete event history is read.
void fresh_deps_visit_sleeping(FreshDeps *d, void (*visit)(uint32_t region, void *user), void *user);
// An untracked region notifies its consumers on every observation, even if its version did not change.
uint32_t fresh_deps_observe(FreshDeps *d, uint32_t region, uint32_t version, int tracked);
// Mapping versions have an independent domain from content generations. Zero is an unavailable/busy observation
// and notifies every dependent node each time; a stable version only notifies nodes linked to its changed region.
uint32_t fresh_deps_observe_mapping(FreshDeps *d, uint32_t region, uint64_t version);
#endif

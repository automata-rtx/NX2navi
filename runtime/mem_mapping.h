// Regional mapping epochs. Writers are serialized by the guest memory lock; readers take no additional lock.
// A publication precedes mutation with busy stamps and follows completed backing/metadata changes with new stamps.
#pragma once
#include <stdint.h>
#include "resource_events.h"

typedef struct {
    uint64_t *region;
    uint64_t serial;
    uint32_t count;
    unsigned shift;
    int disabled;
    ResourceEvents *events; // optional notification stream, owned by the same writer lock as these versions
} MemMapping;

static inline uint64_t mem_mapping_clock(const MemMapping *m) {
    return __atomic_load_n(&m->disabled, __ATOMIC_SEQ_CST) ? 0 :
           __atomic_load_n(&m->serial, __ATOMIC_SEQ_CST);
}

static inline void mem_mapping_publish(MemMapping *m, uint64_t addr, uint64_t size, int busy) {
    if (!size || __atomic_load_n(&m->disabled, __ATOMIC_SEQ_CST)) return;
    uint64_t limit = (uint64_t)m->count << m->shift;
    uint64_t now = __atomic_load_n(&m->serial, __ATOMIC_SEQ_CST);
    if (addr >= limit || size > limit - addr || now > UINT64_MAX - 2) {
        __atomic_store_n(&m->disabled, 1, __ATOMIC_SEQ_CST);
        return;
    }
    uint64_t next = now + 2;
    uint32_t first = (uint32_t)(addr >> m->shift), last = (uint32_t)((addr + size - 1) >> m->shift);
    for (uint32_t r = first; r <= last; r++)
        __atomic_store_n(&m->region[r], next | (busy != 0), __ATOMIC_SEQ_CST);
    __atomic_store_n(&m->serial, next, __ATOMIC_SEQ_CST);
    if (m->events) resource_events_publish(m->events, first, last);
}

static inline uint64_t mem_mapping_version(const MemMapping *m, uint32_t region) {
    if (region >= m->count || !mem_mapping_clock(m)) return 0;
    uint64_t stamp = __atomic_load_n(&m->region[region], __ATOMIC_SEQ_CST);
    if (stamp & 1) return 0;
    return mem_mapping_clock(m) ? (stamp ? stamp : 2) : 0;
}

static inline int mem_mapping_unchanged(const MemMapping *m, uint64_t addr, uint64_t size, uint64_t observed) {
    uint64_t limit = (uint64_t)m->count << m->shift, before = mem_mapping_clock(m);
    if (!observed || !before || observed > before || !size || addr >= limit || size > limit - addr) return 0;
    uint32_t first = (uint32_t)(addr >> m->shift), last = (uint32_t)((addr + size - 1) >> m->shift);
    for (uint32_t r = first; r <= last; r++) {
        uint64_t stamp = mem_mapping_version(m, r);
        if (!stamp || stamp > observed) return 0;
    }
    // A writer publishing during the scan may have changed a region already read. Reject this observation rather
    // than pretending the individually sampled region values constituted one consistent snapshot.
    return mem_mapping_clock(m) == before;
}

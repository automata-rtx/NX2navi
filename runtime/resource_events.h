// Bounded regional change notifications. Each log has one externally serialized writer domain;
// independent readers keep their own cursors and take no producer lock. Missing history means
// rebuild the consumer's observations, never assume that a region stayed unchanged.
#ifndef RESOURCE_EVENTS_H
#define RESOURCE_EVENTS_H
#include <stdint.h>

#define RESOURCE_EVENTS_CAPACITY 4096u
typedef struct {
    uint64_t sequence;
    uint32_t first, last;
} ResourceEvent;
typedef struct {
    ResourceEvent event[RESOURCE_EVENTS_CAPACITY];
    uint64_t head;
    int disabled;
} ResourceEvents;
typedef void (*ResourceEventVisit)(uint32_t first, uint32_t last, void *user);

static inline uint64_t resource_events_head(const ResourceEvents *log) {
    return __atomic_load_n(&log->head, __ATOMIC_ACQUIRE);
}

// The owning writer lock stays held through this entire operation, including publication.
// No allocation, callback or additional lock is allowed on this producer path.
static inline void resource_events_publish(ResourceEvents *log, uint32_t first, uint32_t last) {
    if (__atomic_load_n(&log->disabled, __ATOMIC_ACQUIRE)) return;
    uint64_t old = __atomic_load_n(&log->head, __ATOMIC_RELAXED);
    if (first > last || old == UINT64_MAX) {
        __atomic_store_n(&log->disabled, 1, __ATOMIC_RELEASE);
        return;
    }
    uint64_t next = old + 1;
    ResourceEvent *e = &log->event[(next - 1) % RESOURCE_EVENTS_CAPACITY];
    // The busy stamp must precede payload changes even for a reader of the old wrapped record.
    __atomic_store_n(&e->sequence, 0, __ATOMIC_SEQ_CST);
    __atomic_store_n(&e->first, first, __ATOMIC_SEQ_CST);
    __atomic_store_n(&e->last, last, __ATOMIC_SEQ_CST);
    __atomic_store_n(&e->sequence, next, __ATOMIC_SEQ_CST);
    __atomic_store_n(&log->head, next, __ATOMIC_RELEASE);
}

// Visits only history complete at entry. A concurrent later event remains for the next call.
// Partial callbacks before failure are harmless notifications; the caller must then rebuild.
// Advance the cursor on failure too: a full conservative rebuild replaces the lost history.
static inline int resource_events_read(const ResourceEvents *log, uint64_t *cursor,
                                      ResourceEventVisit visit, void *user) {
    uint64_t end = resource_events_head(log), begin = *cursor;
    *cursor = end;
    if (__atomic_load_n(&log->disabled, __ATOMIC_ACQUIRE) || begin > end ||
        end - begin > RESOURCE_EVENTS_CAPACITY) return 0;
    for (uint64_t i = begin; i != end;) {
        uint64_t wanted = ++i;
        const ResourceEvent *e = &log->event[(wanted - 1) % RESOURCE_EVENTS_CAPACITY];
        if (__atomic_load_n(&e->sequence, __ATOMIC_SEQ_CST) != wanted) return 0;
        uint32_t first = __atomic_load_n(&e->first, __ATOMIC_SEQ_CST);
        uint32_t last = __atomic_load_n(&e->last, __ATOMIC_SEQ_CST);
        if (__atomic_load_n(&e->sequence, __ATOMIC_SEQ_CST) != wanted || first > last) return 0;
        visit(first, last, user);
    }
    return !__atomic_load_n(&log->disabled, __ATOMIC_ACQUIRE);
}
#endif

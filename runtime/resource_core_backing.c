#include "resource_core_backing.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#ifndef RC_BACKING_SINGLE_THREADED
#include "platform/platform.h"
#endif

typedef struct {
    uint64_t id, size;
    uintptr_t provider;
    uint32_t mappings, captures;
    int imported, finalized;
} Backing;

typedef struct {
    uint64_t id, address, size, storage_namespace, storage_offset;
    uint32_t head;
    int storage_writer;
} Map;

typedef struct {
    uint64_t address, end, offset, mapping;
    uint32_t backing, next;
} Segment;

struct RcBackingStore {
#ifndef RC_BACKING_SINGLE_THREADED
    PlatMutex lock;
#endif
    RcBackingConfig limit;
    Backing *backings;
    Map *maps;
    Segment *segments;
    uint32_t free_segment, free_segments;
    uint32_t live_backings, live_maps, live_captures, captured_segments;
};

struct RcBackingCapture {
    RcBackingStore *store;
    RcBackingMapId map;
    uint64_t address, end;
    uint32_t count;
    RcBackingSegment segments[]; // Followed by count private backing slot indices.
};

static void lock_store(RcBackingStore *store) {
#ifndef RC_BACKING_SINGLE_THREADED
    plat_mutex_lock(&store->lock);
#else
    (void)store;
#endif
}

static void unlock_store(RcBackingStore *store) {
#ifndef RC_BACKING_SINGLE_THREADED
    plat_mutex_unlock(&store->lock);
#else
    (void)store;
#endif
}

// Incarnations remain distinct across store resets and across reusable slab slots. Backing 1 belongs to the
// raw guest-address compatibility domain while registry consumers migrate.
static _Atomic uint64_t last_identity = 1;
static uint64_t allocate_ids(uint64_t count) {
    uint64_t last = atomic_load_explicit(&last_identity, memory_order_relaxed);
    do {
        if (count > UINT64_MAX - last) return 0;
    } while (!atomic_compare_exchange_weak_explicit(&last_identity, &last, last + count,
                                                     memory_order_relaxed, memory_order_relaxed));
    return last + 1;
}

static int valid_span(uint64_t start, uint64_t size) {
    return size && start <= UINT64_MAX - size;
}

static int contains(uint64_t base, uint64_t size, uint64_t address, uint64_t length) {
    return address >= base && length && length <= size && address - base <= size - length;
}

static uint64_t max_u64(uint64_t a, uint64_t b) { return a > b ? a : b; }
static uint64_t min_u64(uint64_t a, uint64_t b) { return a < b ? a : b; }

static void *alloc_slots(uint32_t limit, size_t entry_size) {
    if (!limit || limit == UINT32_MAX || limit >= SIZE_MAX / entry_size) return NULL;
    return calloc((size_t)limit + 1, entry_size);
}

RcBackingStore *rc_backing_store_create(RcBackingConfig limit) {
    if (!limit.captures || !limit.captured_segments) return NULL;
    RcBackingStore *store = calloc(1, sizeof *store);
    if (!store) return NULL;
    store->limit = limit;
    store->backings = alloc_slots(limit.backings, sizeof *store->backings);
    store->maps = alloc_slots(limit.maps, sizeof *store->maps);
    store->segments = alloc_slots(limit.current_segments, sizeof *store->segments);
    if (!store->backings || !store->maps || !store->segments) {
        rc_backing_store_destroy(store);
        return NULL;
    }
#ifndef RC_BACKING_SINGLE_THREADED
    plat_mutex_init(&store->lock);
#endif
    for (uint32_t i = 1; i < limit.current_segments; i++) store->segments[i].next = i + 1;
    store->free_segment = 1;
    store->free_segments = limit.current_segments;
    return store;
}

RcBackingResult rc_backing_store_destroy(RcBackingStore *store) {
    if (!store) return RC_BACKING_INVALID;
    if (store->live_maps || store->live_backings || store->live_captures) return RC_BACKING_BUSY;
    free(store->backings);
    free(store->maps);
    free(store->segments);
    free(store);
    return RC_BACKING_OK;
}

RcBackingConfig rc_backing_store_usage(RcBackingStore *store) {
    if (!store) return (RcBackingConfig){0};
    lock_store(store);
    RcBackingConfig usage = {store->live_backings, store->live_maps, store->limit.current_segments - store->free_segments,
                            store->live_captures, store->captured_segments};
    unlock_store(store);
    return usage;
}

static Backing *backing_of(RcBackingStore *store, RcBackingId id) {
    if (!id.id || !id.slot || id.slot > store->limit.backings) return NULL;
    Backing *backing = &store->backings[id.slot];
    return backing->id == id.id ? backing : NULL;
}

static Map *map_of(RcBackingStore *store, RcBackingMapId id) {
    if (!id.id || !id.slot || id.slot > store->limit.maps) return NULL;
    Map *map = &store->maps[id.slot];
    return map->id == id.id ? map : NULL;
}

// Callers reserve capacity before taking a slot. Zero is the sentinel for a missing interval or backing.
static uint32_t free_backing_slot(const RcBackingStore *store) {
    uint32_t i = 1;
    while (store->backings[i].id) i++;
    return i;
}

static uint32_t free_map_slot(const RcBackingStore *store) {
    uint32_t i = 1;
    while (store->maps[i].id) i++;
    return i;
}

static void clear_backing(RcBackingStore *store, Backing *backing) {
    *backing = (Backing){0};
    --store->live_backings;
}

static void collect_imports(RcBackingStore *store) {
    for (uint32_t i = 1; i <= store->limit.backings; i++) {
        Backing *backing = &store->backings[i];
        if (backing->imported && !backing->mappings && !backing->captures) clear_backing(store, backing);
    }
}

RcBackingResult rc_backing_create(RcBackingStore *store, uint64_t size, uintptr_t provider, RcBackingId *out) {
    if (out) *out = (RcBackingId){0};
    if (!store || !size || !out) return RC_BACKING_INVALID;
    lock_store(store);
    RcBackingResult result = RC_BACKING_CAPACITY;
    if (store->live_backings < store->limit.backings) {
        uint64_t id = allocate_ids(1);
        if (id) {
            uint32_t slot = free_backing_slot(store);
            store->backings[slot] = (Backing){.id = id, .size = size, .provider = provider};
            ++store->live_backings;
            *out = (RcBackingId){id, slot};
            result = RC_BACKING_OK;
        } else result = RC_BACKING_EXHAUSTED;
    }
    unlock_store(store);
    return result;
}

static uint32_t *insert_segment(RcBackingStore *store, uint32_t *link, Segment segment) {
    uint32_t slot = store->free_segment;
    store->free_segment = store->segments[slot].next;
    --store->free_segments;
    ++store->backings[segment.backing].mappings;
    segment.next = *link;
    store->segments[slot] = segment;
    *link = slot;
    return &store->segments[slot].next;
}

static void erase_segment(RcBackingStore *store, uint32_t *link) {
    uint32_t slot = *link;
    Segment *segment = &store->segments[slot];
    *link = segment->next;
    --store->backings[segment->backing].mappings;
    segment->next = store->free_segment;
    store->free_segment = slot;
    ++store->free_segments;
}

RcBackingResult rc_backing_finalize(RcBackingStore *store, RcBackingId id) {
    if (!store || !id.id) return RC_BACKING_INVALID;
    lock_store(store);
    Backing *backing = backing_of(store, id);
    RcBackingResult result = RC_BACKING_STALE;
    if (backing && !backing->imported && !backing->finalized) {
        backing->finalized = 1;
        for (uint32_t i = 1; i <= store->limit.maps && backing->mappings; i++) {
            for (uint32_t *at = &store->maps[i].head; *at;) {
                Segment *segment = &store->segments[*at];
                if (segment->backing == id.slot) erase_segment(store, at);
                else at = &segment->next;
            }
        }
        result = RC_BACKING_OK;
    }
    unlock_store(store);
    return result;
}

RcBackingResult rc_backing_reclaim(RcBackingStore *store, RcBackingId id, uintptr_t *provider) {
    if (provider) *provider = 0;
    if (!store || !id.id || !provider) return RC_BACKING_INVALID;
    lock_store(store);
    Backing *backing = backing_of(store, id);
    RcBackingResult result = RC_BACKING_STALE;
    if (backing && !backing->imported) {
        result = RC_BACKING_BUSY;
        if (backing->finalized && !backing->captures) {
            *provider = backing->provider;
            clear_backing(store, backing);
            result = RC_BACKING_OK;
        }
    }
    unlock_store(store);
    return result;
}

int rc_backing_explicit_retained(RcBackingStore *store, RcBackingId *identity) {
    if (!store || !identity || !identity->id) return 0;
    lock_store(store);
    if (!identity->slot) {
        for (uint32_t slot = 1; slot <= store->limit.backings; ++slot) {
            const Backing *candidate = &store->backings[slot];
            if (candidate->id != identity->id || candidate->imported) continue;
            identity->slot = slot;
            break;
        }
    }
    const Backing *backing = backing_of(store, *identity);
    const int retained = backing && !backing->imported;
    unlock_store(store);
    return retained;
}

static RcBackingResult open_map(RcBackingStore *store, Map map, RcBackingMapId *out) {
    lock_store(store);
    RcBackingResult result = RC_BACKING_CAPACITY;
    if (store->live_maps < store->limit.maps) {
        map.id = allocate_ids(1);
        result = RC_BACKING_EXHAUSTED;
        if (map.id) {
            uint32_t slot = free_map_slot(store);
            store->maps[slot] = map;
            ++store->live_maps;
            *out = (RcBackingMapId){map.id, slot};
            result = RC_BACKING_OK;
        }
    }
    unlock_store(store);
    return result;
}

RcBackingResult rc_backing_map_create(RcBackingStore *store, uint64_t address, uint64_t size, RcBackingMapId *out) {
    if (out) *out = (RcBackingMapId){0};
    if (!store || !valid_span(address, size) || !out) return RC_BACKING_INVALID;
    return open_map(store, (Map){.address = address, .size = size}, out);
}

typedef struct {
    uint32_t *link, removed;
    Segment left, right;
} Cut;

static Cut plan_cut(RcBackingStore *store, Map *map, uint64_t address, uint64_t end) {
    Cut cut = {.link = &map->head};
    while (*cut.link && store->segments[*cut.link].end <= address)
        cut.link = &store->segments[*cut.link].next;
    for (uint32_t at = *cut.link; at; at = store->segments[at].next) {
        Segment segment = store->segments[at];
        if (segment.address >= end) break;
        ++cut.removed;
        if (segment.address < address) {
            cut.left = segment;
            cut.left.end = address;
        }
        if (segment.end > end) {
            cut.right = segment;
            cut.right.offset += end - segment.address;
            cut.right.address = end;
        }
    }
    return cut;
}

static uint32_t cut_survivors(Cut cut) { return !!cut.left.backing + !!cut.right.backing; }

static void apply_cut(RcBackingStore *store, Cut cut, Segment replacement) {
    uint32_t *at = cut.link;
    for (uint32_t i = 0; i < cut.removed; i++) erase_segment(store, at);
    if (cut.left.backing) at = insert_segment(store, at, cut.left);
    if (replacement.backing) at = insert_segment(store, at, replacement);
    if (cut.right.backing) insert_segment(store, at, cut.right);
}

RcBackingResult rc_backing_map_replace(RcBackingStore *store, RcBackingMapId id, uint64_t address, uint64_t size,
                                      RcBackingId target, uint64_t offset) {
    if (!store || !id.id || !valid_span(address, size) || (!target.id && (target.slot || offset)))
        return RC_BACKING_INVALID;
    lock_store(store);
    Map *map = map_of(store, id);
    Backing *backing = backing_of(store, target);
    RcBackingResult result;
    if (!map || (target.id && (!backing || backing->finalized))) result = RC_BACKING_STALE;
    else if (map->storage_namespace || !contains(map->address, map->size, address, size) ||
             (backing && (backing->imported || !contains(0, backing->size, offset, size))))
        result = RC_BACKING_INVALID;
    else {
        Cut cut = plan_cut(store, map, address, address + size);
        if (cut_survivors(cut) + !!backing > cut.removed + (uint64_t)store->free_segments)
            result = RC_BACKING_CAPACITY;
        else {
            uint64_t mapping = backing ? allocate_ids(1) : 0;
            if (backing && !mapping) result = RC_BACKING_EXHAUSTED;
            else {
                apply_cut(store, cut, (Segment){address, address + size, offset, mapping, target.slot, 0});
                result = RC_BACKING_OK;
            }
        }
    }
    unlock_store(store);
    return result;
}

// In the scratch list only, next is the request ordinal (plus one) for a new mapping.
static void replace_segments(Segment *segments, size_t *length, RcBackingReplace change, uint32_t request) {
    const uint64_t end = change.address + change.size;
    size_t first = 0;
    while (first < *length && segments[first].end <= change.address) ++first;
    size_t after = first;
    while (after < *length && segments[after].address < end) ++after;
    Segment inserts[3];
    size_t count = 0;
    if (first < after && segments[first].address < change.address) {
        inserts[count] = segments[first];
        inserts[count++].end = change.address;
    }
    if (change.backing.id)
        inserts[count++] = (Segment){change.address, end, change.offset, 0, change.backing.slot, request + 1};
    if (first < after && segments[after - 1].end > end) {
        inserts[count] = segments[after - 1];
        inserts[count].offset += end - inserts[count].address;
        inserts[count++].address = end;
    }
    if (first + count != after)
        memmove(segments + first + count, segments + after, (*length - after) * sizeof *segments);
    memcpy(segments + first, inserts, count * sizeof *segments);
    *length = *length - (after - first) + count;
}

RcBackingResult rc_backing_map_replace_many(RcBackingStore *store, RcBackingMapId id,
                                           const RcBackingReplace *changes, uint32_t count) {
    if (!store || !id.id || (count && !changes)) return RC_BACKING_INVALID;
    if (count == 1) {
        const RcBackingReplace *change = changes;
        return rc_backing_map_replace(store, id, change->address, change->size, change->backing, change->offset);
    }
    lock_store(store);
    Map *map = map_of(store, id);
    RcBackingResult result = !map ? RC_BACKING_STALE : map->storage_namespace ? RC_BACKING_INVALID : RC_BACKING_OK;
    for (uint32_t i = 0; result == RC_BACKING_OK && i < count; ++i) {
        RcBackingReplace c = changes[i];
        Backing *target = backing_of(store, c.backing);
        if (c.backing.id && (!target || target->finalized)) result = RC_BACKING_STALE;
        else if (!contains(map->address, map->size, c.address, c.size) ||
                 (target && (target->imported || !contains(0, target->size, c.offset, c.size))) ||
                 (!target && (c.backing.slot || c.offset))) result = RC_BACKING_INVALID;
    }
    if (result != RC_BACKING_OK || !count) {
        unlock_store(store);
        return result;
    }
    size_t original = 0;
    for (uint32_t at = map->head; at; at = store->segments[at].next) ++original;
    Segment *segments = NULL;
    if ((uint64_t)original + (uint64_t)count * 2 <= SIZE_MAX / sizeof *segments)
        segments = malloc((original + (size_t)count * 2) * sizeof *segments);
    if (!segments) {
        unlock_store(store);
        return RC_BACKING_CAPACITY;
    }
    size_t length = 0;
    for (uint32_t at = map->head; at; at = store->segments[at].next) {
        segments[length] = store->segments[at];
        segments[length++].next = 0;
    }
    uint32_t mappings = 0;
    for (uint32_t i = 0; i < count; ++i) {
        replace_segments(segments, &length, changes[i], mappings);
        mappings += !!changes[i].backing.id;
    }
    if (length > original + store->free_segments) result = RC_BACKING_CAPACITY;
    else {
        // Requests sharing surviving pieces get the same fresh identity. One allocation also checks wrap once.
        uint64_t first = mappings ? allocate_ids(mappings) : 0;
        if (mappings && !first) result = RC_BACKING_EXHAUSTED;
        else {
            while (map->head) erase_segment(store, &map->head);
            uint32_t *link = &map->head;
            for (size_t i = 0; i < length; ++i) {
                Segment segment = segments[i];
                if (segment.next) segment.mapping = first + segment.next - 1;
                link = insert_segment(store, link, segment);
            }
        }
    }
    free(segments);
    unlock_store(store);
    return result;
}

RcBackingResult rc_backing_map_close(RcBackingStore *store, RcBackingMapId id) {
    if (!store || !id.id) return RC_BACKING_INVALID;
    lock_store(store);
    Map *map = map_of(store, id);
    RcBackingResult result = RC_BACKING_STALE;
    if (map) {
        while (map->head) erase_segment(store, &map->head);
        if (map->storage_namespace) collect_imports(store);
        *map = (Map){0};
        --store->live_maps;
        result = RC_BACKING_OK;
    }
    unlock_store(store);
    return result;
}

typedef struct {
    uint64_t end, offset;
    uint32_t backing;
} ImportRun;

// Live import segments are the canonical storage index. Every overlapping live segment has the same
// backing and affine offset for those bytes. Retained snapshots never participate in this search.
static ImportRun import_run(RcBackingStore *store, uint64_t storage, uint64_t position, uint64_t end) {
    ImportRun run = {.end = end};
    for (uint32_t i = 1; i <= store->limit.maps; i++) {
        const Map *map = &store->maps[i];
        if (map->storage_namespace != storage) continue;
        for (uint32_t at = map->head; at; at = store->segments[at].next) {
            const Segment *segment = &store->segments[at];
            uint64_t start = map->storage_offset + (segment->address - map->address);
            if (start > position) {
                if (!run.backing) run.end = min_u64(run.end, start);
                break;
            }
            uint64_t bottom = map->storage_offset + (segment->end - map->address);
            if (bottom > position) {
                if (!run.backing || bottom > run.end) {
                    run = (ImportRun){min_u64(bottom, end), segment->offset + (position - start), segment->backing};
                }
                break;
            }
        }
    }
    return run;
}

static RcBackingResult import_capacity(RcBackingStore *store, uint64_t storage, uint64_t offset,
                                       uint64_t size, uint32_t *new_backings) {
    uint64_t end = offset + size;
    uint32_t count = 0;
    *new_backings = 0;
    for (uint64_t position = offset; position < end;) {
        ImportRun run = import_run(store, storage, position, end);
        if (count == store->free_segments ||
            (!run.backing && *new_backings == store->limit.backings - store->live_backings)) return RC_BACKING_CAPACITY;
        ++count;
        *new_backings += !run.backing;
        position = run.end;
    }
    return RC_BACKING_OK;
}

static void import_segments(RcBackingStore *store, uint32_t *link, uint64_t storage, uint64_t address,
                            uint64_t offset, uint64_t size, uint64_t mapping) {
    uint64_t end = offset + size, next_id = mapping;
    for (uint64_t position = offset; position < end;) {
        ImportRun run = import_run(store, storage, position, end);
        if (!run.backing) {
            run.backing = free_backing_slot(store);
            store->backings[run.backing] = (Backing){.id = ++next_id, .size = run.end - position, .imported = 1};
            ++store->live_backings;
        }
        Segment segment = {.address = address + (position - offset), .end = address + (run.end - offset),
                           .offset = run.offset, .mapping = mapping, .backing = run.backing};
        link = insert_segment(store, link, segment);
        position = run.end;
    }
}

static RcBackingResult import_locked(RcBackingStore *store, uint64_t storage, uint64_t address, uint64_t offset,
                                     uint64_t size, RcBackingMapId *out) {
    if (store->live_maps == store->limit.maps) return RC_BACKING_CAPACITY;
    uint32_t new_backings;
    RcBackingResult result = import_capacity(store, storage, offset, size, &new_backings);
    if (result != RC_BACKING_OK) return result;
    uint64_t id = allocate_ids((uint64_t)new_backings + 2);
    if (!id) return RC_BACKING_EXHAUSTED;
    uint32_t slot = free_map_slot(store);
    Map map = {.id = id, .address = address, .size = size, .storage_namespace = storage, .storage_offset = offset};
    import_segments(store, &map.head, storage, address, offset, size, id + 1);
    store->maps[slot] = map;
    ++store->live_maps;
    *out = (RcBackingMapId){id, slot};
    return RC_BACKING_OK;
}

RcBackingResult rc_backing_map_import(RcBackingStore *store, uint64_t storage, uint64_t address,
                                     uint64_t offset, uint64_t size, RcBackingMapId *out) {
    if (out) *out = (RcBackingMapId){0};
    if (!store || !storage || !out || !valid_span(address, size) || !valid_span(offset, size)) return RC_BACKING_INVALID;
    lock_store(store);
    RcBackingResult result = import_locked(store, storage, address, offset, size, out);
    unlock_store(store);
    return result;
}

RcBackingResult rc_backing_storage_map_create(RcBackingStore *store, uint64_t storage, uint64_t address,
                                              uint64_t offset, uint64_t size, RcBackingMapId *out) {
    if (out) *out = (RcBackingMapId){0};
    if (!store || !storage || !out || !valid_span(address, size) || !valid_span(offset, size)) return RC_BACKING_INVALID;
    Map map = {.address = address, .size = size, .storage_namespace = storage,
               .storage_offset = offset, .storage_writer = 1};
    return open_map(store, map, out);
}

RcBackingResult rc_backing_storage_map_publish(RcBackingStore *store, RcBackingMapId id, uint64_t address,
                                               uint64_t size) {
    if (!store || !id.id || !valid_span(address, size)) return RC_BACKING_INVALID;
    lock_store(store);
    Map *map = map_of(store, id);
    RcBackingResult result = RC_BACKING_STALE;
    if (map) {
        result = RC_BACKING_INVALID;
        if (map->storage_writer && contains(map->address, map->size, address, size)) {
            Cut cut = plan_cut(store, map, address, address + size);
            if (!cut.removed) {
                uint64_t offset = map->storage_offset + (address - map->address);
                uint32_t new_backings;
                result = import_capacity(store, map->storage_namespace, offset, size, &new_backings);
                if (result == RC_BACKING_OK) {
                    uint64_t mapping = allocate_ids((uint64_t)new_backings + 1);
                    if (mapping) import_segments(store, cut.link, map->storage_namespace, address, offset, size, mapping);
                    else result = RC_BACKING_EXHAUSTED;
                }
            }
        }
    }
    unlock_store(store);
    return result;
}

static Cut import_cut(RcBackingStore *store, Map *map, uint64_t storage, uint64_t lo, uint64_t hi) {
    if (map->storage_namespace != storage || lo >= map->storage_offset + map->size || hi <= map->storage_offset)
        return (Cut){0};
    uint64_t first = max_u64(lo, map->storage_offset) - map->storage_offset;
    uint64_t end = min_u64(hi - map->storage_offset, map->size);
    return plan_cut(store, map, map->address + first, map->address + end);
}

RcBackingResult rc_backing_import_invalidate(RcBackingStore *store, uint64_t storage, uint64_t offset, uint64_t size) {
    if (!store || !storage || !valid_span(offset, size)) return RC_BACKING_INVALID;
    lock_store(store);
    uint64_t end = offset + size;
    int64_t delta = 0;
    for (uint32_t i = 1; i <= store->limit.maps; i++) {
        Cut cut = import_cut(store, &store->maps[i], storage, offset, end);
        delta += (int64_t)cut_survivors(cut) - cut.removed;
    }
    RcBackingResult result = RC_BACKING_CAPACITY;
    if (delta <= store->free_segments) {
        // Cuts which free segments fund splits elsewhere in this same atomic mutation.
        for (int expands = 0; expands <= 1; expands++) {
            for (uint32_t i = 1; i <= store->limit.maps; i++) {
                Cut cut = import_cut(store, &store->maps[i], storage, offset, end);
                if (cut.removed && (cut_survivors(cut) > cut.removed) == expands)
                    apply_cut(store, cut, (Segment){0});
            }
        }
        collect_imports(store);
        result = RC_BACKING_OK;
    }
    unlock_store(store);
    return result;
}

// Emit current intersections and explicit holes in address order. The iterator only borrows live topology
// while locked. Cropping a mapping segment keeps the original mapping identity and adjusts the backing offset.
static RcBackingSegment segment_at(RcBackingStore *store, uint32_t *at, uint64_t address, uint64_t end,
                                   uint32_t *backing_slot) {
    while (*at && store->segments[*at].end <= address) *at = store->segments[*at].next;
    const Segment *next = *at ? &store->segments[*at] : NULL;
    *backing_slot = 0;
    if (!next || next->address > address) {
        uint64_t stop = next ? min_u64(end, next->address) : end;
        return (RcBackingSegment){.address = address, .range.size = stop - address};
    }
    *backing_slot = next->backing;
    const Backing *backing = &store->backings[next->backing];
    RcSourceRange range = {backing->id, next->offset + (address - next->address), min_u64(end, next->end) - address};
    return (RcBackingSegment){address, next->mapping, range, backing->provider};
}

RcBackingResult rc_backing_map_runs(RcBackingStore *store, RcBackingMapId id, uint64_t address,
                                   RcBackingRun *runs, uint32_t capacity, uint32_t *count) {
    if (count) *count = 0;
    if (!store || !id.id || !runs || !capacity || !count) return RC_BACKING_INVALID;
    lock_store(store);
    const Map *map = map_of(store, id);
    RcBackingResult result = RC_BACKING_STALE;
    if (map && !contains(map->address, map->size, address, 1)) result = RC_BACKING_INVALID;
    else if (map) {
        uint64_t end = map->address + map->size;
        uint32_t at = map->head;
        while (address < end && *count < capacity) {
            uint32_t slot;
            RcBackingSegment segment = segment_at(store, &at, address, end, &slot);
            RcBackingRun run = {address, segment.range, segment.provider};
            address += segment.range.size;
            for (uint32_t next = slot ? store->segments[at].next : 0; next; next = store->segments[next].next) {
                const Segment *piece = &store->segments[next];
                if (piece->address != address || piece->backing != slot ||
                    piece->offset != run.range.offset + run.range.size) break;
                run.range.size += piece->end - piece->address;
                address = piece->end;
                at = next;
            }
            runs[(*count)++] = run;
        }
        result = RC_BACKING_OK;
    }
    unlock_store(store);
    return result;
}

RcBackingResult rc_backing_map_intervals(RcBackingStore *store, RcBackingMapId id, uint64_t address, uint64_t size,
                                       RcInterval *out, uint32_t capacity, uint32_t *count) {
    if (count) *count = 0;
    if (!store || !id.id || !count || (!!out != !!capacity) || !valid_span(address, size)) return RC_BACKING_INVALID;
    lock_store(store);
    const Map *map = map_of(store, id);
    RcBackingResult result = !map ? RC_BACKING_STALE : RC_BACKING_INVALID;
    if (map && contains(map->address, map->size, address, size)) {
        uint64_t end = address + size;
        uint32_t at = map->head, slot;
        while (address < end && (!out || *count < capacity)) {
            if (*count == UINT32_MAX) { result = RC_BACKING_CAPACITY; goto finish; }
            RcBackingSegment piece = segment_at(store, &at, address, end, &slot);
            if (out) out[*count] = (RcInterval){piece.address, piece.mapping, piece.range};
            (*count)++;
            address += piece.range.size;
        }
        result = RC_BACKING_OK;
    }
finish:
    unlock_store(store);
    return result;
}

static int interval_current_at(RcBackingStore *store, uint32_t *at, RcInterval old, uint64_t address, uint64_t until) {
    RcSourceRange expected = old.range;
    if (expected.backing) expected.offset += address - old.address;
    expected.size = until - address;
    uint32_t slot;
    RcBackingSegment live = segment_at(store, at, address, until, &slot);
    return live.mapping == old.mapping && rc_range_equal(live.range, expected);
}

int rc_backing_map_intervals_current(RcBackingStore *store, RcBackingMapId id, const RcInterval *intervals,
                                    uint32_t count, uint64_t address, uint64_t size) {
    if (!store || !id.id || !intervals || !count || !valid_span(address, size)) return 0;
    lock_store(store);
    const Map *map = map_of(store, id);
    int matches = map && contains(map->address, map->size, address, size);
    uint64_t end = address + size;
    uint32_t at = map ? map->head : 0;
    for (uint32_t i = 0; matches && address < end && i < count; i++) {
        RcInterval old = intervals[i];
        if (!valid_span(old.address, old.range.size) || address < old.address) { matches = 0; break; }
        uint64_t old_end = old.address + old.range.size;
        if (address >= old_end) continue;
        uint64_t until = min_u64(old_end, end);
        matches = interval_current_at(store, &at, old, address, until);
        address = until;
    }
    unlock_store(store);
    return matches && address == end;
}

static RcBackingResult capture_locked(RcBackingStore *store, RcBackingMapId id, uint64_t address, uint64_t size,
                                      RcBackingCaptureMode mode, RcBackingCapture **out) {
    Map *map = map_of(store, id);
    if (!map) return RC_BACKING_STALE;
    if (!contains(map->address, map->size, address, size)) return RC_BACKING_INVALID;
    if (store->live_captures == store->limit.captures) return RC_BACKING_CAPACITY;
    uint64_t end = address + size;
    uint32_t count = 0, at = map->head, backing_slot;
    uint32_t available = store->limit.captured_segments - store->captured_segments;
    for (uint64_t position = address; position < end;) {
        RcBackingSegment segment = segment_at(store, &at, position, end, &backing_slot);
        if (!backing_slot && mode == RC_BACKING_REQUIRE_MAPPED) return RC_BACKING_ABSENT;
        if (count == available) return RC_BACKING_CAPACITY;
        ++count;
        position += segment.range.size;
    }
    size_t entry_size = sizeof(RcBackingSegment) + sizeof(uint32_t);
    if (count > (SIZE_MAX - sizeof(RcBackingCapture)) / entry_size) return RC_BACKING_CAPACITY;
    RcBackingCapture *capture = malloc(sizeof *capture + (size_t)count * entry_size);
    if (!capture) return RC_BACKING_CAPACITY;
    *capture = (RcBackingCapture){store, id, address, end, count};
    uint32_t *indices = (uint32_t *)(capture->segments + count);
    uint64_t position = address;
    at = map->head;
    for (uint32_t i = 0; i < count; i++) {
        RcBackingSegment segment = segment_at(store, &at, position, end, &indices[i]);
        capture->segments[i] = segment;
        if (indices[i]) ++store->backings[indices[i]].captures;
        position += segment.range.size;
    }
    ++store->live_captures;
    store->captured_segments += count;
    *out = capture;
    return RC_BACKING_OK;
}

RcBackingResult rc_backing_capture(RcBackingStore *store, RcBackingMapId id, uint64_t address, uint64_t size,
                                  RcBackingCaptureMode mode, RcBackingCapture **out) {
    if (out) *out = NULL;
    if (!store || !id.id || !out || !valid_span(address, size) ||
        (mode != RC_BACKING_REQUIRE_MAPPED && mode != RC_BACKING_ALLOW_HOLES)) return RC_BACKING_INVALID;
    lock_store(store);
    RcBackingResult result = capture_locked(store, id, address, size, mode, out);
    unlock_store(store);
    return result;
}

const RcBackingSegment *rc_backing_capture_segments(const RcBackingCapture *capture, uint32_t *count) {
    if (count) *count = capture ? capture->count : 0;
    return capture ? capture->segments : NULL;
}

int rc_backing_capture_range_current(RcBackingStore *store, const RcBackingCapture *capture,
                                     uint64_t address, uint64_t size) {
    if (!store || !capture || capture->store != store ||
        !contains(capture->address, capture->end - capture->address, address, size)) return 0;
    uint64_t end = address + size;
    lock_store(store);
    const Map *map = map_of(store, capture->map);
    int matches = !!map;
    uint32_t at = map ? map->head : 0;
    for (uint32_t i = 0; matches && address < end && i < capture->count; i++) {
        const RcBackingSegment *old = &capture->segments[i];
        uint64_t old_end = old->address + old->range.size;
        if (address >= old_end) continue;
        uint64_t until = min_u64(old_end, end);
        matches = interval_current_at(store, &at, (RcInterval){old->address, old->mapping, old->range}, address, until);
        address = until;
    }
    unlock_store(store);
    return matches && address == end;
}

int rc_backing_capture_current(RcBackingStore *store, const RcBackingCapture *capture) {
    return capture && rc_backing_capture_range_current(store, capture, capture->address, capture->end - capture->address);
}

void rc_backing_capture_release(RcBackingCapture **owned) {
    if (!owned || !*owned) return;
    RcBackingCapture *capture = *owned;
    *owned = NULL;
    RcBackingStore *store = capture->store;
    const uint32_t *indices = (const uint32_t *)(capture->segments + capture->count);
    lock_store(store);
    for (uint32_t i = 0; i < capture->count; i++) {
        if (!indices[i]) continue;
        Backing *backing = &store->backings[indices[i]];
        if (!--backing->captures && !backing->mappings && backing->imported) clear_backing(store, backing);
    }
    --store->live_captures;
    store->captured_segments -= capture->count;
    unlock_store(store);
    free(capture);
}

#include "fresh_deps.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { REGIONS = 73, NODES = 257, SHIFT = 12 };
typedef struct { uint32_t id, first, last; int live, dirty, unsupported; } Expected;
static Expected model[NODES];
static uint32_t version[REGIONS];
static int seen[REGIONS], tracked[REGIONS];
static uint32_t seed = 0xa1b2c3d4;
static uint32_t random_word(void) { seed = seed * 1664525u + 1013904223u; return seed; }

static void verify(FreshDeps *d) {
    unsigned expected_regions[REGIONS] = {0}, queue_seen[NODES] = {0};
    for (unsigned i = 0; i < NODES; i++) if (model[i].live) {
        assert(fresh_deps_owner(d, model[i].id) == i + 1);
        assert(fresh_deps_dirty(d, model[i].id) == model[i].dirty);
        if (!model[i].unsupported)
            for (unsigned r = model[i].first; r <= model[i].last; r++) expected_regions[r]++;
    }
    unsigned queued = 0;
    for (uint32_t id = fresh_deps_first(d); id; id = fresh_deps_next(d, id)) {
        uintptr_t owner = fresh_deps_owner(d, id);
        assert(owner && owner <= NODES);
        unsigned i = (unsigned)owner - 1;
        assert(model[i].live && model[i].dirty && model[i].id == id && !queue_seen[i]++);
        assert(++queued <= NODES);
    }
    for (unsigned i = 0; i < NODES; i++) assert(queue_seen[i] == (unsigned)(model[i].live && model[i].dirty));
    uint32_t n;
    const uint32_t *regions = fresh_deps_regions(d, &n);
    unsigned count = 0;
    for (unsigned r = 0; r < REGIONS; r++) if (expected_regions[r]) {
        assert(count < n && regions[count++] == r);
    }
    assert(count == n);
}

static void forget_empty_regions(void) {
    for (unsigned r = 0; r < REGIONS; r++) {
        int used = 0;
        for (unsigned i = 0; i < NODES; i++)
            if (model[i].live && !model[i].unsupported && model[i].first <= r && r <= model[i].last) used = 1;
        if (!used) seen[r] = 0;
    }
}

int main(void) {
    FreshDeps *d = fresh_deps_create(REGIONS, SHIFT);
    assert(d);
    // New observers, duplicate notifications, cross-region ranges, version wrap, tracking loss and recovery.
    uint32_t x = fresh_deps_add(d, 1, (1u << SHIFT) - 1, 2);
    assert(x && fresh_deps_dirty(d, x));
    fresh_deps_observe(d, 0, UINT32_MAX, 1);
    fresh_deps_observe(d, 1, 9, 1);
    fresh_deps_clean(d, x);
    assert(!fresh_deps_observe(d, 1, 9, 1) && !fresh_deps_dirty(d, x));
    assert(fresh_deps_observe(d, 0, 0, 1) && fresh_deps_dirty(d, x));
    fresh_deps_clean(d, x);
    assert(fresh_deps_observe(d, 0, 0, 0) && fresh_deps_dirty(d, x));
    fresh_deps_clean(d, x);
    assert(fresh_deps_observe(d, 0, 0, 0) && fresh_deps_dirty(d, x));
    fresh_deps_remove(d, x);
    assert(!fresh_deps_first(d));
    fresh_deps_destroy(d);
    d = fresh_deps_create(REGIONS, SHIFT);
    assert(d);
    for (unsigned step = 0; step < 100000; step++) {
        unsigned op = random_word() % 7, i = random_word() % NODES;
        Expected *e = &model[i];
        if (op == 0 && !e->live) {
            uint64_t limit = (uint64_t)REGIONS << SHIFT;
            uint64_t addr = random_word() % limit, size = 1 + random_word() % (9u << SHIFT);
            if ((random_word() & 15) == 0) addr = UINT64_MAX - 3, size = 100;
            if ((random_word() & 31) == 0) size = 0;
            int unsupported = !size || addr >= limit || size > limit - addr;
            *e = (Expected){.id = fresh_deps_add(d, i + 1, addr, size), .live = 1, .dirty = 1,
                            .unsupported = unsupported, .first = (uint32_t)(addr >> SHIFT),
                            .last = (uint32_t)((addr + size - 1) >> SHIFT)};
            assert(e->id);
        } else if (op == 1 && e->live) {
            fresh_deps_remove(d, e->id);
            memset(e, 0, sizeof *e);
            forget_empty_regions();
        } else if (op == 2 && e->live) {
            fresh_deps_clean(d, e->id);
            if (!e->unsupported) e->dirty = 0;
        } else if (op == 3 && e->live) {
            fresh_deps_mark(d, e->id); e->dirty = 1;
        } else if (op == 4) {
            unsigned r = random_word() % REGIONS;
            uint32_t v = (random_word() & 3) ? version[r] : random_word();
            int t = (random_word() & 7) != 0;
            if (!seen[r] || version[r] != v || !t || !tracked[r])
                for (unsigned j = 0; j < NODES; j++)
                    if (model[j].live && !model[j].unsupported && model[j].first <= r && r <= model[j].last)
                        model[j].dirty = 1;
            version[r] = v, seen[r] = 1, tracked[r] = t;
            fresh_deps_observe(d, r, v, t);
            // Inactive regions are reset when their first subscriber is inserted.
            forget_empty_regions();
        } else if (op == 5) {
            fresh_deps_mark_all(d);
            for (unsigned j = 0; j < NODES; j++) if (model[j].live) model[j].dirty = 1;
        }
        verify(d);
    }
    for (unsigned i = 0; i < NODES; i++) if (model[i].live) fresh_deps_remove(d, model[i].id);
    uint32_t count;
    fresh_deps_regions(d, &count);
    assert(!count && !fresh_deps_first(d));
    fresh_deps_destroy(d);
    puts("fresh_deps: deterministic cases and 100000 randomized operations passed");
    return 0;
}

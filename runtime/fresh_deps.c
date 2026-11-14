#include "fresh_deps.h"
#include <stdlib.h>
#include <string.h>

typedef struct { uint64_t mapping; uint32_t head, position, version, poll_position; uint8_t observed, tracked, polling; } Region;
typedef struct { uint32_t node, region, next, prev, own_next; } Link;
typedef struct {
    uintptr_t owner;
    uint32_t links, next, prev, free_next;
    uint8_t live, dirty, unsupported;
} Node;
struct FreshDeps {
    Region *region;
    Node *node;
    Link *link;
    uint32_t *active;
    uint32_t *poll;
    uint32_t regions, nodes, links, cnodes, clinks, nactive, cactive;
    uint32_t free_node, free_link, first, last;
    uint32_t npoll, cpoll;
    unsigned shift;
    int sorted, poll_sorted;
};

static int reserve(void **p, uint32_t *cap, uint32_t need, size_t stride) {
    if (need <= *cap) return 1;
    uint32_t n = *cap ? *cap : 32;
    while (n < need) {
        if (n > UINT32_MAX / 2) return 0;
        n *= 2;
    }
    if (n > SIZE_MAX / stride) return 0;
    void *q = realloc(*p, (size_t)n * stride);
    if (!q) return 0;
    memset((char *)q + (size_t)*cap * stride, 0, (size_t)(n - *cap) * stride);
    *p = q, *cap = n;
    return 1;
}

FreshDeps *fresh_deps_create(uint32_t regions, unsigned shift) {
    if (!regions || shift >= 63 || (uint64_t)regions > (UINT64_MAX >> shift)) return NULL;
    FreshDeps *d = calloc(1, sizeof *d);
    if (!d) return NULL;
    d->region = calloc(regions, sizeof *d->region);
    if (!d->region) { free(d); return NULL; }
    d->regions = regions, d->shift = shift, d->sorted = d->poll_sorted = 1;
    return d;
}

void fresh_deps_destroy(FreshDeps *d) {
    if (!d) return;
    free(d->region); free(d->node); free(d->link); free(d->active); free(d->poll); free(d);
}

void fresh_deps_mark(FreshDeps *d, uint32_t id) {
    if (!id || id > d->nodes || !d->node[id].live || d->node[id].dirty) return;
    Node *n = &d->node[id];
    n->dirty = 1, n->prev = d->last, n->next = 0;
    if (d->last) d->node[d->last].next = id;
    else d->first = id;
    d->last = id;
}

static void unqueue(FreshDeps *d, uint32_t id) {
    Node *n = &d->node[id];
    if (!n->dirty) return;
    if (n->prev) d->node[n->prev].next = n->next;
    else d->first = n->next;
    if (n->next) d->node[n->next].prev = n->prev;
    else d->last = n->prev;
    n->dirty = 0, n->prev = n->next = 0;
}

void fresh_deps_clean(FreshDeps *d, uint32_t id) {
    if (id && id <= d->nodes && d->node[id].live && !d->node[id].unsupported) unqueue(d, id);
}

void fresh_deps_sleep_region(FreshDeps *d, uint32_t region) {
    if (region >= d->regions) return;
    Region *r = &d->region[region];
    if (!r->polling) return;
    uint32_t moved = d->poll[--d->npoll];
    d->poll[r->poll_position] = moved;
    d->region[moved].poll_position = r->poll_position;
    r->polling = 0, d->poll_sorted = 0;
}

void fresh_deps_wake_region(FreshDeps *d, uint32_t region) {
    if (region >= d->regions) return;
    Region *r = &d->region[region];
    if (!r->head || r->polling) return;
    // Admission reserves capacity for every referenced region, including sleeping ones.
    r->poll_position = d->npoll;
    d->poll[d->npoll++] = region;
    r->polling = 1, d->poll_sorted = 0;
}

void fresh_deps_wake_all(FreshDeps *d) {
    for (uint32_t i = 0; i < d->nactive; i++) fresh_deps_wake_region(d, d->active[i]);
}

void fresh_deps_visit_sleeping(FreshDeps *d, void (*visit)(uint32_t region, void *user), void *user) {
    for (uint32_t i = 0; i < d->nactive; i++) {
        uint32_t r = d->active[i];
        if (!d->region[r].polling) visit(r, user);
    }
}

void fresh_deps_remove(FreshDeps *d, uint32_t id) {
    if (!id || id > d->nodes || !d->node[id].live) return;
    Node *n = &d->node[id];
    unqueue(d, id);
    for (uint32_t i = n->links; i;) {
        Link *l = &d->link[i];
        uint32_t next = l->own_next;
        Region *r = &d->region[l->region];
        if (l->prev) d->link[l->prev].next = l->next;
        else r->head = l->next;
        if (l->next) d->link[l->next].prev = l->prev;
        if (!r->head) {
            fresh_deps_sleep_region(d, l->region);
            uint32_t moved = d->active[--d->nactive];
            d->active[r->position] = moved;
            d->region[moved].position = r->position;
            r->observed = 0;
            d->sorted = 0;
        }
        l->next = d->free_link, d->free_link = i;
        i = next;
    }
    memset(n, 0, sizeof *n);
    n->free_next = d->free_node, d->free_node = id;
}

uint32_t fresh_deps_add(FreshDeps *d, uintptr_t owner, uint64_t addr, uint64_t size) {
    uint32_t id;
    if (d->free_node) id = d->free_node, d->free_node = d->node[id].free_next;
    else {
        if (d->nodes > UINT32_MAX - 2) return 0;
        if (!reserve((void **)&d->node, &d->cnodes, d->nodes + 2, sizeof *d->node)) return 0;
        id = ++d->nodes;
    }
    d->node[id] = (Node){.owner = owner, .live = 1};
    fresh_deps_mark(d, id);
    uint64_t limit = (uint64_t)d->regions << d->shift;
    if (!size || addr >= limit || size > limit - addr) {
        d->node[id].unsupported = 1;
        return id;
    }
    uint32_t first = (uint32_t)(addr >> d->shift), last = (uint32_t)((addr + size - 1) >> d->shift);
    for (uint32_t r = first; r <= last; r++) {
        uint32_t i;
        if (d->free_link) i = d->free_link, d->free_link = d->link[i].next;
        else {
            if (d->links > UINT32_MAX - 2) goto fail;
            if (!reserve((void **)&d->link, &d->clinks, d->links + 2, sizeof *d->link)) goto fail;
            i = ++d->links;
        }
        Region *reg = &d->region[r];
        if (!reg->head) {
            if (!reserve((void **)&d->active, &d->cactive, d->nactive + 1, sizeof *d->active) ||
                !reserve((void **)&d->poll, &d->cpoll, d->nactive + 1, sizeof *d->poll)) {
                d->link[i].next = d->free_link, d->free_link = i;
                goto fail;
            }
            reg->position = d->nactive;
            d->active[d->nactive++] = r;
            reg->observed = 0, d->sorted = 0;
        }
        d->link[i] = (Link){id, r, reg->head, 0, d->node[id].links};
        if (reg->head) d->link[reg->head].prev = i;
        reg->head = i, d->node[id].links = i;
        fresh_deps_wake_region(d, r);
    }
    return id;
fail:
    fresh_deps_remove(d, id);
    return 0;
}

void fresh_deps_mark_all(FreshDeps *d) {
    for (uint32_t i = 1; i <= d->nodes; i++) fresh_deps_mark(d, i);
}
int fresh_deps_dirty(const FreshDeps *d, uint32_t id) {
    return !id || id > d->nodes || !d->node[id].live || d->node[id].dirty;
}
uintptr_t fresh_deps_owner(const FreshDeps *d, uint32_t id) {
    return id && id <= d->nodes && d->node[id].live ? d->node[id].owner : 0;
}
uint32_t fresh_deps_first(const FreshDeps *d) { return d->first; }
uint32_t fresh_deps_next(const FreshDeps *d, uint32_t id) { return d->node[id].next; }
static int compare_region(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}
const uint32_t *fresh_deps_regions(FreshDeps *d, uint32_t *count) {
    if (!d->sorted) {
        qsort(d->active, d->nactive, sizeof *d->active, compare_region);
        for (uint32_t i = 0; i < d->nactive; i++) d->region[d->active[i]].position = i;
        d->sorted = 1;
    }
    *count = d->nactive;
    return d->active;
}
const uint32_t *fresh_deps_poll_regions(FreshDeps *d, uint32_t *count) {
    if (!d->poll_sorted) {
        qsort(d->poll, d->npoll, sizeof *d->poll, compare_region);
        for (uint32_t i = 0; i < d->npoll; i++) d->region[d->poll[i]].poll_position = i;
        d->poll_sorted = 1;
    }
    *count = d->npoll;
    return d->poll;
}
uint32_t fresh_deps_observe(FreshDeps *d, uint32_t region, uint32_t version, int tracked) {
    if (region >= d->regions) return 0;
    Region *r = &d->region[region];
    uint32_t visits = 0;
    if (!r->observed || r->version != version || !tracked || !r->tracked) {
        for (uint32_t i = r->head; i; i = d->link[i].next) {
            fresh_deps_mark(d, d->link[i].node);
            visits++;
        }
    }
    r->version = version, r->observed = 1, r->tracked = tracked != 0;
    return visits;
}

uint32_t fresh_deps_observe_mapping(FreshDeps *d, uint32_t region, uint64_t version) {
    if (region >= d->regions) return 0;
    Region *r = &d->region[region];
    uint32_t visits = 0;
    if (!version || r->mapping != version) {
        for (uint32_t i = r->head; i; i = d->link[i].next) {
            fresh_deps_mark(d, d->link[i].node);
            visits++;
        }
    }
    r->mapping = version;
    return visits;
}

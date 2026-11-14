#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int fail_after = -1;
static void *checked_realloc(void *p, size_t size) {
    if (fail_after == 0) return NULL;
    if (fail_after > 0) fail_after--;
    return realloc(p, size);
}
#define realloc checked_realloc
#include "../runtime/fresh_deps.c"
#undef realloc

enum { R = 73, N = 120 };
static struct { uint32_t id, first, last; } nodes[N];
static unsigned refs[R], awake[R];
static uint32_t seed = 0x539d77au;
static uint32_t word(void) { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed; }
static void verify(FreshDeps *d) {
    uint32_t na, np, a = 0, p = 0;
    const uint32_t *all = fresh_deps_regions(d, &na), *poll = fresh_deps_poll_regions(d, &np);
    for (uint32_t r = 0; r < R; r++) {
        if (refs[r]) assert(a < na && all[a++] == r);
        if (refs[r] && awake[r]) assert(p < np && poll[p++] == r);
    }
    assert(a == na && p == np);
    for (unsigned i = 0; i < N; i++) if (nodes[i].id) assert(!fresh_deps_dirty(d, nodes[i].id));
}
int main(void) {
    FreshDeps *d = fresh_deps_create(R, 12);
    assert(d);
    for (unsigned step = 0; step < 100000; step++) {
        unsigned i = word() % N, r = word() % R, op = word() % 5;
        if (!op && !nodes[i].id) {
            unsigned last = r + word() % 5; if (last >= R) last = R - 1;
            nodes[i].id = fresh_deps_add(d, i + 1, (uint64_t)r << 12, (uint64_t)(last - r + 1) << 12);
            assert(nodes[i].id);
            nodes[i].first = r, nodes[i].last = last;
            fresh_deps_clean(d, nodes[i].id);
            for (; r <= last; r++) refs[r]++, awake[r] = 1;
        } else if (!op && nodes[i].id) {
            fresh_deps_remove(d, nodes[i].id);
            for (r = nodes[i].first; r <= nodes[i].last; r++) if (!--refs[r]) awake[r] = 0;
            nodes[i].id = 0;
        } else if (op == 1) {
            fresh_deps_sleep_region(d, r); awake[r] = 0;
        } else if (op == 2) {
            fresh_deps_wake_region(d, r); if (refs[r]) awake[r] = 1;
        } else if (op == 3) {
            fresh_deps_wake_all(d);
            for (r = 0; r < R; r++) if (refs[r]) awake[r] = 1;
        } else {
            uint32_t n;
            const uint32_t *poll = fresh_deps_poll_regions(d, &n);
            for (uint32_t j = n; j-- > 0;) if (poll[j] & 1) {
                awake[poll[j]] = 0;
                fresh_deps_sleep_region(d, poll[j]);
            }
        }
        verify(d);
    }
    fresh_deps_destroy(d);
    // Every admission allocation may fail, including after linking some regions. A failed admission leaves no
    // subscriptions and later wake operations require no allocation, even when every referenced region slept.
    for (int failure = 0; failure < 12; failure++) {
        d = fresh_deps_create(R, 12); assert(d);
        fail_after = failure;
        uint32_t id = fresh_deps_add(d, 1, 0, (uint64_t)R << 12), n;
        fail_after = -1;
        fresh_deps_regions(d, &n); assert(n == (id ? R : 0));
        fresh_deps_poll_regions(d, &n); assert(n == (id ? R : 0));
        if (!id) id = fresh_deps_add(d, 1, 0, (uint64_t)R << 12);
        assert(id);
        for (unsigned r = 0; r < R; r++) fresh_deps_sleep_region(d, r);
        fail_after = 0;
        fresh_deps_wake_all(d);
        fresh_deps_poll_regions(d, &n); assert(n == R);
        fresh_deps_remove(d, id);
        fresh_deps_poll_regions(d, &n); assert(!n);
        fail_after = -1;
        fresh_deps_destroy(d);
    }
    puts("fresh polling: 100000 model operations, descending removal, admission failures and allocation-free wake passed");
}

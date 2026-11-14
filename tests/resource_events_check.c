#include "../runtime/resource_events.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static ResourceEvents log_;
static uint8_t observed[256], expected[256];
static uint32_t rng = 17;
static uint32_t next(void) { rng ^= rng << 13; rng ^= rng >> 17; return rng ^= rng << 5; }
static void visit(uint32_t first, uint32_t last, void *user) {
    uint8_t *set = user;
    assert(first <= last && last < 256);
    for (uint32_t i = first; i <= last; i++) set[i] = 1;
}

int main(void) {
    uint64_t cursor = 0, second = 0;
    assert(resource_events_read(&log_, &cursor, visit, observed));
    for (unsigned step = 0; step < 100000; step++) {
        uint32_t first = next() % 256, last = first + next() % (256 - first);
        resource_events_publish(&log_, first, last);
        visit(first, last, expected);
        if (step % 37 == 0) {
            memset(observed, 0, sizeof observed);
            assert(resource_events_read(&log_, &cursor, visit, observed));
            assert(!memcmp(observed, expected, sizeof observed));
            memset(expected, 0, sizeof expected);
        }
    }
    // An independent lagging reader cannot turn overwritten history into a clean answer.
    assert(!resource_events_read(&log_, &second, visit, observed));
    assert(second == resource_events_head(&log_));
    assert(resource_events_read(&log_, &second, visit, observed));
    // An in-flight replacement of the expected record fails closed.
    cursor = resource_events_head(&log_);
    resource_events_publish(&log_, 4, 9);
    uint64_t end = resource_events_head(&log_);
    ResourceEvent *e = &log_.event[(end - 1) % RESOURCE_EVENTS_CAPACITY];
    __atomic_store_n(&e->sequence, 0, __ATOMIC_SEQ_CST);
    assert(!resource_events_read(&log_, &cursor, visit, observed));
    __atomic_store_n(&e->sequence, end, __ATOMIC_SEQ_CST);
    // Reset/wrap cannot manufacture a valid observation in an earlier serial domain.
    cursor = end + 1;
    assert(!resource_events_read(&log_, &cursor, visit, observed));
    log_.head = UINT64_MAX;
    resource_events_publish(&log_, 0, 0);
    assert(log_.disabled && !resource_events_read(&log_, &cursor, visit, observed));
    memset(&log_, 0, sizeof log_);
    resource_events_publish(&log_, 2, 1);
    assert(log_.disabled);
    puts("resource event history: 100000 publications, independent cursors, overflow, busy, reset and wrap passed");
    return 0;
}

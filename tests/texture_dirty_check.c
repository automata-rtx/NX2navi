#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define AGEN_N 64
#define DIRTY_N 64
#define DIRTY_BUCKET 0x80000000u
static uint32_t dirty[DIRTY_N];
static uint64_t ndirty;
static int ntslots = 100;
static int32_t bucket_head[AGEN_N], next[100], *slot_next = next;
static int32_t bucket_ghead[AGEN_N];
static struct { struct { int head; } pub; int gnext; } grp[1];
static int group_calls;
static int calls[100], mutate;
#include "dirty_function.inc"

static void push(uint32_t x) { dirty[ndirty++ % DIRTY_N] = x; }
static void visit(int id, void *arg) {
    (void)arg;
    calls[id]++;
    if (mutate) { mutate = 0; push(DIRTY_BUCKET | 3); }
}
static void visit_group(int g, void *arg) {
    group_calls++;
    for (int id = grp[g].pub.head; id >= 0; id = slot_next[id]) visit(id, arg);
}

int main(void) {
    memset(bucket_head, -1, sizeof bucket_head);
    memset(bucket_ghead, -1, sizeof bucket_ghead);
    memset(next, -1, sizeof next);
    bucket_head[3] = 7;
    next[7] = 45;
    uint64_t pos = 0;
    for (int i = 0; i < 30; i++) push(DIRTY_BUCKET | 3);
    push(7);
    push(99);
    assert(vk_texture_dirty_slots_locked(&pos, visit, NULL, NULL));
    assert(calls[7] == 1 && calls[45] == 1 && calls[99] == 1);
    assert(pos == ndirty);

    push(7);
    mutate = 1;
    assert(vk_texture_dirty_slots_locked(&pos, visit, NULL, NULL));
    assert(ndirty == pos); // callback changes are drained in a second batch
    assert(calls[7] == 3 && calls[45] == 2);

    for (int i = 0; i < DIRTY_N + 1; i++) push(7);
    assert(!vk_texture_dirty_slots_locked(&pos, visit, NULL, NULL));
    assert(pos == ndirty && calls[7] == 3);
    push(99); // ring wrapping still preserves the newest notification
    assert(vk_texture_dirty_slots_locked(&pos, visit, NULL, NULL));
    assert(calls[99] == 2);
    bucket_ghead[3] = 0;
    grp[0].pub.head = 50;
    grp[0].gnext = -1;
    next[50] = 51;
    push(DIRTY_BUCKET | 3);
    push(DIRTY_BUCKET | 3);
    assert(vk_texture_dirty_slots_locked(&pos, visit, visit_group, NULL));
    assert(group_calls == 1 && calls[50] == 1 && calls[51] == 1);
    push(DIRTY_BUCKET | 3);
    push(50);
    assert(vk_texture_dirty_slots_locked(&pos, visit, NULL, NULL));
    assert(calls[50] == 2 && calls[51] == 2);
    puts("Texture dirty checks passed: duplicate buckets/slots, callback mutations, overflow and ring wrap");
    return 0;
}

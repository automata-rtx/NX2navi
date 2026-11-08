// Exercise the installer's queue with synthetic jobs. A parent writes the shared worker-limit byte
// before acknowledging each checkpoint on stdin. No GPU or game files are required.
#include <assert.h>
#include <stdio.h>
#include "shader_workers.h"

static ShaderWork queue;
static size_t claimed[5];

static void finish_one(void *arg) {
    size_t *previous = arg;
    *previous = shader_work_next(&queue, *previous);
}

static void counts(int active, size_t next) {
    for (int i = 0; i < 200; i++) {
        plat_mutex_lock(&queue.lock);
        int matches = queue.active == active && queue.next == next;
        plat_mutex_unlock(&queue.lock);
        if (matches) return;
        plat_sleep_ms(10);
    }
    assert(0 && "unexpected number of active or assigned jobs");
}

static void change(const char *message) {
    puts(message);
    fflush(stdout);
    assert(getchar() == '\n');
}

int main(void) {
    assert(shader_work_open(&queue, 20, 3, 1) && queue.limit && *queue.limit == 12);
    for (size_t i = 0; i < 12; i++) assert(shader_work_next(&queue, SIZE_MAX) == i);
    change("set 8");
    PlatThread *threads[5];
    for (size_t i = 0; i < 5; i++) {
        claimed[i] = i;
        threads[i] = plat_thread_new(finish_one, &claimed[i], "finish shader");
        assert(threads[i]);
        if (i < 4) counts(11 - (int)i, 12);   // four finish without taking any new work
    }
    counts(8, 13);   // the fifth frees the first slot below eight
    change("set 12");
    for (int i = 0; i < 5; i++) plat_thread_join(threads[i]);
    counts(12, 17);   // four workers resume even while the other eight stay busy
    unsigned seen = 0;
    for (int i = 0; i < 5; i++) {
        assert(claimed[i] >= 12 && claimed[i] < 17);
        seen |= 1u << (claimed[i] - 12);
    }
    assert(seen == 31);
    for (size_t i = 17; i < 20; i++) assert(shader_work_next(&queue, 0) == i);
    for (int i = 0; i < 12; i++) assert(shader_work_next(&queue, 0) == SIZE_MAX);
    counts(0, 20);
    return 0;
}

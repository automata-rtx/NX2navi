#include <windows.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include "resource_events.h"

static ResourceEvents events;
static int done;
typedef struct { uint64_t expected, calls, complete, rebuilds; int lag; } Reader;
static void verify_event(uint32_t first, uint32_t last, void *user) {
    Reader *r = user;
    assert(first == r->expected++ && last == first + 17);
    r->calls++;
}
static DWORD WINAPI consume(void *user) {
    Reader *r = user;
    uint64_t cursor = 0;
    do {
        r->expected = cursor + 1;
        int complete = resource_events_read(&events, &cursor, verify_event, r);
        if (complete) {
            assert(r->expected == cursor + 1);
            r->complete++;
        } else r->rebuilds++;
        if (r->lag) Sleep(1);
        else SwitchToThread();
    } while (!__atomic_load_n(&done, __ATOMIC_ACQUIRE) || cursor != resource_events_head(&events));
    assert(cursor == 250000);
    return 0;
}
int main(void) {
    Reader reader[2] = {{0}, {.lag = 1}};
    HANDLE threads[2];
    for (unsigned i = 0; i < 2; i++) {
        threads[i] = CreateThread(NULL, 0, consume, &reader[i], 0, NULL);
        assert(threads[i]);
    }
    for (uint32_t i = 1; i <= 250000; i++) resource_events_publish(&events, i, i + 17);
    __atomic_store_n(&done, 1, __ATOMIC_RELEASE);
    assert(WaitForMultipleObjects(2, threads, TRUE, 10000) == WAIT_OBJECT_0);
    for (unsigned i = 0; i < 2; i++) {
        assert(reader[i].complete);
        CloseHandle(threads[i]);
    }
    printf("resource event concurrency: 250000 publications, two readers, %llu checked records and %llu rebuilds passed\n",
           (unsigned long long)(reader[0].calls + reader[1].calls),
           (unsigned long long)(reader[0].rebuilds + reader[1].rebuilds));
}

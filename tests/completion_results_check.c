#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

typedef SRWLOCK PlatMutex;
static void plat_mutex_lock(PlatMutex *m) { AcquireSRWLockExclusive(m); }
static void plat_mutex_unlock(PlatMutex *m) { ReleaseSRWLockExclusive(m); }
static void plat_cond_broadcast(int *c) { (*c)++; }
#define rt_fatal(...) abort()
typedef struct { uint64_t fence; void (*fn)(void *); void *arg; } Pending;
static PlatMutex st_lock = SRWLOCK_INIT, res_lock = SRWLOCK_INIT;
static Pending *queue, *rqueue;
static int qhead, qn, qcap, rqhead, rqn, rqcap, cv_work;
static uint64_t done;
#include "results.inc"

static int next_result, active;
static void result(void *arg) {
    assert(InterlockedIncrement((LONG *)&active) == 1);
    assert((intptr_t)arg == next_result++);
    Sleep(0);
    assert(InterlockedDecrement((LONG *)&active) == 0);
}
static DWORD WINAPI drain(void *arg) { results_run((uintptr_t)arg); return 0; }

enum { OQ_N = 1024, VK_SUCCESS = 0, VK_NOT_READY = 1,
       VK_QUERY_RESULT_64_BIT = 1, VK_QUERY_RESULT_WITH_AVAILABILITY_BIT = 2 };
typedef int VkDevice;
typedef int VkResult;
static int oq_pool, resets;
static uint32_t oq_next;
static uint8_t oq_busy[OQ_N];
static uint64_t values[OQ_N], available[OQ_N];
static VkResult vkGetQueryPoolResults(VkDevice d, int p, uint32_t first, uint32_t n,
                                     size_t bytes, void *out, size_t stride, int flags) {
    (void)d; (void)p; (void)flags;
    assert(first + n <= OQ_N && bytes >= n * stride);
    int missing = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint64_t *v = (uint64_t *)((char *)out + i * stride);
        v[0] = values[first + i]; v[1] = available[first + i];
        missing |= !v[1];
    }
    return missing ? VK_NOT_READY : VK_SUCCESS;
}
static void vkResetQueryPool(VkDevice d, int p, uint32_t first, uint32_t n) {
    (void)d; (void)p;
    assert(first + n <= OQ_N);
    for (uint32_t i = first; i < first + n; i++) {
        assert(oq_busy[i]); available[i] = 0;
    }
    resets++;
}
static void vk_device_fault_report(int r) { (void)r; abort(); }
#include "queries.inc"

int main(void) {
    for (intptr_t i = 0; i < 1000; i++) vk_completion_add_result(i + 1, result, (void *)i);
    // A completed prefix becomes visible; later values and ordinary work stay queued.
    vk_completion_add(1, result, (void *)9999);
    results_run(100);
    assert(next_result == 100 && rqn == 900 && qn == 1);
    HANDLE a = CreateThread(NULL, 0, drain, (void *)1000, 0, NULL);
    HANDLE b = CreateThread(NULL, 0, drain, (void *)1000, 0, NULL);
    assert(a && b);
    assert(WaitForSingleObject(a, 10000) == WAIT_OBJECT_0);
    assert(WaitForSingleObject(b, 10000) == WAIT_OBJECT_0);
    CloseHandle(a); CloseHandle(b);
    assert(next_result == 1000 && !rqn && qn == 1);
    // Queue another value after both consumers have drained the earlier values.
    vk_completion_add_result(1001, result, (void *)1000);
    results_run(1001);
    assert(next_result == 1001);
    free(queue); free(rqueue);

    for (int i = 0; i < OQ_N; i++) values[i] = 100 + i, available[i] = oq_busy[i] = 1;
    oq_next = 600; oq_batch_mode = 1;
    int avail;
    for (uint32_t q = 0; q < 300; q++) assert(oq_take(0, q, &avail) == 100 + q && avail);
    oq_reset_flush(0);
    assert(OQB.calls == 2 && resets == 2);
    for (int i = 0; i < 300; i++) assert(!oq_busy[i]);
    // A future slot observed unavailable in a batch must be re-read once ready.
    oq_batch_drop();
    available[400] = 0;
    assert(oq_take(0, 399, &avail) == 499 && avail);
    available[400] = 1; values[400] = 900;
    assert(oq_take(0, 400, &avail) == 900 && avail);
    oq_reset_flush(0);
    // Ring wrap and recycling must not return a cached value from a prior use.
    oq_next = 3;
    assert(oq_take(0, 1023, &avail) == 1123 && avail);
    values[0] = 2000; available[0] = oq_busy[0] = 1;
    assert(oq_take(0, 0, &avail) == 2000 && avail);
    oq_reset_flush(0);
    oq_batch_drop(); oq_batch_mode = 0;
    uint64_t calls = OQB.calls;
    for (int q = 500; q < 502; q++) assert(oq_take(0, q, &avail) == 100u + q && avail);
    oq_reset_flush(0);
    assert(OQB.calls == calls + 2);
    puts("Completion ordering and query batching checks passed");
}

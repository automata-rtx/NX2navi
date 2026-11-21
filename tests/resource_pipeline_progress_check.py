"""Host tests for pipeline admission, worker publication and cancellation."""

import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = (ROOT / "runtime/vk_pipeline.c").read_text(encoding="utf-8")
VK_HEADER = (ROOT / "runtime/vk.h").read_text(encoding="utf-8")


def function(name):
    match = re.search(r"^(?:static )?[^\n;{}]*\b" + re.escape(name) + r"\([^;{}]*\)\s*\{", PIPELINE, re.M)
    if not match:
        raise ValueError(name)
    depth = 1
    end = match.end()
    while depth:
        depth += (PIPELINE[end] == "{") - (PIPELINE[end] == "}")
        end += 1
    return PIPELINE[match.start():end]


def section(first, following):
    start = PIPELINE.index(first)
    return PIPELINE[start:PIPELINE.index(following, start + len(first))]


base = r'''
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#define RC_BUDGET_TEST
#include "resource_core_budget.h"
#include "rc_native_rmw_word.h"

#define TEST_MB (1024ull * 1024ull)
#define REQUIRED_MARGIN_MB 256ull
#define ROOM 64ull

enum Wake {
    WAKE_GLOBAL,
    WAKE_UNRELATED_HEAP,
    WAKE_RAM,
    WAKE_COMMIT,
    WAKE_STOP,
    WAKE_LOCAL_STOP,
    WAKE_RECORD,
};
typedef struct { enum Wake wake; uint64_t advance_ns; } WakeStep;
static RcBudgetSample telemetry;
static uint64_t test_clock = 1000000000ull;
static unsigned mutex_depth, budget_depth;
static unsigned wait_count, service_count, driver_calls;
static const WakeStep *wakes;
static volatile int32_t local_stop;
static RcBudgetTicket *record_to_release;
static int device_lost;
static unsigned wake_count;
static int stop_during_service;

void vk_pipeline_stop_preparation(void);
static int vk_device_is_lost(void) { return device_lost; }
static void vk_device_lost_check(VkResult result, const char *operation) {
    assert(operation && *operation);
    if (result == VK_ERROR_DEVICE_LOST) device_lost = 1;
}
static void assert_external(void) { assert(!mutex_depth && !budget_depth); }

uint64_t rc_budget_test_now_ns(void) { return test_clock; }
void rc_budget_test_lock(void) { assert_external(); budget_depth++; }
void rc_budget_test_unlock(void) { assert(budget_depth == 1); budget_depth--; }
void rc_budget_test_sample(RcBudgetSample *out) {
    assert(budget_depth == 1);
    *out = telemetry;
}
uint64_t plat_now_ns(void) { return test_clock; }
void rt_log(const char *text, ...) { (void)text; }
_Noreturn void rt_fatal(const char *text, ...) {
    va_list args;
    va_start(args, text);
    vfprintf(stderr, text, args);
    va_end(args);
    fputc('\n', stderr);
    abort();
}

static uint64_t free_at(uint64_t amount) { return (REQUIRED_MARGIN_MB + amount) * TEST_MB; }
static void change_room(uint64_t ram, uint64_t commit) {
    telemetry.free.amount[RC_BUDGET_RAM] = free_at(ram);
    telemetry.free.amount[RC_BUDGET_COMMIT] = free_at(commit);
}

void rc_budget_test_wait(uint32_t milliseconds) {
    assert_external();
    assert(milliseconds && wait_count < wake_count && !driver_calls);
    WakeStep step = wakes[wait_count++];
    test_clock += step.advance_ns;
    switch (step.wake) {
    case WAKE_GLOBAL:
        resource_core_budget_progress();
        break;
    case WAKE_UNRELATED_HEAP:
        telemetry.free.amount[RC_BUDGET_GPU_FIRST] += 2048 * TEST_MB;
        resource_core_budget_refresh();
        break;
    case WAKE_RAM:
        telemetry.free.amount[RC_BUDGET_RAM] = free_at(ROOM);
        resource_core_budget_refresh();
        break;
    case WAKE_COMMIT:
        telemetry.free.amount[RC_BUDGET_COMMIT] = free_at(ROOM);
        resource_core_budget_refresh();
        break;
    case WAKE_STOP:
        vk_pipeline_stop_preparation();
        break;
    case WAKE_LOCAL_STOP:
        local_stop = 1;
        break;
    case WAKE_RECORD:
        assert(record_to_release && record_to_release->token.ledger);
        assert(resource_core_budget_cancel(record_to_release) == RC_ADMISSION_OK);
        record_to_release = NULL;
        break;
    }
}

static unsigned reclaim_probe(void *context, const RcAdmissionVector *deficit) {
    assert_external();
    assert(!context && (deficit->amount[RC_BUDGET_RAM] || deficit->amount[RC_BUDGET_COMMIT]));
    service_count++;
    if (stop_during_service) {
        change_room(ROOM, ROOM);
        test_clock += 1000000000ull;
        vk_pipeline_stop_preparation();
        return RC_BUDGET_RECLAIM_DESTROYED;
    }
    return RC_BUDGET_RECLAIM_NONE;
}

#include "resource_core_budget.c"

static void reset_ledger(uint64_t ram, uint64_t commit) {
    assert_external();
    resource_core_budget_test_reset();
    telemetry = (RcBudgetSample){.known_mask = (1u << (RC_BUDGET_GPU_FIRST + 1)) - 1};
    telemetry.total.amount[RC_BUDGET_RAM] = 16384 * TEST_MB;
    telemetry.total.amount[RC_BUDGET_COMMIT] = 32768 * TEST_MB;
    telemetry.total.amount[RC_BUDGET_GPU_FIRST] = 8192 * TEST_MB;
    telemetry.free.amount[RC_BUDGET_GPU_FIRST] = 2048 * TEST_MB;
    change_room(ram, commit);
    wait_count = service_count = driver_calls = wake_count = 0;
    stop_during_service = local_stop = device_lost = 0;
    wakes = NULL;
    record_to_release = NULL;
    assert(resource_core_budget_register_reclaimer(RC_BUDGET_BUFFER,
        (1u << RC_BUDGET_RAM) | (1u << RC_BUDGET_COMMIT), reclaim_probe, NULL) == RC_ADMISSION_OK);
    resource_core_budget_refresh();
}

static RcBudgetProducerStats pipeline_stats(void) {
    RcBudgetSnapshot snapshot;
    resource_core_budget_snapshot(&snapshot);
    return snapshot.producer[RC_BUDGET_PIPELINE];
}

static void assert_charge(uint64_t reserved, uint64_t opaque) {
    RcBudgetProducerStats stats = pipeline_stats();
    for (unsigned d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) {
        int constrained = d == RC_BUDGET_RAM || d == RC_BUDGET_COMMIT;
        assert(stats.reserved.amount[d] == (constrained ? reserved * TEST_MB : 0));
        assert(stats.opaque_debt.amount[d] == (constrained ? opaque * TEST_MB : 0));
        assert(!stats.retained.amount[d] && !stats.pending.amount[d]);
    }
}

static void assert_reservation(const RcBudgetTicket *ticket, uint64_t mb) {
    RcAdmissionVector actual;
    assert(resource_core_budget_ticket_cost(ticket, &actual) == RC_ADMISSION_OK);
    for (unsigned d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++)
        assert(actual.amount[d] == ((d == RC_BUDGET_RAM || d == RC_BUDGET_COMMIT) ? mb * TEST_MB : 0));
}
'''

admission = section("// One compiler invocation owns", "static VkResult pipe_create_one(")
optional = "\n".join(function(name) for name in
                     ("gate_enter", "gate_leave", "gate_text", "gate_wait_for"))

test_admission = r'''
static void reset(uint64_t ram, uint64_t commit) {
    assert(!pipe_budget.ticket.token.ledger && !pipe_budget.native_entered);
    pipe_budget_stopping = 0;
    assert(!A.inflight && !A.external && !A.threads && !U.lib_workers);
    A.closed = 0;
    memset(&K, 0, sizeof K);
    reset_ledger(ram, commit);
}

static void check_exact_wait(void) {
    reset(16, 16);
    const WakeStep script[] = {
        {WAKE_GLOBAL, 0},
        {WAKE_GLOBAL, 0},
        {WAKE_UNRELATED_HEAP, 2000000},
        {WAKE_RAM, 2000000},
        {WAKE_COMMIT, 31000000000ull},
    };
    wakes = script;
    wake_count = sizeof script / sizeof script[0];

    assert(pipe_budget_acquire(ROOM, 1, NULL, 0) == PIPE_BUDGET_OWNED);
    assert(wait_count == wake_count && service_count == 1 && !driver_calls);
    RcBudgetProducerStats stats = pipeline_stats();
    assert(stats.admitted == 1 && stats.refused == 1);
    assert_reservation(&pipe_budget.ticket, ROOM);
    assert_charge(ROOM, 0);
    pipe_budget_end(PIPE_BUDGET_OWNED);
    assert_charge(0, 0);
}

static void check_record_availability(void) {
    reset(ROOM, ROOM);
    static RcBudgetTicket other[BUDGET_RECORDS];
    const RcAdmissionVector unrelated = {.amount[RC_BUDGET_STAGING] = 1};
    for (unsigned i = 0; i < BUDGET_RECORDS; i++)
        assert(resource_core_budget_reserve(RC_BUDGET_OTHER, 1, &unrelated, &other[i]) == RC_ADMISSION_OK);

    const WakeStep script[] = {{WAKE_GLOBAL, 0}, {WAKE_UNRELATED_HEAP, 2000000}, {WAKE_RECORD, 0}};
    wakes = script;
    wake_count = sizeof script / sizeof script[0];
    record_to_release = &other[0];
    assert(pipe_budget_acquire(ROOM, 1, NULL, 0) == PIPE_BUDGET_OWNED);
    RcBudgetProducerStats stats = pipeline_stats();
    assert(stats.admitted == 1 && stats.refused == 1 && stats.last_refusal.status == RC_ADMISSION_NO_RECORDS);
    assert(wait_count == wake_count && !service_count && !driver_calls && !record_to_release);
    assert_charge(ROOM, 0);
    pipe_budget_end(PIPE_BUDGET_OWNED);
    for (unsigned i = 1; i < BUDGET_RECORDS; i++)
        assert(resource_core_budget_cancel(&other[i]) == RC_ADMISSION_OK);
    assert_charge(0, 0);
}

static void check_terminal_estimate(void) {
    reset(ROOM, ROOM);
    RcAdmissionStatus reason;
    assert(pipe_budget_try_begin(0, 1, &reason) == PIPE_BUDGET_TERMINAL && reason == RC_ADMISSION_INVALID);
    assert(pipe_budget_acquire(16385, 1, NULL, 0) == PIPE_BUDGET_TERMINAL);
    assert(!wait_count && !service_count && !pipeline_stats().admitted && !driver_calls);
    assert_charge(0, 0);
}

static void check_required_stop(void) {
    reset(16, 16);
    const WakeStep script[] = {{WAKE_GLOBAL, 0}, {WAKE_STOP, 31000000000ull}};
    wakes = script;
    wake_count = sizeof script / sizeof script[0];

    assert(pipe_budget_acquire(ROOM, 1, NULL, 0) == PIPE_BUDGET_STOPPED);
    assert(wait_count == wake_count && !pipe_budget.ticket.token.ledger && !driver_calls);
    assert_charge(0, 0);

    reset(ROOM, ROOM);
    vk_pipeline_stop_preparation();
    RcAdmissionStatus reason;
    assert(pipe_budget_acquire(ROOM, 1, NULL, 0) == PIPE_BUDGET_STOPPED);
    assert(pipe_budget_try_begin(ROOM, 0, &reason) == PIPE_BUDGET_STOPPED);
    assert(!wait_count && !service_count && !pipeline_stats().admitted && !driver_calls);
    assert_charge(0, 0);
}

static void check_stop_during_service(void) {
    reset(16, 16);
    stop_during_service = 1;
    assert(pipe_budget_acquire(ROOM, 1, NULL, 0) == PIPE_BUDGET_STOPPED);
    assert(service_count == 1 && !wait_count && !pipeline_stats().admitted && !driver_calls);
    assert_charge(0, 0);
}

static void check_optional_capacity_and_stop(void) {
    // Enough room inside the protected margin for required 64 MiB, but none
    // above the optional reserve. An optional wait cannot use it or service reclaim.
    reset(ROOM, ROOM);
    const WakeStep deadline[] = {{WAKE_GLOBAL, 31000000000ull}};
    wakes = deadline;
    wake_count = 1;
    double took = 0;
    assert(!gate_wait_for(ROOM, "optional startup", &local_stop, &took));
    assert(took == ROOM && wait_count == 1 && !service_count && !pipeline_stats().admitted);
    assert_charge(0, 0);
    assert(pipe_budget_begin(ROOM) == PIPE_BUDGET_OWNED);
    pipe_budget_end(PIPE_BUDGET_OWNED);

    reset(16, 16);
    const WakeStep canceled[] = {{WAKE_LOCAL_STOP, 0}};
    wakes = canceled;
    wake_count = 1;
    assert(!gate_wait_for(ROOM, "optional repair", &local_stop, &took));
    assert(local_stop && wait_count == 1 && !service_count && !pipeline_stats().admitted);
    assert_charge(0, 0);
}

static void check_nested_and_settlement(void) {
    reset(8192, 8192);
    double took = 0;
    assert(gate_enter(ROOM, &took) && took == ROOM);
    assert_reservation(&pipe_budget.ticket, ROOM);
    assert_charge(ROOM, 0);
    RcBudgetTicket first = pipe_budget.ticket;
    RcAdmissionStatus reason;
    assert(pipe_budget_try_begin(32, 0, &reason) == PIPE_BUDGET_BORROWED && reason == RC_ADMISSION_OK);
    assert(pipe_budget_begin(ROOM) == PIPE_BUDGET_BORROWED);
    pipe_budget_end(PIPE_BUDGET_BORROWED);
    assert(pipe_budget_try_begin(80, 0, &reason) == PIPE_BUDGET_REFUSED && reason == RC_ADMISSION_REFUSED);
    assert(pipe_budget_begin(80) == PIPE_BUDGET_REFUSED);
    assert(!wait_count && !service_count && pipeline_stats().admitted == 1 && !driver_calls);
    assert(first.token.ledger == pipe_budget.ticket.token.ledger &&
           first.token.slot == pipe_budget.ticket.token.slot &&
           first.token.generation == pipe_budget.ticket.token.generation);
    assert_reservation(&first, ROOM);
    assert_charge(ROOM, 0);

    gate_leave(took);
    assert_charge(0, 0);
    assert(!pipe_budget.ticket.token.ledger && !pipe_budget.native_entered);
}
'''

worker_types = r'''
typedef int PlatMutex;
typedef int PlatCond;
typedef struct PlatThread { unsigned joined; } PlatThread;
#define PLAT_COND_INIT 0
typedef struct { unsigned unused; } Layout;
typedef struct { unsigned tag; } GKey;
typedef struct { unsigned tag; const void *vs; uint8_t native_rmw_marker; } VkPipeState;
typedef struct {
    VkPipeline pipeline;
    RcNativeRmwWord native_rmw;
    uint32_t native_rmw_binding;
} VkPipeInfo;
'''
graphics_api_start = VK_HEADER.index("typedef enum {\n    VK_GRAPHICS_READY")
worker_types += VK_HEADER[graphics_api_start:VK_HEADER.index("double vk_pipeline_building_ms", graphics_api_start)]
worker_types += section("typedef struct {\n    uint64_t vs, fs;", "typedef struct Pipe {")
worker_types += r'''
typedef struct {
    struct Job *job;
    VkPipeInfo info;
    VkGraphicsPipelineRevision fast, full;
    uint32_t required_waiters;
    uint64_t t_miss;
    const char *graphics_failure;
    VkResult graphics_error;
    RcAdmissionStatus graphics_admission;
    unsigned required_layout, graphics_unsupported;
    unsigned building, link_queued;
} Pipe;
static struct {
    PlatMutex lock;
    int ready;
    VkDevice device;
    struct { unsigned swaps; } g;
    struct { unsigned graphics, async_built, failures, hits; } stats;
} P;
static struct {
    unsigned loading, left, failed, built, promoted, gate_ended;
    double until;
} K;
static PlatCond done_cv, work_cv, lib_cv, autosave_wake;

static void plat_mutex_lock(PlatMutex *lock) {
    assert(lock == &P.lock);
    assert_external();
    mutex_depth++;
}
static void plat_mutex_unlock(PlatMutex *lock) { assert(lock == &P.lock && mutex_depth == 1); mutex_depth--; }
static void plat_cond_wait(PlatCond *cond, PlatMutex *lock) { (void)cond; (void)lock; abort(); }
static void plat_cond_broadcast(PlatCond *cond) { assert(cond && mutex_depth == 1); }
static void plat_thread_join(PlatThread *thread) { assert_external(); assert(thread && !thread->joined++); }
'''

worker_owners = (
    section("enum { LIB_VI,", "// pre:") +
    section("typedef struct Job {", "// ---- pipeline keys") +
    section("typedef struct LibWait {", "static void u_library_complete(Lib *library);")
)
worker_close = "\n".join(function(name) for name in
                         ("job_unlink", "job_cancel", "u_push", "u_block", "u_unblock", "u_unwait", "u_library_complete", "u_link_owner", "u_cancel",
                          "workers_close", "workers_drain", "vk_pipeline_stop_preparation"))

test_workers = r'''
static void check_drain_graphics(void) {
    reset(ROOM, ROOM);
    Pipe boot = {.building = 1};
    VkPipeline ready = (VkPipeline)(uintptr_t)21;
    Pipe optimized = {.building = 1, .info.pipeline = ready, .fast.pipeline = ready};
    Job *first = calloc(1, sizeof *first);
    Job *second = calloc(1, sizeof *second);
    assert(first && second);
    *first = (Job){.next = second, .p = &boot, .low = 1, .prebuild = 1, .worker = -1};
    *second = (Job){.prev = first, .p = &optimized, .low = 1, .worker = -1};
    boot.job = first;
    optimized.job = second;
    A.low = first;
    A.low_tail = second;
    A.inflight = 2;
    K.left = 1;
    K.loading = 1;

    PlatThread graphics_worker = {0}, library_worker = {0};
    A.threads = U.lib_workers = 1;
    A.thread[0] = &graphics_worker;
    U.thread[0] = &library_worker;
    vk_pipeline_stop_preparation();
    assert(A.closed && !A.low && !A.low_tail && !A.inflight && !K.left && !K.loading && !K.failed);
    assert(!boot.job && !boot.building && !boot.info.pipeline);
    assert(!optimized.job && !optimized.building && optimized.info.pipeline == ready && optimized.fast.pipeline == ready);
    assert(!graphics_worker.joined && !library_worker.joined);
    workers_drain();
    workers_drain();
    assert(graphics_worker.joined == 1 && library_worker.joined == 1);
    assert(!A.threads && !U.lib_workers && !A.inflight && !K.left && !mutex_depth);
}

static void check_library_prerequisites(void) {
    reset(ROOM, ROOM);
    Lib first = {.building = 1}, second = {.building = 1}, already_ready = {0};
    VkPipeline fallback = (VkPipeline)(uintptr_t)52;
    Pipe target = {.building = 1, .link_queued = 1, .info.pipeline = fallback, .fast.pipeline = fallback};
    Job *owner = calloc(1, sizeof *owner);
    LibJob *link = calloc(1, sizeof *link);
    assert(owner && link);
    *owner = (Job){.p = &target, .worker = JOB_WAIT_LINK};
    *link = (LibJob){.p = &target};
    target.job = owner;
    A.inflight = 1;

    plat_mutex_lock(&P.lock);
    Lib *deps[LIB_COUNT] = {&first, &second, &already_ready, &already_ready};
    u_block(link, deps);
    assert(U.blocked == link && link->remaining == 2 && !U.head);
    first.building = 0;
    u_library_complete(&first);
    assert(U.blocked == link && link->remaining == 1 && !U.head && !first.waiters && second.waiters);
    second.building = 0;
    u_library_complete(&second);
    assert(!U.blocked && !link->remaining && U.head == link && U.tail == link && !second.waiters);
    u_library_complete(&second);
    assert(U.head == link && U.tail == link && !link->next);
    plat_mutex_unlock(&P.lock);

    workers_drain();
    assert(!A.inflight && !target.building && !target.link_queued && !target.job);
    assert(target.info.pipeline == fallback && target.fast.pipeline == fallback);
}

static void check_drain_libraries(void) {
    reset(ROOM, ROOM);
    Lib required = {.building = 1}, optional = {.building = 1}, available = {0};
    Pipe ready = {.link_queued = 1}, waiting = {.link_queued = 1};
    LibJob *shader = calloc(1, sizeof *shader);
    LibJob *speculative = calloc(1, sizeof *speculative);
    LibJob *link = calloc(1, sizeof *link);
    LibJob *blocked = calloc(1, sizeof *blocked);
    Job *ready_owner = calloc(1, sizeof *ready_owner), *waiting_owner = calloc(1, sizeof *waiting_owner);
    assert(shader && speculative && link && blocked && ready_owner && waiting_owner);
    *ready_owner = (Job){.p = &ready, .worker = JOB_WAIT_LINK};
    *waiting_owner = (Job){.p = &waiting, .prebuild = 1, .worker = JOB_WAIT_LINK};
    ready.building = waiting.building = 1;
    ready.job = ready_owner;
    waiting.job = waiting_owner;
    A.inflight = 2;
    K.left = 1;
    *shader = (LibJob){.b = &required};
    *speculative = (LibJob){.b = &optional, .optional = 1};
    *link = (LibJob){.p = &ready};
    *blocked = (LibJob){.p = &waiting};
    required.job = shader;
    optional.job = speculative;
    plat_mutex_lock(&P.lock);
    u_push(shader);
    u_push(speculative);
    u_push(link);
    Lib *deps[LIB_COUNT] = {&required, &optional, &available, &available};
    u_block(blocked, deps);
    assert(U.blocked == blocked && blocked->remaining == 2 && required.waiters && optional.waiters);
    plat_mutex_unlock(&P.lock);

    workers_drain();
    assert(!required.job && !required.building && !required.waiters);
    assert(!optional.job && !optional.building && !optional.waiters);
    assert(!ready.link_queued && !ready.job && !ready.building);
    assert(!waiting.link_queued && !waiting.job && !waiting.building && !K.left && !A.inflight);
    assert(!U.head && !U.tail && !U.low && !U.low_tail && !U.blocked && !mutex_depth);
    workers_drain();
    assert(A.closed && !mutex_depth);
}
'''

native_create = function("pipe_create_one")
native_stubs = r'''
static VkResult native_result;
static VkPipeline expected_handle;
static int expected_compute;
static VkResult native_compile(int compute, VkDevice device, VkPipelineCache cache, uint32_t count,
                               const void *create, const VkAllocationCallbacks *allocator, VkPipeline *out) {
    assert(!mutex_depth && !budget_depth && !device_lost && !pipe_budget_stopping);
    assert(pipe_budget.native_entered && pipe_budget.ticket.token.ledger);
    assert(device == P.device && count == 1 && compute == expected_compute && create && !allocator);
    assert(cache == VK_NULL_HANDLE);
    driver_calls++;
    *out = native_result == VK_SUCCESS ? expected_handle : VK_NULL_HANDLE;
    return native_result;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateComputePipelines(
    VkDevice device, VkPipelineCache cache, uint32_t count, const VkComputePipelineCreateInfo *create,
    const VkAllocationCallbacks *allocator, VkPipeline *out) {
    return native_compile(1, device, cache, count, create, allocator, out);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateGraphicsPipelines(
    VkDevice device, VkPipelineCache cache, uint32_t count, const VkGraphicsPipelineCreateInfo *create,
    const VkAllocationCallbacks *allocator, VkPipeline *out) {
    return native_compile(0, device, cache, count, create, allocator, out);
}

static void check_native_settlement(void) {
    reset(8192, 8192);
    P.device = (VkDevice)(uintptr_t)333;
    expected_handle = (VkPipeline)(uintptr_t)444;
    VkComputePipelineCreateInfo compute = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    VkGraphicsPipelineCreateInfo graphics = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};

    assert(pipe_budget_begin(ROOM) == PIPE_BUDGET_OWNED);
    vk_pipeline_stop_preparation();
    VkPipeline out = expected_handle;
    assert(pipe_create_one(2, &compute, &out, VK_NULL_HANDLE) == VK_ERROR_INITIALIZATION_FAILED);
    assert(!out && !driver_calls && !pipe_budget.native_entered);
    pipe_budget_end(PIPE_BUDGET_OWNED);
    assert_charge(0, 0);

    reset(8192, 8192);
    assert(pipe_budget_begin(ROOM) == PIPE_BUDGET_OWNED);
    expected_compute = 1;
    native_result = VK_SUCCESS;
    assert(pipe_create_one(2, &compute, &out, VK_NULL_HANDLE) == VK_SUCCESS && out == expected_handle);
    assert(driver_calls == 1);
    assert_charge(ROOM, 0);
    pipe_budget_end(PIPE_BUDGET_OWNED);
    assert_charge(0, ROOM);
    test_clock += 1000000000ull;
    resource_core_budget_refresh();
    assert_charge(0, 0);

    reset(8192, 8192);
    assert(pipe_budget_begin(ROOM) == PIPE_BUDGET_OWNED);
    expected_compute = 0;
    native_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    assert(pipe_create_one(1, &graphics, &out, VK_NULL_HANDLE) == VK_ERROR_OUT_OF_DEVICE_MEMORY && !out);
    assert(driver_calls == 1 && !device_lost);
    pipe_budget_end(PIPE_BUDGET_OWNED);
    assert_charge(0, ROOM);

    reset(8192, 8192);
    assert(pipe_budget_begin(ROOM) == PIPE_BUDGET_OWNED);
    native_result = VK_ERROR_DEVICE_LOST;
    assert(pipe_create_one(1, &graphics, &out, VK_NULL_HANDLE) == VK_ERROR_DEVICE_LOST && !out);
    assert(driver_calls == 1 && device_lost);
    assert(pipe_create_one(1, &graphics, &out, VK_NULL_HANDLE) == VK_ERROR_INITIALIZATION_FAILED);
    assert(driver_calls == 1);
    pipe_budget_end(PIPE_BUDGET_OWNED);
    assert_charge(0, ROOM);
    assert(pipe_budget_begin(ROOM) == PIPE_BUDGET_STOPPED);
}
'''

warm_stubs = r'''
static int prep_mode;
static unsigned startup_compiler_calls;
static unsigned retired_count;
static VkPipeline expected_retired;
static void *completion_owner;
static void (*completion_cancel)(void *, int);
enum { PERF_CNT_PIPE_ASYNC };
static void ready_note(Pipe *p, int fast) { assert(p && !fast && mutex_depth == 1); }
static void keys_note(const VkPipeState *s) { assert(s->native_rmw_marker); }
static void perf_count(int counter) { assert(counter == PERF_CNT_PIPE_ASYNC && mutex_depth == 1); }
static void prebuild_report(void) { assert(mutex_depth == 1 && !K.left && !K.loading); }

static VkPipeline create_graphics_admitted(const VkPipeState *s, const GKey *k, const Layout *layout, uint32_t fs_off,
                                            VkResult *status, uint32_t marker, RcNativeRmwWord *word,
                                            GraphicsSampledQueries *queries) {
    assert_external();
    assert(A.external == 1 && A.inflight == 1 && !A.closed && s->tag == 917 && k && !layout && !fs_off);
    assert(!marker || marker == 31);
    VkGraphicsPipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    VkPipeline out = VK_NULL_HANDLE;
    startup_compiler_calls++;
    *status = pipe_create_one(1, &ci, &out, VK_NULL_HANDLE);
    if (out && marker)
        *word = (RcNativeRmwWord){.atomic_word = 52141, .execution_model = 5365, .storage_binding = 5, .c0_byte = 0x780};
    if (out) *queries = (GraphicsSampledQueries){.vs = UINT64_C(1) << 2, .fs = UINT64_C(1) << 3};
    return out;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyPipeline(VkDevice device, VkPipeline pipe, const VkAllocationCallbacks *allocator) {
    assert_external();
    assert(!allocator && device == P.device && pipe == expected_retired && pipe);
    retired_count++;
}
static void vk_completion_add_owned(uint64_t scene, void (*done)(void *), void (*cancel)(void *, int), void *owner) {
    assert_external();
    assert(scene == 12 && !completion_owner && done && cancel && owner);
    completion_owner = owner;
    completion_cancel = cancel;
}
static void plat_thread_set_priority(PlatThread *thread, int priority) {
    (void)thread; (void)priority; abort(); // Only queued promotion is exercised in this fixture.
}
#define PLAT_PRIO_NORMAL 0
'''
revision_owner = section("// The immutable info remains", "// ---- graphics pipelines")
revision_owner += section("struct VkGraphicsPipelineWait {", "static int blendable(")
revision_owner += "\n".join(function(name) for name in ("job_push", "promote", "job_wanted",
                                                         "graphics_wait_attach_locked", "graphics_wait_cancel_locked"))
warm_owner = "\n".join(function(name) for name in ("create_graphics", "job_build", "now_ms", "warm_builder"))
test_warm = r'''
static void check_optional_startup_required_seed(void) {
    reset(ROOM, ROOM);
    Pipe boot = {.building = 1};
    Job *job = calloc(1, sizeof *job);
    assert(job);
    *job = (Job){.p = &boot, .low = 1, .prebuild = 1, .worker = -1, .s = {.tag = 917}};
    boot.job = job;
    A.low = A.low_tail = job;
    A.inflight = K.left = 1;
    K.gate_ended = 1; // The first optional refusal already logged its budget snapshot.
    prep_mode = 0;
    warm_builder(&boot);
    assert(A.low == job && A.low_tail == job && A.inflight == 1 && !A.external && K.left == 1);
    assert(boot.job == job && boot.building && !boot.info.pipeline && startup_compiler_calls == 0);
    assert(!P.stats.failures && !K.failed && !K.built && !wait_count && !service_count && K.gate_ended == 2);
    assert_charge(0, 0);

    prep_mode = 1; // The explicit seed uses the same queue and owns a required compiler grant.
    expected_compute = 0;
    native_result = VK_SUCCESS;
    warm_builder(&boot);
    assert(boot.info.pipeline == expected_handle && !boot.building && !boot.job);
    assert(!boot.info.native_rmw_binding && !boot.info.native_rmw.atomic_word);
    assert(boot.full.vs_query_size_lod0_mask == (UINT64_C(1) << 2) &&
           boot.full.fs_query_size_lod0_mask == (UINT64_C(1) << 3));
    assert(!A.low && !A.low_tail && !A.inflight && !A.external && !K.left && K.built == 1);
    assert(startup_compiler_calls == 1 && driver_calls == 1 && P.stats.graphics == 1 && P.stats.async_built == 1);
    assert(!K.failed && !P.stats.failures && !wait_count && !service_count);
    assert_charge(0, ROOM);
    prep_mode = 0;
}

static void check_marker_publication(int success) {
    reset(ROOM, ROOM);
    Pipe selected = {.building = 1, .info = {.native_rmw_binding = 31}};
    Job *job = calloc(1, sizeof *job);
    assert(job);
    *job = (Job){.p = &selected, .worker = -1, .s = {.tag = 917, .native_rmw_marker = 1}};
    selected.job = job;
    A.inflight = A.external = 1;
    expected_compute = 0;
    native_result = success ? VK_SUCCESS : VK_ERROR_OUT_OF_DEVICE_MEMORY;
    plat_mutex_lock(&P.lock);
    job_build(job, 0);
    A.external = 0;
    VkGraphicsPipelineUse use = {0};
    assert(graphics_acquire_locked(&selected, &use) == success);
    if (success) {
        assert(use.info == &selected.info && use.info->native_rmw_binding == 31);
        assert(use.info->native_rmw.c0_byte == 0x780 && use.info->native_rmw.storage_binding == 5);
        assert(use.vs_query_size_lod0_mask == (UINT64_C(1) << 2) &&
               use.fs_query_size_lod0_mask == (UINT64_C(1) << 3));
    } else {
        assert(!selected.info.native_rmw.atomic_word && !selected.full.vs_query_size_lod0_mask);
    }
    plat_mutex_unlock(&P.lock);
    if (success) vk_pipeline_graphics_release(&use);
}

static void check_required_graphics_revision(void) {
    reset(8192, 8192);
    expected_retired = (VkPipeline)(uintptr_t)701;
    VkPipeline optimized = (VkPipeline)(uintptr_t)702;
    VkGraphicsPipelineUse recorded = {0}, duplicate = {0}, parked = {0}, replacement = {0};
    Pipe p = {0};
    plat_mutex_lock(&P.lock);
    const GraphicsSampledQueries fast_queries = {.vs = UINT64_C(1) << 1, .fs = UINT64_C(1) << 4,
        .vertex_uniforms = {.count = 1, .bindings = {{1,0,1}}, .ranges = {{16,4}}}};
    const GraphicsSampledQueries full_queries = {.vs = UINT64_C(1) << 2, .fs = UINT64_C(1) << 5,
        .vertex_uniforms = {.count = 1, .bindings = {{1,0,1}}, .ranges = {{52,4}}}};
    graphics_publish_fast_locked(&p, expected_retired, fast_queries);
    assert(graphics_acquire_locked(&p, &recorded) && graphics_acquire_locked(&p, &duplicate) &&
           graphics_acquire_locked(&p, &parked));
    assert(parked.vs_query_size_lod0_mask == fast_queries.vs && parked.fs_query_size_lod0_mask == fast_queries.fs);
    plat_mutex_unlock(&P.lock);
    vk_pipeline_graphics_record(&recorded, 12);
    vk_pipeline_graphics_record(&duplicate, 12);
    assert(p.fast.users == 2 && !recorded.revision && !duplicate.revision);
    plat_mutex_lock(&P.lock);
    assert(!graphics_publish_full_locked(&p, VK_NULL_HANDLE, full_queries));
    assert(graphics_selected(&p) == expected_retired && p.fast.vs_query_size_lod0_mask == fast_queries.vs &&
           p.fast.fs_query_size_lod0_mask == fast_queries.fs);
    assert(!graphics_publish_full_locked(&p, optimized, full_queries));
    assert(graphics_acquire_locked(&p, &replacement) && replacement.pipeline == optimized);
    assert(parked.pipeline == expected_retired && parked.info == replacement.info);
    assert(parked.vertex_uniforms == &p.fast.vertex_uniforms && replacement.vertex_uniforms == &p.full.vertex_uniforms);
    assert(parked.vertex_uniforms->ranges[0].offset == 16 && replacement.vertex_uniforms->ranges[0].offset == 52);
    assert(parked.vs_query_size_lod0_mask == fast_queries.vs && parked.fs_query_size_lod0_mask == fast_queries.fs &&
           replacement.vs_query_size_lod0_mask == full_queries.vs && replacement.fs_query_size_lod0_mask == full_queries.fs);
    assert(graphics_legacy_locked(&p) == optimized && !p.fast.legacy);
    plat_mutex_unlock(&P.lock);
    vk_pipeline_graphics_release(&parked);
    vk_pipeline_graphics_release(&replacement);
    assert(!retired_count && p.fast.users == 1 && p.fast.pipeline == expected_retired);
    completion_cancel(completion_owner, 1); // Successful native idle after a handed prefix with an abandoned tail.
    assert(retired_count == 1 && !p.fast.pipeline && !p.fast.users && p.full.pipeline == optimized);
    assert(!parked.vs_query_size_lod0_mask && !replacement.vs_query_size_lod0_mask);
    completion_owner = NULL;
}

static void check_required_waiter_promotion(void) {
    reset(8192, 8192);
    Pipe p = {.building = 1};
    Job head = {.worker = -1}, optional = {.p = &p, .worker = -1, .prebuild = 1, .low = 1};
    p.job = &optional;
    VkGraphicsPipelineWait a = {0}, b = {0};
    plat_mutex_lock(&P.lock);
    job_push(&head, 0);
    job_push(&optional, 0);
    graphics_wait_attach_locked(&a, &p);
    graphics_wait_attach_locked(&b, &p);
    graphics_wait_attach_locked(&b, &p);
    assert(p.required_layout && p.required_waiters == 2 && A.head == &optional && A.tail == &head && !A.low);
    graphics_wait_cancel_locked(&a);
    graphics_wait_cancel_locked(&b);
    assert(!p.required_waiters && p.job == &optional && p.building && optional.demanded && A.head == &optional);
    job_unlink(&optional);
    job_unlink(&head);
    plat_mutex_unlock(&P.lock);
}
'''

required_stubs = r'''
static int recorder_probe_contested;
static unsigned cold_layouts;
static Pipe *cached_graphics;
static Pipe cold_graphics;
static VkResult cold_result;

static int plat_mutex_trylock(PlatMutex *lock) {
    if (recorder_probe_contested) return 0;
    plat_mutex_lock(lock);
    return 1;
}
static void plat_cond_wait_ms(PlatCond *cond, PlatMutex *lock, uint32_t ms) {
    (void)cond; (void)lock; (void)ms; abort();
}
static int ensure_init(void) { assert_external(); return P.ready = 1; }
static int workers_start(void) { assert(mutex_depth); return 1; }
static void gkey_fill(const VkPipeState *state, GKey *key) { *key = (GKey){state->tag}; }
static uint64_t key_hash(const void *data, size_t size) {
    assert(size == sizeof(GKey));
    return ((const GKey *)data)->tag;
}
static Pipe *pipe_find(const void *data, size_t size, uint64_t hash) {
    assert(mutex_depth && key_hash(data, size) == hash);
    return hash == 91 ? cached_graphics : NULL;
}
static const char *graphics_shape_unsupported(const VkPipeState *state, const GKey *key) {
    assert_external();
    assert(state->tag == key->tag);
    return key->tag == 92 ? "test unsupported shape" : NULL;
}
static Pipe *pipe_miss(const VkPipeState *state, const GKey *key, uint64_t hash, int async, int *queued, int required) {
    assert(mutex_depth == 1 && !cached_graphics && state->tag == 91 && key->tag == hash && hash == 91);
    assert(!async && required && pipe_budget.ticket.token.ledger && pipe_budget_enter_native());
    cold_layouts++;
    *queued = 0;
    cold_graphics = (Pipe){.graphics_error = cold_result, .graphics_failure = "test layout creation"};
    if (cold_result == VK_SUCCESS)
        assert(!graphics_publish_full_locked(&cold_graphics, expected_handle, (GraphicsSampledQueries){0}));
    return cached_graphics = &cold_graphics;
}
'''
required_owner = section("static VkGraphicsPipelineResult graphics_stopped(", "double vk_pipeline_building_ms(")
test_required = r'''
static void check_typed_graphics_prepare(void) {
    reset(16, 16);
    VkPipeState state = {.tag = 91, .vs = &state};
    VkGraphicsPipelineWait *wait = NULL;
    VkGraphicsPipelineUse use = {0};
    VkGraphicsPipelineFailure failure;
    assert(vk_pipeline_graphics_prepare(&state, 0, &wait, &use, &failure) == VK_GRAPHICS_PENDING);
    assert(!P.ready && !wait && !cold_layouts && !use.revision);

    assert(vk_pipeline_graphics_prepare(&state, 1, &wait, &use, &failure) == VK_GRAPHICS_PENDING);
    assert(P.ready && wait && !wait->pipe && !use.revision && !cold_layouts);
    assert(failure.admission == RC_ADMISSION_REFUSED && failure.peak.amount[RC_BUDGET_RAM] == ROOM * TEST_MB);
    const WakeStep script[] = {{WAKE_GLOBAL, 0}, {WAKE_RAM, 0}, {WAKE_COMMIT, PIPE_BUDGET_RESCAN_NS}};
    wakes = script;
    wake_count = sizeof script / sizeof script[0];
    assert(vk_pipeline_graphics_wait(wait) && wait_count == wake_count && !cold_layouts);
    cold_result = VK_SUCCESS;
    assert(vk_pipeline_graphics_prepare(&state, 1, &wait, &use, &failure) == VK_GRAPHICS_READY);
    assert(use.pipeline == expected_handle && use.revision && !wait && cold_layouts == 1);
    vk_pipeline_graphics_release(&use);
    assert_charge(0, ROOM);

    recorder_probe_contested = 1;
    assert(vk_pipeline_graphics_prepare(&state, 0, &wait, &use, &failure) == VK_GRAPHICS_PENDING);
    assert(!wait && !use.revision);
    recorder_probe_contested = 0;
    assert(vk_pipeline_graphics_prepare(&state, 0, &wait, &use, &failure) == VK_GRAPHICS_READY);
    assert(!wait && use.pipeline == expected_handle && cold_layouts == 1);
    vk_pipeline_graphics_release(&use);

    reset(8192, 8192);
    cached_graphics = NULL;
    state.tag = 92;
    assert(vk_pipeline_graphics_prepare(&state, 1, &wait, &use, &failure) == VK_GRAPHICS_UNSUPPORTED);
    assert(!wait && !use.revision && cold_layouts == 1 && failure.error == VK_ERROR_FEATURE_NOT_PRESENT);
    assert_charge(0, 0);

    state.tag = 91;
    cold_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    assert(vk_pipeline_graphics_prepare(&state, 1, &wait, &use, &failure) == VK_GRAPHICS_TERMINAL);
    assert(!wait && !use.revision && cold_layouts == 2 && failure.error == VK_ERROR_OUT_OF_DEVICE_MEMORY);
    vk_pipeline_stop_preparation();
    assert(vk_pipeline_graphics_prepare(&state, 0, &wait, &use, &failure) == VK_GRAPHICS_STOPPED);
    assert(!wait && !use.revision && cold_layouts == 2);
}
'''

# Reuse the analyzer's valid SPIR-V variants, then exercise the production Vulkan
# create and attestation statements with a controlled native completion result.
def query_owner():
    fixture = (ROOT / "tests/sampled_image_query_check.c").read_text(encoding="utf-8")
    table_start = fixture.index("enum {\n    TABLE_VAR")
    table_end = fixture.index("static void check_table_site(", table_start)
    fixture = fixture[:fixture.index("static void check_site(")] + fixture[table_start:table_end]
    query_type = section("typedef struct {\n    uint64_t candidates, attested;", "static ModuleSampledQueries module_sampled_queries(")
    uniform_stubs = r'''
#include "rc_spirv_uniform_footprint.h"
typedef struct { unsigned unused; } GfxShader;
typedef struct { unsigned set, binding, count; int slot; } GfxShaderRes;
enum { GFX_RES_UBO };
static int gfx_shader_resources(const GfxShader *shader,int kind,const GfxShaderRes **out) {
    assert(shader && kind == GFX_RES_UBO);
    static const GfxShaderRes resource = {.set=0, .binding=1, .count=1, .slot=3};
    *out = &resource;
    return 1;
}
static int gfx_shader_pool_inactive_source(const GfxShader *shader,const uint32_t *words,uint64_t count,
                                          uint32_t set,uint32_t binding_offset) {
    (void)count;(void)set;(void)binding_offset;
    assert(!shader && words);
    return 0;
}
'''
    return "\n".join((fixture, uniform_stubs, query_type, function("module_vertex_uniforms"),
        "#define VK_POOL_TEX 32768", function("compute_pool_query_shape_kinds"), r'''
static void check_final_compute_table(void) {
    Module module = table_module(TABLE_COMPUTE);
    assert(compute_pool_query_shape_kinds(module.words, module.count) == (1u << 2));
    for (size_t i = 5; i < module.count; i += module.words[i] >> 16) {
        uint32_t *op = &module.words[i];
        if (op[0] == (4u << 16 | 71u) && op[1] == TABLE_VAR && op[2] == 33) op[3] = 3;
    }
    assert(!compute_pool_query_shape_kinds(module.words, module.count));
    module = table_module(TABLE_LITERAL);
    assert(!compute_pool_query_shape_kinds(module.words, module.count));
}
static VkResult native_module_status;
static const uint32_t *native_module_words;
static size_t native_module_size;

VKAPI_ATTR VkResult VKAPI_CALL vkCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo *ci,
                                                   const VkAllocationCallbacks *alloc, VkShaderModule *out) {
    assert_external();
    assert(device == P.device && !alloc && ci->sType == VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
    assert(ci->pCode == native_module_words && ci->codeSize == native_module_size);
    if (native_module_status == VK_SUCCESS) *out = (VkShaderModule)(uintptr_t)1;
    return native_module_status;
}

static VkShaderModule check_create_query_module(const uint32_t *w, uint64_t words, ModuleSampledQueries *queries) {
''', section("    if (queries) queries->attested = 0;", "    const uint32_t *orig_w"),
        section("    VkShaderModuleCreateInfo ci =", "    if (native_rmw)"),
        section("    if (queries && r == VK_SUCCESS && m)", "    free(fixed);"), r'''
    return m;
}

static void check_final_query_modules(void) {
    reset(8192, 8192);
    int budget = pipe_budget_begin(ROOM);
    assert(budget == PIPE_BUDGET_OWNED);
    const uint64_t a = UINT64_C(1) << 1, b = UINT64_C(1) << 2;
    Module final = module(NONZERO_LOD, 0, 1, 2);
    native_module_words = final.words;
    native_module_size = final.count * sizeof final.words[0];
    native_module_status = VK_SUCCESS;
    ModuleSampledQueries queries = {.candidates = a | b | (UINT64_C(1) << 63), .stage = RC_SAMPLED_IMAGE_QUERY_VERTEX};
    assert(check_create_query_module(final.words, final.count, &queries) && queries.attested == b);
    final = module(MATCHED, 0, 1, 2);
    native_module_size = final.count * sizeof final.words[0];
    queries.candidates = a;
    assert(check_create_query_module(final.words, final.count, &queries) && queries.attested == a);
    for (unsigned stage = RC_SAMPLED_IMAGE_QUERY_FRAGMENT; stage <= RC_SAMPLED_IMAGE_QUERY_COMPUTE; ++stage) {
        const uint32_t set = stage == RC_SAMPLED_IMAGE_QUERY_FRAGMENT;
        final = stage_module(MATCHED, set, 1, 2, stage);
        native_module_size = final.count * sizeof final.words[0];
        queries = (ModuleSampledQueries){.candidates = a | b, .stage = stage, .set = set};
        assert(check_create_query_module(final.words, final.count, &queries) && queries.attested == (a | b));
        queries.set ^= 1;
        assert(check_create_query_module(final.words, final.count, &queries) && !queries.attested);
    }
    native_module_status = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    queries.attested = a | b;
    assert(!check_create_query_module(final.words, final.count, &queries) && !queries.attested);
    pipe_budget_end(budget);
}
'''))


main = r'''
int main(void) {
    check_exact_wait();
    check_record_availability();
    check_terminal_estimate();
    check_required_stop();
    check_nested_and_settlement();
    check_optional_capacity_and_stop();
    check_stop_during_service();
    check_native_settlement();
    check_drain_graphics();
    check_drain_libraries();
    check_library_prerequisites();
    check_optional_startup_required_seed();
    check_marker_publication(1);
    check_marker_publication(0);
    check_required_graphics_revision();
    check_final_query_modules();
    check_final_compute_table();
    check_required_waiter_promotion();
    check_typed_graphics_prepare();
    return 0;
}
'''


def run():
    cc = Path(os.environ["CC"]).resolve()
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="pipeline-progress-host-") as directory:
        source = Path(directory) / "check.c"
        exe = source.with_suffix(".exe")
        source.write_text("\n".join((base, admission, optional, worker_types, worker_owners, worker_close,
                                      test_admission, native_create, native_stubs, test_workers,
                                      warm_stubs, revision_owner, warm_owner, test_warm, required_stubs,
                                      required_owner, test_required, query_owner(), main)), encoding="utf-8")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", "-I" + str(ROOT / "runtime"),
                        str(source), str(ROOT / "runtime/rc_sampled_image_query.c"),
                        str(ROOT / "runtime/rc_spirv_uniform_footprint.c"), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True, timeout=20)
    print("Pipeline admission, revisions, final stage query proofs, startup and queued shutdown passed")


if __name__ == "__main__":
    run()

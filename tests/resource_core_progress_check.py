"""Exercise the actual parked recorder and native progress dispatcher without Vulkan."""
import os
from pathlib import Path
import subprocess
import tempfile

from pass_end_lock_check import function

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "runtime/vk_sync.c").read_text()
WALKER = (ROOT / "runtime/vk_exec.c").read_text()
names = (
    "vk_sync_try_provider_idle", "vk_sync_end_provider_idle", "submit_lock", "queue_submit",
    "recorder_park", "recorder_resume", "progress_stopped", "split_part", "vk_sync_split", "progress_marker",
    "progress_split", "vk_sync_progress_split", "vk_sync_progress_split_before",
    "progress_capacity_ready", "vk_sync_progress_wait_capacity", "vk_sync_handed_fence",
    "vk_sync_progress_wait_prior", "vk_sync_progress_prepare_pipeline", "vk_sync_progress_wait_upload_staging",
    "wait_executed_signal",
)
prefix = r'''
#include <windows.h>
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "resource_core_budget.h"
typedef SRWLOCK PlatMutex;
typedef CONDITION_VARIABLE PlatCond;
typedef uint64_t VkCommandBuffer;
typedef uint64_t VkSemaphore;
typedef int VkResult;
typedef struct { uint64_t device; } VkCtx;
typedef struct {
    int sType;
    unsigned semaphoreCount;
    VkSemaphore *pSemaphores;
    uint64_t *pValues;
} VkSemaphoreWaitInfo;
typedef struct VkExec {
    int live, recorded, rt_resolved;
    uint64_t fence, dirty;
    VkCommandBuffer cb;
} VkExec;
typedef struct { int sType; VkSemaphore semaphore; uint64_t value, stageMask; } VkSemaphoreSubmitInfo;
typedef struct { VkCommandBuffer commandBuffer; } VkCommandBufferSubmitInfo;
typedef struct { int sType, flags; } VkCommandBufferBeginInfo;
typedef struct {
    int sType;
    unsigned commandBufferInfoCount, signalSemaphoreInfoCount;
    VkCommandBufferSubmitInfo *pCommandBufferInfos;
    VkSemaphoreSubmitInfo *pSignalSemaphoreInfos;
} VkSubmitInfo2;
typedef struct { VkCommandBuffer cb; } CmdSlot;
typedef struct { int sealed; } TextureUseBatch;
typedef struct RcVmaDependency { RcAdmissionVector peak; int selected; } RcVmaDependency;
typedef enum {
    VK_SYNC_PROGRESS_UNAVAILABLE, VK_SYNC_PROGRESS_COMPLETE,
    VK_SYNC_PROGRESS_STOPPED, VK_SYNC_PROGRESS_STOPPED_AFTER_CUT
} VkSyncProgressResult;
enum { VK_SUCCESS, VK_TIMEOUT, VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, VK_NULL_HANDLE = 0 };
enum { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO = 5, VK_STRUCTURE_TYPE_SUBMIT_INFO_2, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT };
enum { PART_CBS = 4, VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    VK_CRUMB_SUBMIT };
static PlatMutex sub_lock = SRWLOCK_INIT;
static PlatCond recorder_resumed = CONDITION_VARIABLE_INIT;
static int recorder_parked;
static _Thread_local uint64_t recording_scene;
static uint64_t issued = 8, done = 5, queued_v = 6, progress_next;
static uint64_t ordinary_parts, covered_ordinary_parts, handed_scene_v;
static uint64_t cur_fence = 9, exec_sig, closed_read;
static struct { uint64_t fence; } recording_results;
static CmdSlot slots[1] = {{20}};
static const int cur_slot = 0;
static TextureUseBatch scene_uses;
enum { HOLD_CAP = 2, HOLD_NS = 500000, PERF_FENCE = 1 };
static int hold_max = HOLD_CAP, held_n;
static uint64_t held_t0, tl_ready_current, tl_held_ready[HOLD_CAP];
static struct {
    int n;
    VkSubmitInfo2 si[HOLD_CAP];
    VkCommandBufferSubmitInfo cbi[HOLD_CAP][PART_CBS + 2];
    VkSemaphoreSubmitInfo sig[HOLD_CAP][2];
    TextureUseBatch *uses[HOLD_CAP];
    uint64_t scene[HOLD_CAP], last;
} H;
static int npart_cbs;
static VkSemaphore progress_tl = 2, exec_tl = 3;
static unsigned splits, markers, prefix_finishes, registration_finishes, waits, services, probes, older_waits;
static unsigned executed_waits;
static unsigned whole_handoffs, bounded_handoffs, results_published_after_loss;
static unsigned pipeline_flushes, pipeline_prepares;
static unsigned staging_waits;
static int staging_available, staging_canceled;
static uint64_t elapsed_ns;
static int stopped, lost, stop_during_wait, lose_on_poll, reusable_after_wait, capacity_after_wait;
static int lose_on_pipeline_flush;
static int stop_during_upload, stop_before_queue, lose_after_queue;
static VkExec *x_cur;
static int rw_chunk, part_start, split_draws, split_at, split_n;
static uint64_t part_t0;
static int recording_failed, fail_drain, fail_pass_end;
static unsigned drains, pass_ends, tail_adoptions;
enum { PW_SPLIT = 11, VK_EXEC_DIRTY_ALL = 15 };
static uint64_t expected_value;
static VkCtx context = {1};
static void plat_mutex_lock(PlatMutex *mutex) { AcquireSRWLockExclusive(mutex); }
static int plat_mutex_trylock(PlatMutex *mutex) { return TryAcquireSRWLockExclusive(mutex); }
static void plat_mutex_unlock(PlatMutex *mutex) { ReleaseSRWLockExclusive(mutex); }
static void plat_cond_wait(PlatCond *cond, PlatMutex *mutex) {
    assert(SleepConditionVariableSRW(cond, mutex, INFINITE, 0));
}
static void plat_cond_broadcast(PlatCond *cond) { WakeAllConditionVariable(cond); }
static int vk_sync_stopping(void) { return stopped; }
static int vk_device_is_lost(void) { return lost; }
static int vk_exec_recording_failed(void) { return recording_failed; }
static uint64_t vk_sync_fence_done(void) { return done; }
static void group_drain(VkExec *ex, int why) {
    assert(ex == x_cur && why == PW_SPLIT && recording_scene == ex->fence && !recorder_parked);
    drains++;
    recording_failed = fail_drain;
}
static void vk_exec_pass_end(VkExec *ex) {
    assert(ex == x_cur && ex->recorded > part_start && !recorder_parked);
    pass_ends++;
    ex->recorded++;
    recording_failed = fail_pass_end;
}
static void vk_exec_draw_new_cb(void) { tail_adoptions++; }
static void vk_device_lost_poll(void) { lost = lose_on_poll; }
static void vk_device_lost_check(VkResult rc, const char *reason) { (void)rc; (void)reason; abort(); }
static void rt_fatal(const char *reason, ...) { (void)reason; abort(); }
static VkCtx *vk_ctx(void) { return &context; }
static uint64_t kern_now_ns(void) { return elapsed_ns; }
static void hold_release(uint64_t value, int aged, int try_only) {
    assert(value == 9 && !aged && !try_only && recorder_parked && !recording_scene);
    assert(plat_mutex_trylock(&sub_lock));
    plat_mutex_unlock(&sub_lock);
    pipeline_flushes++;
    if (lose_on_pipeline_flush) lost = 1;
}
static int vk_staging_prepare_wait(uint64_t upload) {
    assert(upload == 31 && recorder_parked && !recording_scene && issued == 8 && queued_v == 6);
    assert(plat_mutex_trylock(&sub_lock));
    plat_mutex_unlock(&sub_lock);
    staging_waits++;
    if (staging_canceled) stopped = 1;
    return staging_available;
}
static void vk_gpu_passes_part(VkCommandBuffer cb, int end) { assert(cb && (end == 0 || end == 1)); }
static void ts_end(VkCommandBuffer cb, uint64_t scene) { assert(cb && scene == cur_fence); }
static void gfx_tl_note(char kind, uint64_t when, uint64_t part, uint64_t scene) {
    assert((kind == 'X' || kind == 'Y') && !when && part && scene == cur_fence);
}
static int rec_end_cb(VkCommandBuffer cb, int crumb, uint64_t scene) { assert(cb && !crumb && !scene); return 1; }
static void vkEndCommandBuffer(VkCommandBuffer cb) { assert(cb); }
static void vk_upload_flush(int wait) { assert(!wait); if (stop_during_upload) stopped = 1; }
static VkCommandBuffer prologue(VkCtx *c, uint64_t scene) { assert(c == &context && scene == cur_fence); return 0; }
static unsigned part_list(VkCommandBufferSubmitInfo *info, VkCommandBuffer pro, VkCommandBuffer cb, int last) {
    assert(!pro && cb == slots[cur_slot].cb && last);
    info[0].commandBuffer = cb;
    return 1;
}
static void vk_upload_progress(uint64_t *submitted, uint64_t *completed) { *submitted = *completed = 1; }
static TextureUseBatch *vk_texture_uses_seal(uint64_t scene, uint64_t upload) {
    assert(scene == cur_fence && upload == 1 && !scene_uses.sealed);
    scene_uses.sealed = 1;
    return &scene_uses;
}
static void vk_exec_resource_handoff(uint64_t scene) {
    assert(scene == cur_fence && scene_uses.sealed && !closed_read);
    whole_handoffs++;
}
static void vk_exec_resource_handoff_before(uint64_t scene, uint64_t read) {
    assert(scene == cur_fence && scene_uses.sealed && !closed_read);
    closed_read = read;
    bounded_handoffs++;
}
static void ts_submitted(void) {}
static void publish_recording_results(void) {
    assert(!closed_read);
    if (lost) results_published_after_loss++;
}
static void cancel_recording_results(void) { assert(lost && !lose_after_queue); }
static void vk_mem_lazy_handed(void) {}
static void vk_mem_note_submit(uint64_t scene) { assert(scene == cur_fence); }
static CmdSlot *take_slot(VkCtx *c, uint64_t scene) {
    assert(c == &context && scene == cur_fence && !scene_uses.sealed);
    slots[cur_slot].cb = 20 + splits;
    return &slots[cur_slot];
}
static void vkBeginCommandBuffer(VkCommandBuffer cb, VkCommandBufferBeginInfo *begin) { assert(cb && begin->flags); }
static void vk_crumb(VkCommandBuffer cb, int kind, uint64_t scene) {
    assert(cb && kind == VK_CRUMB_SUBMIT && scene == cur_fence);
}
static void ts_begin(VkCommandBuffer cb) { assert(cb); }
static void rec_begin(VkCommandBuffer cb, uint64_t scene) { assert(cb && scene == cur_fence); }
static const char *cfg_get(const char *key) { (void)key; abort(); }
static void rt_log(const char *message, ...) { (void)message; abort(); }
static uint64_t perf_begin(void) { return 1; }
static void perf_end(int kind, uint64_t start) { assert(kind == PERF_FENCE && start == 1); }
static void vk_queue_lock_raw(void) { if (stop_before_queue) stopped = 1; }
static void vk_queue_unlock(void) {}
static void vk_texture_uses_abandon(TextureUseBatch *uses) {
    assert(stopped && uses == &scene_uses && uses->sealed && closed_read == 14);
    uses->sealed = 0;
    closed_read = 0;
}
static VkResult vkWaitSemaphores(uint64_t device, VkSemaphoreWaitInfo *wait, uint64_t ns) {
    if (wait->pSemaphores == &exec_tl) {
        assert(device == 1 && ns <= UINT64_C(100000000) && wait->semaphoreCount == 1);
        executed_waits++;
        elapsed_ns += ns;
        if (*wait->pValues > issued) { if (stop_during_wait) stopped = 1; return VK_TIMEOUT; }
        return VK_SUCCESS;
    }
    assert(device == 1 && ns && recorder_parked && !recording_scene && issued == 8 && queued_v == 6 && done == 5);
    assert(wait->semaphoreCount == 1 && wait->pSemaphores == &progress_tl && *wait->pValues == expected_value);
    if (stop_during_wait) { stopped = 1; return VK_TIMEOUT; }
    return VK_SUCCESS;
}
static int hold_flush_locked(VkCtx *c) {
    assert(c == &context && recording_scene == 9 && !recorder_parked && !H.last);
    assert(H.n == 1 && held_n == 1 && !stopped);
    const VkSubmitInfo2 *submit = H.si;
    if (submit->signalSemaphoreInfoCount) {
        assert(submit->signalSemaphoreInfoCount == 1);
        assert(submit->pSignalSemaphoreInfos->semaphore == progress_tl);
        assert(submit->pSignalSemaphoreInfos->stageMask == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
        expected_value = submit->pSignalSemaphoreInfos->value;
    } else assert(H.scene[0] && !closed_read);
    TextureUseBatch *uses = H.uses[0];
    if (H.scene[0]) {
        assert(H.scene[0] == cur_fence && uses == &scene_uses && uses->sealed);
        assert(submit->commandBufferInfoCount == 1 &&
               submit->pCommandBufferInfos[0].commandBuffer == slots[cur_slot].cb);
        assert(!closed_read || closed_read == 14);
        closed_read = 0;
        uses->sealed = 0;
        splits++;
    } else {
        assert(!submit->commandBufferInfoCount && !uses && !closed_read);
        markers++;
    }
    H.n = held_n = 0;
    if (lose_after_queue) lost = 1;
    return 1;
}
static void vk_exec_draw_update_prefix_completed(VkExec *ex, uint64_t scene) {
    assert(ex->live && scene == 9 && recorder_parked && !recording_scene && !lost && !stopped);
    prefix_finishes++;
}
static void vk_exec_pool_registration_prefix_complete(uint64_t scene) {
    assert(scene == 9 && recorder_parked && prefix_finishes == registration_finishes + 1);
    registration_finishes++;
}
static int fence_wait_impl(uint64_t fence, uint64_t ns) {
    assert(fence == 6 && ns && recorder_parked && !recording_scene);
    older_waits++;
    if (stop_during_wait) { stopped = 1; return 0; }
    return 1;
}
void resource_core_budget_readiness(int required, RcBudgetReadiness *out) {
    assert(required && recorder_parked && !recording_scene);
    *out = (RcBudgetReadiness){.available={{waits && capacity_after_wait ? 12 : 1}}};
}
static int resource_core_vma_dependency_ready(const RcVmaDependency *dependency, const RcBudgetReadiness *now) {
    assert(dependency->selected && dependency->peak.amount[0] == 12);
    return now->available.amount[0] >= dependency->peak.amount[0];
}
void resource_core_budget_service(int required, const RcAdmissionVector *peak, RcBudgetProgress *out) {
    assert(required && recorder_parked && !recording_scene && peak);
    assert(plat_mutex_trylock(&sub_lock)); // the service really runs outside the recorder mutex
    plat_mutex_unlock(&sub_lock);
    assert(peak->amount[0] == 12 || !peak->amount[0]);
    *out = (RcBudgetProgress){.event=++services};
}
void resource_core_budget_wait_since(uint64_t event, uint32_t ms) {
    assert(event == services && ms && recorder_parked);
    waits++;
    if (stop_during_wait) stopped = 1;
}
static int spare_ready(void *context) {
    assert(context == &reusable_after_wait && recorder_parked && !recording_scene);
    probes++;
    return waits && reusable_after_wait;
}
'''

cases = r'''
static HANDLE entrant_started, entrant_entered;
static DWORD WINAPI entrant(void *unused) {
    (void)unused;
    assert(SetEvent(entrant_started));
    submit_lock();
    assert(!recorder_parked && !recording_scene);
    assert(SetEvent(entrant_entered));
    plat_mutex_unlock(&sub_lock);
    return 0;
}
static int prepare_pipeline(void *context) {
    assert(recorder_parked && !recording_scene && issued == 8 && queued_v == 6 && done == 5);
    assert(!vk_sync_try_provider_idle());
    assert(plat_mutex_trylock(&sub_lock));
    plat_mutex_unlock(&sub_lock);
    pipeline_prepares++;
    if (context) {
        HANDLE *other = context;
        *other = CreateThread(NULL, 0, entrant, NULL, 0, NULL);
        assert(*other && WaitForSingleObject(entrant_started, 1000) == WAIT_OBJECT_0);
        assert(WaitForSingleObject(entrant_entered, 10) == WAIT_TIMEOUT);
        return 1;
    }
    stopped = 1;
    return 0;
}
int main(void) {
    assert(vk_sync_handed_fence() == 6);
    assert(vk_sync_progress_wait_prior(6) == VK_SYNC_PROGRESS_UNAVAILABLE);
    assert(vk_sync_progress_prepare_pipeline(prepare_pipeline, NULL) == VK_SYNC_PROGRESS_UNAVAILABLE);
    assert(vk_sync_progress_wait_upload_staging(31) == VK_SYNC_PROGRESS_UNAVAILABLE && !staging_waits);
    plat_mutex_lock(&sub_lock);
    recording_scene = 9;
    assert(vk_sync_progress_wait_prior(9) == VK_SYNC_PROGRESS_UNAVAILABLE);
    assert(vk_sync_progress_wait_prior(8) == VK_SYNC_PROGRESS_UNAVAILABLE);
    assert(!older_waits && vk_sync_progress_wait_prior(6) == VK_SYNC_PROGRESS_COMPLETE && older_waits == 1);
    stop_during_wait = 1;
    assert(vk_sync_progress_wait_prior(6) == VK_SYNC_PROGRESS_STOPPED && older_waits == 2);
    stopped = stop_during_wait = 0;

    VkExec ex = {.live=1};
    VkCommandBuffer cb;
    progress_tl = 0;
    assert(vk_sync_progress_split(&ex, &cb, 0) == VK_SYNC_PROGRESS_UNAVAILABLE && !cb && !splits);
    progress_tl = 2;
    assert(vk_sync_progress_split(&ex, &cb, 0) == VK_SYNC_PROGRESS_COMPLETE && cb == 21);
    assert(prefix_finishes == 1 && registration_finishes == 1 && issued == 8 && done == 5 && queued_v == 6);
    stop_during_wait = 1;
    assert(vk_sync_progress_split(&ex, &cb, 0) == VK_SYNC_PROGRESS_STOPPED_AFTER_CUT && cb == 22);
    assert(prefix_finishes == 1 && registration_finishes == 1);
    assert(vk_sync_progress_split(&ex, &cb, 0) == VK_SYNC_PROGRESS_STOPPED && !cb);
    stopped = stop_during_wait = 0;
    lose_on_poll = 1;
    assert(vk_sync_progress_split(&ex, &cb, 0) == VK_SYNC_PROGRESS_STOPPED_AFTER_CUT && cb == 23);
    assert(lost && prefix_finishes == 1 && registration_finishes == 1);
    lost = lose_on_poll = 0;
    ordinary_parts = 1;
    handed_scene_v = 8;
    assert(vk_sync_progress_split(&ex, &cb, 1) == VK_SYNC_PROGRESS_UNAVAILABLE && !cb && !markers);
    handed_scene_v = 9;
    npart_cbs = 1;
    assert(vk_sync_progress_split(&ex, &cb, 1) == VK_SYNC_PROGRESS_UNAVAILABLE && !cb && !markers);
    npart_cbs = 0;
    assert(vk_sync_progress_split(&ex, &cb, 1) == VK_SYNC_PROGRESS_COMPLETE && !cb && markers == 1);
    assert(splits == 3 && prefix_finishes == 2 && registration_finishes == 2 && covered_ordinary_parts == 1);
    assert(vk_sync_progress_split(&ex, &cb, 1) == VK_SYNC_PROGRESS_UNAVAILABLE && !cb && markers == 1);
    ordinary_parts++;
    stop_during_wait = 1;
    assert(vk_sync_progress_split(&ex, &cb, 1) == VK_SYNC_PROGRESS_STOPPED && !cb && markers == 2);
    assert(covered_ordinary_parts == 1 && prefix_finishes == 2 && registration_finishes == 2);
    stopped = stop_during_wait = 0;

    RcVmaDependency dependency = {.peak={{12}}, .selected=1};
    assert(vk_sync_progress_wait_capacity(NULL, NULL, NULL) == VK_SYNC_PROGRESS_UNAVAILABLE);
    capacity_after_wait = 1;
    assert(vk_sync_progress_wait_capacity(&dependency, NULL, NULL) == VK_SYNC_PROGRESS_COMPLETE);
    assert(waits == 1 && services == 1);
    capacity_after_wait = 0;
    reusable_after_wait = 1;
    waits = services = 0;
    assert(vk_sync_progress_wait_capacity(&dependency, spare_ready, &reusable_after_wait) == VK_SYNC_PROGRESS_COMPLETE);
    assert(waits == 1 && services == 1 && probes >= 3);
    stop_during_wait = 1;
    assert(vk_sync_progress_wait_capacity(&dependency, NULL, NULL) == VK_SYNC_PROGRESS_STOPPED);
    assert(recording_scene == 9 && !recorder_parked && waits == 2 && services == 2);

    assert(vk_sync_progress_prepare_pipeline(prepare_pipeline, NULL) == VK_SYNC_PROGRESS_STOPPED);
    assert(!pipeline_flushes && !pipeline_prepares);
    stopped = 0;
    assert(vk_sync_progress_prepare_pipeline(NULL, NULL) == VK_SYNC_PROGRESS_UNAVAILABLE);
    assert(vk_sync_progress_wait_upload_staging(0) == VK_SYNC_PROGRESS_UNAVAILABLE && !staging_waits);
    assert(vk_sync_progress_wait_upload_staging(31) == VK_SYNC_PROGRESS_UNAVAILABLE && staging_waits == 1);
    staging_available = 1;
    assert(vk_sync_progress_wait_upload_staging(31) == VK_SYNC_PROGRESS_COMPLETE && staging_waits == 2);
    staging_canceled = 1;
    assert(vk_sync_progress_wait_upload_staging(31) == VK_SYNC_PROGRESS_STOPPED && staging_waits == 3);
    assert(stopped && recording_scene == 9 && !recorder_parked && !pipeline_flushes);
    stopped = 0;
    lose_on_pipeline_flush = 1;
    assert(vk_sync_progress_prepare_pipeline(prepare_pipeline, NULL) == VK_SYNC_PROGRESS_STOPPED);
    assert(recording_scene == 9 && !recorder_parked && lost && pipeline_flushes == 1 && !pipeline_prepares);
    lost = lose_on_pipeline_flush = 0;
    assert(vk_sync_progress_prepare_pipeline(prepare_pipeline, NULL) == VK_SYNC_PROGRESS_STOPPED);
    assert(recording_scene == 9 && !recorder_parked && stopped && pipeline_flushes == 2 && pipeline_prepares == 1);
    stopped = 0;
    entrant_started = CreateEventW(NULL, FALSE, FALSE, NULL);
    entrant_entered = CreateEventW(NULL, FALSE, FALSE, NULL);
    HANDLE other = NULL;
    assert(vk_sync_progress_prepare_pipeline(prepare_pipeline, &other) == VK_SYNC_PROGRESS_COMPLETE);
    assert(recording_scene == 9 && !recorder_parked && pipeline_flushes == 3 && pipeline_prepares == 2);
    assert(issued == 8 && queued_v == 6 && done == 5 && splits == 3 && markers == 2);
    assert(WaitForSingleObject(entrant_entered, 10) == WAIT_TIMEOUT); // resume still holds sub_lock
    recording_scene = 0;
    plat_mutex_unlock(&sub_lock);
    assert(WaitForSingleObject(entrant_entered, 1000) == WAIT_OBJECT_0);
    assert(WaitForSingleObject(other, 1000) == WAIT_OBJECT_0);
    CloseHandle(other);
    CloseHandle(entrant_started);
    CloseHandle(entrant_entered);
    assert(vk_sync_try_provider_idle());
    vk_sync_end_provider_idle();
    stopped = 0;
    stop_during_wait = 1;
    assert(wait_executed_signal(&context, 9, UINT64_MAX) == VK_TIMEOUT && executed_waits == 1 && stopped);
    assert(wait_executed_signal(&context, 6, UINT64_MAX) == VK_SUCCESS && executed_waits == 2);
    stopped = stop_during_wait = 0;
    assert(wait_executed_signal(&context, 9, UINT64_C(230000000)) == VK_TIMEOUT && executed_waits == 5);

    plat_mutex_lock(&sub_lock);
    recording_scene = ex.fence = 9;
    x_cur = &ex;
    ex.cb = 99;
    rw_chunk = 1;
    assert(vk_exec_graphics_pool_progress(&ex, 5) == VK_SYNC_PROGRESS_UNAVAILABLE && !drains);
    rw_chunk = 0;
    assert(vk_exec_graphics_pool_progress(&ex, 9) == VK_SYNC_PROGRESS_UNAVAILABLE && !drains);
    assert(vk_exec_graphics_pool_progress(&ex, 8) == VK_SYNC_PROGRESS_UNAVAILABLE && older_waits == 2);
    assert(vk_exec_graphics_pool_progress(&ex, 5) == VK_SYNC_PROGRESS_COMPLETE && older_waits == 2);
    assert(vk_exec_graphics_pool_progress(&ex, 6) == VK_SYNC_PROGRESS_COMPLETE && older_waits == 3);
    assert(!pass_ends && !tail_adoptions && ex.cb == 99);
    fail_drain = 1;
    assert(vk_exec_graphics_pool_progress(&ex, 0) == VK_SYNC_PROGRESS_STOPPED);
    fail_drain = recording_failed = 0;
    assert(vk_exec_graphics_pool_progress(&ex, 0) == VK_SYNC_PROGRESS_UNAVAILABLE && ex.cb == 99);
    assert(!pass_ends && !tail_adoptions && markers == 2 && splits == 3);

    ex.recorded = part_start = 10;
    ordinary_parts = covered_ordinary_parts;
    assert(vk_exec_graphics_pool_progress(&ex, 0) == VK_SYNC_PROGRESS_UNAVAILABLE && markers == 2);
    ordinary_parts++;
    handed_scene_v = 9;
    assert(vk_exec_graphics_pool_progress(&ex, 0) == VK_SYNC_PROGRESS_COMPLETE && markers == 3);
    assert(!pass_ends && !tail_adoptions && ex.cb == 99 && registration_finishes == 3);
    assert(vk_exec_graphics_pool_progress(&ex, 0) == VK_SYNC_PROGRESS_UNAVAILABLE && markers == 3);

    ex.recorded++;
    fail_pass_end = 1;
    assert(vk_exec_graphics_pool_progress(&ex, 0) == VK_SYNC_PROGRESS_STOPPED && splits == 3);
    assert(pass_ends == 1 && registration_finishes == 3 && ex.cb == 99);
    fail_pass_end = recording_failed = 0;
    stop_during_wait = 1;
    assert(vk_exec_graphics_pool_progress(&ex, 0) == VK_SYNC_PROGRESS_STOPPED_AFTER_CUT);
    assert(part_start == ex.recorded && pass_ends == 2 && splits == 4 && tail_adoptions == 1 && ex.cb == 24);
    assert(ex.dirty == VK_EXEC_DIRTY_ALL && !ex.rt_resolved && !split_draws && split_at == INT_MAX && part_t0 == elapsed_ns);
    assert(recording_scene == 9 && !recorder_parked && registration_finishes == 3 && issued == 8 && queued_v == 6);
    stopped = stop_during_wait = 0;
    ex.recorded++;
    assert(vk_exec_graphics_pool_progress(&ex, 0) == VK_SYNC_PROGRESS_COMPLETE);
    assert(part_start == ex.recorded && pass_ends == 3 && splits == 5 && tail_adoptions == 2 && ex.cb == 25);
    assert(recording_scene == 9 && registration_finishes == 4 && issued == 8 && queued_v == 6);

    assert(vk_sync_progress_split_before(&ex, NULL, 14) == VK_SYNC_PROGRESS_UNAVAILABLE);
    assert(vk_sync_progress_split_before(&ex, &cb, 0) == VK_SYNC_PROGRESS_UNAVAILABLE && !cb);
    assert(vk_sync_progress_split_before(&ex, &cb, 13) == VK_SYNC_PROGRESS_UNAVAILABLE && !cb);
    progress_tl = 0;
    assert(vk_sync_progress_split_before(&ex, &cb, 14) == VK_SYNC_PROGRESS_UNAVAILABLE && !cb);
    progress_tl = 2;
    stopped = 1;
    assert(vk_sync_progress_split_before(&ex, &cb, 14) == VK_SYNC_PROGRESS_STOPPED && !cb);
    stopped = 0;
    stop_during_upload = 1;
    assert(vk_sync_progress_split_before(&ex, &cb, 14) == VK_SYNC_PROGRESS_STOPPED && !cb);
    assert(splits == 5 && whole_handoffs == 5 && !bounded_handoffs);
    stopped = stop_during_upload = 0;
    stop_before_queue = 1;
    assert(vk_sync_progress_split_before(&ex, &cb, 14) == VK_SYNC_PROGRESS_STOPPED && !cb);
    assert(splits == 5 && bounded_handoffs == 1 && handed_scene_v == 9 && !scene_uses.sealed && !held_n);
    assert(recording_scene == 9 && !recorder_parked && registration_finishes == 4);
    stopped = stop_before_queue = 0;
    assert(vk_sync_progress_split_before(&ex, &cb, 14) == VK_SYNC_PROGRESS_COMPLETE && cb == 26);
    assert(splits == 6 && bounded_handoffs == 2 && registration_finishes == 5);
    stop_during_wait = 1;
    assert(vk_sync_progress_split_before(&ex, &cb, 14) == VK_SYNC_PROGRESS_STOPPED_AFTER_CUT && cb == 27);
    assert(splits == 7 && bounded_handoffs == 3 && registration_finishes == 5);
    stopped = stop_during_wait = 0;
    lose_on_poll = 1;
    assert(vk_sync_progress_split_before(&ex, &cb, 14) == VK_SYNC_PROGRESS_STOPPED_AFTER_CUT && cb == 28);
    assert(splits == 8 && bounded_handoffs == 4 && registration_finishes == 5 && lost);
    lost = lose_on_poll = 0;
    lose_after_queue = 1;
    assert(vk_sync_progress_split_before(&ex, &cb, 14) == VK_SYNC_PROGRESS_STOPPED_AFTER_CUT && cb == 29);
    assert(splits == 9 && bounded_handoffs == 5 && registration_finishes == 5 && lost);
    assert(results_published_after_loss == 1);
    lost = 0;
    assert(vk_sync_split() == 30);
    assert(lost && splits == 10 && whole_handoffs == 6 && results_published_after_loss == 2);
    assert(recording_scene == 9 && !recorder_parked && issued == 8 && queued_v == 6 && done == 5);
    recording_scene = 0;
    plat_mutex_unlock(&sub_lock);
    puts("recorder progress: ordinary and bounded handoff, native prefix wait, tail adoption, exclusive park and capacity passed");
    return 0;
}
'''

cc = Path(os.environ["CC"])
env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
work = ROOT / "build/test"
work.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="recorder-progress-", dir=work) as temporary:
    fixture = Path(temporary) / "progress.c"
    walker = "\n".join(function(WALKER, name) for name in ("progress_adopt_tail", "vk_exec_graphics_pool_progress"))
    fixture.write_text(prefix + "\n" + "\n".join(function(SOURCE, name) for name in names) + "\n" + walker + "\n" + cases)
    exe = fixture.with_suffix(".exe")
    subprocess.run([str(cc), "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fsanitize-trap=all", "-I", str(ROOT / "runtime"),
                    str(fixture), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

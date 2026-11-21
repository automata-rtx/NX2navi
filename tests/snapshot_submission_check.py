"""Exercise production snapshot admission/stop boundaries without a GPU or game.

Extracts the submission admission, recording ownership transfer and fence wait.
The host fixture controls refusal, stop, queue flushing and completion independently.
"""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def function(text, start):
    begin = text.index(start)
    brace = text.index('{', begin)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[begin:end]


def main():
    text = (ROOT / 'runtime/vk_sync.c').read_text()
    discard = function(text, 'static void discard_readiness(')
    begin = text.index('static uint64_t submit(const')
    end = text.index('    RP_REPLAY_RESET();', begin)
    record = text.index('    int recorded = 1;', end)
    published = text.index('    publish_recording_results();', record)
    submit = text[begin:end] + '''
    (void)c;
    const uint64_t fence = 100;
    CmdSlot *s = &slot;
    tail_entered = own != NULL;
''' + text[record:published] + '''
    recording_results.fence = 0;
    plat_mutex_unlock(&sub_lock);
    return recorded ? fence : 0;
}
'''
    submit_capture = function(text, 'uint64_t vk_sync_submit_capture(')
    submit_bundle = function(text, 'uint64_t vk_sync_submit_bundle(')
    stop = function(text, 'void vk_sync_stop_submissions(void)')
    wait = function(text, 'static int fence_wait_impl(uint64_t value, uint64_t timeout_ns) {')
    c = r'''
#include <assert.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "resource_core_budget.h"
typedef int PlatMutex;
typedef int PlatCond;
typedef int VkResult;
typedef int VkCommandBuffer;
typedef int VkCtx;
typedef struct { VkCommandBuffer cb; uint64_t fence; } CmdSlot;
typedef struct VkUsnapCapture { unsigned id, consumed, canceled; } VkUsnapCapture;
typedef struct VkDrawScratchCredit { unsigned id, consumed, canceled; } VkDrawScratchCredit;
typedef struct {VkUsnapCapture *snapshot; VkDrawScratchCredit *scratch;} VkReplayCaptureBundle;
typedef struct { unsigned discarded; } VkExecReadiness;
typedef enum {
    VK_REPLAY_CAPTURE_WAIT_NONE, VK_REPLAY_CAPTURE_WAIT_BUDGET,
    VK_REPLAY_CAPTURE_WAIT_PRODUCER, VK_REPLAY_CAPTURE_WAIT_DRAIN_OLDER
} VkReplayCaptureWaitKind;
typedef struct {
    RcAdmissionVector peak;
    RcBudgetReadiness before;
    uint64_t producer_epoch;
    VkReplayCaptureWaitKind kind;
} VkReplayCaptureWait;
static CmdSlot slot = {1, 100};
static CmdSlot *slots = &slot;
static int cur_slot, npart_cbs, stopped_recording;
static uint64_t recording_scene;
static struct { uint64_t fence; } recording_results;
enum { VK_SUCCESS=0, VK_NOT_READY=1, VK_EXEC_RECORD_STOPPED=-1 };
#define RP_SCOPE(...)
#define RP_VALUE(...)
#define RP_CALL(id, expression) (expression)
typedef struct { void (*record)(VkCommandBuffer,void *); void *arg; } OwnCommands;
static PlatMutex sub_lock=1, st_lock=2;
static PlatCond cv_done;
static int locked[3], stopping, lost, stop_on_lock, stop_in_service, stop_on_prepared;
static int refused, preflights, flushes, queue_locked, waits, reached_record, tail_entered;
static int producer_refusal, producer_waits;
static VkExecReadiness planned;
static int fixed_blocked, fixed_attempts;
static int notifications, complete_on_wait, leave_unconsumed;
static int record_status, end_calls, ts_drops, result_cancels, texture_cancels, idle_calls;
static int missing_texture;
static jmp_buf expected_fatal;
static int vk_exec_recording_missing_resource(void) { return missing_texture; }
static VkUsnapCapture preflight_owner, first_owner, second_owner;
static VkDrawScratchCredit scratch_owner;
static unsigned recorded_owner, recorded_scratch;
static uint64_t issued=7, done=6, reserved=10, now;
static int32_t submission_stopping;
void vk_sync_stop_submissions(void);
static int vk_sync_stopping(void) { return stopping || submission_stopping; }
static int vk_device_is_lost(void) { return lost; }
static void vk_sync_start(void) {}
static int vk_sync_prepare_draw(void) {
    assert(!locked[1] && !locked[2] && !stopping && !lost);
    fixed_attempts++;
    return !fixed_blocked;
}
static VkCtx *vk_ctx(void) { static VkCtx c;return &c; }
static void plat_mutex_lock(PlatMutex *lock) {
    assert(!locked[*lock]);locked[*lock]=1;
    if(*lock==1 && stop_on_lock) {stop_on_lock=0;stopping=1;}
}
static void plat_mutex_unlock(PlatMutex *lock) {assert(locked[*lock]);locked[*lock]=0;}
static void submit_lock(void) { plat_mutex_lock(&sub_lock); }
static VkResult vk_device_wait_idle(void) { assert(locked[1] && stopped_recording); idle_calls++; return VK_SUCCESS; }
static void vk_device_lost_check(VkResult result,const char *where) { (void)where; assert(result==VK_SUCCESS); }
static void rec_end(void) { assert(locked[1] && stopped_recording); end_calls++; }
static void ts_drop(void) { assert(end_calls==1); ts_drops++; }
static void cancel_recording_results(void) { assert(ts_drops==1 && recording_results.fence); result_cancels++; }
static void vk_texture_uses_cancel(uint64_t fence) { assert(fence==100 && result_cancels==1); texture_cancels++; }
static void plat_cond_broadcast(PlatCond *cv) {(void)cv;notifications++;}
static void resource_core_budget_progress(void) {}
static void vk_usnap_queue_bundle_cancel(VkReplayCaptureBundle *bundle) {
    if (bundle->snapshot) {
        assert(!bundle->snapshot->consumed && !bundle->snapshot->canceled);
        bundle->snapshot->canceled++;
    }
    if (bundle->scratch) {
        assert(!bundle->scratch->consumed && !bundle->scratch->canceled);
        bundle->scratch->canceled++;
    }
    *bundle = (VkReplayCaptureBundle){0};
}
static void vk_exec_readiness_discard(VkExecReadiness *readiness) {
    if (!readiness) return;
    assert(readiness == &planned && !planned.discarded);
    planned.discarded++;
}
static VkResult vk_exec_snapshot_preflight(const uint64_t *h, int n, VkUsnapCapture **capture,
                                           VkExecReadiness **readiness, VkReplayCaptureWait *wait) {
    (void)h;
    (void)n;
    assert(locked[1] && !*capture && !*readiness);
    preflights++;
    *wait = (VkReplayCaptureWait){0};
    if (refused) {
        refused--;
        if (producer_refusal) {
            wait->kind = VK_REPLAY_CAPTURE_WAIT_PRODUCER;
            wait->producer_epoch = 21;
        } else {
            wait->kind = VK_REPLAY_CAPTURE_WAIT_BUDGET;
            wait->peak = (RcAdmissionVector){{4}};
        }
        return VK_NOT_READY;
    }
    *readiness = &planned;
    *capture = &preflight_owner;
    if (stop_on_prepared) stopping = 1;
    return VK_SUCCESS;
}
static int vk_exec_record(VkCommandBuffer cb, const uint64_t *h, int n, uint64_t fence,
                          VkReplayCaptureBundle *bundle, const VkExecReadiness *readiness) {
    (void)h;
    (void)n;
    assert(cb == slot.cb && locked[1] && recording_results.fence == fence && recording_scene == fence);
    VkUsnapCapture *capture = bundle->snapshot;
    assert((capture == &preflight_owner) == (readiness == &planned));
    assert(!readiness || !readiness->discarded);
    VkDrawScratchCredit *credit = bundle->scratch;
    assert(!capture || (!capture->consumed && !capture->canceled));
    assert(!credit || (!credit->consumed && !credit->canceled));
    recorded_owner = capture ? capture->id : 0;
    recorded_scratch = credit ? credit->id : 0;
    reached_record++;
    if (leave_unconsumed) return 0;
    if (capture) capture->consumed++;
    if (credit) credit->consumed++;
    *bundle = (VkReplayCaptureBundle){0};
    return record_status;
}
static void vk_usnap_wait_producer(uint64_t epoch, uint32_t milliseconds) {
    assert(epoch == 21 && milliseconds && !locked[1] && !queue_locked && flushes);
    producer_waits++;
}
static void vk_queue_lock(void) {assert(!locked[1]);queue_locked=1;flushes++;}
static void vk_queue_unlock(void) {assert(queue_locked);queue_locked=0;}
void resource_core_budget_service(int required,const RcAdmissionVector *peak,RcBudgetProgress *out) {
    assert(required && peak->amount[0]==4 && !locked[1] && !queue_locked && flushes);
    *out=(RcBudgetProgress){.event=1,.blocked_mask=1};
    if(stop_in_service)vk_sync_stop_submissions();
}
void resource_core_budget_wait_since(uint64_t event,uint32_t ms) {
    assert(!locked[1] && event==1 && ms);waits++;
}
static void rt_fatal(const char *fmt,...) {
    (void)fmt;
    if (missing_texture) {
        assert(!locked[1] && stopped_recording && end_calls == 1 && result_cancels == 1 && texture_cancels == 1);
        assert(!slot.fence && !npart_cbs);
        longjmp(expected_fatal, 1);
    }
    abort();
}
static void shutdown_present_tail(VkCommandBuffer cb,void *arg) {(void)cb;(void)arg;}
static void hold_release(uint64_t v,int a,int b) {(void)v;(void)a;(void)b;}
static uint64_t kern_now_ns(void) {return ++now;}
static void plat_cond_wait(PlatCond *cv,PlatMutex *lock) {
    (void)cv;assert(locked[*lock] && complete_on_wait);waits++;done=issued;
}
static void plat_cond_wait_ns(PlatCond *cv,PlatMutex *lock,uint64_t ns) {
    assert(ns);plat_cond_wait(cv,lock);
}
'''.replace('static void resource_core_budget_progress', 'void resource_core_budget_progress')
    c += '\n'.join((stop, discard, submit, submit_bundle, submit_capture, wait))
    c += r'''
static void reset(void) {
    assert(!locked[1] && !locked[2] && !recording_results.fence && !recording_scene);
    stopping = lost = stop_on_lock = stop_in_service = stop_on_prepared = refused = 0;
    submission_stopping = preflights = flushes = waits = reached_record = tail_entered = 0;
    notifications = complete_on_wait = leave_unconsumed = 0;
    stopped_recording = end_calls = ts_drops = result_cancels = texture_cancels = idle_calls = 0;
    record_status = 1;
    slot.fence = 100;
    issued = 7; done = 6; reserved = 10;
    preflight_owner = (VkUsnapCapture){.id = 1};
    first_owner = (VkUsnapCapture){.id = 2};
    second_owner = (VkUsnapCapture){.id = 3};
    scratch_owner = (VkDrawScratchCredit){.id = 4};
    recorded_owner = recorded_scratch = 0;
    producer_refusal = producer_waits = 0;
    planned = (VkExecReadiness){0};
}
int main(void) {
    uint64_t reused_handle = 42;
    reset();
    refused = 1;
    assert(vk_sync_submit_capture(NULL, 0, NULL) == 100);
    assert(preflights == 2 && flushes == 1 && waits == 1 && reached_record == 1);
    assert(recorded_owner == preflight_owner.id && preflight_owner.consumed == 1 && !preflight_owner.canceled);
    assert(planned.discarded == 1);

    reset();
    refused = producer_refusal = 1;
    assert(vk_sync_submit_capture(NULL, 0, NULL) == 100);
    assert(preflights == 2 && flushes == 1 && producer_waits == 1 && !waits && reached_record == 1);
    assert(planned.discarded == 1);

    reset();
    record_status = VK_EXEC_RECORD_STOPPED;
    npart_cbs = 2;
    assert(vk_sync_submit_capture(NULL, 0, NULL) == 7 && planned.discarded == 1);
    assert(preflights == 1 && reached_record == 1 && result_cancels == 1 && !npart_cbs);
    assert(vk_sync_submit_capture(NULL, 0, NULL) == 7 && reached_record == 1 && planned.discarded == 1);

    reset();
    stop_on_lock = 1;
    assert(vk_sync_submit_capture(&reused_handle, 1, &first_owner) == 7);
    assert(first_owner.canceled == 1 && !second_owner.canceled && !preflights && !reached_record);

    reset();
    lost = 1;
    assert(vk_sync_submit_capture(&reused_handle, 1, &first_owner) == 7);
    assert(first_owner.canceled == 1 && !preflights && !reached_record);

    reset();
    lost = 1;
    VkReplayCaptureBundle bundle = {&first_owner, &scratch_owner};
    assert(vk_sync_submit_bundle(&reused_handle, 1, bundle) == 7);
    assert(first_owner.canceled == 1 && scratch_owner.canceled == 1 && !reached_record);

    reset();
    assert(vk_sync_submit_bundle(&reused_handle, 1, (VkReplayCaptureBundle){0}) == 100);
    assert(!preflights && reached_record == 1 && !recorded_owner && !recorded_scratch);
    assert(vk_sync_submit_bundle(&reused_handle, 1, bundle) == 100);
    assert(first_owner.consumed == 1 && scratch_owner.consumed == 1 && recorded_scratch == scratch_owner.id);

    reset();
    assert(vk_sync_submit_capture(&reused_handle, 1, &first_owner) == 100);
    assert(recorded_owner == first_owner.id && first_owner.consumed == 1 && !first_owner.canceled);
    assert(vk_sync_submit_capture(&reused_handle, 1, &second_owner) == 100);
    assert(recorded_owner == second_owner.id && second_owner.consumed == 1 && !second_owner.canceled);
    assert(!preflights && reached_record == 2);

    reset();
    fixed_blocked = 1;
    fixed_attempts = 0;
    assert(vk_sync_submit_capture(&reused_handle, 1, &first_owner) == 7);
    assert(fixed_attempts == 1 && first_owner.canceled == 1 && !first_owner.consumed && !preflights && !reached_record);
    fixed_blocked = 0;

    reset();
    leave_unconsumed = 1;
    assert(vk_sync_submit_capture(&reused_handle, 1, &first_owner) == 0);
    assert(recorded_owner == first_owner.id && first_owner.canceled == 1 && !first_owner.consumed && !preflights);

    reset();
    record_status = VK_EXEC_RECORD_STOPPED;
    npart_cbs = 4;
    assert(vk_sync_submit_capture(&reused_handle, 1, &first_owner) == 7);
    assert(first_owner.consumed && stopped_recording && submission_stopping && end_calls == 1 && ts_drops == 1);
    assert(result_cancels == 1 && texture_cancels == 1 && !slot.fence && !npart_cbs);
    assert(submit(NULL, 0, NULL, NULL, 1) == 7 && idle_calls == 1 && !tail_entered);

    reset();
    record_status = VK_EXEC_RECORD_STOPPED;
    missing_texture = 1;
    if (!setjmp(expected_fatal)) {
        vk_sync_submit_capture(&reused_handle, 1, &first_owner);
        assert(0);
    }
    missing_texture = 0;

    reset();
    refused = stop_in_service = 1;
    assert(vk_sync_submit_capture(NULL, 0, NULL) == 7);
    assert(preflights == 1 && notifications && !reached_record);

    reset();
    stop_on_prepared = 1;
    assert(vk_sync_submit_capture(NULL, 0, NULL) == 7);
    assert(preflight_owner.canceled == 1 && !reached_record && planned.discarded == 1);

    reset();
    stopping = 1;
    fixed_attempts = 0;
    assert(submit(NULL, 0, NULL, NULL, 1) == 100 && tail_entered && !preflights && !reached_record);
    assert(!fixed_attempts);

    reset();
    refused = stop_in_service = 1;
    assert(submit(NULL, 0, NULL, NULL, 1) == 100 && tail_entered && preflights == 1 && !reached_record);

    reset();
    stop_on_prepared = 1;
    assert(submit(NULL, 0, NULL, NULL, 1) == 100 && tail_entered && preflight_owner.canceled == 1);
    assert(planned.discarded == 1);

    reset();
    vk_sync_stop_submissions();
    assert(!fence_wait_impl(9, UINT64_MAX) && done == 6 && !waits);
    complete_on_wait = 1;
    assert(fence_wait_impl(7, UINT64_MAX) && done == 7 && waits == 1);
    puts("snapshot submission: pressure, explicit ownership transfer, stop, presentation tail and fence waits passed");
}
'''
    cc = Path(os.environ.get('CC', r'C:\msys64\ucrt64\bin\clang.exe'))
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get('PATH', ''))
    work = ROOT / 'build' / 'test'
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='snapshot-submission-', dir=work) as directory:
        source = Path(directory) / 'check.c'
        executable = Path(directory) / 'check.exe'
        source.write_text(c)
        subprocess.run([str(cc), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror',
                        '-fsanitize=undefined', '-fsanitize-trap=undefined',
                        '-I' + str(ROOT / 'runtime'), str(source), '-o', str(executable)], env=env, check=True)
        subprocess.run([str(executable)], env=env, check=True)


if __name__ == '__main__':
    main()

"""Exercise production completion-owner disposal without a Vulkan device or game files."""
from pathlib import Path
import os
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "runtime/vk_sync.c").read_text()


def function(name):
    match = re.search(r"^(?:static )?(?:void|int) " + name + r"\([^;]*?\) \{", source, re.M)
    assert match, name
    end, depth = match.end(), 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


fixture = r'''
#include <assert.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct { uint64_t fence; void (*fn)(void *); void *arg; void (*cancel)(void *, int); } Pending;
static int st_lock, res_lock, cv_work;
static uint64_t done;
static Pending *queue, *rqueue;
static int qhead, qn, qcap, rqhead, rqn, rqcap;
enum { RESULT_BATCH = 64 };
static _Thread_local struct { uint64_t fence; int n, cap; Pending *items; } recording_results;
static uint64_t exw_value;
static int cb_late, lost;
static int vk_device_is_lost(void) { return lost; }
static void plat_mutex_lock(int *p) { assert(!*p); *p = 1; }
static void plat_mutex_unlock(int *p) { assert(*p); *p = 0; }
static void plat_cond_broadcast(int *p) { (void)p; assert(st_lock); }
static jmp_buf fatal;
static int expect_fatal;
static void rt_fatal(const char *s, ...) { (void)s; if (expect_fatal) longjmp(fatal, 1); abort(); }

typedef int VkResult;
typedef struct { int gfx_queue; } VkCtx;
typedef struct { int handed, abandoned; } TextureUseBatch;
enum { VK_SUCCESS, VK_ERROR_DEVICE_LOST = -4, VK_NULL_HANDLE = 0 };
static void vk_texture_uses_handed(TextureUseBatch *batch) { if (batch) batch->handed++; }
static void vk_texture_uses_abandon(TextureUseBatch *batch) { if (batch) batch->abandoned++; }
static void vk_device_lost_check(VkResult result, const char *where) { (void)where; lost = result == VK_ERROR_DEVICE_LOST; }
static VkResult driver_result;
static unsigned driver_calls;
static VkResult vkQueueSubmit2(int queue_handle, uint32_t count, int *submits, int fence) {
    (void)queue_handle; (void)submits; (void)fence; assert(count); driver_calls++; return driver_result;
}
static int loss_injected, held_n;
static int lost_at_due(void) { return loss_injected; }
static int tl_recording(void) { return 0; }
static uint64_t tl_batch_next, tl_held_ready[2], queued_v, handed_scene_v, uncertain_scene_v;
static uint64_t census_final, census_part, census_uncertain;
static void rc_registry_runtime_native_handoff(uint64_t final, uint64_t part, uint64_t uncertain) {
    census_final = final; census_part = part; census_uncertain = uncertain;
}
static uint64_t plat_now_ns(void) { return 0; }
static void gfx_tl_note(char kind, uint64_t ns, uint64_t a, uint64_t b) { (void)kind; (void)ns; (void)a; (void)b; }
static struct { int n, si[2]; TextureUseBatch *uses[2]; uint64_t scene[2], last, batches, subs; } H;
'''
for name in ("pending_push", "vk_completion_add_owned", "vk_completion_add", "pending_disposable",
             "pending_dispose", "pending_dispose_queue", "pending_dispose_remaining", "publish_recording_results",
             "cancel_recording_results", "vk_completion_add_result_owned", "results_run", "hold_flush_locked"):
    fixture += function(name) + "\n"
fixture += r'''
typedef struct { int completed, canceled, may, enqueue; } Owner;
static Owner nested;
static void complete(void *p) { assert(!st_lock); ((Owner *)p)->completed++; }
static void complete_then_lose(void *p) { assert(cb_late); complete(p); lost = 1; }
static void cancel(void *p, int may) {
    assert(!st_lock); Owner *owner = p;
    owner->canceled++; owner->may = may;
    if (owner->enqueue) vk_completion_add_owned(99, complete, cancel, &nested);
}
static void handover_check(void) {
    static VkCtx ctx;
    static TextureUseBatch split, failed, never;
    static Owner in_flight, unhanded;
    H.n = held_n = 1; H.scene[0] = 9; H.uses[0] = &split;
    assert(hold_flush_locked(&ctx)); // A real split may reach the queue without signaling anything.
    assert(driver_calls == 1 && handed_scene_v == 9 && !queued_v && split.handed == 1 && !H.n);
    assert(!census_final && census_part == 9 && !census_uncertain);
    vk_completion_add_owned(9, complete, cancel, &in_flight);
    vk_completion_add_owned(10, complete, cancel, &unhanded);
    H.n = held_n = 1; H.last = H.scene[0] = 9; H.uses[0] = &failed;
    driver_result = VK_ERROR_DEVICE_LOST;
    expect_fatal = 1;
    if (!setjmp(fatal)) { hold_flush_locked(&ctx); abort(); }
    expect_fatal = 0;
    assert(lost && driver_calls == 2 && !H.n && !queued_v && uncertain_scene_v == 9);
    assert(!failed.abandoned && !failed.handed);
    pending_dispose_remaining(8, uncertain_scene_v, 0);
    assert(!in_flight.completed && !in_flight.canceled && unhanded.canceled == 1 && !unhanded.may);
    pending_dispose_remaining(8, uncertain_scene_v, 1);
    assert(!in_flight.completed && in_flight.canceled == 1 && in_flight.may);
    // Injected loss is ordered before the real queue API. This batch can be abandoned on loss.
    lost = 0; loss_injected = 1;
    H.n = held_n = 1; H.last = H.scene[0] = 10; H.uses[0] = &never;
    expect_fatal = 1;
    if (!setjmp(fatal)) { hold_flush_locked(&ctx); abort(); }
    expect_fatal = 0;
    assert(lost && driver_calls == 2 && H.n == 1 && uncertain_scene_v == 9);
    assert(!hold_flush_locked(&ctx));
    assert(!H.n && never.abandoned == 1 && !never.handed);
}
int main(void) {
    Owner finished = {0}, handed = {0}, never = {.enqueue = 1};
    vk_completion_add_owned(1, complete, cancel, &finished);
    vk_completion_add_owned(2, complete, cancel, &handed);
    vk_completion_add_owned(3, complete, cancel, &never);
    // An undrained device loss releases only never-handed owners, including a callback that adds another owner.
    pending_dispose_remaining(1, 2, 0);
    assert(finished.completed == 1 && !finished.canceled);
    assert(!handed.completed && !handed.canceled && qn == 1);
    assert(never.canceled == 1 && !never.may && !never.completed);
    assert(nested.canceled == 1 && !nested.may);
    pending_dispose_remaining(1, 2, 0);
    assert(qn == 1 && !handed.canceled);
    // A real idle proof releases possible GPU users without claiming successful command execution.
    pending_dispose_remaining(1, 2, 1);
    assert(handed.canceled == 1 && handed.may && !handed.completed && !qn);
    pending_dispose_remaining(1, 2, 1);
    assert(handed.canceled == 1 && finished.completed == 1);
    Owner a = {0}, b = {0}, legacy = {0}, result = {0};
    vk_completion_add_owned(4, complete, cancel, &a);
    vk_completion_add_owned(5, complete, cancel, &b);
    vk_completion_add(6, complete, &legacy);
    recording_results.fence = 4;
    vk_completion_add_result_owned(4, complete, cancel, &result);
    assert(recording_results.n == 1 && !rqn);
    publish_recording_results();
    recording_results.fence = 0;
    pending_dispose_remaining(5, 5, 1);
    assert(a.completed == 1 && b.completed == 1 && result.completed == 1);
    assert(!a.canceled && !b.canceled && !legacy.completed && !qn && !rqn);
    Owner ran = {0}, skipped = {0}, future = {0};
    vk_completion_add_result_owned(7, complete_then_lose, cancel, &ran);
    vk_completion_add_result_owned(7, complete, cancel, &skipped);
    vk_completion_add_result_owned(8, complete, cancel, &future);
    exw_value = 7;
    results_run(7);
    assert(ran.completed == 1 && !ran.canceled && !cb_late);
    assert(!skipped.completed && skipped.canceled == 1 && skipped.may);
    assert(!future.completed && !future.canceled && rqn == 1);
    pending_dispose_remaining(7, 7, 0);
    assert(future.canceled == 1 && !future.may && !rqn);
    lost = 0;
    Owner discarded[RESULT_BATCH + 1] = {0}, prefix_result = {0};
    recording_results.fence = 8;
    vk_completion_add_result_owned(8, complete, cancel, &prefix_result);
    publish_recording_results();
    for (unsigned i = 0; i < sizeof discarded / sizeof *discarded; ++i)
        vk_completion_add_result_owned(8, complete, cancel, &discarded[i]);
    assert(recording_results.n == RESULT_BATCH + 1 && rqn == 1);
    cancel_recording_results();
    recording_results.fence = 0;
    assert(rqn == 1 && !recording_results.items);
    for (unsigned i = 0; i < sizeof discarded / sizeof *discarded; ++i)
        assert(discarded[i].canceled == 1 && !discarded[i].completed && !discarded[i].may);
    pending_dispose_remaining(7, 8, 0);
    assert(!prefix_result.canceled && !prefix_result.completed && rqn == 1);
    pending_dispose_remaining(7, 8, 1);
    assert(prefix_result.canceled && prefix_result.may && !rqn);
    handover_check();
    assert(!st_lock && !res_lock);
    free(queue); free(rqueue);
    puts("resource core completion ownership: drained, split, uncertain handover and result loss passed");
    return 0;
}
'''

cc = Path(os.environ.get("CC", r"C:\Users\Andrew\AppData\Local\Programs\2NX2Navi\toolchain\msys64\ucrt64\bin\clang.exe"))
work = (ROOT / "build/test").resolve()
work.relative_to(ROOT.resolve())
work.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="completion-owner-", dir=work) as directory:
    path = Path(directory)
    test = path / "completion.c"
    test.write_text(fixture)
    exe = path / "completion.exe"
    subprocess.run([str(cc), "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fsanitize-trap=all", str(test), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

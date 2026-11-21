#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "host.h"
#include "vk_exec.h"
#include "vk_mesh_fb.h"

#define handle(type, v) ((type)(uintptr_t)(v))
#define RP_CALL(region, call) (call)
#define RP_SCOPE(scope, tag)
enum { PW_SPLIT, PW_END, RB_SUBMIT };
static struct { int on; } RR;
static void *rw_chunk;
static int split_n, split_draws, split_at, part_start;
static uint64_t part_t0;
static char trace[100];
static unsigned ntrace, nlogs;
static char last_log[512];
static int copy_commands, stop_code, lose_code, fail_in_pass, stop_in_pass, stop_in_copy;
static int expect_begun_tail, stop_on_read;
static VkSyncProgressResult prefix_status;
static VkMeshFbCapacityResult capacity_status;
static VkExec *active;
static const uint64_t scene_id = UINT64_C(0x1200000003);

static void step(char c) {
    assert(ntrace < sizeof trace - 1);
    trace[ntrace++] = c;
    trace[ntrace] = 0;
}
static void group_drain(VkExec *ex, int why);
static void progress_adopt_tail(VkExec *ex, VkCommandBuffer cb);
static void resource_targets_forget(VkExec *ex);
static void runs_break(int why);
static void views_release(VkExec *ex, uint64_t fence);

#include "functions.inc"

static _Thread_local const VkExecReadiness *record_readiness;
static int finish_recording(VkExec *ex, int stopped, int synced, VkTextureRegistrationScope *local_registration) {
    uint64_t fence = ex->fence;
    const VkExecReadiness *readiness = record_readiness;
#include "replay_tail.inc"
    return 1; // The native final scene is handed by vk_sync after vk_exec_record returns healthy.
}

volatile int32_t vk_lost;
int vk_sync_stopping(void) { return stop_code; }
uint64_t kern_now_ns(void) { return 1000; }
int vk_exec_pool_split(uint64_t fence) { assert(fence == scene_id); return 1; }
int vk_mem_mirror_wb_pending(void) { return 1; }
int vk_sync_wb_split(void) { return 1; }
uint64_t vk_sync_handed_fence(void) { abort(); }
uint64_t vk_sync_fence_done(void) { abort(); }
VkSyncProgressResult vk_sync_progress_wait_prior(uint64_t v) { (void)v; abort(); }
void vk_exec_pool_marks(VkExec *ex) { assert(ex == active); step('P'); }
void vk_exec_draw_new_cb(void) { step('T'); }
void vk_exec_draw_prepared_stats_flush(VkExec *ex) { assert(ex == active); step('S'); }
void vk_exec_draw_update_history_end(uint64_t fence) { assert(x_cur == active && fence == scene_id); step('I'); }
void vk_exec_event_sync(int begin) { assert(!begin && !x_cur); step('E'); }
void vk_usnap_end(uint64_t fence) { assert(fence == scene_id); step('U'); }
int vk_mem_mirror_end(VkCommandBuffer cb, uint64_t fence) {
    assert(fence == scene_id && cb == handle(VkCommandBuffer, 3));
    step('W');
    return 1;
}
void vk_texture_registration_record_end(VkTextureRegistrationScope *scope, int resumed) {
    assert(scope == (VkTextureRegistrationScope *)(uintptr_t)1 && !resumed && !x_cur);
    step('D');
}
void vk_texture_registration_scope_end(VkTextureRegistrationScope *scope) {
    assert(scope == (VkTextureRegistrationScope *)(uintptr_t)1 && !x_cur);
    step('O');
}
void vk_exec_barrier(VkExec *ex) { assert(ex == active); step('B'); ex->recorded++; }
void vk_exec_pass_end(VkExec *ex) {
    assert(ex == active);
    if (!ex->in_pass) return;
    assert(x_cur == ex);
    ex->in_pass = 0;
    ex->recorded++;
    step('X');
    if (fail_in_pass) cancel_recording();
    if (stop_in_pass) stop_code = 1;
    if (lose_code) vk_lost = 1;
}
VkSyncProgressResult vk_sync_progress_split(VkExec *ex, VkCommandBuffer *resume, int empty_tail) {
    assert(ex == active && x_cur == ex && !ex->in_pass && !empty_tail && ex->cb == handle(VkCommandBuffer, 1));
    assert(ex->recorded == part_start + copy_commands + expect_begun_tail);
    step('G');
    if (prefix_status == VK_SYNC_PROGRESS_COMPLETE || prefix_status == VK_SYNC_PROGRESS_STOPPED_AFTER_CUT)
        *resume = handle(VkCommandBuffer, 2);
    return prefix_status;
}
VkCommandBuffer vk_sync_split_exec(void) {
    assert(!x_cur && !vk_exec_recording_failed() && !stop_code && !vk_device_is_lost());
    if (copy_commands == 3) assert(active->cb == handle(VkCommandBuffer, 2));
    step('F');
    return handle(VkCommandBuffer, 3);
}
int vk_meshfb_capacity_copy(VkCommandBuffer cb, uint64_t scene) {
    assert(cb == handle(VkCommandBuffer, 1) && scene == scene_id && x_cur == active && !active->in_pass);
    step('C');
    if (stop_in_copy) stop_code = 1;
    return stop_code || vk_device_is_lost() ? -1 : copy_commands;
}
VkMeshFbCapacityResult vk_meshfb_capacity_read(uint64_t scene, VkMeshFbCapacity *out) {
    assert(scene == scene_id && x_cur == active && active->cb == handle(VkCommandBuffer, 2) &&
           prefix_status == VK_SYNC_PROGRESS_COMPLETE && part_start == active->recorded);
    assert(active->dirty == VK_EXEC_DIRTY_ALL && !active->rt_resolved && !split_draws && split_at == INT_MAX && part_t0 == 1000);
    step('R');
    *out = (VkMeshFbCapacity){.scene = scene, .ordinal = 615, .requested_outer = UINT32_MAX,
           .requested_mesh = UINT64_C(0x100000009), .available = 65535, .detail = 9,
           .limit = VK_MESHFB_CAPACITY_LIMIT_EXPANSION, .error = VK_ERROR_MEMORY_MAP_FAILED};
    if (stop_on_read) stop_code = 1;
    return capacity_status;
}
static void group_drain(VkExec *ex, int why) {
    assert(x_cur == ex && ex == active);
    step(why == PW_END ? 'Q' : 'q');
}
static void resource_targets_forget(VkExec *ex) { assert(ex == active && x_cur == ex); step('K'); }
static void runs_break(int why) { assert(why == RB_SUBMIT); step('L'); }
static void views_release(VkExec *ex, uint64_t fence) { assert(ex == active && fence == scene_id); step('V'); }
void rt_log(const char *message, ...) {
    va_list args;
    va_start(args, message);
    vsnprintf(last_log, sizeof last_log, message, args);
    va_end(args);
    nlogs++;
}
void rt_fatal(const char *message, ...) { fprintf(stderr, "%s\n", message); abort(); }

static VkExec exec;
static void reset(int copied, VkSyncProgressResult progress, VkMeshFbCapacityResult result) {
    ntrace = nlogs = 0;
    trace[0] = last_log[0] = 0;
    stop_code = lose_code = vk_lost = fail_in_pass = stop_in_pass = stop_in_copy = stop_on_read = recording_failed = 0;
    probe_pass_context = NULL;
    probe_pass_record = NULL;
    expect_begun_tail = 1;
    part_start = 12;
    copy_commands = copied;
    prefix_status = progress;
    capacity_status = result;
    exec = (VkExec){.cb = handle(VkCommandBuffer, 1), .fence = scene_id, .recorded = part_start,
                    .in_pass = 1, .rt_resolved = 1, .registration_scope = (VkTextureRegistrationScope *)(uintptr_t)1};
    x_cur = active = &exec;
}
static int finish(int stopped) {
    int result = finish_recording(&exec, stopped, 1, exec.registration_scope);
    assert(!x_cur && !exec.cb && !exec.registration_scope);
    return result;
}
static void hook(VkExec *ex, void *opaque) {
    assert(ex == active && opaque == &exec);
    step('H');
    cancel_recording();
}
static void checked(const char *expected) {
    if (strcmp(trace, expected))
        fprintf(stderr, "trace %s expected %s\n", trace, expected);
    assert(!strcmp(trace, expected));
}

static void stop_case(void) {
    reset(3, VK_SYNC_PROGRESS_COMPLETE, VK_MESHFB_CAPACITY_READY);
    stop_in_pass = 1;
    assert(finish(0) == VK_EXEC_RECORD_STOPPED);
    assert(recording_failed == RECORDING_CANCELLED && !nlogs);
    checked("QXKSIDOEUV");
}
static void pass_cancel_case(void) {
    reset(0, VK_SYNC_PROGRESS_COMPLETE, VK_MESHFB_CAPACITY_READY);
    fail_in_pass = 1;
    assert(finish(0) == VK_EXEC_RECORD_STOPPED && recording_failed == RECORDING_CANCELLED && !nlogs);
}
int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "stop")) { stop_case(); return 0; }
    if (argc > 1 && !strcmp(argv[1], "pass")) { pass_cancel_case(); return 0; }
    reset(0, VK_SYNC_PROGRESS_UNAVAILABLE, VK_MESHFB_CAPACITY_UNPROVEN);
    assert(finish(0) == 1 && !vk_exec_recording_failed());
    checked("QXCSIDOEBPFTWUV");

    reset(3, VK_SYNC_PROGRESS_COMPLETE, VK_MESHFB_CAPACITY_READY);
    assert(finish(0) == 1 && !nlogs);
    checked("QXCqGTRSIDOEBPFTWUV");

    reset(3, VK_SYNC_PROGRESS_COMPLETE, VK_MESHFB_CAPACITY_READY);
    exec.in_pass = 0;
    expect_begun_tail = 0;
    assert(finish(0) == 1 && !nlogs);
    checked("QCqGTRSIDOEBPFTWUV");

    reset(3, VK_SYNC_PROGRESS_COMPLETE, VK_MESHFB_CAPACITY_OVERFLOW);
    assert(finish(0) == VK_EXEC_RECORD_STOPPED && vk_exec_recording_missing_resource() && nlogs);
    checked("QXCqGTRKSIDOEUV");
    assert(strstr(last_log, "operation 615") && strstr(last_log, "4294967305"));

    reset(3, VK_SYNC_PROGRESS_COMPLETE, VK_MESHFB_CAPACITY_UNPROVEN);
    assert(finish(0) == VK_EXEC_RECORD_STOPPED && vk_exec_recording_missing_resource() && nlogs);
    checked("QXCqGTRKSIDOEUV");
    assert(strstr(last_log, "-5"));

    reset(-1, VK_SYNC_PROGRESS_UNAVAILABLE, VK_MESHFB_CAPACITY_UNPROVEN);
    assert(finish(0) == VK_EXEC_RECORD_STOPPED && vk_exec_recording_missing_resource());
    checked("QXCKSIDOEUV");

    reset(3, VK_SYNC_PROGRESS_UNAVAILABLE, VK_MESHFB_CAPACITY_UNPROVEN);
    assert(finish(0) == VK_EXEC_RECORD_STOPPED && vk_exec_recording_missing_resource());
    checked("QXCqGKSIDOEUV");

    reset(3, VK_SYNC_PROGRESS_STOPPED_AFTER_CUT, VK_MESHFB_CAPACITY_READY);
    assert(finish(0) == VK_EXEC_RECORD_STOPPED && recording_failed == RECORDING_CANCELLED && !nlogs);
    checked("QXCqGTKSIDOEUV");

    reset(3, VK_SYNC_PROGRESS_COMPLETE, VK_MESHFB_CAPACITY_STOPPED);
    assert(finish(0) == VK_EXEC_RECORD_STOPPED && recording_failed == RECORDING_CANCELLED && !nlogs);
    checked("QXCqGTRKSIDOEUV");

    reset(3, VK_SYNC_PROGRESS_COMPLETE, VK_MESHFB_CAPACITY_READY);
    vk_exec_probe_pass_arm(hook, &exec);
    assert(finish(0) == VK_EXEC_RECORD_STOPPED && recording_failed == RECORDING_CANCELLED && !nlogs);
    checked("QHKSIDOEUV");

    reset(3, VK_SYNC_PROGRESS_COMPLETE, VK_MESHFB_CAPACITY_READY);
    stop_in_copy = 1;
    assert(finish(0) == VK_EXEC_RECORD_STOPPED && recording_failed == RECORDING_CANCELLED && !nlogs);
    checked("QXCKSIDOEUV");

    reset(3, VK_SYNC_PROGRESS_COMPLETE, VK_MESHFB_CAPACITY_READY);
    stop_on_read = 1;
    assert(finish(0) == VK_EXEC_RECORD_STOPPED && recording_failed == RECORDING_CANCELLED && !nlogs);
    checked("QXCqGTRKSIDOEUV");

    reset(3, VK_SYNC_PROGRESS_COMPLETE, VK_MESHFB_CAPACITY_READY);
    lose_code = 1;
    assert(finish(0) == VK_EXEC_RECORD_STOPPED && recording_failed == RECORDING_CANCELLED && !nlogs);
    checked("QXKSIDOEUV");

    stop_case();
    pass_cancel_case();
    int scoped = 0, other = 0;
    vk_exec_probe_mesh_partial(&scoped);
    assert(vk_exec_probe_mesh_partial_allowed(&scoped));
    assert(!vk_exec_probe_mesh_partial_allowed(&other));
    vk_exec_probe_mesh_partial(NULL);
    assert(!vk_exec_probe_mesh_partial_allowed(&scoped));
    puts("transformed mesh replay capacity: actual end sequence, prefix adoption, exact read and cancellation passed");
}

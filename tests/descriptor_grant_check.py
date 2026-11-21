"""Exercise selected non-push descriptor preparation, binding and actual pool ownership."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'runtime' / 'vk_exec_draw.c'

def between(text, start, end):
    part = text.index(start)
    return text[part:text.index(end, part + len(start))]

PREFIX = r'''
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include "resource_core_budget.h"

typedef struct CbRes CbRes;
typedef struct { void *device; int validation; } VkCtx;
typedef struct { unsigned set, stage; } VkPipeBinding;
typedef struct {
    const VkPipeBinding *bindings;
    VkDescriptorSetLayout set_layout, set_layout1;
    unsigned n_sets, n_bindings;
    int push, push1;
} VkPipeInfo;
typedef struct { uint64_t fence, resource_op; CbRes *batch; } VkExec;
typedef struct {
    RcBudgetReadiness before;
    RcAdmissionVector peak;
    int selected;
} RcVmaDependency;
typedef enum {
    VK_SYNC_PROGRESS_UNAVAILABLE, VK_SYNC_PROGRESS_COMPLETE, VK_SYNC_PROGRESS_STOPPED,
    VK_SYNC_PROGRESS_STOPPED_AFTER_CUT
} VkSyncProgressResult;
typedef enum {VK_EXEC_GRAPHICS_READY, VK_EXEC_GRAPHICS_NEEDS_ORDERED,
    VK_EXEC_GRAPHICS_UNAVAILABLE, VK_EXEC_GRAPHICS_STOPPED, VK_EXEC_GRAPHICS_INVALIDATED} VkExecGraphicsResult;
enum { VK_EXEC_PREPARATION_DESCRIPTOR, VK_EXEC_PREPARATION_DESCRIPTOR_SET };
typedef struct { int owner, reason, result, detail; uint32_t role, slot; } VkExecPreparationFailure;
typedef int PlatMutex, PlatOnce, PlatCond;
#define PLAT_ONCE_INIT 0
#define SPARE_POOLS 4
#define RWD_STAGE(...)
#define RWD_CALL(stage, call) (call)
#define rt_log(...) ((void)0)
#define rt_fatal(...) abort()
static PlatMutex spare_lock;
static PlatCond scratch_idle;
static int locked, refuse, has_effect, waits, stopped, fail_create, service_relief, prior_waits, optional_refused;
static unsigned optional_admission;
static uint64_t completed;
static void (*on_prior)(void);
static unsigned reserves, materializes, return_pending, destroyed, resets, creates, allocations;
static uint64_t charge[32];
static VkCtx ctx;
static VkCtx *vk_ctx(void) { return &ctx; }
static void plat_mutex_lock(PlatMutex *lock) { assert(lock && !locked); locked = 1; }
static void plat_mutex_unlock(PlatMutex *lock) { assert(lock && locked); locked = 0; }
static void plat_cond_broadcast(PlatCond *condition) { assert(condition && locked); }
static void plat_cond_wait(PlatCond *condition, PlatMutex *lock) { (void)condition; (void)lock; abort(); }
static void plat_once(PlatOnce *once, void (*fn)(void *), void *arg) { if (!*once) *once = 1, fn(arg); }
static int vk_sync_stopping(void) { return stopped; }
static int vk_device_is_lost(void) { return 0; }
static void vk_device_lost_check(VkResult result, const char *msg) { (void)result; (void)msg; abort(); }
static CbRes *cb_res(VkExec *ex) { return ex->batch; }
static VkSyncProgressResult vk_sync_progress_prepare_pipeline(int (*service)(void *), void *arg) {
    assert(!locked && !has_effect && refuse);
    waits++;
    if (stopped) return VK_SYNC_PROGRESS_STOPPED;
    return service(arg) ? VK_SYNC_PROGRESS_COMPLETE : VK_SYNC_PROGRESS_STOPPED;
}
static VkSyncProgressResult vk_sync_progress_wait_prior(uint64_t scene) {
    assert(!locked && !has_effect && on_prior && scene == 40 && !prior_waits);
    prior_waits++;
    on_prior();
    completed = scene;
    return VK_SYNC_PROGRESS_COMPLETE;
}
static uint64_t vk_sync_fence_done(void) { return completed; }
static RcAdmissionStatus resource_core_budget_register_reclaimer_stub(RcBudgetProducer producer, uint32_t domains,
                                                         RcBudgetReclaimFn callback, void *arg) {
    assert(!locked && producer == RC_BUDGET_BUFFER && domains == 3 && callback && !arg);
    return RC_ADMISSION_OK;
}
#define resource_core_budget_register_reclaimer resource_core_budget_register_reclaimer_stub
void resource_core_budget_readiness(int required, RcBudgetReadiness *before) {
    assert(required && !locked);
    *before = (RcBudgetReadiness){.records_available = !refuse, .usable_mask = 3,
                                  .available = {{refuse ? 0 : UINT64_MAX, refuse ? 0 : UINT64_MAX}}};
}
RcAdmissionStatus resource_core_budget_reserve(RcBudgetProducer producer, int required,
                                              const RcAdmissionVector *peak, RcBudgetTicket *out) {
    assert(!locked && producer == RC_BUDGET_BUFFER && peak->amount[0] == peak->amount[1]);
    reserves++;
    if (!required) optional_admission++;
    if (refuse || (!required && optional_refused)) return RC_ADMISSION_REFUSED;
    assert(reserves < sizeof charge / sizeof *charge);
    charge[reserves] = peak->amount[0];
    *out = (RcBudgetTicket){.token = {.slot = reserves}, .producer = producer};
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_materialize(RcBudgetTicket *ticket, const RcAdmissionVector *actual) {
    assert(!locked && actual->amount[0] == charge[ticket->token.slot]);
    materializes++;
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_cancel(RcBudgetTicket *ticket) {
    assert(!locked && charge[ticket->token.slot]);
    charge[ticket->token.slot] = 0;
    *ticket = (RcBudgetTicket){0};
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_retire(RcBudgetTicket *ticket) {
    assert(!locked && charge[ticket->token.slot]);
    return_pending++;
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_destroyed(RcBudgetTicket *ticket) {
    assert(!locked && return_pending == destroyed && destroyed);
    return resource_core_budget_cancel(ticket);
}
void resource_core_budget_progress(void) { assert(!locked); }
void resource_core_budget_service(int required, const RcAdmissionVector *peak, RcBudgetProgress *out) {
    assert(required && peak->amount[0] && !locked);
    *out = (RcBudgetProgress){0};
    if (service_relief) refuse = 0;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorPool(VkDevice dev, const VkDescriptorPoolCreateInfo *info,
        const VkAllocationCallbacks *cb, VkDescriptorPool *out) {
    assert(!locked && !has_effect && info->maxSets == 256 && info->poolSizeCount == 7 && !dev && !cb);
    creates++;
    if (fail_create) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    *out = (VkDescriptorPool)(uintptr_t)(creates + 20);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateDescriptorSets(VkDevice dev, const VkDescriptorSetAllocateInfo *info,
                                                       VkDescriptorSet *out) {
    assert(!locked && !has_effect && !dev && info->descriptorPool && *info->pSetLayouts && info->descriptorSetCount == 1);
    *out = (VkDescriptorSet)(uintptr_t)(++allocations + 100);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkResetDescriptorPool(VkDevice dev, VkDescriptorPool pool, VkDescriptorPoolResetFlags req) {
    assert(!locked && !has_effect && !dev && pool && !req);
    resets++;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorPool(VkDevice dev, VkDescriptorPool pool, const VkAllocationCallbacks *cb) {
    assert(!locked && !has_effect && !dev && pool && !cb && return_pending == destroyed + 1);
    destroyed++;
}
'''

CB = r'''
static void raw_storage_inputs_release(void *owner) { assert(!owner); }
static void ordinary_storage_inputs_release(struct OrdinaryStorageInputs *owner) { assert(!owner); }
struct CbRes {
    uint64_t fence;
    DrawDescriptorPool *pools;
    DrawDescriptorCell *unused_descriptors;
};
static DrawDescriptorPool *spare_pool[SPARE_POOLS];
static unsigned descriptor_destroying;
static DrawDescriptorPool *active_descriptor_pools;
static int descriptor_stopped, nspare_pool;
static PlatOnce descriptor_reclaimer_once;

typedef struct {
    uint32_t opcode;
    int compatibility, pointer_outputs_ready, fallback, selected_writers;
    struct { const VkPipeInfo *info; } use;
    DrawDescriptorGrant descriptors;
} VkExecGraphicsDraw;
static VkExecPreparationFailure failure;
static void vk_exec_graphics_failure_set(VkExecGraphicsDraw *draw, const VkExecPreparationFailure *cause) {
    assert(draw && !has_effect && !locked && !failure.detail);
    failure = *cause;
}
'''

TEST = r'''
static CbRes *old_scene;
static void release_old_scene(void) { draw_descriptor_pools_complete(old_scene); }

int main(void) {
    const VkPipeBinding bindings[] = {{0, 0}, {1, 4}};
    const VkPipeInfo select = {.bindings = bindings, .n_sets = 2, .n_bindings = 2, .push = 1,
        .set_layout = (VkDescriptorSetLayout)(uintptr_t)1, .set_layout1 = (VkDescriptorSetLayout)(uintptr_t)2};
    VkPipeInfo different = select;
    CbRes a = {.fence = 40}, b = {.fence = 41};
    VkExec ex = {.fence = 40, .resource_op = 52, .batch = &a};
    VkExecGraphicsDraw draw = {.opcode = 26, .use = {.info = &select}};
    assert(draw_descriptor_set_mask(&select) == 2);
    different.n_bindings = 1;
    assert(draw_descriptor_set_mask(&different) == 0);
    ctx.validation = 1;
    assert(draw_descriptor_set_mask(&different) == 2);
    different.push1 = 1;
    assert(draw_descriptor_set_mask(&different) == 0);
    ctx.validation = 0;
    assert(graphics_descriptors_ready(&ex, &draw, 0) == VK_EXEC_GRAPHICS_NEEDS_ORDERED);
    assert(!reserves && !creates && !allocations);
    refuse = service_relief = 1;
    assert(graphics_descriptors_ready(&ex, &draw, 1) == VK_EXEC_GRAPHICS_READY);
    assert(reserves == 2 && creates == 1 && allocations == 1 && materializes == 1 && waits == 1);
    assert(draw.descriptors.info == &select && draw.descriptors.owner == &a && draw.descriptors.set[1]);
    assert(draw_descriptor_grant_current(&ex, &select, &draw.descriptors, ex.resource_op));
    assert(!draw_descriptor_grant_current(&ex, &different, &draw.descriptors, ex.resource_op));
    DrawDescriptorCell *first = draw.descriptors.set[1];
    draw_descriptor_grant_release(&draw.descriptors); // Unpublished canceled ready lease.
    assert(graphics_descriptors_ready(&ex, &draw, 0) == VK_EXEC_GRAPHICS_READY && draw.descriptors.set[1] == first);
    descriptor_binding_grant = &draw.descriptors;
    has_effect = 1;
    assert(!draw_descriptor_bind_set(&ex, &different, 1));
    ex.fence = 41;
    assert(!draw_descriptor_bind_set(&ex, &select, 1));
    ex.fence = 40;
    ex.resource_op++;
    assert(!draw_descriptor_bind_set(&ex, &select, 1));
    ex.resource_op--;
    assert(draw_descriptor_bind_set(&ex, &select, 1) == first->set);
    assert(!draw_descriptor_bind_set(&ex, &select, 1) && allocations == 1);
    descriptor_binding_grant = NULL;
    draw_descriptor_grant_release(&draw.descriptors); // Consumed set stays with native use.
    has_effect = 0;
    assert(!a.unused_descriptors && !resets && !destroyed);
    assert(draw_descriptor_reclaim(NULL, &((RcAdmissionVector){{UINT64_MAX}})) == RC_BUDGET_RECLAIM_NONE);
    // If only the current recorder holds pools, a refusal cannot wait on its own output.
    RcVmaDependency blocked = {.selected = 1, .peak = draw_descriptor_pool_peak()};
    refuse = 1; service_relief = 0;
    assert(draw_descriptor_progress(ex.fence, &blocked) == VK_SYNC_PROGRESS_UNAVAILABLE && !prior_waits && !resets);
    old_scene = &a; on_prior = release_old_scene;
    ex.fence = b.fence; ex.batch = &b;
    assert(graphics_descriptors_ready(&ex, &draw, 1) == VK_EXEC_GRAPHICS_READY);
    assert(prior_waits == 1 && resets == 1 && !nspare_pool && charge[2]);
    refuse = 0;
    assert(creates == 1 && allocations == 2 && materializes == 1 && !nspare_pool);
    draw_descriptor_grant_release(&draw.descriptors);
    draw_descriptor_pools_complete(&b);
    assert(draw_descriptor_reclaim(NULL, &((RcAdmissionVector){{1}})) == RC_BUDGET_RECLAIM_DESTROYED);
    assert(resets == 2 && !nspare_pool && destroyed == 1 && !charge[2]);
    CbRes c = {.fence = 42}; ex.fence = 42; ex.batch = &c;
    fail_create = 1;
    assert(graphics_descriptors_ready(&ex, &draw, 1) == VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(creates == 2 && reserves == 4 && !charge[4] && !c.pools);
    assert(failure.owner == VK_EXEC_PREPARATION_DESCRIPTOR && failure.reason == VK_EXEC_PREPARATION_DESCRIPTOR_SET &&
           failure.detail == VK_ERROR_OUT_OF_DEVICE_MEMORY && failure.role == 4 && failure.slot == 1);
    draw_descriptor_grant_release(&draw.descriptors);

    const VkPipeInfo compute = {.n_sets = 1, .set_layout = select.set_layout};
    DrawDescriptorGrant compute_grant = {0};
    fail_create = 0;
    uint32_t pre = allocations;
    assert(draw_descriptor_grant_prepare(&ex, &compute, &compute_grant, 1) == VK_EXEC_GRAPHICS_READY);
    DrawDescriptorCell *pending = compute_grant.set[0];
    assert(pending && allocations == pre + 1);
    assert(!draw_descriptor_grant_current(&ex, &compute, &compute_grant, ex.resource_op + 1));
    assert(draw_descriptor_grant_prepare(&ex, &compute, &compute_grant, 1) == VK_EXEC_GRAPHICS_READY);
    assert(compute_grant.set[0] == pending && allocations == pre + 1);
    draw.compatibility = 1;
    draw.pointer_outputs_ready = 1;
    draw.descriptors = compute_grant;
    compute_grant = (DrawDescriptorGrant){0};
    select_graphics_descriptors(&draw);
    assert(descriptor_binding_grant == &draw.descriptors);
    has_effect = 1;
    assert(draw_descriptor_bind_set(&ex, &compute, 0) == pending->set);
    assert(!draw_descriptor_bind_set(&ex, &compute, 0) && allocations == pre + 1);
    select_graphics_descriptors(NULL);
    assert(!descriptor_binding_grant);
    draw_descriptor_grant_release(&draw.descriptors);
    has_effect = 0;
    assert(!c.unused_descriptors);
    draw_descriptor_pools_complete(&c);
    // Select both motion sets before publishing a packet. The first placement survives a
    // refusal of the second only as an unused cell, and is immediately available to a required draw.
    CbRes motion = {.fence = 43};
    DrawDescriptorGrant optional = {.owner = &motion};
    assert(draw_descriptor_set_claim(&motion, select.set_layout, 1, 0, &optional.set[0], NULL) == VK_SUCCESS);
    DrawDescriptorCell *first_optional = optional.set[0];
    assert(first_optional && !motion.unused_descriptors);
    motion.pools->allocated = DRAW_DESCRIPTOR_SETS_PER_POOL;
    optional_refused = 1;
    const int observed_waits = waits;
    const unsigned observed_optional = optional_admission;
    assert(!nspare_pool);
    assert(draw_descriptor_set_claim(&motion, select.set_layout1, 1, 0, &optional.set[1], NULL) == VK_NOT_READY);
    assert(waits == observed_waits && optional_admission == observed_optional + 1 && !optional.set[1]);
    draw_descriptor_grant_release(&optional);
    assert(motion.unused_descriptors == first_optional);
    DrawDescriptorCell *required = NULL;
    assert(draw_descriptor_set_claim(&motion, select.set_layout, 0, 1, &required, NULL) == VK_SUCCESS);
    assert(required == first_optional && !motion.unused_descriptors);
    draw_descriptor_pools_complete(&motion);
    vk_exec_draw_descriptor_shutdown();
    puts("descriptor grant: selected layout, unused reuse, cancellation, completion and charge passed");
    return 0;
}
'''


def main():
    text = SOURCE.read_text(encoding='utf-8')
    declarations = between(text, 'typedef struct DrawDescriptorCell', 'struct CbRes {')
    allocator = between(text, 'static const VkDescriptorPoolSize draw_descriptor_sizes', 'static void support_in(')
    header = SOURCE.with_name('vk_exec.h').read_text(encoding='utf-8')
    grant_type = between(header, 'typedef struct {\n    struct CbRes *owner;', '} DrawDescriptorGrant;') + '} DrawDescriptorGrant;'
    prepare = between(text, 'static int draw_descriptor_service(', 'static int graphics_load_pool_members_disjoint(')
    selected = between(text, 'static VkExecGraphicsResult graphics_draw_select(', '} GraphicsCompatibilityPool;')
    assert 'draw_descriptor_grant_current(ex, draw->use.info, &draw->descriptors, ex->resource_op)' in text
    assert 'draw_descriptor_grant_prepare(ex, dispatch->info, &dispatch->descriptors, 1)' in text
    actual_bind = between(text, 'static void bind_resources(', '// ---- draws')
    assert 'result = graphics_descriptors_ready(ex, draw, ordered);' in selected
    compat = between(text, 'static int prepare_draw_phase(VkExec *ex,', 'static int prepare_draw(')
    assert compat.index('graphics_pipeline_ready(ex, draw, 1)') < compat.index('gate = "issue-descriptors"')
    assert compat.index('graphics_descriptors_ready(ex, draw, 1)') < compat.index('graphics_record_state(ex, &draw->state')
    record = between(text, 'static int graphics_record_state(', 'static int graphics_draw_activate(')
    match = re.search(r"^        descriptor_binding_grant = draw[^;]+;", record, re.M)
    assert match, "prepared graphics recorder did not select a descriptor grant"
    select = "static void select_graphics_descriptors(VkExecGraphicsDraw *draw) {\n" + match[0] + "\n}\n"
    assert actual_bind.index('descriptor_binding_grant->info != info') < actual_bind.index('binding_manifest_publish(ex)')
    assert 'set = draw_descriptor_bind_set(ex, info, s);' in actual_bind
    motion = between(text, 'static int mo_desc(', 'static uint32_t mo_vertex_buffers(')
    assert motion.index('draw_descriptor_set_claim(unused.owner, layout, 1, 0') < motion.index('binding_manifest_publish(ex)')
    assert motion.index('draw_descriptor_grant_release(&unused)') < motion.index('binding_manifest_publish(ex)')
    assert 'alloc_set(' not in motion
    cc = Path(os.environ['CC'])
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get('PATH', ''))
    with tempfile.TemporaryDirectory(prefix='draw-descriptor-grant-') as temp:
        path = Path(temp) / 'check.c'
        path.write_text('\n'.join((PREFIX, grant_type, declarations, CB, allocator, prepare, select, TEST)), encoding='utf-8')
        exe = path.with_suffix('.exe')
        subprocess.run([str(cc), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror', '-fsanitize=undefined',
                        '-fsanitize-trap=undefined', '-I' + str(ROOT / 'runtime'), str(path), '-o', str(exe)],
                       env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == '__main__':
    main()

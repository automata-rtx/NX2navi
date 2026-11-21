"""Exercise production alias cohort admission and ordered replay with real shared tickets."""
from pathlib import Path
import argparse
import os
import subprocess
import tempfile

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--runtime", type=Path, default=DEFAULT_ROOT / "runtime")
parser.add_argument("--headers", type=Path)
args = parser.parse_args()
runtime = args.runtime
headers = args.headers or runtime
texture = (runtime / "vk_texture.c").read_text()
replay = (runtime / "vk_exec.c").read_text()
joint = "VkResult vk_texture_alias_read_prepare(" in texture


def definition(source, signature):
    first = source.index(signature)
    return source[first:source.index("\n}\n", first) + 3]


PREFIX = r"""
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define RC_BUDGET_TEST
#include "resource_core_budget.c"

#define MB (1024ull * 1024ull)
typedef uint64_t VkImage, VkDeviceSize;
typedef void *VmaAllocation;
typedef unsigned VkImageType, VkFormat, VkSampleCountFlagBits;
typedef enum {
    VK_SUCCESS, VK_NOT_READY, VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY,
    VK_ERROR_INITIALIZATION_FAILED, VK_ERROR_DEVICE_LOST,
} VkResult;
enum {
    VK_NULL_HANDLE, VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    VK_IMAGE_LAYOUT_UNDEFINED, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT = 16,
    VK_EXEC_ALIAS_READ_LIMIT = 8,
};
typedef struct {
    unsigned sType, flags, imageType, format;
    struct { unsigned w, h, d; } extent;
    unsigned mipLevels, arrayLayers, samples, tiling, usage, initialLayout;
} VkImageCreateInfo;
typedef struct { unsigned usage, flags, memoryTypeBits; } VmaAllocationCreateInfo;
typedef struct { VkDeviceSize size; } VmaAllocationInfo;
typedef struct {
    RcBudgetReadiness before;
    RcAdmissionVector peak, reuse;
    uint64_t released;
    unsigned memory_type;
    int selected, can_reuse, owner_full, reuse_unattempted;
} RcVmaDependency;
typedef struct {
    RcBudgetProducer producer;
    int required;
    RcBudgetTicket *peak_ticket;
    RcVmaDependency *dependency;
} RcVmaRequest;
typedef struct { uint64_t upload; VkDeviceSize size; } VkTextureAliasStagingProgress;
typedef struct { uint64_t scene, ordinal; } RcRegistryPosition;
typedef struct { int vma; } VkCtx;
typedef struct { unsigned flags, type, format, w, h, d, levels, layers, samples, usage; } ImgKey;
typedef struct {
    uint64_t buffer, size;
    int from_cache;
    RcBudgetTicket ticket;
} VkPreparedStaging;
typedef struct VkTextureAliasRead {
    ImgKey placement;
    uint64_t stage_bytes, workspace_bytes, placement_bytes, raw_upload;
    struct { VkImage image; } result;
    int raw_output, target_hold, uploaded;
    VmaAllocation allocation;
    VkPreparedStaging staging;
    RcBudgetTicket workspace_ticket;
    unsigned char *workspace;
} VkTextureAliasRead;
typedef enum { VK_EXEC_RESTORE_READY, VK_EXEC_RESTORE_STOPPED, VK_EXEC_RESTORE_STALE, VK_EXEC_RESTORE_FAILED } VkExecRestoreResult;
typedef enum { VK_SYNC_PROGRESS_UNAVAILABLE, VK_SYNC_PROGRESS_COMPLETE, VK_SYNC_PROGRESS_STOPPED,
               VK_SYNC_PROGRESS_STOPPED_AFTER_CUT } VkSyncProgressResult;
typedef struct {
    uint64_t fence, cb, recorded;
    struct { unsigned width; } tonemap_extent;
    unsigned bc_written_mask;
} VkExec;

typedef enum { NORMAL, REMAP_WAIT, ADVANCE_WAIT, STOP_WAIT, LOSE_STAGING, REMAP_BARRIER, NO_NATIVE_RELEASE, CLOSE_AFTER_CUT, STOP_AFTER_CUT } Transition;
static Transition transition;
static RcBudgetSample telemetry;
static RcRegistryPosition original, position;
static VkExec replay;
static VkCtx device;
static int capture_current, provisional_open, stopping, device_lost, contact, recording_failed, expect_optional;
static unsigned budget_lock_depth;
static uint64_t test_clock = 1000000000;
static uint64_t cached, offered_size, offered_upload, observed_prior, native_revision;
static uint64_t raw_submitted, raw_completed;
static unsigned raw_finishes, scope_checks;
static int abandon_raw;
static unsigned queries, stage_plans, stage_allocations, stage_checks, stage_waits, native_checks, native_waits, collections;
static unsigned creates, destroys, barriers, recorded, prefix, workspace_calls, fail_workspace;
static int reusable[3], compatible[3], growth_allowed[3], owner_full[3], fail_stage;
static VkDeviceSize fail_stage_size;
static RcBudgetTicket image_tickets[3];

uint64_t rc_budget_test_now_ns(void) { return test_clock; }
void rc_budget_test_lock(void) { assert(!budget_lock_depth); budget_lock_depth++; }
void rc_budget_test_unlock(void) { assert(budget_lock_depth == 1); budget_lock_depth--; }
void rc_budget_test_sample(RcBudgetSample *out) { assert(budget_lock_depth); *out = telemetry; }
void rc_budget_test_wait(uint32_t time) { (void)time; abort(); }
static void set_room(uint64_t ram, uint64_t heap) {
    telemetry.total.amount[RC_BUDGET_RAM] = telemetry.total.amount[RC_BUDGET_COMMIT] =
        telemetry.total.amount[RC_BUDGET_GPU_FIRST] = 1024 * MB;
    telemetry.total.amount[RC_BUDGET_STAGING] = 32 * MB;
    telemetry.free.amount[RC_BUDGET_RAM] = 128 * MB + ram;
    telemetry.free.amount[RC_BUDGET_COMMIT] = 160 * MB;
    telemetry.free.amount[RC_BUDGET_GPU_FIRST] = 32 * MB + heap;
    telemetry.known_mask = (1u << RC_BUDGET_RAM) | (1u << RC_BUDGET_COMMIT) |
        (1u << RC_BUDGET_GPU_FIRST) | (1u << RC_BUDGET_STAGING);
}
static void no_reservation(void) {
    RcBudgetSnapshot state;
    resource_core_budget_snapshot(&state);
    for (unsigned d = 0; d < RC_ADMISSION_MAX_DOMAINS; ++d) assert(!state.reserved.amount[d]);
}
static void no_owned_storage(void) {
    RcBudgetSnapshot state;
    resource_core_budget_snapshot(&state);
    for (unsigned d = 0; d < RC_ADMISSION_MAX_DOMAINS; ++d)
        assert(!state.reserved.amount[d] && !state.retained.amount[d] && !state.pending.amount[d]);
}
static VkCtx *vk_ctx(void) { return &device; }
static int vk_sync_stopping(void) { return stopping; }
static int vk_device_is_lost(void) { return device_lost; }
static int vk_exec_recording_failed(void) { return recording_failed; }
static int vk_contact_pending(void) { return contact; }
static void vk_device_lost_check(VkResult result, const char *role) {
    assert(result == VK_ERROR_DEVICE_LOST && role);
    device_lost = 1;
}
static int rc_registry_position_compare(RcRegistryPosition a, RcRegistryPosition b) {
    return a.scene != b.scene || a.ordinal != b.ordinal;
}
static RcRegistryPosition vk_exec_resource_position(void) { return position; }
static void *rc_registry_runtime(void) { return &position; }
static int rc_registry_provisional_read_current(void *registry, RcRegistryPosition read) {
    return registry == &position && provisional_open && !rc_registry_position_compare(read, original);
}
static int exact_source_current(void *scope) {
    assert(scope == &original);
    ++scope_checks;
    return capture_current && !rc_registry_position_compare(position, original);
}
static void mark_transition(void) {
    test_clock += 2000000;
    if (transition == REMAP_WAIT) capture_current = 0;
    if (transition == ADVANCE_WAIT) position.ordinal += 2;
    if (transition == STOP_WAIT) stopping = 1;
}
static VkResult resource_core_vma_image_peak(const RcVmaRequest *request, const VkImageCreateInfo *info,
    const VmaAllocationCreateInfo *settings, VmaAllocationCreateInfo *placement, RcAdmissionVector *peak) {
    unsigned id = info->extent.w;
    assert(id >= 1 && id <= 2 && request->required == !expect_optional && !budget_lock_depth);
    ++queries;
    *placement = *settings;
    placement->memoryTypeBits = 1u << id;
    *peak = (RcAdmissionVector){.amount[RC_BUDGET_GPU_FIRST] = 64};
    *request->dependency = (RcVmaDependency){.selected=1, .memory_type=id, .can_reuse=reusable[id]};
    return VK_SUCCESS;
}
static VkResult resource_core_vma_create_image(const RcVmaRequest *request, int allocator, const VkImageCreateInfo *info,
    const VmaAllocationCreateInfo *placement, VkImage *image, VmaAllocation *allocation, VmaAllocationInfo *actual,
    RcAdmissionStatus *admission) {
    (void)admission;
    unsigned id = info->extent.w;
    assert(id >= 1 && id <= 2 && (!JOINT || !(queries & 1)) &&
           request->required == !expect_optional && allocator == device.vma);
    assert(!*image && !*allocation && !image_tickets[id].token.ledger && placement->memoryTypeBits == 1u << id);
    if (request->dependency) *request->dependency = (RcVmaDependency){.selected=1, .memory_type=id,
        .can_reuse=reusable[id], .owner_full=owner_full[id], .released=native_revision};
    if (owner_full[id]) return VK_NOT_READY;
    const int reuse = reusable[id] && compatible[id];
    if (!reuse && ((placement->flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) || !growth_allowed[id]))
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    const RcAdmissionVector cost = {.amount[RC_BUDGET_GPU_FIRST] = 64};
    if (reuse) {
        const RcAdmissionVector working = {0};
        assert(resource_core_budget_adopt_compatible(RC_BUDGET_TEXTURE, request->required, &cost, &working,
                                                     &image_tickets[id]) == RC_ADMISSION_OK);
    } else {
        assert(resource_core_budget_split(request->peak_ticket, &cost, RC_BUDGET_TEXTURE,
                                          &image_tickets[id]) == RC_ADMISSION_OK);
        assert(resource_core_budget_materialize(&image_tickets[id], &cost) == RC_ADMISSION_OK);
    }
    *image = 100 + id;
    *allocation = (VmaAllocation)(uintptr_t)id;
    actual->size = 64;
    creates++;
    return VK_SUCCESS;
}
static void resource_core_vma_destroy_image(int vma, VkImage image, VmaAllocation allocation) {
    uintptr_t id = (uintptr_t)allocation;
    assert(vma == device.vma && image == 100 + id && id <= 2 && !budget_lock_depth);
    RcBudgetTicket *ticket = &image_tickets[id];
    assert(resource_core_budget_retire(ticket) == RC_ADMISSION_OK);
    assert(resource_core_budget_destroyed(ticket) == RC_ADMISSION_OK);
    destroys++;
}
static void resource_core_vma_dependency_rebase(RcVmaDependency *need, int required) {
    assert(destroys == creates && !expect_optional);
    no_reservation();
    need->released = native_revision;
    resource_core_budget_readiness(required, &need->before);
}
static int resource_core_vma_dependency_ready(const RcVmaDependency *need, const RcBudgetReadiness *now) {
    if (need->owner_full) return now->records_available && need->released != native_revision;
    return need->memory_type == 2 && need->released != native_revision;
}
static VkResult vk_staging_prepare_plan(VkDeviceSize size, VkPreparedStaging *stage, RcAdmissionVector *peak) {
    assert(size && !stage->buffer && !stage->ticket.token.ledger);
    stage_plans++;
    *stage = (VkPreparedStaging){.size=size};
    *peak = (RcAdmissionVector){0};
    if (cached == size) {
        cached = 0;
        stage->buffer = 300 + size;
        stage->from_cache = 1;
    } else peak->amount[RC_BUDGET_RAM] = 32;
    return VK_SUCCESS;
}
static VkResult vk_staging_prepare_allocate(VkPreparedStaging *stage, RcBudgetTicket *ticket, int required) {
    assert(ticket->token.ledger && required == !expect_optional);
    if (stage->buffer) return VK_SUCCESS;
    if (fail_stage && stage->size == fail_stage_size) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    RcAdmissionVector cost = {.amount[RC_BUDGET_RAM] = 32};
    assert(resource_core_budget_split(ticket, &cost, RC_BUDGET_STAGE, &stage->ticket) == RC_ADMISSION_OK);
    assert(resource_core_budget_materialize(&stage->ticket, &cost) == RC_ADMISSION_OK);
    stage->buffer = 200 + stage->size;
    stage_allocations++;
    return VK_SUCCESS;
}
static void vk_staging_prepare_cancel(VkPreparedStaging *stage) {
    if (stage->from_cache) cached = stage->size;
    if (stage->ticket.token.ledger) {
        assert(resource_core_budget_retire(&stage->ticket) == RC_ADMISSION_OK);
        assert(resource_core_budget_destroyed(&stage->ticket) == RC_ADMISSION_OK);
    }
    *stage = (VkPreparedStaging){0};
}
static uint64_t vk_staging_prepare_supplier(VkDeviceSize size, uint64_t after) {
    stage_checks++;
    assert(!expect_optional);
    return offered_size == size && offered_upload > after ? offered_upload : 0;
}
static int vk_staging_prepare_available(VkDeviceSize size) { return cached == size; }
static int vk_texture_alias_read_raw_progress(VkTextureAliasRead *alias, RcRegistryPosition read, uint64_t *upload) {
    *upload=alias->raw_upload;
    return !*upload || (capture_current && !rc_registry_position_compare(read,original));
}
static int vk_texture_alias_read_raw_finish(VkTextureAliasRead *alias, RcRegistryPosition read) {
    if (!alias->raw_upload) return 1;
    assert(!scope_checks && !rc_registry_position_compare(read,original));
    if (abandon_raw || !capture_current || alias->raw_upload>raw_completed) return 0;
    ++raw_finishes;
    alias->raw_upload=0;
    return 1;
}
static void vk_upload_progress(uint64_t *submitted,uint64_t *completed) {
    *submitted=raw_submitted;
    *completed=raw_completed;
}
static VkSyncProgressResult upload_staging_progress(VkExec *ex, uint64_t upload) {
    assert(ex == &replay && upload == offered_upload && !barriers && !recorded);
    no_owned_storage();
    stage_waits++;
    mark_transition();
    if (stopping) return VK_SYNC_PROGRESS_STOPPED;
    if (upload<=raw_submitted) raw_completed=upload;
    if (transition != LOSE_STAGING && offered_size) cached = offered_size;
    fail_stage = 0;
    return VK_SYNC_PROGRESS_COMPLETE;
}
static uint64_t vk_mem_progress_prior(const RcVmaDependency *need, uint64_t scene) {
    assert(need->selected && need->memory_type == 2 && scene == original.scene && !recorded);
    if (owner_full[2]) assert(need->owner_full == 4);
    native_checks++;
    return observed_prior;
}
static void vk_mem_collect(void) {
    no_owned_storage();
    if (observed_prior && transition != NO_NATIVE_RELEASE) {
        compatible[2] = 1;
        owner_full[2] = 0;
        native_revision++;
    }
    collections++;
}
static void vk_exec_barrier(VkExec *ex) {
    assert(ex == &replay && !recorded && !stopping);
    barriers++;
    if (transition == REMAP_BARRIER) capture_current = 0;
}
static void vk_texture_alias_read_record(VkTextureAliasRead *alias, uint64_t cb, uint64_t scene) {
    assert(barriers == 1 && cb == replay.cb && scene == original.scene && alias->staging.buffer && alias->workspace);
    assert(exact_source_current(&original));
    recorded++;
}
static VkSyncProgressResult vk_exec_graphics_pool_progress(VkExec *ex, uint64_t prior) {
    assert(ex == &replay);
    if (prior) {
        assert(prior == observed_prior && prior < original.scene && !recorded && !barriers);
        no_owned_storage();
        native_waits++;
        mark_transition();
        return stopping ? VK_SYNC_PROGRESS_STOPPED : VK_SYNC_PROGRESS_COMPLETE;
    }
    assert(recorded == 2 && replay.recorded == 1);
    prefix++;
    if (transition == STOP_AFTER_CUT) return VK_SYNC_PROGRESS_STOPPED_AFTER_CUT;
    if (transition == CLOSE_AFTER_CUT) provisional_open = 0;
    return VK_SYNC_PROGRESS_COMPLETE;
}
static void *workspace_malloc(size_t size) {
    workspace_calls++;
    if (workspace_calls == fail_workspace) return NULL;
    return malloc(size);
}
#define malloc workspace_malloc
"""

TESTS = r"""
#undef malloc
static VkTextureAliasRead alias[2], *cohort[2] = {&alias[0], &alias[1]};
static void reset(uint64_t ram, uint64_t heap) {
    no_owned_storage();
    resource_core_budget_test_reset();
    set_room(ram, heap);
    original = position = (RcRegistryPosition){9, 40};
    replay = (VkExec){.fence=9, .cb=17};
    capture_current = provisional_open = 1;
    transition = NORMAL;
    stopping = device_lost = recording_failed = contact = expect_optional = 0;
    cached = offered_size = offered_upload = observed_prior = native_revision = 0;
    raw_submitted = raw_completed = raw_finishes = scope_checks = abandon_raw = 0;
    queries = stage_plans = stage_allocations = stage_checks = stage_waits = native_checks = native_waits = collections = 0;
    creates = destroys = barriers = recorded = prefix = workspace_calls = fail_workspace = fail_stage = fail_stage_size = 0;
    for (unsigned id = 1; id <= 2; ++id) {
        growth_allowed[id] = 1;
        reusable[id] = compatible[id] = owner_full[id] = 0;
        alias[id - 1] = (VkTextureAliasRead){.placement={.w=id,.h=1,.d=1,.levels=1,.layers=1,.samples=1},
                                             .stage_bytes=8*id, .workspace_bytes=4};
    }
}
static VkExecRestoreResult prepare(void) {
    return vk_exec_prepare_alias_readback(&replay, cohort, 2, exact_source_current, &original);
}
static void assert_unissued(void) {
    assert(!recorded && !prefix);
    for (unsigned i = 0; i < 2; ++i) {
        assert(!alias[i].allocation && !alias[i].placement_bytes && !alias[i].workspace && !alias[i].staging.buffer);
    }
    no_owned_storage();
}
#if JOINT
static void assert_issued_then_cancel(void) {
    assert(recorded == 2 && prefix == 1 && replay.recorded == 1 && barriers == 1 && exact_source_current(&original));
    no_reservation();
    vk_texture_alias_read_unprepare(cohort, 2); // Simulate caller release after the proven local readback prefix.
    no_owned_storage();
}
#endif
int main(void) {
    reset(96, 96);
    offered_upload = 11; offered_size = 8;
    assert(prepare() == VK_EXEC_RESTORE_FAILED);
    assert(queries == 2 && stage_plans == 2 && !creates && !stage_waits && !stage_checks && !native_waits);
    assert_unissued(); // Either required leg fits by itself. Their joint peak is refused before creating one image.
#if JOINT
    reset(96, 128);
    alias[0].raw_upload=11; alias[1].raw_upload=13;
    offered_upload=raw_submitted=13;
    assert(prepare()==VK_EXEC_RESTORE_READY && stage_waits==1 && raw_finishes==2 && scope_checks);
    assert_issued_then_cancel();

    reset(96, 128);
    alias[0].raw_upload=13; raw_submitted=12;
    assert(prepare()==VK_EXEC_RESTORE_STALE && !stage_waits && !raw_finishes && !scope_checks);
    assert_unissued();

    reset(96, 128);
    alias[0].raw_upload=13; offered_upload=raw_submitted=13; abandon_raw=1;
    assert(prepare()==VK_EXEC_RESTORE_STALE && stage_waits==1 && !raw_finishes && !scope_checks);
    assert_unissued();

    reset(96, 128);
    alias[0].raw_upload=13; offered_upload=raw_submitted=13; transition=ADVANCE_WAIT;
    assert(prepare()==VK_EXEC_RESTORE_STALE && stage_waits==1 && !raw_finishes && !scope_checks);
    assert_unissued();

    reset(40, 128); offered_upload = 11; offered_size = 8;
    assert(prepare() == VK_EXEC_RESTORE_READY);
    assert(stage_waits == 1 && stage_allocations == 1 && creates == 2 && !native_waits);
    assert_issued_then_cancel();
    const Transition rejects[] = { REMAP_WAIT, ADVANCE_WAIT, STOP_WAIT, LOSE_STAGING };
    const VkExecRestoreResult expected[] = { VK_EXEC_RESTORE_STALE, VK_EXEC_RESTORE_STALE,
                                           VK_EXEC_RESTORE_STOPPED, VK_EXEC_RESTORE_FAILED };
    for (unsigned i = 0; i < sizeof rejects / sizeof *rejects; ++i) {
        reset(40, 128); offered_upload = 11; offered_size = 8; transition = rejects[i];
        assert(prepare() == expected[i] && stage_waits == 1 && !creates && !barriers);
        assert_unissued();
    }

    reset(96, 64);
    reusable[1] = reusable[2] = compatible[1] = 1;
    assert(prepare() == VK_EXEC_RESTORE_READY && creates == 3 && destroys == 1 && !native_waits && !stage_waits);
    assert_issued_then_cancel(); // One existing compatible image plus one allocated placement.
    reset(96, 64);
    reusable[1] = reusable[2] = compatible[2] = 1;
    assert(prepare() == VK_EXEC_RESTORE_READY && creates == 2 && !destroys && !native_waits && !stage_waits);
    assert_issued_then_cancel();
    for (unsigned test = 0; test < 3; ++test) {
        reset(96, 64);
        reusable[1] = reusable[2] = compatible[1] = 1;
        growth_allowed[2] = 0;
        observed_prior = test == 0 ? 0 : 7;
        if (test == 2) transition = REMAP_WAIT;
        assert(prepare() == (test == 0 ? VK_EXEC_RESTORE_FAILED : test == 1 ? VK_EXEC_RESTORE_READY : VK_EXEC_RESTORE_STALE));
        assert(native_waits == (test != 0) && collections == (test == 1) && !stage_waits);
        if (test == 1) assert_issued_then_cancel();
        else assert_unissued();
    }
    reset(96, 64); reusable[1] = reusable[2] = compatible[1] = 1; growth_allowed[2] = 0;
    observed_prior = 7; transition = NO_NATIVE_RELEASE;
    assert(prepare() == VK_EXEC_RESTORE_FAILED && native_waits == 1 && collections == 1 && queries == 2);
    assert_unissued(); // Completion provided no new compatible capacity.
    reset(96, 128); owner_full[2] = 1; observed_prior = 7;
    assert(prepare() == VK_EXEC_RESTORE_READY && native_waits == 1 && collections == 1);
    assert_issued_then_cancel();

    reset(96, 128); offered_upload = 11; offered_size = 16; fail_stage = 1; fail_stage_size = 16;
    assert(prepare() == VK_EXEC_RESTORE_READY && stage_waits == 1 && creates == 4 && destroys == 2);
    assert_issued_then_cancel(); // A failure after a previous image and staging member releases them before the wait.
    reset(96, 128); fail_workspace = 2;
    assert(prepare() == VK_EXEC_RESTORE_FAILED && creates == 2 && destroys == 2 && workspace_calls == 2 && !barriers);
    assert_unissued();
    reset(96, 128); transition = REMAP_BARRIER;
    assert(prepare() == VK_EXEC_RESTORE_STALE && creates == 2 && destroys == 2 && barriers == 1);
    assert_unissued();

    reset(96, 128);
    transition = CLOSE_AFTER_CUT;
    assert(prepare() == VK_EXEC_RESTORE_STALE && !provisional_open);
    assert_issued_then_cancel(); // The source still matches, but its original read may no longer issue.

    reset(96, 128);
    transition = STOP_AFTER_CUT;
    assert(prepare() == VK_EXEC_RESTORE_STOPPED);
    assert_issued_then_cancel(); // Issued ownership outlives the caller even when the tail stops.

    reset(40, 128);
    alias[0].target_hold = 1;
    alias[0].result.image = 432;
    alias[1].raw_output = 1;
    offered_upload = 11;
    offered_size = 8;
    assert(prepare() == VK_EXEC_RESTORE_READY && !creates && !queries && stage_waits == 1);
    assert_issued_then_cancel();
    assert(alias[0].result.image == 432 && !alias[1].result.image);

    reset(40, 128); offered_upload = 11; offered_size = 8; expect_optional = 1;
    VkTextureAliasStagingProgress upload;
    RcVmaDependency native;
    assert(vk_texture_alias_read_prepare(cohort, 2, 0, 0, &upload, &native) == VK_NOT_READY);
    assert(!upload.upload && !native.selected && !stage_checks && !stage_waits && !native_waits && !creates);
    assert_unissued();
#endif
    puts("alias readback admission: joint refusal and reuse, proven staging and native progress, remap and cancellation passed");
}
"""

metadata = ""
if joint:
    begin = texture.index("typedef struct {\n    VkImageCreateInfo image;\n    VmaAllocationCreateInfo placement;\n    RcAdmissionVector image_peak")
    end = texture.find("static void image_alias_record_copies(", begin)
    if end < 0:
        end = texture.index("void vk_texture_alias_read_record(", begin)
    selection = texture[begin:end]
    current = "".join(definition(replay, name) for name in (
        "static int alias_readback_position_current(", "static int alias_readback_source_current(",
        "static VkExecRestoreResult alias_readback_raw_progress("))
    if "static VkResult image_alias_plan_storage(" in selection:
        usnap = (headers / "vk_usnap.c").read_text()
        metadata = ("\n#define rt_fatal(...) abort()\n"
                    + definition(usnap, "void vk_replay_metadata_free(") + "#undef rt_fatal\n")
        # The existing injection counts per-image compose workspace allocations.
        selection = selection.replace("static VkResult image_alias_plan_storage(",
                                      "#undef malloc\nstatic VkResult image_alias_plan_storage(")
else:
    selection = definition(texture, "VkResult vk_texture_alias_read_allocate(")
    current = ""
source = (
    f"#define JOINT {int(joint)}\n" + PREFIX
    + definition(texture, "static int restore_peak_add(")
    + definition(texture, "static void image_alias_stage_release(")
    + metadata + selection + current
    + definition(replay, "VkExecRestoreResult vk_exec_prepare_alias_readback(")
    + TESTS
)
cc = Path(os.environ["CC"]).resolve()
env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="alias-readback-admission-") as tmp:
    check = Path(tmp) / "check.c"
    exe = check.with_suffix(".exe")
    check.write_text(source, encoding="utf-8")
    flags = [] if joint else ["-Wno-unused-function"]
    subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", *flags,
                    "-fsanitize=undefined", "-fsanitize-trap=all", "-I", str(headers),
                    str(check), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

#define VK_NO_PROTOTYPES
#include <assert.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include "vk.h"
#include "vk_exec.h"
#include "resource_command_effects.h"
#include "resource_image_write.h"

#define TARGET ((VkImage)(uintptr_t)41)
#define SOURCE ((VkImage)(uintptr_t)42)
#define SCRATCH ((VkImage)(uintptr_t)43)
#define ALLOCATION ((VmaAllocation)(uintptr_t)57)
enum Failure {
    DECLARE_STALE, DECLARE_CAPACITY, SEAL_FAILURE, DECLARE_AND_SEAL_FAILURE,
    CANCEL_BEFORE_RECORD, CANCEL_IN_PASS_END, STOP_BEFORE_RECORD, LOSE_DEVICE_IN_PASS_END,
    CANCEL_AFTER_SOURCE_PREPARE, STOP_AFTER_SCRATCH_ALLOC,
    STOP_DURING_CAPACITY_WAIT, LOSE_DEVICE_ALLOCATING_SCRATCH
};
enum { RECORDING_RESOURCE_UNAVAILABLE = 1, RECORDING_CANCELLED };
typedef struct { int key; } TexKey;
typedef struct { int placeholder; } Tex;
struct VkMemSpanCapture { int active; };
void vk_texture_alias_read_cancel(VkTextureAliasRead *alias) { assert(!alias); }
struct VkMemoryImageHold { int active; };
struct VkTexturePresentSource { int active; };
struct ResourceCommandEffects { int active; };
typedef struct {
    ResourceCommandEffects *commands;
    uint32_t opcode;
} EffectDeclaration;

static enum Failure fault;
static unsigned expected_opcode;
static int recording_failed, stopping, failed_case, scratch_live, scratch_releases, effect_releases;
static int native_records, target_releases, source_releases, declarations, pass_ends, barriers, scope_depth;
static unsigned fatal_calls;
static jmp_buf fatal_return;
static VkExec exec;
static VkCtx device;
static VkMemSpanCapture buffer_source;
static VkMemoryImageHold target_hold;
static VkTexturePresentSource texture_source;
static ResourceCommandEffects commands;
static EffectDeclaration effect;
volatile int32_t vk_lost;

/* Private target layout */

VkCtx *vk_ctx(void) { return &device; }
int vk_exec_recording_failed(void) { return recording_failed != 0; }
int vk_sync_stopping(void) { return stopping; }
static void cancel_recording(void) {
    if (!recording_failed) recording_failed = RECORDING_CANCELLED;
}
void vk_exec_required_graphics_unavailable(VkExec *ex, uint32_t opcode, const char *message) {
    assert(ex == &exec && opcode == expected_opcode && message);
    recording_failed = RECORDING_RESOURCE_UNAVAILABLE;
}
_Noreturn void rt_fatal(const char *message, ...) {
    assert((fault == SEAL_FAILURE || fault == DECLARE_AND_SEAL_FAILURE) && message);
    ++fatal_calls;
    longjmp(fatal_return, 1);
}
RcRegistry *rc_registry_runtime(void) { return (RcRegistry *)(uintptr_t)1; }
RcRegistryResult rc_registry_gpu_capture_end(RcRegistry *r, RcRegistryGpuCapture *capture) {
    (void)r;
    (void)capture;
    abort(); // These fault scenarios exercise device-backed transfers only.
}
RcRegistryPosition vk_exec_resource_position(void) {
    return (RcRegistryPosition){exec.fence, exec.resource_op * 2};
}
RcRegistryResult rc_registry_declare_through(RcRegistry *r, RcRegistryPosition position) {
    assert(r == rc_registry_runtime() && position.scene == exec.fence && position.ordinal == 3);
    return fault == SEAL_FAILURE || fault == DECLARE_AND_SEAL_FAILURE ? RC_REGISTRY_STALE : RC_REGISTRY_OK;
}
static void image_write_input_position(VkExec *ex) {
    assert(ex == &exec && !ex->resource_next);
    ex->resource_op = 1;
}
static EffectDeclaration *effect_declaration_new(VkExec *ex, RcRegistryPosition position) {
    assert(ex == &exec && position.scene == ex->fence && position.ordinal == 3 && !commands.active);
    commands.active = 1;
    effect.commands = &commands;
    return &effect;
}
void vk_exec_effects_release(void *declaration) {
    if (!declaration) return;
    assert(declaration == &effect && commands.active && effect.opcode == expected_opcode);
    commands.active = 0;
    ++effect_releases;
}
static void begin_effect(void *declaration) {
    if (declaration) assert(declaration == &effect && commands.active);
    if (fault == CANCEL_BEFORE_RECORD) cancel_recording();
    if (fault == STOP_BEFORE_RECORD) stopping = 1;
}
int vk_contact_pending(void) { return 0; }
int vk_texture_projected_transfer_current(const VkTextureProjectedTransfer *target) {
    assert(target && target->hold == &target_hold && target_hold.active);
    return 1;
}
RcRegistryResult resource_effects_add_pinned_image(ResourceCommandEffects *current, uint64_t image,
                                                  const RcInterval *mapping, uint32_t count) {
    assert(current == &commands && commands.active && image == (uint64_t)(uintptr_t)TARGET);
    assert(mapping && count == 1 && mapping->range.size == 128);
    ++declarations;
    return fault == DECLARE_CAPACITY || fault == DECLARE_AND_SEAL_FAILURE ? RC_REGISTRY_CAPACITY : RC_REGISTRY_OK;
}
void vk_mem_image_release(VkMemoryImageHold *hold) {
    if (!hold) return;
    assert(hold == &target_hold && hold->active);
    hold->active = 0;
    ++target_releases;
}
void vk_mem_span_capture_release(VkMemSpanCapture **hold) {
    if (!*hold) return;
    assert(*hold == &buffer_source && buffer_source.active);
    buffer_source.active = 0;
    *hold = NULL;
    ++source_releases;
}
void vk_texture_present_release(VkTexturePresentSource *hold) {
    if (!hold) return;
    assert(hold == &texture_source && hold->active);
    hold->active = 0;
    ++source_releases;
}
void vk_texture_write_target_cancel(VkTextureWriteTarget *other) { assert(!other); }
void vk_texture_restore_cancel(VkTextureRestore *other) { assert(!other); }
void resource_core_vma_destroy_image(VmaAllocator allocator, VkImage image, VmaAllocation allocation) {
    assert(allocator == device.vma && image == SCRATCH && allocation == ALLOCATION && scratch_live);
    scratch_live = 0;
    ++scratch_releases;
}
static void target_capture(VkExecProjectedTransfer *plan) {
    VkTextureProjectedTransfer *target = calloc(1, sizeof(*target));
    assert(target && !target_hold.active);
    target->read = vk_exec_resource_position();
    target->hold = &target_hold;
    target_hold.active = 1;
    target->image = TARGET;
    target->mappings = 1;
    target->mapping = calloc(1, sizeof(*target->mapping));
    target->covered = calloc(1, sizeof(*target->covered));
    assert(target->mapping && target->covered);
    target->mapping->range.size = 128;
    target->covered->range.size = 8;
    plan->target = target;
}
static int projected_b2t_capture(VkExec *ex, const p_copy_b2t *payload, VkExecProjectedTransfer *plan) {
    assert(ex == &exec && payload && plan->opcode == OP_COPY_B2T);
    target_capture(plan);
    plan->buffer.source_hold = &buffer_source;
    buffer_source.active = 1;
    return 1;
}

int vk_texture_transfer_input_current(const VkTextureTransferInput *input) {
    assert(!input->texture);
    return 1;
}

static int projected_t2t_capture(VkExec *ex, const p_copy_t2t *payload, VkExecProjectedTransfer *plan) {
    assert(ex == &exec && payload && plan->opcode == OP_COPY_T2T);
    target_capture(plan);
    plan->image.source.image = SOURCE;
    plan->image.source_hold = &texture_source;
    texture_source.active = 1;
    plan->image.scratch_recipe.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    return 1;
}
RcRegistryResult vk_exec_buffer_to_texture_prepare_source(VkExec *ex, VkExecBufferToTexture *copy) {
    assert(ex == &exec && ex->resource_op == 1 && copy->source_hold == &buffer_source && buffer_source.active);
    if (fault == CANCEL_AFTER_SOURCE_PREPARE) cancel_recording();
    return RC_REGISTRY_OK;
}
int vk_exec_buffer_to_texture_source_current(const VkExecBufferToTexture *copy) {
    assert(copy->source_hold == &buffer_source && buffer_source.active);
    return fault != DECLARE_STALE || exec.resource_next == 0;
}
int vk_texture_transfer_read_current(const VkTexturePresentSource *source, VkImage image) {
    assert(source == &texture_source && source->active && image == SOURCE);
    return fault != DECLARE_STALE || exec.resource_op == 0;
}
VkResult vk_exec_texture_to_texture_allocate(VkExecTextureToTexture *copy, RcVmaDependency *dependency) {
    assert(copy->source_hold == &texture_source && texture_source.active && dependency && !scratch_live);
    if (fault == STOP_DURING_CAPACITY_WAIT) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (fault == LOSE_DEVICE_ALLOCATING_SCRATCH) return VK_ERROR_DEVICE_LOST;
    copy->scratch = SCRATCH;
    copy->scratch_memory = ALLOCATION;
    scratch_live = 1;
    if (fault == STOP_AFTER_SCRATCH_ALLOC) stopping = 1;
    return VK_SUCCESS;
}
VkSyncProgressResult vk_exec_texture_capacity(VkExec *ex, const RcVmaDependency *dependency, VkResult result,
        int (*changed)(void *), void *context, uint64_t *observed) {
    assert(ex == &exec && dependency && result == VK_ERROR_OUT_OF_DEVICE_MEMORY && changed && context && observed);
    assert(fault == STOP_DURING_CAPACITY_WAIT && !changed(context));
    stopping = 1;
    return VK_SYNC_PROGRESS_STOPPED;
}
void vk_device_lost_check(VkResult result, const char *message) {
    assert(result == VK_ERROR_DEVICE_LOST && message && fault == LOSE_DEVICE_ALLOCATING_SCRATCH);
    ++fatal_calls;
    longjmp(fatal_return, 1);
}
void resource_image_scope_begin(ResourceImageScope *scope, uint64_t fence) {
    assert(scope && fence == exec.fence && !scope_depth);
    scope_depth = 1;
}
void resource_image_scope_end(ResourceImageScope *scope) {
    assert(scope && scope_depth);
    scope_depth = 0;
}
void vk_exec_pass_end(VkExec *ex) {
    assert(ex == &exec);
    ++pass_ends;
    if (fault == CANCEL_IN_PASS_END) cancel_recording();
    if (fault == LOSE_DEVICE_IN_PASS_END) vk_lost = 1;
}
void vk_exec_barrier(VkExec *ex) { assert(ex == &exec); ++barriers; }
VkImage vk_texture_projected_transfer_image(const VkTextureProjectedTransfer *target) {
    assert(target && target->image == TARGET);
    return target->image;
}
// The producer-specific emitter boundary sees a real plan. For these refusal cases any call here is a failure.
static void projected_b2t_record(VkExec *ex, VkExecProjectedTransfer *plan, VkImage destination) {
    assert(ex == &exec && plan->buffer.source_hold == &buffer_source && destination == TARGET);
    ++native_records;
}
static void projected_t2t_record(VkExec *ex, VkExecProjectedTransfer *plan, VkImage destination) {
    assert(ex == &exec && plan->image.source_hold == &texture_source && plan->image.scratch == SCRATCH &&
           destination == TARGET);
    ++native_records;
}

/* Production functions */
/* Production replay tail */

static const char *const labels[] = {
    "stale declaration input", "writer declaration capacity refusal", "output seal refusal",
    "both declaration and output seal refused", "recording canceled after declaration",
    "recording canceled during pass end", "submission stopped after declaration",
    "device lost during pass end", "recording canceled during source preparation",
    "submission stopped immediately after scratch allocation",
    "submission stopped while waiting for scratch capacity", "terminal device loss allocating scratch"
};

static VkExecProjectedTransfer prepared;

static void run(enum Failure phase, unsigned opcode) {
    fault = phase;
    expected_opcode = opcode;
    exec = (VkExec){.fence=19};
    prepared = (VkExecProjectedTransfer){0};
    recording_failed = stopping = vk_lost = scratch_live = native_records = 0;
    target_releases = source_releases = scratch_releases = effect_releases = declarations = pass_ends = barriers = scope_depth = 0;
    buffer_source.active = target_hold.active = texture_source.active = commands.active = fatal_calls = 0;
    const p_copy_b2t b2t = {0};
    const p_copy_t2t t2t = {0};
    const void *payload = opcode == OP_COPY_B2T ? (const void *)&b2t : &t2t;
    volatile int status = -1;
    int during_preparation = phase >= CANCEL_AFTER_SOURCE_PREPARE;
    int fatal = phase == SEAL_FAILURE || phase == DECLARE_AND_SEAL_FAILURE || phase == LOSE_DEVICE_ALLOCATING_SCRATCH;
    if (!setjmp(fatal_return)) {
        status = prepare_ordered_projected_transfer(&exec, opcode, payload, &prepared);
        if (!during_preparation || status == 1) replay_tail(&exec, prepared);
    }
    int expected_scratch = opcode == OP_COPY_T2T && phase != STOP_DURING_CAPACITY_WAIT && phase != LOSE_DEVICE_ALLOCATING_SCRATCH;
    int expected_declares = !during_preparation && phase != DECLARE_STALE;
    int okay = target_releases == 1 && source_releases == 1 && scratch_releases == expected_scratch &&
        effect_releases == !during_preparation && declarations == expected_declares &&
        native_records == 0 && fatal_calls == (unsigned)fatal && !scope_depth && !commands.active &&
        !target_hold.active && !buffer_source.active && !texture_source.active && !scratch_live &&
        (during_preparation ? status < 0 : status == 1);
    if ((phase == CANCEL_IN_PASS_END || phase == LOSE_DEVICE_IN_PASS_END) && barriers) okay = 0;
    if (during_preparation && status < 0 && !fatal) {
        vk_exec_projected_transfer_cancel(&prepared);
        okay &= target_releases == 1 && source_releases == 1 && scratch_releases == expected_scratch;
    }
    printf("%s %s %s [target %d source %d scratch %d effect %d declares %d emits %d fatal %u prep %d]\n",
           okay ? "PASS" : "FAIL", opcode == OP_COPY_B2T ? "B2T" : "T2T", labels[phase],
           target_releases, source_releases, scratch_releases, effect_releases, declarations, native_records, fatal_calls, status);
    failed_case += !okay;
}

int main(void) {
    for (enum Failure phase = DECLARE_STALE; phase <= LOSE_DEVICE_IN_PASS_END; phase++) {
        run(phase, OP_COPY_B2T);
        run(phase, OP_COPY_T2T);
    }
    run(CANCEL_AFTER_SOURCE_PREPARE, OP_COPY_B2T);
    run(STOP_AFTER_SCRATCH_ALLOC, OP_COPY_T2T);
    run(STOP_DURING_CAPACITY_WAIT, OP_COPY_T2T);
    run(LOSE_DEVICE_ALLOCATING_SCRATCH, OP_COPY_T2T);
    return failed_case != 0;
}

"""Exercise the selected shield resolve and its ordered DLAA recording boundary."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ordered_dispatch_preparation_check import between, function


ROOT = Path(__file__).resolve().parents[1]

PREAMBLE = r'''
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HANDLE(type, n) ((type)(uintptr_t)(n))
#define MAX_BINDINGS 128
#define SUPPORT_BYTES 16
#define ZERO_BYTES 16
enum { GFX_RES_UBO, VK_PIPE_SUPPORT, NVN_STAGE_FS = 1, NVN_FORMAT_RG16F = 2, NVN_DEPTH_FUNC_EQUAL = 3,
       NVN_INDEX_TYPE_UNSIGNED_SHORT = 1, VK_EXEC_DIRTY_ALL = 63, MOTION_PLAYER = 1, RC_BUDGET_BUFFER = 3,
       VMA_MEMORY_USAGE_AUTO = 0, VMA_ALLOCATION_CREATE_MAPPED_BIT = 1,
       VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT = 2, GRAPHICS_PAYLOAD_SUPPORT = 5,
       RC_BUDGET_RAM = 0, RC_BUDGET_COMMIT = 1,
       RC_BUDGET_RECLAIM_PENDING = 1, RC_BUDGET_RECLAIM_BUSY = 2, RC_BUDGET_RECLAIM_DESTROYED = 4,
       CAM_BLOCK = 1024, OCAM_BLOCK = 2048, PW_HOOKS = 10, T_MV1 = 1 };
typedef struct { uint64_t model, part, lifetime; unsigned kind; } MotionOwner;
typedef struct { uint64_t ibuf; int count, base_vertex, itype; } p_draw_idx;
typedef struct { uint32_t stride; } Vstream;
typedef struct {
    int vs, fs, feedback, color_fmt[8];
    uint32_t vattr[16];
    Vstream vstream[16];
    struct { unsigned mask; } chmask;
    struct { int logic_op; } color;
    struct { int dw, df; } ds;
} VkPipeState;
typedef struct { int slot; } GfxShaderRes;
typedef struct {
    uint16_t binding;
    uint8_t stage, kind;
    VkDescriptorType type;
    uint16_t count;
    const GfxShaderRes *res;
    uint8_t set;
} VkPipeBinding;
typedef struct {
    const VkPipeBinding *bindings;
    uint32_t n_bindings, n_sets;
    VkPipelineLayout layout;
    VkDescriptorSetLayout set_layout, set_layout1;
    int pool, push, push1;
} VkPipeInfo;

typedef struct { VkBuffer b[17]; VkDeviceSize o[17], sz[17], str[17]; } MoVb;
static struct { VkBuffer zbuf; } Z = {(VkBuffer)(uintptr_t)101};
typedef void *VmaAllocation;
typedef void *VmaAllocator;
typedef struct { unsigned usage, requiredFlags, flags; } VmaAllocationCreateInfo;
typedef struct { void *pMappedData; } VmaAllocationInfo;
typedef struct { unsigned producer; } RcVmaRequest;
typedef struct { uint64_t amount[2]; unsigned domains; } RcAdmissionVector;
typedef struct { unsigned ready, records_available; uint64_t available; } RcBudgetReadiness;
typedef enum { RC_ADMISSION_OK, RC_ADMISSION_REFUSED, RC_ADMISSION_NO_RECORDS } RcAdmissionStatus;
typedef struct { struct { void *ledger; } token; } RcBudgetTicket;
typedef struct { RcAdmissionVector peak; int selected; } RcVmaDependency;
typedef struct { uint64_t event; unsigned reclaim; } RcBudgetProgress;
typedef struct {
    VmaAllocator vma;
    VkDevice device;
    struct { VkPhysicalDeviceLimits limits; } props;
} VkCtx;
static VkCtx context = {.props.limits = {.minUniformBufferOffsetAlignment = 256, .maxUniformBufferRange = 65536}};

typedef struct { VkBuffer buf; uint8_t *map; VkDeviceSize used, size; } Chunk;
typedef struct UpdateSlab { struct UpdateSlab *next; Chunk chunk; } UpdateSlab;
typedef struct CbRes {
    UpdateSlab *graphics_slabs;
    struct CbRes *graphics_prev, *graphics_next;
} CbRes;
typedef struct DrawCtx { CbRes *res; } DrawCtx;
typedef struct {
    VkCommandBuffer cb;
    uint64_t fence, recorded;
    unsigned dirty;
    DrawCtx *dc;
    unsigned ssbo[5], ubo[5];
} VkExec;
typedef struct { VkExec *ex; uint64_t bytes; } VkDrawScratchWaitContext;
typedef struct { VkDeviceSize bytes, alignment, offsets[5]; unsigned slots; } GraphicsPayloadDemand;
typedef struct VkExecGraphicsScratch VkExecGraphicsScratch;
struct VkExecGraphicsScratch {
    GraphicsPayloadDemand demand;
    CbRes *owner;
    UpdateSlab *slab;
    VkDeviceSize start, prior_used;
    VkExecGraphicsScratch *cohort;
    int exposed, fresh;
};
typedef struct VkShieldResolve VkShieldResolve;
typedef enum { VK_SYNC_PROGRESS_COMPLETE, VK_SYNC_PROGRESS_STOPPED, VK_SYNC_PROGRESS_UNAVAILABLE } VkSyncProgressResult;

typedef struct { VkCommandBuffer slot; } ReplayChunk;
static struct { uint64_t out, drained; } Q;
static ReplayChunk *temporal_hook_chunk, *rw_chunk;
static VkExec *x_cur;
static int recording_scene, earlier_held, handed, parked, stopping, lost, failed, shortage, stop_while_waiting;
static unsigned parks, attempts, waits, pipeline_calls, no_capacity, cancellations, completes, scratch_releases;
static unsigned host_attempts, host_cancels, host_destroys, host_shortage, host_state;
static uint64_t host_peak, host_actual;
static int motion_enabled = 1;
static unsigned native_geometry_allocations, binding_allocations, binds, begins, ends, draws, camera_calls, sdk_calls, clears, new_cb;
static uint64_t clock_ns, frame = 5, callback_done;
static VkDeviceSize demanded;
static VkExecGraphicsScratch *graphics_payload_current, *graphics_payload_bound;
static CbRes *graphics_head, *graphics_tail;
static DrawCtx own_dc, *dc = &own_dc;
static CbRes owner;
static int spare_lock, in_rendering, submitted;
static VkBuffer submitted_buffer;
static uint8_t *submitted_map;
static const VkCommandBuffer command_buffer = (VkCommandBuffer)(uintptr_t)139;

static int vk_sync_stopping(void) { return stopping; }
static int vk_device_is_lost(void) { return lost; }
static int vk_exec_recording_failed(void) { return failed; }
static int progress_stopped(void) { return stopping || lost; }
static VkCtx *vk_ctx(void) { return &context; }
static int motion_owner_enabled(void) { return motion_enabled; }
static int shield_trace(void) { return 0; }
static uint64_t vk_dlaa_scene_frame(void) { return frame; }
static uint64_t vk_sync_fence_done(void) { return callback_done; }
static uint64_t vk_sync_handed_fence(void) { return handed ? 7 : 0; }
static uint64_t kern_now_ns(void) { return clock_ns; }
static void vk_device_lost_check(VkResult result, const char *where) {
    assert(where);
    if (result == VK_ERROR_DEVICE_LOST) lost = 1;
}
static void vk_exec_required_temporal_unavailable(VkExec *ex, const char *why) {
    assert(ex == x_cur && *why && !parked && !in_rendering);
    assert((strstr(why, "metadata") != NULL) == (host_shortage != 0));
    ++no_capacity;
    failed = 1;
}
static void vk_exec_cancel_temporal_preparation(void) {
    assert(!parked && !in_rendering);
    ++cancellations;
    failed = 1;
}
static void rt_log(const char *fmt, ...) { assert(fmt); }
static _Noreturn void rt_fatal(const char *why) { fprintf(stderr, "%s\n", why); abort(); }

static void group_drain(VkExec *ex, int why) {
    assert(x_cur == ex && why == PW_HOOKS && !rw_chunk);
    Q.drained = Q.out;
}
static uint64_t recorder_park(void) {
    assert(!parked && recording_scene && !in_rendering && !camera_calls && !sdk_calls);
    uint64_t scene = recording_scene;
    recording_scene = 0;
    ++parks;
    parked = 1;
    return scene;
}
static void recorder_resume(uint64_t scene) {
    assert(parked && !recording_scene && x_cur->cb == command_buffer);
    parked = 0;
    recording_scene = (int)scene;
}
static void hold_release(uint64_t scene, int a, int b) {
    assert(scene == x_cur->fence && !a && !b && !recording_scene && parked);
    if (rw_chunk) assert(rw_chunk == temporal_hook_chunk && !rw_chunk->slot && Q.out - Q.drained == 1);
    if (earlier_held) earlier_held = 0, handed = 1;
}
static void plat_mutex_lock(int *mutex) { assert(!*mutex); *mutex = 1; }
static void plat_mutex_unlock(int *mutex) { assert(*mutex); *mutex = 0; }
static CbRes *update_current(VkExec *ex) { return ex->dc->res; }
static CbRes *cb_res(VkExec *ex) { ex->dc->res = &owner; return &owner; }
static VkResult update_slab_create(uint64_t bytes, int allow_growth, UpdateSlab **out, RcVmaDependency *dependency) {
    assert(!in_rendering && !camera_calls && allow_growth && host_state == 1);
    ++attempts;
    if (demanded) assert(demanded == bytes);
    demanded = bytes;
    if (shortage && callback_done != 7) {
        *dependency = (RcVmaDependency){.selected = 1, .peak.domains = 1};
        return VK_NOT_READY;
    }
    UpdateSlab *slab = calloc(1, sizeof *slab);
    assert(slab);
    slab->chunk = (Chunk){.buf = HANDLE(VkBuffer, 55), .map = calloc(1, bytes), .size = bytes};
    assert(slab->chunk.map);
    *out = slab;
    return VK_SUCCESS;
}
static int scratch_placement_fits(uint64_t bytes, const RcVmaDependency *dependency) {
    return bytes == demanded && dependency->selected;
}
static void update_release(UpdateSlab *slab) {
    assert(slab && !slab->next && !in_rendering);
    ++scratch_releases;
    free(slab->chunk.map);
    free(slab);
}
static void vk_c0_fill(uint8_t *data, int stage, unsigned ssbo, unsigned ubo, const uint32_t dim[3], uint32_t base) {
    (void)data; (void)stage; (void)ssbo; (void)ubo; (void)dim; (void)base;
    abort(); // A resolve must copy the captured C0 rather than use the active draw state.
}
static RcAdmissionStatus resource_core_budget_reserve(unsigned producer, int required,
                                                       const RcAdmissionVector *peak, RcBudgetTicket *ticket) {
    assert(producer == RC_BUDGET_BUFFER && required && !host_state && !ticket->token.ledger && !camera_calls);
    assert(!peak->domains && peak->amount[RC_BUDGET_RAM] && peak->amount[RC_BUDGET_RAM] == peak->amount[RC_BUDGET_COMMIT]);
    ++host_attempts;
    if (host_peak) assert(host_peak == peak->amount[RC_BUDGET_RAM]);
    host_peak = peak->amount[RC_BUDGET_RAM];
    if (host_shortage && callback_done != 7)
        return host_shortage == 1 ? RC_ADMISSION_REFUSED : RC_ADMISSION_NO_RECORDS;
    host_state = 1;
    ticket->token.ledger = &host_state;
    return RC_ADMISSION_OK;
}
static int rc_admission_accounted(RcAdmissionStatus result) { return result == RC_ADMISSION_OK; }
static RcAdmissionStatus resource_core_budget_materialize(RcBudgetTicket *ticket, const RcAdmissionVector *actual) {
    assert(host_state == 1 && ticket->token.ledger == &host_state && !parked && !camera_calls);
    assert(actual->amount[RC_BUDGET_RAM] == actual->amount[RC_BUDGET_COMMIT] &&
           actual->amount[RC_BUDGET_RAM] <= host_peak && !actual->domains);
    host_actual = actual->amount[RC_BUDGET_RAM];
    host_state = 2;
    return RC_ADMISSION_OK;
}
static RcAdmissionStatus resource_core_budget_cancel(RcBudgetTicket *ticket) {
    assert(host_state == 1 && ticket->token.ledger == &host_state && !host_actual);
    host_state = 0;
    ++host_cancels;
    ticket->token.ledger = NULL;
    return RC_ADMISSION_OK;
}
static RcAdmissionStatus resource_core_budget_retire(RcBudgetTicket *ticket) {
    assert(host_state == 2 && ticket->token.ledger == &host_state && host_actual);
    host_state = 3;
    return RC_ADMISSION_OK;
}
static RcAdmissionStatus resource_core_budget_destroyed(RcBudgetTicket *ticket) {
    assert(host_state == 3 && ticket->token.ledger == &host_state);
    host_state = 0;
    ++host_destroys;
    ticket->token.ledger = NULL;
    return RC_ADMISSION_OK;
}
static void resource_core_budget_readiness(int required, RcBudgetReadiness *now) {
    assert(required && parked && !recording_scene);
    now->ready = callback_done == 7;
    now->records_available = host_shortage != 2 || now->ready;
    now->available = host_shortage != 1 || now->ready ? host_peak : host_peak - 1;
}
static int resource_core_budget_peak_ready(const RcAdmissionVector *peak, const RcBudgetReadiness *now) {
    assert(!peak->domains && peak->amount[0] == host_peak && peak->amount[1] == host_peak && host_shortage);
    return now->records_available && now->available >= peak->amount[0];
}
static void resource_core_budget_service(int required, const RcAdmissionVector *peak, RcBudgetProgress *progress) {
    assert(required && parked);
    assert(host_shortage ? (!peak->domains && peak->amount[0] == host_peak && peak->amount[1] == host_peak) :
                           (peak->domains == 1 && !peak->amount[0] && !peak->amount[1]));
    *progress = (RcBudgetProgress){.event = 42};
}
static int vk_exec_draw_graphics_scratch_spare_ready(void *context) {
    const VkDrawScratchWaitContext *wait = context;
    assert(wait->ex == x_cur && wait->bytes == demanded && parked);
    return 0;
}
static int resource_core_vma_dependency_ready(const RcVmaDependency *dependency, const RcBudgetReadiness *now) {
    assert(dependency->selected);
    return now->ready;
}
static uint64_t vk_exec_draw_graphics_scratch_prior_useful(VkExec *ex, uint64_t bytes, const RcVmaDependency *dependency,
                                                          const RcBudgetReadiness *now) {
    assert(ex == x_cur && bytes == demanded && dependency->selected && !now->ready && parked && !recording_scene);
    return handed ? 7 : 0;
}
static uint64_t vk_exec_draw_scratch_prior_useful(VkExec *ex, uint64_t bytes, const RcVmaDependency *dependency,
                                                const RcBudgetReadiness *now) {
    assert(ex == x_cur && bytes == demanded && dependency->selected && !now->ready && parked);
    return 0;
}
static void resource_core_budget_wait_since(uint64_t event, uint32_t milliseconds) {
    assert(event == 42 && milliseconds && parked && !recording_scene && handed && !callback_done);
    ++waits;
    clock_ns += milliseconds * UINT64_C(1000000);
    if (stop_while_waiting) stopping = 1;
    else callback_done = 7;
}
static VkResult resource_core_vma_create_buffer(const RcVmaRequest *request, VmaAllocator allocator,
                                               const VkBufferCreateInfo *info, const VmaAllocationCreateInfo *placement,
                                               VkBuffer *buffer, VmaAllocation *alloc, VmaAllocationInfo *map, void *extra) {
    (void)allocator; (void)extra;
    assert(request->producer == RC_BUDGET_BUFFER && !in_rendering && !camera_calls);
    assert(info->usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    assert(placement->requiredFlags == VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    *buffer = HANDLE(VkBuffer, 81);
    *alloc = map->pMappedData = calloc(1, info->size);
    ++native_geometry_allocations;
    return VK_SUCCESS;
}
static void resource_core_vma_destroy_buffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation alloc) {
    (void)allocator;
    assert(buffer == HANDLE(VkBuffer, 81));
    free(alloc);
}

static VkPipeBinding bindings[7], rejected[7];
static VkPipeInfo info = {.bindings = bindings, .n_bindings = 7, .n_sets = 2, .push1 = 1};
static VkPipeInfo rejected_info = {.bindings = rejected, .n_bindings = 7, .n_sets = 2, .push1 = 1};
static VkPipeline vk_pipeline_graphics_async(const VkPipeState *state, const VkPipeInfo **out, int *pending, uint32_t ms) {
    assert(!ms && !parked && !camera_calls && !in_rendering);
    assert(state->color_fmt[0] == NVN_FORMAT_RG16F && state->ds.df == NVN_DEPTH_FUNC_EQUAL);
    ++pipeline_calls;
    *pending = state->fs == 2;
    *out = state->fs == 3 ? &rejected_info : &info;
    return *pending ? VK_NULL_HANDLE : HANDLE(VkPipeline, 99);
}
static int nvn_stage_of(int stage) { return stage; }
static void support_fill(uint8_t *data, float w, float h, float size) {
    const float v[4] = {w, h, size, 1};
    memcpy(data, v, sizeof v);
}
static VkDescriptorSet alloc_set(VkExec *ex, VkDescriptorSetLayout layout) {
    (void)layout;
    assert(ex == x_cur && !in_rendering && !camera_calls && !parked && host_state == 2);
    return HANDLE(VkDescriptorSet, 120 + ++binding_allocations);
}
static int vk_pipeline_vertex_input_dynamic(void) { return 0; }
static PFN_vkCmdPushDescriptorSetKHR vk_pipeline_push_fn(void) { return NULL; }

static void vkCmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo *ri) {
    assert(cb == command_buffer && !in_rendering && !parked && camera_calls == 1 && !sdk_calls && host_state == 2);
    assert(ri->pColorAttachments->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD && ri->pDepthAttachment->imageView);
    in_rendering = 1;
    ++begins;
}
static void vkCmdEndRendering(VkCommandBuffer cb) {
    assert(cb == command_buffer && in_rendering);
    in_rendering = 0;
    ++ends;
}
static void vk_exec_draw_new_cb(void) { assert(!in_rendering && x_cur->cb == command_buffer); ++new_cb; }
static void vkCmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint point, VkPipeline pipeline) {
    assert(cb == command_buffer && pipeline);
    assert(point == (in_rendering ? VK_PIPELINE_BIND_POINT_GRAPHICS : VK_PIPELINE_BIND_POINT_COMPUTE));
}
static void vk_pipeline_set_dynamic(VkCommandBuffer cb, const VkPipeState *state) {
    assert(cb == command_buffer && in_rendering && state->ds.df == NVN_DEPTH_FUNC_EQUAL);
}
static void vkCmdSetViewportWithCount(VkCommandBuffer cb, uint32_t n, const VkViewport *viewport) {
    assert(cb == command_buffer && n == 1 && viewport->height == -100 && in_rendering);
}
static void vkCmdSetScissorWithCount(VkCommandBuffer cb, uint32_t n, const VkRect2D *scissor) {
    assert(cb == command_buffer && n == 1 && scissor->extent.width == 200 && in_rendering);
}
static void vkCmdSetDepthBias(VkCommandBuffer cb, float a, float b, float c) {
    assert(cb == command_buffer && a == 1 && b == 2 && c == 3 && in_rendering);
}
static VkDeviceSize first_c0, first_c1, first_current, first_previous;
static void check_uniforms(uint32_t count, const VkWriteDescriptorSet *writes) {
    assert(count == 6 && owner.graphics_slabs && in_rendering);
    const Chunk *chunk = &owner.graphics_slabs->chunk;
    const unsigned start_byte[6] = {0, 0x11, 0x12, 0x33, 0x44, 0x55};
    const uint32_t bytes[6] = {SUPPORT_BYTES, 2048, 8, 512, 512, 512};
    VkDeviceSize offsets[6];
    for (uint32_t i = 0; i < count; ++i) {
        const VkDescriptorBufferInfo *input = writes[i].pBufferInfo;
        assert(input->buffer == chunk->buf && input->offset % 256 == 0 && input->range == bytes[i]);
        assert(input->offset + input->range <= chunk->size && writes[i].dstSet && writes[i].descriptorCount == 1);
        const uint8_t *data = chunk->map + input->offset;
        offsets[i] = input->offset;
        if (!i) {
            const float support[4] = {3, 4, 5, 1};
            assert(!memcmp(data, support, sizeof support));
        } else {
            unsigned byte = draws == 1 && i == 3 ? 0x77 : start_byte[i];
            for (uint32_t j = 0; j < input->range; ++j) assert(data[j] == byte);
        }
    }
    assert(offsets[3] != offsets[4] && offsets[3] != offsets[5] && offsets[4] != offsets[5]);
    if (!draws) {
        first_c0 = offsets[1]; first_c1 = offsets[2]; first_current = offsets[3]; first_previous = offsets[5];
    } else {
        assert(offsets[1] == first_c0 && offsets[2] != first_c1);
        assert(offsets[3] != first_current && offsets[5] == first_previous);
    }
}
static void vkUpdateDescriptorSets(VkDevice device, uint32_t count, const VkWriteDescriptorSet *writes, uint32_t copies,
                                   const VkCopyDescriptorSet *copy) {
    (void)device; (void)copy;
    assert(!copies);
    check_uniforms(count, writes);
}
static void vkCmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint point, VkPipelineLayout layout,
                                    uint32_t first, uint32_t count, const VkDescriptorSet *sets,
                                    uint32_t dynamics, const uint32_t *offsets) {
    (void)layout; (void)offsets;
    assert(cb == command_buffer && in_rendering && !first && count == 1 && *sets && !dynamics);
    assert(point == VK_PIPELINE_BIND_POINT_GRAPHICS);
    ++binds;
}
static const VkBuffer primary_mesh = (VkBuffer)(uintptr_t)81;
static VkBuffer expected_geometry[2];
static void vkCmdBindVertexBuffers2(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer *buffers,
                                    const VkDeviceSize *offsets, const VkDeviceSize *sizes, const VkDeviceSize *strides) {
    assert(cb == command_buffer && !first && n == 17 && strides[1] == 4 && in_rendering);
    assert(buffers[1] == expected_geometry[draws] && sizes[1] == 32 && sizes[16] == ZERO_BYTES);
    const uint8_t *data = buffers[1] == primary_mesh ? submitted_map : owner.graphics_slabs->chunk.map;
    for (unsigned i = 0; i < 32; ++i) assert(data[offsets[1] + i] == (draws ? 0x61 : 0x62));
}
static void vkCmdBindIndexBuffer(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkIndexType type) {
    assert(cb == command_buffer && in_rendering && buffer == expected_geometry[draws]);
    assert(type == (draws ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16));
    const uint8_t *data = buffer == primary_mesh ? submitted_map : owner.graphics_slabs->chunk.map;
    for (unsigned i = 0; i < (draws ? 12u : 6u); ++i) assert(data[offset + i] == (draws ? 0x21 : 0x22));
}
static void vkCmdDrawIndexed(VkCommandBuffer cb, uint32_t count, uint32_t instances, uint32_t first, int32_t base,
                              uint32_t first_instance) {
    assert(cb == command_buffer && in_rendering && count == 3 && instances == 1 && !first && !first_instance && base == 2);
    ++draws;
}

struct Size { VkImageView mv_view; int chars_ok, dist_ok; VkImage mv; };
static struct {
    uint64_t cam_addr;
    int cam_size, reset, cam_slot, cam_prev_ok;
    double cam_prev[16];
    VkPipeline mv_pipe;
    VkPipelineLayout mv_layout;
    struct { VkImage image; } depth;
    unsigned why[3];
} D;
static void vkCmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    assert(cb == command_buffer && !in_rendering && !parked && x && y && z == 1 && !camera_calls);
    ++camera_calls;
}
static void rec_push(PFN_vkCmdPushDescriptorSetKHR fn, VkCommandBuffer cb, VkPipelineBindPoint point,
                      VkPipelineLayout layout, uint32_t set, uint32_t n, const VkWriteDescriptorSet *writes) {
    (void)fn; (void)layout;
    assert(cb == command_buffer);
    if (point == VK_PIPELINE_BIND_POINT_COMPUTE) {
        assert(!in_rendering && !set && n == 10);
        return;
    }
    assert(in_rendering && set == 1 && n == 1 && writes[0].pBufferInfo->range == sizeof(float) * 4);
    const VkDescriptorBufferInfo *input = writes[0].pBufferInfo;
    const float expected[4] = {200, 100, draws ? 0.25f : 0.5f, 0};
    assert(!memcmp(owner.graphics_slabs->chunk.map + input->offset, expected, sizeof expected));
}
static void full_barrier(VkCommandBuffer cb) { assert(cb == command_buffer && !in_rendering); }
static void times_mark(VkCommandBuffer cb, int mark) { assert(cb == command_buffer && mark == T_MV1); }
static void vk_mvcheck_camera(VkCommandBuffer cb, VkImage mv, uint32_t w, uint32_t h, float jx, float jy) {
    (void)mv;
    assert(cb == command_buffer && w == 200 && h == 100 && !jx && !jy);
}
static void vkCmdClearColorImage(VkCommandBuffer cb, VkImage image, VkImageLayout layout,
                                 const VkClearColorValue *color, uint32_t count, const VkImageSubresourceRange *range) {
    (void)image; (void)layout; (void)color; (void)count; (void)range;
    assert(cb == command_buffer);
    ++clears;
}
static float char_bias(void) { return 0; }
'''

CHECKS = r'''
static uint8_t packets[2][4][4096];
static void make_packet(ShieldFrame *f, unsigned at, int which, int old) {
    ShieldDraw *p = &f->draws[at];
    *p = (ShieldDraw){.owner = {1, 10 + at, 3, MOTION_PLAYER}, .camera = 123, .depth = D.depth.image,
        .fence = old ? 7 : at ? 11 : 7, .order = at ? 20 : 10, .draw = {.ibuf = 12 + at, .count = 3,
        .base_vertex = 2, .itype = which ? 2 : NVN_INDEX_TYPE_UNSIGNED_SHORT},
        .state = {.vs = 42, .fs = at == 2 ? 3 : at == 3 ? 2 : 1},
        .viewport = {.x = which ? 0.25f : 0.5f, .y = 100, .height = -100},
        .scissor = {.extent = {200, 100}}, .bias = {1, 2, 3}, .support = {3, 4, 5}, .geom_size = 48,
        .indices = {0, which ? 12 : 6}, .data = packets[old][at]};
    p->state.vstream[1].stride = 4;
    p->vertex[1] = (ShieldSlice){16, 32};
    memset(p->data, which ? 0x21 : 0x22, p->indices.bytes);
    memset(p->data + 16, which ? 0x61 : 0x62, 32);
    static const unsigned slots[] = {0, 1, 3, 4};
    static const unsigned sizes[] = {2048, 8, 512, 512};
    const unsigned bytes[] = {0x11, 0x12, old ? 0x55 : which ? 0x77 : 0x33, 0x44};
    unsigned offset = 64;
    for (unsigned i = 0; i < sizeof slots / sizeof *slots; ++i) {
        p->bank[slots[i]] = (ShieldSlice){offset, sizes[i]};
        memset(p->data + offset, bytes[i], sizes[i]);
        offset += sizes[i];
    }
}

static void init_objects(VkExec *ex) {
    static const GfxShaderRes banks[] = {{0}, {1}, {3}, {4}, {18}, {5}};
    *ex = (VkExec){.cb = command_buffer, .fence = 11, .recorded = 9, .dc = dc};
    x_cur = ex;
    recording_scene = (int)ex->fence;
    D.depth.image = HANDLE(VkImage, 4);
    D.cam_addr = 123;
    D.cam_size = OCAM_BLOCK;
    D.cam_prev_ok = 1;
    D.cam_slot = 0;
    D.mv_pipe = HANDLE(VkPipeline, 17);
    D.reset = 0;
    D.why[1] = 0;
    shield_had_scene = 1;
    shield_last_frame = frame - 1;
    shield_last_camera = D.cam_addr;
    shield_last_depth = D.depth.image;
    shield_width = 200;
    shield_height = 100;
    for (unsigned i = 0; i < 2; ++i) {
        ShieldFrame *f = &shield_frames[(frame - i) & 1];
        *f = (ShieldFrame){.frame = frame - i, .count = 4};
        for (unsigned j = 0; j < 4; ++j) make_packet(f, j, j & 1, i);
    }
    ShieldFrame *now = &shield_frames[frame & 1];
    ShieldDraw tmp = now->draws[0];
    now->draws[0] = now->draws[1];
    now->draws[1] = tmp;
    for (unsigned i = 0; i < 7; ++i)
        bindings[i] = (VkPipeBinding){.binding = (uint16_t)i, .stage = i == 6 ? NVN_STAGE_FS : 0,
            .kind = i ? GFX_RES_UBO : VK_PIPE_SUPPORT, .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .count = 1, .res = i ? &banks[i - 1] : NULL, .set = i == 6};
    memcpy(rejected, bindings, sizeof bindings);
    rejected[5].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    expected_geometry[0] = expected_geometry[1] = HANDLE(VkBuffer, 55);
    demanded = 0;
    attempts = waits = parks = camera_calls = sdk_calls = clears = failed = pipeline_calls = 0;
    draws = begins = ends = new_cb = binds = binding_allocations = shortage = earlier_held = handed = stopping = 0;
    callback_done = 0;
    clock_ns = 0;
    parked = 0;
    stop_while_waiting = 0;
    assert(!host_state);
    host_attempts = host_shortage = 0;
    host_peak = host_actual = 0;
    motion_enabled = 1;
}

static void check_completion(VkExec *ex, int may_have_executed) {
    assert(ex->dc->res == &owner && owner.graphics_slabs);
    if (may_have_executed) {
        assert(submitted_map == owner.graphics_slabs->chunk.map);
        assert(owner.graphics_slabs->chunk.buf == submitted_buffer);
        assert(callback_done < ex->fence && submitted);
        assert(submitted_map[first_c0] == 0x11);
        callback_done = ex->fence;
        ++completes;
    } else {
        assert(!submitted && !camera_calls);
    }
    UpdateSlab *slab = graphics_detach(&owner);
    assert(slab && !slab->next);
    update_release(slab);
    ex->dc->res = NULL;
    submitted = 0;
}

static void check_ready(int pressure, int use_mesh) {
    VkExec ex;
    init_objects(&ex);
    if (pressure) {
        earlier_held = 1;
        shortage = pressure == 1;
        host_shortage = pressure > 1 ? (unsigned)pressure - 1 : 0;
        Q.out = 4, Q.drained = 3;
    }
    ShieldDraw *p = &shield_frames[frame & 1].draws[1];
    p->geometry_id = 1;
    player_mesh.entries[0].size = p->geom_size;
    player_mesh.entries[0].bytes = malloc(p->geom_size);
    assert(player_mesh.entries[0].bytes);
    memcpy(player_mesh.entries[0].bytes, p->data, p->geom_size);
    player_mesh.count = 1;
    if (use_mesh) {
        motion_geometry_promote(1);
        assert(player_mesh.entries[0].uploaded);
        expected_geometry[0] = primary_mesh;
        submitted_map = player_mesh.mapped;
    }
    ReplayChunk first = {0};
    if (pressure) temporal_hook_chunk = rw_chunk = &first;
    assert(eval_hook(&ex) == 1);
    assert(!first.slot && !parked && ex.cb == command_buffer && ex.recorded >= 13);
    assert(pressure ? parks == 1 && waits == 1 && callback_done == 7 : !parks && !waits);
    assert(attempts == (pressure == 1 ? 3u : 1u) && host_attempts == (pressure > 1 ? 3u : 1u));
    assert(host_peak == sizeof(VkShieldResolve) + 2 * sizeof(ShieldPreparedDraw) + sizeof(VkExecGraphicsScratch));
    assert(host_actual == host_peak && !host_state);
    assert(pipeline_calls == 4 && binding_allocations == 2 && binds == 2 && D.cam_slot == 1 && !clears);
    assert(camera_calls == 1 && begins == 1 && ends == 1 && draws == 2 && sdk_calls == 1 && new_cb == 1);
    assert(demanded == (use_mesh ? 5680u : 5936u));
    assert(shield_last_frame == frame && shield_last_camera == D.cam_addr && shield_last_depth == D.depth.image);
    assert(owner.graphics_slabs && owner.graphics_slabs->chunk.used == demanded);
    submitted = 1;
    submitted_buffer = owner.graphics_slabs->chunk.buf;
    submitted_map = owner.graphics_slabs->chunk.map;
    check_completion(&ex, 1);
    temporal_hook_chunk = rw_chunk = NULL;
    assert(player_mesh.entries[0].uploaded);
    resource_core_vma_destroy_buffer(NULL, player_mesh.buffer, player_mesh.allocation);
    free(player_mesh.entries[0].bytes);
    memset(&player_mesh, 0, sizeof player_mesh);
}

static void check_unavailable(int stop, int host) {
    VkExec ex;
    init_objects(&ex);
    unsigned failures = no_capacity, canceled = cancellations, dropped = host_cancels;
    shortage = !host;
    host_shortage = (unsigned)host;
    earlier_held = stop_while_waiting = stop;
    assert(eval_hook(&ex) == 0);
    assert(attempts == (host ? 0u : 2u) && parks == 1 && waits == (unsigned)stop);
    assert(host_attempts == (host ? 2u : 1u) && !host_actual && !host_state && host_cancels == dropped + !host);
    assert(no_capacity == failures + !stop && cancellations == canceled + stop);
    assert(!begins && !ends && !camera_calls && !sdk_calls && !clears && D.cam_slot == 0);
    assert(!owner.graphics_slabs && shield_last_frame == frame - 1 && pipeline_calls == 4 && !binding_allocations);
}

static void check_cancel_and_empty(void) {
    VkExec ex;
    init_objects(&ex);
    VkShieldResolve *plan = NULL;
    unsigned released = scratch_releases, retired = host_destroys;
    assert(vk_shield_resolve_prepare(&ex, D.cam_addr, D.depth.image, 200, 100, 0, &plan));
    assert(plan && owner.graphics_slabs && !camera_calls);
    shield_resolve_cancel(&plan);
    assert(!plan && !owner.graphics_slabs && scratch_releases == released + 1 && !begins);
    shield_resolve_cancel(&plan);
    assert(scratch_releases == released + 1 && host_destroys == retired + 1 && !host_state);
    assert(vk_shield_resolve_prepare(&ex, D.cam_addr, D.depth.image, 200, 100, 0, &plan));
    motion_enabled = 0;
    assert(!vk_shield_resolve_emit(&ex, &plan, D.cam_addr, D.depth.image, VK_NULL_HANDLE, VK_NULL_HANDLE, 200, 100, 0));
    assert(!plan && !owner.graphics_slabs && scratch_releases == released + 2 && host_destroys == retired + 2);
    assert(!host_state && binding_allocations == 4 && !begins);
    motion_enabled = 1;
    // Existing extents serve earlier recorded readers. A refused or canceled tail must preserve them.
    UpdateSlab *earlier = calloc(1, sizeof *earlier);
    assert(earlier);
    earlier->chunk = (Chunk){.buf = HANDLE(VkBuffer, 54), .map = calloc(1, 8192), .size = 8192, .used = 73};
    assert(earlier->chunk.map);
    memset(earlier->chunk.map, 0x7e, 73);
    graphics_slab_adopt(&owner, earlier);
    unsigned previous = attempts;
    assert(vk_shield_resolve_prepare(&ex, D.cam_addr, D.depth.image, 200, 100, 0, &plan));
    assert(attempts == previous && plan->scratch->slab == earlier && plan->scratch->start == 256);
    assert(earlier->chunk.used == plan->scratch->start + demanded);
    shield_resolve_cancel(&plan);
    assert(earlier->chunk.used == 73);
    for (unsigned i = 0; i < 73; ++i) assert(earlier->chunk.map[i] == 0x7e);
    check_completion(&ex, 0);

    D.reset = 1;
    assert(eval_hook(&ex));
    assert(attempts == previous && !begins && !ends && !clears && camera_calls == 1 && sdk_calls == 1);
    assert(shield_last_frame == frame);
    init_objects(&ex);
    shield_frames[(frame - 1) & 1].count = 0;
    assert(eval_hook(&ex));
    assert(!attempts && !pipeline_calls && !begins && !ends && camera_calls == 1 && sdk_calls == 1);
    assert(shield_last_frame == frame);
    init_objects(&ex);
    host_shortage = 1;
    for (unsigned i = 0; i < 4; ++i) shield_frames[frame & 1].draws[i].state.fs = 2;
    assert(eval_hook(&ex));
    assert(!host_attempts && !attempts && pipeline_calls == 4 && !begins && camera_calls == 1 && sdk_calls == 1);
}

int main(void) {
    check_ready(1, 0);
    check_ready(0, 1);
    check_ready(2, 0);
    check_ready(3, 0);
    assert(native_geometry_allocations == 4 && completes == 4 && host_destroys == 4 && !host_state);
    check_unavailable(0, 0);
    check_unavailable(1, 0);
    check_unavailable(0, 1);
    check_unavailable(1, 2);
    check_cancel_and_empty();
    puts("Shield motion: exact retained packets, separate host and native retries, one camera/render/SDK sequence, and completion passed");
}
'''


def main():
    runtime = ROOT / "runtime"
    shield = (runtime / "vk_shield_motion.inc").read_text(encoding="utf-8")
    draw = (runtime / "vk_exec_draw.c").read_text(encoding="utf-8")
    replay = (runtime / "vk_exec.c").read_text(encoding="utf-8")
    sync = (runtime / "vk_sync.c").read_text(encoding="utf-8")
    dlaa = (runtime / "vk_dlaa.c").read_text(encoding="utf-8")

    # Model Vulkan only at its call boundary and splice the real recorder, selected-pass, emit and DLAA handoff code.
    source = [PREAMBLE, function(sync, "vk_sync_progress_prepare_pipeline"), function(replay, "vk_exec_temporal_prepare")]
    source += [function(draw, name) for name in (
        "graphics_payload_align", "graphics_payload_fit", "graphics_slab_adopt", "graphics_unlink_owner", "graphics_detach",
        "graphics_payload_initialize", "graphics_scratch_claim", "vk_exec_draw_graphics_scratch_finish",
    )]
    source += [between(shield, "enum { PLAYER_DRAWS", "static PlatMutex shield_lock"),
               between(shield, "enum { PLAYER_MESH_BYTES", "// A mesh seen by another depth pass"),
               function(shield, "pose_bank"), function(shield, "same_pose"), function(shield, "shield_previous"),
               shield[shield.index("static void motion_geometry_promote("):]]
    move = function(dlaa, "mv_pass")
    prepare = between(move, "    int s = D.cam_slot", "    if (D.cam_size == CAM_BLOCK)")
    update = between(move, "    D.cam_slot ^=", "    VkDescriptorImageInfo zi")
    camera = between(move, "    vkCmdDispatch(cb, (w + 7)", "    // the characters' own motion")
    after = between(move, "    if (D.cam_size == OCAM_BLOCK &&\n        vk_shield_resolve_emit(",
                    "    if (D.cam_size == CAM_BLOCK && vk_mvobj_execute")
    assert move.index("vk_shield_resolve_prepare(") < move.index("D.mvm_slot++") < move.index("vkCmdDispatch(")
    evaluate = function(dlaa, "evaluate")
    check = between(evaluate, "    int masked = 0, mv =", "    VkImageView cv")
    assert evaluate.index("if (mv == MV_PASS_ABORTED)") < evaluate.index("vk_ffx_upscale(") < evaluate.index("evaluated:")
    assert evaluate.index("if (mv == MV_PASS_ABORTED)") < evaluate.index("N.evaluate(") < evaluate.index("vkCmdBlitImage(")
    source += [r'''
enum { MV_PASS_ABORTED = -1 };
static int mv_pass(VkExec *ex, struct Size *e, VkImageView zv, float jx, float jy, float bias, int *masked) {
    (void)bias;
    *masked = 0;
    const float cr_f = 1, *cr = &cr_f;
    const int have_m = 1;
    double cur_vp[16] = {0};
    uint32_t w = 200, h = 100;
''', prepare, "    (void)s;\n", update, camera, after, "    return 1;\n}\n", r'''
static int eval_hook(VkExec *ex) {
    int no_mv = 0;
    VkCommandBuffer cb = ex->cb;
    struct Size size = {.mv_view = HANDLE(VkImageView, 8)}, *e = &size;
    VkImageView zv = HANDLE(VkImageView, 9);
    float jx = 0, jy = 0;
''', check, "    ++sdk_calls;\n    return 1;\n}\n", CHECKS]

    cc = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="shield-motion-scratch-") as tmp:
        c = Path(tmp) / "check.c"
        exe = c.with_suffix(".exe")
        c.write_text("\n".join(source), encoding="utf-8")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", str(c), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True, timeout=20)


if __name__ == "__main__":
    main()

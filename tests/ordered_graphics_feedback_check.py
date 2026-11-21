"""Host check of ordered graphics feedback planning and HUD pipeline shape.

Extract the planner, activation and HUD shape from the production recorder.
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def function(source, name):
    match = re.search(r"^(?:static )?[A-Za-z_][\w \t*]*\b" + re.escape(name) + r"\([^;{}]*\)\s*\{", source, re.M)
    if not match:
        raise ValueError(name)
    depth = 1
    end = match.end()
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


STUBS = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan.h>

enum { VK_FEEDBACK_OFF, VK_FEEDBACK_RESTART, VK_FEEDBACK_LAYOUT };
enum {
    NVN_FORMAT_R8 = 1, NVN_LOGIC_OP_COPY = 0, NVN_BLEND_EQUATION_ADD = 1,
    NVN_BLEND_FUNC_ZERO, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_SRC_ALPHA,
    NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA, NVN_BLEND_FUNC_SRC_COLOR, NVN_BLEND_FUNC_DST_COLOR
};
enum Coverage { REPLACE, SOURCE_OVER, MULTIPLY, MULTIPLY_PLUS_ALPHA, ADD_ALPHA };
typedef struct { int id; } GfxShader;
typedef struct { VkImage image; VkImageAspectFlags aspect; int fb_usage, fb_learned; } VkTexImage;
typedef struct {
    VkImage image;
    const VkTexImage *img;
    uint32_t level, base_layer, layers, width, height, samples, is3d;
    VkFormat format;
    VkImageViewType view_type;
    VkImageAspectFlags aspect;
    int32_t nvn_format;
    uint32_t scale, blocks;
    int exact_snapshot;
    VkImage bc_backing;
    uint64_t bc_address;
    uint32_t bc_level;
} VkExecTarget;
typedef struct {
    VkExecTarget color[8], depth;
    uint64_t fence;
    uint32_t pass_ncolor, pass_layers, fb_dirty, fbl_mask;
    int in_pass;
#if REAL_MESH_RMW || REAL_UBO_PASS_SPLIT
    VkCommandBuffer cb;
    unsigned recorded;
#endif
#if REAL_UBO_PASS_SPLIT
    VkImageView pass_views[9];
    uint32_t pass_w, pass_h, fb_read, pass_serial;
    VkFormat pass_depth, pass_stencil;
    int hud_alpha_attachment;
#endif
    VkExtent2D tonemap_extent;
} VkExec;
typedef struct { VkImage image; uint32_t l0, l1, a0, a1; } SubRange;
typedef struct {
    unsigned target, src_rgb, dst_rgb, src_a, dst_a, eq_rgb, eq_a;
} blend_st;
typedef struct {
    const GfxShader *fs;
    int32_t color_fmt[8];
    struct { unsigned mask; } chmask;
    struct { unsigned logic_op, blend_en; } color;
    blend_st blend[8];
} VkPipeState;
typedef struct { int feedback; } VkCtx;
static VkCtx C;
static const VkCtx *vk_ctx(void) { return &C; }
static struct { unsigned fb_draws, fb_restarts; } vk_exec_frame;
static struct {
    VkImage hud_target;
    uint64_t hl_at, presents;
    struct { uint32_t w, h; VkImageView view; } alpha;
} D;
static unsigned incomplete, begin_count, end_count;
static int tonemap_shape;
#if REAL_HUD_SOURCE
typedef struct {
    VkImage target;
    uint64_t turn;
    struct { uint32_t w,h; VkImageView view; } coverage;
    int alpha_seen;
} VkDlaaHudSource;
static VkDlaaHudSource hud_use;
static struct { VkDlaaHudSource *open; } hud_sources;
static void hud_group_record(const VkExec *ex, VkDlaaHudSource *use) { assert(ex&&use==&hud_use); }
#endif

static void vk_dlaa_hud_incomplete(void) { incomplete++; }
static int has_image(const VkImage *images, int n, VkImage image) {
    for (int i = 0; i < n; i++) if (images[i] == image) return 1;
    return 0;
}
static unsigned vk_pipeline_target_wmask(const VkPipeState *s, int i) {
    return (s->chmask.mask >> (4 * i)) & 15u;
}
static const GfxShader primary = {1}, coverage = {2};
static int variant_available;
static const GfxShader *variant_for(const GfxShader *fs, enum Coverage mode) {
    assert(fs == &primary && mode == REPLACE);
    return variant_available ? &coverage : NULL;
}
'''

SEAM = r'''
static uint32_t pass_mask_from_recorder(const VkExec *ex, int tonemap) {
'''  # Body below is cut verbatim from vk_exec_pass_begin.

ADAPTERS = r'''
static SubRange selected[4];
static unsigned nselected;
static uint32_t feedback_scan(VkExec *ex, const GfxShader *fs) {
    assert(fs);
    uint32_t mask = 0;
    for (unsigned i = 0; i < nselected; i++) mask |= attached_as(ex, &selected[i]);
    return mask;
}
static void vk_exec_pass_end(VkExec *ex) {
    assert(ex->in_pass);
    ex->in_pass = 0;
    end_count++;
}
static void vk_exec_pass_begin(VkExec *ex) {
    assert(!ex->in_pass);
    ex->in_pass = 1;
    ex->fbl_mask = pass_mask_from_recorder(ex, tonemap_shape);
    ex->fb_dirty = 0;
    begin_count++;
}
'''

READY_STUBS = r'''
typedef enum {
    VK_EXEC_GRAPHICS_READY, VK_EXEC_GRAPHICS_NEEDS_ORDERED, VK_EXEC_GRAPHICS_STOPPED,
    VK_EXEC_GRAPHICS_UNAVAILABLE, VK_EXEC_GRAPHICS_NOOP, VK_EXEC_GRAPHICS_COMPAT,
    VK_EXEC_GRAPHICS_INVALIDATED
} VkExecGraphicsResult;
typedef enum {
    VK_GRAPHICS_READY, VK_GRAPHICS_PENDING, VK_GRAPHICS_STOPPED, VK_GRAPHICS_UNSUPPORTED, VK_GRAPHICS_TERMINAL
} VkGraphicsPipelineResult;
typedef enum { VK_SYNC_PROGRESS_COMPLETE, VK_SYNC_PROGRESS_STOPPED, VK_SYNC_PROGRESS_STOPPED_AFTER_CUT,
               VK_SYNC_PROGRESS_UNAVAILABLE } VkSyncProgressResult;
typedef struct { int pool; } VkPipeInfo;
typedef struct { int placeholder; } VkGraphicsUniformFootprints;
typedef struct { const VkPipeInfo *info; VkPipeline pipeline; void *revision;
                 const VkGraphicsUniformFootprints *vertex_uniforms; } VkGraphicsPipelineUse;
typedef struct { const char *reason; int error, admission; } VkGraphicsPipelineFailure;
enum { RC_ADMISSION_REFUSED = 1 };
typedef struct { int pending; } VkGraphicsPipelineWait;
#if REAL_UBO_PASS_SPLIT
typedef struct { uint64_t target_registration; } p_set_rt;
typedef struct {
    VkExecGraphicsPassShape shape;
    p_set_rt rt;
    uint64_t fence;
    int withheld;
    VkCommandBuffer hook_cb, barrier_cb;
} VkExecGraphicsPass;
typedef struct VkTextureAliasRead { int id; } VkTextureAliasRead;
typedef enum { VK_EXEC_GRAPHICS_PASS_READY, VK_EXEC_GRAPHICS_PASS_UNAVAILABLE } VkExecGraphicsPassResult;
#else
typedef struct { int withheld; } VkExecGraphicsPass;
#endif
#if REAL_UBO_MERGE
typedef struct { int current, native_count; } GraphicsUboMerge;
#endif
typedef struct {
    int placeholder;
#if REAL_BYTE_INDEX
    const p_draw_idx *widen_draw;
#endif
#if REAL_UBO_MERGE
    GraphicsUboMerge *ubo_merges;
    uint32_t ubo_merge_count;
    int native_materialized;
    const VkPipeInfo *info;
    const VkGraphicsUniformFootprints *vertex_uniforms;
#endif
} VkExecGraphicsScratch;
typedef struct { int placeholder; } VkExecPoolRequirement;
typedef struct { int buffer; uint64_t offset; } VkMemSpan;
typedef struct { int placeholder; } VkMemSpanCapture;
typedef struct { uint64_t ind; } p_draw_mesh_ind;
#if REAL_MESH_RMW
typedef struct { int *rmw, *rmw_bitmap; } SelectedStorageWriters;
static int selected_word, selected_bitmap, rmw_barriers, bitmap_barriers;
#if REAL_NATIVE_RMW_SPLIT
static unsigned native_order_calls;
static int native_split_refused;
static int vk_exec_record_between_renderings(VkExec *ex, int (*record)(VkCommandBuffer, void *), void *context) {
    ++native_order_calls;
    if (native_split_refused) return 0;
    return record(ex->cb, context);
}
#endif
static int vk_mem_native_rmw_bitmap_order(const int *rmw, VkCommandBuffer cb) {
    assert(rmw == &selected_bitmap && cb == (VkCommandBuffer)(uintptr_t)4);
    bitmap_barriers++;
    return 1;
}

static int vk_mem_native_rmw_word_order(const int *rmw, VkCommandBuffer cb) {
    assert(rmw == &selected_word && cb == (VkCommandBuffer)(uintptr_t)4);
    rmw_barriers++;
    return 1;
}
#endif
typedef struct { int selected; } RcVmaDependency;
typedef struct { int placeholder; } RcBudgetReadiness;
typedef struct { VkExec *ex; uint64_t bytes; } VkDrawScratchWaitContext;
typedef struct {
    VkPipeState state;
    VkGraphicsPipelineUse use;
    VkGraphicsPipelineWait *wait;
    VkExecGraphicsScratch *scratch;
    VkExecGraphicsPass *pass;
#if REAL_MESH_RMW
    SelectedStorageWriters *selected_writers;
#endif
#if REAL_UBO_PASS_SPLIT
    VkTextureAliasRead *attachment_seeds[9];
#endif
    const void *payload;
    uint64_t program_serial;
    uint32_t opcode;
    int activated, mesh, direct_mesh, hud_incomplete;
    GraphicsFeedback feedback;
    VkMemSpan indirect;
    VkMemSpanCapture *indirect_hold;
#if REAL_VERTEX_INPUT
    int exact_index, exact_indirect;
#endif
#if REAL_VERTEX_INPUT || REAL_BYTE_INDEX
    VkMemSpan index;
    VkMemSpanCapture *index_hold;
#endif
#if REAL_BYTE_INDEX
    int compatibility, byte_index_compat, byte_index_qualified;
#endif
    VkDeviceSize mesh_ring_offset;
    void *samples;
    uint32_t sample_count;
} VkExecGraphicsDraw;
#if REAL_SCRATCH_FAILURE
struct VkExecPreparationFailure {
    int owner, reason, result, detail;
    uint32_t role, slot;
    const char *stage;
};
typedef struct VkExecPreparationFailure VkExecPreparationFailure;
typedef enum { VK_EXEC_PREPARATION_MAPPING, VK_EXEC_PREPARATION_PLACEMENT,
               VK_EXEC_PREPARATION_PRESSURE, VK_EXEC_PREPARATION_ORDER } VkExecPreparationReason;
enum { VK_EXEC_PREPARATION_SCRATCH = 3, VK_EXEC_PREPARATION_PIPELINE = 5, VK_MUSE_IB = 1 };
static void vk_exec_graphics_failure_set(VkExecGraphicsDraw *draw, const VkExecPreparationFailure *cause) {
    assert(draw && cause->slot == UINT32_MAX);
    assert(cause->owner == VK_EXEC_PREPARATION_SCRATCH ||
           (cause->owner == VK_EXEC_PREPARATION_PIPELINE && cause->detail == VK_ERROR_INITIALIZATION_FAILED &&
            cause->stage && !strcmp(cause->stage, "compiler-terminal")));
}
#endif
enum { OP_DRAW_MESH_IND = 17, OP_DRAW_IDX, OP_DRAW_IDX_IND };
static int argument_probe, argument_current, input_records, mesh_records;
static VkMemSpanCapture argument_capture;
#define VK_MEM_USE(span) ((void)(span))
#define rt_log(...) ((void)0)

static int waits, prior_waits, builds, activations, writes, failed, grows, stop_requested, hard_requested;
static const VkPipeState *needed_state;
static VkGraphicsPipelineWait pending_pipeline;
static VkExecGraphicsScratch granted_scratch;
#if REAL_UBO_MERGE
static unsigned merge_checks, scratch_finishes;
static int graphics_ubo_merge_current(const VkExec *ex, const GraphicsUboMerge *merge) {
    assert(ex->fence == 7);
    merge_checks++;
    return merge->current;
}
static void vk_exec_draw_graphics_scratch_finish(VkExecGraphicsScratch **scratch) {
    assert(*scratch == &granted_scratch);
    (*scratch)->ubo_merges = NULL;
    (*scratch)->ubo_merge_count = 0;
    *scratch = NULL;
    scratch_finishes++;
}
#endif
#if REAL_UBO_NATIVE_RECORD
#if REAL_UBO_PASS_SPLIT
static unsigned ubo_pass_events, private_copies;
static int deliberately_stale_ubo;
#endif
static int graphics_ubo_merge_record_native(VkExec *ex, VkExecGraphicsScratch *scratch) {
    assert(ex && !ex->in_pass && scratch == &granted_scratch && !writes && !activations);
#if REAL_UBO_PASS_SPLIT
    if (ubo_pass_events) {
        assert(ubo_pass_events == 2);
        ubo_pass_events++;
    }
    if (deliberately_stale_ubo) return 0;
    scratch->native_materialized = 1;
    private_copies++;
#endif
    return 1;
}
#endif
#if REAL_UBO_PASS_SPLIT
static VkTextureAliasRead attachment_seed;
static int vk_exec_graphics_pass_current(const VkExec *ex, const VkExecGraphicsPass *pass) {
    return !pass->withheld && ex->fence == pass->fence && ex->in_pass == pass->shape.in_pass;
}
static void vk_exec_oq_pass_end(VkExec *ex) {
    assert(ex->in_pass && !ubo_pass_events++);
}
static void ubo_end_rendering(VkCommandBuffer cb) {
    assert(cb == (VkCommandBuffer)(uintptr_t)4 && ubo_pass_events++ == 1);
}
static void ubo_resume_rendering(VkCommandBuffer cb, const VkRenderingInfo *rendering) {
    assert(cb == (VkCommandBuffer)(uintptr_t)4 && ubo_pass_events++ == 4);
    assert(rendering->renderArea.extent.width == 129 && rendering->renderArea.extent.height == 128 &&
        rendering->layerCount == 1 && rendering->colorAttachmentCount == 1 && !rendering->pStencilAttachment);
    const VkRenderingAttachmentInfo *color = rendering->pColorAttachments;
    const VkRenderingAttachmentInfo *depth = rendering->pDepthAttachment;
    assert(color->imageView == (VkImageView)(uintptr_t)6 && color->imageLayout == VK_IMAGE_LAYOUT_GENERAL);
    assert(depth->imageView == (VkImageView)(uintptr_t)9 &&
        depth->imageLayout == VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT);
    assert(color->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD && color->storeOp == VK_ATTACHMENT_STORE_OP_STORE);
    assert(depth->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD && depth->storeOp == VK_ATTACHMENT_STORE_OP_STORE);
}
static void memory_barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src, VkAccessFlags2 src_access,
                           VkPipelineStageFlags2 dst, VkAccessFlags2 dst_access) {
    assert(cb == (VkCommandBuffer)(uintptr_t)4 && ubo_pass_events++ == 3);
    assert(src == VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT && src_access == VK_ACCESS_2_MEMORY_WRITE_BIT &&
        dst == VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT &&
        dst_access == (VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT));
}
#endif
static const VkPipeInfo fast_info = {0};
static int pool_on(void) { return 1; }
static int vk_sync_stopping(void) { return stop_requested; }
#if REAL_SCRATCH_WAIT
static int vk_exec_recording_failed(void) { return 0; }
#endif
static int vk_device_is_lost(void) { return 0; }
static void vk_device_lost_check(VkResult result, const char *what) { (void)result; (void)what; }
static VkSyncProgressResult vk_exec_graphics_progress_prepare(VkExec *ex, int (*prepare)(void *), void *context) {
    assert(ex->fence && !writes && !activations);
    return prepare(context) ? VK_SYNC_PROGRESS_COMPLETE : VK_SYNC_PROGRESS_STOPPED;
}
static VkGraphicsPipelineResult vk_pipeline_graphics_prepare(const VkPipeState *state, int allow_start,
    VkGraphicsPipelineWait **wait, VkGraphicsPipelineUse *use, VkGraphicsPipelineFailure *failure) {
    assert(state == needed_state && !writes && !activations);
    if (hard_requested) {
        failure->error = VK_ERROR_INITIALIZATION_FAILED;
        return VK_GRAPHICS_TERMINAL;
    }
    if (stop_requested) return VK_GRAPHICS_STOPPED;
    if (!allow_start) return VK_GRAPHICS_PENDING;
    if (waits < 2) {
        if (!*wait) builds++;
        *wait = &pending_pipeline;
        return VK_GRAPHICS_PENDING;
    }
    *wait = NULL;
    *use = (VkGraphicsPipelineUse){.info=&fast_info,.pipeline=(VkPipeline)(uintptr_t)13,.revision=&pending_pipeline};
    return VK_GRAPHICS_READY;
}
static int vk_pipeline_graphics_wait(VkGraphicsPipelineWait *wait) {
    assert(wait == &pending_pipeline && !writes && !activations);
    waits++;
    return 1;
}
static VkResult vk_exec_draw_graphics_scratch_prepare(VkExec *ex, const VkPipeState *state, const VkPipeInfo *info,
    uint32_t opcode, const void *payload, int ordered, VkExecGraphicsScratch **out, uint64_t *bytes, RcVmaDependency *dep,
    const VkGraphicsUniformFootprints *source) {
    assert(!source);
    assert(ex->fence && state == needed_state && info == &fast_info && !opcode && payload && ordered && !*out && !writes);
    *bytes = 144;
    if (++grows < 3) {
        dep->selected = 1;
        return VK_NOT_READY;
    }
    *out = &granted_scratch;
    return VK_SUCCESS;
}
#if REAL_BYTE_INDEX
static VkExecGraphicsResult graphics_byte_index_select(VkExecGraphicsDraw *draw, const p_draw_idx *indices) {
    (void)draw; (void)indices;
    assert(0);  // The cold feedback test is nonindexed.
    return VK_EXEC_GRAPHICS_UNAVAILABLE;
}
static VkResult graphics_scratch_prepare_indexed(VkExec *ex, const VkPipeState *state, const VkPipeInfo *info,
    uint32_t opcode, const void *payload, const VkMemSpan *span, const VkMemSpanCapture *hold, int ordered,
    int legacy, VkExecGraphicsScratch **out, uint64_t *bytes, RcVmaDependency *dep, VkExecHostInputResult *source) {
    assert(span && !hold && !legacy);
    *source = VK_EXEC_HOST_INPUT_READY;
    return vk_exec_draw_graphics_scratch_prepare(ex, state, info, opcode, payload, ordered, out, bytes, dep, NULL);
}
#endif
static void resource_core_budget_readiness(int required, RcBudgetReadiness *out) { assert(required && out); }
static uint64_t vk_exec_draw_graphics_scratch_prior_useful(VkExec *ex, uint64_t bytes,
    const RcVmaDependency *dep, const RcBudgetReadiness *now) {
    assert(ex->fence && bytes == 144 && dep->selected && now);
    return (uint64_t)grows;
}
static uint64_t vk_sync_fence_done(void) { return 0; }
static VkSyncProgressResult vk_sync_progress_wait_prior(uint64_t older) {
    assert(older == (uint64_t)++prior_waits && !writes);
    return VK_SYNC_PROGRESS_COMPLETE;
}
static int vk_exec_draw_graphics_scratch_spare_ready(void *context) { (void)context; return 0; }
static VkSyncProgressResult vk_sync_progress_wait_capacity(const RcVmaDependency *dep,
    int (*ready)(void *), void *context) { (void)dep; (void)ready; (void)context; assert(0); return VK_SYNC_PROGRESS_UNAVAILABLE; }
static void vk_exec_required_graphics_unavailable(VkExec *ex, uint32_t opcode, const char *why) {
    assert(ex && (!opcode || argument_probe) && why);
    failed++;
}
static int sample_query_images_record(VkExec *ex, void *samples, uint32_t count) {
    assert(ex && !samples && !count && !writes);
    return 1;
}
static int vk_exec_graphics_pass_activate(VkExec *ex, VkExecGraphicsPass *pass) {
    assert(pass && !writes);
    if (pass->withheld) return 0;
    ex->in_pass = 1;
    activations++;
    return 1;
}
#if REAL_VERTEX_INPUT
static int graphics_vertices_record(VkExec *ex,VkExecGraphicsDraw *draw) { assert(ex&&draw);return 1; }
static int vk_mem_span_record_input(const VkMemSpanCapture *hold,const VkMemSpan *span) {
    assert(argument_probe&&hold==&argument_capture&&span->buffer==80&&!input_records);
    return argument_current;
}
static void vk_exec_record_native_input_provider(VkExec *ex,VkMemSpanCapture **hold) {
    assert(ex->fence&&argument_probe&&*hold==&argument_capture&&argument_current);
    *hold=NULL;input_records++;
}
#endif
static int graphics_argument_use(VkExec *ex, uint32_t opcode, uint64_t addr, uint64_t bytes,
                                 const VkMemSpan *span, VkMemSpanCapture **hold) {
    (void)ex; (void)opcode; (void)addr; (void)bytes; (void)span; (void)hold;
    assert(0);
    return 0;
}
static int mesh_ring_entry(VkExec *ex, const p_draw_mesh_ind *d, const VkMemSpan *span, VkDeviceSize *offset) {
    assert(ex->fence&&argument_probe&&d&&span->buffer==80&&input_records==1);
    *offset=48;mesh_records++;return 1;
}
static int graphics_record_state(VkExec *ex, const VkPipeState *state, VkPipeline pipeline, const VkPipeInfo *info,
    uint32_t feedback, int mesh, int fb, int direct_mesh, VkExecPoolRequirement *req, VkExecGraphicsDraw *draw) {
    assert(ex->in_pass && state == needed_state && info == &fast_info && pipeline == (VkPipeline)(uintptr_t)13);
    assert(!feedback && (!mesh || argument_probe) && !fb && !direct_mesh && !req && draw->scratch == &granted_scratch);
    assert(draw->activated && !writes);
    writes++;
    return 1;
}
'''

FIXED_ARGUMENT_STUBS = r'''
typedef enum { RC_REGISTRY_OK, RC_REGISTRY_STALE, RC_REGISTRY_UNSUPPORTED } RcRegistryResult;
static int native_attempts, legacy_attempts, native_supported=1;
static int vk_mem_span_capture_current(const VkMemSpanCapture *hold,const VkMemSpan *span) {
    return hold==&argument_capture && span->buffer==80 && argument_current;
}
static void vk_mem_span_capture_release(VkMemSpanCapture **hold) {
    assert(*hold==&argument_capture);
    *hold=NULL;
}
static RcRegistryResult vk_mem_span_native_input(uint64_t address,uint64_t size,int clamp,VkMemSpan *span,
                                                 VkMemSpanCapture **hold) {
    assert(address==0x4000&&(size==20||size==12)&&!clamp&&!*hold);
    native_attempts++;
    if (!native_supported) return RC_REGISTRY_UNSUPPORTED;
    *span=(VkMemSpan){.buffer=80,.offset=8};*hold=&argument_capture;
    return RC_REGISTRY_OK;
}
static int vk_mem_lookup(uint64_t address,uint64_t size,VkMemSpan *span) {
    assert(!native_supported&&address==0x4000&&size==20);
    legacy_attempts++;
    *span=(VkMemSpan){.buffer=91,.offset=4};
    return 1;
}
static VkMemSpanCapture *vk_mem_span_capture(uint64_t address,uint64_t size,const VkMemSpan *span) {
    assert(!native_supported&&address==0x4000&&size==20&&span->buffer==91);
    return &argument_capture;
}
static int vk_mem_span_held_argument_current(const VkMemSpanCapture *hold,uint64_t address,uint64_t size,
                                             const VkMemSpan *span) {
    assert(address==0x4000&&(size==12||size==20));
    return native_supported ? vk_mem_span_capture_current(hold,span) :
           hold==&argument_capture&&span->buffer==91&&argument_current;
}
'''

READY_TESTS = r'''
#if REAL_FIXED_INDIRECT
static void test_fixed_indirect_activation(void) {
    VkExec ex={.fence=7};
    VkExecGraphicsPass pass={0};
    const p_draw_mesh_ind payload={0x4000};
    argument_probe=argument_current=1;
    failed=0;
    for (int mesh=0;mesh<=1;mesh++) {
        writes=activations=input_records=mesh_records=0;
        VkExecGraphicsDraw draw={.opcode=mesh?OP_DRAW_MESH_IND:OP_DRAW_IDX_IND,.mesh=mesh,
            .payload=&payload,.pass=&pass,.scratch=&granted_scratch,
            .use={.info=&fast_info,.pipeline=(VkPipeline)(uintptr_t)13,.revision=&pending_pipeline}};
        assert(graphics_indirect_select(&draw,payload.ind,mesh?12:20)==VK_EXEC_GRAPHICS_READY && draw.exact_indirect);
        needed_state=&draw.state;
        assert(graphics_draw_activate(&ex,&draw,NULL) && writes==1 && activations==1 &&
               input_records==1 && mesh_records==mesh && !draw.indirect_hold);
    }
    writes=activations=input_records=0;
    VkExecGraphicsDraw rejected={.payload=&payload,.pass=&pass,.use={.info=&fast_info,.pipeline=(VkPipeline)(uintptr_t)13,.revision=&pending_pipeline}};
    assert(graphics_indirect_select(&rejected,payload.ind,20)==VK_EXEC_GRAPHICS_READY);
    argument_current=0;
    assert(graphics_indirect_select(&rejected,payload.ind,20)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!graphics_draw_activate(&ex,&rejected,NULL) && failed==1 && !input_records && !writes && !activations);
    assert(native_attempts==3&&!legacy_attempts);
    vk_mem_span_capture_release(&rejected.indirect_hold);
    native_supported=0;argument_current=1;
    VkExecGraphicsDraw cold={0};
    assert(graphics_indirect_select(&cold,payload.ind,20)==VK_EXEC_GRAPHICS_READY && !cold.exact_indirect);
    assert(native_attempts==4&&legacy_attempts==1&&cold.indirect.buffer==91);
    vk_mem_span_capture_release(&cold.indirect_hold);
    argument_probe=0;
}
#endif

#if REAL_UBO_PASS_SPLIT
static void test_ubo_pass_split(void) {
    VkExec ex = {.fence = 7, .in_pass = 1, .cb = (VkCommandBuffer)(uintptr_t)4, .pass_serial = 43,
        .pass_ncolor = 1, .pass_layers = 1, .pass_w = 129, .pass_h = 128, .pass_depth = VK_FORMAT_D32_SFLOAT,
        .depth = {.image = (VkImage)(uintptr_t)7, .aspect = VK_IMAGE_ASPECT_DEPTH_BIT},
        .pass_views = {[0] = (VkImageView)(uintptr_t)6, [8] = (VkImageView)(uintptr_t)9},
        .fbl_mask = 0x100, .fb_dirty = 0x100, .fb_read = 0x100};
    VkExecGraphicsPass original = {.fence = 7, .rt.target_registration = 17, .shape.in_pass = 1,
                                  .hook_cb = ex.cb};
    const unsigned begins = begin_count, ends = end_count, copies = private_copies;
    GraphicsUboMerge merge = {.current = 1, .native_count = 1};
    granted_scratch = (VkExecGraphicsScratch){.ubo_merges = &merge, .ubo_merge_count = 1};
    VkExecGraphicsDraw draw = {.scratch = &granted_scratch, .pass = &original,
                              .attachment_seeds = {[8] = &attachment_seed}};

    // The active pass already owns the exact D32 seed. An ambient hook on that pass is not new hook work.
    assert(graphics_ubo_merge_record_for_draw(&ex, &draw));
    assert(ex.in_pass && ex.pass_serial == 43 && draw.pass == &original && draw.attachment_seeds[8] == &attachment_seed);
    assert(begin_count == begins && end_count == ends && ubo_pass_events == 5 && private_copies == copies + 1);
    assert(ex.recorded == 4 && !ex.fb_dirty && !ex.fb_read && ex.fbl_mask == 0x100);
    assert(graphics_ubo_merge_record_for_draw(&ex, &draw));
    assert(ubo_pass_events == 5 && private_copies == copies + 1 && ex.recorded == 4);

    // Compatibility uses the same bridge if it kept the active pass but changed an unrelated selection.
    ubo_pass_events = 0;
    granted_scratch.native_materialized = 0;
    assert(graphics_ubo_merge_record_preserving_pass(&ex, &granted_scratch));
    assert(ex.in_pass && draw.attachment_seeds[8] == &attachment_seed && ex.pass_serial == 43);
    assert(begin_count == begins && end_count == ends && ubo_pass_events == 5 && private_copies == copies + 2);

    ubo_pass_events = 0;
    granted_scratch.native_materialized = 0;
    original.withheld = 1;
    assert(!graphics_ubo_merge_record_for_draw(&ex, &draw));
    original.withheld = 0;
    ex.hud_alpha_attachment = 1;
    assert(!graphics_ubo_merge_record_for_draw(&ex, &draw));
    ex.hud_alpha_attachment = 0;
    ex.tonemap_extent.width = 1;
    assert(!graphics_ubo_merge_record_for_draw(&ex, &draw));
    ex.tonemap_extent.width = 0;
    assert(!ubo_pass_events && private_copies == copies + 2 && ex.in_pass && draw.pass == &original);

    // A stale copy still restores the render instance for cancellation. It never records a second copy.
    deliberately_stale_ubo = 1;
    assert(!graphics_ubo_merge_record_for_draw(&ex, &draw));
    deliberately_stale_ubo = 0;
    assert(ubo_pass_events == 5 && private_copies == copies + 2 && ex.in_pass && draw.pass == &original);

    ex.in_pass = original.shape.in_pass = 0;
    ubo_pass_events = 0;
    assert(graphics_ubo_merge_record_for_draw(&ex, &draw));
    assert(!ex.in_pass && !ubo_pass_events && private_copies == copies + 3);
    assert(begin_count == begins && end_count == ends && draw.attachment_seeds[8] == &attachment_seed);
    granted_scratch = (VkExecGraphicsScratch){0};
    puts("Graphics UBO copy: exact color and D32 LOAD, retained seed, one query split, stale and special cases passed");
}
#endif

static void test_required_one_shot(void) {
    VkExec ex = {.fence = 7};
    VkExecGraphicsPass pass = {0};
    VkExecGraphicsDraw draw = {.program_serial = 11, .payload = &ex, .pass = &pass};
    needed_state = &draw.state;
    assert(graphics_pipeline_ready(&ex, &draw, 0) == VK_EXEC_GRAPHICS_NEEDS_ORDERED);
    assert(!writes && !builds && !activations && !grows);
    assert(graphics_pipeline_ready(&ex, &draw, 1) == VK_EXEC_GRAPHICS_READY);
    assert(builds == 1 && waits == 2 && !writes && !activations);
    assert(graphics_scratch_ready(&ex, &draw, 1) == VK_EXEC_GRAPHICS_READY);
    assert(prior_waits == 2 && grows == 3 && !writes && !activations);
#if REAL_UBO_MERGE
    assert(graphics_ubo_merges_current(&ex, NULL) && !merge_checks);
    GraphicsUboMerge merges[] = {{.current = 1}, {.current = 1}};
    granted_scratch.ubo_merges = merges;
    granted_scratch.ubo_merge_count = 2;
    assert(graphics_scratch_ready(&ex, &draw, 1) == VK_EXEC_GRAPHICS_READY);
    assert(merge_checks == 2 && !scratch_finishes && grows == 3);
    merges[1].current = 0;
    assert(graphics_scratch_ready(&ex, &draw, 1) == VK_EXEC_GRAPHICS_READY);
    assert(merge_checks == 4 && scratch_finishes == 1 && grows == 4 && prior_waits == 2);
    assert(draw.scratch == &granted_scratch && !writes && !activations);
#endif
    assert(graphics_draw_activate(&ex, &draw, NULL) && activations == 1 && writes == 1);
    assert(!graphics_draw_activate(&ex, &draw, NULL) && failed == 1 && writes == 1);

    VkExecGraphicsDraw denied = {.payload = &ex, .pass = &pass};
    needed_state = &denied.state;
    writes = activations = 0;
    hard_requested = 1;
    assert(graphics_pipeline_ready(&ex, &denied, 1) == VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!graphics_draw_activate(&ex, &denied, NULL) && failed == 2 && !writes && !activations);
    hard_requested = 0;
    stop_requested = 1;
    assert(graphics_pipeline_ready(&ex, &denied, 1) == VK_EXEC_GRAPHICS_STOPPED);
    assert(!writes && !activations);
}
'''

SAMPLE_STUBS = r'''
typedef enum { VK_TEXTURE_SAMPLE_READY, VK_TEXTURE_SAMPLE_FALLBACK, VK_TEXTURE_SAMPLE_ABSENT,
               VK_TEXTURE_SAMPLE_GONE, VK_TEXTURE_SAMPLE_UNAVAILABLE } VkTextureSampleStatus;
enum { VK_TEXTURE_ALIAS_SAMPLE_FORCE = 1, VK_TEXTURE_ALIAS_SAMPLE_LINEAR_RGBA8_VIEW = 2,
       VK_TEXTURE_ALIAS_SAMPLE_ONLY = 4, VK_TEXTURE_ALIAS_ORDER = 5 };
enum { VK_TEXTURE_READ_FORCE_COMPAT = 2, VK_TEXTURE_READ_LINEAR_RGBA8_VIEW = 4, VK_TEXTURE_READ_EXACT_DETAIL = 8 };
typedef struct {
    const VkTexImage *image;
    uint64_t registration;
    int native_slot, historical, canonical, projected;
    void *query_image;
} VkTextureSample;
static int vk_texture_query_image_current_locked(const void *image) { (void)image; assert(0); return 0; }
typedef struct { uint64_t registration; int native_slot, canonical, unbacked, projected; } VkTextureReadKey;
typedef struct { int exact; } GfxShaderRes;
typedef struct { uint64_t scene, ordinal; } RcRegistryPosition;
typedef struct VkTextureAliasRead VkTextureAliasRead;
typedef struct RcInterval { unsigned unused; } RcInterval;
typedef struct { uint64_t registration; } VkTextureSampleGeometry;
static __attribute__((unused)) int vk_texture_sample_geometry_current_locked(int logical,
    const VkTextureSample *sample, const VkTexImage *image, const VkTextureSampleGeometry *geometry,
    const RcInterval **ranges, uint32_t *count) {
    (void)logical; (void)sample; (void)image; (void)geometry; (void)ranges; (void)count;
    assert(0 && "ordinary feedback checks do not acquire geometry grants");
    return 0;
}
typedef struct {
    const GfxShaderRes *resource;
    uint64_t handle, token;
    VkTextureSample sample;
    VkTexImage image;
    VkTextureAliasRead *alias;
    VkTextureSampleGeometry geometry;
} GraphicsSample;
static __attribute__((unused)) int vk_texture_alias_sample_current_locked(int logical, VkTextureAliasRead *alias) {
    (void)logical; (void)alias; assert(0); return 0;
}
typedef struct { uint64_t serial; } TestDescriptor;
typedef struct { TestDescriptor t; } Tex;
typedef struct {
    struct { uint64_t id; } handle;
    uint64_t publication;
    TestDescriptor descriptor;
} TextureRegistration;
typedef struct { int used,finalized; uint64_t pool_owner; TextureRegistration *registration; Tex *entry; } TexSlot;
enum { VK_TEXTURE_REGISTRATION_CURRENT, VK_TEXTURE_REGISTRATION_EXACT,
       VK_TEXTURE_REGISTRATION_ABSENT = -1, VK_TEXTURE_REGISTRATION_GONE = -2 };
#define NVN_HANDLE_TEX(h) ((int)((h) & 0xfffff))
static const VkTexImage warm_sample = {(VkImage)(uintptr_t)91, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
static TexSlot tslots[1256];
static const int ntslots = sizeof tslots / sizeof *tslots;
static struct { int calls; } binding_path;
enum { VK_TEXTURE_BINDING_SOURCE = 2 };
static int compatibility_current, canonical_current = 1, projected_current;
static int registration_status, native_selection = 258, warm_is_canonical = 1, alias_status;
static unsigned expected_alias_options;
static unsigned registration_calls, native_queries, alias_queries, legacy_tokens;
typedef struct { int phase, status; } VkTextureAliasFailure;
static const RcRegistryPosition sample_read = {7, 14};
static RcRegistryPosition vk_exec_resource_position(void) { return sample_read; }
static int rc_registry_position_valid(RcRegistryPosition read) { return read.scene && read.ordinal; }
static int rc_registry_position_compare(RcRegistryPosition a, RcRegistryPosition b) {
    return a.scene != b.scene || a.ordinal != b.ordinal;
}
static int image_alias_refuse(VkTextureAliasFailure *failure, int phase, int status, RcInterval receiver) {
    (void)receiver;
    if (failure) *failure = (VkTextureAliasFailure){.phase = phase, .status = status};
    return -1;
}
static VkTextureAliasRead *const acquired_alias = (VkTextureAliasRead *)(uintptr_t)17;
static TextureRegistration warm_registration = {.handle.id=741,.publication=4,.descriptor.serial=80};
static Tex warm_owner = {.t.serial=80};
static int texture_registration_resolve(int logical, const TextureRegistration **record, int *native) {
    assert(logical == 731 && record && native);
    *record = &warm_registration;
    *native = native_selection;
    registration_calls++;
    return registration_status;
}
static int image_alias_sample_select_locked(int logical, int native, RcRegistryPosition read,
    VkTextureAliasRead **out, VkTextureSample *sample, VkTexImage *shape, unsigned options, VkTextureAliasFailure *failure) {
    assert(logical == 731 && native == native_selection && options == expected_alias_options && !*out && sample && shape);
    assert(!rc_registry_position_compare(read, sample_read));
    alias_queries++;
    if (alias_status > 0) { *out = acquired_alias; *shape = warm_sample; }
    if (alias_status < 0 && failure) failure->status = 17;
    return alias_status;
}
static uint64_t vk_exec_texture_pool(void) { return 7; }
static uint64_t vk_texture_registration_publication(void) { return 5; }
static Tex *texture_binding_entry(const TexSlot *slot, int native) {
    assert(native == 1255);
    return slot->entry;
}
static Tex *texture_read_key_image(const TexSlot *slot, VkImage image) {
    assert(image == warm_sample.image);
    return slot->entry;
}
static int source_binding_receipt(Tex *owner, RcRegistryPosition *write) {
    assert(owner && !write);
    return canonical_current;
}
static VkTextureSampleStatus vk_texture_sample_query_locked(int native, VkTextureSample *sample) {
    assert(native == 258);
    native_queries++;
    *sample = (VkTextureSample){.image = &warm_sample, .registration = warm_registration.handle.id,
                               .canonical = warm_is_canonical};
    return VK_TEXTURE_SAMPLE_READY;
}
static Tex *texture_projected_sample_locked(const TexSlot *slot,int native,VkImage expected,
                                            VkTextureSample *sample,RcRegistryPosition *next) {
    assert(native==1255 && slot==&tslots[native] && expected==warm_sample.image && !sample && !next);
    return projected_current ? slot->entry : NULL;
}
static int texture_history_sample_locked(int logical,VkTextureSample *out,int *native,int recording) {
    const TextureRegistration *record;
    assert(!recording && out);
    int status=texture_registration_resolve(logical,&record,native);
    if(status==VK_TEXTURE_REGISTRATION_EXACT)
        *out=(VkTextureSample){.image=&warm_sample,.registration=record->handle.id,.native_slot=-1,.historical=status};
    return status;
}
static int vk_texture_history_sample_current_locked(int logical,uint64_t registration,VkImage image) {
    return registration_status==VK_TEXTURE_REGISTRATION_EXACT && logical==731 &&
        registration==warm_registration.handle.id && image==warm_sample.image;
}
static uint64_t token_of(uint64_t handle) { legacy_tokens++; return handle + 1; }
static int vk_texture_binding_receipt_locked(int native, VkImage image, RcRegistryPosition *write) {
    assert(!write);
    return image == warm_sample.image && (native == 258 || (native == 1255 && compatibility_current));
}
static int vk_texture_sample_unbacked_current_locked(VkTextureReadKey key) {
    (void)key; assert(0); return 0;
}
static int vk_texture_registration_expected_locked(int logical, VkImage *image) {
    (void)logical; (void)image;
    return 0;
}

static void test_warm_sample_selection(void) {
    tslots[258] = (TexSlot){.used=1,.pool_owner=7,.registration=&warm_registration,.entry=&warm_owner};
    VkTextureSample sample = {0};
    VkTextureAliasRead *alias = NULL;
    VkTexImage shape;
    alias_status = -1; // The raw alias probe would fail this qualified current source.
    assert(vk_texture_ordered_sample_select_locked(731, sample_read, 0, 0, &sample, &alias, &shape, NULL) == VK_TEXTURE_SAMPLE_READY);
'''

SAMPLE_ASSERTIONS = r'''
    assert(sample.image == &warm_sample && sample.canonical && sample.native_slot == 258 &&
           sample.registration == warm_registration.handle.id && !alias);
    assert(registration_calls == 1 && native_queries == 1 && !alias_queries);
    VkTextureSample rejected = {0};
    VkTextureAliasFailure wrong_read = {0};
    assert(vk_texture_ordered_sample_select_locked(731, (RcRegistryPosition){7, 16}, 0, 0,
            &rejected, &alias, &shape, &wrong_read) == VK_TEXTURE_SAMPLE_UNAVAILABLE);
    assert(wrong_read.phase == VK_TEXTURE_ALIAS_ORDER && wrong_read.status == RC_REGISTRY_STALE);
    assert(registration_calls == 1 && native_queries == 1 && !alias_queries && !alias);
    uint64_t handle = 731;
    @TOKEN@
    GraphicsSample selected = {.handle = handle, .token = token, .sample = sample, .image = *sample.image};
    assert(!legacy_tokens && graphics_sample_shape_current(&selected));
    TextureRegistration replacement = warm_registration;
    replacement.handle.id++;
    tslots[258].registration = &replacement;
    assert(!graphics_sample_shape_current(&selected)); // Same image handle cannot validate the wrong publication.
    tslots[258].registration = &warm_registration;
    canonical_current = 0;
    assert(!graphics_sample_shape_current(&selected));
    canonical_current = 1;

    registration_status = VK_TEXTURE_REGISTRATION_ABSENT;
    assert(vk_texture_ordered_sample_select_locked(731, sample_read, 0, 0, &sample, &alias, &shape, NULL) == VK_TEXTURE_SAMPLE_ABSENT);
    registration_status = VK_TEXTURE_REGISTRATION_GONE;
    assert(vk_texture_ordered_sample_select_locked(731, sample_read, 0, 0, &sample, &alias, &shape, NULL) == VK_TEXTURE_SAMPLE_GONE);
    registration_status = VK_TEXTURE_REGISTRATION_EXACT;
    assert(vk_texture_ordered_sample_select_locked(731, sample_read, 0, 0, &sample, &alias, &shape, NULL) == VK_TEXTURE_SAMPLE_READY);
    assert(sample.historical == VK_TEXTURE_REGISTRATION_EXACT && sample.registration == warm_registration.handle.id &&
           sample.native_slot == -1 && !alias && native_queries == 1 && !alias_queries);
    selected = (GraphicsSample){.handle = handle, .sample = sample, .image = *sample.image};
    assert(graphics_sample_shape_current(&selected));
    selected.sample.registration++;
    assert(!graphics_sample_shape_current(&selected));

    registration_status = VK_TEXTURE_REGISTRATION_CURRENT;
    alias_status = 1; // A forced native/specialty request still asks its actual compatible provider.
    assert(vk_texture_ordered_sample_select_locked(731, sample_read, VK_TEXTURE_READ_FORCE_COMPAT, 0, &sample, &alias, &shape, NULL) ==
           VK_TEXTURE_SAMPLE_READY && alias == acquired_alias && native_queries == 1 && alias_queries == 1);
    alias = NULL;
    warm_is_canonical = 0; // A mutable alias receives its separate explicit source path.
    alias_status = -1;
    VkTextureAliasFailure failure = {0};
    assert(vk_texture_ordered_sample_select_locked(731, sample_read, 0, 0, &sample, &alias, &shape, &failure) == VK_TEXTURE_SAMPLE_UNAVAILABLE);
    assert(failure.status == 17);
    alias_status = 0;
    assert(vk_texture_ordered_sample_select_locked(731, sample_read, 0, 0, &sample, &alias, &shape, NULL) == VK_TEXTURE_SAMPLE_READY && !alias);
    assert(native_queries == 3 && alias_queries == 3);
    alias_status = 1;
    expected_alias_options = VK_TEXTURE_ALIAS_SAMPLE_LINEAR_RGBA8_VIEW;
    assert(vk_texture_ordered_sample_select_locked(731, sample_read, VK_TEXTURE_READ_LINEAR_RGBA8_VIEW, 0, &sample, &alias, &shape, NULL) ==
           VK_TEXTURE_SAMPLE_READY && alias == acquired_alias);
    expected_alias_options = 0;
}
'''

SAMPLE_COMPATIBILITY_CASES = r'''
static void test_noncanonical_sample_admission(void) {
    TextureRegistration registration = {.handle.id = 6540, .publication = 5, .descriptor.serial = 80};
    Tex backing = {.t.serial = 40};
    tslots[1255] = (TexSlot){.used = 1, .pool_owner = 7, .registration = &registration, .entry = &backing};
    const GfxShaderRes *resource = &(GfxShaderRes){.exact = 1};
    uint64_t handle = 1255;
    VkTextureSample sample = {.image = &warm_sample, .registration = 6540, .native_slot = 1255};

    compatibility_current = 1;
    assert(choose_sample_admission(resource, handle, sample, 1) == VK_EXEC_GRAPHICS_COMPAT);
    assert(choose_sample_admission(resource, handle, sample, 0) == VK_EXEC_GRAPHICS_NEEDS_ORDERED);
    backing.t.serial = 80;
    assert(choose_sample_admission(resource, handle, sample, 0) == VK_EXEC_GRAPHICS_READY);
    assert(choose_sample_admission(resource, handle, sample, 1) == VK_EXEC_GRAPHICS_READY);
    compatibility_current = 0;
    assert(choose_sample_admission(resource, handle, sample, 1) == VK_EXEC_GRAPHICS_COMPAT);
    sample.canonical = 1;
    assert(choose_sample_admission(resource, handle, sample, 1) == VK_EXEC_GRAPHICS_READY);
    canonical_current = 0;
    assert(choose_sample_admission(resource, handle, sample, 0) == VK_EXEC_GRAPHICS_NEEDS_ORDERED);
    assert(choose_sample_admission(resource, handle, sample, 1) == VK_EXEC_GRAPHICS_UNAVAILABLE);
    canonical_current = 1;
    sample.canonical = 0;
    sample.projected = 1;
    compatibility_current = 1;
    assert(choose_sample_admission(resource, handle, sample, 1) == VK_EXEC_GRAPHICS_UNAVAILABLE);
    projected_current = 1;
    assert(choose_sample_admission(resource, handle, sample, 1) == VK_EXEC_GRAPHICS_READY);
    projected_current = 0;
}
'''

def sample_identity(draw, texture):
    select = function(draw, "graphics_stage_samples")
    token = next(line for line in select.splitlines() if "uint64_t token = sample.registration" in line)
    shared, test = SAMPLE_STUBS.split("static void test_warm_sample_selection(", 1)
    helper = function(texture, "vk_texture_ordered_sample_select_locked")
    provider = "\n".join(function(texture, name) for name in
                         ("vk_texture_select_sample_locked", "texture_read_key_slot",
                          "texture_read_key_compatibility", "vk_texture_resolved_receipt_locked"))
    chooser = """
static VkExecGraphicsResult choose_sample_admission(const GfxShaderRes *resource, uint64_t handle,
                                                      VkTextureSample sample, int ordered) {
    GraphicsSample selected = {.resource = resource, .handle = handle, .token = token_of(handle),
                               .sample = sample, .image = *sample.image};
    return graphics_sample_admit(&selected, ordered);
}
"""
    return "\n".join((shared, provider, helper, function(draw, "graphics_sample_shape_current"),
                      function(draw, "graphics_sample_admit"), "static void test_warm_sample_selection(" + test,
                      SAMPLE_ASSERTIONS.replace("@TOKEN@", token.strip()), chooser, SAMPLE_COMPATIBILITY_CASES))

TESTS = r'''
static VkExecTarget target(const VkTexImage *image, unsigned mip, unsigned layer) {
    return (VkExecTarget){
        .image = image->image, .img = image, .level = mip, .base_layer = layer, .layers = 1,
        .width = 80, .height = 40, .samples = 1,
    };
}

static void test_feedback(void) {
    VkTexImage color = {(VkImage)(uintptr_t)1, VK_IMAGE_ASPECT_COLOR_BIT, 1, 0};
    VkTexImage depth = {(VkImage)(uintptr_t)2, VK_IMAGE_ASPECT_DEPTH_BIT, 1, 1};
    VkTexImage no_usage = {(VkImage)(uintptr_t)3, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
    VkExec ex = {.pass_ncolor = 3, .pass_layers = 1, .in_pass = 1, .fbl_mask = 0x100};
    ex.color[0] = target(&color, 2, 3);
    ex.color[1] = target(&color, 1, 3);
    ex.color[2] = target(&no_usage, 0, 0);
    ex.depth = target(&depth, 0, 0);
    selected[0] = (SubRange){color.image, 2, 3, 3, 4};
    selected[1] = (SubRange){depth.image, 0, 1, 0, 1};
    selected[2] = (SubRange){no_usage.image, 0, 1, 0, 1};
    nselected = 3;
    assert(attached_as(&ex, &selected[0]) == 1);
    SubRange wrong_layer = {color.image, 2, 3, 2, 3};
    SubRange wrong_image = {(VkImage)(uintptr_t)4, 2, 3, 3, 4};
    assert(attached_as(&ex, &wrong_layer) == 0);
    assert(attached_as(&ex, &wrong_image) == 0);

    C.feedback = VK_FEEDBACK_LAYOUT;
    ex.fb_dirty = 0x107;
    GraphicsFeedback plan = feedback_plan(&ex, feedback_scan(&ex, &primary), 0, 0, C.feedback);
    assert(plan.sampled == 0x105 && plan.learn == 1 && plan.layouts == 0x103 && plan.restart);
    assert(ex.fb_dirty == 0x107 && ex.fbl_mask == 0x100 && !color.fb_learned && !begin_count && !end_count);
    feedback_apply(&ex, &plan, 1);
    assert((plan.sampled & ex.fbl_mask) == 0x101);
    /* Layout belongs to both views of the image, feedback to the actual overlap only. */
    assert(begin_count == 1 && end_count == 1 && ex.fbl_mask == 0x103);
    assert(color.fb_learned && !no_usage.fb_learned && ex.fb_dirty == 0);

    ex.fb_dirty = 1;
    GraphicsFeedback reopen = feedback_plan(&ex, plan.sampled, 1, 0, C.feedback);
    assert(reopen.layouts == 0x103 && !reopen.restart && !reopen.learn && !reopen.dirty);
    assert(feedback_prepare(&ex, &primary) == 0x101 && begin_count == 1);
    ex.fb_dirty = 1;
    tonemap_shape = 1;
    ex.fbl_mask = 0;
    assert(feedback_prepare(&ex, &primary) == 0 && begin_count == 2 && ex.fbl_mask == 0);

    tonemap_shape = 0;
    C.feedback = VK_FEEDBACK_RESTART;
    ex.fb_dirty = 1;
    assert(feedback_prepare(&ex, &primary) == 0 && begin_count == 3 && end_count == 3);
    C.feedback = VK_FEEDBACK_OFF;
    ex.fb_dirty = 1;
    assert(feedback_prepare(&ex, &primary) == 0 && begin_count == 3);
}

static void test_cold_native_feedback(void) {
    VkTexImage color = {(VkImage)(uintptr_t)31, VK_IMAGE_ASPECT_COLOR_BIT, 1, 0};
    VkTexImage depth = {(VkImage)(uintptr_t)32, VK_IMAGE_ASPECT_DEPTH_BIT, 1, 0};
    VkTexImage no_usage = {(VkImage)(uintptr_t)33, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
    VkExec ex = {.pass_ncolor = 2, .pass_layers = 1, .in_pass = 1};
    ex.color[0] = target(&color, 0, 0);
    ex.color[1] = target(&no_usage, 0, 0);
    ex.depth = target(&depth, 0, 0);
    const unsigned all = 0x103;
    const unsigned eligible = 0x101;
    const unsigned before_begin = begin_count;
    const unsigned before_end = end_count;
    C.feedback = VK_FEEDBACK_LAYOUT;
    GraphicsFeedback clean = feedback_plan(&ex, all, 0, 0, C.feedback);
    assert(clean.sampled == all && clean.restart && !clean.dirty);
    assert(clean.learn == eligible && clean.layouts == eligible);
    GraphicsFeedback initial = feedback_plan(&ex, all, 1, 0, C.feedback);
    assert(initial.restart && !initial.dirty && initial.learn == eligible && initial.layouts == eligible);
    assert(!color.fb_learned && !depth.fb_learned && !no_usage.fb_learned);
    assert(begin_count == before_begin && end_count == before_end);
    feedback_apply(&ex, &clean, 1);
    assert(color.fb_learned && depth.fb_learned && !no_usage.fb_learned);
    assert(ex.fbl_mask == eligible && begin_count == before_begin + 1 && end_count == before_end + 1);
    clean = feedback_plan(&ex, all, 0, 0, C.feedback);
    assert(!clean.restart && !clean.learn && clean.layouts == eligible);
    initial = feedback_plan(&ex, all, 1, 0, C.feedback);
    assert(!initial.restart && !initial.learn && initial.layouts == eligible);
    assert(!feedback_plan(&ex, 0, 0, 0, C.feedback).restart);
    assert(!feedback_plan(&ex, all, 1, 1, C.feedback).layouts);
    assert(!feedback_plan(&ex, all, 1, 0, VK_FEEDBACK_RESTART).layouts);
    assert(!feedback_plan(&ex, all, 1, 0, VK_FEEDBACK_OFF).layouts);
}

static void test_hud_shape(void) {
    VkTexImage color = {(VkImage)(uintptr_t)5, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
    VkExec ex = {.pass_ncolor = 1, .pass_layers = 1};
    ex.color[0] = target(&color, 0, 0);
    VkRenderingAttachmentInfo extra = {0};
    assert(!vk_dlaa_hud_attachment(&ex, &extra));

    D.hud_target = color.image;
    D.hl_at = D.presents = 7;
    D.alpha.w = 80;
    D.alpha.h = 40;
    D.alpha.view = (VkImageView)(uintptr_t)6;
#if REAL_HUD_SOURCE
    hud_use=(VkDlaaHudSource){.target=D.hud_target,.turn=D.hl_at,.coverage={D.alpha.w,D.alpha.h,D.alpha.view}};
    hud_sources.open=&hud_use;
#endif
    assert(vk_dlaa_hud_attachment(&ex, &extra) && extra.imageView == D.alpha.view);
    assert(incomplete == 0);

    VkPipeState state = {.fs = &primary, .color_fmt = {77}, .chmask.mask = 0xf};
    variant_available = 0;
    assert(!vk_hud_alpha_pipeline(&state));
    assert(state.color_fmt[0] == 77 && state.color_fmt[1] == NVN_FORMAT_R8);
    assert(state.fs == &primary && !(state.chmask.mask & 0xf0));
    variant_available = 1;
    assert(vk_hud_alpha_pipeline(&state));
    assert(state.fs == &coverage && (state.chmask.mask & 0xf0) == 0x10 && (state.color.blend_en & 2));

    ex.color[0].samples = 4;
    assert(!vk_dlaa_hud_attachment(&ex, &extra) && incomplete == 1);
    ex.color[0].samples = 1;
    D.presents++;
    assert(!vk_dlaa_hud_attachment(&ex, &extra) && incomplete == 1);
}

#if REAL_MESH_RMW
#if REAL_NATIVE_RMW_SPLIT
static void test_rmw_pass_order(void) {
    VkExec ex = {.fence = 7, .in_pass = 1, .cb = (VkCommandBuffer)(uintptr_t)4};
    VkExecGraphicsPass prior = {.fence = 7, .shape = {.in_pass = 1}};
    SelectedStorageWriters writers = {.rmw = &selected_word};
    VkExecGraphicsDraw draw = {.pass = &prior, .selected_writers = &writers};
    assert(graphics_native_rmw_order_for_draw(&ex, &draw));
    assert(ex.in_pass && draw.pass == &prior && native_order_calls == 1 && rmw_barriers == 1);
    prior.withheld = 1;
    assert(!graphics_native_rmw_order_for_draw(&ex, &draw));
    assert(native_order_calls == 1 && rmw_barriers == 1);
    prior.withheld = 0;
    writers.rmw = NULL;
    writers.rmw_bitmap = &selected_bitmap;
    draw.opcode = OP_DRAW_IDX;
    draw.pass = NULL; // Compatibility draw uses the exact rendering already open in the recorder.
    assert(graphics_native_rmw_order_for_draw(&ex, &draw));
    assert(ex.in_pass && !draw.pass && native_order_calls == 2 && bitmap_barriers == 1);
    native_split_refused = 1;
    assert(!graphics_native_rmw_order_for_draw(&ex, &draw));
    assert(native_order_calls == 3 && bitmap_barriers == 1);
    draw.opcode = OP_DRAW_MESH_IND;
    assert(graphics_native_rmw_order_for_draw(&ex, &draw));
    assert(native_order_calls == 3 && bitmap_barriers == 1);
}
#else
static void test_rmw_pass_order(void) {
    VkExec ex = {.fence = 7, .in_pass = 1, .cb = (VkCommandBuffer)(uintptr_t)4};
    VkExecGraphicsPass prior = {.fence = 7, .rt.target_registration = 17, .shape = {
        .in_pass = 1, .pass_w = 128, .pass_h = 128, .pass_layers = 1, .pass_scale = 1, .pass_samples = 1,
    }};
    SelectedStorageWriters writers = {.rmw = &selected_word};
    VkExecGraphicsDraw draw = {.pass = &prior, .selected_writers = &writers, .attachment_seeds = {&attachment_seed}};
    reopen_shape = prior.shape;
    reopen_shape.in_pass = 0;
    reopen_rt = prior.rt;
    reopen_withheld = 0;
    const unsigned ends = end_count;
    assert(graphics_native_rmw_order_for_draw(&ex, &draw));
    assert(!ex.in_pass && draw.pass == &reopened && rmw_barriers == 1 && ex.recorded == 1 && end_count == ends + 1);

    ex.in_pass = 1;
    draw.pass = &prior;
    prior.shape.hud_alpha_attachment = 1;
    assert(!graphics_native_rmw_order_for_draw(&ex, &draw));
    assert(ex.in_pass && draw.pass == &prior && rmw_barriers == 1 && end_count == ends + 1);

    prior.shape.hud_alpha_attachment = 0;
    draw.opcode = OP_DRAW_MESH_IND;
    assert(graphics_native_rmw_order_for_draw(&ex, &draw));
    assert(ex.in_pass && draw.pass == &prior && rmw_barriers == 1 && end_count == ends + 1);

    writers.rmw = NULL;
    writers.rmw_bitmap = &selected_bitmap;
    draw.opcode = OP_DRAW_IDX;
    assert(graphics_native_rmw_order_for_draw(&ex, &draw));
    assert(!ex.in_pass && draw.pass == &reopened && bitmap_barriers == 1 && rmw_barriers == 1 && end_count == ends + 2);
}
#endif
#endif

int main(void) {
    test_feedback();
    test_cold_native_feedback();
    test_hud_shape();
#if REAL_UBO_PASS_SPLIT
    test_ubo_pass_split();
#endif
#if REAL_MESH_RMW
    test_rmw_pass_order();
#endif
    test_required_one_shot();
#if REAL_FIXED_INDIRECT
    test_fixed_indirect_activation();
#endif
    test_warm_sample_selection();
    test_noncanonical_sample_admission();
    return 0;
}
'''


def fixture(sources, draw_source):
    draw = (draw_source or sources / "vk_exec_draw.c").read_text(encoding="utf-8")
    exec_source = (sources / "vk_exec.c").read_text(encoding="utf-8")
    dlaa = (sources / "vk_dlaa.c").read_text(encoding="utf-8")
    hud = (sources / "vk_hud_alpha.c").read_text(encoding="utf-8")
    texture = (sources / "vk_texture.c").read_text(encoding="utf-8")
    begin = function(exec_source, "pass_begin")
    a = begin.index("    VkImage want[9];")
    b = begin.index("    barrier_to_why(ex, want,", a)
    planned = begin.index("    int tonemap = !admitted && vk_dlaa_tonemap_target(ex);", a)
    loop = begin.index("    for (uint32_t i = 0; !tonemap && i < 9; i++)", planned)
    pass_mask = begin[a:planned] + begin[loop:b]
    start = draw.index("typedef struct {\n    uint32_t sampled, dirty, learn, layouts;")
    feedback = draw[start:draw.index("// By-region barrier inside the pass:", start)]
    compiler_at = draw.index("typedef struct {\n    VkExecGraphicsDraw *draw;")
    compiler = draw[compiler_at:draw.index("uint32_t vk_exec_pool_shader_kinds(", compiler_at)]
    byte_index = "static VkExecGraphicsResult graphics_widen_route(" in draw
    ubo_merge = "static int graphics_ubo_merges_current(" in draw
    activate = function(draw, "graphics_draw_activate")
    mesh_rmw = "graphics_native_rmw_order_for_draw(" in activate
    native_rmw_split = mesh_rmw and "vk_exec_record_between_renderings(" in function(draw, "graphics_native_rmw_order_for_draw")
    ubo_pass_split = "graphics_ubo_merge_record_for_draw(" in activate
    pass_shape, ubo_split = "", ""
    if ubo_pass_split:
        exec_header = (sources / "vk_exec.h").read_text(encoding="utf-8")
        pass_shape = re.search(r"^typedef struct(?: VkExecGraphicsPassShape)? \{[^{}]*\} VkExecGraphicsPassShape;", exec_header, re.M).group()
        split = function(draw, "graphics_ubo_merge_record_for_draw")
        assert "graphics_ubo_merge_record_preserving_pass(ex, draw->scratch)" in split, "UBO copy loses the active LOAD receipt"
        assert activate.index("graphics_ubo_merge_record_for_draw(") < activate.index("graphics_vertices_record(")
        compatibility = function(draw, "prepare_draw_phase")
        scratch = compatibility.index("graphics_scratch_ready(")
        materialize = compatibility.index("graphics_ubo_merge_record_preserving_pass(", scratch)
        assert scratch < materialize < compatibility.index("draw->activated = 1", materialize)
        actual_copy = function(draw, "ubo_merges_record_native")
        copy = actual_copy.index("vkCmdCopyBuffer(")
        assert actual_copy.index("VK_ACCESS_TRANSFER_READ_BIT") < copy
        if "VkAccessFlags shader_read" in actual_copy:
            assert actual_copy.index("merge->storage_uniform ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_UNIFORM_READ_BIT") < copy
            assert copy < actual_copy.index(".dstAccessMask = shader_read")
        else:
            assert copy < actual_copy.index("VK_ACCESS_UNIFORM_READ_BIT")
        actual_split = function(exec_source, "vk_exec_record_between_renderings")
        actual_split = actual_split.replace("vk_exec_record_between_renderings", "ubo_record_between_renderings")
        actual_split = actual_split.replace("vkCmdEndRendering(", "ubo_end_rendering(")
        actual_split = actual_split.replace("vkCmdBeginRendering(", "ubo_resume_rendering(")
        wrapper = function(draw, "graphics_ubo_merge_record_preserving_pass")
        wrapper = wrapper.replace("vk_exec_record_between_renderings(", "ubo_record_between_renderings(")
        declaration = re.search(r"^typedef struct \{[^{}]*\} GraphicsUboPassCopy;", draw, re.M).group()
        ubo_split = "\n".join((function(exec_source, "pass_attachment"), function(exec_source, "ordinary_rendering"),
            actual_split, declaration, function(draw, "graphics_ubo_merge_copy_between_passes"), wrapper))
    index_types = ()
    if byte_index:
        nvn_obj = (sources / "nvn_obj.h").read_text(encoding="utf-8")
        nvn = (sources / "nvn.h").read_text(encoding="utf-8")
        vk = (sources / "vk.h").read_text(encoding="utf-8")
        index_types = (
            re.search(r"^typedef struct \{[^{}]*\} p_draw_idx;", nvn_obj, re.M).group(),
            "enum {" + re.search(r"NVN_INDEX_TYPE_UNSIGNED_BYTE\s*=\s*\w+", nvn).group() + "};",
            re.search(r"typedef enum \{[^{}]*\} VkExecHostInputResult;", vk).group(),
        )
    return "\n".join((
        f"#define REAL_FIXED_INDIRECT {int('static VkExecGraphicsResult graphics_indirect_select(' in draw)}",
        f"#define REAL_HUD_SOURCE {int('VkDlaaHudSource *use = hud_sources.open' in function(dlaa,'vk_dlaa_hud_attachment'))}",
        f"#define REAL_BYTE_INDEX {int(byte_index)}",
        f"#define REAL_UBO_MERGE {int(ubo_merge)}",
        f"#define REAL_MESH_RMW {int(mesh_rmw)}",
        f"#define REAL_NATIVE_RMW_SPLIT {int(native_rmw_split)}",
        f"#define REAL_UBO_PASS_SPLIT {int(ubo_pass_split)}",
        f"#define REAL_UBO_NATIVE_RECORD {int(ubo_pass_split or 'graphics_ubo_merge_record_native(' in activate)}",
        f"#define REAL_VERTEX_INPUT {int('static int graphics_vertices_record(' in draw)}",
        f"#define REAL_SCRATCH_FAILURE {int('static VkExecGraphicsResult graphics_scratch_failure(' in draw)}",
        f"#define REAL_SCRATCH_WAIT {int('graphics_scratch_wait(' in draw)}",
        STUBS, function(draw, "feedback_overlap"), function(draw, "attached_as"), SEAM, pass_mask,
        "    (void)wasp;\n    return fbl;\n}",
        ADAPTERS, feedback, function(dlaa, "hud_attachment_target") if "static const VkExecTarget *hud_attachment_target(" in dlaa else "",
        function(dlaa, "vk_dlaa_hud_attachment"),
        function(hud, "vk_hud_alpha_pipeline"), *index_types, pass_shape, READY_STUBS, ubo_split, compiler,
        function(draw, "graphics_widen_route") if byte_index else "",
        function(draw, "graphics_ubo_merges_current") if ubo_merge else "",
        function(draw, "graphics_scratch_failure") if "static VkExecGraphicsResult graphics_scratch_failure(" in draw else "",
        function(draw, "graphics_scratch_wait") if "graphics_scratch_wait(" in draw else "",
        function(draw, "graphics_scratch_revision_current"),
        function(draw, "graphics_scratch_ready"),
        FIXED_ARGUMENT_STUBS if "graphics_indirect_select(" in draw else "",
        function(draw, "graphics_argument_select") if "graphics_argument_select(" in draw else "",
        function(draw, "graphics_indirect_select") if "graphics_indirect_select(" in draw else "",
        function(draw, "graphics_ubo_merge_has_native") if ubo_pass_split else "",
        function(draw, "graphics_ubo_merge_record_for_draw") if ubo_pass_split else "",
        function(draw, "graphics_native_rmw_order") if native_rmw_split else "",
        function(draw, "graphics_native_rmw_order_for_draw") if mesh_rmw else "",
        'static void graphics_marked_runtime_failure(VkExecGraphicsDraw *draw, const char *stage) '
        '{ assert(draw && !strcmp(stage, "issue-order")); }' if "graphics_marked_runtime_failure(" in activate else "",
        # These draws have no replacement LOAD. Their ordinary activation still reaches the real downstream checks.
        "typedef struct EffectDeclaration EffectDeclaration; static const EffectDeclaration *effect_current; "
        "static int graphics_load_effects_issue_current(const VkExecGraphicsDraw *draw, const EffectDeclaration *effects) "
        "{ assert(draw && !effects); return 1; }" if "graphics_load_effects_issue_current(" in activate else "",
        activate,
        READY_TESTS, sample_identity(draw, texture), TESTS,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path(__file__).resolve().parents[1] / "runtime")
    parser.add_argument("--draw-source", type=Path, help="Read draw source from a recorded checkpoint")
    parser.add_argument("--emit", type=Path, help="Only emit extracted C at this path")
    args = parser.parse_args()
    source = fixture(args.sources, args.draw_source)
    if args.emit:
        args.emit.write_text(source, encoding="utf-8")
        return
    cc = os.environ.get("CC") or shutil.which("clang")
    if not cc:
        raise SystemExit("Clang is required (set CC)")
    cc = Path(shutil.which(cc) or cc).resolve()
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="graphics-shape-") as directory:
        c = Path(directory) / "check.c"
        exe = Path(directory) / ("check.exe" if os.name == "nt" else "check")
        c.write_text(source, encoding="utf-8")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=undefined", str(c), "-o", str(exe)],
                       env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True, timeout=20)
    print("Graphics readiness: exact sampled receipts, cold draw, feedback and HUD shape passed")


if __name__ == "__main__":
    main()

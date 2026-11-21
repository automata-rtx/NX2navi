// The op-stream replay of the Vulkan back end. vk_exec.c walks the command
// handles of a submission, keeps render targets, passes, clears and fixed-function state; draws, binds and dispatches
// (vk_exec_draw.c) and copies, counters, events and conditional rendering (vk_exec_xfer.c) plug in through the two
// hooks at the bottom.
// Runs on the submitting thread under vk_sync_submit's lock: no locking inside.
// Contains code Copyright (c) 2026 jhaizhou, Copyright (c) 2026 Peiikai (MIT; see THIRD_PARTY_NOTICES.md).
#pragma once
#include <stdint.h>

#include "nvn_api.h"
#include "vk.h"
#include "vk_pool_publication.h"
#include "resource_core_registry.h"
#include "resource_command_effects.h"
#include "resource_core_vma.h"
#include "texture_resource.h"

// Dynamic state the next draw must record (VkExec.dirty). vk_exec_record sets them all at the start of each command
// buffer; the state ops set theirs; vk_exec_draw.c clears what it recorded.
enum {
    VK_EXEC_DIRTY_VIEWPORT = 1 << 0,     // viewports or depth range
    VK_EXEC_DIRTY_SCISSOR = 1 << 1,
    VK_EXEC_DIRTY_DEPTH_BIAS = 1 << 2,   // polygon offset
    VK_EXEC_DIRTY_BLEND_CONST = 1 << 3,
    VK_EXEC_DIRTY_STENCIL = 1 << 4,      // compare and write masks, references
    VK_EXEC_DIRTY_ALL = (1 << 5) - 1,
};

// One render target of OP_SET_RT, resolved to its vk_texture.c image for the current command buffer.
typedef struct {
    const tex_t *tex;                    // texture pin of ex->rt (nvn_pins_sweep keeps it); NULL: no target in this slot
    VkImage image;                       // VK_NULL_HANDLE: no target or not renderable (logged once)
    VkFormat format;                     // view format: the view's when it reinterprets a same-size format
    VkImageViewType view_type;           // 2D, or 2D_ARRAY for more than one layer
    VkImageAspectFlags aspect;
    uint32_t level, base_layer, layers;  // subresource of the view; of a 3D image: depth planes
    uint8_t is3d;                        // a 3D image (colour only), viewed through 2D views of its slices
    uint8_t blocks;                      // a BC image viewed as one texel per block (VkTexImage.block_views)
    uint32_t width, height, samples;     // extent of that level; with blocks, in blocks
    int32_t nvn_format;                  // NVNformat of the view (VkPipeState.color_fmt / depth_fmt)
    const VkTexImage *img;               // the image (fb_usage, fb_learned)
    uint8_t scale;                       // VkTexImage.scale
    uint8_t exact_snapshot;              // pass-owned shape, with command-buffer-owned attachment view
    VkImage bc_backing;                   // a 3D block stand-in copies its written planes to this image
    uint64_t bc_address;
    uint32_t bc_level;
    VkResult bc_status;                   // failure to realize a supported block stand-in
} VkExecTarget;

// Value identity captured after pass preparation. The actual attachment lifetime remains owned by recording;
// registry proof checks the hold incarnation before accepting any later binding receipt.
typedef struct {
    RcRegistryImageBinding binding;
    RcSourceRange range;
    RcRegistryImageWriteExtent *extents;
    uint32_t extent_count;
    VkPinnedTextureWrite pinned;
    int private_detached;                 // a retained BC backing lost projection before this pass wrote
    VkImageSubresourceRange loaded_planes;
    VkTextureAliasRead *load_seed;
} VkExecTargetBinding;

// Native compute replacement may also fill the captured scene color/depth borders. These exact physical owners
// are captured before declaration sealing; execution must refuse the optional reconstruction if they change.
typedef struct VkDlaaSceneEffects {
    VkImage images[2];
    uint64_t addresses[2];
    RcSourceRange ranges[2];
    RcRegistryImageBinding owners[2];
    ResourceImageEffectRole border[2];
    VkTextureProjectedTransfer *edge[2];
    int complete;
} VkDlaaSceneEffects;
int vk_dlaa_scene_effects(VkDlaaSceneEffects *out);
void vk_dlaa_scene_prepare_edges(VkDlaaSceneEffects *, const VkTextureColorBlitSource *);
void vk_dlaa_scene_release_edges(VkDlaaSceneEffects *);
void vk_dlaa_scene_effects_cancel_unselected(VkDlaaSceneEffects *, uint32_t selected);

void vk_exec_cond_frame_end(uint64_t frame);   // vk_exec_xfer.c: a present, in replay order (NX2NAVI_LOG_RCOND)
extern unsigned vk_cond_clears_run, vk_cond_clears_skip, vk_cond_draws_skip, vk_cond_depth_run;
void vk_exec_cond_why(const p_render_cond *c, char *buf, size_t cap);   // NX2NAVI_LOG_RCOND=ab

#define NVN_STAGE_FS 1
void vk_exec_discard_held(void); // serialized backend teardown entry point
void vk_exec_discard_held_locked(void);
int vk_exec_held_ready_locked(void);
int vk_exec_pool_registration_needs_prefix(uint64_t fence);
void vk_exec_pool_registration_prefix_complete(uint64_t fence);

// The replay state. One instance lives for the run: NVN state is queue state and carries over from one submission to
// the next. The "per command buffer" fields are reset by vk_exec_record.
typedef struct VkExec {
    // ---- per command buffer
    struct DrawCtx *dc;                  // vk_exec_draw.c: the draw context this state records with (NULL until a
                                         // draw op names it: the calling thread's, vk_exec_draw_ctx_use)
    VkCommandBuffer cb;                  // being recorded
    uint64_t fence;                      // timeline value the submission signals (vk_completion_add)
    uint64_t resource_next, resource_op; // scene-local logical operation assignment; input precedes its output
    VkTextureRegistrationScope *registration_scope;
    int recorded;                        // Vulkan commands recorded so far; every handler adds the ones it records
    int tonemap_drawn;                   // the full-screen tonemap draw was actually issued in this pass
    VkExtent2D tonemap_extent;           // nonzero while drawing the guest tonemap at DLSS output size
    int in_pass;                         // inside vkCmdBeginRendering
    uint32_t dirty;                      // VK_EXEC_DIRTY_*
    // Pass marks (vk.h vk_pass_mark_begin): the open pass's record (NULL: none or marks off), vk_exec_frame.draws when
    // it began, and the passes and dispatches this submission marked so far.
    VkPassMark *mark;
    uint64_t mark_draws0;
    uint32_t mark_passes, mark_dispatches;

    // ---- render targets (OP_SET_RT) and the current pass
    p_set_rt rt;                         // last OP_SET_RT payload (a change ends the open pass); rt.n 0, no depth: none
    int rt_resolved;                   // color[] and depth hold this command buffer's images for rt
    VkExecTarget color[8], depth;        // resolved targets
    VkImageView pass_views[9];           // exact native color and depth/stencil views at ordinary pass begin
    VkExecTargetBinding target_bindings[9]; // ordinary guest attachments of the active pass, never stand-ins
    uint32_t target_binding_mask;
    uint32_t bc_target_mask;             // native stand-ins, distinct from ordinary guest attachment writes
    uint32_t bc_written_mask;            // drawn/cleared stand-ins with a real copy back
    uint32_t pass_w, pass_h, pass_layers;            // render area and layer count of the open (or next) pass
    uint32_t pass_scale;                             // its targets' VkTexImage.scale: game viewports and scissors
                                                     // are multiplied by it
    int pass_noatt;                                   // the pass has no attachment: its size follows the draws (vk_exec_noatt_dims)
    VkFormat pass_color[8], pass_depth, pass_stencil; // attachment formats of the pass (VkPipelineRenderingCreateInfo);
                                                      // VK_FORMAT_UNDEFINED: none
    uint32_t pass_ncolor, pass_samples;               // colorAttachmentCount and rasterization samples of the pass

    // ---- fixed-function state (vk_exec.c keeps it; vk_exec_draw.c records it before draws)
    float vp[16][6];                     // OP_VIEWPORTS: x, y, w, h, near, far
    int has_vp;                          // a viewport was set; else the render area
    int vp_jittered;                     // the viewport recorded last took the DLAA jitter (vk_dlaa.c vk_dlaa_viewport)
    p_scissor scissor;                   // OP_SCISSOR (scissor 0)
    int has_scissor;                     // a scissor was set; else unbounded
    float depth_range[2];                // OP_DEPTH_RANGE: overrides the viewports' near and far
    uint32_t st_vmask[2], st_wmask[2], st_ref[2];   // OP_STENCIL per face (0 front, 1 back): compare mask, write mask, ref
    float point_size;                    // OP_BIND_STATE kind 6: NVN point size for the support buffer; 0 = never set = 1
    p_poly_offset poly_offset;           // OP_POLY_OFFSET: depth bias factor, units, clamp
    p_prim_restart prim_restart;         // OP_PRIM_RESTART: applied to indexed strip and fan draws (op_draw_idx)
    float blend_color[4];                // OP_BLEND_COLOR
    int64_t copy_stride;                 // OP_COPY_STRIDE: row stride of buffer <-> texture copies (vk_exec_xfer.c)
    uint64_t tex_pool;                   // OP_SET_TEX_POOL's copied incarnation
    int32_t tex_pool_count;
    const dpool_t *samp_pool;             // OP_SET_SAMP_POOL
    // Pipeline inputs. vk_exec.c writes depth_clamp (OP_DEPTH_CLAMP), raster_discard (OP_RASTER_DISCARD) and, at pass
    // begin, color_fmt, depth_fmt and samples; vk_exec_draw.c owns every other field (shaders, prim, y_flip, state
    // objects, vertex input).
    VkPipeState pipe;

    // ---- vk_exec_draw.c slots (draws, binds, dispatch): left empty by vk_exec.c
    const prog_t *prog;                  // OP_BIND_PROGRAM
    int32_t prog_stages;
    NVNbufferRange vbuf[16];             // OP_BIND_VBUF
    p_bind_buf ubo[8][18], ssbo[8][16];  // OP_BIND_UBO / OP_BIND_SSBO per NVN stage (6 task, 7 mesh) and index
    uint8_t cs_push[1024];                             // OP_PUSH_CC: NVN2 compute push constants (c0 from 0x10)
    uint64_t tex[8][128], samp[8][128], img[8][32];   // OP_BIND_TEX / OP_BIND_SAMP / OP_BIND_IMG handles per stage

    // ---- vk_exec_xfer.c slots (render enable, conditional rendering): left empty by vk_exec.c
    int render_disabled;                 // OP_RENDER_ENABLE false or a failed OP_RENDER_COND: clears and draws skip
    p_render_cond render_cond;           // last OP_RENDER_COND
    int has_render_cond;

    // ---- internal to vk_exec.c
    void *views;                         // image views created for this command buffer (freed at its fence)
    int nviews, cap_views;
    int barrier_at;                      // recorded right after the last full barrier (-1: none yet)
    // NX2NAVI_BARRIER_GAME (vk_exec.c barrier_to_why): the last full barrier was one the replay itself needed (a copy,
    // clear or fill of its own follows it), so the next pass or dispatch gets a full one too; passes and dispatches
    // begun since the last full barrier
    int bar_dirty, bar_passes, bar_dispatches;

    // Feedback loops (vk.h VkCtx.feedback, vk_exec_draw.c feedback_prepare). Learned targets are attached in
    // ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT; they stay in it across consecutive passes and go back to GENERAL at the next
    // full barrier that is not a pass begin.
    VkImage fbl_img[9];                  // images in the feedback layout now
    VkImageAspectFlags fbl_aspect[9];
    int nfbl;
    uint32_t fbl_mask;                   // attachments of the open pass in that layout: bit i colour i, bit 8 depth
    uint32_t fb_dirty;                   // attachments of the open pass written by a draw or clear since the pass began
                                         // or the last in-pass barrier
    uint32_t fb_read;                    // attachments of the open pass a draw sampled (feedback layout) since the pass
                                         // began or the last in-pass read-to-write barrier (feedback_war)
    const void *fb_fs;                   // feedback_scan's memo: fragment shader, texture bind serial, pass serial
    uint32_t fb_serial, fb_pass, fb_sampled;
    uint64_t fb_authority;
    uint32_t tex_serial, pass_serial;    // bumped by every texture or image bind and pass begin
    int hud_alpha_attachment;             // this pass also records game HUD opacity
} VkExec;

// Begins dynamic rendering with the current targets unless a pass is open (a full memory barrier precedes it). Call
// before a draw or an attachment clear.
void vk_exec_pass_begin(VkExec *ex);
// Record non-rendering work between two native instances of the same ordinary LOAD/STORE pass.
// The callback emits to the supplied command buffer without altering replay state or handing off work.
int vk_exec_record_between_renderings(VkExec *, int (*record)(VkCommandBuffer, void *), void *);
void vk_exec_targets_prepare(VkExec *ex); // validates captured pinned attachments before an actual graphics draw
void vk_exec_targets_written(VkExec *ex); // after an actual graphics draw; reuses the pass's captured identities
// Ends the open pass (no-op without one) and marks its targets GPU-written. Call before copies and dispatches.
void vk_exec_pass_end(VkExec *ex);
// Full memory barrier (all commands, all memory); ends the open pass first. Copies and dispatches order themselves with
// it. Skipped when nothing was recorded since the last one.
void vk_exec_barrier(VkExec *ex);
// The barrier before a game dispatch: a full one, or with NX2NAVI_BARRIER_GAME none unless the replay's own commands
// came since the last full barrier (vk_exec.c barrier_to_why). Ends the open pass.
void vk_exec_barrier_dispatch(VkExec *ex);
// NX2NAVI_TRACE=perf, every 600 frames: the barriers per frame by kind.
void vk_exec_barrier_log(uint64_t frame);
// vk_texture.c (NX2NAVI_TEX_ALIAS): commands that must run at the replay's current position. vk_exec_inline_begin returns
// the command buffer being recorded after a full barrier (an open pass is ended; *reopen says so), or VK_NULL_HANDLE
// when this thread is not recording a replay. vk_exec_inline_end begins the pass again when it was open.
VkCommandBuffer vk_exec_inline_begin(int *reopen);
void vk_exec_inline_end(int reopen);
// A test's hook (tex_probe.c): between a call with cb and one with VK_NULL_HANDLE, the calling thread counts as
// recording a replay into cb for submission `fence`, so vk_exec_inline_begin hands cb out. Called from the record
// callback of vk_sync_submit_own, where a replay would be recorded.
void vk_exec_probe_inline(VkCommandBuffer cb, uint64_t fence);
void vk_exec_probe_pass_arm(void (*record)(VkExec *, void *), void *context);
void vk_exec_probe_mesh_partial(const void *payload);
int vk_exec_probe_mesh_partial_allowed(const void *payload);
// vk_exec_draw.c, the replay workers: a draw context of their own (what draws keep from one to the next while they
// record: vk_exec_draw.c DrawCtx), and the one the calling thread records with from now on (NULL: the replay's).
void *vk_exec_draw_ctx_new(void);
void vk_exec_draw_ctx_use(void *ctx);
// Device-wide fixed draw resources are prepared before nonempty replay begins. VK_NOT_READY reports the exact
// required admission peak. The sync helper may wait and must be called outside recording and provider locks.
int vk_exec_draw_fixed_ready(void);
VkResult vk_exec_draw_fixed_prepare(RcAdmissionVector *peak);
int vk_sync_prepare_draw(void);
VkResult vk_exec_draw_storage_standins_prepare(VkExec *, const VkPipeInfo *, int allow_ordered);
void vk_exec_draw_fallback_shutdown(void);
void vk_exec_draw_fixed_shutdown(void); // After replay, upload and completion drain, before VMA teardown.
void vk_exec_draw_mfb_stop(void);       // Close intake and join CPU conversion before replay teardown.
void vk_exec_draw_mfb_shutdown(void);   // After healthy pipeline cache teardown, release transformed shaders.
void vk_exec_draw_native_mesh_shutdown(void); // After ordered recording and count results drain.
// Release reusable mapped draw scratch after replay and completion drain, before VMA teardown.
void vk_exec_draw_scratch_shutdown(void);
void vk_exec_draw_descriptor_shutdown(void);
void vk_exec_draw_bindless_shutdown(void); // After queue/recorder drain and pipeline layout destruction.

typedef struct VkExecDispatchImages VkExecDispatchImages;
typedef struct VkExecBufferScratchReceipt VkExecBufferScratchReceipt;
typedef struct {
    struct CbRes *owner;
    const VkPipeInfo *info;
    uint64_t ordinal;
    struct DrawDescriptorCell *set[2];
    struct RawStorageInputs *raw_storage;
    struct OrdinaryStorageInputs *storage_writes;
    VkResult refusal;
    uint32_t refused_set;
} DrawDescriptorGrant;

// Value-only provenance copied at the original successful or refused selection.
typedef struct {
    RcRegistryPosition read;
    const GfxShader *shader;
    const char *gate, *geometry;
    VkPipeline pipeline;
    const VkGraphicsPipelineRevision *revision;
    VkTextureSampleRefusal legacy;
    VkTextureAliasFailure alias;
    uint64_t program, handle, registration;
    VkImage image;
    uint64_t address;
    uint32_t width, height, depth, levels, layers, samples, format, aspect;
    int logical, stage, set, binding, member, translated_set, translated_binding, status;
    uint32_t matching_sites;
    uint8_t present, native_exact, geometry_only, table;
    uint8_t attachment_seed; // One-based attachment slot when an owned LOAD seed prevents switching.
} VkExecSampledFailure;

enum {
    VK_DISPATCH_TRACE_BINDING = 1u << 0,
    VK_DISPATCH_TRACE_HANDLE = 1u << 1,
    VK_DISPATCH_TRACE_SELECTION = 1u << 2,
    VK_DISPATCH_TRACE_IMAGE = 1u << 3,
    VK_DISPATCH_TRACE_VIEW = 1u << 4,
    VK_DISPATCH_TRACE_REGISTRATION = 1u << 5,
    VK_DISPATCH_TRACE_HISTORY = 1u << 6,
    VK_DISPATCH_TRACE_BYTES = 1u << 7,
    VK_DISPATCH_TRACE_PLANES = 1u << 8,
};
enum {
    VK_DISPATCH_TRACE_HOLD_PRESENT = 1u << 0,
    VK_DISPATCH_TRACE_HOLD_ALIAS = 1u << 1,
    VK_DISPATCH_TRACE_HOLD_STORAGE = 1u << 2,
    VK_DISPATCH_TRACE_HOLD_WRITE = 1u << 3,
    VK_DISPATCH_TRACE_HOLD_PIN = 1u << 4,
    VK_DISPATCH_TRACE_HOLD_QUERY = 1u << 5,
};
typedef struct {
    const char *role;
    uint64_t handle, registration, image, address, bytes;
    uint32_t known, holds, width, height, depth, format, type, levels, layers, samples, aspect;
    uint32_t count, flags, resource_type;
    int set, binding, stage, kind, descriptor_type, slot, member, history;
    int has_view;
    view_t view;
    VkImageSubresourceRange planes;
} VkExecDispatchTraceOperand;

typedef struct {
    // Only static labels and values copied before prepared resource release.
    const char *gate, *detail;
    uint8_t shader_key[32], shader_known, native_exact;
    VkExecDispatchTraceOperand output, input;
} VkExecDispatchInputTrace;

typedef struct {
    const char *stage, *reason;
    int result, subtype, binding, slot, member;
    uint64_t image, peer_image, address, bytes, handle;
    VkExecSampledFailure sampled;
    VkExecDispatchInputTrace trace;
} VkExecDispatchInputFailure;

typedef struct VkExecDispatch {
    const GfxShader *shader;
    const VkPipeInfo *info;
    VkPipeline pipeline;
    const prog_t *program;
    uint64_t program_serial, registration;
    uint32_t opcode;
    const void *payload;
    RcRegistryPosition read_position;
    VkExecDispatchImages *selection;
    VkExecBufferScratchReceipt *scratch;
    DrawDescriptorGrant descriptors;
    VkExecDispatchInputFailure input_failure;
    uint8_t compatibility;
    struct VkExecPoolGrant *pool_grant;
    VkMemSpan indirect;
    VkMemSpanCapture *indirect_hold;
    int indirect_native;
} VkExecDispatch;

void vk_exec_dispatch_input_trace_log(const VkExecDispatch *, const char *stage, int status);

typedef enum {
    VK_EXEC_DISPATCH_STOPPED = -1,
    VK_EXEC_DISPATCH_COMPAT = 0,
    VK_EXEC_DISPATCH_NATIVE = 1,
    VK_EXEC_DISPATCH_NOOP = 2
} VkExecDispatchResult;
typedef enum {
    VK_EXEC_DISPATCH_EFFECT_STOPPED = -1,
    VK_EXEC_DISPATCH_EFFECT_COMPAT = 0,
    VK_EXEC_DISPATCH_EFFECT_READY = 1
} VkExecDispatchEffectResult;

// COMPAT is selected before any immutable dispatch promise. NATIVE transfers its image receipt only after Ready.
VkExecDispatchResult vk_exec_dispatch_select(VkExec *ex, uint32_t opcode, const void *payload,
    const VkMemSpan *retry_span, VkMemSpanCapture *retry_capture, VkExecDispatch *dispatch);
int vk_exec_dispatch_compat_inputs_select(VkExec *ex, uint32_t opcode, const void *payload, VkExecDispatch *dispatch);
const VkPipeBinding *vk_exec_dispatch_compat_storage_binding(const VkExecDispatch *dispatch);
int vk_exec_dispatch_has_compat_inputs(const VkExecDispatch *dispatch);
void vk_exec_dispatch_cancel(VkExecDispatch *dispatch);
VkExecDispatch *vk_exec_dispatch_enter(VkExecDispatch *dispatch);
void vk_exec_dispatch_validate(const VkExec *ex, VkExecDispatch *dispatch, uint32_t opcode, const void *payload);
VkExecDispatchEffectResult vk_exec_dispatch_effect_prepare(VkExec *ex, uint32_t opcode, const void *payload,
                                                    const GfxShader **selected, VkExecDispatchImages **out);
void vk_exec_dispatch_effect_cancel(VkExecDispatchImages **out);
void *vk_exec_effects_declare_dispatch(VkExec *ex, uint32_t opcode, const void *payload, const GfxShader *selected,
                                       VkExecDispatchImages **inputs, RcRegistryPosition position,
                                       VkExecBufferScratchReceipt **scratch);

typedef enum {
    VK_EXEC_SCRATCH_PASS,
    VK_EXEC_SCRATCH_READY,
    VK_EXEC_SCRATCH_NEED_CAPACITY,
    VK_EXEC_SCRATCH_INVALIDATED,
    VK_EXEC_SCRATCH_UNAVAILABLE,
    VK_EXEC_SCRATCH_STOPPED,
} VkExecBufferScratchResult;
typedef struct {
    RcVmaDependency native;
    RcBudgetReadiness host_before;
    RcAdmissionVector host_peak;
    uint64_t bytes;
    int host_only;
} VkExecBufferScratchWait;

// Only admitted layouts with a complete scratch demand produce a receipt.
VkExecBufferScratchResult vk_exec_draw_buffer_scratch_prepare(VkExec *, VkExecDispatch *, VkExecBufferScratchWait *);

typedef struct VkExecGraphicsPass VkExecGraphicsPass;
typedef struct VkExecPreparationFailure VkExecPreparationFailure;
typedef enum {
    VK_EXEC_GRAPHICS_PASS_READY,
    VK_EXEC_GRAPHICS_PASS_NEEDS_ORDERED,
    VK_EXEC_GRAPHICS_PASS_COMPAT,
    VK_EXEC_GRAPHICS_PASS_STOPPED,
    VK_EXEC_GRAPHICS_PASS_UNAVAILABLE,
} VkExecGraphicsPassResult;

typedef struct VkExecGraphicsPassShape {
    VkExecTarget color[8], depth;
    VkFormat pass_color[8], pass_depth, pass_stencil;
    int32_t pipe_color_fmt[8], pipe_depth_fmt;
    uint32_t pass_w, pass_h, pass_layers, pass_scale, pass_ncolor, pass_samples;
    int pass_noatt, in_pass;
    VkExtent2D tonemap_extent;
    int hud_alpha_attachment;
    VkImage feedback_images[9];
    VkImageAspectFlags feedback_aspects[9];
    int feedback_count;
    uint32_t feedback_dirty;
    uint32_t fbl_mask;
} VkExecGraphicsPassShape;

VkExecGraphicsPassResult vk_exec_graphics_pass_capture(VkExec *ex, int allow_ordered, VkExecGraphicsPass **out);
VkExecGraphicsPassResult vk_exec_graphics_pass_capture_seeded(VkExec *, int allow_ordered, VkTextureAliasRead *seeds[9],
                                                             VkExecGraphicsPass **, VkExecPreparationFailure *);
// The transformed mesh operation closes any active pass before its compute phases. Capture its future
// pass without running hooks or recording. Activate only after the caller has closed the prior pass.
VkExecGraphicsPassResult vk_exec_graphics_pass_capture_reopen(VkExec *ex, int allow_ordered, VkExecGraphicsPass **out);
const VkExecGraphicsPassShape *vk_exec_graphics_pass_shape(const VkExecGraphicsPass *pass);
int vk_exec_graphics_pass_reopen_match(const VkExecGraphicsPass *before, const VkExecGraphicsPass *after);
int vk_exec_graphics_pass_reopen_allowed(const VkExecGraphicsPass *pass);
int vk_exec_graphics_pass_attachment_peers(const VkExecGraphicsPass *, VkExecPreparationFailure *);
typedef RcRegistryResult (*VkExecAttachmentAccessCheck)(const void *, VkImage, RcInterval, RcInterval *overlap);
int vk_exec_graphics_pass_attachment_access(const VkExecGraphicsPass *, VkExecAttachmentAccessCheck,
                                           const void *, VkExecPreparationFailure *);
typedef enum {
    VK_EXEC_ATTACHMENT_READ_DISJOINT,
    VK_EXEC_ATTACHMENT_READ_PRE_DRAW,
    VK_EXEC_ATTACHMENT_READ_FRAGMENT,
} VkExecAttachmentRead;
int vk_exec_graphics_pass_attachment_read(const VkExecGraphicsPass *, VkImage, const RcInterval *, uint32_t count,
                                         uint32_t slot, VkExecAttachmentRead, VkExecPreparationFailure *);
int vk_exec_graphics_pass_geometry_read(const VkExec *, const VkExecGraphicsPass *, VkImage, RcInterval, int active);
int vk_exec_graphics_pass_current(const VkExec *ex, const VkExecGraphicsPass *pass);
int vk_exec_graphics_pass_activate(VkExec *ex, VkExecGraphicsPass *pass);
void vk_exec_graphics_pass_cancel(VkExecGraphicsPass **pass);

typedef struct VkExecGraphicsDraw VkExecGraphicsDraw;
typedef struct VkExecGraphicsScratch VkExecGraphicsScratch;
struct VkExecGpaSampleScope;
typedef enum {
    VK_EXEC_GRAPHICS_COMPAT,
    VK_EXEC_GRAPHICS_NOOP,
    VK_EXEC_GRAPHICS_READY,
    VK_EXEC_GRAPHICS_NEEDS_ORDERED,
    VK_EXEC_GRAPHICS_STOPPED,
    VK_EXEC_GRAPHICS_UNAVAILABLE,
    VK_EXEC_GRAPHICS_INVALIDATED,
} VkExecGraphicsResult;

typedef enum {
    VK_EXEC_PREPARATION_NONE,
    VK_EXEC_PREPARATION_IMAGE,
    VK_EXEC_PREPARATION_DESCRIPTOR,
    VK_EXEC_PREPARATION_SCRATCH,
    VK_EXEC_PREPARATION_POOL_IMAGE,
    VK_EXEC_PREPARATION_PIPELINE,
} VkExecPreparationOwner;

typedef enum {
    VK_EXEC_PREPARATION_MAPPING,
    VK_EXEC_PREPARATION_ORDER,
    VK_EXEC_PREPARATION_PLACEMENT,
    VK_EXEC_PREPARATION_PRESSURE,
    VK_EXEC_PREPARATION_ATTACHMENT,
    VK_EXEC_PREPARATION_FEEDBACK,
    VK_EXEC_PREPARATION_DESCRIPTOR_SOURCE,
    VK_EXEC_PREPARATION_DESCRIPTOR_SET,
} VkExecPreparationReason;

typedef struct {
    const char *gate;
    RcRegistryPosition read;
    RcInterval range;
    uint64_t address, bytes;
    unsigned stage, set, binding, marker, sites, descriptor_count, writer_count;
    unsigned revision, mode, retained, compatibility;
    int detail, registry;
    VkMemNativeRmwBitmapFailure bitmap;
} VkExecStorageFailure;

struct VkExecPreparationFailure {
    VkExecPreparationOwner owner;
    VkExecPreparationReason reason;
    VkExecGraphicsResult result;
    int detail; // Original stage result, interpreted according to owner and reason.
    uint32_t role, slot; // UINT32_MAX means the stage or slot does not apply.
    RcInterval range, peer;
    const char *stage;
    uint64_t registration, material_serial;
    VkImage image;
    RcRegistryNativeImageFailure coverage;
    VkTextureSampleRefusal sampled_source;
    VkExecSampledFailure sampled;
    VkExecStorageFailure storage;
    struct {
        RcRegistryPosition prepared, actual, pool_read;
        const char *gate;
        VkPipeline pipeline;
        VkGraphicsPipelineRevision *revision;
        uint64_t pool_generation;
        uint32_t pool_index, pool_kinds, required_kinds, marker, scratch_merges;
        uint8_t prior_pipeline, prior_pool, prior_scratch, has_pool, finite_pool, has_scratch;
    } issue;
    struct {
        RcRegistryPosition position;
        uint32_t member;
        VkTextureSampleStatus status;
        uint8_t present;
    } sample;
    struct {
        RcRegistryPosition position, actual;
        uint64_t address, bytes, offset;
        VkBuffer buffer;
        const char *initial;
        VkExecHostInputResult initial_result;
        int selected_class;
        uint8_t present, ordered, import_used;
    } index;
    struct {
        RcRegistryPosition position;
        uint64_t address, bytes;
        uint32_t guest_format, native_format, width, height, samples, level, layer, layers;
        uint8_t present, seeded, native_checked, preserve;
    } load;
};

void vk_exec_graphics_storage_failure_log(const VkExecStorageFailure *);
void vk_exec_graphics_preparation_unavailable(VkExec *, uint32_t opcode, VkExecGraphicsResult,
    const VkExecPreparationFailure *, const char *reason);
void vk_exec_graphics_failure_set(VkExecGraphicsDraw *, const VkExecPreparationFailure *);
const VkExecPreparationFailure *vk_exec_graphics_failure(const VkExecGraphicsDraw *);

typedef struct VkExecStoragePhase VkExecStoragePhase;
typedef struct {
    const VkPipeInfo *info;
    int stage_override; // A transformed compute module binds the original task or mesh NVN slots.
    int retain_reads;
} VkExecStoragePhaseInput;
typedef struct {
    VkExecStoragePhase *previous;
    uint32_t previous_phase;
} VkExecStorageScope;
struct ResourceCommandEffects;

// One ordered operation owns all its selected placements. An absent phase has no pipeline layout.
VkExecGraphicsResult vk_exec_storage_phase_prepare(VkExec *, uint32_t opcode, const VkExecStoragePhaseInput *,
    uint32_t phases, int ordered, VkExecStoragePhase **);
void vk_exec_storage_phase_declare(const VkExecStoragePhase *, struct ResourceCommandEffects *);
// Scope the actual binder. Begin only after all fallible inputs and immediately before native use.
int vk_exec_storage_phase_enter(VkExec *, VkExecStoragePhase *, uint32_t phase, const VkPipeInfo *, VkExecStorageScope *);
void vk_exec_storage_phase_leave(VkExecStorageScope *);
int vk_exec_storage_phase_begin(VkExec *, VkExecStoragePhase *, uint32_t phase);
void vk_exec_storage_phase_recorded(VkExec *, VkExecStoragePhase *, uint32_t phase);
void vk_exec_storage_phase_cancel(VkExecStoragePhase **);

struct VkGpaPrefixRequest;
int vk_exec_dispatch_uniform_gpa_blocker(VkExec *, const VkExecDispatch *, struct VkGpaPrefixRequest *);
VkExecGraphicsResult vk_exec_dispatch_prepare_native_inputs(VkExec *ex, VkExecDispatch *dispatch,
    RcRegistryPosition attempted, struct VkExecGpaSampleScope **pending);
VkExecGraphicsResult vk_exec_dispatch_prepare_compat_inputs(VkExec *ex, VkExecDispatch *dispatch,
    RcRegistryPosition attempted, struct VkExecGpaSampleScope **pending);

typedef struct VkExecPoolGrant VkExecPoolGrant;
typedef struct {
    const VkPipeInfo *info;
    VkExecPoolGrant *grant;
    uint32_t opcode;
} VkExecPoolBindScope;
uint32_t vk_exec_pool_shader_kinds(const GfxShader *shader);
VkExecGraphicsResult vk_exec_pool_prepare_info(VkExec *ex, uint32_t required_kinds, VkExecPoolGrant **grant,
    int ordered, int allow_prefix, RcRegistryPosition attempted, struct VkExecGpaSampleScope **pending,
    VkExecGraphicsPass *pass);
int vk_exec_pool_grant_current(const VkExec *ex, const VkExecPoolGrant *grant);
// A bind captures native use for that recording part. The operation retains its grant until explicit release.
void vk_exec_pool_grant_release(VkExecPoolGrant **grant);
VkExecPoolBindScope vk_exec_pool_bind_enter(VkExecPoolBindScope scope);

// Workers receive a Ready manifest. An initial group-front probe never grows or records; the submitting
// thread resumes any retained manifest on the original draw if an ordered dependency is needed.
VkExecGraphicsResult vk_exec_graphics_draw_prepare(VkExec *ex, uint32_t opcode, const void *payload,
                                                   int ordered, RcRegistryPosition attempted,
                                                   struct VkExecGpaSampleScope **sample_pending, VkExecGraphicsDraw **draw);
int vk_exec_graphics_draw_current(const VkExec *ex, uint32_t opcode, const void *payload,
                                  VkExecGraphicsDraw *draw);
int vk_exec_graphics_draw_replan(VkExec *ex, uint32_t opcode, const void *payload, VkExecGraphicsDraw *draw);
VkExecGraphicsDraw *vk_exec_graphics_draw_enter(VkExecGraphicsDraw *draw);
void vk_exec_graphics_draw_cancel(VkExecGraphicsDraw **draw);
void vk_exec_graphics_draw_release_gpa_inputs(VkExecGraphicsDraw *draw);
void vk_exec_graphics_draw_release_gpa_inputs_retry(VkExecGraphicsDraw *draw);
VkSyncProgressResult vk_exec_graphics_progress_prepare(VkExec *ex, int (*prepare)(void *), void *context);
// Recorder only, outside pool/provider locks, before emitting the blocked draw. A nonzero scene names
// the older pool occupant. Zero requests proof of this scene's actual prefix use of the occupied sets.
VkSyncProgressResult vk_exec_graphics_pool_progress(VkExec *ex, uint64_t prior_scene);
void vk_exec_required_graphics_unavailable(VkExec *ex, uint32_t opcode, const char *reason);
typedef enum {
    VK_EXEC_RESTORE_READY, VK_EXEC_RESTORE_STOPPED, VK_EXEC_RESTORE_STALE, VK_EXEC_RESTORE_FAILED
} VkExecRestoreResult;
enum { VK_EXEC_ALIAS_READ_LIMIT = 8 };
VkExecRestoreResult vk_exec_prepare_storage_target_bound(VkExec *, VkTextureStorageTarget *, VkTexImage *);
VkExecRestoreResult vk_exec_prepare_alias_readback(VkExec *, VkTextureAliasRead *const *, uint32_t,
    int (*current)(void *), void *scope);
VkExecRestoreResult vk_exec_prepare_alias_texture_read(VkExec *, VkTextureAliasRead *const *, uint32_t,
                                                      int (*current)(void *), void *scope, const VkTexImage **);
VkExecRestoreResult vk_exec_prepare_texture_restore(VkExec *, VkTextureRestore *,
    int (*current)(void *), void *context, const VkTexImage **image, VkResult *failure);
VkSyncProgressResult vk_exec_texture_capacity(VkExec *, const RcVmaDependency *, VkResult,
    int (*spare_ready)(void *), void *context, uint64_t *observed_prior);
void vk_exec_required_graphics_unavailable(VkExec *ex, uint32_t opcode, const char *reason);
VkResult vk_exec_draw_graphics_scratch_prepare(VkExec *ex, const VkPipeState *state, const VkPipeInfo *info,
                                                uint32_t opcode, const void *payload, int allow_growth,
                                                VkExecGraphicsScratch **out,
                                                uint64_t *demand_bytes, RcVmaDependency *dependency, const VkGraphicsUniformFootprints *vertex_uniforms);
VkExecGraphicsScratch *vk_exec_draw_graphics_scratch_enter(VkExecGraphicsScratch *scratch);
void vk_exec_draw_graphics_scratch_finish(VkExecGraphicsScratch **scratch);
int vk_exec_draw_graphics_scratch_spare_ready(void *context);
uint64_t vk_exec_draw_graphics_scratch_prior_useful(VkExec *ex, uint64_t bytes, const RcVmaDependency *dependency,
                                                   const RcBudgetReadiness *now);

// The exclusive scene recorder prepares scratch before declaring and recording the operation. VK_NOT_READY
// leaves the operation untouched and reports its capacity dependency. The recorder owns the immutable payload
// through retries. Update commit and generated-index emission consume the exact grant.
typedef struct VkDrawScratchUpdate VkDrawScratchUpdate;
typedef struct {
    VkMemSpan span;
    VkMemSpanCapture *hold;
    RcInterval *writes;
    uint32_t count;
    int backed;
} VkExecUpdateDestination;
RcRegistryResult vk_exec_draw_update_select(const p_update_ub *, VkExecUpdateDestination *);
void vk_exec_draw_update_drop(VkExecUpdateDestination *);
typedef enum { VK_EXEC_UNIFORM_UPDATE_NONE, VK_EXEC_UNIFORM_UPDATE_READY, VK_EXEC_UNIFORM_UPDATE_STALE }
    VkExecUniformUpdateRead;
VkExecUniformUpdateRead upd_read_uniform(const VkExec *, uint64_t address, uint64_t bytes, void *out);
// A whole native descriptor may consume an inline write only if that write fully covers the selected
// backing and provider at this read position. The caller keeps the exact span through native binding.
VkExecUniformUpdateRead vk_exec_uniform_update_native(const VkExec *, uint64_t address, uint64_t size,
                                                      const VkMemSpan *, const VkMemSpanCapture *);

typedef struct {
    uint64_t offset, bytes;
    const void *data;
    RcRegistryPosition output;
} VkExecUniformUpdateFragment;
// Return ordered, nonoverlapping inline bytes from the actual retained native binding.
// Callers must independently prove the gaps from the original snapshot. Copy borrowed bytes before
// transferring to another owner or advancing the active read/history. Count is zero on failure.
RcRegistryResult vk_exec_uniform_update_fragments(const VkExec *, uint64_t address, uint64_t bytes,
    const VkMemSpan *, const VkMemSpanCapture *, RcRegistryPosition read,
    VkExecUniformUpdateFragment *, uint32_t capacity, uint32_t *count);

// Retain completed continuation storage only while the scene's inline-update journal references it.
// Move the allocation's ticket to the update journal only when it still borrows bytes from this block.
int vk_exec_draw_update_history_defer(void *commands, uint64_t bytes, RcBudgetTicket *budget);
void vk_exec_draw_update_history_end(uint64_t scene);
VkResult vk_exec_draw_update_prepare(VkExec *ex, const p_update_ub *op, VkExecUpdateDestination *,
                                     VkDrawScratchUpdate **out, RcVmaDependency *dependency);
void vk_exec_draw_update_cancel(VkExec *ex, VkDrawScratchUpdate *update);
void vk_exec_draw_update_commit(VkExec *ex, const p_update_ub *op, VkDrawScratchUpdate *update);
VkResult vk_exec_draw_index_prepare(VkExec *ex, const p_draw *op, VkDrawScratchUpdate **out,
                                   RcVmaDependency *dependency);
// Call after the final dispatch attempt. Any unpublished generated-index grant is canceled.
void vk_exec_draw_index_finish(VkExec *ex, VkDrawScratchUpdate *grant);

// Call only after native completion covers the sealed prefix of this scene. The usefulness queries guide
// the recorder's progress decisions. A prior scene is eligible for a fence wait only after its specific
// final native handoff succeeded.
void vk_exec_draw_update_prefix_completed(VkExec *ex, uint64_t scene);
int vk_exec_draw_scratch_prefix_useful(VkExec *ex, uint64_t bytes,
                                     const RcVmaDependency *dependency, const RcBudgetReadiness *readiness);
uint64_t vk_exec_draw_scratch_prior_useful(VkExec *ex, uint64_t bytes,
                                          const RcVmaDependency *dependency, const RcBudgetReadiness *readiness);
typedef struct {
    VkExec *ex;
    uint64_t bytes;
} VkDrawScratchWaitContext;
// The exclusive recorder retains the operation. This read-only readiness probe borrows the context for the call.
int vk_exec_draw_scratch_spare_ready(void *context);

// Queue preparation owns a compatible spare or a pinned request until it materializes, adopts or cancels it.
VkResult vk_exec_draw_update_credit_plan(uint64_t max_update_bytes, VkDrawScratchCredit **credit,
                                        RcAdmissionVector *peak, int *drain_older);
VkResult vk_exec_draw_update_credit_materialize(VkDrawScratchCredit *credit, RcBudgetTicket *joint);
void vk_exec_draw_update_credit_cancel(VkDrawScratchCredit *credit);
void vk_exec_draw_update_credit_adopt(VkExec *ex, VkDrawScratchCredit **credit);

// vk_exec_draw.c, before the chunks of a parallel group are recorded (submitting thread): the stand-ins are made and the
// texture pool set this submission took is brought up to date, which the chunks then only bind. Returns the inline
// uniform updates of the submission so far (vk_exec_upd_top).
uint32_t vk_exec_draw_group_begin(VkExec *ex, VkExecPoolView *pool);
void vk_exec_registration_gone(VkExec *ex, int logical);
void vk_exec_sample_source_refused(VkExec *ex, int logical);
void vk_exec_sampled_binding_refused(VkExec *, int logical, uint32_t opcode, const char *reason,
                                     const VkExecSampledFailure *);
void vk_exec_required_temporal_unavailable(VkExec *ex, const char *reason);
void vk_exec_cancel_temporal_preparation(void);
VkSyncProgressResult vk_exec_temporal_prepare(VkExec *ex, int (*prepare)(void *), void *context);
// Scheduling hint for a completed draw that used the bindless pool. It never authorizes shader,
// descriptor or resource reuse: the canonical leader still performs every normal resolution and validity check.
// Only chunk zero writes its group's record; the replay thread reads it after retirement and owns the cache.
typedef struct {
    const prog_t *program;
    uint64_t serial;
    int32_t stages;
    uint8_t vertex_format[16];
    int32_t color_format[8];
    uint8_t valid, direct_mesh;
} VkExecPoolRequirement;
VkExecPoolRequirement *vk_exec_pool_requirement_begin(void);
void vk_exec_draw_pool_learn(const VkExecPoolRequirement *requirement);
int vk_exec_draw_pool_leader_needed(VkExec *ex);
void vk_exec_draw_pool_schedule_init(void);
int vk_exec_draw_pool_requirement_known(VkExec *ex, int direct_mesh);
void vk_exec_draw_pool_leader_finished(VkExec *ex);
enum { VK_POOL_REFUSE_PRIOR, VK_POOL_REFUSE_ORDERED, VK_POOL_REFUSE_LIMIT, VK_POOL_REFUSE_INVALID,
       VK_POOL_REFUSE_FAMILY, VK_POOL_REFUSE_FORMAT, VK_POOL_REFUSE_DISABLED, VK_POOL_REFUSE_SPECIAL,
       VK_POOL_REFUSE_COUNT };
void vk_exec_draw_pool_refused(VkExec *ex, int reason);
void vk_exec_draw_pool_prefix_op(uint32_t opcode); // opt-in bounded diagnostic; replay thread only
VkExecPoolView *vk_exec_pool_view(void);   // the calling chunk's group, NULL outside parallel recording
VkExecPoolView *vk_exec_pool_local_view(void);   // owned only by the calling chunk; renewed after a later publication
uint64_t vk_exec_pool_generation(void);   // zero while pool state is being refreshed or publication is disabled
int vk_exec_pool_wait(uint64_t fence, uint32_t kinds, VkDescriptorSet *set);   // completed view or ordered first use
// vk_exec.c: of the submission's inline uniform updates (all), those a draw on the calling thread may read.
uint32_t vk_exec_upd_top(uint32_t all);
// vk_exec_xfer.c, parallel groups: vk_exec_xfer_group turns the front pass of the group in slot `group` on (it lists
// the group's occlusion segments) or off (-1); vk_exec_xfer_front replays a counter report or reset, render enable or
// render condition in the front pass (returns the condition's decision, 1 renders); vk_exec_xfer_chunk_state is the
// occlusion counting at a chunk's first op, which vk_exec_xfer_worker hands to the worker side with the chunk's
// decisions and its group's slot (on), or turns it off.
void vk_exec_xfer_group(int group);
int vk_exec_xfer_front(VkExec *fx, uint32_t op, const void *payload);
uint64_t vk_exec_xfer_chunk_state(void);
void vk_exec_xfer_worker(int on, const uint8_t *cond, uint32_t ncond, uint64_t state, uint32_t slot0, uint32_t nslots,
                         int chunk, int group);
// After every chunk of the group in slot `group` is done (submitting thread, groups oldest first): the slots the
// chunks closed go into their segments, in the order of the chunks.
void vk_exec_xfer_group_merge(int group, int nchunks);
// The front pass: whether a draw here counts samples (a query slot), and n slots of the ring in the order of the stream
// for one chunk (the first is returned; vk_exec_xfer_worker hands them to the chunk's worker).
int vk_exec_xfer_counting(void);
uint32_t vk_exec_xfer_slots(uint32_t n);
// vk_exec_draw.c: forget what the command buffer being recorded has bound (pipeline, sets, dynamic state, vertex
// buffers): the replay continues in a new one (vk_sync_split).
void vk_exec_draw_new_cb(void);
// Transfer a native span after successful record_input at the operation's original position.
// The actual command batch owns its allocation until completion or confirmed cancellation.
void vk_exec_record_native_input_provider(VkExec *, VkMemSpanCapture **);
VkMemMirrorInputPolicy vk_exec_mirror_input_policy(const VkExec *, int ordered);
VkSyncProgressResult vk_exec_read_device_pointer(VkExec *, uint64_t address, const VkMemSpan *, VkMemSpanCapture *,
                                                 const RcInterval *, uint32_t count, RcRegistryPosition read, uint64_t *value);
typedef struct VkExecPointerReadback VkExecPointerReadback;
VkResult vk_exec_pointer_phase_prepare(VkExecPointerReadback **);
void vk_exec_pointer_phase_cancel(VkExecPointerReadback **);
VkSyncProgressResult vk_exec_read_device_indices(VkExec *, uint64_t address, uint64_t bytes,
    const VkMemSpan *, VkMemSpanCapture *, const RcInterval *, uint32_t count, RcRegistryPosition,
    VkBuffer destination, VkDeviceSize offset, int *issued);
VkSyncProgressResult vk_exec_read_device_pointer_phase(VkExec *, VkExecPointerReadback **,
    const VkMemSpan *, VkMemSpanCapture *, const RcInterval *, uint32_t,
    const RcRegistryBufferPhaseExtent *, uint32_t, RcRegistryPosition, uint64_t *, VkExecGraphicsPass *);
typedef void (*VkExecNativeInputCopyRecord)(uint64_t fence, void *context);
enum { VK_EXEC_NATIVE_INPUT_COPY_LIMIT = 16, VK_EXEC_NATIVE_INPUT_WORD_LIMIT = 16 };
typedef struct {
    const VkMemSpan *span;
    VkMemSpanCapture *capture;
    const RcInterval *ranges;
    uint32_t range_count;
    VkBuffer snapshot;
    VkDeviceSize bytes;
    const VkBufferCopy *words;
    uint32_t word_count;
} VkExecNativeInputCopy;
VkSyncProgressResult vk_exec_ordered_host_input_prefix(VkExec *, RcRegistryPosition, VkBuffer,
                                                      VkDeviceSize offset, VkDeviceSize size);
VkSyncProgressResult vk_exec_copy_native_inputs(VkExec *, RcRegistryPosition,
    const VkExecNativeInputCopy *, uint32_t count, VkBuffer readback, VkExecNativeInputCopyRecord, void *context);

// A copied queue/chunk state becomes the inputs of a recording context. Invalidate its input certificates without
// discarding immutable recipes or extending the lifetime of submission-owned descriptors.
void vk_exec_draw_state_adopted(VkExec *ex);
// Immutable state blocks hold only assignments into queue state and source offsets into this command copy.
void vk_exec_draw_state_apply(VkExec *ex, const uint8_t *commands, const void *prepared);
NvnExecFn vk_exec_draw_prepared_handler(uint32_t opcode);
// Workers publish context-local preparation counters at chunk completion. The replay thread alone collects them
// into VkExecFrame, so logging/reset never races a worker updating those frame fields.
void vk_exec_draw_prepared_stats_flush(VkExec *ex);
void vk_exec_draw_prepared_stats_collect(void);
// vk_exec_draw.c, before a part of submission fence goes to the queue (vk_exec.c maybe_split): 1 when it may go. The
// texture pool set it binds is then sealed and the next draw that needs the pools takes another; 0 when every other set
// is bound by this submission's earlier parts, and the replay keeps recording into the same part.
int vk_exec_pool_split(uint64_t fence);
// vk_exec_draw.c: the vertex and fragment shader of the last draw recorded into this command buffer (NULL: none yet).
void vk_exec_draw_shaders(const GfxShader **vs, const GfxShader **fs);
// vk_exec_draw.c "pool marks": at the end of a submission that recorded work, after its closing barrier. Hands the
// mark copies the GPU completed to vk_texture.c and, every few such submissions, copies the marks out and clears them.
void vk_exec_pool_marks(VkExec *ex);
// vk_exec.c: the stand-in a 3D BC render target is drawn through goes with its image (vk_exec_image_gone).
void vk_exec_bc_gone(VkImage image);
// vk_hdr.c: with NX2NAVI_HDR=1, notes a draw into a presented window texture (vk_hdr_track: set by the first present).
extern int vk_hdr_track;
void vk_hdr_draw(VkExec *ex, const VkPipeInfo *info);
// Scissor 0 clipped to the render area of the open pass; extent 0: nothing to draw.
VkRect2D vk_exec_scissor(const VkExec *ex);
// vk_exec_draw.c: orders the open pass's earlier sampled reads of the attachments in written (bit i colour i, bit 8
// depth) before a draw or clear writes them; nothing when none of them was sampled since the last such barrier.
void vk_exec_feedback_war(VkExec *ex, uint32_t written);
void vk_exec_noatt_dims(const VkExec *ex, uint32_t *w, uint32_t *h);   // pass size when no attachment is set
// 1: every target of the current pass (any 1: at least one, and not only the depth target while a colour target is
// rendered for the first time) is one the game renders frame after frame: rendered in the last few presented frames,
// or steady (vk_target_frames.h tfr_pass_recent); 0 otherwise, or when the pass has none.
int vk_exec_targets_recent(VkExec *ex, int any);
extern uint64_t vk_exec_steady_passes;   // its answers of 1 that only a steady target gave (vk_target_frames.h)
// Looking ahead (vk_exec.c): called by a draw that found its shader or pipeline missing, before it waits or is skipped.
// The rest of the submission being replayed is read once and the stage libraries of the pipelines its later draws need
// are queued to the workers (vk_pipeline_hint). vk_exec_ahead_op (vk_exec_draw.c) applies one bind or draw op of that
// scan to sx, a copy of the replay state; vk_exec_ahead_begin starts a scan there.
void vk_exec_ahead(VkExec *ex);
void vk_exec_ahead_begin(void);
void vk_exec_ahead_op(VkExec *sx, uint32_t op, const void *payload);
extern uint64_t vk_exec_ahead_draws, vk_exec_ahead_lookups;   // draws the scans saw, and those whose shaders they looked up
// The current pass's targets with their formats, extents and the frames they were last rendered in (diagnostics).
void vk_exec_targets_describe(VkExec *ex, char *buf, size_t cap);

// NX2NAVI_TRACE=gfx: replay counters of the current frame, logged and reset at each present (vk_exec_frame_log).
typedef struct {
    uint64_t passes, draws_seen, draws, clears, copies, dispatches, submits;
    uint64_t fb_draws, fb_barriers, fb_restarts, fb_layouts;   // feedback loops: draws sampling a written target of
                                                               // their pass, in-pass barriers, pass restarts, layout
                                                               // transitions
    uint64_t fb_war_barriers;            // in-pass barriers before a write to a target an earlier draw sampled
    uint64_t ub_updates, ub_pass_breaks; // inline uniform updates, and those that ended an open pass
    uint64_t bind_calls, bind_ns;        // bind_resources: calls and the time spent resolving and pushing their sets
    uint64_t bind_tex_ns;                // of bind_ns, the texture descriptors' lookups
    uint64_t bind_buf_ns;                // of bind_ns, the uniform and storage buffer descriptors' resolution
    uint64_t bind_sig_ns;                // of bind_ns, the per-draw binding signature
    uint64_t bind_push_ns;               // of bind_ns, the set packing, pushes and binds after the binding loop
    uint64_t bind_reused, bind_resolved; // descriptors taken from the per-binding reuse (PB) and resolved again
    uint64_t prep_builds, prep_descriptors, prep_aggregate_skipped, prep_reused, prep_snapshot_reused;
    uint64_t prep_resolved, prep_ordered, prep_checked, prep_proof_descriptors;
    uint64_t prep_renewals, prep_renewed_descriptors;
    uint64_t prep_cache_hits, prep_cache_misses, prep_cache_evictions, prep_cache_probes;
    uint64_t dyn_calls, dyn_left;        // dynamic state calls recorded, and left out as unchanged (B.dyn)
} VkExecFrame;
extern VkExecFrame vk_exec_frame;
void vk_exec_frame_written(const tex_t *t);                    // vk_texture_gpu_written: t rendered or copied into
uint64_t vk_exec_recording_fence(void);                        // fence of the submission this thread records, else 0
void vk_exec_frame_log(const tex_t *presented, uint64_t frame);
// The draws skipped since the last call because a shader or pipeline was still building (vk_exec_draw.c), and whether
// a frame lacking them would be held back at its present (vk_backend.c frame_hold).
uint32_t vk_exec_pending_draws_take(void);
int vk_frame_hold_left(void);
// 1 while the texture pool makes the textures of an area load (vk_exec_draw.c pool_load_backlog): the frames drawn
// meanwhile show black stand-ins, and frame_hold leaves their presents out.
int vk_exec_pool_loading(void);

// NX2NAVI_DUMP=<frame> (vk_dump.c): between presents frame-1 and frame, every pass's targets and every dispatch's storage
// images are copied out and written as PNGs to NX2NAVI_DUMP_DIR (default "dump"), with passes.txt (targets, pipelines,
// textures, samplers) and images.txt (per image value statistics). The hooks run only when vk_dump_armed is set.
extern int vk_dump_armed;
void vk_dump_init(void);                 // reads the environment; vk_exec_record's first call
void vk_dump_pass_end(VkExec *ex);       // after vkCmdEndRendering
// after a draw's binds; mesh: a mesh draw (s->vs is its mesh stage, s->ts its task stage or NULL)
void vk_dump_draw(VkExec *ex, const VkPipeState *s, VkPipeline p, const VkPipeInfo *info, int mesh);
// Around a mesh draw of the dumped frame (after vk_dump_draw): a primitives-generated query, listed in mesh_prims.txt.
// begin returns the query, or -1 when none was begun; end takes that value.
int vk_dump_mesh_begin(VkExec *ex);
void vk_dump_mesh_end(VkExec *ex, int q);
void vk_dump_dispatch(VkExec *ex, const GfxShader *cs, VkPipeline p, const VkPipeInfo *info, const p_dispatch *d);
void vk_dump_copy(VkExec *ex, uint32_t op, const void *payload);   // before a copy or texture clear is replayed
void vk_dump_present(uint64_t frame);    // vk_be_present: waits for the dump frame's copies and closes the logs

// c0, the NVN driver constant bank of one stage (vk_exec_draw.c; NX2NAVI_FIX_C0=0 binds zeros instead): bank is
// VK_C0_BYTES, stage an NVN stage (0 vertex, 1 fragment, 5 compute), ssbo and ubo the 16 bound slots of that stage.
#define VK_C0_BYTES 0x800u
void vk_c0_fill(uint8_t bank[VK_C0_BYTES], int stage, const p_bind_buf ssbo[16], const p_bind_buf ubo[16],
                const uint32_t dispatch[3], uint32_t base_instance);

// Handler hooks. vk_exec.c defines both WEAK: they stop with rt_fatal naming the op.
// vk_exec_draw.c: OP_DRAW, OP_DRAW_IDX, OP_DISPATCH, OP_BIND_* (program, state, vattr, vstream, vbuf, ubo, ssbo, tex,
// samp, img), OP_UPDATE_UB, OP_DRAW_IDX_IND, OP_DISPATCH_IND. payload: nvn_op_payload of the record.
void vk_exec_op_draw(VkExec *ex, uint32_t op, const void *payload);
// The aspect a sampled view of img reads (vk_exec_draw.c get_view): stencil for a view in STENCIL depth-stencil mode.
VkImageAspectFlags vk_exec_view_aspect(const VkTexImage *img, const view_t *v, int hv);
VkImageView vk_exec_target_view(const VkTexImage *img, VkFormat format, VkImageViewType type, uint32_t aspect,
                                uint32_t level, uint32_t base, uint32_t layers, int blocks);   // kept for the image's life
// vk_exec_xfer.c: OP_COPY_B2B, OP_COPY_B2T, OP_COPY_T2B, OP_COPY_T2T, OP_CLEAR_TEX, OP_SIGNAL_EVENT, OP_REPORT_COUNTER,
// OP_RESET_COUNTER, OP_RENDER_ENABLE, OP_RENDER_COND, OP_CLEAR_BUF.
void vk_exec_op_xfer(VkExec *ex, uint32_t op, const void *payload);
// OP_WAIT_EVENT (vk_exec_xfer.c "event order check"): counts the wait; 1 when its condition holds in replay order (the
// value in memory, or the signals replayed so far), 0 when its signal has not been replayed yet.
int vk_exec_wait_event(const void *payload);
int vk_exec_event_met(const void *payload);   // the same test without counting
#define VK_EXEC_EVENTS_SEEN 64
typedef struct {
    struct { volatile uint32_t *val; uint32_t value; } slots[VK_EXEC_EVENTS_SEEN];
    int count, next;
} VkExecEventOrder;
void vk_exec_event_order_copy(VkExecEventOrder *order);
int vk_exec_event_order_met(VkExecEventOrder *order, const p_wait_event *wait, uint32_t memory);
uint32_t vk_exec_event_order_signal(VkExecEventOrder *order, const p_signal_event *signal, uint32_t memory);
int vk_exec_frozen_event(const volatile uint32_t *event, uint32_t *memory);
extern int vk_exec_signalled;                 // set by every replayed SignalEvent; vk_exec.c clears it
// Events the CPU signals (vk_exec_xfer.c "events the CPU signals").
void vk_exec_event_cpu(const void *event);    // any thread: nvnEventSignal is about to write the event's value
void vk_exec_event_sync(int begin);           // vk_exec_record: its mirror sync and uniform copy are taken now / it ends
int vk_exec_event_cpu_known(const void *event);     // the CPU has signalled this event before
int vk_exec_event_holds_now(const void *payload);   // an OP_WAIT_EVENT's condition on the value in memory now
// vk_backend.c submit_sync, before a submission is recorded: waits for the CPU's signal when the submission begins
// with a wait for it (vk_exec.c "the gate").
int vk_exec_gate_wait(const uint64_t *handles, int n);    // zero if a leading wait has not opened for capture
int vk_exec_event_at_sync(void);              // NX2NAVI_EVENT_AT_SYNC
uint64_t vk_exec_event_late_take(void);       // waits only a CPU signal after the copies would have passed, since the last call
// Since the last call, which resets them: waits, waits reached before their signal, those of them a later replayed
// signal met, and the sum of how many submissions later it came.
void vk_exec_event_stats(uint64_t *waits, uint64_t *unmet, uint64_t *late, uint64_t *late_subs);
extern uint64_t vk_exec_submit_seq;   // replay thread: submissions replayed so far
extern uint32_t vk_exec_brk;          // replay thread: bumped by every op that is not state or a draw (vk_exec.c op_linear)
uint32_t vk_exec_brk_now(void);       // vk_exec_brk, or in a parallel group's chunk the chunk's own count
int vk_exec_mesh_run(const void *cur, const void **out, int max);
void vk_exec_mesh_run_stops(char *out, size_t n);
// NX2NAVI_EVO_TRACE=1 (diagnostics, vk_exec.c "event order trace"): one token per step of the replay's event order.
void vk_exec_evt(char what, const void *event, uint32_t value);
void vk_exec_evt_frame(uint64_t frame);   // the game's present call: the frame's tokens are closed and logged
int vk_exec_held(void);               // atomic count of held submissions (event order)
int vk_exec_held_ready(void);         // serialized advisory query for the backend
// vk_dlaa.c: once DLSS has scaled the scene this frame, passes larger than the scene's light target vk_dlaa_swap_img
// sample DLSS's output vk_dlaa_swap_view in its place, so the game's scaling draw scales DLSS's picture.
extern const VkTexImage *vk_dlaa_swap_img;
extern VkImageView vk_dlaa_swap_view;
// Actual backend-owned output selected by swap_view; swap_img names the guest image being substituted.
extern VkImage vk_dlaa_swap_output;
// vk_aa.c, the anti-aliasing setting in place of the game's SMAA (vk_exec_draw.c aa_dispatch). vk_aa_mode: this frame's
// replacement, VK_AA_GAME when the game's SMAA runs as it is. vk_aa_role: which of the game's SMAA passes a compute
// program is, by the hash of its guest code. vk_aa_skip_ok: the edge and weight passes may be left out (the last
// composition was replaced). vk_aa_record: the replacement of the composition into cb, from src (the composite, sampled)
// into dst (the scene target's R32_UINT storage view) of w x h; 0 when it could not. vk_aa_composed: what became of the
// composition this frame.
enum { VK_AA_GAME, VK_AA_OFF, VK_AA_FXAA, VK_AA_SMAA };
enum { VK_AA_ROLE_NONE, VK_AA_ROLE_EDGES, VK_AA_ROLE_WEIGHTS, VK_AA_ROLE_COMPOSE };
int vk_aa_mode(void);
int vk_aa_role(uint64_t guest_hash);
int vk_aa_skip_ok(void);
int vk_aa_record(VkCommandBuffer cb, int mode, VkImageView src, VkImageView dst, uint32_t w, uint32_t h);
void vk_aa_composed(int ok);
// vk_contact.c: a pass about to begin (its targets resolved, no pass open), and a dispatch or copy about to be recorded
void vk_contact_pass_begin(VkExec *ex);
void vk_contact_flush(VkExec *ex);
int vk_contact_pending(void);   // a contact shadows' pass waits for the next transfer outside a pass
// vk_exec_xfer.c, samples-passed counting: a draw about to be recorded in the open pass (begins an occlusion query
// when counting and none is active), and the open pass about to end (ends the active query).
void vk_exec_oq_draw(VkExec *ex);
void vk_exec_oq_pass_end(VkExec *ex);

// Declaration runs in logical stream order, before parallel consumers. The opaque owner outlives retries; emitted
// effects transfer their obligations to actual completion/cancellation owners before this preparation ref is freed.
void *vk_exec_effects_declare(VkExec *, uint32_t opcode, const void *payload, RcRegistryPosition,
                              VkExecDispatch *compat_dispatch);
void *vk_exec_effects_enter(void *declaration);
void vk_exec_effects_leave(void *previous);
// Seal actual recording results only after execution/retry, never during group look-ahead.
void vk_exec_effects_recorded(void *declaration, int emitted);
void vk_exec_effects_release(void *declaration);
RcRegistryPosition vk_exec_resource_position(void);
void vk_exec_resource_handoff(uint64_t fence);
void vk_exec_resource_handoff_before(uint64_t fence, uint64_t keep_open_ordinal);

// Returns complete metadata coverage without resolving/uploading a Vulkan resource. Zero requires conservative authority.
typedef void (*VkExecEffectEmit)(void *, uint64_t, uint64_t, RcRegistryWriterKind);
int vk_exec_xfer_effects(VkExec *, uint32_t opcode, const void *payload, VkExecEffectEmit emit, void *context);
typedef struct VkExecBufferTransfer VkExecBufferTransfer;
typedef struct {
    uint64_t copies, clears, legs, bytes, partial, mirror_twin;
} VkBufferTransferProbeStats;
void vk_exec_buffer_transfer_probe_arm(int arm);
VkBufferTransferProbeStats vk_exec_buffer_transfer_probe_stats(void);
struct ResourceCommandEffects;
RcRegistryResult vk_exec_buffer_transfer_prepare(uint32_t opcode, const void *payload, RcRegistryPosition read,
                                                 VkExecBufferTransfer **);
RcRegistryResult vk_exec_buffer_transfer_declare(const VkExecBufferTransfer *, struct ResourceCommandEffects *,
                                                 RcRegistryPosition output);
void vk_exec_buffer_transfer_release(VkExecBufferTransfer *);
RcRegistryResult vk_exec_buffer_transfer_sources_prepare(VkExec *, VkExecBufferTransfer *);
// Consumes the selected plan on success or refusal.
void *vk_exec_effects_declare_buffer_transfer(VkExec *, uint32_t opcode, const void *payload, VkExecBufferTransfer *,
                                              RcRegistryPosition output);
VkExecBufferTransfer *vk_exec_effects_buffer_transfer(const VkExec *, uint32_t opcode, const void *payload);
typedef struct {
    VkTextureWriteTarget *target;
    VkImageSubresourceRange range;
    VkClearColorValue color;
    uint64_t texels;
} VkExecClearTexture;
// True means the ordered clear has either an exact target or a checked no-effect command.
int vk_exec_clear_capture(const p_clear_tex *, VkExecClearTexture *);
void vk_exec_clear_commit(VkExec *, VkExecClearTexture *);
typedef struct {
    VkTextureWriteTarget *target;
    VkMemSpan source;
    VkMemSpanCapture *source_hold;
    VkTextureAliasRead *source_alias;
    VkBuffer buffer;
    VkBufferImageCopy region;
    RcRegistryPosition read;
    uint64_t size;
    uint32_t offset_alignment;
    VkFormat format;
} VkExecBufferToTexture;
// Negative is a refused exact input, zero is compatibility, positive is a prepared copy or a checked no-op.
int vk_exec_buffer_to_texture_capture(VkExec *, const p_copy_b2t *, VkExecBufferToTexture *);
int vk_exec_clear_alias_candidate(const p_clear_tex *);
int vk_exec_texture_to_texture_alias_candidate(const p_copy_t2t *);
int vk_exec_buffer_to_texture_alias_candidate(const VkExec *, const p_copy_b2t *);
int vk_exec_buffer_to_texture_source_current(const VkExecBufferToTexture *);
RcRegistryResult vk_exec_buffer_to_texture_prepare_source(VkExec *, VkExecBufferToTexture *);
int vk_exec_buffer_to_texture_capture_compat(VkExec *, const p_copy_b2t *, VkExecBufferToTexture *);
void vk_exec_buffer_to_texture_commit(VkExec *, VkExecBufferToTexture *);
void vk_exec_buffer_to_texture_commit_compat(VkExec *, const p_copy_b2t *, VkExecBufferToTexture *);
void vk_exec_buffer_to_texture_cancel(VkExecBufferToTexture *);
// The caller holds the selected source across all T2T destination routes.
int vk_exec_transfer_image_input_capture(uint32_t, const void *, VkTextureTransferInput *, VkTextureAliasFailure *);
void vk_exec_texture_to_texture_compat(VkExec *, const p_copy_t2t *, const VkTextureTransferInput *);
typedef enum { VK_EXEC_IMAGE_COPY, VK_EXEC_IMAGE_RESOLVE, VK_EXEC_IMAGE_BLIT } VkExecImageTransfer;
typedef struct {
    VkTextureWriteTarget *target;
    VkTexturePresentSource *source_hold;
    VkTextureRestore *restore;
    const VkTextureTransferInput *input;
    VkTexImage source;
    VkImageSubresourceLayers source_layers, destination_layers;
    VkOffset3D source_offset, destination_offset;
    VkExtent3D source_extent, destination_extent;
    VkExecImageTransfer op;
    VkFilter filter;
    VkImageCreateInfo scratch_recipe;
    VkOffset3D scratch_offset;
    VkImage scratch;
    VmaAllocation scratch_memory;
} VkExecTextureToTexture;
int vk_exec_texture_to_texture_capture(VkExec *, const p_copy_t2t *, VkExecTextureToTexture *);
VkResult vk_exec_texture_to_texture_allocate(VkExecTextureToTexture *, RcVmaDependency *);
void vk_exec_texture_to_texture_commit(VkExec *, VkExecTextureToTexture *);
void vk_exec_texture_to_texture_cancel(VkExecTextureToTexture *);
typedef struct {
    VkTextureProjectedTransfer *target;
    uint32_t opcode;
    union {
        VkExecBufferToTexture buffer;
        VkExecTextureToTexture image;
    };
} VkExecProjectedTransfer;
int vk_exec_projected_transfer_capture(VkExec *, uint32_t opcode, const void *, VkExecProjectedTransfer *);
int vk_exec_projected_transfer_current(const VkExecProjectedTransfer *);
void vk_exec_projected_transfer_record(VkExec *, VkExecProjectedTransfer *);
void vk_exec_projected_transfer_cancel(VkExecProjectedTransfer *);
void *vk_exec_effects_declare_projected_transfer(VkExec *, const VkExecProjectedTransfer *, RcRegistryPosition output);
typedef struct VkExecTextureBufferWrite VkExecTextureBufferWrite;
typedef struct {
    VkTextureRestore *texture;
    VkTextureAliasRead *alias;
    VkTextureTransferInput input;
    const VkTexImage *image;
    struct MemBackingCapture *destination;
    uint64_t size;
    VkExecTextureBufferWrite *output;
} VkExecRestoreCopy;
int vk_exec_restore_copy_capture(VkExec *, const p_copy_b2t *, VkExecRestoreCopy *, VkTextureAliasFailure *);
int vk_exec_restore_copy_current(const p_copy_b2t *, const VkExecRestoreCopy *);
int vk_exec_restore_copy_prepare(VkExec *, const p_copy_b2t *, VkExecRestoreCopy *);
RcRegistryResult vk_exec_restore_copy_declare(const p_copy_b2t *, VkExecRestoreCopy *,
                                              struct ResourceCommandEffects *, RcRegistryPosition);
void vk_exec_restore_copy_commit(VkExec *, const p_copy_b2t *, VkExecRestoreCopy *);
void *vk_exec_effects_declare_texture_buffer(VkExec *, const p_copy_b2t *, VkExecRestoreCopy *, RcRegistryPosition);
void vk_exec_restore_copy_cancel(VkExecRestoreCopy *);
void *vk_exec_effects_declare_texture_write(VkExec *, uint32_t opcode, const VkTextureWriteTarget *, RcRegistryPosition);

int vk_exec_effects_read_allowed(uint64_t addr, uint64_t size); // reject same-operation writable feedback from ordinary snapshots
int vk_exec_effects_read_allowed_range(RcSourceRange, int private_sample);
int vk_exec_request_gpa_prefix(VkExec *, RcRegistryWrite, RcRegistryPosition producer, RcRegistryPosition read, RcInterval source);
// Scope only the pure sample query after sealing its prospective read. Leave before releasing preparation resources.
// When Leave reports a dependency, discard provisional grants before Settle. Settle adopts any new tail and consumes
// the scope; COMPLETE updates attempted and retries the same read. Cancel consumes a scope that needs no wait.
typedef struct VkExecGpaSampleScope VkExecGpaSampleScope;
VkExecGpaSampleScope *vk_exec_gpa_sample_enter(VkExec *, RcRegistryPosition attempted);
int vk_exec_gpa_sample_leave(VkExecGpaSampleScope *);
VkSyncProgressResult vk_exec_gpa_sample_settle(VkExecGpaSampleScope *, RcRegistryPosition *attempted);
void vk_exec_gpa_sample_cancel(VkExecGpaSampleScope *);
// Both exact and compatible selection can reach the actual native dispatch binder. A partly recorded helper cannot park.
int vk_exec_gpa_dispatch_binding(VkExec *, const GfxShader *, VkPipeline, const VkPipeInfo *,
                                 uint64_t indirect_address, const VkMemSpan *indirect, VkMemSpanCapture *argument,
                                 int repeatable, int selected);
int vk_exec_gpa_dispatch_retrying(const VkExec *);
int vk_exec_effects_unrecorded(const void *effect);
void vk_exec_draw_cancel_unrecorded(VkExec *);

// DLAA (vk_dlaa.c): NVIDIA DLSS at the scene's own size, behind NX2NAVI_DLAA (nx2navi.ini dlaa = 1).
#pragma once
#include "vk.h"
#include "vk_ffx.h"

// The narrowest scene the DLSS pass and the motion passes recognise by its D32F depth: Ultra Performance at a 1080p
// output renders 640 wide, smaller windows less.
#define DLAA_SCENE_MIN_W 320u

int vk_dlaa_wanted(void);   // the setting is on (the device may still lack DLSS)
// DLSS Frame Generation through NGX itself (vk_dlaa.c vk_ngxfg_*, gfx_present.c's presenter): NX2NAVI_FRAMEGEN = dlss
// where Streamline does not run (every build but the Windows one, where these all answer 0).
typedef struct { VkImage img; VkImageView view; VkFormat fmt; uint32_t w, h; } VkNgxImage;   // in GENERAL
int vk_ngxfg_wanted(void);   // this run can generate frames through NGX (DLSS upscaling on at the start)
int vk_ngxfg_max(void);      // generated frames per rendered one the GPU allows (1 = 2x, 5 = 6x), 0 when unavailable
int vk_ngxfg_on(void);       // ... and NX2NAVI_FRAMEGEN = dlss now (the menu switches it live)
void vk_ngxfg_reset(void);   // the next generation starts without history (a resize, the menu)
// Records count generated frames into out[] and a copy of bb into real; returns how many were made (0: none, and real
// is not written).
int vk_ngxfg_evaluate(VkCommandBuffer cb, const VkNgxImage *bb, const VkNgxImage *hudless, const VkNgxImage *alpha, const VkNgxImage *out, int count,
                      const VkNgxImage *real);
int vk_dlaa_fsr3(void);   // the setting is on and its upscaler is AMD FSR 3.1, which loads
int vk_dlaa_needs_restart(void);   // the menu's upscaler cannot start in this run
// Frame generation (vk_fg.cpp): obtains the inputs of the selected composition once. The owning receipt retains
// its native depth image/view and motion generation through the graphics composition submit.
int vk_dlaa_fg_inputs(VkFfxImage *depth, VkFfxImage *mv, float *jx, float *jy, int *reset);
// ... their views, and this frame's camera for DLSS Frame Generation (vk_dlssg.c): matrices for row vectors (v * M, the
// rows stored in order), as Streamline takes them; 0 without a camera that frame.
// znear and zfar: the view distances the depth buffer holds as 0 and 1 (1 and 0 when inverted), from the frame's own
// projection and the device's depth mode (vk_dlaa.c proj_planes); with an infinite far plane (infinite) zfar is
// VK_DLAA_FAR_INFINITE, a distance no surface reaches.
#define VK_DLAA_FAR_INFINITE 1e8f
typedef struct {
    float view_to_clip[16], clip_to_view[16], clip_to_prev_clip[16], prev_clip_to_clip[16];
    float pos[3], up[3], right[3], fwd[3];
    float fov_y, aspect;
    float znear, zfar;
    int infinite, inverted;
} VkDlaaCamera;
int vk_dlaa_fg_views(VkImageView *depth, VkImageView *mv);
// The reconstruction producer seals an exact frame before vk_dlaa_frame resets its replay state.
extern int vk_dlaa_hudless_wanted;
typedef struct VkDlaaHudSource VkDlaaHudSource;
typedef struct {
    VkNgxImage color, alpha;
    uint64_t frame, generation;
    int exact;
} VkDlaaHudImages;
void vk_dlaa_hud_frame(uint64_t frame);
VkDlaaHudSource *vk_dlaa_hud_take(uint64_t frame, const VkTexImage *selected);
// Requires actual native handoff of the recorded source writes. The returned views remain
// valid until the caller releases its source or transfers it to the consuming graphics fence.
const VkDlaaHudImages *vk_dlaa_hud_source_images(const VkDlaaHudSource *source, uint64_t frame);
void vk_dlaa_hud_source_release(VkDlaaHudSource *source);
void vk_dlaa_hud_shutdown(void);
int vk_dlaa_fg_camera(VkDlaaCamera *c);     // ... and its temporal upscaler is AMD FSR 3.1 (NX2NAVI_TAA = fsr3) instead of DLSS
int vk_dlaa_ffx_camera(VkFfxCamera *c);     // the same camera as FSR takes it (vk_ffx.h); 0 without one that frame
// vk_device.c: NGX's instance and device extensions join the lists (room for max names); vk_dlaa_init after the
// device exists brings NGX up and checks that DLSS is available.
void vk_dlaa_instance_exts(const char **ext, uint32_t *n, uint32_t max);
void vk_dlaa_device_exts(const VkExtensionProperties *have, uint32_t nhave, const char **ext, uint32_t *n, uint32_t max);
void vk_dlaa_init(void);
// vk_device.c destroy_all, once the device is idle and before the allocator and the device go (Linux): releases the NGX
// features and parameters, frees every image, buffer, pipeline and query pool DLAA made, resets its state for a later
// device and shuts NGX down for this one. Safe when NGX never came up, when nothing was made, and more than once.
// Windows: releases the temporal size and post groups; motion resources use the same idle proof on both platforms.
void vk_dlaa_release(void);
void vk_dlaa_motion_shutdown(void); // vk_device.c after successful device idle, before allocator teardown.
// 1 when dlaa is on and NGX came up with DLSS (after vk_dlaa_init): a DLSS mode's smaller scene is scaled up by DLSS.
int vk_dlaa_available(void);
// The video memory DLSS reports for its features (NGX's statistics), for the log; -1 while unknown.
int64_t vk_dlaa_vram(void);
// vk_exec.c vk_exec_pass_begin, once the pass's targets are resolved and before its barrier: arms the frame on the
// scene's light target or, at the first quarter-width pass after it, runs DLAA on that target. Each present ends the
// frame (vk_dlaa_frame).
typedef struct VkExec VkExec;
void vk_dlaa_pass(VkExec *ex);
int vk_dlaa_swap_matches(const VkTexImage *selected);
typedef struct VkExecGraphicsPassShape VkExecGraphicsPassShape;
typedef struct VkDlaaPreparedObserver VkDlaaPreparedObserver;
// Only an exact G-buffer or scene arm admits a receipt. Recording consumes a successful receipt once.
int vk_dlaa_prepare_observer(const VkExec *, const VkExecGraphicsPassShape *, VkTextureAliasRead *color_seed,
                            VkTextureAliasRead *depth_seed, VkDlaaPreparedObserver **);
int vk_dlaa_prepared_observer_current(const VkExec *, const VkExecGraphicsPassShape *, const VkDlaaPreparedObserver *);
int vk_dlaa_pass_prepared(VkExec *, VkDlaaPreparedObserver **);
void vk_dlaa_prepared_observer_cancel(VkDlaaPreparedObserver **);
enum {
    VK_DLAA_GRAPHICS_PASS_HOOK = 1 << 0,
    VK_DLAA_GRAPHICS_PASS_TONEMAP = 1 << 1,
    VK_DLAA_GRAPHICS_PASS_HUD = 1 << 2,
    VK_DLAA_GRAPHICS_PASS_OBSERVE = 1 << 3,
};
// Read-only classification of selected attachment identities before beginning rendering.
// Shader texture substitution still uses its actual captured sample provider.
uint32_t vk_dlaa_graphics_pass_work(const VkExec *, const VkExecGraphicsPassShape *);

int vk_dlaa_hud_attachment(const VkExec *ex, VkRenderingAttachmentInfo *attachment);
int vk_hud_alpha_pipeline(VkPipeState *s);
void vk_dlaa_hud_incomplete(void);
void vk_dlaa_hud_write(VkImage image);
void vk_dlaa_hud_draw_emitted(const VkExec *ex);
void vk_dlaa_hud_target_report(const VkExec *ex);
// The exact composition and tag were both unissued, after the previous slot retired.
void vk_dlaa_present_alpha_unrecorded(unsigned slot);
VkNgxImage vk_dlaa_present_alpha(VkCommandBuffer cb, VkExtent2D ext, const int rect[4], const int to[4],
                                 const VkDlaaHudImages *source, unsigned slot);
int vk_dlaa_swap_active(const VkExec *ex);
int vk_dlaa_tonemap_target(const VkExec *ex);
void vk_dlaa_tonemap_begin(VkExec *ex, VkRenderingAttachmentInfo *color, VkRenderingAttachmentInfo *depth,
                           VkRenderingInfo *ri);
void vk_dlaa_tonemap_end(VkExec *ex);
// The forward scene's compute upscaler. Returns 1 when temporal reconstruction filled its output.
struct VkDlaaSceneEffects;
int vk_dlaa_scene_upscale(VkExec *ex, uint32_t sw, uint32_t sh, const VkTexImage *dst,
                         struct VkDlaaSceneEffects *effects, const VkTextureColorBlitSource *output);
void vk_dlaa_frame(void);
int vk_dlaa_temporal_hook_needs_recorder(void);
void vk_dlaa_swap_use(VkExec *ex);
// Captures the latest evaluation only at its own present boundary. Composition
// selects it on its thread and moves a consumed use to its native fence slot.
typedef struct VkDlaaFgSizeUse VkDlaaFgSizeUse;
VkDlaaFgSizeUse *vk_dlaa_fg_capture(void);
// Pure description. 1 describes usable inputs, 0 means this frame lacks them, and -1 means its receipt was invalidated.
int vk_dlaa_fg_describe(const VkDlaaFgSizeUse *, VkFfxImage *depth, VkFfxImage *motion);
// The same pure check plus native views/camera for the already selected composition.
int vk_dlaa_fg_compose_peek(VkFfxImage *depth, VkFfxImage *motion, VkImageView *depth_view, VkImageView *motion_view,
                            VkDlaaCamera *camera, float *jx, float *jy, int *reset);
void vk_dlaa_fg_discard_current(void);
void vk_dlaa_fg_compose_select(VkDlaaFgSizeUse *use);
int vk_dlaa_fg_compose_consumed(const VkDlaaFgSizeUse *use);
void vk_dlaa_fg_release(VkDlaaFgSizeUse *use);
// After replay and composition retirement, before destroying the device allocator.
void vk_dlaa_size_shutdown(void);
// 1 once, at the present of the first frame whose scene has a new size after a setting changed (vk_dlaa.c).
int vk_dlaa_resized_take(void);
// vk_exec_draw.c, each draw before its fixed state is recorded: notes the scene camera block (vk_dlaa_draw), and moves
// the viewport of the scene's draws by the frame's jitter (vk_dlaa_viewport, when the viewport is recorded).
void vk_dlaa_draw(VkExec *ex);
void vk_dlaa_viewport(VkExec *ex, VkViewport *v);
void vk_dlaa_jitter_check(VkExec *ex);   // before each draw's fixed state: records the viewport again where its jitter changes
// vk_mvsurvey.c (NX2NAVI_MV_SURVEY): notes the uniform blocks of each scene draw, beside vk_dlaa_draw; each present
// compares them with the previous frames' and logs a summary now and then.
void vk_mvsurvey_draw(VkExec *ex);
void vk_mvsurvey_frame(void);
void vk_mvprev_draw(VkExec *ex);   // vk_mvprev.c: notes the scene draw's object and bone blocks
// Per-object motion (vk_exec_draw.c): the moving scene draws are listed again during the replay with
// their vertex shader's last-frame variant; vk_dlaa.c mv_pass records them over the camera motion into ex->cb (and
// leaves every binding of the replay to be set again; draws whose camera block is not cam are left out).
// vk_dlaa_prev_camera: last frame's camera block and this frame's jitter, 0 when there is no history.
uint64_t vk_dlaa_scene_frame(void);
typedef struct VkShieldResolve VkShieldResolve;
int vk_shield_resolve_prepare(VkExec *ex, uint64_t camera, VkImage depth, uint32_t w, uint32_t h, int reset,
                              VkShieldResolve **resolve);
int vk_shield_resolve_emit(VkExec *ex, VkShieldResolve **resolve, uint64_t camera, VkImage depth, VkImageView zv,
                           VkImageView mv, uint32_t w, uint32_t h, int reset);
int vk_mvobj_execute(VkExec *ex, uint64_t cam, VkImageView mv, VkImageView depth, VkFormat depth_fmt, uint32_t w,
                     uint32_t h);
int vk_dlaa_prev_camera(VkDescriptorBufferInfo *out, float jitter[2]);
// vk_mvcheck.c (NX2NAVI_MV_CHECK, test): mv_pass hands it the motion image after the camera pass and, after the object
// pass, the motion image again and the scene light target it ran on; it measures how well the vectors fit the picture.
void vk_mvcheck_camera(VkCommandBuffer cb, VkImage mv, uint32_t w, uint32_t h, float jx, float jy);
void vk_mvcheck_final(VkCommandBuffer cb, uint64_t fence, VkImage mv, VkImage cimg, VkFormat cfmt, uint32_t level,
                      uint32_t layer, uint32_t w, uint32_t h, float jx, float jy);
const char *vk_mvobj_frame_masks(void);   // vk_exec_draw.c: the last motion list, per position and changed banks
void vk_mvcheck_output(VkCommandBuffer cb, VkImage out, uint32_t ow, uint32_t oh);   // after DLSS on a check frame
// vk_exec_draw.c, once a draw's fragment shader is known: 1 skips the draw (the game's FXAA while NX2NAVI_FXAA is 0, its
// scaling draw when DLSS wrote the scaled picture).
int vk_dlaa_skip_draw(VkExec *ex, const GfxShader *fs);
int vk_dlaa_skip_draw_depth(VkImage depth, const GfxShader *fs);

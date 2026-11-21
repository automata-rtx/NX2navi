// Vulkan-internal header of the graphics layer. Host code only; the guest never sees it.
#pragma once
#include <vulkan/vulkan.h>
#include <stdint.h>
#include "resource_events.h"
#include "texture_source.h"
#include "resource_core_registry.h"
#include "resource_core_backing.h"
#include "rc_native_rmw_word.h"
#include "rc_native_image_store.h"
#include "resource_core_budget.h"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#include "vk_mem_alloc.h"
#pragma clang diagnostic pop
#include "nvn_obj.h"   // pool_t, tex_t, view_t, samp_par, NVNcopyRegion (memory and textures block)
#include "gfx.h"       // GfxShader, GfxShaderRes, GfxShaderAnswers (pipelines block)

// Required (device choice fails without them): Vulkan 1.3; VK_KHR_swapchain, VK_KHR_push_descriptor,
// VK_EXT_depth_clip_control, VK_EXT_external_memory_host; features timelineSemaphore, dynamicRendering,
// synchronization2, shaderStorageImageWriteWithoutFormat, depthClipControl, shaderInt64, samplerAnisotropy,
// independentBlend, geometryShader, textureCompressionBC.
// Optional (flag set in VkCtx when enabled): the has_* members.
typedef struct VkCtx {
    VkInstance instance;
    VkPhysicalDevice phys;
    VkDevice device;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mem_props;
    uint32_t gfx_family;                    // graphics + compute + presentation on win32
    uint32_t xfer_family;                   // a transfer-only family when the GPU has one, else gfx_family
    VkQueue gfx_queue;                      // graphics + present
    VkQueue xfer_queue;                     // equals gfx_queue when xfer_shared
    int xfer_shared;                        // 1: no separate queue; serialise with gfx_queue
    VmaAllocator vma;
    VkSemaphore timeline;                   // one timeline semaphore; values are issued by vk_sync.c
    VkDebugUtilsMessengerEXT messenger;
    volatile int32_t validation_errors;        // errors seen by the messenger (NX2NAVI_VK_VALIDATION=1)
    int validation;                         // layer active
    int has_index_type_uint8;              // actual enabled KHR or EXT feature
    int has_custom_border_color, has_eds, has_eds2, has_eds3, has_vertex_input_dynamic_state,
        has_conditional_rendering, has_transform_feedback, has_provoking_vertex, has_device_fault, has_memory_budget,
        has_robustness2, has_buffer_device_address, has_fragment_shader_interlock, has_buffer_marker,
        has_nv_checkpoints,                 // VK_NV_device_diagnostic_checkpoints (NX2NAVI_PASS_CRUMBS=1, vk_crumbs.c)
        has_astc_ldr,                       // textureCompressionASTC_LDR (enabled when reported)
        has_calibrated,                     // VK_KHR/EXT_calibrated_timestamps (GPU idle split in the perf line)
        has_hdr_metadata,                   // VK_EXT_hdr_metadata (vk_hdr.c vk_hdr_metadata)
        has_mesh_shader, has_vrs,           // VK_EXT_mesh_shader with taskShader and meshShader (NVN2 task/mesh stages)
        has_maximal_reconvergence,          // VK_KHR_shader_maximal_reconvergence (vk_pipeline.c spirv_reconverge)
        has_pipeline_binary,                // VK_KHR_pipeline_binary with VK_KHR_maintenance5 (vk_pipeline.c pipe_create)
        has_mesh_queries;                   // meshShaderQueries, kept while NX2NAVI_GPU_PASSES or NX2NAVI_DUMP is set
    int mesh_fallback;                      // task and mesh programs run as compute passes and are drawn from buffers
                                            // (mesh_fallback: auto without has_mesh_shader; vk_exec_draw.c "mesh fallback")
    // Feedback loops (maxwell.md 1.9, vk_exec_draw.c feedback_prepare): a draw samples a target of its open pass that
    // an earlier draw of the pass wrote. NX2NAVI_FEEDBACK=0: VK_FEEDBACK_OFF (sampled in GENERAL without a barrier, the
    // old behaviour); VK_FEEDBACK_LAYOUT with VK_EXT_attachment_feedback_loop_layout and dynamicRenderingLocalRead
    // (attachable images get feedback usage, a by-region barrier inside the pass); else (or NX2NAVI_FEEDBACK=restart)
    // VK_FEEDBACK_RESTART (the pass ends and restarts behind the full barrier).
    int feedback;                           // VK_FEEDBACK_*
    uint32_t min_imported_host_alignment;   // VK_EXT_external_memory_host
    VkQueue present_queue;                 // presentation only; gfx_queue on single-queue devices
    int device_index;                       // index in vkEnumeratePhysicalDevices
} VkCtx;
enum { VK_FEEDBACK_OFF, VK_FEEDBACK_RESTART, VK_FEEDBACK_LAYOUT };

// vk_caps.c: fallbacks for GPUs other than the RX 9070, decided at device creation.
typedef struct {
    int float64;                       // shaderFloat64 enabled; 0: shaders are translated with softfloat64
    uint32_t sbuf_align;               // nonzero: storage buffers bind at their address rounded down to this
    VkDeviceSize storage_write_alignment; // enabled robustBufferAccess2 descriptor-range rounding; 0 means no bounded-write proof
    int robust_image_access;          // device-default robustImageAccess or robustImageAccess2 enabled at creation
    uint8_t storage_image_format_support[VK_FORMAT_B10G11R11_UFLOAT_PACK32 + 1]; // device lifetime, indexed by VkFormat
    uint32_t sg_size;                  // requiredSubgroupSize on sg_pin_stages (32)
    VkShaderStageFlags sg_pin_stages;  // 0: no pin, stages run at the device's subgroupSize
    int gpl;                           // VK_EXT_graphics_pipeline_library enabled with fast linking (NX2NAVI_GPL=0/1; unset: off on AMD's driver for RDNA 2 and older)
    int amd;                           // an AMD GPU (vendor 0x1002), any driver
    int amd_sc;                        // AMD's Windows driver branch for RDNA 1 and 2 ("AMD proprietary shader compiler")
    int amd_llpc;                      // AMD's own driver for RDNA 3 and newer (its driverInfo names LLPC)
    int mesa;                          // a Mesa driver the game runs on: RADV, ANV or NVK
    uint32_t driver_id;                // the device's VkDriverId
} VkCaps;
extern VkCaps vk_caps;
void vk_caps_driver(VkPhysicalDevice p);             // first, before vkCreateDevice: vk_caps.amd, amd_sc, amd_llpc, mesa, driver_id
// The card and its driver in one phrase, for a fatal error's summary line ("" before vk_caps_init). Any thread.
const char *vk_caps_summary(void);
void vk_caps_features(VkPhysicalDeviceFeatures *f);   // before vkCreateDevice (NX2NAVI_VK_NO_FLOAT64 clears shaderFloat64)
// Before vkCreateDevice, with the driver's graphics pipeline library feature and properties: sets vk_caps.gpl,
// 1 = enable the extensions (VK_KHR_pipeline_library, VK_EXT_graphics_pipeline_library) and the feature.
int vk_caps_gpl(const VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT *f,
                const VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT *p);
void vk_caps_init(VkCtx *c);                          // after it: pretend limits, fallbacks, the boot report

VkCtx *vk_ctx(void);                        // NULL until vk_init() succeeded
int vk_init(void);                          // idempotent; 1 on success, 0 on failure (reason logged with rt_log)
void vk_shutdown(void);
int vk_selftest(void);                      // 0 on success
// Call with the result of every queue submit, present, acquire or wait. Any result but VK_ERROR_DEVICE_LOST returns at
// once, and the caller goes on with its own handling. A lost device ends the run here, in one place: the first thread
// to arrive sets vk_lost, logs the GPU breadcrumbs (vk_crumbs_report) and what VK_EXT_device_fault reports
// (description, faulting addresses with their precision, vendor codes), then ends the process as rt_fatal does (the
// log written, a save commit under way finished) with exit code RT_LOST_EXIT (dispatch.h). `where` names the call for
// the fatal line. Every later thread stops in this call and never returns to touch the device; the process ends
// within LOST_EXIT_MS (vk_device.c) even when the first thread hangs on a lock one of them holds.
void vk_device_lost_check(VkResult r, const char *where);
// 1 from the moment a thread saw the device lost. The paths that would touch the device (the replay and its submit,
// the uploads, the present, the completion thread's callbacks) read it where they start and return without doing so.
extern volatile int32_t vk_lost;
static inline int vk_device_is_lost(void) { return __atomic_load_n(&vk_lost, __ATOMIC_RELAXED); }
// Asks the device whether it is lost with a call that has a result and needs no lock (the status of a fence nothing
// signals), and ends the run through vk_device_lost_check when it is. The completion thread calls it before it runs
// callbacks: a lost device may report the submissions' timeline as reached (vk_sync.c completion_main).
void vk_device_lost_poll(void);

// ---- GPU breadcrumbs (vk_crumbs.c): with VK_AMD_buffer_marker, each crumb has the GPU write its code into host memory
// twice, when the command processor reaches it and when all work before it has finished. After a lost device the two
// codes bracket the work that hung. Without the extension every call returns at once.
enum { VK_CRUMB_SUBMIT = 1, VK_CRUMB_SUBMIT_END, VK_CRUMB_UPLOAD, VK_CRUMB_UPLOAD_END, VK_CRUMB_ASTC, VK_CRUMB_PRESENT,
       VK_CRUMB_PRESENT_END };
void vk_crumbs_init(VkCtx *c);       // after the allocator exists
void vk_crumbs_release(void);
// Records crumb kind with number n (a timeline value, a batch number; the low 28 bits are kept) into cb. Only the
// thread that records cb may call it: never into a replay part while the recorder thread owns it (vk_rec.h).
void vk_crumb(VkCommandBuffer cb, int kind, uint64_t n);
void vk_crumbs_report(void);         // logs the two codes and what they name

// ---- pass marks (vk_crumbs.c): the replay (vk_exec.c vk_exec_pass_begin / vk_exec_pass_end, vk_exec_draw.c
// op_dispatch) marks where each render pass and dispatch starts and where it ends, so that a device loss names the
// pass the GPU was in. A mark is a number; a ring of VK_PASS_MARKS records keeps what it was. With
// VK_AMD_buffer_marker the GPU writes the number of the last mark it started (TOP_OF_PIPE, before vkCmdBeginRendering
// or the dispatch) and of the last it finished (BOTTOM_OF_PIPE, after vkCmdEndRendering or the dispatch) into the
// breadcrumb buffer; with VK_NV_device_diagnostic_checkpoints the same two points are checkpoints, read with
// vkGetQueueCheckpointData(2)NV after the loss. NX2NAVI_PASS_CRUMBS: unset, AMD buffer marks only; 1, also NVIDIA
// checkpoints (the extension is enabled at device creation); 0, none.
#define VK_PASS_MARKS 4096   // power of two
enum { VK_MARK_PASS = 1, VK_MARK_DISPATCH };
typedef struct {
    uint32_t id;                        // the mark this record belongs to (0: never written)
    uint8_t kind;                       // VK_MARK_*
    uint8_t ncolor;                     // colour attachments
    uint32_t step;                      // index of the pass, or of the dispatch, within its submission, from 0
    uint32_t w, h, d;                   // render area and layers; a dispatch: group counts (0 0 0: indirect)
    uint32_t draws;                     // draws recorded in the pass; UINT32_MAX until its end is recorded
    uint64_t fence;                     // the submission: the timeline value it signals
    VkFormat color[8], depth;           // attachment formats (VK_FORMAT_UNDEFINED: none)
    const GfxShader *first[2], *last[2];// vertex and fragment shader of the first and last draw; a dispatch: first[0]
    uint32_t gp;                        // NX2NAVI_GPU_PASSES: the timestamp pair of this mark plus one; 0 none
    uint32_t layer;                     // NX2NAVI_GPU_PASSES: the first layer of the depth target (a shadow cascade)
} VkPassMark;
void vk_gpu_passes_frame(uint64_t frame);   // vk_crumbs.c, each present: NX2NAVI_GPU_PASSES, GPU time per pass and dispatch
// NX2NAVI_GPU_PASSES also times what a part of a submission spends outside its passes and dispatches (vk_crumbs.c, "the
// gap ledger"). vk_gpu_passes_part: vk_sync.c, at the start (end 0) and the end (end 1) of each part's command buffer.
// vk_gpu_gap_note: the replay is about to record commands of this kind outside a pass or dispatch; units is their size
// (bytes for buffers, texels for images, 0 unknown). Both do nothing outside the timed frames. Replay thread only.
enum {
    VK_GAP_COPY_B2B, VK_GAP_COPY_B2T, VK_GAP_COPY_T2B, VK_GAP_COPY_T2T, VK_GAP_CLEAR_BUF, VK_GAP_CLEAR_TEX,
    VK_GAP_CLEAR_TARGET, VK_GAP_UPSCALER, VK_GAP_WB_MERGE, VK_GAP_WB_COPY, VK_GAP_TWIN, VK_GAP_POOL_MARKS,
    VK_GAP_MESH_RING, VK_GAP_BC_BACK, VK_GAP_WB_BEFORE, VK_GAP_WB_AFTER, VK_GAP_N
};
void vk_gpu_passes_part(VkCommandBuffer cb, int end);
void vk_gpu_gap_note(int kind, uint64_t units);
void vk_gpu_gap_boundary(VkCommandBuffer cb);
int vk_gpu_passes_capturing(void);
uint64_t vk_gpu_passes_capture_id(void);
void vk_gpu_passes_stop_capture(void);
void vk_exec_mesh_counts_begin(uint64_t capture);
void vk_exec_mesh_counts_end(uint64_t capture);
uint32_t vk_exec_mesh_counts(uint64_t capture, const uint32_t **out, uint32_t *missing);
// Pass marks wanted and possible (NX2NAVI_PASS_CRUMBS and the extensions): set by vk_crumbs_init.
extern int vk_pass_marks_on;
// vk_device.c create_device, before vkCreateDevice: 1 when NX2NAVI_PASS_CRUMBS asks for NVIDIA checkpoints.
int vk_pass_marks_want_nv(void);
// The next mark: records its start into cb (through the recorder thread when it owns cb) and returns its ring record,
// id, kind and fence set, for the caller to fill. NULL when pass marks are off. Replay thread only.
VkPassMark *vk_pass_mark_begin(VkCommandBuffer cb, int kind, uint64_t fence);
void vk_pass_mark_end(VkCommandBuffer cb, const VkPassMark *m);   // records the end of mark m into cb
// The recorder thread's side of both (vk_rec.c): the marker and checkpoint commands themselves.
void vk_pass_mark_cmd(VkCommandBuffer cb, uint32_t id, int end);
// AMD only: the last mark the GPU started and the last it finished (0: none); 0 without buffer marks.
int vk_pass_marks_read(uint32_t *started, uint32_t *finished);
const VkPassMark *vk_pass_mark_record(uint32_t id);   // the ring record of mark id; NULL once a later mark took its slot
// The report's text for a started and a finished mark, three lines without a trailing newline: which marks (source:
// the extension they came from), then each with its ring record. vk_pass_marks_report logs it; the probe checks it.
void vk_pass_marks_format(const char *source, uint32_t started, uint32_t finished, char *o, size_t size);
void vk_pass_marks_report(void);    // vk_device.c device_fault_report: the AMD words and the NVIDIA checkpoints, formatted

// ---- window and present (gfx_present.c). Host threads only.
// vk_present_image: blits the crop rectangle {x, y, w, h} of src (NULL, or w/h <= 0: the whole w x h image), scaled
// to the swapchain extent, into the acquired swapchain image and presents it. src is in layout, which is
// VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL (its writes made visible to the transfer stage by the caller's barrier) or
// VK_IMAGE_LAYOUT_GENERAL (vk_texture.c images: the present makes earlier writes visible and orders later writes after
// the blit). All its writes must be submitted to gfx_queue before the call (queue order does the rest). Starts the
// window on first use. A minimised window or a lost swapchain skips the frame. Calls serialise on an internal lock.
void vk_present_image(VkImage src, VkImageLayout layout, VkFormat fmt, int w, int h, const int crop[4]);
// vk_present_image split in two: vk_present_blit submits the blit and leaves the present waiting, vk_present_flip
// queues it. The next blit or a swapchain rebuild queues a waiting present first.
typedef struct VkTexturePresentSource VkTexturePresentSource;
typedef struct VkDlaaFgSizeUse VkDlaaFgSizeUse;
typedef enum {
    VK_PRESENT_SOURCE_UNSENT,
    VK_PRESENT_SOURCE_GFX_SUBMITTED,
    VK_PRESENT_SOURCE_UNCERTAIN,
    VK_PRESENT_SOURCE_PENDING,
} VkPresentSourceHandoff;
typedef struct {
    RcAdmissionVector peak;
    uint64_t older_config;
    VkFence older_graphics;
    int reprepare;
} VkPresentConfigRetry;
typedef struct VkDlaaHudSource VkDlaaHudSource;
// Submitted or uncertain graphics work takes the source and any reconstruction input it consumed.
VkPresentSourceHandoff vk_present_blit(VkImage src, VkImageLayout layout, VkFormat fmt, int w, int h,
                                      const int crop[4], uint64_t frame, VkTexturePresentSource **source,
                                      VkDlaaFgSizeUse **temporal, VkDlaaHudSource **hud,
                                      VkPresentConfigRetry *retry);
VkResult vk_present_config_drain(const VkPresentConfigRetry *retry);
void vk_present_retire_temporal_inputs(void);
// FSR 1 on the present (vk_fsr.c): records EASU (+ RCAS) of src's crop rectangle {x, y, w, h} into the rectangle out
// {x, y, w, h} of dst, a swapchain image in TRANSFER_DST; src is in GENERAL and fmt its format. 1 = EASU enlarged it,
// 2 = a bilinear blit shrank it and RCAS sharpened it (the picture is smaller than the crop), 0 = nothing recorded (the
// setting says bilinear, the sizes are equal, a shrink without sharpening, or FSR is not available): the caller blits.
int vk_fsr_record(VkCommandBuffer cb, unsigned slot, VkImage src, VkFormat fmt, int sw, int sh, const int crop[4],
                  VkImage dst, const int out[4]);
void vk_fsr_release(void);   // before the device goes away
// Colour adjustment on the present (vk_color.c): with NX2NAVI_COLOR_* away from their defaults, records a pass that writes
// src's crop rectangle {x, y, w, h}, adjusted, into an RGBA16F image of that size in GENERAL and returns it (the caller
// scales that one instead, crop {0, 0, w, h}); VK_NULL_HANDLE = nothing recorded. src is in GENERAL, fmt its format.
VkImage vk_color_record(VkCommandBuffer cb, unsigned slot, VkImage src, VkFormat fmt, const int crop[4]);
void vk_color_release(void);   // before the device goes away
// The brightness, contrast, saturation and gamma values (as the settings write them) of a colour profile, NULL for
// custom or an unknown name (vk_color.c).
const char *const *vk_color_preset(const char *name);
// HDR output (vk_hdr.c). vk_hdr_armed: the HDR path is set up for this run (decided at the start). vk_hdr_wanted:
// NX2NAVI_HDR as the menu has it now. vk_hdr_surface_format: the HDR10 surface format when wanted and offered (1). vk_hdr_set_active: gfx_present.c says whether the swapchain is HDR10 (vk_hdr_active). vk_hdr_encode:
// like vk_color_record, src's crop as PQ code values (RGBA16F, crop {0, 0, w, h}). vk_hdr_ui_target / vk_hdr_ui_compose:
// the in-game menu's RGBA8 target, then blended onto the swapchain image. vk_hdr_to_srgb8: 10-bit PQ pixels to sRGB bytes.
int vk_hdr_armed(void);
int vk_hdr_wanted(void);
int vk_hdr_needs_restart(void);   // the menu's HDR row: the value set takes effect at the next start
void vk_hdr_off(void);   // probes that read back 8-bit swapchain pixels: SDR whatever the setting
int vk_hdr_surface_format(const VkSurfaceFormatKHR *f, uint32_t n, VkSurfaceFormatKHR *out);
void vk_hdr_metadata(VkSwapchainKHR sc, int new_sc);
void vk_hdr_set_active(int on);
int vk_hdr_active(void);
VkImage vk_hdr_encode(VkCommandBuffer cb, unsigned slot, VkImage src, VkFormat fmt, const int crop[4]);
int vk_hdr_native_picture(VkFormat f);   // vk_hdr_native, and f is the format of the game's PQ picture
VkImageView vk_hdr_ui_target(VkCommandBuffer cb, uint32_t w, uint32_t h);
void vk_hdr_ui_compose(VkCommandBuffer cb, VkImageView dst_view, VkFormat fmt, uint32_t w, uint32_t h);
void vk_hdr_to_srgb8(const uint8_t *in, uint8_t *out, size_t px, int a2r);
// Luminance statistics of an HDR10 shot inside rect {x, y, w, h}: the darkest pixel, the 0.1st, 1st, 50th and 99th
// percentiles and the brightest pixel in nits, and the share of pixels under 0.01 nits (vk_hdr.c).
void vk_hdr_luminance_stats(const uint8_t *in, int stride, const int rect[4], int a2r, float out[7]);
void vk_hdr_release(void);   // before the device goes away
// The support buffer's HDR settings, a vec4 at 1440 (vk_exec_draw.c support_fill; read by the game's tonemap with
// shaders/hdr_tonemap.frag spliced in, vk_hdr.c tonemap_splice).
void vk_hdr_support(float out[4]);
// vk_contact.c: screen-space contact shadows on the game's composed scene colour (NX2NAVI_CONTACT_SHADOWS)
int vk_contact_strength(void);   // percent, 0 = off
void vk_contact_release(void);   // before the device goes away
int vk_present_flip(void); // true when a host present succeeded; asynchronous presenters report separately
// Every submission to ctx->gfx_queue (and xfer_queue when shared) takes this lock; vk_present_image and the swapchain
// rebuild do. Recursive. Taking it first hands the replay's held submissions to the queue (vk_sync_flush_held); the
// raw forms leave them held, for vk_sync.c itself.
void vk_queue_lock(void);
void vk_queue_unlock(void);
void vk_present_queue_lock(void);
void vk_present_queue_unlock(void);
VkResult vk_device_wait_idle(void);
void vk_queue_lock_raw(void);
int vk_queue_trylock_raw(void);   // 1 when acquired
void vk_sync_flush_held(void);    // queue lock held: the held submissions go to the queue in one call (vk_sync.c)
// Copy of the last presented swapchain image, for probes: on, the next presents read the image back into a host buffer
// and wait for the GPU. *px stays valid until the next present; the format is the swapchain's. Returns 0 with no frame.
// The rectangle {x, y, w, h} of a dw x dh swapchain image a cw x ch picture is presented in (gfx_present.c,
// NX2NAVI_ASPECT); the rest is black.
void gfx_fit_rect(int cw, int ch, int dw, int dh, int out[4]);
void vk_present_capture(int on);
int vk_present_captured(const uint8_t **px, int *w, int *h, VkFormat *fmt);

// The window system's part of device creation (gfx_window_win.c with Win32, gfx_window_sdl.c with SDL): the instance
// extensions a surface needs (their count, at most max), and whether a queue family of pd can present to it.
int gfx_window_vk_extensions(const char **out, int max);
int gfx_window_vk_presentable(VkInstance inst, VkPhysicalDevice pd, uint32_t family);
void gfx_window_resize(int client_w, int client_h);   // programmatic resize; returns after the swapchain is rebuilt
void gfx_vsync_bind_counter(volatile int32_t *counter);  // host-side stand-in for a bound event: incremented per tick
unsigned gfx_window_focus_messages(void);          // OE_MSG_FOCUS_STATE_CHANGED pushes made by the window thread
void gfx_window_release_vk(void);                  // vk_shutdown calls it first: destroys swapchain, surface, present objects
long vk_validation_errors_total(void);             // validation errors since the first vk_init, across vk_shutdown

// ---- memory and textures (vk_memory.c, vk_texture.c)
// Guest or host threads; every call needs vk_init(). vk_backend.c wires these into gfx_vulkan_backend.
int  vk_pool_init(pool_t *p);              // CPU-visible: guest storage imported (VK_EXT_external_memory_host) as one
                                           // VkBuffer; physical: device-local memory; virtual: a canonical address map
int  vk_pool_init_setup(pool_t *p, const pool_t *setup); // guest initialization preserves p until native setup succeeds
void vk_pool_fini(pool_t *p);              // drops images cut from the pool; waits for the GPU if it used the pool
int  vk_pool_map_virtual(pool_t *v, int n, const void *requests);   // nvn_mapping_t[n], checked by the front end
void vk_texture_register(int id, const tex_t *t, const view_t *v);  // texture descriptor slot (copied)
void vk_texture_register_in_pool(uint64_t pool, int id, const tex_t *t, const view_t *v);
void vk_texture_pool_release(uint64_t pool);
int vk_texture_register_in_pool_ordered(uint64_t pool, int id, const tex_t *t, const view_t *v, uint64_t queued_publication);
int vk_texture_pool_release_ordered(uint64_t pool, uint64_t queued_publication);
int vk_texture_finalize_registration_ordered(const tex_t *t, uint64_t queued_publication);
void vk_texture_release_replayed(const tex_t *t);
void vk_sampler_register(int id, const samp_par *s);                // sampler descriptor slot (copied)
void vk_texture_release(const tex_t *t);   // TextureFinalize: the image stays for slots that still name its storage
typedef enum {
    VK_TEXTURE_CPU_READ_OUTSIDE, VK_TEXTURE_CPU_READ_READY, VK_TEXTURE_CPU_READ_STOPPED, VK_TEXTURE_CPU_READ_REFUSED,
} VkTextureCpuReadResult;
int vk_texture_cpu_read_candidate(const tex_t *, const view_t *);
VkTextureCpuReadResult vk_texture_cpu_read_texels(const tex_t *, const view_t *, const NVNcopyRegion *,
    uint8_t *dst, uint64_t row, uint64_t img, int replay_ordered, uint64_t *selected_fence);
void vk_read_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row, uint64_t img);
void vk_write_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, const uint8_t *src, uint64_t row, uint64_t img);
void vk_write_packaged_texture(const tex_t *t, const void *src);
void vk_guest_written(uint64_t addr, uint64_t size);   // a host write write-watch may miss: textures there re-upload

// GPU address -> backing (section 3.1). avail: bytes from offset that stay contiguous in buffer. alloc/alloc_offset:
// the device-local memory behind physical and virtual pool addresses, NULL for imported pools.
// cls: the pool class of the backing (VK_MCLS_*), for the traffic census.
typedef struct {
    VkBuffer buffer;
    VkDeviceSize offset, avail;
    VmaAllocation alloc;
    VkDeviceSize alloc_offset;
    int cls;
    void *pool;   // the imported pool record behind the span (NULL for device memory), for VK_MEM_BOUND
} VkMemSpan;
// Traffic census (vk_memory.c vk_mem_census): bytes the replay binds or copies, by pool class and use. Imported pools by
// their CPU access flag; DEVICE is physical memory (directly or through a virtual pool); SCRATCH the replay's transient
// copies. CPU_READ counts bytes host code reads from pool storage in place of the GPU (misaligned binds, byte indices,
// counter words); GPU_WRITE what copies, fills, counter reports and uniform updates write.
enum { VK_MCLS_NONE, VK_MCLS_HOST_NOACC, VK_MCLS_HOST_UNC, VK_MCLS_HOST_CACHED, VK_MCLS_DEVICE, VK_MCLS_SCRATCH, VK_MCLS_N };
enum { VK_MUSE_VB, VK_MUSE_IB, VK_MUSE_UBO, VK_MUSE_SSBO, VK_MUSE_IND, VK_MUSE_COPY_SRC, VK_MUSE_GPU_WRITE,
       VK_MUSE_CPU_READ, VK_MUSE_N };
void vk_mem_count(int cls, int use, uint64_t bytes);
// NX2NAVI_TRACE=perf dirty census (vk_memory.c): the replay notes each imported pool it binds as a vertex, index, uniform,
// storage or indirect buffer; at the start of every submit the pages the CPU wrote in such pools since the previous
// submit are counted from the shared page generations and logged as `mem: dirty` every 60 frames. Nothing when tracing
// is off: VK_MEM_BOUND is one flag test at the call site.
void vk_mem_bound_note(void *pool);

// The vkGetDeviceProcAddr the runtime fetches device entry points with: the loader's, or Streamline's interposer when
// DLSS Frame Generation is up (vk_device.c sets it right after the device is created). Functions fetched the other way
// bypass the plugins' hooks.
extern PFN_vkGetDeviceProcAddr vk_dpa;

// Every timed submission part's GPU duration summed, never reset: a reader differences it across frames (dynres.c).
uint64_t vk_gpu_busy_all(void);
#define VK_MEM_BOUND(sp) do { if ((rt_trace & RT_TRACE_PERF) && (sp).pool) vk_mem_bound_note((sp).pool); } while (0)
void vk_mem_dirty_submit(void);                   // vk_sync_submit, under its lock, before the replay
// VRAM mirror of buffer-bound pools (vk_memory.c). vk_mirror_on: NX2NAVI_MEMORY=mirror|verify, or the
// device class is discrete and NX2NAVI_MEMORY is unset or auto; set at device selection. The replay (submitting thread
// under vk_sync_submit's lock) calls:
// - VK_MEM_USE at a vertex, index, uniform, storage or indirect binding of a span vk_mem_resolve returned: the pool's
//   mirror is created on first use and sp.buffer becomes it (same offsets); a pool that stays imported keeps its buffer.
// - VK_MEM_PICK where a copy, fill or update reads or writes a span: the mirror if the pool has one, never a new one.
// - vk_mem_gpu_wrote after recording a GPU write of [sp.offset, sp.offset + size) into sp.buffer: a mirror's bytes go
//   back to guest memory at the end of the submission. No-op for imported buffers.
// - vk_mem_mirror_begin at the start of a submission's recording, vk_mem_mirror_end (records the write-backs; returns
//   the commands recorded) after its last barrier.
// vk_sync.c records the copies the mirrors' first accesses queued (vk_mem_mirror_pending) into a prologue command
// buffer with vk_mem_mirror_prologue and submits it right before each part.
extern int vk_mirror_on;
void vk_mem_use(VkMemSpan *sp);
void vk_mem_pick(VkMemSpan *sp);
VkBuffer vk_mem_guest_twin(const VkMemSpan *sp);   // a VRAM mirror span: the pool's imported buffer (guest memory), same offset; else none
int vk_mem_cpu_visible_span(const VkMemSpan *sp, uint64_t addr, uint64_t size); // exact captured buffer writes directly into guest storage
#define VK_MEM_USE(sp) do { if (vk_mirror_on && (sp).pool) vk_mem_use(&(sp)); } while (0)
#define VK_MEM_PICK(sp) do { if (vk_mirror_on && (sp).pool) vk_mem_pick(&(sp)); } while (0)
void vk_mem_gpu_wrote(const VkMemSpan *sp, uint64_t size);
// A shader's storage-buffer write: goes back through the write-back merge (vk_memory.c), which keeps the words the GPU
// did not change.
// Publish proved stores in the retained provider without widening or optional writeback policy.
void vk_mem_gpu_wrote_exact(const VkMemSpan *sp, uint64_t size);
void vk_mem_gpu_wrote_shader(const VkMemSpan *sp, uint64_t size);
// Uniform snapshots are owned by the submission until recording consumes them. Replay never rebuilds a queued copy.
typedef struct VkUsnapCapture VkUsnapCapture;
typedef struct {
    const uint64_t *handles;
    int nhandles;
    const p_bind_buf *inherited;
    size_t ninherited;
} VkUsnapInputs;
typedef struct { p_bind_buf slots[8 * 18]; } VkUsnapBindingState;
int vk_usnap_project(const uint64_t *handles, int n, VkUsnapBindingState *candidate);
typedef struct VkDrawScratchCredit VkDrawScratchCredit;
typedef struct {
    VkUsnapCapture *snapshot;
    VkDrawScratchCredit *scratch;
} VkReplayCaptureBundle;
enum { VK_REPLAY_WORKER_LIMIT = 16 };
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
// Replay metadata holds both RAM and commit until its owner frees the allocation.
VkResult vk_replay_metadata_allocate(size_t bytes, int clear, RcBudgetProducer producer,
                                     RcBudgetTicket *ticket, void **allocation);
void vk_replay_metadata_free(void *allocation, RcBudgetTicket *ticket);
int vk_replay_metadata_peak_add(RcAdmissionVector *peak, uint64_t bytes);
size_t vk_exec_draw_update_credit_host_bytes(const VkDrawScratchCredit *credit);
// The queue owns both outputs after success. A refusal leaves the bundle empty and reports retry evidence.
VkResult vk_usnap_queue_bundle(const VkUsnapInputs *inputs, uint64_t max_update_bytes,
                              VkReplayCaptureBundle *out, VkReplayCaptureWait *wait);
void vk_usnap_queue_bundle_cancel(VkReplayCaptureBundle *bundle);
VkUsnapCapture *vk_usnap_take(const VkUsnapInputs *inputs);
VkResult vk_usnap_preflight(const VkUsnapInputs *inputs, VkUsnapCapture **capture, VkReplayCaptureWait *wait);
void vk_usnap_wait_producer(uint64_t producer_epoch, uint32_t milliseconds);
VkResult vk_usnap_begin(VkUsnapCapture **capture, int n, uint64_t fence);
int vk_sync_stopping(void);
int vk_sync_try_provider_idle(void);
void vk_sync_end_provider_idle(void);
void vk_sync_stop_submissions(void);
uint64_t vk_sync_present_complete(void);
void vk_usnap_cancel(VkUsnapCapture *capture);
uint64_t vk_usnap_reclaim_reusable(void); // shared pressure adapter: count destroyed, never credited capacity
void vk_usnap_end(uint64_t fence);
int vk_usnap_find(uint64_t addr, uint64_t n, uint64_t align, VkDescriptorBufferInfo *out);
typedef struct VkMemSpanCapture VkMemSpanCapture;
// A resolved range of one immutable capture. A region index is only meaningful while capture matches; no region or
// arena pointer is retained. GPU writes invalidate just their overlapping regions at their position in the stream.
typedef struct {
    uint64_t capture, addr, size, align;
    uint32_t region, reserved;
} VkUsnapLease;
typedef enum {
    VK_USNAP_RESOLVE_READY, VK_USNAP_RESOLVE_NATIVE_ONLY, VK_USNAP_RESOLVE_ORDERED_PRODUCER,
    VK_USNAP_RESOLVE_SOURCE_ADVANCED, VK_USNAP_RESOLVE_UNAVAILABLE, VK_USNAP_RESOLVE_NO_CAPACITY,
    VK_USNAP_RESOLVE_MAPPING_STALE, VK_USNAP_RESOLVE_ALIGNMENT
} VkUsnapResolveResult;
// Only READY supplies a descriptor. Producer evidence requires a separate proof of native input coverage.
VkUsnapResolveResult vk_usnap_resolve_at(uint64_t addr, uint64_t n, uint64_t align, RcRegistryPosition,
                                         VkDescriptorBufferInfo *, VkUsnapLease *);
struct VkGpaPrefixRequest;
// Only an earlier GPA projection that blocks this exact retained uniform capture can supply a dependency.
int vk_usnap_gpa_blocker(uint64_t address, uint64_t bytes, uint64_t align, RcRegistryPosition read,
                        struct VkGpaPrefixRequest *);
int vk_usnap_find_lease(uint64_t addr, uint64_t n, uint64_t align, VkDescriptorBufferInfo *out, VkUsnapLease *lease);
int vk_usnap_lease_valid(const VkUsnapLease *lease);
int vk_usnap_find_lease_at(uint64_t addr, uint64_t n, uint64_t align, RcRegistryPosition position,
                           VkDescriptorBufferInfo *out, VkUsnapLease *lease);
int vk_usnap_lease_valid_at(const VkUsnapLease *lease, RcRegistryPosition position);
uint64_t vk_usnap_capture(void);   // active capture identity, 0 without a capture
typedef struct { uint64_t address, size; RcRegistryPosition output; } VkMemBufferInlineProof;
// Borrow the active snapshot producer observation to identify exact latest native-buffer fragments. The caller
// retains the selected native span; this does not create or transfer another registry capture.
RcRegistryResult vk_usnap_buffer_fragments(const VkMemSpanCapture *capture, const VkMemSpan *selected,
    uint64_t address, uint64_t size, RcRegistryPosition read, const VkMemBufferInlineProof *inline_proofs,
    uint32_t inline_count, RcRegistryBufferFragment *out, uint32_t capacity, uint32_t *fragment_count);
// The prepared binding adapter retains the legacy aggregate-proof layout. Canonical snapshot leases validate
// individually at the consuming position, so the snapshot provider declines this aggregate shortcut.
#define VK_USNAP_PROOF_BUCKETS 64u
typedef struct {
    uint64_t capture, serial, epoch, horizon, buckets;
    uint64_t version[VK_USNAP_PROOF_BUCKETS];
    uint32_t building, sealed;
} VkUsnapProof;
void vk_usnap_proof_begin(VkUsnapProof *proof);
int vk_usnap_proof_add(VkUsnapProof *proof, const VkUsnapLease *lease);
int vk_usnap_proof_end(VkUsnapProof *proof);
// Finish into an exclusively owned destination. On failure its previous contents cannot be reused.
int vk_usnap_proof_end_into(VkUsnapProof *building, VkUsnapProof *destination);
int vk_usnap_proof_valid(VkUsnapProof *proof);
int vk_usnap_proof_inherit(VkUsnapProof *building, const VkUsnapProof *prior);
const uint8_t *vk_usnap_map(void);   // a host copy of the current arena, at the offsets vk_usnap_find gives
const uint8_t *vk_usnap_bytes(uint64_t addr, uint64_t n);   // the copy of [addr, addr+n) as of the submit, NULL: none
void vk_usnap_drop(uint64_t addr, uint64_t n);   // notify legacy descriptor caches of a recorded GPU write
uint64_t vk_usnap_gen(void);                      // legacy descriptor-cache generation
// [addr, addr+n) into dst: 1 from this submission's valid retained copy, 2 from a fresh protected import read at
// the current operation, 0 if neither can be proven (dst untouched).
int vk_usnap_read(uint64_t addr, uint64_t n, void *dst);
int vk_usnap_equal_tail(uint64_t addr, uint64_t size, uint64_t skip, const void *expected);
// Why vk_usnap_find finds nothing for [addr, addr+n) in this submission (diagnostics).
enum { VK_USNAP_FOUND, VK_USNAP_NO_SNAPSHOT, VK_USNAP_DROPPED, VK_USNAP_IN_FLIGHT, VK_USNAP_NOT_CPU, VK_USNAP_NOT_BOUND };
int vk_usnap_why(uint64_t addr, uint64_t n);
void vk_usnap_shutdown(void);                     // after recording and completion owners have discharged, before VMA teardown
// vk_mvprev.c: last frame's copy of the scene's object (c7) and bone (c5) blocks, for per-object motion vectors.
// vk_mvprev_step: the distance of the game's other copy of a vertex-stage block the snapshot also copies, 0: none.
// vk_mvprev_find: last frame's block of that object (prev, its address; how, MVPREV_*), 0: none.
enum { MVPREV_NONE, MVPREV_SELF, MVPREV_SIDE, MVPREV_NEARER };
uint64_t vk_mvprev_step(int stage, int index, uint64_t size);
int vk_mvprev_on(void);
void vk_mvprev_want(uint32_t index_mask);   // also keep these binding indices' last-frame copies (bit n: index n)   // DLAA on and NX2NAVI_DLAA_OBJMV not 0, as of the last present
int vk_mvprev_find(int index, uint64_t addr, uint64_t size, uint64_t *prev, int *how);
void vk_mvprev_frame(void);   // each present, beside vk_dlaa_frame
void vk_mem_mirror_begin(void);
int vk_mem_mirror_end(VkCommandBuffer cb, uint64_t fence);
int vk_mem_mirror_pending(void);
int vk_mem_mirror_pressure_waiting(void);
void vk_mem_mirror_service_pressure(void); // outside submission and provider locks
void vk_mem_mirror_prologue(VkCommandBuffer cb, uint64_t scene);
void vk_mem_lazy_part_end(void);   // lazy write-back: a part ends (before its prologue), and it is handed over
void vk_mem_lazy_handed(void);
void vk_mem_lazy_decide(void);     // once the driver is known (after vk_caps_driver): on or off, with one log line
void vk_mem_fatal_note(char *out, size_t n);   // the memory mode and the lazy write-back's state, for a fatal error's line
// Shadow storage (vk_memory.c): vk_sync.c calls vk_mem_shadow_sync right before each submit; the replay notes every GPU
// write into CPU-visible memory at record time (vk_mem_gpu_pending) and reports it at the fence with vk_gpu_written,
// which copies the bytes back into guest memory before telling the texture cache (vk_guest_written).
// fence: the value of the submission the part belongs to (0: none, a probe's own sync), at which what shaders stored
// through the device address table goes back to guest memory (NX2NAVI_SHADOW_TABLE_STORES).
void vk_mem_shadow_sync(uint64_t fence);
// Record time: a shader of the guest submission being recorded reads the device address table (vk_gpa.c). The pools
// such a shader reaches are never resolved, so the next vk_mem_shadow_sync brings the CPU's writes into every shadow
// the table names, once per guest submission.
void vk_mem_shadow_table_read(void);
int vk_mem_shadow_storage(void);   // CPU-visible pools are shadow copies of guest memory, not imports (NX2NAVI_SHADOW)
int vk_mem_shadow_table_stores(void);   // shadow storage keeps what shaders store through the table (NX2NAVI_SHADOW_TABLE_STORES)
void vk_mem_gpu_pending(uint64_t addr, uint64_t size);
void vk_mem_gpu_pending_span(const VkMemSpan *, uint64_t address, uint64_t size);
void vk_gpu_written(uint64_t addr, uint64_t size);
// The same for a GPU write into a VRAM mirror, which went back to guest memory with the submission: the texture checks
// see it, the mirror's next sync does not copy it back up. vk_mem_mirrored at record time, right after the write's
// buffer was picked (VK_MEM_USE / VK_MEM_PICK): 1 when [addr, addr+size) lies in one pool whose mirror is on.
void vk_gpu_written_back(uint64_t addr, uint64_t size);
int vk_mem_mirrored(uint64_t addr, uint64_t size);
int vk_mem_flush_add_probe(void);        // --probe-vkexec: the flush range list; 0 or the number of the failed check
uint64_t vk_mem_mirror_uploaded(void);   // --probe-vkexec: bytes copied into mirrors since the last summary line
int vk_mem_mirror_probe(uint64_t addr);   // --probe-vkexec: the pool at addr gets its mirror as on a first buffer use;
                                          // 1 when it has one
void vk_mem_class_log(const VkPhysicalDeviceProperties *p, const VkPhysicalDeviceMemoryProperties *m);   // once
int vk_mem_class_of(uint64_t addr);               // VK_MCLS_* of the pool at addr (NONE when unbacked)
void vk_mem_census(uint64_t frame);               // at each present: pool census and traffic, logged at frame 1800 and,
                                                  // with NX2NAVI_TRACE=perf, every 1800 frames
void vk_mem_lost_report(void);                    // a line with the video memory in use and its budget (device lost)
float vk_mem_device_used_mb(void);                // device-local memory this process holds (the driver's budget
                                                  // figure), MB; -1 without VK_EXT_memory_budget. Any thread.
void vk_mem_stage_log(const char *stage);         // one line after a boot stage: device memory, process commit
// VRAM in use and its budget, summed over the DEVICE_LOCAL heaps. exact 1: VK_EXT_memory_budget (the whole process),
// 0: this allocator's own allocations against the heap sizes. 0 when unknown.
int vk_mem_vram(uint64_t *used, uint64_t *budget, int *exact);
// VRAM headroom (vk_headroom.c): the VMA callbacks that see new device memory, and the thread that allocates the next
// block of each memory type in use ahead of the render thread.
const VmaDeviceMemoryCallbacks *vk_headroom_callbacks(void);
void vk_headroom_start(void);
void vk_headroom_stop(void);   // before vmaDestroyAllocator
uint64_t vk_headroom_held(void);        // bytes the spare blocks' anchors hold
uint64_t vk_headroom_release(void);     // a refused allocation: the anchors go, none for 30 s; returns their bytes
// A refused allocation: the staging and snapshot buffers kept for reuse and the headroom's spare blocks go back to the
// driver before the caller tries once more (vk_memory.c). Returns the bytes.
uint64_t vk_mem_give_back(void);
void vk_mem_ballast(VmaAllocator vma);   // NX2NAVI_VRAM_BALLAST_MB (test): video memory taken for the session (vk_memory.c)
void vk_mem_ballast_release(VmaAllocator vma);
VkDeviceSize vk_vma_block_size(void);   // VMA's block size on heaps over 1 GiB (NX2NAVI_VMA_BLOCK_MB)
int vk_mem_resolve(uint64_t addr, uint64_t size, VkMemSpan *out);   // 1 when all of [addr, addr+size) is backed
// vk_mem_resolve for a caller that reads guest memory itself or only asks what backs it: with shadow storage the pool
// is not copied in for it (vk_memory.c, mem_resolve).
int vk_mem_lookup(uint64_t addr, uint64_t size, VkMemSpan *out);
// Pin the exact selected native argument, and check the narrow logical range before using it again.
VkMemSpanCapture *vk_mem_span_capture(uint64_t address, uint64_t size, const VkMemSpan *selected);
struct MemBackingCapturePlan;
typedef struct {
    VkMemSpan selected;
    uint64_t publication;
    size_t bytes;
    int nested_host;
} VkMemSpanCapturePlan;
// Uniform snapshots reserve both receipts before retaining either of them.
RcRegistryResult vk_mem_span_capture_plan(const VkMemSpan *, const struct MemBackingCapturePlan *,
                                           VkMemSpanCapturePlan *);
RcRegistryResult vk_mem_span_capture_bounded(const VkMemSpanCapturePlan *, const struct MemBackingCapturePlan *,
                                              VkMemSpanCapture **);
// Exact already-established native import or installed mirror. No optional mirror placement during planning.
typedef struct {
    VkMemSpan imported, selected;
    uint64_t publication, mirror_incarnation, import_scene;
    uint64_t transition_mirror;
    size_t bytes;
    int cold_writer;
} VkMemImportInputPlan;
enum { VK_MEM_IMPORT_WRITER_NONE, VK_MEM_IMPORT_WRITER_COLD, VK_MEM_IMPORT_WRITER_ACTIVE };
RcRegistryResult vk_mem_span_import_input_plan(const VkMemSpan *, const struct MemBackingCapturePlan *,
                                                VkMemImportInputPlan *);
int vk_mem_span_import_writer_candidate(uint64_t address, const VkMemSpan *);
RcRegistryResult vk_mem_span_import_writer_plan(const VkMemSpan *, const struct MemBackingCapturePlan *,
                                                VkMemImportInputPlan *);
RcRegistryResult vk_mem_span_import_input_capture(const VkMemImportInputPlan *, const struct MemBackingCapturePlan *,
                                                   VkMemSpanCapture **);
uint32_t vk_mem_span_import_input_ranges(const VkMemSpanCapture *, const RcInterval **);
RcRegistryResult vk_mem_span_import_writers_publish(VkMemSpanCapture *const *, uint32_t count, RcRegistryPosition read);
void vk_mem_span_import_writer_issued(VkMemSpanCapture *, uint64_t fence);
int vk_mem_span_capture_current(const VkMemSpanCapture *, const VkMemSpan *selected);
// The caller already owns this capture. Another owner can retain the same mapping/provider across a defer.
VkMemSpanCapture *vk_mem_span_capture_retain(VkMemSpanCapture *);
void vk_mem_span_capture_release(VkMemSpanCapture **);
// Select an exact established native input. Optional mirror creation stays in compatibility.
RcRegistryResult vk_mem_span_native_input(uint64_t address, uint64_t size, int clamp, VkMemSpan *,
                                          VkMemSpanCapture **);
typedef enum {
    VK_MEM_MIRROR_EXISTING,    // no new optional decision during speculative preparation
    VK_MEM_MIRROR_IMPORT,      // keep the current import for the recording scene
    VK_MEM_MIRROR_PREPARE,     // actual ordered empty native part with prior work complete
} VkMemMirrorInputPolicy;
RcRegistryResult vk_mem_span_prepare_native_input(uint64_t address, uint64_t size, int clamp, VkMemMirrorInputPolicy,
                                                  VkMemSpan *, VkMemSpanCapture **);
// Retain the actual buffer already selected by a compatibility binder before parking its operation.
RcRegistryResult vk_mem_span_capture_bound_input(uint64_t address, uint64_t size, const VkMemSpan *selected,
                                                  VkMemSpanCapture **);
int vk_mem_span_record_input(VkMemSpanCapture *, const VkMemSpan *);
int vk_mem_span_held_argument_current(const VkMemSpanCapture *,uint64_t address,uint64_t size,const VkMemSpan *);
// Settle the held representation at its original read before another binding can choose a mirror.
RcRegistryResult vk_mem_span_prepare_held_argument(const VkMemSpanCapture *, const VkMemSpan *, int ordered);
int vk_mem_span_record_held_argument(VkMemSpanCapture *,uint64_t address,uint64_t size,const VkMemSpan *);
// Copy a subrange of the prepared provider's mapping. No current address rediscovery.
uint32_t vk_mem_span_input_intervals(const VkMemSpanCapture *, uint64_t address, uint64_t size, RcInterval **);
// Inspect the retained device mapping without claiming a raw-buffer write.
uint32_t vk_mem_span_capture_device_intervals(const VkMemSpanCapture *, const VkMemSpan *, uint64_t address,
                                              uint64_t size, RcInterval **);
int vk_mem_span_input_cpu_visible(const VkMemSpanCapture *);
// Value identity only, from this exact original allocation rather than a new VA selection.
// Verify the held physical provider for an exact positioned buffer result. Supply a complete independent
// initial fill only when its ordered origin is intact; otherwise the registry needs earlier valid results.
int vk_mem_span_buffer_preserve_current(void *, RcRegistryBufferSpan, RcRegistryPosition, int *initialized);
int vk_mem_span_buffer_location(const VkMemSpanCapture *, const VkMemSpan *, RcRegistryBufferLocation *);
// Reflect existing fixed raw initialization boundaries in writable effects without narrowing the MayWrite hull.
enum { VK_MEM_FIXED_WRITER_PARTS = 68 };
uint32_t vk_mem_span_buffer_fixed_partitions(const VkMemSpanCapture *, const VkMemSpan *,
                                             const RcInterval *, uint32_t count, RcInterval *, uint32_t capacity);
typedef struct VkMemNativeRmwWord VkMemNativeRmwWord;
struct RcVmaDependency;
VkResult vk_mem_native_rmw_word_prepare(VkMemSpanCapture *, const VkMemSpan *, uint64_t descriptor_size,
    uint64_t word_address, RcRegistryPosition, struct RcVmaDependency *, VkMemNativeRmwWord **);
int vk_mem_native_rmw_word_current(const VkMemNativeRmwWord *, const VkMemSpan *, uint64_t word_address,
    RcRegistryPosition);
VkDescriptorBufferInfo vk_mem_native_rmw_word_marker(const VkMemNativeRmwWord *);
RcRegistryResult vk_mem_native_rmw_word_declare(VkMemNativeRmwWord *, RcRegistryPosition output);
int vk_mem_native_rmw_word_bind(VkMemNativeRmwWord *, VkCommandBuffer);
int vk_mem_native_rmw_word_order(VkMemNativeRmwWord *, VkCommandBuffer);
void vk_mem_native_rmw_word_issued(VkMemNativeRmwWord **, VkCommandBuffer);
void vk_mem_native_rmw_word_cancel(VkMemNativeRmwWord **);
typedef struct VkMemNativeRmwBitmap VkMemNativeRmwBitmap;
typedef struct {
    const char *gate;
    RcRegistryResult registry;
    RcInterval range;
    VkResult result; // original VkResult from prepare, otherwise zero
    RcRegistryPosition blocker_read;
    uint64_t mirror_offset, blocker_offset, blocker_bytes;
} VkMemNativeRmwBitmapFailure;
// A report is optional. A successful call clears it. Failure carries only evidence from that decision.
VkResult vk_mem_native_rmw_bitmap_prepare_report(VkMemSpanCapture *, const VkMemSpan *, uint64_t descriptor_size,
    VkPipelineStageFlags2 writer_stage, RcRegistryPosition, struct RcVmaDependency *, VkMemNativeRmwBitmap **,
    VkMemNativeRmwBitmapFailure *);
VkResult vk_mem_native_rmw_bitmap_prepare(VkMemSpanCapture *, const VkMemSpan *, uint64_t descriptor_size,
    VkPipelineStageFlags2 writer_stage, RcRegistryPosition, struct RcVmaDependency *, VkMemNativeRmwBitmap **);
int vk_mem_native_rmw_bitmap_current_report(const VkMemNativeRmwBitmap *, const VkMemSpan *, RcRegistryPosition,
    VkMemNativeRmwBitmapFailure *);
int vk_mem_native_rmw_bitmap_current(const VkMemNativeRmwBitmap *, const VkMemSpan *, RcRegistryPosition);
int vk_mem_native_rmw_bitmap_order(VkMemNativeRmwBitmap *, VkCommandBuffer);
VkDescriptorBufferInfo vk_mem_native_rmw_bitmap_marker(const VkMemNativeRmwBitmap *);
int vk_mem_native_rmw_bitmap_bind(VkMemNativeRmwBitmap *, VkCommandBuffer);
RcRegistryResult vk_mem_native_rmw_bitmap_declare(VkMemNativeRmwBitmap *, RcRegistryPosition);
void vk_mem_native_rmw_bitmap_issued(VkMemNativeRmwBitmap **, VkCommandBuffer);
void vk_mem_native_rmw_bitmap_cancel(VkMemNativeRmwBitmap **);
// Value provenance of the mirror actually selected by the native write. The recording owns the span.
int vk_mem_span_mirror_access_location(const VkMemSpan *, RcRegistryBufferLocation *);
// An installed and already synchronized mirror with its real imported twin. This never refreshes it.
int vk_mem_span_gpa_donor(const VkMemSpanCapture *, const VkMemSpan *, RcRegistryBufferLocation *, VkBuffer *twin);
// The caller keeps the captured provider alive and proves/reuses the same span when recording the transfer.
RcRegistryResult vk_mem_span_buffer_fragments(const VkMemSpanCapture *, const VkMemSpan *,
    uint64_t address, uint64_t size, RcRegistryGpuCapture, RcRegistryPosition,
    const VkMemBufferInlineProof *inline_proofs, uint32_t inline_count,
    RcRegistryBufferFragment *, uint32_t capacity, uint32_t *fragment_count);
typedef struct { void *pool; uint64_t mirror_incarnation; } VkMemInputProvider;
VkMemInputProvider vk_mem_span_take_input_provider(VkMemSpanCapture **);
void vk_mem_input_provider_release(VkMemInputProvider);
typedef enum {
    VK_EXEC_HOST_INPUT_READY,
    VK_EXEC_HOST_INPUT_UNSUPPORTED,
    VK_EXEC_HOST_INPUT_BLOCKED,
    VK_EXEC_HOST_INPUT_STALE,
    VK_EXEC_HOST_INPUT_STOPPED,
    VK_EXEC_HOST_INPUT_CAPACITY,
    VK_EXEC_HOST_INPUT_ABSENT,
    VK_EXEC_HOST_INPUT_CHANGED,  // source bytes may have changed during this observation
    VK_EXEC_HOST_INPUT_UNTRACKED, // no exact-page write observation is available
} VkExecHostInputResult;
typedef struct {
    const uint8_t *bytes;
    void *opaque;
    RcInterval mapping; // First exact clipped slice, when the original borrow reached mapping.
    const char *refusal; // The failed check, if recorded.
} VkExecHostInputRead;
// At the input phase of an ordered operation, borrow readable bytes from the exact retained provider.
// The caller retains span ownership through finish/cancel and consumes CPU results only after successful finish.
// The hold protects mapping and GPU producer order. Ordinary CPU store ordering remains with the guest.
VkExecHostInputResult vk_exec_host_input_begin(const VkMemSpanCapture *, const VkMemSpan *, uint64_t address,
                                              uint64_t size, RcRegistryPosition, VkExecHostInputRead *);
// Observe a current coherent import after the actual earlier command prefix, using exact native output
// and CPU page proof. Copy into an already allocated immutable destination before finish.
VkExecHostInputResult vk_exec_host_input_begin_ordered(struct VkExec *, const VkMemSpanCapture *, const VkMemSpan *,
    uint64_t address, uint64_t size, RcRegistryPosition, VkExecHostInputRead *);
VkExecHostInputResult vk_exec_host_input_finish(VkExecHostInputRead *);
void vk_exec_host_input_cancel(VkExecHostInputRead *);
// Translate an NVN storage pointer from its canonical logical bytes. Only a current host import with
// ordered, visible contents qualifies. This never represents a live mirror's native byte view.
VkExecHostInputResult vk_exec_host_pointer_read(uint64_t address, RcRegistryPosition, uint64_t *value);
// Only explicit physical backing IDs from captured device intervals qualify. Finalized pools with
// remaining native users still retain the backing. The slot is cached without taking a provider reference.
int vk_mem_device_backing_reclaimed(RcBackingId *);
// Texture lock held. Bind and retain the exact physical pool provider selected by the span.
typedef struct VkMemImageAliasBaseline VkMemImageAliasBaseline;
// A successful NULL baseline leaves image-only coverage available. A non-NULL baseline owns exact physical buffer
// ranges through native use. Its allocation, mapping and original read must still be current before recording.
RcRegistryResult vk_mem_image_alias_baseline_capture(uint64_t address, uint64_t bytes, const RcInterval *mapping,
    uint32_t count, RcRegistryPosition read, VkMemImageAliasBaseline **out);
RcRegistryResult vk_mem_image_alias_baseline_capture_handed(uint64_t address, uint64_t bytes, const RcInterval *,
    uint32_t count, RcRegistryPosition cut, VkMemImageAliasBaseline **, uint64_t *fill_upload);
RcRegistryResult vk_mem_image_alias_baseline_capture_submitted(uint64_t address, uint64_t bytes, const RcInterval *,
    uint32_t count, RcRegistryPosition read, VkMemImageAliasBaseline **, uint64_t *fill_upload);
// Pre-wait authority qualification only. Native source access still requires the exact fill to complete.
const RcRegistryBufferSpan *vk_mem_image_alias_baseline_submitted_spans(const VkMemImageAliasBaseline *,
    uint64_t fill_upload, uint32_t *count);
const RcRegistryBufferSpan *vk_mem_image_alias_baseline_spans(const VkMemImageAliasBaseline *, uint32_t *count);
int vk_mem_image_alias_baseline_current(const VkMemImageAliasBaseline *);
int vk_mem_image_alias_baseline_source(const VkMemImageAliasBaseline *, uint32_t baseline,
    RcRegistryBufferLocation, uint64_t size, VkBuffer *, VkDeviceSize *);
void vk_mem_image_alias_baseline_release(VkMemImageAliasBaseline **);
typedef enum {
    VK_MEM_IMAGE_ALIAS_DIRECT,
    VK_MEM_IMAGE_ALIAS_PRIVATE_PRESERVE,
    VK_MEM_IMAGE_ALIAS_STALE,
    VK_MEM_IMAGE_ALIAS_FAILED,
    VK_MEM_IMAGE_ALIAS_OCCUPIED
} VkMemImageAliasPlacement;
// Texture lock held. Preplacement is advisory; only a successful bind transfers a physical pool hold to the image.
// Both calls validate the normalized bytes of the whole actual Vulkan requirement, including padding.
// An image returned with STALE can have bound natively before the remap. Retire it with vk_defer_image.
VkMemImageAliasPlacement vk_mem_image_alias_preplacement(uint64_t address, const VkMemoryRequirements *requirements,
    const VkMemSpan *selected, const RcInterval *mapping, uint32_t mapping_count);
VkMemImageAliasPlacement vk_mem_bind_image_alias(VkImage image, uint64_t address, const VkMemoryRequirements *requirements,
    const VkMemSpan *selected, const RcInterval *mapping, uint32_t mapping_count, VkResult *native_failure);
int vk_mem_image_alias_overlaps(uint64_t address, uint64_t size, const VkMemSpan *, VkImage *, uint32_t capacity);
int vk_mem_image_alias_current(VkImage image, uint64_t addr, uint64_t size, VmaAllocation allocation,
                               const RcInterval *intervals, uint32_t count);
typedef struct VkMemoryImageHold VkMemoryImageHold;
typedef struct {
    VkDeviceMemory memory;
    VkDeviceSize offset, size;
} VkImageMemoryPlacement;
int vk_mem_image_hold_placement(const VkMemoryImageHold *, VkImage, VkImageMemoryPlacement *);
VkMemoryImageHold *vk_mem_image_hold(VkImage image);
VkMemoryImageHold *vk_mem_image_hold_retain(VkMemoryImageHold *);
// The caller owns a live hold. Verify the immutable VkImage before extending its lifetime.
VkMemoryImageHold *vk_mem_image_hold_retain_exact(VkMemoryImageHold *, VkImage exact);
int vk_mem_image_hold_retired(const VkMemoryImageHold *);
int vk_mem_image_held(VkImage image);
void vk_defer_image_owned_view(VkImage image, VkImageView view);
void vk_mem_image_release(VkMemoryImageHold *hold);
// Changes whenever vk_mem_resolve plus VK_MEM_USE may give another span for an address: a pool added, finalized or
// mapped, a mirror created, refused or forgotten, a submission started.
uint64_t vk_mem_gen(void);
// Page write generations over guest memory (mem_written_pages plus vk_guest_written). Returns 1 when a page of the
// range was written after generation `since`, 0 when not, -1 when the range is not write-watched (treat as written);
// *now is the generation to pass next time.
int vk_mem_changed(uint64_t addr, uint64_t size, uint32_t since, uint32_t *now);
// The same answer from per-epoch write-watch snapshots of 2 MiB regions: no system call once the regions were queried in
// this epoch (a guest submission). For checks that may see writes made during the epoch one epoch late.
// Optional guest page bounds [lo, hi) enclosing all observed writes. Empty when unavailable.
typedef struct { uint64_t lo, hi; } VkDirtyPages;
int vk_mem_changed_snap(uint64_t addr, uint64_t size, uint32_t since, uint32_t epoch, uint32_t *now,
                        VkDirtyPages *dirty);
void vk_mem_snap_stats(uint64_t *checks, uint64_t *calls, uint64_t *untracked);   // since the last call

// Upload batch: one command buffer on gfx_queue that texture uploads and texel copies record into. begin/end bracket
// the recording (recursive lock); vk_staging (between them) returns a mapped host buffer that lives until the batch
// completes, or, with alloc non-NULL, until the caller's resource_core_vma_destroy_buffer after vk_upload_flush(1).
// Caller-owned staging must use that shared destruction adapter so its capacity ownership settles. The replay calls
// vk_upload_flush(0) before each submission that may read uploaded images; queue order and the batch's closing
// barrier make the uploads visible to it. Returns the batch sequence number; wait=1 blocks until it completed.
// NX2NAVI_FIX_ZERO (default 1): physical pools are zeroed once at creation and GPU-only images with memory of their own
// are cleared when created; 0 leaves both as the driver hands them out.
int vk_fix_zero(void);
VkCommandBuffer vk_upload_begin(void);
void vk_upload_end(void);
uint64_t vk_upload_dependency(void);
// Exact active upload caller only; zero for another command buffer or outside begin/end, even if a batch is open.
uint64_t vk_upload_recording_sequence(VkCommandBuffer cb);
void vk_upload_defer_release(uint64_t sequence, void (*release)(void *), void *owner);
void vk_upload_progress(uint64_t *submitted, uint64_t *completed);
void *vk_staging(VkDeviceSize size, VkBuffer *buf, VmaAllocation *alloc);
// A 256-byte-aligned range of the staging arena owned by the open upload batch (vk_memory.c): *buf at *off. Only
// between vk_upload_begin and vk_upload_end; a large request gets a buffer of its own at offset 0.
void *vk_staging_at(VkDeviceSize size, VkBuffer *buf, VkDeviceSize *off);
typedef struct {
    VkDeviceSize size;
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationCreateInfo placement;
    uint8_t *map;
    uint32_t pool_memory_type;
    int from_cache;
} VkPreparedStaging;
// Plan outside upload recording. A cached span is held immediately and needs no new capacity.
VkResult vk_staging_prepare_plan(VkDeviceSize size, VkPreparedStaging *stage, RcAdmissionVector *peak);
VkResult vk_staging_prepare_allocate(VkPreparedStaging *stage, RcBudgetTicket *peak, int required);
// Claim a compatible completed allocation without an allocation attempt or a wait.
int vk_staging_prepare_reuse(VkDeviceSize size, VkPreparedStaging *stage);
void vk_staging_prepare_cancel(VkPreparedStaging *stage);
VkBuffer vk_staging_prepare_record(VkCommandBuffer cb, VkPreparedStaging *stage);
int vk_staging_prepare_available(VkDeviceSize size);
uint64_t vk_staging_prepare_supplier(VkDeviceSize size, uint64_t after);
int vk_staging_prepare_wait(uint64_t handed_upload);
uint64_t vk_upload_flush(int wait);
// Texture work on the render and replay threads, summed per frame (vk_texture.c vk_texstat_frame, the "texup:" line of
// NX2NAVI_TEXUP_LOG_MS). vk_texstat_time adds the time since t0 (kern_now_ns) and one event; vk_texstat_add adds events
// and bytes. Other threads are not counted. CREATE: new texture entries (image, memory, first transition); UPLOAD: the
// whole upload, of which STAGE (staging range or buffer), COPY (storage into staging: copy, untile, CPU decode) and ASTC
// (vk_astc_record, of which SCRATCH: a new scratch buffer); STG_NEW: staging buffers made by vmaCreateBuffer; DEVMEM: new
// VkDeviceMemory of 32 MiB or more (VMA blocks, dedicated allocations); FLUSH: vk_upload_flush (submit, retire, frees).
enum { VK_TS_CREATE, VK_TS_UPLOAD, VK_TS_STAGE, VK_TS_STG_NEW, VK_TS_COPY, VK_TS_ASTC, VK_TS_SCRATCH, VK_TS_DEVMEM,
       VK_TS_FLUSH, VK_TS_N };
void vk_texstat_time(int kind, uint64_t t0);
void vk_texstat_add(int kind, uint64_t n, uint64_t bytes);
// Deferred destruction: freed once the upload batches recorded so far and the timeline value last noted have completed.
void vk_defer_image(VkImage img, VmaAllocation alloc);   // alloc NULL: vkDestroyImage only
// Same lifetime as vk_defer_image; charge owned allocation bytes until actual allocator destruction completes.
// Pending bytes prevent duplicate reclamation, not a promise of system commit relief (VMA may retain its blocks).
void vk_defer_image_reclaim(VkImage img, VmaAllocation alloc, uint64_t bytes);
// Callback reports actual destruction, after upload and scene use. It can run under the upload lock during drain:
// use only leaf bookkeeping/atomics; never acquire renderer locks or call back into Vulkan/memory here.
void vk_defer_image_notify(VkImage img, VmaAllocation alloc, uint64_t bytes,
                           void (*destroyed)(void *), void *context);
// Material owner has already excluded new users. Destroy only with actual completed consumers and upload.
int vk_mem_image_destroy_completed(VkImage, VmaAllocation, uint64_t scene, uint64_t upload,
                                    void (*destroyed)(void *), void *context);
void vk_texture_retire_drain(void); // After backend destruction at shutdown: reap stable material metadata.
uint64_t vk_mem_reclaim_pending(void);
void vk_defer_view(VkImageView view);
// free(p) once the GPU is past the submission being recorded, as vk_defer_image: a host record other threads may still
// read (a destroyed texture's, vk_texture.c destroy_tex).
void vk_defer_free(void *p);
int vk_exec_image_gone(VkImage image);   // texture lock held: the replay's views of image go with it; returns how many
void vk_defer_buffer(VkBuffer buf, VmaAllocation alloc);
// Exclusive upload users only: the caller relinquishes the buffer after its last recorded upload sequence.
// No scene command, recorded CPU packet or later upload may retain/use it after this call.
void vk_upload_defer_buffer(VkBuffer buf, VmaAllocation alloc, uint64_t last_upload_seq);
// ---- ASTC (vk_astc.c). NX2NAVI_ASTC = auto | gpu | native | cpu | check; see vk_astc.c.
enum { VK_ASTC_CPU, VK_ASTC_GPU, VK_ASTC_NATIVE, VK_ASTC_CHECK };
int vk_astc_mode(void);                 // decided once per device (logged); VK_ASTC_CPU before vk_init
int vk_astc_native(VkFormat astc_fmt);  // 1: images of this ASTC format are created as such and get the raw blocks
int vk_astc_gpu(void);                  // 1: ASTC images decoded to RGBA8 by the compute decoder (vk_astc_record)
int vk_astc_bc(void);                   // 1: those images are BC3 instead, encoded after the decode (NX2NAVI_ASTC_BC=1)
int vk_astc_bc1(void);                  // 1: and the ones whose blocks decode opaque BC1 (NX2NAVI_ASTC_BC1)
// One image region decoded from raw blocks: tight rows of 16-byte blocks at byte src of the source buffer (slices one
// after the other), w x h texels and slices deep, written at offset of level / layer. bl: the blocks at src are the
// level's block-linear storage as the game laid it out, block height 2^th and depth 2^td GOBs (tex_deswizzle.c
// nvn_bl_offset), and the decoder reads each block from its place in it (no CPU deswizzle); src is 16-byte aligned.
typedef struct { VkDeviceSize src; uint32_t w, h, slices, level, layer; VkOffset3D offset; uint8_t bl, th, td; } VkAstcRegion;
// Records into cb, the open upload batch (between vk_upload_begin and vk_upload_end): src (TRANSFER_SRC, src_size
// bytes of blocks) goes to a device-local scratch buffer, a dispatch per region decodes it, and the texels are copied
// into img (VK_IMAGE_LAYOUT_GENERAL). The caller records its usual barriers around an image copy before and after the
// call. readback (optional, host-visible, TRANSFER_DST): also gets the decoded texels, the regions one after the other.
// bc: img is BC3 (TexVkFormat.astc_bc 1) or BC1 (2); the texels are encoded to its blocks before the copy, and readback
// is ignored.
// 1 when recorded. 0 when the decoder is unavailable or cannot take these regions, and -1 when its scratch buffer
// cannot be allocated: nothing is recorded in either case, and after -1 the same call can succeed once memory is free.
int vk_astc_record(VkCommandBuffer cb, VkBuffer src, VkDeviceSize src_size, const VkAstcRegion *rg, int n, uint32_t bw,
                   uint32_t bh, int srgb, int bc, VkImage img, VkImageAspectFlags aspect, VkBuffer readback);
void vk_astc_scratch_fail(int n);   // a test's hook: the next n scratch buffer requests fail (tex_probe.c)
uint64_t vk_texture_upload_retries(void);   // uploads left to be tried again: their decode had no memory (vk_texture.c upload)
// vk_texture.c "split fill", a test's hooks (tex_probe.c): the mode from here on (tex_split_fill: 0, 1 or 2), and how
// many entries started with a copy of the released entry they replaced and how many were filled from their storage.
void vk_texture_split_fill_set(int mode);
void vk_texture_split_counts(uint64_t *copies, uint64_t *fills);
VkDeviceSize vk_astc_region_bytes(const VkAstcRegion *r, uint32_t bw, uint32_t bh);   // source bytes the region reads
void vk_astc_stats(uint64_t *dispatches, uint64_t *texels);   // since start
uint64_t vk_astc_scratch_bytes(void);                          // the decoder's shared scratch now
// NX2NAVI_ASTC_TIME=1: GPU timestamps around each vk_astc_record's dispatches (logged per footprint and the largest one)
void vk_astc_time_enable(int on);
int vk_astc_time_get(uint32_t bw, uint32_t bh, uint64_t *texels, uint64_t *calls, double *us);   // totals so far
void vk_astc_time_summary(void);        // logs the totals
void vk_astc_release(void);             // vk_device.c, before the device goes: frees the decoder's objects
void vk_mem_note_submit(uint64_t timeline_value);  // vk_sync.c: after each submit that signals ctx->timeline with this value
void vk_mem_collect(void);                         // frees what has completed
typedef struct { uint64_t imported_pools, physical_pools, virtual_pools, import_failures, upload_batches; } VkMemStats;
void vk_mem_stats(VkMemStats *s);
void vk_mem_shutdown(void);                        // before vk_shutdown: waits idle, frees every memory and texture object

// Textures. Images are keyed by storage GPU address plus descriptor and stay in VK_IMAGE_LAYOUT_GENERAL. A mirrored
// texture (CPU-written pool storage) is uploaded on first use and again when its storage pages were
// written; once the GPU wrote it (vk_texture_gpu_written) the image is its content. A returned pointer stays valid
// until the next call that may drop textures (release, pool fini); copy what outlives that.
typedef struct {
    VkImage image;
    VkFormat format;               // after the fallbacks of tex_vk_format (ASTC -> RGBA8, RGB -> RGBA)
    VkImageType type;
    VkImageViewType view_type;     // view of the whole image
    VkImageAspectFlags aspect;
    VkImageUsageFlags usage;
    uint32_t width, height, depth, levels, layers, samples;
    uint64_t addr;                 // storage GPU address
    uint8_t preserve_pool_bytes;   // optional alias declined to preserve a selected physical RAW source
    uint8_t rgba8_linear_view;      // exact ordinary 2D SRGB allocation admits an UNORM sampled view
    uint8_t alpha_one;             // X format: sample alpha as 1 through the view swizzle
    uint8_t chan[4];               // TexVkFormat.chan: component order of a stand-in format, applied by the view swizzle
    uint8_t mirrored, gpu_owned;
    uint8_t bound;                 // aliases physical-pool memory (physical and virtual pools)
    uint8_t block_views;           // BC image that renders through uncompressed views of its blocks (BLOCK_TEXEL_VIEW_
                                   // COMPATIBLE + EXTENDED_USAGE: only those views may use COLOR_ATTACHMENT)
    uint8_t swz[4];               // NVN texture swizzle selectors (nvnTextureBuilderSetSwizzle); a view swizzle replaces them
    uint8_t fb_usage;              // created with ATTACHMENT_FEEDBACK_LOOP usage (VK_FEEDBACK_LAYOUT, attachable images)
    uint8_t fb_learned;            // a draw sampled it while it was a written attachment of the open pass: its passes
                                   // attach it in ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT (vk_exec.c vk_exec_pass_begin)
    uint8_t scale;                 // 1, or NX2NAVI_SHADOW_SCALE for a shadow map made larger than the game's (vk_texture.c
                                   // shadow_scale_of): width and height above are the larger ones, the game's regions
                                   // are multiplied by it
    uint8_t skip;                  // texture detail (vk_texture.c detail_skip): the game's top mip levels the image was made
                                   // without. Image level i holds the game's level i + skip, and width, height and levels
                                   // above are the image's, so the game's level skip is their base. 0 everywhere else.
    uint8_t gone;                  // destroyed (vk_texture.c destroy_tex, texture lock held): the record stays readable until
                                   // the GPU is past the submission being recorded (vk_defer_free), as the replay and DLSS
                                   // keep pointers to render targets across a command buffer; no view of it is cached
} VkTexImage;
// Capture the selected physical image and hold its storage until graphics composition completes.
VkTexturePresentSource *vk_texture_present_capture(const tex_t *, VkTexImage *selected);
// Exact guest texture incarnation recorded when the owner was captured, or zero for an untagged owner.
GfxPresentPictureSource vk_texture_present_picture(const VkTexturePresentSource *);
// Fork the physical owner of a live target, before handing its exact image and a view to composition.
VkTexturePresentSource *vk_texture_present_retain_image(const VkTexImage *selected, VkImage exact);
// Extend an established physical placement lease, including an older retained frame. The caller
// proves descriptor/view generation and actual queue order before selecting the image.
VkTexturePresentSource *vk_texture_present_retain_held_image(VkMemoryImageHold *, VkImage exact);
void vk_texture_present_release(VkTexturePresentSource *);
// Capture the explicit transfer image at the ordered read horizon, without resolving or uploading a replacement.
VkTexturePresentSource *vk_texture_transfer_read_capture(const tex_t *, VkTexImage *);
// Returns true when the scene use has taken over the material or pin. A raw image still needs the loan.
int vk_texture_transfer_read_use(const VkTexturePresentSource *, VkImage);
VkTexturePresentSource *vk_texture_transfer_read_select(const tex_t *, VkTexImage *);
int vk_texture_transfer_read_current(const VkTexturePresentSource *, VkImage);
// Negative rejects stale or incomplete projected bytes, zero leaves another layout to the compatibility path.
int vk_texture_transfer_read_prove(const tex_t *, const VkTexImage *, VkImageSubresourceLayers, const NVNcopyRegion *);
struct ResourceCommandEffects;
typedef struct VkTextureProjectedTransfer VkTextureProjectedTransfer;
int vk_texture_alias_volume_transfer(const tex_t *, const view_t *);
int vk_texture_host_volume_transfer(const tex_t *, const view_t *);
int vk_texture_host_color_transfer(const tex_t *, const view_t *);
int vk_texture_projected_transfer_capture(const tex_t *, const view_t *, const NVNcopyRegion *,
                                          VkTexImage *, VkTextureProjectedTransfer **);
// An ordered sequence of disjoint same-image copies. Use these retained regions in order, with a barrier
// between copies, and report their union after the last one. Begin rechecks the original source proof.
int vk_texture_projected_self_copy_capture(const tex_t *, const VkTexImage *, const VkImageCopy *,
                                           uint32_t, VkTextureProjectedTransfer **);
const VkImageCopy *vk_texture_projected_self_copy_regions(const VkTextureProjectedTransfer *, uint32_t *);
int vk_texture_projected_transfer_current(const VkTextureProjectedTransfer *);
VkImage vk_texture_projected_transfer_image(const VkTextureProjectedTransfer *);
RcRegistryResult vk_texture_projected_transfer_declare(const VkTextureProjectedTransfer *,
                                                        struct ResourceCommandEffects *, RcRegistryPosition);
int vk_texture_projected_transfer_begin(VkTextureProjectedTransfer *);
void vk_texture_projected_transfer_written(VkTextureProjectedTransfer *);
void vk_texture_projected_transfer_cancel(VkTextureProjectedTransfer **);
// Fork the already selected ordinary sampled image into the current recording while the texture lock is held.
int vk_texture_record_selected_sample_locked(const VkTexturePresentSource *, VkImage);
// Reserve a previously captured image's actual writer coverage. Texture lock held.
void vk_texture_image_write_prepare_locked(const VkTexImage *, uint64_t logical_size);
void vk_texture_image_write_prepare(VkImage, uint64_t address, uint64_t logical_size);
// A 2D texture a shader samples as a 3D one: its one-slice 3D copy, refreshed when its content changed (vk_texture.c);
// NULL when it does not qualify. Texture lock held.
// Immutable one-layer 3D substitute. Nonnull CB is an ordered, outside-rendering scene copy. NULL CB only looks up.
// required: 0 speculative/no deferral; 1 consuming binding; 2 strict preflight (missing eligible copy defers).
const VkTexImage *vk_texture_flat3d_locked(const VkTexImage *img, uint32_t layer, VkCommandBuffer ordered_cb, int required);
// Select the weak HDR repeat identity when the last draw copied a float picture. Texture lock held.
void vk_hdr_present_source_locked(const VkTexImage *win, VkTexImage *source);
void vk_hdr_image_gone(VkImage image);
const VkTexImage *vk_texture_get(const tex_t *t);   // created and made current; NULL for BUFFER or unmapped formats
const VkTexImage *vk_texture_get_write(const tex_t *t); // private writable contents, before writable view capture
const VkTexImage *vk_texture_get_transfer_write(const tex_t *t);
// Read-only recipe geometry for preparing the buffer's source footprint before a compatible transfer.
int vk_texture_transfer_write_geometry(const tex_t *, VkTexImage *);
int vk_texture_alias_region_transfer(const tex_t *, const view_t *, const NVNcopyRegion *);
int vk_texture_alias_depth_transfer(const tex_t *, const view_t *, const NVNcopyRegion *);
int vk_texture_alias_depth_attachment_range(const tex_t *, const view_t *, VkImageSubresourceRange *);
int vk_texture_alias_attachment_range(const tex_t *, const view_t *, int depth, VkImageSubresourceRange *);
int vk_texture_attachment_base_view(const tex_t *, const view_t *);
int vk_texture_alias_depth_clear_candidate(const tex_t *);
int vk_texture_alias_depth_clear_bound(const tex_t *);
const VkTexImage *vk_texture_get_alias_transfer_write(const tex_t *);
void vk_texture_alias_transfer_written(const tex_t *, VkImage, const view_t *, const NVNcopyRegion *);
void vk_texture_alias_clear_written(const tex_t *, VkImage, const VkImageSubresourceRange *);
void vk_texture_alias_depth_clear_written(const tex_t *, VkImage, const NVNcopyRegion *);
VkImage vk_texture_alias_probe_receiver_image(const tex_t *);
typedef struct VkTextureAliasRead VkTextureAliasRead;
// Receives the actual initialized LOAD receipt and retains its allocation at its original scope.
VkTexturePresentSource *vk_texture_present_retain_alias_image(VkTextureAliasRead *, VkImage exact);
typedef enum {
    VK_TEXTURE_ALIAS_ORDER, VK_TEXTURE_ALIAS_MAPPING, VK_TEXTURE_ALIAS_LAYOUT, VK_TEXTURE_ALIAS_COVERAGE,
    VK_TEXTURE_ALIAS_OWNER, VK_TEXTURE_ALIAS_CAPACITY
} VkTextureAliasPhase;
typedef struct {
    VkTextureAliasPhase phase;
    RcRegistryResult status;
    const char *layout_gate;
    RcInterval receiver;
    uint64_t registration, material_serial;
    VkImage image;
    RcRegistryNativeImageFailure coverage;
    RcRegistryBufferLocation raw_origin;
    RcRegistryResult raw_status;
    uint32_t raw_spans;
    uint8_t raw_attempted;
} VkTextureAliasFailure;
typedef struct {
    VkTextureAliasRead *alias;
    VkTexturePresentSource *resident;
    VkTexImage image;
    const tex_t *texture;
    RcRegistryPosition read;
    RcInterval *mapping, *demand;
    uint32_t mapping_count, demand_count;
} VkTextureTransferInput;

int vk_texture_transfer_source_geometry(const tex_t *, VkTexImage *);
int vk_texture_transfer_input_capture(const tex_t *, const VkTexImage *, VkImageSubresourceLayers, const NVNcopyRegion *,
                                      VkTextureTransferInput *, VkTextureAliasFailure *);
int vk_texture_transfer_input_current(const VkTextureTransferInput *);
int vk_texture_transfer_input_use(const VkTextureTransferInput *);
void vk_texture_transfer_input_cancel(VkTextureTransferInput *);

int vk_texture_alias_read_capture(const tex_t *, VkTextureAliasRead **, VkTexImage *, VkTextureAliasFailure *);
int vk_texture_alias_read_current(VkTextureAliasRead *);
// A direct sample may hold a submitted RAW fill candidate before it has a copyable source.
int vk_texture_alias_read_raw_progress(VkTextureAliasRead *, RcRegistryPosition, uint64_t *upload);
int vk_texture_alias_read_raw_finish(VkTextureAliasRead *, RcRegistryPosition);
int vk_texture_alias_sample_selection_current_locked(int logical, VkTextureAliasRead *);
// A winning image producer can supply a B2B source. OK with no receipt keeps the selected raw buffer.
RcRegistryResult vk_texture_alias_buffer_capture(const VkMemSpanCapture *, const VkMemSpan *, uint64_t address,
    uint64_t size, VkTextureAliasRead **, VkTextureAliasFailure *);
RcRegistryResult vk_texture_alias_buffer_capture_rows(const VkMemSpanCapture *, const VkMemSpan *, uint64_t address,
    uint64_t bytes, uint64_t row_bytes, uint64_t row_stride, VkTextureAliasRead **, VkTextureAliasFailure *);
// The held caller supplies the exact fixed writable span at the original even read.
// OK without a seed proves direct native use. A returned seed owns its exact mixed sources and destination.
RcRegistryResult vk_texture_alias_buffer_write_input_capture(const VkMemSpanCapture *, const VkMemSpan *,
    uint64_t address, uint64_t bytes, VkTextureAliasRead **seed, VkTextureAliasFailure *);
int vk_texture_alias_buffer_write_input_current(const VkMemSpanCapture *, const VkMemSpan *,
    uint64_t address, uint64_t bytes, VkTextureAliasRead *seed);
// Stage after the shared ordered readback. Validate the entire cohort before recording any destination copy.
int vk_texture_alias_buffer_write_input_stage(VkTextureAliasRead *);
void vk_texture_alias_buffer_write_input_record(VkTextureAliasRead *, VkCommandBuffer, uint64_t fence);
// Call only for actual shader issue on the seeded command buffer, before its normal effect commit.
void vk_texture_alias_buffer_write_input_issued(VkTextureAliasRead *, VkCommandBuffer);
// Only after the prefix containing vk_texture_alias_read_record actually completed.
int vk_texture_alias_buffer_source(VkTextureAliasRead *, VkBuffer *, VkDeviceSize *);
int vk_texture_alias_buffer_copy(VkTextureAliasRead *, void *, uint64_t);

struct RcVmaDependency;
typedef struct {
    uint64_t upload;
    VkDeviceSize size;
} VkTextureAliasStagingProgress;
VkResult vk_texture_alias_read_prepare(VkTextureAliasRead *const *, uint32_t count, int required,
    uint64_t after_upload, VkTextureAliasStagingProgress *, struct RcVmaDependency *);
void vk_texture_alias_read_unprepare(VkTextureAliasRead *const *, uint32_t count);
void vk_texture_alias_read_record(VkTextureAliasRead *, VkCommandBuffer, uint64_t);
const VkTexImage *vk_texture_alias_read_upload(VkTextureAliasRead *, VkCommandBuffer, uint64_t);
void vk_texture_alias_read_cancel(VkTextureAliasRead *);
typedef struct VkTextureBoundRelocation VkTextureBoundRelocation;
int vk_texture_bound_relocation_capture(const tex_t *destination, VkTextureBoundRelocation **);
int vk_texture_bound_relocation_current(VkTextureBoundRelocation *);
VkResult vk_texture_bound_relocation_allocate(VkTextureBoundRelocation *, struct RcVmaDependency *);
void vk_texture_bound_relocation_record(VkTextureBoundRelocation *, VkCommandBuffer, uint64_t fence);
int vk_texture_bound_relocation_commit(VkTextureBoundRelocation *);
void vk_texture_bound_relocation_cancel(VkTextureBoundRelocation *);
typedef enum {
    VK_TEXTURE_STORAGE_SEED_REFUSED = -1,
    VK_TEXTURE_STORAGE_SEED_UNSUPPORTED,
    VK_TEXTURE_STORAGE_SEED_CURRENT,
    VK_TEXTURE_STORAGE_SEED_CAPTURED,
} VkTextureStorageSeedResult;
// The caller owns the selected image and exact storage view. Capture stays at the original even read.
// CAPTURED uses the ordinary alias cohort to seed only those planes, before the dispatch records.
VkTextureStorageSeedResult vk_texture_storage_seed_capture(const VkTexImage *, VkImageSubresourceRange,
                                                            VkTextureAliasRead **, VkTextureAliasFailure *);
typedef struct VkTextureStorageTarget VkTextureStorageTarget;
typedef struct VkTextureWriteTarget VkTextureWriteTarget;
enum {
    VK_TEXTURE_STORAGE_CAPTURE_UNSUPPORTED = -2,
    VK_TEXTURE_STORAGE_CAPTURE_INVALIDATED = -1,
    VK_TEXTURE_STORAGE_CAPTURE_ABSENT = 0,
    VK_TEXTURE_STORAGE_CAPTURE_READY = 1,
    VK_TEXTURE_STORAGE_CAPTURE_COW = 2,
    VK_TEXTURE_STORAGE_CAPTURE_BOUND = 3,
};
const VkTexImage *vk_texture_storage_binding_locked(int logical, int exact, view_t *, int *has_view);
int vk_texture_effect_independent_storage(int logical, int exact, int written, VkTexImage *, view_t *, int *has_view, uint64_t *size,
                                          VkTextureStorageTarget **, const char **reason);
// A nonnull material output opts a written binding into exact host copy-on-write preparation.
int vk_texture_effect_storage_capture(int logical, int exact, VkTexImage *, view_t *, int *has_view, uint64_t *size,
                                     VkTextureStorageTarget **, VkTextureWriteTarget **material, int written, int allow_bound, const char **reason);
// The selected bound version is held by the target while its preparation is suspended.
VkTextureBoundRelocation *vk_texture_storage_target_bound_plan(VkTextureStorageTarget *);
int vk_texture_storage_target_bound_commit(VkTextureStorageTarget *, VkTexImage *selected);
int vk_texture_storage_target_current(const VkTextureStorageTarget *);
int vk_texture_storage_target_load_current(const VkTextureStorageTarget *, const VkImageSubresourceRange *);
int vk_texture_storage_target_host_mirror(const VkTextureStorageTarget *);
int vk_texture_storage_target_replaced_stale(const VkTextureStorageTarget *);
VkResult vk_texture_storage_target_prepare_host_cpu(VkTextureStorageTarget *, struct RcVmaDependency *);
// Both retained destinations must be current and physically independent at their original read.
int vk_texture_storage_targets_disjoint(const VkTextureStorageTarget *, const VkTextureStorageTarget *);
int vk_texture_storage_target_disjoint_input(const VkTextureStorageTarget *, const RcInterval *, uint32_t, VkImageMemoryPlacement);
int vk_texture_write_target_disjoint_placement(const VkTextureWriteTarget *, VkImageMemoryPlacement);
VkImage vk_texture_storage_target_image(const VkTextureStorageTarget *, const VkTextureAliasRead *);
int vk_texture_storage_target_declare(const VkTextureStorageTarget *, const VkTextureAliasRead *, struct ResourceCommandEffects *);
// The caller holds the texture lock through use, command recording and the actual output report.
int vk_texture_storage_target_use(const VkTextureStorageTarget *, VkTextureAliasRead *, uint64_t fence, VkCommandBuffer);
int vk_texture_storage_target_read_use(const VkTextureStorageTarget *, const VkImageSubresourceRange *, uint64_t fence);
int vk_texture_storage_target_written(const VkTextureStorageTarget *, VkTextureAliasRead *,
                                      const VkImageSubresourceRange *defined_planes);
// After issuing a retained in-place storage operation, carry its original initialized plane into later native views.
int vk_texture_storage_target_written_preserving(const VkTextureStorageTarget *, const VkImageSubresourceRange *);
void vk_texture_storage_target_release(VkTextureStorageTarget *);
int vk_texture_alias_target_capture(const tex_t *, const view_t *, uint32_t layers, VkTextureAliasRead **,
                                    int ordered, VkTextureAliasFailure *);
const RcInterval *vk_texture_alias_read_intervals(const VkTextureAliasRead *, uint32_t *count);
int vk_texture_alias_target_current(VkTextureAliasRead *, VkImage);
int vk_texture_alias_target_load_current(VkTextureAliasRead *, VkImage, const VkImageSubresourceRange *);
// The LOAD receipt owns this initialized native placement until the last recorded use.
const VkTexImage *vk_texture_alias_target_image(const VkTextureAliasRead *);
const VkTexImage *vk_texture_alias_target_replacement(const VkTextureAliasRead *);
int vk_texture_alias_target_read_prior(VkTextureAliasRead *, VkImage target, VkImage sampled, int fragment);
int vk_texture_alias_target_prepare_replacement(VkTextureAliasRead *, const VkImageSubresourceRange *);
void vk_texture_alias_target_record_replacement(VkTextureAliasRead *, const VkImageSubresourceRange *);
void vk_texture_alias_target_cancel(VkTextureAliasRead *);
VkTextureAliasRead *vk_texture_alias_read_retain(VkTextureAliasRead *);
void vk_texture_alias_read_use(VkTextureAliasRead *, uint64_t fence);
VkImageView vk_texture_alias_read_view_locked(VkTextureAliasRead *, const VkImageViewCreateInfo *);
const VkTexImage *vk_texture_get_target(const tex_t *t);
int vk_texture_attachment_load_ready(const VkTexImage *selected, const VkImageSubresourceRange *);
struct ResourceCommandEffects;
int vk_texture_declare_pin_write(const tex_t *, struct ResourceCommandEffects *);
void vk_texture_explicit_read_use(const tex_t *, VkImage);
typedef struct {
    const tex_t *pin;
    VkImage image;
} VkPinnedTextureWrite;
void vk_texture_pinned_write_capture(const tex_t *, VkImage, VkPinnedTextureWrite *);
int vk_texture_pinned_write_matches_locked(const VkPinnedTextureWrite *, VkImage);
void vk_texture_pinned_write_release(VkPinnedTextureWrite *);
void vk_texture_pinned_write_use(const VkPinnedTextureWrite *);
void vk_texture_pinned_write_prepare(const VkPinnedTextureWrite *);
void vk_texture_pinned_write_recorded(const VkPinnedTextureWrite *);
typedef struct VkTextureTargetLoan VkTextureTargetLoan;
VkTextureTargetLoan *vk_texture_target_loan(const tex_t *, VkImage expected, int materialized);
const VkTexImage *vk_texture_target_loan_image(const VkTextureTargetLoan *);
int vk_texture_target_loan_current(const VkTextureTargetLoan *);
int vk_texture_target_loan_backing_current(const VkTextureTargetLoan *, const RcRegistryImageWriteExtent *, uint32_t);
int vk_texture_target_loan_private(const VkTextureTargetLoan *);
VkPinnedTextureWrite vk_texture_target_loan_take_pin(VkTextureTargetLoan *);
void vk_texture_target_loan_release(VkTextureTargetLoan *);
const VkTexImage *vk_texture_writable_locked(const VkTexImage *img); // texture lock held
struct RcVmaDependency;
int vk_texture_write_target_shape(const tex_t *, VkTexImage *);
// A T2T restore can seed its private target from the old destination's exact nonresident recipe.
int vk_texture_write_target_restore_shape(const tex_t *, VkTexImage *);
// Negative is a changed candidate, zero leaves compatibility available before the write target is owned.
int vk_texture_write_target_capture(const tex_t *, const VkTexImage *, VkTextureWriteTarget **);
int vk_texture_write_target_current(const VkTextureWriteTarget *);
int vk_texture_write_target_needs_allocation(const VkTextureWriteTarget *);
VkResult vk_texture_write_target_allocate(VkTextureWriteTarget *, struct RcVmaDependency *);
void vk_texture_write_target_declare(const VkTextureWriteTarget *, struct ResourceCommandEffects *);
const VkTexImage *vk_texture_write_target_begin(VkTextureWriteTarget *, VkCommandBuffer);
void vk_texture_write_target_written(VkTextureWriteTarget *);
void vk_texture_write_target_clear_written(VkTextureWriteTarget *, const VkImageSubresourceRange *);
// An operation may read or write this same exact private image in its subsequent phases.
void vk_texture_write_target_recorded(VkTextureWriteTarget *);
int vk_texture_write_target_use(const VkTextureWriteTarget *, VkImage);
void vk_texture_write_target_cancel(VkTextureWriteTarget *);
typedef struct {
    VkTextureWriteTarget **targets;
    uint32_t count;
} VkTextureWritePlan;

typedef enum {
    VK_TEXTURE_STORAGE_GONE,
    VK_TEXTURE_STORAGE_PINNED,
    VK_TEXTURE_STORAGE_TARGET,
    VK_TEXTURE_STORAGE_NEW_TARGET
} VkTextureStorageResult;

VkTextureStorageResult vk_texture_effect_ordered_image(int logical, int written, const VkTextureWritePlan *,
    VkTexImage *, view_t *, int *has_view, uint64_t *size, VkPinnedTextureWrite *, VkTextureWriteTarget **);
int vk_texture_effect_ordered_sample(int logical, const VkTextureWritePlan *, VkTexImage *, view_t *, int *has_view,
    VkTextureWriteTarget **, VkTexturePresentSource **);
int vk_texture_write_plan_independent(const VkTextureWritePlan *);
int vk_texture_write_plan_disjoint_pin(const VkTextureWritePlan *, const VkPinnedTextureWrite *);
// Compare one selected private writer's full canonical coverage with an exact retained native buffer input.
int vk_texture_write_disjoint_input(const VkTextureWriteTarget *, const VkPinnedTextureWrite *, const RcInterval *, uint32_t);
RcRegistryResult vk_texture_write_compare_input(const VkTextureWriteTarget *, const VkPinnedTextureWrite *,
                                                const RcInterval *, uint32_t, RcInterval *overlap);
VkResult vk_texture_write_plan_allocate(const VkTextureWritePlan *, struct RcVmaDependency *);
VkImage vk_texture_write_target_prepared_image(const VkTextureWriteTarget *);
void vk_texture_write_target_viewed(VkTextureWriteTarget *);
void vk_texture_write_target_bound(VkTextureWriteTarget *);
int vk_texture_write_target_bound_current(const VkTextureWriteTarget *);
typedef struct VkTextureRestore VkTextureRestore;
VkTextureRestore *vk_texture_restore_capture(const tex_t *, VkTexImage *shape);
// The transfer owns this optional COW target. Reserve its native placement with the restoration before either grows.
void vk_texture_restore_pair_write_target(VkTextureRestore *, VkTextureWriteTarget *);
VkResult vk_texture_restore_prepare(VkTextureRestore *, struct RcVmaDependency *);
int vk_texture_restore_spare_ready(void *);
uint64_t vk_texture_restore_upload_supplier(const VkTextureRestore *, uint64_t after);
const VkTexImage *vk_texture_restore_publish(VkTextureRestore *);
void vk_texture_restore_use(const VkTextureRestore *);
void vk_texture_restore_cancel(VkTextureRestore *);
// The startup probe selects an eligible material using the same residency detach as pressure reclamation.
int vk_texture_restore_probe_evict(const tex_t *);
uint64_t vk_texture_restore_probe_count(void);
uint64_t vk_mem_progress_prior(const struct RcVmaDependency *, uint64_t scene);
int vk_mem_observe_subresources(uint64_t addr, uint64_t size, const TextureSourceStamp *prior, uint32_t epoch,
                               const TextureSourceSpan *spans, uint32_t count, TextureSourceObservation *out);
// A texture slot; a slot whose texture was finalized follows the live texture at the same storage (hle_nvn.c).
const VkTexImage *vk_texture_by_id(int id, view_t *view, int *has_view);
// The replay looks up many slots per draw: it takes the texture lock once around them (vk_texture_lock) and uses the
// _locked lookups; vk_texture_slots_gen changes whenever a slot may resolve to another image or view.
void vk_texture_lock(void);
void vk_texture_unlock(void);
const VkTexImage *vk_texture_by_id_locked(int id, view_t *view, int *has_view);
// Shader-selection metadata only: 1 known, 0 absent, -1 content-dependent alias ambiguity (use the ordered
// resolver). No allocation, guest-byte access, freshness preparation or deferred-state mutation. Output has no
// image/readiness handles; only format, view_type and samples are authoritative for shader answers.
int vk_texture_metadata_locked(int id, VkTexImage *out, view_t *view, int *has_view);
// The replay workers (vk_exec.c "replay workers"): a chunk's lookup that would make or take back a texture entry
// before the chunk's turn returns nothing and says so here; the chunk replays the op again in turn
// (vk_texture_deferred_take clears it). vk_exec_turn_due: the calling thread records a chunk whose earlier chunks may not
// be done yet; vk_exec_turn waits until they are (no-op elsewhere).
int vk_texture_deferred(void);
// Mark a turn-dependent lookup for the recorder's existing retry. Acquires no lock.
int vk_texture_defer_until_turn(void);
int vk_texture_deferred_take(void);
void vk_texture_deferred_stats(uint64_t out[3]);   // since the last call, by what the lookup found: none, retired, released
int vk_exec_turn_due(void);
int vk_exec_in_chunk(void);   // the calling thread records a chunk of a parallel group
// vk_exec_shared: other threads may record now (a chunk, or the submitting thread past a group, NX2NAVI_REPLAY_PIPE),
// so the shared tables lock; vk_exec_behind: the calling thread is that submitting thread; vk_exec_hooks_wait: then
// waits until the group's first chunk ran the pass hooks.
int vk_exec_shared(void);
int vk_exec_behind(void);
// The replay thread, while no group of the replay workers is out (vk_exec.c "side work"): fn(arg, lo, hi) over items
// [0, n) in slices of grain, on the workers and the calling thread at once; returns when every slice is done. Without
// workers, or for one slice, the caller runs it alone.
void vk_exec_par(void (*fn)(void *arg, int lo, int hi), void *arg, int n, int grain);
// Inside a recording chunk, run both halves and join. A recorder waiting on the group can take half 1.
// Returns zero without calling fn when unavailable. Each half must be independent of renderer locks.
int vk_exec_chunk_pair(void (*fn)(void *arg, unsigned half), void *arg);
void vk_exec_hooks_wait(void);
// A pass hook (vk_contact_pass_begin, vk_dlaa_pass) about to change what draws read or to record: the submitting
// thread past a group waits for the group first; a group's hooks run in its first chunk, in turn already.
void vk_exec_hook_turn(void);
// Where the calling thread records in the order of the stream as the replay workers cut it (later records larger).
uint64_t vk_exec_stream_pos(void);
// The epoch of a uniform snapshot drop the calling thread makes (write) or of the drops it sees (vk_usnap.c).
uint64_t vk_exec_epoch(int write);
void vk_exec_turn(void);
VkSampler vk_sampler_by_id_locked(int id);
// The slot's sampler with nearest filtering and no anisotropy: a depth texture read without a compare lands on one texel
// on the Switch although the game binds a linear sampler. VK_NULL_HANDLE
// for an empty slot.
VkSampler vk_sampler_point_by_id_locked(int id);
// The slot's sampler with its compare on, for a depth-compare read (vk_texture.c, at the definition); the slot's own
// sampler when it compares already. VK_NULL_HANDLE for an empty slot.
VkSampler vk_sampler_compare_by_id_locked(int id);
VkSampler vk_sampler_pool_by_id_locked(int id, int comparison);
// Texture detail (vk_texture.c): what texture_detail = auto picks on this device, 0 full, 1 reduced, 2 low (the budget is
// read at the first call). For the menu's row.
int vk_texture_detail_auto(void);
const char *vk_texture_detail_name(void);   // the detail in effect: full, reduced or low; a fatal error's line
// The slot's sampler for a texture made without its top `skip` mip levels (VkTexImage.skip): its LOD clamp counts the
// game's levels, so it moves down by skip. The slot's own sampler when its clamp is open or skip is 0.
VkSampler vk_sampler_detail_by_id_locked(int id, uint32_t skip);
// Slot id resolved to a texture made without its top levels, and a shader binding that reads texels by position or
// asks for the size wants it: the texture is made again with every level and stays so (vk_texture.c "texture detail").
// Returns what the slot resolves to now, as vk_texture_by_id_locked does.
const VkTexImage *vk_texture_exact_by_id_locked(int id, view_t *view, int *has_view);
uint32_t vk_texture_slots_gen(void);
int vk_texture_slot_group_locked(int id); // Diagnostic group identity; caller holds the texture lock.
void vk_texture_create_budget_locked(int n);         // slot lookups create at most n entries (mapped storage only); -1 any
int vk_texture_create_deferred_locked(void);        // a lookup since the last call left a creation to the budget             // one past the highest texture slot registered so far (or more)
int vk_sampler_slot_count_locked(void);
int vk_sampler_slot_registered_locked(int id); // metadata only; does not create a sampler
int vk_sampler_slot_comparison_locked(int id);
int vk_sampler_slot_compare_state_locked(int id, unsigned *mode, unsigned *function);
int vk_sampler_pool_comparison_locked(void);
// Bindless pools (vk_exec_draw.c, "texture and sampler pools"): set 2 of every pipeline layout. Shaders that index
// the texture or sampler pool with handles from memory have their pool arrays moved there (vk_pipeline.c
// spirv_pool_rebind); VK_POOL_KINDS image bindings, one per view kind, then the sampler pool.
#define VK_POOL_TEX 32768                           // as the translator's Accessor.TexturePoolLength
#define VK_POOL_SAMP 4096                           // Accessor.SamplerPoolLength
#define VK_POOL_KINDS 16                            // kind = depth * 8 + dim * 2 + arrayed, dim 1D 2D 3D CUBE
extern volatile uint32_t vk_pool_kinds_used;        // bit k: a pipeline was made with a kind-k pool array
// Texture pool marks (vk_pipeline.c spirv_pool_mark, vk_exec_draw.c "pool marks"): set 2, this binding, a uint per pool
// slot that shaders set to 1 when they index the slot.
#define VK_POOL_MARK_BINDING (VK_POOL_KINDS + 1)
#define VK_POOL_COMPARISON_BINDING (VK_POOL_KINDS + 2)
void vk_pool_marks_decline(void);                    // optional resources refused before the first pool layout
int vk_pool_marks_enabled(void);                     // NX2NAVI_TEX_DISUSE: modules get the marks
extern volatile int32_t vk_pool_marks_missed;        // modules indexing the pool that could not be marked
void vk_pool_marks_read(const uint32_t *marks, uint32_t n);   // vk_texture.c: slots marked since the last read
// The texture pool's ring of descriptor sets (vk_exec_draw.c pool_take_set): a submission's first pool use takes the next
// set and brings it up to date. vk_pool_takes counts the takes whose sync left no slot behind (vk_exec_draw.c pool_sync);
// once it has moved on by more than VK_POOL_SETS since a pool entry changed, every set holds the change (vk_texture.c
// retire_sweep). It may count fewer takes than there were, which only delays that.
#define VK_POOL_SETS 4
uint64_t vk_pool_takes(void);
// Texture lock held: 1 while the texture pool refresh looks its slots up (vk_exec_draw.c pool_sync_entries), 0 after.
// Those lookups leave out a texture evicted by disuse, and are not counted as the texture being used.
void vk_texture_pool_ctx(int on);
VkDescriptorSetLayout vk_pool_set_layout(void);     // the pools' layout (made on first use)
int vk_pool_split_samplers(void);
VkDescriptorSetLayout vk_empty_set_layout(void);    // set 1 of layouts without one
// The upscaler stand-in (vk_tonemap.c): src (sampled) scaled to dw x dh and tonemapped into its own RGBA16F image, which
// it returns for the caller to blit into the target; VK_NULL_HANDLE when it cannot. The caller binds its own compute
// pipeline again afterwards.
VkImage vk_tonemap_record(VkCommandBuffer cb, VkImage src, VkImageView source_view, VkFormat sf, uint32_t dw, uint32_t dh);
void vk_tonemap_shutdown(void); // after the fallback commands have retired, before the device allocator
// What vk_texture_by_id(id) returns can change only when this does: the slot is registered again, or a texture entry at
// its storage address is created, dropped, released or taken back into use. Unlocked; read it before the lookup it
// guards (a lookup may create an entry). 0 when the slot was never registered: nothing is kept under 0.
uint64_t vk_texture_slot_token(int id);
int vk_texture_effect_range(int id, uint64_t *addr, uint64_t *size, uint64_t *token);
// Resolve and preserve the actual writable image during ordered declaration. Returns 1 ready, 0 absent, -1 deferred.
// out is a value snapshot, never a Tex pointer. Its image is held through this recording scene's real completion.
int vk_texture_effect_image(int id, int exact, VkTexImage *out, view_t *view, int *has_view, uint64_t *source_size,
                            VkPinnedTextureWrite *capture);
// The selected compute replacement carries the actual image through command recording and into HUD/FG
// if used. Possible whole-plane projection is optional and requires the same declared physical mapping.
enum { VK_TEXTURE_COLOR_BLIT_MAPPINGS = 8 };
typedef struct {
    VkImage image;
    VkMemoryImageHold *hold;
    RcRegistryPosition read;
    uint64_t address, bytes, serial;
    VkFormat format;
    VkExtent2D extent;
    uint32_t mapping_count;
    RcInterval mapping[VK_TEXTURE_COLOR_BLIT_MAPPINGS];
} VkTextureColorBlitSource;
int vk_texture_effect_color_blit(int id, VkTexImage *, view_t *, int *has_view, uint64_t *source_size,
                                VkTextureColorBlitSource *);
void vk_texture_storage_blit_written(const VkTextureColorBlitSource *, const VkImageBlit *);
// Read only from a current registration's already-selected committed pin. No resolver/upload/COW enters preflight.
 // Capture the resolved sampled instance in the active scene, without writable conversion or a CPU-source claim.
// -1: ordered preparation deferred, 0: unavailable/absent, 1: exact image/view metadata retained through completion.
// The caller must preserve this binding and verify shader-answer metadata before sealing its declaration.
int vk_texture_effect_sample(int id, int exact, VkTexImage *out, view_t *view, int *has_view, uint64_t *source_size);
// Texture lock held: whether a captured physical image still has a current owner for cached view creation.
// A false answer permits only consumer-owned deferred views; it does not revoke an existing image capture.
int vk_texture_image_live_locked(const VkTexImage *captured);

uint64_t vk_texture_same_regs(void);
void vk_texture_bucket_stats(char *out, size_t n);   // diagnosis for the pool log (vk_texture.c)
uint64_t vk_texture_slot_khash_locked(int id);       // vk_texture.c: descriptor hash of a GPU-written slot, 0 otherwise
uint32_t vk_texture_entry_gen(void);                 // vk_texture.c: bumped by every entry change   // registrations that changed nothing (vk_texture_register), for the pool log
// Slots whose token may have changed since *pos (0 at first; then the value these calls leave): fn is called for each
// (some more than once) and *pos advanced. 0 when too many changes came since: the caller looks at every slot. *pos
// from vk_texture_dirty_pos_locked after such a walk. Texture lock held.
// gfn, when given, gets each slot group of a bucket in place of fn for its members.
int vk_texture_dirty_slots_locked(uint64_t *pos, void (*fn)(int id, void *arg), void (*gfn)(int g, void *arg), void *arg);
// Source-proof changes only; no descriptor identity mutation. Zero requires invalidating all source receipts.
int vk_texture_source_dirty_slots_locked(uint64_t *pos, void (*fn)(int id, void *arg), void *arg);
int vk_texture_schedule_dirty_slots_locked(uint64_t *pos, void (*fn)(int id, void *arg), void *arg);
// Slot groups (vk_texture.c): the GPU-written slots of one address bucket with one descriptor hash kh, which resolve
// alike. head the first member, vk_texture_group_next_locked the next (-1: the end), n how many; stamp and fmask the
// texture pool's (vk_exec_draw.c pool_dirty_group: 1 when every member holds the same views, of the kinds fmask; a slot
// that joins goes to the tail, refilled on its own), set 0 by vk_texture_groups_unstamp_locked. Texture lock held.
typedef struct { uint64_t kh, incarnation; int32_t head, n; uint32_t stamp, fmask, association; } VkSlotGroup;
VkSlotGroup *vk_texture_group_locked(int g);
int vk_texture_group_member_locked(int group, uint64_t incarnation, int id);
int vk_texture_group_direct_locked(int group, uint64_t incarnation, int representative,
                                   VkImage expected, uint32_t *entry);
int vk_texture_group_next_locked(int id);
void vk_texture_groups_unstamp_locked(void);
uint64_t vk_texture_dirty_pos_locked(void);
// The bindless pool's entries (vk_texture.c): after a pool fill of slot id, notes the entry it resolved to (1; 0: none)
// and gives its number for vk_texture_pool_hold_locked in *ent (0: none); a submission's freshness check of every
// noted entry (returns how many). Texture lock held.
int vk_texture_pool_note_locked(int id, uint32_t *ent);
int vk_texture_pool_fresh_locked(void);
// A pool slot that held the entry old (0: none) now holds ent: the entries slots hold get the use stamps a lookup
// leaves (vk_texture.c touch) from the check by list. Texture lock held.
void vk_texture_pool_hold_locked(uint32_t old, uint32_t ent);
// A slot whose last lookup found no entry, looked up again: 1 it resolves now; 0 still none, its storage may become
// readable; -1 none until its token changes (a CPU slot whose texture was released after its registration, no storage).
int vk_texture_slot_retry_locked(int id);
// vk_memory.c, for the pool's freshness check: the generation table's lock, shared; with it held, 1 when every 2 MiB
// region of [addr, addr+size) has this epoch's write-watch snapshot and no page stamped after since (nothing in it can
// have changed), 0 when a closer look is due (vk_mem_changed_snap).
void vk_mem_gen_lock_shared(void);
void vk_mem_gen_unlock_shared(void);
int vk_mem_unchanged_locked(uint64_t addr, uint64_t size, uint32_t since, uint32_t epoch);
// The same check's reason when it fails: 1 a region without this epoch's snapshot yet, 2 one untracked, 3 one written
// since, 4 nothing to check (no size, beyond the regions); 0 unchanged.
int vk_mem_unchanged_why_locked(uint64_t addr, uint64_t size, uint32_t since, uint32_t epoch);
// The lock exclusively, for the snapshot of the regions of [addr, addr+size) that have none in epoch yet (the number of
// write-watch queries it took).
void vk_mem_gen_lock(void);
void vk_mem_gen_unlock(void);
uint64_t vk_mem_snap_range_locked(uint64_t addr, uint64_t size, uint32_t epoch);
// Dependency consumers use the same snapshot and generation tables. Observe while gen_lock is held; it does not
// take a new snapshot. A serial change requires checking which regions received a later memory observation.
void vk_mem_dependency_watch(void);
uint64_t vk_mem_dependency_serial(void);
int vk_mem_dependency_region_locked(uint32_t region, uint32_t epoch, uint32_t *version);
int vk_mem_dependency_sleep_locked(uint32_t region);
int vk_mem_dependency_adopt_locked(uint32_t region, uint32_t epoch); // established sleeping membership; complete event history
int vk_mem_dependency_events_locked(uint64_t *cursor, ResourceEventVisit visit, void *user);
// Lock-free preservation of an already established, same-submission tracked proof. Zero refuses reuse. This does
// not query write watch or prove tracking; it only checks publications since the captured observation watermark.
int vk_mem_dependency_range_unchanged(uint64_t addr, uint64_t size, uint64_t observed);
void vk_sampler_dlss_bias(float bias);           // texture LOD bias added for DLSS upscaling (material samplers)
void vk_sampler_aniso_live(void);                   // each present: NX2NAVI_ANISO after a settings change (tiled samplers)
uint32_t vk_sampler_gen(void);                      // changes whenever a sampler slot is registered
void vk_texture_gpu_written(const tex_t *t);        // rendered or copied into by the GPU: no re-upload from storage
// The live image of the given format and img's size at img's storage address, NULL when there is none. Texture lock held.
const VkTexImage *vk_texture_at_locked(const VkTexImage *img, VkFormat format);
// Rank and decode fallback pictures using issued integer storage descriptors on the base packed-color plane.
// This records writable intent, carries no proof of executed stores or initialization, and grants no projection.
void vk_texture_packed_storage_issued(VkImage image, uint64_t address);
void vk_texture_pinned_storage_picture_issued(const VkPinnedTextureWrite *, VkImage);
void vk_texture_write_target_picture_issued(const VkTextureWriteTarget *, VkImage);
int vk_texture_storage_target_picture_written(const VkTextureStorageTarget *, VkTextureAliasRead *,
    const VkImageSubresourceRange *, int preserving, VkImage);
int vk_texture_packed_storage_rank_available_locked(void);
// The query takes a live texture entry under the texture lock and checks its exact recording disposition.
RcRegistryPosition vk_texture_packed_storage_output_locked(const VkTexImage *image, RcRegistryPosition cut);
void vk_texture_storage_written(VkImage image, uint64_t addr); // successful command emission only
int vk_texture_render_target_written(VkImage image, uint64_t addr);
int vk_texture_render_target_load_current(VkImage, uint64_t, const VkImageSubresourceRange *);
int vk_texture_render_target_loaded_written(VkImage, uint64_t, VkTextureAliasRead *seed, const VkImageSubresourceRange *);
// Copy the selected image's existing range owners and native extents. The caller owns the extent vector.
// Typed reuse still depends on live registry proof; never allocate or rebind a hold here.
int vk_texture_image_binding(VkImage image, uint64_t addr, uint64_t size, RcRegistryImageBinding *out,
                             RcRegistryImageWriteExtent **extents, uint32_t *count);
// Capture the full storage range of this exact live physical owner and its recording lifetime. The optional
// persistent hold is broad range ownership, not a claim that guest CPU contents are current.
int vk_texture_image_effect(VkImage image, uint64_t addr, RcSourceRange *range, RcRegistryImageBinding *owner);
void vk_texture_storage_bound_locked(const VkTexImage *img);   // bound as a storage image in the recording submission
uint64_t vk_texture_last_write(const tex_t *t);    // timeline value of the newest submission recorded to write it; 0: none
size_t vk_texture_texel_span(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint64_t row, uint64_t img);   // bytes ReadTexels spans; 0 for a bad region
int vk_texture_small_dirty(void);                   // a small mirrored target holds GPU texels its storage lacks
void vk_texture_write_back_small(uint64_t fence);   // writes those rendered by work up to fence back to storage
void vk_texture_debug(const VkTexImage *img, uint32_t *uploads, int *last_change, int *pages_written);   // vk_dump
int vk_texture_slot_swz(int id, uint8_t swz[4]);    // vk_dump: swizzle of the descriptor in slot id
int vk_texture_read_image(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row,
                          uint64_t img);            // always from the image (GPU copy, waited for); 0 on failure
int vk_texture_cpu_stale(uint64_t addr, uint64_t size);    // a GPU-written mirrored texture there has stale storage
uint64_t vk_texture_dirty_fence(uint64_t addr, uint64_t size);   // newest recorded writer of such a texture; 0: none
void vk_texture_write_back(uint64_t addr, uint64_t size);   // writes those images back into their storage (waits)
void vk_texture_range_gone(uint64_t lo, uint64_t size, VmaAllocation alloc);   // vk_pool_fini: drop the images there
void vk_texture_stats(uint64_t *images, uint64_t *uploads, uint64_t *reuploads);
// Entries now, live and released (TextureFinalize, kept until sweep_released), with the bytes of their own memory
// (images not bound into pool memory).
typedef struct { uint64_t live, released, own_live, own_released, bytes_live, bytes_released; } VkTexCensus;
void vk_texture_census(VkTexCensus *s);
void vk_texture_census_log(uint64_t frame);         // vk_memory.c vram_log: the live entries' memory by kind and format
unsigned vk_texture_reclaim_heap(uint32_t domain, uint64_t bytes); // RcBudgetReclaim status for optional image giveback
int vk_texture_commit_low(void);                    // the system's commit was low at the last check (vk_texture.c)
// vk_mem_vram's device-local use less the free space inside VMA's blocks (vk_memory.c), for the evictions
int vk_mem_vram_live(uint64_t *used, uint64_t *budget);
// Pressure request is lock-free. Drain only at a replay boundary with no active recording groups, before new work.
void vk_exec_texture_publications_request_reclaim(void);
void vk_exec_texture_publications_reclaim(uint64_t incoming_fence);
void vk_exec_texture_publications_clear(void);      // idle renderer teardown, before texture shutdown
void vk_texture_shutdown(void);                     // vk_mem_shutdown calls it
void vk_exec_resource_handoff(uint64_t fence); // close ordered CPU reads before a replay prefix can execute
void vk_exec_resource_handoff_before(uint64_t fence, uint64_t keep_open_ordinal);
void vk_texture_new_submit(void);                   // a queue submit starts: draw bindings check storage again
uint32_t vk_texture_epoch(void);                    // the write-watch snapshot epoch vk_texture_new_submit started
VkSampler vk_sampler_by_id(int id);                 // VK_NULL_HANDLE for an empty slot
VkSampler vk_sampler_for(const samp_par *s);        // cached by content
int vk_sampler_params(int id, samp_par *out);       // parameters of a sampler slot; 0 for an empty slot

// 8-bit RGBA PNG with stored deflate blocks (boot.c); stops with rt_fatal when the file cannot be written.
void rt_write_png(const char *path, const uint8_t *rgba, uint32_t w, uint32_t h);

// ---- submission, synchronisation, completion (vk_sync.c, vk_backend.c)
// Fence values are ctx->timeline values: each submission signals the next one, in submit order. A value counts as done
// once the GPU reached it and the completion thread ran its side effects. Guest or host threads.
void vk_sync_start(void);                  // vk_init (fatal without a device) and the completion thread; idempotent
uint64_t vk_sync_submit(const uint64_t *handles, int n);   // records with vk_exec_record, submits; returns the value
uint64_t vk_sync_submit_capture(const uint64_t *handles, int n, VkUsnapCapture *capture); // always consumes capture
uint64_t vk_sync_submit_bundle(const uint64_t *handles, int n, VkReplayCaptureBundle bundle); // always consumes both
// The probes (--probe-vkmem's table parts, the lost-device probe): one submission whose commands the caller records
// itself where the replay would, with the submission lock, the memory prologue and the fence of any other.
uint64_t vk_sync_submit_own(void (*record)(VkCommandBuffer, void *), void *arg);
uint64_t vk_sync_reserve(void);            // asynchronous replay: the value the next queued submission signals
int vk_sync_fence_wait(uint64_t value, uint64_t timeout_ns);   // 1 when done; 0 on timeout or a value never submitted
// The executed timeline (vk_sync.c): a submission's own commands are through the GPU, its write-backs maybe not yet.
int vk_sync_exec_wait(uint64_t value, uint64_t timeout_ns);
int vk_sync_wb_split(void);                 // NX2NAVI_WB_SPLIT: the write-backs go into a part of their own
// NX2NAVI_TRACE=perf (vk_sync.c "the executed timeline"): what a wait on that timeline left behind since the last call,
// whether the completion thread's callback running now belongs to a value whose waiter went on already, and the
// completion thread's time by callback.
void vk_sync_exec_stats(uint64_t *waits, uint64_t *early, uint64_t *lag_ns, uint64_t *lag_max_ns);
int vk_sync_effect_late(void);
uint64_t vk_sync_cb_value(void);            // the value whose side effect the calling thread is running; 0: not in one
void vk_sync_cb_stats(char *out, size_t n);
void vk_exec_late_stats(uint64_t *events, uint64_t *counters, uint64_t *counts);   // vk_exec_xfer.c: such callbacks since the last call
VkCommandBuffer vk_sync_split_exec(void);   // vk_sync_split for the part before a submission's write-backs
int vk_mem_mirror_wb_pending(void);         // vk_memory.c: vk_mem_mirror_end has write-backs to record
uint64_t vk_sync_fence_done(void);         // highest done value
uint64_t vk_sync_issued(void);             // newest submitted value (the one being recorded is this + 1)
void vk_sync_finish(void);                 // every submitted value done
void vk_sync_shutdown(void);               // finish, stop the completion thread, free the command buffers
// The op-stream replay (vk_exec.c): records the command handles of one submission into cb, which the caller has
// begun and ends. fence is the value this submission will signal. Returns the number of commands recorded into the
// command buffer it holds at the end (0: that one stays empty and is not submitted), or STOPPED when the tail
// must be abandoned. A weak vk_sync.c version accepts empty streams only.
enum { VK_EXEC_RECORD_STOPPED = -1 }; // possible handed prefix, no final scene handoff
int vk_exec_recording_failed(void);  // active native tail cannot continue
int vk_exec_recording_missing_resource(void); // required resource failure rather than normal cancellation
typedef struct VkExecReadiness VkExecReadiness;
// A synchronous readiness plan and capture are valid under the same uninterrupted submission lock.
int vk_exec_record(VkCommandBuffer cb, const uint64_t *handles, int n, uint64_t fence,
                   VkReplayCaptureBundle *bundle, const VkExecReadiness *readiness);
VkResult vk_exec_snapshot_preflight(const uint64_t *handles, int n, VkUsnapCapture **capture,
                                    VkExecReadiness **readiness, VkReplayCaptureWait *wait);
void vk_exec_readiness_discard(VkExecReadiness *readiness);
// Inside vk_exec_record, outside a render pass: ends the command buffer recorded so far, submits it (after the upload
// batch) without a signal, and returns a new begun one for the rest of the submission. The submission's timeline value
// is signalled by its last part, whose signal covers every earlier part in submission order.
VkCommandBuffer vk_sync_split(void);
struct VkExec;
struct RcVmaDependency;
typedef enum {
    VK_SYNC_PROGRESS_UNAVAILABLE,
    VK_SYNC_PROGRESS_COMPLETE,
    VK_SYNC_PROGRESS_STOPPED,
    VK_SYNC_PROGRESS_STOPPED_AFTER_CUT,
} VkSyncProgressResult;
// The replay owner can park outside provider locks without letting another recorder overtake its scene.
// Only an actual successful native final-fence handoff qualifies as an older scene supplier.
uint64_t vk_sync_handed_fence(void);
// Recorder owns an empty current part. Confirm all earlier native scene parts are already done.
int vk_sync_mirror_publication_safe(uint64_t scene);
VkSyncProgressResult vk_sync_progress_wait_prior(uint64_t issued_scene);
VkSyncProgressResult vk_sync_progress_wait_capacity(const struct RcVmaDependency *,
                                                   int (*compatible_spare_ready)(void *), void *context);
// After draining replay workers, prepare the selected pipeline outside the recorder mutex. The callback returns
// nonzero once it owns a result (ready or terminal refusal), or zero on stop. The current scene stays exclusive.
VkSyncProgressResult vk_sync_progress_prepare_pipeline(int (*prepare)(void *), void *context);
VkSyncProgressResult vk_sync_progress_wait_upload_staging(uint64_t handed_upload);
// After draining workers and closing the pass, cut only for eligible prefix scratch. A nonempty cut returns
// the begun unhanded tail on COMPLETE or STOPPED_AFTER_CUT. An empty tail keeps its open command buffer and
// waits on a private queue marker only for a fresh, handed ordinary part from this scene.
VkSyncProgressResult vk_sync_progress_split(struct VkExec *, VkCommandBuffer *resume, int empty_tail);
// Submit a nonempty native part while keeping the active guest read ordinal open.
VkSyncProgressResult vk_sync_progress_split_before(struct VkExec *, VkCommandBuffer *resume,
                                                  uint64_t keep_open_ordinal);
// Inside vk_exec_record, for the replay workers (vk_exec.c): vk_sync_append_later ends the command buffer recorded so
// far, puts it into the part followed by n places for command buffers the workers record (*at the first), and returns
// a new begun one for the rest of the part; vk_sync_append_fill puts the workers' command buffers into their places,
// before the part ends; vk_sync_append_room says whether the part has room for n more. A worker claims and begins a
// command buffer of its own pool for the submission being recorded with vk_sync_worker_begin(w) and ends it with
// vk_sync_worker_end, on its own thread.
int vk_sync_append_room(int n);
VkCommandBuffer vk_sync_append_later(int n, int *at);
void vk_sync_append_fill(int at, const VkCommandBuffer *cbs, int n);
VkCommandBuffer vk_sync_worker_begin(int w);
void vk_sync_worker_end(VkCommandBuffer cb);
extern int vk_sub_in_frame;   // replay: the submission being recorded is the frame's n-th (0 the first; vk_backend.c)
// Runs fn(arg) on the completion thread once fence is done on the GPU (event writes, counter reports, sync state).
// Callbacks run in the order they were added, after those of earlier values; a value already done runs at once there.
void vk_completion_add(uint64_t fence, void (*fn)(void *), void *arg);
// An owned callback also disposes abandoned work. cancel(arg, 0) means the scene never reached the queue;
// cancel(arg, 1) is allowed only after all possibly submitted GPU users were drained. It must not publish the
// execution side effects of commands that may not have run. Submitted users are retained on an undrained loss.
void vk_completion_add_owned(uint64_t fence, void (*fn)(void *), void (*cancel)(void *, int), void *arg);
// The same for a side effect the game's CPU reads (an occlusion count, a counter report, an event value): run before
// the others, and by the wait after a frame's first submission itself (vk_sync.c "the executed timeline").
void vk_completion_add_result(uint64_t fence, void (*fn)(void *), void *arg);
void vk_completion_add_result_owned(uint64_t fence, void (*fn)(void *), void (*cancel)(void *, int), void *arg);
// gfx_vulkan_backend (vk_backend.c): presents go to a present thread and run after the submissions made before them.
void rdoc_init(void);                     // runtime/rdoc.c: NX2NAVI_RDOC debug capture; before vkCreateInstance
int rdoc_active(void);
void rdoc_after_present(uint64_t frame);
void vk_present_drain(void);               // returns once every queued present was submitted to gfx_queue
int vk_replay_async(void);                 // NX2NAVI_REPLAY_ASYNC (default 0): submissions replay on the replay thread
void vk_program_release(const prog_t *p);  // GfxBackend.program_release: vk_pipeline.c; weak no-op in vk_backend.c
void vk_backend_shutdown(void);            // drains replay and completion before destroying snapshots and memory

// ---- pipelines (vk_pipeline.c)
// Binding model (comment block at the top of vk_pipeline.c): one descriptor set, set 0, per pipeline. Binding 0 is
// the translator's support buffer (a uniform buffer every stage shares); the vertex stage keeps its own bindings, the fragment
// stage's follow them (its SPIR-V is renumbered). Set 0 is a push-descriptor set unless the pipeline needs more than
// maxPushDescriptors (32 here) descriptors; VkPipeInfo.bindings says what goes where. No push constants.
// Everything a graphics pipeline depends on, as the op stream carries it. Fields covered by dynamic state are read by
// vk_pipeline_set_dynamic only and left out of the pipeline key.
enum {
    VK_PIPE_RMW_NONE,
    VK_PIPE_RMW_MESH_WORD,
    VK_PIPE_RMW_FRAGMENT_BITMAP,
    VK_PIPE_RMW_MESH_BITMAP,
};
typedef struct VkPipeState {
    const GfxShader *vs, *fs;              // gfx_shader_get results; fs NULL: no fragment stage (depth only). vs is
                                           // the mesh stage of a mesh draw (gfx_shader_stage GFX_STAGE_MS)
    const GfxShader *ts;                   // mesh draws: the task stage, or NULL
    int32_t color_fmt[8];                  // NVNformat of colour target i (OP_SET_RT), 0 none
    int32_t depth_fmt;                     // NVNformat of the depth/stencil target, 0 none
    uint8_t samples;                       // sample count of the targets; 0 or 1 single-sampled
    uint8_t prim;                          // NVN_DRAW_PRIMITIVE_* of the draw
    uint8_t depth_minus_w;                 // nvn_depth_minus_w(): clip-space depth is -w..w
    uint8_t depth_clamp, raster_discard;   // OP_DEPTH_CLAMP, OP_RASTER_DISCARD
    uint8_t y_flip;                        // the replay's viewports have negative heights (upper-left origin): the
                                           // front face is not mirrored
    uint8_t prim_restart;                  // the draw's indices restart its strip or fan at the all-ones value
                                           // (OP_PRIM_RESTART, set by op_draw_idx); dynamic state only
    uint8_t feedback;                      // the fragment shader samples an attachment of the pass in the feedback
                                           // layout: 1 colour, 2 depth/stencil (VK_PIPELINE_CREATE_*_FEEDBACK_LOOP_BIT_EXT)
    uint8_t native_rmw_marker;             // VK_PIPE_RMW_* explicit command-owned atomic marker variant
    blend_st blend[8];                    // OP_BIND_STATE kind 0, indexed by blend_st.target
    color_st color;                        // kinds 1-5
    chmask_st chmask;
    ms_st ms;
    poly_st poly;
    ds_st ds;
    uint32_t vattr[16];                    // OP_BIND_VATTR: va_st words; attribute i feeds vertex shader Location i
    vs_st vstream[16];                     // OP_BIND_VSTREAM
} VkPipeState;
void vk_pipe_state_defaults(VkPipeState *s);   // NVN defaults of every state object (shaders and formats left 0)
enum { VK_PIPE_SUPPORT = GFX_RES_KINDS, VK_PIPE_NATIVE_RMW_MARKER };
#define VK_PIPE_ZERO_BINDING 16                // vertex binding of shader inputs without an attribute: bind a zeroed
                                               // buffer of at least 16 bytes there (stride 0)
typedef struct {
    uint16_t binding;                      // in its set
    uint8_t stage;                         // GFX_STAGE_* that declared it (support buffer: the first stage)
    uint8_t kind;                          // GFX_RES_* or a VK_PIPE_* synthetic descriptor role
    VkDescriptorType type;
    uint16_t count;
    VkShaderStageFlags stages;
    const GfxShaderRes *res;               // stage reflection; NULL for synthetic descriptors
    uint8_t set;                           // descriptor set: 0, or 1 for the fragment stage with independent layouts
} VkPipeBinding;
typedef struct {
    VkPipeline pipeline;                   // VK_NULL_HANDLE when creation failed
    VkPipelineLayout layout;
    VkDescriptorSetLayout set_layout;      // set 0
    int push;                              // 1: write set 0 with vkCmdPushDescriptorSetKHR; 0: allocate and bind a set
    VkPipelineBindPoint bind_point;
    uint32_t n_bindings;
    const VkPipeBinding *bindings;
    uint32_t vs_inputs;                    // Locations the vertex shader reads
    // Independent layouts (vk_caps.gpl): set 0 holds the vertex stage's bindings, set 1 the fragment stage's; bindings
    // are ordered by set. One of the two sets at most is a push set. n_sets 1: everything in set 0 (NX2NAVI_GPL=0, compute).
    VkDescriptorSetLayout set_layout1;
    int push1;
    uint32_t n_sets;
    // Descriptor update template per set (vk_pipeline.c make_tpl): reads one VkDescSlot per descriptor, in binding order,
    // from the start of the set's descriptors. VK_NULL_HANDLE: the set is written with VkWriteDescriptorSet structs.
    VkDescriptorUpdateTemplate tpl[2];
    int pool;                              // a stage indexes the texture or sampler pool: bind set 2 (the pools)
    // Owned compute source whose bytes exactly match the successfully created native shader module.
    // A definite-effect analysis of any other source cannot describe this pipeline.
    const GfxShader *compute_effect_shader;
    RcNativeImageSeedBindings native_image_seed_bindings;
    RcNativeImageStore native_image_store; // certified against the final module's selected compute entry point
    uint64_t compute_query_size_lod0_mask;  // direct set 0 bindings attested by this native compute pipeline
    const GfxShader *compute_pool_domain_shader; // source index proof remains valid after the final table remap
    uint32_t compute_pool_query_shape_kinds; // all uses of the selected table binding are LOD0 queries
    int compute_pool_marks; // the actual final module uses the renderer marker binding
    // Attestation from the actual final native module. A nonzero binding requires the command's
    // zero-initialized storage marker at the selected stage and set. The mesh mode records one word.
    RcNativeRmwWord native_rmw;
    uint32_t native_rmw_binding;
    RcNativeImageSeedBindings native_image_rmw_seed_bindings;
} VkPipeInfo;
// One descriptor of the packed array a descriptor update template reads (VkPipeInfo.tpl), 24 bytes for every type; the
// bytes a type leaves unused are zero, so equal descriptors compare equal byte for byte.
typedef union { VkDescriptorImageInfo image; VkDescriptorBufferInfo buffer; VkBufferView view; uint64_t w[3]; } VkDescSlot;
// Both return the cached pipeline (created on first use; failures are cached and logged once). info: may be NULL.
// Valid until vk_pipeline_shutdown; thread-safe.
VkPipeline vk_pipeline_graphics(const VkPipeState *s, const VkPipeInfo **info);
// Only considers a previously selected writable fragment storage binding after its ordinary import is blocked.
int vk_pipeline_fragment_rmw_candidate(const VkPipeState *, const VkPipeBinding *);
int vk_pipeline_mesh_rmw_bitmap_candidate(const VkPipeState *, const VkPipeBinding *);
// The replay's variant: after vk_pipeline_async_enable a missing pipeline is built by background workers and this
// returns VK_NULL_HANDLE with *pending 1 until it is ready (the draw is skipped); otherwise as vk_pipeline_graphics.
// wait_ms > 0 waits up to that long for a pending pipeline first. Pipelines built in a run are recorded to
// pipeline-keys.bin next to the pipeline cache and prebuilt at the next boot (NX2NAVI_PIPE_PREBUILD=0: not).
VkPipeline vk_pipeline_graphics_async(const VkPipeState *s, const VkPipeInfo **info, int *pending, uint32_t wait_ms);
typedef enum {
    VK_GRAPHICS_READY, VK_GRAPHICS_PENDING, VK_GRAPHICS_STOPPED,
    VK_GRAPHICS_UNSUPPORTED, VK_GRAPHICS_TERMINAL,
} VkGraphicsPipelineResult;
typedef struct VkGraphicsPipelineRevision VkGraphicsPipelineRevision;
typedef struct VkGraphicsPipelineWait VkGraphicsPipelineWait;
// Only the actual selected vertex module can certify an ordinary exclusive uniform binding.
// A bounded result that does not fit these limits leaves the ordinary full input path unchanged.
enum { VK_GRAPHICS_UNIFORM_BINDINGS = 8, VK_GRAPHICS_UNIFORM_RANGES = 16 };
typedef struct { uint32_t offset, bytes; } VkGraphicsUniformRange;
typedef struct { uint16_t binding; uint8_t first, count; } VkGraphicsUniformBinding;
typedef struct {
    uint32_t count;
    VkGraphicsUniformBinding bindings[VK_GRAPHICS_UNIFORM_BINDINGS];
    VkGraphicsUniformRange ranges[VK_GRAPHICS_UNIFORM_RANGES];
} VkGraphicsUniformFootprints;

// Binding 0 is reserved for support, so the fragment query mask can carry this revision's pool proof.
#define VK_GRAPHICS_INACTIVE_POOL_ATTESTED UINT64_C(1)
typedef struct {
    const VkPipeInfo *info;
    VkPipeline pipeline;
    VkGraphicsPipelineRevision *revision;
    uint64_t vs_query_size_lod0_mask;       // set 0 bindings certified by this native pipeline revision
    uint64_t fs_query_size_lod0_mask;       // final fragment layout bindings certified by the same revision
    const VkGraphicsUniformFootprints *vertex_uniforms; // borrowed from this exact retained revision
} VkGraphicsPipelineUse;
typedef struct {
    VkResult error;
    const char *reason;
    RcAdmissionStatus admission;
    RcAdmissionVector peak;
    RcBudgetReadiness before;
} VkGraphicsPipelineFailure;
// Zero-start prepare is a lookup only. The parked producer starts/waits for the
// exact state and owns *wait until Ready, terminal failure or explicit cancel.
VkGraphicsPipelineResult vk_pipeline_graphics_prepare(
    const VkPipeState *state, int allow_start, VkGraphicsPipelineWait **wait,
    VkGraphicsPipelineUse *out, VkGraphicsPipelineFailure *failure);
int vk_pipeline_graphics_wait(VkGraphicsPipelineWait *wait);
void vk_pipeline_graphics_cancel(VkGraphicsPipelineWait **wait);
void vk_pipeline_graphics_release(VkGraphicsPipelineUse *use);
// Move the exact selected revision into the actual scene before native bind.
void vk_pipeline_graphics_record(VkGraphicsPipelineUse *use, uint64_t scene);
// How long the pipeline of s has been building since a draw first missed it, in milliseconds; 0 when it is unknown,
// usable, or no draw has missed it.
double vk_pipeline_building_ms(const VkPipeState *s);
// The stage libraries of a draw the replay has not reached yet, queued to the workers when no pipeline is known for its
// state (vk_pipeline.c "stage libraries ahead of their draws"). Returns the libraries queued.
int vk_pipeline_hint(const VkPipeState *s);
// rt_boot calls it unless --shot is given; NX2NAVI_PIPE_SYNC=1 makes it a no-op. Of the probes only --probe-prewarm
// calls it.
void vk_pipeline_async_enable(void);
int vk_pipeline_async_on(void);   // 1 after vk_pipeline_async_enable: graphics shaders are translated in the background too
// Boot prewarm, before the guest runs (rt_boot): loads the shaders pipeline-keys.bin names, translating the ones not
// cached under the current translator from their stored inputs, and builds every recorded pipeline, on all cores.
// No-op without vk_pipeline_async_enable, with NX2NAVI_PIPE_PREBUILD=0 or with NX2NAVI_GPU=null.
void vk_pipeline_prewarm(void);
// Ends the boot prebuild's crash marker (vk_pipeline.c, "boot prebuild crash marker"): boot_wait calls it when it lets
// the game start while the prewarm goes on, so a crash in play is not taken for one in the boot's builds.
void vk_pipeline_prewarm_marker_clear(void);
typedef struct {
    int phase;                            // 0 starting (device, key file), 1 shaders, 2 pipelines, 3 done
    int64_t sh_done, sh_total, sh_translated;   // gfx_shader_prewarm_progress
    int64_t pipe_done, pipe_total;        // pipelines built or failed of those queued
} VkPrewarmProgress;
void vk_pipeline_prewarm_progress(VkPrewarmProgress *p);   // any thread, while vk_pipeline_prewarm runs or after
VkPipeline vk_pipeline_compute(const GfxShader *cs, const VkPipeInfo **info);
typedef enum {
    VK_COMPUTE_READY, VK_COMPUTE_PENDING, VK_COMPUTE_STOPPED, VK_COMPUTE_TERMINAL,
} VkComputePipelineResult;
typedef enum {
    VK_COMPUTE_BUILD, VK_COMPUTE_FLIGHT, VK_COMPUTE_CAPACITY,
} VkComputePipelineWaitKind;
typedef struct {
    const GfxShader *shader;
    RcAdmissionVector peak;
    RcBudgetReadiness before;
    RcAdmissionStatus admission;
    VkComputePipelineWaitKind kind;
    VkResult error;
    const char *failure;
} VkComputePipelinePending;
// Probe only with allow_create=0. Creating or waiting requires the serialized recorder to be parked.
// A wait receipt names the exact immutable shader and can be retried without retaining a flight pointer.
VkComputePipelineResult vk_pipeline_compute_prepare(const GfxShader *shader, int allow_create,
                                                      VkPipeline *pipeline, const VkPipeInfo **info,
                                                      VkComputePipelinePending *pending);
int vk_pipeline_compute_wait(const VkComputePipelinePending *pending);
// Records every dynamic state that VkPipeState determines, after binding a graphics pipeline: cull mode, front face,
// topology, depth and stencil tests and ops, depth bias enable, rasterizer discard, vertex input, and, where the GPU
// has them dynamic, depth clamp, polygon mode, blend enables, equations, write masks and alpha to coverage. The replay
// sets the rest: vkCmdSetViewportWithCount, vkCmdSetScissorWithCount, depth bias, blend constants, stencil compare and
// write masks and references; vertex buffers with vkCmdBindVertexBuffers2 (strides NULL when
// vk_pipeline_vertex_input_dynamic(), else from VkPipeState.vstream).
void vk_pipeline_set_dynamic(VkCommandBuffer cb, const VkPipeState *s);
// Vertex input as vkCmdSetVertexInputEXT takes it (vk_pipeline.c vertex_input).
typedef struct {
    uint32_t n_attr, n_bind;
    VkVertexInputAttributeDescription2EXT a[32];
    VkVertexInputBindingDescription2EXT b[17];
} VkVtxInput;
// The dynamic state vk_pipeline_set_dynamic_delta last recorded into a command buffer, as the values its calls took.
// have: the groups that still hold what the command buffer has. Zeroed, nothing is known and the next call records
// everything; the owner zeroes it whenever the command buffer begins again or other code records pipelines or dynamic
// state into it (the replay keeps it in vk_exec_draw.c's B, which vk_exec_draw_new_cb clears).
enum { VK_DYN_CORE = 1, VK_DYN_VTX = 2, VK_DYN_BLEND = 4 };   // VTX: topology, primitive restart, vertex input
typedef struct {
    uint32_t have;
    VkCullModeFlags cull;
    VkFrontFace front;
    VkPrimitiveTopology topo;
    VkBool32 dt, dw, st, bias_en, discard, restart, clamp, a2c;
    VkCompareOp dop, scmp[2];
    VkStencilOp sop[2][3];
    VkPolygonMode poly;
    uint32_t nt;                            // colour targets the blend arrays cover
    VkBool32 en[8];
    VkColorBlendEquationEXT eq[8];
    VkColorComponentFlags wm[8];
    VkVtxInput vi;
} VkDynShadow;
// vk_pipeline_set_dynamic recording only what differs from *sh, which it then updates. mesh: the bound pipeline is a
// mesh pipeline, which keeps topology, primitive restart and vertex input static, so binding it dropped them: they are
// left out and marked unknown. Returns the number of state calls recorded and adds the number left out to *skipped.
int vk_pipeline_set_dynamic_delta(VkCommandBuffer cb, const VkPipeState *s, VkDynShadow *sh, int mesh, uint64_t *skipped);
int vk_pipeline_vertex_input_dynamic(void);    // 1: vertex input is VK_EXT_vertex_input_dynamic_state
// Translator answers from the bound state: attr_type from the vertex attribute formats (vs), frag_out_type from the
// colour target formats (fs). Other fields are left as they are; either pointer may be NULL.
void vk_pipeline_answers(const VkPipeState *s, GfxShaderAnswers *vs, GfxShaderAnswers *fs);
int vk_pipeline_tex_type(int nvn_target);      // NVN_TEXTURE_TARGET_* -> GFX_TEX_* (GfxShaderAnswers.tex_type)
uint32_t vk_pipeline_target_wmask(const VkPipeState *s, int i);   // colour write mask of target i the pipeline uses
VkPrimitiveTopology vk_pipeline_topology(int nvn_prim);   // quads -> triangle list, line loop -> strip: the replay
                                                          // rewrites those draws; polygon -> fan
int vk_format_shader_numeric_type(VkFormat format);      // GFX_TYPE_* for a supported renderer format
VkFormat vk_vertex_format(int nvn_format, int *type);     // attribute format (supported fallback), GFX_TYPE_* in *type
PFN_vkCmdPushDescriptorSetKHR vk_pipeline_push_fn(void);  // after the first vk_pipeline_* call
// Descriptor update templates (after the first vk_pipeline_* call): NULL when templates are off (NX2NAVI_DESC_TEMPLATE=0,
// a device below Vulkan 1.1) or, for the push variant, vkCmdPushDescriptorSetWithTemplateKHR is missing.
PFN_vkCmdPushDescriptorSetWithTemplateKHR vk_pipeline_push_tpl_fn(void);
PFN_vkUpdateDescriptorSetWithTemplate vk_pipeline_update_tpl_fn(void);
// Probe only: 1 selects stage-independent layouts for pipelines looked up from now on, 0 the shared set 0; returns
// the previous mode. 1 needs vk_caps.gpl (the independent-sets flag comes with the extension).
int vk_pipeline_set_indep(int on);
typedef struct { uint64_t graphics, compute, failures, hits, layouts, push_layouts, async_built, pending_skips; } VkPipeStats;
void vk_pipeline_stats(VkPipeStats *s);
void vk_pipeline_save(void);               // writes the VkPipelineCache to pipeline-cache.bin next to nx2navi.exe
void vk_pipeline_save_exit(void);          // vk_pipeline_save as the process ends: new builds wait while it waits for
                                           // the running ones
void vk_pipeline_autosave(void);           // vk_pipeline_save at most every 20 s, when pipelines were built since
void vk_pipeline_stop_preparation(void);   // cancel capacity-blocked jobs before joining replay/background workers
void vk_pipeline_shutdown(void);           // waits for background builds, saves, destroys pipelines, layouts and the
                                           // cache; before vk_shutdown
int vk_pipeline_selftest(void);            // one graphics and one compute pipeline from census shaders; 0 on success
// Pipelines a library creates on our device without a pipeline cache of its own (FSR 3.1, vk_ffx.cpp), through the
// pipeline caches kept across runs; as vkCreateComputePipelines (compute 1) or vkCreateGraphicsPipelines with no cache.
VkResult vk_pipeline_create_foreign(VkDevice dev, int compute, uint32_t n, const void *infos,
                                    const VkAllocationCallbacks *alloc, VkPipeline *out);
// How many it made so far: from_cache came without a compile and compiled went into the play cache (both only with a
// play cache, which is what can tell them apart); uncached were created with no cache of ours.
typedef struct { uint64_t made, from_cache, compiled, uncached; } VkForeignPipes;
void vk_pipeline_foreign_counts(VkForeignPipes *o);

#include "vk_rec.h"   // last: its macros route the vkCmd* calls of every file that includes vk.h

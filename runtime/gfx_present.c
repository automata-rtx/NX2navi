// Vulkan surface and swapchain, present blit, 60 Hz vsync thread, focus: the window-system-free half of the window
// (the other half: gfx_window_win.c with Win32, gfx_window_sdl.c with SDL, through gfx_wsys.h).
// Contains code Copyright (c) 2026 Peiikai, Copyright (c) 2026 MacNativeGames contributors (MIT; see
// THIRD_PARTY_NOTICES.md).
//
// Threads: the window thread (the back end's) owns the window and never runs guest code. The vsync thread signals the
// bound system events. Presents come from any host thread (the back end's present worker) and take g_pl; every queue
// submission takes the recursive queue lock (vk_queue_lock). The swapchain is rebuilt lazily: a resize only marks it
// dirty (gfx_window_mark_dirty), the next present (or gfx_window_resize) rebuilds it.
//
// The in-game menu (ui/menu.h) draws over the game image in the present's command buffer, after the blit. The window's
// size and fullscreen state are the settings NX2NAVI_WINDOW_SIZE and NX2NAVI_FULLSCREEN, which the back end applies at start
// and whenever a cfg_set changes them (window_settings_live -> gfx_window_apply_settings).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audren.h"
#include "cfg.h"
#include "coop.h"
#include "flight.h"
#include "fs.h"
#include "gfx.h"
#include "gfx_wsys.h"
#include "nvn_api.h"
#include "ov.h"
#include "perfmon.h"
#include "platform/platform.h"
#include "ui/menu.h"
#include "vk.h"
#include "vk_fg.h"
#include "vk_dlaa.h"
#include "vk_dlssg.h"
#include "resource_core_vma.h"

#define FRAMES 2
#define MAX_IMAGES 16
#define NG_MAX 5   // generated frames per rendered one at most (DLSS Frame Generation through NGX, 6x)
#define VSYNC_NS 1000000000ull
#define VSYNC_HZ 60

// ---- queue lock
// Taking it hands the replay's held submissions to the queue first (vk_sync.c, submit batching), so whatever the
// holder submits or waits for comes after them in queue order.
static PlatRMutex g_ql;
void vk_queue_lock(void) {
    plat_rmutex_lock(&g_ql);
    vk_sync_flush_held();
}
void vk_queue_lock_raw(void) { plat_rmutex_lock(&g_ql); }
int vk_queue_trylock_raw(void) { return plat_rmutex_trylock(&g_ql); }
void vk_queue_unlock(void) { plat_rmutex_unlock(&g_ql); }

// The display queue has no game submissions. Shared-queue devices retain the original lock.
static PlatRMutex g_pql;
void vk_present_queue_lock(void) {
    if (vk_ctx()->present_queue != vk_ctx()->gfx_queue) plat_rmutex_lock(&g_pql);
    else vk_queue_lock();
}
void vk_present_queue_unlock(void) {
    if (vk_ctx()->present_queue != vk_ctx()->gfx_queue) plat_rmutex_unlock(&g_pql);
    else vk_queue_unlock();
}
VkResult vk_device_wait_idle(void) {
    vk_queue_lock();
    vk_present_queue_lock();
    PFN_vkGetDeviceProcAddr get = vk_dlssg_dpa();
    PFN_vkDeviceWaitIdle wait = get ? (PFN_vkDeviceWaitIdle)get(vk_ctx()->device, "vkDeviceWaitIdle") : NULL;
    VkResult r = wait ? wait(vk_ctx()->device) : vkDeviceWaitIdle(vk_ctx()->device);
    vk_present_queue_unlock();
    vk_queue_unlock();
    return r;
}

// ---- window state shared with the back end
static VkSurfaceKHR g_surface;
static volatile int32_t g_dirty;               // the window was resized; swapchain stale
static volatile int32_t g_restart;             // gfx_window_request_restart: closing the window starts the game again
static volatile int32_t g_focused = 1;
static volatile int32_t g_focus_msgs;
static PlatRMutex g_start_lock;       // gfx_window_start / vsync start
static int g_started;

void gfx_window_mark_dirty(void) { plat_xchg32(&g_dirty, 1); }

// The window gained or lost the focus: the game hears it as nn::oe's focus state and a notification message. The line
// carries the local wall time, the only one in the log: when a GPU hang takes the whole system down, the log ends
// without an error, and this time set against the driver's entry in the Windows event log (nvlddmkm 153, Display 4101)
// tells whether an alt-tab or an overlay came before the hang or followed it.
void gfx_window_focus_changed(int in) {
    plat_xchg32(&g_focused, in);
    oe_set_focus(in);                          // pinned to 1 by NX2NAVI_ASSUME_FOCUS inside
    oe_push_message(OE_MSG_FOCUS_STATE_CHANGED);
    plat_inc32(&g_focus_msgs);
    struct tm t;
    plat_local_time(&t);
    rt_log("gfx: window %s the focus at %04d-%02d-%02d %02d:%02d:%02d local time", in ? "gained" : "lost",
           t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
}

// The window was closed: the process ends with it, once a save Commit still running in the background is
// done. After gfx_window_request_restart the executable then starts again with the same command line and environment
// (plat_restart), and the new process waits until this one is gone (main.c) before it moves the log and opens the
// RomFS and the saves.
void gfx_window_closed(void) {
    int restart = plat_cas32(&g_restart, 0, 0) != 0;
    rt_log("gfx: window closed; %s", restart ? "restarting" : "exiting");
    cfg_save();   // a menu change saves 400 ms later on the render thread (option_rows.cpp); closing sooner kept the old file
    // The window closed while the boot still built its pipelines: not the driver crash the marker is there to catch.
    // Left in place, the next start took it for one and every later start skipped the prebuild (vk_pipeline.c pm_skipped).
    vk_pipeline_prewarm_marker_clear();
    vk_pipeline_save_exit();
    prof_write();   // plat_exit runs no atexit handlers
    rtss_limit_restore();   // RivaTuner Statistics Server's profile as it was
    coop_shutdown();   // LEAVE and BYE to the room, before the process ends
    fs_save_flush();   // last: the game's threads still run until plat_exit, and a Commit after this one is refused
    if (restart) {
        cfg_setenv("NX2NAVI_REEXEC", NULL);   // the new process counts its own address-space retries (memory.c)
        if (!plat_restart()) rt_log("gfx: the restart could not start the game again");
    }
    plat_exit(0);
}

void gfx_window_request_restart(void) {
    plat_xchg32(&g_restart, 1);
    gfx_window_request_close();
}

// NX2NAVI_FULLSCREEN=1: borderless window over the monitor. NX2NAVI_WINDOW_SIZE=<w>x<h>: client area of the window (the
// swapchain follows the window; the game image is scaled to it).
int gfx_cfg_fullscreen(void) { return cfg_bool("NX2NAVI_FULLSCREEN"); }

void gfx_cfg_window_size(int *w, int *h) {
    int x = 0, y = 0;
    if (sscanf(cfg_value("NX2NAVI_WINDOW_SIZE"), "%dx%d", &x, &y) != 2 || x < 640 || y < 360 || x > 7680 || y > 4320)
        sscanf(cfg_meta("WINDOW_SIZE")->def, "%dx%d", &x, &y);   // unparsable or out of range: the default
    *w = x;
    *h = y;
}

// The refresh rate of the window's monitor (NX2NAVI_FPS_LIMIT=display), measured by the back end's window thread when the
// window opens, lands on another monitor or the display mode changes; readers take the stored value.
static volatile int32_t g_rate_mhz;   // millihertz, 0 unknown

void gfx_window_set_refresh(double hz) {
    int32_t mhz = (int32_t)(hz * 1000 + 0.5);
    if (plat_xchg32(&g_rate_mhz, mhz) != mhz) rt_log("gfx: monitor refresh rate %.3f Hz", hz);
}

double gfx_window_refresh_hz(void) { return plat_cas32(&g_rate_mhz, 0, 0) / 1000.0; }

// ---- swapchain and present state (g_pl)
static PlatRMutex g_pl;
static VkSwapchainKHR g_sc;
static uint64_t g_sc_generation;
static volatile int32_t g_sc_blocked; // no later work may acquire from this generation
typedef enum { PRESENT_NATIVE, PRESENT_DLSSG, PRESENT_FFX } PresentPath;
static PresentPath g_sc_path;
static VkFormat g_sc_fmt;
static int g_sc_hdr;                        // HDR10 (NX2NAVI_HDR, vk_hdr.c): PQ code values in Rec. 2020
static VkExtent2D g_sc_ext;
static GfxOrdinaryPicture g_ordinary_picture;
static uint64_t g_ordinary_picture_sequence;
static int g_scaler;                        // the last present: 0 the game image 1:1, 1 scaled by the blit, 2 by FSR,
                                            // 3 shrunk by the blit and sharpened by RCAS
static uint32_t g_sc_n;
static VkImage g_sc_img[MAX_IMAGES];
static VkImageView g_sc_view[MAX_IMAGES];   // colour attachment views for the menu overlay
static VkSemaphore g_render_done[MAX_IMAGES];
static VkCommandPool g_pool;
static VkCommandBuffer g_cb[FRAMES];
static VkFence g_fence[FRAMES];
static VkTexturePresentSource *g_slot_source[FRAMES];
static VkDlaaFgSizeUse *g_slot_temporal[FRAMES];
static VkDlaaHudSource *g_slot_hud[FRAMES];
typedef enum { COMPOSE_KNOWN, COMPOSE_UNCERTAIN, COMPOSE_UNCERTAIN_ACQUIRE } ComposeState;
static ComposeState g_compose_state;
static VkSemaphore g_acquire[FRAMES];
static unsigned g_frame;
static int g_capture;
static VkBuffer g_cap_buf;
static VmaAllocation g_cap_alloc;
static uint8_t *g_cap_ptr;
static VkDeviceSize g_cap_size;
static int g_cap_w, g_cap_h, g_cap_valid;
typedef enum { FLIP_NONE, FLIP_COPY, FLIP_READY, FLIP_UNSENT, FLIP_UNCERTAIN } FlipPhase;
typedef enum { SDK_FLIP_LEGACY, SDK_FLIP_RECEIPT, SDK_FLIP_FALLBACK } SdkFlipPath;
typedef struct {
    FlipPhase phase;
    uint64_t generation;
    VkSwapchainKHR swapchain;
    PresentPath path;
    unsigned slot;
    uint32_t index;
    VkQueue queue;
    VkSemaphore render_done;
    SdkFlipPath sdk_path;
    uint64_t sdk_frame, sdk_config, guest_frame, hud_generation;
    int hud_exact;
    GfxOrdinaryPicture picture;
} PendingFlip;
static PendingFlip g_flip;

typedef enum { SDK_INPUTS_FREE, SDK_INPUTS_PENDING, SDK_INPUTS_TIMELINE, SDK_INPUTS_QUARANTINED } SdkInputState;
// g_pl protects this lease of the images owned by the composition slot. Graphics
// completion only covers composition; the SDK can still read the same exact images.
static struct PresentSdkInputs {
    SdkInputState state;
    Nx2SlInputReceipt receipt;
    Nx2SlImage hudless, alpha;
    uint64_t hud_generation;
    VkSemaphore semaphore;
    uint64_t value;
} g_present_sdk_inputs[FRAMES];
// FidelityFX is configured before submit and interpolates later during its present.
// The compositor slot keeps this exact image until the SDK generation finishes.
static struct FfxHudlessUse {
    VkImage image;
    VkSwapchainKHR swapchain;
    uint64_t generation;
} g_ffx_hudless[FRAMES];
// Each slot owns the composed picture until the display copy and FG input consumers finish.
// The game timeline covers composition only; it never waits for the display queue to present.
static struct PresentSlot {
    VkImage image;
    VkImageView view;
    VmaAllocation memory;
    VkExtent2D extent;
    VkFormat format;
    VkCommandBuffer copy;
    VkFence copied;
    VkFence acquired;
    VkSemaphore retired;
    uint64_t retire_value, ready_value;
} g_present_slots[FRAMES];
static VkSemaphore g_snapshot_ready;
static uint64_t g_snapshot_value;
static uint64_t g_present_acquire_ns, g_present_flip_ns, g_present_reuse_ns;
static unsigned g_present_samples;
static void present_require(VkResult result, const char *operation) {
    if (result == VK_SUCCESS) return;
    vk_device_lost_check(result, operation);
    rt_fatal("gfx: %s failed (%d)", operation, (int)result);
}
typedef enum { WSI_SHOWN, WSI_ENQUEUED, WSI_UNSENT, WSI_UNCERTAIN } WsiHandoff;

static int wsi_no_submission(VkResult r) {
    return r == VK_ERROR_OUT_OF_HOST_MEMORY || r == VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

static WsiHandoff wsi_handoff(PresentPath path, VkResult r) {
    if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR) return WSI_SHOWN;
    if (path != PRESENT_NATIVE) return WSI_UNCERTAIN;
    if (wsi_no_submission(r)) return WSI_UNSENT;
    switch (r) {
    case VK_ERROR_OUT_OF_DATE_KHR:
    case VK_ERROR_SURFACE_LOST_KHR:
    case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
#ifdef VK_EXT_present_timing
    case VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT:
#endif
        // A rejection by native WSI still enqueues the semaphore wait.
        return WSI_ENQUEUED;
    default:
        return WSI_UNCERTAIN;
    }
}

static void wsi_block_generation(void) {
    plat_xchg32(&g_sc_blocked, 1);
    plat_xchg32(&g_dirty, 1);
}

static WsiHandoff wsi_record_handoff(PresentPath path, VkResult r, uint64_t generation) {
    WsiHandoff handoff = wsi_handoff(path, r);
    if (handoff == WSI_SHOWN) {
        if (r == VK_SUBOPTIMAL_KHR) plat_xchg32(&g_dirty, 1);
        return handoff;
    }
    if (handoff != WSI_ENQUEUED)
        rt_log("gfx: generation %llu present failed (%d), handoff %s", (unsigned long long)generation,
               (int)r, handoff == WSI_UNSENT ? "unsent" : "uncertain");
    wsi_block_generation();
    return handoff;
}

static VkResult wsi_acquire(VkCtx *c, const PendingFlip *flip, VkSemaphore acquire,
                            VkFence acquired, uint32_t *index) {
    switch (flip->path) {
    case PRESENT_DLSSG:
        return vk_dlssg_acquire(c->device, flip->swapchain, 1000000000ull, acquire, index);
    case PRESENT_FFX:
        return vk_fg_acquire(flip->swapchain, 1000000000ull, acquire, index);
    default:
        return vkAcquireNextImageKHR(c->device, flip->swapchain, 1000000000ull, acquire, acquired, index);
    }
}

static void wsi_bind_image(PendingFlip *flip, uint32_t index) {
    if (index >= g_sc_n || !g_sc_img[index] || !g_render_done[index])
        rt_fatal("gfx: invalid acquired index %u for swapchain generation %llu (%u images)",
                 index, (unsigned long long)flip->generation, g_sc_n);
    flip->index = index;
    flip->render_done = g_render_done[index];
}

// A known-unsent copy/compose left an acquire outstanding. Enqueue only its wait.
// The acquired image and this generation are then abandoned together on rebuild.
static void wsi_settle_acquire(VkQueue queue, VkSemaphore acquire, VkFence fence) {
    VkSemaphoreSubmitInfo wait = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = acquire, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount = acquire ? 1u : 0u, .pWaitSemaphoreInfos = &wait};
    int separate = queue != vk_ctx()->gfx_queue;
    if (separate) vk_present_queue_lock(); else vk_queue_lock();
    VkResult r = vkQueueSubmit2(queue, 1, &si, fence);
    if (separate) vk_present_queue_unlock(); else vk_queue_unlock();
    present_require(r, "settle unsent presentation");
}

static void sync_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to,
                         VkPipelineStageFlags2 s0, VkAccessFlags2 a0, VkPipelineStageFlags2 s1, VkAccessFlags2 a1);

static int present_split(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *v = cfg_get("NX2NAVI_PRESENT_SPLIT");
        enabled = !v || strcmp(v, "0");
        rt_log("gfx: presentation handoff %s", enabled ? "enabled" : "legacy diagnostic");
    }
    // FSR and NGX already own their display scheduling. A device without a spare queue
    // keeps the direct path: an extra copy there would not remove queue serialization.
    return enabled && vk_ctx()->present_queue != vk_ctx()->gfx_queue && !vk_fg_active() && !vk_ngxfg_wanted();
}

static VkResult present_slot_sdk_ready(const struct PresentSlot *p) {
    if (!p->retired || !p->retire_value) return VK_SUCCESS;
    VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1, .pSemaphores = &p->retired, .pValues = &p->retire_value};
    return vkWaitSemaphores(vk_ctx()->device, &wi, 5000000000ull);
}

static VkResult present_inputs_ready(unsigned slot) {
    struct PresentSdkInputs *use = &g_present_sdk_inputs[slot];
    if (use->state == SDK_INPUTS_FREE) return VK_SUCCESS;
    if (use->state != SDK_INPUTS_TIMELINE) return VK_NOT_READY;
    VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1, .pSemaphores = &use->semaphore, .pValues = &use->value};
    VkResult r = vkWaitSemaphores(vk_ctx()->device, &wi, 5000000000ull);
    if (r == VK_SUCCESS) {
        if (rt_trace & RT_TRACE_PRESENT)
            rt_log("presentation: Streamline retired frame %llu generation %llu ticket %llu slot %u value %llu",
                   (unsigned long long)use->receipt.frame, (unsigned long long)use->receipt.generation,
                   (unsigned long long)use->receipt.ticket, slot, (unsigned long long)use->value);
        *use = (struct PresentSdkInputs){0};
    }
    return r;
}

static int present_inputs_match(const PendingFlip *flip, const Nx2SlInputReceipt *receipt) {
    return receipt->size == sizeof *receipt && receipt->version == NX2SL_INPUTS_ABI_V1 &&
           !receipt->flags && !receipt->reserved &&
           receipt->frame == flip->sdk_frame && receipt->sdk_frame == (uint32_t)flip->sdk_frame &&
           receipt->generation == flip->generation && receipt->swapchain == flip->swapchain;
}

static void present_inputs_quarantine(struct PresentSdkInputs *use, const char *reason) {
    use->state = SDK_INPUTS_QUARANTINED;
    rt_log("presentation: retain Streamline inputs for frame %llu generation %llu ticket %llu (%s)",
           (unsigned long long)use->receipt.frame, (unsigned long long)use->receipt.generation,
           (unsigned long long)use->receipt.ticket, reason);
}

static VkDlssgConfigResult present_inputs_tag(PendingFlip *flip, VkCommandBuffer cb, VkExtent2D extent,
    const int rect[4], const Nx2SlImage *hudless, const Nx2SlImage *alpha, VkDlssgConfigPreparation *prepared) {
    struct PresentSdkInputs *use = &g_present_sdk_inputs[flip->slot];
    if (use->state != SDK_INPUTS_FREE) rt_fatal("gfx: Streamline inputs already own this present slot");
    if (!prepared || !prepared->count || prepared->frame != flip->guest_frame) return VK_DLSSG_CONFIG_INVALIDATED;
    flip->sdk_frame = flip->guest_frame;
    flip->sdk_path = SDK_FLIP_RECEIPT;
    use->state = SDK_INPUTS_PENDING;
    use->hud_generation = flip->hud_generation;
    RcAdmissionVector peak;
    int sdk_tag_ok;
    const int expected_off = !prepared->generated;
    VkDlssgConfigResult config = vk_dlssg_frame_inputs_admitted(cb, extent, g_sc_fmt, g_sc_n, rect, hudless, alpha,
        flip->sdk_frame, flip->swapchain, flip->generation, prepared, &use->receipt, &flip->sdk_config, &peak,
        &sdk_tag_ok);
    if (config != VK_DLSSG_CONFIG_READY && config != VK_DLSSG_CONFIG_COMPAT) {
        *use = (struct PresentSdkInputs){0};
        return config;
    }
    if (expected_off && !sdk_tag_ok) {
        static unsigned off_unproven;
        if (off_unproven++ < 8)
            rt_log("presentation: Streamline did not acknowledge off for frame %llu",
                   (unsigned long long)flip->sdk_frame);
    }
    uint32_t supplied = (hudless ? NX2SL_INPUT_HUDLESS : 0) | (alpha ? NX2SL_INPUT_UI_ALPHA : 0);
    uint32_t late = use->receipt.accepted | use->receipt.uncertain;
    if (!present_inputs_match(flip, &use->receipt) || (late & ~supplied) ||
        (use->receipt.accepted & use->receipt.uncertain) || (!!late != !!use->receipt.ticket)) {
        // Preserve the one WSI call when the tag acknowledgement cannot be used.
        // Supplied images can still be owned by the SDK even without a valid ticket.
        flip->sdk_path = SDK_FLIP_FALLBACK;
        if (!late || !supplied) {
            *use = (struct PresentSdkInputs){0};
            return config;
        }
        use->receipt = (Nx2SlInputReceipt){.frame = flip->sdk_frame, .generation = flip->generation,
                                           .swapchain = flip->swapchain, .uncertain = supplied};
        late = supplied;
        present_inputs_quarantine(use, "invalid tag receipt");
    }
    if (late & NX2SL_INPUT_HUDLESS) use->hudless = *hudless;
    if (late & NX2SL_INPUT_UI_ALPHA) use->alpha = *alpha;
    return config;
}

static void present_inputs_abandon(const PendingFlip *flip, int known_unsent) {
    vk_dlssg_config_abandon(flip->sdk_config);
    if (flip->sdk_path != SDK_FLIP_RECEIPT) return;
    struct PresentSdkInputs *use = &g_present_sdk_inputs[flip->slot];
    if (use->state != SDK_INPUTS_PENDING || use->hud_generation != flip->hud_generation ||
        !present_inputs_match(flip, &use->receipt))
        rt_fatal("gfx: missing Streamline inputs for an abandoned flip");
    if (!use->receipt.ticket || (known_unsent && vk_dlssg_cancel_inputs(&use->receipt))) {
        *use = (struct PresentSdkInputs){0};
    } else {
        present_inputs_quarantine(use, known_unsent ? "SDK cancellation failed" : "uncertain composition");
    }
}

static VkResult present_inputs_flip(const PendingFlip *flip, VkPresentInfoKHR *pi, VkResult *native, VkResult *per_image) {
    if (flip->sdk_path == SDK_FLIP_LEGACY) return vk_dlssg_present(flip->queue, pi);
    if (flip->sdk_path == SDK_FLIP_FALLBACK) {
        uint64_t previous = vk_dlssg_current_frame();
        vk_dlssg_present_frame(flip->sdk_frame);
        VkResult r = vk_dlssg_present(flip->queue, pi);
        vk_dlssg_present_frame(previous);
        return r;
    }
    struct PresentSdkInputs *use = &g_present_sdk_inputs[flip->slot];
    if (use->state != SDK_INPUTS_PENDING || use->hud_generation != flip->hud_generation ||
        !present_inputs_match(flip, &use->receipt))
        rt_fatal("gfx: missing Streamline inputs for the pending flip");
    // Preserve the actual SDK lane and require both Vulkan present results.
    VkResult image = VK_RESULT_MAX_ENUM;
    pi->pResults = &image;
    Nx2SlInputCompletion completion = {0};
    VkResult presented = vk_dlssg_present_inputs(flip->queue, pi, &use->receipt, &completion);
    pi->pResults = NULL;
    *native = presented;
    *per_image = image;
    VkResult r = presented;
    if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR) {
        if (image != VK_SUCCESS) r = image == VK_RESULT_MAX_ENUM ? VK_ERROR_UNKNOWN : image;
    }
    if (use->receipt.ticket && (rt_trace & RT_TRACE_PRESENT)) {
        rt_log("presentation: HUD source frame %llu generation %llu lineage %s",
               (unsigned long long)flip->guest_frame, (unsigned long long)flip->hud_generation,
               flip->hud_exact ? "same physical image" : "compatible render path");
        rt_log("presentation: Streamline frame %llu token %u generation %llu slot %u %s results %d/%d "
               "ticket %llu masks 0x%x/0x%x completion %u ticket %llu semaphore 0x%llx value %llu",
               (unsigned long long)flip->sdk_frame, use->receipt.sdk_frame, (unsigned long long)flip->generation,
               flip->slot, flip->queue == vk_ctx()->gfx_queue ? "direct" : "split", (int)presented, (int)image,
               (unsigned long long)use->receipt.ticket, use->receipt.accepted, use->receipt.uncertain,
               completion.flags, (unsigned long long)completion.ticket,
               (unsigned long long)(uintptr_t)completion.semaphore, (unsigned long long)completion.value);
    }
    if (!use->receipt.ticket) {
        *use = (struct PresentSdkInputs){0};
    } else if ((r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR) && completion.size == sizeof completion &&
               completion.version == NX2SL_INPUTS_ABI_V1 && !completion.reserved &&
               completion.ticket == use->receipt.ticket && completion.flags == NX2SL_INPUT_TIMELINE &&
               completion.semaphore && completion.value) {
        use->state = SDK_INPUTS_TIMELINE;
        use->semaphore = completion.semaphore;
        use->value = completion.value;
    } else {
        present_inputs_quarantine(use, "present has no SDK completion");
    }
    return r;
}

static int present_slot_ready(unsigned slot) {
    struct PresentSlot *p = &g_present_slots[slot];
    VkDevice d = vk_ctx()->device;
    uint64_t start = kern_now_ns();
    VkResult r = vkWaitForFences(d, 1, &p->copied, VK_TRUE, 5000000000ull);
    if (kern_now_ns() - start > 25000000ull) {
        static unsigned late;
        if (++late <= 12) rt_log("presentation: delayed copy %.3f ms",
            (kern_now_ns() - start) / 1e6);
    }
    if (r == VK_SUCCESS) r = present_slot_sdk_ready(p);
    g_present_reuse_ns += kern_now_ns() - start;
    if (r != VK_SUCCESS) {
        vk_device_lost_check(r, "presentation slot retirement");
        rt_fatal("gfx: presentation slot retirement failed (%d)", (int)r);
    }
    p->retired = VK_NULL_HANDLE;
    p->retire_value = 0;
    return 1;
}

static void present_slot_drop(struct PresentSlot *p) {
    VkCtx *c = vk_ctx();
    if (p->view) vkDestroyImageView(c->device, p->view, NULL);
    if (p->image) resource_core_vma_destroy_image(c->vma, p->image, p->memory);
    p->view = VK_NULL_HANDLE;
    p->image = VK_NULL_HANDLE;
    p->memory = NULL;
}

static int present_slot_image(unsigned slot) {
    struct PresentSlot *p = &g_present_slots[slot];
    if (p->image && p->extent.width == g_sc_ext.width && p->extent.height == g_sc_ext.height && p->format == g_sc_fmt) return 1;
    // The caller has confirmed both the graphics copy and the SDK retire condition. Keep this
    // generation available until the replacement image and view have both been prepared.
    VkCtx *c = vk_ctx();
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
        .format = g_sc_fmt, .extent = {g_sc_ext.width, g_sc_ext.height, 1}, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                  .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT};
    const RcVmaRequest request = {.producer = RC_BUDGET_OTHER, .required = 1};
    struct PresentSlot replacement = {0};
    VkResult result = resource_core_vma_create_image(&request, c->vma, &ii, &ai, &replacement.image,
                                                        &replacement.memory, NULL, NULL);
    if (result != VK_SUCCESS) {
        vk_device_lost_check(result, "admitting a DLSSG presentation slot");
        return 0;
    }
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = replacement.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = g_sc_fmt,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkImageView view;
    result = vkCreateImageView(c->device, &vi, NULL, &view);
    if (result != VK_SUCCESS) {
        present_slot_drop(&replacement);
        vk_device_lost_check(result, "preparing a DLSSG presentation slot view");
        return 0;
    }
    present_slot_drop(p);
    p->image = replacement.image;
    p->memory = replacement.memory;
    p->view = view;
    p->extent = g_sc_ext;
    p->format = g_sc_fmt;
    return 1;
}

// Called after the renderer has released its source images. Acquisition and display copies
// wait only on the presentation queue, never on the game's completion timeline.
static int present_snapshot_copy(PendingFlip *flip) {
    VkCtx *c = vk_ctx();
    struct PresentSlot *p = &g_present_slots[flip->slot];
    VkSemaphore acquire = g_acquire[flip->slot];
    uint64_t start = kern_now_ns();
    if (flip->path == PRESENT_NATIVE)
        present_require(vkResetFences(c->device, 1, &p->acquired), "reset display acquisition fence");
    uint32_t index;
    VkResult r = wsi_acquire(c, flip, acquire, p->acquired, &index);
    g_present_acquire_ns += kern_now_ns() - start;
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
        vk_device_lost_check(r, "snapshot acquisition");
        wsi_block_generation();
        return 0;
    }
    if (r == VK_SUBOPTIMAL_KHR) plat_xchg32(&g_dirty, 1);
    wsi_bind_image(flip, index);
    VkCommandBuffer cb = p->copy;
    present_require(vkResetCommandBuffer(cb, 0), "reset display copy commands");
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    present_require(vkBeginCommandBuffer(cb, &bi), "begin display copy commands");
    VkImage dst = g_sc_img[flip->index];
    sync_barrier(cb, dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageCopy cp = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .extent = {p->extent.width, p->extent.height, 1}};
    vkCmdCopyImage(cb, p->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
    sync_barrier(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_NONE, 0);
    present_require(vkEndCommandBuffer(cb), "end display copy commands");
    VkSemaphoreSubmitInfo waits[2] = {
        {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = g_snapshot_ready,
         .value = p->ready_value, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
        {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = acquire,
         .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT}};
    VkSemaphoreSubmitInfo sig = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = flip->render_done, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkCommandBufferSubmitInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount = 2, .pWaitSemaphoreInfos = waits, .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cbi, .signalSemaphoreInfoCount = 1, .pSignalSemaphoreInfos = &sig};
    present_require(vkResetFences(c->device, 1, &p->copied), "reset display copy fence");
    vk_present_queue_lock();
    r = vkQueueSubmit2(flip->queue, 1, &si, p->copied);
    vk_present_queue_unlock();
    if (r != VK_SUCCESS) {
        if (!wsi_no_submission(r)) present_require(r, "snapshot display copy");
        wsi_block_generation();
        wsi_settle_acquire(flip->queue, acquire, p->copied);
        return 0;
    }
    flip->phase = FLIP_READY;
    return 1;
}

// ---- NX2NAVI_FLIGHT thumbnails (flight.c; g_pl): each present also blits the swapchain image down into slot frame % g_th_n
// of a host-visible ring. A slot's tag names the frame it holds complete; recording a new frame clears the tag first,
// and the tag is set once the fence of that present has signalled.
#define TH_BYTES ((VkDeviceSize)FLIGHT_THUMB_W * FLIGHT_THUMB_H * 4)
static int g_th_n, g_th_failed;
static uint64_t g_th_frame;                // frame of the next vk_present_image call (vk_thumbs_frame)
static uint64_t *g_th_tag;                 // per ring slot
static uint64_t g_th_pending[FRAMES];      // per fence slot: the frame whose thumbnail its submission writes
static VkImage g_th_img;
static VmaAllocation g_th_img_alloc;
static VkFormat g_th_fmt;
static VkBuffer g_th_buf;
static VmaAllocation g_th_alloc;
static uint8_t *g_th_ptr;

// ---- vsync
static PlatMutex g_vs_lock = PLAT_MUTEX_INIT;
static uint64_t g_vs_ev[16];
static int g_vs_nev;
static volatile int32_t *g_vs_ctr[4];
static int g_vs_nctr;
static volatile int64_t g_ticks;
static volatile int64_t g_vs_start_ns;
static int g_vs_started;

static void vsync_main(void *arg) {
    (void)arg;
    plat_thread_priority(PLAT_PRIO_HIGHEST);
    uint64_t n = 0;
    for (;;) {
        uint64_t start = (uint64_t)g_vs_start_ns;
        uint64_t due = start + (n + 1) * VSYNC_NS / VSYNC_HZ;
        uint64_t now = kern_now_ns();
        if (now > due + 100000000ull) {                  // the process stalled: restart the schedule, no catch-up burst
            g_vs_start_ns = (int64_t)(now - (n + 1) * VSYNC_NS / VSYNC_HZ);
        } else if (due > now) {
            plat_wait_ns(due - now);
        }
        n++;
        plat_xchg64(&g_ticks, (int64_t)n);
        uint64_t ev[16];
        volatile int32_t *ctr[4];
        int ne, nc;
        plat_mutex_lock_shared(&g_vs_lock);
        ne = g_vs_nev;
        nc = g_vs_nctr;
        memcpy(ev, g_vs_ev, sizeof ev);
        memcpy(ctr, g_vs_ctr, sizeof ctr);
        plat_mutex_unlock_shared(&g_vs_lock);
        for (int i = 0; i < nc; i++) plat_inc32(ctr[i]);
        for (int i = 0; i < ne; i++) kern_signal_system_event(ev[i]);
    }
    return;
}

static void vsync_ensure(void) {
    plat_rmutex_lock(&g_start_lock);
    if (!g_vs_started) {
        g_vs_started = 1;
        g_vs_start_ns = (int64_t)kern_now_ns();
        if (!plat_thread_spawn(vsync_main, NULL, "vsync", 0)) rt_fatal("gfx: cannot start the vsync thread");
    }
    plat_rmutex_unlock(&g_start_lock);
}

void gfx_vsync_bind(uint64_t system_event) {
    plat_mutex_lock(&g_vs_lock);
    int have = 0;
    for (int i = 0; i < g_vs_nev; i++) have |= g_vs_ev[i] == system_event;
    if (!have) {
        if (g_vs_nev == 16) rt_fatal("gfx: more than 16 vsync events");
        g_vs_ev[g_vs_nev++] = system_event;
    }
    plat_mutex_unlock(&g_vs_lock);
    vsync_ensure();
}

void gfx_vsync_bind_counter(volatile int32_t *counter) {
    plat_mutex_lock(&g_vs_lock);
    if (g_vs_nctr == 4) rt_fatal("gfx: more than 4 vsync counters");
    g_vs_ctr[g_vs_nctr++] = counter;
    plat_mutex_unlock(&g_vs_lock);
    vsync_ensure();
}

uint64_t gfx_vsync_ticks(void) { return (uint64_t)plat_cas64(&g_ticks, 0, 0); }
uint64_t gfx_vsync_time_ns(uint64_t tick) { return (uint64_t)g_vs_start_ns + tick * VSYNC_NS / VSYNC_HZ; }

// ---- focus
int gfx_window_focused(void) { return g_focused; }
unsigned gfx_window_focus_messages(void) { return (unsigned)g_focus_msgs; }

// Present path (g_pl held): a settings change (cfg_version) reaches the window thread, which applies a changed size or
// fullscreen; a changed NX2NAVI_VK_PRESENT or NX2NAVI_HDR rebuilds the swapchain at this present.
static int g_sc_hdr_wanted;   // vk_hdr_wanted when the swapchain was made (create_swapchain)
static void window_settings_live(void) {
    static unsigned seen;
    static int started;
    static char present[16];   // the NX2NAVI_VK_PRESENT the swapchain was made with
    unsigned v = cfg_version();
    if (started && v == seen) return;
    seen = v;
    res_fov_live();
    const char *e = cfg_value("NX2NAVI_VK_PRESENT");
    if (!started || strncmp(e, present, sizeof present - 1)) {
        snprintf(present, sizeof present, "%s", e);
        if (started) plat_xchg32(&g_dirty, 1);
    }
    if (started && vk_fg_wanted() != vk_fg_active()) plat_xchg32(&g_dirty, 1);   // frame generation switched: its own swapchain
    // HDR switched in the menu: a swapchain of the other format. Compared with what the swapchain was made for, not
    // with what it got: on a display without HDR10 the setting stays on and the swapchain SDR, and that is settled.
    if (started && vk_hdr_wanted() != g_sc_hdr_wanted) plat_xchg32(&g_dirty, 1);
    ov_operation_mode_live();
    audio_channels_live();
    if (started) gfx_window_apply_settings();
    started = 1;
}

// ---- swapchain
static const char *g_present_got = "";   // the NX2NAVI_VK_PRESENT value of the mode the swapchain got; "" before one

static VkPresentModeKHR pick_present_mode(VkPhysicalDevice pd) {
    const char *e = cfg_value("NX2NAVI_VK_PRESENT");
    VkPresentModeKHR want = !strcmp(e, "mailbox") ? VK_PRESENT_MODE_MAILBOX_KHR
                          : !strcmp(e, "immediate") ? VK_PRESENT_MODE_IMMEDIATE_KHR : VK_PRESENT_MODE_FIFO_KHR;
    uint32_t n = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd, g_surface, &n, NULL);
    VkPresentModeKHR m[16];
    if (n > 16) n = 16;
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd, g_surface, &n, m);
    VkPresentModeKHR got = VK_PRESENT_MODE_FIFO_KHR;
    int relaxed = 0, immediate = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (m[i] == want) got = want;
        relaxed |= m[i] == VK_PRESENT_MODE_FIFO_RELAXED_KHR, immediate |= m[i] == VK_PRESENT_MODE_IMMEDIATE_KHR;
    }
    // Mailbox missing (several AMD drivers): plain FIFO holds a frame that misses a refresh to the next one, and the
    // game, paced at 60 on a 60 Hz display, drops to 30 at every slow frame. Relaxed FIFO shows a late frame at once
    // (a tear that once) and waits for the refresh otherwise; immediate where that is missing too.
    if (want == VK_PRESENT_MODE_MAILBOX_KHR && got != want)
        got = relaxed ? VK_PRESENT_MODE_FIFO_RELAXED_KHR : immediate ? VK_PRESENT_MODE_IMMEDIATE_KHR : got;
    const char *name = got == VK_PRESENT_MODE_MAILBOX_KHR ? "mailbox" : got == VK_PRESENT_MODE_IMMEDIATE_KHR ? "immediate" : "fifo";
    __atomic_store_n(&g_present_got, name, __ATOMIC_RELEASE);
    static int logged = -1;   // the mode last logged: a change from the menu shows in the log
    if ((int)got != logged) {
        logged = (int)got;
        rt_log("gfx: present mode %s (NX2NAVI_VK_PRESENT=%s)",
               got == VK_PRESENT_MODE_FIFO_RELAXED_KHR ? "fifo relaxed" : name, e);
    }
    return got;
}

const char *gfx_window_present_mode(void) { return __atomic_load_n(&g_present_got, __ATOMIC_ACQUIRE); }

// Owns the deferred flip until the matching generation consumes or abandons it (g_pl held).
static int flip_locked(void) {
    PendingFlip flip = g_flip;
    if (flip.phase != FLIP_COPY && flip.phase != FLIP_READY) return 0;
    if (flip.generation != g_sc_generation || flip.swapchain != g_sc)
        rt_fatal("gfx: stale flip for swapchain generation %llu (active %llu)",
                 (unsigned long long)flip.generation, (unsigned long long)g_sc_generation);
    g_flip = (PendingFlip){0};
    const int snapshot = flip.phase == FLIP_COPY;
    if (snapshot && !present_snapshot_copy(&flip)) {
        present_inputs_abandon(&flip, 1); // no SDK present was attempted on this lane
        return 0;
    }
    VkPresentInfoKHR pi = {.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .waitSemaphoreCount = 1,
                           .pWaitSemaphores = &flip.render_done, .swapchainCount = 1, .pSwapchains = &flip.swapchain,
                           .pImageIndices = &flip.index};
    uint64_t start = kern_now_ns();
    int separate = flip.queue != vk_ctx()->gfx_queue;
    if (separate) vk_present_queue_lock(); else vk_queue_lock();
    VkResult native = VK_ERROR_UNKNOWN, image = VK_ERROR_UNKNOWN;
    VkResult r = flip.path == PRESENT_DLSSG ? present_inputs_flip(&flip, &pi, &native, &image)
               : flip.path == PRESENT_FFX ? vk_fg_present(&pi) : vkQueuePresentKHR(flip.queue, &pi);
    VkResult idle = VK_SUCCESS;
    if (r != VK_ERROR_DEVICE_LOST && snapshot && flip.path == PRESENT_DLSSG && flip.sdk_path == SDK_FLIP_LEGACY) {
        struct PresentSlot *p = &g_present_slots[flip.slot];
        if (!vk_dlssg_input_retirement(&p->retired, &p->retire_value)) {
            // Default SDK queue mode blocks this queue until FG is through. Missing state must
            // never turn into an unsafe reuse of the input images on the graphics queue.
            uint64_t idle_start = kern_now_ns();
            idle = vkQueueWaitIdle(flip.queue);
            static unsigned fallbacks;
            if (++fallbacks <= 3 || fallbacks % 300 == 0)
                rt_log("presentation: FG retirement fallback %u, queue idle %.3f ms", fallbacks,
                       (kern_now_ns() - idle_start) / 1e6);
        }
    }
    if (separate) vk_present_queue_unlock(); else vk_queue_unlock();
    if (flip.sdk_config) vk_dlssg_config_presented(flip.sdk_config, flip.guest_frame, g_fence[flip.slot], native, image);
    present_require(idle, "FG input retirement fallback");
    g_present_flip_ns += kern_now_ns() - start;
    if (++g_present_samples == 300) {
        rt_log("presentation: %u frames, acquire %.3f ms, present %.3f ms, reuse %.3f ms per frame, split %d",
            g_present_samples, g_present_acquire_ns / 3e8, g_present_flip_ns / 3e8, g_present_reuse_ns / 3e8, snapshot);
        g_present_samples = 0;
        g_present_acquire_ns = g_present_flip_ns = g_present_reuse_ns = 0;
    }
    vk_device_lost_check(r, "the present");
    WsiHandoff handoff = wsi_record_handoff(flip.path, r, flip.generation);
    if (flip.path == PRESENT_NATIVE && handoff == WSI_SHOWN)
        g_ordinary_picture = r == VK_SUCCESS ? flip.picture : (GfxOrdinaryPicture){0};
    if (handoff == WSI_UNSENT || handoff == WSI_UNCERTAIN) {
        flip.phase = handoff == WSI_UNSENT ? FLIP_UNSENT : FLIP_UNCERTAIN;
        g_flip = flip;
    }
    return handoff == WSI_SHOWN;
}

static void free_capture(void) {
    VkCtx *c = vk_ctx();
    if (g_cap_buf && c) vmaDestroyBuffer(c->vma, g_cap_buf, g_cap_alloc);
    g_cap_buf = VK_NULL_HANDLE;
    g_cap_ptr = NULL;
    g_cap_size = 0;
    g_cap_valid = 0;
}

// g_pl is held and the generation passed retirement. The swapchain goes
// before the semaphores; declared SDK paths also own their native destruction.
static void drop_swapchain_locked(VkCtx *c, int release_sdk) {
    for (uint32_t i = 0; i < g_sc_n; i++) {
        if (g_sc_view[i]) vkDestroyImageView(c->device, g_sc_view[i], NULL);
        g_sc_view[i] = VK_NULL_HANDLE;
    }
    if (g_sc && g_sc_path == PRESENT_FFX) vk_fg_destroy_swapchain(g_sc), vk_fg_release();
    else if (g_sc && g_sc_path == PRESENT_DLSSG) vk_dlssg_destroy_swapchain(c->device, g_sc);
    else if (g_sc) vkDestroySwapchainKHR(c->device, g_sc, NULL);
    g_sc = VK_NULL_HANDLE;
    if (release_sdk) {
        // Destroy the proxy swapchain while its SDK hook is live. The SDK
        // still owns the tagged application images until shutdown succeeds.
        if (vk_device_is_lost() || !vk_dlssg_release_checked())
            rt_fatal("gfx: Streamline shutdown did not retire the present inputs");
        memset(g_present_sdk_inputs, 0, sizeof g_present_sdk_inputs);
    }
    for (uint32_t i = 0; i < g_sc_n; i++) {
        if (g_render_done[i]) vkDestroySemaphore(c->device, g_render_done[i], NULL);
        g_render_done[i] = VK_NULL_HANDLE;
    }
    memset(g_sc_img, 0, sizeof g_sc_img);
    g_sc_n = 0;
}

// Returns 1 with a usable swapchain, 0 when the window has no area (minimised) or the create failed, which leaves
// g_dirty set so the next present tries again. Called with g_pl held. g_dirty is cleared before the window's size is
// read: a resize that arrives while the swapchain is being made (AMD's frame generation swapchain takes tens of ms)
// sets it again, where clearing it at the end lost that resize and kept a swapchain of the old size, whose acquires
// then timed out frame after frame behind AMD's presenter (the picture stopped).
static VkResult wsi_images(VkCtx *c, uint32_t *count, VkImage *images) {
    switch (g_sc_path) {
    case PRESENT_FFX: return vk_fg_get_images(g_sc, count, images);
    case PRESENT_DLSSG: return vk_dlssg_get_images(c->device, g_sc, count, images);
    default: return vkGetSwapchainImagesKHR(c->device, g_sc, count, images);
    }
}

// The old WSI generation is already retired at this point. The caller publishes
// the new generation only when all required render-done semaphores exist.
static VkResult prepare_swapchain_images(VkCtx *c, VkFormat format) {
    uint32_t count = 0;
    VkResult r = wsi_images(c, &count, NULL);
    if (r != VK_SUCCESS) return r;
    if (!count || count > MAX_IMAGES) return VK_ERROR_INITIALIZATION_FAILED;
    r = wsi_images(c, &count, g_sc_img);
    if (r != VK_SUCCESS) return r;
    if (!count || count > MAX_IMAGES) return VK_ERROR_INITIALIZATION_FAILED;
    for (uint32_t i = 0; i < count; i++)
        if (!g_sc_img[i]) return VK_ERROR_INITIALIZATION_FAILED;

    // drop_swapchain_locked uses this count even for a partially built candidate.
    g_sc_n = count;
    VkSemaphoreCreateInfo semi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (uint32_t i = 0; i < count; i++) {
        VkSemaphore render_done;
        r = vkCreateSemaphore(c->device, &semi, NULL, &render_done);
        if (r != VK_SUCCESS) return r;
        g_render_done[i] = render_done;
        VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = g_sc_img[i],
                                    .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (vkCreateImageView(c->device, &vi, NULL, &g_sc_view[i]) != VK_SUCCESS) g_sc_view[i] = VK_NULL_HANDLE;
    }
    return VK_SUCCESS;
}

static void ng_drain(void);

static void present_ffx_hold_hudless(const PendingFlip *flip, VkImage image) {
    if (flip->path != PRESENT_FFX || !image) return;
    struct FfxHudlessUse *use = &g_ffx_hudless[flip->slot];
    if (use->image && (use->image != image || use->swapchain != flip->swapchain ||
        use->generation != flip->generation))
        rt_fatal("gfx: replacing a HUD-less input still configured in FidelityFX");
    *use = (struct FfxHudlessUse){image, flip->swapchain, flip->generation};
}

static VkResult present_ffx_finish_hudless(void) {
    unsigned retained = 0;
    for (unsigned i = 0; i < FRAMES; ++i) {
        const struct FfxHudlessUse *use = &g_ffx_hudless[i];
        if (!use->image) continue;
        if (g_sc_path != PRESENT_FFX || !g_sc || use->swapchain != g_sc || use->generation != g_sc_generation)
            return VK_ERROR_INITIALIZATION_FAILED;
        ++retained;
    }
    if (!retained) return VK_SUCCESS;
    VkResult result = vk_fg_finish_inputs(g_sc);
    if (rt_trace & RT_TRACE_PRESENT)
        rt_log("presentation: FidelityFX HUD-less generation %llu, %u slots, SDK retirement %d",
               (unsigned long long)g_sc_generation, retained, (int)result);
    if (result == VK_SUCCESS) memset(g_ffx_hudless, 0, sizeof g_ffx_hudless);
    return result;
}

// Finish the host present worker's use before testing its native submissions.
// For legacy WSI, checked device idle is the existing compatibility boundary.
// The final device shutdown settles v1 inputs through the SDK itself.
static VkResult present_retire_generation(int release_sdk) {
    if (!release_sdk) {
        // An unresolved tag has no native fence to wait on. Avoid repeatedly
        // draining both queues as resize asks to retire this generation.
        for (unsigned i = 0; i < FRAMES; ++i) {
            SdkInputState state = g_present_sdk_inputs[i].state;
            if (state == SDK_INPUTS_PENDING || state == SDK_INPUTS_QUARANTINED) return VK_NOT_READY;
        }
    }
    ng_drain();
    VkResult r = present_ffx_finish_hudless();
    if (r != VK_SUCCESS) return r;
    r = vk_device_wait_idle();
    if (r == VK_SUCCESS && vk_dlssg_has_input_receipts()) vk_dlssg_config_progress(vk_ctx()->device, 1);
    for (unsigned i = 0; i < FRAMES && r == VK_SUCCESS; ++i) {
        r = present_slot_sdk_ready(&g_present_slots[i]);
        if (r == VK_SUCCESS && !release_sdk) r = present_inputs_ready(i);
    }
    if (r == VK_SUCCESS) {
        for (unsigned i = 0; i < FRAMES; ++i) {
            g_present_slots[i].retired = VK_NULL_HANDLE;
            g_present_slots[i].retire_value = 0;
        }
    }
    return r;
}

static void present_take_completed_temporal(unsigned slot, VkDlaaFgSizeUse **completed) {
    if (!g_slot_temporal[slot]) return;
    if (*completed) rt_fatal("gfx: two completed temporal owners for the same presentation slot");
    *completed = g_slot_temporal[slot];
    g_slot_temporal[slot] = NULL;
}

static void present_release_completed_temporal(VkDlaaFgSizeUse *completed[FRAMES]) {
    for (unsigned i = 0; i < FRAMES; ++i) {
        if (completed[i]) vk_dlaa_fg_release(completed[i]);
    }
}

static int create_swapchain(VkDlaaFgSizeUse *completed_temporal[FRAMES]) {
    if (g_compose_state != COMPOSE_KNOWN) return 0;
    flip_locked();   // the acquired image goes out before its swapchain is replaced
    plat_xchg32(&g_dirty, 0);
    VkCtx *c = vk_ctx();
    VkSurfaceCapabilitiesKHR caps;
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(c->phys, g_surface, &caps) != VK_SUCCESS) return plat_xchg32(&g_dirty, 1), 0;
    int cw = 0, ch = 0;
    wsys_client_size(&cw, &ch);
    VkExtent2D ext = caps.currentExtent.width != UINT32_MAX ? caps.currentExtent
                                                            : (VkExtent2D){(uint32_t)cw, (uint32_t)ch};
    if (!ext.width || !ext.height) return plat_xchg32(&g_dirty, 1), 0;
    uint32_t nf = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(c->phys, g_surface, &nf, NULL);
    VkSurfaceFormatKHR fmts[64];
    if (nf > 64) nf = 64;
    vkGetPhysicalDeviceSurfaceFormatsKHR(c->phys, g_surface, &nf, fmts);
    if (!nf) return plat_xchg32(&g_dirty, 1), 0;
    // UNORM: the blit writes the bytes the game wrote (no sRGB encode). NX2NAVI_HDR: HDR10 when the surface offers it.
    VkSurfaceFormatKHR sf = fmts[0];
    const int hdr_wanted = vk_hdr_wanted();
    int hdr = hdr_wanted && vk_hdr_surface_format(fmts, nf, &sf), found = hdr;
    for (uint32_t i = 0; i < nf && !found; i++)
        if (fmts[i].format == VK_FORMAT_B8G8R8A8_UNORM && fmts[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { sf = fmts[i]; found = 1; }
    for (uint32_t i = 0; i < nf && !found; i++)
        if (fmts[i].format == VK_FORMAT_R8G8B8A8_UNORM && fmts[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { sf = fmts[i]; found = 1; }
    // NGX frame generation's presenter holds the generated frames' images too, and the rendered frame waits for it:
    // room for both
    const uint32_t ngx = vk_ngxfg_wanted() ? 1 : 0, most = ngx ? 12 : 8;
    uint32_t count = caps.minImageCount + 1 + (ngx ? 2u + NG_MAX : 0u);
    if (caps.maxImageCount && count > caps.maxImageCount) count = caps.maxImageCount;
    if (count > most) count = most;

    VkResult retired = present_retire_generation(0);
    if (retired != VK_SUCCESS) {
        vk_device_lost_check(retired, "swapchain generation retirement");
        if (retired != VK_NOT_READY)
            rt_log("gfx: generation %llu is still owned after retirement wait failed (%d)",
                   (unsigned long long)g_sc_generation, (int)retired);
        wsi_block_generation();
        return 0;
    }
    // Checked retirement covers the old generation's composition on both
    // presentation paths. Transfer its depth/motion holds before replacing WSI.
    if (completed_temporal) {
        for (unsigned i = 0; i < FRAMES; ++i)
            present_take_completed_temporal(i, &completed_temporal[i]);
    }
    g_flip = (PendingFlip){0};
    vk_queue_lock();
    VkSwapchainKHR old = g_sc;
    VkSwapchainCreateInfoKHR ci = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, .surface = g_surface, .minImageCount = count,
        .imageFormat = sf.format, .imageColorSpace = sf.colorSpace, .imageExtent = ext, .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,   // every surface supports it (the menu overlay draws into it)
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE, .preTransform = caps.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, .presentMode = pick_present_mode(c->phys), .clipped = VK_TRUE,
        .oldSwapchain = old};
    // Frame generation (vk_fg.cpp): AMD's swapchain in this one's place. Its swapchain is replaced only after the old
    // one is gone; a first creation that fails leaves the plain swapchain below.
    VkResult r = VK_ERROR_FEATURE_NOT_PRESENT;
    PresentPath path = PRESENT_NATIVE;
    if (!vk_fg_wanted() && vk_fg_active()) {   // switched off in the menu: AMD's swapchain and contexts go, the plain one follows
        if (old && g_sc_path == PRESENT_FFX) {
            vk_fg_destroy_swapchain(old);
            ci.oldSwapchain = VK_NULL_HANDLE;
            old = g_sc = VK_NULL_HANDLE;
        }
        vk_fg_release();
    }
    if (vk_fg_wanted()) {
        if (old && g_sc_path == PRESENT_FFX) vk_fg_destroy_swapchain(old);
        else if (old && g_sc_path == PRESENT_DLSSG) vk_dlssg_destroy_swapchain(c->device, old);
        else if (old) vkDestroySwapchainKHR(c->device, old, NULL);
        ci.oldSwapchain = VK_NULL_HANDLE;
        old = g_sc = VK_NULL_HANDLE;
        r = vk_fg_create_swapchain(&ci, &g_sc);
        if (r == VK_SUCCESS) path = PRESENT_FFX;
        // once AMD's swapchain context exists every swapchain call goes through it: a plain swapchain in its place would
        // get AMD's acquire and present. Try again at the next present instead.
        if (r != VK_SUCCESS && vk_fg_active()) {
            vk_device_lost_check(r, "create frame generation swapchain");
            static volatile int32_t n_fail;
            if (plat_inc32(&n_fail) <= 5) rt_log("gfx: frame generation swapchain %ux%u failed (%d); trying again", ext.width, ext.height, (int)r);
            g_sc = VK_NULL_HANDLE;
            drop_swapchain_locked(c, 0);   // the old swapchain's views and semaphores
            vk_queue_unlock();
            plat_xchg32(&g_dirty, 1);
            return 0;
        }
    }
    if (r != VK_SUCCESS && vk_dlssg_active()) {
        r = vk_dlssg_create_swapchain(c->device, &ci, &g_sc);
        if (r == VK_SUCCESS) path = PRESENT_DLSSG;
    } else if (r != VK_SUCCESS) {
        r = vkCreateSwapchainKHR(c->device, &ci, NULL, &g_sc);
    }
    if (r != VK_SUCCESS) {
        vk_device_lost_check(r, "create swapchain");
        static volatile int32_t n_failed;
        int32_t nf_log = plat_inc32(&n_failed);
        if (nf_log <= 8 || !(nf_log & 1023)) rt_log("gfx: vkCreateSwapchainKHR %ux%u failed (%d), %d times", ext.width, ext.height, (int)r, nf_log);
        // A create with oldSwapchain retires the old swapchain even when it fails: it can no longer acquire or be an
        // oldSwapchain again. It goes now (the device is idle), and the next present creates one from nothing.
        g_sc = old;   // NULL when a frame generation path destroyed it already: its views and semaphores still go
        drop_swapchain_locked(c, 0);
        vk_queue_unlock();
        plat_xchg32(&g_dirty, 1);
        return 0;
    }
    VkSwapchainKHR made = g_sc;
    g_sc = old;   // as above: NULL when a frame generation path destroyed it
    drop_swapchain_locked(c, 0);
    g_sc = made;
    g_sc_path = path;
    r = prepare_swapchain_images(c, sf.format);
    if (r != VK_SUCCESS) {
        vk_device_lost_check(r, "prepare swapchain images");
        rt_log("gfx: cannot prepare swapchain images for %ux%u (%d)", ext.width, ext.height, (int)r);
        drop_swapchain_locked(c, 0);
        vk_queue_unlock();
        wsi_block_generation();
        return 0;
    }
    g_sc_fmt = sf.format;
    g_sc_hdr_wanted = hdr_wanted;
    static int hdr_was = -1;   // the log says when a later swapchain is of the other kind (the menu's HDR row)
    if (hdr_was >= 0 && hdr_was != hdr)
        rt_log("gfx: the swapchain is now %s (format %d)", hdr ? "HDR10" : "SDR", (int)sf.format);
    hdr_was = hdr;
    g_sc_hdr = hdr;
    vk_hdr_set_active(hdr);
    if (hdr) vk_hdr_metadata(g_sc, 1);
    g_sc_ext = ext;
    free_capture();
    if (++g_sc_generation == 0) rt_fatal("gfx: swapchain generation overflow");
    plat_xchg32(&g_sc_blocked, 0);
    vk_queue_unlock();
    return 1;
}

static void sync_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags2 s0,
                         VkAccessFlags2 a0, VkPipelineStageFlags2 s1, VkAccessFlags2 a1) {
    VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = s0, .srcAccessMask = a0,
                               .dstStageMask = s1, .dstAccessMask = a1, .oldLayout = from, .newLayout = to,
                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .image = img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &d);
}

static int ensure_capture_buffer(void) {
    VkCtx *c = vk_ctx();
    VkDeviceSize need = (VkDeviceSize)g_sc_ext.width * g_sc_ext.height * 4;
    if (g_cap_buf && g_cap_size == need) return 1;
    free_capture();
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = need, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(c->vma, &bi, &ai, &g_cap_buf, &g_cap_alloc, &info) != VK_SUCCESS) { g_cap_buf = VK_NULL_HANDLE; return 0; }
    g_cap_ptr = info.pMappedData;
    g_cap_size = need;
    return g_cap_ptr != NULL;
}

void vk_thumbs_enable(int n) {
    plat_rmutex_lock(&g_pl);
    if (!g_th_n && n > 0 && (g_th_tag = calloc((size_t)n, sizeof *g_th_tag))) g_th_n = n;
    plat_rmutex_unlock(&g_pl);
}

void vk_thumbs_frame(uint64_t frame) {
    plat_rmutex_lock(&g_pl);
    g_th_frame = frame;
    plat_rmutex_unlock(&g_pl);
}

static void free_thumbs(void) {
    VkCtx *c = vk_ctx();
    if (g_th_img && c) vmaDestroyImage(c->vma, g_th_img, g_th_img_alloc);
    if (g_th_buf && c) vmaDestroyBuffer(c->vma, g_th_buf, g_th_alloc);
    g_th_img = VK_NULL_HANDLE, g_th_buf = VK_NULL_HANDLE, g_th_ptr = NULL;
    if (g_th_tag) memset(g_th_tag, 0, sizeof *g_th_tag * (size_t)g_th_n);
    memset(g_th_pending, 0, sizeof g_th_pending);
}

// The ring and the blit target in the swapchain's format; allocated on the first present, the image again when the
// swapchain format changes. A failure turns the thumbnails off for the session.
static int ensure_thumbs(void) {
    if (!g_th_n || g_th_failed || !g_th_frame) return 0;
    VkCtx *c = vk_ctx();
    if (!g_th_buf) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = TH_BYTES * (VkDeviceSize)g_th_n,
                                 .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(c->vma, &bi, &ai, &g_th_buf, &g_th_alloc, &info) != VK_SUCCESS || !info.pMappedData) {
            if (g_th_buf) vmaDestroyBuffer(c->vma, g_th_buf, g_th_alloc);
            g_th_buf = VK_NULL_HANDLE;
            g_th_failed = 1;
            rt_log("gfx: flight thumbnails off: no host-visible ring of %llu MB", (unsigned long long)(bi.size >> 20));
            return 0;
        }
        g_th_ptr = info.pMappedData;
    }
    // an HDR10 swapchain blits into RGBA8: its PQ codes, flat-looking but readable as bytes
    VkFormat tf = g_sc_hdr ? VK_FORMAT_R8G8B8A8_UNORM : g_sc_fmt;
    if (g_th_img && g_th_fmt != tf) {
        vmaDestroyImage(c->vma, g_th_img, g_th_img_alloc);   // the previous present's fence was waited for: idle
        g_th_img = VK_NULL_HANDLE;
    }
    if (!g_th_img) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = tf,
                                .extent = {FLIGHT_THUMB_W, FLIGHT_THUMB_H, 1}, .mipLevels = 1, .arrayLayers = 1,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(c->vma, &ci, &ai, &g_th_img, &g_th_img_alloc, NULL) != VK_SUCCESS) {
            g_th_img = VK_NULL_HANDLE;
            g_th_failed = 1;
            rt_log("gfx: flight thumbnails off: no %ux%u image", FLIGHT_THUMB_W, FLIGHT_THUMB_H);
            return 0;
        }
        g_th_fmt = tf;
    }
    return 1;
}

// dst (TRANSFER_SRC, the swapchain image just written) -> thumbnail image -> ring slot of g_th_frame.
static void record_thumb(VkCommandBuffer cb, VkImage dst, unsigned fence_slot) {
    uint64_t f = g_th_frame;
    unsigned ring = (unsigned)(f % (uint64_t)g_th_n);
    g_th_tag[ring] = 0;   // under g_pl: vk_thumbs_read no longer hands this slot out
    sync_barrier(cb, g_th_img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageBlit region = {
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {(int)g_sc_ext.width, (int)g_sc_ext.height, 1}},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {FLIGHT_THUMB_W, FLIGHT_THUMB_H, 1}}};
    vkCmdBlitImage(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_th_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region,
                   VK_FILTER_LINEAR);
    sync_barrier(cb, g_th_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy bc = {.bufferOffset = TH_BYTES * ring, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                            .imageExtent = {FLIGHT_THUMB_W, FLIGHT_THUMB_H, 1}};
    vkCmdCopyImageToBuffer(cb, g_th_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_th_buf, 1, &bc);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                           .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &d);
    g_th_pending[fence_slot] = f;
    g_th_frame = 0;
}

int vk_thumbs_read(uint64_t frame, uint8_t *rgba) {
    int ok = 0;
    plat_rmutex_lock(&g_pl);
    if (g_th_n && g_th_ptr && frame) {
        unsigned ring = (unsigned)(frame % (uint64_t)g_th_n);
        if (g_th_tag[ring] == frame) {
            VkCtx *c = vk_ctx();
            vmaInvalidateAllocation(c->vma, g_th_alloc, TH_BYTES * ring, TH_BYTES);
            memcpy(rgba, g_th_ptr + TH_BYTES * ring, TH_BYTES);
            ok = g_th_fmt == VK_FORMAT_B8G8R8A8_UNORM || g_th_fmt == VK_FORMAT_B8G8R8A8_SRGB ? 2 : 1;
        }
    }
    plat_rmutex_unlock(&g_pl);
    if (ok == 2)
        for (size_t i = 0; i < TH_BYTES; i += 4) {
            uint8_t t = rgba[i];
            rgba[i] = rgba[i + 2], rgba[i + 2] = t;
        }
    return ok != 0;
}

void vk_present_capture(int on) {
    plat_rmutex_lock(&g_pl);
    g_capture = on;
    if (!on) g_cap_valid = 0;
    plat_rmutex_unlock(&g_pl);
}

int vk_present_captured(const uint8_t **px, int *w, int *h, VkFormat *fmt) {
    plat_rmutex_lock(&g_pl);
    int ok = g_cap_valid;
    if (ok) { *px = g_cap_ptr; *w = g_cap_w; *h = g_cap_h; *fmt = g_sc_fmt; }
    plat_rmutex_unlock(&g_pl);
    return ok;
}

static float g_pic[4];   // gfx_picture_rect
static uint64_t g_game_picture_size;   // publish both dimensions from the same game frame
static volatile int32_t g_pic_ok;

// The captured swapchain image as a PNG (the menu's test hook), RGBA with opaque alpha. From an HDR10 swapchain the
// log also gets the luminance in nits of the picture's rectangle (vk_hdr_luminance_stats: the black floor of a dark
// scene, for one), and <path>.pq10 keeps the raw pixels: "PQ10", width, height, 1 when A2R10G10B10 (u32 each), then
// the pixels as the swapchain holds them.
static void save_capture(const char *path) {
    size_t n = (size_t)g_cap_w * (size_t)g_cap_h * 4;
    uint8_t *px = malloc(n);
    if (!px) return;
    int bgra = g_sc_fmt == VK_FORMAT_B8G8R8A8_UNORM || g_sc_fmt == VK_FORMAT_B8G8R8A8_SRGB;
    if (g_sc_hdr) vk_hdr_to_srgb8(g_cap_ptr, px, n / 4, g_sc_fmt == VK_FORMAT_A2R10G10B10_UNORM_PACK32);
    else for (size_t i = 0; i < n; i += 4) {
        px[i] = g_cap_ptr[i + (bgra ? 2 : 0)];
        px[i + 1] = g_cap_ptr[i + 1];
        px[i + 2] = g_cap_ptr[i + (bgra ? 0 : 2)];
        px[i + 3] = 255;
    }
    rt_write_png(path, px, (uint32_t)g_cap_w, (uint32_t)g_cap_h);
    free(px);
    rt_log("gfx: menu test shot %s (%dx%d)", path, g_cap_w, g_cap_h);
    if (!g_sc_hdr) return;
    int a2r = g_sc_fmt == VK_FORMAT_A2R10G10B10_UNORM_PACK32;
    int rect[4] = {0, 0, g_cap_w, g_cap_h};
    if (__atomic_load_n(&g_pic_ok, __ATOMIC_ACQUIRE)) {
        rect[0] = (int)(g_pic[0] * (float)g_cap_w + 0.5f), rect[1] = (int)(g_pic[1] * (float)g_cap_h + 0.5f);
        rect[2] = (int)(g_pic[2] * (float)g_cap_w + 0.5f), rect[3] = (int)(g_pic[3] * (float)g_cap_h + 0.5f);
        if (rect[2] <= 0 || rect[3] <= 0 || rect[0] + rect[2] > g_cap_w || rect[1] + rect[3] > g_cap_h)
            rect[0] = rect[1] = 0, rect[2] = g_cap_w, rect[3] = g_cap_h;
    }
    float st[7];
    vk_hdr_luminance_stats(g_cap_ptr, g_cap_w, rect, a2r, st);
    rt_log("gfx: shot %s, HDR10 luminance of the picture (%dx%d at %d,%d) in nits: darkest %.5f, 0.1%% %.5f, 1%% %.5f, "
           "median %.3f, 99%% %.1f, brightest %.1f; %.2f%% of pixels under 0.01", path, rect[2], rect[3], rect[0], rect[1],
           st[0], st[1], st[2], st[3], st[4], st[5], 100.0f * st[6]);
    char raw[1200];
    snprintf(raw, sizeof raw, "%s.pq10", path);
    FILE *f = fopen(raw, "wb");
    if (f) {
        uint32_t head[4] = {0x30315150u /* "PQ10" */, (uint32_t)g_cap_w, (uint32_t)g_cap_h, (uint32_t)a2r};
        fwrite(head, sizeof head, 1, f);
        fwrite(g_cap_ptr, 4, (size_t)g_cap_w * (size_t)g_cap_h, f);
        fclose(f);
    }
}

// The rectangle {x, y, w, h} of a dw x dh swapchain image the cw x ch picture goes into: the whole image with
// NX2NAVI_ASPECT=stretch, else (fit, the default) the largest one of the picture's own aspect ratio, centred (a 16:9 picture
// on the Steam Deck's 1280x800 screen gets bars of 40 rows above and below; a 16:9 picture in a 21:9 window, bars at the
// sides).
// Where the last present put the picture, as fractions of the swapchain (x, y, w, h; g_pic): menu_cursor.c's pointer
void gfx_game_picture_size(unsigned *w, unsigned *h) {
    uint64_t size = __atomic_load_n(&g_game_picture_size, __ATOMIC_RELAXED);
    *w = (unsigned)(size >> 32);
    *h = (unsigned)size;
}

int gfx_picture_rect(float out[4]) {
    if (!__atomic_load_n(&g_pic_ok, __ATOMIC_ACQUIRE)) return 0;
    memcpy(out, g_pic, sizeof g_pic);
    return 1;
}

void gfx_fit_rect(int cw, int ch, int dw, int dh, int out[4]) {
    out[0] = out[1] = 0, out[2] = dw, out[3] = dh;
    if (cw <= 0 || ch <= 0 || !strcmp(cfg_value("NX2NAVI_ASPECT"), "stretch")) return;
    if ((int64_t)dw * ch > (int64_t)dh * cw) out[2] = (int)(((int64_t)dh * cw + ch / 2) / ch);   // window wider: pillarbox
    else out[3] = (int)(((int64_t)dw * ch + cw / 2) / cw);                                        // taller: letterbox
    // A side of the window's parity keeps the two bars equal; a picture one pixel short of the window fills it
    out[2] += (dw - out[2]) & 1, out[3] += (dh - out[3]) & 1;
    out[0] = (dw - out[2]) / 2, out[1] = (dh - out[3]) / 2;
}

// ---- Movie overlay (ov_movie2.c): the latest decoded movie frame goes over the game image, fit to the window
// with black bars, while frames keep coming (GFX_MOVIE_HOLD_NS after the last one it disappears). It made movies
// visible before the game's own shaders translated; now the game draws them itself (its YUV compute pass, then the
// upscaler fallback of vk_exec_draw.c), under its menus, so the overlay is off unless NX2NAVI_MOVIE_OVERLAY=1. One staging buffer per frame in flight: the CPU writes slot i only after
// that slot's fence (the present waits it before recording), the GPU copies it into one RGBA8 image.
#define GFX_MOVIE_HOLD_NS 400000000ull
static PlatMutex g_mv_lock = PLAT_MUTEX_INIT;
static uint8_t *g_mv_rgba;
static int g_mv_w, g_mv_h;
static uint64_t g_mv_ns, g_mv_serial;
static VkBuffer g_mvb[FRAMES];
static VmaAllocation g_mva[FRAMES];
static void *g_mvp[FRAMES];
static VkDeviceSize g_mvsz[FRAMES];
static uint64_t g_mv_slot_serial[FRAMES];
static VkImage g_mvi;
static VmaAllocation g_mvia;
static int g_mvi_w, g_mvi_h;

void gfx_movie_frame(const uint8_t *rgba, int w, int h) {
    if (w <= 0 || h <= 0) return;
    plat_mutex_lock(&g_mv_lock);
    size_t need = (size_t)w * h * 4;
    if (w != g_mv_w || h != g_mv_h || !g_mv_rgba) {
        free(g_mv_rgba);
        g_mv_rgba = malloc(need);
        g_mv_w = g_mv_rgba ? w : 0, g_mv_h = g_mv_rgba ? h : 0;
    }
    if (g_mv_rgba) {
        memcpy(g_mv_rgba, rgba, need);
        g_mv_ns = plat_now_ns();
        g_mv_serial++;
    }
    plat_mutex_unlock(&g_mv_lock);
}

void gfx_movie_frame_own(uint8_t *rgba, int w, int h) {
    if (w <= 0 || h <= 0 || !rgba) {
        free(rgba);
        return;
    }
    plat_mutex_lock(&g_mv_lock);
    uint8_t *old = g_mv_rgba;
    g_mv_rgba = rgba;
    g_mv_w = w, g_mv_h = h;
    g_mv_ns = plat_now_ns();
    g_mv_serial++;
    plat_mutex_unlock(&g_mv_lock);
    free(old);
}

void gfx_movie_stop(void) {
    plat_mutex_lock(&g_mv_lock);
    g_mv_ns = 0;
    plat_mutex_unlock(&g_mv_lock);
}

// Records the overlay into cb for swapchain image dst (in TRANSFER_DST_OPTIMAL, left so). 1 when drawn.
static int movie_overlay(VkCommandBuffer cb, int slot, VkImage dst) {
    static int off = -1;
    if (off < 0) {
        const char *v = cfg_get("NX2NAVI_MOVIE_OVERLAY");
        off = !(v && strcmp(v, "0"));
    }
    if (off) return 0;
    VkCtx *c = vk_ctx();
    plat_mutex_lock(&g_mv_lock);
    if (!g_mv_ns || plat_now_ns() - g_mv_ns > GFX_MOVIE_HOLD_NS || !g_mv_rgba) {
        plat_mutex_unlock(&g_mv_lock);
        return 0;
    }
    int w = g_mv_w, h = g_mv_h;
    VkDeviceSize need = (VkDeviceSize)w * h * 4;
    if (g_mvsz[slot] != need) {
        if (g_mvb[slot]) vmaDestroyBuffer(c->vma, g_mvb[slot], g_mva[slot]);
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = need,
                                 .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                               VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(c->vma, &bi, &ai, &g_mvb[slot], &g_mva[slot], &info) != VK_SUCCESS) {
            g_mvb[slot] = VK_NULL_HANDLE, g_mvsz[slot] = 0;
            plat_mutex_unlock(&g_mv_lock);
            return 0;
        }
        g_mvp[slot] = info.pMappedData;
        g_mvsz[slot] = need;
        g_mv_slot_serial[slot] = 0;
    }
    if (g_mv_slot_serial[slot] != g_mv_serial) {
        memcpy(g_mvp[slot], g_mv_rgba, (size_t)need);
        g_mv_slot_serial[slot] = g_mv_serial;
    }
    plat_mutex_unlock(&g_mv_lock);
    if (g_mvi_w != w || g_mvi_h != h) {
        if (g_mvi) {
            vk_queue_lock(), vk_device_wait_idle(), vk_queue_unlock();
            vmaDestroyImage(c->vma, g_mvi, g_mvia);
            g_mvi = VK_NULL_HANDLE;
        }
        VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                                .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {(uint32_t)w, (uint32_t)h, 1},
                                .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(c->vma, &ii, &ai, &g_mvi, &g_mvia, NULL) != VK_SUCCESS) {
            g_mvi = VK_NULL_HANDLE, g_mvi_w = g_mvi_h = 0;
            return 0;
        }
        g_mvi_w = w, g_mvi_h = h;
    }
    sync_barrier(cb, g_mvi, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_PIPELINE_STAGE_2_TRANSFER_BIT, 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferImageCopy bc = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                            .imageExtent = {(uint32_t)w, (uint32_t)h, 1}};
    vkCmdCopyBufferToImage(cb, g_mvb[slot], g_mvi, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bc);
    sync_barrier(cb, g_mvi, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT);
    // the game image's blit wrote dst: black bars first, then the frame
    sync_barrier(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkClearColorValue black = {.float32 = {0, 0, 0, 1}};
    VkImageSubresourceRange all = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &all);
    sync_barrier(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT);
    int to[4];
    gfx_fit_rect(w, h, (int)g_sc_ext.width, (int)g_sc_ext.height, to);
    VkImageBlit region = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .srcOffsets = {{0, 0, 0}, {w, h, 1}},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .dstOffsets = {{to[0], to[1], 0}, {to[0] + to[2], to[1] + to[3], 1}}};
    vkCmdBlitImage(cb, g_mvi, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region,
                   VK_FILTER_LINEAR);
    return 1;
}

// The game image's crop into dst at `to`: colour adjustment (or the HDR10 encode), then FSR or the bilinear blit; dst in
// TRANSFER_DST_OPTIMAL. Returns vk_fsr_record's result (0: the blit scaled it).
static int put_picture(VkCommandBuffer cb, unsigned slot, VkImage src, VkFormat fmt, int w, int h, const int rect[4], VkImage dst,
                       const int to[4], int general, uint32_t *effects) {
    int cw = rect[2], ch = rect[3];
    // Colour adjustment (NX2NAVI_COLOR_*, vk_color.c), or on an HDR10 swapchain the PQ encode (vk_hdr.c; the adjustment is
    // for SDR): a copy of the crop takes the game image's place below
    VkImage graded = !general ? VK_NULL_HANDLE : g_sc_hdr ? vk_hdr_encode(cb, slot, src, fmt, rect)
                                                         : vk_color_record(cb, slot, src, fmt, rect);
    const int grect[4] = {0, 0, cw, ch};
    VkImage psrc = graded ? graded : src;
    VkFormat pfmt = graded ? VK_FORMAT_R16G16B16A16_SFLOAT : fmt;
    const int *prect = graded ? grect : rect;
    // FSR (NX2NAVI_UPSCALER): EASU when the picture's rectangle is larger than the crop, the shrink and RCAS when it is
    // smaller; else the bilinear blit
    int fsr = general ? vk_fsr_record(cb, slot, psrc, pfmt, graded ? cw : w, graded ? ch : h, prect, dst, to) : 0;
    if (effects) {
        if (graded) *effects |= g_sc_hdr ? GFX_PICTURE_HDR_ENCODED : GFX_PICTURE_COLOR_GRADED;
        if (fsr) *effects |= fsr == 2 ? GFX_PICTURE_FSR_SHRUNK : GFX_PICTURE_FSR_ENLARGED;
    }
    if (!fsr && src) {
        if (general)   // a GENERAL source (vk_texture.c images): earlier submissions' writes become visible to the blit here
            sync_barrier(cb, psrc, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        VkImageBlit region = {
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffsets = {{prect[0], prect[1], 0}, {prect[0] + cw, prect[1] + ch, 1}},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{to[0], to[1], 0}, {to[0] + to[2], to[1] + to[3], 1}}};
        vkCmdBlitImage(cb, psrc, general ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_LINEAR);
        if (general)   // later submissions that write the source wait for the blit's read
            sync_barrier(cb, psrc, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 0,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0);
    }
    return fsr;
}

// The display's frame rate with frame generation: the frames the display got (generated ones included: vk_fg.cpp
// vk_fg_presents, vk_dlssg.c vk_dlssg_shown) over half a second, sampled at each of the game's presents
// (gfx_output_fps).
static volatile double g_out_fps;
static uint64_t ng_presented(void);   // frames the NGX frame generation presenter showed (0: it does not run)
static int ng_running(void);          // that presenter is in use
static void output_rate(void) {
    static uint64_t t0, n0;
    uint64_t ng_n = ng_presented();
    uint64_t n = ng_n ? ng_n : vk_dlssg_active() ? vk_dlssg_shown() : vk_fg_presents(g_sc), t = plat_now_ns();
    if (!n) { g_out_fps = 0, t0 = 0; return; }
    if (!t0 || n < n0) { t0 = t, n0 = n; return; }
    if (t - t0 >= 500000000ull) g_out_fps = (double)(n - n0) * 1e9 / (double)(t - t0), t0 = t, n0 = n;
}
double gfx_output_fps(void) { return g_out_fps; }

static struct {
    VkImage image;
    VmaAllocation memory;
    VkImageView view;
    uint32_t width, height;
    VkFormat format;
} g_hudless[FRAMES];
static unsigned g_hudless_slot;
#define g_hl g_hudless[g_hudless_slot].image
#define g_hl_mem g_hudless[g_hudless_slot].memory
#define g_hl_view g_hudless[g_hudless_slot].view
#define g_hl_w g_hudless[g_hudless_slot].width
#define g_hl_h g_hudless[g_hudless_slot].height
#define g_hl_fmt g_hudless[g_hudless_slot].format

static void hudless_drop(unsigned slot) {
    if (g_ffx_hudless[slot].image) rt_fatal("gfx: destroying a HUD-less image before FidelityFX completion");
    VkCtx *c = vk_ctx();
    if (g_hudless[slot].view) vkDestroyImageView(c->device, g_hudless[slot].view, NULL);
    if (g_hudless[slot].image)
        resource_core_vma_destroy_image(c->vma, g_hudless[slot].image, g_hudless[slot].memory);
    memset(&g_hudless[slot], 0, sizeof g_hudless[slot]);
}

static VkImage hudless_record(VkCommandBuffer cb, unsigned slot, int w, int h, const int rect[4], const int to[4],
                              int general, const VkDlaaHudImages *images) {
    if (!images || images->color.w != (uint32_t)w || images->color.h != (uint32_t)h ||
        !(vk_dlssg_active() || vk_fg_active() || ng_running())) return VK_NULL_HANDLE;
    VkImage pic = images->color.img;
    VkFormat pf = images->color.fmt;
    VkCtx *c = vk_ctx();
    if (g_hl && (g_hl_w != g_sc_ext.width || g_hl_h != g_sc_ext.height || g_hl_fmt != g_sc_fmt)) {
        VkResult idle = vk_device_wait_idle();
        if (idle != VK_SUCCESS) {
            vk_device_lost_check(idle, "retiring the composed HUD-less image");
            return VK_NULL_HANDLE;
        }
        hudless_drop(slot);
    }
    if (!g_hl) {
        VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = g_sc_fmt,
                                .extent = {g_sc_ext.width, g_sc_ext.height, 1}, .mipLevels = 1, .arrayLayers = 1,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT};
        const RcVmaRequest request = {.producer = RC_BUDGET_OTHER};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (resource_core_vma_create_image(&request, c->vma, &ii, &ai, &g_hl, &g_hl_mem, NULL, NULL) != VK_SUCCESS)
            return VK_NULL_HANDLE;
        VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = g_hl,
                                    .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = g_sc_fmt,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VkImageView view;
        if (vkCreateImageView(c->device, &vi, NULL, &view) != VK_SUCCESS) {
            hudless_drop(slot);
            return VK_NULL_HANDLE;
        }
        g_hl_view = view;
        g_hl_w = g_sc_ext.width, g_hl_h = g_sc_ext.height, g_hl_fmt = g_sc_fmt;
        rt_log("gfx: frame generation's HUD-less picture at %ux%u", g_hl_w, g_hl_h);
    }
    // the steps' work images were just read for the presented picture: they are written again only after that
    sync_barrier(cb, g_hl, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    if (to[2] != (int)g_sc_ext.width || to[3] != (int)g_sc_ext.height) {
        VkClearColorValue black = {.float32 = {0, 0, 0, 1}};
        VkImageSubresourceRange all = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, g_hl, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &all);
        sync_barrier(cb, g_hl, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT);
    }
    // the steps' per-slot views of their source: a slot of its own (the presented picture's views stay valid for this
    // command buffer); with HDR the PQ encode as for the presented float picture
    put_picture(cb, slot + FRAMES, pic, pf, w, h, rect, g_hl, to, general, NULL);
    sync_barrier(cb, g_hl, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    return g_hl;
}


// ---------------------------------------------------------------- DLSS Frame Generation through NGX (vk_ngxfg_*)
// NX2NAVI_FRAMEGEN = dlss where Streamline does not run (every build but the Windows one; vk_ngxfg_wanted is 0 there
// and none of this is reached). Each frame is composed into an image of its own (ng.bb) instead of a swapchain image.
// At the end of the frame's command buffer DLSS Frame Generation makes the generated frames from it
// (vk_ngxfg_evaluate, the multiplier less one) and the picture is copied aside. The presenter thread then shows them
// in order, each copied into a swapchain image it acquires itself, and the rendered frame last, spaced evenly over the
// time between two rendered frames, as NVIDIA's frame generation guide paces them. Two sets of images take turns: a
// set is written again two frames later, after the presenter's copies from it are done (the timeline ng.copied).
// Every frame goes through the presenter while it runs, so the timelines advance in order.
// Three things froze the picture while this was written, and the code keeps away from each:
// - a frame's own command buffer never waits for a swapchain acquire: on Wayland an acquire's semaphore can signal
//   only after a later present, and every later present would come from that command buffer;
// - a timeline wait is never submitted ahead of its signal on the same queue (ng_wait_signaled);
// - an acquire never blocks while it holds the queue lock the presenter presents with (ng_acquire).
typedef struct { VkImage img; VmaAllocation mem; VkImageView view; } NgImg;
typedef struct { int count, set; uint64_t n, t, generation; VkSwapchainKHR sc; } NgJob;
static struct {
    int tried, on, fresh;         // tried: 0 nothing made, 1 the objects below exist, 2 it cannot run here
    int thread_up;                // the presenter thread runs (started once, kept over a release)
    uint32_t w, h;
    VkFormat fmt;
    NgImg bb, out[2][NG_MAX], real[2];
    VkSemaphore rendered, copied;   // timelines: frame n's command buffer submitted/done, the presenter done with frame n
    uint64_t n;                     // frames handed to the presenter
    VkCommandPool pool;
    VkCommandBuffer cb[2][NG_MAX + 1];   // per shown frame: the generated ones, then the rendered one
    VkSemaphore acq[2][NG_MAX + 1];
    PlatMutex m;
    PlatCond cv;
    int busy, pending;
    NgJob job;
    uint64_t signaled;              // the last value of ng.copied the presenter submitted a signal for (ng.m)
    uint64_t last_t, interval;      // average time between rendered frames (ns)
    volatile uint64_t presents;     // frames presented while it runs, generated ones included
} ng = {.m = PLAT_MUTEX_INIT};

// An acquire that never blocks while it holds the queue lock: the swapchain is shared by the game thread and the
// presenter (both acquire, the presenter presents), and an image is freed only by a present, which needs that lock.
// A blocking acquire under the lock waited for the present it held off: the picture froze (1 s acquire timeouts).
static VkResult ng_acquire(VkSwapchainKHR sc, uint64_t timeout, VkSemaphore sem, uint32_t *idx) {
    VkDevice dev = vk_ctx()->device;
    uint64_t end = plat_now_ns() + timeout;
    for (;;) {
        vk_queue_lock();
        VkResult r = vkAcquireNextImageKHR(dev, sc, 0, sem, VK_NULL_HANDLE, idx);
        vk_queue_unlock();
        if (r != VK_NOT_READY && r != VK_TIMEOUT) return r;
        if (plat_now_ns() >= end) return VK_TIMEOUT;
        plat_sleep_ns(200000);
    }
}

static void ng_drain(void) {   // waits until the presenter has shown everything handed to it
    if (!ng.tried) return;
    plat_mutex_lock(&ng.m);
    while (ng.pending || ng.busy) plat_cond_wait(&ng.cv, &ng.m);
    plat_mutex_unlock(&ng.m);
}

static void ng_sleep_until(uint64_t t) {
    uint64_t now = plat_now_ns();
    if (t > now && t - now < 100000000ull) plat_sleep_ns(t - now);
}

static int ng_present(const PendingFlip *flip) {
    VkPresentInfoKHR pi = {.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .waitSemaphoreCount = 1,
                           .pWaitSemaphores = &flip->render_done, .swapchainCount = 1, .pSwapchains = &flip->swapchain,
                           .pImageIndices = &flip->index};
    vk_present_queue_lock();
    VkResult r = vkQueuePresentKHR(flip->queue, &pi);
    vk_present_queue_unlock();
    vk_device_lost_check(r, "the present of a generated frame");
    if (wsi_record_handoff(PRESENT_NATIVE, r, flip->generation) != WSI_SHOWN) return 0;
    __atomic_add_fetch(&ng.presents, 1, __ATOMIC_RELAXED);
    return 1;
}

static void ng_signal_copied(uint64_t v) {   // after everything submitted before it (frame v's copies included)
    VkSemaphoreSubmitInfo sig = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = ng.copied, .value = v,
                                 .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .signalSemaphoreInfoCount = 1, .pSignalSemaphoreInfos = &sig};
    vk_present_queue_lock();
    VkResult r = vkQueueSubmit2(vk_ctx()->present_queue, 1, &si, VK_NULL_HANDLE);
    vk_present_queue_unlock();
    if (r != VK_SUCCESS) {
        vk_device_lost_check(r, "the generated-frame copied signal");
        rt_fatal("ngxfg: copied signal submit failed (%d)", (int)r);
    }
    plat_mutex_lock(&ng.m);
    ng.signaled = v;
    plat_cond_broadcast(&ng.cv);
    plat_mutex_unlock(&ng.m);
}

// A frame's submission waits for ng.copied >= n - 2 on the GPU. That signal must be submitted first: one queue runs
// its submissions in order, and a wait queued ahead of its signal stopped the queue for good (the picture froze).
static void ng_wait_signaled(uint64_t v) {
    plat_mutex_lock(&ng.m);
    while (ng.signaled < v) plat_cond_wait(&ng.cv, &ng.m);
    plat_mutex_unlock(&ng.m);
}

// Acquires a swapchain image, copies img (GENERAL, the swapchain's size and format) into it once frame n's command
// buffer is done (ng.rendered), and presents it. Only the presenter acquires while it runs: the frame's own command
// buffer never waits for an acquire, whose image the compositor may give back only after a later present.
static int ng_show(VkImage img, const NgJob *job, int slot) {
    if (plat_cas32(&g_sc_blocked, 0, 0)) return 0;
    VkCtx *c = vk_ctx();
    int set = job->set;
    VkSemaphore acquire = ng.acq[set][slot];
    uint32_t j;
    VkResult r = ng_acquire(job->sc, 1000000000ull, acquire, &j);
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
        vk_device_lost_check(r, "the acquire of a generated frame");
        static int told;
        if (told++ < 5) rt_log("ngxfg: presenter acquire failed (%d)", (int)r);
        wsi_block_generation();
        return 0;
    }
    if (r == VK_SUBOPTIMAL_KHR) plat_xchg32(&g_dirty, 1);
    PendingFlip flip = {.generation = job->generation, .swapchain = job->sc, .queue = c->present_queue};
    wsi_bind_image(&flip, j);
    VkCommandBuffer cb = ng.cb[set][slot];
    r = vkResetCommandBuffer(cb, 0);
    if (r != VK_SUCCESS) rt_fatal("ngxfg: presenter command buffer reset failed (%d)", (int)r);
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    r = vkBeginCommandBuffer(cb, &bi);
    if (r != VK_SUCCESS) rt_fatal("ngxfg: presenter command buffer begin failed (%d)", (int)r);
    sync_barrier(cb, g_sc_img[j], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageCopy cp = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                      .extent = {ng.w, ng.h, 1}};
    vkCmdCopyImage(cb, img, VK_IMAGE_LAYOUT_GENERAL, g_sc_img[j], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
    sync_barrier(cb, g_sc_img[j], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_NONE, 0);
    r = vkEndCommandBuffer(cb);
    if (r != VK_SUCCESS) rt_fatal("ngxfg: presenter command buffer end failed (%d)", (int)r);
    VkSemaphoreSubmitInfo wait[2] = {
        {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = acquire, .stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT},
        {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = ng.rendered, .value = job->n,
         .stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT}};
    VkSemaphoreSubmitInfo sig = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = flip.render_done,
                                 .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkCommandBufferSubmitInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .waitSemaphoreInfoCount = 2, .pWaitSemaphoreInfos = wait,
                        .commandBufferInfoCount = 1, .pCommandBufferInfos = &cbi, .signalSemaphoreInfoCount = 1,
                        .pSignalSemaphoreInfos = &sig};
    vk_present_queue_lock();
    r = vkQueueSubmit2(flip.queue, 1, &si, VK_NULL_HANDLE);
    vk_present_queue_unlock();
    if (r != VK_SUCCESS) {
        if (!wsi_no_submission(r)) present_require(r, "generated frame copy");
        wsi_block_generation();
        wsi_settle_acquire(flip.queue, acquire, VK_NULL_HANDLE);
        return 0;
    }
    return ng_present(&flip);
}

static void ng_run(const NgJob *job) {
    if (job->sc != g_sc || job->generation != g_sc_generation)
        rt_fatal("ngxfg: stale job for swapchain generation %llu", (unsigned long long)job->generation);
    VkCtx *c = vk_ctx();
    uint64_t t = job->t, n = job->n;
    int set = job->set, count = job->count, shown = 0;
    if (ng.last_t && t > ng.last_t) {
        uint64_t d = t - ng.last_t;
        if (d > 2000000ull && d < 100000000ull) ng.interval = ng.interval ? (ng.interval * 7 + d) / 8 : d;
    }
    ng.last_t = t;
    uint64_t step = ng.interval / (uint64_t)(count + 1);
    if (n > 2) {   // the command buffers and acquire semaphores of this set: their last use (frame n - 2) is over
        uint64_t v = n - 2;
        VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &ng.copied,
                                  .pValues = &v};
        VkResult r = vkWaitSemaphores(c->device, &wi, 2000000000ull);
        if (r != VK_SUCCESS) {
            vk_device_lost_check(r, "generated-frame image reuse");
            rt_fatal("ngxfg: presenter reuse wait failed (%d), frame %llu", (int)r, (unsigned long long)n);
        }
    }
    for (int i = 0; i < count; i++) {
        if (i) ng_sleep_until(t + (uint64_t)i * step);   // the first one at once: the least latency
        if (!ng_show(ng.out[set][i].img, job, i)) break;
        shown = i + 1;
    }
    static uint64_t jobs, gen_ok, gen_want;
    jobs++, gen_want += (uint64_t)count;
    if (count) ng_sleep_until(t + (uint64_t)count * step);
    if (ng_show(ng.real[set].img, job, NG_MAX)) // the rendered frame, last
        perfmon_shown(kern_now_ns());
    ng_signal_copied(n);   // after the copies out of this set
    gen_ok += (uint64_t)shown;
    if (jobs == 60 || jobs % 1200 == 0)
        rt_log("ngxfg: presenter: %llu rendered frames, %llu of %llu generated frames shown, %llu presents; %.2f ms between rendered frames",
               (unsigned long long)jobs, (unsigned long long)gen_ok, (unsigned long long)gen_want,
               (unsigned long long)__atomic_load_n(&ng.presents, __ATOMIC_RELAXED), (double)ng.interval / 1e6);
}

static void ng_thread(void *arg) {
    (void)arg;
    for (;;) {
        plat_mutex_lock(&ng.m);
        while (!ng.pending) {
            if (!plat_cond_wait_ms(&ng.cv, &ng.m, 1500) && ng.on && ng.n) {   // nothing for 1.5 s: where the GPU stands
                uint64_t rv = 0, cv = 0;
                vkGetSemaphoreCounterValue(vk_ctx()->device, ng.rendered, &rv);
                vkGetSemaphoreCounterValue(vk_ctx()->device, ng.copied, &cv);
                static int told;
                if (told++ < 3)
                    rt_log("ngxfg: watchdog: no frame for 1.5 s; frames handed %llu, copied signal submitted %llu; GPU rendered %llu copied %llu",
                           (unsigned long long)ng.n, (unsigned long long)ng.signaled, (unsigned long long)rv, (unsigned long long)cv);
            }
        }
        // Own the complete job before the producer can fill the next pending slot.
        const NgJob job = ng.job;
        ng.pending = 0, ng.busy = 1;
        plat_cond_broadcast(&ng.cv);
        plat_mutex_unlock(&ng.m);
        ng_run(&job);
        plat_mutex_lock(&ng.m);
        ng.busy = 0;
        plat_cond_broadcast(&ng.cv);
        plat_mutex_unlock(&ng.m);
    }
}

static void ng_free_image(NgImg *im) {
    VkCtx *c = vk_ctx();
    if (im->view) vkDestroyImageView(c->device, im->view, NULL);
    if (im->img) vmaDestroyImage(c->vma, im->img, im->mem);
    memset(im, 0, sizeof *im);
}

static int ng_image(NgImg *im, uint32_t w, uint32_t h, VkFormat f) {
    VkCtx *c = vk_ctx();
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ai, &im->img, &im->mem, NULL) != VK_SUCCESS) return im->img = VK_NULL_HANDLE, 0;
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = im->img, .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                .format = f, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    return vkCreateImageView(c->device, &vi, NULL, &im->view) == VK_SUCCESS;
}

// The presenter and the images for the current swapchain (g_pl held). 0: frames go out the plain way.
static int ng_ready(void) {
    if (!vk_ngxfg_wanted()) return 0;   // the Windows build, or a run that started without DLSS upscaling
    int want = vk_ngxfg_on() && !g_sc_hdr && !vk_fg_active() && !vk_dlssg_active();
    if (!vk_fg_wanted()) vk_dlaa_hudless_wanted = want;   // the HUD-less picture DLSS-G tells the HUD apart by
    if (!want) {   // off (the menu switched it off live): what the presenter holds goes out first
        if (ng.on) ng_drain(), ng.on = 0, rt_log("ngxfg: frame generation off");
        return 0;
    }
    if (ng.on && ng.w == g_sc_ext.width && ng.h == g_sc_ext.height && ng.fmt == g_sc_fmt) return 1;
    VkCtx *c = vk_ctx();
    if (!ng.tried) {
        ng.tried = 1;
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(c->phys, g_sc_fmt, &fp);
        if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
            rt_log("ngxfg: the swapchain format %d is no storage format here: frame generation off", (int)g_sc_fmt);
            return ng.on = 0, ng.tried = 2, 0;
        }
        if (!ng.thread_up) plat_cond_init(&ng.cv);
        VkSemaphoreTypeCreateInfo tl = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
        VkSemaphoreCreateInfo tci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &tl};
        vkCreateSemaphore(c->device, &tci, NULL, &ng.rendered);
        vkCreateSemaphore(c->device, &tci, NULL, &ng.copied);
        VkSemaphoreCreateInfo semi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                       .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = c->gfx_family};
        vkCreateCommandPool(c->device, &pci, NULL, &ng.pool);
        for (int s = 0; s < 2; s++) {
            VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = ng.pool,
                                              .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = NG_MAX + 1};
            vkAllocateCommandBuffers(c->device, &ai, ng.cb[s]);
            for (int i = 0; i <= NG_MAX; i++) vkCreateSemaphore(c->device, &semi, NULL, &ng.acq[s][i]);
        }
        if (!ng.thread_up && !plat_thread_spawn(ng_thread, NULL, "fg present", 0)) {
            rt_log("ngxfg: the presenter thread did not start: frame generation off");
            return ng.on = 0, ng.tried = 2, 0;
        }
        ng.thread_up = 1;
    }
    if (ng.tried == 2) return 0;
    ng_drain();
    present_require(vk_device_wait_idle(), "NGX frame generation image retirement");
    ng_free_image(&ng.bb);
    for (int s = 0; s < 2; s++) {
        ng_free_image(&ng.real[s]);
        for (int i = 0; i < NG_MAX; i++) ng_free_image(&ng.out[s][i]);
    }
    uint32_t w = g_sc_ext.width, h = g_sc_ext.height;
    int ok = ng_image(&ng.bb, w, h, g_sc_fmt);
    for (int s = 0; s < 2 && ok; s++) {
        ok = ng_image(&ng.real[s], w, h, g_sc_fmt);
        for (int i = 0; i < NG_MAX && ok; i++) ok = ng_image(&ng.out[s][i], w, h, g_sc_fmt);
    }
    if (!ok) {
        rt_log("ngxfg: frame generation images (%ux%u) could not be made: frame generation off", w, h);
        return ng.on = 0;
    }
    ng.w = w, ng.h = h, ng.fmt = g_sc_fmt, ng.fresh = 1, ng.on = 1;
    vk_ngxfg_reset();
    rt_log("ngxfg: presenter ready at %ux%u (up to %dx)", w, h, vk_ngxfg_max() + 1);
    return 1;
}

// NX2NAVI_FG_MULT: 2 to 6, or dynamic: only the frames needed to reach the display's refresh rate at the rendered frame
// rate (240 Hz at 60 fps: 4x), between 2x and what the GPU allows, as NVIDIA describes its dynamic multi frame
// generation; a new multiplier must hold for 30 frames before it is taken, so it does not flicker.
static int ng_multiplier(int max) {
    const char *v = cfg_value("NX2NAVI_FG_MULT");
    int m;
    if (!strcmp(v, "dynamic")) {
        static int cur = 2, cand, held;
        double hz = gfx_window_refresh_hz(), ms = (double)ng.interval / 1e6;
        int fit = hz > 0 && ms > 0 ? (int)(hz * ms / 1000.0 + 0.1) : 2;   // shown frames per rendered one that fit
        if (fit < 2) fit = 2;
        if (fit > max + 1) fit = max + 1;
        if (fit != cur) {
            held = fit == cand ? held + 1 : 0, cand = fit;
            if (held >= 30) cur = fit, held = 0;
        } else held = 0;
        m = cur;
    } else {
        m = atoi(v);
    }
    if (m < 2) m = 2;
    if (m > max + 1) m = max + 1;
    return m;
}

// Records the copy of the frame's picture and DLSS-G's frames (the end of present_image's command buffer, dst in
// layout *last). Returns the number of generated frames, -1 when the frame goes out the plain way.
static int ng_record(VkCommandBuffer cb, VkImage dst, VkImageLayout *last, VkPipelineStageFlags2 *stage, VkAccessFlags2 *write,
                     VkImage hudless, const VkNgxImage *alpha, int *set, int gen) {
    if (!ng.on) return -1;
    if (ng.fresh) {   // every image to GENERAL, once
        NgImg *all[1 + 2 + 2 * NG_MAX];
        int k = 0;
        all[k++] = &ng.bb;
        for (int s = 0; s < 2; s++) {
            all[k++] = &ng.real[s];
            for (int i = 0; i < NG_MAX; i++) all[k++] = &ng.out[s][i];
        }
        for (int i = 0; i < k; i++)
            sync_barrier(cb, all[i]->img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, 0,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        ng.fresh = 0;
    }
    *set = (int)((ng.n + 1) & 1);
    // the frame's picture is in dst (ng.bb): to GENERAL for DLSS-G and the copy into this set's rendered-frame image
    sync_barrier(cb, dst, *last, VK_IMAGE_LAYOUT_GENERAL, *stage, *write, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT);
    *last = VK_IMAGE_LAYOUT_GENERAL, *stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, *write = 0;
    int made = 0;
    if (!gen || menu_is_open()) {
        vk_ngxfg_reset();   // generation pauses while the menu is open
    } else {
        int max = vk_ngxfg_max(), want = ng_multiplier(max) - 1;
        VkNgxImage bb = {ng.bb.img, ng.bb.view, ng.fmt, ng.w, ng.h};
        VkNgxImage hl = {hudless, g_hl_view, g_hl_fmt, g_hl_w, g_hl_h};
        VkNgxImage out[NG_MAX], real = {ng.real[*set].img, ng.real[*set].view, ng.fmt, ng.w, ng.h};
        for (int i = 0; i < NG_MAX; i++) out[i] = (VkNgxImage){ng.out[*set][i].img, ng.out[*set][i].view, ng.fmt, ng.w, ng.h};
        int hl_ok = hudless && g_hl_w == ng.w && g_hl_h == ng.h && g_hl_fmt == ng.fmt;
        made = vk_ngxfg_evaluate(cb, &bb, hl_ok ? &hl : NULL, hl_ok ? alpha : NULL, out, want, &real);
    }
    // the rendered frame for the presenter (ng.bb is drawn again by the next frame)
    sync_barrier(cb, ng.real[*set].img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageCopy cp = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                      .extent = {ng.w, ng.h, 1}};
    vkCmdCopyImage(cb, dst, VK_IMAGE_LAYOUT_GENERAL, ng.real[*set].img, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
    sync_barrier(cb, ng.real[*set].img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT);
    *stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT, *write = 0;
    return made;
}

// gfx_window_release_vk proved retirement. Disable the idle watchdog before
// destroying its timelines. The thread stays for a later device (ng_ready).
static void ng_release(void) {
    if (!ng.tried) return;
    plat_mutex_lock(&ng.m);
    ng.on = ng.tried = 0, ng.n = ng.signaled = 0;
    plat_mutex_unlock(&ng.m);
    VkCtx *c = vk_ctx();
    ng_free_image(&ng.bb);
    for (int s = 0; s < 2; s++) {
        ng_free_image(&ng.real[s]);
        for (int i = 0; i < NG_MAX; i++) ng_free_image(&ng.out[s][i]);
    }
    if (ng.rendered) vkDestroySemaphore(c->device, ng.rendered, NULL);
    if (ng.copied) vkDestroySemaphore(c->device, ng.copied, NULL);
    for (int s = 0; s < 2; s++)
        for (int i = 0; i <= NG_MAX; i++)
            if (ng.acq[s][i]) vkDestroySemaphore(c->device, ng.acq[s][i], NULL), ng.acq[s][i] = VK_NULL_HANDLE;
    if (ng.pool) vkDestroyCommandPool(c->device, ng.pool, NULL);   // its command buffers with it
    ng.rendered = ng.copied = VK_NULL_HANDLE, ng.pool = VK_NULL_HANDLE;
}

static uint64_t ng_presented(void) { return ng.on ? __atomic_load_n(&ng.presents, __ATOMIC_RELAXED) : 0; }
static int ng_running(void) { return ng.on; }

// Hands the submitted frame to the presenter (it shows count generated frames, then swapchain image idx).
static void ng_hand(int count, int set, uint64_t n) {
    plat_mutex_lock(&ng.m);
    while (ng.pending) plat_cond_wait(&ng.cv, &ng.m);
    ng.job = (NgJob){.count = count, .set = set, .n = n, .t = plat_now_ns(),
                     .generation = g_sc_generation, .sc = g_sc};
    ng.pending = 1;
    plat_cond_broadcast(&ng.cv);
    plat_mutex_unlock(&ng.m);
}

static VkDlssgConfigResult present_config_prepare(const PendingFlip *flip, const VkDlaaFgSizeUse *temporal,
    const VkDlaaHudImages *hud, VkDlssgConfigPreparation *prepared, VkPresentConfigRetry *retry) {
    if (vk_sync_stopping() || vk_device_is_lost()) return VK_DLSSG_CONFIG_STOPPED;
    VkFfxImage depth = {0}, motion = {0};
    int have = vk_dlaa_fg_describe(temporal, &depth, &motion);
    if (have < 0) return VK_DLSSG_CONFIG_INVALIDATED;
    uint32_t generated = vk_dlssg_config_next_generated(have);
    Nx2SlFgConfigV1 choices[3] = {{.size = sizeof choices[0], .version = NX2SL_FG_MEMORY_ABI_V1,
        .generation = flip->generation, .swapchain = flip->swapchain, .backbuffers = g_sc_n,
        .generated = generated, .mvec_width = motion.width, .mvec_height = motion.height,
        .depth_width = depth.width, .depth_height = depth.height, .color_width = g_sc_ext.width,
        .color_height = g_sc_ext.height, .color = g_sc_fmt, .depth = depth.format, .motion = motion.format}};
    uint32_t count = 1;
    if (generated && have && hud && hud->color.img) {
        choices[count] = choices[0];
        choices[count++].hudless = g_sc_fmt;
        if (hud->alpha.img && vk_dlssg_ui_alpha()) {
            choices[count] = choices[count - 1];
            choices[count].alpha = VK_FORMAT_R16_SFLOAT;
            choices[count++].recomposition = 1;
        }
    }
    VkDlssgConfigResult result = vk_dlssg_config_preflight_variants(choices, count, have, flip->guest_frame,
                                                                  prepared, &retry->peak);
    if (result == VK_DLSSG_CONFIG_DRAIN_REQUIRED &&
        !vk_dlssg_config_drain_eligible(&retry->older_config, &retry->older_graphics))
        rt_fatal("gfx: Streamline configuration drain lost its earlier native owner");
    return result;
}

// The present thread retains this job while retrying. Window work can proceed
// while the prior configuration's actual queues retire.
VkResult vk_present_config_drain(const VkPresentConfigRetry *retry) {
    VkCtx *c = vk_ctx();
    VkResult r = vkWaitForFences(c->device, 1, &retry->older_graphics, VK_TRUE, 5000000000ull);
    if (r != VK_SUCCESS) return r;
    r = vk_device_wait_idle();
    if (r != VK_SUCCESS) return r;
    plat_rmutex_lock(&g_pl);
    uint64_t use;
    VkFence fence;
    if (vk_dlssg_config_drain_eligible(&use, &fence) && use == retry->older_config && fence == retry->older_graphics)
        vk_dlssg_config_progress(c->device, 1);
    plat_rmutex_unlock(&g_pl);
    return VK_SUCCESS;
}

static VkPresentSourceHandoff present_image(VkImage src, VkImageLayout layout, VkFormat fmt,
                                            int w, int h, const int crop[4], int defer, uint64_t frame,
                                            VkTexturePresentSource **owner, VkDlaaFgSizeUse **temporal,
                                            VkDlaaHudSource **hud, VkPresentConfigRetry *retry) {
    VkDlssgConfigPreparation prepared = {0};
    VkPresentConfigRetry direct_retry = {0};
    const VkDlaaHudImages *hud_images = NULL;
    VkPresentSourceHandoff source = VK_PRESENT_SOURCE_UNSENT;
    VkTexturePresentSource *completed_source = NULL;
    VkDlaaFgSizeUse *completed_temporal[FRAMES] = {0};
    VkDlaaHudSource *completed_hud = NULL;
    int general = layout == VK_IMAGE_LAYOUT_GENERAL;
    {   // the presented image's format, when it changes (an HDR picture comes in another one)
        static VkFormat last = VK_FORMAT_UNDEFINED;
        if (src && fmt != last) last = fmt, rt_log("gfx: the game presents %dx%d images of format %d", w, h, (int)fmt);
    }
    if (vk_device_is_lost()) return source;
    gfx_window_start();
    VkCtx *c = vk_ctx();
    plat_rmutex_lock(&g_pl);
    if (g_compose_state != COMPOSE_KNOWN) goto out;
    flip_locked();
    if ((!g_sc || plat_cas32(&g_dirty, 0, 0)) && !wsys_minimized()) create_swapchain(completed_temporal);
    if (wsys_minimized() || !g_sc || plat_cas32(&g_sc_blocked, 0, 0)) goto out;

    unsigned slot = g_frame % FRAMES;
    g_hudless_slot = slot;
    VkResult r = vkWaitForFences(c->device, 1, &g_fence[slot], VK_TRUE, 5000000000ull);
    if (r != VK_SUCCESS) {
        vk_device_lost_check(r, "present slot reuse");
        rt_log("gfx: present fence wait failed (%d)", (int)r);
        goto out;
    }
    completed_source = g_slot_source[slot];
    g_slot_source[slot] = NULL;
    present_take_completed_temporal(slot, &completed_temporal[slot]);
    completed_hud = g_slot_hud[slot];
    g_slot_hud[slot] = NULL;
    if (vk_dlssg_has_input_receipts()) vk_dlssg_config_progress(c->device, 0);
    r = present_inputs_ready(slot);
    if (r != VK_SUCCESS) {
        vk_device_lost_check(r, "Streamline present input retirement");
        if (r != VK_NOT_READY) rt_log("presentation: Streamline input wait failed (%d)", (int)r);
        goto out;
    }
    const int separate = present_split();
    // A native swapchain already owns immutable display images until acquisition. Compose
    // directly into one and present on the separate queue, avoiding an extra output copy.
    // The SDK swapchain keeps staging even while FG is temporarily off: its scheduling
    // must never extend the game's completion timeline through its acquisition waits.
    const int split = separate && g_sc_path == PRESENT_DLSSG;
    if (split && (!present_slot_ready(slot) || !present_slot_image(slot))) goto out;
    uint32_t idx = 0;
    // NGX frame generation: the frame is composed into ng.bb and the presenter shows it (ng_show); no acquire here
    const int ngp = ng_ready();
    if (plat_cas32(&g_sc_blocked, 0, 0)) goto out;
    PendingFlip flip = {.generation = g_sc_generation, .swapchain = g_sc, .path = g_sc_path, .slot = slot,
                        .queue = separate ? c->present_queue : c->gfx_queue, .guest_frame = frame};
    hud_images = hud ? vk_dlaa_hud_source_images(*hud, frame) : NULL;
    if (src && frame && flip.path == PRESENT_DLSSG && vk_dlssg_has_input_receipts()) {
        if (!retry) retry = &direct_retry;
        VkDlssgConfigResult config = present_config_prepare(&flip, temporal ? *temporal : NULL, hud_images,
                                                           &prepared, retry);
        if (config == VK_DLSSG_CONFIG_PENDING || config == VK_DLSSG_CONFIG_DRAIN_REQUIRED) {
            source = VK_PRESENT_SOURCE_PENDING;
            goto out;
        }
        if (config == VK_DLSSG_CONFIG_STOPPED) goto out;
        if (config != VK_DLSSG_CONFIG_READY && config != VK_DLSSG_CONFIG_COMPAT)
            rt_fatal("gfx: Streamline config preflight failed for frame %llu (%d)", (unsigned long long)frame, (int)config);
    }
    r = (ngp || split) ? VK_SUCCESS : wsi_acquire(c, &flip, g_acquire[slot], VK_NULL_HANDLE, &idx);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        wsi_block_generation();
        if (src && retry && flip.path == PRESENT_DLSSG && vk_dlssg_has_input_receipts()) {
            retry->reprepare = 1;
            source = VK_PRESENT_SOURCE_PENDING;
            goto out;
        }
        if (!create_swapchain(completed_temporal)) goto out;
        flip = (PendingFlip){.generation = g_sc_generation, .swapchain = g_sc, .path = g_sc_path, .slot = slot,
                             .queue = present_split() ? c->present_queue : c->gfx_queue, .guest_frame = frame};
        if (src && frame && flip.path == PRESENT_DLSSG && vk_dlssg_has_input_receipts()) {
            if (retry) retry->reprepare = 1;
            source = VK_PRESENT_SOURCE_PENDING;
            goto out;
        }
        r = wsi_acquire(c, &flip, g_acquire[slot], VK_NULL_HANDLE, &idx);
    }
    vk_device_lost_check(r, "the acquire of a swapchain image");
    // not ready: skip the frame. Acquires that keep timing out (1 s each) mean a swapchain nothing will free an
    // image of again: three in a row rebuild it, where the picture used to stop for good.
    static uint32_t n_timeouts;
    if (r == VK_TIMEOUT || r == VK_NOT_READY) {
        if (++n_timeouts >= 3) {
            rt_log("gfx: no swapchain image for %u acquires (%d): the swapchain is made again", n_timeouts, (int)r);
            n_timeouts = 0;
            plat_xchg32(&g_dirty, 1);
        }
    } else {
        n_timeouts = 0;
    }
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
        if (r != VK_TIMEOUT && r != VK_NOT_READY) wsi_block_generation();
        goto out;
    }
    if (r == VK_SUBOPTIMAL_KHR) plat_xchg32(&g_dirty, 1);
    if (!ngp && !split) wsi_bind_image(&flip, idx);
    if (g_sc_hdr) vk_hdr_metadata(g_sc, 0);   // a changed peak or white goes to the display
    if (g_th_pending[slot]) {   // that fence has signalled: the thumbnail its submission wrote is complete
        g_th_tag[g_th_pending[slot] % (uint64_t)g_th_n] = g_th_pending[slot];
        g_th_pending[slot] = 0;
    }

    int cx = 0, cy = 0, cw = w, ch = h;
    if (crop && crop[2] > 0 && crop[3] > 0) { cx = crop[0]; cy = crop[1]; cw = crop[2]; ch = crop[3]; }
    if (cx < 0) cx = 0;
    if (cy < 0) cy = 0;
    if (cx + cw > w) cw = w - cx;
    if (cy + ch > h) ch = h - cy;
    if (cw <= 0 || ch <= 0) cx = cy = 0, cw = w, ch = h;   // a crop outside the image (the game's nvnWindowSetCrop):
                                                             // the whole image, not an inverted blit rectangle
    if (src && cw > 0 && ch > 0)
        __atomic_store_n(&g_game_picture_size, ((uint64_t)cw << 32) | (unsigned)ch, __ATOMIC_RELAXED);
    window_settings_live();
    cam_speed_tick();
    // The menu overlay: laid out before recording (its first frame starts the ImGui back end and uploads the font)
    const char *menu_shot = NULL;
    MenuTarget mt = {c->instance, c->phys, c->device, c->gfx_family, c->gfx_queue,
                     g_sc_hdr ? VK_FORMAT_R8G8B8A8_UNORM : g_sc_fmt, g_sc_ext.width,
                     g_sc_ext.height, cw, ch, g_scaler};
    // without a source image (gfx_present_boot): a black frame with the boot screen
    VkImageView dst_view = split ? g_present_slots[slot].view : ngp ? ng.bb.view : g_sc_view[idx];
    int overlay = dst_view &&
                  (src ? menu_overlay_frame(__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), &mt, &menu_shot)
                       : menu_boot_frame(&mt));
    // NX2NAVI_SHOT_EVERY=<s> with NX2NAVI_SHOT_DIR=<dir>: the presented frame as shot_<n>.png every s seconds
    // (a check of what the window shows, for runs nobody watches)
    static char shot_path[1100];
    {
        static double every = -1;
        static uint64_t next_ns;
        static int n;
        if (every < 0) {
            const char *e = cfg_get("NX2NAVI_SHOT_EVERY");
            every = e ? atof(e) : 0;
            const char *after = cfg_get("NX2NAVI_SHOT_AFTER");   // seconds after the start before the first shot
            next_ns = plat_now_ns() + (uint64_t)((every + (after ? atof(after) : 0)) * 1e9);
        }
        const char *dir = cfg_get("NX2NAVI_SHOT_DIR");
        if (!menu_shot && every > 0 && dir && plat_now_ns() >= next_ns && n < 200) {
            snprintf(shot_path, sizeof shot_path, "%s/shot_%03d.png", dir, n++);
            menu_shot = shot_path;
            next_ns = plat_now_ns() + (uint64_t)(every * 1e9);
        }
    }
    int cap = (g_capture || menu_shot) && ensure_capture_buffer();
    if (cap) g_cap_valid = 0;
    int thumb = src && g_th_n && ensure_thumbs();

    VkCommandBuffer cb = g_cb[slot];
    present_require(vkResetCommandBuffer(cb, 0), "reset presentation commands");
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    present_require(vkBeginCommandBuffer(cb, &bi), "begin presentation commands");
    uint64_t crumb = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    vk_crumb(cb, VK_CRUMB_PRESENT, crumb);
    VkImage dst = split ? g_present_slots[slot].image : ngp ? ng.bb.img : g_sc_img[idx];
    sync_barrier(cb, dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 0,
                 VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    const int rect[4] = {cx, cy, cw, ch};
    int to[4];   // where the picture goes in the swapchain image: {x, y, w, h}
    gfx_fit_rect(cw, ch, (int)g_sc_ext.width, (int)g_sc_ext.height, to);
    if (src && g_sc_ext.width && g_sc_ext.height) {
        for (int i = 0; i < 4; i++) g_pic[i] = (float)to[i] / (float)(i & 1 ? g_sc_ext.height : g_sc_ext.width);
        __atomic_store_n(&g_pic_ok, 1, __ATOMIC_RELEASE);
    }
    if (!src || to[2] != (int)g_sc_ext.width || to[3] != (int)g_sc_ext.height) {   // black bars beside or above and below it
        VkClearColorValue black = {.float32 = {0, 0, 0, 1}};
        VkImageSubresourceRange all = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &all);
        sync_barrier(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT);
    }
    uint32_t picture_effects = 0;
    int picture_replaced = 0;
    int fsr = put_picture(cb, slot, src, fmt, w, h, rect, dst, to, general, &picture_effects);
    if (src) g_scaler = fsr == 2 ? 3 : fsr ? 2 : cw != to[2] || ch != to[3];
    // Frame generation without the HUD in its generated frames: the picture as it would be without the HUD (vk_dlaa.c,
    // DLSS's or FSR 3.1's output before the HUD was drawn over it) goes through the same steps into an image of the
    // swapchain's size, which frame generation compares with the presented one to find the HUD and leave it unwarped
    VkImage hudless = src ? hudless_record(cb, slot, w, h, rect, to, general, hud_images) : VK_NULL_HANDLE;
    if (hudless) {
        flip.hud_generation = hud_images->generation;
        flip.hud_exact = hud_images->exact;
    }
    static int show_hl = -1;   // NX2NAVI_FG_HUDLESS_SHOW=1 (test): the HUD-less picture on screen instead
    if (show_hl < 0) show_hl = cfg_get("NX2NAVI_FG_HUDLESS_SHOW") && !strcmp(cfg_get("NX2NAVI_FG_HUDLESS_SHOW"), "1");
    if (hudless && show_hl) {
        picture_replaced = 1;
        VkImageCopy cp = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .extent = {g_sc_ext.width, g_sc_ext.height, 1}};
        vkCmdCopyImage(cb, hudless, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
        sync_barrier(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT);
    }
    int movie = !g_sc_hdr && src && movie_overlay(cb, slot, dst);   // a movie frame over the game image (SDR swapchains)
    if (movie) picture_effects |= GFX_PICTURE_MOVIE_LAYER;
    // last: the image's layout, and the stage and access of its last write, as each step leaves it
    VkImageLayout last = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    VkPipelineStageFlags2 last_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    VkAccessFlags2 last_write = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    // HDR10: the menu goes into vk_hdr.c's RGBA8 image, which is then blended onto the swapchain image at the HUD's white
    VkImageView menu_view = overlay && g_sc_hdr ? vk_hdr_ui_target(cb, g_sc_ext.width, g_sc_ext.height) : VK_NULL_HANDLE;
    if (overlay && g_sc_hdr && !menu_view) overlay = 0;
    VkNgxImage alpha = {0};
    if (hudless && (vk_dlssg_ui_alpha() || ng_running()) && !movie && !show_hl)
        alpha = vk_dlaa_present_alpha(cb, g_sc_ext, rect, to, hud_images, slot);
    VkPipeline overlay_pipeline = overlay && alpha.img ? menu_overlay_hud_pipeline() : VK_NULL_HANDLE;
    if (overlay && !overlay_pipeline) alpha = (VkNgxImage){0};
    if (overlay) {
        picture_effects |= GFX_PICTURE_MENU_LAYER;
        sync_barrier(cb, dst, last, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, last_stage, last_write,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                                        .imageView = menu_view ? menu_view : dst_view,
                                        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                        .loadOp = menu_view ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
                                        .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
        VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, g_sc_ext}, .layerCount = 1,
                              .colorAttachmentCount = 1, .pColorAttachments = &ca};
        VkRenderingAttachmentInfo attachments[2] = {ca, ca};
        if (overlay_pipeline) {
            attachments[1].imageView = alpha.view;
            attachments[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            ri.pColorAttachments = attachments;
            ri.colorAttachmentCount = 2;
        }
        vkCmdBeginRendering(cb, &ri);
        menu_overlay_draw_with_pipeline(cb, overlay_pipeline);
        vkCmdEndRendering(cb);
        if (alpha.img)
            sync_barrier(cb, alpha.img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT);
        if (menu_view) vk_hdr_ui_compose(cb, dst_view, g_sc_fmt, g_sc_ext.width, g_sc_ext.height);
        last = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        last_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        last_write = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    }
    static int show_alpha = -1;
    if (show_alpha < 0) {
        const char *value = cfg_get("NX2NAVI_FG_UI_ALPHA_SHOW");
        show_alpha = value && !strcmp(value, "1");
    }
    if (alpha.img && show_alpha) {
        picture_replaced = 1;
        VkImageBlit blit = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffsets = {{0, 0, 0}, {(int)g_sc_ext.width, (int)g_sc_ext.height, 1}},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffsets = {{0, 0, 0}, {(int)g_sc_ext.width, (int)g_sc_ext.height, 1}}};
        sync_barrier(cb, dst, last, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, last_stage, last_write,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        vkCmdBlitImage(cb, alpha.img, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
        last = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        last_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        last_write = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        alpha = (VkNgxImage){0};
    }
    if (cap) {
        sync_barrier(cb, dst, last, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, last_stage, last_write,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        VkBufferImageCopy bc = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                .imageExtent = {g_sc_ext.width, g_sc_ext.height, 1}};
        vkCmdCopyImageToBuffer(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_cap_buf, 1, &bc);
        last = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        last_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        last_write = 0;
    }
    if (thumb) {
        if (last != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
            sync_barrier(cb, dst, last, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, last_stage, last_write,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        last = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        last_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        last_write = 0;
        record_thumb(cb, dst, slot);
    }
    vk_dlaa_fg_compose_select(temporal ? *temporal : NULL);
    if (src) {
        // Without async interpolation, same-size reuse is ordered on the game queue. CPU destruction
        // still needs the exact SDK finish, including when composition is rejected after configuration.
        present_ffx_hold_hudless(&flip, hudless);
        vk_fg_frame(cb, g_sc, g_sc_fmt, g_sc_ext, hudless, g_hl_view, to);
    }
    if (src) output_rate();
    if ((!src || !frame) && flip.path == PRESENT_DLSSG && vk_dlssg_has_input_receipts()) {
        // Boot and diagnostic pictures without a frame identity do not configure frame generation.
        flip.sdk_path = SDK_FLIP_FALLBACK;
        flip.sdk_frame = flip.guest_frame;
    } else if (src && flip.path == PRESENT_DLSSG && vk_dlssg_has_input_receipts()) {
        const VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        Nx2SlImage hl = {hudless, g_hl_view, g_hl_fmt, g_hl_w, g_hl_h, VK_IMAGE_LAYOUT_GENERAL, usage};
        Nx2SlImage ui = {alpha.img, alpha.view, alpha.fmt, alpha.w, alpha.h, VK_IMAGE_LAYOUT_GENERAL,
                         usage | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
        int tag_hudless = !movie && hl.image && hl.view;
        VkDlssgConfigResult config = present_inputs_tag(&flip, cb, g_sc_ext, to, tag_hudless ? &hl : NULL,
            tag_hudless && ui.image && ui.view ? &ui : NULL, &prepared);
        if (config != VK_DLSSG_CONFIG_READY && config != VK_DLSSG_CONFIG_COMPAT) {
            vk_dlaa_fg_compose_select(NULL);
            wsi_block_generation();
            if (!vk_device_is_lost()) {
                vk_dlaa_present_alpha_unrecorded(slot);
                if (!split && !ngp) wsi_settle_acquire(c->gfx_queue, g_acquire[slot], VK_NULL_HANDLE);
            }
            if (config == VK_DLSSG_CONFIG_STOPPED) goto out;
            if (config == VK_DLSSG_CONFIG_INVALIDATED && retry) {
                retry->reprepare = 1;
                source = VK_PRESENT_SOURCE_PENDING;
                goto out;
            }
            rt_fatal("gfx: retained Streamline config changed at native tag (%d)", (int)config);
        }
    } else if (src) {
        vk_dlssg_frame(cb, g_sc_ext, to, movie ? VK_NULL_HANDLE : hudless, g_hl_view, g_sc_fmt, alpha.img, alpha.view);
    }
    // through the presenter: the picture stays in ng.bb, and DLSS Frame Generation's frames and the copy for the
    // presenter are recorded; otherwise the swapchain image is ready to present
    int ng_set = 0;
    const int ng_fresh = ng.fresh;
    const int ng_count = ngp ? ng_record(cb, dst, &last, &last_stage, &last_write, movie ? VK_NULL_HANDLE : hudless, alpha.img ? &alpha : NULL, &ng_set, src != VK_NULL_HANDLE) : -1;
    vk_dlaa_fg_compose_select(NULL);
    const int via_ng = ng_count >= 0;
    if (ngp && !via_ng) rt_fatal("gfx: NGX composition has no presenter");
    if (!via_ng) sync_barrier(cb, dst, last,
        split ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        last_stage, last_write, VK_PIPELINE_STAGE_2_NONE, 0);
    vk_crumb(cb, VK_CRUMB_PRESENT_END, crumb);
    present_require(vkEndCommandBuffer(cb), "end presentation commands");

    VkSemaphoreSubmitInfo wait = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = g_acquire[slot],
                                  .stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT};
    VkSemaphoreSubmitInfo sig = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = flip.render_done,
                                 .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkCommandBufferSubmitInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .waitSemaphoreInfoCount = 1, .pWaitSemaphoreInfos = &wait,
                        .commandBufferInfoCount = 1, .pCommandBufferInfos = &cbi, .signalSemaphoreInfoCount = 1,
                        .pSignalSemaphoreInfos = &sig};
    if (split) {
        g_present_slots[slot].ready_value = ++g_snapshot_value;
        si.waitSemaphoreInfoCount = 0;
        sig.semaphore = g_snapshot_ready;
        sig.value = g_snapshot_value;
    }
    const uint64_t ng_n = ng.n + 1;   // through the presenter: this frame's number on its timelines
    if (via_ng) {
        // No acquire to wait for. The frame writes one of the presenter's two image sets: it waits on the GPU until
        // the presenter is done with the frame before last, which used that set, and that signal is submitted first
        // (ng_wait_signaled). It signals the timeline the presenter's copies wait for instead of a render-done semaphore.
        wait = (VkSemaphoreSubmitInfo){.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = ng.copied,
                                       .value = ng_n > 2 ? ng_n - 2 : 0, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
        sig = (VkSemaphoreSubmitInfo){.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = ng.rendered,
                                      .value = ng_n, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
        si.waitSemaphoreInfoCount = ng_n > 2 ? 1u : 0u;
        if (ng_n > 2) ng_wait_signaled(ng_n - 2);
    }
    present_require(vkResetFences(c->device, 1, &g_fence[slot]), "reset presentation fence");
    vk_queue_lock();
    r = vkQueueSubmit2(c->gfx_queue, 1, &si, g_fence[slot]);
    vk_queue_unlock();
    // This submit is the last native use of src. Only composed images are
    // handed to the display queue or SDK below. Keep uncertain use pinned too.
    const int unsent = wsi_no_submission(r);
    if (r == VK_SUCCESS && !via_ng && flip.path == PRESENT_NATIVE &&
        src && frame && owner && w > 0 && h > 0) {
        const GfxPresentPictureSource input = vk_texture_present_picture(*owner);
        flip.picture.size = (GfxOrdinaryPresentSize){
            .generation = flip.generation, .frame = frame, .guest_texture_serial = input.window_texture_serial,
            .source_width = (uint32_t)w, .source_height = (uint32_t)h,
            .output_width = g_sc_ext.width, .output_height = g_sc_ext.height};
        if (!picture_replaced && input.image == (uint64_t)(uintptr_t)src && input.format == (uint32_t)fmt &&
            input.width == (uint32_t)w && input.height == (uint32_t)h &&
            dst == g_sc_img[flip.index] && g_ordinary_picture_sequence != UINT64_MAX) {
            flip.picture.source = input;
            flip.picture.composition = ++g_ordinary_picture_sequence;
            flip.picture.image = (uint64_t)(uintptr_t)dst;
            flip.picture.image_index = flip.index;
            flip.picture.format = g_sc_fmt;
            flip.picture.effects = picture_effects;
            memcpy(flip.picture.source_rect, rect, sizeof flip.picture.source_rect);
            memcpy(flip.picture.output_rect, to, sizeof flip.picture.output_rect);
        }
    }
    if (!unsent && owner) {
        g_slot_source[slot] = *owner;
        *owner = NULL;
    }
    if (!unsent && temporal && *temporal && vk_dlaa_fg_compose_consumed(*temporal)) {
        g_slot_temporal[slot] = *temporal;
        *temporal = NULL;
    }
    if (!unsent && hud && *hud && hudless) {
        g_slot_hud[slot] = *hud;
        *hud = NULL;
    }
    if (r == VK_SUCCESS) {
        source = VK_PRESENT_SOURCE_GFX_SUBMITTED;
        if (via_ng) {
            ng.n = ng_n;
            ng_hand(ng_count, ng_set, ng_n);
        } else {
            flip.phase = split ? FLIP_COPY : FLIP_READY;
            g_flip = flip;
            if (!defer) flip_locked();
        }
    } else {
        present_inputs_abandon(&flip, unsent);
        vk_device_lost_check(r, "the present's submit");
        rt_log("gfx: present submit failed (%d)", (int)r);
        g_th_pending[slot] = 0;
        wsi_block_generation();
        if (!unsent) {
            g_compose_state = via_ng || split ? COMPOSE_UNCERTAIN : COMPOSE_UNCERTAIN_ACQUIRE;
            source = VK_PRESENT_SOURCE_UNCERTAIN;
            goto out;
        }
        if (via_ng) ng.fresh = ng_fresh;
        wsi_settle_acquire(c->gfx_queue, via_ng || split ? VK_NULL_HANDLE : g_acquire[slot], g_fence[slot]);
    }
    g_frame++;
    if (cap && source == VK_PRESENT_SOURCE_GFX_SUBMITTED) {
        r = vkWaitForFences(c->device, 1, &g_fence[slot], VK_TRUE, 5000000000ull);
        if (r == VK_SUCCESS) r = vmaInvalidateAllocation(c->vma, g_cap_alloc, 0, VK_WHOLE_SIZE);
        vk_device_lost_check(r, "presentation capture");
        if (r == VK_SUCCESS) {
            g_cap_w = (int)g_sc_ext.width;
            g_cap_h = (int)g_sc_ext.height;
            g_cap_valid = 1;
            if (menu_shot) save_capture(menu_shot);
        }
    }
out:
    if (prepared.count || prepared.use) vk_dlssg_config_cancel(&prepared);
    plat_rmutex_unlock(&g_pl);
    if (completed_source) vk_texture_present_release(completed_source);
    present_release_completed_temporal(completed_temporal);
    vk_dlaa_hud_source_release(completed_hud);
    return source;
}

void vk_present_image(VkImage src, VkImageLayout layout, VkFormat fmt, int w, int h, const int crop[4]) {
    if (present_image(src, layout, fmt, w, h, crop, 0, vk_dlssg_current_frame(), NULL, NULL, NULL, NULL) ==
        VK_PRESENT_SOURCE_UNCERTAIN)
        rt_fatal("gfx: uncertain presentation source submission");
}

VkPresentSourceHandoff vk_present_blit(VkImage src, VkImageLayout layout, VkFormat fmt, int w, int h,
                                       const int crop[4], uint64_t frame, VkTexturePresentSource **owner,
                                       VkDlaaFgSizeUse **temporal, VkDlaaHudSource **hud, VkPresentConfigRetry *retry) {
    return present_image(src, layout, fmt, w, h, crop, 1, frame, owner, temporal, hud, retry);
}

// Required reconstruction admission can retire a finished composition while the next scene is parked.
void vk_present_retire_temporal_inputs(void) {
    if (vk_device_is_lost() || !plat_rmutex_trylock(&g_pl)) return;
    VkDlaaFgSizeUse *completed[FRAMES] = {0};
    VkDlaaHudSource *hud[FRAMES] = {0};
    if (g_started && g_compose_state == COMPOSE_KNOWN) {
        for (unsigned i = 0; i < FRAMES; ++i) {
            if ((!g_slot_temporal[i] && !g_slot_hud[i]) ||
                vkGetFenceStatus(vk_ctx()->device, g_fence[i]) != VK_SUCCESS) continue;
            completed[i] = g_slot_temporal[i];
            g_slot_temporal[i] = NULL;
            hud[i] = g_slot_hud[i];
            g_slot_hud[i] = NULL;
        }
    }
    plat_rmutex_unlock(&g_pl);
    for (unsigned i = 0; i < FRAMES; ++i) {
        if (completed[i]) vk_dlaa_fg_release(completed[i]);
        vk_dlaa_hud_source_release(hud[i]);
    }
}

// A black frame with the boot screen; there is no borrowed game source.
void gfx_present_boot(void) { vk_present_image(VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_FORMAT_UNDEFINED, 1920, 1080, NULL); }

int vk_present_flip(void) {
    if (vk_device_is_lost()) return 0;
    plat_rmutex_lock(&g_pl);
    int shown = flip_locked();
    plat_rmutex_unlock(&g_pl);
    return shown;
}

static int ordinary_present_current_locked(void) {
    return g_sc && g_sc_path == PRESENT_NATIVE && !vk_device_is_lost() && !wsys_minimized() &&
        !plat_cas32(&g_sc_blocked, 0, 0) && !plat_cas32(&g_dirty, 0, 0) &&
        g_ordinary_picture.size.generation == g_sc_generation;
}

int gfx_ordinary_picture(GfxOrdinaryPicture *picture) {
    plat_rmutex_lock(&g_pl);
    const int current = ordinary_present_current_locked() && g_ordinary_picture.composition;
    *picture = current ? g_ordinary_picture : (GfxOrdinaryPicture){0};
    plat_rmutex_unlock(&g_pl);
    return current;
}

int gfx_ordinary_present_size(GfxOrdinaryPresentSize *size) {
    plat_rmutex_lock(&g_pl);
    const int current = ordinary_present_current_locked() && g_ordinary_picture.size.frame &&
        g_ordinary_picture.size.guest_texture_serial;
    *size = current ? g_ordinary_picture.size : (GfxOrdinaryPresentSize){0};
    plat_rmutex_unlock(&g_pl);
    return current;
}

void gfx_window_resize(int cw, int chh) {
    gfx_window_start();
    wsys_resize(cw, chh);
    VkDlaaFgSizeUse *completed_temporal[FRAMES] = {0};
    plat_rmutex_lock(&g_pl);
    create_swapchain(completed_temporal);
    plat_rmutex_unlock(&g_pl);
    present_release_completed_temporal(completed_temporal);
}

// Called by vk_shutdown before the device goes away; the window and its thread stay.
void gfx_window_release_vk(void) {
    VkTexturePresentSource *completed[FRAMES] = {0};
    VkDlaaFgSizeUse *temporal[FRAMES] = {0};
    VkDlaaHudSource *hud[FRAMES] = {0};
    VkCtx *c = vk_ctx();
    plat_rmutex_lock(&g_pl);
    if (c && g_started) {
        const int release_sdk = vk_dlssg_has_input_receipts();
        present_require(present_retire_generation(release_sdk), "presentation release retirement");
        if (g_compose_state == COMPOSE_UNCERTAIN_ACQUIRE)
            rt_fatal("gfx: no completion proof for an uncertain swapchain acquisition wait");
        memcpy(completed, g_slot_source, sizeof completed);
        memset(g_slot_source, 0, sizeof g_slot_source);
        memcpy(temporal, g_slot_temporal, sizeof temporal);
        memset(g_slot_temporal, 0, sizeof g_slot_temporal);
        memcpy(hud, g_slot_hud, sizeof hud);
        memset(g_slot_hud, 0, sizeof g_slot_hud);
        g_compose_state = COMPOSE_KNOWN;
        g_flip = (PendingFlip){0};
        ng_release();
        if (!g_sc && vk_fg_active()) vk_fg_release();
        drop_swapchain_locked(c, release_sdk);
        menu_vk_release();
        vk_fsr_release();
        vk_color_release();
        vk_hdr_release();
        vk_contact_release();
        free_capture();
        free_thumbs();
        for (int i = 0; i < FRAMES; i++) {
            vkDestroySemaphore(c->device, g_acquire[i], NULL);
            vkDestroyFence(c->device, g_fence[i], NULL);
            vkDestroyFence(c->device, g_present_slots[i].copied, NULL);
            vkDestroyFence(c->device, g_present_slots[i].acquired, NULL);
            present_slot_drop(&g_present_slots[i]);
            hudless_drop(i);
        }
        vkDestroySemaphore(c->device, g_snapshot_ready, NULL);
        memset(g_present_slots, 0, sizeof g_present_slots);
        g_snapshot_ready = VK_NULL_HANDLE;
        g_snapshot_value = 0;
        if (g_pool) vkDestroyCommandPool(c->device, g_pool, NULL);
        if (g_surface) vkDestroySurfaceKHR(c->instance, g_surface, NULL);
        g_surface = VK_NULL_HANDLE; g_pool = VK_NULL_HANDLE;
        g_started = 0;     // the window and its thread stay; the next gfx_window_start makes a new surface
    }
    plat_rmutex_unlock(&g_pl);
    for (unsigned i = 0; i < FRAMES; ++i) {
        if (completed[i]) vk_texture_present_release(completed[i]);
        if (temporal[i]) vk_dlaa_fg_release(temporal[i]);
        vk_dlaa_hud_source_release(hud[i]);
    }
}

void gfx_window_start(void) {
    plat_rmutex_lock(&g_start_lock);
    if (!g_started) {
        if (!vk_init()) rt_fatal("gfx: no usable Vulkan device (see the vk: lines above)");
        VkCtx *c = vk_ctx();
        static int opened;
        if (!opened && !(opened = wsys_open())) rt_fatal("gfx: window creation failed");
        g_surface = wsys_surface(c->instance);
        if (!g_surface) rt_fatal("gfx: no Vulkan surface for the window");

        VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                       .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = c->gfx_family};
        present_require(vkCreateCommandPool(c->device, &pci, NULL, &g_pool), "create presentation command pool");
        VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g_pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = FRAMES};
        present_require(vkAllocateCommandBuffers(c->device, &ai, g_cb), "allocate composition commands");
        VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT};
        VkSemaphoreCreateInfo semi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        for (int i = 0; i < FRAMES; i++) {
            present_require(vkCreateFence(c->device, &fi, NULL, &g_fence[i]), "create presentation fence");
            present_require(vkCreateSemaphore(c->device, &semi, NULL, &g_acquire[i]), "create acquire semaphore");
            present_require(vkCreateFence(c->device, &fi, NULL, &g_present_slots[i].copied), "create display copy fence");
            present_require(vkCreateFence(c->device, &fi, NULL, &g_present_slots[i].acquired), "create acquisition diagnostic fence");
            VkCommandBufferAllocateInfo copy_ai = ai;
            copy_ai.commandBufferCount = 1;
            present_require(vkAllocateCommandBuffers(c->device, &copy_ai, &g_present_slots[i].copy), "allocate display copy commands");
        }
        VkSemaphoreTypeCreateInfo ti = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
        VkSemaphoreCreateInfo ready_ci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &ti};
        if (vkCreateSemaphore(c->device, &ready_ci, NULL, &g_snapshot_ready) != VK_SUCCESS)
            rt_fatal("gfx: cannot create the presentation handoff timeline");
        g_frame = 0;
        g_started = 1;
        plat_rmutex_lock(&g_pl);
        create_swapchain(NULL); // presentation slots are empty at startup
        plat_rmutex_unlock(&g_pl);
        int cw = 0, ch = 0;
        wsys_client_size(&cw, &ch);
        rt_log("gfx: window %dx%d at DPI %u, swapchain %ux%u format %d", cw, ch, gfx_window_dpi(), g_sc_ext.width,
               g_sc_ext.height, (int)g_sc_fmt);
    }
    plat_rmutex_unlock(&g_start_lock);
    vsync_ensure();
}

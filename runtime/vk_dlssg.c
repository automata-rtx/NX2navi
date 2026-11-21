// DLSS Frame Generation (NX2NAVI_FRAMEGEN = dlss; applies after a restart) through nx2navi_dlssg.dll
// (runtime/dlssg/nx2_sl.cpp), which drives NVIDIA Streamline. Like AMD's frame generation (vk_fg.cpp) it needs DLSS or
// FSR 3.1 upscaling on, whose depth, motion vectors and camera it hands over, and an RTX 40 or 50 GPU.
//
// The DLL and Streamline (sl.interposer.dll and its plugins, nvngx_dlssg.dll) sit next to the game. The instance and
// device are created through Streamline's proxies (vk_device.c), so it can add the extensions, features and queues
// DLSS-G needs; the present path calls Streamline's swapchain functions (gfx_present.c), and every present records the
// frame's constants and tags first (vk_dlssg_frame). Generation pauses while the menu is open.
//
// Multi frame generation: NX2NAVI_FRAMEGEN_MULTIPLIER (2, 3 or 4, live) is how many frames the display gets per
// rendered frame, so one, two or three generated ones. Streamline says how many this GPU and driver can generate
// (numFramesToGenerateMax: one where only 2x runs, three on an RTX 50); a higher setting is clamped to it with a log
// line. frame_rate.c divides the frame limit by the multiplier in effect (vk_dlssg_multiplier), so the display gets the
// limit and the game renders a half, a third or a quarter of it.
//
// DLSS-G needs Reflex, whose low-latency mode paces the game (NX2NAVI_REFLEX, default 1; 0 off, 2 on with boost): the
// 1.1.1 sleeps before the engine dispatches the update that samples the pad. Image acquisition supplies a fallback.
// Simulation, host submission, and presentation use the frame number of the image acquired for that update.
// The mode is the swapchain's, so it is set again after each one (vk_dlssg_create_swapchain).
#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "dispatch.h"
#include "host.h"
#include "resource_core_budget.h"
#include "vk.h"
#include "vk_dlaa.h"
#include "vk_dlssg.h"
#include "dlssg/nx2_sl.h"
#include "ui/menu.h"

static struct {
    int wanted, loaded, active, released;
    int fail;            // DLSSG_NO_FILES or DLSSG_UNAVAILABLE when it was wanted and does not run (vk_dlssg_status)
    int frames_started;
    int mode, pending;   // generation as the driver has it, and the frames since the host asked for the other state
    int shown_mult;      // vk_dlssg_multiplier: the frames per game frame as last handed to the DLL (1 while off)
    PFN_nx2sl_frames_max frames_max;   // NULL with an older DLL, which generates one frame
    PFN_nx2sl_init init;
    PFN_nx2sl_instance_proc gipa;
    PFN_nx2sl_device_proc gdpa;
    PFN_nx2sl_available available;
    PFN_nx2sl_log_unavailable log_unavailable;
    PFN_nx2sl_ui_alpha ui_alpha;
    PFN_nx2sl_frame frame;
    PFN_nx2sl_present_rect present_rect;
    PFN_nx2sl_present_marker marker;
    PFN_nx2sl_sleep sleep;
    PFN_nx2sl_wait wait;
    PFN_nx2sl_phase phase;
    PFN_nx2sl_submit_end submit_end;
    PFN_nx2sl_shown shown;
    PFN_nx2sl_input_retirement input_retirement;
    PFN_nx2sl_frame_inputs_v1 frame_inputs;
    PFN_nx2sl_present_inputs_v1 present_inputs;
    PFN_nx2sl_cancel_inputs_v1 cancel_inputs;
    PFN_nx2sl_shutdown_checked_v1 shutdown_checked;
    PFN_nx2sl_estimate_fg_v1 estimate_fg;
    PFN_nx2sl_fg_observed_v1 fg_observed;
    PFN_nx2sl_reflex reflex;
    uint32_t game;   // the game's presents (vk_dlssg_game_present)
    uint64_t cur;    // the frame the present thread blits and presents next (vk_dlssg_present_frame)
    uint32_t started;      // last distinct guest image acquisition, on the frame-loop thread
    uint32_t waited;       // frame already paced by the update or its acquisition fallback
    uint64_t hosted_ns;    // kern_now_ns after the last host present
    PFN_nx2sl_shutdown shutdown;
    PFN_vkCreateSwapchainKHR create_sc;
    PFN_vkDestroySwapchainKHR destroy_sc;
    PFN_vkGetSwapchainImagesKHR images;
    PFN_vkAcquireNextImageKHR acquire;
    PFN_vkQueuePresentKHR present;
} S = {.wanted = -1, .fail = DLSSG_NO_FILES, .shown_mult = 1};

enum { DLSSG_QUOTE_SLOTS = 12 };
typedef enum {
    DLSSG_QUOTED, DLSSG_NO_MEMORY_ABI, DLSSG_NO_SDK_QUOTE, DLSSG_NO_UNIQUE_HEAP, DLSSG_NO_HEAP_TELEMETRY,
    DLSSG_NO_CACHED_DISPOSITION,
} DlssgQuoteStatus;

typedef struct {
    Nx2SlFgConfigV1 key;
    uint64_t bytes;
    uint32_t sdk_limit;
    int heap;
    DlssgQuoteStatus status;
} DlssgConfigQuote;

typedef struct {
    Nx2SlFgConfigV1 key;
    uint64_t use, native_frame, quote_bytes;
    DlssgQuoteStatus status;
    int observed;
} DlssgCompatConfig;

typedef struct {
    Nx2SlFgConfigV1 key, ready;
    DlssgConfigQuote quote[DLSSG_QUOTE_SLOTS];
    DlssgCompatConfig compat;
    unsigned quote_next;
    RcAdmissionVector peak;
    RcBudgetTicket ticket;
    RcBudgetReadiness refusal;
    uint64_t sequence, use, native_frame, estimated;
    VkFence first_graphics;
    int graphics_done, settled, entered, refused;
} DlssgConfigMemory;

static DlssgConfigMemory memory;

static int config_same(const Nx2SlFgConfigV1 *a, const Nx2SlFgConfigV1 *b) {
    return a->generation == b->generation && a->swapchain == b->swapchain &&
        a->backbuffers == b->backbuffers && a->generated == b->generated &&
        a->mvec_width == b->mvec_width && a->mvec_height == b->mvec_height &&
        a->depth_width == b->depth_width && a->depth_height == b->depth_height &&
        a->color_width == b->color_width && a->color_height == b->color_height &&
        a->color == b->color && a->depth == b->depth && a->motion == b->motion &&
        a->hudless == b->hudless && a->alpha == b->alpha && a->recomposition == b->recomposition &&
        a->queue_mode == b->queue_mode;
}

static int config_device_heap(const VkCtx *ctx) {
    if (!ctx || ctx->props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) return -1;
    int heap = -1;
    for (uint32_t i = 0; i < ctx->mem_props.memoryHeapCount; ++i) {
        if (!(ctx->mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) continue;
        if (heap >= 0 || i >= RC_BUDGET_GPU_HEAPS) return -1;
        heap = (int)i;
    }
    return heap;
}

static void config_trace(const char *phase) {
    if (!(rt_trace & RT_TRACE_PRESENT)) return;
    RcBudgetSnapshot ledger;
    resource_core_budget_snapshot(&ledger);
    const VkCtx *ctx = vk_ctx();
    VmaBudget vma[VK_MAX_MEMORY_HEAPS];
    if (ctx && ctx->vma) vmaGetHeapBudgets(ctx->vma, vma);
    rt_log("dlssg: memory %s generation %llu config %llu images %u %ux%u depth %ux%u generated %u "
           "recomposition %u rough VRAM %llu process commit %llu RAM %llu domains %#x fence %#llx",
           phase, (unsigned long long)memory.key.generation, (unsigned long long)memory.use,
           memory.key.backbuffers, memory.key.color_width, memory.key.color_height,
           memory.key.depth_width, memory.key.depth_height, memory.key.generated, memory.key.recomposition,
           (unsigned long long)memory.estimated, (unsigned long long)ledger.sample.process_commit,
           (unsigned long long)ledger.sample.process_resident, ledger.sample.known_mask,
           (unsigned long long)(uintptr_t)memory.first_graphics);
    if (!ctx) return;
    for (uint32_t i = 0; i < ctx->mem_props.memoryHeapCount && i < RC_BUDGET_GPU_HEAPS; ++i) {
        unsigned domain = RC_BUDGET_GPU_FIRST + i;
        rt_log("dlssg: memory %s heap %u requested %llu allowance %llu sampled free %llu "
               "unobserved %llu ledger retained %llu VMA blocks %llu allocations %llu external-inclusive usage %llu",
               phase, i, (unsigned long long)memory.peak.amount[domain],
               (unsigned long long)ledger.optional_available.amount[domain],
               (unsigned long long)ledger.sample.free.amount[domain],
               (unsigned long long)ledger.unobserved.amount[domain],
               (unsigned long long)ledger.retained.amount[domain],
               ctx->vma ? (unsigned long long)vma[i].statistics.blockBytes : 0,
               ctx->vma ? (unsigned long long)vma[i].statistics.allocationBytes : 0,
               ctx->vma ? (unsigned long long)vma[i].usage : 0);
    }
}

static void config_compat_trace(const char *phase) {
    if (!(rt_trace & RT_TRACE_PRESENT)) return;
    static const char *const why[] = {"quoted", "memory ABI unavailable", "no usable SDK quote",
                                    "no unique local heap", "heap telemetry unavailable",
                                    "cached quote disposition unavailable"};
    const DlssgCompatConfig *compat = &memory.compat;
    const Nx2SlFgConfigV1 *key = &compat->key;
    RcBudgetSnapshot sample;
    resource_core_budget_snapshot(&sample);
    rt_log("dlssg: unknown memory %s generation %llu config %llu frame %llu %ux%u depth %ux%u generated %u "
           "recomposition %u reason %s rough VRAM %llu process commit %llu RAM %llu domains %#x",
           phase, (unsigned long long)key->generation, (unsigned long long)compat->use,
           (unsigned long long)compat->native_frame, key->color_width, key->color_height,
           key->depth_width, key->depth_height, key->generated, key->recomposition, why[compat->status],
           (unsigned long long)compat->quote_bytes, (unsigned long long)sample.sample.process_commit,
           (unsigned long long)sample.sample.process_resident, sample.sample.known_mask);
}

static void config_compat_close(const char *phase) {
    if (memory.compat.use) config_compat_trace(phase);
    memory.compat = (DlssgCompatConfig){0};
}

static void config_trim_estimate(void) {
    // Preflight reserves the largest eligible variant; only the selected estimate enters the SDK.
    // If no ledger record is free for the split, keep the original conservative reservation.
    for (unsigned domain = RC_BUDGET_GPU_FIRST; domain < RC_BUDGET_GPU_FIRST + RC_BUDGET_GPU_HEAPS; ++domain) {
        uint64_t bytes = memory.peak.amount[domain];
        if (bytes <= memory.estimated) continue;
        RcAdmissionVector excess = {0};
        excess.amount[domain] = bytes - memory.estimated;
        RcBudgetTicket unused;
        RcAdmissionStatus status = resource_core_budget_split(&memory.ticket, &excess, RC_BUDGET_OTHER, &unused);
        if (status == RC_ADMISSION_NO_RECORDS) return;
        if (status != RC_ADMISSION_OK || resource_core_budget_cancel(&unused) != RC_ADMISSION_OK)
            rt_fatal("dlssg: configuration variant could not return its unused reservation");
        memory.peak.amount[domain] = memory.estimated;
        return;
    }
}

static void config_settle(int used, const char *phase) {
    if (memory.ticket.token.ledger) {
        if (memory.entered) config_trim_estimate();
        RcAdmissionStatus status = memory.entered ? resource_core_budget_opaque_done(&memory.ticket)
                                                  : resource_core_budget_cancel(&memory.ticket);
        if (status != RC_ADMISSION_OK) rt_fatal("dlssg: configuration reservation lost at %s", phase);
    }
    config_trace(phase);
    if (used) memory.ready = memory.key;
    memory.use = memory.native_frame = 0;
    memory.first_graphics = VK_NULL_HANDLE;
    memory.graphics_done = memory.settled = memory.entered = 0;
}

static DlssgConfigQuote *config_find_quote(const Nx2SlFgConfigV1 *key) {
    for (unsigned i = 0; i < DLSSG_QUOTE_SLOTS; ++i)
        if (config_same(key, &memory.quote[i].key)) return &memory.quote[i];
    return NULL;
}

static const DlssgConfigQuote *config_quote(const Nx2SlFgConfigV1 *key) {
    DlssgConfigQuote *quote = config_find_quote(key);
    if (!quote) {
        quote = &memory.quote[memory.quote_next++ % DLSSG_QUOTE_SLOTS];
        *quote = (DlssgConfigQuote){.key = *key, .heap = -1, .status = DLSSG_NO_SDK_QUOTE,
                                   .sdk_limit = UINT32_MAX};
    }

    if (quote->status == DLSSG_NO_SDK_QUOTE) {
        if (!S.estimate_fg || !S.fg_observed) {
            quote->status = DLSSG_NO_MEMORY_ABI;
            return quote;
        }
        uint32_t limit = S.frames_max ? S.frames_max() : 1;
        if (limit == quote->sdk_limit) return quote;
        quote->sdk_limit = limit;
        Nx2SlFgEstimateV1 answer = {.size = sizeof answer, .version = NX2SL_FG_MEMORY_ABI_V1};
        if (!S.estimate_fg(key, &answer) || answer.size != sizeof answer ||
            answer.version != NX2SL_FG_MEMORY_ABI_V1 || answer.flags != NX2SL_FG_ESTIMATE_ROUGH_VRAM ||
            answer.reserved || answer.padding || answer.sdk_result || !answer.vram_bytes) return quote;
        quote->bytes = answer.vram_bytes;
        quote->heap = config_device_heap(vk_ctx());
        quote->status = quote->heap < 0 ? DLSSG_NO_UNIQUE_HEAP : DLSSG_NO_HEAP_TELEMETRY;
        if (quote->heap >= 0) resource_core_budget_refresh();
    }
    if (quote->status == DLSSG_NO_HEAP_TELEMETRY) {
        RcBudgetReadiness now;
        resource_core_budget_readiness(0, &now);
        if (now.usable_mask & (1u << (RC_BUDGET_GPU_FIRST + (unsigned)quote->heap)))
            quote->status = DLSSG_QUOTED;
    }
    return quote;
}

static uint64_t config_compat_enter(const Nx2SlFgConfigV1 *key, uint64_t frame) {
    DlssgCompatConfig *compat = &memory.compat;
    if (!config_same(key, &compat->key)) {
        const DlssgConfigQuote *quote = config_find_quote(key);
        DlssgQuoteStatus status = quote && quote->status != DLSSG_QUOTED ? quote->status : DLSSG_NO_CACHED_DISPOSITION;
        if (!++memory.sequence) rt_fatal("dlssg: configuration identity exhausted");
        config_compat_close("superseded by another unquoted configuration");
        memory.ready = (Nx2SlFgConfigV1){0};
        *compat = (DlssgCompatConfig){.key = *key, .use = memory.sequence, .native_frame = frame,
                                     .quote_bytes = quote ? quote->bytes : 0, .status = status};
        config_compat_trace("entering SDK compatibility");
    }
    if (compat->observed || !S.fg_observed) return 0;
    compat->native_frame = frame;
    return compat->use;
}

static VkDlssgConfigResult config_reserve(const Nx2SlFgConfigV1 *largest, uint64_t estimate,
                                           VkDlssgConfigPreparation *prepared) {
    if (!largest) return VK_DLSSG_CONFIG_READY;
    if (!config_same(largest, &memory.key) || memcmp(&prepared->peak, &memory.peak, sizeof memory.peak)) {
        memory.refused = 0;
        memory.key = *largest;
        memory.peak = prepared->peak;
        memory.estimated = estimate;
        config_trace("configuration envelope");
    }
    if (vk_sync_stopping() || vk_device_is_lost()) return VK_DLSSG_CONFIG_STOPPED;
    RcBudgetReadiness now;
    resource_core_budget_readiness(0, &now);
    if (memory.refused && !resource_core_budget_peak_progress(&prepared->peak, &memory.refusal, &now))
        return VK_DLSSG_CONFIG_PENDING;
    RcAdmissionStatus status = resource_core_budget_reserve(RC_BUDGET_OTHER, 0, &prepared->peak, &memory.ticket);
    if (status == RC_ADMISSION_REFUSED || status == RC_ADMISSION_NO_RECORDS) {
        memory.refused = 1;
        memory.refusal = now;
        return VK_DLSSG_CONFIG_PENDING;
    }
    if (status != RC_ADMISSION_OK) return VK_DLSSG_CONFIG_TERMINAL;
    memory.refused = 0;
    if (vk_sync_stopping() || vk_device_is_lost()) {
        config_settle(0, "stopped before SDK entry");
        return VK_DLSSG_CONFIG_STOPPED;
    }
    if (!++memory.sequence) rt_fatal("dlssg: configuration identity exhausted");
    memory.use = prepared->use = memory.sequence;
    config_trace("reserved before SDK entry");
    return VK_DLSSG_CONFIG_READY;
}

void vk_dlssg_config_abandon(uint64_t use) {
    if (!use || use != memory.use || memory.entered) return;
    config_settle(0, "abandoned before SDK entry");
}

static int config_enter(uint64_t use, uint64_t frame) {
    if (vk_sync_stopping() || vk_device_is_lost()) {
        vk_dlssg_config_abandon(use);
        return 0;
    }
    if (use && use == memory.use && !memory.entered) {
        if (!config_same(&memory.key, &memory.ready)) memory.ready = (Nx2SlFgConfigV1){0};
        config_compat_close("superseded by a quoted configuration");
        memory.entered = 1;
        config_trace("before native configuration entry");
    }
    if (use && use == memory.use && !memory.settled) memory.native_frame = frame;
    return !use || use == memory.use;
}

static int config_key_valid(const Nx2SlFgConfigV1 *key) {
    return key->size == sizeof *key && key->version == NX2SL_FG_MEMORY_ABI_V1 && !key->flags && !key->reserved &&
        key->generation && key->swapchain && key->backbuffers && key->color_width && key->color_height &&
        key->color != VK_FORMAT_UNDEFINED && !key->queue_mode &&
        (!key->generated || (key->mvec_width && key->mvec_height && key->mvec_width == key->depth_width &&
        key->mvec_height == key->depth_height && key->depth != VK_FORMAT_UNDEFINED && key->motion != VK_FORMAT_UNDEFINED &&
        key->recomposition == (key->alpha != VK_FORMAT_UNDEFINED) && (!key->recomposition || key->hudless != VK_FORMAT_UNDEFINED)));
}

static Nx2SlFgConfigV1 config_plain(Nx2SlFgConfigV1 key) {
    key.hudless = key.alpha = VK_FORMAT_UNDEFINED;
    key.recomposition = 0;
    if (!key.generated) {
        key.mvec_width = key.mvec_height = key.depth_width = key.depth_height = 0;
        key.depth = key.motion = VK_FORMAT_UNDEFINED;
    }
    return key;
}

void vk_dlssg_config_cancel(VkDlssgConfigPreparation *prepared) {
    if (!prepared) return;
    vk_dlssg_config_abandon(prepared->use);
    *prepared = (VkDlssgConfigPreparation){0};
}

int vk_dlssg_config_drain_eligible(uint64_t *use, VkFence *first_graphics) {
    *use = 0;
    *first_graphics = VK_NULL_HANDLE;
    if (!memory.use || !memory.entered || !memory.settled || !memory.first_graphics || vk_device_is_lost()) return 0;
    *use = memory.use;
    *first_graphics = memory.first_graphics;
    return 1;
}

static int config_present_accepted(VkResult result) { return result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR; }

void vk_dlssg_config_presented(uint64_t use, uint64_t frame, VkFence graphics, VkResult result, VkResult image) {
    if (!use || !graphics || !config_present_accepted(result) || !config_present_accepted(image) || !S.fg_observed)
        return;
    if (use == memory.compat.use && frame == memory.compat.native_frame && !memory.compat.observed) {
        if (S.fg_observed(frame, &memory.compat.key)) {
            memory.compat.observed = 1;
            resource_core_budget_refresh();
            config_compat_trace("first exact SDK execution observed");
        }
        return;
    }
    if (use != memory.use || !memory.entered || memory.settled || frame != memory.native_frame ||
        !S.fg_observed(frame, &memory.key)) return;
    memory.first_graphics = graphics;
    memory.settled = 1;
    config_trace("first SDK execution observed");
}

void vk_dlssg_config_progress(VkDevice device, int checked_idle) {
    if (!memory.use || !memory.settled || vk_device_is_lost()) return;
    if (!memory.graphics_done && device) {
        VkResult result = vkGetFenceStatus(device, memory.first_graphics);
        if (result == VK_SUCCESS) {
            memory.graphics_done = 1;
            config_trace("first configuration graphics fence completed");
        } else vk_device_lost_check(result, "Streamline configuration first graphics completion");
    }
    // The first exact present used Streamline's blocking presenting-queue mode.
    // A real idle observation on that device also covers its native FG work.
    if (memory.graphics_done && checked_idle && !vk_device_is_lost())
        config_settle(1, "first configuration use retired");
}

static void config_swapchain_destroyed(VkSwapchainKHR swapchain) {
    if (memory.key.swapchain == swapchain) {
        if (memory.use) config_settle(0, "owning swapchain destroyed");
        memory.key = (Nx2SlFgConfigV1){0};
        memory.refused = 0;
    }
    if (memory.ready.swapchain == swapchain) memory.ready = (Nx2SlFgConfigV1){0};
    if (memory.compat.key.swapchain == swapchain) config_compat_close("owning swapchain destroyed");
    for (unsigned i = 0; i < DLSSG_QUOTE_SLOTS; ++i)
        if (memory.quote[i].key.swapchain == swapchain) memory.quote[i] = (DlssgConfigQuote){0};
}

static void sl_line(const char *line) { rt_log("%s", line); }

static int reflex_mode(void) {
    static int m = -1;
    if (m < 0) {
        const char *s = cfg_get("NX2NAVI_REFLEX");
        m = s && *s ? atoi(s) : 1;
        if (m < 0 || m > 2) m = 1;
    }
    return m;
}

int vk_dlssg_wanted(void) {
    if (S.wanted < 0) {
        S.wanted = !strcmp(cfg_value("NX2NAVI_FRAMEGEN"), "dlss");
        if (S.wanted && !vk_dlaa_wanted()) {
            rt_log("dlssg: frame generation needs DLSS or FSR 3.1 upscaling (any mode): off");
            S.wanted = 0;
        }
    }
    return S.wanted;
}

int vk_dlssg_early(void) {
    if (!vk_dlssg_wanted()) return 0;
    if (S.loaded) return S.loaded > 0 && !S.released;
    S.loaded = -1;
#ifdef _WIN32
    wchar_t dir[MAX_PATH], path[MAX_PATH + 32];
    DWORD n = GetModuleFileNameW(NULL, dir, MAX_PATH);
    while (n && dir[n - 1] != L'\\') n--;
    dir[n ? n - 1 : 0] = 0;
    swprintf(path, MAX_PATH + 32, L"%ls\\nx2navi_dlssg.dll", dir);
    HMODULE m = LoadLibraryW(path);
    if (!m) {
        rt_log("dlssg: nx2navi_dlssg.dll is not next to the game: DLSS Frame Generation off");
        return 0;
    }
    PFN_nx2sl_version ver = (PFN_nx2sl_version)(void *)GetProcAddress(m, "nx2sl_version");
    S.input_retirement = (PFN_nx2sl_input_retirement)(void *)GetProcAddress(m, "nx2sl_input_retirement");
    S.init = (PFN_nx2sl_init)(void *)GetProcAddress(m, "nx2sl_init");
    S.gipa = (PFN_nx2sl_instance_proc)(void *)GetProcAddress(m, "nx2sl_instance_proc");
    S.gdpa = (PFN_nx2sl_device_proc)(void *)GetProcAddress(m, "nx2sl_device_proc");
    S.available = (PFN_nx2sl_available)(void *)GetProcAddress(m, "nx2sl_available");
    S.log_unavailable = (PFN_nx2sl_log_unavailable)(void *)GetProcAddress(m, "nx2sl_log_unavailable");
    S.ui_alpha = (PFN_nx2sl_ui_alpha)(void *)GetProcAddress(m, "nx2sl_ui_alpha");
    S.frame = (PFN_nx2sl_frame)(void *)GetProcAddress(m, "nx2sl_frame");
    S.present_rect = (PFN_nx2sl_present_rect)(void *)GetProcAddress(m, "nx2sl_present_rect");
    S.marker = (PFN_nx2sl_present_marker)(void *)GetProcAddress(m, "nx2sl_present_marker");
    S.sleep = (PFN_nx2sl_sleep)(void *)GetProcAddress(m, "nx2sl_sleep");
    S.wait = (PFN_nx2sl_wait)(void *)GetProcAddress(m, "nx2sl_wait");
    S.phase = (PFN_nx2sl_phase)(void *)GetProcAddress(m, "nx2sl_phase");
    S.submit_end = (PFN_nx2sl_submit_end)(void *)GetProcAddress(m, "nx2sl_submit_end");
    S.shown = (PFN_nx2sl_shown)(void *)GetProcAddress(m, "nx2sl_shown");
    S.shutdown = (PFN_nx2sl_shutdown)(void *)GetProcAddress(m, "nx2sl_shutdown");
    S.frame_inputs = (PFN_nx2sl_frame_inputs_v1)(void *)GetProcAddress(m, "nx2sl_frame_inputs_v1");
    S.present_inputs = (PFN_nx2sl_present_inputs_v1)(void *)GetProcAddress(m, "nx2sl_present_inputs_v1");
    S.cancel_inputs = (PFN_nx2sl_cancel_inputs_v1)(void *)GetProcAddress(m, "nx2sl_cancel_inputs_v1");
    S.shutdown_checked = (PFN_nx2sl_shutdown_checked_v1)(void *)GetProcAddress(m, "nx2sl_shutdown_checked_v1");
    S.estimate_fg = (PFN_nx2sl_estimate_fg_v1)(void *)GetProcAddress(m, "nx2sl_estimate_fg_v1");
    S.fg_observed = (PFN_nx2sl_fg_observed_v1)(void *)GetProcAddress(m, "nx2sl_fg_observed_v1");
    if (!(S.frame_inputs && S.present_inputs && S.cancel_inputs && S.shutdown_checked)) {
        S.frame_inputs = NULL;
        S.present_inputs = NULL;
        S.cancel_inputs = NULL;
        S.shutdown_checked = NULL;
        S.estimate_fg = NULL;
        S.fg_observed = NULL;
    }
    S.reflex = (PFN_nx2sl_reflex)(void *)GetProcAddress(m, "nx2sl_reflex");
    S.frames_max = (PFN_nx2sl_frames_max)(void *)GetProcAddress(m, "nx2sl_frames_max");   // optional: 2x only without
    if (!S.frames_max) rt_log("dlssg: nx2navi_dlssg.dll is older than multi frame generation: 2x at most");
    if (!S.ui_alpha) rt_log("dlssg: this nx2navi_dlssg.dll predates UI alpha tagging; using HUD-less color alone");
    if (!ver || ver() != NX2SL_VERSION || !S.init || !S.gipa || !S.gdpa || !S.available || !S.frame || !S.marker ||
        !S.input_retirement || !S.sleep || !S.submit_end || !S.shown || !S.reflex) {
        rt_log("dlssg: nx2navi_dlssg.dll is from another build: DLSS Frame Generation off");
        return 0;
    }
    if (rt_trace & RT_TRACE_PRESENT) rt_log("dlssg: SDK startup allocation has no pre-entry quote");
    if (!S.init(dir, sl_line)) return 0;
    S.loaded = 1;
    if (!S.frame_inputs) rt_log("dlssg: legacy bridge has no exact late-input receipts or configuration memory quote");
    return 1;
#else
    return 0;
#endif
}

VkResult vk_dlssg_create_instance(const VkInstanceCreateInfo *ci, VkInstance *out) {
    PFN_vkCreateInstance f = S.loaded > 0 && !S.released ?
        (PFN_vkCreateInstance)S.gipa()(VK_NULL_HANDLE, "vkCreateInstance") : NULL;
    VkResult result = f ? f(ci, NULL, out) : vkCreateInstance(ci, NULL, out);
    if (result != VK_SUCCESS && S.shutdown_checked && !vk_dlssg_release_checked())
        rt_fatal("dlssg: Streamline shutdown failed after Vulkan instance creation failed");
    return result;
}

// An instance function through the interposer while Streamline is loaded, else NULL (the caller takes the loader's).
// The window surface goes through it: Streamline learns the window a swapchain shows from vkCreateWin32SurfaceKHR, and
// without that its swapchain hook found "no window corresponding to this application" and generated no frames.
PFN_vkVoidFunction vk_dlssg_instance_proc(VkInstance inst, const char *name) {
    return S.loaded > 0 && !S.released ? S.gipa()(inst, name) : NULL;
}

// The interposer's vkCreateDevice finds the device's instance in a map its own vkEnumeratePhysicalDevices fills; a
// physical device enumerated through the loader alone left it a null instance, so the plugins got an empty instance
// table and sl.common called a null GetPhysicalDeviceMemoryProperties while it set up. Enumerate through it as well.
VkResult vk_dlssg_enumerate(VkInstance inst, uint32_t *n, VkPhysicalDevice *pd) {
    PFN_vkEnumeratePhysicalDevices f =
        S.loaded > 0 && !S.released ?
        (PFN_vkEnumeratePhysicalDevices)S.gipa()(inst, "vkEnumeratePhysicalDevices") : NULL;
    return f ? f(inst, n, pd) : vkEnumeratePhysicalDevices(inst, n, pd);
}

VkResult vk_dlssg_create_device(VkInstance inst, VkPhysicalDevice pd, const VkDeviceCreateInfo *ci, VkDevice *out) {
    PFN_vkCreateDevice f = S.loaded > 0 && !S.released ?
        (PFN_vkCreateDevice)S.gipa()(inst, "vkCreateDevice") : NULL;
    VkResult result = f ? f(pd, ci, NULL, out) : vkCreateDevice(pd, ci, NULL, out);
    if (result != VK_SUCCESS && S.shutdown_checked && !vk_dlssg_release_checked())
        rt_fatal("dlssg: Streamline shutdown failed after Vulkan device creation failed");
    return result;
}

void vk_dlssg_device_ready(VkPhysicalDevice physical, VkDevice dev) {
    if (S.loaded <= 0 || S.released) return;
    if (!S.available()) {
        S.fail = DLSSG_UNAVAILABLE;
        if (S.log_unavailable) S.log_unavailable(physical);
        return;
    }
    PFN_vkGetDeviceProcAddr g = S.gdpa();
    S.create_sc = (PFN_vkCreateSwapchainKHR)g(dev, "vkCreateSwapchainKHR");
    S.destroy_sc = (PFN_vkDestroySwapchainKHR)g(dev, "vkDestroySwapchainKHR");
    S.images = (PFN_vkGetSwapchainImagesKHR)g(dev, "vkGetSwapchainImagesKHR");
    S.acquire = (PFN_vkAcquireNextImageKHR)g(dev, "vkAcquireNextImageKHR");
    S.present = (PFN_vkQueuePresentKHR)g(dev, "vkQueuePresentKHR");
    S.active = S.create_sc && S.destroy_sc && S.images && S.acquire && S.present;
    rt_log("dlssg: %s", S.active ? "DLSS Frame Generation on the swapchain" : "the swapchain hooks are missing: off");
    if (S.active) {
        S.reflex(reflex_mode());
        vk_dlaa_hudless_wanted = 1;
    }
}

int vk_dlssg_active(void) { return S.active; }
int vk_dlssg_input_retirement(VkSemaphore *semaphore, uint64_t *value) {
    *semaphore = VK_NULL_HANDLE;
    *value = 0;
    return S.active && S.input_retirement && S.input_retirement(semaphore, value);
}

int vk_dlssg_status(void) {
    if (S.active) return DLSSG_RUNS;
    return vk_dlssg_wanted() ? S.fail : DLSSG_NOT_WANTED;
}

int vk_dlssg_multiplier_of(const char *value, int max, const char **why) {
    const char *w = NULL;
    long n = 2;
    if (value && *value) {
        char *end;
        n = strtol(value, &end, 10);
        if (*end == 'x' || *end == 'X') end++;
        if (*end || end == value || n < 2) w = "is not 2, 3 or 4", n = 2;
        else if (n > 4) w = "is above 4, the most there is", n = 4;
    }
    if (max >= 2 && n > max) w = "is above what this GPU and driver allow", n = max;
    if (why) *why = w;
    return (int)n;
}

// Streamline's limit as a multiplier: 2 to 4, or 0 while it has reported none (the first frames). An older DLL
// generates one frame: 2.
static int gpu_multiplier_max(int *old_dll) {
    if (old_dll) *old_dll = !S.frames_max;
    if (!S.frames_max) return 2;
    uint32_t f = S.frames_max();
    return !f ? 0 : f >= 3 ? 4 : (int)f + 1;
}

int vk_dlssg_multiplier_max(int *old_dll) {
    if (old_dll) *old_dll = 0;
    return S.active ? gpu_multiplier_max(old_dll) : 0;
}

int vk_dlssg_multiplier(void) { return S.active ? __atomic_load_n(&S.shown_mult, __ATOMIC_RELAXED) : 1; }

// The multiplier the setting gives here (present thread), clamped to the GPU's limit; one log line each time the value
// or the limit changes and the value is not taken as it is. Until Streamline reports its limit, one generated frame,
// and no log line about the GPU yet.
static int multiplier_wanted(void) {
    static char last[24];
    static int last_max = -1, m = 2;
    const char *value = cfg_value("NX2NAVI_FRAMEGEN_MULTIPLIER");
    int max = gpu_multiplier_max(NULL);
    if (max == last_max && !strncmp(value, last, sizeof last - 1)) return m;
    snprintf(last, sizeof last, "%s", value);
    last_max = max;
    const char *why;
    int n = vk_dlssg_multiplier_of(value, max, &why);
    if (why) rt_log("dlssg: framegen_multiplier = %s %s (up to %dx here): %dx", value, why, max ? max : 4, n);
    m = max ? n : 2;
    return m;
}

// vk_device.c: while Streamline is loaded, every device entry point the runtime fetches goes through its interposer, so
// its plugins see the calls they hook (Reflex's frame timing and DLSS-G's presents). NULL keeps the loader's.
PFN_vkGetDeviceProcAddr vk_dlssg_dpa(void) { return S.loaded > 0 && !S.released && S.gdpa ? S.gdpa() : NULL; }

VkResult vk_dlssg_create_swapchain(VkDevice d, const VkSwapchainCreateInfoKHR *ci, VkSwapchainKHR *sc) {
    VkResult r = S.create_sc(d, ci, NULL, sc);
    if (rt_trace & RT_TRACE_PRESENT)
        rt_log("dlssg: interposed swapchain %ux%u, minimum %u images, pre-create memory unknown, result %d",
               ci->imageExtent.width, ci->imageExtent.height, ci->minImageCount, (int)r);
    if (r == VK_SUCCESS) S.reflex(reflex_mode());
    return r;
}
void vk_dlssg_destroy_swapchain(VkDevice d, VkSwapchainKHR sc) {
    // The presentation owner drained accepted and possibly-issued work before destroying this swapchain.
    S.destroy_sc(d, sc, NULL);
    if (!vk_device_is_lost()) config_swapchain_destroyed(sc);
}
VkResult vk_dlssg_get_images(VkDevice d, VkSwapchainKHR sc, uint32_t *n, VkImage *img) { return S.images(d, sc, n, img); }
VkResult vk_dlssg_acquire(VkDevice d, VkSwapchainKHR sc, uint64_t t, VkSemaphore sem, uint32_t *idx) {
    return S.acquire(d, sc, t, sem, VK_NULL_HANDLE, idx);
}

VkResult vk_dlssg_present(VkQueue q, const VkPresentInfoKHR *pi) {
    S.marker(0, S.cur);
    uint64_t begin = kern_now_ns();
    VkResult r = S.present(q, pi);
    uint64_t presented = kern_now_ns();
    S.marker(1, S.cur);
    static uint64_t present_ns, state_ns;
    static unsigned samples;
    present_ns += presented - begin;
    state_ns += kern_now_ns() - presented;
    if (++samples == 120) {
        rt_log("presentation: SDK present %.3f ms, state %.3f ms per frame", present_ns / 1.2e8, state_ns / 1.2e8);
        samples = 0;
        present_ns = state_ns = 0;
    }
    __atomic_store_n(&S.hosted_ns, kern_now_ns(), __ATOMIC_RELAXED);
    return r;
}

int vk_dlssg_has_input_receipts(void) { return S.active && S.frame_inputs; }

VkResult vk_dlssg_present_inputs(VkQueue queue, const VkPresentInfoKHR *present,
                                 const Nx2SlInputReceipt *receipt, Nx2SlInputCompletion *out) {
    *out = (Nx2SlInputCompletion){.size = sizeof *out, .version = NX2SL_INPUTS_ABI_V1};
    if (!vk_dlssg_has_input_receipts()) {
        out->flags = NX2SL_INPUT_UNPROVEN;
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (!receipt || !present) {
        out->flags = NX2SL_INPUT_UNPROVEN;
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    VkResult result = S.present_inputs(S.present, queue, present, receipt, out);
    int accepted = (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) && present->swapchainCount == 1 &&
                   (!present->pResults || present->pResults[0] == VK_SUCCESS || present->pResults[0] == VK_SUBOPTIMAL_KHR);
    if (out->size != sizeof *out || out->version != NX2SL_INPUTS_ABI_V1 || out->reserved ||
        out->ticket != receipt->ticket || out->flags > NX2SL_INPUT_UNPROVEN ||
        (out->flags == NX2SL_INPUT_NO_LATE_TAGS && (receipt->accepted || receipt->uncertain)) ||
        (out->flags == NX2SL_INPUT_TIMELINE && (!receipt->ticket || !out->semaphore || !out->value || !accepted))) {
        *out = (Nx2SlInputCompletion){.size = sizeof *out, .version = NX2SL_INPUTS_ABI_V1,
                                     .flags = NX2SL_INPUT_UNPROVEN, .ticket = receipt->ticket};
    }
    if (accepted) __atomic_store_n(&S.hosted_ns, kern_now_ns(), __ATOMIC_RELAXED);
    return result;
}

int vk_dlssg_cancel_inputs(const Nx2SlInputReceipt *receipt) {
    return vk_dlssg_has_input_receipts() && receipt && receipt->ticket && S.cancel_inputs(receipt);
}

// Reflex holds the frame's start until the time the driver computes from the measured render and queue times, so the
// game's CPU work ends about when the GPU is free for it; then the frame's first markers. Once a frame: the game may
// acquire again before it presents. No hold while no host present went out for HOLD_IDLE_NS (the window minimized:
// gfx_present.c skips the host presents while the game runs on): the driver then has no present to time the frame by,
// and the sleep would wait out Streamline's half-second timeout each frame.
#define HOLD_IDLE_NS 200000000ull

// Turning generation on or off is an slDLSSGSetOptions call and Streamline's own rebuild, and it can be seen: around a
// menu, a fade or a swapchain change the depth and motion-vector views come and go for a frame or two. Following them
// frame by frame switched generation off and on dozens of times a second (a run logged 32 changes, in pairs, a frame
// apart), so a change waits until the new state has held for FG_SETTLE_FRAMES frames.
#define FG_SETTLE_FRAMES 12

static int recently_presenting(void) {
    uint64_t hosted = __atomic_load_n(&S.hosted_ns, __ATOMIC_RELAXED);
    return hosted && kern_now_ns() - hosted < HOLD_IDLE_NS;
}

static RtDispatchHook update_hook;

// In 1.1.1 the frame loop calls this virtual method, waits for its update jobs (including the pad reads),
// then acquires the image for their result. The previous image can still be presented during the update.
static void game_update(Cpu *c) {
    uint32_t frame = S.started + 1;
    if (c->x[30] != 0x700346c73cull || !S.active || !S.phase || !S.wait || frame == S.waited) {
        update_hook.original(c);
        return;
    }
    static int announced;
    if (!announced) {
        announced = 1;
        rt_log("dlssg: pacing before the game update; simulation and render submission use separate Reflex markers");
    }
    S.waited = frame;
    int hold = recently_presenting();
    int away = hold ? kern_core_away() : 0;
    S.wait(frame, hold);
    if (hold) kern_core_back(&away);
    S.phase(frame, NX2SL_SIM_START);
    update_hook.original(c);
    S.phase(frame, NX2SL_SIM_END);
}

RtDispatchHook *vk_dlssg_update_hook(void) {
    if (!RT_V111 || strcmp(cfg_value("NX2NAVI_FRAMEGEN"), "dlss")) return NULL;
    update_hook = (RtDispatchHook){.pc = 0x7003b4eb40ull, .wrapper = game_update};
    return &update_hook;
}

void vk_dlssg_frame_start(void) {
    if (!S.active) return;
    uint32_t f = __atomic_load_n(&S.game, __ATOMIC_RELAXED) + 1;
    if (f == S.started) return;
    S.started = f;
    if (S.waited == f) return;
    S.waited = f;
    if (S.wait && S.phase && update_hook.original) S.wait(f, recently_presenting());
    else S.sleep(f, recently_presenting());
}

void vk_dlssg_game_present(void) { __atomic_add_fetch(&S.game, 1, __ATOMIC_RELAXED); }

void vk_dlssg_render_start(uint64_t frame) {
    if (!S.active || !S.phase || !S.wait || !update_hook.original) return;
    // The replay worker (or the synchronous submit path) visits frames in guest submission order.
    static uint64_t last;
    if (last == frame) return;
    last = frame;
    S.phase((uint32_t)frame, NX2SL_RENDER_START);
}

void vk_dlssg_submitted(uint64_t frame) {
    if (S.active) S.submit_end((uint32_t)frame);
}

void vk_dlssg_present_frame(uint64_t frame) { S.cur = frame; }

uint64_t vk_dlssg_shown(void) { return S.active ? S.shown() : 0; }

int vk_dlssg_ui_alpha(void) { return S.active && S.ui_alpha; }
uint64_t vk_dlssg_current_frame(void) { return S.cur; }

typedef struct {
    uint32_t backbuffers;
    VkFormat format;
    VkDlssgConfigPreparation *prepared;
    uint64_t use;
    RcAdmissionVector peak;
    VkDlssgConfigResult result;
} DlssgFrameAdmission;

static int frame_generation_mode(int have, int *mode, int *pending) {
    int want = have && !menu_is_open();
    *mode = S.mode;
    *pending = S.pending;
    if (want == *mode) *pending = 0;
    else if (++*pending >= FG_SETTLE_FRAMES) { *mode = want; *pending = 0; }
    else want = *mode;
    return want;
}

uint32_t vk_dlssg_config_next_generated(int scene_inputs) {
    int mode, pending;
    return scene_inputs == VK_DLSSG_SCENE_VALID && frame_generation_mode(1, &mode, &pending)
        ? (uint32_t)(multiplier_wanted() - 1) : 0;
}

VkDlssgConfigResult vk_dlssg_config_preflight_variants(const Nx2SlFgConfigV1 *choices, uint32_t count,
    int scene_inputs, uint64_t frame, VkDlssgConfigPreparation *prepared, RcAdmissionVector *peak) {
    *peak = (RcAdmissionVector){0};
    if (!choices || !count || count > VK_DLSSG_CONFIG_VARIANTS || !frame || !prepared || prepared->count ||
        !vk_dlssg_has_input_receipts()) return VK_DLSSG_CONFIG_TERMINAL;
    if (vk_sync_stopping() || vk_device_is_lost()) return VK_DLSSG_CONFIG_STOPPED;
    if (scene_inputs != VK_DLSSG_SCENE_VALID && scene_inputs != VK_DLSSG_SCENE_ABSENT)
        return VK_DLSSG_CONFIG_INVALIDATED;
    VkDlssgConfigPreparation plan = {.frame = frame, .count = count,
                                     .missing_scene = scene_inputs == VK_DLSSG_SCENE_ABSENT};
    int wants_generation = frame_generation_mode(!plan.missing_scene, &plan.mode, &plan.pending);
    // Off applies to this inputless frame. The logical debounce and prior configuration can still be on.
    if (wants_generation && !plan.missing_scene) plan.generated = (uint32_t)(multiplier_wanted() - 1);
    if (!plan.generated && count != 1) return VK_DLSSG_CONFIG_INVALIDATED;

    Nx2SlFgConfigV1 common = config_plain(choices[0]);
    uint32_t variants = 0;
    const Nx2SlFgConfigV1 *largest = NULL;
    uint64_t max = 0;
    int found_heap = -1, ready = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (!config_key_valid(&choices[i]) || choices[i].generated != plan.generated)
            return VK_DLSSG_CONFIG_INVALIDATED;
        Nx2SlFgConfigV1 plain = config_plain(choices[i]);
        if (!config_same(&plain, &common)) return VK_DLSSG_CONFIG_INVALIDATED;
        uint32_t kind = choices[i].recomposition ? 4 : choices[i].hudless != VK_FORMAT_UNDEFINED ? 2 : 1;
        if (variants & kind) return VK_DLSSG_CONFIG_INVALIDATED;
        variants |= kind;
        plan.choice[i].key = plan.generated ? choices[i] : plain;
    }

    if (plan.generated && memory.use && memory.entered) {
        // The prior owner's first use has not yet settled. Retrying that exact config is allowed.
        if (count != 1 || !config_same(&plan.choice[0].key, &memory.key))
            return memory.settled ? VK_DLSSG_CONFIG_DRAIN_REQUIRED : VK_DLSSG_CONFIG_PENDING;
        plan.use = memory.use;
        plan.peak = memory.peak;
        plan.choice[0].estimate = memory.estimated;
    } else if (plan.generated) {
        if (memory.use) config_settle(0, "superseded before SDK entry");
        for (uint32_t i = 0; i < count; ++i) {
            VkDlssgConfigChoice *candidate = &plan.choice[i];
            if (config_same(&candidate->key, &memory.ready)) { ready++; continue; }
            // Unknown growth already attempted for this exact configuration can be observed,
            // but cannot be retroactively admitted as if the vendor had never run.
            if (config_same(&candidate->key, &memory.compat.key)) {
                candidate->result = VK_DLSSG_CONFIG_COMPAT;
                continue;
            }
            const DlssgConfigQuote *quote = config_quote(&candidate->key);
            if (quote->status != DLSSG_QUOTED) {
                candidate->result = VK_DLSSG_CONFIG_COMPAT;
                continue;
            }
            if (found_heap >= 0 && found_heap != quote->heap) return VK_DLSSG_CONFIG_TERMINAL;
            found_heap = quote->heap;
            ready++;
            candidate->estimate = quote->bytes;
            if (quote->bytes > max) { max = quote->bytes; largest = &candidate->key; }
        }
        if (max) plan.peak.amount[RC_BUDGET_GPU_FIRST + (unsigned)found_heap] = max;
        *peak = plan.peak;
        VkDlssgConfigResult result = config_reserve(largest, max, &plan);
        if (result != VK_DLSSG_CONFIG_READY) return result;
    } else ready = 1;
    *prepared = plan;
    *peak = plan.peak;
    return ready || plan.use ? VK_DLSSG_CONFIG_READY : VK_DLSSG_CONFIG_COMPAT;
}

static int frame_inputs(VkCommandBuffer cb, VkExtent2D ext, const int rect[4], const Nx2SlImage *hudless,
                        const Nx2SlImage *alpha, uint64_t frame, VkSwapchainKHR swapchain, uint64_t generation,
                        Nx2SlInputReceipt *receipt, DlssgFrameAdmission *admission) {
    if (!S.active) return 0;
    VkFfxImage d = {0}, m = {0};
    VkImageView dv = VK_NULL_HANDLE, mvv = VK_NULL_HANDLE;
    VkDlaaCamera c;
    float jx = 0, jy = 0;
    int reset = 0;
    int scene = admission ? vk_dlaa_fg_compose_peek(&d, &m, &dv, &mvv, &c, &jx, &jy, &reset) :
        vk_dlaa_fg_inputs(&d, &m, &jx, &jy, &reset) && vk_dlaa_fg_views(&dv, &mvv) && vk_dlaa_fg_camera(&c);
    int have = scene == VK_DLSSG_SCENE_VALID;
    int mode, pending, on, mult;
    if (admission && admission->prepared && admission->prepared->frame == frame) {
        mode = admission->prepared->mode;
        pending = admission->prepared->pending;
        mult = (int)admission->prepared->generated + 1;
        on = mult > 1;
    } else {
        on = frame_generation_mode(have, &mode, &pending);
        mult = on ? multiplier_wanted() : 1;
    }   // the pacer's divisor (frame_rate.c)
    Nx2SlImage zd = {d.image, dv, d.format, d.width, d.height, VK_IMAGE_LAYOUT_GENERAL,
                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT};
    Nx2SlImage zm = {m.image, mvv, m.format, m.width, m.height, VK_IMAGE_LAYOUT_GENERAL,
                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT};
    Nx2SlCamera k;
    memset(&k, 0, sizeof k);
    if (have && (!admission || on)) {
        memcpy(k.view_to_clip, c.view_to_clip, 64), memcpy(k.clip_to_view, c.clip_to_view, 64);
        memcpy(k.clip_to_prev_clip, c.clip_to_prev_clip, 64), memcpy(k.prev_clip_to_clip, c.prev_clip_to_clip, 64);
        memcpy(k.pos, c.pos, 12), memcpy(k.up, c.up, 12), memcpy(k.right, c.right, 12), memcpy(k.fwd, c.fwd, 12);
        // the frame's own near and far planes (vk_dlaa.c proj_planes); nx2navi_dlssg.dll tells Streamline the depth is
        // not inverted, which holds for every projection the game has shown (0 near, 1 far)
        k.znear = c.znear, k.zfar = c.zfar, k.fov_y = c.fov_y, k.aspect = c.aspect;
        static int told;
        if (c.inverted && !told++) rt_log("dlssg: the scene's depth is inverted, which Streamline is not told");
        k.jitter[0] = jx, k.jitter[1] = jy;
        k.mv_scale[0] = 1.0f / (float)d.width, k.mv_scale[1] = 1.0f / (float)d.height;   // render pixels
        k.reset = reset;
    }
    // The composed owner supplies exact late-image descriptions for the versioned path.
    int send_scene = have && (!admission || on);
    const Nx2SlImage *hl = send_scene && hudless && hudless->image && hudless->view ? hudless : NULL;
    const Nx2SlImage *ui = hl && alpha && alpha->image && alpha->view ? alpha : NULL;
    int generate = on ? mult - 1 : 0;
    if (admission) {
        if (vk_sync_stopping() || vk_device_is_lost()) {
            admission->result = VK_DLSSG_CONFIG_STOPPED;
            return 0;
        }
        Nx2SlFgConfigV1 key = {.size = sizeof key, .version = NX2SL_FG_MEMORY_ABI_V1,
            .generation = generation, .swapchain = swapchain, .backbuffers = admission->backbuffers,
            .generated = (uint32_t)generate, .color_width = ext.width, .color_height = ext.height,
            .color = admission->format};
        if (generate && have) {
            key.mvec_width = m.width;
            key.mvec_height = m.height;
            key.depth_width = d.width;
            key.depth_height = d.height;
            key.depth = d.format;
            key.motion = m.format;
            key.hudless = hl ? hl->format : VK_FORMAT_UNDEFINED;
            key.alpha = ui ? ui->format : VK_FORMAT_UNDEFINED;
            key.recomposition = ui != NULL;
        }
        VkDlssgConfigPreparation *prepared = admission->prepared;
        const VkDlssgConfigChoice *selected = NULL;
        if (prepared && prepared->frame == frame && prepared->count <= VK_DLSSG_CONFIG_VARIANTS &&
            scene == (prepared->missing_scene ? VK_DLSSG_SCENE_ABSENT : VK_DLSSG_SCENE_VALID)) {
            for (uint32_t i = 0; i < prepared->count; ++i)
                if (config_same(&key, &prepared->choice[i].key)) { selected = &prepared->choice[i]; break; }
        }
        if (!selected || (prepared->use && prepared->use != memory.use) ||
            // The selected scene's fields are immutable. Only the successful claim changes ownership.
            (generate && !vk_dlaa_fg_inputs(&d, &m, &jx, &jy, &reset))) {
            admission->result = VK_DLSSG_CONFIG_INVALIDATED;
            return 0;
        }
        admission->result = selected->result;
        admission->peak = prepared->peak;
        if (selected->estimate && prepared->use) {
            admission->use = prepared->use;
            if (!memory.entered) {
                memory.key = selected->key;
                memory.estimated = selected->estimate;
                config_trim_estimate();
            }
            admission->peak = memory.peak;
        } else vk_dlssg_config_abandon(prepared->use);
        *prepared = (VkDlssgConfigPreparation){0};
        if (!config_enter(admission->use, frame)) {
            admission->result = VK_DLSSG_CONFIG_STOPPED;
            return 0;
        }
        if (generate && admission->result == VK_DLSSG_CONFIG_COMPAT)
            admission->use = config_compat_enter(&key, frame);
    }
    S.mode = mode;
    S.pending = pending;
    if (!admission || generate || !S.frames_started)
        __atomic_store_n(&S.shown_mult, on ? mult : 1, __ATOMIC_RELAXED); // the pacer's divisor (frame_rate.c)
    if (admission && !generate && !S.frames_started) {
        // The SDK is still in its initial off mode. Present the real frame without entering the tag/option path.
        receipt->frame = frame;
        receipt->swapchain = swapchain;
        receipt->generation = generation;
        receipt->sdk_frame = (uint32_t)frame;
        return 1;
    }
    if (generate) S.frames_started = 1;
    if (receipt) {
        Nx2SlFrameInputsV1 input = {.size = sizeof input, .version = NX2SL_INPUTS_ABI_V1,
            .frame = frame, .generation = generation, .swapchain = swapchain, .cb = cb,
            .depth = send_scene ? &zd : NULL, .motion = send_scene ? &zm : NULL, .hudless = hl, .alpha = ui,
            .camera = send_scene ? &k : NULL, .width = ext.width, .height = ext.height, .generated = generate};
        memcpy(input.rect, rect, sizeof input.rect);
        int ok = S.frame_inputs(&input, receipt);
        uint32_t late = receipt->accepted | receipt->uncertain;
        uint32_t supplied = (hl ? NX2SL_INPUT_HUDLESS : 0) | (ui ? NX2SL_INPUT_UI_ALPHA : 0);
        if (receipt->size != sizeof *receipt || receipt->version != NX2SL_INPUTS_ABI_V1 ||
            receipt->flags || receipt->reserved || (receipt->accepted & receipt->uncertain) ||
            (late & ~supplied) || (!!late != !!receipt->ticket) ||
            ((ok || late) && (receipt->frame != frame || receipt->sdk_frame != (uint32_t)frame ||
                             receipt->swapchain != swapchain || receipt->generation != generation))) {
            *receipt = (Nx2SlInputReceipt){.size = sizeof *receipt, .version = NX2SL_INPUTS_ABI_V1,
                .frame = frame, .generation = generation, .swapchain = swapchain, .sdk_frame = (uint32_t)frame,
                .uncertain = supplied};
            return 0; // Keep any requested inputs until checked teardown if the bridge reply is malformed.
        }
        if (admission && !generate && ok) __atomic_store_n(&S.shown_mult, 1, __ATOMIC_RELAXED);
        return ok;
    }
    if (S.ui_alpha) S.ui_alpha(ui);
    if (S.present_rect) S.present_rect(rect);
    return S.frame(cb, have ? &zd : NULL, have ? &zm : NULL, hl, have ? &k : NULL, ext.width, ext.height, generate, (uint32_t)frame);
}

int vk_dlssg_frame_inputs(VkCommandBuffer cb, VkExtent2D extent, const int rect[4],
                          const Nx2SlImage *hudless, const Nx2SlImage *alpha, uint64_t frame,
                          VkSwapchainKHR swapchain, uint64_t generation, Nx2SlInputReceipt *out) {
    *out = (Nx2SlInputReceipt){.size = sizeof *out, .version = NX2SL_INPUTS_ABI_V1};
    if (!vk_dlssg_has_input_receipts()) return 0;
    return frame_inputs(cb, extent, rect, hudless, alpha, frame, swapchain, generation, out, NULL);
}

VkDlssgConfigResult vk_dlssg_frame_inputs_admitted(VkCommandBuffer cb, VkExtent2D extent, VkFormat format,
    uint32_t backbuffers, const int rect[4], const Nx2SlImage *hudless, const Nx2SlImage *alpha,
    uint64_t frame, VkSwapchainKHR swapchain, uint64_t generation, VkDlssgConfigPreparation *prepared,
    Nx2SlInputReceipt *receipt, uint64_t *memory_use, RcAdmissionVector *peak, int *sdk_tag_ok) {
    *receipt = (Nx2SlInputReceipt){.size = sizeof *receipt, .version = NX2SL_INPUTS_ABI_V1};
    DlssgFrameAdmission admission = {.backbuffers = backbuffers, .format = format, .prepared = prepared,
                                     .result = VK_DLSSG_CONFIG_TERMINAL};
    *sdk_tag_ok = vk_dlssg_has_input_receipts() && cb && frame && generation && swapchain && rect &&
        extent.width && extent.height && format != VK_FORMAT_UNDEFINED && backbuffers &&
        frame_inputs(cb, extent, rect, hudless, alpha, frame, swapchain, generation, receipt, &admission);
    vk_dlssg_config_cancel(prepared); // A successful entry already consumed the preflight value.
    *memory_use = admission.use;
    *peak = admission.peak;
    return admission.result;
}

void vk_dlssg_frame(VkCommandBuffer cb, VkExtent2D ext, const int rect[4], VkImage hudless, VkImageView hudless_view,
                    VkFormat hudless_format, VkImage alpha, VkImageView alpha_view) {
    Nx2SlImage hl = {hudless, hudless_view, hudless_format, ext.width, ext.height, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
    Nx2SlImage ui = {alpha, alpha_view, VK_FORMAT_R16_SFLOAT, ext.width, ext.height, VK_IMAGE_LAYOUT_GENERAL,
                     hl.usage | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
    (void)frame_inputs(cb, ext, rect, &hl, &ui, S.cur, VK_NULL_HANDLE, 0, NULL, NULL);
}

int vk_dlssg_release_checked(void) {
    if (S.loaded <= 0 || S.released) return 1;
    if (!S.shutdown_checked || !S.shutdown_checked()) return 0;
    if (memory.use) config_settle(0, "checked SDK shutdown");
    config_compat_close("checked SDK shutdown");
    memory = (DlssgConfigMemory){0};
    S.released = 1;
    S.active = 0;
    return 1;
}

void vk_dlssg_release(void) {
    if (S.shutdown_checked) {
        (void)vk_dlssg_release_checked();
        return;
    }
    if (S.loaded > 0 && S.shutdown) S.shutdown();
    S.active = 0;
}

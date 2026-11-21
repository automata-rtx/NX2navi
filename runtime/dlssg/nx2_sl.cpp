// nx2navi_dlssg.dll: DLSS Frame Generation through NVIDIA Streamline, behind the C interface of nx2_sl.h. Built with
// MSVC against the Streamline SDK's headers (tools/build_dlssg.py; the SDK stays outside the repository), so its C++
// types have the ABI of Streamline's own DLLs. sl.interposer.dll and its plugins are loaded from the game's folder at
// run time; this DLL links none of them.
//
// Manual hooking (the SDK's ProgrammingGuideManualHooking): the game creates its instance and device through the
// interposer's vkCreateInstance and vkCreateDevice, which add what DLSS-G needs, and calls the interposer's
// swapchain functions; everything else goes to the Vulkan loader as before. Each frame of the game sleeps in Reflex
// where it starts and gets the PCL latency markers, the camera constants and a snapshot of the depth and motion vectors (taken
// when tagged), then the present goes through the interposer, which inserts the generated frames: one, two or
// three between two rendered frames (2x, 3x or 4x), up to what the GPU and driver allow (numFramesToGenerateMax).
#define NX2SL_BUILD
#include <windows.h>

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include <atomic>
#include <chrono>
#include <mutex>

#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss_g.h>
#include <sl_helpers.h>
#include <sl_pcl.h>
#include <sl_reflex.h>

#include "nx2_sl.h"

static_assert(sizeof(Nx2SlFrameInputsV1) == 120 && offsetof(Nx2SlFrameInputsV1, frame) == 16);
static_assert(sizeof(Nx2SlInputReceipt) == 64 && offsetof(Nx2SlInputReceipt, ticket) == 16);
static_assert(sizeof(Nx2SlInputCompletion) == 40 && offsetof(Nx2SlInputCompletion, ticket) == 16);

namespace {

HMODULE g_mod;
Nx2SlLog g_log;
PFun_slInit *p_init;
PFun_slShutdown *p_shutdown;
PFun_slIsFeatureLoaded *p_loaded;
PFun_slSetTagForFrame *p_tag;
PFun_slSetConstants *p_constants;
PFun_slGetNewFrameToken *p_token;
PFun_slGetFeatureFunction *p_feature;
PFun_slDLSSGSetOptions *p_dlssg_options;
PFun_slDLSSGGetState *p_dlssg_state;
PFun_slPCLSetMarker *p_marker;
PFun_slReflexSetOptions *p_reflex;
PFun_slReflexGetState *p_reflex_state;
PFun_slReflexSleep *p_sleep;
std::mutex g_token_slot;
std::mutex g_reflex_options;  // GetState and SetOptions are not thread safe
using Clock = std::chrono::steady_clock;
constexpr unsigned pacingWindow = 300;
PFN_vkGetInstanceProcAddr p_gipa;
PFN_vkGetDeviceProcAddr p_gdpa;
int g_available = -1;
bool g_initialized;
Nx2SlInputReceipt g_pending_inputs{};
uint64_t g_next_ticket;

struct InputSample {
    VkSemaphore semaphore;
    uint64_t value;
    bool checked;
};
InputSample g_last_input_sample{};
InputSample g_last_proof{};
struct FgExecutionObservation {
    uint64_t frame;
    Nx2SlFgConfigV1 config;
};
FgExecutionObservation g_fg_prepared{}, g_fg_execution{};
uint64_t g_sample_generation;
int g_frames = -1;                   // the frames to generate as last set (0 off), -1 before the first nx2sl_frame
std::atomic<uint32_t> g_max{0};      // numFramesToGenerateMax as Streamline last reported it, 0 before it did
uint64_t g_presented;
VkSemaphore g_input_fence;
uint64_t g_input_value;
std::atomic<uint64_t> g_shown{0};

void say(const char *fmt, ...) {
    if (!g_log) return;
    char s[512];
    va_list a;
    va_start(a, fmt);
    vsnprintf(s, sizeof s, fmt, a);
    va_end(a);
    g_log(s);
}

void sl_log(sl::LogType type, const char *msg) {
    if (type == sl::LogType::eInfo) return;   // only warnings and errors reach the game's log
    static int n;
    if (n++ < 200) say("dlssg: streamline %s: %s", type == sl::LogType::eError ? "error" : "warning", msg);
}

template <class T> T *proc(const char *name) { return reinterpret_cast<T *>(GetProcAddress(g_mod, name)); }

// The token of frame n. Tokens come from a small ring Streamline reuses, so each call takes its own right before use.
sl::FrameToken *token_locked(uint32_t frame) {
    sl::FrameToken *t = nullptr;
    uint32_t requested = frame;
    return frame && p_token(t, &requested) == sl::Result::eOk && t && (uint32_t)*t == frame ? t : nullptr;
}

sl::FrameToken *token(uint32_t frame) {
    std::lock_guard<std::mutex> lock(g_token_slot);
    return token_locked(frame);
}

sl::Result tag_frame(uint32_t frame, const sl::ViewportHandle &viewport, const sl::ResourceTag *tags,
                     uint32_t count, VkCommandBuffer cb, uint32_t *sdk_frame = nullptr) {
    std::lock_guard<std::mutex> lock(g_token_slot);
    sl::FrameToken *t = token_locked(frame);
    if (!t) return sl::Result::eErrorMissingInputParameter;
    if (sdk_frame) *sdk_frame = (uint32_t)*t;
    return p_tag(*t, viewport, tags, count, (sl::CommandBuffer *)cb);
}

template <typename T> bool input_abi(const T *p) {
    return p && p->size == sizeof(T) && p->version == NX2SL_INPUTS_ABI_V1 && !p->flags && !p->reserved;
}

bool same_inputs(const Nx2SlInputReceipt &a, const Nx2SlInputReceipt &b) {
    return a.ticket == b.ticket && a.frame == b.frame && a.sdk_frame == b.sdk_frame &&
           a.generation == b.generation && a.swapchain == b.swapchain &&
           a.accepted == b.accepted && a.uncertain == b.uncertain;
}

InputSample input_sample(const sl::DLSSGState &s) {
    return {(VkSemaphore)(uintptr_t)s.inputsProcessingCompletionFence,
            s.lastPresentInputsProcessingCompletionFenceValue, true};
}

InputSample query_input_sample() {
    sl::DLSSGState state{};
    return p_dlssg_state(sl::ViewportHandle(0), state, nullptr) == sl::Result::eOk ? input_sample(state) : InputSample{};
}

bool mark(sl::PCLMarker marker, sl::FrameToken &t, uint32_t frame) {
    sl::Result r = p_marker(marker, t);
    if (r != sl::Result::eOk) {
        static std::atomic<unsigned> errors{0};
        if (errors.fetch_add(1, std::memory_order_relaxed) < 8)
            say("dlssg: PCL marker %u rejected on frame %u: %s", (unsigned)marker, frame, sl::getResultAsStr(r));
    }
    return r == sl::Result::eOk;
}

void matrix(sl::float4x4 &m, const float *v) {
    for (int r = 0; r < 4; r++) m[r] = sl::float4(v[r * 4], v[r * 4 + 1], v[r * 4 + 2], v[r * 4 + 3]);
}

sl::Resource resource(const Nx2SlImage *i) {
    sl::Resource r(sl::ResourceType::eTex2d, (void *)i->image, nullptr, (void *)i->view, (uint32_t)i->layout);
    r.width = i->width;
    r.height = i->height;
    r.nativeFormat = (uint32_t)i->format;
    r.mipLevels = 1;
    r.arrayLayers = 1;
    r.usage = (uint32_t)i->usage;
    return r;
}

}  // namespace

extern "C" NX2SL_API int nx2sl_version(void) { return NX2SL_VERSION; }

extern "C" NX2SL_API int nx2sl_init(const wchar_t *dir, Nx2SlLog log) {
    g_log = log;
    wchar_t path[MAX_PATH];
    swprintf(path, MAX_PATH, L"%ls\\sl.interposer.dll", dir);
    g_mod = LoadLibraryW(path);
    if (!g_mod) return say("dlssg: sl.interposer.dll not found next to the game: DLSS Frame Generation off"), 0;
    p_init = proc<PFun_slInit>("slInit");
    p_shutdown = proc<PFun_slShutdown>("slShutdown");
    p_loaded = proc<PFun_slIsFeatureLoaded>("slIsFeatureLoaded");
    p_tag = proc<PFun_slSetTagForFrame>("slSetTagForFrame");
    p_constants = proc<PFun_slSetConstants>("slSetConstants");
    p_token = proc<PFun_slGetNewFrameToken>("slGetNewFrameToken");
    p_feature = proc<PFun_slGetFeatureFunction>("slGetFeatureFunction");
    p_gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(g_mod, "vkGetInstanceProcAddr"));
    p_gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(GetProcAddress(g_mod, "vkGetDeviceProcAddr"));
    if (!p_init || !p_tag || !p_constants || !p_token || !p_feature || !p_gipa || !p_gdpa)
        return say("dlssg: sl.interposer.dll lacks its entry points"), 0;
    static const sl::Feature features[] = {sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL};
    const wchar_t *paths[] = {dir};
    sl::Preferences p{};
    p.showConsole = false;
    p.logLevel = sl::LogLevel::eDefault;
    p.pathsToPlugins = paths;
    p.numPathsToPlugins = 1;
    p.pathToLogsAndData = dir;
    p.logMessageCallback = sl_log;
    // the game tracks its own command buffer state; no plugin downloads at run time. The depth and motion vectors are
    // tagged per frame (slSetTagForFrame, nx2sl_frame), which Streamline refuses without the frame-based flag.
    p.flags = sl::PreferenceFlags::eDisableCLStateTracking | sl::PreferenceFlags::eUseManualHooking |
              sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    p.featuresToLoad = features;
    p.numFeaturesToLoad = 3;
    p.applicationId = 231313132;   // the id NVIDIA's samples use for builds without an issued one (as vk_dlaa.c's NGX)
    p.engine = sl::EngineType::eCustom;
    p.renderAPI = sl::RenderAPI::eVulkan;
    sl::Result r = p_init(p, sl::kSDKVersion);
    if (r != sl::Result::eOk) return say("dlssg: slInit failed (%d): DLSS Frame Generation off", (int)r), 0;
    g_initialized = true;
    say("dlssg: Streamline up");
    return 1;
}

extern "C" NX2SL_API PFN_vkGetInstanceProcAddr nx2sl_instance_proc(void) { return p_gipa; }
extern "C" NX2SL_API PFN_vkGetDeviceProcAddr nx2sl_device_proc(void) { return p_gdpa; }

extern "C" NX2SL_API int nx2sl_available(void) {
    if (g_available >= 0) return g_available;
    bool loaded = false;
    g_available = p_loaded && p_loaded(sl::kFeatureDLSS_G, loaded) == sl::Result::eOk && loaded;
    void *f = nullptr;
    if (g_available && p_feature(sl::kFeatureDLSS_G, "slDLSSGSetOptions", f) == sl::Result::eOk) p_dlssg_options = (PFun_slDLSSGSetOptions *)f;
    if (g_available && p_feature(sl::kFeatureDLSS_G, "slDLSSGGetState", f) == sl::Result::eOk) p_dlssg_state = (PFun_slDLSSGGetState *)f;
    if (p_feature(sl::kFeaturePCL, "slPCLSetMarker", f) == sl::Result::eOk) p_marker = (PFun_slPCLSetMarker *)f;
    if (p_feature(sl::kFeatureReflex, "slReflexSetOptions", f) == sl::Result::eOk) p_reflex = (PFun_slReflexSetOptions *)f;
    if (p_feature(sl::kFeatureReflex, "slReflexGetState", f) == sl::Result::eOk) p_reflex_state = (PFun_slReflexGetState *)f;
    if (p_feature(sl::kFeatureReflex, "slReflexSleep", f) == sl::Result::eOk) p_sleep = (PFun_slReflexSleep *)f;
    if (!p_dlssg_options || !p_dlssg_state || !p_marker || !p_reflex || !p_sleep || !p_reflex_state) g_available = 0;
    // How many frames this GPU and driver generate per rendered frame: known once the device exists (else after the
    // first presents, nx2sl_present_marker)
    sl::DLSSGState s{};
    if (g_available && p_dlssg_state && p_dlssg_state(sl::ViewportHandle(0), s, nullptr) == sl::Result::eOk)
        g_max.store(s.numFramesToGenerateMax);
    if (!g_available) say("dlssg: frame generation unavailable (loaded %d, FG options/state %d/%d, PCL marker %d, Reflex options/sleep/state %d/%d/%d)",
                          loaded, !!p_dlssg_options, !!p_dlssg_state, !!p_marker, !!p_reflex, !!p_sleep, !!p_reflex_state);
    else if (g_max.load()) say("dlssg: DLSS Frame Generation available, up to %ux", g_max.load() + 1);
    else say("dlssg: DLSS Frame Generation available (its most frames per rendered frame not reported yet)");
    return g_available;
}

extern "C" NX2SL_API void nx2sl_log_unavailable(VkPhysicalDevice physical) {
    auto supported = proc<PFun_slIsFeatureSupported>("slIsFeatureSupported");
    if (!supported) return say("dlssg: Streamline has no adapter support query");
    sl::AdapterInfo adapter{};
    adapter.vkPhysicalDevice = physical;
    sl::Result r = supported(sl::kFeatureDLSS_G, adapter);
    say("dlssg: Streamline's support check for frame generation: %s (%d)", sl::getResultAsStr(r), (int)r);
}

// Streamline sets Reflex's mode on the swapchain it was told of last (VK_NV_low_latency2) and drops it while there is
// none, so it is set again after each swapchain.
extern "C" NX2SL_API void nx2sl_reflex(int mode) {
    if (g_available != 1 || !p_reflex) return;
    sl::ReflexOptions o{};
    o.mode = mode == 2 ? sl::ReflexMode::eLowLatencyWithBoost : mode ? sl::ReflexMode::eLowLatency : sl::ReflexMode::eOff;
    std::lock_guard<std::mutex> lock(g_reflex_options);
    sl::Result r = p_reflex(o);
    say("dlssg: Reflex mode request %s, frameLimitUs %u: %s", mode == 2 ? "on with boost" : mode ? "on" : "off",
        o.frameLimitUs, sl::getResultAsStr(r));
}

extern "C" NX2SL_API void nx2sl_wait(uint32_t frame, int hold) {
    if (g_available != 1) return;
    sl::FrameToken *t = token(frame);
    // This window belongs to the game thread. Time inside the SDK call can include CPU scheduling.
    static unsigned starts, calls, errors, idle, missing;
    static Clock::duration sleepTime{};
    if (!hold) ++idle;
    else if (!t) ++missing;
    else {
        auto start = Clock::now();
        sl::Result result = p_sleep(*t);
        sleepTime += Clock::now() - start;
        ++calls;
        if (result != sl::Result::eOk && ++errors == 1)
            say("dlssg: Reflex sleep failed on frame %u: %s", frame, sl::getResultAsStr(result));
    }
    if (++starts == pacingWindow) {
        double ms = std::chrono::duration<double, std::milli>(sleepTime).count();
        say("dlssg: Reflex through game frame %u: sleep calls %u, failures %u, skipped idle %u, missing token %u, mean time in call %.3f ms",
            frame, calls, errors, idle, missing, calls ? ms / calls : 0.0);
        starts = calls = errors = idle = missing = 0;
        sleepTime = {};
    }
}

extern "C" NX2SL_API void nx2sl_phase(uint32_t frame, Nx2SlPhase phase) {
    static const sl::PCLMarker markers[] = {sl::PCLMarker::eSimulationStart, sl::PCLMarker::eSimulationEnd,
                                            sl::PCLMarker::eRenderSubmitStart};
    if (g_available != 1 || (unsigned)phase >= sizeof markers / sizeof *markers) return;
    if (auto *t = token(frame)) mark(markers[phase], *t, frame);
}

// Original bridge ABI for executables that only report image acquisition.
extern "C" NX2SL_API void nx2sl_sleep(uint32_t frame, int hold) {
    nx2sl_wait(frame, hold);
    if (auto *t = g_available == 1 ? token(frame) : nullptr) {
        p_marker(sl::PCLMarker::eSimulationStart, *t);
        p_marker(sl::PCLMarker::eSimulationEnd, *t);
        p_marker(sl::PCLMarker::eRenderSubmitStart, *t);
    }
}

extern "C" NX2SL_API void nx2sl_submit_end(uint32_t frame) {
    sl::FrameToken *t = g_available == 1 ? token(frame) : nullptr;
    if (t) mark(sl::PCLMarker::eRenderSubmitEnd, *t, frame);
}

// Written and consumed on the present thread, once for each backbuffer.
static sl::Extent backbufferRect{};
extern "C" NX2SL_API void nx2sl_present_rect(const int rect[4]) {
    backbufferRect = {(uint32_t)rect[1], (uint32_t)rect[0], (uint32_t)rect[2], (uint32_t)rect[3]};
}

// Set immediately before nx2sl_frame on the present thread. The optional export keeps older hosts compatible.
static Nx2SlImage pendingAlpha{};
extern "C" NX2SL_API void nx2sl_ui_alpha(const Nx2SlImage *alpha) { pendingAlpha = alpha ? *alpha : Nx2SlImage{}; }

static int frame_inputs(VkCommandBuffer cb, const Nx2SlImage *depth, const Nx2SlImage *mv,
                        const Nx2SlImage *hudless, const Nx2SlImage *alpha, const Nx2SlCamera *cam,
                        uint32_t out_w, uint32_t out_h, int enable, uint32_t frame,
                        sl::Extent rect, Nx2SlInputReceipt *receipt, bool *configured = nullptr) {
    if (g_available != 1 || !frame) return 0;
    sl::ViewportHandle vp(0);
    // never more than Streamline allows: one while it has not said
    uint32_t max = g_max.load();
    int n = enable > 0 ? enable : 0;
    if (n > (int)(max ? max : 1)) n = (int)(max ? max : 1);
    bool recompose = n && depth && mv && cam && hudless && alpha && alpha->image && alpha->view;
    bool inputs_accepted = false, off_tags_accepted = false;
    if (n && depth && mv && cam) {
        sl::Constants c{};
        matrix(c.cameraViewToClip, cam->view_to_clip);
        matrix(c.clipToCameraView, cam->clip_to_view);
        matrix(c.clipToPrevClip, cam->clip_to_prev_clip);
        matrix(c.prevClipToClip, cam->prev_clip_to_clip);
        c.jitterOffset = sl::float2(cam->jitter[0], cam->jitter[1]);
        c.mvecScale = sl::float2(cam->mv_scale[0], cam->mv_scale[1]);
        c.cameraPinholeOffset = sl::float2(0, 0);
        c.cameraPos = sl::float3(cam->pos[0], cam->pos[1], cam->pos[2]);
        c.cameraUp = sl::float3(cam->up[0], cam->up[1], cam->up[2]);
        c.cameraRight = sl::float3(cam->right[0], cam->right[1], cam->right[2]);
        c.cameraFwd = sl::float3(cam->fwd[0], cam->fwd[1], cam->fwd[2]);
        c.cameraNear = cam->znear;
        c.cameraFar = cam->zfar;
        c.cameraFOV = cam->fov_y;
        c.cameraAspectRatio = cam->aspect;
        c.depthInverted = sl::Boolean::eFalse;
        c.cameraMotionIncluded = sl::Boolean::eTrue;
        c.motionVectors3D = sl::Boolean::eFalse;
        c.reset = cam->reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        sl::Result constantsResult;
        {
            std::lock_guard<std::mutex> lock(g_token_slot);
            sl::FrameToken *t = token_locked(frame);
            if (!t) return 0;
            constantsResult = p_constants(c, *t, vp);
        }
        if (constantsResult != sl::Result::eOk) recompose = false;
        static unsigned constantErrors;
        if (constantsResult != sl::Result::eOk && constantErrors++ < 8)
            say("dlssg: camera constants rejected on frame %u (%d)", frame, (int)constantsResult);
        sl::Resource rd = resource(depth), rm = resource(mv), rh = hudless ? resource(hudless) : sl::Resource{};
        sl::Extent ed{0, 0, depth->width, depth->height}, em{0, 0, mv->width, mv->height};
        sl::Extent eh = rect ? rect : sl::Extent{0, 0, out_w, out_h};
        // Replay can write the next scene between the blit submission and the paced present.
        // Preserve this frame's depth and motion on the blit command buffer while its color matches.
        sl::Resource ra = recompose ? resource(alpha) : sl::Resource{};
        sl::ResourceTag tags[5] = {sl::ResourceTag(&rd, sl::kBufferTypeDepth, sl::ResourceLifecycle::eOnlyValidNow, &ed),
                                   sl::ResourceTag(&rm, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eOnlyValidNow, &em),
                                   sl::ResourceTag(hudless ? &rh : nullptr, sl::kBufferTypeHUDLessColor, sl::ResourceLifecycle::eValidUntilPresent, &eh)};
        unsigned count = 3;   // a null HUD-less tag also clears the previous picture
        tags[count++] = sl::ResourceTag(recompose ? &ra : nullptr, sl::kBufferTypeUIAlpha,
                                        sl::ResourceLifecycle::eValidUntilPresent, &eh);
        if (rect)
            tags[count++] = sl::ResourceTag(nullptr, sl::kBufferTypeBackbuffer,
                                             sl::ResourceLifecycle::eValidUntilPresent, &rect);
        uint32_t mask = (hudless ? NX2SL_INPUT_HUDLESS : 0) | (recompose ? NX2SL_INPUT_UI_ALPHA : 0);
        uint32_t sdk_frame = 0;
        sl::Result result = tag_frame(frame, vp, tags, count, cb, &sdk_frame);
        inputs_accepted = sdk_frame && result == sl::Result::eOk && constantsResult == sl::Result::eOk;
        if (receipt && sdk_frame) {
            receipt->sdk_frame = sdk_frame;
            if (result == sl::Result::eOk) receipt->accepted = mask;
            else receipt->uncertain = mask;
        }
        if (result != sl::Result::eOk) recompose = false;
        static unsigned tagErrors;
        if (result != sl::Result::eOk && tagErrors++ < 8)
            say("dlssg: frame %u resource tags rejected (%d)", frame, (int)result);
    } else if (n || receipt) {
        // Explicitly clear both late inputs for this SDK frame, including typed Off presents.
        sl::ResourceTag absent[] = {
            {nullptr, sl::kBufferTypeHUDLessColor, sl::ResourceLifecycle::eValidUntilPresent},
            {nullptr, sl::kBufferTypeUIAlpha, sl::ResourceLifecycle::eValidUntilPresent}};
        uint32_t sdk_frame = 0;
        sl::Result result = tag_frame(frame, vp, absent, 2, cb, &sdk_frame);
        if (!sdk_frame) return 0;
        off_tags_accepted = result == sl::Result::eOk;
    } else if (!token(frame)) return 0;
    static bool recomposing = false;
    if (n != g_frames || recompose != recomposing) {
        sl::DLSSGOptions o{};
        o.mode = n ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
        o.enableUserInterfaceRecomposition = recompose ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        o.numFramesToGenerate = n ? (uint32_t)n : 1;
        // Generation goes off for every menu, cutscene and frame without depth and motion, and on again after it. Off
        // frees the generator's resources unless this flag is set, and on allocates them again inside the present, with
        // the queue lock held: a frame of 100 to 150 ms each time, the submitting thread waiting for that lock. With
        // the flag, off only suspends. Whether frame generation runs at all is fixed for the run (vk_dlssg.c), so
        // nothing is held for long that would otherwise be given back; Streamline frees it at shutdown.
        o.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
        sl::Result r = p_dlssg_options(vp, o);
        if (n) say("dlssg: frame generation on, %dx (%d generated per rendered frame) (%d), output %ux%u", n + 1, n, (int)r, out_w, out_h);
        else say("dlssg: frame generation off (%d), output %ux%u", (int)r, out_w, out_h);
        if (r == sl::Result::eOk) {
            if (recompose != recomposing) say("dlssg: UI alpha recomposition %s", recompose ? "on" : "off");
            recomposing = recompose;
            g_frames = n;
        }
    }
    if (configured) *configured = inputs_accepted && enable == n && g_frames == n && recomposing == recompose;
    if (receipt && !n) return off_tags_accepted && !g_frames;
    return 1;
}

extern "C" NX2SL_API int nx2sl_frame(VkCommandBuffer cb, const Nx2SlImage *depth, const Nx2SlImage *mv,
                                     const Nx2SlImage *hudless, const Nx2SlCamera *cam, uint32_t out_w, uint32_t out_h,
                                     int enable, uint32_t frame) {
    Nx2SlImage alpha = pendingAlpha;
    pendingAlpha = {};
    if (g_pending_inputs.ticket) return 0;
    return frame_inputs(cb, depth, mv, hudless, &alpha, cam, out_w, out_h, enable, frame, backbufferRect, nullptr);
}

extern "C" NX2SL_API int nx2sl_frame_inputs_v1(const Nx2SlFrameInputsV1 *in, Nx2SlInputReceipt *out) {
    if (!input_abi(in) || !input_abi(out)) return 0;
    *out = {sizeof *out, NX2SL_INPUTS_ABI_V1};
    if (!in->generation || !in->swapchain || !in->cb || g_pending_inputs.ticket || g_next_ticket == UINT64_MAX) return 0;
    uint32_t frame = (uint32_t)in->frame;
    *out = {sizeof *out, NX2SL_INPUTS_ABI_V1, 0, 0, 0, in->frame, in->generation, in->swapchain, frame, 0, 0};
    sl::Extent rect{(uint32_t)in->rect[1], (uint32_t)in->rect[0], (uint32_t)in->rect[2], (uint32_t)in->rect[3]};
    const Nx2SlImage *hudless = in->hudless && in->hudless->image && in->hudless->view ? in->hudless : nullptr;
    bool configured = false;
    g_fg_prepared = {};
    int okay = frame_inputs(in->cb, in->depth, in->motion, hudless, in->alpha, in->camera,
                            in->width, in->height, in->generated, frame, rect, out, &configured);
    if (okay && configured) {
        auto &config = g_fg_prepared.config;
        g_fg_prepared.frame = in->frame;
        config.generation = in->generation;
        config.swapchain = in->swapchain;
        config.generated = static_cast<uint32_t>(in->generated);
        config.mvec_width = in->motion->width;
        config.mvec_height = in->motion->height;
        config.depth_width = in->depth->width;
        config.depth_height = in->depth->height;
        config.color_width = in->width;
        config.color_height = in->height;
        config.motion = in->motion->format;
        config.depth = in->depth->format;
        if (out->accepted & NX2SL_INPUT_HUDLESS) config.hudless = hudless->format;
        if (out->accepted & NX2SL_INPUT_UI_ALPHA) config.alpha = in->alpha->format;
        config.recomposition = !!(out->accepted & NX2SL_INPUT_UI_ALPHA);
    }
    if (out->accepted || out->uncertain) {
        out->ticket = ++g_next_ticket;
        g_pending_inputs = *out;
    }
    return okay;
}

static InputSample present_marker(int after, uint32_t frame, bool accepted, bool *marked = nullptr,
                                  bool *generated = nullptr) {
    if (after) {
        g_input_fence = VK_NULL_HANDLE;
        g_input_value = 0;
    }
    {
        std::lock_guard<std::mutex> lock(g_token_slot);
        sl::FrameToken *t = g_available == 1 ? token_locked(frame) : nullptr;
        if (!t) return {};
        bool ok = mark(after ? sl::PCLMarker::ePresentEnd : sl::PCLMarker::ePresentStart, *t, frame);
        if (marked) *marked = ok;
    }
    if (!after || !accepted) return {};
    g_presented++;
    // Asked after each present, numFramesActuallyPresented is what that present showed: 2 with one generated frame, 4
    // with three.
    sl::DLSSGState s{};
    static unsigned stateErrors;
    g_input_fence = VK_NULL_HANDLE;
    g_input_value = 0;
    sl::Result result = p_dlssg_state(sl::ViewportHandle(0), s, nullptr);
    if (result != sl::Result::eOk) {
        if (stateErrors++ % pacingWindow == 0)
            say("dlssg: FG state query after present failed: %s", sl::getResultAsStr(result));
        return {};
    }
    if (generated) *generated = g_frames > 0 && s.status == sl::DLSSGStatus::eOk && s.numFramesActuallyPresented > 1;
    g_input_fence = (VkSemaphore)(uintptr_t)s.inputsProcessingCompletionFence;
    g_input_value = s.lastPresentInputsProcessingCompletionFenceValue;
    uint64_t shown = g_shown.fetch_add(s.numFramesActuallyPresented, std::memory_order_relaxed) + s.numFramesActuallyPresented;
    if (s.numFramesToGenerateMax && s.numFramesToGenerateMax != g_max.load()) {   // a 0 now and then is no news
        say("dlssg: up to %ux on this GPU and driver", s.numFramesToGenerateMax + 1);
        g_max.store(s.numFramesToGenerateMax);
    }
    // Count presentations over elapsed time. A single multiplier sample cannot distinguish 167 from 170 FPS.
    static uint64_t startPresent, startShown;
    static Clock::time_point startTime;
    if (!startPresent || stateErrors || g_presented - startPresent >= pacingWindow) {
        auto now = Clock::now();
        if (startPresent && !stateErrors) {
            double seconds = std::chrono::duration<double>(now - startTime).count();
            say("dlssg: through game frame %u: %.2f app presents/s, %.2f Streamline frames presented/s, latest %u frames/present, status %u, FG VSync support %d",
                frame, (g_presented - startPresent) / seconds, (shown - startShown) / seconds,
                s.numFramesActuallyPresented, (unsigned)s.status, (int)s.bIsVsyncSupportAvailable);
        }
        // Query on a live swapchain. The Vulkan backend asks the driver about its sleep and latency reports here.
        sl::ReflexState state{};
        {
            std::lock_guard<std::mutex> lock(g_reflex_options);
            result = p_reflex_state(state);
        }
        if (result == sl::Result::eOk)
            say("dlssg: Reflex driver low latency available %d, latency reports available %d",
                state.lowLatencyAvailable, state.latencyReportAvailable);
        else say("dlssg: Reflex state query failed: %s", sl::getResultAsStr(result));
        startTime = now;
        startPresent = g_presented;
        startShown = shown;
        stateErrors = 0;
    }
    return input_sample(s);
}

extern "C" NX2SL_API void nx2sl_present_marker(int after, uint32_t frame) {
    if (!after) g_fg_execution = {};
    InputSample observed = present_marker(after, frame, true);
    if (after) {
        g_last_input_sample = observed;
        g_sample_generation = 0;
    }
}

static bool fresh_input_sample(InputSample before, InputSample after, bool new_generation) {
    if (!before.checked || !after.checked || !after.semaphore || !after.value) return false;
    // A fresh swapchain observation may introduce a different timeline. A
    // reused numeric value on the same observed timeline still cannot retire it.
    if (g_last_proof.semaphore == after.semaphore && after.value <= g_last_proof.value) return false;
    if (before.semaphore == after.semaphore) return after.value > before.value;
    return !before.semaphore || new_generation;
}

extern "C" NX2SL_API VkResult nx2sl_present_inputs_v1(PFN_vkQueuePresentKHR present, VkQueue queue,
        const VkPresentInfoKHR *pi, const Nx2SlInputReceipt *receipt, Nx2SlInputCompletion *out) {
    g_fg_execution = {};
    if (!input_abi(out)) return VK_ERROR_INITIALIZATION_FAILED;
    *out = {sizeof *out, NX2SL_INPUTS_ABI_V1, NX2SL_INPUT_UNPROVEN};
    if (!input_abi(receipt) || g_available != 1 || !present || !pi || !receipt->generation || !receipt->swapchain ||
        receipt->sdk_frame != (uint32_t)receipt->frame || pi->swapchainCount != 1 || !pi->pSwapchains ||
        pi->pSwapchains[0] != receipt->swapchain) return VK_ERROR_INITIALIZATION_FAILED;
    bool late = receipt->accepted || receipt->uncertain;
    if (late != !!receipt->ticket || (g_pending_inputs.ticket && !same_inputs(*receipt, g_pending_inputs)) ||
        (late && !g_pending_inputs.ticket)) return VK_ERROR_INITIALIZATION_FAILED;
    out->ticket = receipt->ticket;
    if (!late) out->flags = NX2SL_INPUT_NO_LATE_TAGS;
    bool changed_generation = !g_sample_generation || receipt->generation != g_sample_generation;
    InputSample before = g_last_input_sample;
    if (late && (changed_generation || !before.checked)) before = query_input_sample();
    g_pending_inputs = {}; // After any actual present attempt a null tag cannot prove cancellation.
    FgExecutionObservation prepared = g_fg_prepared;
    g_fg_prepared = {};
    bool started = false, ended = false;
    present_marker(0, receipt->sdk_frame, false, &started);
    VkResult result = present(queue, pi);
    bool accepted = (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) &&
                    (!pi->pResults || pi->pResults[0] == VK_SUCCESS || pi->pResults[0] == VK_SUBOPTIMAL_KHR);
    bool generated = false;
    InputSample observed = present_marker(1, receipt->sdk_frame, accepted, &ended, &generated);
    if (accepted && started && ended && generated && prepared.frame == receipt->frame &&
        prepared.config.generation == receipt->generation && prepared.config.swapchain == receipt->swapchain)
        g_fg_execution = prepared;
    g_last_input_sample = observed;
    g_sample_generation = receipt->generation;
    if (late && accepted && started && ended && fresh_input_sample(before, observed, changed_generation)) {
        g_last_proof = observed;
        out->flags = NX2SL_INPUT_TIMELINE;
        out->semaphore = observed.semaphore;
        out->value = observed.value;
    }
    return result;
}

extern "C" NX2SL_API int nx2sl_cancel_inputs_v1(const Nx2SlInputReceipt *receipt) {
    if (!input_abi(receipt) || !receipt->ticket || !same_inputs(*receipt, g_pending_inputs)) return 0;
    const sl::ResourceTag absent[] = {
        {nullptr, sl::kBufferTypeHUDLessColor, sl::ResourceLifecycle::eValidUntilPresent},
        {nullptr, sl::kBufferTypeUIAlpha, sl::ResourceLifecycle::eValidUntilPresent}};
    if (tag_frame(receipt->sdk_frame, sl::ViewportHandle(0), absent, 2, VK_NULL_HANDLE) != sl::Result::eOk) return 0;
    g_pending_inputs = {};
    g_fg_prepared = {};
    return 1;
}

extern "C" NX2SL_API int nx2sl_estimate_fg_v1(const Nx2SlFgConfigV1 *config, Nx2SlFgEstimateV1 *out) {
    if (!out || out->size != sizeof *out || out->version != NX2SL_FG_MEMORY_ABI_V1 || out->flags ||
        out->reserved || out->padding) return 0;
    *out = {sizeof *out, NX2SL_FG_MEMORY_ABI_V1, 0, 0, 0, -1, 0};
    if (!config || config->size != sizeof *config || config->version != NX2SL_FG_MEMORY_ABI_V1 ||
        config->flags || config->reserved || !config->generation || !config->swapchain ||
        !config->backbuffers || !config->generated || !config->color_width || !config->color_height ||
        !config->mvec_width || !config->mvec_height || config->mvec_width != config->depth_width ||
        config->mvec_height != config->depth_height || config->color == VK_FORMAT_UNDEFINED ||
        config->depth == VK_FORMAT_UNDEFINED || config->motion == VK_FORMAT_UNDEFINED ||
        config->recomposition > 1 || (config->recomposition &&
        (config->hudless == VK_FORMAT_UNDEFINED || config->alpha == VK_FORMAT_UNDEFINED)) ||
        config->queue_mode >= static_cast<uint32_t>(sl::DLSSGQueueParallelismMode::eCount) ||
        g_available != 1 || !p_dlssg_state) return 0;
    uint32_t max = g_max.load();
    if (config->generated > (max ? max : 1)) return 0;

    sl::DLSSGOptions options{};
    options.mode = sl::DLSSGMode::eOn;
    options.flags = sl::DLSSGFlags::eRequestVRAMEstimate | sl::DLSSGFlags::eRetainResourcesWhenOff;
    options.numFramesToGenerate = config->generated;
    options.numBackBuffers = config->backbuffers;
    options.mvecDepthWidth = config->mvec_width;
    options.mvecDepthHeight = config->mvec_height;
    options.colorWidth = config->color_width;
    options.colorHeight = config->color_height;
    options.colorBufferFormat = static_cast<uint32_t>(config->color);
    options.depthBufferFormat = static_cast<uint32_t>(config->depth);
    options.mvecBufferFormat = static_cast<uint32_t>(config->motion);
    options.hudLessBufferFormat = static_cast<uint32_t>(config->hudless);
    options.uiBufferFormat = static_cast<uint32_t>(config->alpha);
    options.enableUserInterfaceRecomposition = config->recomposition ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    options.queueParallelismMode = static_cast<sl::DLSSGQueueParallelismMode>(config->queue_mode);

    sl::DLSSGState state{};
    const sl::Result result = p_dlssg_state(sl::ViewportHandle(0), state, &options);
    out->sdk_result = static_cast<int32_t>(result);
    if (result != sl::Result::eOk) return 0;
    // GetState consumes the SDK's present delta even when called for a cold estimate.
    if (state.numFramesActuallyPresented)
        g_shown.fetch_add(state.numFramesActuallyPresented, std::memory_order_relaxed);
    if (!state.estimatedVRAMUsageInBytes) return 0;
    out->flags = NX2SL_FG_ESTIMATE_ROUGH_VRAM;
    out->vram_bytes = state.estimatedVRAMUsageInBytes;
    return 1;
}

extern "C" NX2SL_API int nx2sl_fg_observed_v1(uint64_t frame, const Nx2SlFgConfigV1 *config) {
    if (!config || config->size != sizeof *config || config->version != NX2SL_FG_MEMORY_ABI_V1 ||
        config->flags || config->reserved || !frame || g_fg_execution.frame != frame || config->queue_mode) return 0;
    const auto &actual = g_fg_execution.config;
    // Swapchain shape/count are established by the host's exact generation. Vulkan image tags and applied options
    // are independently checked here, including configurations the SDK declined after tagging.
    if (actual.generation != config->generation || actual.swapchain != config->swapchain ||
        actual.generated != config->generated || actual.recomposition != config->recomposition ||
        actual.mvec_width != config->mvec_width || actual.mvec_height != config->mvec_height ||
        actual.depth_width != config->depth_width || actual.depth_height != config->depth_height ||
        actual.color_width != config->color_width || actual.color_height != config->color_height ||
        actual.depth != config->depth || actual.motion != config->motion ||
        actual.hudless != config->hudless || actual.alpha != config->alpha) return 0;
    g_fg_execution = {};
    return 1;
}

extern "C" NX2SL_API uint32_t nx2sl_frames_max(void) { return g_available == 1 ? g_max.load() : 0; }

extern "C" NX2SL_API int nx2sl_input_retirement(VkSemaphore *semaphore, uint64_t *value) {
    if (g_frames == 0) { // no persistent inputs consumed by this present
        *semaphore = VK_NULL_HANDLE;
        *value = 0;
        return 1;
    }
    *semaphore = g_input_fence;
    *value = g_input_value;
    return g_input_fence && g_input_value;
}

extern "C" NX2SL_API uint64_t nx2sl_shown(void) { return g_shown.load(std::memory_order_relaxed); }

extern "C" NX2SL_API int nx2sl_shutdown_checked_v1(void) {
    if (!g_initialized) return 1;
    if (!p_shutdown || p_shutdown() != sl::Result::eOk) return 0;
    g_initialized = false;
    g_available = -1;
    g_pending_inputs = {};
    g_fg_prepared = {};
    g_fg_execution = {};
    g_last_input_sample = {};
    g_last_proof = {};
    g_sample_generation = 0;
    g_input_fence = VK_NULL_HANDLE;
    g_input_value = 0;
    return 1;
}

extern "C" NX2SL_API void nx2sl_shutdown(void) { (void)nx2sl_shutdown_checked_v1(); }

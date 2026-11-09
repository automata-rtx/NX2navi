// nx2navi_dlssg.dll: DLSS Frame Generation through NVIDIA Streamline, behind the C interface of nx2_sl.h. Built with
// MSVC against the Streamline SDK's headers (tools/build_dlssg.py; the SDK stays outside the repository), so its C++
// types have the ABI of Streamline's own DLLs. sl.interposer.dll and its plugins are loaded from the game's folder at
// run time; this DLL links none of them.
//
// Manual hooking (the SDK's ProgrammingGuideManualHooking): the game creates its instance and device through the
// interposer's vkCreateInstance and vkCreateDevice, which add what DLSS-G needs, and calls the interposer's
// swapchain functions; everything else goes to the Vulkan loader as before. Each frame gets a frame token, the PCL
// latency markers, the camera constants and the depth and motion vector tags (valid until the present), then the
// present goes through the interposer, which inserts the generated frame.
#define NX2SL_BUILD
#include <windows.h>

#include <stdio.h>
#include <string.h>

#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss_g.h>
#include <sl_pcl.h>
#include <sl_reflex.h>

#include "nx2_sl.h"

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
PFun_slReflexSleep *p_sleep;
PFN_vkGetInstanceProcAddr p_gipa;
PFN_vkGetDeviceProcAddr p_gdpa;
sl::FrameToken *g_frame;
int g_available = -1, g_multiplier = -1;
uint64_t g_presented;
uint32_t g_generated;
uint64_t g_output_frames;
uint64_t g_interval_start, g_interval_output;

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
    if (p_feature(sl::kFeatureReflex, "slReflexSleep", f) == sl::Result::eOk) p_sleep = (PFun_slReflexSleep *)f;
    if (!p_dlssg_options || !p_dlssg_state || !p_marker || !p_reflex || !p_sleep) g_available = 0;
    if (g_available) {
        sl::ReflexOptions o{};
        o.mode = sl::ReflexMode::eLowLatency;
        sl::Result r = p_reflex(o);
        if (r != sl::Result::eOk) {
            say("dlssg: Reflex low latency failed (%d)", (int)r);
            g_available = 0;
        }
    }
    say("dlssg: DLSS Frame Generation %s", g_available ? "available" : "not available on this GPU or driver");
    return g_available;
}

extern "C" NX2SL_API int nx2sl_frame(VkCommandBuffer cb, const Nx2SlImage *depth, const Nx2SlImage *mv,
                                     const Nx2SlImage *hudless, const Nx2SlCamera *cam, uint32_t out_w, uint32_t out_h,
                                     int multiplier) {
    if (g_available != 1) return 0;
    if (p_token(g_frame, nullptr) != sl::Result::eOk || !g_frame) return 0;
    p_sleep(*g_frame);
    p_marker(sl::PCLMarker::eSimulationStart, *g_frame);
    p_marker(sl::PCLMarker::eSimulationEnd, *g_frame);
    p_marker(sl::PCLMarker::eRenderSubmitStart, *g_frame);
    sl::ViewportHandle vp(0);
    if (multiplier && depth && mv && cam) {
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
        c.reset = cam->reset || multiplier != g_multiplier ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        sl::Result constants = p_constants(c, *g_frame, vp);
        sl::Resource rd = resource(depth), rm = resource(mv), rh = hudless ? resource(hudless) : sl::Resource{};
        sl::Extent ed{0, 0, depth->width, depth->height}, em{0, 0, mv->width, mv->height};
        const VkRect2D &rect = cam->output_rect;
        sl::Extent picture{(uint32_t)rect.offset.y, (uint32_t)rect.offset.x, rect.extent.width, rect.extent.height};
        // The present worker submits this command buffer before waiting for its flip time.
        // The replay can already be writing the next scene's depth and motion before that flip.
        sl::ResourceTag tags[] = {sl::ResourceTag(&rd, sl::kBufferTypeDepth, sl::ResourceLifecycle::eOnlyValidNow, &ed),
                                   sl::ResourceTag(&rm, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eOnlyValidNow, &em),
                                   sl::ResourceTag(hudless ? &rh : nullptr, sl::kBufferTypeHUDLessColor,
                                                   sl::ResourceLifecycle::eValidUntilPresent, hudless ? &picture : nullptr),
                                   sl::ResourceTag(nullptr, sl::kBufferTypeBackbuffer, sl::ResourceLifecycle::eValidUntilPresent, &picture)};
        sl::Result tagged = p_tag(*g_frame, vp, tags, sizeof tags / sizeof tags[0], (sl::CommandBuffer *)cb);
        if (constants != sl::Result::eOk || tagged != sl::Result::eOk) {
            static int warnings;
            if (warnings++ < 8) say("dlssg: frame rejected: constants %d, resources %d", (int)constants, (int)tagged);
            multiplier = 0;
        }
    }
    if (multiplier != g_multiplier) {
        sl::DLSSGOptions o{};
        // Menus and frames without camera inputs temporarily suspend interpolation.
        // Keep the NGX resources ready for the next gameplay frame.
        o.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
        o.mode = multiplier ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
        o.numFramesToGenerate = multiplier ? multiplier - 1 : 1;
        sl::Result r = p_dlssg_options(vp, o);
        say("dlssg: set multiplier %d (%d), output %ux%u", multiplier, (int)r, out_w, out_h);
        if (r == sl::Result::eOk) {
            g_multiplier = multiplier;
            g_interval_start = g_presented;
            g_interval_output = g_output_frames;
        }
    }
    p_marker(sl::PCLMarker::eRenderSubmitEnd, *g_frame);
    return 1;
}

extern "C" NX2SL_API int nx2sl_present_marker(int after) {
    if (g_available != 1 || !g_frame) return 0;
    p_marker(after ? sl::PCLMarker::ePresentEnd : sl::PCLMarker::ePresentStart, *g_frame);
    if (after) {
        g_presented++;
        sl::DLSSGState s{};
        sl::Result result = p_dlssg_state(sl::ViewportHandle(0), s, nullptr);
        if (result == sl::Result::eOk) {
            g_generated = s.numFramesActuallyPresented;
            g_output_frames += g_generated;
            uint64_t count = g_presented - g_interval_start;
            if (count == 60 || count >= 300) {
                say("dlssg: %llu game presents, %llu output frames in interval, last %u, multiplier %d, status %u",
                    (unsigned long long)count, (unsigned long long)(g_output_frames - g_interval_output),
                    g_generated, g_multiplier, (unsigned)s.status);
                if (count >= 300) {
                    g_interval_start = g_presented;
                    g_interval_output = g_output_frames;
                }
            }
            if (s.numFramesToGenerateMax >= NX2SL_MAX_MULTIPLIER - 1) return NX2SL_MAX_MULTIPLIER;
            if (s.numFramesToGenerateMax) return (int)s.numFramesToGenerateMax + 1;
        } else if (g_presented == 1 || g_presented % 300 == 0) {
            say("dlssg: could not read the presented frame count (%d)", (int)result);
        }
    }
    return 0;
}

extern "C" NX2SL_API void nx2sl_stats(uint64_t *presented, uint32_t *generated_per_frame) {
    *presented = g_presented, *generated_per_frame = g_generated;
}

extern "C" NX2SL_API uint64_t nx2sl_output_frames(void) { return g_output_frames; }

extern "C" NX2SL_API void nx2sl_shutdown(void) {
    if (p_shutdown && g_available >= 0) p_shutdown();
}

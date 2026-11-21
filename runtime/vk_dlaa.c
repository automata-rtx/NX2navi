// DLAA: NVIDIA DLSS at the scene's own size (NX2NAVI_DLAA, nx2navi.ini dlaa). NGX is loaded at run time from the driver's
// NGX core: on Windows _nvngx.dll, found through HKLM\SOFTWARE\NVIDIA Corporation\Global\NGXCore; on Linux
// libnvidia-ngx.so.1, which the NVIDIA driver installs on the library path. NGX loads the DLSS model from the folder of
// the executable: nvngx_dlss.dll on Windows, libnvidia-ngx-dlss.so.<version> on Linux. No NVIDIA SDK code or header is
// compiled in: the few NGX types used are declared here from the SDK ABI and the Linux core's shutdown interface.
//
// The core's parameter object is a C++ class whose virtual functions are called through its vtable: the Set overloads
// in slots 0-7 and the Get overloads in 8-15. MSVC (the Windows core) lays out each group of overloads in reverse
// declaration order, the Itanium C++ ABI (the Linux core, built with GCC) in declaration order. vtable_ok checks the
// layout with a set-then-get round trip before anything relies on it.
//
// vk_dlaa_init brings NGX up on the new device. Temporal size groups retire
// when their replay and graphics-composition uses are done. vk_dlaa_release
// also frees the remaining DLAA resources and shuts down NGX on Linux.
//
// Where DLAA runs in the frame, the jitter and the motion vectors: see the sections below.
#ifdef _WIN32
#include <windows.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "cfg.h"
#include "camera_history.h"
#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "platform/platform.h"
#include "vk.h"
#include "vk_dlaa.h"
#include "vk_dlssg.h"
#include "vk_exec.h"
#include "resource_core_vma.h"
#include "resource_command_effects.h"
#include "resource_image_write.h"
#include "texture_resource.h"
#include "vk_ffx.h"
#include "vk_fg.h"
#include "vk_rec.h"
#include "dlaa_mv_spv.h"

typedef int NgxResult;   // NVSDK_NGX_Result: 1 success, 0xBAD00000 | n failures
#define NGX_OK(r) (((unsigned)(r) & 0xFFF00000u) != 0xBAD00000u)
#define NGX_VERSION_API 0x15
#define NGX_FEATURE_SUPERSAMPLING 1
#define NGX_PERFQ_DLAA 5   // NVSDK_NGX_PerfQuality_Value: 0 performance, 1 balanced, 2 quality, 3 ultra performance
typedef struct NgxParam NgxParam;
typedef struct NgxHandle NgxHandle;
typedef enum { NGX_SR_EMPTY, NGX_SR_OPAQUE_READY, NGX_SR_FAILED } NgxSrState;
typedef struct {
    NgxHandle *handle;
    uint64_t scene;
    NgxSrState state;
} NgxSrFeature;
typedef struct { void *slot[17]; } NgxParamVtbl;
struct NgxParam { const NgxParamVtbl *vt; };
typedef struct {   // NVSDK_NGX_ImageViewInfo_VK
    VkImageView view;
    VkImage image;
    VkImageSubresourceRange range;
    VkFormat format;
    unsigned width, height;
} NgxImageView;
typedef struct {   // NVSDK_NGX_Resource_VK
    union { NgxImageView img; struct { VkBuffer buffer; unsigned bytes; } buf; } r;
    int type;      // 0 image view, 1 buffer
    _Bool rw;
} NgxResourceVK;

typedef NgxResult (*PFN_Init)(unsigned long long app, const wchar_t *path, VkInstance, VkPhysicalDevice, VkDevice,
                              PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr, int version, const void *info);
typedef NgxResult (*PFN_ReqExt)(unsigned *ni, const char ***ie, unsigned *nd, const char ***de);
typedef NgxResult (*PFN_Params)(NgxParam **);
typedef NgxResult (*PFN_DestroyParams)(NgxParam *);
typedef NgxResult (*PFN_Create)(VkCommandBuffer, int feature, NgxParam *, NgxHandle **);
typedef NgxResult (*PFN_Evaluate)(VkCommandBuffer, const NgxHandle *, const NgxParam *, void *callback);
typedef NgxResult (*PFN_Release)(NgxHandle *);
#ifndef _WIN32
// NVSDK_NGX_VULKAN_Shutdown1 as the Linux core exports it: the device and a pointer that receives the devices NGX still
// serves afterwards (vk_dlaa_release).
typedef NgxResult (*PFN_Shutdown1)(VkDevice, unsigned *remaining);
#endif

static struct {
    void *core;   // the NGX core library (an HMODULE on Windows)
    PFN_Init init;
    PFN_ReqExt req_ext;
    PFN_Params caps, alloc_params;
    PFN_DestroyParams destroy_params;
    PFN_Create create;
    PFN_Evaluate evaluate;
    PFN_Release release;
    void *feat_inst_req;   // NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements (vk_ngxfg), NULL when the core lacks it
    int tried, loaded, available;
    int ngx_up;   // vk_dlaa_init brought NGX up: DLSS serves this run
    NgxParam *capsp;   // the capability parameters (vk_dlaa_init), which DLSS's statistics callback reads and fills
    int wanted;   // -1 not read yet
#ifndef _WIN32
    // Linux teardown (vk_dlaa_release)
    PFN_Shutdown1 shutdown1;   // required: core_load leaves DLAA off without it
    VkDevice device;           // what NGX was initialised with: set while vk_dlaa_release owes it a shutdown
    NgxParam *caps_map;        // GetCapabilityParameters' map: allocated for the caller, destroyed by vk_dlaa_release
#endif
} N = {.wanted = -1};

// DLSS Frame Generation through NGX itself (vk_ngxfg_*, at the end of this file): NX2NAVI_FRAMEGEN = dlss where
// Streamline does not run, which is every build but the Windows one.
static struct {
    int max;                     // generated frames per rendered one this GPU allows (MultiFrameCountMax), 0: none
    NgxHandle *feat;
    NgxParam *p;
    uint32_t w, h;               // the picture the feature was created for
    VkFormat fmt;
    int failed, reset, evals;
} FG;

// Whether this run can generate frames through NGX: DLSS upscaling on at the start, as NGX comes up only then. The
// swapchain then gets the presenter's extra images and NGX is asked about frame generation; whether frames are
// generated is NX2NAVI_FRAMEGEN = dlss, read live (vk_ngxfg_on).
int vk_ngxfg_wanted(void) {
#ifdef _WIN32
    return 0;   // Streamline's path there (vk_dlssg.c)
#else
    static int w = -1;
    if (w < 0) w = cfg_bool("NX2NAVI_DLAA") && strcmp(cfg_value("NX2NAVI_TAA"), "fsr3");
    return w;
#endif
}

typedef void (*SetI)(NgxParam *, const char *, int);
typedef void (*SetU)(NgxParam *, const char *, unsigned);
typedef void (*SetF)(NgxParam *, const char *, float);
typedef void (*SetP)(NgxParam *, const char *, void *);
typedef NgxResult (*GetI)(const NgxParam *, const char *, int *);
typedef NgxResult (*GetU)(const NgxParam *, const char *, unsigned *);
typedef NgxResult (*GetF)(const NgxParam *, const char *, float *);
typedef NgxResult (*GetP)(const NgxParam *, const char *, void **);
typedef NgxResult (*GetU64)(const NgxParam *, const char *, unsigned long long *);
// The SDK declares Set(u64), Set(float), Set(double), Set(u32), Set(int), Set(ID3D11Resource *), Set(ID3D12Resource *),
// Set(void *), then the Get overloads in the same order, then Reset.
#ifdef _WIN32
enum { S_P = 0, S_D12, S_D11, S_I, S_U32, S_D, S_F, S_U64, G_P, G_D12, G_D11, G_I, G_U32, G_D, G_F, G_U64 };
#else
enum { S_U64 = 0, S_F, S_D, S_U32, S_I, S_D11, S_D12, S_P, G_U64, G_F, G_D, G_U32, G_I, G_D11, G_D12, G_P };
#endif
static void p_seti(NgxParam *p, const char *n, int v) { ((SetI)p->vt->slot[S_I])(p, n, v); }
static void p_setu(NgxParam *p, const char *n, unsigned v) { ((SetU)p->vt->slot[S_U32])(p, n, v); }
static void p_setf(NgxParam *p, const char *n, float v) { ((SetF)p->vt->slot[S_F])(p, n, v); }
static void p_setp(NgxParam *p, const char *n, void *v) { ((SetP)p->vt->slot[S_P])(p, n, v); }
static int p_geti(const NgxParam *p, const char *n, int *v) { return NGX_OK(((GetI)p->vt->slot[G_I])(p, n, v)); }
static int p_getu(const NgxParam *p, const char *n, unsigned *v) { return NGX_OK(((GetU)p->vt->slot[G_U32])(p, n, v)); }
static int p_getf(const NgxParam *p, const char *n, float *v) { return NGX_OK(((GetF)p->vt->slot[G_F])(p, n, v)); }
static int p_getp(const NgxParam *p, const char *n, void **v) { return NGX_OK(((GetP)p->vt->slot[G_P])(p, n, v)); }
static int p_getull(const NgxParam *p, const char *n, unsigned long long *v) {
    return NGX_OK(((GetU64)p->vt->slot[G_U64])(p, n, v));
}

#ifdef _WIN32
#define DLSS_MODEL "nvngx_dlss.dll"
#else
#define DLSS_MODEL "libnvidia-ngx-dlss.so.<version>"
#endif

// NGX is brought up only when DLSS is wanted at start: its device extensions can only be asked for at device creation,
// and loading the driver's NGX core on every NVIDIA machine would cost start time and change the device with DLAA off.
// So DLSS applies after a restart. FSR 3.1 (amd_fidelityfx_vk.dll, vk_ffx.cpp) asks nothing of the device: NX2NAVI_TAA
// and dlaa follow the setting at once when NGX is not up, and vk_dlaa_frame loads the library the first time it is used.
int vk_dlaa_wanted(void) {
    static int at_start = -1;
    if (at_start < 0) at_start = cfg_bool("NX2NAVI_DLAA");
    return at_start || (!N.ngx_up && cfg_bool("NX2NAVI_DLAA") && !strcmp(cfg_value("NX2NAVI_TAA"), "fsr3"));
}

// The temporal upscaler behind dlaa: NVIDIA DLSS through NGX, once NGX is up, or else AMD FSR 3.1, which takes the same
// inputs. The DLSS mode setting (NX2NAVI_DLSS) is the render scale for both.
static int fsr3(void) { return !N.ngx_up && !strcmp(cfg_value("NX2NAVI_TAA"), "fsr3"); }
static int fsr_frame;   // fsr3() as vk_dlaa_init and then vk_dlaa_frame read it: the upscaler this frame runs with
// FSR 3.1 is the upscaler and its library loads (vk_ffx_available): widescreen.c shrinks the scene only for an upscaler
// that runs.
int vk_dlaa_fsr3(void) { return vk_dlaa_wanted() && fsr3() && vk_ffx_available(); }

// The upscaler the menu has set cannot start in this run: DLSS when NGX was never tried (a device made without its
// extensions), FSR 3.1 while NGX is up and serves the run. A DLSS this PC could not bring up is not waiting for a restart.
int vk_dlaa_needs_restart(void) {
    if (!cfg_bool("NX2NAVI_DLAA")) return 0;
    return strcmp(cfg_value("NX2NAVI_TAA"), "fsr3") ? !N.ngx_up && !N.tried : N.ngx_up;
}

#ifdef _WIN32
// The NGX core's folder when the driver has not written NGXCore\FullPath (seen with driver 617.42, whose key held
// only DLSS override values): the core sits in the driver's own DriverStore folder, the one its Vulkan driver is in.
// That is the display adapter's VulkanDriverName, where the Vulkan loader itself finds the driver.
static int ngx_dir_from_adapter(char *dir, DWORD size) {
    HKEY cls;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}",
                      0, KEY_READ, &cls) != ERROR_SUCCESS)
        return 0;
    int found = 0;
    for (DWORD i = 0; !found; i++) {
        char sub[64], val[MAX_PATH + 1] = {0}, probe[MAX_PATH + 32];
        DWORD sn = sizeof sub, vn = MAX_PATH;
        if (RegEnumKeyExA(cls, i, sub, &sn, NULL, NULL, NULL, NULL) != ERROR_SUCCESS) break;
        if (RegGetValueA(cls, sub, "VulkanDriverName", RRF_RT_REG_SZ | RRF_RT_REG_MULTI_SZ, NULL, val, &vn) != ERROR_SUCCESS)
            continue;
        char *slash = strrchr(val, '\\');
        if (!slash) continue;
        *slash = 0;
        snprintf(probe, sizeof probe, "%s\\_nvngx.dll", val);
        if (GetFileAttributesA(probe) == INVALID_FILE_ATTRIBUTES) continue;
        snprintf(dir, size, "%s", val);
        found = 1;
    }
    RegCloseKey(cls);
    return found;
}
#endif

static int core_load(void) {
    if (N.tried) return N.loaded;
    N.tried = 1;
#ifdef _WIN32
    char dir[MAX_PATH] = {0}, path[MAX_PATH + 32];
    DWORD n = sizeof dir;
    if (RegGetValueA(HKEY_LOCAL_MACHINE, "SOFTWARE\\NVIDIA Corporation\\Global\\NGXCore", "FullPath", RRF_RT_REG_SZ,
                     NULL, dir, &n) != ERROR_SUCCESS && !ngx_dir_from_adapter(dir, sizeof dir)) {
        rt_log("dlaa: the driver has no NGX core (registry NGXCore missing, none beside its Vulkan driver): DLAA off");
        return 0;
    }
    snprintf(path, sizeof path, "%s\\_nvngx.dll", dir);
    N.core = LoadLibraryA(path);
    if (!N.core) {
        rt_log("dlaa: cannot load %s (error %lu): DLAA off", path, GetLastError());
        return 0;
    }
#define FN(f, name) N.f = (void *)GetProcAddress((HMODULE)N.core, name)
#else
    // The driver's NGX core; the dynamic loader finds it where the driver put it (/usr/lib, /usr/lib/x86_64-linux-gnu).
    const char *path = "libnvidia-ngx.so.1";
    N.core = plat_lib_open(path);
    if (!N.core) {
        rt_log("dlaa: cannot load %s (not an NVIDIA driver, or one without NGX): DLAA off", path);
        return 0;
    }
#define FN(f, name) N.f = plat_lib_sym(N.core, name)
#endif
    FN(init, "NVSDK_NGX_VULKAN_Init_Ext2");
#ifndef _WIN32
    FN(shutdown1, "NVSDK_NGX_VULKAN_Shutdown1");   // Linux: checked below
#endif
    FN(req_ext, "NVSDK_NGX_VULKAN_RequiredExtensions");
    FN(caps, "NVSDK_NGX_VULKAN_GetCapabilityParameters");
    FN(alloc_params, "NVSDK_NGX_VULKAN_AllocateParameters");
    FN(destroy_params, "NVSDK_NGX_VULKAN_DestroyParameters");
    FN(create, "NVSDK_NGX_VULKAN_CreateFeature");
    FN(evaluate, "NVSDK_NGX_VULKAN_EvaluateFeature");
    FN(release, "NVSDK_NGX_VULKAN_ReleaseFeature");
    FN(feat_inst_req, "NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements");   // optional: frame generation's
#undef FN
    if (!N.init || !N.req_ext || !N.caps || !N.alloc_params || !N.create || !N.evaluate || !N.release) {
        rt_log("dlaa: the NGX core %s lacks Vulkan entry points: DLAA off", path);
        return 0;
    }
#ifndef _WIN32
    // Without Shutdown1 the device cannot be taken back from NGX (vk_dlaa_release), and the process then segfaults after
    // the Vulkan teardown (NVIDIA 615.78.08): DLAA stays off rather than run with a teardown known to crash. The
    // parameter maps are destroyed there too.
    if (!N.shutdown1 || !N.destroy_params) {
        rt_log("dlaa: the NGX core %s lacks NVSDK_NGX_VULKAN_Shutdown1 or DestroyParameters, which its teardown needs: "
               "DLAA off", path);
        return 0;
    }
#endif
    N.loaded = 1;
    rt_log("dlaa: NGX core %s", path);
    return 1;
}

static int listed(const char *const *list, uint32_t n, const char *s) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(list[i], s)) return 1;
    return 0;
}

// NVSDK_NGX_FeatureDiscoveryInfo with an application id (the layout of the SDK's struct, 64-bit), for the question
// which instance extensions frame generation wants.
typedef struct {
    int sdk, feature;
    struct { int type; union { struct { const char *id; int engine; const char *ver; } project; unsigned long long app; } v; } ident;
    const wchar_t *path;
    const void *info;
} NgxDiscovery;
typedef NgxResult (*PFN_FeatInstReq)(const NgxDiscovery *, uint32_t *, VkExtensionProperties **);
#define NGX_FEATURE_FRAME_GENERATION 11   // NVSDK_NGX_Feature_FrameGeneration
// The names NGX lists are copied: its list may not outlive the call, and the device creation keeps the pointers.
static char fg_names[16][VK_MAX_EXTENSION_NAME_SIZE];
static int fg_nnames;
static const char *fg_keep(const char *name) {
    for (int i = 0; i < fg_nnames; i++)
        if (!strcmp(fg_names[i], name)) return fg_names[i];
    if (fg_nnames >= 16) return NULL;
    snprintf(fg_names[fg_nnames], sizeof fg_names[0], "%s", name);
    return fg_names[fg_nnames++];
}

void vk_dlaa_instance_exts(const char **ext, uint32_t *n, uint32_t max) {
    unsigned ni = 0, nd = 0;
    const char **ie = NULL, **de = NULL;
    if (!vk_dlaa_wanted() || fsr3() || !core_load() || !NGX_OK(N.req_ext(&ni, &ie, &nd, &de))) return;
    for (unsigned i = 0; i < ni; i++)
        if (!listed(ext, *n, ie[i]) && *n < max) ext[(*n)++] = ie[i];
    // Frame generation's own instance extensions (vk_ngxfg_evaluate). Its device extensions are not asked for: the
    // core's GetFeatureDeviceExtensionRequirements answered "not implemented" (0xBAD00012) with NVIDIA's 615 Linux
    // driver, and NGX's general list (vk_dlaa_device_exts) served the feature.
    if (vk_ngxfg_wanted() && N.feat_inst_req) {
        static const wchar_t here[] = L".";
        NgxDiscovery d;
        memset(&d, 0, sizeof d);
        d.sdk = NGX_VERSION_API, d.feature = NGX_FEATURE_FRAME_GENERATION;
        d.ident.type = 0, d.ident.v.app = 231313132ull;   // the application id vk_dlaa_init gives NGX
        d.path = here;
        uint32_t k = 0;
        VkExtensionProperties *pr = NULL;
        if (NGX_OK(((PFN_FeatInstReq)N.feat_inst_req)(&d, &k, &pr)) && pr)
            for (uint32_t i = 0; i < k; i++) {
                const char *nm = fg_keep(pr[i].extensionName);
                if (nm && !listed(ext, *n, nm) && *n < max) ext[(*n)++] = nm, rt_log("ngxfg: instance extension %s", nm);
            }
    }
}

void vk_dlaa_device_exts(const VkExtensionProperties *have, uint32_t nhave, const char **ext, uint32_t *n,
                         uint32_t max) {
    unsigned ni = 0, nd = 0;
    const char **ie = NULL, **de = NULL;
    if (!vk_dlaa_wanted() || fsr3() || !core_load() || !NGX_OK(N.req_ext(&ni, &ie, &nd, &de))) return;
    for (unsigned i = 0; i < nd; i++) {
        int ok = 0;
        for (uint32_t k = 0; k < nhave && !ok; k++) ok = !strcmp(have[k].extensionName, de[i]);
        if (!ok) rt_log("dlaa: the device lacks %s, which NGX asks for", de[i]);
        else if (!listed(ext, *n, de[i]) && *n < max) ext[(*n)++] = de[i];
    }
}

static int vtable_ok(void) {
    NgxParam *p = NULL;
    if (!NGX_OK(N.alloc_params(&p)) || !p) return 0;
    int i = 0;
    unsigned u = 0;
    float f = 0;
    void *v = NULL;
    p_seti(p, "NX2navi.Test.I", -12345);
    p_setu(p, "NX2navi.Test.U", 4000000000u);
    p_setf(p, "NX2navi.Test.F", 1.25f);
    p_setp(p, "NX2navi.Test.P", (void *)p);
    int ok = p_geti(p, "NX2navi.Test.I", &i) && i == -12345 && p_getu(p, "NX2navi.Test.U", &u) && u == 4000000000u &&
             p_getf(p, "NX2navi.Test.F", &f) && f == 1.25f && p_getp(p, "NX2navi.Test.P", &v) && v == (void *)p;
    if (N.destroy_params) N.destroy_params(p);
    if (!ok) rt_log("dlaa: NGX parameter layout check failed (i=%d u=%u f=%g p=%p): DLAA off", i, u, f, v);
    return ok;
}

#ifndef _WIN32
// UTF-8 to UTF-32 wchar_t, without the C library's locale (the runtime never sets one, so mbstowcs would stop at the
// first non-ASCII byte). Returns 0 on malformed input or when out has no room.
static int utf8_to_wide(const char *in, wchar_t *out, size_t cap) {
    const unsigned char *u = (const unsigned char *)in;
    size_t n = 0;
    while (*u) {
        unsigned c = *u++, more = c < 0x80 ? 0 : c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 4;
        if (more == 4) return 0;
        if (more) c &= 0x3F >> more;
        for (unsigned k = 0; k < more; k++, u++) {
            if ((*u & 0xC0) != 0x80) return 0;
            c = c << 6 | (*u & 0x3F);
        }
        if (n + 1 >= cap) return 0;
        out[n++] = (wchar_t)c;
    }
    out[n] = 0;
    return 1;
}
#endif

static int before_bloom;

void vk_dlaa_init(void) {
    before_bloom = cfg_bool("NX2NAVI_DLSS_BEFORE_BLOOM");
    if (vk_dlaa_wanted() && fsr3()) {
        fsr_frame = 1;
        N.available = vk_ffx_load();
        rt_log("dlaa: the temporal upscaler is AMD FSR 3.1%s", N.available ? "" : ", which is not available: the game's own scaling");
        return;
    }
    if (!vk_dlaa_wanted() || !core_load()) return;
#ifdef _WIN32
    wchar_t wdir[MAX_PATH];
    DWORD len = GetModuleFileNameW(NULL, wdir, MAX_PATH);
    while (len && wdir[len - 1] != L'\\') len--;
    wdir[len] = 0;
#else
    // The executable's folder, where NGX looks for libnvidia-ngx-dlss.so and writes its logs, as a wchar_t string:
    // 32-bit code points here, as the Linux core expects.
    char dir[PLAT_PATH_MAX];
    wchar_t wdir[PLAT_PATH_MAX];
    char *cut = plat_exe_path(dir, sizeof dir) ? plat_last_sep(dir) : NULL;
    if (!cut || !utf8_to_wide((cut[1] = 0, dir), wdir, PLAT_PATH_MAX)) {
        rt_log("dlaa: cannot name the executable's folder for NGX: DLAA off");
        return;
    }
#endif
    // 231313132: the application id NVIDIA's DLSS samples use for builds without an NVIDIA-issued id
    const VkCtx *c = vk_ctx();
    NgxResult r = N.init(231313132ull, wdir, c->instance, c->phys, c->device, vkGetInstanceProcAddr,
                         vkGetDeviceProcAddr, NGX_VERSION_API, NULL);
    if (!NGX_OK(r)) {
        rt_log("dlaa: NGX init failed (%08x): DLAA off", (unsigned)r);
        return;
    }
#ifndef _WIN32
    N.device = c->device;   // NGX holds the device from here, also if the layout check below turns DLAA off
#endif
    if (!vtable_ok()) return;
    NgxParam *caps = NULL;
    int avail = 0, driver = 0;
    if (NGX_OK(N.caps(&caps)) && caps) {
#ifndef _WIN32
        N.caps_map = caps;   // allocated for us (the SDK header: free it with DestroyParameters): vk_dlaa_release does
#endif
        p_geti(caps, "SuperSampling.Available", &avail);
        p_geti(caps, "SuperSampling.NeedsUpdatedDriver", &driver);
    }
    N.available = avail != 0;
    N.capsp = caps;
    N.ngx_up = 1;
    rt_log("dlaa: NGX up; DLSS %s%s", avail ? "available" : "NOT available (is " DLSS_MODEL " next to the executable?)",
           driver ? "; the driver needs an update" : "");
    if (vk_ngxfg_wanted() && caps) {
        int fga = 0;
        unsigned mx = 0;
        p_geti(caps, "FrameGeneration.Available", &fga);
        if (!p_getu(caps, "DLSSG.MultiFrameCountMax", &mx)) mx = 1;
        // MultiFrameCountMax said 3 (4x) with the 310.9.16 library on an RTX 5090, yet that library evaluated 5
        // generated frames without an error on NVIDIA's 615 driver, and NVIDIA ships up to 6x on RTX 50 cards: a GPU
        // with multi-frame generation gets up to 5 (6x).
        FG.max = fga ? (mx >= 3 ? 5 : mx < 1 ? 1 : (int)mx) : 0;
        rt_log("ngxfg: DLSS Frame Generation %s (up to %dx)",
               fga ? "available" : "NOT available (is libnvidia-ngx-dlssg.so.<version> next to the executable?)", FG.max + 1);
    }
}

int vk_dlaa_available(void) { return vk_dlaa_wanted() && N.available; }

// ---------------------------------------------------------------- where DLAA runs in NX2navi's frame
// Measured (NX2NAVI_DUMP, frame 2100 at Lookout Landing, 2560x1440 scene): the G-buffer passes draw into five targets with
// the scene depth (d32f at the scene size), one of them the R11G11B10F light target; lighting and the transparents
// then draw into that light target alone, the last time with the scene depth attached; the first pass narrower than a
// quarter of the scene starts the bloom chain, which reads it. After the bloom come the tonemap and post passes, then
// the HUD is drawn into the same target and a last pass copies it to the 8-bit output.
// So: after the G-buffer, a pass at the scene size with exactly one colour target (R11G11B10F or RGBA16F) and the D32F
// depth arms the frame with that pair; the next pass at most a quarter as wide ends the scene (no jitter after it) and,
// when the scene is as wide as the output, runs DLAA on the armed target first. Bloom, tonemap, post and HUD then all
// work on the anti-aliased picture.
// Upscaling (a scene smaller than the output: NX2NAVI_DLSS quality/balanced/performance, or the game's own 5/6): measured
// at a 960x536 scene, 1920x1080 output, the bloom and its composite write back into the light target at the scene size,
// then one full-screen draw (the FXAA shader below, which also scales) reads it into an output-sized R11G11B10F target
// with no depth, and the HUD is drawn over that target at the output size. DLSS runs at the begin of that pass: it
// reads the light target and its result is blitted into the pass's target, whose scaling draw is then skipped. The
// HUD, drawn afterwards into the same target, is never jittered nor passed through DLSS.
// The scaling draw is not always that FXAA shader: at a 1280x720 scene and 1920x1080 output (1.4.3, dlss = quality)
// the game's tonemap (guest fs 7e114c4330dc212a) reads the light target and writes the output-sized target itself,
// over DLSS's picture, so the jittered scene reached the screen and the whole picture shook. So from DLSS's evaluation
// to the frame's end, draws in passes larger than the light target sample DLSS's output in its place
// (vk_dlaa_swap_img, applied by vk_exec_draw.c tex_info): whichever shader scales, it scales DLSS's picture.
#define OCAM_BLOCK 0x3500u   // Ocarina's scene context block (vk_dlaa_draw)
#define OCAM_BYTES 0xb0u     // its rows 0-10: the view and the projection
#define DLSS_SIZES 5         // DLSS features kept, one per scene and output size (D.sizes)
// A size no evaluation used for this many goes with its feature (stale_sizes): DLSS's history buffers and the work
// images at the output size held their VRAM after a switch of the DLSS mode, until four more sizes had come and gone.
#define SIZE_STALE 3600
static struct {
    VkExecTarget color, depth;   // the armed pair, valid in the frame it was armed
    int armed, done, gbuf;       // this frame: armed on the light target, DLAA ran, the G-buffer was drawn
    int ended, up;               // this frame: the scene is drawn (jitter over), DLSS wrote the scaled output
    int early, previous_early;   // the colour domain of this frame and of the current temporal history
    int promoted;                // this frame has an output-sized tonemap result
    int debug_image;             // evaluate wrote a diagnostic in place of reconstructed colour
    uint64_t early_outputs, full_size_outputs;
    VkImage composite;           // guest tonemap destination for this frame
    struct Size {                // a DLSS feature and its work images per scene size (dynamic resolution flips it)
        uint32_t w, h, ow, oh;   // render size, output size
        VkImage out, mv;
        VmaAllocation out_mem, mv_mem;
        VkImageView out_view, mv_view;
        VkImage mask, chars[2];  // the bias mask DLSS reads, the character class mask of this and last frame (R8)
        VmaAllocation mask_mem, chars_mem[2];
        VkImageView mask_view, chars_view[2];
        int chars_cur, chars_ok; // chars[chars_cur] is written next; chars_ok: the other holds last frame's
        VkImage dist[2];         // each pixel's view distance, this and last frame (R32F, dlaa_mv.comp depth_out)
        VmaAllocation dist_mem[2];
        VkImageView dist_view[2];
        int dist_cur, dist_ok;   // as chars_cur and chars_ok
        NgxSrFeature sr;
        uint64_t used;           // evals count when it last ran
        uint64_t generation;
        uint64_t last_scene;
        uint32_t scene_users, present_users;
        unsigned retired, initialized;
    } sizes[DLSS_SIZES], *last;  // last: the one the previous DLAA frame ran with
    struct Size *swap_size;
    struct Size *fg_size;
    VkTexturePresentSource *fg_depth_owner;
    int size_hook_open;
    VkImage last_color;          // the target DLAA ran on last frame: another one means a new history
    NgxParam *params;
    int failed, reset;
    uint64_t evals, frames_missed;
    int share_off;   // frames left that pick the output pass without the mode's share (vk_dlaa_pass, vk_dlaa_frame)
    int shape_logged;   // passes of another shape than the scene's turned down as the output pass (vk_dlaa_pass)
    // jitter (vk_dlaa_viewport)
    VkImage jdepth;              // the scene depth of the last frame DLAA ran on: passes drawing with it are jittered
    uint32_t jw, jh;             // its size: on Ocarina's frame the camera's draws into any D32F depth of it are (jitter_draw)
    float jx, jy;                // the offset the next scene draws take, render pixels (window x right, y down)
    float ux, uy;                // the offset this frame's scene draws took (DLSS and the motion vectors get these)
    uint32_t jframe;
    int jit_ok, jit_used, hold;  // jitter allowed this frame / a draw used it / frames left without it
    uint64_t holds, jittered;   // jitter holds; evaluations of a jittered scene
    uint64_t jdraws, jfixed;    // draws into a scene-size depth that took the jitter / that did not (full-screen)
    // camera (vk_dlaa_draw) and motion vectors (mv_pass)
    VkMemoryImageHold *camera_depth_owner;
    VkImage camera_depth;
    VkFormat camera_depth_format;
    uint32_t camera_depth_level, camera_depth_layer, camera_depth_layers;
    uint64_t cam_addr;           // this frame's scene camera block (vk_dlaa_draw), 0: none seen
    uint32_t cam_size, cam_vp;   // its size and the offset of its view-projection rows
    struct {                     // Ocarina: this frame's scene contexts of light-target draws
        uint64_t addr, seq;      // seq: vk_exec_submit_seq at the last draw that used it
        uint32_t n;              // draws
        int got;                 // cam holds its rows 0-10 as the first draw read them (vk_usnap_read: 1, 2), 0 not yet
        float cam[OCAM_BYTES / 4];
    } views[8];
    int nviews;
    VkBuffer camb;               // two CAM_SLOT slots: this frame's block and last frame's
    VmaAllocation camb_mem;
    VkBuffer mvm;                // MVM_SLOTS pairs of blocks for dlaa_mv.comp: the identity and the reprojection (mv_pass)
    VmaAllocation mvm_mem;
    uint8_t *mvm_map;
    uint32_t mvm_slot;
    double cam_prev[16];         // last frame's view-projection (rows 3-6 of its block), for the reprojection
    int cam_slot, cam_prev_ok;   // the slot this frame writes / the other slot holds last frame's camera
    VkDescriptorSetLayout mv_dsl;
    VkPipelineLayout mv_layout;
    VkPipeline mv_pipe;
    VkSampler mv_samp;
    int mv_ok;                   // -1 not built, 0 failed, 1 ready
    uint64_t mv_frames, mv_none, masked;   // masked: frames with the characters' bias mask
    uint64_t why[3];             // history resets: scene size changed, no camera, scene target changed
    uint64_t target_kept;        // scene target changed and the history stayed (target_changed)
    // frames without a camera and what mv_pass did about them (dlaa_cam_keep, cam_note)
    uint64_t cam_none_draw, cam_none_snap;   // no scene-context draw this frame / one, not in the evaluating submission's copy
    uint64_t cam_at_draw;        // camera taken from the bytes the draw read, where the evaluating submission lacks them
    uint64_t cam_kept, cam_kept_max;   // no camera, history kept with zero motion / runs cut short at KEEP_MAX
    float cut_pos[3], cut_fwd[3];      // last frame's camera position and direction (cam_cut)
    float cut_view[12];
    uint64_t cut_frame;
    uint64_t cut_ns, cuts;             // when it was read; camera cuts that started a new history
    int cut_ok, cut_logged;
    int keep_run;                // frames in a row kept without a camera
    uint64_t gap_logged, miss_logged, target_logged, second_logged;
    uint64_t seq_present;        // vk_exec_submit_seq at the last present (vk_dlaa_frame)
    uint64_t seconds;            // present intervals with a second scene after DLSS ran
    int on;                      // the dlaa setting, read at each present (the menu switches it live)
    int preset, preset_dirty;    // the DLSS model features are created with (preset_value); a change pending
    VkExecTarget ids;            // this frame's G-buffer target 0 (object classes in green), valid with ids_ok
    int ids_ok;
    unsigned cfg_seen;           // cfg_version when preset was read
    uint64_t last_eval_ns;       // plat_now_ns of the previous evaluation (FrameTimeDeltaInMsec)
    VkFfxImage fg_depth, fg_mv;  // the last evaluation's depth and motion vectors, for frame generation
    VkImageView fg_depth_view, fg_mv_view;
    VkDlaaCamera fg_cam;         // this frame's camera (mv_pass), fg_cam_ok when known
    int fg_cam_ok;
    float fg_jx, fg_jy;
    int fg_fresh, fg_reset;      // fresh: not handed out yet (vk_dlaa_fg_inputs)
    uint64_t presents;           // vk_dlaa_frame calls so far
    uint64_t fg_at;             // presents when the depth and motion were recorded
    VkNgxImage present_alpha[2];
    VmaAllocation present_alpha_mem[2];
} D = {.mv_ok = -1, .cfg_seen = ~0u};
typedef struct {
    VkTexImage image;
    VkMemoryImageHold *native;
    VkTextureAliasRead *seed;
    const tex_t *pin;
    uint32_t refs;
    int canceled;
    RcBudgetTicket metadata;
} DlaaTargetOwner;

enum { DLAA_TARGET_COLOR, DLAA_TARGET_DEPTH, DLAA_TARGET_IDS, DLAA_TARGET_COUNT };
static DlaaTargetOwner *frame_targets[DLAA_TARGET_COUNT];

static int dlaa_target_valid(const DlaaTargetOwner *owner) {
    return !owner || !__atomic_load_n(&owner->canceled, __ATOMIC_ACQUIRE);
}

static DlaaTargetOwner *dlaa_target_capture(const VkExecTarget *selected, VkTextureAliasRead *seed) {
    if (!selected->exact_snapshot || !selected->tex || !selected->img || !selected->image ||
        selected->img->image != selected->image || selected->img->gone || selected->bc_backing ||
        (seed && !vk_texture_alias_target_current(seed, selected->image))) return NULL;
    const tex_t *pin = nvn_texture_pin_acquire(selected->tex);
    if (!pin) return NULL;
    void *storage = NULL;
    RcBudgetTicket metadata = {0};
    if (vk_replay_metadata_allocate(sizeof(DlaaTargetOwner), 1, RC_BUDGET_TEXTURE, &metadata, &storage) != VK_SUCCESS) {
        nvn_texture_pin_release(pin);
        return NULL;
    }
    // Caller owns the exact admitted native image while this reference is acquired. The frame never
    // dereferences the pass's copied metadata or the mutable registration entry after adoption.
    DlaaTargetOwner *owner = storage;
    *owner = (DlaaTargetOwner){.image = *selected->img, .pin = pin, .refs = 1, .metadata = metadata};
    if (seed) owner->seed = vk_texture_alias_read_retain(seed);
    else owner->native = vk_mem_image_hold(selected->image);
    // A recorded use can survive later retirement of the original image cache entry.
    owner->image.gone = 1;
    return owner;
}

static void dlaa_target_release(void *opaque) {
    DlaaTargetOwner *owner = opaque;
    if (!owner || __atomic_sub_fetch(&owner->refs, 1, __ATOMIC_ACQ_REL)) return;
    if (owner->native) vk_mem_image_release(owner->native);
    vk_texture_alias_read_cancel(owner->seed);
    nvn_texture_pin_release(owner->pin);
    RcBudgetTicket metadata = owner->metadata;
    vk_replay_metadata_free(owner, &metadata);
}

static void dlaa_target_cancel(void *opaque, int may_have_executed) {
    (void)may_have_executed;
    DlaaTargetOwner *owner = opaque;
    __atomic_store_n(&owner->canceled, 1, __ATOMIC_RELEASE);
    dlaa_target_release(owner);
}

static void dlaa_target_use_cancel(void *opaque, int may_have_executed) {
    (void)may_have_executed;
    dlaa_target_release(opaque);
}

static void dlaa_target_record(VkExec *ex, DlaaTargetOwner *owner, int publishes) {
    if (!owner) return;
    if (!__atomic_add_fetch(&owner->refs, 1, __ATOMIC_RELAXED))
        rt_fatal("dlaa: prepared target owner reference count exhausted");
    vk_completion_add_owned(ex->fence, dlaa_target_release,
        publishes ? dlaa_target_cancel : dlaa_target_use_cancel, owner);
}

static VkExecTarget dlaa_owned_target(const VkExecTarget *selected, DlaaTargetOwner *owner) {
    VkExecTarget target = *selected;
    if (owner) {
        target.img = &owner->image;
        target.tex = owner->pin;
    }
    return target;
}

static void dlaa_frame_target_set(unsigned slot, DlaaTargetOwner *owner) {
    dlaa_target_release(frame_targets[slot]);
    frame_targets[slot] = owner;
}

static int dlaa_camera_depth_matches(const VkExecTarget *depth) {
    return D.camera_depth_owner && D.camera_depth == depth->image && D.camera_depth_format == depth->format &&
        D.camera_depth_level == depth->level && D.camera_depth_layer == depth->base_layer &&
        D.camera_depth_layers == depth->layers;
}

static void dlaa_camera_depth_close(void) {
    if (D.camera_depth_owner) vk_mem_image_release(D.camera_depth_owner);
    D.camera_depth_owner = NULL;
    D.cam_addr = 0;
    D.nviews = 0;
}

static void dlaa_camera_depth_select(const VkExecTarget *depth) {
    if (dlaa_camera_depth_matches(depth)) return;
    dlaa_camera_depth_close();
    // The accepted draw owns this physical image. Holding it prevents handle
    // reuse from matching a camera observed on a different allocation.
    D.camera_depth_owner = vk_mem_image_hold(depth->image);
    D.camera_depth = depth->image;
    D.camera_depth_format = depth->format;
    D.camera_depth_level = depth->level;
    D.camera_depth_layer = depth->base_layer;
    D.camera_depth_layers = depth->layers;
}

static void dlaa_frame_targets_close(void) {
    dlaa_camera_depth_close();
    if (frame_targets[DLAA_TARGET_COLOR]) D.color.img = NULL, D.color.tex = NULL;
    if (frame_targets[DLAA_TARGET_DEPTH]) D.depth.img = NULL, D.depth.tex = NULL;
    if (frame_targets[DLAA_TARGET_IDS]) D.ids.img = NULL, D.ids.tex = NULL;
    for (unsigned i = 0; i < DLAA_TARGET_COUNT; ++i) dlaa_frame_target_set(i, NULL);
}

static int dlaa_targets_armed(void) {
    return D.armed && dlaa_target_valid(frame_targets[DLAA_TARGET_COLOR]) &&
           dlaa_target_valid(frame_targets[DLAA_TARGET_DEPTH]);
}

static VkTexturePresentSource *dlaa_present_target(const DlaaTargetOwner *owner, VkImage exact) {
    if (!owner || !dlaa_target_valid(owner) || owner->image.image != exact) return NULL;
    return owner->seed ? vk_texture_present_retain_alias_image(owner->seed, exact) :
        vk_texture_present_retain_held_image(owner->native, exact);
}

static int dlaa_selected_image_matches(const VkTexImage *current, const VkTexImage *selected,
                                         const DlaaTargetOwner *owner) {
    if (!owner) return current == selected;
    return dlaa_target_valid(owner) && current && !current->gone && selected == &owner->image &&
        current->image == selected->image && current->addr == selected->addr && current->format == selected->format &&
        current->width == selected->width && current->height == selected->height && current->aspect == selected->aspect &&
        current->samples == selected->samples && current->levels == selected->levels && current->layers == selected->layers;
}

static uint64_t size_generation;

struct VkDlaaHudSource {
    VkDlaaHudImages images;
    VkNgxImage coverage;
    VkImage srgb;
    VmaAllocation color_mem, coverage_mem, srgb_mem;
    VkTexturePresentSource *target_owner;
    VkImage target;
    VkFormat target_format;
    uint64_t turn, last_scene;
    uint32_t scenes, presents, invalid, alpha_seen, alpha_bad;
    int initialized;
};

// Two composition fence slots, one active job, one queued job and one recorder.
#define HUD_SOURCE_GROUPS 5
static struct {
    VkDlaaHudSource groups[HUD_SOURCE_GROUPS];
    VkDlaaHudSource *open, *sealed;
    uint64_t generation;
    int forward_pending;
} hud_sources;

static int hud_group_idle(const VkDlaaHudSource *use) {
    return !__atomic_load_n(&use->scenes, __ATOMIC_ACQUIRE) &&
           !__atomic_load_n(&use->presents, __ATOMIC_ACQUIRE);
}

static void hud_group_complete(void *object) {
    VkDlaaHudSource *use = object;
    if (!__atomic_fetch_sub(&use->scenes, 1, __ATOMIC_ACQ_REL))
        rt_fatal("dlaa: HUD source has no recording owner");
    resource_core_budget_progress();
}

static void hud_group_cancel(void *object, int may_have_executed) {
    // Completion owns the proof that an uncertain batch has stopped using the images.
    (void)may_have_executed;
    VkDlaaHudSource *use = object;
    __atomic_store_n(&use->invalid, 1, __ATOMIC_RELEASE);
    hud_group_complete(use);
}

static void hud_group_record(const VkExec *ex, VkDlaaHudSource *use) {
    uint64_t prior = __atomic_load_n(&use->last_scene, __ATOMIC_RELAXED);
    for (;;) {
        if (prior == ex->fence) return;
        if (!ex->fence || prior > ex->fence) rt_fatal("dlaa: HUD source recording is out of order");
        if (__atomic_compare_exchange_n(&use->last_scene, &prior, ex->fence, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            break;
    }
    __atomic_fetch_add(&use->scenes, 1, __ATOMIC_RELAXED);
    vk_completion_add_owned(ex->fence, hud_group_complete, hud_group_cancel, use);
}

void vk_dlaa_hud_frame(uint64_t frame) {
    VkDlaaHudSource *use = hud_sources.open;
    hud_sources.open = NULL;
    hud_sources.sealed = NULL;
    __atomic_store_n(&hud_sources.forward_pending, 0, __ATOMIC_RELEASE);
    if (!use || use->turn != D.presents || __atomic_load_n(&use->invalid, __ATOMIC_ACQUIRE)) return;
    use->images.frame = frame;
    use->images.alpha = (VkNgxImage){0};
    if (D.fg_at == use->turn && __atomic_load_n(&use->alpha_seen, __ATOMIC_ACQUIRE) &&
        !__atomic_load_n(&use->alpha_bad, __ATOMIC_ACQUIRE))
        use->images.alpha = use->coverage;
    hud_sources.sealed = use;
}

VkDlaaHudSource *vk_dlaa_hud_take(uint64_t frame, const VkTexImage *selected) {
    VkDlaaHudSource *use = hud_sources.sealed;
    if (!use || use->images.frame != frame) return NULL;
    hud_sources.sealed = NULL;
    if (!selected || selected->width != use->images.color.w || selected->height != use->images.color.h ||
        __atomic_load_n(&use->invalid, __ATOMIC_ACQUIRE)) return NULL;
    // A distinct SDR output needs a committed final encode witness. Preserve
    // the existing HUD colors on those frames as an explicit compatibility use.
    use->images.exact = use->target_owner && use->target == selected->image &&
                        use->target_format == selected->format;
    if (__atomic_fetch_add(&use->presents, 1, __ATOMIC_ACQ_REL))
        rt_fatal("dlaa: the same HUD source was captured twice");
    return use;
}

const VkDlaaHudImages *vk_dlaa_hud_source_images(const VkDlaaHudSource *use, uint64_t frame) {
    if (!use || use->images.frame != frame || !__atomic_load_n(&use->presents, __ATOMIC_ACQUIRE) ||
        __atomic_load_n(&use->invalid, __ATOMIC_ACQUIRE) || vk_device_is_lost() ||
        __atomic_load_n(&use->last_scene, __ATOMIC_ACQUIRE) > vk_sync_handed_fence()) return NULL;
    return &use->images;
}

void vk_dlaa_hud_source_release(VkDlaaHudSource *use) {
    if (!use) return;
    if (__atomic_fetch_sub(&use->presents, 1, __ATOMIC_ACQ_REL) != 1)
        rt_fatal("dlaa: HUD source has no composition owner");
    resource_core_budget_progress();
}

struct VkDlaaFgSizeUse {
    struct Size *size;
    VkTexturePresentSource *depth_owner;
    uint64_t generation;
    VkFfxImage depth, mv;
    VkImageView depth_view, mv_view;
    VkDlaaCamera camera;
    float jx, jy;
    int reset, camera_ok, consumed;
};
static _Thread_local VkDlaaFgSizeUse *fg_compose;

static void size_drop_present(struct Size *s) {
    if (!__atomic_fetch_sub(&s->present_users, 1, __ATOMIC_ACQ_REL))
        rt_fatal("dlaa: unowned present generation");
    resource_core_budget_progress();
}

static void fg_drop_depth(VkImageView view, VkTexturePresentSource *owner) {
    if (view) vkDestroyImageView(vk_ctx()->device, view, NULL);
    vk_texture_present_release(owner);
}

static void fg_forget_size(void) {
    if (D.fg_size) {
        fg_drop_depth(D.fg_depth_view, D.fg_depth_owner);
        size_drop_present(D.fg_size);
    }
    D.fg_depth_owner = NULL;
    D.fg_depth_view = VK_NULL_HANDLE;
    D.fg_size = NULL;
    D.fg_fresh = 0;
}

void vk_dlaa_fg_discard_current(void) { fg_forget_size(); }

static void fg_publish_size(struct Size *s, const VkExecTarget *depth) {
    fg_forget_size();
    if (!vk_dlssg_active() && !vk_fg_active() && !vk_ngxfg_on()) return;
    static unsigned depth_refusals;
    const DlaaTargetOwner *selected = frame_targets[DLAA_TARGET_DEPTH];
    VkTexturePresentSource *owner = selected && depth->img == &selected->image ?
        dlaa_present_target(selected, depth->image) :
        vk_texture_present_retain_image(depth->img, depth->image);
    if (!owner) {
        if (depth_refusals++ < 8)
            rt_log("dlaa: frame generation input unavailable for frame %llu, exact depth image %p could not be retained",
                   (unsigned long long)(D.presents + 1), (void *)depth->image);
        return;
    }
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = depth->image, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = depth->format,
        .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, depth->level, 1, depth->base_layer, 1}};
    VkImageView view = VK_NULL_HANDLE;
    VkResult r = vkCreateImageView(vk_ctx()->device, &vi, NULL, &view);
    if (r != VK_SUCCESS) {
        vk_texture_present_release(owner);
        if (depth_refusals++ < 8)
            rt_log("dlaa: frame generation input unavailable for frame %llu, depth view creation failed (%d)",
                   (unsigned long long)(D.presents + 1), (int)r);
        vk_device_lost_check(r, "frame generation depth view");
        return;
    }
    __atomic_fetch_add(&s->present_users, 1, __ATOMIC_RELAXED);
    D.fg_size = s;
    D.fg_depth_owner = owner;
    D.fg_depth_view = view;
    D.fg_fresh = 1;
}

VkDlaaFgSizeUse *vk_dlaa_fg_capture(void) {
    struct Size *s = D.fg_size;
    if (!s) return NULL;
    if (!D.fg_fresh || D.failed || D.presents != D.fg_at + 1 ||
        __atomic_load_n(&s->retired, __ATOMIC_ACQUIRE)) {
        fg_forget_size();
        return NULL;
    }
    VkDlaaFgSizeUse *use = malloc(sizeof *use);
    if (!use) rt_fatal("dlaa: cannot retain frame generation inputs");
    *use = (VkDlaaFgSizeUse){.size = s, .depth_owner = D.fg_depth_owner, .generation = s->generation,
        .depth = D.fg_depth, .mv = D.fg_mv,
        .depth_view = D.fg_depth_view, .mv_view = D.fg_mv_view, .camera = D.fg_cam,
        .jx = D.fg_jx, .jy = D.fg_jy, .reset = D.fg_reset, .camera_ok = D.fg_cam_ok};
    D.fg_depth_owner = NULL;
    D.fg_depth_view = VK_NULL_HANDLE;
    D.fg_size = NULL;
    D.fg_fresh = 0;
    return use;
}

void vk_dlaa_fg_compose_select(VkDlaaFgSizeUse *use) {
    if (use && fg_compose) rt_fatal("dlaa: nested frame generation composition owner");
    fg_compose = use;
}

int vk_dlaa_fg_compose_consumed(const VkDlaaFgSizeUse *use) { return use && use->consumed; }

void vk_dlaa_fg_release(VkDlaaFgSizeUse *use) {
    if (!use) return;
    if (use->size->generation != use->generation || use == fg_compose)
        rt_fatal("dlaa: invalid frame generation release");
    fg_drop_depth(use->depth_view, use->depth_owner);
    size_drop_present(use->size);
    free(use);
}

// What this frame drew into a D32F depth at least DLAA_SCENE_MIN_W wide before DLSS ran (vk_dlaa_draw), for the line a
// frame without a camera logs (cam_note); cleared at each present (vk_dlaa_frame).
static struct {
    uint32_t zdraws, zother;      // draws into the scene depth / into another such depth than last frame's (jdepth)
    uint32_t zother_w, zother_h;
    uint32_t ocam, ocam_off;      // of the scene-depth draws: with the scene context at c3 / of those, not into a scene
    VkFormat ocam_off_fmt;        //   colour target (the format of the last such)
    uint32_t c3_other, c3_size;   // with another block at c3 (the size of the last such; 0: none bound)
    uint64_t seq0;                // vk_exec_submit_seq at the frame's first scene-depth draw
    uint32_t one_n, one_w, one_h; // after arming: passes with one colour target and no depth (vk_dlaa_pass), the last's
    VkFormat one_fmt;             //   size and format, for a frame without the DLAA point (vk_dlaa_frame)
    uint32_t second;              // after DLSS ran: passes shaped like the light target's (another scene)
    VkImage second_img;
} F;

int vk_dlaa_fg_views(VkImageView *depth, VkImageView *mv) {
    if (!fg_compose || !fg_compose->consumed) return 0;
    *depth = fg_compose->depth_view;
    *mv = fg_compose->mv_view;
    return *depth && *mv;
}

int vk_dlaa_fg_camera(VkDlaaCamera *c) {
    if (!fg_compose || !fg_compose->consumed || !fg_compose->camera_ok) return 0;
    *c = fg_compose->camera;
    return 1;
}

// Metres in one unit of Ocarina's view space, for FSR's distance heuristics: one. Measured in Hyrule Field, the camera
// orbits Link about 4 units behind him (the nearest scene depth, 0.975, is 1 - 0.1 / 4) at about 3 units a second, and
// travels about 4.7 units a second while he runs; in centimetres (the 0.01 FSR had) he would be 4 cm from the camera.
#define VIEW_METERS 1.0f

static VkFfxCamera ffx_camera(const VkDlaaCamera *camera) {
    return (VkFfxCamera){camera->fov_y, camera->znear, camera->zfar, VIEW_METERS};
}

int vk_dlaa_ffx_camera(VkFfxCamera *c) {
    if (!fg_compose || !fg_compose->consumed || !fg_compose->camera_ok) return 0;
    *c = ffx_camera(&fg_compose->camera);
    return 1;
}

int vk_dlaa_fg_describe(const VkDlaaFgSizeUse *use, VkFfxImage *depth, VkFfxImage *mv) {
    if (!use) return 0;
    if (use->consumed || use->size->generation != use->generation ||
        __atomic_load_n(&use->size->retired, __ATOMIC_ACQUIRE)) return -1;
    if (!use->depth_view || !use->mv_view) return -1;
    if (!use->camera_ok) return 0;
    *depth = use->depth;
    *mv = use->mv;
    return 1;
}

int vk_dlaa_fg_compose_peek(VkFfxImage *depth, VkFfxImage *mv, VkImageView *depth_view, VkImageView *mv_view,
                            VkDlaaCamera *camera, float *jx, float *jy, int *reset) {
    const VkDlaaFgSizeUse *use = fg_compose;
    int describe = vk_dlaa_fg_describe(use, depth, mv);
    if (describe != 1) return describe;
    *depth_view = use->depth_view;
    *mv_view = use->mv_view;
    *camera = use->camera;
    *jx = use->jx;
    *jy = use->jy;
    *reset = use->reset;
    return 1;
}

int vk_dlaa_fg_inputs(VkFfxImage *depth, VkFfxImage *mv, float *jx, float *jy, int *reset) {
    VkDlaaFgSizeUse *use = fg_compose;
    if (!use || use->consumed) return 0;
    if (use->size->generation != use->generation ||
        __atomic_load_n(&use->size->retired, __ATOMIC_ACQUIRE)) return 0;
    use->consumed = 1;
    *depth = use->depth, *mv = use->mv, *jx = use->jx, *jy = use->jy, *reset = use->reset;
    return 1;
}

const VkTexImage *vk_dlaa_swap_img;
VkImageView vk_dlaa_swap_view;
VkImage vk_dlaa_swap_output;

static int dl_active(void) { return N.available && !D.failed && D.on; }
// The upscaler this frame runs with, as vk_dlaa_frame read it. The menu can set NX2NAVI_TAA to dlss between that and
// evaluate; read again there while NGX is not up (DLSS starts only after a restart), it sent evaluate to DLSS's path
// with NGX's functions not loaded, a call to address 0 (issue #325).
static int use_fsr(void) { return !N.ngx_up && fsr_frame; }

int vk_dlaa_temporal_hook_needs_recorder(void) {
    return __atomic_load_n(&D.size_hook_open, __ATOMIC_ACQUIRE) ||
        __atomic_load_n(&hud_sources.forward_pending, __ATOMIC_ACQUIRE);
}

static void size_hook_close(void) {
    __atomic_store_n(&D.size_hook_open, 0, __ATOMIC_RELEASE);
}

static int dl_fail(const char *what, NgxResult r) {
    size_hook_close();
    rt_log("dlaa: %s failed (%08x): DLAA off for this run", what, (unsigned)r);
    D.failed = 1;
    return 0;
}

static VkImageCreateInfo dlaa_image_2d_info(uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage) {
    return (VkImageCreateInfo){.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
        .format = format, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
}

static VkResult dlaa_image_2d_view(VkDevice device, VkImage image, VkFormat format, VkImageView *view) {
    VkImageAspectFlags aspect = format == VK_FORMAT_D32_SFLOAT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageViewCreateInfo info = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format, .subresourceRange = {aspect, 0, 1, 0, 1}};
    return vkCreateImageView(device, &info, NULL, view);
}

enum { DLAA_SIZE_NATIVE_IMAGES = 7 };
typedef struct {
    uint32_t w, h, ow, oh;
    VkImageCreateInfo images[DLAA_SIZE_NATIVE_IMAGES];
    VmaAllocationCreateInfo placements[DLAA_SIZE_NATIVE_IMAGES];
    RcAdmissionVector peak;
} DlaaSizeNativePlan;

typedef enum {
    DLAA_SIZE_NATIVE_READY, DLAA_SIZE_NATIVE_PENDING, DLAA_SIZE_NATIVE_STOPPED, DLAA_SIZE_NATIVE_TERMINAL
} DlaaSizeNativeState;

typedef struct {
    DlaaSizeNativeState state;
    VkResult vk_result;
    RcAdmissionStatus admission;
    RcAdmissionVector peak;
    RcBudgetReadiness before;
    RcVmaDependency child;
} DlaaSizeNativeOutcome;

typedef struct {
    VkImage *image;
    VmaAllocation *allocation;
    VkImageView *view;
} DlaaSizeNativeMember;

static void dlaa_size_native_members(struct Size *size, DlaaSizeNativeMember members[DLAA_SIZE_NATIVE_IMAGES]) {
    const DlaaSizeNativeMember native[DLAA_SIZE_NATIVE_IMAGES] = {
        {&size->out, &size->out_mem, &size->out_view},
        {&size->mv, &size->mv_mem, &size->mv_view},
        {&size->mask, &size->mask_mem, &size->mask_view},
        {&size->chars[0], &size->chars_mem[0], &size->chars_view[0]},
        {&size->chars[1], &size->chars_mem[1], &size->chars_view[1]},
        {&size->dist[0], &size->dist_mem[0], &size->dist_view[0]},
        {&size->dist[1], &size->dist_mem[1], &size->dist_view[1]},
    };
    memcpy(members, native, sizeof native);
}

// A stopped request cannot publish a tentative bundle. Relay native loss to the caller for its loss handling.
static int dlaa_size_native_stopped(DlaaSizeNativeOutcome *outcome) {
    if (outcome->vk_result == VK_ERROR_DEVICE_LOST || vk_device_is_lost()) {
        outcome->state = DLAA_SIZE_NATIVE_STOPPED;
        outcome->vk_result = VK_ERROR_DEVICE_LOST;
    } else if (vk_sync_stopping()) {
        outcome->state = DLAA_SIZE_NATIVE_STOPPED;
        outcome->vk_result = VK_NOT_READY;
    }
    return outcome->state == DLAA_SIZE_NATIVE_STOPPED;
}

// Unpublished groups and old groups with an external retirement proof share this native destruction path.
static void dlaa_size_native_destroy(struct Size *group) {
    const VkCtx *c = vk_ctx();
    DlaaSizeNativeMember members[DLAA_SIZE_NATIVE_IMAGES];
    dlaa_size_native_members(group, members);
    for (unsigned i = 0; i < DLAA_SIZE_NATIVE_IMAGES; ++i) {
        DlaaSizeNativeMember member = members[i];
        if (*member.view) vkDestroyImageView(c->device, *member.view, NULL);
        if (*member.image || *member.allocation)
            resource_core_vma_destroy_image(c->vma, *member.image, *member.allocation);
        *member.view = VK_NULL_HANDLE;
        *member.image = VK_NULL_HANDLE;
        *member.allocation = NULL;
    }
}

static DlaaSizeNativeOutcome dlaa_size_native_plan(uint32_t w, uint32_t h, uint32_t ow, uint32_t oh,
                                                   DlaaSizeNativePlan *out) {
    DlaaSizeNativeOutcome outcome = {.state = DLAA_SIZE_NATIVE_TERMINAL,
        .vk_result = VK_ERROR_INITIALIZATION_FAILED, .admission = RC_ADMISSION_INVALID};
    if (dlaa_size_native_stopped(&outcome)) return outcome;
    const VkCtx *c = vk_ctx();
    if (!c || !out || !w || !h || !ow || !oh) return outcome;
    const VkImageUsageFlags storage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    const VkImageUsageFlags transfer = storage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    DlaaSizeNativePlan next = {.w = w, .h = h, .ow = ow, .oh = oh, .images = {
        dlaa_image_2d_info(ow, oh, VK_FORMAT_R16G16B16A16_SFLOAT, transfer),
        dlaa_image_2d_info(w, h, VK_FORMAT_R16G16_SFLOAT, transfer | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT),
        dlaa_image_2d_info(w, h, VK_FORMAT_R8_UNORM, storage),
        dlaa_image_2d_info(w, h, VK_FORMAT_R8_UNORM, storage),
        dlaa_image_2d_info(w, h, VK_FORMAT_R8_UNORM, storage),
        dlaa_image_2d_info(w, h, VK_FORMAT_R32_SFLOAT, storage),
        dlaa_image_2d_info(w, h, VK_FORMAT_R32_SFLOAT, storage),
    }};
    // Each image has a pinned placement and its own native requirement. A default-pool growth block per
    // member would count backing that these seven long-lived dedicated allocations will not allocate.
    const VmaAllocationCreateInfo dedicated = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT};
    const RcVmaRequest request = {.producer = RC_BUDGET_TEXTURE, .required = 1};
    for (unsigned i = 0; i < DLAA_SIZE_NATIVE_IMAGES; ++i) {
        if (dlaa_size_native_stopped(&outcome)) return outcome;
        const VkImageCreateInfo *image = &next.images[i];
        VkImageFormatProperties properties;
        outcome.vk_result = vkGetPhysicalDeviceImageFormatProperties(c->phys, image->format, image->imageType,
                                                                      image->tiling, image->usage, image->flags, &properties);
        if (outcome.vk_result == VK_SUCCESS &&
            (image->extent.width > properties.maxExtent.width || image->extent.height > properties.maxExtent.height))
            outcome.vk_result = VK_ERROR_FORMAT_NOT_SUPPORTED;
        RcAdmissionVector member;
        if (outcome.vk_result == VK_SUCCESS)
            outcome.vk_result = resource_core_vma_image_peak(&request, image, &dedicated, &next.placements[i], &member);
        if (outcome.vk_result != VK_SUCCESS) {
            if (outcome.vk_result == VK_ERROR_FORMAT_NOT_SUPPORTED || outcome.vk_result == VK_ERROR_FEATURE_NOT_PRESENT ||
                outcome.vk_result == VK_ERROR_OUT_OF_DEVICE_MEMORY)
                outcome.admission = RC_ADMISSION_UNSUPPORTED;
            dlaa_size_native_stopped(&outcome);
            return outcome;
        }
        for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; ++d) {
            if (member.amount[d] > UINT64_MAX - next.peak.amount[d]) {
                outcome.vk_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
                outcome.admission = RC_ADMISSION_OVERFLOW;
                return outcome;
            }
            next.peak.amount[d] += member.amount[d];
        }
    }
    outcome.peak = next.peak;
    // Individual valid allocations can still be impossible simultaneously on the same selected heap.
    for (uint32_t heap = 0; heap < RC_BUDGET_GPU_HEAPS && heap < c->mem_props.memoryHeapCount; ++heap) {
        if (next.peak.amount[RC_BUDGET_GPU_FIRST + heap] > c->mem_props.memoryHeaps[heap].size) {
            outcome.vk_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
            outcome.admission = RC_ADMISSION_UNSUPPORTED;
            return outcome;
        }
    }
    if (dlaa_size_native_stopped(&outcome)) return outcome;
    *out = next;
    outcome.state = DLAA_SIZE_NATIVE_READY;
    outcome.admission = RC_ADMISSION_OK;
    return outcome;
}

// Materialize an unpublished exact plan. The renderer owns the subsequent barrier, validation and publication.
static DlaaSizeNativeOutcome dlaa_size_native_create(const DlaaSizeNativePlan *plan, struct Size *out) {
    DlaaSizeNativeOutcome outcome = {.state = DLAA_SIZE_NATIVE_TERMINAL,
        .vk_result = VK_ERROR_INITIALIZATION_FAILED, .admission = RC_ADMISSION_INVALID};
    if (dlaa_size_native_stopped(&outcome)) return outcome;
    const VkCtx *c = vk_ctx();
    if (!c || !plan || !out) return outcome;
    outcome.peak = plan->peak;
    resource_core_budget_readiness(1, &outcome.before);
    RcBudgetTicket ticket = {0};
    outcome.admission = resource_core_budget_reserve(RC_BUDGET_TEXTURE, 1, &plan->peak, &ticket);
    if (outcome.admission != RC_ADMISSION_OK) {
        if (outcome.admission == RC_ADMISSION_REFUSED || outcome.admission == RC_ADMISSION_NO_RECORDS) {
            outcome.state = DLAA_SIZE_NATIVE_PENDING;
            outcome.vk_result = VK_NOT_READY;
        } else if (outcome.admission == RC_ADMISSION_UNSUPPORTED) {
            outcome.vk_result = VK_ERROR_FEATURE_NOT_PRESENT;
        }
        dlaa_size_native_stopped(&outcome);
        return outcome;
    }

    struct Size next = {.w = plan->w, .h = plan->h, .ow = plan->ow, .oh = plan->oh};
    DlaaSizeNativeMember members[DLAA_SIZE_NATIVE_IMAGES];
    dlaa_size_native_members(&next, members);
    RcVmaRequest request = {.producer = RC_BUDGET_TEXTURE, .required = 1, .peak_ticket = &ticket,
                            .dependency = &outcome.child};
    outcome.vk_result = VK_SUCCESS;
    for (unsigned i = 0; i < DLAA_SIZE_NATIVE_IMAGES; ++i) {
        if (dlaa_size_native_stopped(&outcome)) break;
        DlaaSizeNativeMember member = members[i];
        outcome.vk_result = resource_core_vma_create_image(&request, c->vma, &plan->images[i], &plan->placements[i],
                                                            member.image, member.allocation, NULL, &outcome.admission);
        if (outcome.vk_result != VK_SUCCESS || dlaa_size_native_stopped(&outcome)) break;
        if (!*member.image || !*member.allocation) {
            outcome.vk_result = VK_ERROR_INITIALIZATION_FAILED;
            break;
        }
        VkImageView view = VK_NULL_HANDLE;
        outcome.vk_result = dlaa_image_2d_view(c->device, *member.image, plan->images[i].format, &view);
        if (outcome.vk_result != VK_SUCCESS) break;
        if (!view) {
            outcome.vk_result = VK_ERROR_INITIALIZATION_FAILED;
            break;
        }
        *member.view = view;
    }

    if (resource_core_budget_cancel(&ticket) != RC_ADMISSION_OK)
        rt_fatal("dlaa: work images lost their capacity reservation");
    dlaa_size_native_stopped(&outcome);
    if (outcome.state != DLAA_SIZE_NATIVE_STOPPED && outcome.vk_result == VK_SUCCESS) {
        *out = next;
        outcome.state = DLAA_SIZE_NATIVE_READY;
        outcome.admission = RC_ADMISSION_OK;
        outcome.child = (RcVmaDependency){0};
        return outcome;
    }

    // These images have never reached a recorder. Driver loss and partial materialization can still unwind them.
    dlaa_size_native_destroy(&next);
    if (outcome.state != DLAA_SIZE_NATIVE_STOPPED && outcome.vk_result == VK_NOT_READY &&
        (outcome.admission == RC_ADMISSION_REFUSED || outcome.admission == RC_ADMISSION_NO_RECORDS))
        outcome.state = DLAA_SIZE_NATIVE_PENDING;
    return outcome;
}

static int image2d(uint32_t w, uint32_t h, VkFormat f, VkImageUsageFlags usage, VkImage *img, VmaAllocation *mem,
                   VkImageView *view) {
    const VkCtx *c = vk_ctx();
    VkImageCreateInfo ii = dlaa_image_2d_info(w, h, f, usage);
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ai, img, mem, NULL) != VK_SUCCESS) return 0;
    return dlaa_image_2d_view(c->device, *img, f, view) == VK_SUCCESS;
}

// The guest draws its tonemap at scene size. Give that same draw output-sized attachments
// so its scene sample retains DLSS detail. The depth attachment matches the pipeline
// format; this fullscreen pass has depth testing and writes disabled.
enum { POST_COLOR, POST_DEPTH, POST_IMAGES };
typedef struct {
    struct {
        VkImage image;
        VmaAllocation allocation;
        VkImageView view;
    } member[POST_IMAGES];
    VkExtent2D size;
    RcBudgetTicket metadata;
    uint64_t last_scene;
    uint32_t refs;
    unsigned initialized, canceled;
} DlaaPostImages;

static struct {
    DlaaPostImages *cached, *frame;
    VkExtent2D refused;
} post;

static DlaaPostImages *post_retain(DlaaPostImages *images) {
    if (!__atomic_fetch_add(&images->refs, 1, __ATOMIC_RELAXED))
        rt_fatal("dlaa: tonemap attachments have no owner");
    return images;
}

static void post_release(void *arg) {
    DlaaPostImages *images = arg;
    if (!images) return;
    unsigned refs = __atomic_fetch_sub(&images->refs, 1, __ATOMIC_ACQ_REL);
    if (!refs) rt_fatal("dlaa: tonemap attachment ownership underflow");
    if (refs != 1) return;

    RcBudgetTicket metadata = images->metadata;
    if (resource_core_budget_retire(&metadata) != RC_ADMISSION_OK)
        rt_fatal("dlaa: tonemap attachments lost their metadata reservation");
    const VkCtx *c = vk_ctx();
    for (unsigned i = 0; i < POST_IMAGES; ++i) {
        if (images->member[i].view) vkDestroyImageView(c->device, images->member[i].view, NULL);
        if (images->member[i].image || images->member[i].allocation)
            resource_core_vma_destroy_image(c->vma, images->member[i].image, images->member[i].allocation);
    }
    free(images);
    if (resource_core_budget_destroyed(&metadata) != RC_ADMISSION_OK)
        rt_fatal("dlaa: tonemap attachment metadata destruction failed");
}

static int post_current(const DlaaPostImages *images) {
    return images && !__atomic_load_n(&images->canceled, __ATOMIC_ACQUIRE);
}

static void post_cancel(void *arg, int may_have_executed) {
    (void)may_have_executed; // Sync dispatches cancellation only after native retirement is safe.
    DlaaPostImages *images = arg;
    __atomic_store_n(&images->canceled, 1, __ATOMIC_RELEASE);
    post_release(images);
}

static void post_scene_use(VkExec *ex, DlaaPostImages *images) {
    if (!ex->fence) rt_fatal("dlaa: tonemap attachments require a recording scene");
    uint64_t prior = __atomic_load_n(&images->last_scene, __ATOMIC_RELAXED);
    for (;;) {
        if (prior == ex->fence) return;
        if (prior > ex->fence) rt_fatal("dlaa: invalid tonemap attachment scene");
        if (__atomic_compare_exchange_n(&images->last_scene, &prior, ex->fence, 0,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) break;
    }
    vk_completion_add_owned(ex->fence, post_release, post_cancel, post_retain(images));
}

static void post_frame_end(void) {
    DlaaPostImages *images = post.frame;
    post.frame = NULL;
    post_release(images);
    if (post.cached && !post_current(post.cached)) {
        images = post.cached;
        post.cached = NULL;
        post_release(images);
    }
}

static void release_post(void) {
    if (!vk_ctx() || vk_device_is_lost()) return;
    post_frame_end();
    DlaaPostImages *images = post.cached;
    post.cached = NULL;
    post.refused = (VkExtent2D){0};
    post_release(images);
}

static VkResult post_create(VkExtent2D size, DlaaPostImages **out) {
    const VkCtx *c = vk_ctx();
    if (!size.width || !size.height) return VK_ERROR_INITIALIZATION_FAILED;
    const VkImageCreateInfo info[POST_IMAGES] = {
        dlaa_image_2d_info(size.width, size.height, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT),
        dlaa_image_2d_info(size.width, size.height, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
    };
    const VmaAllocationCreateInfo dedicated = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT};
    const RcAdmissionVector host = {.amount = {[RC_BUDGET_RAM] = sizeof(DlaaPostImages),
                                             [RC_BUDGET_COMMIT] = sizeof(DlaaPostImages)}};
    RcVmaRequest request = {.producer = RC_BUDGET_TEXTURE};
    RcAdmissionVector peak = host;
    VmaAllocationCreateInfo placements[POST_IMAGES];
    for (unsigned i = 0; i < POST_IMAGES; ++i) {
        VkImageFormatProperties limits;
        VkResult result = vkGetPhysicalDeviceImageFormatProperties(c->phys, info[i].format, info[i].imageType,
            info[i].tiling, info[i].usage, info[i].flags, &limits);
        if (result != VK_SUCCESS) return result;
        if (size.width > limits.maxExtent.width || size.height > limits.maxExtent.height)
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        RcAdmissionVector cost;
        result = resource_core_vma_image_peak(&request, &info[i], &dedicated, &placements[i], &cost);
        if (result != VK_SUCCESS) return result;
        for (unsigned d = 0; d < RC_ADMISSION_MAX_DOMAINS; ++d) {
            if (cost.amount[d] > UINT64_MAX - peak.amount[d]) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
            peak.amount[d] += cost.amount[d];
        }
    }

    RcBudgetTicket joint = {0};
    if (resource_core_budget_reserve(request.producer, request.required, &peak, &joint) != RC_ADMISSION_OK)
        return VK_NOT_READY;
    request.peak_ticket = &joint;
    RcBudgetTicket metadata = {0};
    DlaaPostImages *images = NULL;
    VkResult result = VK_NOT_READY;
    if (resource_core_budget_split(&joint, &host, request.producer, &metadata) == RC_ADMISSION_OK) {
        images = calloc(1, sizeof *images);
        result = images ? VK_SUCCESS : VK_ERROR_OUT_OF_HOST_MEMORY;
        if (images) {
            images->refs = 1;
            images->size = size;
            if (!rc_admission_accounted(resource_core_budget_materialize(&metadata, &host)))
                rt_fatal("dlaa: tonemap attachment metadata settlement failed");
            images->metadata = metadata;
        } else if (resource_core_budget_cancel(&metadata) != RC_ADMISSION_OK) {
            rt_fatal("dlaa: tonemap attachment metadata cancellation failed");
        }
    }
    for (unsigned i = 0; result == VK_SUCCESS && i < POST_IMAGES; ++i) {
        if (vk_sync_stopping() || vk_device_is_lost()) {
            result = vk_device_is_lost() ? VK_ERROR_DEVICE_LOST : VK_NOT_READY;
            break;
        }
        result = resource_core_vma_create_image(&request, c->vma, &info[i], &placements[i],
            &images->member[i].image, &images->member[i].allocation, NULL, NULL);
        if (result != VK_SUCCESS) break;
        if (vk_sync_stopping() || vk_device_is_lost()) {
            result = vk_device_is_lost() ? VK_ERROR_DEVICE_LOST : VK_NOT_READY;
            break;
        }
        if (!images->member[i].image || !images->member[i].allocation) {
            result = VK_ERROR_INITIALIZATION_FAILED;
            break;
        }
        VkImageView view = VK_NULL_HANDLE;
        result = dlaa_image_2d_view(c->device, images->member[i].image, info[i].format, &view);
        if (result != VK_SUCCESS) break;
        if (!view) result = VK_ERROR_INITIALIZATION_FAILED;
        else images->member[i].view = view;
    }
    if (resource_core_budget_cancel(&joint) != RC_ADMISSION_OK)
        rt_fatal("dlaa: tonemap attachments lost their admission reservation");
    if (result == VK_SUCCESS && (vk_sync_stopping() || vk_device_is_lost()))
        result = vk_device_is_lost() ? VK_ERROR_DEVICE_LOST : VK_NOT_READY;
    if (result == VK_SUCCESS) *out = images;
    else post_release(images);
    return result;
}

static int post_images(VkExec *ex, VkExtent2D size) {
    if (vk_sync_stopping() || vk_device_is_lost()) return 0;
    DlaaPostImages *images = post.cached;
    if (!post_current(images) || images->size.width != size.width || images->size.height != size.height) {
        VkResult result = post_create(size, &images);
        if (result != VK_SUCCESS) {
            vk_device_lost_check(result, "preparing output-sized tonemap attachments");
            if (post.refused.width != size.width || post.refused.height != size.height) {
                post.refused = size;
                rt_log("dlaa: output-sized tonemap attachments unavailable at %ux%u (%d)",
                       size.width, size.height, (int)result);
            }
            return 0;
        }
        DlaaPostImages *old = post.cached;
        post.cached = images;
        post_release(old);
    }
    post.refused = (VkExtent2D){0};
    post_scene_use(ex, images);
    if (!images->initialized) {
        VkImageMemoryBarrier2 barriers[POST_IMAGES];
        for (unsigned i = 0; i < POST_IMAGES; ++i)
            barriers[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .dstStageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = images->member[i].image,
                .subresourceRange = {i == POST_DEPTH ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT,
                                     0, 1, 0, 1}};
        VkDependencyInfo dependency = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = POST_IMAGES, .pImageMemoryBarriers = barriers};
        vkCmdPipelineBarrier2(ex->cb, &dependency);
        images->initialized = 1;
        ex->recorded++;
    }
    if (post.frame != images) {
        DlaaPostImages *old = post.frame;
        post.frame = post_retain(images);
        post_release(old);
    }
    return 1;
}

// Which features may stay alive when another is made (NX2NAVI_DLSS_IDLE_FREE). A feature made while one with a larger
// input size is alive leaves that larger one drawing alternating bands of rows black or noise from then on, for as long
// as it is used; seen for the pause screen's feature (the picture's size in and out), in two ways:
// - a pause of 3,600 frames frees the gameplay feature (stale_sizes) and the unpause makes it again: every later pause
//   was striped, in each of 8 scripted runs of pause, unpause, pause in which it had been freed and in none of 12 in
//   which it had not, whatever the texture, write-back, HDR and frame generation settings;
// - a DLSS mode picked in the menu makes a feature of a third size at the unpause, with nothing freed: the next pause
//   was striped (performance to ultra performance in play).
// Giving the kept feature its own sizes again at each evaluation changed nothing. A feature made while a smaller one
// is alive leaves the smaller one as it was (the pause screen's after the gameplay one, at every first pause). So when
// a feature is made, the alive ones with a larger input are released first, and after a release since the last
// creation all of them (release_kept); their images stay and each is made again when its size is next evaluated.
// Scripted on one build, each way with the rule and with 2: the long pause and the mode change were striped without it
// and clean with it. The input size may not be what matters: in every striped case the feature made was of another
// model (performance's or ultra performance's) than the pause screen's, and the one run in which balanced's feature,
// of the pause screen's model, was made again beside it without the rule was clean. The rule covers both readings.
//   1 (default)  idle features are freed
//   0            idle features are never freed (about 0.5 GB more video memory after the first pause)
//   2            idle features are freed and nothing is released when a feature is made (as before; shows the fault)
static int feat_released;

static int idle_free(void) {
    static int m = -1;
    if (m < 0) m = cfg_int("NX2NAVI_DLSS_IDLE_FREE");
    return m;
}

static int size_idle(const struct Size *e) {
    return !__atomic_load_n(&e->scene_users, __ATOMIC_ACQUIRE) &&
           !__atomic_load_n(&e->present_users, __ATOMIC_ACQUIRE);
}

static void size_scene_complete(void *arg) {
    struct Size *e = arg;
    if (!__atomic_fetch_sub(&e->scene_users, 1, __ATOMIC_ACQ_REL))
        rt_fatal("dlaa: unowned replay generation");
    resource_core_budget_progress();
}

static void size_scene_cancel(void *arg, int may_have_executed) {
    (void)may_have_executed;  // both paths invalidate the history; vk_sync owns the native retirement proof
    struct Size *e = arg;
    __atomic_store_n(&e->retired, 1, __ATOMIC_RELEASE);
    size_scene_complete(e);
}

static void size_scene_use(VkExec *ex, struct Size *e) {
    uint64_t prior = __atomic_load_n(&e->last_scene, __ATOMIC_RELAXED);
    for (;;) {
        if (prior == ex->fence) return;
        if (prior > ex->fence) rt_fatal("dlaa: replay used an older scene generation");
        if (__atomic_compare_exchange_n(&e->last_scene, &prior, ex->fence, 0,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) break;
    }
    __atomic_fetch_add(&e->scene_users, 1, __ATOMIC_RELAXED);
    vk_completion_add_owned(ex->fence, size_scene_complete, size_scene_cancel, e);
}

void vk_dlaa_swap_use(VkExec *ex) {
    if (!D.swap_size || D.swap_size->out != vk_dlaa_swap_output ||
        __atomic_load_n(&D.swap_size->retired, __ATOMIC_ACQUIRE))
        rt_fatal("dlaa: reconstructed binding lost its native generation");
    size_scene_use(ex, D.swap_size);
}

static void size_mark_retired(struct Size *e) {
    __atomic_store_n(&e->retired, 1, __ATOMIC_RELEASE);
    if (D.last == e) D.last = NULL;
    if (D.fg_size == e) fg_forget_size();
    if (D.swap_size == e) {
        D.swap_size = NULL;
        vk_dlaa_swap_img = NULL, vk_dlaa_swap_view = VK_NULL_HANDLE, vk_dlaa_swap_output = VK_NULL_HANDLE;
    }
}

// Actual SDK recordings share the Size scene receipt. Presentation and later
// ordinary sampling may retain the images without using the SDK handle.
static void ngx_sr_scene_use(VkExec *ex, struct Size *e) {
    if (e->sr.scene > ex->fence) rt_fatal("dlaa: replay used an older SR feature scene");
    size_scene_use(ex, e);
    e->sr.scene = ex->fence;
}

// Only the recorder or the drained teardown owner calls the SDK. Release is
// handle retirement; NGX may keep its own cache and no budget credit follows.
static void ngx_sr_release(NgxSrFeature *sr) {
    if (sr->handle) {
        if (!N.release) rt_fatal("dlaa: release missing for an owned DLSS SR feature");
        NgxResult result = N.release(sr->handle);
        if (!NGX_OK(result)) rt_fatal("dlaa: DLSS SR feature release failed (%08x)", (unsigned)result);
    }
    *sr = (NgxSrFeature){0};
}

static void destroy_size(struct Size *e) {
    if (!size_idle(e)) rt_fatal("dlaa: destroy of an executing size generation");
    int had_feature = e->sr.handle != NULL;
    ngx_sr_release(&e->sr);
    if (had_feature) feat_released = 1;
    dlaa_size_native_destroy(e);
    memset(e, 0, sizeof *e);
}

static int size_collect_retired(void) {
    int collected = 0;
    for (int i = 0; i < DLSS_SIZES; ++i) {
        struct Size *e = &D.sizes[i];
        if (e->generation && __atomic_load_n(&e->retired, __ATOMIC_ACQUIRE) && size_idle(e)) {
            size_mark_retired(e);
            destroy_size(e);
            ++collected;
        }
    }
    return collected;
}

void vk_dlaa_size_shutdown(void) {
    if (!vk_ctx() || vk_device_is_lost()) return;
    vk_dlaa_swap_img = NULL, vk_dlaa_swap_view = VK_NULL_HANDLE, vk_dlaa_swap_output = VK_NULL_HANDLE;
    dlaa_frame_targets_close();
    fg_forget_size();
    for (int i = 0; i < DLSS_SIZES; ++i) {
        struct Size *e = &D.sizes[i];
        if (!e->generation) continue;
        size_mark_retired(e);
        destroy_size(e);
    }
    size_hook_close();
}

typedef struct {
    uint32_t w, h, ow, oh;
    uint64_t scene;
    struct Size *ready;
    DlaaSizeNativeOutcome outcome;
    RcBudgetReadiness observed;
    const char *failure;
} DlaaSizeRequest;

static struct Size *size_find(const DlaaSizeRequest *request) {
    if (D.preset_dirty) return NULL;
    for (int i = 0; i < DLSS_SIZES; ++i) {
        struct Size *e = &D.sizes[i];
        if (e->generation && !__atomic_load_n(&e->retired, __ATOMIC_ACQUIRE) &&
            e->w == request->w && e->h == request->h && e->ow == request->ow && e->oh == request->oh)
            return e;
    }
    return NULL;
}

static struct Size *size_candidate(const DlaaSizeRequest *request, int allow_empty) {
    struct Size *candidate = NULL;
    for (int i = 0; i < DLSS_SIZES; ++i) {
        struct Size *e = &D.sizes[i];
        if (!e->generation) {
            if (allow_empty) return e;
            continue;
        }
        if (e == D.last || e == D.swap_size || e == D.fg_size ||
            __atomic_load_n(&e->last_scene, __ATOMIC_ACQUIRE) == request->scene || !size_idle(e)) continue;
        if (!candidate || e->used < candidate->used) candidate = e;
    }
    return candidate;
}

// A cold size can wait for older accepted users and actual capacity. An
// unchanged condition must not repeatedly materialize and roll back siblings.
#define DLAA_SIZE_NO_PROGRESS_NS UINT64_C(5000000000)
static int prepare_size(void *context) {
    DlaaSizeRequest *request = context;
    DlaaSizeNativePlan plan;
    request->outcome = dlaa_size_native_plan(request->w, request->h, request->ow, request->oh, &plan);
    if (request->outcome.state != DLAA_SIZE_NATIVE_READY)
        return request->outcome.state != DLAA_SIZE_NATIVE_STOPPED;

    int refused = 0, reclaimed = 0;
    uint64_t progress_at = kern_now_ns();
    while (!vk_sync_stopping() && !vk_device_is_lost()) {
        vk_present_retire_temporal_inputs();
        reclaimed |= size_collect_retired() != 0;
        struct Size *slot = size_candidate(request, 1);
        resource_core_budget_readiness(1, &request->observed);
        int records = refused && (request->outcome.admission == RC_ADMISSION_NO_RECORDS ||
                                  request->outcome.child.owner_full);
        int allowed = !refused || reclaimed || (!records &&
            resource_core_budget_peak_progress(&request->outcome.peak, &request->outcome.before, &request->observed));
        if (slot && allowed) {
            struct Size next = {0};
            request->outcome = dlaa_size_native_create(&plan, &next);
            reclaimed = 0;
            if (request->outcome.state == DLAA_SIZE_NATIVE_READY) {
                if (slot->generation) {
                    size_mark_retired(slot);
                    destroy_size(slot);
                }
                if (D.preset_dirty) {
                    for (int i = 0; i < DLSS_SIZES; ++i)
                        if (D.sizes[i].generation) size_mark_retired(&D.sizes[i]);
                    D.preset_dirty = 0;
                }
                if (++size_generation == 0) rt_fatal("dlaa: size generation overflow");
                next.generation = size_generation;
                *slot = next;
                request->ready = slot;
                return 1;
            }
            if (request->outcome.state != DLAA_SIZE_NATIVE_PENDING)
                return request->outcome.state != DLAA_SIZE_NATIVE_STOPPED;
            refused = 1;
            records = request->outcome.admission == RC_ADMISSION_NO_RECORDS || request->outcome.child.owner_full;
        }
        if (refused) {
            // Keep the prior published generation. A distinct inactive group
            // that really retires can return all seven of its native records.
            struct Size *victim = size_candidate(request, 0);
            if (victim) {
                size_mark_retired(victim);
                destroy_size(victim);
                reclaimed = 1;
                progress_at = kern_now_ns();
                continue;
            }
        }
        if (kern_now_ns() - progress_at >= DLAA_SIZE_NO_PROGRESS_NS) {
            request->failure = !slot ? "no retired size slot" : records ? "seven allocation records unavailable" :
                               "full native peak unavailable";
            request->outcome.state = DLAA_SIZE_NATIVE_TERMINAL;
            request->outcome.vk_result = VK_NOT_READY;
            return 1;
        }
        RcBudgetProgress progress;
        resource_core_budget_service(1, &request->outcome.peak, &progress);
        resource_core_budget_wait_since(progress.event, 100);
    }
    request->outcome.state = DLAA_SIZE_NATIVE_STOPPED;
    return 0;
}

static struct Size *size_for(VkExec *ex, uint32_t w, uint32_t h, uint32_t ow, uint32_t oh) {
    DlaaSizeRequest request = {.w = w, .h = h, .ow = ow, .oh = oh, .scene = ex->fence};
    struct Size *e = size_find(&request);
    if (e) return e;
    VkSyncProgressResult progress = vk_exec_temporal_prepare(ex, prepare_size, &request);
    vk_device_lost_check(request.outcome.vk_result, "preparing temporal size images");
    if (progress == VK_SYNC_PROGRESS_COMPLETE && request.ready) return request.ready;
    if (request.ready) size_mark_retired(request.ready);
    if (progress == VK_SYNC_PROGRESS_STOPPED || request.outcome.state == DLAA_SIZE_NATIVE_STOPPED) {
        vk_exec_cancel_temporal_preparation();
    } else if (!vk_sync_stopping() && !vk_device_is_lost()) {
        char diagnostic[240];
        int domain = -1;
        uint64_t needed = 0, available = 0;
        for (unsigned i = 0; i < RC_ADMISSION_MAX_DOMAINS; ++i) {
            uint64_t peak = request.outcome.peak.amount[i], room = request.observed.available.amount[i];
            if (peak <= room) continue;
            domain = (int)i;
            needed = peak;
            available = room;
            break;
        }
        snprintf(diagnostic, sizeof diagnostic,
                 "cannot prepare %ux%u -> %ux%u (%s, park %d, Vulkan %d, budget %d, domain %d needs %llu / has %llu bytes)",
                 w, h, ow, oh, request.failure ? request.failure : "native preparation failed",
                 (int)progress, (int)request.outcome.vk_result, (int)request.outcome.admission, domain,
                 (unsigned long long)needed, (unsigned long long)available);
        vk_exec_required_temporal_unavailable(ex, diagnostic);
    }
    return NULL;
}

static void size_record_begin(VkExec *ex, struct Size *e) {
    size_scene_use(ex, e);
    if (e->initialized) return;
    VkCommandBuffer cb = ex->cb;
    VkImageMemoryBarrier2 ib[7];
    VkImage im[7] = {e->out, e->mv, e->mask, e->chars[0], e->chars[1], e->dist[0], e->dist[1]};
    for (int i = 0; i < 7; i++)
        ib[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = im[i],
                                        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 7,
                           .pImageMemoryBarriers = ib};
    vkCmdPipelineBarrier2(cb, &di);
    e->initialized = 1;
}

static void full_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

static NgxResourceVK res_image(VkImageView v, VkImage img, VkFormat f, uint32_t w, uint32_t h, VkImageAspectFlags a,
                               int rw) {
    NgxResourceVK r;
    memset(&r, 0, sizeof r);
    r.r.img = (NgxImageView){v, img, {a, 0, 1, 0, 1}, f, w, h};
    r.type = 0;
    r.rw = rw;
    return r;
}

// ---------------------------------------------------------------- timing (NX2NAVI_DLAA_TIMES=1)
// GPU timestamps around the camera motion dispatch, the per-object motion pass and DLSS's evaluation, and the CPU
// time of NGX's call and of draining the recorder before it, summed and logged every TIMES_SUMMARY frames:
//   dlaa times: gpu camera motion a ms, object motion b ms, DLSS c ms (frames timed); cpu: recorder drain d ms, NGX
//   evaluate e ms, motion pass recording f ms
// Each frame takes its own slot of TIMES_SLOTS; a slot's results are read when it comes round again (by then the GPU is
// long done with it; not ready: that frame is left out).
enum { T_MV0, T_MV1, T_OBJ, T_DL0, T_DL1, T_N };
#define TIMES_SLOTS 8u
#define TIMES_SUMMARY 600
static struct {
    int on;                      // -1 not read
    VkQueryPool pool;
    uint32_t slot;
    int used[TIMES_SLOTS];       // the slot holds a frame's timestamps (all T_N written)
    double gpu[3];               // ms summed: camera motion, object motion, DLSS
    uint64_t frames, cpu_drain, cpu_ngx, cpu_mv;   // ns summed
} T = {.on = -1};

static int times_on(void) {
    if (T.on < 0) {
        T.on = rt_knob("DLAA_TIMES", "0");
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                    .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = TIMES_SLOTS * T_N};
        if (T.on && vkCreateQueryPool(vk_ctx()->device, &qi, NULL, &T.pool) != VK_SUCCESS) T.on = 0;
    }
    return T.on;
}

static void times_mark(VkCommandBuffer cb, int which) {
    if (times_on()) vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, T.pool, T.slot * T_N + (uint32_t)which);
}

// At the start of a frame's timing (mv_pass): the slot's last results are read and it is reset.
static void times_begin(VkCommandBuffer cb) {
    if (!times_on()) return;
    T.slot = (T.slot + 1) % TIMES_SLOTS;
    if (T.used[T.slot]) {
        uint64_t q[T_N];
        if (vkGetQueryPoolResults(vk_ctx()->device, T.pool, T.slot * T_N, T_N, sizeof q, q, sizeof q[0],
                                  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            double ns = vk_ctx()->props.limits.timestampPeriod / 1e6;   // ticks to ms
            T.gpu[0] += (double)(q[T_MV1] - q[T_MV0]) * ns, T.gpu[1] += (double)(q[T_OBJ] - q[T_MV1]) * ns;
            T.gpu[2] += (double)(q[T_DL1] - q[T_DL0]) * ns;
            if (++T.frames % TIMES_SUMMARY == 0) {
                rt_log("dlaa times: gpu camera motion %.3f ms, object motion %.3f ms, DLSS %.3f ms (%llu frames timed); "
                       "cpu: recorder drain %.3f ms, NGX evaluate %.3f ms, motion pass recording %.3f ms",
                       T.gpu[0] / TIMES_SUMMARY, T.gpu[1] / TIMES_SUMMARY, T.gpu[2] / TIMES_SUMMARY,
                       (unsigned long long)T.frames, T.cpu_drain / 1e6 / TIMES_SUMMARY, T.cpu_ngx / 1e6 / TIMES_SUMMARY,
                       T.cpu_mv / 1e6 / TIMES_SUMMARY);
                T.gpu[0] = T.gpu[1] = T.gpu[2] = 0, T.cpu_drain = T.cpu_ngx = T.cpu_mv = 0;
            }
        }
    }
    T.used[T.slot] = 0;
    vkCmdResetQueryPool(cb, T.pool, T.slot * T_N, T_N);
}

// ---------------------------------------------------------------- motion vectors
// The camera's motion first (runtime/shaders/dlaa_mv.comp): this frame's scene camera block is copied out of the
// replay's uniform snapshot (vk_usnap.c, as of the submit call) into one slot of camb, last frame's stays in the other,
// and the compute pass reprojects every pixel's depth through the two (inv4 and mv_pass below). Then the per-object
// pass (vk_exec_draw.c, vk_mvobj_execute) writes the motion of whatever moves on its own over it.
#define CAM_BYTES 256u   // vec4 rows 0-15: view, view-projection (3-6), projection, inverse view
#define CAM_BLOCK 0x940u   // the whole block, kept for the per-object motion pass's vertex shaders (vk_exec_draw.c)
#define CAM_SLOT 0xa00u    // camb slot stride
#define MVM_SLOTS 16u      // frames the GPU may still be reading the reprojection blocks of

// dlaa_mv.comp's push constants in its layout (std430: dz at offset 32)
typedef struct {
    float size[2], jitter[2], bias, prev_ok, zfit, zprev_ok;
    float dz[4];   // the projection's [2][2], [2][3], [3][2], [3][3]
    float zmask;
} MvPush;

static RcAdmissionVector mv_refused;
static int mv_waiting_capacity;

static void mv_discard(const VkCtx *c) {
    if (D.mv_pipe) vkDestroyPipeline(c->device, D.mv_pipe, NULL);
    if (D.mv_layout) vkDestroyPipelineLayout(c->device, D.mv_layout, NULL);
    if (D.mv_dsl) vkDestroyDescriptorSetLayout(c->device, D.mv_dsl, NULL);
    if (D.mv_samp) vkDestroySampler(c->device, D.mv_samp, NULL);
    if (D.camb) resource_core_vma_destroy_buffer(c->vma, D.camb, D.camb_mem);
    if (D.mvm) resource_core_vma_destroy_buffer(c->vma, D.mvm, D.mvm_mem);
    D.mv_pipe = VK_NULL_HANDLE;
    D.mv_layout = VK_NULL_HANDLE;
    D.mv_dsl = VK_NULL_HANDLE;
    D.mv_samp = VK_NULL_HANDLE;
    D.camb = D.mvm = VK_NULL_HANDLE;
    D.camb_mem = D.mvm_mem = NULL;
    D.mvm_map = NULL;
}

void vk_dlaa_motion_shutdown(void) {
    const VkCtx *c = vk_ctx();
    if (!c) return;
    mv_discard(c);
    mv_waiting_capacity = 0;
    D.mv_ok = -1;
}

static int mv_init(void) {
    if (D.mv_ok >= 0) return D.mv_ok;
    if (vk_sync_stopping() || vk_device_is_lost()) return 0;
    if (mv_waiting_capacity) {
        RcBudgetReadiness now;
        resource_core_budget_readiness(1, &now);
        if (!resource_core_budget_peak_ready(&mv_refused, &now)) return 0;
    }
    const VkCtx *c = vk_ctx();
    RcBudgetTicket joint = {0};
    RcAdmissionVector peak = {0};
    int peak_known = 0;
    VkDescriptorSetLayoutBinding b[10] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 10, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MvPush)};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
                              .minFilter = VK_FILTER_NEAREST, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof dlaa_mv_spv,
                                   .pCode = dlaa_mv_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 2 * CAM_SLOT,
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                  .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT};
    VkBufferCreateInfo mbi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = MVM_SLOTS * 2 * CAM_BYTES,
                              .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT};
    VmaAllocationCreateInfo mai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                   .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT |
                                            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                            VMA_ALLOCATION_CREATE_MAPPED_BIT};
    RcVmaRequest request = {.producer = RC_BUDGET_OTHER, .required = 1};
    VmaAllocationCreateInfo host, device;
    RcAdmissionVector dev_peak;
    VkResult r = resource_core_vma_buffer_peak(&request, &mbi, &mai, &host, &peak);
    if (r != VK_SUCCESS) goto fail;
    r = resource_core_vma_buffer_peak(&request, &bi, &ai, &device, &dev_peak);
    if (r != VK_SUCCESS) goto fail;
    for (unsigned d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) {
        if (dev_peak.amount[d] > UINT64_MAX - peak.amount[d]) {
            r = VK_ERROR_OUT_OF_DEVICE_MEMORY;
            goto fail;
        }
        peak.amount[d] += dev_peak.amount[d];
    }
    peak_known = 1;
    if (resource_core_budget_reserve(request.producer, request.required, &peak, &joint) != RC_ADMISSION_OK) {
        r = VK_NOT_READY;
        goto fail;
    }
    request.peak_ticket = &joint;
    VmaAllocationInfo minfo;
    r = resource_core_vma_create_buffer(&request, c->vma, &mbi, &host, &D.mvm, &D.mvm_mem, &minfo, NULL);
    if (r != VK_SUCCESS) goto fail;
    if (!minfo.pMappedData) {
        r = VK_ERROR_MEMORY_MAP_FAILED;
        goto fail;
    }
    D.mvm_map = minfo.pMappedData;
    r = resource_core_vma_create_buffer(&request, c->vma, &bi, &device, &D.camb, &D.camb_mem, NULL, NULL);
    if (r != VK_SUCCESS) goto fail;
    r = vkCreateDescriptorSetLayout(c->device, &dci, NULL, &D.mv_dsl);
    if (r != VK_SUCCESS) goto fail;
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &D.mv_dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    r = vkCreatePipelineLayout(c->device, &li, NULL, &D.mv_layout);
    if (r != VK_SUCCESS) goto fail;
    r = vkCreateSampler(c->device, &si, NULL, &D.mv_samp);
    if (r != VK_SUCCESS) goto fail;
    r = vkCreateShaderModule(c->device, &mi, NULL, &m);
    if (r != VK_SUCCESS) goto fail;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = D.mv_layout};
    r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &D.mv_pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    if (r != VK_SUCCESS) goto fail;
    if (resource_core_budget_cancel(&joint) != RC_ADMISSION_OK)
        rt_fatal("dlaa: motion buffers lost their joint admission");
    mv_waiting_capacity = 0;
    return D.mv_ok = 1;
fail:
    mv_discard(c); // No command can use a motion provider before mv_ok becomes ready.
    if (joint.token.ledger && resource_core_budget_cancel(&joint) != RC_ADMISSION_OK)
        rt_fatal("dlaa: motion buffers lost their rollback reservation");
    vk_device_lost_check(r, "preparing the motion-vector pass");
    if (r == VK_NOT_READY) {
        if (!mv_waiting_capacity) rt_log("dlaa: motion-vector buffers await required admission");
        if (peak_known) {
            mv_refused = peak;
            mv_waiting_capacity = 1;
        }
    } else {
        mv_waiting_capacity = 0;
        D.mv_ok = 0;
        rt_log("dlaa: the motion-vector pass could not be set up (%d): DLAA runs with zero motion", (int)r);
    }
    return 0;
}

// The reprojection from this frame's clip space to last frame's, last * inverse(this), in double precision: the shader
// took each pixel back to world space through a float inverse of the view-projection, and with NX2navi's world
// coordinates in the thousands and its far plane the result was noise of a fraction of a pixel and more on distant
// surfaces, which DLSS followed into a shimmer like heat haze (roofs and trees, most of all in the upscaling modes).
// The shader is unchanged: its "this frame" block gets the identity and its "last frame" block this matrix, so it
// applies just the one matrix to each pixel. A still camera gives the identity, and zero motion exactly.
static int inv4(const double m[16], double o[16]) {
    double a[4][8];
    for (int r = 0; r < 4; r++)
        for (int k = 0; k < 8; k++) a[r][k] = k < 4 ? m[r * 4 + k] : (k - 4 == r);
    for (int col = 0; col < 4; col++) {
        int piv = col;
        for (int r = col + 1; r < 4; r++)
            if (fabs(a[r][col]) > fabs(a[piv][col])) piv = r;
        if (fabs(a[piv][col]) < 1e-300) return 0;
        for (int k = 0; k < 8; k++) { double t = a[col][k]; a[col][k] = a[piv][k]; a[piv][k] = t; }
        double d = a[col][col];
        for (int k = 0; k < 8; k++) a[col][k] /= d;
        for (int r = 0; r < 4; r++)
            if (r != col) {
                double f = a[r][col];
                for (int k = 0; k < 8; k++) a[r][k] -= f * a[col][k];
            }
    }
    for (int r = 0; r < 4; r++)
        for (int k = 0; k < 4; k++) o[r * 4 + k] = a[r][k + 4];
    return 1;
}

// Ocarina's camera from its scene context block (vk_dlaa_draw): its scene vertex shaders take the world position to
// the camera through rows 0-2 (a 3x4 view) and then to clip space through rows 7-10 (the projection, 4x4), so the
// view-projection is rows 7-10 times rows 0-2.
static void ocam_vp(const float *v, double out[16]) {
    const float *P = v + 7 * 4;
    double V[16] = {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], 0, 0, 0, 1};
    for (int r = 0; r < 4; r++)
        for (int k = 0; k < 4; k++) {
            double acc = 0;
            for (int j = 0; j < 4; j++) acc += (double)P[r * 4 + j] * V[j * 4 + k];
            out[r * 4 + k] = acc;
        }
}

// Frame generation's camera (vk_dlaa_fg_camera) from the Ocarina block: the projection is rows 7-10, the view rows 0-2
// (rows: right, up, back; the camera looks down -z), m the reprojection from this frame's clip space to last frame's.
// Each matrix is handed out transposed, for row vectors.
static void fg_store(float out[16], const double m[16]) {
    for (int r = 0; r < 4; r++)
        for (int k = 0; k < 4; k++) out[r * 4 + k] = (float)m[k * 4 + r];
}
// The view distance in front of the camera at which the projection P (row-major, column vectors, looking down -z) gives
// clip depth z_ndc = k: (P[10] z + P[11]) / (P[14] z + P[15]) = k, solved for z. INFINITY where no distance gives it.
static double plane_dist(const double P[16], double k) {
    double den = P[10] - k * P[14];
    if (fabs(den) <= 1e-9 * (fabs(P[10]) + fabs(P[14]))) return INFINITY;
    return -(k * P[15] - P[11]) / den;
}

// What the depth buffer's 0 and 1 stand for, from the frame's projection and the device's depth mode. With NVN's
// NEAR_IS_MINUS_W (clip depth -w..w, which the pipelines keep through negativeOneToOne, vk_pipeline.c) a fragment
// stores z_ndc * 0.5 + 0.5, so 0 is z_ndc = -1; with NEAR_IS_ZERO it stores z_ndc. Both are 0..1 in the D32F target
// (the viewport's depth range, vk_exec_draw.c), and the -w..w mapping stores exactly what a 0..w projection with the
// same planes would: (f / (f - n)) (1 - n / distance), the convention DLSS, Streamline and FSR take. Ocarina's planes
// are not fixed: in Hyrule Field its z row is (-1.00001, -0.200001), near 0.1 and far about 25,000 (the view's units
// are metres), and the camera before it had far about 1,000. Taken as 0.1 and 1e8 (no far plane), a distance d reads
// as (f - n) d / (f - d): 1,000 as 1,042, 10,000 as 16,650, and the far plane's neighbourhood without bound. So the
// planes come from each evaluation's projection.
static int proj_planes(const double P[16], VkDlaaCamera *c) {
    int minus_w = nvn_depth_minus_w();
    double at0 = plane_dist(P, minus_w ? -1.0 : 0.0), at1 = plane_dist(P, 1.0);   // the distances stored as 0 and 1
    int inverted = !(at0 < at1);   // reversed depth: the near plane is stored as 1
    double n = inverted ? at1 : at0, f = inverted ? at0 : at1;
    if (!(n > 0) || !isfinite(n)) return 0;
    c->inverted = inverted;
    c->znear = (float)n;
    c->infinite = !(f < VK_DLAA_FAR_INFINITE);
    c->zfar = c->infinite ? VK_DLAA_FAR_INFINITE : (float)f;
    // one line at the start and whenever the planes change (the first 20 times)
    static float logged_n, logged_f;
    static int logged_inv = -1, lines;
    if (lines < 20 && (inverted != logged_inv || fabsf(c->znear - logged_n) > 1e-4f * logged_n ||
                       fabsf(c->zfar - logged_f) > 1e-4f * logged_f)) {
        lines++, logged_inv = inverted, logged_n = c->znear, logged_f = c->zfar;
        char fz[32];
        if (c->infinite) snprintf(fz, sizeof fz, "infinite");
        else snprintf(fz, sizeof fz, "%.6g", c->zfar);
        rt_log("dlaa: the scene's projection: near %.6g, far %s, depth %s, clip depth %s (z row %.6g %.6g %.6g %.6g, "
               "w row %.6g %.6g %.6g %.6g)", c->znear, fz, inverted ? "inverted (1 near)" : "0 near, 1 far",
               minus_w ? "-w..w" : "0..w", P[8], P[9], P[10], P[11], P[12], P[13], P[14], P[15]);
    }
    return 1;
}

static int fg_camera(const float *v, const double m[16], VkDlaaCamera *c) {
    double P[16], Pi[16], mi[16];
    for (int i = 0; i < 16; i++) P[i] = v[28 + i];
    // The tagged D32 image stores zero to one. Express the projection and both
    // endpoints of the reprojection in that same basis before taking their inverses.
    double projection[16], reprojection[16];
    memcpy(projection, P, sizeof projection);
    memcpy(reprojection, m, sizeof reprojection);
    if (nvn_depth_minus_w()) {
        for (int i = 0; i < 4; ++i) {
            projection[8 + i] = (P[8 + i] + P[12 + i]) * 0.5;
            reprojection[8 + i] = (m[8 + i] + m[12 + i]) * 0.5;
        }
        for (int i = 0; i < 4; ++i) {
            reprojection[4 * i + 3] -= reprojection[4 * i + 2];
            reprojection[4 * i + 2] *= 2.0;
        }
    }
    if (!inv4(projection, Pi) || !inv4(reprojection, mi) || !proj_planes(P, c)) return 0;
    fg_store(c->view_to_clip, projection), fg_store(c->clip_to_view, Pi);
    fg_store(c->clip_to_prev_clip, reprojection), fg_store(c->prev_clip_to_clip, mi);
    for (int i = 0; i < 3; i++) {
        c->right[i] = v[i], c->up[i] = v[4 + i], c->fwd[i] = -v[8 + i];
        // the position: -R^T t, R the view's 3x3 and t its last column
        c->pos[i] = -(v[i] * v[3] + v[4 + i] * v[7] + v[8 + i] * v[11]);
    }
    c->fov_y = P[5] != 0 ? 2.0f * (float)atan(1.0 / fabs(P[5])) : 1.0f;
    c->aspect = P[0] != 0 ? (float)fabs(P[5] / P[0]) : 1.777f;
    // NX2NAVI_MOUSE_TRACE=1 (test runs, mouse_direct.c): the camera's direction each frame
    static int trace = -1;
    if (trace < 0) {
        const char *e = cfg_get("NX2NAVI_MOUSE_TRACE");
        trace = e && *e && strcmp(e, "0");
    }
    if (trace)
        rt_log("mousetrace: cam t=%.3f fwd=%.6f,%.6f,%.6f pos=%.3f,%.3f,%.3f", plat_now_ns() / 1e6, c->fwd[0], c->fwd[1],
               c->fwd[2], c->pos[0], c->pos[1], c->pos[2]);
    return 1;
}

// Frames without a camera (dlaa_cam_keep, NX2NAVI_DLAA_CAM_KEEP; 1 by default, 0 restores the old handling). Measured in
// Kokiri Forest during dialogue (Saria, dlss = performance): about 1 frame in 20 had no camera, and each started a new
// history twice, on that frame and on the next (nothing to reproject from): 24 to 26 resets per 300 frames. Those were
// frames the event order held past their present (vk_exec.c), which then ran in a submission without a uniform copy
// (the present thread's empty one), so the lookup at DLSS found nothing. With the change: no camera reset there in
// 3600 frames but the first. Two things change:
// - Ocarina's camera rows are taken from the bytes the scene draws themselves read (vk_dlaa_draw, views[].cam), not
//   looked up again in the uniform copy of the submission DLSS runs in, which on some frames has none or lost the
//   block to a GPU write.
// - A frame that has no camera at all keeps DLSS's history: its vectors are zero (the camera taken as still) and last
//   frame's camera stays the one the next frame reprojects from. At most KEEP_MAX frames in a row: a longer stretch
//   is something else (a cut scene with a camera of its own), and a still history there would trail what moves.
#define KEEP_MAX 6

static const char *const usnap_why[] = {"found", "no copy in this submission", "dropped by a GPU write in this submission",
                                        "left out: a GPU write in flight", "not memory the CPU writes",
                                        "not bound in this submission"};

// One line for each of the first frames without a camera: what the frame drew into the scene depth, which submission
// the scene context was seen in and why the evaluating one lacks it, and what was done.
static void cam_note(int view, int kept) {
    uint64_t seq = vk_exec_submit_seq;
    char snap[200] = "";
    if (D.cam_addr)
        snprintf(snap, sizeof snap, "; scene context %#llx last drawn %llu submission(s) before DLSS's, there: %s",
                 (unsigned long long)D.cam_addr, (unsigned long long)(view >= 0 ? seq - D.views[view].seq : 0),
                 usnap_why[vk_usnap_why(D.cam_addr, D.cam_size == OCAM_BLOCK ? OCAM_BYTES : D.cam_size)]);
    rt_log("dlaa: frame %llu without a camera: %llu scene-depth draws over %llu submission(s) (%u into another D32 "
           "depth %ux%u), %u with the scene context (%u not into a scene colour target, format %d), %u with another c3 "
           "block (last size %#x)%s; %s",
           (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1, (unsigned long long)F.zdraws,
           (unsigned long long)(F.zdraws + F.zother ? seq - F.seq0 + 1 : 0), F.zother, F.zother_w, F.zother_h, F.ocam,
           F.ocam_off, (int)F.ocam_off_fmt, F.c3_other, F.c3_size, snap,
           kept == 1 ? "history kept, zero motion" : kept == 2 ? "new history (more than KEEP_MAX in a row)"
                                                 : "new history");
}

// Camera cuts (NX2NAVI_DLAA_CUT, default 1; 0: off). Only a lost camera (mv_pass), another scene size or another
// target started a new history, so DLSS and frame generation (vk_dlaa_fg_inputs) carried their history across a camera
// that jumped: an area load or a spawn moves it 5 to 180 m in one frame. Such a frame now starts a new history, as
// NVIDIA's DLSS guide asks for a camera cut: a move over 5 m, or a turn over 10 degrees together with a move over 0.5 m,
// both per 60 Hz frame and scaled by the time since the last camera. Camera traces of play in the Fire Temple
// (NX2NAVI_MOUSE_TRACE) had a median of 0.05 m and 0.08 degrees a frame; the second test took only the area loads and a
// swing of the camera through a doorway at 12 to 18 degrees and 0.7 to 1.4 m a frame. A turn alone (the mouse camera,
// which orbits about 3 m from Link) does not count. Each cut is logged, up to 24.
// v: the Ocarina scene context's rows 0-2, the view (fg_camera).
static void cam_cut(const float *v) {
    static int on = -1;
    if (on < 0) on = rt_knob("DLAA_CUT", "1");
    float pos[3], fwd[3];
    for (int i = 0; i < 3; i++) {
        fwd[i] = -v[8 + i];
        pos[i] = -(v[i] * v[3] + v[4 + i] * v[7] + v[8 + i] * v[11]);
    }
    uint64_t now = plat_now_ns();
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    if (on && D.cut_ok) {
        double dot = fwd[0] * D.cut_fwd[0] + fwd[1] * D.cut_fwd[1] + fwd[2] * D.cut_fwd[2];
        double turn = acos(dot > 1 ? 1 : dot < -1 ? -1 : dot) * (180.0 / 3.14159265358979);
        double dx = pos[0] - D.cut_pos[0], dy = pos[1] - D.cut_pos[1], dz = pos[2] - D.cut_pos[2];
        double move = sqrt(dx * dx + dy * dy + dz * dz);
        double k = (now - D.cut_ns) / 1e6 / (1000.0 / 60);   // 60 Hz frames since the last camera
        if (k < 0.25) k = 0.25;
        if (k > 4) k = 4;
        int jumped = move > 5.0 * k || (turn > 10.0 * k && move > 0.5 * k);
        if (jumped && !(D.cam_prev_ok && frame == D.cut_frame + 1 && camera_history_continuous(frame, D.cut_view, v))) {
            D.reset = 1, D.cuts++;
            if (D.cut_logged < 24)
                D.cut_logged++, rt_log("dlaa: frame %llu: camera cut (moved %.2f m, turned %.1f degrees in %.1f ms): new "
                                       "history (%llu so far)",
                                       (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1,
                                       move, turn, (now - D.cut_ns) / 1e6, (unsigned long long)D.cuts);
        }
    }
    memcpy(D.cut_pos, pos, sizeof pos), memcpy(D.cut_fwd, fwd, sizeof fwd);
    memcpy(D.cut_view, v, sizeof D.cut_view);
    D.cut_frame = frame;
    D.cut_ns = now, D.cut_ok = 1;
}

// The pass, recorded through the recorder like the replay's own commands. 0: no camera this frame (the vectors stay
// zero and DLSS starts a new history); 2: no camera, but the history is kept (the vectors stay zero, dlaa_cam_keep).
// bias > 0 with this frame's G-buffer classes (D.ids) also writes e->mask, the bias mask (dlaa_mv.comp); *masked says
// whether it did.
enum { MV_PASS_ABORTED = -1 };
static int mv_pass(VkExec *ex, struct Size *e, VkImageView zv, float jx, float jy, float bias, int *masked,
                   VkDlaaCamera *camera, int *camera_ok) {
    *masked = 0;
    uint32_t w = e->w, h = e->h;
    static int keep = -1;
    if (keep < 0) keep = rt_knob("DLAA_CAM_KEEP", "1");
    VkDescriptorBufferInfo cam;
    const int matching_depth = dlaa_camera_depth_matches(&D.depth);
    int view = -1;   // Ocarina: the views[] entry of the camera
    for (int i = 0; matching_depth && D.cam_size == OCAM_BLOCK && D.cam_addr && i < D.nviews; i++)
        if (D.views[i].addr == D.cam_addr) view = i;
    const float *held = keep && view >= 0 && D.views[view].got ? D.views[view].cam : NULL;
    int found = matching_depth && mv_init() && D.cam_addr &&
        vk_usnap_find(D.cam_addr, D.cam_size == OCAM_BLOCK ? OCAM_BYTES : D.cam_size, 4, &cam);
    if (!found && held && D.mv_ok == 1) {
        D.cam_at_draw++;
        if (D.gap_logged < 40) {
            D.gap_logged++;
            rt_log("dlaa: frame %llu: camera read at the draw: scene context %#llx last drawn %llu submission(s) before "
                   "DLSS's, there: %s; the frame's scene-depth draws over %llu submission(s); %d held submissions",
                   (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1,
                   (unsigned long long)D.cam_addr, (unsigned long long)(vk_exec_submit_seq - D.views[view].seq),
                   usnap_why[vk_usnap_why(D.cam_addr, OCAM_BYTES)],
                   (unsigned long long)(F.zdraws + F.zother ? vk_exec_submit_seq - F.seq0 + 1 : 0), vk_exec_held());
        }
    }
    else if (!found) {
        D.mv_none++;
        if (matching_depth && D.cam_addr) D.cam_none_snap++;
        else D.cam_none_draw++;
        e->chars_ok = e->dist_ok = 0;
        int kept = 0;
        if (keep && D.mv_ok == 1 && D.cam_prev_ok) {
            if (D.keep_run < KEEP_MAX) kept = 1, D.keep_run++, D.cam_kept++;
            else kept = 2, D.cam_kept_max++;
        }
        if (D.gap_logged < 40) D.gap_logged++, cam_note(view, kept);
        if (kept == 1) return 2;
        D.cam_prev_ok = 0;
        return 0;
    }
    D.keep_run = 0;
    // this frame's view-projection as the scene draws read it (Ocarina, dlaa_cam_keep) or as the snapshot holds it, for
    // the reprojection; kept as last frame's for the next
    const float *cr = held ? held
                      : D.cam_size == OCAM_BLOCK ? (const float *)vk_usnap_bytes(D.cam_addr, OCAM_BYTES)
                                                 : (const float *)vk_usnap_bytes(D.cam_addr + D.cam_vp, 64);
    double cur_vp[16], m[16], ci[16];
    int have_m = 0;
    if (cr && D.cam_size == OCAM_BLOCK) ocam_vp(cr, cur_vp);
    else if (cr)
        for (int i = 0; i < 16; i++) cur_vp[i] = cr[i];
    if (cr) {
        if (D.cam_prev_ok && inv4(cur_vp, ci)) {
            for (int r = 0; r < 4; r++)
                for (int k = 0; k < 4; k++) {
                    double v = 0;
                    for (int j = 0; j < 4; j++) v += D.cam_prev[r * 4 + j] * ci[j * 4 + k];
                    m[r * 4 + k] = v;
                }
            have_m = 1;
        }
    }
    if (cr && have_m && D.cam_size == OCAM_BLOCK) *camera_ok = fg_camera(cr, m, camera);
    if (cr && D.cam_size == OCAM_BLOCK) cam_cut(cr);
    int s = D.cam_slot, had_prev = D.cam_prev_ok;
    VkShieldResolve *shield = NULL;
    if (had_prev && have_m && D.cam_size == OCAM_BLOCK &&
        !vk_shield_resolve_prepare(ex, D.cam_addr, D.depth.image, w, h, D.reset, &shield)) return MV_PASS_ABORTED;
    VkCommandBuffer cb = ex->cb;
    if (D.cam_size == CAM_BLOCK) {   // the whole block, for the per-object motion pass (the deferred engine's only)
        VkBufferCopy bc = {cam.offset, (VkDeviceSize)s * CAM_SLOT, CAM_BLOCK};
        vkCmdCopyBuffer(cb, cam.buffer, D.camb, 1, &bc);
        full_barrier(cb);
    }
    D.cam_slot ^= 1, D.cam_prev_ok = cr != NULL;
    if (cr) memcpy(D.cam_prev, cur_vp, sizeof cur_vp);
    if (!had_prev || !have_m) {   // nothing to reproject into: zero motion, new history
        e->chars_ok = e->dist_ok = 0;
        return 0;
    }
    VkDescriptorImageInfo zi = {D.mv_samp, zv, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo oi = {VK_NULL_HANDLE, e->mv_view, VK_IMAGE_LAYOUT_GENERAL};
    uint32_t ms = D.mvm_slot++ % MVM_SLOTS;
    float *blk = (float *)(D.mvm_map + (size_t)ms * 2 * CAM_BYTES);   // rows 3-6 of each block are read
    memset(blk, 0, 2 * CAM_BYTES);
    for (int r = 0; r < 4; r++)
        for (int k = 0; k < 4; k++)
            blk[12 + r * 4 + k] = r == k ? 1.0f : 0.0f, blk[64 + 12 + r * 4 + k] = (float)m[r * 4 + k];
    vmaFlushAllocation(vk_ctx()->vma, D.mvm_mem, (VkDeviceSize)ms * 2 * CAM_BYTES, 2 * CAM_BYTES);
    VkDescriptorBufferInfo cur = {D.mvm, (VkDeviceSize)ms * 2 * CAM_BYTES, CAM_BYTES};
    VkDescriptorBufferInfo prev = {D.mvm, (VkDeviceSize)ms * 2 * CAM_BYTES + CAM_BYTES, CAM_BYTES};
    // The class mask needs the G-buffer at the scene size; otherwise bindings 4-7 get the mask images as stand-ins
    // (the shader returns before reading them with bias 0).
    const VkExecTarget *ids = &D.ids;
    int use = bias > 0 && D.ids_ok && dlaa_target_valid(frame_targets[DLAA_TARGET_IDS]) &&
        ids->image && ids->width == w && ids->height == h && !ids->is3d;
    VkImageView iv = use ? vk_exec_target_view(ids->img, ids->format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT,
                                               ids->level, ids->base_layer, 1, 0)
                         : e->mask_view;
    int cur_c = e->chars_cur;
    VkDescriptorImageInfo ii = {D.mv_samp, iv, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo mki = {VK_NULL_HANDLE, e->mask_view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo cpi = {VK_NULL_HANDLE, e->chars_view[cur_c ^ 1], VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo coi = {VK_NULL_HANDLE, e->chars_view[cur_c], VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dpi = {VK_NULL_HANDLE, e->dist_view[e->dist_cur ^ 1], VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo doi = {VK_NULL_HANDLE, e->dist_view[e->dist_cur], VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wr[10] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &zi},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &oi},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 2, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &cur},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 3, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &prev},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 4, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 5, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &mki},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 6, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &cpi},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 7, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &coi},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 8, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &dpi},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 9, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &doi}};
    times_begin(cb);
    times_mark(cb, T_MV0);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.mv_pipe);
    rec_push(vk_pipeline_push_fn(), cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.mv_layout, 0, 10, wr);
    // The depth check (dlaa_mv.comp, depth_prev) where nothing else gives characters their own motion: Ocarina's frame,
    // which has neither the per-object pass nor the class buffer. Its distances take the projection's depth terms from
    // rows 7-10 of the scene context (ocam_vp). NX2NAVI_DLAA_ZFIT=0 (test): the camera's motion alone, as before.
    static int zfit_on = -1;
    if (zfit_on < 0) zfit_on = rt_knob("DLAA_ZFIT", "1");
    int zfit = zfit_on && !use && cr && D.cam_size == OCAM_BLOCK;
    MvPush pc = {{(float)w, (float)h}, {jx, jy}, use ? bias : 0.0f, use && e->chars_ok ? 1.0f : 0.0f,
                 zfit ? 1.0f : 0.0f, zfit && e->dist_ok ? 1.0f : 0.0f,
                 {zfit ? cr[38] : 0.0f, zfit ? cr[39] : 0.0f, zfit ? cr[42] : 0.0f, zfit ? cr[43] : 0.0f},
                 zfit ? bias : 0.0f};
    vkCmdPushConstants(cb, D.mv_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
    full_barrier(cb);
    times_mark(cb, T_MV1);
    vk_mvcheck_camera(cb, e->mv, w, h, jx, jy);   // NX2NAVI_MV_CHECK (test)
    // the characters' own motion over the camera's (vk_exec_draw.c, per-object motion)
    uint64_t c0 = T.on > 0 ? kern_now_ns() : 0;
    if (D.cam_size == OCAM_BLOCK &&
        vk_shield_resolve_emit(ex, &shield, D.cam_addr, D.depth.image, zv, e->mv_view, w, h, D.reset))
        full_barrier(cb);
    if (D.cam_size == CAM_BLOCK && vk_mvobj_execute(ex, D.cam_addr, e->mv_view, zv, D.depth.format, w, h)) full_barrier(cb);
    if (T.on > 0) T.cpu_mv += kern_now_ns() - c0;
    times_mark(cb, T_OBJ);
    if (T.on > 0) T.used[T.slot] = 1;   // the DLSS marks follow in evaluate
    vk_mvcheck_final(cb, ex->fence, e->mv, D.color.image, D.color.format, D.color.level, D.color.base_layer, w, h, jx, jy);
    D.mv_frames++;
    if (use) e->chars_cur ^= 1, e->chars_ok = 1, D.masked++;
    else e->chars_ok = 0;
    if (zfit && bias > 0) D.masked++;
    e->dist_cur ^= 1, e->dist_ok = zfit;
    *masked = use || (zfit && bias > 0);
    return 1;
}

// Last frame's camera block for the per-object motion pass, recorded before this frame's mv_pass: that pass writes
// this frame's into slot cam_slot and leaves the other slot, last frame's, as it is. 0: none (no history).
int vk_dlaa_prev_camera(VkDescriptorBufferInfo *out, float jitter[2]) {
    if (!dl_active() || !D.cam_prev_ok || !D.camb) return 0;
    if (D.cam_size != CAM_BLOCK) return 0;   // the per-object pass knows the deferred engine's block only
    *out = (VkDescriptorBufferInfo){D.camb, (VkDeviceSize)(D.cam_slot ^ 1) * CAM_SLOT, CAM_BLOCK};
    jitter[0] = D.jit_ok ? D.jx : 0.0f, jitter[1] = D.jit_ok ? D.jy : 0.0f;
    return 1;
}

// ---------------------------------------------------------------- reactive mask
// Particles, smoke, sparks and other transparents have no motion vectors of their own: the motion image holds the
// surface's behind them, and DLSS carried their old colour along that surface (trails behind sparks, smoke smeared
// with the camera). DLSS takes a mask of where to trust this frame's colour over the history (NGX
// "DLSS.Input.Bias.Current.Color.Mask"). It is made from the picture itself, at no cost on the CPU: the light target is
// copied when the transparents start (the pass that arms DLAA, vk_dlaa_pass) and compared with what DLSS gets
// (runtime/dlaa_mask_spv.h, tools/make_dlaa_mask.py): the share of a pixel's colour that changed, from MASK_THR on,
// times MASK_GAIN. Off by default (NX2NAVI_DLSS_REACTIVE=1 turns it on, NX2NAVI_DLAA_MASKDEBUG=1 shows the mask instead of
// the picture): the pass that arms also draws the sky and blended fog layers over the solid scene, and the mask
// covered the sky, sunlit trees and roofs and a wide fog region besides the particles (seen with MASKDEBUG at 19:45 in
// the first save), where DLSS would give up its anti-aliasing. It needs the particle draws alone (drawn again into the
// mask, like the motion draws) before it can be on. In the upscaling modes the light target DLSS reads also holds the
// bloom by then; the copy would have to be compared at the end of the scene instead.
#include "dlaa_mask_spv.h"
#define MASK_THR 0.1f
#define MASK_GAIN 2.5f
#define SNAPS 3

static struct {
    struct Snap {                // per light target size and format: its copy at arming, and the mask
        uint32_t w, h;
        VkFormat f;
        VkImage before, mask;
        VmaAllocation before_mem, mask_mem;
        VkImageView before_view, mask_view;
        uint64_t used;
    } s[SNAPS];
    struct Snap *cur;            // this frame's copy, NULL: none
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
    int ok, knob;                // ok: -1 not built, 0 failed, 1 ready; knob: -1 not read
    uint64_t made, missed;       // frames with a mask; frames DLSS ran without the copy
} R = {.ok = -1, .knob = -1};

static int mask_init(void) {
    if (R.knob < 0) R.knob = rt_knob("DLSS_REACTIVE", "0");
    if (!R.knob) return 0;
    if (R.ok >= 0) return R.ok;
    R.ok = 0;
    const VkCtx *c = vk_ctx();
    VkDescriptorSetLayoutBinding b[3] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 3, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof dlaa_mask_spv,
                                   .pCode = dlaa_mask_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(c->device, &dci, NULL, &R.dsl) != VK_SUCCESS) goto fail;
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &R.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    if (vkCreatePipelineLayout(c->device, &li, NULL, &R.layout) != VK_SUCCESS ||
        vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS)
        goto fail;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = R.layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &R.pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    m = VK_NULL_HANDLE;
    if (r != VK_SUCCESS) goto fail;
    return R.ok = 1;
fail:
    if (m) vkDestroyShaderModule(c->device, m, NULL);   // made before a later step failed
    rt_log("dlaa: the reactive mask pass could not be set up: DLSS runs without it");
    return 0;
}

// A snapshot's images and views. The device must be idle: free_snap waits for it, vk_dlaa_release has.
static void destroy_snap(struct Snap *s) {
    const VkCtx *c = vk_ctx();
    if (s->before_view) vkDestroyImageView(c->device, s->before_view, NULL);
    if (s->mask_view) vkDestroyImageView(c->device, s->mask_view, NULL);
    if (s->before) vmaDestroyImage(c->vma, s->before, s->before_mem);
    if (s->mask) vmaDestroyImage(c->vma, s->mask, s->mask_mem);
    memset(s, 0, sizeof *s);
}

static void free_snap(struct Snap *s) {
    vk_queue_lock(), vk_device_wait_idle(), vk_queue_unlock();   // a fourth size: the evicted images may still be in flight
    destroy_snap(s);
}

static struct Snap *snap_for(VkCommandBuffer cb, uint32_t w, uint32_t h, VkFormat f) {
    static uint64_t tick;
    struct Snap *e = NULL, *lru = &R.s[0];
    for (int i = 0; i < SNAPS; i++) {
        if (R.s[i].w == w && R.s[i].h == h && R.s[i].f == f) return R.s[i].used = ++tick, &R.s[i];
        if (!R.s[i].before && !e) e = &R.s[i];
        if (R.s[i].used < lru->used) lru = &R.s[i];
    }
    if (!e) free_snap(e = lru);
    VkImageUsageFlags u = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (!image2d(w, h, f, u, &e->before, &e->before_mem, &e->before_view) ||
        !image2d(w, h, VK_FORMAT_R8_UNORM, u | VK_IMAGE_USAGE_STORAGE_BIT, &e->mask, &e->mask_mem, &e->mask_view)) {
        free_snap(e);
        return NULL;
    }
    VkImageMemoryBarrier2 ib[2];
    VkImage im[2] = {e->before, e->mask};
    for (int i = 0; i < 2; i++)
        ib[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = im[i],
                                        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 2,
                           .pImageMemoryBarriers = ib};
    vkCmdPipelineBarrier2(cb, &di);
    e->w = w, e->h = h, e->f = f, e->used = ++tick;
    return e;
}

// At arming: the light target as it is before the transparents, outside any pass.
static void mask_snapshot(VkExec *ex) {
    R.cur = NULL;
    if (mask_init() <= 0) return;
    const VkExecTarget *t = &D.color;
    struct Snap *s = snap_for(ex->cb, t->width, t->height, t->format);
    if (!s) return;
    vk_exec_barrier(ex);
    VkImageCopy c = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, t->level, t->base_layer, 1},
                     .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .extent = {t->width, t->height, 1}};
    vkCmdCopyImage(ex->cb, t->image, VK_IMAGE_LAYOUT_GENERAL, s->before, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    full_barrier(ex->cb);
    ex->recorded += 2;
    R.cur = s;
}

// Before DLSS: the mask from the copy and the light target (view cv). NULL: none this frame.
static struct Snap *mask_pass(VkCommandBuffer cb, VkImageView cv, uint32_t w, uint32_t h) {
    struct Snap *s = R.cur;
    R.cur = NULL;
    if (mask_init() <= 0) return NULL;
    if (!s || s->w != w || s->h != h || s->f != D.color.format) return R.missed++, NULL;
    VkDescriptorImageInfo bi = {D.mv_samp, s->before_view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo ai = {D.mv_samp, cv, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo mi = {VK_NULL_HANDLE, s->mask_view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wr[3] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &bi},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ai},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 2, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &mi}};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, R.pipe);
    rec_push(vk_pipeline_push_fn(), cb, VK_PIPELINE_BIND_POINT_COMPUTE, R.layout, 0, 3, wr);
    float pc[4] = {(float)w, (float)h, MASK_THR, MASK_GAIN};
    vkCmdPushConstants(cb, R.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
    full_barrier(cb);
    R.made++;
    return s;
}

// ---------------------------------------------------------------- teardown
// Release an unused alpha image or one whose graphics and SDK uses have retired.
static void free_alpha(VkNgxImage *a, VmaAllocation *allocation) {
    VkCtx *c = vk_ctx();
    if (a->view) vkDestroyImageView(c->device, a->view, NULL);
    if (a->img) resource_core_vma_destroy_image(c->vma, a->img, *allocation);
    *a = (VkNgxImage){0};
    *allocation = NULL;
}

// The device is idle here. Release NGX features and parameters before their Vulkan resources, then shut down NGX even
// if initialization succeeded but the layout check failed. The Linux core's Shutdown1 takes an extra output pointer
// compared with the SDK's public wrapper; it reports the number of devices still using NGX.
// ---- the free camera's view of the ground (debug_tools.c dbg_pad). While it flies, nine of the view distances
// mv_pass wrote (a grid over the lower part of the view, where the ground is) go back to the host, and the camera rises
// where the closest is nearer than its clearance: flown through a hill, it showed the void below the map. A slot is
// read FC_SLOTS evaluations after its copy, when the GPU has long been past it.
int dbg_freecam_flying(void);                          // debug_tools.c
void dbg_freecam_ground(const float *dist, int n);
#define FC_N 9
#define FC_SLOTS 4
static VkBuffer fc_buf;
static VmaAllocation fc_mem;
static float *fc_map;
static uint32_t fc_next;
static uint8_t fc_have[FC_SLOTS];

static void fcam_probe(VkCommandBuffer cb, const struct Size *e, uint32_t w, uint32_t h) {
    if (!dbg_freecam_flying() || !e->dist_ok) {   // dist_ok: mv_pass wrote this frame's distances (Ocarina's frame)
        memset(fc_have, 0, sizeof fc_have);
        return;
    }
    const VkCtx *c = vk_ctx();
    if (!fc_buf) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = FC_SLOTS * FC_N * sizeof(float),
                                 .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(c->vma, &bi, &ai, &fc_buf, &fc_mem, &info) != VK_SUCCESS) {
            fc_buf = VK_NULL_HANDLE;
            return;
        }
        fc_map = info.pMappedData;
    }
    uint32_t s = fc_next++ % FC_SLOTS;
    if (fc_have[s]) {
        vmaInvalidateAllocation(c->vma, fc_mem, s * FC_N * sizeof(float), FC_N * sizeof(float));
        dbg_freecam_ground(fc_map + s * FC_N, FC_N);
    }
    // the replay's viewports have negative heights: row 0 is the top of the view
    static const float fx[3] = {0.2f, 0.5f, 0.8f}, fy[3] = {0.6f, 0.78f, 0.96f};
    VkBufferImageCopy r[FC_N];
    for (int i = 0; i < FC_N; i++)
        r[i] = (VkBufferImageCopy){.bufferOffset = (s * FC_N + i) * sizeof(float),
                                   .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                   .imageOffset = {(int32_t)(fx[i % 3] * (w - 1)), (int32_t)(fy[i / 3] * (h - 1)), 0},
                                   .imageExtent = {1, 1, 1}};
    vkCmdCopyImageToBuffer(cb, e->dist[e->dist_cur ^ 1], VK_IMAGE_LAYOUT_GENERAL, fc_buf, FC_N, r);
    fc_have[s] = 1;
}

void vk_dlaa_release(void) {
    camera_history_enable(0);
    vk_dlaa_size_shutdown();
    vk_dlaa_hud_shutdown();
    release_post();
#ifndef _WIN32
    const VkCtx *c = vk_ctx();
    if (!c) return;
    if (FG.feat) N.release(FG.feat);   // frame generation's feature and parameters (vk_ngxfg_evaluate), before NGX goes
    if (FG.p) N.destroy_params(FG.p);
    memset(&FG, 0, sizeof FG);
    if (D.params) N.destroy_params(D.params);
    if (N.caps_map) N.destroy_params(N.caps_map);
    for (int i = 0; i < SNAPS; i++) destroy_snap(&R.s[i]);
    for (unsigned i = 0; i < 2; ++i) free_alpha(&D.present_alpha[i], &D.present_alpha_mem[i]);
    if (fc_buf) vmaDestroyBuffer(c->vma, fc_buf, fc_mem), fc_buf = VK_NULL_HANDLE, fc_map = NULL;
    if (R.pipe) vkDestroyPipeline(c->device, R.pipe, NULL);
    if (R.layout) vkDestroyPipelineLayout(c->device, R.layout, NULL);
    if (R.dsl) vkDestroyDescriptorSetLayout(c->device, R.dsl, NULL);
    if (T.pool) vkDestroyQueryPool(c->device, T.pool, NULL);
    memset(&D, 0, sizeof D), D.mv_ok = -1, D.cfg_seen = ~0u;
    memset(&R, 0, sizeof R), R.ok = -1, R.knob = -1;
    memset(&T, 0, sizeof T), T.on = -1;
    memset(&F, 0, sizeof F);
    vk_dlaa_swap_img = NULL, vk_dlaa_swap_view = VK_NULL_HANDLE, vk_dlaa_swap_output = VK_NULL_HANDLE;
    if (!N.device) return;   // NGX never came up on this device: DLAA off, FSR 3.1, or its init failed
    unsigned left = 0;
    vk_queue_lock();   // NGX may wait for the device itself
    NgxResult r = N.shutdown1(N.device, &left);
    vk_queue_unlock();
    rt_log("dlaa: NGX shut down (%08x, %u devices left)", (unsigned)r, left);
    N.device = NULL, N.caps_map = NULL, N.ngx_up = 0, N.available = 0;
#else
    // Window teardown has settled the Streamline tags before the allocator goes away.
    if (vk_ctx())
        for (unsigned i = 0; i < 2; ++i) free_alpha(&D.present_alpha[i], &D.present_alpha_mem[i]);
#endif
}

// ---------------------------------------------------------------- jitter
// The scene's draws (every pass drawing with the scene depth, until the DLAA point) have their viewport moved by a
// sub-pixel Halton(2,3) offset, which shifts what they rasterise as a jittered projection would; full-screen passes
// that shade by window position are unaffected. Jitter only follows a frame DLSS ran on: a jittered frame that DLSS
// does not resolve shakes on screen, so a missed DLAA point stops the jitter for a second.
static float halton(uint32_t i, uint32_t b) {
    float f = 1, r = 0;
    for (; i; i /= b) f /= (float)b, r += f * (float)(i % b);
    return r;
}

// Whether a draw takes the jitter, before DLSS ran this frame. The deferred frame: every draw into the scene depth.
// Ocarina's frame: every draw that projects through the camera (it binds the scene context at c3, vk_dlaa_draw) into a
// D32F depth of the scene's size, whichever image that is. Ocarina draws its depth pre-pass into the scene depth, copies
// that depth into a second image of the same size, and then shades the same geometry into the light target against
// the copy with the depth test EQUAL. Jittering only the draws into the first image left the shading draws unjittered
// against jittered depth: EQUAL failed on nearly every pixel, the light target kept its clear colour, and the ground
// and characters went white while the sky stayed right. Its full-screen passes drawn with a scene depth
// attached (the bloom composite DLSS reads) do not take the jitter either: they read the jittered scene through their
// own texture coordinates, and moving them too would shift the picture DLSS gets by a second offset.
static int jitter_draw(const VkExec *ex) {
    const VkExecTarget *z = &ex->depth;
    if (!D.jit_ok || D.ended || !z->image) return 0;
    if (D.cam_size != OCAM_BLOCK) return z->image == D.jdepth;
    return z->format == VK_FORMAT_D32_SFLOAT && z->width == D.jw && z->height == D.jh &&
           ex->ubo[0][0].size == OCAM_BLOCK && ex->ubo[0][0].addr;
}

void vk_dlaa_viewport(VkExec *ex, VkViewport *v) {
    ex->vp_jittered = jitter_draw(ex);
    if (!ex->vp_jittered) return;
    v->x += D.jx, v->y += D.jy;
    D.ux = D.jx, D.uy = D.jy, D.jit_used = 1;
}

void vk_dlaa_jitter_check(VkExec *ex) {
    int j = jitter_draw(ex);
    if (j != ex->vp_jittered) ex->dirty |= VK_EXEC_DIRTY_VIEWPORT;
    const VkExecTarget *z = &ex->depth;   // the draws into a scene-size depth, with the jitter or without
    if (D.jit_ok && !D.ended && z->image && z->format == VK_FORMAT_D32_SFLOAT && z->width == D.jw && z->height == D.jh) {
        if (j) D.jdraws++;
        else D.jfixed++;
    }
}

static int scene_color(VkFormat f);

static void dlaa_draw_locked(VkExec *ex);
void vk_dlaa_draw(VkExec *ex) {
    if (!dl_active() || D.ended) return;
    static PlatMutex lock = PLAT_MUTEX_INIT;   // the replay workers' scene draws count at once (vk_exec.c chunks)
    int lk = vk_exec_shared();
    if (lk) plat_mutex_lock(&lock);
    dlaa_draw_locked(ex);
    if (lk) plat_mutex_unlock(&lock);
}

static void dlaa_draw_locked(VkExec *ex) {
    static int survey = -1;   // NX2NAVI_DLAA_SURVEY=1 (test): what is drawn into the armed light target
    if (survey < 0) survey = rt_knob("DLAA_SURVEY", "0");
    static uint64_t dfn[16];   // scene draws writing depth, by depth test (reversed depth: GREATER/GEQUAL)
    if (survey && ex->depth.image && ex->depth.image == D.jdepth && ex->pipe.ds.dt && ex->pipe.ds.dw)
        dfn[ex->pipe.ds.df & 15]++;
    if (survey && D.armed && ex->color[0].image == D.color.image) {
        static uint64_t n[2][2][2], arms, frames, last_frame;   // [blend][depth write][depth test]
        const VkPipeState *s = &ex->pipe;
        n[s->color.blend_en & 1][s->ds.dw][s->ds.dt]++;
        uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (f != last_frame) last_frame = f, frames++;
        static uint64_t ps;
        if (ex->pass_serial != ps) ps = ex->pass_serial, arms++;
        if (frames % 300 == 0) {
            static uint64_t logged;
            if (logged != frames) {
                logged = frames;
                rt_log("dlaa survey: scene draws writing depth by depth test (NVN func 1-8: never less equal lequal "
                       "greater notequal gequal always): %llu %llu %llu %llu %llu %llu %llu %llu",
                       (unsigned long long)dfn[1], (unsigned long long)dfn[2], (unsigned long long)dfn[3],
                       (unsigned long long)dfn[4], (unsigned long long)dfn[5], (unsigned long long)dfn[6],
                       (unsigned long long)dfn[7], (unsigned long long)dfn[8]);
                rt_log("dlaa survey: %llu frames, %.1f passes a frame into the light target after arming; draws a frame "
                       "(blend/depth write/depth test): 000 %.1f 001 %.1f 010 %.1f 011 %.1f 100 %.1f 101 %.1f 110 %.1f 111 %.1f",
                       (unsigned long long)frames, (double)arms / frames, (double)n[0][0][0] / frames, (double)n[0][0][1] / frames,
                       (double)n[0][1][0] / frames, (double)n[0][1][1] / frames, (double)n[1][0][0] / frames,
                       (double)n[1][0][1] / frames, (double)n[1][1][0] / frames, (double)n[1][1][1] / frames);
            }
        }
    }
    // The scene camera of a draw into the scene depth: the deferred engine binds a 0x940-byte block at c4 with the
    // view-projection in rows 3-6 (0x30); Ocarina's opaque scene draws bind a 0x3500-byte scene context at c3, whose
    // rows 0-2 and 7-10 their vertex shaders use as the view and the projection (ocam_vp). Its full-screen passes bind
    // a 0x200-byte block at c3 instead, which is not the camera. Without a camera there is no camera motion and no
    // jitter (next_jitter), so DLSS runs on each frame alone without shaking the picture.
    const p_bind_buf *b = &ex->ubo[0][1];   // c4: the replay keeps user banks from c3 on
    const VkExecTarget *z = &ex->depth;
    if (!z->image || z->width < DLAA_SCENE_MIN_W || z->format != VK_FORMAT_D32_SFLOAT) return;
    if (!F.zdraws && !F.zother) F.seq0 = vk_exec_submit_seq;
    if (D.jdepth && z->image != D.jdepth) F.zother++, F.zother_w = z->width, F.zother_h = z->height;
    else F.zdraws++;
    // The prior depth is still useful to jitter and diagnose the draw. Votes
    // for a current camera belong to the depth this evaluation will read.
    const VkExecTarget *armed = &D.depth;
    if (dlaa_targets_armed() && (z->image != armed->image || z->format != armed->format ||
        z->level != armed->level || z->base_layer != armed->base_layer || z->layers != armed->layers))
        return;
    if (ex->ubo[0][0].size == OCAM_BLOCK && ex->ubo[0][0].addr) {
        F.ocam++;
        if (!ex->color[0].image || !scene_color(ex->color[0].format)) F.ocam_off++, F.ocam_off_fmt = ex->color[0].format;
    } else F.c3_other++, F.c3_size = ex->ubo[0][0].addr ? ex->ubo[0][0].size : 0;
    if (b->size == CAM_BLOCK && b->addr) {
        dlaa_camera_depth_select(z);
        D.cam_addr = b->addr, D.cam_size = CAM_BLOCK, D.cam_vp = 0x30;
    } else if (ex->ubo[0][0].size == OCAM_BLOCK && ex->ubo[0][0].addr && ex->color[0].image &&
             scene_color(ex->color[0].format)) {   // Ocarina: the scene context most light-target draws use this frame
        dlaa_camera_depth_select(z);
        uint64_t a = ex->ubo[0][0].addr;
        int i = 0, best = 0;
        while (i < D.nviews && D.views[i].addr != a) i++;
        if (i == D.nviews && D.nviews < 8) D.views[D.nviews].addr = a, D.views[D.nviews].n = 0, D.views[D.nviews++].got = 0;
        if (i < D.nviews) {
            D.views[i].n++, D.views[i].seq = vk_exec_submit_seq;
            // The camera rows as this draw reads them, from its own submission's copy: DLSS runs in a later submission
            // on some frames, whose copy may not hold the block (mv_pass, dlaa_cam_keep).
            if (!D.views[i].got) D.views[i].got = vk_usnap_read(a, OCAM_BYTES, D.views[i].cam);
        }
        for (int k = 1; k < D.nviews; k++)
            if (D.views[k].n > D.views[best].n) best = k;
        D.cam_addr = D.views[best].addr, D.cam_size = OCAM_BLOCK, D.cam_vp = 0;
    }
}

// NX2NAVI_DLAA_JITTER=0 (test): no jitter. Read once.
static int jitter_on(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("DLAA_JITTER", "1");
    return on;
}

// The jitter moves on right after DLAA ran, not at the present: the replay can record the next frame's scene before
// the present of this one is handled, and a step taken there landed in the middle of a frame (worse after an off/on
// switch, which shifted that order). Jitter needs a frame DLSS ran on; a frame without the DLAA point stops it.
// The sequence is as long as NVIDIA's DLSS guide asks for the upscaling ratio, 8 x (output / scene)^2 phases (18 in
// Quality, 24 in Balanced, 32 in Performance, 72 in Ultra Performance), and 16 at least, the length DLAA has used from the start.
static void next_jitter(uint32_t w, uint32_t ow) {
    int nj = !jitter_on();
    double r = w ? (double)ow / w : 1.0;
    uint32_t phases = (uint32_t)ceil(8.0 * r * r - 1e-6);
    if (phases < 16) phases = 16;
    if (phases > 72) phases = 72;
    uint32_t i = D.jframe++ % phases + 1;
    D.jx = halton(i, 2) - 0.5f, D.jy = halton(i, 3) - 0.5f;
    // Jitter only where the camera is known (vk_dlaa_draw): the deferred engine's block or Ocarina's scene context. On
    // Ocarina's frame the draws that take it are the camera's into every scene-size depth (jitter_draw); taking only the
    // draws into DLSS's own depth image turned the picture white, and jitter was off there until that was found.
    D.jit_ok = !D.hold && !nj && (D.cam_size == CAM_BLOCK || D.cam_size == OCAM_BLOCK);
}

// The DLSS model (NX2NAVI_DLSS_PRESET, live): default takes NVIDIA's model per mode (evaluate: K for DLAA, Quality and
// Balanced, M for Performance, L for Ultra Performance), 0 here. A letter a..m takes that model in every mode: k pins
// K, j is the other transformer model, which NVIDIA describes as slightly less ghosting at the cost of extra flicker, e
// the older CNN model, cheaper on slower RTX cards. The value is an NVSDK_NGX_DLSS_Hint_Render_Preset (1 A ... 13 M),
// read when a feature is created; a change recreates the features (preset_check).
static int preset_value(void) {
    const char *v = cfg_get("NX2NAVI_DLSS_PRESET");
    int c = v && v[0] && !v[1] ? v[0] | 0x20 : 0;
    return c >= 'a' && c <= 'm' ? c - 'a' + 1 : 0;
}

// How strongly DLSS favours the current frame on characters and where they just were (NX2NAVI_DLSS_CHAR_MASK, percent,
// live; 0 off). Where a character's vectors are the camera's only (dlaa_objmv = 0, or a draw the per-object pass
// skips), one that moves on its own drags a faint copy of itself through DLSS's history; the bias mask makes DLSS take
// less of the history there, at the cost of some of its anti-aliasing on those pixels. Ocarina's frame has no class
// buffer to find characters by: there the mask covers the pixels whose depth the camera's motion does not explain
// (dlaa_mv.comp, the depth check).
static float char_bias(void) {
    const char *v = cfg_get("NX2NAVI_DLSS_CHAR_MASK");
    long n = v && *v ? strtol(v, NULL, 10) : 65;
    return n <= 0 ? 0.0f : n >= 100 ? 1.0f : (float)n / 100.0f;
}

// The quality setting DLSS is created with, from the scene's share of the output width.
static int perf_quality(uint32_t w, uint32_t ow) {
    if (w >= ow) return NGX_PERFQ_DLAA;
    double k = (double)w / ow;
    return k >= 0.66 ? 2 : k >= 0.57 ? 1 : k >= 0.49 ? 0 : 3;
}

// Every 300 frames DLSS ran, the history resets and camera gaps of those frames, while there were any (at most 400 lines a
// run): the measure for dlaa_cam_keep at one spot of the game.
static void window_line(void) {
    enum { N_ = 11 };
    static uint64_t last[N_];
    static int lines;
    if (D.evals % 300) return;
    uint64_t now[N_] = {D.why[0], D.why[1], D.why[2], D.mv_none, D.cam_none_draw, D.cam_none_snap, D.cam_kept,
                        D.cam_at_draw, D.frames_missed, D.seconds, D.target_kept}, d[N_], any = 0;
    for (int i = 0; i < N_; i++) d[i] = now[i] - last[i], last[i] = now[i];
    for (int i = 0; i < N_; i++) any |= d[i];
    if (!any || lines >= 400) return;
    lines++;
    rt_log("dlaa: frames %llu-%llu: resets %llu size, %llu camera, %llu target (%llu target changes with the history "
           "kept); %llu without a camera (%llu without a "
           "scene-context draw, %llu not in DLSS's submission), %llu kept with zero motion; %llu cameras read at the draw; "
           "%llu frames without the DLAA point, %llu with a second scene after DLSS ran",
           (unsigned long long)(D.evals - 299), (unsigned long long)D.evals, (unsigned long long)d[0],
           (unsigned long long)d[1], (unsigned long long)d[2], (unsigned long long)d[10], (unsigned long long)d[3],
           (unsigned long long)d[4],
           (unsigned long long)d[5], (unsigned long long)d[6], (unsigned long long)d[7], (unsigned long long)d[8],
           (unsigned long long)d[9]);
}

// dst: the target DLSS's picture goes to, NULL for the light target itself (DLAA in place).
static void hudless(VkExec *ex, VkImage source, VkImage target, const VkTexImage *texture,
                    struct Size *reconstruction, const VkTextureColorBlitSource *compute_output,
                    uint32_t w, uint32_t h);

// Another light target than last frame's. The game draws its scene into one of a few pooled targets of the same size
// and moves to another whenever its pass list changes, in the middle of play with the camera and the scene carrying
// on: 35 times in a walk of a minute through Kokiri Forest, three of them within eight frames on the ladder below
// the tree house.
// The history is the upscaler's own and not in the target, so where this frame's motion vectors are valid (cont: the
// camera of this frame and of the last one, mv_pass 1) the history stays. A new history at each change showed as a few
// soft, grainy frames every time, a flicker over the whole picture. Without valid vectors, and with
// NX2NAVI_DLAA_TARGET_KEEP=0 (test), a new history as before. The first few are logged with the new target's size, to
// tell a target the game made anew from the ones it alternates between.
static void target_changed(const VkExecTarget *t, int cont) {
    static int keep = -1;
    if (keep < 0) keep = rt_knob("DLAA_TARGET_KEEP", "1");
    int kept = keep && cont && D.last_color;
    if (D.last_color && D.target_logged < 16)
        D.target_logged++, rt_log("dlaa: frame %llu: the light target changed from %p to %p (%ux%u format %d): %s",
                                  (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1,
                                  (void *)D.last_color, (void *)t->image, t->width, t->height, (int)t->format,
                                  kept ? "the history stays (the camera carries on)" : "new history");
    if (kept) D.target_kept++;
    else D.reset = 1, D.why[2]++;
    D.last_color = t->image;
}

// DLSS's own video memory, the features and their history (NGX's DLSS statistics callback with the capability
// parameters), for the log's VRAM breakdown (vk_dlaa_vram); -1 until known. Asked on the replay thread, which makes every
// other NGX call, every 600 evaluations.
static volatile int64_t dlss_vram = -1;
int64_t vk_dlaa_vram(void) { return __atomic_load_n(&dlss_vram, __ATOMIC_RELAXED); }

static void query_vram(void) {
    void *cb = NULL;
    unsigned long long n = 0;
    if (fsr3() || !N.capsp || !p_getp(N.capsp, "DLSSGetStatsCallback", &cb) || !cb) return;
    if (NGX_OK(((NgxResult (*)(NgxParam *))cb)(N.capsp)) && p_getull(N.capsp, "SizeInBytes", &n))
        __atomic_store_n(&dlss_vram, (int64_t)n, __ATOMIC_RELAXED);
}

// Cache age selects inactive groups for retirement. Their actual replay and
// presentation users determine when the images and feature can be destroyed.
static void stale_sizes(const struct Size *cur) {
    if (!idle_free()) return;
    for (int i = 0; i < DLSS_SIZES; i++) {
        struct Size *s = &D.sizes[i];
        if (s == cur || !s->generation || __atomic_load_n(&s->retired, __ATOMIC_ACQUIRE) ||
            D.evals < s->used + SIZE_STALE) continue;
        rt_log("dlaa: size %ux%u -> %ux%u went unused for %d frames and was marked for retirement", s->w, s->h, s->ow, s->oh,
               SIZE_STALE);
        size_mark_retired(s);
    }
}

typedef struct {
    uint64_t required_scene;
    int retired;
} NgxSrRetirement;

// This runs during the existing recorder park, after older sealed submissions
// are flushed. Only the final native signal proves every older SDK use retired.
static int ngx_sr_wait_retirement(void *context) {
    NgxSrRetirement *retirement = context;
    uint64_t scene = retirement->required_scene;
    if (scene > vk_sync_handed_fence()) return 1;   // terminal: no final handoff after flush
    while (!vk_sync_stopping() && !vk_device_is_lost()) {
        if (!vk_sync_fence_wait(scene, UINT64_C(100000000))) continue;
        retirement->retired = 1;
        return 1;
    }
    return 0;
}

// Keep the established compatibility order between larger inputs, presets and
// previously freed features. Prove every selected SDK use before releasing any.
static int release_kept(VkExec *ex, const struct Size *making, int all) {
    struct Size *release[DLSS_SIZES];
    int n = 0;
    NgxSrRetirement retirement = {0};
    for (int i = 0; i < DLSS_SIZES; i++) {
        struct Size *s = &D.sizes[i];
        if (s == making || !s->sr.handle) continue;
        if (!all && (uint64_t)s->w * s->h <= (uint64_t)making->w * making->h) continue;
        uint64_t used = s->sr.scene;
        if (!used || used >= ex->fence) {
            vk_exec_required_temporal_unavailable(ex, "an incompatible DLSS SR feature still belongs to the current recording");
            return 0;
        }
        if (used > retirement.required_scene) retirement.required_scene = used;
        release[n++] = s;
    }
    if (!n) return 1;
    if (retirement.required_scene > vk_sync_fence_done()) {
        VkSyncProgressResult result = vk_exec_temporal_prepare(ex, ngx_sr_wait_retirement, &retirement);
        if (result == VK_SYNC_PROGRESS_STOPPED) {
            vk_exec_cancel_temporal_preparation();
            return 0;
        }
        if (result != VK_SYNC_PROGRESS_COMPLETE || !retirement.retired) {
            vk_exec_required_temporal_unavailable(ex, "could not prove the last native scene of an incompatible DLSS SR feature");
            return 0;
        }
    }
    if (vk_sync_stopping() || vk_device_is_lost()) {
        vk_exec_cancel_temporal_preparation();
        return 0;
    }
    for (int i = 0; i < n; ++i) {
        struct Size *s = release[i];
        ngx_sr_release(&s->sr);
        if (D.last == s) D.last = NULL;
    }
    rt_log("dlaa: %d DLSS feature(s) released before the one for %ux%u is made (%s); made again when used", n,
           making->w, making->h, all ? "after a release" : "larger inputs");
    return 1;
}

// The scene's last column and row after a jitter to the left or up. The jitter moves the scene's viewport by under half a
// pixel (vk_dlaa_viewport), which keeps every pixel centre inside it, yet on an RX 7900 XT (AMD's Windows driver) the
// draws of a viewport moved left left the last column out, and of one moved up the last row: the driver evidently limits
// drawing to the whole pixels inside the viewport. In a frame dump (NX2NAVI_DUMP, Hyrule Field, 3416x1440, FSR 4 Native
// AA) the depth's last row kept its clear value through every scene pass and the light target held the fog's colour
// there, which the upscaler blended into a grey line along the bottom and right edges of the picture that flickered with
// the jitter; with NX2NAVI_DLAA_JITTER=0, and without DLSS or FSR, there was no line. Before DLSS or FSR reads the scene,
// such a column or row of the light target and of the depth (which the motion vectors are made from) is filled from the
// one next to it. NX2NAVI_DLAA_EDGE_FILL=0 (test): not filled.
static RcSourceRange dlaa_guest_target(const VkExecTarget *target) {
    if (!target->tex || !target->img || target->img->image != target->image) return (RcSourceRange){0};
    return (RcSourceRange){1, target->img->addr, target->tex->L.size};
}

void vk_dlaa_scene_effects_cancel_unselected(VkDlaaSceneEffects *effects, uint32_t selected) {
    if (effects)
        for (unsigned i=0; i<2; i++)
            if (!(selected & (1u<<i))) resource_effects_image_role_finish(&effects->border[i], 0);
}

static unsigned dlaa_edge_axes(const VkExecTarget *t, float jx, float jy) {
    static int on = -1;
    if (on < 0) on = rt_knob("DLAA_EDGE_FILL", "1");
    if (!on || t->width < 2 || t->height < 2) return 0;
    return (jx < 0 ? 1u : 0u) | (jy < 0 ? 2u : 0u);
}

static uint32_t dlaa_edge_targets(const VkExecTarget *t, const VkExecTarget *z) {
    return (t->image ? 1u : 0u) | (z->image && z->width == t->width && z->height == t->height ? 2u : 0u);
}

static VkImageCopy dlaa_edge_copy(const VkExecTarget *g, unsigned role, unsigned axis) {
    uint32_t w = g->width, h = g->height;
    VkImageSubresourceLayers plane = {role ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT,
                                      g->level, g->base_layer, 1};
    VkOffset3D from = axis ? (VkOffset3D){0, (int32_t)h - 2, 0} : (VkOffset3D){(int32_t)w - 2, 0, 0};
    VkOffset3D to = axis ? (VkOffset3D){0, (int32_t)h - 1, 0} : (VkOffset3D){(int32_t)w - 1, 0, 0};
    return (VkImageCopy){plane, from, plane, to, axis ? (VkExtent3D){w, 1, 1} : (VkExtent3D){1, h, 1}};
}

static uint32_t dlaa_edge_regions(const VkExecTarget *target, unsigned role, unsigned axes, VkImageCopy out[2]) {
    uint32_t n = 0;
    for (unsigned i = 0; i < 2; i++)
        if (axes & (1u << i)) out[n++] = dlaa_edge_copy(target, role, i);
    return n;
}

static void dlaa_edge_completed(void *opaque) {
    VkTextureProjectedTransfer *target = opaque;
    vk_texture_projected_transfer_cancel(&target);
}

static void dlaa_edge_canceled(void *opaque, int may_have_executed) {
    (void)may_have_executed;
    dlaa_edge_completed(opaque);
}

static unsigned edge_fill(VkCommandBuffer cb, const VkExecTarget *t, const VkExecTarget *z, float jx, float jy,
                      VkDlaaSceneEffects *effects) {
    const unsigned axes = dlaa_edge_axes(t, jx, jy);
    uint32_t targets = axes ? dlaa_edge_targets(t, z) : 0;
    // Settle skipped roles before a real copy can bind another claim for the same normalized range.
    vk_dlaa_scene_effects_cancel_unselected(effects, targets);
    if (!targets) return 0;
    const VkExecTarget *scene[2] = {t, z};
    const VkImageCopy *selected[2] = {0};
    uint32_t next[2] = {0};
    for (unsigned i = 0; effects && i < 2; i++) {
        VkTextureProjectedTransfer *edge = effects->edge[i];
        if (!(targets & (1u << i)) || !edge) continue;
        VkImageCopy current[2];
        uint32_t count;
        const uint32_t actual = dlaa_edge_regions(scene[i], i, axes, current);
        const VkImageCopy *retained = vk_texture_projected_self_copy_regions(edge, &count);
        if (count != actual || vk_texture_projected_transfer_image(edge) != scene[i]->image ||
            memcmp(retained, current, count * sizeof *retained) || !vk_texture_projected_transfer_begin(edge)) {
            targets &= ~(1u << i);
            resource_effects_image_role_finish(&effects->border[i], 0);
            continue;
        }
        selected[i] = retained;
    }
    if (!targets) return 0;
    static int told;
    if (!told++) rt_log("dlaa: the scene's last column or row after a jitter to the left or up is filled from the one before it");
    unsigned recorded = 0;
    // The final row copies the already filled column at its corner. Keep the transfer barrier between axes.
    for (unsigned axis = 0; axis < 2; axis++) {
        if (!(axes & (1u << axis))) continue;
        for (unsigned i = 0; i < 2; i++) {
            if (!(targets & (1u << i))) continue;
            const VkExecTarget *g = scene[i];
            recorded++;
            if (selected[i]) {
                vkCmdCopyImage(cb, g->image, VK_IMAGE_LAYOUT_GENERAL, g->image, VK_IMAGE_LAYOUT_GENERAL,
                               1, &selected[i][next[i]++]);
                continue;
            }
            const VkImageCopy region = dlaa_edge_copy(g, i, axis);
            RcSourceRange range = effects ? effects->ranges[i] : dlaa_guest_target(g);
            if (range.size) vk_texture_image_write_prepare(g->image, range.offset, range.size);
            vkCmdCopyImage(cb, g->image, VK_IMAGE_LAYOUT_GENERAL, g->image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
            if (effects) resource_effects_image_role_finish(&effects->border[i], 1);
            if (range.size) {
                resource_image_written((uint64_t)(uintptr_t)g->image, range.offset);
                if (effects && effects->owners[i].hold.id)
                    resource_image_owned((uint64_t)(uintptr_t)g->image, range.offset, effects->owners[i].hold);
                vk_texture_storage_written(g->image, range.offset);
            }
        }
        full_barrier(cb);
    }
    // Both retained entries were locked before the actual copies, in color/depth order.
    for (int i = 1; i >= 0; i--) if (selected[i]) {
        vk_texture_projected_transfer_written(effects->edge[i]);
        resource_effects_image_role_finish(&effects->border[i], 1);
    }
    for (unsigned i = 0; i < 2; i++) if (selected[i]) {
        vk_completion_add_owned(vk_exec_recording_fence(), dlaa_edge_completed, dlaa_edge_canceled, effects->edge[i]);
        effects->edge[i] = NULL;
    }
    return recorded;
}

// Set when the scene the upscaler runs on has another size than the last one; vk_dlaa_resized_take reads it.
static volatile int32_t size_changed;

// Direct NGX SR has no vetted pre-create internal-peak query. Keep the
// supported SDK path explicit and leave the seven app images on native admission.
static int ngx_sr_create_opaque(VkExec *ex, struct Size *e, NgxParam *p,
                                const VkExecTarget *t, const VkExecTarget *z) {
    if (e->sr.state != NGX_SR_EMPTY) rt_fatal("dlaa: DLSS SR creation already owned by this size");
    static int reported;
    if (!reported) {
        reported = 1;
        rt_log("dlaa: direct NGX SR uses opaque SDK allocation compatibility (no proven pre-create capacity bound)");
    }
    const uint32_t w = e->w, h = e->h, ow = e->ow, oh = e->oh;
    p_setu(p, "CreationNodeMask", 1);
    p_setu(p, "VisibilityNodeMask", 1);
    p_setu(p, "Width", w);
    p_setu(p, "Height", h);
    p_setu(p, "OutWidth", ow);
    p_setu(p, "OutHeight", oh);
    p_seti(p, "PerfQualityValue", perf_quality(w, ow));
    // IsHDR | MVLowRes | AutoExposure: linear scene light, motion vectors at the render size, no exposure texture
    p_seti(p, "DLSS.Feature.Create.Flags", (1 << 0) | (1 << 1) | (1 << 6));
    p_seti(p, "DLSS.Enable.Output.Subrects", 0);
    // The model per mode (NVSDK_NGX_DLSS_Hint_Render_Preset: 1 A ... 11 K, 12 L, 13 M), as NVIDIA recommends for
    // DLSS 4.5 (310.x): K (transformer) for DLAA, Quality and Balanced, M for Performance, L for Ultra Performance.
    // M and L cost more on cards without FP8 (RTX 20/30). dlss_preset (D.preset, preset_check) takes one model for
    // every mode instead.
    int pk = D.preset ? D.preset : 11, pm = D.preset ? D.preset : 13, pl = D.preset ? D.preset : 12;
    p_setu(p, "DLSS.Hint.Render.Preset.DLAA", (unsigned)pk);
    p_setu(p, "DLSS.Hint.Render.Preset.UltraQuality", (unsigned)pk);
    p_setu(p, "DLSS.Hint.Render.Preset.Quality", (unsigned)pk);
    p_setu(p, "DLSS.Hint.Render.Preset.Balanced", (unsigned)pk);
    p_setu(p, "DLSS.Hint.Render.Preset.Performance", (unsigned)pm);
    p_setu(p, "DLSS.Hint.Render.Preset.UltraPerformance", (unsigned)pl);
    rec_drain();   // NGX records directly into this actual scene command buffer.
    if (vk_sync_stopping() || vk_device_is_lost()) {
        vk_exec_cancel_temporal_preparation();
        return 0;
    }
    ngx_sr_scene_use(ex, e);
    e->sr.state = NGX_SR_FAILED;
    uint64_t t0 = perf_mark_begin();
    NgxResult result = N.create(ex->cb, NGX_FEATURE_SUPERSAMPLING, p, &e->sr.handle);
    perf_mark_end(PERF_M_NGX, t0, 0);
    ex->recorded++;    // SDK entry may have emitted commands even on failure.
    vk_exec_draw_new_cb();
    if (!NGX_OK(result) || !e->sr.handle) {
        size_mark_retired(e);
        return dl_fail("DLSS feature creation", result);
    }
    e->sr.state = NGX_SR_OPAQUE_READY;
    feat_released = 0;
    D.reset = 1;
    int q = perf_quality(w, ow), pr = q == 0 ? pm : q == 3 ? pl : pk;
    rt_log("dlaa: DLSS feature created at %ux%u -> %ux%u, quality value %d, preset %c (scene format %d, depth format "
           "%d)", w, h, ow, oh, q, 'A' + pr - 1, (int)t->format, (int)z->format);
    return 1;
}

static void dlaa_compute_blit_written(const VkTextureColorBlitSource *output, const VkImageBlit *blit) {
    // Seal the actual guest copy before HUD preparation may hand off a native prefix.
    resource_effects_emitted();
    resource_image_written((uint64_t)(uintptr_t)output->image, output->address);
    vk_texture_storage_blit_written(output, blit);
}

static int evaluate(VkExec *ex, const VkExecTarget *dst, VkDlaaSceneEffects *effects, int early,
                    const VkTextureColorBlitSource *compute_output) {
    const VkExecTarget *t = &D.color, *z = &D.depth;
    if (!dlaa_targets_armed()) return 0;
    const DlaaTargetOwner *destination = !dst ? frame_targets[DLAA_TARGET_COLOR] : NULL;
    if (!dst) dst = t;
    // Writing a pending seed needs its own vendor output claim and physical publication.
    if (destination && destination->seed && vk_texture_alias_target_replacement(destination->seed)) return 0;
    uint32_t w = t->width, h = t->height, ow = dst->width, oh = dst->height;
    if (early) wide_picture(&ow, &oh);
    const int fsr = use_fsr();
    if (!fsr && !N.ngx_up) return 0;
    const int replace_preset = D.preset_dirty;
    struct Size *e = size_for(ex, w, h, ow, oh);
    if (!e) return 0;

    const int cold_sr = !fsr && e->sr.state == NGX_SR_EMPTY;
    if (!fsr && !D.params && !NGX_OK(N.alloc_params(&D.params))) return dl_fail("parameter allocation", 0);
    NgxParam *p = D.params;
    if (cold_sr && (replace_preset || idle_free() != 2) &&
        !release_kept(ex, e, replace_preset || feat_released)) return 0;
    if (!fsr && !cold_sr && e->sr.state != NGX_SR_OPAQUE_READY)
        rt_fatal("dlaa: DLSS selected an unusable SR feature");

    VkCommandBuffer cb = ex->cb;
    vk_exec_barrier(ex);   // what drew the scene is done (no pass is open here)
    if (cold_sr && !ngx_sr_create_opaque(ex, e, p, t, z)) return 0;
    D.early = early;
    if (early) hud_sources.open = NULL;
    if (early != D.previous_early) D.reset = 1;
    D.previous_early = early;
    RESOURCE_IMAGE_SCOPE(image_writes, ex->fence);
    vk_gpu_gap_note(VK_GAP_UPSCALER, (uint64_t)ow * oh);
    size_record_begin(ex, e);
    if (e != D.last) {   // another size's history is from another moment
        D.reset = 1, D.why[0]++, e->chars_ok = 0;
        __atomic_store_n(&size_changed, 1, __ATOMIC_RELEASE);
    }
    dlaa_target_record(ex, frame_targets[DLAA_TARGET_COLOR], 0);
    dlaa_target_record(ex, frame_targets[DLAA_TARGET_DEPTH], 0);
    if (D.ids_ok && dlaa_target_valid(frame_targets[DLAA_TARGET_IDS]))
        dlaa_target_record(ex, frame_targets[DLAA_TARGET_IDS], 0);
    VkImageView zv = vk_exec_target_view(z->img, z->format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, z->level,
                                         z->base_layer, 1, 0);
    float jx = D.jit_used ? D.ux : 0.0f, jy = D.jit_used ? D.uy : 0.0f;
    ex->recorded += edge_fill(cb, t, z, jx, jy, effects);   // before the motion vectors read the depth
    static int no_mv = -1;
    if (no_mv < 0) no_mv = !rt_knob("DLAA_MV", "1");   // NX2NAVI_DLAA_MV=0: zero motion (test)
    VkDlaaCamera camera = {0};
    int camera_ok = 0, masked = 0;
    int mv = no_mv ? 0 : mv_pass(ex, e, zv, jx, jy, char_bias(), &masked, &camera, &camera_ok);
    if (mv == MV_PASS_ABORTED) return 0;
    if (mv != 1) {
        VkClearColorValue zero = {{0, 0, 0, 0}};
        VkImageSubresourceRange all = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, e->mv, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &all);
        full_barrier(cb);
        // no camera: the history cannot be moved along (2: kept as it is for a short gap, mv_pass dlaa_cam_keep)
        if (!no_mv && mv == 0) D.reset = 1, D.why[1]++;
    }
    VkImageView cv = vk_exec_target_view(t->img, t->format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, t->level,
                                         t->base_layer, 1, 0);
    struct Snap *mk = D.mv_samp ? mask_pass(cb, cv, w, h) : NULL;
    int timed = T.on > 0 && T.used[T.slot];   // mv_pass began this frame's slot
    if (timed) times_mark(cb, T_DL0);
    uint64_t c0 = timed ? kern_now_ns() : 0;
    rec_drain();   // NGX records into the command buffer itself: the recorder thread must have made every call so far
    if (timed) T.cpu_drain += kern_now_ns() - c0;
    if (mv == 1) fcam_probe(cb, e, w, h);   // mv_pass's distances, behind its barrier
    if (fsr) {   // FSR 3.1 (vk_ffx.cpp) in DLSS's place: the same images, then the same blit below
        if (t->level || t->base_layer || z->level || z->base_layer) return dl_fail("FSR 3.1 on a target that is not level 0, layer 0", 0);
        if (D.last_color != t->image) target_changed(t, mv == 1);
        uint64_t now = plat_now_ns();
        double dt = D.last_eval_ns ? (now - D.last_eval_ns) / 1e6 : 0;
        D.last_eval_ns = now;
        VkFfxImage fc = {t->image, t->format, w, h, 0}, fz = {z->image, z->format, w, h, 0},
                   fm = {e->mv, VK_FORMAT_R16G16_SFLOAT, w, h, 1}, fo = {e->out, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh, 1};
        VkFfxCamera fcam = camera_ok ? ffx_camera(&camera) : (VkFfxCamera){0};
        if (!vk_ffx_upscale(cb, &fc, &fz, &fm, &fo, jx, jy, D.reset, dt > 0 && dt < 100 ? (float)dt : 16.67f, &fcam))
            return dl_fail("FSR 3.1", 0);
        static int told;
        if (!told++) rt_log("dlaa: FSR 3.1 at %ux%u -> %ux%u (scene format %d, depth format %d)", w, h, ow, oh,
                            (int)t->format, (int)z->format);
        goto evaluated;
    }
    if (D.last_color != t->image) target_changed(t, mv == 1);
    NgxResourceVK rc = res_image(cv, t->image, t->format, w, h, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    NgxResourceVK ro = res_image(e->out_view, e->out, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    NgxResourceVK rz = res_image(zv, z->image, z->format, w, h, VK_IMAGE_ASPECT_DEPTH_BIT, 0);
    NgxResourceVK rm = res_image(e->mv_view, e->mv, VK_FORMAT_R16G16_SFLOAT, w, h, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    p_setp(p, "Color", &rc);
    p_setp(p, "Output", &ro);
    p_setp(p, "Depth", &rz);
    p_setp(p, "MotionVectors", &rm);
    p_setf(p, "Jitter.Offset.X", jx);
    p_setf(p, "Jitter.Offset.Y", jy);
    p_setf(p, "Sharpness", 0.0f);
    p_seti(p, "Reset", D.reset);
    // One bias mask: the reactive one (NX2NAVI_DLSS_REACTIVE, a test) where it ran, else the characters' (mv_pass), else
    // none. The parameter object is reused, so it is cleared explicitly.
    NgxResourceVK rk = mk ? res_image(mk->mask_view, mk->mask, VK_FORMAT_R8_UNORM, w, h, VK_IMAGE_ASPECT_COLOR_BIT, 0)
                          : res_image(e->mask_view, e->mask, VK_FORMAT_R8_UNORM, w, h, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    p_setp(p, "DLSS.Input.Bias.Current.Color.Mask", mk || masked ? (void *)&rk : NULL);
    p_setu(p, "DLSS.Input.Bias.Current.Color.Subrect.Base.X", 0);
    p_setu(p, "DLSS.Input.Bias.Current.Color.Subrect.Base.Y", 0);
    p_setf(p, "MV.Scale.X", 1.0f);
    p_setf(p, "MV.Scale.Y", 1.0f);
    p_setu(p, "DLSS.Render.Subrect.Dimensions.Width", w);
    p_setu(p, "DLSS.Render.Subrect.Dimensions.Height", h);
    p_setf(p, "DLSS.Pre.Exposure", 1.0f);
    p_setf(p, "DLSS.Exposure.Scale", 1.0f);
    // The time since the previous evaluation, which DLSS weighs its history by; 60 fps before the first and after a
    // gap (a pause, a load), where no history carries over anyway.
    uint64_t now = plat_now_ns();
    double dt = D.last_eval_ns ? (now - D.last_eval_ns) / 1e6 : 0;
    D.last_eval_ns = now;
    p_setf(p, "FrameTimeDeltaInMsec", dt > 0 && dt < 100 ? (float)dt : 16.67f);
    c0 = timed ? kern_now_ns() : 0;
    if (vk_sync_stopping() || vk_device_is_lost()) {
        vk_exec_cancel_temporal_preparation();
        return 0;
    }
    ngx_sr_scene_use(ex, e);
    NgxResult r = N.evaluate(cb, e->sr.handle, p, NULL);
    if (timed) T.cpu_ngx += kern_now_ns() - c0;
    if (!NGX_OK(r)) {
        e->sr.state = NGX_SR_FAILED;
        size_mark_retired(e);
        vk_exec_draw_new_cb();
        return dl_fail("DLSS evaluation", r);
    }
evaluated:
    if (timed) times_mark(cb, T_DL1);   // through the recorder, after NGX's own commands
    // Frame generation gets the same new start with this frame's inputs (vk_dlaa_fg_inputs): a frame DLSS starts a
    // new history on (a camera cut, another scene size or target) has no motion from the frame before that holds.
    D.fg_reset = D.reset;
    D.reset = 0;
    D.last = e, e->used = D.evals + 1;
    if (D.evals % 600 == 0) query_vram();
    stale_sizes(e);
    full_barrier(cb);
    vk_mvcheck_output(cb, e->out, ow, oh);   // NX2NAVI_MV_CHECK (test)
    // the result into the scene target (DLAA: the bloom, the post passes and the HUD read it and draw over it) or into
    // the scaling pass's target (the HUD draws over it)
    VkImageBlit b = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                     .srcOffsets = {{0, 0, 0}, {(int32_t)ow, (int32_t)oh, 1}},
                     .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, dst->level, dst->base_layer, 1},
                     .dstOffsets = {{0, 0, 0}, {(int32_t)dst->width, (int32_t)dst->height, 1}}};
    // NX2NAVI_DLAA_MVDEBUG=1 (test): the motion vectors in place of the picture, x in red and y in green, render pixels
    // as scene light (the post passes still run over them)
    static int mvdebug = -1;
    if (mvdebug < 0) mvdebug = rt_knob("DLAA_MVDEBUG", "0");
    static int maskdebug = -1;
    if (maskdebug < 0) maskdebug = rt_knob("DLAA_MASKDEBUG", "0");
    // (NX2NAVI_DLAA_MASKDEBUG=1 shows whichever bias mask DLSS got: the reactive one, or the characters')
    VkImage shown = mvdebug ? e->mv : maskdebug && mk ? mk->mask : maskdebug && masked ? e->mask : e->out;
    D.debug_image = shown != e->out;
    if (D.debug_image) b.srcOffsets[1] = (VkOffset3D){(int32_t)w, (int32_t)h, 1};
    RcSourceRange output = dlaa_guest_target(dst);
    // The captured compute destination has no guest attachment descriptor. Prepare it in this scope,
    // where the actual blit and its ownership notification are recorded.
    if (compute_output && compute_output->bytes)
        vk_texture_image_write_prepare(compute_output->image, compute_output->address, compute_output->bytes);
    else if (output.size) vk_texture_image_write_prepare(dst->image, output.offset, output.size);
    vkCmdBlitImage(cb, shown, VK_IMAGE_LAYOUT_GENERAL, dst->image, VK_IMAGE_LAYOUT_GENERAL, 1, &b,
                   D.early && ow != w ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    if (compute_output) dlaa_compute_blit_written(compute_output, &b);
    else if (output.size) {
        resource_image_written((uint64_t)(uintptr_t)dst->image, output.offset);
        vk_texture_storage_written(dst->image, output.offset);
    }
    full_barrier(cb);
    if (dst != t) {
        D.swap_size = e;
        vk_dlaa_swap_img = t->img;
        vk_dlaa_swap_view = e->out_view;
        vk_dlaa_swap_output = e->out;   // the scaling draw reads DLSS's picture
    }
    ex->recorded += 4;
    // Frame generation needs the tonemapped scene. The early forward resolve takes
    // that copy in vk_dlaa_scene_upscale after the game composites bloom.
    if (!D.early && vk_dlaa_hudless_wanted && (!vk_dlssg_active() || (D.cam_size == OCAM_BLOCK && !D.gbuf))) {
        if (!dst->level && !dst->base_layer)
            hudless(ex, dst->image, dst->image, dst->img, NULL, compute_output, dst->width, dst->height);
        else
            hudless(ex, e->out, VK_NULL_HANDLE, NULL, e, NULL, ow, oh);
    }
    vk_exec_draw_new_cb();   // NGX bound pipelines and descriptor sets of its own: the next draw binds everything again
    D.jdepth = z->image, D.jw = z->width, D.jh = z->height;
    __atomic_store_n(&hud_sources.forward_pending, D.early && vk_dlaa_hudless_wanted, __ATOMIC_RELEASE);
    // for frame generation at this frame's present (vk_dlaa_fg_inputs)
    D.fg_cam = camera;
    D.fg_cam_ok = camera_ok;
    D.fg_depth = (VkFfxImage){z->image, z->format, w, h, 0};
    D.fg_mv = (VkFfxImage){e->mv, VK_FORMAT_R16G16_SFLOAT, w, h, 1};
    D.fg_mv_view = e->mv_view;
    D.fg_jx = jx, D.fg_jy = jy, D.fg_at = D.presents;
    fg_publish_size(e, z);
    size_hook_close();
    if (D.jit_used) D.jittered++;
    next_jitter(w, ow);
    if (++D.evals == 1 || D.evals == 300 || D.evals % 3600 == 0)   // 300: one early line with the frames counted
        rt_log("dlaa: %llu frames anti-aliased (scene %ux%u, output %ux%u; %llu frames without the DLAA point, %llu with camera "
               "motion, %llu with the characters' mask, %llu without a camera, %llu jittered (%llu draws moved, %llu not), "
               "%llu jitter holds; resets: %llu size, %llu camera, %llu target; reactive mask %llu, without its copy %llu; "
               "no camera: %llu without a scene-context draw, %llu not in DLSS's submission; %llu kept with zero motion, "
               "%llu runs past %d; %llu cameras read at the draw)",
               (unsigned long long)D.evals, w, h, ow, oh, (unsigned long long)D.frames_missed,
               (unsigned long long)D.mv_frames, (unsigned long long)D.masked, (unsigned long long)D.mv_none,
               (unsigned long long)D.jittered, (unsigned long long)D.jdraws, (unsigned long long)D.jfixed,
               (unsigned long long)D.holds, (unsigned long long)D.why[0], (unsigned long long)D.why[1],
               (unsigned long long)D.why[2], (unsigned long long)R.made, (unsigned long long)R.missed,
               (unsigned long long)D.cam_none_draw, (unsigned long long)D.cam_none_snap, (unsigned long long)D.cam_kept,
               (unsigned long long)D.cam_kept_max, KEEP_MAX, (unsigned long long)D.cam_at_draw);
    window_line();
    return 1;
}

// The composition slot receives a separate R16F output for Streamline. The source is R8 coverage.
static int alpha_image(VkCommandBuffer cb, VkNgxImage *a, VmaAllocation *allocation, uint32_t w, uint32_t h,
                       VkImageUsageFlags usage, VkFormat format) {
    if (!a->img || a->w != w || a->h != h || a->fmt != format) {
        const VkCtx *c = vk_ctx();
        usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        VkImageCreateInfo info = dlaa_image_2d_info(w, h, format, usage);
        const RcVmaRequest request = {.producer = RC_BUDGET_OTHER};
        VmaAllocationCreateInfo placement = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VmaAllocation memory = NULL;
        VkNgxImage next = {.fmt = format, .w = w, .h = h};
        VkResult result = resource_core_vma_create_image(&request, c->vma, &info, &placement,
                                                        &next.img, &memory, NULL, NULL);
        if (result != VK_SUCCESS) {
            vk_device_lost_check(result, "preparing the present alpha image");
            return 0;
        }
        VkImageView view = VK_NULL_HANDLE;
        result = dlaa_image_2d_view(c->device, next.img, format, &view);
        if (result != VK_SUCCESS || !view) {
            vk_device_lost_check(result, "creating the present alpha view");
            free_alpha(&next, &memory);
            return 0;
        }
        next.view = view;
        if (a->img) {
            // Slot reuse checks Streamline's receipt when available. Keep the device-idle
            // compatibility proof for resizing this input on older SDK paths.
            VkResult idle = vk_device_wait_idle();
            if (idle != VK_SUCCESS) {
                vk_device_lost_check(idle, "retiring the present alpha image");
                free_alpha(&next, &memory);
                return 0;
            }
            free_alpha(a, allocation);
        }
        *a = next;
        *allocation = memory;
    }
    // The clear or blit replaces every pixel. Discard the old contents on this
    // recording, including when a previous composition was never submitted.
    VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = a->img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &di);
    return 1;
}

static void clear_alpha(VkCommandBuffer cb, VkImage image) {
    VkImageSubresourceRange all = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkClearColorValue zero = {0};
    vkCmdClearColorImage(cb, image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &all);
    full_barrier(cb);
}

static void hud_group_drop_target(VkDlaaHudSource *use) {
    vk_texture_present_release(use->target_owner);
    use->target_owner = NULL;
    use->target = VK_NULL_HANDLE;
}

static void hud_group_collect(void) {
    for (unsigned i = 0; i < HUD_SOURCE_GROUPS; ++i) {
        VkDlaaHudSource *use = &hud_sources.groups[i];
        if (use != hud_sources.open && use != hud_sources.sealed && hud_group_idle(use))
            hud_group_drop_target(use);
    }
}

static void hud_group_destroy(VkDlaaHudSource *use) {
    hud_group_drop_target(use);
    const VkCtx *c = vk_ctx();
    if (use->coverage.view) vkDestroyImageView(c->device, use->coverage.view, NULL);
    if (use->coverage.img) resource_core_vma_destroy_image(c->vma, use->coverage.img, use->coverage_mem);
    if (use->images.color.view) vkDestroyImageView(c->device, use->images.color.view, NULL);
    if (use->images.color.img) resource_core_vma_destroy_image(c->vma, use->images.color.img, use->color_mem);
    if (use->srgb) resource_core_vma_destroy_image(c->vma, use->srgb, use->srgb_mem);
    *use = (VkDlaaHudSource){0};
}

void vk_dlaa_hud_shutdown(void) {
    if (!vk_ctx()) return;
    hud_sources.open = hud_sources.sealed = NULL;
    for (unsigned i = 0; i < HUD_SOURCE_GROUPS; ++i) {
        VkDlaaHudSource *use = &hud_sources.groups[i];
        if (!hud_group_idle(use)) rt_fatal("dlaa: HUD source still belongs to replay or composition");
        hud_group_destroy(use);
    }
}

typedef struct {
    uint32_t w, h;
    VkFormat format;
    int coverage;
    VkDlaaHudSource *ready;
    VkResult result;
    RcAdmissionStatus admission;
    RcAdmissionVector peak;
} HudSourceRequest;

static int hud_group_matches(const HudSourceRequest *request, const VkDlaaHudSource *use) {
    const VkNgxImage *color = &use->images.color;
    return color->img && color->w == request->w && color->h == request->h && color->fmt == request->format &&
           !!use->coverage.img == request->coverage && !__atomic_load_n(&use->invalid, __ATOMIC_ACQUIRE);
}

static VkResult hud_group_create(HudSourceRequest *want, VkDlaaHudSource *out) {
    const VkCtx *c = vk_ctx();
    const VkImageUsageFlags transfer = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                       VK_IMAGE_USAGE_SAMPLED_BIT;
    enum { COLOR, COVERAGE, ENCODED, MEMBERS };
    VkImageCreateInfo images[MEMBERS] = {
        dlaa_image_2d_info(want->w, want->h, want->format, transfer | VK_IMAGE_USAGE_STORAGE_BIT),
        dlaa_image_2d_info(want->w, want->h, VK_FORMAT_R8_UNORM, transfer | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT),
        dlaa_image_2d_info(want->w, want->h, VK_FORMAT_R8G8B8A8_SRGB, transfer),
    };
    const int enabled[MEMBERS] = {1, want->coverage, want->format == VK_FORMAT_R8G8B8A8_UNORM};
    VmaAllocationCreateInfo placement[MEMBERS];
    const VmaAllocationCreateInfo dedicated = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT};
    RcVmaRequest request = {.producer = RC_BUDGET_TEXTURE};
    want->peak = (RcAdmissionVector){0};
    for (unsigned i = 0; i < MEMBERS; ++i) {
        if (!enabled[i]) continue;
        VkImageFormatProperties properties;
        VkResult r = vkGetPhysicalDeviceImageFormatProperties(c->phys, images[i].format, images[i].imageType,
            images[i].tiling, images[i].usage, images[i].flags, &properties);
        if (r != VK_SUCCESS) return r;
        if (want->w > properties.maxExtent.width || want->h > properties.maxExtent.height)
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        RcAdmissionVector member;
        r = resource_core_vma_image_peak(&request, &images[i], &dedicated, &placement[i], &member);
        if (r != VK_SUCCESS) return r;
        for (unsigned d = 0; d < RC_ADMISSION_MAX_DOMAINS; ++d) {
            if (member.amount[d] > UINT64_MAX - want->peak.amount[d]) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
            want->peak.amount[d] += member.amount[d];
        }
    }
    RcBudgetTicket ticket = {0};
    want->admission = resource_core_budget_reserve(request.producer, 0, &want->peak, &ticket);
    if (want->admission != RC_ADMISSION_OK) return VK_NOT_READY;
    request.peak_ticket = &ticket;
    VkDlaaHudSource next = {0};
    VkImage *native[MEMBERS] = {&next.images.color.img, &next.coverage.img, &next.srgb};
    VmaAllocation *memory[MEMBERS] = {&next.color_mem, &next.coverage_mem, &next.srgb_mem};
    VkImageView *views[MEMBERS] = {&next.images.color.view, &next.coverage.view, NULL};
    VkResult r = VK_SUCCESS;
    for (unsigned i = 0; i < MEMBERS; ++i) {
        if (!enabled[i]) continue;
        if (vk_sync_stopping() || vk_device_is_lost()) {
            r = vk_device_is_lost() ? VK_ERROR_DEVICE_LOST : VK_NOT_READY;
            break;
        }
        r = resource_core_vma_create_image(&request, c->vma, &images[i], &placement[i],
                                          native[i], memory[i], NULL, &want->admission);
        if (r != VK_SUCCESS) break;
        if (!*native[i] || !*memory[i]) {
            r = VK_ERROR_INITIALIZATION_FAILED;
            break;
        }
        if (views[i]) {
            r = dlaa_image_2d_view(c->device, *native[i], images[i].format, views[i]);
            if (r != VK_SUCCESS) break;
        }
    }
    if (resource_core_budget_cancel(&ticket) != RC_ADMISSION_OK)
        rt_fatal("dlaa: HUD source lost its precharged native reservation");
    if (r != VK_SUCCESS) {
        hud_group_destroy(&next);
        return r;
    }
    next.images.color.fmt = want->format;
    next.images.color.w = want->w;
    next.images.color.h = want->h;
    if (want->coverage) {
        next.coverage.fmt = VK_FORMAT_R8_UNORM;
        next.coverage.w = want->w;
        next.coverage.h = want->h;
    }
    *out = next;
    return VK_SUCCESS;
}

static VkDlaaHudSource *hud_group_find(const HudSourceRequest *want) {
    for (unsigned i = 0; i < HUD_SOURCE_GROUPS; ++i) {
        VkDlaaHudSource *use = &hud_sources.groups[i];
        if (use != hud_sources.open && use != hud_sources.sealed && hud_group_idle(use) &&
            hud_group_matches(want, use)) return use;
    }
    return NULL;
}

static int hud_group_prepare(void *arg) {
    HudSourceRequest *want = arg;
    hud_group_collect();
    VkDlaaHudSource *slot = NULL;
    for (unsigned i = 0; i < HUD_SOURCE_GROUPS; ++i) {
        VkDlaaHudSource *use = &hud_sources.groups[i];
        if (use == hud_sources.open || use == hud_sources.sealed || !hud_group_idle(use)) continue;
        if (!slot || !use->images.color.img) slot = use;
        if (!use->images.color.img) break;
    }
    if (!slot) {
        want->result = VK_NOT_READY;
        return 1;
    }
    VkDlaaHudSource next = {0};
    want->result = hud_group_create(want, &next);
    if (want->result != VK_SUCCESS && want->result != VK_ERROR_DEVICE_LOST && want->coverage &&
        !vk_sync_stopping() && !vk_device_is_lost()) {
        // The alpha-optional fallback is a separately admitted color recipe.
        want->coverage = 0;
        want->ready = hud_group_find(want);
        if (want->ready) {
            want->result = VK_SUCCESS;
            return 1;
        }
        want->result = hud_group_create(want, &next);
    }
    if (want->result == VK_SUCCESS) {
        hud_group_destroy(slot);
        *slot = next;
        want->ready = slot;
    }
    return !vk_sync_stopping() && !vk_device_is_lost();
}

void vk_dlaa_hud_incomplete(void) {
    VkDlaaHudSource *use = hud_sources.open;
    if (use) __atomic_store_n(&use->alpha_bad, 1, __ATOMIC_RELEASE);
}

// Only the exact attachment writes reported for this emitted draw are
// covered. Storage and transfer reports outside this scope still taint alpha.
static _Thread_local VkImage hud_coverage_target;
void vk_dlaa_hud_target_report(const VkExec *ex) {
    hud_coverage_target = ex && ex->in_pass && ex->hud_alpha_attachment ? ex->color[0].image : VK_NULL_HANDLE;
}

void vk_dlaa_hud_write(VkImage image) {
    VkDlaaHudSource *use = hud_sources.open;
    if (use && image && image == use->target && image != hud_coverage_target) vk_dlaa_hud_incomplete();
}

void vk_dlaa_hud_draw_emitted(const VkExec *ex) {
    VkDlaaHudSource *use = hud_sources.open;
    if (!use || !use->coverage.img || !ex->in_pass || ex->hud_alpha_attachment) return;
    for (uint32_t i = 0; i < ex->pass_ncolor; ++i)
        if (ex->color[i].image == use->target) vk_dlaa_hud_incomplete();
}

static const VkExecTarget *hud_attachment_target(const VkDlaaHudSource *use, const VkExecTarget *colors,
    uint32_t count) {
    if (!use || !use->target || !use->coverage.view || use->turn != D.presents) return NULL;
    for (uint32_t i = 0; i < count; ++i) {
        const VkExecTarget *target = &colors[i];
        if (target->image == use->target && !target->level && !target->base_layer) return target;
    }
    return NULL;
}

int vk_dlaa_hud_attachment(const VkExec *ex, VkRenderingAttachmentInfo *a) {
    VkDlaaHudSource *use = hud_sources.open;
    const VkExecTarget *t = hud_attachment_target(use, ex->color, ex->pass_ncolor);
    if (!t) return 0;
    if (ex->pass_ncolor != 1 || ex->pass_layers != 1 || t->samples > 1 ||
        t->width != use->coverage.w || t->height != use->coverage.h) {
        vk_dlaa_hud_incomplete();
        return 0;
    }
    hud_group_record(ex, use);
    __atomic_store_n(&use->alpha_seen, 1, __ATOMIC_RELEASE);
    *a = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = use->coverage.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    return 1;
}

void vk_dlaa_present_alpha_unrecorded(unsigned slot) {
    free_alpha(&D.present_alpha[slot], &D.present_alpha_mem[slot]);
}

VkNgxImage vk_dlaa_present_alpha(VkCommandBuffer cb, VkExtent2D ext, const int rect[4], const int to[4],
                                 const VkDlaaHudImages *source, unsigned slot) {
    if (!source || !source->alpha.img ||
        !alpha_image(cb, &D.present_alpha[slot], &D.present_alpha_mem[slot], ext.width, ext.height,
                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_FORMAT_R16_SFLOAT))
        return (VkNgxImage){0};
    full_barrier(cb);
    if (to[2] != (int)ext.width || to[3] != (int)ext.height) clear_alpha(cb, D.present_alpha[slot].img);
    VkImageBlit region = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffsets = {{rect[0], rect[1], 0}, {rect[0] + rect[2], rect[1] + rect[3], 1}},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstOffsets = {{to[0], to[1], 0}, {to[0] + to[2], to[1] + to[3], 1}}};
    vkCmdBlitImage(cb, source->alpha.img, VK_IMAGE_LAYOUT_GENERAL, D.present_alpha[slot].img, VK_IMAGE_LAYOUT_GENERAL,
                   1, &region, VK_FILTER_LINEAR);
    full_barrier(cb);
    return D.present_alpha[slot];
}

int vk_dlaa_hudless_wanted;

static void hud_group_initialize(VkCommandBuffer cb, VkDlaaHudSource *use) {
    if (use->initialized) return;
    VkImage images[] = {use->images.color.img, use->srgb, use->coverage.img};
    VkImageMemoryBarrier2 barriers[sizeof images / sizeof images[0]];
    uint32_t n = 0;
    for (unsigned i = 0; i < sizeof images / sizeof images[0]; ++i) {
        if (!images[i]) continue;
        barriers[n++] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = images[i], .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    }
    VkDependencyInfo info = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = n,
                             .pImageMemoryBarriers = barriers};
    vkCmdPipelineBarrier2(cb, &info);
    use->initialized = 1;
}

static void hudless(VkExec *ex, VkImage source, VkImage target, const VkTexImage *texture,
                    struct Size *reconstruction, const VkTextureColorBlitSource *compute_output,
                    uint32_t w, uint32_t h) {
    hud_sources.open = NULL;
    if (!source || vk_hdr_native()) return;
    // Hold the actual source before preparation can park the recorder. The
    // guest source is also the coverage target; the private source is SR output.
    VkTexturePresentSource *target_owner = NULL;
    VkFormat target_format = VK_FORMAT_UNDEFINED;
    if (target) {
        if (!texture || target != source) return;
        target_format = texture->format;
        const DlaaTargetOwner *owner = frame_targets[DLAA_TARGET_COLOR];
        if (compute_output) {
            if (compute_output->image != target) return;
            target_owner = vk_texture_present_retain_held_image(compute_output->hold, target);
        } else target_owner = owner && texture == &owner->image ? dlaa_present_target(owner, target) :
            vk_texture_present_retain_image(texture, target);
        if (!target_owner) return;
    } else {
        if (!reconstruction || reconstruction->out != source ||
            __atomic_load_n(&reconstruction->retired, __ATOMIC_ACQUIRE)) return;
        size_scene_use(ex, reconstruction);
    }
    int hdr = vk_hdr_wanted();
    HudSourceRequest want = {.w = w, .h = h,
        .format = hdr ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM,
        .coverage = target && D.cam_size == OCAM_BLOCK && !D.gbuf && (vk_dlssg_ui_alpha() || vk_ngxfg_on())};
    want.ready = hud_group_find(&want);
    if (!want.ready) {
        VkSyncProgressResult progress = vk_exec_temporal_prepare(ex, hud_group_prepare, &want);
        vk_device_lost_check(want.result, "HUD source images");
        if (progress != VK_SYNC_PROGRESS_COMPLETE) {
            vk_texture_present_release(target_owner);
            return;
        }
    }
    if (!want.ready) {
        static unsigned logged;
        if (logged++ < 8)
            rt_log("dlaa: HUD source %ux%u unavailable (VkResult %d, admission %d)", w, h, want.result, want.admission);
        vk_texture_present_release(target_owner);
        return;
    }
    VkDlaaHudSource *use = want.ready;
    hud_group_drop_target(use);
    use->target_owner = target_owner;
    use->target = target;
    use->target_format = target_format;
    use->turn = D.presents;
    if (!++hud_sources.generation) rt_fatal("dlaa: HUD source generation exhausted");
    use->images.generation = hud_sources.generation;
    use->images.alpha = (VkNgxImage){0};
    use->images.frame = 0;
    use->images.exact = 0;
    use->alpha_seen = use->alpha_bad = 0;
    use->last_scene = 0;
    hud_group_record(ex, use);
    VkCommandBuffer cb = ex->cb;
    hud_group_initialize(cb, use);
    if (use->coverage.img) {
        full_barrier(cb);
        clear_alpha(cb, use->coverage.img);
    }
    VkImageBlit region = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffsets = {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstOffsets = {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}}};
    vkCmdBlitImage(cb, source, VK_IMAGE_LAYOUT_GENERAL, hdr ? use->images.color.img : use->srgb,
                   VK_IMAGE_LAYOUT_GENERAL, 1, &region, VK_FILTER_NEAREST);
    full_barrier(cb);
    if (!hdr) {
        VkImageCopy copy = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .extent = {w, h, 1}};
        vkCmdCopyImage(cb, use->srgb, VK_IMAGE_LAYOUT_GENERAL, use->images.color.img, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
        full_barrier(cb);
    }
    hud_sources.open = use;
}

static int scene_color(VkFormat f) {
    return f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || f == VK_FORMAT_R16G16B16A16_SFLOAT;
}

static int samples_scene(const VkExec *ex, int with_bloom) {
    uint64_t scene = ex->tex[NVN_STAGE_FS][0], bloom = ex->tex[NVN_STAGE_FS][1];
    if (!scene || (with_bloom && !bloom)) return 0;
    vk_texture_lock();
    const VkTexImage *selected = vk_texture_current_registration_locked(NVN_HANDLE_TEX(scene), NULL, NULL);
    int match = dlaa_selected_image_matches(selected, D.color.img, frame_targets[DLAA_TARGET_COLOR]);
    if (match && with_bloom) {
        const VkTexImage *b = vk_texture_current_registration_locked(NVN_HANDLE_TEX(bloom), NULL, NULL);
        match = b && b->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && b->width < D.color.width && b->height < D.color.height;
    }
    vk_texture_unlock();
    return match;
}

int vk_dlaa_swap_matches(const VkTexImage *selected) {
    return vk_dlaa_swap_img && dlaa_selected_image_matches(selected, vk_dlaa_swap_img, frame_targets[DLAA_TARGET_COLOR]);
}

int vk_dlaa_swap_active(const VkExec *ex) {
    return ex->in_pass &&
           (D.early ? ex->tonemap_extent.width != 0 : ex->pass_w > vk_dlaa_swap_img->width + 1);
}

static VkImageBlit blit_scene(VkCommandBuffer cb, VkImage src, uint32_t sw, uint32_t sh,
                              VkImage dst, uint32_t dw, uint32_t dh) {
    VkImageBlit region = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .srcOffsets = {{0, 0, 0}, {(int32_t)sw, (int32_t)sh, 1}},
                          .dstOffsets = {{0, 0, 0}, {(int32_t)dw, (int32_t)dh, 1}}};
    vkCmdBlitImage(cb, src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, 1, &region,
                   sw == dw && sh == dh ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);
    return region;
}

static int dlaa_tonemap_targets(const VkExec *ex, const VkExecTarget *color, const VkExecTarget *depth,
    uint32_t color_count) {
    return D.composite && vk_dlaa_swap_img && color[0].image == D.composite && D.last &&
           (D.last->ow != D.color.width || D.last->oh != D.color.height) &&
           color_count == 1 && depth->image == D.depth.image && post_current(post.frame) &&
           post.frame->size.width == D.last->ow && post.frame->size.height == D.last->oh &&
           !ex->pipe.ds.dt && !ex->pipe.ds.dw && !(ex->pipe.color.blend_en & 1) && samples_scene(ex, 1);
}

int vk_dlaa_tonemap_target(const VkExec *ex) {
    return dlaa_tonemap_targets(ex, ex->color, &ex->depth, ex->pass_ncolor);
}

void vk_dlaa_tonemap_begin(VkExec *ex, VkRenderingAttachmentInfo *color, VkRenderingAttachmentInfo *depth,
                           VkRenderingInfo *ri) {
    DlaaPostImages *images = post.frame;
    post_scene_use(ex, images);
    VkRect2D sc = vk_exec_scissor(ex);
    // The viewport can extend beyond the guest target. A clipped or letterboxed
    // composite keeps the guest colours in the portion the draw will not cover.
    int covers = sc.extent.width == ex->pass_w && sc.extent.height == ex->pass_h &&
                 (!ex->has_vp || (ex->vp[0][0] <= 0 && ex->vp[0][1] <= 0 &&
                  ex->vp[0][0] + ex->vp[0][2] >= ex->pass_w && ex->vp[0][1] + ex->vp[0][3] >= ex->pass_h));
    if (!covers) {
        full_barrier(ex->cb);
        blit_scene(ex->cb, ex->color[0].image, ex->pass_w, ex->pass_h, images->member[POST_COLOR].image,
                   images->size.width, images->size.height);
        full_barrier(ex->cb);
        ex->recorded += 3;
    }
    ex->tonemap_drawn = 0;
    color->loadOp = covers ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : VK_ATTACHMENT_LOAD_OP_LOAD;
    color->imageView = images->member[POST_COLOR].view;
    depth->imageView = images->member[POST_DEPTH].view;
    depth->loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth->storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    ri->renderArea.extent = ex->tonemap_extent = images->size;
}

void vk_dlaa_tonemap_end(VkExec *ex) {
    ex->tonemap_extent = (VkExtent2D){0};
    if (!ex->tonemap_drawn) return;
    // Keep the original guest target current for copies and alternate present paths.
    full_barrier(ex->cb);
    const VkExecTarget *target = &ex->color[0];
    RcSourceRange output = dlaa_guest_target(target);
    if (output.size) vk_texture_image_write_prepare(target->image, output.offset, output.size);
    DlaaPostImages *images = post.frame;
    blit_scene(ex->cb, images->member[POST_COLOR].image, images->size.width, images->size.height,
               target->image, ex->pass_w, ex->pass_h);
    if (output.size) {
        resource_image_written((uint64_t)(uintptr_t)target->image, output.offset);
        vk_texture_storage_written(target->image, output.offset);
    }
    full_barrier(ex->cb);
    D.promoted = 1;
    ex->recorded += 3;
}

// The final encode samples the compute upscaler's output even on frames that skip the HUD pass.
static int scene_effects_current(const VkDlaaSceneEffects *effects) {
    if (!effects || !effects->complete) return 0;
    const VkExecTarget *targets[2] = {&D.color,&D.depth};
    for (unsigned i=0;i<2;i++) {
        const VkExecTarget *t=targets[i];
        uint64_t address=t->tex?(uint64_t)(uintptr_t)t->tex->mem:0;
        if (t->image!=effects->images[i] || address!=effects->addresses[i]) return 0;
    }
    return 1;
}
int vk_dlaa_scene_effects(VkDlaaSceneEffects *out) {
    if (!out) return 0;
    *out=(VkDlaaSceneEffects){0};
    // The caller has finished earlier recording workers before sealing this serial dispatch. Snapshot both
    // possible edge-fill outputs even if reconstruction is currently inactive; state can become active later.
    const VkExecTarget targets[2]={D.color,D.depth};
    for (unsigned i=0;i<2;i++) {
        const VkExecTarget *t=&targets[i];
        out->images[i]=t->image;
        out->addresses[i]=t->tex?(uint64_t)(uintptr_t)t->tex->mem:0;
        if (t->image && (!out->addresses[i] ||
            !vk_texture_image_effect(t->image,out->addresses[i],&out->ranges[i],&out->owners[i]))) return 0;
    }
    out->complete=1;
    if (!scene_effects_current(out)) {out->complete=0;return 0;}
    return 1;
}
static int dlaa_edge_mapping(const VkDlaaSceneEffects *scene, unsigned role,
                             RcRegistryImageWriteExtent **out, uint32_t *count) {
    RcRegistryImageBinding unused;
    const RcSourceRange range = scene->ranges[role];
    vk_texture_image_binding(scene->images[role], range.offset, range.size, &unused, out, count);
    uint64_t address = range.offset;
    for (uint32_t i = 0; i < *count; i++) {
        const RcInterval interval = (*out)[i].interval;
        if (!interval.mapping || !rc_range_valid(interval.range) || interval.address != address ||
            interval.range.size > range.size - (address - range.offset)) return 0;
        address += interval.range.size;
    }
    return *count && address - range.offset == range.size;
}

void vk_dlaa_scene_prepare_edges(VkDlaaSceneEffects *scene, const VkTextureColorBlitSource *primary) {
    if (!scene->complete || !dl_active() || !D.jit_used || !primary->mapping_count) return;
    const unsigned axes = dlaa_edge_axes(&D.color, D.ux, D.uy);
    if (!axes) return;
    const VkExecTarget *targets[2] = {&D.color, &D.depth};
    const uint32_t mask = dlaa_edge_targets(&D.color, &D.depth);
    RcRegistryImageWriteExtent *mapping[2] = {0};
    uint32_t counts[2] = {0};
    int disjoint = 1;
    vk_texture_lock();
    for (unsigned i = 0; i < 2 && disjoint; i++) {
        if (!scene->images[i]) continue;
        disjoint = scene->images[i] != primary->image && dlaa_edge_mapping(scene, i, &mapping[i], &counts[i]);
        for (uint32_t j = 0; j < counts[i] && disjoint; j++)
            for (uint32_t k = 0; k < primary->mapping_count && disjoint; k++)
                disjoint = !rc_range_overlap(mapping[i][j].interval.range, primary->mapping[k].range);
    }
    if (scene->images[0] && scene->images[0] == scene->images[1]) disjoint = 0;
    for (uint32_t i = 0; i < counts[0] && disjoint; i++)
        for (uint32_t j = 0; j < counts[1] && disjoint; j++)
            disjoint = !rc_range_overlap(mapping[0][i].interval.range, mapping[1][j].interval.range);
    for (unsigned i = 0; disjoint && i < 2; i++) {
        const VkExecTarget *target = targets[i];
        if (!(mask & (1u << i)) || !target->tex || !target->img) continue;
        VkImageCopy regions[2];
        uint32_t n = dlaa_edge_regions(target, i, axes, regions);
        vk_texture_projected_self_copy_capture(target->tex, target->img, regions, n, &scene->edge[i]);
    }
    vk_texture_unlock();
    free(mapping[0]);
    free(mapping[1]);
}

void vk_dlaa_scene_release_edges(VkDlaaSceneEffects *scene) {
    for (unsigned i = 0; i < 2; i++) vk_texture_projected_transfer_cancel(&scene->edge[i]);
}

int vk_dlaa_scene_upscale(VkExec *ex, uint32_t sw, uint32_t sh, const VkTexImage *dst,
                         VkDlaaSceneEffects *effects, const VkTextureColorBlitSource *output) {
    if (!dl_active() || !output || !output->hold || output->image != dst->image || output->address != dst->addr) return 0;
    vk_exec_hooks_wait();
    if (!dlaa_targets_armed() || D.gbuf || (D.done && !D.early) || sw != D.color.width || sh != D.color.height ||
        dst->image == D.color.image || dst->width < sw || dst->height < sh)
        return 0;
    vk_exec_hook_turn();
    // A sealed declaration cannot acquire a newly selected guest output. The caller's ordinary blit still
    // writes only its captured destination, so it remains valid when optional reconstruction loses this guard.
    if (effects && !scene_effects_current(effects)) return 0;
    if (D.early) {
        if (!D.composite || D.up) return 0;
        vk_dlaa_scene_effects_cancel_unselected(effects, 0);
        DlaaPostImages *images = post.frame;
        int promoted = D.promoted && post_current(images);
        if (promoted) post_scene_use(ex, images);
        vk_exec_barrier(ex);
        VkImage src = promoted ? images->member[POST_COLOR].image : D.composite;
        uint32_t width = promoted ? images->size.width : sw, height = promoted ? images->size.height : sh;
        VkImageBlit blit = blit_scene(ex->cb, src, width, height, dst->image, dst->width, dst->height);
        dlaa_compute_blit_written(output, &blit);
        full_barrier(ex->cb);
        ex->recorded += 2;
        if (vk_dlaa_hudless_wanted)
            hudless(ex, dst->image, dst->image, dst, NULL, output, dst->width, dst->height);
        __atomic_store_n(&hud_sources.forward_pending, 0, __ATOMIC_RELEASE);
        D.up = 1;
        D.early_outputs++;
        if (promoted || (sw == dst->width && sh == dst->height)) D.full_size_outputs++;
        if (D.early_outputs == 1 || D.early_outputs == 300 || D.early_outputs % 3600 == 0)
            rt_log("dlaa: before bloom on %llu of %llu reconstructed frames; %llu tonemapped at output size; scene %ux%u, tonemap %ux%u, output %ux%u",
                   (unsigned long long)D.early_outputs, (unsigned long long)D.evals, (unsigned long long)D.full_size_outputs, sw, sh,
                   width, height, dst->width, dst->height);
        return 1;
    }
    VkExecTarget out = {.image = dst->image, .format = dst->format, .width = dst->width, .height = dst->height, .img = dst};
    if (!evaluate(ex, &out, effects, 0, output)) return 0;
    D.ended = D.done = D.up = 1;
    static int told;
    if (!told++) rt_log("dlaa: resolved the scene at its compute upscale (%ux%u -> %ux%u)", sw, sh, dst->width, dst->height);
    return 1;
}

// Two frame shapes. A deferred frame (the G-buffer, several colour targets at the scene size) arms on the light target
// drawn after it and ends the scene at the first pass a quarter as wide (the bloom), as described above. Ocarina's
// frame has no G-buffer: its light target (B10G11R11 at the scene size, with the D32F scene depth) is drawn into from the
// first opaque pass on, half- and quarter-size effect passes come and go in between. Measured (NX2NAVI_DUMP, Kokiri
// Forest, DLAA in a 2560x1440 picture): the last pass with the scene depth is the bloom composite into a second light
// target (each such pass at the scene's size takes over the arming, with its depth); the game's upscaler (its stand-in
// here, vk_tonemap.c) then fills a light-format target without depth, at the picture's size, which the HUD is drawn
// into; a last draw copies that target into the 8-bit sRGB output target, encoding only, since the game tonemaps before
// its post pass (vk_tonemap.c, vk_hdr.c). vk_dlaa_scene_upscale resolves the armed pair directly into the compute
// upscaler's destination. For frames that reach the graphics hooks without that dispatch, the light-format output
// pass or the final encode still provides the resolve point.
static int srgb8(VkFormat f) { return f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_B8G8R8A8_SRGB; }

// The output target DLAA runs at. With the game's own HDR (vk_hdr.c vk_hdr_native) the same pass, at the same point of
// the frame, tonemaps into its PQ picture (vk_hdr.c vk_hdr_native_picture) instead of the 8-bit sRGB one; matching sRGB
// alone left DLAA off for the whole run there (no frame anti-aliased) and with it the frame generation that takes
// DLAA's depth and motion.
static int output_target(VkFormat f) { return srgb8(f) || vk_hdr_native_picture(f); }

double wide_dlss_share(void);   // widescreen.c: the DLSS mode's share of the picture (1 DLAA), 0 off

static int dlaa_deferred;   // at least one scene-size G-buffer established the deferred frame shape

typedef enum {
    DLAA_PASS_NONE, DLAA_PASS_COMPOSITE, DLAA_PASS_GBUFFER, DLAA_PASS_ARM, DLAA_PASS_BLOOM,
    DLAA_PASS_OUTPUT, DLAA_PASS_NATIVE_OUTPUT, DLAA_PASS_ENCODE, DLAA_PASS_DEFER_WAIT,
    DLAA_PASS_DEFER_OUTPUT, DLAA_PASS_NEEDS_OUTPUT
} DlaaPassAction;

typedef struct {
    const VkExecTarget *color, *depth;
    uint32_t color_count, width, layers;
} DlaaPassTargets;

typedef struct {
    DlaaPassAction action;
    int second_scene, one_color, wrong_shape;
} DlaaPassDecision;

// The same exact view, format, shader input and size predicates drive preview and execution.
// The preview reports the need for an ordered hook without committing this frame's state.
static DlaaPassDecision dlaa_pass_select(const VkExec *ex, DlaaPassTargets targets, const unsigned output[2]) {
    const VkExecTarget *c = &targets.color[0], *z = targets.depth;
    const uint32_t n = targets.color_count;
    DlaaPassDecision decision = {0};
    if (!dl_active()) return decision;
    if (D.early) {
        if (!D.composite && n == 1 && targets.layers == 1 && c->image && c->image != D.color.image &&
            c->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && c->width == D.color.width && c->height == D.color.height &&
            !c->level && !c->base_layer && c->samples <= 1 && z->image == D.depth.image && !ex->pipe.ds.dt &&
            !ex->pipe.ds.dw && !(ex->pipe.color.blend_en & 1) && samples_scene(ex, 1))
            decision.action = DLAA_PASS_COMPOSITE;
        return decision;
    }
    if (D.done) {
        decision.second_scene = n == 1 && c->image && scene_color(c->format) && z->image &&
            z->format == VK_FORMAT_D32_SFLOAT && c->width == D.color.width && c->height == D.color.height &&
            z->width == c->width;
        return decision;
    }
    const int armed = dlaa_targets_armed();
    const int scene = z->image && z->format == VK_FORMAT_D32_SFLOAT && z->width >= DLAA_SCENE_MIN_W;
    // A half-size effects buffer must not replace the armed light/depth pair.
    if (n >= 3 && scene && !(armed && (z->width < D.color.width || z->height < D.color.height)))
        decision.action = DLAA_PASS_GBUFFER;
    else if ((D.gbuf || !dlaa_deferred) && n == 1 && c->image && scene_color(c->format) && scene &&
        c->width == z->width && c->height == z->height &&
        (!armed || (c->width >= D.color.width && c->height >= D.color.height)) && c->samples <= 1 && !c->is3d)
        decision.action = DLAA_PASS_ARM;
    if (decision.action || !armed) return decision;
    if (before_bloom && !dlaa_deferred && !use_fsr() && n == 2 && !z->image &&
        c->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && targets.color[1].format == c->format &&
        c->width == targets.color[1].width && c->height == targets.color[1].height &&
        c->width < D.color.width && c->height < D.color.height && samples_scene(ex, 0)) {
        decision.action = DLAA_PASS_BLOOM;
        return decision;
    }
    if (!dlaa_deferred) {
        decision.one_color = n == 1 && c->image && !z->image && c->image != D.color.image && c->samples <= 1 && !c->is3d;
        if (!decision.one_color) return decision;
        const double share = wide_dlss_share();
        const int native = share <= 0 || share >= 1;
        const int big = D.share_off || native || c->width >= 0.95 * D.color.width / share;
        const int same = c->width == D.color.width && c->height == D.color.height;
        const int larger = c->width > D.color.width && c->height > D.color.height;
        // The output shares the scene aspect ratio. A large square minimap must keep its own content.
        const int shape = (double)c->width * D.color.height <= 1.03 * (double)c->height * D.color.width &&
                          (double)c->height * D.color.width <= 1.03 * (double)c->width * D.color.height;
        decision.wrong_shape = !shape && big && larger;
        if (shape && big && larger) decision.action = DLAA_PASS_OUTPUT;
        else if (same && native && scene_color(c->format)) decision.action = DLAA_PASS_NATIVE_OUTPUT;
        else if (same && output_target(c->format)) decision.action = DLAA_PASS_ENCODE;
        return decision;
    }
    if (!D.ended) {
        if (targets.width && targets.width * 4 <= D.color.width)
            decision.action = !output ? DLAA_PASS_NEEDS_OUTPUT :
                D.color.width >= output[0] ? DLAA_PASS_ENCODE : DLAA_PASS_DEFER_WAIT;
    } else if (n == 1 && c->image && !z->image && scene_color(c->format) && c->width > D.color.width &&
        c->samples <= 1 && !c->is3d) {
        if (!output) decision.action = DLAA_PASS_NEEDS_OUTPUT;
        else if (c->width == output[0] && c->height == output[1]) decision.action = DLAA_PASS_DEFER_OUTPUT;
    }
    return decision;
}

uint32_t vk_dlaa_graphics_pass_work(const VkExec *ex, const VkExecGraphicsPassShape *shape) {
    const DlaaPassTargets targets = {shape->color, &shape->depth, shape->pass_ncolor, shape->pass_w, shape->pass_layers};
    unsigned output[2];
    const unsigned *size = dlaa_deferred && res_output_size_current(output) ? output : NULL;
    const DlaaPassAction action = dlaa_pass_select(ex, targets, size).action;
    uint32_t work = 0;
    if (action == DLAA_PASS_GBUFFER || action == DLAA_PASS_ARM) work = VK_DLAA_GRAPHICS_PASS_OBSERVE;
    else if (action) work = VK_DLAA_GRAPHICS_PASS_HOOK;
    if (dlaa_tonemap_targets(ex, targets.color, targets.depth, targets.color_count)) work |= VK_DLAA_GRAPHICS_PASS_TONEMAP;
    if (hud_attachment_target(hud_sources.open, targets.color, targets.color_count)) work |= VK_DLAA_GRAPHICS_PASS_HUD;
    return work;
}

struct VkDlaaPreparedObserver {
    RcRegistryPosition read;
    VkExecTarget color, depth;
    DlaaTargetOwner *color_owner, *depth_owner;
    RcBudgetTicket metadata;
    DlaaPassAction action;
    int stores_ids;
};

static int dlaa_pass_stores_ids(DlaaPassAction action, const VkExecTarget *c, const VkExecTarget *z) {
    return action == DLAA_PASS_GBUFFER && c->image && c->format == VK_FORMAT_R8G8B8A8_UNORM &&
        c->width == z->width && c->height == z->height && c->samples <= 1 && !c->is3d &&
        (!D.ids_ok || !dlaa_target_valid(frame_targets[DLAA_TARGET_IDS]) || c->width >= D.ids.width);
}

static int dlaa_observer_target_same(const VkExecTarget *a, const VkExecTarget *b) {
    return a->exact_snapshot == b->exact_snapshot && a->tex == b->tex && a->img == b->img &&
        a->image == b->image && a->format == b->format && a->view_type == b->view_type &&
        a->aspect == b->aspect && a->level == b->level && a->base_layer == b->base_layer &&
        a->layers == b->layers && a->width == b->width && a->height == b->height &&
        a->samples == b->samples && a->is3d == b->is3d && a->bc_backing == b->bc_backing;
}

static int dlaa_observer_current(const VkExec *ex, DlaaPassTargets targets, const VkDlaaPreparedObserver *prepared) {
    if (!prepared || ex->fence != prepared->read.scene ||
        rc_registry_position_compare(prepared->read, vk_exec_resource_position()) ||
        !dlaa_observer_target_same(&prepared->color, &targets.color[0]) ||
        !dlaa_observer_target_same(&prepared->depth, targets.depth)) return 0;
    // G-buffer and scene arm are selected before any output-sized resolve decision.
    DlaaPassAction action = dlaa_pass_select(ex, targets, NULL).action;
    return action == prepared->action &&
        prepared->stores_ids == dlaa_pass_stores_ids(action, &targets.color[0], targets.depth);
}

int vk_dlaa_prepared_observer_current(const VkExec *ex, const VkExecGraphicsPassShape *shape,
                                    const VkDlaaPreparedObserver *prepared) {
    const DlaaPassTargets targets = {shape->color, &shape->depth, shape->pass_ncolor, shape->pass_w, shape->pass_layers};
    return dlaa_observer_current(ex, targets, prepared);
}

void vk_dlaa_prepared_observer_cancel(VkDlaaPreparedObserver **in_out) {
    VkDlaaPreparedObserver *prepared = *in_out;
    *in_out = NULL;
    if (!prepared) return;
    dlaa_target_release(prepared->color_owner);
    dlaa_target_release(prepared->depth_owner);
    RcBudgetTicket metadata = prepared->metadata;
    vk_replay_metadata_free(prepared, &metadata);
}

int vk_dlaa_prepare_observer(const VkExec *ex, const VkExecGraphicsPassShape *shape,
                            VkTextureAliasRead *color_seed, VkTextureAliasRead *depth_seed, VkDlaaPreparedObserver **out) {
    *out = NULL;
    RcRegistryPosition read = vk_exec_resource_position();
    if (!read.scene || !read.ordinal || read.scene != ex->fence || shape->in_pass) return 0;
    const DlaaPassTargets targets = {shape->color, &shape->depth, shape->pass_ncolor, shape->pass_w, shape->pass_layers};
    const DlaaPassAction action = dlaa_pass_select(ex, targets, NULL).action;
    if (action != DLAA_PASS_GBUFFER && action != DLAA_PASS_ARM) return 0;
    const VkExecTarget *c = &shape->color[0], *z = &shape->depth;
    if (depth_seed && (z->format != VK_FORMAT_D32_SFLOAT || z->aspect != VK_IMAGE_ASPECT_DEPTH_BIT ||
        z->level || z->base_layer || z->layers != 1 || z->samples != 1)) return 0;
    const int stores_ids = dlaa_pass_stores_ids(action, c, z);

    RcBudgetTicket metadata = {0};
    void *storage = NULL;
    if (vk_replay_metadata_allocate(sizeof(VkDlaaPreparedObserver), 1, RC_BUDGET_TEXTURE, &metadata, &storage) != VK_SUCCESS)
        return 0;
    VkDlaaPreparedObserver *prepared = storage;
    *prepared = (VkDlaaPreparedObserver){.read = read, .color = *c, .depth = *z, .metadata = metadata,
        .action = action, .stores_ids = stores_ids};
    if (stores_ids || action == DLAA_PASS_ARM) {
        prepared->color_owner = dlaa_target_capture(c, color_seed);
        if (!prepared->color_owner) goto unavailable;
    }
    if (action == DLAA_PASS_ARM) {
        prepared->depth_owner = dlaa_target_capture(z, depth_seed);
        if (!prepared->depth_owner) goto unavailable;
    }
    *out = prepared;
    return 1;

unavailable:
    vk_dlaa_prepared_observer_cancel(&prepared);
    return 0;
}

static int dlaa_pass_execute(VkExec *ex, VkDlaaPreparedObserver *prepared) {
    if (!dl_active()) return prepared == NULL;
    const DlaaPassTargets targets = {ex->color, &ex->depth, ex->pass_ncolor, ex->pass_w, ex->pass_layers};
    const VkExecTarget *c = &targets.color[0], *z = targets.depth;
    unsigned out[2], scene[2];
    const unsigned *size = NULL;
    if (!prepared && !D.early && !D.done) {
        // Ocarina takes sizes from this frame. Other supported patch sets have an explicit output size.
        res_sizes(out, scene);
        size = out;
    }
    if (prepared && !dlaa_observer_current(ex, targets, prepared)) return 0;
    const DlaaPassDecision decision = dlaa_pass_select(ex, targets, size);
    const int stores_ids = dlaa_pass_stores_ids(decision.action, c, z);
    DlaaTargetOwner *color = NULL, *depth = NULL;
    if (prepared) {
        color = prepared->color_owner;
        depth = prepared->depth_owner;
        prepared->color_owner = NULL;
        prepared->depth_owner = NULL;
    }
    if (decision.second_scene) F.second++, F.second_img = c->image;
    if (decision.one_color) F.one_n++, F.one_w = c->width, F.one_h = c->height, F.one_fmt = c->format;
    if (decision.wrong_shape && D.shape_logged < 8)
        D.shape_logged++, rt_log("dlaa: a %ux%u pass is not the output pass: its shape is not the scene's %ux%u",
                                 c->width, c->height, D.color.width, D.color.height);
    switch (decision.action) {
    case DLAA_PASS_NONE: case DLAA_PASS_NEEDS_OUTPUT:
        return 1;
    case DLAA_PASS_COMPOSITE:
        vk_exec_hook_turn();
        D.composite = c->image;
        if (!D.debug_image && (D.last->ow != c->width || D.last->oh != c->height) &&
            post_images(ex, (VkExtent2D){D.last->ow, D.last->oh})) {
            D.swap_size = D.last;
            vk_dlaa_swap_img = D.color.img;
            vk_dlaa_swap_view = D.last->out_view;
            vk_dlaa_swap_output = D.last->out;
        }
        return 1;
    case DLAA_PASS_GBUFFER:
        if (!dlaa_deferred)
            rt_log("dlaa: a G-buffer pass (%d colour targets, depth %ux%u, target 0 format %d): the deferred frame shape",
                   targets.color_count, z->width, z->height, c->image ? (int)c->format : 0);
        D.gbuf = 1;
        D.armed = 0;
        dlaa_deferred = 1;
        // Keep the largest matching object-class buffer for the character bias mask.
        if (stores_ids) {
            dlaa_frame_target_set(DLAA_TARGET_IDS, color);
            D.ids = dlaa_owned_target(c, color);
            D.ids_ok = 1;
            dlaa_target_record(ex, color, 1);
        }
        return 1;
    case DLAA_PASS_ARM: {
        int first = !dlaa_targets_armed();
        dlaa_frame_target_set(DLAA_TARGET_COLOR, color);
        dlaa_frame_target_set(DLAA_TARGET_DEPTH, depth);
        D.color = dlaa_owned_target(c, color);
        D.depth = dlaa_owned_target(z, depth);
        D.armed = 1;
        dlaa_target_record(ex, color, 1);
        dlaa_target_record(ex, depth, 1);
        static int told;
        if (first && !told++)
            rt_log("dlaa: first scene armed: light target %ux%u format %d, depth %ux%u", c->width, c->height, (int)c->format,
                   z->width, z->height);
        if (first) mask_snapshot(ex);
        return 1;
    }
    case DLAA_PASS_BLOOM:
        vk_exec_hook_turn();
        if (evaluate(ex, NULL, NULL, 1, NULL)) D.done = D.ended = 1;
        else D.early = 0;
        return 1;
    case DLAA_PASS_NATIVE_OUTPUT: {
        static int told;
        if (!told++)
            rt_log("dlaa: the pass the HUD is drawn into is at the scene's size (%ux%u, format %d): DLSS writes there",
                   c->width, c->height, (int)c->format);
        break;
    }
    case DLAA_PASS_OUTPUT: case DLAA_PASS_ENCODE: case DLAA_PASS_DEFER_WAIT: case DLAA_PASS_DEFER_OUTPUT:
        break;
    }
    vk_exec_hook_turn();
    if (decision.action == DLAA_PASS_DEFER_WAIT) {
        D.ended = 1;
        return 1;
    }
    if (decision.action == DLAA_PASS_ENCODE) {
        if (evaluate(ex, NULL, NULL, 0, NULL)) D.ended = D.done = 1;
        return 1;
    }
    if (evaluate(ex, c, NULL, 0, NULL)) {
        D.done = D.up = 1;
        if (decision.action != DLAA_PASS_DEFER_OUTPUT) D.ended = 1;
    }
    return 1;
}

void vk_dlaa_pass(VkExec *ex) {
    (void)dlaa_pass_execute(ex, NULL);
}

int vk_dlaa_pass_prepared(VkExec *ex, VkDlaaPreparedObserver **in_out) {
    if (!*in_out || !dlaa_pass_execute(ex, *in_out)) return 0;
    vk_dlaa_prepared_observer_cancel(in_out);
    return 1;
}

// NX2NAVI_DLSS, read at each present (the menu switches it live): the scene size of the mode (resolution.c) and the
// texture bias that goes with it while DLSS or FSR 3.1 runs. NVIDIA's DLSS guide asks for log2(render / output) - 1 on
// top of the game's own bias: the render scale's part (res_dlss_lod_bias) samples textures for the output's resolution,
// and the -1 sharper still, since the jittered frames DLSS accumulates resolve the detail one level finer: -1 at DLAA,
// -1.58 in Quality (67 %), -2 in Performance. The -1 comes only with the jitter (jitter_on): without it the same pixel
// samples the same spot every frame, and a sharper level would shimmer instead. The sampler clamps the sum to the
// device's limit (vk_texture.c make_sampler).
#define JITTER_LOD_BIAS (-1.0f)
static void dlss_mode(void) {
    static char mode[32];
    static int bias_on = -1;
    const char *m = dynres_override();   // dynamic resolution's step (dynres.c) while it differs from the setting
    if (!m) m = cfg_value("NX2NAVI_DLSS");
    if (strcmp(m, mode)) {
        snprintf(mode, sizeof mode, "%s", m);
        res_dlss_mode(m);
        bias_on = -1;
    }
    int b = dl_active();
    if (b != bias_on)
        bias_on = b, vk_sampler_dlss_bias(b ? res_dlss_lod_bias() + (jitter_on() ? JITTER_LOD_BIAS : 0.0f) : 0.0f);
}


// The preset, read again when a setting changed (the menu); a different one recreates the features at the next DLAA run.
static void preset_check(void) {
    unsigned v = cfg_version();
    if (v == D.cfg_seen) return;
    D.cfg_seen = v;
    int p = preset_value();
    if (p == D.preset) return;
    if (D.evals) {
        if (p) rt_log("dlaa: DLSS preset %c in every mode; the features are created again", 'A' + p - 1);
        else rt_log("dlaa: DLSS presets per mode; the features are created again");
    }
    D.preset = p, D.preset_dirty = 1, D.reset = 1;
}

// The scene's size after a setting changed (a DLSS or FSR 3.1 mode picked in the menu). The game makes its scene targets
// again at the new size and they start empty, so the first frames drawn into them are wrong: the first one is lit
// without what the frame before it left (a dark scene), and two frames later the scene can be black with the interface
// intact (Ultra Performance and DLAA). Measured by capturing every present over six mode changes, 2026-11-10: bad
// frames 3 and 5 presents after the change, with every other setting the same. evaluate notes a scene of another size;
// vk_dlaa_frame counts the presents since a setting last changed, so a game that draws two sizes in turn holds nothing.
static int setting_presents;            // vk_dlaa_frame: presents left in which that counts as a mode change
#define SETTING_PRESENTS 60

// vk_backend.c frame_hold, each present after vk_dlaa_frame: 1 once when the scene took a new size after a setting
// changed. The previous picture then stays on screen over the frames described above.
int vk_dlaa_resized_take(void) {
    if (!__atomic_exchange_n(&size_changed, 0, __ATOMIC_ACQ_REL) || !setting_presents) return 0;
    setting_presents = 0;
    return 1;
}

uint64_t vk_dlaa_scene_frame(void) {
    return __atomic_load_n(&D.presents, __ATOMIC_RELAXED);
}

void vk_dlaa_frame(void) {
    static unsigned cfg_seen;
    static int cfg_known;
    unsigned v = cfg_version();
    __atomic_add_fetch(&D.presents, 1, __ATOMIC_RELAXED);
    if (D.fg_size && D.presents != D.fg_at + 1) fg_forget_size();
    size_collect_retired();
    if (cfg_known && v != cfg_seen) setting_presents = SETTING_PRESENTS;
    else if (setting_presents) setting_presents--;
    cfg_seen = v, cfg_known = 1;
    D.swap_size = NULL;
    vk_dlaa_swap_img = NULL, vk_dlaa_swap_view = VK_NULL_HANDLE, vk_dlaa_swap_output = VK_NULL_HANDLE;
    post_frame_end();
    dlaa_frame_targets_close();
    D.early = D.promoted = 0, D.composite = VK_NULL_HANDLE;
    dlss_mode();
    preset_check();
    if (!N.ngx_up) {   // FSR 3.1 follows the setting
        fsr_frame = fsr3();
        N.available = fsr_frame && cfg_bool("NX2NAVI_DLAA") && vk_ffx_load();
    }
    int on = N.available && cfg_bool("NX2NAVI_DLAA");
    if (on != D.on) {
        rt_log("dlaa: %s", on ? "on" : "off");
        D.on = on;
        camera_history_enable(on);
        // either way the next DLAA frame starts a new history from an unjittered frame
        D.reset = 1, D.cam_prev_ok = 0, D.jit_ok = D.jit_used = D.armed = D.done = D.gbuf = D.ended = D.up = 0;
        D.hold = 0;
        fg_forget_size();
        hud_sources.open = hud_sources.sealed = NULL;
        memset(&F, 0, sizeof F);
        __atomic_store_n(&D.size_hook_open, on && !D.failed, __ATOMIC_RELEASE);
        return;
    }
    if (!dl_active()) {
        size_hook_close();
        return;
    }
    if (D.share_off) D.share_off--;
    if (!D.done && D.armed) D.share_off = 120;   // vk_dlaa_pass: the output pass without the mode's share for a while
    if (!D.done) {
        D.frames_missed++;
        // the first few after DLSS first ran: what the frame drew, to tell a frame drawn another way from a loading screen
        if (D.evals && D.miss_logged < 24) {
            D.miss_logged++;
            char arm[64] = "never armed";
            if (D.armed) snprintf(arm, sizeof arm, "armed on a %ux%u light target", D.color.width, D.color.height);
            rt_log("dlaa: frame %llu without the DLAA point: %s; %u scene-depth draws (%u into another D32 depth); %u "
                   "later passes with one colour target and no depth (last %ux%u format %d); %llu submissions since the "
                   "last present, %d held submissions waiting for a signal",
                   (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), arm, F.zdraws, F.zother,
                   F.one_n, F.one_w, F.one_h, (int)F.one_fmt, (unsigned long long)(vk_exec_submit_seq - D.seq_present),
                   vk_exec_held());
        }
    }
    if (D.done && F.second) {
        D.seconds++;
        if (D.second_logged < 24)
            D.second_logged++, rt_log("dlaa: frame %llu: a second scene after DLSS ran (%u passes into light target %p; "
                                      "DLSS ran on %p); %llu submissions since the last present, %d held",
                                      (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED),
                                      F.second, (void *)F.second_img, (void *)D.color.image,
                                      (unsigned long long)(vk_exec_submit_seq - D.seq_present), vk_exec_held());
    }
    D.seq_present = vk_exec_submit_seq;
    if (D.jit_used && !D.done) D.hold = 60, D.holds++;   // jittered but not resolved: that frame shook
    else if (D.hold) D.hold--;
    if (!D.done) D.jit_ok = 0;
    D.armed = D.done = D.jit_used = D.gbuf = D.ended = D.up = D.ids_ok = 0;
    memset(&F, 0, sizeof F);
    __atomic_store_n(&D.size_hook_open, 1, __ATOMIC_RELEASE);
}

// ---------------------------------------------------------------- the game's FXAA (NX2NAVI_FXAA, live)
// NX2navi's FXAA is two full-screen passes after the bloom, just before the HUD: one copies the scene target into a
// scratch target, the other draws the anti-aliased result back over the scene (found by diffing frame dumps with the
// game's own switch, ModelSceneExtensionParam EnableFXAA, on and off). With fxaa = 0 the draws of that second pass's
// fragment shader are skipped, so the scene reaches the HUD as it was. Shaders are known by the FNV-1a hash of their
// SPIR-V, as vk_dump.c names them; a translator change gives new hashes and the switch then does nothing.
// With a scene smaller than the output (the game's own 5/6 render scale, or a DLSS mode) the same shader draws the
// scaled picture into the output-sized target, in a pass without the depth attachment the in-place FXAA has: there it
// is always drawn, whatever fxaa says (skipping it would leave the 3D picture out), unless DLSS already wrote that
// target this frame.
// The stock translator's hash, the one with tools/xlate patch 0010 (PCSS, PR #89) applied, the one with patches 0010
// and 0011 (the shadow filter spread) applied, and that one with the zero_locals job flag (Mesa, vk_caps.c).
// A scaling draw this list misses still scales DLSS's picture, not
// the jittered scene: after DLSS ran, passes larger than the light target sample DLSS's output (vk_dlaa_swap_img).
static const uint64_t FXAA_FS[] = {0xf450003ebeb3dc52ull, 0x0a12f4bc2a9a4fbeull, 0x5c43ae18948d52cbull,
                                   0xaad42238801d31d8ull};

static uint64_t spirv_hash(const GfxShader *s) {
    static struct { const GfxShader *s; uint64_t h; } tab[8192];
    uintptr_t k = ((uintptr_t)s >> 4) * 0x9E3779B97F4A7C15ull;
    for (uint32_t i = 0, j = (uint32_t)(k >> 51); i < 8192; i++, j = (j + 1) & 8191) {
        if (tab[j].s == s) return tab[j].h;
        if (!tab[j].s) {
            uint64_t words = 0, h = 0xcbf29ce484222325ull;
            const uint8_t *b = (const uint8_t *)gfx_shader_spirv(s, &words);
            for (uint64_t n = 0; b && n < words * 4; n++) h = (h ^ b[n]) * 0x100000001b3ull;
            tab[j].s = s, tab[j].h = h;
            return h;
        }
    }
    return 0;   // table full: no skip
}

int vk_dlaa_skip_draw_depth(VkImage depth, const GfxShader *fs) {
    static uint64_t frame_read = ~0ull;
    static int fxaa = 1;
    // fxaa = 0 reports once whether it found the game's FXAA draw: the first skip, or a minute of frames (3600) with
    // none, which is what a translator change that moves FXAA_FS looks like on every GPU, with or without DLSS
    static uint64_t fx_off_frames;
    static int fx_found, fx_reported;
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (f != frame_read) {   // once a frame: the menu switches it live
        frame_read = f, fxaa = cfg_bool("NX2NAVI_FXAA");
        if (!fxaa && !fx_found && !fx_reported && ++fx_off_frames == 3600) {
            fx_reported = 1;
            rt_log("dlaa: fxaa = 0 found no draw of the game's FXAA in %llu frames (its fragment shader is not in "
                   "FXAA_FS): the FXAA stays on", (unsigned long long)fx_off_frames);
        }
    }
    if (fxaa && !D.up) return 0;   // nothing to skip: no shader lookup per draw at the defaults
    if (!fs) return 0;
    uint64_t h = spirv_hash(fs);
    int known = 0;
    for (size_t i = 0; i < sizeof FXAA_FS / sizeof FXAA_FS[0]; i++) known |= h == FXAA_FS[i];
    if (!known) return 0;
    if (!depth) {   // the scaling pass, which DLSS replaced
        static int told;
        if (D.up && !told++) rt_log("dlaa: the game's scaling draw left out, DLSS wrote it (fragment shader %016llx)",
                                    (unsigned long long)h);
        return D.up;
    }
    if (!fxaa && !fx_found) {
        fx_found = 1;
        rt_log("dlaa: fxaa = 0 skips the game's FXAA (fragment shader %016llx)", (unsigned long long)h);
    }
    return !fxaa;
}

int vk_dlaa_skip_draw(VkExec *ex, const GfxShader *fs) { return vk_dlaa_skip_draw_depth(ex->depth.image, fs); }


// ---------------------------------------------------------------- DLSS Frame Generation through NGX (vk_ngxfg)
// gfx_present.c records this into the present command buffer once the frame's picture, HUD included, is in bb. It
// generates count frames between the previous call's picture and this one into out[0..count-1] (MultiFrameIndex 1 to
// count) and a copy of bb into real; gfx_present.c's presenter shows them in that order, then the rendered one. The
// inputs are the ones the Streamline path gets (vk_dlssg.c): the depth, motion vectors, jitter and camera of the frame
// DLSS last ran on, and the picture without the HUD.
int vk_ngxfg_max(void) { return vk_ngxfg_wanted() && N.available && !FG.failed ? FG.max : 0; }
int vk_ngxfg_on(void) { return vk_ngxfg_max() && !strcmp(cfg_value("NX2NAVI_FRAMEGEN"), "dlss"); }
void vk_ngxfg_reset(void) { FG.reset = 1; }

int vk_ngxfg_evaluate(VkCommandBuffer cb, const VkNgxImage *bb, const VkNgxImage *hudless, const VkNgxImage *alpha, const VkNgxImage *out, int count,
                      const VkNgxImage *real) {
    if (!vk_ngxfg_max() || count < 1) return 0;
    if (count > FG.max) count = FG.max;
    VkFfxImage d, m;
    VkImageView dv, mvv;
    VkDlaaCamera c;
    float jx = 0, jy = 0;
    int reset = 0;
    if (!vk_dlaa_fg_inputs(&d, &m, &jx, &jy, &reset) || !vk_dlaa_fg_views(&dv, &mvv) || !vk_dlaa_fg_camera(&c)) {
        FG.reset = 1;   // a frame without DLSS's inputs: the next generation has no valid previous frame
        return 0;
    }
    if (!FG.p && !NGX_OK(N.alloc_params(&FG.p))) {
        FG.failed = 1;
        rt_log("ngxfg: parameter allocation failed: frame generation off");
        return 0;
    }
    NgxParam *p = FG.p;
    if (FG.feat && (FG.w != bb->w || FG.h != bb->h || FG.fmt != bb->fmt)) {
        N.release(FG.feat);
        FG.feat = NULL;
    }
    if (!FG.feat) {
        p_setu(p, "CreationNodeMask", 1);
        p_setu(p, "VisibilityNodeMask", 1);
        p_setu(p, "Width", bb->w);
        p_setu(p, "Height", bb->h);
        p_setu(p, "DLSSG.BackbufferFormat", (unsigned)bb->fmt);
        uint64_t t0 = perf_mark_begin();
        NgxResult r = N.create(cb, NGX_FEATURE_FRAME_GENERATION, p, &FG.feat);
        perf_mark_end(PERF_M_NGX, t0, 0);
        if (!NGX_OK(r) || !FG.feat) {
            FG.failed = 1, FG.feat = NULL;
            rt_log("ngxfg: DLSS Frame Generation feature creation failed (%08x): frame generation off", (unsigned)r);
            return 0;
        }
        FG.w = bb->w, FG.h = bb->h, FG.fmt = bb->fmt, FG.reset = 1;
        rt_log("ngxfg: DLSS Frame Generation created at %ux%u (format %d), depth and motion %ux%u", bb->w, bb->h,
               (int)bb->fmt, d.width, d.height);
    }
    NgxResourceVK rb = res_image(bb->view, bb->img, bb->fmt, bb->w, bb->h, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    NgxResourceVK rz = res_image(dv, d.image, d.format, d.width, d.height, VK_IMAGE_ASPECT_DEPTH_BIT, 0);
    NgxResourceVK rm = res_image(mvv, m.image, m.format, m.width, m.height, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    NgxResourceVK rh = hudless ? res_image(hudless->view, hudless->img, hudless->fmt, hudless->w, hudless->h,
                                           VK_IMAGE_ASPECT_COLOR_BIT, 0) : rb;
    NgxResourceVK rr = res_image(real->view, real->img, real->fmt, real->w, real->h, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    p_setp(p, "DLSSG.Backbuffer", &rb);
    p_setp(p, "DLSSG.Depth", &rz);
    p_setp(p, "DLSSG.MVecs", &rm);
    p_setp(p, "DLSSG.HUDLess", hudless ? &rh : NULL);
    p_setp(p, "DLSSG.UI", NULL);
    NgxResourceVK ra = alpha ? res_image(alpha->view, alpha->img, alpha->fmt, alpha->w, alpha->h,
                                          VK_IMAGE_ASPECT_COLOR_BIT, 0) : rb;
    p_setp(p, "DLSSG.UIAlpha", alpha ? &ra : NULL);
    p_setu(p, "DLSSG.UserInterfaceRecompositionEnabled", hudless && alpha);
    p_setp(p, "DLSSG.BidirectionalDistortionField", NULL);
    p_setp(p, "DLSSG.OutputReal", &rr);
    p_setp(p, "DLSSG.OutputDisableInterpolation", NULL);
    p_setp(p, "DLSSG.CameraViewToClip", c.view_to_clip);
    p_setp(p, "DLSSG.ClipToCameraView", c.clip_to_view);
    p_setp(p, "DLSSG.ClipToLensClip", NULL);
    p_setp(p, "DLSSG.ClipToPrevClip", c.clip_to_prev_clip);
    p_setp(p, "DLSSG.PrevClipToClip", c.prev_clip_to_clip);
    p_setf(p, "DLSSG.JitterOffsetX", jx);
    p_setf(p, "DLSSG.JitterOffsetY", jy);
    p_setf(p, "DLSSG.MvecScaleX", 1.0f / (float)m.width);   // the vectors are in render pixels
    p_setf(p, "DLSSG.MvecScaleY", 1.0f / (float)m.height);
    p_setf(p, "DLSSG.CameraPinholeOffsetX", 0.0f);
    p_setf(p, "DLSSG.CameraPinholeOffsetY", 0.0f);
    static const char *const pos[] = {"DLSSG.CameraPosX", "DLSSG.CameraPosY", "DLSSG.CameraPosZ"};
    static const char *const up[] = {"DLSSG.CameraUpX", "DLSSG.CameraUpY", "DLSSG.CameraUpZ"};
    static const char *const rt[] = {"DLSSG.CameraRightX", "DLSSG.CameraRightY", "DLSSG.CameraRightZ"};
    static const char *const fw[] = {"DLSSG.CameraFwdX", "DLSSG.CameraFwdY", "DLSSG.CameraFwdZ"};
    for (int i = 0; i < 3; i++)
        p_setf(p, pos[i], c.pos[i]), p_setf(p, up[i], c.up[i]), p_setf(p, rt[i], c.right[i]), p_setf(p, fw[i], c.fwd[i]);
    p_setf(p, "DLSSG.CameraNear", c.znear);   // the frame's own planes (proj_planes), as vk_dlssg.c gives them
    p_setf(p, "DLSSG.CameraFar", c.zfar);
    p_setf(p, "DLSSG.CameraFOV", c.fov_y);
    p_setf(p, "DLSSG.CameraAspectRatio", c.aspect);
    p_setu(p, "DLSSG.ColorBuffersHDR", 0);
    p_setu(p, "DLSSG.DepthInverted", (unsigned)c.inverted);
    p_setu(p, "DLSSG.CameraMotionIncluded", 1);
    p_setu(p, "DLSSG.Reset", (unsigned)(reset || FG.reset));
    p_setu(p, "DLSSG.AutomodeOverrideReset", 0);
    p_setu(p, "DLSSG.NotRenderingGameFrames", 0);
    p_setu(p, "DLSSG.OrthoProjection", 0);
    p_setu(p, "DLSSG.MvecDilated", 0);
    p_setu(p, "DLSSG.MenuDetectionEnabled", 0);
    struct { const char *base; uint32_t w, h; } rects[] = {
        {"DLSSG.MVecsSubrect", m.width, m.height}, {"DLSSG.DepthSubrect", d.width, d.height},
        {"DLSSG.UIAlphaSubrect", bb->w, bb->h}, {"DLSSG.HUDLessSubrect", bb->w, bb->h}, {"DLSSG.InputBackbufferSubrect", bb->w, bb->h},
        {"DLSSG.OutputInterpolatedSubrect", bb->w, bb->h}, {"DLSSG.OutputRealSubrect", bb->w, bb->h}};
    for (size_t i = 0; i < sizeof rects / sizeof *rects; i++) {
        char k[64];
        snprintf(k, sizeof k, "%sBaseX", rects[i].base), p_setu(p, k, 0);
        snprintf(k, sizeof k, "%sBaseY", rects[i].base), p_setu(p, k, 0);
        snprintf(k, sizeof k, "%sWidth", rects[i].base), p_setu(p, k, rects[i].w);
        snprintf(k, sizeof k, "%sHeight", rects[i].base), p_setu(p, k, rects[i].h);
    }
    p_setu(p, "DLSSG.MultiFrameCount", (unsigned)count);
    int made = 0;
    for (int i = 0; i < count; i++) {
        NgxResourceVK ro = res_image(out[i].view, out[i].img, out[i].fmt, out[i].w, out[i].h, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        p_setp(p, "DLSSG.OutputInterpolated", &ro);
        p_setu(p, "DLSSG.MultiFrameIndex", (unsigned)(i + 1));
        NgxResult r = N.evaluate(cb, FG.feat, p, NULL);
        if (!NGX_OK(r)) {
            static int told;
            if (told++ < 5) rt_log("ngxfg: evaluation %d of %d failed (%08x)", i + 1, count, (unsigned)r);
            break;
        }
        made++;
    }
    full_barrier(cb);
    FG.reset = 0;
    if (made && (++FG.evals == 1 || FG.evals % 3600 == 0))
        rt_log("ngxfg: %d frames with generation (%d generated per rendered frame)", FG.evals, made);
    return made;
}

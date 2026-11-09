// DLSS Frame Generation (NX2NAVI_FRAMEGEN = dlss; applies after a restart) through nx2navi_dlssg.dll
// (runtime/dlssg/nx2_sl.cpp), which drives NVIDIA Streamline. Like AMD's frame generation (vk_fg.cpp) it needs DLSS or
// FSR 3.1 upscaling on, whose depth, motion vectors and camera it hands over, and an RTX 40 or 50 GPU.
//
// The DLL and Streamline (sl.interposer.dll and its plugins, nvngx_dlssg.dll) sit next to the game. The instance and
// device are created through Streamline's proxies (vk_device.c), so it can add the extensions, features and queues
// DLSS-G needs; the present path calls Streamline's swapchain functions (gfx_present.c), and every present records the
// frame's constants and tags first (vk_dlssg_frame). Generation pauses while the menu is open.
#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "vk.h"
#include "vk_dlaa.h"
#include "vk_dlssg.h"
#include "gfx_wsys.h"
#include "dlssg/nx2_sl.h"
#include "ui/menu.h"

static struct {
    int wanted, loaded, active;
    PFN_nx2sl_init init;
    PFN_nx2sl_instance_proc gipa;
    PFN_nx2sl_device_proc gdpa;
    PFN_nx2sl_available available;
    PFN_nx2sl_begin begin;
    PFN_nx2sl_begin_paced begin_paced;
    PFN_nx2sl_reflex_mode reflex_mode;
    int reflex;
    uint64_t last_present_ns;
    PFN_nx2sl_submit submit;
    PFN_nx2sl_suspend suspend;
    PFN_nx2sl_frame frame;
    PFN_nx2sl_present_marker marker;
    PFN_nx2sl_output_frames output_frames;
    PFN_nx2sl_shutdown shutdown;
    PFN_vkCreateSwapchainKHR create_sc;
    PFN_vkDestroySwapchainKHR destroy_sc;
    PFN_vkGetSwapchainImagesKHR images;
    PFN_vkAcquireNextImageKHR acquire;
    PFN_vkQueuePresentKHR present;
} S = {.wanted = -1};

static void sl_line(const char *line) { rt_log("%s", line); }
static int max_multiplier;

int vk_dlssg_max_multiplier(void) { return __atomic_load_n(&max_multiplier, __ATOMIC_RELAXED); }

int vk_dlssg_multiplier(void) {
    int requested = cfg_int("NX2NAVI_DLSSG_MULTIPLIER");
    int limit = vk_dlssg_max_multiplier();
    if (requested < 2 || requested > NX2SL_MAX_MULTIPLIER) requested = 2;
    return requested > limit ? limit : requested;
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
    if (S.loaded) return S.loaded > 0;
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
    S.init = (PFN_nx2sl_init)(void *)GetProcAddress(m, "nx2sl_init");
    S.gipa = (PFN_nx2sl_instance_proc)(void *)GetProcAddress(m, "nx2sl_instance_proc");
    S.gdpa = (PFN_nx2sl_device_proc)(void *)GetProcAddress(m, "nx2sl_device_proc");
    S.available = (PFN_nx2sl_available)(void *)GetProcAddress(m, "nx2sl_available");
    S.begin = (PFN_nx2sl_begin)(void *)GetProcAddress(m, "nx2sl_begin");
    S.begin_paced = (PFN_nx2sl_begin_paced)(void *)GetProcAddress(m, "nx2sl_begin_paced");
    S.reflex_mode = (PFN_nx2sl_reflex_mode)(void *)GetProcAddress(m, "nx2sl_reflex_mode");
    S.submit = (PFN_nx2sl_submit)(void *)GetProcAddress(m, "nx2sl_submit");
    S.suspend = (PFN_nx2sl_suspend)(void *)GetProcAddress(m, "nx2sl_suspend");
    S.frame = (PFN_nx2sl_frame)(void *)GetProcAddress(m, "nx2sl_frame");
    S.marker = (PFN_nx2sl_present_marker)(void *)GetProcAddress(m, "nx2sl_present_marker");
    S.output_frames = (PFN_nx2sl_output_frames)(void *)GetProcAddress(m, "nx2sl_output_frames");
    S.shutdown = (PFN_nx2sl_shutdown)(void *)GetProcAddress(m, "nx2sl_shutdown");
    if (!ver || ver() != NX2SL_VERSION || !S.init || !S.gipa || !S.gdpa || !S.available || !S.frame || !S.marker || !S.begin || !S.submit || !S.suspend) {
        rt_log("dlssg: nx2navi_dlssg.dll is from another build: DLSS Frame Generation off");
        return 0;
    }
    if (!S.init(dir, sl_line)) return 0;
    S.loaded = 1;
    return 1;
#else
    return 0;
#endif
}

VkResult vk_dlssg_create_instance(const VkInstanceCreateInfo *ci, VkInstance *out) {
    PFN_vkCreateInstance f = S.loaded > 0 ? (PFN_vkCreateInstance)S.gipa()(VK_NULL_HANDLE, "vkCreateInstance") : NULL;
    return f ? f(ci, NULL, out) : vkCreateInstance(ci, NULL, out);
}

// An instance function through the interposer while Streamline is loaded, else NULL (the caller takes the loader's).
// The window surface goes through it: Streamline learns the window a swapchain shows from vkCreateWin32SurfaceKHR, and
// without that its swapchain hook found "no window corresponding to this application" and generated no frames.
PFN_vkVoidFunction vk_dlssg_instance_proc(VkInstance inst, const char *name) {
    return S.loaded > 0 ? S.gipa()(inst, name) : NULL;
}

// The interposer's vkCreateDevice finds the device's instance in a map its own vkEnumeratePhysicalDevices fills; a
// physical device enumerated through the loader alone left it a null instance, so the plugins got an empty instance
// table and sl.common called a null GetPhysicalDeviceMemoryProperties while it set up. Enumerate through it as well.
VkResult vk_dlssg_enumerate(VkInstance inst, uint32_t *n, VkPhysicalDevice *pd) {
    PFN_vkEnumeratePhysicalDevices f =
        S.loaded > 0 ? (PFN_vkEnumeratePhysicalDevices)S.gipa()(inst, "vkEnumeratePhysicalDevices") : NULL;
    return f ? f(inst, n, pd) : vkEnumeratePhysicalDevices(inst, n, pd);
}

VkResult vk_dlssg_create_device(VkInstance inst, VkPhysicalDevice pd, const VkDeviceCreateInfo *ci, VkDevice *out) {
    PFN_vkCreateDevice f = S.loaded > 0 ? (PFN_vkCreateDevice)S.gipa()(inst, "vkCreateDevice") : NULL;
    return f ? f(pd, ci, NULL, out) : vkCreateDevice(pd, ci, NULL, out);
}

void vk_dlssg_device_ready(VkDevice dev) {
    if (S.loaded > 0 && S.gdpa) vk_dpa = S.gdpa();
    if (S.loaded <= 0 || !S.available()) return;
    PFN_vkGetDeviceProcAddr g = S.gdpa();
    S.create_sc = (PFN_vkCreateSwapchainKHR)g(dev, "vkCreateSwapchainKHR");
    S.destroy_sc = (PFN_vkDestroySwapchainKHR)g(dev, "vkDestroySwapchainKHR");
    S.images = (PFN_vkGetSwapchainImagesKHR)g(dev, "vkGetSwapchainImagesKHR");
    S.acquire = (PFN_vkAcquireNextImageKHR)g(dev, "vkAcquireNextImageKHR");
    S.present = (PFN_vkQueuePresentKHR)g(dev, "vkQueuePresentKHR");
    S.active = S.create_sc && S.destroy_sc && S.images && S.acquire && S.present;
    if (S.active) vk_dlaa_hudless_wanted = 1;
    rt_log("dlssg: %s", S.active ? "DLSS Frame Generation on the swapchain" : "the swapchain hooks are missing: off");
}

void *vk_dlssg_begin(void) {
    if (!S.active) return NULL;
    if (!S.begin_paced) return S.begin();
    uint64_t last = __atomic_load_n(&S.last_present_ns, __ATOMIC_RELAXED);
    // The driver cannot pace against a window that stopped presenting.
    int recent = last && kern_now_ns() - last < 200000000ull;
    return S.begin_paced(recent && __atomic_load_n(&S.reflex, __ATOMIC_RELAXED));
}
void vk_dlssg_submit(void *token, int end) { if (S.active && token) S.submit(token, end); }
void vk_dlssg_suspend(int suspend) { if (S.active) S.suspend(suspend); }

int vk_dlssg_active(void) { return S.active; }
uint64_t vk_dlssg_output_frames(void) { return S.active && S.output_frames ? S.output_frames() : 0; }

VkResult vk_dlssg_create_swapchain(VkDevice d, const VkSwapchainCreateInfoKHR *ci, VkSwapchainKHR *sc) {
    VkResult r = S.create_sc(d, ci, NULL, sc);
    if (r == VK_SUCCESS) {
        int mode = cfg_int("NX2NAVI_REFLEX");
        if (mode < 0 || mode > 2) mode = 1;
        if (S.reflex_mode && !S.reflex_mode(mode)) mode = 0;
        __atomic_store_n(&S.reflex, mode, __ATOMIC_RELAXED);
        __atomic_store_n(&S.last_present_ns, 0, __ATOMIC_RELAXED);
    }
    return r;
}
void vk_dlssg_destroy_swapchain(VkDevice d, VkSwapchainKHR sc) { S.destroy_sc(d, sc, NULL); }
VkResult vk_dlssg_get_images(VkDevice d, VkSwapchainKHR sc, uint32_t *n, VkImage *img) { return S.images(d, sc, n, img); }
VkResult vk_dlssg_acquire(VkDevice d, VkSwapchainKHR sc, uint64_t t, VkSemaphore sem, uint32_t *idx) {
    return S.acquire(d, sc, t, sem, VK_NULL_HANDLE, idx);
}

VkResult vk_dlssg_present(VkQueue q, const VkPresentInfoKHR *pi) {
    S.marker(0, NULL);
    VkResult r = S.present(q, pi);
    VkResult async = VK_SUCCESS;
    int limit = S.marker(1, &async);
    if (async == VK_ERROR_DEVICE_LOST || (r == VK_SUCCESS && async != VK_SUCCESS)) r = async;
    if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR)
        __atomic_store_n(&S.last_present_ns, kern_now_ns(), __ATOMIC_RELAXED);
    if (limit && limit != vk_dlssg_max_multiplier()) {
        __atomic_store_n(&max_multiplier, limit, __ATOMIC_RELAXED);
        rt_log("dlssg: supported frame generation multipliers up to %dx", limit);
    }
    return r;
}

void vk_dlssg_frame(void *token, VkCommandBuffer cb, VkExtent2D ext, const int to[4], int scene, const Nx2SlImage *hudless, const Nx2SlImage *ui) {
    if (!S.active) return;
    VkFfxImage d = {0}, m = {0};
    VkImageView dv = VK_NULL_HANDLE, mvv = VK_NULL_HANDLE;
    VkImageUsageFlags usage[2] = {0};
    VkDlaaCamera c;
    float jx = 0, jy = 0;
    int reset = 0;
    int inputs = vk_dlaa_fg_inputs(&d, &m, &jx, &jy, &reset);
    int views = inputs && vk_dlaa_fg_views(&dv, &mvv, usage);
    int camera = vk_dlaa_fg_camera(&c);
    int have = scene && views && camera;
    int menu = menu_is_open();
    int on = have && !menu;
    static unsigned frames;
    if (frames++ % 300 == 0)
        rt_log("dlssg: inputs scene=%d depth/motion=%d views=%d camera=%d hudless=%d ui-alpha=%d menu=%d",
               scene, inputs, views, camera, hudless != NULL, ui != NULL, menu);
    Nx2SlImage zd = {d.image, dv, d.format, d.width, d.height, VK_IMAGE_LAYOUT_GENERAL, usage[0]};
    Nx2SlImage zm = {m.image, mvv, m.format, m.width, m.height, VK_IMAGE_LAYOUT_GENERAL, usage[1]};
    Nx2SlCamera k;
    memset(&k, 0, sizeof k);
    if (have) {
        memcpy(k.view_to_clip, c.view_to_clip, 64), memcpy(k.clip_to_view, c.clip_to_view, 64);
        memcpy(k.clip_to_prev_clip, c.clip_to_prev_clip, 64), memcpy(k.prev_clip_to_clip, c.prev_clip_to_clip, 64);
        memcpy(k.pos, c.pos, 12), memcpy(k.up, c.up, 12), memcpy(k.right, c.right, 12), memcpy(k.fwd, c.fwd, 12);
        k.znear = c.znear, k.zfar = c.zfar, k.fov_y = c.fov_y, k.aspect = c.aspect;
        k.jitter[0] = jx, k.jitter[1] = jy;
        k.mv_scale[0] = 1.0f / (float)d.width, k.mv_scale[1] = 1.0f / (float)d.height;   // render pixels
        k.output_rect = (VkRect2D){{to[0], to[1]}, {(uint32_t)to[2], (uint32_t)to[3]}};
        k.reset = reset;
    }
    int result = S.frame(token, cb, have ? &zd : NULL, have ? &zm : NULL, hudless, ui, have ? &k : NULL, ext.width, ext.height,
            on ? vk_dlssg_multiplier() : 0);
    if (result == NX2SL_REBUILD_SWAPCHAIN) gfx_window_mark_dirty();
}

void vk_dlssg_release(void) {
    if (S.loaded > 0 && S.shutdown) S.shutdown();
    S.active = 0;
    __atomic_store_n(&max_multiplier, 0, __ATOMIC_RELAXED);
}

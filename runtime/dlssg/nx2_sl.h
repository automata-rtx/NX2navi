// The C interface of nx2navi_dlssg.dll (runtime/dlssg/nx2_sl.cpp), which runtime/vk_dlssg.c loads for DLSS Frame
// Generation. The DLL is built with MSVC against NVIDIA's Streamline SDK (its C++ types have MSVC's ABI); nothing of
// the SDK is in this repository, and the game itself is built without it.
#pragma once
#include <stdint.h>
#include <vulkan/vulkan.h>

#define NX2SL_VERSION 4

typedef struct {
    VkImage image;
    VkImageView view;
    VkFormat format;
    uint32_t width, height;
    VkImageLayout layout;      // the layout it is in when Streamline reads it (at the present)
    VkImageUsageFlags usage;
} Nx2SlImage;

typedef struct {
    float view_to_clip[16], clip_to_view[16], clip_to_prev_clip[16], prev_clip_to_clip[16];   // row-major
    float pos[3], up[3], right[3], fwd[3];
    float znear, zfar, fov_y, aspect;
    float jitter[2];           // render pixels
    float mv_scale[2];         // what makes the motion vectors fractions of the render size
    int reset;
} Nx2SlCamera;

enum { NX2SL_INPUTS_ABI_V1 = 1 };
enum { NX2SL_INPUT_HUDLESS = 1u << 0, NX2SL_INPUT_UI_ALPHA = 1u << 1 };
enum { NX2SL_INPUT_NO_LATE_TAGS = 0, NX2SL_INPUT_TIMELINE = 1, NX2SL_INPUT_UNPROVEN = 2 };

// Version 1 has no request flags. A host initializes each input and output prefix.
typedef struct {
    uint32_t size, version, flags, reserved;
    uint64_t frame, generation;
    VkSwapchainKHR swapchain;
    VkCommandBuffer cb;
    const Nx2SlImage *depth, *motion, *hudless, *alpha;
    const Nx2SlCamera *camera;
    uint32_t width, height;
    int32_t rect[4], generated;
} Nx2SlFrameInputsV1;

typedef struct {
    uint32_t size, version, flags, reserved;
    uint64_t ticket, frame, generation;
    VkSwapchainKHR swapchain;
    uint32_t sdk_frame, accepted, uncertain;
} Nx2SlInputReceipt;

typedef struct {
    uint32_t size, version, flags, reserved;
    uint64_t ticket;
    VkSemaphore semaphore;
    uint64_t value;
} Nx2SlInputCompletion;

enum { NX2SL_FG_MEMORY_ABI_V1 = 1, NX2SL_FG_ESTIMATE_ROUGH_VRAM = 1u << 0 };

// Input dimensions and formats are native Vulkan values. Optional HUD/alpha formats are VK_FORMAT_UNDEFINED.
typedef struct {
    uint32_t size, version, flags, reserved;
    uint64_t generation;
    VkSwapchainKHR swapchain;
    uint32_t backbuffers, generated;
    uint32_t mvec_width, mvec_height, depth_width, depth_height, color_width, color_height;
    VkFormat color, depth, motion, hudless, alpha;
    uint32_t recomposition, queue_mode;
} Nx2SlFgConfigV1;

typedef struct {
    uint32_t size, version, flags, reserved;
    uint64_t vram_bytes;
    int32_t sdk_result;
    uint32_t padding;
} Nx2SlFgEstimateV1;

typedef void (*Nx2SlLog)(const char *line);

#ifdef __cplusplus
extern "C" {
#endif
#ifdef NX2SL_BUILD
#define NX2SL_API __declspec(dllexport)
#else
#define NX2SL_API
#endif
typedef void (*PFN_nx2sl_present_rect)(const int rect[4]);
typedef int (*PFN_nx2sl_version)(void);
// slInit with DLSS-G, Reflex and PCL for Vulkan; dir: the folder of sl.interposer.dll and the plugins (the game's)
typedef int (*PFN_nx2sl_init)(const wchar_t *dir, Nx2SlLog log);
// The interposer's proc address functions: vkCreateInstance and vkCreateDevice through them add what DLSS-G needs
// (extensions, features, queues) and tell Streamline the device; the swapchain functions through them are its hooks.
typedef PFN_vkGetInstanceProcAddr (*PFN_nx2sl_instance_proc)(void);
typedef PFN_vkGetDeviceProcAddr (*PFN_nx2sl_device_proc)(void);
// 1 when DLSS-G loaded and supports this GPU (after the device exists)
typedef int (*PFN_nx2sl_available)(void);
// Optional: log Streamline's support result for frame generation after availability failed (its general check of
// the system, driver and adapters: Streamline looks at one adapter only when given its LUID, which Vulkan has not).
typedef void (*PFN_nx2sl_log_unavailable)(VkPhysicalDevice physical);
// Reflex's low-latency mode, 0 off, 1 on, 2 on with boost: after the device, and after each swapchain the game creates
// through the interposer (the mode is the swapchain's)
typedef void (*PFN_nx2sl_reflex)(int mode);
// Frames are numbered by the game's presents: frame n ends with its n-th present (vk_dlssg.c vk_dlssg_game_present).
// Where the game starts a frame: with hold, Reflex's sleep, which holds the start until the time the driver computes
// from the measured render and queue times; then the frame's simulation and render-submit-start markers.
typedef void (*PFN_nx2sl_sleep)(uint32_t frame, int hold);
// Separate entry points for hosts that can observe the game update and the host render thread independently.
typedef void (*PFN_nx2sl_wait)(uint32_t frame, int hold);
enum Nx2SlPhase { NX2SL_SIM_START, NX2SL_SIM_END, NX2SL_RENDER_START };
typedef void (*PFN_nx2sl_phase)(uint32_t frame, enum Nx2SlPhase phase);
// The last CPU submission for this frame went out.
typedef void (*PFN_nx2sl_submit_end)(uint32_t frame);
// Each host present, before the present call: the frame's constants and tags, and frame generation on or off. enable is
// the number of frames to generate between two rendered ones (0 off; 1, 2 or 3 for 2x, 3x or 4x), at most what
// nx2sl_frames_max reports; a DLL without nx2sl_frames_max (an older build) generates one for any nonzero value. Then
// nx2sl_present_marker(0) before and (1) after the present.
typedef void (*PFN_nx2sl_ui_alpha)(const Nx2SlImage *alpha);
typedef int (*PFN_nx2sl_frame)(VkCommandBuffer cb, const Nx2SlImage *depth, const Nx2SlImage *mv, const Nx2SlImage *hudless,
                               const Nx2SlCamera *cam, uint32_t out_w, uint32_t out_h, int enable, uint32_t frame);
// The most frames DLSS-G can generate between two rendered ones on this GPU and driver (Streamline's
// numFramesToGenerateMax: 1 on a GPU that does 2x, 3 on one that does 4x); 0 until Streamline reports it. Asked after
// the device exists and refreshed after each present. Optional: an older DLL lacks it, and the game then asks for one.
typedef uint32_t (*PFN_nx2sl_frames_max)(void);
typedef void (*PFN_nx2sl_present_marker)(int after, uint32_t frame);
// The frames the display got, generated ones included, counted after each present (on the present thread)
typedef uint64_t (*PFN_nx2sl_shown)(void);
// Queried on the present thread after the present; the SDK owns this timeline semaphore.
typedef int (*PFN_nx2sl_input_retirement)(VkSemaphore *semaphore, uint64_t *value);
// These four optional exports are negotiated together. No SDK object pointer crosses the bridge.
typedef int (*PFN_nx2sl_frame_inputs_v1)(const Nx2SlFrameInputsV1 *, Nx2SlInputReceipt *);
typedef VkResult (*PFN_nx2sl_present_inputs_v1)(PFN_vkQueuePresentKHR, VkQueue, const VkPresentInfoKHR *,
                                                const Nx2SlInputReceipt *, Nx2SlInputCompletion *);
// Positive removal only while the exact receipt is still the next unattempted present.
typedef int (*PFN_nx2sl_cancel_inputs_v1)(const Nx2SlInputReceipt *);
// Returns 1 only when the SDK is known shut down or was never initialized.
typedef int (*PFN_nx2sl_shutdown_checked_v1)(void);
// Optional independent cold-configuration estimate. The returned bytes are a rough VRAM projection.
typedef int (*PFN_nx2sl_estimate_fg_v1)(const Nx2SlFgConfigV1 *, Nx2SlFgEstimateV1 *);
// Take the same typed present's post-present FG execution observation once, without resampling SDK state.
typedef int (*PFN_nx2sl_fg_observed_v1)(uint64_t frame, const Nx2SlFgConfigV1 *config);
typedef void (*PFN_nx2sl_shutdown)(void);
#ifdef __cplusplus
}
#endif

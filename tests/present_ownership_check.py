"""Exercise WSI receipt ownership against a scripted Vulkan driver on the host."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from bind_recipe_check import function


def between(source, start, end):
    offset = source.index(start)
    return source[offset:source.index(end, offset)]


def wsi_fixture(source):
    declarations = between(source, 'typedef enum { PRESENT_NATIVE', 'static VkCommandPool g_pool')
    declarations += between(source, 'static VkTexturePresentSource *g_slot_source', 'static int g_capture')
    declarations += between(source, 'typedef enum { FLIP_NONE', 'static void present_require')
    helpers = between(source, 'static void present_require', 'static void sync_barrier')
    selected = '\n'.join(function(source, name) for name in (
        'present_slot_sdk_ready', 'present_inputs_ready', 'present_inputs_match', 'present_inputs_quarantine',
        'present_inputs_tag', 'present_inputs_abandon', 'present_inputs_flip',
        'present_config_prepare', 'vk_present_config_drain',
        'flip_locked', 'free_capture', 'wsi_images', 'prepare_swapchain_images', 'drop_swapchain_locked',
        'present_ffx_hold_hudless', 'present_ffx_finish_hudless',
        'present_retire_generation', 'present_take_completed_temporal', 'present_release_completed_temporal',
        'create_swapchain', 'vk_present_retire_temporal_inputs',
        'hudless_drop', 'ordinary_present_current_locked', 'gfx_ordinary_picture', 'gfx_ordinary_present_size', 'gfx_window_resize', 'gfx_window_release_vk', 'gfx_fit_rect',
        'put_picture'))
    # Keep ordinary blit geometry and presentation ownership. DSP/SDK recording remains modeled.
    recording = function(source, 'present_image')
    assert recording.index('present_ffx_hold_hudless(&flip, hudless)') < recording.index('vkQueueSubmit2(c->gfx_queue')
    retirement = function(source, 'present_retire_generation')
    assert retirement.index('present_ffx_finish_hudless()') < retirement.index('vk_device_wait_idle()')
    acquire = recording[:recording.index('    if (g_sc_hdr) vk_hdr_metadata')]
    submit = recording.index('    VkSemaphoreSubmitInfo wait =')
    selects = [line for line in recording.splitlines() if 'vk_dlaa_fg_compose_select(' in line]
    select, unselect = selects[0], selects[-1]
    tag = between(recording, '    if ((!src || !frame) && flip.path == PRESENT_DLSSG', '    // through the presenter:')
    crop_geometry = between(recording, '    int cx =', '    if (src && cw >')
    picture_geometry = between(recording, '    const int rect[4]', '    if (src && g_sc_ext.width')
    compose = acquire + r'''

    const VkCommandBuffer cb = HANDLE(VkCommandBuffer, 1201);
    const unsigned ng_set = 0, ng_count = 0;
    const int via_ng = 0, ng_fresh = 0, cap = 1;
    const char *menu_shot = NULL;
    const VkImage dst = g_sc_img[idx];
    uint32_t picture_effects = 0;
    const int picture_replaced = 0;
''' + crop_geometry + picture_geometry + r'''
    if (native_picture) {
        memcpy(last_picture_rect, to, sizeof last_picture_rect);
        assert(put_picture(cb, slot, src, fmt, w, h, rect, dst, to, general, &picture_effects) == picture_fsr);
    }
    VkImage hudless = mock_hudless[slot].image;
    if (hud_images) {
        assert(hudless);
        flip.hud_generation = hud_images->generation;
        flip.hud_exact = hud_images->exact;
    }
    VkNgxImage alpha = mock_alpha[slot];
    int movie = 0;
''' + select + '\n    record_picture();\n' + tag + unselect + '\n' + recording[submit:]
    return r'''
#include <assert.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk.h"
#include "vk_dlaa.h"
#include "vk_dlssg.h"
#include "vk_fg.h"
#include "resource_core_vma.h"

enum {FRAMES = 2, MAX_IMAGES = 8, NG_MAX = 3};
#define HANDLE(type, value) ((type)(uintptr_t)(value))
static int g_pl, g_started, g_sc_hdr_wanted;
static volatile int32_t g_dirty, g_sc_blocked;
static uint64_t g_sc_generation;
static VkSwapchainKHR g_sc;
static VkSurfaceKHR g_surface;
static VkCommandPool g_pool;
static VkFence g_fence[FRAMES];
static VkSemaphore g_acquire[FRAMES];
static unsigned g_frame, g_hudless_slot;
static uint64_t g_th_pending[FRAMES];
static VkBuffer g_cap_buf;
static VmaAllocation g_cap_alloc;
static uint8_t *g_cap_ptr;
static VkDeviceSize g_cap_size;
static int g_cap_w, g_cap_h, g_cap_valid;
static struct { VkImageView view; VkImage image; VmaAllocation memory; } g_hudless[FRAMES];
#define g_hl_view (g_hudless[g_hudless_slot].view)
#define g_hl (g_hudless[g_hudless_slot].image)
''' + declarations + r'''
enum {ACQUIRE, COMPOSE, SETTLE, FLIP, DRAIN, IDLE, SDK, CREATE, DROP, DROP_RENDER_DONE, CAPTURE,
      TAG, TYPED_FLIP, CANCEL, SHUTDOWN, DROP_HUD};
static unsigned trace[128], ntrace, created, destroyed, frame_slot, acquired_index, resize_width = 800, resize_height = 500;
static unsigned gfx_locked, present_locked, graphics_waited, resets, compose_count, acquire_count, flip_count, settled;
static int minimized, fatal_expected, graphics_submitted, retired;
volatile int32_t vk_lost;
static VkSwapchainKHR acquired_swapchain;
static VkSemaphore acquired_semaphore, graphics_signal;
static VkResult acquire_result, submit_result, display_result, idle_result, sdk_result, capture_result;
static VkCtx test_ctx;
static jmp_buf fatal_return;
static unsigned render_done_drops, sdk_waits, releases;
static unsigned fail_semaphore_after, new_semaphores;
static VkSwapchainKHR semaphore_owner[32], dropped_swapchains[16];
struct VkTexturePresentSource { int completed, released; GfxPresentPictureSource picture; };
GfxPresentPictureSource vk_texture_present_picture(const VkTexturePresentSource *source) {
    return source ? source->picture : (GfxPresentPictureSource){0};
}
static VkTexturePresentSource sources[3], *frontend_source;
static unsigned source_releases;
static int native_picture, picture_blits, last_picture_rect[4], mismatch_picture, picture_fsr, picture_grade;
static VkImage last_picture_source, last_picture_destination;
static VkImageBlit last_picture_copy;
struct VkDlaaFgSizeUse { int consumed, completed, released; };
static VkDlaaFgSizeUse motion[3], *frontend_motion, *selected_motion, *recorded_motion;
static VkDlaaFgSizeUse *motion_fence_owner[FRAMES];
static VkResult motion_fence_status[FRAMES];
static unsigned motion_releases, motion_polls;
static int use_motion, refuse_trylock, split_temporal;
struct VkDlaaHudSource { VkDlaaHudImages images; int completed, released; };
static VkDlaaHudSource hud_source, *frontend_hud;
static Nx2SlImage mock_hudless[FRAMES];
static VkNgxImage mock_alpha[FRAMES];
#define g_hl_fmt (mock_hudless[g_hudless_slot].format)
#define g_hl_w (mock_hudless[g_hudless_slot].width)
#define g_hl_h (mock_hudless[g_hudless_slot].height)
static uint64_t sdk_frame, sdk_ticket, expect_sdk_frame;
unsigned rt_trace;
static unsigned tagged, typed_presented, legacy_tagged, legacy_presented, legacy_retirement,
                canceled, checked_shutdown, destroyed_hudless;
static int sdk_active, receipts, invalid_tag, valid_no_tags, no_timeline, invalid_completion,
           copy_failed, cancel_fails, shutdown_fails;
static VkResult sdk_image_result;
static VkDlssgConfigResult config_result;
static unsigned config_calls, config_tags, config_accepted, config_checks, config_dropped, old_drained;
static int selected_variants, configuration_token, old_drain, external_idle, config_invalidate, invalidated_scene;
static unsigned config_unproven;
static VkPresentConfigRetry config_retry;


int vk_sync_stopping(void) { return 0; }
uint32_t vk_dlssg_config_next_generated(int have) { return have ? 1u : 0u; }
int vk_dlaa_fg_describe(const VkDlaaFgSizeUse *use, VkFfxImage *depth, VkFfxImage *motion) {
    if (!use) return 0;
    assert(!use->consumed && g_pl);
    if (invalidated_scene) return -1;
    *depth = (VkFfxImage){HANDLE(VkImage,1100),VK_FORMAT_D32_SFLOAT,320,180,0};
    *motion = (VkFfxImage){HANDLE(VkImage,1101),VK_FORMAT_R16G16_SFLOAT,320,180,0};
    return 1;
}

VkDlssgConfigResult vk_dlssg_config_preflight_variants(const Nx2SlFgConfigV1 *choices,
    uint32_t count, int have, uint64_t frame, VkDlssgConfigPreparation *prep, RcAdmissionVector *peak) {
    assert(g_pl && !gfx_locked && !present_locked && frame == sdk_frame);
    assert(choices[0].swapchain == g_sc && choices[0].generation == g_sc_generation &&
           choices[0].color == g_sc_fmt && choices[0].backbuffers == g_sc_n);
    assert(choices[0].generated == (unsigned)have && choices[0].alpha == VK_FORMAT_UNDEFINED);
    if (count > 1) {
        assert(choices[1].hudless == g_sc_fmt && choices[1].alpha == VK_FORMAT_UNDEFINED);
        if (count == 3) assert(choices[2].hudless == g_sc_fmt && choices[2].alpha == VK_FORMAT_R16_SFLOAT);
    }
    selected_variants = (int)count;
    config_calls++;
    peak->amount[0] = 4096;
    prep->choice[0].key = choices[0];
    prep->count = count;
    prep->frame = frame;
    prep->generated = choices[0].generated;
    prep->use = (uint64_t)configuration_token;
    prep->choice[0].result = config_result;
    return config_result;
}

void vk_dlssg_config_cancel(VkDlssgConfigPreparation *prep) { memset(prep, 0, sizeof *prep); }
void vk_dlssg_config_abandon(uint64_t use) { if (use) config_dropped++; }
int vk_dlssg_config_drain_eligible(uint64_t *use, VkFence *graphics) {
    assert(g_pl && !gfx_locked && !present_locked);
    if (!old_drain || old_drained) return 0;
    *use = 123;
    *graphics = g_fence[1];
    return 1;
}
void vk_dlssg_config_progress(VkDevice device, int checked_idle) {
    assert(device == test_ctx.device && g_pl && !gfx_locked && !present_locked);
    config_checks++;
    if (old_drain && checked_idle) old_drained++;
}
void vk_dlaa_present_alpha_unrecorded(unsigned slot) {
    assert(g_pl && slot < FRAMES && !graphics_submitted);
    config_dropped++;
}
void vk_dlssg_config_presented(uint64_t use, uint64_t frame, VkFence graphics, VkResult r, VkResult image) {
    assert(g_pl && !gfx_locked && !present_locked && use == (uint64_t)configuration_token &&
           frame == expect_sdk_frame && graphics == g_fence[frame_slot]);
    if (r == VK_SUCCESS && image == VK_SUCCESS) config_accepted++;
    else config_unproven++;
}

static void mark(unsigned event) { assert(ntrace < sizeof trace / sizeof *trace); trace[ntrace++] = event; }
static void expect(unsigned *position, unsigned event) { assert(*position < ntrace && trace[(*position)++] == event); }
static void check_no_driver_progress(unsigned position) { assert(position == ntrace); }
VkCtx *vk_ctx(void) { return &test_ctx; }
uint64_t kern_now_ns(void) { static uint64_t now; return ++now; }
void rt_log(const char *format, ...) { (void)format; }
void rt_fatal(const char *format, ...) {
    (void)format;
    if (fatal_expected) longjmp(fatal_return, 1);
    abort();
}
void vk_device_lost_check(VkResult r, const char *where) { (void)where; if (r == VK_ERROR_DEVICE_LOST) vk_lost = 1; }
void vk_texture_present_release(VkTexturePresentSource *s) {
    assert(!g_pl && s && s->completed && !s->released);
    s->released = 1;
    source_releases++;
}
const VkDlaaHudImages *vk_dlaa_hud_source_images(const VkDlaaHudSource *use, uint64_t frame) {
    assert(g_pl && (use == NULL || (use == &hud_source && use->images.frame == frame && !use->released)));
    return use ? &use->images : NULL;
}
void vk_dlaa_hud_source_release(VkDlaaHudSource *use) {
    if (!use) return;
    assert(!g_pl && !gfx_locked && !present_locked && use->completed && !use->released);
    use->released = 1;
}
void vk_dlaa_fg_compose_select(VkDlaaFgSizeUse *use) {
    assert(g_pl && (!use || !selected_motion));
    selected_motion = use;
}
int vk_dlaa_fg_compose_consumed(const VkDlaaFgSizeUse *use) {
    assert(g_pl && use);
    return use->consumed;
}
void vk_dlaa_fg_release(VkDlaaFgSizeUse *use) {
    if (!use) return;
    assert(!g_pl && !gfx_locked && !present_locked && use->completed && !use->released);
    use->released = 1;
    motion_releases++;
}
const char *cfg_value(const char *key) {
    assert(!strcmp(key, "NX2NAVI_ASPECT"));
    return "fit";
}
VkImage vk_hdr_encode(VkCommandBuffer cb, unsigned slot, VkImage image, VkFormat format, const int crop[4]) {
    (void)cb; (void)slot; (void)image; (void)format; (void)crop;
    abort(); // This test uses the ordinary SDR blit.
}
VkImage vk_color_record(VkCommandBuffer cb, unsigned slot, VkImage image, VkFormat format, const int crop[4]) {
    assert(cb && slot < FRAMES && image && format == VK_FORMAT_R8G8B8A8_UNORM && crop);
    return picture_grade ? HANDLE(VkImage, 777) : VK_NULL_HANDLE;
}
int vk_fsr_record(VkCommandBuffer cb, unsigned slot, VkImage image, VkFormat format, int sw, int sh,
                  const int crop[4], VkImage dst, const int out[4]) {
    assert(cb && slot < FRAMES && image &&
           format == (picture_grade ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM));
    assert(!picture_grade || image == HANDLE(VkImage, 777));
    assert(sw > 0 && sh > 0 && crop && dst && out);
    return picture_fsr;
}
static void sync_barrier(VkCommandBuffer cb, VkImage image, VkImageLayout old, VkImageLayout new,
                         VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                         VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    (void)src_access; (void)dst_access;
    assert(cb && image && old == VK_IMAGE_LAYOUT_GENERAL && new == old && src_stage && dst_stage);
}
void vkCmdBlitImage(VkCommandBuffer cb, VkImage src, VkImageLayout src_layout, VkImage dst,
                    VkImageLayout dst_layout, uint32_t count, const VkImageBlit *regions, VkFilter filter) {
    assert(native_picture && cb && src && dst && count == 1 && regions && filter == VK_FILTER_LINEAR);
    assert(src_layout == VK_IMAGE_LAYOUT_GENERAL && dst_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    last_picture_source = src;
    last_picture_destination = dst;
    last_picture_copy = *regions;
    picture_blits++;
}
static void record_picture(void) {
    // Stand in for the SDK consuming the selected motion image while recording graphics.
    if (selected_motion && use_motion) {
        selected_motion->consumed = 1;
        recorded_motion = selected_motion;
    }
}
void vk_queue_lock(void) { assert(!present_locked && !gfx_locked); gfx_locked = 1; }
void vk_queue_unlock(void) { assert(!present_locked && gfx_locked); gfx_locked = 0; }
void vk_present_queue_lock(void) { assert(!gfx_locked && !present_locked); present_locked = 1; }
void vk_present_queue_unlock(void) { assert(!gfx_locked && present_locked); present_locked = 0; }
static void plat_rmutex_lock(int *m) { assert(m == &g_pl && !*m); *m = 1; }
static void plat_rmutex_unlock(int *m) { assert(m == &g_pl && *m); *m = 0; }
static int plat_rmutex_trylock(int *m) {
    if (refuse_trylock) return 0;
    plat_rmutex_lock(m);
    return 1;
}
static int32_t plat_xchg32(volatile int32_t *p, int32_t n) { int32_t old = *p; *p = n; return old; }
static int32_t plat_cas32(volatile int32_t *p, int32_t n, int32_t expect) {
    int32_t old = *p;
    if (old == expect) *p = n;
    return old;
}
static int32_t plat_inc32(volatile int32_t *p) { return ++*p; }
void gfx_window_start(void) { assert(g_started); }
int wsys_minimized(void) { return minimized; }
void wsys_client_size(int *w, int *h) { *w = (int)resize_width; *h = (int)resize_height; }
void wsys_resize(int w, int h) { resize_width = (unsigned)w; resize_height = (unsigned)h; }
static VkPresentModeKHR pick_present_mode(VkPhysicalDevice p) { assert(p == test_ctx.phys); return VK_PRESENT_MODE_FIFO_KHR; }
static int present_split(void) { return test_ctx.present_queue != test_ctx.gfx_queue; }
static int present_slot_ready(unsigned slot) {
    assert(split_temporal && g_pl && slot == g_frame % FRAMES);
    return 1;
}
static int present_slot_image(unsigned slot) {
    assert(split_temporal && g_pl && slot == g_frame % FRAMES);
    return 1;
}
static int ng_ready(void) { return 0; }
static struct { int fresh; uint64_t n; VkSemaphore copied, rendered; } ng;
static void ng_wait_signaled(uint64_t n) { (void)n; abort(); }
static void ng_hand(uint32_t count, int set, uint64_t n) {
    (void)count; (void)set; (void)n; abort();
}
static void ng_drain(void) { assert(g_pl && !gfx_locked && !present_locked); mark(DRAIN); }
static void teardown_dependency(void) { assert(retired && g_pl); releases++; }
static void ng_release(void) { teardown_dependency(); }
static void menu_vk_release(void) { teardown_dependency(); }
void vk_fsr_release(void) { teardown_dependency(); }
void vk_color_release(void) { teardown_dependency(); }
void vk_hdr_release(void) { teardown_dependency(); }
void vk_contact_release(void) { teardown_dependency(); }
static void free_thumbs(void) { teardown_dependency(); }
static void save_capture(const char *filename) { (void)filename; abort(); }
void resource_core_vma_destroy_image(VmaAllocator a, VkImage image, VmaAllocation memory) {
    assert(g_pl && retired && checked_shutdown && !shutdown_fails && a == test_ctx.vma);
    unsigned slot = image == mock_hudless[0].image ? 0 : 1;
    assert(image == mock_hudless[slot].image && memory == g_hudless[slot].memory);
    assert(g_present_sdk_inputs[slot].state == SDK_INPUTS_FREE);
    mark(DROP_HUD);
    destroyed_hudless++;
}
void vmaDestroyBuffer(VmaAllocator a, VkBuffer buffer, VmaAllocation memory) {
    (void)a; (void)buffer; (void)memory; abort();
}
VkResult vmaInvalidateAllocation(VmaAllocator a, VmaAllocation m, VkDeviceSize offset, VkDeviceSize size) {
    assert(g_pl && graphics_submitted && a == test_ctx.vma && m == g_cap_alloc && !offset && size == VK_WHOLE_SIZE);
    mark(CAPTURE);
    return VK_SUCCESS;
}
int vk_hdr_wanted(void) { return 0; }
int vk_hdr_surface_format(const VkSurfaceFormatKHR *f, uint32_t n, VkSurfaceFormatKHR *out) {
    (void)f; (void)n; (void)out; abort();
}
void vk_hdr_set_active(int on) { assert(!on); }
void vk_hdr_metadata(VkSwapchainKHR sc, int init) { (void)sc; (void)init; abort(); }
int vk_ngxfg_wanted(void) { return 0; }
int vk_fg_wanted(void) { return 0; }
int vk_fg_active(void) { return 0; }
static VkResult ffx_finish_result;
static unsigned ffx_finishes;
VkResult vk_fg_finish_inputs(VkSwapchainKHR sc) {
    assert(g_pl && !gfx_locked && !present_locked && sc == g_sc && g_sc_path == PRESENT_FFX);
    ffx_finishes++;
    return ffx_finish_result;
}
int vk_dlssg_active(void) { return sdk_active; }
int vk_dlssg_has_input_receipts(void) { return sdk_active && receipts; }
uint64_t vk_dlssg_current_frame(void) { return sdk_frame; }
void vk_dlssg_present_frame(uint64_t frame) { sdk_frame = frame; }
static void inspect_sdk_images(const Nx2SlImage *hl, const Nx2SlImage *alpha) {
    unsigned slot = g_frame % FRAMES;
    assert(hl && alpha && hl->image == mock_hudless[slot].image &&
           alpha->image == mock_alpha[slot].img);
    assert(hl->view == mock_hudless[slot].view && alpha->view == mock_alpha[slot].view &&
           hl->width == g_sc_ext.width && alpha->height == g_sc_ext.height &&
           hl->layout == VK_IMAGE_LAYOUT_GENERAL && alpha->layout == VK_IMAGE_LAYOUT_GENERAL &&
           hl->format == g_sc_fmt && alpha->format == VK_FORMAT_R16_SFLOAT);
}
int vk_dlssg_frame_inputs(VkCommandBuffer cb, VkExtent2D extent, const int rect[4],
                          const Nx2SlImage *hl, const Nx2SlImage *alpha, uint64_t frame,
                          VkSwapchainKHR sc, uint64_t generation, Nx2SlInputReceipt *out) {
    assert(g_pl && !gfx_locked && !present_locked && vk_dlssg_has_input_receipts());
    assert(cb == HANDLE(VkCommandBuffer, 1201) && rect[2] == (int)extent.width && frame == sdk_frame);
    inspect_sdk_images(hl, alpha);
    assert(sc == g_sc && generation == g_sc_generation);
    mark(TAG);
    tagged++;
    *out = (Nx2SlInputReceipt){.size = sizeof *out, .version = NX2SL_INPUTS_ABI_V1,
        .frame = frame, .sdk_frame = (uint32_t)frame, .generation = generation, .swapchain = sc};
    if (valid_no_tags) return 1;
    if (invalid_tag) {
        out->uncertain = NX2SL_INPUT_HUDLESS | NX2SL_INPUT_UI_ALPHA;
        return 0;
    }
    out->ticket = ++sdk_ticket;
    out->accepted = NX2SL_INPUT_HUDLESS | NX2SL_INPUT_UI_ALPHA;
    return 1;
}
int vk_dlssg_ui_alpha(void) { return sdk_active; }
VkDlssgConfigResult vk_dlssg_frame_inputs_admitted(VkCommandBuffer cb,VkExtent2D ext,VkFormat format,
    uint32_t backbuffers,const int rect[4],const Nx2SlImage *hudless,const Nx2SlImage *alpha,uint64_t frame,
    VkSwapchainKHR sc,uint64_t generation,VkDlssgConfigPreparation *prepared,Nx2SlInputReceipt *r,
    uint64_t *use,RcAdmissionVector *peak,int *sdk_tag_ok) {
    (void)peak;
    assert(prepared->choice[0].key.generation == generation && prepared->choice[0].key.swapchain == sc &&
           prepared->choice[0].key.color == format && prepared->choice[0].key.backbuffers == backbuffers);
    config_tags++;
    if (config_invalidate) return VK_DLSSG_CONFIG_INVALIDATED;
    *use = prepared->use;
    memset(prepared,0,sizeof *prepared);
    *sdk_tag_ok = vk_dlssg_frame_inputs(cb,ext,rect,hudless,alpha,frame,sc,generation,r);
    return config_result;
}

void vk_dlssg_frame(VkCommandBuffer cb, VkExtent2D extent, const int rect[4], VkImage hl, VkImageView view,
                    VkFormat f, VkImage alpha, VkImageView av) {
    (void)cb; (void)extent; (void)rect; (void)hl; (void)view; (void)f; (void)alpha; (void)av;
    assert(!vk_dlssg_has_input_receipts());
    if (sdk_active) legacy_tagged++;
}
int vk_dlssg_cancel_inputs(const Nx2SlInputReceipt *r) {
    assert(g_pl && !gfx_locked && !present_locked && r->ticket == sdk_ticket && r->generation == g_sc_generation);
    mark(CANCEL);
    canceled++;
    return !cancel_fails;
}
int vk_dlssg_release_checked(void) {
    assert(g_pl && retired && !gfx_locked && !present_locked && !destroyed_hudless && !g_sc);
    mark(SHUTDOWN);
    checked_shutdown++;
    if (shutdown_fails) return 0;
    sdk_active = 0;
    return 1;
}
VkResult vk_fg_acquire(VkSwapchainKHR sc, uint64_t ns, VkSemaphore s, uint32_t *i) {
    (void)sc; (void)ns; (void)s; (void)i; abort();
}
VkResult vk_dlssg_acquire(VkDevice d, VkSwapchainKHR sc, uint64_t ns, VkSemaphore s, uint32_t *i) {
    assert(sdk_active);
    return vkAcquireNextImageKHR(d, sc, ns, s, VK_NULL_HANDLE, i);
}
VkResult vk_fg_present(const VkPresentInfoKHR *pi) { (void)pi; abort(); }
static void inspect_sdk_flip(VkQueue q, const VkPresentInfoKHR *pi) {
    assert(g_pl && sdk_active && q == test_ctx.present_queue && pi->swapchainCount == 1);
    assert((q == test_ctx.gfx_queue && gfx_locked && !present_locked) ||
           (q != test_ctx.gfx_queue && present_locked && !gfx_locked));
    assert(*pi->pSwapchains == g_sc && pi->waitSemaphoreCount == 1 &&
           *pi->pWaitSemaphores == g_render_done[acquired_index] && *pi->pImageIndices == acquired_index);
}
VkResult vk_dlssg_present(VkQueue q, const VkPresentInfoKHR *pi) {
    inspect_sdk_flip(q, pi);
    assert(sdk_frame == expect_sdk_frame && !pi->pResults);
    legacy_presented++;
    return display_result;
}
VkResult vk_dlssg_present_inputs(VkQueue q, const VkPresentInfoKHR *pi, const Nx2SlInputReceipt *r,
                                 Nx2SlInputCompletion *out) {
    inspect_sdk_flip(q, pi);
    assert(pi->pResults && r->frame == expect_sdk_frame && r->generation == g_sc_generation &&
           r->swapchain == g_sc && r->ticket == sdk_ticket);
    mark(TYPED_FLIP);
    typed_presented++;
    if (!invalid_completion) *pi->pResults = sdk_image_result;
    *out = (Nx2SlInputCompletion){.size = sizeof *out, .version = NX2SL_INPUTS_ABI_V1,
        .ticket = r->ticket + (invalid_completion ? 1 : 0), .flags = NX2SL_INPUT_NO_LATE_TAGS};
    if (r->ticket && !no_timeline) {
        out->flags = NX2SL_INPUT_TIMELINE;
        out->semaphore = HANDLE(VkSemaphore, 117);
        out->value = 53;
    }
    return display_result;
}
int vk_dlssg_input_retirement(VkSemaphore *s, uint64_t *v) {
    assert(sdk_active && !receipts && present_locked && legacy_presented == 1);
    *s = HANDLE(VkSemaphore, 117);
    *v = 53;
    legacy_retirement++;
    return 1;
}
VkResult vk_fg_create_swapchain(const VkSwapchainCreateInfoKHR *ci, VkSwapchainKHR *sc) { (void)ci; (void)sc; abort(); }
VkResult vk_dlssg_create_swapchain(VkDevice d, const VkSwapchainCreateInfoKHR *ci, VkSwapchainKHR *sc) {
    assert(sdk_active && !checked_shutdown);
    return vkCreateSwapchainKHR(d, ci, NULL, sc);
}
VkResult vk_fg_get_images(VkSwapchainKHR sc, uint32_t *n, VkImage *i) { (void)sc; (void)n; (void)i; abort(); }
VkResult vk_dlssg_get_images(VkDevice d, VkSwapchainKHR sc, uint32_t *n, VkImage *i) {
    assert(sdk_active && !checked_shutdown);
    return vkGetSwapchainImagesKHR(d, sc, n, i);
}
void vk_fg_destroy_swapchain(VkSwapchainKHR sc) { (void)sc; abort(); }
void vk_dlssg_destroy_swapchain(VkDevice d, VkSwapchainKHR sc) {
    assert(sdk_active && !checked_shutdown && !destroyed_hudless);
    vkDestroySwapchainKHR(d, sc, NULL);
}
void vk_fg_release(void) { abort(); }
VKAPI_ATTR VkResult VKAPI_CALL vkAcquireNextImageKHR(VkDevice d, VkSwapchainKHR sc, uint64_t ns,
                                                     VkSemaphore sem, VkFence fence, uint32_t *i) {
    assert(d == test_ctx.device && sc == g_sc && g_pl && !g_sc_blocked && !gfx_locked && !present_locked);
    assert(ns && !fence && sem == g_acquire[g_frame % FRAMES]);
    mark(ACQUIRE);
    acquire_count++;
    if (acquire_result == VK_SUCCESS || acquire_result == VK_SUBOPTIMAL_KHR) {
        acquired_swapchain = sc;
        acquired_semaphore = sem;
        *i = acquired_index;
    }
    return acquire_result;
}
VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(VkDevice d, uint32_t n, const VkFence *f, VkBool32 all, uint64_t ns) {
    assert(d == test_ctx.device && n == 1 && all && ns);
    if (external_idle && !g_pl) {
        assert(*f == g_fence[1]);
        return VK_SUCCESS;
    }
    assert(g_pl);
    if (graphics_submitted) {
        assert(*f == g_fence[frame_slot] && resets == compose_count);
        return capture_result;
    }
    frame_slot = g_frame % FRAMES;
    assert(*f == g_fence[frame_slot]);
    if (g_slot_source[frame_slot]) g_slot_source[frame_slot]->completed = 1;
    if (g_slot_hud[frame_slot]) g_slot_hud[frame_slot]->completed = 1;
    if (motion_fence_owner[frame_slot]) motion_fence_owner[frame_slot]->completed = 1;
    graphics_waited++;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetFenceStatus(VkDevice d, VkFence fence) {
    assert(g_pl && d == test_ctx.device && !gfx_locked && !present_locked);
    unsigned slot = (unsigned)(uintptr_t)fence - 91;
    assert(slot < FRAMES && fence == g_fence[slot] && (motion_fence_owner[slot] || g_slot_hud[slot]));
    motion_polls++;
    VkResult r = motion_fence_status[slot];
    if (r == VK_SUCCESS) {
        if (motion_fence_owner[slot]) motion_fence_owner[slot]->completed = 1;
        if (g_slot_hud[slot]) g_slot_hud[slot]->completed = 1;
    }
    return r;
}
VKAPI_ATTR VkResult VKAPI_CALL vkResetFences(VkDevice d, uint32_t n, const VkFence *f) {
    assert(g_pl && d == test_ctx.device && n == 1 && *f == g_fence[frame_slot] && graphics_waited > resets);
    resets++;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit2(VkQueue q, uint32_t n, const VkSubmitInfo2 *si, VkFence f) {
    assert(g_pl && gfx_locked && q == test_ctx.gfx_queue && n == 1);
    if (config_invalidate && !si->commandBufferInfoCount) {
        assert(!f && g_sc_blocked && si->waitSemaphoreInfoCount == 1 &&
               si->pWaitSemaphoreInfos[0].semaphore == acquired_semaphore);
        settled++;
        return VK_SUCCESS;
    }
    assert(f == g_fence[frame_slot]);
    if (split_temporal) assert(!si->waitSemaphoreInfoCount);
    else assert(si->waitSemaphoreInfoCount == 1 && si->pWaitSemaphoreInfos[0].semaphore == acquired_semaphore);
    if (!si->commandBufferInfoCount) {
        assert(g_sc_blocked && submit_result == VK_ERROR_OUT_OF_DEVICE_MEMORY && !si->signalSemaphoreInfoCount);
        mark(SETTLE);
        settled++;
        return VK_SUCCESS;
    }
    assert(si->pCommandBufferInfos[0].commandBuffer == HANDLE(VkCommandBuffer, 1201));
    assert(si->signalSemaphoreInfoCount == 1);
    if (split_temporal) {
        assert(si->pSignalSemaphoreInfos[0].semaphore == g_snapshot_ready);
        assert(si->pSignalSemaphoreInfos[0].value == g_present_slots[frame_slot].ready_value);
        assert(si->pSignalSemaphoreInfos[0].value);
    } else {
        assert(!si->pSignalSemaphoreInfos[0].value);
        assert(si->pSignalSemaphoreInfos[0].semaphore == g_render_done[acquired_index]);
    }
    mark(COMPOSE);
    compose_count++;
    graphics_signal = si->pSignalSemaphoreInfos[0].semaphore;
    graphics_submitted = submit_result == VK_SUCCESS;
    if (recorded_motion && submit_result != VK_ERROR_OUT_OF_DEVICE_MEMORY)
        motion_fence_owner[frame_slot] = recorded_motion;
    retired = 0;
    return submit_result;
}
VKAPI_ATTR VkResult VKAPI_CALL vkQueuePresentKHR(VkQueue q, const VkPresentInfoKHR *pi) {
    assert(g_pl && !gfx_locked && present_locked && q == test_ctx.present_queue);
    assert(pi->waitSemaphoreCount == 1 && *pi->pWaitSemaphores == graphics_signal);
    assert(pi->swapchainCount == 1 && *pi->pSwapchains == acquired_swapchain && *pi->pImageIndices == acquired_index);
    mark(FLIP);
    flip_count++;
    return display_result;
}
VKAPI_ATTR VkResult VKAPI_CALL vkQueueWaitIdle(VkQueue q) { (void)q; abort(); }
VkResult vk_device_wait_idle(void) {
    assert((g_pl || external_idle) && !gfx_locked && !present_locked);
    mark(IDLE);
    retired = idle_result == VK_SUCCESS;
    if (retired) {
        for (unsigned i = 0; i < FRAMES; ++i) {
            if (g_slot_source[i]) g_slot_source[i]->completed = 1;
            if (g_slot_hud[i]) g_slot_hud[i]->completed = 1;
            if (motion_fence_owner[i]) motion_fence_owner[i]->completed = 1;
        }
    }
    return idle_result;
}
VKAPI_ATTR VkResult VKAPI_CALL vkWaitSemaphores(VkDevice d, const VkSemaphoreWaitInfo *wi, uint64_t ns) {
    assert(g_pl && (retired || vk_dlssg_has_input_receipts()) && d == test_ctx.device &&
           wi->semaphoreCount == 1 && ns);
    assert(*wi->pSemaphores == HANDLE(VkSemaphore, 117) && *wi->pValues == 53);
    mark(SDK);
    sdk_waits++;
    return sdk_result;
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceCapabilitiesKHR(VkPhysicalDevice p, VkSurfaceKHR s,
                                                                        VkSurfaceCapabilitiesKHR *caps) {
    assert(p == test_ctx.phys && s == g_surface);
    *caps = (VkSurfaceCapabilitiesKHR){.currentExtent = {UINT32_MAX, UINT32_MAX}, .minImageCount = 2};
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceFormatsKHR(VkPhysicalDevice p, VkSurfaceKHR s,
                                                                   uint32_t *n, VkSurfaceFormatKHR *fo) {
    assert(p == test_ctx.phys && s == g_surface && n);
    *n = 1;
    if (fo) *fo = (VkSurfaceFormatKHR){VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateSwapchainKHR(VkDevice d, const VkSwapchainCreateInfoKHR *ci,
                                                   const VkAllocationCallbacks *a, VkSwapchainKHR *out) {
    assert(g_pl && gfx_locked && retired && d == test_ctx.device && !a);
    assert(ci->oldSwapchain == g_sc && ci->surface == g_surface && ci->minImageCount == 3);
    assert(ci->imageExtent.width == resize_width && ci->imageExtent.height == resize_height);
    assert(g_present_slots[0].retire_value == 0);
    mark(CREATE);
    *out = HANDLE(VkSwapchainKHR, 400 + ++created);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetSwapchainImagesKHR(VkDevice d, VkSwapchainKHR sc, uint32_t *n, VkImage *out) {
    assert(g_pl && gfx_locked && d == test_ctx.device && sc == g_sc);
    assert(!out || *n == 3);
    *n = 3;
    if (out)
        for (unsigned i = 0; i < *n; ++i) out[i] = HANDLE(VkImage, (uintptr_t)sc * 10 + i);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateSemaphore(VkDevice d, const VkSemaphoreCreateInfo *ci,
                                                 const VkAllocationCallbacks *a, VkSemaphore *out) {
    static uintptr_t ordinal = 500;
    assert(g_pl && gfx_locked && d == test_ctx.device && ci && !a);
    if (fail_semaphore_after && --fail_semaphore_after == 0) {
        *out = HANDLE(VkSemaphore, 9999); // error must not grant ownership of the returned value
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    assert(new_semaphores < sizeof semaphore_owner / sizeof *semaphore_owner);
    semaphore_owner[new_semaphores++] = g_sc;
    *out = HANDLE(VkSemaphore, ++ordinal);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(VkDevice d, const VkImageViewCreateInfo *ci,
                                                 const VkAllocationCallbacks *a, VkImageView *out) {
    assert(g_pl && gfx_locked && d == test_ctx.device && !a && ci->image);
    *out = HANDLE(VkImageView, (uintptr_t)ci->image + 4000);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(VkDevice d, VkImageView v, const VkAllocationCallbacks *a) {
    assert(g_pl && retired && d == test_ctx.device && v && !a);
}
VKAPI_ATTR void VKAPI_CALL vkDestroySemaphore(VkDevice d, VkSemaphore s, const VkAllocationCallbacks *a) {
    assert(g_pl && retired && d == test_ctx.device && !a);
    if ((uintptr_t)s > 500) {
        unsigned ordinal = (unsigned)((uintptr_t)s - 501);
        assert(ordinal < new_semaphores && semaphore_owner[ordinal]);
        unsigned match = 0;
        while (match < destroyed && dropped_swapchains[match] != semaphore_owner[ordinal]) match++;
        assert(match < destroyed);
        semaphore_owner[ordinal] = VK_NULL_HANDLE;
        render_done_drops++;
        mark(DROP_RENDER_DONE);
    }
}
VKAPI_ATTR void VKAPI_CALL vkDestroyFence(VkDevice d, VkFence f, const VkAllocationCallbacks *a) {
    (void)f;
    assert(g_pl && retired && d == test_ctx.device && !a);
}
VKAPI_ATTR void VKAPI_CALL vkDestroyCommandPool(VkDevice d, VkCommandPool p, const VkAllocationCallbacks *a) {
    assert(g_pl && retired && d == test_ctx.device && p == g_pool && !a);
}
VKAPI_ATTR void VKAPI_CALL vkDestroySwapchainKHR(VkDevice d, VkSwapchainKHR sc, const VkAllocationCallbacks *a) {
    assert(g_pl && retired && d == test_ctx.device && sc == g_sc && !a);
    mark(DROP);
    assert(destroyed < sizeof dropped_swapchains / sizeof *dropped_swapchains);
    dropped_swapchains[destroyed++] = sc;
}
VKAPI_ATTR void VKAPI_CALL vkDestroySurfaceKHR(VkInstance i, VkSurfaceKHR s, const VkAllocationCallbacks *a) {
    assert(g_pl && retired && i == test_ctx.instance && s == g_surface && !a);
}
''' + helpers + r'''
static int present_snapshot_copy(PendingFlip *p) {
    assert(g_pl && p->phase == FLIP_COPY && p->queue != test_ctx.gfx_queue);
    if (copy_failed) {
        wsi_block_generation();
        return 0;
    }
    wsi_bind_image(p, acquired_index);
    p->phase = FLIP_READY;
    return 1;
}
static void present_slot_drop(struct PresentSlot *p) { (void)p; teardown_dependency(); }
''' + selected + compose + function(source, 'vk_present_blit') + r'''
static void setup(void) {
    test_ctx.device = HANDLE(VkDevice, 21);
    test_ctx.gfx_queue = HANDLE(VkQueue, 22);
    test_ctx.present_queue = HANDLE(VkQueue, 23);
    test_ctx.phys = HANDLE(VkPhysicalDevice, 24);
    test_ctx.instance = HANDLE(VkInstance, 25);
    g_surface = HANDLE(VkSurfaceKHR, 71);
    g_pool = HANDLE(VkCommandPool, 75);
    for (unsigned i = 0; i < FRAMES; ++i) {
        g_acquire[i] = HANDLE(VkSemaphore, 101 + i);
        g_fence[i] = HANDLE(VkFence, 91 + i);
    }
    (void)g_scaler;
    g_started = 1;
    gfx_window_resize(800, 500);
    assert(g_sc_generation == 1 && !g_sc_blocked && created == 1 && !destroyed);
    assert(!g_sc_hdr && !g_sc_hdr_wanted && g_sc_fmt == VK_FORMAT_R8G8B8A8_UNORM && !g_cap_ptr && !g_cap_size);
    acquired_index = 2;
    ntrace = 0;
}

static VkPresentSourceHandoff compose_with_motion(unsigned source_index, VkDlaaFgSizeUse **temporal) {
    graphics_submitted = 0;
    g_cap_valid = 0;
    recorded_motion = NULL;
    frontend_source = &sources[source_index];
    const int resized_picture = native_picture && source_index == 1;
    const int crop[4] = {16, 8, 608, 342};
    VkImage image = HANDLE(VkImage, resized_picture ? 72 : 71);
    frontend_source->picture = (GfxPresentPictureSource){
        .window_texture_serial = 7 + source_index,
        .source_texture_serial = resized_picture ? 0 : 7 + source_index,
        .window_image = 71, .image = mismatch_picture ? 999 : (uintptr_t)image,
        .width = resized_picture ? 640 : 320, .height = resized_picture ? 360 : 180,
        .format = VK_FORMAT_R8G8B8A8_UNORM, .handed_scene = 65 + source_index, .handed_ordinal = 804};
    VkPresentSourceHandoff result = vk_present_blit(image, VK_IMAGE_LAYOUT_GENERAL,
                                                   VK_FORMAT_R8G8B8A8_UNORM,
                                                   resized_picture ? 640 : 320, resized_picture ? 360 : 180,
                                                   resized_picture ? crop : NULL,
                                                   sdk_frame, &frontend_source, temporal, &frontend_hud, &config_retry);
    assert(!selected_motion);
    return result;
}

static VkPresentSourceHandoff compose_frame(unsigned source_index) { return compose_with_motion(source_index, NULL); }

static void resize_and_release(void) {
    native_picture = 1;
    assert(compose_frame(0) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    assert(picture_blits == 1 && last_picture_source == HANDLE(VkImage, 71) &&
           last_picture_destination == g_sc_img[2]);
    assert(last_picture_copy.srcOffsets[1].x == 320 && last_picture_copy.srcOffsets[1].y == 180);
    assert(last_picture_rect[0] == 0 && last_picture_rect[1] == 25 &&
           last_picture_rect[2] == 800 && last_picture_rect[3] == 450);
    assert(last_picture_copy.dstOffsets[0].y == 25 && last_picture_copy.dstOffsets[1].y == 475);
    VkSwapchainKHR first_swapchain = g_sc;
    VkSemaphore first_done = g_flip.render_done;
    assert(g_flip.phase == FLIP_READY && g_flip.generation == 1 && g_flip.swapchain == first_swapchain);
    assert(g_flip.queue == test_ctx.present_queue && g_flip.slot == 0 && g_flip.index == 2);
    assert(first_done == g_render_done[2] && g_cap_valid && g_cap_w == 800 && g_cap_h == 500);
    assert(!frontend_source && g_slot_source[0] == &sources[0] && !source_releases);
    unsigned at = 0;
    expect(&at, ACQUIRE);
    expect(&at, COMPOSE);
    expect(&at, CAPTURE);
    check_no_driver_progress(at);
    gfx_window_resize(960, 540);
    expect(&at, FLIP);
    expect(&at, DRAIN);
    expect(&at, IDLE);
    expect(&at, CREATE);
    expect(&at, DROP);
    for (unsigned i = 0; i < 3; ++i) expect(&at, DROP_RENDER_DONE);
    check_no_driver_progress(at);
    assert(g_sc != first_swapchain && g_sc_generation == 2 && g_flip.phase == FLIP_NONE);
    assert(g_sc_ext.width == 960 && g_sc_ext.height == 540 && !g_cap_valid);
    assert(created == 2 && destroyed == 1 && flip_count == 1);
    acquired_index = 0;
    assert(compose_frame(1) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    assert(g_flip.slot == 1 && g_flip.index == 0 && g_flip.render_done != first_done);
    assert(g_flip.swapchain == g_sc && g_flip.generation == 2 && g_cap_w == 960 && g_cap_h == 540);
    assert(picture_blits == 2 && last_picture_source == HANDLE(VkImage, 72) &&
           last_picture_destination == g_sc_img[0]);
    assert(last_picture_copy.srcOffsets[0].x == 16 && last_picture_copy.srcOffsets[0].y == 8 &&
           last_picture_copy.srcOffsets[1].x == 624 && last_picture_copy.srcOffsets[1].y == 350);
    assert(last_picture_copy.dstOffsets[0].x == 0 && last_picture_copy.dstOffsets[0].y == 0 &&
           last_picture_copy.dstOffsets[1].x == 960 && last_picture_copy.dstOffsets[1].y == 540);
    assert(g_slot_source[0] == &sources[0] && g_slot_source[1] == &sources[1] && !source_releases);
    plat_rmutex_lock(&g_pl);
    assert(flip_locked() && !flip_locked());
    plat_rmutex_unlock(&g_pl);
    assert(flip_count == 2);
    acquired_index = 1;
    assert(compose_frame(2) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    assert(source_releases == 1 && sources[0].released && !sources[1].released && !sources[2].released);
    assert(g_slot_source[0] == &sources[2] && g_slot_source[1] == &sources[1]);
    gfx_window_release_vk();
    assert(!g_started && !g_sc && !g_pl && destroyed == 2 && render_done_drops == 6);
    assert(source_releases == 3 && !g_slot_source[0] && !g_slot_source[1]);
    at = ntrace;
    unsigned dependencies = releases;
    gfx_window_release_vk();
    assert(releases == dependencies);
    check_no_driver_progress(at);
}

static void resize_failure_and_recovery(void) {
    GfxOrdinaryPresentSize size;
    assert(!gfx_ordinary_present_size(&size) && !size.generation && !size.frame);
    sdk_frame = 17;
    assert(compose_frame(0) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    assert(!gfx_ordinary_present_size(&size)); // the graphics submission is not a present
    plat_rmutex_lock(&g_pl);
    assert(flip_locked());
    plat_rmutex_unlock(&g_pl);
    assert(gfx_ordinary_present_size(&size) && size.generation == 1 && size.frame == 17 &&
           size.guest_texture_serial == 7 &&
           size.source_width == 320 && size.source_height == 180 &&
           size.output_width == 800 && size.output_height == 500);
    minimized = 1;
    assert(!gfx_ordinary_present_size(&size) && !size.generation);
    minimized = 0;

    // Failed candidate after the old WSI and one of the new image semaphores exist.
    fail_semaphore_after = 2;
    gfx_window_resize(960, 540);
    assert(!g_sc && g_sc_blocked && g_dirty && g_sc_generation == 1 && created == 2 && destroyed == 2);
    assert(g_sc_ext.width == 800 && g_sc_ext.height == 500 && g_sc_n == 0);
    assert(render_done_drops == 4 && new_semaphores == 4);
    assert(!source_releases && g_slot_source[0] == &sources[0]);
    assert(!gfx_ordinary_present_size(&size) && !size.generation && !size.frame);

    // Explicit recovery recreates the prior extent. Before its first accepted
    // frame even a fully constructed WSI generation gives no result.
    gfx_window_resize(800, 500);
    assert(g_sc && !g_sc_blocked && !g_dirty && g_sc_generation == 2 && created == 3);
    assert(!gfx_ordinary_present_size(&size));
    native_picture = 1;
    sdk_frame = 23;
    assert(compose_frame(1) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    display_result = VK_SUBOPTIMAL_KHR;
    plat_rmutex_lock(&g_pl);
    assert(flip_locked());
    plat_rmutex_unlock(&g_pl);
    assert(g_dirty && !gfx_ordinary_present_size(&size));
    display_result = VK_SUCCESS;
    gfx_window_resize(800, 500);
    sdk_frame = 24;
    assert(compose_frame(2) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    plat_rmutex_lock(&g_pl);
    assert(flip_locked());
    plat_rmutex_unlock(&g_pl);
    assert(gfx_ordinary_present_size(&size) && size.generation == 3 && size.frame == 24 &&
           size.guest_texture_serial == 9 &&
           size.source_width == 320 && size.source_height == 180 &&
           size.output_width == 800 && size.output_height == 500);
    gfx_window_release_vk();
    assert(!gfx_ordinary_present_size(&size) && render_done_drops == new_semaphores && source_releases == 3);
}

static void guest_extent_incarnations(void) {
    const unsigned source_width[] = {320, 640, 320}, source_height[] = {180, 360, 180};
    const unsigned output_width[] = {800, 960, 800}, output_height[] = {500, 540, 500};
    native_picture = 1;
    for (unsigned i = 0; i < 3; i++) {
        if (i) gfx_window_resize(output_width[i], output_height[i]);
        sdk_frame = 30 + i;
        GfxOrdinaryPresentSize size;
        GfxOrdinaryPicture picture;
        assert(!gfx_ordinary_present_size(&size));
        assert(!gfx_ordinary_picture(&picture) && !picture.composition);
        assert(compose_frame(i) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
        assert(!gfx_ordinary_present_size(&size));
        assert(!gfx_ordinary_picture(&picture) && !picture.image);
        plat_rmutex_lock(&g_pl);
        assert(flip_locked());
        plat_rmutex_unlock(&g_pl);
        assert(gfx_ordinary_present_size(&size) && size.generation == i + 1 && size.frame == 30 + i);
        assert(size.guest_texture_serial == 7 + i &&
               size.source_width == source_width[i] && size.source_height == source_height[i] &&
               size.output_width == output_width[i] && size.output_height == output_height[i]);
        assert(gfx_ordinary_picture(&picture) && picture.composition == i + 1 &&
               picture.size.frame == sdk_frame && picture.size.generation == g_sc_generation &&
               picture.image == (uintptr_t)last_picture_destination && picture.image_index == acquired_index &&
               picture.format == (uint32_t)g_sc_fmt && picture.source.image == (uintptr_t)last_picture_source &&
               picture.source.window_image == 71 && picture.source.window_texture_serial == 7 + i &&
               picture.source.source_texture_serial == (i == 1 ? 0 : 7 + i) &&
               picture.source.handed_scene == 65 + i && picture.source.handed_ordinal == 804 &&
               picture.source_rect[0] == (i == 1 ? 16 : 0) && picture.source_rect[2] == (i == 1 ? 608 : 320) &&
               !memcmp(picture.output_rect, last_picture_rect, sizeof picture.output_rect));
    }
    gfx_window_release_vk();
    assert(source_releases == 3);
}

static void picture_receipt_mismatch(void) {
    sdk_frame = 43;
    mismatch_picture = 1;
    assert(compose_frame(0) == VK_PRESENT_SOURCE_GFX_SUBMITTED && !frontend_source);
    plat_rmutex_lock(&g_pl);
    assert(flip_locked());
    plat_rmutex_unlock(&g_pl);
    GfxOrdinaryPicture picture;
    assert(!gfx_ordinary_picture(&picture) && !picture.image && !picture.composition);
    GfxOrdinaryPresentSize size;
    assert(gfx_ordinary_present_size(&size) && size.frame == 43 && size.guest_texture_serial == 7 &&
           size.source_width == 320 && size.output_width == 800);
    assert(g_slot_source[0] == &sources[0] && !source_releases);
    gfx_window_release_vk();
    assert(source_releases == 1);
}

static void picture_composed(void) {
    sdk_frame = 58;
    native_picture = picture_grade = picture_fsr = 1;
    assert(compose_frame(0) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    GfxOrdinaryPicture picture;
    assert(!gfx_ordinary_picture(&picture));
    plat_rmutex_lock(&g_pl);
    assert(flip_locked());
    plat_rmutex_unlock(&g_pl);
    assert(gfx_ordinary_picture(&picture) && picture.source.image == 71 &&
           picture.image == (uintptr_t)g_sc_img[acquired_index] && !picture_blits &&
           picture.effects == (GFX_PICTURE_COLOR_GRADED | GFX_PICTURE_FSR_ENLARGED));
    gfx_window_release_vk();
}

static void no_handoff(void) {
    use_motion = 1;
    frontend_motion = &motion[0];
    acquire_result = VK_TIMEOUT;
    assert(compose_with_motion(0, &frontend_motion) == VK_PRESENT_SOURCE_UNSENT);
    assert(acquire_count == 1 && !g_cap_valid && !compose_count && !resets && !g_flip.phase);
    assert(frontend_source == &sources[0] && !g_slot_source[0] && !source_releases);
    assert(frontend_motion == &motion[0] && !motion[0].consumed && !motion_releases);
    acquire_result = VK_SUCCESS;
    submit_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    assert(compose_with_motion(0, &frontend_motion) == VK_PRESENT_SOURCE_UNSENT);
    assert(g_sc_blocked && g_dirty && settled == 1 && !flip_count && !g_cap_valid && !g_flip.phase);
    assert(frontend_source == &sources[0] && !g_slot_source[0] && !source_releases);
    assert(frontend_motion == &motion[0] && motion[0].consumed && !motion_releases);
    vk_present_retire_temporal_inputs();
    assert(!motion_polls && !motion_releases);
    unsigned at = 0;
    expect(&at, ACQUIRE);
    expect(&at, ACQUIRE);
    expect(&at, COMPOSE);
    expect(&at, SETTLE);
    check_no_driver_progress(at);
}

static void uncertain_submit(VkResult result) {
    use_motion = 1;
    frontend_motion = &motion[0];
    motion_fence_status[0] = VK_SUCCESS; // Polling is forbidden without a known submission.
    submit_result = result;
    assert(compose_with_motion(0, &frontend_motion) == VK_PRESENT_SOURCE_UNCERTAIN);
    assert(g_sc_blocked && g_dirty && vk_lost == (result == VK_ERROR_DEVICE_LOST));
    assert(!settled && !g_cap_valid && !g_flip.phase);
    assert(!frontend_source && g_slot_source[0] == &sources[0] && !source_releases);
    assert(!frontend_motion && motion[0].consumed && !motion_releases);
    vk_present_retire_temporal_inputs();
    assert(!motion[0].released && !motion_releases && !motion_polls);
    unsigned at = 0;
    expect(&at, ACQUIRE);
    expect(&at, COMPOSE);
    check_no_driver_progress(at);
    assert(compose_frame(1) == VK_PRESENT_SOURCE_UNSENT);
    assert(frontend_source == &sources[1] && g_slot_source[0] == &sources[0] && !source_releases);
    check_no_driver_progress(at);
    gfx_window_resize(960, 540);
    assert(g_slot_temporal[0] == &motion[0] && !motion_releases && !motion_polls);
    check_no_driver_progress(at);
}

static void failed_present_and_retirement(void) {
    assert(compose_frame(0) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    VkSwapchainKHR previous = g_sc;
    display_result = VK_ERROR_OUT_OF_DATE_KHR;
    idle_result = VK_TIMEOUT;
    g_present_slots[0].retired = HANDLE(VkSemaphore, 117);
    g_present_slots[0].retire_value = 53;
    gfx_window_resize(960, 540);
    assert(g_sc_blocked && g_dirty && g_sc == previous && g_sc_generation == 1);
    assert(!destroyed && created == 1 && !sdk_waits && g_present_slots[0].retire_value == 53);
    assert(!g_flip.phase && flip_count == 1);
    idle_result = VK_SUCCESS;
    sdk_result = VK_TIMEOUT;
    assert(compose_frame(1) == VK_PRESENT_SOURCE_UNSENT);
    assert(acquire_count == 1 && sdk_waits == 1 && !destroyed && g_sc == previous);
    assert(g_present_slots[0].retire_value == 53);
    sdk_result = VK_SUCCESS;
    gfx_window_resize(960, 540);
    assert(!g_sc_blocked && g_sc != previous && g_sc_generation == 2 && destroyed == 1);
    assert(flip_count == 1 && sdk_waits == 2 && !g_present_slots[0].retire_value);
}

static void failed_shutdown(void) {
    use_motion = 1;
    frontend_motion = &motion[0];
    assert(compose_with_motion(0, &frontend_motion) == VK_PRESENT_SOURCE_GFX_SUBMITTED && !frontend_motion);
    VkSwapchainKHR previous = g_sc;
    idle_result = VK_ERROR_DEVICE_LOST;
    fatal_expected = 1;
    if (setjmp(fatal_return) == 0) {
        gfx_window_release_vk();
        abort();
    }
    assert(vk_lost && g_started && g_sc == previous && !destroyed && !releases && g_flip.phase == FLIP_READY);
    assert(g_slot_source[0] == &sources[0] && !source_releases);
    assert(!motion_releases);
}

static void motion_receipts(void) {
    capture_result = VK_TIMEOUT;
    motion_fence_status[0] = motion_fence_status[1] = VK_NOT_READY;
    use_motion = 1;
    frontend_motion = &motion[0];
    assert(compose_with_motion(0, &frontend_motion) == VK_PRESENT_SOURCE_GFX_SUBMITTED && !frontend_motion);
    refuse_trylock = 1;
    vk_present_retire_temporal_inputs();
    assert(!motion_polls && !motion_releases);
    refuse_trylock = 0;
    vk_present_retire_temporal_inputs();
    assert(motion_polls == 1 && !motion_releases);
    frontend_motion = &motion[1];
    assert(compose_with_motion(1, &frontend_motion) == VK_PRESENT_SOURCE_GFX_SUBMITTED && !frontend_motion);
    assert(!motion_releases); // The previous slot's display flip and the new handoff do not prove GPU completion.
    motion_fence_status[0] = VK_ERROR_UNKNOWN;
    motion_fence_status[1] = VK_SUCCESS;
    vk_present_retire_temporal_inputs();
    assert(motion_polls == 3 && motion_releases == 1 && motion[1].released && !motion[0].released);
    motion_fence_status[0] = VK_SUCCESS;
    vk_present_retire_temporal_inputs();
    vk_present_retire_temporal_inputs();
    assert(motion_polls == 4 && motion_releases == 2 && motion[0].released);
    use_motion = 0;
    frontend_motion = &motion[2];
    assert(compose_with_motion(2, &frontend_motion) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    assert(frontend_motion == &motion[2] && !motion[2].consumed);
    vk_present_retire_temporal_inputs();
    gfx_window_release_vk();
    assert(motion_polls == 4 && motion_releases == 2 && !motion[2].released);
}

static void motion_reuse_and_shutdown(void) {
    use_motion = 1;
    frontend_motion = &motion[0];
    assert(compose_with_motion(0, &frontend_motion) == VK_PRESENT_SOURCE_GFX_SUBMITTED && !frontend_motion);
    frontend_motion = &motion[1];
    assert(compose_with_motion(1, &frontend_motion) == VK_PRESENT_SOURCE_GFX_SUBMITTED && !frontend_motion);
    frontend_motion = &motion[2];
    assert(compose_with_motion(2, &frontend_motion) == VK_PRESENT_SOURCE_GFX_SUBMITTED && !frontend_motion);
    assert(motion_releases == 1 && motion[0].released && !motion[1].released && !motion[2].released);
    gfx_window_release_vk();
    gfx_window_release_vk();
    assert(motion_releases == 3 && motion[1].released && motion[2].released && !motion_polls);
}

static void failed_flip(VkResult result, FlipPhase phase) {
    capture_result = VK_TIMEOUT;
    assert(compose_frame(0) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    PendingFlip attempted = g_flip;
    assert(!g_cap_valid && attempted.phase == FLIP_READY && !source_releases);
    display_result = result;
    plat_rmutex_lock(&g_pl);
    assert(!flip_locked() && !flip_locked());
    plat_rmutex_unlock(&g_pl);
    assert(flip_count == 1 && g_sc_blocked && g_dirty && g_flip.phase == phase);
    assert(g_flip.generation == attempted.generation && g_flip.swapchain == attempted.swapchain);
    assert(g_flip.index == attempted.index && g_flip.render_done == attempted.render_done && !source_releases);
    idle_result = VK_TIMEOUT;
    gfx_window_resize(960, 540);
    assert(g_flip.phase == phase && g_sc == attempted.swapchain && g_sc_generation == attempted.generation);
    assert(flip_count == 1 && !destroyed && created == 1 && !source_releases);
}

static void sdk_setup(int split) {
    sdk_active = receipts = 1;
    sdk_frame = expect_sdk_frame = UINT64_C(0x100000043);
    g_sc_path = PRESENT_DLSSG;
    if (!split) test_ctx.present_queue = test_ctx.gfx_queue;
    for (unsigned i = 0; i < FRAMES; ++i) {
        mock_hudless[i] = (Nx2SlImage){HANDLE(VkImage, 1800+i), HANDLE(VkImageView, 1900+i),
            g_sc_fmt, g_sc_ext.width, g_sc_ext.height, VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_USAGE_STORAGE_BIT};
        g_hudless[i].view = mock_hudless[i].view;
        g_hudless[i].image = mock_hudless[i].image;
        g_hudless[i].memory = HANDLE(VmaAllocation, 2100+i);
        mock_alpha[i] = (VkNgxImage){.img = HANDLE(VkImage, 2000+i), .view = HANDLE(VkImageView, 2050+i),
            .w = g_sc_ext.width, .h = g_sc_ext.height, .fmt = VK_FORMAT_R16_SFLOAT};
    }
}

static void mock_split_submission(void) {
    int to[4] = {0, 0, (int)g_sc_ext.width, (int)g_sc_ext.height};
    unsigned slot = g_frame % FRAMES;
    Nx2SlImage alpha = {mock_alpha[slot].img, mock_alpha[slot].view, mock_alpha[slot].fmt,
        mock_alpha[slot].w, mock_alpha[slot].h, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_USAGE_STORAGE_BIT};
    plat_rmutex_lock(&g_pl);
    PendingFlip flip = {.phase = FLIP_COPY, .generation = g_sc_generation, .swapchain = g_sc,
        .path = g_sc_path, .slot = slot, .queue = test_ctx.present_queue, .guest_frame = sdk_frame};
    assert(present_inputs_tag(&flip, HANDLE(VkCommandBuffer, 1201), g_sc_ext, to,
                              &mock_hudless[slot], &alpha, NULL) == VK_DLSSG_CONFIG_INVALIDATED);
    assert(g_present_sdk_inputs[slot].state == SDK_INPUTS_FREE);
    VkDlssgConfigPreparation prep = {.frame = sdk_frame, .count = 1, .generated = 1,
        .choice = {{.key = {.generation = g_sc_generation, .swapchain = g_sc, .color = g_sc_fmt,
                            .backbuffers = g_sc_n}}}};
    assert(present_inputs_tag(&flip, HANDLE(VkCommandBuffer, 1201), g_sc_ext, to,
                              &mock_hudless[slot], &alpha, &prep) == VK_DLSSG_CONFIG_READY);
    g_flip = flip;
    plat_rmutex_unlock(&g_pl);
}

static void stage_flip_now(int show) {
    uint64_t prior = ++sdk_frame; // The app may have selected its next frame while this one was deferred.
    plat_rmutex_lock(&g_pl);
    assert(flip_locked() == show);
    plat_rmutex_unlock(&g_pl);
    assert(sdk_frame == prior);
}

static void ffx_hudless_retirement(void) {
    g_pl = 1;
    g_sc = HANDLE(VkSwapchainKHR, 890);
    g_sc_generation = 17;
    g_sc_path = PRESENT_FFX;
    VkImage a = HANDLE(VkImage, 891), b = HANDLE(VkImage, 892);
    PendingFlip flip = {.path = PRESENT_FFX, .swapchain = g_sc, .generation = 17, .slot = 0};
    present_ffx_hold_hudless(&flip, a);
    present_ffx_hold_hudless(&flip, a); // Same physical image remains ordered on the game queue.
    flip.slot = 1;
    present_ffx_hold_hudless(&flip, b); // Configured before submit, including a known-unsent command.
    assert(g_ffx_hudless[0].image == a && g_ffx_hudless[1].image == b && !ffx_finishes);
    g_sc_generation++;
    assert(present_ffx_finish_hudless() == VK_ERROR_INITIALIZATION_FAILED && !ffx_finishes);
    assert(g_ffx_hudless[0].image == a && g_ffx_hudless[1].image == b);
    g_sc_generation--;

    ffx_finish_result = VK_ERROR_DEVICE_LOST;
    assert(present_ffx_finish_hudless() == VK_ERROR_DEVICE_LOST && ffx_finishes == 1);
    assert(g_ffx_hudless[0].image == a && g_ffx_hudless[1].image == b);

    ffx_finish_result = VK_SUCCESS;
    assert(present_ffx_finish_hudless() == VK_SUCCESS && ffx_finishes == 2);
    assert(!g_ffx_hudless[0].image && !g_ffx_hudless[1].image);
    assert(present_ffx_finish_hudless() == VK_SUCCESS && ffx_finishes == 2);
}

static void sdk_lanes(int split) {
    sdk_setup(split);
    if (split) mock_split_submission();
    else assert(compose_frame(0) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    assert(tagged == 1 && g_present_sdk_inputs[0].state == SDK_INPUTS_PENDING &&
           g_present_sdk_inputs[0].hudless.image == mock_hudless[0].image &&
           g_present_sdk_inputs[0].alpha.view == mock_alpha[0].view);
    stage_flip_now(1);
    assert(typed_presented == 1 && !legacy_presented && g_present_sdk_inputs[0].state == SDK_INPUTS_TIMELINE);
    sdk_result = VK_TIMEOUT;
    // The graphics-read and SDK-read producers are distinct. Even an idle
    // device and no deferred flip cannot replace observing the SDK completion.
    gfx_window_resize(900, 600);
    assert(sdk_waits == 1 && !destroyed && !checked_shutdown && g_sc_generation == 1 &&
           g_present_sdk_inputs[0].state == SDK_INPUTS_TIMELINE);
    sdk_result = VK_SUCCESS;
    gfx_window_resize(900, 600);
    assert(sdk_waits == 2 && destroyed == 1 && !checked_shutdown && g_sc_generation == 2 &&
           g_present_sdk_inputs[0].state == SDK_INPUTS_FREE);
}

static void sdk_temporal_generation_retirement(int split) {
    sdk_setup(split);
    split_temporal = split;
    g_snapshot_ready = HANDLE(VkSemaphore, 92);
    capture_result = VK_TIMEOUT;
    use_motion = 1;
    frontend_motion = &motion[0];
    assert(compose_with_motion(0, &frontend_motion) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    assert(!frontend_motion && g_slot_temporal[0] == &motion[0] && !motion[0].completed);
    assert(g_flip.phase == (split ? FLIP_COPY : FLIP_READY));
    stage_flip_now(1);
    assert(typed_presented == 1 && g_present_sdk_inputs[0].state == SDK_INPUTS_TIMELINE);

    idle_result = VK_TIMEOUT;
    gfx_window_resize(900, 600);
    assert(g_sc_generation == 1 && g_slot_temporal[0] == &motion[0] && !motion_releases && !sdk_waits);

    idle_result = VK_SUCCESS;
    sdk_result = VK_TIMEOUT;
    gfx_window_resize(900, 600);
    assert(g_sc_generation == 1 && g_slot_temporal[0] == &motion[0] && !motion_releases && sdk_waits == 1);

    sdk_result = VK_SUCCESS;
    gfx_window_resize(900, 600);
    assert(g_sc_generation == 2 && !g_slot_temporal[0] && motion_releases == 1 && sdk_waits == 2);
    assert(motion[0].completed && motion[0].released && !motion_polls);
    assert(g_present_sdk_inputs[0].state == SDK_INPUTS_FREE);

    // Reuse this slot with the next generation. The completed owner stays released once;
    // the new hold needs its own graphics proof.
    for (unsigned i = 0; i < FRAMES; ++i) {
        mock_hudless[i].width = mock_alpha[i].w = g_sc_ext.width;
        mock_hudless[i].height = mock_alpha[i].h = g_sc_ext.height;
    }
    g_frame = FRAMES;
    frontend_motion = &motion[1];
    assert(compose_with_motion(1, &frontend_motion) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    assert(!frontend_motion && g_slot_temporal[0] == &motion[1] && motion_releases == 1);
    motion_fence_status[0] = VK_NOT_READY;
    vk_present_retire_temporal_inputs();
    assert(g_slot_temporal[0] == &motion[1] && motion_releases == 1);
    motion_fence_status[0] = VK_SUCCESS;
    vk_present_retire_temporal_inputs();
    assert(!g_slot_temporal[0] && motion[1].released && motion_releases == 2);
    assert(g_present_sdk_inputs[0].state == SDK_INPUTS_PENDING);
}

static void sdk_reuse(void) {
    sdk_setup(0);
    assert(compose_frame(0) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    stage_flip_now(1);
    g_frame = 2;
    sdk_result = VK_TIMEOUT;
    assert(compose_frame(1) == VK_PRESENT_SOURCE_UNSENT);
    assert(tagged == 1 && sdk_waits == 1 && g_present_sdk_inputs[0].state == SDK_INPUTS_TIMELINE);
    sdk_result = VK_SUCCESS;
    assert(compose_frame(1) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    assert(tagged == 2 && sdk_waits == 2 && g_present_sdk_inputs[0].receipt.ticket == 2);
}

static void sdk_abandon(int split, int cancel_failed) {
    sdk_setup(split);
    cancel_fails = cancel_failed;
    if (split) {
        copy_failed = 1;
        mock_split_submission();
        stage_flip_now(0);
    } else {
        submit_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
        assert(compose_frame(0) == VK_PRESENT_SOURCE_UNSENT);
    }
    assert(tagged == 1 && canceled == 1 && !typed_presented && !legacy_presented && !sdk_waits);
    assert(g_present_sdk_inputs[0].state == (cancel_failed ? SDK_INPUTS_QUARANTINED : SDK_INPUTS_FREE));
}

static void sdk_blocked(int malformed_tag, int startup_mode) {
    sdk_setup(0);
    invalid_tag = malformed_tag;
    if (startup_mode == 0) no_timeline = 1;
    if (startup_mode == 1) sdk_image_result = VK_ERROR_OUT_OF_DATE_KHR;
    if (startup_mode == 2) invalid_completion = 1;
    const int show = startup_mode == 0;
    assert(compose_frame(0) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    stage_flip_now(show);
    assert(typed_presented == (unsigned)!malformed_tag && legacy_presented == (unsigned)malformed_tag && !canceled &&
           g_present_sdk_inputs[0].state == SDK_INPUTS_QUARANTINED);
    stage_flip_now(0); // Native present cannot be retried on an ambiguous return.
    assert(typed_presented == (unsigned)!malformed_tag && legacy_presented == (unsigned)malformed_tag && !canceled);
    unsigned progress = ntrace;
    gfx_window_resize(900, 600);
    assert(ntrace == progress && !destroyed && !checked_shutdown && !sdk_waits && g_sc_blocked);
    assert(g_present_sdk_inputs[0].hudless.image == mock_hudless[0].image &&
           g_present_sdk_inputs[0].alpha.view == mock_alpha[0].view);
}

static void sdk_final(void) {
    sdk_blocked(0, 0);
    fatal_expected = shutdown_fails = 1;
    if (!setjmp(fatal_return)) {
        gfx_window_release_vk();
        abort();
    }
    assert(checked_shutdown == 1 && destroyed == 1 && !destroyed_hudless &&
           g_present_sdk_inputs[0].state == SDK_INPUTS_QUARANTINED);
    g_pl = fatal_expected = shutdown_fails = 0;
    gfx_window_release_vk();
    assert(checked_shutdown == 2 && destroyed_hudless == FRAMES &&
           g_present_sdk_inputs[0].state == SDK_INPUTS_FREE);
    unsigned after_shutdown = 0;
    for (unsigned i = 0; i < ntrace; ++i) {
        if (trace[i] == SHUTDOWN) after_shutdown++;
        if (trace[i] == DROP_HUD) assert(after_shutdown == 2);
    }
}

static void sdk_uncertain_submit(void) {
    sdk_setup(0);
    submit_result = VK_ERROR_UNKNOWN;
    assert(compose_frame(0) == VK_PRESENT_SOURCE_UNCERTAIN);
    assert(tagged == 1 && !typed_presented && !legacy_presented && !canceled &&
           g_present_sdk_inputs[0].state == SDK_INPUTS_QUARANTINED);
}

static void sdk_legacy(int split) {
    sdk_setup(split);
    receipts = 0;
    if (split) {
        g_flip = (PendingFlip){.phase = FLIP_COPY, .generation = g_sc_generation,
            .swapchain = g_sc, .path = g_sc_path, .queue = test_ctx.present_queue};
    } else {
        assert(compose_frame(0) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    }
    expect_sdk_frame++;
    stage_flip_now(1);
    assert(!tagged && !typed_presented && legacy_presented == 1 &&
           legacy_tagged == (unsigned)!split && legacy_retirement == (unsigned)split &&
           g_present_sdk_inputs[0].state == SDK_INPUTS_FREE);
}

static void sdk_hud_retirement(void) {
    sdk_setup(0);
    hud_source.images = (VkDlaaHudImages){.generation=81,.frame=sdk_frame,.exact=1};
    frontend_hud = &hud_source;
    assert(compose_frame(0) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    assert(!frontend_hud && g_slot_hud[0]==&hud_source &&
           g_flip.hud_generation == 81 && g_flip.hud_exact);
    stage_flip_now(1);
    assert(g_present_sdk_inputs[0].state == SDK_INPUTS_TIMELINE &&
           g_present_sdk_inputs[0].hud_generation == 81);
    motion_fence_status[0] = VK_NOT_READY;
    vk_present_retire_temporal_inputs();
    assert(!hud_source.released && g_slot_hud[0] == &hud_source);
    motion_fence_status[0] = VK_SUCCESS;
    vk_present_retire_temporal_inputs();
    assert(hud_source.released && !g_slot_hud[0] &&
           g_present_sdk_inputs[0].state == SDK_INPUTS_TIMELINE && !sdk_waits);
}

static void sdk_hud_unsent(void) {
    sdk_setup(0);
    submit_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    hud_source.images = (VkDlaaHudImages){.generation=81,.frame=sdk_frame,.exact=1};
    frontend_hud = &hud_source;
    assert(compose_frame(0) == VK_PRESENT_SOURCE_UNSENT);
    assert(frontend_hud == &hud_source && !g_slot_hud[0] && !hud_source.released);
}

static void sdk_config_pending(void) {
    sdk_setup(0);
    config_result = VK_DLSSG_CONFIG_PENDING;
    hud_source.images = (VkDlaaHudImages){.generation=90,.frame=sdk_frame,.exact=1,
        .color={.img=HANDLE(VkImage,2200)},.alpha={.img=HANDLE(VkImage,2201)}};
    frontend_hud = &hud_source;
    VkDlaaFgSizeUse *t = &motion[0];
    assert(compose_with_motion(0,&t) == VK_PRESENT_SOURCE_PENDING);
    assert(frontend_hud == &hud_source && t == &motion[0] && !t->consumed && !g_slot_hud[0]);
    assert(config_calls == 1 && selected_variants == 3 && config_retry.peak.amount[0] == 4096);
    assert(!acquire_count && !compose_count && !typed_presented && !legacy_presented && !tagged && !config_tags);
    config_result = VK_DLSSG_CONFIG_READY;
    configuration_token = 79;
    assert(compose_with_motion(0,&t) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    assert(g_flip.sdk_config == 79 && !frontend_hud && config_tags == 1);
    stage_flip_now(1);
    assert(config_accepted == 1 && config_checks >= 2 && !config_dropped &&
           g_present_sdk_inputs[0].receipt.ticket);
}

static void sdk_config_old_drain(void) {
    sdk_setup(0);
    config_result = VK_DLSSG_CONFIG_DRAIN_REQUIRED;
    old_drain = 1;
    assert(compose_frame(0) == VK_PRESENT_SOURCE_PENDING);
    assert(config_retry.older_config == 123 && config_retry.older_graphics == g_fence[1] &&
           !acquire_count && !compose_count && !config_tags && !tagged);
    external_idle = 1;
    assert(vk_present_config_drain(&config_retry) == VK_SUCCESS);
    external_idle = 0;
    assert(old_drained == 1 && !g_pl);
    config_result = VK_DLSSG_CONFIG_READY;
    assert(compose_frame(0) == VK_PRESENT_SOURCE_GFX_SUBMITTED && config_tags == 1);
}

static void sdk_config_unproven(void) {
    sdk_setup(0);
    configuration_token = 101;
    sdk_image_result = VK_ERROR_OUT_OF_DATE_KHR;
    assert(compose_frame(0) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    stage_flip_now(0);
    assert(config_unproven == 1 && !config_accepted && g_present_sdk_inputs[0].state == SDK_INPUTS_QUARANTINED);
}

static void sdk_config_changed(void) {
    sdk_setup(0);
    config_invalidate = 1;
    assert(compose_frame(0) == VK_PRESENT_SOURCE_PENDING);
    assert(config_retry.reprepare && acquire_count == 1 && settled == 1 && config_dropped == 1);
    assert(!compose_count && !tagged && !typed_presented && !legacy_presented &&
           frontend_source == &sources[0] && !g_slot_source[0] && !g_flip.phase);
}

static void sdk_missing_inputs(void) {
    sdk_setup(0);
    valid_no_tags = 1;
    assert(compose_frame(0) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    assert(selected_variants == 1 && config_tags == 1 && !g_flip.sdk_config);
    stage_flip_now(1);
    assert(typed_presented == 1 && g_present_sdk_inputs[0].state == SDK_INPUTS_FREE);
    valid_no_tags = 0;
    expect_sdk_frame = sdk_frame;
    VkDlaaFgSizeUse *t = &motion[0];
    assert(compose_with_motion(1,&t) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    stage_flip_now(1);
    assert(typed_presented == 2 && g_present_sdk_inputs[1].state == SDK_INPUTS_TIMELINE);
}

static void sdk_invalidated_scene(void) {
    sdk_setup(0);
    invalidated_scene = fatal_expected = 1;
    VkDlaaFgSizeUse *t = &motion[0];
    if (!setjmp(fatal_return)) {
        compose_with_motion(0,&t);
        abort();
    }
    assert(!config_calls && !acquire_count && !compose_count && !tagged);
}

static void sdk_hud_uncertain(void) {
    sdk_setup(0);
    submit_result = VK_ERROR_UNKNOWN;
    hud_source.images = (VkDlaaHudImages){.generation=93,.frame=sdk_frame,.exact=1};
    frontend_hud = &hud_source;
    assert(compose_frame(0) == VK_PRESENT_SOURCE_UNCERTAIN);
    assert(!frontend_hud && g_slot_hud[0]==&hud_source && !typed_presented &&
           g_present_sdk_inputs[0].state == SDK_INPUTS_QUARANTINED);
    vk_present_retire_temporal_inputs();
    assert(!hud_source.released && !motion_polls && g_slot_hud[0]==&hud_source);
}

static void sdk_no_tag(void) {
    sdk_setup(0);
    valid_no_tags = 1;
    assert(compose_frame(0) == VK_PRESENT_SOURCE_GFX_SUBMITTED);
    stage_flip_now(1);
    assert(tagged == 1 && typed_presented == 1 && !canceled && !sdk_waits &&
           g_present_sdk_inputs[0].state == SDK_INPUTS_FREE);
}

int main(int argc, char **argv) {
    assert(argc == 2);
    setup();
    if (!strcmp(argv[1], "resize")) resize_and_release();
    else if (!strcmp(argv[1], "resize-recovery")) resize_failure_and_recovery();
    else if (!strcmp(argv[1], "guest-extents")) guest_extent_incarnations();
    else if (!strcmp(argv[1], "picture-receipt-mismatch")) picture_receipt_mismatch();
    else if (!strcmp(argv[1], "picture-composed")) picture_composed();
    else if (!strcmp(argv[1], "unsent")) no_handoff();
    else if (!strcmp(argv[1], "uncertain")) uncertain_submit(VK_ERROR_DEVICE_LOST);
    else if (!strcmp(argv[1], "uncertain-without-loss")) uncertain_submit(VK_ERROR_UNKNOWN);
    else if (!strcmp(argv[1], "failed-present")) failed_present_and_retirement();
    else if (!strcmp(argv[1], "failed-shutdown")) failed_shutdown();
    else if (!strcmp(argv[1], "unsent-flip")) failed_flip(VK_ERROR_OUT_OF_DEVICE_MEMORY, FLIP_UNSENT);
    else if (!strcmp(argv[1], "uncertain-flip")) failed_flip(VK_ERROR_UNKNOWN, FLIP_UNCERTAIN);
    else if (!strcmp(argv[1], "motion")) motion_receipts();
    else if (!strcmp(argv[1], "motion-reuse")) motion_reuse_and_shutdown();
    else if (!strcmp(argv[1], "ffx-hud-retirement")) ffx_hudless_retirement();
    else if (!strcmp(argv[1], "sdk-direct")) sdk_lanes(0);
    else if (!strcmp(argv[1], "sdk-split")) sdk_lanes(1);
    else if (!strcmp(argv[1], "sdk-temporal-resize-direct")) sdk_temporal_generation_retirement(0);
    else if (!strcmp(argv[1], "sdk-temporal-resize-split")) sdk_temporal_generation_retirement(1);
    else if (!strcmp(argv[1], "sdk-reuse")) sdk_reuse();
    else if (!strcmp(argv[1], "sdk-compose-unsent")) sdk_abandon(0, 0);
    else if (!strcmp(argv[1], "sdk-split-unsent")) sdk_abandon(1, 0);
    else if (!strcmp(argv[1], "sdk-cancel-failed")) sdk_abandon(1, 1);
    else if (!strcmp(argv[1], "sdk-unproven")) sdk_blocked(0, 0);
    else if (!strcmp(argv[1], "sdk-image-failed")) sdk_blocked(0, 1);
    else if (!strcmp(argv[1], "sdk-completion-malformed")) sdk_blocked(0, 2);
    else if (!strcmp(argv[1], "sdk-tag-malformed")) sdk_blocked(1, 0);
    else if (!strcmp(argv[1], "sdk-final")) sdk_final();
    else if (!strcmp(argv[1], "sdk-no-tag")) sdk_no_tag();
    else if (!strcmp(argv[1], "sdk-hud-retirement")) sdk_hud_retirement();
    else if (!strcmp(argv[1], "sdk-hud-unsent")) sdk_hud_unsent();
    else if (!strcmp(argv[1], "sdk-config-pending")) sdk_config_pending();
    else if (!strcmp(argv[1], "sdk-config-drain")) sdk_config_old_drain();
    else if (!strcmp(argv[1], "sdk-config-unproven")) sdk_config_unproven();
    else if (!strcmp(argv[1], "sdk-config-changed")) sdk_config_changed();
    else if (!strcmp(argv[1], "sdk-inputless")) sdk_missing_inputs();
    else if (!strcmp(argv[1], "sdk-lost-input")) sdk_invalidated_scene();
    else if (!strcmp(argv[1], "sdk-hud-uncertain")) sdk_hud_uncertain();
    else if (!strcmp(argv[1], "sdk-uncertain-submit")) sdk_uncertain_submit();
    else if (!strcmp(argv[1], "sdk-legacy-direct")) sdk_legacy(0);
    else if (!strcmp(argv[1], "sdk-legacy-split")) sdk_legacy(1);
    else abort();
    return 0;
}
'''


def main():
    fixture = wsi_fixture((ROOT / 'runtime/gfx_present.c').read_text())
    cc = os.environ.get('CC') or shutil.which('clang')
    if not cc:
        raise SystemExit('Clang is required (set CC)')
    toolchain = Path(shutil.which(cc) or cc).resolve()
    vma = Path(os.environ.get('VMA_INCLUDE', toolchain.parent.parent / 'include' / 'vma'))
    env = dict(os.environ, PATH=str(toolchain.parent) + os.pathsep + os.environ.get('PATH', ''))
    with tempfile.TemporaryDirectory(prefix='present-ownership-') as directory:
        c = Path(directory) / 'check.c'
        exe = Path(directory) / ('check.exe' if os.name == 'nt' else 'check')
        c.write_text(fixture)
        subprocess.run([str(toolchain), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror',
                        '-fsanitize=undefined', '-fsanitize-trap=undefined',
                        '-I' + str(ROOT / 'runtime'), '-I' + str(vma), str(c), '-o', str(exe)],
                       env=env, check=True)
        for case in ('resize', 'resize-recovery', 'guest-extents', 'picture-receipt-mismatch', 'picture-composed', 'unsent', 'uncertain', 'uncertain-without-loss', 'failed-present', 'failed-shutdown',
                     'unsent-flip', 'uncertain-flip', 'motion', 'motion-reuse',
                     'ffx-hud-retirement', 'sdk-direct', 'sdk-split', 'sdk-reuse', 'sdk-compose-unsent', 'sdk-split-unsent',
                     'sdk-cancel-failed', 'sdk-unproven', 'sdk-image-failed', 'sdk-completion-malformed',
                     'sdk-tag-malformed', 'sdk-final', 'sdk-no-tag', 'sdk-uncertain-submit',
                     'sdk-legacy-direct', 'sdk-legacy-split', 'sdk-hud-retirement', 'sdk-hud-unsent',
                     'sdk-config-pending', 'sdk-config-drain', 'sdk-config-unproven', 'sdk-config-changed',
                     'sdk-inputless', 'sdk-lost-input', 'sdk-hud-uncertain',
                     'sdk-temporal-resize-direct', 'sdk-temporal-resize-split'):
            subprocess.run([str(exe), case], env=env, check=True)
    print('Present ownership: native handoff, resize, temporal fences, Streamline and FidelityFX input retirement passed')


if __name__ == '__main__':
    main()

"""Check reconstruction Size, NGX SR, and frame generation ownership without a Vulkan device."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def section(source, start, end):
    offset = source.index(start)
    return source[offset:source.index(end, offset)]


def generate(source):
    size = section(source, '    struct Size {', ', *last;')
    feature_type = section(source, 'typedef int NgxResult;', 'typedef struct { void *slot[17]; }')
    native = section(source, 'static VkImageCreateInfo dlaa_image_2d_info', 'static int image2d(')
    motion = section(source, 'struct VkDlaaFgSizeUse {', '// What this frame drew')
    motion += section(source, 'int vk_dlaa_fg_views(', 'const VkTexImage *vk_dlaa_swap_img;')
    generation = section(source, 'static int size_idle(', 'static void full_barrier(')
    features = section(source, 'typedef struct {\n    uint64_t required_scene;', '// The scene\'s last column and row')
    features += section(source, 'static int perf_quality(', '// Every 300 frames')
    features += section(source, 'static int ngx_sr_create_opaque(', 'static void dlaa_compute_blit_written(')
    evaluate = section(source, 'static int evaluate(', '\nevaluated:')
    assert evaluate.index('release_kept(ex, e') < evaluate.index('ngx_sr_create_opaque(ex, e')
    assert evaluate.index('ngx_sr_create_opaque(ex, e') < evaluate.index('size_record_begin(ex, e)') < evaluate.index('edge_fill(cb,')
    assert evaluate.index('ngx_sr_scene_use(ex, e)') < evaluate.index('N.evaluate(cb, e->sr.handle')
    select_geometry = section(evaluate, '    struct Size *e = size_for(', '    const int cold_sr')
    select_geometry += section(evaluate, '    VkCommandBuffer cb = ex->cb;', '    if (cold_sr && !ngx_sr_create_opaque')
    select_geometry += section(evaluate, '    D.early = early;', '    VkImageView zv')
    reset = section(source, '    D.fg_reset = D.reset;', '    if (D.evals % 600')
    publish = section(source, '    D.fg_depth = (VkFfxImage)', '    size_hook_close();\n    if (D.jit_used)')
    return r'''
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define RC_BUDGET_TEST
#define RC_VMA_TEST
#include "platform/platform.h"
#include "host.h"
#include "resource_core_vma.h"
#include "vk_exec.h"
#include "vk_dlaa.h"
#include "vk_fg.h"
#include "vk_dlssg.h"
#define HANDLE(type, n) ((type)(uintptr_t)(n))
''' + feature_type + r'''
struct NgxParam { unsigned writes; };
enum {DLSS_SIZES = 5, IMAGE_COUNT = 7, NATIVE_CAP = 80};
static struct {
''' + size + r''', *last;
    struct Size *swap_size, *fg_size;
    VkTexturePresentSource *fg_depth_owner;
    VkFfxImage fg_depth, fg_mv;
    VkImageView fg_depth_view, fg_mv_view;
    VkDlaaCamera fg_cam;
    float fg_jx, fg_jy;
    uint64_t presents, fg_at, evals, why[3];
    int failed, reset, early, previous_early, preset, preset_dirty, fg_fresh, fg_reset, fg_cam_ok, ids_ok;
} D;
static struct { VkDlaaHudSource *open; } hud_sources;
static uint64_t size_generation;

static VkCtx device;
static RcBudgetSample sample;
static RcAdmissionVector previous_charge, group_charge;
static unsigned lock_depth, platform_depth, native_attempts, view_attempts, successful_native, successful_views;
static unsigned native_destroyed, views_destroyed, fail_native, fail_view, barriers;
static unsigned barriers_at_attempt;
static unsigned format_attempts, query_attempts, placement_attempts, stop_native;
static uint32_t w = 1279, h = 720, ow = 1920, oh = 1080;
static uint64_t clock_ns = 1000000000;
static int stop_requested;
static VkResult native_failure;
static VkResult depth_result;
volatile int32_t vk_lost;

enum { OUTPUT_HEAP = RC_BUDGET_GPU_FIRST, SCENE_HEAP };
#define MB (1024ull * 1024ull)

static const VkImageUsageFlags compact = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
static const VkImageUsageFlags transfer = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
static const VkFormat formats[IMAGE_COUNT] = {
    VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16_SFLOAT,
    VK_FORMAT_R8_UNORM, VK_FORMAT_R8_UNORM, VK_FORMAT_R8_UNORM,
    VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_SFLOAT
};
static struct {
    VkFormat format;
    unsigned ordinal;
    int live, view_live;
} allocated[NATIVE_CAP];
struct VkTexturePresentSource { VkImage image; VkImageView view; int released; };
static VkTexturePresentSource depth_providers[16];
static unsigned depth_retain_count, depth_views_made, depth_views_destroyed, depth_releases;
static VkImageView guest_depth_view;

VkCtx *vk_ctx(void) { return &device; }
VkDeviceSize vk_vma_block_size(void) { return 64 * MB; }
int plat_commit_counts_vram(void) { return 1; }
uint64_t rc_budget_test_now_ns(void) { return clock_ns; }
uint64_t kern_now_ns(void) { return clock_ns; }
void rc_budget_test_lock(void) { assert(!lock_depth++); }
void rc_budget_test_unlock(void) { assert(lock_depth-- == 1); }
void rc_budget_test_sample(RcBudgetSample *out) { *out = sample; }
void rc_budget_test_wait(uint32_t ms) { (void)ms; abort(); }
int vk_sync_stopping(void) { return stop_requested; }
void plat_mutex_lock(PlatMutex *lock) { (void)lock; assert(!platform_depth++ && !lock_depth); }
void plat_mutex_unlock(PlatMutex *lock) { (void)lock; assert(platform_depth-- == 1); }
void rt_log(const char *fmt, ...) { (void)fmt; }
void rt_fatal(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    abort();
}
void vk_device_lost_check(VkResult result, const char *operation) {
    (void)operation;
    if (result == VK_ERROR_DEVICE_LOST) vk_lost = 1;
}

#include "resource_core_budget.c"
#include "resource_core_vma.c"

static unsigned allocation_type(VkFormat format) { return format == formats[0] ? 0 : 1; }

static uint64_t bytes_for(unsigned index) {
    static const unsigned bytes_per_texel[IMAGE_COUNT] = {8, 4, 1, 1, 1, 4, 4};
    uint64_t area = index ? (uint64_t)w * h : (uint64_t)ow * oh;
    return (area * bytes_per_texel[index] + 255) & ~UINT64_C(255);
}

static void check_shape(const VkImageCreateInfo *info, unsigned index) {
    assert(index < IMAGE_COUNT && info->sType == VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO);
    assert(info->imageType == VK_IMAGE_TYPE_2D && info->format == formats[index] && !info->flags);
    assert(info->extent.width == (index ? w : ow) && info->extent.height == (index ? h : oh) && info->extent.depth == 1);
    assert(info->mipLevels == 1 && info->arrayLayers == 1 && info->samples == VK_SAMPLE_COUNT_1_BIT);
    assert(info->initialLayout == VK_IMAGE_LAYOUT_UNDEFINED && info->tiling == VK_IMAGE_TILING_OPTIMAL);
    VkImageUsageFlags expected = compact | (index < 2 ? transfer : 0);
    if (index == 1) expected |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    assert(info->usage == expected);
}

static void check_placement(const VkImageCreateInfo *info, const VmaAllocationCreateInfo *create,
                            unsigned index, int pinned) {
    check_shape(info, index);
    assert(create->usage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    assert(create->flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    if (pinned || create->memoryTypeBits) assert(create->memoryTypeBits == 1u << allocation_type(info->format));
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties(
    VkPhysicalDevice phys, VkFormat format, VkImageType type, VkImageTiling tiling,
    VkImageUsageFlags usage, VkImageCreateFlags flags, VkImageFormatProperties *out) {
    assert(!lock_depth && phys == device.phys && type == VK_IMAGE_TYPE_2D);
    assert(tiling == VK_IMAGE_TILING_OPTIMAL && (usage & compact) == compact && !flags);
    assert(format == formats[format_attempts++ % IMAGE_COUNT]);
    *out = (VkImageFormatProperties){.maxExtent = {8192, 8192, 1}, .maxMipLevels = 1,
        .maxArrayLayers = 1, .sampleCounts = VK_SAMPLE_COUNT_1_BIT};
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkGetDeviceImageMemoryRequirements(
    VkDevice d, const VkDeviceImageMemoryRequirements *query, VkMemoryRequirements2 *out) {
    assert(!lock_depth && d == device.device && query->sType == VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS);
    unsigned index = query_attempts++ % IMAGE_COUNT;
    check_shape(query->pCreateInfo, index);
    unsigned type = allocation_type(query->pCreateInfo->format);
    out->memoryRequirements = (VkMemoryRequirements){bytes_for(index), 256, 1u << type};
    ((VkMemoryDedicatedRequirements *)out->pNext)->requiresDedicatedAllocation = VK_FALSE;
}

VkResult vmaFindMemoryTypeIndexForImageInfo(VmaAllocator a, const VkImageCreateInfo *info,
                                            const VmaAllocationCreateInfo *create, uint32_t *type) {
    assert(!lock_depth && a == device.vma);
    unsigned index = placement_attempts++ % IMAGE_COUNT;
    check_placement(info, create, index, 0);
    *type = allocation_type(info->format);
    return VK_SUCCESS;
}

VkResult vmaCreateImage(VmaAllocator a, const VkImageCreateInfo *info,
                        const VmaAllocationCreateInfo *create, VkImage *out,
                        VmaAllocation *memory, VmaAllocationInfo *actual) {
    assert(!lock_depth && !platform_depth && a == device.vma && barriers == barriers_at_attempt);
    unsigned index = native_attempts++;
    check_placement(info, create, index, 1);
    RcBudgetSnapshot ledger;
    resource_core_budget_snapshot(&ledger);
    for (unsigned i = 0; i < RC_ADMISSION_MAX_DOMAINS; ++i)
        assert(ledger.reserved.amount[i] + ledger.retained.amount[i] == previous_charge.amount[i] + group_charge.amount[i]);
    if (native_attempts == fail_native) return native_failure;
    unsigned handle = successful_native++;
    assert(handle < NATIVE_CAP);
    allocated[handle].format = info->format;
    allocated[handle].ordinal = index;
    allocated[handle].live = 1;
    *out = HANDLE(VkImage, handle + 101);
    *memory = HANDLE(VmaAllocation, handle + 201);
    *actual = (VmaAllocationInfo){.size = bytes_for(index), .memoryType = allocation_type(info->format)};
    if (native_attempts == stop_native) stop_requested = 1;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(VkDevice d, const VkImageViewCreateInfo *ci,
                                                 const VkAllocationCallbacks *c, VkImageView *view) {
    assert(!lock_depth && d == device.device && !c);
    if (ci->subresourceRange.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT) {
        assert(depth_retain_count && ci->format == VK_FORMAT_D32_SFLOAT && ci->viewType == VK_IMAGE_VIEW_TYPE_2D);
        VkTexturePresentSource *p = &depth_providers[depth_retain_count - 1];
        assert(p->image == ci->image && !p->released && !p->view);
        assert(ci->subresourceRange.baseMipLevel == 2 && ci->subresourceRange.levelCount == 1);
        assert(ci->subresourceRange.baseArrayLayer == 3 && ci->subresourceRange.layerCount == 1);
        if (depth_result != VK_SUCCESS) return depth_result;
        p->view = *view = HANDLE(VkImageView, 501 + depth_retain_count);
        assert(p->view != guest_depth_view);
        depth_views_made++;
        return VK_SUCCESS;
    }
    assert(barriers == barriers_at_attempt);
    unsigned handle = (unsigned)(uintptr_t)ci->image - 101, index = view_attempts++;
    assert(handle < successful_native && index < IMAGE_COUNT && allocated[handle].live);
    assert(allocated[handle].ordinal == index && allocated[handle].format == ci->format);
    assert(ci->sType == VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO && ci->viewType == VK_IMAGE_VIEW_TYPE_2D);
    VkImageSubresourceRange sub = ci->subresourceRange;
    assert(sub.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT && !sub.baseMipLevel && sub.levelCount == 1);
    assert(!sub.baseArrayLayer && sub.layerCount == 1);
    if (view_attempts == fail_view) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    *view = HANDLE(VkImageView, handle + 301);
    allocated[handle].view_live = 1;
    successful_views++;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(VkDevice d, VkImageView view, const VkAllocationCallbacks *c) {
    assert(!lock_depth && d == device.device && !c);
    if ((uintptr_t)view >= 502) {
        unsigned index = (unsigned)(uintptr_t)view - 502;
        assert(index < depth_retain_count && depth_providers[index].view == view);
        depth_providers[index].view = VK_NULL_HANDLE;
        depth_views_destroyed++;
        return;
    }
    unsigned i = (unsigned)(uintptr_t)view - 301;
    assert(i < successful_native);
    assert(allocated[i].live && allocated[i].view_live);
    allocated[i].view_live = 0;
    views_destroyed++;
}

void vmaDestroyImage(VmaAllocator a, VkImage image, VmaAllocation allocation) {
    unsigned i = (unsigned)(uintptr_t)image - 101;
    assert(!lock_depth && a == device.vma && i < successful_native);
    assert(allocated[i].live && !allocated[i].view_live && allocation == HANDLE(VmaAllocation, i + 201));
    allocated[i].live = 0;
    native_destroyed++;
}

static unsigned scene_captures;
VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo *info) {
    assert(cb && info && info->imageMemoryBarrierCount == IMAGE_COUNT);
    assert(scene_captures > barriers);
    for (unsigned i = 0; i < IMAGE_COUNT; ++i) {
        const VkImageMemoryBarrier2 *b = &info->pImageMemoryBarriers[i];
        unsigned index = (unsigned)(uintptr_t)b->image - 101;
        assert(b->oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && b->newLayout == VK_IMAGE_LAYOUT_GENERAL);
        assert(b->subresourceRange.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT);
        assert(index < successful_native && allocated[index].live && allocated[index].view_live);
        assert(allocated[index].ordinal == i);
    }
    barriers++;
}

// These are other VMA entry points in the shared adapter, outside this fixture's provider.
VKAPI_ATTR void VKAPI_CALL vkGetDeviceBufferMemoryRequirements(
    VkDevice d, const VkDeviceBufferMemoryRequirements *r, VkMemoryRequirements2 *out) {
    (void)d; (void)r; (void)out; abort();
}
VkResult vmaFindMemoryTypeIndexForBufferInfo(VmaAllocator a, const VkBufferCreateInfo *c,
                                             const VmaAllocationCreateInfo *p, uint32_t *type) {
    (void)a; (void)c; (void)p; (void)type; abort();
}
VkResult vmaFindMemoryTypeIndex(VmaAllocator a, uint32_t bits, const VmaAllocationCreateInfo *c, uint32_t *type) {
    (void)a; (void)bits; (void)c; (void)type; abort();
}
VkResult vmaCreateBuffer(VmaAllocator a, const VkBufferCreateInfo *c, const VmaAllocationCreateInfo *p, VkBuffer *out,
                         VmaAllocation *allocation, VmaAllocationInfo *info) {
    (void)a; (void)c; (void)p; (void)out; (void)allocation; (void)info; abort();
}
void vmaDestroyBuffer(VmaAllocator a, VkBuffer buffer, VmaAllocation allocation) {
    (void)a; (void)buffer; (void)allocation; abort();
}
VkResult vmaAllocateMemory(VmaAllocator a, const VkMemoryRequirements *r, const VmaAllocationCreateInfo *c,
                          VmaAllocation *allocation, VmaAllocationInfo *info) {
    (void)a; (void)r; (void)c; (void)allocation; (void)info; abort();
}
VkResult vmaCreatePool(VmaAllocator a, const VmaPoolCreateInfo *create, VmaPool *pool) {
    (void)a; (void)create; (void)pool; abort();
}
void vmaFreeMemory(VmaAllocator a, VmaAllocation allocation) { (void)a; (void)allocation; abort(); }
''' + native + r'''

static void dimensions(uint32_t scene_w, uint32_t scene_h, uint32_t output_w, uint32_t output_h) {
    w = scene_w, h = scene_h, ow = output_w, oh = output_h;
    group_charge = (RcAdmissionVector){0};
    for (unsigned i = 0; i < IMAGE_COUNT; ++i)
        group_charge.amount[i ? SCENE_HEAP : OUTPUT_HEAP] += bytes_for(i);
    group_charge.amount[RC_BUDGET_COMMIT] = group_charge.amount[OUTPUT_HEAP] + group_charge.amount[SCENE_HEAP];
}

static void configure(void) {
    assert(successful_native == native_destroyed && successful_views == views_destroyed);
    resource_core_vma_test_reset();
    resource_core_budget_test_reset();
    memset(&D, 0, sizeof D);
    memset(allocated, 0, sizeof allocated);
    previous_charge = (RcAdmissionVector){0};
    dimensions(1279, 720, 1920, 1080);
    native_attempts = view_attempts = successful_native = successful_views = 0;
    native_destroyed = views_destroyed = fail_native = fail_view = barriers = 0;
    barriers_at_attempt = 0;
    format_attempts = query_attempts = placement_attempts = stop_native = 0;
    stop_requested = vk_lost = 0;
    native_failure = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    device = (VkCtx){.vma = HANDLE(VmaAllocator, 1), .device = HANDLE(VkDevice, 2),
                     .phys = HANDLE(VkPhysicalDevice, 3)};
    device.props.deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    device.mem_props.memoryTypeCount = device.mem_props.memoryHeapCount = 2;
    for (unsigned heap = 0; heap < 2; ++heap) {
        device.mem_props.memoryTypes[heap] = (VkMemoryType){VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, heap};
        device.mem_props.memoryHeaps[heap] = (VkMemoryHeap){8192 * MB, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    }
    sample = (RcBudgetSample){.known_mask = (1u << (SCENE_HEAP + 1)) - 1};
    for (unsigned i = 0; i <= SCENE_HEAP; ++i) {
        sample.total.amount[i] = 8192 * MB;
        sample.free.amount[i] = 4096 * MB;
    }
}

static void assert_retained(const RcAdmissionVector *amount) {
    RcBudgetSnapshot ledger;
    resource_core_budget_snapshot(&ledger);
    for (unsigned d = 0; d < RC_ADMISSION_MAX_DOMAINS; ++d) {
        assert(!ledger.reserved.amount[d] && !ledger.pending.amount[d]);
        assert(ledger.retained.amount[d] == amount->amount[d]);
    }
}

static void check_complete(const struct Size *s) {
    assert(s->out && s->out_view && s->mv && s->mv_view && s->mask && s->mask_view);
    for (unsigned i = 0; i < 2; ++i)
        assert(s->chars[i] && s->chars_view[i] && s->dist[i] && s->dist_view[i]);
}

static DlaaSizeNativeOutcome plan(DlaaSizeNativePlan *prepared) {
    DlaaSizeNativeOutcome result = dlaa_size_native_plan(w, h, ow, oh, prepared);
    assert(result.state == DLAA_SIZE_NATIVE_READY);
    assert(!memcmp(&result.peak, &group_charge, sizeof group_charge));
    assert(query_attempts && placement_attempts);
    return result;
}

static void preserve_old_on_refusal(void) {
    configure();
    DlaaSizeNativePlan prepared;
    plan(&prepared);
    struct Size retained = {0};
    DlaaSizeNativeOutcome result = dlaa_size_native_create(&prepared, &retained);
    assert(result.state == DLAA_SIZE_NATIVE_READY && !barriers);
    assert_retained(&group_charge);
    check_complete(&retained);
    previous_charge = group_charge;
    D.sizes[0] = retained;
    D.last = &D.sizes[0];
    native_attempts = view_attempts = 0;
    dimensions(960, 540, 1600, 900);
    plan(&prepared);
    const unsigned refused_domain = SCENE_HEAP;
    sample.free.amount[refused_domain] = 64 * MB + group_charge.amount[refused_domain] - 1;
    clock_ns += UINT64_C(1000000000);
    resource_core_budget_refresh();
    struct Size unmade = {0};
    result = dlaa_size_native_create(&prepared, &unmade);
    assert(result.state == DLAA_SIZE_NATIVE_PENDING && result.admission == RC_ADMISSION_REFUSED);
    assert(result.before.available.amount[refused_domain] == group_charge.amount[refused_domain] - 1);
    assert(!memcmp(&result.peak, &group_charge, sizeof group_charge));
    assert(!native_attempts && !view_attempts && !barriers && !unmade.out);
    assert(D.last == &D.sizes[0] && D.last->out == retained.out && D.last->mv_view == retained.mv_view);
    assert(D.last->w == 1279 && D.last->h == 720 && D.last->ow == 1920 && D.last->oh == 1080);
    assert_retained(&previous_charge);
    dlaa_size_native_destroy(D.last);
    assert(native_destroyed == IMAGE_COUNT && views_destroyed == IMAGE_COUNT);
    assert_retained(&(RcAdmissionVector){0});
}

static void failure_sweep(void) {
    for (unsigned i = 1; i <= 2 * IMAGE_COUNT; ++i) {
        configure();
        DlaaSizeNativePlan prepared;
        plan(&prepared);
        fail_native = i <= IMAGE_COUNT ? i : 0;
        fail_view = i > IMAGE_COUNT ? i - IMAGE_COUNT : 0;
        struct Size unmade = {0};
        DlaaSizeNativeOutcome result = dlaa_size_native_create(&prepared, &unmade);
        assert(result.state != DLAA_SIZE_NATIVE_READY && result.vk_result == VK_ERROR_OUT_OF_DEVICE_MEMORY);
        assert(!unmade.out && !unmade.out_view && !barriers && !D.sizes[0].out);
        if (fail_native) assert(native_attempts == fail_native && successful_native == fail_native - 1);
        if (fail_view) assert(view_attempts == fail_view && successful_views == fail_view - 1);
        assert(successful_native == native_destroyed && successful_views == views_destroyed);
        assert_retained(&(RcAdmissionVector){0});
    }
}

static void check_stop_and_impossible(void) {
    configure();
    stop_requested = 1;
    DlaaSizeNativePlan prepared = {0};
    assert(dlaa_size_native_plan(w, h, ow, oh, &prepared).state == DLAA_SIZE_NATIVE_STOPPED);
    assert(!prepared.images[0].format && !format_attempts && !native_attempts);
    stop_requested = 0;
    device.mem_props.memoryHeaps[1].size = group_charge.amount[SCENE_HEAP] - 1;
    DlaaSizeNativeOutcome result = dlaa_size_native_plan(w, h, ow, oh, &prepared);
    assert(result.state == DLAA_SIZE_NATIVE_TERMINAL && result.admission == RC_ADMISSION_UNSUPPORTED);
    assert(!prepared.images[0].format && !native_attempts);

    configure();
    plan(&prepared);
    stop_native = 1;
    struct Size unmade = {0};
    result = dlaa_size_native_create(&prepared, &unmade);
    assert(result.state == DLAA_SIZE_NATIVE_STOPPED && !unmade.out && !view_attempts && !barriers);
    assert(successful_native == 1 && native_destroyed == 1);
    assert_retained(&(RcAdmissionVector){0});

    configure();
    plan(&prepared);
    fail_native = 2;
    native_failure = VK_ERROR_DEVICE_LOST;
    result = dlaa_size_native_create(&prepared, &unmade);
    assert(result.state == DLAA_SIZE_NATIVE_STOPPED && result.vk_result == VK_ERROR_DEVICE_LOST);
    assert(!unmade.out && !barriers && successful_native == 1 && native_destroyed == 1);
    assert(successful_views == views_destroyed);
    assert_retained(&(RcAdmissionVector){0});
}

static VkSyncProgressResult parked_result = VK_SYNC_PROGRESS_COMPLETE;
static unsigned parks, parked_retire_calls, unavailable, canceled;
static unsigned feature_releases, fence_waits, rec_drains, draw_invalidations, create_calls, sdk_failed;
static uint64_t scene_done, scene_handed, scene_flush;
static int stop_on_fence_wait, stop_on_rec_drain;
static int parked, sdk_reset, fg_enabled;
static struct {
    uint64_t fence;
    void (*done)(void *);
    void (*cancel)(void *, int);
    void *arg;
} scene_receipt[16];
static const VkTexImage *submitting_depth;
static struct {
    NgxResult (*create)(VkCommandBuffer, int, NgxParam *, NgxHandle **);
    NgxResult (*release)(NgxHandle *);
} N;
static int feat_released;
static volatile int32_t size_changed;
const VkTexImage *vk_dlaa_swap_img;
VkImageView vk_dlaa_swap_view;
VkImage vk_dlaa_swap_output;

int vk_dlssg_active(void) { return fg_enabled; }
int vk_fg_active(void) { return 0; }
int vk_fg_wanted(void) { return 0; }
int vk_ngxfg_on(void) { return 0; }
VkTexturePresentSource *vk_texture_present_retain_image(const VkTexImage *selected, VkImage exact) {
    assert(selected && selected == submitting_depth && selected->aspect == VK_IMAGE_ASPECT_DEPTH_BIT);
    if (selected->image != exact) return NULL;
    assert(depth_retain_count < sizeof depth_providers / sizeof *depth_providers);
    VkTexturePresentSource *owner = &depth_providers[depth_retain_count++];
    assert(!owner->image);
    owner->image = exact;
    return owner;
}
void vk_texture_present_release(VkTexturePresentSource *owner) {
    if (!owner) return;
    assert(owner->image && !owner->view && !owner->released);
    owner->released = 1;
    depth_releases++;
}
static void size_hook_close(void) {}
void vk_present_retire_temporal_inputs(void) { assert(parked); parked_retire_calls++; }
void vk_completion_add_owned(uint64_t fence, void (*done)(void *), void (*cancel)(void *, int), void *arg) {
    assert(!parked && fence && done && cancel && arg);
    assert(scene_captures < sizeof scene_receipt / sizeof *scene_receipt);
    scene_receipt[scene_captures++] = (typeof(scene_receipt[0])){fence, done, cancel, arg};
}
static void retire_scene(uint64_t fence, int stopped) {
    unsigned retired = 0;
    for (unsigned i = 0; i < scene_captures; ++i) {
        if (scene_receipt[i].fence != fence) continue;
        if (stopped) scene_receipt[i].cancel(scene_receipt[i].arg, 0);
        else scene_receipt[i].done(scene_receipt[i].arg);
        scene_receipt[i].fence = 0;
        retired++;
    }
    assert(retired == 1);
}
VkSyncProgressResult vk_exec_temporal_prepare(VkExec *ex, int (*prepare)(void *), void *context) {
    assert(ex->fence && !parked && !stop_requested);
    parked = 1;
    parks++;
    if (scene_flush > scene_handed) scene_handed = scene_flush;
    VkSyncProgressResult proof = prepare(context) ? parked_result : VK_SYNC_PROGRESS_STOPPED;
    parked = 0;
    return proof;
}
void vk_exec_cancel_temporal_preparation(void) { assert(!parked); canceled++; }
void vk_exec_required_temporal_unavailable(VkExec *ex, const char *why) {
    assert(ex->fence && why && !parked);
    unavailable++;
}
uint64_t vk_sync_fence_done(void) { return scene_done; }
uint64_t vk_sync_handed_fence(void) { return scene_handed; }
int vk_sync_fence_wait(uint64_t fence, uint64_t timeout_ns) {
    assert(parked && timeout_ns && fence > scene_done && fence <= scene_handed);
    fence_waits++;
    if (stop_on_fence_wait) { stop_requested = 1; return 0; }
    scene_done = fence;
    return 1;
}
void vk_exec_barrier(VkExec *ex) { assert(ex->cb && !parked); }
void vk_gpu_gap_note(int kind, uint64_t pixels) { assert(kind == VK_GAP_UPSCALER && pixels == (uint64_t)ow * oh); }
static void p_setu(NgxParam *p, const char *key, unsigned value) {
    assert(p && key && value);
    p->writes++;
}
static void p_seti(NgxParam *p, const char *key, int value) {
    if (p) {
        assert(key && value >= 0);
        p->writes++;
    } else {
        assert(!strcmp(key, "Reset"));
        sdk_reset = value;
    }
}
static int dl_fail(const char *what, NgxResult result) { assert(what && !NGX_OK(result)); sdk_failed++; return 0; }
void rec_drain(void) {
    assert(!parked);
    rec_drains++;
    if (stop_on_rec_drain) stop_requested = 1;
}
void vk_exec_draw_new_cb(void) { draw_invalidations++; }
uint64_t perf_mark_begin(void) { return 1; }
void perf_mark_end(int kind, uint64_t start, uint64_t bytes) { assert(kind == PERF_M_NGX && start && !bytes); }
#define RESOURCE_IMAGE_SCOPE(name, f) (void)(f)
// This fixture executes the ordinary SR/FG path. The prepared target ownership
// and its native cancellation are exercised independently by the target receipt fixture.
typedef struct { VkTexImage image; } DlaaTargetOwner;
enum { DLAA_TARGET_COLOR, DLAA_TARGET_DEPTH, DLAA_TARGET_IDS };
static DlaaTargetOwner *frame_targets[3];
static void dlaa_frame_targets_close(void) {
    for (unsigned i=0;i<3;i++) assert(frame_targets[i]==NULL);
}
static void dlaa_target_record(VkExec *ex,DlaaTargetOwner *owner,int publishes) {
    assert(ex && !owner && !publishes);
}
static int dlaa_target_valid(const DlaaTargetOwner *owner) { return owner == NULL; }
static VkTexturePresentSource *dlaa_present_target(const DlaaTargetOwner *owner,VkImage image) {
    (void)owner;(void)image;assert(0);return NULL;
}
''' + motion + generation + features + r'''

static void attempt_size(uint32_t sw, uint32_t sh, uint32_t dw, uint32_t dh) {
    dimensions(sw, sh, dw, dh);
    RcBudgetSnapshot ledger;
    resource_core_budget_snapshot(&ledger);
    previous_charge = ledger.retained;
    native_attempts = view_attempts = 0;
    barriers_at_attempt = barriers;
}

static struct Size *record_evaluate(VkExec *ex, const VkExecTarget *z, int early, int sdk_ready) {
''' + select_geometry + r'''
    (void)cb;
    void *p = NULL;
''' + next(line for line in source.splitlines() if 'p_seti(p, "Reset", D.reset);' in line) + r'''
    if (!sdk_ready) return NULL;
''' + reset + r'''
    const float jx = 0.25f, jy = -0.375f;
''' + publish + r'''
    D.evals++;
    return e;
}

static VkDlaaFgSizeUse *capture_fg(void) {
    D.presents++;
    VkDlaaFgSizeUse *use = vk_dlaa_fg_capture();
    assert(use && !vk_dlaa_fg_capture());
    return use;
}

static void compose_fg(VkDlaaFgSizeUse *use, const struct Size *s, VkImage expected_depth,
                       VkImageView expected_view, int expected_reset) {
    VkFfxImage depth = {0}, vectors = {0};
    VkImageView zv = VK_NULL_HANDLE, mv = VK_NULL_HANDLE;
    VkDlaaCamera camera;
    float jx = 0, jy = 0;
    int reset = -1;
    assert(!vk_dlaa_fg_inputs(&depth, &vectors, &jx, &jy, &reset));
    assert(!vk_dlaa_fg_compose_peek(&depth, &vectors, &zv, &mv, &camera, &jx, &jy, &reset));
    vk_dlaa_fg_compose_select(use);
    assert(vk_dlaa_fg_describe(use, &depth, &vectors) == 1);
    assert(vk_dlaa_fg_compose_peek(&depth, &vectors, &zv, &mv, &camera, &jx, &jy, &reset) == 1);
    assert(zv == expected_view && mv == s->mv_view && camera.znear == 0.125f);
    assert(!vk_dlaa_fg_compose_consumed(use) && !vk_dlaa_fg_views(&zv, &mv));
    assert(vk_dlaa_fg_inputs(&depth, &vectors, &jx, &jy, &reset));
    assert(depth.image == expected_depth && depth.width == s->w && depth.height == s->h);
    assert(vectors.image == s->mv && vectors.width == s->w && vectors.height == s->h);
    assert(jx == 0.25f && jy == -0.375f && reset == expected_reset);
    assert(vk_dlaa_fg_views(&zv, &mv) && zv == expected_view && mv == s->mv_view);
    assert(vk_dlaa_fg_camera(&camera) && camera.znear == 0.125f);
    assert(vk_dlaa_fg_compose_consumed(use) && !vk_dlaa_fg_inputs(&depth, &vectors, &jx, &jy, &reset));
    assert(vk_dlaa_fg_compose_peek(&depth, &vectors, &zv, &mv, &camera, &jx, &jy, &reset) == -1);
    vk_dlaa_fg_compose_select(NULL);
}

static void history_and_depth_view(void) {
    configure();
    fg_enabled = D.fg_cam_ok = 1;
    D.fg_cam.znear = 0.125f;
    VkTexImage depth = {.image = HANDLE(VkImage, 701), .aspect = VK_IMAGE_ASPECT_DEPTH_BIT};
    submitting_depth = &depth;
    VkExecTarget z = {.img = &depth, .image = depth.image, .format = VK_FORMAT_D32_SFLOAT, .level = 2, .base_layer = 3};
    VkExec ex = {.cb = HANDLE(VkCommandBuffer, 7), .fence = 11};
    guest_depth_view = HANDLE(VkImageView, 61);
    struct Size *a = record_evaluate(&ex, &z, 0, 1);
    assert(a && a->generation && a->initialized && a->scene_users == 1 && a->present_users == 1);
    assert(sdk_reset == 1 && scene_captures == 1 && barriers == 1 && parks == 1 && parked_retire_calls);
    size_record_begin(&ex, a);
    assert(a->scene_users == 1 && scene_captures == 1 && barriers == 1);
    VkDlaaFgSizeUse *older = capture_fg();
    VkImageView old_view = older->depth_view;
    VkImage old_image = z.image;
    retire_scene(ex.fence, 0);
    assert(!a->scene_users && a->present_users == 1);

    // The guest changes its backing and its cached D32 view after the old present has queued.
    guest_depth_view = HANDLE(VkImageView, 62);
    depth.image = z.image = HANDLE(VkImage, 702);
    ex.fence++;
    assert(record_evaluate(&ex, &z, 0, 1) == a && !sdk_reset && barriers == 1 && parks == 1);
    VkDlaaFgSizeUse *steady = capture_fg();
    compose_fg(steady, a, z.image, steady->depth_view, 0);
    vk_dlaa_fg_release(steady);
    retire_scene(ex.fence, 0);
    assert(a->present_users == 1 && !depth_providers[0].released && depth_providers[0].view == old_view);

    attempt_size(960, 540, 1920, 1080);
    depth.image = z.image = HANDLE(VkImage, 703);
    ex.fence++;
    struct Size *b = record_evaluate(&ex, &z, 0, 1);
    assert(b && b != a && b->generation > a->generation && sdk_reset && parks == 2 && barriers == 2);
    VkDlaaFgSizeUse *newer = capture_fg();
    compose_fg(older, a, old_image, old_view, 1);
    assert(depth_providers[0].view == old_view && !depth_providers[0].released);
    vk_dlaa_fg_release(older); // The WSI fixture proves this release is gated by the exact composition fence.
    assert(!depth_providers[0].view && depth_providers[0].released && !a->present_users);
    compose_fg(newer, b, z.image, newer->depth_view, 1);
    vk_dlaa_fg_release(newer);
    retire_scene(ex.fence, 0);

    attempt_size(1279, 720, 1920, 1080);
    depth.image = z.image = HANDLE(VkImage, 704);
    ex.fence++;
    assert(record_evaluate(&ex, &z, 0, 1) == a && sdk_reset && parks == 2 && barriers == 2);
    assert(a->scene_users == 1 && D.why[0] == 3);
    // No later frame is allowed to acquire an earlier present's uncaptured publication.
    D.presents += 2;
    assert(!vk_dlaa_fg_capture() && !a->present_users);
    retire_scene(ex.fence, 0);
    assert(size_idle(a) && size_idle(b));
    assert(depth_views_made == depth_views_destroyed && depth_retain_count == depth_releases);
    vk_dlaa_size_shutdown();
    vk_dlaa_size_shutdown();
    assert_retained(&(RcAdmissionVector){0});
}

static void missing_resume_and_cancel(void) {
    configure();
    const unsigned original_barriers = barriers, ready_published = depth_retain_count;
    VkTexImage depth = {.image = HANDLE(VkImage, 705), .aspect = VK_IMAGE_ASPECT_DEPTH_BIT};
    submitting_depth = &depth;
    VkExecTarget z = {.img = &depth, .image = depth.image, .format = VK_FORMAT_D32_SFLOAT, .level = 2, .base_layer = 3};
    VkExec ex = {.cb = HANDLE(VkCommandBuffer, 7), .fence = 21};
    parked_result = VK_SYNC_PROGRESS_UNAVAILABLE;
    assert(!record_evaluate(&ex, &z, 0, 1) && unavailable == 1 && !canceled);
    assert(barriers == original_barriers && !D.last && depth_retain_count == ready_published);
    assert(D.sizes[0].generation && D.sizes[0].retired && !D.sizes[0].initialized);
    assert(size_collect_retired() && !D.sizes[0].generation);
    assert_retained(&(RcAdmissionVector){0});

    parked_result = VK_SYNC_PROGRESS_COMPLETE;
    attempt_size(1279, 720, 1920, 1080);
    ex.fence++;
    fg_enabled = 1;
    depth_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    struct Size *s = record_evaluate(&ex, &z, 0, 1);
    assert(s && s->initialized && !s->present_users && !D.fg_size);
    assert(depth_providers[ready_published].released && !depth_providers[ready_published].view);
    D.presents++;
    assert(!vk_dlaa_fg_capture());
    retire_scene(ex.fence, 1);
    assert(s->retired && size_collect_retired() && !D.last && !s->generation);
    assert_retained(&(RcAdmissionVector){0});

    depth_result = VK_SUCCESS;
    attempt_size(1279, 720, 1920, 1080);
    ex.fence++;
    s = record_evaluate(&ex, &z, 0, 1);
    VkDlaaFgSizeUse *held = capture_fg();
    assert(s && held->size == s && s->scene_users == 1 && s->present_users == 1);
    unsigned destroyed_before = native_destroyed, views_before = depth_views_destroyed;
    retire_scene(ex.fence, 1);
    assert(s->retired && !s->scene_users && !size_collect_retired());
    assert(native_destroyed == destroyed_before && depth_views_destroyed == views_before);
    vk_dlaa_fg_compose_select(held);
    VkFfxImage old_depth, old_mv;
    float jx, jy;
    int reset;
    assert(!vk_dlaa_fg_inputs(&old_depth, &old_mv, &jx, &jy, &reset));
    vk_dlaa_fg_compose_select(NULL);
    vk_dlaa_fg_release(held);
    assert(size_collect_retired() && !s->generation);
    assert(native_destroyed == destroyed_before + IMAGE_COUNT && depth_views_destroyed == views_before + 1);
    assert_retained(&(RcAdmissionVector){0});

    attempt_size(1279, 720, 1920, 1080);
    ex.fence++;
    s = record_evaluate(&ex, &z, 0, 1);
    assert(s && D.fg_size == s && s->present_users == 1);
    D.presents++;
    vk_dlaa_fg_discard_current();
    assert(!D.fg_size && !s->present_users && !vk_dlaa_fg_capture());
    retire_scene(ex.fence, 0);
    vk_dlaa_size_shutdown();
    assert_retained(&(RcAdmissionVector){0});
}

static NgxResult release_feature(NgxHandle *handle) {
    assert(!parked);
    assert(handle == HANDLE(NgxHandle *, feature_releases + 1));
    ++feature_releases;
    return 1;
}

static void check_feature_retirement(void) {
    VkExec ex = {.fence = 40};
    struct Size *earlier = &D.sizes[0], *current = &D.sizes[1];
    earlier->sr = (NgxSrFeature){HANDLE(NgxHandle *, 1), 38, NGX_SR_OPAQUE_READY};
    current->sr = (NgxSrFeature){HANDLE(NgxHandle *, 2), 40, NGX_SR_OPAQUE_READY};
    earlier->last_scene = 40; // A current image use cannot move the feature's native use.
    D.last = earlier;
    N.release = release_feature;
    unsigned unavailable_before = unavailable, canceled_before = canceled;
    assert(!release_kept(&ex, &D.sizes[2], 1));
    assert(unavailable == unavailable_before + 1 && !fence_waits && !feature_releases && D.last == earlier);
    current->sr.scene = 39;
    scene_done = scene_handed = 38;
    assert(!release_kept(&ex, &D.sizes[2], 1));
    assert(unavailable == unavailable_before + 2 && !fence_waits && !feature_releases && D.last == earlier);
    scene_flush = 39;
    stop_on_fence_wait = 1;
    assert(!release_kept(&ex, &D.sizes[2], 1));
    assert(canceled == canceled_before + 1 && fence_waits == 1 && !feature_releases && D.last == earlier);
    stop_requested = stop_on_fence_wait = 0;
    assert(release_kept(&ex, &D.sizes[2], 1));
    assert(fence_waits == 2 && feature_releases == 2 && !earlier->sr.handle && !current->sr.handle && !D.last);
}

static struct Size *creating;
static NgxResult create_result;
static NgxResult create_feature(VkCommandBuffer cb, int feature, NgxParam *p, NgxHandle **out) {
    assert(!parked && cb == HANDLE(VkCommandBuffer, 7) && feature == NGX_FEATURE_SUPERSAMPLING && p->writes == 15);
    assert(rec_drains == create_calls + 1 && draw_invalidations == create_calls);
    assert(creating->sr.state == NGX_SR_FAILED && creating->sr.scene == 41 + create_calls && creating->scene_users == 1);
    create_calls++;
    *out = HANDLE(NgxHandle *, feature_releases + 1);
    return create_result;
}

static void check_feature_create(void) {
    configure();
    scene_captures = 0;
    feat_released = 1;
    N.create = create_feature;
    N.release = release_feature;
    const VkExecTarget t = {.format = VK_FORMAT_B10G11R11_UFLOAT_PACK32}, z = {.format = VK_FORMAT_D32_SFLOAT};
    for (int fail = 0; fail <= 1; ++fail) {
        VkExec ex = {.cb = HANDLE(VkCommandBuffer, 7), .fence = 41 + (unsigned)fail};
        NgxParam params = {0};
        attempt_size(1279, 720, 1920, 1080);
        creating = size_for(&ex, w, h, ow, oh);
        assert(creating && creating->sr.state == NGX_SR_EMPTY);
        unsigned released = feature_releases;
        create_result = fail ? (NgxResult)0xBAD00001u : 1;
        assert(ngx_sr_create_opaque(&ex, creating, &params, &t, &z) == !fail);
        assert(feat_released == fail);
        assert(ex.recorded == 1 && creating->sr.handle && creating->sr.scene == ex.fence && creating->scene_users == 1);
        assert(creating->sr.state == (fail ? NGX_SR_FAILED : NGX_SR_OPAQUE_READY) && creating->retired == (unsigned)fail);
        ngx_sr_scene_use(&ex, creating); // Create and evaluate in one scene share one completion receipt.
        assert(creating->scene_users == 1 && !size_collect_retired() && released == feature_releases);
        retire_scene(ex.fence, 1);
        assert(size_collect_retired() && !creating->sr.handle && feature_releases == released + 1);
        assert(feat_released);
    }
    assert(create_calls == 2 && rec_drains == 2 && draw_invalidations == 2 && sdk_failed == 1);
    assert_retained(&(RcAdmissionVector){0});

    VkExec ex = {.cb = HANDLE(VkCommandBuffer, 7), .fence = 43};
    NgxParam params = {0};
    attempt_size(1279, 720, 1920, 1080);
    creating = size_for(&ex, w, h, ow, oh);
    unsigned canceled_before = canceled;
    stop_on_rec_drain = 1;
    assert(creating && !ngx_sr_create_opaque(&ex, creating, &params, &t, &z));
    assert(stop_requested && canceled == canceled_before + 1 && create_calls == 2 && !creating->scene_users);
    assert(creating->sr.state == NGX_SR_EMPTY && !creating->sr.handle && !ex.recorded);
    vk_dlaa_size_shutdown();
    assert_retained(&(RcAdmissionVector){0});
}

int main(void) {
    failure_sweep();
    check_stop_and_impossible();
    preserve_old_on_refusal();
    history_and_depth_view();
    missing_resume_and_cancel();
    check_feature_retirement();
    check_feature_create();
    puts("Temporal Size: joint allocation, rollback, history, NGX SR custody and exact FG depth view passed");
    return 0;
}
'''


def main():
    source = (ROOT / 'runtime/vk_dlaa.c').read_text()
    cc = os.environ.get('CC') or shutil.which('clang')
    if not cc:
        raise SystemExit('Clang is required (set CC)')
    clang = Path(shutil.which(cc) or cc).resolve()
    vma = Path(os.environ.get('VMA_INCLUDE', clang.parent.parent / 'include' / 'vma'))
    env = dict(os.environ, PATH=str(clang.parent) + os.pathsep + os.environ.get('PATH', ''))
    with tempfile.TemporaryDirectory(prefix='temporal-size-') as directory:
        c = Path(directory) / 'check.c'
        exe = c.with_suffix('.exe' if os.name == 'nt' else '.out')
        c.write_text(generate(source))
        subprocess.run([str(clang), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror',
                        '-fsanitize=undefined', '-fsanitize-trap=undefined',
                        '-I' + str(ROOT / 'runtime'), '-I' + str(vma), str(c), '-o', str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == '__main__':
    main()

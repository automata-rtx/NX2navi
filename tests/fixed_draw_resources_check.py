"""Check fixed draw ownership against the real budget and VMA adapters without a GPU."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from bind_recipe_check import function


def main():
    sync = (ROOT / 'runtime/vk_sync.c').read_text()
    bootstrap = function(sync, 'vk_sync_prepare_draw')
    draw = (ROOT / 'runtime/vk_exec_draw.c').read_text()
    start = draw.index('enum { DV_1D,')
    fixed = draw[start:draw.index('static void twins_release(VkCtx *c);', start)]
    # Specialized formats have their own admission transaction.
    specialized = fixed.index("static const VkClearColorValue DUMMY_STORAGE")
    peak = fixed.index("static int fixed_peak_add", specialized)
    late = fixed.index("static VkImageViewType res_view_type", peak)
    destroy = fixed.index("static void fixed_draw_destroy", late)
    fixed = fixed[:specialized] + fixed[peak:late] + fixed[destroy:]
    zero_bytes = next(line for line in draw.splitlines() if line.startswith('#define ZERO_BYTES '))

    fixture = r'''
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define RC_BUDGET_TEST
#define RC_VMA_TEST
#include "platform/platform.h"
#include "resource_core_vma.h"
#include "vk_exec.h"

static VkCtx device;
static RcBudgetSample sample;
static uint64_t clock_ns = 1000000000;
static unsigned platform_lock_depth, budget_lock_depth;
static unsigned buffer_attempts, image_attempts, native_creates, native_destroys;
static unsigned view_attempts, view_creates, view_destroys, sampler_calls;
static unsigned upload_calls, upload_ends, flushes, barriers, clears;
static unsigned fail_native, fail_view;
static int no_mapping, uncertain_upload, upload_open, extra_depth_shapes;
static int stop_requested, stop_on_wait;
static unsigned queue_handoffs, readiness_waits;
static uint64_t submitted_seq, completed_seq;
volatile int32_t vk_lost;

enum { BUFFER_COUNT = 2, DEPTH_INDEX = 9, MAX_IMAGES = 15, MAX_NATIVE = 17, MAX_VIEWS = 20 };
#define IMAGE_COUNT (extra_depth_shapes ? MAX_IMAGES : 10)
#define NATIVE_COUNT (extra_depth_shapes ? MAX_NATIVE : 12)
#define VIEW_COUNT (extra_depth_shapes ? MAX_VIEWS : 13)
enum { TEST_DEVICE_TYPE, TEST_HOST_TYPE };
#define MB (1024ull * 1024ull)
#define UPLOAD_SEQUENCE 19
#define DEVICE_HEAP RC_BUDGET_GPU_FIRST
#define HOST_HEAP (RC_BUDGET_GPU_FIRST + 1)

static const struct {
    VkImageType type;
    VkImageViewType view;
    uint32_t layers;
    VkSampleCountFlagBits samples;
    VkImageCreateFlags flags;
} shapes[MAX_IMAGES] = {
    {VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D_ARRAY, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE, 6, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, 6, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 1, VK_SAMPLE_COUNT_4_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 1, VK_SAMPLE_COUNT_4_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D_ARRAY, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE, 6, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, 6, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT},
};
static struct {
    uint64_t handle;
    unsigned shape, nviews;
    int is_image, live;
} allocations[MAX_NATIVE];
static struct {
    unsigned owner;
    int live;
} views[MAX_VIEWS];
static unsigned char zero_mapping[65536];
static const VkCommandBuffer upload_cb = (VkCommandBuffer)(uintptr_t)7;
static RcAdmissionVector expected_peak;

VkCtx *vk_ctx(void) { return &device; }
VkDeviceSize vk_vma_block_size(void) { return 64 * MB; }
int plat_commit_counts_vram(void) { return 1; }
uint64_t rc_budget_test_now_ns(void) { return clock_ns; }
void rc_budget_test_lock(void) { assert(!budget_lock_depth && !platform_lock_depth); budget_lock_depth++; }
void rc_budget_test_unlock(void) { assert(budget_lock_depth == 1); budget_lock_depth--; }
void rc_budget_test_sample(RcBudgetSample *out) { *out = sample; }
void rc_budget_test_wait(uint32_t ms) {
    assert(!budget_lock_depth && !platform_lock_depth && ms == 100 && !readiness_waits++);
    assert(!native_creates && !vk_exec_draw_fixed_ready());
    if (stop_on_wait) stop_requested = 1;
    else {
        sample.free.amount[DEVICE_HEAP]++;
        clock_ns += 1000000000;
        resource_core_budget_refresh();
    }
}
static int progress_stopped(void) { return stop_requested || vk_device_is_lost(); }
void vk_queue_lock(void) {
    assert(!platform_lock_depth && !budget_lock_depth && !vk_exec_draw_fixed_ready());
}
void vk_queue_unlock(void) { queue_handoffs++; }
void plat_mutex_lock(PlatMutex *lock) {
    (void)lock;
    assert(!platform_lock_depth && !budget_lock_depth);
    platform_lock_depth++;
}
void plat_mutex_unlock(PlatMutex *lock) { (void)lock; assert(platform_lock_depth == 1); platform_lock_depth--; }
void plat_cond_wait(PlatCond *cond, PlatMutex *lock) { (void)cond; (void)lock; abort(); }
void plat_cond_broadcast(PlatCond *cond) { (void)cond; }
void rt_log(const char *fmt, ...) { (void)fmt; }
_Noreturn void rt_fatal(const char *fmt, ...) {
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
''' + zero_bytes + '\n' + fixed + bootstrap + r'''

static void external_boundary(void) {
    assert(!budget_lock_depth && !platform_lock_depth);
}

static unsigned buffer_type(const VkBufferCreateInfo *info) {
    assert(info->sType == VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO && info->size == ZERO_BYTES);
    VkBufferUsageFlags zero = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
        VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
    VkBufferUsageFlags sink = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    assert(info->usage == zero || info->usage == sink);
    return info->usage == zero ? TEST_HOST_TYPE : TEST_DEVICE_TYPE;
}

static uint64_t image_bytes(const VkImageCreateInfo *info) {
    return (uint64_t)4096 * info->arrayLayers * info->samples;
}

static void report_requirements(VkMemoryRequirements2 *out, uint32_t type, uint64_t bytes) {
    external_boundary();
    out->memoryRequirements = (VkMemoryRequirements){bytes, 256, 1u << type};
    ((VkMemoryDedicatedRequirements *)out->pNext)->requiresDedicatedAllocation = VK_FALSE;
}

VKAPI_ATTR void VKAPI_CALL vkGetDeviceBufferMemoryRequirements(
    VkDevice d, const VkDeviceBufferMemoryRequirements *query, VkMemoryRequirements2 *out) {
    assert(d == device.device);
    report_requirements(out, buffer_type(query->pCreateInfo), query->pCreateInfo->size);
}

VKAPI_ATTR void VKAPI_CALL vkGetDeviceImageMemoryRequirements(
    VkDevice d, const VkDeviceImageMemoryRequirements *query, VkMemoryRequirements2 *out) {
    assert(d == device.device);
    report_requirements(out, TEST_DEVICE_TYPE, image_bytes(query->pCreateInfo));
}

VkResult vmaFindMemoryTypeIndexForBufferInfo(VmaAllocator a, const VkBufferCreateInfo *info,
                                             const VmaAllocationCreateInfo *create, uint32_t *type) {
    external_boundary();
    assert(a == device.vma);
    *type = buffer_type(info);
    if (create->memoryTypeBits) assert(create->memoryTypeBits == (1u << *type));
    return VK_SUCCESS;
}

static VkFormat shape_format(unsigned shape) {
    return shape >= DEPTH_INDEX ? VK_FORMAT_D16_UNORM : VK_FORMAT_R8G8B8A8_UNORM;
}

static int no_depth_compare;

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures(VkPhysicalDevice phys, VkPhysicalDeviceFeatures *features) {
    assert(phys == device.phys);
    *features = (VkPhysicalDeviceFeatures){.imageCubeArray = (VkBool32)extra_depth_shapes};
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties(
    VkPhysicalDevice phys, VkFormat format, VkImageType type, VkImageTiling tiling, VkImageUsageFlags usage,
    VkImageCreateFlags flags, VkImageFormatProperties *properties) {
    assert(phys == device.phys && format == VK_FORMAT_D16_UNORM &&
           (type == VK_IMAGE_TYPE_1D || type == VK_IMAGE_TYPE_2D) && tiling == VK_IMAGE_TILING_OPTIMAL);
    assert(usage == (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
    assert(!flags || flags == VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);
    if (!extra_depth_shapes) return VK_ERROR_FORMAT_NOT_SUPPORTED;
    *properties = (VkImageFormatProperties){.maxArrayLayers = 6, .sampleCounts = VK_SAMPLE_COUNT_1_BIT};
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties2(
    VkPhysicalDevice phys, VkFormat format, VkFormatProperties2 *properties) {
    assert(phys == device.phys && format == VK_FORMAT_D16_UNORM);
    VkFormatProperties3 *format3 = properties->pNext;
    assert(format3->sType == VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3);
    format3->optimalTilingFeatures = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT;
    if (!no_depth_compare) format3->optimalTilingFeatures |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;
}

VkResult vmaFindMemoryTypeIndexForImageInfo(VmaAllocator a, const VkImageCreateInfo *info,
                                            const VmaAllocationCreateInfo *create, uint32_t *type) {
    external_boundary();
    assert(a == device.vma && (info->format == VK_FORMAT_R8G8B8A8_UNORM || info->format == VK_FORMAT_D16_UNORM));
    *type = TEST_DEVICE_TYPE;
    if (create->memoryTypeBits) assert(create->memoryTypeBits == 1u << *type);
    return VK_SUCCESS;
}

static VkResult native_allocate(uint32_t type, uint64_t bytes, const VmaAllocationCreateInfo *create,
                               int is_image, unsigned shape, uint64_t *handle, VmaAllocation *allocation,
                               VmaAllocationInfo *info) {
    external_boundary();
    assert(!vk_exec_draw_fixed_ready() && !upload_calls);
    assert(create->memoryTypeBits == (1u << type));
    assert(create->flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    assert(!(create->flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT));
    RcBudgetSnapshot budget;
    resource_core_budget_snapshot(&budget);
    for (unsigned domain = 0; domain < RC_ADMISSION_MAX_DOMAINS; domain++)
        assert(budget.reserved.amount[domain] + budget.retained.amount[domain] == expected_peak.amount[domain]);
    if (fail_native == buffer_attempts + image_attempts) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    unsigned index = native_creates++;
    assert(index < NATIVE_COUNT);
    *handle = 100 + index;
    *allocation = (VmaAllocation)(uintptr_t)(index + 1);
    *info = (VmaAllocationInfo){.size = bytes, .memoryType = type,
        .pMappedData = type == TEST_HOST_TYPE && !no_mapping ? zero_mapping : NULL};
    allocations[index].handle = *handle;
    allocations[index].shape = shape;
    allocations[index].is_image = is_image;
    allocations[index].live = 1;
    return VK_SUCCESS;
}

VkResult vmaCreateBuffer(VmaAllocator a, const VkBufferCreateInfo *info,
                         const VmaAllocationCreateInfo *create, VkBuffer *out,
                         VmaAllocation *allocation, VmaAllocationInfo *actual) {
    assert(a == device.vma && ++buffer_attempts <= BUFFER_COUNT);
    uint32_t type = buffer_type(info);
    if (type == TEST_HOST_TYPE) {
        assert(create->usage == VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        assert(create->flags & VMA_ALLOCATION_CREATE_MAPPED_BIT);
        assert(create->flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
        assert(create->requiredFlags == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    } else {
        assert(create->usage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    }
    uint64_t handle = 0;
    VkResult result = native_allocate(type, info->size, create, 0, type, &handle, allocation, actual);
    *out = (VkBuffer)(uintptr_t)handle;
    return result;
}

VkResult vmaCreateImage(VmaAllocator a, const VkImageCreateInfo *info,
                        const VmaAllocationCreateInfo *create, VkImage *out,
                        VmaAllocation *allocation, VmaAllocationInfo *actual) {
    assert(a == device.vma && image_attempts < IMAGE_COUNT);
    unsigned shape = image_attempts++;
    assert(info->sType == VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO && info->format == shape_format(shape));
    assert(info->extent.width == 1 && info->extent.height == 1 && info->extent.depth == 1 && info->mipLevels == 1);
    assert(info->imageType == shapes[shape].type && info->flags == shapes[shape].flags);
    assert(info->samples == shapes[shape].samples && info->arrayLayers == shapes[shape].layers);
    assert(info->tiling == VK_IMAGE_TILING_OPTIMAL && info->initialLayout == VK_IMAGE_LAYOUT_UNDEFINED);
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (shape < DEPTH_INDEX && info->samples == VK_SAMPLE_COUNT_1_BIT) usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    assert(info->usage == usage && create->usage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    uint64_t handle = 0;
    VkResult result = native_allocate(TEST_DEVICE_TYPE, image_bytes(info), create, 1, shape, &handle, allocation, actual);
    *out = (VkImage)(uintptr_t)handle;
    return result;
}

static void destroy_allocation(VmaAllocator a, uint64_t handle, VmaAllocation allocation, int image) {
    external_boundary();
    assert(a == device.vma && handle >= 100 && handle < 100 + native_creates);
    unsigned index = (unsigned)handle - 100;
    assert(allocation == (VmaAllocation)(uintptr_t)(index + 1));
    assert(allocations[index].live && allocations[index].is_image == image && !allocations[index].nviews);
    assert(!upload_calls || completed_seq >= UPLOAD_SEQUENCE);
    assert(!vk_lost);
    allocations[index].live = 0;
    native_destroys++;
}

void vmaDestroyBuffer(VmaAllocator a, VkBuffer h, VmaAllocation alloc) {
    destroy_allocation(a, (uint64_t)(uintptr_t)h, alloc, 0);
}

void vmaDestroyImage(VmaAllocator a, VkImage h, VmaAllocation alloc) {
    destroy_allocation(a, (uint64_t)(uintptr_t)h, alloc, 1);
}

VkResult vmaFindMemoryTypeIndex(VmaAllocator a, uint32_t bits, const VmaAllocationCreateInfo *c, uint32_t *type) {
    (void)a; (void)bits; (void)c; (void)type; abort();
}
VkResult vmaAllocateMemory(VmaAllocator a, const VkMemoryRequirements *r, const VmaAllocationCreateInfo *c,
                          VmaAllocation *allocation, VmaAllocationInfo *info) {
    (void)a; (void)r; (void)c; (void)allocation; (void)info; abort();
}
VkResult vmaCreatePool(VmaAllocator a, const VmaPoolCreateInfo *create, VmaPool *pool) {
    (void)a; (void)create; (void)pool; abort();
}
void vmaFreeMemory(VmaAllocator a, VmaAllocation allocation) { (void)a; (void)allocation; abort(); }

static VkResult make_view(uint64_t handle, uint64_t *view) {
    external_boundary();
    assert(!vk_exec_draw_fixed_ready() && !upload_calls);
    assert(handle >= 100 && handle < 100 + native_creates);
    if (++view_attempts == fail_view) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    unsigned i = view_creates++, index = (unsigned)handle - 100;
    assert(i < VIEW_COUNT && allocations[index].live);
    allocations[index].nviews++;
    views[i].live = 1;
    views[i].owner = index;
    *view = 200 + i;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateBufferView(
    VkDevice d, const VkBufferViewCreateInfo *info, const VkAllocationCallbacks *callbacks, VkBufferView *out) {
    assert(d == device.device && !callbacks && info->sType == VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO);
    assert(info->format == VK_FORMAT_R32G32B32A32_SFLOAT && !info->offset && info->range == ZERO_BYTES);
    uint64_t handle = (uint64_t)(uintptr_t)info->buffer, view = 0;
    assert(handle >= 100 && handle < 100 + native_creates);
    assert(!allocations[handle - 100].is_image && allocations[handle - 100].shape == TEST_HOST_TYPE);
    VkResult result = make_view(handle, &view);
    *out = (VkBufferView)(uintptr_t)view;
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(
    VkDevice d, const VkImageViewCreateInfo *info, const VkAllocationCallbacks *callbacks, VkImageView *out) {
    assert(d == device.device && !callbacks && info->sType == VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO);
    uint64_t handle = (uint64_t)(uintptr_t)info->image, view = 0;
    assert(handle >= 100 && handle < 100 + native_creates && allocations[handle - 100].is_image);
    unsigned shape = allocations[handle - 100].shape;
    VkImageSubresourceRange range = info->subresourceRange;
    assert(info->format == shape_format(shape));
    assert(range.aspectMask == (shape >= DEPTH_INDEX ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT));
    assert(!range.baseMipLevel && range.levelCount == 1 && !range.baseArrayLayer && range.layerCount == shapes[shape].layers);
    if (allocations[handle - 100].nviews) {
        assert(shapes[shape].flags == VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);
        assert(info->viewType == VK_IMAGE_VIEW_TYPE_2D_ARRAY);
    } else {
        assert(info->viewType == shapes[shape].view);
    }
    VkResult result = make_view(handle, &view);
    *out = (VkImageView)(uintptr_t)view;
    return result;
}

static void destroy_view(VkDevice d, uint64_t view, const VkAllocationCallbacks *callbacks, int is_image) {
    external_boundary();
    assert(d == device.device && !callbacks && view >= 200 && view < 200 + view_creates);
    unsigned i = (unsigned)view - 200, owner = views[i].owner;
    assert(views[i].live && allocations[owner].live && allocations[owner].is_image == is_image);
    assert(allocations[owner].nviews && (!upload_calls || completed_seq >= UPLOAD_SEQUENCE) && !vk_lost);
    views[i].live = 0;
    allocations[owner].nviews--;
    view_destroys++;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(VkDevice d, VkImageView h, const VkAllocationCallbacks *callbacks) {
    destroy_view(d, (uint64_t)(uintptr_t)h, callbacks, 1);
}

VKAPI_ATTR void VKAPI_CALL vkDestroyBufferView(VkDevice d, VkBufferView h, const VkAllocationCallbacks *callbacks) {
    destroy_view(d, (uint64_t)(uintptr_t)h, callbacks, 0);
}

VkSampler vk_sampler_for(const samp_par *params) {
    external_boundary();
    assert(!vk_exec_draw_fixed_ready() && !upload_calls);
    assert(params->lod_max == 1000.0f && sampler_calls < 2);
    assert(params->cmp_mode == (int)sampler_calls);
    if (sampler_calls) assert(params->cmp_func == NVN_DEPTH_FUNC_ALWAYS);
    return (VkSampler)(uintptr_t)(500 + sampler_calls++);
}

VKAPI_ATTR void VKAPI_CALL vkDestroySampler(VkDevice d, VkSampler h, const VkAllocationCallbacks *callbacks) {
    (void)d; (void)h; (void)callbacks; abort(); // Fixed resources borrow sampler-cache handles.
}
VkCommandBuffer vk_upload_begin(void) {
    external_boundary();
    assert(!vk_exec_draw_fixed_ready() && !upload_open);
    assert(native_creates == NATIVE_COUNT && view_creates == VIEW_COUNT && sampler_calls == 2);
    for (unsigned i = 0; i < sizeof zero_mapping; i++) assert(!zero_mapping[i]);
    upload_calls++;
    upload_open = 1;
    return upload_cb;
}

void vk_upload_end(void) {
    external_boundary();
    assert(upload_open && !vk_exec_draw_fixed_ready());
    upload_open = 0;
    upload_ends++;
}

uint64_t vk_upload_recording_sequence(VkCommandBuffer cb) {
    assert(cb == upload_cb && upload_open && !vk_exec_draw_fixed_ready());
    return UPLOAD_SEQUENCE;
}

uint64_t vk_upload_flush(int wait) {
    external_boundary();
    assert(wait == 1 && !upload_open && upload_calls == upload_ends);
    assert(clears == IMAGE_COUNT && barriers == 2 * IMAGE_COUNT && !vk_exec_draw_fixed_ready());
    flushes++;
    submitted_seq = UPLOAD_SEQUENCE;
    completed_seq = uncertain_upload ? UPLOAD_SEQUENCE - 1 : UPLOAD_SEQUENCE;
    return submitted_seq;
}

void vk_upload_progress(uint64_t *submitted, uint64_t *completed) {
    external_boundary();
    assert(!vk_exec_draw_fixed_ready());
    if (submitted) *submitted = submitted_seq;
    if (completed) *completed = completed_seq;
}

static void check_range(VkImage image, VkImageSubresourceRange range, unsigned which) {
    uint64_t native = (uint64_t)(uintptr_t)image;
    assert(native >= 100 && native < 100 + native_creates);
    unsigned owner = (unsigned)native - 100;
    assert(allocations[owner].is_image && allocations[owner].live && allocations[owner].shape == which);
    assert(range.aspectMask == (which >= DEPTH_INDEX ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT));
    assert(!range.baseMipLevel && range.levelCount == 1);
    assert(!range.baseArrayLayer && range.layerCount == shapes[which].layers);
}

VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo *info) {
    assert(cb == upload_cb && upload_open && !vk_exec_draw_fixed_ready());
    assert(info->imageMemoryBarrierCount == 1 && !info->bufferMemoryBarrierCount && !info->memoryBarrierCount);
    const VkImageMemoryBarrier2 *barrier = info->pImageMemoryBarriers;
    assert(barrier->newLayout == VK_IMAGE_LAYOUT_GENERAL && barrier->srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED &&
           barrier->dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
    unsigned which = barriers / 2;
    check_range(barrier->image, barrier->subresourceRange, which);
    if (barriers % 2) {
        assert(clears == which + 1 && barrier->oldLayout == VK_IMAGE_LAYOUT_GENERAL);
        assert(barrier->srcStageMask == VK_PIPELINE_STAGE_2_CLEAR_BIT && barrier->srcAccessMask == VK_ACCESS_2_TRANSFER_WRITE_BIT);
        assert(barrier->dstStageMask == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT &&
               barrier->dstAccessMask == (VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT));
    } else {
        assert(clears == which && barrier->oldLayout == VK_IMAGE_LAYOUT_UNDEFINED);
        assert(!barrier->srcStageMask && !barrier->srcAccessMask);
        assert(barrier->dstStageMask == VK_PIPELINE_STAGE_2_CLEAR_BIT && barrier->dstAccessMask == VK_ACCESS_2_TRANSFER_WRITE_BIT);
    }
    barriers++;
}

VKAPI_ATTR void VKAPI_CALL vkCmdClearColorImage(
    VkCommandBuffer cb, VkImage image, VkImageLayout layout, const VkClearColorValue *value,
    uint32_t n, const VkImageSubresourceRange *range) {
    assert(cb == upload_cb && upload_open && !vk_exec_draw_fixed_ready());
    assert(layout == VK_IMAGE_LAYOUT_GENERAL && n == 1 && barriers == 2 * clears + 1);
    check_range(image, *range, clears);
    assert(clears < DEPTH_INDEX);
    assert(value->float32[0] == 0.5f && value->float32[1] == 0.5f &&
           value->float32[2] == 0.5f && value->float32[3] == 1.0f);
    clears++;
}

VKAPI_ATTR void VKAPI_CALL vkCmdClearDepthStencilImage(
    VkCommandBuffer cb, VkImage image, VkImageLayout layout, const VkClearDepthStencilValue *value,
    uint32_t n, const VkImageSubresourceRange *range) {
    assert(cb == upload_cb && upload_open && !vk_exec_draw_fixed_ready());
    assert(layout == VK_IMAGE_LAYOUT_GENERAL && n == 1 && clears >= DEPTH_INDEX && barriers == 2 * clears + 1);
    check_range(image, *range, clears);
    assert(value->depth == 0.5f && value->stencil == 0);
    clears++;
}

static void reset_attempt(void) {
    assert(!vk_exec_draw_fixed_ready() && native_creates == native_destroys && view_creates == view_destroys);
    resource_core_vma_test_reset();
    resource_core_budget_test_reset();
    buffer_attempts = image_attempts = native_creates = native_destroys = 0;
    view_attempts = view_creates = view_destroys = sampler_calls = 0;
    upload_calls = upload_ends = flushes = barriers = clears = 0;
    fail_native = fail_view = no_mapping = uncertain_upload = 0;
    submitted_seq = completed_seq = 0;
    stop_requested = stop_on_wait = 0;
    queue_handoffs = readiness_waits = 0;
    memset(allocations, 0, sizeof allocations);
    memset(views, 0, sizeof views);
    memset(zero_mapping, 0x5a, sizeof zero_mapping);
    device = (VkCtx){.vma = (VmaAllocator)(uintptr_t)1, .device = (VkDevice)(uintptr_t)1};
    device.props.deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    device.mem_props.memoryTypeCount = device.mem_props.memoryHeapCount = 2;
    device.mem_props.memoryTypes[TEST_DEVICE_TYPE] = (VkMemoryType){VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};
    device.mem_props.memoryTypes[TEST_HOST_TYPE] = (VkMemoryType){
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1};
    device.mem_props.memoryHeaps[0] = (VkMemoryHeap){8192 * MB, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    device.mem_props.memoryHeaps[1] = (VkMemoryHeap){8192 * MB, 0};
    sample = (RcBudgetSample){.known_mask = (1u << (HOST_HEAP + 1)) - 1};
    for (unsigned domain = 0; domain <= HOST_HEAP; domain++) {
        sample.total.amount[domain] = 8192 * MB;
        sample.free.amount[domain] = 4096 * MB;
    }
    uint64_t image_sum = 0;
    for (unsigned i = 0; i < IMAGE_COUNT; i++)
        image_sum += (uint64_t)4096 * shapes[i].layers * shapes[i].samples;
    expected_peak.amount[DEVICE_HEAP] = ZERO_BYTES + image_sum;
    expected_peak.amount[HOST_HEAP] = expected_peak.amount[RC_BUDGET_RAM] = ZERO_BYTES;
    expected_peak.amount[RC_BUDGET_COMMIT] = 2 * ZERO_BYTES + image_sum;
}

static void assert_charge(const RcAdmissionVector *expected) {
    RcBudgetSnapshot budget;
    resource_core_budget_snapshot(&budget);
    for (unsigned d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) {
        assert(!budget.reserved.amount[d] && !budget.pending.amount[d]);
        assert(budget.retained.amount[d] == expected->amount[d]);
    }
}

static void assert_rollback(void) {
    assert(!vk_exec_draw_fixed_ready() && native_creates == native_destroys && view_creates == view_destroys);
    assert(!upload_calls && !flushes && !clears && !barriers);
    assert_charge(&(RcAdmissionVector){{0}});
}

static void check_joint_refusal(void) {
    reset_attempt();
    sample.free.amount[DEVICE_HEAP] = 64 * MB + expected_peak.amount[DEVICE_HEAP] - 1;
    RcAdmissionVector refused = {{0}};
    assert(vk_exec_draw_fixed_prepare(&refused) == VK_NOT_READY);
    assert(!memcmp(&expected_peak, &refused, sizeof refused));
    assert(!buffer_attempts && !image_attempts && !view_attempts && !sampler_calls);
    assert_rollback();
}

static void check_failure_sweep(void) {
    for (unsigned i = 1; i <= NATIVE_COUNT; i++) {
        reset_attempt();
        fail_native = i;
        assert(vk_exec_draw_fixed_prepare(&(RcAdmissionVector){{0}}) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
        assert(buffer_attempts + image_attempts == i && native_creates == i - 1);
        assert_rollback();
    }
    for (unsigned i = 1; i <= VIEW_COUNT; i++) {
        reset_attempt();
        fail_view = i;
        assert(vk_exec_draw_fixed_prepare(&(RcAdmissionVector){{0}}) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
        assert(view_attempts == i && view_creates == i - 1);
        assert_rollback();
    }
    reset_attempt();
    no_mapping = 1;
    assert(vk_exec_draw_fixed_prepare(&(RcAdmissionVector){{0}}) == VK_ERROR_MEMORY_MAP_FAILED);
    assert(native_creates && !view_attempts);
    assert_rollback();
}

static void check_bootstrap_and_shutdown(void) {
    reset_attempt();
    sample.free.amount[DEVICE_HEAP] = 64 * MB + expected_peak.amount[DEVICE_HEAP] - 1;
    stop_on_wait = 1;
    assert(!vk_sync_prepare_draw());
    assert(queue_handoffs == 1 && readiness_waits == 1 && !native_creates);
    assert_rollback();

    reset_attempt();
    sample.free.amount[DEVICE_HEAP] = 64 * MB + expected_peak.amount[DEVICE_HEAP] - 1;
    assert(vk_sync_prepare_draw());
    assert(queue_handoffs == 1 && readiness_waits == 1);
    fixed_require();
    assert(vk_exec_draw_fixed_ready() && flushes == 1 && completed_seq == UPLOAD_SEQUENCE);
    assert(buffer_attempts == BUFFER_COUNT && image_attempts == IMAGE_COUNT && view_attempts == VIEW_COUNT);
    assert_charge(&expected_peak);
    assert(vk_exec_draw_fixed_prepare(&(RcAdmissionVector){{0}}) == VK_SUCCESS);
    assert(flushes == 1 && native_creates == NATIVE_COUNT && sampler_calls == 2);
    assert(Z.img[DV_CUBE].sview != VK_NULL_HANDLE);
    vk_exec_draw_fixed_shutdown();
    assert(!vk_exec_draw_fixed_ready() && view_destroys == VIEW_COUNT && native_destroys == NATIVE_COUNT);
    assert_charge(&(RcAdmissionVector){{0}});
    vk_exec_draw_fixed_shutdown();
    assert(view_destroys == VIEW_COUNT && native_destroys == NATIVE_COUNT);
}

static void check_uncertain(int lose_device) {
    reset_attempt();
    uncertain_upload = 1;
    assert(vk_exec_draw_fixed_prepare(&(RcAdmissionVector){{0}}) != VK_SUCCESS);
    assert(!vk_exec_draw_fixed_ready() && flushes == 1 && completed_seq == UPLOAD_SEQUENCE - 1);
    vk_exec_draw_fixed_shutdown();
    assert(!view_destroys && !native_destroys);
    assert_charge(&expected_peak);

    completed_seq = UPLOAD_SEQUENCE;
    vk_lost = lose_device;
    vk_exec_draw_fixed_shutdown();
    assert(!vk_exec_draw_fixed_ready());
    if (lose_device) {
        assert(!view_destroys && !native_destroys);
        assert_charge(&expected_peak);
    } else {
        assert(view_destroys == VIEW_COUNT && native_destroys == NATIVE_COUNT);
        assert_charge(&(RcAdmissionVector){{0}});
    }
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "extra-shapes")) {
        extra_depth_shapes = 1;
        check_failure_sweep();
        check_bootstrap_and_shutdown();
    } else if (argc > 1) {
        check_uncertain(!strcmp(argv[1], "lost"));
    } else {
        RcAdmissionVector unsupported;
        no_depth_compare = 1;
        assert(vk_exec_draw_fixed_prepare(&unsupported) == VK_ERROR_FORMAT_NOT_SUPPORTED);
        assert(!vk_exec_draw_fixed_ready() && !native_creates && !clears);
        no_depth_compare = 0;
        check_joint_refusal();
        check_failure_sweep();
        check_bootstrap_and_shutdown();
    }
    return 0;
}
'''

    cc = os.environ.get('CC') or shutil.which('clang')
    if not cc:
        raise SystemExit('Clang is required (set CC)')
    toolchain = Path(shutil.which(cc) or cc).resolve()
    vma = Path(os.environ.get('VMA_INCLUDE', toolchain.parent.parent / 'include' / 'vma'))
    env = dict(os.environ, PATH=str(toolchain.parent) + os.pathsep + os.environ.get('PATH', ''))
    with tempfile.TemporaryDirectory(prefix='fixed-draw-resources-') as directory:
        source = Path(directory) / 'check.c'
        exe = Path(directory) / ('check.exe' if os.name == 'nt' else 'check')
        source.write_text(fixture)
        subprocess.run([str(toolchain), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror',
                        '-fsanitize=undefined', '-fsanitize-trap=undefined',
                        '-I' + str(ROOT / 'runtime'), '-I' + str(vma), str(source), '-o', str(exe)],
                       env=env, check=True)
        for args in ([], ['uncertain'], ['lost'], ['extra-shapes']):
            subprocess.run([str(exe), *args], env=env, check=True)
    print('fixed draw resources: joint admission, native shapes, rollback, upload proof and teardown passed')


if __name__ == '__main__':
    main()

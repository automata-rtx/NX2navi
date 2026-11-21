"""Exercise the actual output-sized tonemap transaction and scene receipts with host-only Vulkan stubs."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def section(source, start, end):
    first = source.index(start)
    return source[first : source.index(end, first)]


PREFIX = r'''
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "host.h"
#include "vk_exec.h"
#include "vk_dlaa.h"
#include "resource_core_vma.h"

#define HANDLE(type, value) ((type)(uintptr_t)(value))

static VkCtx device;
static RcAdmission budget;
static RcAdmissionRecord records[24];
static struct {
    VkImage image;
    VkImageView view;
    VkImageCreateInfo info;
    RcBudgetTicket ticket;
} allocated[32];
static unsigned live, created, views, destroyed, transitions, requests, warnings;
static unsigned native_attempts, view_attempts, fail_native, fail_view, stop_after_native;
static int stopped;
static VkResult idle_result;
static unsigned idle_calls;
volatile int32_t vk_lost;
static RcAdmissionVector requested;

VkCtx *vk_ctx(void) { return &device; }
int vk_sync_stopping(void) { return stopped; }
VkResult vk_device_wait_idle(void) { idle_calls++; return idle_result; }
void vk_device_lost_check(VkResult result, const char *where) {
    assert(where);
    if (result == VK_ERROR_DEVICE_LOST) vk_lost = 1;
}
void rt_log(const char *fmt, ...) { assert(fmt); warnings++; }
_Noreturn void rt_fatal(const char *fmt, ...) { fprintf(stderr, "%s\n", fmt); abort(); }

RcAdmissionStatus resource_core_budget_reserve(RcBudgetProducer producer, int required, const RcAdmissionVector *peak,
                                               RcBudgetTicket *ticket) {
    assert(producer == RC_BUDGET_TEXTURE && !required);
    requests++;
    requested = *peak;
    ticket->producer = producer;
    return rc_admission_reserve(&budget, producer, peak, &ticket->token);
}

RcAdmissionStatus resource_core_budget_split(RcBudgetTicket *from, const RcAdmissionVector *cost,
                                              RcBudgetProducer producer, RcBudgetTicket *out) {
    out->producer = producer;
    return rc_admission_split(&budget, from->token, from->producer, cost, producer, &out->token);
}

RcAdmissionStatus resource_core_budget_materialize(RcBudgetTicket *ticket, const RcAdmissionVector *actual) {
    return rc_admission_materialize(&budget, ticket->token, ticket->producer, actual, &ticket->token);
}

RcAdmissionStatus resource_core_budget_cancel(RcBudgetTicket *ticket) {
    return rc_admission_cancel(&budget, ticket->token, ticket->producer);
}

RcAdmissionStatus resource_core_budget_retire(RcBudgetTicket *ticket) {
    return rc_admission_retire(&budget, ticket->token, ticket->producer, &ticket->token);
}

RcAdmissionStatus resource_core_budget_destroyed(RcBudgetTicket *ticket) {
    return rc_admission_destroyed(&budget, ticket->token, ticket->producer);
}

static RcAdmissionVector image_cost(VkExtent3D extent) {
    uint64_t bytes = (uint64_t)extent.width * extent.height * 4;
    return (RcAdmissionVector){.amount = {[RC_BUDGET_COMMIT] = bytes, [RC_BUDGET_GPU_FIRST] = bytes}};
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties(VkPhysicalDevice physical, VkFormat format,
    VkImageType type, VkImageTiling tiling, VkImageUsageFlags usage, VkImageCreateFlags flags,
    VkImageFormatProperties *properties) {
    assert(physical == device.phys && type == VK_IMAGE_TYPE_2D && tiling == VK_IMAGE_TILING_OPTIMAL && !flags);
    if (format == VK_FORMAT_D32_SFLOAT) assert(usage == VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    else assert(format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && usage ==
        (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
    *properties = (VkImageFormatProperties){.maxExtent = {8192, 8192, 1}};
    return VK_SUCCESS;
}

VkResult resource_core_vma_image_peak(const RcVmaRequest *request, const VkImageCreateInfo *image,
    const VmaAllocationCreateInfo *preferred, VmaAllocationCreateInfo *placement, RcAdmissionVector *peak) {
    assert(!request->required && preferred->flags == VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    *placement = *preferred;
    placement->memoryTypeBits = 1;
    *peak = image_cost(image->extent);
    return VK_SUCCESS;
}

VkResult resource_core_vma_create_image(const RcVmaRequest *request, VmaAllocator allocator,
    const VkImageCreateInfo *image, const VmaAllocationCreateInfo *placement, VkImage *out,
    VmaAllocation *memory, VmaAllocationInfo *info, RcAdmissionStatus *why) {
    assert(allocator == device.vma && !info && !why);
    assert(image->initialLayout == VK_IMAGE_LAYOUT_UNDEFINED && image->samples == VK_SAMPLE_COUNT_1_BIT);
    assert(!*out && !*memory);
    if (++native_attempts == fail_native) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    RcAdmissionVector cost = image_cost(image->extent);
    RcBudgetTicket ticket = {.producer = request->producer};
    if (request->peak_ticket) {
        assert(placement->memoryTypeBits == 1 && request->producer == RC_BUDGET_TEXTURE);
        assert(resource_core_budget_split(request->peak_ticket, &cost, request->producer, &ticket)
            == RC_ADMISSION_OK);
    } else {
        assert(request->producer == RC_BUDGET_OTHER && image->format == VK_FORMAT_R16_SFLOAT);
        if (rc_admission_reserve(&budget, request->producer, &cost, &ticket.token) != RC_ADMISSION_OK)
            return VK_NOT_READY;
    }
    unsigned i = ++created;
    assert(i < sizeof allocated / sizeof *allocated);
    allocated[i].image = *out = HANDLE(VkImage, i);
    allocated[i].info = *image;
    allocated[i].ticket = ticket;
    *memory = HANDLE(VmaAllocation, i);
    assert(resource_core_budget_materialize(&allocated[i].ticket, &cost) == RC_ADMISSION_OK);
    if (native_attempts == stop_after_native) stopped = 1;
    live++;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(VkDevice dev, const VkImageViewCreateInfo *info,
    const VkAllocationCallbacks *allocator, VkImageView *view) {
    unsigned i = (unsigned)(uintptr_t)info->image;
    assert(dev == device.device && !allocator && allocated[i].image && !allocated[i].view);
    assert(info->viewType == VK_IMAGE_VIEW_TYPE_2D && info->format == allocated[i].info.format);
    assert(info->subresourceRange.aspectMask ==
        (info->format == VK_FORMAT_D32_SFLOAT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT));
    if (++view_attempts == fail_view) return VK_ERROR_OUT_OF_HOST_MEMORY;
    allocated[i].view = *view = HANDLE(VkImageView, i);
    views++;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(VkDevice dev, VkImageView view, const VkAllocationCallbacks *allocator) {
    unsigned i = (unsigned)(uintptr_t)view;
    assert(dev == device.device && !allocator && allocated[i].image && allocated[i].view == view);
    allocated[i].view = VK_NULL_HANDLE;
    views--;
}

void resource_core_vma_destroy_image(VmaAllocator allocator, VkImage image, VmaAllocation memory) {
    unsigned i = (unsigned)(uintptr_t)image;
    assert(allocator == device.vma && image && memory == HANDLE(VmaAllocation, i));
    assert(allocated[i].image == image && !allocated[i].view);
    assert(resource_core_budget_retire(&allocated[i].ticket) == RC_ADMISSION_OK);
    assert(resource_core_budget_destroyed(&allocated[i].ticket) == RC_ADMISSION_OK);
    allocated[i].image = VK_NULL_HANDLE;
    destroyed++;
    live--;
}

typedef struct {
    uint64_t fence;
    void (*fn)(void *);
    void *arg;
    void (*cancel)(void *, int);
} Pending;

static Pending pending[16];
static unsigned n_pending;
void vk_completion_add_owned(uint64_t fence, void (*fn)(void *), void (*cancel)(void *, int), void *arg) {
    assert(fence && fn && cancel && arg && n_pending < sizeof pending / sizeof *pending);
    pending[n_pending++] = (Pending){fence, fn, arg, cancel};
}

VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo *info) {
    assert(cb && info->imageMemoryBarrierCount && info->imageMemoryBarrierCount <= 2);
    if (info->imageMemoryBarrierCount == 2) assert(n_pending);
    for (unsigned i = 0; i < info->imageMemoryBarrierCount; ++i) {
        const VkImageMemoryBarrier2 *b = &info->pImageMemoryBarriers[i];
        unsigned handle = (unsigned)(uintptr_t)b->image;
        assert(allocated[handle].view && b->oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
               b->newLayout == VK_IMAGE_LAYOUT_GENERAL);
        if (info->imageMemoryBarrierCount == 2) assert(b->dstStageMask & VK_PIPELINE_STAGE_2_TRANSFER_BIT);
        else assert(allocated[handle].info.format == VK_FORMAT_R16_SFLOAT);
        assert(b->subresourceRange.aspectMask == (i ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT));
    }
    transitions++;
}
'''


CASES = r'''
static void dispose(unsigned index, uint64_t done, uint64_t handed, int idle) {
    Pending item = pending[index];
    if (item.fence && pending_disposable(item, done, handed, idle)) {
        pending[index] = (Pending){0};
        pending_dispose(item, done, handed);
    }
}

static void assert_no_reservations(void) {
    for (unsigned d = 0; d < budget.domain_count; ++d)
        assert(!budget.reserved.amount[d] && !budget.pending.amount[d]);
}

static void check_alpha(VkCommandBuffer cb) {
    VkNgxImage image = {0};
    VmaAllocation memory = NULL;
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    const VkFormat format = VK_FORMAT_R16_SFLOAT;
    unsigned recorded = transitions;
    assert(alpha_image(cb, &image, &memory, 1280, 720, usage, format));
    VkNgxImage old = image;
    VmaAllocation old_memory = memory;
    assert(transitions == recorded + 1 && live == 1 && views == 1 && !idle_calls);

    // If the first command never went to the queue, the next recording still transitions the cached image.
    assert(alpha_image(cb, &image, &memory, 1280, 720, usage, format));
    assert(image.img == old.img && memory == old_memory && transitions == recorded + 2 && !idle_calls);

    unsigned heap = RC_BUDGET_GPU_FIRST;
    budget.domain[heap].limit = budget.retained.amount[heap];
    assert(!alpha_image(cb, &image, &memory, 1920, 1080, usage, format));
    assert(image.img == old.img && memory == old_memory && live == 1 && views == 1 && !idle_calls);
    budget.domain[heap].limit = UINT64_MAX;

    fail_view = view_attempts + 1;
    assert(!alpha_image(cb, &image, &memory, 1920, 1080, usage, format));
    assert(image.img == old.img && live == 1 && views == 1 && !idle_calls);
    fail_view = 0;

    idle_result = VK_TIMEOUT;
    assert(!alpha_image(cb, &image, &memory, 1920, 1080, usage, format));
    assert(image.img == old.img && live == 1 && views == 1 && idle_calls == 1);
    assert(transitions == recorded + 2);
    idle_result = VK_SUCCESS;

    assert(alpha_image(cb, &image, &memory, 1920, 1080, usage, format));
    assert(image.img != old.img && memory != old_memory && image.view && image.w == 1920 && image.h == 1080);
    assert(live == 1 && views == 1 && idle_calls == 2 && transitions == recorded + 3);
    free_alpha(&image, &memory);
    assert(!image.img && !memory && !live && !views);
    assert_no_reservations();
}

int main(void) {
    const RcAdmissionDomain domains[] = {
        {RC_ADMISSION_RAM, 0, UINT64_MAX}, {RC_ADMISSION_COMMIT, 0, UINT64_MAX},
        {RC_ADMISSION_STAGING, 0, UINT64_MAX}, {RC_ADMISSION_GPU_HEAP, 0, UINT64_MAX},
    };
    const unsigned heap = RC_BUDGET_GPU_FIRST;
    assert(rc_admission_init(&budget, 1, domains, sizeof domains / sizeof *domains, records,
                            sizeof records / sizeof *records) == RC_ADMISSION_OK);
    device.device = HANDLE(VkDevice, 1);
    device.phys = HANDLE(VkPhysicalDevice, 2);
    device.vma = HANDLE(VmaAllocator, 3);
    VkExec ex = {.cb = HANDLE(VkCommandBuffer, 4), .fence = 1};
    VkExtent2D first = {1280, 720}, second = {1920, 1080};
    assert(post_images(&ex, first));
    DlaaPostImages *old = post.frame;
    assert(post.cached == old && old->refs == 3 && live == 2 && views == 2 && transitions == 1);
    assert(budget.retained.amount[RC_BUDGET_RAM] == sizeof(DlaaPostImages));
    assert(requested.amount[heap] == (uint64_t)first.width * first.height * 8);
    assert(post_images(&ex, first) && old->refs == 3 && n_pending == 1 && requests == 1 && ex.recorded == 1);

    // A later scene uses the same frame generation even after its initial command has completed.
    ex.fence = 2;
    post_scene_use(&ex, old);
    dispose(0, 1, 2, 0);
    assert(n_pending == 2 && old->refs == 3 && transitions == 1);
    post_frame_end();
    assert(!post.frame && old->refs == 2);

    // Keep the currently cached pair selected through a refusal and each independent failure.
    ex.fence = 3;
    budget.domain[heap].limit = budget.retained.amount[heap] + (uint64_t)second.width * second.height * 8 - 1;
    assert(!post_images(&ex, second) && post.cached == old && !post.frame && native_attempts == 2 && warnings == 1);
    assert_no_reservations();
    budget.domain[heap].limit = UINT64_MAX;
    for (unsigned i = 1; i <= 2; ++i) {
        fail_native = native_attempts + i;
        assert(!post_images(&ex, second) && post.cached == old && live == 2 && views == 2);
        fail_native = 0;
        fail_view = view_attempts + i;
        assert(!post_images(&ex, second) && post.cached == old && live == 2 && views == 2);
        fail_view = 0;
        assert_no_reservations();
    }
    assert(!post.frame && n_pending == 2 && transitions == 1);
    stop_after_native = native_attempts + 1;
    assert(!post_images(&ex, second) && stopped && post.cached == old && live == 2 && views == 2);
    stopped = 0;
    stop_after_native = 0;
    assert_no_reservations();

    // Full publication holds both generations until the last original scene retires.
    assert(post_images(&ex, second) && post.frame != old && post.cached == post.frame);
    assert(post.frame->size.width == second.width && post.frame->size.height == second.height);
    assert(live == 4 && views == 4 && old->refs == 1 && transitions == 2);
    dispose(1, 1, 2, 0);
    assert(live == 4); // A possibly issued scene still owns the exact old pair.
    dispose(1, 2, 2, 0);
    assert(live == 2 && views == 2);

    // Unsubmitted cancellation revokes initialized layout and makes the same size cold again.
    dispose(2, 2, 2, 0);
    assert(!post_current(post.frame));
    post_frame_end();
    assert(!post.cached && !post.frame && !live && !views);
    ex.fence = 4;
    assert(post_images(&ex, second) && transitions == 3 && live == 2);
    post_frame_end();
    dispose(3, 3, 4, 0);
    assert(live == 2);
    dispose(3, 4, 4, 0);
    release_post();
    assert(!post.cached && !post.frame && !live && !views && destroyed == created);
    assert_no_reservations();
    check_alpha(ex.cb);
    for (unsigned d = 0; d < budget.domain_count; ++d) assert(!budget.retained.amount[d]);
    puts("Temporal post and alpha: joint resize, rollback, scene retirement and unsent reuse passed");
    return 0;
}
'''


def main():
    source = (ROOT / "runtime/vk_dlaa.c").read_text()
    sync = (ROOT / "runtime/vk_sync.c").read_text()
    begin = section(source, "void vk_dlaa_tonemap_begin(", "void vk_dlaa_tonemap_end(")
    assert begin.index("post_scene_use(ex, images)") < begin.index("images->member[POST_COLOR]")
    forward = section(source, "int vk_dlaa_scene_upscale(", "// Two frame shapes.")
    assert forward.index("post_scene_use(ex, images)") < forward.index("blit_scene(ex->cb, src")
    frame = section(source, "void vk_dlaa_frame(", "\n    dlss_mode();")
    assert frame.index("post_frame_end();") < frame.index("D.early = D.promoted = 0")
    program = PREFIX
    program += section(source, "static VkImageCreateInfo dlaa_image_2d_info", "enum { DLAA_SIZE_NATIVE_IMAGES")
    program += section(source, "enum { POST_COLOR, POST_DEPTH, POST_IMAGES };", "// Which features may stay")
    program += section(source, "static void free_alpha(", "// The device is idle here.")
    program += section(source, "static int alpha_image(", "static void clear_alpha(")
    program += section(sync, "static int pending_disposable(", "static void pending_dispose_queue(")
    program += CASES
    compiler = os.environ.get("CC") or shutil.which("clang")
    if not compiler:
        raise SystemExit("Clang is required (set CC)")
    clang = Path(shutil.which(compiler) or compiler).resolve()
    vma = Path(os.environ.get("VMA_INCLUDE", clang.parent.parent / "include" / "vma"))
    env = dict(os.environ, PATH=str(clang.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="temporal-post-check-") as directory:
        c = Path(directory) / "check.c"
        exe = c.with_suffix(".exe" if os.name == "nt" else ".out")
        c.write_text(program)
        subprocess.run(
            [str(clang), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
             "-fsanitize-trap=undefined", "-I" + str(ROOT / "runtime"), "-I" + str(vma), str(c), "-o", str(exe)],
            env=env,
            check=True,
        )
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()


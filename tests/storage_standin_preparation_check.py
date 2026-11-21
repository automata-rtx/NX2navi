"""Exercise the exact typed stand-in transaction without a GPU."""

from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

repo = Path(__file__).resolve().parents[1]
draw = (repo / "runtime/vk_exec_draw.c").read_text(encoding="utf-8")
sys.path.insert(0, str(repo / "tests"))
from bind_recipe_check import function

start = draw.index("enum { DV_1D,")
stop = draw.index("static void fixed_draw_destroy(", start)
definition = draw.index("// ---- resource types", stop)
actual = draw[start:stop]
for name in ("res_view_type", "dummy_index"):
    actual += "\n" + function(draw[definition:], name)
before = r"""
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "platform/platform.h"
#include "resource_core_vma.h"
#include "vk_exec.h"
#define ZERO_BYTES 65536
"""
after = r"""
static VkCtx ctx;
volatile int32_t vk_lost;
static unsigned locks, reserves, creates, views, cleared, destroyed, views_destroyed, uploads, waits;
static int parked, refused, fail_view, uncertain;
static uint64_t completed;
static int healthy;
static int fail_format;

VkCtx *vk_ctx(void) { return &ctx; }
int vk_sync_stopping(void) { return 0; }
void vk_texture_lock(void) { assert(!parked && !locks); locks++; }
void vk_texture_unlock(void) { assert(!parked && locks == 1); locks--; }
void rt_fatal(const char *format, ...) { (void)format; abort(); }
VkSyncProgressResult vk_exec_graphics_progress_prepare(VkExec *ex, int (*prepare)(void *), void *context) {
    assert(ex && !parked && !locks);
    parked = 1;
    assert(prepare(context));
    parked = 0;
    return VK_SYNC_PROGRESS_COMPLETE;
}
VkSyncProgressResult vk_exec_texture_capacity(VkExec *ex, const RcVmaDependency *dependency, VkResult failure,
    int (*ready)(void *), void *context, uint64_t *prior) {
    assert(ex && failure != VK_SUCCESS && !ready && !context && prior && !locks && !parked);
    if (!dependency->selected || (failure != VK_NOT_READY && failure != VK_ERROR_OUT_OF_DEVICE_MEMORY &&
        failure != VK_ERROR_OUT_OF_HOST_MEMORY)) return VK_SYNC_PROGRESS_UNAVAILABLE;
    waits++;
    return VK_SYNC_PROGRESS_UNAVAILABLE;
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties(VkPhysicalDevice device,
    VkFormat format, VkImageType type, VkImageTiling tiling, VkImageUsageFlags usage, VkImageCreateFlags flags,
    VkImageFormatProperties *support) {
    assert(device == ctx.phys && (format == VK_FORMAT_R16_UNORM || format == VK_FORMAT_R32_UINT));
    assert(type == VK_IMAGE_TYPE_2D && tiling == VK_IMAGE_TILING_OPTIMAL && !flags);
    assert(usage == (VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT) && parked && !locks);
    if (fail_format) return VK_ERROR_FORMAT_NOT_SUPPORTED;
    *support = (VkImageFormatProperties){.maxArrayLayers = 1, .sampleCounts = VK_SAMPLE_COUNT_1_BIT};
    return VK_SUCCESS;
}
VkResult resource_core_vma_image_peak(const RcVmaRequest *request, const VkImageCreateInfo *info,
    const VmaAllocationCreateInfo *input, VmaAllocationCreateInfo *placement, RcAdmissionVector *peak) {
    assert(parked && !locks && request->required && request->producer == RC_BUDGET_TEXTURE);
    assert(info->extent.width == 1 && info->mipLevels == 1 && input->flags == VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    *placement = *input;
    *peak = (RcAdmissionVector){0};
    peak->amount[RC_BUDGET_GPU_FIRST] = 4096;
    return VK_SUCCESS;
}
void resource_core_budget_readiness(int required, RcBudgetReadiness *out) {
    assert(required && parked && !locks);
    *out = (RcBudgetReadiness){0};
}
RcAdmissionStatus resource_core_budget_reserve(RcBudgetProducer producer, int required,
    const RcAdmissionVector *peak, RcBudgetTicket *out) {
    assert(parked && !locks && producer == RC_BUDGET_TEXTURE && required);
    assert(peak->amount[RC_BUDGET_GPU_FIRST] == 8192);
    reserves++;
    if (refused) return RC_ADMISSION_REFUSED;
    out->producer = producer;
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_cancel(RcBudgetTicket *ticket) {
    assert(parked && !locks && ticket->producer == RC_BUDGET_TEXTURE);
    return RC_ADMISSION_OK;
}
VkResult resource_core_vma_create_image(const RcVmaRequest *request, VmaAllocator allocator,
    const VkImageCreateInfo *info, const VmaAllocationCreateInfo *placement, VkImage *image,
    VmaAllocation *allocation, VmaAllocationInfo *details, RcAdmissionStatus *admission) {
    assert(parked && !locks && allocator == ctx.vma && request->peak_ticket->producer == RC_BUDGET_TEXTURE);
    assert(!details && !admission && placement->flags == VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    assert(info->format == VK_FORMAT_R16_UNORM || info->format == VK_FORMAT_R32_UINT);
    *image = (VkImage)(uintptr_t)(100 + ++creates);
    *allocation = (VmaAllocation)(uintptr_t)creates;
    assert(!SD[DV_2D][6].sview && !SD[DV_2D][11].sview);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(VkDevice d, const VkImageViewCreateInfo *ci,
    const VkAllocationCallbacks *alloc, VkImageView *view) {
    assert(parked && !locks && d == ctx.device && !alloc && ci->viewType == VK_IMAGE_VIEW_TYPE_2D);
    assert(ci->format == VK_FORMAT_R16_UNORM || ci->format == VK_FORMAT_R32_UINT);
    assert(ci->subresourceRange.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT);
    if (++views == 2 && fail_view) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    *view = (VkImageView)(uintptr_t)(200 + views);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(VkDevice device, VkImageView view, const VkAllocationCallbacks *alloc) {
    assert(device == ctx.device && view && !alloc && !locks);
    assert(!uploads || completed >= 77);
    views_destroyed++;
}
void resource_core_vma_destroy_image(VmaAllocator vma, VkImage image, VmaAllocation alloc) {
    assert(vma == ctx.vma && image && alloc && !locks);
    assert(!uploads || completed >= 77);
    destroyed++;
}
VkCommandBuffer vk_upload_begin(void) {
    assert(parked && !locks && creates == 2 && views == 2);
    uploads++;
    return (VkCommandBuffer)(uintptr_t)1;
}
void vk_upload_end(void) { assert(parked && !locks && uploads); }
uint64_t vk_upload_recording_sequence(VkCommandBuffer cb) { assert(cb && parked); return 77; }
uint64_t vk_upload_flush(int wait) { assert(parked && wait && cleared == 2); return 77; }
void vk_upload_progress(uint64_t *submitted, uint64_t *finished) {
    assert(!submitted && finished && !locks);
    *finished = completed = (uncertain && !healthy) ? 0 : 77;
}
VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo *di) {
    assert(cb && di->imageMemoryBarrierCount == 1 && parked && !locks);
}
VKAPI_ATTR void VKAPI_CALL vkCmdClearColorImage(VkCommandBuffer cb, VkImage image,
    VkImageLayout layout, const VkClearColorValue *clear, uint32_t count, const VkImageSubresourceRange *range) {
    assert(cb && image && count == 1 && layout == VK_IMAGE_LAYOUT_GENERAL && range->levelCount == 1);
    assert(!clear->uint32[0] && !clear->uint32[1] && !clear->uint32[2] && !clear->uint32[3]);
    assert(parked && !locks && !SD[DV_2D][6].sview && !SD[DV_2D][11].sview);
    cleared++;
}
int main(int argc, char **argv) {
    assert(argc == 2);
    const char *mode = argv[1];
    const GfxShaderRes a = {.type = 3, .format = 6}, b = {.type = 3, .format = 11};
    const VkPipeBinding bindings[] = {
        {.kind = GFX_RES_IMG, .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .res = &a},
        {.kind = GFX_RES_IMG, .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .res = &b},
        {.kind = GFX_RES_IMG, .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .res = &a},
    };
    const VkPipeInfo layout = {.n_bindings = 3, .bindings = bindings};
    VkExec ex = {0};
    assert(vk_exec_draw_storage_standins_prepare(&ex, &layout, 0) == VK_NOT_READY);
    assert(!reserves && !creates && !uploads);
    refused = !strcmp(mode, "refused");
    fail_view = !strcmp(mode, "view-failure");
    uncertain = !strcmp(mode, "uncertain");
    fail_format = !strcmp(mode, "unavailable-format");
    VkResult result = vk_exec_draw_storage_standins_prepare(&ex, &layout, 1);
    if (refused) assert(result == VK_NOT_READY && !creates && !uploads && waits == 1);
    else if (fail_view) {
        assert(result == VK_ERROR_OUT_OF_DEVICE_MEMORY && creates == 2 && destroyed == 2 && views_destroyed == 1);
        assert(!uploads && !SD[DV_2D][6].sview && !SD[DV_2D][11].sview);
    } else if (fail_format) assert(result == VK_ERROR_FORMAT_NOT_SUPPORTED && !creates && !uploads && !reserves);
    else if (uncertain) {
        assert(result == VK_ERROR_INITIALIZATION_FAILED && uploads == 1 && storage_unproven_count == 2);
        assert(!destroyed && !SD[DV_2D][6].sview && !SD[DV_2D][11].sview);
        healthy = 1;
        storage_standins_shutdown(&ctx);
        assert(destroyed == 2 && views_destroyed == 2);
    } else {
        assert(result == VK_SUCCESS && creates == 2 && views == 2 && reserves == 1 && cleared == 2 && !destroyed);
        assert(storage_dummy(DV_2D, 6) && storage_dummy(DV_2D, 11) && storage_dummy(DV_2D, 6) != storage_dummy(DV_2D, 11));
        assert(vk_exec_draw_storage_standins_prepare(&ex, &layout, 0) == VK_SUCCESS && reserves == 1);
        storage_standins_shutdown(&ctx);
        assert(destroyed == 2 && views_destroyed == 2);
    }
    return 0;
}
"""
cc = os.environ.get("CC") or shutil.which("clang")
if not cc:
    raise SystemExit("Clang is required (set CC)")
clang = Path(shutil.which(cc) or cc).resolve()
vma = Path(os.environ.get("VMA_INCLUDE", clang.parent.parent / "include/vma"))
env = dict(os.environ, PATH=str(clang.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="storage-standin-") as directory:
    source = Path(directory) / "check.c"
    source.write_text(before + actual + after)
    exe = Path(directory) / ("check.exe" if os.name == "nt" else "check")
    subprocess.run([str(clang), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-Wno-unused-variable", "-Wno-unused-function", "-fsanitize=undefined", "-fsanitize-trap=all",
                    "-I" + str(repo/"runtime"), "-I" + str(vma),
                    str(source), "-o", str(exe)], env=env, check=True)
    for mode in ("ordinary", "refused", "view-failure", "uncertain", "unavailable-format"):
        subprocess.run([str(exe), mode], env=env, check=True)
print("Typed storage stand-ins: aggregate admission, independent exact formats, rollback, no early publication and uncertain upload custody PASS")

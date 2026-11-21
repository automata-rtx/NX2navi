"""Check admitted scratch images, tonemap views and scene retirement for fallback blits."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(os.environ.get("NX2_FALLBACK_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
DRAW = Path(os.environ.get("NX2_FALLBACK_DRAW", ROOT / "runtime/vk_exec_draw.c"))


def main():
    draw = DRAW.read_text()
    start = draw.index("// The ordered dispatcher always overwrites these temporary images")
    end = draw.index("// 1 when the bound compute program uses tensor-core", start)
    sync = (ROOT / "runtime/vk_sync.c").read_text()
    pending_type = next(line for line in sync.splitlines() if line.endswith("} Pending;"))
    start_sync = sync.index("static int pending_disposable(")
    end_sync = sync.index("static void pending_dispose_queue(", start_sync)
    fixture = r'''
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "resource_core_vma.h"
#include "platform/platform.h"
#undef vkCmdPipelineBarrier2

typedef struct { uint64_t fence; VkCommandBuffer cb; } VkExec;
static VkCtx ctx;
static PlatMutex *held;
static unsigned reported, progress, refused, reject_view, initializations, allocations, destructions, views;
static RcBudgetReclaimFn reclaim;
typedef struct { VkImage image; VkImageView view; VmaAllocation memory; } NativeImage;
static NativeImage native[48];

void plat_mutex_lock(PlatMutex *mutex) { assert(!held); held = mutex; }
void plat_mutex_unlock(PlatMutex *mutex) { assert(held == mutex); held = NULL; }
void plat_cond_wait(PlatCond *condition, PlatMutex *mutex) { (void)condition; (void)mutex; abort(); }
void plat_cond_broadcast(PlatCond *condition) { assert(condition); }
void plat_once(PlatOnce *once, void (*fn)(void *), void *context) {
    if (!once->state) { once->state = 2; fn(context); }
}
VkCtx *vk_ctx(void) { return &ctx; }
void rt_log(const char *text, ...) { assert(text); reported++; }
_Noreturn void rt_fatal(const char *text, ...) { fputs(text, stderr); abort(); }
void vk_device_lost_check(VkResult result, const char *operation) {
    assert(result != VK_ERROR_DEVICE_LOST && operation);
}
static int knob_tm_dispatch(void) { return 1; }

RcAdmissionStatus resource_core_budget_register_reclaimer(RcBudgetProducer producer, uint32_t domains,
                                                          RcBudgetReclaimFn fn, void *context) {
    assert(!held && !reclaim && producer == RC_BUDGET_TEXTURE && domains && fn && !context);
    assert(!(domains & (1u << RC_BUDGET_STAGING)));
    reclaim = fn;
    return RC_ADMISSION_OK;
}
void resource_core_budget_progress(void) { assert(!held); progress++; }

static NativeImage *find_image(VkImage image) {
    for (unsigned i = 0; i < allocations; i++)
        if (native[i].image == image) return &native[i];
    abort();
}
VkResult resource_core_vma_create_image(const RcVmaRequest *request, VmaAllocator allocator,
        const VkImageCreateInfo *create, const VmaAllocationCreateInfo *placement, VkImage *image,
        VmaAllocation *allocation, VmaAllocationInfo *info, RcAdmissionStatus *admission) {
    assert(!held && allocator == ctx.vma && request->producer == RC_BUDGET_TEXTURE && !request->required);
    assert(!info && !admission && !*image && !*allocation && allocations < 48);
    assert(placement->usage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    assert(create->imageType == VK_IMAGE_TYPE_2D && create->tiling == VK_IMAGE_TILING_OPTIMAL);
    assert(create->extent.width && create->extent.height == 47 && create->extent.depth == 1);
    assert(create->mipLevels == 1 && create->arrayLayers == 1 && create->samples == VK_SAMPLE_COUNT_1_BIT);
    assert((create->usage & (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)) ==
           (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
    assert(!!(create->usage & VK_IMAGE_USAGE_SAMPLED_BIT) == (create->format == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32));
    if (refused) return VK_NOT_READY;
    NativeImage *entry = &native[allocations++];
    entry->image = *image = (VkImage)(uintptr_t)(100 + allocations);
    entry->memory = *allocation = (VmaAllocation)(uintptr_t)(200 + allocations);
    return VK_SUCCESS;
}
RcAdmissionStatus resource_core_vma_allocation_cost(VmaAllocation allocation, RcAdmissionVector *out) {
    assert(!held && allocation);
    *out = (RcAdmissionVector){0};
    out->amount[RC_BUDGET_GPU_FIRST] = 4096;
    return RC_ADMISSION_OK;
}
void resource_core_vma_destroy_image(VmaAllocator allocator, VkImage image, VmaAllocation memory) {
    assert(!held && allocator == ctx.vma);
    NativeImage *entry = find_image(image);
    assert(entry->memory == memory && !entry->view);
    entry->memory = NULL;
    destructions++;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(VkDevice device, const VkImageViewCreateInfo *info,
                                                const VkAllocationCallbacks *callbacks, VkImageView *out) {
    NativeImage *entry = find_image(info->image);
    assert(!held && device == ctx.device && !callbacks && !entry->view && entry->memory);
    assert(info->viewType == VK_IMAGE_VIEW_TYPE_2D && info->format == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32);
    if (reject_view) return VK_ERROR_OUT_OF_HOST_MEMORY;
    entry->view = *out = (VkImageView)(uintptr_t)(300 + ++views);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(VkDevice device, VkImageView view,
                                            const VkAllocationCallbacks *callbacks) {
    assert(!held && device == ctx.device && !callbacks);
    for (unsigned i = 0; i < allocations; i++) if (native[i].view == view) {
        assert(native[i].memory);
        native[i].view = VK_NULL_HANDLE;
        return;
    }
    abort();
}
VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo *dependency) {
    assert(!held && cb && dependency->imageMemoryBarrierCount == 1);
    const VkImageMemoryBarrier2 *barrier = dependency->pImageMemoryBarriers;
    assert(find_image(barrier->image)->memory);
    assert(barrier->oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && barrier->newLayout == VK_IMAGE_LAYOUT_GENERAL);
    assert(barrier->srcStageMask == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    assert(barrier->dstAccessMask == VK_ACCESS_2_TRANSFER_WRITE_BIT);
    initializations++;
}
''' + pending_type + r'''
static Pending callbacks[64];
static unsigned callback_count;
void vk_completion_add_owned(uint64_t fence, void (*fn)(void *), void (*cancel)(void *, int), void *arg) {
    assert(!held && fn && cancel && arg && callback_count < 64);
    callbacks[callback_count++] = (Pending){fence, fn, arg, cancel};
}
''' + sync[start_sync:end_sync] + draw[start:end] + r'''
int main(void) {
    VkExec ex = {.fence = 1, .cb = (VkCommandBuffer)(uintptr_t)1};
    VkImageView view = (VkImageView)(uintptr_t)99;
    VkFormat format = VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    RcAdmissionVector gpu = {0}, commit = {0};
    gpu.amount[RC_BUDGET_GPU_FIRST] = 1;
    commit.amount[RC_BUDGET_COMMIT] = 1;

    refused = 1;
    assert(!fallback_scratch(&ex, 11, 47, format, &view) && !view);
    assert(!allocations && !initializations && !callback_count && reported == 1);
    refused = 0;
    VkImage image = fallback_scratch(&ex, 11, 47, format, &view);
    assert(image && view && allocations == 1 && initializations == 1 && callback_count == 1);
    assert(reclaim(NULL, &gpu) == RC_BUDGET_RECLAIM_NONE && !destructions);
    assert(!pending_disposable(callbacks[0], 0, 1, 0)); // May have executed, no proof yet.
    assert(pending_disposable(callbacks[0], 0, 0, 0));
    pending_dispose(callbacks[0], 0, 0); // Unsent: first layout transition did not execute.

    ex.fence = 2;
    assert(fallback_scratch(&ex, 11, 47, format, &view) == image && view);
    assert(allocations == 1 && initializations == 2);
    ex.fence = 3;
    assert(fallback_scratch(&ex, 11, 47, format, &view) == image && view);
    pending_dispose(callbacks[1], 2, 3);
    assert(reclaim(NULL, &gpu) == RC_BUDGET_RECLAIM_NONE && !destructions);
    assert(pending_disposable(callbacks[2], 2, 3, 1));
    pending_dispose(callbacks[2], 2, 3);
    assert(reclaim(NULL, &commit) == RC_BUDGET_RECLAIM_NONE);
    assert(reclaim(NULL, &gpu) == RC_BUDGET_RECLAIM_DESTROYED && destructions == 1);
    assert(progress == 2);

    ex.fence = 4;
    reject_view = 1;
    assert(fallback_scratch(&ex, 11, 47, format, &view) && !view && reported == 2);
    reject_view = 0;
    for (unsigned i = 1; i < FALLBACK_SCRATCH_SLOTS; ++i)
        assert(fallback_scratch(&ex, 11 + i, 47, VK_FORMAT_R8G8B8A8_UNORM, NULL));
    assert(!fallback_scratch(&ex, 33, 47, format, &view) && !view);
    assert(reported == 3 && allocations == 1 + FALLBACK_SCRATCH_SLOTS);
    pending_dispose(callbacks[3], 4, 4);
    ex.fence = 5;
    assert(fallback_scratch(&ex, 33, 47, format, &view) && view);
    assert(allocations == 2 + FALLBACK_SCRATCH_SLOTS && destructions == 2);
    for (unsigned i = 4; i < callback_count; ++i) pending_dispose(callbacks[i], 5, 5);
    fallback_scratch_shutdown(&ctx);
    assert(destructions == allocations);
    assert(reclaim(NULL, &gpu) == RC_BUDGET_RECLAIM_NONE);
    return 0;
}
'''

    cc = os.environ.get("CC") or shutil.which("clang")
    if not cc:
        raise SystemExit("Clang is required (set CC)")
    compiler = Path(shutil.which(cc) or cc).resolve()
    vma = Path(os.environ.get("VMA_INCLUDE", compiler.parent.parent / "include" / "vma"))
    env = dict(os.environ, PATH=str(compiler.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="fallback-scratch-") as directory:
        source = Path(directory) / "check.c"
        exe = Path(directory) / ("check.exe" if os.name == "nt" else "check")
        source.write_text(fixture)
        subprocess.run([str(compiler), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=undefined",
                        "-I" + str(ROOT / "runtime"), "-I" + str(vma), str(source), "-o", str(exe)],
                       env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)
    print("fallback scratch: admission, repeated transitions, scene retirement and view-first destruction passed")


if __name__ == "__main__":
    main()

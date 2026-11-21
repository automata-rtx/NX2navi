"""Exercise actual separate-present and camera-motion allocation/rollback."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runtime"


def section(s, first, last):
    start = s.index(first)
    return s[start:s.index(last, start + len(first))]


def function(s, name):
    start = re.search(r"^(?:static )?[^;{}\n]*\b" + name + r"\([^;{}]*\)\s*\{", s, re.M)
    assert start, name
    return s[start.start():s.index("\n}", start.end()) + 2]


PRELUDE = r'''
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk.h"
#include "resource_core_vma.h"

#undef vkCreateImageView
#undef vkDestroyImageView
#undef vkCreateDescriptorSetLayout
#undef vkDestroyDescriptorSetLayout
#undef vkCreatePipelineLayout
#undef vkDestroyPipelineLayout
#undef vkCreateSampler
#undef vkDestroySampler
#undef vkCreateShaderModule
#undef vkDestroyShaderModule
#undef vkCreateComputePipelines
#undef vkDestroyPipeline

#define HANDLE(type, n) ((type)(uintptr_t)(n))
#define FRAMES 2
static VkCtx ctx = {.device = HANDLE(VkDevice, 1), .vma = HANDLE(VmaAllocator, 2)};
static VkExtent2D g_sc_ext = {1920, 1080};
static VkFormat g_sc_fmt = VK_FORMAT_R8G8B8A8_UNORM;
static const uint32_t dlaa_mv_spv[] = {0x07230203};
static struct {
    int mv_ok;
    VkBuffer camb, mvm;
    VmaAllocation camb_mem, mvm_mem;
    unsigned char *mvm_map;
    VkPipeline mv_pipe;
    VkPipelineLayout mv_layout;
    VkDescriptorSetLayout mv_dsl;
    VkSampler mv_samp;
} D = {.mv_ok = -1};
static unsigned char mapped_byte;
static int pressure, stopped, missing_map, fail_child, fail_pipeline, fail_view;
static unsigned image_requests, image_destroyed, view_destroyed, buffer_requests, buffer_destroyed;
static unsigned reserve_requests, reservations_canceled, readiness_requests, native_created, native_destroyed;
static int retired, idle, old_image_live = 1, new_image_live, buffers_live;
static RcAdmission ledger; // token owner marker for mocked admission.

VkCtx *vk_ctx(void) { return &ctx; }
int vk_sync_stopping(void) { return stopped; }
volatile int32_t vk_lost;
void vk_device_lost_check(VkResult r, const char *where) {
    assert(r != VK_SUCCESS && where);
}
void rt_log(const char *fmt, ...) { assert(fmt); }
void rt_fatal(const char *fmt, ...) { assert(fmt); abort(); }

VkResult resource_core_vma_create_image(const RcVmaRequest *r, VmaAllocator a,
    const VkImageCreateInfo *i, const VmaAllocationCreateInfo *p, VkImage *out,
    VmaAllocation *memory, VmaAllocationInfo *info, RcAdmissionStatus *why) {
    (void)info; (void)why;
    assert(a == ctx.vma && r->required && !r->peak_ticket && r->producer == RC_BUDGET_OTHER);
    assert(i->extent.width == g_sc_ext.width && i->extent.height == g_sc_ext.height && i->format == g_sc_fmt);
    assert(p->flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    assert(old_image_live && retired && !new_image_live);
    image_requests++;
    if (pressure) return VK_NOT_READY;
    new_image_live = 1;
    *out = HANDLE(VkImage, 20);
    *memory = HANDLE(VmaAllocation, 21);
    return VK_SUCCESS;
}
void resource_core_vma_destroy_image(VmaAllocator a, VkImage image, VmaAllocation memory) {
    assert(a == ctx.vma && retired);
    if (image == HANDLE(VkImage, 10)) {
        assert(old_image_live && new_image_live && memory == HANDLE(VmaAllocation, 11));
        old_image_live = 0;
    } else {
        assert(image == HANDLE(VkImage, 20) && new_image_live && memory == HANDLE(VmaAllocation, 21));
        new_image_live = 0;
    }
    image_destroyed++;
}
VkResult vkCreateImageView(VkDevice device, const VkImageViewCreateInfo *i,
    const VkAllocationCallbacks *cb, VkImageView *out) {
    (void)cb;
    assert(device == ctx.device && i->image == HANDLE(VkImage, 20) && new_image_live && old_image_live);
    *out = HANDLE(VkImageView, fail_view ? 99 : 22); // a failed write does not own the result
    return fail_view ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS;
}
void vkDestroyImageView(VkDevice dev, VkImageView view, const VkAllocationCallbacks *cb) {
    (void)cb;
    assert(dev == ctx.device && (view == HANDLE(VkImageView, 12) || view == HANDLE(VkImageView, 22)));
    view_destroyed++;
}

VkResult resource_core_vma_buffer_peak(const RcVmaRequest *r, const VkBufferCreateInfo *i,
    const VmaAllocationCreateInfo *p, VmaAllocationCreateInfo *placement, RcAdmissionVector *peak) {
    assert(r->required && !r->peak_ticket && r->producer == RC_BUDGET_OTHER);
    assert(i->size == 8192 || i->size == 5120);
    assert(p->flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    *placement = *p;
    placement->memoryTypeBits = i->size == 8192 ? 4 : 8;
    *peak = (RcAdmissionVector){0};
    peak->amount[i->size == 8192 ? RC_BUDGET_RAM : RC_BUDGET_COMMIT] = i->size;
    return VK_SUCCESS;
}
RcAdmissionStatus resource_core_budget_reserve(RcBudgetProducer p, int required,
    const RcAdmissionVector *peak, RcBudgetTicket *ticket) {
    assert(p == RC_BUDGET_OTHER && required && !buffers_live);
    assert(peak->amount[RC_BUDGET_RAM] == 8192 && peak->amount[RC_BUDGET_COMMIT] == 5120);
    reserve_requests++;
    if (pressure) return RC_ADMISSION_REFUSED;
    ticket->token.ledger = &ledger;
    return RC_ADMISSION_OK;
}
void resource_core_budget_readiness(int required, RcBudgetReadiness *out) {
    assert(required);
    readiness_requests++;
    *out = (RcBudgetReadiness){0};
    out->records_available = 1;
    out->usable_mask = (1u << RC_BUDGET_RAM) | (1u << RC_BUDGET_COMMIT);
    out->available.amount[RC_BUDGET_RAM] = pressure ? 4096 : 8192;
    out->available.amount[RC_BUDGET_COMMIT] = 5120;
}
RcAdmissionStatus resource_core_budget_cancel(RcBudgetTicket *ticket) {
    assert(ticket->token.ledger == &ledger);
    ticket->token.ledger = NULL;
    reservations_canceled++;
    return RC_ADMISSION_OK;
}
VkResult resource_core_vma_create_buffer(const RcVmaRequest *r, VmaAllocator a,
    const VkBufferCreateInfo *i, const VmaAllocationCreateInfo *p, VkBuffer *out,
    VmaAllocation *memory, VmaAllocationInfo *info, RcAdmissionStatus *why) {
    (void)why;
    assert(a == ctx.vma && r->required && r->producer == RC_BUDGET_OTHER && r->peak_ticket->token.ledger == &ledger);
    assert(p->memoryTypeBits == (i->size == 8192 ? 4u : 8u));
    assert(i->size == 8192 ? !!info && !buffers_live : !info && buffers_live == 1);
    buffer_requests++;
    if (i->size == 5120 && fail_child) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (info) info->pMappedData = missing_map ? NULL : &mapped_byte;
    *out = HANDLE(VkBuffer, 30 + buffers_live);
    *memory = HANDLE(VmaAllocation, 40 + buffers_live);
    buffers_live++;
    return VK_SUCCESS;
}
void resource_core_vma_destroy_buffer(VmaAllocator a, VkBuffer buffer, VmaAllocation allocation) {
    assert(a == ctx.vma && buffers_live && (D.mv_ok != 1 || idle));
    buffers_live--;
    assert(buffer == HANDLE(VkBuffer, 30 + buffers_live) && allocation == HANDLE(VmaAllocation, 40 + buffers_live));
    buffer_destroyed++;
}
#define NATIVE(create, destroy, object, info) \
    VkResult create(VkDevice dev, const info *i, const VkAllocationCallbacks *cb, object *out) { \
        (void)cb; assert(dev == ctx.device && i && buffers_live == 2); \
        native_created++; *out = HANDLE(object, native_created); return VK_SUCCESS; \
    } \
    void destroy(VkDevice dev, object handle, const VkAllocationCallbacks *cb) { \
        (void)cb; assert(dev == ctx.device && handle && (D.mv_ok != 1 || idle)); native_destroyed++; \
    }
NATIVE(vkCreateDescriptorSetLayout, vkDestroyDescriptorSetLayout, VkDescriptorSetLayout, VkDescriptorSetLayoutCreateInfo)
NATIVE(vkCreatePipelineLayout, vkDestroyPipelineLayout, VkPipelineLayout, VkPipelineLayoutCreateInfo)
NATIVE(vkCreateSampler, vkDestroySampler, VkSampler, VkSamplerCreateInfo)
NATIVE(vkCreateShaderModule, vkDestroyShaderModule, VkShaderModule, VkShaderModuleCreateInfo)
VkResult vkCreateComputePipelines(VkDevice dev, VkPipelineCache cache, uint32_t count,
    const VkComputePipelineCreateInfo *i, const VkAllocationCallbacks *cb, VkPipeline *out) {
    (void)cache; (void)cb;
    assert(dev == ctx.device && count == 1 && i->stage.module && i->layout);
    if (fail_pipeline) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    native_created++;
    *out = HANDLE(VkPipeline, native_created);
    return VK_SUCCESS;
}
void vkDestroyPipeline(VkDevice dev, VkPipeline pipeline, const VkAllocationCallbacks *cb) {
    (void)cb; assert(dev == ctx.device && pipeline && idle); native_destroyed++;
}
'''


TEST = r'''
static void present(void) {
    struct PresentSlot *slot = &g_present_slots[0];
    *slot = (struct PresentSlot){.image = HANDLE(VkImage, 10), .view = HANDLE(VkImageView, 12),
        .memory = HANDLE(VmaAllocation, 11), .extent = g_sc_ext, .format = g_sc_fmt,
        .copy = HANDLE(VkCommandBuffer, 14), .copied = HANDLE(VkFence, 15),
        .retired = HANDLE(VkSemaphore, 16), .retire_value = 17};
    assert(present_slot_image(0) && !image_requests);
    const struct PresentSlot old = *slot;
    g_sc_ext.width = 2560;
    retired = pressure = 1;
    assert(!present_slot_image(0) && !memcmp(slot, &old, sizeof old));
    assert(image_requests == 1 && !image_destroyed && !view_destroyed);
    pressure = 0;
    fail_view = 1;
    assert(!present_slot_image(0) && !memcmp(slot, &old, sizeof old));
    assert(image_requests == 2 && image_destroyed == 1 && !view_destroyed && !new_image_live);
    fail_view = 0;
    assert(present_slot_image(0));
    assert(image_requests == 3 && image_destroyed == 2 && view_destroyed == 1 && !old_image_live);
    assert(slot->extent.width == 2560 && slot->image == HANDLE(VkImage, 20) && slot->view == HANDLE(VkImageView, 22));
    assert(slot->copy == old.copy && slot->copied == old.copied &&
           slot->retired == old.retired && slot->retire_value == old.retire_value);
    assert(present_slot_image(0) && image_requests == 3);
    present_slot_drop(slot);
    assert(image_destroyed == 3 && view_destroyed == 2 && !new_image_live && !slot->memory);
}

int main(int argc, char **argv) {
    assert(argc == 2);
    const char mode = argv[1][0];
    if (mode == 'p') {
        present();
    } else {
        pressure = 1;
        assert(!mv_init() && D.mv_ok == -1 && reserve_requests == 1 && !buffer_requests);
        assert(!mv_init() && readiness_requests == 1 && reserve_requests == 1);
        stopped = 1;
        assert(!mv_init() && readiness_requests == 1);
        stopped = pressure = 0;
        fail_child = mode == 'c';
        missing_map = mode == 'm';
        fail_pipeline = mode == 'f';
        if (fail_child || missing_map || fail_pipeline) {
            assert(!mv_init() && D.mv_ok == 0 && !D.camb && !D.mvm && !D.mvm_map);
            assert(!buffers_live && native_created == native_destroyed);
            assert(buffer_destroyed == (fail_pipeline ? 2u : 1u));
        } else {
            assert(mv_init() && D.mv_ok == 1 && D.mvm_map == &mapped_byte && buffers_live == 2);
            assert(mv_init() && reserve_requests == 2 && buffer_requests == 2 && !buffer_destroyed);
            assert(native_created == 5 && native_destroyed == 1);
        }
        assert(reserve_requests == 2 && reservations_canceled == 1);
        idle = 1;
        vk_dlaa_motion_shutdown();
        vk_dlaa_motion_shutdown();
        assert(D.mv_ok == -1 && !buffers_live && native_created == native_destroyed);
    }
    printf("motion/present allocation (%c) passed\n", mode);
}
'''


def main():
    present = (RUNTIME / "gfx_present.c").read_text()
    motion = (RUNTIME / "vk_dlaa.c").read_text()
    device = (RUNTIME / "vk_device.c").read_text()
    prep = function(present, "present_image")
    assert "!present_slot_ready(slot) || !present_slot_image(slot)" in prep
    release = function(present, "gfx_window_release_vk")
    assert release.index("present_retire_generation(release_sdk)") < release.index("present_slot_drop(")
    shutdown = function(device, "destroy_all")
    assert shutdown.index("if (idle == VK_SUCCESS) vk_dlaa_motion_shutdown();") < shutdown.index("vmaDestroyAllocator(")
    module = "\n".join((
        section(present, "static struct PresentSlot {", "static VkSemaphore g_snapshot_ready;"),
        function(present, "present_slot_drop"), function(present, "present_slot_image"),
        section(motion, "#define CAM_BYTES", "// The reprojection from this frame"),
    ))
    cc = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="motion-present-") as temp:
        src = Path(temp) / "check.c"
        src.write_text("\n".join((PRELUDE, module, TEST)))
        exe = src.with_suffix(".exe")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", "-I" + str(ROOT / "runtime"),
                        "-I" + str(cc.parent.parent / "include/vma"), str(src), "-o", str(exe)], env=env, check=True)
        for mode in ("p", "s", "c", "m", "f"):
            subprocess.run([str(exe), mode], env=env, check=True)


if __name__ == "__main__":
    main()

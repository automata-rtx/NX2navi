"""Exercise the actual bindless ring bootstrap and teardown with controlled native allocation."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runtime/vk_exec_draw.c"


def between(s, first, last):
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

#include "vk_exec.h"
#include "texture_resource.h"
#include "resource_core_vma.h"
#include "platform/platform.h"

#undef vkCreateDescriptorSetLayout
#undef vkDestroyDescriptorSetLayout
#undef vkGetPhysicalDeviceFeatures2
#undef vkGetPhysicalDeviceProperties2
#undef vkGetDescriptorSetLayoutSupport
#undef vkCreateDescriptorPool
#undef vkDestroyDescriptorPool
#undef vkAllocateDescriptorSets
#undef vkUpdateDescriptorSets

static VkCtx context = {.device = (VkDevice)(uintptr_t)2, .vma = (VmaAllocator)(uintptr_t)4};
static unsigned mutex_depth;
static unsigned refused, optional_refused, fail_layout, fail_set_count, fail_child, mapped_missing, native_lost;
static unsigned layout_calls, pools_created, pools_destroyed, layouts_destroyed;
static unsigned reserve_required, reserve_optional, canceled, materialized, retired, destroyed;
static unsigned buffers_created, buffers_destroyed, writes, cleared, safe_destroy;
static unsigned active_pool, active_layout, live_buffer;
static int marks_requested, declined, sampler_capacity, layout_capacity, typed_layout;
static uint32_t readback_memory;
static RcAdmissionVector required_peak, actual_peak;
static uint64_t next_handle = 40;
static PlatMutex pool_lock_;

VkCtx *vk_ctx(void) { return &context; }
volatile int32_t vk_lost;
int vk_pool_marks_enabled(void) { return marks_requested && !declined; }
void vk_pool_marks_decline(void) { assert(!active_layout && !writes); declined = 1; }
void plat_mutex_lock(PlatMutex *p) { assert(p == &pool_lock_ && !mutex_depth++); }
void plat_mutex_unlock(PlatMutex *p) { assert(p == &pool_lock_ && mutex_depth-- == 1); }
void rt_log(const char *format, ...) { assert(format); }
_Noreturn void rt_fatal(const char *format, ...) { fputs(format, stderr); abort(); }
void vk_device_lost_check(VkResult status, const char *what) { (void)status; (void)what; abort(); }
void vk_exec_texture_publications_clear(void) { assert(safe_destroy); cleared++; }

RcAdmissionStatus resource_core_budget_reserve(RcBudgetProducer producer, int required,
    const RcAdmissionVector *peak, RcBudgetTicket *ticket) {
    assert(producer == RC_BUDGET_BUFFER && !ticket->token.ledger);
    if (required) {
        reserve_required++;
        required_peak = *peak;
        if (refused) return RC_ADMISSION_REFUSED;
    } else {
        reserve_optional++;
        assert(peak->amount[RC_BUDGET_GPU_FIRST] && peak->amount[RC_BUDGET_RAM]);
        if (optional_refused) return RC_ADMISSION_REFUSED;
    }
    ticket->token.ledger = (void *)(uintptr_t)1;
    ticket->token.slot = required ? 1 : 2;
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_cancel(RcBudgetTicket *ticket) {
    assert(ticket->token.ledger);
    canceled++;
    *ticket = (RcBudgetTicket){0};
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_materialize(RcBudgetTicket *ticket, const RcAdmissionVector *actual) {
    assert(ticket->token.ledger && ticket->token.slot == 1 && active_pool && active_layout);
    assert(actual->amount[RC_BUDGET_RAM] <= required_peak.amount[RC_BUDGET_RAM]);
    actual_peak = *actual;
    materialized++;
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_retire(RcBudgetTicket *ticket) {
    assert(ticket->token.ledger && materialized && safe_destroy && active_pool);
    retired++;
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_destroyed(RcBudgetTicket *ticket) {
    assert(ticket->token.ledger && !active_pool && !active_layout && !live_buffer && retired == ++destroyed);
    *ticket = (RcBudgetTicket){0};
    return RC_ADMISSION_OK;
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(VkPhysicalDevice device, VkPhysicalDeviceFeatures2 *features) {
    (void)device;
    VkPhysicalDeviceVulkan12Features *physical = features->pNext;
    physical->descriptorBindingPartiallyBound = VK_TRUE;
    physical->descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    physical->descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties2(VkPhysicalDevice device, VkPhysicalDeviceProperties2 *props) {
    (void)device;
    VkPhysicalDeviceVulkan12Properties *indexing = props->pNext;
    indexing->maxPerStageDescriptorUpdateAfterBindSamplers = sampler_capacity ? 1000000 : VK_POOL_SAMP;
    indexing->maxDescriptorSetUpdateAfterBindSamplers = indexing->maxPerStageDescriptorUpdateAfterBindSamplers;
}
VKAPI_ATTR void VKAPI_CALL vkGetDescriptorSetLayoutSupport(VkDevice device, const VkDescriptorSetLayoutCreateInfo *info,
    VkDescriptorSetLayoutSupport *support) {
    assert(device == context.device && info->pBindings[info->bindingCount - 1].binding == VK_POOL_COMPARISON_BINDING);
    support->supported = layout_capacity;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorSetLayout(VkDevice device, const VkDescriptorSetLayoutCreateInfo *info,
    const VkAllocationCallbacks *allocator, VkDescriptorSetLayout *out) {
    assert(device == context.device && !allocator && !active_layout && !active_pool);
    typed_layout = sampler_capacity && layout_capacity;
    assert(info->bindingCount == VK_POOL_KINDS + 1u + (unsigned)typed_layout + (uint32_t)vk_pool_marks_enabled());
    assert(info->pBindings[VK_POOL_KINDS].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER);
    if (vk_pool_marks_enabled()) assert(live_buffer == 5);
    layout_calls++;
    if (fail_layout) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    *out = (VkDescriptorSetLayout)(uintptr_t)++next_handle;
    active_layout = 1;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout layout,
    const VkAllocationCallbacks *allocator) {
    assert(device == context.device && layout && !allocator && active_layout && !active_pool);
    active_layout = 0;
    layouts_destroyed++;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorPool(VkDevice device, const VkDescriptorPoolCreateInfo *info,
    const VkAllocationCallbacks *allocator, VkDescriptorPool *out) {
    assert(device == context.device && !allocator && active_layout && !active_pool);
    assert(info->pPoolSizes[0].descriptorCount == VK_POOL_KINDS * VK_POOL_TEX * info->maxSets &&
           info->pPoolSizes[1].descriptorCount == (unsigned)(1 + typed_layout) * VK_POOL_SAMP * info->maxSets);
    assert(info->poolSizeCount == (vk_pool_marks_enabled() ? 3u : 2u));
    active_pool = info->maxSets;
    pools_created++;
    *out = (VkDescriptorPool)(uintptr_t)++next_handle;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorPool(VkDevice device, VkDescriptorPool pool,
    const VkAllocationCallbacks *allocator) {
    assert(device == context.device && pool && !allocator && active_pool);
    assert(!materialized || safe_destroy);
    active_pool = 0;
    pools_destroyed++;
}
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo *info,
    VkDescriptorSet *sets) {
    assert(device == context.device && info->descriptorPool && info->descriptorSetCount == active_pool);
    if (fail_set_count == active_pool) return VK_ERROR_OUT_OF_POOL_MEMORY;
    for (uint32_t i = 0; i < active_pool; i++) sets[i] = (VkDescriptorSet)(uintptr_t)++next_handle;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSets(VkDevice device, uint32_t count, const VkWriteDescriptorSet *written,
    uint32_t copies, const VkCopyDescriptorSet *from) {
    assert(device == context.device && count == 1 && !copies && !from && active_pool && materialized && live_buffer == 5);
    assert(written->dstSet && written->dstBinding == VK_POOL_MARK_BINDING && written->pBufferInfo->buffer);
    writes++;
}
VkResult resource_core_vma_buffer_peak(const RcVmaRequest *req, const VkBufferCreateInfo *buffer,
    const VmaAllocationCreateInfo *input, VmaAllocationCreateInfo *placed, RcAdmissionVector *cost) {
    assert(!req->required && req->producer == RC_BUDGET_BUFFER);
    assert(buffer->size == VK_POOL_TEX * sizeof(uint32_t));
    assert(input->flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    *placed = *input;
    *cost = (RcAdmissionVector){.amount = {[RC_BUDGET_GPU_FIRST] = buffer->size,
                                            [RC_BUDGET_RAM] = buffer->size, [RC_BUDGET_COMMIT] = buffer->size}};
    return VK_SUCCESS;
}
VkResult resource_core_vma_create_buffer(const RcVmaRequest *req, VmaAllocator allocator,
    const VkBufferCreateInfo *buffer, const VmaAllocationCreateInfo *info, VkBuffer *out,
    VmaAllocation *allocation, VmaAllocationInfo *instance, RcAdmissionStatus *why) {
    assert(!req->required && req->peak_ticket->token.ledger && req->peak_ticket->token.slot == 2);
    assert(allocator == context.vma && !why && !active_layout && reserve_required);
    assert(buffer && info && !*out && !*allocation);
    unsigned attempt = buffers_created + 1;
    if (attempt == fail_child) return native_lost ? VK_ERROR_DEVICE_LOST : VK_ERROR_OUT_OF_DEVICE_MEMORY;
    buffers_created++;
    live_buffer++;
    *out = (VkBuffer)(uintptr_t)++next_handle;
    *allocation = (VmaAllocation)(uintptr_t)++next_handle;
    if (instance) instance->pMappedData = attempt == mapped_missing ? NULL : &readback_memory;
    return VK_SUCCESS;
}
void resource_core_vma_destroy_buffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation) {
    assert(allocator == context.vma && buffer && allocation && live_buffer && !active_pool);
    live_buffer--;
    buffers_destroyed++;
}
'''

TEST = r'''
int main(int argc, char **argv) {
    assert(argc == 2);
    char mode = argv[1][0];
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    marks_requested = mode != 'u' && mode != 'p';
    sampler_capacity = mode == 'p' || mode == 'q';
    layout_capacity = mode == 'p';
    refused = mode == 'r';
    optional_refused = mode == 'o';
    fail_layout = mode == 'l';
    fail_set_count = mode == 's' ? POOL_SETS : 0;
    fail_child = mode == 'd' ? 4 : mode == 'x' ? 2 : 0;
    native_lost = mode == 'x';
    mapped_missing = mode == 'm' ? 3 : 0;
    VkResult result = pool_layout_make(&layout);

    if (mode == 'r') {
        assert(result == VK_NOT_READY && reserve_required == POOL_SETS && !reserve_optional && !layout_calls);
    } else if (mode == 'l') {
        assert(result == VK_ERROR_OUT_OF_DEVICE_MEMORY && layout_calls == 1 && !layout && !active_pool);
        assert(buffers_created == 5 && buffers_destroyed == 5 && canceled == 2 && !live_buffer && !writes);
    } else if (mode == 'x') {
        assert(result == VK_ERROR_DEVICE_LOST && reserve_required == 1 && !active_pool && !layout_calls);
        assert(buffers_created == 1 && buffers_destroyed == 1 && canceled == 2 && !declined);
    } else {
        assert(result == VK_SUCCESS && layout && PL.tables && PL.log == PL.tables->log && PL.nsets == (int)active_pool);
        PL.layout = layout;
        int marked = vk_pool_marks_enabled();
        assert(PL.nsets == (mode == 's' ? POOL_SETS - 1 : POOL_SETS));
        assert(materialized == 1 && active_layout && writes == (marked ? active_pool : 0));
        assert(actual_peak.amount[RC_BUDGET_RAM] == pool_bootstrap_peak(PL.nsets, marked).amount[RC_BUDGET_RAM]);
        if (mode == 's') assert(pools_created == 2 && pools_destroyed == 1 && layouts_destroyed == 1 && reserve_optional == 1);
        if (mode == 'o') assert(declined && !buffers_created);
        if (mode == 'd' || mode == 'm') assert(declined && buffers_created == 3 && buffers_destroyed == 3);
        unsigned reserved = reserve_required;
        assert(vk_pool_set_layout() == layout && reserve_required == reserved && !mutex_depth);
        assert(vk_pool_split_samplers() == (mode == 'p'));
        assert(!retired && !destroyed && live_buffer == (marked ? 5u : 0u));
        safe_destroy = 1;
        pool_nw = 1; // Buffered descriptor update is never submitted during stopped teardown.
        vk_exec_draw_bindless_shutdown();
        assert(retired == 1 && destroyed == 1 && cleared == 1 && !pool_nw && !PL.layout && !PL.nsets && !PL.tables);
    }
    assert(!live_buffer && !active_pool && !active_layout);
    printf("bindless pool bootstrap (%c) passed\n", mode);
}
'''


def main():
    text = SOURCE.read_text()
    module = "\n".join((
        "#define MAX_DESC 1024",
        function(text, "descriptor_pool_opaque_bytes"),
        between(text, "#define POOL_SETS VK_POOL_SETS", "static int pool_native_slot("),
        between(text, "static VkResult pool_marks_allocate(VkCtx *);", "void vk_exec_pool_marks("),
        function(text, "vk_pool_set_layout"),
        function(text, "vk_pool_split_samplers"),
        "static uint32_t pool_nw;\n",
        function(text, "vk_exec_draw_bindless_shutdown"),
    ))
    module = module.replace(function(text, "pool_note_marks_owner"), "")
    slot = between(text, "// Per texture slot, what pool_fill", "enum { POOL_MISSING_HISTORY")
    cc = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="bindless-bootstrap-") as temp:
        source = Path(temp) / "check.c"
        source.write_text("\n".join((PRELUDE, slot, module, TEST)))
        exe = source.with_suffix(".exe")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", "-I" + str(ROOT / "runtime"),
                        "-I" + str(cc.parent.parent / "include/vma"), str(source), "-o", str(exe)], env=env, check=True)
        for mode in ("r", "o", "u", "d", "m", "s", "l", "x", "n", "p", "q"):
            subprocess.run([str(exe), mode], env=env, check=True)


if __name__ == "__main__":
    main()

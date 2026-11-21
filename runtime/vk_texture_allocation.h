// Texture allocation's bounded pressure bridge. Included after Vulkan, shared admission and runtime declarations.
#pragma once

static int texture_allocation_capacity_failure(VkResult result, RcAdmissionStatus admission) {
    return result == VK_NOT_READY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY ||
        admission == RC_ADMISSION_UNSUPPORTED || admission == RC_ADMISSION_OVERFLOW ||
        admission == RC_ADMISSION_GENERATION_EXHAUSTED;
}

static VkResult texture_image_allocate(VkCtx *c, const VkImageCreateInfo *image,
                                        const VmaAllocationCreateInfo *allocation, VkImage *out,
                                        VmaAllocation *memory, int required, const char *purpose, int *capacity_failure) {
    RcVmaRequest request = {.producer = RC_BUDGET_TEXTURE, .required = required};
    RcAdmissionStatus admission;
    if (capacity_failure) *capacity_failure = 0;
    VkResult result = resource_core_vma_create_image(&request, c->vma, image, allocation, out, memory, NULL, &admission);
    if (result == VK_SUCCESS || !texture_allocation_capacity_failure(result, admission)) return result;
    if (capacity_failure) *capacity_failure = 1;
    if (!required) return result;

    // Give back only immediately unused buffers and allocator anchors. No queue wait or future texture retirement
    // is treated as relief. The shared service admits the retry from compatible reuse or newly observed capacity.
    uint64_t released = vk_mem_give_back();
    resource_core_budget_refresh();
    result = resource_core_vma_create_image(&request, c->vma, image, allocation, out, memory, NULL, &admission);
    if (result == VK_SUCCESS) {
        if (capacity_failure) *capacity_failure = 0;
        return result;
    }
    RcBudgetSnapshot sample;
    resource_core_budget_snapshot(&sample);
    rt_fatal("vk: required %s image %ux%ux%u format %d cannot acquire capacity after one reclaim/retry "
             "(result %d, admission %d, released %.1f MiB, available RAM %.1f MiB, commit %.1f MiB, "
             "GPU heaps %.1f/%.1f/%.1f/%.1f/%.1f MiB, telemetry %#x); no substitute texture was published",
             purpose, image->extent.width, image->extent.height, image->extent.depth, (int)image->format,
             (int)result, (int)admission, released / 1048576.0,
             sample.available.amount[RC_BUDGET_RAM] / 1048576.0,
             sample.available.amount[RC_BUDGET_COMMIT] / 1048576.0,
             sample.available.amount[RC_BUDGET_GPU_FIRST] / 1048576.0,
             sample.available.amount[RC_BUDGET_GPU_FIRST + 1] / 1048576.0,
             sample.available.amount[RC_BUDGET_GPU_FIRST + 2] / 1048576.0,
             sample.available.amount[RC_BUDGET_GPU_FIRST + 3] / 1048576.0,
             sample.available.amount[RC_BUDGET_GPU_FIRST + 4] / 1048576.0, sample.sample.known_mask);
    return result;
}

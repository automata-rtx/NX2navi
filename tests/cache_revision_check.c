// Cache-hit creation must not schedule a full rewrite; a save acknowledges only
// the changes captured before its lock was released, and only after a durable write.
#include <assert.h>
#include <stdio.h>
#include "vk_cache_revision.h"

int main(void) {
    VkPipelineCreationFeedback fb = {0};
    VkPipelineCreationFeedbackCreateInfo feedback = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO,
        .pPipelineCreationFeedback = &fb};
    VkComputePipelineCreateInfo compute = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .pNext = &feedback};
    VkGraphicsPipelineCreateInfo graphics = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &feedback};
    const VkPipelineCreationFeedbackFlags hit = VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT |
                                                VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT;
    fb.flags = hit;
    assert(!vk_cache_create_changed(&compute));
    assert(!vk_cache_create_changed(&graphics));
    fb.flags = VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT;
    assert(vk_cache_create_changed(&compute));
    fb.flags = VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT;
    assert(vk_cache_create_changed(&compute)); // hit without VALID is not evidence
    fb.flags = 0;
    assert(vk_cache_create_changed(&compute));
    compute.pNext = NULL;
    assert(vk_cache_create_changed(&compute)); // feedback disabled
    VkPipelineRenderingCreateInfo middle = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .pNext = &feedback};
    graphics.pNext = &middle;
    fb.flags = hit;
    assert(!vk_cache_create_changed(&graphics));
    feedback.pPipelineCreationFeedback = NULL;
    assert(vk_cache_create_changed(&graphics));

    VkCacheRevision revision = {0};
    assert(!vk_cache_pending(&revision));
    vk_cache_changed(&revision);
    uint64_t snapshot = vk_cache_revision(&revision);
    vk_cache_saved(&revision, snapshot, 0);
    assert(vk_cache_pending(&revision)); // failed write remains eligible for retry
    vk_cache_changed(&revision);         // a build completes during the file flush
    vk_cache_saved(&revision, snapshot, 1);
    assert(vk_cache_pending(&revision)); // must not acknowledge that newer build
    vk_cache_saved(&revision, vk_cache_revision(&revision), 1);
    assert(!vk_cache_pending(&revision));
    puts("Cache revision checks passed");
    return 0;
}

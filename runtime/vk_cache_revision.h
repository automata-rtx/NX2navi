#ifndef VK_CACHE_REVISION_H
#define VK_CACHE_REVISION_H

#include <stdint.h>
#include <vulkan/vulkan.h>

// The cache snapshot and its durable file have separate revisions. A build may
// finish after the snapshot lock is released but before the file reaches disk.
typedef struct { uint64_t changed, saved; } VkCacheRevision;

static inline void vk_cache_changed(VkCacheRevision *r) {
    __atomic_add_fetch(&r->changed, 1, __ATOMIC_RELEASE);
}
static inline uint64_t vk_cache_revision(const VkCacheRevision *r) {
    return __atomic_load_n(&r->changed, __ATOMIC_ACQUIRE);
}
static inline int vk_cache_pending(const VkCacheRevision *r) {
    return vk_cache_revision(r) != __atomic_load_n(&r->saved, __ATOMIC_ACQUIRE);
}
// Saves are serialized by vk_pipeline.c's save_lock. A failed write acknowledges nothing.
static inline void vk_cache_saved(VkCacheRevision *r, uint64_t snapshot, int success) {
    if (success) __atomic_store_n(&r->saved, snapshot, __ATOMIC_RELEASE);
}

// Only valid driver feedback can establish that a successful create reused the
// application's cache. Missing feedback remains dirty, including feedback disabled by configuration.
static inline int vk_cache_create_changed(const void *create_info) {
    const VkBaseInStructure *p = create_info;
    for (; p; p = p->pNext) {
        if (p->sType != VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO) continue;
        const VkPipelineCreationFeedbackCreateInfo *f = (const void *)p;
        const VkPipelineCreationFeedback *feedback = f->pPipelineCreationFeedback;
        const VkPipelineCreationFeedbackFlags hit = VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT |
                                                    VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT;
        return !feedback || (feedback->flags & hit) != hit;
    }
    return 1;
}
#endif

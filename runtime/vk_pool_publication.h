// One completed descriptor-pool view for a parallel draw group. The owner resets it only after all consumers finish.
// The payload is immutable after release publication; descriptor-ring lifetime remains tied to the submission fence.
#pragma once
#include <stdint.h>
#include <string.h>

typedef struct {
    VkDescriptorSet set;
    uint64_t fence, generation;
    uint32_t kinds, pending, ready;
} VkExecPoolView;

static inline void vk_pool_view_reset(VkExecPoolView *view) {
    memset(view, 0, sizeof *view);
}

static inline int vk_pool_view_get(const VkExecPoolView *view, uint64_t fence, uint32_t kinds, uint64_t generation,
                                   VkDescriptorSet *set) {
    if (!view || !__atomic_load_n(&view->ready, __ATOMIC_ACQUIRE)) return 0;
    if (view->fence != fence || (view->kinds & kinds) != kinds || !generation ||
        view->generation != generation || !view->set) return 0;
    *set = view->set;
    return 1;
}

// Called under the pool mutex, or before workers can see the group. Refresh, flush and catch-up precede this call.
static inline int vk_pool_view_publish(VkExecPoolView *view, VkDescriptorSet set, uint64_t fence,
                                       uint64_t generation, uint32_t kinds, uint32_t pending) {
    if (!view || __atomic_load_n(&view->ready, __ATOMIC_ACQUIRE) || !set || !generation) return 0;
    view->set = set;
    view->fence = fence;
    view->generation = generation;
    view->kinds = kinds;
    view->pending = pending;   // a complete refresh may intentionally retain budget-deferred stand-ins
    __atomic_store_n(&view->ready, 1, __ATOMIC_RELEASE);
    return 1;
}

// Bounded deferred-slot queue. Slots remain pending until refilled, even when the dirty log has advanced past them.
// The caller holds the pool and texture locks. Rotating before an attempt keeps retries fair under a creation budget.
#pragma once
#include <stdint.h>
#include <string.h>

#ifndef VK_POOL_RETRY_CAP
#define VK_POOL_RETRY_CAP VK_POOL_TEX
#endif

typedef struct {
    uint32_t next[VK_POOL_RETRY_CAP], prev[VK_POOL_RETRY_CAP], attempted[VK_POOL_RETRY_CAP];
    uint32_t head, tail, count, epoch;   // links store slot + 1; zero means no slot
} VkPoolRetry;

static inline int vk_pool_retry_has(const VkPoolRetry *q, uint32_t id) {
    return q->head == id + 1 || q->next[id] || q->prev[id];
}

static inline void vk_pool_retry_add(VkPoolRetry *q, uint32_t id) {
    if (vk_pool_retry_has(q, id)) return;
    q->prev[id] = q->tail;
    if (q->tail) q->next[q->tail - 1] = id + 1;
    else q->head = id + 1;
    q->tail = id + 1;
    q->count++;
}

static inline void vk_pool_retry_remove(VkPoolRetry *q, uint32_t id) {
    if (!vk_pool_retry_has(q, id)) return;
    uint32_t next = q->next[id], prev = q->prev[id];
    if (prev) q->next[prev - 1] = next;
    else q->head = next;
    if (next) q->prev[next - 1] = prev;
    else q->tail = prev;
    q->next[id] = q->prev[id] = 0;
    q->count--;
}

static inline uint32_t vk_pool_retry_rotate(VkPoolRetry *q) {
    uint32_t id = q->head - 1;   // caller checks count before taking a slot
    if (q->count > 1) {
        vk_pool_retry_remove(q, id);
        vk_pool_retry_add(q, id);
    }
    return id;
}

static inline void vk_pool_retry_begin(VkPoolRetry *q) {
    if (q->count) vk_pool_retry_rotate(q);   // rotate the first opportunity across syncs, including all-failed rounds
    if (!++q->epoch) {
        memset(q->attempted, 0, sizeof q->attempted);
        q->epoch = 1;
    }
}

static inline int vk_pool_retry_attempt(VkPoolRetry *q, uint32_t id) {
    if (vk_pool_retry_has(q, id) && q->attempted[id] == q->epoch) return 0;
    q->attempted[id] = q->epoch;
    return 1;
}

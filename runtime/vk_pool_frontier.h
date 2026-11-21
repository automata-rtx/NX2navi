#ifndef VK_POOL_FRONTIER_H
#define VK_POOL_FRONTIER_H
#include <stdint.h>
#include <string.h>

// Indexed deadlines for the shared texture table. Zero is an unresolved member; finite deadlines are
// the first logical read that needs another observation. Members valid without a deadline are absent.
// The caller rebuilds after external invalidation or nonmonotonic scene traversal, under the pool lock.
typedef struct {
    uint32_t ids[VK_POOL_TEX], at[VK_POOL_TEX], count;
    uint64_t deadline[VK_POOL_TEX];
} VkPoolFrontier;
static inline void vk_pool_frontier_reset(VkPoolFrontier *f) { memset(f, 0, sizeof *f); }
static inline int vk_pool_frontier_less(const VkPoolFrontier *f, uint32_t a, uint32_t b) {
    return f->deadline[a] < f->deadline[b] || (f->deadline[a] == f->deadline[b] && a < b);
}
static inline void vk_pool_frontier_swap(VkPoolFrontier *f, uint32_t a, uint32_t b) {
    uint32_t id=f->ids[a];f->ids[a]=f->ids[b];f->ids[b]=id;
    f->at[f->ids[a]]=a+1;f->at[f->ids[b]]=b+1;
}
static inline void vk_pool_frontier_adjust(VkPoolFrontier *f, uint32_t at) {
    while(at && vk_pool_frontier_less(f,f->ids[at],f->ids[(at-1)/2])) {
        uint32_t parent=(at-1)/2;vk_pool_frontier_swap(f,at,parent);at=parent;
    }
    for(;;) {
        uint32_t child=at*2+1;if(child>=f->count)break;
        if(child+1<f->count && vk_pool_frontier_less(f,f->ids[child+1],f->ids[child]))child++;
        if(!vk_pool_frontier_less(f,f->ids[child],f->ids[at]))break;
        vk_pool_frontier_swap(f,at,child);at=child;
    }
}
static inline void vk_pool_frontier_remove(VkPoolFrontier *f, uint32_t id) {
    if(!f->at[id])return;
    uint32_t at=f->at[id]-1;f->at[id]=0;f->count--;
    if(at==f->count)return;
    f->ids[at]=f->ids[f->count];f->at[f->ids[at]]=at+1;vk_pool_frontier_adjust(f,at);
}
static inline void vk_pool_frontier_set(VkPoolFrontier *f, uint32_t id, uint64_t deadline) {
    uint32_t at=f->at[id];f->deadline[id]=deadline;
    if(!at){at=++f->count;f->ids[at-1]=id;f->at[id]=at;}
    vk_pool_frontier_adjust(f,at-1);
}
#endif

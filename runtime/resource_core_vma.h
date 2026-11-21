// Allocation admission and unique capacity ownership shared by image, buffer and staging producers.
#pragma once
#include "vk.h"
#include "resource_core_budget.h"

typedef struct RcVmaDependency {
    RcBudgetReadiness before;
    RcAdmissionVector peak, reuse;
    uint64_t released;
    uint32_t memory_type;
    // When owner records ran out, owner_full names the number the retry must see free together.
    int selected, can_reuse, owner_full, reuse_unattempted;
} RcVmaDependency;

typedef struct {
    RcBudgetProducer producer;
    int required;
    // Required only when VmaAllocationCreateInfo.pool is non-null. These describe that existing pool exactly.
    uint32_t pool_memory_type;
    VkDeviceSize pool_block_size;
    // Optional host companion admitted atomically with GPU allocation growth. On success the caller owns a
    // RESERVED host ticket: materialize after allocating host storage, or cancel if it is no longer needed.
    // Both fields are required together; the output ticket must initially be empty. Pool creation excludes it.
    uint64_t host_bytes;
    RcBudgetTicket *host_ticket;
    // Optional exact incremental request attempted by this call; valid on admission refusal, never a pointer
    // into shared last-refusal statistics. Used by the preparation owner to request domain-specific giveback.
    RcAdmissionVector *refused_peak;
    // Optional progress evidence for the selected default-pool placement. Capture before attempting work;
    // concurrent release followed by refusal must still be observable by the request owner.
    RcVmaDependency *dependency;
    // The operation owner already reserved this exact placement peak with its other preparation inputs.
    // Partition that guarantee before allocating. Existing compatible backing still owns a single charge.
    RcBudgetTicket *peak_ticket;
} RcVmaRequest;

int resource_core_vma_dependency_ready(const RcVmaDependency *dependency, const RcBudgetReadiness *now);
// A failed compound attempt may return its own native placements before parking for actual giveback.
void resource_core_vma_dependency_rebase(RcVmaDependency *dependency, int required);
// A pending native owner can nominate a compatible reuse or owner-slot prerequisite. Actual destruction still
// has to produce new request-specific progress before another allocation call.
int resource_core_vma_dependency_retirement(const RcVmaDependency *, VmaAllocation);

// Query a placement and its worst additional native growth without creating an image or buffer. Pass the
// returned allocation info to the create call so its memory type cannot change under the reserved peak.
// With dependency, the selected default-pool reuse allowance and release evidence also describe the
// non-growing working set. VMA still decides whether an existing slot can accommodate it.
VkResult resource_core_vma_image_peak(const RcVmaRequest *request, const VkImageCreateInfo *image,
                                     const VmaAllocationCreateInfo *allocation, VmaAllocationCreateInfo *placement,
                                     RcAdmissionVector *peak);
VkResult resource_core_vma_buffer_peak(const RcVmaRequest *request, const VkBufferCreateInfo *buffer,
                                      const VmaAllocationCreateInfo *allocation, VmaAllocationCreateInfo *placement,
                                      RcAdmissionVector *peak);

// These never wait for capacity. VK_NOT_READY means admission pressure/bookkeeping capacity; *admission
// distinguishes refusal from unsupported placement. Outputs remain null unless creation and accounting succeed.
VkResult resource_core_vma_create_image(const RcVmaRequest *request, VmaAllocator allocator,
                                        const VkImageCreateInfo *image, const VmaAllocationCreateInfo *allocation,
                                        VkImage *out_image, VmaAllocation *out_allocation, VmaAllocationInfo *info,
                                        RcAdmissionStatus *admission);
VkResult resource_core_vma_create_buffer(const RcVmaRequest *request, VmaAllocator allocator,
                                         const VkBufferCreateInfo *buffer, const VmaAllocationCreateInfo *allocation,
                                         VkBuffer *out_buffer, VmaAllocation *out_allocation, VmaAllocationInfo *info,
                                         RcAdmissionStatus *admission);
// Raw requirements preserve the caller's deliberate suballocation choice. The caller handles mandatory
// dedicated allocation and binding; this path does not reinterpret an image's preferred-dedicated hint.
VkResult resource_core_vma_allocate_memory(const RcVmaRequest *request, VmaAllocator allocator,
                                          const VkMemoryRequirements *requirements,
                                          const VmaAllocationCreateInfo *allocation, VmaAllocation *out_allocation,
                                          VmaAllocationInfo *info, RcAdmissionStatus *admission);
// A fixed-block pool may preallocate backing before any suballocation exists. Charge that growth once, then
// let measured usage account for it; later allocation tickets describe occupied spans inside that backing.
VkResult resource_core_vma_create_pool(const RcVmaRequest *request, VmaAllocator allocator,
                                      const VmaPoolCreateInfo *create, VmaPool *out_pool, RcAdmissionStatus *admission);

// Mark the actual owned allocation pending once the existing lifetime owner queues its retirement. Destruction
// remains the caller's decision after CPU/GPU users complete. Untracked allocations retain their legacy path.
RcAdmissionStatus resource_core_vma_retire(VmaAllocation allocation);
// Occupied footprint for selecting reclaimable capacity, not a promise of physical memory returned on free.
RcAdmissionStatus resource_core_vma_allocation_cost(VmaAllocation allocation, RcAdmissionVector *out);
// Classifies a caller-owned native allocation for reclaim selection. Admission and settlement stay with its owner.
RcAdmissionStatus resource_core_vma_native_footprint(const VmaAllocationInfo *info, RcBudgetProducer producer,
                                                    RcAdmissionVector *out);
void resource_core_vma_destroy_image(VmaAllocator allocator, VkImage image, VmaAllocation allocation);
void resource_core_vma_destroy_buffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation);
void resource_core_vma_free_memory(VmaAllocator allocator, VmaAllocation allocation);

#ifdef RC_VMA_TEST
void resource_core_vma_test_reset(void);
#endif

#include "resource_core_vma.h"
#include "platform/platform.h"
#include "host.h"

#define OWNERS 32768u
#define BUCKETS 65536u
#define NONE UINT32_MAX
enum { BUFFER = 0, IMAGE = 1, RAW_MEMORY = 2 };

typedef struct {
    VmaAllocation allocation;
    RcBudgetTicket ticket;
    uint32_t next, memory_type;
    int default_pool;
} AllocationOwner;

static struct {
    PlatMutex lock;
    AllocationOwner owner[OWNERS];
    uint32_t bucket[BUCKETS], free_head;
    uint64_t default_released[VK_MAX_MEMORY_TYPES];
    int initialized;
} V;

static uint32_t allocation_hash(VmaAllocation allocation) {
    uint64_t value = (uint64_t)(uintptr_t)allocation;
    return (uint32_t)((value >> 4) ^ (value >> 21)) & (BUCKETS - 1);
}

static uint32_t owner_take(void) {
    plat_mutex_lock(&V.lock);
    if (!V.initialized) {
        for (uint32_t i = 0; i < BUCKETS; i++) V.bucket[i] = NONE;
        for (uint32_t i = 0; i < OWNERS; i++) V.owner[i].next = i + 1 < OWNERS ? i + 1 : NONE;
        V.free_head = 0;
        V.initialized = 1;
    }
    uint32_t slot = V.free_head;
    if (slot != NONE) V.free_head = V.owner[slot].next;
    plat_mutex_unlock(&V.lock);
    return slot;
}

static void owner_unused(uint32_t slot) {
    plat_mutex_lock(&V.lock);
    V.owner[slot] = (AllocationOwner){.next = V.free_head};
    V.free_head = slot;
    plat_mutex_unlock(&V.lock);
}

static void owner_publish(uint32_t slot, VmaAllocation allocation, RcBudgetTicket ticket,
                          uint32_t memory_type, int default_pool) {
    plat_mutex_lock(&V.lock);
    uint32_t hash = allocation_hash(allocation);
    for (uint32_t i = V.bucket[hash]; i != NONE; i = V.owner[i].next)
        if (V.owner[i].allocation == allocation) rt_fatal("resource allocation acquired capacity ownership twice");
    V.owner[slot] = (AllocationOwner){allocation, ticket, V.bucket[hash], memory_type, default_pool};
    V.bucket[hash] = slot;
    plat_mutex_unlock(&V.lock);
}

static int owner_detach(VmaAllocation allocation, AllocationOwner *owner) {
    int found = 0;
    plat_mutex_lock(&V.lock);
    if (V.initialized && allocation) {
        uint32_t *link = &V.bucket[allocation_hash(allocation)];
        while (*link != NONE) {
            uint32_t i = *link;
            if (V.owner[i].allocation == allocation) {
                *owner = V.owner[i];
                *link = V.owner[i].next;
                V.owner[i] = (AllocationOwner){.next = V.free_head};
                V.free_head = i;
                found = 1;
                break;
            }
            link = &V.owner[i].next;
        }
    }
    plat_mutex_unlock(&V.lock);
    return found;
}

int resource_core_vma_dependency_ready(const RcVmaDependency *dependency, const RcBudgetReadiness *now) {
    if (!dependency->selected) return 0;
    if (dependency->owner_full) {
        plat_mutex_lock(&V.lock);
        uint32_t slot = V.free_head;
        int remaining = dependency->owner_full;
        while (remaining && slot != NONE) {
            remaining--;
            slot = V.owner[slot].next;
        }
        plat_mutex_unlock(&V.lock);
        return !remaining;
    }
    if (dependency->can_reuse && resource_core_budget_peak_ready(&dependency->reuse, now)) {
        if (__atomic_load_n(&V.default_released[dependency->memory_type], __ATOMIC_ACQUIRE) != dependency->released)
            return 1;
        if (dependency->reuse_unattempted && !resource_core_budget_peak_ready(&dependency->reuse, &dependency->before))
            return 1;
    }
    return resource_core_budget_peak_progress(&dependency->peak, &dependency->before, now);
}

void resource_core_vma_dependency_rebase(RcVmaDependency *dependency, int required) {
    if (dependency->can_reuse) {
        dependency->released = __atomic_load_n(&V.default_released[dependency->memory_type], __ATOMIC_ACQUIRE);
        dependency->reuse_unattempted = 0;
    }
    resource_core_budget_readiness(required, &dependency->before);
}

int resource_core_vma_dependency_retirement(const RcVmaDependency *dependency, VmaAllocation allocation) {
    if (!allocation || !dependency->selected) return 0;
    plat_mutex_lock(&V.lock);
    int eligible = 0;
    if (V.initialized) {
        for (uint32_t i = V.bucket[allocation_hash(allocation)]; i != NONE; i = V.owner[i].next) {
            const AllocationOwner *owner = &V.owner[i];
            if (owner->allocation != allocation) continue;
            eligible = owner->ticket.token.state == RC_ADMISSION_PENDING &&
                (dependency->owner_full || (dependency->can_reuse && owner->default_pool &&
                                            owner->memory_type == dependency->memory_type));
            break;
        }
    }
    plat_mutex_unlock(&V.lock);
    return eligible;
}

static VkResult admission_result(RcAdmissionStatus status) {
    if (status == RC_ADMISSION_REFUSED || status == RC_ADMISSION_NO_RECORDS) return VK_NOT_READY;
    if (status == RC_ADMISSION_UNSUPPORTED) return VK_ERROR_FEATURE_NOT_PRESENT;
    return VK_ERROR_INITIALIZATION_FAILED;
}

static RcAdmissionStatus allocation_cost(const VkCtx *c, uint32_t type, VkDeviceSize bytes,
                                         RcBudgetProducer producer, RcAdmissionVector *out) {
    *out = (RcAdmissionVector){{0}};
    if (type >= c->mem_props.memoryTypeCount) return RC_ADMISSION_UNSUPPORTED;
    uint32_t heap = c->mem_props.memoryTypes[type].heapIndex;
    if (heap >= c->mem_props.memoryHeapCount) return RC_ADMISSION_UNSUPPORTED;
    RcAdmissionStatus status = resource_core_budget_gpu_cost(out, heap, bytes);
    if (status != RC_ADMISSION_OK) return status;
    int host = !(c->mem_props.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ||
        c->props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    if (host) out->amount[RC_BUDGET_RAM] = bytes;
    if (host || plat_commit_counts_vram()) out->amount[RC_BUDGET_COMMIT] = bytes;
    if (producer == RC_BUDGET_STAGE) out->amount[RC_BUDGET_STAGING] = bytes;
    return RC_ADMISSION_OK;
}

RcAdmissionStatus resource_core_vma_native_footprint(const VmaAllocationInfo *info, RcBudgetProducer producer,
                                                    RcAdmissionVector *out) {
    const VkCtx *c = vk_ctx();
    if (!c || !info || !info->size || !out || producer < RC_BUDGET_TEXTURE || producer >= RC_BUDGET_PRODUCERS)
        return RC_ADMISSION_INVALID;
    return allocation_cost(c, info->memoryType, info->size, producer, out);
}

// vk_device.c:vk_init supplies this preferred size; vk_headroom.c:block_size uses the same small-heap rule.
// A default-pool allocation can create at most one preferred block or one dedicated allocation of its required
// size. Custom pools use the supplied fixed block size. Smaller fallback blocks fit inside this bound.
static VkDeviceSize growth_bound(const VkCtx *c, uint32_t type, VkDeviceSize need,
                                 const RcVmaRequest *request, const VmaAllocationCreateInfo *allocation,
                                 int dedicated) {
    if (dedicated || (allocation->flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT)) return need;
    VkDeviceSize block;
    if (allocation->pool) block = request->pool_block_size;
    else {
        uint32_t heap = c->mem_props.memoryTypes[type].heapIndex;
        VkDeviceSize size = c->mem_props.memoryHeaps[heap].size;
        block = size <= (1ull << 30) ? size / 8 : vk_vma_block_size();
        if (block > UINT64_MAX - 31) return 0;
        block = (block + 31) & ~(VkDeviceSize)31;
    }
    return block > need ? block : need;
}

static VkResult native_create(int kind, VmaAllocator allocator, const void *create,
                              const VmaAllocationCreateInfo *allocation, void *out,
                              VmaAllocation *memory, VmaAllocationInfo *info) {
    if (kind == RAW_MEMORY) return vmaAllocateMemory(allocator, create, allocation, memory, info);
    return kind == IMAGE ? vmaCreateImage(allocator, create, allocation, out, memory, info)
                         : vmaCreateBuffer(allocator, create, allocation, out, memory, info);
}

static void native_destroy(int kind, VmaAllocator allocator, void *out, VmaAllocation memory) {
    if (kind == RAW_MEMORY) vmaFreeMemory(allocator, memory);
    else if (kind == IMAGE) {
        vmaDestroyImage(allocator, *(VkImage *)out, memory);
        *(VkImage *)out = VK_NULL_HANDLE;
    } else {
        vmaDestroyBuffer(allocator, *(VkBuffer *)out, memory);
        *(VkBuffer *)out = VK_NULL_HANDLE;
    }
}

// Admission covers the simultaneous peak, not two independently successful checks. Split before the native
// call so bookkeeping exhaustion cannot strand an allocation without a companion reservation.
static RcAdmissionStatus reserve_bundle(const RcVmaRequest *request, const RcAdmissionVector *growth,
                                        RcBudgetTicket *allocation, RcBudgetTicket *host) {
    RcAdmissionVector peak = *growth, companion = {{0}};
    companion.amount[RC_BUDGET_RAM] = companion.amount[RC_BUDGET_COMMIT] = request->host_bytes;
    for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) {
        if (companion.amount[d] > UINT64_MAX - peak.amount[d]) return RC_ADMISSION_OVERFLOW;
        peak.amount[d] += companion.amount[d];
    }
    if (request->refused_peak) *request->refused_peak = peak;
    if (request->dependency) request->dependency->peak = peak;
    RcAdmissionStatus status = request->peak_ticket
        ? resource_core_budget_split(request->peak_ticket, &peak, request->producer, allocation)
        : resource_core_budget_reserve(request->producer, request->required, &peak, allocation);
    if (status == RC_ADMISSION_OK && request->host_bytes) {
        status = resource_core_budget_split(allocation, &companion, request->producer, host);
        if (status != RC_ADMISSION_OK) (void)resource_core_budget_cancel(allocation);
    }
    return status;
}

static void cancel_reserved(RcBudgetTicket *ticket) {
    if (ticket->token.ledger && resource_core_budget_cancel(ticket) != RC_ADMISSION_OK)
        rt_fatal("resource allocation lost its unused reservation");
}

static int placement_fits_heap(const VkCtx *c, uint32_t type, VkDeviceSize bytes) {
    uint32_t heap = c->mem_props.memoryTypes[type].heapIndex;
    return heap < c->mem_props.memoryHeapCount && bytes <= c->mem_props.memoryHeaps[heap].size;
}

static VkResult select_placement(int kind, const RcVmaRequest *request, VkCtx *c, const void *create,
                                 const VmaAllocationCreateInfo *allocation, VkMemoryRequirements2 *requirements,
                                 uint32_t *type, RcAdmissionStatus *admission, int check_heap) {
    if (kind == RAW_MEMORY) requirements->memoryRequirements = *(const VkMemoryRequirements *)create;
    else if (kind == IMAGE) {
        const VkImageCreateInfo *ci = create;
        if (ci->flags & VK_IMAGE_CREATE_DISJOINT_BIT) {
            *admission = RC_ADMISSION_UNSUPPORTED;
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        VkDeviceImageMemoryRequirements query = {.sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS, .pCreateInfo = ci};
        vkGetDeviceImageMemoryRequirements(c->device, &query, requirements);
    } else {
        VkDeviceBufferMemoryRequirements query = {.sType = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS, .pCreateInfo = create};
        vkGetDeviceBufferMemoryRequirements(c->device, &query, requirements);
    }
    VmaAllocationCreateInfo eligible = *allocation;
    if (check_heap) {
        uint32_t compatible = requirements->memoryRequirements.memoryTypeBits, bits = 0;
        if (eligible.memoryTypeBits) compatible &= eligible.memoryTypeBits;
        for (uint32_t i = 0; i < c->mem_props.memoryTypeCount; i++)
            if ((compatible & (1u << i)) && placement_fits_heap(c, i, requirements->memoryRequirements.size))
                bits |= 1u << i;
        if (!bits || (allocation->pool && request->pool_memory_type < c->mem_props.memoryTypeCount &&
                      !(bits & (1u << request->pool_memory_type)))) {
            *admission = RC_ADMISSION_UNSUPPORTED;
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        eligible.memoryTypeBits = bits;
    }
    if (allocation->pool) {
        *type = request->pool_memory_type;
        if (!request->pool_block_size || *type >= c->mem_props.memoryTypeCount ||
            !(requirements->memoryRequirements.memoryTypeBits & (1u << *type))) {
            *admission = RC_ADMISSION_UNSUPPORTED;
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        return VK_SUCCESS;
    }
    if (kind == RAW_MEMORY)
        return vmaFindMemoryTypeIndex(c->vma, requirements->memoryRequirements.memoryTypeBits, &eligible, type);
    return kind == IMAGE ? vmaFindMemoryTypeIndexForImageInfo(c->vma, create, &eligible, type)
                         : vmaFindMemoryTypeIndexForBufferInfo(c->vma, create, &eligible, type);
}

static RcAdmissionStatus growth_cost(const RcVmaRequest *request, VkCtx *c, uint32_t type, VkDeviceSize size,
                                     const VmaAllocationCreateInfo *allocation, int dedicated,
                                     const RcAdmissionVector *footprint, RcAdmissionVector *growth) {
    VkDeviceSize bound = growth_bound(c, type, size, request, allocation, dedicated);
    RcAdmissionStatus status = bound ? allocation_cost(c, type, bound, request->producer, growth) : RC_ADMISSION_OVERFLOW;
    if (status == RC_ADMISSION_OK) growth->amount[RC_BUDGET_STAGING] = footprint->amount[RC_BUDGET_STAGING];
    return status;
}

static void describe_reuse(RcVmaDependency *dependency, uint32_t type, const RcAdmissionVector *footprint, int existing) {
    dependency->selected = 1;
    dependency->memory_type = type;
    dependency->can_reuse = existing;
    dependency->released = __atomic_load_n(&V.default_released[type], __ATOMIC_ACQUIRE);
    dependency->reuse.amount[RC_BUDGET_RAM] = footprint->amount[RC_BUDGET_RAM];
    dependency->reuse.amount[RC_BUDGET_STAGING] = footprint->amount[RC_BUDGET_STAGING];
}

static VkResult query_peak(int kind, const RcVmaRequest *request, const void *create,
                           const VmaAllocationCreateInfo *allocation, VmaAllocationCreateInfo *placement,
                           RcAdmissionVector *peak) {
    VkCtx *c = vk_ctx();
    if (!c || !request || !create || !allocation || !placement || !peak || request->host_bytes)
        return VK_ERROR_INITIALIZATION_FAILED;
    RcVmaDependency *dependency = request->dependency;
    if (dependency) {
        *dependency = (RcVmaDependency){0};
        if (allocation->pool) return VK_ERROR_INITIALIZATION_FAILED;
    }
    VkMemoryDedicatedRequirements dedicated = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 requirements = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &dedicated};
    uint32_t type;
    RcAdmissionStatus status;
    VkResult result = select_placement(kind, request, c, create, allocation, &requirements, &type, &status, 1);
    if (result != VK_SUCCESS) return result;
    RcAdmissionVector footprint;
    status = allocation_cost(c, type, requirements.memoryRequirements.size, request->producer, &footprint);
    if (status != RC_ADMISSION_OK) return admission_result(status);
    status = growth_cost(request, c, type, requirements.memoryRequirements.size, allocation,
                         dedicated.requiresDedicatedAllocation, &footprint, peak);
    if (status != RC_ADMISSION_OK) return admission_result(status);
    *placement = *allocation;
    placement->memoryTypeBits = 1u << type;
    if (dependency) {
        int existing = !dedicated.requiresDedicatedAllocation && !(allocation->flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
        describe_reuse(dependency, type, &footprint, existing);
        dependency->peak = *peak;
    }
    return VK_SUCCESS;
}

VkResult resource_core_vma_image_peak(const RcVmaRequest *request, const VkImageCreateInfo *image,
                                     const VmaAllocationCreateInfo *allocation, VmaAllocationCreateInfo *placement,
                                     RcAdmissionVector *peak) {
    return query_peak(IMAGE, request, image, allocation, placement, peak);
}

VkResult resource_core_vma_buffer_peak(const RcVmaRequest *request, const VkBufferCreateInfo *buffer,
                                      const VmaAllocationCreateInfo *allocation, VmaAllocationCreateInfo *placement,
                                      RcAdmissionVector *peak) {
    return query_peak(BUFFER, request, buffer, allocation, placement, peak);
}

static VkResult create_admitted(int kind, const RcVmaRequest *request, VmaAllocator allocator,
                                const void *create, const VmaAllocationCreateInfo *allocation, void *out,
                                VmaAllocation *out_memory, VmaAllocationInfo *info, RcAdmissionStatus *why) {
    RcAdmissionStatus admission = RC_ADMISSION_INVALID;
    if (why) *why = admission;
    if (!request || !allocator || !create || !allocation || (!out && kind != RAW_MEMORY) || !out_memory)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (kind == IMAGE) *(VkImage *)out = VK_NULL_HANDLE;
    else if (kind == BUFFER) *(VkBuffer *)out = VK_NULL_HANDLE;
    *out_memory = NULL;
    if (request->refused_peak) *request->refused_peak = (RcAdmissionVector){{0}};
    RcVmaDependency *dependency = request->dependency;
    if (dependency) *dependency = (RcVmaDependency){0};
    if (!!request->host_bytes != !!request->host_ticket ||
        (request->host_ticket && request->host_ticket->token.ledger) ||
        (dependency && (allocation->pool || request->host_bytes))) return VK_ERROR_INITIALIZATION_FAILED;
    VkCtx *c = vk_ctx();
    if (!c || allocator != c->vma) return VK_ERROR_INITIALIZATION_FAILED;
    VkMemoryDedicatedRequirements dedicated = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 requirements = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &dedicated};
    uint32_t type;
    VkResult result = select_placement(kind, request, c, create, allocation, &requirements, &type, &admission,
                                       dependency || request->peak_ticket);
    if (result != VK_SUCCESS) {
        if (why) *why = admission;
        return result;
    }
    RcAdmissionVector footprint;
    admission = allocation_cost(c, type, requirements.memoryRequirements.size, request->producer, &footprint);
    if (admission != RC_ADMISSION_OK) {
        if (why) *why = admission;
        return admission_result(admission);
    }
    int existing = !dedicated.requiresDedicatedAllocation && !(allocation->flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    if (dependency) {
        describe_reuse(dependency, type, &footprint, existing);
        resource_core_budget_readiness(request->required, &dependency->before);
    }
    uint32_t slot = owner_take();
    if (slot == NONE) {
        if (dependency) dependency->owner_full = 1;
        if (why) *why = RC_ADMISSION_NO_RECORDS;
        return VK_NOT_READY;
    }
    VmaAllocationCreateInfo selected = *allocation;
    selected.memoryTypeBits = 1u << type; // the native allocation cannot silently move to an unreserved heap
    VmaAllocation memory = NULL;
    VmaAllocationInfo actual;
    RcBudgetTicket ticket = {0}, host = {0};
    result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (existing) {
        VmaAllocationCreateInfo reuse = selected;
        reuse.flags |= VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT;
        result = native_create(kind, allocator, create, &reuse, out, &memory, &actual);
        if (result == VK_SUCCESS) {
            if (actual.memoryType != type) {
                native_destroy(kind, allocator, out, memory);
                owner_unused(slot);
                if (why) *why = RC_ADMISSION_UNSUPPORTED;
                return VK_ERROR_FEATURE_NOT_PRESENT;
            }
            admission = allocation_cost(c, actual.memoryType, actual.size, request->producer, &footprint);
            RcAdmissionVector working = {{0}};
            // Existing VkDeviceMemory is already committed and occupies its heap. Newly touched system-memory
            // pages and a newly occupied staging span still consume their respective discretionary allowances.
            working.amount[RC_BUDGET_RAM] = footprint.amount[RC_BUDGET_RAM];
            working.amount[RC_BUDGET_STAGING] = footprint.amount[RC_BUDGET_STAGING];
            if (request->refused_peak) *request->refused_peak = working;
            if (dependency) dependency->peak = working;
            if (admission == RC_ADMISSION_OK && !request->host_bytes &&
                (!request->peak_ticket || (!working.amount[RC_BUDGET_RAM] && !working.amount[RC_BUDGET_STAGING])))
                admission = resource_core_budget_adopt_compatible(request->producer, request->required,
                                                                  &footprint, &working, &ticket);
            else if (admission == RC_ADMISSION_OK) {
                admission = reserve_bundle(request, &working, &ticket, &host);
                if (admission == RC_ADMISSION_OK)
                    admission = resource_core_budget_materialize_compatible(&ticket, &footprint, &working);
            }
            if (!rc_admission_accounted(admission)) {
                native_destroy(kind, allocator, out, memory);
                cancel_reserved(&ticket);
                cancel_reserved(&host);
                owner_unused(slot);
                if (why) *why = admission;
                return admission_result(admission);
            }
        } else if (result != VK_ERROR_OUT_OF_DEVICE_MEMORY) {
            owner_unused(slot);
            return result;
        }
    }
    if (result != VK_SUCCESS) {
        if (selected.flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) {
            owner_unused(slot);
            return result;
        }
        RcAdmissionVector growth = {{0}};
        admission = growth_cost(request, c, type, requirements.memoryRequirements.size, &selected,
                                dedicated.requiresDedicatedAllocation, &footprint, &growth);
        if (admission == RC_ADMISSION_OK)
            admission = reserve_bundle(request, &growth, &ticket, &host);
        if (admission != RC_ADMISSION_OK) {
            owner_unused(slot);
            if (why) *why = admission;
            return admission_result(admission);
        }
        result = native_create(kind, allocator, create, &selected, out, &memory, &actual);
        if (result != VK_SUCCESS) {
            // A failed driver allocation can still leave allocator blocks or driver bookkeeping behind. Keep the
            // conservative growth debt until the next observation instead of returning stale physical headroom.
            (void)resource_core_budget_opaque_done(&ticket);
            cancel_reserved(&host);
            owner_unused(slot);
            return result;
        }
        if (actual.memoryType != type) {
            native_destroy(kind, allocator, out, memory);
            (void)resource_core_budget_opaque_done(&ticket);
            cancel_reserved(&host);
            owner_unused(slot);
            if (why) *why = RC_ADMISSION_UNSUPPORTED;
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        admission = allocation_cost(c, actual.memoryType, actual.size, request->producer, &footprint);
        if (admission == RC_ADMISSION_OK)
            admission = resource_core_budget_materialize_growth(&ticket, &footprint, &growth);
        if (!rc_admission_accounted(admission)) {
            native_destroy(kind, allocator, out, memory);
            (void)resource_core_budget_opaque_done(&ticket);
            cancel_reserved(&host);
            owner_unused(slot);
            if (why) *why = admission;
            return admission_result(admission);
        }
    }
    owner_publish(slot, memory, ticket, actual.memoryType, !allocation->pool);
    if (request->host_ticket) *request->host_ticket = host;
    *out_memory = memory;
    if (info) *info = actual;
    if (why) *why = admission;
    return VK_SUCCESS;
}

VkResult resource_core_vma_create_image(const RcVmaRequest *request, VmaAllocator allocator,
                                        const VkImageCreateInfo *image, const VmaAllocationCreateInfo *allocation,
                                        VkImage *out_image, VmaAllocation *out_allocation, VmaAllocationInfo *info,
                                        RcAdmissionStatus *admission) {
    return create_admitted(IMAGE, request, allocator, image, allocation, out_image, out_allocation, info, admission);
}

VkResult resource_core_vma_create_buffer(const RcVmaRequest *request, VmaAllocator allocator,
                                         const VkBufferCreateInfo *buffer, const VmaAllocationCreateInfo *allocation,
                                         VkBuffer *out_buffer, VmaAllocation *out_allocation, VmaAllocationInfo *info,
                                         RcAdmissionStatus *admission) {
    return create_admitted(BUFFER, request, allocator, buffer, allocation, out_buffer, out_allocation, info, admission);
}

VkResult resource_core_vma_allocate_memory(const RcVmaRequest *request, VmaAllocator allocator,
                                          const VkMemoryRequirements *requirements,
                                          const VmaAllocationCreateInfo *allocation, VmaAllocation *out_allocation,
                                          VmaAllocationInfo *info, RcAdmissionStatus *admission) {
    return create_admitted(RAW_MEMORY, request, allocator, requirements, allocation, NULL, out_allocation, info, admission);
}

VkResult resource_core_vma_create_pool(const RcVmaRequest *request, VmaAllocator allocator,
                                      const VmaPoolCreateInfo *create, VmaPool *out_pool, RcAdmissionStatus *why) {
    RcAdmissionStatus status = RC_ADMISSION_INVALID;
    if (why) *why = status;
    if (!request || !allocator || !create || !out_pool) return VK_ERROR_INITIALIZATION_FAILED;
    *out_pool = NULL;
    if (request->host_bytes || request->host_ticket || request->peak_ticket) return VK_ERROR_INITIALIZATION_FAILED;
    VkCtx *c = vk_ctx();
    if (!c || allocator != c->vma) return VK_ERROR_INITIALIZATION_FAILED;
    RcBudgetTicket ticket = {0};
    if (create->minBlockCount) {
        if (!create->blockSize || create->minBlockCount > UINT64_MAX / create->blockSize) {
            if (why) *why = RC_ADMISSION_UNSUPPORTED;
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        RcAdmissionVector growth;
        status = allocation_cost(c, create->memoryTypeIndex, create->blockSize * create->minBlockCount,
                                  request->producer, &growth);
        growth.amount[RC_BUDGET_STAGING] = 0; // backing capacity has no occupied staging span yet
        if (status == RC_ADMISSION_OK)
            status = resource_core_budget_reserve(request->producer, request->required, &growth, &ticket);
        if (status != RC_ADMISSION_OK) {
            if (why) *why = status;
            return admission_result(status);
        }
    }
    VkResult result = vmaCreatePool(allocator, create, out_pool);
    if (ticket.token.ledger && resource_core_budget_opaque_done(&ticket) != RC_ADMISSION_OK)
        rt_fatal("resource pool construction lost its capacity owner");
    if (why) *why = RC_ADMISSION_OK;
    return result;
}

RcAdmissionStatus resource_core_vma_retire(VmaAllocation allocation) {
    RcAdmissionStatus status = RC_ADMISSION_OK;
    plat_mutex_lock(&V.lock);
    if (V.initialized && allocation)
        for (uint32_t i = V.bucket[allocation_hash(allocation)]; i != NONE; i = V.owner[i].next)
            if (V.owner[i].allocation == allocation) {
                if (V.owner[i].ticket.token.state == RC_ADMISSION_RETAINED)
                    status = resource_core_budget_retire(&V.owner[i].ticket);
                break;
            }
    plat_mutex_unlock(&V.lock);
    return status;
}

RcAdmissionStatus resource_core_vma_allocation_cost(VmaAllocation allocation, RcAdmissionVector *out) {
    if (!allocation || !out) return RC_ADMISSION_INVALID;
    RcAdmissionStatus status = RC_ADMISSION_STALE;
    plat_mutex_lock(&V.lock);
    if (V.initialized)
        for (uint32_t i = V.bucket[allocation_hash(allocation)]; i != NONE; i = V.owner[i].next)
            if (V.owner[i].allocation == allocation) {
                status = resource_core_budget_ticket_cost(&V.owner[i].ticket, out);
                break;
            }
    plat_mutex_unlock(&V.lock);
    return status;
}

static void destroy_owned(int kind, VmaAllocator allocator, void *handle, VmaAllocation allocation) {
    AllocationOwner owner;
    int owned = owner_detach(allocation, &owner);
    if (owned && owner.ticket.token.state == RC_ADMISSION_RETAINED && resource_core_budget_retire(&owner.ticket) != RC_ADMISSION_OK)
        rt_fatal("resource allocation retirement lost its capacity owner");
    native_destroy(kind, allocator, handle, allocation);
    if (owned && resource_core_budget_destroyed(&owner.ticket) != RC_ADMISSION_OK)
        rt_fatal("resource allocation destruction lost its capacity owner");
    if (owned && owner.default_pool)
        __atomic_add_fetch(&V.default_released[owner.memory_type], 1, __ATOMIC_RELEASE);
}

void resource_core_vma_destroy_image(VmaAllocator allocator, VkImage image, VmaAllocation allocation) {
    destroy_owned(IMAGE, allocator, &image, allocation);
}
void resource_core_vma_destroy_buffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation) {
    destroy_owned(BUFFER, allocator, &buffer, allocation);
}
void resource_core_vma_free_memory(VmaAllocator allocator, VmaAllocation allocation) {
    destroy_owned(RAW_MEMORY, allocator, NULL, allocation);
}

#ifdef RC_VMA_TEST
void resource_core_vma_test_reset(void) { memset(&V, 0, sizeof V); }
#endif

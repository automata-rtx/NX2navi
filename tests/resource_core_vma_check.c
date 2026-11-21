#include <assert.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define RC_BUDGET_TEST
#define RC_VMA_TEST
#include "../runtime/resource_core_vma.h"
#include "../runtime/platform/platform.h"

#define MB (1024ull * 1024ull)
static RcBudgetSample sample;
static uint64_t clock_ns = 1000000000;
static unsigned budget_lock_depth, map_lock_depth;
static jmp_buf fatal_jump;
static int expect_duplicate_owner;
uint64_t rc_budget_test_now_ns(void) { return clock_ns; }
void rc_budget_test_lock(void) { assert(!budget_lock_depth); budget_lock_depth++; }
void rc_budget_test_unlock(void) { assert(budget_lock_depth == 1); budget_lock_depth--; }
void rc_budget_test_sample(RcBudgetSample *out) { *out = sample; }
void rc_budget_test_wait(uint32_t milliseconds) { (void)milliseconds; assert(!"VMA admission must not wait"); }
void plat_mutex_lock(PlatMutex *lock) { (void)lock; assert(!map_lock_depth); map_lock_depth++; }
void plat_mutex_unlock(PlatMutex *lock) { (void)lock; assert(map_lock_depth == 1); map_lock_depth--; }
int plat_commit_counts_vram(void) { return 1; }
void rt_fatal(const char *fmt, ...) {
    if (expect_duplicate_owner) {
        assert(!strcmp(fmt, "resource allocation acquired capacity ownership twice"));
        longjmp(fatal_jump, 1);
    }
    abort();
}
#include "../runtime/resource_core_budget.c"
#include "../runtime/resource_core_vma.c"

static VkCtx context;
static struct {
    VkDeviceSize need, expected_growth;
    uint32_t type, eligible_types, expected_types;
    unsigned reuse_calls, new_calls, destroys, serial, raw_calls, raw_frees;
    int reuse, fail, dedicated, recycle;
    void (*after_free)(VmaAllocation);
    struct { VmaAllocation allocation; int live; } live[32];
} native;

VkCtx *vk_ctx(void) { return &context; }
VkDeviceSize vk_vma_block_size(void) { return 64 * MB; }

static void requirements(VkMemoryRequirements2 *out) {
    out->memoryRequirements = (VkMemoryRequirements){native.need, 256,
        native.eligible_types ? native.eligible_types : 1u << native.type};
    ((VkMemoryDedicatedRequirements *)out->pNext)->requiresDedicatedAllocation = native.dedicated;
}
VKAPI_ATTR void VKAPI_CALL vkGetDeviceImageMemoryRequirements(VkDevice device, const VkDeviceImageMemoryRequirements *query,
                                                             VkMemoryRequirements2 *out) {
    (void)device; (void)query; requirements(out);
}
VKAPI_ATTR void VKAPI_CALL vkGetDeviceBufferMemoryRequirements(VkDevice device, const VkDeviceBufferMemoryRequirements *query,
                                                              VkMemoryRequirements2 *out) {
    (void)device; (void)query; requirements(out);
}
VkResult vmaFindMemoryTypeIndexForImageInfo(VmaAllocator a, const VkImageCreateInfo *i,
                                          const VmaAllocationCreateInfo *c, uint32_t *type) {
    (void)a; (void)i;
    if (native.expected_types) assert(c->memoryTypeBits == native.expected_types);
    *type = native.type; return VK_SUCCESS;
}
VkResult vmaFindMemoryTypeIndexForBufferInfo(VmaAllocator a, const VkBufferCreateInfo *i,
                                           const VmaAllocationCreateInfo *c, uint32_t *type) {
    (void)a; (void)i; (void)c; *type = native.type; return VK_SUCCESS;
}
VkResult vmaFindMemoryTypeIndex(VmaAllocator a, uint32_t bits, const VmaAllocationCreateInfo *c, uint32_t *type) {
    (void)a; (void)c; assert(bits & (1u << native.type)); *type = native.type; return VK_SUCCESS;
}
VkResult vmaCreatePool(VmaAllocator a, const VmaPoolCreateInfo *create, VmaPool *out) {
    (void)a;
    RcBudgetSnapshot observed;
    resource_core_budget_snapshot(&observed);
    assert(observed.reserved.amount[RC_BUDGET_GPU_FIRST + create->memoryTypeIndex] ==
           create->minBlockCount * create->blockSize);
    assert(!observed.reserved.amount[RC_BUDGET_STAGING]);
    *out = (VmaPool)(uintptr_t)1;
    return VK_SUCCESS;
}

static VkResult allocate(const VmaAllocationCreateInfo *create, VmaAllocation *allocation, VmaAllocationInfo *info) {
    assert(!budget_lock_depth && !map_lock_depth);
    assert(create->memoryTypeBits == (1u << native.type));
    if (create->flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) {
        native.reuse_calls++;
        if (!native.reuse) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    } else {
        native.new_calls++;
        RcBudgetSnapshot observed;
        resource_core_budget_snapshot(&observed);
        uint32_t domain = RC_BUDGET_GPU_FIRST + context.mem_props.memoryTypes[native.type].heapIndex;
        assert(observed.reserved.amount[domain] == native.expected_growth);
        if (native.fail) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        sample.free.amount[domain] -= native.expected_growth;
        sample.free.amount[RC_BUDGET_COMMIT] -= native.expected_growth;
        if (native.type == 1) sample.free.amount[RC_BUDGET_RAM] -= native.expected_growth;
    }
    uint32_t index = native.serial;
    if (native.recycle)
        for (uint32_t i = 0; i < native.serial; i++)
            if (!native.live[i].live) { index = i; break; }
    if (index == native.serial) {
        assert(native.serial < 32);
        native.serial++;
    }
    *allocation = (VmaAllocation)(uintptr_t)(0x1000 + index * 16);
    native.live[index].allocation = *allocation;
    native.live[index].live = 1;
    *info = (VmaAllocationInfo){.size = native.need, .memoryType = native.type};
    return VK_SUCCESS;
}
VkResult vmaCreateImage(VmaAllocator a, const VkImageCreateInfo *i, const VmaAllocationCreateInfo *c,
                       VkImage *out, VmaAllocation *allocation, VmaAllocationInfo *info) {
    (void)a; (void)i;
    VkResult result = allocate(c, allocation, info);
    *out = result == VK_SUCCESS ? (VkImage)(uintptr_t)(0x2000 + native.serial * 16) : VK_NULL_HANDLE;
    return result;
}
VkResult vmaCreateBuffer(VmaAllocator a, const VkBufferCreateInfo *i, const VmaAllocationCreateInfo *c,
                        VkBuffer *out, VmaAllocation *allocation, VmaAllocationInfo *info) {
    (void)a; (void)i;
    VkResult result = allocate(c, allocation, info);
    *out = result == VK_SUCCESS ? (VkBuffer)(uintptr_t)(0x3000 + native.serial * 16) : VK_NULL_HANDLE;
    return result;
}
VkResult vmaAllocateMemory(VmaAllocator a, const VkMemoryRequirements *r, const VmaAllocationCreateInfo *c,
                          VmaAllocation *allocation, VmaAllocationInfo *info) {
    (void)a; assert(r->size == native.need); native.raw_calls++;
    return allocate(c, allocation, info);
}
static void release(VmaAllocation allocation) {
    assert(!budget_lock_depth && !map_lock_depth);
    for (unsigned i = 0; i < native.serial; i++)
        if (native.live[i].allocation == allocation) {
            assert(native.live[i].live);
            native.live[i].live = 0;
            native.destroys++;
            if (native.after_free) {
                void (*callback)(VmaAllocation) = native.after_free;
                native.after_free = NULL;
                callback(allocation);
            }
            return;
        }
    assert(!"Unknown native allocation destroyed");
}
void vmaDestroyImage(VmaAllocator a, VkImage image, VmaAllocation allocation) {
    (void)a; (void)image; release(allocation);
}
void vmaDestroyBuffer(VmaAllocator a, VkBuffer buffer, VmaAllocation allocation) {
    (void)a; (void)buffer; release(allocation);
}
void vmaFreeMemory(VmaAllocator a, VmaAllocation allocation) {
    (void)a; native.raw_frees++; release(allocation);
}

static void start(uint64_t room_mb) {
    resource_core_vma_test_reset();
    resource_core_budget_test_reset();
    memset(&native, 0, sizeof native);
    native.need = 4 * MB;
    native.expected_growth = 64 * MB;
    context = (VkCtx){.vma = (VmaAllocator)(uintptr_t)1, .device = (VkDevice)(uintptr_t)1};
    context.props.deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    context.mem_props.memoryTypeCount = context.mem_props.memoryHeapCount = 2;
    context.mem_props.memoryTypes[0] = (VkMemoryType){VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};
    context.mem_props.memoryTypes[1] = (VkMemoryType){VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 1};
    context.mem_props.memoryHeaps[0] = (VkMemoryHeap){8192 * MB, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    context.mem_props.memoryHeaps[1] = (VkMemoryHeap){8192 * MB, 0};
    sample = (RcBudgetSample){.known_mask = (1u << (RC_BUDGET_GPU_FIRST + 2)) - 1};
    sample.total.amount[RC_BUDGET_RAM] = 16384 * MB;
    sample.total.amount[RC_BUDGET_COMMIT] = 32768 * MB;
    sample.free.amount[RC_BUDGET_RAM] = sample.free.amount[RC_BUDGET_COMMIT] = (room_mb + 256) * MB;
    sample.total.amount[RC_BUDGET_STAGING] = sample.free.amount[RC_BUDGET_STAGING] = 64 * MB;
    for (unsigned h = 0; h < 2; h++) {
        sample.total.amount[RC_BUDGET_GPU_FIRST + h] = 8192 * MB;
        sample.free.amount[RC_BUDGET_GPU_FIRST + h] = (room_mb + 64) * MB;
    }
    resource_core_budget_refresh();
}

static VkResult image(RcVmaRequest request, VmaAllocationCreateFlags flags, VkImage *out, VmaAllocation *allocation,
                       RcAdmissionStatus *status) {
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, .flags = flags};
    return resource_core_vma_create_image(&request, context.vma, &ci, &ai, out, allocation, NULL, status);
}

static VkResult staging(VkBuffer *buffer, VmaAllocation *allocation) {
    RcVmaRequest request = {.producer = RC_BUDGET_STAGE, .required = 1,
                            .pool_memory_type = 1, .pool_block_size = 32 * MB};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4 * MB};
    VmaAllocationCreateInfo ai = {.pool = (VmaPool)(uintptr_t)1};
    RcAdmissionStatus status;
    return resource_core_vma_create_buffer(&request, context.vma, &bi, &ai, buffer, allocation, NULL, &status);
}

static VkBuffer replacement_buffer;
static VmaAllocation replacement_allocation;
static void reuse_during_native_free(VmaAllocation released) {
    // Model another producer receiving the allocator's recycled address before the old destroy call returns.
    assert(staging(&replacement_buffer, &replacement_allocation) == VK_SUCCESS);
    assert(replacement_allocation == released);
    RcBudgetSnapshot observed;
    resource_core_budget_snapshot(&observed);
    assert(observed.pending.amount[RC_BUDGET_STAGING] == 4 * MB);
    assert(observed.retained.amount[RC_BUDGET_STAGING] == 4 * MB);
}

static void caller_owned_staging_recycle(void) {
    VkBuffer buffer;
    VmaAllocation allocation;
    RcBudgetSnapshot observed;
    start(256);
    native.type = native.reuse = native.recycle = 1;
    assert(staging(&buffer, &allocation) == VK_SUCCESS);
    // Reproduce the former menu upload cleanup: native memory is free but its capacity owner survives.
    vmaDestroyBuffer(context.vma, buffer, allocation);
    resource_core_budget_snapshot(&observed);
    assert(observed.retained.amount[RC_BUDGET_STAGING] == 4 * MB);
    expect_duplicate_owner = 1;
    if (!setjmp(fatal_jump)) {
        (void)staging(&buffer, &allocation);
        assert(!"Raw staging destruction must reproduce the duplicate-owner fatal on address reuse");
    }
    expect_duplicate_owner = 0;
    assert(!budget_lock_depth && map_lock_depth == 1);
    // The expected fatal abandons that simulated process; reset its fake lock and ledger before the fixed case.
    map_lock_depth = 0;
    start(256);
    native.type = native.reuse = native.recycle = 1;
    for (unsigned i = 0; i < 16; i++) {
        assert(staging(&buffer, &allocation) == VK_SUCCESS);
        assert(native.serial == 1);
        resource_core_vma_destroy_buffer(context.vma, buffer, allocation);
        resource_core_budget_snapshot(&observed);
        assert(!observed.retained.amount[RC_BUDGET_STAGING] && !observed.pending.amount[RC_BUDGET_STAGING]);
    }
    assert(staging(&buffer, &allocation) == VK_SUCCESS);
    native.after_free = reuse_during_native_free;
    resource_core_vma_destroy_buffer(context.vma, buffer, allocation);
    resource_core_budget_snapshot(&observed);
    assert(!observed.pending.amount[RC_BUDGET_STAGING]);
    assert(observed.retained.amount[RC_BUDGET_STAGING] == 4 * MB);
    // Settling the old copied ticket must preserve the replacement's map entry and exact retained charge.
    resource_core_vma_destroy_buffer(context.vma, replacement_buffer, replacement_allocation);
    resource_core_budget_snapshot(&observed);
    assert(!observed.retained.amount[RC_BUDGET_STAGING] && !observed.pending.amount[RC_BUDGET_STAGING]);
    assert(native.serial == 1 && native.destroys == 18);
}

static void required_staging_burst(void) {
    start(4096);
    native.type = native.reuse = 1;
    VkBuffer buffers[18], refused_buffer;
    VmaAllocation allocations[18], refused_allocation;
    // One open batch can hold more occupied spans than the optional scheduling target before it can drain.
    for (unsigned i = 0; i < 18; i++) assert(staging(&buffers[i], &allocations[i]) == VK_SUCCESS);
    RcBudgetSnapshot observed;
    resource_core_budget_snapshot(&observed);
    assert(observed.retained.amount[RC_BUDGET_STAGING] == 72 * MB);
    assert(!observed.optional_available.amount[RC_BUDGET_STAGING] && !native.new_calls);
    assert(observed.unobserved.amount[RC_BUDGET_RAM] == 72 * MB);
    assert(!observed.unobserved.amount[RC_BUDGET_COMMIT] && !observed.unobserved.amount[RC_BUDGET_GPU_FIRST + 1]);
    RcVmaRequest request = {.producer = RC_BUDGET_STAGE, .required = 0,
                            .pool_memory_type = 1, .pool_block_size = 32 * MB};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4 * MB};
    VmaAllocationCreateInfo ai = {.pool = (VmaPool)(uintptr_t)1};
    RcAdmissionStatus status;
    assert(resource_core_vma_create_buffer(&request, context.vma, &bi, &ai,
        &refused_buffer, &refused_allocation, NULL, &status) == VK_NOT_READY);
    resource_core_budget_snapshot(&observed);
    assert(observed.producer[RC_BUDGET_STAGE].last_refusal.capacity_mask == (1u << RC_BUDGET_STAGING));
    // Even a compatible block cannot grant new host working-set growth under real RAM pressure.
    sample.free.amount[RC_BUDGET_RAM] = 256 * MB;
    clock_ns += 200000000;
    resource_core_budget_refresh();
    assert(staging(&refused_buffer, &refused_allocation) == VK_NOT_READY);
    resource_core_budget_snapshot(&observed);
    assert(observed.producer[RC_BUDGET_STAGE].last_refusal.capacity_mask == (1u << RC_BUDGET_RAM));
    assert(observed.retained.amount[RC_BUDGET_STAGING] == 72 * MB && native.destroys == 2);
    for (unsigned i = 0; i < 18; i++) resource_core_vma_destroy_buffer(context.vma, buffers[i], allocations[i]);
    resource_core_budget_snapshot(&observed);
    assert(!observed.retained.amount[RC_BUDGET_STAGING] && !observed.pending.amount[RC_BUDGET_STAGING]);
}

static void companion_peak(void) {
    VkImage handle;
    VmaAllocation allocation;
    RcAdmissionStatus status;
    RcBudgetSnapshot observed;
    RcBudgetTicket host = {0};
    RcVmaRequest request = {.producer = RC_BUDGET_SNAPSHOT, .required = 1,
                            .host_bytes = 80 * MB, .host_ticket = &host};
    // GPU backing alone and the host copy alone fit; their simultaneous commit peak does not.
    start(128);
    assert(image(request, 0, &handle, &allocation, &status) == VK_NOT_READY);
    assert(!native.new_calls && !host.token.ledger && !handle && !allocation);
    resource_core_budget_snapshot(&observed);
    assert(!observed.reserved.amount[RC_BUDGET_COMMIT]);

    start(256);
    assert(image(request, 0, &handle, &allocation, &status) == VK_SUCCESS);
    resource_core_budget_snapshot(&observed);
    assert(observed.reserved.amount[RC_BUDGET_COMMIT] == 80 * MB);
    assert(observed.reserved.amount[RC_BUDGET_RAM] == 80 * MB);
    assert(host.token.state == RC_ADMISSION_RESERVED);
    RcAdmissionVector occupied;
    assert(resource_core_vma_allocation_cost(allocation, &occupied) == RC_ADMISSION_OK);
    assert(occupied.amount[RC_BUDGET_GPU_FIRST] == 4 * MB && !occupied.amount[RC_BUDGET_RAM]);
    RcAdmissionVector host_cost = {{0}};
    host_cost.amount[RC_BUDGET_RAM] = host_cost.amount[RC_BUDGET_COMMIT] = 80 * MB;
    assert(rc_admission_accounted(resource_core_budget_materialize(&host, &host_cost)));
    resource_core_vma_destroy_image(context.vma, handle, allocation);
    assert(resource_core_vma_allocation_cost(allocation, &occupied) == RC_ADMISSION_STALE);
    assert(resource_core_budget_retire(&host) == RC_ADMISSION_OK);
    assert(resource_core_budget_destroyed(&host) == RC_ADMISSION_OK);

    // Existing GPU backing must not be charged as new growth alongside the host reservation.
    start(80);
    native.reuse = 1;
    assert(image(request, 0, &handle, &allocation, &status) == VK_SUCCESS);
    resource_core_budget_snapshot(&observed);
    assert(!observed.unobserved.amount[RC_BUDGET_GPU_FIRST]);
    assert(observed.retained.amount[RC_BUDGET_GPU_FIRST] == 4 * MB);
    assert(observed.reserved.amount[RC_BUDGET_COMMIT] == 80 * MB);
    assert(resource_core_budget_cancel(&host) == RC_ADMISSION_OK);
    resource_core_vma_destroy_image(context.vma, handle, allocation);

    start(256);
    native.fail = 1;
    assert(image(request, 0, &handle, &allocation, &status) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
    resource_core_budget_snapshot(&observed);
    assert(!host.token.ledger && !observed.reserved.amount[RC_BUDGET_COMMIT]);
    assert(observed.unobserved.amount[RC_BUDGET_COMMIT] == 64 * MB);
}

static void exact_placement_progress(void) {
    start(4096);
    RcVmaRequest request = {.producer = RC_BUDGET_BUFFER};
    VkImage same, other, image_handle;
    VmaAllocation same_memory, other_memory, image_memory;
    RcAdmissionStatus status;
    assert(image(request, 0, &same, &same_memory, &status) == VK_SUCCESS);
    // Another type on the same heap is still incompatible with the selected default-pool reuse path.
    context.mem_props.memoryTypes[1].heapIndex = 0;
    native.type = 1;
    assert(image(request, 0, &other, &other_memory, &status) == VK_SUCCESS);
    native.type = 0;
    VkBuffer separate;
    VmaAllocation separate_memory;
    VkBufferCreateInfo buffer_info = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = native.need};
    VmaAllocationCreateInfo allocation_info = {.pool = (VmaPool)(uintptr_t)2};
    RcVmaRequest custom = {.producer = RC_BUDGET_BUFFER, .pool_memory_type = 0, .pool_block_size = 64 * MB};
    assert(resource_core_vma_create_buffer(&custom, context.vma, &buffer_info, &allocation_info,
                                          &separate, &separate_memory, NULL, &status) == VK_SUCCESS);
    sample.free.amount[RC_BUDGET_GPU_FIRST] = 64 * MB;
    clock_ns += 100000000;
    resource_core_budget_refresh();

    RcVmaDependency dependency;
    request.dependency = &dependency;
    assert(image(request, 0, &image_handle, &image_memory, &status) == VK_NOT_READY);
    assert(status == RC_ADMISSION_REFUSED && !image_handle && !image_memory);
    assert(dependency.selected && dependency.peak.amount[RC_BUDGET_GPU_FIRST] == 64 * MB);
    RcBudgetReadiness now;
    resource_core_budget_readiness(0, &now);
    assert(!resource_core_vma_dependency_ready(&dependency, &now));

    assert(!resource_core_vma_dependency_retirement(&dependency, same_memory));
    assert(resource_core_vma_retire(same_memory) == RC_ADMISSION_OK);
    assert(resource_core_vma_retire(other_memory) == RC_ADMISSION_OK);
    assert(resource_core_vma_retire(separate_memory) == RC_ADMISSION_OK);
    assert(resource_core_vma_dependency_retirement(&dependency, same_memory));
    assert(!resource_core_vma_dependency_retirement(&dependency, other_memory));
    assert(!resource_core_vma_dependency_retirement(&dependency, separate_memory));
    RcVmaDependency exhausted = dependency;
    exhausted.owner_full = 1;
    assert(resource_core_vma_dependency_retirement(&exhausted, separate_memory));
    resource_core_budget_readiness(0, &now);
    assert(!resource_core_vma_dependency_ready(&dependency, &now));
    resource_core_vma_destroy_image(context.vma, other, other_memory);
    resource_core_vma_destroy_buffer(context.vma, separate, separate_memory);
    resource_core_budget_readiness(0, &now);
    assert(!resource_core_vma_dependency_ready(&dependency, &now));

    // The sample still permits no physical growth, but a completed same-type release can supply reuse.
    resource_core_vma_destroy_image(context.vma, same, same_memory);
    assert(resource_core_vma_dependency_ready(&dependency, &now));
    native.reuse = 1;
    unsigned growth = native.new_calls;
    assert(image(request, 0, &image_handle, &image_memory, &status) == VK_SUCCESS);
    assert(native.new_calls == growth);
    resource_core_vma_destroy_image(context.vma, image_handle, image_memory);

    native.need = context.mem_props.memoryHeaps[0].size + 1;
    assert(image(request, 0, &image_handle, &image_memory, &status) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
    assert(status == RC_ADMISSION_UNSUPPORTED && !dependency.selected && !image_handle && !image_memory);
    assert(native.new_calls == growth);
    native.eligible_types = 3;
    native.expected_types = 2;
    native.type = 1;
    context.mem_props.memoryTypes[1].heapIndex = 1;
    context.mem_props.memoryHeaps[1].size = 10000 * MB;
    VkImageCreateInfo info = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO};
    VmaAllocationCreateInfo viable;
    RcAdmissionVector viable_peak;
    assert(resource_core_vma_image_peak(&request, &info, &ai, &viable, &viable_peak) == VK_SUCCESS);
    assert(viable.memoryTypeBits == 2 && viable_peak.amount[RC_BUDGET_GPU_FIRST + 1] >= native.need);
}

static void preparation_children_use_one_peak(void) {
    start(4096);
    RcVmaRequest image_request = {.producer = RC_BUDGET_TEXTURE}, stage_request = {.producer = RC_BUDGET_STAGE};
    VkImageCreateInfo image_info = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    VkBufferCreateInfo buffer_info = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = native.need};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO}, image_placement, stage_placement;
    RcAdmissionVector image_peak, stage_peak;
    RcVmaDependency planned;
    image_request.dependency = &planned;
    assert(resource_core_vma_image_peak(&image_request, &image_info, &ai, &image_placement, &image_peak) == VK_SUCCESS);
    assert(planned.selected && planned.can_reuse && planned.memory_type == 0 &&
           planned.peak.amount[RC_BUDGET_GPU_FIRST] == image_peak.amount[RC_BUDGET_GPU_FIRST] &&
           !planned.reuse.amount[RC_BUDGET_GPU_FIRST]);
    RcVmaDependency smaller = planned;
    smaller.reuse.amount[RC_BUDGET_RAM] = MB; // conversion workspace in the caller's aggregate
    smaller.before = (RcBudgetReadiness){.records_available = 1, .usable_mask = 1u << RC_BUDGET_RAM};
    RcBudgetReadiness room = smaller.before;
    room.available.amount[RC_BUDGET_RAM] = MB;
    assert(!resource_core_vma_dependency_ready(&smaller, &room));
    smaller.reuse_unattempted = 1;
    assert(resource_core_vma_dependency_ready(&smaller, &room));
    smaller.before = room;
    assert(!resource_core_vma_dependency_ready(&smaller, &room));
    image_request.dependency = NULL;
    native.type = 1;
    assert(resource_core_vma_buffer_peak(&stage_request, &buffer_info, &ai, &stage_placement, &stage_peak) == VK_SUCCESS);
    assert(image_placement.memoryTypeBits == 1 && stage_placement.memoryTypeBits == 2);
    assert(!native.new_calls && !native.reuse_calls && stage_peak.amount[RC_BUDGET_STAGING] == native.need);

    RcAdmissionVector combined = {{0}};
    for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) combined.amount[d] = image_peak.amount[d] + stage_peak.amount[d];
    combined.amount[RC_BUDGET_RAM] += MB;
    combined.amount[RC_BUDGET_COMMIT] += MB;
    RcBudgetTicket preparation = {0};
    assert(resource_core_budget_reserve(RC_BUDGET_TEXTURE, 0, &combined, &preparation) == RC_ADMISSION_OK);
    image_request.peak_ticket = stage_request.peak_ticket = &preparation;
    VkBuffer buffer;
    VkImage destination;
    VmaAllocation stage, image;
    RcAdmissionStatus status;
    assert(resource_core_vma_create_buffer(&stage_request, context.vma, &buffer_info, &stage_placement,
                                          &buffer, &stage, NULL, &status) == VK_SUCCESS);
    native.type = 0;
    assert(resource_core_vma_create_image(&image_request, context.vma, &image_info, &image_placement,
                                         &destination, &image, NULL, &status) == VK_SUCCESS);
    RcAdmissionVector remaining;
    assert(resource_core_budget_ticket_cost(&preparation, &remaining) == RC_ADMISSION_OK);
    assert(remaining.amount[RC_BUDGET_RAM] == MB && remaining.amount[RC_BUDGET_COMMIT] == MB &&
           !remaining.amount[RC_BUDGET_STAGING] && !remaining.amount[RC_BUDGET_GPU_FIRST]);
    assert(resource_core_budget_cancel(&preparation) == RC_ADMISSION_OK);
    RcBudgetSnapshot charged;
    resource_core_budget_snapshot(&charged);
    assert(charged.producer[RC_BUDGET_TEXTURE].admitted == 1 && !charged.producer[RC_BUDGET_STAGE].admitted);
    assert(charged.retained.amount[RC_BUDGET_STAGING] == native.need && !charged.reserved.amount[RC_BUDGET_RAM]);
    resource_core_vma_destroy_image(context.vma, destination, image);
    resource_core_vma_destroy_buffer(context.vma, buffer, stage);
}

int main(void) {
    RcVmaRequest request = {.producer = RC_BUDGET_TEXTURE, .required = 1};
    VkImage handle, second;
    VmaAllocation allocation, other;
    RcAdmissionStatus status;
    RcBudgetSnapshot observed;
    start(0);
    native.reuse = 1;
    assert(image(request, 0, &handle, &allocation, &status) == VK_SUCCESS);
    resource_core_budget_snapshot(&observed);
    assert(observed.retained.amount[RC_BUDGET_GPU_FIRST] == 4 * MB);
    assert(!observed.unobserved.amount[RC_BUDGET_GPU_FIRST] && !native.new_calls);
    native.reuse = 0;
    assert(image(request, 0, &second, &other, &status) == VK_NOT_READY && status == RC_ADMISSION_REFUSED);
    assert(!second && !other && !native.new_calls);
    assert(resource_core_vma_retire(allocation) == RC_ADMISSION_OK);
    resource_core_budget_snapshot(&observed);
    assert(observed.pending.amount[RC_BUDGET_GPU_FIRST] == 4 * MB && !native.destroys);
    resource_core_vma_destroy_image(context.vma, handle, allocation);

    start(128);
    assert(image(request, 0, &handle, &allocation, &status) == VK_SUCCESS);
    assert(native.reuse_calls == 1 && native.new_calls == 1);
    resource_core_budget_snapshot(&observed);
    assert(observed.unobserved.amount[RC_BUDGET_GPU_FIRST] == 64 * MB);
    assert(observed.retained.amount[RC_BUDGET_GPU_FIRST] == 4 * MB);
    assert(observed.available.amount[RC_BUDGET_GPU_FIRST] == 64 * MB);
    resource_core_vma_destroy_image(context.vma, handle, allocation);
    resource_core_budget_snapshot(&observed);
    assert(observed.available.amount[RC_BUDGET_GPU_FIRST] == 64 * MB);
    clock_ns += 200000000;
    resource_core_budget_refresh();
    resource_core_budget_snapshot(&observed);
    assert(observed.available.amount[RC_BUDGET_GPU_FIRST] == 64 * MB); // driver kept its block

    start(256);
    native.need = native.expected_growth = 128 * MB;
    native.dedicated = 1;
    assert(image(request, 0, &handle, &allocation, &status) == VK_SUCCESS);
    assert(!native.reuse_calls && native.new_calls == 1);
    resource_core_vma_destroy_image(context.vma, handle, allocation);

    start(256);
    native.type = 1;
    native.expected_growth = 32 * MB;
    RcVmaRequest stage = {.producer = RC_BUDGET_STAGE, .required = 1, .pool_memory_type = 1, .pool_block_size = 32 * MB};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4 * MB};
    VmaAllocationCreateInfo ai = {.pool = (VmaPool)(uintptr_t)1};
    VkBuffer buffer;
    assert(resource_core_vma_create_buffer(&stage, context.vma, &bi, &ai, &buffer, &allocation, NULL, &status) == VK_SUCCESS);
    resource_core_budget_snapshot(&observed);
    assert(observed.unobserved.amount[RC_BUDGET_RAM] == 32 * MB);
    assert(observed.retained.amount[RC_BUDGET_STAGING] == 4 * MB);
    resource_core_vma_destroy_buffer(context.vma, buffer, allocation);

    start(128);
    VkMemoryRequirements raw = {native.need, 256, 1u << native.type};
    ai = (VmaAllocationCreateInfo){.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    assert(resource_core_vma_allocate_memory(&request, context.vma, &raw, &ai, &allocation, NULL, &status) == VK_SUCCESS);
    assert(native.raw_calls == 2 && native.reuse_calls == 1 && native.new_calls == 1);
    assert(resource_core_vma_retire(allocation) == RC_ADMISSION_OK);
    resource_core_vma_free_memory(context.vma, allocation);
    assert(native.raw_frees == 1 && native.destroys == 1);

    // Headroom uses the optional lane. Protected required capacity cannot be spent on a speculative block,
    // although probing an already compatible block remains legal and its temporary owner must settle on free.
    start(512);
    RcVmaRequest headroom = {.producer = RC_BUDGET_OTHER, .required = 0};
    raw = (VkMemoryRequirements){native.need, 256, 1u << native.type};
    assert(resource_core_vma_allocate_memory(&headroom, context.vma, &raw, &ai, &allocation, NULL, &status) == VK_NOT_READY);
    assert(status == RC_ADMISSION_REFUSED && !allocation && !native.new_calls);
    native.reuse = 1;
    assert(resource_core_vma_allocate_memory(&headroom, context.vma, &raw, &ai, &allocation, NULL, &status) == VK_SUCCESS);
    resource_core_vma_free_memory(context.vma, allocation);
    resource_core_budget_snapshot(&observed);
    assert(!observed.producer[RC_BUDGET_OTHER].retained.amount[RC_BUDGET_GPU_FIRST]);
    assert(!observed.unobserved.amount[RC_BUDGET_GPU_FIRST] && !native.new_calls);

    start(256);
    VmaPoolCreateInfo pi = {.memoryTypeIndex = 1, .blockSize = 64 * MB, .minBlockCount = 2};
    VmaPool pool;
    assert(resource_core_vma_create_pool(&stage, context.vma, &pi, &pool, &status) == VK_SUCCESS);
    resource_core_budget_snapshot(&observed);
    assert(observed.unobserved.amount[RC_BUDGET_GPU_FIRST + 1] == 128 * MB);
    assert(!observed.retained.amount[RC_BUDGET_STAGING]);

    start(128);
    native.fail = 1;
    assert(image(request, 0, &handle, &allocation, &status) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
    resource_core_budget_snapshot(&observed);
    assert(!handle && !allocation && !observed.reserved.amount[RC_BUDGET_COMMIT]);
    assert(observed.unobserved.amount[RC_BUDGET_COMMIT] == 64 * MB);
    caller_owned_staging_recycle();
    required_staging_burst();
    companion_peak();
    exact_placement_progress();
    preparation_children_use_one_peak();
    assert(!budget_lock_depth && !map_lock_depth);
    puts("resource_core_vma_check: passed");
    return 0;
}

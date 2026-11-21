#include <assert.h>
#include <stdio.h>
#define RC_BUDGET_TEST
#include "vk_mesh_fb.c"

#define HANDLE(type, n) ((type)(uintptr_t)(n))

#define GPU RC_BUDGET_GPU_FIRST
#define HOST_GPU (GPU + 1)
static VkCtx device;
static RcBudgetSample telemetry;
static uint64_t clock_ns = UINT64_C(1000000000);
typedef struct {
    void (*complete)(void *);
    void (*cancel)(void *, int);
    void *arg;
} Result;
static Result result[16];
static int stopped, lock_depth, missing_map, missing_push, fail_scan, map_coherent;
static unsigned plan_calls, scan_calls, scan_destroys, buffer_attempts, fail_buffer, destroyed, idles, invalidates, barriers, copies;
static unsigned active_buffers, buffer_index;
static VkResult idle_result = VK_SUCCESS;
static uint32_t reports[MESHFB_RECS * 4 + CAPACITY_WORDS];
static unsigned capacity_copies, capacity_barriers, cap_invalidates, raw_copies, normalize_calls, scan_dispatches;
static uint32_t raw_fill;
static VkResult invalidation_result;
static MeshFbScanParams captured_params;
static RcBudgetTicket owned[MFB_BUFFERS];
static VkBufferCopy report_copy;
volatile int32_t vk_lost;

VkCtx *vk_ctx(void) { return &device; }
int vk_sync_stopping(void) { return stopped; }
void vk_device_lost_check(VkResult r, const char *where) {
    assert(r == VK_ERROR_DEVICE_LOST && where);
    vk_lost = 1;
}
void vk_completion_add_result_owned(uint64_t scene, void (*complete)(void *), void (*cancel)(void *, int), void *arg) {
    assert(scene < 16 && !result[scene].arg);
    result[scene] = (Result){complete, cancel, arg};
}
VkResult vk_device_wait_idle(void) { idles++; return idle_result; }
int cfg_int(const char *name) { assert(!strcmp(name, "NX2NAVI_MESH_FALLBACK_MB")); return 128; }
void rt_log(const char *fmt, ...) { assert(fmt); }
void rt_fatal(const char *fmt, ...) { fprintf(stderr,"%s\n",fmt); abort(); }
uint64_t rc_budget_test_now_ns(void) { return clock_ns; }
void rc_budget_test_lock(void) { assert(!lock_depth++); }
void rc_budget_test_unlock(void) { assert(lock_depth-- == 1); }
void rc_budget_test_sample(RcBudgetSample *out) { assert(lock_depth == 1); *out = telemetry; }
void rc_budget_test_wait(uint32_t ms) { (void)ms; abort(); }
#include "resource_core_budget.c"

static VKAPI_ATTR void VKAPI_CALL mock_push(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout layout,
                                          uint32_t set, uint32_t n, const VkWriteDescriptorSet *w) {
    (void)cb; (void)bp; (void)layout; (void)set; (void)n; (void)w; abort();
}
static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_proc(VkDevice d, const char *name) {
    assert(d == device.device && !strcmp(name, "vkCmdPushDescriptorSetKHR") && !lock_depth);
    return missing_push ? NULL : (PFN_vkVoidFunction)mock_push;
}
PFN_vkGetDeviceProcAddr vk_dpa = get_proc;
static void assert_native_permit(void) {
    RcBudgetSnapshot ledger;
    assert(!lock_depth);
    resource_core_budget_snapshot(&ledger);
    assert(ledger.producer[RC_BUDGET_PIPELINE].reserved.amount[RC_BUDGET_RAM] == 64 * MIB);
    assert(!ledger.producer[RC_BUDGET_BUFFER].reserved.amount[GPU]);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorSetLayout(VkDevice d, const VkDescriptorSetLayoutCreateInfo *ci,
        const VkAllocationCallbacks *a, VkDescriptorSetLayout *out) {
    assert_native_permit();
    assert(d == device.device && ci->bindingCount == 4 && !a);
    scan_calls++;
    *out = HANDLE(VkDescriptorSetLayout, 1);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineLayout(VkDevice d, const VkPipelineLayoutCreateInfo *ci,
        const VkAllocationCallbacks *a, VkPipelineLayout *out) {
    assert_native_permit();
    assert(d == device.device && ci->setLayoutCount == 1 && ci->pPushConstantRanges->size == sizeof(MeshFbScanParams) && !a);
    *out = HANDLE(VkPipelineLayout, 2);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateShaderModule(VkDevice d, const VkShaderModuleCreateInfo *ci,
        const VkAllocationCallbacks *a, VkShaderModule *out) {
    assert_native_permit();
    assert(d == device.device && ci->codeSize && !a);
    *out = HANDLE(VkShaderModule, 3);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateComputePipelines(VkDevice d, VkPipelineCache cache, uint32_t n,
        const VkComputePipelineCreateInfo *ci, const VkAllocationCallbacks *a, VkPipeline *out) {
    assert_native_permit();
    assert(d == device.device && !cache && n == 1 && ci->stage.module == HANDLE(VkShaderModule, 3) && !a);
    if (fail_scan) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    *out = HANDLE(VkPipeline, 4);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyShaderModule(VkDevice d, VkShaderModule m, const VkAllocationCallbacks *a) {
    assert(d == device.device && m == HANDLE(VkShaderModule, 3) && !a);
}
VKAPI_ATTR void VKAPI_CALL vkDestroyPipeline(VkDevice d, VkPipeline p, const VkAllocationCallbacks *a) {
    assert(d == device.device && p == HANDLE(VkPipeline, 4) && !a);
    scan_destroys++;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyPipelineLayout(VkDevice d, VkPipelineLayout l, const VkAllocationCallbacks *a) {
    assert(d == device.device && l == HANDLE(VkPipelineLayout, 2) && !a);
}
VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorSetLayout(VkDevice d, VkDescriptorSetLayout s, const VkAllocationCallbacks *a) {
    assert(d == device.device && s == HANDLE(VkDescriptorSetLayout, 1) && !a);
}

static RcAdmissionVector exact_cost(const VkBufferCreateInfo *ci, int host) {
    RcAdmissionVector cost = {{0}};
    cost.amount[host ? HOST_GPU : GPU] = ci->size;
    if (host) cost.amount[RC_BUDGET_RAM] = cost.amount[RC_BUDGET_COMMIT] = ci->size;
    return cost;
}
VkResult resource_core_vma_buffer_peak(const RcVmaRequest *request, const VkBufferCreateInfo *ci,
    const VmaAllocationCreateInfo *allocation, VmaAllocationCreateInfo *placement, RcAdmissionVector *peak) {
    assert(request->required && request->producer == RC_BUDGET_BUFFER && !request->peak_ticket && !lock_depth);
    assert(allocation->flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    uint64_t stats = MESHFB_RECS * STAT_BYTES, atom = device.props.limits.nonCoherentAtomSize;
    const uint64_t sizes[] = {16 * MIB, 112 * MIB, 112 * MIB / 128, CAPACITY_REC_AT + CAPACITY_BYTES,
                              ((stats + atom - 1) / atom) * atom + CAPACITY_BYTES};
    unsigned i = plan_calls++ % MFB_BUFFERS;
    assert(ci->size == sizes[i]);
    assert((i == MFB_STAT) == !!(allocation->flags & VMA_ALLOCATION_CREATE_MAPPED_BIT));
    *placement = *allocation;
    placement->memoryTypeBits = i == MFB_STAT ? 2 : 1;
    *peak = exact_cost(ci, i == MFB_STAT);
    return VK_SUCCESS;
}
VkResult resource_core_vma_create_buffer(const RcVmaRequest *request, VmaAllocator allocator,
    const VkBufferCreateInfo *ci, const VmaAllocationCreateInfo *placement, VkBuffer *out, VmaAllocation *memory,
    VmaAllocationInfo *info, RcAdmissionStatus *admission) {
    assert(allocator == device.vma && request->peak_ticket && request->dependency && !lock_depth);
    RcBudgetSnapshot ledger;
    resource_core_budget_snapshot(&ledger);
    assert(!ledger.producer[RC_BUDGET_PIPELINE].reserved.amount[RC_BUDGET_RAM]);
    unsigned i = buffer_index++;
    assert(i < MFB_BUFFERS && placement->memoryTypeBits == (i == MFB_STAT ? 2u : 1u));
    buffer_attempts++;
    if (fail_buffer == i + 1) {
        *admission = RC_ADMISSION_NO_RECORDS;
        *request->dependency = (RcVmaDependency){.owner_full = 1};
        return VK_NOT_READY;
    }
    RcAdmissionVector cost = exact_cost(ci, i == MFB_STAT);
    assert(resource_core_budget_split(request->peak_ticket, &cost, request->producer, &owned[i]) == RC_ADMISSION_OK);
    assert(resource_core_budget_materialize(&owned[i], &cost) == RC_ADMISSION_OK);
    *admission = RC_ADMISSION_OK;
    *out = HANDLE(VkBuffer, 10 + i);
    *memory = HANDLE(VmaAllocation, 20 + i);
    *info = (VmaAllocationInfo){.memoryType = i == MFB_STAT, .size = ci->size,
                                 .pMappedData = i == MFB_STAT && !missing_map ? reports : NULL};
    if (i == MFB_STAT) assert(device.mem_props.memoryTypes[1].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    active_buffers++;
    return VK_SUCCESS;
}
void resource_core_vma_destroy_buffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation memory) {
    unsigned i = (unsigned)(uintptr_t)buffer - 10;
    assert(i < MFB_BUFFERS && allocator == device.vma && memory == HANDLE(VmaAllocation,20 + i));
    assert(resource_core_budget_retire(&owned[i]) == RC_ADMISSION_OK);
    assert(resource_core_budget_destroyed(&owned[i]) == RC_ADMISSION_OK);
    active_buffers--;
    destroyed++;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaInvalidateAllocation(VmaAllocator allocator, VmaAllocation alloc,
                                                          VkDeviceSize at, VkDeviceSize size) {
    assert(allocator == device.vma && alloc == HANDLE(VmaAllocation,20 + MFB_STAT) && !map_coherent);
    if (size == CAPACITY_BYTES) {
        assert(at == M.storage.capacity_stat_at && !(at % device.props.limits.nonCoherentAtomSize));
        cap_invalidates++;
        return invalidation_result;
    }
    assert(at % STAT_BYTES == 0 && size == STAT_BYTES);
    invalidates++;
    return VK_SUCCESS;
}
void rc_vkCmdCopyBuffer(VkCommandBuffer cb, VkBuffer src, VkBuffer dst, uint32_t n, const VkBufferCopy *r) {
    assert(cb && n == 1);
    if (src == HANDLE(VkBuffer, 99)) {
        assert(dst == M.storage.buffer[MFB_REC] && r->srcOffset == 24 && r->dstOffset % REC_BYTES == REQUEST_AT && r->size == 4);
        raw_copies++;
        return;
    }
    assert(src == M.storage.buffer[MFB_REC] && dst == M.storage.buffer[MFB_STAT]);
    if (r->srcOffset == CAPACITY_REC_AT) {
        assert(r->dstOffset == M.storage.capacity_stat_at && r->size == CAPACITY_BYTES);
        capacity_copies++;
        return;
    }
    assert(r->srcOffset % REC_BYTES == STAT_AT && r->dstOffset == r->srcOffset / REC_BYTES * STAT_BYTES && r->size == STAT_BYTES);
    report_copy = *r;
    copies++;
}
void rc_vkCmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo *d) {
    assert(cb && d->bufferMemoryBarrierCount == 1);
    const VkBufferMemoryBarrier2 *b = d->pBufferMemoryBarriers;
    if (b->size == CAPACITY_BYTES) {
        assert(b->srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED && b->dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
        if (b->buffer == M.storage.buffer[MFB_REC]) {
            assert(capacity_barriers == 2 * capacity_copies && b->offset == CAPACITY_REC_AT);
            assert(b->srcStageMask == (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT) &&
                   b->srcAccessMask == (VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT) &&
                   b->dstStageMask == VK_PIPELINE_STAGE_2_COPY_BIT && b->dstAccessMask == VK_ACCESS_2_TRANSFER_READ_BIT);
        } else {
            assert(capacity_barriers + 1 == 2 * capacity_copies);
            assert(b->buffer == M.storage.buffer[MFB_STAT] && b->offset == M.storage.capacity_stat_at &&
                   b->srcStageMask == VK_PIPELINE_STAGE_2_COPY_BIT && b->srcAccessMask == VK_ACCESS_2_TRANSFER_WRITE_BIT &&
                   b->dstStageMask == VK_PIPELINE_STAGE_2_HOST_BIT && b->dstAccessMask == VK_ACCESS_2_HOST_READ_BIT);
        }
        capacity_barriers++;
        return;
    }
    assert(copies == barriers + 1);
    assert(b->srcStageMask == VK_PIPELINE_STAGE_2_COPY_BIT && b->srcAccessMask == VK_ACCESS_2_TRANSFER_WRITE_BIT);
    assert(b->dstStageMask == VK_PIPELINE_STAGE_2_HOST_BIT && b->dstAccessMask == VK_ACCESS_2_HOST_READ_BIT);
    assert(b->buffer == M.storage.buffer[MFB_STAT] && b->offset == report_copy.dstOffset && b->size == report_copy.size);
    barriers++;
}
void rc_vkCmdFillBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkDeviceSize n, uint32_t v) {
    assert(cb && b == M.storage.buffer[MFB_REC] && !(o % 4));
    if (o == CAPACITY_REC_AT) assert(n == CAPACITY_BYTES && !v);
    else if (o % REC_BYTES == REQUEST_AT) {
        assert(n == sizeof(uint32_t));
        raw_fill = v;
    } else assert(o % REC_BYTES == 0 && n == REQUEST_AT && !v);
}
void rc_vkCmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline pipe) {
    assert(cb && bp == VK_PIPELINE_BIND_POINT_COMPUTE && pipe == M.scan.pipeline);
}
void rc_vkCmdPushConstants(VkCommandBuffer cb, VkPipelineLayout layout, VkShaderStageFlags stage,
                           uint32_t at, uint32_t bytes, const void *data) {
    assert(cb && layout == M.scan.layout && stage == VK_SHADER_STAGE_COMPUTE_BIT && !at && bytes == sizeof captured_params);
    memcpy(&captured_params, data, bytes);
}
void rc_vkCmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    assert(cb && x == 1 && y == 1 && z == 1);
    scan_dispatches++;
}
void rec_push(PFN_vkCmdPushDescriptorSetKHR fn, VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout layout,
              uint32_t set, uint32_t n, const VkWriteDescriptorSet *w) {
    assert(fn == M.scan.push && cb && bp == VK_PIPELINE_BIND_POINT_COMPUTE && layout == M.scan.layout && !set && n == 4);
    assert(w[0].pBufferInfo->buffer == M.storage.buffer[MFB_REC] && w[0].pBufferInfo->range == VK_WHOLE_SIZE &&
           w[1].pBufferInfo->range == M.storage.task_bytes && w[2].pBufferInfo->range == M.storage.list_bytes &&
           w[3].pBufferInfo->range == M.storage.mesh_bytes);
    normalize_calls++;
}

static void refresh(void) { clock_ns += UINT64_C(1000000000); resource_core_budget_refresh(); }
static void reset(void) {
    assert(vk_meshfb_shutdown() == VK_SUCCESS && !active_buffers);
    resource_core_budget_test_reset();
    stopped = vk_lost = missing_map = missing_push = fail_scan = map_coherent = 0;
    plan_calls = scan_calls = scan_destroys = buffer_attempts = fail_buffer = destroyed = idles = copies = barriers = invalidates = buffer_index = 0;
    memset(reports, 0, sizeof reports);
    telemetry = (RcBudgetSample){.known_mask = (1u << (HOST_GPU + 1)) - 1};
    for (uint32_t i = 0; i <= HOST_GPU; i++) {
        telemetry.total.amount[i] = 8192 * MIB;
        telemetry.free.amount[i] = 4096 * MIB;
    }
    device = (VkCtx){.device = HANDLE(VkDevice,1), .vma = HANDLE(VmaAllocator,2)};
    device.mem_props.memoryTypeCount = 2;
    device.mem_props.memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    device.props.limits.maxStorageBufferRange = 128 * MIB;
    device.props.limits.nonCoherentAtomSize = 16;
    device.props.limits.maxComputeWorkGroupCount[0] = 65535;
    capacity_copies = capacity_barriers = cap_invalidates = raw_copies = normalize_calls = scan_dispatches = raw_fill = 0;
    invalidation_result = VK_SUCCESS;
    idle_result = VK_SUCCESS;
    memset(result, 0, sizeof result);
    refresh();
}

static void joint_and_stats(void) {
    reset();
    VkMeshFbFailure failure;
    telemetry.free.amount[RC_BUDGET_RAM] = 144 * MIB;
    telemetry.free.amount[GPU] = 192 * MIB;
    refresh();
    assert(vk_meshfb_prepare(0,&failure) == VK_MESHFB_PENDING && !plan_calls && !scan_calls);
    assert(vk_meshfb_prepare(1,&failure) == VK_MESHFB_PENDING && plan_calls == 5 && !scan_calls && !buffer_attempts);
    assert(failure.admission == RC_ADMISSION_REFUSED && failure.peak.amount[RC_BUDGET_RAM] == 64 * MIB);
    telemetry.free.amount[RC_BUDGET_RAM] = 4096 * MIB;
    refresh();
    assert(vk_meshfb_prepare(1,&failure) == VK_MESHFB_PENDING && plan_calls == 5 && scan_calls == 1 && !buffer_attempts);
    assert(failure.before.available.amount[GPU] == 128 * MIB && failure.peak.amount[GPU] == 128 * MIB + 112 * MIB / 128 + CAPACITY_REC_AT + CAPACITY_BYTES);
    assert(failure.peak.amount[HOST_GPU] == MESHFB_RECS * STAT_BYTES + CAPACITY_BYTES);
    telemetry.free.amount[GPU] = 4096 * MIB;
    refresh();
    fail_buffer = 3;
    assert(vk_meshfb_prepare(1,&failure) == VK_MESHFB_PENDING && !active_buffers && destroyed == 2);
    assert(failure.admission == RC_ADMISSION_NO_RECORDS && failure.child.owner_full && scan_calls == 1 && !scan_destroys);
    assert(vk_meshfb_prepare(0,&failure) == VK_MESHFB_PENDING && scan_calls == 1);
    refresh();
    fail_buffer = buffer_index = 0;
    assert(vk_meshfb_prepare(1,&failure) == VK_MESHFB_READY && plan_calls == 5 && active_buffers == 5);
    assert(vk_meshfb_prepare(0,&failure) == VK_MESHFB_READY && scan_calls == 1);
    assert(vk_meshfb_mesh_words() == 112 * MIB / 4);
    assert(vk_meshfb_buffer(MESHFB_CB_TASK).range == 16 * MIB);
    assert(vk_meshfb_buffer(MESHFB_CB_LIST).range == 112 * MIB / 128);
    VkCommandBuffer cb = HANDLE(VkCommandBuffer,99);
    VkBuffer rec;
    int cmds;
    VkMeshFbArguments args = {.scene = 1, .ordinal = 2, .slot_words = 512, .vertices = 3};
    assert(vk_meshfb_capacity_begin(cb, args.scene) == 1);
    assert(vk_meshfb_record(cb, VK_NULL_HANDLE, 0, 1, &args, &rec, &cmds) == 0);
    assert(vk_meshfb_note_for_scene(cb, 0, 1) == 2);
    assert(vk_meshfb_record(cb, VK_NULL_HANDLE, 0, 1, &args, &rec, &cmds) == REC_BYTES);
    assert(vk_meshfb_note_for_scene(cb, REC_BYTES, 2) == 2);
    reports[0] = UINT32_MAX;
    result[1].complete(result[1].arg);
    assert(!vk_meshfb_short() && !invalidates);  // The same noncoherent atom still receives scene 2 writes.
    result[2].complete(result[2].arg);
    assert(vk_meshfb_short() == 1 && invalidates == 2);
    assert(vk_meshfb_short() == 1 && invalidates == 2);
    assert(vk_meshfb_note_for_scene(cb, 0, 3) == 2);
    M.next = MESHFB_RECS;
    assert(vk_meshfb_record(cb, VK_NULL_HANDLE, 0, 1, &args, &rec, &cmds) == 0);
    assert(M.reports_lost == 1 && vk_meshfb_short() == 1 && invalidates == 2);
    assert(vk_meshfb_note_for_scene(cb, 0, 4) == 0 && !result[4].arg);
    result[3].cancel(result[3].arg, 0);
    assert(vk_meshfb_note_for_scene(cb, 0, 5) == 2);
    result[5].cancel(result[5].arg, 1);
    assert(vk_meshfb_note_for_scene(cb, 0, 6) == 0 && !result[6].arg);
    assert(vk_meshfb_note_for_scene(cb, REC_BYTES, 7) == 2);
    result[7].complete(result[7].arg);
    // A later completion leaves canceled or ambiguous copies unproved and the shared atom unsafe.
    assert(vk_meshfb_short() == 1 && invalidates == 2);
    idle_result = VK_TIMEOUT;
    assert(vk_meshfb_shutdown() == VK_TIMEOUT && active_buffers == 5 && !scan_destroys && idles == 1);
    vk_lost = 1;
    assert(vk_meshfb_shutdown() == VK_ERROR_DEVICE_LOST && active_buffers == 5 && idles == 1);
    vk_lost = 0;
    idle_result = VK_SUCCESS;
    assert(vk_meshfb_shutdown() == VK_SUCCESS && !active_buffers && scan_destroys == 1 && idles == 2);
    assert(vk_meshfb_shutdown() == VK_SUCCESS && idles == 2);
}

static void failures(void) {
    VkMeshFbFailure failure;
    reset();
    missing_map = 1;
    assert(vk_meshfb_prepare(1,&failure) == VK_MESHFB_TERMINAL && failure.error == VK_ERROR_MEMORY_MAP_FAILED);
    assert(destroyed == 5 && !active_buffers && scan_destroys == 1);
    assert(vk_meshfb_prepare(0,&failure) == VK_MESHFB_TERMINAL && failure.error == VK_ERROR_MEMORY_MAP_FAILED);
    reset();
    fail_scan = 1;
    assert(vk_meshfb_prepare(1,&failure) == VK_MESHFB_TERMINAL && failure.error == VK_ERROR_OUT_OF_DEVICE_MEMORY && !buffer_attempts);
    reset();
    missing_push = 1;
    assert(vk_meshfb_prepare(1,&failure) == VK_MESHFB_UNSUPPORTED && failure.error == VK_ERROR_FEATURE_NOT_PRESENT);
    assert(!scan_calls && !buffer_attempts);
    reset();
    stopped = 1;
    assert(vk_meshfb_prepare(1,&failure) == VK_MESHFB_STOPPED && !plan_calls && !scan_calls && !buffer_attempts);
}
static void required_capacity_receipt(void) {
    reset();
    device.props.limits.nonCoherentAtomSize = 65536;
    VkMeshFbFailure failure;
    VkMeshFbArguments direct = {.slot_words = 540};
    assert(!vk_meshfb_requested_limit(&direct));
    assert(vk_meshfb_prepare(1, &failure) == VK_MESHFB_READY);
    assert(vk_meshfb_requested_limit(&direct) == 112 * MIB / 4 / 540);
    direct.flags = VK_MESHFB_ARGUMENT_TASK;
    direct.task_words = 72;
    assert(vk_meshfb_requested_limit(&direct) == 16 * MIB / 4 / 72);
    device.props.limits.maxComputeWorkGroupCount[0] = 13;
    assert(vk_meshfb_requested_limit(&direct) == 13);
    direct.task_words = 1;
    assert(!vk_meshfb_requested_limit(&direct));
    device.props.limits.maxComputeWorkGroupCount[0] = 65535;
    assert(M.storage.capacity_stat_at == 65536 && M.storage.capacity_stat_at >= MESHFB_RECS * STAT_BYTES);
    VkCommandBuffer cb = HANDLE(VkCommandBuffer, 99);
    VkMeshFbCapacity outcome;
    assert(vk_meshfb_capacity_copy(cb, 1) == 0);
    assert(vk_meshfb_capacity_read(1, &outcome) == VK_MESHFB_CAPACITY_UNPROVEN);
    uint64_t scene = UINT64_C(0x8912345600000001);
    uint64_t ordinal = UINT64_C(0xabc000000000007);
    VkMeshFbArguments args = {.scene = scene, .ordinal = ordinal, .flags = VK_MESHFB_ARGUMENT_TASK,
                              .task_words = 72, .task_fan = 32, .slot_words = 540, .vertices = 6};
    assert(vk_meshfb_capacity_begin(cb, scene) == 1 && vk_meshfb_capacity_begin(cb, scene) == 0);
    assert(vk_meshfb_capacity_begin(cb, scene + 1) == -1 && vk_meshfb_capacity_copy(cb, scene + 1) == -1);
    VkBuffer buffer;
    int commands;
    VkDeviceSize at = vk_meshfb_record(cb, HANDLE(VkBuffer,99), 24, UINT32_MAX, &args, &buffer, &commands);
    assert(at == 0 && raw_copies == 1 && !raw_fill && commands == 2 && buffer == M.storage.buffer[MFB_REC]);
    vk_meshfb_normalize(cb, at, &args);
    assert(normalize_calls == 1 && scan_dispatches == 1 && captured_params.phase == SCAN_NORMALIZE &&
           captured_params.max_groups == 65535 && captured_params.task_words == 72 &&
           captured_params.flags == VK_MESHFB_ARGUMENT_TASK && captured_params.capacity == CAPACITY_REC_AT / 4 &&
           captured_params.scene_low == (uint32_t)scene && captured_params.scene_high == (uint32_t)(scene >> 32));
    vk_meshfb_scan(cb, at, &args);
    assert(normalize_calls == 2 && scan_dispatches == 2 && captured_params.phase == SCAN_ASSEMBLE &&
           captured_params.ordinal_high == (uint32_t)(ordinal >> 32));
    M.stat_owner[0] = STAT_PENDING;
    assert(vk_meshfb_note_for_scene(cb, at, scene) == 0);   // diagnostics were busy
    assert(vk_meshfb_capacity_copy(cb, scene) == 3 && capacity_copies == 1 && capacity_barriers == 2);
    assert(vk_meshfb_capacity_begin(cb, scene) == -1 && vk_meshfb_capacity_copy(cb, scene) == -1);
    assert(vk_meshfb_capacity_read(scene, &outcome) == VK_MESHFB_CAPACITY_UNPROVEN); // filled-only copy has no GPU cookie
    assert(M.capacity_scene == scene && M.capacity_copied && cap_invalidates == 1 && !invalidates);

    uint32_t *native = reports + M.storage.capacity_stat_at / 4;
    native[0] = (uint32_t)scene;
    native[1] = (uint32_t)(scene >> 32);
    native[2] = VK_MESHFB_CAPACITY_LIMIT_EXPANSION;
    native[3] = 37;
    native[4] = 9;
    native[5] = 1;
    native[6] = 65535;
    native[8] = (uint32_t)ordinal;
    native[9] = (uint32_t)(ordinal >> 32);
    invalidation_result = VK_ERROR_MEMORY_MAP_FAILED;
    assert(vk_meshfb_capacity_read(scene, &outcome) == VK_MESHFB_CAPACITY_UNPROVEN &&
           outcome.error == VK_ERROR_MEMORY_MAP_FAILED && M.capacity_scene == scene);
    invalidation_result = VK_SUCCESS;
    assert(vk_meshfb_capacity_read(scene, &outcome) == VK_MESHFB_CAPACITY_OVERFLOW);
    assert(outcome.scene == scene && outcome.ordinal == ordinal && outcome.requested_outer == 37 &&
           outcome.requested_mesh == UINT64_C(0x100000009) && outcome.available == 65535 &&
           outcome.limit == VK_MESHFB_CAPACITY_LIMIT_EXPANSION && cap_invalidates == 3 && !invalidates);
    assert(!M.capacity_scene && !M.capacity_copied && vk_meshfb_capacity_read(scene, &outcome) == VK_MESHFB_CAPACITY_UNPROVEN);

    args.scene = ++scene;
    args.flags = VK_MESHFB_ARGUMENT_PROBE_PARTIAL;
    assert(vk_meshfb_capacity_begin(cb, scene) == 1);
    assert(vk_meshfb_record(cb, VK_NULL_HANDLE, 0, UINT32_MAX, &args, &buffer, &commands) == REC_BYTES &&
           raw_fill == UINT32_MAX);
    assert(vk_meshfb_capacity_copy(cb, scene) == 3 && capacity_copies == 2);
    assert(vk_meshfb_capacity_read(scene, &outcome) == VK_MESHFB_CAPACITY_UNPROVEN); // a prior cookie is stale
    memset(native, 0, CAPACITY_BYTES);
    native[0] = (uint32_t)scene;
    native[1] = (uint32_t)(scene >> 32);
    assert(vk_meshfb_capacity_read(scene, &outcome) == VK_MESHFB_CAPACITY_READY && !outcome.limit);

    assert(vk_meshfb_capacity_begin(cb, scene + 1) == 1);
    assert(vk_meshfb_capacity_copy(cb, scene + 1) == 3);
    stopped = 1;
    assert(vk_meshfb_capacity_read(scene + 1, &outcome) == VK_MESHFB_CAPACITY_STOPPED &&
           M.capacity_scene == scene + 1);
    assert(vk_meshfb_shutdown() == VK_SUCCESS && !active_buffers && !M.capacity_scene);
}

int main(void) {
    joint_and_stats();
    failures();
    required_capacity_receipt();
    puts("mesh fallback joint VMA, optional reports, required capacity receipt and checked shutdown passed");
}

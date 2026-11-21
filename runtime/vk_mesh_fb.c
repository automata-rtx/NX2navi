// The mesh fallback's buffers and records. A task or mesh program runs as a compute dispatch that writes a storage
// buffer (mesh_fb.c), and the draw reads it:
//   task buffer   a task workgroup's payload at its index, payload-size apart
//   list          per mesh workgroup after a task stage: its task and its index among that task's (meshfb_scan.comp)
//   mesh buffer   a mesh workgroup's vertex lines and primitive area at its slot
//   records       per draw 64 bytes: the indirect arguments of its task dispatch, its mesh dispatch and its draw
// The three buffers are reused by every draw: a draw's dispatches write them and its vertex stage has read them
// before the next draw's dispatches run, with a barrier in between (vk_exec_draw.c mfb_draw), so their size is what
// the largest draw needs, not what a frame's draws need together. mesh_fallback_mb sets it: an eighth for the task
// buffer, the rest for the mesh buffer. Workgroups past either buffer's end write nothing and draw nothing (the
// modules check the buffer's length); the draws that happened to are counted and the log names the size they needed.
// A record is used by one draw and written again only after MESHFB_RECS further draws.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "mesh_fb.h"
#include "vk.h"
#include "vk_mesh_fb.h"
#include "meshfb_scan_spv.h"

enum { MESHFB_RECS = 4096, REC_BYTES = 64, STAT_BYTES = 8, STAT_AT = 48, REQUEST_AT = 56 };
enum { CAPACITY_WORDS = 10, CAPACITY_BYTES = CAPACITY_WORDS * sizeof(uint32_t) };
#define CAPACITY_REC_AT ((VkDeviceSize)MESHFB_RECS * REC_BYTES)
enum { SCAN_NORMALIZE, SCAN_ASSEMBLE };
typedef struct {
    uint32_t rec, capacity, flags, task_words, task_fan, slot_words, vertices, max_groups, phase;
    uint32_t ordinal_low, ordinal_high, scene_low, scene_high;
} MeshFbScanParams;
_Static_assert(sizeof(MeshFbScanParams) == 13 * sizeof(uint32_t), "mesh fallback scan push constants");
enum { MFB_TASK, MFB_MESH, MFB_LIST, MFB_REC, MFB_STAT, MFB_BUFFERS };
enum { STAT_IDLE, STAT_PENDING, STAT_COMPLETE, STAT_UNKNOWN };

// Same existing shared opaque allowance as one ordinary graphics/compute compile.
#define SCAN_COMPILER_BYTES (UINT64_C(64) << 20)

typedef struct {
    PFN_vkCmdPushDescriptorSetKHR push;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipeline;
} MeshFbScan;

typedef struct {
    VkBufferCreateInfo buffers[MFB_BUFFERS];
    VmaAllocationCreateInfo placements[MFB_BUFFERS];
    RcAdmissionVector peak;
    VkDeviceSize capacity_stat_at;
    uint64_t mb;
} MeshFbPlan;

typedef struct {
    VkBuffer buffer[MFB_BUFFERS];
    VmaAllocation allocation[MFB_BUFFERS];
    VkDeviceSize task_bytes, mesh_bytes, list_bytes;  // descriptor spans
    const uint32_t *stat_map;
    VkDeviceSize capacity_stat_at;
    uint32_t stat_atom_rows;  // zero for coherent memory, otherwise every neighbor that must be retired
} MeshFbStorage;

static struct {
    MeshFbPlan plan;
    MeshFbScan scan;
    MeshFbStorage storage;
    uint32_t cap_slots[MESHFB_RECS], slot_words[MESHFB_RECS], cap_tasks[MESHFB_RECS];
    unsigned stat_owner[MESHFB_RECS];  // completion can change this while the recorder owns the other fields
    uint8_t noted[MESHFB_RECS];
    uint32_t next;
    uint64_t draws, short_draws, worst_bytes, reports_lost;
    int stat_error_logged;
    uint64_t capacity_scene;
    int capacity_copied;
} M;

// The stop/drain owner serializes shutdown with all replay and preparation.
// Competing cold probes never start a second scan or allocation transaction.
static int installing;
static VkMeshFbResult published = VK_MESHFB_PENDING;
static VkMeshFbFailure terminal_failure;

static int meshfb_stopped(void) { return vk_sync_stopping() || vk_device_is_lost(); }

static VkMeshFbResult stopped_result(VkMeshFbFailure *failure) {
    failure->error = vk_device_is_lost() ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;
    failure->reason = "mesh fallback preparation stopped";
    return VK_MESHFB_STOPPED;
}

static VkMeshFbResult native_result(VkResult result, const char *reason, VkMeshFbFailure *failure) {
    failure->error = result;
    failure->reason = reason;
    if (result == VK_ERROR_DEVICE_LOST) return VK_MESHFB_STOPPED;
    if (result == VK_NOT_READY) return VK_MESHFB_PENDING;
    if (result == VK_ERROR_FEATURE_NOT_PRESENT) return VK_MESHFB_UNSUPPORTED;
    return VK_MESHFB_TERMINAL;
}

static VkMeshFbResult reserve_result(RcAdmissionStatus status, const char *reason, VkMeshFbFailure *failure) {
    failure->admission = status;
    VkResult result = status == RC_ADMISSION_REFUSED || status == RC_ADMISSION_NO_RECORDS ? VK_NOT_READY :
                      status == RC_ADMISSION_UNSUPPORTED ? VK_ERROR_FEATURE_NOT_PRESENT : VK_ERROR_OUT_OF_DEVICE_MEMORY;
    return native_result(result, reason, failure);
}

static void meshfb_scan_destroy(VkCtx *c, MeshFbScan *scan) {
    if (scan->pipeline) vkDestroyPipeline(c->device, scan->pipeline, NULL);
    if (scan->layout) vkDestroyPipelineLayout(c->device, scan->layout, NULL);
    if (scan->dsl) vkDestroyDescriptorSetLayout(c->device, scan->dsl, NULL);
    *scan = (MeshFbScan){0};
}

static VkResult meshfb_scan_create(VkCtx *c, MeshFbScan *scan) {
    VkDescriptorSetLayoutBinding bindings[4];
    for (uint32_t i = 0; i < 4; i++)
        bindings[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, .bindingCount = 4, .pBindings = bindings};
    VkResult result = vkCreateDescriptorSetLayout(c->device, &dci, NULL, &scan->dsl);
    if (result != VK_SUCCESS) return result;
    VkPushConstantRange range = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MeshFbScanParams)};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &scan->dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &range};
    result = vkCreatePipelineLayout(c->device, &li, NULL, &scan->layout);
    if (result != VK_SUCCESS) return result;
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof meshfb_scan_spv, .pCode = meshfb_scan_spv};
    VkShaderModule module = VK_NULL_HANDLE;
    result = vkCreateShaderModule(c->device, &mi, NULL, &module);
    if (result != VK_SUCCESS) return result;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = module, .pName = "main"}, .layout = scan->layout};
    result = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &scan->pipeline);
    vkDestroyShaderModule(c->device, module, NULL);
    return result;
}

static VkMeshFbResult prepare_scan(VkCtx *c, VkMeshFbFailure *failure) {
    MeshFbScan scan = {.push = (PFN_vkCmdPushDescriptorSetKHR)vk_dpa(c->device, "vkCmdPushDescriptorSetKHR")};
    if (!scan.push) return native_result(VK_ERROR_FEATURE_NOT_PRESENT, "mesh scan needs push descriptors", failure);
    failure->peak.amount[RC_BUDGET_RAM] = failure->peak.amount[RC_BUDGET_COMMIT] = SCAN_COMPILER_BYTES;
    resource_core_budget_readiness(1, &failure->before);
    RcBudgetTicket ticket = {0};
    RcAdmissionStatus admission = resource_core_budget_reserve(RC_BUDGET_PIPELINE, 1, &failure->peak, &ticket);
    if (admission != RC_ADMISSION_OK) return reserve_result(admission, "mesh scan compiler capacity", failure);
    if (meshfb_stopped()) {
        if (resource_core_budget_cancel(&ticket) != RC_ADMISSION_OK)
            rt_fatal("vk: mesh scan lost its unused compiler reservation");
        return stopped_result(failure);
    }
    VkResult result = meshfb_scan_create(c, &scan);
    if (resource_core_budget_opaque_done(&ticket) != RC_ADMISSION_OK)
        rt_fatal("vk: mesh scan lost its compiler reservation");
    if (result != VK_SUCCESS || meshfb_stopped()) {
        meshfb_scan_destroy(c, &scan);
        return result != VK_SUCCESS ? native_result(result, "creating the mesh scan pipeline", failure) : stopped_result(failure);
    }
    M.scan = scan;
    return VK_MESHFB_READY;
}

static void meshfb_storage_destroy(VkCtx *c, MeshFbStorage *storage) {
    for (int i = MFB_BUFFERS - 1; i >= 0; --i)
        if (storage->buffer[i]) resource_core_vma_destroy_buffer(c->vma, storage->buffer[i], storage->allocation[i]);
    *storage = (MeshFbStorage){0};
}

static int storage_peak_add(RcAdmissionVector *total, const RcAdmissionVector *part) {
    for (unsigned i = 0; i < RC_ADMISSION_MAX_DOMAINS; i++) {
        if (part->amount[i] > UINT64_MAX - total->amount[i]) return 0;
        total->amount[i] += part->amount[i];
    }
    return 1;
}

static VkMeshFbResult plan_storage(VkMeshFbFailure *failure) {
    uint64_t mb = (uint64_t)cfg_int("NX2NAVI_MESH_FALLBACK_MB");
    if (mb < 16) mb = 16;
    if (mb > 4096) mb = 4096;
    VkDeviceSize task = (mb << 20) / 8, mesh = (mb << 20) - task, list = mesh / 128;
    VkDeviceSize atom = vk_ctx()->props.limits.nonCoherentAtomSize;
    if (atom < sizeof(uint32_t)) atom = sizeof(uint32_t);
    VkDeviceSize stat_bytes = (VkDeviceSize)MESHFB_RECS * STAT_BYTES;
    if (atom % sizeof(uint32_t) || atom > UINT64_MAX - stat_bytes - CAPACITY_BYTES)
        return native_result(VK_ERROR_FEATURE_NOT_PRESENT, "mesh fallback capacity report alignment", failure);
    VkDeviceSize tail = stat_bytes % atom;
    VkDeviceSize capacity_stat_at = tail ? stat_bytes + atom - tail : stat_bytes;
    const VkBufferUsageFlags st = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkBufferCreateInfo buffers[MFB_BUFFERS] = {
        [MFB_TASK] = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = task, .usage = st},
        [MFB_MESH] = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = mesh, .usage = st},
        [MFB_LIST] = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = list, .usage = st},
        [MFB_REC] = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = CAPACITY_REC_AT + CAPACITY_BYTES,
            .usage = st | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT},
        [MFB_STAT] = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = capacity_stat_at + CAPACITY_BYTES,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT},
    };
    const VmaAllocationCreateInfo device = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT};
    const VmaAllocationCreateInfo mapped = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT};
    MeshFbPlan plan = {.mb = mb, .capacity_stat_at = capacity_stat_at};
    memcpy(plan.buffers, buffers, sizeof buffers);
    RcVmaRequest request = {.producer = RC_BUDGET_BUFFER, .required = 1};
    for (int i = 0; i < MFB_BUFFERS; i++) {
        RcAdmissionVector part;
        VkResult result = resource_core_vma_buffer_peak(&request, &plan.buffers[i], i == MFB_STAT ? &mapped : &device,
                                                        &plan.placements[i], &part);
        if (result != VK_SUCCESS) return native_result(result, "choosing pinned mesh fallback buffer placement", failure);
        if (!storage_peak_add(&plan.peak, &part))
            return native_result(VK_ERROR_OUT_OF_DEVICE_MEMORY, "mesh fallback buffer peak overflow", failure);
    }
    M.plan = plan;
    return VK_MESHFB_READY;
}

static VkMeshFbResult prepare_storage(VkCtx *c, const MeshFbPlan *plan, VkMeshFbFailure *failure) {
    *failure = (VkMeshFbFailure){.peak = plan->peak};
    RcVmaRequest request = {.producer = RC_BUDGET_BUFFER, .required = 1};
    resource_core_budget_readiness(1, &failure->before);
    RcBudgetTicket ticket = {0};
    RcAdmissionStatus admission = resource_core_budget_reserve(request.producer, request.required, &failure->peak, &ticket);
    if (admission != RC_ADMISSION_OK) return reserve_result(admission, "joint mesh fallback buffer capacity", failure);
    request.peak_ticket = &ticket;
    request.dependency = &failure->child;
    MeshFbStorage storage = {0};
    VkMeshFbResult outcome = VK_MESHFB_READY;
    for (int i = 0; i < MFB_BUFFERS; i++) {
        if (meshfb_stopped()) { outcome = stopped_result(failure); break; }
        VmaAllocationInfo info;
        VkResult result = resource_core_vma_create_buffer(&request, c->vma, &plan->buffers[i], &plan->placements[i],
                                                           &storage.buffer[i], &storage.allocation[i], &info, &failure->admission);
        if (result != VK_SUCCESS) { outcome = native_result(result, "creating a reserved mesh fallback buffer", failure); break; }
        if (i == MFB_STAT) {
            if (!info.pMappedData || info.memoryType >= c->mem_props.memoryTypeCount || info.offset) {
                outcome = native_result(VK_ERROR_MEMORY_MAP_FAILED, "mesh fallback statistics mapping unavailable", failure);
                break;
            }
            storage.stat_map = info.pMappedData;
            if (!(c->mem_props.memoryTypes[info.memoryType].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                VkDeviceSize rows = c->props.limits.nonCoherentAtomSize / STAT_BYTES;
                storage.stat_atom_rows = rows < 1 ? 1 : rows > MESHFB_RECS ? MESHFB_RECS : (uint32_t)rows;
            }
        }
    }
    if (resource_core_budget_cancel(&ticket) != RC_ADMISSION_OK)
        rt_fatal("vk: mesh fallback lost its joint buffer reservation");
    if (outcome == VK_MESHFB_READY && meshfb_stopped()) outcome = stopped_result(failure);
    if (outcome != VK_MESHFB_READY) {
        meshfb_storage_destroy(c, &storage);
        return outcome;
    }
    VkDeviceSize range = c->props.limits.maxStorageBufferRange & ~(VkDeviceSize)3;
    VkDeviceSize task = plan->buffers[MFB_TASK].size, mesh = plan->buffers[MFB_MESH].size;
    VkDeviceSize list = plan->buffers[MFB_LIST].size;
    storage.task_bytes = task < range ? task : range;
    storage.mesh_bytes = mesh < range ? mesh : range;
    storage.list_bytes = list < range ? list : range;
    storage.capacity_stat_at = plan->capacity_stat_at;
    M.storage = storage;   // No CPU initialization is needed. A report is read only after its own GPU copy.
    rt_log("vk: mesh fallback: %llu MiB for task payloads, %llu MiB for mesh output (mesh_fallback_mb = %llu)",
           (unsigned long long)(storage.task_bytes >> 20), (unsigned long long)(storage.mesh_bytes >> 20),
           (unsigned long long)plan->mb);
    return VK_MESHFB_READY;
}

VkMeshFbResult vk_meshfb_prepare(int allow_start, VkMeshFbFailure *failure) {
    VkMeshFbFailure unused;
    if (!failure) failure = &unused;
    *failure = (VkMeshFbFailure){.error = VK_NOT_READY};
    if (meshfb_stopped()) return stopped_result(failure);
    VkMeshFbResult state = __atomic_load_n(&published, __ATOMIC_ACQUIRE);
    if (state != VK_MESHFB_PENDING) {
        if (state == VK_MESHFB_READY) failure->error = VK_SUCCESS;
        else *failure = terminal_failure;
        return state;
    }
    if (!allow_start) return VK_MESHFB_PENDING;
    int expected = 0;
    if (!__atomic_compare_exchange_n(&installing, &expected, 1, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
        failure->reason = "another recorder owns mesh fallback preparation";
        return VK_MESHFB_PENDING;
    }
    state = __atomic_load_n(&published, __ATOMIC_ACQUIRE);
    if (state == VK_MESHFB_PENDING) {
        VkCtx *c = vk_ctx();
        if (!c) state = native_result(VK_ERROR_INITIALIZATION_FAILED, "no device for mesh fallback", failure);
        else if (meshfb_stopped()) state = stopped_result(failure);
        else {
            state = M.plan.mb ? VK_MESHFB_READY : plan_storage(failure);
            if (state == VK_MESHFB_READY && !M.scan.pipeline) state = prepare_scan(c, failure);
            if (state == VK_MESHFB_READY) state = prepare_storage(c, &M.plan, failure);
        }
        if (state == VK_MESHFB_TERMINAL || state == VK_MESHFB_UNSUPPORTED || state == VK_MESHFB_STOPPED) {
            if (c) meshfb_scan_destroy(c, &M.scan);   // No draw has acquired this generation.
            terminal_failure = *failure;
        }
        if (state != VK_MESHFB_PENDING) __atomic_store_n(&published, state, __ATOMIC_RELEASE);
    } else if (state != VK_MESHFB_READY) *failure = terminal_failure;
    if (state == VK_MESHFB_READY) *failure = (VkMeshFbFailure){.error = VK_SUCCESS};
    __atomic_store_n(&installing, 0, __ATOMIC_RELEASE);
    return state;
}

VkDescriptorBufferInfo vk_meshfb_buffer(int cb_slot) {
    if (__atomic_load_n(&published, __ATOMIC_ACQUIRE) != VK_MESHFB_READY) return (VkDescriptorBufferInfo){0};
    if (cb_slot == MESHFB_CB_TASK) return (VkDescriptorBufferInfo){M.storage.buffer[MFB_TASK], 0, M.storage.task_bytes};
    if (cb_slot == MESHFB_CB_LIST) return (VkDescriptorBufferInfo){M.storage.buffer[MFB_LIST], 0, M.storage.list_bytes};
    return (VkDescriptorBufferInfo){M.storage.buffer[MFB_MESH], 0, M.storage.mesh_bytes};
}

uint64_t vk_meshfb_mesh_words(void) {
    return __atomic_load_n(&published, __ATOMIC_ACQUIRE) == VK_MESHFB_READY ? M.storage.mesh_bytes / 4 : 0;
}

static unsigned stat_owner(uint32_t i) {
    return __atomic_load_n(&M.stat_owner[i], __ATOMIC_ACQUIRE);
}

static int stat_busy(unsigned owner) {
    return owner == STAT_PENDING || owner == STAT_UNKNOWN;
}

static void stat_finished(void *arg) {
    unsigned *owner = arg;
    __atomic_store_n(owner, STAT_COMPLETE, __ATOMIC_RELEASE);
}

static void stat_canceled(void *arg, int may_have_executed) {
    unsigned *owner = arg;
    __atomic_store_n(owner, may_have_executed ? STAT_UNKNOWN : STAT_IDLE, __ATOMIC_RELEASE);
}

static int stat_readable(uint32_t i) {
    if (stat_owner(i) != STAT_COMPLETE) return 0;
    uint32_t rows = M.storage.stat_atom_rows;
    if (!rows) return 1;
    uint32_t first = i / rows * rows;
    for (uint32_t j = first; j < first + rows; j++)
        if (stat_busy(stat_owner(j))) return 0;
    VkResult result = vmaInvalidateAllocation(vk_ctx()->vma, M.storage.allocation[MFB_STAT],
                                               (VkDeviceSize)i * STAT_BYTES, STAT_BYTES);
    if (result == VK_SUCCESS) return 1;
    if (!M.stat_error_logged) {
        M.stat_error_logged = 1;
        rt_log("vk: mesh fallback: statistics invalidation failed (%d)", (int)result);
    }
    return 0;
}

// The GPU record ring is queue-ordered. Host statistics have a separate readback proof.
static void record_check(uint32_t i) {
    if (!M.noted[i]) return;
    if (stat_owner(i) == STAT_IDLE) {
        M.noted[i] = 0;  // The copy was canceled before it could reach the native queue.
        return;
    }
    if (!stat_readable(i)) return;
    M.noted[i] = 0;
    uint32_t want = M.storage.stat_map[2 * i], tasks = M.storage.stat_map[2 * i + 1];
    if (want <= M.cap_slots[i] && tasks <= M.cap_tasks[i]) return;
    uint64_t need = (uint64_t)want * M.slot_words[i] * 4;
    if (need > M.worst_bytes) M.worst_bytes = need;
    uint64_t n = ++M.short_draws;
    if (n <= 4 || !(n & (n - 1))) {
        char from[96] = "";
        if (tasks) snprintf(from, sizeof from, " from %u task workgroups (room for %u)", tasks, M.cap_tasks[i]);
        rt_log("vk: mesh fallback: a draw's programs emitted %u mesh workgroups of %u bytes%s, and the mesh buffer holds %u: "
               "the full request exceeded the effective capacity (%llu such draws of %llu, largest reported payload %llu bytes)",
               want, M.slot_words[i] * 4, from, M.cap_slots[i], (unsigned long long)n,
               (unsigned long long)M.draws, (unsigned long long)M.worst_bytes);
    }
}

int vk_meshfb_capacity_begin(VkCommandBuffer cb, uint64_t scene) {
    if (!cb || !scene || meshfb_stopped() || __atomic_load_n(&published, __ATOMIC_ACQUIRE) != VK_MESHFB_READY)
        return -1;
    if (M.capacity_scene) return M.capacity_scene == scene && !M.capacity_copied ? 0 : -1;
    vkCmdFillBuffer(cb, M.storage.buffer[MFB_REC], CAPACITY_REC_AT, CAPACITY_BYTES, 0);
    M.capacity_scene = scene;
    return 1;
}

static int meshfb_arguments_valid(const VkMeshFbArguments *args) {
    return args && args->scene == M.capacity_scene && args->scene && args->ordinal && !M.capacity_copied &&
           !(args->flags & ~(VK_MESHFB_ARGUMENT_TASK | VK_MESHFB_ARGUMENT_PROBE_PARTIAL)) &&
           args->slot_words && args->vertices &&
           (!(args->flags & VK_MESHFB_ARGUMENT_TASK) || (args->task_words >= 2 && args->task_fan));
}

static uint32_t meshfb_dispatch_room(VkDeviceSize rows) {
    uint32_t limit = vk_ctx()->props.limits.maxComputeWorkGroupCount[0];
    return rows < limit ? (uint32_t)rows : limit;
}

static uint32_t meshfb_task_room(uint32_t task_words) {
    if (task_words < 2) return 0;
    uint32_t room = meshfb_dispatch_room(M.storage.task_bytes / 4 / task_words);
    uint32_t packed = UINT32_C(1) << MESHFB_LIST_TASK_BITS;
    return room < packed ? room : packed;
}

static uint32_t meshfb_mesh_room(uint32_t slot_words, int with_tasks) {
    if (!slot_words) return 0;
    uint32_t room = meshfb_dispatch_room(M.storage.mesh_bytes / 4 / slot_words);
    return with_tasks && room > M.storage.list_bytes / 4 ? (uint32_t)(M.storage.list_bytes / 4) : room;
}

uint32_t vk_meshfb_requested_limit(const VkMeshFbArguments *args) {
    if (!args || __atomic_load_n(&published, __ATOMIC_ACQUIRE) != VK_MESHFB_READY ||
        (args->flags & ~(VK_MESHFB_ARGUMENT_TASK | VK_MESHFB_ARGUMENT_PROBE_PARTIAL))) return 0;
    return args->flags & VK_MESHFB_ARGUMENT_TASK ? meshfb_task_room(args->task_words) :
                                                  meshfb_mesh_room(args->slot_words, 0);
}

VkDeviceSize vk_meshfb_record(VkCommandBuffer cb, VkBuffer src, VkDeviceSize src_off, uint32_t requested,
                              const VkMeshFbArguments *args, VkBuffer *rec, int *cmds) {
    if (__atomic_load_n(&published, __ATOMIC_ACQUIRE) != VK_MESHFB_READY || !meshfb_arguments_valid(args))
        rt_fatal("vk: recording a transformed mesh without its open scene or valid arguments");
    uint32_t i = M.next++ % MESHFB_RECS;
    VkDeviceSize at = (VkDeviceSize)i * REC_BYTES;
    if (M.noted[i]) {
        record_check(i);
        if (M.noted[i]) {
            M.noted[i] = 0;
            M.reports_lost++;  // GPU reuse stays ordered even if the CPU cannot read the old diagnostic report.
        }
    }
    M.draws++;
    M.slot_words[i] = args->slot_words;
    int with_tasks = (args->flags & VK_MESHFB_ARGUMENT_TASK) != 0;
    M.cap_slots[i] = meshfb_mesh_room(args->slot_words, with_tasks);
    M.cap_tasks[i] = with_tasks ? meshfb_task_room(args->task_words) : 0;
    *rec = M.storage.buffer[MFB_REC];
    // Until normalization, all indirect commands have zero groups/instances. Do not overlap the raw write.
    vkCmdFillBuffer(cb, *rec, at, REQUEST_AT, 0);
    if (src) {
        VkBufferCopy cp = {src_off, at + REQUEST_AT, sizeof(uint32_t)};
        vkCmdCopyBuffer(cb, src, *rec, 1, &cp);
    } else {
        vkCmdFillBuffer(cb, *rec, at + REQUEST_AT, sizeof(uint32_t), requested);
    }
    *cmds = 2;
    return at;
}

static void meshfb_scan_phase(VkCommandBuffer cb, VkDeviceSize at, const VkMeshFbArguments *args, uint32_t phase) {
    if (!meshfb_arguments_valid(args) || at % REC_BYTES || at >= CAPACITY_REC_AT ||
        (phase == SCAN_ASSEMBLE && !(args->flags & VK_MESHFB_ARGUMENT_TASK)))
        rt_fatal("vk: invalid transformed mesh scan phase");
    VkDescriptorBufferInfo bi[4] = {
        {M.storage.buffer[MFB_REC], 0, VK_WHOLE_SIZE}, {M.storage.buffer[MFB_TASK], 0, M.storage.task_bytes},
        {M.storage.buffer[MFB_LIST], 0, M.storage.list_bytes}, {M.storage.buffer[MFB_MESH], 0, M.storage.mesh_bytes}
    };
    VkWriteDescriptorSet w[4];
    for (int k = 0; k < 4; k++)
        w[k] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = (uint32_t)k,
                                      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      .pBufferInfo = &bi[k]};
    MeshFbScanParams pc = {.rec = (uint32_t)(at / 4), .capacity = (uint32_t)(CAPACITY_REC_AT / 4),
        .flags = args->flags, .task_words = args->task_words, .task_fan = args->task_fan,
        .slot_words = args->slot_words, .vertices = args->vertices,
        .max_groups = vk_ctx()->props.limits.maxComputeWorkGroupCount[0], .phase = phase,
        .ordinal_low = (uint32_t)args->ordinal, .ordinal_high = (uint32_t)(args->ordinal >> 32),
        .scene_low = (uint32_t)args->scene, .scene_high = (uint32_t)(args->scene >> 32)};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, M.scan.pipeline);
    rec_push(M.scan.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, M.scan.layout, 0, 4, w);
    vkCmdPushConstants(cb, M.scan.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(cb, 1, 1, 1);
}

void vk_meshfb_normalize(VkCommandBuffer cb, VkDeviceSize at, const VkMeshFbArguments *args) {
    meshfb_scan_phase(cb, at, args, SCAN_NORMALIZE);
}
void vk_meshfb_scan(VkCommandBuffer cb, VkDeviceSize at, const VkMeshFbArguments *args) {
    meshfb_scan_phase(cb, at, args, SCAN_ASSEMBLE);
}

static void meshfb_buffer_dependency(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                                     VkPipelineStageFlags2 source, VkAccessFlags2 source_access,
                                     VkPipelineStageFlags2 target, VkAccessFlags2 target_access) {
    VkBufferMemoryBarrier2 barrier = {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = source, .srcAccessMask = source_access, .dstStageMask = target, .dstAccessMask = target_access,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buffer, .offset = offset, .size = size};
    VkDependencyInfo dependency = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &barrier};
    vkCmdPipelineBarrier2(cb, &dependency);
}

int vk_meshfb_capacity_copy(VkCommandBuffer cb, uint64_t scene) {
    if (!M.capacity_scene) return 0;
    if (!cb || M.capacity_scene != scene || M.capacity_copied || meshfb_stopped()) return -1;
    VkBuffer source = M.storage.buffer[MFB_REC], target = M.storage.buffer[MFB_STAT];
    meshfb_buffer_dependency(cb, source, CAPACITY_REC_AT, CAPACITY_BYTES,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
        VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferCopy copy = {CAPACITY_REC_AT, M.storage.capacity_stat_at, CAPACITY_BYTES};
    vkCmdCopyBuffer(cb, source, target, 1, &copy);
    meshfb_buffer_dependency(cb, target, copy.dstOffset, copy.size, VK_PIPELINE_STAGE_2_COPY_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    M.capacity_copied = 1;
    return 3;
}

static uint64_t meshfb_u64(const uint32_t *words) { return (uint64_t)words[1] << 32 | words[0]; }

VkMeshFbCapacityResult vk_meshfb_capacity_read(uint64_t scene, VkMeshFbCapacity *out) {
    *out = (VkMeshFbCapacity){.scene = scene, .error = VK_SUCCESS};
    if (meshfb_stopped()) {
        if (vk_device_is_lost()) out->error = VK_ERROR_DEVICE_LOST;
        return VK_MESHFB_CAPACITY_STOPPED;
    }
    if (!scene || M.capacity_scene != scene || !M.capacity_copied) {
        out->error = VK_ERROR_INITIALIZATION_FAILED;
        return VK_MESHFB_CAPACITY_UNPROVEN;
    }
    if (M.storage.stat_atom_rows) {
        out->error = vmaInvalidateAllocation(vk_ctx()->vma, M.storage.allocation[MFB_STAT],
                                             M.storage.capacity_stat_at, CAPACITY_BYTES);
        if (out->error == VK_ERROR_DEVICE_LOST) {
            vk_device_lost_check(out->error, "reading transformed mesh scene capacity");
            return VK_MESHFB_CAPACITY_STOPPED;
        }
        if (out->error != VK_SUCCESS) return VK_MESHFB_CAPACITY_UNPROVEN;
    }
    uint32_t report[CAPACITY_WORDS];
    const uint8_t *mapped = (const uint8_t *)M.storage.stat_map;
    memcpy(report, mapped + M.storage.capacity_stat_at, sizeof report);
    uint64_t ordinal = meshfb_u64(report + 8);
    if (meshfb_u64(report) != scene || report[2] > VK_MESHFB_CAPACITY_LIMIT_EXPANSION || (!!report[2] != !!ordinal)) {
        out->error = VK_ERROR_INITIALIZATION_FAILED;
        return VK_MESHFB_CAPACITY_UNPROVEN;
    }
    out->limit = (VkMeshFbCapacityLimit)report[2];
    out->requested_outer = report[3];
    out->requested_mesh = meshfb_u64(report + 4);
    out->available = report[6];
    out->detail = report[7];
    out->ordinal = ordinal;
    M.capacity_scene = 0;
    M.capacity_copied = 0;
    return out->limit ? VK_MESHFB_CAPACITY_OVERFLOW : VK_MESHFB_CAPACITY_READY;
}


int vk_meshfb_note_for_scene(VkCommandBuffer cb, VkDeviceSize at, uint64_t scene) {
    uint32_t i = (uint32_t)(at / REC_BYTES);
    if (!scene || i >= MESHFB_RECS) rt_fatal("vk: invalid mesh fallback readback");
    if (stat_busy(stat_owner(i))) {
        M.reports_lost++;
        return 0;
    }
    VkBufferCopy cp = {at + STAT_AT, (VkDeviceSize)i * STAT_BYTES, STAT_BYTES};
    vkCmdCopyBuffer(cb, M.storage.buffer[MFB_REC], M.storage.buffer[MFB_STAT], 1, &cp);
    VkBufferMemoryBarrier2 visible = {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = M.storage.buffer[MFB_STAT], .offset = cp.dstOffset, .size = cp.size};
    VkDependencyInfo dependency = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &visible};
    vkCmdPipelineBarrier2(cb, &dependency);
    __atomic_store_n(&M.stat_owner[i], STAT_PENDING, __ATOMIC_RELEASE);
    M.noted[i] = 1;
    // A discarded recording never publishes a result, even if a later signal passes this scene.
    vk_completion_add_result_owned(scene, stat_finished, stat_canceled, &M.stat_owner[i]);
    return 2;
}

uint64_t vk_meshfb_short(void) {
    if (__atomic_load_n(&published, __ATOMIC_ACQUIRE) != VK_MESHFB_READY || vk_device_is_lost()) return 0;
    for (uint32_t i = 0; i < MESHFB_RECS; i++) record_check(i);
    return M.short_draws;
}

unsigned vk_meshfb_probe_pending_reports(void) {
    unsigned pending = 0;
    for (uint32_t i = 0; i < MESHFB_RECS; i++) pending += stat_owner(i) == STAT_PENDING;
    return pending;
}

VkResult vk_meshfb_shutdown(void) {
    if (vk_device_is_lost()) return VK_ERROR_DEVICE_LOST;
    if (__atomic_load_n(&installing, __ATOMIC_ACQUIRE)) return VK_NOT_READY;
    if (!M.scan.pipeline) {
        M.plan = (MeshFbPlan){0};
        __atomic_store_n(&published, VK_MESHFB_PENDING, __ATOMIC_RELEASE);
        return VK_SUCCESS;
    }
    VkCtx *c = vk_ctx();
    if (!c) return VK_ERROR_INITIALIZATION_FAILED;
    int ready = __atomic_load_n(&published, __ATOMIC_ACQUIRE) == VK_MESHFB_READY;
    if (ready) {
        VkResult idle = vk_device_wait_idle();
        if (idle != VK_SUCCESS) return idle;
        if (vk_device_is_lost()) return VK_ERROR_DEVICE_LOST;
        if (vk_meshfb_short())
            rt_log("vk: mesh fallback: %llu of %llu reported requests exceeded effective capacity (largest mesh payload %llu bytes)",
                   (unsigned long long)M.short_draws, (unsigned long long)M.draws, (unsigned long long)M.worst_bytes);
        uint64_t omitted = M.reports_lost;
        for (uint32_t i = 0; i < MESHFB_RECS; i++) omitted += M.noted[i] != 0;
        if (omitted) rt_log("vk: mesh fallback: %llu statistics reports retired without a completed readback",
                            (unsigned long long)omitted);
    }
    meshfb_scan_destroy(c, &M.scan);
    meshfb_storage_destroy(c, &M.storage);
    memset(&M, 0, sizeof M);
    __atomic_store_n(&published, VK_MESHFB_PENDING, __ATOMIC_RELEASE);
    return VK_SUCCESS;
}

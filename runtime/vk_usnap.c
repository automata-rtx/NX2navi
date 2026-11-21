// Uniform blocks as of the submit call. NVN starts the GPU when nvnQueueSubmitCommands returns and the GPU reads the
// constants soon after; here the replay records for several milliseconds, and the GPU reads a pool's bytes when it
// runs the part that uses them (the VRAM mirror's sync copies run then, vk_memory.c "VRAM mirror"). Meanwhile the game,
// having handed the frame over, rewrites per-object uniform blocks on other threads with no fence: late draws read
// another object's matrix (phantom rocks, a flat far field; NX2navi 1.2.1's producer f_70008c6460, 14-24 ms after the
// submit). So at the start of a submission's recording every uniform range it binds is copied into an arena of its own,
// and the uniform bindings of the replay use the copy.
// - Only ranges of imported pools (memory the CPU writes) are copied, aligned out to 256 bytes (the largest
//   minUniformBufferOffsetAlignment) and at most 64 KiB from the bound address, overlapping ones merged.
// - Immutable copied bytes retain their mapping and a pre-copy producer observation. Each binding checks its actual
//   ordered input position, including preceding GPU writes to aliased ranges and writes that settled after the copy.
// - The arena goes back to a free list after final graphics completion or cancellation with no outstanding GPU users.
// Synchronous preflight reserves capacity; vk_exec_record takes the copy at its observation cutoff. With
// NX2NAVI_REPLAY_ASYNC the submitting thread copies before queuing the exact capture for replay, which can lag up to
// a frame. Copying on that thread's later replay would observe intervening guest writes: in a
// graveyard the block of the shadow pass came out with another block's values in the frame its allocator moved on to
// the next stretch of memory, about every 364 frames, and that frame's sunlight was all in shadow (a dark frame with
// the interface intact). NX2NAVI_UBO_SNAPSHOT=0 binds live memory as before; =ab alternates (NX2NAVI_AB).
#include <stdlib.h>
#include <string.h>

#include "host.h"
#include "nvn_api.h"
#include "resource_core_vma.h"
#include "resource_host_write.h"
#include "vk.h"
#include "vk_uniform_custody.h"
#include "vk_uniform_payload.h"
#include "vk_exec.h"
#include "vk_gpa.h"
#include "motion_owner.h"
#include "replay_phase_trace.h"
#include "vk_dlaa.h"
#include "platform/platform.h"

#define ALIGN 256u
#define ARENA_MIN (4u << 20)
#define CALL_DEPTH 16
#define PERMISSION_LOCAL_RANGES 8

enum {
    ARENA_REUSABLE, ARENA_RESERVED, ARENA_CAPTURING, ARENA_QUEUED, ARENA_RECORDING,
    ARENA_PENDING_COMPLETION, ARENA_RETIRING
};

typedef struct Arena {
    struct Arena *next;
    VkBuffer buf;
    VmaAllocation alloc;
    uint8_t *map;
    uint8_t *cpu;    // with DLSS (vk_dlaa_available), the same bytes in host memory for the CPU's reads (vk_usnap_bytes):
                     // map is often device memory through the BAR, written fast but read back at about a microsecond a
                     // cache line. Without DLSS nothing reads them back, and the bytes go into map in one copy.
    uint64_t size;
    RcBudgetTicket metadata_budget;
    RcBudgetTicket host_budget;
    RcAdmissionVector native_cost;
    unsigned state, cpu_current, cpu_needed;
} Arena;

typedef struct { uint64_t lo, hi; } Rng;
typedef struct {
    uint64_t lo, hi, off;
    MemBackingCapture *mapping;
    VkMemSpanCapture *import;
    VkUsnapResolveResult unavailable;
} Reg;
typedef struct {
    MemBackingCapturePlan host;
    VkMemSpanCapturePlan import;
} ReceiptPlan;
typedef struct {
    RcRegistryGpuCapture read;
    RcSourceRange *coverage;
    uint32_t count;
    size_t coverage_bytes;
    RcBudgetTicket coverage_budget;
} ProducerObservation;

static struct {
    int mode;                   // rt_knob: 0 off, 1 on, 2 ab
    MotionOwnerSnapshot *owners;
    int on, expected;           // retained capture and whether CPU uniforms require a copy
    uint64_t fence, gen, capture;
    Arena *arena;
    Rng *col;                   // collected ranges
    int ncol, ccol, gather_retry_capacity;
    RcBudgetTicket col_budget;
    Rng *sort_tmp;
    int sort_cap;
    RcBudgetTicket sort_budget;
    uint64_t gather_alignment;
    Reg *reg;                   // ordered by start and end
    int nreg, overlapping;
    RcBudgetTicket region_budget, receipt_budget;
    RcSourceRange *permission_ranges;
    uint32_t permission_capacity;
    ProducerObservation producer;
    // since the last summary
    uint64_t subs, regions, bytes, copy_ns, hits, refused, off_subs;
} U;

static Arena *free_arenas;
static PlatMutex free_lock = PLAT_MUTEX_INIT;
static PlatCond free_idle = PLAT_COND_INIT;
static unsigned retiring_lists;
static int arenas_stopped;
// Inventory collection uses shared sort scratch and a shared neighbour cache.
static PlatMutex gather_lock = PLAT_MUTEX_INIT;
// Only fragmented retained mappings share this pre-admitted scratch across recording workers.
static PlatMutex permission_lock = PLAT_MUTEX_INIT;

// This owner travels with one replay request. Its identity survives queueing without depending on the final fence.
struct VkUsnapCapture {
    MotionOwnerSnapshot *owners;
    Arena *arena;
    Reg *reg;
    ReceiptPlan *receipts;
    RcSourceRange *permission_ranges;
    uint32_t permission_capacity;
    ProducerObservation producer;
    int nreg, region_cap, receipt_cap, enabled, overlapping;
    RcBudgetTicket header_budget, region_budget, receipt_plan_budget, receipt_budget;
    size_t receipt_peak, receipt_bytes;
    unsigned state;
    uint64_t id, total, refused, ns;
    uint64_t alignment;
};
static uint64_t next_capture_id;

static VkResult metadata_allocate(size_t size, int clear, RcBudgetTicket *joint,
                                  RcBudgetTicket *ticket, void **allocation);
static RcAdmissionVector arena_host_cost(uint64_t size);

static int by_backing_offset(const void *left, const void *right) {
    const RcSourceRange *a = left, *b = right;
    if (a->backing != b->backing) return a->backing < b->backing ? -1 : 1;
    return a->offset < b->offset ? -1 : a->offset > b->offset;
}

static uint64_t snapshot_alignment(void) {
    const VkPhysicalDeviceLimits *limits = &vk_ctx()->props.limits;
    uint64_t alignment = ALIGN;
    if (limits->minUniformBufferOffsetAlignment > alignment) alignment = limits->minUniformBufferOffsetAlignment;
    if (limits->minStorageBufferOffsetAlignment > alignment) alignment = limits->minStorageBufferOffsetAlignment;
    return alignment;
}

static uint64_t align_down(uint64_t value, uint64_t alignment) {
    return value - value % alignment;
}

static int align_up(uint64_t value, uint64_t alignment, uint64_t *out) {
    if (!alignment || value > UINT64_MAX - (alignment - 1)) return 0;
    *out = ((value + alignment - 1) / alignment) * alignment;
    return 1;
}

static uint32_t normalize_ranges(RcSourceRange *ranges, uint32_t count) {
    if (count < 2) return count;
    qsort(ranges, count, sizeof *ranges, by_backing_offset);
    uint32_t n = 1;
    for (uint32_t i = 1; i < count; i++) {
        RcSourceRange *tail = &ranges[n - 1], next = ranges[i];
        if (tail->backing == next.backing && next.offset <= tail->offset + tail->size) {
            uint64_t end = next.offset + next.size;
            if (end > tail->offset + tail->size) tail->size = end - tail->offset;
        } else ranges[n++] = next;
    }
    return n;
}

static RcRegistryResult producer_reserve(ProducerObservation *observation, const Reg *regions, int nreg,
                                           uint32_t foreground_slots, uint64_t *epoch,
                                           VkResult *metadata_result, RcAdmissionVector *peak) {
    uint32_t count = 0;
    for (int i = 0; i < nreg; i++) {
        uint32_t n;
        mem_backing_intervals(regions[i].mapping, &n);
        if (n > UINT32_MAX - count) return RC_REGISTRY_CAPACITY;
        count += n;
    }
    if (!count) return RC_REGISTRY_INVALID;
    if ((uint64_t)count * sizeof(RcSourceRange) > SIZE_MAX) return RC_REGISTRY_CAPACITY;
    size_t bytes = (size_t)count * sizeof(RcSourceRange);
    RcSourceRange *ranges;
    if (metadata_result) {
        void *allocation;
        *metadata_result = metadata_allocate(bytes, 0, NULL, &observation->coverage_budget, &allocation);
        if (*metadata_result != VK_SUCCESS) {
            *peak = arena_host_cost(bytes);
            return RC_REGISTRY_CAPACITY;
        }
        ranges = allocation;
    } else {
        ranges = malloc(bytes);
        if (!ranges) return RC_REGISTRY_CAPACITY;
    }
    uint32_t at = 0;
    for (int i = 0; i < nreg; i++) {
        uint32_t n;
        const RcInterval *mapped = mem_backing_intervals(regions[i].mapping, &n);
        for (uint32_t j = 0; j < n; j++) ranges[at++] = mapped[j].range;
    }
    count = normalize_ranges(ranges, count);
    RcRegistryResult result = rc_registry_gpu_capture_reserve(rc_registry_runtime(), ranges, count, foreground_slots,
                                                              &observation->read, epoch);
    if (result == RC_REGISTRY_OK) {
        observation->coverage = ranges;
        observation->count = count;
        observation->coverage_bytes = bytes;
    } else if (metadata_result) vk_replay_metadata_free(ranges, &observation->coverage_budget);
    else free(ranges);
    return result;
}

static VkResult producer_failure(RcRegistryResult result) {
    if (result == RC_REGISTRY_CAPACITY) return VK_ERROR_OUT_OF_HOST_MEMORY;
    if (result == RC_REGISTRY_EXHAUSTED) return VK_ERROR_TOO_MANY_OBJECTS;
    return VK_ERROR_INITIALIZATION_FAILED;
}

static void producer_end(ProducerObservation *observation) {
    if (observation->read.id &&
        rc_registry_gpu_capture_end(rc_registry_runtime(), &observation->read) != RC_REGISTRY_OK)
        rt_fatal("vk_usnap: releasing a producer observation without ownership");
    if (observation->coverage_budget.token.ledger)
        vk_replay_metadata_free(observation->coverage, &observation->coverage_budget);
    else free(observation->coverage);
    *observation = (ProducerObservation){0};
    vk_uniform_payload_reap();
}

static void source_release(Reg *source) {
    vk_mem_span_capture_release(&source->import);
    mem_backing_release(&source->mapping);
}

static void regions_release(Reg *regions, int count, RcBudgetTicket *ticket,
                             RcSourceRange *permission_ranges, RcBudgetTicket *receipts) {
    if (receipts->token.ledger && resource_core_budget_retire(receipts) != RC_ADMISSION_OK)
        rt_fatal("vk_usnap: source receipt capacity retirement failed");
    for (int i = 0; i < count; i++) source_release(&regions[i]);
    free(permission_ranges);
    if (receipts->token.ledger && resource_core_budget_destroyed(receipts) != RC_ADMISSION_OK)
        rt_fatal("vk_usnap: source receipt capacity destruction failed");
    vk_replay_metadata_free(regions, ticket);
}

static int cpu_pool_span(uint64_t addr, uint64_t size, VkMemSpan *span) {
    return size && size <= UINT64_MAX - addr && vk_mem_lookup(addr, size, span) && !span->alloc && span->pool &&
        span->avail >= size && (span->cls == VK_MCLS_HOST_CACHED || span->cls == VK_MCLS_HOST_UNC);
}

typedef enum {
    SOURCE_RETAINED, SOURCE_NO_CPU_POOL, SOURCE_REJECTED, SOURCE_MAPPING_STALE, SOURCE_NO_CAPACITY,
    SOURCE_PLAN_CHANGED
} SourceRetainResult;

static SourceRetainResult source_receipt_plan(const Reg *source, ReceiptPlan *plan) {
    VkMemSpan span;
    uint64_t size = source->hi - source->lo;
    if (!cpu_pool_span(source->lo, size, &span)) return SOURCE_NO_CPU_POOL;
    if (nvn_gpu_only(source->lo)) return SOURCE_REJECTED;
    RcBackingResult mapped = mem_backing_capture_plan(source->lo, size, &plan->host);
    if (mapped == RC_BACKING_STALE || mapped == RC_BACKING_ABSENT) return SOURCE_PLAN_CHANGED;
    if (mapped == RC_BACKING_CAPACITY || mapped == RC_BACKING_EXHAUSTED) return SOURCE_NO_CAPACITY;
    if (mapped != RC_BACKING_OK) return SOURCE_REJECTED;
    RcRegistryResult result = vk_mem_span_capture_plan(&span, &plan->host, &plan->import);
    if (result == RC_REGISTRY_OK) return SOURCE_RETAINED;
    if (result == RC_REGISTRY_STALE) return SOURCE_PLAN_CHANGED;
    return result == RC_REGISTRY_CAPACITY ? SOURCE_NO_CAPACITY : SOURCE_REJECTED;
}

static VkResult source_receipt_fill(Reg *source, const ReceiptPlan *plan) {
    RcRegistryResult result = vk_mem_span_capture_bounded(&plan->import, &plan->host, &source->import);
    if (result == RC_REGISTRY_INVALID) return VK_ERROR_INITIALIZATION_FAILED;
    if (result != RC_REGISTRY_OK)
        return result == RC_REGISTRY_CAPACITY ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_INCOMPLETE;

    RcBackingResult mapped = mem_backing_capture_bounded(&plan->host, &source->mapping);
    if (mapped == RC_BACKING_OK &&
        vk_mem_canonical_source_qualify_expected(source->mapping, source->import) == RC_REGISTRY_OK)
        return VK_SUCCESS;
    source_release(source);
    if (mapped == RC_BACKING_INVALID) return VK_ERROR_INITIALIZATION_FAILED;
    return mapped == RC_BACKING_CAPACITY ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_INCOMPLETE;
}

static int source_receipt_bytes(const ReceiptPlan *plan, size_t *bytes) {
    if (plan->host.bytes > SIZE_MAX - plan->import.bytes) return 0;
    *bytes = plan->host.bytes + plan->import.bytes;
    return 1;
}

static uint64_t permission_scratch_bytes(uint32_t count) {
    return count > PERMISSION_LOCAL_RANGES ? (uint64_t)count * sizeof(RcSourceRange) : 0;
}

static RcAdmissionVector arena_host_cost(uint64_t size) {
    RcAdmissionVector cost = {{0}};
    cost.amount[RC_BUDGET_RAM] = cost.amount[RC_BUDGET_COMMIT] = size;
    return cost;
}

static int peak_sum(RcAdmissionVector *result, const RcAdmissionVector *left, const RcAdmissionVector *right) {
    RcAdmissionVector sum;
    for (unsigned d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) {
        if (right->amount[d] > UINT64_MAX - left->amount[d]) return 0;
        sum.amount[d] = left->amount[d] + right->amount[d];
    }
    *result = sum;
    return 1;
}

int vk_replay_metadata_peak_add(RcAdmissionVector *peak, uint64_t bytes) {
    RcAdmissionVector cost = arena_host_cost(bytes);
    return peak_sum(peak, peak, &cost);
}

void vk_replay_metadata_free(void *allocation, RcBudgetTicket *ticket) {
    if (!ticket->token.ledger) {
        if (allocation) rt_fatal("vk: replay metadata has no capacity owner");
        return;
    }
    if (resource_core_budget_retire(ticket) != RC_ADMISSION_OK)
        rt_fatal("vk: replay metadata retirement failed");
    free(allocation);
    if (resource_core_budget_destroyed(ticket) != RC_ADMISSION_OK)
        rt_fatal("vk: replay metadata destruction failed");
}
static void arena_cancel_host_reservation(Arena *a) {
    if (a->host_budget.token.ledger && resource_core_budget_cancel(&a->host_budget) != RC_ADMISSION_OK)
        rt_fatal("vk_usnap: CPU snapshot reservation cancellation failed");
}
static void arena_retire(Arena *a, unsigned from) {
    if (a->state != from) rt_fatal("vk_usnap: retiring an arena with a capture owner");
    a->state = ARENA_RETIRING;
}
static void arena_destroy(Arena *a) {
    if (a->state != ARENA_RETIRING) rt_fatal("vk_usnap: destroying an arena before retirement");
    if (a->cpu) {
        if (resource_core_budget_retire(&a->host_budget) != RC_ADMISSION_OK)
            rt_fatal("vk_usnap: CPU snapshot lost its capacity owner");
        free(a->cpu);
        if (resource_core_budget_destroyed(&a->host_budget) != RC_ADMISSION_OK)
            rt_fatal("vk_usnap: CPU snapshot capacity destruction failed");
    } else arena_cancel_host_reservation(a);
    resource_core_vma_destroy_buffer(vk_ctx()->vma, a->buf, a->alloc);
    RcBudgetTicket metadata = a->metadata_budget;
    vk_replay_metadata_free(a, &metadata);
}
// free_lock held. Once detached, the list owns its allocations until all native destroys have returned.
static Arena *arena_detach_all(void) {
    Arena *list = free_arenas;
    free_arenas = NULL;
    if (list) {
        retiring_lists++;
        for (Arena *a = list; a; a = a->next) arena_retire(a, ARENA_REUSABLE);
    }
    return list;
}
static uint64_t arena_destroy_list(Arena *list) {
    if (!list) return 0;
    uint64_t count = 0;
    while (list) {
        Arena *next = list->next;
        arena_destroy(list);
        list = next;
        count++;
    }
    plat_mutex_lock(&free_lock);
    if (!--retiring_lists) plat_cond_broadcast(&free_idle);
    plat_mutex_unlock(&free_lock);
    return count;
}
// Detach under the reuse lock and destroy outside it. The count is diagnostic; OS/heap samples establish relief.
uint64_t vk_usnap_reclaim_reusable(void) {
    plat_mutex_lock(&free_lock);
    Arena *list = arenas_stopped ? NULL : arena_detach_all();
    plat_mutex_unlock(&free_lock);
    return arena_destroy_list(list);
}

static void arena_shutdown(void) {
    plat_mutex_lock(&free_lock);
    arenas_stopped = 1;
    while (retiring_lists) plat_cond_wait(&free_idle, &free_lock);
    Arena *list = arena_detach_all();
    plat_mutex_unlock(&free_lock);
    arena_destroy_list(list);
}
static unsigned arena_reclaim(void *context, const RcAdmissionVector *deficit) {
    (void)context;
    RcAdmissionVector remaining = *deficit;
    Arena *victims = NULL;
    plat_mutex_lock(&free_lock);
    if (arenas_stopped) {
        plat_mutex_unlock(&free_lock);
        return RC_BUDGET_RECLAIM_NONE;
    }
    for (Arena **link = &free_arenas; *link;) {
        Arena *a = *link;
        RcAdmissionVector cost = a->native_cost;
        if (a->cpu) {
            cost.amount[RC_BUDGET_RAM] += a->size;
            cost.amount[RC_BUDGET_COMMIT] += a->size;
        }
        int relevant = 0;
        for (unsigned d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++)
            relevant |= remaining.amount[d] && cost.amount[d];
        if (!relevant) { link = &a->next; continue; }
        // The CPU header alone is too small to justify throwing away a reusable GPU allocation.
        RcAdmissionVector header = arena_host_cost(sizeof *a);
        if (!peak_sum(&cost, &cost, &header)) rt_fatal("vk_usnap: arena capacity overflow");
        *link = a->next;
        arena_retire(a, ARENA_REUSABLE);
        a->next = victims; victims = a;
        // This bounds victim selection only. The service must observe actual relief before admitting work.
        for (unsigned d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++)
            remaining.amount[d] = remaining.amount[d] > cost.amount[d] ? remaining.amount[d] - cost.amount[d] : 0;
    }
    if (victims) retiring_lists++;
    plat_mutex_unlock(&free_lock);
    return arena_destroy_list(victims) ? RC_BUDGET_RECLAIM_DESTROYED : RC_BUDGET_RECLAIM_NONE;
}
static void arena_register_reclaimer(void *unused) {
    (void)unused;
    uint32_t domains = ((1u << RC_ADMISSION_MAX_DOMAINS) - 1) & ~(1u << RC_BUDGET_STAGING);
    if (resource_core_budget_register_reclaimer(RC_BUDGET_SNAPSHOT, domains, arena_reclaim, NULL) != RC_ADMISSION_OK)
        rt_fatal("vk_usnap: registering snapshot reclamation failed");
}

static int arena_host_materialize(Arena *a) {
    uint8_t *cpu = malloc((size_t)a->size);
    if (!cpu) {
        arena_cancel_host_reservation(a);
        return 0;
    }
    // Touch each page while its RAM allowance is still reserved. A refresh between preflight and capture
    // could otherwise observe an untouched malloc and clear the unobserved charge for its working set.
    volatile uint8_t *pages = cpu;
    for (uint64_t off = 0; off < a->size; off += 4096) pages[off] = 0;
    pages[a->size - 1] = 0;
    RcAdmissionVector actual = arena_host_cost(a->size);
    if (!rc_admission_accounted(resource_core_budget_materialize(&a->host_budget, &actual)))
        rt_fatal("vk_usnap: CPU snapshot capacity settlement failed");
    a->cpu = cpu;
    return 1;
}
static VkResult arena_admission_failure(RcAdmissionStatus status) {
    if (status == RC_ADMISSION_REFUSED) return VK_NOT_READY;
    // The pressure service only resolves byte deficits. A full owner table cannot wait on a zero-byte request.
    if (status == RC_ADMISSION_NO_RECORDS) return VK_ERROR_TOO_MANY_OBJECTS;
    if (status == RC_ADMISSION_UNSUPPORTED) return VK_ERROR_FEATURE_NOT_PRESENT;
    return VK_ERROR_INITIALIZATION_FAILED;
}

static void metadata_touch_pages(void *allocation, size_t size) {
    volatile uint8_t *pages = allocation;
    for (size_t off = 0;; off += 4096) {
        pages[off] = 0;
        if (size - off <= 4096) break;
    }
    pages[size - 1] = 0;
}

static VkResult replay_metadata_allocate(size_t size, int clear, RcBudgetProducer producer, RcBudgetTicket *joint,
                                         RcBudgetTicket *ticket, void **allocation) {
    *allocation = NULL;
    if (!size || ticket->token.ledger) return VK_ERROR_INITIALIZATION_FAILED;
    RcAdmissionVector cost = arena_host_cost(size);
    RcAdmissionStatus admitted = joint
        ? resource_core_budget_split(joint, &cost, producer, ticket)
        : resource_core_budget_reserve(producer, 1, &cost, ticket);
    if (admitted != RC_ADMISSION_OK) return arena_admission_failure(admitted);

    void *data = malloc(size);
    if (!data) {
        if (resource_core_budget_cancel(ticket) != RC_ADMISSION_OK)
            rt_fatal("vk: replay metadata reservation cancellation failed");
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    // Commit and touch the reserved capacity before settling RAM and commit. Entries beyond the live
    // count are written by their owner before use.
    if (clear) memset(data, 0, size);
    metadata_touch_pages(data, size);
    if (!rc_admission_accounted(resource_core_budget_materialize(ticket, &cost)))
        rt_fatal("vk: replay metadata settlement failed");
    *allocation = data;
    return VK_SUCCESS;
}

VkResult vk_replay_metadata_allocate(size_t size, int clear, RcBudgetProducer producer,
                                     RcBudgetTicket *ticket, void **allocation) {
    return replay_metadata_allocate(size, clear, producer, NULL, ticket, allocation);
}

static VkResult metadata_allocate(size_t size, int clear, RcBudgetTicket *joint,
                                  RcBudgetTicket *ticket, void **allocation) {
    return replay_metadata_allocate(size, clear, RC_BUDGET_SNAPSHOT, joint, ticket, allocation);
}

static VkResult metadata_resize(void *old, size_t copy_bytes, size_t new_bytes, RcBudgetTicket *ticket,
                                RcAdmissionVector *peak, void **allocation) {
    RcBudgetTicket replacement = {0};
    VkResult result = metadata_allocate(new_bytes, 0, NULL, &replacement, allocation);
    if (result != VK_SUCCESS) {
        *peak = arena_host_cost(new_bytes);
        return result;
    }
    if (copy_bytes) memcpy(*allocation, old, copy_bytes);
    vk_replay_metadata_free(old, ticket);
    *ticket = replacement;
    return VK_SUCCESS;
}

static int metadata_next_capacity(int capacity, size_t element_size) {
    if (capacity > INT_MAX / 2) return 0;
    int next = capacity ? capacity * 2 : 1024;
    return (size_t)next <= SIZE_MAX / element_size ? next : 0;
}
static int arena_gpu_growth_blocked(const RcAdmissionVector *peak, uint64_t shadow_bytes) {
    RcBudgetSnapshot budget;
    resource_core_budget_snapshot(&budget);
    if (shadow_bytes > budget.available.amount[RC_BUDGET_RAM] ||
        shadow_bytes > budget.available.amount[RC_BUDGET_COMMIT]) return 0;
    for (unsigned d = RC_BUDGET_GPU_FIRST; d < RC_ADMISSION_MAX_DOMAINS; d++)
        if ((budget.sample.known_mask & (1u << d)) && peak->amount[d] > budget.available.amount[d]) return 1;
    return 0;
}
static void arena_return(Arena *a, unsigned from);

static VkResult arena_take_reusable(uint64_t need, int host, Arena **out) {
    static PlatOnce reclaimer_once = PLAT_ONCE_INIT;
    plat_once(&reclaimer_once, arena_register_reclaimer, NULL);
    *out = NULL;
    if (!need || need > SIZE_MAX - ((1u << 20) - 1)) return VK_ERROR_OUT_OF_HOST_MEMORY;
    plat_mutex_lock(&free_lock);
    if (arenas_stopped) {
        plat_mutex_unlock(&free_lock);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    Arena **fit = &free_arenas, **without_shadow = NULL;
    for (; *fit; fit = &(*fit)->next) {
        Arena *candidate = *fit;
        if (candidate->size < need) continue;
        if (!host || candidate->cpu) break;
        if (!without_shadow || candidate->size < (*without_shadow)->size) without_shadow = fit;
    }
    if (!*fit && without_shadow) fit = without_shadow;
    Arena *a = *fit;
    if (a) {
        *fit = a->next;
        a->next = NULL;
        a->cpu_needed = host;
        a->state = ARENA_RESERVED;
    }
    plat_mutex_unlock(&free_lock);
    *out = a;
    return VK_SUCCESS;
}

static VkBufferCreateInfo arena_buffer_info(uint64_t need) {
    uint64_t size = need < ARENA_MIN ? ARENA_MIN : (need + (1u << 20) - 1) & ~(uint64_t)((1u << 20) - 1);
    // Storage usage supports readonly SSBO bindings of captured uniform-backed data; transfer source lets DLAA
    // copy out the camera block.
    return (VkBufferCreateInfo){.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
        .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
}

static VmaAllocationCreateInfo arena_allocation_info(void) {
    // Prefer device-local memory through the BAR; the GPU reads this on every draw.
    return (VmaAllocationCreateInfo){.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
}

static VkResult arena_fill_shadow(Arena *a, RcBudgetTicket *joint, RcAdmissionVector *refused_peak) {
    if (!a->cpu_needed || a->cpu) return VK_SUCCESS;
    RcAdmissionVector cost = arena_host_cost(a->size);
    *refused_peak = cost;
    RcAdmissionStatus admitted = joint
        ? resource_core_budget_split(joint, &cost, RC_BUDGET_SNAPSHOT, &a->host_budget)
        : resource_core_budget_reserve(RC_BUDGET_SNAPSHOT, 1, &cost, &a->host_budget);
    if (admitted != RC_ADMISSION_OK) return arena_admission_failure(admitted);
    return arena_host_materialize(a) ? VK_SUCCESS : VK_ERROR_OUT_OF_HOST_MEMORY;
}

static VkResult arena_create(const VkBufferCreateInfo *bi, VmaAllocationCreateInfo ai, int host,
                             RcBudgetTicket *joint, RcAdmissionVector *refused_peak, Arena **out) {
    RcBudgetTicket metadata = {0};
    void *allocation;
    *refused_peak = arena_host_cost(sizeof(Arena));
    VkResult result = metadata_allocate(sizeof(Arena), 1, joint, &metadata, &allocation);
    if (result != VK_SUCCESS) return result;
    Arena *a = allocation;
    a->metadata_budget = metadata;
    a->size = bi->size;
    VmaAllocationInfo info;
    RcVmaRequest request = {.producer = RC_BUDGET_SNAPSHOT, .required = 1,
        .host_bytes = host ? a->size : 0, .host_ticket = host ? &a->host_budget : NULL,
        .refused_peak = refused_peak, .peak_ticket = joint};
    RcAdmissionStatus admitted;
    VmaAllocator allocator = vk_ctx()->vma;
    result = resource_core_vma_create_buffer(&request, allocator, bi, &ai, &a->buf, &a->alloc, &info, &admitted);
    if (!joint && result == VK_NOT_READY && admitted == RC_ADMISSION_REFUSED &&
        arena_gpu_growth_blocked(refused_peak, request.host_bytes)) {
        // Standalone admission can retry with a dedicated host-preferred placement.
        RcAdmissionVector primary_peak = *refused_peak;
        ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        ai.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        result = resource_core_vma_create_buffer(&request, allocator, bi, &ai, &a->buf, &a->alloc, &info, &admitted);
        if (result == VK_ERROR_FEATURE_NOT_PRESENT) {
            *refused_peak = primary_peak;
            result = VK_NOT_READY;
            admitted = RC_ADMISSION_REFUSED;
        }
    }
    if (result != VK_SUCCESS) {
        RcAdmissionVector header = arena_host_cost(sizeof *a);
        if (!peak_sum(refused_peak, refused_peak, &header)) result = VK_ERROR_OUT_OF_HOST_MEMORY;
        vk_replay_metadata_free(a, &metadata);
        return result == VK_NOT_READY ? arena_admission_failure(admitted) : result;
    }
    a->map = info.pMappedData;
    if (!a->map || (host && !arena_host_materialize(a))) {
        result = a->map ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_ERROR_MEMORY_MAP_FAILED;
        arena_retire(a, ARENA_REUSABLE);
        arena_destroy(a);
        return result;
    }
    if (resource_core_vma_allocation_cost(a->alloc, &a->native_cost) != RC_ADMISSION_OK)
        rt_fatal("vk_usnap: native snapshot lost its capacity owner");
    a->cpu_needed = host;
    a->state = ARENA_RESERVED;
    rt_log("vk_usnap: new arena of %.1f MiB", a->size / 1048576.0);
    *out = a;
    return VK_SUCCESS;
}

typedef struct {
    VkBufferCreateInfo buffer;
    VmaAllocationCreateInfo placement;
    int host;
} QueuedArenaPlan;

static VkResult arena_queue_placement(QueuedArenaPlan *plan, VmaMemoryUsage usage, RcAdmissionVector *peak) {
    VmaAllocationCreateInfo preferred = arena_allocation_info();
    preferred.usage = usage;
    preferred.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    RcVmaRequest request = {.producer = RC_BUDGET_SNAPSHOT, .required = 1};
    VkResult result = resource_core_vma_buffer_peak(&request, &plan->buffer, &preferred, &plan->placement, peak);
    if (result != VK_SUCCESS) return result;
    RcAdmissionVector header = arena_host_cost(sizeof(Arena));
    if (!peak_sum(peak, peak, &header)) return VK_ERROR_OUT_OF_HOST_MEMORY;
    if (plan->host) {
        RcAdmissionVector shadow = arena_host_cost(plan->buffer.size);
        if (!peak_sum(peak, peak, &shadow)) return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    return VK_SUCCESS;
}

static VkResult arena_queue_growth_plan(QueuedArenaPlan *plan, const RcAdmissionVector *scratch,
                                        const RcBudgetReadiness *before, RcAdmissionVector *peak) {
    RcAdmissionVector native_and_shadow;
    VkResult result = arena_queue_placement(plan, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, &native_and_shadow);
    if (result != VK_SUCCESS) return result;
    if (!peak_sum(peak, scratch, &native_and_shadow)) return VK_ERROR_OUT_OF_HOST_MEMORY;
    int gpu_blocked = 0;
    for (unsigned d = RC_BUDGET_GPU_FIRST; d < RC_ADMISSION_MAX_DOMAINS; d++)
        gpu_blocked |= (before->usable_mask & (1u << d)) && peak->amount[d] > before->available.amount[d];
    if (gpu_blocked && !resource_core_budget_peak_ready(peak, before)) {
        // Select at most one alternative before taking the joint grant. The chosen memory type stays pinned.
        QueuedArenaPlan alternate = *plan;
        RcAdmissionVector alt_peak;
        if (arena_queue_placement(&alternate, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, &alt_peak) == VK_SUCCESS &&
            peak_sum(&alt_peak, scratch, &alt_peak) && resource_core_budget_peak_ready(&alt_peak, before)) {
            *plan = alternate;
            *peak = alt_peak;
        }
    }
    return VK_SUCCESS;
}

static VkResult arena_queue_plan(VkUsnapCapture *t, QueuedArenaPlan *plan, const RcAdmissionVector *scratch,
                                 const RcBudgetReadiness *before, RcAdmissionVector *peak) {
    if (!t || !t->total) return VK_SUCCESS;
    plan->host = vk_dlaa_available();
    VkResult result = arena_take_reusable(t->total, plan->host, &t->arena);
    if (result != VK_SUCCESS) return result;
    if (t->arena) {
        if (!plan->host || t->arena->cpu) return VK_SUCCESS;
        RcAdmissionVector shadow = arena_host_cost(t->arena->size);
        return peak_sum(peak, scratch, &shadow) ? VK_SUCCESS : VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    plan->buffer = arena_buffer_info(t->total);
    return arena_queue_growth_plan(plan, scratch, before, peak);
}

// A failed smaller replacement can still describe useful pressure if the old reusable native allocation can
// actually cover every known deficit. This is only a wait/reclaim hint: the old allocation remains charged until
// the ordinary reclaimer destroys it and a fresh budget sample observes the relief.
static int arena_smaller_peak_reclaimable(const Arena *a, const RcAdmissionVector *peak,
                                          const RcBudgetReadiness *now) {
    if (!a || !peak || !now || !now->records_available) return 0;
    int heap_deficit = 0;
    for (unsigned d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) {
        uint64_t requested = peak->amount[d];
        if (!requested) continue;
        if (!(now->usable_mask & (1u << d))) return 0;
        uint64_t available = now->available.amount[d];
        if (requested <= available) continue;
        uint64_t deficit = requested - available;
        if (d >= RC_BUDGET_GPU_FIRST) heap_deficit = 1;
        if (a->native_cost.amount[d] < deficit) return 0;
    }
    return heap_deficit;
}

// After the full shadow of a reusable arena was refused, a smaller fresh arena may still fit.
// Price its header, placement and shadow together while the reusable arena remains charged.
static int arena_reserve_smaller_shadow(const Arena *a, uint64_t need, const RcAdmissionVector *scratch,
                                        QueuedArenaPlan *plan, RcAdmissionVector *peak, RcBudgetTicket *joint) {
    if (!a || !a->cpu_needed || a->cpu) return 0;
    QueuedArenaPlan smaller = {.buffer = arena_buffer_info(need), .host = 1};
    if (smaller.buffer.size >= a->size) return 0;
    RcBudgetReadiness now;
    resource_core_budget_readiness(1, &now);
    RcAdmissionVector smaller_peak;
    if (arena_queue_growth_plan(&smaller, scratch, &now, &smaller_peak) != VK_SUCCESS) return 0;
    if (!resource_core_budget_peak_ready(&smaller_peak, &now)) {
        if (arena_smaller_peak_reclaimable(a, &smaller_peak, &now)) *peak = smaller_peak;
        return 0;
    }
    RcAdmissionStatus admitted = resource_core_budget_reserve(RC_BUDGET_SNAPSHOT, 1, &smaller_peak, joint);
    if (admitted != RC_ADMISSION_OK) {
        if (admitted == RC_ADMISSION_REFUSED) {
            resource_core_budget_readiness(1, &now);
            if (arena_smaller_peak_reclaimable(a, &smaller_peak, &now)) *peak = smaller_peak;
        }
        return 0;
    }
    *plan = smaller;
    *peak = smaller_peak;
    return 1;
}

static Arena *arena_try_get(uint64_t need, VkResult *result, RcAdmissionVector *refused_peak) {
    *refused_peak = (RcAdmissionVector){{0}};
    int host = vk_dlaa_available();
    Arena *a;
    *result = arena_take_reusable(need, host, &a);
    if (*result != VK_SUCCESS) return NULL;
    if (a) {
        *result = arena_fill_shadow(a, NULL, refused_peak);
        if (*result == VK_SUCCESS) return a;
        QueuedArenaPlan smaller;
        RcBudgetTicket joint = {0};
        RcAdmissionVector no_scratch = {{0}};
        int replace = *result == VK_NOT_READY &&
            arena_reserve_smaller_shadow(a, need, &no_scratch, &smaller, refused_peak, &joint);
        arena_return(a, ARENA_RESERVED);
        if (!replace) return NULL;
        a = NULL;
        *result = arena_create(&smaller.buffer, smaller.placement, smaller.host, &joint, refused_peak, &a);
        if (joint.token.ledger && resource_core_budget_cancel(&joint) != RC_ADMISSION_OK)
            rt_fatal("vk_usnap: smaller arena lost its capacity reservation");
        return a;
    }
    VkBufferCreateInfo buffer = arena_buffer_info(need);
    *result = arena_create(&buffer, arena_allocation_info(), host, NULL, refused_peak, &a);
    return a;
}

static VkResult arena_queue_materialize(VkUsnapCapture *t, const QueuedArenaPlan *plan, RcBudgetTicket *joint) {
    if (!t || !t->total) return VK_SUCCESS;
    RcAdmissionVector attempted;
    if (t->arena) return arena_fill_shadow(t->arena, joint, &attempted);
    return arena_create(&plan->buffer, plan->placement, plan->host, joint, &attempted, &t->arena);
}

// The dedicated peak was refused. Try an existing block in the default arena placement when
// its heap or commit is already held. Reserve the incremental working set together with scratch.
static VkResult arena_queue_try_compatible(VkUsnapCapture *t, const QueuedArenaPlan *plan,
                                           const RcAdmissionVector *scratch, RcAdmissionVector *peak,
                                           RcBudgetTicket *joint) {
    RcBudgetReadiness now;
    resource_core_budget_readiness(1, &now);
    int heap_blocked = 0;
    for (unsigned d = RC_BUDGET_GPU_FIRST; d < RC_ADMISSION_MAX_DOMAINS; d++)
        heap_blocked |= peak->amount[d] && (!(now.usable_mask & (1u << d)) || peak->amount[d] > now.available.amount[d]);
    int commit_blocked = peak->amount[RC_BUDGET_COMMIT] > now.available.amount[RC_BUDGET_COMMIT];
    if (!heap_blocked && !commit_blocked) return VK_INCOMPLETE;

    QueuedArenaPlan compatible = *plan;
    VmaAllocationCreateInfo preferred = arena_allocation_info();
    RcVmaDependency dependency;
    RcVmaRequest request = {.producer = RC_BUDGET_SNAPSHOT, .required = 1, .dependency = &dependency};
    RcAdmissionVector reuse_peak;
    VkResult result = resource_core_vma_buffer_peak(&request, &compatible.buffer, &preferred,
                                                     &compatible.placement, &reuse_peak);
    if (result != VK_SUCCESS || !dependency.can_reuse) return VK_INCOMPLETE;
    compatible.placement.flags |= VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT;
    RcAdmissionVector host = arena_host_cost(compatible.host ? compatible.buffer.size : 0);
    RcAdmissionVector header = arena_host_cost(sizeof(Arena));
    if (!peak_sum(&reuse_peak, scratch, &dependency.reuse) ||
        !peak_sum(&reuse_peak, &reuse_peak, &host) ||
        !peak_sum(&reuse_peak, &reuse_peak, &header)) return VK_INCOMPLETE;
    if (!heap_blocked && (reuse_peak.amount[RC_BUDGET_COMMIT] >= peak->amount[RC_BUDGET_COMMIT] ||
        !resource_core_budget_peak_ready(&reuse_peak, &now))) return VK_INCOMPLETE;

    RcAdmissionStatus admitted = resource_core_budget_reserve(RC_BUDGET_SNAPSHOT, 1, &reuse_peak, joint);
    if (admitted == RC_ADMISSION_UNSUPPORTED) return VK_INCOMPLETE;
    if (admitted != RC_ADMISSION_OK) {
        *peak = reuse_peak;
        return arena_admission_failure(admitted);
    }
    result = arena_queue_materialize(t, &compatible, joint);
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY) return VK_INCOMPLETE;
    *peak = reuse_peak;
    return result;
}

static void arena_return(Arena *a, unsigned from) {
    if (!a || a->state != from) rt_fatal("vk_usnap: invalid arena ownership transfer");
    a->state = ARENA_REUSABLE;
    a->cpu_current = 0;
    plat_mutex_lock(&free_lock);
    a->next = free_arenas, free_arenas = a;
    plat_mutex_unlock(&free_lock);
    resource_core_budget_progress();
}
static void arena_done(void *arg) {   // completion thread: the GPU has read it
    Arena *a = arg;
    arena_return(a, ARENA_PENDING_COMPLETION);
}
static void arena_cancel(void *arg, int may_have_executed) {
    // vk_sync only disposes a possibly submitted callback after actual device-idle proof. Device loss retains
    // those callbacks instead; cancellation here must never be used as a substitute for GPU retirement.
    (void)may_have_executed;
    arena_return(arg, ARENA_PENDING_COMPLETION);
}

static VkResult collect_range(uint64_t lo, uint64_t hi, RcAdmissionVector *peak) {
    if (U.ncol == U.ccol) {
        int starting_capacity = metadata_next_capacity(U.ccol, sizeof *U.col);
        int capacity = U.gather_retry_capacity ? U.gather_retry_capacity : starting_capacity;
        if (!capacity) return VK_ERROR_OUT_OF_HOST_MEMORY;
        void *allocation;
        VkResult result = metadata_resize(U.col, (size_t)U.ncol * sizeof *U.col,
            (size_t)capacity * sizeof *U.col, &U.col_budget, peak, &allocation);
        if (result == VK_NOT_READY && capacity != starting_capacity) {
            // A different submission can reach the collector before the original retry.
            capacity = starting_capacity;
            result = metadata_resize(NULL, 0, (size_t)capacity * sizeof *U.col, &U.col_budget, peak, &allocation);
        }
        if (result != VK_SUCCESS) return result;
        U.col = allocation;
        U.ccol = capacity;
        U.gather_retry_capacity = 0;
    }
    U.col[U.ncol++] = (Rng){lo, hi};
    return VK_SUCCESS;
}

// The same binding can supply a UBO or readonly SSBO. Store both ends when the device limits differ so a
// legal shorter read can still be retained when the longer interpretation crosses the source pool.
static void binding_extents(uint64_t address, uint64_t size, uint64_t *first, uint64_t *second) {
    const VkPhysicalDeviceLimits *limits = &vk_ctx()->props.limits;
    uint64_t uniform = size < limits->maxUniformBufferRange ? size : limits->maxUniformBufferRange;
    uint64_t storage_alignment = limits->minStorageBufferOffsetAlignment;
    uint64_t lead = storage_alignment > 4 ? address % storage_alignment : 0;
    uint64_t storage_limit = limits->maxStorageBufferRange;
    uint64_t storage = 0;
    if (size <= UINT64_MAX - lead && storage_limit > lead) {
        storage_limit -= lead;
        storage = size < storage_limit ? size : storage_limit;
    }
    if (uniform > UINT64_MAX - address) uniform = 0;
    if (storage > UINT64_MAX - address) storage = 0;
    *first = uniform > storage ? uniform : storage;
    *second = uniform < storage ? uniform : storage;
}

static VkResult collect_binding(const p_bind_buf *b, RcAdmissionVector *peak) {
    // op_bind_ubo ignores these records; do not let an ignored descriptor expand the required snapshot.
    if (b->stage < 0 || b->stage >= 8 || b->index < 0 || b->index >= 18) return VK_SUCCESS;
    if (b->addr && b->size) {
        // the vertex stage's object and bone blocks: also the game's other copy on either side, which still
        // holds last frame's (vk_mvprev.c)
        uint64_t step = b->size <= UINT64_MAX - 255 ? vk_mvprev_step(b->stage, b->index, b->size) : 0;
        // which neighbours lie in an imported pool, per block and vk_mem_gen: the same blocks are bound by
        // many draws, and two resolves for each binding were a quarter of the snapshot's time
        static struct { uint64_t addr, size, gen, alignment; int ok; } nc[1024];
        uint32_t ci = (uint32_t)((b->addr ^ b->size << 48) * 0x9E3779B97F4A7C15ull >> 54);
        uint64_t gen = step ? vk_mem_gen() : 0;
        uint64_t alignment = U.gather_alignment ? U.gather_alignment : ALIGN;
        int known = step && nc[ci].addr == b->addr && nc[ci].size == b->size && nc[ci].gen == gen &&
            nc[ci].alignment == alignment, ok = 0;
        for (int k = step ? -1 : 0; k <= (step ? 1 : 0); k++) {
            if ((k < 0 && b->addr < step) || (k > 0 && step > UINT64_MAX - b->addr)) continue;
            uint64_t at = k < 0 ? b->addr - step : k > 0 ? b->addr + step : b->addr;
            uint64_t n, shorter;
            binding_extents(at, b->size, &n, &shorter);
            if (!n) continue;
            // a neighbour past its pool would make the whole merged region fail the pool check below, and
            // the bindings in it read live memory again
            VkMemSpan sp;
            uint64_t lo = align_down(at, alignment);
            if (k && known) {
                if (!(nc[ci].ok >> (k + 1) & 1)) continue;
            } else if (k) {
                if (!cpu_pool_span(lo, at + n - lo, &sp) &&
                    (!shorter || shorter == n || !cpu_pool_span(lo, at + shorter - lo, &sp))) continue;
                ok |= 1 << (k + 1);
            }
            VkResult result = collect_range(lo, at + n, peak);
            if (result == VK_SUCCESS && shorter && shorter != n) result = collect_range(lo, at + shorter, peak);
            if (result != VK_SUCCESS) return result;
        }
        if (step && !known) nc[ci].addr = b->addr, nc[ci].size = b->size, nc[ci].gen = gen,
            nc[ci].alignment = alignment, nc[ci].ok = ok;
    }
    return VK_SUCCESS;
}

static VkResult collect(const op_hdr *h, int depth, RcAdmissionVector *peak) {
    NvnRecordPlan plan;
    if (nvn_record_plan((uintptr_t)h, &plan)) {
        // These are unique capture inputs, not cached bytes or pool resolutions. Neighbour eligibility, imported
        // storage and in-flight writes are still determined for this submission at the original capture boundary.
        for (size_t i = 0; i < plan.ncapture; i++) {
            VkResult result = collect_binding(
                (const p_bind_buf *)(plan.commands + plan.capture_offsets[i] + sizeof(op_hdr)), peak);
            if (result != VK_SUCCESS) return result;
        }
        return VK_SUCCESS;
    }
    // Raw probe commands and reference mode retain the complete interpreter, including CALL/JUMP support.
    for (;;) {
        if (h->size < sizeof *h || (h->size & 7) || h->op >= OP_COUNT_)
            rt_fatal("vk_usnap: corrupt command record at %p: op %u size %u", (const void *)h, h->op, h->size);
        const void *a = nvn_op_payload(h);
        if (h->op == OP_END) return VK_SUCCESS;
        if (h->op == OP_JUMP) {
            h = (const op_hdr *)(uintptr_t)((const p_jump *)a)->target;
            continue;
        }
        VkResult result = VK_SUCCESS;
        if (h->op == OP_CALL && depth < CALL_DEPTH)
            result = collect(nvn_op_first(((const p_jump *)a)->target), depth + 1, peak);
        if (h->op == OP_BIND_UBO) result = collect_binding(a, peak);
        if (result != VK_SUCCESS) return result;
        h = nvn_op_next(h);
    }
}

static void project_binding(VkUsnapBindingState *state, const p_bind_buf *binding) {
    // As in op_bind_ubo, a valid slot is replaced even when this operation unbinds it.
    if (binding->stage >= 0 && binding->stage < 8 && binding->index >= 0 && binding->index < 18)
        state->slots[binding->stage * 18 + binding->index] = *binding;
}

static int project_handle(VkUsnapBindingState *state, uint64_t handle) {
    uintptr_t base = (uintptr_t)handle;
    uint64_t bytes = nvn_submission_handle_bytes(handle);
    if (!base || (uint64_t)base != handle || (base & 7) || (bytes & 7) || bytes < 2 * sizeof(op_hdr) ||
        bytes > UINTPTR_MAX - base) return 0;
    const op_hdr *begin = nvn_op_first(handle);
    const uint8_t *commands = (const uint8_t *)begin;
    size_t last = (size_t)bytes - sizeof(op_hdr);
    const op_hdr *end = (const op_hdr *)(commands + last);
    if (begin->op != OP_BEGIN || begin->size != sizeof *begin || end->op != OP_END || end->size != sizeof *end)
        return 0;

    const size_t *offsets;
    size_t count;
    size_t bind_bytes = sizeof(op_hdr) + sizeof(p_bind_buf);
    if (nvn_record_uniform_offsets(handle, &offsets, &count)) {
        // Indexes contain every bind in command order, including a repeated value after an intervening bind.
        if ((count && !offsets) || count > (last - sizeof *begin) / bind_bytes) return 0;
        size_t after = sizeof *begin;
        for (size_t i = 0; i < count; i++) {
            size_t at = offsets[i];
            if ((at & 7) || at < after || at > last || bind_bytes > last - at) return 0;
            const op_hdr *h = (const op_hdr *)(commands + at);
            if (h->op != OP_BIND_UBO || h->size != bind_bytes) return 0;
            project_binding(state, nvn_op_payload(h));
            after = at + bind_bytes;
        }
        return 1;
    }

    // Only retained flat commands can be projected. The raw interpreter's external call/jump targets
    // have no owned extent here.
    for (size_t at = sizeof *begin; at < last;) {
        size_t remaining = last - at;
        if (remaining < sizeof(op_hdr)) return 0;
        const op_hdr *h = (const op_hdr *)(commands + at);
        if (h->size < sizeof *h || (h->size & 7) || h->size > remaining || h->op >= OP_COUNT_ ||
            h->op == OP_BEGIN || h->op == OP_END || h->op == OP_JUMP || h->op == OP_CALL) return 0;
        if (h->op == OP_BIND_UBO) {
            if (h->size != bind_bytes) return 0;
            project_binding(state, nvn_op_payload(h));
        }
        at += h->size;
    }
    return 1;
}

int vk_usnap_project(const uint64_t *handles, int n, VkUsnapBindingState *candidate) {
    if (!candidate || n < 0 || (n && !handles)) return 0;
    VkUsnapBindingState projected = *candidate;
    for (int i = 0; i < n; i++)
        if (!project_handle(&projected, handles[i])) return 0;
    *candidate = projected;
    return 1;
}

static int by_lo(const void *a, const void *b) {
    uint64_t x = ((const Rng *)a)->lo, y = ((const Rng *)b)->lo;
    return x < y ? -1 : x > y;
}
// Sorts the collected ranges by lo. A town frame binds about 10,000 uniform ranges and qsort took half a millisecond
// of the copy's 0.96 ms on the game thread, so they go through an LSD radix sort on (lo - lowest) / alignment, three
// passes of 11 bits; qsort is left for a span that does not fit in 33 bits (2 TiB).
#define RADIX_BITS 11
static VkResult sort_col(RcAdmissionVector *peak) {
    int n = U.ncol;
    if (n < 2) return VK_SUCCESS;
    uint64_t alignment = U.gather_alignment ? U.gather_alignment : ALIGN;
    uint64_t lo = UINT64_MAX, hi = 0;
    for (int i = 0; i < n; i++) {
        if (U.col[i].lo < lo) lo = U.col[i].lo;
        if (U.col[i].lo > hi) hi = U.col[i].lo;
    }
    if ((hi - lo) / alignment >> (3 * RADIX_BITS)) {
        qsort(U.col, (size_t)n, sizeof *U.col, by_lo);
        return VK_SUCCESS;
    }
    if (U.sort_cap < n) {
        vk_replay_metadata_free(U.sort_tmp, &U.sort_budget);
        U.sort_tmp = NULL;
        U.sort_cap = 0;
        size_t bytes = (size_t)U.ccol * sizeof *U.sort_tmp;
        void *allocation;
        VkResult result = metadata_allocate(bytes, 0, NULL, &U.sort_budget, &allocation);
        if (result == VK_NOT_READY) {
            // Sorting can finish without extra capacity when the radix workspace does not fit.
            qsort(U.col, (size_t)n, sizeof *U.col, by_lo);
            return VK_SUCCESS;
        }
        if (result != VK_SUCCESS) {
            *peak = arena_host_cost(bytes);
            return result;
        }
        U.sort_tmp = allocation;
        U.sort_cap = U.ccol;
    }
    Rng *src = U.col, *dst = U.sort_tmp;
    for (int pass = 0; pass < 3; pass++) {
        int shift = pass * RADIX_BITS, count[1 << RADIX_BITS] = {0};
        for (int i = 0; i < n; i++) count[(src[i].lo - lo) / alignment >> shift & ((1 << RADIX_BITS) - 1)]++;
        for (int k = 0, sum = 0; k < 1 << RADIX_BITS; k++) {
            int c = count[k];
            count[k] = sum, sum += c;
        }
        for (int i = 0; i < n; i++) dst[count[(src[i].lo - lo) / alignment >> shift & ((1 << RADIX_BITS) - 1)]++] = src[i];
        Rng *t = src;
        src = dst, dst = t;
    }
    if (src != U.col) memcpy(U.col, src, (size_t)n * sizeof *src);   // three passes end in tmp
    return VK_SUCCESS;
}

static void summary(void) {
    if (U.subs + U.off_subs < 3600) return;
    double s = U.subs ? (double)U.subs : 1.0;
    rt_log("vk_usnap: %llu submissions with a snapshot, %llu without; per submission %.0f ranges, %.1f KiB copied in "
           "%.3f ms, %.0f uniform bindings from the copy, %.1f ranges without qualified copies",
           (unsigned long long)U.subs, (unsigned long long)U.off_subs, U.regions / s, U.bytes / s / 1024.0,
           U.copy_ns / s / 1e6, U.hits / s, U.refused / s);
    U.subs = U.off_subs = U.regions = U.bytes = U.copy_ns = U.hits = U.refused = 0;
}

// copy_regions's work, sliced for the replay workers (vk_exec_par): regions of the game's memory into the arena, then
// the host copy into the device copy in COPY_SLICE pieces.
#define COPY_SLICE (64u << 10)
typedef struct { uint8_t *dst, *map; Reg *reg; uint64_t total, alignment; } CopyJob;
static void copy_some(void *arg, int lo, int hi) {
    const CopyJob *j = arg;
    for (int i = lo; i < hi; i++) {
        Reg *r = &j->reg[i];
        if (r->unavailable) continue;
        uint64_t size = r->hi - r->lo;
        uint64_t padded;
        if (!align_up(size, j->alignment, &padded)) rt_fatal("vk_usnap: copied range alignment overflow");
        uint8_t *dst = j->dst + r->off;
        RcRegistryResult qualified = vk_mem_canonical_source_qualify_expected(r->mapping, r->import);
        int held = qualified == RC_REGISTRY_OK && mem_backing_read_hold(r->mapping);
        if (held) {
            memcpy(dst, (const void *)(uintptr_t)r->lo, size);
            mem_read_release();
            memset(dst + size, 0, padded - size);
        } else {
            r->unavailable = qualified == RC_REGISTRY_STALE || !mem_backing_current(r->mapping) ?
                VK_USNAP_RESOLVE_MAPPING_STALE : VK_USNAP_RESOLVE_UNAVAILABLE;
            memset(dst, 0, padded);
        }
    }
}

static void copy_down(void *arg, int lo, int hi) {
    const CopyJob *j = arg;
    uint64_t a = (uint64_t)lo * COPY_SLICE, b = (uint64_t)hi * COPY_SLICE;
    if (b > j->total) b = j->total;
    memcpy(j->map + a, j->dst + a, b - a);
}

// Synchronous replay can split the copy over its workers. Each source stays mapped during memcpy.
static VkResult copy_regions(VkUsnapCapture *t, int par) {
    Arena *a = t->arena;
    if (t->state != ARENA_RESERVED || a->state != ARENA_RESERVED)
        rt_fatal("vk_usnap: copying an arena still owned by another consumer");
    uint64_t start = kern_now_ns();
    uint8_t *dst = a->map;
    if (a->cpu_needed) {
        if (!a->cpu) rt_fatal("vk_usnap: CPU snapshot was not admitted before capture");
        dst = a->cpu;
        t->owners = motion_owner_acquire();
    }
    RcRegistryResult observed = rc_registry_gpu_capture_start(rc_registry_runtime(), t->producer.read);
    if (observed != RC_REGISTRY_OK) return producer_failure(observed);
    t->state = a->state = ARENA_CAPTURING;
    CopyJob j = {dst, a->map, t->reg, t->total, t->alignment};
    if (par) vk_exec_par(copy_some, &j, t->nreg, 48);
    else copy_some(&j, 0, t->nreg);
    if (dst != a->map) {
        int slices = (int)((t->total + COPY_SLICE - 1) / COPY_SLICE);
        if (par) vk_exec_par(copy_down, &j, slices, 2);
        else copy_down(&j, 0, slices);
    }
    VkResult result = vmaFlushAllocation(vk_ctx()->vma, a->alloc, 0, t->total);
    if (result != VK_SUCCESS) {
        if (result == VK_ERROR_DEVICE_LOST)
            vk_device_lost_check(result, "flushing a uniform snapshot");
        // The copied bytes have not been published to a consumer. Return both owners
        // to the cancellable reservation state so queue/preflight failure cleanup
        // can release the capture and return the reusable arena exactly once.
        t->state = a->state = ARENA_RESERVED;
        a->cpu_current = 0;
        return result;
    }
    a->cpu_current = dst == a->cpu;
    t->ns += kern_now_ns() - start;
    return VK_SUCCESS;
}

static int snapshot_enabled(void) {
    int mode = __atomic_load_n(&U.mode, __ATOMIC_RELAXED);
    if (!mode) {
        int configured = rt_knob("UBO_SNAPSHOT", "1") + 1;
        if (__atomic_compare_exchange_n(&U.mode, &mode, configured, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            mode = configured;
    }
    return RT_KNOB_ON(mode - 1);
}

static int capture_group_end(int first, int limit, int merge_touching, uint64_t *hi) {
    *hi = U.col[first].hi;
    int i = first + 1;
    for (; i < limit; i++) {
        uint64_t lo = U.col[i].lo;
        if (lo > *hi || (!merge_touching && lo == *hi)) break;
        if (U.col[i].hi > *hi) *hi = U.col[i].hi;
    }
    return i;
}

static VkResult capture_region(VkUsnapCapture *t, uint64_t lo, uint64_t hi, int keep_refused, int replan,
                               RcAdmissionVector *peak, SourceRetainResult *source) {
    Reg region = {.lo = lo, .hi = hi};
    ReceiptPlan plan = {0};
    SourceRetainResult retained = source_receipt_plan(&region, &plan);
    if (retained == SOURCE_PLAN_CHANGED) {
        if (replan) return VK_INCOMPLETE;
        retained = SOURCE_MAPPING_STALE;
    }
    *source = retained;
    if (retained != SOURCE_RETAINED && (retained == SOURCE_NO_CPU_POOL || !keep_refused)) return VK_SUCCESS;
    if (t->nreg == t->region_cap) {
        int capacity = metadata_next_capacity(t->region_cap, sizeof *t->reg);
        void *allocation;
        VkResult result = capacity ? metadata_resize(t->reg, (size_t)t->nreg * sizeof *t->reg,
            (size_t)capacity * sizeof *t->reg, &t->region_budget, peak, &allocation) : VK_ERROR_OUT_OF_HOST_MEMORY;
        if (result != VK_SUCCESS) return result;
        t->reg = allocation;
        t->region_cap = capacity;
    }
    if (t->nreg == t->receipt_cap) {
        int capacity = t->receipt_cap ? metadata_next_capacity(t->receipt_cap, sizeof *t->receipts) : 64;
        void *allocation;
        VkResult result = capacity ? metadata_resize(t->receipts, (size_t)t->nreg * sizeof *t->receipts,
            (size_t)capacity * sizeof *t->receipts, &t->receipt_plan_budget, peak, &allocation)
            : VK_ERROR_OUT_OF_HOST_MEMORY;
        if (result != VK_SUCCESS) return result;
        t->receipts = allocation;
        t->receipt_cap = capacity;
    }
    if (t->nreg && lo < t->reg[t->nreg - 1].hi) t->overlapping = 1;
    if (retained != SOURCE_RETAINED) {
        region.unavailable = retained == SOURCE_NO_CAPACITY ? VK_USNAP_RESOLVE_NO_CAPACITY :
            retained == SOURCE_MAPPING_STALE ? VK_USNAP_RESOLVE_MAPPING_STALE : VK_USNAP_RESOLVE_UNAVAILABLE;
    } else {
        size_t bytes;
        if (!source_receipt_bytes(&plan, &bytes) || bytes > SIZE_MAX - t->receipt_peak)
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        t->receipt_peak += bytes;
        if (plan.host.count > t->permission_capacity) t->permission_capacity = plan.host.count;
    }
    t->receipts[t->nreg] = plan;
    t->reg[t->nreg++] = region;
    return VK_SUCCESS;
}

// An invalid crossing range can connect independent bindings on either side of a pool boundary or hole.
// Walk the original candidates. Only a full exact lookup establishes containment in a logical pool.
static VkResult capture_pool_pieces(VkUsnapCapture *t, int first, int end, int replan, RcAdmissionVector *peak) {
    int merge = 1;
    for (int at = first, next; at < end; at = next) {
        Rng input = U.col[at];
        next = at + 1;
        if (t->nreg && input.hi <= t->reg[t->nreg - 1].hi) continue;
        VkMemSpan span;
        if (!cpu_pool_span(input.lo, input.hi - input.lo, &span)) {
            t->refused++;
            continue;
        }
        uint64_t hi = input.hi;
        for (; merge && next < end && U.col[next].lo <= hi; next++) {
            Rng candidate = U.col[next];
            if (candidate.hi <= hi) continue;
            if (cpu_pool_span(input.lo, candidate.hi - input.lo, &span)) hi = candidate.hi;
            else if (cpu_pool_span(candidate.lo, candidate.hi - candidate.lo, &span)) break;
        }
        SourceRetainResult retained;
        VkResult result = capture_region(t, input.lo, hi, 0, replan, peak, &retained);
        if (result != VK_SUCCESS) return result;
        if (retained != SOURCE_RETAINED) {
            // A proof failure for the union must not prevent trying the individual bindings.
            next = at + 1;
            merge = 0;
            if (hi != input.hi) {
                result = capture_region(t, input.lo, input.hi, 0, replan, peak, &retained);
                if (result != VK_SUCCESS) return result;
            }
        }
        if (retained != SOURCE_RETAINED) t->refused++;
    }
    return VK_SUCCESS;
}

// Async replay also carries an explicit disabled choice so a later A/B phase cannot change it.
static VkResult capture_inventory_plan(const VkUsnapInputs *inputs, int retain_disabled, int replan,
                                       VkUsnapCapture **out, RcAdmissionVector *peak) {
    if (inputs->nhandles <= 0) {
        size_t i = 0;
        while (i < inputs->ninherited && (!inputs->inherited[i].addr || !inputs->inherited[i].size))
            i++;
        if (i == inputs->ninherited) return VK_SUCCESS;
    }
    int enabled = snapshot_enabled();
    if (!enabled && !retain_disabled) return VK_SUCCESS;
    uint64_t start = kern_now_ns();
    RcBudgetTicket header = {0};
    void *allocation;
    VkResult result = metadata_allocate(sizeof(VkUsnapCapture), 1, NULL, &header, &allocation);
    if (result != VK_SUCCESS) {
        *peak = arena_host_cost(sizeof(VkUsnapCapture));
        return result;
    }
    VkUsnapCapture *t = *out = allocation;
    t->header_budget = header;
    t->id = __atomic_add_fetch(&next_capture_id, 1, __ATOMIC_RELAXED);
    if (!t->id) rt_fatal("vk_usnap: capture identity exhausted");
    t->state = ARENA_RESERVED;
    t->enabled = enabled;
    if (!enabled) return VK_SUCCESS;
    Rng *discard_col = NULL, *discard_sort = NULL;
    RcBudgetTicket discard_col_budget = {0}, discard_sort_budget = {0};
    RP_CALL(RP_USNAP_GATHER_LOCK, plat_mutex_lock(&gather_lock));
    {
        RP_SCOPE(rp_gather, RP_USNAP_GATHER);
        t->alignment = U.gather_alignment = snapshot_alignment();
        U.ncol = 0;
        for (size_t i = 0; result == VK_SUCCESS && i < inputs->ninherited; i++)
            result = collect_binding(&inputs->inherited[i], peak);
        for (int i = 0; result == VK_SUCCESS && i < inputs->nhandles; i++)
            if (inputs->handles[i]) result = collect(nvn_op_first(inputs->handles[i]), 0, peak);
        RP_VALUE(RPK_REGIONS_IN, U.ncol);
    }
    if (result == VK_NOT_READY && U.col) {
        // Collection growth was refused. No consumer can use this partial inventory.
        // Rebuild directly at the refused size after pressure service observes the released buffers.
        U.gather_retry_capacity = metadata_next_capacity(U.ccol, sizeof *U.col);
        discard_col = U.col;
        discard_col_budget = U.col_budget;
        U.col = NULL;
        U.col_budget = (RcBudgetTicket){0};
        U.ncol = U.ccol = 0;
        discard_sort = U.sort_tmp;
        discard_sort_budget = U.sort_budget;
        U.sort_tmp = NULL;
        U.sort_budget = (RcBudgetTicket){0};
        U.sort_cap = 0;
    }
    if (result == VK_SUCCESS) result = RP_CALL(RP_USNAP_SORT, sort_col(peak));
    RP_SCOPE(rp_validate, RP_USNAP_VALIDATE);
    for (int first = 0, end; result == VK_SUCCESS && first < U.ncol; first = end) {
        uint64_t hi;
        end = capture_group_end(first, U.ncol, 1, &hi);
        SourceRetainResult retained;
        result = capture_region(t, U.col[first].lo, hi, end == first + 1, replan, peak, &retained);
        if (result != VK_SUCCESS) break;
        if (retained == SOURCE_RETAINED || end == first + 1) {
            t->refused += retained != SOURCE_RETAINED;
            continue;
        }
        result = capture_pool_pieces(t, first, end, replan, peak);
    }
    plat_mutex_unlock(&gather_lock);
    vk_replay_metadata_free(discard_col, &discard_col_budget);
    vk_replay_metadata_free(discard_sort, &discard_sort_budget);
    t->ns = kern_now_ns() - start;
    return result;
}

static void capture_free_receipt_plan(VkUsnapCapture *t) {
    vk_replay_metadata_free(t->receipts, &t->receipt_plan_budget);
    t->receipts = NULL;
    t->receipt_cap = 0;
}

static VkResult capture_fill_receipts(VkUsnapCapture *t, int replan, RcAdmissionVector *peak) {
    if (!t) return VK_SUCCESS;
    if (!t->receipt_peak) {
        capture_free_receipt_plan(t);
        return VK_SUCCESS;
    }
    uint64_t scratch_peak = permission_scratch_bytes(t->permission_capacity);
    if (scratch_peak > SIZE_MAX - t->receipt_peak) return VK_ERROR_OUT_OF_HOST_MEMORY;
    RcAdmissionVector cost = arena_host_cost(t->receipt_peak + scratch_peak);
    RcAdmissionStatus admitted = resource_core_budget_reserve(RC_BUDGET_SNAPSHOT, 1, &cost, &t->receipt_budget);
    if (admitted != RC_ADMISSION_OK) {
        *peak = cost;
        return arena_admission_failure(admitted);
    }

    uint64_t start = kern_now_ns();
    VkResult result = VK_SUCCESS;
    uint32_t permission_capacity = 0;
    for (int i = 0; i < t->nreg; i++) {
        Reg *region = &t->reg[i];
        if (region->unavailable) continue;
        const ReceiptPlan *plan = &t->receipts[i];
        result = source_receipt_fill(region, plan);
        if (result == VK_INCOMPLETE && !replan) {
            // One fresh inventory has already been attempted. Fail closed for this unstable source.
            region->unavailable = VK_USNAP_RESOLVE_MAPPING_STALE;
            t->refused++;
            result = VK_SUCCESS;
            continue;
        }
        if (result != VK_SUCCESS) break;
        t->receipt_bytes += plan->host.bytes + plan->import.bytes;
        if (plan->host.count > permission_capacity) permission_capacity = plan->host.count;
        uint64_t padded;
        if (!align_up(region->hi - region->lo, t->alignment, &padded) || padded > SIZE_MAX - t->total) {
            result = VK_ERROR_OUT_OF_HOST_MEMORY;
            break;
        }
        region->off = t->total;
        t->total += padded;
    }
    uint64_t scratch_bytes = result == VK_SUCCESS ? permission_scratch_bytes(permission_capacity) : 0;
    t->permission_capacity = 0;
    if (scratch_bytes) {
        t->permission_ranges = malloc((size_t)scratch_bytes);
        if (t->permission_ranges) {
            metadata_touch_pages(t->permission_ranges, (size_t)scratch_bytes);
            t->permission_capacity = permission_capacity;
        } else {
            scratch_bytes = 0;
            result = VK_ERROR_OUT_OF_HOST_MEMORY;
        }
    }
    if (t->receipt_bytes) {
        RcAdmissionVector actual = arena_host_cost(t->receipt_bytes + scratch_bytes);
        if (!rc_admission_accounted(resource_core_budget_materialize(&t->receipt_budget, &actual)))
            rt_fatal("vk_usnap: source receipt capacity settlement failed");
    } else if (resource_core_budget_cancel(&t->receipt_budget) != RC_ADMISSION_OK) {
        rt_fatal("vk_usnap: source receipt reservation cancellation failed");
    }
    capture_free_receipt_plan(t);
    t->ns += kern_now_ns() - start;
    return result;
}

static void capture_dispose(VkUsnapCapture *t) {
    if (!t) return;
    producer_end(&t->producer);
    if (t->arena) arena_return(t->arena, t->state);
    motion_owner_release(t->owners);
    capture_free_receipt_plan(t);
    if (t->reg) regions_release(t->reg, t->nreg, &t->region_budget, t->permission_ranges, &t->receipt_budget);
    RcBudgetTicket header = t->header_budget;
    vk_replay_metadata_free(t, &header);
}

static VkResult capture_inventory(const VkUsnapInputs *inputs, int retain_disabled, VkUsnapCapture **out,
                                  RcAdmissionVector *peak) {
    VkResult result = VK_INCOMPLETE;
    for (int replan = 1; replan >= 0; replan--) {
        result = capture_inventory_plan(inputs, retain_disabled, replan, out, peak);
        if (result == VK_SUCCESS) result = capture_fill_receipts(*out, replan, peak);
        if (result != VK_INCOMPLETE) break;
        capture_dispose(*out);
        *out = NULL;
    }
    if (result == VK_INCOMPLETE) return VK_ERROR_INITIALIZATION_FAILED;
    if (*out) {
        RP_VALUE(RPK_REGIONS_KEPT, (*out)->nreg);
        RP_VALUE(RPK_BYTES, (*out)->total);
        RP_VALUE(RPK_REFUSED, (*out)->refused);
    }
    return result;
}

// A budget retry rebuilds this attempt from the current inputs. Include metadata already held when the
// remaining peak was refused. Shared inventory scratch and a reused arena stay charged across attempts.
static VkResult capture_budget_failure(const VkUsnapCapture *t, VkReplayCaptureWait *wait) {
    wait->kind = VK_REPLAY_CAPTURE_WAIT_BUDGET;
    if (!t) return VK_NOT_READY;
    RcAdmissionVector header = arena_host_cost(sizeof *t);
    RcAdmissionVector regions = arena_host_cost((size_t)t->region_cap * sizeof *t->reg);
    RcAdmissionVector coverage = arena_host_cost(t->producer.coverage_bytes);
    RcAdmissionVector plans = arena_host_cost((size_t)t->receipt_cap * sizeof *t->receipts);
    uint64_t scratch_bytes = t->permission_ranges ? permission_scratch_bytes(t->permission_capacity) : 0;
    RcAdmissionVector receipts = arena_host_cost(t->receipt_bytes + scratch_bytes);
    if (!peak_sum(&wait->peak, &wait->peak, &header) ||
        !peak_sum(&wait->peak, &wait->peak, &regions) ||
        !peak_sum(&wait->peak, &wait->peak, &coverage) ||
        !peak_sum(&wait->peak, &wait->peak, &plans) ||
        !peak_sum(&wait->peak, &wait->peak, &receipts)) return VK_ERROR_OUT_OF_HOST_MEMORY;
    return VK_NOT_READY;
}

static VkResult capture_prepare_producer(VkUsnapCapture *t, int queued, VkReplayCaptureWait *wait) {
    if (!t || !t->total) return VK_SUCCESS;
    // The scene snapshot and one serial host-input borrow per recording thread can overlap.
    uint32_t foreground_slots = queued ? VK_REPLAY_WORKER_LIMIT + 2 : 0;
    VkResult metadata = VK_SUCCESS;
    RcRegistryResult result = producer_reserve(&t->producer, t->reg, t->nreg, foreground_slots,
                                              &wait->producer_epoch, &metadata, &wait->peak);
    if (metadata != VK_SUCCESS) return metadata;
    if (result == RC_REGISTRY_OK) return VK_SUCCESS;
    if (result != RC_REGISTRY_BUSY) return producer_failure(result);
    wait->kind = VK_REPLAY_CAPTURE_WAIT_PRODUCER;
    return VK_NOT_READY;
}

void vk_usnap_wait_producer(uint64_t producer_epoch, uint32_t milliseconds) {
    rc_registry_gpu_capture_wait(rc_registry_runtime(), producer_epoch, milliseconds);
}

static VkResult capture_queue(VkUsnapCapture *t) {
    if (!t) return VK_SUCCESS;
    if (t->arena) {
        VkResult result = RP_CALL(RP_USNAP_COPY, copy_regions(t, 0));
        if (result != VK_SUCCESS) return result;
        t->arena->state = ARENA_QUEUED;
    }
    t->state = ARENA_QUEUED;
    return VK_SUCCESS;
}

void vk_usnap_queue_bundle_cancel(VkReplayCaptureBundle *bundle) {
    VkReplayCaptureBundle owned = *bundle;
    *bundle = (VkReplayCaptureBundle){0};
    vk_usnap_cancel(owned.snapshot);
    if (owned.scratch) vk_exec_draw_update_credit_cancel(owned.scratch);
}

VkResult vk_usnap_queue_bundle(const VkUsnapInputs *inputs, uint64_t max_update_bytes,
                              VkReplayCaptureBundle *out, VkReplayCaptureWait *wait) {
    if (out->snapshot || out->scratch) rt_fatal("vk_usnap: queue bundle already owns captures");
    *wait = (VkReplayCaptureWait){0};
    if (vk_device_is_lost()) return VK_ERROR_DEVICE_LOST;
    if (vk_sync_stopping()) return VK_ERROR_INITIALIZATION_FAILED;
    resource_core_budget_readiness(1, &wait->before);

    VkReplayCaptureBundle owned = {0};
    QueuedArenaPlan plan = {0};
    RcAdmissionVector scratch_peak = {{0}};
    RcBudgetTicket joint = {0};
    VkResult result = VK_SUCCESS;
    if (max_update_bytes) {
        int drain_older = 0;
        result = vk_exec_draw_update_credit_plan(max_update_bytes, &owned.scratch, &scratch_peak, &drain_older);
        wait->peak = scratch_peak;
        if (result == VK_SUCCESS && drain_older) result = VK_NOT_READY;
        if (result == VK_NOT_READY && drain_older) wait->kind = VK_REPLAY_CAPTURE_WAIT_DRAIN_OLDER;
    }
    if (result == VK_SUCCESS) {
        result = capture_inventory(inputs, 1, &owned.snapshot, &wait->peak);
        if (result == VK_SUCCESS) result = capture_prepare_producer(owned.snapshot, 1, wait);
    }
    if (result == VK_SUCCESS) {
        if (max_update_bytes) {
            RcBudgetReadiness current;
            resource_core_budget_readiness(1, &current);
            result = arena_queue_plan(owned.snapshot, &plan, &scratch_peak, &current, &wait->peak);
        } else if (owned.snapshot && owned.snapshot->total) {
            // A standalone request can take compatible existing VMA backing.
            owned.snapshot->arena = arena_try_get(owned.snapshot->total, &result, &wait->peak);
        }
    }
    if (result == VK_SUCCESS && max_update_bytes) {
        int needs_growth = 0;
        for (unsigned d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) needs_growth |= wait->peak.amount[d] != 0;
        int compatible_arena = 0;
        if (needs_growth) {
            RcAdmissionStatus admitted = resource_core_budget_reserve(RC_BUDGET_SNAPSHOT, 1, &wait->peak, &joint);
            if (admitted != RC_ADMISSION_OK) result = arena_admission_failure(admitted);
            if (admitted == RC_ADMISSION_REFUSED && owned.snapshot &&
                arena_reserve_smaller_shadow(owned.snapshot->arena, owned.snapshot->total, &scratch_peak,
                                             &plan, &wait->peak, &joint)) {
                arena_return(owned.snapshot->arena, ARENA_RESERVED);
                owned.snapshot->arena = NULL;
                result = VK_SUCCESS;
            } else if ((admitted == RC_ADMISSION_REFUSED || admitted == RC_ADMISSION_UNSUPPORTED) &&
                       owned.snapshot && owned.snapshot->total && !owned.snapshot->arena) {
                VkResult reuse = arena_queue_try_compatible(owned.snapshot, &plan, &scratch_peak, &wait->peak, &joint);
                if (reuse != VK_INCOMPLETE) {
                    result = reuse;
                    compatible_arena = result == VK_SUCCESS;
                }
            }
        }
        if (result == VK_SUCCESS && owned.scratch) result = vk_exec_draw_update_credit_materialize(owned.scratch, &joint);
        if (result == VK_SUCCESS && !compatible_arena) result = arena_queue_materialize(owned.snapshot, &plan, &joint);
        if (result != VK_SUCCESS && compatible_arena) {
            Arena *a = owned.snapshot->arena;
            owned.snapshot->arena = NULL;
            arena_retire(a, ARENA_RESERVED);
            arena_destroy(a);
        }
    }
    if (joint.token.ledger && resource_core_budget_cancel(&joint) != RC_ADMISSION_OK)
        rt_fatal("vk_usnap: queue bundle lost its joint capacity reservation");
    if (result == VK_SUCCESS && vk_device_is_lost()) result = VK_ERROR_DEVICE_LOST;
    if (result == VK_SUCCESS && vk_sync_stopping()) result = VK_ERROR_INITIALIZATION_FAILED;
    if (result == VK_SUCCESS) result = capture_queue(owned.snapshot);
    if (result == VK_NOT_READY && !wait->kind) result = capture_budget_failure(owned.snapshot, wait);
    if (result == VK_NOT_READY && wait->kind == VK_REPLAY_CAPTURE_WAIT_BUDGET &&
        !vk_replay_metadata_peak_add(&wait->peak, vk_exec_draw_update_credit_host_bytes(owned.scratch)))
        result = VK_ERROR_OUT_OF_HOST_MEMORY;
    if (result == VK_SUCCESS) *out = owned;
    else vk_usnap_queue_bundle_cancel(&owned);
    return result;
}

VkUsnapCapture *vk_usnap_take(const VkUsnapInputs *inputs) {
    int retried_clear = 0;
    for (;;) {
        if (vk_sync_stopping() || vk_device_is_lost()) return NULL;
        VkReplayCaptureBundle owned = {0};
        VkReplayCaptureWait wait;
        VkResult result = vk_usnap_queue_bundle(inputs, 0, &owned, &wait);
        if (result == VK_SUCCESS) return owned.snapshot;
        if (vk_sync_stopping() || vk_device_is_lost()) return NULL;
        if (result != VK_NOT_READY)
            rt_fatal("vk_usnap: required snapshot preparation failed (%d)", (int)result);
        if (wait.kind == VK_REPLAY_CAPTURE_WAIT_PRODUCER) {
            vk_usnap_wait_producer(wait.producer_epoch, 16);
            continue;
        }
        vk_queue_lock();
        vk_queue_unlock();
        RcBudgetProgress progress;
        resource_core_budget_service(1, &wait.peak, &progress);
        if (!progress.blocked_mask && !retried_clear) retried_clear = 1;
        else {
            resource_core_budget_wait_since(progress.event, 16);
            retried_clear = 0;
        }
    }
}

// Runs before recording, while the submission inventory is serialized. The submission owner handles
// pressure waits outside its lock and retries preflight against the current inventory.
VkResult vk_usnap_preflight(const VkUsnapInputs *inputs, VkUsnapCapture **capture, VkReplayCaptureWait *wait) {
    *wait = (VkReplayCaptureWait){0};
    if (*capture) {
        if ((*capture)->state != ARENA_QUEUED && (*capture)->state != ARENA_RESERVED)
            rt_fatal("vk_usnap: preflight received a capture already being consumed");
        return VK_SUCCESS;
    }
    VkUsnapCapture *t = NULL;
    VkResult result = capture_inventory(inputs, 0, &t, &wait->peak);
    if (result == VK_SUCCESS) result = capture_prepare_producer(t, 0, wait);
    if (result == VK_SUCCESS && t && t->total) {
        t->arena = RP_CALL(RP_USNAP_ARENA, arena_try_get(t->total, &result, &wait->peak));
    }
    if (result == VK_NOT_READY && !wait->kind) result = capture_budget_failure(t, wait);
    if (result == VK_SUCCESS) *capture = t;
    else capture_dispose(t);
    return result;
}

void vk_usnap_cancel(VkUsnapCapture *capture) {
    if (capture && capture->state != ARENA_RESERVED && capture->state != ARENA_QUEUED)
        rt_fatal("vk_usnap: canceling a capture already being consumed");
    capture_dispose(capture);
}

VkResult vk_usnap_begin(VkUsnapCapture **capture, int n, uint64_t fence) {
    RP_SCOPE(rp_snapshot, RP_USNAP_BEGIN);
    RP_VALUE(RPK_HANDLES, n);
    (void)n;
    VkUsnapCapture *t = *capture;
    *capture = NULL;
    if (U.arena || U.on || U.expected || U.owners || U.reg || U.permission_ranges || U.producer.read.id)
        rt_fatal("vk_usnap: previous recording still owns its capture");
    U.fence = fence;
    U.gen++;
    U.capture = 0;
    U.nreg = 0;
    U.overlapping = 0;
    summary();

    int queued = t && t->state == ARENA_QUEUED;
    if (t && !queued && t->state != ARENA_RESERVED)
        rt_fatal("vk_usnap: replay received a capture already being consumed");
    if (t && t->arena && !queued) {
        VkResult result = RP_CALL(RP_USNAP_COPY, copy_regions(t, 1));
        if (result != VK_SUCCESS) {
            capture_dispose(t);
            return result;
        }
    }
    U.expected = t && t->enabled;
    if (!t || !t->nreg) {
        U.off_subs += !t || !t->enabled;
        RP_VALUE(RPK_OUTCOME, t && t->enabled ? RPO_NO_REGIONS : RPO_DISABLED);
        if (t) U.refused += t->refused;
        capture_dispose(t);
        return VK_SUCCESS;
    }

    RP_SCOPE(rp_adopt, RP_USNAP_ADOPT);
    U.refused += t->refused;
    for (int i = 0; i < t->nreg; i++) {
        if (t->reg[i].unavailable && t->reg[i].mapping) U.refused++;
    }
    U.nreg = t->nreg;
    U.overlapping = t->overlapping;
    if (U.nreg) {
        if (t->arena && t->arena->state != t->state) rt_fatal("vk_usnap: invalid recording capture owner");
        U.arena = t->arena;
        if (U.arena) U.arena->state = ARENA_RECORDING;
        t->arena = NULL;
        U.reg = t->reg;
        t->reg = NULL;
        U.region_budget = t->region_budget;
        t->region_budget = (RcBudgetTicket){0};
        U.receipt_budget = t->receipt_budget;
        t->receipt_budget = (RcBudgetTicket){0};
        U.permission_ranges = t->permission_ranges;
        U.permission_capacity = t->permission_capacity;
        t->permission_ranges = NULL;
        U.producer = t->producer;
        t->producer = (ProducerObservation){0};
        U.owners = t->owners;
        t->owners = NULL;
        U.capture = t->id;
        U.on = 1;
        U.subs++;
        U.regions += (uint64_t)U.nreg;
        U.bytes += t->total;
        U.copy_ns += t->ns;
    }
    RP_END(rp_adopt);
    RP_VALUE(RPK_OUTCOME, !U.on ? RPO_NO_REGIONS : queued ? RPO_ADOPTED : RPO_FRESH_COPY);
    capture_dispose(t);
    return VK_SUCCESS;
}

void vk_usnap_end(uint64_t fence) {
    if (U.fence != fence) rt_fatal("vk_usnap: recording fence changed before snapshot release");
    Arena *a = U.arena;
    U.arena = NULL;
    U.on = U.expected = 0;
    U.capture = 0;
    U.gen++;
    producer_end(&U.producer);
    regions_release(U.reg, U.nreg, &U.region_budget, U.permission_ranges, &U.receipt_budget);
    U.permission_ranges = NULL;
    U.permission_capacity = 0;
    U.reg = NULL;
    U.nreg = 0;
    U.overlapping = 0;
    motion_owner_release(U.owners), U.owners = NULL;
    if (a) {
        if (a->state != ARENA_RECORDING) rt_fatal("vk_usnap: recording lost its arena");
        // Final completion covers every split part. The sync owner also handles cancellation of an abandoned tail.
        a->state = ARENA_PENDING_COMPLETION;
        vk_completion_add_owned(fence, arena_done, arena_cancel, a);
    }
}

// First region whose end is above a.
static int first_above(uint64_t a) {
    int lo = 0, hi = U.nreg;
    while (lo < hi) {
        int m = lo + (hi - lo) / 2;
        if (U.reg[m].hi <= a) lo = m + 1;
        else hi = m;
    }
    return lo;
}

static RcRegistryResult region_permission(const Reg *r, uint64_t addr, uint64_t size, RcRegistryPosition read,
                                          RcRegistryGpuReadEvidence *evidence) {
    if (evidence) *evidence = RC_GPU_CAPTURE_NO_EVIDENCE;
    if (!U.producer.read.id) return RC_REGISTRY_CAPACITY;
    if (!rc_registry_position_valid(read)) return RC_REGISTRY_INVALID;
    if (vk_mem_canonical_source_qualify_expected(r->mapping, r->import) != RC_REGISTRY_OK)
        return RC_REGISTRY_STALE;
    uint32_t count;
    const RcInterval *map = mem_backing_intervals(r->mapping, &count);
    RcSourceRange local[PERMISSION_LOCAL_RANGES];
    RcSourceRange *ranges = local;
    uint32_t n = 0, first = 0;
    uint64_t end = addr + size;
    while (first < count && map[first].address + map[first].range.size <= addr) first++;
    for (uint32_t i = first; i < count && map[i].address < end; i++) n++;
    if (!n || (uint64_t)n * sizeof *ranges > SIZE_MAX) return RC_REGISTRY_INVALID;
    if (n > sizeof local / sizeof local[0]) {
        if (!U.permission_ranges || n > U.permission_capacity) return RC_REGISTRY_INVALID;
        plat_mutex_lock(&permission_lock);
        ranges = U.permission_ranges;
    }
    uint64_t covered = addr;
    for (uint32_t i = 0; i < n; i++) {
        const RcInterval *m = &map[first + i];
        uint64_t lo = m->address > addr ? m->address : addr;
        uint64_t hi = m->address + m->range.size;
        if (hi > end) hi = end;
        if (lo != covered || hi <= lo) break;
        ranges[i] = (RcSourceRange){m->range.backing, m->range.offset + (lo - m->address), hi - lo};
        covered = hi;
    }
    RcRegistryResult result = covered == end ?
        rc_registry_gpu_capture_read_evidence(rc_registry_runtime(), U.producer.read, ranges, n, read, evidence) :
        RC_REGISTRY_INVALID;
    if (ranges != local) plat_mutex_unlock(&permission_lock);
    if (!mem_backing_current_range(r->mapping, addr, size)) {
        if (evidence) *evidence = RC_GPU_CAPTURE_NO_EVIDENCE;
        return RC_REGISTRY_STALE;
    }
    if (result == RC_REGISTRY_STALE && evidence && *evidence == RC_GPU_CAPTURE_NO_EVIDENCE)
        return RC_REGISTRY_INVALID;
    return result;
}

VkUsnapResolveResult vk_usnap_resolve_at(uint64_t addr, uint64_t n, uint64_t align, RcRegistryPosition read,
                                         VkDescriptorBufferInfo *out, VkUsnapLease *lease) {
    if (lease) *lease = (VkUsnapLease){0};
    if (!addr || !n || n > UINT64_MAX - addr) return VK_USNAP_RESOLVE_UNAVAILABLE;
    if (!align) return VK_USNAP_RESOLVE_ALIGNMENT;
    if (!U.expected) return VK_USNAP_RESOLVE_NATIVE_ONLY;
    // Most captures have disjoint regions. Overlapping pool-contained pieces instead use the first
    // region whose end covers the entire request, so the selection is independent of a worker's hint.
    static _Thread_local int hint;   // per thread: the replay workers look up at once
    int i = hint;
    if (U.overlapping) {
        i = first_above(addr + n - 1);
        if (i >= U.nreg || addr < U.reg[i].lo) i = first_above(addr);
    } else if (i >= U.nreg || addr < U.reg[i].lo || addr >= U.reg[i].hi) {
        i = first_above(addr);
    }
    if (i >= U.nreg || addr + n <= U.reg[i].lo) {
        VkMemSpan span;
        if (nvn_gpu_only(addr) || (vk_mem_lookup(addr, n, &span) && (span.alloc || span.cls == VK_MCLS_DEVICE)))
            return VK_USNAP_RESOLVE_NATIVE_ONLY;
        return VK_USNAP_RESOLVE_UNAVAILABLE;
    }
    const Reg *r = &U.reg[i];
    if (addr < r->lo || addr + n > r->hi) return VK_USNAP_RESOLVE_UNAVAILABLE;
    if (r->unavailable) return r->unavailable;
    uint64_t off = r->off + (addr - r->lo);
    if (off % align) return VK_USNAP_RESOLVE_ALIGNMENT;
    RcRegistryGpuReadEvidence evidence;
    RcRegistryResult result = region_permission(r, addr, n, read, &evidence);
    if (result == RC_REGISTRY_CAPACITY) return VK_USNAP_RESOLVE_NO_CAPACITY;
    if (evidence == RC_GPU_CAPTURE_EARLIER_BUFFER) return VK_USNAP_RESOLVE_ORDERED_PRODUCER;
    if (evidence == RC_GPU_CAPTURE_EARLIER_VISIBLE) return VK_USNAP_RESOLVE_SOURCE_ADVANCED;
    if (result == RC_REGISTRY_STALE) return VK_USNAP_RESOLVE_MAPPING_STALE;
    if (result != RC_REGISTRY_OK || !U.arena) return VK_USNAP_RESOLVE_UNAVAILABLE;
    if (out) *out = (VkDescriptorBufferInfo){U.arena->buf, off, n};
    if (lease) *lease = (VkUsnapLease){U.capture, addr, n, align, (uint32_t)i, 0};
    __atomic_add_fetch(&U.hits, 1, __ATOMIC_RELAXED);
    hint = i;
    return VK_USNAP_RESOLVE_READY;
}

int vk_usnap_gpa_blocker(uint64_t address, uint64_t bytes, uint64_t align, RcRegistryPosition read,
                        VkGpaPrefixRequest *request) {
    if (!request || !U.expected || !U.arena || !U.producer.read.id || U.fence != read.scene ||
        !address || !bytes || !align ||
        bytes > UINT64_MAX - address || !rc_registry_position_valid(read)) return 0;
    uint64_t end = address + bytes;
    int index = first_above(U.overlapping ? end - 1 : address);
    if (U.overlapping && (index >= U.nreg || address < U.reg[index].lo)) index = first_above(address);
    if (index >= U.nreg) return 0;
    const Reg *region = &U.reg[index];
    if (region->unavailable || !region->mapping || address < region->lo || end > region->hi ||
        (region->off + (address - region->lo)) % align ||
        region_permission(region, address, bytes, read, NULL) != RC_REGISTRY_BLOCKED) return 0;

    uint32_t count;
    const RcInterval *mapping = mem_backing_intervals(region->mapping, &count);
    RcRegistry *registry = rc_registry_runtime();
    for (uint32_t i = 0; i < count; ++i) {
        const RcInterval *extent = &mapping[i];
        uint64_t begin = extent->address > address ? extent->address : address;
        uint64_t limit = extent->address + extent->range.size;
        if (limit > end) limit = end;
        if (begin >= limit) continue;
        RcInterval source = {begin, extent->mapping,
            {extent->range.backing, extent->range.offset + (begin - extent->address), limit - begin}};
        RcRegistryReadBlocker blocker;
        if (!source.mapping || !rc_registry_diagnose_read_blocker(registry, source.range, read, &blocker) ||
            blocker.guard_reason != RC_REGISTRY_GUARD_GPA_PROJECTION ||
            blocker.reason != RC_REGISTRY_DIAG_SOURCE_PENDING_BUFFER || blocker.sample.position.scene != read.scene ||
            rc_registry_position_compare(blocker.sample.position, read) >= 0) continue;
        if (vk_mem_canonical_source_qualify_expected(region->mapping, region->import) != RC_REGISTRY_OK ||
            !mem_backing_current_range(region->mapping, address, bytes)) return 0;
        *request = (VkGpaPrefixRequest){.guard = blocker.sample.writer, .producer = blocker.sample.position,
                                        .read = read, .source = source};
        return 1;
    }
    return 0;
}

int vk_usnap_find_lease_at(uint64_t addr, uint64_t n, uint64_t align, RcRegistryPosition read,
                          VkDescriptorBufferInfo *out, VkUsnapLease *lease) {
    return vk_usnap_resolve_at(addr, n, align, read, out, lease) == VK_USNAP_RESOLVE_READY;
}

RcRegistryResult vk_usnap_buffer_fragments_with_payloads(const VkMemSpanCapture *capture, const VkMemSpan *selected,
    uint64_t address, uint64_t size, RcRegistryPosition read, const VkMemBufferInlineProof *inline_proofs,
    uint32_t inline_count, RcRegistryBufferFragment *out, RcRegistryBufferPayload **payloads, uint32_t capacity,
    uint32_t *fragment_count) {
    if (fragment_count) *fragment_count = 0;
    if (!fragment_count || !capture || !selected || !size || size > UINT64_MAX - address ||
        !rc_registry_position_valid(read)) return RC_REGISTRY_INVALID;
    if (!U.expected || !U.arena || !U.producer.read.id || U.fence != read.scene)
        return RC_REGISTRY_UNSUPPORTED;
    if (rc_registry_position_compare(read, vk_exec_resource_position())) return RC_REGISTRY_STALE;
    if (payloads)
        return vk_mem_span_buffer_fragments_with_payloads(capture, selected, address, size, U.producer.read, read,
            inline_proofs, inline_count, out, payloads, capacity, fragment_count);
    return vk_mem_span_buffer_fragments(capture, selected, address, size, U.producer.read, read,
        inline_proofs, inline_count, out, capacity, fragment_count);
}

RcRegistryResult vk_usnap_buffer_fragments(const VkMemSpanCapture *capture, const VkMemSpan *selected,
    uint64_t address, uint64_t size, RcRegistryPosition read, const VkMemBufferInlineProof *inline_proofs,
    uint32_t inline_count, RcRegistryBufferFragment *out, uint32_t capacity, uint32_t *fragment_count) {
    return vk_usnap_buffer_fragments_with_payloads(capture, selected, address, size, read, inline_proofs,
        inline_count, out, NULL, capacity, fragment_count);
}

int vk_usnap_find_lease(uint64_t addr, uint64_t n, uint64_t align, VkDescriptorBufferInfo *out, VkUsnapLease *lease) {
    return vk_usnap_find_lease_at(addr, n, align, vk_exec_resource_position(), out, lease);
}

int vk_usnap_find(uint64_t addr, uint64_t n, uint64_t align, VkDescriptorBufferInfo *out) {
    return vk_usnap_find_lease(addr, n, align, out, NULL);
}

static const Reg *lease_region(const VkUsnapLease *lease) {
    if (!lease || !U.on || !lease->capture || lease->capture != U.capture || lease->region >= (uint32_t)U.nreg)
        return NULL;
    const Reg *r = &U.reg[lease->region];
    return lease->addr >= r->lo && lease->addr < r->hi && lease->size && lease->size <= r->hi - lease->addr &&
           lease->align && !((r->off + lease->addr - r->lo) % lease->align) ? r : NULL;
}

int vk_usnap_lease_valid_at(const VkUsnapLease *lease, RcRegistryPosition read) {
    const Reg *r = lease_region(lease);
    return r && !r->unavailable && region_permission(r, lease->addr, lease->size, read, NULL) == RC_REGISTRY_OK;
}

int vk_usnap_lease_valid(const VkUsnapLease *lease) {
    return vk_usnap_lease_valid_at(lease, vk_exec_resource_position());
}
uint64_t vk_usnap_capture(void) { return U.on ? U.capture : 0; }

// Keep the descriptor-proof API while prepared consumers move to ordered per-lease checks.
void vk_usnap_proof_begin(VkUsnapProof *p) { p->sealed = p->building = 0; }
int vk_usnap_proof_add(VkUsnapProof *p, const VkUsnapLease *lease) { (void)p; (void)lease; return 0; }
int vk_usnap_proof_end_into(VkUsnapProof *p, VkUsnapProof *dst) { (void)p; dst->sealed = 0; return 0; }
int vk_usnap_proof_end(VkUsnapProof *p) { return vk_usnap_proof_end_into(p, p); }
int vk_usnap_proof_valid(VkUsnapProof *p) { (void)p; return 0; }
int vk_usnap_proof_inherit(VkUsnapProof *p, const VkUsnapProof *prior) { (void)p; (void)prior; return 0; }

void vk_usnap_drop(uint64_t addr, uint64_t n) {
    // Existing descriptor memo users use this generation as a hint. The actual source
    // decision now comes exclusively from the ordered registry and captured mapping.
    if (U.on && n && n <= UINT64_MAX - addr) __atomic_add_fetch(&U.gen, 1, __ATOMIC_RELAXED);
}
uint64_t vk_usnap_gen(void) { return __atomic_load_n(&U.gen, __ATOMIC_RELAXED); }

static int cpu_written(uint64_t addr, uint64_t n) {   // the same test vk_usnap_begin keeps regions by
    VkMemSpan sp;
    return cpu_pool_span(addr, n, &sp);
}

typedef struct {
    ProducerObservation producer;
    Reg source;
    RcRegistryPosition read;
    RcBudgetTicket budget;
    void *scratch;
} FreshRead;

static int fresh_current(const FreshRead *source) {
    const ProducerObservation *p = &source->producer;
    return p->read.id && vk_mem_canonical_source_qualify_expected(source->source.mapping, source->source.import) ==
            RC_REGISTRY_OK &&
        rc_registry_gpu_capture_read_at(rc_registry_runtime(), p->read, p->coverage, p->count, source->read) ==
            RC_REGISTRY_OK && mem_backing_current(source->source.mapping);
}

static void fresh_release(FreshRead *source) {
    int retained = source->budget.token.state == RC_ADMISSION_RETAINED;
    if (retained && resource_core_budget_retire(&source->budget) != RC_ADMISSION_OK)
        rt_fatal("vk_usnap: fresh read capacity retirement failed");
    producer_end(&source->producer);
    source_release(&source->source);
    free(source->scratch);
    source->scratch = NULL;
    if (source->budget.token.ledger) {
        RcAdmissionStatus result = retained ? resource_core_budget_destroyed(&source->budget)
                                             : resource_core_budget_cancel(&source->budget);
        if (result != RC_ADMISSION_OK) rt_fatal("vk_usnap: fresh read capacity release failed");
    }
}

static int fresh_retain(FreshRead *source, size_t scratch_size, RcAdmissionVector *cost) {
    for (unsigned attempt = 0; attempt < 2; attempt++) {
        ReceiptPlan plan;
        SourceRetainResult planned = source_receipt_plan(&source->source, &plan);
        if (planned == SOURCE_PLAN_CHANGED) continue;
        if (planned != SOURCE_RETAINED) return 0;
        size_t bytes;
        uint64_t coverage = (uint64_t)plan.host.count * sizeof(RcSourceRange);
        if (!source_receipt_bytes(&plan, &bytes) || coverage > SIZE_MAX - bytes ||
            scratch_size > SIZE_MAX - bytes - coverage) return 0;
        *cost = arena_host_cost(bytes + coverage + scratch_size);
        if (resource_core_budget_reserve(RC_BUDGET_SNAPSHOT, 1, cost, &source->budget) != RC_ADMISSION_OK) return 0;
        VkResult result = source_receipt_fill(&source->source, &plan);
        if (result == VK_SUCCESS) return 1;
        fresh_release(source);
        if (result != VK_INCOMPLETE) return 0;
    }
    return 0;
}

static int fresh_begin(uint64_t addr, uint64_t n, size_t scratch_size, FreshRead *source) {
    source->read = vk_exec_resource_position();
    if (!n || n > UINT64_MAX - addr || !rc_registry_position_valid(source->read)) return 0;
    uint64_t end = addr + n;
    source->source = (Reg){.lo = addr, .hi = end};
    RcAdmissionVector cost;
    if (!fresh_retain(source, scratch_size, &cost)) return 0;
    for (int i = first_above(addr); i < U.nreg && U.reg[i].lo < end; i++) {
        const Reg *r = &U.reg[i];
        uint64_t lo = addr > r->lo ? addr : r->lo, hi = end < r->hi ? end : r->hi;
        // A stale producer can have become CPU-visible. An ambiguous old proof or a
        // remapped original source cannot supply the captured input's identity.
        RcRegistryGpuReadEvidence evidence;
        RcRegistryResult result = r->unavailable ? RC_REGISTRY_INVALID :
            region_permission(r, lo, hi - lo, source->read, &evidence);
        if (r->unavailable || (result != RC_REGISTRY_OK && evidence != RC_GPU_CAPTURE_EARLIER_VISIBLE)) {
            fresh_release(source);
            return 0;
        }
    }
    if (producer_reserve(&source->producer, &source->source, 1, 0, NULL, NULL, NULL) == RC_REGISTRY_OK &&
        rc_registry_gpu_capture_start(rc_registry_runtime(), source->producer.read) == RC_REGISTRY_OK &&
        fresh_current(source)) {
        if (scratch_size) {
            source->scratch = malloc(scratch_size);
            if (source->scratch) metadata_touch_pages(source->scratch, scratch_size);
        }
        if (!scratch_size || source->scratch) {
            if (!rc_admission_accounted(resource_core_budget_materialize(&source->budget, &cost)))
                rt_fatal("vk_usnap: fresh read capacity settlement failed");
            return 1;
        }
    }
    fresh_release(source);
    return 0;
}

int vk_usnap_read(uint64_t addr, uint64_t n, void *dst) {
    const uint8_t *snapshot = vk_usnap_bytes(addr, n);
    if (snapshot) { memcpy(dst, snapshot, n); return 1; }
    if (n > SIZE_MAX) return 0;
    FreshRead source = {0};
    uint8_t local[256];
    size_t scratch_size = n > sizeof local ? (size_t)n : 0;
    if (!fresh_begin(addr, n, scratch_size, &source)) return 0;
    void *tmp = scratch_size ? source.scratch : local;
    int ok = mem_backing_read_hold(source.source.mapping);
    if (ok) {
        memcpy(tmp, (const void *)(uintptr_t)addr, n);
        mem_read_release();
        ok = fresh_current(&source);
    }
    if (ok) memcpy(dst, tmp, n);
    fresh_release(&source);
    return ok ? 2 : 0;
}

int vk_usnap_equal_tail(uint64_t addr, uint64_t size, uint64_t skip, const void *expected) {
    if (skip >= size || size > SIZE_MAX) return 0;
    const uint8_t *snapshot = vk_usnap_bytes(addr, size);
    if (snapshot) return !memcmp(snapshot + skip, (const uint8_t *)expected + skip, size - skip);
    FreshRead source = {0};
    if (!fresh_begin(addr, size, 0, &source)) return 0;
    int equal = mem_backing_read_hold(source.source.mapping);
    if (equal) {
        equal = !memcmp((const void *)(uintptr_t)(addr + skip), (const uint8_t *)expected + skip, size - skip);
        mem_read_release();
        equal = equal && fresh_current(&source);
    }
    fresh_release(&source);
    return equal;
}

int vk_usnap_why(uint64_t addr, uint64_t n) {
    if (!U.on) return VK_USNAP_NO_SNAPSHOT;
    if (!n || n > UINT64_MAX - addr) return VK_USNAP_NOT_BOUND;
    int i = first_above(U.overlapping ? addr + n - 1 : addr);
    if (i < U.nreg && addr >= U.reg[i].lo && addr + n <= U.reg[i].hi) {
        const Reg *r = &U.reg[i];
        if (r->unavailable) return VK_USNAP_DROPPED;
        RcRegistryResult result = region_permission(r, addr, n, vk_exec_resource_position(), NULL);
        if (result == RC_REGISTRY_BLOCKED) return VK_USNAP_IN_FLIGHT;
        return result == RC_REGISTRY_OK && !r->unavailable ? VK_USNAP_FOUND : VK_USNAP_DROPPED;
    }
    return cpu_written(addr, n) ? VK_USNAP_NOT_BOUND : VK_USNAP_NOT_CPU;
}

// the host copy where there is one (DLSS), else the device copy
static uint8_t *host_copy(void) { return U.arena->cpu_current ? U.arena->cpu : U.arena->map; }

const uint8_t *vk_usnap_map(void) { return U.arena ? host_copy() : NULL; }

const uint8_t *vk_usnap_bytes(uint64_t addr, uint64_t n) {
    VkDescriptorBufferInfo bi;
    return vk_usnap_find(addr, n, 1, &bi) ? host_copy() + bi.offset : NULL;
}

// The caller drains or cancels submission owners before closing the arena pool.
void vk_usnap_shutdown(void) {
    if (U.arena || U.on || U.expected || U.owners || U.reg || U.permission_ranges || U.producer.read.id)
        rt_fatal("vk_usnap: shutdown while replay still owns a capture");
    motion_owner_shutdown();
    arena_shutdown();
    vk_replay_metadata_free(U.col, &U.col_budget);
    U.col = NULL;
    U.ncol = U.ccol = U.gather_retry_capacity = 0;
    vk_replay_metadata_free(U.sort_tmp, &U.sort_budget);
    U.sort_tmp = NULL;
    U.sort_cap = 0;
    vk_uniform_payload_reap();
}

// NX2NAVI_TRACE=perf "mem: sizes" (perf.c): reusable uniform snapshot arenas.
void vk_usnap_sizes(PerfSizes *s) {
    uint64_t n = 0, b = 0;
    plat_mutex_lock(&free_lock);
    for (const Arena *a = free_arenas; a; a = a->next) n++, b += a->size;
    plat_mutex_unlock(&free_lock);
    perf_size(s, "free uniform snapshot arenas", n, 0, b);
}

// The producer publishes sorted, disjoint ranges. Zero means this binding has no owner.
static unsigned motion_owner_slot(uint64_t address, uint64_t size) {
    unsigned lo = 0, hi = U.owners->count;
    if (!hi || address < U.owners->bindings[0].address) return 0;
    const MotionOwnerBinding *last = &U.owners->bindings[hi - 1];
    if (address >= last->address && address - last->address >= last->size) return 0;
    while (lo < hi) {
        unsigned mid = lo + (hi - lo) / 2;
        if (U.owners->bindings[mid].address <= address) lo = mid + 1;
        else hi = mid;
    }
    if (!lo) return 0;
    const MotionOwnerBinding *b = &U.owners->bindings[lo - 1];
    if (size > b->size || address - b->address > b->size - size) return 0;
    return lo;
}

int vk_usnap_motion_owner(uint64_t address, uint64_t size, MotionOwner *out) {
    if (!U.on || !U.owners || !size || address > UINT64_MAX - size) return 0;
    // Consecutive scene draws often bind the same camera, material and shape data.
    // Ownership is immutable within a capture. Readability is checked separately at the draw's position.
    typedef struct { uint64_t capture, address, size; unsigned slot; } OwnerHint;
    static _Thread_local OwnerHint hints[64];
    unsigned index = (unsigned)(((address >> 8) * UINT64_C(0x9e3779b97f4a7c15)) >> 58);
    OwnerHint *hint = &hints[index];
    if (hint->capture != U.capture || hint->address != address || hint->size != size) {
        *hint = (OwnerHint){U.capture, address, size, motion_owner_slot(address, size)};
    }
    if (!hint->slot) return 0;
    *out = U.owners->bindings[hint->slot - 1].owner;
    return 1;
}

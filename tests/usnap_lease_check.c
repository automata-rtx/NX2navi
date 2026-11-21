// Snapshot adapters use the production registry and host allocations instead of Vulkan arenas.
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdarg.h>
#include "../runtime/vk_usnap.c"
#define RC_REGISTRY_SINGLE_THREADED 1
#define rc_registry_runtime unused_runtime_registry
#include "../runtime/resource_core_registry.c"
#undef rc_registry_runtime

// Buffer donor selection is exercised with its actual adapters by ordered_uniform_merge_check.py.
RcRegistryResult vk_mem_span_buffer_fragments(const VkMemSpanCapture *capture, const VkMemSpan *span,
    uint64_t address, uint64_t size, RcRegistryGpuCapture token, RcRegistryPosition read,
    const VkMemBufferInlineProof *inline_proofs, uint32_t inline_count,
    RcRegistryBufferFragment *fragments, uint32_t capacity, uint32_t *count) {
    (void)capture; (void)span; (void)address; (void)size; (void)token; (void)read;
    (void)inline_proofs; (void)inline_count; (void)fragments; (void)capacity; (void)count;
    assert(!"snapshot fixture must not request a native buffer donor");
    return RC_REGISTRY_UNSUPPORTED;
}

static uint64_t epoch = 1, completed, ticks;
static RcRegistry *registry;
static VkCtx ctx;
static uint8_t *guest;
static size_t guest_size;
static uint64_t imported_pool_boundary;
static uint64_t imported_pool_gap, imported_pool_rounding, imported_pool_overlap;
static uint64_t indexed_handle;
static RcAdmission budget;
static RcAdmissionRecord budget_records[32];
static unsigned allocations, destructions, host_reservations, host_materializations, flushes;
static uint64_t handed;
static int omit_device_mapping, first_range_unmapped, snapshot_mode = 1;
static int reject_first_hold, ab_phase;
static unsigned read_holds, failed_holds;
static unsigned mappings_live, imports_live, remap_on_release;
static uint64_t mapping_id = 1;
static VkResult injected_flush_result = VK_SUCCESS;
static unsigned device_lost_flush_checks;
static unsigned receipt_plans, receipt_fills, nested_receipt_fills;
static unsigned extra_receipt_intervals;
static int change_receipt_count, truncate_receipt_fill, nested_import_receipt;
static size_t refuse_receipt_bytes;
static unsigned budget_reservations;
static RcAdmissionToken watched_gather[2];
static unsigned gather_locked, gather_retires;
static RcBudgetReclaimFn snapshot_reclaimer;
static int model_gpu_peak;
static uint32_t known_heaps;
static unsigned alternate_creates;
static RcAdmissionStatus primary_admission, alternate_admission;
struct VkDrawScratchCredit {
    RcAdmissionVector cost;
    RcBudgetTicket ticket, metadata_budget;
    int ready;
};
static unsigned scratch_materializations, scratch_ready;
static int queue_reuse_model, queue_compatible, queue_existing_scratch, queue_scratch_failure;
static int queue_bar_placement;
static unsigned compatible_probes, queued_growth_attempts;
static int joint_testing, expect_scratch_ready, drain_older;
static const Arena *expect_reusable_at_copy;
static VmaAllocation joint_native;
static RcBudgetTicket joint_native_ticket;
static RcAdmissionVector joint_native_cost;
typedef struct {
    uint64_t fence;
    void (*done)(void *);
    void (*cancel)(void *, int);
    void *arg;
} SnapshotCallback;
static SnapshotCallback callbacks[16];
static unsigned ncallbacks;
volatile int32_t vk_lost;
enum { PRESSURE_STOP = 1, PRESSURE_RETRY };
static int stopping, pressure_action;
static unsigned queue_flushes, pressure_calls;
int vk_sync_stopping(void) { return stopping; }
void vk_queue_lock(void) { queue_flushes++; }
void vk_queue_unlock(void) {}
RcAdmissionStatus resource_core_budget_register_reclaimer(RcBudgetProducer p, uint32_t domains,
                                                         RcBudgetReclaimFn fn, void *context) {
    assert(p == RC_BUDGET_SNAPSHOT && domains && fn && !context && !snapshot_reclaimer);
    snapshot_reclaimer = fn;
    return RC_ADMISSION_OK;
}
void resource_core_budget_progress(void) {}
void resource_core_budget_snapshot(RcBudgetSnapshot *out) {
    *out = (RcBudgetSnapshot){0};
    out->sample.known_mask = known_heaps;
    unsigned count = queue_reuse_model ? budget.domain_count : RC_BUDGET_COMMIT + 1;
    for (unsigned d = 0; d < count; d++) {
        uint64_t held = budget.reserved.amount[d] + budget.retained.amount[d] + budget.pending.amount[d];
        uint64_t limit = budget.domain[d].limit;
        out->available.amount[d] = limit > held ? limit - held : 0;
    }
}
void resource_core_budget_readiness(int required, RcBudgetReadiness *out) {
    assert(required);
    RcBudgetSnapshot snapshot;
    resource_core_budget_snapshot(&snapshot);
    uint32_t usable = queue_reuse_model ? ((1u << RC_BUDGET_GPU_FIRST) - 1) | known_heaps
                                        : (1u << budget.domain_count) - 1;
    *out = (RcBudgetReadiness){.available = snapshot.available, .usable_mask = usable,
                              .records_available = budget.free_head != UINT32_MAX};
}
void resource_core_budget_service(int required, const RcAdmissionVector *peak, RcBudgetProgress *out) {
    assert(required && pressure_action && peak->amount[RC_BUDGET_RAM]);
    pressure_calls++;
    *out = (RcBudgetProgress){.event = 1, .blocked_mask = 1};
    if (pressure_action == PRESSURE_STOP) {
        stopping = 1;
    } else {
        budget.domain[RC_BUDGET_RAM].limit = UINT64_MAX;
        first_range_unmapped = 1;
        memset(guest, 0x72, guest_size);
        out->blocked_mask = 0;
        out->reclaim = RC_BUDGET_RECLAIM_DESTROYED;
    }
}
void resource_core_budget_wait_since(uint64_t event, uint32_t ms) {
    assert(event == 1 && ms && pressure_action == PRESSURE_STOP && stopping);
}
RcAdmissionStatus resource_core_vma_allocation_cost(VmaAllocation allocation, RcAdmissionVector *out) {
    assert(allocation);
    *out = allocation == joint_native ? joint_native_cost : (RcAdmissionVector){{0}};
    return RC_ADMISSION_OK;
}
static size_t uniform_offsets[] = {8, 8 + 8 + sizeof(p_bind_buf)};
static struct {
    uint64_t handle, bytes;
    const size_t *offsets;
    size_t noffsets;
    unsigned ordered_reads;
} projection_record;
VkCtx *vk_ctx(void) { return &ctx; }
int vk_exec_shared(void) { return 1; }
uint64_t vk_exec_epoch(int write) { (void)write; return epoch; }
RcRegistry *rc_registry_runtime(void) { return registry; }
RcRegistryPosition vk_exec_resource_position(void) {
    RcRegistryPosition read = {U.fence, 2};
    if (U.fence && rc_registry_position_compare(read, registry->declared) > 0)
        assert(rc_registry_declare_through(registry, read) == RC_REGISTRY_OK);
    return read;
}
uint64_t vk_sync_fence_done(void) { return completed; }
uint64_t kern_now_ns(void) { return ++ticks; }
int rt_knob(const char *name, const char *def) { (void)name; (void)def; return snapshot_mode; }
int rt_ab(void) { return ab_phase; }
static int temporal, async_replay, owner_acquires;
static MotionOwnerSnapshot owner_snapshot;
int vk_dlaa_available(void) { return temporal; }
int gfx_replay_async(void) { return async_replay; }
MotionOwnerSnapshot *motion_owner_acquire(void) {
    owner_acquires++;
    owner_snapshot.refs++;
    return &owner_snapshot;
}
void motion_owner_release(MotionOwnerSnapshot *snapshot) {
    if (snapshot) {
        assert(snapshot == &owner_snapshot && snapshot->refs);
        snapshot->refs--;
    }
}
void motion_owner_shutdown(void) { assert(!owner_snapshot.refs); }
uint64_t vk_mvprev_step(int stage, int index, uint64_t size) { (void)stage; (void)index; (void)size; return 0; }
uint64_t vk_mem_gen(void) { return 1; }
int vk_mem_lookup(uint64_t addr, uint64_t n, VkMemSpan *out) {
    uint64_t lo = (uintptr_t)guest;
    if (addr < lo || addr - lo >= guest_size) return 0;
    uint64_t offset = addr - lo;
    if (first_range_unmapped && offset < 1024) return 0;
    uint64_t right_start = imported_pool_boundary + imported_pool_gap - imported_pool_overlap;
    uint64_t pool_start = 0, pool_end = guest_size;
    if (offset < right_start) {
        if (offset >= imported_pool_boundary) return 0;
        pool_end = imported_pool_boundary;
    } else pool_start = right_start;
    if (offset < pool_start) return 0;
    if (n > pool_end - offset) return 0;
    *out = (VkMemSpan){.buffer = (VkBuffer)(uintptr_t)(pool_start ? 2 : 1),
        .pool = guest + pool_start, .offset = offset - pool_start,
        .avail = pool_end - offset + (pool_end == imported_pool_boundary ? imported_pool_rounding : 0),
        .cls = VK_MCLS_HOST_CACHED};
    return 1;
}
int vk_mem_resolve(uint64_t a, uint64_t n, VkMemSpan *out) { return vk_mem_lookup(a, n, out); }
int mem_read_hold(uint64_t addr, uint64_t n) {
    if (joint_testing) assert(scratch_ready == (unsigned)expect_scratch_ready && joint_native && free_arenas == expect_reusable_at_copy);
    VkMemSpan span;
    if (!vk_mem_lookup(addr, n, &span) || (reject_first_hold && addr == (uintptr_t)guest)) {
        failed_holds++;
        return 0;
    }
    read_holds++;
    return 1;
}
void mem_read_release(void) {
    assert(read_holds);
    read_holds--;
    if (remap_on_release) { remap_on_release = 0; mapping_id++; }
}

struct MemBackingCapture { uint64_t address, size, mapping; uint32_t count; RcInterval parts[]; };
struct VkMemSpanCapture { uint64_t address, size, mapping; MemBackingCapture *host; };

enum { FIXTURE_MAX_INTERVALS = 16 };
static uint32_t fixture_intervals(uint64_t addr, uint64_t size, RcInterval parts[FIXTURE_MAX_INTERVALS]) {
    uint64_t offset = (addr - (uintptr_t)guest) % (guest_size / 2);
    uint64_t first = offset < 512 && offset + size > 512 ? 512 - offset : size;
    uint32_t count = 1;
    parts[0] = (RcInterval){addr, mapping_id, {2, offset, first}};
    if (first < size) parts[count++] = (RcInterval){addr + first, mapping_id, {2, offset + first, size - first}};
    for (unsigned i = 0; i < extra_receipt_intervals && parts[count - 1].range.size > 1; ++i) {
        assert(count < FIXTURE_MAX_INTERVALS);
        RcInterval *tail = &parts[count - 1];
        RcInterval next = *tail;
        uint64_t cut = tail->range.size / (extra_receipt_intervals - i + 1);
        if (!cut) break;
        tail->range.size = cut;
        next.address += cut;
        next.range.offset += cut;
        next.range.size -= cut;
        parts[count++] = next;
    }
    return count;
}

RcBackingResult mem_backing_capture(uint64_t addr, uint64_t size, MemBackingCapture **out) {
    VkMemSpan span;
    assert(!read_holds);
    *out = NULL;
    if (!vk_mem_lookup(addr, size, &span)) return RC_BACKING_ABSENT;
    RcInterval parts[FIXTURE_MAX_INTERVALS];
    uint32_t count = fixture_intervals(addr, size, parts);
    MemBackingCapture *mapping = calloc(1, sizeof *mapping + count * sizeof *parts);
    assert(mapping);
    *mapping = (MemBackingCapture){.address = addr, .size = size, .mapping = mapping_id, .count = count};
    memcpy(mapping->parts, parts, count * sizeof *parts);
    mappings_live++;
    *out = mapping;
    return RC_BACKING_OK;
}

RcBackingResult mem_backing_capture_plan(uint64_t addr, uint64_t size, MemBackingCapturePlan *out) {
    VkMemSpan span;
    receipt_plans++;
    if (!vk_mem_lookup(addr, size, &span)) return RC_BACKING_ABSENT;
    RcInterval parts[FIXTURE_MAX_INTERVALS];
    uint32_t count = fixture_intervals(addr, size, parts);
    *out = (MemBackingCapturePlan){addr, size, mapping_id, sizeof(MemBackingCapture) + count * sizeof *parts, count};
    return RC_BACKING_OK;
}

RcBackingResult mem_backing_capture_bounded(const MemBackingCapturePlan *plan, MemBackingCapture **out) {
    receipt_fills++;
    if (change_receipt_count) {
        change_receipt_count = 0;
        extra_receipt_intervals = 1;
    }
    RcInterval parts[FIXTURE_MAX_INTERVALS];
    if (plan->latest_mapping != mapping_id || fixture_intervals(plan->address, plan->size, parts) != plan->count)
        return RC_BACKING_STALE;
    assert(plan->bytes == sizeof(MemBackingCapture) + plan->count * sizeof *parts);
    RcBackingResult result = mem_backing_capture(plan->address, plan->size, out);
    if (result == RC_BACKING_OK && truncate_receipt_fill) {
        truncate_receipt_fill = 0;
        mem_backing_release(out);
        return RC_BACKING_STALE;
    }
    return result;
}
const RcInterval *mem_backing_intervals(const MemBackingCapture *mapping, uint32_t *count) {
    *count = mapping ? mapping->count : 0;
    return mapping ? mapping->parts : NULL;
}
int mem_backing_current_range(const MemBackingCapture *mapping, uint64_t addr, uint64_t n) {
    VkMemSpan span;
    return mapping && mapping->mapping == mapping_id && addr >= mapping->address &&
        addr - mapping->address < mapping->size && n <= mapping->size - (addr - mapping->address) &&
        vk_mem_lookup(addr, n, &span);
}
int mem_backing_current(const MemBackingCapture *mapping) {
    return mapping && mem_backing_current_range(mapping, mapping->address, mapping->size);
}
int mem_backing_read_hold(const MemBackingCapture *mapping) {
    return mem_backing_current(mapping) && mem_read_hold(mapping->address, mapping->size);
}
void mem_backing_release(MemBackingCapture **mapping) {
    if (!*mapping) return;
    assert(mappings_live && !read_holds);
    mappings_live--;
    free(*mapping);
    *mapping = NULL;
}
VkMemSpanCapture *vk_mem_span_capture(uint64_t addr, uint64_t n, const VkMemSpan *selected) {
    VkMemSpan span;
    assert(vk_mem_lookup(addr, n, &span));
    assert(selected->buffer == span.buffer && selected->pool == span.pool && selected->offset == span.offset);
    VkMemSpanCapture *capture = malloc(sizeof *capture);
    assert(capture);
    *capture = (VkMemSpanCapture){.address = addr, .size = n, .mapping = mapping_id};
    imports_live++;
    return capture;
}
void vk_mem_span_capture_release(VkMemSpanCapture **import) {
    if (!*import) return;
    assert(imports_live && !read_holds);
    imports_live--;
    mem_backing_release(&(*import)->host);
    free(*import);
    *import = NULL;
}
RcRegistryResult vk_mem_span_capture_plan(const VkMemSpan *selected, const MemBackingCapturePlan *host,
                                          VkMemSpanCapturePlan *out) {
    *out = (VkMemSpanCapturePlan){.selected = *selected, .publication = mapping_id, .nested_host = nested_import_receipt,
        .bytes = sizeof(VkMemSpanCapture) + (nested_import_receipt ? host->bytes : 0)};
    return RC_REGISTRY_OK;
}

RcRegistryResult vk_mem_span_capture_bounded(const VkMemSpanCapturePlan *plan, const MemBackingCapturePlan *host,
                                             VkMemSpanCapture **out) {
    if (plan->publication != mapping_id || plan->nested_host != nested_import_receipt) return RC_REGISTRY_STALE;
    assert(plan->bytes == sizeof(VkMemSpanCapture) + (plan->nested_host ? host->bytes : 0));
    *out = vk_mem_span_capture(host->address, host->size, &plan->selected);
    if (plan->nested_host) {
        RcBackingResult setup = mem_backing_capture_bounded(host, &(*out)->host);
        if (setup != RC_BACKING_OK) {
            vk_mem_span_capture_release(out);
            return RC_REGISTRY_STALE;
        }
        nested_receipt_fills++;
    }
    return RC_REGISTRY_OK;
}
RcRegistryResult vk_mem_canonical_source_qualify_expected(const MemBackingCapture *source,
                                                          const VkMemSpanCapture *import) {
    assert(!read_holds);
    return source && import && import->mapping == mapping_id && source->address == import->address &&
        source->size == import->size && mem_backing_current(source) ? RC_REGISTRY_OK : RC_REGISTRY_STALE;
}
void vk_device_lost_check(VkResult result, const char *where) {
    assert(result == VK_ERROR_DEVICE_LOST && where);
    device_lost_flush_checks++;
}
int nvn_record_plan(uint64_t h, NvnRecordPlan *out) {
    *out = (NvnRecordPlan){0};
    if (projection_record.handle && h == projection_record.handle && projection_record.offsets) {
        out->commands = (const uint8_t *)(uintptr_t)h;
        out->bytes = (size_t)projection_record.bytes;
        out->capture_offsets = projection_record.offsets;
        out->ncapture = 2; // The unique capture index has no entry for the repeated last binding.
        return 1;
    }
    if (h != indexed_handle) return 0;
    out->commands = (const uint8_t *)(uintptr_t)h;
    out->capture_offsets = uniform_offsets;
    out->ncapture = 2;
    return 1;
}
uint64_t nvn_submission_handle_bytes(uint64_t handle) {
    return handle == projection_record.handle ? projection_record.bytes : 0;
}
int nvn_record_uniform_offsets(uint64_t handle, const size_t **offsets, size_t *count) {
    if (handle != projection_record.handle || !projection_record.offsets) return 0;
    projection_record.ordered_reads++;
    *offsets = projection_record.offsets;
    *count = projection_record.noffsets;
    return 1;
}
void vk_exec_par(void (*fn)(void *, int, int), void *arg, int n, int slice) { (void)slice; fn(arg, 0, n); }
void vk_completion_add_owned(uint64_t fence, void (*fn)(void *), void (*cancel)(void *, int), void *arg) {
    assert(fence > completed && fn && cancel && ncallbacks < sizeof callbacks / sizeof callbacks[0]);
    callbacks[ncallbacks++] = (SnapshotCallback){fence, fn, cancel, arg};
}

static SnapshotCallback remove_callback(unsigned i) {
    assert(i < ncallbacks);
    SnapshotCallback callback = callbacks[i];
    --ncallbacks;
    memmove(callbacks + i, callbacks + i + 1, (ncallbacks - i) * sizeof *callbacks);
    return callback;
}

static void complete_snapshots(uint64_t fence) {
    assert(completed <= fence && fence <= handed);
    while (ncallbacks && callbacks[0].fence <= fence) {
        SnapshotCallback callback = remove_callback(0);
        callback.done(callback.arg);
    }
    completed = fence;
}

static void cancel_snapshot(unsigned i, int idle) {
    SnapshotCallback callback = remove_callback(i);
    assert(callback.fence > completed && (callback.fence > handed || idle));
    callback.cancel(callback.arg, callback.fence <= handed);
}

static VkResult preflight_with_peak(const VkUsnapInputs *inputs, VkUsnapCapture **capture, RcAdmissionVector *peak) {
    VkReplayCaptureWait wait;
    VkResult result = vk_usnap_preflight(inputs, capture, &wait);
    *peak = wait.peak;
    if (result == VK_NOT_READY) assert(wait.kind == VK_REPLAY_CAPTURE_WAIT_BUDGET);
    return result;
}

static void begin_snapshot(const uint64_t *handles, int n, uint64_t fence) {
    VkUsnapCapture *capture = NULL;
    RcAdmissionVector peak;
    const VkUsnapInputs inputs = {.handles = handles, .nhandles = n};
    assert(preflight_with_peak(&inputs, &capture, &peak) == VK_SUCCESS);
    vk_usnap_begin(&capture, n, fence);
    assert(!capture);
}

static void end_completed_snapshot(uint64_t fence) {
    vk_usnap_end(fence);
    handed = fence;
    complete_snapshots(fence);
}
RcAdmissionStatus resource_core_budget_reserve(RcBudgetProducer p,int required,const RcAdmissionVector *cost,RcBudgetTicket *t) {
    assert(required && !t->token.ledger);
    budget_reservations++;
    if (cost->amount[RC_BUDGET_RAM] >= ARENA_MIN) host_reservations++;
    if (refuse_receipt_bytes && cost->amount[RC_BUDGET_RAM] == refuse_receipt_bytes) return RC_ADMISSION_REFUSED;
    if (queue_reuse_model) {
        for (unsigned d = RC_BUDGET_GPU_FIRST; d < budget.domain_count; d++)
            if (cost->amount[d] && !(known_heaps & (1u << d))) return RC_ADMISSION_UNSUPPORTED;
    }
    RcAdmissionStatus s=rc_admission_reserve(&budget,p,cost,&t->token);if(s==RC_ADMISSION_OK)t->producer=p;return s;
}
RcAdmissionStatus resource_core_budget_split(RcBudgetTicket *parent, const RcAdmissionVector *portion,
                                             RcBudgetProducer owner, RcBudgetTicket *child) {
    assert(parent->token.ledger && !child->token.ledger);
    RcAdmissionStatus status = rc_admission_split(&budget, parent->token, parent->producer, portion, owner,
                                                   &child->token);
    if (status == RC_ADMISSION_OK) child->producer = owner;
    return status;
}
RcAdmissionStatus resource_core_budget_materialize(RcBudgetTicket *t,const RcAdmissionVector *cost) {
    if (cost->amount[RC_BUDGET_RAM] >= ARENA_MIN) host_materializations++;
    return rc_admission_materialize(&budget,t->token,t->producer,cost,&t->token);
}
RcAdmissionStatus resource_core_budget_cancel(RcBudgetTicket *t) {
    RcAdmissionStatus s=rc_admission_cancel(&budget,t->token,t->producer);if(s==RC_ADMISSION_OK)*t=(RcBudgetTicket){0};return s;
}
RcAdmissionStatus resource_core_budget_retire(RcBudgetTicket *t) {
    for (unsigned i = 0; i < 2; i++) {
        const RcAdmissionToken *watch = &watched_gather[i];
        if (watch->ledger && watch->slot == t->token.slot && watch->generation == t->token.generation) {
            assert(!gather_locked);
            gather_retires++;
        }
    }
    return rc_admission_retire(&budget,t->token,t->producer,&t->token);
}
RcAdmissionStatus resource_core_budget_destroyed(RcBudgetTicket *t) {
    RcAdmissionStatus s=rc_admission_destroyed(&budget,t->token,t->producer);if(s==RC_ADMISSION_OK)*t=(RcBudgetTicket){0};return s;
}
size_t vk_exec_draw_update_credit_host_bytes(const VkDrawScratchCredit *credit) {
    return credit ? sizeof *credit : 0;
}

VkResult vk_exec_draw_update_credit_plan(uint64_t bytes, VkDrawScratchCredit **credit,
                                        RcAdmissionVector *peak, int *drain) {
    assert(joint_testing && !*credit && !scratch_ready);
    *peak = (RcAdmissionVector){0};
    *drain = drain_older;
    if (*drain) return VK_NOT_READY;
    if (!bytes) return VK_SUCCESS;
    RcBudgetTicket metadata = {0};
    void *allocation;
    VkResult result = vk_replay_metadata_allocate(sizeof **credit, 1, RC_BUDGET_BUFFER, &metadata, &allocation);
    if (result != VK_SUCCESS) {
        *peak = arena_host_cost(sizeof **credit);
        return result;
    }
    *credit = allocation;
    (*credit)->metadata_budget = metadata;
    if (!queue_existing_scratch) {
        *peak = (*credit)->cost = arena_host_cost(bytes);
        if (queue_reuse_model) peak->amount[RC_BUDGET_GPU_FIRST + 1] = (*credit)->cost.amount[RC_BUDGET_GPU_FIRST + 1] = bytes;
    }
    return VK_SUCCESS;
}
VkResult vk_exec_draw_update_credit_materialize(VkDrawScratchCredit *credit, RcBudgetTicket *joint) {
    assert(joint_testing && credit && !scratch_ready && !read_holds);
    if (queue_scratch_failure) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (!queue_existing_scratch) {
        assert(resource_core_budget_split(joint, &credit->cost, RC_BUDGET_BUFFER, &credit->ticket) == RC_ADMISSION_OK);
        RcAdmissionStatus accounted = resource_core_budget_materialize(&credit->ticket, &credit->cost);
        assert(rc_admission_accounted(accounted));
        scratch_materializations++;
    }
    scratch_ready++;
    credit->ready = 1;
    memset(guest, 0x35, guest_size);
    return VK_SUCCESS;
}
void vk_exec_draw_update_credit_cancel(VkDrawScratchCredit *credit) {
    assert(joint_testing && credit);
    if (credit->ticket.token.ledger) {
        assert(resource_core_budget_retire(&credit->ticket) == RC_ADMISSION_OK);
        assert(resource_core_budget_destroyed(&credit->ticket) == RC_ADMISSION_OK);
    }
    if (credit->ready) {
        assert(scratch_ready == 1);
        scratch_ready--;
    }
    RcBudgetTicket metadata = credit->metadata_budget;
    vk_replay_metadata_free(credit, &metadata);
}
void plat_mutex_lock(PlatMutex *p) {
    if (p == &gather_lock) {
        assert(!gather_locked);
        gather_locked = 1;
    }
}
void plat_mutex_unlock(PlatMutex *p) {
    if (p == &gather_lock) {
        assert(gather_locked);
        gather_locked = 0;
    }
}
void plat_once(PlatOnce *once, void (*fn)(void *), void *arg) {
    if (!once->state) {
        fn(arg);
        once->state = 2;
    }
}
void plat_cond_wait(PlatCond *cv, PlatMutex *lock) { (void)cv; (void)lock; assert(0); }
void plat_cond_broadcast(PlatCond *cv) { (void)cv; }
void rt_log(const char *fmt, ...) { (void)fmt; }
_Noreturn void rt_fatal(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    abort();
}
void perf_size(PerfSizes *s, const char *name, uint64_t count, uint64_t limit, uint64_t bytes) {
    (void)s; (void)name; (void)count; (void)limit; (void)bytes;
}
VKAPI_ATTR VkResult VKAPI_CALL vmaCreateBuffer(VmaAllocator a, const VkBufferCreateInfo *bi,
        const VmaAllocationCreateInfo *ai, VkBuffer *b, VmaAllocation *allocation, VmaAllocationInfo *info) {
    (void)a; (void)ai;
    void *data = malloc((size_t)bi->size);
    assert(data);
    allocations++;
    *b = (VkBuffer)data, *allocation = (VmaAllocation)data;
    memset(info, 0, sizeof *info);
    info->pMappedData = omit_device_mapping ? NULL : data;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vmaDestroyBuffer(VmaAllocator a, VkBuffer b, VmaAllocation allocation) {
    (void)a; (void)b; destructions++; free(allocation);
}
static RcAdmissionVector queue_native_cost(uint64_t bytes, int host_preferred) {
    RcAdmissionVector cost = arena_host_cost(bytes);
    if (queue_reuse_model) cost.amount[RC_BUDGET_GPU_FIRST + (unsigned)host_preferred] = bytes;
    if (queue_bar_placement && !host_preferred) cost.amount[RC_BUDGET_RAM] = 0;
    return cost;
}
VkResult resource_core_vma_buffer_peak(const RcVmaRequest *request, const VkBufferCreateInfo *bi,
                                      const VmaAllocationCreateInfo *ai, VmaAllocationCreateInfo *placement,
                                      RcAdmissionVector *peak) {
    assert(joint_testing && request->producer == RC_BUDGET_SNAPSHOT && request->required && !request->host_bytes);
    int alternate = ai->usage == VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    *placement = *ai;
    placement->memoryTypeBits = 1u << alternate;
    *peak = queue_native_cost(bi->size, alternate);
    if (request->dependency) {
        assert(queue_reuse_model && !alternate && !(ai->flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT));
        *request->dependency = (RcVmaDependency){.can_reuse = 1};
        request->dependency->reuse.amount[RC_BUDGET_RAM] = bi->size;
    } else assert(ai->flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    return VK_SUCCESS;
}
VkResult resource_core_vma_create_buffer(const RcVmaRequest *request,VmaAllocator allocator,
        const VkBufferCreateInfo *bi,const VmaAllocationCreateInfo *ai,VkBuffer *b,VmaAllocation *a,
        VmaAllocationInfo *info,RcAdmissionStatus *admission) {
    assert(request->producer == RC_BUDGET_SNAPSHOT && request->required);
    assert(ai->flags & VMA_ALLOCATION_CREATE_MAPPED_BIT);
    assert(ai->flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    int alternate = ai->usage == VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    if (alternate) {
        assert((model_gpu_peak || queue_reuse_model) && (ai->flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT));
        alternate_creates++;
    } else {
        assert(ai->usage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    }
    RcAdmissionVector host_cost = arena_host_cost(request->host_bytes);
    if (request->peak_ticket) {
        assert(joint_testing && !joint_native && ai->memoryTypeBits == (1u << alternate));
        int compatible = (ai->flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) != 0;
        RcAdmissionVector growth = queue_native_cost(bi->size, alternate);
        joint_native_cost = growth;
        if (compatible) {
            assert(queue_reuse_model && !alternate && !scratch_ready &&
                   !(ai->flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT));
            compatible_probes++;
            if (!queue_compatible) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
            growth.amount[RC_BUDGET_COMMIT] = growth.amount[RC_BUDGET_GPU_FIRST] = 0;
            assert(!budget.reserved.amount[RC_BUDGET_GPU_FIRST]);
        } else {
            assert(ai->flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
            assert(scratch_ready == (unsigned)expect_scratch_ready);
            queued_growth_attempts++;
        }
        RcAdmissionVector cost;
        assert(peak_sum(&cost, &growth, &host_cost));
        if (request->refused_peak) *request->refused_peak = cost;
        assert(resource_core_budget_split(request->peak_ticket, &cost, request->producer, &joint_native_ticket) ==
               RC_ADMISSION_OK);
        if (request->host_bytes)
            assert(resource_core_budget_split(&joint_native_ticket, &host_cost, request->producer, request->host_ticket) ==
                   RC_ADMISSION_OK);
        assert(vmaCreateBuffer(allocator, bi, ai, b, a, info) == VK_SUCCESS);
        joint_native = *a;
        RcAdmissionStatus accounted = resource_core_budget_materialize(&joint_native_ticket, &joint_native_cost);
        assert(rc_admission_accounted(accounted));
        if (admission) *admission = RC_ADMISSION_OK;
        return VK_SUCCESS;
    }
    RcAdmissionVector peak = host_cost;
    if (model_gpu_peak) peak.amount[RC_BUDGET_GPU_FIRST + (unsigned)alternate] = bi->size;
    if (request->refused_peak) *request->refused_peak = peak;
    RcAdmissionStatus status = alternate ? alternate_admission : primary_admission;
    if (status == RC_ADMISSION_OK && request->host_bytes)
        status = resource_core_budget_reserve(request->producer, request->required, &host_cost, request->host_ticket);
    if (admission) *admission = status;
    if (status == RC_ADMISSION_UNSUPPORTED) return VK_ERROR_FEATURE_NOT_PRESENT;
    if (status != RC_ADMISSION_OK) return VK_NOT_READY;
    return vmaCreateBuffer(allocator, bi, ai, b, a, info);
}
void resource_core_vma_destroy_buffer(VmaAllocator allocator,VkBuffer b,VmaAllocation a) {
    if (a == joint_native) {
        assert(joint_native);
        assert(resource_core_budget_retire(&joint_native_ticket) == RC_ADMISSION_OK);
        assert(resource_core_budget_destroyed(&joint_native_ticket) == RC_ADMISSION_OK);
        joint_native = NULL;
    }
    vmaDestroyBuffer(allocator,b,a);
}
VKAPI_ATTR VkResult VKAPI_CALL vmaFlushAllocation(VmaAllocator a, VmaAllocation allocation, VkDeviceSize off, VkDeviceSize n) {
    (void)a;
    assert(allocation && off == 0 && n);
    flushes++;
    return injected_flush_result;
}

static void check_flush_failure_cleanup(const VkUsnapInputs *inputs) {
    int saved_temporal = temporal, saved_mode = snapshot_mode;
    temporal = 1;
    snapshot_mode = 1;
    vk_usnap_reclaim_reusable();

    VkReplayCaptureBundle bundle = {0};
    VkReplayCaptureWait wait;
    uint64_t baseline_ram = budget.retained.amount[RC_BUDGET_RAM];
    uint64_t baseline_commit = budget.retained.amount[RC_BUDGET_COMMIT];
    unsigned before_allocations = allocations, before_destructions = destructions;
    unsigned before_flushes = flushes, before_lost = device_lost_flush_checks;
    injected_flush_result = VK_ERROR_OUT_OF_HOST_MEMORY;
    VkResult result = vk_usnap_queue_bundle(inputs, 0, &bundle, &wait);
    injected_flush_result = VK_SUCCESS;
    assert(result == VK_ERROR_OUT_OF_HOST_MEMORY);
    assert(!bundle.snapshot && !bundle.scratch && free_arenas);
    assert(free_arenas->state == ARENA_REUSABLE && free_arenas->buf && free_arenas->cpu);
    assert(allocations == before_allocations + 1 && destructions == before_destructions);
    assert(flushes == before_flushes + 1 && device_lost_flush_checks == before_lost);
    assert(!mappings_live && !imports_live && !registry->active_gpu_captures);
    assert(!budget.reserved.amount[RC_BUDGET_RAM] && !budget.pending.amount[RC_BUDGET_RAM]);
    assert(vk_usnap_reclaim_reusable() == 1 && destructions == before_destructions + 1);
    assert(!budget.reserved.amount[RC_BUDGET_RAM] && !budget.pending.amount[RC_BUDGET_RAM]);
    assert(budget.retained.amount[RC_BUDGET_RAM] == baseline_ram &&
           budget.retained.amount[RC_BUDGET_COMMIT] == baseline_commit);

    VkUsnapCapture *capture = NULL;
    RcAdmissionVector peak;
    assert(preflight_with_peak(inputs, &capture, &peak) == VK_SUCCESS && capture && capture->arena);
    before_allocations = allocations;
    before_destructions = destructions;
    before_flushes = flushes;
    injected_flush_result = VK_ERROR_OUT_OF_HOST_MEMORY;
    uint64_t fence = completed + 1;
    result = vk_usnap_begin(&capture, inputs->nhandles, fence);
    injected_flush_result = VK_SUCCESS;
    assert(result == VK_ERROR_OUT_OF_HOST_MEMORY && !capture);
    assert(!U.arena && !U.reg && !U.expected && !U.producer.read.id);
    assert(free_arenas && free_arenas->state == ARENA_REUSABLE && free_arenas->buf && free_arenas->cpu);
    assert(allocations == before_allocations && destructions == before_destructions);
    assert(flushes == before_flushes + 1 && device_lost_flush_checks == before_lost);
    assert(!mappings_live && !imports_live && !registry->active_gpu_captures);
    assert(!budget.reserved.amount[RC_BUDGET_RAM] && !budget.pending.amount[RC_BUDGET_RAM]);
    vk_usnap_end(fence);
    assert(vk_usnap_reclaim_reusable() == 1 && destructions == before_destructions + 1);
    assert(budget.retained.amount[RC_BUDGET_RAM] == baseline_ram &&
           budget.retained.amount[RC_BUDGET_COMMIT] == baseline_commit);

    // Device loss still takes the dedicated terminal-reporting hook. The fixture hook
    // returns so the ownership cleanup can be observed in this process.
    assert(preflight_with_peak(inputs, &capture, &peak) == VK_SUCCESS && capture && capture->arena);
    before_destructions = destructions;
    before_lost = device_lost_flush_checks;
    injected_flush_result = VK_ERROR_DEVICE_LOST;
    fence = completed + 1;
    result = vk_usnap_begin(&capture, inputs->nhandles, fence);
    injected_flush_result = VK_SUCCESS;
    assert(result == VK_ERROR_DEVICE_LOST && !capture);
    assert(device_lost_flush_checks == before_lost + 1 && free_arenas && free_arenas->state == ARENA_REUSABLE);
    assert(!mappings_live && !imports_live && !registry->active_gpu_captures);
    vk_usnap_end(fence);
    assert(vk_usnap_reclaim_reusable() == 1 && destructions == before_destructions + 1);
    assert(budget.retained.amount[RC_BUDGET_RAM] == baseline_ram &&
           budget.retained.amount[RC_BUDGET_COMMIT] == baseline_commit);

    temporal = saved_temporal;
    snapshot_mode = saved_mode;
}

static uint64_t gather_bytes(void) {
    return ((uint64_t)U.ccol + U.sort_cap) * sizeof(Rng);
}

static void check_gather_pressure(void) {
    enum { OLD_CAPACITY = 8192 };
    p_bind_buf *bindings = malloc((2 * OLD_CAPACITY + 1) * sizeof *bindings);
    assert(bindings);
    for (size_t i = 0; i <= 2 * OLD_CAPACITY; i++)
        bindings[i] = (p_bind_buf){.stage = 0, .index = 3,
            .addr = (uintptr_t)guest + (i % 2 ? 2048 : 0), .size = 256};
    VkUsnapInputs inputs = {.inherited = bindings, .ninherited = OLD_CAPACITY};
    VkUsnapCapture *capture = NULL;
    VkReplayCaptureWait wait;
    vk_usnap_reclaim_reusable();
    int saved_temporal = temporal;
    temporal = 0;
    assert(vk_usnap_preflight(&inputs, &capture, &wait) == VK_SUCCESS && capture && capture->nreg == 2);
    vk_usnap_cancel(capture);
    capture = NULL;

    Rng *old_col = U.col, *old_sort = U.sort_tmp;
    assert(U.ccol == OLD_CAPACITY && U.sort_cap == OLD_CAPACITY && old_col && old_sort);
    watched_gather[0] = U.col_budget.token;
    watched_gather[1] = U.sort_budget.token;
    assert(vk_usnap_preflight(&inputs, &capture, &wait) == VK_SUCCESS && capture && capture->nreg == 2);
    assert(U.col == old_col && U.sort_tmp == old_sort && !gather_retires);
    vk_usnap_cancel(capture);
    capture = NULL;

    uint64_t old_bytes = OLD_CAPACITY * sizeof(Rng);
    uint64_t target_bytes = 2 * old_bytes;
    uint64_t limit = target_bytes + old_bytes / 2;
    uint64_t saved_ram = budget.domain[RC_BUDGET_RAM].limit;
    uint64_t saved_commit = budget.domain[RC_BUDGET_COMMIT].limit;
    budget.domain[RC_BUDGET_RAM].limit = budget.domain[RC_BUDGET_COMMIT].limit = limit;
    inputs.ninherited++;
    assert(vk_usnap_preflight(&inputs, &capture, &wait) == VK_NOT_READY && !capture);
    assert(wait.kind == VK_REPLAY_CAPTURE_WAIT_BUDGET);
    assert(wait.peak.amount[RC_BUDGET_RAM] == target_bytes + sizeof(VkUsnapCapture) &&
           wait.peak.amount[RC_BUDGET_COMMIT] == wait.peak.amount[RC_BUDGET_RAM]);

    // Neither the old buffers plus their replacement nor a geometric rebuild fits.
    // The aborted collection releases both old tickets before the next attempt.
    VkResult retried = vk_usnap_preflight(&inputs, &capture, &wait);
    if (retried != VK_SUCCESS)
        fprintf(stderr, "collection retry result %d, retained gather bytes %llu\n",
                (int)retried, (unsigned long long)gather_bytes());
    assert(retried == VK_SUCCESS && capture && capture->nreg == 2);
    assert(gather_retires == 2 && U.ccol == 2 * OLD_CAPACITY && !U.sort_tmp && !U.sort_budget.token.ledger);
    assert(capture->reg[0].lo == (uintptr_t)guest && capture->reg[1].lo == (uintptr_t)guest + 2048);
    vk_usnap_cancel(capture);
    capture = NULL;
    assert(!budget.pending.amount[RC_BUDGET_RAM] && !budget.reserved.amount[RC_BUDGET_RAM]);

    // The same budget still permits a later capture. Raising it restores the radix workspace.
    Rng *rebuilt = U.col;
    assert(vk_usnap_preflight(&inputs, &capture, &wait) == VK_SUCCESS && capture && capture->nreg == 2);
    assert(U.col == rebuilt && !U.sort_tmp && gather_retires == 2);
    vk_usnap_cancel(capture);
    capture = NULL;
    budget.domain[RC_BUDGET_RAM].limit = saved_ram;
    budget.domain[RC_BUDGET_COMMIT].limit = saved_commit;
    assert(vk_usnap_preflight(&inputs, &capture, &wait) == VK_SUCCESS && capture && capture->nreg == 2);
    assert(U.col == rebuilt && U.sort_tmp && U.sort_cap == U.ccol && gather_retires == 2);
    vk_usnap_cancel(capture);
    capture = NULL;
    assert(vk_usnap_reclaim_reusable() == 1 && budget.retained.amount[RC_BUDGET_RAM] == gather_bytes());

    // A smaller capture can arrive first after a larger target is refused.
    watched_gather[0] = U.col_budget.token;
    watched_gather[1] = U.sort_budget.token;
    budget.domain[RC_BUDGET_RAM].limit = budget.domain[RC_BUDGET_COMMIT].limit = 2 * target_bytes + 4096;
    inputs.ninherited = 2 * OLD_CAPACITY + 1;
    assert(vk_usnap_preflight(&inputs, &capture, &wait) == VK_NOT_READY && !capture);
    assert(wait.peak.amount[RC_BUDGET_RAM] == 2 * target_bytes + sizeof(VkUsnapCapture));
    assert(gather_retires == 4 && !U.col && !U.sort_tmp);
    budget.domain[RC_BUDGET_RAM].limit = budget.domain[RC_BUDGET_COMMIT].limit = old_bytes;
    inputs.ninherited = 1;
    assert(vk_usnap_preflight(&inputs, &capture, &wait) == VK_SUCCESS && capture && capture->nreg == 1);
    assert(U.ccol == 1024 && !U.sort_tmp);
    vk_usnap_cancel(capture);
    assert(vk_usnap_reclaim_reusable() == 1 && budget.retained.amount[RC_BUDGET_RAM] == gather_bytes());
    budget.domain[RC_BUDGET_RAM].limit = saved_ram;
    budget.domain[RC_BUDGET_COMMIT].limit = saved_commit;

    inputs.ninherited = 2;
    capture = NULL;
    assert(vk_usnap_preflight(&inputs, &capture, &wait) == VK_SUCCESS && capture && capture->nreg == 2);
    vk_usnap_cancel(capture);
    assert(vk_usnap_reclaim_reusable() == 1 && U.col && U.sort_tmp);
    assert(budget.retained.amount[RC_BUDGET_RAM] == gather_bytes());
    watched_gather[0] = U.col_budget.token;
    watched_gather[1] = U.sort_budget.token;
    temporal = saved_temporal;
    free(bindings);
    puts("snapshot gather: original and retry capacity, lock-safe disposal and continued capture passed");
}

static uint64_t capture_metadata_bytes(const VkUsnapCapture *capture) {
    return sizeof *capture + (uint64_t)capture->region_cap * sizeof(Reg) +
        capture->producer.coverage_bytes + capture->receipt_bytes +
        permission_scratch_bytes(capture->permission_capacity);
}

static void check_host_placement(const VkUsnapInputs *inputs) {
    assert(temporal && !free_arenas);
    VkUsnapCapture *capture = NULL;
    RcAdmissionVector peak;
    unsigned before_alloc = allocations, before_alternate = alternate_creates;
    model_gpu_peak = 1;
    known_heaps = 3u << RC_BUDGET_GPU_FIRST;
    primary_admission = RC_ADMISSION_REFUSED;

    // A known device-heap deficit permits one independently admitted host-visible placement.
    assert(preflight_with_peak(inputs, &capture, &peak) == VK_SUCCESS);
    assert(capture && capture->arena->cpu && allocations == before_alloc + 1);
    assert(alternate_creates == before_alternate + 1);
    vk_usnap_cancel(capture);
    capture = NULL;
    assert(vk_usnap_reclaim_reusable() == 1);

    // If both placements are refused, the pressure request describes the alternate's footprint.
    alternate_admission = RC_ADMISSION_REFUSED;
    assert(preflight_with_peak(inputs, &capture, &peak) == VK_NOT_READY);
    assert(!capture && !peak.amount[RC_BUDGET_GPU_FIRST] && peak.amount[RC_BUDGET_GPU_FIRST + 1] == ARENA_MIN);
    assert(alternate_creates == before_alternate + 2 && allocations == before_alloc + 1);

    alternate_admission = RC_ADMISSION_UNSUPPORTED;
    assert(preflight_with_peak(inputs, &capture, &peak) == VK_NOT_READY);
    assert(!capture && peak.amount[RC_BUDGET_GPU_FIRST] == ARENA_MIN && !peak.amount[RC_BUDGET_GPU_FIRST + 1]);
    assert(alternate_creates == before_alternate + 3);

    known_heaps = 0;
    assert(preflight_with_peak(inputs, &capture, &peak) == VK_NOT_READY);
    known_heaps = 3u << RC_BUDGET_GPU_FIRST;
    budget.domain[RC_BUDGET_COMMIT].limit = 0;
    assert(preflight_with_peak(inputs, &capture, &peak) == VK_NOT_READY);
    budget.domain[RC_BUDGET_COMMIT].limit = UINT64_MAX;
    primary_admission = RC_ADMISSION_NO_RECORDS;
    assert(preflight_with_peak(inputs, &capture, &peak) == VK_ERROR_TOO_MANY_OBJECTS);
    assert(!capture && alternate_creates == before_alternate + 3 && allocations == before_alloc + 1);
    assert(!budget.reserved.amount[RC_BUDGET_RAM] && budget.retained.amount[RC_BUDGET_RAM] == gather_bytes());
    model_gpu_peak = 0;
    known_heaps = 0;
    primary_admission = alternate_admission = RC_ADMISSION_OK;
}

static void check_inherited_snapshot(void) {
    const op_hdr commands[] = {{OP_BEGIN, sizeof(op_hdr)}, {OP_END, sizeof(op_hdr)}};
    const uint64_t handle = (uintptr_t)commands;
    const uint64_t addr = (uintptr_t)guest + 4096;
    const p_bind_buf inherited = {.stage = 0, .index = 3, .addr = addr, .size = 256};
    const VkUsnapInputs sync = {.inherited = &inherited, .ninherited = 1};
    const VkUsnapInputs async = {.handles = &handle, .nhandles = 1, .inherited = &inherited, .ninherited = 1};
    const VkUsnapInputs control = {.handles = &handle, .nhandles = 1};
    VkUsnapCapture *capture = NULL;
    RcAdmissionVector peak;
    VkDescriptorBufferInfo buffer;
    VkUsnapLease lease;

    // A continuation can carry an inherited UBO with no incoming commands.
    uint64_t sync_fence = completed + 1;
    memset(guest + 4096, 0x81, 256);
    unsigned before_flush = flushes;
    assert(preflight_with_peak(&sync, &capture, &peak) == VK_SUCCESS && capture);
    assert(flushes == before_flush);
    memset(guest + 4096, 0x82, 256);
    vk_usnap_begin(&capture, 0, sync_fence);
    assert(!capture && flushes == before_flush + 1);
    assert(vk_usnap_find_lease(addr, 256, 256, &buffer, &lease) && vk_usnap_lease_valid(&lease));
    assert(vk_usnap_bytes(addr, 1)[0] == 0x82);
    end_completed_snapshot(sync_fence);
    assert(!vk_usnap_lease_valid(&lease));

    // The same empty command handle without inherited bindings carries no snapshot bytes.
    VkUsnapCapture *empty = vk_usnap_take(&control);
    assert(empty && !empty->arena && flushes == before_flush + 1);
    vk_usnap_cancel(empty);

    memset(guest + 4096, 0x83, 256);
    capture = vk_usnap_take(&async);
    memset(guest + 4096, 0x84, 256);
    VkUsnapCapture *canceled = vk_usnap_take(&async);
    assert(capture && canceled && capture->arena && canceled->arena != capture->arena);
    vk_usnap_cancel(canceled);
    assert(vk_usnap_reclaim_reusable() == 1);

    uint64_t async_fence = sync_fence + 1;
    memset(guest + 4096, 0x85, 256);
    vk_usnap_begin(&capture, 1, async_fence);
    assert(!capture && flushes == before_flush + 3 && !vk_usnap_lease_valid(&lease));
    assert(vk_usnap_find_lease(addr, 256, 256, &buffer, &lease) && vk_usnap_lease_valid(&lease));
    assert(vk_usnap_bytes(addr, 1)[0] == 0x83);
    end_completed_snapshot(async_fence);
    assert(!vk_usnap_lease_valid(&lease));
}

static void check_projected_snapshot(void) {
    typedef struct {
        op_hdr begin;
        struct {
            op_hdr header;
            p_bind_buf binding;
        } binds[3];
        op_hdr end;
    } Commands;
    const uint64_t addr_a = (uintptr_t)guest + 8192;
    const uint64_t addr_b = (uintptr_t)guest + 12288;
    const p_bind_buf a = {.stage = 0, .index = 3, .addr = addr_a, .size = 256};
    const p_bind_buf b = {.stage = 0, .index = 3, .addr = addr_b, .size = 256};
    Commands commands = {
        .begin = {OP_BEGIN, sizeof(op_hdr)},
        .binds = {
            {{OP_BIND_UBO, sizeof commands.binds[0]}, a},
            {{OP_BIND_UBO, sizeof commands.binds[1]}, b},
            {{OP_BIND_UBO, sizeof commands.binds[2]}, a},
        },
        .end = {OP_END, sizeof(op_hdr)},
    };
    const uint64_t handle = (uintptr_t)&commands;
    size_t offsets[] = {offsetof(Commands, binds[0]), offsetof(Commands, binds[1]), offsetof(Commands, binds[2])};
    projection_record = (typeof(projection_record)){
        .handle = handle, .bytes = sizeof commands, .offsets = offsets, .noffsets = 3,
    };
    VkUsnapBindingState candidate = {0};

    // The unique capture inventory contains A and B, while final state requires the last A.
    assert(vk_usnap_project(&handle, 1, &candidate));
    assert(projection_record.ordered_reads && candidate.slots[a.index].addr == addr_a);

    // The queued command contains no binding and imports only the projected last value.
    const op_hdr next[] = {{OP_BEGIN, sizeof(op_hdr)}, {OP_END, sizeof(op_hdr)}};
    const uint64_t next_handle = (uintptr_t)next;
    const VkUsnapInputs queued = {
        .handles = &next_handle, .nhandles = 1,
        .inherited = candidate.slots, .ninherited = sizeof candidate.slots / sizeof candidate.slots[0],
    };
    memset(guest + 8192, 0x91, 256);
    memset(guest + 12288, 0x92, 256);
    VkUsnapCapture *capture = vk_usnap_take(&queued);
    assert(capture);
    memset(guest + 8192, 0x93, 256);
    uint64_t fence = completed + 1;
    vk_usnap_begin(&capture, 1, fence);
    assert(!capture && vk_usnap_bytes(addr_a, 1)[0] == 0x91 && !vk_usnap_bytes(addr_b, 1));
    end_completed_snapshot(fence);

    // The bounded fallback produces the same ordered result and observes an explicit unbind.
    projection_record.offsets = NULL;
    candidate.slots[a.index] = b;
    assert(vk_usnap_project(&handle, 1, &candidate) && candidate.slots[a.index].addr == addr_a);
    commands.binds[2].binding.addr = 0;
    commands.binds[2].binding.size = 0;
    assert(vk_usnap_project(&handle, 1, &candidate));
    assert(!candidate.slots[a.index].addr && !candidate.slots[a.index].size);
    commands.binds[2].binding = a;

    // A later failure discards staged updates, including a valid preceding command stream.
    const uint64_t mixed[] = {handle, 1};
    candidate.slots[a.index] = b;
    assert(!vk_usnap_project(mixed, 2, &candidate) && candidate.slots[a.index].addr == addr_b);
    projection_record.bytes = sizeof commands - sizeof commands.end;
    assert(!vk_usnap_project(&handle, 1, &candidate) && candidate.slots[a.index].addr == addr_b);
    projection_record.bytes = sizeof commands;
    commands.binds[2].header.op = OP_COUNT_;
    assert(!vk_usnap_project(&handle, 1, &candidate) && candidate.slots[a.index].addr == addr_b);
    commands.binds[2].header.op = OP_BIND_UBO;
    commands.binds[2].header.size = sizeof(op_hdr);
    assert(!vk_usnap_project(&handle, 1, &candidate) && candidate.slots[a.index].addr == addr_b);
    commands.binds[2].header.size = sizeof commands.binds[2];

    projection_record.offsets = offsets;
    offsets[2] = offsets[1];
    assert(!vk_usnap_project(&handle, 1, &candidate) && candidate.slots[a.index].addr == addr_b);
    projection_record = (typeof(projection_record)){0};
}

static void check_queue_bundle(const VkUsnapInputs *inputs) {
    U.mode = 0;
    snapshot_mode = temporal = joint_testing = expect_scratch_ready = 1;
    first_range_unmapped = reject_first_hold = 0;
    vk_usnap_reclaim_reusable();
    VkReplayCaptureBundle bundle = {0};
    VkReplayCaptureWait wait;
    VkUsnapCapture *inventory = NULL;
    assert(vk_usnap_preflight(inputs, &inventory, &wait) == VK_SUCCESS);
    uint64_t metadata = capture_metadata_bytes(inventory);
    vk_usnap_cancel(inventory);
    assert(vk_usnap_reclaim_reusable() == 1);
    uint64_t gather = gather_bytes();
    unsigned before_reserves = host_reservations, before_allocations = allocations;
    unsigned before_flushes = flushes, before_materializations = scratch_materializations;
    memset(guest, 0x19, guest_size);

    refuse_receipt_bytes = sizeof(VkDrawScratchCredit);
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_NOT_READY);
    assert(wait.kind == VK_REPLAY_CAPTURE_WAIT_BUDGET && !bundle.scratch && !bundle.snapshot);
    assert(wait.peak.amount[RC_BUDGET_RAM] == sizeof(VkDrawScratchCredit) &&
           wait.peak.amount[RC_BUDGET_COMMIT] == sizeof(VkDrawScratchCredit));
    assert(budget.retained.amount[RC_BUDGET_RAM] == gather && !budget.reserved.amount[RC_BUDGET_RAM]);
    refuse_receipt_bytes = 0;

    // Scratch, arena, and metadata each fit. Scratch plus arena would fit without its metadata.
    budget.domain[RC_BUDGET_RAM].limit = gather + 3 * ARENA_MIN + sizeof(Arena);
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_NOT_READY);
    uint64_t joint_peak = 3 * ARENA_MIN + sizeof(Arena) + metadata + sizeof(VkDrawScratchCredit);
    assert(wait.peak.amount[RC_BUDGET_RAM] == joint_peak && wait.kind == VK_REPLAY_CAPTURE_WAIT_BUDGET);
    assert(wait.peak.amount[RC_BUDGET_COMMIT] == joint_peak);
    assert(wait.before.available.amount[RC_BUDGET_RAM] == 3 * ARENA_MIN + sizeof(Arena));
    assert(!bundle.snapshot && !bundle.scratch && !scratch_ready && !joint_native);
    assert(host_reservations == before_reserves + 1 && flushes == before_flushes);
    assert(allocations == before_allocations && scratch_materializations == before_materializations);
    assert(!budget.reserved.amount[RC_BUDGET_RAM] && budget.retained.amount[RC_BUDGET_RAM] == gather && guest[0] == 0x19);
    budget.domain[RC_BUDGET_RAM].limit = UINT64_MAX;

    drain_older = 1;
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_NOT_READY);
    assert(wait.kind == VK_REPLAY_CAPTURE_WAIT_DRAIN_OLDER && !bundle.snapshot && !bundle.scratch);
    assert(host_reservations == before_reserves + 1 && allocations == before_allocations && flushes == before_flushes);
    drain_older = 0;

    // Scratch settles first. Failure to map the second allocation discharges both native and scratch ownership.
    unsigned before_destroy = destructions;
    omit_device_mapping = 1;
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_ERROR_MEMORY_MAP_FAILED);
    assert(!bundle.snapshot && !bundle.scratch && !scratch_ready && !joint_native);
    assert(allocations == before_allocations + 1 && destructions == before_destroy + 1 && flushes == before_flushes);
    assert(scratch_materializations == before_materializations + 1 && host_reservations == before_reserves + 2);
    assert(!budget.reserved.amount[RC_BUDGET_RAM] && budget.retained.amount[RC_BUDGET_RAM] == gather);
    omit_device_mapping = 0;

    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_SUCCESS);
    assert(bundle.snapshot && bundle.snapshot->arena->cpu && bundle.scratch && scratch_ready);
    assert(host_reservations == before_reserves + 3 && flushes == before_flushes + 1);
    assert(!budget.reserved.amount[RC_BUDGET_RAM] && budget.retained.amount[RC_BUDGET_RAM] == gather + joint_peak);
    memset(guest, 0x61, guest_size);
    uint64_t fence = completed + 1;
    vk_usnap_begin(&bundle.snapshot, inputs->nhandles, fence);
    assert(!bundle.snapshot && vk_usnap_bytes((uintptr_t)guest, 1)[0] == 0x35);
    assert(budget.retained.amount[RC_BUDGET_RAM] == gather + joint_peak - sizeof(VkUsnapCapture));
    end_completed_snapshot(fence);
    vk_usnap_queue_bundle_cancel(&bundle);
    vk_usnap_queue_bundle_cancel(&bundle);
    assert(!scratch_ready && free_arenas &&
           budget.retained.amount[RC_BUDGET_RAM] == gather + 2 * ARENA_MIN + sizeof(Arena));

    before_allocations = allocations;
    before_reserves = host_reservations;
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_SUCCESS);
    assert(wait.peak.amount[RC_BUDGET_RAM] == ARENA_MIN && allocations == before_allocations);
    assert(host_reservations == before_reserves + 1 && budget.retained.amount[RC_BUDGET_RAM] == gather + joint_peak);
    vk_usnap_queue_bundle_cancel(&bundle);

    expect_scratch_ready = 0;
    before_reserves = host_reservations;
    assert(vk_usnap_queue_bundle(inputs, 0, &bundle, &wait) == VK_SUCCESS && bundle.snapshot && !bundle.scratch);
    assert(!wait.peak.amount[RC_BUDGET_RAM] && host_reservations == before_reserves && allocations == before_allocations);
    vk_usnap_queue_bundle_cancel(&bundle);
    assert(vk_usnap_reclaim_reusable() == 1 && !joint_native);

    U.mode = 0;
    snapshot_mode = ab_phase = 0;
    before_flushes = flushes;
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_SUCCESS);
    assert(bundle.snapshot && !bundle.snapshot->enabled && !bundle.snapshot->arena && bundle.scratch);
    assert(flushes == before_flushes);
    vk_usnap_queue_bundle_cancel(&bundle);
    before_reserves = host_reservations;
    assert(vk_usnap_queue_bundle(&(VkUsnapInputs){0}, 0, &bundle, &wait) == VK_SUCCESS);
    assert(!bundle.snapshot && !bundle.scratch && host_reservations == before_reserves && flushes == before_flushes);

    stopping = 1;
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_ERROR_INITIALIZATION_FAILED);
    assert(!bundle.snapshot && !bundle.scratch && host_reservations == before_reserves);
    assert(!budget.reserved.amount[RC_BUDGET_RAM] && budget.retained.amount[RC_BUDGET_RAM] == gather);
    stopping = joint_testing = U.mode = 0;
    snapshot_mode = 1;
}

static void check_queue_compatible_arena(const VkUsnapInputs *inputs) {
    const unsigned arena_heap = RC_BUDGET_GPU_FIRST, scratch_heap = arena_heap + 1;
    const uint32_t both_heaps = (1u << arena_heap) | (1u << scratch_heap);
    U.mode = 0;
    snapshot_mode = temporal = joint_testing = expect_scratch_ready = queue_reuse_model = 1;
    VkReplayCaptureBundle bundle = {0};
    VkReplayCaptureWait wait;
    assert(!free_arenas && !joint_native);
    uint64_t gather = gather_bytes();
    unsigned probes = compatible_probes, grows = queued_growth_attempts;

    // With known heap room, keep the original jointly reserved dedicated path.
    known_heaps = both_heaps;
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_SUCCESS);
    assert(queued_growth_attempts == ++grows && compatible_probes == probes);
    uint64_t metadata = capture_metadata_bytes(bundle.snapshot) + sizeof(VkDrawScratchCredit);
    vk_usnap_queue_bundle_cancel(&bundle);
    assert(vk_usnap_reclaim_reusable() == 1);

    // Both heaps have room. Only the native arena's new commit would exceed the limit.
    uint64_t reuse_commit = 2 * ARENA_MIN + sizeof(Arena);
    budget.domain[RC_BUDGET_COMMIT].limit = gather + metadata + reuse_commit;
    queue_compatible = 1;
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_SUCCESS);
    assert(compatible_probes == ++probes && queued_growth_attempts == grows);
    assert(!wait.peak.amount[arena_heap] && wait.peak.amount[scratch_heap] == ARENA_MIN);
    assert(wait.peak.amount[RC_BUDGET_COMMIT] == reuse_commit);
    vk_usnap_queue_bundle_cancel(&bundle);
    assert(vk_usnap_reclaim_reusable() == 1);

    // A miss preserves the original dedicated wait. With even less commit, skip the probe.
    queue_compatible = 0;
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_NOT_READY);
    assert(compatible_probes == ++probes && queued_growth_attempts == grows);
    assert(wait.kind == VK_REPLAY_CAPTURE_WAIT_BUDGET && wait.peak.amount[arena_heap] == ARENA_MIN);
    assert(wait.peak.amount[RC_BUDGET_COMMIT] == metadata + reuse_commit + ARENA_MIN);
    assert(!bundle.snapshot && !bundle.scratch && !joint_native && !scratch_ready);
    budget.domain[RC_BUDGET_COMMIT].limit--;
    queue_compatible = 1;
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_NOT_READY);
    assert(compatible_probes == probes && queued_growth_attempts == grows);
    assert(!bundle.snapshot && !bundle.scratch && !joint_native && !scratch_ready);
    assert(!budget.reserved.amount[RC_BUDGET_COMMIT] && budget.retained.amount[RC_BUDGET_COMMIT] == gather);
    budget.domain[RC_BUDGET_COMMIT].limit = UINT64_MAX;

    // Existing host-preferred dedicated placement remains available when its full peak fits.
    budget.domain[arena_heap].limit = 0;
    unsigned alternatives = alternate_creates;
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_SUCCESS);
    assert(queued_growth_attempts == ++grows && alternate_creates == alternatives + 1 && compatible_probes == probes);
    vk_usnap_queue_bundle_cancel(&bundle);
    assert(vk_usnap_reclaim_reusable() == 1);

    // The arena heap has no growth headroom. The known scratch heap can still grow, and the
    // compatible arena needs its RAM working set, DLAA shadow and header but no new native commit.
    budget.domain[scratch_heap].limit = ARENA_MIN;
    queue_compatible = 1;
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_SUCCESS);
    assert(compatible_probes == ++probes && queued_growth_attempts == grows);
    assert(!wait.peak.amount[arena_heap] && wait.peak.amount[scratch_heap] == ARENA_MIN);
    assert(wait.peak.amount[RC_BUDGET_RAM] == 3 * ARENA_MIN + sizeof(Arena));
    assert(wait.peak.amount[RC_BUDGET_COMMIT] == 2 * ARENA_MIN + sizeof(Arena));
    assert(budget.retained.amount[arena_heap] == ARENA_MIN && budget.retained.amount[scratch_heap] == ARENA_MIN);
    assert(!vk_usnap_reclaim_reusable());
    vk_usnap_queue_bundle_cancel(&bundle);
    assert(budget.retained.amount[arena_heap] == ARENA_MIN && !budget.retained.amount[scratch_heap]);
    RcAdmissionVector deficit = {{0}};
    deficit.amount[arena_heap] = 1;
    assert(snapshot_reclaimer(NULL, &deficit) == RC_BUDGET_RECLAIM_DESTROYED);
    assert(!budget.retained.amount[arena_heap] && !joint_native);

    // An incompatible VMA block keeps the original growth wait and releases the private reservation.
    queue_compatible = 0;
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_NOT_READY);
    assert(compatible_probes == ++probes && queued_growth_attempts == grows);
    assert(wait.kind == VK_REPLAY_CAPTURE_WAIT_BUDGET && wait.peak.amount[arena_heap] == ARENA_MIN);
    assert(!bundle.snapshot && !bundle.scratch && !joint_native && !scratch_ready);
    assert(!budget.reserved.amount[RC_BUDGET_RAM] && budget.retained.amount[RC_BUDGET_RAM] == gather);

    // Unknown heap allowance forbids growth. Existing arena backing plus a reusable scratch chunk
    // can still proceed, and a VMA miss keeps the original unsupported result.
    known_heaps = 0;
    queue_existing_scratch = 1;
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_ERROR_FEATURE_NOT_PRESENT);
    assert(compatible_probes == ++probes && !bundle.snapshot && !bundle.scratch && !joint_native);
    queue_compatible = 1;
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_SUCCESS);
    assert(compatible_probes == ++probes && !wait.peak.amount[arena_heap] && !wait.peak.amount[scratch_heap]);
    assert(wait.peak.amount[RC_BUDGET_COMMIT] == ARENA_MIN + sizeof(Arena));
    vk_usnap_queue_bundle_cancel(&bundle);
    assert(vk_usnap_reclaim_reusable() == 1);

    // Native scratch growth still requires a known allowance. If scratch fails after the
    // compatible arena is acquired, both owners and the joint reservation are unwound.
    queue_existing_scratch = 0;
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_ERROR_FEATURE_NOT_PRESENT);
    assert(compatible_probes == probes && !bundle.snapshot && !bundle.scratch);
    known_heaps = both_heaps;
    queue_scratch_failure = 1;
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
    assert(compatible_probes == ++probes && queued_growth_attempts == grows);
    assert(!bundle.snapshot && !bundle.scratch && !joint_native && !scratch_ready && !free_arenas);
    for (unsigned d = 0; d < budget.domain_count; d++) {
        assert(!budget.reserved.amount[d] && !budget.pending.amount[d]);
        assert(budget.retained.amount[d] == (d <= RC_BUDGET_COMMIT ? gather : 0));
        budget.domain[d].limit = UINT64_MAX;
    }
    queue_reuse_model = queue_compatible = queue_scratch_failure = joint_testing = temporal = 0;
    known_heaps = 0;
}

static void check_oversized_shadow(const VkUsnapInputs *inputs) {
    assert(!free_arenas && !joint_native);
    VkResult result;
    RcAdmissionVector refused;
    VkUsnapCapture *capture = NULL;
    temporal = 0;
    Arena *large = arena_try_get(4 * ARENA_MIN, &result, &refused);
    assert(result == VK_SUCCESS && large && large->size == 4 * ARENA_MIN && !large->cpu);
    assert(!large->native_cost.amount[RC_BUDGET_RAM]);
    arena_return(large, ARENA_RESERVED);

    temporal = joint_testing = 1;
    expect_scratch_ready = 0;
    uint64_t retained = gather_bytes() + sizeof(Arena);
    uint64_t previous_ram = budget.domain[RC_BUDGET_RAM].limit;
    unsigned before_alloc = allocations, before_grow = queued_growth_attempts;
    budget.domain[RC_BUDGET_RAM].limit = retained + ARENA_MIN;
    VkReplayCaptureWait wait;
    assert(vk_usnap_preflight(inputs, &capture, &wait) == VK_NOT_READY);
    assert(wait.kind == VK_REPLAY_CAPTURE_WAIT_BUDGET &&
           wait.peak.amount[RC_BUDGET_RAM] >= large->size);
    assert(!capture && free_arenas == large && allocations == before_alloc && !joint_native);

    // The old native arena has no releasable RAM here, but a new minimum arena and shadow fit.
    budget.domain[RC_BUDGET_RAM].limit = retained + 3 * ARENA_MIN;
    assert(vk_usnap_preflight(inputs, &capture, &wait) == VK_SUCCESS);
    assert(capture && capture->arena != large && capture->arena->size == ARENA_MIN && capture->arena->cpu);
    assert(free_arenas == large && !large->cpu && allocations == before_alloc + 1);
    assert(queued_growth_attempts == before_grow + 1 && !budget.reserved.amount[RC_BUDGET_RAM]);
    vk_usnap_cancel(capture);
    assert(vk_usnap_reclaim_reusable() == 2 && !joint_native);

    temporal = 0;
    large = arena_try_get(4 * ARENA_MIN, &result, &refused);
    assert(result == VK_SUCCESS && large && !large->cpu);
    arena_return(large, ARENA_RESERVED);
    temporal = expect_scratch_ready = 1;
    budget.domain[RC_BUDGET_RAM].limit = retained + 4 * ARENA_MIN;
    before_alloc = allocations;
    before_grow = queued_growth_attempts;
    VkReplayCaptureBundle bundle = {0};
    queue_scratch_failure = 1;
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
    assert(!bundle.snapshot && !bundle.scratch && !scratch_ready && !budget.reserved.amount[RC_BUDGET_RAM]);
    assert(free_arenas == large && allocations == before_alloc && !joint_native);

    queue_scratch_failure = 0;
    expect_reusable_at_copy = large;
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_SUCCESS);
    expect_reusable_at_copy = NULL;
    assert(bundle.snapshot && bundle.snapshot->arena != large && bundle.snapshot->arena->size == ARENA_MIN);
    assert(bundle.snapshot->arena->cpu && bundle.scratch && scratch_ready);
    assert(free_arenas == large && !large->cpu && allocations == before_alloc + 1);
    assert(wait.peak.amount[RC_BUDGET_RAM] == 3 * ARENA_MIN + sizeof(Arena));
    assert(queued_growth_attempts == before_grow + 1 && !budget.reserved.amount[RC_BUDGET_RAM]);
    vk_usnap_queue_bundle_cancel(&bundle);
    assert(vk_usnap_reclaim_reusable() == 2 && !joint_native && !scratch_ready);
    assert(budget.retained.amount[RC_BUDGET_RAM] == gather_bytes());
    budget.domain[RC_BUDGET_RAM].limit = previous_ram;
    temporal = joint_testing = expect_scratch_ready = 0;
}

static void check_oversized_shadow_heap_reclaim(const VkUsnapInputs *inputs) {
    const unsigned arena_heap = RC_BUDGET_GPU_FIRST, scratch_heap = arena_heap + 1;
    const uint32_t both_heaps = (1u << arena_heap) | (1u << scratch_heap);
    const uint64_t large_bytes = 4 * ARENA_MIN;
    const uint64_t gather = gather_bytes();
    assert(!free_arenas && !joint_native && !scratch_ready);
    queue_bar_placement = queue_reuse_model = joint_testing = 1;
    known_heaps = both_heaps;

    // Install a real retained receipt for a reusable dedicated BAR arena. Its native allocation
    // occupies a heap and system commit, while its header is the only retained RAM.
    QueuedArenaPlan large_plan = {.buffer = arena_buffer_info(large_bytes)};
    RcAdmissionVector peak;
    RcBudgetTicket joint = {0};
    Arena *large = NULL;
    assert(arena_queue_placement(&large_plan, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, &peak) == VK_SUCCESS);
    assert(resource_core_budget_reserve(RC_BUDGET_SNAPSHOT, 1, &peak, &joint) == RC_ADMISSION_OK);
    assert(arena_create(&large_plan.buffer, large_plan.placement, 0, &joint, &peak, &large) == VK_SUCCESS);
    assert(resource_core_budget_cancel(&joint) == RC_ADMISSION_OK);
    assert(!large->cpu && !large->native_cost.amount[RC_BUDGET_RAM]);
    assert(large->native_cost.amount[arena_heap] == large_bytes);
    arena_return(large, ARENA_RESERVED);

    temporal = expect_scratch_ready = 1;
    budget.domain[RC_BUDGET_RAM].limit = gather + 3 * ARENA_MIN;
    budget.domain[arena_heap].limit = large_bytes + ARENA_MIN / 2;
    // The alternate native placement shares the scratch heap and cannot fit with the scratch grant.
    budget.domain[scratch_heap].limit = ARENA_MIN;
    VkReplayCaptureBundle bundle = {0};
    VkReplayCaptureWait unknown, unclear, wait;

    known_heaps = 1u << scratch_heap;
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &unknown) == VK_NOT_READY);
    assert(unknown.kind == VK_REPLAY_CAPTURE_WAIT_BUDGET && !bundle.snapshot && !bundle.scratch);
    assert(free_arenas == large && !large->cpu);

    known_heaps = both_heaps;
    budget.domain[scratch_heap].limit = ARENA_MIN / 2;
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &unclear) == VK_NOT_READY);
    assert(unclear.kind == VK_REPLAY_CAPTURE_WAIT_BUDGET && !bundle.snapshot && !bundle.scratch);
    assert(free_arenas == large && !large->cpu);
    budget.domain[scratch_heap].limit = ARENA_MIN;

    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &wait) == VK_NOT_READY);
    assert(wait.kind == VK_REPLAY_CAPTURE_WAIT_BUDGET && !bundle.snapshot && !bundle.scratch);
    assert(free_arenas == large && !large->cpu && !scratch_ready);
    RcBudgetReadiness before, refreshed;
    resource_core_budget_readiness(1, &before);
    assert(!resource_core_budget_peak_ready(&wait.peak, &before));
    assert(wait.peak.amount[arena_heap] == ARENA_MIN && before.available.amount[arena_heap] == ARENA_MIN / 2);
    RcAdmissionVector deficit = {{0}};
    for (unsigned d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++)
        if (wait.peak.amount[d] > before.available.amount[d])
            deficit.amount[d] = wait.peak.amount[d] - before.available.amount[d];
    unsigned destroyed = destructions;
    assert(snapshot_reclaimer(NULL, &deficit) == RC_BUDGET_RECLAIM_DESTROYED);
    assert(destructions == destroyed + 1 && !free_arenas && !joint_native);
    resource_core_budget_readiness(1, &refreshed);
    assert(resource_core_budget_peak_progress(&wait.peak, &before, &refreshed));

    VkReplayCaptureWait retry;
    assert(vk_usnap_queue_bundle(inputs, ARENA_MIN, &bundle, &retry) == VK_SUCCESS);
    assert(bundle.snapshot && bundle.snapshot->arena->size == ARENA_MIN && bundle.snapshot->arena->cpu);
    assert(bundle.scratch && scratch_ready);
    const uint64_t metadata = capture_metadata_bytes(bundle.snapshot) + sizeof(VkDrawScratchCredit);
    RcAdmissionVector smaller = arena_host_cost(2 * ARENA_MIN + sizeof(Arena) + metadata);
    smaller.amount[RC_BUDGET_COMMIT] += ARENA_MIN;
    smaller.amount[arena_heap] = smaller.amount[scratch_heap] = ARENA_MIN;
    RcAdmissionVector original = arena_host_cost(large_bytes + ARENA_MIN + metadata);
    original.amount[scratch_heap] = ARENA_MIN;
    for (unsigned d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) {
        assert(wait.peak.amount[d] == smaller.amount[d]);
        assert(unknown.peak.amount[d] == original.amount[d]);
        assert(unclear.peak.amount[d] == original.amount[d]);
    }
    vk_usnap_queue_bundle_cancel(&bundle);
    assert(vk_usnap_reclaim_reusable() == 1 && !joint_native && !scratch_ready);
    for (unsigned d = 0; d < budget.domain_count; d++) {
        assert(!budget.reserved.amount[d] && !budget.pending.amount[d]);
        assert(budget.retained.amount[d] == (d <= RC_BUDGET_COMMIT ? gather : 0));
        budget.domain[d].limit = UINT64_MAX;
    }
    queue_bar_placement = queue_reuse_model = joint_testing = temporal = expect_scratch_ready = 0;
    known_heaps = 0;
}

static void check_adjacent_imported_pools(void) {
    imported_pool_boundary = 8192;
    uint8_t *left = guest + imported_pool_boundary - ALIGN;
    const p_bind_buf adjacent[] = {
        {.stage = 0, .index = 3, .addr = (uintptr_t)left, .size = ALIGN},
        {.stage = 0, .index = 4, .addr = (uintptr_t)(left + ALIGN), .size = ALIGN},
    };
    const VkUsnapInputs inputs = {.inherited = adjacent, .ninherited = 2};
    uint8_t expected[2][ALIGN], actual[ALIGN];
    memset(expected[0], 0x39, ALIGN);
    memset(expected[1], 0x68, ALIGN);
    memcpy(left, expected, sizeof expected);

    // Each UBO fits its own imported pool even though their address ranges touch.
    VkUsnapCapture *capture = vk_usnap_take(&inputs);
    assert(capture);
    memset(left, 0xa5, sizeof expected);
    uint64_t fence = completed + 1;
    vk_usnap_begin(&capture, 0, fence);
    assert(!capture);
    VkDescriptorBufferInfo buffer;
    VkUsnapLease lease;
    for (size_t i = 0; i < sizeof adjacent / sizeof *adjacent; i++) {
        assert(vk_usnap_find_lease(adjacent[i].addr, ALIGN, ALIGN, &buffer, &lease));
        assert(buffer.range == ALIGN && vk_usnap_lease_valid(&lease));
        assert(vk_usnap_read(adjacent[i].addr, ALIGN, actual) == 1);
        assert(memcmp(actual, expected[i], sizeof actual) == 0);
    }
    assert(!vk_usnap_find_lease((uintptr_t)left, sizeof expected, ALIGN, &buffer, &lease));
    end_completed_snapshot(fence);

    const p_bind_buf spanning = {.stage = 0, .index = 3, .addr = (uintptr_t)left, .size = sizeof expected};
    // A crossing overlap must not suppress the two independently capturable pool-contained bindings.
    const p_bind_buf overlap[] = {adjacent[0], adjacent[1], spanning};
    const VkUsnapInputs mixed = {.inherited = overlap, .ninherited = 3};
    capture = vk_usnap_take(&mixed);
    assert(capture && capture->enabled);
    fence = completed + 1;
    vk_usnap_begin(&capture, 0, fence);
    assert(!capture && vk_usnap_capture());
    assert(vk_usnap_find_lease(adjacent[0].addr, adjacent[0].size, ALIGN, &buffer, &lease));
    assert(vk_usnap_find_lease(adjacent[1].addr, adjacent[1].size, ALIGN, &buffer, &lease));
    assert(!vk_usnap_find_lease(spanning.addr, spanning.size, ALIGN, &buffer, &lease));
    end_completed_snapshot(fence);

    // A single UBO spanning both pools cannot supply a snapshot from either half.
    const VkUsnapInputs crossing = {.inherited = &spanning, .ninherited = 1};
    capture = vk_usnap_take(&crossing);
    assert(capture && capture->enabled);
    fence = completed + 1;
    vk_usnap_begin(&capture, 0, fence);
    assert(!capture && !vk_usnap_capture());
    assert(!vk_usnap_find_lease(spanning.addr, spanning.size, ALIGN, &buffer, &lease));
    end_completed_snapshot(fence);
    imported_pool_boundary = 0;
}

static void check_device_offset_alignment(void) {
    const p_bind_buf input = {.stage = 0, .index = 3, .addr = (uintptr_t)(guest + 4096), .size = 128};
    const VkUsnapInputs inputs = {.inherited = &input, .ninherited = 1};
    VkUsnapCapture *capture = vk_usnap_take(&inputs);
    assert(capture && capture->alignment == 512);
    uint64_t fence = completed + 1;
    vk_usnap_begin(&capture, 0, fence);
    VkDescriptorBufferInfo buffer;
    VkUsnapLease lease;
    assert(vk_usnap_find_lease(input.addr, input.size, 512, &buffer, &lease));
    assert(!(buffer.offset % 512) && buffer.range == input.size);
    end_completed_snapshot(fence);
}

static void check_pool_gap(void) {
    imported_pool_boundary = 8192;
    imported_pool_gap = 1024;
    uint64_t left = (uintptr_t)guest + imported_pool_boundary - 512;
    uint64_t right = (uintptr_t)guest + imported_pool_boundary + imported_pool_gap;
    const p_bind_buf bindings[] = {
        {.stage = 0, .index = 3, .addr = left, .size = 512},
        {.stage = 0, .index = 4, .addr = left, .size = right - left + 512},
        {.stage = 0, .index = 5, .addr = right, .size = 512},
    };
    VkUsnapInputs inputs = {.inherited = bindings, .ninherited = 3};
    for (unsigned rounded = 0; rounded <= 1; ++rounded) {
        // Import padding that extends into the later pool is still outside the first pool's logical range.
        imported_pool_rounding = rounded ? 2048 : 0;
        memset((void *)(uintptr_t)right, 0x6a, 512);
        VkUsnapCapture *capture = vk_usnap_take(&inputs);
        assert(capture && capture->nreg == 2);
        memset((void *)(uintptr_t)right, 0x97, 512);
        uint64_t fence = completed + 1;
        assert(vk_usnap_begin(&capture, 0, fence) == VK_SUCCESS);
        VkDescriptorBufferInfo buffer;
        assert(vk_usnap_find(left, 512, 512, &buffer));
        assert(vk_usnap_find(right, 512, 512, &buffer));
        assert(vk_usnap_bytes(right, 512)[511] == 0x6a);
        assert(!vk_usnap_find(left, bindings[1].size, 512, &buffer));
        end_completed_snapshot(fence);
    }
    imported_pool_boundary = imported_pool_gap = imported_pool_rounding = 0;
}

static void check_large_binding_extents(void) {
    enum { uniform_bytes = 64u << 10, storage_bytes = 96u << 10 };
    ctx.props.limits.maxStorageBufferRange = storage_bytes;
    uint64_t base = (uintptr_t)guest;
    uint64_t addr = base + 256;
    const p_bind_buf binding = {.stage = 0, .index = 3, .addr = addr, .size = 112u << 10};
    VkUsnapInputs inputs = {.inherited = &binding, .ninherited = 1};
    memset(guest, 0x37, storage_bytes);
    VkUsnapCapture *capture = vk_usnap_take(&inputs);
    assert(capture && capture->total == storage_bytes);
    memset(guest, 0x83, storage_bytes);
    uint64_t fence = completed + 1;
    assert(vk_usnap_begin(&capture, 0, fence) == VK_SUCCESS);
    VkDescriptorBufferInfo buffer;
    VkUsnapLease lease;
    assert(vk_usnap_find_lease(base, storage_bytes, 512, &buffer, &lease));
    assert(buffer.range == storage_bytes && vk_usnap_lease_valid(&lease));
    assert(vk_usnap_bytes(base, storage_bytes)[storage_bytes - 1] == 0x37);
    assert(!vk_usnap_find(addr, binding.size, 512, &buffer));
    end_completed_snapshot(fence);

    // The same bind can serve a smaller legal UBO even if its storage extent would cross a pool.
    imported_pool_boundary = 80u << 10;
    capture = vk_usnap_take(&inputs);
    assert(capture && capture->total == uniform_bytes + 512);
    fence = completed + 1;
    assert(vk_usnap_begin(&capture, 0, fence) == VK_SUCCESS);
    assert(vk_usnap_find(addr, uniform_bytes, 256, &buffer));
    assert(!vk_usnap_find(base, storage_bytes, 512, &buffer));
    end_completed_snapshot(fence);
    imported_pool_boundary = 0;

    // The inventory also honors a device whose uniform limit exceeds the storage limit.
    ctx.props.limits.maxUniformBufferRange = storage_bytes;
    ctx.props.limits.maxStorageBufferRange = uniform_bytes;
    capture = vk_usnap_take(&inputs);
    fence = completed + 1;
    assert(capture && vk_usnap_begin(&capture, 0, fence) == VK_SUCCESS);
    assert(vk_usnap_find(addr, storage_bytes, 256, &buffer));
    end_completed_snapshot(fence);
    ctx.props.limits.maxUniformBufferRange = uniform_bytes;
    ctx.props.limits.maxStorageBufferRange = 128u << 20;
}

static void check_overlapping_pool_bindings(void) {
    imported_pool_boundary = 8192;
    imported_pool_overlap = 512;
    uint64_t left = (uintptr_t)guest + imported_pool_boundary - 1024;
    uint64_t right = left + 512;
    const p_bind_buf bindings[] = {
        {.stage = 0, .index = 3, .addr = left, .size = 1024},
        {.stage = 0, .index = 4, .addr = right, .size = 1024},
    };
    VkUsnapInputs inputs = {.inherited = bindings, .ninherited = 2};
    VkUsnapCapture *capture = vk_usnap_take(&inputs);
    assert(capture && capture->nreg == 2 && capture->overlapping);
    uint64_t fence = completed + 1;
    assert(vk_usnap_begin(&capture, 0, fence) == VK_SUCCESS);
    VkDescriptorBufferInfo a, b, common;
    assert(vk_usnap_find(left, 1024, 512, &a));
    assert(vk_usnap_find(right, 1024, 512, &b) && a.offset != b.offset);
    assert(vk_usnap_find(right, 512, 512, &common) && common.offset == a.offset + 512);
    assert(!vk_usnap_find(left, 1536, 512, &common));
    end_completed_snapshot(fence);
    imported_pool_boundary = imported_pool_overlap = 0;
}

static void check_receipt_accounting(void) {
    vk_usnap_reclaim_reusable();
    uint64_t base = budget.retained.amount[RC_BUDGET_RAM];
    uint64_t addr = (uintptr_t)guest + 16384;
    const p_bind_buf binding = {.stage = 0, .index = 3, .addr = addr, .size = 1024};
    const VkUsnapInputs inputs = {.inherited = &binding, .ninherited = 1};
    size_t host = sizeof(MemBackingCapture) + sizeof(RcInterval);
    size_t regular = sizeof(VkMemSpanCapture) + host;
    VkUsnapCapture *capture = NULL;
    VkReplayCaptureWait wait;
    unsigned plans = receipt_plans, fills = receipt_fills, before_flushes = flushes;
    refuse_receipt_bytes = regular;
    assert(vk_usnap_preflight(&inputs, &capture, &wait) == VK_NOT_READY);
    assert(wait.kind == VK_REPLAY_CAPTURE_WAIT_BUDGET && !capture);
    uint64_t peak = sizeof(VkUsnapCapture) + 1024 * sizeof(Reg) + 64 * sizeof(ReceiptPlan) + regular;
    assert(wait.peak.amount[RC_BUDGET_RAM] == peak && wait.peak.amount[RC_BUDGET_COMMIT] == peak);
    assert(receipt_plans == plans + 1 && receipt_fills == fills && flushes == before_flushes);
    assert(!mappings_live && !imports_live && budget.retained.amount[RC_BUDGET_RAM] == base);
    refuse_receipt_bytes = 0;

    plans = receipt_plans;
    change_receipt_count = 1;
    assert(vk_usnap_preflight(&inputs, &capture, &wait) == VK_SUCCESS);
    assert(capture && capture->receipt_bytes == regular + sizeof(RcInterval));
    assert(receipt_plans == plans + 2 && receipt_fills == fills + 2);
    assert(mappings_live == 1 && imports_live == 1 && !capture->receipts);
    vk_usnap_cancel(capture);
    capture = NULL;
    assert(!mappings_live && !imports_live && vk_usnap_reclaim_reusable() == 1);
    assert(budget.retained.amount[RC_BUDGET_RAM] == base);
    extra_receipt_intervals = 0;

    nested_import_receipt = truncate_receipt_fill = 1;
    unsigned nested = nested_receipt_fills;
    plans = receipt_plans;
    assert(vk_usnap_preflight(&inputs, &capture, &wait) == VK_SUCCESS);
    assert(capture && capture->receipt_bytes == regular + host);
    assert(receipt_plans == plans + 2 && nested_receipt_fills == nested + 1);
    assert(mappings_live == 2 && imports_live == 1);
    uint64_t retained = budget.retained.amount[RC_BUDGET_RAM];
    uint64_t fence = completed + 1;
    assert(vk_usnap_begin(&capture, 0, fence) == VK_SUCCESS);
    assert(budget.retained.amount[RC_BUDGET_RAM] == retained - sizeof(VkUsnapCapture));
    uint8_t check[512];
    plans = receipt_plans;
    assert(vk_usnap_read(addr, sizeof check, check) == 1 && receipt_plans == plans);
    end_completed_snapshot(fence);
    assert(!mappings_live && !imports_live && vk_usnap_reclaim_reusable() == 1);
    assert(budget.retained.amount[RC_BUDGET_RAM] == base);
    nested_import_receipt = 0;

    // A larger fresh read owns its heap temporary, receipts and coverage only for this read.
    fence = completed + 1;
    assert(vk_usnap_begin(&capture, 0, fence) == VK_SUCCESS);
    memset((void *)(uintptr_t)addr, 0x39, sizeof check);
    memset(check, 0x91, sizeof check);
    refuse_receipt_bytes = regular + sizeof(RcSourceRange) + sizeof check;
    fills = receipt_fills;
    assert(!vk_usnap_read(addr, sizeof check, check) && check[0] == 0x91 && receipt_fills == fills);
    refuse_receipt_bytes = 0;
    assert(vk_usnap_read(addr, sizeof check, check) == 2 && check[sizeof check - 1] == 0x39);
    assert(!mappings_live && !imports_live && budget.retained.amount[RC_BUDGET_RAM] == base);
    end_completed_snapshot(fence);
}

static void check_fragmented_permission_admission(void) {
    const uint64_t addr = (uintptr_t)guest + 20480;
    const p_bind_buf binding = {.stage = 0, .index = 3, .addr = addr, .size = 1024};
    const VkUsnapInputs inputs = {.inherited = &binding, .ninherited = 1};
    extra_receipt_intervals = PERMISSION_LOCAL_RANGES;
    uint32_t count = 1 + extra_receipt_intervals;
    uint64_t scratch = permission_scratch_bytes(count);
    uint64_t receipts = sizeof(MemBackingCapture) + count * sizeof(RcInterval) + sizeof(VkMemSpanCapture);
    refuse_receipt_bytes = receipts + scratch;
    VkUsnapCapture *value = NULL;
    VkReplayCaptureWait wait;
    unsigned fills = receipt_fills;
    assert(vk_usnap_preflight(&inputs, &value, &wait) == VK_NOT_READY);
    assert(wait.kind == VK_REPLAY_CAPTURE_WAIT_BUDGET && !value && receipt_fills == fills);
    assert(wait.peak.amount[RC_BUDGET_RAM] == sizeof(VkUsnapCapture) + 1024 * sizeof(Reg) +
        64 * sizeof(ReceiptPlan) + receipts + scratch);
    refuse_receipt_bytes = 0;
    assert(vk_usnap_preflight(&inputs, &value, &wait) == VK_SUCCESS);
    assert(value && value->permission_ranges && value->permission_capacity == count);
    uint64_t fence = completed + 1;
    assert(vk_usnap_begin(&value, 0, fence) == VK_SUCCESS);
    budget.domain[RC_BUDGET_RAM].limit = budget.retained.amount[RC_BUDGET_RAM];
    unsigned reserves = budget_reservations;
    VkUsnapLease lease;
    VkDescriptorBufferInfo info;
    assert(vk_usnap_find_lease(addr, binding.size, 256, &info, &lease));
    assert(info.range == binding.size && vk_usnap_lease_valid(&lease) && budget_reservations == reserves);
    uint32_t mapped_count;
    const RcInterval *mapped = mem_backing_intervals(U.reg[0].mapping, &mapped_count);
    assert(mapped_count == count);
    RcRegistryWrite producer;
    RcRegistryPosition write = {fence, 3}, read = {fence, 4};
    assert(rc_registry_write_begin_at(registry, mapped[count - 1].range, RC_REGISTRY_BUFFER, write,
                                      &producer) == RC_REGISTRY_OK);
    assert(rc_registry_declare_through(registry, read) == RC_REGISTRY_OK);
    assert(vk_usnap_resolve_at(addr, binding.size, 256, read, &info, NULL) == VK_USNAP_RESOLVE_ORDERED_PRODUCER);
    assert(vk_usnap_resolve_at(addr, mapped[count - 1].address - addr, 256, read, &info, NULL) ==
           VK_USNAP_RESOLVE_READY);
    VkGpaPrefixRequest request = {0};
    assert(!vk_usnap_gpa_blocker(addr, binding.size, 256, read, &request));
    assert(rc_registry_write_cancel(registry, producer) == RC_REGISTRY_OK && budget_reservations == reserves);
    write.ordinal = 5; read.ordinal = 6;
    assert(rc_registry_write_guard_begin_reason_at(registry, RC_REGISTRY_BUFFER, write,
           RC_REGISTRY_GUARD_GPA_PROJECTION, &producer) == RC_REGISTRY_OK);
    assert(rc_registry_declare_through(registry, read) == RC_REGISTRY_OK);
    assert(vk_usnap_gpa_blocker(addr + 256, 16, 256, read, &request));
    assert(request.guard.id == producer.id && request.guard.index == producer.index &&
           !rc_registry_position_compare(request.producer, write) && !rc_registry_position_compare(request.read, read));
    assert(request.source.address == addr + 256 && request.source.mapping == mapping_id &&
           rc_range_equal(request.source.range, (RcSourceRange){2, 20480 + 256, 16}));
    assert(!vk_usnap_gpa_blocker(addr + 257, 16, 256, read, &request));
    assert(!vk_usnap_gpa_blocker(addr + 256, binding.size, 256, read, &request));
    assert(!vk_usnap_gpa_blocker(addr, binding.size, 256, (RcRegistryPosition){fence, 4}, &request));
    assert(!vk_usnap_gpa_blocker(addr, binding.size, 256, (RcRegistryPosition){fence + 1, 6}, &request));
    mapping_id++;
    assert(!vk_usnap_gpa_blocker(addr, binding.size, 256, read, &request));
    mapping_id--;
    assert(rc_registry_write_cancel(registry, producer) == RC_REGISTRY_OK);
    assert(!vk_usnap_gpa_blocker(addr, binding.size, 256, read, &request));
    assert(vk_usnap_resolve_at(addr, binding.size, 256, read, &info, NULL) == VK_USNAP_RESOLVE_READY);
    assert(budget_reservations == reserves);
    end_completed_snapshot(fence);
    assert(!U.permission_ranges && !U.permission_capacity && !mappings_live && !imports_live);
    budget.domain[RC_BUDGET_RAM].limit = UINT64_MAX;
    extra_receipt_intervals = 0;
}

int main(void) {
    RcAdmissionDomain domains[] = {
        {RC_ADMISSION_RAM, 0, UINT64_MAX}, {RC_ADMISSION_COMMIT, 0, UINT64_MAX},
        {RC_ADMISSION_STAGING, 0, UINT64_MAX},
        {RC_ADMISSION_GPU_HEAP, 0, UINT64_MAX}, {RC_ADMISSION_GPU_HEAP, 1, UINT64_MAX}
    };
    assert(rc_admission_init(&budget, 1, domains, sizeof domains / sizeof *domains, budget_records, 32) == RC_ADMISSION_OK);
    ctx.props.limits.minUniformBufferOffsetAlignment = 256;
    ctx.props.limits.minStorageBufferOffsetAlignment = 512;
    ctx.props.limits.maxUniformBufferRange = 64u << 10;
    ctx.props.limits.maxStorageBufferRange = 128u << 20;
    enum { recorder_slots = VK_REPLAY_WORKER_LIMIT + 2, observation_slots = recorder_slots + 4 };
    registry = rc_registry_create((RcRegistryConfig){128, 64, observation_slots});
    assert(registry);
    guest_size = 256u << 10;
    uint8_t *allocation = malloc(guest_size + 1024);
    assert(allocation);
    guest = (uint8_t *)(((uintptr_t)allocation + 1023) & ~(uintptr_t)1023);
    memset(guest, 0x11, guest_size);
    struct { op_hdr begin, bind; p_bind_buf b; op_hdr bind2; p_bind_buf b2; op_hdr end; } cmd = {
        {OP_BEGIN, 8}, {OP_BIND_UBO, 8 + sizeof(p_bind_buf)}, {0, 3, (uintptr_t)guest, 1024},
        {OP_BIND_UBO, 8 + sizeof(p_bind_buf)}, {0, 4, (uintptr_t)guest + 2048, 512}, {OP_END, 8}};
    uint64_t handle = (uintptr_t)&cmd;
    const VkUsnapInputs inputs = {.handles = &handle, .nhandles = 1};
    VkDescriptorBufferInfo bi;
    VkUsnapLease lease;
    begin_snapshot(&handle, 1, 1);
    assert(vk_usnap_find_lease((uintptr_t)guest + 256, 128, 256, &bi, &lease));
    assert(vk_usnap_lease_valid(&lease) && bi.range == 128 && bi.offset == 256);
    VkBuffer first_buffer = bi.buffer;
    assert(vk_usnap_bytes((uintptr_t)guest, 1)[0] == 0x11);
    uint64_t capture = vk_usnap_capture(), gen = vk_usnap_gen();
    vk_usnap_drop((uintptr_t)guest + 300, 4);
    assert(vk_usnap_capture() == capture && vk_usnap_gen() != gen && vk_usnap_lease_valid(&lease));
    VkUsnapProof proof;
    vk_usnap_proof_begin(&proof);
    assert(!vk_usnap_proof_add(&proof, &lease) && !vk_usnap_proof_end(&proof) && !vk_usnap_proof_valid(&proof));

    // This guest alias maps to the same backing bytes that the first draw bound.
    MemBackingCapture *alias = NULL;
    assert(mem_backing_capture((uintptr_t)guest + guest_size / 2 + 300, 4, &alias) == RC_BACKING_OK);
    uint32_t nalias;
    RcSourceRange written = mem_backing_intervals(alias, &nalias)[0].range;
    assert(nalias == 1 && written.backing == 2 && written.offset == 300);
    mem_backing_release(&alias);
    RcRegistryWrite producer;
    assert(rc_registry_write_begin_at(registry, written, RC_REGISTRY_BUFFER, (RcRegistryPosition){1, 5},
                                      &producer) == RC_REGISTRY_OK);
    assert(rc_registry_declare_through(registry, (RcRegistryPosition){1, 6}) == RC_REGISTRY_OK);
    assert(vk_usnap_lease_valid_at(&lease, (RcRegistryPosition){1, 4}));
    assert(!vk_usnap_lease_valid_at(&lease, (RcRegistryPosition){1, 6}));
    VkUsnapLease unaffected;
    assert(vk_usnap_find_lease_at((uintptr_t)guest, 128, 256, (RcRegistryPosition){1, 6}, &bi, &unaffected));
    assert(rc_registry_write_cancel(registry, producer) == RC_REGISTRY_OK);
    assert(vk_usnap_lease_valid_at(&lease, (RcRegistryPosition){1, 6}));

    RcRegistryWrite opaque;
    assert(rc_registry_write_guard_begin_reason_at(registry, RC_REGISTRY_BUFFER, (RcRegistryPosition){1, 7},
            RC_REGISTRY_GUARD_GPA_PROJECTION, &opaque) == RC_REGISTRY_OK);
    assert(rc_registry_declare_through(registry, (RcRegistryPosition){1, 8}) == RC_REGISTRY_OK);
    assert(vk_usnap_lease_valid_at(&lease, (RcRegistryPosition){1, 6}));
    assert(!vk_usnap_lease_valid_at(&lease, (RcRegistryPosition){1, 8}));
    assert(rc_registry_write_cancel(registry, opaque) == RC_REGISTRY_OK);
    end_completed_snapshot(1);
    assert(!vk_usnap_capture() && !vk_usnap_lease_valid(&lease));

    // Reusing an arena and the exact same region index cannot revive an old lease. Indexed gathering must produce
    // the same capture as the raw command walk, including its bytes and alignment rules.
    indexed_handle = handle, epoch = 1;
    memset(guest, 0x22, guest_size);
    begin_snapshot(&handle, 1, 2);
    assert(vk_usnap_capture() != capture && !vk_usnap_lease_valid(&lease));
    assert(!vk_usnap_proof_valid(&proof));
    VkUsnapLease same_region;
    assert(vk_usnap_find_lease(lease.addr, lease.size, lease.align, &bi, &same_region));
    assert(bi.buffer == first_buffer && same_region.region == lease.region && same_region.capture != lease.capture);
    assert(vk_usnap_find_lease((uintptr_t)guest, 1024, 256, &bi, &lease));
    assert(vk_usnap_bytes((uintptr_t)guest, 1)[0] == 0x22);
    for (uint64_t off = 0; off < 1400; off += 7) {
        for (uint64_t n = 0; n < 600; n += 13) {
            int expected = n && off + n <= 1024 && off % 16 == 0;
            VkUsnapLease probe;
            int got = vk_usnap_find_lease((uintptr_t)guest + off, n, 16, &bi, &probe);
            assert(got == expected && vk_usnap_lease_valid(&probe) == expected);
        }
    }
    VkUsnapLease bad;
    assert(!vk_usnap_find_lease(UINT64_MAX - 7, 32, 16, &bi, &bad));
    assert(!vk_usnap_find_lease((uintptr_t)guest, 16, 0, &bi, &bad));
    end_completed_snapshot(2);

    // Async capture is taken before the game overwrites its uniform block and adopted later by the replay.
    VkUsnapCapture *queued = vk_usnap_take(&inputs);
    assert(queued);
    memset(guest, 0x33, guest_size);
    assert(rc_registry_cpu_changed(registry, (RcSourceRange){2, 0, 1024}) == RC_REGISTRY_OK);
    vk_usnap_begin(&queued, 1, 3);
    assert(!queued);
    assert(vk_usnap_bytes((uintptr_t)guest, 1)[0] == 0x22);
    assert(!vk_usnap_lease_valid(&lease));
    end_completed_snapshot(3);

    // A remapped queued input cannot fall through to a fresh read of its replacement.
    VkUsnapCapture *remapped = vk_usnap_take(&inputs);
    assert(remapped && remapped->producer.read.id);
    mapping_id++;
    vk_usnap_begin(&remapped, 1, 4);
    uint8_t read_byte = 0x9a;
    assert(!vk_usnap_bytes((uintptr_t)guest, 1));
    assert(vk_usnap_resolve_at((uintptr_t)guest, 1, 1, vk_exec_resource_position(), &bi, NULL) ==
           VK_USNAP_RESOLVE_MAPPING_STALE);
    assert(!vk_usnap_read((uintptr_t)guest, 1, &read_byte) && read_byte == 0x9a);
    end_completed_snapshot(4);

    // An unused queued scope discharges its registry token and imported mapping owners.
    VkUsnapCapture *unused = vk_usnap_take(&inputs);
    assert(unused && unused->producer.read.id && imports_live && mappings_live);
    RcRegistryGpuCapture canceled_token = unused->producer.read;
    vk_usnap_cancel(unused);
    assert(!imports_live && !mappings_live);
    RcSourceRange clipped = {2, 0, 1};
    assert(rc_registry_gpu_capture_read_at(registry, canceled_token, &clipped, 1, (RcRegistryPosition){5, 2}) ==
           RC_REGISTRY_STALE);

    // A full observation table refuses before creating or flushing an arena or publishing a queue owner.
    RcRegistryGpuCapture full[observation_slots] = {0};
    for (unsigned i = 0; i < sizeof full / sizeof full[0]; i++)
        assert(rc_registry_gpu_capture_begin(registry, &clipped, 1, &full[i]) == RC_REGISTRY_OK);
    VkReplayCaptureBundle waiting_bundle = {0};
    VkReplayCaptureWait wait;
    unsigned refused_flushes = flushes, refused_allocations = allocations;
    VkUsnapCapture *waiting_capture = NULL;
    assert(vk_usnap_preflight(&inputs, &waiting_capture, &wait) == VK_NOT_READY);
    assert(wait.kind == VK_REPLAY_CAPTURE_WAIT_PRODUCER && !waiting_capture);
    uint64_t refused_epoch = wait.producer_epoch;
    assert(vk_usnap_queue_bundle(&inputs, 0, &waiting_bundle, &wait) == VK_NOT_READY);
    assert(wait.kind == VK_REPLAY_CAPTURE_WAIT_PRODUCER && wait.producer_epoch == refused_epoch);
    assert(!waiting_bundle.snapshot && !waiting_bundle.scratch && !imports_live && !mappings_live);
    assert(refused_flushes == flushes && refused_allocations == allocations);
    // Releasing a foreground slot advances progress, yet leaves the queue still full.
    assert(rc_registry_gpu_capture_end(registry, &full[0]) == RC_REGISTRY_OK);
    assert(vk_usnap_queue_bundle(&inputs, 0, &waiting_bundle, &wait) == VK_NOT_READY);
    assert(wait.kind == VK_REPLAY_CAPTURE_WAIT_PRODUCER && wait.producer_epoch != refused_epoch);
    assert(vk_usnap_preflight(&inputs, &waiting_capture, &wait) == VK_SUCCESS && waiting_capture);
    assert(!gpu_capture_of(registry, waiting_capture->producer.read)->active && flushes == refused_flushes);
    memset(guest, 0x54, guest_size);
    assert(vk_usnap_begin(&waiting_capture, 1, 5) == VK_SUCCESS);
    assert(!waiting_capture && vk_usnap_bytes((uintptr_t)guest, 1)[0] == 0x54);
    end_completed_snapshot(5);
    for (unsigned i = 1; i < sizeof full / sizeof full[0]; i++)
        assert(rc_registry_gpu_capture_end(registry, &full[i]) == RC_REGISTRY_OK);

    assert(owner_acquires == 0);
    temporal = 1;
    begin_snapshot(&handle, 1, 8);
    assert(U.owners == &owner_snapshot && owner_snapshot.refs == 1);
    end_completed_snapshot(8);
    async_replay = 1;
    VkUsnapCapture *queued_owner = vk_usnap_take(&inputs);
    assert(queued_owner);
    vk_usnap_begin(&queued_owner, 1, 9);
    assert(!queued_owner);
    assert(U.owners == &owner_snapshot && owner_snapshot.refs == 1 && owner_acquires == 2);
    end_completed_snapshot(9);

    // Each queue item owns its bytes even when adjacent items reuse the same command handles.
    memset(guest, 0x41, guest_size);
    VkUsnapCapture *first = vk_usnap_take(&inputs);
    memset(guest, 0x42, guest_size);
    VkUsnapCapture *second = vk_usnap_take(&inputs);
    assert(first && second && first->id != second->id && first->arena != second->arena);
    memset(guest, 0x43, guest_size);
    VkUsnapCapture *canceled = vk_usnap_take(&inputs);
    assert(canceled && canceled->arena != first->arena && canceled->arena != second->arena);
    vk_usnap_cancel(canceled);
    vk_usnap_begin(&first, 1, 10);
    assert(!first && vk_usnap_bytes((uintptr_t)guest, 1)[0] == 0x41);
    vk_usnap_end(10);
    vk_usnap_begin(&second, 1, 11);
    assert(!second && vk_usnap_bytes((uintptr_t)guest, 1)[0] == 0x42);
    vk_usnap_end(11);
    assert(ncallbacks == 2 && callbacks[0].fence == 10 && callbacks[1].fence == 11);
    unsigned before_destroy = destructions;
    assert(vk_usnap_reclaim_reusable() == 1 && destructions == before_destroy + 1);
    assert(ncallbacks == 2 && !vk_usnap_reclaim_reusable());
    handed = 10; // A split part reached the driver without signaling its scene fence.
    cancel_snapshot(1, 0); // Fence 11 never reached the driver.
    assert(vk_usnap_reclaim_reusable() == 1);
    assert(ncallbacks == 1 && !vk_usnap_reclaim_reusable());
    cancel_snapshot(0, 1); // Physical device idle discharges fence 10.
    assert(vk_usnap_reclaim_reusable() == 1 && !ncallbacks);
    assert(budget.retained.amount[RC_BUDGET_RAM] == gather_bytes() && !budget.pending.amount[RC_BUDGET_RAM]);

    VkUsnapCapture *unrecorded = vk_usnap_take(&inputs);
    assert(unrecorded && owner_snapshot.refs == 1);
    vk_usnap_cancel(unrecorded);
    assert(!owner_snapshot.refs && vk_usnap_reclaim_reusable() == 1);

    // A shadowless arena can keep its native allocation when a new CPU shadow is refused.
    temporal = 0;
    begin_snapshot(&handle, 1, 12);
    Arena *shadowless = U.arena;
    end_completed_snapshot(12);
    RcAdmissionVector peak;
    VkUsnapCapture *preflight = NULL;
    assert(preflight_with_peak(&inputs, &preflight, &peak) == VK_SUCCESS);
    uint64_t metadata = capture_metadata_bytes(preflight);
    vk_usnap_cancel(preflight);
    preflight = NULL;
    unsigned before_alloc = allocations, before_host = host_reservations;
    unsigned before_flush = flushes, before_materialize = host_materializations;
    temporal = 1;
    budget.domain[RC_BUDGET_RAM].limit = budget.retained.amount[RC_BUDGET_RAM] + shadowless->size;
    assert(preflight_with_peak(&inputs, &preflight, &peak) == VK_NOT_READY);
    assert(!preflight && free_arenas == shadowless && allocations == before_alloc);
    assert(flushes == before_flush && host_materializations == before_materialize);
    assert(peak.amount[RC_BUDGET_RAM] == shadowless->size + metadata &&
           peak.amount[RC_BUDGET_COMMIT] == shadowless->size + metadata &&
           !budget.reserved.amount[RC_BUDGET_RAM] &&
           budget.retained.amount[RC_BUDGET_RAM] == gather_bytes() + sizeof(Arena));
    budget.domain[RC_BUDGET_RAM].limit = UINT64_MAX;
    begin_snapshot(&handle, 1, 13);
    assert(allocations == before_alloc && host_reservations == before_host + 2);
    assert(U.arena == shadowless && U.arena->cpu &&
           budget.retained.amount[RC_BUDGET_RAM] == gather_bytes() + sizeof(Arena) +
               U.arena->size + metadata - sizeof(VkUsnapCapture));
    assert(U.arena->cpu[0] == U.arena->map[0] && host_materializations == before_materialize + 1);
    end_completed_snapshot(13);
    before_host = host_reservations;
    begin_snapshot(&handle, 1, 14);
    end_completed_snapshot(14);
    assert(host_reservations == before_host);

    // Hold the complete arena while obtaining a shadowless sibling, then return the sibling last.
    VkUsnapCapture *ready = vk_usnap_take(&inputs);
    assert(ready && ready->arena == shadowless);
    temporal = 0;
    VkUsnapCapture *bare = vk_usnap_take(&inputs);
    assert(bare && !bare->arena->cpu && bare->arena != ready->arena);
    Arena *bare_arena = bare->arena;
    vk_usnap_cancel(ready);
    vk_usnap_cancel(bare);
    assert(free_arenas == bare_arena);
    before_alloc = allocations;
    before_host = host_reservations;
    temporal = 1;
    budget.domain[RC_BUDGET_RAM].limit = budget.retained.amount[RC_BUDGET_RAM] + metadata + 64 * sizeof(ReceiptPlan);
    assert(preflight_with_peak(&inputs, &preflight, &peak) == VK_SUCCESS);
    assert(preflight && preflight->arena == shadowless && allocations == before_alloc &&
           host_reservations == before_host);
    vk_usnap_cancel(preflight);
    preflight = NULL;
    budget.domain[RC_BUDGET_RAM].limit = UINT64_MAX;

    // A synchronous reservation copies bytes only when recording reaches the observation cutoff.
    memset(guest, 0x61, guest_size);
    before_flush = flushes;
    assert(preflight_with_peak(&inputs, &preflight, &peak) == VK_SUCCESS && preflight);
    assert(!preflight->owners && flushes == before_flush);
    temporal = 0;
    memset(guest, 0x62, guest_size);
    vk_usnap_begin(&preflight, 1, 15);
    assert(!preflight && U.arena->cpu_current && U.owners && flushes == before_flush + 1);
    assert(vk_usnap_bytes((uintptr_t)guest, 1)[0] == 0x62 && U.arena->map[0] == 0x62);
    end_completed_snapshot(15);

    // An arena with an old shadow returns its current mapped bytes when there are no host readers.
    memset(guest, 0x55, guest_size);
    begin_snapshot(&handle, 1, 16);
    assert(U.arena == shadowless && U.arena->cpu && !U.arena->cpu_current);
    assert(vk_usnap_bytes((uintptr_t)guest, 1)[0] == 0x55);
    end_completed_snapshot(16);

    // A capture made during an earlier write cannot expose its pre-write bytes after completion.
    // A newer capture of the same range after completion may expose the writeback bytes.
    memset(guest, 0x71, guest_size);
    begin_snapshot(&handle, 1, 17);
    RcRegistryWrite writeback;
    RcRegistryPosition output = {17, 3};
    assert(rc_registry_write_begin_at(registry, (RcSourceRange){2, 0, 1024}, RC_REGISTRY_BUFFER, output,
                                      &writeback) == RC_REGISTRY_OK);
    assert(rc_registry_declare_through(registry, output) == RC_REGISTRY_OK);
    assert(rc_registry_handoff_through(registry, output) == RC_REGISTRY_OK);
    assert(rc_registry_write_submit(registry, writeback, 17) == RC_REGISTRY_OK);
    vk_usnap_end(17);
    handed = 17;
    VkUsnapCapture *before_writeback = vk_usnap_take(&inputs);
    assert(before_writeback && completed == 16);
    memset(guest, 0x73, 1024);
    assert(rc_registry_write_complete(registry, writeback, 1) == RC_REGISTRY_OK);
    complete_snapshots(17);
    VkUsnapCapture *after_writeback = vk_usnap_take(&inputs);
    assert(after_writeback && after_writeback->id != before_writeback->id);
    vk_usnap_begin(&after_writeback, 1, 18);
    assert(!after_writeback && vk_usnap_bytes((uintptr_t)guest, 1)[0] == 0x73);
    end_completed_snapshot(18);
    vk_usnap_begin(&before_writeback, 1, 19);
    assert(!before_writeback && !vk_usnap_bytes((uintptr_t)guest, 1));
    assert(vk_usnap_resolve_at((uintptr_t)guest, 1, 1, vk_exec_resource_position(), &bi, NULL) ==
           VK_USNAP_RESOLVE_SOURCE_ADVANCED);
    assert(vk_usnap_bytes((uintptr_t)guest + 2048, 1)[0] == 0x71);
    read_byte = 0x9a;
    assert(vk_usnap_read((uintptr_t)guest, 1, &read_byte) == 2 && read_byte == 0x73);
    assert(vk_usnap_equal_tail((uintptr_t)guest, 1, 0, &read_byte));
    remap_on_release = 1;
    read_byte = 0x9a;
    assert(!vk_usnap_read((uintptr_t)guest, 1, &read_byte) && read_byte == 0x9a && !remap_on_release);
    end_completed_snapshot(19);
    vk_usnap_reclaim_reusable();

    // A native success with no mapping discharges the unmaterialized CPU reservation and native object.
    temporal = 1;
    omit_device_mapping = 1;
    before_alloc = allocations;
    before_destroy = destructions;
    before_materialize = host_materializations;
    before_flush = flushes;
    assert(preflight_with_peak(&inputs, &preflight, &peak) == VK_ERROR_MEMORY_MAP_FAILED);
    assert(!preflight && allocations == before_alloc + 1 && destructions == before_destroy + 1);
    assert(host_materializations == before_materialize && flushes == before_flush && !free_arenas);
    assert(!budget.reserved.amount[RC_BUDGET_RAM] && budget.retained.amount[RC_BUDGET_RAM] == gather_bytes());

    // A demand-pressure retry observes changed input mappings and bytes before taking the capture.
    omit_device_mapping = 0;
    pressure_action = PRESSURE_RETRY;
    budget.domain[RC_BUDGET_RAM].limit = 0;
    VkUsnapCapture *retried = vk_usnap_take(&inputs);
    assert(retried && pressure_calls == 1 && queue_flushes == 1 && flushes == before_flush + 1);
    vk_usnap_begin(&retried, 1, 20);
    assert(!retried && !vk_usnap_bytes((uintptr_t)guest, 1));
    assert(vk_usnap_bytes((uintptr_t)guest + 2048, 1)[0] == 0x72);
    end_completed_snapshot(20);
    vk_usnap_reclaim_reusable();

    pressure_action = PRESSURE_STOP;
    budget.domain[RC_BUDGET_RAM].limit = 0;
    assert(!vk_usnap_take(&inputs) && stopping && pressure_calls == 2 && queue_flushes == 2);
    assert(!owner_snapshot.refs && !free_arenas);
    budget.domain[RC_BUDGET_RAM].limit = UINT64_MAX;
    stopping = pressure_action = 0;

    // Disabled replay transports an explicit item with no captured bytes.
    U.mode = 0;
    snapshot_mode = 2;
    VkUsnapCapture *disabled = vk_usnap_take(&inputs);
    assert(disabled && !disabled->arena && !vk_usnap_take(&(VkUsnapInputs){0}));
    before_alloc = allocations;
    ab_phase = 1;
    vk_usnap_begin(&disabled, 1, 21);
    assert(!disabled && !vk_usnap_capture() && allocations == before_alloc);
    end_completed_snapshot(21);

    // Successful preflight cannot authorize dereferencing a range whose mapping hold later fails.
    first_range_unmapped = 0;
    assert(preflight_with_peak(&inputs, &preflight, &peak) == VK_SUCCESS && preflight);
    unsigned before_failed_hold = failed_holds;
    reject_first_hold = 1;
    vk_usnap_begin(&preflight, 1, 22);
    assert(!preflight && failed_holds == before_failed_hold + 1 && !read_holds);
    assert(!vk_usnap_bytes((uintptr_t)guest, 1));
    assert(vk_usnap_bytes((uintptr_t)guest + 2048, 1)[0] == 0x72);
    end_completed_snapshot(22);
    vk_usnap_reclaim_reusable();
    check_host_placement(&inputs);
    check_inherited_snapshot();
    check_projected_snapshot();
    check_queue_bundle(&inputs);
    check_queue_compatible_arena(&inputs);
    check_oversized_shadow(&inputs);
    check_oversized_shadow_heap_reclaim(&inputs);
    check_adjacent_imported_pools();
    check_device_offset_alignment();
    check_pool_gap();
    check_large_binding_extents();
    check_overlapping_pool_bindings();
    check_receipt_accounting();
    check_fragmented_permission_admission();

    check_flush_failure_cleanup(&inputs);

    // Losing registry authority after preflight cancels the inactive owner at begin without copying bytes.
    RcRegistry *prior_registry = registry;
    registry = rc_registry_create((RcRegistryConfig){16, 8, 2});
    assert(registry);
    VkReplayCaptureWait failed_start;
    unsigned flushes_before_failure = flushes;
    assert(vk_usnap_preflight(&inputs, &preflight, &failed_start) == VK_SUCCESS && preflight);
    rc_registry_fail_closed(registry);
    uint64_t abandoned_fence = completed + 1;
    assert(vk_usnap_begin(&preflight, 1, abandoned_fence) == VK_ERROR_INITIALIZATION_FAILED);
    assert(!preflight && !U.arena && !U.reg && !U.expected && flushes == flushes_before_failure);
    assert(!imports_live && !mappings_live && !registry->active_gpu_captures);
    vk_usnap_end(abandoned_fence);
    rc_registry_destroy(registry);
    registry = prior_registry;

    check_gather_pressure();
    vk_usnap_shutdown();
    assert(gather_retires == 6 && !U.col && !U.sort_tmp && !U.ccol && !U.sort_cap && !gather_locked);
    assert(snapshot_reclaimer && snapshot_reclaimer(NULL, &peak) == RC_BUDGET_RECLAIM_NONE);
    assert(allocations == destructions && !budget.retained.amount[RC_BUDGET_RAM] &&
           !budget.reserved.amount[RC_BUDGET_RAM] && !budget.pending.amount[RC_BUDGET_RAM]);
    assert(!mappings_live && !imports_live && !registry->active_gpu_captures);
    rc_registry_destroy(registry);
    free(allocation);
    puts("snapshot leases: canonical producer, mapping, completion, joint admission and observation passed");
    return 0;
}

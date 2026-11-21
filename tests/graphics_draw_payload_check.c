#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "platform/platform.h"
#include "resource_core_vma.h"
#include "resource_host_write.h"
#include "vk_bind_recipe.h"
#include "vk_gpa.h"
#include "nvn_api.h"
#include "vk_texture_lifetime.h"

#define RWT_COMPLETION(...)
#define RWT_CHAIN(...)
#include "graphics_payload_types_under_test.h"

typedef struct DrawCtx {
    CbRes *res;
    uint64_t res_fence, prep_incarnation;
    int mfb_stage;
    uint32_t c0_binst;
    struct {
        int sup_valid;
        float sup_in[3];
        uint8_t support[SUPPORT_BYTES];
        VkBuffer sup_buf;
        VkDeviceSize sup_off;
    } B;
} DrawCtx;
static _Thread_local DrawCtx *dc;
static _Thread_local VkExecGraphicsScratch *graphics_payload_current, *graphics_payload_bound;

static VkCtx context;
volatile int32_t vk_lost;
static uint64_t done, handed;
static int stopping, refuse, stop_during_create;

VkCtx *vk_ctx(void) { return &context; }
int vk_sync_stopping(void) { return stopping; }
int vk_exec_recording_failed(void) { return 0; }
uint64_t vk_sync_fence_done(void) { return done; }
uint64_t vk_sync_handed_fence(void) { return handed; }

static PlatMutex *held_lock;
void plat_mutex_lock(PlatMutex *lock) { assert(!held_lock); held_lock = lock; }
void plat_mutex_unlock(PlatMutex *lock) { assert(held_lock == lock); held_lock = NULL; }
void plat_cond_broadcast(PlatCond *cond) { (void)cond; }
void plat_cond_wait(PlatCond *cond, PlatMutex *lock) { (void)cond; (void)lock; abort(); }
void plat_once(PlatOnce *once, void (*fn)(void *), void *arg) {
    if (!once->state) {
        fn(arg);
        once->state = 2;
    }
}
_Noreturn void rt_fatal(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    abort();
}

void rt_log(const char *fmt, ...) { assert(fmt); }

static unsigned pin_releases;
static int index_provider;
void vk_mem_input_provider_release(VkMemInputProvider provider) {
    assert(provider.pool == (void *)&index_provider);
    pin_releases++;
}

void vk_device_lost_check(VkResult result, const char *where) { (void)result; (void)where; abort(); }

static const float settings[4];
static const float *pcss_params(void) { return settings; }
static const float *shadow_spread_params(void) { return settings; }
static int c0_on(void) { return 1; }
int vk_hdr_armed(void) { return 1; }
void vk_hdr_support(float out[4]) { memcpy(out, (const float[]){120, 0, 0, 1}, 4 * sizeof(float)); }
static uint8_t *scratch(VkExec *ex, uint64_t size, VkBuffer *buf, VkDeviceSize *off, ScratchUse use) {
    (void)ex; (void)size; (void)buf; (void)off; (void)use;
    assert(0 && "the chosen graphics layout must use its prepared support span");
    abort();
}

static RcBudgetReclaimFn reclaimer;
static void *reclaim_context;
RcAdmissionStatus resource_core_budget_register_reclaimer(RcBudgetProducer producer, uint32_t domains,
                                                          RcBudgetReclaimFn reclaim, void *arg) {
    assert(!held_lock && !reclaimer && producer == RC_BUDGET_BUFFER && (domains & (1u << RC_BUDGET_RAM)));
    reclaimer = reclaim;
    reclaim_context = arg;
    return RC_ADMISSION_OK;
}
void resource_core_budget_progress(void) { assert(!held_lock); }

typedef struct { VkBuffer buffer; uint8_t *map; VkDeviceSize size; } Native;
static Native natives[8];
static unsigned attempts, creates, destroys;
VkResult resource_core_vma_create_buffer(const RcVmaRequest *request, VmaAllocator allocator,
        const VkBufferCreateInfo *buffer, const VmaAllocationCreateInfo *allocation,
        VkBuffer *out, VmaAllocation *owner, VmaAllocationInfo *info, RcAdmissionStatus *admission) {
    assert(!held_lock && allocator == context.vma && request->required && request->producer == RC_BUDGET_BUFFER);
    assert(!admission && request->dependency && buffer->size == CHUNK_BYTES);
    assert(buffer->usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    assert(allocation->flags & VMA_ALLOCATION_CREATE_MAPPED_BIT);
    assert(allocation->requiredFlags == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    attempts++;
    if (refuse) {
        *request->dependency = (RcVmaDependency){.selected = 1, .memory_type = 0};
        request->dependency->peak.amount[RC_BUDGET_RAM] = buffer->size;
        return VK_NOT_READY;
    }
    assert(creates < sizeof natives / sizeof natives[0]);
    Native *native = &natives[creates++];
    *native = (Native){.buffer = (VkBuffer)(uintptr_t)creates, .map = malloc((size_t)buffer->size), .size = buffer->size};
    assert(native->map);
    *out = native->buffer;
    *owner = (VmaAllocation)native;
    *info = (VmaAllocationInfo){.size = native->size, .pMappedData = native->map};
    if (stop_during_create) stopping = 1;
    return VK_SUCCESS;
}
RcAdmissionStatus resource_core_vma_allocation_cost(VmaAllocation allocation, RcAdmissionVector *out) {
    const Native *native = (const Native *)allocation;
    assert(native->map);
    *out = (RcAdmissionVector){0};
    out->amount[RC_BUDGET_RAM] = native->size;
    return RC_ADMISSION_OK;
}
void resource_core_vma_destroy_buffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation) {
    Native *native = (Native *)allocation;
    assert(!held_lock && allocator == context.vma && native->map && native->buffer == buffer);
    free(native->map);
    native->map = NULL;
    destroys++;
}
VKAPI_ATTR void VKAPI_CALL vkGetDeviceBufferMemoryRequirements(VkDevice device,
        const VkDeviceBufferMemoryRequirements *request, VkMemoryRequirements2 *out) {
    assert(!held_lock && device == context.device);
    out->memoryRequirements = (VkMemoryRequirements){.size = request->pCreateInfo->size,
        .alignment = 256, .memoryTypeBits = 1};
}
VKAPI_ATTR void VKAPI_CALL vmaGetAllocationInfo2(VmaAllocator allocator, VmaAllocation allocation,
                                                VmaAllocationInfo2 *info) {
    (void)allocator; (void)allocation; (void)info; abort();
}

// Unrelated completion payloads stay empty.
void vk_gpa_write_abandon(VkGpaWriteScope *scope, int may) { (void)scope; (void)may; abort(); }
void vk_gpa_write_complete(VkGpaWriteScope *scope, uint64_t fence) { (void)scope; (void)fence; abort(); }
void vk_gpa_write_cancel(VkGpaWriteScope *scope) { (void)scope; abort(); }
void resource_host_write_dispose(ResourceHostWrite *host, uint64_t fence, int may, int cpu_visible) {
    (void)host; (void)fence; (void)may; (void)cpu_visible; abort();
}
RcRegistryResult rc_registry_write_abandon(RcRegistry *r, RcRegistryWrite w, int may) {
    (void)r; (void)w; (void)may; abort();
}
RcRegistryResult rc_registry_write_cancel(RcRegistry *r, RcRegistryWrite w) { (void)r; (void)w; abort(); }
RcRegistryResult rc_registry_write_submit(RcRegistry *r, RcRegistryWrite w, uint64_t fence) {
    (void)r; (void)w; (void)fence; abort();
}
RcRegistryResult rc_registry_write_complete(RcRegistry *r, RcRegistryWrite w, int visible) {
    (void)r; (void)w; (void)visible; abort();
}
void vk_mem_recorded_provider_release(void *provider) { (void)provider; abort(); }
void vk_mem_buffer_write_completed(void *provider, VkBuffer buffer, uint64_t address, uint64_t size, int mirrored) {
    (void)provider; (void)buffer; (void)address; (void)size; (void)mirrored; abort();
}
VKAPI_ATTR VkResult VKAPI_CALL vkResetDescriptorPool(VkDevice device, VkDescriptorPool pool,
                                                   VkDescriptorPoolResetFlags flags) {
    (void)device; (void)pool; (void)flags; abort();
}
VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorPool(VkDevice device, VkDescriptorPool pool,
                                                  const VkAllocationCallbacks *allocator) {
    (void)device; (void)pool; (void)allocator; abort();
}
void vk_texture_publication_release(TextureImagePublication *publication) { assert(!publication); }

typedef struct {
    uint64_t fence;
    void (*done)(void *);
    void (*cancel)(void *, int);
    void *owner;
} Completion;
static Completion final[2];
void vk_completion_add_owned(uint64_t fence, void (*complete)(void *), void (*cancel)(void *, int), void *owner) {
    assert(!held_lock && fence >= 1 && fence <= 2 && !final[fence - 1].owner);
    final[fence - 1] = (Completion){fence, complete, cancel, owner};
}


typedef struct { int unused; } EffectStorageCapture;
#include "mfb_payload_types_under_test.h"

struct VkExecGraphicsDraw {
    MfbDraw *fallback;
    DrawDescriptorGrant descriptors;
    VkPipeState state;
    VkGraphicsPipelineUse use;
    VkExecGraphicsScratch *scratch, *widened_index;
    RcRegistryPosition read_position;
    VkExecPreparationFailure failure;
    const void *payload;
    uint32_t opcode;
    VkMemSpanCapture *index_hold;
    int compatibility, exact_index, restart;
    VkMemSpan index;
};

static uint64_t waited_bytes;

static int mfb_transform_stopped(void) { return vk_sync_stopping() || vk_device_is_lost(); }

static VkExecGraphicsResult graphics_scratch_ready(VkExec *ex, VkExecGraphicsDraw *draw, int ordered) {
    (void)ex; (void)draw; (void)ordered;
    abort();  // Neither tested late cohort can require a second allocation.
}

void resource_core_budget_readiness(int required, RcBudgetReadiness *out) {
    assert(required);
    *out = (RcBudgetReadiness){.records_available=1,.usable_mask=1u<<RC_BUDGET_RAM};
}

VkSyncProgressResult vk_sync_progress_wait_prior(uint64_t fence) {
    (void)fence;
    abort();  // The current scene cannot be its own supplier.
}

VkSyncProgressResult vk_sync_progress_wait_capacity(const RcVmaDependency *dependency, int (*spare_ready)(void *),
                                                     void *context) {
    VkDrawScratchWaitContext *wait = context;
    assert(spare_ready && dependency->selected && dependency->peak.amount[RC_BUDGET_RAM] == CHUNK_BYTES);
    waited_bytes = wait->bytes;
    return VK_SYNC_PROGRESS_STOPPED;
}

static RcRegistryPosition byte_read;
RcRegistryPosition vk_exec_resource_position(void) { return byte_read; }

struct VkMemSpanCapture { int id; };
static const uint8_t index_source[] = {2, 255, 253, 6};
static const uint64_t index_address = UINT64_C(0x100000000);
static VkMemSpanCapture index_owner = {41}, import_owner = {42};
static int read_begins, read_finishes, read_cancels, binding_errors, index_binds;
static int can_freeze = 1, protection_valid = 1, force_snapshot, snapshot_valid = 1;
static VkExecHostInputResult reject_begin = VK_EXEC_HOST_INPUT_READY, reject_finish = VK_EXEC_HOST_INPUT_READY;
static RcRegistryResult select_result;
static int select_calls, select_device, select_mirror;
static const uint64_t device_address = RT_GPU_VA_BASE + 0x31;
static const uint8_t device_indices[] = {7, 255, 19};
static int native_refs, native_pins, native_copies, native_records, native_barriers, native_waits;
static int native_proof = 1, native_current = 1, lose_after_wait;
static VkExec *x_cur;
static int rw_chunk;
enum { PW_OP };
static VkSyncProgressResult native_progress = VK_SYNC_PROGRESS_COMPLETE;

static VkBuffer bound_index;
static VkDeviceSize bound_offset;
static VkIndexType bound_index_type;
static int legacy_reads, legacy_unlocks, resolve_calls, actual_draws, storage_writes, selected_restart = 1;
static int legacy_mapped = 1, mapping_current = 1, legacy_readable = 1, warnings;
static int imported_pool, other_pool, import_refs, import_borrows;
static VkExecHostInputResult import_admission = VK_EXEC_HOST_INPUT_READY, import_finish = VK_EXEC_HOST_INPUT_READY;
static void *legacy_pool = &imported_pool;
static VkExecFrame vk_exec_frame_fixture;
#define vk_exec_frame vk_exec_frame_fixture
static _Thread_local VkExecGraphicsDraw *graphics_current;

VkExecHostInputResult vk_exec_host_input_begin(const VkMemSpanCapture *owned, const VkMemSpan *span, uint64_t address,
                                  uint64_t size, RcRegistryPosition read, VkExecHostInputRead *out) {
    assert(!span->alloc && span->avail >= size && size == sizeof index_source &&
           !rc_registry_position_compare(read, byte_read) && !out->bytes && !out->opaque);
    if (owned == &import_owner) {
        assert(owned == &import_owner && import_refs==1 && span->pool==&imported_pool &&
               span->buffer==(VkBuffer)(uintptr_t)37);
        import_borrows++;
        if (import_admission != VK_EXEC_HOST_INPUT_READY) return import_admission;
        legacy_reads++;
        *out = (VkExecHostInputRead){.bytes=index_source, .opaque=&import_refs};
        return VK_EXEC_HOST_INPUT_READY;
    }
    assert(owned == &index_owner && owned->id == 41 &&
           (address == index_address || address == (uintptr_t)index_source));
    read_begins++;
    if (reject_begin != VK_EXEC_HOST_INPUT_READY) return reject_begin;
    *out = (VkExecHostInputRead){.bytes=index_source, .opaque=&read_begins};
    return VK_EXEC_HOST_INPUT_READY;
}
VkExecHostInputResult vk_exec_host_input_finish(VkExecHostInputRead *owned) {
    assert(owned->bytes == index_source);
    if (owned->opaque == &import_refs) {
        assert(import_refs == 1 && legacy_reads == legacy_unlocks + 1);
        *owned = (VkExecHostInputRead){0};
        legacy_unlocks++;
        return import_finish;
    }
    assert(owned->opaque == &read_begins);
    *owned = (VkExecHostInputRead){0};
    read_finishes++;
    return reject_finish;
}
void vk_exec_host_input_cancel(VkExecHostInputRead *owned) {
    assert(owned->bytes == index_source && owned->opaque == &read_begins);
    *owned = (VkExecHostInputRead){0};
    read_cancels++;
}
int mem_freeze(uint64_t address) {
    assert(address == index_address || address == ((uintptr_t)index_source & ~((UINT64_C(1) << 21) - 1)));
    return can_freeze;
}
int mem_frozen_stable(uint64_t address) { assert(address == index_address); return 1; }
uint64_t mem_protection_serial(void) { return 17; }
int mem_protection_range_unchanged(uint64_t addr, uint64_t n, uint64_t stamp) {
    assert(addr == index_address && n == sizeof index_source && stamp == 17);
    return protection_valid;
}
const uint8_t *vk_usnap_map(void) { return force_snapshot ? index_source : NULL; }
int vk_usnap_find_lease_at(uint64_t addr, uint64_t n, uint64_t align, RcRegistryPosition read,
                           VkDescriptorBufferInfo *out, VkUsnapLease *lease) {
    assert(force_snapshot && addr == index_address && n == sizeof index_source && align == 1 &&
           !rc_registry_position_compare(read, byte_read));
    *out = (VkDescriptorBufferInfo){ .offset = 0, .range = n };
    *lease = (VkUsnapLease){.capture = 18};
    return 1;
}
int vk_usnap_lease_valid_at(const VkUsnapLease *lease, RcRegistryPosition read) {
    assert(lease->capture == 18 && !rc_registry_position_compare(read, byte_read));
    return snapshot_valid;
}
RcRegistryResult vk_mem_span_native_input(uint64_t address, uint64_t size, int clamp,
                                          VkMemSpan *span, VkMemSpanCapture **owned) {
    assert(!clamp && !*owned);
    if (select_device) assert(address == device_address && size == sizeof device_indices);
    else assert((address == index_address || address == (uintptr_t)index_source) &&
                size == (context.has_index_type_uint8 ? 3 : sizeof index_source));
    select_calls++;
    if (select_result != RC_REGISTRY_OK) return select_result;
    *span = (VkMemSpan){.buffer = (VkBuffer)(uintptr_t)(select_mirror ? 94 : 37), .avail = size,
        .cls = select_device ? VK_MCLS_DEVICE : VK_MCLS_HOST_CACHED,
        .pool = select_mirror ? &imported_pool : NULL, .offset = select_device ? 0x31 : 0,
        .alloc = select_device ? (VmaAllocation)(uintptr_t)38 : NULL};
    native_refs++;
    *owned = &index_owner;
    return RC_REGISTRY_OK;
}
int vk_mem_span_capture_current(const VkMemSpanCapture *hold, const VkMemSpan *span) {
    return mapping_current && hold == &index_owner && span;
}
int vk_mem_span_held_argument_current(const VkMemSpanCapture *hold, uint64_t address, uint64_t count,
                                      const VkMemSpan *span) {
    assert(count == (context.has_index_type_uint8 ? 3 : select_device ? sizeof device_indices : sizeof index_source));
    assert(address == (select_device ? device_address : index_address) || address == (uintptr_t)index_source);
    return vk_mem_span_capture_current(hold, span);
}
RcRegistryResult vk_mem_span_prepare_held_argument(const VkMemSpanCapture *hold, const VkMemSpan *span, int ordered) {
    assert(context.has_index_type_uint8 && hold == &index_owner && span->buffer == (VkBuffer)(uintptr_t)37 && ordered == 1);
    return vk_mem_span_capture_current(hold, span) ? RC_REGISTRY_OK : RC_REGISTRY_STALE;
}
void vk_exec_required_graphics_unavailable(VkExec *ex, uint32_t op, const char *reason) {
    assert(ex && op == OP_DRAW_IDX && reason);
    binding_errors++;
}
void vk_mem_count(int cls, int use, uint64_t bytes) {
    assert((cls == VK_MCLS_HOST_CACHED && use == VK_MUSE_CPU_READ && bytes == sizeof index_source) ||
           (cls == VK_MCLS_HOST_CACHED && use == VK_MUSE_IB && bytes == 3 && context.has_index_type_uint8) ||
           (cls == VK_MCLS_SCRATCH && use == VK_MUSE_IB && (bytes == 2 * sizeof index_source || bytes == 2 * sizeof device_indices)));
}
static int restart_for(const VkExec *ex, int prim, int itype) {
    assert(ex && prim == NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP && itype == NVN_INDEX_TYPE_UNSIGNED_BYTE);
    return selected_restart;
}
static int prepare_draw(VkExec *ex, int prim, int32_t count, int restart) {
    assert(ex && prim == NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP && (count == 4 || count == 3) && restart == selected_restart);
    return 1;
}
static int graphics_argument_use(VkExec *ex, uint32_t op, uint64_t address, uint64_t size,
                                 VkMemSpan *span, VkMemSpanCapture **hold) {
    assert(context.has_index_type_uint8 && ex && op == OP_DRAW_IDX);
    if (!vk_mem_span_held_argument_current(*hold, address, size, span)) return 0;
    vk_mem_span_capture_release(hold);
    return 1;
}
static void storage_images_written(VkExec *ex) { assert(ex); storage_writes++; }
static void shield_capture(VkExec *ex, const p_draw_idx *draw) { (void)ex; (void)draw; abort(); }
static void mvobj_draw(VkExec *ex, const p_draw_idx *d, VkBuffer b, VkDeviceSize at) {
    (void)ex; (void)d; (void)b; (void)at; abort();
}
static struct {
    int index_captured;
    struct {int narrowed;} *capture[8];
    VkDescriptorBufferInfo indices;
} *effect_current;
#define EffectDeclaration __typeof__(*effect_current)
#define W_INDEX 0
#undef VK_MEM_USE
#define VK_MEM_USE(span) do { (void)(span); abort(); } while (0)
#undef VK_MEM_BOUND
#define VK_MEM_BOUND(span) do { assert(context.has_index_type_uint8 && (span).buffer); } while (0)
static void warn_once(int type, const char *fmt, ...) { assert(type == W_INDEX && fmt); warnings++; }
const int rt_game_version = 0;
int vk_mem_lookup(uint64_t address, uint64_t size, VkMemSpan *out) {
    assert(address == (uintptr_t)index_source && size == sizeof index_source);
    resolve_calls++;
    if (!legacy_mapped) return 0;
    *out = (VkMemSpan){.buffer = (VkBuffer)(uintptr_t)37, .avail = sizeof index_source, .cls = VK_MCLS_HOST_CACHED,
                       .pool = legacy_pool};
    return 1;
}
int vk_mem_resolve(uint64_t address, uint64_t size, VkMemSpan *out) {
    (void)address; (void)size; (void)out; abort(); // Native and unsupported branches in this fixture never resolve a GPU input.
}
int vk_mem_cpu_visible_span(const VkMemSpan *span, uint64_t address, uint64_t size) {
    assert(address == (uintptr_t)index_source && size == sizeof index_source && !span->alloc);
    return legacy_readable;
}
VkMemSpanCapture *vk_mem_span_capture(uint64_t address, uint64_t size, const VkMemSpan *span) {
    assert(address == (uintptr_t)index_source && size == sizeof index_source &&
           !import_refs && span->pool == &imported_pool && span->buffer == (VkBuffer)(uintptr_t)37);
    import_refs++;
    return &import_owner;
}
void vk_mem_span_capture_release(VkMemSpanCapture **owned) {
    if (!*owned) return;
    if (*owned == &index_owner) {
        assert(native_refs > 0);
        native_refs--;
    } else {
        assert(*owned == &import_owner && import_refs == 1 && legacy_reads == legacy_unlocks);
        import_refs--;
    }
    *owned = NULL;
}

uint32_t vk_mem_span_input_intervals(const VkMemSpanCapture *provider, uint64_t address, uint64_t size,
                                     RcInterval **out) {
    assert(provider == &index_owner && address == device_address && size == sizeof device_indices);
    *out = calloc(1, sizeof **out);
    assert(*out);
    (*out)->range.offset = 0x31;
    (*out)->range.size = size;
    return 1;
}
static void group_drain(VkExec *ex, int why) { assert(ex == x_cur && why == PW_OP); }
RcRegistry *rc_registry_runtime(void) { return (RcRegistry *)(uintptr_t)1; }
int rc_registry_provisional_read_current(RcRegistry *registry, RcRegistryPosition read) {
    return registry == rc_registry_runtime() && !rc_registry_position_compare(read, byte_read);
}
static int device_pointer_current(const VkMemSpan *span, const VkMemSpanCapture *provider,
                                  const RcInterval *ranges, uint32_t count, RcRegistryPosition read) {
    assert(span->buffer == (VkBuffer)(uintptr_t)37 && span->alloc && provider == &index_owner &&
           count == 1 && ranges[0].range.offset == 0x31 && ranges[0].range.size == sizeof device_indices);
    return native_proof && native_current && mapping_current && !rc_registry_position_compare(read, byte_read);
}
int vk_mem_span_record_input(VkMemSpanCapture *provider, const VkMemSpan *span) {
    assert(provider == &index_owner && native_refs == 1 && span->alloc);
    native_records++;
    return 1;
}
VkMemSpanCapture *vk_mem_span_capture_retain(VkMemSpanCapture *provider) {
    assert(provider == &index_owner && native_refs > 0);
    native_refs++;
    return provider;
}
void vk_exec_record_native_input_provider(VkExec *ex, VkMemSpanCapture **provider) {
    assert(ex == x_cur && !ex->in_pass && *provider == &index_owner && native_refs == 2);
    CbRes *res = ex->dc->res;
    if (!res->ninput_providers) {
        res->input_providers = calloc(1, sizeof *res->input_providers);
        assert(res->input_providers);
        res->input_providers[0].pool = (void *)&index_provider;
        res->ninput_providers = res->cinput_providers = 1;
        native_pins++;
    }
    vk_mem_span_capture_release(provider);
}
void vk_exec_pass_end(VkExec *ex) { ex->in_pass = 0; }
void vk_exec_barrier(VkExec *ex) { assert(!ex->in_pass && ex == x_cur); }
VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer cb, VkBuffer source, VkBuffer destination,
                                         uint32_t count, const VkBufferCopy *copy) {
    UpdateSlab *slab = x_cur->dc->res->graphics_slabs;
    assert(cb == x_cur->cb && source == (VkBuffer)(uintptr_t)37 && destination == slab->chunk.buf && count == 1 &&
           copy->srcOffset == 0x31 && copy->dstOffset % 2 == 0 && copy->size == sizeof device_indices &&
           copy->dstOffset + 2 * copy->size == slab->chunk.used && native_pins == 1 && !pin_releases);
    memcpy(slab->chunk.map + copy->dstOffset, device_indices, copy->size);
    native_copies++;
}
VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo *dependency) {
    assert(cb == x_cur->cb && dependency->bufferMemoryBarrierCount == 1);
    const VkBufferMemoryBarrier2 *barrier = dependency->pBufferMemoryBarriers;
    assert(barrier->srcAccessMask == VK_ACCESS_2_TRANSFER_WRITE_BIT && barrier->dstAccessMask == VK_ACCESS_2_HOST_READ_BIT &&
           barrier->size == sizeof device_indices && native_copies == native_barriers + 1);
    native_barriers++;
}
VkSyncProgressResult vk_exec_graphics_pool_progress(VkExec *ex, uint64_t prior) {
    assert(ex == x_cur && !prior && native_records == native_copies && native_copies == native_barriers &&
           native_copies == native_waits + 1 && native_refs == 1);
    native_waits++;
    if (lose_after_wait) native_current = 0;
    return native_progress;
}
static int graphics_storage_begin(VkExec *ex, VkExecGraphicsDraw *draw) { assert(ex && draw == graphics_current); return 1; }
static void graphics_storage_recorded(VkExec *ex, VkExecGraphicsDraw *draw) { assert(ex && draw == graphics_current); }

VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexed(VkCommandBuffer cb, uint32_t count, uint32_t instances,
                                             uint32_t first, int32_t base, uint32_t instance) {
    assert(cb && (count == 4 || count == 3) && instances == 1 && first == 0 && base == 0 && instance == 7);
    actual_draws++;
}
VKAPI_ATTR void VKAPI_CALL vkCmdBindIndexBuffer(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset,
                                                VkIndexType type) {
    assert(cb && buffer);
    assert(context.has_index_type_uint8 ? type == VK_INDEX_TYPE_UINT8_KHR :
           type == VK_INDEX_TYPE_UINT16 && offset % sizeof(uint16_t) == 0);
    bound_index_type = type;
    index_binds++;
    bound_index = buffer;
    bound_offset = offset;
}

struct TwinUse;
static void twin_use_release(struct TwinUse *use) { assert(!use); }

static VkExecBufferScratchReceipt *buffer_scratch_recording(const VkExec *ex) { assert(ex); return NULL; }
static uint8_t *buffer_scratch_view(const VkExecBufferScratchReceipt *receipt, unsigned slot, VkDescriptorBufferInfo *view) {
    assert(!receipt && slot == GRAPHICS_PAYLOAD_SUPPORT && view);
    return NULL;
}
static void buffer_scratch_expose(VkExec *ex, VkExecBufferScratchReceipt *receipt) { (void)ex; (void)receipt; abort(); }
static void draw_descriptor_pools_complete(CbRes *owner) { assert(owner); }
void vk_exec_draw_update_drop(VkExecUpdateDestination *destination) { assert(destination && !destination->hold); }
void vk_replay_metadata_free(void *allocation, RcBudgetTicket *ticket) { free(allocation); assert(!ticket->token.ledger); }
static int graphics_ubo_merge_plan(VkExec *ex, const VkPipeInfo *info, VkExecGraphicsScratch *plan,
                                   RcVmaDependency *dependency) {
    assert(ex && info && plan && dependency && !plan->ubo_merge_count);
    return 1;
}
static void graphics_ubo_merges_release(VkExecGraphicsScratch *grant) { assert(!grant->ubo_merge_count); }
static int graphics_ubo_merge_current(const VkExec *ex, const GraphicsUboMerge *merge) {
    (void)ex; (void)merge; abort();
}
VkUsnapResolveResult vk_usnap_resolve_at(uint64_t addr, uint64_t size, uint64_t alignment, RcRegistryPosition read,
                                         VkDescriptorBufferInfo *out, VkUsnapLease *lease) {
    (void)addr; (void)size; (void)alignment; (void)read; (void)out; (void)lease; abort();
}
void vk_exec_graphics_failure_set(VkExecGraphicsDraw *draw, const VkExecPreparationFailure *failure) {
    if (!draw->failure.owner) draw->failure = *failure;
}
static int host_reserved;
RcAdmissionStatus resource_core_budget_reserve(RcBudgetProducer producer, int required, const RcAdmissionVector *cost,
                                               RcBudgetTicket *out) {
    assert(producer == RC_BUDGET_BUFFER && required && cost->amount[RC_BUDGET_RAM] == sizeof(VkExecGraphicsScratch) &&
           cost->amount[RC_BUDGET_COMMIT] == sizeof(VkExecGraphicsScratch) && !out->token.ledger);
    out->token.ledger = (const RcAdmission *)(uintptr_t)1;
    host_reserved++;
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_materialize(RcBudgetTicket *ticket, const RcAdmissionVector *cost) {
    assert(ticket->token.ledger && cost->amount[RC_BUDGET_RAM] == sizeof(VkExecGraphicsScratch));
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_retire(RcBudgetTicket *ticket) {
    assert(ticket->token.ledger && host_reserved > 0);
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_destroyed(RcBudgetTicket *ticket) {
    assert(ticket->token.ledger && host_reserved > 0);
    host_reserved--;
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_cancel(RcBudgetTicket *ticket) {
    return resource_core_budget_destroyed(ticket);
}

RcAdmissionStatus resource_core_budget_ticket_cost(const RcBudgetTicket *ticket,RcAdmissionVector *out) {
    (void)ticket;(void)out;abort();
}
#include "graphics_payload_under_test.h"

static void check_shape(void) {
    static const GfxShaderRes zero = {.slot = 0}, array = {.slot = -1}, nonzero = {.slot = 1};
    const VkPipeBinding bindings[] = {
        {.stage = GFX_STAGE_TS, .kind = VK_PIPE_SUPPORT, .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
        {.binding = 1, .stage = GFX_STAGE_TS, .kind = GFX_RES_UBO, .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .res = &zero},
        {.binding = 2, .stage = GFX_STAGE_MS, .kind = GFX_RES_UBO, .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .count = 3, .res = &array},
        {.stage = GFX_STAGE_FS, .kind = VK_PIPE_SUPPORT, .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .set = 1},
        {.binding = 1, .stage = GFX_STAGE_FS, .kind = GFX_RES_UBO, .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .count = 3, .res = &nonzero, .set = 1},
        {.binding = 2, .stage = GFX_STAGE_FS, .kind = GFX_RES_UBO, .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .res = &zero, .set = 1},
    };
    VkPipeInfo info = {.bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS, .n_sets = 2,
        .n_bindings = sizeof bindings / sizeof bindings[0], .bindings = bindings};
    VkBindRecipe *recipe = vk_bind_recipe_create(&info, 0);
    assert(recipe && recipe->info == &info);
    uint32_t slots = 0;
    for (uint32_t i = 0; i < recipe->n_ops; i++) {
        const VkBindRecipeOp *op = &recipe->op[i];
        if (op->kind == VK_BR_SUPPORT) slots |= 1u << GRAPHICS_PAYLOAD_SUPPORT;
        if (op->kind != VK_BR_UBO) continue;
        for (uint32_t k = 0; k < op->count; k++)
            if (recipe->input[op->first + k].slot == 0) slots |= 1u << op->stage;
    }
    free(recipe);

    GraphicsPayloadDemand demand;
    unsigned support = GRAPHICS_PAYLOAD_SUPPORT;
    assert(graphics_payload_demand(&info, 16, 1, &demand));
    assert(slots == ((1u << NVN_STAGE_FS) | (1u << NVN_STAGE_TASK) | (1u << NVN_STAGE_MESH) | (1u << support)));
    assert(demand.slots == slots && demand.bytes == 7680 && demand.alignment == 256);
    assert(demand.offsets[NVN_STAGE_FS] == 0 && demand.offsets[NVN_STAGE_TASK] == 2048 &&
           demand.offsets[NVN_STAGE_MESH] == 4096 && demand.offsets[support] == 6144);
    assert(graphics_payload_demand(&info, 4096, 0, &demand));
    assert(demand.bytes == 4096 && demand.slots == (1u << support));
    Chunk tail = {.size = 8191, .used = 7};
    VkDeviceSize offset = 13;
    assert(!graphics_payload_fit(&demand, &tail, &offset) && offset == 13 && tail.used == 7);
    tail.size++;
    assert(graphics_payload_fit(&demand, &tail, &offset) && offset == 4096 && tail.used == 7);
    assert(!graphics_payload_demand(&info, 384, 1, &demand));
    assert(!graphics_payload_demand(&info, UINT64_C(1) << 63, 1, &demand));
    assert(demand.bytes == 4096 && demand.slots == (1u << support));
}


static void check_mfb_payload(VkExec *ex, const VkPipeState *state, const VkPipeInfo *graphics) {
    const GfxShaderRes zero = {.slot = 0};
    const VkPipeBinding phase_binding = {.stage = GFX_STAGE_CS, .kind = GFX_RES_UBO,
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .res = &zero};
    const VkPipeInfo phase = {.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE, .n_sets = 1,
        .n_bindings = 1, .bindings = &phase_binding};
    const p_draw_mesh draw_payload = {.count = 1};
    MfbPhaseUse use = {.shader = (const GfxShader *)(uintptr_t)2, .info = &phase};
    MfbDraw mfb = {.task = use, .mesh = use};
    VkExecGraphicsDraw draw = {.state = *state, .use.info = graphics, .opcode = OP_DRAW_MESH,
        .payload = &draw_payload, .fallback = &mfb};
    CbRes *owner = ex->dc->res;
    UpdateSlab *slab = owner->graphics_slabs;
    VkDeviceSize prior = slab->chunk.used;
    ex->resource_op = ++ex->resource_next;
    ex->ssbo[NVN_STAGE_TASK][0] = (p_bind_buf){.addr = 0x11110000, .size = 48};
    ex->ssbo[NVN_STAGE_MESH][0] = (p_bind_buf){.addr = 0x22220000, .size = 96};

    slab->chunk.used = CHUNK_BYTES;
    refuse = 1;
    unsigned tries = attempts;
    assert(mfb_scratch_cohort_ready(ex, &draw) == VK_EXEC_GRAPHICS_STOPPED);
    assert(attempts == tries + 1 && waited_bytes == 16384);
    assert(!mfb.scratch_owner && !mfb.task.scratch && !mfb.mesh.scratch && !draw.scratch &&
           !mfb.scratch_prepared && slab->chunk.used == CHUNK_BYTES);
    refuse = 0;
    slab->chunk.used = prior;

    assert(mfb_scratch_cohort_ready(ex, &draw) == VK_EXEC_GRAPHICS_READY);
    assert(mfb.scratch_owner->start == prior && mfb.scratch_owner->demand.bytes == 16384);
    assert(slab->chunk.used == prior + 16384 && mfb.task.scratch->stage_override == NVN_STAGE_TASK &&
           mfb.mesh.scratch->stage_override == NVN_STAGE_MESH && draw.scratch->stage_override == 0);
    assert(mfb.task.scratch->start == prior && mfb.mesh.scratch->start == prior + 4096 &&
           draw.scratch->start == prior + 8192);
    assert(mfb.task.scratch->ordinal == ex->resource_op && mfb.mesh.scratch->ordinal == ex->resource_op &&
           draw.scratch->ordinal == ex->resource_op);
    VkDescriptorBufferInfo c0;
    uint64_t addr;
    const uint8_t *bank = graphics_payload_view(mfb.task.scratch, NVN_STAGE_TASK, &c0);
    memcpy(&addr, bank + 0x670, sizeof addr);
    assert(addr == 0x11110000 && c0.offset == prior);
    assert(!graphics_payload_view(mfb.task.scratch, NVN_STAGE_MESH, &c0));
    bank = graphics_payload_view(mfb.mesh.scratch, NVN_STAGE_MESH, &c0);
    memcpy(&addr, bank + 0x770, sizeof addr);
    assert(addr == 0x22220000 && c0.offset == prior + 4096);
    vk_exec_draw_graphics_scratch_finish(&draw.scratch);
    vk_exec_draw_graphics_scratch_finish(&mfb.mesh.scratch);
    vk_exec_draw_graphics_scratch_finish(&mfb.task.scratch);
    vk_exec_draw_graphics_scratch_finish(&mfb.scratch_owner);
    assert(slab->chunk.used == prior);

    mfb = (MfbDraw){.task = use, .mesh = use, .final_mode = MFB_FINAL_PASS_COMPAT};
    assert(mfb_scratch_cohort_ready(ex, &draw) == VK_EXEC_GRAPHICS_READY);
    assert(mfb.late_capacity == 12288 && mfb.late_offset == 8192 && !draw.scratch &&
           mfb.scratch_owner->demand.bytes == 20480);
    assert(mfb_late_scratch_ready(ex, &draw) == VK_EXEC_GRAPHICS_READY);
    assert(draw.scratch->start == prior + mfb.late_offset && draw.scratch->demand.bytes == 8192 &&
           draw.scratch->ordinal == ex->resource_op && draw.scratch->cohort == mfb.scratch_owner);
    mfb.mesh.scratch->exposed = 1;
    vk_exec_draw_graphics_scratch_finish(&draw.scratch);
    vk_exec_draw_graphics_scratch_finish(&mfb.mesh.scratch);
    vk_exec_draw_graphics_scratch_finish(&mfb.task.scratch);
    vk_exec_draw_graphics_scratch_finish(&mfb.scratch_owner);
    vk_exec_draw_update_prefix_completed(ex, ex->fence);
    assert(slab->chunk.used == prior + 20480 && attempts == tries + 1 && owner->graphics_slabs == slab);
}

static void check_byte_preflight(VkExec *ex, VkPipeState state, const VkPipeInfo *info, int warm) {
    p_draw_idx index = {.prim = NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP, .count = 4,
        .itype = NVN_INDEX_TYPE_UNSIGNED_BYTE, .ibuf = index_address, .base_instance = 7};
    VkExecGraphicsScratch plan = {0};
    uint32_t base = 0;
    RcVmaDependency dependency = {0};
    assert(graphics_payload_plan(ex, &state, info, OP_DRAW_IDX, &index, 1, &plan, &base, &dependency, NULL));
    assert(base == 7 && plan.demand.bytes == 8192 && !plan.widen_draw);
    uint64_t saved_op = ex->resource_op;
    ex->resource_op = saved_op ? saved_op : ex->resource_next + 1;
    byte_read = (RcRegistryPosition){ex->fence, 2 * ex->resource_op};
    VkExecGraphicsDraw draw = {.read_position = byte_read, .restart = 1};
    select_result = RC_REGISTRY_OK;
    if (!warm) {
        assert(graphics_widen_prepare(ex, &draw, &index, 0) == VK_EXEC_GRAPHICS_NEEDS_ORDERED);
        assert(!draw.widened_index && draw.index_hold && !read_begins && !index_binds && !ex->dc->res);
        vk_mem_span_capture_release(&draw.index_hold);
        ex->resource_op = saved_op;
        return;
    }

    UpdateSlab *slab = ex->dc->res->graphics_slabs;
    uint64_t prior = slab->chunk.used;
    select_result = RC_REGISTRY_CAPACITY;
    assert(graphics_widen_prepare(ex, &draw, &index, 0) == VK_EXEC_GRAPHICS_UNAVAILABLE && !draw.index_hold);
    select_result = RC_REGISTRY_STALE;
    assert(graphics_widen_prepare(ex, &draw, &index, 0) == VK_EXEC_GRAPHICS_INVALIDATED && !draw.index_hold);
    select_result = RC_REGISTRY_OK;
    reject_begin = VK_EXEC_HOST_INPUT_BLOCKED;
    assert(graphics_widen_prepare(ex, &draw, &index, 0) == VK_EXEC_GRAPHICS_UNAVAILABLE && draw.index_hold);
    assert(slab->chunk.used == prior && !draw.widened_index && !index_binds);
    reject_begin = VK_EXEC_HOST_INPUT_UNTRACKED;
    assert(graphics_widen_prepare(ex, &draw, &index, 1) == VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!resolve_calls && !read_cancels && slab->chunk.used == prior);
    reject_begin = VK_EXEC_HOST_INPUT_READY;
    reject_finish = VK_EXEC_HOST_INPUT_CHANGED;
    assert(graphics_widen_prepare(ex, &draw, &index, 1) == VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!resolve_calls && read_finishes == 1 && !draw.widened_index && slab->chunk.used == prior);

    // A ready exact host receipt suffices even when the old protection proof glitches.
    can_freeze = protection_valid = 0;
    reject_finish = VK_EXEC_HOST_INPUT_READY;
    assert(graphics_widen_prepare(ex, &draw, &index, 0) == VK_EXEC_GRAPHICS_READY);
    const uint16_t *host_widened = (const uint16_t *)(slab->chunk.map + prior);
    assert(host_widened[0] == 2 && host_widened[1] == UINT16_MAX &&
           host_widened[2] == 253 && host_widened[3] == 6 && !resolve_calls);
    vk_exec_draw_graphics_scratch_finish(&draw.widened_index);
    assert(slab->chunk.used == prior);
    can_freeze = protection_valid = 1;
    reject_finish = VK_EXEC_HOST_INPUT_UNSUPPORTED;
    assert(graphics_widen_prepare(ex, &draw, &index, 1) == VK_EXEC_GRAPHICS_INVALIDATED && !resolve_calls && read_finishes == 3);
    reject_finish = VK_EXEC_HOST_INPUT_READY;

    force_snapshot = 1;
    snapshot_valid = 0;
    assert(graphics_widen_prepare(ex, &draw, &index, 0) == VK_EXEC_GRAPHICS_INVALIDATED && slab->chunk.used == prior);
    snapshot_valid = 1;
    assert(graphics_widen_prepare(ex, &draw, &index, 0) == VK_EXEC_GRAPHICS_READY && !draw.index_hold);
    VkExecGraphicsScratch *grant = draw.widened_index;
    assert(grant->start == prior && grant->widen_bytes == 8 && !grant->exposed);
    const uint16_t *widened = (const uint16_t *)(slab->chunk.map + prior);
    assert(widened[0] == 2 && widened[1] == UINT16_MAX && widened[2] == 253 && widened[3] == 6);
    mapping_current = 0;
    int selected = select_calls;
    assert(graphics_widen_prepare(ex, &draw, &index, 1) == VK_EXEC_GRAPHICS_READY && draw.widened_index == grant);
    assert(select_calls == selected && !resolve_calls);
    p_draw_idx different = index;
    assert(graphics_widen_prepare(ex, &draw, &different, 1) == VK_EXEC_GRAPHICS_INVALIDATED);
    mapping_current = 1;
    force_snapshot = 0;

    VkCommandBuffer saved_cb = ex->cb;
    ex->cb = (VkCommandBuffer)(uintptr_t)12;
    graphics_current = &draw;
    op_draw_idx(ex, &index);
    assert(actual_draws == 1 && storage_writes == 1 && index_binds == 1 && grant->exposed &&
           bound_index == slab->chunk.buf && bound_offset == prior);
    assert(!graphics_widen_bind(ex, &index, 1) && binding_errors == 1 && index_binds == 1);
    vk_exec_draw_graphics_scratch_finish(&draw.widened_index);
    uint64_t exposed = slab->chunk.used;
    assert(exposed == prior + 8);

    index.ibuf = (uintptr_t)index_source;
    select_mirror = 1;
    can_freeze = 0;
    reject_begin = VK_EXEC_HOST_INPUT_UNSUPPORTED;
    draw.compatibility = 1;
    legacy_pool = &other_pool;
    assert(graphics_widen_prepare(ex, &draw, &index, 1) == VK_EXEC_GRAPHICS_INVALIDATED && !draw.widened_index);
    legacy_pool = &imported_pool;
    import_admission = VK_EXEC_HOST_INPUT_BLOCKED;
    assert(graphics_widen_prepare(ex, &draw, &index, 1) == VK_EXEC_GRAPHICS_UNAVAILABLE && !draw.widened_index);
    import_admission = VK_EXEC_HOST_INPUT_READY;
    import_finish = VK_EXEC_HOST_INPUT_STALE;
    assert(graphics_widen_prepare(ex, &draw, &index, 1) == VK_EXEC_GRAPHICS_INVALIDATED && !draw.widened_index);
    import_finish = VK_EXEC_HOST_INPUT_READY;
    assert(graphics_widen_prepare(ex, &draw, &index, 1) == VK_EXEC_GRAPHICS_READY && !draw.index_hold && !import_refs);
    assert(import_borrows == 3 && legacy_reads == 2 && legacy_unlocks == 2);
    assert(draw.widened_index->start == exposed);
    op_draw_idx(ex, &index);
    assert(actual_draws == 2 && index_binds == 2 && bound_offset == exposed);
    vk_exec_draw_graphics_scratch_finish(&draw.widened_index);
    exposed = slab->chunk.used;

    select_mirror = 0;
    select_device = 1;
    index.ibuf = device_address;
    index.count = sizeof device_indices;
    x_cur = ex;
    ex->in_pass = 1;
    assert(graphics_widen_prepare(ex, &draw, &index, 0) == VK_EXEC_GRAPHICS_NEEDS_ORDERED);
    assert(!native_copies && draw.index_hold && ex->in_pass);
    native_proof = 0;
    assert(graphics_widen_prepare(ex, &draw, &index, 1) == VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!native_copies && slab->chunk.used == exposed && ex->in_pass);
    native_proof = 1;
    lose_after_wait = 1;
    assert(graphics_widen_prepare(ex, &draw, &index, 1) == VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(native_copies == 1 && slab->chunk.used == exposed && !ex->in_pass);
    native_current = 1;
    lose_after_wait = 0;
    draw.restart = selected_restart = 0;
    assert(graphics_widen_prepare(ex, &draw, &index, 1) == VK_EXEC_GRAPHICS_READY && !draw.index_hold);
    widened = (const uint16_t *)(slab->chunk.map + exposed);
    assert(widened[0] == 7 && widened[1] == 255 && widened[2] == 19);
    assert(draw.widened_index->start == exposed && !draw.widened_index->exposed && native_copies == 2 && native_pins == 1);
    vk_exec_draw_graphics_scratch_finish(&draw.widened_index);
    assert(slab->chunk.used == exposed);
    assert(graphics_widen_prepare(ex, &draw, &index, 1) == VK_EXEC_GRAPHICS_READY && !draw.index_hold);
    op_draw_idx(ex, &index);
    assert(actual_draws == 3 && index_binds == 3 && bound_offset == exposed);
    vk_exec_draw_graphics_scratch_finish(&draw.widened_index);
    assert(slab->chunk.used == exposed + 6);
    exposed = slab->chunk.used;
    native_progress = VK_SYNC_PROGRESS_STOPPED_AFTER_CUT;
    assert(graphics_widen_prepare(ex, &draw, &index, 1) == VK_EXEC_GRAPHICS_STOPPED);
    assert(!draw.widened_index && slab->chunk.used == exposed + 6 && native_pins == 1 && native_waits == 4 && !pin_releases);
    vk_mem_span_capture_release(&draw.index_hold);
    assert(!native_refs && !host_reserved && !warnings);
    graphics_current = NULL;
    x_cur = NULL;
    ex->cb = saved_cb;
    ex->resource_op = saved_op;
}

static void check_custody(void) {
    context.vma = (VmaAllocator)(uintptr_t)1;
    context.device = (VkDevice)(uintptr_t)1;
    context.props.limits.minUniformBufferOffsetAlignment = 4096;
    context.mem_props.memoryTypeCount = context.mem_props.memoryHeapCount = 1;
    context.mem_props.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    context.mem_props.memoryHeaps[0].size = UINT64_C(1) << 30;
    DrawCtx recorder = {0}, later_recorder = {0};
    VkExec ex = {.dc = &recorder, .fence = 1}, later = {.dc = &later_recorder, .fence = 2};
    dc = &recorder;
    const GfxShaderRes c0_resource = {.slot = 0};
    const VkPipeBinding bindings[] = {
        {.stage = GFX_STAGE_VS, .kind = VK_PIPE_SUPPORT, .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
        {.binding = 1, .stage = GFX_STAGE_VS, .kind = GFX_RES_UBO, .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .res = &c0_resource},
    };
    const VkPipeInfo info = {.bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS, .n_bindings = 2, .bindings = bindings};
    const VkPipeState state = {.vs = (const GfxShader *)(uintptr_t)1};
    const p_draw draw = {.count = 3, .base_instance = 19};
    ex.ssbo[NVN_STAGE_VS][0] = (p_bind_buf){.addr = 0x12345678, .size = 80};
    VkExecGraphicsScratch *grant = NULL;
    RcVmaDependency dependency;
    uint64_t bytes;

    check_byte_preflight(&ex, state, &info, 0);
    assert(vk_exec_draw_graphics_scratch_prepare(&ex, &state, &info, OP_DRAW, &draw, 0, &grant, &bytes, &dependency, NULL)
           == VK_NOT_READY);
    assert(bytes == 8192 && !grant && !dependency.selected && !attempts && !ex.resource_next && !recorder.res);
    refuse = 1;
    assert(vk_exec_draw_graphics_scratch_prepare(&ex, &state, &info, OP_DRAW, &draw, 1, &grant, &bytes, &dependency, NULL)
           == VK_NOT_READY);
    assert(bytes == 8192 && !grant && dependency.selected && dependency.peak.amount[RC_BUDGET_RAM] == CHUNK_BYTES &&
           attempts == 1 && !ex.resource_next && !recorder.res);
    const RcVmaDependency refused = dependency;
    refuse = 0;

    stop_during_create = 1;
    assert(vk_exec_draw_graphics_scratch_prepare(&ex, &state, &info, OP_DRAW, &draw, 1, &grant, &bytes, &dependency, NULL)
           == VK_ERROR_INITIALIZATION_FAILED);
    assert(stopping && !grant && !ex.resource_next && !recorder.res && attempts == 2 && nspare_chunk == 1);
    stop_during_create = stopping = 0;
    assert(vk_exec_draw_graphics_scratch_prepare(&ex, &state, &info, OP_DRAW, &draw, 0, &grant, &bytes, &dependency, NULL)
           == VK_SUCCESS);
    assert(grant && grant->owner == recorder.res && final[0].owner == recorder.res && !ex.resource_next && attempts == 2);
    VkDescriptorBufferInfo c0;
    const uint8_t *bank = graphics_payload_view(grant, NVN_STAGE_VS, &c0);
    uint64_t addr;
    uint32_t first;
    assert(bank && c0.range == VK_C0_BYTES && c0.offset == 0);
    memcpy(&addr, bank + 0x110, sizeof addr);
    memcpy(&first, bank + 0x644, sizeof first);
    assert(addr == 0x12345678 && first == 19);

    ex.resource_op = ++ex.resource_next;
    ex.in_pass = ex.has_vp = 1;
    ex.vp[0][2] = 100;
    ex.vp[0][3] = 120;
    ex.pass_scale = 2;
    ex.point_size = 9;
    graphics_payload_bound = grant;
    VkDescriptorBufferInfo support = support_descriptor(&ex);
    const uint8_t *mapped_support = grant->slab->chunk.map + support.offset;
    float width, point, hdr;
    memcpy(&width, mapped_support + 160, sizeof width);
    memcpy(&point, mapped_support + 1392, sizeof point);
    memcpy(&hdr, mapped_support + 1440, sizeof hdr);
    assert(support.buffer == c0.buffer && support.offset == 4096 && support.range == SUPPORT_BYTES);
    assert(width == 200 && point == 9 && hdr == 120);
    graphics_payload_bound = NULL;
    vk_exec_draw_graphics_scratch_finish(&grant);
    assert(!grant && recorder.res->graphics_slabs->chunk.used == 8192);

    assert(vk_exec_draw_graphics_scratch_prepare(&ex, &state, &info, OP_DRAW, &draw, 0, &grant, &bytes, &dependency, NULL)
           == VK_SUCCESS);
    assert(grant->start == 8192 && grant->slab->chunk.used == 16384 && attempts == 2);
    vk_exec_draw_graphics_scratch_finish(&grant);
    assert(!grant && recorder.res->graphics_slabs->chunk.used == 8192);

    UpdateSlab *inline_slab = NULL;
    assert(update_slab_create(256, 1, &inline_slab, &dependency) == VK_SUCCESS);
    VkBuffer inline_buffer = inline_slab->chunk.buf;
    update_publish(recorder.res, inline_slab);
    vk_exec_draw_update_prefix_completed(&ex, ex.fence);
    assert(!recorder.res->update_slabs && recorder.res->graphics_slabs->chunk.used == 8192);
    assert(nspare_chunk == 1 && spare_chunk[0].buf == inline_buffer && inline_buffer != c0.buffer);
    memcpy(&width, mapped_support + 160, sizeof width);
    assert(width == 200);

    const RcBudgetReadiness readiness = {.records_available = 1, .usable_mask = 1u << RC_BUDGET_RAM};
    assert(!vk_exec_draw_graphics_scratch_prior_useful(&later, bytes, &refused, &readiness));
    handed = ex.fence;
    assert(vk_exec_draw_graphics_scratch_prior_useful(&later, bytes, &refused, &readiness) == ex.fence);
    assert(!vk_exec_draw_graphics_scratch_prior_useful(&ex, bytes, &refused, &readiness));

    assert(vk_exec_draw_graphics_scratch_prepare(&later, &state, &info, OP_DRAW, &draw, 0, &grant, &bytes, &dependency, NULL)
           == VK_SUCCESS);
    assert(grant->slab->chunk.buf == inline_buffer && !nspare_chunk && attempts == 3);
    vk_exec_draw_graphics_scratch_finish(&grant);
    assert(!grant && !later_recorder.res->graphics_slabs && nspare_chunk == 1);

    assert(vk_exec_draw_graphics_scratch_prepare(&later, &state, &info, OP_DRAW, &draw, 0, &grant, &bytes, &dependency, NULL)
           == VK_SUCCESS);
    graphics_payload_bound = grant;
    assert(support_descriptor(&later).buffer == inline_buffer);
    graphics_payload_bound = NULL;
    vk_exec_draw_graphics_scratch_finish(&grant);
    assert(!grant && later_recorder.res->graphics_slabs && !nspare_chunk);

    dc = &recorder;
    check_mfb_payload(&ex, &state, &info);
    check_byte_preflight(&ex, state, &info, 1);

    done = ex.fence;
    final[0].done(final[0].owner);
    assert(pin_releases == 1);
    assert(nspare_chunk == 1 && spare_chunk[0].buf == c0.buffer);
    final[1].cancel(final[1].owner, 0);
    assert(!graphics_head && !graphics_tail && nspare_chunk == 2);
    assert(spare_chunk[1].buf == inline_buffer);
    RcAdmissionVector deficit = {0};
    deficit.amount[RC_BUDGET_RAM] = UINT64_MAX;
    assert(reclaimer(reclaim_context, &deficit) == RC_BUDGET_RECLAIM_DESTROYED);
    assert(creates == destroys && !nspare_chunk);
    vk_exec_draw_scratch_shutdown();
}

static void check_native_byte_input(void) {
    VkExec ex = {.cb = (VkCommandBuffer)(uintptr_t)12};
    DrawCtx recorder = {0};
    ex.dc = &recorder;
    const p_draw_idx indices = {.prim = NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP, .itype = NVN_INDEX_TYPE_UNSIGNED_BYTE,
        .count = 3, .base_instance = 7, .ibuf = index_address};
    VkExecGraphicsDraw draw = {.compatibility = 1};
    const int before_draws = actual_draws, before_waits = native_waits, reads = read_begins;
    const int before_binds = index_binds, before_errors = binding_errors;
    select_device = select_mirror = 0;
    select_result = RC_REGISTRY_OK;
    selected_restart = mapping_current = context.has_index_type_uint8 = 1;
    can_freeze = protection_valid = 0; // unrelated protection does not participate in native index binding
    assert(graphics_prepare_byte_indices(&ex, &draw, &indices, 1) == VK_EXEC_GRAPHICS_READY);
    assert(draw.index_hold && draw.index.avail == 3 && !draw.widened_index);
    graphics_current = &draw;
    op_draw_idx(&ex, &indices);
    assert(bound_index_type == VK_INDEX_TYPE_UINT8_KHR && bound_index == (VkBuffer)(uintptr_t)37 && bound_offset == 0);
    assert(actual_draws == before_draws + 1 && index_binds == before_binds + 1 && !draw.index_hold && !native_refs);
    assert(native_waits == before_waits && read_begins == reads);

    // Remap after preparation must not switch to a freshly looked-up source.
    assert(graphics_prepare_byte_indices(&ex, &draw, &indices, 1) == VK_EXEC_GRAPHICS_READY);
    mapping_current = 0;
    op_draw_idx(&ex, &indices);
    assert(actual_draws == before_draws + 1 && index_binds == before_binds + 1 && draw.index_hold);
    vk_mem_span_capture_release(&draw.index_hold);
    op_draw_idx(&ex, &indices);
    assert(binding_errors == before_errors + 1 && index_binds == before_binds + 1);
    graphics_current = NULL;
    context.has_index_type_uint8 = 0;
}

int main(void) {
    check_shape();
    check_custody();
    check_native_byte_input();
    puts("graphics draw payload: exact packing, refusal/stop, late support, final-only custody passed");
    puts("byte indices: admitted host and native sources, retained retry bytes, exact bind and cancellation passed");
}

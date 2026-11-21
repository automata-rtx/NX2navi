"""Host check of mapped draw scratch, ordered uniform updates, and scene completion."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from bind_recipe_check import function


def section(source, first, following):
    start = source.index(first)
    return source[start:source.index(following, start + len(first))]


def main():
    draw = (ROOT / 'runtime/vk_exec_draw.c').read_text()
    replay = (ROOT / 'runtime/vk_exec.c').read_text()
    sync = (ROOT / 'runtime/vk_sync.c').read_text()
    vma = (ROOT / 'runtime/resource_core_vma.c').read_text()
    usnap = (ROOT / 'runtime/vk_usnap.c').read_text()
    usnap = usnap[usnap.index('static RcAdmissionVector arena_host_cost(uint64_t size) {'):]
    metadata = '\n'.join(function(usnap, name) for name in (
        'arena_host_cost', 'peak_sum', 'vk_replay_metadata_peak_add', 'vk_replay_metadata_free',
        'arena_admission_failure', 'metadata_touch_pages', 'replay_metadata_allocate', 'vk_replay_metadata_allocate',
    ))
    chunk_size = next(line for line in draw.splitlines() if line.startswith('#define CHUNK_BYTES '))
    owners = section(draw, '// ---- resources of one command buffer', 'typedef struct {\n    VkPipelineLayout layout;')
    # These forward declarations belong to graphics payload planning, outside the extracted allocator.
    graphics_only = section(owners, 'static GraphicsUboMerge *graphics_ubo_merge_find(', 'struct VkDrawScratchCredit {')
    owners = owners.replace(graphics_only, '', 1)
    owners = owners.replace(function(owners, 'descriptor_pool_opaque_bytes'), '')
    descriptor_cache = section(owners, 'static DrawDescriptorPool *spare_pool[', 'static Chunk spare_chunk[')
    owners = owners.replace(descriptor_cache, '', 1)
    owners = owners.replace('static _Thread_local DrawDescriptorGrant *descriptor_binding_grant;\n', '', 1)
    owners = owners.replace('static int nspare_pool, nspare_chunk;', 'static int nspare_chunk;', 1)
    provider = section(draw, 'static unsigned scratch_detach_spares(', 'static void cb_finish(')
    provider = provider.replace('static void twin_use_release(struct TwinUse *use);',
                                'static void twin_use_release(struct TwinUse *use) { assert(!use); }')
    callbacks = 'static void draw_descriptor_pools_complete(CbRes *r) { assert(!r->pools); }\n'
    callbacks += '\n'.join(function(draw, name) for name in
                          ('cb_finish', 'cb_done', 'cb_cancel', 'vk_exec_draw_new_cb', 'cb_res'))
    callbacks += '\n' + section(draw, 'static uint64_t scratch_span_bytes(uint64_t requested);', 'static void buffer_preserve_release(')
    updates = section(draw, 'static uint64_t scratch_span_bytes(uint64_t bytes) {', '// ---- bind ops') + function(draw, 'update_declare')
    updates += '\n' + function(draw[draw.index('static void graphics_slab_adopt(CbRes *r,'):], 'graphics_slab_adopt')
    updates += '\n' + function(draw[draw.index('uint64_t vk_exec_draw_graphics_scratch_prior_useful(VkExec *ex,'):],
                              'vk_exec_draw_graphics_scratch_prior_useful')
    draw_index = section(draw, 'static void op_draw(VkExec *ex, const p_draw *d) {', '// Primitive restart')
    group_declarations = '\n'.join(next(line for line in replay.splitlines() if line.startswith(marker))
                                   for marker in ('#define GROUP_PEEK ', 'enum { GE_RUN,'))
    group_scan = '\n'.join(function(replay, name) for name in ('run_op', 'group_peek', 'group_peek_plan'))
    pending_type = next(line for line in sync.splitlines() if line.endswith('} Pending;'))
    pending = '\n'.join(function(sync, name) for name in
                        ('pending_push', 'vk_completion_add_owned', 'pending_disposable', 'pending_dispose',
                         'pending_dispose_queue', 'pending_dispose_remaining'))
    footprint = '\n'.join(function(vma, name) for name in ('allocation_cost', 'resource_core_vma_native_footprint'))

    fixture = r'''
#include <assert.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "platform/platform.h"
#include "resource_core_vma.h"
#include "resource_host_write.h"
#include "resource_command_effects.h"
#include "vk_exec.h"
#include "vk_gpa.h"
#include "vk_texture_lifetime.h"

#define RWT_COMPLETION(...)
#define RWT_CHAIN(...)
#define RWT_MAYBE_DRAW(...)
#define RESOURCE_IMAGE_SCOPE(...)

static VkCtx context;
static int context_ready = 1;
static unsigned context_reads;
VkCtx *vk_ctx(void) {
    context_reads++;
    return context_ready ? &context : NULL;
}
int plat_commit_counts_vram(void) { return 0; }
int vk_gpu_passes_capturing(void) { return 0; }

static PlatMutex *lock_stack[4];
static unsigned lock_depth;
void plat_mutex_lock(PlatMutex *lock) {
    assert(lock_depth < sizeof lock_stack / sizeof lock_stack[0]);
    for (unsigned i = 0; i < lock_depth; i++) assert(lock_stack[i] != lock);
    lock_stack[lock_depth++] = lock;
}
void plat_mutex_unlock(PlatMutex *lock) {
    assert(lock_depth && lock_stack[lock_depth - 1] == lock);
    lock_depth--;
}
void plat_cond_broadcast(PlatCond *condition) { (void)condition; }
void plat_cond_wait(PlatCond *condition, PlatMutex *lock) { (void)condition; (void)lock; abort(); }
void plat_once(PlatOnce *once, void (*fn)(void *), void *arg) {
    if (!once->state) {
        fn(arg);
        once->state = 2;
    }
}
void rt_log(const char *fmt, ...) { (void)fmt; }
static jmp_buf fatal_return;
static int expect_fatal;
_Noreturn void rt_fatal(const char *fmt, ...) {
    if (expect_fatal) longjmp(fatal_return, 1);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    abort();
}
uint64_t vk_mem_give_back(void) { abort(); }

static RcBudgetReclaimFn reclaim;
static void *reclaim_context;
static unsigned progress_calls;
RcAdmissionStatus resource_core_budget_register_reclaimer(RcBudgetProducer producer, uint32_t domains,
                                                          RcBudgetReclaimFn fn, void *arg) {
    assert(!lock_depth && !reclaim && producer == RC_BUDGET_BUFFER && fn);
    assert(domains & (1u << RC_BUDGET_RAM));
    assert(domains & (1u << RC_BUDGET_COMMIT));
    assert(domains & (1u << RC_BUDGET_GPU_FIRST));
    assert(!(domains & (1u << RC_BUDGET_STAGING)));
    reclaim = fn;
    reclaim_context = arg;
    return RC_ADMISSION_OK;
}
void resource_core_budget_progress(void) {
    assert(!lock_depth);
    progress_calls++;
}

enum { NATIVE_PADDING = 512 };
typedef struct {
    VkBuffer buffer;
    uint8_t *map;
    unsigned type;
    VkDeviceSize occupied;
    VkBool32 dedicated;
    RcAdmissionVector cost;
    RcAdmissionToken ticket;
    int live;
} Native;
static Native native[16];
static unsigned creates, destroys, native_type;
static int through_destroy_adapter;
enum { GRANT_READY, GRANT_PENDING, GRANT_HARD, GRANT_UNMAPPED, GRANT_PLACEMENT_OOM };
static int grant_mode, grant_dedicated;
static unsigned compat_optional, compat_refused, compat_waited;
static VkResult placement_failure = VK_ERROR_OUT_OF_DEVICE_MEMORY;
static unsigned update_attempts, update_materialized, update_retired, peak_queries;
static uint32_t requirement_type_override;
static RcBudgetReadiness configured_readiness;
static RcVmaDependency configured_dependency;

void resource_core_budget_readiness(int required, RcBudgetReadiness *out) {
    assert(required && !lock_depth);
    *out = configured_readiness;
}
VKAPI_ATTR void VKAPI_CALL vkGetDeviceBufferMemoryRequirements(VkDevice device,
        const VkDeviceBufferMemoryRequirements *request, VkMemoryRequirements2 *out) {
    assert(device == context.device && !lock_depth && request->pCreateInfo);
    out->memoryRequirements = (VkMemoryRequirements){.size = request->pCreateInfo->size + NATIVE_PADDING,
        .alignment = 256, .memoryTypeBits = requirement_type_override ? requirement_type_override : 1u << native_type};
}
static RcAdmission update_ledger;
static RcAdmissionRecord update_tickets[32];

RcAdmissionStatus resource_core_budget_reserve(RcBudgetProducer p, int required, const RcAdmissionVector *cost,
                                              RcBudgetTicket *ticket) {
    assert(required && p == RC_BUDGET_BUFFER && !lock_depth);
    assert(cost->amount[RC_BUDGET_RAM] == cost->amount[RC_BUDGET_COMMIT]);
    ticket->producer = p;
    return rc_admission_reserve(&update_ledger, p, cost, &ticket->token);
}
RcAdmissionStatus resource_core_budget_split(RcBudgetTicket *t, const RcAdmissionVector *cost, RcBudgetProducer p,
                                             RcBudgetTicket *part) {
    (void)t; (void)cost; (void)p; (void)part; abort();
}
RcAdmissionStatus resource_core_budget_materialize(RcBudgetTicket *t, const RcAdmissionVector *cost) {
    return rc_admission_materialize(&update_ledger, t->token, t->producer, cost, &t->token);
}
RcAdmissionStatus resource_core_budget_cancel(RcBudgetTicket *t) {
    RcAdmissionStatus result = rc_admission_cancel(&update_ledger, t->token, t->producer);
    if (result == RC_ADMISSION_OK) *t = (RcBudgetTicket){0};
    return result;
}
RcAdmissionStatus resource_core_budget_retire(RcBudgetTicket *t) {
    assert(!lock_depth);
    return rc_admission_retire(&update_ledger, t->token, t->producer, &t->token);
}
RcAdmissionStatus resource_core_budget_destroyed(RcBudgetTicket *t) {
    RcAdmissionStatus result = rc_admission_destroyed(&update_ledger, t->token, t->producer);
    if (result == RC_ADMISSION_OK) *t = (RcBudgetTicket){0};
    return result;
}
VKAPI_ATTR VkResult VKAPI_CALL vmaCreateBuffer(VmaAllocator allocator, const VkBufferCreateInfo *buffer,
        const VmaAllocationCreateInfo *allocation, VkBuffer *out, VmaAllocation *owner, VmaAllocationInfo *info) {
    assert(context_ready && !lock_depth && allocator == context.vma && creates < sizeof native / sizeof native[0]);
    assert(buffer->usage == (VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    assert(allocation->usage == VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    assert(allocation->flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
    assert(allocation->flags & VMA_ALLOCATION_CREATE_MAPPED_BIT);
    assert((allocation->requiredFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
           (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    Native *memory = &native[creates++];
    *memory = (Native){.buffer = (VkBuffer)(uintptr_t)creates, .map = malloc((size_t)buffer->size),
                       .type = native_type, .occupied = buffer->size + NATIVE_PADDING, .live = 1};
    assert(memory->map);
    *out = memory->buffer;
    *owner = (VmaAllocation)memory;
    *info = (VmaAllocationInfo){.size = memory->occupied, .memoryType = native_type, .pMappedData = memory->map};
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vmaDestroyBuffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation owner) {
    Native *memory = (Native *)owner;
    assert(through_destroy_adapter && context_ready && !lock_depth && allocator == context.vma);
    assert(memory && memory->live && memory->buffer == buffer);
    memory->live = 0;
    free(memory->map);
    memory->map = NULL;
    destroys++;
}
void resource_core_vma_destroy_buffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation owner) {
    assert(!through_destroy_adapter);
    Native *memory = (Native *)owner;
    if (memory->ticket.ledger)
        assert(rc_admission_retire(&update_ledger, memory->ticket, RC_BUDGET_BUFFER, &memory->ticket) == RC_ADMISSION_OK);
    through_destroy_adapter = 1;
    vmaDestroyBuffer(allocator, buffer, owner);
    through_destroy_adapter = 0;
    if (memory->ticket.ledger) {
        assert(rc_admission_destroyed(&update_ledger, memory->ticket, RC_BUDGET_BUFFER) == RC_ADMISSION_OK);
        memory->ticket = (RcAdmissionToken){0};
        update_retired++;
    }
}
VkResult resource_core_vma_create_buffer(const RcVmaRequest *request, VmaAllocator allocator,
        const VkBufferCreateInfo *buffer, const VmaAllocationCreateInfo *allocation,
        VkBuffer *out, VmaAllocation *owner, VmaAllocationInfo *info, RcAdmissionStatus *admission) {
    assert(!lock_depth && request->producer == RC_BUDGET_BUFFER);
    assert(!request->dependency || !request->peak_ticket);
    compat_optional += !request->required;
    assert(!request->host_bytes && !request->host_ticket);
    update_attempts++;
    if (admission) *admission = grant_mode == GRANT_PENDING ? RC_ADMISSION_REFUSED : RC_ADMISSION_OK;
    if (grant_mode == GRANT_PENDING || grant_mode == GRANT_PLACEMENT_OOM) {
        assert(request->dependency);
        *request->dependency = configured_dependency;
        if (request->refused_peak) *request->refused_peak = configured_dependency.peak;
        return grant_mode == GRANT_PENDING ? VK_NOT_READY : placement_failure;
    }
    if (grant_mode == GRANT_HARD) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    VmaAllocationInfo declared = {.size = buffer->size + NATIVE_PADDING, .memoryType = native_type};
    RcAdmissionVector cost;
    assert(resource_core_vma_native_footprint(&declared, RC_BUDGET_BUFFER, &cost) == RC_ADMISSION_OK);
    RcAdmissionToken ticket;
    if (request->peak_ticket) {
        assert(allocation->flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
        assert(allocation->memoryTypeBits == (1u << native_type));
        assert(rc_admission_split(&update_ledger, request->peak_ticket->token, request->peak_ticket->producer,
                                  &cost, RC_BUDGET_BUFFER, &ticket) == RC_ADMISSION_OK);
    } else {
        assert(rc_admission_reserve(&update_ledger, RC_BUDGET_BUFFER, &cost, &ticket) == RC_ADMISSION_OK);
    }
    VkResult result = vmaCreateBuffer(allocator, buffer, allocation, out, owner, info);
    assert(result == VK_SUCCESS);
    Native *memory = (Native *)*owner;
    memory->cost = cost;
    memory->dedicated = grant_dedicated || (allocation->flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    assert(rc_admission_materialize(&update_ledger, ticket, RC_BUDGET_BUFFER, &cost, &memory->ticket) == RC_ADMISSION_OK);
    update_materialized++;
    if (grant_mode == GRANT_UNMAPPED) info->pMappedData = NULL;
    return result;
}
VkResult resource_core_vma_buffer_peak(const RcVmaRequest *request, const VkBufferCreateInfo *buffer,
        const VmaAllocationCreateInfo *allocation, VmaAllocationCreateInfo *placement, RcAdmissionVector *peak) {
    assert(!lock_depth && request->required && request->producer == RC_BUDGET_BUFFER);
    assert(allocation->flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    assert(buffer->size && buffer->size % 256 == 0);
    *placement = *allocation;
    placement->memoryTypeBits = 1u << native_type;
    VmaAllocationInfo selected = {.size = buffer->size + NATIVE_PADDING, .memoryType = native_type};
    assert(resource_core_vma_native_footprint(&selected, RC_BUDGET_BUFFER, peak) == RC_ADMISSION_OK);
    peak_queries++;
    return VK_SUCCESS;
}
RcAdmissionStatus resource_core_vma_allocation_cost(VmaAllocation allocation, RcAdmissionVector *out) {
    const Native *memory = (const Native *)allocation;
    assert(memory && memory->live && memory->ticket.ledger);
    *out = memory->cost;
    return RC_ADMISSION_OK;
}
VKAPI_ATTR void VKAPI_CALL vmaGetAllocationInfo2(VmaAllocator allocator, VmaAllocation allocation,
                                               VmaAllocationInfo2 *info) {
    const Native *memory = (const Native *)allocation;
    assert(allocator == context.vma && memory->live && !lock_depth);
    *info = (VmaAllocationInfo2){.allocationInfo = {.size = memory->occupied, .memoryType = memory->type,
                                                   .pMappedData = memory->map},
                                .dedicatedMemory = memory->dedicated, .blockSize = memory->occupied};
}

void vk_gpa_write_abandon(VkGpaWriteScope *scope, int may) { (void)scope; (void)may; abort(); }
void vk_gpa_write_complete(VkGpaWriteScope *scope, uint64_t fence) { (void)scope; (void)fence; abort(); }
void vk_gpa_write_cancel(VkGpaWriteScope *scope) { (void)scope; abort(); }
void resource_host_write_dispose(ResourceHostWrite *host, uint64_t fence, int may, int cpu_visible) {
    (void)host; (void)fence; (void)may; (void)cpu_visible; abort();
}
RcRegistry *rc_registry_runtime(void) { return (RcRegistry *)(uintptr_t)1; }
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
'''
    fixture += '\n'.join((chunk_size, pending_type, owners, footprint)) + r'''
static PlatMutex st_lock = PLAT_MUTEX_INIT, res_lock = PLAT_MUTEX_INIT;
static PlatCond cv_work = PLAT_COND_INIT;
static uint64_t done;
static Pending *queue, *rqueue;
static int qhead, qn, qcap, rqhead, rqn;

typedef struct DrawCtx {
    CbRes *res;
    uint64_t res_fence, prep_incarnation;
    int B;
    uint32_t c0_binst;
} DrawCtx;
static _Thread_local DrawCtx *dc;
static void prepared_stats_flush(DrawCtx *context) { assert(context == dc); }

volatile int32_t vk_lost;
int vk_sync_stopping(void) { return 0; }
int vk_exec_recording_failed(void) { return 0; }
void vk_device_lost_check(VkResult result, const char *where) { (void)result; (void)where; abort(); }
static uint64_t issued_scene, handed_scene;
uint64_t vk_sync_fence_done(void) { return done; }
uint64_t vk_sync_handed_fence(void) { return handed_scene; }
uint64_t vk_sync_issued(void) { return issued_scene; }

enum { W_UB, DEST_OFFSET = 1024 };
static const uint64_t destination = 0x10000;
static const VkBuffer destination_buffer = (VkBuffer)(uintptr_t)0x10000;
static int unbacked, exact_destination, remapped_destination;
static unsigned input_frees, native_resolves, refused_updates, declarations, raw_declarations;
struct ResourceCommandEffects { unsigned alive; };
static ResourceCommandEffects effects={1};
RcRegistryResult resource_effects_add_intervals(ResourceCommandEffects *e,const RcInterval *plans,uint32_t count,
 const RcInterval *actual,uint32_t n,RcRegistryWriterKind kind){
 assert(e==&effects&&e->alive&&plans==actual&&count==1&&n==1&&kind==RC_REGISTRY_BUFFER&&plans[0].mapping==83);
 declarations++;return RC_REGISTRY_OK;
}
void resource_effects_add(ResourceCommandEffects *e,RcSourceRange plan,RcSourceRange actual,RcRegistryWriterKind kind){
 assert(e==&effects&&kind==RC_REGISTRY_BUFFER&&plan.backing==RC_GUEST_COMPATIBILITY_BACKING&&rc_range_equal(plan,actual));
 raw_declarations++;
}
static unsigned warnings;
static const p_update_ub *emitting;
static VkExec *emitting_scene;
static VkBuffer copy_source;
static VkDeviceSize expected_offset;
static unsigned emitted_count;
static char emitted[16];
VkExecFrame vk_exec_frame;
int vk_mirror_on = 1;

static void event(char kind) {
    assert(emitted_count < sizeof emitted);
    emitted[emitted_count++] = kind;
}
static void warn_once(int key, const char *fmt, ...) {
    assert(key == W_UB && fmt && !emitted_count);
    warnings++;
}
int vk_mem_lookup(uint64_t address, uint64_t bytes, VkMemSpan *out) {
    assert(!lock_depth && address == destination && bytes);
    if (unbacked) return 0;
    *out = (VkMemSpan){.buffer = destination_buffer, .offset = DEST_OFFSET, .avail = bytes,
                       .cls = VK_MCLS_HOST_CACHED, .pool = &context};
    return 1;
}
int vk_mem_resolve(uint64_t address, uint64_t bytes, VkMemSpan *out) {
    native_resolves++;
    return vk_mem_lookup(address, bytes, out);
}
struct VkMemSpanCapture { VkMemSpan span; };
RcRegistryResult vk_mem_span_prepare_native_input(uint64_t address,uint64_t bytes,int clamp,VkMemMirrorInputPolicy policy,
                                                  VkMemSpan *span,VkMemSpanCapture **hold){
    assert(!clamp&&!*hold&&policy==VK_MEM_MIRROR_IMPORT);
    if (!exact_destination || !vk_mem_lookup(address,bytes,span)) return RC_REGISTRY_UNSUPPORTED;
    *hold=malloc(sizeof **hold);assert(*hold);**hold=(VkMemSpanCapture){*span};return RC_REGISTRY_OK;
}
void vk_mem_span_capture_release(VkMemSpanCapture **hold){if(*hold){free(*hold);*hold=NULL;input_frees++;}}
int vk_mem_span_capture_current(const VkMemSpanCapture *hold,const VkMemSpan *span){
    assert(hold&&span&&span->buffer==hold->span.buffer);return !remapped_destination;
}
int vk_mem_span_record_input(VkMemSpanCapture *hold,const VkMemSpan *span){
    return vk_mem_span_capture_current(hold,span);
}
uint32_t vk_mem_span_input_intervals(const VkMemSpanCapture *hold,uint64_t address,uint64_t size,RcInterval **out){
    assert(hold&&size==hold->span.avail&&!*out);*out=malloc(sizeof **out);assert(*out);
    **out=(RcInterval){address,83,{99,hold->span.offset,size}};return 1;
}
int vk_mem_span_input_cpu_visible(const VkMemSpanCapture *hold){assert(hold);return 1;}
int vk_mem_span_buffer_location(const VkMemSpanCapture *hold,const VkMemSpan *span,RcRegistryBufferLocation *value){
    assert(vk_mem_span_capture_current(hold,span)&&span->buffer==destination_buffer);
    *value=(RcRegistryBufferLocation){17,3,span->offset};
    return 1;
}
void resource_effects_buffer_prepare_native(const RcInterval *ranges,uint32_t count,uint64_t address,
                                             RcRegistryBufferLocation value){
    assert(count==1&&ranges[0].mapping==83&&ranges[0].address==address&&address==emitting->dst);
    assert(value.allocation==17&&value.incarnation==3&&value.offset==DEST_OFFSET);
    event('L');
}

void vk_mem_input_provider_release(VkMemInputProvider provider){assert(!provider.pool);}
void vk_exec_record_native_input_provider(VkExec *ex,VkMemSpanCapture **hold){
    assert(ex&&*hold);vk_mem_span_capture_release(hold);
}
int resource_effects_buffer_prepare_intervals(const RcInterval *ranges,uint32_t count,int visible){
    assert(ranges&&count==1&&ranges[0].mapping==83&&visible);return 1;
}
int resource_effects_buffer_prepare(uint64_t address,uint64_t size,int visible){
    assert(address==destination&&size&&visible);return 1;
}
void vk_exec_required_graphics_unavailable(VkExec *ex,uint32_t opcode,const char *reason){
    assert(ex && reason);
    if (opcode == OP_UPDATE_UB) refused_updates++;
    else { assert(!opcode && strstr(reason, "compatibility buffer")); compat_refused++; }
}
void vk_exec_barrier(VkExec *ex) { assert(ex == emitting_scene); ex->in_pass = 0; event('B'); }
void vk_mem_pick(VkMemSpan *span) { assert(span->buffer == destination_buffer); event('P'); }
static int note_written(VkExec *ex, uint64_t address, uint64_t bytes, const VkMemSpan *span) {
    assert(ex == emitting_scene && address == emitting->dst && bytes == emitting->size);
    assert(span->buffer == destination_buffer && span->offset == DEST_OFFSET);
    event('D');
    return 1;
}
static void note_buffer_written(VkExec *ex,uint64_t address,uint64_t size,const VkMemSpan *span,
                                int visible,const RcInterval *captured,uint32_t count){
    assert(visible&&count==1&&captured[0].mapping==83&&captured[0].address==address&&captured[0].range.size==size);
    note_written(ex,address,size,span);
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer command, VkBuffer source, VkBuffer target,
                                           uint32_t count, const VkBufferCopy *copies) {
    assert(command == emitting_scene->cb && target == destination_buffer && count == 1);
    assert(copies->srcOffset == expected_offset && copies->dstOffset == DEST_OFFSET && copies->size == emitting->size);
    for (unsigned i = 0; i < creates; i++)
        if (native[i].buffer == source) {
            assert(native[i].live && !memcmp(native[i].map + expected_offset, emitting->data, (size_t)emitting->size));
            copy_source = source;
            event('C');
            return;
        }
    abort();
}
static void commit_recorded_writes(VkExec *ex) { assert(ex == emitting_scene); event('M'); }
void vk_mem_gpu_wrote(const VkMemSpan *span, uint64_t size) {
    assert(span->buffer == destination_buffer && size == emitting->size);
    event('W');
}
void vk_usnap_drop(uint64_t address, uint64_t size) {
    assert(address == emitting->dst && size == emitting->size);
    event('S');
}
void vk_mem_count(int cls, int use, uint64_t bytes) {
    assert(cls == VK_MCLS_HOST_CACHED && use == VK_MUSE_GPU_WRITE && bytes == emitting->size);
    event('N');
}
static int replay_op_timing(void) { return 0; }
int resource_prep_mode(void) { return 0; }
int rec_batch_begin(VkCommandBuffer cb) { (void)cb; abort(); }
void rec_batch_end(int entered) { (void)entered; abort(); }
uint64_t plat_now_ns(void) { abort(); }
void perf_replay_op_at(int category, uint32_t operation, uint64_t elapsed) {
    (void)category; (void)operation; (void)elapsed; abort();
}
static void cancel_unrecorded_writes(VkExec *ex) { (void)ex; }
static struct { int watched; } UPD;
static void upd_begin(const VkExec *ex){assert(ex&&ex==emitting_scene);}
RcRegistryResult rc_registry_produced_scene_begin(RcRegistry *r,uint64_t scene){
    assert(r==rc_registry_runtime()&&scene==emitting_scene->fence);return RC_REGISTRY_OK;
}
static void upd_note(const VkExec *ex, const p_update_ub *op, VkMemSpan span, VkExecUpdateDestination *target) {
    assert(ex == emitting_scene && op == emitting && span.buffer==destination_buffer && !target->hold);
    event('U');
}
''' + r'''
typedef struct {
    unsigned prepares, indexed, plain, binds, writes, storage_begins, storage_records;
    VkBuffer buffer;
    VkDeviceSize offset;
    uint32_t count, instances, first, first_instance;
    int32_t vertex_offset;
} DrawTrace;
static DrawTrace draw_trace;
static VkExec *observed_scene;
static const p_draw *observed_draw;
static unsigned deferred_draws;
static VkExecGraphicsDraw *graphics_current;

static int graphics_storage_begin(VkExec *ex, VkExecGraphicsDraw *draw) {
    assert(ex == observed_scene && !draw && !draw_trace.indexed && !draw_trace.plain);
    assert(!draw_trace.storage_begins++ && !draw_trace.storage_records);
    return 1;
}
static void graphics_storage_recorded(VkExec *ex, VkExecGraphicsDraw *draw) {
    assert(ex == observed_scene && !draw && draw_trace.storage_begins == 1);
    assert(draw_trace.indexed + draw_trace.plain == 1 && !draw_trace.storage_records++);
}

static int prepare_draw(VkExec *ex, int primitive, int count, int restart) {
    assert(ex == observed_scene && primitive == observed_draw->prim && count == observed_draw->count && !restart);
    draw_trace.prepares++;
    if (deferred_draws) { deferred_draws--; return 0; }
    return 1;
}
VKAPI_ATTR void VKAPI_CALL vkCmdBindIndexBuffer(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset,
                                               VkIndexType type) {
    assert(cb == observed_scene->cb && buffer && offset % 256 == 0 && type == VK_INDEX_TYPE_UINT32);
    assert(!draw_trace.binds++ && !draw_trace.indexed && !draw_trace.plain);
    draw_trace.buffer = buffer;
    draw_trace.offset = offset;
}
VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexed(VkCommandBuffer cb, uint32_t count, uint32_t instances,
                                           uint32_t first, int32_t vertex_offset, uint32_t first_instance) {
    assert(cb == observed_scene->cb && draw_trace.binds == 1 && !draw_trace.plain && !draw_trace.indexed++);
    assert(draw_trace.storage_begins == 1 && !draw_trace.storage_records);
    draw_trace.count = count;
    draw_trace.instances = instances;
    draw_trace.first = first;
    draw_trace.vertex_offset = vertex_offset;
    draw_trace.first_instance = first_instance;
}
VKAPI_ATTR void VKAPI_CALL vkCmdDraw(VkCommandBuffer cb, uint32_t count, uint32_t instances,
                                    uint32_t first, uint32_t first_instance) {
    assert(cb == observed_scene->cb && !draw_trace.binds && !draw_trace.indexed && !draw_trace.plain++);
    assert(draw_trace.storage_begins == 1 && !draw_trace.storage_records);
    draw_trace.count = count;
    draw_trace.instances = instances;
    draw_trace.first = first;
    draw_trace.first_instance = first_instance;
}
static void storage_images_written(VkExec *ex) {
    assert(ex == observed_scene && draw_trace.indexed + draw_trace.plain == 1 && !draw_trace.writes++);
    assert(draw_trace.storage_begins == 1 && draw_trace.storage_records == 1);
}
''' + '\n'.join((metadata, pending, provider, callbacks, updates, draw_index, group_declarations, group_scan)) + r'''

static unsigned reclaim_domain(unsigned domain, uint64_t bytes) {
    assert(reclaim && !lock_depth);
    RcAdmissionVector deficit = {{0}};
    deficit.amount[domain] = bytes;
    return reclaim(reclaim_context, &deficit);
}

static Native *backing(VkBuffer buffer) {
    for (unsigned i = 0; i < creates; i++)
        if (native[i].buffer == buffer) return &native[i];
    abort();
}

static unsigned live_on_heap(unsigned type) {
    unsigned count = 0;
    for (unsigned i = 0; i < creates; i++) count += native[i].live && native[i].type == type;
    return count;
}

static void set_context(void) {
    context.vma = (VmaAllocator)(uintptr_t)1;
    context.device = (VkDevice)(uintptr_t)1;
    context.props.deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    context.mem_props.memoryHeapCount = context.mem_props.memoryTypeCount = 2;
    context.mem_props.memoryTypes[0] = (VkMemoryType){
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0,
    };
    context.mem_props.memoryTypes[1] = (VkMemoryType){
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1,
    };
    context.mem_props.memoryHeaps[0].size = 1ull << 30;
    context.mem_props.memoryHeaps[1] = (VkMemoryHeap){1ull << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
}

static void advance(uint64_t completed, uint64_t handed, int idle) {
    assert(done <= completed && completed <= handed);
    done = completed;
    pending_dispose_remaining(completed, handed, idle);
}

VkSyncProgressResult vk_sync_progress_wait_prior(uint64_t earlier) {
    assert(!lock_depth && earlier > done && earlier <= handed_scene && grant_mode == GRANT_PENDING);
    compat_waited++;
    done = earlier;
    // Complete only the known handed predecessor; the current recorder's callback stays live.
    pending_dispose_queue(&queue, &qhead, &qn, earlier, UINT64_MAX, 0);
    return VK_SYNC_PROGRESS_COMPLETE;
}

static int spare_ready(VkExec *ex, const p_update_ub *op) {
    VkDrawScratchWaitContext context = {.ex = ex, .bytes = op->size};
    return vk_exec_draw_scratch_spare_ready(&context);
}

static p_update_ub *make_update(size_t bytes, unsigned seed) {
    p_update_ub *op = malloc(sizeof *op + bytes);
    assert(op);
    op->dst = destination;
    op->size = bytes;
    for (size_t i = 0; i < bytes; i++) op->data[i] = (uint8_t)(seed + i);
    return op;
}

static VkResult prepare_update_once(VkExec *ex,const p_update_ub *op,VkDrawScratchUpdate **grant,RcVmaDependency *dep){
    VkExecUpdateDestination output={0};
    RcRegistryResult selected=vk_exec_draw_update_select(op,&output);
    assert(selected==RC_REGISTRY_OK);
    VkResult result=vk_exec_draw_update_prepare(ex,op,&output,grant,dep);
    vk_exec_draw_update_drop(&output);
    return result;
}
#define vk_exec_draw_update_prepare prepare_update_once

static void expect_update_rejected(VkExec *ex, const p_update_ub *op, VkDrawScratchUpdate *grant) {
    assert(!emitted_count);
    expect_fatal = 1;
    if (!setjmp(fatal_return)) {
        vk_exec_draw_update_commit(ex, op, grant);
        abort();
    }
    expect_fatal = 0;
    assert(!emitted_count && grant->owner == ex->dc->res);
}

static VkBuffer emit_update(VkExec *ex, const p_update_ub *op, VkDrawScratchUpdate *grant, VkDeviceSize offset) {
    assert(!emitted_count);
    emitting = op;
    emitting_scene = ex;
    expected_offset = offset;
    copy_source = VK_NULL_HANDLE;
    int before = ex->recorded;
    ex->resource_op = ++ex->resource_next;
    const char *expected=exact_destination ? "BDLCMWSNU" : "BPDCMWSNU";
    if (exact_destination) assert(update_declare(ex,op,&effects)==RC_REGISTRY_OK&&grant->declared);
    vk_exec_draw_update_commit(ex, op, grant);
    assert(emitted_count == strlen(expected) && !memcmp(emitted, expected, emitted_count));
    assert(ex->recorded == before + 1 && copy_source && !grant->owner);
    emitted_count = 0;
    emitting = NULL;
    emitting_scene = NULL;
    return copy_source;
}

static void check_update_preparation(void) {
    const RcAdmissionDomain domains[] = {
        {RC_ADMISSION_RAM, 0, UINT64_MAX}, {RC_ADMISSION_COMMIT, 0, UINT64_MAX},
        {RC_ADMISSION_STAGING, 0, UINT64_MAX}, {RC_ADMISSION_GPU_HEAP, 0, UINT64_MAX},
        {RC_ADMISSION_GPU_HEAP, 1, UINT64_MAX},
    };
    assert(rc_admission_init(&update_ledger, 1, domains, sizeof domains / sizeof domains[0],
                             update_tickets, sizeof update_tickets / sizeof update_tickets[0]) == RC_ADMISSION_OK);
    DrawCtx recorder = {0};
    VkExec ex = {.dc = &recorder, .fence = 1, .resource_op = 2, .cb = (VkCommandBuffer)(uintptr_t)1};
    p_update_ub *zero = make_update(0, 0), *small = make_update(1, 0x61), *tail = make_update(257, 0x42);
    VkDrawScratchUpdate *grant = NULL;
    RcVmaDependency dependency = {0};

    assert(vk_exec_draw_update_prepare(&ex, zero, &grant, &dependency) == VK_SUCCESS && !grant);
    assert(!update_attempts && !ex.recorded && !emitted_count && !warnings);
    unbacked = 1;
    assert(vk_exec_draw_update_prepare(&ex, small, &grant, &dependency) == VK_SUCCESS);
    assert(grant && !grant->slab && !recorder.res->update_slabs && !update_attempts && !emitted_count);
    ex.resource_op = ++ex.resource_next;
    vk_exec_draw_update_commit(&ex, small, grant);
    grant = NULL;
    assert(!ex.recorded && !emitted_count && warnings == 1 && !recorder.res->update_pending.owner);
    unbacked = 0;

    grant_mode = GRANT_PENDING;
    configured_dependency.selected = 1;
    configured_dependency.peak.amount[RC_BUDGET_RAM] = CHUNK_BYTES;
    assert(vk_exec_draw_update_prepare(&ex, small, &grant, &dependency) == VK_NOT_READY);
    assert(!grant && dependency.selected && dependency.peak.amount[RC_BUDGET_RAM] == CHUNK_BYTES);
    assert(!creates && !ex.recorded && !emitted_count && !recorder.res->update_slabs);
    grant_mode = GRANT_HARD;
    assert(vk_exec_draw_update_prepare(&ex, small, &grant, &dependency) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
    assert(!grant && !creates && !ex.recorded && !emitted_count && !recorder.res->update_slabs);
    grant_mode = GRANT_UNMAPPED;
    assert(vk_exec_draw_update_prepare(&ex, small, &grant, &dependency) == VK_ERROR_MEMORY_MAP_FAILED);
    assert(!grant && creates == 1 && destroys == 1 && !ex.recorded && !emitted_count);
    assert(!recorder.res->update_pending.owner && !recorder.res->update_slabs);
    assert(!update_ledger.retained.amount[RC_BUDGET_RAM] && !update_ledger.pending.amount[RC_BUDGET_RAM]);

    grant_mode = GRANT_READY;
    assert(vk_exec_draw_update_prepare(&ex, small, &grant, &dependency) == VK_SUCCESS);
    assert(grant && grant->fresh && !grant->offset && grant->aligned == 256 && !recorder.res->update_slabs);
    VkBuffer reserved = grant->slab->chunk.buf;
    assert(creates == 2 && backing(reserved)->ticket.ledger && !ex.recorded && !emitted_count);
    vk_exec_draw_update_cancel(&ex, grant);
    grant = NULL;
    assert(!recorder.res->update_pending.owner && !recorder.res->update_slabs && nspare_chunk == 1);
    assert(spare_ready(&ex, small));
    unsigned before_reuse = update_attempts;
    assert(vk_exec_draw_update_prepare(&ex, small, &grant, &dependency) == VK_SUCCESS && grant);
    assert(grant->slab->chunk.buf == reserved && update_attempts == before_reuse && !nspare_chunk);

    ex.resource_op = grant->ordinal;
    p_update_ub *different = make_update(1, 0x61);
    expect_update_rejected(&ex, different, grant);
    free(different);
    ex.resource_op++;
    expect_update_rejected(&ex, small, grant);
    ex.resource_op--;
    ex.fence++;
    expect_update_rejected(&ex, small, grant);
    ex.fence--;
    ex.in_pass = 1;
    assert(emit_update(&ex, small, grant, 0) == reserved);
    grant = NULL;
    assert(vk_exec_frame.ub_updates == 1 && vk_exec_frame.ub_pass_breaks == 1);
    assert(recorder.res->update_slabs->chunk.used == 256 && backing(reserved)->map[0] == 0x61);


    assert(vk_exec_draw_update_prepare(&ex, tail, &grant, &dependency) == VK_SUCCESS);
    assert(grant && !grant->fresh && grant->offset == 256 && grant->aligned == 512);
    assert(recorder.res->update_slabs->chunk.used == 256 && !emitted_count && ex.recorded == 1);
    vk_exec_draw_update_cancel(&ex, grant);
    grant = NULL;
    assert(recorder.res->update_slabs->chunk.used == 256 && !recorder.res->update_pending.owner);
    assert(vk_exec_draw_update_prepare(&ex, tail, &grant, &dependency) == VK_SUCCESS);
    assert(emit_update(&ex, tail, grant, 256) == reserved && update_attempts == before_reuse);
    assert(recorder.res->update_slabs->chunk.used == 768 && backing(reserved)->map[0] == 0x61);

    exact_destination=1;
    unsigned resolves_before=native_resolves;
    assert(vk_exec_draw_update_prepare(&ex,small,&grant,&dependency)==VK_SUCCESS&&grant->destination.hold);
    assert(emit_update(&ex,small,grant,768)==reserved && input_frees==1 && declarations==1 && raw_declarations==1);
    assert(native_resolves==resolves_before);
    grant=NULL;
    assert(vk_exec_draw_update_prepare(&ex,small,&grant,&dependency)==VK_SUCCESS&&grant->destination.hold);
    ex.resource_op=++ex.resource_next;
    assert(update_declare(&ex,small,&effects)==RC_REGISTRY_OK&&grant->declared);
    remapped_destination=1;
    vk_exec_draw_update_commit(&ex,small,grant);
    assert(refused_updates==1 && !emitted_count && input_frees==2 && native_resolves==resolves_before);
    assert(!grant->owner && recorder.res->update_slabs->chunk.used==1024);
    remapped_destination=exact_destination=0;

    advance(1, 1, 0);
    assert(!qn && nspare_chunk == 1 && backing(reserved)->live);
    assert(reclaim_domain(RC_BUDGET_RAM, CHUNK_BYTES) == RC_BUDGET_RECLAIM_DESTROYED);
    assert(update_materialized == update_retired && !backing(reserved)->live && !nspare_chunk);

    free(tail);
    free(small);
    free(zero);
}

static void check_update_prefix(void) {
    DrawCtx recorder = {0};
    VkExec ex = {.dc = &recorder, .fence = 2, .cb = (VkCommandBuffer)(uintptr_t)2};
    p_update_ub *full = make_update(CHUNK_BYTES, 0x21);
    p_update_ub *small = make_update(32, 0x72);
    p_update_ub *large = make_update(CHUNK_BYTES + 1, 0x32);
    VkDrawScratchUpdate *grant = NULL;
    RcVmaDependency dependency;
    assert(vk_exec_draw_update_prepare(&ex, full, &grant, &dependency) == VK_SUCCESS && grant);
    VkBuffer old_prefix = emit_update(&ex, full, grant, 0);
    assert(backing(old_prefix)->live && recorder.res->update_slabs->chunk.used == CHUNK_BYTES);

    grant = NULL;
    grant_mode = GRANT_PENDING;
    configured_dependency = (RcVmaDependency){.selected = 1, .memory_type = 1};
    configured_dependency.peak.amount[RC_BUDGET_GPU_FIRST + 1] = CHUNK_BYTES;
    assert(vk_exec_draw_update_prepare(&ex, small, &grant, &dependency) == VK_NOT_READY && !grant);
    const RcBudgetReadiness now = {.records_available = 1, .usable_mask = 1u << (RC_BUDGET_GPU_FIRST + 1)};
    assert(!vk_exec_draw_scratch_prefix_useful(&ex, small->size, &(RcVmaDependency){0}, &now));
    assert(!vk_exec_draw_scratch_prefix_useful(&ex, large->size, &dependency, &now));
    assert(vk_exec_draw_scratch_prefix_useful(&ex, small->size, &dependency, &now));
    assert(!spare_ready(&ex, small) && !nspare_chunk && done == 1);

    // Driver OOM can park for this full prefix only if the selected placement could actually fit.
    grant_mode = GRANT_PLACEMENT_OOM;
    configured_readiness = now;
    native_type = 1;
    unsigned before_create = creates;
    assert(vk_exec_draw_update_prepare(&ex, small, &grant, &dependency) == VK_NOT_READY && !grant);
    placement_failure = VK_ERROR_OUT_OF_HOST_MEMORY;
    assert(vk_exec_draw_update_prepare(&ex, small, &grant, &dependency) == VK_NOT_READY && !grant);
    placement_failure = VK_ERROR_DEVICE_LOST;
    assert(vk_exec_draw_update_prepare(&ex, small, &grant, &dependency) == VK_ERROR_DEVICE_LOST && !grant);
    placement_failure = VK_ERROR_OUT_OF_DEVICE_MEMORY;

    VkMemoryHeap *heap = &context.mem_props.memoryHeaps[1];
    VkDeviceSize physical = heap->size;
    heap->size = CHUNK_BYTES + NATIVE_PADDING - 1;
    assert(vk_exec_draw_update_prepare(&ex, small, &grant, &dependency) == VK_ERROR_OUT_OF_DEVICE_MEMORY && !grant);
    heap->size = physical;
    requirement_type_override = 1u << 0;
    assert(vk_exec_draw_update_prepare(&ex, small, &grant, &dependency) == VK_ERROR_OUT_OF_DEVICE_MEMORY && !grant);
    requirement_type_override = 0;
    context.mem_props.memoryTypes[1].propertyFlags &= ~VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    assert(vk_exec_draw_update_prepare(&ex, small, &grant, &dependency) == VK_ERROR_OUT_OF_DEVICE_MEMORY && !grant);
    context.mem_props.memoryTypes[1].propertyFlags |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    native_type = 0;
    assert(creates == before_create && !emitted_count && ex.recorded == 1);
    assert(!recorder.res->update_pending.owner && recorder.res->update_slabs->chunk.buf == old_prefix);

    unsigned before_destroy = destroys, before_attempts = update_attempts;
    expect_fatal = 1;
    if (!setjmp(fatal_return)) {
        vk_exec_draw_update_prefix_completed(&ex, 1);
        abort();
    }
    expect_fatal = 0;
    assert(backing(old_prefix)->live && recorder.res->update_slabs && !nspare_chunk && qn == 1);

    // The synchronization owner now proves this exact submitted prefix idle, without completing the scene.
    vk_exec_draw_update_prefix_completed(&ex, 2);
    assert(!recorder.res->update_slabs && nspare_chunk == 1 && qn == 1 && done == 1);
    assert(spare_ready(&ex, small) && !spare_ready(&ex, large));
    assert(backing(old_prefix)->ticket.ledger && destroys == before_destroy);
    assert(vk_exec_draw_update_prepare(&ex, small, &grant, &dependency) == VK_SUCCESS && grant);
    assert(!grant->offset && !nspare_chunk && update_attempts == before_attempts);
    assert(emit_update(&ex, small, grant, 0) == old_prefix);

    advance(2, 2, 0);
    assert(!qn && nspare_chunk == 1);
    assert(reclaim_domain(RC_BUDGET_RAM, CHUNK_BYTES) == RC_BUDGET_RECLAIM_DESTROYED);
    assert(!backing(old_prefix)->live && !nspare_chunk && destroys == before_destroy + 1);
    free(large);
    free(small);
    free(full);
}

static void check_update_prior(void) {
    DrawCtx older[2] = {{0}}, current = {0}, abandoned = {0};
    VkExec unrelated = {.dc = &older[0], .fence = 3, .cb = (VkCommandBuffer)(uintptr_t)3};
    VkExec compatible = {.dc = &older[1], .fence = 4, .cb = (VkCommandBuffer)(uintptr_t)4};
    VkExec ex = {.dc = &current, .fence = 5, .cb = (VkCommandBuffer)(uintptr_t)5};
    VkExec abandoned_ex = {.dc = &abandoned, .fence = 6};
    p_update_ub *large = make_update(CHUNK_BYTES + 1, 0x64);
    p_update_ub *small = make_update(32, 0x16);
    VkDrawScratchUpdate *grant = NULL;
    RcVmaDependency dependency;
    grant_mode = GRANT_READY;
    native_type = 0;
    grant_dedicated = 1;
    assert(vk_exec_draw_update_prepare(&unrelated, large, &grant, &dependency) == VK_SUCCESS && grant);
    VkBuffer unrelated_buffer = emit_update(&unrelated, large, grant, 0);
    grant_dedicated = 0;
    native_type = 1;
    assert(vk_exec_draw_update_prepare(&compatible, large, &grant, &dependency) == VK_SUCCESS && grant);
    VkBuffer compatible_buffer = emit_update(&compatible, large, grant, 0);
    assert(vk_exec_draw_update_prepare(&ex, large, &grant, &dependency) == VK_SUCCESS && grant);
    VkBuffer current_buffer = emit_update(&ex, large, grant, 0);

    configured_dependency = (RcVmaDependency){.selected = 1, .memory_type = 1};
    configured_dependency.peak.amount[RC_BUDGET_GPU_FIRST + 1] = CHUNK_BYTES;
    grant_mode = GRANT_PENDING;
    assert(vk_exec_draw_update_prepare(&ex, large, &grant, &dependency) == VK_NOT_READY && !grant);
    const RcBudgetReadiness now = {.records_available = 1, .usable_mask = 1u << (RC_BUDGET_GPU_FIRST + 1)};
    issued_scene = 5;
    handed_scene = 2;
    assert(!vk_exec_draw_scratch_prior_useful(&ex, large->size, &dependency, &now));
    handed_scene = 3;
    assert(!vk_exec_draw_scratch_prior_useful(&ex, large->size, &dependency, &now));
    handed_scene = 4;
    assert(vk_exec_draw_scratch_prior_useful(&ex, large->size, &dependency, &now) == 4);
    assert(!vk_exec_draw_scratch_prior_useful(&compatible, large->size, &dependency, &now));

    // This recording has no releasable prefix. Only the useful older scene's final handoff changes the result.
    grant_mode = GRANT_PLACEMENT_OOM;
    configured_readiness = now;
    handed_scene = 3;
    assert(vk_exec_draw_update_prepare(&abandoned_ex, large, &grant, &dependency) ==
           VK_ERROR_OUT_OF_DEVICE_MEMORY && !grant);
    handed_scene = 4;
    assert(vk_exec_draw_update_prepare(&abandoned_ex, large, &grant, &dependency) == VK_NOT_READY && !grant);
    assert(!abandoned.res && !abandoned_ex.recorded && !emitted_count);

    RcVmaDependency placement = dependency;
    placement.can_reuse = 1;
    placement.peak = (RcAdmissionVector){{0}};
    assert(vk_exec_draw_scratch_prior_useful(&ex, large->size, &placement, &now) == 4);
    placement.memory_type = 0;
    assert(!vk_exec_draw_scratch_prior_useful(&ex, large->size, &placement, &now));
    vk_lost = 1;
    assert(!vk_exec_draw_scratch_prior_useful(&ex, large->size, &dependency, &now));
    vk_lost = 0;
    assert(backing(compatible_buffer)->live && backing(unrelated_buffer)->live && backing(current_buffer)->live);

    // Already completed and current scenes cannot be chosen for an older-fence wait.
    done = 4;
    handed_scene = 5;
    assert(!vk_exec_draw_scratch_prior_useful(&ex, large->size, &dependency, &now));
    advance(4, 5, 0);
    assert(!backing(compatible_buffer)->live && !backing(unrelated_buffer)->live && backing(current_buffer)->live);

    grant_mode = GRANT_READY;
    native_type = 0;
    assert(vk_exec_draw_update_prepare(&abandoned_ex, small, &grant, &dependency) == VK_SUCCESS && grant);
    VkBuffer unpublished = grant->slab->chunk.buf;
    assert(grant->fresh && !abandoned.res->update_slabs && !abandoned_ex.recorded);
    advance(5, 5, 0); // Completes current and cancels the unhanded CbRes plus its unpublished span.
    assert(!qn && !backing(current_buffer)->live && backing(unpublished)->live && nspare_chunk == 1);
    assert(reclaim_domain(RC_BUDGET_RAM, CHUNK_BYTES) == RC_BUDGET_RECLAIM_DESTROYED);
    assert(creates == destroys && update_materialized == update_retired && !update_head && !update_tail);
    assert(!update_ledger.retained.amount[RC_BUDGET_RAM] && !update_ledger.pending.amount[RC_BUDGET_RAM]);
    free(small);
    free(large);
}

static void materialize_credit(VkDrawScratchCredit *credit, const RcAdmissionVector *peak) {
    RcBudgetTicket joint = {.producer = RC_BUDGET_OTHER};
    assert(rc_admission_reserve(&update_ledger, joint.producer, peak, &joint.token) == RC_ADMISSION_OK);
    assert(vk_exec_draw_update_credit_materialize(credit, &joint) == VK_SUCCESS);
    RcAdmissionRecord *parent;
    assert(rc_admission_lookup(&update_ledger, joint.token, joint.producer, &parent) == RC_ADMISSION_OK);
    for (unsigned d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) assert(!parent->amount.amount[d]);
    assert(rc_admission_cancel(&update_ledger, joint.token, joint.producer) == RC_ADMISSION_OK);
}

static void check_update_credit(void) {
    VkDrawScratchCredit *first = NULL, *second = NULL, *queued = NULL;
    RcAdmissionVector peak;
    int drain;
    unsigned before_create = creates, before_destroy = destroys, before_peak = peak_queries;
    unsigned before_materialized = update_materialized;
    size_t header = sizeof(VkDrawScratchCredit);
    assert(rc_admission_set_limit(&update_ledger, RC_BUDGET_RAM, header - 1) == RC_ADMISSION_OK);
    assert(vk_exec_draw_update_credit_plan(513, &first, &peak, &drain) == VK_NOT_READY);
    assert(!first && !drain && !scratch_credit.receipts && !update_ledger.retained.amount[RC_BUDGET_RAM]);
    assert(peak.amount[RC_BUDGET_RAM] == header && peak.amount[RC_BUDGET_COMMIT] == header);
    assert(rc_admission_set_limit(&update_ledger, RC_BUDGET_RAM, UINT64_MAX) == RC_ADMISSION_OK);
    assert(vk_exec_draw_update_credit_plan(0, &first, &peak, &drain) == VK_SUCCESS);
    assert(!first && !drain && peak_queries == before_peak && !scratch_credit.receipts);

    assert(vk_exec_draw_update_credit_plan(513, &first, &peak, &drain) == VK_SUCCESS && first);
    assert(!drain && scratch_credit.receipts == 1 && !scratch_credit.chunk.alloc && peak_queries == before_peak + 1);
    uint64_t dropped_incarnation = first->incarnation;
    assert(vk_exec_draw_update_credit_host_bytes(first) == header &&
           update_ledger.retained.amount[RC_BUDGET_RAM] == header);
    assert(vk_exec_draw_update_credit_plan(257, &second, &peak, &drain) == VK_NOT_READY && !second && !drain);
    assert(vk_exec_draw_update_credit_plan(769, &second, &peak, &drain) == VK_NOT_READY && !second && drain);
    assert(creates == before_create && peak_queries == before_peak + 1);
    vk_exec_draw_update_credit_cancel(first);
    first = NULL;
    assert(!scratch_credit.receipts && !scratch_credit.capacity && creates == before_create);
    assert(!update_ledger.retained.amount[RC_BUDGET_RAM]);

    assert(vk_exec_draw_update_credit_plan(513, &first, &peak, &drain) == VK_SUCCESS && first && !drain);
    assert(first->incarnation != dropped_incarnation && first->bytes == 768);
    assert(peak.amount[RC_BUDGET_RAM] == 768 + NATIVE_PADDING);
    materialize_credit(first, &peak);
    VkBuffer shared = scratch_credit.chunk.buf;
    Native *memory = backing(shared);
    assert(first->ready && memory->dedicated && memory->ticket.ledger && creates == before_create + 1);
    assert(scratch_credit.chunk.size == 768 && scratch_credit.capacity == 768 && !nspare_chunk);

    assert(vk_exec_draw_update_credit_plan(257, &second, &peak, &drain) == VK_SUCCESS && second && !drain);
    assert(second != first && second->incarnation == first->incarnation && !peak.amount[RC_BUDGET_RAM]);
    assert(vk_exec_draw_update_credit_materialize(second, NULL) == VK_SUCCESS && second->ready);
    assert(vk_exec_draw_update_credit_plan(32, &queued, &peak, &drain) == VK_SUCCESS && queued && !drain);
    assert(vk_exec_draw_update_credit_materialize(queued, NULL) == VK_SUCCESS);
    assert(scratch_credit.receipts == 3);
    vk_exec_draw_update_credit_cancel(queued);
    queued = NULL;
    assert(scratch_credit.receipts == 2 && scratch_credit.chunk.buf == shared);
    assert(vk_exec_draw_update_credit_plan(769, &queued, &peak, &drain) == VK_NOT_READY && !queued && drain);
    assert(peak_queries == before_peak + 2 && creates == before_create + 1 && update_materialized == before_materialized + 1);

    DrawCtx older = {0}, newer = {0};
    VkExec first_scene = {.dc = &older, .fence = 7, .cb = (VkCommandBuffer)(uintptr_t)7};
    VkExec second_scene = {.dc = &newer, .fence = 8, .cb = (VkCommandBuffer)(uintptr_t)8};
    VkDrawScratchCredit *older_receipt = first, *newer_receipt = second;
    vk_exec_draw_update_credit_adopt(&first_scene, &first);
    vk_exec_draw_update_credit_adopt(&second_scene, &second);
    assert(!first && !second && qn == 2 && older_receipt->owner == older.res && newer_receipt->owner == newer.res);
    assert(older.res->update_credit == older_receipt && newer.res->update_credit == newer_receipt && older.res != newer.res);

    p_update_ub *fill = make_update(513, 0x81), *next = make_update(257, 0x72);
    VkDrawScratchUpdate *grant = NULL;
    RcVmaDependency dependency;
    unsigned before_attempts = update_attempts;
    assert(vk_exec_draw_update_prepare(&first_scene, fill, &grant, &dependency) == VK_SUCCESS);
    assert(grant && grant->protected_span && !grant->offset && !scratch_credit.borrower);
    vk_exec_draw_update_cancel(&first_scene, grant);
    assert(!scratch_credit.borrower && !scratch_credit.chunk.used);
    assert(vk_exec_draw_update_prepare(&first_scene, fill, &grant, &dependency) == VK_SUCCESS);
    assert(emit_update(&first_scene, fill, grant, 0) == shared);
    assert(scratch_credit.borrower == older.res && scratch_credit.chunk.used == 768);
    assert(!older.res->update_slabs && !spare_ready(&second_scene, next) && update_attempts == before_attempts);
    assert(reclaim_domain(RC_BUDGET_RAM, 768) == RC_BUDGET_RECLAIM_NONE);

    assert(vk_exec_draw_update_credit_plan(769, &queued, &peak, &drain) == VK_NOT_READY && !queued && drain);
    grant_mode = GRANT_PENDING;
    configured_dependency = (RcVmaDependency){.selected = 1, .memory_type = 1};
    assert(vk_exec_draw_update_prepare(&second_scene, next, &grant, &dependency) == VK_NOT_READY && !grant);
    assert(!newer.res->update_pending.owner && !newer.res->update_slabs && newer.res->fence == second_scene.fence);
    const RcBudgetReadiness now = {.records_available = 1};
    assert(vk_exec_draw_scratch_prefix_useful(&first_scene, fill->size, &dependency, &now));
    assert(!vk_exec_draw_scratch_prefix_useful(&second_scene, next->size, &dependency, &now));
    vk_exec_draw_update_prefix_completed(&first_scene, first_scene.fence);
    assert(!scratch_credit.borrower && !scratch_credit.chunk.used && scratch_credit.receipts == 2);
    assert(spare_ready(&first_scene, fill) && spare_ready(&second_scene, next) && !nspare_chunk && !lock_depth);
    assert(creates == before_create + 1 && destroys == before_destroy && memory->ticket.ledger);

    assert(vk_exec_draw_update_prepare(&first_scene, fill, &grant, &dependency) == VK_SUCCESS);
    assert(emit_update(&first_scene, fill, grant, 0) == shared && !spare_ready(&second_scene, next));
    assert(vk_exec_draw_update_prepare(&second_scene, next, &grant, &dependency) == VK_NOT_READY && !grant);
    issued_scene = first_scene.fence;
    handed_scene = first_scene.fence - 1;
    assert(!vk_exec_draw_scratch_prior_useful(&second_scene, next->size, &dependency, &now));
    handed_scene = first_scene.fence;
    assert(vk_exec_draw_scratch_prior_useful(&second_scene, next->size, &dependency, &now) == first_scene.fence);
    assert(!spare_ready(&second_scene, next) && memory->live && memory->map[0] == 0x81);

    // Service just the known older completion; the younger recording has no native handoff yet.
    assert(done < handed_scene && queue[qhead].fence == handed_scene);
    pending_dispose_queue(&queue, &qhead, &qn, handed_scene, UINT64_MAX, 0);
    assert(qn == 1 && queue[qhead].fence == second_scene.fence && scratch_credit.receipts == 1);
    assert(!scratch_credit.borrower && spare_ready(&second_scene, next) && !nspare_chunk);
    assert(!vk_exec_draw_scratch_prior_useful(&second_scene, next->size, &dependency, &now));
    assert(done < handed_scene && memory->ticket.ledger && memory->live);
    done = handed_scene;

    before_attempts = update_attempts;
    assert(vk_exec_draw_update_prepare(&second_scene, next, &grant, &dependency) == VK_SUCCESS);
    assert(grant && grant->protected_span && !grant->offset && update_attempts == before_attempts);
    assert(emit_update(&second_scene, next, grant, 0) == shared && scratch_credit.borrower == newer.res);
    advance(7, 8, 0);
    assert(qn == 1 && scratch_credit.borrower == newer.res && memory->live && !nspare_chunk);
    assert(update_ledger.retained.amount[RC_BUDGET_RAM] == memory->cost.amount[RC_BUDGET_RAM] + header);
    assert(update_ledger.retained.amount[RC_BUDGET_COMMIT] == memory->cost.amount[RC_BUDGET_COMMIT] + header);
    assert(reclaim_domain(RC_BUDGET_RAM, 768) == RC_BUDGET_RECLAIM_NONE);
    advance(8, 8, 0);
    assert(!qn && !scratch_credit.receipts && !scratch_credit.borrower && !scratch_credit.chunk.alloc);
    assert(!memory->live && !memory->ticket.ledger && creates == before_create + 1 && destroys == before_destroy + 1);

    grant_mode = GRANT_READY;
    assert(vk_exec_draw_update_credit_plan(96, &queued, &peak, &drain) == VK_SUCCESS && queued && !drain);
    materialize_credit(queued, &peak);
    VkBuffer cancelled = scratch_credit.chunk.buf;
    VkExec discarded = {.dc = &newer, .fence = 9};
    vk_exec_draw_update_credit_adopt(&discarded, &queued);
    assert(!queued && qn == 1 && scratch_credit.receipts == 1 && backing(cancelled)->live);
    advance(8, 8, 0);
    assert(!qn && !scratch_credit.receipts && !scratch_credit.borrower && !backing(cancelled)->live);
    vk_exec_draw_update_credit_cancel(NULL);
    assert(creates == before_create + 2 && destroys == before_destroy + 2 && update_materialized == update_retired);
    assert(!update_ledger.retained.amount[RC_BUDGET_RAM] && !update_ledger.reserved.amount[RC_BUDGET_RAM]);

    free(next);
    free(fill);
}

static void observe_draw(VkExec *ex, const p_draw *draw) {
    draw_trace = (DrawTrace){0};
    observed_scene = ex;
    observed_draw = draw;
}

static void assert_index_draw(uint32_t count, uint32_t instances, uint32_t base_instance) {
    assert(draw_trace.binds == 1 && draw_trace.indexed == 1 && !draw_trace.plain && draw_trace.writes == 1);
    assert(draw_trace.count == count && draw_trace.instances == instances && draw_trace.first_instance == base_instance);
    assert(!draw_trace.first && !draw_trace.vertex_offset);
}

static void check_generated_indices(void) {
    p_draw max_quads = {.prim = NVN_DRAW_PRIMITIVE_QUADS, .count = INT32_MAX};
    p_draw max_loop = {.prim = NVN_DRAW_PRIMITIVE_LINE_LOOP, .count = INT32_MAX};
    assert(nvn_generated_draw_index_bytes(&max_quads) == UINT64_C(12884901864));
    assert(nvn_generated_draw_index_bytes(&max_loop) == UINT64_C(8589934592));

    DrawCtx recorder = {0};
    VkExec ex = {.dc = &recorder, .fence = 10, .cb = (VkCommandBuffer)(uintptr_t)10};
    dc = &recorder;
    const p_draw plain[] = {
        {.prim = NVN_DRAW_PRIMITIVE_LINE_LOOP, .count = 0},
        {.prim = NVN_DRAW_PRIMITIVE_QUADS, .first = 37, .count = 3, .base_instance = 11, .instances = 2},
        {.prim = NVN_DRAW_PRIMITIVE_LINE_LOOP, .first = -1, .count = 1, .base_instance = -2},
    };
    VkDrawScratchUpdate *grant = NULL;
    RcVmaDependency dependency;
    unsigned before_attempts = update_attempts;
    for (unsigned i = 0; i < sizeof plain / sizeof plain[0]; i++) {
        const p_draw *draw = &plain[i];
        assert(vk_exec_draw_index_prepare(&ex, draw, &grant, &dependency) == VK_SUCCESS && !grant);
        ex.resource_op = ++ex.resource_next;
        observe_draw(&ex, draw);
        int before_recorded = ex.recorded;
        op_draw(&ex, draw);
        vk_exec_draw_index_finish(&ex, grant);
        assert(!draw_trace.binds && !draw_trace.indexed && !recorder.res);
        assert(ex.recorded == before_recorded + !!draw->count);
        if (!draw->count) {
            assert(!draw_trace.prepares && !draw_trace.plain && !draw_trace.writes);
        } else {
            assert(draw_trace.prepares == 1 && draw_trace.plain == 1 && draw_trace.writes == 1);
            assert(draw_trace.count == (uint32_t)draw->count && draw_trace.first == (uint32_t)draw->first);
            assert(draw_trace.first_instance == (uint32_t)draw->base_instance);
            assert(draw_trace.instances == (uint32_t)(draw->instances ? draw->instances : 1));
        }
    }
    assert(update_attempts == before_attempts && !qn);

    const p_draw quads = {
        .prim = NVN_DRAW_PRIMITIVE_QUADS, .first = 17, .count = 9, .base_instance = 5, .instances = 3,
    };
    configured_dependency = (RcVmaDependency){.selected = 1, .memory_type = 0};
    configured_dependency.peak.amount[RC_BUDGET_RAM] = CHUNK_BYTES;
    grant_mode = GRANT_PENDING;
    assert(vk_exec_draw_index_prepare(&ex, &quads, &grant, &dependency) == VK_NOT_READY && !grant);
    assert(!qn && !recorder.res && update_attempts == before_attempts + 1);

    // A descriptor retained outside the ordered lane must survive every private prefix cut.
    VkDescriptorBufferInfo retained = {.range = 16};
    grant_mode = GRANT_READY;
    uint8_t *generic = scratch(&ex, retained.range, &retained.buffer, &retained.offset, SCRATCH_REQUIRED);
    memset(generic, 0x66, (size_t)retained.range);

    grant_mode = GRANT_READY;
    assert(vk_exec_draw_index_prepare(&ex, &quads, &grant, &dependency) == VK_SUCCESS);
    assert(grant && grant->opcode == OP_DRAW && grant->payload == &quads && grant->bytes == 48);
    assert(grant->aligned == 256 && !grant->offset && grant->fresh && !recorder.res->update_slabs);
    VkBuffer index_buffer = grant->slab->chunk.buf;
    assert(index_buffer != retained.buffer);
    ex.resource_op = ++ex.resource_next;
    int before_recorded = ex.recorded;
    observe_draw(&ex, &quads);
    deferred_draws = 2;
    op_draw(&ex, &quads);
    vk_exec_draw_new_cb();
    op_draw(&ex, &quads);
    assert(draw_trace.prepares == 2 && !draw_trace.binds && !draw_trace.indexed && !draw_trace.writes);
    assert(ex.recorded == before_recorded && grant->payload == &quads && !grant->slab->chunk.used);
    vk_exec_draw_index_finish(&ex, grant);
    assert(!recorder.res->update_pending.owner && !recorder.res->update_slabs && nspare_chunk == 1);

    before_attempts = update_attempts;
    assert(vk_exec_draw_index_prepare(&ex, &quads, &grant, &dependency) == VK_SUCCESS && grant);
    assert(grant->slab->chunk.buf == index_buffer && update_attempts == before_attempts);
    ex.resource_op = ++ex.resource_next;
    observe_draw(&ex, &quads);
    deferred_draws = 1;
    op_draw(&ex, &quads);
    assert(!draw_trace.binds && !draw_trace.writes && grant->payload == &quads);
    vk_exec_draw_new_cb();
    op_draw(&ex, &quads);
    assert(draw_trace.prepares == 2 && draw_trace.buffer == index_buffer && !draw_trace.offset);
    assert_index_draw(12, 3, 5);
    assert(ex.recorded == before_recorded + 2 && recorder.c0_binst == 5);
    const uint32_t quad_indices[] = {17, 18, 19, 17, 19, 20, 21, 22, 23, 21, 23, 24};
    assert(!memcmp(backing(index_buffer)->map, quad_indices, sizeof quad_indices));
    vk_exec_draw_index_finish(&ex, grant);
    assert(!recorder.res->update_pending.owner && recorder.res->update_slabs->chunk.used == 256);

    const p_draw loop = {.prim = NVN_DRAW_PRIMITIVE_LINE_LOOP, .first = -2, .count = 4, .base_instance = -3};
    assert(vk_exec_draw_index_prepare(&ex, &loop, &grant, &dependency) == VK_SUCCESS && grant);
    assert(!grant->fresh && grant->offset == 256 && grant->bytes == 20 && grant->aligned == 256);
    ex.resource_op = ++ex.resource_next;
    observe_draw(&ex, &loop);
    op_draw(&ex, &loop);
    assert_index_draw(5, 1, (uint32_t)-3);
    assert(draw_trace.offset == 256 && draw_trace.buffer == index_buffer);
    const uint32_t loop_indices[] = {UINT32_MAX - 1, UINT32_MAX, 0, 1, UINT32_MAX - 1};
    assert(!memcmp(backing(index_buffer)->map + 256, loop_indices, sizeof loop_indices));
    assert(!memcmp(backing(index_buffer)->map, quad_indices, sizeof quad_indices));
    vk_exec_draw_index_finish(&ex, grant);

    const p_draw full_loop = {
        .prim = NVN_DRAW_PRIMITIVE_LINE_LOOP, .first = 19, .count = CHUNK_BYTES / sizeof(uint32_t) - 1,
        .base_instance = 7, .instances = 2,
    };
    assert(nvn_generated_draw_index_bytes(&full_loop) == CHUNK_BYTES);
    grant_mode = GRANT_PENDING;
    assert(vk_exec_draw_index_prepare(&ex, &full_loop, &grant, &dependency) == VK_NOT_READY && !grant);
    const RcBudgetReadiness now = {.records_available = 1, .usable_mask = 1u << RC_BUDGET_RAM};
    assert(vk_exec_draw_scratch_prefix_useful(&ex, CHUNK_BYTES, &dependency, &now));
    VkDrawScratchWaitContext wait = {.ex = &ex, .bytes = CHUNK_BYTES};
    assert(!vk_exec_draw_scratch_spare_ready(&wait) && !nspare_chunk);
    assert(reclaim_domain(RC_BUDGET_RAM, CHUNK_BYTES) == RC_BUDGET_RECLAIM_NONE);

    // The known-idle ordered prefix returns its generated slab, while the generic descriptor is still live.
    vk_exec_draw_update_prefix_completed(&ex, ex.fence);
    assert(vk_exec_draw_scratch_spare_ready(&wait) && !recorder.res->update_slabs);
    assert(qn == 1 && nspare_chunk == 1 && backing(retained.buffer)->live && recorder.res->graphics_slabs && !recorder.res->graphics_slabs->next);
    for (size_t i = 0; i < retained.range; i++) assert(generic[i] == 0x66);
    ex.cb = (VkCommandBuffer)(uintptr_t)11;
    vk_exec_draw_new_cb();
    before_attempts = update_attempts;
    assert(vk_exec_draw_index_prepare(&ex, &full_loop, &grant, &dependency) == VK_SUCCESS && grant);
    assert(grant->slab->chunk.buf == index_buffer && update_attempts == before_attempts && !nspare_chunk);
    ex.resource_op = ++ex.resource_next;
    observe_draw(&ex, &full_loop);
    op_draw(&ex, &full_loop);
    assert_index_draw((uint32_t)full_loop.count + 1, 2, 7);
    assert(draw_trace.buffer == index_buffer && !draw_trace.offset);
    const uint32_t *full = (const uint32_t *)backing(index_buffer)->map;
    assert(full[0] == 19 && full[1] == 20 && full[full_loop.count - 1] == 19 + (uint32_t)full_loop.count - 1);
    assert(full[full_loop.count] == 19);
    vk_exec_draw_index_finish(&ex, grant);
    for (size_t i = 0; i < retained.range; i++) assert(generic[i] == 0x66);

    advance(ex.fence, ex.fence, 0);
    assert(!qn && nspare_chunk == 2);
    assert(reclaim_domain(RC_BUDGET_RAM, UINT64_MAX) == RC_BUDGET_RECLAIM_DESTROYED);
    assert(!nspare_chunk && creates == destroys && update_materialized == update_retired);
    observed_scene = NULL;
    observed_draw = NULL;
    grant_mode = GRANT_READY;
}

static void check_generated_group_boundary(void) {
    typedef struct { op_hdr header; p_draw draw; uint32_t pad; } DrawRecord;
    struct { DrawRecord draws[7]; op_hdr end; } recording = {.end = {OP_END, sizeof(op_hdr)}};
    for (unsigned i = 0; i < 7; i++) {
        recording.draws[i].header = (op_hdr){OP_DRAW, sizeof(DrawRecord)};
        recording.draws[i].draw = (p_draw){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .count = 3};
    }
    recording.draws[0].draw.prim = NVN_DRAW_PRIMITIVE_QUADS; // Three vertices use the native nonindexed draw.
    recording.draws[1].draw = (p_draw){.prim = NVN_DRAW_PRIMITIVE_LINE_LOOP, .count = 1};
    const op_hdr *start = &recording.draws[0].header, *ordered = &recording.draws[3].header;
    const op_hdr *tail = &recording.draws[4].header;
    NvnRecordSpan spans[] = {
        {.end = 3 * sizeof(DrawRecord), .ops = 3, .draws = 3, .kind = NVN_PLAN_RUN},
        {.begin = 3 * sizeof(DrawRecord), .end = 4 * sizeof(DrawRecord), .ops = 1, .draws = 1},
        {.begin = 4 * sizeof(DrawRecord), .end = 7 * sizeof(DrawRecord), .ops = 3, .draws = 3, .kind = NVN_PLAN_RUN},
        {.begin = 7 * sizeof(DrawRecord), .end = sizeof recording, .ops = 1, .kind = NVN_PLAN_SERIAL},
    };
    NvnRecordPlan plan = {.commands = (const uint8_t *)&recording, .bytes = sizeof recording,
        .spans = spans, .nspans = sizeof spans / sizeof spans[0]};
    VkExec ex = {0};
    const int primitive[] = {NVN_DRAW_PRIMITIVE_QUADS, NVN_DRAW_PRIMITIVE_LINE_LOOP};
    for (unsigned i = 0; i < sizeof primitive / sizeof primitive[0]; i++) {
        recording.draws[3].draw = (p_draw){.prim = primitive[i], .count = 4};
        spans[1].kind = nvn_record_span_class(ordered);
        assert(spans[1].kind == NVN_PLAN_GENERATED_INDEX);
        const op_hdr *from[] = {start, ordered, tail};
        const op_hdr *stop[] = {ordered, ordered, &recording.end};
        const uint32_t expected[] = {3, 0, 3};
        const int reasons[] = {GE_INDEX, GE_INDEX, GE_RUN};
        for (unsigned j = 0; j < sizeof expected / sizeof expected[0]; j++) {
            const op_hdr *raw_end = NULL, *planned_end = NULL;
            int raw_why, planned_why;
            uint32_t raw_op, planned_op, planned_draws;
            assert(group_peek(&ex, from[j], &raw_end, &raw_why, &raw_op) == expected[j]);
            assert(group_peek_plan(&ex, &plan, from[j], &planned_end, &planned_why, &planned_op, &planned_draws));
            assert(raw_end == stop[j] && planned_end == raw_end && planned_draws == expected[j]);
            assert(raw_why == reasons[j] && planned_why == raw_why && planned_op == raw_op);
        }
    }
}

static void check_compatibility_pressure(void) {
    DrawCtx old_ctx = {0}, new_ctx = {0};
    VkExec old = {.dc = &old_ctx, .fence = 89};
    VkExec next = {.dc = &new_ctx, .fence = 90};
    VkBuffer held, out = (VkBuffer)(uintptr_t)123;
    VkDeviceSize offset;
    uint8_t *original = scratch(&old, CHUNK_BYTES, &held, &offset, SCRATCH_REQUIRED_PRIOR);
    assert(original && !offset && backing(held)->ticket.ledger && !nspare_chunk);
    original[0] = 0x2a;
    original[CHUNK_BYTES - 1] = 0x73;
    unsigned attempts = update_attempts, allocations = creates;
    grant_mode = GRANT_PENDING;
    configured_dependency = (RcVmaDependency){.selected = 1, .memory_type = 0};
    configured_dependency.peak.amount[RC_BUDGET_RAM] = CHUNK_BYTES;
    configured_readiness = (RcBudgetReadiness){.records_available = 1, .usable_mask = 1u << RC_BUDGET_RAM};
    handed_scene = old.fence - 1;
    assert(!scratch(&next, 17, &out, &offset, SCRATCH_REQUIRED_PRIOR));
    assert(update_attempts == attempts + 1 && !compat_waited && compat_refused == 1 && creates == allocations);
    assert(!new_ctx.res && out == (VkBuffer)(uintptr_t)123 && original[0] == 0x2a);
    handed_scene = old.fence;
    assert(!scratch(&old, 17, &out, &offset, SCRATCH_REQUIRED_PRIOR));
    assert(!compat_waited && compat_refused == 2);
    assert(!scratch(&next, 17, &out, &offset, SCRATCH_OPTIONAL));
    assert(compat_optional == 1 && !compat_waited && compat_refused == 2);
    assert(!scratch(&next, 17, &out, &offset, SCRATCH_REQUIRED));
    assert(!compat_waited && compat_refused == 3 && original[CHUNK_BYTES - 1] == 0x73);
    uint8_t *reused = scratch(&next, 17, &out, &offset, SCRATCH_REQUIRED_PRIOR);
    assert(reused == original && out == held && !offset && compat_waited == 1 && compat_refused == 3);
    assert(update_attempts == attempts + 5 && creates == allocations && qn == 1 && new_ctx.res->graphics_slabs);
    assert(new_ctx.res->graphics_slabs->chunk.used == 256 && backing(held)->ticket.ledger);
    reused[0] = 0x44;
    advance(next.fence, next.fence, 0);
    assert(!qn && nspare_chunk == 1 && backing(held)->live && reused[0] == 0x44);
    assert(reclaim_domain(RC_BUDGET_RAM, CHUNK_BYTES) == RC_BUDGET_RECLAIM_DESTROYED);
    assert(!backing(held)->live && !nspare_chunk);

    grant_mode = GRANT_UNMAPPED;
    DrawCtx cold = {0};
    VkExec stopped = {.dc = &cold, .fence = 91};
    assert(!scratch(&stopped, 256, &out, &offset, SCRATCH_REQUIRED_PRIOR));
    assert(!cold.res && compat_waited == 1 && compat_refused == 4 && creates == destroys);
    advance(stopped.fence, stopped.fence, 0);
    assert(!qn && !nspare_chunk);
    grant_mode = GRANT_READY;
}

int main(void) {
    enum { SPLIT_SCENE = 100, UNCERTAIN_SCENE, UNSENT_SCENE, LARGE_SCENE, FINAL_SCENE };
    set_context();
    check_update_preparation();
    check_update_prefix();
    check_update_prior();
    check_update_credit();
    check_generated_indices();
    check_generated_group_boundary();
    assert(creates == destroys && !qn && !nspare_chunk);
    check_compatibility_pressure();
    memset(native, 0, sizeof native);
    creates = destroys = progress_calls = 0;
    DrawCtx main_ctx = {0}, worker_ctx = {0}, other_ctx = {0};
    VkExec main = {.dc = &main_ctx, .fence = SPLIT_SCENE};
    VkExec worker = {.dc = &worker_ctx, .fence = SPLIT_SCENE};
    VkExec other = {.dc = &other_ctx, .fence = SPLIT_SCENE};
    VkBuffer first, split, parallel, device_local;
    VkDeviceSize offset;

    uint8_t *first_map = scratch(&main, 1, &first, &offset, SCRATCH_REQUIRED);
    assert(!offset && first_map == backing(first)->map && creates == 1 && qn == 1);
    first_map[0] = 0x51;
    CbRes *scene_owner = main_ctx.res;
    dc = &main_ctx;
    vk_exec_draw_new_cb(); // The split changes the recording buffer, while the scene's reads continue.
    uint8_t *split_map = scratch(&main, 257, &split, &offset, SCRATCH_REQUIRED);
    assert(split == first && offset == 256 && split_map == first_map + 256);
    assert(main_ctx.res == scene_owner && main_ctx.res->graphics_slabs->chunk.used == 768 && creates == 1 && qn == 1);
    assert(first_map[0] == 0x51);
    split_map[0] = 0x62;

    assert(scratch(&worker, 256, &parallel, &offset, SCRATCH_REQUIRED) && !offset && parallel != first);
    assert(worker_ctx.res != scene_owner && worker_ctx.res->fence == SPLIT_SCENE);
    native_type = 1;
    assert(scratch(&other, 256, &device_local, &offset, SCRATCH_REQUIRED) && !offset && device_local != first && device_local != parallel);
    assert(creates == 3 && qn == 3 && !nspare_chunk);
    assert(reclaim_domain(RC_BUDGET_RAM, CHUNK_BYTES) == RC_BUDGET_RECLAIM_NONE);

    advance(SPLIT_SCENE - 1, SPLIT_SCENE, 0);
    Chunk unavailable = {0};
    assert(qn == 3 && !nspare_chunk && !progress_calls && !destroys);
    assert(!scratch_take_spare(CHUNK_BYTES, &unavailable));
    assert(first_map[0] == 0x51 && split_map[0] == 0x62);
    advance(SPLIT_SCENE, SPLIT_SCENE, 0);
    assert(!qn && nspare_chunk == 3 && progress_calls == 3 && !destroys);

    // Staging pressure cannot select draw chunks. Actual VMA padding lets one host chunk cover this deficit.
    assert(reclaim_domain(RC_BUDGET_STAGING, CHUNK_BYTES) == RC_BUDGET_RECLAIM_NONE);
    assert(reclaim_domain(RC_BUDGET_RAM, CHUNK_BYTES + NATIVE_PADDING) == RC_BUDGET_RECLAIM_DESTROYED);
    assert(nspare_chunk == 2 && live_on_heap(0) == 1 && live_on_heap(1) == 1 && destroys == 1);
    assert(reclaim_domain(RC_BUDGET_GPU_FIRST + 2, CHUNK_BYTES) == RC_BUDGET_RECLAIM_NONE);

    main.fence = UNCERTAIN_SCENE;
    worker.fence = UNSENT_SCENE;
    VkBuffer uncertain, unsent;
    uint8_t *uncertain_map = scratch(&main, 256, &uncertain, &offset, SCRATCH_REQUIRED);
    assert(uncertain_map && !offset && backing(uncertain)->live);
    uncertain_map[0] = 0x73;
    assert(scratch(&worker, 256, &unsent, &offset, SCRATCH_REQUIRED) && !offset && unsent != uncertain);
    assert(creates == 3 && qn == 2 && !nspare_chunk);

    // The shared disposal rule releases only the never-handed scene on undrained loss.
    advance(SPLIT_SCENE, UNCERTAIN_SCENE, 0);
    assert(qn == 1 && queue[qhead].fence == UNCERTAIN_SCENE && nspare_chunk == 1 && progress_calls == 4);
    unsigned unsent_heap = RC_BUDGET_GPU_FIRST + backing(unsent)->type;
    unsigned uncertain_heap = RC_BUDGET_GPU_FIRST + backing(uncertain)->type;
    assert(reclaim_domain(uncertain_heap, CHUNK_BYTES) == RC_BUDGET_RECLAIM_NONE);
    assert(reclaim_domain(unsent_heap, CHUNK_BYTES) == RC_BUDGET_RECLAIM_DESTROYED);
    assert(!backing(unsent)->live && backing(uncertain)->live && uncertain_map[0] == 0x73);
    assert(!scratch_take_spare(CHUNK_BYTES, &unavailable));

    // A drained device supplies physical proof even though the uncertain scene never signaled its final fence.
    advance(SPLIT_SCENE, UNCERTAIN_SCENE, 1);
    assert(!qn && nspare_chunk == 1 && progress_calls == 5);
    advance(SPLIT_SCENE, UNCERTAIN_SCENE, 1);
    assert(nspare_chunk == 1 && progress_calls == 5);

    other.fence = LARGE_SCENE;
    VkBuffer large;
    assert(scratch(&other, CHUNK_BYTES + 1, &large, &offset, SCRATCH_REQUIRED) && !offset && large != uncertain);
    assert(creates == 4 && other_ctx.res->graphics_slabs->chunk.size == CHUNK_BYTES + 256);
    assert(nspare_chunk == 1 && reclaim_domain(uncertain_heap, CHUNK_BYTES) == RC_BUDGET_RECLAIM_DESTROYED);
    assert(backing(large)->live && qn == 1 && !nspare_chunk);
    advance(LARGE_SCENE, LARGE_SCENE, 0);
    assert(!backing(large)->live && !qn && !nspare_chunk && destroys == creates);


    native_type = 0;
    main.fence = FINAL_SCENE;
    VkBuffer final;
    assert(scratch(&main, 32, &final, &offset, SCRATCH_REQUIRED) && !offset);
    advance(FINAL_SCENE, FINAL_SCENE, 0);
    assert(!qn && nspare_chunk == 1 && backing(final)->live);
    vk_exec_draw_scratch_shutdown();
    assert(!nspare_chunk && !backing(final)->live && destroys == creates);

    context_ready = 0;
    unsigned reads_at_stop = context_reads;
    assert(reclaim_domain(RC_BUDGET_RAM, CHUNK_BYTES) == RC_BUDGET_RECLAIM_NONE);
    vk_exec_draw_scratch_shutdown();
    assert(context_reads == reads_at_stop && destroys == creates && !lock_depth);
    free(queue);
    free(rqueue);
    puts("mapped draw scratch: compatibility admission, exact prior reuse, updates and retirement passed");
    return 0;
}
'''

    cc = os.environ.get('CC') or shutil.which('clang')
    if not cc:
        raise SystemExit('Clang is required (set CC)')
    toolchain = Path(shutil.which(cc) or cc).resolve()
    vma_include = Path(os.environ.get('VMA_INCLUDE', toolchain.parent.parent / 'include' / 'vma'))
    env = dict(os.environ, PATH=str(toolchain.parent) + os.pathsep + os.environ.get('PATH', ''))
    with tempfile.TemporaryDirectory(prefix='mapped-draw-scratch-') as directory:
        source = Path(directory) / 'check.c'
        executable = Path(directory) / ('check.exe' if os.name == 'nt' else 'check')
        source.write_text(fixture)
        subprocess.run([str(toolchain), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror',
                        '-fsanitize=undefined', '-fsanitize-trap=undefined',
                        '-I' + str(ROOT / 'runtime'), '-I' + str(vma_include), str(source), '-o', str(executable)],
                       env=env, check=True)
        return subprocess.run([str(executable)], env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())

"""Check the production ordered UBO planner, scratch initialization and descriptor.

The fixture models a retained native allocation and Vulkan activity in host
memory, and checks the real source, scratch ownership, and effect proof paths.
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    match = re.search(r"^(?:static )?[A-Za-z_][\w \t*]*\b" + re.escape(name) + r"\([^;{}]*\)\s*\{", source, re.M)
    if match is None:
        raise ValueError(f"Missing production function {name}")
    start = match.end()
    depth = 1
    stop = start
    while depth:
        depth += (source[stop] == "{") - (source[stop] == "}")
        stop += 1
    return source[match.start():stop] + "\n"


PREFIX = r'''
#include <assert.h>
#include <stdio.h>
#include <string.h>
#define RC_REGISTRY_SINGLE_THREADED 1
#define rc_registry_runtime registry_unused_runtime
#include "resource_core_registry.c"
#undef rc_registry_runtime
#include "resource_core_budget.h"

#define OK(result) assert((result) == RC_REGISTRY_OK)
#define PERMISSION_LOCAL_RANGES 4
enum { BACKING = 3, ALIGN = 256, OUT_OFFSET = 256, MAX_BYTES = 64 * 1024, MAX_UPDATE = 64 };
typedef struct {
    uint64_t address, offset, size, update_offset, update_size;
    uint64_t scene, update_op, read_op;
} FixtureRange;
static FixtureRange input;
enum { VK_MCLS_DEVICE, VK_MCLS_SCRATCH, VK_MUSE_GPU_WRITE, VK_MUSE_UBO, VK_MUSE_CPU_READ, VK_MUSE_SSBO };
enum { GFX_RES_UBO, VK_PIPE_SUPPORT, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, GFX_RES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
enum { GFX_STAGE_VS, GFX_STAGE_FS, GFX_STAGE_TS, GFX_STAGE_MS, GFX_STAGE_CS };
enum { NVN_STAGE_VS, NVN_STAGE_FS, NVN_STAGE_TASK, NVN_STAGE_MESH, NVN_STAGE_CS };
enum { VK_C0_BYTES = 2048, SUPPORT_BYTES = 16, MAX_BINDINGS = 64, MAX_DESC = 1024 };
enum { OP_DRAW, OP_DRAW_IDX, OP_DRAW_IDX_IND, OP_DRAW_MESH, OP_DRAW_MESH_IND, OP_UPDATE_UB,
       NVN_INDEX_TYPE_UNSIGNED_BYTE };
typedef enum { VK_PIPELINE_BIND_POINT_GRAPHICS, VK_PIPELINE_BIND_POINT_COMPUTE } VkPipelineBindPoint;
typedef enum {
    VK_EXEC_HOST_INPUT_READY, VK_EXEC_HOST_INPUT_STOPPED, VK_EXEC_HOST_INPUT_UNSUPPORTED, VK_EXEC_HOST_INPUT_CAPACITY
} VkExecHostInputResult;
typedef enum {
    VK_SUCCESS, VK_NOT_READY, VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY,
    VK_ERROR_INITIALIZATION_FAILED, VK_ERROR_DEVICE_LOST, VK_ERROR_TOO_MANY_OBJECTS, VK_ERROR_FEATURE_NOT_PRESENT
} VkResult;
typedef struct { int selected; RcAdmissionVector peak, reuse; RcBudgetReadiness before; } RcVmaDependency;
typedef void *VmaAllocation;
typedef struct { int slot, ptr, flags; } GfxShaderRes;
typedef int PlatMutex;
#define PLAT_MUTEX_INIT 0
enum { VK_SHADER_STAGE_VERTEX_BIT = 1, VK_SHADER_STAGE_FRAGMENT_BIT = 16 };
typedef struct { unsigned stage, count; int kind, type; const GfxShaderRes *res; unsigned set, binding, stages; } VkPipeBinding;
typedef struct {
    uint32_t n_bindings, n_sets;
    VkPipelineBindPoint bind_point;
    const VkPipeBinding *bindings;
} VkPipeInfo;
typedef void GfxShader;
typedef struct { uint64_t serial; } prog_t;
typedef struct { const GfxShader *vs, *fs, *ts; int prim_restart; } VkPipeState;
typedef struct { int base_instance; } p_draw;
typedef struct { int base_instance, count, itype; } p_draw_idx;
typedef struct {
    uint64_t minUniformBufferOffsetAlignment, maxUniformBufferRange, maxStorageBufferRange, minStorageBufferOffsetAlignment;
} VkPhysicalDeviceLimits;
typedef struct { VkPhysicalDeviceLimits limits; } VkPhysicalDeviceProperties;
typedef struct { int phys; VkPhysicalDeviceProperties props; } VkCtx;
typedef uint64_t VkBuffer, VkDeviceSize;
enum { SNAPSHOT = 1, NATIVE, OTHER_MIRROR, STAGING, MERGED };
typedef struct { VkBuffer buffer; uint64_t offset, range; } VkDescriptorBufferInfo;
typedef struct { uint64_t capture, addr, size, align; uint32_t region, reserved; } VkUsnapLease;
typedef enum {
    VK_USNAP_RESOLVE_READY, VK_USNAP_RESOLVE_NATIVE_ONLY, VK_USNAP_RESOLVE_ORDERED_PRODUCER,
    VK_USNAP_RESOLVE_SOURCE_ADVANCED, VK_USNAP_RESOLVE_MAPPING_STALE, VK_USNAP_RESOLVE_NO_CAPACITY,
    VK_USNAP_RESOLVE_ALIGNMENT, VK_USNAP_RESOLVE_UNAVAILABLE
} VkUsnapResolveResult;
typedef enum { VK_EXEC_UNIFORM_UPDATE_NONE, VK_EXEC_UNIFORM_UPDATE_READY, VK_EXEC_UNIFORM_UPDATE_STALE }
    VkExecUniformUpdateRead;

typedef struct { RcInterval interval; } MemBackingCapture;
typedef struct { VkBuffer buf; uint8_t *cpu, *map; int cpu_current; } Arena;
typedef struct {
    uint64_t lo, hi, off;
    MemBackingCapture *mapping;
    void *import;
    VkUsnapResolveResult unavailable;
} Reg;
static struct {
    uint64_t capture, hits, gen;
    int expected, overlapping, on, nreg;
    Reg *reg;
    Arena *arena;
    struct { RcRegistryGpuCapture read; } producer;
    RcSourceRange *permission_ranges;
    uint32_t permission_capacity;
} U;
static int permission_lock, spare_lock;
static void plat_mutex_lock(int *lock) { assert(lock && !(*lock)++); }
static void plat_mutex_unlock(int *lock) { assert(lock && (*lock)-- == 1); }

typedef struct { VkBuffer buffer; uint64_t offset, avail; void *alloc; int cls; } VkMemSpan;
typedef struct { uint64_t addr, size; } p_bind_buf;
typedef struct DrawCtx DrawCtx;
typedef struct VkExecGraphicsScratch VkExecGraphicsScratch;
typedef struct VkExecBufferScratchReceipt VkExecBufferScratchReceipt;
typedef struct {
    uint64_t cb, fence, resource_op, resource_next;
    int in_pass, recorded;
    const prog_t *prog;
    DrawCtx *dc;
    p_bind_buf ubo[8][18], ssbo[8][16];
} VkExec;
typedef struct { uint64_t dst, size; uint8_t data[MAX_UPDATE]; } p_update_ub;
typedef struct { VkMemSpan span; uint64_t mapping, address; } VkMemSpanCapture;
typedef struct { uint64_t address, size; RcRegistryPosition output; } VkMemBufferInlineProof;
typedef struct { VkMemSpan span; VkMemSpanCapture *hold; RcInterval *writes; uint32_t count; } VkExecUpdateDestination;
typedef struct { uint64_t srcOffset, dstOffset, size; } VkBufferCopy;
typedef struct { uint64_t fence; void (*fn)(void *); void (*cancel)(void *, int); void *arg; } Pending;

static RcRegistry *registry;
static Arena arena = {.buf = SNAPSHOT};
static Reg region;
static MemBackingCapture mapping;
static VkExec recorder;
static uint64_t live_mapping;
static VkBuffer selected, written_buffer;
static RcRegistryWrite update_ticket;
static RcRegistryBufferSpan update_native;
static unsigned native_holds, copies, barriers, notes, commits, gpu_writes;
static unsigned published, retired, completed, canceled, uncertain;
static unsigned budget_refusals, scratch_creates, scratch_releases, refused, merge_copies;
static int recording_failed, capacity_refused, remap_during_copy;
static uint8_t guest[MAX_BYTES];
static uint8_t buffers[MERGED + 1][OUT_OFFSET + MAX_BYTES + ALIGN];

static uint8_t original_byte(uint64_t offset) { return (uint8_t)(0x21 + offset % 93); }
static struct { unsigned ub_updates, ub_pass_breaks; } vk_exec_frame;

static RcRegistry *rc_registry_runtime(void) { return registry; }
static RcRegistryPosition vk_exec_resource_position(void) {
    return (RcRegistryPosition){recorder.fence, recorder.resource_op * 2};
}
static _Noreturn void rt_fatal(const char *message) { fputs(message, stderr); abort(); }
static RcRegistryResult vk_mem_canonical_source_qualify_expected(const MemBackingCapture *m, const void *import) {
    assert(m == &mapping && import == &region);
    if (remap_during_copy && buffers[MERGED][OUT_OFFSET] != 0xe9) {
        remap_during_copy = 0;
        live_mapping++;
    }
    return m->interval.mapping == live_mapping ? RC_REGISTRY_OK : RC_REGISTRY_STALE;
}
static int mem_backing_current_range(const MemBackingCapture *m, uint64_t address, uint64_t size) {
    assert(m == &mapping && address >= input.address && address - input.address + size <= m->interval.range.size);
    return m->interval.mapping == live_mapping;
}
static const RcInterval *mem_backing_intervals(const MemBackingCapture *m, uint32_t *count) {
    assert(m == &mapping);
    *count = 1;
    return &m->interval;
}
static int nvn_gpu_only(uint64_t addr) { (void)addr; return 0; }
static int vk_mem_lookup(uint64_t addr, uint64_t size, VkMemSpan *span) {
    (void)addr; (void)size; (void)span;
    abort(); // Every tested address is covered by its captured region.
}

static int vk_mem_span_capture_current(const VkMemSpanCapture *, const VkMemSpan *);
static VkMemSpanCapture *vk_mem_span_capture_retain(const VkMemSpanCapture *source) {
    VkMemSpanCapture *hold = malloc(sizeof *hold);
    assert(hold && source);
    *hold = *source;
    native_holds++;
    return hold;
}
static int vk_mem_span_record_input(VkMemSpanCapture *hold, const VkMemSpan *span) {
    return vk_mem_span_capture_current(hold, span);
}
#define VK_MEM_BOUND(span) do { assert((span).buffer); } while (0)

static uint32_t vk_exec_upd_top(uint32_t n) { return n; }
static RcRegistryResult vk_mem_span_native_input(uint64_t addr, uint64_t size, int clamp,
                                                 VkMemSpan *span, VkMemSpanCapture **hold) {
    assert(!clamp && !*hold && addr >= input.address && addr - input.address + size <= mapping.interval.range.size);
    *span = (VkMemSpan){selected, input.offset + addr - input.address, size, NULL, VK_MCLS_DEVICE};
    *hold = malloc(sizeof **hold);
    assert(*hold);
    **hold = (VkMemSpanCapture){*span, live_mapping, addr};
    native_holds++;
    return RC_REGISTRY_OK;
}
static int vk_mem_span_capture_current(const VkMemSpanCapture *hold, const VkMemSpan *span) {
    return hold && hold->mapping == live_mapping && selected == span->buffer &&
        hold->span.buffer == span->buffer && hold->span.offset == span->offset && hold->span.avail == span->avail;
}
static void vk_mem_span_capture_release(VkMemSpanCapture **hold) {
    if (!*hold) return;
    assert(native_holds);
    native_holds--;
    free(*hold);
    *hold = NULL;
}
static uint32_t vk_mem_span_input_intervals(const VkMemSpanCapture *hold, uint64_t addr, uint64_t size,
                                            RcInterval **out) {
    assert(!*out && addr == hold->address && size == hold->span.avail);
    *out = malloc(sizeof **out);
    assert(*out);
    **out = (RcInterval){addr, hold->mapping, {BACKING, input.offset + addr - input.address, size}};
    return 1;
}
static int vk_mem_span_input_cpu_visible(const VkMemSpanCapture *hold) { assert(hold); return 0; }
static int vk_mem_cpu_visible_span(const VkMemSpan *span, uint64_t addr, uint64_t size) {
    assert(span && addr && size);
    return 0;
}
static RcRegistryResult vk_usnap_buffer_fragments(const VkMemSpanCapture *hold, const VkMemSpan *span,
    uint64_t address, uint64_t size, RcRegistryPosition read, const VkMemBufferInlineProof *proofs,
    uint32_t proof_count, RcRegistryBufferFragment *out, uint32_t capacity, uint32_t *count) {
    assert(address && size && out && capacity && count && !rc_registry_position_compare(read, vk_exec_resource_position()));
    *count = 0;
    if (!vk_mem_span_capture_current(hold, span)) return RC_REGISTRY_STALE;
    for (uint32_t i = 0; i < proof_count; ++i) {
        assert(proofs[i].address >= address && proofs[i].size &&
               proofs[i].address - address + proofs[i].size <= size &&
               rc_registry_position_compare(proofs[i].output, read) < 0);
    }
    RcRegistryBufferSpan demand = {{address, hold->mapping,
        {BACKING, input.offset + address - input.address, size}}, {span->buffer, live_mapping, span->offset}};
    return rc_registry_gpu_capture_buffer_fragments(registry, U.producer.read, &demand, 1, read, out, capacity, count);
}
static int vk_mem_span_buffer_location(const VkMemSpanCapture *hold, const VkMemSpan *span,
                                       RcRegistryBufferLocation *location) {
    if (!vk_mem_span_capture_current(hold, span)) return 0;
    *location = (RcRegistryBufferLocation){span->buffer, live_mapping, span->offset};
    return 1;
}
static void resource_effects_buffer_prepare_native(const RcInterval *ranges, uint32_t count, uint64_t address,
                                                    RcRegistryBufferLocation location) {
    assert(update_ticket.id && !update_native.native.allocation && count == 1 && ranges->address == address &&
           location.allocation == selected && location.offset == ranges->range.offset);
    update_native = (RcRegistryBufferSpan){*ranges, location};
}
static void vk_exec_record_native_input_provider(VkExec *ex, VkMemSpanCapture **hold) {
    assert(ex == &recorder && *hold);
    // The test-local native recorder releases this transient capture. Its later
    // descriptor hold is separately exercised at the completion boundary.
    vk_mem_span_capture_release(hold);
}
static void vk_exec_barrier(VkExec *ex) { assert(ex == &recorder); barriers++; ex->in_pass = 0; }
#define VK_MEM_PICK(span) do { (void)(span); abort(); } while (0)
static int note_written(VkExec *ex, uint64_t addr, uint64_t size, const VkMemSpan *span) {
    (void)ex; (void)addr; (void)size; (void)span;
    abort(); // No fallback destination.
}
static void note_buffer_written(VkExec *ex, uint64_t addr, uint64_t size, const VkMemSpan *span,
                                int visible, const RcInterval *ranges, uint32_t count) {
    assert(ex == &recorder && !visible && count == 1 && ranges->address == addr &&
           ranges->range.offset == input.offset + addr - input.address && ranges->range.size == size &&
           span->buffer == selected && span->offset == input.offset + addr - input.address && barriers == 1 && !copies);
    notes++;
}
static void vkCmdCopyBuffer(uint64_t cb, VkBuffer src, VkBuffer dst, uint32_t count, const VkBufferCopy *copy) {
    if (dst == MERGED) {
        assert(cb == recorder.cb && src == NATIVE && count == 1);
        memcpy(buffers[dst] + copy->dstOffset,
               buffers[src] + OUT_OFFSET + copy->srcOffset - input.offset, copy->size);
        merge_copies++;
        return;
    }
    assert(cb == recorder.cb && src == STAGING && dst == selected && count == 1 && notes == 1 && !commits);
    assert(copy->srcOffset == ALIGN && copy->size == input.update_size);
    assert(copy->dstOffset == input.offset + input.update_offset);
    memcpy(buffers[dst] + OUT_OFFSET + (copy->dstOffset - input.offset), buffers[src] + copy->srcOffset, copy->size);
    written_buffer = dst;
    copies++;
}
static void commit_recorded_writes(VkExec *ex) {
    assert(ex == &recorder && copies == 1 && update_native.native.allocation == written_buffer);
    OK(rc_registry_buffer_write_record(registry, update_ticket, update_native,
        (RcRegistryPosition){ex->fence, ex->resource_op * 2 + 1}));
    update_ticket = (RcRegistryWrite){0};
    update_native = (RcRegistryBufferSpan){0};
    commits++;
}
static void vk_mem_gpu_wrote(const VkMemSpan *span, uint64_t size) {
    assert(span->buffer == written_buffer && span->avail == size && commits == 1);
    gpu_writes++;
}
static void vk_mem_count(int cls, int use, uint64_t size) {
    if (use == VK_MUSE_CPU_READ) {
        assert(cls == VK_MCLS_SCRATCH && (size == input.size - (input.size == 16 ? 4 : 0) ||
               (input.size > 96 && (size == 12 || size == input.size / 2)) ||
               (input.size == MAX_BYTES && input.read_op == 5 && (size == 16 || size == 4))));
        merge_copies++;
    } else assert(cls == VK_MCLS_DEVICE && use == VK_MUSE_GPU_WRITE && size && gpu_writes == 1);
}
'''


GRAPHICS_STUBS = r'''
struct CbRes {
    UpdateSlab *graphics_slabs;
    CbRes *graphics_prev, *graphics_next;
    uint64_t fence;
};
struct DrawCtx {
    CbRes *res;
    uint64_t res_fence;
    int pb_spoiled;
    VkUsnapLease *prep_capture;
};
static CbRes scratch_owner, *graphics_head, *graphics_tail;
static DrawCtx draw_context, *dc = &draw_context;
static VkExecGraphicsScratch *graphics_payload_current, *graphics_payload_bound;
static struct { uint64_t ubo_max, ubo_align, sbo_max, sbo_align; } LIM;
static const VkCtx device = {.phys = 5, .props.limits = {
    .minUniformBufferOffsetAlignment = ALIGN, .maxUniformBufferRange = MAX_BYTES, .maxStorageBufferRange = 8 * MAX_BYTES,
}};
static unsigned device_queries, readiness_queries;
static RcAdmission budget;
static RcAdmissionRecord charges[8];

static const VkCtx *vk_ctx(void) { return &device; }
static void vkGetPhysicalDeviceProperties(int phys, VkPhysicalDeviceProperties *out) {
    assert(phys == device.phys);
    *out = (VkPhysicalDeviceProperties){.limits.minStorageBufferOffsetAlignment = ALIGN / 2};
    device_queries++;
}
RcAdmissionStatus resource_core_budget_ticket_cost(const RcBudgetTicket *ticket, RcAdmissionVector *out) {
    assert(ticket->token.ledger == &budget && ticket->token.state == RC_ADMISSION_RETAINED);
    RcAdmissionRecord *record;
    RcAdmissionStatus status = rc_admission_lookup(&budget, ticket->token, ticket->producer, &record);
    if (status == RC_ADMISSION_OK) *out = record->amount;
    return status;
}
void resource_core_budget_readiness(int required, RcBudgetReadiness *out) {
    assert(required);
    *out = (RcBudgetReadiness){.usable_mask = 3, .records_available = budget.free_head != RC_ADMISSION_NO_SLOT};
    for (unsigned domain = 0; domain < budget.domain_count; domain++)
        assert(rc_admission_available(&budget, domain, &out->available.amount[domain]) == RC_ADMISSION_OK);
    readiness_queries++;
}
RcAdmissionStatus resource_core_budget_reserve(RcBudgetProducer producer, int required,
                                                      const RcAdmissionVector *peak, RcBudgetTicket *out) {
    assert(producer == RC_BUDGET_BUFFER && required && !out->token.ledger);
    assert(peak->amount[RC_BUDGET_RAM] == peak->amount[RC_BUDGET_COMMIT]);
    RcAdmissionStatus result = rc_admission_reserve(&budget, producer, peak, &out->token);
    if (result == RC_ADMISSION_OK) out->producer = producer;
    else budget_refusals++;
    return result;
}
RcAdmissionStatus resource_core_budget_materialize(RcBudgetTicket *ticket, const RcAdmissionVector *actual) {
    return rc_admission_materialize(&budget, ticket->token, ticket->producer, actual, &ticket->token);
}
RcAdmissionStatus resource_core_budget_cancel(RcBudgetTicket *ticket) {
    RcAdmissionStatus result = rc_admission_cancel(&budget, ticket->token, ticket->producer);
    if (result == RC_ADMISSION_OK) *ticket = (RcBudgetTicket){0};
    return result;
}
RcAdmissionStatus resource_core_budget_retire(RcBudgetTicket *ticket) {
    return rc_admission_retire(&budget, ticket->token, ticket->producer, &ticket->token);
}
RcAdmissionStatus resource_core_budget_destroyed(RcBudgetTicket *ticket) {
    RcAdmissionStatus result = rc_admission_destroyed(&budget, ticket->token, ticket->producer);
    if (result == RC_ADMISSION_OK) *ticket = (RcBudgetTicket){0};
    return result;
}

static CbRes *cb_res(VkExec *ex) {
    assert(ex == &recorder && ex->dc == dc && !dc->res);
    scratch_owner.fence = ex->fence;
    dc->res = &scratch_owner;
    dc->res_fence = ex->fence;
    return dc->res;
}
static VkResult update_slab_create(uint64_t bytes, int allow_growth, UpdateSlab **out,
                                   RcVmaDependency *dependency) {
    assert(allow_growth && !*out && dependency && bytes <= sizeof buffers[MERGED]);
    if (capacity_refused) {
        dependency->selected = 1;
        return VK_NOT_READY;
    }
    assert(!scratch_owner.graphics_slabs);
    memset(buffers[MERGED], 0xe9, sizeof buffers[MERGED]);
    *out = calloc(1, sizeof **out);
    assert(*out);
    (*out)->chunk = (Chunk){.buf = MERGED, .map = buffers[MERGED], .size = sizeof buffers[MERGED]};
    scratch_creates++;
    return VK_SUCCESS;
}
static int scratch_placement_fits(uint64_t bytes, const RcVmaDependency *dependency) {
    assert(bytes && dependency->selected);
    return 1;
}
static void update_release(UpdateSlab *slab) {
    assert(slab && !slab->next && slab->chunk.buf == MERGED);
    scratch_releases++;
    free(slab);
}
static int vk_exec_recording_failed(void) { return recording_failed; }
static int vk_device_is_lost(void) { return 0; }
static int vk_sync_stopping(void) { return 0; }
static int nvn_stage_of(unsigned stage) {
    assert(stage == GFX_STAGE_VS || stage == GFX_STAGE_CS);
    return stage == GFX_STAGE_VS ? NVN_STAGE_VS : NVN_STAGE_CS;
}
static void count_use(DrawCtx *draw, int cls, int use, uint64_t size) {
    assert(draw == dc && (cls == VK_MCLS_SCRATCH || cls == VK_MCLS_DEVICE) && (use == VK_MUSE_UBO || use == VK_MUSE_SSBO) && size);
}
static void vk_exec_required_graphics_unavailable(VkExec *ex, uint32_t op, const char *reason) {
    assert(ex == &recorder && !op && reason);
    recording_failed = 1;
    refused++;
}
static int plat_inc32(volatile int32_t *value) { return ++*value; }
static void rt_log(const char *format, ...) { assert(format); }

static void vk_c0_fill(uint8_t *map, int stage, const p_bind_buf *ssbo, const p_bind_buf *ubo,
                       const uint32_t *dimensions, uint32_t instance) {
    (void)map; (void)stage; (void)ssbo; (void)ubo; (void)dimensions; (void)instance;
    abort(); // The fixture has one cbuf3 binding and no C0 or support bindings.
}

typedef struct {
    VkExec *ex;
    int stage, direct_dispatch, c0_ready;
    const GfxShader *shader;
    VkExecBufferScratchReceipt *buffer_scratch;
    RcRegistryPosition read_position;
    VkDescriptorBufferInfo uniform[32];
    const uint8_t *uniform_bytes[32];
    uint32_t uniforms, leased;
    GraphicsUboMerge *merged[32];
    VkExecGraphicsScratch *merged_owner[32];
    VkUsnapLease lease[32];
    uint8_t c0[VK_C0_BYTES];
    uint32_t groups[3], base_instance;
} EffectInputCapture;

static int c0_on(void) { return 0; }
static VkExecBufferScratchReceipt *recording_scratch;
static VkExecBufferScratchReceipt *buffer_scratch_recording(const VkExec *ex) { assert(ex == &recorder); return recording_scratch; }
static void buffer_scratch_expose(VkExec *ex, VkExecBufferScratchReceipt *receipt) {
    assert(ex == &recorder && receipt == recording_scratch);
    receipt->exposed = 1;
}

static struct { uint64_t sbuf_align; } vk_caps = {16};
static int ubo_ptr_res(const GfxShaderRes *r) { return r->ptr == 2; }
static const p_bind_buf *ssbo_direct_checked(const VkExec *ex, unsigned stage, const GfxShaderRes *resource,
    int *ubo, p_bind_buf *scratch, int *refusal) {
    assert(ex == &recorder && stage < 8 && resource && scratch && !*refusal);
    *ubo = resource->slot == 0;
    return *ubo ? &ex->ubo[stage][0] : NULL;
}

enum { VK_ACCESS_MEMORY_WRITE_BIT=1, VK_ACCESS_TRANSFER_READ_BIT=2, VK_ACCESS_TRANSFER_WRITE_BIT=4,
    VK_ACCESS_UNIFORM_READ_BIT=8, VK_ACCESS_SHADER_READ_BIT=16, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT=1,
    VK_PIPELINE_STAGE_TRANSFER_BIT=2, VK_NULL_HANDLE=0, VK_STRUCTURE_TYPE_MEMORY_BARRIER=1 };
typedef unsigned VkAccessFlags;
typedef struct {int sType; VkAccessFlags srcAccessMask, dstAccessMask;} VkMemoryBarrier;
static unsigned storage_dependencies;
static void vkCmdPipelineBarrier(uint64_t cb, unsigned from, unsigned to, unsigned flags, unsigned n,
    const VkMemoryBarrier *barrier, unsigned buffers, const void *b, unsigned images, const void *i) {
    assert(cb == recorder.cb && !flags && n == 1 && !buffers && !images && !b && !i);
    if (to == VK_PIPELINE_STAGE_TRANSFER_BIT) {
        assert(from == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT && barrier->srcAccessMask == VK_ACCESS_MEMORY_WRITE_BIT &&
               barrier->dstAccessMask == VK_ACCESS_TRANSFER_READ_BIT);
    } else {
        assert(from == VK_PIPELINE_STAGE_TRANSFER_BIT && to == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT &&
               barrier->srcAccessMask == VK_ACCESS_TRANSFER_WRITE_BIT &&
               (barrier->dstAccessMask == VK_ACCESS_SHADER_READ_BIT || barrier->dstAccessMask == VK_ACCESS_UNIFORM_READ_BIT));
        storage_dependencies += barrier->dstAccessMask == VK_ACCESS_SHADER_READ_BIT;
    }
}
static uint8_t *buffer_scratch_view(const VkExecBufferScratchReceipt *scratch, unsigned slot,
                                   VkDescriptorBufferInfo *view) {
    (void)scratch; (void)slot; (void)view;
    abort();
}
static int find_stage(DrawCtx *draw, const void *prog, int stage, const uint8_t **ctl, const uint8_t **code,
                      uint32_t *length) {
    (void)draw; (void)prog; (void)stage; (void)ctl; (void)code; (void)length;
    abort();
}
static int ctl_nvn2(const uint8_t *ctl) { (void)ctl; abort(); }
static void c0_capture_bytes(uint8_t *c0, const VkExec *ex, int stage, const uint32_t *groups, uint32_t instance,
                             int nvn2, const uint8_t *ctl) {
    (void)c0; (void)ex; (void)stage; (void)groups; (void)instance; (void)nvn2; (void)ctl;
    abort();
}
'''


CASES = r'''
typedef struct {
    VkDescriptorBufferInfo descriptor;
    VkMemSpanCapture *provider;
    CbRes *owner;
} Publication;

static void dispose(void *opaque) {
    Publication *publication = opaque;
    assert(publication->descriptor.buffer);
    if (publication->owner) {
        assert(publication->owner->graphics_slabs);
        update_release(graphics_detach(publication->owner));
    }
    vk_mem_span_capture_release(&publication->provider);
    *publication = (Publication){0};
    retired++;
}
static void complete(void *opaque) { completed++; dispose(opaque); }
static void cancel(void *opaque, int may_have_executed) {
    canceled++;
    uncertain += may_have_executed;
    dispose(opaque);
}
static void finish(Pending *pending, uint64_t done, uint64_t handed, int drained) {
    if (!pending->fence || !pending_disposable(*pending, done, handed, drained)) return;
    pending_dispose(*pending, done, handed);
    *pending = (Pending){0};
}

static const uint8_t *descriptor_bytes(VkDescriptorBufferInfo descriptor) {
    assert(descriptor.buffer && descriptor.buffer <= MERGED);
    uint64_t offset = descriptor.offset;
    if (descriptor.buffer == NATIVE || descriptor.buffer == OTHER_MIRROR) {
        assert(offset >= input.offset);
        offset = OUT_OFFSET + offset - input.offset;
    }
    assert(descriptor.range && offset + descriptor.range <= sizeof buffers[descriptor.buffer]);
    return buffers[descriptor.buffer] + offset;
}

static FixtureRange small_range(uint64_t bytes) {
    return (FixtureRange){.address = 0x1000, .offset = ALIGN, .size = bytes, .update_offset = bytes / 4,
                          .update_size = bytes / 4, .scene = 5, .update_op = 1, .read_op = 2};
}

static void begin(FixtureRange range) {
    assert(!registry && !native_holds && !scratch_owner.graphics_slabs && !graphics_head && !graphics_tail);
    const RcAdmissionDomain domains[] = {
        {RC_ADMISSION_RAM, 0, UINT64_MAX}, {RC_ADMISSION_COMMIT, 0, UINT64_MAX},
    };
    assert(rc_admission_init(&budget, 1, domains, 2, charges, sizeof charges / sizeof charges[0]) == RC_ADMISSION_OK);
    draw_context = (DrawCtx){0};
    scratch_owner = (CbRes){0};
    memset(&LIM, 0, sizeof LIM);
    recording_failed = 0;
    registry = rc_registry_create((RcRegistryConfig){96, 96, 2});
    assert(registry);
    input = range;
    live_mapping = 1;
    mapping = (MemBackingCapture){{input.address, live_mapping, {BACKING, input.offset, input.size}}};
    region = (Reg){input.address, input.address + input.size, OUT_OFFSET, &mapping, &region, 0};
    U.expected = U.on = U.nreg = 1;
    U.capture = 1;
    U.arena = &arena;
    arena.map = buffers[SNAPSHOT];
    U.reg = &region;
    recorder = (VkExec){.cb = 42, .fence = input.scene, .resource_op = input.update_op, .in_pass = 1, .dc = dc};
    recorder.ubo[NVN_STAGE_VS][0] = (p_bind_buf){input.address, input.size};
    OK(rc_registry_gpu_capture_begin(registry, &mapping.interval.range, 1, &U.producer.read));
    OK(rc_registry_declare_through(registry, vk_exec_resource_position()));
    memset(buffers, 0xe9, sizeof buffers);
    for (uint64_t i = 0; i < input.size; i++) guest[i] = original_byte(i);
    memcpy(buffers[SNAPSHOT] + OUT_OFFSET, guest, input.size);
    memcpy(buffers[NATIVE] + OUT_OFFSET, guest, input.size);
    memcpy(buffers[OTHER_MIRROR] + OUT_OFFSET, guest, input.size);
    selected = NATIVE;
    copies = barriers = notes = commits = gpu_writes = 0;
    written_buffer = 0;
}

static void end(void) {
    vk_exec_draw_update_history_end(recorder.fence);
    assert(!UPD.fence && !UPD.n && !native_holds && !scratch_owner.graphics_slabs);
    assert(!graphics_head && !graphics_tail && !graphics_payload_bound && !graphics_payload_current);
    assert(!budget.retained.amount[RC_BUDGET_RAM] && !budget.reserved.amount[RC_BUDGET_RAM] &&
           !budget.pending.amount[RC_BUDGET_RAM]);
    OK(rc_registry_gpu_capture_end(registry, &U.producer.read));
    rc_registry_destroy(registry);
    registry = NULL;
}

static RcRegistryWrite declare(uint64_t offset, uint64_t size, uint64_t ordinal) {
    RcRegistryWrite output;
    OK(rc_registry_write_begin_at(registry, (RcSourceRange){BACKING, input.offset + offset, size},
                                  RC_REGISTRY_BUFFER, (RcRegistryPosition){recorder.fence, ordinal}, &output));
    return output;
}
static void advance(uint64_t operation) {
    assert(operation > recorder.resource_op);
    recorder.resource_op = operation;
    OK(rc_registry_declare_through(registry, vk_exec_resource_position()));
}

static void emit_inline(RcRegistryWrite output_ticket, uint64_t offset, uint64_t size) {
    update_ticket = output_ticket;
    OK(rc_registry_declare_through(registry, (RcRegistryPosition){recorder.fence, recorder.resource_op * 2 + 1}));
    void *storage;
    RcBudgetTicket command_budget = {0};
    assert(vk_replay_metadata_allocate(sizeof(p_update_ub), 0, RC_BUDGET_SNAPSHOT, &command_budget, &storage) == VK_SUCCESS);
    p_update_ub *op = storage;
    assert(size <= sizeof op->data);
    op->dst = input.address + offset;
    op->size = size;
    for (uint64_t i = 0; i < size; i++) op->data[i] = (uint8_t)(0xa1 + i);
    memcpy(buffers[STAGING] + ALIGN, op->data, size);
    VkExecUpdateDestination output = {0};
    OK(vk_mem_span_native_input(op->dst, size, 0, &output.span, &output.hold));
    output.count = vk_mem_span_input_intervals(output.hold, op->dst, size, &output.writes);
    uint64_t old_generation = U.gen;
    update_begin(&recorder);
    update_emit(&recorder, op, output.span, &output, STAGING, ALIGN);
    assert(copies == 1 && commits == 1 && gpu_writes == 1 && !output.hold && !output.writes &&
           U.gen == old_generation + 1);
    // The recorder owns the admitted bytes until its last use of the inline payload.
    assert(vk_exec_draw_update_history_defer(op, sizeof *op, &command_budget) && !command_budget.token.ledger);
}

static RcRegistryResult inline_fragments(uint64_t total, VkExecUniformUpdateFragment *parts, uint32_t capacity,
                                          uint32_t *count) {
    VkMemSpan span;
    VkMemSpanCapture *hold = NULL;
    OK(vk_mem_span_native_input(input.address, total, 0, &span, &hold));
    RcRegistryResult result = vk_exec_uniform_update_fragments(&recorder, input.address, total, &span, hold,
        vk_exec_resource_position(), parts, capacity, count);
    vk_mem_span_capture_release(&hold);
    return result;
}

static const GfxShaderRes shader_ubo = {.slot = 3};
static const VkPipeBinding bindings[] = {
    {.stage = GFX_STAGE_VS, .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .kind = GFX_RES_UBO, .res = &shader_ubo},
    {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .kind = VK_PIPE_SUPPORT},
};
static const VkPipeInfo pipeline = {.bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS,
    .n_bindings = sizeof bindings / sizeof bindings[0], .bindings = bindings};

static VkResult prepare_draw(VkExecGraphicsScratch **grant, uint64_t *demand, RcVmaDependency *dep) {
    const VkPipeState state = {0};
    const p_draw op = {0};
    return vk_exec_draw_graphics_scratch_prepare(&recorder, &state, &pipeline, OP_DRAW, &op, 1, grant, demand, dep, NULL);
}

static VkExecGraphicsScratch plan_merge(void) {
    // Reserve preceding bytes to exercise a nonzero merge descriptor offset.
    VkExecGraphicsScratch plan = {.demand = {.alignment = ALIGN, .bytes = OUT_OFFSET},
                                  .info = &pipeline, .ordinal = recorder.resource_op};
    RcVmaDependency dependency = {0};
    assert(graphics_ubo_merge_plan(&recorder, &pipeline, &plan, &dependency) && !dependency.selected);
    return plan;
}

static VkExecGraphicsScratch plan_checked(void) {
    VkExecGraphicsScratch plan = plan_merge();
    GraphicsUboMerge *merge = graphics_ubo_merge_find(&plan, NVN_STAGE_VS, 3, input.address, input.size);
    assert(merge && plan.ubo_merge_count == 1 && merge->fragment_count == 1 && merge->gap_count == 2);
    assert(graphics_ubo_merge_current(&recorder, merge));
    const GraphicsUboFragment *fragment = &merge->fragments[0];
    assert(fragment->offset == input.update_offset && fragment->bytes == input.update_size);
    assert(fragment->output.scene == input.scene && fragment->output.ordinal == input.update_op * 2 + 1);
    assert(merge->offset == OUT_OFFSET && plan.demand.bytes >= OUT_OFFSET + input.size);
    assert(native_holds == 1 && budget.retained.amount[RC_BUDGET_RAM] == sizeof(GraphicsUboMerge) + sizeof(p_update_ub));
    return plan;
}

static int bind_uniform(unsigned slot, int required, VkDescriptorBufferInfo *descriptor, const uint8_t **bytes) {
    return uniform_buffer_descriptor(&recorder, NULL, NVN_STAGE_VS, slot, input.address, input.size, ALIGN, required,
                                     VK_MUSE_UBO, descriptor, bytes);
}

static void expect_mixed(Publication out, uint64_t total, uint64_t offset, uint64_t length, uint8_t first) {
    assert(out.descriptor.offset % ALIGN == 0 && out.descriptor.range == total);
    const uint8_t *actual = descriptor_bytes(out.descriptor);
    for (uint64_t i = 0; i < total; i++) {
        uint8_t expected = original_byte(i);
        if (i >= offset && i - offset < length) expected = (uint8_t)(first + i - offset);
        assert(actual[i] == expected);
    }
}

static void inline_merge(FixtureRange range, int cancel_scene) {
    begin(range);
    uint64_t total = input.size, offset = input.update_offset, length = input.update_size;
    VkDescriptorBufferInfo old = {0};
    VkUsnapLease full = {0};
    const uint8_t *old_bytes = NULL;
    dc->prep_capture = &full;
    assert(bind_uniform(3, 1, &old, &old_bytes));
    assert(old.buffer == SNAPSHOT && old.range == total && old_bytes == descriptor_bytes(old));
    assert(old_bytes[offset] == original_byte(offset) && full.capture);
    dc->prep_capture = NULL;

    memset(guest, 0x19, total);
    RcRegistryWrite output = declare(offset, length, input.update_op * 2 + 1);
    emit_inline(output, offset, length);
    advance(input.read_op);
    RcRegistryPosition read = vk_exec_resource_position();
    assert(vk_usnap_resolve_at(input.address, total, ALIGN, read, NULL, NULL) == VK_USNAP_RESOLVE_ORDERED_PRODUCER);
    assert(!vk_usnap_lease_valid_at(&full, read) && old_bytes[offset] == original_byte(offset));
    uint8_t miss[MAX_BYTES] = {0};
    assert(upd_read_uniform(&recorder, input.address, total, miss) == VK_EXEC_UNIFORM_UPDATE_STALE && !miss[0]);
    expect_mixed((Publication){.descriptor = {NATIVE, input.offset, total}}, total, offset, length, 0xa1);

    unsigned pre_refusals = budget_refusals, pre_readiness = readiness_queries, pre_device = device_queries;
    RcVmaDependency dependency = {0};
    VkExecGraphicsScratch *grant = NULL;
    uint64_t demand = 0;
    uint64_t metadata = sizeof(GraphicsUboMerge);
    const uint64_t journal_bytes = sizeof(p_update_ub);

    // Exercise the cold limits query and the public preparation status.
    assert(!LIM.ubo_max);
    assert(rc_admission_set_limit(&budget, RC_BUDGET_RAM, 0) == RC_ADMISSION_OK);
    assert(prepare_draw(&grant, &demand, &dependency) == VK_NOT_READY && !grant && demand);
    assert(LIM.ubo_max == MAX_BYTES && LIM.ubo_align == ALIGN && LIM.sbo_align == ALIGN / 2 &&
           device_queries == pre_device + 1);
    assert(dependency.selected && dependency.peak.amount[RC_BUDGET_RAM] == metadata &&
           dependency.peak.amount[RC_BUDGET_COMMIT] == metadata);
    assert(dependency.before.records_available && !dependency.before.available.amount[RC_BUDGET_RAM]);
    assert(!native_holds && !scratch_owner.graphics_slabs && rc_admission_used_at(&budget, RC_BUDGET_RAM) == journal_bytes);
    assert(rc_admission_set_limit(&budget, RC_BUDGET_RAM, metadata + journal_bytes) == RC_ADMISSION_OK);

    // Metadata now fits while the scratch receipt's incremental host charge does not.
    assert(prepare_draw(&grant, &demand, &dependency) == VK_NOT_READY && !grant &&
           demand >= OUT_OFFSET + total);
    assert(dependency.selected && dependency.peak.amount[RC_BUDGET_RAM] == metadata + sizeof(VkExecGraphicsScratch) &&
           dependency.peak.amount[RC_BUDGET_COMMIT] == metadata + sizeof(VkExecGraphicsScratch));
    assert(dependency.before.records_available && !dependency.before.available.amount[RC_BUDGET_RAM]);
    assert(!native_holds && !scratch_owner.graphics_slabs && rc_admission_used_at(&budget, RC_BUDGET_RAM) == journal_bytes &&
           rc_admission_used_at(&budget, RC_BUDGET_COMMIT) == journal_bytes);
    assert(budget_refusals == pre_refusals + 2 && readiness_queries == pre_readiness + 2);
    assert(rc_admission_set_limit(&budget, RC_BUDGET_RAM, UINT64_MAX) == RC_ADMISSION_OK);

    uint32_t fragments = 7;
    assert(inline_fragments(total, NULL, 0, &fragments) == RC_REGISTRY_CAPACITY && !fragments);
    VkExecUniformUpdateFragment slice;
    selected = OTHER_MIRROR;
    assert(inline_fragments(total, &slice, 1, &fragments) == RC_REGISTRY_STALE && !fragments && !native_holds);
    VkExecGraphicsScratch wrong = plan_merge();
    assert(!wrong.ubo_merge_count && !native_holds);
    graphics_ubo_merges_release(&wrong);
    selected = NATIVE;

    dependency = (RcVmaDependency){0};
    VkExecGraphicsScratch plan = plan_checked();
    unsigned pre_creates = scratch_creates, pre_releases = scratch_releases;
    capacity_refused = 1;
    assert(graphics_scratch_claim(&recorder, plan, 0, 1, &grant, &dependency) == VK_NOT_READY &&
           dependency.selected && !grant && !native_holds && !scratch_owner.graphics_slabs);
    assert(budget.retained.amount[RC_BUDGET_RAM] == journal_bytes && scratch_creates == pre_creates);
    capacity_refused = 0;
    plan = plan_checked();
    remap_during_copy = 1;
    assert(graphics_scratch_claim(&recorder, plan, 0, 1, &grant, &dependency) == VK_ERROR_INITIALIZATION_FAILED);
    assert(!remap_during_copy && live_mapping == 2 && !grant && !native_holds && !scratch_owner.graphics_slabs);
    assert(budget.retained.amount[RC_BUDGET_RAM] == journal_bytes && scratch_creates == pre_creates + 1 &&
           scratch_releases == pre_releases + 1);
    assert(vk_usnap_resolve_at(input.address, total, ALIGN, read, NULL, NULL) == VK_USNAP_RESOLVE_MAPPING_STALE);
    live_mapping = 1;

    plan = plan_checked();
    assert(graphics_scratch_claim(&recorder, plan, 0, 1, &grant, &dependency) == VK_SUCCESS);
    assert(grant && !grant->exposed && scratch_owner.graphics_slabs);
    vk_exec_draw_graphics_scratch_finish(&grant);
    assert(!grant && !native_holds && !scratch_owner.graphics_slabs && budget.retained.amount[RC_BUDGET_RAM] == journal_bytes);
    assert(scratch_releases == pre_releases + 2);

    // A future writer leaves this read valid and blocks the later observation.
    RcRegistryWrite later = declare(offset + 4, 4, read.ordinal + 1);
    plan = plan_checked();
    assert(graphics_scratch_claim(&recorder, plan, 0, 1, &grant, &dependency) == VK_SUCCESS);
    graphics_payload_bound = grant;
    Publication good = {.owner = &scratch_owner};
    VkDescriptorBufferInfo refused_descriptor = {0};
    const uint8_t *scratch_bytes = NULL;
    assert(!bind_uniform(4, 0, &refused_descriptor, &scratch_bytes));
    assert(!refused_descriptor.buffer && !scratch_bytes && !grant->exposed);
    assert(bind_uniform(3, 1, &good.descriptor, &scratch_bytes));
    assert(good.descriptor.buffer == MERGED && good.descriptor.offset == OUT_OFFSET);
    assert(scratch_bytes == descriptor_bytes(good.descriptor) && grant->exposed && !recording_failed);
    expect_mixed(good, total, offset, length, 0xa1);
    assert(guest[0] == 0x19 && scratch_bytes[0] == original_byte(0));
    assert(buffers[MERGED][OUT_OFFSET - 1] == 0xe9 && buffers[MERGED][OUT_OFFSET + total] == 0xe9);
    EffectInputCapture effect = {.ex = &recorder, .stage = NVN_STAGE_VS, .read_position = read};
    assert(effect_capture_uniform(&effect, 3) && effect_input_current(&effect));
    assert(effect.uniform[3].buffer == good.descriptor.buffer && effect.uniform[3].offset == good.descriptor.offset &&
           effect.uniform[3].range == total && effect.uniform_bytes[3] == scratch_bytes && !effect.leased);
    published++;

    advance(input.read_op + 1);
    assert(!effect_input_current(&effect));
    assert(inline_fragments(total, &slice, 1, &fragments) == RC_REGISTRY_BLOCKED && !fragments && native_holds == 1);
    assert(!bind_uniform(3, 1, &refused_descriptor, &scratch_bytes));
    assert(!refused_descriptor.buffer && !scratch_bytes && recording_failed);
    OK(rc_registry_write_cancel(registry, later));

    graphics_payload_bound = NULL;
    vk_exec_draw_graphics_scratch_finish(&grant);
    assert(!grant && !native_holds && scratch_owner.graphics_slabs && budget.retained.amount[RC_BUDGET_RAM] == journal_bytes);
    assert(scratch_releases == pre_releases + 2);
    vk_exec_draw_update_history_end(recorder.fence);
    assert(!rc_admission_used_at(&budget, RC_BUDGET_RAM) && !rc_admission_used_at(&budget, RC_BUDGET_COMMIT));
    Pending work = {recorder.fence, complete, cancel, &good};
    if (cancel_scene) {
        finish(&work, 0, 0, 0);
        assert(!good.descriptor.buffer && !scratch_owner.graphics_slabs);
        OK(rc_registry_write_cancel(registry, output));
        assert(vk_usnap_resolve_at(input.address, total, ALIGN, vk_exec_resource_position(), &old, NULL) ==
               VK_USNAP_RESOLVE_READY);
        assert(descriptor_bytes(old)[offset] == original_byte(offset));
    } else {
        finish(&work, 0, recorder.fence, 0);
        assert(work.fence && scratch_owner.graphics_slabs);
        expect_mixed(good, total, offset, length, 0xa1);
        finish(&work, recorder.fence, recorder.fence, 0);
        assert(!good.descriptor.buffer && !scratch_owner.graphics_slabs);
        OK(rc_registry_write_cancel(registry, output));
    }
    finish(&work, recorder.fence, recorder.fence, 1);
    end();
}

/* The test's native write receipt identifies the physical destination. Equal
 * canonical backing coverage and a current mapping alone are insufficient. */
static int consume_native(uint64_t total, Publication *out) {
    VkMemSpan span;
    VkMemSpanCapture *hold = NULL;
    OK(vk_mem_span_native_input(input.address, total, 0, &span, &hold));
    RcInterval *ranges = NULL;
    uint32_t count = vk_mem_span_input_intervals(hold, input.address, total, &ranges);
    int ordered = rc_registry_read_device_buffer_at(registry, ranges, count, vk_exec_resource_position()) ==
                  RC_REGISTRY_OK;
    free(ranges);
    assert(ordered); // The registry cannot distinguish these native mirrors.
    if (span.buffer != written_buffer || !vk_mem_span_capture_current(hold, &span)) {
        vk_mem_span_capture_release(&hold);
        return 0;
    }
    *out = (Publication){.descriptor = {span.buffer, span.offset, total}, .provider = hold};
    published++;
    return 1;
}

static void native_merge(void) {
    const uint64_t total = 64, offset = 16, length = 16;
    begin(small_range(total));
    RcRegistryWrite output = declare(offset, length, input.update_op * 2 + 1);
    // Model a non-inline ordered buffer fill on the selected provider.
    for (uint64_t i = 0; i < length; i++) buffers[NATIVE][OUT_OFFSET + offset + i] = (uint8_t)(0xc1 + i);
    written_buffer = NATIVE;
    advance(input.read_op);
    assert(vk_usnap_resolve_at(input.address, total, ALIGN, vk_exec_resource_position(), NULL, NULL) ==
           VK_USNAP_RESOLVE_ORDERED_PRODUCER);
    uint8_t miss[MAX_BYTES];
    assert(upd_read_uniform(&recorder, input.address + offset, length, miss) == VK_EXEC_UNIFORM_UPDATE_NONE);
    Publication native = {0}, wrong = {0};
    selected = OTHER_MIRROR;
    assert(!consume_native(total, &wrong) && !native_holds && !wrong.descriptor.buffer);
    selected = NATIVE;
    assert(consume_native(total, &native) && native_holds == 1);
    expect_mixed(native, total, offset, length, 0xc1);
    Pending work = {recorder.fence, complete, cancel, &native};
    finish(&work, 0, recorder.fence, 0);
    assert(work.fence && native_holds == 1);
    finish(&work, 0, recorder.fence, 1); // Queue idle supports retirement, not successful execution.
    assert(!work.fence && !native_holds && !native.descriptor.buffer);
    assert(uncertain == 1);
    OK(rc_registry_write_cancel(registry, output));
    end();
}

typedef enum { ACCESS_ONLY, EXACT_NARROW, PRESERVED_MIRROR, PRESERVED_MIRROR_MISSING, PRESERVED_MIRROR_FOREIGN } BufferResult;

static void storage_uniform_selection(VkPipelineBindPoint point, BufferResult result) {
    FixtureRange range = small_range(16);
    const int preserved = result >= PRESERVED_MIRROR;
    const int produced = result == EXACT_NARROW || result == PRESERVED_MIRROR;
    if (preserved) range.read_op++;
    begin(range);
    RcRegistryWrite predecessor = {0};
    if (preserved) {
        const RcRegistryPosition prior_output = {recorder.fence, recorder.resource_op * 2 + 1};
        const uint64_t seed_bytes = result == PRESERVED_MIRROR_MISSING ? 12 : 16;
        predecessor = declare(0, seed_bytes, prior_output.ordinal);
        OK(rc_registry_declare_through(registry, prior_output));
        const RcRegistryBufferSpan prior = {{input.address, live_mapping, {BACKING, input.offset, seed_bytes}},
            {result == PRESERVED_MIRROR_FOREIGN ? OTHER_MIRROR : NATIVE, live_mapping, input.offset}};
        OK(rc_registry_buffer_access_record(registry, predecessor, prior, prior_output));
        OK(rc_registry_buffer_write_record(registry, predecessor, prior, prior_output));
        for (unsigned i = 0; i < seed_bytes; ++i) buffers[NATIVE][OUT_OFFSET + i] = (uint8_t)(0x60 + i);
        advance(recorder.resource_op + 1);
    }
    const unsigned stage = point == VK_PIPELINE_BIND_POINT_GRAPHICS ? NVN_STAGE_VS : NVN_STAGE_CS;
    const GfxShaderRes storage = {.slot = 0};
    const VkPipeBinding binding = {.stage = point == VK_PIPELINE_BIND_POINT_GRAPHICS ? GFX_STAGE_VS : GFX_STAGE_CS,
        .kind = GFX_RES_SSBO, .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .count = 1, .res = &storage};
    const VkPipeInfo info = {.bind_point = point, .n_bindings = 1, .bindings = &binding};
    recorder.ubo[stage][0] = (p_bind_buf){input.address + 4, 12};
    const RcRegistryPosition output = {recorder.fence, recorder.resource_op * 2 + 1};
    const unsigned write_start = preserved ? 0 : 4;
    const unsigned write_size = preserved ? 16 : 4;
    RcRegistryWrite writer = declare(write_start, write_size, output.ordinal);
    OK(rc_registry_declare_through(registry, output));
    RcRegistryBufferSpan span = {{input.address + write_start, live_mapping, {BACKING, input.offset + write_start, write_size}},
                                 {NATIVE, live_mapping, input.offset + write_start}};
    OK(rc_registry_buffer_access_record(registry, writer, span, output));
    for (uint64_t i = 0; i < 4; ++i) buffers[NATIVE][OUT_OFFSET + 4 + i] = (uint8_t)(0x91 + i);
    if (result == EXACT_NARROW) OK(rc_registry_buffer_write_record(registry, writer, span, output));
    else if (preserved) {
        RcRegistryResult recorded = rc_registry_buffer_preserved_record(registry, writer, span, vk_exec_resource_position(), 0);
        assert(recorded == (produced ? RC_REGISTRY_OK : RC_REGISTRY_BLOCKED));
    }
    // Current guest data differs: both unchanged gaps must use the original capture, not CPU re-import.
    memset(guest, 0x77, 16);
    advance(input.read_op);
    RcRegistryPosition read = vk_exec_resource_position();
    assert(vk_usnap_resolve_at(input.address, 16, ALIGN, read, NULL, NULL) == VK_USNAP_RESOLVE_ORDERED_PRODUCER);
    UboMergePlan plan = {0};
    GraphicsPayloadDemand demand = {.alignment = ALIGN, .bytes = OUT_OFFSET};
    assert(ubo_merge_plan(&recorder, &info, recorder.resource_op, &demand, &plan, NULL));
    assert(plan.count == (unsigned)produced);
    if (produced) {
        assert(plan.entries[0].storage_uniform == &storage && plan.entries[0].fragment_count == 1 &&
               plan.entries[0].native_count == 1 && plan.entries[0].gap_count == (preserved ? 0u : 2u));
        assert(plan.entries[0].fragments[0].output.ordinal == output.ordinal);
        assert(graphics_ubo_merge_current(&recorder, &plan.entries[0]));
        recorder.ubo[stage][0].size--;
        assert(!graphics_ubo_merge_current(&recorder, &plan.entries[0]));
        recorder.ubo[stage][0].size++;
        assert(ubo_merges_initialize(&recorder, &plan, buffers[MERGED]));
        unsigned dependencies = storage_dependencies;
        recorder.in_pass = 0;
        VkExecGraphicsScratch graphics = {.start = 0, .ubo_merges = plan.entries, .ubo_merge_count = plan.count};
        UpdateSlab slab = {.chunk = {.buf=MERGED, .map=buffers[MERGED]}};
        graphics.slab = &slab;
        VkExecBufferScratchReceipt compute = {.ubo = plan, .chunk = slab.chunk, .info = &info, .collecting=1};
        VkDescriptorBufferInfo value = {0};
        if (point == VK_PIPELINE_BIND_POINT_GRAPHICS) {
            int recorded = 0;
            assert(ubo_merges_record_native(&recorder, &plan, MERGED, 0, &recorded) && recorded);
            graphics_payload_bound = &graphics;
            graphics.native_materialized = 1;
        } else {
            recording_scratch = &compute;
            assert(dispatch_materialize_ubo_merges(&recorder, &info));
            compute.collecting = 0;
        }
        assert(uniform_buffer_descriptor(&recorder, NULL, stage, 0, input.address, 16, LIM.sbo_align, 1,
                                         VK_MUSE_SSBO, &value, NULL));
        assert(value.buffer == MERGED && value.offset == OUT_OFFSET && storage_dependencies == dependencies + 1);
        if (preserved) {
            const uint8_t *bytes = descriptor_bytes(value);
            for (unsigned i = 0; i < 16; ++i)
                assert(bytes[i] == (uint8_t)(i >= 4 && i < 8 ? 0x91 + i - 4 : 0x60 + i));
        } else expect_mixed((Publication){.descriptor=value}, 16, 4, 4, 0x91);
        graphics_payload_bound = NULL;
        recording_scratch = NULL;
    } else {
        assert(rc_registry_buffer_access_check(registry, span, read) ==
            (result == PRESERVED_MIRROR_FOREIGN ? RC_REGISTRY_BLOCKED : RC_REGISTRY_OK));
        VkDescriptorBufferInfo value = {0};
        assert(!uniform_buffer_descriptor(&recorder, NULL, stage, 0, input.address, 16, LIM.sbo_align, 1,
                                          VK_MUSE_SSBO, &value, NULL) && !value.buffer);
    }
    ubo_merges_release(&plan);
    OK(rc_registry_write_cancel(registry, writer));
    if (preserved) {
        OK(rc_registry_write_cancel(registry, predecessor));
        assert(vk_usnap_resolve_at(input.address, 16, ALIGN, read, NULL, NULL) == VK_USNAP_RESOLVE_READY);
    }
    end();
}

static void many_ordered_bindings(VkPipelineBindPoint point) {
    enum { INPUTS = 6, BYTES = 16 };
    FixtureRange range = small_range(INPUTS * ALIGN);
    begin(range);
    const unsigned stage = point == VK_PIPELINE_BIND_POINT_GRAPHICS ? NVN_STAGE_VS : NVN_STAGE_CS;
    const unsigned shader_stage = point == VK_PIPELINE_BIND_POINT_GRAPHICS ? GFX_STAGE_VS : GFX_STAGE_CS;
    const GfxShaderRes first = {.slot = 3};
    const VkPipeBinding array = {.stage = shader_stage, .count = INPUTS, .kind = GFX_RES_UBO,
                                 .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .res = &first};
    const VkPipeInfo info = {.bind_point = point, .n_bindings = 1, .bindings = &array};
    const RcRegistryPosition output = {recorder.fence, recorder.resource_op * 2 + 1};
    RcRegistryWrite writers[INPUTS];
    for (unsigned i = 0; i < INPUTS; ++i) writers[i] = declare(i * ALIGN + 4, 4, output.ordinal);
    OK(rc_registry_declare_through(registry, output));
    for (unsigned i = 0; i < INPUTS; ++i) {
        const uint64_t from = i * ALIGN;
        recorder.ubo[stage][i] = (p_bind_buf){input.address + from, BYTES};
        const RcRegistryBufferSpan written = {{input.address + from + 4, live_mapping, {BACKING, input.offset + from + 4, 4}},
                                              {NATIVE, live_mapping, input.offset + from + 4}};
        OK(rc_registry_buffer_write_record(registry, writers[i], written, output));
        memset(buffers[NATIVE] + OUT_OFFSET + from + 4, (int)(0xa0 + i), 4);
    }
    advance(input.read_op);
    UboMergePlan plan = {0};
    GraphicsPayloadDemand demand = {.alignment = ALIGN, .bytes = OUT_OFFSET};
    RcVmaDependency dependency = {0};
    const uint64_t expected_cost = INPUTS * sizeof(GraphicsUboMerge);
    assert(rc_admission_set_limit(&budget, RC_BUDGET_RAM, expected_cost - 1) == RC_ADMISSION_OK);
    assert(!ubo_merge_plan(&recorder, &info, recorder.resource_op, &demand, &plan, &dependency));
    assert(dependency.selected && dependency.peak.amount[RC_BUDGET_RAM] == expected_cost && !native_holds);
    assert(rc_admission_set_limit(&budget, RC_BUDGET_RAM, expected_cost) == RC_ADMISSION_OK);
    assert(ubo_merge_plan(&recorder, &info, recorder.resource_op, &demand, &plan, NULL));
    assert(plan.count == INPUTS && native_holds == INPUTS);
    assert(ubo_merges_initialize(&recorder, &plan, buffers[MERGED]));
    recorder.in_pass = 0;
    int recorded = 0;
    assert(ubo_merges_record_native(&recorder, &plan, MERGED, 0, &recorded) && recorded);

    UpdateSlab slab = {.chunk = {.buf=MERGED,.map=buffers[MERGED]}};
    VkExecGraphicsScratch graphics = {.ubo_merges=plan.entries,.ubo_merge_count=plan.count,.slab=&slab,.native_materialized=1};
    VkExecBufferScratchReceipt compute = {.ubo=plan,.chunk=slab.chunk,.info=&info,.ubo_materialized=1};
    if (point == VK_PIPELINE_BIND_POINT_GRAPHICS) graphics_payload_bound = &graphics;
    else recording_scratch = &compute;
    for (unsigned i = 0; i < INPUTS; ++i) {
        VkDescriptorBufferInfo desc = {0};
        const uint64_t from = i * ALIGN;
        assert(uniform_buffer_descriptor(&recorder, NULL, stage, 3 + i, input.address + from, BYTES,
                                          LIM.ubo_align, 1, VK_MUSE_UBO, &desc, NULL));
        assert(desc.buffer == MERGED && desc.offset == OUT_OFFSET + from && desc.range == BYTES);
        const uint8_t *actual = descriptor_bytes(desc);
        for (unsigned j = 0; j < BYTES; ++j)
            assert(actual[j] == (j >= 4 && j < 8 ? 0xa0 + i : original_byte(from + j)));
    }
    graphics_payload_bound = NULL;
    recording_scratch = NULL;
    ubo_merges_release(&plan);
    for (unsigned i = 0; i < INPUTS; ++i) OK(rc_registry_write_cancel(registry, writers[i]));
    end();
}

static void many_ordered_fragments(void) {
    enum { FRAGMENTS = 40, BYTES = FRAGMENTS * 8 };
    begin(small_range(BYTES));
    const RcRegistryPosition output = {recorder.fence, recorder.resource_op * 2 + 1};
    RcRegistryWrite writers[FRAGMENTS];
    for (unsigned i = 0; i < FRAGMENTS; ++i) writers[i] = declare(i * 8 + 4, 4, output.ordinal);
    OK(rc_registry_declare_through(registry, output));
    for (unsigned i = 0; i < FRAGMENTS; ++i) {
        const uint64_t start = i * 8 + 4;
        const RcRegistryBufferSpan written = {{input.address + start, live_mapping, {BACKING, input.offset + start, 4}},
                                              {NATIVE, live_mapping, input.offset + start}};
        OK(rc_registry_buffer_write_record(registry, writers[i], written, output));
        memset(buffers[NATIVE] + OUT_OFFSET + start, (int)(0x80 + i), 4);
    }
    advance(input.read_op);
    UboMergePlan plan = {0};
    GraphicsPayloadDemand demand = {.alignment = ALIGN, .bytes = OUT_OFFSET};
    RcVmaDependency dependency = {0};
    assert(rc_admission_set_limit(&budget, RC_BUDGET_RAM, sizeof(GraphicsUboMerge)) == RC_ADMISSION_OK);
    assert(!ubo_merge_plan(&recorder, &pipeline, recorder.resource_op, &demand, &plan, &dependency));
    assert(dependency.selected && dependency.peak.amount[RC_BUDGET_RAM] > sizeof(GraphicsUboMerge) && !native_holds);
    const uint64_t exact_budget = dependency.peak.amount[RC_BUDGET_RAM];
    ubo_merges_release(&plan);
    assert(!budget.retained.amount[RC_BUDGET_RAM]);
    assert(rc_admission_set_limit(&budget, RC_BUDGET_RAM, exact_budget) == RC_ADMISSION_OK);
    assert(ubo_merge_plan(&recorder, &pipeline, recorder.resource_op, &demand, &plan, NULL));
    assert(plan.count == 1 && plan.entries[0].overflow && plan.entries[0].fragment_count == FRAGMENTS &&
           plan.entries[0].native_count == FRAGMENTS && plan.entries[0].gap_count == FRAGMENTS);
    assert(ubo_merges_initialize(&recorder, &plan, buffers[MERGED]));
    recorder.in_pass = 0;
    int copied = 0;
    assert(ubo_merges_record_native(&recorder, &plan, MERGED, 0, &copied) && copied);
    for (unsigned i = 0; i < BYTES; ++i)
        assert(buffers[MERGED][OUT_OFFSET + i] == (i % 8 >= 4 ? 0x80 + i / 8 : original_byte(i)));
    assert(graphics_ubo_merge_current(&recorder, plan.entries));
    live_mapping++;
    assert(!graphics_ubo_merge_current(&recorder, plan.entries));
    live_mapping--;
    ubo_merges_release(&plan);
    for (unsigned i = 0; i < FRAGMENTS; ++i) OK(rc_registry_write_cancel(registry, writers[i]));
    end();
}

static void fixed_vertex_footprint(void) {
    FixtureRange range = small_range(MAX_BYTES);
    range.read_op = 5;
    begin(range);
    RcRegistryWrite first = declare(16, 16, 3);
    input.update_offset = 16;
    input.update_size = 16;
    emit_inline(first, 16, 16);
    advance(2);
    copies = commits = gpu_writes = notes = barriers = 0;
    input.update_offset = 52;
    input.update_size = 4;
    RcRegistryWrite second = declare(52, 4, 5);
    emit_inline(second, 52, 4);
    advance(3);
    // This unpositioned result is a real blocker for a whole read and disjoint from both actual loads.
    RcRegistryWrite hole = declare(40, 4, 7);
    advance(4);
    VkPipeBinding binds[2] = {{.stage = GFX_STAGE_VS, .stages = VK_SHADER_STAGE_VERTEX_BIT, .binding = 1,
        .kind = GFX_RES_UBO, .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .count = 1, .res = &shader_ubo}};
    VkPipeInfo info = {.bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS, .n_bindings = 1, .bindings = binds};
    const VkGraphicsUniformFootprints proof = {.count = 1,.bindings = {{.binding = 1, .first = 0, .count = 2}},
        .ranges = {{16, 16}, {52, 4}}};
    assert(graphics_uniform_footprint(&info, binds, &proof, MAX_BYTES));
    assert(!graphics_uniform_footprint(&info, binds, &proof, 55));
    binds[0].stages |= VK_SHADER_STAGE_FRAGMENT_BIT;
    assert(!graphics_uniform_footprint(&info, binds, &proof, MAX_BYTES));
    binds[0].stages = VK_SHADER_STAGE_VERTEX_BIT;
    binds[1] = binds[0];
    info.n_bindings = 2;
    assert(!graphics_uniform_footprint(&info, binds, &proof, MAX_BYTES));
    info.n_bindings = 1;
    assert(!plan_merge().ubo_merge_count);

    VkExecGraphicsScratch plan = {.demand = {.alignment = ALIGN, .bytes = OUT_OFFSET},
        .info = &info, .ordinal = recorder.resource_op, .vertex_uniforms = &proof};
    assert(graphics_ubo_merge_plan(&recorder, &info, &plan, NULL));
    assert(plan.ubo_merge_count == 2 && plan.demand.bytes == OUT_OFFSET + MAX_BYTES && native_holds == 2);
    assert(plan.ubo_merges[0].offset == OUT_OFFSET + 16 && plan.ubo_merges[1].offset == OUT_OFFSET + 52);
    assert(plan.ubo_merges[0].fragments[0].output.ordinal == 3 && plan.ubo_merges[1].fragments[0].output.ordinal == 5);
    UpdateSlab slab = {.chunk = {.buf = MERGED, .map = buffers[MERGED]}};
    plan.slab = &slab;
    assert(graphics_payload_initialize(&recorder, &plan, 0));
    graphics_payload_bound = &plan;
    VkDescriptorBufferInfo descriptor = {0};
    const uint8_t *host = buffers[MERGED];
    assert(!graphics_ubo_merge_find(&plan, NVN_STAGE_VS, 3, input.address, MAX_BYTES));
    assert(!graphics_ubo_merge_find(&plan, NVN_STAGE_VS, 3, input.address + 16, 16));
    assert(!uniform_buffer_descriptor(&recorder, &binds[0], NVN_STAGE_VS, 3, input.address, MAX_BYTES, ALIGN,
                                      0, VK_MUSE_SSBO, &descriptor, &host) && !descriptor.buffer && !host);
    VkPipeBinding other = binds[0];
    other.binding = 2;
    assert(!uniform_buffer_descriptor(&recorder, &other, NVN_STAGE_VS, 3, input.address, MAX_BYTES, ALIGN,
                                      0, VK_MUSE_UBO, &descriptor, &host) && !descriptor.buffer && !host);
    assert(!uniform_buffer_descriptor(&recorder, NULL, NVN_STAGE_VS, 3, input.address, MAX_BYTES, ALIGN,
                                      0, VK_MUSE_UBO, &descriptor, &host) && !descriptor.buffer && !host);
    assert(uniform_buffer_descriptor(&recorder, &binds[0], NVN_STAGE_VS, 3, input.address, MAX_BYTES, ALIGN,
                                     1, VK_MUSE_UBO, &descriptor, &host) && !host && plan.exposed);
    assert(descriptor.buffer == MERGED && descriptor.offset == OUT_OFFSET && descriptor.range == MAX_BYTES);
    const uint8_t *actual = descriptor_bytes(descriptor);
    for (unsigned i = 0; i < MAX_BYTES; ++i) {
        unsigned expected = i >= 16 && i < 32 ? 0xa1 + i - 16 : i >= 52 && i < 56 ? 0xa1 + i - 52 : 0xe9;
        assert(actual[i] == expected);
    }
    live_mapping++;
    assert(!graphics_ubo_merge_current(&recorder, &plan.ubo_merges[0]));
    live_mapping--;
    recorder.ubo[NVN_STAGE_VS][0].size--;
    assert(!graphics_ubo_merge_current(&recorder, &plan.ubo_merges[1]));
    recorder.ubo[NVN_STAGE_VS][0].size++;
    graphics_payload_bound = NULL;
    graphics_ubo_merges_release(&plan);

    RcRegistryWrite touching = declare(52, 4, 9);
    advance(5);
    plan = (VkExecGraphicsScratch){.demand = {.alignment = ALIGN}, .info = &info,
        .ordinal = recorder.resource_op, .vertex_uniforms = &proof};
    assert(graphics_ubo_merge_plan(&recorder, &info, &plan, NULL) && !plan.ubo_merge_count && !native_holds);
    assert(!plan.vertex_uniforms);
    graphics_ubo_merges_release(&plan);
    OK(rc_registry_write_cancel(registry, touching));
    OK(rc_registry_write_cancel(registry, hole));
    OK(rc_registry_write_cancel(registry, first));
    OK(rc_registry_write_cancel(registry, second));
    end();
    puts("fixed final VS footprint: two native-positioned inline spans, forbidden old holes, original descriptor offsets and range, stale and shared refusal");
}

int main(void) {
    inline_merge(small_range(32), 1);
    inline_merge(small_range(64), 0);
    // Field scene 7, program 4134 VS cbuf 3 (guest UBO 0), read ordinal 760.
    inline_merge((FixtureRange){.address = UINT64_C(0x1586ac000), .offset = UINT64_C(0x286ac000),
                                .size = 64 * 1024, .update_offset = 0x20, .update_size = 0x40,
                                .scene = 7, .update_op = 7, .read_op = 380}, 0);
    native_merge();
    assert(published == 4 && retired == 4 && completed == 2 && canceled == 2 && uncertain == 1);
    assert(scratch_creates == 9 && scratch_releases == 9 && refused == 3 && merge_copies == 6);
    assert(budget_refusals == 6 && readiness_queries == 6 && device_queries == 3);
    for (BufferResult result = ACCESS_ONLY; result <= PRESERVED_MIRROR_FOREIGN; ++result) {
        storage_uniform_selection(VK_PIPELINE_BIND_POINT_GRAPHICS, result);
        storage_uniform_selection(VK_PIPELINE_BIND_POINT_COMPUTE, result);
    }
    many_ordered_bindings(VK_PIPELINE_BIND_POINT_GRAPHICS);
    many_ordered_bindings(VK_PIPELINE_BIND_POINT_COMPUTE);
    many_ordered_fragments();
    fixed_vertex_footprint();
    puts("ordered buffer inputs: six independent bindings and 40 native fragments with original snapshot gaps, admission and remap");
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT / "runtime")
    args = parser.parse_args()
    COMBINED = args.runtime

    def replace(text, old, new):
        assert text.count(old) == 1, old
        return text.replace(old, new)


    def extract(source, *names):
        return "".join(function(source, name) for name in names)


    prefix = PREFIX
    prefix = replace(prefix, "    uint64_t capture, hits, gen;", "    uint64_t capture, hits, gen, fence;")
    prefix = replace(prefix, "typedef struct { VkMemSpan span; uint64_t mapping, address; } VkMemSpanCapture;", """
    typedef enum { POOL_IMPORT, POOL_PHYS } FixturePoolKind;
    typedef struct { FixturePoolKind kind; uint64_t acc_sub; } PoolRec;
    typedef struct { VkMemSpan span; uint64_t mapping, address, size; const PoolRec *provider; } VkMemSpanCapture;
    static PoolRec imported = {POOL_IMPORT, 0};
    static PlatMutex pool_lock;
    static uint64_t msub;
    static int cross_scene_fixture;
    #define plat_mutex_lock_shared plat_mutex_lock
    #define plat_mutex_unlock_shared plat_mutex_unlock
    """)
    prefix = replace(prefix, "    **hold = (VkMemSpanCapture){*span, live_mapping, addr};",
        "    **hold = (VkMemSpanCapture){*span, live_mapping, addr, size, &imported};")
    prefix = replace(prefix, "    assert(!*out && addr == hold->address && size == hold->span.avail);",
        "    assert(!*out && addr >= hold->address && addr - hold->address <= hold->size &&\n"
        "           size <= hold->size - (addr - hold->address));")
    prefix = replace(prefix, "static int vk_mem_span_input_cpu_visible(const VkMemSpanCapture *hold) { assert(hold); return 0; }",
        "static int vk_mem_span_input_cpu_visible(const VkMemSpanCapture *hold) { assert(hold); return cross_scene_fixture; }")
    prefix = replace(prefix, "    assert(span && addr && size);\n    return 0;\n}",
        "    assert(span && addr && size);\n    return cross_scene_fixture;\n}")
    prototypes = (COMBINED / "vk_uniform_payload.h").read_text()
    prototypes = prototypes[prototypes.index("typedef struct VkUniformPayload"):prototypes.index("#endif")]
    prefix = replace(prefix, "static RcRegistryResult vk_usnap_buffer_fragments(",
        "typedef struct VkRecordingPartDisposition VkRecordingPartDisposition;\n" + prototypes + """
    VkRecordingPartDisposition *vk_recording_part_disposition_capture(uint64_t);
    int vk_recording_part_disposition_handed(const VkRecordingPartDisposition *);
    void vk_recording_part_disposition_release(VkRecordingPartDisposition *);
    VkResult vk_replay_metadata_allocate(size_t, int, RcBudgetProducer, RcBudgetTicket *, void **);
    void vk_replay_metadata_free(void *, RcBudgetTicket *);
    RcRegistryResult vk_mem_span_buffer_fragments_with_payloads(const VkMemSpanCapture *, const VkMemSpan *,
        uint64_t, uint64_t, RcRegistryGpuCapture, RcRegistryPosition, const VkMemBufferInlineProof *,
        uint32_t, RcRegistryBufferFragment *, RcRegistryBufferPayload **, uint32_t, uint32_t *);
    static RcRegistryResult vk_usnap_buffer_fragments_fixture(""")
    prefix = replace(prefix,
        "    assert(address && size && out && capacity && count && !rc_registry_position_compare(read, vk_exec_resource_position()));",
        """    assert(address && size && out && capacity && count && !rc_registry_position_compare(read, vk_exec_resource_position()));
        if (cross_scene_fixture)
            return vk_mem_span_buffer_fragments_with_payloads(hold, span, address, size, U.producer.read, read,
                proofs, proof_count, out, NULL, capacity, count);""")
    prefix = replace(prefix, "    *location = (RcRegistryBufferLocation){span->buffer, live_mapping, span->offset};",
        "    *location = (RcRegistryBufferLocation){span->buffer, cross_scene_fixture ? 0 : live_mapping, span->offset};")
    prefix = replace(prefix, "    assert(ex == &recorder && !visible && count == 1 && ranges->address == addr &&",
        "    assert(ex == &recorder && visible == cross_scene_fixture && count == 1 && ranges->address == addr &&")
    prefix = replace(prefix, "static void commit_recorded_writes(VkExec *ex) {",
        "static void commit_recorded_writes_payload(VkExec *ex, RcRegistryBufferPayload *payload) {")
    prefix = replace(prefix, "    OK(rc_registry_buffer_write_record(registry, update_ticket, update_native,\n        (RcRegistryPosition){ex->fence, ex->resource_op * 2 + 1}));",
        """    const RcRegistryPosition output = {ex->fence, ex->resource_op * 2 + 1};
        if (payload) OK(rc_registry_buffer_write_payload_record(registry, update_ticket, update_native, output, payload));
        else OK(rc_registry_buffer_write_record(registry, update_ticket, update_native, output));""")
    prefix = replace(prefix, "static int vk_mem_span_buffer_location(",
        "static int span_capture_match(const VkMemSpanCapture *hold, const VkMemSpan *span) {\n"
        "    return vk_mem_span_capture_current(hold, span);\n}\n"
        "static int vk_mem_span_buffer_location(")

    draw = (COMBINED / "vk_exec_draw.c").read_text()
    usnap = (COMBINED / "vk_usnap.c").read_text()
    memory = (COMBINED / "vk_memory.c").read_text()
    sync = (COMBINED / "vk_sync.c").read_text()
    header = (COMBINED / "vk_exec.h").read_text()
    effects = (COMBINED / "resource_command_effects.c").read_text()
    texture = (COMBINED / "vk_texture.c").read_text()
    part_layout = (COMBINED / "vk_texture_uses.h").read_text()
    effect_header = (COMBINED / "resource_command_effects.h").read_text()
    effect_types = """
    typedef struct ResourceCommandEffects ResourceCommandEffects;
    typedef struct SharedImageClaim SharedImageClaim;
    typedef struct { ResourceCommandEffects *owner; } ResourceImageEffectRole;
    typedef int (*ResourceBufferPreserveCheck)(void *, RcRegistryBufferSpan, RcRegistryPosition, int *);
    """
    effect_types += effects[effects.index("typedef struct EffectBufferPreservation"):
                            effects.index("typedef struct SharedImageClaim")]
    effect_types += effects[effects.index("typedef struct EffectClaim"):
                            effects.index("enum { IMAGE_TRACE_SLOTS")]
    effect_types += "\nstatic EffectActual fixture_actual;\nstatic EffectClaim fixture_claim;\n"
    effect_types += "static ResourceCommandEffects fixture_effects;\n"
    prefix = replace(prefix, "static void resource_effects_buffer_prepare_native(",
        effect_types + "static void resource_effects_buffer_prepare_native(")
    prefix = replace(prefix, "    update_native = (RcRegistryBufferSpan){*ranges, location};",
        """    update_native = (RcRegistryBufferSpan){*ranges, location};
        fixture_actual = (EffectActual){.interval = *ranges, .prepared = 1, .visible = cross_scene_fixture,
            .native_buffer = location};
        fixture_claim = (EffectClaim){.effect = *ranges, .descriptor = *ranges, .kind = RC_REGISTRY_BUFFER,
            .ticket = update_ticket, .actual = &fixture_actual};
        fixture_effects = (ResourceCommandEffects){.position = {recorder.fence, recorder.resource_op * 2 + 1},
            .claims = &fixture_claim};
        current = &fixture_effects;""")
    prefix = replace(prefix, function(prefix, "commit_recorded_writes_payload"), r"""
    static void resource_effects_opaque_commit(void) {}
    static void effect_gpa_commit(void) {}
    static void effect_capture_dump(void) {
        assert(copies == 1 && update_native.native.allocation == written_buffer && fixture_actual.recorded);
        update_ticket = (RcRegistryWrite){0};
        update_native = (RcRegistryBufferSpan){0};
        current = NULL;
        commits++;
    }
    """)
    prefix += extract(effects, "resource_effects_buffers_commit_payload")
    graphics = replace(GRAPHICS_STUBS, "    CbRes *graphics_prev, *graphics_next;",
        "    CbRes *graphics_prev, *graphics_next;\n"
        "    unsigned committed_written, nwritten, committed_gpa;\n    void *gpa_writes;")
    part_start = part_layout.index("enum { TEXTURE_BATCH_RECORDING")
    part_end = part_layout.index("\n", part_start)
    part = part_layout[part_start:part_end] + "\n"
    part_start = part_layout.index("typedef struct VkRecordingPartDisposition {")
    part_end = part_layout.index("} TextureUsePartDisposition;", part_start) + len("} TextureUsePartDisposition;")
    part += part_layout[part_start:part_end] + "\n"
    part += (ROOT / "tests/ordered_uniform_payload_part.inc").read_text()
    part += extract(part_layout, "texture_use_part_retain")
    part += extract(texture, "vk_recording_part_disposition_capture", "vk_recording_part_disposition_handed",
                    "vk_recording_part_disposition_release")

    payload = (COMBINED / "vk_uniform_payload.inc").read_text()
    payload = payload[payload.index("struct VkUniformPayload"):]
    adapters = (ROOT / "tests/ordered_uniform_payload_custody.inc").read_text()
    journal_start = draw.index("// Each pointer-read payload names a native inline producer")
    journal_end = draw.index("static int uniform_pointer_value(", journal_start)
    fragment_end = header.index("} VkExecUniformUpdateFragment;") + len("} VkExecUniformUpdateFragment;")
    fragment_start = header.rindex("typedef struct {", 0, fragment_end)
    definitions_start = draw.index("typedef struct {\n    VkBuffer buf;")
    graphics_start = draw.index("struct VkExecGraphicsScratch {", definitions_start)
    graphics_end = draw.index("\n};", graphics_start) + len("\n};")
    vk = (COMBINED / "vk.h").read_text()
    proof_start = vk.index("enum { VK_GRAPHICS_UNIFORM_BINDINGS")
    proof_end = vk.index("} VkGraphicsUniformFootprints;", proof_start) + len("} VkGraphicsUniformFootprints;")
    fixture = prefix + payload + vk[proof_start:proof_end] + "\n" + header[fragment_start:fragment_end] + "\n" + draw[journal_start:journal_end]
    fixture += draw[definitions_start:graphics_end] + graphics + adapters + part
    fixture += extract(draw, "commit_recorded_writes_payload")
    fixture += extract(memory, "vk_mem_span_buffer_fragments_with_payloads", "vk_mem_span_buffer_fragments")
    fixture += extract(usnap, "vk_usnap_buffer_fragments_with_payloads")
    start = draw.index("struct GraphicsUboOverflow {")
    fixture += draw[start:draw.index("static void graphics_ubo_overflow_release(", start)]
    fixture += r"""
    typedef struct { uint64_t address, bytes, alignment; } UniformAddressRange;
    static uint32_t uniform_effect_opcode(const VkExec *ex, RcRegistryPosition read) {
        (void)ex; (void)read; return 0;
    }
    static uint64_t vk_usnap_capture(void) { return 0; }
    """
    fixture += extract(draw, "storage_uniform_address_range", "readonly_storage_uniform_range")
    fixture += extract(usnap, "first_above", "region_permission", "vk_usnap_resolve_at", "lease_region",
                       "vk_usnap_lease_valid_at", "vk_usnap_drop", "host_copy", "vk_usnap_map")
    fixture += extract(draw,
        "limits_init", "update_begin", "update_emit", "graphics_payload_align", "graphics_payload_demand_at",
        "graphics_payload_demand", "ubo_merge_find", "graphics_ubo_merge_find", "graphics_ubo_footprint_find",
        "graphics_ubo_overflow_release", "graphics_ubo_overflow_allocate", "graphics_ubo_capacity",
        "graphics_ubo_fragments", "graphics_ubo_sealed", "graphics_ubo_gaps", "graphics_ubo_sealed_gaps",
        "graphics_ubo_fragment_payloads_release", "graphics_ubo_merge_release", "ubo_merges_release",
        "fixed_peak_add", "ubo_merge_budget_peak", "graphics_ubo_merges_release",
        "graphics_ubo_current_fragments_collect", "graphics_ubo_payload_fragments", "graphics_ubo_fragments_collect",
        "graphics_ubo_merge_matches", "graphics_ubo_merge_current", "uniform_binding_range", "uniform_range_requires_order", "graphics_uniform_footprint",
        "ordered_buffer_demand", "ubo_merge_select", "ubo_merge_plan_for", "ubo_merge_plan", "graphics_ubo_merge_plan", "graphics_payload_fit", "graphics_payload_view",
        "uniform_native_descriptor", "uniform_native_update_descriptor", "uniform_resolve_name", "uniform_input_refused", "uniform_buffer_descriptor",
        "effect_capture_uniform", "effect_input_current", "update_current", "graphics_unlink_owner", "graphics_detach",
        "graphics_slab_adopt", "ubo_merges_initialize", "graphics_payload_initialize", "graphics_scratch_claim",
        "vk_exec_draw_graphics_scratch_finish", "graphics_payload_plan", "vk_exec_draw_graphics_scratch_prepare",
        "compute_ubo_merges_current", "ubo_merges_record_native", "dispatch_materialize_ubo_merges").replace(
            "vk_usnap_buffer_fragments(", "vk_usnap_buffer_fragments_fixture(")
    fixture += extract(sync, "pending_disposable", "pending_dispose")
    cases = CASES
    cases = replace(cases, "    input = range;", "    input = range;\n    U.fence = range.scene;")
    cases = replace(cases, "int main(void) {",
        (ROOT / "tests/ordered_uniform_payload_cases.inc").read_text() + "\nint main(void) {")
    cases = replace(cases, "    many_ordered_fragments();", "    many_ordered_fragments();\n    cross_scene_cases();")
    fixture += cases

    cc = os.environ.get("CC") or shutil.which("clang")
    if not cc:
        raise SystemExit("Clang is required (set CC)")
    compiler = Path(shutil.which(cc) or cc).resolve()
    env = dict(os.environ, PATH=str(compiler.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="ordered-uniform-merge-") as temporary:
        code = Path(temporary) / "check.c"
        exe = code.with_suffix(".exe") if os.name == "nt" else code.with_suffix("")
        code.write_text(fixture, encoding="utf-8")
        subprocess.run([str(compiler), "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", "-I" + str(args.runtime.resolve()),
                        str(code), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()

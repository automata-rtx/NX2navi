"""Exercise the exact native compute C0/support receipt before effect declaration."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(os.environ.get("NX2_SCRATCH_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
SOURCE = Path(os.environ.get("NX2_SCRATCH_SOURCE", ROOT / "runtime"))
DRAW = SOURCE / "vk_exec_draw.c"


def extract(source, name):
    match = re.search(r"^(?:static )?[^\n]*\b" + name + r"\([^;{]*\)\s*\{", source, re.M)
    if not match:
        raise ValueError(name)
    return source[match.start():source.index("\n}", match.end()) + 2]


def section(source, start, end):
    at = source.index(start)
    return source[at:source.index(end, at + len(start))]


def main():
    source = DRAW.read_text()
    replay = (SOURCE / "vk_exec.c").read_text()
    defines = "\n".join(line for line in source.splitlines()
                        if line.startswith(("#define MAX_BINDINGS ", "#define CHUNK_BYTES ",
                                            "#define SUPPORT_BYTES ", "#define NVN_STAGE_CS ")))
    definitions = section(source, "typedef struct {\n    VkBuffer buf;\n    VmaAllocation alloc;",
                          "struct VkExecGraphicsScratch {")
    fixture = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "vk_exec.h"
#include "resource_core_vma.h"
#include "vk_gpa.h"
''' + defines + "\n" + definitions + r'''
struct CbRes { Chunk *chunks; int nchunks, cchunks; UpdateSlab *graphics_slabs; };
struct DrawCtx { CbRes *res; uint64_t res_fence; };
static CbRes cb_owner;
static struct DrawCtx *dc;
typedef struct { VkExecBufferScratchReceipt *buffer_scratch; VkPipeline compat_compute_pipeline;
                 uint32_t opcode; const GfxShader *compat_compute_shader; } EffectDeclaration;
static EffectDeclaration *effect_current;
static VkExecDispatch *dispatch_current;
static VkExec *current;
static unsigned parks, failures, counted, copied, barriers;
static int parked, recover_host;
static VkBufferCopy last_copy;
static VkBuffer from_buffer, to_buffer;
static VkCtx context;
static RcAdmission ledger;
static unsigned native_allocations, native_refusals, released, cancellations, materializations, destroyed, fills, legacy_drops;
static int refused, host_refused, covered_current = 1, alternate_nvn2, stopping;
static Chunk spare;
static uint8_t mapping[CHUNK_BYTES];
volatile int32_t vk_lost;
static const uint8_t control = 37;

VkCtx *vk_ctx(void) { return &context; }
_Noreturn void rt_fatal(const char *format, ...) { (void)format; abort(); }
int vk_sync_stopping(void) { return stopping; }
int vk_exec_recording_failed(void) { return 0; }
void vk_device_lost_check(VkResult result, const char *operation) { assert(result != VK_ERROR_DEVICE_LOST && operation); }
static int c0_on(void) { return 1; }
static int pool_on(void) { return 1; }
static int nvn_stage_of(int stage) { (void)stage; abort(); }
static int shader_skipped(const GfxShader *shader) { assert(shader); return 0; }
RcRegistryPosition vk_exec_resource_position(void) {
    return (RcRegistryPosition){current->fence, current->resource_op * 2};
}
static int dispatch_read_current(VkExec *ex, const VkExecDispatch *dispatch) {
    return covered_current && ex == current && ex->prog == dispatch->program &&
        !rc_registry_position_compare(dispatch->read_position, vk_exec_resource_position()) &&
        (dispatch->opcode != OP_DISPATCH_IND || dispatch->indirect_hold);
}
static int find_stage(struct DrawCtx *draw, const prog_t *program, int stage, const uint8_t **ctl,
                      const uint8_t **code, uint32_t *length) {
    assert(draw && program && stage == NVN_STAGE_CS && code && length);
    *ctl = &control;
    return 1;
}
static int ctl_nvn2(const uint8_t *ctl) { assert(ctl == &control); return alternate_nvn2; }
static void c0_capture_bytes(uint8_t bytes[VK_C0_BYTES], const VkExec *ex, int st, const uint32_t groups[3],
                             uint32_t instance, int nvn2, const uint8_t *ctl) {
    assert(ex && st == NVN_STAGE_CS && !instance && nvn2 == alternate_nvn2 && ctl == &control);
    memset(bytes, 0x55, VK_C0_BYTES);
    memcpy(bytes + (nvn2 ? 0 : 0x200), groups, 3 * sizeof *groups);
    fills++;
}
static CbRes *cb_res(VkExec *ex) {
    ex->dc->res = &cb_owner;
    ex->dc->res_fence = ex->fence;
    return &cb_owner;
}
static void *grow(void *data, int *capacity, size_t item) {
    *capacity += 2;
    void *next = realloc(data, (size_t)*capacity * item);
    assert(next);
    return next;
}
static void scratch_chunk_specs(uint64_t size, VkBufferCreateInfo *buffer, VmaAllocationCreateInfo *placement) {
    assert(size == CHUNK_BYTES);
    *buffer = (VkBufferCreateInfo){.size = size};
    *placement = (VmaAllocationCreateInfo){0};
}
static int scratch_take_spare(uint64_t bytes, Chunk *out) {
    if (!spare.alloc || bytes != CHUNK_BYTES) return 0;
    *out = spare;
    spare = (Chunk){0};
    return 1;
}
static void scratch_return_chunk(VmaAllocator allocator, const Chunk *chunk) {
    assert(allocator == context.vma && !spare.alloc && chunk->alloc && chunk->size == CHUNK_BYTES);
    spare = *chunk;
    released++;
}
void resource_core_budget_readiness(int required, RcBudgetReadiness *out) {
    assert(required);
    *out = (RcBudgetReadiness){.records_available = 1, .usable_mask = 3};
    if (!host_refused)
        out->available.amount[RC_BUDGET_RAM] = out->available.amount[RC_BUDGET_COMMIT] = sizeof(VkExecBufferScratchReceipt);
}
RcAdmissionStatus resource_core_budget_reserve(RcBudgetProducer producer, int required, const RcAdmissionVector *host,
                                               RcBudgetTicket *ticket) {
    assert(producer == RC_BUDGET_BUFFER && required && !ticket->token.ledger);
    assert(host->amount[RC_BUDGET_COMMIT] == sizeof(VkExecBufferScratchReceipt));
    assert(host->amount[RC_BUDGET_RAM] == sizeof(VkExecBufferScratchReceipt));
    for (uint32_t d = RC_BUDGET_GPU_FIRST; d < RC_ADMISSION_MAX_DOMAINS; ++d)
        assert(!host->amount[d]);
    counted++;
    if (host_refused) return RC_ADMISSION_REFUSED;
    ticket->token.ledger = &ledger;
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_materialize(RcBudgetTicket *ticket, const RcAdmissionVector *host) {
    assert(ticket->token.ledger == &ledger && host->amount[RC_BUDGET_RAM] == sizeof(VkExecBufferScratchReceipt));
    materializations++;
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_cancel(RcBudgetTicket *ticket) {
    assert(ticket->token.ledger == &ledger);
    cancellations++;
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_retire(RcBudgetTicket *ticket) {
    assert(ticket->token.ledger == &ledger && ticket->token.state != RC_ADMISSION_PENDING);
    ticket->token.state = RC_ADMISSION_PENDING;
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_destroyed(RcBudgetTicket *ticket) {
    assert(ticket->token.state == RC_ADMISSION_PENDING);
    assert(ticket->token.ledger == &ledger);
    destroyed++;
    return RC_ADMISSION_OK;
}

VkResult resource_core_vma_buffer_peak(const RcVmaRequest *request, const VkBufferCreateInfo *buffer,
        const VmaAllocationCreateInfo *allocation, VmaAllocationCreateInfo *placement, RcAdmissionVector *peak) {
    assert(request->producer == RC_BUDGET_BUFFER && request->required && request->dependency && buffer->size == CHUNK_BYTES);
    *placement = *allocation;
    placement->memoryTypeBits = 4;
    *peak = (RcAdmissionVector){0};
    peak->amount[RC_BUDGET_COMMIT] = CHUNK_BYTES;
    request->dependency->selected = 1;
    request->dependency->memory_type = 2;
    request->dependency->peak = *peak;
    return VK_SUCCESS;
}
VkResult resource_core_vma_create_buffer(const RcVmaRequest *request, VmaAllocator allocator,
        const VkBufferCreateInfo *buffer, const VmaAllocationCreateInfo *placement, VkBuffer *out,
        VmaAllocation *memory, VmaAllocationInfo *info, RcAdmissionStatus *admission) {
    assert(request->required && request->producer == RC_BUDGET_BUFFER && allocator == context.vma);
    assert(placement->memoryTypeBits == 4 && buffer->size == CHUNK_BYTES && request->host_bytes == sizeof(VkExecBufferScratchReceipt));
    assert(request->refused_peak && !*out && !*memory);
    request->refused_peak->amount[RC_BUDGET_COMMIT] = CHUNK_BYTES + request->host_bytes;
    if (refused) {
        native_refusals++;
        *admission = RC_ADMISSION_REFUSED;
        return VK_NOT_READY;
    }
    assert(!native_allocations++);
    *out = (VkBuffer)(uintptr_t)500;
    *memory = (VmaAllocation)(uintptr_t)700;
    *info = (VmaAllocationInfo){.pMappedData = mapping};
    request->host_ticket->token.ledger = &ledger;
    *admission = RC_ADMISSION_OK;
    return VK_SUCCESS;
}
RcAdmissionStatus resource_core_vma_allocation_cost(VmaAllocation memory, RcAdmissionVector *cost) {
    if (memory == (VmaAllocation)(uintptr_t)3) return RC_ADMISSION_STALE;
    assert(memory == (VmaAllocation)(uintptr_t)700);
    *cost = (RcAdmissionVector){0};
    cost->amount[RC_BUDGET_COMMIT] = CHUNK_BYTES;
    return RC_ADMISSION_OK;
}
void resource_core_vma_destroy_buffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation memory) {
    assert(allocator == context.vma && buffer == (VkBuffer)(uintptr_t)4 && memory == (VmaAllocation)(uintptr_t)3);
    legacy_drops++;
}
int resource_core_vma_dependency_ready(const RcVmaDependency *dependency, const RcBudgetReadiness *ready) {
    assert(dependency->selected && ready);
    return !refused;
}
int vk_exec_draw_graphics_scratch_spare_ready(void *opaque) {
    const VkDrawScratchWaitContext *wait = opaque;
    assert(wait->ex == current && wait->bytes);
    return spare.alloc != NULL;
}
void vk_mem_collect(void) { assert(parked); }
void resource_core_budget_service(int required, const RcAdmissionVector *peak, RcBudgetProgress *out) {
    assert(required && parked && peak->amount[RC_BUDGET_RAM] == sizeof(VkExecBufferScratchReceipt));
    out->event = 37;
    if (recover_host) host_refused = 0;
}
void resource_core_budget_wait_since(uint64_t event, uint32_t ms) {
    assert(parked && event == 37 && ms == 100);
}
VkSyncProgressResult vk_exec_graphics_progress_prepare(VkExec *ex, int (*body)(void *), void *context) {
    assert(!parked && ex == current && !ex->resource_op);
    parks++;
    parked = 1;
    assert(body(context));
    parked = 0;
    return VK_SYNC_PROGRESS_COMPLETE;
}
void vk_exec_required_graphics_unavailable(VkExec *ex, uint32_t op, const char *reason) {
    assert(ex == current && !parked && !ex->resource_op && op == OP_DISPATCH && reason);
    failures++;
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer cb, VkBuffer src, VkBuffer dst, uint32_t count,
                                          const VkBufferCopy *copy) {
    assert(cb == current->cb && count == 1);
    copied++;
    from_buffer = src;
    to_buffer = dst;
    last_copy = *copy;
}
void vk_exec_barrier(VkExec *ex) {
    assert(ex == current && copied == barriers + 1 && (unsigned)ex->recorded == copied);
    barriers++;
}
'''

    fixture += r"""
struct VkExecDispatchImages { const void *selected_writers, *pointer_sources; uint32_t pointer_count; };
static VkExecDispatchImages selection;
static int selected_storage_writers_device_current(const VkExec *ex, const void *writers, const VkPipeInfo *info,
    RcRegistryPosition read, const void *pointers, uint32_t pointer_count) {
    assert(ex && !writers && info && read.scene == ex->fence && !pointers && !pointer_count);
    return 1;
}
static int dispatch_compat_read_current(const VkExec *ex, const VkExecDispatch *dispatch) {
    assert(dispatch->compatibility);
    return ex == current && !rc_registry_position_compare(vk_exec_resource_position(), dispatch->read_position);
}
static int ubo_merge_plan(VkExec *ex, const VkPipeInfo *info, uint64_t ordinal, GraphicsPayloadDemand *demand,
                          UboMergePlan *plan, RcVmaDependency *dependency) {
    assert(ex && info && ordinal == ex->resource_op && demand->alignment && !plan->count && dependency);
    return 1;
}
static void ubo_merges_release(UboMergePlan *plan) { assert(!plan->count && !plan->entries); }
static int ubo_merge_budget_peak(const UboMergePlan *plan, RcAdmissionVector *peak) {
    assert(!plan->count && peak);
    return 1;
}

static int ubo_merges_initialize(VkExec *ex, const UboMergePlan *plan, uint8_t *map) {
    assert(ex && !plan->count && map);
    return 1;
}
static int compute_ubo_merges_current(const VkExec *ex, const VkExecBufferScratchReceipt *receipt) {
    assert(ex && receipt && !receipt->ubo.count);
    return 1;
}
"""
    functions = ("graphics_payload_align", "graphics_payload_demand_at", "graphics_payload_fit", "fixed_peak_add",
                 "buffer_scratch_view", "buffer_scratch_borrowed", "buffer_scratch_expose", "buffer_scratch_cancel",
                 "buffer_scratch_adopt", "buffer_scratch_recording", "buffer_scratch_indirect_c0",
                 "buffer_scratch_fixed_dispatch", "buffer_scratch_try_borrow", "buffer_scratch_allocate",
                 "vk_exec_draw_buffer_scratch_prepare")
    fixture += "\n".join(extract(source, name) for name in functions)
    fixture += section(replay, "typedef struct {\n    VkDrawScratchWaitContext spare;", "static VkExecDispatchResult prepare_ordered_dispatch(")
    fixture += r'''
int main(void) {
    struct DrawCtx draw = {0};
    prog_t program = {.serial = 9};
    VkExec ex = {.dc = &draw, .fence = 11, .prog = &program, .resource_op = 1,
                 .cb = (VkCommandBuffer)(uintptr_t)12};
    dc = &draw;
    current = &ex;
    p_dispatch direct = {8, 2, 3, 0};
    GfxShaderRes ubo = {.slot = 0};
    VkPipeBinding bindings[] = {
        {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .kind = VK_PIPE_SUPPORT},
        {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .kind = GFX_RES_UBO, .stage = GFX_STAGE_CS, .res = &ubo}
    };
    VkPipeInfo info = {.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE, .n_bindings = 2, .bindings = bindings};
    VkExecDispatch dispatch = {.read_position = {11, 2}, .info = &info, .opcode = OP_DISPATCH, .payload = &direct,
        .program = &program, .program_serial = 9, .shader = (const GfxShader *)(uintptr_t)1,
        .selection = &selection};
    VkExecBufferScratchWait wait;
    context.props.limits.minUniformBufferOffsetAlignment = 1024;
    refused = 1;
    spare = (Chunk){.alloc = (VmaAllocation)(uintptr_t)3, .buf = (VkBuffer)(uintptr_t)4, .size = CHUNK_BYTES};
    assert(vk_exec_draw_buffer_scratch_prepare(&ex, &dispatch, &wait) == VK_EXEC_SCRATCH_NEED_CAPACITY);
    assert(!dispatch.scratch && !fills && !spare.alloc && !cb_owner.nchunks && native_refusals == 1 && legacy_drops == 1);
    assert(wait.native.selected && !wait.host_only && wait.native.before.records_available);
    assert(wait.native.peak.amount[RC_BUDGET_COMMIT] == CHUNK_BYTES + sizeof(VkExecBufferScratchReceipt));

    refused = 0;
    assert(vk_exec_draw_buffer_scratch_prepare(&ex, &dispatch, &wait) == VK_EXEC_SCRATCH_READY);
    VkExecBufferScratchReceipt *receipt = dispatch.scratch;
    assert(receipt && fills == 1 && !cb_owner.nchunks && native_allocations == 1 && materializations == 1);
    VkDescriptorBufferInfo c0, support;
    uint8_t *bank = buffer_scratch_view(receipt, NVN_STAGE_CS, &c0);
    assert(bank && c0.offset == 0 && c0.range == VK_C0_BYTES);
    assert(buffer_scratch_view(receipt, GRAPHICS_PAYLOAD_SUPPORT, &support) && support.offset % 1024 == 0);
    assert(support.offset >= c0.range && support.buffer == c0.buffer);
    assert(!memcmp(bank + 0x200, (uint32_t[]){8, 2, 3}, 12));
    buffer_scratch_cancel(&dispatch.scratch);
    buffer_scratch_cancel(&dispatch.scratch);
    assert(!dispatch.scratch && spare.alloc && released == 1 && destroyed == 1);

    host_refused = 1;
    assert(vk_exec_draw_buffer_scratch_prepare(&ex, &dispatch, &wait) == VK_EXEC_SCRATCH_NEED_CAPACITY);
    assert(!dispatch.scratch && spare.alloc && fills == 1 && wait.host_only && !wait.native.selected);
    assert(wait.host_peak.amount[RC_BUDGET_COMMIT] == sizeof(VkExecBufferScratchReceipt));
    ex.resource_op = 0;
    assert(prepare_dispatch_scratch(&ex, &dispatch, dispatch.read_position) == VK_EXEC_DISPATCH_STOPPED);
    assert(!dispatch.scratch && !ex.resource_op && !ex.resource_next && parks == 1 && failures == 1);
    recover_host = 1;
    assert(prepare_dispatch_scratch(&ex, &dispatch, dispatch.read_position) == VK_EXEC_DISPATCH_NATIVE);
    assert(dispatch.scratch && !ex.resource_op && !ex.resource_next && parks == 2 && failures == 1);

    ex.resource_op = 1;
    receipt = dispatch.scratch;
    EffectDeclaration effect = {
        .buffer_scratch = buffer_scratch_adopt(&dispatch.scratch, &ex, dispatch.opcode, dispatch.payload,
                                               dispatch.shader, (RcRegistryPosition){11, 3})
    };
    assert(effect.buffer_scratch == receipt && !dispatch.scratch);
    assert(!buffer_scratch_adopt(&dispatch.scratch, &ex, dispatch.opcode, dispatch.payload,
                                 dispatch.shader, (RcRegistryPosition){11, 3}));
    assert(!buffer_scratch_recording(&ex));
    effect_current = &effect;
    dispatch_current = &dispatch;
    assert(buffer_scratch_recording(&ex) == receipt);
    buffer_scratch_expose(&ex, receipt);
    buffer_scratch_expose(&ex, receipt);
    assert(cb_owner.nchunks == 1 && !spare.alloc && receipt->exposed);
    buffer_scratch_cancel(&effect.buffer_scratch);
    assert(!spare.alloc && destroyed == 2);

    p_dispatch_ind indirect = {24, 0};
    dispatch.opcode = OP_DISPATCH_IND;
    dispatch.payload = &indirect;
    dispatch.indirect = (VkMemSpan){.buffer = (VkBuffer)(uintptr_t)33, .offset = 60};
    dispatch.indirect_hold = (VkMemSpanCapture *)(uintptr_t)3;
    ex.resource_next = 1;
    ex.resource_op = 2;
    dispatch.read_position.ordinal = 4;
    assert(vk_exec_draw_buffer_scratch_prepare(&ex, &dispatch, &wait) == VK_EXEC_SCRATCH_PASS);
    dispatch.indirect_native = 1;
    for (alternate_nvn2 = 0; alternate_nvn2 <= 1; alternate_nvn2++) {
        uint64_t used = cb_owner.chunks[0].used;
        assert(vk_exec_draw_buffer_scratch_prepare(&ex, &dispatch, &wait) == VK_EXEC_SCRATCH_READY);
        receipt = dispatch.scratch;
        bank = buffer_scratch_view(receipt, NVN_STAGE_CS, &c0);
        uint32_t offset = alternate_nvn2 ? 0 : 0x200;
        assert(bank && c0.offset >= used && c0.offset % 1024 == 0 && receipt->indirect_offset == offset);
        assert(!memcmp(bank + offset, (uint32_t[]){0, 0, 0}, 12));
        assert(receipt->borrowed == &cb_owner && cb_owner.nchunks == 1 && native_allocations == 1);
        effect.buffer_scratch = buffer_scratch_adopt(&dispatch.scratch, &ex, dispatch.opcode, dispatch.payload,
                                                    dispatch.shader, (RcRegistryPosition){11, 5});
        buffer_scratch_expose(&ex, receipt);
        assert(buffer_scratch_indirect_c0(&ex, &dispatch.indirect));
        assert(from_buffer == dispatch.indirect.buffer && to_buffer == c0.buffer && copied == barriers);
        assert(last_copy.srcOffset == 60 && last_copy.dstOffset == c0.offset + offset && last_copy.size == 12);
        buffer_scratch_cancel(&effect.buffer_scratch);
        assert(cb_owner.chunks[0].used > used && !spare.alloc && !cancellations);
    }

    alternate_nvn2 = 1;
    assert(vk_exec_draw_buffer_scratch_prepare(&ex, &dispatch, &wait) == VK_EXEC_SCRATCH_READY);
    uint64_t prior = dispatch.scratch->prior;
    buffer_scratch_cancel(&dispatch.scratch);
    assert(cb_owner.chunks[0].used == prior && native_allocations == 1 && copied == 2 && destroyed == 5);

    UpdateSlab slab = {.chunk = cb_owner.chunks[0]};
    cb_owner.graphics_slabs = &slab;
    cb_owner.chunks[0].used = cb_owner.chunks[0].size;
    prior = slab.chunk.used;
    assert(vk_exec_draw_buffer_scratch_prepare(&ex, &dispatch, &wait) == VK_EXEC_SCRATCH_READY);
    assert(dispatch.scratch->borrowed_slab == &slab && slab.chunk.used > prior);
    buffer_scratch_cancel(&dispatch.scratch);
    assert(slab.chunk.used == prior && native_allocations == 1 && destroyed == 6);

    covered_current = 0;
    assert(vk_exec_draw_buffer_scratch_prepare(&ex, &dispatch, &wait) == VK_EXEC_SCRATCH_INVALIDATED);
    covered_current = 1;
    unsigned prior_count = counted;
    ubo.slot = 1;
    assert(vk_exec_draw_buffer_scratch_prepare(&ex, &dispatch, &wait) == VK_EXEC_SCRATCH_PASS);
    bindings[1].kind = GFX_RES_SSBO;
    assert(vk_exec_draw_buffer_scratch_prepare(&ex, &dispatch, &wait) == VK_EXEC_SCRATCH_PASS);
    dispatch.selection = NULL;
    assert(vk_exec_draw_buffer_scratch_prepare(&ex, &dispatch, &wait) == VK_EXEC_SCRATCH_PASS);
    assert(!dispatch.scratch && counted == prior_count && native_allocations == 1);
    free(cb_owner.chunks);
    return 0;
}
'''

    cc = os.environ.get("CC") or shutil.which("clang")
    if not cc:
        raise SystemExit("Clang is required (set CC)")
    clang = Path(shutil.which(cc) or cc).resolve()
    vma = Path(os.environ.get("VMA_INCLUDE", clang.parent.parent / "include" / "vma"))
    env = dict(os.environ, PATH=str(clang.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="buffer-scratch-receipt-") as temporary:
        folder = Path(temporary)
        test = folder / "check.c"
        executable = folder / ("check.exe" if os.name == "nt" else "check")
        test.write_text(fixture)
        subprocess.run([str(clang), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=undefined",
                        "-I" + str(SOURCE), "-I" + str(ROOT / "runtime"), "-I" + str(vma), str(test), "-o", str(executable)],
                       env=env, check=True)
        subprocess.run([str(executable)], env=env, check=True)
    print("buffer scratch receipt: selected layout, host-only recovery, indirect copy and barrier, reuse and cancellation passed")


if __name__ == "__main__":
    main()

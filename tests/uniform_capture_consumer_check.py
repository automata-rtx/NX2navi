"""Exercise typed UBO descriptors and pointer reads without a live fallback."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(os.environ.get("NX2_UBO_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
DRAW = Path(os.environ.get("NX2_UBO_DRAW", ROOT / "runtime/vk_exec_draw.c"))


def extract(source, name):
    match = re.search(r"^(?:static )?[^\n]*\b" + name + r"\([^;{]*\)\s*\{", source, re.M)
    if not match:
        raise ValueError(name)
    return source[match.start():source.index("\n}", match.end()) + 2]


def main():
    source = DRAW.read_text()
    binding = extract(source, "bind_resources")
    arena = extract(Path(ROOT / "runtime/vk_usnap.c").read_text(), "arena_buffer_info")
    assert "VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT" in arena
    assert "VK_BUFFER_USAGE_STORAGE_BUFFER_BIT" in arena
    assert "VK_BUFFER_USAGE_TRANSFER_SRC_BIT" in arena
    ssbo = extract(source, "ssbo_info")
    assert "if (ubo && b->addr && !written)" in ssbo
    assert "uniform_buffer_descriptor(ex" in ssbo
    assert "bi.offset % align" in ssbo and "bi.range > max" in ssbo
    descriptor = extract(source, "uniform_buffer_descriptor")
    assert "uniform_native_update_descriptor(ex, addr, size, alignment, read, use, out)" in descriptor
    for path in ("prepared_resolve(", "ubo_info(ex,"):
        lookup = binding.index(path)
        guard = binding.index("vk_exec_recording_failed()", lookup)
        cancel = binding.index("binding_manifest_cancel(ex)", guard)
        assert lookup < guard < cancel < binding.index("binding_manifest_publish(ex)")
    optional = extract(source, "mo_desc")
    assert optional.index("if (!o[k].buffer.buffer)") < optional.index("binding_manifest_publish(ex)")
    fixture = r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "vk_exec.h"
#include "platform/platform.h"

#undef VK_MEM_BOUND
#define VK_MEM_BOUND(span) do { assert((span).buffer); } while (0)

enum { NVN_STAGE_VS = 0, NVN_STAGE_CS = 5 };
typedef struct DrawCtx {
    int pb_spoiled, mo_ovr_on;
    VkUsnapLease *prep_capture;
    VkDescriptorBufferInfo mo_ovr[8][32];
    struct { uint8_t c0[3][VK_C0_BYTES]; } B;
} DrawCtx;
typedef struct {
    uint64_t address, bytes, descriptor_address, descriptor_bytes;
    VkDeviceSize descriptor_offset;
    VkDeviceSize offset;
    unsigned stage, slot;
    uint32_t native_count;
    const VkPipeBinding *descriptor_binding;
} GraphicsUboMerge;
typedef struct {
    struct { VkBuffer buf; uint8_t *map; } chunk;
} MergeSlab;
struct VkExecGraphicsScratch {
    MergeSlab *slab;
    VkDeviceSize start;
    GraphicsUboMerge *ubo_merges;
    uint32_t ubo_merge_count;
    int exposed, native_materialized;
    const void *vertex_uniforms;
};
struct VkExecBufferScratchReceipt {
    struct { GraphicsUboMerge *entries; uint32_t count; } ubo;
    struct { VkBuffer buf; uint8_t *map; } chunk;
    VkDeviceSize start;
    int ubo_materialized, collecting;
};
static VkExecGraphicsScratch *graphics_payload_current, *graphics_payload_bound;
static uint8_t snapshots[64], native_bytes[64], merged_bytes[64];
static uint64_t span_size;
static unsigned queries, maps, selectors, records, retained, released, refused, read_begins, read_ends, publications;
static unsigned native_update_queries, native_proofs, merge_checks, merged_inputs, terminal_probes;
static RcRegistryResult native_proof = RC_REGISTRY_OK;
static int recording_failed, native_current = 1, lease_valid = 1, reclassify;
static int native_update_ready, merge_current;
static VkUsnapResolveResult resolve_result;
static VkExecUniformUpdateRead update_result;
static VkExecHostInputResult host_result = VK_EXEC_HOST_INPUT_READY, finish_result = VK_EXEC_HOST_INPUT_READY;
static struct { uint64_t ubo_max, ubo_align; } LIM = {32, 16};
static const VkBuffer native_buffer = (VkBuffer)(uintptr_t)0x25, snap_buffer = (VkBuffer)(uintptr_t)0x56;
static const VkBuffer merged_buffer = (VkBuffer)(uintptr_t)0x72;

static void count_use(DrawCtx *draw, int cls, int use, uint64_t size) {
    assert(draw && (cls == VK_MCLS_SCRATCH || cls == VK_MCLS_HOST_CACHED) && use == VK_MUSE_UBO && size == 32);
}
void rt_log(const char *format, ...) { if (strstr(format, "required uniform input")) terminal_probes++; }
_Noreturn void rt_fatal(const char *format, ...) { (void)format; abort(); }
int vk_exec_recording_failed(void) { return recording_failed; }
void vk_exec_required_graphics_unavailable(VkExec *ex, uint32_t op, const char *reason) {
    assert(ex && op == 99 && reason);
    recording_failed = 1;
    refused++;
}
RcRegistryPosition vk_exec_resource_position(void) { return (RcRegistryPosition){5, 18}; }

VkUsnapResolveResult vk_usnap_resolve_at(uint64_t addr, uint64_t n, uint64_t align, RcRegistryPosition pos,
                                        VkDescriptorBufferInfo *out, VkUsnapLease *lease) {
    assert(addr && n == 32 && (align == 1 || align == 16) && pos.scene == 5 && pos.ordinal == 18);
    queries++;
    if (resolve_result == VK_USNAP_RESOLVE_READY) {
        *out = (VkDescriptorBufferInfo){snap_buffer, 16, n};
        if (lease) *lease = (VkUsnapLease){.capture = 7, .addr = addr, .size = n};
    }
    return resolve_result;
}
const uint8_t *vk_usnap_map(void) { maps++; return snapshots; }
int vk_usnap_lease_valid_at(const VkUsnapLease *lease, RcRegistryPosition read) {
    return lease->capture == 7 && read.scene == 5 && lease_valid;
}
RcRegistryResult vk_mem_span_native_input(uint64_t addr, uint64_t size, int clamp,
                                          VkMemSpan *span, VkMemSpanCapture **hold) {
    assert(addr && size == 32 && !clamp && !*hold);
    selectors++;
    if (reclassify) resolve_result = VK_USNAP_RESOLVE_MAPPING_STALE;
    span_size = size;
    *hold = (VkMemSpanCapture *)(uintptr_t)3;
    *span = (VkMemSpan){.buffer = native_buffer, .offset = 16, .avail = size, .cls = VK_MCLS_HOST_CACHED};
    return RC_REGISTRY_OK;
}
int vk_mem_span_capture_current(const VkMemSpanCapture *hold, const VkMemSpan *span) {
    assert(hold && span->buffer == native_buffer && span->avail == span_size);
    return native_current;
}
int vk_mem_span_record_input(VkMemSpanCapture *hold, const VkMemSpan *span) {
    assert(vk_mem_span_capture_current(hold, span));
    records++;
    return 1;
}
uint32_t vk_mem_span_input_intervals(const VkMemSpanCapture *hold, uint64_t addr, uint64_t size,
                                     RcInterval **ranges) {
    assert(hold && addr && size == span_size && !*ranges);
    *ranges = malloc(sizeof **ranges);
    assert(*ranges);
    **ranges = (RcInterval){addr, 11, {17, 16, size}};
    return 1;
}
RcRegistry *rc_registry_runtime(void) { return (RcRegistry *)(uintptr_t)1; }
RcRegistryResult rc_registry_read_device_buffer_at(RcRegistry *registry, const RcInterval *ranges,
                                                   uint32_t count, RcRegistryPosition read) {
    assert(registry == rc_registry_runtime() && count == 1 && ranges->address && ranges->mapping == 11 &&
           ranges->range.backing == 17 && ranges->range.offset == 16 && ranges->range.size == span_size &&
           read.scene == 5 && read.ordinal == 18);
    native_proofs++;
    return native_proof;
}
void vk_exec_record_native_input_provider(VkExec *ex, VkMemSpanCapture **hold) {
    assert(ex && *hold && records);
    retained++;
    *hold = NULL;
}
void vk_mem_span_capture_release(VkMemSpanCapture **hold) {
    assert(*hold);
    *hold = NULL;
    released++;
}
VkExecHostInputResult vk_exec_host_input_begin(const VkMemSpanCapture *span, const VkMemSpan *selected,
    uint64_t address, uint64_t size, RcRegistryPosition read, VkExecHostInputRead *out) {
    assert(span && selected->buffer == native_buffer && address && size == sizeof(uint64_t) && span_size == 32 && read.scene == 5);
    read_begins++;
    if (host_result == VK_EXEC_HOST_INPUT_READY)
        *out = (VkExecHostInputRead){.bytes = native_bytes + 8, .opaque = (void *)(uintptr_t)1};
    return host_result;
}
VkExecHostInputResult vk_exec_host_input_finish(VkExecHostInputRead *input) {
    assert(input->opaque && input->bytes == native_bytes + 8);
    read_ends++;
    *input = (VkExecHostInputRead){0};
    return finish_result;
}
void vk_exec_host_input_cancel(VkExecHostInputRead *input) {
    assert(input->opaque);
    *input = (VkExecHostInputRead){0};
}
VkExecUniformUpdateRead upd_read_uniform(const VkExec *ex, uint64_t address, uint64_t bytes, void *out) {
    assert(ex && address && bytes == sizeof(uint64_t));
    if (update_result == VK_EXEC_UNIFORM_UPDATE_READY) {
        uint64_t value = UINT64_C(0x1234);
        memcpy(out, &value, bytes);
    }
    return update_result;
}

static int c0_on(void) { return 0; }
static VkDescriptorBufferInfo c0_info(VkExec *ex, int stage, int required) { (void)ex; (void)stage; (void)required; abort(); }
static uint8_t *graphics_payload_view(const VkExecGraphicsScratch *scratch, unsigned slot, VkDescriptorBufferInfo *out) {
    (void)scratch; (void)slot; (void)out; abort();
}
static int graphics_ubo_merge_current(const VkExec *ex, const GraphicsUboMerge *merge) {
    assert(ex->fence == 5 && ex->resource_op == 9 && merge == graphics_payload_bound->ubo_merges);
    merge_checks++;
    return merge_current;
}
static VkExecBufferScratchReceipt *buffer_scratch_recording(const VkExec *ex) {
    (void)ex; abort();
}
static uint8_t *buffer_scratch_view(const VkExecBufferScratchReceipt *receipt, unsigned slot, VkDescriptorBufferInfo *out) {
    (void)receipt; (void)slot; (void)out; abort();
}
static void buffer_scratch_expose(VkExec *ex, VkExecBufferScratchReceipt *receipt) {
    (void)ex; (void)receipt; abort();
}
static int pool_index_bind_input(VkExec *ex, int stage, int slot, const GfxShaderRes *resource,
                                 VkDescriptorBufferInfo *out) {
    assert(ex && stage == NVN_STAGE_VS && slot == 4 && !resource && out);
    return 0;
}
static int effect_uniform_descriptor(VkExec *ex, int st, int slot, const VkPipeBinding *binding,
                                     VkDescriptorBufferInfo *out) {
    assert(binding && binding->set == 1 && binding->binding == 7);
    assert(ex && st == NVN_STAGE_VS && slot == 4 && out);
    return 0;
}
static int uniform_native_update_descriptor(VkExec *ex, uint64_t addr, uint64_t size, uint64_t alignment,
                                            RcRegistryPosition read, int use, VkDescriptorBufferInfo *out) {
    assert(ex && addr && size == 32 && alignment == 16 && read.scene == 5 && use == VK_MUSE_UBO && out);
    native_update_queries++;
    if (!native_update_ready) return 0;
    *out = (VkDescriptorBufferInfo){native_buffer, 16, size};
    return 1;
}
static void effect_uniform_record(VkExec *ex, int st, int slot, VkDescriptorBufferInfo out, const uint8_t *bytes) {
    assert(ex && st == NVN_STAGE_VS && slot == 4);
    if (out.buffer == merged_buffer) {
        assert(graphics_payload_bound->exposed && out.offset == 32 && out.range == 32 && bytes == merged_bytes + 32);
        merged_inputs++;
    } else assert(out.buffer == snap_buffer && out.offset == 16 && bytes == snapshots + 16);
}
static void limits_init(void) { abort(); }
static int find_stage(DrawCtx *draw, const prog_t *prog, int st, const uint8_t **ctl, const uint8_t **code, uint32_t *len) {
    (void)draw; (void)prog; (void)st; (void)ctl; (void)code; (void)len; abort();
}
static uint32_t ctl_c1(const uint8_t *ctl, uint32_t len) { (void)ctl; (void)len; abort(); }
static int ctl_nvn2(const uint8_t *ctl) { (void)ctl; abort(); }
static VkDescriptorBufferInfo zero_info(uint64_t max) { (void)max; abort(); }
static int bind_range(VkExec *ex, uint64_t address, uint64_t size, uint64_t align, uint64_t max,
                      int use, int written, int required, VkDescriptorBufferInfo *out) {
    (void)ex; (void)address; (void)size; (void)align; (void)max; (void)use; (void)written; (void)required; (void)out;
    abort();
}

typedef struct { uint64_t address, bytes, alignment; } UniformAddressRange;
static const VkPipeBinding fixture_binding = {.kind = GFX_RES_UBO, .set = 1, .binding = 7,
                                               .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
static uint32_t uniform_effect_opcode(const VkExec *ex, RcRegistryPosition read) {
    assert(ex && read.scene == 5 && read.ordinal == 18);
    return 99;
}
uint64_t vk_usnap_capture(void) { return 7; }
'''
    names = ("ubo_merge_find", "graphics_ubo_merge_find", "graphics_ubo_footprint_find", "uniform_native_descriptor", "uniform_resolve_name", "uniform_input_refused", "uniform_buffer_descriptor",
             "ubo_info_for", "ubo_info", "uniform_pointer_value", "ubo_ptr")
    fixture += "\n".join(extract(source, name) for name in names)
    fixture += r'''
static VkDescriptorBufferInfo consume(VkExec *ex) {
    VkDescriptorBufferInfo descriptor = ubo_info(ex, &fixture_binding, NVN_STAGE_VS, 4);
    if (!vk_exec_recording_failed() && descriptor.buffer) publications++;
    return descriptor;
}

int main(void) {
    DrawCtx draw = {0};
    VkExec ex = {.dc = &draw, .fence = 5, .resource_op = 9};
    GfxShaderRes pointer = {.cbuf_slot = 4, .handle = 2};
    uint8_t guest[32] = {0};
    ex.ubo[NVN_STAGE_VS][1] = (p_bind_buf){.addr = (uintptr_t)guest, .size = sizeof guest};
    const VkUsnapResolveResult unavailable[] = {
        VK_USNAP_RESOLVE_NO_CAPACITY, VK_USNAP_RESOLVE_MAPPING_STALE,
        VK_USNAP_RESOLVE_UNAVAILABLE, VK_USNAP_RESOLVE_ORDERED_PRODUCER, VK_USNAP_RESOLVE_SOURCE_ADVANCED,
        VK_USNAP_RESOLVE_ALIGNMENT
    };
    uint64_t value;
    for (unsigned i = 0; i < sizeof unavailable / sizeof unavailable[0]; i++) {
        resolve_result = unavailable[i];
        assert(!consume(&ex).buffer && recording_failed && draw.pb_spoiled);
        recording_failed = 0;
        assert(!ubo_ptr(&ex, NVN_STAGE_VS, &pointer, &value) && !value);
    }
    assert(!maps && !selectors && !read_begins && !publications && refused == 6);

    resolve_result = VK_USNAP_RESOLVE_ORDERED_PRODUCER;
    native_update_ready = 1;
    assert(consume(&ex).buffer == native_buffer && !recording_failed && publications == 1 && native_update_queries == 3);
    native_update_ready = 0;

    draw.mo_ovr_on = 1;
    assert(!ubo_info(&ex, &fixture_binding, NVN_STAGE_VS, 4).buffer && !recording_failed && refused == 6);
    draw.mo_ovr_on = 0;
    resolve_result = VK_USNAP_RESOLVE_NATIVE_ONLY;
    native_current = 0;
    assert(!consume(&ex).buffer && recording_failed && selectors == 1 && !records && !native_proofs &&
           released == 1 && !retained);
    recording_failed = 0;
    native_current = 1;
    reclassify = 1;
    assert(!consume(&ex).buffer && recording_failed && !records && !native_proofs && released == 2 && !retained);
    recording_failed = 0;
    reclassify = 0;
    resolve_result = VK_USNAP_RESOLVE_NATIVE_ONLY;
    native_proof = RC_REGISTRY_BLOCKED;
    assert(!consume(&ex).buffer && recording_failed && native_proofs == 1 && !records && released == 3 && !retained);
    recording_failed = 0;
    native_proof = RC_REGISTRY_OK;
    assert(consume(&ex).buffer == native_buffer && native_proofs == 2 && records == 1 &&
           retained == 1 && publications == 2 && !maps);
    uint64_t want = UINT64_C(0xf81a);
    memcpy(native_bytes + 8, &want, sizeof want);
    assert(ubo_ptr(&ex, NVN_STAGE_VS, &pointer, &value) && value == want && read_ends == 1);
    finish_result = VK_EXEC_HOST_INPUT_STALE;
    assert(!ubo_ptr(&ex, NVN_STAGE_VS, &pointer, &value) && !value && read_ends == 2);
    finish_result = VK_EXEC_HOST_INPUT_CHANGED;
    assert(!ubo_ptr(&ex, NVN_STAGE_VS, &pointer, &value) && !value && read_ends == 3);

    resolve_result = VK_USNAP_RESOLVE_READY;
    VkUsnapLease lease = {0};
    draw.prep_capture = &lease;
    assert(consume(&ex).buffer == snap_buffer && lease.capture == 7 && publications == 3);
    memcpy(snapshots + 16 + 8, &want, sizeof want);
    assert(ubo_ptr(&ex, NVN_STAGE_VS, &pointer, &value) && value == want);
    lease_valid = 0;
    assert(!ubo_ptr(&ex, NVN_STAGE_VS, &pointer, &value) && !value);
    unsigned before = queries;
    update_result = VK_EXEC_UNIFORM_UPDATE_STALE;
    assert(!ubo_ptr(&ex, NVN_STAGE_VS, &pointer, &value) && !value && queries == before);
    update_result = VK_EXEC_UNIFORM_UPDATE_READY;
    assert(ubo_ptr(&ex, NVN_STAGE_VS, &pointer, &value) && value == UINT64_C(0x1234) && queries == before);

    MergeSlab slab = {.chunk = {merged_buffer, merged_bytes}};
    GraphicsUboMerge merge = {.address = (uintptr_t)guest, .bytes = sizeof guest, .offset = 16,
                               .stage = NVN_STAGE_VS, .slot = 4};
    VkExecGraphicsScratch scratch = {.slab = &slab, .start = 16, .ubo_merges = &merge, .ubo_merge_count = 1};
    lease = (VkUsnapLease){0};
    graphics_payload_bound = &scratch;
    resolve_result = VK_USNAP_RESOLVE_ORDERED_PRODUCER;
    unsigned snapshot_maps = maps;
    assert(!consume(&ex).buffer && recording_failed && merge_checks == 1 && !scratch.exposed && !merged_inputs);
    recording_failed = 0;
    merge_current = 1;
    merge.slot = 3;
    assert(!consume(&ex).buffer && recording_failed && merge_checks == 1 && !scratch.exposed && !merged_inputs);
    recording_failed = 0;
    merge.slot = 4;
    VkDescriptorBufferInfo merged = consume(&ex);
    assert(merged.buffer == merged_buffer && merged.offset == 32 && merged.range == sizeof guest);
    assert(scratch.exposed && !recording_failed && maps == snapshot_maps && merge_checks == 2 && merged_inputs == 1);
    assert(publications == 4 && !lease.capture && terminal_probes == refused);
    return 0;
}
'''
    cc = os.environ.get("CC") or shutil.which("clang")
    if not cc:
        raise SystemExit("Clang is required (set CC)")
    clang = Path(shutil.which(cc) or cc).resolve()
    vma = Path(os.environ.get("VMA_INCLUDE", clang.parent.parent / "include" / "vma"))
    env = dict(os.environ, PATH=str(clang.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="uniform-capture-consumer-") as temporary:
        folder = Path(temporary)
        test = folder / "check.c"
        executable = folder / ("check.exe" if os.name == "nt" else "check")
        test.write_text(fixture)
        subprocess.run([str(clang), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=undefined",
                        "-I" + str(ROOT / "runtime"), "-I" + str(vma), str(test), "-o", str(executable)],
                       env=env, check=True)
        subprocess.run([str(executable)], env=env, check=True)
    print("UBO/SSBO snapshot consumers: typed failures, native spans, merged scratch, pointer reads and arena usage passed")


if __name__ == "__main__":
    main()

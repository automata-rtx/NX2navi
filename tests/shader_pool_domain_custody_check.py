"""Verify domain selection and actual descriptor binding share the retained storage snapshot."""

import os
from pathlib import Path
import subprocess
import tempfile

from bind_recipe_check import function


ROOT = Path(__file__).resolve().parents[1]
PREFIX = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { POOL_INDEX_C0, POOL_INDEX_STORAGE, POOL_INDEX_UNIFORM, POOL_INDEX_INLINE,
       VK_EXEC_UNIFORM_UPDATE_READY, GFX_RES_SSBO = 1, RC_REGISTRY_OK = 0, RC_REGISTRY_BLOCKED = 7 };
typedef struct { uint64_t scene, ordinal; } RcRegistryPosition;
typedef struct { uint64_t backing, offset, size; } RcSourceRange;
typedef struct { uint64_t address,mapping; RcSourceRange range; } RcInterval;
typedef struct RcRegistry RcRegistry;
enum { RC_GUEST_COMPATIBILITY_BACKING = 1 };
static int rc_range_valid(RcSourceRange r) {
    return r.backing && r.size && r.size <= UINT64_MAX-r.offset;
}
typedef struct { uint64_t addr, size; } p_bind_buf;
typedef struct { unsigned binding, flags, ptr; } GfxShaderRes;
enum { VK_PIPELINE_BIND_POINT_GRAPHICS = 1, VK_PIPELINE_BIND_POINT_COMPUTE = 2,
       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7 };
typedef struct { const GfxShaderRes *res; int stage, kind, type; unsigned count; } VkPipeBinding;
typedef struct { int bind_point; unsigned n_bindings; VkPipeBinding bindings[2]; } VkPipeInfo;
typedef struct { unsigned owner; } VkMemSpanCapture;
typedef struct { uint64_t buffer, offset, range; } VkDescriptorBufferInfo;
enum { VK_MCLS_HOST_CACHED = 3, VK_MCLS_DEVICE = 4, VK_MCLS_HOST_UNC = 5 };
typedef struct { uint64_t buffer; int cls,alloc; } VkMemSpan;
typedef struct { uint64_t allocation, incarnation, offset; } RcRegistryBufferLocation;
typedef struct { RcInterval interval; RcRegistryBufferLocation native; } RcRegistryBufferSpan;
typedef struct {
    VkMemSpan span;
    VkMemSpanCapture *native;
    RcInterval *mapped;
    uint32_t mapped_count, visible;
} EffectStorageCapture;
typedef struct { unsigned token; uint64_t buffer; } PoolIndexSnapshot;
typedef struct PoolIndexInput {
    const GfxShaderRes *resource;
    struct PoolIndexInput *source;
    p_bind_buf bound;
    VkDescriptorBufferInfo descriptor;
    VkMemSpan span;
    VkMemSpanCapture *native;
    RcInterval *ranges;
    uint32_t range_count;
    PoolIndexSnapshot *snapshot;
    const uint8_t *bytes;
    uint32_t byte_offset, word, word_count, offsets[16];
    int kind, stage, slot, whole_range, gpu_words_ready, recorded, recorded_copy, guard_original, scalar_original;
    int lease, immutable_import,import_required;
} PoolIndexInput;
typedef struct { uint32_t count; int finite; PoolIndexInput values[2]; } PoolIndexInputs;
typedef struct {PoolIndexInputs inputs;} ComputePoolIndexInputs;
typedef struct {ComputePoolIndexInputs *compute_inputs; const VkPipeInfo *compute_info; RcRegistryPosition read;} VkExecPoolGrant;
static struct {VkExecPoolGrant *grant; const VkPipeInfo *info; unsigned opcode;} pool_binding;

typedef struct { unsigned opcode; RcRegistryPosition read_position; PoolIndexInputs *pool_index_inputs; } VkExecGraphicsDraw;
typedef struct { int pb_spoiled, mo_ovr_on, sf_mvprev; } DrawState;
typedef struct { uint64_t fence; p_bind_buf ssbo[8][3], ubo[8][3]; DrawState *dc; } VkExec;
typedef struct { VkExec *ex; VkExecGraphicsDraw *draw; int stage; } PoolIndexRead;
static struct { uint64_t storage_write_alignment, sbuf_align; } vk_caps = {1, 16};
static unsigned consulted, scene_uses, refusals, native_copies, authority_checks;
static int remapped, blocked, image_owned;
static uint64_t mirror_incarnation;
static RcRegistryPosition original = {5, 16};
static VkExecGraphicsDraw *graphics_current;
static PoolIndexSnapshot snapshot = {.token=99,.buffer=71};
enum { VK_SUCCESS=0,VK_EXEC_HOST_INPUT_READY=0 };

static int nvn_gpu_only(uint64_t address) { (void)address;return 0; }
static int resource_effects_image_trace_enabled(void) { return 0; }
static void pool_index_import_report(RcRegistryPosition read,const PoolIndexInput *input,uint64_t address,
    RcRegistryBufferLocation native,const char *phase,int required,int placement,int host,const char *refusal) {
    (void)read;(void)input;(void)address;(void)native;(void)phase;(void)required;(void)placement;(void)host;(void)refusal;abort();
}


static const p_bind_buf *effect_storage_fixed(const VkExec *ex, int stage, const GfxShaderRes *resource) {
    assert(stage == 4 && resource->binding == 19);
    return &ex->ssbo[stage][1];
}
static int vk_exec_pool_index_read_word(VkExec *ex, VkExecGraphicsDraw *draw, int stage, int kind,
        const GfxShaderRes *resource, uint32_t offset, uint32_t *word) {
    assert(ex && draw == graphics_current && stage == 4 && kind == GFX_RES_SSBO && resource->binding == 19 && !offset);
    consulted++;
    PoolIndexInput *input = &draw->pool_index_inputs->values[0];
    if (!input->bytes) { input->word_count = 1; return 2; }
    *word = 12;
    return 1;
}
static int pool_index_c0_word(const VkExec *ex, int stage, uint32_t offset, uint32_t *word) {
    (void)ex;(void)stage;(void)offset;(void)word;abort();
}
static int vk_usnap_lease_valid_at(const int *lease, RcRegistryPosition read) { (void)lease;(void)read;abort(); }
static int vk_mem_span_capture_current(const VkMemSpanCapture *capture, const VkMemSpan *span) {
    assert(capture && capture->owner == 12 && span->buffer == 40);
    return !remapped;
}
static int vk_exec_uniform_update_native(const VkExec *ex, uint64_t address, uint64_t range,
    const VkMemSpan *span, const VkMemSpanCapture *native) {
    (void)ex;(void)address;(void)range;(void)span;(void)native;abort();
}
static int unrelated_authority;
static RcRegistry *rc_registry_runtime(void) { return (RcRegistry *)(uintptr_t)1; }
static int rc_registry_read_device_buffer_at(RcRegistry *registry,const RcInterval *range,uint32_t count,RcRegistryPosition read) {
    assert(registry == rc_registry_runtime() && count == 1 && read.scene == original.scene && read.ordinal == original.ordinal);
    if (range->address != 8)
        assert(range->range.backing == 13 && range->mapping == 409);
    authority_checks++;
    return (blocked || (unrelated_authority && range->range.size != 4)) ? RC_REGISTRY_BLOCKED : RC_REGISTRY_OK;
}
static int vk_mem_span_buffer_location(const VkMemSpanCapture *capture,const VkMemSpan *span,
                                        RcRegistryBufferLocation *location) {
    if (!vk_mem_span_capture_current(capture,span)) return 0;
    *location = (RcRegistryBufferLocation){29,mirror_incarnation,32};
    return 1;
}
static int word_busy, word_foreign, scalar_image;
static unsigned scalar_checks, affine_checks;
static const uint64_t guard_offset = 8, scalar_offset = 2917648, guest_base = UINT64_C(0x1b6cb0000);
static int rc_registry_read_range_at(RcRegistry *registry, RcSourceRange range, RcRegistryPosition read) {
    assert(registry == rc_registry_runtime() && range.size == 4 && range.backing == 13 &&
        (range.offset == 0x1000 + scalar_offset || range.offset == 0x1000 + guard_offset) &&
        read.ordinal == original.ordinal);
    ++scalar_checks;
    return word_busy ? RC_REGISTRY_BLOCKED : RC_REGISTRY_OK;
}
static int rc_registry_buffer_access_check(RcRegistry *registry, RcRegistryBufferSpan span, RcRegistryPosition read) {
    assert(registry == rc_registry_runtime() && span.interval.range.size == 4 && span.interval.mapping == 409 &&
        span.native.offset == 32 + scalar_offset && span.native.allocation == 29 && span.native.incarnation == 11 &&
        read.ordinal == original.ordinal);
    ++affine_checks;
    return word_foreign ? RC_REGISTRY_BLOCKED : RC_REGISTRY_OK;
}
static int rc_registry_native_image_buffer_requires_coverage(RcRegistry *registry,const RcInterval *range,uint32_t count,
                                                              RcRegistryPosition read) {
    assert(registry == rc_registry_runtime() && count == 1 && read.scene == original.scene && read.ordinal == original.ordinal);
    return range->range.size == 4 ? scalar_image : image_owned;
}
static int pool_index_bind_active(const VkExec *ex) {
    assert(ex); return graphics_current->pool_index_inputs->finite;
}
static unsigned native_retains, native_registrations;
static VkMemSpanCapture *vk_mem_span_capture_retain(VkMemSpanCapture *owner) {
    assert(owner && owner->owner == 12);
    native_retains++;
    return owner;
}
static int vk_mem_span_record_input(VkMemSpanCapture *hold, const VkMemSpan *span) {
    assert(hold && hold->owner == 12 && span->buffer == 40);
    return 1;
}
static void vk_exec_record_native_input_provider(VkExec *ex, VkMemSpanCapture **hold) {
    assert(ex && hold && *hold);
    *hold = NULL;
    native_registrations++;
}
static void vk_mem_span_capture_release(VkMemSpanCapture **hold) { assert(hold && !*hold); }
static void vk_exec_required_graphics_unavailable(VkExec *ex,unsigned opcode,const char *reason) {
    assert(ex && opcode == 26 && reason); refusals++;
}
static int rc_registry_position_compare(RcRegistryPosition a,RcRegistryPosition b) {
    return a.scene!=b.scene || a.ordinal!=b.ordinal;
}
static RcRegistryPosition vk_exec_resource_position(void) { return original; }
static int nvn_stage_of(int stage) { return stage; }
static int pool_index_inputs_current(const VkExec *,RcRegistryPosition,const PoolIndexInputs *);
static int vk_exec_pool_index_inputs_current(const VkExec *ex,const VkExecGraphicsDraw *draw) {
    return pool_index_inputs_current(ex,draw->read_position,draw->pool_index_inputs);
}
static int compute_pool_output_bindings_current(VkExec *ex,const VkPipeInfo *info,const PoolIndexInput *input) {
    (void)ex;(void)info;(void)input;abort();
}
static int effect_overlap;
static int vk_exec_effects_read_allowed_range(RcSourceRange range,int output) {
    assert(range.backing==13 && (range.size==0x980000 || range.size==0x580020) && !output);
    return !effect_overlap;
}
static void pool_index_snapshot_record(uint64_t fence,void *context) {
    assert(fence == 80 && context == &snapshot && snapshot.token == 99); scene_uses++;
}
'''

TEST = r'''
int main(void) {
    static const uint32_t data[] = {12, 12, 0xfffff, 0};
    const GfxShaderRes shader = {.binding = 19};
    RcInterval range = {.address=8};
    VkMemSpanCapture owner = {12};
    PoolIndexInputs inputs = {.count = 1, .values = {{.resource=&shader, .bound={0x400,16},
        .descriptor={40,0,16}, .span={.buffer=40}, .native=&owner,.ranges=&range,.range_count=1,
        .kind=POOL_INDEX_STORAGE,.stage=4,.slot=1}}};
    DrawState state = {0};
    VkExec ex = {.fence=80, .ssbo[4][1]={0x400,16}, .dc=&state};
    VkExecGraphicsDraw draw = {.opcode=26,.read_position=original,.pool_index_inputs=&inputs};
    graphics_current = &draw;
    PoolIndexRead attempt = {&ex,&draw,4};
    const uint8_t *bytes = NULL;
    uint32_t size = 0;
    VkDescriptorBufferInfo descriptor = {0};
    assert(!pool_index_bind_input(&ex, 4, 1, &shader, &descriptor));
    assert(pool_draw_index_domain(&attempt, &shader, &bytes, &size) == 2 && consulted == 1);
    PoolIndexInput *input = &inputs.values[0];
    assert(input->whole_range && input->word_count == 1 && !bytes && !size);
    // The ordered native copy made the shader snapshot and readback from the same retained source.
    input->descriptor = (VkDescriptorBufferInfo){71,0,16};
    input->bytes = (const uint8_t *)data;
    input->snapshot = &snapshot;
    input->gpu_words_ready = 1;
    input->recorded_copy = 1;
    pool_index_snapshot_record(ex.fence, &snapshot);
    native_copies++;
    assert(pool_draw_index_domain(&attempt, &shader, &bytes, &size) == 1 && bytes == (const uint8_t *)data && size == 16);
    inputs.finite = 1;
    assert(pool_index_bind_input(&ex, 4, 1, &shader, &descriptor) && descriptor.buffer == 71 && descriptor.range == 16);
    assert(input->recorded && state.pb_spoiled && scene_uses == 1 && native_copies == 1 && authority_checks == 1);
    blocked = 1;
    assert(pool_index_bind_input(&ex, 4, 1, &shader, &descriptor) && !descriptor.buffer && refusals == 1);
    blocked = 0;
    remapped = 1;
    assert(pool_index_bind_input(&ex, 4, 1, &shader, &descriptor) && !descriptor.buffer && refusals == 2);
    assert(scene_uses == 1 && native_copies == 1 && consulted == 2);
    remapped = 0;
    vk_caps.storage_write_alignment = 0;
    assert(!pool_draw_index_domain(&attempt, &shader, &bytes, &size) && consulted == 2);
    // Real guard geometry: b20 reads byte 8 of a full 0x980000-byte device mirror.
    range = (RcInterval){.address=guest_base,.mapping=409,.range={13,0x1000,0x980000}};
    EffectStorageCapture fixed = {.span={.buffer=40,.cls=VK_MCLS_HOST_CACHED},.native=&owner,.mapped=&range,.mapped_count=1};
    int scalar = 1;
    assert(!pool_index_storage_source_ready(&fixed,original,1,&scalar) && !scalar);
    assert(pool_index_storage_source_ready(&fixed,original,0,&scalar) && !scalar);
    fixed.visible=1; scalar=1;
    assert(pool_index_storage_source_ready(&fixed,original,1,&scalar) && !scalar);
    fixed.span.alloc=1; scalar=1;
    assert(!pool_index_storage_source_ready(&fixed,original,1,&scalar) && !scalar);
    fixed.visible=0; fixed.span.alloc=0;
    mirror_incarnation = 11;
    scalar = 1;
    assert(pool_index_storage_source_ready(&fixed,original,1,&scalar) && scalar);
    remapped = 1;
    assert(!pool_index_storage_source_ready(&fixed,original,1,&scalar) && !scalar);
    remapped = 0;

    input->span = fixed.span;
    input->guard_original = input->scalar_original = 1;
    input->whole_range = input->recorded = input->recorded_copy = 0;
    input->snapshot = NULL;
    input->bytes = NULL;
    ex.ssbo[4][1] = input->bound = (p_bind_buf){guest_base,0x980000};
    input->descriptor = (VkDescriptorBufferInfo){40,32,0x980000};
    input->word_count = 1;
    input->offsets[0] = guard_offset;
    RcInterval word = {0};
    assert(pool_index_scalar_word(input,guard_offset,&word));
    assert(word.address == guest_base + guard_offset && word.mapping == 409 && word.range.size == 4);
    unrelated_authority = 1;
    assert(rc_registry_read_device_buffer_at(rc_registry_runtime(),input->ranges,1,original) == RC_REGISTRY_BLOCKED);
    assert(pool_index_bind_input(&ex,4,1,&shader,&descriptor) &&
           descriptor.buffer == 40 && descriptor.offset == 32 && descriptor.range == 0x980000);
    assert(native_retains == 1 && native_registrations == 1 && scene_uses == 1);
    VkPipeInfo native_layout={.bind_point=VK_PIPELINE_BIND_POINT_GRAPHICS,.n_bindings=1,
        .bindings={{&shader,4,GFX_RES_SSBO,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1}}};
    assert(pool_index_bind_layout(&ex,&native_layout));
    image_owned=1; // A true image dependency elsewhere still requires a composed full shader input.
    assert(!pool_index_bind_layout(&ex,&native_layout));
    image_owned=0;
    effect_overlap=1;
    assert(!pool_index_bind_layout(&ex,&native_layout));
    effect_overlap=0;
    native_layout.bindings[0].count=2;
    assert(!pool_index_bind_layout(&ex,&native_layout));
    scalar_image = 1;
    assert(pool_index_bind_input(&ex,4,1,&shader,&descriptor) && !descriptor.buffer && refusals == 3);
    scalar_image = 0;

    // A separate scalar table index uses the same exact authority path and retained binding.
    range.range.size = 0x580020;
    ex.ssbo[4][1] = input->bound = (p_bind_buf){guest_base,0x580020};
    input->guard_original = input->recorded = 0;
    input->descriptor.range = 0x580020;
    input->offsets[0] = scalar_offset;
    assert(pool_index_scalar_word(input,scalar_offset,&word));
    assert(word.address == guest_base + scalar_offset && word.mapping == 409 && word.range.size == 4);
    assert(!pool_index_scalar_word(input,0x58001e,&word));
    assert(pool_index_bind_input(&ex,4,1,&shader,&descriptor) &&
        descriptor.buffer == 40 && descriptor.offset == 32 && descriptor.range == 0x580020);
    assert(native_retains == 2 && native_registrations == 2 && scalar_checks == 6 && !affine_checks);
    word_busy = 1;
    assert(pool_index_bind_input(&ex,4,1,&shader,&descriptor) && descriptor.buffer == 40);
    assert(affine_checks == 1 && native_registrations == 2);
    word_foreign = 1;
    assert(pool_index_bind_input(&ex,4,1,&shader,&descriptor) && !descriptor.buffer);
    assert(affine_checks == 2 && native_registrations == 2);
    remapped = 1;
    assert(pool_index_bind_input(&ex,4,1,&shader,&descriptor) && !descriptor.buffer);
    assert(affine_checks == 2 && native_registrations == 2);
    // A raw import remains mutable. The retained replica alone can supply the shader decision and full descriptor.
    remapped=0;
    mirror_incarnation=0;
    input->scalar_original=input->recorded=input->recorded_copy=0;
    input->immutable_import=1;
    input->snapshot=&snapshot;
    input->descriptor=(VkDescriptorBufferInfo){snapshot.buffer,0,0x580020};
    native_layout.bindings[0].count=1;
    const unsigned before_replica=scene_uses;
    assert(pool_index_bind_layout(&ex,&native_layout));
    assert(pool_index_bind_input(&ex,4,1,&shader,&descriptor) && descriptor.buffer==snapshot.buffer &&
           !descriptor.offset && descriptor.range==0x580020 && scene_uses==before_replica+1);
    assert(pool_index_bind_input(&ex,4,1,&shader,&descriptor) && scene_uses==before_replica+1);
    input->descriptor.buffer=40;
    assert(!pool_index_bind_layout(&ex,&native_layout));
    input->descriptor.buffer=snapshot.buffer;
    native_layout.bindings[1]=native_layout.bindings[0];
    native_layout.n_bindings=2;
    assert(!pool_index_bind_layout(&ex,&native_layout));
    native_layout.n_bindings=1;
    effect_overlap=1;
    assert(!pool_index_bind_layout(&ex,&native_layout));
    puts("Table domain custody: scalar retained direct, immutable full raw replica, final binding, alias and remap refusal");
}
'''


def main():
    draw = (ROOT / "runtime/vk_exec_draw.c").read_text(encoding="utf-8")
    beginning = draw.index("static int pool_index_scalar_word(")
    draw = draw[beginning:]
    names = ("pool_index_scalar_word", "pool_index_scalar_authority", "pool_index_scalar_current",
             "pool_index_input_current", "pool_index_inputs_current", "pool_index_storage_source_ready",
             "pool_draw_index_domain", "pool_index_bind_layout", "pool_index_bind_input")
    merged = PREFIX + "\n".join(function(draw, name) for name in names) + TEST
    compiler = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(compiler.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="table-domain-custody-") as temp:
        source = Path(temp) / "check.c"
        source.write_text(merged, encoding="utf-8")
        exe = source.with_suffix(".exe" if os.name == "nt" else "")
        subprocess.run([str(compiler), "-std=gnu11", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                        "-fsanitize-trap=all", str(source), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()

"""Host check of ordered direct and indirect compute preparation and native recording order."""

import os
import re
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "runtime"
REPLAY = SOURCES / "vk_exec.c"


def function(source, name):
    match = re.search(r"^(?:static )?[A-Za-z_][\w \t*]*\b" + re.escape(name) + r"\([^;{}]*\)\s*\{", source, re.M)
    if not match:
        raise ValueError(name)
    depth = 1
    end = match.end()
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]



def between(source, start, following):
    i = source.index(start)
    return source[i:source.index(following, i + len(start))]


DRAW_STUBS = r'''
#include <assert.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include "resource_core_budget.h"

static int n_probe_sh;
static uint64_t pool_probe_exact_dispatches;

typedef struct { int32_t x, y, z; uint64_t caller; } p_dispatch;
typedef struct { uint64_t ind, caller; } p_dispatch_ind;
typedef struct { uint64_t serial; } prog_t;
typedef struct { uint32_t tag; } GfxShader;
typedef struct { int handle, slot, pool, type, count, flags, ptr; } GfxShaderRes;
typedef struct {const GfxShaderRes *res; int kind; VkDescriptorType type; unsigned stage,count;} VkPipeBinding;
typedef struct { VkPipelineBindPoint bind_point; int pool; const GfxShader *compute_effect_shader; } VkPipeInfo;
typedef struct RcEffectProgram RcEffectProgram;
typedef struct SelectedStorageWriters SelectedStorageWriters;
typedef struct { int checked; } VkPassMark;
typedef struct { VkBuffer buffer; VkDeviceSize offset; int cls; } VkMemSpan;
typedef struct { uint64_t scene,ordinal; } RcRegistryPosition;
typedef struct VkMemSpanCapture { VkMemSpan span; uint64_t address, size; unsigned mapping; } VkMemSpanCapture;
typedef enum { RC_REGISTRY_OK, RC_REGISTRY_UNSUPPORTED, RC_REGISTRY_STALE, RC_REGISTRY_BUSY } RcRegistryResult;
typedef struct VkExecPoolGrant { int held; } VkExecPoolGrant;
typedef struct { const VkPipeInfo *info; VkExecPoolGrant *grant; uint32_t opcode; } VkExecPoolBindScope;
typedef struct VkExecGpaSampleScope { int owned; struct { int uniform; } prepare; } VkExecGpaSampleScope;
typedef struct { RcRegistryPosition producer, read; int guard; struct Source { int range; } source; } VkGpaPrefixRequest;
typedef enum { VK_EXEC_GRAPHICS_READY, VK_EXEC_GRAPHICS_COMPAT, VK_EXEC_GRAPHICS_NEEDS_ORDERED,
    VK_EXEC_GRAPHICS_UNAVAILABLE, VK_EXEC_GRAPHICS_STOPPED, VK_EXEC_GRAPHICS_INVALIDATED } VkExecGraphicsResult;
static VkExecPoolBindScope binding;

typedef struct DrawCtx {
    uint32_t c0_dim[3];
    struct { VkBuffer c0_buf[3]; VkDeviceSize c0_off[3]; int c0_valid[3]; } B;
} DrawCtx;
typedef struct {
    DrawCtx *dc;
    prog_t *prog;
    VkCommandBuffer cb;
    uint64_t tex[1][128], img[1][32];
    uint64_t fence, registration_scope, resource_op, resource_next;
    int recorded, dirty, rt_resolved, in_pass;
} VkExec;

enum { NVN_STAGE_CS, W_PROGRAM, W_PIPELINE, W_INDIRECT, GFX_RES_TEX, GFX_RES_IMG, GFX_RES_UBO, GFX_RES_SSBO, VK_MUSE_IND };
enum Event {
    E_BARRIER, E_USE, E_PIPE_BIND, E_UNIFORM_COPY, E_RESOURCES, E_COPY, E_COPY_BARRIER,
    E_DISPATCH, E_INDIRECT, E_GPA_PROOF, E_WRITES, E_INDIRECT_COUNT, E_INDIRECT_BOUND,
};
static const GfxShader old_shader = {7}, later_shader = {9};
static const GfxShader *visible_shader;
static VkPipeInfo compute_info = {.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE};
static const VkPipeline compute_pipeline = (VkPipeline)(uintptr_t)41;
static const VkBuffer arg_buffer = (VkBuffer)(uintptr_t)42, const_buffer = (VkBuffer)(uintptr_t)43;
static const VkBuffer mirror_buffer = (VkBuffer)(uintptr_t)58;
static int native_indirect, remap_on_bind, remap_after_scratch;
static VkBuffer input_buffer(void) { return native_indirect ? mirror_buffer : arg_buffer; }
static const VkCommandBuffer scene_cb = (VkCommandBuffer)(uintptr_t)44;
static const VkDeviceSize arg_offset = 20, c0_offset = 512;
static const uint64_t argument_address = 0x3000;
static VkPassMark mark;
static int twin_dispatch, vk_dump_armed;
typedef struct {
    VkMemSpan selected;
    VkMemSpanCapture *capture;
    int owned;
} TwinIndirectReceipt;
static TwinIndirectReceipt *twin_indirect;
static int twin_may_bind(const VkPipeInfo *info) { assert(info);return native_indirect; }
static void twin_indirect_end(TwinIndirectReceipt *receipt) {
    if(receipt==twin_indirect) { assert(receipt->capture&&!receipt->owned);twin_indirect=NULL; }
    else assert(!receipt->capture&&!twin_indirect);
}
static VkResult standins_result;
static unsigned standins_checked;
static struct { uint64_t bind_ns, bind_calls, dispatches; } vk_exec_frame;
static enum Event events[32];
static unsigned nevents, warns, selections, preparation;
static int deferred_binding, nvn2_mode, missing_indirect;
static int compat_samples, compat_sample_refusal, compat_raw_storage, compat_image_bytes;
static unsigned compat_bindings;
static int compat_storage, compat_storage_refused;
static unsigned compat_storage_prepares, compat_storage_releases;
static int compat_selected(void) { return compat_samples || compat_storage; }
static unsigned touches, lookups, mapping, captures, capture_releases, native_receipts, physical_adoptions, references;
static unsigned native_provider_uses, native_provider_completions;
static VkMemSpanCapture captured_span;
static VkMemSpanCapture *completed_span;
static void (*complete_span)(void *);
static int replace_on_settle, prospective, compat_ambient, ambient_bound;
static int uniform_blocked, uniform_probes, uniform_requests;
static unsigned grants, grant_releases, settles, pending_cancels, pool_calls, warnings;
static VkExecPoolGrant pool_grant;
static int publish_disjoint_output, later_pool_authority;
static VkExecGpaSampleScope sample_scope;
static int split_draws;
static VkExec *active;

static void record(enum Event event) {
    assert(nevents < sizeof events / sizeof events[0]);
    events[nevents++] = event;
}
static void expect(const enum Event *expected, size_t n) {
    assert(nevents == n);
    for (size_t i = 0; i < n; i++) assert(events[i] == expected[i]);
}

#define RWT_CALL(region, expression) (expression)
#define VK_MEM_USE(span) do { assert(!native_indirect && (span).buffer == arg_buffer); record(E_USE); } while (0)
#define VK_MEM_BOUND(span) do { assert((span).buffer == input_buffer()); record(E_INDIRECT_BOUND); } while (0)

static void warn_once(int kind, const char *message, ...) {
    assert(kind == W_PROGRAM || kind == W_PIPELINE || kind == W_INDIRECT);
    assert(message && *message);
    warns++;
}
static int prog_lead_on(void) { return 0; }
static void prog_lead(prog_t *program, int stage) { (void)program; (void)stage; abort(); }
static void vk_contact_flush(VkExec *ex) { assert(ex == active); }
static int twin_compare_resource(const GfxShaderRes *resource) { return resource->type == 0x403; }
static void cb_res(VkExec *ex) { assert(ex == active); }
static void fixed_require(void) {}
static int aa_dispatch(VkExec *ex, int x, int y, int z, int *specialized) {
    assert(specialized && !*specialized);
    assert(ex == active);
    assert((x == 2 && y == 3 && z == 5) || (!x && !y && !z));
    return 0;
}
static int vk_texture_deferred(void) { return deferred_binding == 2; }
static int native_tensor(VkExec *ex) { assert(ex == active); return 1; }
static void untranslated_dispatch_fallback(VkExec *ex) { (void)ex; abort(); }
static const GfxShader *shader_for(VkExec *ex, int stage, int *present, int *pending, unsigned wait) {
    assert(ex == active && stage == NVN_STAGE_CS && present && !pending && !wait);
    selections++;
    *present = 1;
    return visible_shader;
}
static int knob_log_cs_binds(void) { return 0; }
static int gfx_shader_resources(const GfxShader *s, int kind, const GfxShaderRes **out) {
    static const GfxShaderRes ordinary_texture = {.slot = 3, .type = 3};
    static const GfxShaderRes ordinary_storage = {.slot = 4, .count = 1};
    assert(s == &old_shader && out);
    if (kind == GFX_RES_SSBO && compat_raw_storage) { *out = &ordinary_storage; return 1; }
    if (kind == GFX_RES_IMG) {
        static const GfxShaderRes image={.slot=2,.count=1,.flags=4};
        *out = compat_storage ? &image : NULL;
        return compat_storage;
    }
    if (kind != GFX_RES_TEX) { assert(kind == GFX_RES_UBO || kind == GFX_RES_SSBO); *out = NULL; return 0; }
    *out = &ordinary_texture;
    return compat_samples;
}
static char dispatch_input_line[768];
static void rt_log(const char *format, ...) {
    assert(format);
    warnings++;
    if (!strstr(format, "required dispatch input read=")) return;
    assert(!dispatch_input_line[0]);
    va_list args;
    va_start(args, format);
    const int length = vsnprintf(dispatch_input_line, sizeof dispatch_input_line, format, args);
    va_end(args);
    assert(length > 0 && (size_t)length < sizeof dispatch_input_line);
}

static VkPipeline vk_pipeline_compute(const GfxShader *shader, const VkPipeInfo **info) {
    assert(shader == visible_shader && info && !*info);
    preparation++;
    *info = &compute_info;
    return compute_pipeline;
}
static int pool_on(void) { return 1; }
static int shader_skipped(const GfxShader *shader) { assert(shader == &old_shader); return 0; }
static void vk_exec_barrier_dispatch(VkExec *ex) { assert(ex == active); record(E_BARRIER); }
static void vk_exec_barrier(VkExec *ex) { assert(ex == active); record(E_COPY_BARRIER); }
static int flat3d_preflight(VkExec *ex, const GfxShader *const *shaders, const int *stages, int n) {
    assert(ex == active && n == 1 && *stages == NVN_STAGE_CS && *shaders == &old_shader);
    return 1;
}
static void bind_pipeline(VkExec *ex, VkPipelineBindPoint point, VkPipeline pipe) {
    assert(ex == active && point == VK_PIPELINE_BIND_POINT_COMPUTE && pipe == compute_pipeline);
    record(E_PIPE_BIND);
}
static int vk_exec_gpa_dispatch_retrying(const VkExec *ex) { assert(ex == active); return 0; }
static int vk_exec_gpa_dispatch_binding(VkExec *ex,const GfxShader *shader,VkPipeline pipeline,
                                        const VkPipeInfo *info,uint64_t address,const VkMemSpan *indirect,VkMemSpanCapture *argument,
                                        int repeatable,int selected) {
    assert(ex==active && shader==&old_shader && pipeline==compute_pipeline && info==&compute_info && repeatable);
    assert(selected==0 || selected==1);
    if (!selected && compat_ambient) ambient_bound = 1;
    assert(indirect ? address==argument_address && indirect->buffer==input_buffer() && indirect->offset==arg_offset : !address);
    assert(native_indirect ? argument==&captured_span : !argument);
    return 1;
}
static uint64_t bt_now(void) { return 9; }
static void verify_descriptor_binding(VkExec *ex, const VkPipeInfo *info);
static void verify_compat_sample_binding(VkExec *ex, const VkPipeInfo *info);
static int dispatch_materialize_ubo_merges(VkExec *ex, const VkPipeInfo *info);
static int dispatch_query_samples_record(VkExec *ex, const VkPipeInfo *info, const void *samples) {
    assert(ex && info);
    (void)samples;
    return 1;
}
static void bind_resources(VkExec *ex, const VkPipeInfo *info) {
    assert(ex == active && info == &compute_info && twin_dispatch);
    verify_descriptor_binding(ex, info);
    verify_compat_sample_binding(ex, info);
    assert(dispatch_materialize_ubo_merges(ex, info));
    assert(!info->pool || (binding.info == info && binding.grant && binding.grant->held));
    record(E_RESOURCES);
    if (native_indirect) assert(twin_indirect && twin_indirect->capture==&captured_span &&
                               twin_indirect->selected.buffer==mirror_buffer);
    if (remap_on_bind) mapping++;
    if (!ex->dc->c0_dim[0]) {
        assert(!ex->dc->B.c0_valid[2]);
        ex->dc->B.c0_buf[2] = const_buffer;
        ex->dc->B.c0_off[2] = c0_offset;
        ex->dc->B.c0_valid[2] = 1;
    }
    if (deferred_binding == 1) deferred_binding = 2;
}
static VkPassMark *dispatch_mark(VkExec *ex, const GfxShader *shader, uint32_t x, uint32_t y, uint32_t z) {
    assert(ex == active && shader == &old_shader);
    assert((x == 2 && y == 3 && z == 5) || (!x && !y && !z));
    return &mark;
}
static void vk_pass_mark_end(VkCommandBuffer cb, VkPassMark *actual) { assert(cb == active->cb && actual == &mark); }
// The real GPA table/settlement implementation is covered by gpa_check. Verify the direct emitter passes the
// exact selected arguments only after the actual dispatch. Indirect emission cannot request this proof.
static void dispatch_gpa_definite_written(const p_dispatch *payload, const GfxShader *shader,
                                            VkPipeline pipeline, const VkPipeInfo *info) {
    assert(payload->x == 2 && payload->y == 3 && payload->z == 5 && shader == &old_shader &&
           pipeline == compute_pipeline && info == &compute_info && nevents && events[nevents - 1] == E_DISPATCH);
    record(E_GPA_PROOF);
}
static void storage_images_written(VkExec *ex) {
    assert(ex == active && nevents && (events[nevents - 1] == E_GPA_PROOF || events[nevents - 1] == E_INDIRECT));
    record(E_WRITES);
}
static void vk_dump_dispatch(VkExec *ex, const GfxShader *shader, VkPipeline pipe, const VkPipeInfo *info,
                             const p_dispatch *op) {
    (void)ex; (void)shader; (void)pipe; (void)info; (void)op; abort();
}
static int vk_mem_lookup(uint64_t address, uint64_t size, VkMemSpan *span) {
    assert(address == argument_address && size == sizeof(VkDispatchIndirectCommand));
    lookups++;
    if (missing_indirect) return 0;
    *span = (VkMemSpan){arg_buffer, arg_offset, 71};
    return 1;
}
static int vk_mem_resolve(uint64_t address, uint64_t size, VkMemSpan *span) {
    touches++;
    if (missing_indirect) return 0;
    assert(address == argument_address && size == sizeof(VkDispatchIndirectCommand));
    *span = (VkMemSpan){arg_buffer, arg_offset, 71};
    return 1;
}
static int find_stage(DrawCtx *dc, prog_t *prog, int stage, const uint8_t **ctl, const uint8_t **code, uint32_t *len) {
    static const uint8_t ctl_data = 1;
    assert(dc == active->dc && prog == active->prog && stage == NVN_STAGE_CS);
    *ctl = &ctl_data;
    *code = NULL;
    *len = 0;
    return 1;
}
static int ctl_nvn2(const uint8_t *ctl) { assert(ctl && *ctl == 1); return nvn2_mode; }

VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer cb, VkBuffer src, VkBuffer dst, uint32_t count,
                                           const VkBufferCopy *ranges) {
    assert(cb == active->cb && src == input_buffer() && dst == const_buffer && count == 1);
    assert(ranges->srcOffset == arg_offset && ranges->dstOffset == c0_offset + (nvn2_mode ? 0 : 0x200));
    assert(ranges->size == sizeof(VkDispatchIndirectCommand));
    record(E_COPY);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    assert(cb == active->cb && x == 2 && y == 3 && z == 5);
    assert(active->dc->c0_dim[0] == x && active->dc->c0_dim[1] == y && active->dc->c0_dim[2] == z);
    record(E_DISPATCH);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDispatchIndirect(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset) {
    assert(cb == active->cb && buffer == input_buffer() && offset == arg_offset && !active->dc->B.c0_valid[2]);
    record(E_INDIRECT);
}
static void vk_mem_count(int cls, int category, uint64_t bytes) {
    assert(cls == 71 && category == VK_MUSE_IND && bytes == sizeof(VkDispatchIndirectCommand));
    record(E_INDIRECT_COUNT);
}
'''

PREFIX = r'''
#include <assert.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct { uint32_t op, size; } op_hdr;
typedef struct { uint32_t opcode,ops; } NvnExecStep;
static struct { uint64_t drained,out; } Q;
static struct { uintptr_t shader; VkMemSpanCapture *argument; VkMemSpan argument_span; int selected, native_argument; }
    gpa_retry_dispatch;
static RcRegistryPosition gpa_attempted;
enum { RECORDING_RESOURCE_UNAVAILABLE = 1 };
static int rc_registry_position_compare(RcRegistryPosition a,RcRegistryPosition b) {
    return a.scene != b.scene || a.ordinal != b.ordinal;
}
typedef struct { int unused; } RcRegistry;
typedef enum { VK_SYNC_PROGRESS_UNAVAILABLE,VK_SYNC_PROGRESS_COMPLETE,VK_SYNC_PROGRESS_STOPPED,VK_SYNC_PROGRESS_STOPPED_AFTER_CUT } VkSyncProgressResult;
enum { OP_DISPATCH,OP_DISPATCH_IND,PW_OP };
enum { SCENE=19,SCOPE=23,ORDINAL=7,PEAK=64*1024*1024 };
enum { VK_AA_GAME,VK_AA_ROLE_NONE };
static DrawCtx draw_context,*dc=&draw_context;
static prog_t program={17};
static const op_hdr walk_stopped;
static VkExec exec,*x_cur=&exec;
static VkExecDispatch caller;
static VkExecDispatch *dispatch_current=&caller;
static DrawDescriptorGrant *descriptor_binding_grant;
static DrawDescriptorGrant *compat_bound_descriptors(void);
static int descriptor_refused;
static uint32_t descriptor_prepared, descriptor_consumed;
static void verify_descriptor_binding(VkExec *ex, const VkPipeInfo *info) {
    if (!descriptor_binding_grant) return;
    DrawDescriptorGrant *expected = dispatch_current ? &dispatch_current->descriptors : compat_bound_descriptors();
    assert(descriptor_binding_grant == expected && descriptor_binding_grant->info == info &&
           descriptor_binding_grant->ordinal == ex->resource_op && descriptor_prepared && !descriptor_consumed);
    if (!deferred_binding) descriptor_consumed++;
}
static RcRegistry registry;
static int transaction,split_draws;
#define RWT_CHAIN(scope,stage)
static uint64_t visible_scope;
static const void *selected_payload;
static uint32_t selected_opcode;
static unsigned selects,drains,waits,parks,probes,builds,ins,outs,declarations,emits,releases;
static unsigned compiler_delays,selection_cancels;
static unsigned read_redeclarations, held_prepares, held_reads;
static int stop,loss,failed,recorder_parked,stop_wait,stop_handoff,fail_drain,terminal_probe,terminal_build,cache,noop;
static int replace_scope_while_waiting;
static uint64_t recording_scene;
static int sub_lock,recorder_resumed,submit_locked;
static unsigned handoffs,resumes;
#define parked recorder_parked
static VkExecDispatchEffectResult effect_answer;
typedef struct { unsigned count; } VkTextureWritePlan;
typedef struct { struct { VkMemSpanCapture *native; VkMemSpan span; } output; } DeviceStoragePointer;
static int device_shader_pointers_current(const DeviceStoragePointer *pointers,uint32_t count) {
    assert(!pointers && !count);return 1;
}
static void device_storage_pointers_free(DeviceStoragePointer *pointers,uint32_t count) {
    assert(!pointers && !count);
}
static int device_storage_pointer_value_current(const VkExec *ex, const DeviceStoragePointer *pointer) {
    assert(ex); (void)pointer; abort();
}
typedef struct {
    const VkPipeInfo *info;
    RcRegistryPosition read;
    int ready;
} ComputeSamples;
typedef struct { int *independent; int uploaded,written,phase_seen; } EffectImageCapture;
static int storage_target;
static EffectImageCapture compatibility_image;
struct VkExecDispatchImages {
    int owned;
    VkTextureWritePlan write_plan;
    DeviceStoragePointer *pointer_sources;
    uint32_t pointer_count;
    SelectedStorageWriters *selected_writers;
    ComputeSamples *samples;
    EffectImageCapture *images;
    uint32_t count;
    int storage_admission;
};
static VkExecDispatchImages selection;
typedef struct {
    ComputeSamples *compute_samples;
    const GfxShader *compat_compute_shader;
    VkPipeline compat_compute_pipeline;
    const prog_t *program;
    uint64_t program_serial;
    uint32_t opcode;
    VkExecBufferScratchReceipt *buffer_scratch;
    RcRegistryPosition position;
    DrawDescriptorGrant compat_descriptors;
    EffectImageCapture *images;
    VkTextureWritePlan write_plan;
    uint32_t image_count;
    int storage_admission;
    DeviceStoragePointer *pointer_sources;
    uint32_t pointer_count;
} EffectDeclaration;
static EffectDeclaration compat_effect, *effect_current;
static DrawDescriptorGrant *compat_bound_descriptors(void) {
    assert(effect_current == &compat_effect);
    return &compat_effect.compat_descriptors;
}
static ComputeSamples selected_samples;
static unsigned sample_preparations, sample_materializations, sample_releases, compat_selection_releases;
static const RcEffectProgram *gfx_shader_effects(const GfxShader *shader) { (void)shader; return NULL; }
static VkExecGraphicsResult selected_storage_writers_prepare(VkExec *ex, const VkPipeInfo *info, int ordered,
    const GfxShader *shader, const RcEffectProgram *program, const p_dispatch *direct, SelectedStorageWriters **out,
    const DeviceStoragePointer *pointers, uint32_t count, const void *override, const void *failure) {
    assert(ex && info && ordered && shader && !program && !*out && !pointers && !count && !override && !failure);
    (void)direct;
    return VK_EXEC_GRAPHICS_READY;
}
static VkExecGraphicsResult device_storage_pointer_outputs_prepare(VkExec *ex, const VkPipeInfo *info,
    const DeviceStoragePointer *pointers, uint32_t count, const SelectedStorageWriters *writers) {
    assert(ex == &exec && info == &compute_info && !pointers && !count && !writers);
    return VK_EXEC_GRAPHICS_READY;
}
static int selected_storage_writers_device_current(const VkExec *ex, const SelectedStorageWriters *writers,
    const VkPipeInfo *info, RcRegistryPosition read, const DeviceStoragePointer *pointers, uint32_t count) {
    assert(ex == &exec && !writers && info == &compute_info && read.scene == SCENE && read.ordinal == 2 * ORDINAL &&
        !pointers && !count);
    return 1;
}
static int vk_aa_mode(void) { return VK_AA_GAME; }
static int vk_aa_role(uint64_t hash) { (void)hash; abort(); }
static uint64_t cs_guest_hash(const prog_t *prog) { (void)prog; abort(); }
static uint64_t vk_texture_registration_publication(void) {
    return exec.registration_scope ? exec.registration_scope : visible_scope;
}
static uint64_t vk_texture_registration_scope_token(void) {
    return exec.registration_scope == visible_scope ? 0 : exec.registration_scope;
}
static jmp_buf fatal_jump;
static int expect_fatal;
static char fatal_message[256];

static void no_effects(void) { assert(!outs&&!declarations&&!emits&&!releases&&!nevents); }
static void before_ready(void) { no_effects(); assert(!touches); }

static void draw_descriptor_grant_release(DrawDescriptorGrant *grant) {
    *grant = (DrawDescriptorGrant){0};
}
static int draw_descriptor_grant_current(const VkExec *ex, const VkPipeInfo *info, const DrawDescriptorGrant *grant,
                                         uint64_t ordinal) {
    assert(ex == &exec && ordinal == ORDINAL);
    return grant->info == info && grant->ordinal == ordinal;
}
static VkExecGraphicsResult draw_descriptor_grant_prepare(VkExec *ex, const VkPipeInfo *info,
                                                           DrawDescriptorGrant *grant, int ordered) {
    before_ready();
    assert(ex == &exec && info == &compute_info && ordered && ex->fence == SCENE && ex->resource_op == ORDINAL);
    descriptor_prepared++;
    assert(!grant->info && !descriptor_consumed);
    if (descriptor_refused) {
        grant->refusal = VK_ERROR_OUT_OF_DEVICE_MEMORY;
        grant->refused_set = 0;
        return VK_EXEC_GRAPHICS_UNAVAILABLE;
    }
    grant->info = info;
    grant->ordinal = ex->resource_op;
    return VK_EXEC_GRAPHICS_READY;
}
static VkExecGraphicsResult dispatch_buffers_prepare(VkExec *ex,VkExecDispatch *dispatch) {
    assert(draw_descriptor_grant_current(ex,dispatch->info,&dispatch->descriptors,ex->resource_op));
    return VK_EXEC_GRAPHICS_READY;
}
static const void *nvn_op_payload(const op_hdr *op) { return op+1; }
static int vk_sync_stopping(void) { return stop; }
static int vk_device_is_lost(void) { return loss; }
static int vk_exec_recording_failed(void) { return failed; }
static _Noreturn void rt_fatal(const char *fmt, ...) {
    assert(expect_fatal);
    va_list ap; va_start(ap,fmt); vsnprintf(fatal_message,sizeof fatal_message,fmt,ap); va_end(ap);
    longjmp(fatal_jump,1);
}
static int rc_registry_position_valid(RcRegistryPosition p) { return p.scene && p.ordinal; }
static RcRegistryPosition vk_exec_resource_position(void);
static void vk_exec_turn(void) { abort(); }
static int vk_exec_behind(void) { return 0; }
static void compute_samples_release(ComputeSamples **samples) {
    if (!*samples) return;
    assert(*samples == &selected_samples);
    *samples = NULL;
    sample_releases++;
}
static int compute_samples_current(const VkExec *ex, const VkPipeInfo *info, const ComputeSamples *samples) {
    assert(ex == &exec && info == &compute_info);
    if (!samples) return 1;
    return samples == &selected_samples && samples->ready && samples->info == info &&
        !rc_registry_position_compare(samples->read, vk_exec_resource_position());
}
static VkExecGraphicsResult compute_samples_select(VkExec *ex, VkExecDispatch *dispatch, RcRegistryPosition attempted,
    VkExecGpaSampleScope **pending) {
    RcRegistryPosition read = vk_exec_resource_position();
    assert(ex == &exec && pending && !*pending && !declarations && !outs);
    assert(!rc_registry_position_compare(read, dispatch->read_position));
    assert(!attempted.scene || !rc_registry_position_compare(read, attempted));
    if (compat_samples) {
        assert(dispatch->selection != &selection && !dispatch->selection->samples);
        sample_preparations++;
        if (compat_sample_refusal) return VK_EXEC_GRAPHICS_UNAVAILABLE;
        selected_samples = (ComputeSamples){.info = dispatch->info, .read = read};
        dispatch->selection->samples = &selected_samples;
    }
    return VK_EXEC_GRAPHICS_READY;
}
static VkExecGraphicsResult compute_aliases_prepare(VkExec *ex, ComputeSamples *samples) {
    assert(ex == &exec);
    if (samples) {
        assert(samples == &selected_samples && !samples->ready &&
            !rc_registry_position_compare(samples->read, vk_exec_resource_position()));
        samples->ready = 1;
        sample_materializations++;
    }
    return VK_EXEC_GRAPHICS_READY;
}
static void verify_compat_sample_binding(VkExec *ex, const VkPipeInfo *info) {
    if (!compat_selected()) return;
    assert(!dispatch_current && effect_current == &compat_effect && declarations == 1);
    if (compat_samples) {
        assert(sample_preparations == 1 && sample_materializations == 1 && !sample_releases);
        assert(compute_samples_current(ex, info, effect_current->compute_samples));
        compat_bindings++;
    }
    if (compat_storage)
        assert(effect_current->image_count == 1 && effect_current->images == &compatibility_image &&
               storage_target && compatibility_image.uploaded && !compat_storage_releases);
}
static void vk_exec_cancel_temporal_preparation(void) { stop=1; }
static void vk_exec_required_graphics_unavailable(VkExec *ex, uint32_t opcode, const char *reason) {
    assert(ex == &exec && opcode == selected_opcode && reason);
    failed = RECORDING_RESOURCE_UNAVAILABLE;
}
static VkResult vk_exec_draw_storage_standins_prepare(VkExec *ex,const VkPipeInfo *info,int ordered) {
    assert(ex==&exec&&info==&compute_info&&(ordered==0||ordered==1));
    standins_checked++;
    return standins_result;
}
static VkMemSpanCapture *vk_mem_span_capture(uint64_t address,uint64_t size,const VkMemSpan *span) {
    assert(address==argument_address&&size==12&&span->buffer==arg_buffer&&span->offset==arg_offset&&!captures);
    captured_span = (VkMemSpanCapture){.span=*span,.address=address,.size=size,.mapping=mapping}; captures++;references=1;
    return &captured_span;
}
static RcRegistryResult vk_mem_span_native_input(uint64_t address,uint64_t size,int clamp,VkMemSpan *span,
                                                 VkMemSpanCapture **out) {
    assert(address==argument_address&&size==12&&!clamp&&!*out);
    if (!native_indirect) return RC_REGISTRY_UNSUPPORTED;
    native_receipts++;
    assert(!references);
    *span=(VkMemSpan){mirror_buffer,arg_offset,71};
    captured_span=(VkMemSpanCapture){.span=*span,.address=address,.size=size,.mapping=mapping};
    references=1;*out=&captured_span;
    return RC_REGISTRY_OK;
}
static VkMemSpanCapture *vk_mem_span_capture_retain(VkMemSpanCapture *hold) {
    assert(hold==&captured_span&&references);
    references++;
    return hold;
}
static int vk_mem_span_capture_current(const VkMemSpanCapture *hold,const VkMemSpan *span) {
    return hold && hold == &captured_span && hold->mapping == mapping &&
        hold->span.buffer == span->buffer && hold->span.offset == span->offset;
}
static void vk_mem_span_capture_release(VkMemSpanCapture **hold) {
    if (*hold) {
        assert(*hold==&captured_span&&references);
        if (!--references) capture_releases++;
        *hold=NULL;
    }
}
static int vk_mem_span_held_argument_current(const VkMemSpanCapture *hold, uint64_t address,
                                             uint64_t size, const VkMemSpan *span) {
    return vk_mem_span_capture_current(hold, span) && hold->address == address && hold->size == size;
}
static RcRegistryResult vk_mem_span_prepare_held_argument(const VkMemSpanCapture *hold, const VkMemSpan *span, int ordered) {
    assert(ordered && !declarations && !outs && vk_exec_resource_position().ordinal == 2 * ORDINAL);
    held_prepares++;
    return vk_mem_span_held_argument_current(hold, argument_address, 12, span) ? RC_REGISTRY_OK : RC_REGISTRY_STALE;
}
static int vk_mem_span_record_held_argument(VkMemSpanCapture *hold, uint64_t address, uint64_t bytes, const VkMemSpan *span) {
    if (!vk_mem_span_held_argument_current(hold, address, bytes, span)) return 0;
    held_reads++;
    record(E_USE);
    return 1;
}
static void vk_exec_record_native_input_provider(VkExec *ex,VkMemSpanCapture **hold) {
    assert(ex==&exec&&native_indirect&&references&&*hold==&captured_span);
    physical_adoptions++;
    native_provider_uses++;
    vk_mem_span_capture_release(hold);
}
static void graphics_argument_release(void *owner) {
    VkMemSpanCapture *hold=owner; vk_mem_span_capture_release(&hold);
}
static void graphics_argument_cancel(void *owner,int may_have_executed) {
    (void)may_have_executed; graphics_argument_release(owner);
}
static void vk_completion_add_owned(uint64_t scene,void (*done)(void *),void (*cancel)(void *,int),void *owner) {
    assert(scene==SCENE&&done==graphics_argument_release&&cancel==graphics_argument_cancel&&
           owner==&captured_span&&!completed_span);
    complete_span=done; completed_span=owner;
}
static void finish_capture(void) {
    if (completed_span) {
        complete_span(completed_span);
        completed_span=NULL;
    }
    native_provider_completions += native_provider_uses;
    native_provider_uses = 0;
}
static uint32_t vk_exec_pool_shader_kinds(const GfxShader *shader) {
    assert(shader==&old_shader); return 7;
}
static int vk_exec_pool_grant_current(const VkExec *ex,const VkExecPoolGrant *grant) {
    assert(ex==&exec);
    if (!grant || !grant->held) return 0;
    RcRegistryPosition read = vk_exec_resource_position();
    assert(read.scene == SCENE && read.ordinal == ORDINAL * 2 &&
           "pool grant validation must name the dispatch's original read");
    return !later_pool_authority;
}
static void vk_exec_pool_grant_release(VkExecPoolGrant **out) {
    if (*out) {
        assert(*out==&pool_grant&&pool_grant.held);
        pool_grant.held=0; *out=NULL; grant_releases++;
    }
}
static VkExecPoolBindScope vk_exec_pool_bind_enter(VkExecPoolBindScope next) {
    VkExecPoolBindScope old=binding;
    binding=next; return old;
}
static VkExecGraphicsResult vk_exec_pool_prepare_info(VkExec *ex,uint32_t kinds,VkExecPoolGrant **out,
    int ordered,int allow_prefix,RcRegistryPosition attempted,VkExecGpaSampleScope **pending,void *pass) {
    assert(ex==&exec&&kinds==7&&ordered&&!pass&&!*out);
    pool_calls++;
    if (allow_prefix) {
        before_ready();
        assert(pending && !*pending && vk_exec_resource_position().ordinal==ORDINAL*2);
        if (prospective) {
            prospective--;
            assert(!sample_scope.owned);
            sample_scope.owned=1; *pending=&sample_scope;
            return VK_EXEC_GRAPHICS_NEEDS_ORDERED;
        }
        assert(!settles || (attempted.scene==SCENE&&attempted.ordinal==ORDINAL*2));
    } else {
        assert(!pending && compat_ambient && ambient_bound);
        return VK_EXEC_GRAPHICS_NEEDS_ORDERED;
    }
    *out=&pool_grant; pool_grant.held=1; grants++;
    return VK_EXEC_GRAPHICS_READY;
}
static VkExecGraphicsResult compute_pool_prepare(VkExec *ex,VkExecDispatch *dispatch,
    RcRegistryPosition attempted,VkExecGpaSampleScope **pending) {
    return vk_exec_pool_prepare_info(ex,vk_exec_pool_shader_kinds(dispatch->shader),&dispatch->pool_grant,
        1,1,attempted,pending,NULL);
}
static int vk_exec_shared(void) { return 0; }
static int vk_gpa_write_scene_may_need_prefix(uint64_t scene) { assert(scene == SCENE); return uniform_blocked; }
static int vk_exec_dispatch_uniform_gpa_blocker(VkExec *ex, const VkExecDispatch *dispatch, VkGpaPrefixRequest *request) {
    before_ready();
    assert(ex == &exec && dispatch->selection == &selection && dispatch->shader == &old_shader &&
           dispatch->read_position.scene == SCENE && dispatch->read_position.ordinal == 2 * ORDINAL);
    uniform_probes++;
    if (!uniform_blocked) return 0;
    *request = (VkGpaPrefixRequest){.producer = {SCENE, 5}, .read = dispatch->read_position,
                                    .guard = 9, .source = {11}};
    return 1;
}
static VkExecGpaSampleScope *vk_exec_gpa_sample_enter(VkExec *ex, RcRegistryPosition attempted) {
    before_ready();
    assert(ex == &exec && !attempted.ordinal && !sample_scope.owned && !uniform_requests);
    sample_scope = (VkExecGpaSampleScope){.owned = 1};
    return &sample_scope;
}
static int vk_exec_request_gpa_prefix(VkExec *ex, int guard, RcRegistryPosition producer,
                                      RcRegistryPosition read, struct Source source) {
    before_ready();
    assert(ex == &exec && guard == 9 && producer.ordinal == 5 && source.range == 11 &&
           read.scene == SCENE && read.ordinal == 2 * ORDINAL);
    uniform_requests++;
    return 1;
}
static int vk_exec_gpa_sample_leave(VkExecGpaSampleScope *scope) {
    assert(scope == &sample_scope && scope->owned && scope->prepare.uniform && uniform_requests == 1);
    return 1;
}
static void vk_gpa_report_write_blocker(int guard, int source) { assert(guard == 9 && source == 11); }
static VkSyncProgressResult vk_exec_gpa_sample_settle(VkExecGpaSampleScope *sample,RcRegistryPosition *attempted) {
    assert(sample==&sample_scope && sample->owned && !exec.resource_op && !declarations && !emits &&
           !attempted->ordinal && exec.resource_next+1==ORDINAL);
    if (sample->prepare.uniform) {
        assert(uniform_blocked && uniform_requests == 1 && !pool_calls);
        uniform_blocked = 0;
    }
    *sample = (VkExecGpaSampleScope){0}; settles++;
    *attempted=(RcRegistryPosition){SCENE,ORDINAL*2};
    if (replace_on_settle) mapping++;
    return VK_SYNC_PROGRESS_COMPLETE;
}
static void vk_exec_gpa_sample_cancel(VkExecGpaSampleScope *sample) {
    assert(sample==&sample_scope&&sample->owned);
    sample->owned=0; pending_cancels++;
}
static RcRegistry *rc_registry_runtime(void) { return &registry; }
static int rc_registry_declare_through(RcRegistry *r,RcRegistryPosition pos) {
    assert(r==&registry&&pos.scene==SCENE&&!parked);
    if (pos.ordinal==2*ORDINAL) {
        before_ready();
        if (ins) {
            assert(exec.resource_next == ORDINAL && exec.resource_op == ORDINAL && !outs);
            read_redeclarations++;
        } else {
            assert(exec.resource_next+1 == ORDINAL);
            ins++;
        }
    } else {
        assert(pos.ordinal==2*ORDINAL+1&&ins==1&&exec.resource_next==ORDINAL&&!outs); outs++;
    }
    return RC_REGISTRY_OK;
}
static void group_drain(VkExec *ex,int reason) {
    before_ready(); assert(ex==&exec&&reason==PW_OP&&!parked);
    drains++; if(fail_drain)failed=1;
}
static VkExecDispatchEffectResult vk_exec_dispatch_effect_prepare(VkExec *ex,uint32_t op,const void *payload,
    const GfxShader **shader,VkExecDispatchImages **out) {
    before_ready();
    assert(ex==&exec&&ex->registration_scope==SCOPE&&ex->prog==&program&&ex->resource_op==ORDINAL);
    assert(payload==selected_payload&&op==selected_opcode&&dispatch_current==(compat_selected() ? NULL : &caller)&&ins==1&&!selection.owned);
    assert(ex->resource_next+1==ORDINAL&&lookups==(op==OP_DISPATCH_IND && !native_indirect));
    selects++;
    *shader=visible_shader;
    *out=&selection;
    selection.owned=1;
    return effect_answer;
}
static void dispatch_compat_storage_release(VkExecDispatchImages *selection);
static void vk_exec_dispatch_effect_cancel(VkExecDispatchImages **out) {
    assert(out&&!parked);
    if (!*out) return;
    if (*out == &selection) {
        assert(selection.owned == 1);
        selection.owned = 0;
        selection_cancels++;
    } else {
        compute_samples_release(&(*out)->samples);
        if ((*out)->count) dispatch_compat_storage_release(*out);
        free(*out);
        compat_selection_releases++;
    }
    *out = NULL;
}
static VkComputePipelineResult vk_pipeline_compute_prepare(const GfxShader *shader,int create,VkPipeline *pipe,
        const VkPipeInfo **out,VkComputePipelinePending *pending) {
    before_ready();
    assert(shader==&old_shader&&create==parked&&submit_locked==!parked&&!*pipe&&!*out&&!failed);
    assert(ins==1&&!exec.resource_op&&exec.resource_next+1==ORDINAL);
    if(create)builds++;else probes++;
    *pending=(VkComputePipelinePending){.shader=shader,.kind=VK_COMPUTE_CAPACITY};
    pending->peak.amount[RC_BUDGET_RAM]=pending->peak.amount[RC_BUDGET_COMMIT]=PEAK;
    if((!create&&terminal_probe)||(create&&terminal_build)){
        pending->failure="mock exact compiler refusal";pending->admission=RC_ADMISSION_REFUSED;pending->error=VK_ERROR_OUT_OF_DEVICE_MEMORY;
        return VK_COMPUTE_TERMINAL;
    }
    if((create&&builds>compiler_delays)||(!create&&cache)){
        *pipe=compute_pipeline;*out=&compute_info;return VK_COMPUTE_READY;
    }
    return VK_COMPUTE_PENDING;
}
static int vk_pipeline_compute_wait(const VkComputePipelinePending *pending) {
    before_ready();
    assert(parked&&pending->shader==&old_shader&&pending->peak.amount[RC_BUDGET_RAM]==PEAK&&
        pending->peak.amount[RC_BUDGET_COMMIT]==PEAK);
    visible_shader=&later_shader;visible_scope=SCOPE+100;waits++;
    if(replace_scope_while_waiting)exec.registration_scope=visible_scope;
    if(stop_wait)stop=1;
    return !stop;
}
static void plat_mutex_unlock(int *lock) {
    before_ready();
    assert(lock==&sub_lock&&submit_locked&&recorder_parked&&!recording_scene&&drains==1&&probes==1);
    assert(!exec.resource_op&&exec.resource_next+1==ORDINAL);
    submit_locked=0;parks++;
}
static void plat_mutex_lock(int *lock) {
    assert(lock==&sub_lock&&!submit_locked&&recorder_parked&&!recording_scene);
    submit_locked=1;
}
static void plat_cond_broadcast(int *condition) {
    assert(condition==&recorder_resumed&&submit_locked&&!recorder_parked&&recording_scene==SCENE);
    resumes++;
}
static void hold_release(uint64_t limit,int wait,int pad) {
    before_ready();
    assert(limit==SCENE&&!wait&&!pad&&!submit_locked&&parked&&!recording_scene);
    handoffs++;
    if(stop_handoff)stop=1;
}
typedef struct { uint32_t count; } TestUboMerge;
struct VkExecBufferScratchReceipt {
    const VkPipeInfo *info;
    RcRegistryPosition read;
    TestUboMerge ubo;
    struct { VkBuffer buf; } chunk;
    VkDeviceSize start;
    int exposed, ubo_materialized;
};
static VkExecBufferScratchReceipt scratch_receipt, *effect_scratch;
static VkExecBufferScratchReceipt *buffer_scratch_adopt(VkExecBufferScratchReceipt **source,
    const VkExec *ex, uint32_t opcode, const void *payload, const GfxShader *shader, RcRegistryPosition output) {
    if (!*source) return NULL;
    assert(ex && opcode == selected_opcode && payload && shader && output.scene == ex->fence && output.ordinal == ORDINAL * 2 + 1);
    VkExecBufferScratchReceipt *scratch = *source;
    *source = NULL;
    return scratch;
}
static int request_scratch;
static unsigned scratch_admissions, scratch_releases, scratch_exposures;
static int compute_ubo_merges_current(const VkExec *ex, const VkExecBufferScratchReceipt *scratch) {
    return !scratch || (scratch == &scratch_receipt && scratch->info == &compute_info &&
        scratch->read.scene == ex->fence && scratch->read.ordinal == ORDINAL * 2);
}
static VkExecBufferScratchReceipt *buffer_scratch_recording(VkExec *ex) {
    assert(ex == &exec);
    if (effect_scratch) assert(!rc_registry_position_compare(effect_scratch->read, vk_exec_resource_position()));
    return effect_scratch;
}
static int ubo_merges_record_native(VkExec *ex, const TestUboMerge *ubo, VkBuffer buffer, VkDeviceSize start, int *copied) {
    assert(ex == &exec && effect_scratch && ubo == &effect_scratch->ubo && ubo->count == 1 &&
        buffer == const_buffer && start == c0_offset && !*copied);
    record(E_UNIFORM_COPY);
    *copied = 1;
    return 1;
}
static void buffer_scratch_expose(VkExec *ex, VkExecBufferScratchReceipt *scratch) {
    assert(ex == &exec && scratch == effect_scratch && !scratch->exposed);
    scratch->exposed = 1;
    scratch_exposures++;
}
static void buffer_scratch_cancel(VkExecBufferScratchReceipt **receipt) {
    if (!*receipt) return;
    assert(*receipt == &scratch_receipt);
    *receipt = NULL;
    scratch_releases++;
}
static int buffer_scratch_indirect_c0(VkExec *ex, const VkMemSpan *argument) {
    assert(ex == &exec && argument);
    return 0;
}
static VkExecDispatchResult prepare_dispatch_scratch(VkExec *ex, VkExecDispatch *dispatch, RcRegistryPosition read) {
    assert(ex == &exec && !parked && !ex->resource_op && ex->resource_next + 1 == ORDINAL);
    assert(dispatch->selection && (dispatch->compatibility ? compat_selected() : dispatch->selection == &selection));
    assert(read.scene == SCENE && read.ordinal == 2 * ORDINAL);
    assert(!dispatch->scratch && !declarations && !outs);
    if (request_scratch) {
        scratch_receipt = (VkExecBufferScratchReceipt){.info = dispatch->info, .read = read,
            .ubo.count = 1, .chunk.buf = const_buffer, .start = c0_offset};
        dispatch->scratch = &scratch_receipt;
        scratch_admissions++;
    }
    if (remap_after_scratch) mapping++;
    return VK_EXEC_DISPATCH_NATIVE;
}
static void *vk_exec_effects_declare_dispatch(VkExec *ex,uint32_t op,const void *payload,const GfxShader *shader,
    VkExecDispatchImages **inputs,RcRegistryPosition pos,VkExecBufferScratchReceipt **scratch) {
    assert(scratch && !effect_scratch && (!*scratch || *scratch == &scratch_receipt));
    no_effects();
    assert(!parked&&!stop&&!loss&&!failed&&ex==&exec&&payload==selected_payload&&op==selected_opcode);
    assert(shader==&old_shader&&*inputs==&selection&&selection.owned==1&&dispatch_current==&caller);
    assert(!touches && !completed_span && !held_reads &&
           held_prepares==(op==OP_DISPATCH_IND));
    assert(ex->resource_op==ORDINAL&&ex->resource_next==ORDINAL&&pos.scene==SCENE&&pos.ordinal==2*ORDINAL+1);
    *inputs=NULL;selection.owned=2;declarations++;
    effect_scratch = *scratch;
    *scratch = NULL;
    if (publish_disjoint_output) later_pool_authority=1;
    return &transaction;
}
typedef struct { int *target; } VkExecClearTexture;
typedef struct { int *target; } VkExecBufferToTexture;
typedef struct { int *target; } VkExecTextureToTexture;
typedef struct { int dummy; } VkExecRestoreCopy, VkExecProjectedTransfer, VkExecBufferTransfer;
typedef int VkTextureWriteTarget;
static void *resource_declare_texture_write(VkExec *ex,uint32_t op,int *target) { (void)ex;(void)op;(void)target;abort(); }
static void *vk_exec_effects_declare_buffer_transfer(VkExec *ex, uint32_t op, const void *payload,
    VkExecBufferTransfer *buffer, RcRegistryPosition output) {
    (void)ex; (void)op; (void)payload; (void)buffer; (void)output; abort();
}
static void *resource_declare_projected_transfer(VkExec *ex, VkExecProjectedTransfer *transfer) {
    (void)ex; (void)transfer; abort();
}
static void *vk_exec_effects_declare_texture_buffer(VkExec *ex, const void *payload, VkExecRestoreCopy *restore,
    RcRegistryPosition output) {
    (void)ex; (void)payload; (void)restore; (void)output; abort();
}
static int dispatch_compat_inputs_transfer(VkExec *, EffectDeclaration *, uint32_t, const void *, VkExecDispatch *);
static void *vk_exec_effects_declare(VkExec *ex, uint32_t op, const void *payload, RcRegistryPosition output,
                                     VkExecDispatch *compat_dispatch) {
    assert(compat_selected() && compat_dispatch && !dispatch_current && read_redeclarations == 1 &&
        !outs && !declarations && sample_preparations == (unsigned)compat_samples &&
        sample_materializations == (unsigned)compat_samples);
    assert(op == selected_opcode && payload == selected_payload && output.scene == SCENE &&
        output.ordinal == ORDINAL * 2 + 1);
    compat_effect = (EffectDeclaration){.program = ex->prog, .program_serial = ex->prog->serial, .opcode = op, .position = output};
    assert(dispatch_compat_inputs_transfer(ex, &compat_effect, op, payload, compat_dispatch));
    assert(compat_effect.compute_samples == (compat_samples ? &selected_samples : NULL) &&
           !compat_dispatch->selection->samples);
    assert(compat_effect.image_count == (unsigned)compat_storage &&
           compat_effect.images == (compat_storage ? &compatibility_image : NULL) &&
           !compat_dispatch->selection->count && !compat_dispatch->selection->images);
    declarations++;
    return &compat_effect;
}
static void *vk_exec_effects_enter(void *effect) {
    assert(effect==(noop?NULL:compat_selected()?(void *)&compat_effect:&transaction)&&outs==1);
    assert(!effect_current);
    if (compat_selected()) effect_current = effect;
    return NULL;
}
static void op_dispatch(VkExec *ex,const p_dispatch *d);
static void op_dispatch_ind(VkExec *ex,const p_dispatch_ind *d);
static void emit_dispatch(VkExec *ex,const op_hdr *h) {
    if (compat_selected()) assert(!dispatch_current && effect_current == &compat_effect);
    else {
        assert(dispatch_current!=&caller&&dispatch_current->shader==&old_shader&&dispatch_current->registration==SCOPE);
        assert(dispatch_current->payload==selected_payload&&dispatch_current->pipeline==compute_pipeline);
    }
    assert(declarations==1&&!emits&&!parked);
    if(h->op==OP_DISPATCH)op_dispatch(ex,nvn_op_payload(h));else op_dispatch_ind(ex,nvn_op_payload(h));
    for(unsigned i=0;i<nevents;i++)emits+=events[i]==E_DISPATCH||events[i]==E_INDIRECT;
}
static void vk_exec_effects_recorded(void *e,int v) {
    assert(e==(noop?NULL:compat_selected()?(void *)&compat_effect:&transaction)&&!v&&emits==!noop);
}
static void vk_exec_effects_leave(void *prior) { assert(!prior); effect_current = NULL; }
static void vk_exec_effects_release(void *e) {
    releases++;
    if (e == &compat_effect) {
        draw_descriptor_grant_release(&compat_effect.compat_descriptors);
        compute_samples_release(&compat_effect.compute_samples);
        if (compat_effect.image_count) {
            VkExecDispatchImages ownership={.images=compat_effect.images,.count=compat_effect.image_count};
            dispatch_compat_storage_release(&ownership);
            compat_effect.images=NULL;
            compat_effect.image_count=0;
        }
    }
    else if (e) { assert(e == &transaction && selection.owned == 2); selection.owned = 0; }
    else assert(noop);
    buffer_scratch_cancel(&effect_scratch);
}
'''

TESTS = r'''
static const struct { op_hdr op; p_dispatch payload; } direct = {{OP_DISPATCH,sizeof direct},{2,3,5,149}};
static const struct { op_hdr op; p_dispatch_ind payload; } indirect = {{OP_DISPATCH_IND,sizeof indirect},{0x3000,151}};

static const struct { op_hdr op; p_dispatch payload; } zero = {{OP_DISPATCH,sizeof zero},{0,3,5,149}};

static void reset(const op_hdr *op) {
    finish_capture();
    captures=capture_releases=0; mapping=1;
    native_indirect=remap_on_bind=remap_after_scratch=0;
    native_receipts=physical_adoptions=0; assert(!references);
    native_provider_completions = 0;
    standins_checked=0;standins_result=VK_SUCCESS;
    replace_on_settle=prospective=compat_ambient=ambient_bound=0;
    uniform_blocked=uniform_probes=uniform_requests=0;
    grants=grant_releases=settles=pending_cancels=pool_calls=warnings=0;
    gpa_attempted=(RcRegistryPosition){0}; compute_info.pool=0;
    publish_disjoint_output=later_pool_authority=0;
    assert(!pool_grant.held && !sample_scope.owned && !binding.info);
    memset(&draw_context,0,sizeof draw_context);
    exec=(VkExec){.fence=SCENE,.resource_next=ORDINAL-1,.registration_scope=SCOPE,.prog=&program,
                  .cb=scene_cb,.dc=&draw_context};
    active=&exec;
    dispatch_current=&caller;visible_shader=&old_shader;visible_scope=SCOPE;
    selected_payload=nvn_op_payload(op);selected_opcode=op->op;
    selects=drains=waits=parks=probes=builds=ins=outs=declarations=emits=releases=0;
    nevents=warns=selections=preparation=touches=lookups=0;
    retained_storage=storage_texture_locked=storage_begun=storage_emitted=storage_written=0;
    storage_access_invalid=storage_access_checks=storage_seed_requests=0;
    compiler_delays=selection_cancels=0;
    read_redeclarations=held_prepares=held_reads=0;
    compat_samples=compat_sample_refusal=compat_bindings=0;
    compat_raw_storage=compat_image_bytes=0;
    compat_storage=compat_storage_refused=0;
    compat_storage_prepares=compat_storage_releases=0;
    assert(!storage_target);
    sample_preparations=sample_materializations=sample_releases=compat_selection_releases=0;
    request_scratch=scratch_admissions=scratch_releases=scratch_exposures=0;
    assert(!effect_current && !effect_scratch);
    selection=(VkExecDispatchImages){0};
    effect_answer=VK_EXEC_DISPATCH_EFFECT_READY;
    vk_exec_frame.dispatches=split_draws=0;
    deferred_binding=nvn2_mode=missing_indirect=0;
    stop=loss=failed=parked=stop_wait=stop_handoff=fail_drain=terminal_probe=terminal_build=cache=noop=0;
    replace_scope_while_waiting=0;
    descriptor_refused=descriptor_prepared=descriptor_consumed=0;
    recording_scene=SCENE;submit_locked=1;handoffs=resumes=0;
    expect_fatal=0;memset(fatal_message,0,sizeof fatal_message);
    dispatch_input_line[0] = 0;
}
static void no_operation(void) {
    no_effects();
    assert(!exec.resource_op&&exec.resource_next+1==ORDINAL&&dispatch_current==&caller&&!parked&&submit_locked);
}
static void check_retained(void) {
    const op_hdr *ops[]={&direct.op,&indirect.op,&indirect.op};
    for(unsigned i=0;i<3;i++){
        const op_hdr *op=ops[i];reset(op);compiler_delays=2;nvn2_mode=i==2;
        assert(!vk_texture_registration_scope_token());
        NvnExecStep step={.opcode=op->op,.ops=1};
        assert(run_selected(&exec,op,i?&step:NULL));
        assert(selects==1&&probes==1&&parks==1&&handoffs==1&&resumes==1&&drains==1&&waits==2&&builds==3);
        assert(ins==1&&outs==1&&declarations==1&&emits==1&&releases==1);
        assert(descriptor_prepared == 1 && descriptor_consumed == 1 && !descriptor_binding_grant);
        assert(!parked&&!selection.owned&&!selection_cancels&&!exec.resource_op&&exec.resource_next==ORDINAL);
        assert(dispatch_current==&caller&&visible_shader==&later_shader&&visible_scope!=SCOPE);
        assert(vk_texture_registration_scope_token()==SCOPE&&vk_texture_registration_publication()==SCOPE);
        assert(!preparation&&!selections&&!warns&&lookups==(i!=0)&&!touches);
        assert(held_prepares == (i != 0) && held_reads == (i != 0));
        static const enum Event direct_order[]={E_BARRIER,E_PIPE_BIND,E_RESOURCES,E_DISPATCH,E_GPA_PROOF,E_WRITES};
        static const enum Event indirect_order[]={E_BARRIER,E_USE,E_PIPE_BIND,E_RESOURCES,E_COPY,E_COPY_BARRIER,
            E_INDIRECT,E_WRITES,E_INDIRECT_COUNT,E_INDIRECT_BOUND};
        if(i)expect(indirect_order,sizeof indirect_order/sizeof indirect_order[0]);
        else expect(direct_order,sizeof direct_order/sizeof direct_order[0]);
        assert(exec.recorded==(i?2:1));
    }
}
static void check_descriptor_refusal(void) {
    reset(&direct.op);
    descriptor_refused = 1;
    retained_storage = 1;
    assert(!run_selected(&exec, &direct.op, NULL));
    assert(failed && descriptor_prepared == 1 && !descriptor_consumed && !descriptor_binding_grant);
    assert(!outs && !declarations && !emits && !releases && !selection.owned);
    assert(!storage_begun && !storage_emitted && !storage_written);
}
static void check_retained_storage_emission(void) {
    reset(&direct.op);
    retained_storage = 1;
    assert(run_selected(&exec, &direct.op, NULL));
    assert(storage_access_checks == 1 && storage_seed_requests == 1);
    assert(emits == 1 && storage_begun == 1 && storage_emitted == 1 && storage_written == 1 &&
           !storage_texture_locked && exec.recorded == 1);
    assert(!dispatch_input_line[0]);
}
static void check_invalid_storage_cannot_seed(void) {
    reset(&direct.op);
    retained_storage = storage_access_invalid = 1;
    assert(!run_selected(&exec, &direct.op, NULL));
    assert(failed && storage_access_checks == 1 && !storage_seed_requests && !storage_begun &&
           !storage_emitted && !storage_written && !emits && !declarations);
    char read[96], status[64];
    snprintf(read, sizeof read, "required dispatch input read=%llu/%llu", (unsigned long long)SCENE,
             2ull * ORDINAL);
    snprintf(status, sizeof status, "status=%d stage-result=%d", VK_EXEC_GRAPHICS_UNAVAILABLE,
             VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(strstr(dispatch_input_line, read) && strstr(dispatch_input_line, "path=native stage=storage-access") &&
           strstr(dispatch_input_line, status));
}
static void check_incompatible_registration(void) {
    reset(&direct.op);compiler_delays=1;replace_scope_while_waiting=1;
    assert(!run_selected(&exec,&direct.op,NULL) && failed && warnings==1);
    assert(parks==1&&resumes==1&&waits==1&&vk_texture_registration_publication()!=SCOPE);
    assert(selection_cancels==1&&!selection.owned);
    no_operation();
}
static void check_cached_and_compat(void) {
    reset(&direct.op);cache=1;
    assert(run_selected(&exec,&direct.op,NULL));
    assert(selects==1&&probes==1&&!builds&&!parks&&!drains&&declarations==1&&emits==1&&!selection.owned);

    reset(&direct.op);effect_answer=VK_EXEC_DISPATCH_EFFECT_COMPAT;
    VkExecDispatch dispatch={0};
    assert(prepare_ordered_dispatch(&exec,&direct.op,&dispatch,&gpa_attempted,NULL,NULL)==VK_EXEC_DISPATCH_COMPAT);
    assert(selects==1&&selections==1&&selection_cancels==1&&!selection.owned&&!dispatch.shader&&!probes&&!parks);
    no_operation();
}
static void check_compat_sample_handoff(void) {
    reset(&direct.op);
    cache = compat_samples = 1;
    effect_answer = VK_EXEC_DISPATCH_EFFECT_COMPAT;
    dispatch_current = NULL;
    assert(run_selected(&exec, &direct.op, NULL));
    assert(probes == 1 && selects == 1 && selections == 1 && !preparation);
    assert(ins == 1 && read_redeclarations == 1 && outs == 1 && declarations == 1 && emits == 1 && releases == 1);
    assert(sample_preparations == 1 && sample_materializations == 1 && compat_bindings == 1 && sample_releases == 1 &&
        compat_selection_releases == 1);
    assert(!effect_current && !dispatch_current && descriptor_prepared == 1 && descriptor_consumed == 1 && !effect_scratch);
    static const enum Event order[] = {E_BARRIER,E_PIPE_BIND,E_RESOURCES,E_DISPATCH,E_GPA_PROOF,E_WRITES};
    expect(order, sizeof order / sizeof order[0]);

    reset(&direct.op);
    cache = compat_samples = compat_sample_refusal = 1;
    effect_answer = VK_EXEC_DISPATCH_EFFECT_COMPAT;
    dispatch_current = NULL;
    assert(!run_selected(&exec, &direct.op, NULL));
    assert(failed && sample_preparations == 1 && !sample_materializations && !compat_bindings && !sample_releases);
    assert(compat_selection_releases == 1);
    assert(ins == 1 && !read_redeclarations && !effect_current && !dispatch_current);
    no_effects();
}
static void check_compat_storage_handoff(void) {
    static const enum Event order[] = {E_BARRIER,E_PIPE_BIND,E_RESOURCES,E_DISPATCH,E_GPA_PROOF,E_WRITES};
    for (int has_sample=0; has_sample<=1; has_sample++) {
        reset(&direct.op);
        cache=compat_storage=1;
        compat_samples=has_sample;
        retained_storage=1;
        exec.img[NVN_STAGE_CS][2]=31;
        effect_answer=VK_EXEC_DISPATCH_EFFECT_COMPAT;
        dispatch_current=NULL;
        assert(run_selected(&exec,&direct.op,NULL));
        assert(drains==1 && ins==1 && read_redeclarations==1 && outs==1 && declarations==1 &&
               emits==1 && releases==1 && exec.recorded==1);
        assert(compat_storage_prepares==1 && compat_storage_releases==1 && compat_selection_releases==1 && !storage_target);
        assert(sample_preparations==(unsigned)has_sample && sample_materializations==(unsigned)has_sample &&
               sample_releases==(unsigned)has_sample && compat_bindings==(unsigned)has_sample);
        assert(storage_begun==1 && storage_emitted==1 && storage_written==1 && !storage_texture_locked &&
               !effect_current && !dispatch_current);
        expect(order,sizeof order/sizeof order[0]);
    }

    reset(&direct.op);
    cache=compat_storage=compat_storage_refused=1;
    retained_storage=1;
    exec.img[NVN_STAGE_CS][2]=31;
    effect_answer=VK_EXEC_DISPATCH_EFFECT_COMPAT;
    dispatch_current=NULL;
    assert(!run_selected(&exec,&direct.op,NULL));
    assert(failed && drains==1 && ins==1 && compat_storage_prepares==1 && compat_storage_releases==1 &&
           compat_selection_releases==1 && !storage_target);
    assert(!outs && !declarations && !emits && !storage_begun && !storage_emitted && !storage_written);
    assert(strstr(dispatch_input_line,"path=compat stage=compat-storage-output"));
}
static void check_compat_raw_selection(void) {
    reset(&direct.op);
    exec.resource_op=ORDINAL;
    compat_raw_storage=1;
    VkExecDispatch dispatch={0};
    assert(!vk_exec_dispatch_compat_inputs_select(&exec,direct.op.op,nvn_op_payload(&direct.op),&dispatch));
    assert(!dispatch.selection);
    compat_image_bytes=1;
    assert(vk_exec_dispatch_compat_inputs_select(&exec,direct.op.op,nvn_op_payload(&direct.op),&dispatch));
    assert(dispatch.compatibility && dispatch.selection && dispatch.read_position.scene==SCENE &&
           dispatch.read_position.ordinal==2*ORDINAL);
    vk_exec_dispatch_cancel(&dispatch);
    assert(!dispatch.selection && compat_selection_releases==1);
    exec.resource_op=0;
    no_operation();
}
static void check_native_scratch_handoff(void) {
    reset(&direct.op);
    cache = request_scratch = 1;
    assert(run_selected(&exec, &direct.op, NULL));
    assert(scratch_admissions == 1 && scratch_exposures == 1 && scratch_releases == 1 && !effect_scratch);
    assert(scratch_receipt.ubo_materialized && exec.recorded == 2);
    static const enum Event order[] = {E_BARRIER,E_PIPE_BIND,E_UNIFORM_COPY,E_RESOURCES,E_DISPATCH,E_GPA_PROOF,E_WRITES};
    expect(order, sizeof order / sizeof order[0]);
}
static void check_stop(void) {
    reset(&indirect.op);stop=1;
    assert(!run_selected(&exec,&indirect.op,NULL));
    assert(!ins&&!selects&&!probes&&!drains&&!parks&&!selection_cancels);no_operation();
    reset(&indirect.op);compiler_delays=1;stop_wait=1;
    assert(!run_selected(&exec,&indirect.op,NULL));
    assert(selects==1&&probes==1&&parks==1&&waits==1&&builds==1&&selection_cancels==1);
    assert(handoffs==1&&resumes==1&&recording_scene==SCENE);no_operation();

    reset(&indirect.op);stop_handoff=1;
    assert(!run_selected(&exec,&indirect.op,NULL));
    assert(selects==1&&probes==1&&drains==1&&parks==1&&handoffs==1&&resumes==1&&!builds&&!waits);
    assert(selection_cancels==1&&recording_scene==SCENE);no_operation();

    reset(&direct.op);fail_drain=1;
    assert(!run_selected(&exec,&direct.op,NULL));
    assert(selects==1&&probes==1&&drains==1&&!parks&&!builds&&selection_cancels==1);no_operation();
}
static void check_terminal(void) {
    for(unsigned on_build=0;on_build<2;on_build++){
        reset(&indirect.op);terminal_probe=!on_build;terminal_build=on_build;expect_fatal=1;
        if(!setjmp(fatal_jump)){run_selected(&exec,&indirect.op,NULL);abort();}
        assert(strstr(fatal_message,"mock exact compiler refusal")&&strstr(fatal_message,"Vulkan -2"));
        assert(selects==1&&probes==1&&parks==on_build&&builds==on_build&&selection_cancels==1);no_operation();
    }
    reset(&direct.op);recording_scene=0;expect_fatal=1;
    if(!setjmp(fatal_jump)){run_selected(&exec,&direct.op,NULL);abort();}
    assert(strstr(fatal_message,"no active recorder"));
    assert(selects==1&&probes==1&&!parks&&!builds&&selection_cancels==1);no_operation();
}
static void check_indirect_retry_and_terminal(void) {
    reset(&indirect.op);cache=1;
    VkExecDispatch dispatch={0};
    assert(prepare_ordered_dispatch(&exec,&indirect.op,&dispatch,&gpa_attempted,NULL,NULL)==VK_EXEC_DISPATCH_NATIVE);
    exec.resource_op = ORDINAL;
    vk_exec_dispatch_validate(&exec,&dispatch,OP_DISPATCH_IND,&indirect.payload);
    exec.resource_op = ORDINAL; // The fixture emulates the walker's actual operation assignment.
    VkExecDispatch *prior=vk_exec_dispatch_enter(&dispatch);
    deferred_binding=1;
    op_dispatch_ind(&exec,&indirect.payload);
    assert(!exec.recorded&&nevents==4&&events[3]==E_RESOURCES);
    deferred_binding=0;
    op_dispatch_ind(&exec,&indirect.payload);
    unsigned copies=0,dispatches=0,writers=0;
    for(unsigned i=0;i<nevents;i++){
        copies+=events[i]==E_COPY;dispatches+=events[i]==E_INDIRECT;writers+=events[i]==E_WRITES;
    }
    assert(copies==1&&dispatches==1&&writers==1&&exec.recorded==2);
    assert(probes==1&&lookups==1&&!touches&&held_reads==2&&!selections&&!preparation);
    vk_exec_dispatch_enter(prior);
    vk_exec_dispatch_cancel(&dispatch);
    assert(selection_cancels==1&&!selection.owned&&dispatch_current==&caller);

    reset(&indirect.op);cache=1;remap_after_scratch=1;request_scratch=1;expect_fatal=1;
    if(!setjmp(fatal_jump)){run_selected(&exec,&indirect.op,NULL);abort();}
    assert(strstr(fatal_message,"changed after pipeline preparation"));
    assert(probes==1&&lookups==1&&!touches&&!held_reads&&!warns&&!exec.recorded&&selection_cancels==1&&!selection.owned);
    assert(scratch_admissions==1&&scratch_releases==1&&!scratch_exposures&&!effect_scratch);
    assert(exec.resource_op==ORDINAL);
    exec.resource_op=0; // The fixture intercepts the fatal before normal control can leave validation.
    no_operation();

    reset(&indirect.op);missing_indirect=1;noop=1;
    assert(run_selected(&exec,&indirect.op,NULL));
    assert(lookups==1&&!touches&&!held_reads&&warns==1&&!selects&&!probes&&!declarations&&!emits);
}
static void check_standin_refusal(void) {
    reset(&indirect.op);cache=1;standins_result=VK_ERROR_FORMAT_NOT_SUPPORTED;
    assert(!run_selected(&exec,&indirect.op,NULL));
    assert(standins_checked==1&&failed&&captures==1&&capture_releases==1&&lookups==1&&!touches);
    assert(ins==1&&!outs&&!selection.owned&&selection_cancels==1);
    no_operation();
}
static void check_noop(void) {
    reset(&zero.op);noop=1;
    assert(run_selected(&exec,&zero.op,NULL));
    assert(!selects&&!selection_cancels&&!probes&&!parks&&!declarations&&!emits);
    assert(ins==1&&outs==1&&dispatch_current==&caller&&vk_exec_frame.dispatches==1&&split_draws==1);
}
static void check_pool_prefix_and_mapping(void) {
    reset(&indirect.op);cache=1;compute_info.pool=1;prospective=1;
    publish_disjoint_output=1;
    assert(run_selected(&exec,&indirect.op,NULL));
    assert(later_pool_authority && !failed);
    assert(probes==1&&selects==1&&pool_calls==2&&settles==1&&
           !pending_cancels&&grants==1&&grant_releases==1&&declarations==1&&emits==1&&completed_span);
    assert(captures==1&&!capture_releases);
    finish_capture();
    assert(capture_releases==1);

    reset(&indirect.op);cache=1;compute_info.pool=1;prospective=1;replace_on_settle=1;
    assert(!run_selected(&exec,&indirect.op,NULL));
    assert(settles==1&&pool_calls==1&&failed&&captures==1&&capture_releases==1&&!completed_span);
    no_operation();

    reset(&direct.op);cache=1;compute_info.pool=1;prospective=2;
    assert(!run_selected(&exec,&direct.op,NULL));
    assert(settles==1&&pool_calls==2&&pending_cancels==1&&failed&&!grants);
    no_operation();

    reset(&direct.op);compute_info.pool=1;compat_ambient=1;
    dispatch_current=NULL;
    exec.resource_op=++exec.resource_next;
    op_dispatch(&exec,&direct.payload);
    assert(ambient_bound && pool_calls==1 && !failed && !grants && !exec.recorded);
    expect((const enum Event[]){E_BARRIER,E_PIPE_BIND},2);
    dispatch_current=&caller;
}
static void check_uniform_prefix_before_dispatch_output(void) {
    reset(&indirect.op); cache = 1; native_indirect = 1; uniform_blocked = 1;
    assert(run_selected(&exec, &indirect.op, NULL));
    assert(uniform_probes == 1 && uniform_requests == 1 && settles == 1 && !uniform_blocked);
    assert(ins == 1 && outs == 1 && selects == 1 && emits == 1 && physical_adoptions == 1 && !references);
    assert(gpa_attempted.scene == SCENE && gpa_attempted.ordinal == 2 * ORDINAL);
    reset(&direct.op); cache = 1; uniform_blocked = 1;
    assert(run_selected(&exec, &direct.op, NULL));
    assert(uniform_probes == 1 && uniform_requests == 1 && settles == 1 && emits == 1);
}
static void check_retained_actual_mirror(void) {
    reset(&indirect.op);cache=1;native_indirect=1;compute_info.pool=1;prospective=1;
    assert(run_selected(&exec,&indirect.op,NULL));
    assert(native_receipts==1 && physical_adoptions==1 && !lookups && !touches && !captures);
    assert(settles==1&&pool_calls==2&&declarations==1&&emits==1&&!references&&capture_releases==1);
    assert(native_provider_uses == 1 && !native_provider_completions);
    finish_capture();
    assert(!native_provider_uses && native_provider_completions == 1);

    reset(&indirect.op);cache=1;native_indirect=1;
    VkExecDispatch dispatch={0}, retry={0};
    assert(prepare_ordered_dispatch(&exec,&indirect.op,&dispatch,&gpa_attempted,NULL,NULL)==VK_EXEC_DISPATCH_NATIVE);
    VkMemSpanCapture *retained=vk_mem_span_capture_retain(dispatch.indirect_hold);
    VkMemSpan original=dispatch.indirect;
    vk_exec_dispatch_cancel(&dispatch);
    assert(references==1&&!capture_releases);
    ins=0; // The production GPA boundary reopens the same logical input prefix.
    assert(prepare_ordered_dispatch(&exec,&indirect.op,&retry,&gpa_attempted,&original,retained)==VK_EXEC_DISPATCH_NATIVE);
    assert(retry.indirect_hold==retained && native_receipts==1 && !touches && !lookups);
    exec.resource_op = ORDINAL;
    vk_exec_dispatch_validate(&exec,&retry,OP_DISPATCH_IND,&indirect.payload);
    exec.resource_op = ORDINAL;
    VkExecDispatch *prior=vk_exec_dispatch_enter(&retry);
    remap_on_bind=1;
    op_dispatch_ind(&exec,&indirect.payload);
    assert(failed && !exec.recorded && !physical_adoptions && !native_provider_uses);
    vk_exec_dispatch_enter(prior);
    vk_exec_dispatch_cancel(&retry);
    vk_mem_span_capture_release(&retained);
    assert(!references&&capture_releases==1);
}
int main(void) {
    check_retained();
    check_retained_storage_emission();
    check_invalid_storage_cannot_seed();
    check_incompatible_registration();
    check_descriptor_refusal();
    check_cached_and_compat();
    check_compat_sample_handoff();
    check_compat_storage_handoff();
    check_compat_raw_selection();
    check_native_scratch_handoff();
    check_stop();
    check_terminal();
    check_noop();
    check_standin_refusal();
    check_indirect_retry_and_terminal();
    check_pool_prefix_and_mapping();
    check_retained_actual_mirror();
    check_uniform_prefix_before_dispatch_output();
    return 0;
}
'''


def main():
    source = REPLAY.read_text(encoding="utf-8")
    draw = (SOURCES / "vk_exec_draw.c").read_text(encoding="utf-8")
    sync = (SOURCES / "vk_sync.c").read_text(encoding="utf-8")
    header = (SOURCES / "vk_exec.h").read_text(encoding="utf-8")
    vk_header = (SOURCES / "vk.h").read_text(encoding="utf-8")
    compiler_types = between(vk_header, "typedef enum {\n    VK_COMPUTE_READY", "// Probe only with allow_create=0")
    dispatch_types = between(header, "typedef struct VkExecDispatchImages", "// COMPAT is selected")
    dispatch_types = "\n".join((
        "typedef struct VkGraphicsPipelineRevision VkGraphicsPipelineRevision;",
        "typedef struct {int unused;} VkTextureSampleRefusal, VkTextureAliasFailure;",
        dispatch_types,
        "static void selected_sampled_failure_log(const VkExecSampledFailure *failure, uint32_t opcode, const char *path) "
        "{ (void)failure; (void)opcode; (void)path; }",
    ))
    # Focus this walker on ordered preparation and issue. Storage, sample provider selection,
    # and scratch allocation have separate fixtures with the corresponding physical owners.
    draw_stubs = r"""
typedef struct { VkMemSpan span; VkMemSpanCapture *native; } RawStorageInput;
static int prepared_buffer_requires_image(const VkMemSpan *span,uint64_t address,uint64_t bytes,RcRegistryPosition read) {
    (void)span;(void)address;(void)bytes;(void)read;return 0;
}
static int dispatch_indirect_requires_image(const void *payload,RcRegistryPosition read) {
    (void)payload;(void)read;return 0;
}
static const RawStorageInput *prepared_buffer_argument(const DrawDescriptorGrant *grant,uint64_t address,uint64_t bytes) {
    (void)grant;(void)address;(void)bytes;return NULL;
}
static int prepared_buffer_argument_record(VkExec *ex,const DrawDescriptorGrant *grant,const RawStorageInput *input,
    VkMemSpan *argument) {
    (void)ex;(void)grant;(void)input;(void)argument;abort();
}
static unsigned retained_storage,storage_texture_locked,storage_begun,storage_emitted,storage_written;
static unsigned storage_access_invalid,storage_access_checks,storage_seed_requests;
static int dispatch_storage_outputs_present(void) {
    if (compat_storage) assert(effect_current == &compat_effect && effect_current->images==&compatibility_image &&
                               effect_current->image_count==1 && storage_target);
    return retained_storage;
}
static int dispatch_storage_outputs_begin(VkExec *ex) {
    assert(ex == active && storage_texture_locked && !storage_begun && !storage_written &&
           events[nevents - 1] == E_RESOURCES);
    storage_begun++;
    return 1;
}
static void dispatch_storage_outputs_written(VkExec *ex) {
    assert(ex == active && storage_texture_locked && storage_begun && storage_emitted && !storage_written &&
           events[nevents - 1] == E_DISPATCH);
    storage_written++;
}
static int readonly_storage_pointer(const GfxShaderRes *resource) {
    return resource->ptr && resource->count<=1 && !(resource->flags&1);
}
static const VkPipeBinding *vk_exec_dispatch_compat_storage_binding(const VkExecDispatch *dispatch) {
    static const VkPipeBinding binding={.kind=GFX_RES_IMG,.type=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,.count=1};
    assert(dispatch->info==&compute_info && dispatch->pipeline==compute_pipeline);
    return dispatch->compatibility && compat_storage ? &binding : NULL;
}
static VkExecGraphicsResult dispatch_compat_storage_prepare(VkExec *ex,VkExecDispatch *dispatch) {
    if (!compat_storage) return VK_EXEC_GRAPHICS_READY;
    assert(ex==&exec && dispatch->compatibility && dispatch->info==&compute_info && ex->img[NVN_STAGE_CS][2]==31 &&
           !rc_registry_position_compare(dispatch->read_position,vk_exec_resource_position()) &&
           drains==1 && ins==1 && !outs && !declarations && !dispatch->selection->count);
    assert(!compat_storage_prepares && !storage_target);
    assert(!compat_samples || (dispatch->selection->samples==&selected_samples && selected_samples.ready));
    compat_storage_prepares++;
    storage_target=1;
    compatibility_image=(EffectImageCapture){.independent=&storage_target,.uploaded=!compat_storage_refused};
    dispatch->selection->images=&compatibility_image;
    dispatch->selection->count=1;
    return compat_storage_refused ? VK_EXEC_GRAPHICS_UNAVAILABLE : VK_EXEC_GRAPHICS_READY;
}
static int vk_texture_storage_target_current(const int *target) {
    assert(compat_storage && target==&storage_target);
    return *target;
}
static int dispatch_storage_seed_current(const EffectImageCapture *image,int uploaded) {
    assert(image==&compatibility_image && uploaded && storage_target);
    return image->uploaded;
}
static int dispatch_compat_storage_current(const VkExecDispatchImages *selection) {
    return !selection->count || (vk_texture_storage_target_current(selection->images[0].independent) &&
        dispatch_storage_seed_current(&selection->images[0],1));
}

static void dispatch_compat_storage_release(VkExecDispatchImages *selection) {
    if (!selection->count) return;
    assert(selection->count==1 && selection->images==&compatibility_image && storage_target);
    selection->count=0;
    selection->images=NULL;
    compat_storage_releases++;
    storage_target=0;
}
static int dispatch_storage_outputs_disjoint(const VkExec *ex, const VkExecDispatch *dispatch) {
    assert(ex == &exec && dispatch->selection);
    return 1;
}
static VkExecGraphicsResult dispatch_native_storage_peers_prepare(VkExec *ex, VkExecDispatch *dispatch) {
    assert(ex == &exec && dispatch->selection);
    return VK_EXEC_GRAPHICS_READY;
}
static VkExecGraphicsResult dispatch_native_storage_volume_prepare(VkExec *ex, VkExecDispatch *dispatch) {
    assert(ex == &exec && dispatch->selection);
    return VK_EXEC_GRAPHICS_READY;
}
static VkExecGraphicsResult dispatch_storage_overwrite_select(VkExecDispatch *dispatch) {
    assert(dispatch->selection);
    storage_access_checks += retained_storage;
    return storage_access_invalid ? VK_EXEC_GRAPHICS_UNAVAILABLE : VK_EXEC_GRAPHICS_READY;
}
static VkExecGraphicsResult dispatch_storage_seeds_prepare(VkExec *ex, VkExecDispatch *dispatch) {
    assert(ex == active && dispatch->selection && !storage_begun && !storage_access_invalid);
    storage_seed_requests += retained_storage;
    return VK_EXEC_GRAPHICS_READY;
}
static void vk_texture_lock(void) { assert(retained_storage && !storage_texture_locked); storage_texture_locked = 1; }
static void vk_texture_unlock(void) { assert(storage_texture_locked); storage_texture_locked = 0; }
static void resource_effects_emitted(void) {
    assert(storage_texture_locked && storage_begun && !storage_emitted && events[nevents - 1] == E_DISPATCH);
    storage_emitted++;
}
static int dispatch_first_write_begin(VkExec *ex, uint32_t opcode) {
    assert(ex == &exec && opcode == selected_opcode);
    return 1;
}
static void dispatch_first_write_recorded(void) {}
static VkExecGraphicsResult dispatch_first_write_prepare(VkExec *ex, const VkTextureWritePlan *plan) {
    assert(ex == &exec && !plan->count);
    return VK_EXEC_GRAPHICS_READY;
}
static VkExecGraphicsResult dispatch_device_pointers_prepare(VkExec *ex, VkExecDispatch *dispatch) {
    assert(ex == &exec && !dispatch->selection->pointer_count);
    return VK_EXEC_GRAPHICS_READY;
}
"""
    draw_stubs += r"""
typedef struct {uint64_t address, bytes, alignment;} UniformAddressRange;
typedef struct {uint64_t address;} p_bind_buf;
#define GFX_STAGE_CS 5u
static struct {uint64_t ubo_max;} LIM;
static void limits_init(void) { LIM.ubo_max = 65536; }
static int uniform_binding_range(const VkExec *ex, const VkPipeBinding *binding, unsigned stage,
                                 unsigned member, UniformAddressRange *out) {
    (void)ex; (void)binding; (void)stage; (void)member; (void)out;
    abort();
}
static int readonly_storage_uniform_range(const VkExec *ex, unsigned stage, const GfxShaderRes *resource,
                                          UniformAddressRange *out) {
    assert(ex==&exec && stage==NVN_STAGE_CS && compat_raw_storage && resource->slot==4 && out);
    return 0;
}
static __attribute__((unused)) const p_bind_buf *raw_storage_fixed(const VkExec *ex,const VkPipeBinding *binding) {
    static const p_bind_buf raw={.address=0x8000};
    assert(ex==&exec && binding->res->slot==4 && binding->kind==GFX_RES_SSBO && binding->count==1 &&
           binding->type==VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && binding->stage==GFX_STAGE_CS);
    return &raw;
}
static __attribute__((unused)) int raw_storage_requires_image(const p_bind_buf *buffer,RcRegistryPosition read) {
    assert(buffer->address==0x8000 && read.scene==SCENE && read.ordinal==2*ORDINAL);
    return compat_image_bytes;
}
static int uniform_range_requires_order(const UniformAddressRange *source,RcRegistryPosition read) {
    (void)source;(void)read;abort();
}
"""
    draw_handlers = draw_stubs + "\n".join(function(draw, name) for name in (
        "graphics_argument_record", "dispatch_log_compute", "dispatch_materialize_ubo_merges",
        "dispatch_bind_selected_resources", "dispatch_direct_emit", "dispatch_indirect_emit",
        "vk_exec_dispatch_enter", "dispatch_inputs_match", "vk_exec_dispatch_cancel",
        "dispatch_device_pointers_current", "dispatch_read_current",
        "vk_exec_dispatch_has_compat_inputs", "dispatch_compat_read_current", "dispatch_compat_inputs_transfer",
        "dispatch_record_input_stage",
        "vk_exec_dispatch_prepare_compat_inputs", "vk_exec_dispatch_prepare_native_inputs", "vk_exec_dispatch_validate",
        "dispatch_indirect_span", "dispatch_admit_held_argument", "vk_exec_dispatch_select",
        "vk_exec_dispatch_compat_inputs_select", "dispatch_compat_pipeline", "dispatch_storage_standins",
        "op_dispatch", "op_dispatch_ind"))
    dispatch_work = between(source, "typedef struct {\n    VkExecDispatch *dispatch;", "} OrderedDispatchWork;")
    helpers = dispatch_work + "} OrderedDispatchWork;\n" + "\n".join(function(source, name) for name in
        ("prepare_dispatch_parked", "prepare_dispatch_uniform_gpa", "prepare_dispatch_inputs", "prepare_ordered_dispatch"))
    helpers = helpers.replace("&recording_failed", "&failed")
    sync_helpers = "\n".join(function(sync, name) for name in
                            ("recorder_park", "recorder_resume", "progress_stopped", "vk_sync_progress_prepare_pipeline"))
    authority = "\n".join(function(source, name) for name in
                          ("vk_exec_resource_position", "resource_seal", "resource_assign", "resource_declare_noop",
                           "resource_declare"))
    walk = source[source.index("static const op_hdr *walk(", source.index(dispatch_work)):]
    prepare = between(walk, "        VkExecDispatch dispatch", "        if (step && step->opcode == NVN_EXEC_STATE")
    prepare = prepare.replace("&recording_failed", "&failed")
    effect = between(walk, "            void *effect;", "            RcRegistryPosition read = vk_exec_resource_position()")
    noop = "            if" + between(walk, " (exact_dispatch == VK_EXEC_DISPATCH_NOOP) {", "            } else if (is_buffer_transfer) {")
    finish = between(walk, "            vk_exec_effects_recorded(effect,0);", "            ex->resource_op = 0;")
    finish = finish.replace("            vk_exec_graphics_draw_enter(prior_graphics);\n", "")
    invoke = r'''
static const op_hdr *run_selected_impl(VkExec *ex,const op_hdr *h,const NvnExecStep *step) {
    const void *a=nvn_op_payload(h);
    VkExecClearTexture clear={0};
    int exact_clear=0;
    VkExecBufferToTexture b2t={0};
    int exact_b2t=0;
    VkExecTextureToTexture t2t={0};
    int exact_t2t=0;
    const int is_buffer_transfer=0, projected_transfer=0;
    VkExecBufferTransfer *buffer_transfer=NULL;
    VkExecProjectedTransfer projected={0};
    VkExecRestoreCopy restored={0};
    enum { OP_COPY_T2B = 100 };
    const int graphics_noop=0;
''' + prepare + effect + noop + "} else emit_dispatch(ex,h);\n" + finish + r'''
    ex->resource_op = 0;
    return NULL;
}
static int run_selected(VkExec *ex,const op_hdr *h,const NvnExecStep *step) {
    return run_selected_impl(ex,h,step)!=&walk_stopped;
}
'''
    cc = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="ordered-dispatch-walker-") as tmp:
        c = Path(tmp) / "check.c"
        exe = c.with_suffix(".exe")
        c.write_text("\n".join((DRAW_STUBS, dispatch_types, compiler_types, PREFIX, draw_handlers,
                                authority, sync_helpers, helpers, invoke, TESTS)), encoding="utf-8")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", "-I" + str(ROOT / "runtime"),
                        str(c), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True, timeout=20)
    print("Ordered dispatch: native and compatibility input custody, recorder park, one emission and cancellation passed")


if __name__ == "__main__":
    main()

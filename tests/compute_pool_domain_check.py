"""Exercise verified compute shader domains through finite table selection and descriptor binding."""

import os
import re
from pathlib import Path
import subprocess
import tempfile

from shader_pool_domain_check import FIXTURE as DOMAIN_FIXTURE

ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    declaration = re.search(r"^(?:static )?[A-Za-z_][\w \t*]*\b" + name + r"\([^;{}]*\)\s*\{", source, re.M)
    if not declaration:
        raise ValueError(f"Missing production definition {name}")
    end = declaration.end()
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[declaration.start():end] + "\n"

# Keep the production parser and existing ordered, read-only SSBO handle expression.
# Exercise the ordinary sampler lane from a compute module with explicit LOD.
SHADER = DOMAIN_FIXTURE[:DOMAIN_FIXTURE.index("static uint32_t *instruction(")]
SHADER = SHADER.replace("COORD_TYPE, COORD, COMPARED", "COORD_TYPE, COORD, COMPARED, COLOR_TYPE")
SHADER = SHADER.replace("0, 64, 0,", "0, 65, 0,")
SHADER = SHADER.replace("I(15, 4, ENTRY, 0x6e69616d, 0),",
                        "I(15, 5, ENTRY, 0x6e69616d, 0), I(16, ENTRY, 17, 1, 1, 1),")
SHADER = SHADER.replace("I(25, IMAGE, F32, 1, 1, 0, 0, 1, 0)", "I(25, IMAGE, F32, 1, 0, 0, 0, 1, 0)")
SHADER = SHADER.replace("I(89, F32, COMPARED, COMBINED, COORD, FLOAT_ZERO)",
                        "I(88, COLOR_TYPE, COMPARED, COMBINED, COORD, 2, FLOAT_ZERO)")
SHADER = SHADER.replace("I(23, COORD_TYPE, F32, 2),", "I(23, COORD_TYPE, F32, 2), I(23, COLOR_TYPE, F32, 4),")
SHADER = SHADER.replace(".type=0x403", ".type=3")
SHADER = SHADER[:SHADER.index("static int reader_status")]
SHADER = SHADER.replace("COMPARED, COLOR_TYPE", "COMPARED, COLOR_TYPE, C0_COUNT, OUT_BLOCK, OUT_BLOCK_PTR, OUT_PTR, OUT, OUT_ELEMENT")
SHADER = SHADER.replace("0, 65, 0,", "0, 71, 0,")
SHADER = SHADER.replace("I(71, UNUSED, 0),",
                        "I(71, UNUSED, 0), I(71, OUT, 34, 0), I(71, OUT, 33, 12), I(71, OUT_BLOCK, 2), I(72, OUT_BLOCK, 0, 35, 0),")
SHADER = SHADER.replace("I(43, U32, MAXIMUM, 7)", "I(43, U32, MAXIMUM, 32767)")
SHADER = SHADER.replace("I(43, U32, COUNT, 8)", "I(43, U32, COUNT, 32768), I(43, U32, C0_COUNT, 8)")
SHADER = SHADER.replace("I(28, C0_ROWS, U32, COUNT)", "I(28, C0_ROWS, U32, C0_COUNT)")
SHADER = SHADER.replace("I(33, FUNCTION_TYPE, VOID)", r"""
    I(30, OUT_BLOCK, COLOR_TYPE), I(32, OUT_BLOCK_PTR, 12, OUT_BLOCK), I(32, OUT_PTR, 12, COLOR_TYPE),
    I(59, OUT_BLOCK_PTR, OUT, 12), I(33, FUNCTION_TYPE, VOID)""")
SHADER = SHADER.replace("1u << 16 | 253, 1u << 16 | 56,",
                        "I(65, OUT_PTR, OUT_ELEMENT, OUT, ZERO), I(62, OUT_ELEMENT, COMPARED), 1u << 16 | 253, 1u << 16 | 56,")
SHADER = SHADER.replace("static GfxShaderRes buffer = {.binding=7, .count=1, .slot=1};",
                        "static GfxShaderRes buffers[] = {{.binding=7,.count=1,.slot=1},{.binding=12,.count=1,.flags=1}};")
SHADER = SHADER.replace("&c0, &buffer, textures", "&c0, buffers, textures")
SHADER = SHADER.replace(".count=8, .type=3", ".count=32768, .type=3")
SHADER = SHADER.replace("const int counts[] = {1, 1, 2, 0}", "const int counts[] = {1, 2, 2, 0}")
SHADER = SHADER.replace("{0x40100004, 9, 1, 1}", "{0x40100004, 32772, 1, 1}")


QUERY_SHADER = SHADER[SHADER.index("static const uint32_t module[]"):
                      SHADER.index("static GfxShaderRes c0")]
QUERY_SHADER = QUERY_SHADER.replace("uint32_t module[]", "uint32_t query_module[]")
QUERY_SHADER = QUERY_SHADER.replace("0, 71, 0,", "0, 72, 0,")
QUERY_SHADER = QUERY_SHADER.replace("I(23, COLOR_TYPE, F32, 4)", "I(23, COLOR_TYPE, U32, 2)")
QUERY_SHADER = QUERY_SHADER.replace("I(88, COLOR_TYPE, COMPARED, COMBINED, COORD, 2, FLOAT_ZERO)",
                                    "I(100, IMAGE, 71, COMBINED), I(103, COLOR_TYPE, COMPARED, 71, ZERO)")

MODEL = r'''
#include <stdlib.h>
#include <string.h>

#include "rc_sampled_image_query.h"

enum { NVN_STAGE_CS=5, VK_POOL_TEX=32768, VK_POOL_KINDS=16, VK_POOL_SAMP=4096, VK_POOL_MARK_BINDING=17,
       VK_EXEC_POOL_MEMBER_LIMIT=144, VK_MEM_MIRROR_EXISTING,
       VK_IMAGE_ASPECT_COLOR_BIT=1, VK_IMAGE_LAYOUT_GENERAL=1, VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET=10,
       VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE=4, VK_DESCRIPTOR_TYPE_SAMPLER=5 };
#define VK_NULL_HANDLE UINT64_C(0)
#define VK_WHOLE_SIZE UINT64_MAX
#define NVN_HANDLE_TEX(h) ((int)((h)&0xfffff))
enum { VK_NOT_READY=1, VK_ERROR_DEVICE_LOST=-4, VK_ERROR_FEATURE_NOT_PRESENT=-8,
       RC_BUDGET_RAM=0, RC_BUDGET_COMMIT=1, RC_ADMISSION_OK=0, VK_TEXTURE_REGISTRATION_ABSENT=1,
       VK_TEXTURE_REGISTRATION_CURRENT=2 };
typedef uint64_t VkDescriptorPool, VkDescriptorSet, VkDevice, VkImage, VkImageView, VkBuffer, VkSampler;
typedef uint64_t VkCommandBuffer, VkDescriptorSetLayout;
typedef int VkResult;
typedef struct {int refs;} VkTextureRegistrationScope;
typedef struct {VkDevice device;} VkCtx;
typedef struct {int unused;} PlatMutex;
typedef struct {uint64_t amount[2];} RcAdmissionVector;
typedef struct {VkSampler sampler; VkImageView imageView; int imageLayout;} VkDescriptorImageInfo;
enum {
       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER=2, VK_SUCCESS=0, RC_BUDGET_BUFFER=1,
       VK_PIPELINE_BIND_POINT_GRAPHICS=1, VK_PIPELINE_BIND_POINT_COMPUTE=2 };
typedef struct {uint64_t scene,ordinal;} RcRegistryPosition;
typedef struct {uint64_t buffer,offset,range;} VkDescriptorBufferInfo;
typedef struct {int sType; VkDescriptorSet dstSet; uint32_t dstBinding,dstArrayElement,descriptorCount;
    int descriptorType; const VkDescriptorImageInfo *pImageInfo; const VkDescriptorBufferInfo *pBufferInfo;} VkWriteDescriptorSet;
typedef struct {uint64_t image;} VkTexImage;
typedef struct {int retired,slot,recorded;} VkTextureQueryImage;
typedef struct {struct {uint32_t aspectMask;} planes;} VkTextureSampleGeometry;
typedef struct {int historical; VkTextureQueryImage *query_image;} VkTextureSample;
typedef struct {const GfxShader *shader;const GfxShaderRes *resource;uint64_t handle;
    VkTextureSample sample;VkTexImage image;VkTextureSampleGeometry geometry;int seen;} GraphicsSample;
typedef struct {int reason;} VkExecSampledFailure;
typedef struct {int reason;} VkExecPreparationFailure;
typedef struct {uint64_t buffer;} VkMemSpan;
typedef struct {int id;} VkMemSpanCapture;
typedef struct {const uint8_t *data;} PoolIndexReadback;
typedef struct {int id; uint64_t buffer;} PoolIndexSnapshot;
typedef struct {int owned;} RcBudgetTicket;
typedef struct {int unused;} VkExecGpaSampleScope;
typedef struct {int mo_ovr_on,sf_mvprev,pb_spoiled,flat3d_pool_ready; VkDescriptorBufferInfo storage_buffers[2];} DrawCtx;
typedef struct VkExec {DrawCtx *dc; uint64_t fence, tex_pool; const dpool_t *samp_pool; VkCommandBuffer cb;
    struct {unsigned width;} tonemap_extent;} VkExec;
typedef struct {
    int kind, stage, count, type, set, binding;
    const GfxShaderRes *res;
} VkPipeBinding;
typedef struct {
    int bind_point;
    unsigned n_bindings;
    const VkPipeBinding *bindings;
    const GfxShader *compute_pool_domain_shader;
    int pool,compute_pool_marks;
    uint32_t compute_pool_query_shape_kinds;
} VkPipeInfo;
typedef enum {VK_EXEC_GRAPHICS_READY,VK_EXEC_GRAPHICS_UNAVAILABLE,VK_EXEC_GRAPHICS_INVALIDATED,
              VK_EXEC_GRAPHICS_STOPPED,VK_EXEC_GRAPHICS_NEEDS_ORDERED} VkExecGraphicsResult;
typedef enum {POOL_INDEX_C0, POOL_INDEX_UNIFORM, POOL_INDEX_INLINE, POOL_INDEX_STORAGE} PoolIndexInputKind;
typedef struct PoolIndexInput {
    const GfxShaderRes *resource;
    struct PoolIndexInput *source;
    VkDescriptorBufferInfo descriptor;
    VkMemSpan span;
    VkMemSpanCapture *native;
    RcInterval *ranges;
    uint32_t range_count;
    PoolIndexSnapshot *snapshot;
    PoolIndexReadback *readback;
    const uint8_t *bytes;
    unsigned word_count;
    p_bind_buf bound;
    int stage,slot,recorded,recorded_copy,whole_range,gpu_words_ready,guard_original,scalar_original;
    PoolIndexInputKind kind;
    int immutable_import,import_required;
} PoolIndexInput;
typedef struct PoolIndexInputs {PoolIndexInput values[1]; unsigned count; int finite;} PoolIndexInputs;
typedef struct ComputePoolIndexInputs {
    RcBudgetTicket budget,ranges_budget;
    RcInterval *ranges;
    PoolIndexInputs inputs;
} ComputePoolIndexInputs;
typedef struct {
    struct {uint32_t slot,kinds;} entries[VK_EXEC_POOL_MEMBER_LIMIT];
    uint32_t count;
    uint8_t domain,comparison_absent;
} VkExecPoolIndices;
typedef struct {
    int finite,held;
    uint32_t index,kinds;
    RcRegistryPosition read;
    ComputePoolIndexInputs *compute_inputs;
    const VkPipeInfo *compute_info;
    VkExecPoolIndices indices;
    VkDescriptorSet set;
    VkTextureRegistrationScope *scope_lease;
    uint64_t scene,pool,publication;
    struct ComputePoolQuerySet *compute_query;
} VkExecPoolGrant;
typedef struct {
    VkMemSpan span;
    VkMemSpanCapture *native;
    RcInterval *mapped;
    unsigned mapped_count;
    int recorded,written;
    const GfxShaderRes *resource;
    VkDescriptorBufferInfo descriptor;
} EffectStorageCapture;
typedef struct {EffectStorageCapture storage; int cold_writer;} SelectedStorageWriter;
typedef struct {SelectedStorageWriter writer;} SelectedStorageWriters;
typedef struct {SelectedStorageWriters *selected_writers; int opaque_storage_effects;} ComputeSelection;
typedef struct {SelectedStorageWriters *selected_writers;} EffectDeclaration;
typedef struct {int native_bindings; EffectStorageCapture output;} EffectInputCapture;
typedef struct {
    const GfxShader *shader;
    const VkPipeInfo *info;
    RcRegistryPosition read_position;
    VkExecPoolGrant *pool_grant;
    const ComputeSelection *selection;
    struct {VkExecSampledFailure sampled;} input_failure;
} VkExecDispatch;
typedef struct {PoolIndexInputs *pool_index_inputs; RcRegistryPosition read_position; unsigned opcode;} VkExecGraphicsDraw;
typedef struct {const VkPipeInfo *info; VkExecPoolGrant *grant; unsigned opcode;} VkExecPoolBindScope;
struct GfxShader {ShaderPoolIndices *plan; unsigned writes;};
static VkExecPoolBindScope pool_binding;
static VkExecGraphicsDraw *graphics_current;
static RcRegistryPosition position={7,8};
static struct {int partially_bound,marks_ready,marks_owner_valid,marks_mixed,split_samplers;VkDescriptorSetLayout layout;
    uint64_t marks_owner; VkBuffer marks; uint32_t set_registration_gone[1];
    int set_comparison_bound[1],sealed,registration_bound,set_prefix_complete[1];
    void *publication[1];} PL={.partially_bound=1,.layout=10};
#define POOL_IMAGE_ENTRIES (VK_POOL_KINDS * VK_POOL_TEX)
static struct {VkSampler samp;} Z={.samp=99};
static int vk_dlaa_swap_img;
static uint64_t pool_finite_native_bindings;
static PlatMutex pool_mt;
static VkTextureRegistrationScope scope={1};
static dpool_t sampler_pool;
static VkCtx ctx={.device=17};
static unsigned sampler_generation=3;
static unsigned shape_captures,carrier_allocations,carrier_frees,carrier_records,private_allocs,private_frees;
static unsigned private_charges,sampler_writes,marker_writes,image_writes,canonical_updates;
static int texture_lock, pool_lock, absent_zero=1, missing_slot=-1, stale_slot=-1, alloc_refused;
static VkDescriptorSet private_set;
static void (*native_release)(void *);
static void (*native_cancel)(void *,int);
static void *native_owned;
static VkImageView bound_views[VK_POOL_TEX];
static struct {uint64_t storage_write_alignment,sbuf_align;} vk_caps={4,4};
static const p_bind_buf input_binding={.addr=64,.size=sizeof handles}, output_binding={.addr=256,.size=16};
static VkMemSpanCapture input_native={5}, replica_original={6}, output_native={7};
static PoolIndexSnapshot replica={.id=1,.buffer=191};
static unsigned queries, captures, refills, whole_calls, snapshot_drops, registered, copied, reports, override_guard;
static unsigned metadata_count, metadata_fail;
static RcInterval actual_mapping={.range={11,64,16}};
static EffectDeclaration effect;
static EffectDeclaration *effect_current=&effect;
static EffectInputCapture bound_effects={
    .native_bindings=1,.output={.mapped=&actual_mapping,.mapped_count=1,.recorded=1,.written=1,
       .resource=&buffers[1],.descriptor={73,0,16}}
};
static EffectInputCapture *effect_scope_capture(VkExec *ex,int stage) {
    assert(ex && stage==NVN_STAGE_CS);return &bound_effects;
}
static EffectStorageCapture *effect_storage_find(EffectInputCapture *capture,unsigned set,unsigned binding) {
    assert(capture==&bound_effects && !set && binding==12);return &capture->output;
}
static int vk_replay_metadata_allocate(size_t size,int clear,int producer,RcBudgetTicket *ticket,void **out) {
    assert(size && producer==RC_BUDGET_BUFFER && !ticket->owned && !*out);
    if (metadata_fail && !--metadata_fail) return 1;
    *out=clear?calloc(1,size):malloc(size);
    assert(*out);
    ticket->owned=1;
    metadata_count++;
    return VK_SUCCESS;
}
static void vk_replay_metadata_free(void *data,RcBudgetTicket *ticket) {
    assert(!!data==ticket->owned);
    if (!data) return;
    assert(metadata_count--);
    ticket->owned=0;
    free(data);
}
static int domain_live=1, overlap, cpu_visible;
static int whole_result=VK_EXEC_GRAPHICS_UNAVAILABLE, finite_result=VK_EXEC_GRAPHICS_READY;
static uint32_t kinds=1u<<2;
static int rc_registry_position_compare(RcRegistryPosition a, RcRegistryPosition b) {
    return a.scene!=b.scene || a.ordinal!=b.ordinal;
}
static RcRegistryPosition vk_exec_resource_position(void) {return position;}
static void *rc_registry_runtime(void) {return NULL;}
static int rc_registry_provisional_read_current(void *r,RcRegistryPosition read) {
    assert(!r);return !rc_registry_position_compare(position,read);
}
static int rc_registry_position_open(void *r,RcRegistryPosition read) {
    return rc_registry_provisional_read_current(r,read);
}
static int vk_sync_stopping(void) {return 0;}
static int vk_device_is_lost(void) {return 0;}
static int vk_exec_recording_failed(void) {return 0;}
static uint32_t vk_exec_pool_shader_kinds(const GfxShader *s) {assert(s);return kinds;}
uint32_t gfx_shader_guest_writes(const GfxShader *s) {return s->writes;}
int gfx_shader_has_pool_index_domain(const GfxShader *s) {return s->plan && s->plan->domain_source;}
int gfx_shader_pool_domain_indices(const GfxShader *s,GfxShaderPoolReadDomain read,GfxShaderPoolReadWord word,void *ctx,
    GfxShaderPoolIndex *out,uint32_t capacity,uint32_t *count,int *comparison) {
    assert(!word && !comparison);
    queries++;
    return shader_pool_domain_eval(s->plan,read,word,ctx,out,capacity,count,comparison);
}
static uint16_t pool_kind_type(int kind) {assert(kind==2);return 3;}
int gfx_shader_resources(const GfxShader *shader,int kind,const GfxShaderRes **out) {
    assert(shader && kind==GFX_RES_TEX);*out=textures;return 2;
}
static const p_bind_buf *effect_storage_fixed(const VkExec *ex,int stage,const GfxShaderRes *r) {
    assert(ex && stage==NVN_STAGE_CS);
    return r==&buffers[0] ? &input_binding : r==&buffers[1] ? &output_binding : NULL;
}
static SelectedStorageWriter *selected_storage_writer(SelectedStorageWriters *writers,int stage,const GfxShaderRes *r) {
    assert(stage==NVN_STAGE_CS && r==&buffers[1]);
    return writers ? &writers->writer : NULL;
}
static int effect_storage_capture_span(const GfxShaderRes *r,uint64_t addr,uint64_t size,int policy,EffectStorageCapture *out) {
    assert(r==&buffers[1] && addr==256 && size==16 && policy==VK_MEM_MIRROR_EXISTING);
    out->span.buffer=73;out->native=&output_native;
    out->mapped_count=1;out->mapped=malloc(sizeof *out->mapped);
    *out->mapped=(RcInterval){.range={11,overlap?8:64,16}};
    return 1;
}

typedef struct { uint64_t allocation,incarnation,offset; } RcRegistryBufferLocation;
typedef int VkExecHostInputResult;
enum { VK_EXEC_HOST_INPUT_READY=0 };
static int rc_registry_native_image_buffer_requires_coverage(void *r,const RcInterval *range,unsigned n,RcRegistryPosition read) { (void)r;(void)range;(void)n;(void)read;abort(); }
static int nvn_gpu_only(uint64_t address) { (void)address;return 0; }
static int resource_effects_image_trace_enabled(void) { return 0; }
static void pool_index_import_report(RcRegistryPosition read,const PoolIndexInput *input,uint64_t address,
    RcRegistryBufferLocation native,const char *phase,int required,VkResult placement,VkExecHostInputResult host,const char *refusal) {
    (void)read;(void)input;(void)address;(void)native;(void)phase;(void)required;(void)placement;(void)host;(void)refusal;abort();
}
static int vk_mem_span_buffer_location(const VkMemSpanCapture *capture,const VkMemSpan *span,
        RcRegistryBufferLocation *out) { (void)capture;(void)span;(void)out;abort(); }
static int vk_mem_span_capture_current(const VkMemSpanCapture *capture,const VkMemSpan *span) {
    return capture && span->buffer && (capture!=&input_native || domain_live);
}
static void vk_mem_span_capture_release(VkMemSpanCapture **capture) {*capture=NULL;}
static int pool_index_storage_capture(VkExec *ex,RcRegistryPosition read,PoolIndexInputs *inputs,PoolIndexInput *input,int required) {
    assert(!required);
    assert(ex && !inputs->count && input->resource==&buffers[0] && !rc_registry_position_compare(read,position));
    captures++;
    input->span.buffer=71;input->native=&input_native;
    input->descriptor.range=sizeof handles;
    input->kind=POOL_INDEX_STORAGE;
    input->range_count=1;input->ranges=malloc(sizeof *input->ranges);
    *input->ranges=(RcInterval){.range={11,0,sizeof handles}};
    if(cpu_visible) {
        input->bytes=(const uint8_t *)handles;
        input->snapshot=&replica;
        input->descriptor.buffer=191;
    }
    return 1;
}
static int pool_index_input_current(const VkExec *ex,RcRegistryPosition read,const PoolIndexInput *input) {
    return ex && !rc_registry_position_compare(read,position) && vk_mem_span_capture_current(input->native,&input->span);
}
static int pool_index_inputs_resolve(VkExec *ex,RcRegistryPosition read,PoolIndexInputs *inputs,VkExecGraphicsDraw *draw) {
    assert(ex && !rc_registry_position_compare(read,position) && !draw && inputs->count==1);
    PoolIndexInput *input=inputs->values;
    assert(input->whole_range && input->word_count==1 && !input->bytes);
    copied++;
    input->bytes=(const uint8_t *)handles;
    input->snapshot=&replica;
    input->descriptor.buffer=191;
    input->gpu_words_ready=input->recorded_copy=1;
    return 1;
}
static int vk_exec_effects_read_allowed_range(RcSourceRange range,int private_sample) {
    assert(range.backing==11 && !private_sample);return !override_guard;
}
static int nvn_stage_of(int stage) {return stage;}
static void pool_index_snapshot_release(void *arg) {if(arg) {assert(arg==&replica);snapshot_drops++;}}
static void pool_index_readback_release(void *arg) {assert(!arg);}
static void pool_index_snapshot_record(uint64_t fence,void *arg) {
    assert(fence==7 && arg==&replica);registered++;
}
static void vk_exec_required_graphics_unavailable(VkExec *ex,unsigned opcode,const char *reason) {
    assert(ex && opcode==9 && reason);reports++;
}
static VkMemSpanCapture *vk_mem_span_capture_retain(VkMemSpanCapture *in) {assert(in==&replica_original);return in;}
static int vk_mem_span_record_input(VkMemSpanCapture *c,VkMemSpan *p) {return c && p;}
static void vk_exec_record_native_input_provider(VkExec *ex,VkMemSpanCapture **in) {assert(ex&&in);}
static int pool_index_inputs_current(const VkExec *ex,RcRegistryPosition read,const PoolIndexInputs *inputs);
static void compute_pool_index_inputs_release(ComputePoolIndexInputs *inputs);
static void plat_mutex_lock(PlatMutex *lock){assert(lock==&pool_mt && !pool_lock++);}
static void plat_mutex_unlock(PlatMutex *lock){assert(lock==&pool_mt && pool_lock--==1);}
static void vk_texture_lock(void){assert(!texture_lock++);}
static void vk_texture_unlock(void){assert(texture_lock--==1);}
static VkCtx *vk_ctx(void){return &ctx;}
static VkTextureRegistrationScope *vk_exec_texture_registration_scope(void){return &scope;}
static uint64_t vk_texture_registration_scope_publication(const VkTextureRegistrationScope *actual){
    assert(actual==&scope && scope.refs);return 19;
}
static VkTextureRegistrationScope *vk_texture_registration_scope_retain(VkTextureRegistrationScope *actual){
    assert(actual==&scope && scope.refs==1);scope.refs++;return actual;
}
static void vk_texture_registration_scope_end(VkTextureRegistrationScope *actual){
    if(actual){assert(actual==&scope && scope.refs==2);scope.refs--;}
}
static int graphics_pool_identity_locked(const VkExec *ex,const VkExecPoolGrant *grant){
    assert(ex && grant && pool_lock && !grant->compute_query);return !rc_registry_position_compare(grant->read,position);
}
static int graphics_pool_grant_read_current(const VkExec *ex,const VkExecPoolGrant *grant){
    assert(ex);
    assert(pool_lock && grant && !grant->compute_query);return !rc_registry_position_compare(grant->read,position);
}
static int graphics_pool_comparison_feedback(const VkExec *ex,const VkPipeInfo *info,const VkExecPoolGrant *grant) {
    (void)ex;(void)info;(void)grant;return 0;
}
static int graphics_pool_layout_matches(const VkExec *ex,const VkExecPoolGrant *grant){
    assert(ex && grant && pool_lock && !grant->compute_query);return 1;
}
static void graphics_pool_release_locked(VkExecPoolGrant *grant){assert(pool_lock && !grant->compute_query);grant->held=0;}
static void vk_texture_publication_use(void *publication){(void)publication;canonical_updates++;}
static void binding_manifest_cancel(VkExec *ex){assert(ex);}
static VkDescriptorSet pool_sync_draw(VkExec *ex){(void)ex;abort();}
static int vk_texture_deferred(void){return 0;}
static int flat3d_pool_consuming;
static uint32_t vk_sampler_gen(void){assert(texture_lock);return sampler_generation;}
static int vk_sampler_slot_count_locked(void){assert(texture_lock);return 3;}
static VkSampler vk_sampler_by_id_locked(int id){assert(texture_lock && id<3);return id==1 ? 0 : 100+id;}
static VkSampler vk_sampler_pool_by_id_locked(int id,int comparison){assert(!comparison);return vk_sampler_by_id_locked(id);}
static VkImageView pool_absent_view(uint32_t kind){assert(kind==2&&texture_lock);return 6000;}
static int vk_texture_registration_expected_locked(int slot,void *unused){
    assert(texture_lock && !unused);
    return !slot && absent_zero ? VK_TEXTURE_REGISTRATION_ABSENT : VK_TEXTURE_REGISTRATION_CURRENT;
}
static int sample_geometry_capture_locked(GraphicsSample *sample,const char **reason){
    assert(texture_lock && *reason==NULL && sample->resource==textures && sample->shader && sample->seen);
    shape_captures++;
    if((int)sample->handle==missing_slot)return 0;
    sample->geometry.planes.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;
    if(sample->handle==1)sample->image.image=501;
    return 1;
}
static VkResult vk_texture_query_image_create_locked(int slot,const VkTextureSample *sample,
    const VkTextureSampleGeometry *geometry,VkTextureQueryImage **out){
    assert(texture_lock && sample && geometry->planes.aspectMask==VK_IMAGE_ASPECT_COLOR_BIT && !*out);
    *out=calloc(1,sizeof **out);
    assert(*out);
    (*out)->slot=slot;
    carrier_allocations++;
    return VK_SUCCESS;
}
static void graphics_sample_release(GraphicsSample *sample){
    if(sample->sample.query_image){free(sample->sample.query_image);carrier_frees++;}
}
static int graphics_sample_shape_current(const GraphicsSample *sample){
    assert(texture_lock);
    if(sample->sample.historical==VK_TEXTURE_REGISTRATION_ABSENT)
        return vk_texture_registration_expected_locked((int)sample->handle,NULL)==VK_TEXTURE_REGISTRATION_ABSENT;
    return sample->sample.query_image && sample->sample.query_image->slot!=stale_slot;
}
static int sample_query_images_record(VkExec *ex,GraphicsSample *samples,uint32_t count){
    assert(ex->cb && !texture_lock && count==4);
    for(uint32_t i=0;i<count;i++) if(samples[i].sample.query_image){
        assert(!samples[i].sample.query_image->recorded);
        samples[i].sample.query_image->recorded=1;
        carrier_records++;
    }
    return 1;
}
static VkImageView vk_texture_query_image_view(const VkExec *ex,const VkTextureQueryImage *query,VkImage *image){
    assert(ex->cb && image && texture_lock && query->recorded);
    *image=600+query->slot;
    return 36000+query->slot;
}
static uint64_t descriptor_pool_opaque_bytes(uint64_t count){assert(count>VK_POOL_TEX);return 64*count;}
static int resource_core_budget_reserve(int producer,int required,const RcAdmissionVector *cost,RcBudgetTicket *ticket){
    assert(producer==RC_BUDGET_BUFFER && required==1 && cost->amount[0] && cost->amount[0]==cost->amount[1] && !ticket->owned);
    ticket->owned=1;private_charges++;return RC_ADMISSION_OK;
}
static int resource_core_budget_materialize(RcBudgetTicket *ticket,const RcAdmissionVector *cost){
    assert(ticket->owned==1 && cost->amount[0]);ticket->owned=2;return RC_ADMISSION_OK;
}
static int rc_admission_accounted(int result){return result==RC_ADMISSION_OK;}
static int resource_core_budget_cancel(RcBudgetTicket *ticket){
    assert(ticket->owned==1 && private_charges);ticket->owned=0;private_charges--;return RC_ADMISSION_OK;
}
static int resource_core_budget_retire(RcBudgetTicket *ticket){assert(ticket->owned==2);ticket->owned=3;return RC_ADMISSION_OK;}
static int resource_core_budget_destroyed(RcBudgetTicket *ticket){
    assert(ticket->owned==3 && private_charges);ticket->owned=0;private_charges--;return RC_ADMISSION_OK;
}
static VkResult pool_alloc_sets(VkCtx *c,VkDescriptorSetLayout layout,uint32_t count,int marks,
    VkDescriptorPool *pool,VkDescriptorSet *set){
    assert(c==&ctx && layout==PL.layout && count==1 && marks==PL.marks_ready);
    if(alloc_refused)return VK_NOT_READY;
    *pool=500+private_allocs;
    *set=1200+private_allocs++;
    private_set=*set;
    return VK_SUCCESS;
}
static void vkDestroyDescriptorPool(VkDevice device,VkDescriptorPool pool,void *allocator){
    assert(device==ctx.device && pool>=500 && !allocator);private_frees++;
}
static void vkUpdateDescriptorSets(VkDevice device,uint32_t count,const VkWriteDescriptorSet *writes,unsigned copied_count,
    const void *copies){
    assert(device==ctx.device && count && !copied_count && !copies && texture_lock);
    for(uint32_t i=0;i<count;i++){
        const VkWriteDescriptorSet *w=&writes[i];
        assert(w->dstSet==private_set && w->sType==VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
        if(w->dstBinding==VK_POOL_KINDS){
            assert(w->descriptorCount==VK_POOL_SAMP && w->descriptorType==VK_DESCRIPTOR_TYPE_SAMPLER);
            assert(w->pImageInfo[0].sampler==100 && w->pImageInfo[1].sampler==99 &&
                   w->pImageInfo[2].sampler==102 && w->pImageInfo[VK_POOL_SAMP-1].sampler==99);
            sampler_writes++;
        }else if(w->dstBinding==VK_POOL_MARK_BINDING){
            assert(w->descriptorCount==1 && w->descriptorType==VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && w->pBufferInfo->buffer==PL.marks);
            marker_writes++;
        }else{
            assert(w->dstBinding==2 && w->descriptorCount==1 && w->descriptorType==VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
            assert(w->pImageInfo->imageLayout==VK_IMAGE_LAYOUT_GENERAL);
            bound_views[w->dstArrayElement]=w->pImageInfo->imageView;
            image_writes++;
        }
    }
}
static void vk_completion_add_owned(uint64_t scene,void (*complete)(void *),void (*cancel)(void *,int),void *owner){
    assert(scene==7 && complete && cancel && owner && !native_owned);
    native_release=complete;native_cancel=cancel;native_owned=owner;
}
static void vk_device_lost_check(VkResult r,const char *site){(void)r;(void)site;abort();}
void rt_fatal(const char *site,...){(void)site;abort();}
static void compute_pool_record_refusal(VkExecDispatch *dispatch,const VkExecPreparationFailure *failure){
    assert(dispatch && failure);dispatch->input_failure.sampled.reason=failure->reason;
}
void vk_exec_pool_grant_release(VkExecPoolGrant **out);
static VkExecGraphicsResult pool_prepare_indices(VkExec *ex,uint32_t offered,VkExecPoolGrant **out,
    int ordered,int allow_prefix,RcRegistryPosition attempted,VkExecGpaSampleScope **pending,
    void *pass,const VkExecPoolIndices *indices,void *failure) {
    assert(ex && offered==kinds && ordered && allow_prefix && !attempted.scene && !pending && !pass && failure);
    assert(position.ordinal==8);
    if(indices) {
        assert(indices->domain && indices->count==4);
        assert(indices->entries[0].slot==4 && indices->entries[1].slot==32767 &&
               indices->entries[2].slot==1 && indices->entries[3].slot==0);
        for(unsigned i=0;i<indices->count;i++)assert(indices->entries[i].kinds==(1u<<2));
        refills++;
        if(finite_result!=VK_EXEC_GRAPHICS_READY)return finite_result;
    } else {
        whole_calls++;
        if(whole_result!=VK_EXEC_GRAPHICS_READY)return whole_result;
    }
    vk_exec_pool_grant_release(out);
    *out=calloc(1,sizeof **out);
    (*out)->read=position;
    (*out)->held=1;
    if(indices){(*out)->indices=*indices;(*out)->finite=1;}
    return VK_EXEC_GRAPHICS_READY;
}
'''

TEST = r'''
static void check_shader_store_refusals(void) {
    // Mutations retain the real index expression and change only the storage writer.
    uint32_t altered[sizeof module / sizeof module[0] + 4];
    memcpy(altered, module, sizeof module);
    uint32_t *store = NULL;
    for (size_t at = 5; at < sizeof module / sizeof module[0]; at += pi_len(altered + at))
        if (pi_op(altered + at) == PX_STORE && altered[at + 1] == OUT_ELEMENT) store = altered + at;
    assert(store);
    store[1] = RAW_PTR;
    store[2] = RAW_VALUE;
    assert(!shader_pool_indices_build(altered, sizeof module / sizeof module[0], resources, counts));
    buffers[0].flags = 1; // Making the handle table writable cannot permit a domain.
    assert(!shader_pool_indices_build(altered, sizeof module / sizeof module[0], resources, counts));
    buffers[0].flags = 0;

    memcpy(altered, module, sizeof module);
    const uint32_t pointer = altered[3]++;
    const size_t at = (size_t)(store - altered);
    memmove(store + 4, store, sizeof module - at * sizeof *altered);
    store[0] = 4u << 16 | PX_COPY;
    store[1] = OUT_PTR;
    store[2] = pointer;
    store[3] = OUT_ELEMENT;
    store[5] = pointer;
    assert(!shader_pool_indices_build(altered, sizeof altered / sizeof altered[0], resources, counts));

    buffers[1].ptr = 1; // An otherwise identified output cannot borrow a fixed writer receipt.
    assert(!shader_pool_indices_build(module, sizeof module / sizeof module[0], resources, counts));
    buffers[1].ptr = 0;
}

static void run_query_dispatch(VkExec *ex,VkPipeInfo *pipe,VkExecDispatch *dispatch) {
    // The fixture models the native pool declaration remap independently of the source domain analysis.
    uint32_t final[sizeof query_module/sizeof query_module[0]+8];
    memcpy(final,query_module,5*sizeof *final);
    uint32_t used=5;
    for(uint32_t at=5;at<sizeof query_module/sizeof query_module[0];){
        const uint32_t *word=query_module+at;
        uint32_t op=word[0]&0xffff,count=word[0]>>16;
        if(op==71 && count==4 && word[2]==33 && (word[1]==POOL || word[1]==SAMPLER_POOL)){
            final[used++]=4u<<16|71;
            final[used++]=word[1]; final[used++]=34; final[used++]=2;
            final[used++]=word[0]; final[used++]=word[1]; final[used++]=33;
            final[used++]=word[1]==POOL ? 2 : VK_POOL_KINDS;
        } else {memcpy(final+used,word,count*sizeof *final);used+=count;}
        at+=count;
    }
    GfxShader query={shader_pool_indices_build(query_module,sizeof query_module/sizeof query_module[0],resources,counts),
        RC_EFFECT_WRITES_BUFFER};
    assert(query.plan && query.plan->domain_source==&buffers[0]);
    uint32_t attested=compute_pool_query_shape_kinds(final,used);
    assert(attested==(1u<<2));
    dispatch->shader=pipe->compute_pool_domain_shader=&query;
    finite_result=VK_EXEC_GRAPHICS_UNAVAILABLE;
    cpu_visible=0;
    assert(whole_result==VK_EXEC_GRAPHICS_UNAVAILABLE);
    const unsigned prior_whole=whole_calls,prior_finite=refills,prior_copied=copied;
    assert(compute_pool_prepare(ex,dispatch,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!dispatch->pool_grant && !shape_captures && whole_calls==prior_whole+1 && refills==prior_finite+1);

    pipe->compute_pool_query_shape_kinds=attested;
    for(int marked=0;marked<2;marked++){
        cpu_visible=marked;
        pipe->compute_pool_marks=PL.marks_ready=marked;
        PL.marks=marked ? 908 : 0;
        const unsigned before_whole=whole_calls,before_finite=refills;
        assert(compute_pool_prepare(ex,dispatch,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_READY);
        VkExecPoolGrant *grant=dispatch->pool_grant;
        assert(grant&&grant->compute_query&&grant->compute_inputs&&grant->finite&&scope.refs==2&&metadata_count==3);
        assert(whole_calls==before_whole && refills==before_finite);
        pool_binding=(VkExecPoolBindScope){pipe,grant,9};
        assert(pool_index_bind_active(ex) && pool_index_bind_layout(ex,pipe));
        actual_mapping.range.offset=8;
        assert(!pool_index_bind_layout(ex,pipe));
        actual_mapping.range.offset=64;
        VkDescriptorBufferInfo input;
        assert(pool_index_bind_input(ex,NVN_STAGE_CS,-1,&buffers[0],&input) && input.buffer==191);
        VkDescriptorSet actual=0;
        assert(binding_manifest_prepare_pool(ex,pipe,&actual) && actual==private_set && actual==grant->set);
        assert(bound_views[0]==6000 && bound_views[1]==36001 && bound_views[4]==36004 &&
               bound_views[32767]==68767 && !bound_views[2] && !canonical_updates);
        assert(sampler_writes==(unsigned)marked+1 && marker_writes==(unsigned)marked);
        assert(carrier_records==3u*((unsigned)marked+1u) && image_writes==4u*((unsigned)marked+1u));
        assert(binding_manifest_prepare_pool(ex,pipe,&actual) && actual==private_set && native_owned);
        ex->cb++;
        assert(!binding_manifest_prepare_pool(ex,pipe,&actual) && !actual);
        ex->cb--;
        pool_binding=(VkExecPoolBindScope){0};
        const unsigned before_free=private_frees;
        vk_exec_pool_grant_release(&dispatch->pool_grant);
        assert(!dispatch->pool_grant && scope.refs==1 && metadata_count==1 && private_frees==before_free);
        if(marked)native_cancel(native_owned,1);
        else native_release(native_owned);
        native_owned=NULL;
        assert(private_frees==before_free+1&&!metadata_count&&!private_charges&&carrier_allocations==carrier_frees);
    }
    assert(PL.marks_owner_valid && PL.marks_owner==ex->tex_pool && !PL.marks_mixed);
    assert(copied==prior_copied+2);

    const unsigned previously_written=image_writes, previous_releases=carrier_frees;
    assert(compute_pool_prepare(ex,dispatch,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_READY);
    pool_binding=(VkExecPoolBindScope){pipe,dispatch->pool_grant,9};
    VkDescriptorSet actual=0;
    VkPipeInfo wrong=*pipe;
    assert(!binding_manifest_prepare_pool(ex,&wrong,&actual) && !actual);
    position.ordinal+=2;
    assert(!binding_manifest_prepare_pool(ex,pipe,&actual) && !actual);
    position=dispatch->read_position;
    ++sampler_generation;
    assert(!binding_manifest_prepare_pool(ex,pipe,&actual) && !actual);
    --sampler_generation;
    absent_zero=0;
    assert(!binding_manifest_prepare_pool(ex,pipe,&actual) && !actual);
    absent_zero=1;
    stale_slot=4;
    assert(!binding_manifest_prepare_pool(ex,pipe,&actual) && !actual);
    stale_slot=-1;
    pool_binding=(VkExecPoolBindScope){0};
    vk_exec_pool_grant_release(&dispatch->pool_grant);
    assert(!native_owned && carrier_frees==previous_releases+3 && image_writes==previously_written&&!metadata_count);

    missing_slot=32767;
    assert(compute_pool_prepare(ex,dispatch,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!dispatch->pool_grant && carrier_allocations==carrier_frees && !private_charges);
    missing_slot=-1;
    alloc_refused=1;
    assert(compute_pool_prepare(ex,dispatch,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!dispatch->pool_grant && carrier_allocations==carrier_frees && !private_charges && !native_owned);
    free(query.plan);
    puts("Compute query table: final fixed kind2 proof, exact full replica, three private carriers and robust-zero absence, sampler and marks, cancellation and stale rejection passed");
}

int main(void) {
    check_shader_store_refusals();
    GfxShader shader={shader_pool_indices_build(module,sizeof module / sizeof module[0],resources,counts),RC_EFFECT_WRITES_BUFFER};
    assert(shader.plan && shader.plan->domain_source==&buffers[0]);
    GfxShaderPoolIndex static_index;
    uint32_t static_count;
    assert(!shader_pool_indices_eval(shader.plan,NULL,NULL,&static_index,1,&static_count) && !static_count);
    const VkPipeBinding bindings[]={
        {.kind=GFX_RES_SSBO,.stage=NVN_STAGE_CS,.count=1,.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.binding=7,.res=&buffers[0]},
        {.kind=GFX_RES_SSBO,.stage=NVN_STAGE_CS,.count=1,.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.binding=12,.res=&buffers[1]},
    };
    VkPipeInfo pipe={.bind_point=VK_PIPELINE_BIND_POINT_COMPUTE,.n_bindings=2,.bindings=bindings,
        .compute_pool_domain_shader=&shader,.pool=1};
    DrawCtx dc={.storage_buffers={{0},{73,0,16}}};
    VkExec ex={.dc=&dc,.fence=7,.tex_pool=83,.samp_pool=&sampler_pool,.cb=34};
    const ComputeSelection selection={0};
    VkExecDispatch dispatch={.shader=&shader,.info=&pipe,.read_position={7,8},.selection=&selection};
    unsigned expected_drop=0;
    whole_result=VK_EXEC_GRAPHICS_READY;
    assert(compute_pool_prepare(&ex,&dispatch,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_READY);
    assert(dispatch.pool_grant && !dispatch.pool_grant->finite && !refills && !queries && !captures);
    vk_exec_pool_grant_release(&dispatch.pool_grant);
    whole_result=VK_EXEC_GRAPHICS_UNAVAILABLE;
    for(cpu_visible=0; cpu_visible<2; cpu_visible++) {
        unsigned previous_refills=refills;
        assert(compute_pool_prepare(&ex,&dispatch,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_READY);
        assert(dispatch.pool_grant->finite && dispatch.pool_grant->compute_info==&pipe &&
               dispatch.pool_grant->compute_inputs->inputs.values[0].snapshot==&replica);
        assert(metadata_count==2);
        assert(refills==previous_refills+1 && copied==1);
        assert(compute_pool_prepare(&ex,&dispatch,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_READY);
        assert(refills==previous_refills+1);
        pool_binding=(VkExecPoolBindScope){&pipe,dispatch.pool_grant,9};
        assert(pool_index_bind_active(&ex) && pool_index_bind_layout(&ex,&pipe));
        VkPipeInfo remapped=pipe;
        assert(!pool_index_bind_layout(&ex,&remapped));
        override_guard=1;
        assert(!pool_index_bind_layout(&ex,&pipe));
        override_guard=0;
        actual_mapping.range.offset=8;
        assert(!pool_index_bind_layout(&ex,&pipe)); // The native issued storage differs from early proof.
        actual_mapping.range.offset=64;
        dc.storage_buffers[1].buffer++;
        assert(!pool_index_bind_layout(&ex,&pipe));
        dc.storage_buffers[1].buffer--;
        bound_effects.output.recorded=0;
        assert(!pool_index_bind_layout(&ex,&pipe));
        bound_effects.output.recorded=1;
        VkDescriptorBufferInfo desc;
        assert(pool_index_bind_input(&ex,NVN_STAGE_CS,-1,&buffers[0],&desc) && desc.buffer==191 && desc.range==sizeof handles);
        assert(dc.pb_spoiled && registered==(unsigned)cpu_visible);
        domain_live=0;
        assert(!pool_index_bind_layout(&ex,&pipe));
        assert(pool_index_bind_input(&ex,NVN_STAGE_CS,-1,&buffers[0],&desc) && !desc.buffer);
        assert(!vk_exec_pool_grant_current(&ex,dispatch.pool_grant) && reports==(unsigned)cpu_visible+1);
        domain_live=1;
        pool_binding=(VkExecPoolBindScope){0};
        vk_exec_pool_grant_release(&dispatch.pool_grant);
        assert(snapshot_drops==++expected_drop && !metadata_count);
    }
    metadata_fail=1;
    assert(compute_pool_prepare(&ex,&dispatch,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!dispatch.pool_grant && !metadata_count && snapshot_drops==expected_drop);
    metadata_fail=2;
    assert(compute_pool_prepare(&ex,&dispatch,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!dispatch.pool_grant && !metadata_count && snapshot_drops==++expected_drop);
    overlap=1;
    unsigned admitted=refills;
    assert(compute_pool_prepare(&ex,&dispatch,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(refills==admitted && !dispatch.pool_grant && snapshot_drops==++expected_drop);
    overlap=0;
    pipe.compute_pool_domain_shader=NULL;
    unsigned attempted_queries=queries;
    assert(compute_pool_prepare(&ex,&dispatch,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(queries==attempted_queries && !dispatch.pool_grant);
    pipe.compute_pool_domain_shader=&shader;
    kinds|=1u<<3;
    assert(compute_pool_prepare(&ex,&dispatch,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(queries==attempted_queries && !dispatch.pool_grant);
    kinds=1u<<2;
    position.ordinal++;
    unsigned attempted_whole=whole_calls;
    assert(compute_pool_prepare(&ex,&dispatch,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(whole_calls==attempted_whole);
    position=dispatch.read_position;
    run_query_dispatch(&ex,&pipe,&dispatch);
    free(shader.plan);
    puts("Compute pool domain: fixed 32768 table and separate writer, exact replica binding, source writes, overlap and stale refusals passed");
}
'''


def main():
    source = Path(os.environ.get("COMPUTE_POOL_SOURCE", ROOT / "runtime/vk_exec_draw.c")).read_text(encoding="utf-8")
    pipeline = Path(os.environ.get("COMPUTE_POOL_PIPELINE_SOURCE", ROOT / "runtime/vk_pipeline.c")).read_text(encoding="utf-8")
    query_set = source[source.index("typedef struct ComputePoolQuerySet {"):source.index("struct ComputeSamples {")]
    assert "vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, 2, 1, &pool" in function(source, "bind_resources")
    assert "binding_manifest_prepare_pool(ex, info, &pool)" in function(source, "bind_resources")
    assert "vk_exec_pool_bind_enter" in function(source, "dispatch_bind_selected_resources")
    declaration = source[source.index("typedef struct {\n    VkExec *ex;\n    VkExecDispatch *dispatch;\n    ComputePoolIndexInputs *owner;"):
                         source.index("static int compute_pool_domain_disjoint(")]
    names = ("pool_sampler_descriptors", "pool_index_inputs_clear", "compute_pool_index_inputs_release", "pool_index_inputs_current",
             "vk_exec_pool_index_inputs_current", "pool_index_output_disjoint", "compute_pool_output_bindings_current",
             "pool_index_bind_active", "pool_index_bind_layout", "pool_index_bind_input", "compute_pool_domain_disjoint",
             "compute_pool_read_domain", "compute_pool_domain_indices", "pool_note_marks_owner",
             "compute_pool_query_capable", "compute_pool_query_release", "compute_pool_query_cancel",
             "compute_pool_query_shape_current_locked", "compute_pool_query_current", "compute_pool_query_bind",
             "vk_exec_pool_grant_current", "vk_exec_pool_grant_release", "graphics_pool_bind", "binding_manifest_prepare_pool",
             "compute_pool_query_resource", "compute_pool_query_descriptors", "compute_pool_query_prepare", "compute_pool_prepare")
    compiler = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(compiler.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="compute-pool-domain-") as directory:
        harness = Path(directory) / "check.c"
        exe = harness.with_suffix(".exe")
        harness.write_text(SHADER + QUERY_SHADER + MODEL + query_set + declaration +
                           "\n".join(function(source, name) for name in names) +
                           function(pipeline, "compute_pool_query_shape_kinds") + TEST, encoding="utf-8")
        subprocess.run([str(compiler), "-std=gnu11", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", "-I", str(ROOT / "runtime"),
                        str(harness), str(ROOT / "runtime/rc_sampled_image_query.c"), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()

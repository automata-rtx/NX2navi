"""Exercise ordinary fixed compute texture selection at the dispatched operation's read."""
import os
from pathlib import Path
import runpy
import subprocess
import tempfile

TESTS = Path(__file__).resolve().parent
function = runpy.run_path(str(TESTS / "ordered_graphics_feedback_check.py"))["function"]
DRAW = Path(os.environ.get("DIRECT_COMPUTE_SAMPLE_SOURCE", TESTS.parent / "runtime/vk_exec_draw.c")).read_text(encoding="utf-8")

PREFIX = r'''
#include <stdint.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
enum { NVN_STAGE_CS, GFX_STAGE_CS, GFX_RES_TEX, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
       VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, VK_IMAGE_VIEW_TYPE_3D, VK_IMAGE_TYPE_2D,
       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER=111,
       GFX_RES_IMG, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, OP_DISPATCH=27, OP_DISPATCH_IND, RC_EFFECT_WRITES_IMAGE=2, RC_EFFECT_WRITES_UNKNOWN=128,
       VK_TEXTURE_READ_FORCE_COMPAT=1, VK_TEXTURE_READ_REQUIRED=2, VK_TEXTURE_READ_LINEAR_RGBA8_VIEW=4,
       VK_TEXTURE_READ_EXACT_DETAIL=8, RC_BUDGET_BUFFER=1, VK_SUCCESS=0 };
#define NVN_HANDLE_TEX(h) ((int)(h))
#define VK_NULL_HANDLE 0
typedef uint64_t VkImage;
typedef int VkResult, VkPipeline;
typedef unsigned VkFormat;
enum {VK_FORMAT_UNDEFINED};
typedef struct {uint64_t memory,offset,size;} VkImageMemoryPlacement;
typedef struct { uint64_t scene,ordinal; } RcRegistryPosition;
typedef struct {uint64_t serial;} prog_t;
typedef struct {int exists;} GfxShader;
typedef struct {int type,slot,pool,exact,count,separate,flags;} GfxShaderRes;
typedef struct {uint32_t aspect,level0,levels,layer0,layers;} view_t;
typedef struct {uint64_t registration;int native_slot,canonical,projected;} VkTextureReadKey;
typedef struct {uint64_t address;} RcInterval;
typedef struct {VkImage image;uint32_t aspect,level0,levels,layer0,layers;} ViewKey;
typedef struct {int type,skip,bound,mirrored;VkImage image;uint64_t addr;unsigned format;} VkTexImage;
typedef struct {int live, recorded;} VkTextureQueryImage;
typedef struct {uint64_t registration,history_capture;int registration_only,association_only;VkImage image;} VkTextureSampleGeometry;
typedef enum {VK_TEXTURE_SAMPLE_GEOMETRY_READY, VK_TEXTURE_SAMPLE_GEOMETRY_PROVIDER} VkTextureSampleGeometryResult;
typedef struct {const VkTexImage *image;uint64_t registration;int native_slot,historical,canonical,projected,has_view;
    view_t view; VkTextureQueryImage *query_image;} VkTextureSample;
typedef enum {VK_TEXTURE_SAMPLE_READY,VK_TEXTURE_SAMPLE_PREPARE,VK_TEXTURE_SAMPLE_FALLBACK,VK_TEXTURE_SAMPLE_DEFERRED,
    VK_TEXTURE_SAMPLE_UNAVAILABLE,VK_TEXTURE_SAMPLE_ABSENT,VK_TEXTURE_SAMPLE_GONE,VK_TEXTURE_SAMPLE_CANONICAL_REFUSED,
    VK_TEXTURE_SAMPLE_UNBACKED} VkTextureSampleStatus;
enum { VK_TEXTURE_REGISTRATION_ABSENT=-1, VK_TEXTURE_REGISTRATION_EXACT=1 };
typedef enum { VK_EXEC_GRAPHICS_READY,VK_EXEC_GRAPHICS_INVALIDATED,VK_EXEC_GRAPHICS_UNAVAILABLE,
               VK_EXEC_GRAPHICS_NEEDS_ORDERED,VK_EXEC_GRAPHICS_STOPPED } VkExecGraphicsResult;
typedef enum { VK_EXEC_RESTORE_READY,VK_EXEC_RESTORE_UNAVAILABLE,VK_EXEC_RESTORE_STOPPED,VK_EXEC_RESTORE_STALE } VkExecRestoreResult;
#define VK_EXEC_ALIAS_READ_LIMIT 4
typedef struct {int live;} VkTextureAliasRead, VkTexturePresentSource, VkExecGpaSampleScope, RcBudgetTicket;
typedef struct {
    int status;
    const char *layout_gate;
    VkImage image;
    struct { uint64_t address; struct { uint64_t size; } range; } receiver;
} VkTextureAliasFailure;
enum {RC_REGISTRY_STALE = 2};
typedef struct {int result;} VkTextureSampleRefusal;
typedef struct {const GfxShader *shader;const GfxShaderRes *resource;int stage,member;uint64_t handle,token;
                VkTextureSample sample;VkTexImage image;VkTextureAliasRead *alias;VkTexturePresentSource *hold;
                VkTextureSampleGeometry geometry;} GraphicsSample;
typedef struct {const GfxShaderRes *res;int kind, stage, type;uint32_t count;unsigned binding,set;} VkPipeBinding;
typedef struct {int x,y,z;} p_dispatch;
typedef struct {uint32_t aspectMask;} VkImageSubresourceRange;
typedef struct {uint32_t scalar,set,binding;} RcNativeImageStore;
enum { RC_NATIVE_IMAGE_SEED_MAX_BINDINGS=8, RC_NATIVE_IMAGE_SEED_READ=1, RC_NATIVE_IMAGE_SEED_WRITE=2, RC_NATIVE_IMAGE_SEED_ATOMIC=4,
       RC_NATIVE_IMAGE_SEED_2D=0, RC_NATIVE_IMAGE_SEED_3D=1 };
typedef struct {uint8_t binding,image_format,scalar,access,dimension;} RcNativeImageSeedBinding;
typedef struct {RcNativeImageSeedBinding entries[RC_NATIVE_IMAGE_SEED_MAX_BINDINGS];uint8_t count;} RcNativeImageSeedBindings;
typedef enum {
    RESOURCE_IMAGE_TRACE_ADMISSION_UNOBSERVED, RESOURCE_IMAGE_TRACE_STORAGE_LAYOUT,
    RESOURCE_IMAGE_TRACE_STORAGE_BINDING, RESOURCE_IMAGE_TRACE_STORAGE_PEER,
    RESOURCE_IMAGE_TRACE_STORAGE_TARGET, RESOURCE_IMAGE_TRACE_STORAGE_PLANE, RESOURCE_IMAGE_TRACE_STORAGE_SELECTED,
} ResourceImageTraceAdmission;
typedef struct {uint32_t n_bindings;VkPipeBinding bindings[4];int pool,bind_point;uint64_t compute_query_size_lod0_mask;
                const GfxShader *compute_effect_shader; VkPipeline pipeline; RcNativeImageStore native_image_store;
                RcNativeImageSeedBindings native_image_seed_bindings,native_image_rmw_seed_bindings;} VkPipeInfo;
typedef int VkExecPoolGrant;
typedef uint64_t VkCommandBuffer;
typedef struct {const VkPipeInfo *info;VkExecPoolGrant *grant;uint32_t opcode;} VkExecPoolBindScope;
typedef struct {const VkPipeInfo *info;RcRegistryPosition read;RcBudgetTicket budget;uint32_t capacity,count;
                GraphicsSample entries[];} ComputeSamples;
typedef struct {VkImage prepared;} VkTextureWriteTarget;
typedef struct {VkTextureWriteTarget **targets;uint32_t count;} VkTextureWritePlan;
typedef struct {const GfxShaderRes *resource;int sampled,written,has_view,slot,stage,full_overwrite,phase_seen,owns_write,preserve_native;uint64_t handle,size;
                VkTexImage image;view_t view;void *independent;VkTextureWriteTarget *write_target;
                VkTexturePresentSource *read_hold; VkTextureAliasRead *storage_seed; VkImageSubresourceRange issued_planes;} EffectImageCapture;
typedef struct {int current;} DeviceStoragePointer;
typedef struct {ComputeSamples *samples;uint32_t count;EffectImageCapture *images;
                DeviceStoragePointer *pointer_sources;uint32_t pointer_count;ResourceImageTraceAdmission storage_admission;
                int opaque_storage_effects,bound_storage,host_mirror_storage;VkTextureWritePlan write_plan;} VkExecDispatchImages;
typedef struct {RcRegistryPosition read;uint64_t program,handle,registration;VkPipeline pipeline;
    int logical,member,status,native_exact,present;const char *gate,*geometry;
    VkTextureAliasFailure alias;VkTextureSampleRefusal legacy;} VkExecSampledFailure;
typedef struct {const char *stage,*reason;int subtype,binding,slot,member,result;
                uint64_t handle,image,peer_image,address,bytes;
    struct {const char *gate;} trace;VkExecSampledFailure sampled;} VkExecDispatchInputFailure;
static void sampled_failure_site(VkExecSampledFailure *failure,const GfxShader *shader,
    const GfxShaderRes *resource,const VkPipeInfo *info) { (void)shader; assert(failure->present && resource && info); }
static void sampled_failure_shape(VkExecSampledFailure *failure,const VkTexImage *image) { assert(failure->present && image); }
typedef struct {const VkPipeInfo *info; int current;} VkExecBufferScratchReceipt;
typedef struct {const VkPipeInfo *info; uint64_t ordinal; unsigned cell;} DrawDescriptorGrant;
typedef struct {VkExecDispatchImages *selection;const VkPipeInfo *info;RcRegistryPosition read_position;
                const GfxShader *shader;const prog_t *program;uint64_t program_serial,registration;
                uint32_t opcode;const void *payload;VkPipeline pipeline;VkExecDispatchInputFailure input_failure;
                VkExecBufferScratchReceipt *scratch;VkExecPoolGrant *pool_grant;DrawDescriptorGrant descriptors;
                int compatibility;} VkExecDispatch;
enum {VK_PIPELINE_BIND_POINT_COMPUTE};
typedef struct {
    const GfxShaderRes *resource;
    uint64_t handle;
    VkImage image;
    struct {uint64_t imageView;} descriptor;
} BindingImageSelection;
typedef struct {int mo_ovr_on;uint32_t n_storage_images;BindingImageSelection storage_images[4];} DrawCtx;
typedef struct {uint64_t fence, resource_op,tex[1][128],img[1][32];DrawCtx *dc;const prog_t *prog;
    struct {int width;} tonemap_extent; VkCommandBuffer cb;} VkExec;
typedef struct {ComputeSamples *compute_samples;const prog_t *program;uint64_t program_serial;uint32_t opcode;
                const GfxShader *compat_compute_shader; VkPipeline compat_compute_pipeline;
                VkExecBufferScratchReceipt *buffer_scratch; RcRegistryPosition position; DrawDescriptorGrant compat_descriptors; EffectImageCapture *images; uint32_t image_count; DeviceStoragePointer *pointer_sources; uint32_t pointer_count;
                ResourceImageTraceAdmission storage_admission;VkTextureWritePlan write_plan;} EffectDeclaration;
#if HAS_COMPAT
static EffectDeclaration *effect_current;
static int queued, materialized, refused, stopped, skip_shader;
static RcRegistryPosition read;
static unsigned descriptor_acquires, descriptor_cancels;
static int descriptor_refused;
static int draw_descriptor_grant_current(const VkExec *ex, const VkPipeInfo *info, const DrawDescriptorGrant *grant,
                                         uint64_t ordinal) {
    return ex && grant->info == info && grant->ordinal == ordinal && grant->cell == 1;
}
static VkExecGraphicsResult draw_descriptor_grant_prepare(VkExec *ex, const VkPipeInfo *info, DrawDescriptorGrant *grant,
                                                             int ordered) {
    assert(ordered && ex->resource_op == read.ordinal / 2);
    if (descriptor_refused) return VK_EXEC_GRAPHICS_UNAVAILABLE;
    if (!grant->cell) {
        *grant = (DrawDescriptorGrant){info, ex->resource_op, 1};
        descriptor_acquires++;
    }
    return VK_EXEC_GRAPHICS_READY;
}
static void draw_descriptor_grant_release(DrawDescriptorGrant *grant) {
    descriptor_cancels += grant->cell;
    *grant = (DrawDescriptorGrant){0};
}
static int device_shader_pointers_current(const DeviceStoragePointer *sources,uint32_t count) {
    return !count || (count==1 && sources && sources->current);
}
static void device_storage_pointers_free(DeviceStoragePointer *sources,uint32_t count) {
    assert(!!sources==!!count);free(sources);
}
static VkExecGraphicsResult dispatch_device_pointers_prepare(VkExec *ex,VkExecDispatch *dispatch) {
    assert(ex && dispatch->selection);
    return VK_EXEC_GRAPHICS_READY;
}
static VkExecGraphicsResult dispatch_buffers_prepare(VkExec *ex,VkExecDispatch *dispatch) {
    assert(draw_descriptor_grant_current(ex,dispatch->info,&dispatch->descriptors,ex->resource_op));
    return VK_EXEC_GRAPHICS_READY;
}
static int compute_ubo_merges_current(const VkExec *ex, const VkExecBufferScratchReceipt *scratch) {
    assert(ex);
    return !scratch || scratch->current;
}
static VkExecBufferScratchReceipt *buffer_scratch_adopt(VkExecBufferScratchReceipt **scratch, const VkExec *ex,
    uint32_t opcode, const void *payload, const GfxShader *shader, RcRegistryPosition position) {
    assert(ex && opcode && payload && shader);
    VkExecBufferScratchReceipt *kept = *scratch;
    if (kept) assert(kept->current && position.scene == ex->fence && position.ordinal == read.ordinal + 1);
    *scratch = NULL;
    return kept;
}
static int vk_sync_stopping(void){return stopped;}
static int vk_device_is_lost(void){return 0;}
static int vk_exec_recording_failed(void){return 0;}
static int pool_on(void){return 1;}
static int shader_skipped(const GfxShader *shader){assert(shader);return skip_shader;}
static uint64_t vk_texture_registration_publication(void){return 140;}
static void rt_log(const char *message, ...) {(void)message;}
static void vk_exec_required_graphics_unavailable(VkExec *ex,uint32_t opcode,const char *reason){
    assert(ex && opcode && reason); ++refused;
}
#endif
#if PREVIOUS_UNIFORM_GATE
static int dispatch_needs_sample_cohort(const VkExec *ex,const VkExecDispatch *dispatch){assert(ex&&dispatch);return 0;}
#endif
static int lock_count, reports, failure, allocations, releases, all_current, aged;
static int geometry_checks, query_creates, query_releases, query_records, query_bound, selected_redirect, historical_query;
static int query_current=1, carrier_fails;
static VkTextureQueryImage query_carrier;
static VkExecDispatch *dispatch_current;
static DrawDescriptorGrant *descriptor_binding_grant;
static int fallback, gone, resolves, retained, hold_releases;
static void *vk_dlaa_swap_img;
static VkTextureAliasRead raw_receipt;
static int alias_captures, alias_cancels, remap_during_materialization;
#if HAS_COMPAT_STORAGE
static VkTextureAliasRead storage_seed;
static unsigned storage_seed_cancels;
#endif
static int pending_raw, pending_valid=1;
static VkTexturePresentSource resident={1}, historical={1};
static const VkTexImage image={.image=77};
static const VkTexImage comparison_image={.image=620,.format=124,.addr=UINT64_C(0x800193f800)};
static int comparison_select;
static RcRegistryPosition read={9,10};
static void vk_texture_lock(void){assert(!lock_count++);}
static void vk_texture_unlock(void){assert(lock_count--==1);}
static int texture_deferred;
static int vk_texture_deferred(void){return texture_deferred;}
static void vk_texture_deferred_take(void){assert(texture_deferred);texture_deferred=0;}
static int vk_replay_metadata_allocate(size_t bytes,int clear,int producer,RcBudgetTicket *budget,void **out){
    assert(bytes && clear && producer==RC_BUDGET_BUFFER);*out=calloc(1,bytes);assert(*out);++allocations;budget->live=1;return VK_SUCCESS;
}
static RcRegistryPosition vk_exec_resource_position(void){return read;}
static int rc_registry_position_compare(RcRegistryPosition a,RcRegistryPosition b){return a.scene!=b.scene || a.ordinal!=b.ordinal;}
static int graphics_sample_shape_current(const GraphicsSample *sample){
    assert(sample&&lock_count);
    return all_current && (!sample->sample.query_image || query_current) && (!sample->alias || !pending_raw);
}
static int vk_texture_alias_sample_selection_current_locked(int logical,VkTextureAliasRead *alias) {
    assert(logical==16 && alias==&raw_receipt && alias->live && lock_count);
    return all_current && (!pending_raw || pending_valid);
}
static int twin_compare_resource(const GfxShaderRes *resource){return resource->type==98||resource->type==0x503;}
static int res_view_type(int type){return type;}
static uint64_t token_of(uint64_t handle){assert(handle);return handle+100;}
static VkTexturePresentSource *vk_texture_present_retain_image(const VkTexImage *selected,VkImage chosen){
    assert((selected==&image&&chosen==image.image)||
           (comparison_select&&selected==&comparison_image&&chosen==comparison_image.image));
    ++retained;
    return &resident;
}
static VkTexturePresentSource *vk_texture_history_sample_retain_locked(int logical,uint64_t registration,VkImage chosen) {
    assert(aged && logical==17 && registration==70 && chosen==image.image);
    ++retained;
    return &historical;
}
static int sample_geometry_capture_locked(GraphicsSample *sample,const char **reason) {
    assert(sample&&reason&&sample->handle==17&&lock_count);
    ++geometry_checks;
    if (historical_query) {
        sample->sample=(VkTextureSample){.registration=70,.historical=VK_TEXTURE_REGISTRATION_EXACT};
        sample->geometry=(VkTextureSampleGeometry){.registration=70,.history_capture=123};
        return 1;
    }
    if (selected_redirect) { *reason="associated-shape-needs-selection"; return 0; }
    sample->sample=(VkTextureSample){.image=&image,.registration=78};
    sample->geometry.registration=78;
    sample->image=image;
    sample->hold=vk_texture_present_retain_image(&image,image.image);
    return 1;
}
static const char *vk_texture_sample_geometry_status_name(VkTextureSampleGeometryResult result) {
    assert(result == VK_TEXTURE_SAMPLE_GEOMETRY_PROVIDER);
    return "association-refused";
}
static VkTextureSampleGeometryResult vk_texture_sample_associated_geometry_locked(int logical,
    const VkTextureSample *source,VkTextureSample *sample,VkTextureSampleGeometry *geometry) {
    assert(logical==17 && source && source->image==&image && source->registration==78 && lock_count);
    if (!selected_redirect) return VK_TEXTURE_SAMPLE_GEOMETRY_PROVIDER;
    *sample=*source;
    *geometry=(VkTextureSampleGeometry){.registration=78,.association_only=1,.image=image.image};
    return VK_TEXTURE_SAMPLE_GEOMETRY_READY;
}
static VkResult vk_texture_query_image_create_locked(int logical,const VkTextureSample *sample,
    const VkTextureSampleGeometry *geometry,VkTextureQueryImage **out) {
    assert(logical==17&&lock_count&&!*out&&!query_carrier.live);
    if (historical_query) {
        assert(!sample->image&&sample->historical==VK_TEXTURE_REGISTRATION_EXACT&&
               geometry->registration==70&&geometry->history_capture==123);
    } else {
        assert(sample->image==&image&&geometry->registration==78);
    }
    if (carrier_fails) return -1;
    query_carrier=(VkTextureQueryImage){.live=1};
    *out=&query_carrier;
    ++query_creates;
    return VK_SUCCESS;
}
static void vk_texture_query_image_release(VkTextureQueryImage **image) {
    if (!*image) return;
    assert(*image==&query_carrier && query_carrier.live);
    query_carrier.live=0;
    *image=NULL;
    ++query_releases;
}
static int vk_texture_query_image_view(const VkExec *ex,const VkTextureQueryImage *query,VkImage *native) {
    assert(ex->cb==55 && query==&query_carrier && query->live && !native);
    return query->recorded;
}
static int vk_texture_query_image_record_locked(VkExec *ex,VkTextureQueryImage *query) {
    assert(lock_count && ex->cb==55 && query==&query_carrier && query->live && !query->recorded);
    if (!query_current) return 0;
    query->recorded=1;
    ++query_records;
    return 1;
}
static int vk_exec_record_between_renderings(VkExec *ex,int (*record)(VkCommandBuffer,void *),void *work) {
    assert(!lock_count && ex->cb==55);
    return record(ex->cb,work);
}
typedef struct {
    VkExec *ex;
    GraphicsSample *samples;
    uint32_t count;
    int complete;
} SampleQueryCommands;
static int retained_original;
static int retained_compat_sample_current(const VkExec *ex,const GraphicsSample *sample) {
    assert(ex&&sample&&lock_count);
    return retained_original && sample->hold && !sample->alias;
}
static void retain_compat_sample(GraphicsSample *sample) {assert(sample&&lock_count);}
static void retained_sample_refusal(VkExec *ex,const GraphicsSample *sample,RcRegistryPosition read,
    uint32_t opcode,int stage,const char *reason) {
    assert(sample && read.scene && stage==NVN_STAGE_CS);
    vk_exec_required_graphics_unavailable(ex,opcode,reason);
}
static int compute_sample(VkExec *,uint32_t,const GfxShaderRes *,uint64_t,VkTextureSample *,
    VkTextureAliasRead **,const VkTexturePresentSource **);
static VkExecGraphicsResult vk_exec_pool_prepare_info(VkExec *ex,int kinds,VkExecPoolGrant **out,
    int required,int ordered,RcRegistryPosition attempted,void *pending,void *failure_out) {
    (void)ex;(void)kinds;(void)out;(void)required;(void)ordered;(void)attempted;(void)pending;(void)failure_out;abort();
}
static int vk_exec_pool_shader_kinds(const GfxShader *shader){(void)shader;abort();}
static void vk_exec_pool_grant_release(VkExecPoolGrant **grant){assert(!*grant);}
static int vk_exec_pool_grant_current(VkExec *ex,VkExecPoolGrant *grant){(void)ex;(void)grant;abort();}
static void vk_exec_cancel_temporal_preparation(void){abort();}
static VkExecPoolBindScope vk_exec_pool_bind_enter(VkExecPoolBindScope next){
    assert(!next.info && !next.grant);return (VkExecPoolBindScope){0};
}
static void bind_resources(VkExec *ex,const VkPipeInfo *info){
    const DrawDescriptorGrant *expected=dispatch_current ? &dispatch_current->descriptors : &effect_current->compat_descriptors;
    assert(descriptor_binding_grant==expected && draw_descriptor_grant_current(ex,info,expected,ex->resource_op));
    VkTextureSample sampled={0};
    VkTextureAliasRead *alias=NULL;
    const VkTexturePresentSource *source=NULL;
    vk_texture_lock();
    assert(compute_sample(ex,0,info->bindings[0].res,17,&sampled,&alias,&source)==2);
    assert(sampled.query_image==&query_carrier && query_carrier.recorded && source==&resident && !alias);
    vk_texture_unlock();
    ++query_bound;
}
static int linear_control = 1, exact_control;
static int defer_selected_sample, unavailable_selected_sample;
static VkTextureSampleStatus vk_texture_ordered_sample_select_locked(int logical,RcRegistryPosition sample_read,
    unsigned flags,unsigned options,VkTextureSample *sample,VkTextureAliasRead **alias,VkTexImage *shape,
    VkTextureAliasFailure *failure_out) {
    const unsigned expected_flags = (linear_control ? VK_TEXTURE_READ_LINEAR_RGBA8_VIEW : 0) |
                                    (exact_control ? VK_TEXTURE_READ_EXACT_DETAIL : 0);
    assert(lock_count && flags==expected_flags && failure_out && !options &&
           !rc_registry_position_compare(read,sample_read));
    ++reports;
    if (logical==15) {
        if (!failure) return VK_TEXTURE_SAMPLE_ABSENT;
        *failure_out = (VkTextureAliasFailure){.status=91,.layout_gate="provider-layout",.image=42,.receiver={512,{16}}};
        return VK_TEXTURE_SAMPLE_UNAVAILABLE;
    }
    if (logical==16) {
        assert(!raw_receipt.live && !*alias);
        raw_receipt.live=1;
        ++alias_captures;
        *alias=&raw_receipt;
        shape->image=0;
        sample->registration=77;
        return VK_TEXTURE_SAMPLE_READY;
    }
    assert(logical==17);
    if (comparison_select) {
        *sample=(VkTextureSample){.image=&comparison_image,.registration=190,.projected=1,.has_view=1,
                                 .native_slot=106,.view={2,0,1,0,3}};
        return VK_TEXTURE_SAMPLE_READY;
    }
    sample->image=&image;
    if (unavailable_selected_sample) {
        *failure_out=(VkTextureAliasFailure){.status=91,.layout_gate="peer-proof-unavailable"};
        return VK_TEXTURE_SAMPLE_UNAVAILABLE;
    }
    if (defer_selected_sample) {
        texture_deferred=1;
        return VK_TEXTURE_SAMPLE_DEFERRED;
    }
    if (gone) return VK_TEXTURE_SAMPLE_GONE;
    if (fallback) return VK_TEXTURE_SAMPLE_FALLBACK;
    if(aged) {
        sample->historical=VK_TEXTURE_REGISTRATION_EXACT;sample->native_slot=-1;sample->registration=70;
    } else {
        sample->canonical=1;sample->registration=78;
    }
    return VK_TEXTURE_SAMPLE_READY;
}
static VkTextureSampleStatus vk_texture_resolve_sample_locked(int slot,unsigned flags,VkTextureSample *out){
    const unsigned expected_flags = VK_TEXTURE_READ_REQUIRED | (linear_control ? VK_TEXTURE_READ_LINEAR_RGBA8_VIEW : 0) |
                                    (exact_control ? VK_TEXTURE_READ_EXACT_DETAIL : 0);
    assert(fallback && slot==17 && flags==expected_flags && lock_count);
    ++resolves;
    *out=(VkTextureSample){.image=&image,.native_slot=11,.has_view=1,.view={1,1,1,0,1}};
    return VK_TEXTURE_SAMPLE_FALLBACK;
}
static VkTextureSampleStatus vk_texture_resolve_sample_report_locked(int slot,unsigned flags,VkTextureSample *out,
    VkTextureSampleRefusal *refused) {
    assert(refused);return vk_texture_resolve_sample_locked(slot,flags,out);
}
static int graphics_sample_wait(VkExec *ex,int logical,RcRegistryPosition attempted,VkExecGpaSampleScope **pending){
    (void)ex;(void)logical;(void)attempted;(void)pending;abort();
}
static void vk_texture_alias_read_cancel(VkTextureAliasRead *alias) {
    if (!alias) return;
#if HAS_COMPAT_STORAGE
    if (alias == &storage_seed) {
        assert(alias->live);
        alias->live = 0;
        storage_seed_cancels++;
        return;
    }
#endif
    assert(alias==&raw_receipt && alias->live);
    alias->live=0;
    ++alias_cancels;
}
static void vk_texture_present_release(VkTexturePresentSource *hold){
    if(hold){assert(hold==&resident||hold==&historical);++hold_releases;}
}
static void vk_replay_metadata_free(void *out,RcBudgetTicket *budget){
    assert(out&&budget->live);++releases;free(out);
}
#if HAS_STORAGE_SAMPLE_RECEIPT
static int vk_dlaa_swap_active(const VkExec *ex){assert(ex);return 1;}
static int texture_view_key(const VkTexImage *image,const view_t *view,int hv,uint16_t type,int storage,ViewKey *out){
    assert(type == 0 || type == 3 || type == 0x503);
    if(!image->image)return 0;
    *out=(ViewKey){image->image,hv?view->aspect:1,hv?view->level0:0,storage?1:hv?view->levels:2,
                   hv?view->layer0:0,hv?view->layers:1};
    return 1;
}
static VkImage vk_texture_write_target_prepared_image(const VkTextureWriteTarget *target){return target->prepared;}
#endif
#if HAS_INPUT_REPORT
static VkExecGraphicsResult dispatch_record_input_stage(VkExecDispatch *dispatch,const char *stage,VkExecGraphicsResult result){
    assert(dispatch&&stage);
    if(result!=VK_EXEC_GRAPHICS_READY)dispatch->input_failure.stage=stage,dispatch->input_failure.result=result;
    return result;
}
#endif
#if HAS_COMPAT
static const VkTexImage assembled={.image=199};
static VkExecRestoreResult vk_exec_prepare_alias_texture_read(VkExec *ex, VkTextureAliasRead **aliases,
    uint32_t count, int (*current)(void *), void *position, const VkTexImage **images) {
    // The raw material fixture offers exactly the original retained provider. No VA lookup is available here.
    assert(ex && count==1 && aliases[0]==&raw_receipt && aliases[0]->live && !lock_count);
    if (remap_during_materialization || !pending_valid) all_current=0;
    if (all_current) pending_raw=0; // The recorder must settle before the strict callback.
    if (!current(position)) return VK_EXEC_RESTORE_STALE;
    ++materialized;
    images[0]=&assembled;
    return VK_EXEC_RESTORE_READY;
}
static int alias_sample_view_locked(VkTextureAliasRead *alias,const VkTexImage *image,const view_t *view,int hv,int type) {
    assert(alias==&raw_receipt && image->image==assembled.image && view && !hv && type==3 && lock_count);
    return 1;
}
static void vk_texture_alias_read_use(VkTextureAliasRead *alias,uint64_t scene) {
    assert(alias==&raw_receipt && scene==read.scene && lock_count);
    ++queued;
}
#if HAS_COMPAT_STORAGE
static const GfxShader storage_shader={1};
static const GfxShaderRes storage_binding={.type=3,.slot=3,.flags=4,.count=1};
static const VkImage storage_original=309, storage_image=310;
static int storage_target_live, storage_target_current=1, storage_seed_uploaded=1;
static int storage_capture_result=1, storage_planes_exact=1, storage_inventory=1, storage_opaque;
static int storage_capture_defers;
static VkExecGraphicsResult storage_access_result=VK_EXEC_GRAPHICS_READY, storage_seed_result=VK_EXEC_GRAPHICS_READY;
static unsigned storage_captures, storage_accesses, storage_seeds, storage_releases, storage_answer_lookups;
static uint32_t storage_tex_queries, storage_image_queries=1u<<3;
static int storage_answers_match=1;
typedef struct {uint8_t tex_type[32], img_type[32], img_format[32];} GfxShaderAnswers;
static const VkTexImage storage_shape={.image=309,.addr=0x19000};

static unsigned gfx_shader_guest_writes(const GfxShader *shader) {
    return shader!=&storage_shader || storage_opaque ? RC_EFFECT_WRITES_UNKNOWN : RC_EFFECT_WRITES_IMAGE;
}
static int rc_native_image_seed_store(const RcNativeImageSeedBinding *binding,RcNativeImageStore *proof) {
    if (!(binding->access&RC_NATIVE_IMAGE_SEED_WRITE)) return 0;
    *proof=(RcNativeImageStore){.scalar=binding->scalar,.binding=binding->binding};
    return proof->scalar!=0;
}
static int gfx_shader_write_inventory_complete(const GfxShader *shader) {
    assert(shader==&storage_shader);
    return storage_inventory;
}
static void gfx_shader_slot_dependencies(const GfxShader *shader,uint32_t *textures,uint32_t *images) {
    assert(shader==&storage_shader);
    *textures=storage_tex_queries; *images=storage_image_queries;
}
static int find_stage(const void *dc,const prog_t *prog,int stage,const uint8_t **control,
    const uint8_t **code,uint32_t *length) {
    static const uint8_t ctl=9;
    assert(dc && prog->serial==83 && stage==NVN_STAGE_CS);
    *control=&ctl; *code=NULL; *length=0;
    return 1;
}
static uint32_t slot_mask_of(const void *dc,uint64_t program,int stage) {
    assert(dc && program==83 && stage==NVN_STAGE_CS);
    return 1u<<3;
}
static int answers_for(VkExec *ex,int stage,const uint8_t *control,uint32_t mask,GfxShaderAnswers *answers) {
    assert(ex->dc && stage==NVN_STAGE_CS && *control==9 && mask==(1u<<3));
    *answers=(GfxShaderAnswers){.img_type={[3]=50,[11]=60},.img_format={[3]=70,[11]=80}};
    return 1;
}
static int bound_chosen,bound_pending,bound_prepares;
static int gfx_type_of(const VkTexImage *image,const view_t *view,int has_view) {
    assert(image->image==(bound_chosen && !bound_pending?storage_original+1:storage_original) && view->levels==1 && has_view);
    return 3;
}
static uint8_t storage_image_format_answer(const VkTexImage *image,const view_t *view,int has_view,int written) {
    assert(image->image==(bound_chosen && !bound_pending?storage_original+1:storage_original) && view->layers==1 && has_view && written);
    return 15;
}
static const GfxShader *shader_for_resolve_answers_mode(VkExec *ex,int stage,int *present,int *pending,unsigned wait,
    const GfxShaderAnswers *answers,int lookup_only) {
    assert(ex->prog->serial==83 && stage==NVN_STAGE_CS && present && !pending && !wait && lookup_only);
    assert(answers->img_type[3]==3 && answers->img_format[3]==15 &&
           answers->img_type[11]==60 && answers->img_format[11]==80);
    storage_answer_lookups++;
    return storage_answers_match ? &storage_shader : NULL;
}
enum { VK_TEXTURE_STORAGE_CAPTURE_UNSUPPORTED=-2,VK_TEXTURE_STORAGE_CAPTURE_INVALIDATED=-1,
       VK_TEXTURE_STORAGE_CAPTURE_ABSENT=0,VK_TEXTURE_STORAGE_CAPTURE_READY=1,VK_TEXTURE_STORAGE_CAPTURE_COW=2,VK_TEXTURE_STORAGE_CAPTURE_BOUND=3 };
static int vk_texture_effect_independent_storage(int logical,int exact,int written,VkTexImage *image,view_t *view,
    int *has_view,uint64_t *size,void **target,const char **reason) {
    (void)exact;(void)written;
    assert(!lock_count && !*target);
    storage_captures++;
    if (logical==32) {
        assert(storage_target_live);
        return 0;
    }
    assert(logical==31 && !storage_target_live);
    if (storage_capture_result <= 0) {
        texture_deferred=storage_capture_defers;
        if (reason && storage_capture_result==VK_TEXTURE_STORAGE_CAPTURE_UNSUPPORTED) *reason="storage-target-material";
        return storage_capture_result;
    }
    *image=storage_shape; *view=(view_t){1,0,1,0,1}; *has_view=1; *size=64;
    *target=&storage_target_live; storage_target_live=1;
    return 1;
}
static int vk_texture_storage_target_replaced_stale(const void *target) {(void)target;return 0;}
static VkTextureWriteTarget host_material;
static int host_live,host_chosen,host_prepares,host_cancels;
static VkExecRestoreResult bound_disposition;
static void *vk_texture_storage_target_bound_plan(void *target) {
    return bound_pending ? target : NULL;
}
static VkExecRestoreResult vk_exec_prepare_storage_target_bound(VkExec *ex,void *target,VkTexImage *image) {
    assert(ex && target && bound_pending && image->image==storage_shape.image);
    bound_prepares++;
    if(bound_disposition!=VK_EXEC_RESTORE_READY)return bound_disposition;
    bound_pending=0;
    image->image=storage_shape.image+1;
    return VK_EXEC_RESTORE_READY;
}
static int vk_texture_effect_storage_capture(int logical,int exact,VkTexImage *image,view_t *view,
    int *has_view,uint64_t *size,void **target,VkTextureWriteTarget **material,int written,int allow_bound,const char **reason) {
    if (bound_chosen && logical==31) {
        assert(allow_bound && !*target);
        const int capture=vk_texture_effect_independent_storage(logical,exact,written,image,view,has_view,size,target,reason);
        assert(capture==VK_TEXTURE_STORAGE_CAPTURE_READY);
        bound_pending=1;
        return VK_TEXTURE_STORAGE_CAPTURE_BOUND;
    }
    if (!host_chosen) return vk_texture_effect_independent_storage(logical,exact,written,image,view,has_view,size,target,reason);
    assert(material && !*material && !*target && logical==31 && !host_live);
    *image=storage_shape; *view=(view_t){1,0,1,0,1}; *has_view=1; *size=64;
    *material=&host_material;
    host_live=1;
    return VK_TEXTURE_STORAGE_CAPTURE_COW;
}
static void vk_texture_write_target_cancel(VkTextureWriteTarget *target) {
    assert(host_live && target==&host_material);
    host_live=0;host_cancels++;target->prepared=0;
}
static int vk_texture_write_target_current(const VkTextureWriteTarget *target) {
    return target==&host_material && host_live;
}
static int vk_texture_write_plan_independent(const VkTextureWritePlan *plan) {
    return plan->count==1 && plan->targets[0]==&host_material && host_live;
}
static VkExecGraphicsResult dispatch_first_write_prepare(VkExec *ex,const VkTextureWritePlan *plan) {
    assert(ex&&vk_texture_write_plan_independent(plan) && !host_material.prepared);
    host_material.prepared=902;
    host_prepares++;
    return VK_EXEC_GRAPHICS_READY;
}
static int vk_texture_storage_target_current(const void *target) {
    assert(target==&storage_target_live && storage_target_live);
    return storage_target_current;
}
static void vk_texture_storage_target_release(void *target) {
    if (!target) return;
    assert(target==&storage_target_live && storage_target_live && !storage_seed.live);
    storage_target_live=0;
    storage_releases++;
}
static int vk_texture_alias_read_current(const VkTextureAliasRead *seed) {
    assert(seed==&storage_seed);
    return seed->live;
}
static VkImage vk_texture_storage_target_image(const void *target,const VkTextureAliasRead *seed) {
    assert(target==&storage_target_live && storage_target_live && seed==&storage_seed && seed->live);
    return storage_image;
}
static int vk_texture_alias_target_current(const VkTextureAliasRead *seed,VkImage selected) {
    assert(seed==&storage_seed && selected==storage_image);
    return seed->live && storage_seed_uploaded;
}
#if HAS_HELD_STORAGE_LOAD
static int vk_texture_storage_target_load_current(const void *target,const VkImageSubresourceRange *planes) {
    assert(target==&storage_target_live && storage_target_live && planes->aspectMask==1);
    return 0;
}
#else
static int vk_texture_render_target_load_current(VkImage image,uint64_t addr,const VkImageSubresourceRange *planes) {
    assert(image==storage_original && addr==storage_shape.addr && planes->aspectMask==1);
    return 0;
}
#endif
static int dispatch_storage_planes(const EffectImageCapture *capture,VkImageSubresourceRange *planes,uint32_t extent[2]) {
    assert(capture->independent==&storage_target_live && storage_target_live && capture->image.image==(bound_chosen?storage_original+1:storage_original) &&
           capture->view.levels==1 && capture->view.layers==1);
    *planes=(VkImageSubresourceRange){1}; extent[0]=8; extent[1]=8;
    return storage_planes_exact;
}
static void vk_exec_dispatch_input_trace_log(const VkExecDispatch *dispatch,const char *stage,VkExecGraphicsResult result) {
    (void)dispatch; (void)stage; (void)result;
}

static int storage_integer_view(const VkTexImage *image,const view_t *view,int has_view,VkFormat *format) {
    (void)view; (void)has_view;
    *format=image->format;
    return 0;
}
static int nvn_gpu_only(uint64_t address) {(void)address; return 0;}
static const struct { int robust_image_access; } vk_caps = {1};
static int dispatch_storage_seed_read_format(const RcNativeImageSeedBinding *proof,unsigned format) {
    return proof->scalar && !format;
}
static VkExecGraphicsResult dispatch_storage_access_refused(const VkExecDispatch *dispatch,
    const EffectImageCapture *capture, const char *reason) {
    (void)dispatch; (void)capture; assert(reason);
    return VK_EXEC_GRAPHICS_UNAVAILABLE;
}
static VkExecGraphicsResult dispatch_storage_overwrite_select(VkExecDispatch *dispatch) {
    const EffectImageCapture *capture=dispatch->selection->images;
    if (capture->write_target) {
        assert(capture->write_target==&host_material && host_live && capture->image.image==host_material.prepared);
        return VK_EXEC_GRAPHICS_READY;
    }
    assert(dispatch->selection->count==1 && dispatch->shader==&storage_shader && !dispatch->info->native_image_store.scalar &&
           dispatch->info->native_image_seed_bindings.count==1 &&
           capture->resource==&storage_binding && capture->written && capture->stage==NVN_STAGE_CS && capture->slot==3 &&
           capture->handle==31 && capture->independent==&storage_target_live && storage_target_live);
    storage_accesses++;
    return storage_access_result;
}
static VkExecGraphicsResult dispatch_storage_seeds_prepare(VkExec *ex,VkExecDispatch *dispatch) {
    EffectImageCapture *capture=dispatch->selection->images;
    if (capture->write_target) {
        assert(ex && capture->write_target==&host_material && host_live && !capture->storage_seed);
        return VK_EXEC_GRAPHICS_READY;
    }
    assert(dispatch->selection->count==1 && capture->independent==&storage_target_live &&
           storage_target_live && !storage_seed.live && !capture->issued_planes.aspectMask && !capture->full_overwrite &&
           !rc_registry_position_compare(dispatch->read_position,read) && ex->resource_op==read.ordinal/2);
    storage_seeds++;
    if (storage_seed_result!=VK_EXEC_GRAPHICS_READY) return storage_seed_result;
    capture->issued_planes.aspectMask=1;
    capture->storage_seed=&storage_seed;
    storage_seed.live=1;
    return VK_EXEC_GRAPHICS_READY;
}
static void rt_fatal(const char *reason) { fprintf(stderr,"%s\n",reason); abort(); }
#endif
#endif
'''

CASES = r'''
#if HAS_COMPAT_STORAGE
static void check_compat_storage(void) {
    VkExecDispatchImages selection={0};
    const prog_t program={83};
    const p_dispatch payload={2,3,1};
    const VkPipeInfo info={.n_bindings=1,.pipeline=44,
        .bindings={{&storage_binding,GFX_RES_IMG,GFX_STAGE_CS,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,2,0}},
        .native_image_seed_bindings={.count=1,.entries={{.binding=2,.scalar=1,.access=RC_NATIVE_IMAGE_SEED_WRITE}}}};
    VkExecDispatch dispatch={.selection=&selection,.info=&info,.read_position=read,.shader=&storage_shader,
        .program=&program,.program_serial=program.serial,.registration=140,.opcode=OP_DISPATCH,.payload=&payload,.pipeline=44,
        .compatibility=1};
    VkExec ex={.fence=read.scene,.resource_op=read.ordinal/2,.img={{[3]=31}},.prog=&program};
    __typeof__(*ex.dc) ctx={0}; ex.dc=&ctx;
    const RcRegistryPosition none={0};
    EffectDeclaration effect={.position={read.scene,read.ordinal+1}};
    assert(vk_exec_dispatch_compat_storage_binding(&dispatch)==&info.bindings[0]);

    // The native eligibility attempt found no writable material. Compatibility still owes two reads
    // before gathering any destination, including a current already-held earlier member on retry.
    const GfxShaderRes peer={.type=3,.slot=0,.count=2};
    VkPipeInfo with_samples=info;
    with_samples.n_bindings=2;
    with_samples.bindings[1]=(VkPipeBinding){&peer,GFX_RES_TEX,GFX_STAGE_CS,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,2,1,0};
    dispatch.info=&with_samples;
    ex.tex[NVN_STAGE_CS][0]=16;
    ex.tex[NVN_STAGE_CS][1]=17;
    defer_selected_sample=1;
    int reports_before=reports;
    VkExecGraphicsResult initial_peer=vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL);
    if (initial_peer!=VK_EXEC_GRAPHICS_INVALIDATED)
        fprintf(stderr,"original sample preflight result=%d stage=%s reason=%s\n",
                initial_peer,dispatch.input_failure.stage,dispatch.input_failure.reason);
    assert(initial_peer==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!strcmp(dispatch.input_failure.stage,"sample-capture") &&
           !strcmp(dispatch.input_failure.reason,"sample-selection-unsettled"));
    assert(!texture_deferred && !storage_captures && !selection.count &&
           selection.samples && selection.samples->count==1 &&
           selection.samples->entries[0].alias==&raw_receipt &&
           reports==reports_before+2);
    ComputeSamples *sample_owner=selection.samples;
    defer_selected_sample=0;
    assert(compute_samples_select(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY &&
           selection.samples==sample_owner && selection.samples->count==2 &&
           selection.samples->entries[0].alias==&raw_receipt &&
           selection.samples->entries[1].hold==&resident && reports==reports_before+3);
    assert(compute_aliases_prepare(&ex,selection.samples)==VK_EXEC_GRAPHICS_READY);
    assert(compute_samples_current(&ex,dispatch.info,selection.samples));
    read.ordinal+=2;
    assert(compute_samples_select(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_INVALIDATED &&
           !selection.samples && !raw_receipt.live);
    read=dispatch.read_position;

    // Retry cancellation drops the original pin and leaves both writer and descriptor unissued.
    defer_selected_sample=1;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_INVALIDATED &&
           selection.samples && raw_receipt.live);
    compute_samples_release(&selection.samples);
    assert(!raw_receipt.live && !storage_captures && !dispatch.descriptors.cell);
    defer_selected_sample=0;
    unavailable_selected_sample=1;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE &&
           !strcmp(dispatch.input_failure.stage,"sample-capture") &&
           !strcmp(dispatch.input_failure.reason,"peer-proof-unavailable"));
    assert(!selection.samples && !raw_receipt.live && !storage_captures && !dispatch.descriptors.cell);
    unavailable_selected_sample=0;
    ex.tex[NVN_STAGE_CS][0]=ex.tex[NVN_STAGE_CS][1]=0;
    dispatch.info=&info;

    const GfxShader other_shader={1};
    dispatch.shader=&other_shader;
    assert(!vk_exec_dispatch_compat_storage_binding(&dispatch));
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
    assert(!selection.count && !storage_captures && selection.storage_admission==RESOURCE_IMAGE_TRACE_STORAGE_LAYOUT);
    draw_descriptor_grant_release(&dispatch.descriptors);
    dispatch.shader=&storage_shader;
    dispatch.pipeline=45;
    assert(!vk_exec_dispatch_compat_storage_binding(&dispatch));
    dispatch.pipeline=44;
    VkPipeInfo wrong_binding=info;
    dispatch.info=&wrong_binding;
    wrong_binding.bindings[0].binding=3;
    assert(!vk_exec_dispatch_compat_storage_binding(&dispatch));
    wrong_binding.bindings[0].binding=2;
    wrong_binding.bindings[0].set=1;
    assert(!vk_exec_dispatch_compat_storage_binding(&dispatch));
    dispatch.info=&info;
    storage_inventory=0;
    assert(!vk_exec_dispatch_compat_storage_binding(&dispatch));
    storage_inventory=1;
    storage_opaque=1;
    assert(!vk_exec_dispatch_compat_storage_binding(&dispatch));
    storage_opaque=0;
    VkPipeInfo multiple=info;
    const GfxShaderRes independent={.type=3,.slot=4,.flags=4,.count=1};
    multiple.n_bindings=2;
    multiple.bindings[1]=(VkPipeBinding){&independent,GFX_RES_IMG,GFX_STAGE_CS,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,3,0};
    multiple.native_image_seed_bindings.entries[1]=(RcNativeImageSeedBinding){.binding=3,.scalar=1,.access=RC_NATIVE_IMAGE_SEED_WRITE};
    multiple.native_image_seed_bindings.count=2;
    dispatch.info=&multiple;
    assert(vk_exec_dispatch_compat_storage_binding(&dispatch)==&multiple.bindings[0]);
    multiple.native_image_seed_bindings.entries[1].access=RC_NATIVE_IMAGE_SEED_READ;
    assert(!vk_exec_dispatch_compat_storage_binding(&dispatch));
    const GfxShaderRes readonly={.type=3,.slot=4,.count=1};
    multiple.bindings[1].res=&readonly;
    assert(vk_exec_dispatch_compat_storage_binding(&dispatch)==&multiple.bindings[0]);
    multiple.bindings[1].binding=2;
    assert(!vk_exec_dispatch_compat_storage_binding(&dispatch));
    dispatch.info=&info;

    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
    EffectImageCapture *adopted=selection.images;
    assert(selection.count==1 && adopted->independent==&storage_target_live && adopted->storage_seed==&storage_seed &&
           storage_captures==1 && storage_accesses==1 && storage_seeds==1 && !storage_releases);
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
    assert(selection.images==adopted && storage_captures==1 && storage_accesses==1 && storage_seeds==1 &&
           storage_answer_lookups==1);

    storage_seed_uploaded=0;
    assert(!dispatch_compat_inputs_transfer(&ex,&effect,dispatch.opcode,dispatch.payload,&dispatch));
    assert(!effect.images && !effect.image_count && !effect.compat_descriptors.cell && selection.images==adopted);
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!selection.images && !selection.count && !dispatch.descriptors.cell &&
           !storage_target_live && storage_releases==1 && storage_seed_cancels==1);
    storage_seed_uploaded=1;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
    adopted=selection.images;
    storage_target_current=0;
    assert(!dispatch_compat_inputs_transfer(&ex,&effect,dispatch.opcode,dispatch.payload,&dispatch));
    assert(!effect.images && selection.images==adopted);
    storage_target_current=1;
    DeviceStoragePointer *pointer=calloc(1,sizeof *pointer);
    assert(pointer);
    selection.pointer_sources=pointer;
    selection.pointer_count=1;
    assert(!dispatch_compat_inputs_transfer(&ex,&effect,dispatch.opcode,dispatch.payload,&dispatch));
    pointer->current=1;
    assert(dispatch_compat_inputs_transfer(&ex,&effect,dispatch.opcode,dispatch.payload,&dispatch));
    assert(effect.pointer_sources==pointer && effect.pointer_count==1 &&
           !selection.pointer_sources && !selection.pointer_count);
    device_storage_pointers_free(effect.pointer_sources,effect.pointer_count);
    effect.pointer_sources=NULL;effect.pointer_count=0;
    assert(effect.storage_admission==RESOURCE_IMAGE_TRACE_STORAGE_SELECTED);
    assert(effect.images==adopted && effect.image_count==1 && effect.compat_descriptors.cell &&
           effect.images[0].storage_seed==&storage_seed && !dispatch.descriptors.cell && !selection.count &&
           !selection.images && storage_target_live && storage_seed.live);
    assert(!dispatch_compat_inputs_transfer(&ex,&effect,dispatch.opcode,dispatch.payload,&dispatch));
    dispatch_compat_storage_release(&selection);
    assert(storage_releases==1 && storage_seed_cancels==1 && storage_target_live);
    VkExecDispatchImages owner={.count=effect.image_count,.images=effect.images};
    dispatch_compat_storage_release(&owner);
    draw_descriptor_grant_release(&effect.compat_descriptors);
    assert(!owner.images && !owner.count && !storage_target_live && !storage_seed.live &&
           storage_releases==2 && storage_seed_cancels==2);

    storage_access_result=VK_EXEC_GRAPHICS_UNAVAILABLE;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!strcmp(dispatch.input_failure.stage,"compat-storage-output") && storage_accesses==3 && storage_seeds==2 &&
           selection.count==1 && !dispatch.descriptors.cell && !storage_seed.live);
    dispatch_compat_storage_release(&selection);
    assert(storage_releases==3 && storage_seed_cancels==2);
    storage_access_result=VK_EXEC_GRAPHICS_READY;
    storage_seed_result=VK_EXEC_GRAPHICS_INVALIDATED;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!selection.images && !selection.count && !dispatch.descriptors.cell &&
           storage_accesses==4 && storage_seeds==3 && storage_releases==4 && storage_seed_cancels==2);
    storage_seed_result=VK_EXEC_GRAPHICS_READY;
    storage_capture_result=-1;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!selection.images && !selection.count && !storage_target_live && storage_accesses==4);
    storage_capture_result=0;
    storage_capture_defers=1;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!texture_deferred && !selection.images && !selection.count &&
           !strcmp(dispatch.input_failure.reason,"storage-target-deferred"));
    storage_capture_defers=0;
    storage_capture_result=VK_TEXTURE_STORAGE_CAPTURE_UNSUPPORTED;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!selection.images && !selection.count && !storage_target_live &&
           !strcmp(dispatch.input_failure.reason,"storage-target-material"));
    storage_capture_result=1;
    storage_planes_exact=0;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!selection.images && !selection.count && !storage_target_live && storage_releases==5 &&
           !strcmp(dispatch.input_failure.reason,"storage-target-view") && !dispatch.descriptors.cell);
    storage_planes_exact=1;
    storage_answers_match=0;
    unsigned answers_before=storage_answer_lookups;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!selection.images && !selection.count && !storage_target_live && storage_releases==6 &&
           storage_accesses==4 && storage_answer_lookups==answers_before+1 && !dispatch.descriptors.cell);
    dispatch.info=&multiple;
    multiple.native_image_seed_bindings.entries[1].access=RC_NATIVE_IMAGE_SEED_WRITE;
    multiple.bindings[1].binding=3;
    multiple.bindings[1].res=&independent;
    ex.img[NVN_STAGE_CS][4]=32;
    const unsigned before_pair=storage_captures;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
    assert(storage_captures==before_pair+2 && storage_releases==7 && !storage_target_live &&
           !selection.images && !selection.count && selection.storage_admission==RESOURCE_IMAGE_TRACE_STORAGE_TARGET);
    draw_descriptor_grant_release(&dispatch.descriptors);
    dispatch.info=&info;
    storage_answers_match=1;
    host_chosen=1;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
    assert(host_live && host_prepares==1 && !host_cancels && !storage_target_live &&
           selection.write_plan.count==1 && selection.images->write_target==&host_material &&
           selection.images->image.image==host_material.prepared);
    EffectDeclaration host_effect={.position={read.scene,read.ordinal+1}};
    assert(dispatch_compat_inputs_transfer(&ex,&host_effect,dispatch.opcode,dispatch.payload,&dispatch));
    assert(!selection.write_plan.count && !selection.images && host_effect.write_plan.count==1 && host_live);
    VkExecDispatchImages host_owner={.count=host_effect.image_count,.images=host_effect.images,.write_plan=host_effect.write_plan};
    dispatch_compat_storage_release(&host_owner);
    draw_descriptor_grant_release(&host_effect.compat_descriptors);
    assert(!host_live && host_cancels==1 && !host_owner.images && !host_owner.write_plan.count);

    // A real selected bound result carries its holder across a failed prefix, or reaches the ordinary
    // storage seed and descriptor with only the successfully relocated physical image.
    host_chosen=0;bound_chosen=1;
    bound_disposition=VK_EXEC_RESTORE_STALE;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(bound_prepares==1 && !selection.images && !selection.bound_storage && !dispatch.descriptors.cell);
    bound_pending=0;
    bound_disposition=VK_EXEC_RESTORE_READY;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
    assert(bound_prepares==2 && !bound_pending && selection.bound_storage && selection.count==1 &&
           selection.images->image.image==storage_shape.image+1 && dispatch.descriptors.cell);
    dispatch_compat_storage_release(&selection);
    draw_descriptor_grant_release(&dispatch.descriptors);
    assert(!selection.images && !selection.bound_storage && !storage_target_live);

    // Native opaque 2D collection must see the same selected result without recording a copy yet.
    dispatch.compatibility=0;
    selection.opaque_storage_effects=1;
    assert(dispatch_native_storage_inventory_prepare(&ex,&dispatch)==VK_EXEC_GRAPHICS_READY);
    assert(bound_prepares==2 && bound_pending && selection.bound_storage && !selection.opaque_storage_effects);
    assert(dispatch_bound_storage_prepare(&ex,&dispatch)==VK_EXEC_GRAPHICS_READY);
    assert(bound_prepares==3 && selection.images->image.image==storage_shape.image+1);
    dispatch_compat_storage_release(&selection);
    bound_chosen=0;selection.opaque_storage_effects=1;
    assert(dispatch_native_storage_inventory_prepare(&ex,&dispatch)==VK_EXEC_GRAPHICS_READY);
    assert(bound_prepares==3 && !selection.images && selection.opaque_storage_effects);
    // The same preflight can transfer both exact b1 peers and a later-realized COW target once.
    selection.opaque_storage_effects=0;
    dispatch.compatibility=1;
    dispatch.info=&with_samples;
    ex.tex[NVN_STAGE_CS][0]=16;
    ex.tex[NVN_STAGE_CS][1]=17;
    host_chosen=1;
    const int retained_reports=reports;
    VkExecGraphicsResult paired=vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL);
    if (paired!=VK_EXEC_GRAPHICS_READY)
        fprintf(stderr,"paired COW result=%d stage=%s reason=%s\n",
                paired,dispatch.input_failure.stage,dispatch.input_failure.reason);
    assert(paired==VK_EXEC_GRAPHICS_READY);
    assert(reports==retained_reports+2 && selection.samples && selection.samples->count==2 &&
           selection.count==1 && selection.images[0].write_target==&host_material && host_live);
    ComputeSamples *adopted_samples=selection.samples;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
    assert(reports==retained_reports+2 && selection.samples==adopted_samples);
    effect=(EffectDeclaration){.position={read.scene,read.ordinal+1}};
    assert(dispatch_compat_inputs_transfer(&ex,&effect,dispatch.opcode,dispatch.payload,&dispatch));
    assert(effect.compute_samples==adopted_samples && !selection.samples &&
           effect.image_count==1 && effect.images[0].write_target==&host_material);
    assert(!dispatch_compat_inputs_transfer(&ex,&effect,dispatch.opcode,dispatch.payload,&dispatch));
    compute_samples_release(&effect.compute_samples);
    VkExecDispatchImages adopted_owner={.count=effect.image_count,.images=effect.images,.write_plan=effect.write_plan};
    dispatch_compat_storage_release(&adopted_owner);
    draw_descriptor_grant_release(&effect.compat_descriptors);
    assert(!host_live && !raw_receipt.live && allocations==releases && retained==hold_releases);
    puts("Compatibility sampled peers: original-cut defer/retry, stale, refusal, cancellation and first COW transfer passed");
    puts("Compatibility storage: selected output and seed custody, stale transfer, paired rollback and cancellation passed");
    puts("Bound storage adoption: stale preissue rollback, newly selected descriptor, native delayed adoption and opaque compatibility passed");
}
#endif

int main(void) {
    const GfxShaderRes ordinary={.slot=4,.type=3}, compare={.slot=7,.type=98}, explicit={.slot=8};
    const VkPipeInfo info={.n_bindings=3,.bindings={{&ordinary,GFX_RES_TEX,GFX_STAGE_CS,100,3,1,0},
        {&compare,GFX_RES_TEX,GFX_STAGE_CS,100,1,2,0},{&explicit,GFX_RES_TEX,GFX_STAGE_CS,100,1,3,0}}};
    VkExecDispatchImages selection={.count=1,
        .images=(EffectImageCapture[]){{.resource=&explicit,.sampled=1,.handle=5}}};
    VkExecDispatch dispatch={.selection=&selection,.info=&info,.read_position=read};
    VkExec ex={.fence=read.scene,.tex={{[4]=16,[5]=17,[6]=15,[7]=99,[8]=5}}};
    __typeof__(*ex.dc) ctx={0};ex.dc=&ctx;
    all_current=1;
    const RcRegistryPosition none={0};
    assert(compute_samples_select(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
    // There are no bound UBOs. Ordinary adjacent members still select their actual provider or absence.
    assert(allocations==1 && reports==3 && selection.samples->count==3);
    const GraphicsSample *samples=selection.samples->entries;
    pending_raw=1;
    assert(!compute_samples_current(&ex,&info,selection.samples)); // A retained candidate cannot bind.
    assert(samples[0].alias==&raw_receipt && !samples[0].image.image && samples[0].member==0);
    assert(samples[1].hold==&resident && samples[1].image.image==77 && samples[1].member==1);
    assert(samples[2].sample.historical==VK_TEXTURE_REGISTRATION_ABSENT && samples[2].member==2);
    assert(compute_samples_select(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
    assert(reports==3 && allocations==1); // The exact pending candidate survives adjacent retained samples.
    pending_raw=0;
    read.ordinal+=2;
    assert(compute_samples_select(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_INVALIDATED && !selection.samples);
    read=dispatch.read_position;
    failure=1;
    assert(compute_samples_select(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE && !selection.samples);
    assert(allocations==2 && reports==6 && releases==2 && !lock_count);
    assert(!strcmp(dispatch.input_failure.reason, "provider-layout") && dispatch.input_failure.subtype==91 &&
        dispatch.input_failure.image==42 && dispatch.input_failure.address==512 && dispatch.input_failure.bytes==16);
    failure=0;aged=1;
    assert(compute_samples_select(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
    const GraphicsSample *old=&selection.samples->entries[1];
    assert(old->hold==&historical && old->sample.historical==VK_TEXTURE_REGISTRATION_EXACT &&
           old->sample.native_slot==-1 && old->sample.registration==70);
    compute_samples_release(&selection.samples);
#if HAS_COMPAT_STORAGE
    selection.images=NULL;
    selection.count=0;
    ex.tex[NVN_STAGE_CS][8]=0;
#endif
#if HAS_COMPAT
    const prog_t program={22};
    const GfxShader shader={1};
    dispatch.shader=&shader;
    dispatch.program=ex.prog=&program;
    dispatch.program_serial=program.serial;
    dispatch.registration=140;
    dispatch.pipeline=5;
    dispatch.opcode=26;
    dispatch.payload=&ordinary;
    ex.resource_op=read.ordinal/2;
    aged=0;
    int previous_reports=reports;
    skip_shader=1;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
    assert(reports==previous_reports && !selection.samples);
    skip_shader=0;
    stopped=1;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_STOPPED);
    assert(reports==previous_reports && !selection.samples);
    stopped=0;
    failure=1;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!selection.samples && !materialized && !queued);
    failure=0;
    remap_during_materialization=1;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!selection.samples && !materialized && !queued && !raw_receipt.live && alias_captures==alias_cancels);
    remap_during_materialization=0;
    all_current=1;
    previous_reports=reports;
    pending_raw=1;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
    assert(!pending_raw && materialized==1 && reports==previous_reports+3 && !queued);
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
    assert(materialized==1 && reports==previous_reports+3 && descriptor_acquires==1 && !descriptor_cancels);
    EffectDeclaration effect={.program=&program,.program_serial=program.serial,.opcode=dispatch.opcode};
    assert(!dispatch_compat_inputs_transfer(&ex,&effect,dispatch.opcode,&info,&dispatch));
    assert(vk_exec_dispatch_has_compat_inputs(&dispatch) && !effect.compute_samples);
    read.ordinal+=2;
    assert(!dispatch_compat_inputs_transfer(&ex,&effect,dispatch.opcode,dispatch.payload,&dispatch));
    assert(vk_exec_dispatch_has_compat_inputs(&dispatch) && !effect.compute_samples);
    read=dispatch.read_position;
    assert(dispatch_compat_inputs_transfer(&ex,&effect,dispatch.opcode,dispatch.payload,&dispatch));
    assert(!vk_exec_dispatch_has_compat_inputs(&dispatch) && effect.compute_samples &&
           effect.compat_descriptors.info==&info && effect.compat_descriptors.cell==1 && !dispatch.descriptors.cell);
    assert(!dispatch_compat_inputs_transfer(&ex,&effect,dispatch.opcode,dispatch.payload,&dispatch));
    const GfxShader *shader_bound=NULL;
    const VkPipeInfo *info_bound=NULL;
    VkPipeline pipeline=0;
    assert(dispatch_compat_pipeline(&ex,dispatch.opcode,&shader_bound,&info_bound,&pipeline)==0);
    effect_current=&effect;
    assert(dispatch_compat_pipeline(&ex,dispatch.opcode,&shader_bound,&info_bound,&pipeline)==1);
    assert(shader_bound==&shader && info_bound==&info && pipeline==dispatch.pipeline);
    assert(reports==previous_reports+3);
    VkTextureSample computed={0};
    VkTextureAliasRead *alias=NULL;
    const VkTexturePresentSource *hold=NULL;
    vk_texture_lock();
    assert(!compute_sample(&ex,0,&compare,99,&computed,&alias,&hold));
    assert(!compute_sample(&ex,0,&explicit,5,&computed,&alias,&hold));
    assert(compute_sample(&ex,0,&ordinary,16,&computed,&alias,&hold)==1);
    assert(computed.registration==77 && computed.image->image==assembled.image && alias==&raw_receipt && !hold);
    vk_texture_alias_read_use(alias,ex.fence);
    assert(compute_sample(&ex,1,&ordinary,17,&computed,&alias,&hold)==1);
    assert(computed.registration==78 && computed.image->image==image.image && !alias && hold==&resident);
    vk_texture_unlock();
    assert(reports==previous_reports+3 && queued==1 && !refused);
    all_current=0;
    assert(dispatch_compat_pipeline(&ex,dispatch.opcode,&shader_bound,&info_bound,&pipeline)==-1);
    assert(refused==1 && queued==1);
    compute_samples_release(&effect.compute_samples);
    assert(allocations==releases && !raw_receipt.live && alias_captures==alias_cancels);
    // A compatibility pipeline may have a retained original buffer with no sampled images.
    VkExecBufferScratchReceipt buffer = {.info = &info, .current = 1};
    dispatch.scratch = &buffer;
    draw_descriptor_grant_release(&effect.compat_descriptors);
    const VkPipeInfo no_samples = {0};
    dispatch.info = &no_samples;
    descriptor_refused = 1;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE &&
           !dispatch.descriptors.cell);
    assert(dispatch.input_failure.stage && !strcmp(dispatch.input_failure.stage,"descriptor-grant"));
    descriptor_refused = 0;
    dispatch.info = &info;
    assert(draw_descriptor_grant_prepare(&ex,&info,&dispatch.descriptors,1)==VK_EXEC_GRAPHICS_READY);
    effect = (EffectDeclaration){.program=&program, .program_serial=program.serial, .opcode=dispatch.opcode,
                                 .position={read.scene, read.ordinal + 1}};
    assert(vk_exec_dispatch_has_compat_inputs(&dispatch) && !selection.samples && !effect.buffer_scratch);
    buffer.current = 0;
    assert(!dispatch_compat_inputs_transfer(&ex,&effect,dispatch.opcode,dispatch.payload,&dispatch));
    assert(dispatch.scratch==&buffer && !effect.buffer_scratch);
    buffer.current = 1;
    assert(dispatch_compat_inputs_transfer(&ex,&effect,dispatch.opcode,dispatch.payload,&dispatch));
    assert(!dispatch.scratch && effect.buffer_scratch==&buffer && !effect.compute_samples &&
           !dispatch.descriptors.cell && effect.compat_descriptors.cell==1);
    assert(!dispatch_compat_inputs_transfer(&ex,&effect,dispatch.opcode,dispatch.payload,&dispatch));
    effect_current = &effect;
    const GfxShader *compat_shader = NULL;
    const VkPipeInfo *compat_info = NULL;
    VkPipeline compat_pipeline = 0;
    assert(dispatch_compat_pipeline(&ex,dispatch.opcode,&compat_shader,&compat_info,&compat_pipeline)==1 &&
           compat_shader==dispatch.shader && compat_info==dispatch.info && compat_pipeline==dispatch.pipeline);
    buffer.current = 0;
    int previous_refused = refused;
    assert(dispatch_compat_pipeline(&ex,dispatch.opcode,&compat_shader,&compat_info,&compat_pipeline)==-1 &&
           refused==previous_refused+1);
    effect_current = NULL;
    draw_descriptor_grant_release(&effect.compat_descriptors);
    assert(descriptor_acquires == 2 && descriptor_cancels == 2);
    puts("Compatibility compute: ordered descriptor placement follows the original shader and buffer/sampled owners");
#else
    fputs("Compatibility dispatch has no retained ordinary sample preparation or binding handoff\n", stderr);
    return 1;
#endif
    all_current=1;
    linear_control=0;
    const GfxShaderRes storage={0}, sampled={.slot=5,.exact=1};
    const VkPipeInfo storage_info={.n_bindings=1,.bindings={{&sampled,GFX_RES_TEX,GFX_STAGE_CS,100,1,4,0}}};
    const view_t plane={1,0,1,0,1};
    VkExecDispatchImages storage_selection={.count=1,.images=(EffectImageCapture[3]){{.resource=&storage,.written=1,.handle=12,
        .image={.image=88,.addr=0x800},.size=16,.has_view=1,.view=plane,.independent=&resident}}};
    dispatch.selection=&storage_selection;
    dispatch.info=&storage_info;
    exact_control=1;
    fallback=1;
    int retains_before=retained,releases_before=hold_releases;
    assert(compute_samples_select(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
    // FALLBACK carries the exact view the ordinary binder would use. Keep its native owner through issue.
    assert(storage_selection.samples && storage_selection.samples->count==1 && resolves==1 && retained==retains_before+1);
    const GraphicsSample *direct=&storage_selection.samples->entries[0];
    assert(direct->hold==&resident && direct->image.image==77 && direct->sample.view.level0==1);
#if HAS_STORAGE_SAMPLE_RECEIPT
    EffectImageCapture *writer=&storage_selection.images[0];
    assert(dispatch_storage_outputs_disjoint(&ex,&dispatch));
    writer->image.image=77; // A different mip of a retained image is physically disjoint.
    assert(dispatch_storage_outputs_disjoint(&ex,&dispatch));
    writer->view.level0=1; // Same image, same subresource is actual feedback.
    assert(!dispatch_storage_outputs_disjoint(&ex,&dispatch));
    assert(!strcmp(dispatch.input_failure.reason,"sample-view-conflict"));
    EffectImageCapture *effect_sample=&storage_selection.images[1];
    storage_selection.count=2;
    *effect_sample=(EffectImageCapture){.sampled=1,.resource=&sampled,.handle=17,.image={.image=99},
        .view=plane,.has_view=1,.read_hold=&historical};
    assert(dispatch_storage_outputs_disjoint(&ex,&dispatch)); // Binder gives the effect receipt precedence.
    effect_sample->read_hold=NULL;
    assert(!dispatch_storage_outputs_disjoint(&ex,&dispatch));
    assert(!strcmp(dispatch.input_failure.reason,"effect-sample-owner-missing"));
    effect_sample->write_target=&(VkTextureWriteTarget){.prepared=105};
    effect_sample->image.image=77; // Ignore the predecessor when the phase binds its new image.
    assert(dispatch_storage_outputs_disjoint(&ex,&dispatch));
    effect_current = &(EffectDeclaration){.compute_samples=storage_selection.samples,.opcode=27};
    vk_texture_lock();
    assert(compute_sample(&ex,0,&sampled,17,&computed,&alias,&hold)==1 && computed.image->image==77 &&
           computed.view.level0==1 && hold==&resident);
    all_current=0;
    retained_original=1;
    assert(compute_sample(&ex,0,&sampled,17,&computed,&alias,&hold)==1 && hold==&resident);
    retained_original=0;
    assert(compute_sample(&ex,0,&sampled,17,&computed,&alias,&hold)==-1);
    vk_texture_unlock();
    effect_current=NULL;
    storage_selection.count=1;
#endif
    compute_samples_release(&storage_selection.samples);
    assert(!storage_selection.samples && hold_releases==releases_before+1);
#if HAS_STORAGE_SAMPLE_RECEIPT
    assert(!dispatch_storage_outputs_disjoint(&ex,&dispatch));
    assert(!strcmp(dispatch.input_failure.reason,"sample-receipt-missing"));
    assert(compute_samples_select(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!storage_selection.samples && !strcmp(dispatch.input_failure.reason,"sample-binding-receipt-stale"));
    gone=1;
    assert(compute_samples_select(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!storage_selection.samples && dispatch.input_failure.subtype==VK_TEXTURE_SAMPLE_GONE);
    assert(allocations==releases && retained==hold_releases);
#endif
    // Same final direct binding proof serves compatibility and native, while the original overlaps output.
    GfxShaderRes query_resource={.slot=5,.type=3};
    VkPipeInfo query_info={.n_bindings=1,.bindings={{.res=&query_resource,.kind=GFX_RES_TEX,
        .stage=GFX_STAGE_CS,.type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,.count=1,.binding=2}},
        .compute_query_size_lod0_mask=UINT64_C(1)<<2};
    all_current=1;gone=0;fallback=0;linear_control=1;exact_control=0;
    ex.cb=55;
    dispatch.info=&query_info;
    dispatch.selection=&storage_selection;
    const int previous_texel_reports=reports;
    for (int native=0;native<2;native++) {
        assert(draw_descriptor_grant_prepare(&ex,&query_info,&dispatch.descriptors,1)==VK_EXEC_GRAPHICS_READY);
        assert(compute_samples_select(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
        ComputeSamples *query_samples=storage_selection.samples;
        assert(query_samples && query_samples->count==1 && query_samples->entries[0].geometry.registration==78);
        assert(query_samples->entries[0].sample.query_image==&query_carrier && reports==previous_texel_reports);
        storage_selection.images[0].image.image=77;
        assert(dispatch_storage_outputs_disjoint(&ex,&dispatch));
        GraphicsSample *query=&query_samples->entries[0];
        const GraphicsSample resident_query=*query;
        query->image.image=0;
        query->hold=NULL;
        assert(!dispatch_storage_outputs_disjoint(&ex,&dispatch));
        query->geometry.registration_only=1;
        assert(dispatch_storage_outputs_disjoint(&ex,&dispatch));
        *query=resident_query;
        query_info.compute_query_size_lod0_mask=0;
        assert(!dispatch_storage_outputs_disjoint(&ex,&dispatch));
        assert(!strcmp(dispatch.input_failure.reason,"sample-query-proof"));
        query_info.compute_query_size_lod0_mask=UINT64_C(1)<<2;
        // Prepare never records the carrier. The actual binding entry records once before descriptor use.
        assert(!query_carrier.recorded);
        EffectDeclaration query_effect={.program=dispatch.program,.program_serial=dispatch.program_serial,
            .opcode=dispatch.opcode,.position={read.scene,read.ordinal+1}};
        if(native) {
            query_effect.compute_samples=query_samples;
            storage_selection.samples=NULL;
            dispatch_current=&dispatch;
        } else {
#if HAS_COMPAT_STORAGE
            // This compatibility query has no storage output. The synthetic peer above is only for disjointness.
            VkExecDispatchImages query_selection={.samples=storage_selection.samples};
            storage_selection.samples=NULL;
            dispatch.selection=&query_selection;
#endif
            assert(dispatch_compat_inputs_transfer(&ex,&query_effect,dispatch.opcode,dispatch.payload,&dispatch));
#if HAS_COMPAT_STORAGE
            dispatch.selection=&storage_selection;
#endif
        }
        effect_current=&query_effect;
        assert(dispatch_bind_selected_resources(&ex,dispatch.opcode,dispatch.shader,&query_info));
        assert(dispatch_bind_selected_resources(&ex,dispatch.opcode,dispatch.shader,&query_info));
        assert(query_records==native+1);
        compute_samples_release(&query_effect.compute_samples);
        draw_descriptor_grant_release(native ? &dispatch.descriptors : &query_effect.compat_descriptors);
        effect_current=NULL;
        dispatch_current=NULL;
    }
    assert(geometry_checks==2 && query_creates==2 && query_releases==2 && query_bound==4);
    assert(compute_samples_select(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
    query_current=0;
    assert(compute_samples_select(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!storage_selection.samples && query_creates==3 && query_releases==3 && query_records==2);
    query_current=1;

    query_info.bindings[0].set=1;
    assert(!compute_query_geometry_binding(&ex,&query_info,&query_info.bindings[0]));
    query_info.bindings[0].set=0;
    query_info.bindings[0].count=2;
    assert(!compute_query_geometry_binding(&ex,&query_info,&query_info.bindings[0]));
    query_info.bindings[0].count=1;
    query_resource.separate=1;
    assert(!compute_query_geometry_binding(&ex,&query_info,&query_info.bindings[0]));
    query_resource.separate=0;
    ex.dc->mo_ovr_on=1;
    assert(!compute_query_geometry_binding(&ex,&query_info,&query_info.bindings[0]));
    ex.dc->mo_ovr_on=0;
    storage_selection.images[1]=(EffectImageCapture){.resource=&query_resource,.sampled=1,.handle=17};
    storage_selection.count=2;
    assert(compute_samples_select(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
    assert(!storage_selection.samples && query_creates==3);
    storage_selection.count=1;
    carrier_fails=1;
    assert(compute_samples_select(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
    assert(storage_selection.samples && !storage_selection.samples->entries[0].sample.query_image);
    assert(query_creates==3 && reports==previous_texel_reports+1);
    compute_samples_release(&storage_selection.samples);
    carrier_fails=0;
    selected_redirect=1;
    assert(compute_samples_select(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
    assert(storage_selection.samples && storage_selection.samples->entries[0].sample.query_image==&query_carrier &&
           storage_selection.samples->entries[0].geometry.association_only && reports==previous_texel_reports+2);
    assert(dispatch_storage_outputs_disjoint(&ex,&dispatch));
    compute_samples_release(&storage_selection.samples);
    assert(!query_carrier.live && query_creates==4 && query_releases==4);
    selected_redirect=0;
    historical_query=1;
    assert(compute_samples_select(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
    assert(storage_selection.samples);
    const GraphicsSample *historical_shape=&storage_selection.samples->entries[0];
    assert(historical_shape->sample.query_image==&query_carrier && historical_shape->geometry.history_capture==123 &&
           !historical_shape->sample.image && !historical_shape->hold && reports==previous_texel_reports+2);
    assert(dispatch_storage_outputs_disjoint(&ex,&dispatch));
    compute_samples_release(&storage_selection.samples);
    assert(!query_carrier.live && query_creates==5 && query_releases==5);
#if HAS_STORAGE_SAMPLE_RECEIPT
    // Same shader binding/member as the recorded compatibility stop. Verify its ordinary
    // binder consumes the original held source and view through the dispatch receipt.
    const GfxShaderRes depth_compare={.slot=0,.count=1,.type=0x503};
    const VkPipeInfo compare_info={.n_bindings=1,.bindings={{.res=&depth_compare,.kind=GFX_RES_TEX,
        .stage=GFX_STAGE_CS,.type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,.count=1,.binding=5}}};
    comparison_select=1;
    linear_control=0;
    historical_query=0;
    ex.tex[NVN_STAGE_CS][0]=17;
    dispatch.info=&compare_info;
    const int comparison_reports=reports;
    assert(compute_samples_select(&ex,&dispatch,none,NULL)==VK_EXEC_GRAPHICS_READY);
    assert(storage_selection.samples&&storage_selection.samples->count==1&&reports==comparison_reports+1);
    assert(dispatch_storage_outputs_disjoint(&ex,&dispatch));
    effect_current=&(EffectDeclaration){.compute_samples=storage_selection.samples,.opcode=OP_DISPATCH};
    vk_texture_lock();
    assert(compute_sample(&ex,0,&depth_compare,17,&computed,&alias,&hold)==1&&
           computed.image->image==comparison_image.image&&computed.registration==190&&
           computed.native_slot==106&&computed.view.aspect==2&&computed.view.levels==1&&
           computed.view.layers==3&&!alias&&hold==&resident);
    vk_texture_unlock();
    assert(reports==comparison_reports+1);
    GraphicsSample *held_comparison=&storage_selection.samples->entries[0];
    held_comparison->sample.registration=0; // Legacy GPU fallback left only the original slot/token/hold.
    held_comparison->sample.projected=0;
    assert(dispatch_storage_outputs_disjoint(&ex,&dispatch));
    held_comparison->sample.native_slot++;
    assert(!dispatch_storage_outputs_disjoint(&ex,&dispatch));
    assert(!strcmp(dispatch.input_failure.reason,"sample-comparison-mapping"));
    compute_samples_release(&storage_selection.samples);
    comparison_select=0;
    linear_control=1;
    effect_current=NULL;
    puts("Direct compute comparison: b5 retained depth and original view reached the ordinary binder");
#endif
    assert(allocations==releases && retained==hold_releases);

#if HAS_COMPAT_STORAGE
    check_compat_storage();
#endif
    puts("Direct compute query: final binding proof, compatibility and native issue, storage peers, remap and rollback passed");
    puts("Direct compute: raw receipt, current and historical custody, absence, specialty, retry and refusal passed");
}
'''

TRACE_STUBS = r"""
static void *dispatch_trace_begin(VkExecDispatch *dispatch,const char *reason,const char *detail,
                                  const EffectImageCapture *output) {
    (void)dispatch; (void)reason; (void)detail; (void)output; return NULL;
}
static void dispatch_storage_target_trace(VkExecDispatch *dispatch,const EffectImageCapture *capture,
    const VkPipeBinding *binding,const char *reason,const char *detail) {
    (void)dispatch; (void)capture; (void)binding; (void)reason; (void)detail;
}
static int dispatch_storage_conflict(VkExecDispatch *,const EffectImageCapture *,const char *,VkImage,int,int);
static int dispatch_storage_input_conflict(VkExecDispatch *dispatch, const EffectImageCapture *output,
    const char *reason, const char *detail, const VkPipeBinding *binding, int member, uint64_t handle, VkImage peer,
    const EffectImageCapture *effect, const GraphicsSample *sample, const VkTexImage *tested_image) {
    (void)detail; (void)handle; (void)effect; (void)sample; (void)tested_image;
    return dispatch_storage_conflict(dispatch,output,reason,peer,binding->binding,member);
}
static int dispatch_storage_peer_conflict(VkExecDispatch *dispatch, const EffectImageCapture *output,
    const EffectImageCapture *peer, const char *reason) {
    return dispatch_storage_conflict(dispatch,output,reason,peer->image.image,-1,-1);
}
"""

names = ("graphics_sample_selection_current", "compute_samples_current_locked", "graphics_sample_release", "compute_samples_release",
         "sample_associated_geometry_capture_locked", "sample_redirect_query_carrier_locked")
has_storage_receipt = "dispatch_storage_sample_view_disjoint(" in DRAW
if has_storage_receipt:
    names += ("effect_image_lookup", "compute_sample_receipt", "dispatch_storage_conflict",
              "dispatch_storage_sample_view_disjoint", "compute_sample_selection_failure")
if "dispatch_compat_samples_required(" in DRAW:
    PREFIX += "const VkPipeBinding *vk_exec_dispatch_compat_storage_binding(const VkExecDispatch *);\n"
    required = ("dispatch_compat_samples_required",)
else:
    required = ()
names += ("sample_query_record_outside_rendering", "sample_query_images_record", "compute_query_overrides_inactive",
          "dispatch_query_samples_record", "dispatch_bind_selected_resources", "compute_sample_members",
          "compute_query_geometry_binding", "compute_sample_selected") + required + ("compute_samples_select",)
if has_storage_receipt:
    PREFIX += "static int vk_texture_storage_targets_disjoint(const void *a,const void *b){return a&&b&&a!=b;}\n"
    PREFIX += r"""
static int vk_texture_storage_target_disjoint_input(const void *target,const RcInterval *intervals,
    uint32_t count,VkImageMemoryPlacement input_placement) {
    return target&&intervals&&count&&input_placement.memory!=(uintptr_t)target;
}
static int twin_format(const VkTexImage *image){(void)image;return 0;}
static int pool_comparison_format(const ViewKey *view,uint64_t *features){(void)view;*features=0;return 1;}
static const RcInterval *vk_texture_alias_read_intervals(const VkTextureAliasRead *alias,uint32_t *count){
    static const RcInterval mapped = {.address=0x1100};
    *count=alias==&raw_receipt && raw_receipt.live;
    return *count ? &mapped : NULL;
}
static const RcInterval *vk_texture_selected_sample_intervals_locked(VkTextureReadKey key,VkImage selected,uint32_t *count){
    static const RcInterval mapped = {.address=0x2100};
    *count=(key.registration==78 && selected==image.image) ||
           (comparison_select && key.registration==190 && key.native_slot==106 && key.projected &&
            selected==comparison_image.image);
    return *count ? &mapped : NULL;
}
static const RcInterval *vk_texture_native_sample_intervals_locked(RcRegistryPosition at,
    const VkTextureSample *sample,const void *held,VkImage expected,VkImageMemoryPlacement *placement,uint32_t *count) {
    static const RcInterval extent={.address=UINT64_C(0x800193f800)};
    assert(lock_count&&sample&&held);
    *placement=(VkImageMemoryPlacement){0};
    *count=comparison_select&&!rc_registry_position_compare(at,read)&&
        !sample->registration&&sample->native_slot==106&&expected==comparison_image.image;
    return *count?&extent:NULL;
}
static int vk_texture_write_target_disjoint_placement(const VkTextureWriteTarget *target,VkImageMemoryPlacement placement){
    assert(target);
    return !placement.memory;
}
static int vk_texture_write_disjoint_input(const VkTextureWriteTarget *target,const void *pin,const RcInterval *intervals,uint32_t count){
    assert(target&&!pin);return intervals&&count;
}
"""
    if "dispatch_storage_mirror_buffers_disjoint(" in DRAW:
        PREFIX += r"""
static int vk_texture_storage_target_host_mirror(const void *target) {(void)target; return 0;}
static int dispatch_storage_mirror_input_disjoint(const void *target,const RcInterval *intervals,
    uint32_t count,VkImageMemoryPlacement input_placement) {
    (void)input_placement;
    (void)target; (void)intervals; (void)count; assert(0); return 0;
}
static int dispatch_storage_mirror_buffers_disjoint(const VkExec *ex,VkExecDispatch *dispatch,const EffectImageCapture *output) {
    (void)ex; (void)dispatch; (void)output; assert(0); return 0;
}
"""
    PREFIX += TRACE_STUBS
    names += ("dispatch_storage_material_sample_disjoint",)
    PREFIX += r"""
static int dispatch_preserving_storage_inputs(const VkExec *ex,VkExecDispatch *dispatch,const EffectImageCapture *output) {
    (void)ex; (void)dispatch; (void)output; assert(0); return 0;
}
"""
    names += ("dispatch_storage_output_disjoint", "dispatch_storage_outputs_disjoint")
has_compat = "vk_exec_dispatch_prepare_compat_inputs(" in DRAW
has_compat_storage = "vk_exec_dispatch_compat_storage_binding(" in DRAW
if has_compat_storage:
    names += ("dispatch_storage_seed_binding", "compute_storage_binding_unused", "dispatch_unbound_storage_writer",
              "dispatch_storage_volume_required", "dispatch_storage_rmw_binding", "dispatch_storage_indirect_rmw", "dispatch_storage_binding_shape", "dispatch_storage_binding_proof",
              "dispatch_storage_seed_current",
              "dispatch_compat_storage_release", "dispatch_storage_image_access", "dispatch_compat_storage_inventory",
              "dispatch_native_storage_volume_pending", "vk_exec_dispatch_compat_storage_binding",
              "dispatch_compat_storage_answers_current", "dispatch_compat_storage_current",
              "dispatch_storage_read_peer_prepare", "effect_write_plan_add", "dispatch_storage_inventory_capture",
              "dispatch_bound_storage_prepare", "dispatch_compat_storage_prepare", "dispatch_native_storage_inventory_prepare")
if has_compat:
    names += ("compute_samples_current", "compute_sample", "dispatch_inputs_match", "compute_aliases_current",
              "compute_aliases_prepare", "vk_exec_dispatch_has_compat_inputs", "dispatch_compat_read_current",
              "dispatch_compat_inputs_transfer", "vk_exec_dispatch_prepare_compat_inputs", "dispatch_compat_pipeline")
    PREFIX += """typedef struct { const VkExec *ex; const ComputeSamples *samples; } ComputeSamplesRead;\n"""
    replay = Path(os.environ.get("DIRECT_COMPUTE_REPLAY_SOURCE", TESTS.parent / "runtime/vk_exec.c")).read_text(encoding="utf-8")
    prepare = function(replay, "prepare_ordered_dispatch")
    assert prepare.index("rc_registry_declare_through(rc_registry_runtime(), input)") < prepare.index(
        "vk_exec_dispatch_compat_inputs_select(") < prepare.index("prepare_dispatch_inputs(ex, h, dispatch, input, attempted, 1)")
    assert "compatibility ? vk_exec_dispatch_prepare_compat_inputs(" in function(replay, "prepare_dispatch_inputs")
    assert prepare.count("prepare_dispatch_scratch(ex, dispatch, input)") == 2
    assert "compute && vk_exec_dispatch_has_compat_inputs(&dispatch) ? &dispatch : NULL" in replay
    declare = function(replay, "resource_declare")
    assert declare.index("}, compat_dispatch)") < declare.index("resource_seal(ex)")
    effects = function(DRAW, "vk_exec_effects_declare")
    assert "dispatch_compat_inputs_transfer(ex, d, opcode, payload, compat_dispatch)" in effects

    if has_compat_storage:
        stage = function(DRAW, "effect_stage")
        assert stage.index("d->compat_compute_shader && d->image_count") < stage.index("effect_stage_image_inputs(")
        retained = stage[stage.index("d->compat_compute_shader && d->image_count"):stage.index("if(!present)")]
        assert "effect_stage_image_views(d, ex, 0, 1)" in retained
        assert "effect_stage_publish(d, ex, st, op, payload, shader, 0)" in retained
        assert "effect_images_truncate(d,0)" in function(DRAW, "vk_exec_effects_release")
        abandoned = function(DRAW, "vk_exec_dispatch_effect_cancel")
        assert "vk_texture_alias_read_cancel(receipt->images[i].storage_seed)" in abandoned
        assert "vk_texture_storage_target_release(receipt->images[i].independent)" in abandoned
    for op in ("op_dispatch", "op_dispatch_ind"):
        assert "dispatch_compat_pipeline(" in function(DRAW, op)
    assert "vk_texture_alias_read_use(alias, ex->fence)" in function(DRAW, "tex_info_full_sample")
    release = function(DRAW, "vk_exec_effects_release")
    assert "compute_samples_release(&d->compute_samples)" in release
    assert "draw_descriptor_grant_release(&d->compat_descriptors)" in release
    assert "&effect_current->compat_descriptors" in function(DRAW, "dispatch_bind_selected_resources")
if "dispatch_unbound_storage_writer(" in DRAW:
    i = names.index("dispatch_bind_selected_resources")
    dependencies = ("dispatch_storage_seed_binding", "compute_storage_binding_unused", "dispatch_unbound_storage_writer")
    names = names[:i] + dependencies + tuple(name for name in names[i:] if name not in dependencies)
source = (f"#define PREVIOUS_UNIFORM_GATE {int('dispatch_needs_sample_cohort(' in DRAW)}\n"
          f"#define HAS_COMPAT {int(has_compat)}\n"
          f"#define HAS_COMPAT_STORAGE {int(has_compat_storage)}\n"
          f"#define HAS_STORAGE_SAMPLE_RECEIPT {int(has_storage_receipt)}\n"
          f"#define HAS_HELD_STORAGE_LOAD {int('vk_texture_storage_target_load_current(' in DRAW)}\n"
          f"#define HAS_INPUT_REPORT {int('dispatch_record_input_stage(' in DRAW)}\n"
          + PREFIX + "\n".join(function(DRAW, name) for name in names) + CASES)
cc = Path(os.environ["CC"])
env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ["PATH"])
with tempfile.TemporaryDirectory(prefix="direct-compute-sample-") as temporary:
    c = Path(temporary) / "check.c"
    exe = c.with_suffix(".exe")
    c.write_text(source, encoding="utf-8")
    subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                    "-fsanitize-trap=all", str(c), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

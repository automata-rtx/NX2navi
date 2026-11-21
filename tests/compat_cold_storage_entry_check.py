"""Exercise the normal compatibility entry with cold registered images, including a true 3D storage mip."""
import os
from pathlib import Path
import runpy
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
runtime = Path(os.environ.get("COLD_STORAGE_SOURCE", ROOT / "runtime"))
headers = Path(os.environ.get("COLD_STORAGE_HEADERS", runtime))
function = runpy.run_path(str(ROOT / "tests/ordered_graphics_feedback_check.py"))["function"]
draw = (runtime / "vk_exec_draw.c").read_text(encoding="utf-8")
texture = (runtime / "vk_texture.c").read_text(encoding="utf-8")

PREFIX = r'''

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "resource_command_effects.h"
#include "rc_native_image_store.h"
typedef uint64_t VkImage,VkCommandBuffer;
typedef struct {uint64_t memory,offset,size;} VkImageMemoryPlacement;
typedef uint32_t VkImageUsageFlags,VkFormat;
typedef struct {
    int type,skip;VkImage image;uint64_t addr;uint32_t usage,aspect,format;
    uint32_t view_type,width,height,depth,levels,layers,samples,scale,fb_learned;
    int mirrored,gone,bound,block_views,gpu_owned,preserve_pool_bytes;
} VkTexImage;
typedef struct {
    uint32_t aspect,level0,levels,layer0,layers,set_layers,set_target,target;
    uint32_t set_dsmode,set_format,format,set_levels,base_level,min_layer,pad1;
} view_t;
typedef struct {uint32_t aspectMask,baseMipLevel,levelCount,baseArrayLayer,layerCount;} VkImageSubresourceRange;
typedef struct {uint64_t serial;} prog_t;
typedef struct {int unused;} GfxShader, VkExecGpaSampleScope;
typedef struct {uint64_t address,bytes,alignment;} UniformAddressRange;
typedef struct {uint64_t addr,size;} p_bind_buf;
typedef struct VkTextureWriteTarget VkTextureWriteTarget;
typedef struct {VkTextureWriteTarget **targets;uint32_t count;} VkTextureWritePlan;
typedef struct {int slot,flags,count,pool,type,exact,ptr;uint32_t binding,set;} GfxShaderRes;
typedef struct {const GfxShaderRes *res;int kind,stage,type;uint32_t count,binding,set;} VkPipeBinding;
typedef enum {VK_EXEC_GRAPHICS_READY,VK_EXEC_GRAPHICS_INVALIDATED,VK_EXEC_GRAPHICS_UNAVAILABLE,
              VK_EXEC_GRAPHICS_STOPPED} VkExecGraphicsResult;
enum { VK_PIPELINE_BIND_POINT_COMPUTE=0, GFX_RES_TEX=0, GFX_RES_IMG, GFX_RES_UBO, GFX_RES_SSBO, NVN_STAGE_CS=0, GFX_STAGE_CS=0,
       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE=3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
       VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, OP_DISPATCH=27,OP_DISPATCH_IND,
       VK_IMAGE_TYPE_2D=1,VK_IMAGE_TYPE_3D,VK_IMAGE_VIEW_TYPE_2D,VK_IMAGE_VIEW_TYPE_3D,VK_IMAGE_VIEW_TYPE_2D_ARRAY,
       NVN_TEXTURE_TARGET_3D,VK_FORMAT_UNDEFINED=0,VK_FORMAT_R8_UNORM=9,VK_AA_GAME=0,VK_AA_ROLE_NONE=0,RC_EFFECT_WRITES_IMAGE=4,RC_EFFECT_WRITES_UNKNOWN=128,
       VK_TEXTURE_REGISTRATION_ABSENT=3,VK_TEXTURE_REGISTRATION_EXACT,VK_FORMAT_R32_UINT=98,
       NVN_TEXTURE_TARGET_2D=1,NVN_FORMAT_R32UI=11,NVN_FORMAT_R11G11B10F=0x3f,NVN_FORMAT_RGB9E5F=0x40,
       VK_FORMAT_B10G11R11_UFLOAT_PACK32=122,VK_FORMAT_E5B9G9R9_UFLOAT_PACK32=123,
       VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT=8 };
#define NVN_HANDLE_TEX(handle) ((int)(handle))
#define VK_NULL_HANDLE 0
typedef struct {uint32_t n_bindings;VkPipeBinding bindings[5];int pool,pipeline,bind_point;
                RcNativeImageStore native_image_store;RcNativeImageSeedBindings native_image_seed_bindings,native_image_rmw_seed_bindings;} VkPipeInfo;
typedef struct {int x,y,z;} p_dispatch;
typedef struct {int live;} VkTextureAliasRead;
typedef struct {uint64_t imageView;int imageLayout;} VkDescriptorImageInfo;
typedef struct {uint64_t registration;int native_slot,canonical,projected;} VkTextureReadKey;
typedef struct VkTextureStorageTarget VkTextureStorageTarget;
typedef struct TextureStorageCpuLoad TextureStorageCpuLoad;
typedef struct VkTextureBoundRelocation VkTextureBoundRelocation;
typedef struct {
    const GfxShaderRes *resource;uint64_t handle,size;
    int stage,slot,written,has_view,sampled,full_overwrite,phase_seen,owns_write,preserve_native;
    VkTexImage image;view_t view;VkDescriptorImageInfo descriptor;
    VkTextureStorageTarget *independent;VkTextureAliasRead *storage_seed;
    VkTextureWriteTarget *write_target;const void *read_hold;VkImageSubresourceRange issued_planes;
} EffectImageCapture;
typedef struct {VkImage image;uint32_t aspect,level0,levels,layer0,layers,type,format;} ViewKey;
typedef struct {int historical,has_view,native_slot,canonical,projected;uint64_t registration;void *query_image;view_t view;} VkTextureSample;
typedef struct {
    VkTexImage image; uint64_t handle; const void *hold; VkTextureAliasRead *alias;
    VkTextureSample sample;
    struct {uint64_t registration,history_capture;VkImage image;int association_only,registration_only;} geometry;
} GraphicsSample;
typedef struct {GraphicsSample one;RcRegistryPosition read;const VkPipeInfo *info;} ComputeSamples;
typedef struct {
    uint32_t count,pointer_count;void *pointer_sources;int opaque_storage_effects,bound_storage,host_mirror_storage;
    EffectImageCapture *images;ComputeSamples *samples;ResourceImageTraceAdmission storage_admission;
    VkTextureWritePlan write_plan;
} VkExecDispatchImages;
typedef struct {int live;} DrawDescriptorGrant;
typedef struct {
    const char *stage,*reason;
    int binding,slot,member,subtype;
    uint64_t handle,image,peer_image,address,bytes;
    struct {const char *gate;} trace;
} VkExecDispatchInputFailure;
static int dispatch_storage_seed_current(const EffectImageCapture *,int uploaded);
typedef struct {
    const GfxShader *shader;const prog_t *program;uint64_t program_serial,registration;
    uint32_t opcode;const void *payload;RcRegistryPosition read_position;VkExecDispatchImages *selection;
    int compatibility,pipeline;const VkPipeInfo *info;DrawDescriptorGrant descriptors;
    VkExecDispatchInputFailure input_failure;
} VkExecDispatch;
typedef struct {
    const GfxShaderRes *resource;
    uint64_t handle;
    VkImage image;
    VkDescriptorImageInfo descriptor;
} BindingImageSelection;
typedef struct {unsigned n_storage_images;BindingImageSelection storage_images[4];} DrawCtx;
typedef struct {DrawCtx *dc;const prog_t *prog;uint64_t fence,cb,img[1][32],tex[1][128];int bar_dirty;} VkExec;
typedef struct {uint32_t image_count;EffectImageCapture *images;} EffectDeclaration;
static EffectDeclaration *effect_current;
enum {VK_IMAGE_LAYOUT_GENERAL=1};
static int dispatch_storage_binding_proof(const VkPipeInfo *,const VkPipeBinding *,int,RcNativeImageStore *);

static const GfxShader shader;
static const GfxShaderRes storage_bindings[]={
    {.slot=3,.flags=4,.count=1,.type=3,.binding=3},{.slot=4,.count=1,.type=3,.binding=4}
};
#define storage_binding storage_bindings[0]
#define readonly_binding storage_bindings[1]
static const GfxShaderRes volume_binding={.slot=3,.flags=4,.count=1,.type=4};
static const GfxShaderRes sample_binding={.slot=8,.count=1,.type=3};
static const GfxShaderRes comparison_binding={.slot=0,.count=1,.type=0x503};
static const uint64_t comparison_handle=0x1a6028f9;
static RcRegistryPosition read={9,22};
static int cold_normal_entry=1,lock_count,conflicting_peer, volume_view_override;
static uint32_t storage_shape = VK_IMAGE_VIEW_TYPE_2D;
static const struct VkTestCtx {struct {struct {uint32_t maxComputeWorkGroupCount[3];} limits;} props;}
    test_ctx = {.props.limits.maxComputeWorkGroupCount={65535,65535,65535}};
static const struct VkTestCtx *vk_ctx(void) { return &test_ctx; }
static const VkImage storage_original=309, storage_alternate=310, readonly_original=408;
static int readonly_enabled,readonly_initialized,readonly_format_ok=1,readonly_registered,readonly_resolve_deferred;
static unsigned cold_seeds, sample_releases;
static DrawCtx state, *dc=&state;
static struct {int ubo_max;uint64_t sbo_max;} LIM={1,4096};
static void *vk_dlaa_swap_img;
static VkImage vk_dlaa_swap_output;
static uint64_t vk_dlaa_swap_view;
static unsigned issue_refusals, writable_uses, writable_publications, hud_write_calls;
static struct {int robust_image_access;} vk_caps={1};
static void vk_texture_lock(void){lock_count++;}
static void vk_texture_unlock(void){assert(lock_count-- > 0);}
static RcRegistryPosition vk_exec_resource_position(void){return read;}
static uint64_t vk_texture_registration_publication(void){return 140;}
static int tex_deferred, cold_refused, cold_absent, cold_historical, cold_redirect;
static int vk_texture_deferred(void){return tex_deferred;}
int vk_texture_deferred_take(void) {
    const int deferred=tex_deferred;
    tex_deferred=0;
    return deferred;
}
static int native_tensor(const VkExec *ex){return ex->prog!=NULL;}
static int vk_aa_mode(void){return VK_AA_GAME;}
static uint64_t cs_guest_hash(const prog_t *program){return program->serial;}
static int vk_aa_role(uint64_t hash){(void)hash;return VK_AA_ROLE_NONE;}
static const GfxShader *shader_for(VkExec *ex,int stage,int *present,int *pending,unsigned wait){
    assert(ex->prog&&stage==NVN_STAGE_CS&&present&&!pending&&!wait);*present=1;return &shader;
}
static int shader_skipped(const GfxShader *selected){assert(selected==&shader);return 0;}
static int memoized_rmw_binding=-1, no_sample_reflection;
static unsigned memo_queries, buffer_preparations;
static int gfx_shader_indirect_image_rmw_binding(const GfxShader *selected) {
    assert(selected==&shader);
    memo_queries++;
    return memoized_rmw_binding;
}
static int gfx_shader_resources(const GfxShader *selected,int kind,const GfxShaderRes **resources){
    assert(selected==&shader);
    *resources=kind==GFX_RES_TEX&&!no_sample_reflection?&sample_binding:
        kind==GFX_RES_IMG?(storage_shape==VK_IMAGE_VIEW_TYPE_2D?&storage_binding:&volume_binding):NULL;
    return kind==GFX_RES_IMG ? 1+readonly_enabled : *resources!=NULL;
}
static int twin_compare_resource(const GfxShaderRes *r){return r->type==0x503;}
static VkFormat twin_format(const VkTexImage *image){return image->aspect==1?124:0;}
static int pool_comparison_format(const ViewKey *view,uint64_t *features){
    *features=0;
    return view->format==124&&view->aspect==2&&view->type==VK_IMAGE_VIEW_TYPE_2D_ARRAY&&
        view->level0==0&&view->levels==1&&view->layer0==0&&view->layers==3;
}
static int dispatch_indirect_requires_image(const void *payload,RcRegistryPosition at){(void)payload;(void)at;return 0;}
static void limits_init(void){LIM.ubo_max=1;}
static RcInterval rmw_input_mapping;
static int uniform_binding_range(const VkExec *ex,const VkPipeBinding *binding,int stage,unsigned member,UniformAddressRange *source) {
    (void)ex;(void)stage;(void)member;
    if (binding->kind != GFX_RES_SSBO || binding->binding != 4 || !rmw_input_mapping.address) return 0;
    *source=(UniformAddressRange){rmw_input_mapping.address,rmw_input_mapping.range.size,4};
    return 1;
}
static int uniform_range_requires_order(const UniformAddressRange *source,RcRegistryPosition at){(void)source;(void)at;return 0;}
static int readonly_storage_pointer(const GfxShaderRes *r){(void)r;return 0;}
static int readonly_storage_uniform_range(const VkExec *ex,int stage,const GfxShaderRes *r,UniformAddressRange *source) {
    (void)ex;(void)stage;(void)r;(void)source;return 0;
}
static const p_bind_buf *raw_storage_fixed(const VkExec *ex,const VkPipeBinding *binding){(void)ex;(void)binding;return NULL;}
static int raw_storage_requires_image(const p_bind_buf *bound,RcRegistryPosition at){(void)bound;(void)at;return 0;}
static void rt_fatal(const char *message){fputs(message,stderr);abort();}
static void rt_log(const char *format,...){assert(format);}
static int host_material, rmw_test, rmw_atomic_feature;
static VkImage rmw_predecessor;
static unsigned gfx_shader_guest_writes(const GfxShader *selected){assert(selected==&shader);return host_material||rmw_test?RC_EFFECT_WRITES_IMAGE:0;}
static int gfx_shader_write_inventory_complete(const GfxShader *selected){assert(selected==&shader);return 1;}
static int vk_sync_stopping(void){return 0;}
static int vk_device_is_lost(void){return 0;}
static int vk_exec_recording_failed(void){return 0;}
static int pool_on(void){return 0;}
static VkExecGraphicsResult dispatch_record_input_stage(VkExecDispatch *d,const char *stage,VkExecGraphicsResult result) {
    if(result!=VK_EXEC_GRAPHICS_READY)d->input_failure.stage=stage;
    return result;
}
static VkExecGraphicsResult dispatch_device_pointers_prepare(VkExec *ex,VkExecDispatch *d) {
    assert(ex&&d&&!d->selection->pointer_count);return VK_EXEC_GRAPHICS_READY;
}
static void device_storage_pointers_free(void *sources,unsigned count){assert(!sources&&!count);}
static VkExecGraphicsResult compute_samples_select(VkExec *ex,VkExecDispatch *d,RcRegistryPosition attempted,VkExecGpaSampleScope **pending){
    (void)attempted;(void)pending;
    assert(!d->selection->samples && !rc_registry_position_compare(d->read_position,read));
    if (no_sample_reflection) {
        assert(!ex->tex[0][8] && (d->info->n_bindings==0 || d->info->bindings[0].kind!=GFX_RES_TEX));
        return VK_EXEC_GRAPHICS_READY;
    }
    assert(ex->tex[0][8]==17);
    ComputeSamples *samples=calloc(1,sizeof *samples);
    assert(samples);
    samples->read=d->read_position;
    samples->info=d->info;
    samples->one.image.image=conflicting_peer?storage_original:77;
    samples->one.hold=samples;
    d->selection->samples=samples;
    return VK_EXEC_GRAPHICS_READY;
}
static int compute_samples_current_locked(const VkExec *ex,const ComputeSamples *samples,int uploaded) {
    assert(lock_count && ex && samples && uploaded);
    return !rc_registry_position_compare(samples->read,read);
}
static void compute_samples_release(ComputeSamples **samples){
    if(*samples){assert((*samples)->one.hold==*samples);free(*samples);*samples=NULL;sample_releases++;}
}
static VkExecGraphicsResult compute_aliases_prepare(VkExec *ex,ComputeSamples *samples){
    if (no_sample_reflection) {
        assert(ex && !samples);
        return VK_EXEC_GRAPHICS_READY;
    }
    assert(ex && samples && !rc_registry_position_compare(samples->read,read));
    return VK_EXEC_GRAPHICS_READY;
}
static VkExecGraphicsResult draw_descriptor_grant_prepare(VkExec *ex,const VkPipeInfo *info,DrawDescriptorGrant *grant,int ordered){
    assert(ex&&info&&ordered&&!grant->live);grant->live=1;return VK_EXEC_GRAPHICS_READY;
}
static void draw_descriptor_grant_release(DrawDescriptorGrant *grant){grant->live=0;}
static VkExecGraphicsResult dispatch_buffers_prepare(VkExec *ex,VkExecDispatch *d) {
    assert(ex&&d->descriptors.live);
    buffer_preparations++;
    return VK_EXEC_GRAPHICS_READY;
}
static int dispatch_compat_storage_answers_current(VkExec *ex,const VkExecDispatch *d){
    assert(ex && d->selection->count==(unsigned)(1+readonly_enabled));
    assert(!d->compatibility || (no_sample_reflection ? !d->selection->samples : !!d->selection->samples));
    return 1;
}
static void vk_texture_alias_read_cancel(VkTextureAliasRead *seed){if(seed){assert(seed->live);seed->live=0;}}

static VkImage vk_texture_write_target_prepared_image(const VkTextureWriteTarget *);
static int vk_texture_write_target_current(const VkTextureWriteTarget *);
static VkExecGraphicsResult dispatch_first_write_prepare(VkExec *,const VkTextureWritePlan *);
static int vk_texture_write_plan_independent(const VkTextureWritePlan *);
static int vk_dlaa_swap_active(const VkExec *ex){(void)ex;return 0;}
static int vk_dlaa_swap_matches(const VkTexImage *image){(void)image;assert(0);return 0;}
static void vk_dlaa_swap_use(VkExec *ex){(void)ex;assert(0);}
static const VkTexImage *vk_texture_current_registration_locked(int logical,view_t *view,int *has_view) {
    (void)logical;(void)view;(void)has_view;assert(0);return NULL;
}
static void tex_info_full(VkExec *ex,int stage,uint32_t member,const GfxShaderRes *resource,uint64_t handle,
                          int storage,int separate,VkDescriptorImageInfo *out,VkImage *image,
                          int require,VkTextureReadKey *chosen,int *recorded) {
    (void)ex;(void)stage;(void)member;(void)resource;(void)handle;(void)storage;(void)separate;
    (void)out;(void)image;(void)require;(void)chosen;(void)recorded;
    assert(0);
}
static EffectImageCapture *effect_image_capture(const GfxShaderRes *r,uint64_t handle) {
    if(effect_current) for(uint32_t i=0;i<effect_current->image_count;i++) {
        EffectImageCapture *capture=&effect_current->images[i];
        if(capture->resource==r&&capture->handle==handle)return capture;
    }
    return NULL;
}
static void vk_exec_required_graphics_unavailable(const VkExec *ex,unsigned op,const char *reason) {
    assert(ex&&(op==OP_DISPATCH||(rmw_test&&op==OP_DISPATCH_IND))&&reason&&lock_count);
    issue_refusals++;
}
static void vk_dlaa_hud_write(VkImage image){assert(image==storage_original);hud_write_calls++;}
static int texel_bytes(VkFormat format) { return format == VK_FORMAT_R8_UNORM ? 1 : 4; }

static int storage_integer_view(const VkTexImage *image,const view_t *view,int has_view,VkFormat *format) {
    (void)view; (void)has_view;
    *format=image->format;
    return 0;
}


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

static int dispatch_storage_format_features(VkFormat format,uint32_t *native_format) {
    if (rmw_test && format==VK_FORMAT_R32_UINT) {
        *native_format=33;
        return 2u | (rmw_atomic_feature ? 16u : 0u);
    }
    assert(format==VK_FORMAT_R8_UNORM);
    *native_format=15;
    return 2u|4u|(readonly_format_ok?8u:0u);
}
static const GraphicsSample *compute_sample_receipt(const ComputeSamples *samples,uint32_t member,
                                                      const GfxShaderRes *r,uint64_t handle) {
    assert(!member);
    if (!samples) return NULL;
    assert((r==&sample_binding&&handle==17)||(r==&comparison_binding&&handle==comparison_handle));
    return &samples->one;
}
static int compute_query_geometry_binding(const VkExec *ex,const VkPipeInfo *info,const VkPipeBinding *binding){
    (void)ex;(void)info;(void)binding;return 0;
}
static int texture_view_key(const VkTexImage *image,const view_t *view,int hv,uint16_t type,int storage,ViewKey *key) {
    assert((type==3||type==4||type==0x503)&&image->image);
    *key=(ViewKey){.image=image->image,.aspect=hv?view->aspect:1,.level0=hv?view->level0:0,
                  .levels=hv?view->levels:1,.layer0=hv?view->layer0:0,.layers=hv?view->layers:1,
                  .type=image->view_type,.format=image->format};
    (void)storage;return 1;
}
'''

ALLOCATION = r'''

typedef struct {uint64_t id;} RegistrationHandle;
typedef struct {int live,scene_uses;} VkMemoryImageHold;
typedef struct {
    int target,width,height,depth,levels,samples;
    uint32_t format,swizzle_r,swizzle_g,swizzle_b,swizzle_a,layout_reserved,stride;
} nvn_texdesc;
typedef struct {uint64_t addr;nvn_texdesc d;} TexKey;
typedef struct {uint64_t size,layer_size;unsigned align,layers,bpb,tileH,tileD;} nvn_texlayout;
typedef struct {void *mem;uint64_t serial;nvn_texlayout L;nvn_texdesc d;} tex_t;
typedef struct {uint32_t vk,ideal;int depth,stencil,expand_rgb,alpha_one,chan[4],astc_bw,astc_bc;} TexVkFormat;
static int __attribute__((unused)) tex_vk_format(int guest,TexVkFormat *out) {
    if (guest==NVN_FORMAT_R11G11B10F) *out=(TexVkFormat){.vk=122,.ideal=122};
    else if (guest==NVN_FORMAT_RGB9E5F) *out=(TexVkFormat){.vk=123,.ideal=123};
    else return 0;
    return 1;
}
static int nvn_tex_is_linear(const nvn_texdesc *image){assert(image);return 0;}
static int nvn_fmt_bpb(uint32_t format,int *width,int *height){
    *width=*height=1;
    if (format==NVN_FORMAT_R11G11B10F || format==NVN_FORMAT_RGB9E5F) return 4;
    return format==VK_FORMAT_R8_UNORM;
}
typedef struct {tex_t descriptor;RegistrationHandle handle;view_t view;int has_view;} TextureRegistration;
typedef struct Tex Tex;
typedef struct {
    const TextureRegistration *registration;
    Tex *ce,*owned;
    view_t cview;
    uint32_t cgen,cepoch;
    int finalized,used,sample_owned,cheld,chas_view;
} TexSlot;
typedef struct {TexSlot *slot;Tex *entry;int associated;} TextureSlotSelection;
typedef int VkTextureSampleStatus,VkTextureSampleRefusal;
enum {VK_TEXTURE_STORAGE_CAPTURE_UNSUPPORTED=-2,VK_TEXTURE_STORAGE_CAPTURE_INVALIDATED=-1,TEX_PRIVATE_STORAGE_OUTPUT=2,
      VK_TEXTURE_STORAGE_CAPTURE_ABSENT=0,VK_TEXTURE_STORAGE_CAPTURE_READY=1,VK_TEXTURE_STORAGE_CAPTURE_COW=2,VK_TEXTURE_STORAGE_CAPTURE_BOUND=3,VK_TEXTURE_REGISTRATION_GONE=5};
struct Tex {
    VkTexImage pub;
    TexKey key;
    tex_t t;
    TexVkFormat f;
    struct {uint32_t flags;} ik;
    uint64_t content_revision,sample_general_revision;
    void *alloc, *bound_to, *image_host_backing;
    VkTextureWriteTarget *pending_write;
    struct { uint64_t id; } image_authority;
    struct { int possible; } *native_output;
    void *image_mapped_holds;
    int alias_fragment_owner, storage_registration_owned, source_canonical;
    Tex *next_at;
    unsigned og_start;
    uint32_t image_mapped_hold_count;
    uint32_t image_device_count, image_private_mapping_count;
    const RcInterval *image_private_mapping;
    int pin_owned,pin_detached,material,released,retiring,superseded,overwritten,uploaded,up_retry,host_storage_init;
};
typedef struct {int proof;} VkTextureAliasLayout;
enum { VK_TEXTURE_REGISTRATION_CURRENT=2, VK_IMAGE_USAGE_STORAGE_BIT=4, VK_IMAGE_USAGE_TRANSFER_DST_BIT=8,VK_IMAGE_USAGE_TRANSFER_SRC_BIT=16,
       VK_IMAGE_ASPECT_COLOR_BIT=1, HOST_STORAGE_COW=2 };

static TextureRegistration cold_registration,readonly_registration;
static TexSlot tslots[]={{.registration=&cold_registration,.used=1},{.registration=&readonly_registration,.used=1}};
static const int ntslots=2;
static const uint32_t sinfo[]={0,1};
static uint32_t agen[2];
static int pool_ctx,in_lookup,resolve_held;
static Tex *texture_registration_source(const TexSlot *slot){return slot->owned;}
static int slot_absence_valid(const TexSlot *slot,uint32_t gen){assert(slot&&gen);return 0;}
static Tex cold_entry,*cold_selected,readonly_entry,redirect_entry,replacement_entry;
static unsigned slot_resolves, exact_slot_resolves, refreshes;
static uint64_t cold_mapped_cookie,readonly_mapped_cookie;
static unsigned cold_native_creates,cold_native_acquires,cold_native_releases;
static unsigned readonly_native_acquires,readonly_native_releases;
static const int by_key, by_addr;
static int word_alias_enabled,word_alias_initialized,word_alias_exact_active;
static const RcInterval word_mapping={0x19000,5,{37,0xb64200,0xb64000}};
static const RcInterval cold_mapping[]={ {0x19000,5,{37,0x1200,32}}, {0x19020,6,{39,0x4800,32}} };
static const RcInterval readonly_mapping[]={ {0x29000,17,{46,0x6200,32}}, {0x29020,18,{46,0x6220,32}} };
static const RcInterval readonly_overlap[]={ {0x29000,17,{46,0x6200,32}}, {0x29020,18,{39,0x481f,32}} };
static VkMemoryImageHold cold_native_hold,readonly_native_hold;
static void lock(void){vk_texture_lock();}
static void unlock(void){vk_texture_unlock();}
static uint64_t vk_exec_texture_pool(void){assert(lock_count);return 307;}
static int texture_registration_resolve(int logical,const TextureRegistration **history,int *native) {
    assert(lock_count && (logical==31 || (readonly_enabled && logical==32)));
    *native=logical-31;
    *history=tslots[*native].registration;
    if(logical==31&&cold_historical) return VK_TEXTURE_REGISTRATION_EXACT;
    if(logical==31&&cold_absent) {
        *history=NULL;
        *native=-1;
        return VK_TEXTURE_REGISTRATION_ABSENT;
    }
    return VK_TEXTURE_REGISTRATION_CURRENT;
}
typedef struct {uint64_t buffer,offset,avail;} VkMemSpan;
typedef struct {int current;RcInterval interval;} VkMemSpanCapture;
typedef struct {VkMemSpan span;VkMemSpanCapture *native;uint64_t address,bytes;} RawStorageInput;
static int vk_mem_span_held_argument_current(const VkMemSpanCapture *native,uint64_t address,uint64_t bytes,
    const VkMemSpan *span) {
    return native && native->current && span->buffer==155 && native->interval.address==address && bytes==12 &&
        native->interval.range.size==bytes;
}
static uint32_t vk_mem_span_input_intervals(const VkMemSpanCapture *native,uint64_t address,uint64_t bytes,
    RcInterval **out) {
    assert(native&&native->current&&native->interval.address==address&&native->interval.range.size==bytes);
    *out=malloc(sizeof **out);
    assert(*out);
    **out=native->interval;
    return 1;
}
static int stale_test_active, stale_map_missing;
static int vk_mem_lookup(uint64_t address,uint64_t size,VkMemSpan *backing){
    assert(lock_count&&backing&&address==0x19000&&size==64);
    if (stale_test_active && !stale_map_missing) *backing=(VkMemSpan){.buffer=155,.avail=size};
    return cold_refused==2 || (stale_test_active && !stale_map_missing);
}
static uint32_t vk_mem_device_intervals(const VkMemSpan *span,uint64_t address,uint64_t bytes,RcInterval **out) {
    assert(stale_test_active && span->buffer==155 && address==0x19000 && bytes==64);
    *out=malloc(sizeof cold_mapping);
    assert(*out);
    memcpy(*out,cold_mapping,sizeof cold_mapping);
    (*out)[0].mapping=cold_mapped_cookie;
    return 2;
}
static int host_mapping=1, host_authority=1, host_source=1;
static unsigned host_captures, host_cancels;
static int nvn_gpu_only(uintptr_t address){return (address==0x19000&&!host_material)||address==0x29000;}
static void make_key(const tex_t *descriptor,TexKey *key) {
    assert(descriptor->mem && lock_count);
    *key=(TexKey){(uintptr_t)descriptor->mem,descriptor->d};
}
static Tex *nvn_hm_get(const int *map,const void *query) {
    assert(lock_count && (map==&by_key || map==&by_addr));
    if (map==&by_addr) return cold_selected;
    const TexKey *key=query;
    if (key->addr==0x29000) return readonly_registered ? &readonly_entry : NULL;
    assert(key->addr==0x19000);
    if (word_alias_enabled) {
        if (key->d.format==NVN_FORMAT_RGB9E5F) return word_alias_exact_active ? &cold_entry : NULL;
        if (key->d.format==NVN_FORMAT_R11G11B10F) return &redirect_entry;
        return NULL;
    }
    if (key->d.swizzle_r==9) return &redirect_entry;
    return cold_selected;
}
static void nvn_hm_put(const int *map,const void *key,Tex *entry) {
    assert(lock_count);
    if (map==&by_addr) assert(*(const uint64_t *)key==entry->key.addr);
    else { assert(map==&by_key && !memcmp(key,&entry->key,sizeof entry->key)); cold_selected=entry; }
}
static int replacement_fail, replacement_invalidate;
static unsigned replacement_attempts, replacement_destroyed, predecessor_detached, predecessor_readers;
static unsigned n_site[3];
enum { PERF_CNT_TEX_NEW, FLIGHT_TEX_NEW };
static Tex *create_tex(const tex_t *t,const TexKey *key,int defer_init,int private_output,int *capacity_failure) {
    assert(lock_count && defer_init && private_output==TEX_PRIVATE_STORAGE_OUTPUT && cold_selected==&cold_entry);
    replacement_attempts++;
    *capacity_failure=replacement_fail;
    if (replacement_fail) return NULL;
    replacement_entry=cold_entry;
    replacement_entry.pub.image=storage_alternate;
    replacement_entry.pub.preserve_pool_bytes=1;
    replacement_entry.pub.gpu_owned=0;
    replacement_entry.key=*key;
    replacement_entry.t=*t;
    replacement_entry.content_revision=0;
    replacement_entry.image_private_mapping=cold_mapping;
    replacement_entry.released=replacement_entry.overwritten=replacement_entry.superseded=0;
    replacement_entry.alias_fragment_owner=replacement_entry.image_mapped_hold_count=0;
    replacement_entry.image_mapped_holds=NULL;
    replacement_entry.native_output=NULL;
    if (replacement_invalidate) cold_mapped_cookie++;
    return &replacement_entry;
}
static void gwin_remove(Tex *entry) {assert(entry==&replacement_entry);}
static void free_tex(Tex *entry,int retained) {
    assert(!retained && (entry==&cold_entry || entry==&replacement_entry));
    entry->pub.gone=1;
}
static void destroy_tex(Tex *entry) {
    assert(lock_count && entry==cold_selected);
    if (entry==&cold_entry) {
        assert(predecessor_readers && entry->native_output && entry->image_mapped_holds);
        predecessor_detached++;
    } else {
        assert(entry==&replacement_entry);
        replacement_destroyed++;
    }
    cold_selected=NULL;
    agen[0]++;
    free_tex(entry,0);
}
static void init_new(Tex *entry) {assert(entry==&replacement_entry && entry->pub.preserve_pool_bytes);}
static void origin_made(Tex *entry) {assert(entry==&replacement_entry);}
static void perf_count(int action) {assert(action==PERF_CNT_TEX_NEW);}
static void flight_ev(Tex *entry,int type,unsigned start,unsigned end) {
    assert(entry==&replacement_entry && type==FLIGHT_TEX_NEW && !start && !end);
}
static void tex_gen_bump(uint64_t address) {assert(address==0x19000);agen[0]++;}
static void touch(Tex *entry) {assert(entry==&replacement_entry);}
static int storage_seed_cold(const Tex *entry) {
    return !entry->uploaded && !entry->up_retry && !entry->material && !entry->pin_owned &&
        !entry->pub.gpu_owned && !entry->alias_fragment_owner;
}
static void slot_result_view(const TexSlot *slot,const Tex *entry,view_t *view,int *has_view) {
    assert(lock_count && ((slot==tslots && (entry==&cold_entry||entry==&redirect_entry||entry==&replacement_entry)) ||
                          (slot==&tslots[1] && entry==&readonly_entry)));
    *view=(view_t){.aspect=1,.level0=entry->pub.type==VK_IMAGE_TYPE_3D?1:0,.levels=1,.layers=1};
    if (entry->pub.type==VK_IMAGE_TYPE_3D) {
        view->set_levels=1;
        view->base_level=1;
    }
    if (slot==tslots && volume_view_override==1) view->set_layers=1;
    if (slot==tslots && volume_view_override==2) {
        view->set_target=1;
        view->target=VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    }
    if (slot==tslots && volume_view_override==3) view->layer0=1;
    if (slot==tslots && volume_view_override==4) view->levels=2;
    if (slot==tslots && volume_view_override==5) {
        view->set_target=1;
        view->target=NVN_TEXTURE_TARGET_3D;
    }
    if (word_alias_enabled) view->set_format=1,view->format=NVN_FORMAT_R32UI;
    else if (entry==&redirect_entry) view->level0=1;
    *has_view=1;
}
static int storage_target_identity_current(const Tex *entry);
static Tex *slot_resolve(TexSlot *slot,view_t *view,int *has_view,int private_output,uint32_t *installed) {
    assert(in_lookup && installed && !*installed && (private_output==0||private_output==TEX_PRIVATE_STORAGE_OUTPUT));
    *view=(view_t){0};
    *has_view=0;
    slot_resolves++;
    if (slot==tslots && cold_redirect) {
        slot_result_view(slot,&redirect_entry,view,has_view);
        return &redirect_entry;
    }
    if (slot==tslots && cold_selected) {
        slot_result_view(slot,cold_selected,view,has_view);
        return cold_selected;
    }
    if (slot==&tslots[1] && readonly_registered) {
        slot_result_view(slot,&readonly_entry,view,has_view);
        return &readonly_entry;
    }
    if (slot==&tslots[1]) {
        assert(lock_count&&readonly_enabled&&!readonly_registered&&readonly_resolve_deferred);
        tex_deferred=1;
        return NULL;
    }
    assert(cold_normal_entry && lock_count && slot==tslots && !cold_selected && !cold_native_hold.live);
    if (cold_refused) return NULL;
    cold_native_creates++;
    cold_entry=(Tex){.pub={.type=storage_shape==VK_IMAGE_VIEW_TYPE_3D?VK_IMAGE_TYPE_3D:VK_IMAGE_TYPE_2D,
       .view_type=storage_shape,.width=65,.height=33,.depth=storage_shape==VK_IMAGE_VIEW_TYPE_3D?9:1,.levels=3,
       .layers=1,.samples=1,.image=storage_original,.addr=0x19000,
       .usage=VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
       .aspect=VK_IMAGE_ASPECT_COLOR_BIT,.format=VK_FORMAT_R8_UNORM},
       .t=slot->registration->descriptor,.f={.vk=VK_FORMAT_R8_UNORM,.ideal=VK_FORMAT_R8_UNORM},.content_revision=67,.alloc=&cold_entry,
       .image_private_mapping_count=2,.image_private_mapping=cold_mapping};
    make_key(&cold_entry.t,&cold_entry.key);
    cold_selected=&cold_entry;
    slot_result_view(slot,cold_selected,view,has_view);
    return cold_selected;
}
static void texture_require_exact_locked(Tex *entry) {
    assert(lock_count && entry->pub.skip);
    exact_slot_resolves++;
    entry->pub.skip=0;
    agen[0]++;
}
static void texture_slot_selected_view(const TextureSlotSelection *,view_t *,int *);
static const VkTexImage *texture_slot_refresh_locked(TextureSlotSelection selected,view_t *view,
    int *has_view,VkTextureSampleStatus *status,VkTextureSampleRefusal *refusal) {
    assert(lock_count && !status && !refusal);
    refreshes++;
    if (!selected.entry || tex_deferred) return NULL;
    texture_slot_selected_view(&selected,view,has_view);
    return &selected.entry->pub;
}
typedef struct {
    RcRegistryPosition read;
    uint64_t pool,registration;
    int logical,native,has_view;
    view_t view;
} TextureWriteStorageSelection;
struct VkTextureWriteTarget {
    Tex *entry;
    VkImage selected,prepared;
    struct {int instance;} source_use;
    struct {const RcRegistrySourceProof *proofs;uint32_t proof_count;} *cpu_seed;
    TextureWriteStorageSelection storage;
};
static int mem_backing_current(const void *backing){return backing&&host_mapping;}
static int source_mapping_domain_current(const Tex *entry){return entry==&cold_entry&&host_mapping;}
static int source_permission_at(Tex *entry,RcRegistryPosition cut,void *out) {
    assert(lock_count&&entry==&cold_entry&&!out&&!rc_registry_position_compare(cut,read));
    return host_authority;
}
RcRegistry *rc_registry_runtime(void){return NULL;}
int rc_registry_source_proofs_current(RcRegistry *registry,const RcRegistrySourceProof *proofs,
    uint32_t count,RcRegistryPosition cut) {
    (void)registry;(void)proofs;(void)count;(void)cut;
    assert(0); // The separate ordered CPU seed fixture exercises CPU proof custody.
    return 0;
}
static int write_target_material_candidate(const Tex *entry,int allow_cold) {
    assert(lock_count&&!allow_cold);
    return entry==&cold_entry&&entry->material&&entry->pub.mirrored;
}
static const tex_t *nvn_texture_pin_acquire(const tex_t *descriptor) {
    assert(lock_count&&descriptor==&cold_registration.descriptor);
    return descriptor;
}
static void nvn_texture_pin_release(const tex_t *descriptor) {
    if (descriptor) assert(lock_count&&descriptor==&cold_registration.descriptor);
}
static const tex_t *nvn_texture_pin_retain_attachment(uint64_t serial,const Tex *entry) {
    (void)serial; (void)entry;
    assert(0); // The dedicated host-private fixture exercises pinned initialized LOAD.
    return NULL;
}
static Tex *nvn_texture_pin_attachment(const tex_t *pin) {
    (void)pin;
    assert(0);
    return NULL;
}
static void texture_use_pin(const Tex *entry,const tex_t *pin) {
    (void)entry; (void)pin;
    assert(0);
}
static int from_storage(const Tex *entry) {return entry->pub.mirrored;}
static int content_image_eligible(const Tex *entry) {assert(entry==&cold_entry);return 1;}
static TextureStorageCpuLoad *storage_host_cpu_load(Tex *entry,RcRegistryPosition read,const RcInterval *mapping,uint32_t count) {
    (void)entry; (void)read; (void)mapping; (void)count;
    assert(0); // The dedicated host fixture exercises this receipt.
    return NULL;
}
static void storage_host_cpu_load_release(TextureStorageCpuLoad *load) {assert(!load);}

static int storage_host_shape(const Tex *entry,const view_t *view,int has_view,VkImageSubresourceRange *plane) {
    (void)entry; (void)view; (void)has_view; (void)plane;
    assert(0);
    return 0;
}
static int storage_host_initialized(const Tex *entry,RcRegistryPosition position) {
    (void)entry; (void)position;
    assert(0);
    return 0;
}
static const RcInterval *storage_host_mapping(const Tex *entry,uint32_t *count) {
    (void)entry; (void)count;
    assert(0);
    return NULL;
}
static int storage_host_planes_current(const Tex *entry,const VkTextureStorageTarget *target,
    const VkImageSubresourceRange *planes) {
    (void)entry; (void)target; (void)planes;
    assert(0);
    return 0;
}
static int storage_target_capture_bound_locked(int logical,int native,const TextureRegistration *record,
    const TextureSlotSelection *selected,const TexKey *key,const view_t *view,int has_view,
    VkTextureStorageTarget **out,const char **reason) {
    (void)logical; (void)native; (void)record; (void)selected; (void)key; (void)view; (void)has_view; (void)out; (void)reason;
    assert(0); // Tested with the production lower bound capture in the separate custody fixture.
    return VK_TEXTURE_STORAGE_CAPTURE_UNSUPPORTED;
}
static void vk_texture_bound_relocation_cancel(VkTextureBoundRelocation *plan) { assert(!plan); }
static int write_target_capture(const tex_t *pin,const VkTexImage *shape,VkTextureWriteTarget **out,
    int cpu_allowed,Tex *selected) {
    assert(lock_count&&pin==&cold_registration.descriptor&&selected==&cold_entry&&shape==&selected->pub&&cpu_allowed);
    host_captures++;
    if(!host_source)return 0;
    *out=calloc(1,sizeof **out);
    assert(*out);
    **out=(VkTextureWriteTarget){.entry=selected,.selected=shape->image,.source_use.instance=1};
    selected->pending_write=*out;
    selected->pin_owned=1;
    return 1;
}
static void vk_texture_write_target_cancel(VkTextureWriteTarget *target) {
    lock();
    assert(target->entry->pending_write==target);
    target->entry->pending_write=NULL;
    target->entry->pin_owned=0;
    host_cancels++;
    unlock();
    free(target);
}
static int write_target_storage_current_locked(const VkTextureWriteTarget *);
static int write_target_current_locked(const VkTextureWriteTarget *target) {
    return target->entry->pending_write==target&&write_target_storage_current_locked(target);
}
static int vk_texture_write_target_current(const VkTextureWriteTarget *target) {
    lock();
    const int current=write_target_current_locked(target);
    unlock();
    return current;
}
static VkImage vk_texture_write_target_prepared_image(const VkTextureWriteTarget *target) {
    assert(target&&target->prepared);
    return target->prepared;
}
static VkExecGraphicsResult dispatch_first_write_prepare(VkExec *ex,const VkTextureWritePlan *plan) {
    assert(ex&&plan->count==1&&vk_texture_write_target_current(plan->targets[0]));
    plan->targets[0]->prepared=storage_alternate;
    return VK_EXEC_GRAPHICS_READY;
}
static int vk_texture_write_plan_independent(const VkTextureWritePlan *plan) {
    return plan->count==1&&plan->targets[0];
}
static int graphics_sample_shape_current(const GraphicsSample *sample) {
    return sample->hold&&sample->image.image;
}
static const RcInterval *vk_texture_alias_read_intervals(const VkTextureAliasRead *alias,uint32_t *count) {
    (void)alias;(void)count;
    assert(0);
    return NULL;
}
static int host_sample_conflict, comparison_mapped;
static const VkImage comparison_image=620;
static RcInterval comparison_mapping;
static const RcInterval *vk_texture_selected_sample_intervals_locked(VkTextureReadKey key,VkImage image,uint32_t *count) {
    assert(lock_count);
    if (image==comparison_image) {
        // The receipt must name the memory behind the selected view at the original read.
        *count=comparison_mapped&&key.registration==190&&key.native_slot==106&&key.projected;
        return *count?&comparison_mapping:NULL;
    }
    assert(image==77);
    *count=1;
    return host_sample_conflict?cold_mapping:readonly_mapping;
}
static VkImageMemoryPlacement comparison_placement;
static const RcInterval *vk_texture_native_sample_intervals_locked(RcRegistryPosition at,
    const VkTextureSample *sample,const void *held,VkImage expected,VkImageMemoryPlacement *placement,uint32_t *count) {
    assert(lock_count&&sample&&held);
    *placement=comparison_placement;
    *count=comparison_mapped&&expected==comparison_image&&
        !rc_registry_position_compare(at,read)&&!sample->registration&&sample->native_slot==106;
    return *count?&comparison_mapping:NULL;
}
static int vk_texture_write_target_disjoint_placement(const VkTextureWriteTarget *target,VkImageMemoryPlacement placement){
    assert(target);
    return !placement.memory;
}
typedef void *VmaAllocation;
static int texture_image_placement(VmaAllocation allocation,VkImageMemoryPlacement *placement) {
    assert(allocation==cold_entry.alloc);
    *placement=(VkImageMemoryPlacement){1,2000,500};
    return 1;
}
static int vk_texture_write_disjoint_input(const VkTextureWriteTarget *target,const void *pinned,
    const RcInterval *intervals,uint32_t count) {
    assert(lock_count&&target->prepared&&!pinned);
    return count==1&&intervals==readonly_mapping;
}
static int image_alias_material_layout(const Tex *entry,VkTextureAliasLayout *layout){
    assert(lock_count && (entry==&cold_entry || entry==&readonly_entry || entry==&redirect_entry || entry==&replacement_entry) && layout);
    return entry->pub.format==VK_FORMAT_R8_UNORM||(rmw_test&&entry->pub.format==VK_FORMAT_R32_UINT)||
        (word_alias_enabled&&(entry->pub.format==VK_FORMAT_B10G11R11_UFLOAT_PACK32||
                             entry->pub.format==VK_FORMAT_E5B9G9R9_UFLOAT_PACK32));
}
static int image_alias_placement_current(const Tex *entry) {
    assert(lock_count && (entry==&cold_entry || entry==&readonly_entry || entry==&redirect_entry || entry==&replacement_entry));
    const uint64_t cookie=entry==&readonly_entry?readonly_mapped_cookie:cold_mapped_cookie;
    return cookie==entry->image_private_mapping[0].mapping;
}
static int __attribute__((unused)) image_alias_source_current(const Tex *entry) {
    assert(word_alias_enabled && entry==&redirect_entry);
    return image_alias_placement_current(entry);
}
static const RcInterval *__attribute__((unused)) image_alias_plane_demand(Tex *entry,
    const VkImageSubresourceRange *plane,uint32_t *count) {
    assert(word_alias_enabled && lock_count && entry==&redirect_entry);
    if (plane->aspectMask!=1 || plane->baseMipLevel || plane->levelCount!=1 ||
        plane->baseArrayLayer || plane->layerCount!=1) return NULL;
    *count=1;
    return &word_mapping;
}
static int __attribute__((unused)) image_alias_resident_current(const Tex *entry,const RcInterval *demand,uint32_t count,
    RcRegistryPosition at) {
    assert(word_alias_enabled && lock_count && entry==&redirect_entry);
    return demand==&word_mapping && count==1 && word_alias_initialized && !rc_registry_position_compare(at,read);
}
static int image_alias_guest_mapping_current(uint64_t address,uint64_t bytes,const RcInterval *expected,uint32_t count){
    if (word_alias_enabled)
        return address==0x19000 && bytes==word_mapping.range.size && count==1 && expected &&
            expected->address==word_mapping.address && expected->mapping==cold_mapped_cookie &&
            rc_range_equal(expected->range,word_mapping.range);
    assert(lock_count && (address==0x19000||address==0x29000) && bytes==64 && count==2);
    const int writer=address==0x19000;
    const RcInterval *active=writer?cold_mapping:readonly_entry.image_private_mapping;
    if(expected[0].mapping!=(writer?cold_mapped_cookie:readonly_mapped_cookie)) return 0;
    for (uint32_t i=0;i<count;++i) {
        if(expected[i].address!=active[i].address || expected[i].mapping!=active[i].mapping ||
           !rc_range_equal(expected[i].range,active[i].range)) return 0;
    }
    return 1;
}
static VkMemoryImageHold *vk_mem_image_hold(VkImage image) {
    assert(lock_count && (image==storage_original||image==readonly_original||image==storage_alternate));
    VkMemoryImageHold *hold=image==readonly_original?&readonly_native_hold:&cold_native_hold;
    assert(!hold->live);
    hold->live=1;
    if(image==readonly_original) readonly_native_acquires++; else cold_native_acquires++;
    return hold;
}
static int vk_mem_image_hold_retired(const VkMemoryImageHold *hold) {
    assert(lock_count && (hold==&cold_native_hold||hold==&readonly_native_hold));
    return !hold->live;
}
static void vk_mem_image_release(VkMemoryImageHold *hold) {
    assert(!lock_count && (hold==&cold_native_hold||hold==&readonly_native_hold) && hold->live);
    if(hold==&cold_native_hold) cold_native_releases++; else readonly_native_releases++;
    hold->live=0;
}
static int image_alias_load_current(Tex *entry,const VkImageSubresourceRange *planes,RcRegistryPosition position) {
    assert(lock_count && (entry==&readonly_entry || entry==&cold_entry || entry==&redirect_entry || entry==&replacement_entry) &&
           planes&&planes->aspectMask==VK_IMAGE_ASPECT_COLOR_BIT);
    if (entry==&cold_entry && rmw_test)
        return rmw_predecessor==entry->pub.image && !rc_registry_position_compare(position,read) && !planes->baseMipLevel &&
            planes->levelCount==1 && !planes->baseArrayLayer && planes->layerCount==1;
    return entry==&readonly_entry && readonly_initialized && !rc_registry_position_compare(position,read);
}
typedef struct {uint64_t fence;} TextureUseBatch;
typedef struct {VkMemoryImageHold *hold;VkImage image;} TextureNativeUse;
static TextureUseBatch batch;
static TextureUseBatch *texture_use_active(uint64_t fence,int create) {
    assert(lock_count && create && fence==read.scene);
    batch.fence=fence;
    return &batch;
}
static unsigned stale_part_releases;
static void texture_use_part_release(void *part) {assert(part==&batch);stale_part_releases++;}
static void texture_native_use_batch(TextureUseBatch *active,const TextureNativeUse *input) {
    assert(lock_count&&active==&batch&&active->fence==read.scene);
    assert(input->hold==&readonly_native_hold&&input->image==readonly_original&&input->hold->live);
    input->hold->scene_uses++;
}
'''

SEED = r'''


static int vk_texture_storage_target_host_mirror(const VkTextureStorageTarget *target) {
    assert(!target || !target->host_cpu);
    return 0;
}
typedef RcInterval MemBackingCapture;
enum {RC_BACKING_OK};
static int mem_backing_capture(uint64_t address,uint64_t bytes,MemBackingCapture **out) {
    assert(rmw_test&&address==rmw_input_mapping.address&&bytes==rmw_input_mapping.range.size);
    *out=&rmw_input_mapping;
    return RC_BACKING_OK;
}

static const RcInterval *mem_backing_intervals(const MemBackingCapture *capture,uint32_t *count) {
    *count=1;
    return capture;
}
static void mem_backing_release(MemBackingCapture **capture) { *capture=NULL; }
static int ubo_ptr_res(const GfxShaderRes *resource) {(void)resource;return 0;}
static const p_bind_buf *ssbo_direct_checked(const VkExec *ex,int stage,const GfxShaderRes *resource,
    int *uniform,p_bind_buf *temporary,int *refused) {
    (void)ex; (void)stage; (void)resource; (void)uniform; (void)temporary; (void)refused;
    return NULL;
}
static UniformAddressRange storage_uniform_address_range(const p_bind_buf *source) {
    return (UniformAddressRange){source->addr,source->size,4};
}
static int dispatch_storage_mirror_buffers_disjoint(const VkExec *ex,VkExecDispatch *dispatch,const EffectImageCapture *output) {
    (void)ex; (void)dispatch; (void)output; assert(0); return 0;
}
static VkExecGraphicsResult dispatch_storage_host_cpu_prepare(VkExec *ex,VkTextureStorageTarget *target) {
    (void)ex; (void)target; assert(0); return VK_EXEC_GRAPHICS_UNAVAILABLE;
}

static VkImage vk_texture_storage_target_image(const VkTextureStorageTarget *target,const VkTextureAliasRead *seed) {
    assert(target && (!seed || (seed->live&&target->image==storage_original)));
    return target->image;
}
static int vk_texture_alias_read_current(const VkTextureAliasRead *seed){return seed->live;}
static int vk_texture_alias_target_current(const VkTextureAliasRead *seed,VkImage image) {
    assert(image==storage_original);return seed->live;
}
static int vk_texture_storage_target_use(const VkTextureStorageTarget *target,VkTextureAliasRead *seed,
                                          uint64_t fence,VkCommandBuffer cb){
    (void)cb;
    assert(lock_count&&target->image==storage_original&&target->read.scene==fence);
    assert((seed&&seed->live)||(!seed&&rmw_test&&rmw_predecessor==storage_original));
    assert(cold_native_hold.live);
    cold_native_hold.scene_uses++;
    writable_uses++;
    return 1;
}
static int vk_texture_storage_target_written_preserving(const VkTextureStorageTarget *target,const VkImageSubresourceRange *planes) {
    assert(lock_count&&rmw_test&&rmw_predecessor==storage_original&&target->image==storage_original&&
        planes&&planes->aspectMask==VK_IMAGE_ASPECT_COLOR_BIT&&!planes->baseMipLevel&&planes->levelCount==1);
    writable_publications++;
    return 1;
}
static int vk_texture_storage_target_written(const VkTextureStorageTarget *target,VkTextureAliasRead *seed,
                                              const VkImageSubresourceRange *planes) {
    assert(lock_count&&target->image==storage_original&&seed&&seed->live&&planes->aspectMask==1);
    writable_publications++;
    return 1;
}
typedef struct {int phase,status;} VkTextureAliasFailure;
typedef enum {VK_TEXTURE_STORAGE_SEED_CURRENT,VK_TEXTURE_STORAGE_SEED_CAPTURED,
              VK_TEXTURE_STORAGE_SEED_UNSUPPORTED,VK_TEXTURE_STORAGE_SEED_REFUSED} VkTextureStorageSeedResult;
typedef enum {VK_EXEC_RESTORE_READY,VK_EXEC_RESTORE_STOPPED,VK_EXEC_RESTORE_STALE} VkExecRestoreResult;
static VkExecRestoreResult vk_exec_prepare_storage_target_bound(VkExec *ex,VkTextureStorageTarget *target,VkTexImage *image) {
    (void)ex; (void)target; (void)image;
    assert(0); // No bound producer is declared by these compatibility and COW cases.
    return VK_EXEC_RESTORE_STALE;
}
enum {VK_TEXTURE_ALIAS_LAYOUT,VK_TEXTURE_ALIAS_CAPACITY};
static VkTextureAliasRead captured_seed;
static VkTexImage seed_shape;
static int volume_seed_stale;

static VkTextureStorageSeedResult vk_texture_storage_seed_capture(const VkTexImage *image,
    VkImageSubresourceRange planes,VkTextureAliasRead **seed,VkTextureAliasFailure *failure) {
    assert(image->image==storage_original&&*seed==NULL&&failure&&!captured_seed.live);
    assert(cold_native_hold.live&&planes.aspectMask==VK_IMAGE_ASPECT_COLOR_BIT);
    if (image->type==VK_IMAGE_TYPE_3D) {
        assert(image->width==65&&image->height==33&&image->depth==9);
        assert(planes.baseMipLevel==1&&planes.levelCount==1&&planes.baseArrayLayer==0&&planes.layerCount==1);
    } else {
        assert(planes.baseMipLevel==0&&planes.levelCount==1&&planes.baseArrayLayer==0&&planes.layerCount==1);
    }
    seed_shape=*image;
    captured_seed.live=1;
    *seed=&captured_seed;
    cold_seeds++;
    return VK_TEXTURE_STORAGE_SEED_CAPTURED;
}
static VkExecRestoreResult vk_exec_prepare_alias_texture_read(VkExec *ex,VkTextureAliasRead *const *aliases,
    uint32_t count,int (*current)(void *),void *opaque,const VkTexImage **images) {
    const VkExecDispatch *d=opaque;
    const EffectImageCapture *capture=d->selection->images;
    const VkTextureStorageTarget *target=capture->independent;
    assert(ex&&d->selection->count==(unsigned)(1+readonly_enabled)&&capture->written&&!capture->full_overwrite);
    assert(count==1&&aliases[0]==&captured_seed&&aliases[0]->live&&current(opaque));
    assert(target->mapping_count==2&&target->mapping[0].range.backing==37&&target->mapping[1].range.backing==39);
    assert(!rc_registry_position_compare(target->read,d->read_position));
    if(volume_seed_stale) return VK_EXEC_RESTORE_STALE;
    images[0]=&seed_shape;
    return VK_EXEC_RESTORE_READY;
}
static void dispatch_storage_seed_refused(const VkExecDispatch *d,const EffectImageCapture *capture,
    VkTextureStorageSeedResult result,const VkTextureAliasFailure *failure) {
    (void)d;(void)capture;(void)result;(void)failure;
    assert(0);
}
static uint64_t alias_resource_view_locked(VkTextureAliasRead *alias,const VkTexImage *image,
    const view_t *view,int hv,int type,int storage) {
    (void)alias;(void)image;(void)view;(void)hv;(void)type;(void)storage;
    assert(0);
    return 0;
}

'''

CASES = r'''


static VkExecGraphicsResult prepare_peer(VkExec *ex,const p_dispatch *groups,const VkPipeInfo *info,VkExecDispatch *d) {
    assert(vk_exec_dispatch_compat_inputs_select(ex,OP_DISPATCH,groups,d));
    d->info=info;
    d->pipeline=info->pipeline;
    return vk_exec_dispatch_prepare_compat_inputs(ex,d,(RcRegistryPosition){0},NULL);
}

static void release_peer(VkExecDispatch *dispatch) {
    dispatch_compat_storage_release(dispatch->selection);
    compute_samples_release(&dispatch->selection->samples);
    draw_descriptor_grant_release(&dispatch->descriptors);
    free(dispatch->selection);
    *dispatch=(VkExecDispatch){0};
}

static VkPipeInfo readonly_test_pipeline(void) {
    return (VkPipeInfo){.n_bindings=3,.pipeline=44,
        .bindings={{&sample_binding,GFX_RES_TEX,GFX_STAGE_CS,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,1,0},
                   {&storage_binding,GFX_RES_IMG,GFX_STAGE_CS,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,2,0},
                   {&readonly_binding,GFX_RES_IMG,GFX_STAGE_CS,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,3,0}},
        .native_image_seed_bindings={.count=2,.entries={
            {.binding=2,.scalar=RC_NATIVE_IMAGE_FLOAT32,.access=RC_NATIVE_IMAGE_SEED_WRITE},
            {.binding=3,.scalar=RC_NATIVE_IMAGE_FLOAT32,.access=RC_NATIVE_IMAGE_SEED_READ}}}};
}

static void test_readonly_storage_peer(void) {
    const p_dispatch groups={2,3,1};
    const prog_t program={83};
    DrawCtx cx={0};
    VkExec ex={.dc=&cx,.fence=read.scene,.prog=&program,.tex={{[8]=17}},.img={{[3]=31,[4]=32}}};
    const VkPipeInfo info=readonly_test_pipeline();
    readonly_registration=(TextureRegistration){.descriptor={.mem=(void *)(uintptr_t)0x29000,.serial=111,.L.size=64},.handle={189}};
    readonly_entry=(Tex){.pub={.type=VK_IMAGE_TYPE_2D,.view_type=VK_IMAGE_VIEW_TYPE_2D,.width=64,.height=64,.depth=1,.levels=1,
        .layers=1,.samples=1,.image=readonly_original,.addr=0x29000,
        .usage=VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,.aspect=VK_IMAGE_ASPECT_COLOR_BIT,
        .format=VK_FORMAT_R8_UNORM},.key={.addr=0x29000},.t=readonly_registration.descriptor,.content_revision=201,.alloc=&readonly_entry,
        .image_private_mapping_count=2,.image_private_mapping=readonly_mapping};
    readonly_mapped_cookie=readonly_mapping[0].mapping;
    readonly_enabled=readonly_registered=readonly_initialized=1;
    const unsigned initial_seeds=cold_seeds;
    VkExecDispatch dispatch={0};

    // Readonly absence cannot become a zero-filled substitute or an output seed.
    readonly_initialized=0;
    assert(prepare_peer(&ex,&groups,&info,&dispatch)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!dispatch.descriptors.live&&cold_seeds==initial_seeds);
    release_peer(&dispatch);
    readonly_initialized=1;
    readonly_format_ok=0;
    assert(prepare_peer(&ex,&groups,&info,&dispatch)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!dispatch.descriptors.live&&cold_seeds==initial_seeds);
    release_peer(&dispatch);
    readonly_format_ok=1;
    vk_caps.robust_image_access=0;
    assert(prepare_peer(&ex,&groups,&info,&dispatch)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!dispatch.descriptors.live&&cold_seeds==initial_seeds);
    release_peer(&dispatch);
    vk_caps.robust_image_access=1;

    readonly_entry.image_private_mapping=readonly_overlap;
    assert(prepare_peer(&ex,&groups,&info,&dispatch)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!strcmp(dispatch.input_failure.reason,"storage-peer-mapping")&&!dispatch.descriptors.live);
    assert(cold_seeds==initial_seeds);
    release_peer(&dispatch);
    readonly_entry.image_private_mapping=readonly_mapping;

    for (int issue=0;issue<2;++issue) {
        assert(prepare_peer(&ex,&groups,&info,&dispatch)==VK_EXEC_GRAPHICS_READY);
        assert(dispatch.selection->count==2 && dispatch.descriptors.live &&
               dispatch.selection->samples->one.image.image==77);
        EffectImageCapture *output=&dispatch.selection->images[0], *peer=&dispatch.selection->images[1];
        assert(output->written&&output->storage_seed&&output->issued_planes.aspectMask==1);
        assert(!peer->written&&!peer->storage_seed&&peer->issued_planes.aspectMask==1&&
               peer->independent->image==readonly_original&&peer->independent->registration==189);
        assert(!peer->phase_seen && !peer->descriptor.imageView);
        EffectDeclaration effect={.image_count=2,.images=dispatch.selection->images};
        effect_current=&effect;
        // Stand in for Vulkan creating the exact selected views. The actual binder must consume ours.
        output->descriptor=(VkDescriptorImageInfo){701,VK_IMAGE_LAYOUT_GENERAL};
        peer->descriptor=(VkDescriptorImageInfo){702,VK_IMAGE_LAYOUT_GENERAL};
        cx.n_storage_images=1;
        cx.storage_images[0]=(BindingImageSelection){&storage_binding,31,storage_original,output->descriptor};

        if(!issue) {
            vk_texture_lock();
            assert(!dispatch_storage_outputs_begin(&ex,OP_DISPATCH));
            vk_texture_unlock();
            assert(issue_refusals==1&&!readonly_native_hold.scene_uses&&!writable_uses);
        }
        VkDescriptorImageInfo selected={0};
        VkImage bound=0;
        tex_info(&ex,NVN_STAGE_CS,0,&readonly_binding,32,1,0,&selected,&bound,NULL,NULL);
        assert(peer->phase_seen&&selected.imageView==702&&selected.imageLayout==VK_IMAGE_LAYOUT_GENERAL&&bound==readonly_original);
        if(!issue) {
            readonly_mapped_cookie++;
            assert(!dispatch_compat_storage_current(dispatch.selection));
            vk_texture_lock();
            assert(!dispatch_storage_outputs_begin(&ex,OP_DISPATCH));
            vk_texture_unlock();
            assert(issue_refusals==2&&!readonly_native_hold.scene_uses&&!writable_uses);
            readonly_mapped_cookie--;
        }
        vk_texture_lock();
        assert(dispatch_storage_outputs_begin(&ex,OP_DISPATCH));
        assert(readonly_native_hold.scene_uses==1&&cold_native_hold.scene_uses==1&&writable_uses==(unsigned)(issue+1));
        assert(hud_write_calls==writable_uses);
        if(issue) dispatch_storage_outputs_written(&ex,OP_DISPATCH);
        vk_texture_unlock();
        assert(writable_publications==(unsigned)issue);
        effect_current=NULL;
        release_peer(&dispatch);
        assert(!readonly_native_hold.live&&!cold_native_hold.live);
        // Known-unhanded disposal releases scene custody separately from the retained declaration.
        assert(readonly_native_hold.scene_uses==1&&cold_native_hold.scene_uses==1);
        readonly_native_hold.scene_uses--;
        cold_native_hold.scene_uses--;
    }
    assert(readonly_native_acquires==readonly_native_releases&&cold_native_acquires==cold_native_releases&&!lock_count);
    assert(cold_seeds==initial_seeds+2&&writable_publications==1);
    puts("Readonly storage peer: exact view/use, canonical overlap, stale remap, and unissued custody passed");
}


static void test_native_readonly_storage_peer(void) {
    const p_dispatch groups={2,3,1};
    const prog_t program={83};
    const VkPipeInfo info=readonly_test_pipeline();
    VkExec ex={.prog=&program,.fence=read.scene,.tex={{[8]=17}},.img={{[3]=31,[4]=32}}};
    const unsigned initial_read_acquires=readonly_native_acquires, initial_read_releases=readonly_native_releases;
    const unsigned initial_write_acquires=cold_native_acquires, initial_write_releases=cold_native_releases;
    const unsigned initial_publications=writable_publications, initial_uses=writable_uses;
    const unsigned initial_seeds=cold_seeds;

    for (int deferred=0;deferred<2;++deferred) {
        const unsigned reader_start=readonly_native_acquires, released_start=readonly_native_releases;
        VkExecDispatchImages *selection=calloc(1,sizeof *selection);
        assert(selection);
        selection->images=calloc(1,sizeof *selection->images);
        assert(selection->images);
        selection->count=1;
        EffectImageCapture *writer=&selection->images[0];
        *writer=(EffectImageCapture){.resource=&storage_binding,.handle=31,.slot=3,.stage=NVN_STAGE_CS,.written=1};
        // The native shader effect pass has already retained its writer. No shader answer queried the
        // read-only slot, so its capture is absent until the late final attested inventory arrives.
        assert(vk_texture_effect_independent_storage(31,storage_binding.exact,1,&writer->image,&writer->view,&writer->has_view,
                                                    &writer->size,&writer->independent,NULL)==VK_TEXTURE_STORAGE_CAPTURE_READY);
        VkTextureStorageTarget *const original_writer=writer->independent;
        VkExecDispatch dispatch={.info=&info,.selection=selection,.shader=&shader,.program=&program,
            .program_serial=program.serial,.registration=140,.opcode=OP_DISPATCH,.payload=&groups,
            .read_position=read,.pipeline=info.pipeline};
        assert(!dispatch.compatibility&&cold_native_hold.live&&!readonly_native_hold.live&&selection->count==1);

        if(!deferred) {
            // Capture succeeded, but plane validation rejected. Only the temporary source lease may be released.
            readonly_initialized=0;
            assert(dispatch_native_storage_peers_prepare(&ex,&dispatch)==VK_EXEC_GRAPHICS_UNAVAILABLE);
            assert(selection->count==1&&selection->images[0].independent==original_writer);
            assert(cold_native_hold.live&&!readonly_native_hold.live);
            assert(readonly_native_acquires==reader_start+1&&readonly_native_releases==released_start+1);
            readonly_initialized=1;
        } else {
            readonly_registered=0;
            tslots[1].ce=NULL;
            readonly_resolve_deferred=1;
            assert(dispatch_native_storage_peers_prepare(&ex,&dispatch)==VK_EXEC_GRAPHICS_INVALIDATED);
            assert(selection->count==1&&selection->images[0].independent==original_writer);
            assert(cold_native_hold.live&&!readonly_native_hold.live&&!vk_texture_deferred());
            assert(readonly_native_acquires==reader_start&&readonly_native_releases==released_start);
            readonly_registered=1;
            readonly_resolve_deferred=0;
        }

        assert(dispatch_native_storage_peers_prepare(&ex,&dispatch)==VK_EXEC_GRAPHICS_READY);
        assert(selection->storage_admission==RESOURCE_IMAGE_TRACE_STORAGE_SELECTED&&selection->count==2);
        assert(selection->images[0].independent==original_writer&&selection->images[0].written);
        const EffectImageCapture *peer=&selection->images[1];
        assert(peer->resource==&readonly_binding&&peer->handle==32&&!peer->written&&peer->independent&&
               peer->issued_planes.aspectMask==VK_IMAGE_ASPECT_COLOR_BIT&&!peer->storage_seed);
        const VkTextureStorageTarget *const original_peer=peer->independent;
        const unsigned read_acquires=readonly_native_acquires;
        assert(dispatch_native_storage_peers_prepare(&ex,&dispatch)==VK_EXEC_GRAPHICS_READY);
        assert(selection->count==2&&selection->images[1].independent==original_peer&&readonly_native_acquires==read_acquires);
        assert(dispatch_storage_seed_current(&selection->images[1],1));

        if(!deferred) {
            // A retained capture outside the final inventory must fail its matched-closure check.
            EffectImageCapture *larger=realloc(selection->images,3*sizeof *larger);
            assert(larger);
            selection->images=larger;
            selection->images[2]=(EffectImageCapture){.independent=original_writer};
            selection->count=3;
            assert(dispatch_native_storage_peers_prepare(&ex,&dispatch)==VK_EXEC_GRAPHICS_UNAVAILABLE);
            selection->count=2;
            selection->images[2]=(EffectImageCapture){0};
            assert(dispatch_native_storage_peers_prepare(&ex,&dispatch)==VK_EXEC_GRAPHICS_READY);
            assert(readonly_native_acquires==read_acquires);
        }
        assert(compute_samples_select(&ex,&dispatch,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_READY);
        assert(dispatch_storage_outputs_disjoint(&ex,&dispatch));
        assert(dispatch_storage_overwrite_select(&dispatch)==VK_EXEC_GRAPHICS_READY);
        assert(dispatch_storage_seeds_prepare(&ex,&dispatch)==VK_EXEC_GRAPHICS_READY);
        assert(dispatch_compat_storage_current(selection));

        // The command is never recorded. Receipt cancellation consumes both actual holds and the write seed.
        release_peer(&dispatch);
        assert(!cold_native_hold.live&&!readonly_native_hold.live&&!lock_count);
        assert(!cold_native_hold.scene_uses&&!readonly_native_hold.scene_uses);
        assert(writable_publications==initial_publications&&writable_uses==initial_uses);
    }
    assert(readonly_native_acquires==initial_read_acquires+3&&readonly_native_releases==initial_read_releases+3);
    assert(cold_native_acquires==initial_write_acquires+2&&cold_native_releases==initial_write_releases+2);
    assert(cold_seeds==initial_seeds+2);
    puts("Native readonly peer: late inventory closure, repeat, deferred retry and unpublished cancellation passed");
}

static VkPipeInfo volume_test_pipeline(void) {
    return (VkPipeInfo){.n_bindings=2,.pipeline=44,
        .bindings={{&sample_binding,GFX_RES_TEX,GFX_STAGE_CS,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,1,0},
                   {&volume_binding,GFX_RES_IMG,GFX_STAGE_CS,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,2,0}},
        .native_image_seed_bindings={.count=1,.entries={{.binding=2,.image_format=15,
            .scalar=RC_NATIVE_IMAGE_FLOAT32,.access=RC_NATIVE_IMAGE_SEED_WRITE,.dimension=RC_NATIVE_IMAGE_SEED_3D}}}};
}

static VkExecDispatch native_volume_dispatch(const VkExec *ex, const p_dispatch *groups, const VkPipeInfo *info) {
    VkExecDispatchImages *selection=calloc(1,sizeof *selection);
    assert(selection);
    selection->opaque_storage_effects=1;
    return (VkExecDispatch){.selection=selection,.shader=&shader,.program=ex->prog,.program_serial=ex->prog->serial,
             .registration=vk_texture_registration_publication(),.opcode=OP_DISPATCH,.payload=groups,
             .read_position=read,.info=info,.pipeline=info->pipeline};
}

static void test_volume(void) {
    const p_dispatch groups={2,3,4};
    const prog_t program={83};
    VkExec ex={.prog=&program,.fence=read.scene,.tex={{[8]=17}},.img={{[3]=31}}};
    const VkPipeInfo info=volume_test_pipeline();
    VkExecDispatch dispatch={0};
    readonly_enabled=0;
    cold_selected=NULL;
    tslots[0].ce=NULL;
    storage_shape=VK_IMAGE_VIEW_TYPE_3D;
    cold_registration.descriptor.d=(nvn_texdesc){.target=NVN_TEXTURE_TARGET_3D,.width=65,.height=33,
        .depth=9,.levels=3,.samples=1,.format=VK_FORMAT_R8_UNORM};
    cold_registration.descriptor.L.layers=1;
    const unsigned initial_seeds=cold_seeds, initial_published=writable_publications, initial_uses=writable_uses;
    const unsigned physical_acquires=cold_native_acquires;

    for (int mismatch=0;mismatch<3;++mismatch) {
        VkPipeInfo wrong=info;
        if (mismatch==0) wrong.native_image_seed_bindings.count=0;
        if (mismatch==1) wrong.native_image_seed_bindings.entries[0].dimension=RC_NATIVE_IMAGE_SEED_2D;
        if (mismatch==2) wrong.native_image_seed_bindings.entries[0].binding=3;
        const VkExecGraphicsResult expected = mismatch == 2 ? VK_EXEC_GRAPHICS_UNAVAILABLE : VK_EXEC_GRAPHICS_READY;
        assert(prepare_peer(&ex,&groups,&wrong,&dispatch)==expected);
        assert(!vk_exec_dispatch_compat_storage_binding(&dispatch));
        assert(!dispatch.selection->count&&!cold_native_hold.live&&cold_seeds==initial_seeds);
        release_peer(&dispatch);

        dispatch=native_volume_dispatch(&ex,&groups,&wrong);
        assert(!vk_exec_dispatch_compat_storage_binding(&dispatch));
        assert(dispatch_native_storage_inventory_prepare(&ex,&dispatch)==expected);
        assert(dispatch.selection->opaque_storage_effects&&!dispatch.selection->count&&!cold_native_hold.live);
        release_peer(&dispatch);
    }
    assert(cold_native_acquires==physical_acquires&&!cold_selected);

    assert(prepare_peer(&ex,&groups,&info,&dispatch)==VK_EXEC_GRAPHICS_READY);
    assert(vk_exec_dispatch_compat_storage_binding(&dispatch)==&info.bindings[1]);
    EffectImageCapture *writer=&dispatch.selection->images[0];
    assert(dispatch.selection->count==1&&dispatch.selection->storage_admission==RESOURCE_IMAGE_TRACE_STORAGE_SELECTED);
    assert(writer->written&&writer->independent&&writer->storage_seed&&!writer->full_overwrite);
    assert(writer->issued_planes.baseMipLevel==1&&writer->issued_planes.layerCount==1&&
           writer->independent->volume_mip.baseMipLevel==1&&writer->independent->volume_mip.layerCount==1);
    assert(cold_seeds==initial_seeds+1&&cold_native_hold.live&&dispatch_compat_storage_current(dispatch.selection));
    cold_mapped_cookie++;
    assert(!dispatch_compat_storage_current(dispatch.selection));
    cold_mapped_cookie--;
    assert(dispatch_compat_storage_current(dispatch.selection));
    release_peer(&dispatch);

    const GfxShaderRes peer={.slot=4,.flags=4,.count=1,.type=4};
    VkPipeInfo multiple=info;
    multiple.n_bindings++;
    multiple.bindings[2]=(VkPipeBinding){&peer,GFX_RES_IMG,GFX_STAGE_CS,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,3,0};

    // A declared output absent from the exact executable inventory is not a writer.
    const unsigned seeds_before_unused=cold_seeds;
    ex.img[0][4]=32;
    assert(prepare_peer(&ex,&groups,&multiple,&dispatch)==VK_EXEC_GRAPHICS_READY);
    assert(dispatch.selection->count==1&&dispatch.selection->images[0].slot==3&&cold_seeds==seeds_before_unused+1);
    release_peer(&dispatch);

    // If that same binding can write, its missing handle invalidates the entire cohort.
    multiple.native_image_seed_bindings.count=2;
    multiple.native_image_seed_bindings.entries[1]=multiple.native_image_seed_bindings.entries[0];
    multiple.native_image_seed_bindings.entries[1].binding=3;
    ex.img[0][4]=0;
    dispatch=native_volume_dispatch(&ex,&groups,&multiple);
    assert(dispatch_native_storage_inventory_prepare(&ex,&dispatch)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!dispatch.selection->count&&!cold_native_hold.live&&cold_native_acquires==cold_native_releases);
    assert(cold_seeds==seeds_before_unused+1&&writable_publications==initial_published);
    release_peer(&dispatch);

    ex.img[0][3]=0;
    dispatch=native_volume_dispatch(&ex,&groups,&multiple);
    const unsigned no_acquire=cold_native_acquires;
    assert(dispatch_native_storage_inventory_prepare(&ex,&dispatch)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!dispatch.selection->count&&cold_native_acquires==no_acquire);
    release_peer(&dispatch);
    ex.img[0][3]=31;

    // Storage view restrictions fail with no ordered input, no seed and no leaked capture.
    for (volume_view_override=1;volume_view_override<=4;++volume_view_override) {
        agen[0]++;
        assert(prepare_peer(&ex,&groups,&info,&dispatch)==VK_EXEC_GRAPHICS_UNAVAILABLE);
        assert(!dispatch.selection->count&&!cold_native_hold.live&&cold_seeds==initial_seeds+2);
        assert(!strcmp(dispatch.input_failure.reason,"storage-target-view"));
        release_peer(&dispatch);
    }
    volume_view_override=5; // Explicit target override to the actual 3D target is legal.
    for(int malformed=0;malformed<3;++malformed) {
        VkTexImage original=cold_entry.pub;
        if(malformed==0) cold_entry.pub.view_type=VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        if(malformed==1) cold_entry.pub.type=VK_IMAGE_TYPE_2D;
        if(malformed==2) cold_entry.f.vk=0;
        dispatch=native_volume_dispatch(&ex,&groups,&info);
        assert(vk_exec_dispatch_compat_storage_binding(&dispatch)==&info.bindings[1]);
        agen[0]++;
        assert(dispatch_native_storage_inventory_prepare(&ex,&dispatch)==VK_EXEC_GRAPHICS_UNAVAILABLE);
        assert(dispatch.selection->opaque_storage_effects&&!dispatch.selection->count&&!cold_native_hold.live);
        release_peer(&dispatch);
        cold_entry.pub=original;
        cold_entry.f.vk=VK_FORMAT_R8_UNORM;
    }

    vk_caps.robust_image_access=0;
    agen[0]++;
    assert(prepare_peer(&ex,&groups,&info,&dispatch)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!dispatch.descriptors.live&&cold_seeds==initial_seeds+2);
    release_peer(&dispatch);
    vk_caps.robust_image_access=1;

    dispatch=native_volume_dispatch(&ex,&groups,&info);
    assert(vk_exec_dispatch_compat_storage_binding(&dispatch)==&info.bindings[1]);
    assert(dispatch_native_storage_inventory_prepare(&ex,&dispatch)==VK_EXEC_GRAPHICS_READY);
    assert(!dispatch.selection->opaque_storage_effects&&dispatch.selection->count==1&&cold_native_hold.live);
    assert(compute_samples_select(&ex,&dispatch,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_READY);
    writer=&dispatch.selection->images[0];
    VkTextureStorageTarget *held=writer->independent;
    assert(held&&!writer->storage_seed&&!writer->full_overwrite);
    assert(!vk_exec_dispatch_compat_storage_binding(&dispatch));
    assert(dispatch_native_storage_inventory_prepare(&ex,&dispatch)==VK_EXEC_GRAPHICS_READY);
    assert(dispatch.selection->count==1&&dispatch.selection->images[0].independent==held);
    assert(dispatch_native_storage_peers_prepare(&ex,&dispatch)==VK_EXEC_GRAPHICS_READY);
    assert(dispatch_storage_outputs_disjoint(&ex,&dispatch));
    assert(dispatch_storage_overwrite_select(&dispatch)==VK_EXEC_GRAPHICS_READY);
    assert(!writer->full_overwrite&&!writer->storage_seed&&cold_seeds==initial_seeds+2);
    volume_seed_stale=1;
    assert(dispatch_storage_seeds_prepare(&ex,&dispatch)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!writer->storage_seed&&!writer->issued_planes.aspectMask&&!captured_seed.live&&cold_native_hold.live);
    assert(cold_seeds==initial_seeds+3);
    volume_seed_stale=0;
    assert(dispatch_storage_seeds_prepare(&ex,&dispatch)==VK_EXEC_GRAPHICS_READY);
    assert(dispatch_compat_storage_current(dispatch.selection)&&cold_seeds==initial_seeds+4);
    release_peer(&dispatch);
    assert(!cold_native_hold.live&&!lock_count&&cold_native_acquires==cold_native_releases);
    assert(writable_publications==initial_published&&writable_uses==initial_uses);
    puts("Compute volume: compatibility/native closure, full selected 3D mip seed, and refusal controls passed");
}

#if HAVE_WRITER_CHECK
static void test_actual_volume_descriptors(void) {
    GfxShaderRes second={.slot=4,.type=4,.flags=4,.count=1}, third=second, unused=second;
    third.slot=5;
    unused.slot=6;
    VkPipeInfo info=volume_test_pipeline();
    info.n_bindings=5;
    info.bindings[2]=(VkPipeBinding){&second,GFX_RES_IMG,GFX_STAGE_CS,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,3,0};
    info.bindings[3]=(VkPipeBinding){&third,GFX_RES_IMG,GFX_STAGE_CS,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,9,0};
    info.bindings[4]=(VkPipeBinding){&unused,GFX_RES_IMG,GFX_STAGE_CS,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,12,0};
    info.native_image_seed_bindings.count=3;
    info.native_image_seed_bindings.entries[1]=info.native_image_seed_bindings.entries[0];
    info.native_image_seed_bindings.entries[2]=info.native_image_seed_bindings.entries[0];
    info.native_image_seed_bindings.entries[1].binding=3;
    info.native_image_seed_bindings.entries[2].binding=9;
    DrawCtx bound={.n_storage_images=3,.storage_images={
        {.resource=&volume_binding,.handle=31,.image=309,.descriptor={.imageView=10}},
        {.resource=&second,.handle=32,.image=310,.descriptor={.imageView=11}},
        {.resource=&third,.handle=33,.image=311,.descriptor={.imageView=12}}}};
    VkExec ex={.dc=&bound,.img={{[3]=31,[4]=32,[5]=33,[6]=999}}};
    uint32_t member=99;
    uint64_t handle=99;
    assert(!dispatch_unbound_storage_writer(&ex,&info,&member,&handle));

    // The Vulkan view of the stand-in is valid. Its absent physical image makes it ineligible.
    bound.storage_images[2].image=VK_NULL_HANDLE;
    assert(dispatch_unbound_storage_writer(&ex,&info,&member,&handle)==&info.bindings[3]&&member==0&&handle==33);
    bound.storage_images[2].image=311;
    ex.img[0][5]=41;
    assert(dispatch_unbound_storage_writer(&ex,&info,&member,&handle)==&info.bindings[3]&&handle==41);
    ex.img[0][5]=33;
    bound.storage_images[2].descriptor.imageView=VK_NULL_HANDLE;
    assert(dispatch_unbound_storage_writer(&ex,&info,&member,&handle)==&info.bindings[3]);
    bound.storage_images[2].descriptor.imageView=12;
    assert(!dispatch_unbound_storage_writer(&ex,&info,&member,&handle));

    // Without executable proof the extra reflected writer cannot be presumed unused.
    info.native_image_seed_bindings.count=0;
    assert(dispatch_unbound_storage_writer(&ex,&info,&member,&handle)==&info.bindings[4]&&handle==999);
}
#endif

static void test_shared_writable_selection(void) {
    readonly_enabled=0;
    storage_shape=VK_IMAGE_VIEW_TYPE_2D;
    volume_view_override=0;
    cold_registration.descriptor.d=(nvn_texdesc){.width=65,.height=33,.depth=1,.levels=3,.samples=1,
        .format=VK_FORMAT_R8_UNORM};
    tslots[0].ce=NULL;
    cold_selected=NULL;
    cold_redirect=0;
    VkTexImage capture={0};
    view_t view={0};
    uint64_t size=0;
    int has_view=0;
    VkTextureStorageTarget *target=NULL;
    const char *reason=NULL;
    assert(vk_texture_effect_independent_storage(31,0,1,&capture,&view,&has_view,&size,&target,&reason)==
           VK_TEXTURE_STORAGE_CAPTURE_READY);
    assert(target&&cold_selected==&cold_entry&&target->serial==cold_registration.descriptor.serial&&
           target->key.addr==0x19000&&size==64&&!reason);
    vk_texture_storage_target_release(target);
    target=NULL;

    // The ordinary slot selected a live swizzle variant at the same normalized bytes.
    redirect_entry=cold_entry;
    redirect_entry.pub.image=storage_alternate;
    redirect_entry.t.serial=204;
    redirect_entry.t.d.swizzle_r=9;
    redirect_entry.key.d.swizzle_r=9;
    redirect_entry.alloc=&redirect_entry;
    redirect_entry.pub.skip=1;
    cold_redirect=1;
    agen[0]++;
    const unsigned before_resolves=slot_resolves, before_exact=exact_slot_resolves, before_refresh=refreshes;
    assert(vk_texture_effect_independent_storage(31,1,1,&capture,&view,&has_view,&size,&target,&reason)==
           VK_TEXTURE_STORAGE_CAPTURE_READY);
    assert(target&&capture.image==storage_alternate&&view.level0==1&&has_view&&!capture.skip&&!reason&&
           target->serial==204&&target->registration_serial==53&&
           target->key.d.swizzle_r==9&&target->registration_key.d.swizzle_r==0);
    assert(target->mapping_count==2&&rc_range_equal(target->mapping[0].range,cold_mapping[0].range)&&
           rc_range_equal(target->mapping[1].range,cold_mapping[1].range));
    assert(slot_resolves==before_resolves+2&&exact_slot_resolves==before_exact+1&&refreshes==before_refresh);
    view_t selected={0};
    int selected_view=0;
    vk_texture_lock();
    assert(vk_texture_storage_binding_locked(31,1,&selected,&selected_view)==&redirect_entry.pub);
    vk_texture_unlock();
    assert(selected_view==has_view&&selected.level0==view.level0&&slot_resolves==before_resolves+2&&
           refreshes==before_refresh+1&&vk_texture_storage_target_current(target));

    // Revalidation uses only the held selection and its original map/read/view.
    cold_redirect=0;
    assert(vk_texture_storage_target_current(target));
    tslots[0].cview.level0=0;
    assert(!vk_texture_storage_target_current(target));
    tslots[0].cview.level0=1;
    cold_mapped_cookie++;
    assert(!vk_texture_storage_target_current(target));
    cold_mapped_cookie--;
    cold_historical=1;
    assert(!vk_texture_storage_target_current(target));
    cold_historical=0;
    agen[0]++;
    assert(!vk_texture_storage_target_current(target)&&slot_resolves==before_resolves+2);
    lock();
    assert(storage_target_current_locked(target,1)==&redirect_entry);
    tslots[0].ce=&cold_entry;
    assert(!storage_target_current_locked(target,1));
    unlock();
    vk_texture_storage_target_release(target);
    target=NULL;

    cold_historical=1;
    assert(vk_texture_effect_independent_storage(31,1,1,&capture,&view,&has_view,&size,&target,&reason)==
           VK_TEXTURE_STORAGE_CAPTURE_UNSUPPORTED);
    assert(!target&&!strcmp(reason,"storage-target-historical-registration"));
    cold_historical=0;
    cold_redirect=1;
    agen[0]++;
    redirect_entry.key.d.width++;
    assert(vk_texture_effect_independent_storage(31,1,1,&capture,&view,&has_view,&size,&target,&reason)==
           VK_TEXTURE_STORAGE_CAPTURE_UNSUPPORTED);
    assert(!target&&!strcmp(reason,"storage-target-ambiguous-alias"));
    redirect_entry.key.d.width--;
    tslots[0].owned=&redirect_entry;
    assert(vk_texture_effect_independent_storage(31,1,1,&capture,&view,&has_view,&size,&target,&reason)==
           VK_TEXTURE_STORAGE_CAPTURE_UNSUPPORTED);
    assert(!target&&!strcmp(reason,"storage-target-ambiguous-alias"));
    assert(!cold_native_hold.live&&!lock_count&&cold_native_acquires==cold_native_releases);
    puts("Writable selection: ordinary exact alias and effective view match retained output at the original scope");
}

static void test_registered_packed_word_alias(void) {
    word_alias_enabled=word_alias_initialized=1;
    host_material=0;
    cold_redirect=1;
    cold_mapped_cookie=word_mapping.mapping;
    tslots[0].owned=NULL;
    tslots[0].ce=NULL;
    cold_registration=(TextureRegistration){
        .handle={35922},.has_view=1,.view={.set_format=1,.format=NVN_FORMAT_R32UI},
        .descriptor={.mem=(void *)(uintptr_t)0x19000,.serial=14271,
            .d={.target=NVN_TEXTURE_TARGET_2D,.format=NVN_FORMAT_RGB9E5F,.width=2580,.height=1080,.depth=1,.levels=1},
            .L={0xb64000,0xb64000,512,1,4,4,0}}};
    redirect_entry=(Tex){.pub={.type=VK_IMAGE_TYPE_2D,.view_type=VK_IMAGE_VIEW_TYPE_2D,
        .width=2580,.height=1080,.depth=1,.levels=1,.layers=1,.samples=1,.image=storage_alternate,.addr=0x19000,
        .usage=VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .aspect=VK_IMAGE_ASPECT_COLOR_BIT,.format=VK_FORMAT_B10G11R11_UFLOAT_PACK32,.gpu_owned=1},
        .t=cold_registration.descriptor,.f={.vk=122,.ideal=122},.ik={.flags=VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT},
        .content_revision=31,.alloc=&redirect_entry,.image_private_mapping_count=1,.image_private_mapping=&word_mapping};
    redirect_entry.t.serial=14158;
    redirect_entry.t.d.format=NVN_FORMAT_R11G11B10F;
    redirect_entry.t.d.swizzle_r=9;
    lock();
    make_key(&redirect_entry.t,&redirect_entry.key);
    unlock();
    cold_entry=redirect_entry; // A distinct exact-key live candidate used only for the conflicting-peer check.
    cold_entry.pub.image=storage_original;
    VkTexImage image;
    view_t view;
    int has_view=0;
    uint64_t size=0;
    VkTextureStorageTarget *target=NULL;
    const char *reason=NULL;
    assert(vk_texture_effect_storage_capture(31,1,&image,&view,&has_view,&size,&target,NULL,1,1,&reason)==
           VK_TEXTURE_STORAGE_CAPTURE_READY);
    assert(target && !reason && target->registration==35922 && target->registration_serial==14271 &&
           target->serial==14158 && image.image==storage_alternate && image.format==VK_FORMAT_B10G11R11_UFLOAT_PACK32 &&
           has_view && view.set_format && view.format==NVN_FORMAT_R32UI && size==0xb64000);
    assert(target->hold==&cold_native_hold && target->mapping_count==1 && target->mapping[0].mapping==5);
    const unsigned resolves=slot_resolves;
    const VkImageSubresourceRange plane={1,0,1,0,1}, sibling={1,1,1,0,1};
    assert(vk_texture_storage_target_load_current(target,&plane) && !vk_texture_storage_target_load_current(target,&sibling));
    word_alias_initialized=0;
    assert(!vk_texture_storage_target_current(target));
    word_alias_initialized=1;
    cold_mapped_cookie++;
    assert(!vk_texture_storage_target_current(target));
    cold_mapped_cookie--;
    word_alias_exact_active=1;
    assert(!vk_texture_storage_target_current(target));
    word_alias_exact_active=0;
    redirect_entry.t.serial++;
    assert(!vk_texture_storage_target_current(target));
    redirect_entry.t.serial--;
    cold_native_hold.live=0;
    assert(!vk_texture_storage_target_current(target));
    cold_native_hold.live=1;
    assert(vk_texture_storage_target_current(target)&&slot_resolves==resolves);
    cold_native_hold.scene_uses=1; // A previously recorded reader of the donor retains its own ownership.
    vk_texture_storage_target_release(target);
    target=NULL;
    assert(!cold_native_hold.live && cold_native_hold.scene_uses==1);
    cold_native_hold.scene_uses=0;

    for (int mode=0;mode<6;mode++) {
        word_alias_exact_active=mode==0;
        word_alias_initialized=mode!=1;
        cold_registration.descriptor.d.stride=mode==2?64:0;
        cold_registration.descriptor.L.tileH=mode==3?3:4;
        cold_registration.view.format=mode==4?NVN_FORMAT_RGB9E5F:NVN_FORMAT_R32UI;
        redirect_entry.ik.flags=mode==5?0:VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
        assert(vk_texture_effect_storage_capture(31,1,&image,&view,&has_view,&size,&target,NULL,1,1,&reason)==
               VK_TEXTURE_STORAGE_CAPTURE_UNSUPPORTED);
        assert(!target && !strcmp(reason,"storage-target-ambiguous-alias"));
    }
    // If the exact registration owns its own live 9E5 placement, issue through that physical image.
    word_alias_exact_active=word_alias_initialized=1;
    cold_registration.descriptor.d.stride=0;
    cold_registration.descriptor.L.tileH=4;
    cold_registration.view.format=NVN_FORMAT_R32UI;
    cold_redirect=0;
    cold_selected=&cold_entry;
    cold_entry.t=cold_registration.descriptor;
    cold_entry.pub.format=VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    cold_entry.f=(TexVkFormat){.vk=123,.ideal=123};
    lock();
    make_key(&cold_entry.t,&cold_entry.key);
    unlock();
    agen[0]++;
    assert(vk_texture_effect_storage_capture(31,1,&image,&view,&has_view,&size,&target,NULL,1,1,&reason)==
           VK_TEXTURE_STORAGE_CAPTURE_READY);
    assert(target->serial==14271 && image.image==storage_original && image.format==VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 &&
           view.format==NVN_FORMAT_R32UI && redirect_entry.pub.image==storage_alternate);
    const VkDescriptorImageInfo descriptor={.imageView=1234,.imageLayout=VK_IMAGE_LAYOUT_GENERAL};
    EffectImageCapture issued={.resource=&storage_binding,.handle=31,.written=1,.preserve_native=1,
        .image=image,.independent=target,.descriptor=descriptor,.issued_planes=plane};
    EffectDeclaration effects={.image_count=1,.images=&issued};
    DrawCtx bindings={.n_storage_images=1,.storage_images={{&storage_binding,31,storage_original,descriptor}}};
    VkExec exec={.dc=&bindings,.fence=read.scene};
    const unsigned writes=writable_publications;
    effect_current=&effects;
    rmw_test=1;
    rmw_predecessor=storage_original;
    lock();
    assert(dispatch_storage_outputs_begin(&exec,OP_DISPATCH));
    dispatch_storage_outputs_written(&exec,OP_DISPATCH);
    unlock();
    assert(writable_publications==writes+1 && exec.bar_dirty && cold_native_hold.scene_uses==1);
    effect_current=NULL;
    rmw_test=0;
    rmw_predecessor=VK_NULL_HANDLE;
    vk_texture_storage_target_release(target);
    assert(!lock_count && !cold_native_hold.live && cold_native_acquires==cold_native_releases);
    puts("Packed word aliases preserve held donor and exact 9E5 owner selections and issue on their own native images");
}

static void test_shared_host_material_selection(void) {
    host_material=1;
    cold_redirect=0;
    tslots[0].owned=NULL;
    tslots[0].ce=NULL;
    cold_selected=&cold_entry;
    cold_entry.pub.image=storage_original;
    cold_entry.pub.type=VK_IMAGE_TYPE_2D;
    cold_entry.pub.mirrored=1;
    cold_entry.material=cold_entry.uploaded=1;
    cold_entry.image_private_mapping_count=0;
    cold_entry.image_host_backing=&cold_entry;
    cold_entry.t=cold_registration.descriptor;
    lock();
    make_key(&cold_entry.t,&cold_entry.key);
    unlock();

    VkTexImage image={0};
    view_t view={0};
    uint64_t size=0;
    int has_view=0;
    VkTextureStorageTarget *independent=NULL;
    VkTextureWriteTarget *material=NULL;
    const char *reason=NULL;
    unsigned calls=slot_resolves;

    assert(vk_texture_effect_independent_storage(31,0,1,&image,&view,&has_view,&size,&independent,&reason)==
           VK_TEXTURE_STORAGE_CAPTURE_UNSUPPORTED);
    assert(!strcmp(reason,"storage-target-source-class")&&!host_captures&&!cold_entry.pending_write);
    assert(vk_texture_effect_storage_capture(31,0,&image,&view,&has_view,&size,&independent,&material,1,1,&reason)==
           VK_TEXTURE_STORAGE_CAPTURE_COW);
    assert(material&&!independent&&!reason&&image.image==storage_original&&size==64&&has_view);
    assert(slot_resolves==calls+1&&host_captures==1&&cold_entry.pending_write==material);

    lock();
    assert(write_target_current_locked(material));
    host_mapping=0;
    assert(!write_target_current_locked(material));
    host_mapping=1;
    host_authority=0;
    assert(!write_target_current_locked(material));
    host_authority=1;
    ++read.ordinal;
    assert(!write_target_current_locked(material));
    --read.ordinal;
    tslots[0].cview.level0++;
    assert(!write_target_current_locked(material));
    tslots[0].cview.level0--;
    cold_registration.handle.id++;
    assert(!write_target_current_locked(material));
    cold_registration.handle.id--;
    agen[0]++;
    assert(!write_target_current_locked(material));
    agen[0]--;
    assert(write_target_current_locked(material));
    vk_texture_write_target_cancel(material);
    unlock();
    material=NULL;
    assert(!cold_entry.pending_write&&!lock_count&&host_cancels==1);

    host_source=0;
    assert(vk_texture_effect_storage_capture(31,0,&image,&view,&has_view,&size,&independent,&material,1,1,&reason)==
           VK_TEXTURE_STORAGE_CAPTURE_UNSUPPORTED);
    assert(!strcmp(reason,"storage-target-material-source")&&!material&&!independent);
    host_source=1;
    cold_entry.uploaded=0;
    assert(vk_texture_effect_storage_capture(31,0,&image,&view,&has_view,&size,&independent,&material,1,1,&reason)==
           VK_TEXTURE_STORAGE_CAPTURE_UNSUPPORTED);
    assert(!strcmp(reason,"storage-target-material-cold")&&host_captures==2&&!cold_entry.pending_write);
    cold_entry.uploaded=1;
    const prog_t program={83};
    const p_dispatch groups={2,3,1};
    const VkPipeInfo info={.n_bindings=2,.pipeline=44,
        .bindings={{&sample_binding,GFX_RES_TEX,GFX_STAGE_CS,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,1,0},
                   {&storage_binding,GFX_RES_IMG,GFX_STAGE_CS,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,2,0}},
        .native_image_seed_bindings={.count=1,.entries={{.binding=2,.scalar=RC_NATIVE_IMAGE_FLOAT32,
            .access=RC_NATIVE_IMAGE_SEED_WRITE}}}};
    VkExec ex={.prog=&program,.tex={{[8]=17}},.img={{[3]=31}}};
    VkExecDispatch dispatch={0};
    assert(prepare_peer(&ex,&groups,&info,&dispatch)==VK_EXEC_GRAPHICS_READY);
    EffectImageCapture *output=&dispatch.selection->images[0];
    assert(dispatch.selection->write_plan.count==1&&dispatch.selection->count==1&&dispatch.descriptors.live&&
           output->owns_write&&output->write_target==dispatch.selection->write_plan.targets[0]&&
           output->image.image==storage_alternate&&cold_entry.pub.image==storage_original&&
           !output->independent&&!output->full_overwrite&&!output->storage_seed);
    assert(dispatch_compat_storage_current(dispatch.selection));
    release_peer(&dispatch);
    assert(!cold_entry.pending_write&&host_cancels==2);

    host_sample_conflict=1;
    assert(prepare_peer(&ex,&groups,&info,&dispatch)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!strcmp(dispatch.input_failure.reason,"material-storage-sample-mapping")&&!dispatch.descriptors.live);
    release_peer(&dispatch);
    assert(!cold_entry.pending_write&&host_cancels==3);
    puts("Compatibility host COW: prepared descriptor uses planned image, normalized sample overlap refuses");
    puts("Host storage COW selection: one selected image/view, write opt-in, original read/source/mapping revalidation");
}

static void test_comparison_storage_peer(void) {
    const RcRegistryPosition saved_read=read;
    read=(RcRegistryPosition){14449,42666};
    const p_dispatch groups={2,3,4};
    const prog_t program={83};
    VkExec ex={.prog=&program,.fence=read.scene,.tex={{[0]=comparison_handle}},.img={{[3]=31}}};
    VkPipeInfo info=volume_test_pipeline();
    info.bindings[0]=(VkPipeBinding){&comparison_binding,GFX_RES_TEX,GFX_STAGE_CS,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,5,0};
    storage_shape=VK_IMAGE_VIEW_TYPE_3D;
    volume_view_override=0;
    readonly_enabled=0;
    cold_redirect=0;
    cold_selected=NULL;
    tslots[0].ce=NULL;
    cold_registration.descriptor.d=(nvn_texdesc){.target=NVN_TEXTURE_TARGET_3D,.width=65,.height=33,
        .depth=9,.levels=3,.samples=1,.format=VK_FORMAT_R8_UNORM};
    cold_registration.descriptor.L.layers=1;
    VkExecDispatch dispatch=native_volume_dispatch(&ex,&groups,&info);
    assert(dispatch_native_storage_inventory_prepare(&ex,&dispatch)==VK_EXEC_GRAPHICS_READY);
    assert(dispatch.selection->count==1);
    EffectImageCapture *output=&dispatch.selection->images[0];
    assert(output->independent&&cold_native_hold.live);

    assert(!dispatch_storage_outputs_disjoint(&ex,&dispatch));
    if (strcmp(dispatch.input_failure.reason,"sample-receipt-missing"))
        fprintf(stderr,"b5 read %llu/%llu refused before member: %s (member %d)\n",
                (unsigned long long)read.scene,(unsigned long long)read.ordinal,
                dispatch.input_failure.reason,dispatch.input_failure.member);
    assert(dispatch.input_failure.binding==5&&dispatch.input_failure.member==0&&
           !strcmp(dispatch.input_failure.reason,"sample-receipt-missing"));
    ComputeSamples sample={.read=read,.one={.handle=comparison_handle,.image={.image=comparison_image,.format=124,
        .aspect=2,.type=VK_IMAGE_TYPE_2D,.view_type=VK_IMAGE_VIEW_TYPE_2D_ARRAY,.width=1600,.height=1600,
        .layers=3,.levels=1,.samples=1,.addr=UINT64_C(0x800193f800)},.hold=&sample,.sample={
            .registration=190,.native_slot=106,.projected=1,.has_view=1,
            .view={.aspect=2,.level0=0,.levels=1,.layer0=0,.layers=3}}}};
    dispatch.selection->samples=&sample;
    dispatch.compatibility=1;
    comparison_mapped=1;
    comparison_mapping=(RcInterval){UINT64_C(0x800193f800),88,{39,0x4820,0xf3c000}};
    assert(dispatch_storage_outputs_disjoint(&ex,&dispatch)); // Adjacent bytes in the same normalized backing.
    sample.one.sample.registration=0;
    sample.one.sample.projected=0;
    comparison_placement=(VkImageMemoryPlacement){1,4000,300};
    assert(dispatch_storage_outputs_disjoint(&ex,&dispatch));
    comparison_placement=(VkImageMemoryPlacement){1,2000,300};
    assert(!dispatch_storage_outputs_disjoint(&ex,&dispatch));
    assert(!strcmp(dispatch.input_failure.reason,"sample-comparison-mapping"));
    comparison_placement=(VkImageMemoryPlacement){1,2500,300};

    comparison_mapping.range.offset--;
    assert(!dispatch_storage_outputs_disjoint(&ex,&dispatch));
    assert(dispatch.input_failure.binding==5&&dispatch.input_failure.member==0&&
           dispatch.input_failure.peer_image==comparison_image&&
           !strcmp(dispatch.input_failure.reason,"sample-comparison-mapping"));
    comparison_mapping.range.offset++;
    assert(dispatch_storage_outputs_disjoint(&ex,&dispatch));
    comparison_mapped=0;
    assert(!dispatch_storage_outputs_disjoint(&ex,&dispatch));
    assert(!strcmp(dispatch.input_failure.reason,"sample-comparison-mapping"));
    comparison_mapped=1;
    sample.one.sample.native_slot++;
    assert(!dispatch_storage_outputs_disjoint(&ex,&dispatch));
    assert(!strcmp(dispatch.input_failure.reason,"sample-comparison-mapping"));
    sample.one.sample.native_slot--;
    cold_mapped_cookie++;
    assert(!dispatch_storage_outputs_disjoint(&ex,&dispatch));
    assert(!strcmp(dispatch.input_failure.reason,"sample-comparison-mapping"));
    cold_mapped_cookie--;
    sample.one.image.aspect=1;
    assert(!dispatch_storage_outputs_disjoint(&ex,&dispatch));
    assert(!strcmp(dispatch.input_failure.reason,"sample-comparison-twin-unprepared"));
    sample.one.image.aspect=2;
    sample.one.sample.view.layers=2;
    assert(!dispatch_storage_outputs_disjoint(&ex,&dispatch));
    assert(!strcmp(dispatch.input_failure.reason,"sample-comparison-view-unavailable"));
    sample.one.sample.view.layers=3;
    sample.one.hold=NULL;
    assert(!dispatch_storage_outputs_disjoint(&ex,&dispatch));
    assert(!strcmp(dispatch.input_failure.reason,"sample-owner-missing"));
    sample.one.hold=&sample;

    info.bindings[0].type=VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    assert(!dispatch_storage_outputs_disjoint(&ex,&dispatch));
    assert(!strcmp(dispatch.input_failure.reason,"sample-texel-provider-unimplemented"));
    info.bindings[0].type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    const GfxShaderRes unrepresented={.slot=-1,.count=1,.type=3};
    info.bindings[0].res=&unrepresented;
    assert(!dispatch_storage_outputs_disjoint(&ex,&dispatch));
    assert(!strcmp(dispatch.input_failure.reason,"sample-handle-source-unimplemented"));
    info.bindings[0].res=&comparison_binding;
    ex.tex[0][0]=0;
    assert(!dispatch_storage_outputs_disjoint(&ex,&dispatch));
    assert(!strcmp(dispatch.input_failure.reason,"sample-comparison-standin-unprepared"));
    info.bindings[0]=(VkPipeBinding){&sample_binding,GFX_RES_TEX,GFX_STAGE_CS,
        VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,1,5,0};
    assert(!ex.tex[0][8]&&dispatch_storage_outputs_disjoint(&ex,&dispatch));
    info.pool=1;
    assert(!dispatch_storage_outputs_disjoint(&ex,&dispatch));
    assert(!strcmp(dispatch.input_failure.reason,"bindless-input"));
    info.pool=0;

    dispatch.selection->samples=NULL;
    release_peer(&dispatch);
    assert(!lock_count&&!cold_native_hold.live);
    read=saved_read;
    puts("Compatibility comparison peer: b5 D16 array, adjacent normalized bytes, overlap, remap and unsupported controls passed");
}

static void test_indirect_atomic_storage(void) {
    const unsigned seed_count=cold_seeds, stopped=issue_refusals, published=writable_publications, used=writable_uses;
    const prog_t program={83};
    const uint64_t indirect_address=0x39000;
    DrawCtx cx={0};
    VkExec ex={.dc=&cx,.fence=read.scene,.prog=&program,.img={{[3]=31}}};
    const GfxShaderRes input={.slot=1};
    const VkPipeInfo info={.n_bindings=2,.pipeline=44,
        .bindings={{&storage_binding,GFX_RES_IMG,GFX_STAGE_CS,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,3,0},
                   {&input,GFX_RES_SSBO,GFX_STAGE_CS,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,4,0}},
        .native_image_rmw_seed_bindings={.count=1,.entries={{.binding=3,.image_format=33,
            .scalar=RC_NATIVE_IMAGE_UINT32,.access=RC_NATIVE_IMAGE_SEED_READ|RC_NATIVE_IMAGE_SEED_WRITE|RC_NATIVE_IMAGE_SEED_ATOMIC}}}};
    assert(cold_selected==&cold_entry&&!cold_native_hold.live);
    const Tex saved=cold_entry;
    host_material=0;
    tslots[0].ce=NULL;
    cold_entry=(Tex){.pub={.type=VK_IMAGE_TYPE_2D,.view_type=VK_IMAGE_VIEW_TYPE_2D,.width=4,.height=4,.depth=1,
        .levels=1,.layers=1,.samples=1,.image=storage_original,.addr=0x19000,.aspect=VK_IMAGE_ASPECT_COLOR_BIT,
        .usage=VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .format=VK_FORMAT_R32_UINT},.t=cold_registration.descriptor,.f={.vk=VK_FORMAT_R32_UINT,.ideal=VK_FORMAT_R32_UINT},.content_revision=73,
        .alloc=&cold_entry,.image_private_mapping_count=2,.image_private_mapping=cold_mapping};
    lock();
    make_key(&cold_entry.t,&cold_entry.key);
    unlock();
    rmw_test=1;

    VkExecDispatch control={0};
    const unsigned first_query=memo_queries, first_buffer=buffer_preparations;
    const p_dispatch direct={1,1,1};
    ex.tex[0][8]=17;
    assert(vk_exec_dispatch_compat_inputs_select(&ex,OP_DISPATCH_IND,&indirect_address,&control));
    assert(memo_queries==first_query); // Existing sampled indirect route needs no memo.
    release_peer(&control);

    ex.tex[0][8]=0;
    no_sample_reflection=1;
    assert(vk_exec_dispatch_compat_inputs_select(&ex,OP_DISPATCH,&direct,&control));
    assert(memo_queries==first_query); // Existing direct storage route needs no memo.
    release_peer(&control);
    assert(!vk_exec_dispatch_compat_inputs_select(&ex,OP_DISPATCH_IND,&indirect_address,&control));
    assert(!control.selection&&memo_queries==first_query+1); // Final source proof unavailable.
    memoized_rmw_binding=5;
    assert(!vk_exec_dispatch_compat_inputs_select(&ex,OP_DISPATCH_IND,&indirect_address,&control));
    assert(!control.selection&&memo_queries==first_query+2); // Reflected image is unrelated.
    memoized_rmw_binding=3;
    ex.img[0][3]=0;
    assert(!vk_exec_dispatch_compat_inputs_select(&ex,OP_DISPATCH_IND,&indirect_address,&control));
    assert(!control.selection&&memo_queries==first_query+3);
    ex.img[0][3]=31;

    VkPipeInfo image_only=info;
    image_only.n_bindings=1;
    VkPipeInfo unattested=image_only;
    unattested.native_image_rmw_seed_bindings.count=0;
    assert(vk_exec_dispatch_compat_inputs_select(&ex,OP_DISPATCH_IND,&indirect_address,&control) &&
        "memoized storage-only indirect must enter ordered preparation");
    assert(control.compatibility&&control.program==&program&&control.program_serial==program.serial&&
        control.registration==140&&control.payload==&indirect_address&&control.opcode==OP_DISPATCH_IND&&
        !rc_registry_position_compare(control.read_position,read)&&memo_queries==first_query+4);
    control.info=&unattested;
    control.pipeline=unattested.pipeline;
    assert(!vk_exec_dispatch_compat_storage_binding(&control));
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&control,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_READY);
    assert(control.descriptors.live&&!control.selection->count&&!control.selection->samples&&
        !cold_native_hold.live&&buffer_preparations==first_buffer+1);
    release_peer(&control);

    rmw_predecessor=storage_original;
    rmw_atomic_feature=1;
    assert(vk_exec_dispatch_compat_inputs_select(&ex,OP_DISPATCH_IND,&indirect_address,&control));
    control.info=&image_only;
    control.pipeline=image_only.pipeline;
    assert(vk_exec_dispatch_compat_storage_binding(&control)==&image_only.bindings[0]);
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&control,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_READY);
    assert(control.descriptors.live&&control.selection->count==1&&!control.selection->samples&&
        control.selection->images[0].preserve_native&&cold_native_hold.live&&buffer_preparations==first_buffer+2);
    release_peer(&control);
    assert(!cold_native_hold.live&&!cold_native_hold.scene_uses&&writable_publications==published);

    for (unsigned step=0;step<5;++step) {
        VkExecDispatch dispatch={0};
        rmw_predecessor=step==1 ? VK_NULL_HANDLE : step==2 ? storage_alternate : storage_original;
        rmw_atomic_feature=step!=0;
        rmw_input_mapping=(RcInterval){0x49000,8,{37,step==3?0x121f:0x1220,12}};
        assert(vk_exec_dispatch_compat_inputs_select(&ex,OP_DISPATCH_IND,&indirect_address,&dispatch));
        dispatch.info=&info;
        dispatch.pipeline=info.pipeline;
        assert(vk_exec_dispatch_compat_storage_binding(&dispatch)==&info.bindings[0]);
        const VkExecGraphicsResult result=vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,(RcRegistryPosition){0},NULL);
        if (step!=4) {
            assert(result==VK_EXEC_GRAPHICS_UNAVAILABLE&&!dispatch.descriptors.live&&!cold_native_hold.scene_uses);
            if (step==3) assert(!strcmp(dispatch.input_failure.reason,"retained-storage-buffer-mapping"));
            release_peer(&dispatch);
            continue;
        }
        assert(result==VK_EXEC_GRAPHICS_READY&&dispatch.descriptors.live&&dispatch.selection->count==1);
        assert(!dispatch.selection->samples&&buffer_preparations==first_buffer+3&&memo_queries==first_query+10);
        EffectImageCapture *writer=dispatch.selection->images;
        assert(writer->preserve_native&&writer->written&&!writer->full_overwrite&&!writer->storage_seed&&
            writer->issued_planes.aspectMask==VK_IMAGE_ASPECT_COLOR_BIT&&cold_native_hold.live&&cold_seeds==seed_count);

        VkMemSpanCapture command={1,{indirect_address,5,{37,0x1220,12}}};
        RawStorageInput argument={.span={155,0,12},.native=&command,.address=indirect_address,.bytes=12};
        vk_texture_lock();
        assert(dispatch_preserving_argument_disjoint(writer,&argument)); // Adjacent physical bytes are valid.
        command.interval.range.offset--;
        assert(!dispatch_preserving_argument_disjoint(writer,&argument));
        command.interval.range.offset++;
        command.current=0;
        assert(!dispatch_preserving_argument_disjoint(writer,&argument));
        command.current=1;
        argument.span.offset=2;
        assert(!dispatch_preserving_argument_disjoint(writer,&argument));

        EffectDeclaration effects={.image_count=1,.images=writer};
        effect_current=&effects;
        writer->descriptor=(VkDescriptorImageInfo){703,VK_IMAGE_LAYOUT_GENERAL};
        cx.n_storage_images=1;
        cx.storage_images[0]=(BindingImageSelection){&storage_binding,31,storage_original,writer->descriptor};
        cx.storage_images[0].descriptor.imageLayout=0;
        assert(!dispatch_storage_outputs_begin(&ex,OP_DISPATCH_IND));
        cx.storage_images[0].descriptor.imageLayout=VK_IMAGE_LAYOUT_GENERAL;
        rmw_predecessor=storage_alternate;
        assert(!dispatch_storage_outputs_begin(&ex,OP_DISPATCH_IND));
        assert(issue_refusals==stopped+2&&!cold_native_hold.scene_uses&&writable_publications==published);
        rmw_predecessor=storage_original;
        assert(dispatch_storage_outputs_begin(&ex,OP_DISPATCH_IND));
        assert(writable_uses==used+1&&cold_native_hold.scene_uses==1&&writable_publications==published);
        dispatch_storage_outputs_written(&ex,OP_DISPATCH_IND);
        assert(ex.bar_dirty&&writable_publications==published+1);
        vk_texture_unlock();
        effect_current=NULL;
        release_peer(&dispatch);
        assert(!cold_native_hold.live&&cold_native_hold.scene_uses==1);
        cold_native_hold.scene_uses--;
    }
    assert(cold_seeds==seed_count&&!cold_native_hold.live&&!lock_count);
    cold_entry=saved;
    rmw_test=rmw_atomic_feature=0;
    memoized_rmw_binding=-1;
    no_sample_reflection=0;
    rmw_predecessor=VK_NULL_HANDLE;
    rmw_input_mapping=(RcInterval){0};
    puts("Indirect storage atomic: original argument, typed feature, preserved predecessor and exact issue passed");
    puts("Memoized indirect entry: storage-only ordered prep, ordinary paths and unavailable final-proof controls passed");
}

static void dispose_stale_placement(VkImage image,uint64_t serial,uint64_t revision,int known_unhanded) {
    StoragePlacementWrite *write=malloc(sizeof *write);
    assert(write);
    *write=(StoragePlacementWrite){.key=replacement_entry.key,.image=image,.serial=serial,.revision=revision};
    storage_placement_write_dispose(&batch,write,known_unhanded);
}
static void test_stale_storage_replacement(int volume,int released,int overwritten,int reregistered) {
    stale_test_active=1;
    host_material=0;
    cold_redirect=cold_historical=0;
    word_alias_enabled=0;
    const int type=volume ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    const uint32_t view_type=volume ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
    cold_registration=(TextureRegistration){.descriptor={.mem=(void *)(uintptr_t)0x19000,.serial=345,.L={.size=64,.layers=1},
        .d={.target=volume?NVN_TEXTURE_TARGET_3D:NVN_TEXTURE_TARGET_2D,.width=8,.height=8,
            .depth=volume?4:1,.levels=volume?2:1,.samples=1,.format=VK_FORMAT_R8_UNORM}}, .handle={666}};
    cold_mapped_cookie=cold_mapping[0].mapping;
    tslots[0]=(TexSlot){.registration=&cold_registration,.used=1};
    cold_entry=(Tex){.pub={.type=type,.view_type=view_type,.width=8,.height=8,.depth=volume?4:1,
        .levels=volume?2:1,.layers=1,.samples=1,.image=storage_original,.addr=0x19000,.aspect=VK_IMAGE_ASPECT_COLOR_BIT,
        .usage=VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .format=VK_FORMAT_R8_UNORM,.gpu_owned=1}, .t=cold_registration.descriptor,
        .f={.vk=VK_FORMAT_R8_UNORM,.ideal=VK_FORMAT_R8_UNORM},.content_revision=654,.alloc=&cold_entry,
        .image_private_mapping_count=2,.image_private_mapping=cold_mapping,.released=released,.overwritten=overwritten};
    static const RcInterval prior_mapping[]={{0x19000,9,{22,0x5000,32}},{0x19020,8,{23,0x9000,32}}};
    if (reregistered) {
        cold_entry.t.serial--;
        cold_entry.image_private_mapping=prior_mapping;
    }
    cold_entry.native_output=calloc(1,sizeof *cold_entry.native_output);
    assert(cold_entry.native_output);
    cold_entry.native_output->possible=1;
    cold_entry.alias_fragment_owner=1;
    cold_entry.image_mapped_holds=&predecessor_readers;
    cold_entry.image_mapped_hold_count=2;
    cold_selected=&cold_entry;
    predecessor_readers=1;
    lock();
    make_key(&cold_entry.t,&cold_entry.key);
    unlock();
    const unsigned attempts=replacement_attempts, detached=predecessor_detached, reset=replacement_destroyed;
    VkTexImage image={0};
    view_t view={0};
    int has_view=0;
    uint64_t bytes=0;
    VkTextureStorageTarget *target=NULL;
    const char *reason=NULL;
    assert(vk_texture_effect_independent_storage(31,1,0,&image,&view,&has_view,&bytes,&target,&reason)==
           VK_TEXTURE_STORAGE_CAPTURE_INVALIDATED);
    assert(!target && replacement_attempts==attempts && cold_selected==&cold_entry);
    cold_historical=1;
    assert(vk_texture_effect_independent_storage(31,1,1,&image,&view,&has_view,&bytes,&target,&reason)==
           VK_TEXTURE_STORAGE_CAPTURE_UNSUPPORTED);
    cold_historical=0;
    stale_map_missing=1;
    assert(vk_texture_effect_independent_storage(31,1,1,&image,&view,&has_view,&bytes,&target,&reason)==
           VK_TEXTURE_STORAGE_CAPTURE_INVALIDATED);
    stale_map_missing=0;
    cold_entry.bound_to=&cold_entry;
    assert(vk_texture_effect_independent_storage(31,1,1,&image,&view,&has_view,&bytes,&target,&reason)==
           VK_TEXTURE_STORAGE_CAPTURE_UNSUPPORTED);
    cold_entry.bound_to=NULL;
    cold_entry.pending_write=(VkTextureWriteTarget *)(uintptr_t)1;
    assert(vk_texture_effect_independent_storage(31,1,1,&image,&view,&has_view,&bytes,&target,&reason)==
           VK_TEXTURE_STORAGE_CAPTURE_INVALIDATED);
    cold_entry.pending_write=NULL;
    assert(!target && replacement_attempts==attempts && cold_selected==&cold_entry);
    replacement_fail=1;
    assert(vk_texture_effect_independent_storage(31,1,1,&image,&view,&has_view,&bytes,&target,&reason)==
           VK_TEXTURE_STORAGE_CAPTURE_INVALIDATED);
    replacement_fail=0;
    replacement_invalidate=1;
    assert(vk_texture_effect_independent_storage(31,1,1,&image,&view,&has_view,&bytes,&target,&reason)==
           VK_TEXTURE_STORAGE_CAPTURE_INVALIDATED);
    replacement_invalidate=0;
    cold_mapped_cookie--;
    assert(replacement_attempts==attempts+2 && cold_selected==&cold_entry &&
           replacement_destroyed==reset && predecessor_detached==detached);
    assert(vk_texture_effect_independent_storage(31,1,1,&image,&view,&has_view,&bytes,&target,&reason)==
           VK_TEXTURE_STORAGE_CAPTURE_READY);
    assert(target && vk_texture_storage_target_replaced_stale(target) && image.image==storage_alternate &&
           image.type==type && image.preserve_pool_bytes && !image.gpu_owned && !reason);
    assert(target->serial==345 && target->registration_serial==345 && target->registration==666);
    assert(texture_release_matches(&replacement_entry,345));
    assert(!texture_release_matches(&replacement_entry,344));
    VkImageSubresourceRange planes={VK_IMAGE_ASPECT_COLOR_BIT,volume?1:0,1,0,1};
    assert(vk_texture_storage_target_current(target) && !vk_texture_storage_target_load_current(target,&planes));
    assert(predecessor_detached==detached+1 && predecessor_readers==1 && replacement_attempts==attempts+3);
    ++cold_registration.handle.id;
    assert(!vk_texture_storage_target_current(target));
    --cold_registration.handle.id;
    if (volume) {
        // Emulate issue advancing only the new placement. Held old physical content remains
        // independent, even when its late part was unhanded or the current part was handed.
        replacement_entry.content_revision=2;
        replacement_entry.pub.gpu_owned=1;
        vk_texture_storage_target_release(target);
        const unsigned part_releases=stale_part_releases;
        dispose_stale_placement(storage_original,cold_entry.t.serial,654,1);
        dispose_stale_placement(storage_alternate,344,2,1);
        dispose_stale_placement(storage_alternate,345,1,1);
        dispose_stale_placement(storage_alternate,345,2,0);
        assert(cold_selected==&replacement_entry && replacement_destroyed==reset && predecessor_readers==1);
        dispose_stale_placement(storage_alternate,345,2,1);
        assert(stale_part_releases==part_releases+5);
    } else {
        vk_texture_storage_target_release(target);
    }
    assert(!cold_selected && replacement_destroyed==reset+1 && predecessor_readers==1);
    free(cold_entry.native_output);
    predecessor_readers=0;
    stale_test_active=0;
}

int main(void) {
    cold_registration=(TextureRegistration){.descriptor={.mem=(void *)(uintptr_t)0x19000,.serial=53,.L.size=64}, .handle={91}};
    cold_mapped_cookie=cold_mapping[0].mapping;
    const prog_t program={83};
    const p_dispatch payload={2,3,1};
    const VkPipeInfo info={.n_bindings=2,.pipeline=44,
        .bindings={{&sample_binding,GFX_RES_TEX,GFX_STAGE_CS,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,1,0},
                   {&storage_binding,GFX_RES_IMG,GFX_STAGE_CS,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,2,0}},
        .native_image_seed_bindings={.count=1,.entries={{.binding=2,.scalar=RC_NATIVE_IMAGE_FLOAT32,
            .access=RC_NATIVE_IMAGE_SEED_WRITE}}}};
    VkExec ex={.prog=&program,.tex={{[8]=17}},.img={{[3]=31}}};
    VkExecDispatch dispatch={0};

    assert(!cold_selected&&!cold_native_creates);
    tex_deferred=1;
    assert(!vk_exec_dispatch_compat_inputs_select(&ex,OP_DISPATCH,&payload,&dispatch));
    assert(!dispatch.selection&&!cold_selected&&!cold_native_creates);
    assert(vk_texture_deferred_take()==1&&!vk_texture_deferred());
    cold_refused=1;
    assert(vk_exec_dispatch_compat_inputs_select(&ex,OP_DISPATCH,&payload,&dispatch));
    dispatch.info=&info;
    dispatch.pipeline=info.pipeline;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_READY);
    assert(dispatch.selection->storage_admission==RESOURCE_IMAGE_TRACE_STORAGE_TARGET&&
           !dispatch.selection->count&&!dispatch.selection->images&&!vk_texture_deferred());
    assert(!cold_selected&&!cold_native_creates&&!cold_native_acquires&&!cold_seeds);
    compute_samples_release(&dispatch.selection->samples);
    draw_descriptor_grant_release(&dispatch.descriptors);
    free(dispatch.selection);
    cold_refused=0;
    const unsigned before_absence=sample_releases;
    cold_absent=1;
    assert(prepare_peer(&ex,&payload,&info,&dispatch)==VK_EXEC_GRAPHICS_READY);
    assert(!dispatch.selection->count&&!cold_native_hold.live);
    release_peer(&dispatch);
    cold_absent=0;
    assert(sample_releases==before_absence+1);
    cold_refused=2;
    assert(prepare_peer(&ex,&payload,&info,&dispatch)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!dispatch.selection->count&&!cold_native_hold.live&&
           !strcmp(dispatch.input_failure.reason,"storage-target-placement-unavailable"));
    release_peer(&dispatch);
    cold_refused=0;

    assert(vk_exec_dispatch_compat_inputs_select(&ex,OP_DISPATCH,&payload,&dispatch));
    assert(dispatch.compatibility&&dispatch.selection&&!cold_selected&&!cold_native_hold.live);
    dispatch.info=&info;
    dispatch.pipeline=info.pipeline;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_READY);
    assert(dispatch.selection->storage_admission==RESOURCE_IMAGE_TRACE_STORAGE_SELECTED&&dispatch.descriptors.live);
    assert(cold_selected&&cold_native_creates==1&&cold_native_acquires==1&&cold_native_hold.live&&cold_seeds==1);
    assert(dispatch.selection->count==1&&dispatch.selection->images[0].independent->registration==91);
    assert(dispatch.selection->samples->one.image.image==77&&dispatch.selection->samples->one.hold);
    cold_mapped_cookie++;
    assert(!dispatch_compat_storage_current(dispatch.selection));
    cold_mapped_cookie--;
    cold_registration.handle.id++;
    assert(!dispatch_compat_storage_current(dispatch.selection));
    cold_registration.handle.id--;
    assert(dispatch_compat_storage_current(dispatch.selection));
    dispatch_compat_storage_release(dispatch.selection);
    compute_samples_release(&dispatch.selection->samples);
    draw_descriptor_grant_release(&dispatch.descriptors);
    free(dispatch.selection);
    assert(!cold_native_hold.live&&cold_native_releases==1&&sample_releases==4);

    conflicting_peer=1;
    assert(vk_exec_dispatch_compat_inputs_select(&ex,OP_DISPATCH,&payload,&dispatch));
    dispatch.info=&info;
    dispatch.pipeline=info.pipeline;
    assert(vk_exec_dispatch_prepare_compat_inputs(&ex,&dispatch,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!strcmp(dispatch.input_failure.reason,"sample-view-conflict")&&!dispatch.descriptors.live);
    assert(cold_native_creates==1&&cold_native_acquires==2&&cold_seeds==1);
    dispatch_compat_storage_release(dispatch.selection);
    compute_samples_release(&dispatch.selection->samples);
    free(dispatch.selection);
    assert(!cold_native_hold.live&&cold_native_releases==2&&sample_releases==5&&!lock_count);
    puts("Normal compatibility selection: cold target refusal, exact target, disjoint sample, held seed, remap and cancellation passed");
    conflicting_peer=0;
    test_readonly_storage_peer();
    test_native_readonly_storage_peer();
    test_volume();
#if HAVE_WRITER_CHECK
    test_actual_volume_descriptors();
#endif
    test_comparison_storage_peer();
    test_shared_writable_selection();
    test_shared_host_material_selection();
    test_indirect_atomic_storage();
    test_registered_packed_word_alias();
    test_stale_storage_replacement(0,1,1,0);
    test_stale_storage_replacement(0,1,1,1);
    test_stale_storage_replacement(0,1,0,1);
    test_stale_storage_replacement(1,0,1,1);
    test_stale_storage_replacement(1,1,0,1);
    test_stale_storage_replacement(1,1,1,0);
    puts("Dormant 2D/3D writes: new and reused registration, remap, old finalizer, refusal and cancellation passed");
}
'''

begin = texture.index("struct VkTextureStorageTarget {")
end = texture.index("\n};", begin) + len("\n};")
trace_end = texture.index("} TextureStorageRefusal;") + len("} TextureStorageRefusal;")
trace_key_end = texture.index("} TextureStorageRefusalKey;")
trace_begin = texture.rindex("typedef struct {", 0, trace_key_end)
trace = texture[trace_begin:trace_end] + r"""
int resource_effects_image_trace_enabled(void) { return 0; }
static void storage_target_trace_refusal(const TextureStorageRefusal *trace,int result,const char *why) {
    (void)trace;(void)result;(void)why;assert(0); // This fixture keeps tracing disabled.
}
"""
lower = ("same_but_swizzle", "texture_binding_entry", "texture_slot_select_locked", "texture_slot_selected_view",
         "texture_storage_select_locked", "vk_texture_storage_binding_locked", "storage_volume_shape",
         "storage_volume_mip", "storage_target_shape", "storage_target_shape_refusal", "storage_target_volume_view",
         "storage_target_identity_current", "write_target_storage_current_locked", "storage_trace_key_relation",
         "storage_trace_refresh_result", "storage_trace_effective", "storage_stale_predecessor", "storage_target_replace_stale_locked",
         "vk_texture_effect_storage_capture", "vk_texture_storage_target_replaced_stale",
         "vk_texture_effect_independent_storage",
         "storage_target_registration_locked", "storage_target_current_locked", "vk_texture_storage_target_bound_plan",
         "vk_texture_storage_target_current", "vk_texture_storage_target_load_current",
         "vk_texture_storage_targets_disjoint", "texture_placement_disjoint", "vk_texture_storage_target_disjoint_input",
         "vk_texture_storage_target_release", "vk_texture_storage_target_read_use")
word_helpers = ("storage_word_view_plane", "storage_word_format", "storage_word_alias_initialized", "storage_word_alias_capture")
if all(name in texture for name in word_helpers):
    index = lower.index("storage_target_identity_current") + 1
    lower = lower[:index] + word_helpers + lower[index:]
common = ("effect_image_lookup", "dispatch_storage_seed_current", "dispatch_inputs_match", "dispatch_compat_read_current", "dispatch_compat_storage_release",
          "dispatch_storage_store_format",
          "dispatch_storage_seed_binding", "dispatch_storage_rmw_binding", "dispatch_storage_indirect_rmw", "dispatch_storage_binding_shape", "dispatch_storage_binding_proof", "dispatch_storage_seed_read_format",
          "dispatch_storage_access_refused", "dispatch_storage_overwrite_select",
          "dispatch_storage_image_access", "dispatch_compat_storage_inventory",
          "dispatch_native_storage_volume_pending", "vk_exec_dispatch_compat_storage_binding",
          "dispatch_storage_read_peer_prepare",
          "dispatch_compat_storage_current", "compute_sample_members", "dispatch_storage_conflict", "dispatch_storage_sample_view_disjoint",
          "dispatch_storage_mirror_input_disjoint", "dispatch_storage_material_sample_disjoint", "effect_write_plan_add",
          "dispatch_preserving_buffer_disjoint",
          "dispatch_preserving_storage_inputs", "dispatch_storage_output_disjoint", "dispatch_storage_outputs_disjoint",
          "dispatch_storage_seed_sources_current", "dispatch_storage_seeds_prepare")
entry = ("tex_info", "dispatch_preserving_argument_disjoint", "dispatch_storage_outputs_begin", "dispatch_storage_outputs_written",
         "dispatch_storage_inventory_capture", "dispatch_bound_storage_prepare", "dispatch_compat_storage_prepare",
         "dispatch_native_storage_inventory_prepare", "dispatch_native_storage_peers_prepare",
         "dispatch_compat_indirect_storage_candidate", "vk_exec_dispatch_compat_inputs_select",
         "vk_exec_dispatch_prepare_compat_inputs")
check_writer = "static const VkPipeBinding *dispatch_unbound_storage_writer(" in draw
if check_writer:
    binder = function(draw, "dispatch_bind_selected_resources")
    assert binder.index("bind_resources(ex, info)") < binder.index("dispatch_unbound_storage_writer(")
    direct = function(draw, "dispatch_direct_emit")
    assert direct.index("if (!bound) return;") < direct.index("vkCmdDispatch(")
    common = tuple(item for name in common for item in (
        (name, "compute_storage_binding_unused", "dispatch_unbound_storage_writer") if name == "dispatch_storage_seed_binding"
        else ("dispatch_storage_volume_required", name) if name == "dispatch_native_storage_volume_pending"
        else (name,)))
placement_decl = texture[texture.rfind("typedef struct {", 0, texture.index("} StoragePlacementWrite;")):
                         texture.index("} StoragePlacementWrite;") + len("} StoragePlacementWrite;")]
source = (f"#define HAVE_WRITER_CHECK {int(check_writer)}\n" +
    PREFIX + ALLOCATION + texture[begin:end] + trace + "\n".join(function(texture, name) for name in lower) +
    function(draw, "dispatch_storage_view_planes") + function(draw, "dispatch_storage_planes") + SEED +
    "\n".join(function(draw, name) for name in common) +
    "\n".join(function(draw, name) for name in entry) + placement_decl + "\n" +
    function(texture, "storage_placement_write_dispose") + function(texture,"texture_release_matches") + CASES
)
compiler = Path(os.environ["CC"])
env = dict(os.environ, PATH=str(compiler.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="cold-storage-compat-") as folder:
    path = Path(folder) / "check.c"
    exe = path.with_suffix(".exe")
    path.write_text(source, encoding="utf-8")
    subprocess.run([str(compiler), "-std=gnu11", "-Wall", "-Wextra", "-Werror", "-O1", "-fsanitize=undefined",
                    "-fsanitize-trap=all", f"-I{headers}", str(path), str(headers / "rc_native_image_store.c"),
                    "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

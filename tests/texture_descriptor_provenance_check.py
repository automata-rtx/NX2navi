"""The published image identity must describe the descriptor's actual view, including stand-ins."""
from pathlib import Path
import os
import subprocess
import tempfile
from bind_recipe_check import function

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ["CC"])
source = (ROOT / "runtime/vk_exec_draw.c").read_text()
prefix = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
typedef uint64_t VkImage, VkImageView, VkSampler;
typedef int VkImageLayout;
#define VK_NULL_HANDLE 0
#define VK_IMAGE_LAYOUT_GENERAL 1
#define VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT 2
#define VK_IMAGE_ASPECT_DEPTH_BIT 1
#define VK_IMAGE_ASPECT_STENCIL_BIT 2
#define VK_IMAGE_VIEW_TYPE_3D 3
#define NVN_HANDLE_TEX(handle) ((int)((handle)&0xfffff))
enum { VK_TEXTURE_REGISTRATION_CURRENT, VK_TEXTURE_REGISTRATION_EXACT,
       VK_TEXTURE_REGISTRATION_ABSENT=-1, VK_TEXTURE_REGISTRATION_GONE=-2 };
typedef struct {VkSampler sampler;VkImageView imageView;VkImageLayout imageLayout;} VkDescriptorImageInfo;
typedef struct {uint16_t type;int flags,exact,format,pool;} GfxShaderRes;
typedef struct {int dummy;} view_t;
typedef struct {VkImage image;int skip,aspect;} VkTexImage;
typedef struct {int unused;} ViewKey;
typedef struct {int mo_ovr_on;} TestDraw;
typedef struct {int nfbl;TestDraw *dc;uint64_t fence;} VkExec;
enum {NVN_STAGE_VS, NVN_STAGE_CS};
typedef struct {int live;} VkTexturePresentSource;
typedef struct {VkImage image;} VkTextureAliasRead;
typedef struct {uint64_t registration;int native_slot;uint8_t canonical,refused,unbacked,projected;} VkTextureReadKey;
typedef struct VkTextureQueryImage VkTextureQueryImage;
typedef struct {const VkTexImage *image;view_t view;int has_view,native_slot,historical;uint64_t registration;
    int canonical,projected;VkTextureQueryImage *query_image;} VkTextureSample;
static VkImageView vk_texture_query_image_view(const VkExec *ex,const VkTextureQueryImage *query,VkImage *image){
    (void)ex;(void)query;(void)image;assert(0);return 0;
}
static int sampled_rgba8_linear_request(const VkTexImage *image,const view_t *view,int hv,uint16_t type){
    (void)image;(void)view;(void)hv;(void)type;return 0;
}
typedef enum {VK_TEXTURE_SAMPLE_READY,VK_TEXTURE_SAMPLE_FALLBACK,VK_TEXTURE_SAMPLE_GONE,
              VK_TEXTURE_SAMPLE_ABSENT,VK_TEXTURE_SAMPLE_CANONICAL_REFUSED,VK_TEXTURE_SAMPLE_PREPARE,
              VK_TEXTURE_SAMPLE_DEFERRED,VK_TEXTURE_SAMPLE_UNAVAILABLE,VK_TEXTURE_SAMPLE_UNBACKED} VkTextureSampleStatus;
enum {VK_TEXTURE_READ_REQUIRED=1,VK_TEXTURE_READ_FORCE_COMPAT=2};
static int vk_dlaa_swap_img,twin_dispatch, required_unavailable;
static int twin_compare_resource(const GfxShaderRes *r){assert(r&&!twin_dispatch);return 0;}
static void twin_capture_view(const VkTexImage *image,const view_t *view,int has_view,uint64_t handle,
    const GfxShaderRes *r,VkTextureReadKey key,int historical,VkDescriptorImageInfo *out) {
    (void)has_view; (void)key; (void)historical; assert(image&&view&&handle&&r&&out&&!twin_dispatch);
}
static void vk_exec_required_graphics_unavailable(VkExec *ex,int op,const char *why) {
    assert(ex&&!op&&why);required_unavailable++;
}
static struct {struct {VkImageView view;} img[1];VkSampler samp_shadow,samp;} Z={{{9}},5,6};
static VkTexImage texture={7,2,0};
static const VkTexImage old_texture={18,0,0};
static const VkTexImage archived_child={19,0,0};
static int archived_child_ready;
static int history,raw_lookups,slot_offset,notes,failed,typed_ready,typed_refused,source_failures,selections;
static int gpa_pending, deferred, pending_requests;
static int vk_texture_deferred(void) {return deferred;}
static int vk_texture_sample_wait_gpa_locked(VkExec *ex,int logical) {
    assert(ex&&logical==1);
    if (!gpa_pending) return 0;
    pending_requests++;
    return deferred=1;
}
static void vk_exec_registration_gone(VkExec *ex,int logical){assert(ex&&logical==1);failed++;}
static int available=1,view_available=1;
static VkImage selected_image=7;
static int flat3d_pool_consuming;
static VkTextureSampleStatus vk_texture_history_flat3d_locked(int logical,const VkTexImage **out,void *plan){
    assert(logical==1&&!plan&&history==VK_TEXTURE_REGISTRATION_EXACT);
    *out=archived_child_ready?&archived_child:NULL;
    return *out?VK_TEXTURE_SAMPLE_READY:VK_TEXTURE_SAMPLE_PREPARE;
}
static const VkTexImage *vk_texture_by_id_locked(int id,view_t *v,int *hv);
typedef struct {int unused;} VkTextureSampleRefusal;
static void sampled_binding_refusal(VkExec *ex, int stage, uint32_t member, const GfxShaderRes *resource, uint64_t handle,
    VkTextureSampleStatus status, const VkTextureSample *sample, const VkTextureSampleRefusal *refused,
    int geometry, const char *reason) {
    (void)ex; (void)stage; (void)member; (void)resource; (void)handle; (void)status;
    (void)sample; (void)refused; (void)geometry; (void)reason;
    if (status==VK_TEXTURE_SAMPLE_CANONICAL_REFUSED) source_failures++;
}
static VkTextureSampleStatus vk_texture_resolve_sample_report_locked(int id,unsigned flags,VkTextureSample *out, VkTextureSampleRefusal *refused){
    (void)refused;
    assert(id==1);selections++;
    *out=(VkTextureSample){.native_slot=1+slot_offset,.historical=history};
    if(history==VK_TEXTURE_REGISTRATION_EXACT){out->image=&old_texture;return VK_TEXTURE_SAMPLE_READY;}
    if(history==VK_TEXTURE_REGISTRATION_GONE)return VK_TEXTURE_SAMPLE_GONE;
    if(history==VK_TEXTURE_REGISTRATION_ABSENT)return VK_TEXTURE_SAMPLE_ABSENT;
    if(typed_refused)return VK_TEXTURE_SAMPLE_CANONICAL_REFUSED;
    if(typed_ready){out->image=&texture;out->registration=382;out->canonical=!(flags&VK_TEXTURE_READ_FORCE_COMPAT);return VK_TEXTURE_SAMPLE_READY;}
    out->image=vk_texture_by_id_locked(out->native_slot,&out->view,&out->has_view);
    return VK_TEXTURE_SAMPLE_FALLBACK;
}
static int vk_texture_registration_expected_locked(int id,VkImage *image){assert(id==1&&!image);return history;}
static int issue_reconstructed, reconstruction_view_current=1, prepared_sample_state, view_compatible=1;
static VkTexImage reconstructed={41,0,0};
static VkTextureAliasRead alias_input={41};
static int compute_sample(VkExec *ex,uint32_t member,const GfxShaderRes *resource,uint64_t handle,
                          VkTextureSample *out,VkTextureAliasRead **alias,const VkTexturePresentSource **source) {
    (void)ex; (void)member; (void)resource; (void)handle; (void)out; (void)alias; (void)source; return 0;
}
static VkTexturePresentSource exact_source={1};
static int exact_source_uses, alias_uses;
static int vk_texture_record_selected_sample_locked(const VkTexturePresentSource *source,VkImage image) {
    assert(source==&exact_source && source->live && image==7); ++exact_source_uses; return 1;
}
static void vk_texture_alias_read_use(VkTextureAliasRead *alias,uint64_t fence) {
    assert(alias==&alias_input && fence==31); ++alias_uses;
}
static int graphics_sample(VkExec *ex,int stage,uint32_t member,const GfxShaderRes *resource,uint64_t handle,
                           VkTextureSample *out,VkTextureAliasRead **alias,const VkTexturePresentSource **source) {
    assert(ex&&stage==NVN_STAGE_VS&&!member&&resource&&handle==1);
    if (prepared_sample_state == 2) {
        *out=(VkTextureSample){.image=&texture,.canonical=1,.native_slot=9,.registration=382};
        *source=&exact_source;
        return 1;
    }
    if (prepared_sample_state == 3) {
        *out=(VkTextureSample){.native_slot=9,.registration=382};
        return 1;
    }
    if (prepared_sample_state == 1) {
        *out=(VkTextureSample){.historical=VK_TEXTURE_REGISTRATION_ABSENT,.native_slot=-1};
        return 1;
    }
    if (prepared_sample_state == -1) return -1;
    if (!issue_reconstructed) return 0;
    *out=(VkTextureSample){.image=&reconstructed,.registration=483,.native_slot=9};
    *alias=&alias_input;
    return 1;
}
static int texture_view_key(const VkTexImage *image,const view_t *view,int hv,uint16_t type,int storage,ViewKey *key) {
    assert(image&&view&&key&&!hv&&!storage); (void)type; return view_compatible;
}
static int comparison_capable, comparison_checks;
static int pool_comparison_format(const ViewKey *key,uint64_t *features) {
    assert(key&&features);comparison_checks++;
    *features=comparison_capable;
    return comparison_capable;
}
static VkImageView alias_sample_view_locked(VkTextureAliasRead *alias,const VkTexImage *image,const view_t *view,
                                           int has_view,uint16_t type) {
    assert(alias==&alias_input&&image->image==alias->image&&view&&!has_view&&!type);
    return reconstruction_view_current?43:0;
}
static int registered_slot_locked(int id){return id+slot_offset;}
static void vk_texture_registration_note_locked(int id,const VkTexImage *image,const view_t *view,int hv){
    assert(id==1+slot_offset&&image&&view&&!hv);notes++;
}
static const VkTexImage *vk_texture_by_id_locked(int id,view_t *v,int *hv){
    assert(id==1+slot_offset);raw_lookups++;*v=(view_t){0};*hv=0;return available?&texture:NULL;
}
static const VkTexImage *vk_texture_exact_by_id_locked(int id,view_t *v,int *hv){return vk_texture_by_id_locked(id,v,hv);}
static const VkTexImage *vk_texture_writable_locked(const VkTexImage *i){return i;}
static VkImageView get_view_actual(VkExec *e,const VkTexImage *i,const view_t *v,int hv,uint16_t type,int storage,VkImage *actual,int required,VkTextureReadKey key){
    (void)key;
    (void)e;(void)v;(void)hv;(void)type;(void)storage;(void)required;
    *actual=i==&old_texture||i==&archived_child?i->image:selected_image;return view_available?8:0;
}
static int exact_child;
static int vk_texture_flat3d_selected_current_locked(VkTextureReadKey key,VkImage image){
    return exact_child&&key.registration&&image==17;
}
static int in_fbl(VkExec *e,VkImage i){(void)e;return i==7;}
static int vk_exec_view_aspect(const VkTexImage *i,const view_t *v,int hv){(void)i;(void)v;(void)hv;return 0;}
static VkSampler vk_sampler_point_by_id_locked(int id){(void)id;return 3;}
static VkSampler vk_sampler_compare_by_id_locked(int id){(void)id;return 3;}
static VkSampler vk_sampler_detail_by_id_locked(int id,int skip){(void)id;(void)skip;return 3;}
static VkSampler vk_sampler_by_id_locked(int id){(void)id;return 3;}
static int res_view_type(uint16_t type){return type==3?VK_IMAGE_VIEW_TYPE_3D:0;}
static int dummy_index(int type,int ms){(void)type;(void)ms;return 0;}
static VkImageView storage_dummy(int di,int format){(void)di;(void)format;return 10;}
'''
main = r'''
int main(void){
    TestDraw dc={0};VkExec ex={.nfbl=1,.dc=&dc,.fence=31};GfxShaderRes r={0};VkDescriptorImageInfo descriptor;VkImage image;
    assert(tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,0,NULL,NULL)==1);
    assert(image==7 && descriptor.imageView==8 && descriptor.imageLayout==2);
    view_available=0;
    assert(!tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,0,NULL,NULL));
    assert(!image && descriptor.imageView==9 && descriptor.imageLayout==1);
    assert(!tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,1,0,&descriptor,&image,1,NULL,NULL));
    assert(!image && descriptor.imageView==10 && !descriptor.sampler && descriptor.imageLayout==1);
    available=0;view_available=1;
    assert(!tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,0,&descriptor,&image,1,NULL,NULL));
    assert(!image && descriptor.imageView==9 && descriptor.imageLayout==1);
    available=1;
    assert(tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,1,NULL,NULL)==1);
    assert(image==7 && descriptor.imageView==8 && descriptor.imageLayout==2);
    selected_image=17;
    assert(tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,1,NULL,NULL)==1);
    assert(image==17 && descriptor.imageView==8 && descriptor.imageLayout==1);
    int previous=raw_lookups;history=VK_TEXTURE_REGISTRATION_EXACT;
    assert(!tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,1,NULL,NULL));
    assert(image==18 && descriptor.imageView==8 && raw_lookups==previous);
    history=VK_TEXTURE_REGISTRATION_ABSENT;
    assert(!tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,1,NULL,NULL));
    assert(!image && descriptor.imageView==9 && raw_lookups==previous);
    assert(!tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,1,0,&descriptor,&image,1,NULL,NULL));
    assert(!image && descriptor.imageView==10 && raw_lookups==previous && !failed);
    history=VK_TEXTURE_REGISTRATION_GONE;
    descriptor.imageView=31;
    assert(!tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,1,NULL,NULL));
    assert(!image && descriptor.imageView==31 && raw_lookups==previous && failed==1);
    history=VK_TEXTURE_REGISTRATION_EXACT;
    assert(!tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,1,0,&descriptor,&image,1,NULL,NULL));
    assert(!image && descriptor.imageView==31 && raw_lookups==previous && failed==2);
    history=0;slot_offset=8;selected_image=7;
    assert(tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,0,NULL,NULL)==1);
    assert(image==7&&raw_lookups==previous+1&&notes);
    typed_ready=1;
    VkTextureReadKey chosen;
    for(int demanded=0;demanded<=1;demanded++){
        assert(tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,demanded,&chosen,NULL)==1);
        assert(chosen.registration==382&&chosen.native_slot==9&&chosen.canonical&&image==7&&raw_lookups==previous+1);
    }
    VkTextureSample cache={0};
    int before=selections;
    assert(tex_info_full_sample(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,0,&chosen,NULL,&cache,NULL)==1);
    r.type=1;
    assert(tex_info_full_sample(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,0,&chosen,NULL,&cache,NULL)==1);
    assert(selections==before+1&&cache.canonical&&chosen.canonical&&image==7);
    r.type=3;selected_image=17;
    assert(tex_info_full_sample(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,0,&chosen,NULL,&cache,NULL)==1);
    assert(selections==before+2&&!cache.canonical&&!chosen.registration&&image==17);
    exact_child=1;
    assert(tex_info_full_sample(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,0,&chosen,NULL,&cache,NULL)==1);
    assert(selections==before+3&&!cache.canonical&&chosen.registration==382&&chosen.canonical&&image==17);
    // Selected ordinary views consume one exact sample, independent of any late selector/cache state.
    r.type=0; selected_image=7;
    const int captured_at=selections;
    VkTextureSample retained={.image=&texture,.registration=382,.native_slot=9,.canonical=1};
    assert(tex_info_full_sample(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,0,&chosen,&retained,&cache,NULL)==1);
    assert(image==7&&chosen.registration==382&&selections==captured_at);
    view_available=0;
    assert(!tex_info_full_sample(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,0,&chosen,&retained,&cache,NULL));
    assert(chosen.refused&&!image&&selections==captured_at);
    view_compatible=0;
    assert(!tex_info_full_sample(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,0,&chosen,&retained,&cache,NULL));
    assert(!chosen.refused&&!image&&selections==captured_at);
    view_available=view_compatible=1;
    r.type=0x403;r.pool=1;
    assert(!tex_info_full_sample(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,0,&chosen,&retained,&cache,NULL));
    assert(!image&&chosen.refused&&comparison_checks==1&&selections==captured_at);
    comparison_capable=1;
    assert(tex_info_full_sample(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,0,&chosen,&retained,&cache,NULL)==1);
    assert(image==7&&!chosen.refused&&comparison_checks==2&&selections==captured_at);
    r.type=0;r.pool=0;
    retained.image=NULL;
    assert(!tex_info_full_sample(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,0,&chosen,&retained,&cache,NULL));
    assert(chosen.unbacked&&chosen.registration==382&&!image&&selections==captured_at);
    prepared_sample_state=1;
    assert(!tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,1,&chosen,NULL));
    assert(!image&&!chosen.registration&&selections==captured_at);
    prepared_sample_state=3;
    assert(!tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,1,&chosen,NULL));
    assert(!image&&chosen.unbacked&&chosen.registration==382&&selections==captured_at);
    prepared_sample_state=-1;
    assert(!tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,1,&chosen,NULL));
    assert(selections==captured_at); // Missing prepared receipt cannot reopen ordinary slot lookup.
    prepared_sample_state=2;
    int recorded=0, selected_notes=notes;
    assert(tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,1,&chosen,&recorded)==1);
    assert(recorded&&exact_source_uses==1&&notes==selected_notes&&selections==captured_at);
    view_available=0;
    assert(!tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,0,&chosen,&recorded));
    assert(!recorded&&exact_source_uses==1); // no descriptor was issued with the selected image
    view_available=1;
    prepared_sample_state=0;
    r.type=3;
    cache=(VkTextureSample){0};history=VK_TEXTURE_REGISTRATION_EXACT;
    int failures_before=failed;
    assert(!tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,1,&chosen,NULL)&&failed==failures_before+1&&!image);
    archived_child_ready=1;
    assert(!tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,1,&chosen,NULL)&&image==19&&failed==failures_before+1);
    assert(raw_lookups==previous+1);
    history=0;
    r.type=0;selected_image=7;
    typed_refused=1;
    assert(!tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,0,&chosen,NULL));
    assert(chosen.refused&&!source_failures&&!image&&raw_lookups==previous+1);
    assert(!tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,1,&chosen,NULL));
    assert(chosen.refused&&source_failures==1&&!image&&raw_lookups==previous+1);
    gpa_pending=1;
    assert(!tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,1,&chosen,NULL));
    assert(deferred&&pending_requests==1&&source_failures==1&&!chosen.refused&&!image);
    assert(!tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,1,&chosen,NULL));
    assert(deferred&&pending_requests==1&&source_failures==1&&!chosen.refused&&!image);
    deferred=0;gpa_pending=0;issue_reconstructed=1;
    const int prior_notes=notes;
    assert(!tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,1,&chosen,NULL));
    assert(image==41&&descriptor.imageView==43&&descriptor.imageLayout==VK_IMAGE_LAYOUT_GENERAL&&alias_uses==1);
    assert(!chosen.registration&&notes==prior_notes); // A reconstruction supplies the draw, with no guest write.
    reconstruction_view_current=0;
    assert(!tex_info_full(&ex,NVN_STAGE_VS,0,&r,1,0,1,&descriptor,&image,1,&chosen,NULL));
    assert(!image&&descriptor.imageView==43&&notes==prior_notes&&required_unavailable==1); // Stale view cannot switch to a stand-in.
    puts("descriptor provenance: incompatible/absent views retain only stand-in identity; valid views retain exact image");
}
'''
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
work = ROOT / "build/test"
work.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="descriptor-provenance-", dir=work) as temporary:
    path = Path(temporary) / "check.c"
    definition = source[source.index("static int tex_info_full_sample("):]
    path.write_text(prefix + function(definition,"tex_info_full_sample") + function(definition,"tex_info_full") + main)
    exe = path.with_suffix(".exe")
    subprocess.run([str(CC), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fsanitize-trap=all", str(path), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

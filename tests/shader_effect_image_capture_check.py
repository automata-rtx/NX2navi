"""Exercise production specialization image captures and final reflection agreement without Vulkan."""
from pathlib import Path
import os, re, subprocess, tempfile
from bind_recipe_check import function

ROOT=Path(__file__).resolve().parents[1]
CC=Path(os.environ['CC'])
source=(ROOT/'runtime/vk_exec_draw.c').read_text()
def extract(name):
    match=re.search(r'^[^\n]*\b'+name+r'\([^;{]*\)\s*\{',source,re.M)
    assert match,name
    return function(source[match.start():],name)
prefix=r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
typedef uint64_t VkImage, VkImageView, VkSampler;
typedef unsigned VkImageViewType;
static const struct {VkSampler samp;} Z={8};
typedef unsigned VkFormat;
#define VK_FORMAT_UNDEFINED 0
#define VK_NULL_HANDLE 0
#define VK_IMAGE_LAYOUT_GENERAL 1
#define GFX_RES_IMG 1
#define GFX_RES_TEX 2
#define NVN_HANDLE_TEX(h) ((int)(h))
#define NVN_STAGE_CS 0
#define OP_DISPATCH 5
#define VK_IMAGE_TYPE_2D 3
#define VK_IMAGE_VIEW_TYPE_2D 3
#define VK_IMAGE_VIEW_TYPE_2D_ARRAY 4
#define VK_IMAGE_USAGE_STORAGE_BIT 8
#define VK_IMAGE_USAGE_SAMPLED_BIT 16
#define VK_IMAGE_ASPECT_COLOR_BIT 1
#define RC_EFFECT_WRITES_UNKNOWN 4
typedef enum {VK_EXEC_DISPATCH_EFFECT_STOPPED=-1,VK_EXEC_DISPATCH_EFFECT_COMPAT,VK_EXEC_DISPATCH_EFFECT_READY} VkExecDispatchEffectResult;
typedef struct {int active;} ResourceImageScope;
static void resource_image_scope_begin(ResourceImageScope*s,uint64_t f){assert(f);s->active=1;}
static void resource_image_scope_end(ResourceImageScope*s){(void)s;}
typedef struct {unsigned type,format,exact,flags,count,pool,separate;int slot;} GfxShaderRes;
typedef struct {const GfxShaderRes *r;int n,pool;const GfxShaderRes *sampled;int n_sampled;} GfxShader;
typedef struct {uint8_t tex_type[32],img_type[32],img_format[32];} GfxShaderAnswers;
typedef struct {unsigned format,set_format,set_levels,base_level,set_layers,min_layer;} view_t;
typedef struct {uint64_t image,addr;unsigned skip,format,type,gone,samples,usage,levels,layers,block_views,aspect;} VkTexImage;
static const VkTexImage *vk_dlaa_swap_img;
typedef struct {uint64_t sampler,imageView;unsigned imageLayout;} VkDescriptorImageInfo;
typedef struct {int narrowed,native_bindings;} EffectInputCapture;
typedef struct {unsigned aspectMask,baseMipLevel,levelCount,baseArrayLayer,layerCount;} VkImageSubresourceRange;
typedef struct {uint64_t pin,image;} VkPinnedTextureWrite;
typedef struct {uint64_t image;unsigned held,viewed,bound,began;int serial;} VkTextureWriteTarget;
typedef struct VkTextureStorageTarget VkTextureStorageTarget;
typedef struct VkTextureAliasRead VkTextureAliasRead;
typedef struct {VkTextureWriteTarget **targets;uint32_t count;} VkTextureWritePlan;
typedef struct {unsigned held;uint64_t image;} VkTexturePresentSource;
static VkTexturePresentSource historical_read;
static unsigned sampled_uses;
typedef enum {VK_TEXTURE_STORAGE_GONE,VK_TEXTURE_STORAGE_PINNED,VK_TEXTURE_STORAGE_TARGET,VK_TEXTURE_STORAGE_NEW_TARGET}
    VkTextureStorageResult;
typedef struct {const GfxShaderRes *resource;uint64_t handle,size;int stage,slot,written,has_view,owns_write,sampled;VkTextureWriteTarget *write_target;VkTexturePresentSource *read_hold;view_t view;VkTexImage image;VkDescriptorImageInfo descriptor;VkPinnedTextureWrite pinned;VkTextureStorageTarget *independent;VkTextureAliasRead *storage_seed;} EffectImageCapture;
typedef struct DeviceStoragePointer DeviceStoragePointer;
typedef struct VkExecDispatchImages {EffectImageCapture *images;uint32_t count;VkTextureWritePlan write_plan;DeviceStoragePointer *pointer_sources;uint32_t pointer_count;void *samples,*selected_writers;int opaque_storage_effects,storage_admission;} VkExecDispatchImages;
static void device_storage_pointers_free(DeviceStoragePointer *sources,uint32_t count){assert(!sources&&!count);}
static void compute_samples_release(void **samples){assert(samples && !*samples);}
static void selected_storage_writers_release(void **writers){assert(writers && !*writers);}
typedef struct {EffectImageCapture *images;uint32_t image_count,opcode;const GfxShader *shader[8];EffectInputCapture *capture[8];int gpa_unknown,native_bindings;void *commands;VkTextureWritePlan write_plan;DeviceStoragePointer *pointer_sources;uint32_t pointer_count; struct VkExecBufferScratchReceipt *buffer_scratch;
    void *selected_writers,*compute_samples;int selected_writers_owned,storage_admission;
} EffectDeclaration;
typedef struct {uint64_t serial;} Program;
typedef struct {uint64_t img[8][32],tex[8][128],fence,cb;void *dc;Program *prog;int in_pass;struct {unsigned width;} tonemap_extent;} VkExec;
typedef struct {uint64_t scene,ordinal;} RcRegistryPosition;
typedef struct {int unused;} ResourceCommandEffects;
static EffectDeclaration *effect_current;
static struct {unsigned revoked_inputs,opaque_stages;} effect_stats;
static int shader_has_storage_writer(const GfxShader *s){assert(s);return 0;}
static void effect_declare_cold_storage(EffectDeclaration *d,int stage){(void)d;(void)stage;abort();}
static int dispatch_storage_planes(const EffectImageCapture *capture,VkImageSubresourceRange *range,uint32_t extent[2]){
    (void)capture;(void)range;(void)extent;abort();
}
static unsigned locks,reads,writes,views,opaque,order[16],norder,retired_views,pins,releases,shader_reads,publishes;
static int live=1,allow_first,allow_second,pair_disjoint=1;
static VkTextureWriteTarget output[2];
static unsigned first_canceled,first_recorded,draw_unavailable;
static int storage_first_write_probe;
static uint64_t storage_first_write_count,storage_sample_alias_count;
static int result=1,stopping,metadata_available=1;
static VkTexImage selected={.image=11,.addr=0x1000,.format=4,.type=VK_IMAGE_TYPE_2D,.samples=1,
                           .usage=VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,
                           .levels=2,.layers=2,.aspect=VK_IMAGE_ASPECT_COLOR_BIT};
static const GfxShader *translated;
static int gfx_shader_resources(const GfxShader*s,int kind,const GfxShaderRes**r){
    static GfxShaderRes pool={.pool=1};
    if(kind==GFX_RES_IMG){*r=s->r;return s->n;}
    *r=s->pool?&pool:s->sampled;return s->pool?1:s->n_sampled;
}
static int find_stage(void*a,void*b,int s,const uint8_t**ctl,const uint8_t**code,uint32_t*len){(void)a;(void)b;(void)s;*ctl=*code=NULL;*len=0;return 1;}
static void vk_texture_lock(void){assert(!locks);locks++;}
static void vk_texture_unlock(void){assert(locks==1);locks--;}
static int answers_for(VkExec*x,int s,const uint8_t*c,unsigned q,GfxShaderAnswers*a){(void)x;(void)s;(void)c;for(int i=0;i<32;i++)if(q&(1u<<i))a->img_type[i]=9;return 1;}
static int capture(int id,int exact,VkTexImage*i,view_t*v,int*hv,uint64_t*size,int written){assert(locks);(void)exact;order[norder++]=written;writes+=written;reads+=!written;if(result<1)return result;*i=selected;i->addr+=(uint64_t)(id-7)*0x1000;*v=(view_t){4,1};*hv=1;*size=64;return 1;}
static int vk_texture_effect_image(int id,int e,VkTexImage*i,view_t*v,int*h,uint64_t*z,VkPinnedTextureWrite*p){assert(!p->pin);return capture(id,e,i,v,h,z,1);}
static int vk_texture_effect_sample(int id,int e,VkTexImage*i,view_t*v,int*h,uint64_t*z){return capture(id,e,i,v,h,z,0);}
enum { VK_TEXTURE_STORAGE_CAPTURE_ABSENT=0, VK_TEXTURE_STORAGE_CAPTURE_READY=1 };
static int vk_texture_effect_independent_storage(int id,int exact,int written,VkTexImage*i,view_t*v,int*h,uint64_t*z,
                                                 VkTextureStorageTarget **out,const char **reason){
    (void)id;(void)exact;(void)written;(void)i;(void)v;(void)h;(void)z;(void)reason;assert(!*out);return VK_TEXTURE_STORAGE_CAPTURE_ABSENT;
}
static int vk_texture_storage_target_current(const VkTextureStorageTarget *t){(void)t;abort();}
static VkImage vk_texture_storage_target_image(const VkTextureStorageTarget *target,const VkTextureAliasRead *seed){
    (void)target;(void)seed;abort();
}
static void vk_texture_storage_target_release(VkTextureStorageTarget *t){assert(!t);}
static void vk_texture_alias_read_cancel(VkTextureAliasRead *seed){assert(!seed);}
static int capture_ready_pin(int id,VkTexImage*i,view_t*v,int*h,uint64_t*z,VkPinnedTextureWrite*p){
    assert(locks && !p->pin);if(id==9)return 0;
    *i=selected;*v=(view_t){4,1};*h=1;*z=64;*p=(VkPinnedTextureWrite){id,i->image};pins++;return 1;
}
static void vk_texture_pinned_write_release(VkPinnedTextureWrite*p){if(!p->pin)return;assert(pins);*p=(VkPinnedTextureWrite){0};pins--;releases++;}
static VkTextureStorageResult vk_texture_effect_ordered_image(int id,int written,const VkTextureWritePlan *plan,
    VkTexImage*i,view_t*v,int*h,uint64_t*z,VkPinnedTextureWrite*p,VkTextureWriteTarget **out){
    assert(locks);
    int number=id==9?0:id==10&&allow_second?1:-1;
    if(number<0)return capture_ready_pin(id,i,v,h,z,p)?VK_TEXTURE_STORAGE_PINNED:VK_TEXTURE_STORAGE_GONE;
    if(!allow_first)return VK_TEXTURE_STORAGE_GONE;
    *i=selected;i->addr=0x3000+(uint64_t)number*0x1000;*v=(view_t){4,1};*h=1;*z=64;
    for(uint32_t k=0;k<plan->count;k++)if(plan->targets[k]->serial==id){
        assert(plan->targets[k]==&output[number] && output[number].held);
        *out=plan->targets[k];return VK_TEXTURE_STORAGE_TARGET;
    }
    if(!written)return VK_TEXTURE_STORAGE_GONE;
    assert(!output[number].held);output[number]=(VkTextureWriteTarget){.held=1,.serial=id};*out=&output[number];
    return VK_TEXTURE_STORAGE_NEW_TARGET;
}
static int vk_texture_effect_ordered_sample(int logical,const VkTextureWritePlan *plan,VkTexImage *image,view_t *v,
    int *hv,VkTextureWriteTarget **out,VkTexturePresentSource **hold){
    *image=selected;
    *v=(view_t){4,1};*hv=1;
    if(logical==19){
        assert(!historical_read.held);
        historical_read=(VkTexturePresentSource){1,88};
        *hold=&historical_read;
        image->image=historical_read.image;image->addr=0x3000;return 1;
    }
    for(uint32_t i=0;i<plan->count;i++)if(plan->targets[i]->serial==logical){
        *out=plan->targets[i];return 1;
    }
    return 0;
}
static int vk_texture_transfer_read_use(const VkTexturePresentSource *source,uint64_t image){
    assert(locks && effect_current && source->held && source->image==image);sampled_uses++;return 1;
}
static void vk_texture_present_release(VkTexturePresentSource *source){if(source){assert(source->held);source->held=0;}}
static VkSampler vk_sampler_by_id_locked(int id){assert(locks);return id?(VkSampler)id+10:0;}
static void vk_texture_write_target_cancel(VkTextureWriteTarget *target){
    if(!target)return;
    assert((target==&output[0] || target==&output[1]) && target->held && !target->began);
    target->held=0;first_canceled++;
}
static uint64_t vk_texture_write_target_prepared_image(const VkTextureWriteTarget *target){
    assert(locks && target->held && !target->began);return target->image;
}
static void vk_texture_write_target_viewed(VkTextureWriteTarget *target){
    assert(locks && !effect_current && target->held && !target->began);target->viewed++;
}
static int vk_texture_write_target_bound_current(const VkTextureWriteTarget *target){
    assert(locks);return target->bound && target->held && !target->began;
}
static const VkTexImage *vk_texture_write_target_begin(VkTextureWriteTarget *target, uint64_t command){
    assert(locks && effect_current && command && target->viewed && target->image && target->bound && !target->began);
    static VkTexImage actual;actual.image=target->image;target->began=1;return &actual;
}
static void vk_texture_write_target_written(VkTextureWriteTarget *target){
    assert(effect_current && target->began && target->held);target->held=0;first_recorded++;
}
static void vk_exec_required_graphics_unavailable(VkExec *ex,uint32_t opcode,const char *why){assert(ex&&opcode&&why);draw_unavailable++;}
static void vk_dlaa_hud_write(uint64_t image){
    assert((image==output[0].image && output[0].began) || (image==output[1].image && output[1].began));
}
static int shader_skipped(const GfxShader *shader){(void)shader;return 0;}
static int vk_texture_write_plan_independent(const VkTextureWritePlan *plan){return plan->count<2 || pair_disjoint;}
static int vk_texture_write_plan_disjoint_pin(const VkTextureWritePlan *plan,const VkPinnedTextureWrite *pin){
    assert(plan->count && output[0].held);return pin->pin && pin->pin!=9;
}
static int vk_texture_pinned_write_matches_locked(const VkPinnedTextureWrite*p,uint64_t image){return locks && p->pin && p->image==image;}
static int gfx_type_of(const VkTexImage*i,const view_t*v,int h){(void)v;(void)h;return i->type;}
static int int_view_format(unsigned f,VkFormat*out){if(out)*out=f;return f;}
static unsigned texel_bytes(unsigned f){return f;}
enum ShaderAnswerResource { SHADER_ANSWER_TEXTURE, SHADER_ANSWER_IMAGE, SHADER_ANSWER_COMPUTE_IMAGE };
static uint8_t storage_image_format_answer(const VkTexImage *image,const view_t *view,int hv,int compute) {
    (void)compute;
    VkFormat format;
    return hv && view->set_format && int_view_format(view->format,&format) && texel_bytes(format) &&
        texel_bytes(format)==texel_bytes(image->format) ? (uint8_t)view->format : 0;
}
static int vk_texture_image_live_locked(const VkTexImage*i){assert(locks&&i->image);return live;}
static VkImageView get_view(VkExec*x,const VkTexImage*i,const view_t*v,int h,unsigned type,int storage){
    assert(locks && (storage || i->image==52 || i->image==88));
    (void)x;(void)v;(void)h;(void)type;views++;retired_views+=i->gone;return i->image+100;
}
static void fixed_require(void){} // Fixture stand-ins are already ready.
static unsigned res_view_type(unsigned t){return t;}
static unsigned dummy_index(unsigned t,int a){return t+(unsigned)a;}
static VkImageView storage_dummy(unsigned d,unsigned f){return 900+d+f;}
static void resource_effects_opaque(void*c){(void)c;opaque++;}
static void rt_fatal(const char*s){fputs(s,stderr);abort();}
static uint32_t slot_mask_of(void*dc,uint64_t serial,int stage){(void)dc;assert(serial && stage==NVN_STAGE_CS);return 3;}
static void shader_answer_local(const uint8_t*ctl,GfxShaderAnswers*a){(void)ctl;(void)a;}
static int answer_of_handle(uint64_t h,enum ShaderAnswerResource image,uint8_t*t,uint8_t*f,int metadata_only){
    assert(locks && metadata_only);*t=image && h?9:0;*f=0;return metadata_available;
}
static const GfxShader *shader_for_resolve_answers(VkExec*x,int stage,int*present,int*pending,uint32_t wait,const GfxShaderAnswers*a){
    (void)x;assert(!locks && stage==NVN_STAGE_CS && !pending && !wait && a);
    if (++shader_reads % 2 == 0) assert(a->img_type[0]==VK_IMAGE_TYPE_2D && a->img_format[0]==4);
    *present=1;return translated;
}
static unsigned gfx_shader_guest_writes(const GfxShader*s){(void)s;return 1;}
static int gfx_shader_write_inventory_complete(const GfxShader*s){(void)s;return 1;}
static void gfx_shader_slot_dependencies(const GfxShader*s,uint32_t*t,uint32_t*i){(void)s;*t=0;*i=1;}
static int vk_sync_stopping(void){return stopping;}
static int vk_device_is_lost(void){return 0;}
static int vk_exec_recording_failed(void){return 0;}
static EffectDeclaration *effect_declaration_new(VkExec*x,RcRegistryPosition p){(void)x;assert(p.scene && p.ordinal);return calloc(1,sizeof(EffectDeclaration));}
static ResourceCommandEffects *resource_effects_enter(void*p){assert(!p);return NULL;}
static void resource_effects_leave(ResourceCommandEffects*p){assert(!p);}
static void effect_stage_publish(EffectDeclaration*d,VkExec*x,int st,uint32_t op,const void*p,const GfxShader*s,uint32_t first){
    (void)x;(void)p;assert(st==NVN_STAGE_CS && !first && s==translated && d->image_count>=2);
    assert(op==OP_DISPATCH && d->opcode==op);
    assert(d->images[0].written);
    if(d->write_plan.count) {
        for(uint32_t i=0;i<2;i++){
            const EffectImageCapture *image=&d->images[i];
            assert(image->write_target && image->write_target->viewed && !image->write_target->began);
            assert(image->image.image==image->write_target->image && image->descriptor.imageView==image->image.image+100);
        }
    } else {
        assert(d->images[0].pinned.pin==8 && d->images[0].image.image==11);
        assert(d->images[0].descriptor.imageView==111);
    }
    publishes++;
}
typedef struct VkExecBufferScratchReceipt VkExecBufferScratchReceipt;
static VkExecBufferScratchReceipt *buffer_scratch_adopt(VkExecBufferScratchReceipt **scratch,
    const VkExec *ex, uint32_t op, const void *payload, const GfxShader *shader, RcRegistryPosition position) {
    (void)ex; (void)payload; (void)position;
    assert(scratch && !*scratch && op == OP_DISPATCH && shader == translated);
    return NULL;
}
static void effect_declaration_finish(EffectDeclaration*d){assert(!d->gpa_unknown);}
'''
body='\n'.join(extract(n) for n in ('effect_images_truncate','effect_write_plan_add','effect_stage_image_inputs','effect_stage_image_match',
    'effect_stage_image_views','effect_command_images_compatible','effect_validate_command_images',
    'effect_ordered_image_supported','effect_first_write_compatible','effect_first_write_samples','vk_exec_dispatch_effect_prepare',
    'vk_exec_dispatch_effect_cancel','vk_exec_effects_declare_dispatch','dispatch_first_write_begin',
    'dispatch_first_write_recorded'))
tests=r'''
int main(void){
 VkExecBufferScratchReceipt *scratch=NULL;
 VkExec ex={.fence=1};ex.img[0][0]=7;ex.img[0][1]=8;
 GfxShaderRes r[2]={{.slot=0,.type=3},{.slot=1,.type=3,.flags=4}};
 GfxShader initial={r,2,0};EffectDeclaration d={0};GfxShaderAnswers a={0};
 assert(effect_stage_image_inputs(&d,&ex,0,&initial,1,&a,0)==1);
 assert(d.image_count==2&&order[0]==1&&order[1]==0&&reads==1&&writes==1&&views==0);
 assert(a.img_type[0]==3&&a.img_format[0]==4&&d.images[1].size==64);
 // Slot replacement after the capture cannot affect the selected physical view or specialization answers.
 selected.image=77;selected.type=6;ex.img[0][0]=9;
 assert(effect_stage_image_match(&d,0,&initial,0,1));effect_stage_image_views(&d,&ex,0,0);
 assert(d.images[1].image.image==11&&d.images[1].descriptor.imageView==111&&a.img_type[0]==3);
 live=0;effect_stage_image_views(&d,&ex,0,0);assert(retired_views==2);live=1;
 // Final exact-level requirements and output inventories must agree with every retained record.
 d.images[1].image.skip=1;r[0].exact=1;assert(!effect_stage_image_match(&d,0,&initial,0,1));
 d.images[1].image.skip=0;r[0].exact=0;initial.n=1;assert(!effect_stage_image_match(&d,0,&initial,0,1));initial.n=2;
 r[1].flags=0;assert(!effect_stage_image_match(&d,0,&initial,0,1));r[1].flags=4;
 free(d.images);d=(EffectDeclaration){0};ex.img[0][0]=0;initial.n=1;
 assert(effect_stage_image_inputs(&d,&ex,0,&initial,1,&a,0)==1&&d.image_count==1&&!d.images[0].handle);
 ex.img[0][0]=7;assert(effect_stage_image_match(&d,0,&initial,0,1));effect_stage_image_views(&d,&ex,0,0);
 assert(!d.images[0].image.image&&d.images[0].descriptor.imageView>=900);
 free(d.images);d=(EffectDeclaration){0};result=-1;
 assert(effect_stage_image_inputs(&d,&ex,0,&initial,1,&a,0)==-1&&!locks);
 free(d.images);d=(EffectDeclaration){0};result=1;initial.pool=1;
 assert(effect_stage_image_inputs(&d,&ex,0,&initial,1,&a,0)==-2&&!d.image_count);initial.pool=0;
 // A later-stage COW must revoke all forced stages, not just the readonly half of the command.
 EffectImageCapture pair[2]={{.image={11,0x1000},.size=64},{.written=1,.image={22,0x1020},.size=64}};
 EffectInputCapture inputs[2]={{1,1},{1,1}};d=(EffectDeclaration){.images=pair,.image_count=2,.shader={&initial,&initial},.capture={&inputs[0],&inputs[1]}};
 effect_validate_command_images(&d);assert(!d.image_count&&!d.shader[0]&&!d.shader[1]&&!inputs[0].narrowed&&!inputs[1].narrowed&&!inputs[0].native_bindings&&!inputs[1].native_bindings&&d.gpa_unknown&&opaque==1);
 d.image_count=2;d.gpa_unknown=0;pair[1].image.image=11;effect_validate_command_images(&d);assert(d.image_count==2&&!d.gpa_unknown&&opaque==1);
 pair[1].image.image=22;pair[1].image.addr=0x2000;effect_validate_command_images(&d);assert(d.image_count==2&&opaque==1);
 // Prepare checks reflection and retains actual providers without touching native descriptor or effect state.
 Program program={1};ex.prog=&program;initial.n=2;translated=&initial;
 selected.image=11;selected.type=VK_IMAGE_TYPE_2D;
 VkExecDispatchImages *receipt=NULL;const GfxShader *shader=NULL;
 unsigned pre_views=views,pre_reads=reads,pre_writes=writes;
 assert(vk_exec_dispatch_effect_prepare(&ex,0,NULL,&shader,&receipt)==VK_EXEC_DISPATCH_EFFECT_READY);
 assert(receipt->count==2 && pins==2 && shader_reads==2 && !publishes && shader==translated);
 assert(views==pre_views && reads==pre_reads && writes==pre_writes);
 // The compatibility cache and current slot now refer elsewhere. The exact receipt must still commit its image.
 live=0;selected.image=99;ex.img[0][0]=9;ex.img[0][1]=10;
 EffectDeclaration *commit=vk_exec_effects_declare_dispatch(&ex,OP_DISPATCH,NULL,shader,&receipt,(RcRegistryPosition){1,2},&scratch);
 assert(!receipt && pins==2 && views==pre_views+2 && publishes==1 && shader_reads==2);
 assert(commit->images[1].image.image==11 && commit->images[1].descriptor.imageView==111 && retired_views==2);
 effect_images_truncate(commit,0);free(commit->images);free(commit);assert(!pins && releases==2);

 // An unavailable required input rolls back a partial capture. Cancellation of a complete receipt is idempotent.
 assert(vk_exec_dispatch_effect_prepare(&ex,0,NULL,&shader,&receipt)==VK_EXEC_DISPATCH_EFFECT_COMPAT);
 assert(!receipt && !shader && !pins && releases==3 && publishes==1);
 ex.img[0][0]=7;shader_reads=0;
 assert(vk_exec_dispatch_effect_prepare(&ex,0,NULL,&shader,&receipt)==VK_EXEC_DISPATCH_EFFECT_READY);
 vk_exec_dispatch_effect_cancel(&receipt);vk_exec_dispatch_effect_cancel(&receipt);
 assert(!pins && releases==5 && !receipt && publishes==1 && views==pre_views+2);
 metadata_available=0;
 assert(vk_exec_dispatch_effect_prepare(&ex,0,NULL,&shader,&receipt)==VK_EXEC_DISPATCH_EFFECT_COMPAT && !pins);
 metadata_available=1;allow_first=1;shader_reads=0;selected.image=11;
 ex.img[0][0]=ex.img[0][1]=9;ex.cb=2;
 assert(vk_exec_dispatch_effect_prepare(&ex,0,NULL,&shader,&receipt)==VK_EXEC_DISPATCH_EFFECT_READY);
 assert(receipt->write_plan.count==1 && receipt->write_plan.targets[0]==&output[0] && output[0].held);
 assert(receipt->images[0].write_target==receipt->images[1].write_target && !pins && !output[0].began);
 vk_exec_dispatch_effect_cancel(&receipt);assert(first_canceled==1 && !output[0].held);
 ex.img[0][1]=10;r[0].flags=4;allow_second=1;pair_disjoint=0;
 assert(vk_exec_dispatch_effect_prepare(&ex,0,NULL,&shader,&receipt)==VK_EXEC_DISPATCH_EFFECT_COMPAT);
 assert(!receipt && first_canceled==3 && !output[0].held && !output[1].held);
 pair_disjoint=1;
 GfxShaderRes sampled[2]={{.slot=2,.type=3},{.slot=3,.type=3,.separate=1}};
 initial.sampled=sampled;initial.n_sampled=2;ex.tex[0][2]=9;ex.tex[0][3]=19;
 assert(vk_exec_dispatch_effect_prepare(&ex,0,NULL,&shader,&receipt)==VK_EXEC_DISPATCH_EFFECT_READY);
 assert(receipt->count==4 && receipt->images[2].write_target==&output[0] && historical_read.held);
 assert(receipt->write_plan.count==2 && receipt->write_plan.targets[0]!=receipt->write_plan.targets[1]);
 output[0].image=52;output[1].image=53;
 selected.image=199;ex.img[0][0]=ex.img[0][1]=55;
 commit=vk_exec_effects_declare_dispatch(&ex,OP_DISPATCH,NULL,shader,&receipt,(RcRegistryPosition){1,4},&scratch);
 assert(commit->images[0].image.image==52 && commit->images[1].image.image==53 && !output[0].began && !output[1].began);
 assert(commit->images[2].image.image==52 && commit->images[2].descriptor.sampler==Z.samp);
 assert(commit->images[3].image.image==88 && !commit->images[3].descriptor.sampler && !sampled_uses);
 assert(dispatch_first_write_begin(&ex,OP_DISPATCH) && !output[0].began); // phase has not entered
 effect_current=commit;output[0].bound=1;
 assert(!dispatch_first_write_begin(&ex,OP_DISPATCH) && draw_unavailable==1 && !output[0].began && !output[1].began);
 output[1].bound=1;ex.in_pass=1;
 assert(!dispatch_first_write_begin(&ex,OP_DISPATCH) && draw_unavailable==2 && !output[0].began && !output[1].began);
 ex.in_pass=0;
 storage_first_write_probe=1;
 assert(dispatch_first_write_begin(&ex,OP_DISPATCH) && output[0].began && output[1].began && sampled_uses==1);
 assert(storage_sample_alias_count==1);
 dispatch_first_write_recorded();
 assert(first_recorded==2 && !commit->write_plan.count && storage_first_write_count==2);
 effect_current=NULL;effect_images_truncate(commit,0);free(commit->write_plan.targets);free(commit->images);free(commit);
 assert(!output[0].held && !output[1].held && !historical_read.held && first_canceled==3 && first_recorded==2);
 stopping=1;
 assert(vk_exec_dispatch_effect_prepare(&ex,0,NULL,&shader,&receipt)==VK_EXEC_DISPATCH_EFFECT_STOPPED && !pins);
 puts("storage image effects: exact answers, ordinal-free preparation, retained replacement, cancellation and compatibility passed");
}
'''
env=dict(os.environ,PATH=str(CC.parent)+os.pathsep+os.environ.get('PATH',''))
with tempfile.TemporaryDirectory(prefix='shader-image-capture-',dir=ROOT/'build/test') as tmp:
    p=Path(tmp)/'check.c';p.write_text(prefix+body+tests);exe=p.with_suffix('.exe')
    subprocess.run([str(CC),'-std=gnu11','-O1','-Wall','-Wextra','-Werror','-Wno-missing-field-initializers','-fsanitize=undefined','-fsanitize-trap=all',str(p),'-o',str(exe)],env=env,check=True)
    subprocess.run([str(exe)],env=env,check=True)

"""Production tensor replacement output capture, exact bounds and partial-write checks; no Vulkan/game run."""
from pathlib import Path
import os
import subprocess
import tempfile
from bind_recipe_check import function

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ['CC'])
draw = (ROOT / 'runtime/vk_exec_draw.c').read_text()
dlaa = (ROOT / 'runtime/vk_dlaa.c').read_text()
texture = (ROOT / 'runtime/vk_texture.c').read_text()

prefix = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "resource_core_registry.h"
typedef uint64_t VkImage;
typedef uint64_t VkCommandBuffer;
typedef unsigned VkImageAspectFlags;
typedef struct {int32_t x,y,z;} VkOffset3D;
typedef struct {uint32_t width,height,depth;} VkExtent3D;
typedef struct {unsigned aspectMask,mipLevel,baseArrayLayer,layerCount;} Subresource;
typedef struct {Subresource srcSubresource;VkOffset3D srcOffset;Subresource dstSubresource;VkOffset3D dstOffset;VkExtent3D extent;} VkImageCopy;
#define VK_IMAGE_ASPECT_COLOR_BIT 1
#define VK_IMAGE_ASPECT_DEPTH_BIT 2
#define VK_IMAGE_TYPE_2D 2
#define VK_IMAGE_LAYOUT_GENERAL 1
#define VK_AA_GAME 0
#define VK_AA_ROLE_NONE 0
#define NVN_STAGE_CS 0
#define NVN_HANDLE_TEX(h) ((int)(h))
#define RESOURCE_IMAGE_SCOPE(name,fence) (void)(fence)
typedef struct {
    unsigned set_format,format,set_levels,base_level,nlevels,set_layers,min_layer,nlayers,set_target,target,set_swizzle,set_dsmode;
} view_t;
typedef struct {VkImage image;uint64_t addr;unsigned width,height,aspect,type,samples,format,gone,levels,layers,skip;} VkTexImage;
typedef struct {uint32_t width,height;} VkExtent2D;
typedef Subresource VkImageSubresourceLayers;
typedef struct {Subresource srcSubresource,dstSubresource; VkOffset3D srcOffsets[2],dstOffsets[2];} VkImageBlit;
#define NVN_TEXTURE_TARGET_2D 2
#define VK_TEXTURE_COLOR_BLIT_MAPPINGS 8
typedef struct {
    VkImage image;
    void *hold;
    RcRegistryPosition read;
    uint64_t address,bytes,serial;
    uint32_t format;
    VkExtent2D extent;
    uint32_t mapping_count;
    RcInterval mapping[VK_TEXTURE_COLOR_BLIT_MAPPINGS];
} VkTextureColorBlitSource;
typedef struct {void *mem;struct {uint64_t size;} L;} tex_t;
typedef struct {const tex_t *pin;VkImage image;} VkPinnedTextureWrite;
typedef struct {VkImage image;unsigned width,height,level,base_layer,format;const tex_t *tex;const VkTexImage *img;} VkExecTarget;
typedef struct {void *owner;} ResourceImageEffectRole;
typedef struct VkTextureProjectedTransfer VkTextureProjectedTransfer;
typedef struct VkDlaaSceneEffects {VkImage images[2];uint64_t addresses[2];RcSourceRange ranges[2];RcRegistryImageBinding owners[2];ResourceImageEffectRole border[2];VkTextureProjectedTransfer *edge[2];int complete;} VkDlaaSceneEffects;
typedef struct {VkTexImage output;uint64_t size;VkDlaaSceneEffects scene;VkTextureColorBlitSource blit;} EffectFallbackCapture;
typedef struct {uint64_t serial;} Program;
typedef struct {void *dc;Program *prog;uint64_t img[8][32],fence;VkCommandBuffer cb;unsigned recorded;} VkExec;
typedef struct {void *commands;EffectFallbackCapture *fallback;Program *program;uint64_t program_serial;RcRegistryPosition position;} EffectDeclaration;
typedef struct Tex {
    VkTexImage pub;
    struct {struct {uint64_t size;} L;uint64_t serial;struct {unsigned width,height,target,format;} d;} t;
    RcRegistryWrite image_authority;
    struct Tex *next_at;
    const RcInterval *image_private_mapping;
    uint32_t image_private_mapping_count,image_mapped_hold_count;
    int independent;
} Tex;
typedef struct { int unused; } VkTextureAliasLayout;
static int image_alias_material_layout(const Tex *e,VkTextureAliasLayout *layout){assert(layout);return e->independent;}
static unsigned color_holds;
static void *vk_mem_image_hold(VkImage image){assert(image);color_holds++;return (void*)(uintptr_t)image;}
static void vk_mem_image_release(void *hold){if(hold){assert(color_holds);color_holds--;}}
static RcRegistryPosition read_position={1,2};
static RcRegistryPosition vk_exec_resource_position(void){return read_position;}
static void texture_color_blit_capture(const Tex *,const view_t *,int,VkTextureColorBlitSource *);
static RcRegistry *registry;
#define rc_registry_runtime() registry
static Tex *owners, *slots[32];
static int by_addr,locks,turns,pins,touches,prepared,deferred,role,mode,fallback_on=1,capture_ok=1;
static uint8_t control[256];
static unsigned pin_images[64],pin_count;
static void lock(void){locks++;}
static void unlock(void){assert(locks);locks--;}
static void vk_texture_lock(void){lock();}
static void vk_texture_unlock(void){unlock();}
static Tex *nvn_hm_get(void *map,const uint64_t *addr){(void)map;(void)addr;return owners;}
static void touch(Tex *e){assert(locks&&e);touches++;}
static void vk_texture_use_images(const VkImage *images,size_t count){assert(locks);for(size_t i=0;i<count;i++){assert(pin_count<64);pin_images[pin_count++]=(unsigned)images[i];pins++;}}
static int find_stage(void *a,Program*b,int st,const uint8_t **ctl,const uint8_t **code,uint32_t *len){(void)a;(void)b;(void)st;*ctl=control;*code=NULL;*len=0;return 1;}
static int ctl_nvn2(const uint8_t *p){return p==control;}
static const VkTexImage *vk_texture_current_registration_locked(int id,view_t*v,int*hv){assert(locks);(void)v;*hv=0;return id>0&&id<32&&slots[id]?&slots[id]->pub:NULL;}
static int vk_texture_effect_color_blit(int id,VkTexImage*out,view_t*v,int*hv,uint64_t*size,VkTextureColorBlitSource *color) {
    assert(locks);prepared++;if(!capture_ok||id<=0||id>=32||!slots[id])return 0;
    Tex *e=slots[id];*out=e->pub;*size=e->t.L.size;*hv=0;touch(e);
    vk_texture_use_images(&out->image,1);texture_color_blit_capture(e,v,*hv,color);return 1;
}
static int vk_texture_deferred(void){return deferred;}
static int vk_texture_image_live_locked(const VkTexImage *image){assert(locks);for(Tex*e=owners;e;e=e->next_at)if(e->pub.image==image->image&&e->pub.addr==image->addr&&!e->pub.gone)return 1;return 0;}
static const VkTexImage *vk_texture_writable_locked(const VkTexImage*i){return i;}
static int rt_knob(const char*n,const char*d){(void)d;return !strcmp(n,"CS_FALLBACK")?fallback_on:1;}
static int vk_aa_mode(void){return mode;}
static int vk_aa_role(uint64_t h){(void)h;return role;}
static uint64_t cs_guest_hash(Program*p){return p?p->serial:0;}
static void vk_exec_turn(void){assert(!locks);turns++;}
static int nvn_gpu_only(uint64_t a){return a>=UINT64_C(0xf000000000000000);}
static void effect_declare_host(EffectDeclaration *d,const void *native,uint64_t a,uint64_t n,
                                uint64_t lo,uint64_t length,RcRegistryWriterKind kind){
    (void)d;assert(!native&&a==lo&&n==length&&kind==RC_REGISTRY_IMAGE);
}
static struct {RcSourceRange ranges[8];RcRegistryWrite tickets[8];unsigned count;RcRegistryPosition position;} emitted;
static void resource_effects_add(void*c,RcSourceRange r,RcSourceRange full,RcRegistryWriterKind kind){(void)c;assert(rc_range_equal(r,full)&&kind==RC_REGISTRY_IMAGE);unsigned i=emitted.count++;assert(i<8);emitted.ranges[i]=r;assert(rc_registry_write_begin_at(registry,r,kind,emitted.position,&emitted.tickets[i])==RC_REGISTRY_OK);}
static RcRegistryResult resource_effects_add_pinned_image(void *commands,uint64_t image,
        const RcInterval *intervals,uint32_t count) {
    assert(image&&intervals&&count);
    for(uint32_t i=0;i<count;i++)
        resource_effects_add(commands,intervals[i].range,intervals[i].range,RC_REGISTRY_IMAGE);
    return RC_REGISTRY_OK;
}

static unsigned issued_roles, skipped_roles;
static void resource_effects_image_role_begin(void *commands,ResourceImageEffectRole *role) {
    assert(!role->owner);role->owner=commands?commands:role;
}
static void resource_effects_image_role_end(void *commands,ResourceImageEffectRole *role) {
    assert(role->owner==(commands?commands:role));
}
static void resource_effects_image_role_finish(ResourceImageEffectRole *role,int issued) {
    if (!role->owner) return;
    role->owner=NULL;
    if (issued) issued_roles++;
    else skipped_roles++;
}
static struct {VkExecTarget color,depth;int armed,gbuf,done,early,composite,up,promoted,ended,jit_used;float ux,uy;uint64_t early_outputs,full_size_outputs,evals;} D;
enum { POST_COLOR };
typedef struct {struct {VkImage image;} member[1];struct {unsigned width,height;} size;} DlaaPostImages;
static struct {DlaaPostImages *frame;} post;
static int post_current(DlaaPostImages *images){return images!=NULL;}
static void post_scene_use(VkExec *ex,DlaaPostImages *images){assert(ex&&images);}
static unsigned copy_count,write_count,owned_count,storage_count,prepare_count,events[64],event_count,barriers,evaluate_count,blits;
static int active=1,mutate_hook,evaluate_result;
static void rt_log(const char*s,...){(void)s;}
static void resource_image_prepare(uint64_t image,uint64_t addr,uint64_t size){assert(image&&addr&&size);prepare_count++;events[event_count++]=1;}
static void vk_texture_image_write_prepare(VkImage image,uint64_t addr,uint64_t size) {
    resource_image_prepare(image,addr,size);
}
static void vk_texture_image_write_prepare_locked(const VkTexImage *image,uint64_t size) {
    assert(locks);resource_image_prepare(image->image,image->addr,size);
}
static void resource_image_written(uint64_t image,uint64_t addr){assert(image&&addr);write_count++;events[event_count++]=3;}
static void resource_image_owned(uint64_t image,uint64_t addr,RcRegistryWrite hold){assert(image&&addr&&hold.id);owned_count++;events[event_count++]=4;}
static void vk_texture_storage_written(VkImage image,uint64_t addr){assert(image&&addr);storage_count++;events[event_count++]=5;}
static struct {VkImage image;unsigned barrier;VkImageCopy region;} recorded_edges[4];
static unsigned monitor_edges,recorded_edges_count;
static void vkCmdCopyImage(VkCommandBuffer cb,VkImage src,int sl,VkImage dst,int dl,unsigned n,const VkImageCopy*r) {
    assert(cb&&src==dst&&sl==dl&&n==1&&r->extent.width&&r->extent.height);
    if(monitor_edges) {
        assert(recorded_edges_count<4);
        recorded_edges[recorded_edges_count++]=(typeof(*recorded_edges)){src,barriers,*r};
    }
    copy_count++;events[event_count++]=2;
}
static void full_barrier(VkCommandBuffer cb){assert(cb);barriers++;}
static int dl_active(void){return active;}
static void vk_exec_hooks_wait(void){}
static void vk_exec_hook_turn(void){if(mutate_hook)D.color.image++;}
static void vk_exec_barrier(VkExec*e){assert(e);}
static VkImageBlit blit_scene(VkCommandBuffer cb,VkImage s,unsigned sw,unsigned sh,VkImage d,unsigned dw,unsigned dh) {
    assert(cb&&s&&d&&sw&&sh&&dw&&dh);blits++;events[event_count++]=6;
    return (VkImageBlit){.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.dstOffsets={{0,0,0},{dw,dh,1}}};
}
static int vk_dlaa_hudless_wanted;
static void hudless(VkExec *ex,VkImage s,VkImage d,const VkTexImage *texture,void *reconstruction,
                    const VkTextureColorBlitSource *output,unsigned w,unsigned h) {
    (void)reconstruction;assert(ex&&s==d&&texture&&output&&output->image==d&&output->hold&&w&&h);
    assert(event_count>=4 && events[event_count-1]==8);
    events[event_count++]=9;
}
static void resource_effects_emitted(void){events[event_count++]=7;}
static void vk_texture_storage_blit_written(const VkTextureColorBlitSource *output,const VkImageBlit *region) {
    assert(output->image&&output->address&&output->hold&&region->dstSubresource.aspectMask==VK_IMAGE_ASPECT_COLOR_BIT);
    assert(event_count>=3&&events[event_count-3]==6&&events[event_count-2]==7&&events[event_count-1]==3);
    events[event_count++]=8;
}
static struct { int forward_pending; } hud_sources;

struct VkTextureProjectedTransfer { VkImage image; uint32_t count; VkImageCopy copies[2]; unsigned issued; };
static unsigned exact_prepared,exact_reported,exact_released,pending_edges;
static VkImage stale_edge;
static int vk_texture_image_binding(VkImage image,uint64_t address,uint64_t bytes,RcRegistryImageBinding *out,
                                    RcRegistryImageWriteExtent **extents,uint32_t *count) {
    *out=(RcRegistryImageBinding){0};*extents=NULL;*count=0;
    for(Tex *e=owners;e;e=e->next_at) if(e->pub.image==image && e->pub.addr==address && e->t.L.size==bytes) {
        *count=e->image_private_mapping_count;
        if(!*count) return 0;
        *extents=calloc(*count,sizeof **extents);assert(*extents);
        for(uint32_t i=0;i<*count;i++) (*extents)[i].interval=e->image_private_mapping[i];
        return 0; // Exact placement exists independently of an old compatibility-range hold.
    }
    return 0;
}
static int vk_texture_projected_self_copy_capture(const tex_t *tex,const VkTexImage *im,const VkImageCopy *boxes,
                                                 uint32_t count,VkTextureProjectedTransfer **out) {
    assert(locks && tex && im && count && count<=2 && !*out);
    if(im->image==stale_edge) return -1;
    *out=calloc(1,sizeof **out);assert(*out);
    (*out)->image=im->image;(*out)->count=count;
    memcpy((*out)->copies,boxes,count*sizeof *boxes);
    exact_prepared++;
    return 1;
}
static const VkImageCopy *vk_texture_projected_self_copy_regions(const VkTextureProjectedTransfer *r,uint32_t *n) {
    *n=r->count;return r->copies;
}
static VkImage vk_texture_projected_transfer_image(const VkTextureProjectedTransfer *r){return r->image;}
static int vk_texture_projected_transfer_begin(VkTextureProjectedTransfer *r) {
    if(r->image==stale_edge)return 0;
    lock();r->issued=copy_count;return 1;
}
static void vk_texture_projected_transfer_written(VkTextureProjectedTransfer *r) {
    assert(locks && copy_count-r->issued>=r->count);exact_reported++;unlock();
}
static RcRegistryResult vk_texture_projected_transfer_declare(const VkTextureProjectedTransfer *r,void *commands,RcRegistryPosition output) {
    assert(r && rc_registry_position_valid(output));
    for(Tex *e=owners;e;e=e->next_at) if(e->pub.image==r->image)
        return resource_effects_add_pinned_image(commands,r->image,e->image_private_mapping,e->image_private_mapping_count);
    return RC_REGISTRY_STALE;
}
static void vk_texture_projected_transfer_cancel(VkTextureProjectedTransfer **r) {
    if(*r){free(*r);*r=NULL;exact_released++;}
}
static struct {void(*complete)(void *);void(*cancel)(void *,int);void *data;} edge_callbacks[2];
static uint64_t vk_exec_recording_fence(void){return 1;}
static void vk_completion_add_owned(uint64_t fence,void(*complete)(void *),void(*cancel)(void *,int),void *data) {
    assert(fence==1 && !locks && pending_edges<2);
    edge_callbacks[pending_edges++]=(typeof(*edge_callbacks)){complete,cancel,data};
}
'''

body = function(texture, 'vk_texture_image_effect')
body += function(texture, 'texture_color_blit_capture') + function(texture, 'texture_color_blit_complete')
body += function(dlaa, 'dlaa_compute_blit_written')
evaluate_body = function(dlaa, 'evaluate')
assert evaluate_body.index("vkCmdBlitImage(cb, shown") < evaluate_body.index("dlaa_compute_blit_written(") < evaluate_body.index("hudless(")
body += function(dlaa, 'dlaa_guest_target')
body += "\n".join(function(dlaa, name) for name in (
    'scene_effects_current', 'vk_dlaa_scene_effects', 'vk_dlaa_scene_effects_cancel_unselected',
    'dlaa_edge_axes', 'dlaa_edge_targets', 'dlaa_edge_copy', 'dlaa_edge_regions', 'dlaa_edge_mapping',
    'vk_dlaa_scene_prepare_edges', 'vk_dlaa_scene_release_edges', 'dlaa_edge_completed', 'dlaa_edge_canceled', 'edge_fill'))
body += function(draw, 'fallback_destination_locked') + function(draw, 'effect_emit') + function(draw, 'effect_tensor_outputs')
body += r'''
static int dlaa_targets_armed(void){return D.armed;}
static int evaluate(VkExec*ex,const VkExecTarget*out,VkDlaaSceneEffects*effects,int early,
                    const VkTextureColorBlitSource *output) {
    assert(out->image&&!early);evaluate_count++;ex->recorded+=edge_fill(ex->cb,&D.color,&D.depth,-0.25f,-0.25f,effects);
    if(evaluate_result) {
        VkImageBlit region=blit_scene(ex->cb,99,64,64,out->image,out->width,out->height);
        dlaa_compute_blit_written(output,&region);
    }
    return evaluate_result;
}
'''
body += function(dlaa, 'vk_dlaa_scene_upscale')
# The actual destination-selection branch must keep consuming the captured physical image after registration changes.
start = draw.index('    EffectFallbackCapture *captured=effect_current?effect_current->fallback:NULL;')
end = draw.index('    out = dst;', start)
body += 'static EffectDeclaration *effect_current;\nstatic VkTexImage used;\nstatic void use_destination(VkExec *ex){const VkTexImage *dst=NULL;\n'
body += draw[start:end]
body += '(void)legacy_output;if(dst)used=*dst;vk_texture_unlock();}\n'

tests = r'''
static RcRegistryHandle source(uint64_t a){RcRegistryHandle h;assert(rc_registry_source(registry,a,(RcSourceRange){1,a,64},&h)==RC_REGISTRY_OK);return h;}
static int visible(RcRegistryHandle h){RcRegistryObservation o;return rc_registry_observe(registry,h,1,&o)==RC_REGISTRY_OK;}
static void begin(unsigned scene){emitted=(typeof(emitted)){.position={scene,3}};assert(rc_registry_declare_through(registry,(RcRegistryPosition){scene,2})==RC_REGISTRY_OK);}
static void cancel(void){for(unsigned i=0;i<emitted.count;i++)assert(rc_registry_write_cancel(registry,emitted.tickets[i])==RC_REGISTRY_OK);}
int main(void){
 registry=rc_registry_create((RcRegistryConfig){128,128,8});assert(registry);
 Tex target={.pub={11,0x1000,80,80,1,2,1,3},.t.L.size=2048};
 Tex color={.pub={21,0x3000,64,64,1,2,1,3},.t.L.size=1024};
 Tex depth={.pub={22,0x4000,64,64,2,2,1,4},.t.L.size=512};
 Tex stale={.pub={31,0x5000,160,160,1,2,1,3},.t.L.size=4096};
 target.next_at=&color;color.next_at=&depth;depth.next_at=&stale;owners=&target;
 slots[1]=&target;slots[2]=&stale;Program program={7};VkExec ex={.prog=&program,.cb=77,.fence=1};ex.img[0][0]=1;ex.img[0][1]=2;
 tex_t ct={(void*)(uintptr_t)0x3000},zt={(void*)(uintptr_t)0x4000};
 D.color=(VkExecTarget){.image=21,.width=64,.height=64,.tex=&ct};D.depth=(VkExecTarget){.image=22,.width=64,.height=64,.tex=&zt};D.armed=1;
 RcRegistryHandle unrelated=source(0x9000),destination=source(0x1000),border=source(0x3000);
 assert(rc_registry_hold_begin(registry,(RcSourceRange){1,0x3000,1024},RC_REGISTRY_IMAGE,&color.image_authority)==RC_REGISTRY_OK);
 EffectDeclaration d={.program=&program,.program_serial=7};control[0xb6]=0;begin(1);
 assert(effect_tensor_outputs(&d,&ex)&&turns==1&&prepared==1&&pins==3&&touches==3&&!locks);
 assert(d.fallback->output.image==11&&d.fallback->size==2048&&emitted.count==3);
 assert(emitted.ranges[0].offset==0x1000&&emitted.ranges[0].size==2048&&emitted.ranges[1].offset==0x3000&&emitted.ranges[2].offset==0x4000);
 assert(pin_images[0]==11&&pin_images[1]==21&&pin_images[2]==22);
 assert(visible(unrelated)&&!visible(destination)); // No full-space BUFFER/IMAGE dependency reaches unrelated bytes.
 // No command used the captured border: canceling the declaration removes its pending ownership only.
 cancel();assert(visible(destination)&&!visible(border));assert(rc_registry_hold_release(registry,color.image_authority,1)==RC_REGISTRY_OK);color.image_authority=(RcRegistryWrite){0};assert(visible(border));
 // Exact captured destination remains selected even if the descriptor is re-registered and the logical image retires.
 slots[1]=&stale;target.pub.gone=1;effect_current=&d;use_destination(&ex);assert(used.image==11&&used.gone&&prepare_count==1);target.pub.gone=0;slots[1]=&target;
 // Immutable owner metadata is separate from range ownership. Losing the hold is safe; missing exact owner is not.
 VkDlaaSceneEffects scene;assert(vk_dlaa_scene_effects(&scene)&&!scene.owners[0].hold.id);
 D.color.image=999;assert(!vk_dlaa_scene_effects(&scene)&&!scene.complete);D.color.image=21;
 assert(vk_dlaa_scene_effects(&scene));event_count=0;prepare_count=0;scene.owners[0]=(RcRegistryImageBinding){21,{.id=1}};
 // evaluate may fail after border copies. Every actual copy is reported before the ordinary dst-only fallback.
 assert(!vk_dlaa_scene_upscale(&ex,64,64,&d.fallback->output,&scene,&d.fallback->blit));
 assert(evaluate_count==1&&copy_count==4&&prepare_count==4&&write_count==4&&owned_count==2&&storage_count==4&&barriers==2&&!blits);
 for(unsigned i=0;i<event_count;){assert(events[i++]==1&&events[i++]==2&&events[i++]==3);if(events[i]==4)i++;assert(events[i++]==5);}
 // A prior hook selecting another physical image refuses before any edge write; caller may blit captured dst only.
 unsigned before=copy_count;mutate_hook=1;assert(!vk_dlaa_scene_upscale(&ex,64,64,&d.fallback->output,&scene,&d.fallback->blit));assert(copy_count==before&&evaluate_count==1);mutate_hook=0;D.color.image=21;
 tex_t moved={(void*)(uintptr_t)0x3010};D.color.tex=&moved;assert(!vk_dlaa_scene_upscale(&ex,64,64,&d.fallback->output,&scene,&d.fallback->blit)&&copy_count==before);D.color.tex=&ct;
 scene.border[0].owner=&scene;scene.border[1].owner=&scene;
 edge_fill(ex.cb,&D.color,&D.depth,0,0,&scene);
 assert(copy_count==before&&skipped_roles==2&&!issued_roles);
 scene.border[0].owner=&scene;scene.border[1].owner=&scene;
 D.depth.width--;event_count=0;
 edge_fill(ex.cb,&D.color,&D.depth,-0.25f,0,&scene);
 D.depth.width++;
 assert(copy_count==before+1&&skipped_roles==3&&issued_roles==1);
 vk_dlaa_scene_effects_cancel_unselected(&scene,0);
 assert(skipped_roles==3&&issued_roles==1);

 // The real early-composite adapter records the selected guest write immediately, before its HUD handoff.
 scene.border[0].owner=&scene;scene.border[1].owner=&scene;
 D.early=1;D.composite=93;D.up=0;vk_dlaa_hudless_wanted=1;event_count=0;
 assert(vk_dlaa_scene_upscale(&ex,64,64,&d.fallback->output,&scene,&d.fallback->blit));
 assert(event_count==5&&events[0]==6&&events[1]==7&&events[2]==3&&events[3]==8&&events[4]==9);
 assert(skipped_roles==5&&issued_roles==1);
 D.early=D.composite=D.up=vk_dlaa_hudless_wanted=0;

 // Both actual edge roles can consume disjoint exact prepared transfers and hand their holds to the fence.
 VkTextureColorBlitSource primary={.image=11,.mapping_count=1,.mapping={{0x1000,7,{2,0,2048}}}};
 RcInterval color_map={0x3000,8,{2,4096,1024}},depth_map={0x4000,9,{3,0,512}};
 color.image_private_mapping=&color_map;color.image_private_mapping_count=1;
 depth.image_private_mapping=&depth_map;depth.image_private_mapping_count=1;
 D.color.img=&color.pub;D.depth.img=&depth.pub;
 D.jit_used=1;D.ux=D.uy=-0.25f;
 assert(vk_dlaa_scene_effects(&scene));
 vk_dlaa_scene_prepare_edges(&scene,&primary);
 assert(scene.edge[0] && scene.edge[1] && exact_prepared==2);
 scene.border[0].owner=&scene;scene.border[1].owner=&scene;
 event_count=0; before=copy_count; unsigned prior_barriers=barriers;
 monitor_edges=1;recorded_edges_count=0;
 assert(edge_fill(ex.cb,&D.color,&D.depth,D.ux,D.uy,&scene)==4);
 monitor_edges=0;
 assert(copy_count==before+4 && barriers==prior_barriers+2 && exact_reported==2 && pending_edges==2);
 assert(recorded_edges_count==4);
 for(unsigned i=0;i<4;i++) {
     assert(recorded_edges[i].image==(i%2?D.depth.image:D.color.image));
     assert(recorded_edges[i].barrier==prior_barriers+i/2);
     assert(recorded_edges[i].region.extent.width==(i<2?1u:64u));
     assert(recorded_edges[i].region.extent.height==(i<2?64u:1u));
 }
 assert(!scene.edge[0] && !scene.edge[1] && !exact_released);
 vk_dlaa_scene_release_edges(&scene);
 assert(!exact_released);
 edge_callbacks[0].complete(edge_callbacks[0].data);edge_callbacks[1].cancel(edge_callbacks[1].data,0);
 assert(exact_released==2);pending_edges=0;
 // The physical primary alias and normalized sibling alias cases leave old role handling alone.
 primary.image=D.color.image;
 vk_dlaa_scene_prepare_edges(&scene,&primary);assert(!scene.edge[0] && !scene.edge[1]);
 primary.image=11;depth_map.range=(RcSourceRange){2,4096,512};
 vk_dlaa_scene_prepare_edges(&scene,&primary);assert(!scene.edge[0] && !scene.edge[1]);
 depth_map.range=(RcSourceRange){3,0,512};
 // A stale capture cancels only that role. Its exact sibling and unrelated primary remain available.
 vk_dlaa_scene_prepare_edges(&scene,&primary);assert(scene.edge[0] && scene.edge[1]);
 scene.border[0].owner=&scene;scene.border[1].owner=&scene;stale_edge=scene.images[0];
 event_count=0;before=copy_count;unsigned before_skips=skipped_roles;
 assert(edge_fill(ex.cb,&D.color,&D.depth,D.ux,D.uy,&scene)==2);
 assert(copy_count==before+2 && skipped_roles==before_skips+1 && exact_reported==3 && pending_edges==1);
 assert(scene.edge[0] && !scene.edge[1]);
 vk_dlaa_scene_release_edges(&scene);
 edge_callbacks[0].complete(edge_callbacks[0].data);
 assert(exact_released==4);pending_edges=0;stale_edge=0;
 D.jit_used=0;D.color.img=NULL;D.depth.img=NULL;
 // Only an exact independently placed logical plane, actual complete box and the original mapping qualify.
 Tex plane={.pub={.image=11,.addr=UINT64_C(0xf000000000001000),.width=80,.height=80,.aspect=1,.type=2,
    .samples=1,.format=3,.levels=1,.layers=1},.t={.L.size=25600,.serial=42,.d={80,80,2,63}},
    .image_private_mapping_count=1,.image_mapped_hold_count=1,.independent=1};
 RcInterval mapping={.address=plane.pub.addr,.mapping=88,.range={19,0,25600}};
 plane.image_private_mapping=&mapping;
 view_t identity={.set_format=1,.format=63};
 VkTextureColorBlitSource exact;
 texture_color_blit_capture(&plane,&identity,1,&exact);
 VkImageBlit exact_box={.dstSubresource={1,0,0,1},.dstOffsets={{0,0,0},{80,80,1}}};
 assert(texture_color_blit_complete(&plane,&exact,&exact_box));
 --exact_box.dstOffsets[1].x;assert(!texture_color_blit_complete(&plane,&exact,&exact_box));++exact_box.dstOffsets[1].x;
 exact_box.dstSubresource.mipLevel=1;
 assert(!texture_color_blit_complete(&plane,&exact,&exact_box));exact_box.dstSubresource.mipLevel=0;
 ++mapping.mapping;assert(!texture_color_blit_complete(&plane,&exact,&exact_box));--mapping.mapping;
 ++read_position.ordinal;assert(!texture_color_blit_complete(&plane,&exact,&exact_box));--read_position.ordinal;
 ++plane.t.serial;assert(!texture_color_blit_complete(&plane,&exact,&exact_box));--plane.t.serial;
 vk_mem_image_release(exact.hold);
 identity.format=62;
 texture_color_blit_capture(&plane,&identity,1,&exact);
 assert(!exact.mapping_count&&!texture_color_blit_complete(&plane,&exact,&exact_box));vk_mem_image_release(exact.hold);
 vk_mem_image_release(d.fallback->blit.hold);free(d.fallback);d.fallback=NULL;
 // The real declaration chooses exact physical claims for both enabled retained edges and for the primary.
 Tex *saved_owners=owners;
 plane.next_at=&color;owners=&plane;slots[1]=&plane;
 D.jit_used=1;D.color.img=&color.pub;D.depth.img=&depth.pub;
 begin(2);d.position=emitted.position;
 assert(effect_tensor_outputs(&d,&ex)&&d.fallback->scene.edge[0]&&d.fallback->scene.edge[1]);
 assert(emitted.count==3&&emitted.ranges[0].backing==19&&emitted.ranges[1].backing==2&&emitted.ranges[2].backing==3);
 cancel();vk_dlaa_scene_release_edges(&d.fallback->scene);
 vk_mem_image_release(d.fallback->blit.hold);free(d.fallback);d.fallback=NULL;
 D.jit_used=0;D.color.img=NULL;D.depth.img=NULL;owners=saved_owners;slots[1]=&target;
 // Inactive reconstruction still inventories potential border outputs; changing activation cannot add undeclared writes.
 active=0;begin(3);assert(effect_tensor_outputs(&d,&ex)&&emitted.count==3);cancel();vk_mem_image_release(d.fallback->blit.hold);free(d.fallback);d.fallback=NULL;active=1;
 // Descriptor count bounds exclude stale larger images; zero means slot zero, then a real two-entry count selects larger.
 control[0xb6]=2;begin(4);assert(effect_tensor_outputs(&d,&ex)&&d.fallback->output.image==31);cancel();vk_mem_image_release(d.fallback->blit.hold);free(d.fallback);d.fallback=NULL;
 stale.pub.samples=4;begin(5);assert(effect_tensor_outputs(&d,&ex)&&d.fallback->output.image==11);cancel();vk_mem_image_release(d.fallback->blit.hold);free(d.fallback);d.fallback=NULL;stale.pub.samples=1;
 // Missing inventory, deferred capture or another substitution never claim a complete empty write set.
 D.depth.image=999;begin(6);assert(!effect_tensor_outputs(&d,&ex)&&!emitted.count&&!d.fallback);D.depth.image=22;
 deferred=1;assert(!effect_tensor_outputs(&d,&ex)&&!emitted.count);deferred=0;
 capture_ok=0;assert(!effect_tensor_outputs(&d,&ex)&&!emitted.count);capture_ok=1;
 mode=1;role=1;int oldturns=turns;assert(!effect_tensor_outputs(&d,&ex)&&turns==oldturns);role=0;mode=0;
 // GPU-only output still has a captured physical lifetime, while no CPU authority exclusion is invented for it.
 D.color=(VkExecTarget){0};D.depth=(VkExecTarget){0};control[0xb6]=0;target.pub.addr=UINT64_C(0xf000000000001000);begin(7);
 assert(effect_tensor_outputs(&d,&ex)&&!emitted.count&&d.fallback->output.image==11&&visible(unrelated));vk_mem_image_release(d.fallback->blit.hold);free(d.fallback);
 assert(!color_holds);
 rc_registry_destroy(registry);
 puts("tensor outputs: exact selector, retained destination, complete inventory, partial edge writes, races and cancellation passed");
}
'''

env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get('PATH', ''))
with tempfile.TemporaryDirectory(prefix='tensor-output-', dir=ROOT / 'build/test') as tmp:
    path = Path(tmp) / 'check.c'
    path.write_text(prefix + body + tests)
    exe = path.with_suffix('.exe')
    subprocess.run([str(CC), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror',
                    '-Wno-missing-field-initializers', '-fsanitize=undefined', '-fsanitize-trap=all', '-DRC_REGISTRY_SINGLE_THREADED',
                    '-I' + str(ROOT / 'runtime'), str(path), str(ROOT / 'runtime/resource_core_registry.c'),
                    '-o', str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

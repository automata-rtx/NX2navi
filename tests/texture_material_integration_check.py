"""Exercise production texture ownership, content transactions and storage preflight without a game.

GPU allocation/recording, source observation and hashing are controlled boundaries in the content fixture.
It validates publication/ownership ordering, not Vulkan execution or cryptographic hash quality.
"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'build/test/texture-material-integration'
CC = Path(os.environ.get('CC', r'C:\msys64\ucrt64\bin\clang.exe'))
if not CC.is_file():
    CC = Path(r'C:\Users\Andrew\AppData\Local\Programs\2NX2Navi\toolchain\msys64\ucrt64\bin\clang.exe')
sys.path.insert(0, str(ROOT / 'tools'))
from check_content import GENERATED_MARKER

OWNER_FIXTURE = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef uintptr_t VkImage;
typedef void *VmaAllocation;
typedef struct { uint32_t flags,type,format,w,h,d,levels,layers,samples,usage; } ImgKey;
#include "vk_texture_material.h"

static const ImgKey image_key = {.format=37,.w=256,.h=128,.d=1,.levels=4,.layers=8,.samples=1,.usage=7};
static void assert_retained_queue(const TextureMaterialLedger *ledger) {
    uint64_t count=0,bytes=0;
    const TextureMaterial *previous=NULL;
    int cursor_found=ledger->retained_scan_next==NULL;
    for(const TextureMaterial *m=ledger->retained_head;m;m=m->next) {
        assert(++count<=ledger->owners); // Also detects a linked-list cycle.
        assert(m->ledger==ledger && !m->retired && !m->live_refs && m->retained_queued && m->resident.image);
        assert(m->prev==previous);
        bytes+=m->resident.bytes;
        cursor_found|=m==ledger->retained_scan_next;
        previous=m;
    }
    assert(count==ledger->retained_owners && bytes==ledger->retained_bytes && cursor_found);
}

static void retained_queue_transitions(void) {
    TextureMaterialLedger ledger={0};
    TextureMaterial a,b,c;
    texture_material_init(&a,&ledger,1,(void *)(uintptr_t)11,&image_key,1024,0);
    texture_material_init(&b,&ledger,2,(void *)(uintptr_t)12,&image_key,2048,0);
    texture_material_init(&c,&ledger,3,(void *)(uintptr_t)13,&image_key,4096,0);
    assert_retained_queue(&ledger);
    assert(texture_material_scan_next(&ledger)==&a);
    assert(texture_material_scan_next(&ledger)==&c && ledger.retained_scan_next==&b);
    // One live association removes a physical owner even while another association remains released.
    texture_material_acquire(&b,1);
    assert(b.refs==2 && b.live_refs==1 && !b.retained_queued && ledger.retained_scan_next==&a);
    assert_retained_queue(&ledger);
    assert(texture_material_scan_next(&ledger)==&a);
    assert(texture_material_scan_next(&ledger)==&c && ledger.retained_scan_next==&a);
    // Retiring exactly the next victim repairs the cursor before its address can be freed or reused.
    assert(texture_material_release(&a,0));
    assert(texture_material_retire(&a).bytes==1024 && !ledger.retained_scan_next);
    assert(texture_material_scan_next(&ledger)==&c);
    assert_retained_queue(&ledger);
    texture_material_acquire(&c,0);
    assert(c.refs==2 && ledger.retained_owners==1);
    texture_material_set_live(&c,1);
    assert(!ledger.retained_head && !ledger.retained_scan_next && !ledger.retained_owners);
    assert(!texture_material_scan_next(&ledger));
    assert(!texture_material_release(&c,0));
    texture_material_set_live(&c,0);
    assert(ledger.retained_head==&c && ledger.retained_scan_next==&c);
    assert_retained_queue(&ledger);
    assert(texture_material_release(&c,0));
    texture_material_retire(&c);
    assert(!texture_material_release(&b,1));
    assert(b.refs==1 && !b.live_refs && b.retained_queued);
    assert_retained_queue(&ledger);
    assert(texture_material_release(&b,0));
    texture_material_retire(&b);
    assert_retained_queue(&ledger);
    assert(!ledger.owners && !ledger.retained_head && !texture_material_scan_next(&ledger));
}

static void basic_ownership(void) {
    TextureMaterialLedger ledger = {0};
    TextureMaterial m;
    texture_material_init(&m,&ledger,101,(void *)(uintptr_t)201,&image_key,4096,1);
    assert(m.id && m.refs==1 && m.live_refs==1 && !m.ready && !m.cached);
    assert(ledger.owners==1 && ledger.resident_bytes==4096 && ledger.live_bytes==4096 && !ledger.retained_bytes);
    assert(texture_material_exclusive(&m));
    m.ready=1;
    assert(!texture_material_cache(&m,1));
    assert(!texture_material_exclusive(&m));
    texture_material_acquire(&m,1);
    texture_material_acquire(&m,0);
    assert(m.refs==3 && m.live_refs==2 && ledger.resident_bytes==4096 && ledger.live_bytes==4096);
    texture_material_set_live(&m,0);
    assert(m.live_refs==1 && !ledger.retained_bytes);
    assert(!texture_material_release(&m,1));
    assert(m.refs==2 && !m.live_refs && !ledger.live_bytes && ledger.retained_bytes==4096);
    assert(!texture_material_release(&m,0));
    assert(!texture_material_release(&m,0));
    assert(!m.refs && m.cached && !texture_material_unreferenced(&m));
    assert(m.resident.image==101 && m.resident.alloc==(void *)(uintptr_t)201 && m.ready);
    assert(ledger.owners==1 && ledger.resident_bytes==4096 && ledger.retained_bytes==4096);
    // A different logical source can reacquire the retained owner without another physical allocation.
    texture_material_acquire(&m,1);
    assert(m.refs==1 && m.live_refs==1 && ledger.live_bytes==4096 && !ledger.retained_bytes);
    assert(!texture_material_cache(&m,0) && texture_material_exclusive(&m));
    texture_material_note_use(&m,19,8);
    texture_material_note_use(&m,7,9);
    assert(m.resident.last_use_fence==19 && m.resident.last_use_epoch==9);
    assert(texture_material_release(&m,1));
    // Releasing logical references does not claim physical reclamation.
    assert(ledger.resident_bytes==4096 && ledger.retained_bytes==4096 && !m.retired);
    assert(texture_material_retire(&m).bytes==4096);
    assert(m.retired && !m.ready && !ledger.owners && !ledger.resident_bytes && !ledger.retained_bytes);
}

static void separate_versions(void) {
    TextureMaterialLedger ledger={0};
    TextureMaterial old,changed;
    texture_material_init(&old,&ledger,1,(void *)(uintptr_t)11,&image_key,1024,1);
    old.ready=1;
    texture_material_cache(&old,1);
    texture_material_acquire(&old,1);
    texture_material_note_use(&old,70,30);
    texture_material_init(&changed,&ledger,2,(void *)(uintptr_t)12,&image_key,1024,1);
    assert(changed.id!=old.id && ledger.resident_bytes==2048);
    assert(!texture_material_release(&old,1));
    // The other source and previously recorded reads still name the original immutable representation.
    assert(old.resident.image==1 && old.refs==1 && old.live_refs==1 && old.resident.last_use_fence==70 && old.ready);
    assert(changed.resident.image==2 && texture_material_exclusive(&changed));
    assert(!texture_material_release(&old,1));
    assert(ledger.live_bytes==1024 && ledger.retained_bytes==1024);
    assert(texture_material_cache(&old,0));
    assert(texture_material_retire(&old).bytes==1024);
    assert(ledger.owners==1 && ledger.resident_bytes==1024 && changed.resident.image==2);
    assert(texture_material_release(&changed,1));
    assert(texture_material_retire(&changed).bytes==1024 && !ledger.resident_bytes);
}

static void independent_residency(void) {
    TextureMaterialLedger ledger={0};
    TextureMaterial m;
    texture_material_init(&m,&ledger,101,(void *)(uintptr_t)201,&image_key,4096,1);
    m.ready=1;
    texture_material_cache(&m,1);
    texture_material_acquire(&m,0);
    texture_material_note_use(&m,71,30);
    uint64_t representation_id=m.id;
    TextureMaterialResidency captured=texture_material_capture(&m);
    // A finalized association needs exact old contents. Current guest bytes alone cannot recover those.
    assert(!texture_material_take_residency(&m).image);
    // This isolated fixture supplies a retained immutable payload before permitting loss of the last image.
    unsigned char recovery_payload[4096]={0}; assert(sizeof recovery_payload==m.resident.bytes);
    assert(rc_lifetime_recovery_set(&m.lifetime,m.lifetime.id,RC_LIFETIME_RECOVERY_IMMUTABLE)==RC_LIFETIME_OK);
    TextureMaterialResidency deferred_old=texture_material_take_residency(&m);
    assert(deferred_old.image==captured.image && deferred_old.alloc==captured.alloc && deferred_old.id==captured.id);
    assert(deferred_old.last_use_fence==71 && deferred_old.last_use_epoch==30);
    assert(!m.resident.image && !m.resident.id && !m.resident.bytes && !m.retained_queued);
    assert(m.id==representation_id && m.ready && m.cached && m.refs==2 && m.live_refs==1);
    assert(!memcmp(&m.key,&image_key,sizeof image_key));
    assert(ledger.owners==1 && !ledger.resident_bytes && !ledger.live_bytes && !ledger.retained_bytes);
    // A second physical take yields no ownership transfer and therefore no second deferred destruction.
    TextureMaterialResidency empty=texture_material_take_residency(&m);
    assert(!empty.image && !empty.alloc && !empty.id && !empty.bytes);
    assert(!texture_material_capture(&m).image);
    assert(!texture_material_release(&m,1));
    texture_material_acquire(&m,1);
    texture_material_set_live(&m,0);
    assert(!texture_material_cache(&m,0));
    assert(!texture_material_cache(&m,1));
    assert(!texture_material_release(&m,0));
    assert(!texture_material_release(&m,0));
    assert(!m.refs && !m.live_refs && m.cached && m.ready && !ledger.retained_owners);
    assert_retained_queue(&ledger);
    // Metadata survives without logical users or VRAM. Reinstall the same representation under a new instance ID.
    TextureMaterialResidency installed=texture_material_install_residency(&m,102,(void *)(uintptr_t)202,8192);
    assert(installed.id && installed.id!=captured.id && installed.image==102 && installed.bytes==8192);
    assert(m.id==representation_id && m.ready && m.cached && ledger.owners==1);
    assert(ledger.resident_bytes==8192 && ledger.retained_bytes==8192 && !ledger.live_bytes && m.retained_queued);
    texture_material_note_use(&m,99,44);
    assert(captured.image==101 && captured.alloc==(void *)(uintptr_t)201 && captured.bytes==4096);
    assert(captured.id==deferred_old.id && captured.last_use_fence==71 && captured.last_use_epoch==30);
    assert(installed.last_use_fence==0 && installed.last_use_epoch==0);
    assert_retained_queue(&ledger);
    TextureMaterialResidency deferred_new=texture_material_take_residency(&m);
    assert(deferred_new.image==102 && deferred_new.id==installed.id && deferred_new.last_use_fence==99);
    assert(!ledger.resident_bytes && !ledger.retained_bytes && !ledger.retained_owners);
    assert(texture_material_cache(&m,0));
    TextureMaterialResidency retired=texture_material_retire(&m);
    assert(!retired.image && !retired.id && !retired.bytes); // Both allocations were already transferred once.
    assert(m.retired && !m.ready && !ledger.owners);
    assert_retained_queue(&ledger);
}

static uint32_t rng=0x38a74de1u;
static uint32_t next_random(void) { rng ^= rng<<13; rng ^= rng>>17; rng ^= rng<<5; return rng; }
static void churn(void) {
    enum { N=12, ASSOCIATIONS=96 };
    TextureMaterialLedger ledger={0};
    TextureMaterial m[N]={0};
    int owner[ASSOCIATIONS],live[ASSOCIATIONS]={0};
    for(unsigned i=0;i<ASSOCIATIONS;i++)owner[i]=-1;
    for(unsigned step=0;step<20000;step++) {
        unsigned a=next_random()%ASSOCIATIONS, op=next_random()%4;
        if(owner[a]<0) {
            unsigned k=next_random()%N;
            if(!m[k].ledger || m[k].retired) {
                texture_material_init(&m[k],&ledger,step+1,(void *)(uintptr_t)(step+100),&image_key,(k+1)*4096,1);
                // The churn fixture supplies an immutable recovery provider for deliberately evicted live data.
                assert(rc_lifetime_recovery_set(&m[k].lifetime,m[k].lifetime.id,RC_LIFETIME_RECOVERY_IMMUTABLE)==RC_LIFETIME_OK);
                m[k].ready=1; owner[a]=(int)k; live[a]=1;
            } else {
                live[a]=(int)(next_random()&1); texture_material_acquire(&m[k],live[a]);owner[a]=(int)k;
            }
        } else {
            TextureMaterial *p=&m[owner[a]];
            if(op==0) { live[a]=!live[a];texture_material_set_live(p,live[a]); }
            else if(op==1) { texture_material_cache(p,!p->cached); }
            else { texture_material_release(p,live[a]);owner[a]=-1;live[a]=0; }
        }
        // Cache eviction and physical retirement are independent of association release.
        unsigned k=next_random()%N;
        if(m[k].ledger && !m[k].retired && !(next_random()%8))texture_material_cache(&m[k],0);
        if(m[k].ledger && !m[k].retired && !(next_random()%8)) {
            if(m[k].resident.image) texture_material_take_residency(&m[k]);
            else texture_material_install_residency(&m[k],step+1,(void *)(uintptr_t)(step+100),(k+1)*4096);
        }
        for(unsigned i=0;i<N;i++)if(m[i].ledger && !m[i].retired && texture_material_unreferenced(&m[i]))texture_material_retire(&m[i]);
        while(ledger.pending_head) {
            TextureMaterialInstance *instance=ledger.pending_head;
            assert(texture_material_instance_defer_ready(instance));
            texture_material_instance_defer_begin(instance);
            texture_material_destroyed(instance);
            (void)texture_material_instance_reap(instance);
        }
        uint64_t bytes=0,active=0,retained=0,owners=0;
        for(unsigned i=0;i<N;i++) {
            unsigned refs=0,live_refs=0;
            for(unsigned j=0;j<ASSOCIATIONS;j++)if(owner[j]==(int)i){refs++;live_refs+=(unsigned)live[j];}
            if(!m[i].ledger || m[i].retired){assert(!refs);continue;}
            assert(m[i].refs==refs && m[i].live_refs==live_refs);
            assert(m[i].retained_queued==(!live_refs && m[i].resident.image!=0));
            bytes+=m[i].resident.bytes;owners++;
            if(live_refs)active+=m[i].resident.bytes;else retained+=m[i].resident.bytes;
        }
        assert(ledger.owners==owners && ledger.resident_bytes==bytes && ledger.live_bytes==active && ledger.retained_bytes==retained);
        assert_retained_queue(&ledger);
        if(ledger.retained_owners) {
            TextureMaterial *candidate=texture_material_scan_next(&ledger);
            assert(candidate && candidate->retained_queued && !candidate->live_refs);
        } else assert(!texture_material_scan_next(&ledger));
    }
    for(unsigned i=0;i<ASSOCIATIONS;i++)if(owner[i]>=0)texture_material_release(&m[owner[i]],live[i]);
    for(unsigned i=0;i<N;i++)if(m[i].ledger && !m[i].retired){texture_material_cache(&m[i],0);texture_material_retire(&m[i]);}
    assert(!ledger.owners && !ledger.resident_bytes && !ledger.live_bytes && !ledger.retained_bytes);
    assert_retained_queue(&ledger);
}

int main(void) {
    basic_ownership();separate_versions();retained_queue_transitions();independent_residency();churn();
    puts("texture material ownership: retained identity, separate versions and unique budget accounting passed");
    return 0;
}
'''


STORAGE_FIXTURE = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
enum { GFX_RES_IMG=1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE=2, MAX_DESC=256,
VK_NULL_HANDLE=0,VK_IMAGE_LAYOUT_GENERAL=1,VK_IMAGE_VIEW_TYPE_2D=2,VK_IMAGE_VIEW_TYPE_2D_ARRAY=3 };
enum { VK_TEXTURE_REGISTRATION_CURRENT, VK_TEXTURE_REGISTRATION_EXACT,
       VK_TEXTURE_REGISTRATION_GONE=-2 };
typedef struct { int slot,exact,flags,type,format; } GfxShaderRes;
typedef struct { unsigned kind,type,count,stage; const GfxShaderRes *res; } VkPipeBinding;
typedef struct { unsigned n_bindings; VkPipeBinding bindings[4]; } VkPipeInfo;
typedef unsigned VkImage;
typedef unsigned VkImageView,VkImageViewType;
typedef struct { unsigned sampler,imageView,imageLayout; } VkDescriptorImageInfo;
typedef struct {uint64_t registration;int native_slot,refused;} VkTextureReadKey;
typedef struct { const GfxShaderRes *resource;uint64_t handle;VkDescriptorImageInfo descriptor;VkImage image; } BindingImageSelection;
typedef struct {const void *pin;VkImage image;} VkPinnedTextureWrite;
typedef struct { unsigned n_storage_writes,n_storage_images;BindingImageSelection storage_images[MAX_DESC];
struct { VkImage image; uint64_t addr; const VkPinnedTextureWrite *pinned; } storage_writes[MAX_DESC]; } DrawCtx;
typedef struct { uint64_t img[6][32]; DrawCtx *dc; } VkExec;
static void commit_recorded_writes(VkExec *ex) { (void)ex; }
static void resource_effects_emitted(void) {}
static void vk_exec_targets_written(VkExec *ex) { (void)ex; }
static void vk_dlaa_hud_target_report(VkExec *ex) { (void)ex; }
static void vk_dlaa_hud_draw_emitted(VkExec *ex) { (void)ex; }
static void resource_image_written(uint64_t image,uint64_t addr){(void)image;(void)addr;}
typedef struct { unsigned skip,image; uint64_t addr; } VkTexImage;
typedef struct {const GfxShaderRes *resource;uint64_t handle,size;VkTexImage image;VkDescriptorImageInfo descriptor;
                VkPinnedTextureWrite pinned;int written,sampled;void *write_target,*independent;} EffectImageCapture;
static void vk_texture_write_target_bound(void *target){assert(!target);}
typedef struct SelectedStorageWriters SelectedStorageWriters;
typedef struct {
    EffectImageCapture *images;
    uint32_t image_count;
    int first_write_bound;
    SelectedStorageWriters *selected_writers;
} EffectDeclaration;
static void selected_storage_writers_issued(VkExec *ex, EffectDeclaration *declaration) {
    assert(ex && (!declaration || !declaration->selected_writers));
}
static EffectDeclaration *effect_current;
static int storage_phase_current;
static EffectImageCapture captured;
static int use_capture;
static unsigned capture_prepares,pinned_prepares,pinned_writes,pinned_reads,hud_writes;
static EffectImageCapture *effect_image_capture(const GfxShaderRes *r,uint64_t h){return use_capture && captured.resource==r && captured.handle==h?&captured:NULL;}
static void vk_texture_image_write_prepare_locked(const VkTexImage *image,uint64_t size){
 assert(image->image==101&&image->addr==1&&size==64);capture_prepares++;}
static void vk_texture_pinned_write_prepare(const VkPinnedTextureWrite *p){assert(p->pin&&p->image==101);pinned_prepares++;}
static void vk_texture_pinned_write_recorded(const VkPinnedTextureWrite *p){assert(p->pin&&p->image==101);pinned_writes++;}
static void vk_texture_pinned_write_use(const VkPinnedTextureWrite *p){assert(p->pin&&p->image==201);pinned_reads++;}
static void vk_dlaa_hud_write(VkImage image){assert(image==101||image==102);hud_writes++;}
typedef int view_t;
static VkTexImage images[4];
static unsigned locks,unlocks,lookups,exact_calls,writable_calls,bound_calls;
static int depth,deferred,fail_at,registration,recording_failed;
static int vk_exec_recording_failed(void){return recording_failed;}
static int vk_texture_registration_expected_locked(int logical,VkImage *expected){
    assert(depth&&logical>=0&&!expected);return registration;
}
static void vk_exec_registration_gone(VkExec *ex,int logical){
    assert(ex&&logical==1&&depth);recording_failed=1;
}
static void rt_fatal(const char *message,...) { (void)message; abort(); }
static int nvn_stage_of(unsigned stage) { assert(stage<6);return (int)stage; }
#define NVN_HANDLE_TEX(h) ((h)&0xfffff)
static void vk_texture_lock(void){assert(!depth);depth=1;locks++;}
static void vk_texture_unlock(void){assert(depth==1);depth=0;unlocks++;}
static const VkTexImage *vk_texture_by_id_locked(uint64_t id,void *v,void *hv){
    if(v)*(view_t *)v=0;if(hv)*(int *)hv=0;assert(depth==1 && id<4);lookups++;
    if((int)id==fail_at){deferred=1;return NULL;}
    return &images[id];
}
static const VkTexImage *vk_texture_current_registration_locked(int logical,void *v,void *hv){
    return vk_texture_by_id_locked((uint64_t)logical,v,hv);
}
static int registered_slot_locked(int logical){return logical;}
static const VkTexImage *vk_texture_exact_by_id_locked(uint64_t id,void *v,void *hv){
    (void)v;(void)hv;assert(depth==1 && id<4);exact_calls++;images[id].skip=0;return &images[id];
}
static const VkTexImage *vk_texture_writable_locked(const VkTexImage *img){
    assert(depth==1);writable_calls++;((VkTexImage *)img)->image=100+(unsigned)(img-images);return img;
}
static int vk_texture_deferred(void){return deferred;}
static unsigned get_view(VkExec *ex,const VkTexImage *img,view_t *v,int hv,int type,int storage){
    (void)ex;(void)v;(void)hv;(void)type;assert(storage);return img->image;
}
static unsigned selected_calls,full_resolves;
static void binding_manifest_select(VkExec *ex,uint32_t index,VkImage image){(void)ex;(void)index;(void)image;selected_calls++;}
static unsigned res_view_type(int type){(void)type;return VK_IMAGE_VIEW_TYPE_2D;}
static unsigned dummy_index(unsigned type,int ms){(void)type;(void)ms;return 0;}
static unsigned storage_dummy(unsigned index,int format){(void)index;(void)format;return 88;}
static void vk_exec_required_graphics_unavailable(VkExec *ex,uint32_t opcode,const char *message){(void)ex;(void)opcode;(void)message;abort();}
static void tex_info_full(VkExec *ex,int stage,uint32_t member,const GfxShaderRes *r,uint64_t h,int storage,int separate,
VkDescriptorImageInfo *out,VkImage *image,int demanded,VkTextureReadKey *key,int *recorded){(void)stage;(void)member;(void)recorded;
(void)ex;(void)r;(void)h;(void)storage;(void)separate;assert(demanded);full_resolves++;
*out=(VkDescriptorImageInfo){0,999,1};*image=999;if(key)*key=(VkTextureReadKey){0};}
static const VkTexImage *vk_dlaa_swap_img;
static VkImageView vk_dlaa_swap_view;
static VkImage vk_dlaa_swap_output;
static int vk_dlaa_swap_active(VkExec *ex){(void)ex;return 0;}
static int vk_dlaa_swap_matches(const VkTexImage *source){(void)source;return 0;}
static void vk_dlaa_swap_use(VkExec *ex){(void)ex;abort();}
static void vk_texture_storage_written(VkImage image,uint64_t addr){
    assert(image>=100 && addr==image-100);bound_calls++;
}
/* production preflight inserted here */
static void reset(void){
    locks=unlocks=lookups=exact_calls=writable_calls=bound_calls=0;
    depth=deferred=registration=recording_failed=0;
    fail_at=-1;use_capture=0;capture_prepares=selected_calls=full_resolves=0;
    pinned_prepares=pinned_writes=pinned_reads=hud_writes=0;effect_current=NULL;storage_phase_current=0;
    for(unsigned i=0;i<4;i++)images[i]=(VkTexImage){.skip=1,.image=7,.addr=i};
}
int main(void){
    DrawCtx cx={0};VkExec ex={.dc=&cx};GfxShaderRes r={.slot=0,.exact=1,.flags=4};
    VkPipeInfo info={.n_bindings=1,.bindings={{.kind=9,.type=2,.count=1,.stage=3,.res=&r}}};
    ex.img[3][0]=1;ex.img[3][1]=2;
    reset();assert(prepare_storage_images(&ex,&info));assert(!locks && !lookups);
    info.bindings[0].kind=GFX_RES_IMG;info.bindings[0].type=9;
    reset();assert(prepare_storage_images(&ex,&info));assert(!locks && !lookups);
    info.bindings[0].type=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;info.bindings[0].count=2;
    reset();r.flags=0;assert(prepare_storage_images(&ex,&info));assert(!locks && !cx.n_storage_writes);r.flags=4;
    reset();assert(prepare_storage_images(&ex,&info));
    assert(locks==1 && unlocks==1 && !depth && lookups==2 && exact_calls==2 && writable_calls==2 && !bound_calls);
    assert(cx.n_storage_writes==2&&cx.n_storage_images==2&&selected_calls==2);
    VkDescriptorImageInfo descriptor;VkImage selected;
    images[1].image=777; // Installation must consume the frozen writable selection, not resolve the slot again.
    tex_info(&ex,3,0,&r,1,1,0,&descriptor,&selected,NULL,NULL);
    assert(selected==101&&descriptor.imageView==101&&!full_resolves&&lookups==2&&writable_calls==2);
    images[1].image=101;
    storage_images_written(&ex);assert(bound_calls==2 && !cx.n_storage_writes);
    storage_images_written(&ex);assert(bound_calls==2);
    assert(images[1].image==101 && images[2].image==102 && images[3].image==7);
    reset();fail_at=2;assert(!prepare_storage_images(&ex,&info));
    assert(locks==1 && unlocks==1 && !depth && lookups==2 && writable_calls==1 && !bound_calls && !cx.n_storage_writes);
    reset();r.slot=31;info.bindings[0].count=3;ex.img[3][31]=3;
    assert(prepare_storage_images(&ex,&info));assert(lookups==1 && writable_calls==1);
    reset();r.slot=-1;info.bindings[0].count=2;
    assert(prepare_storage_images(&ex,&info));assert(lookups==1 && writable_calls==1);
    reset();r.slot=0;info.bindings[0].count=1;use_capture=1;
    captured=(EffectImageCapture){.resource=&r,.handle=1,.size=64,.image={.image=101,.addr=1},.descriptor={.imageView=55}};
    assert(prepare_storage_images(&ex,&info));
    assert(!lookups&&!writable_calls&&locks==1&&unlocks==1&&capture_prepares==1&&cx.n_storage_writes==1);
    captured.image.image=199;captured.descriptor.imageView=99;
    tex_info(&ex,3,0,&r,1,1,0,&descriptor,&selected,NULL,NULL);
    assert(selected==101&&descriptor.imageView==55&&!full_resolves); // exact front capture was copied once.
    storage_images_written(&ex);assert(bound_calls==1); // immutable front binding used despite mutable slot image.
    reset();use_capture=1;
    captured=(EffectImageCapture){.resource=&r,.handle=1,.size=64,.image={.image=101,.addr=1},
                                  .descriptor={.imageView=55},.pinned={&r,101},.written=1};
    assert(prepare_storage_images(&ex,&info));
    assert(pinned_prepares==1&&!capture_prepares&&!lookups&&!bound_calls&&!pinned_writes);
    EffectImageCapture exact[] = {captured,{.descriptor={.imageView=66},.pinned={&r,201}}};
    EffectDeclaration declaration={.images=exact,.image_count=2};effect_current=&declaration;
    images[1].image=777;
    storage_images_written(&ex);
    assert(pinned_writes==1&&pinned_reads==1&&hud_writes==1&&!bound_calls&&!cx.n_storage_writes);
    reset();use_capture=1;captured=exact[0];storage_phase_current=1;
    assert(prepare_storage_images(&ex,&info)&&cx.n_storage_images==1);
    assert(!pinned_prepares&&!capture_prepares&&!lookups&&!cx.n_storage_writes);
    reset();info.bindings[0].count=2;ex.img[3][0]=ex.img[3][1]=1;
    assert(prepare_storage_images(&ex,&info));
    assert(cx.n_storage_images==1&&cx.n_storage_writes==1&&writable_calls==1&&selected_calls==1);
    reset();registration=VK_TEXTURE_REGISTRATION_EXACT;
    assert(!prepare_storage_images(&ex,&info)&&recording_failed&&!cx.n_storage_images&&!lookups);
    assert(!cx.n_storage_writes&&!selected_calls&&!writable_calls&&locks==1&&unlocks==1);
    puts("texture writable preflight: sampled bypass, storage arrays, exactness and balanced deferral passed");
    return 0;
}
'''


CONTENT_FIXTURE = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>
typedef uintptr_t VkImage,VkCommandBuffer,VkBuffer;
typedef int VkResult;
typedef uint64_t VkDeviceSize;
typedef void *VmaAllocation;
typedef unsigned VkImageType,VkFormat,VkSampleCountFlagBits;
typedef struct { unsigned flags,type,format,w,h,d,levels,layers,samples,usage; } ImgKey;
#include "vk_texture_material.h"
#include "texture_source.h"
#include "resource_core_budget.h"
#include "resource_core_registry.h"
#include "resource_texture_source.h"
RcRegistry *rc_registry_runtime(void){return NULL;}
static RcRegistryPosition vk_exec_resource_position(void){return (RcRegistryPosition){0};}
static uint64_t source_revision=1,source_mapping=1,source_protection=1;
static unsigned source_race_observation;
int rc_registry_read_observation_current(RcRegistry *r,const RcRegistryReadObservation *o){(void)r;(void)o;return 1;}
static unsigned read_commits;
RcRegistryResult rc_registry_read_commit(RcRegistry *r,const RcRegistryReadObservation *o,
 const RcPreparedInput *i,RcRegistryReadVersion *out){(void)r;read_commits++;*out=(RcRegistryReadVersion){1,*o,*i};return RC_REGISTRY_OK;}

int rc_registry_observation_current(RcRegistry *r,const RcRegistryObservation *o){(void)r;return !o->handle.id||o->revision==source_revision;}
static RcBudgetAvailability available={.progress=1,.available.amount={[RC_BUDGET_RAM]=UINT64_C(1)<<30,
    [RC_BUDGET_COMMIT]=UINT64_C(1)<<30,[RC_BUDGET_GPU_FIRST]=UINT64_C(1)<<30,[RC_BUDGET_STAGING]=UINT64_C(1)<<30}};
static unsigned availability_calls,reserve_calls;
static int snapshot_refused, pool_ctx;
static unsigned snapshot_materializations, snapshot_copy_mark;
static int test_snapshot_admission;
static uint64_t snapshot_ram, snapshot_commit;
static int restore_growth_refused, restore_reuse_plan, paired_failure;
static uint64_t paired_attempt_release, paired_peak_bytes;
static unsigned restore_reuse_attempts;
void resource_core_budget_availability(RcBudgetAvailability *out){availability_calls++;*out=available;}
RcAdmissionStatus resource_core_budget_reserve(RcBudgetProducer producer,int required,
    const RcAdmissionVector *cost,RcBudgetTicket *out){
    reserve_calls++;
    paired_peak_bytes=cost->amount[RC_BUDGET_GPU_FIRST];
    if(test_snapshot_admission && producer==RC_BUDGET_SNAPSHOT &&
       (!required || cost->amount[RC_BUDGET_RAM]>snapshot_ram || cost->amount[RC_BUDGET_COMMIT]>snapshot_commit))
        return RC_ADMISSION_REFUSED;
    if(snapshot_refused || (restore_growth_refused && cost->amount[RC_BUDGET_GPU_FIRST]))
        return RC_ADMISSION_REFUSED;
    (void)required;(void)cost;*out=(RcBudgetTicket){.producer=producer};out->token.ledger=(RcAdmission *)(uintptr_t)1;
    return RC_ADMISSION_OK;}
RcAdmissionStatus resource_core_budget_materialize(RcBudgetTicket *t,const RcAdmissionVector *cost){
    assert(t->token.ledger);
    if(t->producer==RC_BUDGET_SNAPSHOT) snapshot_materializations++;
    (void)cost;return RC_ADMISSION_OK;}
RcAdmissionStatus resource_core_budget_retire(RcBudgetTicket *t){assert(t->token.ledger);return RC_ADMISSION_OK;}
RcAdmissionStatus resource_core_budget_destroyed(RcBudgetTicket *t){assert(t->token.ledger);*t=(RcBudgetTicket){0};return RC_ADMISSION_OK;}
RcAdmissionStatus resource_core_budget_cancel(RcBudgetTicket *t){return resource_core_budget_destroyed(t);}
enum { VK_NULL_HANDLE=0,VK_IMAGE_TYPE_2D=1,VK_IMAGE_USAGE_TRANSFER_SRC_BIT=1,VK_IMAGE_USAGE_TRANSFER_DST_BIT=2,
NVN_MEMORY_POOL_FLAGS_VIRTUAL=4,NVN_MEMORY_POOL_FLAGS_PHYSICAL=8,VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO=1,VK_IMAGE_TILING_OPTIMAL=1,
VK_IMAGE_LAYOUT_UNDEFINED=0,VK_IMAGE_LAYOUT_GENERAL=1,VK_SUCCESS=0,ALL_RW=7,VK_PIPELINE_STAGE_2_COPY_BIT=8,
VK_ACCESS_2_TRANSFER_READ_BIT=1,VK_ACCESS_2_TRANSFER_WRITE_BIT=2,VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE=1 };
enum {VK_NOT_READY=1,VK_ERROR_OUT_OF_DEVICE_MEMORY=-1,VK_ERROR_OUT_OF_HOST_MEMORY=-2,
      VK_ERROR_INITIALIZATION_FAILED=-3,VK_ERROR_FORMAT_NOT_SUPPORTED=-4,VK_ERROR_TOO_MANY_OBJECTS=-5};
#define UPLOAD_PART (32ull<<20)
typedef struct { unsigned width,height,depth; } Extent;
typedef struct { unsigned sType,flags,imageType,format; Extent extent; unsigned mipLevels,arrayLayers,samples,tiling,usage,initialLayout; } VkImageCreateInfo;
typedef struct { unsigned aspectMask,mipLevel,baseArrayLayer,layerCount; } VkImageSubresourceLayers;
typedef struct { VkImageSubresourceLayers srcSubresource,dstSubresource; Extent extent; } VkImageCopy;
typedef struct {VkDeviceSize bufferOffset;unsigned bufferRowLength,bufferImageHeight;VkImageSubresourceLayers imageSubresource;Extent imageExtent;} VkBufferImageCopy;
typedef struct { unsigned vma; } VkCtx;
typedef struct { uint64_t size; } VmaAllocationInfo;
typedef struct {unsigned usage,flags;} VmaAllocationCreateInfo;
#define VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT 1u
typedef struct {RcBudgetReadiness before;RcAdmissionVector peak,reuse;uint64_t released;
                uint32_t memory_type;int selected,owner_full,can_reuse,reuse_unattempted;} RcVmaDependency;
typedef struct {RcBudgetProducer producer;RcBudgetTicket *peak_ticket;int required;RcVmaDependency *dependency;} RcVmaRequest;
typedef struct {uint8_t *map;VkBuffer buffer;} VkPreparedStaging;
typedef struct VkTextureRestore VkTextureRestore;
typedef struct VkTextureWriteTarget {
    TextureMaterialUse source_use;
    VkTextureRestore *seed;
    void *cpu_seed;
    int seeded, viewed, bound;
    ImgKey key;
    VkImage image;
    VmaAllocation allocation;
    uint64_t bytes;
} VkTextureWriteTarget;
typedef struct {VkTextureWriteTarget **targets;uint32_t count;} VkTextureWritePlan;
// This fixture covers joint immutable-image COW. Dirty CPU seed staging has its own producer fixture.
static int write_target_cpu_peak(void *seed, RcAdmissionVector *peak) { (void)seed; (void)peak; abort(); }
static int write_target_cpu_stage(VkTextureWriteTarget *target, RcBudgetTicket *peak) {
    (void)target; (void)peak; abort();
}
static void write_target_cpu_rollback(void *seed) { assert(!seed); }
static int texture_reclaim_on_heap(VmaAllocation allocation,uint32_t domain){return !domain||(uintptr_t)allocation==domain;}
typedef struct { uint64_t addr; unsigned layout; } TexKey;
typedef struct { VkImage image; uint64_t addr; unsigned fb_learned,gone,gpu_owned,bound,mirrored,skip,samples,type,levels,layers,usage,width,height,aspect; } VkTexImage;
typedef struct { unsigned depth,stencil,astc_bw,expand_rgb; } TexVkFormat;
typedef struct {unsigned format,width,height;} nvn_texdesc;
typedef struct { uint8_t *mem; unsigned pool_flags; struct { uint64_t size; } L; nvn_texdesc d; uint64_t serial; } tex_t;
typedef struct { const TextureSourceSpan *sample_spans; uint32_t sample_count; } TextureSourceMapping;
typedef struct Tex { TexKey key; VkTexImage pub; TexVkFormat f; ImgKey ik,material_intent;
tex_t t; const tex_t *source_recipe;
TextureMaterial *material; VmaAllocation alloc,bound_to; uint64_t bytes,source_copy_size,up_retry,upload_mapping,upload_protection;
struct Tex *material_prev,*material_next;
uint64_t checked_authority; unsigned checked;
RcBudgetTicket source_budget;
RcBudgetAvailability source_retry;uint32_t source_retry_domains,source_retry_submit;
RcAdmissionVector source_retry_peak;
uint8_t source_retry_source;uint64_t source_retry_handle,source_retry_revision,source_retry_guard;
TextureSourceStamp source_retry_stamp;
int last_ch,ph;
uint8_t *source_copy; uint8_t source_digest[32]; TextureSourceStamp source_stamp;
TextureSourceStamp source_copy_stamp;
TextureSourceAuthority source_authority,source_copy_authority;
RcRegistryPosition native_write;
RcRegistrySourceAccess source_access;
TextureSourceMapping *source_mapping;
RcRegistryReadVersion source_copy_version,material_version;
unsigned source_refusal,source_refusal_result;
uint8_t source_snapshot_mismatch;
unsigned material_writable,gpu_dirty,sfill,flat3d,superseded,overwritten,retiring,released,uploaded,
source_digest_valid,source_blocked,source_canonical,source_borrowed,material_restoring,bc1_try,material_prepared,reusable,reused,content_slot,n_up,chash_ok;
struct Tex *next_at; } Tex;
typedef struct { Tex *e; uint32_t refs; } PRef;
static PRef *pref;
static int npref;
static TextureMaterialLedger material_ledger;
static void content_use_image_add(TextureMaterialInstance *instance){(void)instance;}
static void content_use_image_remove(TextureMaterialInstance *instance){(void)instance;}
static uint64_t rel_bytes;static unsigned submit_epoch=1;
static unsigned by_addr,by_key;
static Tex *head;
static unsigned native_lock,restore_batch_uses;
static void lock(void){assert(!native_lock);native_lock=1;}
static void unlock(void){assert(native_lock);native_lock=0;}
static void make_key(const tex_t *t,TexKey *key){*key=(TexKey){(uintptr_t)t->mem,9};}
void resource_core_budget_readiness(int required,RcBudgetReadiness *out){
 assert(required);*out=(RcBudgetReadiness){available.available,UINT32_MAX,1};}
struct TextureUseBatch {void *head;unsigned buckets[1];};
static TextureUseBatch *texture_use_active(uint64_t fence,int create){
 static TextureUseBatch batch;assert(native_lock&&fence==17&&create);return &batch;}
static void texture_use_insert(void **head,unsigned *buckets,TextureMaterialInstance *instance,void *none){
 (void)head;(void)buckets;assert(native_lock&&instance->value.image&&!none);restore_batch_uses++;}
static unsigned generations,retired,purged,copies,allocations,held,hashes,observes;
static unsigned n_checks;
static VkImage last_retired;static uint64_t reclaim_credit;
static int allocation_failure,invalid_mapping,dirty_after,expect_fatal,tex_deferred,decode_failure,source_changed,provider_failure;
static int stage_failure;
static unsigned stage_attempts,stage_records,stage_cancels;
static uint8_t recovery_stage[128];
static uint8_t *change_guest_on_stage;
static jmp_buf fatal_jump;
static TextureSourceStamp current_stamp={1,1,1,3};
static uint8_t physical[2048][64];static unsigned next_image=100;
static unsigned mip(unsigned n,unsigned l){return n>>l ? n>>l:1;}
static int nvn_gpu_only(uint64_t addr){(void)addr;return 0;}
static Tex *nvn_hm_get(unsigned *map,const void *key){
 const uint64_t *addr=key;(void)map;return head && head->key.addr==*addr?head:NULL;}
static void vk_exec_image_gone(VkImage image){assert(image);purged++;}
static void vk_defer_image_reclaim(VkImage image,VmaAllocation alloc,uint64_t credit){
assert(image && alloc);retired++;last_retired=image;reclaim_credit+=credit;}
static void vk_defer_image(VkImage image,VmaAllocation alloc){vk_defer_image_reclaim(image,alloc,0);}
static void vk_defer_image_notify(VkImage image,VmaAllocation alloc,uint64_t credit,void (*done)(void *),void *arg){
vk_defer_image_reclaim(image,alloc,credit);done(arg);}
typedef struct { VkImage image; } VkMemoryImageHold;
static VkMemoryImageHold reclamation_view_hold;
static VkImage blocked_raw_image;
static unsigned completed_image_destroys;
static VkMemoryImageHold *vk_mem_image_hold(VkImage image) {
    assert(image && !reclamation_view_hold.image);
    reclamation_view_hold = (VkMemoryImageHold){image};
    return &reclamation_view_hold;
}
static void vk_mem_image_release(VkMemoryImageHold *hold) {
    assert(hold == &reclamation_view_hold && hold->image);
    hold->image = 0;
}
static int vk_mem_image_held(VkImage image) { return image == blocked_raw_image; }
static int vk_mem_image_destroy_completed(VkImage image, VmaAllocation alloc, uint64_t scene, uint64_t upload,
                                          void (*done)(void *), void *arg) {
    assert(image && alloc && !reclamation_view_hold.image && image != blocked_raw_image && (scene || upload));
    retired++; last_retired = image; completed_image_destroys++;
    done(arg);
    return 1;
}
static uint64_t vk_upload_dependency(void){return 1;}
static uint64_t vk_upload_recording_sequence(VkCommandBuffer cb){assert(cb);return 9;}
static uint64_t vk_exec_recording_fence(void){return 17;}
static void tex_gen_bump(uint64_t addr){assert(addr);generations++;}
static void image_owner_replaced(Tex *e,VkImage next){(void)e;(void)next;}
static int source_spans_init(Tex *e){return e->t.L.size!=0;}
static int source_observe(Tex *e,const TextureSourceStamp *prior,TextureSourceObservation *out){
observes++;if(source_race_observation==observes){source_revision++;source_race_observation=0;}
e->source_authority=(TextureSourceAuthority){.legacy={.handle.id=1,.revision=source_revision,
    .content_revision=source_revision,.guard=1,.range={1,e->pub.addr,e->t.L.size}}};
*out=(TextureSourceObservation){.stamp=current_stamp};
if(invalid_mapping && observes>1){out->stamp.mapping++;return TEX_SOURCE_UNKNOWN;}
if(dirty_after && observes>1){out->stamp.generation++;return TEX_SOURCE_DIRTY;}
if(source_changed && texture_source_stamp_valid(prior) && prior->generation!=current_stamp.generation)return TEX_SOURCE_DIRTY;
return texture_source_stamp_valid(prior)?TEX_SOURCE_CLEAN:TEX_SOURCE_UNKNOWN;}
static int source_observe_prior(Tex *e,const TextureSourceStamp *prior,TextureSourceObservation *out,
                                const TextureSourceAuthority *capture){
 assert(capture);return source_observe(e,prior,out);
}
static void source_commit(Tex *e,const TextureSourceStamp *stamp,int same_contents){(void)same_contents;e->source_stamp=*stamp;}
static int source_permission_at(Tex *e,RcRegistryPosition p,RcRegistryReadObservation *out){
 (void)p;if(out)*out=e->source_access._content.ordered;return rc_registry_position_valid(e->source_access._content.ordered.position);}
static int source_permission_current(Tex *e){return texture_authority_current(rc_registry_runtime(),&e->source_access._content,vk_exec_resource_position());}
static int mem_read_hold(uint64_t addr,uint64_t size){assert(addr && size && !held);held=1;return 1;}
static const RcRegistrySourceAccess *source_access_anchor(const Tex *e){return &e->source_access;}
static int source_snapshot_hold(Tex *e){return mem_read_hold(e->key.addr,e->t.L.size);}
/* production snapshot copy */
enum { SOURCE_REFUSAL_SNAPSHOT_CAPACITY, SOURCE_REFUSAL_SNAPSHOT_MAPPING, SOURCE_REFUSAL_SNAPSHOT_CHANGED,
       SOURCE_REFUSAL_SNAPSHOT_COMMIT, SOURCE_REFUSAL_SNAPSHOT_MISSING, SOURCE_REFUSAL_RACE };
static void source_note_refusal(Tex *e,int stage,RcRegistryResult result){
 if(!e->source_refusal){e->source_refusal=stage;e->source_refusal_result=result;}
}
static int source_snapshot_current(Tex *e,const TextureSourceAuthority *proof){
 (void)e;return texture_authority_current(rc_registry_runtime(),proof,vk_exec_resource_position());}
static void mem_read_release(void){
 assert(held);held=0;
 if(test_snapshot_admission) snapshot_copy_mark=snapshot_materializations;
}
static int mem_mapping_range_unchanged(uint64_t addr,uint64_t size,uint64_t stamp){assert(addr&&size);return stamp>=source_mapping;}
static int mem_protection_range_unchanged(uint64_t addr,uint64_t size,uint64_t stamp){assert(addr&&size);return stamp>=source_protection;}
static int content_image_eligible(const Tex *e);
/* production snapshot verification */
static uint64_t kern_now_ns(void){return 1;}
typedef struct { uint8_t digest[32]; } blake3_hasher;
static void blake3_hasher_init(blake3_hasher *h){memset(h,0,sizeof *h);}
static void blake3_hasher_update(blake3_hasher *h,const void *p,size_t n){
const uint8_t *b=p;hashes++;for(size_t i=0;i<n;i++)h->digest[i%32]=(uint8_t)(h->digest[i%32]*33+b[i]);}
static void blake3_hasher_finalize(blake3_hasher *h,uint8_t *out,size_t n){assert(n==32);memcpy(out,h->digest,n);}
static VkCtx *vk_ctx(void){static VkCtx ctx;return &ctx;}
static VkImage suballoc_image(VkCtx *c,const VkImageCreateInfo *ci,VmaAllocation *alloc){(void)c;(void)ci;(void)alloc;return 0;}
static int vma_image(VkCtx *c,const VkImageCreateInfo *ci,const ImgKey *k,VkImage *out,VmaAllocation *alloc){
(void)c;(void)ci;(void)k;allocations++;if(allocation_failure)return -1;*out=next_image++;*alloc=(void *)(uintptr_t)*out;return VK_SUCCESS;}
#define SUBALLOC_MAX 1024
static int resource_core_vma_image_peak(const RcVmaRequest *req,const VkImageCreateInfo *ci,
 const VmaAllocationCreateInfo *ai,VmaAllocationCreateInfo *placement,RcAdmissionVector *peak){
 assert(req->producer==RC_BUDGET_TEXTURE&&!req->peak_ticket&&(ci->extent.width==4||ci->extent.width==8));
 *placement=*ai;*peak=(RcAdmissionVector){.amount[RC_BUDGET_GPU_FIRST]=ci->extent.width*16};
 if(req->dependency)*req->dependency=(RcVmaDependency){.selected=1,.can_reuse=restore_reuse_plan,
     .memory_type=0,.released=paired_attempt_release};
 return VK_SUCCESS;
}
static int resource_core_vma_create_image(const RcVmaRequest *req,unsigned vma,const VkImageCreateInfo *ci,
 const VmaAllocationCreateInfo *ai,VkImage *out,VmaAllocation *allocation,VmaAllocationInfo *info,RcAdmissionStatus *status){
 (void)vma;assert((req->peak_ticket ? req->peak_ticket->token.ledger != NULL :
     (ai->flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) != 0) && ai->usage==VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
 if(status)*status=RC_ADMISSION_OK;
 if(ai->flags&VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT){
    assert(restore_growth_refused && restore_reuse_plan);restore_reuse_attempts++;
    assert(req->dependency);*req->dependency=(RcVmaDependency){.selected=1,.can_reuse=1,.released=12};
 }
 if(ci->extent.width==8 && paired_failure){
    assert(status && req->dependency && info);
    *req->dependency=(RcVmaDependency){.selected=1,.memory_type=0,.can_reuse=restore_reuse_plan,
        .released=paired_attempt_release,.owner_full=paired_failure==2};
    if(paired_failure>=2){*status=RC_ADMISSION_NO_RECORDS;return VK_NOT_READY;}
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
 }
 if(info)info->size=128;
 return vma_image(vk_ctx(),ci,NULL,out,allocation);
}
static void resource_core_vma_destroy_image(unsigned vma,VkImage image,VmaAllocation allocation){
 (void)vma;assert(image&&allocation);retired++;last_retired=image;paired_attempt_release++;
}
static void resource_core_vma_dependency_rebase(RcVmaDependency *dep, int required){
 assert(required);
 dep->released=paired_attempt_release;
 dep->reuse_unattempted=0;
 resource_core_budget_readiness(required,&dep->before);
}
static int vk_staging_prepare_plan(VkDeviceSize bytes,VkPreparedStaging *stage,RcAdmissionVector *peak){
 assert((bytes==64||bytes==128)&&!stage->map);
 *peak=(RcAdmissionVector){.amount[RC_BUDGET_STAGING]=bytes};return VK_SUCCESS;
}
static int vk_staging_prepare_allocate(VkPreparedStaging *stage,RcBudgetTicket *peak,int required){
 assert((required==0||required==1)&&peak->token.ledger&&!stage->map);stage_attempts++;if(stage_failure)return -1;
 if(change_guest_on_stage){change_guest_on_stage[0]++;change_guest_on_stage=NULL;}
 stage->map=recovery_stage;return VK_SUCCESS;
}
static void vk_staging_prepare_cancel(VkPreparedStaging *stage){if(stage->map)stage_cancels++;stage->map=NULL;}
static VkBuffer vk_staging_prepare_record(VkCommandBuffer cb,VkPreparedStaging *stage){
 assert(cb&&stage->map==recovery_stage);stage->map=NULL;stage_records++;return 1;
}
static int vk_staging_prepare_available(VkDeviceSize size){assert(size==64||size==128);return 0;}
static uint64_t vk_staging_prepare_supplier(VkDeviceSize size,uint64_t after){assert(size==64||size==128);(void)after;return 0;}
static int nvn_fmt_bpb(unsigned format,int *w,int *h){(void)format;*w=*h=1;return 4;}
static int nvn_tex_is_linear(const nvn_texdesc *d){(void)d;return 0;}
static int nvn_deswizzle(const nvn_texdesc *d,uint32_t lv,uint32_t ly,const uint8_t *src,uint8_t *dst,uint32_t row){
 assert(d->width==4&&d->height==4&&!lv&&!ly&&row==16&&src&&dst);if(decode_failure)return 1;
 memcpy(dst,src,64);return 0;
}
static void expand_rgb(unsigned format,uint8_t *src,uint8_t *dst,size_t texels){(void)format;(void)src;(void)dst;(void)texels;abort();}
static uint32_t content_recovery_domains(const Tex *e){(void)e;return (1u<<RC_BUDGET_RAM)|(1u<<RC_BUDGET_COMMIT)|(1u<<RC_BUDGET_GPU_FIRST);}
static void rt_fatal(const char *fmt,...){(void)fmt;assert(expect_fatal);longjmp(fatal_jump,1);}
static void vmaGetAllocationInfo(unsigned vma,VmaAllocation alloc,VmaAllocationInfo *out){(void)vma;assert(alloc);out->size=64;}
static VkCommandBuffer vk_upload_begin(void){return 1;}
static void vk_upload_end(void){}
static void content_image_forget(Tex *e);
static void init_image(Tex *e,VkCommandBuffer cb){assert(cb);
/* production image initialization invalidation */
memset(physical[e->pub.image],0,64);}
static void barrier(VkCommandBuffer cb,unsigned a,unsigned b,unsigned c){(void)a;(void)b;(void)c;assert(cb);}
static void vkCmdCopyImage(VkCommandBuffer cb,VkImage src,unsigned sl,VkImage dst,unsigned dl,unsigned n,const VkImageCopy *regions){
(void)sl;(void)dl;assert(cb && src!=dst && n==1 && regions[0].extent.width==4);copies++;memcpy(physical[dst],physical[src],64);}
static void vkCmdCopyBufferToImage(VkCommandBuffer cb,VkBuffer src,VkImage dst,unsigned layout,unsigned n,const VkBufferImageCopy *r){
 assert(cb&&src==1&&layout==VK_IMAGE_LAYOUT_GENERAL&&n==1&&r->imageExtent.width==4&&r->imageExtent.height==4);
 assert(r->bufferOffset+64<=sizeof recovery_stage);
 memcpy(physical[dst],recovery_stage+r->bufferOffset,64);
}
static void tex_image_prepare(const Tex *e,uint64_t size){assert(e->pub.image&&e->pub.addr&&size);}
static void *content_malloc(size_t n);
#define malloc content_malloc
#include "vk_texture_content.h"
#undef malloc
static void *content_malloc(size_t n){if(provider_failure && n==sizeof(MaterialSource))return NULL;return malloc(n);}
static void content_upload_reconstructed(Tex *e){
assert(e->material_restoring && e->source_borrowed && e->source_recipe && e->source_recipe->mem==e->source_copy);
TextureSourceObservation observation={0};
/* production upload material provenance */
if(decode_failure){e->up_retry=1;content_source_discard(e);return;}
memcpy(physical[e->pub.image],e->source_copy,64);e->uploaded=1;e->n_up++;
int no_mem=0,source_complete=1;const tex_t *t=e->source_recipe;
/* production upload material publication */
content_source_discard(e);}
static int up_wait(Tex *e){(void)e;return 0;}
static int vk_exec_turn_due(void){return 0;}
static void upload(Tex *e,const void *dirty,int tracked,const TextureSourceObservation *observed){
(void)dirty;(void)tracked;(void)observed;assert(e->source_copy && !e->material);
if(decode_failure){e->uploaded=0;e->up_retry=1;content_source_discard(e);return;}
memcpy(physical[e->pub.image],e->source_copy,64);e->uploaded=1;e->n_up++;
source_commit(e,&e->source_copy_stamp,0);content_image_publish(e,e->source_digest,e->source_copy_size);content_source_discard(e);}
static VkResult vk_texture_write_target_allocate(VkTextureWriteTarget *target, RcVmaDependency *dependency) {
    (void)target;(void)dependency;
    abort(); // paired fixture starts from a nonresident source
}
/* production paired write target */
/* production optional recovery */
/* production exact restore */
/* production compound storage admission */
/* production residency requirement */
/* production clean recovery loop */
static Tex make_tex(uint64_t addr,uint8_t *bytes){
Tex e={0};e.key=(TexKey){addr,9};e.pub=(VkTexImage){.addr=addr,.mirrored=1,.samples=1,.type=1,.levels=1,.layers=1,.usage=3,.width=4,.height=4,.aspect=1};
e.t.mem=bytes;e.t.L.size=64;e.t.d=(nvn_texdesc){37,4,4};e.ik=(ImgKey){.type=1,.format=37,.w=4,.h=4,.d=1,.levels=1,.layers=1,.samples=1,.usage=3};e.material_intent=e.ik;return e;}
static void publish_snapshot(Tex *e,VkImage image){
e->pub.image=image;e->alloc=(void *)(uintptr_t)image;e->bytes=64;e->uploaded=1;
assert(e->source_copy && e->source_digest_valid);memcpy(physical[image],e->source_copy,64);
source_commit(e,&e->source_copy_stamp,0);
content_image_publish(e,e->source_digest,64);content_source_discard(e);assert(e->material && !e->alloc && !e->bytes);}
static void required_snapshot_admission(void){
 uint8_t bytes[64];memset(bytes,0x5a,sizeof bytes);
 Tex selected=make_tex((uintptr_t)bytes,bytes);
 selected.source_canonical=1;
 snapshot_ram=snapshot_commit=sizeof bytes;
 test_snapshot_admission=1;
 unsigned materializations=snapshot_materializations;
 assert(!content_early_prepare_policy(&selected,MATERIAL_PREPARE_OPTIONAL) && !selected.source_copy);
 pool_ctx=1;
 assert(!content_early_prepare(&selected) && !selected.source_copy);
 assert(pool_ctx==1);pool_ctx=0;
 snapshot_ram--;
 assert(!content_early_prepare(&selected) && !selected.source_copy);
 snapshot_ram++;
 snapshot_commit--;
 assert(!content_early_prepare(&selected) && !selected.source_copy);
 snapshot_commit++;
 assert(!content_early_prepare(&selected) && selected.source_copy && !held);
 assert(!memcmp(selected.source_copy,bytes,sizeof bytes));
 assert(snapshot_copy_mark==materializations && snapshot_materializations==materializations+1);
 content_source_discard(&selected);
 test_snapshot_admission=0;
}
static void ordered_restore_contract(void){
 uint8_t bytes[64];memset(bytes,0xb4,sizeof bytes);
 Tex selected=make_tex((uintptr_t)bytes,bytes);selected.t.serial=73;head=&selected;
 observes=0;assert(!content_early_prepare(&selected));publish_snapshot(&selected,199);
 selected.source_canonical=1;
 TextureMaterial *material=selected.material;
 submit_epoch++;assert(vk_texture_restore_probe_evict(&selected.t) && !material->resident.image);
 VkTexImage shape;VkTextureRestore *restore=vk_texture_restore_capture(&selected.t,&shape);
 assert(restore && !shape.image && restore->source->restore_refs==1 && material->refs==2);
 // The selected immutable provider survives changing the address and its stored bytes while parked.
 head=NULL;memset(bytes,0x79,sizeof bytes);
 unsigned alloc_before=allocations,stage_before=stage_attempts,record_before=stage_records,retire_before=retired;
 uint64_t restored_before=vk_texture_restore_probe_count();
 RcVmaDependency dependency;
 snapshot_refused=1;
 assert(vk_texture_restore_prepare(restore,&dependency)==VK_NOT_READY && dependency.selected);
 assert(dependency.peak.amount[RC_BUDGET_GPU_FIRST]==64 && dependency.peak.amount[RC_BUDGET_STAGING]==64);
 assert(allocations==alloc_before && stage_attempts==stage_before && stage_records==record_before);
 assert(!restore->copy.ticket.token.ledger && restore->source->restore_refs==1);
 snapshot_refused=0;allocation_failure=1;
 assert(vk_texture_restore_prepare(restore,&dependency)==VK_ERROR_OUT_OF_DEVICE_MEMORY);
 assert(stage_attempts==stage_before && !restore->copy.ticket.token.ledger);
 allocation_failure=0;stage_failure=1;
 assert(vk_texture_restore_prepare(restore,&dependency)==VK_ERROR_OUT_OF_DEVICE_MEMORY);
 assert(stage_attempts==stage_before+1 && retired==retire_before+1 && !restore->image && !restore->copy.ticket.token.ledger);
 stage_failure=0;allocation_failure=1;restore_growth_refused=restore_reuse_plan=1;
 assert(vk_texture_restore_prepare(restore,&dependency)==VK_NOT_READY && dependency.can_reuse && dependency.released==12);
 assert(restore_reuse_attempts==1 && stage_attempts==stage_before+1 && !restore->copy.ticket.token.ledger);
 assert(dependency.reuse.amount[RC_BUDGET_GPU_FIRST]==0 && dependency.reuse.amount[RC_BUDGET_STAGING]==64);
 allocation_failure=0;
 assert(vk_texture_restore_prepare(restore,&dependency)==VK_SUCCESS && stage_records==record_before);
 assert(restore_reuse_attempts==2);restore_growth_refused=restore_reuse_plan=0;
 assert(!material->resident.image && restore->source->restore_refs==1);
 const VkTexImage *picture=vk_texture_restore_publish(restore);
 assert(picture && picture->image && vk_texture_restore_probe_count()==restored_before+1);
 assert(physical[picture->image][0]==0xb4 && physical[picture->image][63]==0xb4);
 assert(stage_records==record_before+1 && !restore->image && !restore->copy.ticket.token.ledger);
 vk_texture_restore_use(restore);assert(restore_batch_uses==1);
 vk_texture_restore_cancel(restore);
 assert(!((MaterialSource *)material->provider)->restore_refs && material->refs==1);
 content_release(&selected,1);content_collect(UINT64_MAX);
 assert(!material_snapshot_bytes && !material_ledger.owners);
}

static void paired_restore_contract(void){
 uint8_t bytes[64];memset(bytes,0x72,sizeof bytes);
 Tex selected=make_tex((uintptr_t)bytes,bytes);selected.t.serial=83;head=&selected;
 observes=0;assert(!content_early_prepare(&selected));publish_snapshot(&selected,205);
 selected.source_canonical=1;submit_epoch++;
 assert(vk_texture_restore_probe_evict(&selected.t));
 VkTexImage shape;VkTextureRestore *restore=vk_texture_restore_capture(&selected.t,&shape);
 assert(restore && !shape.image);
 uint8_t prior_bytes[64];memset(prior_bytes,0x94,sizeof prior_bytes);
 Tex prior=make_tex((uintptr_t)prior_bytes,prior_bytes);prior.t.serial=84;head=&prior;
 observes=0;assert(!content_early_prepare(&prior));publish_snapshot(&prior,206);
 prior.source_canonical=1;submit_epoch++;
 assert(vk_texture_restore_probe_evict(&prior.t));
 VkTextureRestore *seed=vk_texture_restore_capture(&prior.t,&shape);
 assert(seed && !shape.image);
 VkTextureWriteTarget target={.key=prior.ik,.seed=seed};
 target.key.w=8;
 vk_texture_restore_pair_write_target(restore,&target);
 uint64_t initial_records=stage_records;
 unsigned initial_alloc=allocations, initial_stage=stage_attempts;
 RcVmaDependency dependency;
 snapshot_refused=1;
 assert(vk_texture_restore_prepare(restore,&dependency)==VK_NOT_READY);
 assert(paired_peak_bytes==192 && dependency.peak.amount[RC_BUDGET_STAGING]==128);
 assert(dependency.peak.amount[RC_BUDGET_RAM]==2*sizeof(VkBufferImageCopy)+sizeof(TextureMaterialInstance));
 assert(allocations==initial_alloc && stage_attempts==initial_stage && !target.image);
 snapshot_refused=0;restore_growth_refused=restore_reuse_plan=1;
 paired_failure=1;
 uint64_t before=paired_attempt_release;
 assert(vk_texture_restore_prepare(restore,&dependency)==VK_NOT_READY && dependency.can_reuse);
 assert(dependency.released==paired_attempt_release && paired_attempt_release==before+1);
 assert(!restore->image && !target.image && stage_attempts==initial_stage && !restore->copy.ticket.token.ledger);
 assert(!restore->copy.regions && !seed->copy.regions && !target.seeded);
 paired_failure=2;
 assert(vk_texture_restore_prepare(restore,&dependency)==VK_NOT_READY && dependency.owner_full==2);
 assert(!restore->staging_missing && !restore->image && !target.image && !restore->copy.ticket.token.ledger);
 paired_failure=3;
 assert(vk_texture_restore_prepare(restore,&dependency)==VK_ERROR_TOO_MANY_OBJECTS && !dependency.selected);
 assert(!restore->image && !target.image && !restore->copy.ticket.token.ledger);
 paired_failure=0;restore_growth_refused=restore_reuse_plan=0;
 assert(vk_texture_restore_prepare(restore,&dependency)==VK_SUCCESS);
 assert(restore->image && target.image && target.bytes==128 && stage_records==initial_records);
 const VkTexImage *image=vk_texture_restore_publish(restore);
 assert(image && image->image && physical[image->image][63]==0x72 && target.image);
 assert(target.seeded && physical[target.image][63]==0x94 && !prior.material->resident.image);
 assert(stage_records==initial_records+1 && allocations==initial_alloc+5 && !restore->copy.ticket.token.ledger);
 assert(!seed->copy.regions && !restore->copy.regions);
 vk_texture_restore_cancel(restore);
 assert(target.image); // the transfer, not source publication, owns its admitted destination
 uint64_t released=paired_attempt_release;
 write_target_release_allocation(&target);
 assert(released==paired_attempt_release); // the uploaded seed retires with native completion
 vk_texture_restore_cancel(seed);
 content_release(&prior,1);content_release(&selected,1);content_collect(UINT64_MAX);
 assert(!material_snapshot_bytes && !material_ledger.owners);
}
static void multiple_storage_contract(void){
 VkTextureWriteTarget a={.key.w=4},b={.key.w=8};
 VkTextureWriteTarget *targets[]={&a,&b};
 const VkTextureWritePlan plan={targets,2};
 RcVmaDependency dependency;
 unsigned alloc_before=allocations,stage_before=stage_attempts;
 snapshot_refused=1;
 assert(vk_texture_write_plan_allocate(&plan,&dependency)==VK_NOT_READY);
 assert(paired_peak_bytes==192 && !a.image && !b.image && allocations==alloc_before);
 snapshot_refused=0;restore_growth_refused=restore_reuse_plan=1;
 uint64_t release_before=paired_attempt_release;
 paired_failure=1; // The zero-cost reusable placement failed on the second owner.
 assert(vk_texture_write_plan_allocate(&plan,&dependency)==VK_NOT_READY);
 assert(!a.image && !b.image && release_before+1==dependency.released && dependency.can_reuse);
 paired_failure=2;
 assert(vk_texture_write_plan_allocate(&plan,&dependency)==VK_NOT_READY && dependency.owner_full==2);
 assert(!a.image && !b.image);
 paired_failure=0;restore_growth_refused=restore_reuse_plan=0;
 assert(vk_texture_write_plan_allocate(&plan,&dependency)==VK_SUCCESS);
 assert(a.image && b.image && a.image!=b.image && paired_peak_bytes==192 && stage_attempts==stage_before);
 unsigned after=allocations;
 assert(vk_texture_write_plan_allocate(&plan,&dependency)==VK_SUCCESS && allocations==after);
 write_target_release_allocation(&a);write_target_release_allocation(&b);
}
int main(void){
uint8_t bytes[64];memset(bytes,3,sizeof bytes);Tex a=make_tex(4096,bytes);
assert(!content_early_prepare(&a));assert(a.source_copy && material_snapshot_bytes==64 && hashes==1 && !held);
uint8_t original[64];memcpy(original,a.source_copy,64);memset(bytes,9,64);
assert(!memcmp(a.source_copy,original,64));publish_snapshot(&a,1);TextureMaterial *old=a.material;
assert(!memcmp(physical[1],original,64) && old->refs==1 && old->cached && material_ledger.resident_bytes==64);
Tex b=make_tex(8192,original);observes=0;unsigned adopted_generation=generations;
assert(content_early_prepare(&b));assert(generations==adopted_generation+1 && b.n_up==1);
assert(b.material==old && old->refs==2 && !b.source_copy && material_snapshot_bytes==64 && !allocations);
assert(old->provider && material_ledger.provider_bytes==64 && rc_input_exact(&old->input));
// Verifying the same immutable owner must preserve every binding-facing identity and lifetime count.
VkTexImage unchanged_pub=b.pub;unsigned unchanged_uploads=b.n_up,unchanged_generation=generations;
unsigned unchanged_refs=old->refs,unchanged_live=old->live_refs,unchanged_hashes=hashes;
uint64_t unchanged_resident=material_ledger.resident_bytes;
observes=0;current_stamp.generation++;assert(content_early_prepare(&b));
assert(hashes==unchanged_hashes && b.source_stamp.generation==current_stamp.generation);
assert(!memcmp(&b.pub,&unchanged_pub,sizeof b.pub) && b.n_up==unchanged_uploads && generations==unchanged_generation);
assert(old->refs==unchanged_refs && old->live_refs==unchanged_live && material_ledger.resident_bytes==unchanged_resident);
content_image_forget(&b);observes=0;current_stamp.generation++;
assert(content_early_prepare(&b));assert(hashes==unchanged_hashes+1 && !b.source_copy && material_snapshot_bytes==64);
assert(b.source_stamp.generation==current_stamp.generation && b.source_digest_valid);
assert(!memcmp(&b.pub,&unchanged_pub,sizeof b.pub) && b.n_up==unchanged_uploads && generations==unchanged_generation);
assert(old->refs==unchanged_refs && old->live_refs==unchanged_live && material_ledger.resident_bytes==unchanged_resident);
assert(material_stats.same_owner_hits==2);
unsigned hash_count=hashes;Tex same=make_tex(8192,original);observes=0;
assert(content_early_prepare(&same));assert(hashes==hash_count && same.material==old && old->refs==3);
assert(material_stats.same_owner_hits==2);
content_release(&same,0);assert(!retired);
unsigned before=generations;content_write_begin(&b,1);
assert(b.pub.image!=1 && !b.material && a.material==old && old->refs==1 && old->cached);
assert(copies==1 && generations==before+1 && !memcmp(physical[b.pub.image],physical[1],64));
physical[b.pub.image][0]=44;assert(physical[1][0]==3);
Tex replace=make_tex(24576,original);observes=0;assert(content_early_prepare(&replace));
content_write_begin(&replace,0);assert(copies==1 && !replace.material && replace.pub.image!=1 && physical[replace.pub.image][0]==0);
assert(old->refs==1 && a.material==old);
Tex incompatible=make_tex(28672,original);incompatible.material_intent.format++;
observes=0;assert(!content_early_prepare(&incompatible) && incompatible.source_copy && !incompatible.material);
content_source_discard(&incompatible);
Tex physical_tex=make_tex(36864,original);physical_tex.t.pool_flags=NVN_MEMORY_POOL_FLAGS_PHYSICAL;
assert(!content_image_eligible(&physical_tex) && !content_early_prepare(&physical_tex) && !physical_tex.source_copy);
physical_tex.t.pool_flags=NVN_MEMORY_POOL_FLAGS_VIRTUAL;
assert(!content_image_eligible(&physical_tex) && !content_early_prepare(&physical_tex) && !physical_tex.source_copy);
Tex writable=make_tex(32768,original);writable.material_writable=1;unsigned before_hash=hashes;
assert(!content_early_prepare(&writable) && !writable.source_copy && hashes==before_hash);
unsigned cow_purges=purged;assert(cow_purges==2 && !retired);
allocation_failure=expect_fatal=1;VkImage prior=a.pub.image;
if(!setjmp(fatal_jump)){content_write_begin(&a,1);assert(0);}
assert(a.material==old && a.pub.image==prior && old->refs==1 && !retired && purged==cow_purges);
allocation_failure=expect_fatal=0;
assert(!content_release(&a,0));assert(old->refs==0 && old->cached && !retired);
assert(content_reclaim(64,0)==64);
assert(retired==1 && last_retired==1 && reclaim_credit==64 && !material_ledger.resident_bytes);
content_collect(UINT64_MAX);assert(!material_snapshot_bytes && !material_ledger.provider_bytes);
Tex stale=make_tex(8192,original);observes=0;assert(!content_early_prepare(&stale));assert(stale.source_copy);content_source_discard(&stale);
Tex bad=make_tex(12288,original);observes=0;invalid_mapping=1;
assert(!content_early_prepare(&bad) && !bad.source_copy && !material_snapshot_bytes && !held);invalid_mapping=0;
const TextureSourceSpan sample_spans[]={{3,7},{32,20}};
TextureSourceMapping shaped={sample_spans,2};
Tex sparse=make_tex(147456,original);sparse.source_mapping=&shaped;
sparse.material_intent.format+=17;
observes=0;assert(!content_early_prepare(&sparse) && sparse.source_copy && !held);
for(unsigned i=0;i<64;i++)assert(sparse.source_copy[i]==((i>=3&&i<10)||(i>=32&&i<52)?original[i]:0));
content_source_discard(&sparse);assert(!material_snapshot_bytes);
Tex race=make_tex(16384,original);observes=0;dirty_after=1;
assert(!content_early_prepare(&race));assert(!race.source_copy && !material_snapshot_bytes && !held);
content_source_discard(&race);dirty_after=0;
// Live physical eviction keeps one raw provider and every logical association; an old recording owns its own
// instance until completion/cancellation, even after the same material is reconstructed into another image.
Tex live=make_tex(45056,original),peer=make_tex(49152,original);observes=0;
assert(!content_early_prepare(&live));publish_snapshot(&live,2);TextureMaterial *material=live.material;
observes=0;assert(content_early_prepare(&peer) && peer.material==material);
assert(material->associations==&peer && peer.material_next==&live && live.material_prev==&peer);
TextureMaterialUse recording={0};
assert(texture_material_use_begin(material,&recording)==RC_LIFETIME_OK);
uint64_t identity=material->id,instance=material->resident.id;unsigned frees_before=retired;
submit_epoch++;assert(content_reclaim(64,1)==64);
assert(!live.pub.image && !peer.pub.image && !live.uploaded && !peer.uploaded);
assert(material->refs==2 && material->provider && material_snapshot_bytes==64);
assert(material_ledger.pending_bytes==64 && !material_ledger.deferred_bytes && retired==frees_before);
texture_material_set_live(material,0);live.released=1;live.t.mem=NULL;
unsigned restore_allocations=allocations,restore_uploads=live.n_up;
assert(content_restore(&live,material,NULL));
assert(allocations==restore_allocations+1 && live.n_up==restore_uploads+1);
assert(material->id==identity && material->resident.id!=instance && live.pub.image && live.uploaded);
assert(!memcmp(physical[live.pub.image],original,64) && !memcmp(physical[recording.instance->value.image],original,64));
assert(material_snapshot_bytes==64 && material_ledger.provider_bytes==64 && !live.source_copy);
observes=0;assert(content_early_prepare(&peer) && peer.pub.image==live.pub.image);
for(unsigned alias=0;alias<128;alias++) {
 assert(content_require_resident(&live) && content_require_resident(&peer));
 assert(live.pub.image==peer.pub.image && live.material==material && peer.material==material);
}
assert(allocations==restore_allocations+1 && live.n_up==restore_uploads+1 && material_stats.restored==1);
assert(material_snapshot_bytes==64 && material_ledger.provider_bytes==64);
assert(texture_material_use_cancel(&recording)==RC_LIFETIME_OK);content_collect(UINT64_MAX);
assert(!material_ledger.pending_bytes && !material_ledger.deferred_bytes && retired==frees_before+1);
// Allocation and conversion failures preserve the nonresident identity/provider and expose no partial image.
submit_epoch++;assert(content_reclaim(64,1)==64);content_collect(UINT64_MAX);
allocation_failure=1;tex_deferred=0;
assert(!content_restore(&peer,material,&current_stamp) && tex_deferred && !peer.pub.image);
assert(peer.material==material && material->id==identity && material_snapshot_bytes==64);
allocation_failure=0;decode_failure=1;tex_deferred=0;
unsigned failed_retired=retired;restore_allocations=allocations;
assert(!content_restore(&peer,material,&current_stamp) && tex_deferred && !peer.pub.image && !material->resident.image);
assert(material_snapshot_bytes==64 && allocations==restore_allocations+1 && retired==failed_retired+1);decode_failure=0;tex_deferred=0;
// Finalization after physical eviction remains an exact obligation, and restoration ignores changed guest bytes.
texture_material_set_live(material,0);peer.released=1;peer.t.mem=bytes;
memset(bytes,77,64);assert(content_restore(&peer,material,NULL));
assert(!memcmp(physical[peer.pub.image],original,64) && material->id==identity);
// Shared pressure cannot collect earlier native users or return another heap's charge.
const VkImage to_reclaim = material->resident.image;
const uint64_t use_fence = 31, ready_upload = material->resident.instance->upload_sequence;
material->resident.last_use_fence = use_fence;
submit_epoch++;
assert(!content_reclaim_completed(64,1,(uint32_t)to_reclaim+1,use_fence,ready_upload));
assert(!content_reclaim_completed(64,1,(uint32_t)to_reclaim,use_fence-1,ready_upload));
if (ready_upload) assert(!content_reclaim_completed(64,1,(uint32_t)to_reclaim,use_fence,ready_upload-1));
blocked_raw_image = to_reclaim;
assert(!content_reclaim_completed(64,1,(uint32_t)to_reclaim,use_fence,ready_upload));
blocked_raw_image = 0;
assert(texture_material_use_begin(material,&recording)==RC_LIFETIME_OK);
assert(!content_reclaim_completed(64,1,(uint32_t)to_reclaim,use_fence,ready_upload));
assert(texture_material_use_cancel(&recording)==RC_LIFETIME_OK);
assert(content_reclaim_completed(64,1,(uint32_t)to_reclaim,use_fence,ready_upload)==RC_BUDGET_RECLAIM_DESTROYED);
assert(completed_image_destroys==1 && !material->resident.image && material->provider);
assert(!content_reclaim_completed(64,1,(uint32_t)to_reclaim,use_fence,ready_upload));
content_collect(UINT64_MAX);
assert(!material_ledger.pending_bytes && !material_ledger.deferred_bytes);
assert(content_restore(&peer,material,NULL) && !memcmp(physical[peer.pub.image],original,64));
// Host pressure may drop raw data only while an available image still preserves those exact obligations.
assert(content_drop_providers(64)==64 && !material_snapshot_bytes && !material->provider);
submit_epoch++;assert(!content_reclaim(64,1) && material->resident.image);
content_release(&live,0);content_release(&peer,0);
assert(content_reclaim(64,0)==64);content_collect(UINT64_MAX);
assert(!material_ledger.owners && !material_ledger.resident_owners && !material_ledger.pending_instances);
assert(!tex_deferred && material_stats.evicted==3 && material_stats.restored==3);
// Current reacquisition works without a content-index entry; changed source bytes create another material while a
// finalized peer can still reconstruct the previous exact version. No regeneration reads through that peer.
uint8_t changing[64];memcpy(changing,original,64);
Tex current=make_tex(53248,changing),finalized=make_tex(57344,original);observes=0;
assert(!content_early_prepare(&current));publish_snapshot(&current,3);
observes=0;assert(content_early_prepare(&finalized));material=current.material;identity=material->id;
texture_material_set_live(material,0);finalized.released=1;finalized.t.mem=NULL;
submit_epoch++;assert(content_reclaim(64,1)==64);content_collect(UINT64_MAX);
content_unpin(material->index_slot-1,0);assert(!material->cached && material->refs==2);
observes=0;assert(content_require_resident(&current) && current.material==material && material->id==identity);
assert(!memcmp(physical[current.pub.image],original,64));
submit_epoch++;assert(content_reclaim(64,1)==64);content_collect(UINT64_MAX);
// A reported write of equal bytes still reinstalls the same representation after equality verification.
source_changed=1;current_stamp.generation++;observes=0;
assert(content_require_resident(&current) && current.material==material && material->id==identity);
submit_epoch++;assert(content_reclaim(64,1)==64);content_collect(UINT64_MAX);
memset(changing,23,64);current_stamp.generation++;observes=0;
assert(content_require_resident(&current) && !current.material && current.source_copy && current.source_digest_valid);
VkImage replacement=current.pub.image;publish_snapshot(&current,replacement);
assert(current.material->id!=identity && current.material->input.payload!=material->input.payload);
assert(!memcmp(physical[replacement],changing,64) && material_snapshot_bytes==128);
// Failed cutoff/mapping publication cannot publish the scratch observation as the material's current read version.
current.source_authority.ordered.position=(RcRegistryPosition){9,10};
current.source_access._content.ordered.position=(RcRegistryPosition){8,6};
current.source_stamp=(TextureSourceStamp){0};unsigned before_commit=read_commits;
content_version(&current,current.material);assert(!current.material_version.id && read_commits==before_commit);
current.source_stamp=current_stamp;content_version(&current,current.material);
assert(read_commits==before_commit+1 && current.material_version.origin.position.scene==8);
assert(content_require_resident(&finalized) && finalized.material==material && material->id==identity);
assert(!memcmp(physical[finalized.pub.image],original,64));source_changed=0;
content_release(&current,1);content_release(&finalized,1);content_collect(UINT64_MAX);
assert(!material_snapshot_bytes && !material_ledger.owners && !material_ledger.resident_owners);
// A clean compatibility upload is not permanently excluded after optional host admission once refused.
uint8_t retry_bytes[64];memset(retry_bytes,51,64);
Tex retry=make_tex(61440,retry_bytes);retry.pub.image=90;retry.alloc=(void *)(uintptr_t)90;
retry.bytes=64;retry.uploaded=1;retry.source_stamp=current_stamp;memset(physical[90],27,64);
available.available.amount[RC_BUDGET_RAM]=available.available.amount[RC_BUDGET_COMMIT]=0;available.progress++;
snapshot_refused=1;assert(!content_early_prepare(&retry) && retry.source_retry_domains && !retry.source_copy);
unsigned retry_reserves=reserve_calls,retry_hashes=hashes,advisories=availability_calls;
for(unsigned i=0;i<100;i++)assert(make_fresh_source(&retry));
assert(reserve_calls==retry_reserves && hashes==retry_hashes && availability_calls==advisories);
submit_epoch++;for(unsigned i=0;i<100;i++)assert(make_fresh_source(&retry));
assert(reserve_calls==retry_reserves && availability_calls==advisories+1);
// Unrelated GPU relief cannot make a refused host capture fit.
available.available.amount[RC_BUDGET_GPU_FIRST]++;available.progress++;submit_epoch++;
assert(make_fresh_source(&retry) && reserve_calls==retry_reserves);
snapshot_refused=0;available.available.amount[RC_BUDGET_RAM]=available.available.amount[RC_BUDGET_COMMIT]=8192;
available.progress++;submit_epoch++;stage_failure=1;
unsigned no_image=allocations, no_record=stage_records, new_stage=stage_attempts;
assert(make_fresh_source(&retry)&&retry.pub.image==90&&!retry.material&&!retry.source_copy);
assert(allocations==no_image&&stage_records==no_record&&stage_attempts==new_stage+1);
available.available.amount[RC_BUDGET_GPU_FIRST+1]++;available.progress++;submit_epoch++;
assert(make_fresh_source(&retry)&&stage_attempts==new_stage+1); // Wrong heap cannot rearm the whole prepared request.
stage_failure=0;available.available.amount[RC_BUDGET_STAGING]++;available.progress++;submit_epoch++;allocation_failure=1;
assert(make_fresh_source(&retry) && retry.pub.image==90 && retry.uploaded && !retry.material && !retry.source_copy);
assert(!material_snapshot_bytes && physical[90][0]==27&&stage_records==no_record&&stage_cancels==1);allocation_failure=0;
unsigned failed_allocations=allocations;retry_reserves=reserve_calls;
// Returning to the same allowance after other reservations is not new room for this refusal.
available.progress++;submit_epoch++;
for(unsigned i=0;i<100;i++)assert(make_fresh_source(&retry));
assert(allocations==failed_allocations && reserve_calls==retry_reserves);
available.available.amount[RC_BUDGET_GPU_FIRST]++;available.progress++;submit_epoch++;decode_failure=1;
assert(make_fresh_source(&retry) && retry.pub.image==90 && retry.uploaded && !retry.material && !retry.source_copy);
assert(!material_snapshot_bytes && physical[90][0]==27);decode_failure=0;
// A failure after material/index ownership was established must detach it before restoring the old Tex value.
available.available.amount[RC_BUDGET_GPU_FIRST]++;available.progress++;submit_epoch++;provider_failure=1;
assert(make_fresh_source(&retry) && retry.pub.image==90 && retry.uploaded && !retry.material && !retry.source_copy);
content_collect(UINT64_MAX);assert(!material_snapshot_bytes && !material_ledger.owners && physical[90][0]==27);
for(unsigned i=0;i<MATERIAL_SLOTS;i++)assert(!material_slots[i].material);
provider_failure=0;change_guest_on_stage=retry_bytes;
available.available.amount[RC_BUDGET_GPU_FIRST]++;available.progress++;submit_epoch++;
assert(make_fresh_source(&retry) && retry.material && retry.material->provider && retry.pub.image!=90);
assert(retry_bytes[0]==52&&physical[retry.pub.image][0]==51);retry_bytes[0]=51;
assert(!memcmp(physical[retry.pub.image],retry_bytes,64) && physical[90][0]==27 && !retry.source_retry_domains);
retry_hashes=hashes;failed_allocations=allocations;
submit_epoch++;for(unsigned i=0;i<100;i++)assert(make_fresh_source(&retry));
assert(hashes==retry_hashes && allocations==failed_allocations);
// A matching immutable provider adopts its actual image; no new image is certified from the old compatibility one.
Tex matching=make_tex(65536,retry_bytes);matching.pub.image=91;matching.alloc=(void *)(uintptr_t)91;
matching.bytes=64;matching.uploaded=1;matching.source_stamp=current_stamp;
content_recovery_wait(&matching,(1u<<RC_BUDGET_RAM)|(1u<<RC_BUDGET_COMMIT));
available.available.amount[RC_BUDGET_RAM]++;available.progress++;submit_epoch++;
assert(make_fresh_source(&matching) && matching.material==retry.material && matching.pub.image==retry.pub.image);
assert(allocations==failed_allocations && material_snapshot_bytes==64);
content_release(&matching,0);content_release(&retry,1);content_collect(UINT64_MAX);
assert(!material_snapshot_bytes && !material_ledger.owners);
// An optional hit on a nonresident exact provider must not enter required reconstruction and retire the fallback.
Tex sleeping=make_tex(98304,retry_bytes);assert(!content_early_prepare(&sleeping));publish_snapshot(&sleeping,95);
TextureMaterial *sleeping_material=sleeping.material;submit_epoch++;assert(content_reclaim(64,1)==64);
content_collect(UINT64_MAX);assert(!sleeping.pub.image && sleeping_material->provider);
Tex delayed=make_tex(102400,retry_bytes);delayed.pub.image=96;delayed.alloc=(void *)(uintptr_t)96;
delayed.bytes=64;delayed.uploaded=1;delayed.source_stamp=current_stamp;memset(physical[96],29,64);
content_recovery_wait(&delayed,(1u<<RC_BUDGET_RAM)|(1u<<RC_BUDGET_COMMIT));
available.available.amount[RC_BUDGET_RAM]++;available.progress++;submit_epoch++;
unsigned old_retired=retired;failed_allocations=allocations;allocation_failure=1;
assert(make_fresh_source(&delayed) && delayed.pub.image==96 && delayed.uploaded && !delayed.material);
assert(allocations==failed_allocations && retired==old_retired && !delayed.source_copy && material_snapshot_bytes==64);
assert(delayed.source_retry_domains && physical[96][0]==29);allocation_failure=0;
assert(content_restore(&sleeping,sleeping_material,&current_stamp));
available.available.amount[RC_BUDGET_GPU_FIRST]++;available.progress++;submit_epoch++;
assert(make_fresh_source(&delayed) && delayed.material==sleeping_material && delayed.pub.image==sleeping.pub.image);
content_release(&delayed,0);content_release(&sleeping,1);content_collect(UINT64_MAX);
assert(!material_snapshot_bytes && !material_ledger.owners);
Tex bounded[5];uint8_t bounded_bytes[5][64];
for(unsigned i=0;i<5;i++){
memset(bounded_bytes[i],61+(int)i,64);bounded[i]=make_tex(69632+4096*i,bounded_bytes[i]);
bounded[i].pub.image=10+i;bounded[i].alloc=(void *)(uintptr_t)(10+i);bounded[i].bytes=64;
bounded[i].uploaded=1;bounded[i].source_stamp=current_stamp;
content_recovery_wait(&bounded[i],(1u<<RC_BUDGET_RAM)|(1u<<RC_BUDGET_COMMIT));}
available.available.amount[RC_BUDGET_RAM]++;available.progress++;submit_epoch++;
for(unsigned i=0;i<5;i++)assert(make_fresh_source(&bounded[i]));
assert(material_recovery.starts==4 && material_recovery.bytes==256);
for(unsigned i=0;i<4;i++)assert(bounded[i].material);
assert(!bounded[4].material && bounded[4].source_retry_domains);
submit_epoch++;assert(make_fresh_source(&bounded[4]) && bounded[4].material);
for(unsigned i=0;i<5;i++)content_release(&bounded[i],1);
content_collect(UINT64_MAX);assert(!material_snapshot_bytes && !material_ledger.owners && n_checks>=400);
// An authority race after the owned copy must not erase the recovery advisory. The retry remembers the
// pre-race identity even though the subsequent clean observation already sees the newer source revision.
uint8_t race_bytes[64];memset(race_bytes,81,64);
Tex raced=make_tex(110592,race_bytes);raced.pub.image=97;raced.alloc=(void *)(uintptr_t)97;
raced.bytes=64;raced.uploaded=1;raced.source_stamp=current_stamp;memset(physical[97],42,64);
content_recovery_wait(&raced,(1u<<RC_BUDGET_RAM)|(1u<<RC_BUDGET_COMMIT));
available.available.amount[RC_BUDGET_RAM]++;available.progress++;submit_epoch++;
source_race_observation=observes+3;
assert(make_fresh_source(&raced) && raced.source_retry_source && !raced.source_retry_domains);
assert(raced.source_retry_revision+1==source_revision && !raced.source_copy && !raced.material);
assert(raced.pub.image==97 && physical[97][0]==42 && !material_snapshot_bytes);
retry_reserves=reserve_calls;retry_hashes=hashes;uint64_t unchanged_progress=available.progress;
for(unsigned i=0;i<100;i++)assert(make_fresh_source(&raced));
assert(reserve_calls==retry_reserves && hashes==retry_hashes);
submit_epoch++;assert(make_fresh_source(&raced) && raced.material && raced.material->provider);
assert(available.progress==unchanged_progress && raced.pub.image!=97 && physical[97][0]==42);
assert(!memcmp(physical[raced.pub.image],race_bytes,64) && !raced.source_retry_source);
content_release(&raced,1);content_collect(UINT64_MAX);assert(!material_snapshot_bytes && !material_ledger.owners);
// A stable proof, unrelated capacity/registry activity, and globally advanced clean cutoffs do not retry.
Tex stable=make_tex(114688,race_bytes);stable.pub.image=98;stable.alloc=(void *)(uintptr_t)98;
stable.bytes=64;stable.uploaded=1;stable.source_stamp=current_stamp;
TextureSourceObservation retry_observation;source_observe(&stable,&stable.source_stamp,&retry_observation);
content_recovery_wait_source(&stable,&retry_observation.stamp,&stable.source_authority);
retry_reserves=reserve_calls;retry_hashes=hashes;
for(unsigned epoch=0;epoch<3;epoch++){
 available.progress++;available.available.amount[RC_BUDGET_GPU_FIRST]++;
 current_stamp.generation++;current_stamp.mapping++;current_stamp.protection++;submit_epoch++;
 for(unsigned i=0;i<100;i++)assert(make_fresh_source(&stable));
}
assert(reserve_calls==retry_reserves && hashes==retry_hashes && stable.source_retry_source);
// Only a change to this source's mapping, not another range's mapping serial, rearms the capture.
source_mapping=++current_stamp.mapping;submit_epoch++;
assert(make_fresh_source(&stable) && stable.material && !stable.source_retry_source);
content_release(&stable,1);content_collect(UINT64_MAX);assert(!material_snapshot_bytes && !material_ledger.owners);
// Source-event retries share the same bounded start/byte quota as capacity retries and need no budget event.
Tex events[5];uint8_t event_bytes[5][64];
for(unsigned i=0;i<5;i++){
 memset(event_bytes[i],91+(int)i,64);events[i]=make_tex(118784+4096*i,event_bytes[i]);
 events[i].pub.image=20+i;events[i].alloc=(void *)(uintptr_t)(20+i);events[i].bytes=64;
 events[i].uploaded=1;events[i].source_stamp=current_stamp;
 source_observe(&events[i],&events[i].source_stamp,&retry_observation);
 content_recovery_wait_source(&events[i],&retry_observation.stamp,&events[i].source_authority);
}
source_revision++;submit_epoch++;unchanged_progress=available.progress;
for(unsigned i=0;i<5;i++)assert(make_fresh_source(&events[i]));
assert(material_recovery.starts==4 && material_recovery.bytes==256 && available.progress==unchanged_progress);
for(unsigned i=0;i<4;i++)assert(events[i].material);
assert(!events[4].material && events[4].source_retry_source);
submit_epoch++;assert(make_fresh_source(&events[4]) && events[4].material);
for(unsigned i=0;i<5;i++)content_release(&events[i],1);
content_collect(UINT64_MAX);assert(!material_snapshot_bytes && !material_ledger.owners);
required_snapshot_admission();ordered_restore_contract();paired_restore_contract();multiple_storage_contract();
content_shutdown();assert(!material_snapshot_bytes && !rel_bytes);
puts("texture content integration: immutable provider, live eviction, exact reacquisition, COW and final-owner retirement passed");return 0;
}
'''


SOURCE_INTEGRATION_FIXTURE = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "texture_source.h"
typedef uint64_t VkDeviceSize;
typedef struct { uint64_t bufferOffset; struct { unsigned mipLevel,baseArrayLayer; } imageSubresource; } Region;
typedef struct { uint64_t src; unsigned level,layer; } AstcRegion;
typedef struct {uint64_t lo,hi;} VkDirtyPages;
typedef struct { int uploaded,last_ch,source_blocked,source_canonical,source_refusal,source_refusal_result; void *material,*source_copy;
struct {uint64_t addr;} pub;
unsigned source_count;TextureSourceStamp source_stamp,source_authority_stamp;TextureSourceSpan *source_spans; } Tex;
enum { SOURCE_REFUSAL_SNAPSHOT_MISSING=10, SOURCE_REFUSAL_SNAPSHOT_MAPPING, SOURCE_REFUSAL_SNAPSHOT_CHANGED,
SOURCE_REFUSAL_SNAPSHOT_COMMIT, SOURCE_REFUSAL_RACE, SRC_FOREIGN=41, SRC_MAPPING=42, SRC_CAPACITY=43,
RC_REGISTRY_UNSUPPORTED=7,RC_REGISTRY_STALE=2 };
static int spans_ok=1,overlap,wait_upload,query_state,turn_due,eligible=1,early_hit,tex_deferred;
static int promote,missing_mapping,capture,capture_denied;
static int previous_borrowed, previous_copy_changed;
static unsigned n_checks,commits,early_calls,uploads,source_checks,discards;
static int source_upload_retain_current(Tex *e){
 (void)e;return previous_borrowed || !previous_copy_changed;}
static void content_source_discard(Tex *e){
 assert(e->source_copy&&!previous_borrowed);e->source_copy=NULL;discards++;
}
static int source_spans_init(Tex *e){(void)e;return spans_ok;}
static int up_wait(Tex *e){(void)e;return wait_upload;}
static int source_observe(Tex *e,const TextureSourceStamp *prior,TextureSourceObservation *out){
(void)prior;source_checks++;e->source_canonical|=promote;e->source_blocked=overlap||missing_mapping;
e->source_refusal=overlap?SRC_FOREIGN:missing_mapping?SRC_MAPPING:0;
e->source_refusal_result=0;
*out=(TextureSourceObservation){.stamp=e->source_blocked?(TextureSourceStamp){0}:(TextureSourceStamp){1,1,1,9}};
e->source_authority_stamp=out->stamp;return query_state;}
static int source_stamp_equal(const TextureSourceStamp *a,const TextureSourceStamp *b){
return a->mapping==b->mapping && a->protection==b->protection && a->lineage==b->lineage && a->generation==b->generation;}
static void source_note_refusal(Tex *e,int stage,int result){
if(!e->source_refusal){e->source_refusal=stage;e->source_refusal_result=result;}}
static void source_commit(Tex *e,const TextureSourceStamp *stamp,int same_contents){(void)same_contents;commits++;e->source_stamp=*stamp;}
static void content_version(Tex *e,void *m){assert(e->material==m);}
static int vk_exec_turn_due(void){return turn_due;}
static int content_image_eligible(Tex *e){(void)e;return eligible;}
static int content_early_prepare(Tex *e){
early_calls++;if(capture)e->source_copy=e;
if(capture_denied)source_note_refusal(e,SRC_CAPACITY,0);
return early_hit;}
static int content_recovery_due(Tex *e){(void)e;return 0;}
static void content_recover(Tex *e,const TextureSourceObservation *o){(void)e;(void)o;assert(0);}
static void upload(Tex *e,const void *dirty,int tracked,const TextureSourceObservation *observed){
(void)e;(void)dirty;(void)tracked;assert(observed->stamp.generation==9);uploads++;}
/* production freshness function */
/* production upload source preparation */
static void guard_tests(void){
Tex e={.uploaded=1};turn_due=1;query_state=TEX_SOURCE_CLEAN;
assert(make_fresh_source(&e) && commits==1 && !tex_deferred && !early_calls && !uploads);
query_state=TEX_SOURCE_DIRTY;assert(make_fresh_source(&e));assert(tex_deferred && !early_calls && !uploads);
tex_deferred=0;eligible=0;e.material=&e;assert(make_fresh_source(&e));assert(tex_deferred && !early_calls && !uploads);
tex_deferred=0;e.material=NULL;assert(make_fresh_source(&e));assert(!tex_deferred && early_calls==1 && uploads==1);
turn_due=0;eligible=1;early_hit=1;assert(make_fresh_source(&e));assert(early_calls==2 && uploads==1);
early_hit=0;query_state=TEX_SOURCE_UNKNOWN;assert(make_fresh_source(&e));assert(uploads==2 && n_checks==6);
}
static void upload_source_tests(void){
Tex e={0};TextureSourceObservation selected={0};
unsigned early=early_calls,checks=source_checks;
overlap=1;assert(!make_fresh_source(&e) && !e.source_canonical && e.source_refusal==SRC_FOREIGN);
overlap=0;promote=capture=1;eligible=0;
assert(source_upload_prepare(&e,0,&selected) && e.source_canonical && e.source_copy==&e);
assert(early_calls==early+1 && source_checks==checks+2 && !e.source_blocked && !e.source_refusal);

e=(Tex){0};turn_due=1;
assert(!source_upload_prepare(&e,0,&selected) && tex_deferred && e.source_canonical && !e.source_copy);
assert(early_calls==early+1 && !e.source_refusal);
turn_due=tex_deferred=capture=0;

e=(Tex){0};overlap=1;
assert(!source_upload_prepare(&e,0,&selected) && e.source_refusal==SRC_FOREIGN && e.source_blocked);
e=(Tex){0};overlap=0;missing_mapping=1;
assert(!source_upload_prepare(&e,0,&selected) && e.source_refusal==SRC_MAPPING && e.source_blocked);
assert(early_calls==early+1);

e=(Tex){0};missing_mapping=0;capture_denied=1;
assert(!source_upload_prepare(&e,0,&selected) && e.source_refusal==SRC_CAPACITY && e.source_blocked);
assert(early_calls==early+2);
e=(Tex){0};capture_denied=0;early_hit=1;
assert(!source_upload_prepare(&e,0,&selected) && e.source_canonical && !e.source_blocked && !e.source_refusal);
assert(early_calls==early+3);

e=(Tex){.source_canonical=1};early_hit=0;
assert(!source_upload_prepare(&e,0,&selected) && e.source_refusal==SOURCE_REFUSAL_SNAPSHOT_MISSING);
assert(early_calls==early+4);
e=(Tex){0};promote=0;
assert(source_upload_prepare(&e,0,&selected) && !e.source_canonical && !e.source_copy);
assert(early_calls==early+4);
// Re-observation after an earlier capture lost its cutoff can recover exactly once at this operation.
TextureSourceObservation prior={.stamp={1,1,1,5}};
e=(Tex){.source_canonical=1,.source_refusal=SOURCE_REFUSAL_SNAPSHOT_CHANGED,.source_refusal_result=RC_REGISTRY_STALE};
checks=source_checks;capture=1;
assert(source_upload_prepare(&e,&prior,&selected) && e.source_copy==&e && selected.stamp.generation==9);
assert(source_checks==checks+1 && early_calls==early+5 && !e.source_refusal);
// No unconditional recapture or provenance erasure for an actual capacity or unresolved writer refusal.
e=(Tex){.source_canonical=1,.source_refusal=SRC_CAPACITY};
checks=source_checks;
assert(!source_upload_prepare(&e,&prior,&selected) && e.source_refusal==SRC_CAPACITY && e.source_blocked);
assert(source_checks==checks && early_calls==early+5);
e=(Tex){.source_canonical=1,.source_refusal=SOURCE_REFUSAL_SNAPSHOT_CHANGED,.source_refusal_result=RC_REGISTRY_STALE};
overlap=1;
assert(!source_upload_prepare(&e,&prior,&selected) && e.source_refusal==SRC_FOREIGN && e.source_blocked);
assert(source_checks==checks+1 && early_calls==early+5);
}
static void displaced_copy_tests(void) {
 unsigned old_byte=12;
 Tex e={.source_canonical=1,.source_copy=&old_byte};
 TextureSourceObservation out={0};
 unsigned before=early_calls;
 overlap=0;capture=1;previous_copy_changed=1;
 assert(source_upload_prepare(&e,NULL,&out) && e.source_copy==&e && early_calls==before+1 && discards==1);
 e=(Tex){.source_canonical=1,.source_copy=&old_byte};
 previous_copy_changed=0;
 assert(source_upload_prepare(&e,NULL,&out) && e.source_copy==&old_byte && early_calls==before+1 && discards==1);
 previous_copy_changed=1;previous_borrowed=1;
 assert(source_upload_prepare(&e,NULL,&out) && e.source_copy==&old_byte && early_calls==before+1 && discards==1);
 previous_borrowed=0;missing_mapping=1;
 assert(!source_upload_prepare(&e,NULL,&out) && !e.source_copy && e.source_refusal==SRC_MAPPING && discards==2);
 e=(Tex){.source_canonical=1,.source_copy=&old_byte};
 missing_mapping=0;overlap=1;
 assert(!source_upload_prepare(&e,NULL,&out) && !e.source_copy && e.source_refusal==SRC_FOREIGN && discards==3);
 overlap=0;
}
static const uint64_t layer_bytes=UINT64_C(2000)*3200*2;
static VkDeviceSize vk_astc_region_bytes(const AstcRegion *region,unsigned bw,unsigned bh){
(void)region;(void)bw;(void)bh;return layer_bytes;}
static void packing_test(unsigned mask,int gdsw,int snapshot,int raw_pages){
TextureSourceSpan spans[8];Region rg[8];AstcRegion ar[8];
for(unsigned i=0;i<8;i++){spans[i]=(TextureSourceSpan){i*layer_bytes,layer_bytes};
rg[i]=(Region){.bufferOffset=i*layer_bytes,.imageSubresource={0,i}};
ar[i]=(AstcRegion){.src=i*layer_bytes,.level=0,.layer=i};}
Tex value={.uploaded=1,.pub={4096},.source_count=8,.source_spans=spans};Tex *e=&value;
TextureSourceObservation observation={.stamp={4,5,6,7},.dirty={mask,0,0,0}};
const TextureSourceObservation *observed=raw_pages?NULL:&observation;
const VkDirtyPages pages={e->pub.addr+2*layer_bytes,e->pub.addr+4*layer_bytes};
const VkDirtyPages *dirty=raw_pages?&pages:NULL;
unsigned all_regions=8,n=8,k;int check=0,bc1=0,acheck=0,sk=0,partial=0;
int can_retain=gdsw;unsigned bw=4,bh=4;
uint64_t total=layer_bytes*8,src_bytes=total,prior_map=raw_pages?11:0,prior_protection=raw_pages?12:0;
uint64_t full_bytes=total;
/* production selective packing */
if(raw_pages){
    assert(partial==!snapshot && n==(snapshot?8:2));
    assert(prior_map==11 && prior_protection==12);
    assert(ar[0].layer==(snapshot?0:2) && ar[1].layer==(snapshot?1:3));
    return;
}
unsigned wanted=0;for(unsigned i=0;i<8;i++)wanted+=(mask>>i)&1;
int should_partial=!snapshot && wanted && wanted<8;
assert(partial==should_partial);
if(should_partial){
assert(n==wanted && src_bytes==layer_bytes*wanted && prior_map==4 && prior_protection==5);
if(!gdsw)assert(total==layer_bytes*wanted);
unsigned out=0;for(unsigned i=0;i<8;i++)if((mask>>i)&1){
if(gdsw)assert(ar[out].layer==i && ar[out].src==layer_bytes*i);
else assert(rg[out].imageSubresource.baseArrayLayer==i && rg[out].bufferOffset==layer_bytes*out);
out++;}
}else assert(n==8 && src_bytes==layer_bytes*8 && total==layer_bytes*8);
}
int main(void){guard_tests();upload_source_tests();displaced_copy_tests();for(unsigned mask=0;mask<256;mask++){
packing_test(mask,0,0,0);packing_test(mask,1,0,0);packing_test(mask,0,1,0);}
packing_test(0,1,0,1);packing_test(0,1,1,1);
puts("texture source integration: required late capture, worker turn, and sparse and raw-page upload masks passed");return 0;}
'''


def source_integration_fixture():
    source = (ROOT / 'runtime/vk_texture.c').read_text()
    start = source.index('static int make_fresh_source(')
    end = source.index('\n}\n', start) + 3
    function = source[start:end]
    start = source.index('static int source_upload_prepare(')
    end = source.index('\n}\n', start) + 3
    source_preparation = source[start:end]
    start = source.index('    int select = e->uploaded && !snapshot')
    end = source.index('    // A CPU decode or copy stages', start)
    packing = source[start:end]
    return SOURCE_INTEGRATION_FIXTURE.replace('/* production freshness function */', function).replace(
        '/* production upload source preparation */', source_preparation).replace(
        '/* production selective packing */', packing)


def content_fixture():
    source = (ROOT / 'runtime/vk_texture.c').read_text()
    start = source.index('static int content_require_resident(')
    end = source.index('\n// Whether e\'s storage', start)
    result = CONTENT_FIXTURE.replace('/* production residency requirement */', source[start:end])
    start = source.index('static void source_snapshot_copy(')
    end = source.index('\n}\n', start) + 3
    result = result.replace('/* production snapshot copy */', source[start:end])
    start = source.index('static int source_snapshot_verify(')
    end = source.index('\n}\n', start) + 3
    result = result.replace('/* production snapshot verification */', source[start:end])
    begin = source.index('static void init_image(Tex *e, VkCommandBuffer cb) {')
    begin = source.index('\n', begin) + 1
    finish = source.index('    VkImageMemoryBarrier2 ib', begin)
    result = result.replace('/* production image initialization invalidation */', source[begin:finish])
    begin = source.index('    int snapshot = e->source_copy != NULL;', source.index('static void upload(Tex *e,'))
    finish = source.index('    static volatile int32_t n_bad', begin)
    result = result.replace('/* production upload material provenance */', source[begin:finish])
    begin = source.index('    if (e->uploaded && !no_mem && source_complete)', finish)
    finish = source.index('    content_source_discard(e);', begin)
    result = result.replace('/* production upload material publication */', source[begin:finish])
    end = source.index('static int content_require_resident(')
    start = source.rindex('typedef struct {', 0, source.index('} TextureRecoveryCopy;'))
    restore_start = source.index('struct VkTextureRestore {', start)
    recover_start = source.index('// Promote only after a private image', restore_start)
    result = result.replace('/* production exact restore */', source[restore_start:recover_start])
    begin = source.index('int vk_texture_write_target_needs_allocation(')
    finish = source.index('VkResult vk_texture_write_target_allocate(', begin)
    result = result.replace('/* production paired write target */', source[begin:finish])
    begin = source.index('VkResult vk_texture_write_plan_allocate(')
    finish = source.index('VkImage vk_texture_write_target_prepared_image(', begin)
    result = result.replace('/* production compound storage admission */', source[begin:finish])
    result = result.replace('/* production optional recovery */',
                            '#define malloc content_malloc\n' + source[start:restore_start] + source[recover_start:end] + '#undef malloc\n')
    start = source.index('static int make_fresh_source(')
    end = source.index('\n}\n', start) + 3
    return result.replace('/* production clean recovery loop */', source[start:end])


def storage_fixture():
    source = (ROOT / 'runtime/vk_exec_draw.c').read_text()
    start = source.index('static int prepare_storage_images(')
    end = source.index('\n}\n', start) + 3
    begin = source.index('static void storage_images_written(', end)
    finish = source.index('\n}\n', begin) + 3
    body = source[start:end] + '\n' + source[begin:finish]
    begin = source.index('static void tex_info(')
    finish = source.index('\n}\n', begin) + 3
    body = source[begin:finish] + '\n' + body
    # The preflight must execute before any retained descriptor packet can be accepted.
    bind = source[source.index('static void bind_resources('):]
    preflight = bind.index('prepare_storage_images(ex, info)')
    for consumption in ('binding_recipe(ex, info)', 'prepared_instance(ex, recipe)', 'pb_entry(cx, info, g)'):
        assert preflight < bind.index(consumption), consumption
    assert 'bind_sig(ex, info)' not in bind
    return STORAGE_FIXTURE.replace('/* production preflight inserted here */', body)


def compile_and_run(name, source):
    marker = GENERATED_MARKER.decode() if isinstance(GENERATED_MARKER, bytes) else GENERATED_MARKER
    path = OUT / (name + '.c')
    path.write_text('// ' + marker + '\n' + source)
    exe = OUT / (name + '.exe')
    env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get('PATH', ''))
    subprocess.run([str(CC), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=undefined', '-fsanitize-trap=all', '-I', str(ROOT / 'runtime'),
                    str(path), '-o', str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    OUT.resolve().relative_to(ROOT.resolve())
    with tempfile.TemporaryDirectory(prefix='check-', dir=OUT) as temporary:
        OUT = Path(temporary)
        compile_and_run('ownership', OWNER_FIXTURE)
        compile_and_run('storage_preflight', storage_fixture())
        compile_and_run('content_transactions', content_fixture())
        compile_and_run('source_integration', source_integration_fixture())

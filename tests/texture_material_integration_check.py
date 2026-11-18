"""Exercise production texture ownership, content transactions and storage preflight without a game.

GPU allocation/recording, source observation and hashing are controlled boundaries in the content fixture.
It validates publication/ownership ordering, not Vulkan execution or cryptographic hash quality.
"""
from pathlib import Path
import os
import subprocess
import sys

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
    assert(m.image==101 && m.alloc==(void *)(uintptr_t)201 && m.ready);
    assert(ledger.owners==1 && ledger.resident_bytes==4096 && ledger.retained_bytes==4096);
    // A different logical source can reacquire the retained owner without another physical allocation.
    texture_material_acquire(&m,1);
    assert(m.refs==1 && m.live_refs==1 && ledger.live_bytes==4096 && !ledger.retained_bytes);
    assert(!texture_material_cache(&m,0) && texture_material_exclusive(&m));
    texture_material_note_use(&m,19,8);
    texture_material_note_use(&m,7,9);
    assert(m.last_use_fence==19 && m.last_use_epoch==9);
    assert(texture_material_release(&m,1));
    // Releasing logical references does not claim physical reclamation.
    assert(ledger.resident_bytes==4096 && ledger.retained_bytes==4096 && !m.retired);
    assert(texture_material_retire(&m)==4096);
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
    assert(old.image==1 && old.refs==1 && old.live_refs==1 && old.last_use_fence==70 && old.ready);
    assert(changed.image==2 && texture_material_exclusive(&changed));
    assert(!texture_material_release(&old,1));
    assert(ledger.live_bytes==1024 && ledger.retained_bytes==1024);
    assert(texture_material_cache(&old,0));
    assert(texture_material_retire(&old)==1024);
    assert(ledger.owners==1 && ledger.resident_bytes==1024 && changed.image==2);
    assert(texture_material_release(&changed,1));
    assert(texture_material_retire(&changed)==1024 && !ledger.resident_bytes);
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
        for(unsigned i=0;i<N;i++)if(m[i].ledger && !m[i].retired && texture_material_unreferenced(&m[i]))texture_material_retire(&m[i]);
        uint64_t bytes=0,active=0,retained=0,owners=0;
        for(unsigned i=0;i<N;i++) {
            unsigned refs=0,live_refs=0;
            for(unsigned j=0;j<ASSOCIATIONS;j++)if(owner[j]==(int)i){refs++;live_refs+=(unsigned)live[j];}
            if(!m[i].ledger || m[i].retired){assert(!refs);continue;}
            assert(m[i].refs==refs && m[i].live_refs==live_refs);
            bytes+=m[i].bytes;owners++;
            if(live_refs)active+=m[i].bytes;else retained+=m[i].bytes;
        }
        assert(ledger.owners==owners && ledger.resident_bytes==bytes && ledger.live_bytes==active && ledger.retained_bytes==retained);
    }
    for(unsigned i=0;i<ASSOCIATIONS;i++)if(owner[i]>=0)texture_material_release(&m[owner[i]],live[i]);
    for(unsigned i=0;i<N;i++)if(m[i].ledger && !m[i].retired){texture_material_cache(&m[i],0);texture_material_retire(&m[i]);}
    assert(!ledger.owners && !ledger.resident_bytes && !ledger.live_bytes && !ledger.retained_bytes);
}

int main(void) {
    basic_ownership();separate_versions();churn();
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
enum { GFX_RES_IMG=1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE=2, MAX_DESC=256 };
typedef struct { int slot,exact,flags,type; } GfxShaderRes;
typedef struct { unsigned kind,type,count,stage; const GfxShaderRes *res; } VkPipeBinding;
typedef struct { unsigned n_bindings; VkPipeBinding bindings[4]; } VkPipeInfo;
typedef unsigned VkImage;
typedef struct { unsigned n_storage_writes; struct { VkImage image; uint64_t addr; } storage_writes[MAX_DESC]; } DrawCtx;
typedef struct { uint64_t img[6][32]; DrawCtx *dc; } VkExec;
typedef struct { unsigned skip,image; uint64_t addr; } VkTexImage;
typedef int view_t;
static VkTexImage images[4];
static unsigned locks,unlocks,lookups,exact_calls,writable_calls,bound_calls;
static int depth,deferred,fail_at;
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
static void vk_texture_storage_written(VkImage image,uint64_t addr){
    assert(image>=100 && addr==image-100);bound_calls++;
}
/* production preflight inserted here */
static void reset(void){
    locks=unlocks=lookups=exact_calls=writable_calls=bound_calls=0;
    depth=deferred=0;fail_at=-1;
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
    assert(cx.n_storage_writes==2);storage_images_written(&ex);assert(bound_calls==2 && !cx.n_storage_writes);
    storage_images_written(&ex);assert(bound_calls==2);
    assert(images[1].image==101 && images[2].image==102 && images[3].image==7);
    reset();fail_at=2;assert(!prepare_storage_images(&ex,&info));
    assert(locks==1 && unlocks==1 && !depth && lookups==2 && writable_calls==1 && !bound_calls && !cx.n_storage_writes);
    reset();r.slot=31;info.bindings[0].count=3;ex.img[3][31]=3;
    assert(prepare_storage_images(&ex,&info));assert(lookups==1 && writable_calls==1);
    reset();r.slot=-1;info.bindings[0].count=2;
    assert(prepare_storage_images(&ex,&info));assert(lookups==1 && writable_calls==1);
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
typedef uintptr_t VkImage,VkCommandBuffer;
typedef void *VmaAllocation;
typedef unsigned VkImageType,VkFormat,VkSampleCountFlagBits;
typedef struct { unsigned flags,type,format,w,h,d,levels,layers,samples,usage; } ImgKey;
#include "vk_texture_material.h"
#include "texture_source.h"
enum { VK_NULL_HANDLE=0,VK_IMAGE_TYPE_2D=1,VK_IMAGE_USAGE_TRANSFER_SRC_BIT=1,VK_IMAGE_USAGE_TRANSFER_DST_BIT=2,
NVN_MEMORY_POOL_FLAGS_VIRTUAL=4,VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO=1,VK_IMAGE_TILING_OPTIMAL=1,
VK_IMAGE_LAYOUT_UNDEFINED=0,VK_IMAGE_LAYOUT_GENERAL=1,VK_SUCCESS=0,ALL_RW=7,VK_PIPELINE_STAGE_2_COPY_BIT=8,
VK_ACCESS_2_TRANSFER_READ_BIT=1,VK_ACCESS_2_TRANSFER_WRITE_BIT=2 };
typedef struct { unsigned width,height,depth; } Extent;
typedef struct { unsigned sType,flags,imageType,format; Extent extent; unsigned mipLevels,arrayLayers,samples,tiling,usage,initialLayout; } VkImageCreateInfo;
typedef struct { unsigned aspectMask,mipLevel,baseArrayLayer,layerCount; } VkImageSubresourceLayers;
typedef struct { VkImageSubresourceLayers srcSubresource,dstSubresource; Extent extent; } VkImageCopy;
typedef struct { unsigned vma; } VkCtx;
typedef struct { uint64_t size; } VmaAllocationInfo;
typedef struct { uint64_t addr; unsigned layout; } TexKey;
typedef struct { VkImage image; uint64_t addr; unsigned fb_learned,gone,gpu_owned,bound,mirrored,skip,samples,type,levels,layers,usage,width,height,aspect; } VkTexImage;
typedef struct { unsigned depth,stencil; } TexVkFormat;
typedef struct Tex { TexKey key; VkTexImage pub; TexVkFormat f; ImgKey ik,material_intent;
struct { uint8_t *mem; unsigned pool_flags; struct { uint64_t size; } L; } t;
TextureMaterial *material; VmaAllocation alloc,bound_to; uint64_t bytes,source_copy_size,up_retry,upload_mapping,upload_protection;
uint8_t *source_copy; uint8_t source_digest[32]; TextureSourceStamp source_stamp;
unsigned material_writable,gpu_dirty,sfill,flat3d,superseded,overwritten,retiring,released,uploaded,
source_digest_valid,bc1_try,material_prepared,reusable,reused,content_slot,n_up,chash_ok;
struct Tex *next_at; } Tex;
static TextureMaterialLedger material_ledger;
static uint64_t rel_bytes;static unsigned submit_epoch=1;
static unsigned by_addr;
static Tex *head;
static unsigned generations,retired,purged,copies,allocations,held,hashes,observes;
static VkImage last_retired;static uint64_t reclaim_credit;
static int allocation_failure,invalid_mapping,dirty_after,expect_fatal;
static jmp_buf fatal_jump;
static TextureSourceStamp current_stamp={1,1,1,3};
static uint8_t physical[2048][64];static unsigned next_image=100;
static unsigned mip(unsigned n,unsigned l){return n>>l ? n>>l:1;}
static int nvn_gpu_only(uint64_t addr){(void)addr;return 0;}
static Tex *nvn_hm_get(unsigned *map,const uint64_t *addr){(void)map;return head && head->key.addr==*addr?head:NULL;}
static int content_source_gpu_overlap(const Tex *e){return e->gpu_dirty;}
static void vk_exec_image_gone(VkImage image){assert(image);purged++;}
static void vk_defer_image_reclaim(VkImage image,VmaAllocation alloc,uint64_t credit){
assert(image && alloc);retired++;last_retired=image;reclaim_credit+=credit;}
static void vk_defer_image(VkImage image,VmaAllocation alloc){vk_defer_image_reclaim(image,alloc,0);}
static uint64_t vk_exec_recording_fence(void){return 17;}
static void tex_gen_bump(uint64_t addr){assert(addr);generations++;}
static int source_spans_init(Tex *e){return e->t.L.size!=0;}
static int source_observe(Tex *e,const TextureSourceStamp *prior,TextureSourceObservation *out){
(void)e;observes++;*out=(TextureSourceObservation){.stamp=current_stamp};
if(invalid_mapping && observes>1){out->stamp.mapping++;return TEX_SOURCE_UNKNOWN;}
if(dirty_after && observes>1){out->stamp.generation++;return TEX_SOURCE_DIRTY;}
return texture_source_stamp_valid(prior)?TEX_SOURCE_CLEAN:TEX_SOURCE_UNKNOWN;}
static void source_commit(Tex *e,const TextureSourceStamp *stamp){e->source_stamp=*stamp;}
static int mem_read_hold(uint64_t addr,uint64_t size){assert(addr && size && !held);held=1;return 1;}
static void mem_read_release(void){assert(held);held=0;}
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
static void rt_fatal(const char *fmt,...){(void)fmt;assert(expect_fatal);longjmp(fatal_jump,1);}
static void vmaGetAllocationInfo(unsigned vma,VmaAllocation alloc,VmaAllocationInfo *out){(void)vma;assert(alloc);out->size=64;}
static VkCommandBuffer vk_upload_begin(void){return 1;}
static void vk_upload_end(void){}
static void init_image(Tex *e,VkCommandBuffer cb){assert(cb);memset(physical[e->pub.image],0,64);}
static void barrier(VkCommandBuffer cb,unsigned a,unsigned b,unsigned c){(void)a;(void)b;(void)c;assert(cb);}
static void vkCmdCopyImage(VkCommandBuffer cb,VkImage src,unsigned sl,VkImage dst,unsigned dl,unsigned n,const VkImageCopy *regions){
(void)sl;(void)dl;assert(cb && src!=dst && n==1 && regions[0].extent.width==4);copies++;memcpy(physical[dst],physical[src],64);}
#include "vk_texture_content.h"
static Tex make_tex(uint64_t addr,uint8_t *bytes){
Tex e={0};e.key=(TexKey){addr,9};e.pub=(VkTexImage){.addr=addr,.mirrored=1,.samples=1,.type=1,.levels=1,.layers=1,.usage=3,.width=4,.height=4,.aspect=1};
e.t.mem=bytes;e.t.L.size=64;e.ik=(ImgKey){.type=1,.format=37,.w=4,.h=4,.d=1,.levels=1,.layers=1,.samples=1,.usage=3};e.material_intent=e.ik;return e;}
static void publish_snapshot(Tex *e,VkImage image){
e->pub.image=image;e->alloc=(void *)(uintptr_t)image;e->bytes=64;e->uploaded=1;
assert(e->source_copy && e->source_digest_valid);memcpy(physical[image],e->source_copy,64);
content_image_publish(e,e->source_digest,64);content_source_discard(e);assert(e->material && !e->alloc && !e->bytes);}
int main(void){
uint8_t bytes[64];memset(bytes,3,sizeof bytes);Tex a=make_tex(4096,bytes);
assert(!content_early_prepare(&a));assert(a.source_copy && material_snapshot_bytes==64 && hashes==1 && !held);
uint8_t original[64];memcpy(original,a.source_copy,64);memset(bytes,9,64);
assert(!memcmp(a.source_copy,original,64));publish_snapshot(&a,1);TextureMaterial *old=a.material;
assert(!memcmp(physical[1],original,64) && old->refs==1 && old->cached && material_ledger.resident_bytes==64);
Tex b=make_tex(8192,original);observes=0;assert(content_early_prepare(&b));
assert(b.material==old && old->refs==2 && !b.source_copy && !material_snapshot_bytes && !allocations);
unsigned hash_count=hashes;Tex same=make_tex(8192,original);observes=0;
assert(content_early_prepare(&same));assert(hashes==hash_count && same.material==old && old->refs==3);
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
Tex writable=make_tex(32768,original);writable.material_writable=1;unsigned before_hash=hashes;
assert(!content_early_prepare(&writable) && !writable.source_copy && hashes==before_hash);
unsigned cow_purges=purged;assert(cow_purges==2 && !retired);
allocation_failure=expect_fatal=1;VkImage prior=a.pub.image;
if(!setjmp(fatal_jump)){content_write_begin(&a,1);assert(0);}
assert(a.material==old && a.pub.image==prior && old->refs==1 && !retired && purged==cow_purges);
allocation_failure=expect_fatal=0;
assert(!content_release(&a,0));assert(old->refs==0 && old->cached && !retired);
material_reclaim_cursor=old->index_slot-1;assert(content_reclaim(64)==64);
assert(retired==1 && last_retired==1 && reclaim_credit==64 && !material_ledger.resident_bytes);
Tex stale=make_tex(8192,original);observes=0;assert(!content_early_prepare(&stale));assert(stale.source_copy);content_source_discard(&stale);
Tex bad=make_tex(12288,original);observes=0;invalid_mapping=1;
assert(!content_early_prepare(&bad) && !bad.source_copy && !material_snapshot_bytes && !held);invalid_mapping=0;
Tex race=make_tex(16384,original);observes=0;dirty_after=1;
assert(!content_early_prepare(&race));assert(race.source_copy && race.source_stamp.generation==current_stamp.generation);
content_source_discard(&race);dirty_after=0;
content_shutdown();assert(!material_snapshot_bytes && !rel_bytes);
puts("texture content integration: immutable snapshot, certified reuse, transactional COW and final-owner retirement passed");return 0;
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
typedef struct { int uploaded,last_ch; void *material,*source_copy; unsigned source_count;
TextureSourceStamp source_stamp;TextureSourceSpan *source_spans; } Tex;
static int spans_ok=1,overlap,wait_upload,query_state,turn_due,eligible=1,early_hit,tex_deferred;
static unsigned n_checks,commits,early_calls,uploads;
static int source_spans_init(Tex *e){(void)e;return spans_ok;}
static int content_source_gpu_overlap(Tex *e){(void)e;return overlap;}
static int up_wait(Tex *e){(void)e;return wait_upload;}
static int source_observe(Tex *e,const TextureSourceStamp *prior,TextureSourceObservation *out){
(void)e;(void)prior;*out=(TextureSourceObservation){.stamp={1,1,1,9}};return query_state;}
static void source_commit(Tex *e,const TextureSourceStamp *stamp){commits++;e->source_stamp=*stamp;}
static int vk_exec_turn_due(void){return turn_due;}
static int content_image_eligible(Tex *e){(void)e;return eligible;}
static int content_early_prepare(Tex *e){(void)e;early_calls++;return early_hit;}
static void upload(Tex *e,const void *dirty,int tracked,const TextureSourceObservation *observed){
(void)e;(void)dirty;(void)tracked;assert(observed->stamp.generation==9);uploads++;}
/* production freshness function */
static void guard_tests(void){
Tex e={.uploaded=1};turn_due=1;query_state=TEX_SOURCE_CLEAN;
assert(make_fresh_source(&e) && commits==1 && !tex_deferred && !early_calls && !uploads);
query_state=TEX_SOURCE_DIRTY;assert(make_fresh_source(&e));assert(tex_deferred && !early_calls && !uploads);
tex_deferred=0;eligible=0;e.material=&e;assert(make_fresh_source(&e));assert(tex_deferred && !early_calls && !uploads);
tex_deferred=0;e.material=NULL;assert(make_fresh_source(&e));assert(!tex_deferred && early_calls==1 && uploads==1);
turn_due=0;eligible=1;early_hit=1;assert(make_fresh_source(&e));assert(early_calls==2 && uploads==1);
early_hit=0;query_state=TEX_SOURCE_UNKNOWN;assert(make_fresh_source(&e));assert(uploads==2 && n_checks==6);
}
static void packing_test(unsigned mask,int gdsw,int snapshot){
const uint64_t layer_bytes=UINT64_C(2000)*3200*2;
TextureSourceSpan spans[8];Region rg[8];AstcRegion ar[8];
for(unsigned i=0;i<8;i++){spans[i]=(TextureSourceSpan){i*layer_bytes,layer_bytes};
rg[i]=(Region){.bufferOffset=i*layer_bytes,.imageSubresource={0,i}};
ar[i]=(AstcRegion){.src=i*layer_bytes,.level=0,.layer=i};}
Tex value={.uploaded=1,.source_count=8,.source_spans=spans};Tex *e=&value;
TextureSourceObservation observation={.stamp={4,5,6,7},.dirty={mask,0,0,0}};
const TextureSourceObservation *observed=&observation;
unsigned all_regions=8,n=8,k;int check=0,bc1=0,acheck=0,sk=0,partial=0;
uint64_t total=layer_bytes*8,src_bytes=total,prior_map=0,prior_protection=0;
/* production selective packing */
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
int main(void){guard_tests();for(unsigned mask=0;mask<256;mask++){
packing_test(mask,0,0);packing_test(mask,1,0);packing_test(mask,0,1);}
puts("texture source integration: worker-turn deferral and all sparse masks of eight-layer 97.7 MiB upload passed");return 0;}
'''


def source_integration_fixture():
    source = (ROOT / 'runtime/vk_texture.c').read_text()
    start = source.index('static int make_fresh_source(')
    end = source.index('\nstatic void make_fresh(', start)
    function = source[start:end]
    start = source.index('    int select = e->uploaded && !snapshot')
    end = source.index('    if (!partial && can_retain', start)
    packing = source[start:end]
    return SOURCE_INTEGRATION_FIXTURE.replace('/* production freshness function */', function).replace(
        '/* production selective packing */', packing)


def storage_fixture():
    source = (ROOT / 'runtime/vk_exec_draw.c').read_text()
    start = source.index('static int prepare_storage_images(')
    end = source.index('\n// Writes the pipeline', start)
    body = source[start:end]
    # The preflight must execute before any retained descriptor packet can be accepted.
    bind = source[source.index('static void bind_resources('):]
    preflight = bind.index('prepare_storage_images(ex, info)')
    for consumption in ('binding_recipe(ex, info)', 'prepared_instance(ex, recipe)', 'bind_sig(ex, info)', 'pb_entry(cx, info, g)'):
        assert preflight < bind.index(consumption), consumption
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
    compile_and_run('ownership', OWNER_FIXTURE)
    compile_and_run('storage_preflight', storage_fixture())
    compile_and_run('content_transactions', CONTENT_FIXTURE)
    compile_and_run('source_integration', source_integration_fixture())

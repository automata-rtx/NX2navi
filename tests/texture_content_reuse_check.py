"""Exercise the production materialized-texture donor index and recorded copy contract without a GPU."""
from pathlib import Path
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'build/test/texture-content-reuse'
CC=Path(os.environ.get('CC',r'C:\msys64\ucrt64\bin\clang.exe'))
if not CC.is_file():CC=Path(r'C:\Users\Andrew\AppData\Local\Programs\2NX2Navi\toolchain\msys64\ucrt64\bin\clang.exe')
sys.path.insert(0,str(ROOT/'tools'))
from check_content import GENERATED_MARKER

FIXTURE=r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef uintptr_t VkImage;
typedef uintptr_t VkCommandBuffer;
typedef struct { uint32_t aspectMask,mipLevel,baseArrayLayer,layerCount; } VkImageSubresourceLayers;
typedef struct { uint32_t width,height,depth; } VkExtent3D;
typedef struct { int32_t x,y,z; } VkOffset3D;
typedef struct { VkImageSubresourceLayers srcSubresource; VkOffset3D srcOffset;
                 VkImageSubresourceLayers dstSubresource; VkOffset3D dstOffset; VkExtent3D extent; } VkImageCopy;
enum { VK_IMAGE_TYPE_2D=2, VK_IMAGE_USAGE_TRANSFER_SRC_BIT=1, VK_IMAGE_USAGE_TRANSFER_DST_BIT=2,
       VK_IMAGE_LAYOUT_GENERAL=7, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT=1, VK_PIPELINE_STAGE_2_COPY_BIT=2,
       VK_ACCESS_2_MEMORY_READ_BIT=4,VK_ACCESS_2_MEMORY_WRITE_BIT=8,VK_ACCESS_2_TRANSFER_READ_BIT=16,
       VK_ACCESS_2_TRANSFER_WRITE_BIT=32 };
#define ALL_RW VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT
typedef struct { uint64_t addr; uint32_t format,width,height,swizzle,levels,layers; uint8_t has_layout,layout[8],pad[7]; } TexKey;
typedef struct { uint32_t flags,type,format,w,h,d,levels,layers,samples,usage; } ImgKey;
typedef struct Tex {
    struct { VkImage image; uint32_t mirrored,gpu_owned,gone,skip,samples,type,levels,layers,usage,aspect,width,height; } pub;
    struct { uint32_t astc_bw,depth,stencil; } f;
    TexKey key; ImgKey ik;
    uintptr_t alloc,bound_to,flat3d;
    uint32_t gpu_dirty,sfill,superseded,overwritten,retiring,uploaded,up_retry,content_slot,touched;
    struct Tex *next_at;
} Tex;
static int by_addr;
static Tex *registered[20000];static unsigned registered_count;
static void *nvn_hm_get(const int *map,const uint64_t *addr) {
    assert(map==&by_addr);for(unsigned i=0;i<registered_count;i++)if(registered[i]&&registered[i]->key.addr==*addr)return registered[i];return NULL;
}
static uint64_t clock_ns;
static uint64_t kern_now_ns(void) { return ++clock_ns; }
static uint32_t mip(uint32_t n,uint32_t level) { n>>=level;return n?n:1; }
static void touch(Tex *e) { e->touched++; }
static unsigned step,copies,barriers,copy_count;
static VkImage copied_from,copied_to;
static VkImageCopy copied_regions[32];
static void barrier(VkCommandBuffer cb,uint64_t source,uint64_t src_access,uint64_t dest,uint64_t dst_access) {
    assert(cb==123);barriers++;
    if(step==0){assert(source==VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);assert(src_access==(VK_ACCESS_2_MEMORY_READ_BIT|VK_ACCESS_2_MEMORY_WRITE_BIT));
        assert(dest==VK_PIPELINE_STAGE_2_COPY_BIT&&dst_access==(VK_ACCESS_2_TRANSFER_READ_BIT|VK_ACCESS_2_TRANSFER_WRITE_BIT));}
    else {assert(step==2&&source==VK_PIPELINE_STAGE_2_COPY_BIT&&src_access==VK_ACCESS_2_TRANSFER_WRITE_BIT);
        assert(dest==VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT&&dst_access==(VK_ACCESS_2_MEMORY_READ_BIT|VK_ACCESS_2_MEMORY_WRITE_BIT));}
    step++;
}
static void vkCmdCopyImage(VkCommandBuffer cb,VkImage source,int sl,VkImage dest,int dl,uint32_t count,const VkImageCopy *regions) {
    assert(cb==123&&step==1&&sl==VK_IMAGE_LAYOUT_GENERAL&&dl==VK_IMAGE_LAYOUT_GENERAL&&source!=dest&&count<=32);
    copied_from=source;copied_to=dest;copy_count=count;memcpy(copied_regions,regions,count*sizeof *regions);copies++;step++;
}
#include "vk_texture_content.h"
static void setup(Tex *e,uint64_t addr,VkImage image) {
    *e=(Tex){0};e->key=(TexKey){.addr=addr,.format=42,.width=65,.height=33,.swizzle=123,.levels=4,.layers=3};
    e->ik=(ImgKey){.type=VK_IMAGE_TYPE_2D,.format=77,.w=65,.h=33,.d=1,.levels=4,.layers=3,.samples=1,.usage=3};
    e->pub.image=image;e->pub.mirrored=1;e->pub.samples=1;e->pub.type=VK_IMAGE_TYPE_2D;e->pub.levels=4;e->pub.layers=3;
    e->pub.usage=3;e->pub.aspect=1;e->pub.width=65;e->pub.height=33;e->f.astc_bw=4;e->alloc=1;e->uploaded=1;
    assert(registered_count<20000);registered[registered_count++]=e;
}
static void reset(void) {
    memset(material_slots,0,sizeof material_slots);memset(&material_stats,0,sizeof material_stats);
    memset(registered,0,sizeof registered);registered_count=0;step=copies=barriers=copy_count=0;
}
static void blocked(Tex *dst,Tex *src,const uint8_t *digest) {
    assert(!content_image_eligible(src));assert(!content_image_find(dst,digest,4096));
    unsigned before=copies;assert(!content_image_copy(dst,src,123,4096));assert(copies==before);
}
int main(void) {
    Tex a,b,c;uint8_t digest[32]={1,2,3,4},other[32]={1,2,3,4};other[31]=99;
    reset();setup(&a,0x10000,1);setup(&b,0x20000,2);setup(&c,0x30000,3);
    assert(content_image_eligible(&a)&&content_image_eligible(&b));
    assert(!content_image_find(&b,digest,4096));
    content_image_publish(&a,digest,4096);assert(a.content_slot&&content_image_find(&b,digest,4096)==&a);
    assert(!content_image_find(&a,digest,4096));assert(!content_image_find(&b,digest,4095));
    assert(!content_image_find(&b,other,4096)); // same direct-map bucket, different full digest
    // The guest address is deliberately excluded; descriptor/layout and actual VkImage creation parameters are not.
    b.key.swizzle++;assert(!content_image_find(&b,digest,4096));b.key.swizzle--;
    b.key.layout[3]=1;assert(!content_image_find(&b,digest,4096));b.key.layout[3]=0;
    b.key.has_layout=1;assert(!content_image_find(&b,digest,4096));b.key.has_layout=0;
    b.ik.format++;assert(!content_image_find(&b,digest,4096));b.ik.format--;
    b.ik.layers++;assert(!content_image_find(&b,digest,4096));b.ik.layers--;
    a.ik.format++;assert(!content_image_find(&b,digest,4096));a.ik.format--;
    a.uploaded=0;assert(!content_image_find(&b,digest,4096));a.uploaded=1;
    a.up_retry=1;assert(!content_image_find(&b,digest,4096));a.up_retry=0;
    // Conservative exclusions are rechecked on lookup and before recording, even if invalidation was delayed.
#define BLOCK(field,value) do { __typeof__(a.field) saved=a.field;a.field=(value);blocked(&b,&a,digest);a.field=saved; } while(0)
    BLOCK(pub.image,0);BLOCK(alloc,0);BLOCK(pub.mirrored,0);BLOCK(pub.gpu_owned,1);BLOCK(gpu_dirty,1);
    BLOCK(bound_to,1);BLOCK(sfill,1);BLOCK(flat3d,1);BLOCK(superseded,1);BLOCK(overwritten,1);BLOCK(retiring,1);
    BLOCK(pub.gone,1);BLOCK(pub.skip,1);BLOCK(f.astc_bw,0);BLOCK(f.depth,1);BLOCK(f.stencil,1);
    BLOCK(pub.samples,2);BLOCK(pub.type,3);BLOCK(pub.levels,0);BLOCK(pub.levels,33);BLOCK(pub.layers,0);
    BLOCK(pub.usage,VK_IMAGE_USAGE_TRANSFER_SRC_BIT);BLOCK(pub.usage,VK_IMAGE_USAGE_TRANSFER_DST_BIT);BLOCK(next_at,&c);
#undef BLOCK
    registered[0]=NULL;assert(!content_image_find(&b,digest,4096));registered[0]=&a;
    VkImage old=b.pub.image;b.pub.image=a.pub.image;assert(!content_image_find(&b,digest,4096));b.pub.image=old;
    // Mutation removes the old publication before a later lookup can use it.
    content_image_forget(&a);assert(!a.content_slot&&!content_image_find(&b,digest,4096));
    content_image_publish(&a,other,4096);assert(!content_image_find(&b,digest,4096));
    // A collision transfers the nonowning slot; forgetting its former owner cannot clear the new owner.
    content_image_publish(&c,digest,4096);assert(!a.content_slot&&c.content_slot&&material_stats.collisions==1);
    content_image_forget(&a);assert(content_image_find(&b,digest,4096)==&c);
    // Retired/freed pointer is removed while still valid. No cache-owned lifetime survives this step.
    unsigned slot=c.content_slot;content_image_forget(&c);assert(!material_slots[slot-1].owner);
    memset(&c,0xDD,sizeof c);registered[2]=NULL;assert(!content_image_find(&b,digest,4096));
    content_image_publish(&a,digest,4096);
    // Copy records every mip/layer and both barriers; source/destination keep distinct image ownership.
    assert(content_image_copy(&b,&a,123,4096));assert(step==3&&barriers==2&&copies==1&&copy_count==4);
    assert(copied_from==a.pub.image&&copied_to==b.pub.image&&a.touched==1);
    uint64_t texels=0;
    for(unsigned level=0;level<4;level++) {
        VkImageCopy *r=&copied_regions[level];assert(r->srcSubresource.mipLevel==level&&r->dstSubresource.mipLevel==level);
        assert(r->srcSubresource.layerCount==3&&r->dstSubresource.layerCount==3);
        assert(!r->srcSubresource.baseArrayLayer&&!r->dstSubresource.baseArrayLayer&&r->srcSubresource.aspectMask==1);
        assert(!r->srcOffset.x&&!r->srcOffset.y&&!r->srcOffset.z&&!r->dstOffset.x&&!r->dstOffset.y&&!r->dstOffset.z);
        assert(r->extent.width==mip(65,level)&&r->extent.height==mip(33,level)&&r->extent.depth==1);
        texels+=(uint64_t)r->extent.width*r->extent.height*3;
    }
    assert(material_stats.hits==1&&material_stats.copied_bytes==4096&&material_stats.avoided_texels==texels);
    assert(material_stats.copy_ns>0&&!content_image_copy(&b,&a,0,4096));
    b.gpu_dirty=1;assert(!content_image_copy(&b,&a,123,4096));b.gpu_dirty=0;
    // Thousands of publications stay bounded by the fixed index and never retain previous-owner links.
    reset();Tex *many=calloc(10000,sizeof *many);assert(many);
    for(unsigned i=0;i<10000;i++) { setup(&many[i],0x10000ull+i*4096,i+1);uint8_t d[32]={0};memcpy(d,&i,sizeof i);content_image_publish(&many[i],d,4096); }
    unsigned owners=0;
    for(unsigned i=0;i<MATERIAL_SLOTS;i++)if(material_slots[i].owner){owners++;assert(material_slots[i].owner->content_slot==i+1);}
    assert(owners==MATERIAL_SLOTS);
    for(unsigned i=0;i<10000;i++)content_image_forget(&many[i]);
    for(unsigned i=0;i<MATERIAL_SLOTS;i++)assert(!material_slots[i].owner);
    free(many);puts("texture content reuse: identity, exclusions, bounded donors, invalidation and recorded copies passed");return 0;
}
'''

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    marker=GENERATED_MARKER.decode() if isinstance(GENERATED_MARKER,bytes) else GENERATED_MARKER
    src=OUT/'content.c';src.write_text('// '+marker+'\n'+FIXTURE)
    exe=OUT/'content.exe'
    env=dict(os.environ,PATH=str(CC.parent)+os.pathsep+os.environ.get('PATH',''))
    subprocess.run([str(CC),'-std=gnu11','-O1','-Wall','-Wextra','-Werror','-fsanitize=undefined','-fsanitize-trap=all',
                    '-I',str(ROOT/'runtime'),str(src),'-o',str(exe)],env=env,check=True)
    subprocess.run([str(exe)],env=env,check=True)

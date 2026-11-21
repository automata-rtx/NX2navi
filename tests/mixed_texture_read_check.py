"""Exercise the texture consumer of an exact mixed native image and buffer snapshot."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

from bind_recipe_check import function

PREFIX = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "resource_core_admission.h"

#define VK_TEXTURE_ALIAS_MAX_BYTES (64u * 1024u * 1024u)
#define VK_TEXTURE_ALIAS_MAX_PLANES 4
#define VK_IMAGE_LAYOUT_GENERAL 1
#define VK_PIPELINE_STAGE_2_COPY_BIT 2
#define VK_PIPELINE_STAGE_2_HOST_BIT 4
#define VK_ACCESS_2_TRANSFER_READ_BIT 8
#define VK_ACCESS_2_TRANSFER_WRITE_BIT 16
#define VK_ACCESS_2_HOST_READ_BIT 32
#define ALL_RW 64,128

typedef uintptr_t VkImage, VkBuffer, VkCommandBuffer;
typedef uint64_t VkDeviceSize;
enum { VK_SUCCESS, VK_NOT_READY, RC_BUDGET_TEXTURE };
typedef struct {VkDeviceSize srcOffset,dstOffset,size;} VkBufferCopy;
typedef struct {VkDeviceSize bufferOffset;} VkBufferImageCopy;
typedef struct {uint64_t scene,ordinal;} RcRegistryPosition;
typedef struct {uint64_t backing,offset,size;} RcSourceRange;
typedef struct {uint64_t address,mapping; RcSourceRange range;} RcInterval;
typedef struct {uint64_t id;uint32_t index;} RcRegistryWrite;
typedef struct {uint64_t image;RcRegistryWrite hold;} RcRegistryImageBinding;
typedef struct {RcInterval interval;RcRegistryImageBinding binding;RcRegistryPosition output;uint32_t demand;}
    RcRegistryNativeImageFragment;
typedef struct {uint64_t allocation,incarnation,offset;} RcRegistryBufferLocation;
typedef struct {RcInterval interval;RcRegistryBufferLocation native;} RcRegistryBufferSpan;
typedef struct {
    RcRegistryNativeImageFragment fragment;
    RcRegistryBufferLocation native_buffer;
    uint32_t baseline;
} RcRegistryNativeImageSnapshotFragment;
typedef enum {
    RC_REGISTRY_OK, RC_REGISTRY_UNSUPPORTED, RC_REGISTRY_STALE, RC_REGISTRY_INVALID,
    RC_REGISTRY_CAPACITY, RC_REGISTRY_BLOCKED
} RcRegistryResult;
typedef struct {
    RcInterval interval;
    RcRegistryImageBinding selected;
    RcRegistryPosition selected_output;
    uint32_t demand;
} RcRegistryNativeImageFailure;
typedef struct {RcInterval receiver;RcRegistryNativeImageFailure coverage;} VkTextureAliasFailure;
typedef struct {VkImage image;uint64_t addr;} VkTexImage;
typedef struct {VkBuffer buffer;} VkMemSpan;
typedef struct {unsigned unused;} nvn_texdesc;
typedef struct {unsigned value;} TexVkFormat;
typedef struct {nvn_texdesc d;uint64_t serial; struct {size_t size;} L;} tex_t;
typedef struct {struct {VkBufferImageCopy copy;} planes[VK_TEXTURE_ALIAS_MAX_PLANES];uint32_t count;uint64_t bytes;}
    VkTextureAliasLayout;
typedef struct {int live, uses;} VkMemoryImageHold;
typedef struct {RcAdmissionToken token;} RcBudgetTicket;
typedef struct {int live; RcRegistryBufferSpan span;} VkMemImageAliasBaseline;
typedef struct {
    VkTexImage pub;
    tex_t t;
    TexVkFormat f;
    uint64_t content_revision,sample_general_revision;
    int pending_write,pin_detached;
} Tex;
typedef struct {
    const Tex *entry;
    VkTexImage image;
    nvn_texdesc descriptor;
    TexVkFormat format;
    uint64_t guest_bytes;
    void *allocation;
    RcInterval *mapping;
    uint32_t mapping_count;
    RcRegistryWrite *holds;
} NativeTextureOutput;
typedef struct {
    const uint8_t *guest;
    uint64_t address,bytes;
} VkTextureAliasBytes;
typedef struct {
    RcSourceRange covered;
    uint64_t writer_address;
    uint32_t source,demand;
} VkTextureAliasSlice;
'''

STUBS = r'''
typedef struct VkTextureAliasRead {
    const tex_t *receiver;
    uint64_t key, receiver_logical;
    RcRegistryPosition read;
    VkTexImage result;
    int registration_only, transfer_source, raw_output, cpu_handed, sparse_buffer_rows;
    VkMemSpan raw_source;
    uint64_t predecessor_serial;
    VkImage predecessor;
    VkMemoryImageHold *target_hold, *predecessor_hold;
    uint8_t storage_seed;
    RcInterval *receiver_mapping, *demands;
    RcRegistryNativeImageSnapshotFragment *fragments, *current;
    VkTextureAliasSlice *slices;
    VkMemImageAliasBaseline *baseline;
    uint64_t raw_upload;
    VkTextureAliasLayout receiver_layout;
    ImageAliasBufferCopy *buffer_copies;
    uint32_t mapping_count,demand_count,fragment_count,source_count,buffer_count,owners;
    uint64_t stage_bytes,upload_offset,workspace_bytes,guest_bytes;
#if SCALABLE_SOURCES
    ImageAliasSource *sources;
    VkTextureAliasBytes *source_bytes;
    RcBudgetTicket source_ticket;
#else
    ImageAliasSource sources[IMAGE_ALIAS_MAX_SOURCES];
#endif
    struct {VkBuffer buffer;uint8_t *map;} staging;
    uint8_t *workspace;
} VkTextureAliasRead;

static int by_key, alias_fragment_owners, bound_alias_owners;
static int locked, registry, stale_provider, diagnostics, offset_drift, mapping_drift;
static RcRegistryPosition position={8,10};
static uint64_t pending_fill;
static int fill_complete;
static unsigned image_copies, buffer_copies, barriers, buffer_borrows, baseline_releases, markers;
static const Tex donor={.pub={17,0x2200},.t={.L={4}}};
static NativeTextureOutput projection={.entry=&donor};
static VkMemoryImageHold image_owner={.live=1};
enum { MANY_DONORS = 9 };
static Tex more_donors[MANY_DONORS];
static NativeTextureOutput more_projections[MANY_DONORS];
static VkMemoryImageHold more_holds[MANY_DONORS];
static RcAdmission budget;
static RcAdmissionRecord budget_records[4];
static VkTextureAliasRead *recorded;
static void (*completion)(void *);
static uint8_t stage_bytes[256];
static const unsigned char image_bytes[4]={55,56,57,58};
static const unsigned char raw_bytes[12]={11,12,13,99,99,99,99,21,22,23,24,25};

static int rc_range_valid(RcSourceRange r){return r.backing&&r.size&&r.offset<=UINT64_MAX-r.size;}
static int rc_range_equal(RcSourceRange a,RcSourceRange b){return a.backing==b.backing&&a.offset==b.offset&&a.size==b.size;}
static int rc_registry_position_compare(RcRegistryPosition a,RcRegistryPosition b){return a.scene!=b.scene||a.ordinal!=b.ordinal;}
static int *rc_registry_runtime(void){return &registry;}
static void lock(void){locked++;}
static void unlock(void){assert(locked-- > 0);}
static RcRegistryPosition vk_exec_resource_position(void){return position;}
static const void *nvn_hm_get(const int *map,const void *key){
    if(map==&alias_fragment_owners){
        const VkImage image=*(const VkImage *)key;
        if(image==donor.pub.image)return &projection;
        if(image>=200&&image<200+MANY_DONORS)return &more_projections[image-200];
        return NULL;
    }
    assert(map==&by_key||map==&bound_alias_owners);
    return NULL;
}
static int image_alias_bound_source_current(const Tex *e,VkImage image){(void)e;(void)image;return 0;}
static int image_alias_source_current(const Tex *e){return e && (e==&donor || (e->pub.image>=200&&e->pub.image<200+MANY_DONORS));}
static int image_alias_layout_for(const VkTexImage *image,const nvn_texdesc *descriptor,const TexVkFormat *format,
                                 uint64_t guest_bytes,VkTextureAliasLayout *layout){
    assert(image->image && descriptor && format && guest_bytes==4);
    *layout=(VkTextureAliasLayout){.count=1,.bytes=4};return 1;
}
static VkMemoryImageHold *vk_mem_image_hold(VkImage image){
    VkMemoryImageHold *owner=image==donor.pub.image?&image_owner:&more_holds[image-200];
    if(!owner->live)return NULL;
    owner->uses++;
    return owner;
}
static int vk_mem_image_hold_retired(const VkMemoryImageHold *hold){return !hold->live;}
static int image_alias_sample_receiver_locked(int logical,const VkTextureAliasRead *alias){
    return logical==19&&alias->receiver_logical==19;
}
static int image_alias_guest_mapping_current(uint64_t address,uint64_t bytes,const RcInterval *mapping,uint32_t count){
    return address==0x1100&&bytes==12&&mapping&&count==1&&mapping->mapping==8&&!mapping_drift;
}
static uint32_t vk_mem_device_intervals(const VkMemSpan *span,uint64_t address,uint64_t bytes,RcInterval **out){
    assert(locked&&span->buffer==31&&address==0x1100&&bytes==12);
    *out=malloc(sizeof **out);assert(*out);
    **out=(RcInterval){address,8+mapping_drift,{9,80,bytes}};
    return 1;
}
static int image_alias_predecessor_matches(const VkTextureAliasRead *alias,const Tex *e){
    (void)alias;(void)e;assert(0);return 0;
}
static int storage_seed_receiver_current(const VkTextureAliasRead *alias,const Tex *e){
    (void)alias;(void)e;assert(0);return 0;
}
static int vk_mem_image_alias_baseline_current(const VkMemImageAliasBaseline *owner){
    return owner&&owner->live&&(!pending_fill||fill_complete);
}
static const RcRegistryBufferSpan *vk_mem_image_alias_baseline_submitted_spans(
    const VkMemImageAliasBaseline *owner,uint64_t upload,uint32_t *count){
    assert(upload&&upload==pending_fill);
    *count=owner&&owner->live;
    return *count ? &owner->span : NULL;
}
static const RcRegistryBufferSpan *vk_mem_image_alias_baseline_spans(const VkMemImageAliasBaseline *owner,uint32_t *count){
    assert(owner&&owner->live);*count=1;return &owner->span;
}
static int vk_mem_image_alias_baseline_source(const VkMemImageAliasBaseline *owner,uint32_t baseline,
    RcRegistryBufferLocation native,uint64_t size,VkBuffer *buffer,VkDeviceSize *offset){
    assert(owner);buffer_borrows++;
    if(!vk_mem_image_alias_baseline_current(owner)||baseline||native.allocation!=17||native.incarnation!=0||
        native.offset<0x41||size>12||native.offset-0x41>12-size)return 0;
    *buffer=31;*offset=native.offset+offset_drift;return 1;
}
static void vk_mem_image_alias_baseline_release(VkMemImageAliasBaseline **baseline){
    if(*baseline){assert((*baseline)->live);(*baseline)->live=0;baseline_releases++;}
    *baseline=NULL;
}
static const RcRegistryNativeImageSnapshotFragment snapshot[]={
    {.fragment={{0x4400,12,{9,80,3}}, {0}, {0},0}, .native_buffer={17,0,0x41}, .baseline=0},
    {.fragment={{0x2200,11,{9,83,4}}, {17,{7,2}}, {8,5},0}, .baseline=UINT32_MAX},
    {.fragment={{0x4407,12,{9,87,5}}, {0}, {8,7},0}, .native_buffer={17,0,0x48}, .baseline=0}
};
static RcRegistryResult rc_registry_native_image_snapshot_mixed(int *r,const RcInterval *demands,uint32_t count,
    const RcRegistryBufferSpan *baselines,uint32_t baseline_count,RcRegistryPosition read,
    RcRegistryNativeImageSnapshotFragment *out,uint32_t capacity,uint32_t *received,RcRegistryNativeImageFailure *failure){
    assert(locked&&r==&registry&&demands&&count==1&&capacity>=3&&read.scene==8&&read.ordinal==10&&!failure);
    if(!baselines||!baseline_count)return RC_REGISTRY_BLOCKED;
    assert(baselines[0].native.allocation==17);
    memcpy(out,snapshot,sizeof snapshot);*received=3;
    if(stale_provider)out[0].native_buffer.incarnation++;
    return RC_REGISTRY_OK;
}
#define rc_registry_native_image_snapshot_handed(...) (abort(),RC_REGISTRY_INVALID)
static const char *cfg_get(const char *key){assert(!strcmp(key,"NX2NAVI_PROBE_CURRENT_IMAGE_ALIAS"));return diagnostics?"24":NULL;}
static void rt_log(const char *message,...){assert(strstr(message,"alias raw fragment"));markers++;}
static void image_alias_release(void *arg){
    VkTextureAliasRead *alias=arg;assert(alias==recorded&&alias->owners);
    if(!--alias->owners)vk_mem_image_alias_baseline_release(&alias->baseline);
}
static void image_alias_cancel(void *arg,int may_have_executed){(void)may_have_executed;image_alias_release(arg);}
static void vk_completion_add_owned(uint64_t fence,void (*done)(void *),void (*cancel)(void *,int),void *context){
    assert(fence==8&&!completion&&done&&cancel);completion=done;recorded=context;
}
static void barrier(VkCommandBuffer cb,uint64_t source,uint64_t source_access,uint64_t dest,uint64_t access){
    assert(cb==77&&source&&source_access&&dest&&access);barriers++;
}
static void vkCmdCopyImageToBuffer(VkCommandBuffer cb,VkImage image,int layout,VkBuffer dst,uint32_t count,const VkBufferImageCopy *reg){
    assert(cb==77&&image==17&&layout==VK_IMAGE_LAYOUT_GENERAL&&dst==32&&count==1&&reg->bufferOffset==0);
    memcpy(stage_bytes+reg->bufferOffset,image_bytes,sizeof image_bytes);image_copies++;
}
static void vkCmdCopyBuffer(VkCommandBuffer cb,VkBuffer src,VkBuffer dst,uint32_t count,const VkBufferCopy *reg){
    assert(cb==77&&src==31&&dst==32&&count==1);
    assert((reg->srcOffset==0x41&&reg->dstOffset==4&&reg->size==3)||
           (reg->srcOffset==0x48&&reg->dstOffset==7&&reg->size==5));
    assert(recorded&&recorded->baseline&&recorded->baseline->live&&recorded->owners==2);
    memcpy(stage_bytes+reg->dstOffset,raw_bytes+(reg->srcOffset-0x41),reg->size);buffer_copies++;
}
static int texture_alias_project(const nvn_texdesc *desc,const VkTextureAliasLayout *layout,
    uint8_t *guest,uint8_t *linear,int forward){
    assert(desc&&layout->bytes==4&&!forward);memcpy(guest,linear,layout->bytes);return 1;
}
#if SCALABLE_SOURCES
static int vk_replay_metadata_allocate(size_t bytes,int clear,int producer,RcBudgetTicket *ticket,void **out){
    assert(producer==RC_BUDGET_TEXTURE && bytes && !ticket->token.ledger);
    const RcAdmissionVector demand={{bytes,bytes}};
    if(rc_admission_reserve(&budget,producer,&demand,&ticket->token)!=RC_ADMISSION_OK)return VK_NOT_READY;
    *out=clear?calloc(1,bytes):malloc(bytes);assert(*out);
    RcAdmissionToken retained;
    assert(rc_admission_materialize(&budget,ticket->token,producer,&demand,&retained)==RC_ADMISSION_OK);
    ticket->token=retained;
    return VK_SUCCESS;
}
static void vk_replay_metadata_free(void *memory,RcBudgetTicket *ticket){
    RcAdmissionToken pending;
    assert(rc_admission_retire(&budget,ticket->token,RC_BUDGET_TEXTURE,&pending)==RC_ADMISSION_OK);
    free(memory);
    assert(rc_admission_destroyed(&budget,pending,RC_BUDGET_TEXTURE)==RC_ADMISSION_OK);
    *ticket=(RcBudgetTicket){0};
}
static void vk_mem_image_release(VkMemoryImageHold *owner){if(owner){assert(owner->uses);owner->uses--;}}
#endif
'''

MAIN = r'''
static void release_source_plan(VkTextureAliasRead *alias) {
#if SCALABLE_SOURCES
    image_alias_sources_release(alias);
    alias->sources=NULL;
    alias->source_bytes=NULL;
#else
    free(alias->buffer_copies);
#endif
    alias->buffer_copies=NULL;
    alias->source_count=alias->buffer_count=0;
    alias->stage_bytes=0;
    alias->workspace_bytes=alias->guest_bytes;
}

static void check_many_sources(void) {
    enum { FRAGMENTS=MANY_DONORS+1, BYTES=FRAGMENTS*4 };
    RcInterval demand={0x4000,8,{9,800,BYTES}};
    RcRegistryNativeImageSnapshotFragment fragments[FRAGMENTS]={0};
    VkTextureAliasSlice slices[FRAGMENTS];
    uint8_t workspace[BYTES+MANY_DONORS*4]={0}, expected[BYTES]={0};
    VkMemImageAliasBaseline owner={.live=1,.span={{0x4400,12,{9,80,12}},{17,0,0x41}}};
    VkTextureAliasRead alias={.result={0,demand.address},.read={8,10},.demands=&demand,.demand_count=1,
        .fragments=fragments,.fragment_count=FRAGMENTS,.slices=slices,.baseline=&owner,
        .guest_bytes=BYTES,.workspace_bytes=BYTES,.workspace=workspace,.staging={32,stage_bytes}};
    for(unsigned i=0;i<FRAGMENTS;i++){
        const RcSourceRange part={9,800+i*4,4};
        if(i==4){
            fragments[i]=(RcRegistryNativeImageSnapshotFragment){.fragment={{0x4400,12,part},{0},{8,5},0},
                .native_buffer={17,0,0x41},.baseline=0};
            memcpy(expected+i*4,raw_bytes,4);
        }else{
            const unsigned donor_slot=MANY_DONORS-1-(i-(i>4));
            Tex *owner_entry=&more_donors[donor_slot];
            *owner_entry=(Tex){.pub={200+donor_slot,0x5000+donor_slot*0x100},.t={.L={4}}};
            more_projections[donor_slot]=(NativeTextureOutput){.entry=owner_entry};
            more_holds[donor_slot].live=1;
            fragments[i]=(RcRegistryNativeImageSnapshotFragment){
                .fragment={{owner_entry->pub.addr,11,part},{owner_entry->pub.image,{20+donor_slot,0}},{8,5},0},
                .baseline=UINT32_MAX};
            memset(expected+i*4,30+donor_slot,4);
        }
    }
#if SCALABLE_SOURCES
    const uint64_t metadata=MANY_DONORS*sizeof(ImageAliasSource)+sizeof(ImageAliasBufferCopy)+
                            FRAGMENTS*sizeof(VkTextureAliasBytes);
    const uint64_t peak=metadata+FRAGMENTS*sizeof(uint64_t);
    assert(rc_admission_set_limit(&budget,0,peak-1)==RC_ADMISSION_OK);
    assert(image_alias_capture_sources(&alias,1,NULL)==RC_REGISTRY_CAPACITY);
    assert(!alias.sources&&!rc_admission_used_at(&budget,0)&&!rc_admission_used_at(&budget,1));
    assert(rc_admission_set_limit(&budget,0,peak)==RC_ADMISSION_OK);
    more_holds[0].live=0; // Preserve earlier donor holds until this attempted owner is canceled.
    assert(image_alias_capture_sources(&alias,1,NULL)==RC_REGISTRY_STALE);
    assert(more_holds[MANY_DONORS-1].uses==1);
    release_source_plan(&alias);
    for(unsigned i=0;i<MANY_DONORS;i++)assert(more_holds[i].uses==0);
    assert(!rc_admission_used_at(&budget,0)&&!rc_admission_used_at(&budget,1));
    more_holds[0].live=1;
#endif
    assert(image_alias_capture_sources(&alias,1,NULL)==RC_REGISTRY_OK);
    assert(alias.source_count==MANY_DONORS&&alias.buffer_count==1);
    alias.upload_offset=(alias.stage_bytes+15)&~UINT64_C(15);
    assert(alias.upload_offset<sizeof stage_bytes);
    for(unsigned i=0;i<MANY_DONORS;i++){
        const ImageAliasSource *source=&alias.sources[i];
        assert(source->hold&&source->hold->uses==1);
        memset(stage_bytes+source->stage_offset,30+(int)(source->image.image-200),4);
    }
    memcpy(stage_bytes+alias.buffer_copies[0].region.dstOffset,raw_bytes,4);
    assert(image_alias_compose_guest(&alias)&&!memcmp(workspace,expected,BYTES));
#if SCALABLE_SOURCES
    assert(rc_admission_used_at(&budget,0)==metadata&&rc_admission_used_at(&budget,1)==metadata);
#endif
    release_source_plan(&alias);
#if SCALABLE_SOURCES
    for(unsigned i=0;i<MANY_DONORS;i++)assert(more_holds[i].uses==0);
    assert(!rc_admission_used_at(&budget,0)&&!rc_admission_used_at(&budget,1));
#endif
}

int main(void){
    const RcAdmissionDomain limits[]={{RC_ADMISSION_RAM,0,64u<<20},{RC_ADMISSION_COMMIT,0,64u<<20}};
    assert(rc_admission_init(&budget,1,limits,2,budget_records,4)==RC_ADMISSION_OK);
    const tex_t receiver={.serial=4};
    VkMemImageAliasBaseline owner={.live=1,.span={{0x4400,12,{9,80,12}},{17,0,0x41}}};
    RcInterval demand={0x1100,8,{9,80,12}};
    RcRegistryNativeImageSnapshotFragment fragments[3], scratch[3];
    memcpy(fragments,snapshot,sizeof fragments);
    VkTextureAliasSlice slices[3];
    uint8_t workspace[32]={0};
    VkTextureAliasRead alias={
        .receiver=&receiver, .receiver_logical=19,.registration_only=1,.read={8,10},.result={0,0x1100},
        .receiver_mapping=&demand,.mapping_count=1,.demands=&demand,.demand_count=1,
        .fragments=fragments,.fragment_count=3,.slices=slices,.current=scratch,.receiver_layout={.bytes=16},
        .guest_bytes=12,.workspace_bytes=12,.workspace=workspace,.owners=1,.staging={32,stage_bytes}
    };
    VkTextureAliasFailure failure={0};
    // Gaps cannot silently select unretained raw Vulkan memory.
    assert(image_alias_capture_sources(&alias,1,&failure)==RC_REGISTRY_INVALID&&!buffer_borrows);
    release_source_plan(&alias);
    alias.baseline=&owner;
    fragments[0].native_buffer.offset=0x4c;
    assert(image_alias_capture_sources(&alias,1,&failure)==RC_REGISTRY_STALE&&buffer_borrows==1);
    release_source_plan(&alias);
    fragments[0]=snapshot[0];
    fragments[0].fragment.interval.range.offset=79;
    assert(image_alias_capture_sources(&alias,1,&failure)==RC_REGISTRY_INVALID&&buffer_borrows==1);
    release_source_plan(&alias);
    fragments[0]=snapshot[0];
    offset_drift=1;
    assert(image_alias_capture_sources(&alias,1,&failure)==RC_REGISTRY_STALE&&buffer_borrows==2);
    release_source_plan(&alias);
    offset_drift=0;
    alias.stage_bytes=VK_TEXTURE_ALIAS_MAX_BYTES;
    assert(image_alias_capture_sources(&alias,1,&failure)==RC_REGISTRY_CAPACITY&&buffer_borrows==3);
    release_source_plan(&alias);
    assert(image_alias_capture_sources(&alias,1,&failure)==RC_REGISTRY_OK);
    assert(alias.source_count==1&&alias.buffer_count==2&&alias.stage_bytes==25&&alias.workspace_bytes==16);
    assert(image_alias_read_current_with(&alias,alias.read,scratch,3));
    release_source_plan(&alias);
    alias.raw_upload=pending_fill=19;
    assert(image_alias_capture_sources(&alias,1,&failure)==RC_REGISTRY_OK);
    assert(alias.source_count==1&&!alias.buffer_count&&alias.stage_bytes==4&&alias.workspace_bytes==16);
    unsigned initial_borrows=buffer_borrows;
    uint64_t observed=0;
    assert(vk_texture_alias_read_raw_progress(&alias,alias.read,&observed)&&observed==pending_fill);
    assert(!image_alias_read_current_with(&alias,alias.read,scratch,3));
    assert(!vk_texture_alias_read_raw_finish(&alias,alias.read)&&buffer_borrows==initial_borrows);
    fill_complete=1;
    stale_provider=1;
    assert(!vk_texture_alias_read_raw_finish(&alias,alias.read)&&buffer_borrows==initial_borrows);
    stale_provider=0;
    mapping_drift=1;
    assert(!vk_texture_alias_read_raw_finish(&alias,alias.read)&&buffer_borrows==initial_borrows);
    mapping_drift=0;
    assert(vk_texture_alias_read_raw_finish(&alias,alias.read)&&buffer_borrows==initial_borrows+2);
    assert(!alias.raw_upload&&alias.source_count==1&&alias.buffer_count==2&&alias.stage_bytes==32&&
           alias.upload_offset==16&&alias.workspace_bytes==16);
    pending_fill=0;
    assert(image_alias_read_current_with(&alias,alias.read,scratch,3));
    stale_provider=1;
    assert(!image_alias_read_current_with(&alias,alias.read,scratch,3)); // Same bytes, different native incarnation.
    stale_provider=0;
    owner.live=0;
    assert(!image_alias_read_current_with(&alias,alias.read,scratch,3));
    owner.live=1;
    alias.raw_output=1;
    alias.raw_source=(VkMemSpan){31};
    alias.receiver=NULL; // A buffer gather must not rediscover a texture receiver.
    assert(image_alias_read_current_with(&alias,alias.read,scratch,3));
    mapping_drift=1;
    assert(!image_alias_read_current_with(&alias,alias.read,scratch,3));
    mapping_drift=0;
    diagnostics=1;
    vk_texture_alias_read_record(&alias,77,8);
    assert(completion&&barriers==2&&image_copies==1&&buffer_copies==2&&markers==2&&alias.owners==2);
    const uint8_t expected[12]={11,12,13,55,56,57,58,21,22,23,24,25};
    assert(image_alias_compose_guest(&alias)&&!memcmp(workspace,expected,sizeof expected));

    // A pitched buffer gather must copy only its retained rows. The workspace's old content
    // must not enter the private buffer through an unused pitch gap.
    RcInterval rows[2] = {{0x1100,8,{9,80,3}},{0x1107,8,{9,87,5}}};
    VkTextureAliasSlice row_slices[2] = {slices[0],slices[2]};
    row_slices[1].demand=1;
    VkTextureAliasRead pitched=alias;
    pitched.demands=rows;
    pitched.demand_count=2;
    pitched.slices=row_slices;
    pitched.fragment_count=2;
    pitched.sparse_buffer_rows=1;
    const uint8_t pitched_expected[12]={11,12,13,0,0,0,0,21,22,23,24,25};
    assert(image_alias_compose_guest(&pitched)&&!memcmp(workspace,pitched_expected,sizeof pitched_expected));

    uint32_t part=slices[2].demand;
    slices[2].demand++;
    assert(!image_alias_compose_guest(&alias)); // An uncovered/extra slice may not be treated as zeros.
    slices[2].demand=part;
    completion(recorded);
    assert(alias.owners==1&&owner.live&&!baseline_releases);
    image_alias_release(&alias);
    assert(!alias.baseline&&!owner.live&&baseline_releases==1);
    release_source_plan(&alias);

    // A completely covered ordinary image still needs neither a raw lease nor an extra copy.
    VkTextureAliasRead image_only={
        .result={0,0x1103},.demands=&demand,.demand_count=1,.fragments=&fragments[1],.fragment_count=1,
        .slices=slices,.guest_bytes=4,.workspace_bytes=4,.workspace=workspace,.staging={32,stage_bytes}
    };
    demand=(RcInterval){0x1103,8,{9,83,4}};
    const RcRegistryNativeImageFragment producer = fragments[1].fragment;
    RcInterval storage = producer.interval;
    RcRegistryWrite old_hold = producer.binding.hold;
    projection = (NativeTextureOutput){.image=donor.pub,.descriptor=donor.t.d,.format=donor.f,.guest_bytes=4,
        .allocation=&image_owner,.mapping=&storage,.mapping_count=1,.holds=&old_hold};
    storage.mapping++;
    assert(image_alias_capture_sources(&image_only,1,&failure)==RC_REGISTRY_STALE);
    release_source_plan(&image_only);
    storage.mapping--;
    assert(image_alias_capture_sources(&image_only,1,&failure)==RC_REGISTRY_OK);
    assert(image_only.source_count==1&&!image_only.baseline&&!image_only.buffer_count&&image_only.stage_bytes==4);
    memcpy(stage_bytes,image_bytes,sizeof image_bytes);
    assert(image_alias_compose_guest(&image_only)&&!memcmp(workspace,image_bytes,sizeof image_bytes));
    release_source_plan(&image_only);
    check_many_sources();
    puts("Mixed texture read: exact raw offsets, image overlay, cover, revalidation and native completion owner passed");
    return 0;
}
'''


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texture-source", type=Path, default=root / "runtime/vk_texture.c")
    parser.add_argument("--layout-source", type=Path, default=root / "runtime/vk_texture_alias_bytes.h")
    args = parser.parse_args()
    texture = args.texture_source.read_text(encoding="utf-8")
    layout = args.layout_source.read_text(encoding="utf-8")

    def extract(name):
        declaration = re.search(r"^[a-zA-Z_0-9 *]+[ *]" + name + r"\([^;{]*\)\s*\{", texture, re.M)
        assert declaration, name
        return function(texture[declaration.start():], name)

    scalable = "image_alias_source_storage(" in texture
    start = texture.index("} ImageAliasBufferCopy;")
    start = texture.rindex("typedef struct {", 0, start) if scalable else texture.index("enum { IMAGE_ALIAS_MAX_SOURCES")
    types = texture[start:texture.index("struct VkTextureAliasRead {")]
    sources = ("image_alias_compare_handles", "image_alias_source_storage", "image_alias_source_index",
               "image_alias_capture_source_table", "image_alias_sources_release") if scalable else ()
    snapshot = ("image_alias_snapshot",) if "static RcRegistryResult image_alias_snapshot(" in texture else ()
    record = ("image_alias_record_copies",) if "static void image_alias_record_copies(" in texture else ()
    source = f"#define SCALABLE_SOURCES {int(scalable)}\n" + PREFIX + types + STUBS + "\n".join(map(extract, (
        "image_alias_stage_buffer", "native_texture_output_contains", *sources,
        "image_alias_capture_sources", "image_alias_fragment_equal",
        *snapshot, "image_alias_read_candidate_with", "image_alias_read_current_with",
        "image_alias_finish_upload_layout", "vk_texture_alias_read_raw_progress", "vk_texture_alias_read_raw_finish",
        *record, "vk_texture_alias_read_record",
    ))) + function(layout, "texture_alias_compose") + extract("image_alias_compose_guest") + MAIN
    cc = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="mixed-image-buffer-") as directory:
        cfile = Path(directory) / "check.c"
        binary = cfile.with_suffix(".exe")
        cfile.write_text(source, encoding="utf-8")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=undefined", "-I", str(root / "runtime"),
                        str(cfile), "-o", str(binary)],
                       env=env, check=True)
        subprocess.run([str(binary)], env=env, check=True, timeout=20)


if __name__ == "__main__":
    main()


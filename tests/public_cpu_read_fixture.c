#include <assert.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "resource_core_registry.h"
#include "fixture_abi.h"

#define OK(call) assert((call) == RC_REGISTRY_OK)
#define RT_ADDRESS_LIMIT (UINT64_C(1)<<63)
#define VK_WHOLE_SIZE UINT64_MAX
enum { NVN_TEXTURE_TARGET_2D=1, NVN_TEXTURE_TARGET_3D, NVN_FORMAT_R8=1, NVN_FORMAT_RGBA8=0x25,
    VK_FORMAT_R8_UNORM=9, VK_FORMAT_R8G8B8A8_UNORM=37, VK_IMAGE_ASPECT_COLOR_BIT=1,
    VK_IMAGE_ASPECT_DEPTH_BIT=2, VK_IMAGE_ASPECT_STENCIL_BIT=4,
    VK_IMAGE_TYPE_2D=1, VK_IMAGE_USAGE_TRANSFER_SRC_BIT=1, VK_IMAGE_LAYOUT_GENERAL=1, VK_SUCCESS=0,
    VK_NOT_READY=1, VK_ERROR_DEVICE_LOST=-4, VK_PIPELINE_STAGE_2_COPY_BIT=2, VK_PIPELINE_STAGE_2_HOST_BIT=4,
    VK_ACCESS_2_TRANSFER_READ_BIT=2, VK_ACCESS_2_TRANSFER_WRITE_BIT=4, VK_ACCESS_2_HOST_READ_BIT=8, ALL_RW=1,
    ST_FREE=0, ST_INACCESSIBLE=1, PERM_R=1, PERM_RW=3, RC_BACKING_OK=0 };
typedef uint64_t VkImage, VkBuffer, VkCommandBuffer, VkPipelineStageFlags2, VkDeviceSize, VkImageView;
typedef struct {uint64_t srcOffset,dstOffset,size;} VkBufferCopy;
typedef struct {uint32_t aspectMask,baseMipLevel,levelCount,baseArrayLayer,layerCount;} VkImageSubresourceRange;
typedef struct {int unused;} VkImageViewCreateInfo, ImgKey, RcVmaDependency;
#define VK_POOL_KINDS 1
#define IMAGE_ALIAS_MAX_WRITE_FRAGMENTS 32
#define RC_BUDGET_TEXTURE 1
#define VK_ERROR_OUT_OF_HOST_MEMORY (-1)
#define VK_MCLS_DEVICE 2
typedef uint32_t VkImageAspectFlags, VkFormat;
typedef int VkResult;
typedef void *VmaAllocation;
typedef struct {uint32_t aspectMask,mipLevel,baseArrayLayer,layerCount;} Subresource, VkImageSubresourceLayers;
typedef struct {int x,y,z;} Offset;
typedef struct {uint32_t width,height,depth;} Extent;
typedef struct {uint64_t bufferOffset; Subresource imageSubresource; Offset imageOffset; Extent imageExtent;} VkBufferImageCopy;
typedef struct {int width,height,depth,xoffset,yoffset,zoffset;} NVNcopyRegion;
typedef struct {int set_target,set_dsmode,set_format,format,set_levels,base_level,set_layers,min_layer; unsigned nlevels,nlayers;} view_t;
typedef struct {int width,height,target,levels,samples,format;uint32_t depth;} nvn_texdesc;
typedef struct {int vk,ideal,depth,stencil,astc_bw,expand_rgb;} TexVkFormat;
typedef struct {nvn_texdesc d; uint64_t addr;} TexKey;
typedef struct {nvn_texdesc d;struct {uint64_t size; int layers;} L; uint8_t *mem; uint64_t serial;} tex_t;
typedef struct {VkImage image; uint64_t addr; VkFormat format;
    uint32_t type,aspect,scale,skip,depth,width,height,samples,levels,layers,usage;
    int gone,gpu_owned,fb_learned;} VkTexImage;
typedef struct {VkTexImage pub; tex_t t;RcRegistryPosition native_write,sample_general_inline;
    uint64_t sample_general_revision,sample_general_upload,wr_fence,content_revision;
    void *alloc, *bound_to,*pending_write;
    int pin_detached,overwritten,superseded,released,retiring,alias_fragment_owner;
    RcInterval *alias_full_demand,*image_private_mapping; uint32_t alias_full_count,image_private_mapping_count;
    TexVkFormat f;} Tex;
typedef struct {int cls;uint64_t allocation;} VkMemSpan;
typedef struct {Tex *entry;void *allocation;VkTexImage image;nvn_texdesc descriptor;TexVkFormat format;
    uint64_t guest_bytes;uint32_t mapping_count;RcInterval *mapping;RcRegistryWrite *holds;} NativeTextureOutput;
typedef struct {int used;} RcBudgetTicket;
typedef struct {uint8_t *map;VkBuffer buffer;VmaAllocation allocation;} VkPreparedStaging;
typedef struct {uint64_t upload;VkDeviceSize size;} VkTextureAliasStagingProgress;
typedef struct {RcInterval receiver;RcRegistryNativeImageFailure coverage;} VkTextureAliasFailure;
typedef struct {RcRegistryBufferSpan span;RcRegistryPosition cut;uint64_t fill;} VkMemImageAliasBaseline;
typedef struct VkTextureAliasRead VkTextureAliasRead;
typedef void (*NvnTexelStorageRange)(void *,uint64_t,uint64_t);
typedef struct {int retired;} VkMemoryImageHold;
typedef struct {uint64_t address,size,identity;} MemBackingCapture;
typedef struct {uint64_t base,size; int state,perm;} RtMemoryInfo;
typedef struct {uintptr_t vma;} VkCtx;
static RcRegistry *registry;
static RcRegistry *fixture_registry(void){return registry;}
static RcRegistryPosition producer={7,3};
static RcRegistryWrite writer, image_owner;
static RcInterval original_map={UINT64_C(0x8000000000),91,{27,0,16}}, current_map;
static Tex entry;
static NativeTextureOutput donor;
static int alias_fragment_owners, bound_alias_owners;
static VkMemImageAliasBaseline *borrowed_raw;
static uint64_t earliest_upload, latest_upload;
static int raw_reads, fill_waits, restore_attempts;
static void (*late_release)(void *);
static void *late_owner;
static uint8_t raw_bytes[64];
static VkCtx ctx;
static VkMemoryImageHold image_hold;
static uint64_t fence_done, protection, destination_mapping;
static int alias_current, lost, uploads, writes, waits, origin_holds, destination_holds, captures, frees;
static int staged, destroyed, deferred, legacy, blocked, invalidate_result, read_only;
static int texture_lock, memory_lock;
static int test_case;
static NVNcopyRegion *guest_region;
enum {SUCCESS, SOURCE_REMAP, CANCEL, READONLY, DEST_REMAP, PROTECT, LOST, INVALIDATE, NEW_WRITER, NO_LAYOUT, UPLOAD_LOST, REGION_MUTATED, RGBA, QUEUED, RESERVED, EVENT_HELD, COMPARE, MIXED_R8, MIXED_RGBA, MIXED_RESIDENT, RAW_R8, MIXED_REMAP, MIXED_CUT_MOVED, MIXED_READONLY,
    MIXED_DEST_REMAP, MIXED_PROTECT, MIXED_INVALIDATE, MIXED_RETIRED, MIXED_UNCERTAIN, MIXED_UNACCEPTED, MIXED_ABSENT};
static uint8_t pixels[64]={41,42,43,44, 51,52,53,54, 61,62,63,64, 71,72,73,74};
static uint8_t stage_bytes[160], pending_bytes[160];
static jmp_buf refusal;

static int mixed(void){return test_case>=MIXED_R8;}
static int rgba(void){return test_case==RGBA || test_case==MIXED_RGBA;}
static void lock(void){assert(mixed() || !texture_lock);++texture_lock;}
static void unlock(void){assert(texture_lock>0);--texture_lock;}
static int mapping_guard;
static void plat_mutex_lock_shared(int *which){assert(which==&mapping_guard && !memory_lock++);}
static void plat_mutex_unlock_shared(int *which){assert(which==&mapping_guard && memory_lock--==1);}
#define lock mapping_guard
static int host_backing_range_current_locked(const MemBackingCapture *capture,uint64_t address,uint64_t size){
    assert(capture && address==capture->address && size==capture->size && memory_lock==1);
    return capture->identity==destination_mapping;
}
static void query_locked(uint64_t address,RtMemoryInfo *out){
    *out=(RtMemoryInfo){address,4096,2,read_only?PERM_R:PERM_RW};
}
#include "fixture_memory.h"
#undef lock
static int mem_backing_capture(uint64_t address,uint64_t size,MemBackingCapture **out){
    assert((mixed() || texture_lock) && address && size);
    *out=malloc(sizeof **out);
    assert(*out);
    **out=(MemBackingCapture){address,size,destination_mapping};
    ++captures;
    return RC_BACKING_OK;
}
static void mem_backing_release(MemBackingCapture **out){if(*out){free(*out);*out=NULL;++frees;}}
static uint64_t mem_protection_serial(void){return protection;}
static int mem_protection_range_unchanged(uint64_t address,uint64_t size,uint64_t captured){
    assert(address && size && texture_lock);
    return protection==captured;
}
static void mem_read_release(void){plat_mutex_unlock_shared(&mapping_guard);++destination_holds;}
static int nvn_gpu_only(uint64_t address){return address==original_map.address;}
static int nvn_fmt_bpb(int format,int *width,int *height){*width=*height=1;return format==NVN_FORMAT_R8?1:4;}
static void make_key(const tex_t *t,TexKey *out){*out=(TexKey){.d=t->d,.addr=(uintptr_t)t->mem};}
static int by_key,rl,rcv,rq_n,r_busy,ck_reads,ck_diff,ck_diff_unexplained,ck_diff_idle;
enum {DR_READ_TEXELS, DR_CHECK};
static uint64_t dr_calls[2],dr_tcalls[2],dr_waits[2],dr_twaits[2],dr_ns[2],dr_tns[2];
static void *nvn_hm_get(int *map,const void *key){
    assert(texture_lock);
    if(map==&alias_fragment_owners) {
        assert(mixed() && *(const VkImage *)key==entry.pub.image);
        return &donor;
    }
    assert(map==&by_key && ((const TexKey *)key)->addr==(uintptr_t)entry.t.mem);
    return mixed() && test_case!=MIXED_RESIDENT ? NULL : &entry;
}
static int vk_mem_lookup(uint64_t address,uint64_t size,VkMemSpan *span){
    assert(texture_lock && span && address==original_map.address && size==original_map.range.size);
    *span=(VkMemSpan){VK_MCLS_DEVICE,800};
    return alias_current;
}
static unsigned vk_mem_device_intervals(const VkMemSpan *span,uint64_t address,uint64_t size,RcInterval **out){
    assert(span && address==current_map.address && size==original_map.range.size);
    *out=NULL;
    if (mixed() && span->allocation!=800) return 0;
    *out=malloc(sizeof **out);assert(*out);**out=current_map;return 1;
}
static VkCtx *vk_ctx(void){return &ctx;}
static VkMemoryImageHold *vk_mem_image_hold(VkImage image){
    assert(texture_lock && image==entry.pub.image && !origin_holds++);
    return &image_hold;
}
static int vk_mem_image_hold_retired(const VkMemoryImageHold *hold){assert(hold==&image_hold);return hold->retired;}
static void vk_mem_image_release(VkMemoryImageHold *hold){
    if (!hold) return;
    assert(!texture_lock && hold==&image_hold && origin_holds--==1);
}
static int vk_sync_stopping(void){return 0;}
static int vk_device_is_lost(void){return lost;}
static void vk_sync_start(void){}
static uint64_t vk_sync_fence_done(void){return fence_done;}
static uint64_t vk_sync_issued(void){return producer.scene-(test_case==RESERVED);}
static uint64_t vk_sync_handed_fence(void){return producer.scene-(test_case==MIXED_UNACCEPTED);}
static void vk_queue_lock(void){
    assert(mixed() && !texture_lock);
    OK(rc_registry_handoff_through(registry,producer));
}
static void vk_queue_unlock(void){assert(mixed() && !texture_lock);}
static int vk_exec_held(void){return test_case==EVENT_HELD;}
static int vk_sync_fence_wait(uint64_t scene,uint64_t timeout){
    assert(!texture_lock && origin_holds==(test_case!=RAW_R8) && scene==producer.scene && timeout==UINT64_MAX);
    ++waits;
    if(test_case==LOST){lost=1;return 0;}
    if(test_case==CANCEL){
        assert(rc_registry_image_producer_projection_abandon(registry,writer,producer,
            (RcRegistryImageBinding){entry.pub.image,image_owner},0));
        OK(rc_registry_write_cancel(registry,writer));
        fence_done=scene;
        return 1;
    }
    if (test_case!=RAW_R8) {
        OK(rc_registry_handoff_through(registry,producer));
        rc_registry_image_producer_native_completed(registry,writer,producer,(RcRegistryImageBinding){entry.pub.image,image_owner});
        OK(rc_registry_write_cancel(registry,writer));
    }
    if(test_case==MIXED_RETIRED) image_hold.retired=1;
    fence_done=scene;
    if(test_case==SOURCE_REMAP)current_map.mapping++;
    if(test_case==NEW_WRITER)entry.content_revision++;
    if(test_case==REGION_MUTATED)guest_region->xoffset=0;
    return 1;
}
static void vk_upload_progress(uint64_t *submitted,uint64_t *completed){
    if(submitted)*submitted=mixed() ? latest_upload : 1;
    if(completed)*completed=mixed() ? earliest_upload : 1;
}
static void *vk_staging(size_t size,VkBuffer *buffer,VmaAllocation *allocation){
    assert(texture_lock && origin_holds==1 && size<=sizeof stage_bytes && ++staged==destroyed+1);
    *buffer=6;*allocation=stage_bytes;return stage_bytes;
}
static VkCommandBuffer vk_upload_begin(void){return 5;}
static void vk_upload_end(void){}
static uint64_t vk_upload_recording_sequence(VkCommandBuffer cb){
    assert(mixed() && cb==5 && texture_lock && latest_upload==1);return 2;
}
static void vk_upload_defer_release(uint64_t seq,void (*release)(void *),void *owner){
    assert(mixed() && seq==2 && release && owner && !late_release && !late_owner);
    late_release=release;
    late_owner=owner;
    ++deferred;
}
static void barrier(VkCommandBuffer cb,VkPipelineStageFlags2 stage,uint64_t access,...){
    assert(cb==5 && stage && access);
}
static void vkCmdCopyImageToBuffer(VkCommandBuffer cb,VkImage image,int layout,VkBuffer buffer,
    unsigned count,const VkBufferImageCopy *regions){
    const VkBufferImageCopy *part=regions;
    assert(texture_lock && fence_done==producer.scene && cb==5 && image==entry.pub.image &&
        layout==VK_IMAGE_LAYOUT_GENERAL && buffer==6 && count==1 && part->imageSubresource.mipLevel==0 &&
        part->imageSubresource.aspectMask==VK_IMAGE_ASPECT_COLOR_BIT);
    const unsigned pixel_width=rgba() ? 4 : 1;
    const unsigned row=part->imageExtent.width*pixel_width;
    uint8_t *transfer=mixed() ? pending_bytes : stage_bytes;
    for(uint32_t y=0;y<part->imageExtent.height;y++)
        memcpy(transfer+part->bufferOffset+y*row,pixels+pixel_width*(4*(part->imageOffset.y+y)+part->imageOffset.x),row);
    ++writes;
}
static void vkCmdCopyBuffer(VkCommandBuffer cb,VkBuffer src,VkBuffer dst,uint32_t count,const VkBufferCopy *copy){
    assert(mixed() && texture_lock && cb==5 && src==9 && dst==6 && count==1 && earliest_upload==1 &&
        copy->srcOffset+copy->size<=sizeof raw_bytes && copy->dstOffset+copy->size<=sizeof pending_bytes);
    memcpy(pending_bytes+copy->dstOffset,raw_bytes+copy->srcOffset,copy->size);
    ++raw_reads;
}
static uint64_t vk_upload_flush(int wait){
    if (mixed()) {
        if (!wait) { assert(!texture_lock && !latest_upload); latest_upload=1; return 1; }
        assert(texture_lock && raw_reads && !uploads && earliest_upload==1);
        ++uploads;
        if(test_case==MIXED_UNCERTAIN) {lost=1;return 0;}
        if(test_case==MIXED_PROTECT) ++protection;
        if(test_case==MIXED_DEST_REMAP) ++destination_mapping;
        memcpy(stage_bytes,pending_bytes,sizeof stage_bytes);
        latest_upload=earliest_upload=2;
        return 2;
    }
    assert(wait && texture_lock && writes==uploads+1);
    ++uploads;
    if(test_case==PROTECT)++protection;
    if(test_case==DEST_REMAP)++destination_mapping;
    if(test_case==UPLOAD_LOST){lost=1;return 0;}
    return 2;
}
static VkResult vmaInvalidateAllocation(uintptr_t vma,VmaAllocation alloc,uint64_t offset,uint64_t size){
    assert(vma==ctx.vma && alloc==stage_bytes && !offset);
    if(mixed())assert(earliest_upload==2 && size>0 && size<sizeof stage_bytes);
    else assert(size==VK_WHOLE_SIZE);
    return invalidate_result;
}
static void resource_core_vma_destroy_buffer(uintptr_t vma,VkBuffer buffer,VmaAllocation alloc){
    assert(vma==ctx.vma && buffer==6 && alloc==stage_bytes);++destroyed;
}
static void vk_defer_buffer(VkBuffer buffer,VmaAllocation alloc){
    assert(buffer==6 && alloc==stage_bytes && lost);++deferred;
}
static void vk_device_lost_check(VkResult result,const char *where){assert(result==VK_ERROR_DEVICE_LOST && where);lost=1;}
static void shrink_rgb(int format,const void *source,void *target,size_t width){
    (void)format;(void)source;(void)target;(void)width;abort();
}
static void log_limited(volatile int32_t *n,const char *format,...){assert(n&&format);}
static int32_t plat_inc32(volatile int32_t *n){return ++*n;}
static void rt_log(const char *format,...){assert(format);}
static void rt_fatal(const char *format,...){assert(format);++blocked;longjmp(refusal,1);}
static int gfx_replay_async(void){return 1;}
static void plat_mutex_lock(int *m){assert(m==&rl);++*m;}
static void plat_mutex_unlock(int *m){assert(m==&rl && --*m==0);}
static uint64_t kern_now_ns(void){return 0;}
static int replay_check(void){return test_case==COMPARE;}
static uint64_t perf_begin(void){return 1;}
static int core_away_unlocked(void){assert(rl==1);return 1;}
static void core_back_unlocked(int *away){assert(rl==1&&*away==1);}
static void wait_counted(int kind,uint64_t start){
    assert(rl==1 && start<=1);
    ++dr_waits[kind];++dr_twaits[kind];++dr_ns[kind];++dr_tns[kind];
}
static void publish_producer(void){
    if(test_case==RAW_R8||test_case==MIXED_ABSENT) {
        OK(rc_registry_declare_through(registry,producer));
        return;
    }
    OK(rc_registry_write_begin_at(registry,original_map.range,RC_REGISTRY_IMAGE,producer,&writer));
    OK(rc_registry_image_producer_add(registry,writer,producer));
    OK(rc_registry_declare_through(registry,producer));
    const unsigned pixel=rgba()?4:1;
    const RcSourceRange parts[]={{original_map.range.backing,5*pixel,pixel},
                                {original_map.range.backing,9*pixel,pixel}};
    const RcRegistryNativeImageWrite write={{original_map,image_owner},mixed()?parts:&original_map.range,mixed()?2:1};
    OK(rc_registry_native_image_write_fragments(registry,entry.pub.image,&write,1,producer));
    RcRegistryImageBinding binding={entry.pub.image,image_owner};
    OK(rc_registry_image_producer_resolve(registry,writer,producer,&binding,0));
}
static void plat_cond_wait(int *condition,int *mutex){
    assert(condition==&rcv && mutex==&rl && rl==1 && rq_n==1 && test_case==QUEUED && !texture_lock);
    producer=(RcRegistryPosition){8,3};
    entry.native_write=(RcRegistryPosition){8,2};
    entry.wr_fence=8;
    ++entry.content_revision;
    publish_producer();
    pixels[5]=152;pixels[6]=153;pixels[9]=162;pixels[10]=163;
    rq_n=0;
}
static uint64_t vk_texture_last_write(const tex_t *t){
    assert(t->d.target==NVN_TEXTURE_TARGET_3D||entry.pub.fb_learned||test_case==MIXED_ABSENT);return 0;
}
static size_t vk_texture_texel_span(const tex_t *t,const view_t *v,const NVNcopyRegion *r,uint64_t row,uint64_t img){
    assert(t && !v && r && row && img && test_case==COMPARE);return 7;
}
static void vk_read_texels(const tex_t *t,const view_t *v,const NVNcopyRegion *r,uint8_t *dst,uint64_t row,uint64_t img){
    (void)v;(void)r;(void)dst;(void)row;(void)img;assert(t->d.target==NVN_TEXTURE_TARGET_3D||entry.pub.fb_learned||test_case==MIXED_ABSENT);++legacy;longjmp(refusal,1);
}
typedef struct {uintptr_t args[6];} Cpu;
static void *arg_p(Cpu *c,int i){return (void *)c->args[i];}
static uint64_t arg_x(Cpu *c,int i){return c->args[i];}
#define NVN_BLOCKING() ((void)0)
#define NVN_API(name) static void name(Cpu *c)

static int nvn_tex_is_linear(const nvn_texdesc *desc){return desc->target==NVN_TEXTURE_TARGET_2D;}
static uint32_t nvn_tex_linear_stride(const nvn_texdesc *desc){
    return (uint32_t)desc->width*(desc->format==NVN_FORMAT_R8?1:4);
}
static int nvn_deswizzle_geom(const nvn_texdesc *desc,unsigned level,unsigned layer,...){
    (void)desc;(void)level;(void)layer;abort();
}
static unsigned gob_off(uint32_t x,uint32_t y){(void)x;(void)y;abort();}
static uint64_t nvn_bl_offset(uint32_t x,uint32_t y,uint32_t z,uint32_t row,uint32_t rows,unsigned th,unsigned td){
    (void)x;(void)y;(void)z;(void)row;(void)rows;(void)th;(void)td;abort();
}
#include "fixture_box_layout.h"
#include "fixture_alias_layout.h"
#include "fixture_alias_types.h"

static RcRegistryResult vk_mem_image_alias_baseline_capture_handed(uint64_t address,uint64_t size,const RcInterval *map,
    uint32_t count,RcRegistryPosition cut,VkMemImageAliasBaseline **out,uint64_t *fill){
    assert(mixed() && texture_lock && !borrowed_raw && !*out && count==1 && address==map->address &&
        size==map->range.size && !rc_registry_position_compare(cut,producer) && latest_upload==1 && !earliest_upload);
    if(test_case==MIXED_ABSENT)return RC_REGISTRY_OK;
    borrowed_raw=malloc(sizeof *borrowed_raw);assert(borrowed_raw);
    *borrowed_raw=(VkMemImageAliasBaseline){.span={.interval=*map,.native={800,0,0}},.cut=cut,.fill=1};
    *out=borrowed_raw;
    *fill=1;
    return RC_REGISTRY_OK;
}
static void vk_mem_image_alias_baseline_release(VkMemImageAliasBaseline **out){
    if(*out) {assert(*out==borrowed_raw);free(*out);borrowed_raw=*out=NULL;}
}
static const RcRegistryBufferSpan *vk_mem_image_alias_baseline_submitted_spans(const VkMemImageAliasBaseline *raw,
    uint64_t fill,uint32_t *count) {
    (void)raw; (void)fill; (void)count;
    abort(); // The handed CPU read must use only its completed baseline.
}
static const RcRegistryBufferSpan *vk_mem_image_alias_baseline_spans(const VkMemImageAliasBaseline *raw,uint32_t *count){
    assert(raw==borrowed_raw);
    *count=raw && earliest_upload>=raw->fill;
    return *count?&raw->span:NULL;
}
static int vk_mem_image_alias_baseline_current(const VkMemImageAliasBaseline *raw){
    return raw && raw==borrowed_raw && raw->span.interval.mapping==current_map.mapping &&
        !rc_registry_position_compare(raw->cut,rc_registry_handed_cutoff(registry));
}
static int vk_mem_image_alias_baseline_source(const VkMemImageAliasBaseline *raw,uint32_t index,
    RcRegistryBufferLocation native,uint64_t size,VkBuffer *buffer,VkDeviceSize *offset){
    if(!vk_mem_image_alias_baseline_current(raw)||index||earliest_upload<raw->fill||
        native.allocation!=raw->span.native.allocation || native.incarnation || native.offset+size>raw->span.interval.range.size)
        return 0;
    *buffer=9;
    *offset=native.offset;
    return 1;
}
static int vk_staging_prepare_wait(uint64_t sequence){
    assert(mixed() && !texture_lock && sequence==1 && borrowed_raw && latest_upload==1 && !earliest_upload);
    ++fill_waits;
    earliest_upload=1;
    if(test_case==MIXED_REMAP)++current_map.mapping;
    if(test_case==MIXED_CUT_MOVED){
        const RcRegistryPosition next={producer.scene+1,3};
        OK(rc_registry_declare_through(registry,next));
        OK(rc_registry_handoff_through(registry,next));
    }
    return 1;
}
static VkResult vk_texture_alias_read_prepare(VkTextureAliasRead *const *aliases,uint32_t count,int required,
    uint64_t previous,VkTextureAliasStagingProgress *progress,RcVmaDependency *native){
    VkTextureAliasRead *alias=*aliases;
    assert(mixed() && !texture_lock && count==1 && required && !previous && alias->stage_bytes<sizeof stage_bytes);
    *progress=(VkTextureAliasStagingProgress){0};
    *native=(RcVmaDependency){0};
    alias->workspace=calloc(1,alias->workspace_bytes);
    assert(alias->workspace);
    memset(stage_bytes,0xee,sizeof stage_bytes);memset(pending_bytes,0xdd,sizeof pending_bytes);
    alias->staging=(VkPreparedStaging){stage_bytes,6,stage_bytes};
    ++staged;
    ++restore_attempts;
    return VK_SUCCESS;
}
static VkResult vk_replay_metadata_allocate(size_t bytes,int clear,int producer,RcBudgetTicket *ticket,void **out){
    assert(bytes&&producer==RC_BUDGET_TEXTURE&&!ticket->used);
    *out=clear?calloc(1,bytes):malloc(bytes);
    assert(*out);
    ticket->used=1;
    return VK_SUCCESS;
}
static void vk_replay_metadata_free(void *memory,RcBudgetTicket *ticket){
    assert(memory && ticket->used);
    ticket->used=0;
    free(memory);
}
static void image_alias_release(void *opaque){
    VkTextureAliasRead *alias=opaque;
    assert(alias->owners==1 && mixed());
    for(uint32_t i=0;i<alias->source_count;i++)vk_mem_image_release(alias->sources[i].hold);
    if(alias->sources)vk_replay_metadata_free(alias->sources,&alias->source_ticket);
    if(alias->staging.buffer) {assert(earliest_upload==2 || (lost && deferred));++destroyed;}
    free(alias->workspace);
    free(alias->demands);
    free(alias->receiver_mapping);
    free(alias->fragments);
    free(alias->current);
    free(alias->slices);
    vk_mem_image_alias_baseline_release(&alias->baseline);
    free(alias);
}
static int image_alias_layout_for(const VkTexImage *image,const nvn_texdesc *desc,const TexVkFormat *format,
    uint64_t bytes,VkTextureAliasLayout *layout){
    assert(image==&entry.pub && desc==&entry.t.d && format==&entry.f && bytes==original_map.range.size);
    return texture_alias_layout(desc,image,layout);
}
static int image_alias_bound_source_current(const Tex *entry_,VkImage image){(void)entry_;(void)image;abort();}
static int storage_seed_receiver_current(const VkTextureAliasRead *alias,const Tex *destination){(void)alias;(void)destination;abort();}
static int image_alias_sample_receiver_locked(int logical,const VkTextureAliasRead *alias){(void)logical;(void)alias;abort();}
static int image_alias_predecessor_matches(const VkTextureAliasRead *alias,const Tex *entry_){(void)alias;(void)entry_;abort();}
static const char *cfg_get(const char *key){assert(!strcmp(key,"NX2NAVI_PROBE_CURRENT_IMAGE_ALIAS"));return NULL;}

#include "fixture_texture.h"
#include "fixture_backend.h"
static struct {void (*read_texels)(const tex_t *,const view_t *,const NVNcopyRegion *,uint8_t *,uint64_t,uint64_t);}
    backend={vk_be_read_texels};
static typeof(backend) *gfx_backend(void){return &backend;}
#include "fixture_nvn.h"

static void begin_case(int which){
    test_case=which;
    static const uint8_t initial[]={41,42,43,44, 51,52,53,54, 61,62,63,64, 71,72,73,74};
    memcpy(pixels,initial,sizeof initial);
    producer=(RcRegistryPosition){7,3};
    registry=rc_registry_create((RcRegistryConfig){32,32,32});
    assert(registry);
    original_map.range.size=rgba() ? 64 : 16;
    current_map=original_map;
    const tex_t texture={.d={4,4,NVN_TEXTURE_TARGET_2D,1,1,rgba() ? NVN_FORMAT_RGBA8 : NVN_FORMAT_R8,1},
                         .L={original_map.range.size,1},
                         .mem=(uint8_t*)(uintptr_t)original_map.address,.serial=10};
    entry=(Tex){.t=texture,.native_write={7,2},.sample_general_inline={7,1},
        .sample_general_revision=190,.wr_fence=7,.content_revision=800,.alloc=&entry,
        .alias_fragment_owner=1,.alias_full_demand=&original_map,.alias_full_count=1,
        .image_private_mapping_count=1,.image_private_mapping=&original_map,
        .pub={.image=18,.addr=original_map.address,.format=rgba() ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_R8_UNORM,.type=VK_IMAGE_TYPE_2D,
            .aspect=VK_IMAGE_ASPECT_COLOR_BIT,.scale=1,.width=4,.height=4,.samples=1,.levels=1,.layers=1,
            .usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT,.gpu_owned=1}};
    donor=(NativeTextureOutput){.entry=&entry};
    if (which!=RAW_R8 && which!=MIXED_ABSENT)
        OK(rc_registry_image_hold_begin_extents(registry,entry.pub.image,&original_map,1,&image_owner));
    publish_producer();
    assert(!rc_registry_position_handed(registry,producer));
    if(which==MIXED_RESIDENT)OK(rc_registry_handoff_through(registry,producer));
    if(which==QUEUED){
        OK(rc_registry_handoff_through(registry,producer));
        rc_registry_image_producer_native_completed(registry,writer,producer,(RcRegistryImageBinding){entry.pub.image,image_owner});
        OK(rc_registry_write_cancel(registry,writer));
    }
    if(which==NO_LAYOUT)entry.sample_general_inline=(RcRegistryPosition){7,5};
    alias_current=1; fence_done=0; protection=20; destination_mapping=70;
    lost=uploads=writes=waits=origin_holds=destination_holds=captures=frees=0;
    staged=destroyed=deferred=blocked=legacy=0;
    rq_n=which==QUEUED;
    ck_reads=ck_diff=ck_diff_unexplained=ck_diff_idle=0;
    memset(dr_calls,0,sizeof dr_calls);memset(dr_tcalls,0,sizeof dr_tcalls);
    memset(dr_waits,0,sizeof dr_waits);memset(dr_twaits,0,sizeof dr_twaits);
    read_only=which==READONLY || which==MIXED_READONLY;
    image_hold=(VkMemoryImageHold){0};
    assert(!late_owner && !late_release && !borrowed_raw);
    latest_upload=earliest_upload=0;
    raw_reads=fill_waits=restore_attempts=0;
    for(unsigned i=0;i<sizeof raw_bytes;i++)raw_bytes[i]=(uint8_t)(120+i);
    if (mixed()) for(unsigned i=0;i<sizeof pixels;i++)pixels[i]=(uint8_t)(40+i);
    invalidate_result=which==INVALIDATE || which==MIXED_INVALIDATE ? -3 : VK_SUCCESS;
}
static uint8_t result[24];
int main(void){
    for(int which=SUCCESS;which<=COMPARE;++which){
        begin_case(which);
        memset(result,0xa6,sizeof result);
        NVNcopyRegion box={.width=2,.height=2,.depth=1,.xoffset=1,.yoffset=1};
        guest_region=&box;
        if(which==RGBA)for(unsigned i=0;i<sizeof pixels;i++)pixels[i]=(uint8_t)(90+i);
        Cpu cpu={{(uintptr_t)&entry.t,0,(uintptr_t)&box,(uintptr_t)result,which==RGBA ? 12 : 5,which==RGBA ? 24 : 10}};
        if (!setjmp(refusal)) nvnTextureReadTexelsStrided(&cpu);
        assert(ck_reads==(which==COMPARE) && !ck_diff && !ck_diff_unexplained && !ck_diff_idle);
        assert(waits==(which!=EVENT_HELD && which!=RESERVED) &&
               !legacy && !texture_lock && !memory_lock && !origin_holds && captures==frees && staged==destroyed+deferred);
        assert(dr_calls[DR_READ_TEXELS]==1 && dr_waits[DR_READ_TEXELS]==(which==QUEUED));
        if(which==QUEUED)assert(producer.scene==8 && fence_done==8 && !rq_n);
        if(which==SUCCESS||which==REGION_MUTATED||which==QUEUED||which==COMPARE){
            assert(writes==1+(which==COMPARE) && uploads==writes && !blocked && destination_holds==writes);
            const int delta=which==QUEUED ? 100 : 0;
            assert(result[0]==52+delta && result[1]==53+delta && result[5]==62+delta && result[6]==63+delta);
            for(unsigned i=0;i<sizeof result;i++) if(i!=0&&i!=1&&i!=5&&i!=6)assert(result[i]==0xa6);
        } else if(which==RGBA) {
            assert(writes==1 && uploads==1 && !blocked && destination_holds==1);
            assert(!memcmp(result,pixels+20,8) && !memcmp(result+12,pixels+36,8));
            for(unsigned i=0;i<sizeof result;i++)if((i>=8&&i<12)||i>=20)assert(result[i]==0xa6);
        } else {
            for(unsigned i=0;i<sizeof result;i++)assert(result[i]==0xa6);
            assert(blocked==(which!=LOST && which!=UPLOAD_LOST));
            if(which==UPLOAD_LOST)assert(writes==1 && uploads==1 && deferred==1 && !destroyed);
            if(which==SOURCE_REMAP||which==CANCEL||which==LOST||which==NEW_WRITER||which==NO_LAYOUT||which==RESERVED||which==EVENT_HELD)
                assert(!writes&&!uploads);
        }
        rc_registry_destroy(registry);
    }
    begin_case(SUCCESS);
    entry.t.d.target=NVN_TEXTURE_TARGET_3D;
    const NVNcopyRegion one={.width=1,.height=1,.depth=1};
    Cpu cpu={{(uintptr_t)&entry.t,0,(uintptr_t)&one,(uintptr_t)result}};
    if(!setjmp(refusal))nvnTextureReadTexels(&cpu);
    assert(legacy==1 && !blocked&&!waits&&!writes);
    rc_registry_destroy(registry);
    begin_case(SUCCESS);
    entry.pub.fb_learned=1;
    if(!setjmp(refusal))nvnTextureReadTexels(&cpu);
    assert(legacy==1 && !blocked&&!waits&&!writes);
    rc_registry_destroy(registry);
    for (int which=MIXED_R8;which<=MIXED_ABSENT;++which) {
        begin_case(which);
        memset(result,0xa6,sizeof result);
        const NVNcopyRegion box={.width=2,.height=2,.depth=1,.xoffset=1,.yoffset=1};
        const unsigned pixel=rgba()?4:1, pitch=2*pixel+3;
        Cpu request={{(uintptr_t)&entry.t,0,(uintptr_t)&box,(uintptr_t)result,pitch,2*pitch}};
        if(!setjmp(refusal))nvnTextureReadTexelsStrided(&request);
        assert(!memory_lock && !texture_lock && captures==frees);
        if(which<=RAW_R8) {
            assert(!blocked && !legacy && waits==1 && fill_waits==1 && uploads==1 && destroyed==1 &&
                destination_holds==1 && writes==(which!=RAW_R8) && raw_reads==2 && earliest_upload==2);
            for(unsigned y=0;y<2;y++) {
                const unsigned source=(5+4*y)*pixel;
                assert(!memcmp(result+y*pitch,(which==RAW_R8?raw_bytes:pixels)+source,pixel));
                assert(!memcmp(result+y*pitch+pixel,raw_bytes+source+pixel,pixel));
            }
        } else {
            assert(blocked==(which!=MIXED_UNCERTAIN&&which!=MIXED_ABSENT) && legacy==(which==MIXED_ABSENT));
            if(which==MIXED_UNCERTAIN) {
                assert(late_owner && late_release && !destroyed && borrowed_raw && origin_holds==1);
                late_release(late_owner);
                late_release=NULL;
                late_owner=NULL;
            }
            if(which==MIXED_REMAP||which==MIXED_CUT_MOVED||which==MIXED_UNACCEPTED||which==MIXED_ABSENT)
                assert(!writes && !raw_reads && !waits && !restore_attempts && !destination_holds);
            if(which==MIXED_RETIRED) assert(waits==1 && !uploads && !writes && !raw_reads);
        }
        for(unsigned i=0;i<sizeof result;i++)
            if(which>RAW_R8 || (i>=2*pixel && i<pitch) || i>=pitch+2*pixel) assert(result[i]==0xa6);
        assert(!borrowed_raw && !origin_holds && staged==destroyed);
        rc_registry_destroy(registry);
    }
    puts("public NVN ReadTexels: cross-provider R8 and RGBA8 rows, raw control, resident mixed coverage, stale cutoff, remaps, protection, and uncertain custody passed");
    puts("public NVN ReadTexels: queued producer drain, issued-only wait, exact strided bytes/guards, remaps, cancellation, protection, staging failure, shutdown, diagnostics and legacy routing passed");
}

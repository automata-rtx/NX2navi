#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "resource_core_registry.h"

#define OK(expr) assert((expr) == RC_REGISTRY_OK)
#define RAW_BUFFER ((VkBuffer)(uintptr_t)19)
#define MIXED_BUFFER ((VkBuffer)(uintptr_t)20)
#define TARGET ((VkImage)(uintptr_t)22)
enum { W_FORMAT, W_REGION, W_BUFFER, VK_GAP_COPY_B2T, VK_MUSE_COPY_SRC };
enum { VK_MEM_MIRROR_IMPORT, VK_MCLS_DEVICE, IMAGE_ALIAS_MAX_WRITE_FRAGMENTS = 16, VK_TEXTURE_ALIAS_MAX_BYTES = 16384 };
enum { OP_COPY_B2T = 1 };
typedef struct {
    VkImage image;
    VkFormat format;
    VkImageAspectFlags aspect;
    VkImageType type;
    VkImageUsageFlags usage;
    uint32_t width, height, depth, levels, layers, samples;
    int scale;
    uint64_t addr;
} VkTexImage;
typedef struct { struct { int format; } d; } tex_t;
typedef struct { int set_levels, base_level, set_layers, min_layer, set_format, format; } view_t;
typedef struct { int xoffset, yoffset, zoffset, width, height, depth; } NVNcopyRegion;
typedef struct { uint64_t src; const tex_t *tex; view_t view; NVNcopyRegion r; int has_view; } p_copy_b2t;
typedef struct { VkBuffer buffer; uint64_t offset, avail; int cls; } VkMemSpan;
typedef struct { uint64_t mapping; } VkMemSpanCapture;
typedef struct { unsigned live; } VkTextureWriteTarget;
typedef struct {
    int64_t copy_stride;
    int recorded;
    uint64_t fence;
    VkCommandBuffer cb;
} VkExec;
typedef struct { VkFormat vk, ideal; int expand_rgb, astc_bw; } TexVkFormat;
typedef struct { RcRegistryNativeImageFailure coverage; } VkTextureAliasFailure;
typedef struct { uint32_t fragment; } Slice;
typedef struct { RcRegistryBufferSpan span; } VkMemImageAliasBaseline;
typedef struct VkTextureAliasRead {
    uint32_t owners, mapping_count, demand_count, fragment_count;
    RcRegistryPosition read;
    struct { uint64_t addr; } result;
    uint64_t guest_bytes, workspace_bytes, stage_bytes, upload_offset;
    VkMemSpan raw_source;
    int raw_output, sparse_buffer_rows;
    RcInterval *receiver_mapping, *demands;
    RcRegistryNativeImageSnapshotFragment *fragments, *current;
    Slice *slices;
    VkMemImageAliasBaseline *baseline;
} VkTextureAliasRead;
typedef enum { VK_EXEC_RESTORE_READY, VK_EXEC_RESTORE_STOPPED, VK_EXEC_RESTORE_STALE, VK_EXEC_RESTORE_FAILED } VkExecRestoreResult;
/* TRANSFER TYPES */
int vk_exec_buffer_to_texture_source_current(const VkExecBufferToTexture *);

static const VkTexImage actual = {.image=TARGET,.format=VK_FORMAT_R8_UNORM,.aspect=VK_IMAGE_ASPECT_COLOR_BIT,
    .type=VK_IMAGE_TYPE_3D,.usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT,.width=8,.height=4,.depth=4,
    .levels=1,.layers=1,.samples=1,.scale=1,.addr=0xc000};
static const tex_t volume = {{1}};
static uint8_t destination[4][4][8], staging[12];
static const uint8_t raw[] = {1,2,3,4,5,6,7,8,9,10,11,12};
static const uint8_t authored[] = {41,42,43,44};
static RcRegistry *registry;
static uint64_t mapping, hull;
static unsigned declared, warnings, refusals, stopped, issued, mixed_issues, selects, releases, checks,
                reads, owner_release, native_recordings, mixed_cancels, writes, later_offset;
static VkMemSpanCapture span_hold;
static VkTextureAliasRead *issued_alias;
static unsigned vk_lost;

#define rc_registry_runtime() registry
static RcRegistryPosition vk_exec_resource_position(void) { return (RcRegistryPosition){1,4}; }
static uint32_t max1(uint32_t n) { return n ? n : 1; }
static int tex_vk_format(int format, TexVkFormat *out) {
    assert(format==1);
    *out=(TexVkFormat){.vk=VK_FORMAT_R8_UNORM,.ideal=VK_FORMAT_R8_UNORM};
    return 1;
}
static int nvn_fmt_bpb(int format,int *bw,int *bh) {
    assert(format==1);
    *bw=*bh=1;
    return 1;
}
static void warn_once(int kind,const char *fmt,...) { assert(kind>=0&&fmt); warnings++; }
static void rt_log(const char *fmt,...) { assert(fmt); }
static int vk_texture_deferred(void) { return 0; }
static int vk_sync_stopping(void) { return stopped; }
static int vk_device_is_lost(void) { return vk_lost; }
static int vk_exec_recording_failed(void) { return refusals!=0; }
static void vk_exec_required_graphics_unavailable(VkExec *ex,int opcode,const char *reason) {
    assert(ex&&opcode==OP_COPY_B2T&&reason&&declared);
    ++refusals;
}
static const VkTexImage *vk_texture_get_transfer_write(const tex_t *target) {
    assert(target==&volume&&declared);
    return &actual;
}
static const VkTexImage *vk_texture_get_alias_transfer_write(const tex_t *target) { (void)target; abort(); }
static const VkTexImage *vk_texture_get(const tex_t *target) { (void)target; abort(); }
static int vk_texture_transfer_write_geometry(const tex_t *target,VkTexImage *out) {
    assert(target==&volume&&!declared);
    *out=actual;
    out->image=VK_NULL_HANDLE;
    return 1;
}
static int vk_texture_alias_region_transfer(const tex_t *target,const view_t *view,const NVNcopyRegion *box) {
    assert(target==&volume&&!view&&box->zoffset==2&&box->depth==1);
    return 0;
}
static int vk_mem_lookup(uint64_t address,uint64_t size,VkMemSpan *out) {
    assert(address==0x9000&&size==hull&&!declared);
    *out=(VkMemSpan){RAW_BUFFER,64,size,VK_MCLS_DEVICE};
    return 1;
}
static RcRegistryResult vk_mem_span_prepare_native_input(uint64_t address,uint64_t bytes,int clamp,int policy,
    VkMemSpan *out,VkMemSpanCapture **hold) {
    assert(!clamp&&policy==VK_MEM_MIRROR_IMPORT&&!*hold&&!span_hold.mapping);
    assert(vk_mem_lookup(address,bytes,out));
    selects++;
    span_hold.mapping=mapping;
    *hold=&span_hold;
    return RC_REGISTRY_OK;
}
static int vk_mem_span_capture_current(const VkMemSpanCapture *hold,const VkMemSpan *span) {
    return hold==&span_hold && hold->mapping==mapping && span->buffer==RAW_BUFFER && span->offset==64;
}
static int vk_mem_span_record_input(VkMemSpanCapture *hold,const VkMemSpan *span) {
    ++checks;
    return vk_mem_span_capture_current(hold,span);
}
static void vk_mem_span_capture_release(VkMemSpanCapture **hold) {
    if (!*hold) return;
    assert(*hold==&span_hold&&span_hold.mapping);
    span_hold.mapping=0;
    *hold=NULL;
    releases++;
}
static uint32_t vk_mem_span_capture_device_intervals(const VkMemSpanCapture *hold,const VkMemSpan *span,
    uint64_t address,uint64_t bytes,RcInterval **out) {
    assert(vk_mem_span_capture_current(hold,span)&&address==0x9000&&bytes==hull);
    *out=malloc(sizeof **out);
    assert(*out);
    **out=(RcInterval){address,mapping,{9,64,bytes}};
    return 1;
}
static RcRegistryResult vk_mem_image_alias_baseline_capture(uint64_t address,uint64_t bytes,const RcInterval *maps,
    uint32_t count,RcRegistryPosition read,VkMemImageAliasBaseline **out) {
    assert(count==1&&maps->address==address&&maps->mapping==mapping&&maps->range.size==bytes);
    assert(!rc_registry_position_compare(read,vk_exec_resource_position()));
    *out=malloc(sizeof **out);
    assert(*out);
    **out=(VkMemImageAliasBaseline){.span={*maps,{19,1,64}}};
    return RC_REGISTRY_OK;
}
static const RcRegistryBufferSpan *vk_mem_image_alias_baseline_spans(VkMemImageAliasBaseline *baseline,uint32_t *count) {
    *count=1;
    return &baseline->span;
}
static void lock(void) {}
static void unlock(void) {}
static RcRegistryResult image_alias_capture_sources(VkTextureAliasRead *alias,int retain,VkTextureAliasFailure *failure) {
    assert(retain&&failure&&alias->raw_output&&alias->fragment_count>=2);
    return RC_REGISTRY_OK;
}
static void image_alias_release(VkTextureAliasRead *alias) {
    if (--alias->owners) return;
    free(alias->receiver_mapping);
    free(alias->demands);
    free(alias->fragments);
    free(alias->current);
    free(alias->slices);
    free(alias->baseline);
    free(alias);
    ++owner_release;
}
static int vk_texture_alias_read_current(VkTextureAliasRead *alias) {
    return alias->receiver_mapping[0].mapping==mapping;
}
static VkExecRestoreResult vk_exec_prepare_alias_readback(VkExec *ex,VkTextureAliasRead *const *aliases,uint32_t count,
    int (*current)(void *),void *scope) {
    assert(ex&&count==1&&aliases[0]->owners==1&&!declared);
    assert(current(scope));
    ++reads;
    return stopped ? VK_EXEC_RESTORE_STOPPED : VK_EXEC_RESTORE_READY;
}
static int vk_texture_alias_buffer_source(VkTextureAliasRead *alias,VkBuffer *buffer,VkDeviceSize *offset) {
    assert(vk_texture_alias_read_current(alias) && !declared);
    const RcRegistryNativeImageSnapshotFragment *f=alias->fragments;
    memset(staging,0,sizeof staging);
    for (uint32_t i=0;i<alias->fragment_count;i++) {
        const RcRegistryNativeImageFragment *piece=&f[i].fragment;
        uint64_t to=piece->interval.range.offset-64;
        assert(to<=hull&&piece->interval.range.size<=hull-to);
        if (!piece->binding.image) {
            assert(f[i].native_buffer.allocation==19 && f[i].native_buffer.offset-64==to);
            memcpy(staging+to,raw+to,piece->interval.range.size);
        } else {
            assert(piece->binding.image==10 && piece->output.ordinal==3 &&
                   piece->interval.range.offset>=later_offset);
            memcpy(staging+to,authored+(piece->interval.range.offset-later_offset),piece->interval.range.size);
        }
    }
    *buffer=MIXED_BUFFER;
    *offset=32;
    return 1;
}
static void vk_texture_alias_read_cancel(VkTextureAliasRead *alias) {
    if (!alias) return;
    ++mixed_cancels;
    image_alias_release(alias);
}
static void vk_texture_alias_read_use(VkTextureAliasRead *alias,uint64_t fence) {
    assert(alias&&alias->owners==1&&fence==1&&declared&&!issued_alias);
    ++alias->owners;
    issued_alias=alias;
}
static void vk_texture_write_target_cancel(VkTextureWriteTarget *target) { assert(!target); }
static void vk_exec_record_native_input_provider(VkExec *ex,VkMemSpanCapture **hold) {
    assert(ex&&declared&&*hold==&span_hold);
    ++native_recordings;
    span_hold.mapping=0;
    *hold=NULL;
}
#define RESOURCE_IMAGE_SCOPE(owner,fence) assert((fence)==1&&declared)
static void vk_contact_flush(VkExec *ex) { assert(ex&&declared); }
static void vk_gpu_gap_note(int kind,uint64_t area) { assert(kind==VK_GAP_COPY_B2T&&area==8); }
static void vk_mem_count(int cls,int use,uint64_t bytes) { assert(cls==VK_MCLS_DEVICE&&use==VK_MUSE_COPY_SRC&&bytes==hull); }
static void vk_exec_pass_end(VkExec *ex) { assert(ex&&declared); }
static void vk_exec_barrier(VkExec *ex) { assert(ex&&declared); }
static void vk_dlaa_hud_write(VkImage image) { assert(image==TARGET); }
static void resource_image_written(uint64_t image,uint64_t address) { assert(image==22&&address==0xc000); }
static void vk_texture_gpu_written(const tex_t *target) { assert(target==&volume); }
static void vk_texture_alias_transfer_written(const tex_t *target,VkImage image,const view_t *view,const NVNcopyRegion *r) {
    assert(target==&volume&&image==TARGET&&!view&&r->zoffset==2&&r->depth==1);
    ++writes;
}
static void vkCmdCopyBufferToImage(VkCommandBuffer cb,VkBuffer input,VkImage output,VkImageLayout layout,
    uint32_t count,const VkBufferImageCopy *r) {
    assert(cb&&declared&&output==TARGET&&layout==VK_IMAGE_LAYOUT_GENERAL&&count==1);
    assert(r->imageSubresource.aspectMask==VK_IMAGE_ASPECT_COLOR_BIT&&r->imageSubresource.baseArrayLayer==0&&
           r->imageSubresource.layerCount==1&&r->imageSubresource.mipLevel==0);
    assert(r->imageOffset.x==2&&r->imageOffset.y==1&&r->imageOffset.z==2);
    assert(r->imageExtent.width==4&&r->imageExtent.height==2&&r->imageExtent.depth==1&&r->bufferImageHeight==2);
    const uint8_t *bytes;
    if (input==RAW_BUFFER) { assert(r->bufferOffset==64); bytes=raw; }
    else { assert(input==MIXED_BUFFER&&r->bufferOffset==32&&issued_alias); bytes=staging; ++mixed_issues; }
    for (unsigned y=0;y<2;y++) memcpy(destination[2][1+y]+2,bytes+y*r->bufferRowLength,4);
    ++issued;
}
/* TRANSFER FUNCTIONS */

static void prior_image(uint64_t offset,int current) {
    const RcRegistryPosition output={1,current ? 3 : 5};
    const RcInterval image={0x1000,23,{9,64,hull}};
    const RcSourceRange bytes={9,offset,4};
    RcRegistryWrite hold, producer;
    OK(rc_registry_image_hold_begin_extents(registry,10,&image,1,&hold));
    OK(rc_registry_write_begin_at(registry,image.range,RC_REGISTRY_IMAGE,output,&producer));
    OK(rc_registry_image_producer_add(registry,producer,output));
    if (!current) return; // A later pending output is outside the selected read.
    OK(rc_registry_declare_through(registry,vk_exec_resource_position()));
    const RcRegistryNativeImageWrite named={{.interval=image,.hold=hold},&bytes,1};
    OK(rc_registry_native_image_write_fragments(registry,10,&named,1,output));
    OK(rc_registry_image_producer_resolve(registry,producer,output,&(RcRegistryImageBinding){10,hold},0));
}
enum Scenario { RAW, TIGHT_MIXED, PADDED_MIXED, PADDED_NONDEMAND, LATER, REMAP_BEFORE_PREPARE, REMAP_BEFORE_ISSUE, STOPPED };
static void dispatch(enum Scenario scenario) {
    const int expected_issue = scenario < REMAP_BEFORE_PREPARE;
    const int expected_mix = scenario==TIGHT_MIXED || scenario==PADDED_MIXED;
    const int has_image = scenario!=RAW;
    const unsigned stride=scenario==TIGHT_MIXED ? 4 : 8;
    hull=stride+4;
    mapping=27;
    declared=refusals=stopped=0;
    memset(destination,0xa7,sizeof destination);
    registry=rc_registry_create((RcRegistryConfig){16,16,4});
    assert(registry);
    later_offset=scenario==PADDED_NONDEMAND ? 68 : 64+stride;
    if (has_image) prior_image(later_offset,scenario!=LATER);
    if (!has_image || scenario==LATER) OK(rc_registry_declare_through(registry,vk_exec_resource_position()));
    VkExec ex={.fence=1,.cb=(VkCommandBuffer)(uintptr_t)1,.copy_stride=stride};
    const p_copy_b2t op={.src=0x9000,.tex=&volume,.r={2,1,2,4,2,1}};
    VkExecBufferToTexture copy={0};
    assert(vk_exec_buffer_to_texture_capture_compat(&ex,&op,&copy)==1&&copy.source_hold&&!copy.target);
    if (scenario==REMAP_BEFORE_PREPARE) ++mapping;
    if (scenario==STOPPED) stopped=1;
    const RcRegistryResult result=vk_exec_buffer_to_texture_prepare_source(&ex,&copy);
    if (scenario==REMAP_BEFORE_PREPARE) assert(result==RC_REGISTRY_STALE);
    else if (scenario==STOPPED) assert(result==RC_REGISTRY_BUSY);
    else {
        assert(result==RC_REGISTRY_OK);
        if (scenario==REMAP_BEFORE_ISSUE) ++mapping;
        declared=1;
        vk_exec_buffer_to_texture_commit_compat(&ex,&op,&copy);
        assert(refusals==!expected_issue);
    }
    assert(ex.recorded==expected_issue);
    vk_exec_buffer_to_texture_cancel(&copy);
    assert(!copy.source_hold&&!copy.source_alias&&!span_hold.mapping);
    if (issued_alias) {
        assert(expected_mix && issued_alias->owners==1);
        image_alias_release(issued_alias);
        issued_alias=NULL;
    }
    for (unsigned z=0;z<4;z++) for (unsigned y=0;y<4;y++) for (unsigned x=0;x<8;x++) {
        uint8_t want=0xa7;
        if (expected_issue&&z==2&&y>=1&&y<=2&&x>=2&&x<=5)
            want=expected_mix&&y==2 ? authored[x-2] : raw[(y-1)*stride+x-2];
        assert(destination[z][y][x]==want);
    }
    rc_registry_destroy(registry);
}
int main(void) {
    const RcInterval split[]={{100,1,{9,40,2}},{102,2,{10,16,14}}};
    RcInterval demand[3];
    assert(image_alias_buffer_rows(split,2,100,12,4,8,demand,3)==3);
    assert(demand[0].address==100&&demand[0].range.size==2&&demand[1].range.size==2);
    assert(demand[2].address==108&&demand[2].mapping==2&&demand[2].range.offset==22&&demand[2].range.size==4);
    assert(!image_alias_buffer_rows(split,2,100,12,4,8,demand,2));
    for (enum Scenario scenario=RAW;scenario<=STOPPED;scenario++) dispatch(scenario);
    assert(issued==5 && writes==5 && raw[4]==5 && mixed_issues==2 && reads==4 &&
           selects==8 && native_recordings==5 && releases==3 && mixed_cancels==5 && checks==10 && owner_release==6 && !warnings);
    puts("B2T original source: raw, mixed 3D slice, exact pitched rows, padding-only and later writes, remap, stop and retained disposal passed");
}

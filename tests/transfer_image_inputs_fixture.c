#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define VK_NO_PROTOTYPES
#include "tex.h"
#include "vk.h"
#include "vk_exec.h"
#include "vk_texture_alias_bytes.h"
#include "resource_command_effects.h"
#include "resource_host_write.h"

#define OLD ((VkImage)(uintptr_t)10)
#define NEW ((VkImage)(uintptr_t)14)
#define REBUILT ((VkImage)(uintptr_t)22)
#define DST ((VkImage)(uintptr_t)31)
#define BUFFER ((VkBuffer)(uintptr_t)75)
#define OUTPUT UINT64_C(0x20000)
#define BASE (RT_GPU_VA_BASE+UINT64_C(0x1000))
#define OFFSET UINT64_C(0x1000)
#define OK(e) assert((e)==RC_REGISTRY_OK)
enum { W_FORMAT, W_REGION, W_COPY, IMAGE_ALIAS_MAX_WRITE_FRAGMENTS = 8192 };
typedef struct { unsigned flags, type, format, w,h,d,levels,layers,samples,usage; } ImgKey;
typedef struct { uintptr_t addr; } TexKey;
typedef struct {
    const tex_t *texture;
    TexKey key;
    ImgKey placement;
    VkTexImage image;
    const RcInterval *mapping, *transfer_demand;
    const VkImageSubresourceRange *load_planes;
    uint32_t mapping_count, transfer_demand_count;
    int allow_submitted_raw;
} ImageAliasReceiver;
struct VkTextureAliasRead { int current; uint64_t pending; };
struct VkTexturePresentSource { VkImage selected; };
static RcRegistry *registry;
static RcRegistryPosition read = {1,8};
static tex_t origin, target;
static VkTexImage resident, destination;
static uint8_t before[64], newer[64], combined[64], outbytes[64], raw[64];
static unsigned warnings, uses, assemblies, native_commits, buffer_commits, buffer_effects, handoffs, failures;
static uint8_t pitched[28];
static int stale_destination, stale_location;
static VkExecTextureBufferWrite *buffer_tail;
static int locked, remap;
static struct VkTexturePresentSource hold;
static struct VkTextureAliasRead assembled;

#define rc_registry_runtime fixture_registry
static RcRegistry *fixture_registry(void) { return registry; }
void rt_fatal(const char *message, ...) { (void)message; abort(); }
RcRegistryPosition vk_exec_resource_position(void) { return read; }
static int max1(int v) { return v > 0 ? v : 1; }
static void lock(void) { ++locked; }
static void unlock(void) { assert(locked); --locked; }
static void make_key(const tex_t *t, TexKey *k) { k->addr=(uintptr_t)t->mem; }
static void warn_once(int group, const char *msg, ...) { assert(group>=0&&msg); ++warnings; }
int vk_texture_deferred(void) { return 0; }
static int image_alias_refuse(VkTextureAliasFailure *f, int phase, RcRegistryResult s, RcInterval i) {
    *f=(VkTextureAliasFailure){.phase=phase,.status=s,.receiver=i};
    return -1;
}
static int image_alias_refuse_layout(VkTextureAliasFailure *f,RcInterval i,const char *reason) {
    f->layout_gate=reason;
    return image_alias_refuse(f,VK_TEXTURE_ALIAS_LAYOUT,RC_REGISTRY_UNSUPPORTED,i);
}
int tex_vk_format(int f,TexVkFormat *out) {
    assert(f==NVN_FORMAT_R8);
    *out=(TexVkFormat){.vk=VK_FORMAT_R8_UNORM,.ideal=VK_FORMAT_R8_UNORM};
    return 1;
}
int nvn_fmt_bpb(int f,int *width,int *height) {
    assert(f==NVN_FORMAT_R8);
    if(width) *width=1;
    if(height) *height=1;
    return 1;
}
static int vk_mem_lookup_stub(uint64_t address,uint64_t size,VkMemSpan *out) {
    assert(address==BASE && size==origin.L.size && out);
    return !remap;
}
static unsigned vk_mem_device_intervals_stub(VkMemSpan *span,uint64_t base,uint64_t length,RcInterval **mapped) {
    assert(span && base==BASE && length<=64 && mapped);
    *mapped=malloc(sizeof **mapped);
    **mapped=(RcInterval){base,27,{9,OFFSET,length}};
    return 1;
}
#define vk_mem_lookup vk_mem_lookup_stub
#define vk_mem_device_intervals vk_mem_device_intervals_stub
static int image_alias_guest_mapping_current(uint64_t address,uint64_t bytes,const RcInterval *m,uint32_t count) {
    assert(address==BASE && bytes==origin.L.size && m && count==1);
    return !remap && m->address==address && m->mapping==27 && m->range.offset==OFFSET;
}
int vk_texture_transfer_source_geometry(const tex_t *t,VkTexImage *out) {
    assert(t==&origin); *out=resident; return 1;
}
int vk_texture_transfer_write_geometry(const tex_t *t,VkTexImage *out) {
    assert(t==&target); *out=destination; return 1;
}
VkTexturePresentSource *vk_texture_transfer_read_capture(const tex_t *t,VkTexImage *shape) {
    assert(t==&origin&&locked); hold.selected=resident.image; *shape=resident; return &hold;
}
int vk_texture_transfer_read_current(const VkTexturePresentSource *h,VkImage im) {
    return h && h->selected==im && im==resident.image;
}
int vk_texture_record_selected_sample_locked(const VkTexturePresentSource *h,VkImage im) {
    assert(locked&&vk_texture_transfer_read_current(h,im)); ++uses; return 1;
}
void vk_texture_present_release(VkTexturePresentSource *h) {
    if(h) { assert(h==&hold); h->selected=0; }
}
void vk_texture_alias_read_cancel(VkTextureAliasRead *a) { if(a) { assert(a==&assembled); a->current=0; } }
int vk_texture_alias_read_current(VkTextureAliasRead *a) { return a==&assembled && a->current && !remap && !a->pending; }
int vk_texture_alias_read_raw_progress(VkTextureAliasRead *a,RcRegistryPosition at,uint64_t *upload) {
    if(a!=&assembled || !a->current || remap || rc_registry_position_compare(at,read)) return 0;
    *upload=a->pending;
    return 1;
}
const VkTexImage *vk_texture_get(const tex_t *t) { assert(t==&origin); return &resident; }
const VkTexImage *vk_texture_get_transfer_write(const tex_t *t) { assert(t==&target); return &destination; }
const VkTexImage *vk_texture_get_alias_transfer_write(const tex_t *t) { assert(t==&target); return &destination; }
int vk_texture_alias_region_transfer(const tex_t *t,const view_t *view,const NVNcopyRegion *region) {
    assert(t==&target&&region); (void)view; return 0;
}
void vk_exec_required_graphics_unavailable(VkExec *ex,uint32_t op,const char *reason) {
    assert(ex && (op==OP_COPY_T2T || op==OP_COPY_T2B) && reason); ++failures;
}
void vk_gpu_gap_note(int cat,uint64_t count) { assert((cat==VK_GAP_COPY_T2T || cat==VK_GAP_COPY_T2B) && count); }
void vk_dlaa_hud_write(VkImage image) { assert(image==DST); }
void vk_exec_pass_end(VkExec *ex) { assert(ex); }
void vk_exec_barrier(VkExec *ex) { assert(ex); }
void vk_texture_explicit_read_use(const tex_t *tex,VkImage image) { assert(tex==&origin&&image==OLD); }
void resource_image_written(uint64_t im,uint64_t adr) { assert(im==(uintptr_t)DST&&adr==destination.addr); }
void vk_texture_gpu_written(const tex_t *t) { assert(t==&target); }
void vk_texture_alias_transfer_written(const tex_t *t,VkImage i,const view_t *v,const NVNcopyRegion *r) {
    (void)t;(void)i;(void)v;(void)r;assert(0);
}
int rt_knob(const char *name,const char *value) { assert(name&&value); return 0; }
void rt_log(const char *s,...) { assert(s); }

static void copy_image(VkCommandBuffer cb,VkImage source,VkImageLayout sl,VkImage dest,VkImageLayout dl,
                       uint32_t count,const VkImageCopy *r) {
    assert(cb&&sl==VK_IMAGE_LAYOUT_GENERAL&&dl==VK_IMAGE_LAYOUT_GENERAL&&count==1&&dest==DST);
    assert(source==OLD||source==REBUILT);
    const uint8_t *input=source==OLD ? before : combined;
    assert(r->srcOffset.x==1 && r->srcOffset.y==1 && r->dstOffset.x==1 && r->dstOffset.y==1);
    assert(r->extent.width==4&&r->extent.height==2&&r->extent.depth==1);
    for (uint32_t y=0;y<r->extent.height;y++)
        memcpy(outbytes+(r->dstOffset.z*4+r->dstOffset.y+y)*8+1,
               input+(r->srcOffset.z*4+r->srcOffset.y+y)*8+1,4);
    native_commits++;
}
#undef vkCmdCopyImage
#undef vkCmdResolveImage
#undef vkCmdBlitImage
#define vkCmdCopyImage copy_image
#define vkCmdResolveImage unexpected_resolve
#define vkCmdBlitImage unexpected_blit
static PFN_vkCmdResolveImage unexpected_resolve;
static PFN_vkCmdBlitImage unexpected_blit;

// Fixture-specific Vulkan staging. Provider choice and composition use the real registry and byte engine.
static int image_alias_read_prepare_locked(const ImageAliasReceiver *r,const void *entry,VkTextureAliasRead **alias,
    VkTexImage *shape,int force,int target_entry,VkTextureAliasFailure *failure) {
    assert(locked && !entry && force && !target_entry && failure);
    assert(r->transfer_demand_count==2 && r->mapping_count==1 && r->load_planes->levelCount==1);
    RcRegistryBufferSpan baseline={r->mapping[0],{54,1,0}};
    RcRegistryNativeImageSnapshotFragment fragments[8];
    uint32_t count;
    OK(rc_registry_native_image_snapshot_mixed(registry,r->transfer_demand,r->transfer_demand_count,&baseline,1,
                                               read,fragments,8,&count,NULL));
    VkTextureAliasSlice slices[8];
    const VkTextureAliasBytes bytes[]={{before,BASE,64},{newer,BASE,64},{raw,0,64}};
    for(uint32_t i=0;i<count;i++) {
        const RcRegistryNativeImageSnapshotFragment *f=&fragments[i];
        const RcRegistryNativeImageFragment *g=&f->fragment;
        const unsigned which=f->native_buffer.allocation ? 2 : g->binding.image==(uintptr_t)NEW ? 1 : 0;
        const uint64_t at=which==2 ? f->native_buffer.offset : g->interval.address;
        slices[i]=(VkTextureAliasSlice){g->interval.range,at,which,g->demand};
    }
    memset(combined,0,sizeof combined);
    assert(texture_alias_compose(bytes,3,combined,BASE,r->mapping[0].range.size,
                                 r->transfer_demand,r->transfer_demand_count,slices,count));
    *shape=r->image;
    shape->image=REBUILT;
    assembled.current=1;
    *alias=&assembled;
    ++assemblies;
    return 1;
}

// Linear row/slice coordinates for a small lossless R8 test texture.
int nvn_texels_box_ranges(const nvn_texdesc *d,unsigned level,unsigned layer,uint32_t x,uint32_t y,uint32_t z,
                          uint32_t w,uint32_t h,void (*range)(void*,uint64_t,uint64_t),void *context) {
    assert(d==&origin.d&&level==0&&layer==0);
    if(x+w>8 || y+h>4 || z>(d->target==NVN_TEXTURE_TARGET_3D)) return -1;
    for(uint32_t row=0;row<h;row++) range(context,(z*4+y+row)*8+x,w);
    return 0;
}
/* BUFFER_TYPES */
struct VkMemSpanCapture { int selected; };
static struct VkMemSpanCapture buffer_hold = {1};
static const struct MemBackingCapture *const output_map = (void *)(uintptr_t)7;

static void copy_image_to_buffer(VkCommandBuffer cb,VkImage source,VkImageLayout layout,VkBuffer dest,
                                 uint32_t count,const VkBufferImageCopy *r) {
    assert(cb && layout==VK_IMAGE_LAYOUT_GENERAL && dest==BUFFER && count==1);
    assert(source==OLD || source==REBUILT);
    assert(r->bufferOffset==4 && r->bufferRowLength==7 && r->bufferImageHeight==0);
    assert(r->imageExtent.width==4 && r->imageExtent.height==2 && r->imageExtent.depth==1);
    const uint8_t *bytes=source==OLD ? before : combined;
    for(uint32_t y=0;y<r->imageExtent.height;y++)
        memcpy(pitched+r->bufferOffset+(uint64_t)y*r->bufferRowLength,
            bytes+((r->imageOffset.z*4+r->imageOffset.y+y)*8+r->imageOffset.x),4);
    buffer_commits++;
}
#undef vkCmdCopyImageToBuffer
#define vkCmdCopyImageToBuffer copy_image_to_buffer

RcRegistryResult vk_mem_canonical_source_qualify(const struct MemBackingCapture *cap,uint64_t address,uint64_t size) {
    assert(cap==output_map && address==OUTPUT && size==11);
    return stale_destination ? RC_REGISTRY_STALE : RC_REGISTRY_OK;
}
int vk_mem_span_capture_current(const VkMemSpanCapture *cap,const VkMemSpan *span) {
    assert(cap==&buffer_hold && span->buffer==BUFFER && span->offset==4);
    return !stale_destination;
}
int vk_mem_span_buffer_location(const VkMemSpanCapture *cap,const VkMemSpan *span,RcRegistryBufferLocation *location) {
    assert(cap==&buffer_hold && span->buffer==BUFFER && span->offset==4);
    *location=(RcRegistryBufferLocation){81, stale_location ? 2u : 1u,4};
    return 1;
}
int resource_effects_buffer_prepare_intervals(const RcInterval *m,uint32_t n,int visible) {
    assert(m && n==1 && m->address==OUTPUT && m->range.size==11 && !visible);
    return 1;
}
void resource_effects_buffer_prepare_native(const RcInterval *m,uint32_t n,uint64_t address,RcRegistryBufferLocation location) {
    assert(m && n==1 && address==OUTPUT && location.allocation==81 && location.incarnation==1 && location.offset==4);
    buffer_effects++;
}
void resource_effects_buffers_commit(void) { assert(buffer_effects==buffer_commits); }
int vk_exec_recording_failed(void) { return 0; }
void vk_contact_flush(VkExec *ex) { assert(ex); }
void vk_texture_restore_use(const VkTextureRestore *restore) { (void)restore; abort(); }
int vk_texture_transfer_read_use(const VkTexturePresentSource *source,VkImage image) {
    (void)source; (void)image; abort(); // This selected lane takes its use from the exact input receipt.
}
void vk_usnap_drop(uint64_t address,uint64_t size) { assert(address==OUTPUT && size==11); }
void vk_mem_gpu_pending_span(const VkMemSpan *span,uint64_t address,uint64_t size) {
    (void)span; (void)address; (void)size; abort();
}
static void texture_buffer_writeback(const VkExecTextureBufferWrite *write) { assert(write==buffer_tail); }
static void texture_buffer_write_done(void *opaque) { (void)opaque; abort(); }
static void texture_buffer_write_cancel(void *opaque,int maybe) { (void)opaque; (void)maybe; abort(); }
void vk_completion_add_owned(uint64_t fence,void (*done)(void *),void (*cancel)(void *,int),void *opaque) {
    assert(fence==1 && done==texture_buffer_write_done && cancel==texture_buffer_write_cancel && opaque==buffer_tail);
    handoffs++;
}
void vk_mem_count(int cls,int use,uint64_t bytes) {
    (void)cls; assert(use==VK_MUSE_GPU_WRITE && bytes==8);
}

/* EXTRACT */

static void assert_overlap_geometry(const VkTextureTransferInput *input) {
    VkTexImage same=input->image, distinct=same;
    distinct.image=NEW;  // The normalized address is identical. The Vulkan allocation is different.
    const Sub src={.img=&same,.aspect=VK_IMAGE_ASPECT_COLOR_BIT,.w=4,.h=2,.nz=1},
              dst={.img=&same,.aspect=VK_IMAGE_ASPECT_COLOR_BIT,.x=2,.w=4,.h=2,.nz=1};
    VkExecTextureToTexture copy={.op=VK_EXEC_IMAGE_COPY,
        .source_layers={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.source_extent={4,2,1}};
    image_copy_prepare_overlap(&copy,&src,&dst,&same);
    assert(copy.scratch_recipe.sType==VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO &&
        copy.scratch_recipe.extent.width==4 && copy.scratch_recipe.extent.height==2);
    copy.scratch_recipe=(VkImageCreateInfo){0};
    Sub other=dst;
    other.img=&distinct;
    image_copy_prepare_overlap(&copy,&src,&other,&distinct);
    assert(!copy.scratch_recipe.sType);
}

static void emit_buffer(const p_copy_b2t *op,VkTextureTransferInput *input,int replaced,int refuse) {
    RcInterval mapping={OUTPUT,32,{71,256,11}};
    VkExecTextureBufferWrite write={.image=input->image.image,.hold=&buffer_hold,
        .address=OUTPUT,.location={81,1,4},.span={.buffer=BUFFER,.offset=4},.mapped=&mapping,.mapped_count=1,.gpu_only=1,
        .pitch={.row_bytes=4,.stride=7,.rows=2,.size=11},
        .region={.bufferOffset=4,.bufferRowLength=7,
                 .imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.imageOffset={1,1,op->r.zoffset},
                 .imageExtent={4,2,1}}};
    VkExecRestoreCopy copy={.input=*input,.size=11,.destination=(struct MemBackingCapture *)output_map,.output=&write};
    VkExec ex={.cb=(VkCommandBuffer)(uintptr_t)1,.fence=1};
    const unsigned buffers=buffer_commits, completed=handoffs;
    if(replaced) {
        assert(input->alias==&assembled);
        assembled.pending=11;
        assert(restore_copy_capture_current(op,&copy));
        assert(!vk_exec_restore_copy_current(op,&copy)); // Pending identity still isn't issue permission.
        stale_destination=1;
        assert(!restore_copy_capture_current(op,&copy));
        stale_destination=0;
        remap=1;
        assert(!restore_copy_capture_current(op,&copy));
        remap=0;
        assembled.pending=0; // The existing native prefix has finished this particular selected producer.
    }
    memset(pitched,0xa7,sizeof pitched);
    buffer_tail=&write;
    remap=refuse;
    vk_exec_restore_copy_commit(&ex,op,&copy);
    assert(ex.recorded==!refuse && buffer_commits==buffers+!refuse && handoffs==completed+!refuse);
    assert((copy.output==NULL)==!refuse);
    const uint8_t *source=replaced ? combined : before;
    for(unsigned i=0;i<sizeof pitched;i++) {
        const unsigned at=op->r.zoffset*32+9;
        uint8_t want=0xa7;
        if(!refuse && i>=4&&i<8) want=source[at+i-4];
        if(!refuse && i>=11&&i<15) want=source[at+8+i-11];
        assert(pitched[i]==want);
    }
    if(!refuse) {
        copy.output=&write;
        stale_location=1;
        vk_exec_restore_copy_commit(&ex,op,&copy);
        stale_location=0;
        assert(buffer_commits==buffers+1 && handoffs==completed+1);
    }
    remap=0;
    buffer_tail=NULL;
}


static void put_image(VkImage image,RcSourceRange extent,RcRegistryPosition output) {
    RcInterval owner={BASE,27,{9,OFFSET,origin.L.size}};
    RcRegistryWrite hold, author;
    OK(rc_registry_image_hold_begin_extents(registry,(uintptr_t)image,&owner,1,&hold));
    OK(rc_registry_write_begin_at(registry,owner.range,RC_REGISTRY_IMAGE,output,&author));
    OK(rc_registry_image_producer_add(registry,author,output));
    OK(rc_registry_declare_through(registry,output));
    const RcRegistryNativeImageWrite pieces={{.interval=owner,.hold=hold},&extent,1};
    OK(rc_registry_native_image_write_fragments(registry,(uintptr_t)image,&pieces,1,output));
    OK(rc_registry_image_producer_resolve(registry,author,output,&(RcRegistryImageBinding){(uintptr_t)image,hold},0));
}

static void transfer(int volume,int replaced,int need_refusal) {
    const unsigned z=!!volume;
    const uint64_t first=(z*4+1)*8+1;
    registry=rc_registry_create((RcRegistryConfig){32,32,8});
    assert(registry);
    origin=(tex_t){.mem=(void*)(uintptr_t)BASE,.serial=1,.L={.size=volume ? 64 : 32},
        .d={.target=volume ? NVN_TEXTURE_TARGET_3D : NVN_TEXTURE_TARGET_2D,
            .format=NVN_FORMAT_R8,.width=8,.height=4,.depth=volume ? 2 : 1,.levels=1}};
    resident=(VkTexImage){.image=OLD,.addr=BASE,.aspect=VK_IMAGE_ASPECT_COLOR_BIT,.scale=1,.levels=1,.layers=1,
        .type=volume ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,.format=VK_FORMAT_R8_UNORM,.samples=1,
        .width=8,.height=4,.depth=volume ? 2 : 1,.usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT,.gpu_owned=1};
    target=origin;
    target.mem=(void*)(uintptr_t)0x10000;
    target.serial=3;
    destination=resident;
    destination.image=DST;
    destination.addr=0x10000;
    destination.usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    for(unsigned i=0;i<64;i++) before[i]=i+1,newer[i]=201,raw[i]=77,outbytes[i]=0xa7;
    put_image(OLD,(RcSourceRange){9,OFFSET+(replaced ? first+8 : 0),replaced ? 2 : origin.L.size},(RcRegistryPosition){1,3});
    if(replaced) put_image(NEW,(RcSourceRange){9,OFFSET+first+1,2},(RcRegistryPosition){1,5});
    OK(rc_registry_declare_through(registry,read));

    p_copy_t2t op={.src=&origin,.dst=&target,.sr={1,1,z,4,2,1},.dr={1,1,z,4,2,1}};
    const p_copy_b2t t2b={.tex=&origin,.r=op.sr,.src=0x20000};
    VkTextureAliasFailure failure={0};
    VkTextureTransferInput input={0};
    assert(vk_exec_transfer_image_input_capture(OP_COPY_T2B,&t2b,&input,&failure)==1);
    assert(input.image.image==(replaced ? REBUILT : OLD));
    emit_buffer(&t2b,&input,replaced,need_refusal);
    if(!volume&&!replaced) assert_overlap_geometry(&input);
    vk_texture_transfer_input_cancel(&input);
    assert(vk_exec_transfer_image_input_capture(OP_COPY_T2T,&op,&input,&failure)==1);
    assert(input.image.image==(replaced ? REBUILT : OLD));
    VkExec ex={.cb=(VkCommandBuffer)(uintptr_t)1};
    const unsigned issued=native_commits;
    remap=need_refusal;
    vk_exec_texture_to_texture_compat(&ex,&op,&input);
    assert(native_commits==issued+!need_refusal);
    for(unsigned i=0;i<64;i++) {
        uint8_t expected=0xa7;
        if(!need_refusal) {
            if(i>=first&&i<first+4)
                expected=replaced ? (i==first+1||i==first+2 ? 201 : 77) : before[i];
            if(i>=first+8&&i<first+12) expected=replaced && i>=first+10 ? 77 : before[i];
        }
        assert(outbytes[i]==expected);
    }
    remap=0;
    vk_texture_transfer_input_cancel(&input);
    rc_registry_destroy(registry);
}

int main(void) {
    transfer(0,0,0);
    transfer(0,1,0);
    transfer(1,1,0);
    transfer(1,1,1);
    assert(buffer_commits==3 && buffer_effects==3 && handoffs==3 && native_commits==3 &&
        assemblies==6 && uses==2 && failures==5 && !warnings && !locked);
    // Ordinary CPU/import source bypasses the GPU logical-image selector.
    origin.mem=(void *)(uintptr_t)0x8000;
    VkTextureTransferInput none={0};
    const p_copy_b2t t2b={.tex=&origin,.r={1,1,0,4,2,1}};
    VkTextureAliasFailure failure={0};
    assert(!vk_exec_transfer_image_input_capture(OP_COPY_T2B,&t2b,&none,&failure) && !none.texture);
    // A GPU-only raw mapping without any overlapping projected writer also takes the existing route.
    origin.mem=(void *)(uintptr_t)BASE;
    origin.L.size=32;
    resident.type=VK_IMAGE_TYPE_2D;
    resident.depth=1;
    registry=rc_registry_create((RcRegistryConfig){32,32,8});
    assert(registry);
    OK(rc_registry_declare_through(registry,read));
    assert(!vk_exec_transfer_image_input_capture(OP_COPY_T2B,&t2b,&none,&failure) && !none.texture && assemblies==6);
    rc_registry_destroy(registry);
    puts("Ordered T2T and T2B emitters used the retained source: old and replaced images, raw pitch gaps, 3D, overlap and stale mapping.");
}

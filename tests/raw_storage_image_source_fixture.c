#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "resource_command_effects.h"

enum { GFX_RES_SSBO=1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_MCLS_DEVICE, VK_MCLS_SCRATCH,
       VK_MUSE_SSBO, VK_MUSE_CPU_READ, VK_MEM_MIRROR_EXISTING, VK_GPA_CB_SLOT=120, VK_EXEC_ALIAS_READ_LIMIT=16,
       RC_BUDGET_BUFFER, VK_SUCCESS=0, VK_ERROR_DEVICE_LOST=-4, VK_ERROR_OUT_OF_DEVICE_MEMORY=-2 };
enum { VS=0, CS=5, RAW_BUFFER=7, COMPOSED_BUFFER=99, BYTES=64 };
static const uint64_t base=UINT64_C(0xf000000000020000);
typedef uint64_t VkBuffer, VkDeviceSize;
typedef void *VmaAllocation;
typedef int VkResult;
typedef uint32_t VkBufferUsageFlags;
typedef struct VkDrawIndexedIndirectCommand {
    uint32_t indexCount, instanceCount, firstIndex;
    int32_t vertexOffset;
    uint32_t firstInstance;
} VkDrawIndexedIndirectCommand;
typedef struct { uint64_t srcOffset, dstOffset, size; } VkBufferCopy;
#define VK_NULL_HANDLE 0
typedef struct { VkBuffer buffer; uint64_t offset,range; } VkDescriptorBufferInfo;
typedef struct { uint64_t addr,size; } p_bind_buf;
typedef struct { uint32_t flags,ptr,cbuf_slot,slot,count; } GfxShaderRes;
typedef struct { uint32_t kind,type,count,stage,set,binding; const GfxShaderRes *res; } VkPipeBinding;
typedef struct { uint32_t n_bindings; VkPipeBinding bindings[2]; } VkPipeInfo;
typedef struct { uint64_t fence; struct { int pb_spoiled; } *dc; p_bind_buf ssbo[8][16]; } VkExec;
typedef struct { VkBuffer buffer; uint64_t offset,avail; int cls; VmaAllocation alloc; } VkMemSpan;
typedef struct { uint64_t mapping,backing; unsigned owners; } VkMemSpanCapture;
typedef struct { uint64_t addr,size; VkMemSpan span; VkMemSpanCapture *native; RcInterval *mapped;
                 uint32_t mapped_count; } EffectStorageCapture;
typedef struct {
    const GfxShaderRes *resource;
    RcRegistryPosition read;
    uint64_t value;
    int stage, current;
} DeviceStoragePointer;
typedef struct { EffectStorageCapture storage; } SelectedStorageWriter;
typedef struct {
    void *rmw, *rmw_bitmap;
    uint64_t rmw_word;
    uint32_t count;
    SelectedStorageWriter entries[1];
} SelectedStorageWriters;
typedef struct {
    ResourceCommandEffects *commands;
    RcRegistryPosition position;
    SelectedStorageWriters *selected_writers;
    const DeviceStoragePointer *pointer_sources;
    uint32_t pointer_count;
    uint32_t gpa_count;
    int gpa_unknown, gpa_recorded;
    struct {uint64_t addr, size;} gpa_effects[1];
} EffectDeclaration;
static EffectDeclaration *effect_current;
typedef struct { RcRegistryNativeImageFailure coverage; } VkTextureAliasFailure;
typedef struct VkTextureAliasRead {
    RcRegistryPosition read;
    uint64_t mapping,backing,guest_bytes,upload_offset;
    int raw_output;
    struct { VmaAllocation allocation; } staging;
    uint32_t count;
    RcRegistryNativeImageSnapshotFragment fragments[8];
    RcInterval mapping_interval;
    uint8_t workspace[BYTES];
} VkTextureAliasRead;
typedef struct RawStorageInputs RawStorageInputs;
typedef struct { RawStorageInputs *raw_storage; } DrawDescriptorGrant;
typedef struct { uint64_t size; VkBufferUsageFlags usage; } VkBufferCreateInfo;
typedef int VmaAllocationCreateInfo;
typedef struct { void *pMappedData; } VmaAllocationInfo;
typedef struct { int producer,required; } RcVmaRequest;
typedef enum { VK_EXEC_GRAPHICS_READY, VK_EXEC_GRAPHICS_INVALIDATED, VK_EXEC_GRAPHICS_UNAVAILABLE,
               VK_EXEC_GRAPHICS_NEEDS_ORDERED, VK_EXEC_GRAPHICS_STOPPED } VkExecGraphicsResult;
typedef enum { VK_EXEC_RESTORE_READY, VK_EXEC_RESTORE_STALE, VK_EXEC_RESTORE_STOPPED } VkExecRestoreResult;
static struct { uint64_t ubo_max,sbo_align,sbo_max; } LIM={64,16,BYTES};
static struct { uint64_t sbuf_align; } vk_caps={16};
static struct { void *vma; } context;
static RcRegistry *registry;
static RcRegistryPosition position;
static uint64_t mapping=21, backing=9;
static uint8_t raw[BYTES+2], detached[2][BYTES];
static uint64_t allocation_size;
static VkBufferUsageFlags allocation_usage;
static unsigned live_inputs,live_aliases,live_buffers,placed_count,readbacks,recorded_raw;
static int remap_in_progress,capacity_denied,stop_in_progress;
static void *native_bytes;
static int host_freeze, stop_native_freeze;
static unsigned host_frozen, native_frozen;
typedef enum { VK_EXEC_HOST_INPUT_READY, VK_EXEC_HOST_INPUT_STALE, VK_EXEC_HOST_INPUT_STOPPED,
               VK_EXEC_HOST_INPUT_UNSUPPORTED, VK_EXEC_HOST_INPUT_BLOCKED, VK_EXEC_HOST_INPUT_ABSENT,
               VK_EXEC_HOST_INPUT_CHANGED, VK_EXEC_HOST_INPUT_UNTRACKED } VkExecHostInputResult;
typedef struct {const uint8_t *bytes;} VkExecHostInputRead;
typedef enum { VK_SYNC_PROGRESS_COMPLETE, VK_SYNC_PROGRESS_STOPPED,
               VK_SYNC_PROGRESS_STOPPED_AFTER_CUT } VkSyncProgressResult;
typedef struct {
    const VkMemSpan *span;
    VkMemSpanCapture *capture;
    const RcInterval *ranges;
    uint32_t range_count;
    VkBuffer snapshot;
    uint64_t bytes;
    const VkBufferCopy *words;
    uint32_t word_count;
} VkExecNativeInputCopy;
static void (*finish)(void *);
static void (*cancel)(void *,int);
static void *completion_owner;

static RcRegistryPosition vk_exec_resource_position(void) { return position; }
static uint64_t host_pointer;
static const DeviceStoragePointer *device_storage_pointer(const DeviceStoragePointer *p,uint32_t count,
    int stage,const GfxShaderRes *r) {
    for (uint32_t i=0;i<count;i++) if (p[i].stage==stage&&p[i].resource==r) return &p[i];
    return NULL;
}
static int device_storage_pointer_value_current(const VkExec *ex, const DeviceStoragePointer *p) {
    return ex->fence==p->read.scene && p->current && !rc_registry_position_compare(p->read,position);
}
static int ptr_addr_checked(const VkExec *ex,int stage,const GfxShaderRes *r,uint64_t *address) {
    const DeviceStoragePointer *p=effect_current?
        device_storage_pointer(effect_current->pointer_sources,effect_current->pointer_count,stage,r):NULL;
    if (p && !device_storage_pointer_value_current(ex,p)) return 0;
    *address=p?p->value:host_pointer;
    return 1;
}
static RcRegistry *fixture_registry(void) { return registry; }
#define rc_registry_runtime() fixture_registry()
static int nvn_gpu_only(uint64_t addr) { return addr>=base; }
static void limits_init(void) { abort(); }
static int nvn_stage_of(uint32_t stage) { return (int)stage; }
static int ssbo_slot(const GfxShaderRes *resource,int stage,int *uniform) {
    assert(stage==VS || stage==CS);*uniform=0;return resource->slot;
}
static int ssbo_bound_recipe(const VkExec *ex,int stage,const GfxShaderRes *resource,p_bind_buf *out,int *uniform) {
    int slot=ssbo_slot(resource,stage,uniform);
    if (slot<0 || slot>=16) return 0;
    *out=ex->ssbo[stage][slot];
    return 1;
}
static int vk_mem_lookup(uint64_t addr,uint64_t bytes,VkMemSpan *out) {
    if (addr<base || addr-base>=BYTES || bytes>BYTES-(addr-base)) return 0;
    *out=(VkMemSpan){.buffer=RAW_BUFFER,.offset=16+addr-base,.avail=BYTES-(addr-base),.cls=VK_MCLS_DEVICE,.alloc=raw};return 1;
}
static RcInterval extent(uint64_t size) { return (RcInterval){base,mapping,{backing,0,size}}; }
static uint32_t vk_mem_device_intervals(const VkMemSpan *span,uint64_t address,uint64_t bytes,RcInterval **out) {
    assert(span->buffer==RAW_BUFFER && address>=base && address-base<=BYTES && bytes<=BYTES-(address-base));
    *out=malloc(sizeof **out);assert(*out);
    **out=(RcInterval){address,mapping,{backing,address-base,bytes}};return 1;
}
static int effect_storage_capture_span(const GfxShaderRes *resource,uint64_t address,uint64_t bytes,int policy,
                                       EffectStorageCapture *out) {
    assert(!resource->flags && policy==VK_MEM_MIRROR_EXISTING && address==base && (bytes==0 || bytes==BYTES));
    *out=(EffectStorageCapture){.addr=base,.size=BYTES};
    assert(vk_mem_lookup(address,BYTES,&out->span));
    vk_mem_device_intervals(&out->span,address,BYTES,&out->mapped);
    out->native=malloc(sizeof *out->native);assert(out->native);
    *out->native=(VkMemSpanCapture){mapping,backing,1};live_inputs++;return 1;
}
static void vk_mem_span_capture_release(VkMemSpanCapture **native) {
    if (!*native) return;
    assert(live_inputs && (*native)->owners);
    if (!--(*native)->owners) { live_inputs--;free(*native); }
    *native=NULL;
}
static VkMemSpanCapture *vk_mem_span_capture_retain(VkMemSpanCapture *native) {
    assert(native && native->owners);native->owners++;return native;
}
static int vk_mem_span_capture_current(const VkMemSpanCapture *native,const VkMemSpan *span) {
    return native && native->mapping==mapping && native->backing==backing && span->buffer==RAW_BUFFER;
}
static int vk_mem_span_record_input(VkMemSpanCapture *native,const VkMemSpan *span) {
    assert(vk_mem_span_capture_current(native,span));recorded_raw++;return 1;
}

// Vulkan readback bytes, staging handles and physical holds are modeled. The winner selection is the real registry.
static RcRegistryResult vk_texture_alias_buffer_capture(const VkMemSpanCapture *native,const VkMemSpan *span,
    uint64_t address,uint64_t bytes,VkTextureAliasRead **out,VkTextureAliasFailure *failure) {
    assert(address>=base && bytes<=BYTES-(address-base) && !*out && vk_mem_span_capture_current(native,span));
    const RcRegistryBufferSpan baseline={{address,mapping,{backing,address-base,bytes}},{19,2,span->offset}};
    VkTextureAliasRead local={.read=position,.mapping=mapping,.backing=backing,.guest_bytes=bytes,.raw_output=1,
        .mapping_interval=baseline.interval};
    RcRegistryResult result=rc_registry_native_image_snapshot_mixed(registry,&baseline.interval,1,&baseline,1,
        position,local.fragments,8,&local.count,failure?&failure->coverage:NULL);
    if (result!=RC_REGISTRY_OK) return result;
    for (uint32_t i=0;i<local.count;i++) if (local.fragments[i].fragment.binding.image) {
        *out=malloc(sizeof local);assert(*out);**out=local;live_aliases++;break;
    }
    return result;
}
static const RcInterval *vk_texture_alias_read_intervals(const VkTextureAliasRead *alias, uint32_t *count) {
    *count=1;return &alias->mapping_interval;
}
static void vk_texture_alias_read_cancel(VkTextureAliasRead *alias) {
    if (!alias) return;
    assert(live_aliases);live_aliases--;free(alias);
}
static int vk_texture_alias_read_current(VkTextureAliasRead *alias) {
    return alias && alias->mapping==mapping && alias->backing==backing &&
        !rc_registry_position_compare(alias->read,position);
}
static int image_alias_compose_guest(VkTextureAliasRead *alias) {
    assert(readbacks && alias->guest_bytes<=BYTES);
    for (uint32_t i=0;i<alias->count;i++) {
        const RcRegistryNativeImageFragment *frag=&alias->fragments[i].fragment;
        const uint64_t image=frag->binding.image;
        assert(!image || image==100 || image==200);
        const uint8_t *bytes=image?detached[image==200]:raw+1;
        memcpy(alias->workspace+frag->interval.range.offset-alias->mapping_interval.range.offset,
               bytes+frag->interval.range.offset,frag->interval.range.size);
    }
    return 1;
}
#define vk_ctx() (&context)
static VkResult vmaInvalidateAllocation(void *vma,VmaAllocation allocation,uint64_t offset,uint64_t bytes) {
    (void)vma;(void)allocation;assert(!offset && !bytes);return VK_SUCCESS;
}
static void scratch_chunk_specs(uint64_t size,VkBufferCreateInfo *buffer,VmaAllocationCreateInfo *allocation) {
    *buffer=(VkBufferCreateInfo){.size=size};*allocation=1;
}
static VkResult resource_core_vma_create_buffer(const RcVmaRequest *request,void *vma,const VkBufferCreateInfo *info,
    const VmaAllocationCreateInfo *allocation,VkBuffer *buffer,VmaAllocation *owner,VmaAllocationInfo *mapped,void *status) {
    (void)vma;(void)status;
    assert(request->required && request->producer==RC_BUDGET_BUFFER && info->size<=BYTES && *allocation==1);
    if (capacity_denied) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    uint8_t *bytes=malloc(info->size+2);assert(bytes&&!native_bytes);
    memset(bytes,0xd9,info->size+2);*buffer=COMPOSED_BUFFER;*owner=bytes;native_bytes=mapped->pMappedData=bytes+1;
    allocation_size=info->size;allocation_usage=info->usage;
    live_buffers++;placed_count++;return VK_SUCCESS;
}
static void resource_core_vma_destroy_buffer(void *vma,VkBuffer buffer,VmaAllocation owner) {
    (void)vma;uint8_t *allocation=owner;
    assert(live_buffers && buffer==COMPOSED_BUFFER && allocation+1==native_bytes);
    assert(allocation[0]==0xd9 && allocation[allocation_size+1]==0xd9);
    free(owner);native_bytes=NULL;live_buffers--;
}
static void vk_device_lost_check(VkResult result,const char *stage) { (void)result;(void)stage;abort(); }
static int vk_device_is_lost(void) { return 0; }
static int vk_sync_stopping(void) { return 0; }
static VkExecRestoreResult vk_exec_prepare_alias_readback(VkExec *ex,VkTextureAliasRead *const *images,uint32_t count,
    int (*current)(void *),void *scope) {
    assert(ex->fence==position.scene && count==1 && images[0] && current(scope));
    if (stop_in_progress) return VK_EXEC_RESTORE_STOPPED;
    if (remap_in_progress) { mapping++;backing++;assert(!current(scope));return VK_EXEC_RESTORE_STALE; }
    readbacks++;return VK_EXEC_RESTORE_READY;
}
static void vk_completion_add_owned(uint64_t fence,void (*done)(void *),void (*abandoned)(void *,int),void *owner) {
    assert(fence==position.scene && !completion_owner);
    finish=done;cancel=abandoned;completion_owner=owner;
}
static int host_untracked, host_changed;
static VkExecHostInputResult vk_exec_host_input_begin(const VkMemSpanCapture *hold,const VkMemSpan *span,
    uint64_t address,uint64_t bytes,RcRegistryPosition read,VkExecHostInputRead *out) {
    assert(bytes==20 && address+bytes==base+BYTES && !rc_registry_position_compare(read,position));
    assert(vk_mem_span_capture_current(hold,span));
    if (host_untracked) return VK_EXEC_HOST_INPUT_UNTRACKED;
    if (!host_freeze) return VK_EXEC_HOST_INPUT_UNSUPPORTED;
    out->bytes=raw+1+address-base;host_frozen++;return VK_EXEC_HOST_INPUT_READY;
}
static VkExecHostInputResult vk_exec_host_input_finish(VkExecHostInputRead *read) {
    assert(read->bytes);return host_changed?VK_EXEC_HOST_INPUT_CHANGED:VK_EXEC_HOST_INPUT_READY;
}
static uint32_t vk_mem_span_input_intervals(const VkMemSpanCapture *native,uint64_t address,uint64_t size,
                                           RcInterval **out) {
    VkMemSpan span;
    assert(vk_mem_lookup(address,size,&span) && vk_mem_span_capture_current(native,&span));
    return vk_mem_device_intervals(&span,address,size,out);
}
static VkSyncProgressResult vk_exec_copy_native_inputs(VkExec *ex,RcRegistryPosition read,
    const VkExecNativeInputCopy *input,uint32_t count,VkBuffer dest,void (*record)(uint64_t,void *),void *ctx) {
    assert(count==1 && dest==COMPOSED_BUFFER && !rc_registry_position_compare(read,position));
    assert(input->range_count==1 && input->words[0].size==input->bytes && input->bytes==20 &&
        !input->snapshot && !input->words[0].srcOffset && !input->words[0].dstOffset);
    assert(vk_mem_span_capture_current(input->capture,input->span));
    assert(rc_registry_read_device_buffer_at(registry,input->ranges,input->range_count,read)==RC_REGISTRY_OK);
    record(ex->fence,ctx);native_frozen++;
    if (stop_native_freeze) return VK_SYNC_PROGRESS_STOPPED_AFTER_CUT;
    memcpy(native_bytes,raw+1+input->ranges[0].range.offset,input->bytes);
    return VK_SYNC_PROGRESS_COMPLETE;
}
static void complete(int abandoned) {
    if (abandoned) cancel(completion_owner,0);else finish(completion_owner);
    completion_owner=NULL;
}
static void rt_log(const char *format,...) { (void)format; }
static void rt_fatal(const char *reason,...) { (void)reason;abort(); }
typedef int PlatMutex;
#define PLAT_MUTEX_INIT 0
static void plat_mutex_lock(PlatMutex *p) { assert(!*p);*p=1; }
static void plat_mutex_unlock(PlatMutex *p) { assert(*p);*p=0; }

#ifdef BASELINE
static int vk_mem_resolve(uint64_t address,uint64_t bytes,VkMemSpan *span) { return vk_mem_lookup(address,bytes,span); }
#define VK_MEM_USE(span) ((void)(span))
#define VK_MEM_BOUND(span) ((void)(span))
static int note_written(VkExec *ex,uint64_t address,uint64_t bytes,VkMemSpan *span) {
    (void)ex;(void)address;(void)bytes;(void)span;abort();
}
static void vk_exec_required_graphics_unavailable(VkExec *ex,unsigned opcode,const char *reason) {
    (void)ex;(void)opcode;(void)reason;abort();
}
static void vk_mem_gpu_wrote_shader(const VkMemSpan *span,uint64_t bytes) { (void)span;(void)bytes;abort(); }
static void vk_usnap_drop(uint64_t address,uint64_t bytes) { (void)address;(void)bytes;abort(); }
static void count_use(void *dc,int cls,int use,uint64_t bytes) { (void)dc;(void)cls;(void)use;(void)bytes; }
static void vk_mem_count(int cls,int use,uint64_t bytes) { (void)cls;(void)use;(void)bytes; }
static void *scratch(VkExec *ex,uint64_t size,VkBuffer *buffer,VkDeviceSize *offset) { (void)ex;(void)size;(void)buffer;(void)offset;abort(); }
#endif

#ifdef BASELINE
static void raw_storage_inputs_release(void *owner) { assert(!owner); }
#else
static void raw_storage_inputs_release(void *);
#endif
static void draw_descriptor_grant_release(DrawDescriptorGrant *grant) {
    raw_storage_inputs_release(grant->raw_storage);grant->raw_storage=NULL;
}
/* Production functions */

static void producer(uint64_t image,uint64_t start,uint64_t ordinal) {
    const RcRegistryPosition output={1,ordinal};
    const RcInterval original={.address=base+image*4096,.mapping=image+1,.range={9,0,44}};
    const RcSourceRange authored={9,start,12};
    RcRegistryImageWriteExtent owner={.interval=original};
    RcRegistryWrite writer;
    assert(rc_registry_image_hold_begin_extents(registry,image,&original,1,&owner.hold)==RC_REGISTRY_OK);
    assert(rc_registry_write_begin_at(registry,original.range,RC_REGISTRY_IMAGE,output,&writer)==RC_REGISTRY_OK);
    assert(rc_registry_image_producer_add(registry,writer,output)==RC_REGISTRY_OK);
    assert(rc_registry_declare_through(registry,output)==RC_REGISTRY_OK);
    const RcRegistryNativeImageWrite projection={owner,&authored,1};
    assert(rc_registry_native_image_write_fragments(registry,image,&projection,1,output)==RC_REGISTRY_OK);
    const RcRegistryImageBinding binding={image,owner.hold};
    assert(rc_registry_image_producer_resolve(registry,writer,output,&binding,0)==RC_REGISTRY_OK);
    assert(rc_registry_handoff_through(registry,output)==RC_REGISTRY_OK);
    assert(rc_registry_write_submit(registry,writer,1)==RC_REGISTRY_OK);
    rc_registry_image_producer_native_completed(registry,writer,output,binding);
    assert(rc_registry_write_complete_effects(registry,writer,NULL,0,NULL,0)==RC_REGISTRY_OK);
    // The real image hold outlives its registration. There is no live guest texture to reselect here.
}

static void check_contents(VkDescriptorBufferInfo descriptor) {
    const uint8_t *contents=descriptor.buffer==COMPOSED_BUFFER?native_bytes:raw+1;
    assert(descriptor.range==BYTES && descriptor.offset%16==0 && contents);
    for (unsigned i=0;i<BYTES;i++) {
        const uint8_t value=i>=29&&i<41?detached[1][i]:i>=9&&i<21?detached[0][i]:raw[1+i];
        if (contents[i]!=value) { fputs("FAIL shader descriptor used stale physical buffer bytes\n",stderr);abort(); }
    }
    assert(raw[0]==0xa6 && raw[BYTES+1]==0xa6);
}

#if !defined(BASELINE)
static void check_pointer_target(VkExec *ex) {
    const GfxShaderRes reader={.ptr=1,.slot=3,.count=1};
    const VkPipeInfo info={1,{{GFX_RES_SSBO,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,CS,0,3,&reader}}};
    VkDescriptorBufferInfo descriptor={0};
    DrawDescriptorGrant grant={0};
    DeviceStoragePointer pointer={.resource=&reader,.read=position,.value=base,.stage=CS,.current=1};
    EffectDeclaration scope={.position={position.scene,position.ordinal+1},
                             .pointer_sources=&pointer,.pointer_count=1};
    scope.commands=resource_effects_create(scope.position);
    effect_current=&scope;
    p_bind_buf unused;
    assert(!raw_storage_fixed(ex,&info.bindings[0],&unused));
    assert(raw_storage_bind(ex,&grant,&info.bindings[0],&descriptor)==RAW_STORAGE_MISSING);
    assert(raw_storage_prepare(ex,&info,&grant,0,&pointer,1)==VK_EXEC_GRAPHICS_NEEDS_ORDERED);
    assert(raw_storage_prepare(ex,&info,&grant,1,&pointer,1)==VK_EXEC_GRAPHICS_READY);
    pointer.value=base+16;
    assert(raw_storage_bind(ex,&grant,&info.bindings[0],&descriptor)==RAW_STORAGE_MISSING);
    pointer.value=base;
    pointer.current=0;
    assert(raw_storage_bind(ex,&grant,&info.bindings[0],&descriptor)==RAW_STORAGE_MISSING);
    pointer.current=1;
    mapping++;
    assert(raw_storage_bind(ex,&grant,&info.bindings[0],&descriptor)==RAW_STORAGE_MISSING);
    mapping--;
    assert(raw_storage_bind(ex,&grant,&info.bindings[0],&descriptor)==RAW_STORAGE_BOUND);
    check_contents(descriptor);
    raw_storage_inputs_release(grant.raw_storage);grant.raw_storage=NULL;
    complete(1);
    assert(!live_inputs && !live_aliases && !live_buffers);
    resource_effects_release(scope.commands);

    // Compatibility may resolve the same scalar through an eligible host table.
    host_pointer=base;
    scope.pointer_sources=NULL;
    scope.pointer_count=0;
    scope.commands=resource_effects_create(scope.position);
    const RcInterval output={base+4096,mapping+1,{backing,9,4}};
    assert(resource_effects_add_intervals(scope.commands,&output,1,&output,1,RC_REGISTRY_BUFFER)==RC_REGISTRY_OK);
    assert(raw_storage_prepare(ex,&info,&grant,1,NULL,0)==VK_EXEC_GRAPHICS_READY);
    assert(raw_storage_bind(ex,&grant,&info.bindings[0],&descriptor)==RAW_STORAGE_OVERLAP);
    raw_storage_inputs_release(grant.raw_storage);grant.raw_storage=NULL;
    assert(!live_inputs && !live_aliases && !live_buffers && !completion_owner);
    resource_effects_release(scope.commands);
    scope.commands=resource_effects_create(scope.position);
    assert(raw_storage_prepare(ex,&info,&grant,1,NULL,0)==VK_EXEC_GRAPHICS_READY);
    assert(raw_storage_bind(ex,&grant,&info.bindings[0],&descriptor)==RAW_STORAGE_BOUND);
    check_contents(descriptor);
    raw_storage_inputs_release(grant.raw_storage);
    complete(0);
    resource_effects_release(scope.commands);
    effect_current=NULL;
    host_pointer=0;
    puts("Selected read-only storage addresses retain mixed image and raw bytes through normal descriptor binding");
}

static void check_shader_writer_alias(VkExec *ex, const GfxShaderRes *reader) {
    const GfxShaderRes writable={.flags=1,.slot=1};
    const VkPipeInfo shader={2,{{GFX_RES_SSBO,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,CS,0,3,reader},
        {GFX_RES_SSBO,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,CS,0,4,&writable}}};
    const uint64_t other=base+4096;
    ex->ssbo[CS][1]=(p_bind_buf){other,BYTES};
    DrawDescriptorGrant grant={0};
    assert(raw_storage_prepare(ex,&shader,&grant,1,NULL,0)==VK_EXEC_GRAPHICS_READY);
    VkDescriptorBufferInfo descriptor={0};
    EffectDeclaration scope={.position={position.scene,position.ordinal+1}};
    scope.commands=resource_effects_create(scope.position);
    effect_current=&scope;
    const RcInterval overlap={other,mapping+1,{backing,8,4}};
    assert(resource_effects_add_intervals(scope.commands,&overlap,1,&overlap,1,RC_REGISTRY_BUFFER)==RC_REGISTRY_OK);
    if (raw_storage_bind(ex,&grant,&shader.bindings[0],&descriptor)!=-2 || descriptor.buffer)
        { fputs("FAIL private shader buffer bound despite same-shader normalized writer overlap\n",stderr);abort(); }
    assert(!completion_owner);
    resource_effects_release(scope.commands);
    effect_current=NULL;
    assert(raw_storage_bind(ex,&grant,&shader.bindings[0],&descriptor)==-2 && !descriptor.buffer);
    effect_current=&scope;

    scope.commands=resource_effects_create(scope.position);
    assert(resource_effects_opaque_intervals(scope.commands,NULL)==RC_REGISTRY_OK);
    assert(raw_storage_bind(ex,&grant,&shader.bindings[0],&descriptor)==-2 && !descriptor.buffer);
    resource_effects_release(scope.commands);

    scope.commands=resource_effects_create(scope.position);
    const RcInterval disjoint={other,mapping+1,{backing,BYTES,4}};
    assert(resource_effects_add_intervals(scope.commands,&disjoint,1,&disjoint,1,RC_REGISTRY_BUFFER)==RC_REGISTRY_OK);
    VkMemSpanCapture native={mapping,backing,1};
    SelectedStorageWriters marked={.rmw_bitmap=&marked,.count=1,.entries={{{.native=&native,
        .mapped=(RcInterval *)&overlap,.mapped_count=1,.span={.buffer=RAW_BUFFER}}}}};
    scope.selected_writers=&marked;
    assert(raw_storage_bind(ex,&grant,&shader.bindings[0],&descriptor)==-2 && !descriptor.buffer);
    marked.rmw=&marked;
    marked.rmw_bitmap=NULL;
    marked.rmw_word=other;
    assert(raw_storage_bind(ex,&grant,&shader.bindings[0],&descriptor)==-2 && !descriptor.buffer);
    RcInterval both={other,mapping+1,{backing,BYTES-4,8}};
    marked.entries[0].storage.mapped=&both;
    marked.rmw_word+=4;
    assert(raw_storage_bind(ex,&grant,&shader.bindings[0],&descriptor)==1);
    assert(descriptor.buffer==COMPOSED_BUFFER);
    check_contents(descriptor);
    effect_current=NULL;
    resource_effects_release(scope.commands);
    raw_storage_inputs_release(grant.raw_storage);
    complete(0);
    assert(!live_inputs && !live_aliases && !live_buffers);
}
static void check_projected_u8(VkExec *ex) {
    const uint64_t address=base+18, bytes=6;
    const uint8_t prior=detached[0][18];
    detached[0][18]=UINT8_MAX;
    VkMemSpan span;
    assert(vk_mem_lookup(address,bytes,&span));
    VkMemSpanCapture *hold=malloc(sizeof *hold);assert(hold);
    *hold=(VkMemSpanCapture){mapping,backing,1};live_inputs++;
    uint16_t out[8]={0xcafe};
    out[7]=0xbabe;
    assert(graphics_widen_projected(ex,&span,hold,address,bytes,position,out+1,1)==VK_EXEC_HOST_INPUT_READY);
    for (unsigned i=0;i<bytes;i++) {
        const uint16_t expected=i==0?UINT16_MAX:i<3?detached[0][18+i]:raw[19+i];
        assert(out[1+i]==expected);
    }
    assert(out[0]==0xcafe && out[7]==0xbabe);
    assert(!completion_owner && !live_buffers && !live_aliases);
    remap_in_progress=1;
    assert(graphics_widen_projected(ex,&span,hold,address,bytes,position,out+1,0)==VK_EXEC_HOST_INPUT_STALE);
    mapping--;backing--;remap_in_progress=0;
    assert(out[0]==0xcafe && out[7]==0xbabe && out[1]==UINT16_MAX);
    vk_mem_span_capture_release(&hold);
    assert(!live_inputs && !live_buffers && !live_aliases);
    detached[0][18]=prior;
}
static void check_direct_index(VkExec *ex) {
    const VkPipeInfo pipeline={0};
    for (uint64_t size=2;size<=4;size*=2) {
        const uint64_t address=base+12;
        VkMemSpan span;
        assert(vk_mem_lookup(address,3*size,&span) && span.offset%size==0);
        VkMemSpanCapture *hold=malloc(sizeof *hold);assert(hold);
        *hold=(VkMemSpanCapture){mapping,backing,1};live_inputs++;
        const RawStorageInput argument={.span=span,.native=hold,.address=address,.bytes=3*size};
        DrawDescriptorGrant grant={0};
        assert(prepared_buffer_inputs(ex,&pipeline,&grant,1,&argument,1,0)==VK_EXEC_GRAPHICS_READY);
        const RawStorageInput *prepared=prepared_buffer_argument(&grant,address,3*size);
        assert(prepared && prepared->image && prepared->bytes==3*size);
        VkMemSpan recorded=span;
        assert(prepared_buffer_argument_record(ex,&grant,prepared,&recorded));
        assert(recorded.buffer==COMPOSED_BUFFER && recorded.offset%size==0 && recorded.avail==3*size);
        for (unsigned i=0;i<3*size;i++) {
            const uint8_t expected=i<9?detached[0][12+i]:raw[13+i];
            assert(((uint8_t *)native_bytes)[recorded.offset+i]==expected);
        }
        vk_mem_span_capture_release(&hold);
        raw_storage_inputs_release(grant.raw_storage);
        complete(size==2);
        assert(!live_inputs && !live_aliases && !live_buffers);
    }
}
static void check_index_command_cut(VkExec *ex) {
    const uint64_t address=base+44;
    const VkDrawIndexedIndirectCommand command={.indexCount=7,.instanceCount=2,.firstIndex=24,.vertexOffset=-3,.firstInstance=13};
    uint8_t original[sizeof command];
    memcpy(original,raw+1+44,sizeof original);
    memcpy(raw+1+44,&command,sizeof command);
    VkMemSpan span;
    assert(vk_mem_lookup(address,sizeof command,&span));
    VkMemSpanCapture *hold=malloc(sizeof *hold);assert(hold);
    *hold=(VkMemSpanCapture){mapping,backing,1};live_inputs++;
    const RawStorageInput input={.span=span,.native=hold,.address=address,.bytes=sizeof command,.freeze=1};
    // Once exact host proof refuses (unwatchable or changed mid-borrow), a
    // native copy of the same live import is not an original-command receipt.
    const unsigned prior_native=native_frozen;
    for (int reason=0;reason<2;reason++) {
        host_untracked=!reason;
        host_freeze=host_changed=reason;
        DrawDescriptorGrant grant={0};
        assert(prepared_buffer_inputs(ex,NULL,&grant,1,&input,1,0x100)==VK_EXEC_GRAPHICS_UNAVAILABLE);
        assert(!grant.raw_storage && !live_buffers && !completion_owner && native_frozen==prior_native);
    }
    host_untracked=host_changed=0;
    for (unsigned i=0;i<3;i++) {
        host_freeze=i==0;
        stop_native_freeze=i==2;
        DrawDescriptorGrant grant={0};
        if (stop_native_freeze) {
            assert(prepared_buffer_inputs(ex,NULL,&grant,1,&input,1,0x100)==VK_EXEC_GRAPHICS_STOPPED);
            assert(!grant.raw_storage && live_buffers==1 && completion_owner);
            complete(1);
            break;
        }
        assert(prepared_buffer_inputs(ex,NULL,&grant,1,&input,1,0x100)==VK_EXEC_GRAPHICS_READY);
        const RawStorageInput *picked=prepared_buffer_argument(&grant,address,sizeof command);
        assert(picked && !picked->image && picked->freeze && !picked->offset);
        assert(!memcmp(native_bytes,&command,sizeof command));
        VkDrawIndexedIndirectCommand copied;
        memcpy(&copied,native_bytes,sizeof copied);
        uint64_t offset,bytes;
        assert(indexed_draw_reached_bytes(&copied,2,&offset,&bytes) && offset==48 && bytes==14);
        assert(indexed_draw_reached_bytes(&copied,4,&offset,&bytes) && offset==96 && bytes==28);
        assert(copied.vertexOffset==-3 && copied.firstInstance==13);
        copied.instanceCount=0;
        assert(indexed_draw_reached_bytes(&copied,4,&offset,&bytes) && !bytes);
        copied.instanceCount=1;copied.indexCount=0;
        assert(indexed_draw_reached_bytes(&copied,4,&offset,&bytes) && !bytes);
        copied.indexCount=UINT32_MAX;copied.firstIndex=UINT32_MAX;
        assert(indexed_draw_reached_bytes(&copied,4,&offset,&bytes) && offset+bytes==UINT64_C(34359738360));
        assert(indexed_draw_reached_bytes(&copied,1,&offset,&bytes) && offset+bytes==UINT64_C(8589934590));
        assert(!indexed_draw_reached_bytes(&copied,3,&offset,&bytes));
        VkMemSpan recorded=span;
        assert(prepared_buffer_argument_record(ex,&grant,picked,&recorded) && recorded.buffer==COMPOSED_BUFFER);
        mapping++;
        assert(!prepared_buffer_argument_record(ex,&grant,picked,&recorded));
        mapping--;
        raw_storage_inputs_release(grant.raw_storage);
        assert(live_inputs==1 && live_buffers==1);
        complete(0);
    }
    stop_native_freeze=host_freeze=0;
    assert(host_frozen==2 && native_frozen==2);
    vk_mem_span_capture_release(&hold);
    memcpy(raw+1+44,original,sizeof original);
    assert(!live_buffers && !live_inputs && !live_aliases);
}
static void check_indirect_argument(VkExec *ex) {
    // The physical command has x=1. The retained image authored the exact command's x=3.
    const uint64_t address=base+12;
    const uint32_t physical[3]={1,1,1}, projected[3]={3,1,1};
    uint8_t original_raw[12], original_image[12];
    memcpy(original_raw,raw+13,12);
    memcpy(original_image,detached[0]+12,12);
    memcpy(raw+13,physical,12);
    memcpy(detached[0]+12,projected,12);
    const VkPipeInfo pipeline={0};
    DrawDescriptorGrant grant={0};
    VkMemSpan span;
    assert(vk_mem_lookup(address,12,&span));
    VkMemSpanCapture *hold=malloc(sizeof *hold);assert(hold);
    *hold=(VkMemSpanCapture){mapping,backing,1};live_inputs++;
    const RawStorageInput argument={.span=span,.native=hold,.address=address,.bytes=12};
    assert(prepared_buffer_requires_image(&span,address,12,position));
    assert(prepared_buffer_inputs(ex,&pipeline,&grant,1,&argument,1,0x100)==VK_EXEC_GRAPHICS_READY);
    assert(allocation_usage==0x100);
    const RawStorageInput *ready=prepared_buffer_argument(&grant,address,12);
    assert(ready && ready->native==hold && ready->span.buffer==RAW_BUFFER);
    assert(!prepared_buffer_argument(&grant,address,11));
    assert(!prepared_buffer_argument(&grant,address+4,12));
    const uint64_t lead=span.offset & 15;
    assert(memcmp(raw+13,(uint8_t *)native_bytes+lead,12));
    assert(!memcmp((uint8_t *)native_bytes+lead,projected,12));
    VkMemSpan selected=span;
    mapping++;
    assert(!prepared_buffer_argument_record(ex,&grant,ready,&selected));
    assert(!completion_owner && selected.buffer==RAW_BUFFER);
    mapping--;
    assert(prepared_buffer_argument_record(ex,&grant,ready,&selected));
    assert(selected.buffer==COMPOSED_BUFFER && selected.offset==lead && selected.avail==12);
    vk_mem_span_capture_release(&hold);
    raw_storage_inputs_release(grant.raw_storage);
    assert(live_inputs==1 && live_buffers==1 && completion_owner);
    complete(1);
    assert(!live_inputs && !live_aliases && !live_buffers);
    assert(raw[0]==0xa6 && raw[BYTES+1]==0xa6);
    memcpy(raw+13,original_raw,12);
    memcpy(detached[0]+12,original_image,12);
}
#endif

int main(void) {
    registry=rc_registry_create((RcRegistryConfig){16,32,4});assert(registry);
    memset(raw,0xa6,sizeof raw);
    for (unsigned i=0;i<BYTES;i++) {raw[i+1]=(uint8_t)i;detached[0][i]=(uint8_t)(50+i);detached[1][i]=(uint8_t)(120+i);}
    producer(100,9,3);producer(200,29,5);
    position=(RcRegistryPosition){1,6};
    assert(rc_registry_declare_through(registry,position)==RC_REGISTRY_OK);
    VkExec ex={.fence=1};
    VkDescriptorBufferInfo descriptor;
#ifdef BASELINE
    assert(!context.vma && vk_caps.sbuf_align==LIM.sbo_align);
    assert(!allocation_usage && !host_frozen && !native_frozen);
    uint64_t target=base;
#ifdef POINTER_BASELINE
    const GfxShaderRes reader={.ptr=1,.slot=3,.count=1};
    const VkPipeBinding binding={GFX_RES_SSBO,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,CS,0,3,&reader};
    const DeviceStoragePointer pointer={.resource=&reader,.read=position,.value=base,.stage=CS,.current=1};
    EffectDeclaration scope={.pointer_sources=&pointer,.pointer_count=1};
    effect_current=&scope;
    p_bind_buf unused;
    assert(ptr_addr_checked(&ex,CS,&reader,&target) && target==base && !raw_storage_fixed(&ex,&binding,&unused));
    const RcInterval demand=extent(BYTES);
    assert(rc_registry_native_image_buffer_requires_coverage(registry,&demand,1,position));
#endif
    assert(bind_range(&ex,target,0,LIM.sbo_align,LIM.sbo_max,VK_MUSE_SSBO,0,&descriptor));
    assert(!placed_count && !recorded_raw);
    check_contents(descriptor);
#else
    const GfxShaderRes resource={.slot=0};
    const VkPipeInfo graphics={1,{{GFX_RES_SSBO,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VS,0,2,&resource}}};
    const VkPipeInfo compute={1,{{GFX_RES_SSBO,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,CS,0,3,&resource}}};
    ex.ssbo[VS][0]=ex.ssbo[CS][0]=(p_bind_buf){base,BYTES};
    check_pointer_target(&ex);
    check_shader_writer_alias(&ex,&resource);
    check_indirect_argument(&ex);
    check_direct_index(&ex);
    check_projected_u8(&ex);
    readbacks=0;
    EffectDeclaration scope={.position={1,7}};
    scope.commands=resource_effects_create(scope.position);
    effect_current=&scope;
    DrawDescriptorGrant grant={0};
    assert(raw_storage_bind(&ex,&grant,&graphics.bindings[0],&descriptor)==-1);
    assert(raw_storage_prepare(&ex,&graphics,&grant,0,NULL,0)==VK_EXEC_GRAPHICS_NEEDS_ORDERED);
    assert(!live_inputs && !live_buffers);
    capacity_denied=1;
    assert(raw_storage_prepare(&ex,&graphics,&grant,1,NULL,0)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!live_inputs && !live_aliases && !readbacks);
    capacity_denied=0;
    stop_in_progress=1;
    assert(raw_storage_prepare(&ex,&graphics,&grant,1,NULL,0)==VK_EXEC_GRAPHICS_STOPPED);
    assert(!live_inputs && !live_aliases && !live_buffers);
    stop_in_progress=0;
    for (unsigned i=0;i<2;i++) {
        const VkPipeInfo *pipeline=i?&compute:&graphics;
        assert(raw_storage_prepare(&ex,pipeline,&grant,1,NULL,0)==VK_EXEC_GRAPHICS_READY);
        assert(raw_storage_bind(&ex,&grant,&pipeline->bindings[0],&descriptor)==1);
        assert(descriptor.buffer==COMPOSED_BUFFER);
        check_contents(descriptor);
        raw_storage_inputs_release(grant.raw_storage);grant.raw_storage=NULL;
        assert(completion_owner && live_buffers==1);
        complete(i);
        assert(!live_inputs && !live_aliases && !live_buffers);
    }
    assert(readbacks==2);
    assert(raw_storage_prepare(&ex,&compute,&grant,1,NULL,0)==VK_EXEC_GRAPHICS_READY);
    mapping++;backing++;
    assert(raw_storage_bind(&ex,&grant,&compute.bindings[0],&descriptor)==-1);
    assert(!completion_owner);
    raw_storage_inputs_release(grant.raw_storage);grant.raw_storage=NULL;
    assert(!live_inputs && !live_aliases && !live_buffers);
    mapping--;backing--;
    remap_in_progress=1;
    assert(raw_storage_prepare(&ex,&compute,&grant,1,NULL,0)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!grant.raw_storage && !live_inputs && !live_aliases && !live_buffers);
    mapping--;backing--;remap_in_progress=0;

    resource_effects_release(scope.commands);
    effect_current=NULL;
    const RcRegistryPosition output={1,7};
    RcRegistryWrite raw_writer;
    assert(rc_registry_write_begin_at(registry,extent(BYTES).range,RC_REGISTRY_BUFFER,output,&raw_writer)==RC_REGISTRY_OK);
    assert(rc_registry_declare_through(registry,output)==RC_REGISTRY_OK);
    const RcRegistryBufferSpan newer={extent(BYTES),{19,2,16}};
    assert(rc_registry_buffer_write_record(registry,raw_writer,newer,output)==RC_REGISTRY_OK);
    position.ordinal=8;
    assert(rc_registry_declare_through(registry,position)==RC_REGISTRY_OK);
    const unsigned allocated_before=placed_count;
    scope.position.ordinal=9;
    scope.commands=resource_effects_create(scope.position);
    effect_current=&scope;
    const RcInterval alias={base+4096,mapping+1,{backing,0,4}};
    assert(resource_effects_add_intervals(scope.commands,&alias,1,&alias,1,RC_REGISTRY_BUFFER)==RC_REGISTRY_OK);
    assert(raw_storage_prepare(&ex,&compute,&grant,1,NULL,0)==VK_EXEC_GRAPHICS_READY);
    assert(raw_storage_bind(&ex,&grant,&compute.bindings[0],&descriptor)==1 && descriptor.buffer==RAW_BUFFER);
    assert(recorded_raw==1 && placed_count==allocated_before);
    raw_storage_inputs_release(grant.raw_storage);complete(0);
    assert(!live_inputs && !live_aliases && !live_buffers);
    resource_effects_release(scope.commands);
    effect_current=NULL;
    check_index_command_cut(&ex);
    puts("Indexed indirect command preserved its exact host or ordered-native snapshot and derived a bounded range");
    puts("Raw graphics and compute storage: mixed bytes, disjoint writers, same-shader aliases, raw provider, remap and teardown passed");
#endif
    rc_registry_destroy(registry);
}

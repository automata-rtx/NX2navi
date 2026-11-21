#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "vk_exec.h"
#include "resource_core_vma.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_tex.h"
#include "tex.h"

#define RC_REGISTRY_SINGLE_THREADED 1
#define rc_registry_runtime unused_registry_runtime
#include "resource_core_registry.c"
#undef rc_registry_runtime

#define HANDLE(type, id) ((type)(uintptr_t)(id))
#define OK(call) assert((call) == RC_REGISTRY_OK)
struct VkTextureRegistrationScope { RcRegistryScope horizon; uint64_t publication; };
struct MemBackingCapture { RcInterval interval; };
typedef struct { uint64_t addr; nvn_texdesc d; } TexKey;
typedef struct Tex {
    TexKey key;
    VkTexImage pub;
    tex_t t;
    TexVkFormat f;
    VmaAllocation alloc, bound_to;
    int released, gpu_dirty, retiring, superseded, overwritten, pin_detached;
    const RcInterval *image_device_intervals;
    uint32_t image_device_count, image_private_mapping_count;
    uint64_t image_device_size;
    MemBackingCapture *image_host_backing;
    void *pending_write;
    RcInterval origin;
    struct Tex *next_at;
} Tex;
typedef struct {
    RcRegistryHandle handle;
    RcRegistryCapture retained_record;
    uint64_t publication, exact_instance;
    tex_t descriptor;
    view_t view, exact_view;
    VkTexImage exact;
    int has_view, has_exact_view;
} TextureRegistration;
typedef struct {
    const TextureRegistration *registration;
    uint64_t pool_owner;
    int logical_id, used, finalized, chas_view;
    view_t cview;
    Tex *entry;
} TexSlot;
typedef struct { struct { uint64_t id; } value; } TextureMaterialInstance;
typedef struct { struct { const TextureMaterialInstance *instance; } use; } TextureUseNode;
static RcRegistry *texture_registrations;
static TextureImagePublication *texture_registration_images;
static TextureRegistration records[2];
static VkTextureRegistrationScope fixture_scope;
static TexSlot tslots[1];
static const int ntslots = 1;
static int by_key, by_addr;
static Tex *address_peer, *own_entry;
static struct MemBackingCapture host_capture;
static RcBackingResult backing_result = RC_BACKING_OK;
static uint32_t mapping_count = 1;
static int mapping_stable = 1;
static unsigned source_allocation_reads, mapping_lookups;
static int binding_attested;
static uint32_t expected_width = 32, expected_height = 16;
static VkFormat expected_format = VK_FORMAT_D32_SFLOAT;
static VkImageAspectFlags expected_aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
static const TextureRegistration *historical_owner;
static TextureMaterialInstance historical_instance;
static TextureUseNode historical_use;

static VkCtx ctx;
static VkExec ex;
static const VmaAllocation original_alloc = HANDLE(VmaAllocation, 11);
static const VmaAllocation carrier_alloc = HANDLE(VmaAllocation, 12);
static const VkImage original_image = HANDLE(VkImage, 13);
static const VkImage carrier_image = HANDLE(VkImage, 14);
static const VkImageView carrier_view = HANDLE(VkImageView, 15);
static const VkDeviceMemory original_memory = HANDLE(VkDeviceMemory, 16);
static const VkDeviceMemory carrier_memory = HANDLE(VkDeviceMemory, 17);
static RcRegistryPosition read = {9, 30};
static RcInterval mapping = {.address=0x5000, .mapping=20, .range={7, 64, 32768}};
static VkTexImage original;
static VkTextureSample sample;
static VkTextureSampleGeometry receipt;
static unsigned creates, views, destroys, view_destroys, host_releases, host_cancels, barriers;
static int floats = 1, sealed, dedicated = 1, aliased, admission_blocked, view_blocked, stale;

typedef struct {
    uint64_t fence;
    void (*fn)(void *);
    void *arg;
    void (*cancel)(void *, int);
} Pending;
static Pending pending;

VkCtx *vk_ctx(void) { return &ctx; }
volatile int32_t vk_lost;
int vk_sync_stopping(void) { return 0; }
int vk_exec_recording_failed(void) { return 0; }
uint64_t vk_exec_recording_fence(void) { return ex.fence; }
VkTextureRegistrationScope *vk_exec_texture_registration_scope(void) { return &fixture_scope; }
uint64_t vk_texture_registration_scope_publication(const VkTextureRegistrationScope *scope) { return scope->publication; }
uint64_t vk_texture_registration_scope_token(void) { return 1; }
uint64_t vk_exec_texture_pool(void) { return 23; }
RcRegistryPosition vk_exec_resource_position(void) { return read; }
RcRegistry *rc_registry_runtime(void) { return texture_registrations; }
int vk_texture_pool_slot_locked(uint64_t pool,int logical) { return pool == 23 && logical == 6 ? 0 : -1; }
static const TextureUseNode *texture_publication_owner_locked(TextureImagePublication *items,uint32_t index,VkImage image) {
    return historical_owner && items == texture_registration_images && index == historical_owner->handle.index &&
        image == historical_owner->exact.image ? &historical_use : NULL;
}
static Tex *texture_binding_entry(const TexSlot *slot,int native) {
    assert(slot == &tslots[0] && native == 0);
    return slot->entry;
}
static Tex *texture_registration_source(const TexSlot *slot) {
    assert(slot == &tslots[0]);
    return NULL;
}
static void slot_result_view(const TexSlot *slot,const Tex *entry,view_t *view,int *has_view) {
    assert(slot == &tslots[0] && entry);
    *view=slot->registration->view;
    *has_view=slot->registration->has_view;
}
int vk_texture_binding_receipt_locked(int native,VkImage image,RcRegistryPosition *next) {
    assert(!native && tslots[0].entry && tslots[0].entry->pub.image == image && !next);
    return binding_attested;
}
static const RcInterval *image_backing_intervals(const Tex *entry,uint32_t *count) {
    *count=1;
    return &entry->origin;
}
static int image_alias_guest_mapping_current(uint64_t address,uint64_t bytes,const RcInterval *intervals,uint32_t count) {
    assert(address == mapping.address && bytes == mapping.range.size && count == 1);
    return intervals->address == address && intervals->mapping == mapping.mapping &&
        rc_range_equal(intervals->range,mapping.range);
}
int vk_mem_image_alias_current(VkImage image,uint64_t address,uint64_t bytes,VmaAllocation allocation,
    const RcInterval *intervals,uint32_t count) {
    (void)image; (void)address; (void)bytes; (void)allocation; (void)intervals; (void)count; abort();
}
static void make_key(const tex_t *t,TexKey *key) { *key=(TexKey){(uintptr_t)t->mem,t->d}; }
static Tex *nvn_hm_get(const int *table,const void *key) {
    assert(key);
    if (table == &by_key) return own_entry;
    assert(table == &by_addr && *(const uint64_t *)key == mapping.address);
    return address_peer;
}
static int tex_mirrored(const tex_t *t) { return !nvn_gpu_only((uintptr_t)t->mem) && t->d.format != NVN_FORMAT_DEPTH32F; }
static unsigned umax(unsigned a,unsigned b) { return a > b ? a : b; }
const char *cfg_get(const char *name) { assert(!strcmp(name,"NX2NAVI_MINIMAL_LAYOUT")); return NULL; }
int tex_vk_format(int nvn_format,TexVkFormat *format) {
    if (nvn_format == NVN_FORMAT_DEPTH32F)
        *format=(TexVkFormat){.vk=VK_FORMAT_D32_SFLOAT,.ideal=VK_FORMAT_D32_SFLOAT,.depth=1};
    else if (nvn_format == NVN_FORMAT_RGBA8)
        *format=(TexVkFormat){.vk=VK_FORMAT_R8G8B8A8_UNORM,.ideal=VK_FORMAT_R8G8B8A8_UNORM};
    else return 0;
    return 1;
}
int tex_vk_format_supported(const TexVkFormat *format) {
    return format->vk == VK_FORMAT_D32_SFLOAT || format->vk == VK_FORMAT_R8G8B8A8_UNORM;
}
uint64_t mem_mapping_serial(void) { return 51; }
uint64_t mem_protection_serial(void) { return 52; }
int mem_mapping_range_unchanged(uint64_t address,uint64_t bytes,uint64_t serial) {
    assert(address == mapping.address && bytes == mapping.range.size && serial == 51);
    return mapping_stable;
}
int mem_protection_range_unchanged(uint64_t address,uint64_t bytes,uint64_t serial) {
    assert(address == mapping.address && bytes == mapping.range.size && serial == 52);
    return 1;
}
RcBackingResult mem_backing_capture_plan(uint64_t address,uint64_t bytes,MemBackingCapturePlan *plan) {
    assert(address == mapping.address && bytes == mapping.range.size);
    mapping_lookups++;
    *plan=(MemBackingCapturePlan){.address=address,.size=bytes,.count=mapping_count};
    return backing_result;
}
RcBackingResult mem_backing_capture_bounded(const MemBackingCapturePlan *plan,MemBackingCapture **out) {
    assert(plan->count == 1 && plan->address == mapping.address && !*out);
    host_capture.interval=mapping;
    *out=&host_capture;
    return RC_BACKING_OK;
}
const RcInterval *mem_backing_intervals(const MemBackingCapture *capture,uint32_t *count) {
    assert(capture == &host_capture);
    *count=1;
    return &capture->interval;
}
int mem_backing_current(const MemBackingCapture *capture) {
    assert(capture == &host_capture);
    return mapping_stable;
}
void mem_backing_release(MemBackingCapture **capture) { *capture=NULL; }
int vk_mem_lookup(uint64_t address,uint64_t bytes,VkMemSpan *span) {
    assert(address == mapping.address && bytes == mapping.range.size && nvn_gpu_only(address));
    mapping_lookups++;
    *span=(VkMemSpan){.cls=VK_MCLS_DEVICE};
    return backing_result == RC_BACKING_OK;
}
uint32_t vk_mem_device_intervals(const VkMemSpan *span,uint64_t address,uint64_t bytes,RcInterval **out) {
    assert(span->cls == VK_MCLS_DEVICE && address == mapping.address && bytes == mapping.range.size);
    *out=malloc(sizeof **out);
    assert(*out);
    **out=mapping;
    return 1;
}
int vk_texture_sample_geometry_locked(int logical,VkTextureSample *out,VkTextureSampleGeometry *geometry) {
    assert(logical==6);
    *out=sample;
    *geometry=receipt;
    if (stale) geometry->mapping.mapping++;
    return 1;
}
int vk_format_shader_numeric_type(VkFormat format) {
    assert(format==VK_FORMAT_D32_SFLOAT || format==VK_FORMAT_R8G8B8A8_UNORM);
    return floats ? GFX_TYPE_FLOAT : GFX_TYPE_UINT;
}
static VkFormatFeatureFlags features_of(VkFormat format) {
    assert(format==expected_format);
    return VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
}
void vk_device_lost_check(VkResult result,const char *where) {
    assert(result!=VK_ERROR_DEVICE_LOST && where);
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties(VkPhysicalDevice phys,VkFormat format,
    VkImageType type,VkImageTiling tiling,VkImageUsageFlags usage,VkImageCreateFlags flags,VkImageFormatProperties *out) {
    assert(phys==ctx.phys && format==expected_format && type==VK_IMAGE_TYPE_2D &&
           tiling==VK_IMAGE_TILING_OPTIMAL && usage==VK_IMAGE_USAGE_SAMPLED_BIT && !flags);
    *out=(VkImageFormatProperties){.maxExtent={64,64,1},.maxMipLevels=1,.maxArrayLayers=1,.sampleCounts=VK_SAMPLE_COUNT_1_BIT};
    return VK_SUCCESS;
}
VkResult resource_core_vma_create_image(const RcVmaRequest *request,VmaAllocator allocator,
    const VkImageCreateInfo *image,const VmaAllocationCreateInfo *placement,VkImage *native,VmaAllocation *allocation,
    VmaAllocationInfo *info,RcAdmissionStatus *admission) {
    assert(request->producer==RC_BUDGET_TEXTURE && !request->required && request->host_bytes && request->host_ticket &&
           allocator==ctx.vma && !info && !admission && image->format==expected_format &&
           image->extent.width==expected_width && image->extent.height==expected_height && image->extent.depth==1 &&
           image->mipLevels==1 && image->arrayLayers==1 && image->samples==VK_SAMPLE_COUNT_1_BIT &&
           image->usage==VK_IMAGE_USAGE_SAMPLED_BIT && !image->flags && !image->pNext &&
           image->initialLayout==VK_IMAGE_LAYOUT_UNDEFINED &&
           placement->flags==VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT && !placement->pool);
    creates++;
    if (admission_blocked) return VK_NOT_READY;
    *native=carrier_image;
    *allocation=carrier_alloc;
    return VK_SUCCESS;
}
VMA_CALL_PRE void VMA_CALL_POST vmaGetAllocationInfo2(VmaAllocator allocator,VmaAllocation allocation,VmaAllocationInfo2 *out) {
    assert(allocator==ctx.vma && allocation==carrier_alloc);
    *out=(VmaAllocationInfo2){.allocationInfo={.deviceMemory=aliased ? original_memory : carrier_memory,.size=4096},
        .dedicatedMemory=dedicated,.blockSize=4096};
}
VMA_CALL_PRE void VMA_CALL_POST vmaGetAllocationInfo(VmaAllocator allocator,VmaAllocation allocation,VmaAllocationInfo *out) {
    assert(allocator==ctx.vma && allocation==original_alloc);
    source_allocation_reads++;
    *out=(VmaAllocationInfo){.deviceMemory=original_memory,.size=65536,.offset=256};
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(VkDevice device,const VkImageViewCreateInfo *create,
    const VkAllocationCallbacks *allocator,VkImageView *view) {
    assert(device==ctx.device && !allocator && create->image==carrier_image && create->format==expected_format &&
           create->viewType==VK_IMAGE_VIEW_TYPE_2D && create->subresourceRange.aspectMask==expected_aspect &&
           !create->subresourceRange.baseMipLevel && create->subresourceRange.levelCount==1 &&
           !create->subresourceRange.baseArrayLayer && create->subresourceRange.layerCount==1);
    views++;
    if (view_blocked) return VK_ERROR_OUT_OF_HOST_MEMORY;
    *view=carrier_view;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(VkDevice device,VkImageView view,const VkAllocationCallbacks *allocator) {
    assert(device==ctx.device && view==carrier_view && !allocator);
    view_destroys++;
}
void resource_core_vma_destroy_image(VmaAllocator allocator,VkImage image,VmaAllocation allocation) {
    assert(allocator==ctx.vma && image==carrier_image && allocation==carrier_alloc);
    destroys++;
}
RcAdmissionStatus resource_core_budget_cancel(RcBudgetTicket *ticket) {
    assert(ticket);
    host_cancels++;
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_materialize(RcBudgetTicket *ticket,const RcAdmissionVector *cost) {
    assert(ticket && cost->amount[RC_BUDGET_RAM] && cost->amount[RC_BUDGET_RAM]==cost->amount[RC_BUDGET_COMMIT]);
    return RC_ADMISSION_OK;
}
void vk_replay_metadata_free(void *owner,RcBudgetTicket *ticket) {
    assert(ticket && owner);
    host_releases++;
    free(owner);
}
void vk_completion_add_owned(uint64_t fence,void (*done)(void *),void (*cancel)(void *,int),void *arg) {
    assert(fence==read.scene && done && cancel && arg && !pending.fn);
    pending=(Pending){fence,done,arg,cancel};
}
VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier2(VkCommandBuffer cb,const VkDependencyInfo *dependency) {
    assert(cb==ex.cb && sealed && pending.fn && dependency->imageMemoryBarrierCount==1 &&
           !dependency->bufferMemoryBarrierCount && !dependency->memoryBarrierCount);
    const VkImageMemoryBarrier2 *barrier=dependency->pImageMemoryBarriers;
    assert(barrier->image==carrier_image && barrier->srcStageMask==VK_PIPELINE_STAGE_2_NONE &&
           barrier->dstStageMask==VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT &&
           barrier->dstAccessMask==VK_ACCESS_2_SHADER_SAMPLED_READ_BIT &&
           barrier->oldLayout==VK_IMAGE_LAYOUT_UNDEFINED && barrier->newLayout==VK_IMAGE_LAYOUT_GENERAL &&
           barrier->subresourceRange.aspectMask==expected_aspect &&
           !barrier->subresourceRange.baseMipLevel && barrier->subresourceRange.levelCount==1);
    barriers++;
}
_Noreturn void rt_fatal(const char *message,...) { fputs(message,stderr); abort(); }

/* Extract carrier */
/* Extract retirement */

static void settle(uint64_t complete,uint64_t handed,int drained) {
    assert(pending_disposable(pending,complete,handed,drained));
    pending_dispose(pending,complete,handed);
    pending=(Pending){0};
}

static void publish_cold(TextureRegistration *registration,uint64_t serial,uint64_t publication) {
    registration->descriptor=(tex_t){.mem=(void *)(uintptr_t)mapping.address,.serial=serial,
        .d={.target=NVN_TEXTURE_TARGET_2D,.format=NVN_FORMAT_DEPTH32F,.width=128,.height=64,.depth=1,.levels=3},
        .L={.layers=1,.size=mapping.range.size}};
    registration->publication=publication;
    OK(rc_registry_register(texture_registrations,texture_registration_key(23,6),serial,mapping.range,
                            (uintptr_t)registration,&registration->handle));
    tslots[0]=(TexSlot){.registration=registration,.pool_owner=23,.logical_id=6,.used=1};
}

static void association_cases(const TextureRegistration *registration) {
    Tex peer={.t=registration->descriptor,.alloc=original_alloc,.origin=mapping,.image_private_mapping_count=1};
    peer.t.serial=90; // A different object can supply the selected representation.
    peer.pub=original;
    peer.pub.addr=mapping.address;
    peer.key=(TexKey){mapping.address,peer.t.d};
    assert(tex_vk_format(peer.t.d.format,&peer.f));
    address_peer=&peer;
    VkTextureSample associated;
    VkTextureSampleGeometry geometry;
    assert(vk_texture_sample_associated_geometry_locked(6,NULL,&associated,&geometry)==VK_TEXTURE_SAMPLE_GEOMETRY_READY);
    assert(geometry.association_only && !geometry.registration_only && !associated.image && !geometry.image &&
        !geometry.allocation && !geometry.layout_revision && geometry.extent.height==16);

    tslots[0].entry=&peer;
    tslots[0].cview=registration->view;
    tslots[0].chas_view=registration->has_view;
    assert(vk_texture_sample_associated_geometry_locked(6,NULL,&associated,&geometry)==VK_TEXTURE_SAMPLE_GEOMETRY_READY);
    VkTextureQueryImage *carrier=NULL;
    const unsigned native_reads=source_allocation_reads;
    assert(vk_texture_query_image_create_locked(6,&associated,&geometry,&carrier)==VK_SUCCESS);
    assert(source_allocation_reads==native_reads); // Consensus metadata needs neither a source allocation nor a layout.
    peer.pub.height=48;
    assert(!vk_texture_query_image_current_locked(carrier));
    vk_texture_query_image_release(&carrier);
    assert(vk_texture_sample_associated_geometry_locked(6,NULL,&associated,&geometry)==VK_TEXTURE_SAMPLE_GEOMETRY_ALTERNATE);

    const VkTextureSample candidate={.image=&peer.pub,.view=registration->view,.has_view=1,.native_slot=0,
        .registration=registration->handle.id};
    assert(vk_texture_sample_associated_geometry_locked(6,&candidate,&associated,&geometry)==VK_TEXTURE_SAMPLE_GEOMETRY_PROVIDER);
    binding_attested=1;
    assert(vk_texture_sample_associated_geometry_locked(6,&candidate,&associated,&geometry)==VK_TEXTURE_SAMPLE_GEOMETRY_READY);
    assert(geometry.association_only && !geometry.registration_only && associated.image==&peer.pub &&
        geometry.material==90 && geometry.extent.width==32 && geometry.extent.height==12 &&
        geometry.planes.baseMipLevel==2 && !geometry.layout_revision && geometry.image==peer.pub.image);
    expected_height=12;
    assert(vk_texture_query_image_create_locked(6,&associated,&geometry,&carrier)==VK_SUCCESS);
    assert(source_allocation_reads==native_reads+1);
    tslots[0].cview.base_level=1;
    assert(!vk_texture_query_image_record_locked(&ex,carrier));
    tslots[0].cview=registration->view;
    mapping.mapping++;
    assert(!vk_texture_query_image_current_locked(carrier));
    mapping.mapping--;
    assert(vk_texture_query_image_record_locked(&ex,carrier));
    VkImage actual=VK_NULL_HANDLE;
    assert(vk_texture_query_image_view(&ex,carrier,&actual)==carrier_view && actual==carrier_image &&
        actual!=peer.pub.image);
    vk_texture_query_image_release(&carrier);
    assert(!pending_disposable(pending,8,9,0));
    settle(9,9,0);
    binding_attested=0;
    tslots[0].entry=NULL;
    address_peer=NULL;
}

static void cold_cases(void) {
    TextureRegistration *registration=&records[0];
    publish_cold(registration,43,42);
    fixture_scope.publication=42;
    OK(rc_registry_scope_begin(texture_registrations,&fixture_scope.horizon));
    VkTextureSample cold;
    VkTextureSampleGeometry geometry;
    assert(vk_texture_sample_registered_geometry_locked(6,&cold,&geometry)==VK_TEXTURE_SAMPLE_GEOMETRY_READY);
    assert(geometry.extent.width==128 && geometry.extent.height==64 && geometry.planes.levelCount==3);
    assert(cold.has_view && cold.view.base_level==0 && cold.view.nlevels==3 && cold.view.nlayers==1);

    // Publish the selected mip as a new immutable registration. The old scope sees its finalized predecessor.
    OK(rc_registry_finalize(texture_registrations,texture_registration_key(23,6),43));
    registration=&records[1];
    registration->has_view=1;
    registration->view=(view_t){.set_levels=1,.base_level=2,.nlevels=1};
    publish_cold(registration,44,43);
    assert(vk_texture_sample_registered_geometry_locked(6,&cold,&geometry)==VK_TEXTURE_SAMPLE_GEOMETRY_GONE);
    OK(rc_registry_scope_end(texture_registrations,fixture_scope.horizon));
    OK(rc_registry_scope_begin(texture_registrations,&fixture_scope.horizon));
    fixture_scope.publication=43;
    assert(vk_texture_sample_registered_geometry_locked(6,&cold,&geometry)==VK_TEXTURE_SAMPLE_GEOMETRY_READY);
    assert(!cold.image && !cold.canonical && !cold.projected && cold.registration==registration->handle.id &&
        geometry.registration_only && !geometry.image && !geometry.allocation && !geometry.layout_revision &&
        geometry.format==VK_FORMAT_D32_SFLOAT && geometry.planes.aspectMask==VK_IMAGE_ASPECT_DEPTH_BIT &&
        geometry.extent.width==32 && geometry.extent.height==16 &&
        cold.view.base_level==2 && cold.view.nlevels==1 && cold.view.nlayers==1);
    const VkTextureSample saved_sample=cold;
    const VkTextureSampleGeometry saved=geometry;
    assert(!vk_texture_sample_geometry_current_locked(6,&cold,&original,&geometry,NULL,NULL));

    Tex peer={.key={.addr=mapping.address,.d=registration->descriptor.d}};
    peer.key.d.levels=1;
    address_peer=&peer;
    assert(vk_texture_sample_registered_geometry_locked(6,&cold,&geometry)==VK_TEXTURE_SAMPLE_GEOMETRY_ALTERNATE);
    peer.key.d.width=50;
    assert(vk_texture_sample_registered_geometry_locked(6,&cold,&geometry)==VK_TEXTURE_SAMPLE_GEOMETRY_READY);
    tslots[0].entry=&peer;
    assert(vk_texture_sample_registered_geometry_locked(6,&cold,&geometry)==VK_TEXTURE_SAMPLE_GEOMETRY_ALTERNATE);
    tslots[0].entry=NULL;
    own_entry=&peer;
    assert(vk_texture_sample_registered_geometry_locked(6,&cold,&geometry)==VK_TEXTURE_SAMPLE_GEOMETRY_ALTERNATE);
    own_entry=NULL;
    address_peer=NULL;
    mapping_count=2;
    assert(vk_texture_sample_registered_geometry_locked(6,&cold,&geometry)==VK_TEXTURE_SAMPLE_GEOMETRY_MAPPING);
    mapping_count=1;
    backing_result=RC_BACKING_ABSENT;
    assert(vk_texture_sample_registered_geometry_locked(6,&cold,&geometry)==VK_TEXTURE_SAMPLE_GEOMETRY_UNBACKED);
    mapping_stable=0;
    assert(vk_texture_sample_registered_geometry_locked(6,&cold,&geometry)==VK_TEXTURE_SAMPLE_GEOMETRY_MAPPING);
    mapping_stable=1;
    backing_result=RC_BACKING_OK;

    VkTextureQueryImage *carrier=NULL;
    const unsigned native_reads=source_allocation_reads, released=destroys;
    assert(vk_texture_query_image_create_locked(6,&saved_sample,&saved,&carrier)==VK_SUCCESS && carrier);
    assert(source_allocation_reads==native_reads); // Registration-only create never consults source allocation.
    mapping.range.backing++;
    assert(!vk_texture_query_image_current_locked(carrier) && !vk_texture_query_image_record_locked(&ex,carrier));
    assert(vk_texture_sample_registered_geometry_locked(6,&cold,&geometry)==VK_TEXTURE_SAMPLE_GEOMETRY_READY);
    assert(geometry.mapping.mapping==saved.mapping.mapping && geometry.mapping.range.backing!=saved.mapping.range.backing &&
        geometry.format==saved.format && geometry.extent.width==saved.extent.width);
    mapping.range.backing--;
    assert(vk_texture_query_image_current_locked(carrier) && vk_texture_query_image_record_locked(&ex,carrier));
    VkImage actual=VK_NULL_HANDLE;
    assert(vk_texture_query_image_view(&ex,carrier,&actual)==carrier_view && actual==carrier_image);
    vk_texture_query_image_release(&carrier);
    assert(destroys==released && !pending_disposable(pending,8,9,0));
    settle(8,9,1);
    assert(destroys==released+1);
    association_cases(registration);
    OK(rc_registry_scope_end(texture_registrations,fixture_scope.horizon));
}

static void historical_cases(void) {
    texture_registrations=rc_registry_create((RcRegistryConfig){8,8,4});
    assert(texture_registrations);
    OK(rc_registry_declare_through(texture_registrations,(RcRegistryPosition){read.scene,read.ordinal+1}));
    TextureRegistration *old=&records[0];
    const uint64_t key=texture_registration_key(23,6);
    *old=(TextureRegistration){.publication=61,.exact_instance=72,.has_view=1,.has_exact_view=1,
        .descriptor={.mem=(void *)(uintptr_t)mapping.address,.serial=55,.pool_flags=NVN_MEMORY_POOL_FLAGS_CPU_CACHED,
            .d={.target=NVN_TEXTURE_TARGET_2D,.format=NVN_FORMAT_RGBA8,.width=96,.height=48,.depth=1,.levels=3},
            .L={.layers=1,.size=mapping.range.size}},
        .view={.set_levels=1,.base_level=1,.nlevels=1},.exact_view={.set_levels=1,.base_level=1,.nlevels=1},
        .exact={.image=original_image,.format=VK_FORMAT_R8G8B8A8_UNORM,.type=VK_IMAGE_TYPE_2D,
            .view_type=VK_IMAGE_VIEW_TYPE_2D,.aspect=VK_IMAGE_ASPECT_COLOR_BIT,.width=96,.height=48,.depth=1,
            .levels=3,.layers=1,.samples=1,.usage=VK_IMAGE_USAGE_SAMPLED_BIT}};
    OK(rc_registry_register(texture_registrations,key,55,mapping.range,(uintptr_t)old,&old->handle));
    tslots[0]=(TexSlot){.registration=old,.pool_owner=23,.logical_id=6,.used=1};
    OK(rc_registry_scope_begin(texture_registrations,&fixture_scope.horizon));
    fixture_scope.publication=61;
    const VkTextureRegistrationScope historical_scope=fixture_scope;
    OK(rc_registry_capture(texture_registrations,old->handle,&old->retained_record));
    historical_owner=old;
    historical_instance.value.id=old->exact_instance;
    historical_use.use.instance=&historical_instance;

    OK(rc_registry_finalize(texture_registrations,key,55));
    mapping.mapping++;
    mapping.range.backing++; // Same guest address now denotes a different allocation and registration.
    records[1]=(TextureRegistration){0};
    publish_cold(&records[1],56,62);
    const unsigned looked_up=mapping_lookups, source_reads=source_allocation_reads;
    VkTextureSample geometry_sample;
    VkTextureSampleGeometry geometry;
    assert(vk_texture_sample_historical_geometry_locked(6,&geometry_sample,&geometry)==VK_TEXTURE_SAMPLE_GEOMETRY_READY);
    assert(geometry_sample.historical==VK_TEXTURE_REGISTRATION_EXACT && !geometry_sample.image &&
        geometry_sample.native_slot==-1 && geometry_sample.registration==old->handle.id &&
        geometry_sample.view.base_level==1 && geometry_sample.view.nlevels==1 &&
        geometry.history_capture==old->retained_record.id && geometry.history_scope.id==historical_scope.horizon.id &&
        geometry.material==72 && geometry.format==VK_FORMAT_R8G8B8A8_UNORM && geometry.extent.width==48 &&
        geometry.extent.height==24 && !geometry.image && !geometry.allocation && !geometry.layout_revision &&
        !geometry.mapping.mapping && !geometry.registration_only && !geometry.association_only);
    const VkTextureSample held_sample=geometry_sample;
    const VkTextureSampleGeometry held=geometry;
    expected_format=VK_FORMAT_R8G8B8A8_UNORM;
    expected_aspect=VK_IMAGE_ASPECT_COLOR_BIT;
    expected_width=48;
    expected_height=24;
    VkTextureQueryImage *carrier=NULL;
    assert(vk_texture_query_image_create_locked(6,&held_sample,&held,&carrier)==VK_SUCCESS);
    historical_instance.value.id++;
    assert(vk_texture_sample_historical_geometry_locked(6,&geometry_sample,&geometry)==VK_TEXTURE_SAMPLE_GEOMETRY_GONE &&
        !vk_texture_query_image_current_locked(carrier));
    historical_instance.value.id--;
    const uint64_t capture=old->retained_record.id;
    old->retained_record.id=0;
    assert(vk_texture_sample_historical_geometry_locked(6,&geometry_sample,&geometry)==VK_TEXTURE_SAMPLE_GEOMETRY_GONE);
    old->retained_record.id=capture;
    OK(rc_registry_scope_begin(texture_registrations,&fixture_scope.horizon));
    fixture_scope.publication=62;
    assert(vk_texture_sample_historical_geometry_locked(6,&geometry_sample,&geometry)==VK_TEXTURE_SAMPLE_GEOMETRY_REGISTRATION);
    assert(!vk_texture_query_image_record_locked(&ex,carrier));
    OK(rc_registry_scope_end(texture_registrations,fixture_scope.horizon));
    fixture_scope=historical_scope;
    assert(vk_texture_query_image_current_locked(carrier) && vk_texture_query_image_record_locked(&ex,carrier));
    VkImage actual=VK_NULL_HANDLE;
    assert(vk_texture_query_image_view(&ex,carrier,&actual)==carrier_view && actual==carrier_image && actual!=old->exact.image);
    const unsigned retained_images=destroys;
    vk_texture_query_image_release(&carrier);
    assert(destroys==retained_images && !pending_disposable(pending,8,9,0));
    settle(9,9,0);
    assert(destroys==retained_images+1 && mapping_lookups==looked_up && source_allocation_reads==source_reads);
    OK(rc_registry_capture_release(texture_registrations,old->retained_record));
    OK(rc_registry_scope_end(texture_registrations,historical_scope.horizon));
    rc_registry_destroy(texture_registrations);
}

int main(void) {
    texture_registrations=rc_registry_create((RcRegistryConfig){8,8,4});
    assert(texture_registrations);
    OK(rc_registry_declare_through(texture_registrations,read));
    fixture_scope.publication=41;
    ctx=(VkCtx){.vma=HANDLE(VmaAllocator,6),.phys=HANDLE(VkPhysicalDevice,7),.device=HANDLE(VkDevice,8)};
    ex=(VkExec){.cb=HANDLE(VkCommandBuffer,4),.fence=read.scene,.recorded=21};
    original=(VkTexImage){.image=original_image,.format=VK_FORMAT_D32_SFLOAT,.type=VK_IMAGE_TYPE_2D,
        .view_type=VK_IMAGE_VIEW_TYPE_2D,.aspect=VK_IMAGE_ASPECT_DEPTH_BIT,.width=128,.height=64,.depth=1,.levels=3,
        .layers=1,.samples=1,.usage=VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT};
    sample=(VkTextureSample){.image=&original,.native_slot=12,.registration=43,
        .view={.set_levels=1,.base_level=2,.nlevels=1},.has_view=1};
    receipt=(VkTextureSampleGeometry){.scope=&fixture_scope,.read=read,.pool=23,.publication=41,.registration=43,
        .material=5,.layout_revision=1,.image=original_image,.allocation=original_alloc,.mapping=mapping,
        .planes={VK_IMAGE_ASPECT_DEPTH_BIT,2,1,0,1},.extent={32,16},.format=VK_FORMAT_D32_SFLOAT,.native_slot=12};
    VkTextureQueryImage *query=NULL;
    VkImage actual=original_image;
    stale=1;
    assert(vk_texture_query_image_create_locked(6,&sample,&receipt,&query)==VK_ERROR_INITIALIZATION_FAILED && !creates);
    stale=0;
    floats=0;
    assert(vk_texture_query_image_create_locked(6,&sample,&receipt,&query)==VK_ERROR_FORMAT_NOT_SUPPORTED && !creates);
    floats=1;
    admission_blocked=1;
    assert(vk_texture_query_image_create_locked(6,&sample,&receipt,&query)==VK_NOT_READY && !destroys && !host_cancels);
    admission_blocked=0;
    dedicated=0;
    assert(vk_texture_query_image_create_locked(6,&sample,&receipt,&query)==VK_ERROR_FEATURE_NOT_PRESENT &&
           destroys==1 && host_cancels==1 && !views);
    dedicated=1;
    aliased=1;
    assert(vk_texture_query_image_create_locked(6,&sample,&receipt,&query)==VK_ERROR_FEATURE_NOT_PRESENT &&
           destroys==2 && host_cancels==2 && !views);
    aliased=0;
    view_blocked=1;
    assert(vk_texture_query_image_create_locked(6,&sample,&receipt,&query)==VK_ERROR_OUT_OF_HOST_MEMORY &&
           destroys==3 && host_cancels==3 && views==1 && !view_destroys);
    view_blocked=0;
    assert(vk_texture_query_image_create_locked(6,&sample,&receipt,&query)==VK_SUCCESS && query);
    assert(!vk_texture_query_image_view(&ex,query,&actual) && !actual);
    ex.in_pass=1;
    assert(!vk_texture_query_image_record_locked(&ex,query) && !barriers && !pending.fn);
    ex.in_pass=0;
    stale=1;
    assert(!vk_texture_query_image_current_locked(query) && !vk_texture_query_image_record_locked(&ex,query));
    stale=0;
    read.ordinal+=2;
    assert(!vk_texture_query_image_record_locked(&ex,query));
    read.ordinal-=2;
    sealed=1;
    OK(rc_registry_declare_through(texture_registrations,(RcRegistryPosition){read.scene,read.ordinal+1}));
    assert(!rc_registry_provisional_read_current(texture_registrations,read));
    assert(vk_texture_query_image_record_locked(&ex,query) && vk_texture_query_image_record_locked(&ex,query) &&
           barriers==1 && ex.recorded==21);
    assert(vk_texture_query_image_view(&ex,query,&actual)==carrier_view && actual==carrier_image && actual!=original_image);
    VkExec other=ex;
    other.cb=HANDLE(VkCommandBuffer,40);
    assert(!vk_texture_query_image_view(&other,query,&actual) && !actual &&
           !vk_texture_query_image_record_locked(&other,query));
    vk_texture_query_image_release(&query);
    assert(!query && destroys==3 && !view_destroys && !host_releases);
    assert(!pending_disposable(pending,8,9,0)); // Handed but incomplete commands retain the image and view.
    settle(8,9,1);
    assert(destroys==4 && view_destroys==1 && host_releases==1);

    assert(vk_texture_query_image_create_locked(6,&sample,&receipt,&query)==VK_SUCCESS);
    vk_texture_query_image_release(&query); // Unrecorded optional preparation.
    assert(destroys==5 && view_destroys==2 && host_releases==2 && !pending.fn);
    assert(vk_texture_query_image_create_locked(6,&sample,&receipt,&query)==VK_SUCCESS &&
           vk_texture_query_image_record_locked(&ex,query));
    vk_texture_query_image_release(&query);
    settle(8,8,0); // Never handed.
    assert(destroys==6 && view_destroys==3 && host_releases==3);
    assert(vk_texture_query_image_create_locked(6,&sample,&receipt,&query)==VK_SUCCESS &&
           vk_texture_query_image_record_locked(&ex,query));
    settle(9,9,0); // Native completion can also precede release of the preparation owner.
    assert(destroys==6);
    vk_texture_query_image_release(&query);
    assert(destroys==7 && view_destroys==4 && host_releases==4);
    cold_cases();
    rc_registry_destroy(texture_registrations);
    historical_cases();
    puts("Query carriers preserve resident, cold, associated and retained historical geometry through native retirement");
}

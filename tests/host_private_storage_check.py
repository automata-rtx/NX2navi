"""Exercise production host-private storage shape, preparation and issue rules."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile


def function(source, name):
    match = re.search(r"^(?:static )?[A-Za-z_][\w \t*]*\b" + re.escape(name) + r"\([^;{}]*\)\s*\{", source, re.M)
    if not match:
        raise ValueError(name)
    depth = 1
    end = match.end()
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


DEFINITIONS = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include "resource_core_registry.h"
#include "resource_core_backing.h"

typedef struct {VkDeviceMemory memory; uint64_t offset,size;} VkImageMemoryPlacement;

typedef void *VmaAllocation;
static int texture_image_placement(VmaAllocation allocation,VkImageMemoryPlacement *placement) {
    assert(allocation);
    *placement=(VkImageMemoryPlacement){(VkDeviceMemory)(uintptr_t)allocation,0,4096};
    return 1;
}

enum {TEXTURE_BATCH_RECORDING, TEXTURE_BATCH_SEALED, TEXTURE_BATCH_HANDED, TEXTURE_BATCH_ABANDONED};
enum {NVN_TEXTURE_TARGET_2D = 1, NVN_TEXTURE_TARGET_3D = 3, HOST_STORAGE_ZERO = 1, HOST_STORAGE_COW = 2};
typedef struct {uint32_t refs, handover;} TextureUsePartDisposition;
typedef struct {TextureUsePartDisposition *part;} TextureUseBatch;
typedef struct { int target, width, height, depth, levels, samples, format, linear; } nvn_texdesc;
typedef struct {
    unsigned set_layers, set_dsmode, set_target, set_format, set_levels;
    unsigned target, format, base_level, min_layer, nlayers;
} view_t;
typedef struct {
    VkImage image;
    VkImageType type;
    VkImageViewType view_type;
    VkFormat format;
    VkImageAspectFlags aspect;
    VkImageUsageFlags usage;
    uint64_t addr;
    uint32_t bound, width, height, depth, layers, levels, samples, scale, skip, block_views;
    unsigned mirrored, fb_learned, preserve_pool_bytes, gpu_owned;
} VkTexImage;
typedef struct { uint64_t addr; } TexKey;
typedef struct {
    VkTexImage pub;
    TexKey key;
    struct {nvn_texdesc d; struct {int layers; uint64_t size;} L; uint64_t serial;} t;
    struct {VkFormat ideal, vk; unsigned expand_rgb, astc_bw;} f;
    void *alloc, *material, *bound_to, *pending_write, *image_host_backing;
    unsigned pin_detached, image_device_count, image_private_mapping_count;
    unsigned host_storage_init, pin_owned, gpu_dirty;
    void *snap;
    uint64_t content_revision;
    RcRegistryWrite image_authority, *image_mapped_holds;
    uint32_t image_mapped_hold_count;
    TextureUsePartDisposition *host_storage_init_part;
    RcRegistryPosition sample_general_inline;
    uint64_t sample_general_upload, sample_general_revision;
} Tex;
typedef struct { uint64_t mapping, protection, lineage; uint32_t generation; } TextureSourceStamp;
typedef struct { uint32_t count; VkBufferImageCopy *regions; } TextureRecoveryCopy;
typedef struct { VkBuffer buffer; } VkPreparedStaging;
typedef struct {
    TextureRecoveryCopy copy;
    VkPreparedStaging staging;
    RcRegistryGpuCapture source;
    TextureSourceStamp stamp;
    uint32_t references, range_count;
    int recorded;
    RcSourceRange ranges[2];
} TextureStorageCpuLoad;
typedef struct {
    VkImageSubresourceRange volume_mip,word_alias_plane;
    RcRegistryPosition read;
    VkImage image;
    void *hold, *host_pin;
    uint32_t mapping_count;
    RcInterval mapping[2];
    uint64_t host_layout;
    TextureStorageCpuLoad *host_cpu;
    view_t view;
    int has_view;
} VkTextureStorageTarget;
typedef struct { VkImageSubresourceRange load_planes; } VkTextureAliasRead;
typedef struct { void *hold; VkImage image; VkTextureAliasRead *alias; } TextureNativeUse;
typedef struct { int replacement; } ResourceImageTraceDetails;

static Tex image;
static TextureUseBatch active;
static TextureUsePartDisposition part, other_part;
static uint64_t active_scene, upload_dependency;
static RcInterval spans[2];
static int mapped, exact, image_published, host_owned, pin_written, pin_used, native_used, prepared;
static int late_store, protection_changed, registry_blocked, small_shared_page;
static uint32_t watched_generation, copies, barriers, completion_refs, mapped_releases, raw_releases, retired;
static VkCommandBuffer command = (VkCommandBuffer)(uintptr_t)4451;
static uint64_t source_id = 377;
static void *posted_owner;
static void (*posted_disposition)(void *, void *, int);
static const int by_key;
#define GENERAL_RW (VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT)
#define ALL_RW VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, GENERAL_RW
static int fills_no_page(const Tex *entry) { assert(entry==&image); return small_shared_page; }
static int texture_source_stamp_valid(const TextureSourceStamp *stamp) {
    return stamp->mapping && stamp->protection && stamp->lineage && stamp->generation;
}
static int mem_mapping_range_unchanged(uint64_t addr, uint64_t bytes, uint64_t stamp) {
    assert(addr==image.pub.addr && bytes==image.t.L.size && stamp==3);
    return mapped;
}
static int mem_protection_range_unchanged(uint64_t addr, uint64_t bytes, uint64_t stamp) {
    assert(addr==image.pub.addr && bytes==image.t.L.size && stamp==4);
    return !protection_changed;
}
static int vk_mem_changed(uint64_t addr, uint64_t bytes, uint32_t cutoff, uint32_t *now) {
    assert(addr==image.pub.addr && bytes==image.t.L.size && cutoff==watched_generation);
    *now=cutoff+9;
    return late_store;
}
RcRegistry *rc_registry_runtime(void) { return (RcRegistry *)(uintptr_t)12; }
RcRegistryResult rc_registry_gpu_capture_read_at(RcRegistry *registry, RcRegistryGpuCapture source,
    const RcSourceRange *ranges, uint32_t count, RcRegistryPosition read) {
    assert(registry==rc_registry_runtime() && source.id==source_id && ranges && count==2 &&
           read.scene==51 && read.ordinal==12);
    return registry_blocked ? RC_REGISTRY_BLOCKED : RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_gpu_capture_end(RcRegistry *registry, RcRegistryGpuCapture *source) {
    assert(registry==rc_registry_runtime() && source->id==source_id);
    *source=(RcRegistryGpuCapture){0};
    return RC_REGISTRY_OK;
}
static void storage_host_cpu_load_release(void *load) { assert(load); }
static void storage_host_cpu_load_cancel(void *load,int possible) { (void)possible; assert(load); }
static void vk_completion_add_owned(uint64_t fence,void (*done)(void *),void (*cancel)(void *,int),void *load) {
    assert(fence==51 && done==storage_host_cpu_load_release && cancel==storage_host_cpu_load_cancel && load);
    completion_refs++;
}
static void barrier(VkCommandBuffer cb,VkPipelineStageFlags2 src,VkAccessFlags2 sa,VkPipelineStageFlags2 dst,VkAccessFlags2 da) {
    assert(cb==command);
    assert((barriers==0 && src==VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT && sa==GENERAL_RW &&
            dst==VK_PIPELINE_STAGE_2_COPY_BIT && da==VK_ACCESS_2_TRANSFER_WRITE_BIT) ||
           (barriers==1 && src==VK_PIPELINE_STAGE_2_COPY_BIT && sa==VK_ACCESS_2_TRANSFER_WRITE_BIT &&
            dst==VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT && da==GENERAL_RW));
    barriers++;
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer buffer,VkImage destination,
    VkImageLayout layout,uint32_t count,const VkBufferImageCopy *regions) {
    assert(cb==command && buffer==(VkBuffer)(uintptr_t)17 && destination==image.pub.image &&
           layout==VK_IMAGE_LAYOUT_GENERAL && count==2 && regions && barriers==1);
    copies++;
}
static void *texture_use_part_retain(TextureUseBatch *batch) {
    assert(batch==&active && batch->part==&part);return batch->part;
}
static void texture_use_part_release(void *body) { assert(body==&part); }
static int texture_use_part_known_unhanded(const void *body) { assert(body==&part);return 1; }
static void resource_effects_retain_part_disposition(void *body,int (*unhanded)(const void *),
    void *owner,void (*release)(void *,void *,int)) {
    assert(body==&part && unhanded==texture_use_part_known_unhanded && owner && release && !posted_owner);
    posted_owner=owner;
    posted_disposition=release;
}
static Tex *nvn_hm_get(const int *map,const TexKey *key) {
    assert(map==&by_key && key->addr==image.key.addr);
    return &image;
}
RcRegistryResult rc_registry_image_hold_release_extents(RcRegistry *registry, const RcRegistryWrite *holds,
                                                         uint32_t count,int cpu_visible) {
    assert(registry==rc_registry_runtime() && holds==image.image_mapped_holds && count==2 && cpu_visible);
    mapped_releases++;
    return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_hold_release(RcRegistry *registry,RcRegistryWrite claim,int cpu_visible) {
    assert(registry==rc_registry_runtime() && claim.id==image.image_authority.id && cpu_visible);
    raw_releases++;
    return RC_REGISTRY_OK;
}
static void destroy_tex(Tex *entry) {
    assert(entry==&image && !entry->image_mapped_holds && !entry->image_authority.id);
    retired++;
}
static void gpu_written_prepared_at(Tex *entry,const uint32_t *cutoff) {
    assert(entry==&image && cutoff && *cutoff==watched_generation);
    assert(!entry->pub.gpu_owned && !entry->image_authority.id && !entry->image_mapped_holds);
    entry->pub.gpu_owned=entry->gpu_dirty=1;
    entry->content_revision++;
    entry->image_authority=(RcRegistryWrite){241, 9};
    entry->image_mapped_holds=calloc(2,sizeof *entry->image_mapped_holds);
    assert(entry->image_mapped_holds);
    entry->image_mapped_hold_count=2;
}
static void rt_fatal(const char *reason) { (void)reason; assert(0); }
enum { GFX_RES_UBO, GFX_RES_SSBO, NVN_STAGE_CS=0, GFX_STAGE_CS=0 };
typedef struct { unsigned slot; } GfxShaderRes;
typedef struct { const GfxShaderRes *res; unsigned kind, stage, type, binding, count; } VkPipeBinding;
typedef struct { unsigned n_bindings; VkPipeBinding bindings[2]; } VkPipeInfo;
typedef struct { uint64_t addr, size; } p_bind_buf;
typedef struct { p_bind_buf ubo[1][18]; } VkExec;
typedef struct { const VkPipeInfo *info; } VkExecDispatch;
typedef struct { VkTextureStorageTarget *independent; } EffectImageCapture;
typedef struct { uint64_t address, bytes, alignment; } UniformAddressRange;
typedef struct { int unused; } MemBackingCapture, VkMemSpan;
static struct { uint64_t ubo_max; } LIM;
static MemBackingCapture host_input;
static RcInterval uniform_mapping;
static int uniform_unmapped, uniform_provider_mismatch;
typedef enum {VK_USNAP_RESOLVE_READY, VK_USNAP_RESOLVE_NATIVE_ONLY, VK_USNAP_RESOLVE_ORDERED_PRODUCER} VkUsnapResolveResult;
static VkUsnapResolveResult uniform_provider;
static RcRegistryPosition vk_exec_resource_position(void) {return (RcRegistryPosition){51,12};}
static VkUsnapResolveResult vk_usnap_resolve_at(uint64_t address,uint64_t bytes,uint64_t alignment,RcRegistryPosition read,
                                               void *descriptor,void *lease) {
    assert(address==uniform_mapping.address && bytes==uniform_mapping.range.size && alignment==16 &&
           read.scene==51 && read.ordinal==12 && !descriptor && !lease);
    return uniform_provider;
}
static RcRegistryResult vk_mem_canonical_source_qualify(const MemBackingCapture *source,uint64_t address,uint64_t bytes) {
    assert(source==&host_input && address==uniform_mapping.address && bytes==uniform_mapping.range.size);
    return uniform_provider_mismatch ? RC_REGISTRY_STALE : RC_REGISTRY_OK;
}
static const char *uniform_refusal;
static int nvn_gpu_only(uint64_t address) { return address==0x100000000; }
static int vk_mem_lookup(uint64_t address,uint64_t bytes,VkMemSpan *span) {
    assert(address==uniform_mapping.address && bytes==uniform_mapping.range.size && span);
    return !uniform_unmapped;
}
static uint32_t vk_mem_device_intervals(const VkMemSpan *span,uint64_t address,uint64_t bytes,RcInterval **out) {
    assert(span && address==uniform_mapping.address && bytes==uniform_mapping.range.size);
    *out=malloc(sizeof **out);
    assert(*out);
    **out=uniform_mapping;
    return 1;
}
static RcBackingResult mem_backing_capture(uint64_t address,uint64_t bytes,MemBackingCapture **out) {
    assert(address==uniform_mapping.address && bytes==uniform_mapping.range.size);
    if (uniform_unmapped) return RC_BACKING_STALE;
    *out=&host_input;
    return RC_BACKING_OK;
}
static void mem_backing_release(MemBackingCapture **capture) {
    assert(!*capture || *capture==&host_input);
    *capture=NULL;
}
static void limits_init(void) {LIM.ubo_max=4096;}
static int uniform_binding_range(const VkExec *ex,const VkPipeBinding *binding,unsigned stage,
                                 unsigned member,UniformAddressRange *range) {
    const unsigned slot=binding->res->slot+member-3;
    assert(stage==NVN_STAGE_CS && slot<18);
    *range=(UniformAddressRange){ex->ubo[stage][slot].addr,ex->ubo[stage][slot].size,16};
    return range->address && range->bytes;
}
static int dispatch_storage_conflict(VkExecDispatch *dispatch,const EffectImageCapture *output,const char *reason,
                                     VkImage image_id,unsigned binding,int member) {
    assert(dispatch && output && !image_id && binding==8 && member>=-1);
    uniform_refusal=reason;
    return 0;
}
static int __attribute__((unused)) dispatch_storage_input_conflict(VkExecDispatch *dispatch,
    const EffectImageCapture *output,const char *reason,const char *detail,const VkPipeBinding *binding,
    int member,uint64_t handle,VkImage image_id,const EffectImageCapture *effect,const void *sample,
    const VkTexImage *tested) {
    (void)detail; (void)handle; (void)effect; (void)sample; (void)tested;
    return dispatch_storage_conflict(dispatch,output,reason,image_id,binding->binding,member);
}
static int nvn_tex_is_linear(const nvn_texdesc *d) { return d->linear; }
static int format_compressed(VkFormat format) { return format == VK_FORMAT_BC1_RGBA_UNORM_BLOCK; }
static int nvn_fmt_bpb(int fmt, int *width, int *height) { *width = *height = 1; return fmt == 1; }
static uint64_t vk_upload_dependency(void) { return upload_dependency; }
static TextureUseBatch *texture_use_active(uint64_t scene, int create) {
    (void)create;
    return active_scene == scene ? &active : 0;
}
static int mem_backing_current(void *identity) {
    return identity == &host_input ? !uniform_unmapped : mapped && identity == spans;
}
static const RcInterval *mem_backing_intervals(void *identity, uint32_t *count) {
    if (identity == &host_input) {
        *count=1;
        return &uniform_mapping;
    }
    assert(identity == spans);
    *count = 2;
    return spans;
}
static void lock(void) {}
static void unlock(void) {}
static Tex *storage_target_current_locked(const VkTextureStorageTarget *target, int issued) {
    (void)issued;
    return exact && mapped && target && target->image == image.pub.image ? &image : NULL;
}
static VkImage vk_texture_storage_target_image(const VkTextureStorageTarget *target, const VkTextureAliasRead *seed) {
    return target && !seed ? target->image : VK_NULL_HANDLE;
}
static int image_alias_load_current(Tex *entry, const VkImageSubresourceRange *planes, RcRegistryPosition read) {
    (void)entry; (void)planes; (void)read; assert(0); return 0;
}
static void texture_use_pin(Tex *entry, void *pin) {
    assert(entry == &image && pin == &part);
    pin_used++;
}
static void texture_native_use_batch(TextureUseBatch *batch, const TextureNativeUse *choice) {
    assert(batch == &active && choice->hold == &image && choice->image == image.pub.image && !choice->alias);
    native_used++;
}
static void resource_image_prepare_pinned(uint64_t identity, uint64_t addr, uint64_t size,
    const RcInterval *mapping, uint32_t count, int detached) {
    assert(identity == (uint64_t)(uintptr_t)image.pub.image && addr == image.pub.addr &&
           size == image.t.L.size && count == 2 && !detached);
    assert(!memcmp(spans, mapping, sizeof spans));
    prepared++;
}
static void resource_image_written(uint64_t identity, uint64_t addr) {
    assert(identity == (uint64_t)(uintptr_t)image.pub.image && addr == image.pub.addr);
    image_published++;
}
static void vk_texture_storage_bound_locked(const VkTexImage *entry) {
    assert(entry == &image.pub);
    host_owned++;
}
static void explicit_written(Tex *entry) { assert(entry == &image); pin_written++; }
static void resource_effects_image_trace_details(void *effects, uint64_t image_id, ResourceImageTraceDetails details) {
    (void)effects; (void)image_id; (void)details; assert(0);
}
static void image_alias_promote_target_locked(VkTextureAliasRead *seed, Tex *entry) {
    (void)seed; (void)entry; assert(0);
}
static int image_alias_whole_range(const VkTexImage *entry, const VkImageSubresourceRange *planes) {
    (void)entry; (void)planes; assert(0); return 0;
}
static void image_alias_replacement_record_part(VkTextureAliasRead *seed, TextureUseBatch *batch) {
    (void)seed; (void)batch; assert(0);
}
static int image_alias_private_planes_written_locked(Tex *entry, const VkImageSubresourceRange *planes, RcRegistryPosition output) {
    (void)entry; (void)planes; (void)output; assert(0); return 0;
}
static int image_alias_planes_written_locked(Tex *entry, const VkImageSubresourceRange *planes) {
    (void)entry; (void)planes; assert(0); return 0;
}
'''


CASES = r'''
static void dimensions(int volume) {
    image = (Tex){
        .pub = {.image = (VkImage)(uintptr_t)99, .addr = 0x80018000,
            .type = volume ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
            .view_type = volume ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R8G8_UNORM, .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
            .width = 64, .height = 36, .depth = volume ? 48 : 1,
            .levels = 3, .layers = 1, .samples = 1, .scale = 1,
            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT},
        .t = {.d = {.target = volume ? NVN_TEXTURE_TARGET_3D : NVN_TEXTURE_TARGET_2D, .format = 1,
                    .width = 64, .height = 36, .depth = volume ? 48 : 1, .levels = 3, .samples = 1},
              .L = {.layers = 1, .size = 27648}},
        .f = {.ideal = VK_FORMAT_R8G8_UNORM, .vk = VK_FORMAT_R8G8_UNORM},
        .alloc = &image, .image_host_backing = spans, .host_storage_init = HOST_STORAGE_ZERO,
        .sample_general_revision = 83, .sample_general_upload = 19,
    };
}

static void layout_and_init(void) {
    const RcRegistryPosition read = {51, 12};
    upload_dependency = 18;
    assert(!storage_host_initialized(&image, read));
    upload_dependency = 19;
    assert(storage_host_initialized(&image, read));
    image.host_storage_init = 0;
    assert(!storage_host_initialized(&image, read)); // a layout transition has no texels
    image.host_storage_init = HOST_STORAGE_ZERO;
    image.sample_general_upload = 0;
    image.sample_general_inline = (RcRegistryPosition){51, 11};
    image.host_storage_init_part = &part;
    active_scene = 51;
    active.part = &part;
    part.handover = TEXTURE_BATCH_RECORDING;
    assert(storage_host_initialized(&image, read));
    assert(!storage_host_initialized(&image, (RcRegistryPosition){51, 10}));
    assert(!storage_host_initialized(&image, (RcRegistryPosition){52, 2}));
    active.part = &other_part;
    assert(!storage_host_initialized(&image, read));
    part.handover = TEXTURE_BATCH_SEALED;
    assert(!storage_host_initialized(&image, (RcRegistryPosition){52, 2}));
    part.handover = TEXTURE_BATCH_ABANDONED;
    assert(!storage_host_initialized(&image, (RcRegistryPosition){90, 2}));
    part.handover = TEXTURE_BATCH_HANDED;
    assert(storage_host_initialized(&image, (RcRegistryPosition){52, 2}));
    image.sample_general_revision = 0;
    assert(!storage_host_initialized(&image, (RcRegistryPosition){52, 2}));
}

static void shape_and_use(int volume) {
    dimensions(volume);
    active_scene = 51;
    exact = mapped = 1;
    const uint32_t target_type = volume ? NVN_TEXTURE_TARGET_3D : NVN_TEXTURE_TARGET_2D;
    const view_t view = {.set_levels = 1, .base_level = 1, .set_target = 1, .target = target_type};
    VkImageSubresourceRange plane;
    assert(storage_host_shape(&image, &view, 1, &plane));
    assert(plane.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT && plane.baseMipLevel == 1 &&
           plane.levelCount == 1 && plane.baseArrayLayer == 0 && plane.layerCount == 1);
    VkTextureStorageTarget held = {.host_layout = 83, .view = view, .has_view = 1, .read = {51, 12},
                                    .image = image.pub.image, .hold = &image, .mapping_count = 2};
    memcpy(held.mapping, spans, sizeof spans);
    if (volume) held.volume_mip = plane;
    view_t wrong_view = view;
    wrong_view.target = volume ? NVN_TEXTURE_TARGET_2D : NVN_TEXTURE_TARGET_3D;
    assert(!storage_host_shape(&image, &wrong_view, 1, &plane));
    wrong_view = view;
    wrong_view.set_layers = wrong_view.min_layer = 1;
    assert(!storage_host_shape(&image, &wrong_view, 1, &plane));
    wrong_view = view;
    wrong_view.base_level = image.pub.levels;
    assert(!storage_host_shape(&image, &wrong_view, 1, &plane));
    if (volume) {
        const unsigned depth = image.pub.depth;
        image.pub.depth--;
        assert(!storage_host_shape(&image, &view, 1, &plane));
        image.pub.depth = depth;
    }
    assert(storage_host_shape(&image, &view, 1, &plane));
    VkImageSubresourceRange wrong_plane = plane;
    wrong_plane.baseMipLevel = 0;
    assert(!vk_texture_storage_target_load_current(&held, &wrong_plane));
    assert(!vk_texture_storage_target_written(&held, NULL, &wrong_plane));
    assert(!vk_texture_storage_target_written(&held, NULL, NULL));
    uint32_t count = 0;
    assert(storage_host_mapping(&image, &count) == spans && count == 2);
    image.material = &image;
    assert(!storage_host_mapping(&image, &count) && count == 0);
    image.material = NULL;
    image.bound_to = &image;
    assert(!storage_host_mapping(&image, &count));
    image.bound_to = NULL;
    mapped = 0;
    assert(!storage_host_mapping(&image, &count));
    assert(!vk_texture_storage_target_load_current(&held, &plane));
    assert(!vk_texture_storage_target_written(&held, NULL, &plane));
    assert(image_published == volume && host_owned == volume);
    mapped = 1;
    assert(vk_texture_storage_target_load_current(&held, &plane));
    assert(!vk_texture_storage_target_use(&held, NULL, 52, VK_NULL_HANDLE));
    assert(vk_texture_storage_target_use(&held, NULL, 51, VK_NULL_HANDLE));
    assert(vk_texture_storage_target_read_use(&held, &plane, 51));
    assert(image_published == volume); // preparing and retaining does not write
    assert(vk_texture_storage_target_written(&held, NULL, &plane));
    assert(image_published == volume + 1 && host_owned == volume + 1);
    if (!volume) {
        image.pub.mirrored = 1; // an already adopted, fully seeded and still retained material-free pin
        image.host_storage_init = HOST_STORAGE_COW;
        held.host_pin = &part;
        assert(vk_texture_storage_target_load_current(&held, &plane));
        assert(vk_texture_storage_target_use(&held, NULL, 51, VK_NULL_HANDLE));
        assert(vk_texture_storage_target_written(&held, NULL, &plane));
        assert(image_published == 1 && host_owned == 1 && pin_written == 1 && pin_used == 1);
    }
}
static void mirrored_cpu_load(void) {
    dimensions(0);
    image.pub.mirrored=1;
    image.host_storage_init=0;
    image.key.addr=image.pub.addr;
    image.t.serial=135;
    active.part=&part;
    watched_generation=13;
    upload_dependency=image.sample_general_upload;
    const view_t view={.set_levels=1,.base_level=1};
    VkBufferImageCopy regions[2]={0};
    TextureStorageCpuLoad load={.copy={.count=2,.regions=regions},
        .staging={.buffer=(VkBuffer)(uintptr_t)17},.source={source_id,1},
        .stamp={.mapping=3,.protection=4,.lineage=5,.generation=watched_generation},
        .references=1,.range_count=2,.ranges={spans[0].range,spans[1].range}};
    VkTextureStorageTarget held={.image=image.pub.image,.hold=&image,.read={51,12},
        .host_layout=image.sample_general_revision,.host_cpu=&load,.view=view,.has_view=1,.mapping_count=2};
    memcpy(held.mapping,spans,sizeof spans);
    VkImageSubresourceRange planes;
    assert(storage_host_shape(&image,&view,1,&planes));
    assert(vk_texture_storage_target_host_mirror(&held));
    assert(vk_texture_storage_target_load_current(&held,&planes));
    assert(!vk_texture_storage_target_written(&held,NULL,&planes)); // staging owns bytes, no command yet
    late_store=1;
    assert(!vk_texture_storage_target_load_current(&held,&planes));
    assert(!vk_texture_storage_target_use(&held,NULL,51,command));
    late_store=0;
    protection_changed=1;
    assert(!vk_texture_storage_target_load_current(&held,&planes));
    protection_changed=0;
    registry_blocked=1;
    assert(!vk_texture_storage_target_load_current(&held,&planes));
    registry_blocked=0;
    small_shared_page=1;
    assert(!vk_texture_storage_target_load_current(&held,&planes));
    small_shared_page=0;
    mapped=0;
    assert(!vk_texture_storage_target_load_current(&held,&planes));
    mapped=1;
    image.pub.gpu_owned=1;
    assert(!vk_texture_storage_target_load_current(&held,&planes));
    image.pub.gpu_owned=0;
    RcInterval input={.address=0x80048000,.mapping=200,.range={spans[0].range.backing,spans[0].range.offset,32}};
    assert(!vk_texture_storage_target_disjoint_input(&held,&input,1,(VkImageMemoryPlacement){0}));
    input.range.offset+=spans[0].range.size;
    assert(vk_texture_storage_target_disjoint_input(&held,&input,1,(VkImageMemoryPlacement){0}));
    input.mapping=0;
    assert(vk_texture_storage_target_disjoint_input(&held,&input,1,(VkImageMemoryPlacement){0}));
    assert(!dispatch_storage_mirror_input_disjoint(&held,&input,1,(VkImageMemoryPlacement){0}));
    input.mapping=200;
    assert(dispatch_storage_mirror_input_disjoint(&held,&input,1,(VkImageMemoryPlacement){0}));
    input.range.backing=RC_GUEST_COMPATIBILITY_BACKING;
    assert(!dispatch_storage_mirror_input_disjoint(&held,&input,1,(VkImageMemoryPlacement){0}));
    input.range.backing=7;
    input.address=UINT64_MAX-15;
    assert(!dispatch_storage_mirror_input_disjoint(&held,&input,1,(VkImageMemoryPlacement){0}));
    GfxShaderRes descriptor={.slot=4};
    VkPipeInfo pipe={.n_bindings=1,
        .bindings={{&descriptor,GFX_RES_UBO,GFX_STAGE_CS,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,8,1}}};
    VkExecDispatch dispatch={.info=&pipe};
    EffectImageCapture output={.independent=&held};
    VkExec ex={.ubo={{[1]={0x80048000,32}}}};
    uniform_mapping=(RcInterval){.address=ex.ubo[0][1].addr,.mapping=24,.range={12,0x4000,32}};
    assert(dispatch_storage_mirror_buffers_disjoint(&ex,&dispatch,&output));
    uniform_mapping.range=spans[0].range;
    uniform_mapping.range.size=32;
    assert(!dispatch_storage_mirror_buffers_disjoint(&ex,&dispatch,&output) &&
           !strcmp(uniform_refusal,"host-storage-uniform-mapping"));
    uniform_mapping.range.backing=15;
    uniform_provider=VK_USNAP_RESOLVE_ORDERED_PRODUCER;
    assert(!dispatch_storage_mirror_buffers_disjoint(&ex,&dispatch,&output));
    uniform_provider=VK_USNAP_RESOLVE_NATIVE_ONLY;
    assert(dispatch_storage_mirror_buffers_disjoint(&ex,&dispatch,&output));
    uniform_provider_mismatch=1;
    assert(!dispatch_storage_mirror_buffers_disjoint(&ex,&dispatch,&output));
    uniform_provider_mismatch=0;
    ex.ubo[0][1].addr=uniform_mapping.address=0x100000000;
    assert(dispatch_storage_mirror_buffers_disjoint(&ex,&dispatch,&output));
    uniform_unmapped=1;
    assert(!dispatch_storage_mirror_buffers_disjoint(&ex,&dispatch,&output));
    uniform_unmapped=0;
    descriptor.slot=1;
    assert(!dispatch_storage_mirror_buffers_disjoint(&ex,&dispatch,&output) &&
           !strcmp(uniform_refusal,"host-storage-program-constants"));
    descriptor.slot=4;
    pipe.bindings[0].kind=GFX_RES_SSBO;
    assert(!dispatch_storage_mirror_buffers_disjoint(&ex,&dispatch,&output) &&
           !strcmp(uniform_refusal,"host-storage-buffer-input"));
    assert(!vk_texture_storage_target_use(&held,NULL,51,VK_NULL_HANDLE));
    assert(vk_texture_storage_target_use(&held,NULL,51,command));
    assert(copies==1 && barriers==2 && completion_refs==1 && load.references==2);
    assert(!vk_texture_storage_target_use(&held,NULL,51,command));
    late_store=1; // the already staged and recorded bytes had the earlier cut
    assert(vk_texture_storage_target_written(&held,NULL,&planes));
    late_store=0;
    assert(!load.source.id && posted_owner && image.pub.gpu_owned && image.gpu_dirty);
    FirstMirroredWrite *stale=malloc(sizeof *stale);
    assert(stale);
    *stale=(FirstMirroredWrite){image.key,image.pub.image,image.t.serial,image.content_revision-1};
    first_mirrored_write_dispose(&part,stale,1);
    assert(!mapped_releases && !raw_releases && !retired);
    posted_disposition(&part,posted_owner,1);
    posted_owner=NULL;
    assert(mapped_releases==1 && raw_releases==1 && retired==1);
}

int main(void) {
    spans[0] = (RcInterval){.address = 0x80018000, .mapping = 9, .range = {5, 0x8000, 16384}};
    spans[1] = (RcInterval){.address = 0x8001c000, .mapping = 11, .range = {7, 0x4000, 11264}};
    dimensions(0);
    layout_and_init();
    shape_and_use(0);
    shape_and_use(1);
    assert(native_used == 4 && prepared == 3);
    mirrored_cpu_load();
    puts("host-private storage: volume/pin and first mirrored CPU authority, inline staging, normalized overlap and known-unhanded rollback passed");
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument("--cc", default=os.environ.get("CC", "clang"))
    args = parser.parse_args()
    source = (args.sources / "vk_texture.c").read_text(encoding="utf-8")
    funcs = (
        "storage_volume_shape", "storage_volume_mip", "storage_target_volume_view",
        "storage_host_initialized", "storage_host_mapping", "storage_host_cpu_candidate", "storage_host_cpu_load_current",
        "storage_host_shape", "storage_host_planes_match", "storage_host_planes_current",
        "vk_texture_storage_target_load_current", "vk_texture_storage_target_host_mirror", "texture_placement_disjoint", "vk_texture_storage_target_disjoint_input",
        "vk_texture_storage_target_use", "vk_texture_storage_target_read_use",
        "first_mirrored_write_dispose", "first_mirrored_write_recorded", "vk_texture_storage_target_written",
    )
    with tempfile.TemporaryDirectory() as work:
        translation = Path(work) / "check.c"
        binary = Path(work) / "check.exe"
        start = source.index("typedef struct {\n    TexKey key;\n    VkImage image;\n    uint64_t serial, revision;\n} FirstMirroredWrite;")
        end = source.index("} FirstMirroredWrite;", start) + len("} FirstMirroredWrite;")
        draw = (args.sources / "vk_exec_draw.c").read_text(encoding="utf-8")
        draw_funcs = ("dispatch_storage_mirror_input_disjoint", "dispatch_storage_mirror_uniform_disjoint",
                      "dispatch_storage_mirror_buffers_disjoint")
        translation.write_text("\n\n".join((DEFINITIONS, source[start:end], *(function(source, name) for name in funcs),
                                           *(function(draw, name) for name in draw_funcs), CASES)))
        command = [args.cc, "-std=gnu11", "-Wall", "-Wextra", "-Werror", "-Wno-sign-compare",
                   "-fsanitize=undefined", "-fsanitize-trap=all"]
        subprocess.run([*command, "-I", str(args.headers), str(translation), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()

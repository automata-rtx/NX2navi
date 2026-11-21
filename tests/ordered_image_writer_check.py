"""Check exact transfer eligibility and deferred installation of a mixed LOAD attachment."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from present_source_check import ROOT, function, functions, section


def fixture(root):
    tex, xfer, replay, draw = ((root / name).read_text() for name in
        ('vk_texture.c', 'vk_exec_xfer.c', 'vk_exec.c', 'vk_exec_draw.c'))
    for entry, capture in (('prepare_ordered_clear', 'vk_exec_clear_capture'),
                           ('prepare_ordered_buffer_to_texture', 'vk_exec_buffer_to_texture_capture'),
                           ('prepare_ordered_texture_to_texture', 'vk_exec_texture_to_texture_capture')):
        source = function(replay, entry)
        assert source.index('image_write_input_position(ex)') < source.index('prepare_bound_texture_relocation')
        assert source.index('prepare_bound_texture_relocation') < source.index(capture)
    source = function(replay, 'vk_exec_graphics_pass_capture_seeded')
    assert (source.index('capture_attachment_load_seeds(ex, allow_ordered, 1, seeds, failure)')
            < source.index('if (result != VK_EXEC_GRAPHICS_PASS_READY) return result;')
            < source.index('graphics_pass_capture(ex, allow_ordered, 0, seeds, out, failure)'))
    source = function(replay, 'capture_attachment_load_seeds')
    assert (source.index('if (relocate_color &&')
            < source.index('prepare_bound_texture_relocation(ex, target, allow_ordered)')
            < source.index('vk_texture_alias_target_capture(target, view, layers, &seeds[i], allow_ordered, &refused)'))
    source = function(xfer, 'vk_exec_clear_commit')
    assert source.index('vkCmdClearColorImage') < source.index('vk_texture_write_target_clear_written')
    source = function(xfer, 'clear_tex')
    assert source.index('vkCmdClearColorImage') < source.index('vk_texture_alias_clear_written')
    assert source.index('vkCmdClearDepthStencilImage') < source.index('vk_texture_alias_clear_written')
    assert source.index('vk_texture_gpu_written') < source.index('vk_texture_alias_clear_written')
    source = function(tex, 'write_target_finish')
    assert source.index('write_target_recorded') < source.index('image_alias_planes_written_locked')
    assert source.index('image_alias_planes_written_locked') < source.index('nvn_texture_pin_release')
    source = function(xfer, 'vk_exec_texture_to_texture_compat')
    assert source.index('image_copy_record') < source.index('vk_texture_alias_transfer_written')
    source = function(draw, 'graphics_record_state')
    assert source.index('vk_exec_targets_prepare(ex)') < source.index('if (vk_exec_recording_failed())')
    source = function(replay, 'clear_attachment')
    assert (source.index('resource_attachment_prepare(ex, slot)') < source.rindex('vk_exec_recording_failed()')
            < source.index('vkCmdClearAttachments(') < source.index('resource_attachment_written(ex, slot)'))
    source = function(draw, 'storage_images_written')
    assert source.index('resource_effects_emitted()') < source.index('vk_exec_targets_written(ex)')
    assert 'vk_exec_graphics_draw_release_gpa_inputs(' in function(draw, 'vk_exec_graphics_draw_cancel')
    assert 'graphics_release_gpa_inputs(' in function(draw, 'vk_exec_graphics_draw_release_gpa_inputs')
    assert 'vk_texture_alias_target_cancel(' in function(draw, 'graphics_release_gpa_inputs')
    assert 'vk_texture_alias_target_cancel(' in function(replay, 'attachment_clear_load_cancel')
    assert 'image_alias_replacement_record_part(' not in function(tex, 'vk_texture_alias_target_prepare_replacement')
    seed = function(tex, 'image_alias_read_prepare_locked')
    assert 'alias->result = receiver->image;' in seed and 'alias->placement = receiver->placement;' in seed
    assert 'alias->predecessor_hold = vk_mem_image_hold(original);' in seed
    assert '.image = e->pub' in function(tex, 'image_alias_read_capture_locked')
    assert '.usage = key->usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT' in function(tex, 'image_alias_placement_plan')
    assert '.newLayout = VK_IMAGE_LAYOUT_GENERAL' in function(tex, 'vk_texture_alias_read_upload')
    readback = function(replay, 'vk_exec_prepare_alias_readback')
    assert readback.index('vk_exec_barrier(ex);') < readback.index('vk_texture_alias_read_record(')
    capture = function(replay, 'graphics_pass_capture')
    assert capture.index('if (seed)') < capture.index('vk_texture_alias_target_image(seed)') < capture.index('vk_texture_target_loan(tex')
    assert 'if (first_seed == 9 && !after_barrier' in capture
    select = function(draw, 'graphics_draw_select')
    assert select.index('graphics_pool_prepare(') < select.index('graphics_load_peers_prepare(') < select.index('graphics_descriptors_ready(')
    plane_layout = (root / 'vk_texture_alias_bytes.h').read_text()
    clear_planes = section(plane_layout, 'enum { VK_TEXTURE_ALIAS_MAX_PLANES', '#define VK_TEXTURE_ALIAS_MAX_BYTES')
    clear_planes += section(plane_layout, 'typedef struct {\n    VkBufferImageCopy copy;', '// Every plane in the host stage')
    clear_planes += functions(plane_layout, 'texture_alias_selected_planes', 'texture_alias_clear_planes')
    return r"""
#include <assert.h>
#include <stdio.h>
#include "vk_exec.h"
#include "nvn_hmap.h"
#include "resource_image_write.h"
#include "resource_host_write.h"
#include "tex.h"
#define HANDLE(type, value) ((type)(uintptr_t)(value))
enum { VK_TEXTURE_ALIAS_MAX_BYTES = 1024*1024 };
int nvn_fmt_bpb(int f, int *w, int *h) {
    *w = *h = 1;
    return f == NVN_FORMAT_R8 ? 1 : f == NVN_FORMAT_R32UI ? 4 : 0;
}
int tex_vk_format(int f, TexVkFormat *format) {
    if (f != NVN_FORMAT_R8 && f != NVN_FORMAT_R32UI) return 0;
    *format = (TexVkFormat){.vk = f == NVN_FORMAT_R8 ? VK_FORMAT_R8_UNORM : VK_FORMAT_R32_UINT};
    format->ideal = format->vk;
    return 1;
}
typedef struct { int unused; } Sub;
static int sub_format(const tex_t *t, Sub *ignored) {
    (void)ignored;
    return t && (t->d.format == NVN_FORMAT_R8 || t->d.format == NVN_FORMAT_R32UI);
}
""" + clear_planes + functions(tex, 'vk_texture_alias_region_transfer', 'alias_depth_format',
                  'vk_texture_alias_depth_attachment_range', 'vk_texture_alias_attachment_range') + '\n' + functions(
    xfer, 'int_format', 'vk_exec_clear_alias_candidate', 'vk_exec_texture_to_texture_alias_candidate') + '\n' + section(
    tex, 'typedef struct {\n    uint64_t addr;', '// What makes two images interchangeable') + '\n' + (
        next(line for line in tex.splitlines() if line.endswith('} ImgKey;'))) + r"""
typedef struct Tex {
    tex_t t;
    VkTexImage pub;
    VmaAllocation alloc;
    uint64_t bytes, content_revision, bound_to;
    int released, superseded, overwritten, retiring, pin_owned;
    void *material, *alias_fragment_owner;
    ImgKey ik;
    RcInterval *image_private_mapping, *alias_full_demand;
    uint32_t image_private_mapping_count, alias_full_count;
    RcInterval *alias_plane_demand;
    uint32_t alias_plane_count;
    VkImageSubresourceRange alias_plane_range;
    TexKey key;
} Tex;
struct VkTextureAliasRead {
    const tex_t *receiver;
    TexKey key;
    VkTexImage result;
    VmaAllocation allocation;
    uint64_t guest_bytes, placement_bytes, predecessor_serial, predecessor_revision;
    uint32_t mapping_count, demand_count;
    RcInterval *receiver_mapping, *demands;
    VkMemoryImageHold *target_hold;
    VkImage predecessor;
    VkImageSubresourceRange load_planes;
    int uploaded, target_replaced, predecessor_dormant, upload_canceled, target_created;
};
static nvn_hmap by_key = {.klen = sizeof(TexKey)};
static int locked, original_current, output_current, gpu_emitted, ownership_replacements, image_retired, prepared;
static int native_depth_current, depth_sources_ready, depth_source_captures;
static VkTextureAliasRead depth_seed;
static const RcInterval depth_mapping = {
    .address = 0x8000000000, .mapping = 17, .range = {18, 0x400, 8192}
};
static int part_owner, part_releases, alias_retains, alias_releases, discarded;
static void (*discard)(void *, void *, int);
static void *discard_owner;
static VkImage emitted_image;
static unsigned resumes, rebinds, n_restarts;
static uint64_t alias_dormant_target_replacements;
static void lock(void) { assert(!locked); locked = 1; }
static void unlock(void) { assert(locked); locked = 0; }
int vk_texture_alias_read_current(VkTextureAliasRead *seed) {
    return original_current && seed->predecessor && !seed->target_replaced;
}
int vk_texture_render_target_load_current(VkImage actual, uint64_t address, const VkImageSubresourceRange *planes) {
    assert(planes && planes->aspectMask == VK_IMAGE_ASPECT_COLOR_BIT && planes->baseMipLevel == 1 &&
           planes->baseArrayLayer == 1 && planes->layerCount == 1);
    return output_current && actual == emitted_image && address;
}
void resource_image_prepare_mapped(uint64_t image, uint64_t address, uint64_t size,
    const RcInterval *map, uint32_t count) {
    assert(image == 87 && address && size == 8192 && map && count == 1 && !gpu_emitted);
    prepared++;
}
static int image_owner_replaced(Tex *e, VkImage next) {
    assert(locked && gpu_emitted && e->pub.image == HANDLE(VkImage, 85) && next == emitted_image);
    ++ownership_replacements;
    return 0;
}
static void compatibility_invalidate(Tex *e) { assert(locked && e); }
VkMemoryImageHold *vk_mem_image_hold(VkImage image) {
    assert(image == emitted_image);
    return HANDLE(VkMemoryImageHold *, 80);
}
int vk_exec_image_gone(VkImage old) { assert(locked && old == HANDLE(VkImage, 85)); return 1; }
void vk_defer_image_reclaim(VkImage old, VmaAllocation a, VkDeviceSize bytes) {
    assert(locked && old == HANDLE(VkImage, 85) && a == HANDLE(VmaAllocation, 90) && bytes == 8192);
    image_retired++;
}
static void tex_gen_bump(uint64_t address) { assert(locked && address); }
static void texture_resume_use(Tex *e) {
    assert(locked && gpu_emitted && e->released);
    e->released=0;
    ++resumes;
}
static void texture_rebind_object(Tex *e, const tex_t *replacement) {
    assert(locked && gpu_emitted && !e->released && e->t.serial != replacement->serial);
    e->t = *replacement;
    ++rebinds;
}
void vk_dlaa_hud_write(VkImage image) { assert(image == emitted_image); }
static void gpu_written_prepared(Tex *e) {
    assert(locked && gpu_emitted && e->pub.image == emitted_image && prepared);
    ++e->content_revision;
}
RcRegistryPosition vk_exec_resource_position(void) { return (RcRegistryPosition){2, 8}; }
static int image_alias_planes_written_locked(Tex *e, const VkImageSubresourceRange *planes) {
    assert(locked && gpu_emitted && e->pub.image == emitted_image && planes && planes->layerCount == 1);
    assert(e->image_private_mapping_count == 1 && e->alias_plane_count == 3 && !e->alias_full_count);
    assert(e->alias_plane_range.layerCount == 2 && e->pub.preserve_pool_bytes);
    output_current = 1;
    return 1;
}
typedef int TextureUseBatch;
static TextureUseBatch *texture_use_active(uint64_t scene,int create) { assert(scene==2 && create);return &part_owner; }
static void *texture_use_part_retain(TextureUseBatch *part) { assert(part==&part_owner);return part; }
static int texture_use_part_known_unhanded(const void *part) { return part==&part_owner; }
static void texture_use_part_release(void *part) { assert(part==&part_owner);part_releases++; }
VkTextureAliasRead *vk_texture_alias_read_retain(VkTextureAliasRead *seed) { assert(seed);alias_retains++;return seed; }
static void image_alias_release(void *seed) { assert(seed && alias_releases<alias_retains);alias_releases++; }
void resource_effects_emitted(void) { assert(gpu_emitted); }
void resource_effects_retain_part_disposition(void *part,int (*known_unhanded)(const void *),void *owner,
                                             void (*release)(void *,void *,int)) {
    assert(part==&part_owner && known_unhanded(part) && !discard);discard=release;discard_owner=owner;
}
static void destroy_tex(Tex *e) { assert(locked && e->pub.image==emitted_image);e->pub.gone=1;discarded++; }
static void settle_replacement(int known_unhanded) {
    void (*callback)(void *,void *,int)=discard;
    void *owner=discard_owner;
    discard=NULL;discard_owner=NULL;
    assert(callback&&owner);callback(&part_owner,owner,known_unhanded);
}
void rt_fatal(const char *fmt, ...) { (void)fmt; abort(); }
static void make_key(const tex_t *t, TexKey *key) {
    *key = (TexKey){.addr=(uintptr_t)t->mem,.d=t->d};
}
static int image_alias_load_current(Tex *entry, const VkImageSubresourceRange *planes, RcRegistryPosition read) {
    assert(locked && entry->pub.fb_learned && planes->aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT);
    assert(read.scene == 2 && read.ordinal == 8);
    return native_depth_current;
}
int vk_mem_lookup(uint64_t address, uint64_t size, VkMemSpan *span) {
    assert(locked && address == depth_mapping.address && size == depth_mapping.range.size && span);
    return 1;
}
uint32_t vk_mem_device_intervals(const VkMemSpan *span, uint64_t address, uint64_t size, RcInterval **out) {
    assert(locked && span && address == depth_mapping.address && size == depth_mapping.range.size);
    *out = malloc(sizeof **out);
    assert(*out);
    **out = depth_mapping;
    return 1;
}
RcRegistry *rc_registry_runtime(void) { return HANDLE(RcRegistry *, 1); }
int rc_registry_native_image_has_prior_projection(RcRegistry *registry, const RcInterval *mapping,
    uint32_t count, RcRegistryPosition read) {
    assert(locked && registry == rc_registry_runtime() && count == 1 &&
           mapping->mapping == depth_mapping.mapping && read.scene == 2 && read.ordinal == 8);
    return 1;
}
static int image_alias_single_color_view(const tex_t *t, const view_t *view) {
    (void)t; (void)view;
    abort(); // This fixture selects only depth planes.
}
int rc_registry_native_image_has_prior_bound_projection(RcRegistry *registry, const RcInterval *mapping,
    uint32_t count, RcRegistryPosition read) {
    (void)registry; (void)mapping; (void)count; (void)read;
    abort(); // The earlier independent depth producer already establishes relevance.
}
RcRegistryResult vk_mem_image_alias_baseline_capture(uint64_t address, uint64_t bytes, const RcInterval *mapping,
    uint32_t count, RcRegistryPosition read, VkMemImageAliasBaseline **out) {
    (void)address; (void)bytes; (void)mapping; (void)count; (void)read; (void)out;
    abort(); // Every case has a real prior image projection, so there is no raw fallback.
}
void vk_mem_image_alias_baseline_release(VkMemImageAliasBaseline **baseline) { assert(!*baseline); }
int vk_mem_image_held(VkImage image) { assert(locked && image); return 0; }
static Tex *entry_of_mode(const tex_t *t, int private_output, uint32_t *generation) {
    (void)t; (void)private_output; (void)generation;
    abort(); // The original target is already selected.
}
enum { DROP_REPLACED };
static void drop_tex(Tex *entry, int why) { (void)entry; (void)why; abort(); }
static int image_alias_refuse(VkTextureAliasFailure *failure, VkTextureAliasPhase phase,
    RcRegistryResult status, RcInterval receiver) {
    *failure = (VkTextureAliasFailure){.phase=phase,.status=status,.receiver=receiver};
    return -1;
}
static int image_alias_read_capture_locked(const tex_t *t, const Tex *entry, VkTextureAliasRead **out,
    VkTexImage *shape, int force, int target, const VkImageSubresourceRange *planes,
    VkMemImageAliasBaseline **baseline, VkTextureAliasFailure *failure) {
    assert(locked && t->serial == entry->t.serial && entry->pub.image && force && target == 2 && !*baseline);
    assert(planes->aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT && planes->baseMipLevel == 0 &&
           planes->levelCount == 1 && planes->baseArrayLayer == 0 && planes->layerCount == 1);
    ++depth_source_captures;
    if (!depth_sources_ready)
        return image_alias_refuse(failure, VK_TEXTURE_ALIAS_COVERAGE, RC_REGISTRY_BLOCKED, depth_mapping);
    *shape = entry->pub;
    shape->image = VK_NULL_HANDLE;
    depth_seed = (VkTextureAliasRead){.receiver=t,.result=*shape,.predecessor=entry->pub.image,
        .predecessor_serial=t->serial,.predecessor_revision=entry->content_revision,.load_planes=*planes};
    *out = &depth_seed;
    return 1;
}
""" + functions(tex, 'image_alias_whole_range', 'image_alias_predecessor_matches', 'vk_texture_alias_target_current',
                'vk_texture_alias_target_load_current',
                'vk_texture_alias_target_replacement', 'vk_texture_alias_target_read_prior',
                'vk_texture_alias_target_prepare_replacement',
                'image_alias_replacement_part_dispose', 'image_alias_replacement_record_part',
                'image_alias_promote_target_locked', 'vk_texture_alias_target_record_replacement',
                'vk_texture_alias_read_intervals', 'vk_texture_alias_target_capture') + r"""
struct VkExecGraphicsPass {
    VkExecGraphicsPassShape shape;
    VkTextureTargetLoan *loan[9];
    VkExecTargetBinding bindings[9];
};
static int has_image(const VkImage *list, int n, VkImage image) {
    for (int i=0; i<n; ++i) if (list[i] == image) return 1;
    return 0;
}
int vk_texture_target_loan_backing_current(const VkTextureTargetLoan *loan, const RcRegistryImageWriteExtent *e, uint32_t count) {
    return loan && e && count == 1;
}
""" + functions(replay, 'graphics_attachment_failure', 'graphics_attachment_peer_failure',
    'vk_exec_graphics_pass_attachment_peers', 'vk_exec_graphics_pass_attachment_read', 'graphics_pass_feedback') + r"""
static void check_attachment_peers(VkTextureAliasRead *pending) {
    RcInterval target = {.mapping=17, .range={18,0x400,0x80}};
    RcInterval disjoint = {.mapping=28, .range={18,0x480,0x80}};
    RcInterval overlapping = {.mapping=38, .range={18,0x420,0x10}};
    RcInterval different = {.mapping=48, .range={19,0x400,0x80}};
    RcRegistryImageWriteExtent color = {.interval=disjoint}, depth = {.interval=different};
    RcInterval *original = pending->receiver_mapping;
    pending->receiver_mapping = &target;
    VkExecGraphicsPass pass = {.shape={.color={{.image=pending->result.image},{.image=HANDLE(VkImage,88)}},
        .depth={.image=HANDLE(VkImage,89)}}, .loan={[1]=HANDLE(VkTextureTargetLoan*,71),[8]=HANDLE(VkTextureTargetLoan*,72)},
        .bindings={{.load_seed=pending}, {.extents=&color,.extent_count=1},
                                                [8]={.extents=&depth,.extent_count=1}}};
    VkExecPreparationFailure why = {0};
    assert(vk_exec_graphics_pass_attachment_peers(&pass,&why) && !why.owner);
    assert(vk_exec_graphics_pass_attachment_read(&pass,HANDLE(VkImage,91),&disjoint,1,17,VK_EXEC_ATTACHMENT_READ_DISJOINT,&why) && !why.owner);
    assert(!vk_exec_graphics_pass_attachment_read(&pass,HANDLE(VkImage,91),&overlapping,1,19,VK_EXEC_ATTACHMENT_READ_DISJOINT,&why));
    assert(why.owner == VK_EXEC_PREPARATION_IMAGE && why.reason == VK_EXEC_PREPARATION_FEEDBACK &&
           why.detail == 19 && why.range.mapping == 17 && why.peer.mapping == 38);
    why = (VkExecPreparationFailure){0};
    assert(!vk_exec_graphics_pass_attachment_read(&pass,HANDLE(VkImage,91),NULL,0,20,VK_EXEC_ATTACHMENT_READ_DISJOINT,&why));
    assert(why.reason == VK_EXEC_PREPARATION_MAPPING && why.detail == 20);
    why = (VkExecPreparationFailure){0};
    assert(vk_exec_graphics_pass_attachment_read(&pass,pending->predecessor,&overlapping,1,19,
                                                VK_EXEC_ATTACHMENT_READ_PRE_DRAW,&why) && !why.owner);
    assert(vk_exec_graphics_pass_attachment_read(&pass,HANDLE(VkImage,91),&overlapping,1,19,
                                                VK_EXEC_ATTACHMENT_READ_FRAGMENT,&why) && !why.owner);
    assert(!vk_exec_graphics_pass_attachment_read(&pass,pending->predecessor,&overlapping,1,19,
                                                 VK_EXEC_ATTACHMENT_READ_FRAGMENT,&why));
    assert(why.reason == VK_EXEC_PREPARATION_FEEDBACK);
    why = (VkExecPreparationFailure){0};
    assert(!vk_exec_graphics_pass_attachment_read(&pass,pending->result.image,&disjoint,1,19,
                                                 VK_EXEC_ATTACHMENT_READ_PRE_DRAW,&why));
    assert(why.reason == VK_EXEC_PREPARATION_FEEDBACK);
    why = (VkExecPreparationFailure){0};
    const VmaAllocation allocation = pending->allocation;
    pending->allocation = NULL;
    assert(!vk_exec_graphics_pass_attachment_read(&pass,HANDLE(VkImage,91),&overlapping,1,19,
                                                 VK_EXEC_ATTACHMENT_READ_PRE_DRAW,&why));
    pending->allocation = allocation;
    why = (VkExecPreparationFailure){0};
    pending->upload_canceled = 1;
    assert(!vk_exec_graphics_pass_attachment_read(&pass,HANDLE(VkImage,91),&overlapping,1,19,
                                                 VK_EXEC_ATTACHMENT_READ_PRE_DRAW,&why));
    pending->upload_canceled = 0;
    why = (VkExecPreparationFailure){0};
    original_current = 0;
    assert(!vk_exec_graphics_pass_attachment_read(&pass,HANDLE(VkImage,91),&overlapping,1,19,
                                                 VK_EXEC_ATTACHMENT_READ_PRE_DRAW,&why));
    original_current = 1;
    why = (VkExecPreparationFailure){0};
    const RcInterval all_planes[] = {overlapping, {.mapping=38, .range={18,0x420,0}}};
    assert(!vk_exec_graphics_pass_attachment_read(&pass,HANDLE(VkImage,91),all_planes,2,19,
                                                 VK_EXEC_ATTACHMENT_READ_PRE_DRAW,&why));
    assert(why.reason == VK_EXEC_PREPARATION_MAPPING);
    why = (VkExecPreparationFailure){0};
    // The same retained comparison applies to an independently allocated depth replacement.
    pass.bindings[0].load_seed = NULL;
    pass.bindings[8].load_seed = pending;
    pass.shape.depth.image = pending->result.image;
    assert(vk_exec_graphics_pass_attachment_read(&pass,HANDLE(VkImage,91),&overlapping,1,19,
                                                VK_EXEC_ATTACHMENT_READ_PRE_DRAW,&why) && !why.owner);
    assert(!vk_exec_graphics_pass_attachment_read(&pass,pending->result.image,&overlapping,1,19,
                                                 VK_EXEC_ATTACHMENT_READ_PRE_DRAW,&why));
    assert(why.reason == VK_EXEC_PREPARATION_FEEDBACK && why.slot == 8);
    pass.bindings[0].load_seed = pending;
    pass.bindings[8].load_seed = NULL;
    why = (VkExecPreparationFailure){0};
    depth.interval = overlapping;
    assert(!vk_exec_graphics_pass_attachment_peers(&pass,&why));
    assert(why.reason == VK_EXEC_PREPARATION_ATTACHMENT && why.detail == 8 && why.peer.mapping == 38);
    pending->receiver_mapping = original;
}
static void check_learned_depth_load(void) {
    const tex_t tex = {.mem=HANDLE(uint8_t *,depth_mapping.address),.serial=48,.L={.size=8192,.layers=1},
        .d={.format=NVN_FORMAT_DEPTH32F,.target=NVN_TEXTURE_TARGET_2D,.width=16,.height=16,.levels=1,.samples=1}};
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    const VkImage preceding = HANDLE(VkImage,65);
    Tex old = {.t=tex,.pub={.image=preceding,.aspect=VK_IMAGE_ASPECT_DEPTH_BIT,.levels=1,.layers=1,
                             .fb_learned=1,.fb_usage=1,.usage=usage},
               .content_revision=7,.ik={.usage=usage}};
    make_key(&tex,&old.key);
    assert(!nvn_hm_put(&by_key,&old.key,&old));
    VkTextureAliasRead *seed = NULL;
    VkTextureAliasFailure why = {0};
    native_depth_current = 1;
    assert(vk_texture_alias_target_capture(&tex,NULL,1,&seed,1,&why) == 0 && !seed);
    native_depth_current = 0;
    assert(vk_texture_alias_target_capture(&tex,NULL,1,&seed,0,&why) == 2 && !seed);
    assert(depth_source_captures == 0);
    old.ik.usage &= ~VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
    assert(vk_texture_alias_target_capture(&tex,NULL,1,&seed,1,&why) < 0 && !seed);
    assert(why.phase == VK_TEXTURE_ALIAS_OWNER && !depth_source_captures);
    old.ik.usage = usage;
    old.pub.fb_usage = 0;
    assert(vk_texture_alias_target_capture(&tex,NULL,1,&seed,1,&why) < 0 && !seed);
    assert(why.phase == VK_TEXTURE_ALIAS_OWNER && !depth_source_captures);
    old.pub.fb_usage = 1;
    assert(vk_texture_alias_target_capture(&tex,NULL,1,&seed,1,&why) < 0 && !seed);
    assert(why.phase == VK_TEXTURE_ALIAS_COVERAGE && depth_source_captures == 1);
    depth_sources_ready = 1;
    why = (VkTextureAliasFailure){0};
    assert(vk_texture_alias_target_capture(&tex,NULL,1,&seed,1,&why) == 1 && seed == &depth_seed);
    assert(depth_source_captures == 2 && !seed->result.image && seed->result.fb_learned &&
           seed->result.fb_usage && seed->result.usage == usage && !seed->target_created);
    assert(old.pub.image == preceding && image_alias_predecessor_matches(seed,&old) && old.content_revision == 7);

    seed->result.image = HANDLE(VkImage,66);
    seed->allocation = HANDLE(VmaAllocation,67);
    seed->uploaded = original_current = 1;
    assert(vk_texture_alias_target_read_prior(seed,seed->result.image,preceding,0));
    assert(!vk_texture_alias_target_read_prior(seed,seed->result.image,preceding,1));
    VkExecGraphicsPassShape shape = {.depth={.image=seed->result.image,.img=&seed->result}};
    graphics_pass_feedback(&shape);
    assert(shape.fbl_mask == (1u << 8) && shape.feedback_count == 1 &&
           shape.feedback_images[0] == seed->result.image && shape.feedback_aspects[0] == VK_IMAGE_ASPECT_DEPTH_BIT);
    assert(old.pub.image == preceding && old.pub.fb_learned && !gpu_emitted);
    // Free the fixture's local association while preserving the existing generic fates below.
    assert(nvn_hm_del(&by_key,&old.key));
}

int main(void) {
    check_learned_depth_load();
    tex_t gpu = {.mem=HANDLE(uint8_t*,0x8000000000), .serial=5, .L.size=8192,
        .d={.width=128,.height=64,.target=NVN_TEXTURE_TARGET_2D,.levels=1,.samples=1,.format=NVN_FORMAT_R32UI}};
    tex_t cpu = gpu;
    cpu.mem = HANDLE(uint8_t*,0x10000);
    const NVNcopyRegion full = {0,0,0,128,64,1};
    p_clear_tex clear = {.tex=&gpu,.is_uint=1,.r=full,.mask=0xf,.c.u={1,0,0,0}};
    assert(vk_exec_clear_alias_candidate(&clear));
    clear.mask = 0x7;
    assert(!vk_exec_clear_alias_candidate(&clear));
    clear.mask = 0xf;
    clear.r.width--;
    assert(!vk_exec_clear_alias_candidate(&clear));
    clear.r = full;
    clear.is_uint = 0;
    assert(!vk_exec_clear_alias_candidate(&clear));
    clear.c.u[0] = 0;
    assert(vk_exec_clear_alias_candidate(&clear));
    clear.has_view = clear.view.set_format = 1;
    clear.view.format = gpu.d.format;
    assert(vk_exec_clear_alias_candidate(&clear));
    clear.view.format = NVN_FORMAT_R8;
    assert(!vk_exec_clear_alias_candidate(&clear));
    clear.view.format = gpu.d.format;
    clear.view.set_levels = 1;
    clear.view.base_level = 1;
    assert(!vk_exec_clear_alias_candidate(&clear));
    VkTexImage array = {.levels = 2, .layers = 2, .aspect = VK_IMAGE_ASPECT_COLOR_BIT};
    VkTextureAliasLayout planes = {.count = 4, .bytes = 64, .planes = {
        {{.bufferOffset = 0, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}},1,16,1},
        {{.bufferOffset = 16, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,1}},1,16,1},
        {{.bufferOffset = 32, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,1,0,1}},1,16,1},
        {{.bufferOffset = 48, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,1,1,1}},1,16,1}}};
    assert(!texture_alias_clear_planes((VkImageSubresourceRange){VK_IMAGE_ASPECT_DEPTH_BIT,1,1,1,1}, &array, &planes));
    assert(!texture_alias_clear_planes((VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT,1,2,1,1}, &array, &planes));
    assert(texture_alias_clear_planes((VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT,1,1,0,1}, &array, &planes));
    assert(planes.count == 1 && planes.bytes == 48 && planes.planes[0].copy.bufferOffset == 32);
    p_copy_t2t copy = {.src=&cpu,.dst=&gpu,.sr=full,.dr=full};
    assert(vk_exec_texture_to_texture_alias_candidate(&copy));
    copy.dr.width--;
    assert(vk_exec_texture_to_texture_alias_candidate(&copy));
    copy.has_dv = copy.dv.set_format = 1;
    assert(!vk_exec_texture_to_texture_alias_candidate(&copy));
    copy.has_dv = 0;
    copy.src = NULL;
    assert(!vk_exec_texture_to_texture_alias_candidate(&copy));
    copy.src = &gpu;
    copy.dst = &cpu;
    assert(!vk_exec_texture_to_texture_alias_candidate(&copy));

    const VkImageSubresourceRange prepared_planes = {VK_IMAGE_ASPECT_COLOR_BIT,1,1,1,2};
    const VkImageSubresourceRange rendered_planes = {VK_IMAGE_ASPECT_COLOR_BIT,1,1,1,1};
    tex_t layered=gpu;
    layered.d.levels=2;layered.d.target=NVN_TEXTURE_TARGET_2D_ARRAY;layered.L.layers=3;
    view_t view={.set_levels=1,.base_level=1,.set_layers=1,.min_layer=1};
    VkImageSubresourceRange selected_planes;
    assert(vk_texture_alias_attachment_range(&layered,&view,0,&selected_planes));
    assert(selected_planes.aspectMask==VK_IMAGE_ASPECT_COLOR_BIT && selected_planes.baseMipLevel==1 &&
           selected_planes.baseArrayLayer==1 && selected_planes.layerCount==2);
    view.nlayers=3;
    assert(!vk_texture_alias_attachment_range(&layered,&view,0,&selected_planes));
    view.nlayers=1;view.set_format=1;view.format=NVN_FORMAT_R8;
    assert(!vk_texture_alias_attachment_range(&layered,&view,0,&selected_planes));
    view.set_format=0;layered.d.samples=4;
    assert(!vk_texture_alias_attachment_range(&layered,&view,0,&selected_planes));
    layered.d.samples=1;layered.d.target=NVN_TEXTURE_TARGET_3D;
    assert(!vk_texture_alias_attachment_range(&layered,&view,0,&selected_planes));
    layered.d.target=NVN_TEXTURE_TARGET_CUBEMAP;layered.L.layers=6;
    view.set_target=1;view.target=NVN_TEXTURE_TARGET_2D;view.min_layer=5;
    assert(vk_texture_alias_attachment_range(&layered,&view,0,&selected_planes) && selected_planes.layerCount==1);
    Tex old = {.t=gpu,.pub={.image=HANDLE(VkImage,85),.addr=(uintptr_t)gpu.mem,.levels=2,.layers=3,
                           .fb_usage=1,.fb_learned=1},.alloc=HANDLE(VmaAllocation,90),
        .bytes=8192,.content_revision=4,.key.addr=(uintptr_t)gpu.mem,.released=1,.superseded=1};
    assert(!nvn_hm_put(&by_key, &old.key, &old));
    RcInterval mapping[1] = {0}, demands[3] = {0};
    tex_t recreated = gpu;
    recreated.serial++;
    VkTextureAliasRead pending = {.receiver=&recreated,.key=old.key,
        .result={.image=HANDLE(VkImage,87),.addr=old.pub.addr},.allocation=HANDLE(VmaAllocation,97),
        .guest_bytes=8192,.placement_bytes=32768,.mapping_count=1,.demand_count=3,
        .receiver_mapping=mapping,.demands=demands,.predecessor=old.pub.image,
        .predecessor_serial=gpu.serial,.predecessor_revision=4,.predecessor_dormant=1,.load_planes=prepared_planes};
    assert(image_alias_predecessor_matches(&pending,&old));
    old.pub.gone=1;assert(!image_alias_predecessor_matches(&pending,&old));old.pub.gone=0;
    old.released=0;assert(!image_alias_predecessor_matches(&pending,&old));old.released=1;
    original_current = 1;
    assert(!vk_texture_alias_target_prepare_replacement(&pending,&rendered_planes)); // no upload yet
    pending.uploaded = 1;
    original_current = 0;
    assert(!vk_texture_alias_target_prepare_replacement(&pending,&rendered_planes)); // stale real input
    original_current = 1;
    assert(!vk_texture_alias_target_prepare_replacement(&pending,&(VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT,0,1,1,1}));
    assert(!vk_texture_alias_target_prepare_replacement(&pending,&(VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT,1,1,1,3}));
    assert(vk_texture_alias_target_prepare_replacement(&pending,&rendered_planes));
    check_attachment_peers(&pending);
    assert(old.t.serial == gpu.serial && old.released && old.superseded && !resumes && !rebinds);
    assert(old.pub.image == HANDLE(VkImage,85) && prepared == 1 && !ownership_replacements && !image_retired);
    gpu_emitted = 1, emitted_image = pending.result.image;
    vk_texture_alias_target_record_replacement(&pending,&rendered_planes);
    assert(pending.target_replaced && !pending.allocation && !pending.demands && !pending.receiver_mapping);
    assert(!old.released && !old.superseded && old.t.serial==recreated.serial && resumes==1 && rebinds==1);
    assert(alias_dormant_target_replacements==1 && n_restarts==1);
    assert(old.pub.image == emitted_image && old.alloc == HANDLE(VmaAllocation,97) && old.bytes == 32768);
    assert(old.pub.fb_usage && old.pub.fb_learned);
    assert(old.image_private_mapping == mapping && old.alias_plane_demand == demands && old.content_revision == 5);
    assert(image_retired == 1 && ownership_replacements == 1 && vk_texture_alias_target_current(&pending, emitted_image));
    assert(!vk_texture_alias_target_replacement(&pending));
    assert(discard && discard_owner==&pending);
    settle_replacement(0);
    assert(!discarded && part_releases==1 && alias_releases==1); // Handed replacement remains available.
    image_alias_replacement_record_part(&pending,&part_owner);
    old.pub.image=HANDLE(VkImage,98);
    settle_replacement(1);
    assert(!discarded && part_releases==2 && alias_releases==2); // Later image at the same key survives.
    image_alias_replacement_record_part(&pending,&part_owner);
    old.pub.image=emitted_image;
    settle_replacement(1);
    assert(discarded==1 && old.pub.gone && part_releases==3 && alias_releases==alias_retains);
    free(by_key.vals); free(by_key.keys);
    puts("Ordered image writers: exact transfer eligibility, mixed LOAD disjoint peers, learned D32 independent LOAD and feedback, predecessor preserved until issue");
}
"""


def main():
    cc = Path(os.environ.get('CC') or shutil.which('clang') or '').resolve()
    if not cc.is_file():
        raise SystemExit('Clang is required (set CC)')
    vma = Path(os.environ.get('VMA_INCLUDE', cc.parent.parent / 'include' / 'vma'))
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get('PATH', ''))
    root = Path(env.get('IMAGE_WRITER_CHECK_SOURCE', ROOT / 'runtime'))
    with tempfile.TemporaryDirectory(prefix='image-writer-') as temp:
        source = Path(temp) / 'check.c'
        exe = Path(temp) / ('check.exe' if os.name == 'nt' else 'check')
        source.write_text(fixture(root))
        subprocess.run([str(cc), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror',
                        '-fsanitize=undefined', '-fsanitize-trap=undefined',
                        '-I' + str(root), '-I' + str(ROOT / 'runtime'), '-I' + str(vma), str(source), '-o', str(exe)],
                       env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == '__main__':
    main()

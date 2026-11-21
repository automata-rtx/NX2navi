"""Exercise geometry-only draw peer validation and descriptor issue on the production gates."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from ordered_graphics_feedback_check import function


STUBS = r"""
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "resource_core_observation.h"

typedef uintptr_t VkImage, VkImageView, VkSampler, VkCommandBuffer;
typedef uint32_t VkImageLayout, VkImageViewType;
typedef struct { VkSampler sampler; VkImageView imageView; VkImageLayout imageLayout; } VkDescriptorImageInfo;
typedef struct { VkImage image; } VkRenderingAttachmentInfo;
typedef struct { VkImage depth; } VkRenderingInfo;
typedef struct { int mo_ovr_on; } DrawCtx;
typedef struct { VkImage image, bc_backing; } VkExecTarget;
typedef struct { RcInterval interval; } RcRegistryImageWriteExtent;
typedef struct { const RcRegistryImageWriteExtent *extents; uint32_t count; } VkTextureTargetLoan;
typedef struct { const RcInterval *intervals; uint32_t count; } VkTextureAliasRead;
typedef struct {
    struct { uint64_t image; } binding;
    const RcRegistryImageWriteExtent *extents;
    uint32_t extent_count;
    const VkTextureAliasRead *load_seed;
} VkExecTargetBinding;
typedef struct {
    VkExecTarget color[8], depth;
    VkImage feedback_images[9];
    int feedback_count;
} VkExecGraphicsPassShape;
typedef struct {
    VkExecGraphicsPassShape shape;
    VkExecTargetBinding bindings[9];
    const VkTextureTargetLoan *loan[9];
    int activated;
} VkExecGraphicsPass;
typedef struct {
    VkExecTarget color[8], depth;
    VkExecTargetBinding target_bindings[9];
    uint32_t pass_ncolor, target_binding_mask, fb_dirty, fb_read;
    int in_pass, nfbl, hud_alpha_attachment;
    int recorded;
    VkCommandBuffer cb;
    uint64_t fence;
    const DrawCtx *dc;
    VkImage fbl_img[9];
    struct { uint32_t width, height; } tonemap_extent;
} VkExec;
typedef struct { int slot, pool, type, exact, flags, format; } GfxShaderRes;
typedef struct { VkImage image; int aspect, skip; } VkTexImage;
typedef struct { int unused; } view_t;
typedef struct { int unused; } ViewKey;
typedef struct {
    VkImage source, image;
    VkImageView view;
    VkCommandBuffer cb;
    uint64_t fence, registration;
} VkTextureQueryImage;
typedef struct {
    const VkTexImage *image;
    uint64_t registration;
    int native_slot, historical, canonical, projected, has_view;
    view_t view;
    VkTextureQueryImage *query_image;
} VkTextureSample;
typedef struct { uint64_t registration; RcInterval mapping; int registration_only; } VkTextureSampleGeometry;
typedef struct { uint64_t registration; int native_slot, unbacked, canonical, projected, refused; } VkTextureReadKey;
typedef struct { VkImage image; } VkTexturePresentSource;
typedef struct {
    const GfxShaderRes *resource;
    uint64_t handle, token;
    int stage, member;
    VkTextureSample sample;
    VkTexImage image;
    VkTextureSampleGeometry geometry;
    VkTextureAliasRead *alias;
    const VkTexturePresentSource *hold;
} GraphicsSample;
typedef struct {
    const GfxShaderRes *res;
    uint32_t stage, kind, type, set, count, binding;
} VkPipeBinding;
typedef struct { uintptr_t layout; uint32_t n_sets, n_bindings; VkPipeBinding bindings[4]; } VkPipeInfo;
typedef struct {
    const void *revision;
    const VkPipeInfo *info;
    uint64_t vs_query_size_lod0_mask, fs_query_size_lod0_mask;
} VkGraphicsPipelineUse;
typedef struct {
    const VkExecGraphicsPass *pass;
    VkGraphicsPipelineUse use;
    GraphicsSample *samples;
    uint32_t sample_count, opcode;
    int mesh, geometry_recording_attested, compatibility, feedback_selected;
} VkExecGraphicsDraw;

enum { GFX_RES_TEX, GFX_RES_IMG, GFX_STAGE_VS = 0, GFX_STAGE_FS = 1, NVN_STAGE_VS = 0, NVN_STAGE_MESH = 6,
       NVN_STAGE_TASK = 7, NVN_STAGE_FS = 1, NVN_STAGE_CS = 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER = 1 };
enum { VK_TEXTURE_REGISTRATION_GONE = -2, VK_TEXTURE_REGISTRATION_ABSENT = -1, VK_TEXTURE_REGISTRATION_EXACT = 1 };
typedef enum { VK_TEXTURE_SAMPLE_READY, VK_TEXTURE_SAMPLE_FALLBACK, VK_TEXTURE_SAMPLE_UNBACKED, VK_TEXTURE_SAMPLE_PREPARE,
               VK_TEXTURE_SAMPLE_DEFERRED, VK_TEXTURE_SAMPLE_UNAVAILABLE, VK_TEXTURE_SAMPLE_CANONICAL_REFUSED } VkTextureSampleStatus;
enum { VK_TEXTURE_READ_REQUIRED = 1, VK_TEXTURE_READ_FORCE_COMPAT = 2, VK_IMAGE_ASPECT_DEPTH_BIT = 2,
       VK_IMAGE_ASPECT_STENCIL_BIT = 4, VK_IMAGE_VIEW_TYPE_2D = 1, VK_IMAGE_VIEW_TYPE_3D = 2, VK_NULL_HANDLE = 0,
       VK_IMAGE_LAYOUT_GENERAL = 1, VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT = 2,
       VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT = 1, VK_ACCESS_2_MEMORY_WRITE_BIT = 2, VK_ACCESS_2_MEMORY_READ_BIT = 4 };
#define NVN_HANDLE_TEX(h) ((int)((h) & 0xfffff))

static const VkExecGraphicsDraw *graphics_current;
static int vk_dlaa_swap_active(const VkExec *ex) { (void)ex; return 1; }
static uint64_t vk_texture_current_registration_locked(int slot, void *view, void *has_view) {
    assert(slot == 17 && !view && !has_view);
    return 301;
}
static int vk_dlaa_swap_matches(uint64_t token) { return token == 301; }

static VkImage current_image = 101;
static uint64_t current_registration = 301;
static unsigned unavailable;
static const char *reason;
static VkExec *recording;
static unsigned transitions, render_ends, render_begins, render_barriers, texture_locked;
static unsigned ordinary_reads;
static int ordinary_texel_control;

static const RcInterval *vk_texture_alias_read_intervals(const VkTextureAliasRead *seed, uint32_t *count) {
    *count = seed ? seed->count : 0;
    return seed ? seed->intervals : NULL;
}
static int vk_texture_target_loan_backing_current(const VkTextureTargetLoan *loan,
    const RcRegistryImageWriteExtent *extents, uint32_t count) {
    return loan && loan->extents == extents && loan->count == count;
}
static const VkExecGraphicsPassShape *vk_exec_graphics_pass_shape(const VkExecGraphicsPass *pass) {
    return &pass->shape;
}
static int vk_texture_sample_geometry_current_locked(int logical, const VkTextureSample *sample,
    const VkTexImage *image, const VkTextureSampleGeometry *geometry, const RcInterval **ranges, uint32_t *count) {
    assert(logical == 17 && !ranges && !count);
    return image->image == current_image && geometry->registration == current_registration &&
        sample->registration == current_registration;
}
static int vk_texture_query_image_current_locked(const VkTextureQueryImage *query) {
    return query && query->source == current_image && query->registration == current_registration;
}
static VkImageView vk_texture_query_image_view(const VkExec *ex, const VkTextureQueryImage *query, VkImage *image) {
    if (image) *image = 0;
    if (!query || !query->cb || query->cb != ex->cb || query->fence != ex->fence) return 0;
    if (image) *image = query->image;
    return query->view;
}
static int vk_texture_query_image_record_locked(VkExec *ex, VkTextureQueryImage *query) {
    assert(texture_locked && !ex->in_pass && ex == recording && ex->cb && !query->cb);
    if (!vk_texture_query_image_current_locked(query)) return 0;
    assert(query->image != ex->depth.image && query->image != query->source);
    query->cb = ex->cb;
    query->fence = ex->fence;
    transitions++;
    return 1;
}
static void vk_texture_lock(void) { assert(!texture_locked++); }
static void vk_texture_unlock(void) { assert(texture_locked-- == 1); }
static VkRenderingInfo ordinary_rendering(VkExec *ex, VkRenderingAttachmentInfo *color,
    VkRenderingAttachmentInfo *depth, VkRenderingAttachmentInfo *stencil) {
    assert(ex == recording && ex->in_pass && color && depth && stencil);
    return (VkRenderingInfo){.depth = ex->depth.image};
}
static void vk_exec_oq_pass_end(VkExec *ex) { assert(ex == recording && ex->in_pass); }
static void vkCmdEndRendering(VkCommandBuffer cb) {
    assert(cb == recording->cb && recording->in_pass);
    render_ends++;
}
static void memory_barrier(VkCommandBuffer cb, int src_stage, int src_access, int dst_stage, int dst_access) {
    assert(cb == recording->cb && !recording->in_pass && src_stage == VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT &&
        dst_stage == src_stage && src_access == VK_ACCESS_2_MEMORY_WRITE_BIT &&
        dst_access == (VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT));
    render_barriers++;
}
static void vkCmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo *rendering) {
    assert(cb == recording->cb && !recording->in_pass && rendering->depth == recording->depth.image);
    render_begins++;
}
static int vk_texture_registration_expected_locked(int logical, void *record) {
    (void)logical; (void)record; assert(0); return 0;
}
static int vk_texture_history_sample_current_locked(int logical, uint64_t registration, VkImage image) {
    (void)logical; (void)registration; (void)image; assert(0); return 0;
}
static int vk_texture_sample_unbacked_current_locked(VkTextureReadKey key) {
    (void)key; assert(0); return 0;
}
static int vk_texture_alias_sample_current_locked(int logical, VkTextureAliasRead *alias) {
    (void)logical; (void)alias; assert(0); return 0;
}
static int vk_texture_resolved_receipt_locked(VkTextureReadKey key, VkImage image, void *edge) {
    assert(ordinary_texel_control && !edge);
    return key.registration == current_registration && image == current_image;
}
static uint64_t token_of(uint64_t handle) { (void)handle; assert(0); return 0; }
static int vk_texture_binding_receipt_locked(int slot, VkImage image, void *edge) {
    (void)slot; (void)image; (void)edge; assert(0); return 0;
}
static int graphics_prepared_sample_current(VkExec *ex, const VkExecGraphicsDraw *draw, const GraphicsSample *sample) {
    (void)ex; (void)draw; (void)sample; assert(0); return 0;
}
static void vk_exec_required_graphics_unavailable(VkExec *ex, uint32_t opcode, const char *message) {
    assert(ex && (opcode == 26 || !opcode)); ++unavailable; reason = message;
}

static int vk_dlaa_swap_img, twin_dispatch, flat3d_pool_consuming;
static struct { VkSampler samp, samp_shadow; struct { VkImageView view; } img[1]; } Z;
static int vk_texture_deferred(void) { return 0; }
static int registered_slot_locked(int logical) { (void)logical; abort(); }
static const VkTexImage *vk_texture_by_id_locked(int id, view_t *view, int *has_view) {
    (void)id; (void)view; (void)has_view; abort();
}
static int twin_compare_resource(const GfxShaderRes *res) { (void)res; abort(); }
static VkImageViewType res_view_type(int type) { assert(type == 3); return VK_IMAGE_VIEW_TYPE_2D; }
static int compute_sample(VkExec *ex, uint32_t member, const GfxShaderRes *res, uint64_t handle, VkTextureSample *sample,
    VkTextureAliasRead **alias, const VkTexturePresentSource **source) {
    (void)ex; (void)member; (void)res; (void)handle; (void)sample; (void)alias; (void)source; abort();
}
typedef struct {int unused;} VkTextureSampleRefusal;
static void sampled_binding_refusal(VkExec *ex, int stage, uint32_t member, const GfxShaderRes *resource, uint64_t handle,
    VkTextureSampleStatus status, const VkTextureSample *sample, const VkTextureSampleRefusal *refused,
    int geometry, const char *message) {
    (void)ex; (void)stage; (void)member; (void)resource; (void)handle; (void)status;
    (void)sample; (void)refused;
    assert(geometry && message);
    unavailable++;
    reason=message;
}
static VkTextureSampleStatus vk_texture_resolve_sample_report_locked(int logical, unsigned flags, VkTextureSample *sample, VkTextureSampleRefusal *refused) {
    (void)refused;
    (void)logical; (void)flags; (void)sample; abort();
}
static int vk_texture_sample_wait_gpa_locked(VkExec *ex, int logical) { (void)ex; (void)logical; abort(); }
static void vk_exec_registration_gone(VkExec *ex, int logical) { (void)ex; (void)logical; abort(); }
static VkTextureSampleStatus vk_texture_history_flat3d_locked(int logical, const VkTexImage **image, void *plan) {
    (void)logical; (void)image; (void)plan; abort();
}
static const VkTexImage *vk_texture_exact_by_id_locked(int id, view_t *view, int *has_view) {
    (void)id; (void)view; (void)has_view; abort();
}
static const VkTexImage *vk_texture_writable_locked(const VkTexImage *image) { (void)image; abort(); }
static VkImageView alias_sample_view_locked(VkTextureAliasRead *alias, const VkTexImage *image,
    const view_t *view, int has_view, int type) {
    (void)alias; (void)image; (void)view; (void)has_view; (void)type; abort();
}
static VkImageView get_view_actual(VkExec *ex, const VkTexImage *img, const view_t *view, int has_view,
    int type, int storage, VkImage *image, int demanded, VkTextureReadKey key) {
    assert(ordinary_texel_control && ex == recording && img->image == ex->depth.image &&
        view && !has_view && type == 3 && !storage && demanded && key.registration == current_registration);
    *image = img->image;
    return 701;
}
static int sampled_rgba8_linear_request(const VkTexImage *image, const view_t *view, int has_view, int type) {
    (void)image; (void)view; (void)has_view; (void)type; abort();
}
static int pool_comparison_format(const ViewKey *key, uint64_t *features) {
    (void)key;(void)features;abort(); // The fixture only exercises direct bindings.
}
static int texture_view_key(const VkTexImage *image, const view_t *view, int has_view,
    int type, int storage, ViewKey *key) {
    (void)image; (void)view; (void)has_view; (void)type; (void)storage; (void)key; abort();
}
static int vk_texture_record_selected_sample_locked(const VkTexturePresentSource *source, VkImage image) {
    assert(ordinary_texel_control && source->image == image && image == current_image);
    ordinary_reads++;
    return 1;
}
static void vk_texture_alias_read_use(VkTextureAliasRead *alias, uint64_t fence) { (void)alias; (void)fence; abort(); }
static void twin_capture_view(const VkTexImage *image, const view_t *view, int has_view, uint64_t handle,
    const GfxShaderRes *res, VkTextureReadKey key, int historical, VkDescriptorImageInfo *out) {
    assert(ordinary_texel_control && image->image == current_image && view && !has_view && handle == 17 &&
        res->type == 3 && key.registration == current_registration && !historical && out);
}
static void vk_texture_registration_note_locked(int slot, const VkTexImage *image, const view_t *view, int has_view) {
    (void)slot; (void)image; (void)view; (void)has_view; abort();
}
static int vk_texture_flat3d_selected_current_locked(VkTextureReadKey key, VkImage image) {
    (void)key; (void)image; abort();
}
static int vk_exec_view_aspect(const VkTexImage *image, const view_t *view, int has_view) {
    (void)view; (void)has_view;
    return image->aspect;
}
static VkSampler vk_sampler_point_by_id_locked(int id) { assert(id == 0); return 70; }
static VkSampler vk_sampler_compare_by_id_locked(int id) { (void)id; abort(); }
static VkSampler vk_sampler_detail_by_id_locked(int id, int skip) { (void)id; (void)skip; abort(); }
static VkSampler vk_sampler_by_id_locked(int id) { (void)id; abort(); }
static int dummy_index(VkImageViewType type, int ms) { (void)type; (void)ms; abort(); }
static VkImageView storage_dummy(int id, int format) { (void)id; (void)format; abort(); }
"""

CASES = r"""
int main(void) {
    const GfxShaderRes resource = {.slot = 7, .type = 3, .exact = 1}, other = {.slot = 8};
    const VkTexturePresentSource held = {.image = 101};
    const VkPipeBinding selected = {.res = &resource, .stage = GFX_STAGE_VS, .kind = GFX_RES_TEX,
        .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .count = 1, .binding = 2};
    VkPipeInfo info = {.layout = 1, .n_sets = 1, .n_bindings = 1, .bindings = {selected}};
    const int first_revision = 1, changed_revision = 2;
    const RcInterval demand = {.mapping = 13, .range = {4, 1024, 64}};
    RcInterval seed_range = {.mapping = 30, .range = {4, 1088, 64}};
    RcRegistryImageWriteExtent depth_extent = {.interval = {.mapping = 31, .range = {5, 1024, 64}}};
    const VkTextureTargetLoan depth_loan = {.extents = &depth_extent, .count = 1};
    VkTextureAliasRead seed = {.intervals = &seed_range, .count = 1};
    VkExecGraphicsPass pass = {
        .shape = {.color = {{201, 0}}, .depth = {202, 0}},
        .bindings = {{.binding.image = 201, .load_seed = &seed},
                     [8] = {.binding.image = 202, .extents = &depth_extent, .extent_count = 1}},
        .loan = {[8] = &depth_loan},
    };
    GraphicsSample sample = {.resource = &resource, .handle = 17, .sample.registration = 301,
        .image = {.image = 101, .aspect = VK_IMAGE_ASPECT_DEPTH_BIT}, .geometry = {.registration = 301, .mapping = demand}, .hold = &held};
    VkExecGraphicsDraw draw = {.pass = &pass, .samples = &sample, .sample_count = 1, .opcode = 26,
        .use = {.revision = &first_revision, .info = &info, .vs_query_size_lod0_mask = UINT64_C(1) << 2},
        .feedback_selected = 1};
    VkExec ex = {0};

    assert(vk_exec_graphics_pass_geometry_read(&ex, &pass, 101, demand, 0));
    assert(graphics_geometry_final_current(&ex, &draw, 0));
    assert(!vk_exec_graphics_pass_geometry_read(&ex, NULL, 101, demand, 0));
    assert(!vk_exec_graphics_pass_geometry_read(&ex, &pass, 0, demand, 0));
    assert(!vk_exec_graphics_pass_geometry_read(&ex, &pass, 101, (RcInterval){0}, 0));
    assert(!vk_exec_graphics_pass_geometry_read(&ex, &pass, 201, demand, 0));
    pass.bindings[0].binding.image = 999;
    assert(!graphics_geometry_final_current(&ex, &draw, 0));
    pass.bindings[0].binding.image = 201;
    seed_range.range.offset--;
    assert(!graphics_geometry_final_current(&ex, &draw, 0)); // Backing overlap through another mapping.
    seed_range.range.offset++;
    seed_range.mapping = 0;
    assert(!graphics_geometry_final_current(&ex, &draw, 0));
    seed_range.mapping = 30;
    seed_range.range.size = 0;
    assert(!graphics_geometry_final_current(&ex, &draw, 0));
    seed_range.range.size = 64;
    seed.intervals = NULL;
    assert(!graphics_geometry_final_current(&ex, &draw, 0));
    seed.intervals = &seed_range;
    pass.loan[8] = NULL;
    assert(!graphics_geometry_final_current(&ex, &draw, 0));
    pass.loan[8] = &depth_loan;
    depth_extent.interval.range.backing = 4;
    assert(!graphics_geometry_final_current(&ex, &draw, 0));
    depth_extent.interval.range.backing = 5;
    pass.shape.color[0].bc_backing = 444;
    assert(!graphics_geometry_final_current(&ex, &draw, 0));
    pass.shape.color[0].bc_backing = 0;
    pass.shape.feedback_count = 1;
    pass.shape.feedback_images[0] = 101;
    assert(!graphics_geometry_final_current(&ex, &draw, 0));
    pass.shape.feedback_count = 0;
    ex.nfbl = 1;
    ex.fbl_img[0] = 101;
    assert(!graphics_geometry_final_current(&ex, &draw, 0));
    ex.nfbl = 0;
    ex.tonemap_extent.width = 1920;
    assert(!graphics_geometry_final_current(&ex, &draw, 0));
    ex.tonemap_extent.width = 0;
    ex.hud_alpha_attachment = 1;
    assert(!graphics_geometry_final_current(&ex, &draw, 0));
    ex.hud_alpha_attachment = 0;

    assert(graphics_geometry_final_binding(&draw.use, &sample));
    draw.use.revision = &changed_revision;
    draw.use.vs_query_size_lod0_mask = UINT64_C(1) << 1;
    assert(!graphics_geometry_final_binding(&draw.use, &sample)); // Changed pipeline lacks exact attestation.
    draw.use.vs_query_size_lod0_mask = UINT64_C(1) << 2;
    info.bindings[0].binding = 1;
    assert(!graphics_geometry_final_binding(&draw.use, &sample));
    info.bindings[0] = selected;
    info.bindings[0].res = &other;
    assert(!graphics_geometry_final_binding(&draw.use, &sample));
    info.bindings[0] = selected;
    info.n_bindings = 2;
    info.bindings[1] = (VkPipeBinding){.res = &other, .kind = GFX_RES_IMG};
    assert(!graphics_geometry_final_binding(&draw.use, &sample)); // Even a distinct storage image is vetoed.
    info.bindings[1] = selected;
    assert(!graphics_geometry_final_binding(&draw.use, &sample));
    info.n_bindings = 1;
    assert(graphics_geometry_final_current(&ex, &draw, 0));

    graphics_current = &draw;
    VkTextureSample out = {0};
    VkTextureAliasRead *alias = NULL;
    const VkTexturePresentSource *source = NULL;
    assert(graphics_sample(&ex, NVN_STAGE_VS, 0, &resource, 17, &out, &alias, &source) == 2);
    assert(out.image == &sample.image && out.registration == 301 && source == &held && !alias);
    assert(graphics_sample(&ex, NVN_STAGE_VS, 1, &resource, 17, &out, &alias, &source) == -1);
    assert(!strcmp(reason, "prepared sampled image receipt missing"));
    assert(graphics_sample(&ex, NVN_STAGE_VS, 0, &resource, 18, &out, &alias, &source) == -1);
    current_registration++;
    assert(graphics_sample(&ex, NVN_STAGE_VS, 0, &resource, 17, &out, &alias, &source) == -1);
    assert(!strcmp(reason, "retained texture geometry changed"));
    current_registration--;
    current_image++;
    assert(!graphics_geometry_final_current(&ex, &draw, 0));
    current_image--;

    ex.in_pass = 1;
    ex.pass_ncolor = 1;
    ex.color[0] = (VkExecTarget){101, 0};
    assert(!graphics_geometry_final_current(&ex, &draw, 0)); // Previous active pass owns this native image.
    ex.color[0] = pass.shape.color[0];
    ex.depth = pass.shape.depth;
    ex.target_bindings[0] = pass.bindings[0];
    ex.target_bindings[8] = pass.bindings[8];
    ex.target_binding_mask = 1u;
    pass.activated = 1;
    assert(!graphics_geometry_final_current(&ex, &draw, 1)); // Active depth identity was not published.
    ex.target_binding_mask |= 1u << 8;
    assert(graphics_geometry_final_current(&ex, &draw, 1));
    draw.geometry_recording_attested = 1;
    assert(graphics_sample(&ex, NVN_STAGE_VS, 0, &resource, 17, &out, &alias, &source) == 2);
    ex.target_bindings[8].binding.image = 999;
    assert(graphics_sample(&ex, NVN_STAGE_VS, 0, &resource, 17, &out, &alias, &source) == -1);
    assert(unavailable == 4 && !strcmp(reason, "retained texture geometry changed"));

    const DrawCtx context = {0};
    ex.dc = &context;
    ex.cb = 80;
    ex.fence = 3;
    recording = &ex;
    pass.shape.depth.image = ex.depth.image = 101;
    pass.bindings[8].binding.image = ex.target_bindings[8].binding.image = 101;
    ex.nfbl = 1;
    ex.fbl_img[0] = 101;
    ex.fb_dirty = ex.fb_read = 1;
    draw.geometry_recording_attested = 0;
    assert(graphics_sample_shape_current(&sample)); // Exact source is the active attachment.
    assert(!graphics_geometry_peers_current(&ex, &draw, &sample, 1));
    assert(!graphics_geometry_final_current(&ex, &draw, 1));
    VkDescriptorImageInfo descriptor = {0};
    VkImage native = 0;
    VkTextureReadKey key = {0};
    int recorded_source = 0;
    assert(!tex_info_full_sample(&ex, NVN_STAGE_VS, 0, &resource, 17, 0, 0, &descriptor, &native, 1,
        &key, NULL, NULL, &recorded_source) && !descriptor.imageView && !native && !recorded_source);

    VkTextureQueryImage carrier = {.source = 101, .image = 501, .view = 601, .registration = 301};
    sample.sample.query_image = &carrier;
    assert(graphics_sample_shape_current(&sample) && graphics_geometry_peers_current(&ex, &draw, &sample, 1));
    assert(graphics_geometry_final_current(&ex, &draw, 1));
    assert(!tex_info_full_sample(&ex, NVN_STAGE_VS, 0, &resource, 17, 0, 0, &descriptor, &native, 1,
        &key, NULL, NULL, &recorded_source) && !descriptor.imageView && !native);
    assert(!strcmp(reason, "retained texture query carrier unavailable"));

    draw.use.vs_query_size_lod0_mask = UINT64_C(1) << 1;
    assert(!graphics_geometry_final_current(&ex, &draw, 1));
    assert(!tex_info_full_sample(&ex, NVN_STAGE_VS, 0, &resource, 17, 0, 0, &descriptor, &native, 1,
        &key, NULL, NULL, &recorded_source) && !descriptor.imageView && !native);
    assert(!strcmp(reason, "retained texture geometry changed"));
    draw.use.vs_query_size_lod0_mask = UINT64_C(1) << 2;
    current_registration++;
    assert(!graphics_sample_shape_current(&sample) && !graphics_geometry_final_current(&ex, &draw, 1));
    assert(!sample_query_images_record(&ex, &sample, 1));
    assert(!transitions && !carrier.cb && ex.in_pass && ex.recorded == 3);
    current_registration--;

    assert(sample_query_images_record(&ex, &sample, 1));
    assert(transitions == 1 && ex.in_pass && !ex.fb_read && !ex.fb_dirty && ex.recorded == 7);
    assert(render_ends == 2 && render_begins == 2 && render_barriers == 2 && !texture_locked);
    assert(sample_query_images_record(&ex, &sample, 1) && transitions == 1 && render_ends == 2);
    assert(graphics_geometry_final_current(&ex, &draw, 1));
    draw.geometry_recording_attested = 1;
    assert(!tex_info_full_sample(&ex, NVN_STAGE_VS, 0, &resource, 17, 0, 0, &descriptor, &native, 1,
        &key, NULL, NULL, &recorded_source));
    assert(native == carrier.image && native != ex.depth.image && descriptor.imageView == carrier.view &&
        descriptor.imageLayout == VK_IMAGE_LAYOUT_GENERAL && descriptor.sampler == 70 && recorded_source && !key.registration);

    draw.geometry_recording_attested = 0;
    draw.use.vs_query_size_lod0_mask = UINT64_C(1) << 1;
    descriptor = (VkDescriptorImageInfo){0};
    assert(!tex_info_full_sample(&ex, NVN_STAGE_VS, 0, &resource, 17, 0, 0, &descriptor, &native, 1,
        &key, NULL, NULL, &recorded_source) && !descriptor.imageView && !native && !recorded_source);
    assert(!strcmp(reason, "retained texture geometry changed")); // The carrier view already exists.
    draw.use.vs_query_size_lod0_mask = UINT64_C(1) << 2;
    vk_dlaa_swap_img = 1;
    assert(!graphics_geometry_final_current(&ex, &draw, 1));
    vk_dlaa_swap_img = 0;
    assert(graphics_geometry_final_current(&ex, &draw, 1));
    sample.stage = 2;
    info.bindings[0].stage = GFX_STAGE_FS;
    info.bindings[0].binding = 5;
    assert(!graphics_geometry_final_current(&ex, &draw, 1)); // Vertex proof cannot authorize this fragment binding.
    draw.use.fs_query_size_lod0_mask = UINT64_C(1) << 2;
    assert(!graphics_geometry_final_current(&ex, &draw, 1)); // The packed layout moved the fragment binding.
    draw.use.fs_query_size_lod0_mask = UINT64_C(1) << 5;
    assert(graphics_geometry_final_current(&ex, &draw, 1));
    info.n_sets = 2;
    assert(!graphics_geometry_final_current(&ex, &draw, 1)); // An independent fragment layout uses set 1.
    info.bindings[0].set = 1;
    info.bindings[0].binding = 2;
    draw.use.fs_query_size_lod0_mask = UINT64_C(1) << 2;
    assert(graphics_geometry_final_current(&ex, &draw, 1));
    descriptor = (VkDescriptorImageInfo){0};
    assert(!tex_info_full_sample(&ex, NVN_STAGE_FS, 0, &resource, 17, 0, 0, &descriptor, &native, 1,
        &key, NULL, NULL, &recorded_source) && native == carrier.image && descriptor.imageView == carrier.view);
    draw.use.fs_query_size_lod0_mask = UINT64_C(1) << 3;
    assert(!graphics_geometry_final_current(&ex, &draw, 1));
    draw.use.fs_query_size_lod0_mask = UINT64_C(1) << 2;
    descriptor = (VkDescriptorImageInfo){0};
    current_registration++;
    assert(!graphics_geometry_final_current(&ex, &draw, 1));
    assert(!tex_info_full_sample(&ex, NVN_STAGE_FS, 0, &resource, 17, 0, 0, &descriptor, &native, 1,
        &key, NULL, NULL, &recorded_source) && !descriptor.imageView && !native && !recorded_source);
    assert(!strcmp(reason, "retained texture geometry changed") && !ordinary_reads);
    current_registration--;

    // A positively selected cold registration has no image or original hold.
    sample.geometry.registration_only = 1;
    sample.image.image = carrier.source = current_image = 0;
    sample.hold = NULL;
    assert(graphics_geometry_final_current(&ex, &draw, 1));
    assert(!tex_info_full_sample(&ex, NVN_STAGE_FS, 0, &resource, 17, 0, 0, &descriptor, &native, 1,
        &key, NULL, NULL, &recorded_source));
    assert(native == carrier.image && descriptor.imageView == carrier.view && recorded_source && !key.registration);
    sample.image.image = carrier.source = current_image = 101;
    sample.hold = &held;
    ordinary_texel_control = 1;
    sample.geometry = (VkTextureSampleGeometry){0};
    sample.sample.query_image = NULL;
    assert(graphics_sample_shape_current(&sample) && sample_query_images_record(&ex, &sample, 1));
    assert(!tex_info_full_sample(&ex, NVN_STAGE_FS, 0, &resource, 17, 0, 0, &descriptor, &native, 1,
        &key, NULL, NULL, &recorded_source));
    assert(native == ex.depth.image && descriptor.imageView == 701 && ordinary_reads == 1 && recorded_source &&
        descriptor.imageLayout == VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT &&
        key.registration == current_registration && transitions == 1);

    puts("geometry draw: disjoint peers, active attachment isolation, exact descriptor admission and native transition");
    return 0;
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path(__file__).resolve().parents[1] / "runtime")
    args = parser.parse_args()
    draw = (args.sources / "vk_exec_draw.c").read_text(encoding="utf-8")
    recorder = (args.sources / "vk_exec.c").read_text(encoding="utf-8")
    selected = [function(recorder, name) for name in (
        "vk_exec_graphics_pass_geometry_read", "vk_exec_record_between_renderings",
    )]
    selected.extend(function(draw, name) for name in (
        "in_fbl", "graphics_sample_shape_current", "graphics_geometry_peers_current",
        "graphics_geometry_final_binding", "graphics_geometry_final_current", "graphics_sample", "tex_info_full_sample",
    ))
    query_commands = re.search(r"^typedef struct \{[^{}]*\} SampleQueryCommands;", draw, re.M)
    assert query_commands
    selected.append(query_commands.group())
    selected.extend(function(draw, name) for name in (
        "sample_query_record_outside_rendering", "sample_query_images_record",
    ))
    cc = os.environ.get("CC") or shutil.which("clang")
    if not cc:
        raise SystemExit("Clang is required (set CC)")
    with tempfile.TemporaryDirectory(prefix="geometry-draw-") as temp:
        work = Path(temp)
        c = work / "check.c"
        c.write_text("\n".join((STUBS, *selected, CASES)), encoding="utf-8")
        exe = work / ("check.exe" if os.name == "nt" else "check")
        subprocess.run([cc, "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                        "-fsanitize-trap=undefined", "-I", str(args.sources.resolve()), str(c), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True, timeout=20)


if __name__ == "__main__":
    main()

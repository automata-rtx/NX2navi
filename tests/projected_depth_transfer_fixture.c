#define VK_NO_PROTOTYPES
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "tex.h"
#include "vk.h"
#include "vk_exec.h"

#define SOURCE ((VkImage)(uintptr_t)41)
#define DESTINATION ((VkImage)(uintptr_t)42)
#define SCRATCH ((VkImage)(uintptr_t)43)
#define VK_TEXTURE_ALIAS_MAX_BYTES (64u * 1024u * 1024u)
#define warn_once(category, ...) ignore_warning(__VA_ARGS__)

struct VkTexturePresentSource { int ref; };
static struct VkTexturePresentSource source_hold;
static VkTexImage selected;
static VkCtx device;
static const tex_t *source_tex, *destination_tex;
static NVNcopyRegion expected_source, expected_destination;
static int proved, destination_captures, copies, blits, scratch_barriers, blit_features = 1;
static int source_authoritative = 1, same_image;
static VkImageAspectFlags selected_aspect=VK_IMAGE_ASPECT_DEPTH_BIT;

static void ignore_warning(const char *format, ...) { (void)format; }
VkCtx *vk_ctx(void) { return &device; }
int vk_texture_deferred_take(void) { return 0; }
void vk_exec_barrier(VkExec *ex) { assert(ex); }
static uint32_t max1(uint32_t value) { return value ? value : 1; }
static inline int tex_mirrored(const tex_t *tex) { return tex && !nvn_gpu_only((uintptr_t)tex->mem); }

int tex_vk_format(int format, TexVkFormat *out) {
    if (format == NVN_FORMAT_RGBA32F) {
        *out = (TexVkFormat){.vk = VK_FORMAT_R32G32B32A32_SFLOAT, .ideal = VK_FORMAT_R32G32B32A32_SFLOAT};
        return 1;
    }
    assert(format == NVN_FORMAT_DEPTH32F);
    *out = (TexVkFormat){.vk = VK_FORMAT_D32_SFLOAT, .ideal = VK_FORMAT_D32_SFLOAT, .depth = 1};
    return 1;
}

int nvn_fmt_bpb(int format, int *width, int *height) {
    assert(format == NVN_FORMAT_DEPTH32F || format == NVN_FORMAT_RGBA32F);
    if (width) *width = 1;
    if (height) *height = 1;
    return format == NVN_FORMAT_RGBA32F ? 16 : 4;
}

VkTexturePresentSource *vk_texture_transfer_read_capture(const tex_t *tex, VkTexImage *image) {
    assert(tex == source_tex);
    *image = selected;
    return &source_hold;
}

VkTexturePresentSource *vk_texture_transfer_read_select(const tex_t *tex, VkTexImage *image) {
    return vk_texture_transfer_read_capture(tex, image);
}

int vk_texture_transfer_read_prove(const tex_t *tex, const VkTexImage *image, VkImageSubresourceLayers layers,
                                  const NVNcopyRegion *box) {
    assert(tex == source_tex && image->image == SOURCE && layers.aspectMask == selected_aspect &&
           layers.mipLevel == 0 && layers.baseArrayLayer == 0 && layers.layerCount == 1);
    assert(box->xoffset == expected_source.xoffset && box->yoffset == expected_source.yoffset &&
           box->zoffset == 0 && box->width == expected_source.width && box->height == expected_source.height && box->depth == 1);
    ++proved;
    return source_authoritative ? 1 : -1;
}

int vk_texture_projected_transfer_capture(const tex_t *tex, const view_t *view, const NVNcopyRegion *box,
                                         VkTexImage *shape, VkTextureProjectedTransfer **out) {
    assert(tex == destination_tex && view && view->set_format && view->format == tex->d.format);
    assert(proved && shape->format == selected.format && shape->aspect == selected_aspect);
    assert(box->xoffset == expected_destination.xoffset && box->yoffset == expected_destination.yoffset &&
           box->zoffset == 0 && box->width == expected_destination.width &&
           box->height == expected_destination.height && box->depth == 1);
    shape->image = same_image ? SOURCE : DESTINATION;
    *out = (VkTextureProjectedTransfer *)(uintptr_t)1;
    ++destination_captures;
    return 1;
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties(VkPhysicalDevice phys, VkFormat format,
                                                              VkFormatProperties *props) {
    assert(phys == device.phys && format == VK_FORMAT_D32_SFLOAT);
    *props = (VkFormatProperties){.optimalTilingFeatures = blit_features ?
        VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT : 0};
}

VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo *dependency) {
    assert(!cb && dependency->imageMemoryBarrierCount == 1 &&
           dependency->pImageMemoryBarriers->image == SCRATCH &&
           dependency->pImageMemoryBarriers->subresourceRange.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT);
    ++scratch_barriers;
}

VKAPI_ATTR void VKAPI_CALL vkCmdCopyImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst,
                                         VkImageLayout dl, uint32_t count, const VkImageCopy *region) {
    assert(!cb && (src == SOURCE || src == SCRATCH) && (dst == DESTINATION || dst == SOURCE || dst == SCRATCH));
    assert(sl == VK_IMAGE_LAYOUT_GENERAL && dl == VK_IMAGE_LAYOUT_GENERAL && count == 1);
    assert(region->srcSubresource.aspectMask == selected_aspect &&
           region->dstSubresource.aspectMask == selected_aspect &&
           region->extent.depth == 1 && region->extent.width == (unsigned)expected_source.width &&
           region->extent.height == (unsigned)expected_source.height);
    ++copies;
}

VKAPI_ATTR void VKAPI_CALL vkCmdBlitImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst,
                                         VkImageLayout dl, uint32_t count, const VkImageBlit *region, VkFilter filter) {
    assert(!cb && src == SOURCE && dst == DESTINATION && sl == dl && count == 1 && filter == VK_FILTER_NEAREST &&
           region->srcSubresource.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT &&
           region->dstSubresource.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT);
    ++blits;
}

VKAPI_ATTR void VKAPI_CALL vkCmdResolveImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst,
                                            VkImageLayout dl, uint32_t count, const VkImageResolve *region) {
    (void)cb; (void)src; (void)sl; (void)dst; (void)dl; (void)count; (void)region;
    assert(!"ordinary D32 cannot use vkCmdResolveImage");
}

/* Production functions */


int vk_texture_transfer_input_current(const VkTextureTransferInput *input) {
    assert(!input->texture);
    return 1;
}

static int test_mirrored_full_color_route(void) {
    tex_t source = {.mem = (void *)(uintptr_t)RT_GPU_VA_BASE,
        .d = {.target = NVN_TEXTURE_TARGET_2D, .format = NVN_FORMAT_RGBA32F, .width = 1, .height = 1, .levels = 1},
        .L = {.layers = 1, .size = 512}};
    tex_t destination = source;
    destination.mem = (void *)(uintptr_t)0x41000000;
    destination.d.flags = NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET;
    destination.L.size = 128;
    source_tex = &source; destination_tex = &destination;
    selected_aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    selected = (VkTexImage){.image = SOURCE, .type = VK_IMAGE_TYPE_2D, .width = 1, .height = 1,
        .levels = 1, .layers = 1, .samples = 1, .format = VK_FORMAT_R32G32B32A32_SFLOAT, .aspect = selected_aspect};
    expected_source = expected_destination = (NVNcopyRegion){0,0,0,1,1,1};
    const view_t identity = {.set_format = 1, .format = NVN_FORMAT_RGBA32F, .set_levels = 1, .nlevels = 1};
    p_copy_t2t op = {.src = &source, .dst = &destination, .dv = identity, .has_dv = 1,
        .sr = expected_source, .dr = expected_destination};
    VkExecProjectedTransfer plan = {0};
    VkExec exec = {0};
    if (projected_t2t_capture(&exec, &op, &plan) != 1 || !plan.target) {
        fputs("FAIL ordinary full mirrored color T2T has no selected output\n", stderr);
        return 1;
    }
    assert(plan.image.op == VK_EXEC_IMAGE_COPY && plan.image.source_hold == &source_hold && proved == 1 &&
           destination_captures == 1 && !plan.image.scratch_recipe.sType);
    image_copy_record(&exec, &plan.image, SOURCE, DESTINATION);
    assert(copies == 1);
    plan = (VkExecProjectedTransfer){0};
    op.dv.format = NVN_FORMAT_R8;
    assert(projected_t2t_capture(&exec, &op, &plan) == 0 && !plan.target && destination_captures == 1);
    puts("PASS ordinary color T2T selected actual source and destination plus identity-view refusal");
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--host-color")) return test_mirrored_full_color_route();
    tex_t source = {.mem = (void *)(uintptr_t)RT_GPU_VA_BASE,
        .d = {.target = NVN_TEXTURE_TARGET_2D, .format = NVN_FORMAT_DEPTH32F, .width = 128, .height = 128, .levels = 1},
        .L = {.layers = 1, .size = 65536}};
    tex_t destination = source;
    destination.mem = (void *)(uintptr_t)(RT_GPU_VA_BASE + 65536);
    source_tex = &source;
    destination_tex = &destination;
    selected = (VkTexImage){.image = SOURCE, .type = VK_IMAGE_TYPE_2D, .width = 128, .height = 128,
        .levels = 1, .layers = 1, .samples = 1, .format = VK_FORMAT_D32_SFLOAT, .aspect = VK_IMAGE_ASPECT_DEPTH_BIT};
    const view_t identity = {.set_format = 1, .format = NVN_FORMAT_DEPTH32F, .set_levels = 1, .nlevels = 1,
                              .set_layers = 1, .nlayers = 1};
    expected_source = (NVNcopyRegion){5, 7, 0, 13, 17, 1};
    expected_destination = (NVNcopyRegion){11, 19, 0, 13, 17, 1};
    p_copy_t2t op = {.src = &source, .dst = &destination, .sv = identity, .dv = identity, .has_sv = 1, .has_dv = 1,
                      .sr = expected_source, .dr = expected_destination, .flags = NVN_COPY_FLAGS_LINEAR_FILTER_BIT};
    VkExecProjectedTransfer plan = {0};
    VkExec exec = {0};
    if (projected_t2t_capture(&exec, &op, &plan) != 1 || !plan.target) {
        fputs("FAIL actual D32 partial copy does not get a projected receipt\n", stderr);
        return 1;
    }
    assert(plan.image.op == VK_EXEC_IMAGE_COPY && plan.image.source_hold == &source_hold &&
           !plan.image.scratch_recipe.sType && destination_captures == 1);
    image_copy_record(&exec, &plan.image, SOURCE, DESTINATION);
    assert(copies == 1);

    same_image = 1;
    plan = (VkExecProjectedTransfer){0};
    assert(projected_t2t_capture(&exec, &op, &plan) == 1 && plan.image.scratch_recipe.format == VK_FORMAT_D32_SFLOAT);
    plan.image.scratch = SCRATCH;
    VkImage staged = texture_copy_stage_source(&exec, &plan.image);
    assert(staged == SCRATCH && scratch_barriers == 1 && copies == 2);
    image_copy_record(&exec, &plan.image, staged, SOURCE);
    assert(copies == 3);
    same_image = 0;

    op.dr = expected_destination = (NVNcopyRegion){11, 19, 0, 21, 23, 1};
    plan = (VkExecProjectedTransfer){0};
    assert(projected_t2t_capture(&exec, &op, &plan) == 1 && plan.image.op == VK_EXEC_IMAGE_BLIT);
    image_copy_record(&exec, &plan.image, SOURCE, DESTINATION);
    assert(blits == 1);

    plan = (VkExecProjectedTransfer){0};
    source_authoritative = 0;
    assert(projected_t2t_capture(&exec, &op, &plan) == -1 && !plan.target);
    source_authoritative = 1;
    blit_features = 0;
    assert(projected_t2t_capture(&exec, &op, &plan) == 0 && !plan.target);
    op.dv.base_level = 1;
    assert(projected_t2t_capture(&exec, &op, &plan) == 0 && !plan.target);
    op.dv.base_level = 0;
    selected.samples = 4;
    assert(projected_t2t_capture(&exec, &op, &plan) == 1 && !plan.target && destination_captures == 3);
    puts("PASS D32 box proof, exact retained transfer, depth overlap scratch, legal blit and unproved-input refusal");
}

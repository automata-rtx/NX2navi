#define VK_NO_PROTOTYPES
#include <assert.h>
#include <stdio.h>
#include "nvn_api.h"
#include "vk.h"
#include "vk_texture_alias_bytes.h"

enum { IMAGE_ALIAS_MAX_WRITE_FRAGMENTS = 8192 };
const int rt_nvn_api_version[2] = {200, 0};
const char *cfg_get(const char *name) { (void)name; return NULL; }

typedef struct {
    tex_t t;
    VkTexImage pub;
    TexVkFormat fmt;
    const RcInterval *image_private_mapping;
    uint32_t image_private_mapping_count, alias_full_count;
    RcInterval *alias_plane_demand;
    VkImageSubresourceRange alias_plane_range;
    uint32_t alias_plane_count;
    int bound_to;
} Tex;

static uint64_t producer_begin, producer_end;
static unsigned range_queries;

static int image_alias_exact_layout(const Tex *, VkTextureAliasLayout *);
static int image_alias_source_current(const Tex *entry) { return entry && !entry->bound_to; }
static int image_alias_whole_load_current(const Tex *entry, RcRegistryPosition read) {
    assert(entry && entry->bound_to && read.scene == 7 && read.ordinal == 20);
    return 1;
}
RcRegistry *rc_registry_runtime(void) { return (RcRegistry *)(uintptr_t)1; }
int rc_registry_native_image_covers(RcRegistry *registry, uint64_t image, const RcInterval *demand,
                                    uint32_t count, RcRegistryPosition read) {
    assert(registry == rc_registry_runtime() && image == 41 && read.scene == 7 && read.ordinal == 20 && count);
    ++range_queries;
    for (uint32_t i = 0; i < count; ++i) {
        assert(demand[i].mapping == 9 && demand[i].range.backing == 12);
        uint64_t off = demand[i].range.offset;
        if (off < producer_begin || off > producer_end || demand[i].range.size > producer_end - off) return 0;
    }
    return 1;
}

/* Production functions */

static int image_alias_exact_layout(const Tex *entry, VkTextureAliasLayout *layout) {
    return image_alias_layout_for(&entry->pub, &entry->t.d, &entry->fmt, entry->t.L.size, layout);
}

static Tex depth_image(uint32_t width, uint32_t height, uint32_t layers, uint32_t levels, int guest, VkFormat native) {
    Tex entry = {.t = {.mem = (void *)(uintptr_t)RT_GPU_VA_BASE,
        .d = {.width=width, .height=height, .depth=layers, .levels=levels, .format=guest,
              .target=layers == 1 ? NVN_TEXTURE_TARGET_2D : NVN_TEXTURE_TARGET_2D_ARRAY}},
        .pub = {.image=(VkImage)(uintptr_t)41, .type=VK_IMAGE_TYPE_2D, .width=width, .height=height, .depth=1,
                .layers=layers, .levels=levels, .samples=1, .format=native, .aspect=VK_IMAGE_ASPECT_DEPTH_BIT,
                .usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                .addr=RT_GPU_VA_BASE},
        .fmt = {.vk=native, .ideal=native, .depth=1}};
    nvn_tex_layout(&entry.t.d, &entry.t.L);
    return entry;
}

static void check_coverage(Tex *entry, uint32_t level, uint32_t layer, uint64_t expected_bytes) {
    const RcInterval mapping = {RT_GPU_VA_BASE, 9, {12, 0x200, entry->t.L.size}};
    entry->image_private_mapping = &mapping;
    entry->image_private_mapping_count = 1;
    const view_t view = {.set_format=1, .format=entry->t.d.format, .set_target=1, .target=NVN_TEXTURE_TARGET_2D,
        .set_levels=1, .base_level=level, .nlevels=1, .set_layers=1, .min_layer=layer, .nlayers=1};
    VkImageSubresourceRange selected;
    assert(vk_texture_alias_depth_attachment_range(&entry->t, &view, &selected));
    assert(selected.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT && selected.baseMipLevel == level &&
           selected.levelCount == 1 && selected.baseArrayLayer == layer && selected.layerCount == 1);
    VkTextureAliasLayout layout;
    assert(image_alias_exact_layout(entry, &layout) && texture_alias_clear_planes(selected, &entry->pub, &layout));
    assert(layout.count == 1 && layout.planes[0].copy.imageSubresource.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT &&
           layout.planes[0].copy.imageSubresource.mipLevel == level &&
           layout.planes[0].copy.imageSubresource.baseArrayLayer == layer);
    assert((uint64_t)layout.planes[0].row_bytes * layout.planes[0].rows == expected_bytes);
    uint32_t count;
    const RcInterval *demand = image_alias_plane_demand(entry, &selected, &count);
    assert(demand && count);
    uint64_t total = 0;
    producer_begin = mapping.range.offset + layer * entry->t.L.layer_size + nvn_tex_level_offset(&entry->t.d, level);
    producer_end = level + 1 == entry->t.d.levels ? mapping.range.offset + (layer + 1) * entry->t.L.layer_size :
        mapping.range.offset + layer * entry->t.L.layer_size + nvn_tex_level_offset(&entry->t.d, level + 1);
    for (uint32_t i = 0; i < count; ++i) {
        assert(demand[i].range.offset >= producer_begin && demand[i].range.offset + demand[i].range.size <= producer_end);
        total += demand[i].range.size;
    }
    assert(total == expected_bytes);
    const RcRegistryPosition read = {7,20};
    assert(image_alias_load_current(entry, &selected, read));
    VkImageSubresourceRange adjacent = selected;
    if (layer) {
        adjacent.baseArrayLayer--;
        assert(!image_alias_load_current(entry, &adjacent, read));
        adjacent = selected;
    }
    if (level) {
        adjacent.baseMipLevel--;
        assert(!image_alias_load_current(entry, &adjacent, read));
    }
    assert(image_alias_load_current(entry, &selected, read));
    free(entry->alias_plane_demand);
    entry->alias_plane_demand = NULL;
    entry->alias_plane_count = 0;
}

int main(void) {
    Tex stock = depth_image(1600, 1600, 3, 1, NVN_FORMAT_DEPTH16, VK_FORMAT_D16_UNORM);
    assert(stock.t.L.size == 0xf3c000 && stock.t.L.layer_size == 0x514000);
    check_coverage(&stock, 0, 1, 1600 * 1600 * 2);
    assert(producer_end - producer_begin > 1600 * 1600 * 2);
    Tex layered_mips = depth_image(65, 67, 3, 2, NVN_FORMAT_DEPTH16, VK_FORMAT_D16_UNORM);
    check_coverage(&layered_mips, 1, 2, 32 * 33 * 2);
    Tex d32 = depth_image(128, 128, 1, 1, NVN_FORMAT_DEPTH32F, VK_FORMAT_D32_SFLOAT);
    check_coverage(&d32, 0, 0, 128 * 128 * 4);
    assert(range_queries == 9);
    VkImageSubresourceRange range;
    view_t not_depth = {.set_format=1, .format=NVN_FORMAT_DEPTH32F};
    assert(!vk_texture_alias_depth_attachment_range(&stock.t, &not_depth, &range));
    stock.pub.aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
    VkTextureAliasLayout layout;
    assert(!image_alias_exact_layout(&stock, &layout));
    puts("PASS native D16 selected layer/mip and D32 byte footprints; neighboring layers, mips and padding excluded");
}

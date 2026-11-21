// Exact guest texture byte projection for retained native alias material.
#pragma once
#include <stdlib.h>
#include "tex.h"
#include "resource_core_observation.h"

enum { VK_TEXTURE_ALIAS_MAX_PLANES = 128 };
#define VK_TEXTURE_ALIAS_MAX_BYTES (64u * 1024u * 1024u)

typedef struct {
    VkBufferImageCopy copy;
    uint32_t rows, row_bytes, slices;
} VkTextureAliasPlane;

typedef struct {
    VkTextureAliasPlane planes[VK_TEXTURE_ALIAS_MAX_PLANES];
    uint32_t count;
    uint64_t bytes;
} VkTextureAliasLayout;

// Every plane in the host stage is tightly packed and begins at a legal Vulkan copy offset.
// The Tex owner has already checked the actual image format, shape, usage and lossless representation.
static int texture_alias_layout(const nvn_texdesc *desc, const VkTexImage *image, VkTextureAliasLayout *out) {
    *out = (VkTextureAliasLayout){0};
    if (!image->levels || !image->layers || image->levels > VK_TEXTURE_ALIAS_MAX_PLANES / image->layers ||
        image->levels >= 32 || image->levels != (unsigned)desc->levels ||
        image->width != (unsigned)desc->width || image->height != (unsigned)desc->height) return 0;
    int block_width, block_height;
    unsigned bpb = nvn_fmt_bpb(desc->format, &block_width, &block_height);
    if (!bpb || bpb > 16 || (bpb & (bpb - 1)) || block_width <= 0 || block_height <= 0) return 0;
    for (uint32_t lv = 0; lv < image->levels; lv++) {
        uint32_t w = image->width >> lv, h = image->height >> lv, depth = image->depth >> lv;
        if (!w) w = 1;
        if (!h) h = 1;
        if (!depth) depth = 1;
        uint64_t rows = (h + (uint64_t)block_height - 1) / block_height;
        uint64_t row_bytes = (w + (uint64_t)block_width - 1) / block_width * bpb;
        if (rows > UINT32_MAX || row_bytes > UINT32_MAX ||
            rows > VK_TEXTURE_ALIAS_MAX_BYTES / row_bytes) return 0;
        for (uint32_t layer = 0; layer < image->layers; layer++) {
            uint64_t size = rows * row_bytes;
            uint64_t aligned = (out->bytes + 15) & ~UINT64_C(15);
            if (aligned > VK_TEXTURE_ALIAS_MAX_BYTES ||
                size > (VK_TEXTURE_ALIAS_MAX_BYTES - aligned) / depth) return 0;
            VkTextureAliasPlane *plane = &out->planes[out->count++];
            *plane = (VkTextureAliasPlane){
                .copy = {.bufferOffset = aligned,
                    .imageSubresource = {image->aspect, lv, layer, 1},
                    .imageExtent = {w, h, depth}},
                .rows = rows, .row_bytes = row_bytes, .slices = depth,
            };
            out->bytes = aligned + size * depth;
        }
    }
    return out->count != 0;
}

// A temporary sampled receiver exposes only the registered view's subresources. Keep the exact buffer offsets
// for those planes; omitted planes carry neither pixel proof nor a native copy.
static int texture_alias_selected_planes(const view_t *view, const VkTexImage *image, VkTextureAliasLayout *layout) {
    if (!view || (!view->set_levels && !view->set_layers)) return 1;
    uint32_t mip = view->set_levels ? view->base_level : 0;
    uint32_t layer = view->set_layers ? view->min_layer : 0;
    if (mip >= image->levels || layer >= image->layers) return 0;
    uint32_t levels = image->levels - mip, layers = image->layers - layer;
    if (view->set_levels && view->nlevels && view->nlevels < levels) levels = view->nlevels;
    if (view->set_layers && view->nlayers && view->nlayers < layers) layers = view->nlayers;
    uint32_t kept = 0;
    layout->bytes = 0;
    for (uint32_t i = 0; i < layout->count; ++i) {
        const VkTextureAliasPlane plane = layout->planes[i];
        uint32_t m = plane.copy.imageSubresource.mipLevel, l = plane.copy.imageSubresource.baseArrayLayer;
        if (m < mip || m - mip >= levels || l < layer || l - layer >= layers) continue;
        layout->planes[kept++] = plane;
        layout->bytes = plane.copy.bufferOffset + (uint64_t)plane.rows * plane.row_bytes * plane.slices;
    }
    layout->count = kept;
    return kept != 0;
}

// Like a sampled view, an issued clear names complete planes. Validate the literal native subresource range
// before translating it to the shared selector; clipping an invalid range would grant unwritten bytes.
static int texture_alias_clear_planes(VkImageSubresourceRange range, const VkTexImage *image,
                                      VkTextureAliasLayout *layout) {
    if (range.aspectMask != image->aspect || !range.levelCount || !range.layerCount ||
        range.baseMipLevel >= image->levels || range.levelCount > image->levels - range.baseMipLevel ||
        range.baseArrayLayer >= image->layers || range.layerCount > image->layers - range.baseArrayLayer) return 0;
    const view_t view = {.set_levels = 1, .base_level = range.baseMipLevel, .nlevels = range.levelCount,
                         .set_layers = 1, .min_layer = range.baseArrayLayer, .nlayers = range.layerCount};
    return texture_alias_selected_planes(&view, image, layout);
}

static int texture_alias_view_covered(VkImageSubresourceRange range, const VkTextureAliasLayout *layout) {
    if ((range.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT && range.aspectMask != VK_IMAGE_ASPECT_DEPTH_BIT) ||
        !range.levelCount || !range.layerCount || range.levelCount > layout->count / range.layerCount) return 0;
    uint32_t found = 0;
    for (uint32_t i = 0; i < layout->count; ++i) {
        const VkImageSubresourceLayers layers = layout->planes[i].copy.imageSubresource;
        if (layers.aspectMask == range.aspectMask &&
            layers.mipLevel >= range.baseMipLevel && layers.mipLevel - range.baseMipLevel < range.levelCount &&
            layers.baseArrayLayer >= range.baseArrayLayer && layers.baseArrayLayer - range.baseArrayLayer < range.layerCount)
            found++;
    }
    return found == range.levelCount * range.layerCount;
}

static int texture_alias_project(const nvn_texdesc *desc, const VkTextureAliasLayout *layout,
                                 uint8_t *guest, uint8_t *linear, int to_linear) {
    for (uint32_t i = 0; i < layout->count; i++) {
        const VkTextureAliasPlane *p = &layout->planes[i];
        const uint32_t level = p->copy.imageSubresource.mipLevel, layer = p->copy.imageSubresource.baseArrayLayer;
        const uint64_t slice = (uint64_t)p->row_bytes * p->rows;
        for (uint32_t z = 0; z < p->slices; z++)
            if (nvn_texels_box(desc, level, layer, 0, 0, z, p->row_bytes, p->rows, guest,
                    linear + p->copy.bufferOffset + slice * z, p->row_bytes, to_linear) != 0)
                return 0;
    }
    return 1;
}

typedef struct {
    RcInterval *demands;
    const RcInterval *mapping;
    uint64_t address, bytes;
    uint32_t capacity, count, maps;
    int failed;
} VkTextureAliasDemand;

static void texture_alias_demand_range(void *context, uint64_t offset, uint64_t bytes) {
    VkTextureAliasDemand *demand = context;
    if (demand->failed) return;
    if (!bytes || offset >= demand->bytes || bytes > demand->bytes - offset) {
        demand->failed = 1;
        return;
    }
    uint64_t address = demand->address + offset;
    // Region order can restart for the next mip, layer or 3D slice.
    uint32_t lo = 0, hi = demand->maps;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        const RcInterval *current = &demand->mapping[mid];
        if (current->address + current->range.size <= address) lo = mid + 1;
        else hi = mid;
    }
    for (uint32_t i = lo; bytes && i < demand->maps; i++) {
        const RcInterval *current = &demand->mapping[i];
        if (current->address > address || demand->count == demand->capacity) break;
        uint64_t within = address - current->address, take = current->range.size - within;
        if (take > bytes) take = bytes;
        demand->demands[demand->count++] = (RcInterval){.address = address, .mapping = current->mapping,
            .range = {current->range.backing, current->range.offset + within, take}};
        address += take;
        bytes -= take;
    }
    if (bytes) demand->failed = 1;
}

static int texture_alias_demand_setup(const RcInterval *mapping, uint32_t map_count, uint64_t address,
                                      uint64_t bytes, RcInterval *out, uint32_t capacity, VkTextureAliasDemand *demand) {
    if (!map_count || !capacity || !bytes || bytes > UINT64_MAX - address) return 0;
    for (uint32_t i = 0; i < map_count; i++) {
        const RcInterval *map = &mapping[i];
        if (!map->mapping || !rc_range_valid(map->range) || map->range.size > UINT64_MAX - map->address ||
            (i && mapping[i - 1].address + mapping[i - 1].range.size > map->address)) return 0;
    }
    *demand = (VkTextureAliasDemand){.demands = out, .mapping = mapping, .address = address, .bytes = bytes,
        .capacity = capacity, .maps = map_count};
    return 1;
}

static int texture_alias_demand_box_plane(const nvn_texdesc *desc, const NVNcopyRegion *box, uint32_t pixel_bytes,
    uint32_t mip_level, uint32_t array_layer, const RcInterval *mapping, uint32_t map_count, uint64_t address,
    uint64_t bytes, RcInterval *out, uint32_t capacity, uint32_t *count) {
    *count = 0;
    VkTextureAliasDemand demand;
    const int volume = desc->target == NVN_TEXTURE_TARGET_3D;
    if (box->xoffset < 0 || box->yoffset < 0 || box->width <= 0 || box->height <= 0) return 0;
    uint32_t z = 0, slices = 1;
    if (volume) {
        if (array_layer || mip_level >= 32 || desc->depth <= 0 || box->zoffset < 0 || box->depth <= 0) return 0;
        uint32_t available = (uint32_t)desc->depth >> mip_level;
        if (!available) available = 1;
        z = (uint32_t)box->zoffset;
        slices = (uint32_t)box->depth;
        if (z >= available || slices > available - z) return 0;
    }
    const uint64_t x = (uint64_t)(uint32_t)box->xoffset * pixel_bytes;
    const uint64_t width = (uint64_t)(uint32_t)box->width * pixel_bytes;
    if (x > UINT32_MAX || !width || width > UINT32_MAX ||
        !texture_alias_demand_setup(mapping, map_count, address, bytes, out, capacity, &demand)) return 0;
    for (uint32_t slice = 0; slice < slices; slice++)
        if (nvn_texels_box_ranges(desc, mip_level, array_layer, (uint32_t)x, box->yoffset, z + slice,
                (uint32_t)width, box->height, texture_alias_demand_range, &demand) != 0 || demand.failed) return 0;
    if (!demand.count) return 0;
    *count = demand.count;
    return 1;
}

static int texture_alias_demand_box(const nvn_texdesc *desc, const NVNcopyRegion *box, uint32_t pixel_bytes,
    const RcInterval *mapping, uint32_t map_count, uint64_t address, uint64_t bytes, RcInterval *out, uint32_t capacity,
    uint32_t *count) {
    return texture_alias_demand_box_plane(desc, box, pixel_bytes, 0, 0, mapping, map_count, address, bytes, out, capacity, count);
}

static int texture_alias_demand(const nvn_texdesc *desc, const VkTextureAliasLayout *layout,
                                const RcInterval *mapping, uint32_t map_count, uint64_t address, uint64_t bytes,
                                RcInterval *out, uint32_t capacity, uint32_t *count) {
    VkTextureAliasDemand demand;
    *count = 0;
    if (!texture_alias_demand_setup(mapping, map_count, address, bytes, out, capacity, &demand)) return 0;
    for (uint32_t i = 0; !demand.failed && i < layout->count; i++) {
        const VkTextureAliasPlane *p = &layout->planes[i];
        for (uint32_t z = 0; !demand.failed && z < p->slices; z++)
            if (nvn_texels_slice_ranges(desc, p->copy.imageSubresource.mipLevel, p->copy.imageSubresource.baseArrayLayer,
                    z, texture_alias_demand_range, &demand)) return 0;
    }
    if (demand.failed || !demand.count) return 0;
    *count = demand.count;
    return 1;
}

typedef struct {
    const uint8_t *guest;
    uint64_t address, bytes;
} VkTextureAliasBytes;

typedef struct {
    RcSourceRange covered;
    uint64_t writer_address;
    uint32_t source, demand;
} VkTextureAliasSlice;

// A full ordered partition is required. A gap, a missing retained image or an overlapping claim can
// never be filled with zero bytes or a stale owner. Guest padding was excluded from the demand.
static int texture_alias_compose(const VkTextureAliasBytes *sources, uint32_t source_count,
                                 uint8_t *target_guest, uint64_t address, uint64_t bytes,
                                 const RcInterval *demands, uint32_t demand_count,
                                 const VkTextureAliasSlice *slices, uint32_t slice_count) {
    uint32_t next = 0;
    for (uint32_t i = 0; i < demand_count; i++) {
        const RcInterval *demand = &demands[i];
        if (!rc_range_valid(demand->range) || demand->address < address || demand->range.size > bytes ||
            demand->address - address > bytes - demand->range.size) return 0;
        uint64_t dst = demand->address - address, offset = demand->range.offset, remaining = demand->range.size;
        while (remaining) {
            if (next == slice_count) return 0;
            const VkTextureAliasSlice *slice = &slices[next++];
            if (slice->demand != i || slice->source >= source_count || !rc_range_valid(slice->covered) ||
                slice->covered.backing != demand->range.backing || slice->covered.offset != offset ||
                slice->covered.size > remaining) return 0;
            const VkTextureAliasBytes *source = &sources[slice->source];
            if (!source->guest || slice->writer_address < source->address || slice->covered.size > source->bytes ||
                slice->writer_address - source->address > source->bytes - slice->covered.size) return 0;
            memcpy(target_guest + dst, source->guest + (slice->writer_address - source->address), slice->covered.size);
            dst += slice->covered.size;
            offset += slice->covered.size;
            remaining -= slice->covered.size;
        }
    }
    return next == slice_count;
}

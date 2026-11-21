// Tegra X1 block-linear <-> linear copies.
// Contains code Copyright (c) 2026 jhaizhou (MIT; see THIRD_PARTY_NOTICES.md).
//
// Layout (from public descriptions):
//   GOB = 64 bytes x 8 rows = 512 bytes; offset of (x byte, y row) within a GOB =
//     (x%64/32)*256 + (y%8/2)*64 + (x%32/16)*32 + (y%2)*16 + x%16
//   block = 1 GOB wide x 2^th GOBs high x 2^td GOBs deep, ordered y first then z inside a block; blocks fill the level
//   in x -> y -> z row-major order.
// Level and layer start offsets and the block height and depth of a level come from nvn_tex.c; this file only does the
// addressing inside a level and the copy.
//
// Switch 2 (NVN2 titles, NVN API major version 200 and up) stores the 16-byte pieces of a GOB in another order:
//     (x%64/32)*256 + (y%8/4)*128 + (x%32/16)*64 + (y%4)*16 + x%16
// (four rows of one 16-byte column before the next column). Found on Ocarina of Time's packaged textures: with it the
// ASTC 4x4 title logo and BC4 masks decode without seams (an edge-continuity search over all 120 orders of the five
// address bits picks it for both; OCARINA-BRINGUP.md round 18). Block and level arithmetic are unchanged.
#include "host.h"
#include "tex.h"
#include <string.h>

static int gob_nvn2(void) {
    static int v = -1;
    if (v < 0) v = rt_nvn_api_version[0] >= 200;
    return v;
}

static uint32_t gob_off(uint32_t x, uint32_t y) {
    if (gob_nvn2()) return (x % 64 / 32) * 256 + (y % 8 / 4) * 128 + (x % 32 / 16) * 64 + (y % 4) * 16 + x % 16;
    return (x % 64 / 32) * 256 + (y % 8 / 2) * 64 + (x % 32 / 16) * 32 + (y % 2) * 16 + x % 16;
}

uint64_t nvn_bl_offset(uint32_t x, uint32_t y, uint32_t z, uint32_t row_bytes, uint32_t rows, unsigned th, unsigned td) {
    uint32_t bh = 1u << th, bd = 1u << td;
    uint64_t gobs_x = (row_bytes + 63) / 64, blocks_y = (rows + 8 * bh - 1) / (8 * bh);
    uint64_t block = 512ull * bh * bd;                                  // bytes in one block
    return (z / bd) * gobs_x * blocks_y * block                        // which block layer (along depth)
         + (y / (8 * bh)) * gobs_x * block                             // which block row
         + (x / 64) * block                                            // which block column
         + ((z % bd) * bh + (y % (8 * bh)) / 8) * 512ull               // which GOB within the block
         + gob_off(x, y);
}

static uint32_t mip(uint32_t s, unsigned l) { s >>= l; return s ? s : 1; }

// Geometry of one level: bytes per row of blocks, block rows, slices, where the level and layer start in the storage,
// and the level's block height and depth. -1 for textures the copy does not handle.
int nvn_deswizzle_geom(const nvn_texdesc *d, unsigned level, unsigned layer,
                       uint32_t *row_bytes, uint32_t *rows, uint32_t *slices, uint64_t *base, unsigned *th, unsigned *td, uint32_t *bpb) {
    if (d->target == 5 || d->target == 6 || d->target == 10 || nvn_tex_is_linear(d) || d->samples > 1) return -1;   // MS / BUFFER / LINEAR
    int bw, bh; *bpb = (uint32_t)nvn_fmt_bpb(d->format, &bw, &bh);
    if (!*bpb || level >= (d->levels ? d->levels : 1)) return -1;
    nvn_texlayout L; nvn_tex_layout(d, &L);
    if (layer >= L.layers) return -1;
    int one_d = d->target == 0 || d->target == 3;
    uint32_t w = mip(d->width, level), h = one_d ? 1 : mip(d->height, level);
    *slices = d->target == 2 ? mip(d->depth ? d->depth : 1, level) : 1;
    *row_bytes = (w + bw - 1) / bw * *bpb; *rows = (h + bh - 1) / bh;
    *base = (uint64_t)layer * L.layer_size + nvn_tex_level_geometry(d, &L, level, th, td);
    return 0;
}

// nvn_bl_offset per row instead of per 16 bytes: within a row, GOB column k starts k blocks further, and the four
// 16-byte segments of a GOB row (x % 64 = 0, 16, 32, 48) sit at 0, 32, 256 and 288 of it (Switch 2: 0, 64, 256, 320).
void tex_bl_copy(uint8_t *bl, uint8_t *lin, size_t pitch, uint32_t row_bytes, uint32_t rows, uint32_t slices,
                 unsigned th, unsigned td, int to_linear) {
    static const uint32_t seg_x1[4] = {0, 32, 256, 288}, seg_nvn2[4] = {0, 64, 256, 320};
    const uint32_t *seg = gob_nvn2() ? seg_nvn2 : seg_x1;
    uint64_t block = 512ull << th << td;
    for (uint32_t z = 0; z < slices; z++)
        for (uint32_t y = 0; y < rows; y++) {
            uint8_t *row = lin + ((size_t)z * rows + y) * pitch, *g = bl + nvn_bl_offset(0, y, z, row_bytes, rows, th, td);
            uint32_t x = 0;
            if (to_linear)
                for (; x + 64 <= row_bytes; x += 64, g += block)
                    for (int s = 0; s < 4; s++) memcpy(row + x + 16 * s, g + seg[s], 16);
            else
                for (; x + 64 <= row_bytes; x += 64, g += block)
                    for (int s = 0; s < 4; s++) memcpy(g + seg[s], row + x + 16 * s, 16);
            for (int s = 0; x < row_bytes; s++, x += 16) {   // a partial GOB at the row's end
                uint32_t n = row_bytes - x < 16 ? row_bytes - x : 16;
                if (to_linear) memcpy(row + x, g + seg[s], n); else memcpy(g + seg[s], row + x, n);
            }
        }
}

static int move(const nvn_texdesc *d, unsigned level, unsigned layer, uint8_t *bl, uint8_t *lin, size_t pitch, int to_linear) {
    if (nvn_tex_is_linear(d)) {
        if (level || layer || d->samples > 1 || (d->target != 0 && d->target != 1 && d->target != 7)) return -1;
        int bw, bh, bpb = nvn_fmt_bpb(d->format, &bw, &bh);
        if (!bpb) return -1;
        size_t row = (size_t)((d->width + bw - 1) / bw) * bpb;
        uint32_t rows = d->target == 0 ? 1 : (d->height + bh - 1) / bh, stride = nvn_tex_linear_stride(d);
        if (pitch < row || stride < row) return -1;
        for (uint32_t y = 0; y < rows; y++) {
            if (to_linear) memcpy(lin + y * pitch, bl + (size_t)y * stride, row);
            else memcpy(bl + (size_t)y * stride, lin + y * pitch, row);
        }
        return 0;
    }
    uint32_t row_bytes, rows, slices, bpb; uint64_t base; unsigned th, td;
    if (nvn_deswizzle_geom(d, level, layer, &row_bytes, &rows, &slices, &base, &th, &td, &bpb)) return -1;
    if (pitch < row_bytes) return -1;
    tex_bl_copy(bl + base, lin, pitch, row_bytes, rows, slices, th, td, to_linear);
    return 0;
}

typedef struct {
    uint32_t row_bytes, rows, stride;
    uint64_t base;
    unsigned th, td;
} TexelBoxLayout;

static int texel_box_layout(const nvn_texdesc *d, unsigned level, unsigned layer, uint32_t x, uint32_t y, uint32_t z,
                            uint32_t wb, uint32_t h, TexelBoxLayout *layout) {
    *layout = (TexelBoxLayout){0};
    if (nvn_tex_is_linear(d)) {
        if (level || layer || z || d->samples > 1 || (d->target != 0 && d->target != 1 && d->target != 7)) return -1;
        int bw, bh, bpb = nvn_fmt_bpb(d->format, &bw, &bh);
        if (!bpb) return -1;
        uint64_t row = ((d->width + (uint64_t)bw - 1) / bw) * bpb;
        layout->rows = d->target == 0 ? 1 : (d->height + (uint64_t)bh - 1) / bh;
        layout->stride = nvn_tex_linear_stride(d);
        if (layout->stride < row) return -1;
        layout->row_bytes = (uint32_t)row;
    } else {
        uint32_t slices, bpb;
        if (nvn_deswizzle_geom(d, level, layer, &layout->row_bytes, &layout->rows, &slices, &layout->base,
                                &layout->th, &layout->td, &bpb) || z >= slices) return -1;
    }
    return x + (uint64_t)wb > layout->row_bytes || y + (uint64_t)h > layout->rows ? -1 : 0;
}

int nvn_texels_box(const nvn_texdesc *d, unsigned level, unsigned layer, uint32_t x, uint32_t y, uint32_t z, uint32_t wb,
                   uint32_t h, void *storage, void *rows, size_t pitch, int to_linear) {
    uint8_t *st = storage, *ln = rows;
    TexelBoxLayout layout;
    if (texel_box_layout(d, level, layer, x, y, z, wb, h, &layout)) return -1;
    if (nvn_tex_is_linear(d)) {
        for (uint32_t r = 0; r < h; r++) {
            uint8_t *p = st + (size_t)(y + r) * layout.stride + x, *q = ln + r * pitch;
            if (to_linear) memcpy(q, p, wb); else memcpy(p, q, wb);
        }
        return 0;
    }
    for (uint32_t r = 0; r < h; r++) {
        uint8_t *q = ln + r * pitch;
        for (uint32_t i = 0; i < wb;) {
            uint32_t xb = x + i, n = 16 - xb % 16;   // contiguous up to the end of its 16-byte segment
            if (n > wb - i) n = wb - i;
            uint8_t *p = st + layout.base + nvn_bl_offset(xb, y + r, z, layout.row_bytes, layout.rows, layout.th, layout.td);
            if (to_linear) memcpy(q + i, p, n); else memcpy(p, q + i, n);
            i += n;
        }
    }
    return 0;
}

typedef struct {
    NvnTexelStorageRange emit;
    void *context;
    uint64_t offset, end;
} TexelRanges;

static void texel_range(TexelRanges *r, uint64_t offset, uint64_t size) {
    if (r->end != offset) {
        if (r->end > r->offset) r->emit(r->context, r->offset, r->end - r->offset);
        r->offset = offset;
    }
    r->end = offset + size;
}

int nvn_texels_box_ranges(const nvn_texdesc *d, unsigned level, unsigned layer, uint32_t x, uint32_t y, uint32_t z,
                          uint32_t wb, uint32_t h, NvnTexelStorageRange emit, void *context) {
    TexelBoxLayout layout;
    if (texel_box_layout(d, level, layer, x, y, z, wb, h, &layout)) return -1;
    if (!wb || !h || !emit) return 0;
    TexelRanges ranges = {.emit = emit, .context = context};
    if (nvn_tex_is_linear(d)) {
        for (uint32_t row = 0; row < h; row++)
            texel_range(&ranges, (uint64_t)(y + row) * layout.stride + x, wb);
    } else {
        // Use the actual GOB ordering from the copy address function, including the Switch 2 permutation.
        uint8_t segment_x[32], segment_y[32];
        for (uint32_t yy = 0; yy < 8; yy++)
            for (uint32_t xx = 0; xx < 64; xx += 16) {
                uint32_t i = gob_off(xx, yy) / 16;
                segment_x[i] = (uint8_t)xx;
                segment_y[i] = (uint8_t)yy;
            }
        uint32_t block_rows = 8u << layout.th;
        uint64_t end_x = (uint64_t)x + wb, end_y = (uint64_t)y + h;
        for (uint64_t by = y / block_rows * block_rows; by < end_y; by += block_rows) {
            uint64_t first_y = by > (y & ~7u) ? by : (y & ~7u);
            uint64_t last_y = by + block_rows < end_y ? by + block_rows : end_y;
            for (uint64_t gx = x & ~63u; gx < end_x; gx += 64)
                for (uint64_t gy = first_y; gy < last_y; gy += 8) {
                    uint64_t gob = layout.base + nvn_bl_offset((uint32_t)gx, (uint32_t)gy, z,
                        layout.row_bytes, layout.rows, layout.th, layout.td);
                    if (gx >= x && gx + 64 <= end_x && gy >= y && gy + 8 <= end_y) {
                        texel_range(&ranges, gob, 512);
                        continue;
                    }
                    for (uint32_t i = 0; i < 32; i++) {
                        uint64_t sy = gy + segment_y[i], sx = gx + segment_x[i];
                        if (sy < y || sy >= end_y || sx + 16 <= x || sx >= end_x) continue;
                        uint64_t lo = sx < x ? x : sx, hi = sx + 16 < end_x ? sx + 16 : end_x;
                        texel_range(&ranges, gob + i * 16 + lo - sx, hi - lo);
                    }
                }
        }
    }
    if (ranges.end > ranges.offset) emit(context, ranges.offset, ranges.end - ranges.offset);
    return 0;
}

int nvn_texels_slice_ranges(const nvn_texdesc *d, unsigned level, unsigned layer, uint32_t z,
                            NvnTexelStorageRange emit, void *context) {
    TexelBoxLayout layout;
    if (texel_box_layout(d, level, layer, 0, 0, z, 0, 0, &layout)) return -1;
    return nvn_texels_box_ranges(d, level, layer, 0, 0, z, layout.row_bytes, layout.rows, emit, context);
}

int nvn_deswizzle(const nvn_texdesc *d, unsigned level, unsigned layer, const void *src, void *dst, size_t dst_pitch) {
    return move(d, level, layer, (uint8_t *)src, (uint8_t *)dst, dst_pitch, 1);
}
int nvn_swizzle(const nvn_texdesc *d, unsigned level, unsigned layer, const void *src, size_t src_pitch, void *dst) {
    return move(d, level, layer, (uint8_t *)dst, (uint8_t *)src, src_pitch, 0);
}

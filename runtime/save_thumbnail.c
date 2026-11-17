#include "gfx.h"

// The manual and autosave draw in main+0x338b7b0 scales each source axis to the configured
// capture size. Keep the middle 16:9 of that image inside the existing 543x306 JPEG viewport.
void save_thumbnail_crop(Cpu *c) {
    uint64_t args = c->x[29] - 56;
    c->x[2] = args;   // displaced instruction, including for other capture targets
    if (LD32(c->x[19] + 1088) > 1) return;

    // The capture draw can already be sampling a resized copy of the frame. Its texture
    // dimensions and the host window's dimensions can both differ from the game picture.
    unsigned source_width, source_height;
    gfx_game_picture_size(&source_width, &source_height);
    uint64_t w9 = (uint64_t)source_width * 9, h16 = (uint64_t)source_height * 16;
    if (!w9 || !h16 || w9 == h16) return;
    int axis = w9 > h16 ? 0 : 1;
    float zoom = axis == 0 ? (float)w9 / h16 : (float)h16 / w9;

    const float *bounds = (const float *)(uintptr_t)(LD64(args + 8) + 8);
    float *scale = (float *)(uintptr_t)LD64(args + 16);
    float width = bounds[2] - bounds[0], height = bounds[3] - bounds[1];
    if (!(width > 0 && height > 0 && scale[0] > 0 && scale[1] > 0)) return;

    float center = (axis == 0 ? width : height) * 0.5f;
    scale[axis] *= zoom;

    // The offset can point at a shared constant. Use the stack space of the render state
    // already bound by the caller for a private copy, then zoom around the viewport center.
    const float *offset = (const float *)(uintptr_t)LD64(args + 24);
    float *local = (float *)(uintptr_t)(c->sp + 16);
    memcpy(local, offset, 2 * sizeof(float));
    local[axis] = center + (local[axis] - center) * zoom;
    ST64(args + 24, (uintptr_t)local);
}

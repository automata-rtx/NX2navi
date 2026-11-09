// Attachment history for draws whose shader or pipeline is compiling.
// Contains code Copyright (c) 2026 jhaizhou, Copyright (c) 2026 Peiikai (MIT; see THIRD_PARTY_NOTICES.md).
#include <stdio.h>
#include <stdlib.h>
#include "host.h"
#include "vk_exec.h"

// ---- target history
// The presented frame (gfx_counters.frames + 1; 0: never) each target subresource was last rendered into, and the one
// before. prepare_draw lets a draw into a target the previous frame did not render (a one-shot draw: a volume or LUT
// generated once) wait for its shaders and pipeline instead of skipping it. Per subresource, not per image: a volume
// may be generated a few slices per frame. Direct-mapped: a slot another subresource took reads as not rendered, so a
// collision makes a draw wait rather than skip.
typedef struct {
    VkImage image;
    uint32_t level, layer;
    uint64_t last, prev;
    unsigned roles, prev_roles;   // scene passes that used this depth subresource in each frame
} TargetFrames;
static TargetFrames tfr[1024];

static TargetFrames *tfr_slot(const VkExecTarget *t) {
    uint64_t h = ((uint64_t)(uintptr_t)t->image ^ (uint64_t)t->level << 56 ^ (uint64_t)t->base_layer << 40) *
                 0x9E3779B97F4A7C15ull;
    return &tfr[h >> 54];
}

static int tfr_match(const TargetFrames *e, const VkExecTarget *t) {
    return e->image == t->image && e->level == t->level && e->layer == t->base_layer;
}

static void tfr_note(const VkExecTarget *t, uint64_t stamp, unsigned role) {
    TargetFrames *e = tfr_slot(t);
    if (!tfr_match(e, t)) *e = (TargetFrames){.image = t->image, .level = t->level, .layer = t->base_layer};
    if (e->last != stamp) {
        e->prev = e->last, e->last = stamp;
        e->prev_roles = e->roles;
        e->roles = 0;
    }
    e->roles |= role;
}

static int tfr_recent(const VkExecTarget *t, uint64_t stamp) {
    const TargetFrames *e = tfr_slot(t);
    return stamp > 1 && tfr_match(e, t) && (e->last == stamp ? e->prev : e->last) == stamp - 1;
}

// The scene lighting and half-resolution effects can acquire fresh colour images each frame
// while retaining their depth image. Record the pass on that depth subresource so a new
// colour image can reuse the previous frame's classification.
static unsigned scene_pass_role(const VkExec *ex) {
    const VkExecTarget *z = &ex->depth;
    if (!z->image || z->format != VK_FORMAT_D32_SFLOAT || z->level || z->base_layer || z->layers != 1 ||
        z->samples != 1 || (ex->pass_ncolor != 1 && ex->pass_ncolor != 2)) return 0;
    VkFormat format = ex->pass_ncolor == 1 ? VK_FORMAT_B10G11R11_UFLOAT_PACK32 : VK_FORMAT_R16G16B16A16_SFLOAT;
    for (unsigned i = 0; i < ex->pass_ncolor; i++) {
        const VkExecTarget *c = &ex->color[i];
        if (!c->image || c->format != format || c->width != z->width || c->height != z->height ||
            c->level || c->base_layer || c->layers != 1 || c->samples != 1 || c->is3d || c->blocks) return 0;
    }
    unsigned w, h;
    wide_picture(&w, &h);
    // Scene targets follow the picture aspect, including ultrawide and rounded render sizes.
    // Small inventory captures, cube faces and square shadow maps can also share a depth buffer.
    int64_t aspect_error = (int64_t)z->width * h - (int64_t)z->height * w;
    if (z->width < w / 8 || z->width > w || llabs(aspect_error) > (int64_t)8 * (w + h)) return 0;
    return ex->pass_ncolor == 1 ? 1u : 2u;
}

// No frame before the current one rendered the target (a slot another subresource took reads the same).
static int tfr_first(const VkExecTarget *t, uint64_t stamp) {
    const TargetFrames *e = tfr_slot(t);
    return !tfr_match(e, t) || !(e->last == stamp ? e->prev : e->last);
}

// With any, a recent depth target alone does not make the pass per-frame when a colour target is rendered for the
// first time: the game's item icon capture (an inventory icon) draws into a new colour target with the
// per-frame depth buffer, once. Its draws skipped there stay missing from the icon until the item is captured again.
int vk_exec_targets_recent(const VkExec *ex, int any, int effects) {
    uint64_t stamp = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    int n = 0, color = 0, depth = 0, color_first = 0;
    for (int i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if (!t->image) continue;
        n++;
        if (tfr_recent(t, stamp)) *(i < 8 ? &color : &depth) = 1;
        else if (!any) return 0;
        else if (i < 8 && tfr_first(t, stamp)) color_first = 1;
    }
    if (!any) return n > 0;
    if (color || (depth && !color_first)) return 1;
    if (!depth || !effects) return 0;
    const TargetFrames *e = tfr_slot(&ex->depth);
    unsigned previous_roles = e->last == stamp ? e->prev_roles : e->roles;
    return (previous_roles & scene_pass_role(ex)) != 0;
}

// One line naming each target of the current pass: format, extent, layer and the two frames it was last rendered in
// (prepare_draw's pipeline wait log).
void vk_exec_targets_describe(const VkExec *ex, char *buf, size_t cap) {
    size_t n = 0;
    buf[0] = 0;
    for (int i = 0; i < 9 && n < cap; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if (!t->image) continue;
        const TargetFrames *e = tfr_slot(t);
        int m = tfr_match(e, t);
        int w = snprintf(buf + n, cap - n, "%s%c%d fmt %d %ux%u L%u seen %llu/%llu", n ? "; " : "", i < 8 ? 'c' : 'd',
                         i < 8 ? i : 0, (int)t->format, t->width, t->height, t->base_layer,
                         (unsigned long long)(m ? e->last : 0), (unsigned long long)(m ? e->prev : 0));
        if (w < 0) break;
        n += (size_t)w;
    }
    if (!n) snprintf(buf, cap, "no targets");
}

void vk_exec_targets_note(const VkExec *ex) {
    uint64_t stamp = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    for (int i = 0; i < 8; i++)
        if (ex->color[i].image) tfr_note(&ex->color[i], stamp, 0);
    if (ex->depth.image) tfr_note(&ex->depth, stamp, scene_pass_role(ex));
}

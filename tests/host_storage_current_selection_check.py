"""Exercise original cold texture slot publication and retained host targets."""
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
    depth, end = 1, match.end()
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


PREFIX = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include "resource_core_registry.h"

enum { NVN_TEXTURE_TARGET_2D = 1, NVN_TEXTURE_TARGET_3D = 3, TEX_PRIVATE_STORAGE_OUTPUT = 2 };
typedef struct { unsigned swizzle_r, swizzle_g, swizzle_b, swizzle_a; int target, width, height, depth, levels, samples, format, linear; } nvn_texdesc;
typedef struct { uint64_t addr; nvn_texdesc d; } TexKey;
typedef struct {
    unsigned set_layers, set_dsmode, set_target, set_format, set_levels;
    unsigned target, format, base_level, min_layer, nlayers, pad1;
} view_t;
typedef struct {
    VkImage image;
    VkImageType type;
    VkImageViewType view_type;
    VkFormat format;
    VkImageUsageFlags usage;
    VkImageAspectFlags aspect;
    uint64_t addr;
    uint32_t width, height, depth, levels, layers, samples, scale;
    int skip, bound, block_views, mirrored, fb_learned, preserve_pool_bytes, gone;
} VkTexImage;
typedef struct { uint64_t serial, address; nvn_texdesc d; struct { uint32_t layers; uint64_t size; } L; } tex_t;
typedef struct { RcInterval span; int current; } Backing;
typedef struct Tex {
    TexKey key;
    tex_t t;
    VkTexImage pub;
    struct { VkFormat ideal, vk; int expand_rgb, astc_bw; } f;
    uint64_t content_revision, sample_general_revision;
    void *alloc, *material, *bound_to, *pending_write;
    Backing *image_host_backing;
    const RcInterval *image_private_mapping;
    uint32_t image_device_count, image_private_mapping_count;
    int pin_owned, pin_detached, released, retiring, superseded, overwritten;
} Tex;
typedef struct { tex_t descriptor; struct { uint64_t id; } handle; } TextureRegistration;
typedef struct {
    tex_t descriptor;
    TexKey key;
    TextureRegistration *registration;
    Tex *ce, *owned;
    view_t cview;
    uint32_t cgen, cepoch;
    int used, finalized, cheld, chas_view, sample_owned;
} TexSlot;
typedef struct { TexSlot *slot; Tex *entry; int associated; } TextureSlotSelection;
typedef struct {
    TexKey key, registration_key;
    RcRegistryPosition read;
    VkImage image;
    VkImageSubresourceRange word_alias_plane;
    uint64_t serial, registration, registration_serial, size, revision, host_layout;
    int native, bound_relocated, has_view;
    void *bound_plan;
    void *host_pin;
    view_t view;
    uint32_t mapping_count;
    RcInterval mapping[1];
} VkTextureStorageTarget;
typedef struct { int returned_selected, effective; } StorageTrace;

static const int by_key;
static TexSlot tslots[2];
static TextureRegistration registrations[2], replacement;
static Tex entries[2], other;
static Tex *owners[2];
static Backing backings[2];
static uint64_t sinfo[2];
static uint32_t agen[65536];
static int ntslots = 2, pool_ctx, in_lookup, resolve_held, resolves, bad_backing, wrong_key, wrong_serial, wrong_owner, unrelated_resolve_bump;
static const RcRegistryPosition original_read = {14604,42708};
static int re_registered, unrelated_after_install;

static __attribute__((unused)) int nvn_gpu_only(uint64_t address) { return address == 0; }
static __attribute__((unused)) void make_key(const tex_t *descriptor, TexKey *key) {
    *key = (TexKey){.addr = descriptor->address, .d = descriptor->d};
}
static int storage_target_identity_current(const Tex *entry);
static int __attribute__((unused)) storage_word_alias_initialized(Tex *entry,VkImageSubresourceRange plane,
    RcRegistryPosition read) {
    (void)entry; (void)plane; (void)read; assert(0); return 0;
}

static Tex *texture_registration_source(const TexSlot *slot) { return slot->owned; }
static int slot_absence_valid(const TexSlot *slot, uint32_t gen) { (void)slot; (void)gen; return 0; }
static int __attribute__((unused)) from_storage(const Tex *e) { return e->pub.mirrored; }
static int nvn_tex_is_linear(const nvn_texdesc *d) { return d->linear; }
static int nvn_fmt_bpb(int fmt, int *w, int *h) { *w = *h = 1; return fmt != 0; }
static int format_compressed(VkFormat f) { return f == VK_FORMAT_BC1_RGB_UNORM_BLOCK; }
static int mem_backing_current(const Backing *b) { return b && b->current; }
static const RcInterval *mem_backing_intervals(const Backing *b, uint32_t *count) {
    assert(b);
    *count = 1;
    return &b->span;
}
static Tex *nvn_hm_get(const int *table, const TexKey *key) {
    assert(table == &by_key);
    for (int i = 0; i < 2; ++i)
        if (owners[i] && !memcmp(key, &entries[i].key, sizeof *key)) return owners[i];
    return NULL;
}
static void slot_result_view(TexSlot *slot, const Tex *e, view_t *view, int *has_view) {
    assert(slot && e);
    *view = (view_t){.set_target = 1, .target = e->t.d.target, .set_levels = 1};
    *has_view = 1;
}
static Tex *slot_resolve(TexSlot *slot, view_t *view, int *has_view, int private_output
#if HAS_INSTALLATION
    , uint32_t *installed_generation
#endif
) {
    assert(in_lookup && (private_output == 0 || private_output == TEX_PRIVATE_STORAGE_OUTPUT));
    int i = (int)(slot - tslots);
    Tex *e = &entries[i];
    resolves++;
    if (!owners[i]) {
        e->key = slot->key;
        if (wrong_key) e->key.d.swizzle_r++;
        e->t = slot->descriptor;
        if (wrong_serial) e->t.serial++;
        e->f.vk = e->f.ideal = (VkFormat)e->key.d.format;
        e->pub = (VkTexImage){.image = (VkImage)(uintptr_t)(87 + i), .type = VK_IMAGE_TYPE_3D,
            .view_type = VK_IMAGE_VIEW_TYPE_3D, .format = e->f.vk, .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            .addr = e->key.addr, .width = 64, .height = 36, .depth = (uint32_t)e->t.d.depth,
            .levels = 1, .layers = 1, .samples = 1, .scale = 1};
        e->alloc = e;
        e->sample_general_revision = 100 + i;
        backings[i] = (Backing){.span = {e->key.addr, 401 + i, {103, 0x100000 * (1 + i), e->t.L.size}}, .current = !bad_backing};
        e->image_host_backing = &backings[i];
        owners[i] = wrong_owner ? &other : e;
        // entry_of_mode installs this same native owner and bumps its allocation-address generation.
        agen[sinfo[i] & 0xffff]++;
#if HAS_INSTALLATION
        *installed_generation = agen[sinfo[i] & 0xffff];
#endif
        if (unrelated_after_install) agen[sinfo[i] & 0xffff]++;
        if (re_registered) {
            replacement = *slot->registration;
            replacement.handle.id++;
            slot->registration = &replacement;
        }
    } else if (unrelated_resolve_bump) agen[sinfo[i] & 0xffff]++;
    slot_result_view(slot, e, view, has_view);
    return e;
}
static const VkTexImage *texture_slot_refresh_locked(TextureSlotSelection sel, view_t *view, int *has_view, void *s, void *f) {
    (void)sel; (void)view; (void)has_view; (void)s; (void)f; assert(0); return NULL;
}
static void texture_require_exact_locked(Tex *e) { (void)e; assert(0); }
static void storage_trace_effective(StorageTrace *t, uint64_t reg, const TexKey *k, Tex *e, const view_t *v, int hv) {
    (void)t; (void)reg; (void)k; (void)e; (void)v; (void)hv; assert(0);
}
static int image_alias_placement_current(const Tex *e) { (void)e; assert(0); return 0; }
static int image_alias_guest_mapping_current(uint64_t a, uint64_t s, const RcInterval *m, uint32_t n) {
    (void)a; (void)s; (void)m; (void)n; assert(0); return 0;
}
static Tex *nvn_texture_pin_attachment(void *pin) { (void)pin; assert(0); return NULL; }
static TexSlot *storage_target_registration_locked(const VkTextureStorageTarget *t) {
    TexSlot *s = &tslots[t->native];
    return !s->finalized && s->registration->handle.id == t->registration &&
        !rc_registry_position_compare(original_read, t->read) &&
        !memcmp(&s->key, &t->registration_key, sizeof s->key) ? s : NULL;
}
'''

CAPTURE_HEAD = r'''
static int capture(int native, VkTextureStorageTarget *target) {
    TexSlot *slot = &tslots[native];
    const TexKey registration_key = slot->key;
    const uint64_t registration_serial __attribute__((unused)) = slot->descriptor.serial;
    const uint64_t registration = slot->registration->handle.id;
    const int exact = 1, gpu_only = 0, host_private = 1, trace_selected = 0;
    int result, has_view[1];
    view_t view[1];
    const char *why;
    StorageTrace trace = {0};
'''

CAPTURE_TAIL = r'''
    *target = (VkTextureStorageTarget){.key = entry->key, .registration_key = registration_key,
        .read = original_read, .image = entry->pub.image, .serial = entry->t.serial, .registration = registration,
        .registration_serial = slot->descriptor.serial, .size = entry->t.L.size, .revision = entry->content_revision,
        .host_layout = entry->sample_general_revision, .native = native, .has_view = *has_view, .view = *view, .mapping_count = count};
    target->mapping[0] = mapping[0];
    result = 1;
done:
    (void)why;
    return result;
}
'''

CASES = r'''
static void init(void) {
    memset(tslots, 0, sizeof tslots);
    memset(entries, 0, sizeof entries);
    memset(owners, 0, sizeof owners);
    re_registered = unrelated_after_install = 0;
    resolves = bad_backing = wrong_key = wrong_serial = wrong_owner = unrelated_resolve_bump = 0;
    for (int i = 0; i < 2; i++) {
        uint64_t address = i ? UINT64_C(0x27158b000) : UINT64_C(0x27149a000);
        sinfo[i] = agen_idx(address);
        agen[sinfo[i]] = 9011 + i;
        nvn_texdesc d = {.width = 64, .height = 36, .depth = i ? 48 : 96, .levels = 1, .samples = 1,
            .target = NVN_TEXTURE_TARGET_3D, .format = i ? 16 : 122};
        const tex_t descriptor = {.d = d, .serial = 4283 + i, .address = address,
            .L = {.layers = 1, .size = i ? 0x3c000 : 0xf0000}};
        registrations[i] = (TextureRegistration){.descriptor = descriptor, .handle = {37098 + i}};
        tslots[i] = (TexSlot){.used = 1, .registration = &registrations[i],
            .key = {.addr = address, .d = d}, .descriptor = descriptor};
    }
    assert(sinfo[0] != sinfo[1]);
}
static void check_original_sample_pair(void) {
    init();
    for (int i = 0; i < 2; ++i) {
        nvn_texdesc *shape = &tslots[i].descriptor.d;
        shape->target = NVN_TEXTURE_TARGET_2D;
        shape->depth = 1;
        tslots[i].key.d = *shape;
        registrations[i].descriptor = tslots[i].descriptor;
    }
    const TextureSlotSelection b1 = texture_slot_select_locked(0, 0);
    const TextureSlotSelection b2 = texture_slot_select_locked(1, 0);
    assert(b1.entry == &entries[0] && b2.entry == &entries[1] && b1.entry->pub.image != b2.entry->pub.image);
    assert(texture_binding_entry(b1.slot, 0) == b1.entry && texture_binding_entry(b2.slot, 1) == b2.entry);
    assert(resolves == 2);
    assert(texture_slot_select_locked(0, 0).entry == b1.entry &&
           texture_slot_select_locked(1, 0).entry == b2.entry && resolves == 2);

    init();
    unrelated_after_install = 1;
    assert(texture_slot_select_locked(0, 0).entry == &entries[0]);
    assert(!texture_binding_entry(&tslots[0], 0));

    init();
    re_registered = 1;
    assert(texture_slot_select_locked(0, 0).entry == &entries[0]);
    assert(!texture_binding_entry(&tslots[0], 0));

    init();
    bad_backing = 1;
    assert(texture_slot_select_locked(0, 0).entry == &entries[0]);
    assert(!texture_binding_entry(&tslots[0], 0));
    init();
    wrong_serial = 1;
    const TextureSlotSelection private_output = texture_slot_select_locked(0, TEX_PRIVATE_STORAGE_OUTPUT);
    assert(private_output.entry == &entries[0] && texture_binding_entry(private_output.slot, 0) == private_output.entry);
    puts("ordinary cold samples: one exact selection per original slot; later generation, registration and backing changes rejected");
}
int main(void) {
    check_original_sample_pair();
    init();
    VkTextureStorageTarget a, b;
    assert(capture(0, &a) == 1 && capture(1, &b) == 1);
    assert(resolves == 2 && a.image != b.image && a.read.ordinal == b.read.ordinal);
    assert(storage_target_current_locked(&a, 0) == &entries[0] && storage_target_current_locked(&b, 0) == &entries[1]);
    backings[0].span.mapping++;
    assert(!storage_target_current_locked(&a, 0) && storage_target_current_locked(&b, 0));
    backings[0].span.mapping--;
    tslots[0].registration->handle.id++;
    assert(!storage_target_current_locked(&a, 0));
    tslots[0].registration->handle.id--;
    agen[sinfo[0]]++; // later unrelated cache invalidation is not restamped at bind
    assert(!storage_target_current_locked(&a, 0) && storage_target_current_locked(&b, 0));
    assert(resolves == 2);

    init();
    assert(capture(0, &a) == 1);
    assert(capture(0, &a) == 1 && resolves == 1);
    agen[sinfo[0]]++; // choose normally again with the same live exact owner
    unrelated_resolve_bump = 1;
    assert(!capture(0, &a) && tslots[0].ce == &entries[0] && tslots[0].cgen != agen[sinfo[0]]);
    unrelated_resolve_bump = 0;
    assert(capture(0, &a) == 1 && resolves == 3);

    init();
    bad_backing = 1;
    assert(!capture(0, &a) && tslots[0].cgen != agen[sinfo[0]]);
    init();
    wrong_key = 1; // swizzle alias can be a candidate, but it cannot borrow the fresh cache generation
    assert(!capture(0, &a) && tslots[0].cgen != agen[sinfo[0]]);
    init();
    wrong_serial = 1;
    assert(!capture(0, &a) && tslots[0].cgen != agen[sinfo[0]]);
    init();
    wrong_owner = 1;
    assert(!capture(0, &a) && tslots[0].cgen != agen[sinfo[0]]);
    puts("host storage placement: two exact cold world volumes, same read, stale registration/backing/cache, alias and owner controls passed");
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument("--cc", default=os.environ.get("CC", "clang"))
    parser.add_argument("--expect-failure", choices=["sample", "cold", "warm"])
    args = parser.parse_args()
    source = args.source.read_text()
    body = function(source, "vk_texture_effect_storage_capture")
    # Use the real selection and post-shape receipt gate. Earlier unpublished class/refusal routing is independent.
    selections = body.index("    const TextureSlotSelection selected")
    generation = body.rfind("    const uint32_t selection_generation", 0, selections)
    selections = body[generation if generation >= 0 else selections:body.index("    Tex *entry = selected.entry;", selections) + len("    Tex *entry = selected.entry;")]
    placement = body[body.rindex("    result = VK_TEXTURE_STORAGE_CAPTURE_INVALIDATED;"):body.index('    why = "storage-target-capacity";')]
    placement = placement.replace("VK_TEXTURE_STORAGE_CAPTURE_INVALIDATED", "0")
    shape = r'''
    VkImageSubresourceRange plane = {0};
    if (!storage_host_shape(entry, view, *has_view, &plane)) return 0;
    assert(plane.baseMipLevel == 0 && plane.levelCount == 1 && plane.baseArrayLayer == 0 && plane.layerCount == 1);
'''
    names = ("agen_idx", "same_but_swizzle", "texture_binding_entry", "texture_slot_select_locked",
             "texture_slot_selected_view", "texture_storage_select_locked", "storage_volume_shape", "storage_volume_mip",
             "storage_target_volume_view", "storage_host_shape", "storage_target_identity_current", "storage_host_mapping",
             "storage_target_current_locked")
    with tempfile.TemporaryDirectory() as folder:
        c = Path(folder) / "check.c"
        exe = c.with_suffix(".exe")
        installation = int("installed == agen[sinfo[id] & 0xffff]" in source)
        c.write_text(f"#define HAS_INSTALLATION {installation}\
" + PREFIX + "\n".join(function(source, name) for name in names) + CAPTURE_HEAD + selections +
                     shape + placement + CAPTURE_TAIL + CASES)
        subprocess.run([args.cc, "-std=gnu11", "-Wall", "-Wextra", "-Werror", "-Wno-sign-compare",
                        "-fsanitize=undefined", "-fsanitize-trap=all", "-I", str(args.headers), str(c), "-o", str(exe)], check=True)
        run = subprocess.run([str(exe)], text=True, capture_output=True)
        if args.expect_failure:
            needle = {"sample": "texture_binding_entry(b1.slot, 0) == b1.entry", "cold": "capture(0, &a) == 1", "warm": "!capture(0, &a) && tslots[0].ce"}[args.expect_failure]
            assert run.returncode != 0 and needle in run.stderr, run
            print(f"expected {args.expect_failure} placement regression reproduced")
        else:
            if run.returncode: print(run.stderr)
            run.check_returncode()
            print(run.stdout.strip())


if __name__ == "__main__":
    main()

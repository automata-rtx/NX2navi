"""Host check of an exact ordered 2D sampled-image geometry receipt."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

TESTS = Path(__file__).resolve().parent


PRELUDE = r"""
#define VK_NO_PROTOTYPES
#include <assert.h>
#include <stdio.h>
#include "nvn_api.h"
#include "texture_resource.h"
#include "tex.h"

typedef struct { uint64_t address; int format, levels; } TexKey;
typedef struct {
    VkTexImage pub;
    TexKey key;
    tex_t t;
    TexVkFormat f;
    VmaAllocation alloc, bound_to;
    RcRegistryPosition sample_general_inline;
    uint64_t sample_general_upload, sample_general_revision;
    uint8_t host_storage_init;
    int image_device_count, image_private_mapping_count;
    int released, retiring, superseded, overwritten, pin_detached;
    void *pending_write, *image_host_backing;
    RcInterval origin;
} Tex;
typedef struct {
    struct { uint64_t id; } handle;
    uint64_t publication;
    tex_t descriptor;
    view_t view;
    int has_view;
} TextureRegistration;
typedef struct {
    int used, finalized, logical_id, chas_view;
    uint64_t pool_owner;
    TextureRegistration *registration;
    view_t cview;
    Tex *entry;
} TexSlot;
struct VkTextureRegistrationScope { uint64_t publication; };

enum { POOL = 13, FIRST = 23, SECOND = 29 };
static TexSlot tslots[2];
static int ntslots = 2;
static Tex depth, color;
static TextureRegistration registrations[2];
static VkTextureRegistrationScope captured_scope = {50}, other_scope = {50};
static VkTextureRegistrationScope *active_scope = &captured_scope;
static uint64_t active_pool = POOL;
static RcRegistryPosition resource_read = {901, 30};
static RcInterval actual_mapping[2];
static uint64_t submitted_upload, issued_scene, handed_scene;
static int stop, fail_recording, unavailable_layout, backend_map_reads, exact_host_mapping;
static unsigned cutoff_kind;

volatile int32_t vk_lost;
VkTextureRegistrationScope *vk_exec_texture_registration_scope(void) { return active_scope; }
uint64_t vk_exec_texture_pool(void) { return active_pool; }
uint64_t vk_texture_registration_scope_publication(const VkTextureRegistrationScope *scope) {
    return scope ? scope->publication : 0;
}
RcRegistryPosition vk_exec_resource_position(void) { return resource_read; }
uint64_t vk_sync_issued(void) { return issued_scene; }
uint64_t vk_sync_handed_fence(void) { return handed_scene; }
void vk_upload_progress(uint64_t *submitted, uint64_t *completed) {
    assert(submitted && !completed);
    *submitted = submitted_upload;
}
int vk_sync_stopping(void) { return stop; }
int vk_exec_recording_failed(void) { return fail_recording; }
static int texture_registration_resolve(int logical, const TextureRegistration **record, int *native) {
    *record = NULL;
    *native = logical == FIRST ? 0 : logical == SECOND ? 1 : -1;
    return *native < 0 ? VK_TEXTURE_REGISTRATION_ABSENT : (int)cutoff_kind;
}
static Tex *texture_binding_entry(const TexSlot *slot, int native) {
    assert(slot == &tslots[native]);
    return slot->entry;
}
static const Tex *texture_registration_source(const TexSlot *slot) {
    (void)slot;
    return NULL; // The stock compatibility candidate is resolved through the cached resident entry.
}
static void slot_result_view(const TexSlot *slot, const Tex *entry, view_t *view, int *has_view) {
    (void)slot; (void)entry; (void)view; (void)has_view;
    abort();
}
static void make_key(const tex_t *descriptor, TexKey *key) {
    *key = (TexKey){(uintptr_t)descriptor->mem, descriptor->d.format, descriptor->d.levels};
}
static const RcInterval *image_backing_intervals(const Tex *e, uint32_t *count) {
    *count = (uint32_t)e->image_private_mapping_count;
    return *count ? &e->origin : NULL;
}
static int image_alias_guest_mapping_current(uint64_t address, uint64_t bytes, const RcInterval *expected, uint32_t count) {
    const int index = address == depth.pub.addr ? 0 : 1;
    const RcInterval *current = &actual_mapping[index];
    ++backend_map_reads;
    assert(bytes && count == 1 && expected);
    return address == current->address && bytes == current->range.size &&
           current->mapping == expected->mapping && rc_range_equal(current->range, expected->range);
}
#define mem_backing_current(value) ((value) && exact_host_mapping)
static uint32_t umax(uint32_t a, uint32_t b) { return a > b ? a : b; }
void rt_fatal(const char *format, ...) { (void)format; abort(); }
"""


CASES = r"""
static void install(Tex *e, unsigned i, int target, int guest_format, VkFormat format, VkImageAspectFlags aspect,
                    uint32_t width, uint32_t height, uint32_t mips) {
    const uint64_t address = UINT64_C(0x8000000000) + i * 0x40000;
    TextureRegistration *r = &registrations[i];
    *r = (TextureRegistration){.handle.id = 100 + i, .publication = 49,
        .descriptor = {.mem = (void *)(uintptr_t)address, .serial = 200 + i,
            .d = {.target = target, .format = guest_format, .width = width, .height = height, .levels = mips},
            .L = {.layers = 1, .size = 0x10000}}};
    actual_mapping[i] = (RcInterval){.address = address, .mapping = 500 + i, .range = {10, i * 0x40000, 0x10000}};
    *e = (Tex){.pub = {.image = (VkImage)(uintptr_t)(1000 + i), .addr = address,
            .type = VK_IMAGE_TYPE_2D, .view_type = VK_IMAGE_VIEW_TYPE_2D, .width = width, .height = height,
            .depth = 1, .layers = 1, .levels = mips, .samples = 1, .scale = 1, .format = format,
            .aspect = aspect, .usage = VK_IMAGE_USAGE_SAMPLED_BIT},
        .t = r->descriptor, .f = {.vk = format, .ideal = format, .depth = aspect == VK_IMAGE_ASPECT_DEPTH_BIT},
        .alloc = (VmaAllocation)(uintptr_t)(2000 + i), .image_private_mapping_count = 1, .origin = actual_mapping[i]};
    make_key(&r->descriptor, &e->key);
    tslots[i] = (TexSlot){.used = 1, .pool_owner = POOL, .logical_id = i ? SECOND : FIRST, .registration = r, .entry = e};
}

static int current(int logical, const VkTextureSample *sample, const VkTexImage *exact,
                   const VkTextureSampleGeometry *receipt) {
    const RcInterval *ranges = NULL;
    uint32_t count = 51;
    const int result = vk_texture_sample_geometry_current_locked(logical, sample, exact, receipt, &ranges, &count);
    if (result) {
        assert(ranges == &receipt->mapping && count == 1);
    } else {
        assert(!ranges && count == 0);
    }
    return result;
}

int main(void) {
    install(&depth, 0, NVN_TEXTURE_TARGET_2D, NVN_FORMAT_DEPTH32F, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_ASPECT_DEPTH_BIT, 128, 128, 1);
    install(&color, 1, NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_ASPECT_COLOR_BIT, 320, 240, 3);
    VkTextureSample sample = {0}, other = {0};
    VkTextureSampleGeometry receipt = {0}, secondary = {0};
    assert(vk_texture_sample_geometry_report_locked(FIRST, &sample, &receipt) ==
           VK_TEXTURE_SAMPLE_GEOMETRY_LAYOUT_UNOBSERVED);
    assert(!strcmp(vk_texture_sample_geometry_status_name(VK_TEXTURE_SAMPLE_GEOMETRY_LAYOUT_UNOBSERVED),
                   "layout-init-unobserved"));
    sample_general_record(&depth, (RcRegistryPosition){901, 30}, 0);
    assert(!vk_texture_sample_geometry_locked(FIRST, &sample, &receipt)); // Same operation has not ordered the transition.
    sample_general_record(&depth, (RcRegistryPosition){900, 44}, 0);
    assert(!vk_texture_sample_geometry_locked(FIRST, &sample, &receipt)); // A recorded earlier scene is insufficient.
    issued_scene = 900;
    assert(vk_texture_sample_geometry_report_locked(FIRST, &sample, &receipt) ==
           VK_TEXTURE_SAMPLE_GEOMETRY_LAYOUT_UNORDERED); // Issued can still be held by batching.
    assert(!strcmp(vk_texture_sample_geometry_status_name(VK_TEXTURE_SAMPLE_GEOMETRY_LAYOUT_UNORDERED),
                   "layout-init-unordered"));
    handed_scene = 900;
    assert(vk_texture_sample_geometry_locked(FIRST, &sample, &receipt));
    assert(sample.image == &depth.pub && !sample.has_view && !sample.projected && !sample.canonical &&
           sample.registration == registrations[0].handle.id && sample.native_slot == 0);
    assert(receipt.extent.width == 128 && receipt.extent.height == 128 && receipt.planes.baseMipLevel == 0 &&
           receipt.planes.levelCount == 1 && receipt.planes.baseArrayLayer == 0 && receipt.planes.layerCount == 1);
    assert(receipt.scope == &captured_scope && receipt.read.scene == 901 && receipt.read.ordinal == 30 &&
           receipt.pool == POOL && receipt.material == depth.t.serial && receipt.planes.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT);
    VkTexImage exact = *sample.image;
    sample.image = NULL; // Draw custody retains the native image separately.
    assert(current(FIRST, &sample, &exact, &receipt));
    assert(!current(SECOND, &sample, &exact, &receipt));

    sample_general_record(&color, (RcRegistryPosition){0}, 11);
    submitted_upload = 10;
    assert(vk_texture_sample_geometry_report_locked(SECOND, &other, &secondary) ==
           VK_TEXTURE_SAMPLE_GEOMETRY_LAYOUT_UNORDERED);
    submitted_upload = 11;
    assert(vk_texture_sample_geometry_locked(SECOND, &other, &secondary));
    assert(secondary.extent.width == 320 && secondary.extent.height == 240 && secondary.planes.levelCount == 3 &&
           secondary.image != receipt.image && secondary.mapping.mapping != receipt.mapping.mapping);
    assert(current(FIRST, &sample, &exact, &receipt));

    registrations[1].has_view = tslots[1].chas_view = 1;
    registrations[1].view = tslots[1].cview = (view_t){.set_levels = 1, .base_level = 1, .set_layers = 1};
    assert(vk_texture_sample_geometry_locked(SECOND, &other, &secondary)); // Zero counts mean remaining layers and mips.
    assert(secondary.extent.width == 160 && secondary.extent.height == 120 &&
           secondary.planes.levelCount == 2 && secondary.planes.baseMipLevel == 1 &&
           secondary.planes.layerCount == 1 && secondary.planes.baseArrayLayer == 0);
    ++tslots[1].cview.base_level;
    assert(!vk_texture_sample_geometry_locked(SECOND, &other, &secondary));
    tslots[1].cview = registrations[1].view;
    tslots[1].cview.set_dsmode = 1;
    assert(!vk_texture_sample_geometry_locked(SECOND, &other, &secondary));
    tslots[1].cview = registrations[1].view;
    tslots[1].cview.set_format = 1;
    tslots[1].cview.format = NVN_FORMAT_RGBA8_SRGB;
    assert(!vk_texture_sample_geometry_locked(SECOND, &other, &secondary));

    active_scope = &other_scope;
    assert(!current(FIRST, &sample, &exact, &receipt));
    active_scope = &captured_scope;
    ++active_pool;
    assert(!current(FIRST, &sample, &exact, &receipt));
    active_pool = POOL;
    resource_read.ordinal += 2;
    assert(!current(FIRST, &sample, &exact, &receipt));
    resource_read.ordinal -= 2;
    exact.width *= 2;
    assert(!current(FIRST, &sample, &exact, &receipt));
    exact.width /= 2;
    depth.pub.fb_learned = 1;
    assert(!current(FIRST, &sample, &exact, &receipt));
    depth.pub.fb_learned = 0;
    depth.pub.scale = 2;
    assert(!current(FIRST, &sample, &exact, &receipt));
    assert(vk_texture_sample_geometry_report_locked(FIRST, &other, &secondary) == VK_TEXTURE_SAMPLE_GEOMETRY_SHAPE);
    depth.pub.scale = 1;
    depth.pub.skip = 1;
    assert(!current(FIRST, &sample, &exact, &receipt));
    depth.pub.skip = 0;
    depth.bound_to = (VmaAllocation)(uintptr_t)7;
    assert(!current(FIRST, &sample, &exact, &receipt));
    assert(vk_texture_sample_geometry_report_locked(FIRST, &other, &secondary) == VK_TEXTURE_SAMPLE_GEOMETRY_PROVIDER);
    depth.bound_to = NULL;
    stop = 1;
    assert(!current(FIRST, &sample, &exact, &receipt));
    stop = 0;
    cutoff_kind = (unsigned)VK_TEXTURE_REGISTRATION_GONE;
    assert(!current(FIRST, &sample, &exact, &receipt));
    cutoff_kind = VK_TEXTURE_REGISTRATION_CURRENT;
    ++registrations[0].handle.id;
    assert(!current(FIRST, &sample, &exact, &receipt));
    --registrations[0].handle.id;
    ++actual_mapping[0].mapping;
    assert(!current(FIRST, &sample, &exact, &receipt));
    assert(vk_texture_sample_geometry_report_locked(FIRST, &other, &secondary) == VK_TEXTURE_SAMPLE_GEOMETRY_MAPPING);
    depth.origin = actual_mapping[0];
    assert(!current(FIRST, &sample, &exact, &receipt)); // A valid new mapping doesn't validate the old receipt.
    assert(vk_texture_sample_geometry_locked(FIRST, &sample, &receipt));
    assert(current(FIRST, &sample, &exact, &receipt));
    sample_general_record(&depth, (RcRegistryPosition){901, 8}, 0);
    assert(!current(FIRST, &sample, &exact, &receipt)); // Reinitializing the same VkImage invalidates its old token.
    assert(vk_texture_sample_geometry_locked(FIRST, &sample, &receipt));
    assert(current(FIRST, &sample, &exact, &receipt));
    assert(backend_map_reads > 0 && !unavailable_layout);
    puts("Exact ordered geometry: two independent views, LOD-0 extent, initialization and stale identity controls passed.");
    return 0;
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=TESTS.parent / "runtime")
    parser.add_argument("--base", type=Path, default=TESTS.parent / "runtime")
    args = parser.parse_args()
    base = args.base.resolve()
    sys.path.insert(0, str(base.parent / "tests"))
    from ordered_graphics_feedback_check import function

    source = (args.sources / "vk_texture.c").read_text(encoding="utf-8")
    assert "e->sample_general_revision = 0;" in function(source, "image_owner_replaced")
    initialize = function(source, "init_image")
    assert initialize.index("e->sample_general_revision = 0;") < initialize.index("vkCmdPipelineBarrier2")
    assert ".oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL" in initialize
    new = function(source, "init_new")
    assert new.count("sample_general_record(") == 2
    assert new.index("init_image(e, cb);") < new.index("sample_general_record(e, vk_exec_resource_position(), 0);")
    assert new.rindex("init_image(e, cb);") < new.index("vk_upload_recording_sequence(cb)") < new.index("vk_upload_end();")
    names = ("sample_general_record", "sample_general_current", "sample_geometry_view", "sample_geometry_mapping",
             "vk_texture_sample_geometry_report_locked", "vk_texture_sample_geometry_locked",
             "vk_texture_sample_geometry_status_name", "vk_texture_sample_geometry_current_locked")
    compiler = os.environ.get("CC") or shutil.which("clang")
    if not compiler:
        raise SystemExit("Clang is required (set CC)")
    cc = Path(shutil.which(compiler) or compiler).resolve()
    vma = Path(os.environ.get("VMA_INCLUDE", cc.parent.parent / "include/vma"))
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="sampled-geometry-") as directory:
        c = Path(directory) / "check.c"
        exe = c.with_suffix(".exe" if os.name == "nt" else "")
        c.write_text(PRELUDE + "static uint64_t sample_general_revision;\n" +
                     "\n".join(function(source, name) for name in names) + CASES, encoding="utf-8")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", f"-I{args.sources}", f"-I{base}", f"-I{vma}",
                        str(c), "-o", str(exe)], check=True, timeout=60, env=env)
        subprocess.run([str(exe)], check=True, timeout=20, env=env)


if __name__ == "__main__":
    main()

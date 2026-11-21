"""Check preserved native RAW placement and LOAD qualification through the actual replay helpers."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from present_source_check import ROOT, function, section


def make_fixture(runtime):
    texture = (runtime / "vk_texture.c").read_text()
    replay = (runtime / "vk_exec.c").read_text()
    start = function(texture, "start_at")
    assert start.index("e->pub.preserve_pool_bytes") < start.index("content_write_begin")
    init = function(texture, "init_image")
    assert init.index("vkCmdPipelineBarrier2") < init.index("e->pub.preserve_pool_bytes") < init.index("vkCmdClearDepthStencilImage")
    create = function(texture, "create_tex")
    assert create.index("bind_pool_memory") < create.index("VK_MEM_IMAGE_ALIAS_STALE") < create.index("if (!img && tex_mirrored")
    raw = function(texture, "image_alias_read_prepare_locked")
    assert raw.index("alias->baseline = *receiver->prepared_baseline") < raw.index("vk_mem_image_alias_baseline_capture")
    assert "alias->baseline ? RC_REGISTRY_OK" in raw
    guard = section(function(replay, "pass_begin"),
                    "    // This pass always LOADs", "    // A parallel group's targets")
    prepare = section(function(replay, "resource_attachment_prepare"),
                      "    // An exact LOAD/STORE", "    vk_texture_image_write_prepare")
    prepare = prepare.replace("binding->", "t->")
    return r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef uint64_t VkImage;
typedef unsigned VkImageAspectFlags;
typedef struct {unsigned aspectMask, baseMipLevel, levelCount, baseArrayLayer, layerCount;} VkImageSubresourceRange;
typedef int VkResult;
typedef int VkImageCreateInfo;
typedef struct {uint64_t size, alignment; unsigned memoryTypeBits;} VkMemoryRequirements;
typedef struct {int device;} VkCtx;
typedef struct {void *alloc; uint64_t avail;} VkMemSpan;
typedef struct {uint64_t address, bytes;} RcInterval;
typedef struct {uint64_t scene, ordinal;} RcRegistryPosition;
typedef struct {
    VkImage image;
    uint64_t addr;
    int preserve_pool_bytes, gpu_owned;
    unsigned format, aspect, scale, levels, layers;
} VkTexImage;
typedef struct {unsigned unused;} ImgKey;
typedef struct {
    ImgKey ik;
    VkTexImage pub;
    void *bound_to;
    uint64_t image_device_size;
    RcInterval *image_device_intervals, *image_private_mapping, *alias_full_demand;
    uint32_t image_device_count, image_private_mapping_count, alias_full_count;
    int released, retiring, superseded, overwritten;
} Tex;
typedef enum {
    VK_MEM_IMAGE_ALIAS_DIRECT, VK_MEM_IMAGE_ALIAS_PRIVATE_PRESERVE,
    VK_MEM_IMAGE_ALIAS_STALE, VK_MEM_IMAGE_ALIAS_FAILED, VK_MEM_IMAGE_ALIAS_OCCUPIED
} VkMemImageAliasPlacement;

enum { VK_SUCCESS = 0, VK_ERROR_OUT_OF_DEVICE_MEMORY = -2, VK_FORMAT_D32_SFLOAT = 126,
       VK_FORMAT_D16_UNORM = 124, VK_IMAGE_ASPECT_COLOR_BIT = 1, VK_IMAGE_ASPECT_DEPTH_BIT = 2,
       VK_IMAGE_ASPECT_STENCIL_BIT = 4 };
#define VK_NULL_HANDLE 0
static unsigned img_reqs, alias_fragment_owners, n_bind_skips;
static VkMemoryRequirements *cached;
static VkMemoryRequirements native_req = {16384, 4096, 12};
static const uint64_t addr = 0x250000, image_size = 16384, logical_size = 8192;
static unsigned creates, binds, advice, mapping_calls;
static int fits = 1, stale_mapping, drift, registry_current, producer_current, locked, refusals;
static int direct_stale;
static VkResult create_error;
static Tex *owner;
typedef struct {Tex *entry;} NativeTextureOutput;
static NativeTextureOutput projection;
static VkMemImageAliasPlacement early, final;
static void *nvn_hm_get(unsigned *table, const void *key) {
    assert(key);
    if (table == &img_reqs) return cached;
    assert(table == &alias_fragment_owners && locked);
    projection.entry = owner;
    return owner && *(const VkImage *)key == owner->pub.image ? &projection : NULL;
}
static void nvn_hm_put(unsigned *table, const ImgKey *key, VkMemoryRequirements *write) {
    assert(table == &img_reqs && key && !cached);
    cached = write;
}
static void rt_fatal(const char *error) { fprintf(stderr, "%s\n", error); abort(); }
static VkResult vkCreateImage(int device, const VkImageCreateInfo *ci, void *callbacks, VkImage *out) {
    assert(device == 1 && ci && !callbacks && !*out);
    creates++;
    if (!create_error) *out = 17;
    return create_error;
}
static void vkGetImageMemoryRequirements(int device, VkImage image, VkMemoryRequirements *out) {
    assert(device == 1 && image == 17);
    *out = native_req;
    if (drift) out->memoryTypeBits ^= 4;
}
static int pool_fits(VkCtx *ctx, const Tex *e, const VkMemoryRequirements *req, uint64_t guest_size, VkMemSpan *s) {
    assert(ctx->device == 1 && e->pub.addr == addr && req->size == image_size && guest_size == logical_size);
    *s = (VkMemSpan){&owner, image_size};
    return fits;
}
static unsigned vk_mem_device_intervals(const VkMemSpan *s, uint64_t a, uint64_t n, RcInterval **out) {
    assert(s->alloc == &owner && n <= s->avail && a == addr);
    mapping_calls++;
    if (stale_mapping) return 0;
    *out = malloc(sizeof **out);
    assert(*out);
    **out = (RcInterval){a, n};
    return 1;
}
static VkMemImageAliasPlacement vk_mem_image_alias_preplacement(uint64_t a, const VkMemoryRequirements *req,
    const VkMemSpan *s, const RcInterval *ranges, uint32_t count) {
    assert(a == addr && req->size == image_size && s->alloc == &owner && count == 1);
    assert(ranges[0].address == addr && ranges[0].bytes == image_size);
    advice++;
    return early;
}
static VkMemImageAliasPlacement vk_mem_bind_image_alias(VkImage image, uint64_t a, const VkMemoryRequirements *req,
    const VkMemSpan *s, const RcInterval *ranges, uint32_t count, VkResult *v) {
    assert(image == 17 && a == addr && req->size == image_size && s->alloc == &owner && count == 1 && *v == VK_SUCCESS);
    assert(ranges[0].address == addr && ranges[0].bytes == image_size);
    binds++;
    return final;
}
static int vk_mem_image_alias_current(VkImage image, uint64_t a, uint64_t bytes, void *alloc,
    const RcInterval *ranges, uint32_t count) {
    assert(image == 17 && a == addr && bytes == image_size && alloc == &owner && count == 1);
    assert(ranges[0].address == addr && ranges[0].bytes == image_size);
    return !direct_stale;
}
static void lock(void) { assert(!locked); locked = 1; }
static void unlock(void) { assert(locked); locked = 0; }
static int image_alias_source_current(const Tex *e) { return producer_current && e == owner; }
static int rc_registry_runtime(void) { return 88; }
static int outside_replay;
static RcRegistryPosition vk_exec_resource_position(void) {
    return outside_replay ? (RcRegistryPosition){0} : (RcRegistryPosition){7, 14};
}
static int rc_registry_position_valid(RcRegistryPosition read) { return read.scene && read.ordinal; }
static int image_alias_ordinary_color(const Tex *entry) { (void)entry; abort(); } // Fixture exercises raw placement.
static int image_alias_load_current(Tex *, const VkImageSubresourceRange *, RcRegistryPosition);
static int rc_registry_native_image_covers(int registry, uint64_t image, const RcInterval *demands,
    unsigned count, RcRegistryPosition read) {
    assert(registry == 88 && image == 17 && demands && count == 1 && read.scene == 7 && read.ordinal == 14);
    return registry_current;
}
static int rc_registry_native_image_covers_handed(int registry, uint64_t image,
    const RcInterval *demands, unsigned count) {
    assert(registry == 88 && image == 17 && demands && count == 1 && outside_replay);
    return registry_current;
}
''' + function(texture, "bind_pool_memory") + "\n" + function(texture, "preserve_pool_contents_current") + \
        "\n" + function(texture, "preserve_pool_contents_handed") + "\n" + function(texture, "vk_texture_attachment_load_ready") + r'''
typedef struct {int current; VkImage image;} VkTextureAliasRead;
typedef struct {
    uint64_t serial;
    void *mem;
    struct {uint64_t size;} L;
    struct {int format;} d;
} AttachmentTexture;
enum { OP_CLEAR_DS = 10, VK_EXEC_PREPARATION_IMAGE = 1, VK_EXEC_PREPARATION_ATTACHMENT = 4,
       VK_EXEC_GRAPHICS_UNAVAILABLE = 5, LOAD_ATTACHMENT_SEED_CURRENT = 0x101,
       LOAD_ATTACHMENT_NO_CURRENT_SOURCE = 0x108 };
typedef struct {
    int owner, reason, result, detail;
    uint32_t role, slot;
    VkImage image;
    uint64_t material_serial;
    const char *stage;
    struct {
        RcRegistryPosition position;
        int present, seeded, native_checked, preserve;
        uint64_t address, bytes;
        unsigned guest_format, native_format, width, height, samples, level, layer, layers;
    } load;
} VkExecPreparationFailure;
typedef struct {
    const VkTexImage *img;
    const AttachmentTexture *tex;
    VkImage image, bc_backing;
    unsigned aspect, format, blocks, level, base_layer, layers, is3d, width, height, samples;
} VkExecTarget;
typedef struct {
    VkTextureAliasRead *load_seed;
    VkImageSubresourceRange loaded_planes;
    struct {uint64_t offset;} range;
} VkExecTargetBinding;
typedef struct {VkExecTargetBinding bindings[9];} VkExecGraphicsPass;
typedef struct {unsigned pass_ncolor, pass_layers; VkExecTarget color[8], depth;} VkExec;

''' + function(replay, "attachment_load_planes") + r'''
static int image_alias_load_current(Tex *e, const VkImageSubresourceRange *planes, RcRegistryPosition read) {
    return planes && planes->aspectMask && rc_registry_position_valid(read) && preserve_pool_contents_current(e);
}

static int vk_texture_alias_target_load_current(const VkTextureAliasRead *seed, VkImage image,
                                                  const VkImageSubresourceRange *planes) {
    return seed && seed->current && seed->image == image && planes && planes->levelCount;
}
static int vk_texture_render_target_load_current(VkImage image, uint64_t address, const VkImageSubresourceRange *planes) {
    return owner && owner->pub.image == image && owner->pub.addr == address &&
        image_alias_load_current(owner, planes, vk_exec_resource_position());
}
static void vk_exec_required_graphics_unavailable(VkExec *ex, unsigned opcode, const char *reason) {
    assert(ex && !opcode && reason);
}
static VkExecPreparationFailure last_failure;
static void graphics_preparation_unavailable(VkExec *ex, unsigned op, int result,
    const VkExecPreparationFailure *failure, const char *reason) {
    assert(ex && op == OP_CLEAR_DS && reason && !locked && result == VK_EXEC_GRAPHICS_UNAVAILABLE &&
        failure->owner == VK_EXEC_PREPARATION_IMAGE && failure->reason == VK_EXEC_PREPARATION_ATTACHMENT);
    last_failure = *failure;
    refusals++;
}
static int pass_issued;
static void open_pass(VkExec *ex, VkExecGraphicsPass *admitted) {
    const unsigned opcode = OP_CLEAR_DS;
''' + guard + r'''
    pass_issued++;
}
static int complete_load(uint32_t slot, VkExec *ex, VkExecTargetBinding *t, const VkExecTarget *selected, VkImage image) {
    VkImageSubresourceRange range;
    const VkImageSubresourceRange *planes = attachment_load_planes(ex, selected, slot, &range);
    t->loaded_planes = (VkImageSubresourceRange){0};
''' + prepare.replace("return;", "return 0;") + r'''
    return t->loaded_planes.aspectMask != 0;
}
int main(void) {
    VkCtx ctx = {1};
    VkImageCreateInfo create = 1;
    VkResult failure = VK_SUCCESS;
    VkImage image = 0;
    Tex e = {.pub = {.addr = addr}};

    final = VK_MEM_IMAGE_ALIAS_PRIVATE_PRESERVE;
    assert(bind_pool_memory(&ctx, &e, &create, logical_size, &image, &failure) == VK_MEM_IMAGE_ALIAS_PRIVATE_PRESERVE);
    assert(image == 17 && creates == 1 && binds == 1 && advice == 0 && mapping_calls == 2);
    assert(e.pub.preserve_pool_bytes && !e.bound_to && !e.image_device_count);
    assert(e.image_private_mapping_count == 1 && e.image_private_mapping[0].bytes == logical_size);
    free(e.image_private_mapping);
    e = (Tex){.pub = {.addr = addr}};
    image = 0;

    early = VK_MEM_IMAGE_ALIAS_PRIVATE_PRESERVE;
    assert(bind_pool_memory(&ctx, &e, &create, logical_size, &image, &failure) == early);
    assert(!image && creates == 1 && binds == 1 && advice == 1 && e.pub.preserve_pool_bytes);
    free(e.image_private_mapping);
    e = (Tex){.pub = {.addr = addr}};

    early = VK_MEM_IMAGE_ALIAS_DIRECT;
    final = VK_MEM_IMAGE_ALIAS_STALE;
    assert(bind_pool_memory(&ctx, &e, &create, logical_size, &image, &failure) == final);
    assert(creates == 2 && binds == 2 && advice == 2 && !e.bound_to && !e.pub.preserve_pool_bytes);
    image = 0;
    drift = 1;
    assert(bind_pool_memory(&ctx, &e, &create, logical_size, &image, &failure) == VK_MEM_IMAGE_ALIAS_STALE);
    assert(creates == 3 && binds == 2);
    drift = 0;
    image = 0;
    fits = 0;
    assert(bind_pool_memory(&ctx, &e, &create, logical_size, &image, &failure) == VK_MEM_IMAGE_ALIAS_FAILED);
    assert(!image && creates == 3 && n_bind_skips == 1);
    fits = 1;
    stale_mapping = 1;
    assert(bind_pool_memory(&ctx, &e, &create, logical_size, &image, &failure) == VK_MEM_IMAGE_ALIAS_STALE);
    assert(!image && creates == 3 && binds == 2);
    stale_mapping = 0;
    final = VK_MEM_IMAGE_ALIAS_DIRECT;
    direct_stale = 1;
    assert(bind_pool_memory(&ctx, &e, &create, logical_size, &image, &failure) == VK_MEM_IMAGE_ALIAS_STALE);
    assert(image == 17 && !e.bound_to && !e.image_device_count);
    direct_stale = 0;
    image = 0;
    assert(bind_pool_memory(&ctx, &e, &create, logical_size, &image, &failure) == final);
    assert(e.bound_to == &owner && e.image_device_count == 1 && e.image_device_size == image_size);
    assert(e.image_device_intervals[0].bytes == image_size && !e.pub.preserve_pool_bytes);
    free(e.image_device_intervals);
    free(cached);

    VkTexImage old_color = {.image = 11, .addr = 0x300000};
    VkTexImage depth = {.image = 17, .addr = addr, .preserve_pool_bytes = 1, .format = VK_FORMAT_D32_SFLOAT,
        .aspect = VK_IMAGE_ASPECT_DEPTH_BIT, .levels = 1, .layers = 1};
    VkTexImage peer = {.image = 17, .addr = addr, .preserve_pool_bytes = 1, .format = 9,
        .aspect = VK_IMAGE_ASPECT_COLOR_BIT, .levels = 1, .layers = 1};
    RcInterval demands = {addr, logical_size};
    e = (Tex){.pub = depth, .alias_full_demand = &demands, .alias_full_count = 1};
    owner = &e;
    VkExec pass = {.pass_ncolor = 2, .pass_layers = 1,
        .color = {{.img = &old_color, .image = 11},
                  {.img = &peer, .image = 17, .format = 9, .layers = 1, .samples = 1, .aspect = VK_IMAGE_ASPECT_COLOR_BIT}},
        .depth = {.img = &depth, .image = 17, .layers = 1, .samples = 1, .aspect = VK_IMAGE_ASPECT_DEPTH_BIT,
                  .format = VK_FORMAT_D32_SFLOAT}};
    open_pass(&pass, NULL);
    assert(refusals == 1 && !pass_issued && last_failure.slot == 1 &&
        last_failure.detail == LOAD_ATTACHMENT_NO_CURRENT_SOURCE && last_failure.image == 17);
    VkTextureAliasRead ready_seed = {1, 17};
    VkExecGraphicsPass selected = {.bindings[1].load_seed = &ready_seed};
    pass.depth.image = 0;
    open_pass(&pass, &selected);
    assert(refusals == 1 && pass_issued == 1);
    ready_seed.image = 20;
    open_pass(&pass, &selected);
    assert(refusals == 2 && pass_issued == 1 && last_failure.slot == 1 &&
        last_failure.detail == LOAD_ATTACHMENT_SEED_CURRENT && last_failure.image == 17);

    pass.color[1] = (VkExecTarget){0};
    pass.depth.image = 17;
    producer_current = registry_current = 1;  // actual earlier whole native depth clear
    open_pass(&pass, NULL);
    assert(refusals == 2 && pass_issued == 2);
    VkExecTargetBinding target = {.range.offset = addr};
    assert(complete_load(8, &pass, &target, &pass.depth, 17));
    open_pass(&pass, NULL);
    assert(pass_issued == 3);   // the selected attachment retained its whole depth version
    pass.depth.aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
    assert(!complete_load(8, &pass, &target, &pass.depth, 17));
    pass.depth.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    pass.depth.level = 1;
    assert(!complete_load(8, &pass, &target, &pass.depth, 17));
    pass.depth.level = 0;
    registry_current = 0;   // a later conflicting guest writer cannot preserve this LOAD
    open_pass(&pass, NULL);
    assert(refusals == 3 && pass_issued == 3);
    assert(!complete_load(8, &pass, &target, &pass.depth, 17));
    outside_replay = 1;
    assert(!preserve_pool_contents_current(&e));
    assert(!preserve_pool_contents_handed(&e));
    e.pub.gpu_owned = 1;
    assert(!preserve_pool_contents_handed(&e) && !preserve_pool_contents_current(&e));
    registry_current = 1;
    assert(preserve_pool_contents_handed(&e) && !preserve_pool_contents_current(&e));
    e.overwritten = 1;
    assert(!preserve_pool_contents_handed(&e));
    puts("RAW placement and LOAD: preserve versus direct/failure/stale, exact seed only, D32 close/reopen and packed/mip/conflict refusals");
}
'''


def main():
    cc = Path(os.environ.get("CC") or shutil.which("clang") or "").resolve()
    if not cc.is_file():
        raise SystemExit("Clang is required (set CC)")
    runtime = Path(os.getenv("IMAGE_PLACEMENT_CHECK_SOURCE", ROOT / "runtime"))
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="native-image-placement-") as temp:
        source = Path(temp) / "check.c"
        exe = Path(temp) / ("check.exe" if os.name == "nt" else "check")
        source.write_text(make_fixture(runtime), encoding="utf-8")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=undefined", str(source), "-o", str(exe)],
                       env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()


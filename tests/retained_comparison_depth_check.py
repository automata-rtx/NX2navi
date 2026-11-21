from pathlib import Path
import os
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TEXTURE = Path(os.environ.get("DEPTH_COMPARE_TEXTURE_SOURCE", ROOT / "runtime/vk_texture.c"))
DRAW = Path(os.environ.get("DEPTH_COMPARE_DRAW_SOURCE", ROOT / "runtime/vk_exec_draw.c"))


def function(source, name):
    match = re.search(r"^(?:static )?[^;\n]*\b" + name + r"\([^;{}]*\)\s*\{", source, re.M)
    assert match, name
    return source[match.start(): source.index("\n}", match.end()) + 2]


MODEL = r'''
#include <assert.h>
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct {
    VkImage image;
    VkImageType type;
    VkFormat format;
    VkImageAspectFlags aspect;
    VkImageUsageFlags usage;
    uint32_t width, height, depth, levels, layers, samples;
} VkTexImage;
typedef struct {
    VkBufferImageCopy copy;
    uint32_t rows, row_bytes, slices;
} VkTextureAliasPlane;
enum { VK_TEXTURE_ALIAS_MAX_PLANES = 128 };
typedef struct {
    VkTextureAliasPlane planes[VK_TEXTURE_ALIAS_MAX_PLANES];
    uint32_t count;
    uint64_t bytes;
} VkTextureAliasLayout;
typedef struct {
    VkImageCreateFlags flags;
    VkFormat format;
    VkImageUsageFlags usage;
    uint32_t w,h,levels,layers;
} ImgKey;
typedef struct {
    VkTexImage result;
    ImgKey placement;
    VkTextureAliasLayout receiver_layout;
    struct {unsigned char *map;} staging;
    VkBuffer staged;
    void *allocation, *target_hold;
    int transfer_source, raw_output, storage_seed, uploaded, registration_only, comparison_depth;
    unsigned source_count;
    struct {VkTexImage image;} sources[2];
    uint64_t upload_offset;
} VkTextureAliasRead;
#define VK_POOL_TEX 32768
#define VK_POOL_KINDS 16
#define POOL_DEPTH_TWIN_KINDS ((1u << 10) | (1u << 11))
#define NVN_TEXTURE_SWIZZLE_R 2
enum { VK_TEXTURE_ALIAS_LAYOUT = 1, VK_TEXTURE_ALIAS_OWNER, VK_TEXTURE_ALIAS_CAPACITY };
enum { RC_REGISTRY_UNSUPPORTED = 7, RC_REGISTRY_STALE, RC_REGISTRY_CAPACITY };
typedef struct {uint64_t tag;} RcInterval;
typedef struct {int target;} view_t;
typedef struct {
    const VkTexImage *image;
    view_t view;
    int has_view;
    uint64_t registration;
} VkTextureSample;
typedef enum { VK_TEXTURE_SAMPLE_READY, VK_TEXTURE_SAMPLE_FALLBACK } VkTextureSampleStatus;
typedef struct {
    int phase, status;
    uint64_t registration;
    VkImage image;
    const char *layout_gate;
    RcInterval receiver;
} VkTextureAliasFailure;
typedef struct {
    VkTextureAliasRead *read;
    VkTextureSample sample;
    VkTexImage image;
    uint32_t slot, kinds;
} PoolAliasMember;
typedef struct {
    PoolAliasMember members[2];
    unsigned count, refused;
    const char *stage;
} PoolAliases;
typedef struct {
    VkFormat format;
    VkImageAspectFlags aspect;
    uint32_t swz;
} ViewKey;
typedef struct TextureNativeUse {
    struct TextureNativeUse *next;
    const VkTextureAliasRead *alias;
    uint32_t descriptor;
} TextureNativeUse;
typedef struct { const TextureNativeUse *native; } TextureImagePublication;
typedef struct { VkImage image, bc_backing; } VkExecTarget;
typedef struct { VkExecTarget color[8], depth; } VkExecGraphicsPassShape;
typedef struct { size_t nfbl; VkImage fbl_img[9]; } VkExec;
typedef struct {
    VkExecGraphicsPassShape *pass;
    VkExec *ex;
    PoolAliases *aliases, *comparison_aliases;
    uint64_t read;
    uint32_t kinds;
} GraphicsPoolRestore;
static struct {int split_samplers;} PL={.split_samplers=1};
static uint32_t pool_request_kinds(const GraphicsPoolRestore *request, uint32_t slot) {
    assert(slot == 3);
    return request->kinds;
}
static int pool_comparison_format(const ViewKey *key, uint64_t *features) {
    assert(key->format == VK_FORMAT_R16_UNORM || key->format == VK_FORMAT_R32_SFLOAT);
    *features = 0;
    return 0;
}
static uint16_t pool_kind_type(uint32_t kind) {return kind;}
static int texture_view_key(const VkTexImage *source, const view_t *view, int has_view, uint16_t kind, int storage,
                            ViewKey *key) {
    (void)view;(void)has_view;
    assert((kind==10 || kind==11) && !storage);
    *key=(ViewKey){source->format,source->aspect,NVN_TEXTURE_SWIZZLE_R};
    return 1;
}
static VkTextureAliasRead captured;
static VkTextureSample issued_sample;
static int capture_refused;
static unsigned captures, cancellations;
static int read_lost;
static VkTextureSampleStatus vk_texture_comparison_alias_capture_locked(int slot, uint64_t read,
    VkTextureSample *sample, VkTextureAliasRead **alias, VkTexImage *shape, VkTextureAliasFailure *failure) {
    (void)failure;
    assert(slot==3 && read==73);
    ++captures;
    if (capture_refused) return VK_TEXTURE_SAMPLE_FALLBACK;
    *sample=issued_sample;
    if (read_lost) ++sample->registration;
    *shape=captured.result;
    *alias=&captured;
    return VK_TEXTURE_SAMPLE_READY;
}
static void vk_texture_alias_read_cancel(VkTextureAliasRead *alias) {
    assert(alias==&captured);
    ++cancellations;
}
static const RcInterval *vk_texture_alias_read_intervals(const VkTextureAliasRead *alias, uint32_t *count) {
    assert(alias==&captured);
    static const RcInterval origin={91};
    *count=1;
    return &origin;
}
static PoolAliasMember *pool_alias_member(PoolAliases *aliases, uint32_t id) {
    return aliases->count && aliases->members[0].slot==id ? aliases->members : NULL;
}
static VkResult pool_aliases_append(PoolAliases *aliases, PoolAliasMember member) {
    assert(aliases->count==0);
    aliases->members[aliases->count++]=member;
    return VK_SUCCESS;
}
static void pool_alias_refuse(PoolAliases *aliases, uint32_t id, uint32_t kind, const char *gate, const VkTextureAliasFailure *failure) {
    assert(id==3 && kind==10 && failure->status);
    ++aliases->refused;
    aliases->stage=gate;
}
typedef struct {VkPhysicalDevice phys;} VkCtx;
static VkExecGraphicsPassShape *vk_exec_graphics_pass_shape(VkExecGraphicsPassShape *shape) { return shape; }
static VkCtx *vk_ctx(void) {
    static VkCtx context;
    return &context;
}
static VkFormatFeatureFlags2 format_features;
static uint32_t native_layers;
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties2(VkPhysicalDevice physical, VkFormat format, VkFormatProperties2 *p) {
    assert(physical == vk_ctx()->phys && (format == VK_FORMAT_D16_UNORM || format == VK_FORMAT_D32_SFLOAT));
    VkFormatProperties3 *format3 = p->pNext;
    format3->optimalTilingFeatures = format_features;
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties(VkPhysicalDevice physical, VkFormat format,
    VkImageType type, VkImageTiling tiling, VkImageUsageFlags usage, VkImageCreateFlags flags, VkImageFormatProperties *p) {
    assert(physical == vk_ctx()->phys && (format == VK_FORMAT_D16_UNORM || format == VK_FORMAT_D32_SFLOAT));
    assert(type == VK_IMAGE_TYPE_2D && tiling == VK_IMAGE_TILING_OPTIMAL && !flags);
    assert(usage == (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
    *p = (VkImageFormatProperties){.maxExtent={64,64,1}, .maxMipLevels=8, .maxArrayLayers=native_layers,
                                  .sampleCounts=VK_SAMPLE_COUNT_1_BIT};
    return VK_SUCCESS;
}
'''

TEST = r'''
static VkTextureAliasRead color(VkFormat format, unsigned bpp, unsigned char *stage) {
    VkTextureAliasRead alias = {.result={.format=format,.type=VK_IMAGE_TYPE_2D,.aspect=VK_IMAGE_ASPECT_COLOR_BIT,
        .usage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,.width=2,.height=1,.depth=1,.levels=2,.layers=2,.samples=1},
        .placement={.format=format,.w=2,.h=1,.levels=2,.layers=2},
        .receiver_layout={.count=2,.bytes=32},.staging={stage},.upload_offset=16};
    for (unsigned i=0; i<2; ++i) {
        alias.receiver_layout.planes[i]=(VkTextureAliasPlane){.copy={.bufferOffset=16*i,
            .imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,i,0,2},.imageExtent={2-i,1,1}},
            .rows=1,.row_bytes=(2-i)*bpp,.slices=2};
    }
    return alias;
}
int main(void) {
    format_features = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT |
                      VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;
    native_layers = 4;
    unsigned char bytes[64] = {0};
    VkTextureAliasRead r16=color(VK_FORMAT_R16_UNORM,2,bytes), r32=color(VK_FORMAT_R32_SFLOAT,4,bytes);
    VkTexImage depth;
    assert(vk_texture_alias_depth_compare_locked(&r16,&depth));
    assert(depth.format==VK_FORMAT_D16_UNORM && depth.aspect==VK_IMAGE_ASPECT_DEPTH_BIT && depth.layers==2 && depth.levels==2);
    assert(depth.usage==(VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
    assert(r16.placement.format==depth.format && r16.receiver_layout.planes[1].copy.imageSubresource.aspectMask==VK_IMAGE_ASPECT_COLOR_BIT);
    assert(image_alias_sample_depth_values_valid(&r16));

    VkTextureAliasRead original=r32;
    native_layers=1;
    assert(!vk_texture_alias_depth_compare_locked(&r32,&depth) && !memcmp(&r32,&original,sizeof r32));
    native_layers=4;
    format_features &= ~VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;
    assert(!vk_texture_alias_depth_compare_locked(&r32,&depth));
    format_features |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;
    r32.target_hold=(void *)1;
    assert(!vk_texture_alias_depth_compare_locked(&r32,&depth));
    r32.target_hold=NULL;
    r32.receiver_layout.planes[1].copy.bufferOffset=18;
    assert(!vk_texture_alias_depth_compare_locked(&r32,&depth));
    r32.receiver_layout.planes[1].copy.bufferOffset=16;
    assert(vk_texture_alias_depth_compare_locked(&r32,&depth));
    assert(depth.format==VK_FORMAT_D32_SFLOAT && image_alias_sample_depth_values_valid(&r32));
    const uint32_t normalized[] = {UINT32_C(0x3f800000),UINT32_C(0x3f000000),UINT32_C(0x80000000)};
    memcpy(bytes+16,normalized,sizeof normalized);
    assert(image_alias_sample_depth_values_valid(&r32));
    const uint32_t invalid[] = {UINT32_C(0x3f800001),UINT32_C(0xbf800000),UINT32_C(0x7fc00000)};
    for (unsigned i=0; i<3; ++i) {
        memcpy(bytes+32,&invalid[i],sizeof invalid[i]);
        assert(!image_alias_sample_depth_values_valid(&r32));
    }
    VkBufferImageCopy regions[VK_TEXTURE_ALIAS_MAX_PLANES];
    copy_regions(&r32,regions);
    assert(regions[0].imageSubresource.aspectMask==VK_IMAGE_ASPECT_DEPTH_BIT && regions[0].bufferOffset==16);
    assert(regions[1].imageSubresource.aspectMask==VK_IMAGE_ASPECT_DEPTH_BIT && regions[1].bufferOffset==32 &&
           regions[1].imageSubresource.mipLevel==1 && regions[1].imageSubresource.layerCount==2);
    copy_regions(&original,regions);
    assert(regions[1].imageSubresource.aspectMask==VK_IMAGE_ASPECT_COLOR_BIT);
    assert(!vk_texture_alias_depth_compare_locked(&original,NULL));

    const VkImage source=(VkImage)(uintptr_t)7, other=(VkImage)(uintptr_t)8;
    r32.source_count=1;
    r32.sources[0].image.image=source;
    VkExecGraphicsPassShape pass={.color={{.image=other}}};
    VkExec ex={0};
    GraphicsPoolRestore request={.pass=&pass,.ex=&ex};
    assert(pool_alias_depth_before_draw(&request,&r32));
    pass.color[0].bc_backing=source;
    assert(!pool_alias_depth_before_draw(&request,&r32));
    pass.color[0].bc_backing=VK_NULL_HANDLE;
    pass.depth.image=source;
    assert(!pool_alias_depth_before_draw(&request,&r32));
    pass.depth.image=VK_NULL_HANDLE;
    ex.nfbl=1; ex.fbl_img[0]=source;
    assert(!pool_alias_depth_before_draw(&request,&r32));

    // The same logical texture can keep its ordinary color provider while only its paired
    // comparison kinds acquire a second retained image at the identical original cut.
    ex.nfbl=0;
    PoolAliases ordinary={0}, comparison={0};
    request.aliases=&ordinary;
    request.comparison_aliases=&comparison;
    request.read=73;
    request.kinds=1u<<2;
    VkTextureAliasRead color_source=color(VK_FORMAT_R16_UNORM,2,bytes);
    captured=color_source;
    issued_sample=(VkTextureSample){.image=&color_source.result,.registration=17};
    assert(!pool_prepare_depth_alias_locked(&request,3,1u<<2|1u<<10,&issued_sample));
    assert(!captures && !comparison.count);
    request.kinds|=1u<<10;
    assert(pool_prepare_depth_alias_locked(&request,3,1u<<2|1u<<10,&issued_sample)==1u<<10);
    assert(!ordinary.count && comparison.count==1 && captures==1 && !cancellations);
    assert(comparison.members[0].image.format==VK_FORMAT_D16_UNORM && color_source.result.format==VK_FORMAT_R16_UNORM);
    assert(pool_prepare_depth_alias_locked(&request,3,1u<<10,&issued_sample)==1u<<10 && captures==1);
    VkTextureAliasRead source_owner=captured;
    source_owner.source_count=1;
    source_owner.sources[0].image.image=source;
    const TextureNativeUse entry={.alias=&source_owner,.descriptor=10*VK_POOL_TEX+3};
    const TextureImagePublication publication={.native=&entry};
    assert(vk_texture_publication_comparison_source_locked(&publication,1u<<10,source));
    assert(!vk_texture_publication_comparison_source_locked(&publication,1u<<2,source));
    assert(!vk_texture_publication_comparison_source_locked(&publication,1u<<10,other));
    comparison=(PoolAliases){0};
    captured=color_source;
    read_lost=1;
    assert(pool_prepare_depth_alias_locked(&request,3,1u<<10,&issued_sample)==-1 && cancellations==1);
    assert(!comparison.count && comparison.refused && !strcmp(comparison.stage,"comparison-twin-original"));
    read_lost=0;
    comparison=(PoolAliases){0};
    capture_refused=1;
    assert(pool_prepare_depth_alias_locked(&request,3,1u<<10,&issued_sample)==-1 && cancellations==1);
    assert(!comparison.count && comparison.refused && !strcmp(comparison.stage,"comparison-twin-source"));
    puts("Retained comparison alias: source admission, separate cohorts, D16/D32 planes, values, feedback and upload passed");
}
'''


def main():
    texture, draw = TEXTURE.read_text(), DRAW.read_text()
    upload = function(texture, "vk_texture_alias_read_upload")
    start = upload.index("    for (uint32_t i = 0; i < alias->receiver_layout.count; i++)")
    end = upload.index("    vkCmdCopyBufferToImage", start)
    regions = "static void copy_regions(VkTextureAliasRead *alias, VkBufferImageCopy *regions) {\n" + upload[start:end] + "}\n"
    src = MODEL + "\n".join(function(texture, name) for name in (
        "vk_texture_color_depth_format", "vk_texture_alias_depth_compare_locked",
        "image_alias_sample_depth_values_valid", "vk_texture_alias_read_uses_image",
        "vk_texture_alias_comparison_source", "vk_texture_publication_comparison_source_locked"
    )) + "\n".join(function(draw, name) for name in (
        "pool_alias_depth_before_draw", "pool_alias_depth_required",
        "pool_alias_depth_select_locked", "pool_prepare_depth_alias_locked"
    )) + regions + TEST
    compiler = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(compiler.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="retained-comparison-depth-") as tmp:
        source, executable = Path(tmp)/"check.c", Path(tmp)/"check.exe"
        source.write_text(src)
        subprocess.run([str(compiler), "-std=gnu11", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", str(source), "-o", str(executable)],
                       env=env, check=True)
        subprocess.run([str(executable)], env=env, check=True)


if __name__ == "__main__":
    main()

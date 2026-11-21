"""Check the actual ordered texture copy, resolve and blit selection."""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "runtime/vk_exec_xfer.c").read_text(encoding="utf-8")


def between(source, begin, end):
    first = source.index(begin)
    return source[first:source.index(end, first)]


ADAPTER = r'''
#define VK_NO_PROTOTYPES
#include <assert.h>
#include <stdlib.h>
#include "tex.h"
#include "vk.h"
#include "vk_exec.h"

struct VkTexturePresentSource { int held; };
struct VkTextureRestore { int held; };
struct VkTextureWriteTarget { int held; };
enum { W_FORMAT, W_REGION };
static VkTexturePresentSource resident;
static VkTextureRestore restore;
static VkTextureWriteTarget target;
static VkCtx context;
static VkTexImage src_shape, dst_shape;
static tex_t input, output;
static int source_calls, restore_calls, capture_calls, releases, cold, cold_target, contact, paired;
static VkFormatFeatureFlags features;
VkCtx *vk_ctx(void) { return &context; }
static void warn_once(int kind, const char *fmt, ...) { assert(kind >= 0 && fmt); }
void rt_fatal(const char *fmt, ...) { assert(fmt); abort(); }
static uint32_t max1(uint32_t value) { return value ? value : 1; }
int tex_vk_format(int format, TexVkFormat *f) {
    *f=(TexVkFormat){.vk=VK_FORMAT_R8G8B8A8_UNORM,.ideal=VK_FORMAT_R8G8B8A8_UNORM};
    return format==NVN_FORMAT_RGBA8;
}
int nvn_fmt_bpb(int format, int *bw, int *bh) {
    assert(format==NVN_FORMAT_RGBA8);
    *bw=*bh=1;
    return 4;
}
int vk_contact_pending(void) { return contact; }
int vk_texture_write_target_shape(const tex_t *t, VkTexImage *out) {
    assert(t==&output);
    *out=dst_shape;
    return !cold_target;
}
int vk_texture_write_target_restore_shape(const tex_t *t, VkTexImage *out) {
    assert(t==&output && cold_target && !dst_shape.image && !dst_shape.gpu_owned);
    *out=dst_shape;
    return 1;
}
int vk_texture_write_target_capture(const tex_t *t,const VkTexImage *shape,VkTextureWriteTarget **out) {
    assert(t==&output && shape->image==dst_shape.image && !target.held && (resident.held || restore.held));
    capture_calls++;
    target.held=1;
    *out=&target;
    return 1;
}
int vk_texture_write_target_needs_allocation(const VkTextureWriteTarget *captured) {
    assert(captured==&target && target.held);
    return !dst_shape.gpu_owned;
}
VkTexturePresentSource *vk_texture_transfer_read_capture(const tex_t *t, VkTexImage *out) {
    assert(t==&input && !resident.held && !target.held);
    source_calls++;
    if (cold) return NULL;
    *out=src_shape;
    resident.held=1;
    return &resident;
}
VkTextureRestore *vk_texture_restore_capture(const tex_t *t,VkTexImage *out) {
    assert(t==&input && cold && !restore.held && !target.held);
    restore_calls++;
    *out=src_shape;
    out->image=VK_NULL_HANDLE;
    restore.held=1;
    return &restore;
}
void vk_texture_restore_pair_write_target(VkTextureRestore *restoring, VkTextureWriteTarget *writing) {
    assert(restoring==&restore && restore.held && writing==&target && target.held && !dst_shape.gpu_owned);
    paired++;
}
void vk_exec_texture_to_texture_cancel(VkExecTextureToTexture *copy) {
    if (copy->source_hold) { assert(resident.held); resident.held=0; releases++; }
    if (copy->restore) { assert(restore.held); restore.held=0; releases++; }
    if (copy->target) { assert(target.held); target.held=0; }
    *copy=(VkExecTextureToTexture){0};
}
void vkGetPhysicalDeviceFormatProperties(VkPhysicalDevice phys,VkFormat fmt,VkFormatProperties *out) {
    assert(phys==context.phys && fmt==src_shape.format);
    *out=(VkFormatProperties){.optimalTilingFeatures=features};
}
static p_copy_t2t reset(void) {
    assert(!resident.held && !restore.held && !target.held);
    source_calls=restore_calls=capture_calls=releases=cold=cold_target=contact=paired=0;
    input.d.format=output.d.format=NVN_FORMAT_RGBA8;
    dst_shape=(VkTexImage){.image=(VkImage)(uintptr_t)4,.usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .type=VK_IMAGE_TYPE_2D,.format=VK_FORMAT_R8G8B8A8_UNORM,.aspect=VK_IMAGE_ASPECT_COLOR_BIT,
        .width=32,.height=32,.depth=1,.levels=2,.layers=2,.samples=1,.scale=1,.gpu_owned=1};
    src_shape=dst_shape;
    src_shape.usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    features=VK_FORMAT_FEATURE_BLIT_SRC_BIT|VK_FORMAT_FEATURE_BLIT_DST_BIT|VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    return (p_copy_t2t){.src=&input,.dst=&output,.has_sv=1,.has_dv=1,
        .sv={.set_levels=1,.base_level=1},.dv={.set_levels=1,.base_level=1},
        .sr={2,3,0,5,5,1},.dr={4,5,0,5,5,1}};
}
'''

CHECKS = r'''
int main(void) {
    VkExec ex={0};
    VkExecTextureToTexture copy={0};
    p_copy_t2t op=reset();
    assert(vk_exec_texture_to_texture_capture(&ex,&op,&copy)==1 && copy.op==VK_EXEC_IMAGE_COPY && copy.target);
    assert(source_calls==1 && capture_calls==1 && !restore_calls);
    assert(copy.scratch_recipe.sType && copy.scratch_recipe.extent.width==5 && copy.scratch_recipe.extent.height==5);
    assert(copy.scratch_offset.x==2 && copy.scratch_offset.y==3 && copy.scratch_recipe.arrayLayers==1);
    vk_exec_texture_to_texture_cancel(&copy);

    op=reset();
    op.dr.zoffset=1;
    assert(vk_exec_texture_to_texture_capture(&ex,&op,&copy)==1 && !copy.scratch_recipe.sType);
    vk_exec_texture_to_texture_cancel(&copy);

    op=reset();
    op.dr.xoffset=8;
    assert(vk_exec_texture_to_texture_capture(&ex,&op,&copy)==1 && !copy.scratch_recipe.sType);
    vk_exec_texture_to_texture_cancel(&copy);

    op=reset();
    dst_shape.gpu_owned=0;
    assert(vk_exec_texture_to_texture_capture(&ex,&op,&copy)==1 && copy.target && !copy.scratch_recipe.sType);
    vk_exec_texture_to_texture_cancel(&copy);

    op=reset();
    op.dr.xoffset=7;
    op.dr.width=2;
    op.flags=NVN_COPY_FLAGS_LINEAR_FILTER_BIT;
    assert(vk_exec_texture_to_texture_capture(&ex,&op,&copy)==1 && copy.op==VK_EXEC_IMAGE_BLIT);
    assert(copy.scratch_recipe.extent.width==16 && copy.scratch_recipe.extent.height==16);
    assert(!copy.scratch_offset.x && !copy.scratch_offset.y);
    vk_exec_texture_to_texture_cancel(&copy);

    op=reset();
    src_shape.image=(VkImage)(uintptr_t)7;
    src_shape.samples=4;
    assert(vk_exec_texture_to_texture_capture(&ex,&op,&copy)==1 && copy.op==VK_EXEC_IMAGE_RESOLVE);
    assert(copy.target && !copy.scratch_recipe.sType);
    vk_exec_texture_to_texture_cancel(&copy);

    op=reset();
    cold=1;
    dst_shape.gpu_owned=0;
    assert(vk_exec_texture_to_texture_capture(&ex,&op,&copy)==1 && copy.target && copy.restore);
    assert(restore_calls==1 && capture_calls==1 && paired==1);
    vk_exec_texture_to_texture_cancel(&copy);
    op=reset();
    cold=cold_target=1;
    dst_shape.image=VK_NULL_HANDLE;
    dst_shape.gpu_owned=0;
    assert(vk_exec_texture_to_texture_capture(&ex,&op,&copy)==1 && copy.target && copy.restore);
    assert(restore_calls==1 && capture_calls==1 && paired==1);
    vk_exec_texture_to_texture_cancel(&copy);
    op=reset();
    cold=1;
    dst_shape.gpu_owned=1;
    assert(vk_exec_texture_to_texture_capture(&ex,&op,&copy)==1 && copy.target && copy.restore && restore_calls==1 && !paired);
    vk_exec_texture_to_texture_cancel(&copy);

    op=reset();
    op.sr.depth=2;
    op.dr.zoffset=1;
    assert(vk_exec_texture_to_texture_capture(&ex,&op,&copy)==1 && !copy.target && !capture_calls && releases==1);
    op=reset();
    op.dr.width=3;
    features=0;
    assert(vk_exec_texture_to_texture_capture(&ex,&op,&copy)==0 && !copy.target && !capture_calls && releases==1);
    return 0;
}
'''

region = between(SOURCE, "typedef struct {\n    const VkTexImage *img;", "static int sub_resolve(")
layers = between(SOURCE, "static VkImageSubresourceLayers sub_layers(", "static int buffer_layout(")
plan = between(SOURCE, "static int image_copy_same_layers(", "VkResult vk_exec_texture_to_texture_allocate(")
cc = Path(os.environ["CC"])
env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="ordered-texture-copy-") as tmp:
    check = Path(tmp) / "check.c"
    binary = check.with_suffix(".exe")
    check.write_text("\n".join((ADAPTER, region, layers, plan, CHECKS)), encoding="utf-8")
    subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
                    "-fsanitize=undefined", "-fsanitize-trap=all", "-I", str(ROOT / "runtime"),
                    "-I", str(cc.parent.parent / "include/vma"), str(check), "-o", str(binary)], env=env, check=True)
    subprocess.run([str(binary)], env=env, check=True)
print("Ordered T2T: exact copy, overlap scratch, filter footprint, resolve, cold source and veto passed")

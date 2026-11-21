"""Exercise exact CopyBufferToTexture input, layout, and native handoff."""
from pathlib import Path
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "runtime/vk_exec_xfer.c").read_text(encoding="utf-8")
HEADER = (ROOT / "runtime/vk_exec.h").read_text(encoding="utf-8")
DRAW = (ROOT / "runtime/vk_exec_draw.c").read_text(encoding="utf-8")


def between(source, begin, end):
    first = source.index(begin)
    return source[first:source.index(end, first)]


TYPES = r'''
#include <assert.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "resource_core_registry.h"

typedef struct {
    VkImage image;
    VkImageAspectFlags aspect;
    VkImageType type;
    VkImageUsageFlags usage;
    uint32_t levels, layers, width, height, depth, samples;
    int scale;
    VkFormat format;
} VkTexImage;
typedef struct { struct { int format; } d; } tex_t;
typedef struct { int set_levels, base_level, set_layers, min_layer, set_format, format; } view_t;
typedef struct { int xoffset, yoffset, zoffset, width, height, depth; } NVNcopyRegion;
typedef struct { uint64_t src; const tex_t *tex; view_t view; NVNcopyRegion r; int has_view; } p_copy_b2t;
typedef struct { VkBuffer buffer; uint64_t offset, avail; int cls; } VkMemSpan;
typedef struct { int held; } VkMemSpanCapture;
typedef struct { int held; } VkTextureWriteTarget;
typedef struct { int vk, ideal, expand_rgb, astc_bw; } TexVkFormat;
typedef struct { void *pool; uint64_t mirror_incarnation; } VkMemInputProvider;
typedef struct {
    uint64_t fence;
    VkMemInputProvider *input_providers;
    int ninput_providers, cinput_providers;
} CbRes;
typedef struct {
    CbRes *res;
    uint64_t res_fence, prep_incarnation;
    int B;
} DrawCtx;
typedef struct {
    DrawCtx *dc;
    uint64_t fence;
    int64_t copy_stride;
    int recorded, rt_resolved, bc_written_mask;
    VkExtent2D tonemap_extent;
    VkCommandBuffer cb;
} VkExec;
typedef struct { int unused; } VkTextureAliasRead;
typedef struct { RcRegistryNativeImageFailure coverage; } VkTextureAliasFailure;
typedef enum { VK_EXEC_RESTORE_READY, VK_EXEC_RESTORE_STOPPED, VK_EXEC_RESTORE_STALE, VK_EXEC_RESTORE_FAILED } VkExecRestoreResult;
enum { VK_MEM_MIRROR_IMPORT, OP_COPY_B2T };
'''

ADAPTERS = r'''
enum { W_BUFFER, W_REGION, W_FORMAT, VK_GAP_COPY_B2T, VK_MUSE_COPY_SRC };
static const VkTexImage shape = {.image=(VkImage)(uintptr_t)17, .aspect=VK_IMAGE_ASPECT_COLOR_BIT,
    .type=VK_IMAGE_TYPE_2D, .usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT, .width=32, .height=32, .depth=1,
    .levels=2, .layers=3, .samples=1, .scale=1};
static const VkMemSpan selection = {.buffer=(VkBuffer)(uintptr_t)21,.offset=64,.avail=8192,.cls=19};
static VkMemSpanCapture held;
static VkTextureWriteTarget retained;
static int backed, candidate, native, contact, live, effect, progress;
static int queries, selections, targets, released, native_holds, commands, writes;
static DrawCtx default_draw, existing_draw;
static DrawCtx *dc=&default_draw;
static uint64_t active_scene=6;
static unsigned callbacks, provider_releases;
static CbRes *cleanup;
static int fail_calloc, expect_fatal;
static jmp_buf fatal;
static const tex_t texture = {{1}}, unsupported = {{2}};
static uint32_t max1(uint32_t n) { return n ? n : 1; }
static void warn_once(int kind, const char *format, ...) { assert(kind >= 0 && format); }
static int tex_vk_format(int format, TexVkFormat *out) {
    *out=(TexVkFormat){.vk=1,.ideal=1,.expand_rgb=format==2};
    return 1;
}
static int nvn_fmt_bpb(int format, int *bw, int *bh) {
    assert(format == 1);
    *bw=*bh=1;
    return 4;
}
static int vk_contact_pending(void) { return contact; }
static int vk_texture_write_target_shape(const tex_t *t, VkTexImage *out) {
    assert(t==&texture);
    *out=shape;
    return candidate;
}
static int vk_mem_lookup(uint64_t address, uint64_t size, VkMemSpan *out) {
    assert(address==0x9000 && size==140);
    assert(!effect);
    queries++;
    *out=selection;
    return backed;
}
static RcRegistryResult vk_mem_span_prepare_native_input(uint64_t address,uint64_t size,int clamp,int policy,VkMemSpan *out,
                                                         VkMemSpanCapture **hold) {
    assert(address==0x9000 && size==140 && !clamp && policy==VK_MEM_MIRROR_IMPORT && !held.held && !effect);
    selections++;
    if (!native) return RC_REGISTRY_UNSUPPORTED;
    *out=selection;
    held.held=1;
    *hold=&held;
    return RC_REGISTRY_OK;
}
static int vk_mem_span_capture_current(const VkMemSpanCapture *capture,const VkMemSpan *span) {
    assert(capture==&held && held.held==1 && span->buffer==selection.buffer);
    return live;
}
static void vk_mem_span_capture_release(VkMemSpanCapture **hold) {
    if (!*hold) return;
    assert(*hold==&held && held.held==1);
    held.held=0;*hold=NULL;released++;
}
static int vk_texture_write_target_capture(const tex_t *t,const VkTexImage *image,VkTextureWriteTarget **out) {
    assert(t==&texture && image->image==shape.image && held.held==1 && !effect);
    targets++;
    retained.held=1;
    *out=&retained;
    return 1;
}
static RcRegistryPosition vk_exec_resource_position(void) { return (RcRegistryPosition){6,2}; }
static int vk_exec_recording_failed(void) { return 0; }
static int vk_sync_stopping(void) { return 0; }
static int vk_device_is_lost(void) { return 0; }
static void vk_exec_required_graphics_unavailable(VkExec *ex,int opcode,const char *reason) {
    (void)ex; (void)opcode; (void)reason; abort();
}
static void rt_log(const char *fmt,...) { (void)fmt; abort(); }
static int vk_texture_write_target_current(const VkTextureWriteTarget *target) {
    return target==&retained && retained.held==1;
}
static RcRegistryResult vk_texture_alias_buffer_capture_rows(const VkMemSpanCapture *hold,const VkMemSpan *span,
    uint64_t address,uint64_t bytes,uint64_t row,uint64_t stride,VkTextureAliasRead **alias,VkTextureAliasFailure *failure) {
    assert(hold==&held && hold->held==1 && span->buffer==selection.buffer && !effect && !*alias && failure);
    assert(address==0x9000 && bytes==140 && row==20 && stride==24);
    return RC_REGISTRY_OK;
}
static int vk_texture_alias_read_current(VkTextureAliasRead *alias) { (void)alias; abort(); }
static void vk_texture_alias_read_use(VkTextureAliasRead *alias,uint64_t fence) { (void)alias; (void)fence; abort(); }
static void vk_texture_alias_read_cancel(VkTextureAliasRead *alias) { assert(!alias); }
static VkExecRestoreResult vk_exec_prepare_alias_readback(VkExec *ex,VkTextureAliasRead *const *aliases,uint32_t count,
    int (*current)(void *),void *scope) {
    (void)ex; (void)aliases; (void)count; (void)current; (void)scope; abort();
}
static int vk_texture_alias_buffer_source(VkTextureAliasRead *alias,VkBuffer *buffer,VkDeviceSize *offset) {
    (void)alias; (void)buffer; (void)offset; abort();
}
static int vk_mem_span_record_input(const VkMemSpanCapture *capture,const VkMemSpan *span) {
    if (!vk_mem_span_capture_current(capture,span)) return 0;
    progress=1;
    return 1;
}
#define RESOURCE_IMAGE_SCOPE(name,fence) assert((fence)==6 && effect && progress)
static void vk_gpu_gap_note(int kind,uint64_t pixels) { assert(kind==VK_GAP_COPY_B2T && pixels==15); }
static void vk_mem_count(int cls,int use,uint64_t size) { assert(cls==19 && use==VK_MUSE_COPY_SRC && size==140); }
static void vk_exec_pass_end(VkExec *ex) { assert(ex->rt_resolved && !commands); }
static void vk_exec_barrier(VkExec *ex) { assert(ex->rt_resolved && !commands); }
static const VkTexImage *vk_texture_write_target_begin(VkTextureWriteTarget *target,VkCommandBuffer cb) {
    assert(target==&retained && target->held==1 && cb && effect && !commands);
    return &shape;
}
static void vk_dlaa_hud_write(VkImage image) { assert(image==shape.image); }
static uint64_t vk_exec_recording_fence(void) { return active_scene; }
static void rt_fatal(const char *reason, ...) {
    assert(expect_fatal && reason);
    longjmp(fatal,1);
}
static void *failing_calloc(size_t count,size_t size) { return fail_calloc ? NULL : calloc(count,size); }
static void *grow(void *old,int *capacity,size_t size) {
    assert(!old && !*capacity);
    *capacity=8;
    void *items=malloc(size*(size_t)*capacity);
    assert(items);
    return items;
}
static VkMemInputProvider vk_mem_span_take_input_provider(VkMemSpanCapture **hold) {
    assert(*hold==&held && held.held==1 && effect);
    held.held=2;*hold=NULL;native_holds++;
    return (VkMemInputProvider){&held,9};
}
static void vk_mem_input_provider_release(VkMemInputProvider provider) {
    assert(provider.pool==&held && provider.mirror_incarnation==9 && held.held==2);
    held.held=0;provider_releases++;
}
static void cb_done(void *owner) {
    CbRes *r=owner;
    assert(r==cleanup && r->ninput_providers==1 && r->fence==active_scene);
    vk_mem_input_provider_release(r->input_providers[0]);
    free(r->input_providers);
    free(r);
    cleanup=NULL;
}
static void cb_cancel(void *owner,int may_have_executed) { assert(!may_have_executed); cb_done(owner); }
static void vk_completion_add_owned(uint64_t scene,void (*done)(void *),void (*cancel)(void *,int),void *owner) {
    assert(scene==active_scene && done==cb_done && cancel==cb_cancel && owner && !cleanup);
    cleanup=owner;callbacks++;
}
static void vkCmdCopyBufferToImage(VkCommandBuffer cb,VkBuffer input,VkImage output,VkImageLayout layout,
                                  uint32_t count,const VkBufferImageCopy *r) {
    assert(cb && input==selection.buffer && output==shape.image && layout==VK_IMAGE_LAYOUT_GENERAL && count==1);
    assert(held.held==2 && native_holds==1 && !commands && !writes && effect);
    assert(r->bufferOffset==64 && r->bufferRowLength==6 && r->bufferImageHeight==3);
    assert(r->imageSubresource.aspectMask==VK_IMAGE_ASPECT_COLOR_BIT && r->imageSubresource.mipLevel==1);
    assert(r->imageSubresource.baseArrayLayer==1 && r->imageSubresource.layerCount==2);
    assert(r->imageOffset.x==2 && r->imageOffset.y==3 && r->imageExtent.width==5 && r->imageExtent.height==3);
    commands++;
}
static void vk_texture_write_target_written(VkTextureWriteTarget *target) {
    assert(target==&retained && retained.held==1 && effect && commands==1 && !writes);
    retained.held=0;
    writes++;
}
static void vk_texture_write_target_cancel(VkTextureWriteTarget *target) {
    if (!target) return;
    assert(target==&retained && retained.held==1 && !effect && !commands);
    retained.held=0;
}
'''

CHECKS = r'''
int main(void) {
    BufferImagePitch pitch;
    assert(buffer_image_pitch(5, 3, 2, 1, 1, 4, 24, &pitch) == COPY_PITCH_OK);
    assert(buffer_image_runs(&pitch) == 6 && buffer_image_run_bytes(&pitch) == 20 && pitch.size == 140);
    assert(buffer_image_pitch(5, 3, 2, 1, 1, 4, 20, &pitch) == COPY_PITCH_OK);
    assert(buffer_image_runs(&pitch) == 1 && buffer_image_run_bytes(&pitch) == 120);
    VkExec ex={.copy_stride=24,.rt_resolved=1,.fence=6,.cb=(VkCommandBuffer)(uintptr_t)1};
    p_copy_b2t op={.src=0x9000,.tex=&texture,
        .view={.set_levels=1,.base_level=1,.set_layers=1,.min_layer=1},
        .has_view=1,.r={2,3,0,5,3,2}};
    VkExecBufferToTexture copy={0};
    candidate=1;
    op.tex=&unsupported;
    assert(vk_exec_buffer_to_texture_capture(&ex,&op,&copy)==1 && !copy.target && !queries);
    op.tex=&texture;
    ex.copy_stride=14;
    assert(vk_exec_buffer_to_texture_capture(&ex,&op,&copy)==1 && !copy.target && !queries);
    ex.copy_stride=24;
    assert(vk_exec_buffer_to_texture_capture(&ex,&op,&copy)==1 && !copy.target && queries==1 && !selections);
    backed=1;
    assert(vk_exec_buffer_to_texture_capture(&ex,&op,&copy)<0 && !copy.target && selections==1 && !targets);
    vk_exec_buffer_to_texture_cancel(&copy);
    contact=1;
    assert(vk_exec_buffer_to_texture_capture(&ex,&op,&copy)==0 && selections==1 && !targets);
    contact=0;
    native=1;
    assert(vk_exec_buffer_to_texture_capture(&ex,&op,&copy)==1 && copy.target && selections==2 && targets==1);
    assert(!vk_exec_buffer_to_texture_source_current(&copy) && vk_exec_buffer_to_texture_prepare_source(&ex,&copy)==RC_REGISTRY_STALE);
    vk_exec_buffer_to_texture_cancel(&copy);
    assert(!copy.target && !copy.source_hold && released==1 && !held.held && !retained.held);
    live=1;
    assert(vk_exec_buffer_to_texture_capture(&ex,&op,&copy)==1 && copy.target && selections==3 && targets==2);
    assert(vk_exec_buffer_to_texture_source_current(&copy) && vk_exec_buffer_to_texture_prepare_source(&ex,&copy)==RC_REGISTRY_OK);
    assert(!ex.dc && !default_draw.res); // a transfer-only command buffer with no draw/dispatch setup
    effect=1;
    vk_exec_buffer_to_texture_commit(&ex,&copy);
    assert(!copy.target && !copy.source_hold && ex.recorded==1 && commands==1 && writes==1);
    assert(!retained.held && held.held==2 && native_holds==1 && released==1);
    assert(ex.dc==&default_draw && default_draw.res==cleanup && cleanup->fence==6 && callbacks==1);
    cb_cancel(cleanup,0);
    assert(!held.held && provider_releases==1);
    // The abandoned scene number is never reused. The same context gets a fresh CbRes for a new recording.
    ex.fence=++active_scene;
    held.held=1;
    VkMemSpanCapture *next=&held;
    vk_exec_record_native_input_provider(&ex,&next);
    assert(!next && cleanup->fence==7 && default_draw.prep_incarnation==2 && callbacks==2);
    cb_done(cleanup);

    ex.dc=&existing_draw;
    ex.fence=++active_scene;
    held.held=1;
    next=&held;
    expect_fatal=fail_calloc=1;
    if (!setjmp(fatal)) { vk_exec_record_native_input_provider(&ex,&next); assert(0); }
    assert(ex.dc==&existing_draw && !cleanup && held.held==1 && next==&held && callbacks==2);
    expect_fatal=fail_calloc=0;
    vk_exec_record_native_input_provider(&ex,&next);
    assert(ex.dc==&existing_draw && !next && callbacks==3 && cleanup->fence==8);
    cb_done(cleanup);
    assert(provider_releases==3);
    return 0;
}
'''

types = between(HEADER, "typedef struct {\n    VkTextureWriteTarget *target;\n    VkMemSpan source;", "// Negative is")
region = between(SOURCE, "typedef struct {\n    const VkTexImage *img;", "static int sub_resolve(")
layout = between(SOURCE, "static VkImageSubresourceLayers sub_layers(", "// A VRAM mirror supplies")
transaction = (between(HEADER, "int vk_exec_buffer_to_texture_source_current(", "void vk_exec_buffer_to_texture_commit(") +
    "void vk_exec_buffer_to_texture_cancel(VkExecBufferToTexture *);\n" +
    between(SOURCE, "static int buffer_to_texture_ready(", "int vk_exec_buffer_to_texture_capture_compat(") +
    between(SOURCE, "int vk_exec_buffer_to_texture_capture(", "int vk_exec_restore_copy_capture("))
cb_owner = between(DRAW, "static CbRes *cb_res(", "static const DescriptorPacket *descriptor_packet(")
provider = between(DRAW, "void vk_exec_record_native_input_provider(", "static int graphics_argument_use(")
cc = Path(os.environ["CC"])
env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="ordered-buffer-texture-") as tmp:
    check = Path(tmp) / "check.c"
    binary = check.with_suffix(".exe")
    check.write_text("\n".join((TYPES, types, ADAPTERS, "#define calloc failing_calloc", cb_owner, provider,
                                "#undef calloc", region, layout, transaction, CHECKS)), encoding="utf-8")
    subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fsanitize-trap=all", "-I", str(ROOT / "runtime"), str(check), "-o", str(binary)], env=env, check=True)
    subprocess.run([str(binary)], env=env, check=True)
print("Ordered B2T: format, stride, source cancellation, fresh transfer-only recording and one pitched native write passed")

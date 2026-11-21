"""Check actual BC stand-in admission, cache ownership and native upload retirement."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
from tempfile import TemporaryDirectory

from present_source_check import ROOT, function




def extract(source):
    new = "bc_standin_find(" in source
    data = source[source.index("typedef struct", source.index("// ---- 3D BC") if "// ---- 3D BC" in source else source.index("typedef struct { VkImage bc, img;")):
                  source.index("static void image_barrier")]
    if new:
        data = source[source.rindex("typedef struct {",0,source.index("} BcStandIn;")):source.index("static void image_barrier")]
    return data + "\n".join(function(source, name) for name in (
        *(("bc_standin_find","bc_upload_release") if new else ()), "bc_standin","vk_exec_bc_gone","bc_target"))

BEFORE = r"""
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "vk_exec.h"
#include "resource_core_vma.h"
#define H(type,integer) ((type)(uintptr_t)(integer))
typedef int PlatMutex;
#define PLAT_MUTEX_INIT 0
static void plat_mutex_lock(PlatMutex *lock) { assert(!*lock); *lock = 1; }
static void plat_mutex_unlock(PlatMutex *lock) { assert(*lock); *lock = 0; }
void rt_fatal(const char *format, ...) { (void)format; abort(); }
void rt_log(const char *format, ...) { (void)format; }
static unsigned level_extent(unsigned dims,unsigned mind) { return dims >> mind; }
static void image_barrier(VkCommandBuffer cb,VkImage image,VkImageLayout layout,uint32_t layers) {
    assert(cb == H(VkCommandBuffer,7) && image == H(VkImage,19) && layout == VK_IMAGE_LAYOUT_UNDEFINED && layers == 64);
}
static void memory_barrier(VkCommandBuffer cb,VkPipelineStageFlags2 src,VkAccessFlags2 access,
                           VkPipelineStageFlags2 dst,VkAccessFlags2 visibility) {
    assert(cb == H(VkCommandBuffer,7) && src && access && dst && visibility);
}
static const VkTexImage backing = {.image=H(VkImage,14),.type=VK_IMAGE_TYPE_3D,
                                  .width=64,.height=64,.depth=64,.levels=1};
static const view_t view = {.set_format=1,.format=NVN_FORMAT_RG32UI};
static const tex_t texture = {.serial=88,.d={.width=64,.height=64,.depth=64,.format=NVN_FORMAT_RGTC1_UNORM}};
struct VkTextureTargetLoan { uint32_t entry; };
static VkTextureTargetLoan loan = {.entry=45};
static unsigned outstanding, admitted, raw, copies, borrowed, destroyed, deferred, fault;
static VkResult native_status;
static void (*upload_callback)(void *);
static void *upload_owner;
static int uploading, turn_due, competing_prepare, pending_refused;
static _Thread_local int tex_deferred;
int vk_exec_turn_due(void) { return turn_due; }
static int bc_target(const tex_t *, const view_t *, int, const VkTexImage *, VkExecTarget *);
static VkCtx c = {.vma=H(VmaAllocator,88)};
VkCtx *vk_ctx(void) { return &c; }
VkTextureTargetLoan *vk_texture_target_loan(const tex_t *t,VkImage exact,int materialized) {
    assert(t==&texture && exact==backing.image && materialized);
    assert(!borrowed); borrowed++; return &loan;
}
const VkTexImage *vk_texture_target_loan_image(const VkTextureTargetLoan *t) {
    assert(t==&loan && borrowed==1); return &backing;
}
int vk_texture_target_loan_current(const VkTextureTargetLoan *t) {
    assert(t==&loan && borrowed==1); return fault!=1;
}
void vk_texture_target_loan_release(VkTextureTargetLoan *t) {
    if (!t) return;
    assert(t==&loan && borrowed==1 && !uploading);
    --borrowed;
}
static VkResult actual_make(VmaAllocator allocator,const VkImageCreateInfo *ci,const VmaAllocationCreateInfo *params,
                            VkImage *image,VmaAllocation *allocation) {
    assert(allocator==c.vma && ci->extent.width==16 && ci->extent.height==16 && ci->extent.depth==1 &&
           ci->arrayLayers==64 && ci->format==VK_FORMAT_R32G32_UINT &&
           (ci->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) && params->usage==VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    *image = VK_NULL_HANDLE;
    *allocation = NULL;
    if (native_status != VK_SUCCESS) return native_status;
    if (competing_prepare) {
        competing_prepare=0;
        turn_due=1;
        VkExecTarget pending={0};
        assert(bc_target(&texture,&view,1,&backing,&pending) && !pending.image &&
               pending.bc_status==VK_NOT_READY && tex_deferred && admitted==1);
        pending_refused++;
        turn_due=tex_deferred=0;
    }
    assert(!outstanding);
    outstanding++;
    *image=H(VkImage,19);
    *allocation=H(VmaAllocation,82);
    if (fault==1) vk_exec_bc_gone(backing.image);
    return VK_SUCCESS;
}
VkResult vmaCreateImage(VmaAllocator allocator,const VkImageCreateInfo *ci,const VmaAllocationCreateInfo *params,
                         VkImage *image,VmaAllocation *allocation,VmaAllocationInfo *info) {
    assert(!info); raw++;
    return actual_make(allocator,ci,params,image,allocation);
}
VkResult resource_core_vma_create_image(const RcVmaRequest *r,VmaAllocator allocator,const VkImageCreateInfo *ci,
  const VmaAllocationCreateInfo *params,VkImage *image,VmaAllocation *allocation,VmaAllocationInfo *info,RcAdmissionStatus *why) {
    assert(r->required && r->producer==RC_BUDGET_TEXTURE && !info && !why && borrowed==1);
    admitted++;
    return actual_make(allocator,ci,params,image,allocation);
}
void resource_core_vma_destroy_image(VmaAllocator allocator,VkImage image,VmaAllocation mem) {
    assert(allocator==c.vma && image==H(VkImage,19) && mem==H(VmaAllocation,82) && outstanding==1);
    outstanding--; destroyed++;
}
void vk_defer_image(VkImage image,VmaAllocation mem) {
    if (!image) { assert(!mem); return; }  // an invalidated pending stand-in has no native owner
    assert(image==H(VkImage,19) && mem==H(VmaAllocation,82) && outstanding==1 && !deferred);
    deferred++;
}
static void settle(void) {
    if (upload_callback) {
        upload_callback(upload_owner);
        upload_callback=NULL; upload_owner=NULL;
    }
    if (deferred) {
        --deferred;
        resource_core_vma_destroy_image(c.vma,H(VkImage,19),H(VmaAllocation,82));
    }
}
void vk_device_lost_check(VkResult result,const char *label) { (void)result; (void)label; abort(); }
VkCommandBuffer vk_upload_begin(void) {
    assert(outstanding==1 && !uploading); uploading=1; return H(VkCommandBuffer,7);
}
void vk_upload_end(void) {
    assert(uploading); uploading=0;
    if (fault==2) vk_exec_bc_gone(backing.image);
}
uint64_t vk_upload_recording_sequence(VkCommandBuffer cb) { assert(uploading && cb==H(VkCommandBuffer,7));return 22; }
void vk_upload_defer_release(uint64_t seq,void (*release)(void *),void *owner) {
    assert(uploading && seq==22 && owner==&loan && !upload_callback); upload_callback=release; upload_owner=owner;
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyImage(VkCommandBuffer cb,VkImage source,VkImageLayout sl,VkImage dst,VkImageLayout dl,
                                         uint32_t count,const VkImageCopy *copy) {
    assert(uploading && cb==H(VkCommandBuffer,7) && source==backing.image &&
           sl==VK_IMAGE_LAYOUT_GENERAL && dst==H(VkImage,19) && dl==sl && count==1 && copy->extent.width==64 &&
           copy->extent.height==64 && copy->extent.depth==64 && copy->dstSubresource.layerCount==64);
    copies++;
}
unsigned nvn_fmt_bpb(int format,int *bw,int *bh) {
    assert(format==NVN_FORMAT_RGTC1_UNORM);
    *bw=*bh=4;
    return 8;
}
static int view_format(const tex_t *t,const view_t *v,int blocks,VkFormat *out) {
    assert(t==&texture && v->set_format && blocks);
    *out=VK_FORMAT_R32G32_UINT;
    return 1;
}
enum { W_LEVEL,W_BLOCK_VIEW };
static void warn_once(int kind,const char *fmt, ...) { (void)kind; (void)fmt; }
"""

AFTER = r"""
int main(void) {
    VkExecTarget out = {0};
    turn_due=1;
    assert(bc_target(&texture,&view,1,&backing,&out) && !out.image && out.bc_status==VK_NOT_READY &&
           tex_deferred && !admitted && !raw && !copies && !borrowed && !nbcs);
    turn_due=tex_deferred=0;
    competing_prepare=1;
    out=(VkExecTarget){0};
    assert(bc_target(&texture,&view,1,&backing,&out) && out.image==H(VkImage,19) &&
           out.bc_backing==backing.image && out.width==16 && out.height==16 && out.layers==64);
    assert(admitted==1 && !raw && copies==1 && borrowed==1 && nbcs==1 && !destroyed);
    assert(pending_refused==1);
    VkExecTarget cached = {0};
    turn_due=1;
    assert(bc_target(&texture,&view,1,&backing,&cached) && cached.image==out.image &&
           admitted==1 && copies==1 && borrowed==1 && nbcs==1 && !tex_deferred);
    turn_due=0;
    vk_exec_bc_gone(backing.image);
    assert(!nbcs && deferred==1 && !destroyed && borrowed==1); // queue custody through real upload
    settle();
    assert(!deferred && destroyed==1 && !borrowed && !outstanding);
    out = (VkExecTarget){0};
    native_status=VK_NOT_READY;
    assert(bc_target(&texture,&view,1,&backing,&out) && out.bc_status==VK_NOT_READY &&
           !out.image && !nbcs && !borrowed && !outstanding && destroyed==1 && copies==1 && !tex_deferred);
    native_status=VK_ERROR_OUT_OF_DEVICE_MEMORY;
    assert(bc_target(&texture,&view,1,&backing,&out) && out.bc_status==VK_ERROR_OUT_OF_DEVICE_MEMORY &&
           !out.image && !nbcs && !borrowed && !outstanding && destroyed==1 && copies==1);
    native_status=VK_SUCCESS; fault=1;
    assert(bc_target(&texture,&view,1,&backing,&out) && out.bc_status==VK_ERROR_INITIALIZATION_FAILED &&
           !out.image && !nbcs && !borrowed && !outstanding && destroyed==2 && copies==1); // no recorded upload
    fault=2;
    assert(bc_target(&texture,&view,1,&backing,&out) && out.bc_status==VK_ERROR_INITIALIZATION_FAILED &&
           !out.image && !nbcs && borrowed==1 && outstanding==1 && destroyed==2 && copies==2 && deferred==1);
    settle();
    assert(!deferred && destroyed==3 && !borrowed && !outstanding);
    fault=0; out=(VkExecTarget){0};
    view_t unsupported=view; unsupported.set_format=0;
    assert(bc_target(&texture,&unsupported,1,&backing,&out) && !out.image && !out.bc_status && admitted==5);
    puts("BC stand-in: shared admission, cold/pending turn retry, exact cache hit, native upload lease, known refusal, unissued and recorded retirement passed");
}
"""

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,default=ROOT/"runtime"/"vk_exec.c")
    parser.add_argument("--headers",type=Path,default=ROOT/"runtime")
    a=parser.parse_args()
    src=a.source.read_text()
    texture=a.headers/"vk_texture.c"
    deferred=function(texture.read_text(),"vk_texture_defer_until_turn") if "vk_texture_defer_until_turn(" in src else ""
    cc=Path(os.environ.get("CC") or shutil.which("clang") or "").resolve()
    if not cc.is_file(): raise SystemExit("Clang is required (set CC)")
    env=dict(os.environ,PATH=str(cc.parent)+os.pathsep+os.environ["PATH"])
    with TemporaryDirectory(prefix="bc-standin-") as work:
        c, exe=Path(work)/"check.c",Path(work)/"check.exe"
        c.write_text(BEFORE+deferred+extract(src)+AFTER)
        headers=[]
        if a.headers.resolve()!=(ROOT/"runtime").resolve():
            path=Path(work)/"input-headers.json"
            path.write_text(json.dumps({"version":0,"case-sensitive":False,"use-external-names":False,
                "roots":[{"type":"file","name":(ROOT/"runtime"/name).as_posix(),
                          "external-contents":(a.headers/name).resolve().as_posix()} for name in ("vk.h","vk_exec.h")]}))
            headers=["-ivfsoverlay",str(path)]
        subprocess.run([str(cc),*headers,"-std=gnu11","-O1","-Wall","-Wextra","-Werror","-fsanitize=undefined",
            "-fsanitize-trap=undefined","-I"+str(ROOT/"runtime"),
            "-I"+str(cc.parent.parent/"include/vma"),str(c),"-o",str(exe)],env=env,check=True)
        subprocess.run([str(exe)],env=env,check=True)
    if "bc_standin_find(" in src:
        assert function(src,"graphics_pass_capture").index("return bc_target_refused") < function(src,"graphics_pass_capture").index("VkImage native = peer_materialized")
        assert function(src,"pass_begin").index("bc_target_refused") < function(src,"pass_begin").index("vkCmdBeginRendering(")
        assert function(src,"clear_color").index("bc_target_refused") < function(src,"clear_color").index("warn_once(W_CLEAR_TARGET")
        memory = (ROOT/"runtime/vk_memory.c").read_text()
        assert "case DEAD_IMAGE_VMA: resource_core_vma_destroy_image" in memory
        assert "resource_core_vma_retire(alloc)" in function(memory,"defer_notify")

if __name__=="__main__":
    main()





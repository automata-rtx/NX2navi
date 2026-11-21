"""Exercise captured attachment values and per-draw notifications from production helpers."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pass_end_lock_check import function

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ["CC"])
source = (ROOT / "runtime/vk_exec.c").read_text()
header = (ROOT / "runtime/vk_exec.h").read_text()
end = header.index("} VkExecTargetBinding;") + len("} VkExecTargetBinding;")
binding_type = header[header.rfind("typedef struct {", 0, end):end]
begin = function(source, "pass_begin")
tail = begin[begin.index("    // Retain the actual guest attachments"):begin.index("    vk_exec_frame.passes++;")]
reset = begin[begin.index("    resource_targets_forget(ex);"):].splitlines()[0]
fixture = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define RC_REGISTRY_SINGLE_THREADED 1
#include "resource_core_registry.c"
typedef uint64_t VkImage;
typedef uint32_t VkImageAspectFlags;
enum {VK_IMAGE_ASPECT_COLOR_BIT=1, VK_IMAGE_ASPECT_DEPTH_BIT=2, VK_FORMAT_D16_UNORM=124, VK_FORMAT_D32_SFLOAT=126};
typedef struct {uint32_t aspectMask,baseMipLevel,levelCount,baseArrayLayer,layerCount;} VkImageSubresourceRange;
typedef struct {VkImage image;uint64_t addr;uint32_t format,scale,aspect,levels,layers,preserve_pool_bytes;} VkTexImage;
typedef struct {struct {uint64_t size;} L;} tex_t;
typedef struct {VkImage image;VkTexImage *img;tex_t *tex;VkImage bc_backing;
    uint32_t aspect,format,blocks,level,base_layer,layers,samples,is3d;} VkExecTarget;
typedef struct {const tex_t *pin;VkImage image;} VkPinnedTextureWrite;
typedef struct {VkTexImage image;unsigned owners,uses;} VkTextureAliasRead;
''' + binding_type + r'''
typedef struct {int in_pass;unsigned pass_ncolor,pass_layers;uint64_t cb,fence;VkExecTarget color[8],depth;
 struct {unsigned width;} tonemap_extent;
 VkExecTargetBinding target_bindings[9];uint32_t target_binding_mask,bc_target_mask,bc_written_mask;} VkExec;
typedef struct {VkPinnedTextureWrite pinned;unsigned release_count;} VkTextureTargetLoan;
typedef struct {unsigned release_count;} VkMemoryImageHold;
typedef struct {VkTextureTargetLoan *loan[9];VkMemoryImageHold *standin[8];VkExecTargetBinding bindings[9];} VkExecGraphicsPass;
static const tex_t *old_pin;
static unsigned queries,begin_calls,live_writes,raw_prepares,historic_writes,mapped_writes,seed_writes;
static unsigned pin_retains,pin_uses,pin_prepares,pin_records,pin_releases;
static void (*loan_completed)(void *);
static void (*loan_canceled)(void *,int);
static void *loan_context;
static void vk_completion_add_owned(uint64_t fence,void (*done)(void *),void (*cancel)(void *,int),void *context) {
    assert(fence==5 && !loan_context && done && cancel && context);
    loan_completed=done;loan_canceled=cancel;loan_context=context;
}
static void vk_texture_alias_read_cancel(VkTextureAliasRead *seed) {
    if (seed) { assert(seed->owners); seed->owners--; }
}
static void seed_done(void *opaque) { vk_texture_alias_read_cancel(opaque); }
static void seed_cancel(void *opaque,int executed) { (void)executed; seed_done(opaque); }
static void vk_texture_alias_read_use(VkTextureAliasRead *seed,uint64_t fence) {
    assert(seed && seed->owners && !seed->uses++);
    seed->owners++;
    vk_completion_add_owned(fence,seed_done,seed_cancel,seed);
}
static const VkTexImage *vk_texture_alias_target_replacement(const VkTextureAliasRead *seed) {
    if (seed) assert(seed->image.image==13);
    return NULL; // This case keeps an in-place seeded native image.
}
static int vk_texture_alias_target_load_current(VkTextureAliasRead *seed,VkImage image,const VkImageSubresourceRange *planes) {
    return seed && seed->owners && seed->image.image==image && planes->baseMipLevel==1 && planes->baseArrayLayer==1 &&
        planes->layerCount==1;
}
static int vk_texture_alias_target_prepare_replacement(VkTextureAliasRead *seed,const VkImageSubresourceRange *planes) {
    (void)seed;(void)planes;abort();
}
static void vk_texture_alias_target_record_replacement(VkTextureAliasRead *seed,const VkImageSubresourceRange *planes) {
    (void)seed;(void)planes;abort();
}
static int vk_texture_render_target_load_current(VkImage image,uint64_t address,const VkImageSubresourceRange *planes) {
    (void)image;(void)address;assert(!planes);return 0;
}
static int vk_texture_render_target_loaded_written(VkImage image,uint64_t address,VkTextureAliasRead *seed,
                                                    const VkImageSubresourceRange *planes) {
    assert(image==13&&address==300&&seed&&seed->uses==1&&seed->owners&&planes&&planes->baseMipLevel==1&&
           planes->baseArrayLayer==1&&planes->layerCount==1);seed_writes++;return 1;
}
static void vk_exec_required_graphics_unavailable(VkExec *ex,unsigned op,const char *reason) {
    (void)ex;(void)op;(void)reason;abort();
}
static void vk_texture_target_loan_release(VkTextureTargetLoan *loan) {
    assert(loan&&!loan->pinned.pin&&!loan->release_count);
    loan->release_count++;
}
static void vk_mem_image_release(VkMemoryImageHold *hold) {
    assert(hold && !hold->release_count);
    hold->release_count++;
}
static VkPinnedTextureWrite vk_texture_target_loan_take_pin(VkTextureTargetLoan *loan) {
    VkPinnedTextureWrite pinned=loan->pinned;loan->pinned=(VkPinnedTextureWrite){0};return pinned;
}
static void vk_texture_image_write_prepare(VkImage image,uint64_t addr,uint64_t size){
    assert((image==13&&addr==300)||(image==14&&addr==400));assert(size==64);raw_prepares++;
}
static void resource_image_written(uint64_t image,uint64_t addr){
    assert(image==14&&addr==400);historic_writes++;
}
static int vk_texture_render_target_written(VkImage image,uint64_t addr){
    assert(begin_calls);
    if(image==14){assert(addr==400);return 0;}
    assert(image==13&&addr==300);live_writes++;return 1;
}
static void vk_texture_pinned_write_capture(const tex_t *t,VkImage image,VkPinnedTextureWrite *out){
    if(t!=old_pin)return;
    assert(image==11&&!out->pin);*out=(VkPinnedTextureWrite){t,image};pin_retains++;
}
static void vk_texture_pinned_write_use(const VkPinnedTextureWrite *capture){
    assert(capture->pin==old_pin&&capture->image==11);pin_uses++;
}
static void vk_texture_pinned_write_prepare(const VkPinnedTextureWrite *capture){
    assert(capture->pin==old_pin&&capture->image==11);pin_prepares++;
}
static void vk_texture_pinned_write_recorded(const VkPinnedTextureWrite *capture){
    assert(capture->pin==old_pin&&capture->image==11&&pin_prepares>pin_records);pin_records++;
}
static void vk_texture_pinned_write_release(VkPinnedTextureWrite *capture){
    if(capture->pin){assert(capture->pin==old_pin&&capture->image==11);pin_releases++;}
    *capture=(VkPinnedTextureWrite){0};
}
static int vk_texture_image_binding(uint64_t image,uint64_t addr,uint64_t size,RcRegistryImageBinding *out,
                                    RcRegistryImageWriteExtent **extents,uint32_t *count){
 queries++;assert(image<32);
 *extents=malloc(sizeof **extents);assert(*extents);
 **extents=(RcRegistryImageWriteExtent){{addr,image+100,{image+200,0,size}},{0}};
 *count=1;
 *out=(RcRegistryImageBinding){0};return 0;
}
static void vkCmdBeginRendering(uint64_t cb,const int *ri){assert(cb&&ri);begin_calls++;}
static void resource_effects_image_record_extents(uint64_t image,const RcRegistryImageWriteExtent *extents,uint32_t count){
 assert(image==14 && count==1 && extents && extents->interval.range.size==64);
 assert(extents->interval.mapping==image+100 && extents->interval.range.backing==image+200);
 assert(extents->interval.address==400);
 mapped_writes++;
}
''' + "\n".join(function(source, name) for name in
               ("resource_target_prepare", "resource_target_written", "resource_target_capture",
                "resource_target_pin", "resource_target_pinned", "resource_targets_forget", "attachment_load_planes",
                "resource_attachment_prepare", "resource_attachment_written", "vk_exec_targets_prepare", "vk_exec_targets_written",
                "graphics_target_loan_done",
                "graphics_target_loan_cancel", "graphics_standin_done", "graphics_standin_cancel",
                "graphics_pass_record_target")) + r'''
static void begin_targets(VkExec *ex,int tonemap,VkExecGraphicsPass *admitted){
 int ri=0;
''' + reset + "\n" + tail + r'''
 ex->in_pass=1;
}
int main(void){
 VkTexImage images[4]={{.image=11,.addr=100},{.image=12,.addr=200},
    {.image=13,.addr=300,.aspect=VK_IMAGE_ASPECT_COLOR_BIT,.levels=2,.layers=3},{.image=14,.addr=400}};
 tex_t texture={{64}}, pinned={{64}};
 old_pin=&pinned;
 VkExec ex={.pass_ncolor=4,.cb=1};
 ex.color[0]=(VkExecTarget){.image=11,.img=&images[0],.tex=&pinned};
 ex.color[1]=(VkExecTarget){.image=22,.img=&images[1],.tex=&texture}; // Native stand-in, never the guest image.
 ex.color[2]=(VkExecTarget){.image=23};
 ex.color[3]=(VkExecTarget){.image=14,.img=&images[3],.tex=&texture};
 ex.depth=(VkExecTarget){.image=13,.img=&images[2],.tex=&texture};
 begin_targets(&ex,0,NULL);
 assert(ex.target_binding_mask==0x109&&queries==2&&pin_retains==1&&pin_uses==1);
 assert(!raw_prepares&&!pin_prepares&&!live_writes&&!pin_records&&!historic_writes);
 // Later metadata changes cannot retarget notifications from an already selected/retained attachment.
 images[0]=(VkTexImage){.image=21,.addr=100};texture.L.size=4096;ex.color[0]=(VkExecTarget){.image=21,.img=&images[0],.tex=&texture};
 ex.in_pass=0;vk_exec_targets_prepare(&ex);vk_exec_targets_written(&ex);assert(!raw_prepares&&!pin_records);
 ex.in_pass=1;
 for(unsigned i=0;i<2;i++){
    vk_exec_targets_prepare(&ex);
    assert(raw_prepares==2*(i+1)&&pin_prepares==i+1&&pin_records==i);
    vk_exec_targets_written(&ex);
 }
 assert(queries==2&&pin_records==2&&live_writes==2&&historic_writes==2&&mapped_writes==2);
 assert(pin_uses==1&&!pin_releases);
 // A private tonemap pass resets prior captured guest targets and emits no guest notifications.
 begin_targets(&ex,1,NULL);assert(!ex.target_binding_mask&&pin_releases==1);
 vk_exec_targets_prepare(&ex);vk_exec_targets_written(&ex);
 assert(queries==2&&raw_prepares==4&&pin_records==2&&mapped_writes==2);
 // A prepared pass transfers its selected physical pin to the actual load and scene exactly once.
 ex.fence=5;
 VkExecGraphicsPass pass={0};
 VkTextureTargetLoan selected={.pinned={&pinned,11}};
 pass.loan[0]=&selected;
 pass.bindings[0].binding.image=11;
 begin_targets(&ex,0,&pass);
 assert(ex.target_binding_mask==1&&resource_target_pinned(&ex,0)->image==11);
 assert(!pass.loan[0]&&!pass.bindings[0].binding.image&&queries==2&&pin_uses==2);
 assert(pin_prepares==2&&pin_records==2); // LOAD did not declare or record a guest write.
 assert(loan_context==&selected&&!selected.release_count);
 resource_targets_forget(&ex);
 assert(pin_releases==2&&!selected.release_count);
 loan_completed(loan_context);
 assert(selected.release_count==1);
 loan_context=NULL;
 VkTextureTargetLoan unsent={0};
 pass.loan[3]=&unsent;
 pass.bindings[3].binding.image=14;
 begin_targets(&ex,0,&pass);
 assert(ex.target_binding_mask==8&&queries==2&&!pass.loan[3]&&loan_context==&unsent);
 resource_targets_forget(&ex);
 loan_canceled(loan_context,0);
 assert(unsent.release_count==1);
 loan_context=NULL;
 VkTextureAliasRead seed={.image=images[2],.owners=1};
 ex.pass_layers=1;
 ex.color[2]=(VkExecTarget){.image=13,.img=&seed.image,.tex=&texture,.aspect=VK_IMAGE_ASPECT_COLOR_BIT,
                            .level=1,.base_layer=1,.layers=2,.samples=1};
 pass.bindings[2]=(VkExecTargetBinding){.binding={.image=13},.range={1,300,64},.load_seed=&seed};
 begin_targets(&ex,0,&pass);
 assert(ex.target_binding_mask==4 && seed.owners==2 && !seed_writes && loan_context==&seed);
 assert(ex.target_bindings[2].load_seed==&seed&&!pass.bindings[2].load_seed);
 vk_exec_targets_prepare(&ex);
 assert(ex.target_bindings[2].loaded_planes.layerCount==1&&!seed_writes);
 vk_exec_targets_written(&ex);
 assert(seed_writes==1);
 resource_targets_forget(&ex);
 assert(seed.owners==1);
 loan_completed(loan_context);
 assert(!seed.owners);
 puts("target binding: LOAD transfers the actual loan or seed without a write; issue and retirement preserve exact backing");
}
'''
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="target-binding-", dir=ROOT / "build/test") as tmp:
    path = Path(tmp) / "check.c"
    path.write_text(fixture)
    exe = path.with_suffix(".exe")
    subprocess.run([str(CC), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-Wno-unused-function", "-fsanitize=undefined", "-fsanitize-trap=all",
                    "-I", str(ROOT / "runtime"), str(path), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

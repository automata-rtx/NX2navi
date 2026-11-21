"""Exercise selected attachment pass admission when SDK availability differs from actual hook work."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from present_source_check import ROOT, function, section


def main():
    cc = Path(os.environ.get("CC") or shutil.which("clang") or "").resolve()
    if not cc.is_file():
        raise SystemExit("Clang is required (set CC)")
    path = Path(os.getenv("SEEDED_PASS_CHECK_SOURCE", ROOT / "runtime" / "vk_exec.c"))
    source = path.read_text(encoding="utf-8")
    planning = function(source, "vk_exec_graphics_pass_capture_seeded")
    assert planning.index("capture_attachment_load_seeds(") < planning.index("return graphics_pass_capture(")
    assert "graphics_pass_hooks_active()" not in planning

    begin = function(source, "pass_begin")
    assert begin.index("graphics_pass_selected_hooks") < begin.index("resource_targets_forget")
    begin_order = ("vk_exec_hooks_wait()", "vk_dlaa_pass_prepared(ex, &admitted->observer)", "vkCmdBeginRendering")
    assert sorted(begin.index(name) for name in begin_order) == [begin.index(name) for name in begin_order]
    assert "vk_dlaa_prepare_observer(" not in begin
    cancel = function(source, "vk_exec_graphics_pass_cancel")
    assert cancel.index("vk_dlaa_prepared_observer_cancel") < cancel.index("vk_texture_alias_read_cancel")
    prefix = r'''
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk.h"
#include "vk_exec.h"
#include "vk_dlaa.h"
#define H(type, value) ((type)(uintptr_t)(value))

struct VkTextureAliasRead { VkTexImage image; int refs, replacement; };
struct VkTextureTargetLoan { VkTexImage image; };
const VkTexImage *vk_dlaa_swap_img;
static tex_t receiver, peer_texture;
static struct VkTextureAliasRead seed;
static struct VkTextureTargetLoan loan;
static uint32_t work;
static struct {unsigned idx;} later_chunk = {1}, *rw_chunk;
struct VkDlaaPreparedObserver { VkImage image; unsigned active; };
static struct VkDlaaPreparedObserver observer;
static int observers, observer_admission_failed, observer_scope_changed;
static int sdk=1, contact, opens, loans, stopped, predecessor_dormant=1, predecessor_resumes, peer_resolves;
static VkImage inspected;
static const RcInterval mapped = {0x80004000,3,{4,0x4000,0x2000}};

int vk_contact_strength(void) { return contact; }
int vk_dlaa_available(void) { return sdk; }
uint32_t vk_dlaa_graphics_pass_work(const VkExec *ex,const VkExecGraphicsPassShape *shape) {
    assert(ex->rt.color[0]==&receiver);
    inspected=shape->color[0].image;
    assert(shape->pass_ncolor==(unsigned)ex->rt.n && shape->pass_w==128 && shape->pass_h==64);
    return work;
}
int vk_dlaa_prepare_observer(const VkExec *ex,const VkExecGraphicsPassShape *shape,VkTextureAliasRead *actual_seed,
    VkTextureAliasRead *depth_seed,VkDlaaPreparedObserver **out) {
    assert(!depth_seed);
    assert(work==VK_DLAA_GRAPHICS_PASS_OBSERVE && ex->fence==5 && !observer.active &&
           !shape->in_pass && actual_seed==&seed && shape->color[0].image==seed.image.image && seed.refs==2);
    *out=NULL;
    if(observer_admission_failed) return 0;
    observer=(VkDlaaPreparedObserver){.image=shape->color[0].image,.active=1};
    *out=&observer;
    observers++;
    return 1;
}
int vk_dlaa_prepared_observer_current(const VkExec *ex,const VkExecGraphicsPassShape *shape,
    const VkDlaaPreparedObserver *actual) {
    return ex->fence==5 && !observer_scope_changed && actual==&observer && actual->active &&
           shape->color[0].image==actual->image;
}
void vk_dlaa_prepared_observer_cancel(VkDlaaPreparedObserver **out) {
    if(!*out) return;
    assert(*out==&observer && observer.active && seed.refs==2 && observers>0);
    observer.active=0;
    *out=NULL;
    observers--;
}
int vk_sync_stopping(void) { return stopped; }
volatile int32_t vk_lost;
int vk_exec_recording_failed(void) { return 0; }
int vk_exec_turn_due(void) { return 0; }
int vk_texture_deferred(void) { return 0; }
void rt_fatal(const char *fmt, ...) { fputs(fmt, stderr); abort(); }
void vk_exec_noatt_dims(const VkExec *ex,uint32_t *w,uint32_t *h) { (void)ex; *w=128; *h=64; }
static uint32_t rt_color_count(const p_set_rt *rt) { assert(rt->n==1 || rt->n==2); return (uint32_t)rt->n; }
static void target_shape(const VkExec *ex,VkExecGraphicsPassShape *shape) {
    shape->pass_ncolor=rt_color_count(&ex->rt);
    shape->pass_w=128; shape->pass_h=64; shape->pass_layers=1; shape->pass_samples=1;
}
static void graphics_pass_feedback(VkExecGraphicsPassShape *shape) { shape->fbl_mask=0; }
static void resolve_target_image(const tex_t *tex,const view_t *view,int has_view,int depth,
    const VkTexImage *image,int make_standin,VkExecTarget *target) {
    assert((tex==&receiver || tex==&peer_texture) && view && !has_view && !depth && !make_standin);
    *target=(VkExecTarget){.tex=tex,.img=image,.image=image->image,.width=128,.height=64,.layers=1,.samples=1};
}
static VkResult peer_standin_status;
static void resolve_target(const tex_t *tex,const view_t *view,int has_view,int depth,VkExecTarget *target) {
    if (tex==&receiver) {
        predecessor_dormant=0;
        predecessor_resumes++;
    } else {
        assert(tex==&peer_texture);
        peer_resolves++;
    }
    resolve_target_image(tex,view,has_view,depth,&loan.image,0,target);
    if (tex==&peer_texture && peer_standin_status!=VK_SUCCESS) {
        target->image=VK_NULL_HANDLE;
        target->bc_status=peer_standin_status;
    }
}
static void resolve_rt(VkExec *ex) {
    resolve_target(&receiver,&ex->rt.cview[0],0,0,&ex->color[0]);
    ex->pass_ncolor=1; ex->pass_w=128; ex->pass_h=64; ex->pass_layers=1; ex->pass_samples=1; ex->rt_resolved=1;
}
void vk_exec_pass_end(VkExec *ex) { ex->in_pass=0; }
void vk_exec_pass_begin(VkExec *ex) { resolve_rt(ex); opens++; ex->in_pass=1; }
const VkTexImage *vk_texture_alias_target_replacement(const VkTextureAliasRead *a) {
    return a && a->replacement ? &a->image : NULL;
}
const VkTexImage *vk_texture_alias_target_image(const VkTextureAliasRead *a) {
    assert(a==&seed && a->refs);
    return a->image.image ? &a->image : NULL;
}
int vk_texture_alias_target_current(VkTextureAliasRead *a,VkImage image) {
    return a==&seed && image==a->image.image && (!a->replacement || predecessor_dormant);
}
VkTextureAliasRead *vk_texture_alias_read_retain(VkTextureAliasRead *a) { assert(a==&seed); a->refs++; return a; }
void vk_texture_alias_read_cancel(VkTextureAliasRead *a) { if (a) { assert(a==&seed && a->refs>1); a->refs--; } }
const RcInterval *vk_texture_alias_read_intervals(const VkTextureAliasRead *a,uint32_t *count) {
    assert(a==&seed && a->refs); *count=1; return &mapped;
}
VkTextureTargetLoan *vk_texture_target_loan(const tex_t *tex,VkImage image,int materialized) {
    assert((tex==&receiver || tex==&peer_texture) && (!image || materialized) &&
           (!materialized || image==loan.image.image));
    loans++; return &loan;
}
const VkTexImage *vk_texture_target_loan_image(const VkTextureTargetLoan *a) { assert(a==&loan); return &a->image; }
void vk_texture_target_loan_release(VkTextureTargetLoan *a) { if (a) { assert(a==&loan && loans); loans--; } }
int vk_texture_target_loan_current(const VkTextureTargetLoan *a) { return a==&loan; }
int vk_texture_target_loan_private(const VkTextureTargetLoan *a) { return a==&loan; }
int vk_texture_target_loan_backing_current(const VkTextureTargetLoan *a,const RcRegistryImageWriteExtent *ranges,uint32_t count) {
    return a==&loan && ranges==NULL && count==0;
}
int vk_texture_image_binding(VkImage image,uint64_t address,uint64_t size,RcRegistryImageBinding *result,
    RcRegistryImageWriteExtent **ranges,uint32_t *count) {
    assert(image && address==mapped.address && size==mapped.range.size && result);
    *ranges=NULL; *count=0; return 0;
}
static VkMemoryImageHold *bc_target_hold(const VkExecTarget *target) { (void)target; abort(); }
void vk_mem_image_release(VkMemoryImageHold *hold) { assert(!hold); }
'''
    extracted = source[source.index("struct VkExecGraphicsPass {"):source.index("static int graphics_pass_hooks_active(")]
    extracted += section(source, "enum {\n    LOAD_ATTACHMENT_SEED_IMAGE", "static VkExecGraphicsPassResult graphics_attachment_failure")
    extracted += "\n".join(function(source, name) for name in
        ("graphics_pass_hooks_active", "graphics_pass_selected_hooks", "vk_exec_graphics_pass_cancel",
         "graphics_attachment_failure", "bc_target_refused", "graphics_pass_capture", "vk_exec_graphics_pass_current"))
    cases = r'''
int main(void) {
    VkExec ex={.fence=5,.rt={.n=2,.color={&receiver,&peer_texture}},.cb=H(VkCommandBuffer,1)};
    receiver.L.size=peer_texture.L.size=mapped.range.size;
    seed=(VkTextureAliasRead){.image={.image=H(VkImage,17),.addr=mapped.address},.refs=1,.replacement=1};
    loan=(VkTextureTargetLoan){.image={.image=H(VkImage,19),.addr=mapped.address}};
    VkTextureAliasRead *seeds[9]={&seed};
    VkExecGraphicsPass *selected=NULL;
    VkExecPreparationFailure failure={0};
    assert(graphics_pass_capture(&ex,1,0,seeds,&selected,&failure)==VK_EXEC_GRAPHICS_PASS_READY);
    assert(!opens && !ex.in_pass && selected && !failure.owner && seed.refs==2 && inspected==seed.image.image);
    assert(!predecessor_resumes && predecessor_dormant && peer_resolves==1);
    assert(!selected->loan[0] && selected->loan[1]==&loan &&
           selected->bindings[0].load_seed==&seed && vk_exec_graphics_pass_current(&ex,selected));
    work=VK_DLAA_GRAPHICS_PASS_OBSERVE;
    assert(!vk_exec_graphics_pass_current(&ex,selected));
    vk_exec_graphics_pass_cancel(&selected);
    assert(graphics_pass_capture(&ex,1,0,seeds,&selected,&failure)==VK_EXEC_GRAPHICS_PASS_READY);
    assert(selected->selected_hook_work==VK_DLAA_GRAPHICS_PASS_OBSERVE && selected->observer==&observer &&
           selected->bindings[0].load_seed==&seed && !opens && observers==1 && vk_exec_graphics_pass_current(&ex,selected));
    observer_scope_changed=1;
    assert(!vk_exec_graphics_pass_current(&ex,selected));
    observer_scope_changed=0;
    work=VK_DLAA_GRAPHICS_PASS_HOOK;
    assert(!vk_exec_graphics_pass_current(&ex,selected));
    vk_exec_graphics_pass_cancel(&selected);
    assert(!selected && !observers && seed.refs==1);
    work=VK_DLAA_GRAPHICS_PASS_OBSERVE;
    rw_chunk=&later_chunk;
    failure=(VkExecPreparationFailure){0};
    assert(graphics_pass_capture(&ex,1,0,seeds,&selected,&failure)==VK_EXEC_GRAPHICS_PASS_UNAVAILABLE);
    assert(!selected && !opens && !observers && (unsigned)failure.detail==(LOAD_ATTACHMENT_HOOK_WORK|work));
    rw_chunk=NULL;
    observer_admission_failed=1;
    failure=(VkExecPreparationFailure){0};
    assert(graphics_pass_capture(&ex,1,0,seeds,&selected,&failure)==VK_EXEC_GRAPHICS_PASS_UNAVAILABLE);
    assert(!selected && !opens && !observers && (unsigned)failure.detail==LOAD_ATTACHMENT_HOOK_OBSERVER);
    observer_admission_failed=0;
    // Already open and using the same native attachments: no repeat begin observer.
    ex.in_pass=1;
    ex.pass_ncolor=2; ex.pass_w=128; ex.pass_h=64; ex.pass_layers=1; ex.pass_samples=1;
    resolve_target_image(&receiver,&ex.rt.cview[0],0,0,&seed.image,0,&ex.color[0]);
    resolve_target_image(&peer_texture,&ex.rt.cview[1],0,0,&loan.image,0,&ex.color[1]);
    failure=(VkExecPreparationFailure){0};
    assert(graphics_pass_capture(&ex,1,0,seeds,&selected,&failure)==VK_EXEC_GRAPHICS_PASS_READY);
    assert(!selected->observer && !observers && vk_exec_graphics_pass_current(&ex,selected));
    vk_exec_graphics_pass_cancel(&selected);
    ex.in_pass=0;
    const uint32_t kinds[]={VK_DLAA_GRAPHICS_PASS_HOOK,VK_DLAA_GRAPHICS_PASS_TONEMAP,VK_DLAA_GRAPHICS_PASS_HUD};
    for (unsigned i=0;i<sizeof kinds/sizeof *kinds;i++) {
        work=kinds[i]; failure=(VkExecPreparationFailure){0};
        assert(graphics_pass_capture(&ex,1,0,seeds,&selected,&failure)==VK_EXEC_GRAPHICS_PASS_UNAVAILABLE);
        assert(!selected && seed.refs==1 && !opens && failure.owner==VK_EXEC_PREPARATION_IMAGE &&
               failure.slot==0 && failure.detail==(LOAD_ATTACHMENT_HOOK_WORK | (int)work) && failure.range.mapping==mapped.mapping);
    }
    work=0;
    assert(!predecessor_resumes && predecessor_dormant);
    predecessor_dormant=0;
    failure=(VkExecPreparationFailure){0};
    assert(graphics_pass_capture(&ex,1,0,seeds,&selected,&failure)==VK_EXEC_GRAPHICS_PASS_UNAVAILABLE);
    assert(!selected && !loans && failure.detail==LOAD_ATTACHMENT_SEED_CURRENT);
    seed.replacement=0;
    failure=(VkExecPreparationFailure){0};
    assert(graphics_pass_capture(&ex,1,0,seeds,&selected,&failure)==VK_EXEC_GRAPHICS_PASS_READY);
    assert(!selected->loan[0] && selected->bindings[0].load_seed==&seed);
    vk_exec_graphics_pass_cancel(&selected);
    peer_standin_status=VK_NOT_READY;
    failure=(VkExecPreparationFailure){0};
    assert(graphics_pass_capture(&ex,1,0,seeds,&selected,&failure)==VK_EXEC_GRAPHICS_PASS_UNAVAILABLE);
    assert(!selected && !opens && !loans && seed.refs==1 && failure.detail==VK_NOT_READY &&
           failure.reason==VK_EXEC_PREPARATION_PRESSURE && failure.slot==1);
    peer_standin_status=VK_SUCCESS;
    ex.rt.n=1;
    ex.rt.color[1]=NULL;
    assert(graphics_pass_capture(&ex,1,1,NULL,&selected,NULL)==VK_EXEC_GRAPHICS_PASS_READY);
    assert(!opens && vk_exec_graphics_pass_current(&ex,selected));
    work=VK_DLAA_GRAPHICS_PASS_TONEMAP;
    assert(!vk_exec_graphics_pass_current(&ex,selected));
    vk_exec_graphics_pass_cancel(&selected);
    assert(graphics_pass_capture(&ex,1,1,NULL,&selected,NULL)==VK_EXEC_GRAPHICS_PASS_COMPAT);
    assert(!opens && !loans);
    assert(graphics_pass_capture(&ex,1,0,NULL,&selected,NULL)==VK_EXEC_GRAPHICS_PASS_READY);
    assert(opens==1 && selected->hook_cb==ex.cb && vk_exec_graphics_pass_current(&ex,selected));
    vk_exec_graphics_pass_cancel(&selected);
    assert(!loans && seed.refs==1);
    puts("Selected LOAD pass: exact dormant seed with ordinary peer, in-place seed, real hook refusal, reopening and existing active temporal route passed");
}
'''
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    runtime = path.parent if (path.parent / "vk_exec.h").is_file() else ROOT / "runtime"
    vma = cc.parent.parent / "include" / "vma"
    overlay = os.getenv("SEEDED_PASS_CHECK_CLANG_OVERLAY")
    compiler_scope = ["-ivfsoverlay", overlay] if overlay else []
    if overlay: runtime = ROOT / "runtime"
    with tempfile.TemporaryDirectory(prefix="seeded-pass-hooks-") as tmp:
        c = Path(tmp) / "check.c"
        exe = c.with_suffix(".exe")
        c.write_text(prefix + extracted + cases, encoding="utf-8")
        subprocess.run([str(cc), *compiler_scope, "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=undefined", f"-I{runtime}", f"-I{vma}",
                        str(c), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()

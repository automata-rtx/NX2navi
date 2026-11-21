"""Exercise the production compatibility pass and descriptor decisions with modeled Vulkan operations."""
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DRAW = Path(os.environ.get("COMPAT_DESCRIPTORS_DRAW_SOURCE", ROOT / "runtime/vk_exec_draw.c"))
REPLAY = ROOT / "runtime/vk_exec.c"


def section(source, start, end):
    at = source.index(start)
    return source[at:source.index(end, at + len(start))]


def block(source, start):
    begin = source.index(start)
    opening = source.index("{", begin)
    level = 1
    at = opening + 1
    while level:
        level += (source[at] == "{") - (source[at] == "}")
        at += 1
    return source[begin:at]


PREFIX = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef enum {VK_EXEC_GRAPHICS_READY, VK_EXEC_GRAPHICS_COMPAT, VK_EXEC_GRAPHICS_INVALIDATED, VK_EXEC_GRAPHICS_UNAVAILABLE,
              VK_EXEC_GRAPHICS_NEEDS_ORDERED, VK_EXEC_GRAPHICS_STOPPED, VK_EXEC_GRAPHICS_NOOP} VkExecGraphicsResult;
typedef enum {VK_EXEC_PREPARATION_PIPELINE, VK_EXEC_PREPARATION_POOL_IMAGE, VK_EXEC_PREPARATION_DESCRIPTOR,
              VK_EXEC_PREPARATION_SCRATCH} VkExecPreparationOwner;
typedef struct { int n; } GfxShader;
typedef struct {const GfxShader *vs, *fs, *ts; unsigned format, native_rmw_marker;} VkPipeState;
typedef struct {unsigned layout, pool;} VkPipeInfo;
typedef struct {const VkPipeInfo *info; unsigned revision, pipeline, vertex_uniforms;} VkGraphicsPipelineUse;
typedef struct {const VkPipeInfo *info; unsigned ordinal, cell;} DrawDescriptorGrant;
typedef struct {unsigned width, height;} VkExtent2D;
typedef struct {VkExtent2D extent;} VkRect2D;
typedef struct {
    unsigned in_pass, pass_noatt, pass_w, pass_h, requested, dirty, rt_resolved;
    unsigned fbl_mask, hud_alpha_attachment, attachment, loaded, load_ready, resource_op, resource_next;
    VkExtent2D tonemap_extent;
} VkExec;
typedef struct {
    VkPipeState state;
    VkGraphicsPipelineUse use;
    DrawDescriptorGrant descriptors;
    unsigned selected_writers, pointer_count, pointer_outputs_ready, pool_grant, scratch;
    unsigned compatibility, activated, opcode;
    struct { void *streams; } vertices;
    struct { unsigned owner; const char *stage; } failure;
} VkExecGraphicsDraw;
typedef struct {unsigned attempted; void *pending; VkExecGraphicsResult result; int saturated;} GraphicsCompatibilityPool;
typedef struct { unsigned sampled; } GraphicsFeedback;
typedef void VkExecPoolRequirement;
typedef unsigned RcRegistryPosition;
enum { VK_EXEC_DIRTY_VIEWPORT=1, VK_EXEC_DIRTY_SCISSOR=2, SK_SCISSOR=3,
       COMPAT_ISSUE_PIPELINE=1, COMPAT_ISSUE_POOL=2, COMPAT_ISSUE_SCRATCH=4 };
#define RWT_STAGE_END(stage) ((void)0)

static GfxShader vs={1}, fs={2};
static VkPipeInfo first_layout={11,0}, second_layout={12,0};
static VkPipeInfo *available_layout;
static unsigned available_revision, opens, closes, loads, draw_emits, allocations, returns, pipeline_releases;
static int refusal, loss, deferred, provide_footprint, scratch_prepared;

static void pass_begin(VkExec *ex, void *prepared, unsigned opcode) {
    assert(!prepared && !opcode);
    if (ex->in_pass) return;
    // Model the native LOAD's current content receipt independently from the draw and its descriptor cells.
    if (ex->attachment && !ex->load_ready) { loss=1; return; }
    ex->loaded=ex->attachment;
    if (ex->attachment) ++loads;
    ex->in_pass=1;
    ex->pass_w=ex->pass_h=ex->requested;
    ++opens;
}
static void vk_exec_pass_end(VkExec *ex) {
    assert(ex->in_pass);
    ex->in_pass=0;
    ++closes;
}
static void vk_exec_noatt_dims(VkExec *ex, uint32_t *w, uint32_t *h) { *w=*h=ex->requested; }
static VkRect2D vk_exec_scissor(VkExec *ex) { return (VkRect2D){.extent={ex->pass_w,ex->pass_h}}; }
static int draw_skipped(VkExec *ex,int mesh,int cause) { (void)mesh; assert(ex && cause==SK_SCISSOR); return 0; }
static int vk_texture_deferred(void) { return deferred; }
static int vk_exec_recording_failed(void) { return loss; }
static VkExecGraphicsResult graphics_compat_pool_prepare_members(VkExec *ex, VkExecGraphicsDraw *draw,
    const GfxShader *const shaders[3], const GraphicsCompatibilityPool *prepare) {
    assert(ex && draw->compatibility && shaders[0] == &vs && shaders[1] == &fs && !shaders[2] && prepare);
    return VK_EXEC_GRAPHICS_READY;
}
static void vk_exec_pool_grant_release(unsigned *grant) { *grant=0; }
static void vk_pipeline_graphics_release(VkGraphicsPipelineUse *use) {
    assert(use->revision);
    *use=(VkGraphicsPipelineUse){0};
    ++pipeline_releases;
}
static VkExecGraphicsResult graphics_pipeline_ready(VkExec *ex,VkExecGraphicsDraw *draw,int ordered) {
    assert(ex->in_pass && ordered);
    if (!draw->use.revision)
        draw->use=(VkGraphicsPipelineUse){available_layout, available_revision, available_revision+100, provide_footprint};
    return VK_EXEC_GRAPHICS_READY;
}
static VkExecGraphicsResult graphics_storage_pipeline_ready(VkExec *ex,VkExecGraphicsDraw *draw,int ordered) {
    assert(!draw->selected_writers);
    return graphics_pipeline_ready(ex,draw,ordered);
}
static VkExecGraphicsResult graphics_storage_pointer_outputs_ready(VkExec *ex,VkExecGraphicsDraw *draw) {
    (void)ex;(void)draw; assert(0); return VK_EXEC_GRAPHICS_UNAVAILABLE;
}
static VkExecGraphicsResult graphics_pool_prepare(VkExec *ex,VkExecGraphicsDraw *draw,int ordered,
    int preparing,unsigned attempted,void *pending,int fallback) {
    assert(ex->in_pass && draw->use.info && ordered && !pending && !attempted && !fallback);
    (void)preparing;
    return VK_EXEC_GRAPHICS_READY;
}
static int graphics_load_pool_disjoint(VkExec *ex,VkExecGraphicsDraw *draw,int replacement) {
    assert(ex && draw && !replacement && !draw->pool_grant);
    return 1;
}
static int draw_descriptor_grant_current(const VkExec *ex,const VkPipeInfo *info,const DrawDescriptorGrant *g,unsigned op) {
    assert(ex->resource_op == op);
    return g->info==info && g->ordinal==op && g->cell;
}
static void retire_grant(DrawDescriptorGrant *g) {
    if(g->cell) ++returns;
    *g=(DrawDescriptorGrant){0};
}
static VkExecGraphicsResult graphics_descriptors_ready(VkExec *ex,VkExecGraphicsDraw *draw,int ordered) {
    assert(ordered && draw->use.info && ex->in_pass);
    retire_grant(&draw->descriptors);
    if (refusal) return VK_EXEC_GRAPHICS_UNAVAILABLE;
    draw->descriptors=(DrawDescriptorGrant){draw->use.info,ex->resource_op,++allocations};
    return VK_EXEC_GRAPHICS_READY;
}
static VkExecGraphicsResult graphics_buffers_prepare(VkExec *ex,VkExecGraphicsDraw *draw,int ordered) {
    assert(ordered && draw_descriptor_grant_current(ex,draw->use.info,&draw->descriptors,ex->resource_op));
    return VK_EXEC_GRAPHICS_READY;
}
static int graphics_vertices_current(const VkExec *ex,const VkExecGraphicsDraw *draw) {
    assert(ex && draw);return 1;
}
static int graphics_vertices_record(VkExec *ex,VkExecGraphicsDraw *draw) {
    assert(ex && draw);return 1;
}
static VkExecGraphicsResult graphics_marked_manifest(VkExecGraphicsDraw *draw,VkExecGraphicsResult result,const char *why) {
    assert(draw && !strcmp(why,"manifest-compat-bindings"));
    return result;
}
static VkExecGraphicsResult graphics_scratch_ready(VkExec *ex,VkExecGraphicsDraw *draw,int ordered) {
    assert(ex && ordered && draw_descriptor_grant_current(ex, draw->use.info, &draw->descriptors,ex->resource_op));
    if (draw->use.vertex_uniforms && ex->resource_next < ex->resource_op) ++scratch_prepared;
    return VK_EXEC_GRAPHICS_READY;
}
static void graphics_compat_issue_unavailable(VkExec *ex,VkExecGraphicsDraw *draw,VkExecGraphicsResult result,
    VkExecPreparationOwner owner,const char *gate,unsigned prior) {
    assert(ex->in_pass && draw->compatibility && result == VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(owner == VK_EXEC_PREPARATION_DESCRIPTOR && !strcmp(gate,"issue-descriptors") && prior);
    loss=1;
}
static int graphics_compat_arguments_current(VkExec *ex,VkExecGraphicsDraw *draw) {
    assert(ex->resource_next == ex->resource_op && draw->compatibility);
    return 1;
}
static int ordinary_storage_writes_record(VkExec *ex,const VkPipeInfo *info,const void *pass,int graphics) { assert(ex && info && !pass && graphics); return 1; }
static int graphics_native_rmw_order_for_draw(VkExec *ex,VkExecGraphicsDraw *draw) { assert(ex && !draw->selected_writers);return 1; }
static void graphics_marked_runtime_failure(VkExecGraphicsDraw *draw,const char *why) { (void)draw;(void)why; assert(0); }
static void vk_exec_required_graphics_unavailable(VkExec *ex,unsigned opcode,const char *why) {
    assert(ex && opcode && why); loss=1;
}
static int graphics_ubo_merge_has_native(unsigned s) { return s != 0; }
static int graphics_ubo_merge_record_preserving_pass(VkExec *ex,unsigned s) { (void)ex;(void)s;assert(0);return 0; }
static int flat3d_preflight(VkExec *ex,const GfxShader **shaders,const int *stages,int n) {
    (void)ex;(void)shaders;(void)stages;(void)n;assert(0);return 0;
}
static int graphics_compat_mesh_ring_entry(VkExec *ex,VkExecGraphicsDraw *draw) { (void)ex;(void)draw;assert(0);return 0; }
static void feedback_apply(VkExec *ex,const GraphicsFeedback *feedback,int ordered) {
    assert(ex->in_pass && !feedback->sampled && ordered);
}
static void vk_dlaa_hud_incomplete(void) { assert(0); }
static int graphics_record_state(VkExec *ex,const VkPipeState *state,unsigned pipeline,const VkPipeInfo *info,
                      unsigned fbl,int mesh,int fb,int direct_mesh,VkExecPoolRequirement *requirement,VkExecGraphicsDraw *draw) {
    assert(ex->in_pass && ex->loaded == ex->attachment && state == &draw->state && !fbl && !mesh && !fb &&
           !direct_mesh && !requirement && pipeline==draw->use.pipeline && draw->activated);
    assert(draw_descriptor_grant_current(ex, info, &draw->descriptors, ex->resource_op));
    draw->descriptors.cell=0;
    ++draw_emits;
    return 1;
}
'''

CHECK = r'''
static void reset(void) {
    available_layout=&first_layout;
    available_revision=1;
    opens=closes=loads=draw_emits=allocations=returns=pipeline_releases=0;
    refusal=loss=deferred=0;
}

static int prepare_compat(VkExec *ex,VkExecGraphicsDraw *draw,VkExecGraphicsResult *result) {
    GraphicsCompatibilityPool prepare={.result=VK_EXEC_GRAPHICS_COMPAT};
    int r=compatibility_control(ex,draw,&prepare);
    *result=prepare.result;
    return r;
}

int main(void) {
    VkExecGraphicsResult status;
    reset();
    VkExec ex={.attachment=7,.load_ready=1,.requested=64,.resource_next=8,.resource_op=9};
    VkExecGraphicsDraw draw={.compatibility=1,.opcode=26};
    assert(!prepare_compat(&ex,&draw,&status) && status==VK_EXEC_GRAPHICS_COMPAT);
    assert(ex.in_pass && opens==1 && loads==1 && allocations==1 && !draw_emits && !draw.activated);
    available_revision=2; // An async revision cannot replace a retained pipeline with the same state.
    ex.resource_next=ex.resource_op;
    assert(compatibility_control(&ex,&draw,NULL));
    assert(opens==1&&loads==1&&allocations==1&&draw_emits==1&&draw.use.revision==1&&!pipeline_releases);

    reset();
    ex=(VkExec){.attachment=8,.load_ready=1,.requested=64,.resource_next=8,.resource_op=9};
    draw=(VkExecGraphicsDraw){.compatibility=1,.opcode=26};
    assert(!prepare_compat(&ex,&draw,&status) && status==VK_EXEC_GRAPHICS_COMPAT);
    // A real target change between phases closes the previous pass and changes the required pipeline state.
    vk_exec_pass_end(&ex);
    ex.requested=128;
    ex.attachment=9;
    available_layout=&second_layout;
    available_revision=2;
    ex.resource_next=ex.resource_op;
    assert(compatibility_control(&ex,&draw,NULL));
    assert(opens==2&&closes==1&&loads==2&&allocations==2&&returns==1&&pipeline_releases==1&&draw_emits==1);

    reset();
    ex=(VkExec){.attachment=7,.load_ready=1,.requested=64,.resource_next=8,.resource_op=9};
    draw=(VkExecGraphicsDraw){.compatibility=1,.opcode=26};
    refusal=1;
    assert(!prepare_compat(&ex,&draw,&status) && status==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(ex.in_pass&&opens==1&&loads==1&&!draw.activated&&!draw_emits&&!allocations&&!loss);
    vk_pipeline_graphics_release(&draw.use);
    retire_grant(&draw.descriptors);
    assert(!returns);

    // The recording owner can release an unissued placement while the open attachment LOAD remains intact.
    refusal=0;
    assert(!prepare_compat(&ex,&draw,&status) && status==VK_EXEC_GRAPHICS_COMPAT);
    retire_grant(&draw.descriptors);
    vk_pipeline_graphics_release(&draw.use);
    assert(returns==1&&allocations==1&&opens==1&&loads==1&&!draw_emits);
    draw=(VkExecGraphicsDraw){.compatibility=1,.opcode=26};
    assert(!prepare_compat(&ex,&draw,&status) && status==VK_EXEC_GRAPHICS_COMPAT);
    ex.resource_next=ex.resource_op;
    assert(compatibility_control(&ex,&draw,NULL));
    assert(opens==1&&loads==1&&draw_emits==1);

    reset();
    provide_footprint=1;
    ex=(VkExec){.attachment=7,.load_ready=1,.requested=64,.resource_next=8,.resource_op=9};
    draw=(VkExecGraphicsDraw){.compatibility=1,.opcode=26};
    assert(!prepare_compat(&ex,&draw,&status) && status==VK_EXEC_GRAPHICS_COMPAT);
    assert(draw.use.vertex_uniforms && scratch_prepared==1 && !draw.activated && !draw_emits);
    ex.resource_next=ex.resource_op;
    assert(compatibility_control(&ex,&draw,NULL) && scratch_prepared==1);
    puts("Compatibility graphics: original LOAD, retained revision, changed layout, descriptor refusal and cancellation passed");
    return 0;
}
'''


def main():
    draw = DRAW.read_text(encoding="utf-8")
    replay = REPLAY.read_text(encoding="utf-8")
    opening = block(replay, "void vk_exec_pass_begin(VkExec *ex)")
    scope = section(draw, "static int prepare_draw_phase(VkExec *ex,", "static int prepare_draw(")
    preamble = section(scope, "    int early_indices = ", "    if (!ex->prog)")
    early = block(scope, "    if (early_indices)")
    deferred = section(scope, "    if (!ex->in_pass) vk_exec_pass_begin(ex);", "    RWT_STAGE_END(work_shader)")
    selection_start = scope.index("    if (draw) {\n        unsigned prior")
    selection = block(scope[selection_start:], "    if (draw)")
    fixture = PREFIX + opening + r'''
static int compatibility_control(VkExec *ex,VkExecGraphicsDraw *draw,GraphicsCompatibilityPool *prepare) {
    const GfxShader *vs=&(GfxShader){1}, *fs=&(GfxShader){2}, *ts=NULL;
    const GfxShader *flat_shaders[]={vs,fs,ts};
    const int flat_stages[]={0,1,2};
    int mesh=0,direct_mesh=0,flat_needed=0,indirect_mesh=0,hud_incomplete=0;
    unsigned fbl=0;
    GraphicsFeedback feedback={0};
    VkExecPoolRequirement *requirement=NULL;
''' + preamble + early + deferred + r'''
    if (vk_exec_recording_failed()) return 0;
    VkPipeState s={.vs=vs,.fs=fs,.ts=ts,.format=ex->attachment};
''' + selection + "\n    return 0;\n}\n" + CHECK
    # Shader identity must remain stable across both invocations.
    fixture = fixture.replace("static GfxShader vs={1}, fs={2};", "static GfxShader test_vs={1}, test_fs={2};")
    fixture = fixture.replace("shaders[0] == &vs && shaders[1] == &fs", "shaders[0] == &test_vs && shaders[1] == &test_fs")
    fixture = fixture.replace("const GfxShader *vs=&(GfxShader){1}, *fs=&(GfxShader){2}, *ts=NULL;",
                              "const GfxShader *vs=&test_vs, *fs=&test_fs, *ts=NULL;")
    cc = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ["PATH"])
    with tempfile.TemporaryDirectory(prefix="compat-graphics-descriptors-") as temp:
        c = Path(temp) / "check.c"
        exe = c.with_suffix(".exe")
        c.write_text(fixture,encoding="utf-8")
        subprocess.run([str(cc),"-std=gnu11","-Wall","-Wextra","-Werror","-O1","-fsanitize=undefined",
                        "-fsanitize-trap=all",str(c),"-o",str(exe)],env=env,check=True)
        subprocess.run([str(exe)],env=env,check=True)


if __name__ == "__main__":
    main()

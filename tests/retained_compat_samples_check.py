"""Check the ordinary sample handoff into compatibility graphics."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

from bind_recipe_check import function

PREFIX = r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
typedef uint64_t VkImage;
enum {NVN_STAGE_VS, NVN_STAGE_MESH, NVN_STAGE_TASK, NVN_STAGE_FS};
typedef struct {int live;} VkTexturePresentSource;
typedef struct {uint64_t scene,ordinal;} RcRegistryPosition;
typedef struct {int unused;} GfxShader;
typedef struct {uint16_t type; int pool,slot;} GfxShaderRes;
typedef struct {VkImage image; uint32_t levels;} VkTexImage;
typedef struct {uint32_t base;} view_t;
typedef struct {VkImage image;uint32_t level0,levels,layer0,layers;} ViewKey;
typedef struct {VkImage image;uint32_t l0,l1,a0,a1;} SubRange;
typedef struct {const VkTexImage *image;view_t view;int has_view,historical;} VkTextureSample;
typedef struct {unsigned used;} VkTextureAliasRead;
typedef struct {
   const GfxShader *shader;
   const GfxShaderRes *resource;
   uint64_t handle;
   VkTextureSample sample;
   VkTexImage image;
   VkTextureAliasRead *alias;
   VkTexturePresentSource *hold;
   int stage, member;
   unsigned version;
#if TEST_QUERY_GEOMETRY
   struct {uint64_t registration;} geometry;
#endif
} GraphicsSample;
typedef struct {
    int owner;
    struct { uint8_t present, attachment_seed; } sampled;
} VkExecPreparationFailure;
typedef struct {
   RcRegistryPosition read_position;
   VkExecPreparationFailure failure;
   void *attachment_seeds[9];
   uint32_t opcode;
   int feedback_selected,compatibility,mesh;
#if TEST_QUERY_GEOMETRY
   int use, geometry_present, geometry_recording_attested;
#endif
   uint32_t sample_count;
   GraphicsSample *samples;
} VkExecGraphicsDraw;
typedef struct {uint64_t fence;VkImage attachment;uint32_t level;} VkExec;
static VkExecGraphicsDraw *graphics_current;
static RcRegistryPosition read_position={44,9};
#if TEST_QUERY_GEOMETRY
// Compatibility only consumes ordinary receipts in this fixture.
static int graphics_geometry_final_binding(const int *use, const GraphicsSample *sample) {
    (void)use; (void)sample;
    abort();
}
static int graphics_geometry_peers_current(const VkExec *ex, const VkExecGraphicsDraw *draw,
    const GraphicsSample *sample, int active) {
    (void)ex; (void)draw; (void)sample; (void)active;
    abort();
}
static void graphics_sample_release(GraphicsSample *sample) {
    (void)sample;
    abort();
}
#endif
static unsigned sample_version=3, lock_count, refusals, releases, sample_releases, alias_releases;
static int rc_registry_position_compare(RcRegistryPosition a,RcRegistryPosition b) {
    return a.scene!=b.scene || a.ordinal!=b.ordinal;
}
static RcRegistryPosition vk_exec_resource_position(void){return read_position;}
static void vk_texture_lock(void){assert(!lock_count++);}
static void vk_texture_unlock(void){assert(lock_count--==1);}
static int graphics_sample_shape_current(const GraphicsSample *sample) {
    assert(lock_count==1);
    return sample->version==sample_version;
}
static int texture_view_key(const VkTexImage *image,const view_t *view,int has_view,
                            uint16_t type,int storage,ViewKey *key) {
    assert(lock_count&&image&&type==3&&!storage);
    *key=(ViewKey){image->image,has_view?view->base:0,image->levels,0,1};
    return 1;
}
static uint32_t attached_as(const VkExec *ex,const SubRange *range) {
    return ex->attachment==range->image && ex->level>=range->l0 && ex->level<range->l1;
}
static void vk_exec_required_graphics_unavailable(VkExec *ex,uint32_t opcode,const char *reason) {
    assert(ex->fence==44&&opcode==26&&reason);
    refusals++;
}
static int retained_original;
static int retained_compat_sample_current(const VkExec *ex,const GraphicsSample *sample) {
    assert(ex && sample && lock_count);
    return retained_original && sample->hold && !sample->alias;
}
static void retained_sample_refusal(VkExec *ex,const GraphicsSample *sample,RcRegistryPosition read,
    uint32_t opcode,int stage,const char *reason) {
    assert(sample && read.scene==44 && stage==NVN_STAGE_FS);
    vk_exec_required_graphics_unavailable(ex,opcode,reason);
}
static void vk_exec_graphics_draw_release_gpa_inputs(VkExecGraphicsDraw *draw) {
    releases++;
    for(uint32_t i=0;i<draw->sample_count;i++) {
        if(draw->samples[i].alias) alias_releases++;
        sample_releases++;
    }
    free(draw->samples);
    draw->samples=NULL;
    draw->sample_count=0;
    draw->feedback_selected=0;
}
static void graphics_release_gpa_inputs(VkExecGraphicsDraw *draw,int keep_arguments) {
    assert(keep_arguments);
    vk_exec_graphics_draw_release_gpa_inputs(draw);
}
'''

MAIN = r'''
int main(void) {
    const GfxShader fragment={1}, changed_shader={2};
    const GfxShaderRes color={.type=3,.slot=8}, absent={.type=3,.slot=9}, alias_res={.type=3,.slot=10};
    VkTextureAliasRead alias;
    VkExec ex={44,81,3};
    VkExecGraphicsDraw draw={.read_position=read_position,.failure={.owner=7},.opcode=26,.feedback_selected=1,
                            .sample_count=3,.samples=calloc(3,sizeof *draw.samples)};
    assert(draw.samples);
    graphics_current=&draw;
    draw.samples[0]=(GraphicsSample){.shader=&fragment,.stage=2,.resource=&color,.handle=17,
        .image={81,1},.sample={.view={3},.has_view=1},.version=3};
    draw.samples[1]=(GraphicsSample){.shader=&fragment,.stage=2,.resource=&absent,.handle=18,
        .sample={.historical=-1},.version=3};
    draw.samples[2]=(GraphicsSample){.shader=&fragment,.stage=2,.resource=&alias_res,.handle=19,
        .alias=&alias,.image={82,1},.version=3};
    alias=(VkTextureAliasRead){0};
    draw.attachment_seeds[1]=&ex;
    assert(!graphics_compatibility_begin(&draw));
    assert(draw.sample_count==3&&!draw.compatibility&&!releases&&!sample_releases&&draw.failure.owner==7);
    draw.attachment_seeds[1]=NULL;
    assert(graphics_compatibility_begin(&draw));
    assert(draw.compatibility&&graphics_samples_active()&&!draw.feedback_selected&&draw.sample_count==3&&
           releases==1&&!sample_releases&&!draw.failure.owner);
    SubRange range={0};
    vk_texture_lock();
    assert(graphics_prepared_feedback(&ex,&fragment,&color,0,17,&range)&&attached_as(&ex,&range));
    ex.level=2; // A disjoint mip selected by the same sampler.
    assert(graphics_prepared_feedback(&ex,&fragment,&color,0,17,&range)&&!attached_as(&ex,&range));
    ex.attachment=82; // Independent alias and an absent member cannot create feedback.
    range=(SubRange){0};
    assert(graphics_prepared_feedback(&ex,&fragment,&alias_res,0,19,&range)&&!attached_as(&ex,&range));
    assert(graphics_prepared_feedback(&ex,&fragment,&absent,0,18,&range)&&!attached_as(&ex,&range));
    assert(!graphics_prepared_feedback(&ex,&changed_shader,&color,0,17,&range)&&!refusals);
    assert(!graphics_prepared_feedback(&ex,&fragment,&color,1,17,&range)&&!refusals); // uncovered neighbor
    vk_texture_unlock();
    VkTextureSample bound;
    VkTextureAliasRead *used=NULL;
    const VkTexturePresentSource *owner=NULL;
    vk_texture_lock();
    assert(graphics_sample(&ex,NVN_STAGE_FS,0,&color,17,&bound,&used,&owner)==1&&bound.image==&draw.samples[0].image&&!used);
    assert(graphics_sample(&ex,NVN_STAGE_FS,0,&absent,18,&bound,&used,&owner)==1&&!bound.image&&bound.historical==-1&&!used);
    assert(graphics_sample(&ex,NVN_STAGE_FS,0,&alias_res,19,&bound,&used,&owner)==1&&used==&alias&&!alias.used);
    assert(graphics_sample(&ex,NVN_STAGE_FS,0,&color,99,&bound,&used,&owner)==0); // Different selected material is a different request.
    assert(!graphics_sample(&ex,NVN_STAGE_VS,0,&color,17,&bound,&used,&owner));
    assert(!graphics_sample(&ex,NVN_STAGE_FS,1,&color,17,&bound,&used,&owner));
    read_position.ordinal++;
    assert(graphics_sample(&ex,NVN_STAGE_FS,0,&color,17,&bound,&used,&owner)==-1&&refusals==1);
    vk_texture_unlock();
    read_position=draw.read_position;
    sample_version++;
    vk_texture_lock();
    VkTexturePresentSource original={1};
    draw.samples[0].hold=&original;
    retained_original=1; // Original native bytes survive the old logical version.
    assert(graphics_sample(&ex,NVN_STAGE_FS,0,&color,17,&bound,&used,&owner)==1&&owner==&original);
    retained_original=0;
    assert(graphics_prepared_feedback(&ex,&fragment,&color,0,17,&range)&&refusals==2);
    vk_texture_unlock();
    vk_exec_graphics_draw_release_gpa_inputs(&draw);
    vk_exec_graphics_draw_release_gpa_inputs(&draw);
    assert(sample_releases==3&&alias_releases==1&&!draw.samples);

    draw=(VkExecGraphicsDraw){.read_position=read_position,.opcode=26,.sample_count=1,
                              .samples=calloc(1,sizeof *draw.samples)};
    assert(draw.samples);
    // The unrelated compatibility route retains a qualified prefix and leaves unselected siblings uncovered.
    assert(graphics_compatibility_begin(&draw));
    assert(draw.compatibility&&draw.sample_count==1&&graphics_samples_active()&&sample_releases==3);
    vk_exec_graphics_draw_release_gpa_inputs(&draw);
    assert(sample_releases==4&&alias_releases==1);
    puts("Compatibility graphics: selected image, exact feedback, absence, failure and retirement passed");
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draw-source", type=Path,
                        default=Path(__file__).resolve().parents[1] / "runtime/vk_exec_draw.c")
    args = parser.parse_args()
    source = args.draw_source.read_text(encoding="utf-8")
    geometry = "static void graphics_drop_geometry_samples(" in source
    sections = ["graphics_samples_active", "graphics_prepared_sample_current",
                "graphics_prepared_feedback", "graphics_sample"]
    if geometry:
        sections.append("graphics_drop_geometry_samples")
    sections.append("graphics_compatibility_begin")
    def definition(name):
        declaration = re.search(r"^[a-zA-Z_0-9 *]+[ *]" + name + r"\([^;{]*\)\s*\{", source, re.M)
        assert declaration, name
        return function(source[declaration.start():], name)

    code = f"#define TEST_QUERY_GEOMETRY {int(geometry)}\n" + PREFIX + \
        "\n".join(definition(name) for name in sections) + MAIN
    assert "graphics_prepared_feedback(ex, fs, &res[r], k, h, &selected)" in function(source, "feedback_scan")
    assert "graphics_compatibility_begin(draw)" in function(source, "vk_exec_graphics_draw_prepare")
    cc = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="compat-samples-") as directory:
        c = Path(directory) / "check.c"
        exe = c.with_suffix(".exe")
        c.write_text(code, encoding="utf-8")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=undefined", str(c), "-o", str(exe)],
                       env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True, timeout=20)


if __name__ == "__main__":
    main()


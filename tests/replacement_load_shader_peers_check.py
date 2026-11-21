"""Run the replacement LOAD stages with selected shader outputs and controlled native owners."""
import os
from pathlib import Path
import subprocess
import tempfile

from ordered_graphics_feedback_check import function

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"

STUBS = r'''
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "resource_command_effects.h"

typedef uint64_t VkImage;
typedef struct VkExec VkExec;
typedef struct EffectDeclaration EffectDeclaration;
typedef struct {int live;RcInterval range;} VkTexHold;
typedef VkTexHold VkTextureWriteTarget;
typedef struct {VkTexHold *pin;VkImage image;} VkPinnedTextureWrite;
typedef struct {VkImage image,bc_backing;uint32_t aspect;} VkExecTarget;
typedef struct {const RcInterval *intervals;uint32_t count;} VkTextureAliasRead;
typedef struct {VkTextureAliasRead *load_seed;RcRegistryImageWriteExtent *extents;uint32_t extent_count;} VkExecTargetBinding;
typedef enum {VK_EXEC_GRAPHICS_READY, VK_EXEC_GRAPHICS_COMPAT, VK_EXEC_GRAPHICS_UNAVAILABLE} VkExecGraphicsResult;
typedef enum {VK_EXEC_PREPARATION_MAPPING,VK_EXEC_PREPARATION_ATTACHMENT} VkExecPreparationReason;
enum {VK_EXEC_PREPARATION_IMAGE=1};
typedef struct {
    int owner,result,detail;
    VkExecPreparationReason reason;
    unsigned role,slot;
    const char *stage;
    RcInterval range,peer;
} VkExecPreparationFailure;
typedef struct {
    struct {VkExecTarget color[8],depth;} shape;
    VkExecTargetBinding bindings[9];
    int loan[9];
} VkExecGraphicsPass;
typedef RcRegistryResult (*VkExecAttachmentAccessCheck)(const void *,VkImage,RcInterval,RcInterval *);
typedef struct {unsigned flags,set,binding,slot,pool;} GfxShaderRes;
typedef struct {unsigned writes;int complete;} GfxShader;
typedef struct {const GfxShaderRes *res;int stage,kind,type;unsigned count,set,binding;} VkPipeBinding;
typedef struct {uint32_t n_bindings;VkPipeBinding *bindings;} VkPipeInfo;
typedef struct {
    uint32_t mapped_count; int written;
    RcInterval *mapped;
    const GfxShaderRes *resource;
    VkTexHold *native;
    int span;
} EffectStorageCapture;
typedef struct {EffectStorageCapture storage;int stage;} SelectedStorageWriter;
typedef struct {SelectedStorageWriter entries[1];int rmw,rmw_bitmap;} SelectedStorageWriters;
typedef struct {int native_bindings;EffectStorageCapture storage;} EffectInputCapture;
typedef struct {
    VkTextureWriteTarget *write_target;
    VkPinnedTextureWrite pinned;
    struct {VkImage image;} image;
    struct {uint64_t imageView;} descriptor;
    const GfxShaderRes *resource;
    uint64_t handle;
    int sampled,stage,written;
} EffectImageCapture;
typedef struct {
    struct {uint32_t count;EffectImageCapture *images;} bindings;
    RcRegistryPosition read;
} VkExecStoragePhase;
struct VkExec { uint64_t img[8][32];RcRegistryPosition read; };
typedef struct {
    const void *program,*fallback;
    VkExecGraphicsPass *pass;
    uint64_t program_serial;
    unsigned opcode;
    RcRegistryPosition read_position;
    int mesh,compatibility;
    VkTextureAliasRead *attachment_seeds[9];
    struct {const GfxShader *vs,*ts,*fs;} state;
    struct {const VkPipeInfo *info;} use;
    VkExecPreparationFailure failure;
} VkExecGraphicsDraw;
struct EffectDeclaration {
    const void *program;
    RcRegistryPosition position;
    ResourceCommandEffects *commands;
    ResourceImageEffectRole draw_attachments;
    const GfxShader *shader[8];
    EffectInputCapture *capture[8];
    VkExecStoragePhase *storage_phase;
    SelectedStorageWriters *selected_writers;
    int gpa_unknown,gpa_count,native_bindings,attachment_effects_current;
    unsigned opcode;
};
enum {NVN_STAGE_VS, NVN_STAGE_FS, NVN_STAGE_TASK, NVN_STAGE_MESH, GFX_RES_SSBO, GFX_RES_IMG,
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, RC_EFFECT_WRITES_BUFFER=1, RC_EFFECT_WRITES_GPA=2,
      RC_EFFECT_WRITES_IMAGE=4, RC_EFFECT_WRITES_UNKNOWN=8};
struct ResourceCommandEffects {RcInterval write;int opaque;};

static unsigned refusals;
static const char *terminal_gate;
static const VkTextureAliasRead *vk_texture_alias_target_replacement(const VkTextureAliasRead *seed){return seed;}
static const RcInterval *vk_texture_alias_read_intervals(const VkTextureAliasRead *seed,uint32_t *count){
    *count=seed?seed->count:0;return seed?seed->intervals:NULL;
}
static int vk_texture_target_loan_backing_current(int loan,const RcRegistryImageWriteExtent *extents,unsigned count){
    return loan && extents && count;
}
static uint32_t gfx_shader_guest_writes(const GfxShader *shader){return shader->writes;}
static int gfx_shader_write_inventory_complete(const GfxShader *shader){return shader->complete;}
static int nvn_stage_of(int stage){return stage;}
static int vk_mem_span_capture_current(const VkTexHold *owner,const int *span){return owner->live && *span==13;}
static int storage_phase_position_current(const VkExec *ex,const VkExecStoragePhase *storage){
    return storage->read.scene==ex->read.scene && storage->read.ordinal==ex->read.ordinal;
}
static SelectedStorageWriter *selected_storage_writer(SelectedStorageWriters *writers,int stage,const GfxShaderRes *resource){
    return writers && writers->entries[0].stage==stage && writers->entries[0].storage.resource==resource ?
        writers->entries:NULL;
}
static EffectStorageCapture *effect_storage_find(EffectInputCapture *capture,uint32_t set,uint32_t binding) {
    const GfxShaderRes *r=capture->storage.resource;
    return r && r->set==set && r->binding==binding ? &capture->storage:NULL;
}
static EffectImageCapture *effect_image_lookup(EffectImageCapture *images,uint32_t count,
                                                const GfxShaderRes *resource,uint64_t handle){
    for (uint32_t i=0;i<count;++i) if(images[i].resource==resource&&images[i].handle==handle) return &images[i];
    return NULL;
}
static RcRegistryResult vk_texture_write_compare_input(const VkTextureWriteTarget *target,const VkPinnedTextureWrite *pin,
    const RcInterval *input,uint32_t count,RcInterval *overlap){
    const VkTexHold *owner=target?target:pin->pin;
    assert(count==1);
    if (!owner->live) return RC_REGISTRY_STALE;
    if (rc_range_overlap(owner->range.range,input->range)) {*overlap=owner->range;return RC_REGISTRY_BLOCKED;}
    return RC_REGISTRY_OK;
}
RcRegistryResult resource_effects_shader_write_peer(const ResourceCommandEffects *effects,
    const ResourceImageEffectRole *attachment,RcInterval peer,RcInterval *overlap){
    assert(attachment->owner==effects);
    if (effects->opaque) return RC_REGISTRY_UNSUPPORTED;
    if (rc_range_overlap(effects->write.range,peer.range)) {*overlap=effects->write;return RC_REGISTRY_BLOCKED;}
    return RC_REGISTRY_OK;
}
static int graphics_attachment_peer_failure(VkExecPreparationFailure *f,unsigned slot,VkExecPreparationReason reason,
    int detail,RcInterval attached,RcInterval overlap){
    assert(!f->owner);
    *f=(VkExecPreparationFailure){.owner=VK_EXEC_PREPARATION_IMAGE,.reason=reason,
         .result=VK_EXEC_GRAPHICS_UNAVAILABLE,.detail=detail,.slot=slot,.range=attached,.peer=overlap};
    return 0;
}
static void vk_exec_graphics_failure_set(VkExecGraphicsDraw *draw,const VkExecPreparationFailure *failure){
    if(!draw->failure.owner) draw->failure=*failure;
}
static void vk_exec_graphics_preparation_unavailable(VkExec *ex,uint32_t opcode,VkExecGraphicsResult result,
    const VkExecPreparationFailure *failure,const char *gate){
    assert(ex&&opcode==26&&result==VK_EXEC_GRAPHICS_UNAVAILABLE&&failure->owner&&gate);
    ++refusals;terminal_gate=gate;
}
static void rt_log(const char *format,...) { assert(format && strstr(format,"replacement LOAD")); }
'''

CASES = r'''
int main(void){
    const RcRegistryPosition read={31,14};
    const unsigned opcode=26;
    const RcInterval load={0x8000000000,4,{80,0x400,128}};
    const RcInterval depth={0x8000010000,5,{80,0x800,64}};
    RcInterval writer={0x8000020000,6,{80,0x900,48}};
    const RcInterval legal=writer;
    VkTextureAliasRead attach_seed={&load,1};
    RcRegistryImageWriteExtent depth_extent={.interval=depth};
    VkExecGraphicsPass pass={
        .shape={.color={{.image=1}},.depth={.image=2}},.loan={[8]=1},
        .bindings={{.load_seed=&attach_seed},[8]={.extents=&depth_extent,.extent_count=1}},
    };
    GfxShaderRes resource={.flags=1,.set=1,.binding=4};
    VkPipeBinding binding={.res=&resource,.stage=NVN_STAGE_FS,.kind=GFX_RES_SSBO,.set=1,.binding=1,.count=1};
    VkPipeInfo info={.n_bindings=1,.bindings=&binding};
    GfxShader shader={.writes=RC_EFFECT_WRITES_BUFFER,.complete=1};
    VkExecGraphicsDraw draw={.pass=&pass,.state={.fs=&shader},.use={&info},
        .program=&info,.read_position=read,.opcode=opcode,.attachment_seeds={&attach_seed}};
    VkExec ex={.read=read};
    ResourceCommandEffects effects={.write=writer};
    VkTexHold capture={.live=1,.range=writer};
    EffectInputCapture selected={.native_bindings=1,.storage={.native=&capture,.span=13,.written=1,
                                                            .mapped=&writer,.mapped_count=1,.resource=&resource}};
    EffectDeclaration declaration={
        .program=draw.program,.position={read.scene,read.ordinal+1},.opcode=opcode,
        .commands=&effects,.draw_attachments={&effects},.shader={[NVN_STAGE_FS]=&shader},
        .capture={[NVN_STAGE_FS]=&selected},.native_bindings=1,
    };
    // The ordinary fixed raw descriptor uses its own retained output, with no imported selected-writer wrapper.
    assert(!declaration.selected_writers && !graphics_load_effects_issue_current(&draw,&declaration));
    assert(graphics_load_effects_required(&draw));
    declaration.attachment_effects_current=graphics_load_effects_select(&ex,&draw,&declaration);
    assert(declaration.attachment_effects_current && graphics_load_effects_issue_current(&draw,&declaration));
    declaration.position.ordinal++;
    assert(!graphics_load_effects_issue_current(&draw,&declaration));
    declaration.position.ordinal--;
    assert(!graphics_load_effects_issue_current(&draw,NULL));
    assert(!refusals);
    declaration.attachment_effects_current=0;
    effects.write.range.offset=depth.range.offset;
    assert(!graphics_load_effects_select(&ex,&draw,&declaration));
    assert(refusals==1&&draw.failure.slot==8&&draw.failure.peer.mapping==writer.mapping&&
           !graphics_load_effects_issue_current(&draw,&declaration));
    draw.failure=(VkExecPreparationFailure){0};
    effects.write=legal;
    effects.opaque=1;
    assert(!graphics_load_effects_select(&ex,&draw,&declaration));
    assert(refusals==2&&draw.failure.detail==RC_REGISTRY_UNSUPPORTED&&!draw.failure.peer.mapping);
    draw.failure=(VkExecPreparationFailure){0};
    effects.opaque=0;
    capture.live=0;
    assert(!graphics_load_effects_select(&ex,&draw,&declaration));
    assert(refusals==3&&!strcmp(terminal_gate,"shader-buffer-owner"));
    draw.failure=(VkExecPreparationFailure){0};
    capture.live=1;
    shader.complete=0;
    assert(!graphics_load_effects_select(&ex,&draw,&declaration));
    assert(refusals==4&&!strcmp(terminal_gate,"shader-write-inventory"));
    draw.failure=(VkExecPreparationFailure){0};
    shader.complete=1;
    shader.writes=RC_EFFECT_WRITES_GPA;
    declaration.gpa_unknown=1;
    assert(!graphics_load_effects_select(&ex,&draw,&declaration));
    assert(refusals==5&&!strcmp(terminal_gate,"shader-gpa-unbounded"));
    draw.failure=(VkExecPreparationFailure){0};
    declaration.gpa_unknown=0;
    shader.writes=0;
    // An independently captured readonly storage image is a positive access peer, not an opaque writer.
    resource.flags=0;
    binding.kind=GFX_RES_IMG;
    binding.type=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ex.img[NVN_STAGE_FS][0]=41;
    EffectImageCapture image={.image={.image=3},.descriptor={.imageView=31},.resource=&resource,
                             .stage=NVN_STAGE_FS,.handle=41,.pinned={.pin=&capture,.image=3}};
    VkExecStoragePhase images={.read=read,.bindings={1,&image}};
    declaration.storage_phase=&images;
    assert(graphics_load_effects_select(&ex,&draw,&declaration));
    image.image.image=1;
    assert(!graphics_load_effects_select(&ex,&draw,&declaration));
    assert(refusals==6&&draw.failure.slot==0&&draw.failure.detail==RC_REGISTRY_BLOCKED);
    draw.failure=(VkExecPreparationFailure){0};
    image.image.image=3;
    capture.range.range.offset=load.range.offset;
    assert(!graphics_load_effects_select(&ex,&draw,&declaration));
    assert(refusals==7&&draw.failure.peer.mapping==writer.mapping);
    puts("Replacement LOAD uses the original selected raw buffer and image peers through issue.");
}
'''


def main():
    draw = (RUNTIME / "vk_exec_draw.c").read_text()
    replay = (RUNTIME / "vk_exec.c").read_text()
    declare = function(draw, "vk_exec_effects_declare")
    assert declare.index("effect_declaration_finish(d)") < declare.index("graphics_load_effects_select(")
    activate = function(draw, "graphics_draw_activate")
    assert activate.index("graphics_load_effects_issue_current(") < activate.index("graphics_native_rmw_order_for_draw(")
    selected = [function(replay, "vk_exec_graphics_pass_attachment_access")]
    selected += [function(draw, name) for name in (
        "graphics_load_replacement", "graphics_load_effects_required", "graphics_load_effects_issue_current",
        "graphics_load_effects_check", "graphics_load_effects_unavailable", "graphics_load_effects_select",
    )]
    cc = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ["PATH"])
    with tempfile.TemporaryDirectory(prefix="replacement-load-writes-") as work:
        path = Path(work) / "fixture.c"
        executable = Path(work) / ("fixture.exe" if os.name == "nt" else "fixture")
        path.write_text(STUBS + "\n".join(selected) + CASES, encoding="utf-8")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", f"-I{RUNTIME}",
                        str(path), "-o", str(executable)], check=True, env=env)
        subprocess.run([str(executable)], check=True, env=env, timeout=20)


if __name__ == "__main__":
    main()

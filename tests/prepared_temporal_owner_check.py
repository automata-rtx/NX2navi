
"""Exercise the admitted temporal native target receipts independently of SDK execution."""
import os
import shutil
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
path = Path(os.getenv("PREPARED_TEMPORAL_CHECK_SOURCE", root / "runtime/vk_dlaa.c"))
text = path.read_text(encoding="utf-8")
owners = text[text.index("typedef struct {\n    VkTexImage image;\n    VkMemoryImageHold *native;"):text.index("static uint64_t size_generation;")]

observer = text[text.index("struct VkDlaaPreparedObserver {"):text.index("static int dlaa_pass_execute(")]
consume = text[text.index("int vk_dlaa_pass_prepared("):text.index("// NX2NAVI_DLSS, read at each present")]

STUBS = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef uint64_t VkImage;
typedef struct { int refs; } tex_t;
typedef struct {
    VkImage image;
    uint64_t addr;
    unsigned format,width,height,aspect,samples,levels,layers,gone;
} VkTexImage;
typedef struct { VkImage image; int refs; } VkMemoryImageHold;
typedef struct { VkImage image; int refs, current; } VkTextureAliasRead;
typedef struct { VkImage image; } VkTexturePresentSource;
typedef struct { int active; } RcBudgetTicket;
enum { VK_SUCCESS, RC_BUDGET_TEXTURE };
typedef struct {
    VkImage image,bc_backing;
    int exact_snapshot;
    const tex_t *tex;
    const VkTexImage *img;
    uint32_t format,view_type,aspect,level,base_layer,layers,width,height,samples,is3d;
} VkExecTarget;

typedef struct { uint64_t fence; } VkExec;
typedef struct {
    VkExecTarget color[8],depth;
    unsigned pass_ncolor,pass_w,pass_layers,in_pass;
} VkExecGraphicsPassShape;
typedef struct { uint64_t scene,ordinal; } RcRegistryPosition;
typedef enum { DLAA_PASS_NONE, DLAA_PASS_GBUFFER, DLAA_PASS_ARM, DLAA_PASS_OUTPUT } DlaaPassAction;
typedef struct { const VkExecTarget *color,*depth; unsigned n,w,layers; } DlaaPassTargets;
typedef struct { DlaaPassAction action; } DlaaPassDecision;
typedef struct VkDlaaPreparedObserver VkDlaaPreparedObserver;
enum { VK_FORMAT_R8G8B8A8_UNORM=4,VK_FORMAT_D32_SFLOAT=126,VK_IMAGE_ASPECT_DEPTH_BIT=2 };
static RcRegistryPosition ordered;
static DlaaPassAction selected_action;
static RcRegistryPosition vk_exec_resource_position(void) { return ordered; }
static int rc_registry_position_compare(RcRegistryPosition a,RcRegistryPosition b) {
    return a.scene!=b.scene || a.ordinal!=b.ordinal;
}
static DlaaPassDecision dlaa_pass_select(const VkExec *ex,DlaaPassTargets targets,const unsigned *o) {
    assert(ex&&targets.color&&targets.depth&&!o);
    return (DlaaPassDecision){selected_action};
}
static struct {
    VkExecTarget color, depth, ids;
    int armed, ids_ok, nviews;
    VkMemoryImageHold *camera_depth_owner;
    VkImage camera_depth;
    uint32_t camera_depth_format, camera_depth_level, camera_depth_layer, camera_depth_layers;
    uint64_t cam_addr;
} D;
static VkMemoryImageHold base={27,0}, resized={29,0};
static VkTexturePresentSource present;
static int output_refs, seed_before_transfer, raw_before_transfer, metadata_live, refuse_metadata;
static struct {
    void (*complete)(void *);
    void (*cancel)(void *,int);
    void *arg;
} native[8];
static unsigned count;

static int vk_replay_metadata_allocate(size_t size,int clear,int producer,RcBudgetTicket *ticket,void **out) {
    assert(size&&clear&&producer==RC_BUDGET_TEXTURE&&!ticket->active);
    if(refuse_metadata)return 1;
    *out=calloc(1,size);assert(*out);ticket->active=1;metadata_live++;return VK_SUCCESS;
}
static void vk_replay_metadata_free(void *allocation,RcBudgetTicket *ticket) {
    assert(ticket->active&&metadata_live);ticket->active=0;metadata_live--;free(allocation);
}
static const tex_t *nvn_texture_pin_acquire(const tex_t *t) { ((tex_t*)t)->refs++; return t; }
static void nvn_texture_pin_release(const tex_t *t) { assert(t&&t->refs>0); ((tex_t*)t)->refs--; }
static VkMemoryImageHold *vk_mem_image_hold(VkImage image) {
    VkMemoryImageHold *hold = image == base.image ? &base : &resized;
    assert(hold->image == image);
    hold->refs++;
    return hold;
}
static void vk_mem_image_release(VkMemoryImageHold *hold) { assert(hold&&hold->refs>0); hold->refs--; }
static int vk_texture_alias_target_current(VkTextureAliasRead *seed,VkImage image) {
    return seed&&seed->current&&seed->image==image;
}
static VkTextureAliasRead *vk_texture_alias_read_retain(VkTextureAliasRead *seed) {
    assert(seed&&seed->refs>0);seed->refs++;return seed;
}
static void vk_texture_alias_read_cancel(VkTextureAliasRead *seed) {
    if(seed){assert(seed->refs>0);seed->refs--;}
}
static VkTexturePresentSource *vk_texture_present_retain_alias_image(VkTextureAliasRead *seed,VkImage exact) {
    assert(seed&&seed->refs>0&&exact==seed->image);seed_before_transfer=seed->refs;output_refs++;
    present.image=exact;return &present;
}
static VkTexturePresentSource *vk_texture_present_retain_held_image(VkMemoryImageHold *hold,VkImage exact) {
    assert(hold&&hold->refs>0&&exact==hold->image);raw_before_transfer=hold->refs;output_refs++;
    present.image=exact;return &present;
}
static void vk_completion_add_owned(uint64_t fence,void (*complete)(void *),void (*cancel)(void *,int),void *arg) {
    assert(fence==5&&count<sizeof native/sizeof native[0]);
    native[count].complete=complete;native[count].cancel=cancel;native[count++].arg=arg;
}
static void rt_fatal(const char *msg) { assert(msg);abort(); }
'''

EXECUTE = r'''
static VkExecGraphicsPassShape current_shape;
static int executed;
static int dlaa_pass_execute(VkExec *ex,VkDlaaPreparedObserver *p) {
    if(!vk_dlaa_prepared_observer_current(ex,&current_shape,p))return 0;
    assert(p->action==DLAA_PASS_ARM&&p->color_owner&&p->depth_owner);
    dlaa_frame_target_set(DLAA_TARGET_COLOR,p->color_owner);
    dlaa_frame_target_set(DLAA_TARGET_DEPTH,p->depth_owner);
    p->color_owner=p->depth_owner=NULL;
    executed++;
    return 1;
}
'''

TEST = r'''
int main(void){
    tex_t pin={0};
    VkTexImage selected={.image=17,.addr=4096,.format=4,.width=1280,.height=720,.aspect=1,.samples=1,.levels=1,.layers=1};
    VkExecTarget target={.image=17,.exact_snapshot=1,.tex=&pin,.img=&selected};
    VkTextureAliasRead seed={17,1,1};
    const VkExec ex={.fence=5};
    assert(!dlaa_target_capture(&target,NULL)==0 || base.refs); // A raw test uses its actual allocated image below.
    DlaaTargetOwner *held=dlaa_target_capture(&target,&seed);
    assert(held&&pin.refs==1&&seed.refs==2&&!base.refs&&metadata_live==1);
    D.color=dlaa_owned_target(&target,held);
    dlaa_frame_target_set(DLAA_TARGET_COLOR,held);
    D.armed=1;
    assert(dlaa_targets_armed()&&D.color.img!=&selected&&D.color.img->gone);
    const VkTexImage before=selected;
    selected.image=19;
    assert(D.color.img->image==17&&dlaa_selected_image_matches(&before,D.color.img,held));
    assert(!dlaa_selected_image_matches(&selected,D.color.img,held));
    dlaa_target_record((VkExec*)&ex,held,1);
    vk_texture_alias_read_cancel(&seed); // original prepared pass cancels its own acquisition
    assert(seed.refs==1&&pin.refs==1&&count==1);
    assert(dlaa_present_target(held,17)&&output_refs==1&&seed_before_transfer==1);
    assert(!dlaa_present_target(held,19));
    dlaa_frame_targets_close();
    assert(seed.refs==1&&pin.refs==1&&!D.color.img); // recording is still the allocation owner
    native[0].complete(native[0].arg);
    assert(!seed.refs&&!pin.refs&&!metadata_live);

    selected.image=27;target.image=27;
    DlaaTargetOwner *raw=dlaa_target_capture(&target,NULL);
    assert(raw&&pin.refs==1&&base.refs==1);
    D.color=dlaa_owned_target(&target,raw);
    dlaa_frame_target_set(DLAA_TARGET_COLOR,raw);
    dlaa_target_record((VkExec*)&ex,raw,0);
    dlaa_target_record((VkExec*)&ex,raw,1);
    assert(dlaa_present_target(raw,27)&&output_refs==2&&raw_before_transfer==1);
    native[1].cancel(native[1].arg,0);
    assert(dlaa_targets_armed()&&dlaa_present_target(raw,27)); // Canceling a reader keeps the prior source.
    native[2].cancel(native[2].arg,0);
    assert(!dlaa_targets_armed()&&!dlaa_present_target(raw,27)&&base.refs==1&&pin.refs==1);
    dlaa_frame_targets_close();
    assert(!base.refs&&!pin.refs&&!metadata_live);

    seed.refs=1;seed.current=0;
    selected.image=17;target.image=17;
    assert(!dlaa_target_capture(&target,&seed)&&seed.refs==1&&!pin.refs);
    seed.current=1;refuse_metadata=1;
    assert(!dlaa_target_capture(&target,&seed)&&seed.refs==1&&!pin.refs&&!metadata_live);

    refuse_metadata=0;
    VkExecGraphicsPassShape shape={.color={target},.depth={.image=27,.exact_snapshot=1,.tex=&pin},.pass_ncolor=1};
    VkTexImage depth=selected;depth.image=27;shape.depth.img=&depth;
    VkDlaaPreparedObserver *prepared=NULL;
    ordered=(RcRegistryPosition){5,8};
    selected_action=DLAA_PASS_OUTPUT;
    assert(!vk_dlaa_prepare_observer(&ex,&shape,&seed,NULL,&prepared)&&!prepared&&!metadata_live);
    selected_action=DLAA_PASS_ARM;
    shape.in_pass=1;
    assert(!vk_dlaa_prepare_observer(&ex,&shape,&seed,NULL,&prepared)&&!prepared&&!metadata_live);
    shape.in_pass=0;
    assert(vk_dlaa_prepare_observer(&ex,&shape,&seed,NULL,&prepared));
    assert(prepared&&metadata_live==3&&seed.refs==2&&base.refs==1&&pin.refs==2&&!executed);
    assert(vk_dlaa_prepared_observer_current(&ex,&shape,prepared));
    ++ordered.ordinal;
    assert(!vk_dlaa_prepared_observer_current(&ex,&shape,prepared));
    --ordered.ordinal;
    shape.depth.base_layer=1;
    assert(!vk_dlaa_prepared_observer_current(&ex,&shape,prepared));
    shape.depth.base_layer=0;
    selected_action=DLAA_PASS_GBUFFER;
    assert(!vk_dlaa_prepared_observer_current(&ex,&shape,prepared));
    assert(!vk_dlaa_pass_prepared((VkExec*)&ex,&prepared)&&prepared&&!executed);
    vk_dlaa_prepared_observer_cancel(&prepared);
    assert(!prepared&&!metadata_live&&seed.refs==1&&!base.refs&&!pin.refs);
    vk_dlaa_prepared_observer_cancel(&prepared);
    selected_action=DLAA_PASS_ARM;
    assert(vk_dlaa_prepare_observer(&ex,&shape,&seed,NULL,&prepared));
    current_shape=shape;
    assert(vk_dlaa_pass_prepared((VkExec*)&ex,&prepared)&&!prepared&&executed==1);
    assert(!vk_dlaa_pass_prepared((VkExec*)&ex,&prepared)&&executed==1);
    assert(metadata_live==2&&seed.refs==2&&base.refs==1&&pin.refs==2);
    dlaa_frame_targets_close();
    assert(!metadata_live&&seed.refs==1&&!base.refs&&!pin.refs);
    shape.depth.format=VK_FORMAT_D32_SFLOAT;
    shape.depth.aspect=VK_IMAGE_ASPECT_DEPTH_BIT;
    shape.depth.layers=shape.depth.samples=1;
    depth.format=VK_FORMAT_D32_SFLOAT;
    depth.aspect=VK_IMAGE_ASPECT_DEPTH_BIT;
    VkTextureAliasRead depth_seed={.image=27,.refs=1,.current=1};
    shape.depth.aspect=1;
    assert(!vk_dlaa_prepare_observer(&ex,&shape,&seed,&depth_seed,&prepared) && !prepared && !metadata_live);
    shape.depth.aspect=VK_IMAGE_ASPECT_DEPTH_BIT;
    depth_seed.image=29;
    assert(!vk_dlaa_prepare_observer(&ex,&shape,&seed,&depth_seed,&prepared) && !prepared &&
           seed.refs==1 && depth_seed.refs==1 && !pin.refs && !metadata_live);
    depth_seed.image=27;
    assert(vk_dlaa_prepare_observer(&ex,&shape,&seed,&depth_seed,&prepared) && !base.refs &&
           seed.refs==2 && depth_seed.refs==2 && metadata_live==3);
    vk_dlaa_prepared_observer_cancel(&prepared);
    assert(seed.refs==1 && depth_seed.refs==1 && !metadata_live);
    assert(vk_dlaa_prepare_observer(&ex,&shape,&seed,&depth_seed,&prepared));
    current_shape=shape;
    assert(vk_dlaa_pass_prepared((VkExec*)&ex,&prepared) && !prepared && executed==2);
    DlaaTargetOwner *dep=frame_targets[DLAA_TARGET_DEPTH];
    assert(dep && dep->seed==&depth_seed && !dep->native && dep->image.format==VK_FORMAT_D32_SFLOAT);
    dlaa_target_record((VkExec*)&ex,dep,1);
    vk_texture_alias_read_cancel(&depth_seed);
    depth.image=29; depth_seed.current=0; // The established owner outlives the initial receipt and later remap.
    assert(dlaa_present_target(dep,27) && !dlaa_present_target(dep,29));
    dlaa_frame_targets_close();
    assert(depth_seed.refs==1 && metadata_live==1);
    native[3].complete(native[3].arg);
    assert(!depth_seed.refs && !metadata_live && !pin.refs);

    VkExecTarget camera_depth = shape.depth;
    assert(!dlaa_camera_depth_matches(&camera_depth));
    dlaa_camera_depth_select(&camera_depth);
    assert(D.camera_depth_owner == &base && base.refs == 1 && dlaa_camera_depth_matches(&camera_depth));
    D.cam_addr = 0x6000;
    D.nviews = 2;
    dlaa_camera_depth_select(&camera_depth);
    assert(base.refs == 1 && D.cam_addr == 0x6000 && D.nviews == 2);
    camera_depth.base_layer++;
    assert(!dlaa_camera_depth_matches(&camera_depth));
    dlaa_camera_depth_select(&camera_depth);
    assert(base.refs == 1 && !D.cam_addr && !D.nviews);
    VkMemoryImageHold *previous = vk_mem_image_hold(camera_depth.image);
    camera_depth.image = resized.image;
    camera_depth.width = 1920;
    camera_depth.height = 1080;
    dlaa_camera_depth_select(&camera_depth);
    assert(D.camera_depth_owner == &resized && base.refs == 1 && resized.refs == 1);
    dlaa_frame_targets_close();
    assert(base.refs == 1 && !resized.refs && !D.camera_depth_owner);
    vk_mem_image_release(previous);
    assert(!base.refs);
    puts("prepared temporal target: exact color and D32 seed, camera depth replacement and native fate passed");
}
'''.replace("    assert(!dlaa_target_capture(&target,NULL)==0 || base.refs); // A raw test uses its actual allocated image below.\n", "")

compiler = os.getenv("CC") or shutil.which("clang")
if not compiler:
    raise SystemExit("Clang is required (set CC)")
cc = Path(shutil.which(compiler) or compiler).resolve()
env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="temporal-target-owner-") as directory:
    c = Path(directory) / "check.c"
    out = c.with_suffix(".exe")
    c.write_text(STUBS + owners + observer + EXECUTE + consume + TEST, encoding="utf-8")
    subprocess.run([str(cc), "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fsanitize-trap=all", str(c), "-o", str(out)], env=env, check=True)
    subprocess.run([str(out)], env=env, check=True)

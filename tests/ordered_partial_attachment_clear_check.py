"""Check original-read partial-clear input retention and selected attachment emission."""
import os
from pathlib import Path
import subprocess
import tempfile

from ordered_graphics_feedback_check import function

ROOT = Path(__file__).resolve().parents[1]

STUB = r"""
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { OP_CLEAR_DS=10, OP_CLEAR_COLOR=9,VK_IMAGE_ASPECT_COLOR_BIT=1,VK_IMAGE_ASPECT_DEPTH_BIT=2,
       NVN_TEXTURE_DEPTH_STENCIL_MODE_DEPTH=0,NVN_TEXTURE_TARGET_2D=2 };
enum { VK_EXEC_PREPARATION_IMAGE=1,VK_EXEC_PREPARATION_ATTACHMENT=4,VK_EXEC_GRAPHICS_UNAVAILABLE=5,
       RC_REGISTRY_STALE=2, VK_NULL_HANDLE=0 };
enum { VK_EXEC_PREPARATION_MAPPING, VK_EXEC_PREPARATION_ORDER, VK_EXEC_PREPARATION_PLACEMENT,
       VK_EXEC_PREPARATION_PRESSURE, VK_EXEC_GRAPHICS_PASS_NEEDS_ORDERED, VK_EXEC_GRAPHICS_PASS_STOPPED,
       VK_EXEC_RESTORE_STOPPED };
typedef int VkExecGraphicsPassResult;
typedef int VkExecPreparationReason;
typedef int VkExecRestoreResult;
enum { VK_EXEC_GRAPHICS_PASS_READY,VK_EXEC_GRAPHICS_PASS_UNAVAILABLE,VK_EXEC_RESTORE_READY,VK_EXEC_RESTORE_STALE };
typedef struct {int width,height,format,target,samples;} TexDesc;
typedef struct {TexDesc d;struct {unsigned layers;uint64_t size;} L;void *mem;} tex_t;
typedef struct {int set_levels,set_layers,set_format,set_target,base_level,nlevels,min_layer,nlayers,
                    format,target,set_dsmode,dsmode;} view_t;
typedef struct {int n; const tex_t *color[8],*depth; view_t cview[8],dview; int has_cview[8],has_dview;} p_set_rt;
typedef struct {int x,y,w,h;} p_scissor;
typedef struct {int32_t xoffset,yoffset,zoffset,width,height,depth;} NVNcopyRegion;
typedef struct {uint32_t op,size;} op_hdr;
typedef struct {int depth_mask,stencil_mask;} p_clear_ds;
typedef struct {int index,mask;} p_clear_color;
typedef struct {unsigned image,aspect;} VkTexImage;
typedef struct {uint32_t aspectMask,baseMipLevel,levelCount,baseArrayLayer,layerCount;} VkImageSubresourceRange;
typedef struct {unsigned image,level,base_layer,layers,scale,blocks,bc_backing,width,height;} VkExecTarget;
typedef struct {struct {int32_t x,y;} offset;struct {uint32_t width,height;} extent;} VkRect2D;
typedef struct {VkRect2D rect;uint32_t baseArrayLayer,layerCount;} VkClearRect;
typedef struct {uint32_t aspectMask,colorAttachment;} VkClearAttachment;
typedef struct {int live;} VkExecGraphicsPass;
typedef struct {int live;} VkTextureAliasRead;
typedef struct {uint64_t mapping;struct {uint64_t backing;}range;} RcInterval;
typedef struct {uint64_t scene,ordinal;} RcRegistryPosition;
typedef struct {int phase,status,coverage; RcInterval receiver; unsigned image; uint64_t registration,material_serial;}
    VkTextureAliasFailure;
typedef struct {int owner,reason,result,detail,coverage; RcInterval range; unsigned role,slot,image;
    uint64_t registration,material_serial; const char *stage;
    struct { RcRegistryPosition position; uint64_t address,bytes; unsigned present,seeded,width,height,samples,guest_format; } load;
} VkExecPreparationFailure;
typedef struct {
    int in_pass,render_disabled,has_scissor,rt_resolved,cb,recorded;
    uint64_t resource_op;
    uint32_t pass_layers,target_binding_mask,bc_target_mask,bc_written_mask,fb_dirty;
    p_scissor scissor;
    VkExecTarget color[8],depth;
    p_set_rt rt;
} VkExec;
typedef struct {const VkExec *ex;p_set_rt rt;VkTextureAliasRead *seed[9];} AttachmentSeedScope;

static struct {op_hdr h;p_clear_ds p;} depth_op={{OP_CLEAR_DS,sizeof(depth_op)},{1,0}};
static struct {op_hdr h;p_clear_color p;} color_op={{OP_CLEAR_COLOR,sizeof(color_op)},{0,15}};
static tex_t depth={.d={.width=128,.height=128,.format=52,.target=2},.L={.layers=1}},
             color={.d={.width=128,.height=128,.format=63,.target=2},.L={.layers=1}};
static VkTextureAliasRead seed, color_seed;
static VkExecGraphicsPass pass;
static VkTexImage image;
static unsigned entered,captures,reads,releases,failures,native_clear,opened,prepares,writes,fragments;
static int observed_slot,current=1,peers=1,typed_capture=1,reconstruction=1,projection=1;
static const char *failed_stage;
static int color_capture, prepared_mask, materializations;
static VkExec *ex_live;

static const void *nvn_op_payload(const op_hdr *h){return h+1;}
static uint32_t rt_color_count(const p_set_rt *r){return (uint32_t)r->n;}
static int vk_texture_alias_depth_attachment_range(const tex_t *t,const view_t *v,VkImageSubresourceRange *range) {
    if (t!=&depth) return 0;
    *range=(VkImageSubresourceRange){2,v && v->set_levels ? (unsigned)v->base_level : 0,1,0,1};
    return 1;
}
static int vk_texture_alias_attachment_range(const tex_t *t,const view_t *v,int is_depth,VkImageSubresourceRange *range) {
    if (is_depth) return vk_texture_alias_depth_attachment_range(t,v,range);
    if (t!=&color) return 0;
    *range=(VkImageSubresourceRange){1,v && v->set_levels ? (unsigned)v->base_level : 0,1,0,1};
    return 1;
}
static uint32_t level_extent(uint32_t size,uint32_t level) {return size>>level ? size>>level : 1;}
static RcRegistryPosition vk_exec_resource_position(void) {return (RcRegistryPosition){1,5720};}
static int vk_texture_alias_region_transfer(const tex_t *t,const view_t *v,const NVNcopyRegion *r){
    return t==&color && !v && r->width==t->d.width && r->height==t->d.height;
}
static void image_write_input_position(VkExec *ex){assert(!ex->resource_op);ex->resource_op=5720;entered++;}
static int vk_texture_alias_read_current(VkTextureAliasRead *s){return (s==&seed || s==&color_seed) && s->live && current;}
static int vk_texture_alias_target_capture(const tex_t *t,const view_t *v,uint32_t layers,VkTextureAliasRead **out,int ordered,
                                           VkTextureAliasFailure *failure){
    assert(layers==1);
    assert(ex_live->resource_op==5720 && ordered && v && v->set_format && t->d.format==v->format && failure);
    assert(!*out && !seed.live);
    observed_slot=t==&depth?8:0;
    if (!observed_slot) {
        if (color_capture) { assert(!color_seed.live); color_seed.live=1;*out=&color_seed; }
        return color_capture;
    }
    captures++;
    if(typed_capture>0){seed.live=1;*out=&seed;}
    if(typed_capture<0)*failure=(VkTextureAliasFailure){.phase=3,.status=4,.receiver={.mapping=9,.range={.backing=6}}};
    return typed_capture;
}
static int vk_exec_prepare_alias_texture_read(VkExec *ex,VkTextureAliasRead *const *seeds,uint32_t count,
    int (*validate)(void *),void *opaque,const VkTexImage **out){
    assert(ex->resource_op==5720 && count==1u+(unsigned)color_capture && color_seed.live==color_capture);
    assert(seeds[count-1]==&seed && seed.live);
    if (color_capture) assert(seeds[0]==&color_seed);
    reads++;
    if(!reconstruction || !validate(opaque)) return VK_EXEC_RESTORE_STALE;
    for (unsigned i=0;i<count;i++) out[i]=&image;
    return VK_EXEC_RESTORE_READY;
}
static int graphics_pass_capture(VkExec *ex,int ordered,int reopen,VkTextureAliasRead *const *seeds,
                                 VkExecGraphicsPass **out,void *failure){
    (void)failure;
    assert(ex->resource_op==5720 && ordered && !reopen && !ex->rt_resolved && !pass.live);
    prepared_mask=0;
    for (int i=0;i<9;i++) if (seeds[i]) {
        assert((i==0 && seeds[i]==&color_seed) || (i==8 && seeds[i]==&seed));
        prepared_mask|=1<<i;
    }
    pass.live=1;*out=&pass;
    return VK_EXEC_GRAPHICS_PASS_READY;
}
static int vk_exec_graphics_pass_attachment_peers(const VkExecGraphicsPass *p,void *failure){
    (void)failure;assert(p==&pass && p->live);return peers;
}
static void vk_exec_graphics_pass_cancel(VkExecGraphicsPass **p){
    if(!*p)return;
    assert(*p==&pass && pass.live);
    pass.live=0;*p=NULL;
}
static void vk_texture_alias_target_cancel(VkTextureAliasRead *s){
    if(!s)return;
    assert((s==&seed || s==&color_seed) && s->live && !pass.live);
    s->live=0;releases++;
}
static int vk_sync_stopping(void){return 0;}
static int vk_device_is_lost(void){return 0;}
static int vk_texture_deferred(void){return 0;}
static int vk_exec_recording_failed(void){return 0;}
static void graphics_preparation_unavailable(VkExec *ex,uint32_t opcode,int result,
    const VkExecPreparationFailure *why,const char *reason){
    assert(ex==ex_live && ex->resource_op==5720 && opcode==OP_CLEAR_DS && reason);
    assert(result==VK_EXEC_GRAPHICS_UNAVAILABLE && why->owner==VK_EXEC_PREPARATION_IMAGE &&
           (why->slot==8 || why->slot==UINT32_MAX) && why->stage);
    if (!strcmp(why->stage,"attachment-load-capture"))
        assert(why->detail==4 && why->range.mapping==9 && why->range.range.backing==6);
    failed_stage=why->stage;failures++;
}
static int graphics_pass_hooks_active(void){return 0;}
static int prepare_bound_texture_relocation(VkExec *ex,const tex_t *t,int ordered){
    (void)ex;(void)t;(void)ordered;assert(0);return -1;
}
static int graphics_attachment_failure(VkExecPreparationFailure *f,int reason,int result,unsigned slot,int detail){
    *f=(VkExecPreparationFailure){.owner=VK_EXEC_PREPARATION_IMAGE,.reason=reason,.result=result,
        .slot=slot,.detail=detail,.stage="attachment-load-preparation"};
    return result;
}
static int graphics_attachment_peer_failure(VkExecPreparationFailure *f,unsigned slot,int reason,int detail,
                                           RcInterval selected,RcInterval peer){
    (void)peer;
    graphics_attachment_failure(f,reason,VK_EXEC_GRAPHICS_PASS_UNAVAILABLE,slot,detail);
    f->range=selected;return 0;
}
static const VkTexImage *vk_texture_get_alias_transfer_write(const tex_t *t){
    assert(t==&color);materializations++;image.image=17;return &image;
}
static void resolve_target_image(const tex_t *t,const view_t *v,int has_view,int depth_slot,const VkTexImage *native,
                                int allow_standin,VkExecTarget *target){
    assert(t==&color && v && has_view && !depth_slot && native==&image && !allow_standin);
    *target=(VkExecTarget){.image=native->image,.layers=1,.scale=1,.width=t->d.width,.height=t->d.height};
}
static void pass_begin(VkExec *ex,VkExecGraphicsPass *admitted,uint32_t opcode){
    assert(!admitted && opcode==OP_CLEAR_DS);
    if(ex->in_pass)return;
    ex->in_pass=1;opened++;
}
static VkRect2D vk_exec_scissor(VkExec *ex){
    return (VkRect2D){{0,0},{(uint32_t)ex->scissor.w,(uint32_t)ex->scissor.h}};
}
static void vk_exec_feedback_war(VkExec *ex,uint32_t bit){assert(ex->in_pass && bit==256);}
static void resource_attachment_prepare(VkExec *ex,unsigned slot){
    assert(ex->in_pass && slot==8 && prepares==writes);prepares++;
}
static int resource_attachment_written(VkExec *ex,unsigned slot){
    assert(ex->in_pass && slot==8 && prepares==writes+1 && native_clear==prepares);writes++;
    return projection;
}
static void vkCmdClearAttachments(int cb,uint32_t count,const VkClearAttachment *attachment,uint32_t rects,
                                  const VkClearRect *rect){
    assert(cb==7 && count==1 && rects==1 && attachment->aspectMask==2 && rect->layerCount==1 &&
           rect->rect.extent.width==16 && rect->rect.extent.height==16);
    native_clear++;
}
static void clear_image_fragment(const VkExecTarget *t,VkRect2D rect,uint32_t layers,uint32_t aspect){
    assert(t==&ex_live->depth && rect.extent.width==16 && rect.extent.height==16 && layers==1 && aspect==2);
    fragments++;
}
"""

CASES = r"""
int main(void){
    const view_t depth_view={.set_format=1,.format=52};
    view_t authored={.set_levels=1,.set_layers=1,.set_target=1,.target=NVN_TEXTURE_TARGET_2D};
    assert(vk_texture_attachment_base_view(&depth,&authored));
    authored.nlevels=1;authored.nlayers=1;
    assert(vk_texture_attachment_base_view(&depth,&authored));
    authored.base_level=1;
    assert(!vk_texture_attachment_base_view(&depth,&authored));
    authored.base_level=0;authored.min_layer=1;
    assert(!vk_texture_attachment_base_view(&depth,&authored));
    authored.min_layer=0;authored.nlayers=2;
    assert(!vk_texture_attachment_base_view(&depth,&authored));
    authored.nlayers=1;authored.set_dsmode=1;authored.dsmode=1;
    assert(!vk_texture_attachment_base_view(&depth,&authored));
    authored.dsmode=0;authored.target=3;
    assert(!vk_texture_attachment_base_view(&depth,&authored));
    authored.target=NVN_TEXTURE_TARGET_2D;authored.set_format=1;authored.format=63;
    assert(!vk_texture_attachment_base_view(&depth,&authored));
    const view_t color_view={.set_format=1,.format=63};
    VkExec ex={.has_scissor=1,.scissor={0,0,16,16},.pass_layers=1,.cb=7,.target_binding_mask=0x101,
                 .rt={.n=1,.depth=&depth,.color={[0]=&color},.has_dview=1,.dview=depth_view,
                      .has_cview={[0]=1},.cview={[0]=color_view}}};
    ex_live=&ex;
    assert(partial_clear_slot(&ex,&depth_op.h)==8 && partial_clear_slot(&ex,&color_op.h)==0);
    ex.scissor.w=1720;ex.scissor.h=720;
    color.d.width=depth.d.width=1720;color.d.height=depth.d.height=720;
    ex.rt.dview.set_levels=1;ex.rt.dview.base_level=0;ex.rt.dview.nlevels=0;
    ex.rt.dview.set_layers=1;ex.rt.dview.min_layer=0;ex.rt.dview.nlayers=1;
    VkExecTarget native={0};
    assert(clear_whole_target_requested(&ex,0)==&color && partial_clear_slot(&ex,&color_op.h)==-1);
    assert(clear_direct_attachment(&ex,0,&native) && native.image==17 && materializations==1);
    assert(!opened && !reads && !ex.rt_resolved);
    ex.rt.dview.base_level=1;
    assert(!clear_direct_attachment(&ex,0,&native) && materializations==1);
    ex.rt.dview.base_level=0;
    color.d.width=color.d.height=depth.d.width=depth.d.height=128;
    ex.scissor.w=ex.scissor.h=128;
    assert(partial_clear_slot(&ex,&depth_op.h)==-1 && clear_whole_target_requested(&ex,8)==&depth);
    ex.scissor.w=ex.scissor.h=16;
    ex.render_disabled=1;
    assert(partial_clear_slot(&ex,&depth_op.h)==-1);
    ex.render_disabled=0;
    depth_op.p.depth_mask=0;
    assert(partial_clear_slot(&ex,&depth_op.h)==-1);
    depth_op.p.depth_mask=1;
    AttachmentClearLoad load={0};
    assert(prepare_attachment_clear_load(&ex,&depth_op.h,8,&load) && load.pass==&pass &&
           entered==1 && captures==1 && reads==1 && !ex.resource_op && !native_clear);
    attachment_clear_load_cancel(&load);
    assert(!pass.live && releases==1 && !seed.live);
    peers=0;
    assert(!prepare_attachment_clear_load(&ex,&depth_op.h,8,&load) && !pass.live && releases==2);
    peers=1;current=0;
    assert(!prepare_attachment_clear_load(&ex,&depth_op.h,8,&load) && !pass.live && releases==3);
    current=1;typed_capture=0;
    assert(prepare_attachment_clear_load(&ex,&depth_op.h,8,&load) && !pass.live && !seed.live);
    assert(failures==2 && !ex.resource_op && !native_clear);
    typed_capture=-1;
    assert(!prepare_attachment_clear_load(&ex,&depth_op.h,8,&load) && !pass.live &&
           failures==3 && !strcmp(failed_stage,"attachment-load-capture") && !native_clear);
    color_capture=1;typed_capture=1;
    assert(prepare_attachment_clear_load(&ex,&color_op.h,0,&load) && load.pass==&pass &&
           prepared_mask==0x101 && color_seed.live && seed.live && !native_clear);
    unsigned before=releases;
    attachment_clear_load_cancel(&load);
    assert(releases==before+2 && !color_seed.live && !seed.live);

    const VkClearAttachment attachment={.aspectMask=VK_IMAGE_ASPECT_DEPTH_BIT};
    ex.scissor.w=0;
    clear_attachment(&ex,&attachment);
    assert(!opened && !native_clear);
    ex.scissor.w=16;
    clear_attachment(&ex,&attachment);
    assert(opened==1 && native_clear==1 && prepares==1 && writes==1 && !fragments &&
           ex.fb_dirty==256 && ex.recorded==1);
    projection=0;
    clear_attachment(&ex,&attachment);
    assert(opened==1 && native_clear==2 && writes==2 && fragments==1);
    puts("Partial color/D32 clear uses one ordered LOAD receipt; whole/empty, cancellation, unaffected peers and regional fallback passed");
}
"""


def main():
    source = (ROOT / "runtime/vk_exec.c").read_text(encoding="utf-8")
    texture = (ROOT / "runtime/vk_texture.c").read_text(encoding="utf-8")
    op = source[source.index("        AttachmentClearLoad clear_load = {0};"):]
    assert op.index("prepare_attachment_clear_load") < op.index("effect = resource_declare") < op.index("if (clear_load.pass)")
    assert op.index("vk_exec_graphics_pass_activate(ex, clear_load.pass)") < op.index("attachment_clear_load_cancel(&clear_load)")
    declaration = source[source.index("typedef struct {\n    VkExecGraphicsPass *pass;\n    VkTextureAliasRead *seeds[9];"):]
    declaration = declaration[:declaration.index("static void attachment_clear_load_cancel")]
    selected = ("graphics_attachment_seeds_current","capture_attachment_load_seeds","clear_attachment_extent","clear_whole_target_requested",
                "partial_clear_slot","clear_ordinary_attachment","clear_direct_attachment","attachment_clear_load_cancel","prepare_attachment_clear_load","clear_attachment")
    text = STUB + declaration + function(texture,"vk_texture_attachment_base_view") + \
        "\n".join(function(source,n) for n in selected) + CASES
    with tempfile.TemporaryDirectory(prefix="partial-attachment-clear-") as work:
        directory = Path(work)
        cfile = directory / "check.c"
        binary = directory / ("check.exe" if os.name == "nt" else "check")
        cfile.write_text(text, encoding="utf-8")
        subprocess.run([os.environ["CC"],"-std=c11","-Wall","-Wextra","-Werror","-fsanitize=undefined",
                       "-fsanitize-trap=all",str(cfile),"-o",str(binary)], check=True)
        subprocess.run([str(binary)], check=True)

if __name__=="__main__":
    main()


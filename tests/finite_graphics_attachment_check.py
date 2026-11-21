"""Exercise attachment compatibility of a retained finite sampled descriptor publication."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

from present_source_check import ROOT, functions

FIXTURE = r'''
#include <assert.h>
#include <stdint.h>

enum {VK_POOL_TEX=8,VK_POOL_KINDS=4};
typedef uint64_t VkImage, VkImageView;
typedef unsigned VkImageAspectFlags;
enum {VK_IMAGE_ASPECT_DEPTH_BIT=2,VK_IMAGE_ASPECT_STENCIL_BIT=4};
typedef struct { uint64_t scene, ordinal; } RcRegistryPosition;
typedef struct { unsigned mapping; } RcInterval;
typedef struct { int canonical, projected; } VkTextureReadKey;
typedef struct { unsigned selected_kinds; VkTextureReadKey selected; } PoolSlot;
typedef struct { VkImage image,bc_backing; unsigned aspect,level,base_layer,layers,is3d; } VkExecTarget;
typedef struct { VkExecTarget color[8], depth; unsigned pass_ncolor; } VkExecGraphicsPassShape;
typedef VkExecGraphicsPassShape VkExecGraphicsPass;
typedef VkExecGraphicsPassShape VkExec;
typedef struct { unsigned aspect,level0,levels,layer0,layers; } ViewKey;
typedef struct ViewNode { const struct ViewNode *next; ViewKey k; VkImageView view; } ViewNode;
typedef int TextureImagePublication, VkTextureAliasRead, GfxShader;
typedef struct {
    unsigned held,index,finite,kinds;
    RcRegistryPosition read;
    struct {unsigned comparison_absent,count; struct {unsigned slot,kinds;} entries[8];} indices;
} VkExecPoolGrant;
typedef struct { int pool; } VkPipeInfo;
enum { VK_EXEC_PREPARATION_POOL_IMAGE=1,VK_EXEC_PREPARATION_FEEDBACK=2,VK_EXEC_GRAPHICS_UNAVAILABLE=3 };
typedef struct {VkImage image;} VkExecSampledFailure;
typedef struct {
    unsigned owner,reason,result,detail,role,slot;
    VkImage image;
    VkExecSampledFailure sampled;
    const char *stage;
} VkExecPreparationFailure;
typedef struct {
    VkExecPoolGrant *pool_grant;
    VkExecGraphicsPass *pass;
    VkExecPreparationFailure failure;
    uint64_t program_serial;
    struct {const GfxShader *vs,*fs,*ts;} state;
    struct {const VkPipeInfo *info; uint64_t pipeline; const void *revision;} use;
} VkExecGraphicsDraw;
typedef enum { VK_EXEC_ATTACHMENT_READ_DISJOINT, VK_EXEC_ATTACHMENT_READ_PRE_DRAW, VK_EXEC_ATTACHMENT_READ_FRAGMENT } VkExecAttachmentRead;
static int pool_mt, texture_locked, proof_current=1, alias_current=1, publication_current=1;
static int check_count, last_mode;
static unsigned alias_entry;
static VkTextureAliasRead source_alias;
static struct { unsigned klen; } views_by_image={1};
static struct {
    VkImage image[VK_POOL_TEX*VK_POOL_KINDS];
    VkImageView view[VK_POOL_TEX*VK_POOL_KINDS];
    PoolSlot ps[VK_POOL_TEX];
    const TextureImagePublication *publication[1];
} PL;
static ViewNode view_first,view_second;
static VkImage expected_image;
static const RcInterval span={17};

static void vk_texture_lock(void){ assert(pool_mt&&!texture_locked); texture_locked=1; }
static void vk_texture_unlock(void){ assert(pool_mt&&texture_locked); texture_locked=0; }
static void plat_mutex_lock(int *lock){ assert(lock==&pool_mt && !*lock); *lock=1; }
static void plat_mutex_unlock(int *lock){ assert(lock==&pool_mt && *lock && !texture_locked); *lock=0; }
static const ViewNode *nvn_hm_get(const void *table, const VkImage *image) {
    assert(texture_locked&&table==&views_by_image);
    return *image==expected_image? &view_first : 0;
}
static const VkExecGraphicsPassShape *vk_exec_graphics_pass_shape(const VkExecGraphicsPass *pass){return pass;}
static int graphics_pool_identity_locked(const VkExec *ex,const VkExecPoolGrant *grant){
    assert(pool_mt&&ex&&grant&&grant->held&&!grant->index);
    return publication_current;
}
static int graphics_pool_grant_read_current(const VkExec *ex,const VkExecPoolGrant *grant){
    assert(ex);
    assert(pool_mt&&grant->read.scene==13&&grant->read.ordinal==14);
    return alias_current;
}
static void vk_exec_graphics_failure_set(VkExecGraphicsDraw *draw,const VkExecPreparationFailure *failure){
    assert(draw&&!draw->failure.owner&&failure->sampled.image);
    draw->failure=*failure;
}
static VkExecSampledFailure sampled_pool_receipt(const VkExecPreparationFailure *failure,RcRegistryPosition read,
    uint64_t program,uint64_t pipeline,const void *revision,const GfxShader *const shaders[3]) {
    assert(failure->detail==failure->slot&&read.scene==13&&read.ordinal==14&&program==41);
    (void)pipeline;(void)revision;(void)shaders;
    return (VkExecSampledFailure){0};
}
static VkTextureAliasRead *vk_texture_publication_alias_locked(const TextureImagePublication *p,unsigned descriptor,VkImage image){
    assert(texture_locked&&p==PL.publication[0]&&PL.image[descriptor]==image);
    return descriptor==alias_entry? &source_alias : 0;
}
static const RcInterval *vk_texture_alias_read_intervals(VkTextureAliasRead *alias,unsigned *count) {
    assert(texture_locked&&alias==&source_alias);
    *count=1;
    return &span;
}
static const RcInterval *vk_texture_selected_sample_intervals_locked(VkTextureReadKey key,VkImage image,unsigned *count){
    assert(texture_locked&&image);
    (void)key;
    *count=1;
    return &span;
}
static int vk_texture_resolved_receipt_locked(VkTextureReadKey key,VkImage image,void *unused){
    assert(texture_locked&&image&&!unused&&(key.canonical||key.projected));
    return proof_current;
}
static int vk_exec_graphics_pass_attachment_read(const VkExecGraphicsPass *pass,VkImage image,
    const RcInterval *intervals,unsigned count,unsigned slot,VkExecAttachmentRead mode,VkExecPreparationFailure *failure){
    assert(texture_locked&&pass&&image&&intervals==&span&&count==1&&slot<VK_POOL_TEX);
    ++check_count;
    last_mode=mode;
    if (mode==VK_EXEC_ATTACHMENT_READ_FRAGMENT) return 1;
    failure->reason=VK_EXEC_PREPARATION_FEEDBACK;
    return 0;
}
'''

CHECK = r'''
int main(void) {
    const unsigned id=4, kind=1;
    const unsigned entry=kind*VK_POOL_TEX+id;
    int publication=1;
    PL.publication[0]=&publication;
    PL.image[entry]=expected_image=10;
    PL.view[entry]=51;
    PL.image[kind*VK_POOL_TEX+1]=22; // A registered, unselected table entry.
    PL.view[kind*VK_POOL_TEX+1]=66;
    view_second=(ViewNode){.view=51,.k={.aspect=1,.level0=2,.levels=1,.layer0=4,.layers=2}};
    view_first=(ViewNode){.view=50,.k={.aspect=1,.level0=0,.levels=1,.layer0=2,.layers=1},.next=&view_second};
    VkExecPoolGrant grant={.held=1,.kinds=1u<<kind,.finite=1,.read={13,14},
        .indices={.count=1,.entries={{id,1u<<kind}}}};
    VkPipeInfo info={.pool=1};
    VkExecGraphicsPass pass={.pass_ncolor=1,.color={{.image=22,.aspect=1,.level=2,.base_layer=4,.layers=1}}};
    VkExec ex=pass;
    VkExecGraphicsDraw draw={.pool_grant=&grant,.pass=&pass,.use={.info=&info},.program_serial=41};

    // The proof carries only the original finite domain. Other table cells do not block it.
    assert(graphics_load_pool_disjoint(&ex,&draw,0)&&!check_count&&!draw.failure.owner);
    pass.color[0]=(VkExecTarget){.image=10,.aspect=1,.level=2,.base_layer=3,.layers=1};
    assert(graphics_load_pool_disjoint(&ex,&draw,0)&&!draw.failure.owner);
    pass.color[0].base_layer=5;
    assert(!graphics_load_pool_disjoint(&ex,&draw,0));
    assert(draw.failure.detail==id&&draw.failure.slot==0&&draw.failure.role==kind&&draw.failure.sampled.image==10);
    draw.failure=(VkExecPreparationFailure){0};
    pass.color[0].level=3;
    assert(graphics_load_pool_disjoint(&ex,&draw,0));
    pass.color[0].level=2;
    pass.color[0].aspect=2;
    assert(graphics_load_pool_disjoint(&ex,&draw,0)); // Distinct aspect.
    PL.view[entry]=99;
    assert(!graphics_load_pool_disjoint(&ex,&draw,0)); // Unknown selected view on an actual target.
    draw.failure=(VkExecPreparationFailure){0};
    PL.view[entry]=51;

    // Compatibility uses attachments actually open at its issue boundary.
    draw.pass=0;
    ex.color[0]=(VkExecTarget){0}; ex.pass_ncolor=0;
    view_second.k.aspect=2;
    ex.depth=(VkExecTarget){.image=10,.aspect=2,.level=2,.base_layer=5,.layers=1};
    assert(!graphics_load_pool_disjoint(&ex,&draw,0)&&draw.failure.slot==8);
    draw.failure=(VkExecPreparationFailure){0};
    ex.depth.aspect=VK_IMAGE_ASPECT_STENCIL_BIT;
    assert(!graphics_load_pool_disjoint(&ex,&draw,0)&&draw.failure.slot==8);
    draw.failure=(VkExecPreparationFailure){0};
    ex.depth.level=3;
    assert(graphics_load_pool_disjoint(&ex,&draw,0));

    // Exact canonical and recomposed prior pixels can use a GENERAL descriptor and independent LOAD target.
    draw.pass=&pass;
    pass.color[0]=(VkExecTarget){.image=22,.aspect=1,.level=2,.base_layer=4,.layers=1};
    PL.ps[id]=(PoolSlot){.selected_kinds=1u<<kind,.selected={.projected=1}};
    assert(graphics_load_pool_disjoint(&ex,&draw,1)&&check_count==1&&last_mode==VK_EXEC_ATTACHMENT_READ_FRAGMENT);
    proof_current=0;
    assert(!graphics_load_pool_disjoint(&ex,&draw,1)&&last_mode==VK_EXEC_ATTACHMENT_READ_DISJOINT);
    alias_entry=entry;
    draw.failure=(VkExecPreparationFailure){0};
    assert(graphics_load_pool_disjoint(&ex,&draw,1)&&last_mode==VK_EXEC_ATTACHMENT_READ_FRAGMENT);
    alias_current=0;
    assert(!graphics_load_pool_disjoint(&ex,&draw,1));
    alias_current=1;
    publication_current=0;
    assert(!graphics_load_pool_disjoint(&ex,&draw,0));
    publication_current=1;
    grant.indices.comparison_absent=1;
    assert(graphics_load_pool_disjoint(&ex,&draw,1)); // A typed absence uses its own neutral image.
    grant.indices.comparison_absent=0;
    grant.finite=0;
    assert(graphics_load_pool_disjoint(&ex,&draw,0)); // An unknown domain gains no grant here.
    assert(!graphics_load_pool_disjoint(&ex,&draw,1)&&last_mode==VK_EXEC_ATTACHMENT_READ_DISJOINT);
    assert(!pool_mt&&!texture_locked);
}
'''

parser = argparse.ArgumentParser()
parser.add_argument('--draw-source', type=Path, default=ROOT/'runtime/vk_exec_draw.c')
args = parser.parse_args()
draw = args.draw_source.read_text()
assert draw.index('result = graphics_load_peers_prepare(ex, draw)') < draw.index('result = graphics_descriptors_ready(ex, draw, ordered)')
assert 'if (result == VK_EXEC_GRAPHICS_READY && !graphics_load_pool_disjoint(ex, draw, 0))' in draw
cc = Path(os.environ['CC'])
env = dict(os.environ, PATH=str(cc.parent)+os.pathsep+os.environ['PATH'])
with tempfile.TemporaryDirectory(prefix='finite-attachment-') as tmp:
    c = Path(tmp)/'fixture.c'
    exe = Path(tmp)/('fixture.exe' if os.name=='nt' else 'fixture')
    c.write_text(FIXTURE+functions(draw, 'feedback_target', 'graphics_pool_same_attachment',
        'graphics_load_pool_members_disjoint', 'graphics_load_pool_disjoint')+CHECK)
    subprocess.run([str(cc),'-std=gnu11','-Wall','-Wextra','-Werror','-fsanitize=undefined',
                    '-fsanitize-trap=all',str(c),'-o',str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)
print('Finite graphics table: exact original overlap, prior owner, view, attachment class and stale controls passed')

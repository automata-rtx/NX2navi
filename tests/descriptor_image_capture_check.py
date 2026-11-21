"""Check production cached descriptor image metadata and completion-owner capture adapters."""
from pathlib import Path
import os
import subprocess
import tempfile
from bind_recipe_check import function

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ["CC"])
source = (ROOT / "runtime/vk_exec_draw.c").read_text()
fixture = r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
typedef uint64_t VkImage;
typedef uint64_t VkDescSlot;
typedef int TextureImagePublication;
typedef struct {TextureImagePublication *images;} CbRes;
typedef struct {int cls,use,has;uint64_t n;} PbCount;
#define PB_DESC 4
typedef struct {uint8_t ok[4];uint64_t in[4][4];VkDescSlot d[4];VkImage images[4];PbCount cnt[4];} PbEntry;
typedef struct {uint64_t h;uint16_t type;struct {VkImage image;} r;} FmEnt;
typedef struct {VkImage bind_images[4];PbEntry *pb_e;int pb_key_ok,pb_spoiled;uint64_t pb_key_cur[4];PbCount pb_cnt_cur,*pb_cap;CbRes *res;FmEnt *fm;} DrawCtx;
typedef struct {DrawCtx *dc;uint64_t tex[8][128];} VkExec;
typedef struct {int slot,separate,pool,count;uint16_t type;} GfxShaderRes;
typedef struct {int kind,type,stage;uint32_t count;const GfxShaderRes *res;} VkPipeBinding;
typedef struct {uint32_t n_bindings;VkPipeBinding bindings[4];} VkPipeInfo;
enum {GFX_RES_TEX=4,GFX_RES_UBO=5,VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER=6,VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER=7};
enum {NVN_STAGE_FS=0};
typedef struct {GfxShaderRes resources[4];int count;} GfxShader;
static int gfx_shader_resources(const GfxShader *shader,int kind,const GfxShaderRes **out) {
    assert(kind==GFX_RES_TEX);*out=shader->resources;return shader->count;
}
#define MAX_DESC 4
#define NVN_HANDLE_TEX(h) ((int)((h)&0xfffff))
static int nvn_stage_of(int stage) {return stage;}
static int lock_depth,logical_position=2,next_write=5;
static void vk_texture_lock(void) {lock_depth++;}
static void vk_texture_unlock(void) {assert(lock_depth>0);lock_depth--;}
static int vk_texture_binding_current_locked(int id,VkImage image) {
    assert(lock_depth>0);
    return id==1 && image==42 && (!next_write || logical_position<next_write);
}
enum {VK_BR_TEXTURE=1,VK_BR_IMAGE=2,VK_BR_UBO=3};
typedef struct {uint32_t kind,first,count;} VkBindRecipeOp;
typedef struct {uint32_t n_ops;VkBindRecipeOp op[4];} VkBindRecipe;
typedef struct {struct {VkImage image;} texture;} Slot;
typedef struct {const VkBindRecipe *recipe;Slot slot[4];} PreparedBindings;
static struct {int bind_reused,bind_resolved;} vk_exec_frame;
static int pb_mode=1,created,added;
static VkImage captured[4];
static int pb_key(const VkExec *ex,const VkPipeBinding *b,uint32_t k,uint64_t key[4]) {
    (void)ex;(void)b;memset(key,0,32);key[0]=k+1;return 1;
}
static void count_use(DrawCtx *cx,int cls,int use,uint64_t n) {(void)cx;(void)cls;(void)use;(void)n;}
static void rt_fatal(const char *s,...) {fputs(s,stderr);abort();}
static CbRes *cb_res(VkExec *ex) {return ex->dc->res;}
static TextureImagePublication *vk_texture_publication_create(const VkImage *p,size_t n,const TextureImagePublication *old) {
    assert(!p&&!n&&!old);created++;static int object;return &object;
}
static void vk_texture_publication_add_and_use(TextureImagePublication *pub,const VkImage *images,size_t n) {
    assert(pub&&n<=4);memcpy(captured,images,n*sizeof *images);added++;
}
'''
fixture += "\n".join(function(source, name) for name in
                        ("binding_image_current", "binding_images_current", "feedback_bindings_current", "pb_take", "pb_keep",
                         "binding_images_prepared", "binding_images_publish"))
fixture += r'''
int main(void) {
    CbRes owner={0};PbEntry entry={0};DrawCtx cx={.res=&owner,.pb_e=&entry};VkExec ex={.dc=&cx};
    VkPipeBinding binding={.kind=GFX_RES_TEX};VkDescSlot out=0,descriptor=87;
    assert(!pb_take(&ex,&binding,1,0,&out));cx.bind_images[1]=42;pb_keep(&cx,1,&descriptor);
    cx.bind_images[1]=0;assert(pb_take(&ex,&binding,1,0,&out));
    assert(out==87&&cx.bind_images[1]==42); // PB preserves exact image alongside opaque view bytes.
    // A predeclared write can be crossed with unchanged global stamps and the same physical image.
    // Cached descriptor acceptance must still consult its logical source association.
    logical_position=8;assert(!pb_take(&ex,&binding,1,0,&out));assert(!entry.ok[1]);
    next_write=0;pb_keep(&cx,1,&descriptor);assert(pb_take(&ex,&binding,1,0,&out));
    GfxShaderRes sampled={.slot=0};
    VkPipeInfo info={.n_bindings=2,.bindings={{.kind=GFX_RES_UBO,.count=1},
        {.kind=GFX_RES_TEX,.count=2,.res=&sampled}}};
    ex.tex[0][0]=ex.tex[0][1]=1;cx.bind_images[1]=cx.bind_images[2]=42;
    assert(binding_images_current(&ex,&info));
    ex.tex[0][1]=2;assert(!binding_images_current(&ex,&info)); // Same VkImage cannot prove another source current.
    ex.tex[0][1]=1;next_write=5;assert(!binding_images_current(&ex,&info));
    next_write=0;assert(binding_images_current(&ex,&info));assert(!lock_depth);
    GfxShader shader={.resources={{.slot=0,.count=2}},.count=1};
    cx.fm=calloc(8192,sizeof *cx.fm);assert(cx.fm);
    cx.fm[(1ull*0x9E3779B97F4A7C15ull)>>51]=(FmEnt){.h=1,.r={42}};
    assert(feedback_bindings_current(&ex,&shader));
    shader.resources[0].type=4;assert(!feedback_bindings_current(&ex,&shader));
    shader.resources[0].type=0;
    next_write=5;assert(!feedback_bindings_current(&ex,&shader));
    next_write=0;ex.tex[0][1]=2;assert(!feedback_bindings_current(&ex,&shader));
    free(cx.fm);cx.fm=NULL;assert(!lock_depth);
    VkBindRecipe recipe={.n_ops=3,.op={{VK_BR_UBO,0,1},{VK_BR_TEXTURE,1,2},{VK_BR_IMAGE,3,1}}};
    PreparedBindings prepared={.recipe=&recipe,.slot={{{99}},{{41}},{{42}},{{43}}}};
    memset(cx.bind_images,0,sizeof cx.bind_images);binding_images_prepared(&cx,&prepared);
    assert(!cx.bind_images[0]&&cx.bind_images[1]==41&&cx.bind_images[2]==42&&cx.bind_images[3]==43);
    binding_images_publish(&ex,4);binding_images_publish(&ex,4);
    assert(created==1&&added==2&&!memcmp(captured,cx.bind_images,sizeof captured));
    puts("descriptor image capture: PB identity, prepared arrays and single additive completion-owner publication passed");
}
'''
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="descriptor-images-", dir=ROOT / "build/test") as tmp:
    path = Path(tmp) / "images.c"
    path.write_text(fixture)
    exe = path.with_suffix(".exe")
    subprocess.run([str(CC), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fsanitize-trap=all", str(path), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

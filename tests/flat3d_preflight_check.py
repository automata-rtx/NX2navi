"""Exercise production derived-image preflight ordering without launching the game."""
from pathlib import Path
import os
import subprocess
import tempfile
from bind_recipe_check import function

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "runtime/vk_exec_draw.c").read_text()
prefix = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define VK_NULL_HANDLE 0
#define VK_POOL_TEX 8
#define VK_POOL_KINDS 2
#define NVN_HANDLE_TEX(h) ((int)((h)&0xfffff))
enum {GFX_RES_TEX, VK_IMAGE_TYPE_2D=2, VK_IMAGE_TYPE_3D=3};
typedef uint64_t VkImage;
typedef struct {int type,skip,id;VkImage image;} VkTexImage;
typedef struct {int set_layers;unsigned min_layer;} view_t;
typedef struct {uint64_t registration;} VkTextureReadKey;
typedef struct {int type,pool,count,slot,exact;} GfxShaderRes;
typedef struct {GfxShaderRes r[4];int count;} GfxShader;
typedef struct {int flat3d_pool_ready;} DrawCtx;
typedef struct {uint64_t tex[3][128],tex_pool,fence;int cb,in_pass,recorded,bar_dirty;DrawCtx *dc;} VkExec;
static DrawCtx draw;
static VkExec *current;
static VkTexImage images[8];static view_t views[8];
static int present[8],ready[8],metadata_unknown[8],deferred,locks,turns,barriers,copies,lookups,meta_calls;
static int fail_copy,fail_resolve,exact_calls,pool_count=8;static unsigned copied_mask;
typedef struct {int id,ready;} VkTextureHistorical3D;
typedef struct {int selected;} RcVmaDependency;
typedef int VkResult,VkSyncProgressResult,VkTextureSampleStatus;
enum {VK_SUCCESS,VK_NOT_READY,VK_ERROR_DEVICE_LOST,VK_SYNC_PROGRESS_COMPLETE,VK_SYNC_PROGRESS_UNAVAILABLE,
 VK_TEXTURE_SAMPLE_READY,VK_TEXTURE_SAMPLE_ABSENT,VK_TEXTURE_SAMPLE_PREPARE,VK_TEXTURE_SAMPLE_FALLBACK};
static VkTextureHistorical3D history;
static VkTexImage old_child={.type=VK_IMAGE_TYPE_3D,.id=4,.image=903};
static int history_id=-1,history_ready,history_tries,history_copies,history_refusals,history_valid;
static uint64_t scope_token;
static int history_pressure,waiting,pool_mt,pool_locked,pool_retry;
static uint32_t pool_defer[VK_POOL_TEX],pool_ready_generation;
static unsigned invalidated;
static struct {VkImage image[VK_POOL_TEX*VK_POOL_KINDS];int ready,registration_live,nsets,registration_bound,sealed,retry;
 uint32_t kinds;uint64_t registration_pool,registration_scope,cur_fence,set_fence[2];} PL;
static void plat_mutex_lock(int *lock){assert(lock==&pool_mt&&!pool_locked&&!locks);pool_locked=1;}
static void plat_mutex_unlock(int *lock){assert(lock==&pool_mt&&pool_locked);pool_locked=0;}
static void pool_member_invalidate(uint32_t id){assert(pool_locked);invalidated|=1u<<id;}
static void vk_pool_retry_add(int *retry,uint32_t id){assert(retry==&pool_retry&&pool_locked);*retry|=1u<<id;}
static uint16_t pool_kind_type(int k){return k==1?4:3;}
static uint64_t vk_texture_registration_scope_token(void){return scope_token;}
static int vk_exec_in_chunk(void){return 0;}
static int vk_sync_stopping(void){return 0;}
static int vk_device_is_lost(void){return 0;}
static void vk_device_lost_check(VkResult status,const char *label){(void)status;(void)label;assert(0);}
static void vk_exec_registration_gone(VkExec *ex,int logical){
 assert(ex==current&&(logical==history_id||logical==-history_id-1));history_refusals++;
}
static VkTextureSampleStatus vk_texture_history_flat3d_locked(int logical,const VkTexImage **image,VkTextureHistorical3D **plan){
 assert(locks&&scope_token);*image=NULL;
 if(logical!=history_id)return VK_TEXTURE_SAMPLE_FALLBACK;
 if(history_ready){*image=&old_child;return VK_TEXTURE_SAMPLE_READY;}
 if(plan){history=(VkTextureHistorical3D){.id=logical};*plan=&history;}
 return VK_TEXTURE_SAMPLE_PREPARE;
}
static int vk_texture_history_flat3d_current(void *plan){assert(plan==&history&&!locks);return history_valid;}
static VkResult vk_texture_history_flat3d_prepare(void *plan,RcVmaDependency *dependency){
 assert(plan==&history&&!locks&&!history_ready);
 history_tries++;dependency->selected=1;
 if(history_pressure&&!waiting)return VK_NOT_READY;
 history.ready=1;
 return VK_SUCCESS;
}
static VkSyncProgressResult vk_exec_texture_capacity(VkExec *ex,const RcVmaDependency *dependency,VkResult result,
 int(*changed)(void*),void *context,uint64_t *prior){
 assert(ex==current&&dependency->selected&&result==VK_NOT_READY&&history_tries==1&&!locks&&!history_copies&&!*prior);
 assert(!changed(context));waiting++;
 return VK_SYNC_PROGRESS_COMPLETE;
}
static int vk_texture_history_flat3d_record_locked(VkTextureHistorical3D *plan,int cb){
 assert(locks&&plan==&history&&history.ready&&cb==current->cb&&!current->in_pass);
 history_ready=1;history_copies++;
 return 1;
}
static void vk_texture_history_flat3d_cancel(VkTextureHistorical3D *plan){assert(plan==&history&&!locks);}
static int gfx_shader_resources(const GfxShader*s,int kind,const GfxShaderRes**r){assert(kind==GFX_RES_TEX);*r=s->r;return s->count;}
static void vk_texture_lock(void){assert(!locks);locks++;}
static void vk_texture_unlock(void){assert(locks==1);locks--;}
static int vk_texture_pool_count_locked(uint64_t owner){assert(locks&&!owner);return pool_count;}
static int vk_texture_metadata_locked(int id,VkTexImage*m,view_t*v,int*hv){
 assert(locks&&id>=0&&id<8);meta_calls++;*m=images[id];*v=views[id];*hv=1;
 return metadata_unknown[id]?-1:present[id];
}
static const VkTexImage *vk_texture_current_registration_locked(int id,view_t*v,int*hv){
 assert(locks&&id>=0&&id<8);lookups++;*v=views[id];*hv=1;
 if(fail_resolve)deferred=1;
 return present[id]?&images[id]:NULL;
}
static const VkTexImage *vk_texture_current_registration_read_locked(int id,view_t*v,int*hv,VkTextureReadKey *key){
 *key=(VkTextureReadKey){.registration=(uint64_t)id+1};
 return vk_texture_current_registration_locked(id,v,hv);
}
static int registered_slot_locked(int id){return id;}
static const VkTexImage *vk_texture_exact_by_id_locked(int id,view_t*v,int*hv){
 exact_calls++;images[id].skip=0;return vk_texture_current_registration_locked(id,v,hv);
}
static int vk_texture_deferred(void){return deferred;}
static int vk_texture_deferred_take(void){int old=deferred;deferred=0;return old;}
static const VkTexImage *vk_texture_flat3d_selected_locked(const VkTexImage*s,VkTextureReadKey key,unsigned layer,int cb,int required){
 assert(locks&&layer==views[s->id].min_layer);
 assert(!key.registration||key.registration==(uint64_t)s->id+1);
 if(!cb){assert(required==2);if(ready[s->id])return s;deferred=1;return NULL;}
 assert(required==1&&cb==current->cb&&!current->in_pass&&turns&&barriers);
 if(fail_copy){deferred=1;return NULL;}
 if(!ready[s->id]){ready[s->id]=1;copies++;copied_mask|=1u<<s->id;}
 return s;
}
static void vk_exec_turn(void){assert(!locks);turns++;}
static void vk_exec_barrier(VkExec*ex){assert(!locks&&turns);barriers++;ex->in_pass=0;}
static void reset(VkExec*ex){
 memset(ex,0,sizeof *ex);memset(&draw,0,sizeof draw);ex->dc=&draw;ex->cb=77;ex->in_pass=1;current=ex;
 memset(present,0,sizeof present);memset(ready,0,sizeof ready);memset(metadata_unknown,0,sizeof metadata_unknown);
 for(int i=0;i<8;i++){images[i]=(VkTexImage){.type=VK_IMAGE_TYPE_2D,.id=i};views[i]=(view_t){1,0};}
 deferred=locks=turns=barriers=copies=lookups=meta_calls=fail_copy=fail_resolve=exact_calls=0;copied_mask=0;
}
'''
tests = r'''
int main(void){
 VkExec ex;GfxShader shader={.r={{.type=3,.slot=0}},.count=1};
 const GfxShader*shaders[]={&shader,NULL};int stages[]={0,1};
 reset(&ex);assert(flat3d_preflight(&ex,shaders,stages,2));
 assert(!locks&&!meta_calls&&!turns&&!copies&&ex.in_pass); // ordinary bindings leave the pass alone
 shader.r[0].type=4;present[1]=1;ex.tex[0][0]=1;views[1].min_layer=3;
 assert(flat3d_preflight(&ex,shaders,stages,2));
 assert(copies==1&&copied_mask==2&&turns==1&&barriers==1&&!locks&&!deferred&&ex.recorded);
 ex.in_pass=1;assert(flat3d_preflight(&ex,shaders,stages,2));
 assert(copies==1&&turns==1&&ex.in_pass); // prepared CPU copy does not force another boundary
 reset(&ex);present[1]=1;ex.tex[0][0]=1;fail_resolve=1;
 assert(!flat3d_preflight(&ex,shaders,stages,2)&&deferred&&!locks&&!turns&&!copies);
 reset(&ex);present[1]=1;ex.tex[0][0]=1;fail_copy=1;
 assert(!flat3d_preflight(&ex,shaders,stages,2)&&deferred&&!locks&&turns==1&&!copies&&ex.recorded);
 reset(&ex);present[1]=1;ex.tex[0][0]=1;images[1].skip=1;shader.r[0].exact=1;
 assert(flat3d_preflight(&ex,shaders,stages,2)&&exact_calls==1&&copies==1);
 reset(&ex);shader=(GfxShader){.r={{.type=4,.pool=1},{.type=4,.pool=1}},.count=2};
 present[0]=present[3]=present[7]=1;images[7].type=VK_IMAGE_TYPE_3D;
 assert(flat3d_preflight(&ex,shaders,stages,2)&&copied_mask==((1u<<0)|(1u<<3))&&copies==2);
 assert(meta_calls==16&&lookups==4&&!locks); // all reachable slots, duplicate pool reflection scanned once per phase
 assert(draw.flat3d_pool_ready);
 reset(&ex);shader=(GfxShader){.r={{.type=4,.pool=1},{.type=4,.pool=1,.exact=1}},.count=2};
 present[3]=1;images[3].skip=1;
 assert(flat3d_preflight(&ex,shaders,stages,2)&&exact_calls==1&&copies==1&&!images[3].skip);
 assert(meta_calls==32); // ordinary pool reflection cannot suppress a later full-mip requirement
 reset(&ex);shader=(GfxShader){.r={{.type=4,.slot=0}},.count=1};
 present[1]=1;metadata_unknown[1]=1;ex.tex[0][0]=1;
 assert(flat3d_preflight(&ex,shaders,stages,2)&&copies==1); // ambiguous metadata resolves rather than pretending absent
 reset(&ex);shader.r[0].pool=2;
 assert(flat3d_preflight(&ex,shaders,stages,2)&&!meta_calls&&!locks);
 assert(!draw.flat3d_pool_ready);
 reset(&ex);shader.r[0].pool=0;ex.tex[0][0]=4;
 history_id=4;scope_token=19;history_valid=history_pressure=1;
 int needs_order=0;
 assert(flat3d_stage(&ex,&shader,0,0,1,&needs_order)&&needs_order&&!history_tries&&!history_copies&&!lookups);
 assert(flat3d_preflight(&ex,shaders,stages,2)&&history_copies==1&&history_tries==2&&waiting==1&&!lookups);
 assert(flat3d_preflight(&ex,shaders,stages,2)&&history_copies==1&&history_tries==2&&!history_refusals);
 // The same archive now supplies a bindless kind. Only that logical entry is queued for republishing.
 shader.r[0].pool=1;PL.ready=1;PL.registration_scope=19;PL.registration_live=8;PL.kinds=3;
 PL.nsets=2;PL.registration_bound=1;PL.cur_fence=ex.fence=8;PL.set_fence[0]=8;
 PL.image[4]=91;PL.image[VK_POOL_TEX+4]=92;
 assert(flat3d_preflight(&ex,shaders,stages,2)&&history_copies==1&&PL.sealed&&PL.retry);
 assert(PL.image[4]==91&&PL.image[VK_POOL_TEX+4]==92&&invalidated==(1u<<4)&&pool_retry==(1u<<4));
 assert(!pool_locked&&!locks);
 puts("flat3D preflight: ordered copies, pass preservation, deferred retry, exact mip and reachable pool coverage passed");
}
'''
cc = Path(os.environ.get("CC", "C:/Users/Andrew/AppData/Local/Programs/2NX2Navi/toolchain/msys64/ucrt64/bin/clang.exe"))
env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="flat3d-preflight-", dir=ROOT / "build/test") as temp:
    path = Path(temp) / "check.c"
    names = ("flat3d_history_changed", "flat3d_history_prepare", "flat3d_history_pool_changed",
             "flat3d_stage", "flat3d_preflight")
    path.write_text(prefix + "\n".join(function(source, name) for name in names) + tests)
    exe = path.with_suffix(".exe")
    subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                    "-fsanitize-trap=all", str(path), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

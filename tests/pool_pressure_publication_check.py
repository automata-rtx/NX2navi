"""Production dormant-manifest reclamation and safe wanted-descriptor reconstruction, CPU only."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bind_recipe_check import function

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ["CC"])
source = (ROOT / "runtime/vk_exec_draw.c").read_text()
fixture = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "replay_work_trace.h"
#define VK_POOL_TEX 4
#define VK_POOL_KINDS 2
#define VK_POOL_SAMP 1
#define VK_NULL_HANDLE 0
#define POOL_LOG 16
typedef uint64_t VkDescriptorSet;
typedef struct {uint64_t fence;} VkExec;
typedef struct {int alive,retired;} TextureImagePublication;
static struct {
 int nsets,cur,ready,sealed,wanted_reconcile,registration_bound;
 uint64_t cur_fence,set_fence[3],set_synced[3],sets[3],set,logn,last[16],log[16];
 TextureImagePublication *publication[3];
 uint32_t publication_updates,set_prepare_refs[3];unsigned char set_rebuild[3],set_prefix_complete[3];
 unsigned char set_comparison_sampler0[3],set_comparison_bound[3];
 uint64_t samp[VK_POOL_SAMP];
 uint64_t tok[VK_POOL_TEX],frame,dirty_pos;
 uint64_t image[VK_POOL_KINDS*VK_POOL_TEX],view[VK_POOL_KINDS*VK_POOL_TEX];
 int ps_storage[VK_POOL_TEX],*ps;
 uint64_t switches,waits,rewrites,catchup,written;
} PL;
typedef struct {int target_set;} GraphicsPoolRestore;
static GraphicsPoolRestore *graphics_pool_restore;
static uint32_t pool_reclaim_requested,pool_defer[VK_POOL_TEX];
static uint64_t pool_reclaim_sets,pool_reclaim_sweeps,pool_ready_generation,pool_ready_authority;
static struct {uint64_t scene;} pool_order;
static struct {uint64_t generation;} pool_frontier_state;
static int pool_memo[8],pool_member[VK_POOL_TEX],pool_mt,lock_depth;
static struct {uint32_t mask;} pool_retry;
static uint64_t completed=5;static int done_queries,releases,waits,fill_calls,budget,fail_source,deferred;
static _Noreturn void rt_fatal(const char *message){puts(message);abort();}
static uint64_t set_values[3][VK_POOL_KINDS*VK_POOL_TEX+VK_POOL_SAMP];
static void plat_mutex_lock(int *lock){assert(lock==&pool_mt&&!lock_depth);lock_depth++;}
static void plat_mutex_unlock(int *lock){assert(lock==&pool_mt&&lock_depth);lock_depth--;}
static void vk_texture_lock(void){}
static void vk_texture_unlock(void){}
static void pool_association_clear(void){}
static uint64_t vk_sync_fence_done(void){done_queries++;return completed;}
static int vk_texture_publication_has_retired(TextureImagePublication *p){assert(lock_depth&&p->alive);return p->retired;}
static void vk_texture_publication_release(TextureImagePublication *p){assert(lock_depth&&p->alive);p->alive=0;releases++;}
static int pool_current(void){return !PL.wanted_reconcile&&!PL.set_rebuild[PL.cur]&&PL.publication[PL.cur];}
typedef struct {uint64_t (*fence_done)(void);int (*fence_wait)(uint64_t,uint64_t);} GfxBackend;
static int fence_wait(uint64_t value,uint64_t timeout){(void)timeout;waits++;completed=value;return 1;}
static const GfxBackend *gfx_backend(void){static const GfxBackend b={vk_sync_fence_done,fence_wait};return &b;}
static void pool_flush(void){}
static void pool_write_to(VkDescriptorSet set,uint32_t kind,uint32_t id,uint64_t view,uint64_t sampler){
 assert(kind==VK_POOL_KINDS&&id<VK_POOL_SAMP&&!view);
 set_values[set-1][VK_POOL_KINDS*VK_POOL_TEX+id]=sampler;
}
static void pool_entry_to(VkDescriptorSet set,uint32_t entry){
 if(entry<VK_POOL_KINDS*VK_POOL_TEX) set_values[set-1][entry]=PL.view[entry];
 else pool_write_to(set,VK_POOL_KINDS,entry-VK_POOL_KINDS*VK_POOL_TEX,0,PL.samp[entry-VK_POOL_KINDS*VK_POOL_TEX]);
}
static void vk_texture_create_budget_locked(int value){budget=value;}
static int pool_create_budget(void){return 2;}
static uint64_t pool_slot_token(int id){return 100+(uint64_t)id;}
static uint64_t vk_texture_dirty_pos_locked(void){return 19;}
static void vk_texture_groups_unstamp_locked(void){}
static int vk_texture_deferred(void){return deferred;}
static void vk_pool_retry_add(void *p,uint32_t id){assert(p==&pool_retry);pool_retry.mask|=1u<<id;}
static void vk_pool_retry_remove(void *p,uint32_t id){assert(p==&pool_retry);pool_retry.mask&=~(1u<<id);}
static int pool_fill(VkExec *ex,uint32_t id,uint32_t mask,uint64_t token){
 (void)ex;fill_calls++;int failed=token&&(fail_source==1||budget==0);
 for(unsigned k=0;k<VK_POOL_KINDS;k++)if(mask>>k&1){
  unsigned e=k*VK_POOL_TEX+id;PL.image[e]=token&&!failed?1000+id:0;
  PL.view[e]=PL.image[e]?PL.image[e]+100:7;
 }
 return !failed;
}
'''
for name in ("vk_exec_texture_publications_request_reclaim", "vk_exec_texture_publications_reclaim",
             "pool_can_stay", "pool_next_set", "pool_take_set", "pool_restore_comparison_sampler0",
             "pool_catch_up_to", "pool_catch_up", "pool_reconcile_wanted"):
    signature = "void " if name.startswith("vk_") else "static int " if name in {
        "pool_can_stay", "pool_next_set", "pool_take_set"} else "static void "
    selected = function(source[source.index(signature + name + "("):], name)
    if name == "pool_catch_up":
        # The trailing diagnostic log reads unrelated profiling counters; retain all descriptor/cursor operations.
        selected = selected[:selected.index("    if (PL.switches % 3600")] + "}\n"
    fixture += selected
fixture += r'''
int main(void){
 TextureImagePublication pubs[3]={{1,1},{1,1},{1,0}};
 PL.nsets=3;PL.cur=2;PL.set=3;PL.cur_fence=5;PL.ready=1;PL.ps=PL.ps_storage;
 for(int i=0;i<3;i++){PL.sets[i]=i+1;PL.publication[i]=&pubs[i];PL.set_fence[i]=i==1?6:4;}
 pool_ready_generation=99;pool_ready_authority=77;pool_order.scene=5;
 vk_exec_texture_publications_reclaim(7);assert(!done_queries&&!releases); // No pressure: no fence query/scan.
 vk_exec_texture_publications_request_reclaim();vk_exec_texture_publications_reclaim(7);
 assert(releases==1&&!pubs[0].alive&&pubs[1].alive&&pubs[2].alive);
 assert(!PL.publication[0]&&PL.set_rebuild[0]);
 assert(PL.ready&&PL.wanted_reconcile&&!pool_ready_generation&&!pool_ready_authority&&!pool_order.scene);
 assert(pool_reclaim_requested); // In-flight retired manifest remains protected and gets a later opportunity.
 completed=6;vk_exec_texture_publications_reclaim(6);assert(releases==1&&pool_reclaim_requested);
 vk_exec_texture_publications_reclaim(7);assert(releases==2&&!pool_reclaim_requested&&pubs[2].alive);
 VkExec ex={7};uint64_t from=0;assert(pool_take_set(&ex,&from));assert(PL.cur==0&&!waits);
 for(unsigned e=0;e<VK_POOL_KINDS*VK_POOL_TEX;e++)PL.image[e]=PL.view[e]=999;
 fail_source=1;pool_reconcile_wanted(&ex,1,2,0);
 assert(!PL.wanted_reconcile&&pool_retry.mask==3&&budget==-1);
 for(unsigned e=0;e<VK_POOL_KINDS*VK_POOL_TEX;e++)assert(!PL.image[e]&&PL.view[e]==7);
 // Even an unchanged change-log cursor cannot skip overwriting the retired set's raw descriptors.
 PL.logn=9;pool_catch_up(9,9);assert(!PL.set_rebuild[0]&&PL.rewrites==1);
 for(unsigned e=0;e<VK_POOL_KINDS*VK_POOL_TEX;e++)assert(set_values[0][e]==7);
 // A retired private sampler must be replaced even when the shared log has no sampler update.
 PL.samp[0]=71;set_values[0][VK_POOL_KINDS*VK_POOL_TEX]=91;
 PL.set_comparison_sampler0[0]=PL.set_comparison_bound[0]=1;
 pool_catch_up(9,9);
 assert(set_values[0][VK_POOL_KINDS*VK_POOL_TEX]==71&&PL.rewrites==1);
 assert(!PL.set_comparison_sampler0[0]&&!PL.set_comparison_bound[0]);
 // A retry remains live; once preparation succeeds, the new physical image replaces the temporary stand-in.
 fail_source=0;PL.wanted_reconcile=1;pool_reconcile_wanted(&ex,3,2,0);
 assert(!pool_retry.mask&&PL.image[0]==1000&&PL.image[4]==1000);
 assert(!PL.image[2]&&!PL.image[6]); // Shrunk/high slots never revive old descriptors.
 assert(PL.tok[0]==100&&PL.tok[1]==101&&!PL.tok[2]);
 // A single-set backend still takes the full rebuild path after safe completed retirement.
 PL.nsets=1;PL.cur=0;PL.set_rebuild[0]=1;PL.cur_fence=6;
 assert(pool_take_set(&ex,&from));pool_catch_up(9,9);assert(!PL.set_rebuild[0]);
 assert(set_values[0][0]==1100&&set_values[0][4]==1100&&!lock_depth);
 assert(pool_reclaim_sets==2&&pool_reclaim_sweeps==2&&fill_calls>0);
 puts("pool pressure: completed dormant holds release; active/useful generations survive; full descriptor repair preserves retries");
}
'''
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="pool-pressure-", dir=ROOT / "build/test") as temporary:
    path = Path(temporary) / "check.c"
    path.write_text(fixture)
    exe = path.with_suffix(".exe")
    subprocess.run([str(CC), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fsanitize-trap=all", "-I", str(ROOT / "runtime"),
                    str(path), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

"""Exercise actual descriptor history, ring selection and indexed publication integration without a GPU."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bind_recipe_check import function

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ["CC"])
source = Path(os.getenv("POOL_PUBLICATION_DRAW_SOURCE", ROOT / "runtime/vk_exec_draw.c")).read_text()
owner_source = Path(os.getenv("POOL_PUBLICATION_OWNER_SOURCE", ROOT / "runtime/vk_texture_uses.h")).read_text()
fixture = r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "replay_work_trace.h"
#define VK_POOL_KINDS 2
#define VK_POOL_TEX 4
#define RT_TRACE_PERF 1
static unsigned rt_trace=RT_TRACE_PERF;
static uint64_t pool_diag_image_changes[VK_POOL_TEX];
#define VK_POOL_SAMP 2
#define VK_POOL_COMPARISON_BINDING (VK_POOL_KINDS+1)
#define POOL_IMAGE_ENTRIES (VK_POOL_KINDS*VK_POOL_TEX)
#define POOL_DESCRIPTOR_ENTRIES (POOL_IMAGE_ENTRIES+2*VK_POOL_SAMP)
#define POOL_LOG 8
#define POOL_WBATCH 3
#define VK_NULL_HANDLE 0
#define VK_IMAGE_LAYOUT_GENERAL 1
#define VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET 1
#define VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE 1
#define VK_DESCRIPTOR_TYPE_SAMPLER 2
typedef uint64_t VkDescriptorSet;
typedef uint64_t VkImage;
typedef uint64_t VkImageView;
typedef uint64_t VkSampler;
typedef struct {uint64_t fence;} VkExec;
typedef struct {VkSampler sampler;VkImageView imageView;int imageLayout;} VkDescriptorImageInfo;
typedef struct {
 int sType;VkDescriptorSet dstSet;uint32_t dstBinding,dstArrayElement,descriptorCount;int descriptorType;
 const VkDescriptorImageInfo *pImageInfo;
} VkWriteDescriptorSet;
typedef struct VkTextureAliasRead {unsigned refs;} VkTextureAliasRead;
typedef struct VkMemoryImageHold {VkImage image;unsigned refs;int retired;} VkMemoryImageHold;
typedef struct TextureNativeUse {
 struct TextureNativeUse *next;
 VkImage image;
 uint32_t descriptor;
 VkMemoryImageHold *hold;
 VkTextureAliasRead *alias;
} TextureNativeUse;
typedef struct {
 int id,alive,pending_native;uint32_t limit,entry_limit;uint64_t last_batch;
 VkImage image[VK_POOL_KINDS*VK_POOL_TEX];
 TextureNativeUse *native;
} TextureImagePublication;
enum {TEXTURE_USE_LIMIT=32};
static struct {unsigned nodes;} texture_uses;
typedef struct {uint64_t id;TextureNativeUse *native;} TextureUseBatch;
static TextureUseBatch in_flight={.id=13};
static VkMemoryImageHold native_source={.image=9876};
static unsigned native_image_queries, native_retains;
static VkMemoryImageHold *vk_mem_image_hold(VkImage image) {
 native_image_queries++;
 if(image!=native_source.image||native_source.retired)return NULL;
 native_source.refs++;return &native_source;
}
static VkMemoryImageHold *vk_mem_image_hold_retain_exact(VkMemoryImageHold *source,VkImage image) {
 assert(source==&native_source);
 if(source->image!=image||!source->refs)return NULL;
 native_retains++;source->refs++;return source;
}
static int vk_mem_image_hold_retired(VkMemoryImageHold *hold) {assert(hold);return hold->retired;}
static VkTextureAliasRead *vk_texture_alias_read_retain(VkTextureAliasRead *alias) {
 if(alias)alias->refs++;
 return alias;
}
static uint64_t vk_exec_recording_fence(void){return 0;}
static TextureUseBatch *texture_use_active(uint64_t fence,int create){assert(!fence&&!create);return NULL;}
static VkMemoryImageHold *vk_mem_image_hold_retain(VkMemoryImageHold *hold) {
 assert(hold==&native_source&&hold->refs);hold->refs++;return hold;
}
static void vk_mem_image_release(VkMemoryImageHold *hold) {
 assert(hold==&native_source&&hold->refs);hold->refs--;
}
static void vk_texture_alias_read_cancel(VkTextureAliasRead *alias) {
 assert(alias&&alias->refs);alias->refs--;
}
static const TextureImagePublication *pool_transfer_images;
static void texture_native_uses_free(TextureNativeUse **);
static void texture_native_use_batch(TextureUseBatch *,const TextureNativeUse *);
static struct {
 int nsets,cur,all,ready,sealed,publication_exhausted,wanted_reconcile,registration_bound;
 uint64_t completed_generation,authority,cur_fence;
 VkDescriptorSet sets[3],set;
 TextureImagePublication *publication[3];
 uint32_t publication_updates,set_registration_gone[3],set_prepare_refs[3];
 uint8_t set_rebuild[3],set_registration_missing[3],set_prefix_complete[3];
 uint8_t split_samplers,set_comparison_sampler0[3],set_comparison_bound[3];
 uint64_t set_generation[3];
 uint64_t set_fence[3],set_synced[3],logn,last[POOL_DESCRIPTOR_ENTRIES];
 uint32_t log[POOL_LOG];
 VkImage image[VK_POOL_KINDS*VK_POOL_TEX];VkImageView view[VK_POOL_KINDS*VK_POOL_TEX];VkSampler samp[VK_POOL_SAMP],comparison[VK_POOL_SAMP];
 uint64_t switches,waits,catchup,written,rewrites,writes;
} PL;
typedef struct {struct {VkImage image;} image;VkTextureAliasRead *read;} PoolAliasMember;
typedef struct {int target_set;PoolAliasMember *aliases;} GraphicsPoolRestore;
static PoolAliasMember *pool_alias_member(PoolAliasMember *member,uint32_t id) {
 (void)id;return member;
}
static GraphicsPoolRestore *graphics_pool_restore;
enum {POOL_SOURCE_REFUSED=2};
static VkWriteDescriptorSet pool_w[POOL_WBATCH];
static VkDescriptorImageInfo pool_ii[POOL_WBATCH];
static uint32_t pool_nw;
static uint64_t pool_ready_generation,pool_ready_authority;
static int pool_mt,locked,created,released,used,shared=1,current_proof=1,native_writes,updates,changes,waits,commits;
static uint64_t completed=100;
static uint64_t vk_sync_fence_done(void){return completed;}
static int vk_sync_stopping(void){return 0;}
static int vk_device_is_lost(void){return 0;}
static TextureImagePublication objects[32];
static VkImageView native_view[3][VK_POOL_KINDS*VK_POOL_TEX];
static VkSampler native_sampler[3][VK_POOL_SAMP];
static VkDescriptorSet chosen;
static int registration_failures, deferred, prefix_candidate, prefix_requests;
static int vk_texture_deferred(void){return deferred;}
static void vk_exec_registration_gone(VkExec *ex,int slot){assert(ex&&slot==-1&&!locked);registration_failures++;}
static void vk_exec_sample_source_refused(VkExec *ex,int slot){assert(ex&&slot==-1&&!locked);registration_failures+=4;}
static int vk_texture_sample_wait_gpa_locked(VkExec *ex,int slot){
 assert(ex&&slot==0&&!locked);
 if(!prefix_candidate)return 0;
 prefix_requests++;return deferred=1;
}
static void pool_order_publish(void){}
static void vk_texture_lock(void){}
static void vk_texture_unlock(void){}
static void pool_association_clear(void){}
static int vk_exec_shared(void){return shared;}
static void plat_mutex_lock(int *p){assert(p==&pool_mt&&!locked);locked=1;}
static void plat_mutex_unlock(int *p){assert(p==&pool_mt&&locked);locked=0;}
__attribute__((noreturn)) static void rt_fatal(const char *s){fputs(s,stderr);abort();}
static TextureImagePublication *vk_texture_publication_create_indexed(uint32_t limit){
 assert(locked&&limit==VK_POOL_KINDS*VK_POOL_TEX&&created<32);
 TextureImagePublication *p=&objects[created++];p->id=created;p->alive=1;p->limit=p->entry_limit=limit;return p;
}
static void vk_texture_publication_update(TextureImagePublication *p,uint32_t entry,VkImage image){
 assert(locked&&p&&p->alive&&entry<p->limit);updates++;p->pending_native++;
 changes+=p->image[entry]!=image;p->image[entry]=image;
}
static void vk_texture_publication_commit_updates(TextureImagePublication *p){
 // Every ownership update has a corresponding native descriptor write before this commit.
 // This fails if a full queue flush happens after acquiring the next not-yet-queued image.
 assert(locked&&p&&p->alive&&!p->pending_native);commits++;
}
static void vk_texture_publication_release(TextureImagePublication *p){
 assert(locked);
 if(!p)return;
 assert(p->alive&&!p->pending_native);
 p->alive=0;released++;
 texture_native_uses_free(&p->native);
}
static void vk_texture_publication_use(TextureImagePublication *p){
 assert((locked||!shared)&&p&&p->alive);used=p->id;
 for(const TextureNativeUse *n=p->native;n;n=n->next)texture_native_use_batch(&in_flight,n);
}
static VkDescriptorSet pool_sync_draw_resolve(VkExec *ex){(void)ex;assert(!locked);return chosen;}
static struct {int device;} ctx;
static typeof(ctx) *vk_ctx(void){return &ctx;}
static void vkUpdateDescriptorSets(int device,uint32_t count,const VkWriteDescriptorSet *w,uint32_t copies,const void *copy){
 (void)device;assert(locked&&!copies&&!copy);
 for(uint32_t i=0;i<count;i++){
  unsigned set=(unsigned)w[i].dstSet-1;assert(set<(unsigned)PL.nsets&&w[i].descriptorCount==1);native_writes++;
  if(w[i].dstBinding<VK_POOL_KINDS){
   TextureImagePublication *p=PL.publication[set];assert(p&&p->alive&&p->pending_native>0);
   p->pending_native--;
   native_view[set][w[i].dstBinding*VK_POOL_TEX+w[i].dstArrayElement]=w[i].pImageInfo->imageView;
  }else native_sampler[set][w[i].dstArrayElement]=w[i].pImageInfo->sampler;
 }
}
static int pool_current(void){return current_proof;}
typedef struct {uint64_t(*fence_done)(void);int(*fence_wait)(uint64_t,uint64_t);} GfxBackend;
static uint64_t fence_done(void){return completed;}
static int fence_wait(uint64_t f,uint64_t timeout){assert(timeout==UINT64_MAX);waits++;completed=f;return 1;}
static const GfxBackend *gfx_backend(void){static const GfxBackend b={fence_done,fence_wait};return &b;}
'''
for name in ("texture_native_uses_free", "texture_native_use_batch", "vk_texture_publication_alias_locked",
             "texture_publication_native_retain",
             "vk_texture_publication_retain_alias_locked", "vk_texture_publication_copy_native_locked"):
    fixture += "\n" + function(owner_source, name)
for name in ("pool_sampler_descriptors", "pool_flush", "pool_write_to", "pool_write", "pool_set_image", "pool_entry_to",
             "pool_can_stay", "pool_next_set", "pool_take_set",
             "vk_exec_pool_split", "pool_restore_comparison_sampler0", "pool_catch_up_to", "pool_catch_up",
             "pool_sync_complete", "pool_sync_draw", "vk_exec_texture_publications_clear"):
    prefix = "void " if name == "vk_exec_texture_publications_clear" else "int " if name == "vk_exec_pool_split" else "static "
    if prefix == "static ":
        prefix += "VkDescriptorSet " if name == "pool_sync_draw" else "int " if name in {
            "pool_can_stay", "pool_next_set", "pool_take_set"} else "void "
    if name == "pool_sampler_descriptors":
        prefix = "static uint32_t "
    signature = prefix + name + ("(void) {" if name == "vk_exec_texture_publications_clear" else "(")
    selected = function(source[source.index(signature):], name)
    if name == "pool_catch_up":
        selected = selected[:selected.index("    if (PL.switches % 3600")] + "}\n"
    fixture += selected
fixture += r'''
static void assert_set(int set){
 TextureImagePublication *p=PL.publication[set];assert(p&&p->alive);
 for(unsigned e=0;e<VK_POOL_KINDS*VK_POOL_TEX;e++)assert(p->image[e]==PL.image[e]&&native_view[set][e]==PL.view[e]);
 for(unsigned s=0;s<VK_POOL_SAMP;s++)assert(native_sampler[set][s]==PL.samp[s]);
}
int main(void){
 PL.nsets=3;PL.cur=0;PL.set=1;PL.cur_fence=1;PL.authority=7;
 for(int i=0;i<3;i++)PL.sets[i]=i+1;
 plat_mutex_lock(&pool_mt);
 // Initial all-set writes have no log history; each native set still acquires its exact owner.
 PL.all=1;
 for(unsigned k=0;k<VK_POOL_KINDS;k++)for(unsigned i=0;i<VK_POOL_TEX;i++)pool_set_image(k,i,100+k*10+i,1000+k*10+i);
 for(unsigned s=0;s<VK_POOL_SAMP;s++){PL.samp[s]=700+s;pool_write(VK_POOL_KINDS,s,0,PL.samp[s]);}
 PL.all=0;pool_flush();assert(created==3&&PL.logn==0&&!PL.publication_updates);
 for(int i=0;i<3;i++)assert_set(i);
 TextureImagePublication *owner=PL.publication[0];int before=updates,changed=changes,native=native_writes;
 pool_sync_complete();pool_sync_complete();assert(created==3&&!released&&PL.publication[0]==owner&&updates==before);
 uint64_t identity_serial=pool_diag_image_changes[0];
 pool_set_image(0,0,PL.image[0],PL.view[0]);assert(!pool_nw&&updates==before&&native_writes==native);
 assert(pool_diag_image_changes[0]==identity_serial);
 // A view-only change writes Vulkan but retains the same exact image hold and owner.
 pool_set_image(0,0,PL.image[0],9000);pool_flush();
 assert(pool_diag_image_changes[0]==identity_serial+1);
 assert(changes==changed&&created==3&&!released&&PL.publication[0]==owner&&native_view[0][0]==9000);
 assert(native_view[1][0]==1000&&PL.publication[1]->image[0]==100);
 // Same raw view, different physical image must enter the history and update ownership.
 uint64_t cursor=PL.logn;pool_set_image(0,0,501,9000);pool_flush();
 assert(pool_diag_image_changes[0]==identity_serial+2);
 assert(PL.logn==cursor+1&&changes==changed+1&&owner->image[0]==501&&native_view[0][0]==9000);
 // Repeated writes to one slot deduplicate at catch-up; the target receives the final association.
 pool_set_image(0,0,502,9001);pool_set_image(0,0,503,9002);pool_flush();
 PL.cur=1;PL.set=2;before=updates;pool_catch_up(0,PL.logn);
 assert(updates==before+1&&PL.publication[1]->image[0]==503);assert_set(1);
 // Lost history rebuilds descriptors AND exact ownership, including both image kinds and samplers.
 PL.cur=0;PL.set=1;
 for(unsigned n=0;n<POOL_LOG+3;n++)pool_set_image(n%2,n%VK_POOL_TEX,800+n,1800+n);
 pool_flush();PL.samp[1]=999;pool_write(VK_POOL_KINDS,1,0,999);pool_flush();
 PL.cur=2;PL.set=3;before=updates;pool_catch_up(0,PL.logn);
 assert(PL.rewrites==1&&updates==before+VK_POOL_KINDS*VK_POOL_TEX);assert_set(2);
 // Explicit rebuild repairs a reclaimed set even with an unchanged log cursor.
 vk_texture_publication_release(PL.publication[2]);PL.publication[2]=NULL;PL.set_rebuild[2]=1;
 memset(native_view[2],0,sizeof native_view[2]);pool_catch_up(PL.logn,PL.logn);
 assert(!PL.set_rebuild[2]&&created==4&&released==1);assert_set(2);
 // Sealed/current reuse is read-only; stale proof rotates and waits before any set update.
 VkExec ex={.fence=20};PL.cur=0;PL.set=1;PL.cur_fence=20;PL.set_fence[0]=20;PL.set_fence[1]=10;completed=9;
 assert(vk_exec_pool_split(20)&&PL.sealed);uint64_t from=999;before=updates;
 assert(!pool_take_set(&ex,&from)&&PL.cur==0&&PL.sealed&&updates==before&&!waits);
 current_proof=0;assert(pool_take_set(&ex,&from)&&PL.cur==1&&!PL.sealed&&waits==1);
 pool_catch_up(from,PL.logn);assert_set(1);pool_sync_complete();
 plat_mutex_unlock(&pool_mt);
 // Binding the older ring set uses that set's owner, never PL.cur's owner.
 chosen=1;assert(pool_sync_draw(&ex)==1&&used==PL.publication[0]->id&&!locked);
 deferred=1;
 chosen=3;assert(!pool_sync_draw(&ex)&&used==PL.publication[0]->id&&!registration_failures);
 deferred=0;
 assert(!pool_sync_draw(&ex)&&used==PL.publication[0]->id&&!registration_failures); // No current scene owner.
 PL.set_fence[2]=ex.fence;
 PL.set_registration_gone[2]=1;
 chosen=3;assert(!pool_sync_draw(&ex)&&used==PL.publication[0]->id&&registration_failures==1&&!locked);
 PL.set_registration_missing[2]=POOL_SOURCE_REFUSED;
 assert(!pool_sync_draw(&ex)&&registration_failures==5&&!locked);
 prefix_candidate=1;
 assert(!pool_sync_draw(&ex)&&registration_failures==5&&prefix_requests==1&&deferred&&!locked);
 assert(used==PL.publication[0]->id); // Pending writes no descriptor-set use into this scene.
 deferred=0;
 PL.set_registration_gone[2]=0;
 chosen=3;assert(pool_sync_draw(&ex)==3&&used==PL.publication[2]->id&&!locked);
 vk_exec_texture_publications_clear();assert(released==created&&!pool_ready_generation&&!pool_ready_authority&&!locked);
 for(int i=0;i<3;i++)assert(!PL.publication[i]);
 // A missing owner on a genuinely empty set is created once; completion needs no full scan.
 plat_mutex_lock(&pool_mt);PL.cur=0;pool_sync_complete();before=updates;pool_sync_complete();
 assert(updates==before&&created==5);plat_mutex_unlock(&pool_mt);vk_exec_texture_publications_clear();assert(released==created&&commits>0);
 // Carry the admitted cell's actual native or reconstructed lease across the ring. The
 // source is retired from lookup, so obtaining a new physical hold would fail.
 plat_mutex_lock(&pool_mt);
 PL.cur=0;PL.set=PL.sets[0];PL.logn=0;
 pool_set_image(0,0,native_source.image,401);
 assert(texture_publication_native_retain(PL.publication[0],0,native_source.image,NULL,NULL));
 VkTextureAliasRead reconstructed={.refs=1};
 PoolAliasMember alias_member={.image={8765},.read=&reconstructed};
 GraphicsPoolRestore requirement={.aliases=&alias_member};
 graphics_pool_restore=&requirement;
 pool_set_image(0,1,alias_member.image.image,402);
 graphics_pool_restore=NULL;
 pool_flush();
 assert(native_source.refs==1&&native_image_queries==1&&reconstructed.refs==2);
 native_source.retired=1;
 pool_transfer_images=PL.publication[0];
 PL.cur=1;PL.set=PL.sets[1];
 pool_catch_up(0,PL.logn);
 assert(PL.publication[1]->image[0]==native_source.image&&native_view[1][0]==401);
 assert(PL.publication[1]->image[1]==8765&&native_view[1][1]==402);
 assert(native_source.refs==2&&native_retains==1&&native_image_queries==1&&reconstructed.refs==3);
 assert(!vk_texture_publication_copy_native_locked(PL.publication[1],pool_transfer_images,1,native_source.image));
 vk_texture_publication_release(PL.publication[0]);PL.publication[0]=NULL;
 pool_transfer_images=NULL;
 assert(native_source.refs==1&&reconstructed.refs==2);
 plat_mutex_unlock(&pool_mt);
 PL.set_fence[1]=ex.fence;
 chosen=2;assert(pool_sync_draw(&ex)==2&&used==PL.publication[1]->id);
 vk_exec_texture_publications_clear();
 assert(native_source.refs==1&&reconstructed.refs==2&&texture_uses.nodes==2&&released==created);
 texture_native_uses_free(&in_flight.native);
 assert(!texture_uses.nodes&&!native_source.refs&&reconstructed.refs==1);
 puts("pool publication: initial all-set ownership, stable view-only updates, image identity changes, history loss, flush commits, sealed rotation and teardown passed");
}
'''
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="pool-publication-", dir=ROOT / "build/test") as tmp:
    path = Path(tmp) / "publication.c"
    path.write_text(fixture)
    exe = path.with_suffix(".exe")
    subprocess.run([str(CC), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fsanitize-trap=all", "-I", str(ROOT / "runtime"),
                    str(path), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)


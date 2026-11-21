"""Exercise production upload checkpoints, batch retirement and command-buffer ownership without a GPU."""
from pathlib import Path
import os
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "runtime/vk_memory.c").read_text()


def function(name):
    match = re.search(r"^(?:static )?[^\n]*\b" + name + r"\([^;]*?\)\s*\{", SOURCE, re.M)
    assert match, name
    begin = SOURCE.index("{", match.start())
    depth, end = 1, begin + 1
    while depth:
        depth += (SOURCE[end] == "{") - (SOURCE[end] == "}")
        end += 1
    return SOURCE[match.start():end]


PREFIX = r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#define MB (1024ull*1024ull)
typedef uint64_t VkDeviceSize,VkBuffer,VmaAllocation,VkFence,VkCommandBuffer,VkImage,VkImageView;
typedef int VkResult;
typedef struct { int device,gfx_family,gfx_queue,vma,timeline; } VkCtx;
typedef struct { int sType,flags,queueFamilyIndex; } VkCommandPoolCreateInfo;
typedef struct { int sType,commandPool,level,commandBufferCount; } VkCommandBufferAllocateInfo;
typedef struct { int sType; } VkFenceCreateInfo;
typedef struct { int sType,flags; } VkCommandBufferBeginInfo;
typedef struct { int sType,commandBufferCount; VkCommandBuffer *pCommandBuffers; } VkSubmitInfo;
typedef struct { uint64_t amount[3]; } Vec;
typedef struct { struct {Vec total;} sample; } RcBudgetSnapshot;
typedef struct { void *pMappedData; } VmaAllocationInfo;
typedef struct { VkDeviceSize size; VkBuffer buffer; VmaAllocation allocation; void *map; int from_cache; } VkPreparedStaging;
enum {VK_SUCCESS=0,VK_NOT_READY=1,VK_TRUE=1,RC_BUDGET_STAGING=2,
 VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO=3,VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
 VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,VK_STRUCTURE_TYPE_SUBMIT_INFO,
 VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,VK_COMMAND_BUFFER_LEVEL_PRIMARY,VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
 VK_CRUMB_UPLOAD,VK_CRUMB_UPLOAD_END,VK_TS_FLUSH};
enum {VK_TIMEOUT=90,VK_ERROR_DEVICE_LOST=-4,VK_NULL_HANDLE=0};
#define VK_WHOLE_SIZE UINT64_MAX
typedef struct {VkBuffer buf;VmaAllocation alloc;VkDeviceSize size;} Staging;
typedef struct {VkBuffer buf;VmaAllocation alloc;uint8_t *map;} StgChunk;
typedef struct {VkCommandBuffer cb;VkFence fence;uint64_t seq;Staging *stg;int nstg,cap;StgChunk *chunk;int nchunk,cap_chunk;VkDeviceSize chunk_used,staging_bytes;} Batch;
#define STG_CACHE 24
#define STG_CACHE_BYTES (192*MB)
#define ARENA_KEEP 8
#define ARENA_CHUNK (8*MB)
#define UPLOAD_BATCH_MIN (16*MB)
#define UPLOAD_BATCH_MAX (64*MB)
#define UPLOAD_INFLIGHT_MIN (64*MB)
#define UPLOAD_INFLIGHT_MAX (256*MB)
static Staging stg_cache[STG_CACHE];static int stg_ncache;static VkDeviceSize stg_cache_bytes;
static StgChunk arena_free[ARENA_KEEP];static int narena_free;
static Batch open_b,*inflight,*spare;static int open_active,ninflight,cap_inflight,nspare,cap_spare;
static uint64_t next_seq,done_seq,submitted_seq,inflight_staging_bytes,upload_checkpoints,upload_pressure_waits;
static uint32_t upload_depth;
static uint64_t upload_batch_window=UPLOAD_BATCH_MAX,upload_inflight_window=UPLOAD_INFLIGHT_MAX;
static int cmd_lock,cmd_once,cmd_pool,lock_depth;
static VkCtx context={1,1,1,1,1};static uint64_t target=512*MB,handle,submitted[4096],signaled[4096],cb_open[4096];
static unsigned submits,waits,destroyed;static uint64_t maximum;
enum {DEAD_IMAGE=1,DEAD_IMAGE_VMA,DEAD_VIEW,DEAD_BUFFER_VMA,DEAD_HOST,DEAD_UPLOAD_CALLBACK};
enum {REC_VIEW,REC_BUFFER,REC_IMAGE,RC_ADMISSION_OK=0};
typedef struct OwnedImageView {VkImageView view;struct OwnedImageView *next;} OwnedImageView;
typedef struct {unsigned readers;OwnedImageView *views;void *pool;} VkMemoryImageHold;
static uint64_t reclaim_pending_bytes;
typedef struct {int kind;uint64_t handle;VmaAllocation alloc;uint64_t seq,tl,reclaim_bytes;void(*destroyed)(void*);void*context;
                VkMemoryImageHold *image_hold;} Dead;
static Dead *dead;static int ndead,cap_dead,npwait;static uint64_t scene_completed;
static unsigned retired_buffers,upload_dead,scene_dead,upload_callback;
static void rt_fatal(const char *fmt,...){fprintf(stderr,"%s\n",fmt);abort();}
static VkCtx *vk_ctx(void){return &context;}
static void cmd_init(void *p){(void)p;}
static void resource_core_budget_progress(void){}
static void plat_once(int *o,void(*f)(void*),void*p){if(!*o){*o=1;f(p);}}
static void plat_rmutex_lock(int *l){(void)l;lock_depth++;}
static void plat_rmutex_unlock(int *l){(void)l;assert(lock_depth);lock_depth--;}
static uint64_t kern_now_ns(void){return 1;}
static int vk_device_is_lost(void){return 0;}
static int stopped;
static int vk_sync_stopping(void){return stopped;}
static void vk_device_lost_check(VkResult r,const char*s){(void)r;(void)s;}
static void vk_texstat_time(int n,uint64_t t){(void)n;(void)t;}
static void vk_queue_lock(void){}
static void vk_queue_unlock(void){}
static void vk_crumb(VkCommandBuffer cb,int kind,uint64_t seq){(void)cb;(void)kind;(void)seq;}
static int vkCreateCommandPool(int d,const VkCommandPoolCreateInfo*i,void*a,int*p){(void)d;(void)i;(void)a;*p=1;return 0;}
static int vkAllocateCommandBuffers(int d,const VkCommandBufferAllocateInfo*i,VkCommandBuffer*p){(void)d;(void)i;*p=++handle;return 0;}
static int vkCreateFence(int d,const VkFenceCreateInfo*i,void*a,VkFence*p){(void)d;(void)i;(void)a;*p=++handle;return 0;}
static int vkResetCommandBuffer(VkCommandBuffer cb,int flags){(void)flags;assert(!cb_open[cb]);return 0;}
static int vkResetFences(int d,int n,const VkFence*f){(void)d;(void)n;submitted[*f]=signaled[*f]=0;return 0;}
static int vkBeginCommandBuffer(VkCommandBuffer cb,const VkCommandBufferBeginInfo*i){(void)i;assert(!cb_open[cb]);cb_open[cb]=1;return 0;}
static int vkEndCommandBuffer(VkCommandBuffer cb){assert(!upload_depth&&cb_open[cb]);cb_open[cb]=0;return 0;}
static int vkQueueSubmit(int q,int n,const VkSubmitInfo*s,VkFence f){(void)q;(void)n;assert(!cb_open[*s->pCommandBuffers]);submitted[f]=1;submits++;return 0;}
static int vkGetFenceStatus(int d,VkFence f){(void)d;return signaled[f]?VK_SUCCESS:VK_NOT_READY;}
static int vkWaitForFences(int d,int n,const VkFence*f,int all,uint64_t timeout){(void)d;(void)n;(void)all;(void)timeout;assert(lock_depth&&submitted[*f]);signaled[*f]=1;waits++;return 0;}
static void vmaFlushAllocation(int v,VmaAllocation a,uint64_t o,uint64_t n){(void)v;(void)a;(void)o;(void)n;}
static void vmaGetAllocationInfo(int v,VmaAllocation allocation,VmaAllocationInfo *info){
  assert(v==context.vma&&allocation&&lock_depth);
  info->pMappedData=(void *)(uintptr_t)allocation;
}
static void resource_core_vma_destroy_buffer(int v,VkBuffer b,VmaAllocation a){
  assert(v==context.vma&&a);
  destroyed++;
  if(b==700)upload_dead++;
  if(b==701)scene_dead++;
}
static void vkDestroyImage(int device,VkImage image,void *allocator){(void)device;(void)image;(void)allocator;abort();}
static void vkDestroyImageView(int device,VkImageView view,void *allocator){(void)device;(void)view;(void)allocator;abort();}
static void resource_core_vma_destroy_image(int v,VkImage image,VmaAllocation a){(void)v;(void)image;(void)a;abort();}
static void image_placement_remove(VkMemoryImageHold *hold){(void)hold;abort();}
static void destroy_rec(VkCtx *c,void *pool){(void)c;(void)pool;abort();}
static int resource_core_vma_retire(VmaAllocation a){assert(a);retired_buffers++;return RC_ADMISSION_OK;}
static int vkGetSemaphoreCounterValue(int d,int s,uint64_t*v){(void)d;(void)s;*v=scene_completed;return 0;}
static void pool_wait_collect(uint64_t tl){(void)tl;}
static void rec_check_handle(int kind,uint64_t h,int gone,uint64_t tl,uint64_t now){(void)kind;(void)h;(void)gone;(void)tl;(void)now;}
static void finish_upload_owner(void *owner){
  assert(owner==&upload_callback && !lock_depth && !upload_depth);
  ++upload_callback;
}
static void resource_core_budget_snapshot(RcBudgetSnapshot*b){*b=(RcBudgetSnapshot){0};b->sample.total.amount[RC_BUDGET_STAGING]=target;}
static void grow(void **p,int *cap,size_t size){*cap=*cap?*cap*2:16;*p=realloc(*p,*cap*size);assert(*p);}
static uint64_t vk_upload_flush(int wait);
'''

MAIN = r'''
static void stage_named(uint64_t n,VkBuffer buffer){
 assert(upload_depth && open_active && cb_open[open_b.cb]);
 if(open_b.nstg==open_b.cap)grow((void**)&open_b.stg,&open_b.cap,sizeof *open_b.stg);
 open_b.stg[open_b.nstg++]=(Staging){buffer,buffer,n};open_b.staging_bytes+=n;
 uint64_t total=open_b.staging_bytes+inflight_staging_bytes;if(total>maximum)maximum=total;
}
static void stage(uint64_t n){stage_named(n,1);}
static void exact_compatible_supplier(void){
 stg_ncache=stg_cache_bytes=0;
 unsigned waited=waits;
 vk_upload_begin();stage_named(2*MB,41);vk_upload_end();
 uint64_t unrelated=vk_upload_flush(0);
 vk_upload_begin();stage_named(512,42);vk_upload_end();
 uint64_t compatible=vk_upload_flush(0);
 vk_upload_begin();stage_named(256,43);vk_upload_end();
 VkPreparedStaging picked={0};
 assert(compatible>unrelated && vk_staging_prepare_supplier(256,0)==compatible);
 assert(!vk_staging_prepare_supplier(256,compatible)); // The next batch is still owned by its recorder.
 assert(!vk_staging_prepare_wait(compatible+1) && waits==waited);
 assert(!vk_staging_prepare_reuse(256,&picked) && !picked.buffer && waits==waited);
 stopped=1;
 assert(!vk_staging_prepare_wait(compatible) && waits==waited);
 stopped=0;
 assert(ninflight==2&&inflight[0].seq==unrelated);
 signaled[inflight[0].fence]=1;
 assert(vk_staging_prepare_wait(compatible) && done_seq==compatible && waits==waited+1);
 vk_upload_begin();
 assert(!vk_staging_prepare_reuse(256,&picked) && !picked.buffer);
 vk_upload_end();
 assert(vk_staging_prepare_reuse(256,&picked) && picked.buffer==42 && picked.size==512 &&
        picked.from_cache && picked.map==(void *)(uintptr_t)42 && waits==waited+1);
 assert(!vk_staging_prepare_reuse(256,&(VkPreparedStaging){0})); // Claim transferred the one compatible buffer.
 vk_upload_flush(1);

 stg_ncache=STG_CACHE-1;
 stg_cache_bytes=stg_ncache*MB;
 for(int i=0;i<stg_ncache;i++)stg_cache[i]=(Staging){50,50,MB};
 vk_upload_begin();stage_named(2*MB,44);vk_upload_end();
 unrelated=vk_upload_flush(0);
 vk_upload_begin();stage_named(512,45);vk_upload_end();
 compatible=vk_upload_flush(0);
 assert(compatible>unrelated && !vk_staging_prepare_supplier(256,0)); // It would be destroyed, not cached.
 vk_upload_flush(1);
 assert(!vk_staging_prepare_reuse(256,&(VkPreparedStaging){0}) && !lock_depth);
}
int main(void){
 VkCommandBuffer outer=vk_upload_begin(),inner=vk_upload_begin();assert(outer==inner);
 assert(vk_upload_recording_sequence(outer)==next_seq&&!vk_upload_recording_sequence(outer+1000));
 stage(64*MB);vk_upload_end();assert(!submits&&upload_depth==1&&cb_open[outer]);
 vk_upload_end();assert(submits==1&&!upload_depth&&!open_active&&!lock_depth&&inflight_staging_bytes==64*MB);
 assert(!vk_upload_recording_sequence(outer));
 // External readback allocation has no upload lock; a pressure drain acquires its own and waits only submitted work.
 upload_drain_submitted();assert(!lock_depth&&!ninflight);
 outer=vk_upload_begin();stage(64*MB);vk_upload_end();
 outer=vk_upload_begin();stage(8*MB);upload_drain_submitted();
 assert(upload_depth==1&&lock_depth==1&&cb_open[outer]&&open_b.staging_bytes==8*MB);vk_upload_end();
 // A long preparation stream progresses without a scene submission and never reuses unsignaled staging.
 for(int i=0;i<400;i++){vk_upload_begin();stage(8*MB);vk_upload_end();assert(inflight_staging_bytes<=256*MB);}
 assert(waits&&upload_pressure_waits&&destroyed&&upload_checkpoints>40&&maximum<=320*MB&&stg_cache_bytes<=STG_CACHE_BYTES);
 vk_upload_flush(1);assert(!ninflight&&!inflight_staging_bytes&&done_seq==submitted_seq);
 // Spare command-buffer bookkeeping starts with zero owned staging, not its previous batch's count.
 outer=vk_upload_begin();assert(!open_b.staging_bytes&&!open_b.nstg&&!open_b.nchunk);vk_upload_end();
 assert(open_active&&!vk_upload_recording_sequence(outer));
 // A shrinking shared target checkpoints at16MiB and drains previously submitted backlog down to64MiB.
 target=32*MB;
 for(int i=0;i<20;i++){vk_upload_begin();stage(16*MB);vk_upload_end();assert(inflight_staging_bytes<=64*MB);}
 // One indivisible operation may exceed the window. It completes recording first, then its true fence retires it.
 unsigned before=waits;outer=vk_upload_begin();stage(512*MB);assert(cb_open[outer]);vk_upload_end();
 assert(waits>before&&!inflight_staging_bytes&&!open_active);
 vk_upload_flush(1);assert(!lock_depth&&!upload_depth);
 // Upload-only retirement waits for its true upload fence, without borrowing the unrelated scene timeline.
 outer=vk_upload_begin();stage(8*MB);uint64_t last=next_seq;
 vk_upload_defer_buffer(700,700,last);assert(retired_buffers==1&&ndead==1&&dead[0].seq==last&&!dead[0].tl);
 if(ndead==cap_dead)grow((void**)&dead,&cap_dead,sizeof *dead);
 dead[ndead++]=(Dead){.kind=DEAD_BUFFER_VMA,.handle=701,.alloc=701,.seq=last,.tl=99};
 vk_upload_defer_release(last,finish_upload_owner,&upload_callback);
 collect_locked(&context);assert(!upload_dead&&!scene_dead&&!upload_callback&&ndead==3&&cb_open[outer]);
 vk_mem_collect();assert(!upload_callback&&ndead==3); // The upload is unhanded while its recorder still owns it.
 vk_upload_end();vk_upload_flush(0);assert(!upload_dead&&!scene_dead&&!upload_callback);
 vk_mem_collect();assert(!upload_callback&&ndead==3); // Submitted still means its exact upload fence may be pending.
 upload_drain_submitted();assert(upload_dead==1&&!scene_dead&&!upload_callback&&ndead==2&&done_seq==last);
 vk_mem_collect();assert(upload_callback==1&&!scene_dead&&ndead==1);
 vk_mem_collect();assert(upload_callback==1&&!scene_dead&&ndead==1);
 scene_completed=99;collect_locked(&context);assert(scene_dead==1&&!ndead);
 exact_compatible_supplier();
 for(int i=0;i<nspare;i++){free(spare[i].stg);free(spare[i].chunk);}free(spare);free(inflight);
 free(dead);
 puts("upload staging: exact handed compatible supplier, cached claim, cancellation and bounded retirement passed");
}
'''


def main():
    code = PREFIX + function("staging_cache_room") + function("free_staging") + function("retire")
    code += function("destroy_dead") + function("collect_locked_out") + function("collect_locked")
    for name in ("upload_wait_oldest_locked", "upload_checkpoint_locked", "upload_drain_submitted", "vk_upload_begin", "vk_upload_end", "vk_upload_flush", "vk_upload_defer_buffer", "vk_upload_recording_sequence"):
        # Keep a private linkage declaration in the fixture without changing the actual function body.
        text = function(name)
        if name == "vk_upload_flush":
            text = "static " + text
        code += "\n" + text
    for name in ("staging_size_fits", "staging_cache_take", "staging_prepare_take_locked", "vk_staging_prepare_reuse",
                 "vk_staging_prepare_supplier", "vk_staging_prepare_wait", "vk_upload_defer_release", "vk_mem_collect"):
        code += "\n" + function(name)
    code += MAIN
    cc = Path(os.environ.get("CC", "C:/Users/Andrew/AppData/Local/Programs/2NX2Navi/toolchain/msys64/ucrt64/bin/clang.exe"))
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="upload-staging-", dir=ROOT / "build/test") as tmp:
        source = Path(tmp) / "check.c"
        source.write_text(code)
        exe = source.with_suffix(".exe")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                        "-fsanitize-trap=all", str(source), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()

"""Actual mirror-span and lazy-copy adapters preserve partial, canceled and newer GPU authority."""
from pathlib import Path
import os
import subprocess
import tempfile
from bind_recipe_check import function

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ["CC"])
source = (ROOT / "runtime/vk_memory.c").read_text()
prefix = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "resource_core_registry.h"
static RcRegistry *registry;
#define rc_registry_runtime() registry
enum {POOL_IMPORT=1,POOL_PHYS,POOL_VIRT};
typedef uint64_t MemBackingCapture;
typedef struct {
    int kind;uint8_t *smap;uint64_t import_lo,import_hi,buf,mbuf,backing;
    MemBackingCapture *host_backing;
    uint64_t host_mapping_cutoff,*tpg;
    uint32_t ntpg;
    unsigned refs;
    uint8_t *base;
} PoolRec;
typedef struct {void *pool,*alloc;uint64_t buffer,offset,avail;} VkMemSpan;
typedef struct {PoolRec *buf;} pidx_ent;
static int pool_index,pool_lock;
static pidx_ent lookup;
static const pidx_ent *pidx_find(const int *index,uint64_t addr,uint64_t size) {(void)index;(void)addr;(void)size;return &lookup;}
static void plat_mutex_lock_shared(int *lock) {(void)lock;}
static void plat_mutex_unlock_shared(int *lock) {(void)lock;}
static uint64_t guest_host_base;
static int nvn_gpu_only(uint64_t a) {return a>=0x100000&&a!=guest_host_base;}
static uint32_t copy_pool_host_intervals(const PoolRec *,uint64_t,uint64_t,RcInterval **);
static uint32_t copy_pool_host_intervals(const PoolRec *pool,uint64_t addr,uint64_t size,RcInterval **out) {
    *out=NULL;
    if(!pool->backing)return 0;
    *out=malloc(sizeof **out);assert(*out);
    **out=(RcInterval){addr,4,{pool->backing,addr-pool->import_lo,size}};
    return 1;
}
static void rt_fatal(const char *s,...) {(void)s;abort();}
typedef struct {void (*done)(void *);void (*cancel)(void *,int);void *arg;uint64_t fence;} Callback;
static Callback callback;
static void pool_host_visibility(const RcRegistryVisibility *proof,const PoolRec *pool,uint64_t addr,uint64_t size) {
    if(!pool)return;
    RcInterval *parts;uint32_t count=copy_pool_host_intervals(pool,addr,size,&parts);
    for(uint32_t i=0;i<count;i++) rc_registry_visibility_apply(registry,proof,parts[i].range);
    free(parts);
}
static void vk_completion_add_owned(uint64_t f,void(*done)(void *),void(*cancel)(void *,int),void *arg) {
    assert(!callback.arg);callback=(Callback){done,cancel,arg,f};
}
typedef uint64_t VkCommandBuffer;
typedef struct {uint64_t srcOffset,dstOffset,size;} VkBufferCopy;
typedef struct {PoolRec *r;uint64_t off,size;} MRange;
static VkBufferCopy *mregions;
static int cregions,copy_commands;
static void grow(void **a,int *cap,size_t size) {*cap=*cap?*cap*2:8;*a=realloc(*a,*cap*size);assert(*a);}
static int range_cmp(const void *a,const void *b) {
    const MRange *x=a,*y=b;
    if(x->r!=y->r)return (uintptr_t)x->r<(uintptr_t)y->r?-1:1;
    return x->off<y->off?-1:x->off>y->off;
}
static void vkCmdCopyBuffer(VkCommandBuffer cb,uint64_t a,uint64_t b,uint32_t n,const VkBufferCopy *r) {
    (void)cb;(void)a;(void)b;assert(n&&r);copy_commands++;
}
'''
start = source.index("typedef struct { RcRegistryVisibility proof;")
end = source.index("// Records the ranges", start)
fixture = prefix + source[start:end] + function(source, "copy_ranges") + function(source, "vk_mem_cpu_visible_span")
fixture += r'''
static RcRegistryWrite writer(uint64_t lo,uint64_t n) {
    RcRegistryWrite w={0};assert(rc_registry_write_begin(registry,(RcSourceRange){1,lo,n},RC_REGISTRY_BUFFER,&w)==RC_REGISTRY_OK);return w;
}
static void finish(RcRegistryWrite w,uint64_t fence) {
    assert(rc_registry_write_submit(registry,w,fence)==RC_REGISTRY_OK);
    assert(rc_registry_write_complete(registry,w,0)==RC_REGISTRY_OK);
}
static int blocked_range(uint64_t backing,uint64_t lo) {
    static uint64_t serial=1;RcRegistryHandle h;RcRegistryObservation o;
    assert(rc_registry_source(registry,serial++,(RcSourceRange){backing,lo,1},&h)==RC_REGISTRY_OK);
    return rc_registry_observe(registry,h,1,&o)!=RC_REGISTRY_OK;
}
static int blocked(uint64_t lo) {return blocked_range(1,lo);}
static CpuVisibility job(uint64_t fence) {
    CpuVisibility j={0};assert(rc_registry_visibility_begin(registry,RC_REGISTRY_VISIBILITY_ORDERED,&j.proof)==RC_REGISTRY_OK);
    j.proof.submission=fence;return j;
}
static void run_callback(int canceled,int may) {
    Callback c=callback;callback=(Callback){0};
    if(canceled)c.cancel(c.arg,may);else c.done(c.arg);
}
static void mirrors(void) {
    registry=rc_registry_create((RcRegistryConfig){128,128,8});assert(registry);
    PoolRec p={.kind=POOL_IMPORT,.import_lo=0x1000,.import_hi=0x6000,.buf=1,.mbuf=2,.backing=2};
    lookup.buf=&p;VkMemSpan captured={.pool=&p,.buffer=1,.offset=0x1000,.avail=0x4000};
    assert(vk_mem_cpu_visible_span(&captured,0x2000,0x100));
    captured.buffer=2;assert(!vk_mem_cpu_visible_span(&captured,0x2000,0x100));captured.buffer=1;
    p.smap=(uint8_t *)&p;assert(!vk_mem_cpu_visible_span(&captured,0x2000,0x100));p.smap=NULL;
    p.kind=POOL_PHYS;assert(!vk_mem_cpu_visible_span(&captured,0x2000,0x100));p.kind=POOL_IMPORT;
    PoolRec replacement=p;lookup.buf=&replacement;assert(!vk_mem_cpu_visible_span(&captured,0x2000,0x100));lookup.buf=&p;
    assert(!vk_mem_cpu_visible_span(&captured,0x2100,0x100)); // Same raw buffer, wrong captured offset.
    RcRegistryWrite old=writer(0x1000,0x5000), captured_old={0}, later={0};
    assert(rc_registry_write_begin(registry,(RcSourceRange){2,0,0x5000},RC_REGISTRY_BUFFER,&captured_old)==RC_REGISTRY_OK);
    assert(rc_registry_write_begin(registry,(RcSourceRange){3,0,0x5000},RC_REGISTRY_BUFFER,&later)==RC_REGISTRY_OK);
    CpuVisibility j=job(4);
    MRange spans[]={{&p,0x3800,0x1000},{&p,0x1000,0x800},{&p,0x4800,0x2000},{&p,0x9000,0x100}};
    uint64_t bytes=0,copies=0;
    assert(copy_ranges(1,spans,4,0,&bytes,&copies,&j)==1);
    assert(bytes==0x2000&&copies==2&&j.count==4&&copy_commands==1); // Exact raw and canonical copies.
    finish(old,4);finish(captured_old,4);finish(later,4);cpu_visibility_queue(&j,4);
    p.import_lo=0x8000;p.import_hi=0xd000;p.backing=3;
    run_callback(0,0);
    assert(!blocked_range(2,0x1000)&&blocked_range(2,0x1800)&&blocked_range(3,0x1000));
    assert(!blocked(0x2000)&&!blocked(0x2700)&&!blocked(0x4800)&&!blocked(0x5fff));
    assert(blocked(0x1000)&&blocked(0x2800)&&blocked(0x4000)); // Gaps and unrecorded/lazy portions remain foreign.
    p.import_lo=0x1000;p.import_hi=0x6000;
    RcRegistryWrite newer=writer(0x3000,0x100);j=job(5);finish(newer,6);
    cpu_visibility_add(&j,&p,0x2000,0x100);cpu_visibility_queue(&j,5);run_callback(0,0);
    assert(blocked(0x3000)); // Reserved earlier but actually executed in a later scene.
    j=job(7);cpu_visibility_add(&j,&p,0,0x100);cpu_visibility_queue(&j,7);run_callback(1,1);
    assert(blocked(0x1000)); // Idle disposal of potentially executed partial scene proves no exact copy.
    j=job(8);p.smap=(uint8_t *)&p;cpu_visibility_add(&j,&p,0,0x5000);assert(!j.count);p.smap=NULL;
    p.kind=POOL_PHYS;cpu_visibility_add(&j,&p,0,0x100);assert(!j.count);cpu_visibility_queue(&j,8);
    p.kind=POOL_IMPORT;
    RcRegistryWrite image={0};
    assert(rc_registry_write_begin(registry,(RcSourceRange){1,0x4000,0x100},RC_REGISTRY_IMAGE,&image)==RC_REGISTRY_OK);
    finish(image,9);j=job(10);cpu_visibility_add(&j,&p,0x3000,0x100);cpu_visibility_queue(&j,10);run_callback(0,0);
    assert(blocked(0x4000)); // A buffer copy cannot certify an image's different backing, even when completed.
    rc_registry_destroy(registry);free(mregions);mregions=NULL;cregions=0;
}
'''

# Extract the complete lazy-copy function with a minimal Vulkan/queue harness. It must publish nothing on loss,
# and it may certify only completed producers observed before its own copy submission.
fixture += r'''
typedef uint64_t VkBuffer;
typedef int VkResult;
typedef struct {uint64_t device,gfx_queue;} VkCtx;
typedef struct {int sType;uint32_t flags;} VkCommandBufferBeginInfo;
typedef struct {int sType;uint32_t commandBufferCount;const VkCommandBuffer *pCommandBuffers;} VkSubmitInfo;
enum {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO=1,VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT=2,
VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT=4,VK_ACCESS_2_MEMORY_WRITE_BIT=8,VK_PIPELINE_STAGE_2_COPY_BIT=16,
VK_ACCESS_2_TRANSFER_READ_BIT=32,VK_ACCESS_2_TRANSFER_WRITE_BIT=64,VK_PIPELINE_STAGE_2_HOST_BIT=128,
VK_ACCESS_2_HOST_READ_BIT=256,VK_ACCESS_2_MEMORY_READ_BIT=512,VK_STRUCTURE_TYPE_SUBMIT_INFO=1024,VK_SUCCESS=0,VK_TRUE=1};
static uint64_t lz_pool,lz_fence;static VkCommandBuffer lz_cb;static int lost,submits;
static RcRegistryWrite pending_lazy;
static int vk_device_is_lost(void) {return lost;}
static VkCtx *vk_ctx(void) {static VkCtx c;return &c;}
static void vkResetCommandPool(uint64_t a,uint64_t b,int c) {(void)a;(void)b;(void)c;}
static void vkBeginCommandBuffer(uint64_t a,const VkCommandBufferBeginInfo *b) {(void)a;(void)b;}
static void mbarrier(uint64_t a,int b,int c,int d,int e) {(void)a;(void)b;(void)c;(void)d;(void)e;}
static void vkEndCommandBuffer(uint64_t a) {(void)a;}
static void vk_queue_lock(void) {}
static void vk_queue_unlock(void) {}
static void vk_sync_flush_held(void) {}
static VkResult vkQueueSubmit(uint64_t q,int n,const VkSubmitInfo *s,uint64_t f) {
    (void)q;(void)n;(void)s;(void)f;submits++;return VK_SUCCESS;
}
static VkResult vkWaitForFences(uint64_t d,int n,uint64_t *f,int all,uint64_t timeout) {
    (void)d;(void)n;(void)f;(void)all;(void)timeout;
    if(pending_lazy.id){finish(pending_lazy,20);pending_lazy=(RcRegistryWrite){0};}return VK_SUCCESS;
}
static void vk_device_lost_check(VkResult r,const char *s) {(void)r;(void)s;}
static void vkResetFences(uint64_t d,int n,uint64_t *f) {(void)d;(void)n;(void)f;}

enum { SB_PAGE=16 };
static uint64_t current_mapping=1;
static unsigned host_pins, stamps;
static int shadow_lock, gen_lock, s_check;
static uint64_t s_both_gpu, s_tab_wb, s_tab_wb_seen, s_tab_wb_jobs, s_tab_wb_ticks;
static int ntpools;
static PoolRec *tpools[2];
static int mem_read_hold(uint64_t address,uint64_t size) {
    assert(address && size && !host_pins);host_pins++;return 1;
}
static int mem_mapping_range_unchanged(uint64_t address,uint64_t size,uint64_t mapping) {
    (void)address;(void)size;return current_mapping==mapping;
}
static int mem_backing_read_hold_at(const MemBackingCapture *capture,uint64_t address,uint64_t size) {
    return mem_mapping_range_unchanged(address,size,*capture)&&mem_read_hold(address,size);
}
static void mem_read_release(void) {assert(host_pins==1);host_pins--;}
static void plat_mutex_lock(int *lock) {assert(!*lock);*lock=1;}
static void plat_mutex_unlock(int *lock) {assert(*lock);*lock=0;}
static uint8_t *sbase_find(PoolRec *p,uint64_t page) {assert(shadow_lock&&p->refs);return page==p->import_lo?p->base:NULL;}
static int mem_frozen(uint64_t page) {(void)page;return 0;}
static void mem_thaw(uint64_t page,uint64_t n) {(void)page;(void)n;assert(host_pins);}
static void plat_mem_prepare_write(uint64_t page,uint64_t n) {(void)page;(void)n;assert(host_pins);}
static uint64_t both_changed(const uint8_t *d,const uint8_t *s,const uint8_t *base,uint64_t n) {
    (void)d;(void)s;(void)base;(void)n;return 0;
}
static uint64_t merge_page(uint8_t *d,const uint8_t *s,uint8_t *base,uint64_t n) {
    assert(host_pins&&shadow_lock);uint64_t moved=0;
    for(uint64_t i=0;i<n;i++)if(s[i]!=base[i])d[i]=base[i]=s[i],moved++;
    return moved;
}
static uint32_t generation_next(void) {return 1;}
static void stamp_page_as(uint64_t page,uint32_t generation,int back) {
    (void)page;(void)generation;(void)back;assert(gen_lock&&host_pins);stamps++;
}
static void pool_native_acquire(PoolRec *p) {assert(shadow_lock&&p->refs);p->refs++;}
static void destroy_rec(VkCtx *context,PoolRec *p) {(void)context;assert(p->refs);p->refs--;}
static int shadow_bulk_write(void) {return 0;}
static int wb_no_reup(void) {return 0;}
static uint64_t qpc(void) {return 0;}
'''
fixture += function(source, "lazy_fetch")
fixture += function(source, "pool_host_write_hold")
table_job = source.index("} TableWritePage;")
start = source.rindex("typedef struct {", 0, table_job)
fixture += source[start:source.index("static void gpu_written(", table_job)]
fixture += r'''
static void lazy(void) {
    registry=rc_registry_create((RcRegistryConfig){64,64,8});assert(registry);
    finish(writer(0x1000,0x1000),10);
    pending_lazy=writer(0x3000,0x100);
    lazy_fetch(1,2,0,0x100,0x1000,NULL);
    assert(submits==1&&!blocked(0x1000)&&blocked(0x3000));
    // This pending producer becomes foreign during the wait; the earlier completed-only proof cannot remove it.
    pending_lazy=writer(0x5000,0x100);
    lazy_fetch(1,2,0,0x100,0x5000,NULL);assert(blocked(0x5000));
    lost=1;int before=submits;lazy_fetch(1,2,0,0x100,0x5000,NULL);
    assert(submits==before&&blocked(0x5000));
    lost=0;lazy_fetch(1,2,0,0x100,0x5000,NULL);assert(!blocked(0x5000));
    PoolRec recorded={.import_lo=0x5000,.backing=2};
    RcRegistryWrite old={0},successor={0};
    assert(rc_registry_write_begin(registry,(RcSourceRange){2,0,0x100},RC_REGISTRY_BUFFER,&old)==RC_REGISTRY_OK);
    assert(rc_registry_write_begin(registry,(RcSourceRange){3,0,0x100},RC_REGISTRY_BUFFER,&successor)==RC_REGISTRY_OK);
    finish(old,20);finish(successor,20);
    lazy_fetch(1,2,0,0x100,0x5000,&recorded);
    assert(!blocked_range(2,0)&&blocked_range(3,0));
    rc_registry_destroy(registry);
}
static void table_queue(uint64_t fence) {plat_mutex_lock(&shadow_lock);table_wb_queue(fence);plat_mutex_unlock(&shadow_lock);}
static void table_shadow(void) {
    registry=rc_registry_create((RcRegistryConfig){64,64,8});assert(registry);
    _Alignas(SB_PAGE) uint8_t guest[SB_PAGE]={0};
    uint8_t image[SB_PAGE]={0},base[SB_PAGE]={0};
    image[1]=8;image[3]=9;guest[2]=17;
    uint64_t address=(uint64_t)(uintptr_t)guest;
    guest_host_base=address;
    MemBackingCapture old_mapping=1;
    PoolRec original={.smap=image,.import_lo=address,.import_hi=address+SB_PAGE,.backing=2,
        .host_backing=&old_mapping,.tpg=&address,.ntpg=1,.refs=1,.base=base};
    PoolRec replacement=original;
    replacement.backing=3;
    tpools[0]=&original;ntpools=1;
    RcRegistryWrite old={0},successor={0};
    assert(rc_registry_write_begin(registry,(RcSourceRange){2,0,SB_PAGE},RC_REGISTRY_BUFFER,&old)==RC_REGISTRY_OK);
    assert(rc_registry_write_begin(registry,(RcSourceRange){3,0,SB_PAGE},RC_REGISTRY_BUFFER,&successor)==RC_REGISTRY_OK);
    finish(old,3);finish(successor,3);
    table_queue(4);
    assert(original.refs==2);
    tpools[0]=&replacement; // The current table changed after recording.
    run_callback(0,0);
    assert(original.refs==1 && replacement.refs==1 && s_tab_wb==2 && s_tab_wb_jobs==1);
    assert(stamps==1&&s_tab_wb_seen==SB_PAGE&&!s_tab_wb_ticks&&!s_both_gpu);
    assert(!host_pins && guest[1]==8 && guest[2]==17 && guest[3]==9);
    assert(!blocked_range(2,1)&&blocked_range(2,2)&&!blocked_range(2,3)&&blocked_range(3,1));

    tpools[0]=&original;image[1]=55;
    table_queue(5);
    destroy_rec(vk_ctx(),&original); // Pool retirement leaves the queued owner's final native ref.
    current_mapping++;
    guest[1]=21;tpools[0]=&replacement;
    run_callback(0,0);
    assert(!original.refs&&replacement.refs==1&&guest[1]==21&&base[1]==8);
    table_queue(6);run_callback(1,1);
    assert(replacement.refs==1&&guest[1]==21&&s_tab_wb==2&&s_tab_wb_jobs==2);
    assert(stamps==1&&s_tab_wb_seen==2*SB_PAGE);
    rc_registry_destroy(registry);
}
int main(void) {
    mirrors();lazy();table_shadow();
    puts("visibility adapters: exact mirrors, lazy backing, GPA byte runs, replacement and cancellation passed");
}
'''

env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="visibility-adapters-", dir=ROOT / "build/test") as tmp:
    path = Path(tmp) / "visibility.c"
    path.write_text(fixture)
    exe = path.with_suffix(".exe")
    subprocess.run([str(CC), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fsanitize-trap=all", "-DRC_REGISTRY_SINGLE_THREADED",
                    "-I", str(ROOT / "runtime"), str(path), str(ROOT / "runtime/resource_core_registry.c"),
                    "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

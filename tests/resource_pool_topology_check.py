"""Run the production Vulkan pool adapter against the canonical store and a model of the GPA table."""
import os
from pathlib import Path
import subprocess
import tempfile

from bind_recipe_check import function


ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ["CC"])
source = (ROOT / "runtime/vk_memory.c").read_text()
fixture = r'''
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nvn_pool_index.h"
#define RC_BACKING_SINGLE_THREADED 1
#include "resource_core_backing.c"

enum { POOL_IMPORT=1, POOL_PHYS, POOL_VIRT, MS_ON=1, VK_MCLS_DEVICE=1, VK_SUCCESS=0 };
enum { NVN_MEMORY_POOL_FLAGS_GPU_NO_ACCESS=1, NVN_MEMORY_POOL_FLAGS_PHYSICAL=2, NVN_MEMORY_POOL_FLAGS_VIRTUAL=4 };
#define NVN_VPOOL_PAGE UINT64_C(65536)
#define VK_NULL_HANDLE 0
#define rt_log(...) ((void)0)
#define rt_fatal(...) abort()
typedef uintptr_t VkBuffer;
typedef struct MemBackingCapture MemBackingCapture;
static int mem_backing_current(const MemBackingCapture *capture) { return capture != NULL; }
static void mem_backing_release(MemBackingCapture **capture) { *capture = NULL; }
static const RcInterval *mem_backing_intervals(const MemBackingCapture *capture, uint32_t *count) {
    (void)capture; *count = 0; return NULL;
}
typedef struct { unsigned flags; uint64_t size; void *mem; } pool_t;
typedef struct { const pool_t *physical_pool; int64_t virtual_offset, physical_offset; uint64_t size; } nvn_mapping_t;
typedef struct { uintptr_t device, vma, gfx_queue; } VkCtx;
typedef struct { VkBuffer buffer; uint64_t offset, alloc_offset, avail; uintptr_t alloc; int cls; } VkMemSpan;
typedef struct PoolRec {
    const pool_t *key;
    int kind, mstate;
    unsigned flags, gpa_rec, csb;
    int32_t used;
    uint64_t lo, size, import_lo, import_hi, publication;
    uint64_t physical_origin_upload, physical_origin_bytes;
    void *physical_raw_uses;
    MemBackingCapture *host_backing;
    uint64_t host_mapping_cutoff;
    uint32_t native_refs;
    uint8_t gpa_host_coherent;
    int unpublished;
    RcBackingId backing;
    RcBackingMapId map;
    VkBuffer buf, mbuf;
    uintptr_t mem, alloc, malloc_, salloc, smap;
    struct { void *base; } *sb;
} PoolRec;

static int pool_lock, cap_phys, nphys, shadow_on, n_import_failures, fail_allocation, destroyed;
static pidx pool_index;
static PoolRec **phys_list;
static uint64_t pool_gen;
static RcBackingStore *pool_backings;
static VkCtx ctx;
static PoolRec *retiring[8];
static uint32_t pending;
static uintptr_t last_buffer;
static VkCtx *vk_ctx(void) { return &ctx; }
static void plat_mutex_lock(int *lock) { assert(!*lock); ++*lock; }
static void plat_mutex_unlock(int *lock) { assert(*lock==1); --*lock; }
#define plat_mutex_lock_shared plat_mutex_lock
#define plat_mutex_unlock_shared plat_mutex_unlock
static long plat_inc32(int *value) { return ++*value; }
static void plat_xchg32(int32_t *value, int32_t replacement) { *value=replacement; }
static int alloc_physical(VkCtx *c, PoolRec *r) {
    (void)c;
    if (fail_allocation) return 0;
    r->buf=++last_buffer;
    r->alloc=r->buf+100;
    return 1;
}
static int import_pool(VkCtx *c, PoolRec *r, uint64_t addr, uint64_t size) {
    (void)c; (void)r; (void)addr; (void)size;
    return 0;
}
// Model the native initialization boundary. The extracted initializer must record the fill
// and barrier before pool_init first publishes the backing through the address index.
typedef uintptr_t VkCommandBuffer;
typedef struct { int sType; uint64_t srcStageMask, srcAccessMask, dstStageMask, dstAccessMask; } VkMemoryBarrier2;
typedef struct { int sType; uint32_t memoryBarrierCount; const VkMemoryBarrier2 *pMemoryBarriers; } VkDependencyInfo;
enum { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 = 1, VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
       VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT = 1, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT = 2,
       VK_ACCESS_2_TRANSFER_WRITE_BIT = 1, VK_ACCESS_2_MEMORY_READ_BIT = 2, VK_ACCESS_2_MEMORY_WRITE_BIT = 4 };
static struct { VkBuffer buffer; uint64_t bytes, sequence; int phase; } origin;
static int vk_fix_zero(void) { return 1; }
static VkCommandBuffer vk_upload_begin(void) { assert(!origin.phase && !pool_lock); origin.phase = 1; return 1; }
static void vkCmdFillBuffer(VkCommandBuffer cb, VkBuffer buffer, uint64_t offset, uint64_t bytes, uint32_t data) {
    assert(cb == 1 && origin.phase == 1 && buffer && !offset && bytes && !(bytes & 3) && !data);
    origin.buffer = buffer; origin.bytes = bytes; origin.phase = 2;
}
static void vkCmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo *dependency) {
    assert(cb == 1 && origin.phase == 2 && dependency->sType == VK_STRUCTURE_TYPE_DEPENDENCY_INFO &&
           dependency->memoryBarrierCount == 1);
    const VkMemoryBarrier2 *barrier = dependency->pMemoryBarriers;
    assert(barrier->sType == VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 &&
           barrier->srcStageMask == VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT && barrier->srcAccessMask == VK_ACCESS_2_TRANSFER_WRITE_BIT &&
           barrier->dstStageMask == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT &&
           barrier->dstAccessMask == (VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT));
    origin.phase = 3;
}
static uint64_t vk_upload_recording_sequence(VkCommandBuffer cb) { assert(cb == 1 && origin.phase == 3); return ++origin.sequence; }
static void vk_upload_end(void) { assert(origin.phase == 3); origin.phase = 0; }
static void pool_authority_begin(PoolRec *r) {
    if (r->kind != POOL_PHYS) return;
    assert(r->backing.id && !r->publication && r->buf == origin.buffer && !origin.phase &&
           r->physical_origin_upload == origin.sequence && r->physical_origin_bytes == origin.bytes);
}
static void pool_authority_end(PoolRec *r) { (void)r; }
static void mem_thaw(uint64_t addr, uint64_t size) { (void)addr; (void)size; }
static void mem_lazy_release(uint64_t addr, uint64_t size) { (void)addr; (void)size; }
static int mem_mapping_range_unchanged(uint64_t a,uint64_t n,uint64_t stamp) {
    (void)a; (void)n; (void)stamp; return 0;
}
static void shadow_metadata_destroy(PoolRec *r) { (void)r; }
static void shadow_table_add(PoolRec *r) { (void)r; }
static void shadow_forget(PoolRec *r) { (void)r; }
static void mirror_forget(PoolRec *r) { (void)r; }
static void mirror_pressure_remove(PoolRec *r) { assert(pool_lock && r->unpublished); }
static void resource_core_vma_destroy_buffer(uintptr_t vma, VkBuffer buf, uintptr_t alloc) {
    (void)vma; (void)buf; (void)alloc;
}
static void resource_core_vma_free_memory(uintptr_t vma, uintptr_t alloc) { (void)vma; assert(alloc); }
static void vkDestroyBuffer(uintptr_t device, VkBuffer buf, void *unused) {
    (void)device; (void)unused;
    assert(buf);
    destroyed++;
}
static void vkFreeMemory(uintptr_t device, uintptr_t mem, void *unused) { (void)device; (void)mem; (void)unused; }
static void vk_texture_range_gone(uint64_t lo, uint64_t size, uintptr_t alloc) { (void)lo; (void)size; (void)alloc; }
static int pool_fini_idle(void) { return 0; }
static void pool_fini_wait(void) {}
static void vk_upload_flush(int wait) { (void)wait; }
static void vk_queue_lock(void) {}
static void vk_queue_unlock(void) {}
static void vkQueueWaitIdle(uintptr_t queue) { (void)queue; }
static void vk_mem_collect(void) {}
static uint64_t vk_sync_issued(void) { return 2; }
static void pool_retiring(PoolRec *r) { retiring[pending++]=r; }
static void pool_wait_add(PoolRec *r) { assert(pending && r==retiring[pending-1]); }
static void pool_retire(void *arg) { (void)arg; abort(); }
static void vk_completion_add(uint64_t fence, void (*fn)(void *), void *arg) {
    assert(fence==3 && fn==pool_retire && pending && retiring[pending-1]==arg);
}
static uint32_t vk_gpa_map(uint64_t lo, uint64_t hi, VkBuffer buf, uint64_t base, int visibility,
                               const RcInterval *host, uint32_t count) {
    (void)lo; (void)hi; (void)buf; (void)base; (void)visibility; (void)host; (void)count;
    return 0;
}
static void vk_gpa_unmap(uint64_t lo, uint64_t hi, uint32_t rec, uint32_t (*cover)(uint64_t)) {
    (void)lo; (void)hi; (void)rec; (void)cover;
    abort();
}
enum { V=32, W=48 };
static struct { VkBuffer buffer; uint64_t offset; } gpa[64];
static uint32_t projection_calls;
typedef struct { uint64_t lo, hi; VkBuffer buffer; RcSourceRange backing; uint64_t delta; } VkGpaVirtualBinding;
typedef struct { const VkGpaVirtualBinding *bindings; uint32_t count; } VkGpaVirtualUpdate;
static int projection_reserved, projection_limit, projection_cancels, projection_disabled;
static RcBackingResult vk_gpa_virtual_prepare(VkGpaVirtualBinding *bindings, uint32_t count, VkGpaVirtualUpdate *out) {
    assert(!projection_reserved && pool_lock);
    *out=(VkGpaVirtualUpdate){0};
    if (projection_disabled) return RC_BACKING_OK;
    for (uint32_t i=0; i<count; ++i) {
        const VkGpaVirtualBinding *b=&bindings[i];
        assert(b->lo%NVN_VPOOL_PAGE==0 && b->hi%NVN_VPOOL_PAGE==0 && b->hi/NVN_VPOOL_PAGE<=64);
        assert(b->lo/NVN_VPOOL_PAGE>=V);
        if (b->buffer) assert(rc_range_valid(b->backing) && b->backing.size==b->hi-b->lo);
        else assert(!b->backing.backing);
        if (b->buffer && projection_limit && b->lo/NVN_VPOOL_PAGE >= (unsigned)projection_limit)
            return RC_BACKING_CAPACITY;
    }
    *out=(VkGpaVirtualUpdate){bindings,count};
    projection_reserved=1;
    return RC_BACKING_OK;
}
static void project(VkGpaVirtualBinding b) {
    assert(pool_lock);
    projection_calls++;
    for (uint64_t addr=b.lo; addr<b.hi; addr+=NVN_VPOOL_PAGE) {
        gpa[addr/NVN_VPOOL_PAGE].buffer=b.buffer;
        gpa[addr/NVN_VPOOL_PAGE].offset=b.buffer ? b.backing.offset+addr-b.lo : 0;
    }
}
static void vk_gpa_virtual_commit(VkGpaVirtualUpdate *update) {
    if (!update->count) return;
    assert(projection_reserved);
    for (uint32_t i=0; i<update->count; ++i) project(update->bindings[i]);
    *update=(VkGpaVirtualUpdate){0};
    projection_reserved=0;
}
static void vk_gpa_virtual_cancel(VkGpaVirtualUpdate *update) {
    if (!update->count) return;
    assert(projection_reserved);
    *update=(VkGpaVirtualUpdate){0};
    projection_reserved=0;
    ++projection_cancels;
}
static void vk_gpa_virtual_unmap(uint64_t lo, uint64_t hi) {
    assert(!projection_reserved);
    project((VkGpaVirtualBinding){.lo=lo,.hi=hi});
}
static void pool_fini(pool_t *p);
'''
fixture += "\n".join(function(source, name) for name in (
    "index_put", "phys_find", "pool_topology_close", "pool_topology_open", "pool_topology_unpublish", "destroy_rec",
    "physical_origin_initialize", "pool_init", "gpa_cover", "pool_topology_replace", "vk_pool_map_virtual", "pool_device_span",
))
fixture += function(source[source.index("static void pool_fini(pool_t *p) {"):], "pool_fini")
fixture += r'''
static VkMemSpan lookup(PoolRec *r, uint64_t addr, uint64_t size) {
    VkMemSpan span={0};
    PoolRec *provider = NULL;
    plat_mutex_lock(&pool_lock);
    if (pool_device_span(r,addr,size,&span,&provider)) assert(provider && provider->buf == span.buffer);
    plat_mutex_unlock(&pool_lock);
    return span;
}

int main(void) {
    uint64_t page=NVN_VPOOL_PAGE;
    pool_t a={NVN_MEMORY_POOL_FLAGS_PHYSICAL,4*page,(void *)(uintptr_t)page};
    pool_t b={NVN_MEMORY_POOL_FLAGS_PHYSICAL,4*page,(void *)(uintptr_t)(8*page)};
    pool_t v={NVN_MEMORY_POOL_FLAGS_VIRTUAL,6*page,(void *)(uintptr_t)(V*page)};
    pool_t w={NVN_MEMORY_POOL_FLAGS_VIRTUAL,2*page,(void *)(uintptr_t)(W*page)};
    assert(pool_init(&a,&a) && pool_init(&b,&b) && pool_init(&v,&v) && pool_init(&w,&w));
    PoolRec *ar=phys_find(&a), *br=phys_find(&b);
    PoolRec *vr=pidx_find(&pool_index,V*page,page)->buf, *wr=pidx_find(&pool_index,W*page,page)->buf;
    assert(ar && br && ar->buf!=br->buf && ar->backing.id!=br->backing.id);
    assert(ar->publication && br->publication > ar->publication && origin.sequence == 2);
    const nvn_mapping_t initial[]={{&a,0,page,page},{&a,page,2*page,2*page}};
    const nvn_mapping_t other[]={ {&a,0,2*page,page} };
    assert(vk_pool_map_virtual(&v,2,initial) && vk_pool_map_virtual(&w,1,other));
    VkMemSpan span=lookup(vr,V*page+16,2*page);
    assert(span.buffer==ar->buf && span.offset==page+16 && span.avail==3*page-16 && ar->used);
    assert(gpa[V+2].buffer==ar->buf && gpa[V+2].offset==3*page && gpa[W].buffer==ar->buf);
    RcBackingCapture *exact, *alias;
    assert(rc_backing_capture(pool_backings,vr->map,(V+1)*page,2*page,RC_BACKING_REQUIRE_MAPPED,&exact)==RC_BACKING_OK);
    assert(rc_backing_capture(pool_backings,wr->map,W*page,page,RC_BACKING_REQUIRE_MAPPED,&alias)==RC_BACKING_OK);
    uint32_t n;
    const RcBackingSegment *old=rc_backing_capture_segments(exact,&n);
    assert(n==1 && rc_range_overlap(old[0].range,rc_backing_capture_segments(alias,NULL)[0].range));

    const nvn_mapping_t batch[]={{&b,0,0,page},{NULL,page,0,page},{&b,2*page,page,page}};
    uint64_t previous_gen=pool_gen;
    uint32_t previous_calls=projection_calls;
    projection_limit=V+2;
    assert(!vk_pool_map_virtual(&v,3,batch));
    assert(!projection_reserved && !projection_cancels && projection_calls==previous_calls && pool_gen==previous_gen);
    assert(lookup(vr,V*page,3*page).buffer==ar->buf && rc_backing_capture_current(pool_backings,exact));
    assert(gpa[V].buffer==ar->buf && gpa[V+1].buffer==ar->buf && gpa[V+2].buffer==ar->buf);
    projection_limit=0;

    RcBackingMapId saved_map=vr->map;
    vr->map.id++;
    assert(!vk_pool_map_virtual(&v,3,batch));
    vr->map=saved_map;
    assert(!projection_reserved && projection_cancels==1 && projection_calls==previous_calls && pool_gen==previous_gen);
    assert(lookup(vr,V*page,3*page).buffer==ar->buf && rc_backing_capture_current(pool_backings,exact));

    const nvn_mapping_t overlay[]={{&b,3*page,0,2*page},{NULL,3*page,0,page},{&b,3*page,2*page,page}};
    assert(vk_pool_map_virtual(&v,3,overlay));
    assert(lookup(vr,(V+3)*page,page).offset==2*page && gpa[V+3].offset==2*page);
    assert(lookup(vr,(V+4)*page,page).offset==page && gpa[V+4].offset==page);
    const nvn_mapping_t clear_extra={NULL,3*page,0,2*page};
    assert(vk_pool_map_virtual(&v,1,&clear_extra));

    projection_disabled=1;
    const nvn_mapping_t without_projection={&b,5*page,0,page};
    previous_calls=projection_calls;
    assert(vk_pool_map_virtual(&v,1,&without_projection) && projection_calls==previous_calls && !gpa[V+5].buffer);
    assert(lookup(vr,(V+5)*page,page).buffer==br->buf);
    assert(vk_pool_map_virtual(&v,1,&(nvn_mapping_t){NULL,5*page,0,page}));
    projection_disabled=0;
    const nvn_mapping_t remap={&b,page,0,page};
    assert(vk_pool_map_virtual(&v,1,&remap));
    assert(!lookup(vr,V*page,2*page).buffer);
    assert(lookup(vr,(V+1)*page,page).buffer==br->buf);
    assert(lookup(vr,(V+2)*page,page).buffer==ar->buf);
    assert(gpa[V+1].buffer==br->buf && gpa[V+1].offset==0);
    assert(!rc_backing_capture_current(pool_backings,exact) && rc_backing_capture_current(pool_backings,alias));
    uint32_t calls=projection_calls;
    const nvn_mapping_t overflow={&b,0,4*page,page};
    assert(!vk_pool_map_virtual(&v,1,&overflow) && projection_calls==calls);
    assert(lookup(vr,V*page,page).buffer==ar->buf);

    pool_fini(&a);
    assert(!phys_find(&a) && pending==1 && !destroyed && retiring[0]==ar);
    assert(!lookup(vr,V*page,page).buffer && !lookup(wr,W*page,page).buffer);
    assert(lookup(vr,(V+1)*page,page).buffer==br->buf);
    assert(!gpa[V].buffer && !gpa[V+2].buffer && !gpa[W].buffer && gpa[V+1].buffer==br->buf);
    assert(!rc_backing_capture_current(pool_backings,alias) && old[0].provider==(uintptr_t)ar);
    uintptr_t provider;
    assert(rc_backing_reclaim(pool_backings,ar->backing,&provider)==RC_BACKING_BUSY);
    rc_backing_capture_release(&alias);
    rc_backing_capture_release(&exact);
    destroy_rec(vk_ctx(),retiring[0]);

    pool_t replacement={NVN_MEMORY_POOL_FLAGS_PHYSICAL,2*page,b.mem};
    fail_allocation=1;
    assert(!pool_init(&b,&replacement) && phys_find(&b)==br && gpa[V+1].buffer==br->buf);
    fail_allocation=0;
    assert(pool_init(&b,&replacement) && b.size==4*page && pending==2 && !gpa[V+1].buffer);
    PoolRec *fresh=phys_find(&b);
    assert(fresh!=br && fresh->backing.id!=br->backing.id && fresh->size==2*page);
    assert(fresh->publication > br->publication && origin.sequence == 3);
    assert(!lookup(vr,(V+1)*page,page).buffer);
    destroy_rec(vk_ctx(),retiring[1]);
    assert(vk_pool_map_virtual(&v,1,&remap) && gpa[V+1].buffer==fresh->buf);
    const nvn_mapping_t unmap={NULL,page,2*page,page};
    assert(vk_pool_map_virtual(&v,1,&unmap) && !gpa[V+1].buffer && !lookup(vr,(V+1)*page,page).buffer);
    pool_fini(&v);
    pool_fini(&w);
    pool_fini(&b);
    assert(pending==3 && destroyed==2 && !pool_index.n && !nphys);
    destroy_rec(vk_ctx(),retiring[2]);
    assert(destroyed==3 && rc_backing_store_destroy(pool_backings)==RC_BACKING_OK);
    free(pool_index.e); free(pool_index.maxhi); free(phys_list);
    puts("native pool topology: affine aliases, partial remap, invalid requests, GPA cleanup and retirement passed");
}
'''

env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
work = ROOT / "build/test"
work.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="pool-topology-", dir=work) as tmp:
    path = Path(tmp) / "topology.c"
    path.write_text(fixture)
    exe = path.with_suffix(".exe")
    subprocess.run([str(CC), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                    "-fsanitize-trap=all", "-I", str(ROOT / "runtime"), str(path), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

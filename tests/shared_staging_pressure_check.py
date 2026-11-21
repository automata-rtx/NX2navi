"""Verify that required admission can reclaim only completed, useful staging capacity."""
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parent))
from upload_staging_progress_check import ROOT, SOURCE, function


FIXTURE = r'''
#include <stdlib.h>
#define main shared_budget_controls
#include "resource_core_budget_check.c"
#undef main

typedef uint64_t VkBuffer, VkDeviceSize, VmaAllocation, VkFence;
typedef struct { int device, vma; } VkCtx;
typedef struct { VkBuffer buf; VmaAllocation alloc; VkDeviceSize size; } Staging;
typedef struct { VkBuffer buf; VmaAllocation alloc; } StgChunk;
typedef struct {
    VkFence fence;
    uint64_t seq;
    Staging *stg;
    int nstg;
    StgChunk *chunk;
    int nchunk;
    uint64_t chunk_used, staging_bytes;
} Batch;
/* production capacity macros */
static Staging stg_cache[STG_CACHE];
static int stg_ncache;
static VkDeviceSize stg_cache_bytes;
static StgChunk arena_free[ARENA_KEEP];
static int narena_free;
static Batch open_b, *inflight, *spare;
static uint32_t upload_depth;
static int ninflight, nspare, cap_spare, cmd_lock, lock_busy, stopped;
static uint64_t inflight_staging_bytes, done_seq, signaled;
static VkCtx ctx;
static RcBudgetTicket allocation[8];
static RcAdmissionVector costs[8];
static unsigned reclaimed, releases_blocks;

enum { VK_SUCCESS=0, VK_NOT_READY=1, VK_TRUE=1 };
static VkCtx *vk_ctx(void) { return &ctx; }
static int vk_sync_stopping(void) { return stopped; }
static int vk_device_is_lost(void) { return 0; }
static int plat_rmutex_trylock(int *lock) {
    assert(!lock_depth && !*lock);
    if (lock_busy) return 0;
    *lock=1;
    return 1;
}
static void plat_rmutex_unlock(int *lock) { assert(*lock); *lock=0; }
static void rt_fatal(const char *message) { fprintf(stderr,"%s\n",message); abort(); }
static int vkWaitForFences(int d,int n,VkFence *f,int all,uint64_t timeout) {
    (void)d; (void)n; (void)f; (void)all; (void)timeout;
    abort(); // Shared pressure must never wait to make a batch eligible.
}
static int vkGetFenceStatus(int device,VkFence fence) {
    (void)device;
    return fence <= signaled ? VK_SUCCESS : VK_NOT_READY;
}
static RcAdmissionStatus resource_core_vma_allocation_cost(VmaAllocation index,RcAdmissionVector *out) {
    assert(cmd_lock && !lock_depth && index < 8);
    return resource_core_budget_ticket_cost(&allocation[index],out);
}
static void observed_allocation(VmaAllocation index,int destroy) {
    for (unsigned d=0;d<RC_ADMISSION_MAX_DOMAINS;d++) {
        if (d==RC_BUDGET_STAGING) continue;
        if (destroy) observed.free.amount[d]+=costs[index].amount[d];
        else observed.free.amount[d]-=costs[index].amount[d];
    }
}
static void resource_core_vma_destroy_buffer(int vma,VkBuffer buffer,VmaAllocation index) {
    (void)vma;
    assert(!cmd_lock && !lock_depth && buffer==index && !(reclaimed & (1u<<index)));
    OK(resource_core_budget_retire(&allocation[index]));
    assert(snapshot().pending.amount[RC_BUDGET_STAGING] >= costs[index].amount[RC_BUDGET_STAGING]);
    reclaimed |= 1u << index;
    OK(resource_core_budget_destroyed(&allocation[index]));
    if (releases_blocks) observed_allocation(index,1);
}
static void allocate_staging(VmaAllocation index,unsigned mb,unsigned heap) {
    costs[index]=cost(mb,0,mb);
    OK(resource_core_budget_gpu_cost(&costs[index],heap,mb*MB));
    OK(resource_core_budget_reserve(RC_BUDGET_STAGE,1,&costs[index],&allocation[index]));
    OK(resource_core_budget_materialize(&allocation[index],&costs[index]));
    observed_allocation(index,0);
    fresh();
}
'''

MAIN = r'''
static void staging_pressure(void) {
    start(100);
    observed.known_mask |= 1u << (RC_BUDGET_GPU_FIRST+1);
    observed.total.amount[RC_BUDGET_GPU_FIRST+1] = 8192*MB;
    observed.free.amount[RC_BUDGET_GPU_FIRST+1] = observed.free.amount[RC_BUDGET_GPU_FIRST];
    fresh();
    cmd_init(NULL);
    cmd_init(NULL); // Process initialization must not register a duplicate provider.
    allocate_staging(1,24,1);
    allocate_staging(2,24,0);
    allocate_staging(3,8,0);
    allocate_staging(4,4,0);
    Staging open_stage={4,4,4*MB};
    open_b=(Batch){.stg=&open_stage,.nstg=1,.staging_bytes=4*MB};
    stg_cache[0]=(Staging){1,1,24*MB};
    stg_cache[1]=(Staging){2,2,24*MB};
    stg_ncache=2;
    stg_cache_bytes=48*MB;
    StgChunk active_chunk={3,3};
    Batch active={.fence=1,.seq=7,.chunk=&active_chunk,.nchunk=1,.staging_bytes=8*MB};
    inflight=&active; ninflight=1; inflight_staging_bytes=8*MB;

    RcAdmissionVector demand=cost(0,88,0);
    RcBudgetProgress progress;
    RcBudgetTicket consumer={0};
    IS(resource_core_budget_reserve(RC_BUDGET_TEXTURE,1,&demand,&consumer),RC_ADMISSION_REFUSED);
    resource_core_budget_service(0,&demand,&progress);
    assert(!reclaimed && stg_ncache==2 && progress.blocked_mask);
    lock_busy=1;
    resource_core_budget_service(1,&demand,&progress);
    unsigned busy_result=progress.reclaim;
    assert(!reclaimed && progress.blocked_mask);
    lock_busy=0;
    upload_depth=1;
    resource_core_budget_service(1,&demand,&progress);
    unsigned active_upload_result=progress.reclaim;
    assert(upload_depth==1 && !reclaimed && progress.blocked_mask);
    upload_depth=0;
    stopped=1;
    resource_core_budget_service(1,&demand,&progress);
    assert(!progress.reclaim && !reclaimed && progress.blocked_mask);
    stopped=0;

    releases_blocks=1;
    resource_core_budget_service(1,&demand,&progress);
    if (progress.blocked_mask || reclaimed!=(1u<<2))
        fprintf(stderr,"shared staging never supplied required heap: mask=%#x reclaimed=%#x\n",
                progress.blocked_mask,reclaimed);
    assert(progress.reclaim==RC_BUDGET_RECLAIM_DESTROYED && !progress.blocked_mask);
    assert(busy_result==RC_BUDGET_RECLAIM_BUSY && active_upload_result==RC_BUDGET_RECLAIM_BUSY);
    assert(reclaimed==(1u<<2) && stg_ncache==1 && stg_cache[0].alloc==1 && stg_cache_bytes==24*MB);
    assert(ninflight==1 && !narena_free && active.nchunk==1 && !done_seq);
    assert(open_b.nstg==1 && open_b.stg[0].alloc==4 && allocation[4].token.state==RC_ADMISSION_RETAINED);
    OK(resource_core_budget_reserve(RC_BUDGET_TEXTURE,1,&demand,&consumer));
    OK(resource_core_budget_cancel(&consumer));
    assert(!consumer.token.ledger && !snapshot().reserved.amount[RC_BUDGET_GPU_FIRST]);

    // Resource pressure cannot borrow in-flight work.
    RcAdmissionVector large=cost(0,96,0);
    resource_core_budget_service(1,&large,&progress);
    assert(progress.blocked_mask==(1u<<RC_BUDGET_GPU_FIRST) && !progress.reclaim);
    cmd_lock=1;
    retire(&ctx,0);
    assert(active.nchunk==1 && !done_seq);
    signaled=1;
    uint64_t before=clock_ns;
    retire(&ctx,0);
    cmd_lock=0;
    assert(done_seq==7 && narena_free==1 && !ninflight && !inflight_staging_bytes);
    resource_core_budget_wait_since(progress.event,100);
    assert(clock_ns==before); // Actual completion exposes an eligible cache row and wakes required admission.

    releases_blocks=0; // Vulkan allocator retained the block, so measured/physical allowance stays short.
    resource_core_budget_service(1,&large,&progress);
    assert(progress.reclaim==RC_BUDGET_RECLAIM_DESTROYED && progress.blocked_mask==(1u<<RC_BUDGET_GPU_FIRST));
    assert(reclaimed==((1u<<2)|(1u<<3)) && !narena_free && stg_cache[0].alloc==1);
    assert(!snapshot().pending.amount[RC_BUDGET_STAGING]);
    IS(resource_core_budget_reserve(RC_BUDGET_TEXTURE,1,&large,&consumer),RC_ADMISSION_REFUSED);
    observed_allocation(3,1);
    fresh();
    OK(resource_core_budget_reserve(RC_BUDGET_TEXTURE,1,&large,&consumer));
    OK(resource_core_budget_cancel(&consumer));
    free(spare);
}
int main(void) {
    staging_pressure();
    shared_budget_controls();
    puts("required staging pressure: exact heap, completed cache, observed relief, cancellation and wake passed");
}
'''


def main():
    macros = [re.search(rf"^#define {name} .*$", SOURCE, re.M).group(0)
              for name in ("STG_CACHE", "STG_CACHE_BYTES", "ARENA_KEEP", "ARENA_CHUNK")]
    code = FIXTURE.replace("/* production capacity macros */","\n".join(macros))
    if "static unsigned staging_budget_reclaim(" in SOURCE:
        code += "\n" + "\n".join(function(name) for name in (
            "staging_reclaim_matches", "staging_reclaim_take", "staging_reclaim_destroy", "staging_budget_reclaim"))
    code += "\n" + "\n".join(function(name) for name in ("cmd_init","grow","staging_cache_room","free_staging","retire"))
    code += MAIN
    cc = Path(os.environ.get("CC", "C:/Users/Andrew/AppData/Local/Programs/2NX2Navi/toolchain/msys64/ucrt64/bin/clang.exe"))
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="required-staging-") as tmp:
        source = Path(tmp) / "check.c"
        executable = source.with_suffix(".exe")
        source.write_text(code, encoding="utf8")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
                        "-fsanitize=undefined", "-fsanitize-trap=all", "-I", str(ROOT / "tests"),
                        str(source), "-o", str(executable)], env=env, check=True)
        subprocess.run([str(executable)], env=env, check=True)


if __name__ == "__main__":
    main()

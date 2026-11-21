"""Check live mirror retry and publication with production replay functions and fake Vulkan boundaries."""
from pathlib import Path
import os
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MEMORY = (ROOT / 'runtime/vk_memory.c').read_text()


def function(name):
    match = re.search(r'^(?:static )?[^\n]*\b' + name + r'\([^;{}]*\)\s*\{', MEMORY, re.M)
    assert match, name
    depth, end = 1, match.end()
    while depth:
        depth += (MEMORY[end] == '{') - (MEMORY[end] == '}')
        end += 1
    return MEMORY[match.start():end]


PREFIX = r'''
#include <assert.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nvn_pool_index.h"
#include "resource_core_budget.h"
/* dependency type */
typedef unsigned VkBuffer,VmaAllocation,VkResult;
typedef struct {unsigned sType,size,usage,sharingMode;} VkBufferCreateInfo;
typedef struct {unsigned usage,requiredFlags;} VmaAllocationCreateInfo;
typedef struct {RcBudgetProducer producer;RcVmaDependency *dependency;} RcVmaRequest;
typedef struct {int vma;} VkCtx;
enum { POOL_IMPORT=1,MS_NONE=0,MS_ON,MS_REFUSED,MS_CREATING,MS_PENDING,
 MR_BUDGET,MR_UNTRACKED,MR_OVERLAP,MR_ALLOC,MR_GPA_PROJECTION,MR_N,
 VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,VK_SHARING_MODE_EXCLUSIVE,VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,VK_SUCCESS,VK_NOT_READY,VK_ERROR_OUT_OF_DEVICE_MEMORY,
 VK_ERROR_OUT_OF_HOST_MEMORY,VK_ERROR_DEVICE_LOST };
enum { SNAP_SHIFT=12,SNAP_REGIONS=16 };
typedef struct PoolRec {
    uint64_t lo,import_lo,import_hi;
    int kind,unpublished;
    _Atomic uint8_t mstate;
    _Atomic uint32_t mirror_checked_sub;
    _Atomic uint64_t mirror_import_only_scene;
    uint64_t mirror_incarnation;
    void *host_backing;
    RcVmaDependency mirror_dependency;
    struct PoolRec *mirror_pressure_prev,*mirror_pressure_next;
    uint8_t refused_why,twin,gpa_host_coherent;
    VkBuffer buf,mbuf;
    VmaAllocation malloc_;
    void *smap;
    uint32_t mgen,acc_sub,gpa_rec,vd_use_sub;
} PoolRec;
typedef struct {PoolRec *pool; VkBuffer buffer;} VkMemSpan;
static pidx pool_index;
static int pool_lock,gen_lock,locks;
static int stop,lost,mepoch,replace_during_create,vk_mirror_on=1;
static uint32_t gen_now=3,msub=1;
static uint64_t mirror_gen;
static int64_t m_untwinned_n,m_untwinned_bytes,m_live_n,m_live_bytes,m_refused_n[MR_N],m_refused_bytes[MR_N],m_retries;
static struct {unsigned sync_queries,first_n,first_bytes;} mrep;
static uint8_t snap_untracked[SNAP_REGIONS];
static PoolRec *mirror_pressure_first,*mirror_pressure_last;
static PoolRec *mirrors[4],**mlist=mirrors,*replacement;
static int nm,mcap=4,pend,npend,cpend;
static VkCtx ctx={1};
static unsigned attempts,destructions,copies,remaps,observations,import_selected;
static int host_mapping_current=1,host_held,lose_host_during_create,deny_projection;
static unsigned service_calls,evictions,pressure_refs,service_busy,idle_owned;
static RcBudgetReclaimFn reclaimer;
static RcBudgetReadiness availability;
static VkResult native_result=VK_NOT_READY;
static VkCtx *vk_ctx(void){return &ctx;}
static int vk_sync_stopping(void){return stop;}
static int vk_device_is_lost(void){return lost;}
static int mem_backing_read_hold(void *mapping){
    assert(mapping&&pool_lock&&!host_held);
    if (!host_mapping_current) return 0;
    host_held=1;return 1;
}
static void mem_read_release(void){assert(pool_lock&&host_held);host_held=0;}
static void vk_device_lost_check(VkResult r,const char *where){(void)where;if(r==VK_ERROR_DEVICE_LOST)lost=1;}
static void plat_mutex_lock(int *l){assert(!*l);*l=1;locks++;}
static void plat_mutex_unlock(int *l){assert(*l);*l=0;locks--;}
#define plat_mutex_lock_shared plat_mutex_lock
#define plat_mutex_unlock_shared plat_mutex_unlock
static int64_t plat_inc64(int64_t *p){return ++*p;}
static int64_t plat_dec64(int64_t *p){return --*p;}
static int64_t plat_add64(int64_t *p,int64_t n){return *p+=n;}
static void grow(void **p,int *n,size_t size){(void)p;(void)n;(void)size;abort();}
static void rt_fatal(const char *message){(void)message;abort();}
static int evict_enabled(void){return 1;}
typedef unsigned PlatOnce;
#define PLAT_ONCE_INIT 0
static void plat_once(PlatOnce *once,void(*fn)(void*),void *arg){if(!*once){*once=1;fn(arg);}}
static void pool_native_acquire(PoolRec *r){assert(pool_lock&&r->mstate==MS_PENDING);pressure_refs++;}
static void destroy_rec(VkCtx *c,PoolRec *r){assert(c==&ctx&&r&&pressure_refs&&!locks);pressure_refs--;}
static int vk_sync_try_provider_idle(void){assert(!locks&&!idle_owned);idle_owned=1;return 1;}
static void vk_sync_end_provider_idle(void){assert(idle_owned==1);idle_owned--;}
void resource_core_budget_snapshot(RcBudgetSnapshot *out){assert(idle_owned);*out=(RcBudgetSnapshot){0};}
RcAdmissionStatus resource_core_budget_register_reclaimer(RcBudgetProducer provider,uint32_t domains,
          RcBudgetReclaimFn fn,void *context){
    assert(!locks&&provider==RC_BUDGET_BUFFER&&((1u<<RC_BUDGET_GPU_FIRST)&domains)&&!context);
    reclaimer=fn;return RC_ADMISSION_OK;
}
void resource_core_budget_service(int required,const RcAdmissionVector *peak,RcBudgetProgress *out){
    assert(!locks&&!required&&reclaimer&&peak->amount[RC_BUDGET_GPU_FIRST]==4096);
    service_calls++;
    out->reclaim=service_busy?RC_BUDGET_RECLAIM_BUSY:reclaimer(NULL,peak);
}
static uint64_t mirror_evict(uint32_t domain,uint64_t bytes,const char *reason){
    assert(idle_owned&&domain==RC_BUDGET_GPU_FIRST&&bytes==4096&&reason);evictions++;return 0;
}
static int twin_wanted(const PoolRec *r){assert(r->buf);return 1;}
static int snap_current(uint32_t k,uint32_t ep){(void)k;(void)ep;return 1;}
static int snap_regions(uint64_t lo,uint64_t hi,uint32_t ep){(void)lo;(void)hi;(void)ep;return 1;}
static unsigned pool_usage(VkCtx *c){assert(c==&ctx);return 1;}
static void mirror_sync(PoolRec *r){r->acc_sub=msub;}
static void range_add(int *p,int *n,int *c,PoolRec *r,uint64_t at,uint64_t size){
    (void)p;(void)n;(void)c;assert(pool_lock&&r->mstate==MS_CREATING&&at==0&&size==4096);copies++;
}
static unsigned vk_gpa_remap_exact(uint64_t lo,uint64_t hi,unsigned old,VkBuffer buf,uint64_t at,int keep,int visible,
                                   int *uses_twin){
    assert(pool_lock&&lo==at&&hi-lo==4096&&old==1&&buf==17&&keep&&visible&&uses_twin);
    *uses_twin=!deny_projection;
    if (deny_projection) return 0;
    remaps++;return 2;
}
void resource_core_budget_readiness(int required,RcBudgetReadiness *out){assert(!required);*out=availability;observations++;}
static int resource_core_vma_dependency_ready(const RcVmaDependency *d,const RcBudgetReadiness *r){
    return resource_core_budget_peak_progress(&d->peak,&d->before,r);
}
static VkResult resource_core_vma_create_buffer(const RcVmaRequest *req,int vma,const VkBufferCreateInfo *bi,
    const VmaAllocationCreateInfo *ai,VkBuffer *out,VmaAllocation *alloc,void *info,RcAdmissionStatus *status){
    (void)info;assert(!locks&&vma==1&&bi->size==4096&&ai->requiredFlags==VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    assert(req->producer==RC_BUDGET_BUFFER);attempts++;
    *req->dependency=(RcVmaDependency){.selected=1,.before=availability,.peak.amount[RC_BUDGET_GPU_FIRST]=4096};
    if(native_result!=VK_SUCCESS){*status=RC_ADMISSION_REFUSED;return native_result;}
    *out=17;*alloc=19;*status=RC_ADMISSION_OK;
    if(lose_host_during_create)host_mapping_current=0;
    if(replace_during_create){
        PoolRec *old=pool_index.e[0].buf;old->unpublished=1;
        pool_index.e[0].buf=replacement;
    }
    return VK_SUCCESS;
}
static void resource_core_vma_destroy_buffer(int vma,VkBuffer b,VmaAllocation a){
    assert(!locks&&vma==1&&b==17&&a==19);destructions++;
}
static int refuse(PoolRec *r,int why,const char *reason){
    (void)reason;r->refused_why=why;m_refused_n[why]++;m_refused_bytes[why]+=4096;return MS_REFUSED;
}
static _Thread_local PoolRec *mirror_pressure_demand;
'''

MAIN = r'''
static pidx_ent entry;
static uintptr_t high;
static PoolRec make_pool(void){return (PoolRec){.kind=POOL_IMPORT,.buf=1,.lo=4096,.import_lo=4096,.import_hi=8192,.gpa_rec=1,.gpa_host_coherent=1};}
static void select_pool(PoolRec *pool){
    entry=(pidx_ent){.lo=4096,.hi=8192,.buf=pool};high=8192;
    pool_index=(pidx){.e=&entry,.maxhi=&high,.n=1};
}
static void demand(PoolRec *r,VkBuffer expect){
    VkMemSpan span={.pool=r,.buffer=r->buf};
    mem_use_locked(&span);assert(span.buffer==expect);assert(!locks);
    if(expect==r->buf)import_selected++;
}
int main(void){
    availability=(RcBudgetReadiness){.records_available=1,.usable_mask=~0u};
    PoolRec pool=make_pool();select_pool(&pool);
    demand(&pool,1);assert(pool.mstate==MS_PENDING&&attempts==1&&mirror_pressure_first==&pool);
    service_busy=1;vk_mem_mirror_service_pressure();
    assert(service_calls==1&&!evictions&&!pressure_refs&&mirror_pressure_first==&pool);
    service_busy=0;vk_mem_mirror_service_pressure();
    assert(service_calls==2&&evictions==1&&!pressure_refs&&!mirror_pressure_first);
    vk_mem_mirror_service_pressure();assert(service_calls==2);
    for(msub=2;msub<650;msub++){demand(&pool,1);demand(&pool,1);}
    assert(attempts==1&&observations==648&&import_selected==1297);
    availability.available.amount[RC_BUDGET_GPU_FIRST+1]=8192;
    demand(&pool,1);assert(attempts==1);
    availability.available.amount[RC_BUDGET_GPU_FIRST]=8192;
    demand(&pool,1);assert(attempts==1); // Same submission keeps the imported provider.
    msub++;native_result=VK_SUCCESS;
    demand(&pool,17);assert(attempts==2&&copies==1&&remaps==1&&nm==1&&pool.mstate==MS_ON);
    assert(!mirror_pressure_first&&!mirror_pressure_last);

    PoolRec old=make_pool(),fresh=make_pool();
    select_pool(&old);replacement=&fresh;replace_during_create=1;msub++;
    demand(&old,1);assert(attempts==3&&destructions==1&&old.mstate==MS_REFUSED&&old.unpublished);
    assert(pool_index.e[0].buf==&fresh&&fresh.mstate==MS_NONE&&!old.mbuf&&!old.malloc_);
    assert(copies==1&&remaps==1&&nm==1);

    replace_during_create=0;
    PoolRec unmapped=make_pool();
    unmapped.host_backing=&unmapped;
    select_pool(&unmapped);
    lose_host_during_create=1;msub++;
    mirror_attempt_locked(&unmapped,1);
    assert(attempts==4&&destructions==2&&unmapped.mstate==MS_REFUSED&&!unmapped.mbuf&&!host_held);
    assert(copies==1&&remaps==1&&nm==1);
    lose_host_during_create=0;host_mapping_current=1;
    PoolRec exact=make_pool();
    exact.host_backing=&exact;
    select_pool(&exact);
    mirror_attempt_locked(&exact,1);
    assert(attempts==5&&copies==2&&remaps==2&&nm==2&&exact.mstate==MS_ON&&!host_held);

    PoolRec projection_refused=make_pool();
    projection_refused.host_backing=&projection_refused;
    select_pool(&projection_refused);
    deny_projection=1;msub++;
    mirror_attempt_locked(&projection_refused,1);
    assert(attempts==6&&destructions==3&&projection_refused.mstate==MS_REFUSED&&!projection_refused.mbuf);
    assert(projection_refused.gpa_rec==1&&!projection_refused.mirror_incarnation&&!host_held);
    assert(copies==2&&remaps==2&&nm==2); // No new native upload after a refused projection.
    deny_projection=0;
    select_pool(&fresh);
    native_result=VK_ERROR_DEVICE_LOST;msub++;
    demand(&fresh,1);assert(lost&&fresh.mstate==MS_REFUSED&&!mirror_pressure_first);
    PoolRec after=make_pool();select_pool(&after);lost=0;stop=1;msub++;
    demand(&after,1);assert(attempts==7&&!after.mbuf&&!mirror_pressure_first);
    puts("mirror preparation: relevant progress, submit-stable import, exact publication and cancellation passed");
}
'''


def main():
    vma = (ROOT / 'runtime/resource_core_vma.h').read_text()
    dependency = re.search(r'typedef struct RcVmaDependency \{[^}]*\} RcVmaDependency;', vma, re.S)[0]
    names = ('mirror_pool_entry', 'mirror_pool_alone', 'mirror_pressure_remove', 'mirror_pressure_add',
             'mirror_reclaim', 'mirror_pressure_register', 'vk_mem_mirror_service_pressure',
             'mirror_finish_backing', 'mirror_create_backing', 'mirror_create', 'mirror_retry_due',
             'mirror_import_scene', 'mirror_attempt_locked', 'mem_use_locked')
    code = PREFIX.replace('/* dependency type */', dependency) + '\n'.join(map(function, names)) + MAIN
    cc = Path(os.environ['CC'])
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get('PATH', ''))
    with tempfile.TemporaryDirectory(prefix='mirror-preparation-') as directory:
        source = Path(directory) / 'check.c'
        source.write_text(code)
        exe = source.with_suffix('.exe')
        subprocess.run([str(cc), '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror', '-fsanitize=undefined',
                        '-fsanitize-trap=all', '-I' + str(ROOT / 'runtime'), str(source), '-o', str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == '__main__':
    main()

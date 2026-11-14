#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define VK_POOL_TEX 16
#define VK_POOL_SAMP 4
#define VK_POOL_KINDS 2
#define VK_NULL_HANDLE 0
#define POOL_LOAD_END 2
#include "vk_pool_retry.h"
typedef uintptr_t VkSampler;
typedef uintptr_t VkDescriptorSet;
#include "vk_pool_publication.h"
typedef struct { uint64_t group,local,prepared,adopted,retry_attempts,retry_success; } PoolStats;
typedef struct DrawCtx { PoolStats pool_stats; } DrawCtx;
static DrawCtx fixture_context,*dc=&fixture_context;
typedef struct { uint64_t fence; DrawCtx *dc; } VkExec;
static VkExecPoolView group_view,local_view;
static unsigned pool_sync_calls,lock_calls;
static int pool_mt;
static int vk_exec_shared(void){return 1;}
static int vk_exec_in_chunk(void){return 1;}
static VkExecPoolView *vk_exec_pool_view(void){return &group_view;}
static VkExecPoolView *vk_exec_pool_local_view(void){return &local_view;}
static int vk_exec_pool_wait(uint64_t fence,uint32_t kinds,VkDescriptorSet *set){(void)fence;(void)kinds;(void)set;return 0;}
static void plat_mutex_lock(int *p){(void)p;lock_calls++;}
static void plat_mutex_unlock(int *p){(void)p;}
static struct {
    uint64_t tok[VK_POOL_TEX], set_synced[2], frame, frame_ns, cur_fence, logn, completed_generation;
    uint64_t n_defer_skips,n_visit,n_refill,n_by_reg,n_by_addr,n_frame_stops,n_walk[2],walk_visit[2],walk_refill[2],walk_ns[2];
    uint64_t n_ep_ent,n_unres_walks,unres_ns,n_ep,ep_ns,n_fresh,n_slots,n_full,full_ns,dirty_pos,sync_ns,catch_ns;
    uint64_t n_gwalk,n_gnouni,n_grep,n_gchanged,n_gskip,n_gskip_slots,n_gchk_pending,n_gchk_writes,n_gchk_bad,n_gchk_groups;
    uint32_t kinds,gen,epoch,sgen,unres_mg;
    int ready,all,retry,nunres,unres_more,gmode,cur,sealed,layout,publication_exhausted;
    uint8_t unres_on[VK_POOL_TEX],unres_in[VK_POOL_TEX];
    int unres[VK_POOL_TEX];
    VkSampler samp[VK_POOL_SAMP];
    VkDescriptorSet set;
} PL;
static struct { VkSampler samp; } Z={1};
static struct { uint64_t frames; } gfx_counters;
static VkPoolRetry pool_retry;
static uint32_t pool_defer[VK_POOL_TEX],vk_pool_kinds_used,gen,ep,sg,mg;
static uint64_t pool_ready_generation,pool_takes,pool_load_made,pool_load_frame0,ticks,pool_retry_peak;
static int pool_loading,slot_count,dirty[VK_POOL_TEX],ndirty,walk_ok,limit,budget,alias_ready;
static unsigned need[VK_POOL_TEX],filled[VK_POOL_TEX],fills[VK_POOL_TEX],unresolved_calls;
static uint64_t tokens[VK_POOL_TEX];
static uint64_t kern_now_ns(void){return ++ticks;}
static uint64_t pool_create_ns(void){return 100;}
static int pool_create_budget(void){return limit;}
static int pool_load_backlog(void){return 1000;}
static int pool_on(void){return 1;}
static int pool_fast(void){return 1;}
static int pool_group(void){return 0;}
static void vk_texture_lock(void){}
static void vk_texture_unlock(void){}
static void vk_texture_pool_ctx(int x){(void)x;}
static uint32_t vk_texture_slots_gen(void){return gen;}
static uint32_t vk_texture_epoch(void){return ep;}
static uint32_t vk_sampler_gen(void){return sg;}
static uint32_t mem_map_gen(void){return mg;}
static int vk_texture_slot_count_locked(void){return slot_count;}
static int vk_sampler_slot_count_locked(void){return 0;}
static VkSampler vk_sampler_by_id_locked(int id){(void)id;return 0;}
static uint64_t vk_texture_slot_token(int id){assert(id>=0&&id<slot_count);return tokens[id];}
static void vk_texture_create_budget_locked(int n){budget=n;}
static int vk_texture_create_deferred_locked(void){return 0;}
static void vk_texture_groups_unstamp_locked(void){}
static int vk_texture_pool_fresh_locked(void){return slot_count;}
static void vk_texture_by_id_locked(int id,void *a,void *b){(void)id;(void)a;(void)b;}
static int vk_texture_slot_retry_locked(int id){(void)id;unresolved_calls++;return alias_ready;}
static uint64_t vk_texture_dirty_pos_locked(void){return PL.dirty_pos+1;}
static int vk_texture_dirty_slots_locked(uint64_t *pos,void(*slot)(int,void*),void(*group)(int,void*),void *arg){
    (void)group;if(!walk_ok)return 0;(*pos)++;
    for(int i=0;i<ndirty;i++)slot(dirty[i],arg);ndirty=0;return 1;
}
typedef struct { int head,n; uint32_t stamp,fmask; } VkSlotGroup;
static VkSlotGroup slot_group;
static int group_next[VK_POOL_TEX];
static VkSlotGroup *vk_texture_group_locked(int g){(void)g;return &slot_group;}
static int vk_texture_group_next_locked(int id){return group_next[id];}
static int pool_group_check(void){return 0;}
static void pool_write(uint32_t k,uint32_t id,uintptr_t view,VkSampler samp){(void)k;(void)id;(void)view;(void)samp;}
static void pool_flush(void){}
static void rt_log(const char *s,...){(void)s;}
static int vk_pool_set_layout(void){PL.layout=1;return 1;}
static int pool_take_set(VkExec *ex,uint64_t *from){*from=0;PL.cur_fence=ex->fence;PL.set=ex->fence+1;pool_sync_calls++;return 0;}
static void pool_catch_up(uint64_t a,uint64_t b){(void)a;(void)b;}
static int pool_fill(VkExec *ex,uint32_t id,uint32_t mask,uint64_t tok){
    (void)ex;(void)tok;assert(pool_ready_generation==0);fills[id]++;
    if((need[id]&mask)&&budget==0)return 0;
    if(need[id]&mask){if(budget>0)budget--;need[id]&=~mask;}
    filled[id]|=mask;return 1;
}
typedef struct { VkExec *ex; uint32_t used; int n,retry,spent; } PoolDirty;
static void pool_sync_entries(VkExec *ex);
static void pool_sync(VkExec *ex);
#include "pool_under_test.h"

static void reset(void){
    memset(&PL,0,sizeof PL);memset(&pool_retry,0,sizeof pool_retry);memset(pool_defer,0,sizeof pool_defer);
    memset(need,0,sizeof need);memset(filled,0,sizeof filled);memset(fills,0,sizeof fills);
    PL.ready=PL.layout=1;PL.kinds=PL.gen=PL.epoch=PL.sgen=1;PL.dirty_pos=1;
    vk_pool_kinds_used=gen=ep=sg=mg=1;slot_count=VK_POOL_TEX;walk_ok=1;ndirty=0;limit=0;
    gfx_counters.frames=1;pool_ready_generation=pool_takes=0;alias_ready=unresolved_calls=0;pool_loading=0;
    vk_pool_view_reset(&group_view);vk_pool_view_reset(&local_view);pool_sync_calls=lock_calls=0;memset(dc,0,sizeof *dc);
    for(int i=0;i<slot_count;i++)PL.tok[i]=tokens[i]=((uint64_t)1<<32)|1;
}
static void change(int id){tokens[id]++;dirty[ndirty++]=id;gen++;need[id]=vk_pool_kinds_used;}
static void sync_next(VkExec *ex){ex->fence++;ep++;gfx_counters.frames++;pool_sync(ex);}
static void queue_oracle(void){
    VkPoolRetry q={0};uint32_t rng=1;uint8_t present[VK_POOL_TEX]={0};
    for(int k=0;k<100000;k++){
        rng=rng*1664525u+1013904223u;unsigned id=(rng>>8)%VK_POOL_TEX;
        if(rng&1){vk_pool_retry_add(&q,id);present[id]=1;}else{vk_pool_retry_remove(&q,id);present[id]=0;}
        unsigned count=0,seen[VK_POOL_TEX]={0},prev=0;
        for(uint32_t p=q.head;p;p=q.next[p-1]){assert(p<=VK_POOL_TEX&&!seen[p-1]++);assert(q.prev[p-1]==prev);prev=p;count++;}
        assert(count==q.count&&prev==q.tail);
        for(unsigned i=0;i<VK_POOL_TEX;i++)assert(!!vk_pool_retry_has(&q,i)==present[i]);
    }
    memset(&q,0,sizeof q);for(unsigned i=0;i<4;i++)vk_pool_retry_add(&q,i);
    unsigned opportunities=0;for(unsigned k=0;k<4;k++){vk_pool_retry_begin(&q);opportunities|=1u<<(q.head-1);}
    assert(opportunities==15);q.epoch=UINT32_MAX;memset(q.attempted,0xff,sizeof q.attempted);vk_pool_retry_begin(&q);
    assert(q.epoch==1&&vk_pool_retry_attempt(&q,0)&&!vk_pool_retry_attempt(&q,0));
}
int main(void){
    queue_oracle();reset();VkExec ex={0};
    change(4);dirty[ndirty++]=4;pool_sync(&ex);
    assert(pool_retry.count==1&&PL.retry&&fills[4]==1&&PL.n_full==0&&pool_takes==0);
    // A re-registration is resolved using its new token, even though its earlier dirty entry was consumed.
    tokens[4]++;limit=1;sync_next(&ex);
    assert(!PL.retry&&!pool_retry.count&&PL.tok[4]==tokens[4]&&fills[4]==2&&PL.n_full==0&&pool_takes==1);
    // New kinds retain failed refills despite an unchanged descriptor token, then finish through the sparse queue.
    reset();ex.fence=0;vk_pool_kinds_used=3;need[3]=2;pool_sync(&ex);
    assert(PL.retry&&vk_pool_retry_has(&pool_retry,3)&&PL.n_full==1);
    limit=1;sync_next(&ex);assert(!PL.retry&&filled[3]==3&&PL.n_full==1);
    // A deferred group representative with an unchanged token cannot certify that all members are current.
    reset();ex.fence=0;slot_group=(VkSlotGroup){.head=0,.n=2,.stamp=1,.fmask=1};group_next[0]=1;group_next[1]=-1;
    need[0]=1;vk_pool_retry_add(&pool_retry,0);vk_pool_retry_begin(&pool_retry);budget=0;
    PoolDirty pd={&ex,1,slot_count,0,0};pool_dirty_group(0,&pd);
    assert(!slot_group.stamp&&vk_pool_retry_has(&pool_retry,0)&&!PL.n_gskip&&fills[0]==1);
    reset();ex.fence=0;limit=1;vk_pool_kinds_used=3;pool_sync(&ex);assert(PL.n_full==1);
    // Ring overflow still takes the conservative full scan.
    walk_ok=0;change(2);sync_next(&ex);assert(PL.n_full==2&&PL.tok[2]==tokens[2]);
    // An obsolete queued slot cannot keep retirement blocked forever.
    reset();ex.fence=0;change(12);pool_sync(&ex);assert(PL.retry);slot_count=8;sync_next(&ex);assert(!PL.retry);
    // Old pending slots progress ahead of continuously arriving dirty slots under a one-entry budget.
    reset();ex.fence=0;change(0);pool_sync(&ex);limit=1;
    for(int id=1;id<8;id++){change(id);sync_next(&ex);assert(!vk_pool_retry_has(&pool_retry,(uint32_t)(id-1)));assert(PL.retry);}
    sync_next(&ex);assert(!PL.retry&&pool_takes==1);
    // The old frame budget remains authoritative: no re-creation attempt in the same exhausted frame.
    reset();ex.fence=0;change(5);pool_sync(&ex);unsigned before=fills[5];PL.frame_ns=1000;limit=1;
    pool_sync(&ex);assert(fills[5]==before&&PL.retry);sync_next(&ex);assert(!PL.retry);
    // Unchanged mappings do not suppress periodic unresolved-alias retries.
    reset();ex.fence=0;PL.nunres=1;PL.unres[0]=6;PL.unres_on[6]=PL.unres_in[6]=1;PL.unres_mg=mg;
    for(int i=0;i<9;i++)sync_next(&ex);assert(unresolved_calls>0&&PL.nunres==1);
    alias_ready=1;for(int i=0;i<9;i++)sync_next(&ex);assert(!PL.nunres);
    // A later global completion leaves the group immutable. One private adoption avoids per-draw locking/refresh.
    reset();ex.fence=1;pool_sync_draw(&ex);assert(pool_sync_calls==1&&group_view.ready);
    VkExecPoolView immutable=group_view;
    pool_sync(&ex);assert(pool_sync_calls==2);pool_sync_draw(&ex);
    unsigned locked=lock_calls;
    for(int i=0;i<100;i++)pool_sync_draw(&ex);
    assert(pool_sync_calls==2&&lock_calls==locked&&!memcmp(&immutable,&group_view,sizeof immutable));
    vk_pool_kinds_used=3;pool_sync_draw(&ex);assert(pool_sync_calls==3&&local_view.kinds==3);
    locked=lock_calls;for(int i=0;i<100;i++)pool_sync_draw(&ex);
    assert(pool_sync_calls==3&&lock_calls==locked&&!memcmp(&immutable,&group_view,sizeof immutable));
    PL.completed_generation=UINT64_MAX;pool_sync(&ex);assert(!vk_exec_pool_generation()&&PL.publication_exhausted);
    puts("pool retry: actual sync preserves dirty overlap, registration, new kinds, overflow, obsolete slots, budgets, streaming fairness, retirement and alias retry; 100000 queue operations passed");
    return 0;
}

#include <assert.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "host.h"
#include "vk_exec.h"

#define MAX_CB 8
#define MAX_EVENTS 256
#define MAX_CALLBACKS 64
enum { EV_BEGIN, EV_END, EV_DRAW, EV_CONTACT, EV_BARRIER };
typedef struct { int kind; uint32_t value; uint64_t weight; } Event;
static struct { Event e[MAX_EVENTS]; unsigned n; } streams[MAX_CB];
static struct { uint64_t fence; void (*fn)(void *); void *arg; } callbacks[MAX_CALLBACKS];
static unsigned callback_count;
static int gpu_finished, groups_merged, contact_pending;
static uint64_t records[4][2];
static uint64_t gpu_values[16384], gpu_members[16384];
static uint8_t gpu_available[16384], gpu_begun[16384];
static unsigned resets[16384];
static VkCtx context;
static uint32_t ring_start;
static int batch_mode=1;
static unsigned leaders_accepted,leaders_refused;
unsigned rt_trace;

static void event(VkCommandBuffer cb, int kind, uint32_t value, uint64_t weight) {
    unsigned i = (unsigned)(uintptr_t)cb;
    assert(i < MAX_CB && streams[i].n < MAX_EVENTS);
    streams[i].e[streams[i].n++] = (Event){kind,value,weight};
}

void rt_log(const char *fmt, ...) { (void)fmt; }
void rt_fatal(const char *fmt, ...) {
    va_list ap; va_start(ap,fmt); vfprintf(stderr,fmt,ap); va_end(ap); fputc('\n',stderr); abort();
}
int rt_knob(const char *key,const char *def) { (void)key; (void)def; return 0; }
const char *nvn_op_name(uint32_t op) { (void)op; return "unexpected fixture opcode"; }
int cfg_bool(const char *key) { (void)key; return 1; }
uint64_t kern_now_ns(void) { return 1000000000; }
VkCtx *vk_ctx(void) { return &context; }
int vk_sync_effect_late(void) { return 0; }
void vk_device_lost_poll(void) {}
void vk_device_lost_check(VkResult value,const char *where) { (void)where; assert(value == VK_SUCCESS); }
void vk_mem_count(int cls,int use,uint64_t size) { (void)cls; (void)use; (void)size; }
int vk_mem_resolve(uint64_t addr,uint64_t size,VkMemSpan *out) {
    if (addr < (uint64_t)(uintptr_t)records || addr + size > (uint64_t)(uintptr_t)(records + 4)) return 0;
    memset(out,0,sizeof *out); return 1;
}
int vk_contact_pending(void) { return contact_pending; }
void vk_contact_flush(VkExec *ex) {
    if (contact_pending) { assert(!ex->in_pass); contact_pending=0; event(ex->cb,EV_CONTACT,0,0); }
}
void vk_exec_turn(void) {}
static void unexpected_transfer(VkExec *ex,const void *payload,...) {
    (void)ex; (void)payload; assert(!"Unrelated transfer reached query fixture");
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(VkPhysicalDevice device,VkPhysicalDeviceFeatures2 *f) {
    (void)device; f->features.occlusionQueryPrecise=1;
    ((VkPhysicalDeviceVulkan12Features *)f->pNext)->hostQueryReset=1;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateQueryPool(VkDevice device,const VkQueryPoolCreateInfo *info,
                                               const VkAllocationCallbacks *alloc,VkQueryPool *pool) {
    (void)device; (void)info; (void)alloc; *pool=(VkQueryPool)(uintptr_t)1; return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyQueryPool(VkDevice d,VkQueryPool p,const VkAllocationCallbacks *a) {
    (void)d; (void)p; (void)a;
}
VKAPI_ATTR void VKAPI_CALL vkCmdBeginQuery(VkCommandBuffer cb,VkQueryPool pool,uint32_t q,VkQueryControlFlags flags) {
    (void)pool; (void)flags; event(cb,EV_BEGIN,q,0);
}
VKAPI_ATTR void VKAPI_CALL vkCmdEndQuery(VkCommandBuffer cb,VkQueryPool pool,uint32_t q) {
    (void)pool; event(cb,EV_END,q,0);
}
VKAPI_ATTR void VKAPI_CALL vkResetQueryPool(VkDevice d,VkQueryPool p,uint32_t first,uint32_t count) {
    (void)d; (void)p; assert(gpu_finished && groups_merged);
    for (uint32_t q=first;q<first+count;q++) { assert(q<16384); resets[q]++; gpu_available[q]=0; }
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetQueryPoolResults(VkDevice d,VkQueryPool p,uint32_t first,uint32_t count,
                                                   size_t bytes,void *data,VkDeviceSize stride,VkQueryResultFlags flags) {
    (void)d; (void)p; (void)bytes; (void)flags; assert(gpu_finished && groups_merged);
    for (uint32_t i=0;i<count;i++) {
        uint64_t *out=(uint64_t *)((uint8_t *)data + i*stride);
        assert(first+i<16384); out[0]=gpu_values[first+i]; out[1]=gpu_available[first+i];
    }
    return VK_SUCCESS;
}

#include "query_under_test.h"

void vk_completion_add_result(uint64_t fence,void (*fn)(void *),void *arg) {
    assert(callback_count<MAX_CALLBACKS && !gpu_finished);
    callbacks[callback_count++]=(__typeof__(callbacks[0])){fence,fn,arg};
}

typedef struct { uint32_t op; int arg; unsigned id; } Op;
typedef struct {
    uint64_t fence,seq,members,value;
    int kind,reset,record;
} Result;
typedef struct {
    Result result[MAX_CALLBACKS]; unsigned n;
    uint64_t record_values[4][2];
    int counting,reported,keep;
    unsigned contacts;
} Observation;

static void reset_fixture(void) {
    free(oq_seg); oq_seg=NULL; oq_nseg=oq_capseg=0;
    for(unsigned g=0;g<GS_GROUPS;g++) {
        free(GS[g].s);
        for(unsigned c=0;c<CP_CHUNKS;c++) { free(CP[g][c].seg); free(CP[g][c].q); }
    }
    memset(GS,0,sizeof GS); memset(CP,0,sizeof CP); memset(&XW,0,sizeof XW); XW.open=-1;
    memset((void *)oq_busy,0,sizeof oq_busy); memset(&OQB,0,sizeof OQB); memset(pending,0,sizeof pending);
    memset(streams,0,sizeof streams); memset(callbacks,0,sizeof callbacks); memset(records,0,sizeof records);
    memset(gpu_values,0,sizeof gpu_values); memset(gpu_members,0,sizeof gpu_members);
    memset(gpu_available,0,sizeof gpu_available); memset(gpu_begun,0,sizeof gpu_begun); memset(resets,0,sizeof resets);
    oq_state=1; oq_pool=(VkQueryPool)(uintptr_t)1; oq_open=-1; oq_next=ring_start; oq_count=0;
    oq_counting=oq_reported=oq_keep=oq_lost=0; report_seq=0; cond_frame=7; gs_on=-1;
    callback_count=0; gpu_finished=groups_merged=contact_pending=0; oq_batch_mode=batch_mode;
}

static void barrier(VkExec *ex) {
    if(ex->in_pass) { vk_exec_oq_pass_end(ex); ex->in_pass=0; }
    event(ex->cb,EV_BARRIER,0,0);
}

static void execute(VkExec *ex,Op op,int front) {
    if(op.op==OP_DRAW) {
        if(front || ex->render_disabled) return;
        ex->in_pass=1; vk_exec_oq_draw(ex);
        event(ex->cb,EV_DRAW,op.id,(uint64_t)op.arg); return;
    }
    p_reset_counter reset={.type=NVN_COUNTER_TYPE_SAMPLES_PASSED};
    p_report_counter report={.type=op.op==OP_REPORT_COUNTER && op.arg<0 ? NVN_COUNTER_TYPE_TIMESTAMP : NVN_COUNTER_TYPE_SAMPLES_PASSED};
    report.addr=(uint64_t)(uintptr_t)records[op.arg<0 ? -op.arg-1 : op.arg];
    p_flag enable={.value=op.arg};
    const void *payload=op.op==OP_RESET_COUNTER ? (const void *)&reset :
                        op.op==OP_RENDER_ENABLE ? (const void *)&enable : (const void *)&report;
    if(front) vk_exec_xfer_front(ex,op.op,payload);
    else op_xfer_impl(ex,op.op,payload);
}

typedef struct { op_hdr h; uint8_t payload[64]; } Command;
static Command commands[128];
static struct { uint64_t drained,out; } Q;
static struct { uint64_t serial_draws; } RW;
static void ahead_set_rt(VkExec *ex,const p_set_rt *p) {
    (void)ex; (void)p; assert(!"Unrelated target binding reached query fixture");
}
void vk_exec_ahead_op(VkExec *ex,uint32_t op,const void *payload) {
    (void)ex; (void)op; (void)payload; assert(!"Unrelated predicted state reached query fixture");
}
int vk_exec_draw_pool_leader_needed(VkExec *ex) { (void)ex; return 1; }
int vk_exec_draw_pool_requirement_known(VkExec *ex) { return !ex->render_disabled; }
void vk_exec_draw_pool_leader_finished(VkExec *ex) { (void)ex; leaders_accepted++; }
void vk_exec_draw_pool_refused(VkExec *ex,int reason) { (void)ex; (void)reason; }
void vk_exec_draw_pool_prefix_op(uint32_t op) { (void)op; }
void vk_exec_barrier(VkExec *ex) { barrier(ex); }

static void make_commands(const Op *ops,unsigned n) {
    assert(n<128); memset(commands,0,sizeof commands);
    for(unsigned i=0;i<n;i++) {
        Op op=ops[i]; commands[i].h.op=op.op; commands[i].h.size=sizeof(Command);
        void *payload=commands[i].payload;
        if(op.op==OP_DRAW) { uint32_t values[2]={(uint32_t)op.arg,op.id}; memcpy(payload,values,sizeof values); }
        else if(op.op==OP_RENDER_ENABLE) *(p_flag *)payload=(p_flag){.value=op.arg};
        else if(op.op==OP_RESET_COUNTER) *(p_reset_counter *)payload=(p_reset_counter){.type=NVN_COUNTER_TYPE_SAMPLES_PASSED};
        else {
            assert(op.op==OP_REPORT_COUNTER);
            *(p_report_counter *)payload=(p_report_counter){
                .type=op.arg<0 ? NVN_COUNTER_TYPE_TIMESTAMP : NVN_COUNTER_TYPE_SAMPLES_PASSED,
                .addr=(uint64_t)(uintptr_t)records[op.arg<0 ? -op.arg-1 : op.arg]};
        }
    }
}
static void op_one(VkExec *ex,const op_hdr *h) {
    const void *payload=nvn_op_payload(h);
    if(h->op==OP_DRAW) {
        const uint32_t *v=payload; execute(ex,(Op){OP_DRAW,(int)v[0],v[1]},0);
    } else op_xfer_impl(ex,h->op,payload);
}
#include "query_leader_under_test.h"

static void group(VkExec *replay,const Op *ops,unsigned n,unsigned cut,int reverse) {
    if(!n) return;
    if(cut>n) cut=n;
    unsigned start[2]={0,cut}, end[2]={cut,n};
    if(!cut || cut==n) { start[0]=0; end[0]=n; start[1]=end[1]=n; }
    unsigned chunks=end[1]>start[1] ? 2 : 1;
    uint64_t state[2]; uint32_t slots[2]={0},slot0[2]; int disabled[2];
    vk_exec_xfer_group(0); groups_merged=0;
    for(unsigned c=0;c<chunks;c++) {
        state[c]=vk_exec_xfer_chunk_state(); disabled[c]=replay->render_disabled;
        for(unsigned k=start[c];k<end[c];k++) {
            if(ops[k].op==OP_DRAW) slots[c]+=vk_exec_xfer_counting() && !replay->render_disabled;
            else execute(replay,ops[k],1);
        }
    }
    vk_exec_xfer_group(-1);
    for(unsigned c=0;c<chunks;c++) slot0[c]=vk_exec_xfer_slots(slots[c]);
    for(unsigned j=0;j<chunks;j++) {
        unsigned c=reverse ? chunks-1-j : j;
        VkExec ex={.cb=(VkCommandBuffer)(uintptr_t)(2+c),.fence=replay->fence,.render_disabled=disabled[c]};
        vk_exec_xfer_worker(1,NULL,0,state[c],slot0[c],slots[c],(int)c,0);
        for(unsigned k=start[c];k<end[c];k++) execute(&ex,ops[k],0);
        barrier(&ex);
        vk_exec_xfer_worker(0,NULL,0,0,0,0,0,0);
    }
    vk_exec_xfer_group_merge(0,(int)chunks); groups_merged=1;
}

static void gpu_execute(Observation *out) {
    assert(groups_merged);
    for(unsigned cb=0;cb<MAX_CB;cb++) {
        int open=-1;
        for(unsigned i=0;i<streams[cb].n;i++) {
            Event e=streams[cb].e[i];
            if(e.kind==EV_BEGIN) { assert(open<0 && !gpu_begun[e.value]); open=(int)e.value; gpu_begun[e.value]=1; }
            else if(e.kind==EV_END) { assert(open==(int)e.value); gpu_available[e.value]=1; open=-1; }
            else if(e.kind==EV_DRAW && open>=0) { gpu_values[open]+=e.weight; gpu_members[open]|=1ull<<e.value; }
            else if(e.kind==EV_CONTACT) out->contacts++;
        }
        assert(open<0);
    }
    gpu_finished=1;
}

static int record_index(uint64_t addr) {
    if(!addr) return -1;
    for(int i=0;i<4;i++) if(addr==(uint64_t)(uintptr_t)records[i]) return i;
    assert(!"Unknown guest record"); return -1;
}

static Observation run(const Op *ops,unsigned n,unsigned cut,int reverse,int leader,int prior) {
    reset_fixture(); Observation out={0};
    // Use a real eligible group size, retaining semantic IDs for the interesting draws.
    // Zero-sample padding makes the production split threshold exercise honest while
    // leaving independent arithmetic expectations unchanged.
    Op expanded[128]; memcpy(expanded,ops,n*sizeof *ops);
    uint32_t draws=0;
    for(unsigned i=0;i<n;i++) draws+=ops[i].op==OP_DRAW;
    while(draws<=2*CHUNK_DRAWS) { expanded[n++]=(Op){OP_DRAW,0,63}; draws++; }
    ops=expanded;
    VkExec ex={.cb=(VkCommandBuffer)(uintptr_t)1,.fence=17};
    oq_counting=prior; oq_count=prior ? 5 : 0;
    if(prior) execute(&ex,(Op){OP_DRAW,3,0},0);
    unsigned begin=0;
    if(leader) {
        make_commands(ops,n); Q.drained=Q.out=0;
        uint32_t before=draws;
        const op_hdr *after=pool_leader_run(&ex,&commands[0].h,&commands[n].h,&draws);
        begin=(unsigned)((const uint8_t *)after-(const uint8_t *)commands)/sizeof(Command);
        assert(draws==before-(begin!=0));
        if(!begin) leaders_refused++;
        if(ops[0].op==OP_DRAW) assert(begin==1); // positive actual-splitter coverage cannot silently fall back
#ifdef REQUIRE_COUNTER_PREFIX
        if(ops[0].op==OP_REPORT_COUNTER || ops[0].op==OP_RESET_COUNTER) assert(begin>0);
#endif
    }
    barrier(&ex); // original group_start boundary, after any canonical leader
    group(&ex,ops+begin,n-begin,cut>begin ? cut-begin : 0,reverse);
    ex.cb=(VkCommandBuffer)(uintptr_t)4;
    execute(&ex,(Op){OP_REPORT_COUNTER,3,0},0); // consumes a worker tail when the group has no report
    barrier(&ex); groups_merged=1;
    assert(!gpu_finished);
    for(unsigned i=0;i<4;i++) assert(records[i][0]==0 && records[i][1]==0);
    gpu_execute(&out);
    // Samples-passed reports remain unknown until their newest callback, rather than reading
    // an older zero from guest memory. Timestamp records resolve to zero within this frame.
    for(unsigned i=0;i<4;i++) {
        uint64_t value=99,addr=(uint64_t)(uintptr_t)records[i];
        const Pending *p=pending_slot(addr,0);
        if(p) assert(counter_word(addr,&value)==(p->unknown ? 2 : 1) && value==0);
    }
    for(unsigned i=0;i<callback_count;i++) {
        Result *r=&out.result[out.n++]; r->fence=callbacks[i].fence;
        assert(r->fence==17);
        uint64_t addr;
        if(callbacks[i].fn==oq_cb) {
            OqSegment *s=callbacks[i].arg; r->kind=1; r->reset=s->reset; r->seq=s->seq; addr=s->addr;
            for(unsigned q=0;q<s->n;q++) { assert(gpu_available[s->q[q]]); r->members|=gpu_members[s->q[q]]; }
        } else {
            assert(callbacks[i].fn==counter_cb); CounterWrite *w=callbacks[i].arg;
            r->kind=2; r->seq=w->seq; addr=w->addr;
        }
        r->record=record_index(addr);
        callbacks[i].fn(callbacks[i].arg);
        if(r->record>=0) {
            r->value=records[r->record][0];
            const Pending *p=pending_slot(addr,0);
            assert(p && p->written>=r->seq);
            uint64_t value=99;
            int status=counter_word(addr,&value);
            if(p->unknown && p->written<p->seq) assert(status==2 && value==0);
            else if(!p->unknown && p->frame==cond_frame) assert(status==1 && value==0);
            else assert(status==1 && value==records[r->record][0]);
        }
    }
    oq_reset_flush(context.device);
    for(unsigned q=0;q<16384;q++) if(gpu_begun[q]) assert(resets[q]==1 && !oq_busy[q]);
    out.counting=oq_counting; out.reported=oq_reported; out.keep=oq_keep;
    memcpy(out.record_values,records,sizeof records);
    assert(!oq_nseg && !GS[0].tail_wait);
    return out;
}

static void pending_visibility_check(void) {
    reset_fixture();
    VkExec ex={.cb=(VkCommandBuffer)(uintptr_t)1,.fence=41};
    records[0][0]=1;
    p_report_counter stamp={.type=NVN_COUNTER_TYPE_TIMESTAMP,.addr=(uint64_t)(uintptr_t)records[0]};
    report_counter(&ex,&stamp);
    ex.fence=42;
    report_counter(&ex,&stamp);
    Pending *p=pending_slot(stamp.addr,0);
    assert(p && p->seq==2 && !p->written && callback_count==2);
    assert(callbacks[0].fence==41 && callbacks[1].fence==42);
    assert(records[0][0]==1); // enqueue does not publish to guest memory
    uint64_t value=99;
    assert(counter_word(stamp.addr,&value)==1 && value==0);
    groups_merged=gpu_finished=1;
    callbacks[0].fn(callbacks[0].arg);
    assert(p->written==1 && p->written<p->seq);
    records[0][0]=1; // the guest may rearm after the older callback
    assert(counter_word(stamp.addr,&value)==1 && value==0);
    callbacks[1].fn(callbacks[1].arg);
    assert(p->written==2 && records[0][0]==0);
    records[0][0]=1;
    assert(counter_word(stamp.addr,&value)==1 && value==0);
    cond_frame++;
    assert(counter_word(stamp.addr,&value)==1 && value==1);

    reset_fixture();
    ex.fence=43;
    p_report_counter samples={.type=NVN_COUNTER_TYPE_SAMPLES_PASSED,.addr=(uint64_t)(uintptr_t)records[0]};
    p_render_cond condition={.addr=samples.addr,.mode=1};
    report_counter(&ex,&samples);
    assert(counter_word(samples.addr,&value)==2); // no samples available yet: conservatively render
    assert(cond_enabled(&condition));
    groups_merged=gpu_finished=1;
    callbacks[0].fn(callbacks[0].arg);
    assert(counter_word(samples.addr,&value)==1 && value==0);
    assert(!cond_enabled(&condition));
}

static void conservative_visibility_check(void) {
    for(int busy=0;busy<2;busy++) {
        reset_fixture(); Observation out={0};
        VkExec ex={.cb=(VkCommandBuffer)(uintptr_t)1,.fence=99};
        oq_counting=1;
        uint32_t q=oq_next%OQ_N;
        if(busy) oq_busy[q]=1;
        execute(&ex,(Op){OP_DRAW,0,1},0); barrier(&ex);
        execute(&ex,(Op){OP_REPORT_COUNTER,0,0},0);
        groups_merged=1; gpu_execute(&out);
        if(!busy) gpu_available[q]=0; // completed query unavailable: never cull on an unknown zero
        assert(callback_count==1);
        callbacks[0].fn(callbacks[0].arg);
        oq_reset_flush(context.device);
        assert(records[0][0]==1);
        if(busy) assert(!gpu_begun[q] && oq_busy[q]); // another owner's slot is not consumed/reset
        else assert(gpu_begun[q] && resets[q]==1 && !oq_busy[q]);
    }
}

static void equal(const Observation *a,const Observation *b,const char *name,unsigned cut,int reverse) {
    if(memcmp(a,b,sizeof *a)) {
        fprintf(stderr,"Mismatch %s cut %u reverse %d callbacks %u/%u\n",name,cut,reverse,a->n,b->n);
        for(unsigned i=0;i<a->n || i<b->n;i++) fprintf(stderr,"  %u: members %llx/%llx value %llu/%llu reset %d/%d\n",i,
            (unsigned long long)a->result[i].members,(unsigned long long)b->result[i].members,
            (unsigned long long)a->result[i].value,(unsigned long long)b->result[i].value,a->result[i].reset,b->result[i].reset);
        abort();
    }
}

int main(void) {
    static const struct { const char *name; int prior; unsigned n; Op ops[10]; } cases[]={
        {"already counting",1,4,{{OP_DRAW,7,1},{OP_DRAW,11,2},{OP_DRAW,13,3},{OP_REPORT_COUNTER,0,0}}},
        {"reset prefix",0,5,{{OP_RESET_COUNTER,0,0},{OP_DRAW,7,1},{OP_DRAW,11,2},{OP_DRAW,13,3},{OP_REPORT_COUNTER,0,0}}},
        {"report reset prefix",1,6,{{OP_REPORT_COUNTER,0,0},{OP_RESET_COUNTER,0,0},{OP_DRAW,7,1},{OP_DRAW,11,2},{OP_REPORT_COUNTER,1,0},{OP_DRAW,13,3}}},
        {"worker tail",1,3,{{OP_DRAW,7,1},{OP_DRAW,11,2},{OP_DRAW,13,3}}},
        {"repeated report",1,6,{{OP_DRAW,7,1},{OP_REPORT_COUNTER,0,0},{OP_REPORT_COUNTER,1,0},{OP_DRAW,11,2},{OP_REPORT_COUNTER,2,0},{OP_DRAW,13,3}}},
        {"timestamp prefix",1,4,{{OP_REPORT_COUNTER,-1,0},{OP_DRAW,7,1},{OP_DRAW,11,2},{OP_REPORT_COUNTER,1,0}}},
        {"same-address reports",1,6,{{OP_DRAW,7,1},{OP_REPORT_COUNTER,0,0},{OP_RESET_COUNTER,0,0},{OP_DRAW,11,2},{OP_REPORT_COUNTER,0,0},{OP_DRAW,13,3}}},
        {"disabled leader",1,5,{{OP_RENDER_ENABLE,0,0},{OP_DRAW,7,1},{OP_RENDER_ENABLE,1,0},{OP_DRAW,11,2},{OP_REPORT_COUNTER,0,0}}},
        {"two resets",1,5,{{OP_RESET_COUNTER,0,0},{OP_RESET_COUNTER,0,0},{OP_DRAW,7,1},{OP_DRAW,11,2},{OP_REPORT_COUNTER,0,0}}}
    };
    unsigned comparisons=0;
    for(unsigned mode=0;mode<4;mode++) {
      ring_start=(mode&1) ? OQ_N-2 : 0; batch_mode=(int)(mode>>1);
      for(unsigned c=0;c<sizeof cases/sizeof cases[0];c++) {
        for(unsigned cut=1;cut<cases[c].n;cut++) for(int reverse=0;reverse<2;reverse++) {
            Observation control=run(cases[c].ops,cases[c].n,cut,reverse,0,cases[c].prior);
            Observation leader=run(cases[c].ops,cases[c].n,cut,reverse,1,cases[c].prior);
            equal(&control,&leader,cases[c].name,cut,reverse); comparisons++;
            if(c==1) assert(leader.record_values[0][0]==31);
            if(c==2) assert(leader.record_values[0][0]==8 && leader.record_values[1][0]==18);
        }
      }
    }
    pending_visibility_check();
    conservative_visibility_check();
    reset_fixture();
    printf("query ordering: %u production-splitter comparisons (%u accepted, %u refused) with ring wrap/batching; overlapping-fence reports, pending/rearmed records and unavailable/busy conservative visibility passed\n",comparisons,leaders_accepted,leaders_refused);
    return 0;
}

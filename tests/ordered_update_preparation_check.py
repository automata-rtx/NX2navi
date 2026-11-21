"""Exercise ordered scratch preparation for uniform updates without a GPU."""
from pathlib import Path
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'runtime/vk_exec.c').read_text()

PREFIX = r'''
#include <assert.h>
#include <limits.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include "resource_core_budget.h"
#include "resource_core_registry.h"
typedef struct {int selected;} RcVmaDependency;
typedef uint64_t VkCommandBuffer;
typedef struct {VkCommandBuffer cb;uint64_t fence;int recorded,in_pass,dirty,rt_resolved;} VkExec;
typedef struct {uint32_t op,size;} op_hdr;
typedef struct {uint64_t dst,size;uint8_t data[1];} p_update_ub;
typedef struct {int32_t prim,first,count,base_instance,instances;} p_draw;
typedef struct {VkExec *ex;uint64_t bytes;} VkDrawScratchWaitContext;
typedef struct {int retained;} VkDrawScratchUpdate;
typedef struct { void *hold; uint64_t span; } VkExecUpdateDestination;
typedef enum {VK_SYNC_PROGRESS_UNAVAILABLE,VK_SYNC_PROGRESS_COMPLETE,VK_SYNC_PROGRESS_STOPPED,
    VK_SYNC_PROGRESS_STOPPED_AFTER_CUT} VkSyncProgressResult;
typedef enum {VK_SUCCESS,VK_NOT_READY,VK_ERROR_TOO_LARGE} VkResult;
enum {VK_NULL_HANDLE=0,VK_EXEC_DIRTY_ALL=127};
enum {OP_UPDATE_UB,OP_DRAW};
static const void *nvn_op_payload(const op_hdr *h){return h+1;}
static int part_start,split_draws,split_at,split_n;
static uint64_t part_t0,done,older;
static int remap_after_wait;
static int calls,prior_waits,cuts,empty_markers,services,cancels,passes,new_buffers,lost,stop,prefix,refused_attempts;
static VkSyncProgressResult cut_result,service_result;
static VkResult first_result;
static uint64_t cut_tail;
static VkDrawScratchUpdate scratch;
static int stop_after_grant,failed,captures,drops,remapped,refusals;
static jmp_buf crash;
static uint64_t kern_now_ns(void){return 123;}
static void vk_exec_draw_new_cb(void){new_buffers++;}
static int vk_sync_stopping(void){return stop;}
static int vk_device_is_lost(void){return lost;}
static void rt_fatal(const char *fmt,...){assert(fmt);failed++;longjmp(crash,1);}
static RcRegistryResult vk_exec_draw_update_select(const p_update_ub *u,VkExecUpdateDestination *d){
    assert(u->dst==0x2000&&!d->hold); captures++;d->hold=&scratch;d->span=37;return RC_REGISTRY_OK;
}
static void vk_exec_draw_update_drop(VkExecUpdateDestination *d){if(d->hold)drops++;*d=(VkExecUpdateDestination){0};}
static int vk_mem_span_capture_current(const void *hold,const uint64_t *span){
    assert(hold==&scratch&&*span==37);return !remapped;
}
static void vk_exec_required_graphics_unavailable(VkExec *ex,uint32_t op,const char *reason){
    assert(ex->fence==7&&op==OP_UPDATE_UB&&reason);refusals++;
}
static VkResult vk_exec_draw_update_prepare(VkExec *ex,const p_update_ub *u,VkExecUpdateDestination *d,
                                            VkDrawScratchUpdate **g,RcVmaDependency *dep){
    assert(ex->fence==7&&u->dst==0x2000&&u->size==1&&u->data[0]==99&&!*g);
    assert(d->hold==&scratch&&d->span==37);
    if(++calls<=refused_attempts&&first_result!=VK_SUCCESS){dep->selected=1;return first_result;}
    if(remapped)return VK_ERROR_TOO_LARGE;
    *d=(VkExecUpdateDestination){0};
    *g=&scratch;scratch.retained=1;if(stop_after_grant)stop=1;return VK_SUCCESS;
}
static void vk_exec_draw_update_cancel(VkExec *ex,VkDrawScratchUpdate *r){
    assert(ex->fence==7&&r==&scratch&&scratch.retained);scratch.retained=0;cancels++;
}
static VkResult vk_exec_draw_index_prepare(VkExec *ex,const p_draw *op,VkDrawScratchUpdate **grant,RcVmaDependency *dep){
    (void)ex;(void)op;(void)grant;(void)dep;assert(0);return VK_ERROR_TOO_LARGE;
}
static void vk_exec_draw_index_finish(VkExec *ex,VkDrawScratchUpdate *r){(void)ex;(void)r;assert(0);}
void resource_core_budget_readiness(int required,RcBudgetReadiness *out){assert(required);*out=(RcBudgetReadiness){0};}
static uint64_t vk_exec_draw_scratch_prior_useful(VkExec *ex,uint64_t bytes,const RcVmaDependency *dep,
          const RcBudgetReadiness *ready){assert(ex->fence==7&&bytes==1&&dep->selected&&ready);return older;}
static int vk_exec_draw_scratch_prefix_useful(VkExec *ex,uint64_t bytes,const RcVmaDependency *dep,
          const RcBudgetReadiness *ready){assert(ex->fence==7&&bytes==1&&dep->selected&&ready);return prefix;}
static int vk_exec_draw_scratch_spare_ready(void *p){const VkDrawScratchWaitContext *c=p;assert(c->ex->fence==7&&c->bytes==1);return 0;}
static uint64_t vk_sync_fence_done(void){return done;}
static VkSyncProgressResult vk_sync_progress_wait_prior(uint64_t f){assert(f==older&&f>done);done=f;prior_waits++;remapped=remap_after_wait;return VK_SYNC_PROGRESS_COMPLETE;}
static void vk_exec_pass_end(VkExec *ex){assert(ex->in_pass);ex->in_pass=0;passes++;}
static VkSyncProgressResult vk_sync_progress_split(VkExec *ex,VkCommandBuffer *tail,int empty_tail){
    assert(ex->cb==44&&!*tail);
    if(empty_tail){assert(ex->recorded==part_start&&!cut_tail);empty_markers++;}
    else assert(!ex->in_pass&&ex->recorded>part_start);
    cuts++;*tail=cut_tail;return cut_result;
}
static VkSyncProgressResult vk_sync_progress_wait_capacity(const RcVmaDependency *dep,int(*ready)(void *),void *op){
    assert(dep->selected&&ready&&!ready(op));services++;return service_result;
}
'''

MAIN = r'''
static VkExec reset(void){
    calls=prior_waits=cuts=empty_markers=services=cancels=passes=new_buffers=lost=stop=prefix=stop_after_grant=failed=0;
    captures=drops=refusals=remapped=remap_after_wait=0;
    refused_attempts=1;
    older=done=cut_tail=part_t0=0;part_start=0;split_draws=9;split_at=10;split_n=32;scratch.retained=0;
    first_result=VK_NOT_READY;cut_result=service_result=VK_SYNC_PROGRESS_UNAVAILABLE;
    return (VkExec){.cb=44,.fence=7,.recorded=3,.in_pass=1,.rt_resolved=1};
}
static int prepare_update(VkExec *ex,VkDrawScratchUpdate **grant){
    const struct {op_hdr h;p_update_ub payload;} update={
        .h={OP_UPDATE_UB,sizeof(op_hdr)+sizeof(p_update_ub)},.payload={0x2000,1,{99}}
    };
    return prepare_ordered_scratch(ex,&update.h,update.payload.size,grant);
}
int main(void){
    VkDrawScratchUpdate *grant=NULL;VkExec ex=reset();older=5;
    assert(prepare_update(&ex,&grant)&&grant==&scratch&&calls==2&&prior_waits==1&&!cuts&&!services);

    assert(captures==1&&!drops);

    grant=NULL;ex=reset();older=5;remap_after_wait=1;
    assert(!prepare_update(&ex,&grant)&&!grant&&captures==1&&calls==2&&prior_waits==1&&drops==1&&refusals==1);

    // Required progress also works when a single descriptor set prevents every ordinary asynchronous split.
    grant=NULL;ex=reset();prefix=1;cut_tail=55;cut_result=VK_SYNC_PROGRESS_COMPLETE;
    assert(prepare_update(&ex,&grant)&&grant==&scratch&&calls==2&&cuts==1&&!prior_waits&&!services);
    assert(ex.cb==55&&part_start==ex.recorded&&part_t0==123&&passes==1&&new_buffers==1&&ex.dirty==VK_EXEC_DIRTY_ALL);
    assert(!split_draws&&split_at==split_n);

    grant=NULL;ex=reset();prefix=1;cut_tail=55;cut_result=VK_SYNC_PROGRESS_STOPPED_AFTER_CUT;
    assert(!prepare_update(&ex,&grant)&&!grant&&calls==1&&cuts==1&&!services);
    assert(ex.cb==55&&part_start==ex.recorded&&new_buffers==1&&captures==1&&drops==1); // Abandon the actual tail, even after a stop.

    grant=NULL;ex=reset();prefix=1;service_result=VK_SYNC_PROGRESS_COMPLETE;
    assert(prepare_update(&ex,&grant)&&calls==2&&cuts==1&&services==1);
    assert(ex.cb==44&&!part_start&&!new_buffers); // An unavailable cut leaves the original part open.

    grant=NULL;ex=reset();part_start=ex.recorded;prefix=1;refused_attempts=2;
    cut_result=service_result=VK_SYNC_PROGRESS_COMPLETE;
    assert(prepare_update(&ex,&grant)&&calls==3&&cuts==1&&empty_markers==1&&services==1);
    assert(ex.cb==44&&ex.in_pass&&!new_buffers&&!passes); // Prove prior parts without reopening the current tail.

    grant=NULL;ex=reset();ex.recorded=0;prefix=1;service_result=VK_SYNC_PROGRESS_COMPLETE;
    assert(prepare_update(&ex,&grant)&&calls==2&&!cuts&&services==1);

    grant=NULL;ex=reset();service_result=VK_SYNC_PROGRESS_COMPLETE;
    assert(prepare_update(&ex,&grant)&&calls==2&&services==1&&!cuts&&!prior_waits);

    grant=NULL;ex=reset();first_result=VK_SUCCESS;stop_after_grant=1;
    assert(!prepare_update(&ex,&grant)&&!grant&&cancels==1&&!scratch.retained&&!services);

    grant=NULL;ex=reset();first_result=VK_ERROR_TOO_LARGE;
    if(!setjmp(crash)){prepare_update(&ex,&grant);assert(0);}
    assert(failed==1&&calls==1&&captures==1&&drops==1&&!services&&!cuts&&!prior_waits);
    puts("ordered uniform update: actual progress, exact operation, tail adoption and grant cancellation passed");
}
'''


def main():
    begin = SOURCE.index('static void progress_adopt_tail(VkExec *ex, VkCommandBuffer cb) {')
    adopt_end = SOURCE.index('\n}\n', begin) + len('\n}\n')
    body = SOURCE.index('static int prepare_ordered_scratch(', begin)
    end = SOURCE.index('\n}\n', body) + len('\n}\n')
    assert 'vk_exec_pool_split' not in SOURCE[begin:end]
    walk = SOURCE[end:SOURCE.index('int vk_exec_record(', end)]
    prepare = walk.index('prepare_ordered_scratch(ex, h, scratch_bytes, &scratch_grant)')
    declare = walk.index('resource_declare(ex, h,')
    assert walk.index('u->size > h->size - header') < walk.index('scratch_bytes = u->size') < prepare
    assert walk.index('scratch_bytes = nvn_generated_draw_index_bytes(a)') < prepare
    commit = 'vk_exec_draw_update_commit(ex, a, scratch_grant)'
    assert prepare < declare < walk.index(commit) < walk.index('vk_exec_effects_recorded(effect,0)')
    assert walk.count(commit) == 1
    assert 'return &walk_stopped;' in walk and 'if (stop == &walk_stopped) return stop;' in walk
    cc = Path(os.environ['CC'])
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get('PATH', ''))
    with tempfile.TemporaryDirectory(prefix='ordered-update-') as directory:
        code = Path(directory) / 'check.c'
        code.write_text(PREFIX + SOURCE[begin:adopt_end] + SOURCE[body:end] + MAIN)
        exe = code.with_suffix('.exe')
        subprocess.run([str(cc), '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror', '-fsanitize=undefined',
                        '-fsanitize-trap=all', '-I' + str(ROOT / 'runtime'), str(code), '-o', str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == '__main__':
    main()

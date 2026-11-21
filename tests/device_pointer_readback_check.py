"""Exercise actual bounded device scalar staging, read order and native-cut ownership."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=ROOT/'runtime/vk_exec.c')
parser.add_argument('--runtime', type=Path, default=ROOT/'runtime')
args = parser.parse_args()
text = args.source.read_text()
pass_struct = text[text.index('struct VkExecGraphicsPass {'):text.index('static int graphics_pass_hooks_active(')]
pass_start = text.index('int vk_exec_graphics_pass_current(')
pass_end = text.index('\n}', pass_start)+len('\n}')
progress_start = text.find('static VkSyncProgressResult upload_staging_progress(')
progress = text[progress_start:text.index('static void progress_adopt_tail(', progress_start)] if progress_start >= 0 else ''

start = text.index('struct VkExecPointerReadback {\n    VkPreparedStaging stage;\n    uint32_t owners;')
end = text.index('VkMemMirrorInputPolicy vk_exec_mirror_input_policy(', start)
phase_start = text.index('VkResult vk_exec_pointer_phase_prepare(')
phase_end = text.index('void vk_exec_resource_handoff(', phase_start)

FIXTURE = r'''
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk.h"
#include "vk_exec.h"
static VkExec execution, *x_cur=&execution;
static unsigned rw_chunk;
static int recording, coherent, proof_open, stopped, mutate_after_cut, parked;
enum StagingState {
    ALLOCATE, NO_SUPPLIER, RETURNED, ALREADY_RETURNED, ALLOCATION_REMAP,
    LOST_CAPACITY, REMAPPED, CANCELED, VACATED, ADVANCED,
};
static enum StagingState staging_state;
void rt_fatal(const char *format, ...) { fputs(format, stderr); abort(); }
static unsigned groups, staging_plans, staging_allocs, staging_releases, waits, reads, native_uses, copies, barriers, invalidates;
static unsigned parks, reuse_checks, supplier_checks;
static VkSyncProgressResult completion;
struct VkMemSpanCapture { unsigned refs; };
static VkMemSpanCapture capture={1};
static VkCtx context;
static RcRegistryPosition position;
static const uint64_t gpu_va=UINT64_C(0x8000004000), sample=UINT64_C(0x1111222233334444);
static uint64_t mapped_value;
static const VkBuffer source_buffer=(VkBuffer)(uintptr_t)101, stage_buffer=(VkBuffer)(uintptr_t)102;
static const VmaAllocation device_allocation=(VmaAllocation)(uintptr_t)11, staging_allocation=(VmaAllocation)(uintptr_t)12;
static struct {void(*done)(void*);void(*cancel)(void*,int);void *arg;} pending;
static const RcInterval range={UINT64_C(0x8000004000),2,{91,16,8}};
enum {PW_OP, PW_SPLIT};
static int part_start;
static unsigned phase_notes, phase_queries, adopted;
enum PassChange { SAME_PASS, RT_CHANGED, LOAN_REPLACED, SERIAL_CHANGED, HOOK_STARTED, ONLY_EXTENT_CHANGED };
static enum PassChange change_after_cut;
static int loan_current, hooks;
static uint32_t noatt_width;
static VkTextureTargetLoan *const target_loan=(VkTextureTargetLoan *)(uintptr_t)51;
static int graphics_pass_hooks_active(void){return hooks;}
static uint32_t graphics_pass_selected_hooks(const VkExec *ex,const VkExecGraphicsPassShape *shape){
    assert(ex==x_cur&&shape->pass_noatt);return hooks;
}
typedef struct VkDlaaPreparedObserver VkDlaaPreparedObserver;
static int vk_dlaa_prepared_observer_current(const VkExec *ex,const VkExecGraphicsPassShape *shape,
                                             const VkDlaaPreparedObserver *observer){
    (void)ex;(void)shape;assert(!observer);return 0;
}
int vk_texture_target_loan_current(const VkTextureTargetLoan *loan){assert(loan==target_loan);return loan_current;}
int vk_texture_target_loan_private(const VkTextureTargetLoan *loan){assert(loan==target_loan);return 0;}
const VkTexImage *vk_texture_alias_target_replacement(const VkTextureAliasRead *seed){assert(!seed);return NULL;}
int vk_texture_alias_target_current(VkTextureAliasRead *seed,VkImage image){(void)image;assert(!seed);return 0;}
int vk_texture_target_loan_backing_current(const VkTextureTargetLoan *loan,const RcRegistryImageWriteExtent *extents,
    uint32_t count){assert(loan==target_loan&&!extents&&!count);return loan_current;}
void vk_exec_noatt_dims(const VkExec *ex,uint32_t *w,uint32_t *h){assert(ex==x_cur);*w=noatt_width;*h=16;}

static RcRegistryBufferPhaseOutcome phase_outcome;
static const RcRegistryBufferPhaseExtent task={.writer={9,1},.interval={UINT64_C(0x8000004000),2,{91,16,8}}};
VkCtx *vk_ctx(void){return &context;}
RcRegistry *rc_registry_runtime(void){return (RcRegistry *)(uintptr_t)1;}
static void group_drain(VkExec *ex,int why){assert(ex==x_cur&&(why==PW_OP||why==PW_SPLIT));groups++;}
int vk_sync_stopping(void){return stopped;}
volatile int32_t vk_lost;
int vk_exec_recording_failed(void){return !recording;}
void vk_device_lost_check(VkResult result,const char *reason){assert(result!=VK_SUCCESS&&reason);vk_lost=1;}
RcRegistryPosition vk_exec_resource_position(void){return position;}
int rc_registry_provisional_read_current(RcRegistry *r,RcRegistryPosition read){
    assert(r==rc_registry_runtime());return proof_open&&read.scene==position.scene&&read.ordinal==position.ordinal;
}
RcRegistryResult rc_registry_read_device_buffer_at(RcRegistry *r,const RcInterval *ranges,uint32_t count,RcRegistryPosition read){
    assert(r==rc_registry_runtime()&&count==1&&ranges==&range&&read.scene==position.scene&&read.ordinal==position.ordinal);
    reads++;return proof_open?RC_REGISTRY_OK:RC_REGISTRY_BLOCKED;
}
int vk_mem_span_capture_current(const VkMemSpanCapture *value,const VkMemSpan *span){
    assert(value==&capture&&value->refs&&span->buffer==source_buffer);return coherent;
}
int vk_mem_span_record_input(VkMemSpanCapture *value,const VkMemSpan *span){
    return vk_mem_span_capture_current(value,span);
}
VkMemSpanCapture *vk_mem_span_capture_retain(VkMemSpanCapture *value){assert(value==&capture&&value->refs==1);value->refs++;return value;}
void vk_exec_record_native_input_provider(VkExec *ex,VkMemSpanCapture **value){
    assert(ex==x_cur&&*value==&capture&&capture.refs==2);capture.refs--;*value=NULL;native_uses++;
}
VkResult vk_staging_prepare_plan(VkDeviceSize size,VkPreparedStaging *stage,RcAdmissionVector *peak){
    assert(size==8&&!stage->buffer);stage->size=size;peak->amount[RC_BUDGET_RAM]=65536;staging_plans++; return VK_SUCCESS;
}
VkResult vk_staging_prepare_allocate(VkPreparedStaging *stage,RcBudgetTicket *budget,int required){
    assert(stage->size==8&&required&&!budget&&!stage->buffer);
    staging_allocs++;
    if(staging_state==ALLOCATION_REMAP)coherent=0;
    if(staging_state!=ALLOCATE)return VK_NOT_READY;
    stage->buffer=stage_buffer;stage->allocation=staging_allocation;stage->map=(uint8_t *)&mapped_value;return VK_SUCCESS;
}
int vk_staging_prepare_reuse(VkDeviceSize size,VkPreparedStaging *stage){
    assert(size==8&&!stage->buffer&&!parked&&!copies);
    reuse_checks++;
    if(staging_state!=ALREADY_RETURNED && (staging_state!=RETURNED || !waits))return 0;
    *stage=(VkPreparedStaging){.size=size,.buffer=stage_buffer,.allocation=staging_allocation,
        .map=(uint8_t *)&mapped_value,.from_cache=1};
    return 1;
}
void vk_staging_prepare_cancel(VkPreparedStaging *stage){
    if(stage->buffer){assert(stage->buffer==stage_buffer);staging_releases++;}
    *stage=(VkPreparedStaging){0};
}
uint64_t vk_staging_prepare_supplier(VkDeviceSize size,uint64_t after){
    assert(size==8&&!after&&!parked&&!copies);
    supplier_checks++;
    return staging_state==NO_SUPPLIER ? 0 : 11;
}
int vk_staging_prepare_wait(uint64_t supplier){
    assert(supplier==11&&parked&&!copies);waits++;
    assert(position.scene==5&&position.ordinal==24);
    if(staging_state==REMAPPED)coherent=0;
    if(staging_state==CANCELED)stopped=1;
    if(staging_state==ADVANCED)position.ordinal+=2;
    return staging_state!=VACATED;
}
VkSyncProgressResult vk_sync_progress_wait_upload_staging(uint64_t upload){
    assert(!parked&&!copies&&recording&&upload);
    parked=1;parks++;
    const int retired=vk_staging_prepare_wait(upload);
    parked=0;
    if(stopped)return VK_SYNC_PROGRESS_STOPPED;
    return retired?VK_SYNC_PROGRESS_COMPLETE:VK_SYNC_PROGRESS_UNAVAILABLE;
}
void vk_exec_pass_end(VkExec *ex){assert(x_cur==ex);ex->in_pass=0;}
void vk_exec_barrier(VkExec *ex){assert(x_cur==ex&&!ex->in_pass);barriers++;}
void vk_completion_add_owned(uint64_t fence,void(*done)(void*),void(*cancel)(void*,int),void *arg){
    assert(fence==x_cur->fence&&!pending.arg&&done&&cancel&&arg);pending.done=done;pending.cancel=cancel;pending.arg=arg;
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer cmd,VkBuffer src,VkBuffer dst,uint32_t count,const VkBufferCopy *copy){
    assert(cmd==x_cur->cb&&src==source_buffer&&dst==stage_buffer&&count==1&&copy->srcOffset==16&&copy->size==8);copies++;
}
VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier2(VkCommandBuffer cmd,const VkDependencyInfo *info){
    assert(cmd==x_cur->cb&&info->bufferMemoryBarrierCount==1&&info->pBufferMemoryBarriers->buffer==stage_buffer);
    assert(info->pBufferMemoryBarriers->dstStageMask==VK_PIPELINE_STAGE_2_HOST_BIT);barriers++;
}
VkSyncProgressResult vk_exec_graphics_pool_progress(VkExec *ex,uint64_t prior){
    assert(ex==x_cur&&!prior&&copies==1&&barriers==2&&ex->recorded==1&&pending.arg);
    if(completion==VK_SYNC_PROGRESS_COMPLETE)mapped_value=sample;
    if(mutate_after_cut)coherent=0;
    return completion;
}
VkSyncProgressResult vk_sync_progress_split_before(VkExec *ex,VkCommandBuffer *tail,uint64_t ordinal){
    assert(ex==x_cur&&ordinal==position.ordinal&&copies==1&&barriers==2&&ex->recorded==2&&pending.arg);
    if(completion==VK_SYNC_PROGRESS_COMPLETE)mapped_value=sample;
    if(mutate_after_cut)coherent=0;
    switch(change_after_cut){
    case SAME_PASS: break;
    case RT_CHANGED: ex->rt.color[0]=(const tex_t *)(uintptr_t)73; break;
    case LOAN_REPLACED: loan_current=0; break;
    case SERIAL_CHANGED: ex->pass_serial++; break;
    case HOOK_STARTED: hooks=1; break;
    case ONLY_EXTENT_CHANGED: noatt_width++; break;
    }
    if(completion==VK_SYNC_PROGRESS_COMPLETE||completion==VK_SYNC_PROGRESS_STOPPED_AFTER_CUT)
        *tail=(VkCommandBuffer)(uintptr_t)81;
    return completion;
}
static void progress_adopt_tail(VkExec *ex,VkCommandBuffer cb){
    assert(ex==x_cur&&cb==(VkCommandBuffer)(uintptr_t)81&&!adopted);ex->cb=cb;adopted++;
}
RcRegistryResult rc_registry_buffer_phase_note(RcRegistry *r,const RcRegistryBufferPhaseExtent *receipt,
    uint32_t count,RcRegistryPosition read,RcRegistryBufferPhaseOutcome outcome){
    assert(r==rc_registry_runtime()&&receipt==&task&&count==1&&read.ordinal==position.ordinal&&adopted==1&&!phase_notes);
    phase_notes++;phase_outcome=outcome;return RC_REGISTRY_OK;
}
RcRegistryResult rc_registry_read_device_buffer_phase(RcRegistry *r,const RcInterval *input,uint32_t input_count,
    const RcRegistryBufferPhaseExtent *receipt,uint32_t count,RcRegistryPosition read){
    assert(r==rc_registry_runtime()&&input==&range&&input_count==1&&receipt==&task&&count==1&&
           read.ordinal==position.ordinal&&phase_notes==1&&phase_outcome==RC_REGISTRY_BUFFER_PHASE_COMPLETED);
    phase_queries++;return RC_REGISTRY_OK;
}
VkResult vmaInvalidateAllocation(VmaAllocator allocator,VmaAllocation allocation,VkDeviceSize start,VkDeviceSize size){
    assert(allocator==context.vma&&allocation==staging_allocation&&!start&&size==8&&mapped_value==sample);invalidates++;return VK_SUCCESS;
}
'''

CASES = r'''
static void reset(void){
    assert(!pending.arg&&capture.refs==1);
    execution=(VkExec){.fence=5,.resource_op=12,.cb=(VkCommandBuffer)(uintptr_t)71,.in_pass=1};
    position=(RcRegistryPosition){5,24};
    recording=coherent=proof_open=1; stopped=vk_lost=mutate_after_cut=parked=0;
    staging_state=ALLOCATE;
    groups=staging_plans=staging_allocs=staging_releases=waits=reads=native_uses=copies=barriers=invalidates=0;
    parks=reuse_checks=supplier_checks=0;
    phase_notes=phase_queries=adopted=part_start=hooks=0;
    change_after_cut=SAME_PASS; loan_current=1;noatt_width=16;
    mapped_value=0;completion=VK_SYNC_PROGRESS_COMPLETE;
}
static VkSyncProgressResult read_pointer(uint64_t *value){
    VkMemSpan span={.buffer=source_buffer,.alloc=device_allocation,.offset=16,.avail=8,.cls=VK_MCLS_DEVICE};
    return vk_exec_read_device_pointer(x_cur,gpu_va,&span,&capture,&range,1,position,value);
}
static void actual_task_read(void){
    const VkMemSpan span={.buffer=source_buffer,.alloc=device_allocation,.offset=16,.avail=8,.cls=VK_MCLS_DEVICE};
    for(unsigned test=0;test<9;test++){
        reset();execution.in_pass=0;execution.resource_next=execution.resource_op;
        execution.pass_serial=31;
        const VkCommandBuffer prefix=execution.cb;
        VkExecGraphicsPass captured={.fence=execution.fence,.barrier_cb=prefix,.barrier_pass=31,
            .barrier_was_open=1,.check_selected_hooks=1,.shape={.pass_noatt=1,.pass_w=16,.pass_h=16},
            .bindings={{.binding={.image=255}}},.loan={target_loan}};
        const VkExecGraphicsPass original=captured;
        assert(vk_exec_graphics_pass_current(x_cur,&captured));
        VkExecPointerReadback *staging=NULL;
        assert(vk_exec_pointer_phase_prepare(&staging)==VK_SUCCESS&&staging&&staging_allocs==1);
        uint64_t value=7;
        assert(vk_exec_read_device_pointer_phase(x_cur,&staging,&span,&capture,&range,1,&task,1,position,&value,&captured)==
               VK_SYNC_PROGRESS_UNAVAILABLE&&staging&&value==7&&!copies&&!phase_notes);
        assert(captured.barrier_cb==prefix&&vk_exec_graphics_pass_current(x_cur,&captured));
        execution.recorded=1;
        execution.cb=(VkCommandBuffer)(uintptr_t)72;
        assert(vk_exec_read_device_pointer_phase(x_cur,&staging,&span,&capture,&range,1,&task,1,position,&value,&captured)==
               VK_SYNC_PROGRESS_UNAVAILABLE&&staging&&value==7&&!copies&&!phase_notes);
        execution.cb=prefix;
        completion=test==1?VK_SYNC_PROGRESS_STOPPED_AFTER_CUT:test==2?VK_SYNC_PROGRESS_UNAVAILABLE:VK_SYNC_PROGRESS_COMPLETE;
        mutate_after_cut=test==3;
        if(test>=4)change_after_cut=(enum PassChange)(test-3);
        VkSyncProgressResult expected=test>=3?VK_SYNC_PROGRESS_UNAVAILABLE:completion;
        assert(vk_exec_read_device_pointer_phase(x_cur,&staging,&span,&capture,&range,1,&task,1,position,&value,&captured)==expected);
        assert(!staging&&copies==1&&barriers==2&&pending.arg&&adopted==(test!=2)&&phase_notes==(test!=2));
        assert(value==(test==0?sample:7)&&phase_queries==(test==0||test>=4));
        assert(captured.barrier_cb==(test==0?execution.cb:prefix));
        assert(!captured.activated&&!memcmp(&captured.shape,&original.shape,sizeof captured.shape)&&
               !memcmp(captured.loan,original.loan,sizeof captured.loan)&&
               !memcmp(captured.bindings,original.bindings,sizeof captured.bindings));
        if(test==0){
            assert(vk_exec_graphics_pass_current(x_cur,&captured));
            assert(!vk_exec_graphics_pass_current(x_cur,&original));
        }
        if(test==0||test>=3){
            assert(phase_outcome==RC_REGISTRY_BUFFER_PHASE_COMPLETED&&staging_releases==1&&invalidates==1);
            pending.done(pending.arg);
        }else{
            assert(!staging_releases&&!invalidates);
            if(test==1) assert(phase_outcome==RC_REGISTRY_BUFFER_PHASE_POSSIBLY_ISSUED);
            pending.cancel(pending.arg,test==1);
        }
        pending.arg=NULL;assert(staging_releases==1&&capture.refs==1);
    }
}
int main(void){
    actual_task_read();
    uint64_t value=7;
    reset();proof_open=0;
    assert(read_pointer(&value)==VK_SYNC_PROGRESS_UNAVAILABLE&&value==7&&!staging_allocs&&!copies);
    reset();staging_state=RETURNED;
    assert(read_pointer(&value)==VK_SYNC_PROGRESS_COMPLETE&&value==sample&&staging_allocs==1&&waits==1);
    assert(parks==1&&reuse_checks==2&&supplier_checks==1&&staging_plans==1);
    assert(groups==2&&copies==1&&barriers==2&&staging_releases==1&&native_uses==1&&invalidates==1&&position.ordinal==24);
    pending.done(pending.arg);pending.arg=NULL;assert(staging_releases==1);
    reset();staging_state=ALREADY_RETURNED;
    assert(read_pointer(&value)==VK_SYNC_PROGRESS_COMPLETE&&value==sample&&staging_allocs==1&&!parks&&!waits&&!supplier_checks);
    pending.done(pending.arg);pending.arg=NULL;
    reset();staging_state=NO_SUPPLIER;value=7;
    assert(read_pointer(&value)==VK_SYNC_PROGRESS_UNAVAILABLE&&value==7&&staging_allocs==1&&!copies&&!pending.arg);
    assert(!parks&&!waits&&reuse_checks==1&&supplier_checks==1);
    reset();staging_state=ALLOCATION_REMAP;
    assert(read_pointer(&value)==VK_SYNC_PROGRESS_UNAVAILABLE&&value==7&&staging_allocs==1&&!copies&&!pending.arg);
    assert(!parks&&!waits&&!reuse_checks&&!supplier_checks);
    for(enum StagingState state=LOST_CAPACITY;state<=ADVANCED;state++){
        reset();staging_state=state;value=7;
        const VkSyncProgressResult expected=state==CANCELED?VK_SYNC_PROGRESS_STOPPED:VK_SYNC_PROGRESS_UNAVAILABLE;
        assert(read_pointer(&value)==expected&&value==7&&staging_allocs==1&&staging_plans==1);
        assert(parks==1&&waits==1&&!parked&&!copies&&!pending.arg&&!staging_releases&&!native_uses);
        assert(reuse_checks==(state==LOST_CAPACITY?2u:1u));
    }
    reset();completion=VK_SYNC_PROGRESS_STOPPED_AFTER_CUT;value=3;
    assert(read_pointer(&value)==completion&&value==3&&pending.arg&&copies==1&&!staging_releases&&!invalidates);
    pending.cancel(pending.arg,1);pending.arg=NULL;assert(staging_releases==1);
    reset();mutate_after_cut=1;
    assert(read_pointer(&value)==VK_SYNC_PROGRESS_UNAVAILABLE&&value==3&&invalidates==1&&staging_releases==1);
    pending.done(pending.arg);pending.arg=NULL;
    puts("device storage pointer: exact local task cut, parked staging custody, unchanged read, remap and cancellation passed");
}
'''

cc=Path(os.environ['CC'])
env=dict(os.environ,PATH=str(cc.parent)+os.pathsep+os.environ.get('PATH',''))
with tempfile.TemporaryDirectory(prefix='device-pointer-readback-') as directory:
    source=Path(directory)/'check.c';exe=source.with_suffix('.exe')
    source.write_text(FIXTURE+pass_struct+text[pass_start:pass_end]+progress+
                      text[start:end]+text[phase_start:phase_end]+CASES)
    include=ROOT/'runtime'
    subprocess.run([str(cc),'-std=c11','-O1','-Wall','-Wextra','-Werror','-fsanitize=undefined','-fsanitize-trap=all',
                   '-I'+str(args.runtime),'-I'+str(include),'-I'+str(cc.parent.parent/'include/vma'),str(source),'-o',str(exe)],env=env,check=True)
    subprocess.run([str(exe)],env=env,check=True)

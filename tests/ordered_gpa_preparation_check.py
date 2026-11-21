"""Exercise the recorder's actual GPA-input park, handoff and retry boundaries."""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "runtime/vk_exec.c").read_text()
begin = SOURCE.index("typedef struct {\n    uintptr_t shader, pipeline, info;")
end = SOURCE.index("static int prepare_ordered_scratch(", begin)
walk = SOURCE[SOURCE.index("static const op_hdr *walk(", end):]
cancel = walk.index("vk_exec_effects_release(effect);")
resume = walk.index("prepare_ordered_gpa_source(ex, &gpa_prepare)")
assert cancel < resume < walk.index("gpa_attempted = gpa_prepare.request.read;")
assert walk.index("gpa_attempted = gpa_prepare.request.read;") < walk.index("gpa_retry_dispatch = gpa_prepare.dispatch;")
assert "!retry && !gpa_prepare.request.guard.id" in walk
draw = (ROOT / "runtime/vk_exec_draw.c").read_text()
for operation, emit in (("direct", "vkCmdDispatch("), ("indirect", "vkCmdDispatchIndirect(")):
    begin_dispatch = draw.index(f"static void dispatch_{operation}_emit(")
    end_dispatch = draw.index("\n}", begin_dispatch)
    native = draw[begin_dispatch:end_dispatch]
    assert native.index("flat3d_preflight(") < native.index("vk_exec_gpa_dispatch_binding(")
    assert native.index("vk_exec_gpa_dispatch_binding(") < native.index("dispatch_bind_selected_resources(")
    assert native.index("dispatch_bind_selected_resources(") < native.index("if (!bound") < native.index(emit)
binder_start = draw.index("static int dispatch_bind_selected_resources(")
binder = draw[binder_start:draw.index("\n}", binder_start)]
assert binder.index("bind_resources(") < binder.index("return !vk_texture_deferred() && !vk_exec_recording_failed();")
assert "|| vk_gpa_write_scene_may_need_prefix(ex->fence))" in SOURCE

prefix = r'''
#include <assert.h>
#include <stdio.h>
#include <vulkan/vulkan.h>
#include "resource_core_backing.h"
#include "resource_core_registry.h"
typedef struct {RcRegistryWrite guard;RcRegistryPosition producer,read;RcInterval source;} VkGpaPrefixRequest;
typedef struct {int variant;} GfxShader, VkPipeInfo;
typedef struct {
    VkBuffer buffer;
    VkDeviceSize offset, avail;
    void *alloc;
    VkDeviceSize alloc_offset;
    int cls;
    void *pool;
} VkMemSpan;
enum {RECORDING_RESOURCE_UNAVAILABLE = 1};
typedef struct {VkImage image;} VkTexImage;
typedef struct {VkImage image;const VkTexImage *img;} VkExecTarget;
typedef struct {
    uint64_t fence,resource_op,resource_next;
    int recorded;
    VkCommandBuffer cb;
    VkExtent2D tonemap_extent;
    unsigned hud_alpha_attachment,pass_ncolor;
    VkExecTarget color[8];
} VkExec;
typedef enum {VK_SYNC_PROGRESS_UNAVAILABLE,VK_SYNC_PROGRESS_COMPLETE,VK_SYNC_PROGRESS_STOPPED,
    VK_SYNC_PROGRESS_STOPPED_AFTER_CUT} VkSyncProgressResult;
typedef struct {RcInterval extent;} MemBackingCapture;
typedef struct VkExecGpaSampleScope VkExecGpaSampleScope;
typedef struct {VkMemSpan span;uint64_t mapping;} VkMemSpanCapture;
static VkMemSpanCapture argument, promoted;
static uint64_t argument_mapping=215;
static unsigned argument_holds;
static int bound_available;
static unsigned bound_captures;
static int provisional_current, defer_consumed;
static int vk_texture_deferred_take(void){defer_consumed++;return 1;}
static VkMemSpanCapture *vk_mem_span_capture(uint64_t address,uint64_t size,const VkMemSpan *selected){
    assert(address==0x3000&&size==sizeof(VkDispatchIndirectCommand)&&!argument_holds);
    argument=(VkMemSpanCapture){*selected,argument_mapping};argument_holds++;return &argument;
}
static RcRegistryResult vk_mem_span_capture_bound_input(uint64_t address,uint64_t size,const VkMemSpan *selected,
                                                        VkMemSpanCapture **out){
    assert(!*out);
    if(!bound_available)return RC_REGISTRY_UNSUPPORTED;
    *out=vk_mem_span_capture(address,size,selected);
    bound_captures++;return RC_REGISTRY_OK;
}
static int vk_mem_span_capture_current(const VkMemSpanCapture *capture,const VkMemSpan *selected){
    assert((capture==&argument||capture==&promoted)&&argument_holds>0);
    return argument_mapping==capture->mapping&&(!selected||
        (selected->buffer==capture->span.buffer&&selected->offset==capture->span.offset&&selected->pool==capture->span.pool));
}
static VkMemSpanCapture *vk_mem_span_capture_retain(VkMemSpanCapture *capture){
    assert(capture==&argument&&argument_holds>0);
    argument_holds++;return capture;
}
static void vk_mem_span_capture_release(VkMemSpanCapture **capture){
    if(*capture){assert(*capture==&argument&&argument_holds>0);argument_holds--;*capture=NULL;}
}
static VkExec *x_cur;
static struct {uint64_t drained,out;} Q;
static int recording_failed, shared, stopped, lost, recorded_effect, rewinds, barriers, cuts, settled, canceled;
static VkSyncProgressResult cut_result;
static RcRegistryResult reopen_result;
static uint64_t publication=77;
static MemBackingCapture captured={.extent={0x20000,13,{14,0x7000,16}}};
static const RcRegistryPosition producer={400,41},read={400,134};
static int vk_exec_shared(void){return shared;}
static int vk_sync_stopping(void){return stopped;}
static int vk_device_is_lost(void){return lost;}
static int vk_exec_recording_failed(void){return recording_failed;}
static RcRegistryPosition vk_exec_resource_position(void){return (RcRegistryPosition){x_cur->fence,x_cur->resource_op*2};}
static int vk_exec_effects_unrecorded(const void *e){return e && !recorded_effect;}
static int vk_gpa_write_prefix_ready(const VkGpaPrefixRequest *r){
    return r->guard.id==10 && !rc_registry_position_compare(r->producer,producer) &&
           !rc_registry_position_compare(r->read,read);
}
static uint64_t vk_texture_registration_publication(void){return publication;}

static RcBackingResult mem_backing_capture(uint64_t a,uint64_t bytes,MemBackingCapture **out){
    assert(a==0x20000 && bytes==16);*out=&captured;return RC_BACKING_OK;
}
static const RcInterval *mem_backing_intervals(const MemBackingCapture *c,uint32_t *n){*n=1;return &c->extent;}
static int mem_backing_current(const MemBackingCapture *c){return c==&captured;}
static void mem_backing_release(MemBackingCapture **c){*c=NULL;}
static void vk_exec_draw_cancel_unrecorded(VkExec *ex){assert(ex->resource_next==67);canceled++;}
RcRegistry *rc_registry_runtime(void){return NULL;}
int rc_registry_provisional_read_current(RcRegistry *r,RcRegistryPosition position){
    assert(!r&&!rc_registry_position_compare(position,read));return provisional_current;
}
RcRegistryResult rc_registry_reopen_unemitted(RcRegistry *r,RcRegistryPosition position){
    assert(!r && !rc_registry_position_compare(position,read));rewinds++;return reopen_result;
}
static void vk_exec_barrier(VkExec *ex){assert(ex==x_cur && ex->resource_next==66);}
VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier2(VkCommandBuffer cb,const VkDependencyInfo *dep){
    assert(cb==x_cur->cb && dep->memoryBarrierCount==1);
    const VkMemoryBarrier2 *mb=dep->pMemoryBarriers;
    assert(mb->srcStageMask==VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT && mb->srcAccessMask==VK_ACCESS_2_MEMORY_WRITE_BIT);
    assert(mb->dstStageMask==VK_PIPELINE_STAGE_2_HOST_BIT && mb->dstAccessMask==VK_ACCESS_2_HOST_READ_BIT);
    barriers++;
}
static VkSyncProgressResult vk_sync_progress_split(VkExec *ex,VkCommandBuffer *tail,int empty){
    assert(!empty && ex==x_cur && ex->resource_next*2+1==read.ordinal-1 && barriers>cuts);
    cuts++;*tail=(VkCommandBuffer)(uintptr_t)9;return cut_result;
}
static void progress_adopt_tail(VkExec *ex,VkCommandBuffer cb){ex->cb=cb;}
static int vk_gpa_write_prefix_complete(const VkGpaPrefixRequest *r){
    assert(cut_result==VK_SYNC_PROGRESS_COMPLETE && cuts && vk_gpa_write_prefix_ready(r));settled++;return 1;
}
static void rt_log(const char *format,...){assert(format);}
'''

cases = r'''
int main(void){
    VkExec ex={.fence=400,.resource_op=67,.resource_next=67,.cb=(VkCommandBuffer)(uintptr_t)7};x_cur=&ex;
    GpaReadPreparation work={.ex=&ex,.effect=&ex};gpa_read_preparation=&work;
    RcRegistryWrite guard={10,1};RcInterval range=captured.extent;
    shared=1;assert(!vk_exec_request_gpa_prefix(&ex,guard,producer,read,range));shared=0;
    Q.out=1;assert(!vk_exec_request_gpa_prefix(&ex,guard,producer,read,range));Q.drained=1;
    recorded_effect=1;assert(!vk_exec_request_gpa_prefix(&ex,guard,producer,read,range));recorded_effect=0;
    work.attempted=read;assert(!vk_exec_request_gpa_prefix(&ex,guard,producer,read,range));
    work.attempted=(RcRegistryPosition){0};
    assert(vk_exec_request_gpa_prefix(&ex,guard,producer,read,range) && !cuts && work.registration==77);
    reopen_result=RC_REGISTRY_BUSY;
    assert(!prepare_ordered_gpa_source(&ex,&work) && !cuts && !settled && recording_failed && ex.resource_next==67);
    recording_failed=0;reopen_result=RC_REGISTRY_OK;
    cut_result=VK_SYNC_PROGRESS_COMPLETE;
    assert(prepare_ordered_gpa_source(&ex,&work) && cuts==1 && settled==1 && ex.resource_next==66);
    assert(ex.cb==(VkCommandBuffer)(uintptr_t)9); // actual begun tail, previous read can be assigned again
    ex.resource_next=67;cut_result=VK_SYNC_PROGRESS_STOPPED_AFTER_CUT;
    assert(!prepare_ordered_gpa_source(&ex,&work) && cuts==2 && settled==1 && !recording_failed);
    ex.resource_next=67;cut_result=VK_SYNC_PROGRESS_COMPLETE;captured.extent.mapping++;
    assert(!prepare_ordered_gpa_source(&ex,&work) && cuts==3 && settled==2 && recording_failed);
    assert(rewinds==4 && canceled==4 && barriers==3);
    ex.resource_next=67;
    recording_failed=0;captured.extent.mapping--;
    GfxShader shader={0};
    VkPipeInfo pipeline_info={0};
    VkPipeline pipeline=(VkPipeline)(uintptr_t)41;
    VkMemSpan indirect={.buffer=(VkBuffer)(uintptr_t)3,.offset=6,.pool=&ex};
    GpaReadPreparation compute={.ex=&ex,.effect=&ex,.compute=1};gpa_read_preparation=&compute;
    assert(!vk_exec_request_gpa_prefix(&ex,guard,producer,read,range)); // no actual compute binder
    assert(vk_exec_gpa_dispatch_binding(&ex,&shader,pipeline,&pipeline_info,0x3000,&indirect,NULL,0,0));
    assert(!vk_exec_request_gpa_prefix(&ex,guard,producer,read,range)); // alternate helper recorded or considered
    assert(vk_exec_gpa_dispatch_binding(&ex,&shader,pipeline,&pipeline_info,0x3000,&indirect,NULL,1,0));
    ex.recorded++; // a binder that emitted native work before the refused input cannot be rewound
    assert(!vk_exec_request_gpa_prefix(&ex,guard,producer,read,range));
    ex.recorded--;
    assert(vk_exec_request_gpa_prefix(&ex,guard,producer,read,range) && prepare_ordered_gpa_source(&ex,&compute));
    assert(ex.resource_next==66 && cuts==4 && settled==3);
    ex.resource_next=67;
    GpaReadPreparation replay={.ex=&ex,.effect=&ex,.compute=1,.attempted=read,.prior_dispatch=&compute.dispatch};
    gpa_read_preparation=&replay;
    assert(vk_exec_gpa_dispatch_retrying(&ex));
    assert(vk_exec_gpa_dispatch_binding(&ex,&shader,pipeline,&pipeline_info,0x3000,&indirect,NULL,1,0));
    assert(replay.binding_checked && !vk_exec_request_gpa_prefix(&ex,guard,producer,read,range)); // same ordinal once
    promoted=argument; // The first binder chose this exact span through compatibility. The mirror is installed now.
    assert(vk_exec_gpa_dispatch_binding(&ex,&shader,pipeline,&pipeline_info,0x3000,&indirect,&promoted,1,0));
    promoted.mapping++;
    assert(!vk_exec_gpa_dispatch_binding(&ex,&shader,pipeline,&pipeline_info,0x3000,&indirect,&promoted,1,0));
    assert(recording_failed);
    recording_failed=0;
    assert(!vk_exec_gpa_dispatch_binding(&ex,&shader,pipeline,&pipeline_info,0x3000,&indirect,NULL,1,1) && recording_failed);
    recording_failed=0;
    argument_mapping++; // A re-created provider can deliberately reuse all native handle values.
    assert(!vk_exec_gpa_dispatch_binding(&ex,&shader,pipeline,&pipeline_info,0x3000,&indirect,NULL,1,0) && recording_failed);
    recording_failed=0;
    assert(!prepare_ordered_gpa_source(&ex,&compute) && cuts==5 && settled==4 && recording_failed);
    gpa_dispatch_release(&compute.dispatch);
    assert(!argument_holds);

    recording_failed=0;ex.resource_next=66;ex.resource_op=67;gpa_read_preparation=NULL;
    RcRegistryPosition attempted={0};
    VkExecGpaSampleScope *scope=vk_exec_gpa_sample_enter(&ex,attempted);
    assert(scope&&!vk_exec_gpa_sample_enter(&ex,attempted));
    assert(!vk_exec_request_gpa_prefix(&ex,guard,producer,read,range));
    provisional_current=1;
    ex.recorded++;
    assert(!vk_exec_request_gpa_prefix(&ex,guard,producer,read,range));
    ex.recorded--;
    assert(vk_exec_request_gpa_prefix(&ex,guard,producer,read,range));
    assert(vk_exec_gpa_sample_leave(scope) && !gpa_read_preparation);
    const int rewinds_before=rewinds, canceled_before=canceled;
    assert(vk_exec_gpa_sample_settle(scope,&attempted)==VK_SYNC_PROGRESS_COMPLETE);
    assert(!rc_registry_position_compare(attempted,read) && ex.resource_next==66 && ex.resource_op==67);
    assert(rewinds==rewinds_before && canceled==canceled_before && cuts==6 && settled==5 && defer_consumed==1);
    scope=vk_exec_gpa_sample_enter(&ex,attempted);
    assert(scope&&!vk_exec_request_gpa_prefix(&ex,guard,producer,read,range));
    assert(!vk_exec_gpa_sample_leave(scope));
    vk_exec_gpa_sample_cancel(scope);
    // Native binding borrows the actual receipt, and only an accepted scope retains a second owner.
    ex.resource_next=67;ex.resource_op=67;recording_failed=0;
    VkMemSpanCapture *native=vk_mem_span_capture(0x3000,sizeof(VkDispatchIndirectCommand),&indirect);
    GpaReadPreparation exact={.ex=&ex,.effect=&ex,.compute=1};gpa_read_preparation=&exact;
    assert(vk_exec_gpa_dispatch_binding(&ex,&shader,pipeline,&pipeline_info,0x3000,&indirect,native,1,1));
    assert(argument_holds==1 && !exact.dispatch.argument);
    assert(vk_exec_request_gpa_prefix(&ex,guard,producer,read,range));
    assert(argument_holds==2 && exact.dispatch.argument==native && exact.dispatch.native_argument);
    vk_mem_span_capture_release(&native);
    assert(argument_holds==1 && prepare_ordered_gpa_source(&ex,&exact));
    ex.resource_next=67;
    GpaReadPreparation exact_retry={.ex=&ex,.effect=&ex,.compute=1,.attempted=read,.prior_dispatch=&exact.dispatch};
    gpa_read_preparation=&exact_retry;
    assert(vk_exec_gpa_dispatch_binding(&ex,&shader,pipeline,&pipeline_info,0x3000,&indirect,exact.dispatch.argument,1,1));
    assert(!vk_exec_gpa_dispatch_binding(&ex,&shader,pipeline,&pipeline_info,0x3000,&indirect,NULL,1,1) && recording_failed);
    recording_failed=0;argument_mapping++;
    assert(!vk_exec_gpa_dispatch_binding(&ex,&shader,pipeline,&pipeline_info,0x3000,&indirect,exact.dispatch.argument,1,1));
    assert(recording_failed);
    recording_failed=0;argument_mapping--;promoted=argument;
    assert(!vk_exec_gpa_dispatch_binding(&ex,&shader,pipeline,&pipeline_info,0x3000,&indirect,&promoted,1,1));
    assert(recording_failed); // An earlier exact promise still requires the same retained receipt.
    gpa_dispatch_release(&exact.dispatch);assert(!argument_holds);
    recording_failed=0;bound_available=1;
    GpaReadPreparation cold={.ex=&ex,.effect=&ex,.compute=1};gpa_read_preparation=&cold;
    assert(vk_exec_gpa_dispatch_binding(&ex,&shader,pipeline,&pipeline_info,0x3000,&indirect,NULL,1,1));
    assert(vk_exec_request_gpa_prefix(&ex,guard,producer,read,range) && cold.dispatch.native_argument);
    assert(bound_captures==1 && cold.dispatch.argument==&argument && argument_holds==1);
    assert(prepare_ordered_gpa_source(&ex,&cold));
    ex.resource_next=67;
    GpaReadPreparation resumed={.ex=&ex,.effect=&ex,.compute=1,.attempted=read,.prior_dispatch=&cold.dispatch};
    gpa_read_preparation=&resumed;
    assert(vk_exec_gpa_dispatch_binding(&ex,&shader,pipeline,&pipeline_info,0x3000,&indirect,cold.dispatch.argument,1,1));
    gpa_dispatch_release(&cold.dispatch);assert(!argument_holds);
    puts("ordered GPA source: actual prefix, same-position indirect retry, helper refusal, stop and mapping reuse passed");
    return 0;
}
'''

cc = Path(os.environ["CC"])
env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="ordered-gpa-") as temporary:
    source = Path(temporary) / "gpa.c"
    source.write_text(prefix + SOURCE[begin:end] + cases)
    exe = source.with_suffix(".exe")
    subprocess.run([str(cc), "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                    "-fsanitize-trap=all", "-I" + str(ROOT / "runtime"), str(source), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

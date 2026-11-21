"""Check real graphics pointer selection, transformed phase limits and shared native copy ownership."""
from pathlib import Path
import os
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
draw = (root / "runtime/vk_exec_draw.c").read_text()
replay = (root / "runtime/vk_exec.c").read_text()
header = (root / "runtime/vk_exec.h").read_text()

def between(source, begin, end, final=False):
    at = source.rindex(begin) if final else source.index(begin)
    return source[at:source.index(end, at)]

mode = between(draw, "typedef enum {\n    DEVICE_POINTER_COMPUTE", "static VkExecGraphicsResult device_shader_pointers_prepare(")
selection = between(draw, "static VkExecGraphicsResult device_shader_pointers_prepare(",
                    "static VkExecGraphicsResult dispatch_device_pointers_prepare(", final=True)
output_selection = between(draw, "// A pointer table supplies an address",
                           "static EffectStorageCapture *effect_storage_move(")
phase = between(draw, "static int mfb_pointer_input_disjoint(", "static const GfxShaderRes *mfb_task_table_binding(") + between(
    draw, "static VkExecGraphicsResult mfb_device_pointers_prepare(", "static VkExecGraphicsResult mfb_dependencies_ready(")
copy_api = between(header, "typedef void (*VkExecNativeInputCopyRecord)", "// A copied queue/chunk state becomes")
copy_impl = between(replay, "VkSyncProgressResult vk_exec_copy_native_inputs(", "VkMemMirrorInputPolicy vk_exec_mirror_input_policy(")
copy_current = between(replay, "static int device_pointer_current(", "static void device_pointer_record_copy(")
def check_replay_order():
    graphics = between(draw, "static VkExecGraphicsResult graphics_draw_select(",
                       "static int prepare_draw_phase(")
    ordered = between(graphics, "result = graphics_pool_prepare(", "result = graphics_load_peers_prepare(")
    assert ordered.index("graphics_pool_prepare(") < ordered.index("graphics_storage_pipeline_ready(")
    assert ordered.index("graphics_storage_pipeline_ready(") < ordered.index("graphics_storage_pointer_outputs_ready(")
    compatibility = between(draw, "static int prepare_draw_phase(VkExec *ex", "static int prepare_draw(")
    assert compatibility.index("DEVICE_POINTER_GRAPHICS_COMPAT") < compatibility.index("graphics_required_device_inputs(")
    assert compatibility.index("graphics_required_device_inputs(") < compatibility.index("graphics_storage_pipeline_ready(")
    compute = between(draw, "VkExecGraphicsResult vk_exec_dispatch_prepare_native_inputs(",
                      "void vk_exec_dispatch_validate(")
    assert compute.index("dispatch_device_pointers_prepare(") < compute.index("selected_storage_writers_prepare(")
    assert compute.index("selected_storage_writers_prepare(") < compute.index("device_storage_pointer_outputs_prepare(")


SELECTION_FIXTURE = r'''#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

enum { GFX_RES_SSBO=1, NVN_STAGE_VS=0, NVN_STAGE_FS=1, NVN_STAGE_CS=5, NVN_STAGE_TASK=6, NVN_STAGE_MESH=7, MAX_BINDINGS=8 };
typedef enum { VK_EXEC_GRAPHICS_READY, VK_EXEC_GRAPHICS_NEEDS_ORDERED, VK_EXEC_GRAPHICS_UNAVAILABLE,
               VK_EXEC_GRAPHICS_INVALIDATED,VK_EXEC_GRAPHICS_STOPPED } VkExecGraphicsResult;
typedef enum { VK_SYNC_PROGRESS_UNAVAILABLE,VK_SYNC_PROGRESS_COMPLETE,VK_SYNC_PROGRESS_STOPPED,
               VK_SYNC_PROGRESS_STOPPED_AFTER_CUT } VkSyncProgressResult;
typedef enum { RC_REGISTRY_OK,RC_REGISTRY_STALE } RcRegistryResult;
typedef struct { uint64_t addr,size; } p_bind_buf;
typedef struct { uint32_t ptr,count,flags,guest_slot; } GfxShaderRes;
typedef struct { const GfxShaderRes *resources; int count, writes, complete; } GfxShader;
typedef struct { p_bind_buf ubo[8][16],ssbo[8][16]; unsigned cb; } VkExec;
typedef struct { uint64_t address,avail; } VkMemSpan;
typedef struct { int alive; } VkMemSpanCapture;
typedef struct { uint64_t mapping; } RcInterval;
typedef struct { uint64_t scene,ordinal; } RcRegistryPosition;
typedef struct {
    VkMemSpan span;
    VkMemSpanCapture *native;
    RcInterval *mapped;
    uint32_t mapped_count;
} EffectStorageCapture;
typedef struct {
    const GfxShaderRes *resource;
    VkMemSpan span;
    VkMemSpanCapture *capture;
    RcInterval *ranges;
    RcRegistryPosition read;
    uint64_t value;
    uint32_t count;
    int stage;
    p_bind_buf binding;
    int prepared,task_input;
    EffectStorageCapture output;
} DeviceStoragePointer;
static int readonly_storage_pointer(const GfxShaderRes *resource) {
    return resource->ptr && resource->count<=1 && !(resource->flags&1);
}
static unsigned sources,targets,copies,releases;
static int lost,output_supported=1,remap;
static const uint64_t guest_gpu_window=0x8000000000;
static int gfx_shader_resources(const GfxShader *s,int kind,const GfxShaderRes **out) {
    assert(kind==GFX_RES_SSBO);
    *out=s->resources;
    return s->count;
}
static int ssbo_slot(const GfxShaderRes *r,int stage,int *ubo) {
    assert(stage>=0&&stage<8);
    *ubo=0;
    return (int)r->guest_slot;
}
static int nvn_gpu_only(uint64_t address){return address>=guest_gpu_window;}
static const DeviceStoragePointer *device_storage_pointer(const DeviceStoragePointer *p,uint32_t n,int stage,const GfxShaderRes *r) {
    for (uint32_t i=0;i<n;i++) if(p[i].stage==stage&&p[i].resource==r) return &p[i];
    return NULL;
}
static int device_shader_pointers_current(const DeviceStoragePointer *pointer,uint32_t count){
    assert(pointer&&count==1&&pointer->capture&&pointer->capture->alive&&pointer->read.ordinal==8);
    return !remap;
}
static RcRegistryResult vk_mem_span_native_input(uint64_t address,uint64_t bytes,int clamp,VkMemSpan *s,VkMemSpanCapture **out) {
    assert(bytes==8&&!clamp&&!*out&&nvn_gpu_only(address));
    *s=(VkMemSpan){address,8};
    *out=malloc(sizeof **out);assert(*out);
    **out=(VkMemSpanCapture){1};sources++;
    return RC_REGISTRY_OK;
}
static void vk_mem_span_capture_release(VkMemSpanCapture **out){if(*out) {free(*out);*out=NULL;releases++;}}
static uint32_t vk_mem_span_input_intervals(const VkMemSpanCapture *c,uint64_t address,uint64_t bytes,RcInterval **out){
    assert(c&&c->alive&&nvn_gpu_only(address)&&bytes==8);
    *out=malloc(sizeof **out);assert(*out);**out=(RcInterval){address};
    return 1;
}
static VkSyncProgressResult vk_exec_read_device_pointer(VkExec *ex,uint64_t address,const VkMemSpan *span,VkMemSpanCapture *c,
    const RcInterval *ranges,uint32_t count,RcRegistryPosition read,uint64_t *out) {
    assert(ex&&span->address==address&&ranges->mapping==address&&count==1&&c->alive&&read.scene==3&&read.ordinal==8);
    copies++;
    *out=address+0x4000;
    return lost?VK_SYNC_PROGRESS_STOPPED_AFTER_CUT:VK_SYNC_PROGRESS_COMPLETE;
}
static int vk_exec_mirror_input_policy(const VkExec *ex,int ordered){assert(ex&&ordered==1);return 2;}
static int effect_storage_capture_span(const GfxShaderRes *r,uint64_t address,uint64_t bytes,int policy,EffectStorageCapture *out) {
    assert(r&&address>=0x4000&&!bytes&&policy==2);
    if (!output_supported) return 0;
    out->native=malloc(sizeof *out->native);assert(out->native);
    out->span=(VkMemSpan){address,32};*out->native=(VkMemSpanCapture){1};
    targets++;return 1;
}

__DEVICE_POINTER_SELECT__
enum { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER=7 };
typedef struct { int kind,type,count,stage; const GfxShaderRes *res; } VkPipeBinding;
typedef struct { unsigned n_bindings; const VkPipeBinding *bindings; } VkPipeInfo;
typedef struct { int selected; } SelectedStorageWriters;
static int nvn_stage_of(int stage){return stage;}
static int device_storage_pointer_value_current(const VkExec *ex,const DeviceStoragePointer *pointer) {
    const p_bind_buf *table=&ex->ssbo[pointer->stage][pointer->resource->guest_slot];
    return pointer->prepared&&table->addr==pointer->binding.addr&&table->size==pointer->binding.size&&
        device_shader_pointers_current(pointer,1);
}
static int vk_mem_span_capture_current(const VkMemSpanCapture *native,const VkMemSpan *span){
    assert(native&&span->address>=0x4000);return native->alive;
}
static int selected_storage_writer(SelectedStorageWriters *writers,int stage,const GfxShaderRes *resource){
    assert(stage>=0&&stage<8&&resource->flags==1);return writers&&writers->selected;
}
__DEVICE_POINTER_OUTPUTS__
typedef struct {const GfxShader *shader;const VkPipeInfo *info;} MfbPhaseUse;
typedef struct { MfbPhaseUse task,mesh; const GfxShader *ts,*ms; unsigned selected_cb; } MfbDraw;
typedef struct {uint64_t identity;int current;} VkTextureWriteTarget;
typedef struct {unsigned pin;} VkPinnedTextureWrite;
typedef struct {int written;VkTextureWriteTarget *write_target;VkPinnedTextureWrite pinned;} EffectImageCapture;
typedef struct {unsigned first,end;} StoragePhaseUse;
typedef struct {
    uint32_t count;StoragePhaseUse phases[2];
    struct {EffectImageCapture *images;} bindings;
} VkExecStoragePhase;
static int vk_texture_write_disjoint_input(const VkTextureWriteTarget *target,const VkPinnedTextureWrite *pin,
    const RcInterval *input,uint32_t count) {
    if (!target||pin||!target->current||!input||!count) return 0;
    for (uint32_t i=0;i<count;i++) if (input[i].mapping==target->identity) return 0;
    return 1;
}
typedef struct {
    MfbDraw *fallback;
    VkExecStoragePhase *storage;
    const GfxShader *pointer_shaders[8];
    DeviceStoragePointer *pointer_sources;
    uint32_t pointer_count;
    RcRegistryPosition read_position;
    struct {const GfxShader *fs;} state;
} VkExecGraphicsDraw;
static void release(DeviceStoragePointer **p,uint32_t *count){
    for (uint32_t i=0;i<*count;i++){
        vk_mem_span_capture_release(&(*p)[i].capture);
        vk_mem_span_capture_release(&(*p)[i].output.native);
        free((*p)[i].ranges);
    }
    free(*p);*p=NULL;*count=0;
}
static void graphics_select_pointer_shaders(VkExecGraphicsDraw *draw,const GfxShader *const selected[8]){
    for(int i=0;i<8;i++) draw->pointer_shaders[i]=selected[i];
}
static VkExecGraphicsResult graphics_required_device_inputs(VkExec *ex,VkExecGraphicsDraw *draw){
    assert(ex&&draw->pointer_count==2&&draw->pointer_sources[0].stage==NVN_STAGE_TASK&&
           draw->pointer_sources[1].stage==NVN_STAGE_MESH&&!draw->pointer_sources[0].prepared&&
           !draw->pointer_sources[1].prepared);
    static RcInterval selected_target[2]={{0x4400},{0x4800}};
    for (unsigned i=0;i<2;i++) {
        draw->pointer_sources[i].value=selected_target[i].mapping;
        draw->pointer_sources[i].output.mapped=&selected_target[i];
        draw->pointer_sources[i].output.mapped_count=1;
        draw->pointer_sources[i].prepared=1;
    }
    ex->cb=48;return VK_EXEC_GRAPHICS_READY;
}
static int gfx_shader_guest_writes(const GfxShader *s){return s->writes;}
static int mfb_selected_compute_writes_known(const MfbPhaseUse *phase,const EffectStorageCapture *external){
    return !external&&phase->shader->complete;
}
static int mfb_phase_guest_writes_known(const GfxShader *shader,const VkPipeInfo *info,
    const VkExecStoragePhase *storage,uint32_t phase){
    (void)info;return shader->complete&&(!shader->writes||(storage&&phase<storage->count));
}
static unsigned phase_refusals;
static VkExecGraphicsResult mfb_storage_phase_ready(VkExec *ex,VkExecGraphicsDraw *draw,int early_final){
    assert(ex&&draw&&!early_final);return VK_EXEC_GRAPHICS_READY;
}
static VkExecGraphicsResult mfb_task_pointer_prepare(VkExec *ex,VkExecGraphicsDraw *draw,int early_final){
    assert(ex&&draw&&draw->pointer_count==2&&!early_final);
    phase_refusals++;return VK_EXEC_GRAPHICS_UNAVAILABLE;
}
__MFB_POINTER_PHASE__
int main(void) {
    VkExec ex={0};
    const GfxShaderRes src={1,1,0,3}, sink={9,1,1,3};
    GfxShader task={&src,1,0,1},mesh={&src,1,0,1},output={&sink,1,1,1};
    RcRegistryPosition read={3,8};
    DeviceStoragePointer *inputs=NULL;
    uint32_t count=0;
    ex.ssbo[NVN_STAGE_VS][3]=(p_bind_buf){guest_gpu_window+0x100,16};
    ex.ssbo[NVN_STAGE_FS][3]=(p_bind_buf){guest_gpu_window+0x200,24};
    ex.ssbo[NVN_STAGE_CS][3]=ex.ssbo[NVN_STAGE_FS][3];
    ex.ssbo[NVN_STAGE_TASK][3]=(p_bind_buf){guest_gpu_window+0x300,16};
    ex.ssbo[NVN_STAGE_MESH][3]=(p_bind_buf){guest_gpu_window+0x400,16};
    assert(device_shader_pointers_prepare(&ex,&task,NVN_STAGE_VS,read,&inputs,&count,0,DEVICE_POINTER_GRAPHICS)==VK_EXEC_GRAPHICS_NEEDS_ORDERED);
    assert(!inputs&&!count&&!sources&&!copies);
    assert(device_shader_pointers_prepare(&ex,&task,NVN_STAGE_VS,read,&inputs,&count,1,DEVICE_POINTER_GRAPHICS)==VK_EXEC_GRAPHICS_READY);
    assert(count==1&&sources==1&&!copies&&!inputs[0].prepared);
    assert(device_shader_pointers_prepare(&ex,&task,NVN_STAGE_VS,read,&inputs,&count,1,DEVICE_POINTER_GRAPHICS)==VK_EXEC_GRAPHICS_READY);
    ex.ssbo[NVN_STAGE_VS][3].addr+=16;
    assert(device_shader_pointers_prepare(&ex,&task,NVN_STAGE_VS,read,&inputs,&count,1,DEVICE_POINTER_GRAPHICS)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!copies);release(&inputs,&count);

    MfbDraw phases={.task={.shader=&task},.mesh={.shader=&mesh},.ts=&task,.ms=&mesh};
    VkExecGraphicsDraw draw={.fallback=&phases,.read_position=read};
    assert(mfb_device_pointers_prepare(&ex,&draw,0)==VK_EXEC_GRAPHICS_READY&&phases.selected_cb==48);
    release(&draw.pointer_sources,&draw.pointer_count);
    task.writes=1;phases.selected_cb=10;
    assert(mfb_device_pointers_prepare(&ex,&draw,0)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(draw.pointer_count==2 && phases.selected_cb==10 && !copies && phase_refusals==1);
    release(&draw.pointer_sources,&draw.pointer_count);
    VkTextureWriteTarget produced={.identity=0x9999,.current=1};
    EffectImageCapture captured={.written=1,.write_target=&produced};
    VkExecStoragePhase proof={.count=1,.phases={{0,1}},.bindings={&captured}};
    draw.storage=&proof;
    assert(mfb_device_pointers_prepare(&ex,&draw,0)==VK_EXEC_GRAPHICS_READY&&phases.selected_cb==48);
    release(&draw.pointer_sources,&draw.pointer_count);
    produced.identity=ex.ssbo[NVN_STAGE_MESH][3].addr;
    assert(mfb_device_pointers_prepare(&ex,&draw,0)==VK_EXEC_GRAPHICS_UNAVAILABLE&&phase_refusals==2);
    release(&draw.pointer_sources,&draw.pointer_count);
    produced.identity=0x4800;
    assert(mfb_device_pointers_prepare(&ex,&draw,0)==VK_EXEC_GRAPHICS_UNAVAILABLE&&phase_refusals==2);
    release(&draw.pointer_sources,&draw.pointer_count);
    produced.identity=0x9999;produced.current=0;
    assert(mfb_device_pointers_prepare(&ex,&draw,0)==VK_EXEC_GRAPHICS_UNAVAILABLE&&phase_refusals==3);
    release(&draw.pointer_sources,&draw.pointer_count);
    draw.storage=NULL;
    task.writes=0;task.complete=0;
    assert(mfb_device_pointers_prepare(&ex,&draw,0)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(draw.pointer_count==2&&!copies&&phase_refusals==4);
    release(&draw.pointer_sources,&draw.pointer_count);
    task.complete=1;phases.task.shader=&output;
    assert(mfb_device_pointers_prepare(&ex,&draw,0)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!draw.pointer_count&&!copies);
    phases.task.shader=&task;
    lost=1;
    assert(device_shader_pointers_prepare(&ex,&output,NVN_STAGE_CS,read,&inputs,&count,1,DEVICE_POINTER_COMPUTE)==VK_EXEC_GRAPHICS_STOPPED);
    assert(!count&&copies==1);release(&inputs,&count);
    lost=0;
    assert(device_shader_pointers_prepare(&ex,&output,NVN_STAGE_CS,read,&inputs,&count,1,DEVICE_POINTER_COMPUTE)==VK_EXEC_GRAPHICS_READY);
    assert(count==1&&inputs[0].prepared&&!inputs[0].output.native&&!targets&&copies==2);
    VkPipeBinding binding={GFX_RES_SSBO,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,NVN_STAGE_CS,&sink};
    VkPipeInfo info={1,&binding};
    SelectedStorageWriters selected={1};
    assert(device_storage_pointer_outputs_prepare(&ex,&info,inputs,count,&selected)==VK_EXEC_GRAPHICS_READY);
    assert(!inputs[0].output.native&&!targets);
    output_supported=0;
    assert(device_storage_pointer_outputs_prepare(&ex,&info,inputs,count,NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE&&!targets);
    output_supported=1;
    assert(device_storage_pointer_outputs_prepare(&ex,&info,inputs,count,NULL)==VK_EXEC_GRAPHICS_READY);
    assert(inputs[0].output.native&&inputs[0].output.span.address==guest_gpu_window+0x4208&&targets==1);
    assert(device_storage_pointer_outputs_prepare(&ex,&info,inputs,count,NULL)==VK_EXEC_GRAPHICS_READY&&targets==1);
    inputs[0].output.native->alive=0;
    assert(device_storage_pointer_outputs_prepare(&ex,&info,inputs,count,NULL)==VK_EXEC_GRAPHICS_INVALIDATED&&targets==1);
    release(&inputs,&count);

    assert(device_shader_pointers_prepare(&ex,&task,NVN_STAGE_FS,read,&inputs,&count,1,DEVICE_POINTER_GRAPHICS_COMPAT)==VK_EXEC_GRAPHICS_READY);
    assert(count==1 && !inputs[0].prepared && inputs[0].resource==&src);
    release(&inputs,&count);
    assert(device_shader_pointers_prepare(&ex,&output,NVN_STAGE_FS,read,&inputs,&count,1,DEVICE_POINTER_GRAPHICS_COMPAT)==VK_EXEC_GRAPHICS_READY);
    assert(count==1&&!inputs[0].prepared&&!inputs[0].output.native&&copies==2);
    binding.stage=NVN_STAGE_FS;
    assert(device_storage_pointer_outputs_prepare(&ex,&info,inputs,count,NULL)==VK_EXEC_GRAPHICS_INVALIDATED&&targets==1);
    inputs[0].prepared=1;inputs[0].value=guest_gpu_window+0x4248;
    ex.ssbo[NVN_STAGE_FS][3].size--;
    assert(device_storage_pointer_outputs_prepare(&ex,&info,inputs,count,NULL)==VK_EXEC_GRAPHICS_INVALIDATED&&targets==1);
    ex.ssbo[NVN_STAGE_FS][3].size++;
    assert(device_storage_pointer_outputs_prepare(&ex,&info,inputs,count,NULL)==VK_EXEC_GRAPHICS_READY);
    assert(inputs[0].output.native&&inputs[0].output.span.address==guest_gpu_window+0x4248&&targets==2);
    release(&inputs,&count);
    assert(releases==sources+targets);
    puts("graphics/MFB pointer preparation: selected stages, before-effects batch, disjoint exact outputs, overlap and unknown refusal, native stop and ordinary compute passed");
}
'''

MIXED_FIXTURE = r'''#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <vulkan/vulkan.h>
typedef struct { uint64_t scene,ordinal; } RcRegistryPosition;
typedef struct { uint64_t backing,offset,size; } RcSourceRange;
typedef struct { uint64_t address,mapping;RcSourceRange range; } RcInterval;
typedef struct { VkBuffer buffer; VkDeviceSize offset,avail; } VkMemSpan;
typedef struct { int ref; VkBuffer buffer; } VkMemSpanCapture;
typedef struct { VkCommandBuffer cb; uint64_t fence,recorded; } VkExec;
typedef enum { VK_SYNC_PROGRESS_UNAVAILABLE,VK_SYNC_PROGRESS_COMPLETE,VK_SYNC_PROGRESS_STOPPED,
               VK_SYNC_PROGRESS_STOPPED_AFTER_CUT } VkSyncProgressResult;
typedef enum { RC_REGISTRY_OK, RC_REGISTRY_STALE } RcRegistryResult;
__NATIVE_COPY_API__
#define PW_OP 1
enum { SNAPSHOT_BYTES=0x580020, COMPARE_BYTES=98368, DIRECT_BYTES=32 };
enum { COMPLETE, REMAP_AFTER_PREFIX, STOP_AFTER_PREFIX, STOP_UNHANDED, STOP_DURING_DRAIN };

static VkExec execution={.cb=(VkCommandBuffer)1,.fence=42}, *x_cur=&execution;
static RcRegistryPosition read={42,6326}, current={42,6326};
static RcInterval mapped[]={{0x10000,1,{8,0,SNAPSHOT_BYTES}},{0x90000,2,{10,0,DIRECT_BYTES}}};
static VkMemSpanCapture a={1,(VkBuffer)1}, b={1,(VkBuffer)2}, *part_providers[2];
static int rw_chunk, stop, mode, ranged;
static unsigned copies, barriers, adoptions, provider_syncs, drains, prefix, owner_record, releases, part_count;
static uint8_t readback_data[COMPARE_BYTES+16+1];
static int vk_sync_stopping(void){return stop;}
static int vk_device_is_lost(void){return 0;}
static int vk_exec_recording_failed(void){return stop;}
static void *rc_registry_runtime(void){return &execution;}
static int rc_registry_position_compare(RcRegistryPosition a,RcRegistryPosition b){
    return a.scene!=b.scene || a.ordinal!=b.ordinal;
}
static RcRegistryPosition vk_exec_resource_position(void){return current;}
static int rc_registry_provisional_read_current(void *r,RcRegistryPosition p){
    return r==&execution && !rc_registry_position_compare(read,p);
}
static RcRegistryResult rc_registry_read_device_buffer_at(void *r,const RcInterval *m,uint32_t n,RcRegistryPosition p){
    assert(r==&execution && n==1 && !rc_registry_position_compare(read,p));
    return m->mapping==(m->address==0x10000 ? 1u : 2u) ? RC_REGISTRY_OK : RC_REGISTRY_STALE;
}
static int vk_mem_span_capture_current(const VkMemSpanCapture *h,const VkMemSpan *s){
    return h->ref && h->buffer==s->buffer;
}
__NATIVE_COPY_CURRENT__
static void group_drain(VkExec *ex,int why){
    assert(ex==&execution && why==PW_OP);
    drains++;
    if (mode==STOP_DURING_DRAIN) stop=1;
}
static void vk_exec_pass_end(VkExec *ex){assert(ex==&execution);}
static int vk_mem_span_record_input(VkMemSpanCapture *h,const VkMemSpan *s){
    assert(vk_mem_span_capture_current(h,s)); provider_syncs++; return 1;
}
static VkMemSpanCapture *vk_mem_span_capture_retain(VkMemSpanCapture *h){h->ref++;return h;}
static void vk_exec_record_native_input_provider(VkExec *ex,VkMemSpanCapture **h){
    assert(ex==&execution && *h && (*h)->ref==2 && part_count<2);
    part_providers[part_count++]=*h;
    *h=NULL;
    adoptions++;
}
static void settle_part(void) {
    while (part_count) {
        VkMemSpanCapture *h=part_providers[--part_count];
        assert(h->ref==2);
        h->ref--;
        releases++;
    }
}
static void record(uint64_t fence,void *context){
    assert(fence==42 && context==&execution && adoptions==2 && provider_syncs==2 && !copies);
    assert(a.ref==2 && b.ref==2 && part_count==2);
    owner_record++;
}
static void vk_exec_barrier(VkExec *ex){assert(ex==&execution && owner_record==1 && part_count==2);}
static VkSyncProgressResult vk_exec_graphics_pool_progress(VkExec *ex,uint64_t prior){
    assert(ex==&execution && !prior && copies==3 && barriers==2 && part_count==2);
    if (mode==STOP_UNHANDED) return VK_SYNC_PROGRESS_STOPPED;
    prefix++;
    if (mode==STOP_AFTER_PREFIX) return VK_SYNC_PROGRESS_STOPPED_AFTER_CUT;
    if (mode==REMAP_AFTER_PREFIX) mapped[1].mapping=3;
    return VK_SYNC_PROGRESS_COMPLETE;
}
static uint8_t native_byte(VkBuffer source,VkDeviceSize offset){
    return (uint8_t)((uintptr_t)source*19 + offset*3 + (offset>>8)*7);
}
static void put_fixture_bytes(VkBuffer source,VkDeviceSize base,const VkBufferCopy *copy) {
    assert(copy->dstOffset+copy->size < sizeof readback_data);
    for (VkDeviceSize i=0;i<copy->size;i++)
        readback_data[copy->dstOffset+i]=native_byte(source,base+copy->srcOffset+i);
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer cb,VkBuffer src,VkBuffer dst,uint32_t n,const VkBufferCopy *r) {
    assert(cb==execution.cb && part_count==2);
    switch(copies++){
    case 0: assert(src==(VkBuffer)1 && dst==(VkBuffer)3 && n==1 &&
                   r[0].srcOffset==32 && r[0].dstOffset==0 && r[0].size==SNAPSHOT_BYTES); break;
    case 1:
        assert(src==(VkBuffer)3 && dst==(VkBuffer)5 && r[0].dstOffset==0);
        if (ranged) {
            assert(n==1 && r[0].srcOffset==8 && r[0].size==COMPARE_BYTES);
            put_fixture_bytes((VkBuffer)1,32,r);
        } else
            assert(n==2 && r[0].srcOffset==0x2c0270 && r[0].size==4 &&
                   r[1].srcOffset==0x2c0278 && r[1].dstOffset==4 && r[1].size==4);
        break;
    case 2:
        assert(src==(VkBuffer)2 && dst==(VkBuffer)5);
        if (ranged) {
            assert(n==1 && r[0].srcOffset==20 && r[0].dstOffset==COMPARE_BYTES && r[0].size==16);
            put_fixture_bytes(src,0,r);
        } else
            assert(n==2 && r[0].srcOffset==16 && r[0].dstOffset==8 && r[0].size==4 &&
                   r[1].srcOffset==20 && r[1].dstOffset==12 && r[1].size==4);
        break;
    default: assert(0);
    }
}
VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier2(VkCommandBuffer cb,const VkDependencyInfo *d) {
    assert(cb==execution.cb && d->bufferMemoryBarrierCount==1);
    const VkBufferMemoryBarrier2 *barrier=d->pBufferMemoryBarriers;
    if (!barriers++)
        assert(copies==1 && barrier->buffer==(VkBuffer)3 && barrier->size==SNAPSHOT_BYTES &&
               barrier->dstAccessMask==(VK_ACCESS_2_TRANSFER_READ_BIT|VK_ACCESS_2_SHADER_STORAGE_READ_BIT));
    else
        assert(copies==3 && barrier->buffer==(VkBuffer)5 && !barrier->offset &&
               barrier->size==(ranged ? COMPARE_BYTES+16 : 16) && barrier->dstAccessMask==VK_ACCESS_2_HOST_READ_BIT);
}
__NATIVE_COPY_IMPL__

static void expect_refused(VkExecNativeInputCopy *plan,VkSyncProgressResult result) {
    assert(vk_exec_copy_native_inputs(&execution,read,plan,2,(VkBuffer)5,record,&execution)==result);
    assert(!copies && !adoptions && !provider_syncs && !part_count && !releases && !owner_record);
}
int main(void) {
    VkMemSpan first={(VkBuffer)1,32,SNAPSHOT_BYTES}, second={(VkBuffer)2,16,DIRECT_BYTES};
    VkBufferCopy ab[]={{0x2c0270,0,4},{0x2c0278,4,4}}, cd[]={{0,8,4},{4,12,4}};
    VkExecNativeInputCopy plan[]={{&first,&a,mapped,1,(VkBuffer)3,first.avail,ab,2},
                                 {&second,&b,mapped+1,1,VK_NULL_HANDLE,8,cd,2}};
    mapped[1].mapping=3;
    expect_refused(plan,VK_SYNC_PROGRESS_UNAVAILABLE);
    mapped[1].mapping=2;
    second.buffer=(VkBuffer)8;
    expect_refused(plan,VK_SYNC_PROGRESS_UNAVAILABLE);
    second.buffer=(VkBuffer)2;
    current.ordinal-=2;
    expect_refused(plan,VK_SYNC_PROGRESS_UNAVAILABLE);
    current=read;
    cd[0].dstOffset=4;
    expect_refused(plan,VK_SYNC_PROGRESS_UNAVAILABLE);
    cd[0].dstOffset=8;
    for (int test=COMPLETE;test<=STOP_UNHANDED;test++) {
        mode=test;
        const VkSyncProgressResult expected[]={VK_SYNC_PROGRESS_COMPLETE,VK_SYNC_PROGRESS_UNAVAILABLE,
                                              VK_SYNC_PROGRESS_STOPPED_AFTER_CUT,VK_SYNC_PROGRESS_STOPPED};
        prefix=mode==STOP_UNHANDED ? 1 : 0; // An unhanded tail follows an older handed prefix.
        assert(vk_exec_copy_native_inputs(&execution,read,plan,2,(VkBuffer)5,record,&execution)==expected[test]);
        assert(copies==3 && barriers==2 && prefix==1 && part_count==2 && a.ref==2 && b.ref==2 && owner_record==1);
        settle_part();
        settle_part();
        assert(releases==2 && a.ref==1 && b.ref==1 && !part_count);
        copies=barriers=prefix=owner_record=provider_syncs=adoptions=releases=0;
        mapped[1].mapping=2;
    }
    mode=STOP_DURING_DRAIN;
    expect_refused(plan,VK_SYNC_PROGRESS_STOPPED);
    mode=COMPLETE;
    stop=0;

    ranged=1;
    plan[0].word_count=plan[1].word_count=1;
    plan[1].bytes=DIRECT_BYTES;
    ab[0]=(VkBufferCopy){8,0,COMPARE_BYTES};
    cd[0]=(VkBufferCopy){4,COMPARE_BYTES,16};
    const VkBufferCopy invalid[]={{8,0,0},{8,0,COMPARE_BYTES-1},{6,0,COMPARE_BYTES},
                                 {SNAPSHOT_BYTES,0,4},{SNAPSHOT_BYTES-4,0,8},{UINT64_MAX-3,0,8},
                                 {8,4,COMPARE_BYTES}};
    unsigned prior_drains=drains;
    for (unsigned i=0;i<sizeof invalid/sizeof *invalid;i++) {
        ab[0]=invalid[i];
        expect_refused(plan,VK_SYNC_PROGRESS_UNAVAILABLE);
    }
    ab[0]=(VkBufferCopy){8,0,COMPARE_BYTES};
    first.offset=UINT64_MAX-3;
    expect_refused(plan,VK_SYNC_PROGRESS_UNAVAILABLE);
    first.offset=32;
    plan[0].snapshot=(VkBuffer)1;
    expect_refused(plan,VK_SYNC_PROGRESS_UNAVAILABLE);
    plan[0].snapshot=(VkBuffer)5;
    expect_refused(plan,VK_SYNC_PROGRESS_UNAVAILABLE);
    plan[0].snapshot=(VkBuffer)3;
    plan[0].bytes=SNAPSHOT_BYTES+4;
    expect_refused(plan,VK_SYNC_PROGRESS_UNAVAILABLE);
    plan[0].bytes=SNAPSHOT_BYTES;
    VkMemSpan large={(VkBuffer)1,0,UINT64_MAX-3};
    VkBufferCopy huge={0,0,UINT64_MAX-3}, extra={0,UINT64_MAX-3,4};
    VkExecNativeInputCopy wrapping[]={{&large,&a,mapped,1,(VkBuffer)3,large.avail,&huge,1},
                                     {&second,&b,mapped+1,1,VK_NULL_HANDLE,8,&extra,1}};
    expect_refused(wrapping,VK_SYNC_PROGRESS_UNAVAILABLE);
    assert(drains==prior_drains);
    readback_data[COMPARE_BYTES+16]=0xc7;
    assert(vk_exec_copy_native_inputs(&execution,read,plan,2,(VkBuffer)5,record,&execution)==VK_SYNC_PROGRESS_COMPLETE);
    for (unsigned i=0;i<COMPARE_BYTES;i++) assert(readback_data[i]==native_byte((VkBuffer)1,40+i));
    for (unsigned i=0;i<16;i++) assert(readback_data[COMPARE_BYTES+i]==native_byte((VkBuffer)2,20+i));
    assert(readback_data[COMPARE_BYTES+16]==0xc7);
    settle_part();
    assert(releases==2 && a.ref==1 && b.ref==1);
    puts("ordered native input copy: scalar and 98,368-byte snapshot/direct subranges; invalid source, alignment, overflow and stale reads; handed and unhanded provider custody");
}
'''

def main():
    check_replay_order()
    cc = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="graphics-pointers-") as tmp:
        tests = {
            "selection": SELECTION_FIXTURE.replace("__DEVICE_POINTER_SELECT__", mode + selection)
                .replace("__DEVICE_POINTER_OUTPUTS__", output_selection)
                .replace("__MFB_POINTER_PHASE__", phase),
            "mixed": MIXED_FIXTURE.replace("__NATIVE_COPY_API__", copy_api)
                .replace("__NATIVE_COPY_CURRENT__", copy_current).replace("__NATIVE_COPY_IMPL__", copy_impl),
        }
        for name, text in tests.items():
            path = Path(tmp) / (name + ".c")
            path.write_text(text)
            exe = path.with_suffix(".exe")
            subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                            "-fsanitize-trap=all", str(path), "-o", str(exe)], env=env, check=True)
            subprocess.run([str(exe)], env=env, check=True)
if __name__ == "__main__":
    main()

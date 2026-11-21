"""Exercise the native argument receipt against actual backing-map incarnations."""
from pathlib import Path
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
memory_path = Path(os.getenv("INDIRECT_SPAN_MEMORY_SOURCE", ROOT / "runtime/vk_memory.c"))
source = memory_path.read_text()
qualified = memory_path.with_name("resource_host_write.h").read_text()
qualification_types = qualified[qualified.index("typedef enum VkMemCanonicalSourceQualificationStage {"):
                                qualified.index("// Qualifies an ordinary tracked import")]
device = source.index("static int pool_device_span(")
device_end = source.index("\n}", device) + 2
mirror = source.index("static int span_record_mirror_input(",source.index("static void mem_use_locked("))
mirror_end = source.index("\n}", mirror) + 2
pending_scene = source.index("static uint64_t mirror_scene(void) {")
import_scene = source.index("static uint64_t mirror_import_scene(const PoolRec *pool) {", pending_scene)
pending_scene_end = source.index("\n}", import_scene) + 2
prepare_mirror = source.index("static int mirror_prepare_input(", source.index("static void mem_use_locked("))
prepare_mirror_end = source.index("\n}", prepare_mirror) + 2
sync = (ROOT / "runtime/vk_sync.c").read_text()
publish_safe = sync.index("int vk_sync_mirror_publication_safe(")
publish_safe_end = sync.index("\n}", publish_safe) + 2
exec_source = (ROOT / "runtime/vk_exec.c").read_text()
mirror_policy = exec_source.index("VkMemMirrorInputPolicy vk_exec_mirror_input_policy(")
mirror_policy_end = exec_source.index("\n}", mirror_policy) + 2
begin = source.index("struct VkMemSpanCapture {")
snapshot = source.index("// pool_lock held. Indexed containment is the logical pool range", begin)
capture_helpers = source[begin:snapshot]
for declaration in (
    "static int mirror_pool_entry(const PoolRec *);",
    "static int mirror_pool_alone(int);",
    "static int mirror_transition_ready_locked(const PoolRec *, RcRegistryPosition);",
    "static void mirror_retire_completed(void *);",
    "static void mirror_retire_canceled(void *, int may_have_executed);",
):
    assert capture_helpers.count(declaration) == 1
    capture_helpers = capture_helpers.replace(declaration, "")
import_release = source.index("static void import_writer_release(", snapshot)
import_release_end = source.index("\n}", import_release) + 2
bound_input = source.index("RcRegistryResult vk_mem_span_capture_bound_input(", import_release_end)
end = source.index("static PoolRec *pool_image_alias_provider_locked(", bound_input)
vectors = source.index("static int host_interval_vectors_match(", end)
vectors_end = source.index("\n}", vectors) + 2
expected = source.index("RcRegistryResult vk_mem_canonical_source_qualify_expected(", vectors_end)
expected_end = source.index("\n}", expected) + 2
protected = source.index("typedef struct {\n    const VkMemSpanCapture *span;")
protected_end = source.index("int vk_mem_span_buffer_location(", protected)
generation = source.index("static uint32_t generation_next(void)")
generation_end = source.index("\n}", generation) + 2
stamp = source.index("enum StampResult {")
stamp_end = source.index("// Collect writes pending", stamp)
input_mapping = source.index("uint32_t vk_mem_span_input_intervals(", protected_end)
physical_origin = source.index("static int physical_initial_fill_covers(", protected_end)
physical_origin_end = source.index("\n}", physical_origin) + 2
input_mapping_end = source.index("static uint32_t copy_pool_host_intervals(", input_mapping)
host_write = (ROOT / "runtime/resource_host_write.c").read_text()
slice_start = host_write.index("uint32_t resource_host_interval_slice(")
slice_end = host_write.index("\n}", slice_start) + 2
draw = Path(os.getenv("INDIRECT_SPAN_DRAW_SOURCE", ROOT / "runtime/vk_exec_draw.c")).read_text()
argument_release = draw.index("static void graphics_argument_release(")
argument_record_end = draw.index("void vk_exec_record_native_input_provider(", argument_release)
argument_use = draw.index("static int graphics_argument_use(", argument_record_end)
argument_use_end = draw.index("\n}", argument_use) + 2
argument_select = draw.index("static VkExecGraphicsResult graphics_argument_select(")
argument_select_end = draw.index("static VkExecGraphicsResult graphics_indirect_select(", argument_select)
compatibility = draw.index("static int graphics_compatibility_begin(")
compatibility_end = draw.index("\n}", compatibility) + 2
held_compatibility = "static void graphics_release_gpa_inputs(" in draw
if held_compatibility:
    release = draw.index("    if (!keep_arguments) {", draw.index("static void graphics_release_gpa_inputs("))
    release_end = draw.index("    vk_exec_graphics_vertices_release(", release)
    mesh = draw.index("static int graphics_compat_mesh_ring_entry(")
    mesh_slot = draw[mesh:draw.index("\n}", mesh) + 2]
    validate = draw.index("static int graphics_compat_arguments_current(")
    validate_arguments = draw[validate:draw.index("\n}", validate) + 2]
    release_compat = "static void graphics_release_gpa_inputs(VkExecGraphicsDraw *draw, int keep_arguments) {\n"
    release_compat += draw[release:release_end] + "}\n"
    release_compat += "static void vk_exec_graphics_draw_release_gpa_inputs(VkExecGraphicsDraw *draw) { graphics_release_gpa_inputs(draw, 0); }\n"
    retry = draw.index("void vk_exec_graphics_draw_release_gpa_inputs_retry(")
    retry_end = draw.index("    // The pending dependency belongs", retry)
    release_compat += draw[retry:retry_end] + "}\n"
else:
    release = draw.index("    vk_mem_span_capture_release(&draw->index_hold);",
                         draw.index("void vk_exec_graphics_draw_release_gpa_inputs(VkExecGraphicsDraw *draw) {"))
    release_end = draw.index("    vk_exec_graphics_vertices_release(", release)
    mesh_slot = ""
    validate_arguments = ""
    release_compat = "static void vk_exec_graphics_draw_release_gpa_inputs(VkExecGraphicsDraw *draw) {\n"
    release_compat += draw[release:release_end] + "}\n"
bounded = draw.index("static int indexed_draw_reached_bytes(")
bounded_end = draw.index("static int indexed_draw_projection_possible(", bounded)
snapshot = draw.index("static VkExecGraphicsResult indexed_draw_snapshot_prepare(")
snapshot_end = draw.index("\n}", snapshot) + 2
indexed = draw.index("static void op_draw_idx_ind(")
indexed_end = draw.index("\n}", indexed) + 2
index_types = draw.index("static int graphics_index_needs_widening(")
index_types_end = draw.index("static VkExecGraphicsResult graphics_argument_select(", index_types)
direct_indexed = draw.index("static void op_draw_idx(")
direct_indexed_end = draw.index("\n}", direct_indexed) + 2
import_scope = "static RcRegistryResult dispatch_admit_held_argument(" in draw
if import_scope:
    dispatch_argument = draw.index("static RcRegistryResult dispatch_admit_held_argument(")
    dispatch_ready = draw[dispatch_argument:draw.index("\n}", dispatch_argument) + 2]
else:
    dispatch_ready = ""
motion = draw.index("static uint32_t mo_vertex_buffers(")
motion_end = draw.index("\n}", motion) + 2
vertices = draw.index("typedef struct {\n    NVNbufferRange input;")
vertex_bind = draw.index("static int bind_vertex_buffers(", vertices)
vertex_bind_end = draw.index("\n}", vertex_bind) + 2
provider = draw.index("void vk_exec_record_native_input_provider(")
provider_end = draw.index("\n}", provider) + 2
vertex_helpers = draw.index("static uint32_t graphics_vertex_stream_mask(const VkPipeState *state,")
vertex_helpers_end = draw.index("typedef struct {\n    VkExec *ex;\n    int stage, present;", vertex_helpers)
prefix = r"""
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>
#define RC_BACKING_SINGLE_THREADED 1
#include "resource_core_backing.c"
#define RC_REGISTRY_SINGLE_THREADED 1
#define rc_registry_runtime rc_registry_unused_global
#include "resource_core_registry.c"
#undef rc_registry_runtime
typedef struct {
    RcBackingStore *store;
    RcBackingCapture *capture;
    uint32_t count;
    RcInterval intervals[8];
} MemBackingCapture;
static RcBackingStore *mock_host_store;
static RcBackingMapId mock_host_map;
static uint64_t newer_mapping_address;
static unsigned host_captures;
static RcBackingResult mem_backing_capture(uint64_t address,uint64_t size,MemBackingCapture **out) {
    *out=NULL;
    RcBackingCapture *core=NULL;
    RcBackingResult result=rc_backing_capture(mock_host_store,mock_host_map,address,size,RC_BACKING_REQUIRE_MAPPED,&core);
    if(result!=RC_BACKING_OK)return result;
    MemBackingCapture *capture=calloc(1,sizeof *capture);assert(capture);
    capture->store=mock_host_store;capture->capture=core;
    const RcBackingSegment *segments=rc_backing_capture_segments(core,&capture->count);
    assert(capture->count<=8);
    for(uint32_t i=0;i<capture->count;i++)
        capture->intervals[i]=(RcInterval){segments[i].address,segments[i].mapping,segments[i].range};
    *out=capture;host_captures++;return RC_BACKING_OK;
}
static const RcInterval *mem_backing_intervals(const MemBackingCapture *source,uint32_t *count) {
    *count=source?source->count:0;
    return source?source->intervals:NULL;
}
static void mem_backing_release(MemBackingCapture **owned) {
    if(!*owned)return;
    rc_backing_capture_release(&(*owned)->capture);
    free(*owned);*owned=NULL;host_captures--;
}
static int mem_mapping_range_unchanged(uint64_t address,uint64_t size,uint64_t cutoff) {
    assert(cutoff==1);
    return !newer_mapping_address||newer_mapping_address<address||newer_mapping_address>=address+size;
}
static int nvn_gpu_only(uint64_t address) { (void)address;return 0; }
static int mem_backing_current_range(const MemBackingCapture *c,uint64_t address,uint64_t size) {
    return c && rc_backing_capture_range_current(c->store,c->capture,address,size);
}
static int mem_backing_current(const MemBackingCapture *c) {
    return c && rc_backing_capture_current(c->store,c->capture);
}
typedef struct {
    VkBuffer buffer;
    VkDeviceSize offset,avail;
    void *alloc;
    VkDeviceSize alloc_offset;
    int cls;
    void *pool;
} VkMemSpan;
typedef struct VkMemSpanCapture VkMemSpanCapture;
typedef struct {void *pool;uint64_t mirror_incarnation;} VkMemInputProvider;
typedef enum {VK_MEM_MIRROR_EXISTING,VK_MEM_MIRROR_IMPORT,VK_MEM_MIRROR_PREPARE} VkMemMirrorInputPolicy;
void vk_mem_span_capture_release(VkMemSpanCapture **);
typedef struct PoolRec {
    uint64_t lo,size,import_lo,mirror_incarnation,host_mapping_cutoff;
    unsigned refs,mirror_input_refs;
    uint32_t acc_sub,vd_use_sub,mirror_checked_sub;
    uint64_t mirror_import_only_scene,import_writer_fence,import_writer_scene;
    uint64_t import_hi,physical_origin_upload,physical_origin_bytes;
    uint32_t import_writer_refs;
    int kind,mstate,unpublished;
    volatile int32_t used;
    int stouched;
    VkBuffer buf,mbuf;
    void *alloc, *smap;
    RcBackingId backing;
    RcBackingMapId map;
    MemBackingCapture *host_backing;
} PoolRec;
enum { POOL_IMPORT,POOL_PHYS,POOL_VIRT,MS_ON,MS_REFUSED,MS_PENDING,MS_NONE,MS_CREATING,MS_DORMANT,VK_MCLS_DEVICE,VK_MCLS_HOST_CACHED,VK_MCLS_HOST_UNC };
static int vk_mirror_on;
typedef struct { int has_index_type_uint8; } VkCtx;
typedef struct {PoolRec *buf;} pidx_ent;
static pidx_ent pool_index;
static RcBackingStore *pool_backings;
static int pool_lock,held,use_lock,gen_lock,mirror_syncs;
static uint32_t msub=11;
static int shared=1;
static int vk_exec_shared(void){return shared;}
static void plat_mutex_lock(int *lock){assert((lock==&use_lock||lock==&pool_lock||lock==&gen_lock)&&!*lock&&!held);*lock=1;}
static void plat_mutex_unlock(int *lock){assert((lock==&use_lock||lock==&pool_lock||lock==&gen_lock)&&*lock&&!held);*lock=0;}
static void mirror_sync(PoolRec *pool){
    assert(use_lock&&held&&pool->mstate==MS_ON&&pool->mirror_input_refs);
    pool->acc_sub=msub;mirror_syncs++;
}
static void plat_mutex_lock_shared(int *lock){assert(lock==&pool_lock&&!held++);}
static void plat_mutex_unlock_shared(int *lock){assert(lock==&pool_lock&&held--==1);}
static void plat_xchg32(volatile int32_t *dst,int32_t v){*dst=v;}
static VkCtx byte_index_device;
static VkCtx *vk_ctx(void){return byte_index_device.has_index_type_uint8 ? &byte_index_device : NULL;}
static void shadow_touch(PoolRec *r){assert(held&&r->smap);r->stouched++;}
static int vk_mem_lookup(uint64_t address, uint64_t size, VkMemSpan *span);
static const pidx_ent *pidx_find(const pidx_ent *index,uint64_t address,uint64_t size) {
    assert((held||pool_lock)&&index==&pool_index);
    const PoolRec *r=index->buf;
    return r&&address>=r->lo&&address-r->lo<=r->size&&size<=r->size-(address-r->lo)?index:NULL;
}
static void pool_native_acquire(PoolRec *r){assert(r->refs);r->refs++;}
static void destroy_rec(VkCtx *c,PoolRec *r){assert((!c||c==&byte_index_device)&&!held&&r->refs);r->refs--;}
typedef enum {
    VK_EXEC_HOST_INPUT_READY,
    VK_EXEC_HOST_INPUT_UNSUPPORTED,
    VK_EXEC_HOST_INPUT_BLOCKED,
    VK_EXEC_HOST_INPUT_STALE,
    VK_EXEC_HOST_INPUT_STOPPED,
    VK_EXEC_HOST_INPUT_CAPACITY,
    VK_EXEC_HOST_INPUT_ABSENT,
    VK_EXEC_HOST_INPUT_CHANGED, VK_EXEC_HOST_INPUT_UNTRACKED,
} VkExecHostInputResult;
typedef struct {
    const uint8_t *bytes;
    void *opaque;
    RcInterval mapping;
    const char *refusal;
} VkExecHostInputRead;
// At the input phase of an ordered operation, borrow readable bytes from the exact retained provider.
// The caller retains span ownership through finish/cancel and consumes CPU results only after successful finish.

static RcRegistry *registry;
static RcRegistryPosition position;
static int stop, readable=1, read_held;
static uint64_t replace_on_release;
static RcBackingId replacement_backing;
static RcRegistry *rc_registry_runtime(void){return registry;}
static RcRegistryPosition vk_exec_resource_position(void){return position;}
static int vk_sync_stopping(void){return stop;}
static int vk_device_is_lost(void){return 0;}
static int progress_stopped(void){return stop;}
static uint64_t recording_scene,cur_fence,ordinary_parts,covered_ordinary_parts,exec_sig,done,issued;
static int npart_cbs,held_n;
static uint64_t vk_sync_fence_done(void){return done;}
static void import_writer_release(VkMemSpanCapture *, uint64_t);
static void mirror_retire_check(PoolRec *pool,int completed) { assert(!completed&&pool->mstate!=MS_DORMANT); }
static int physical_use_preserve_locked(PoolRec *pool,uint64_t offset,uint64_t size) {
    assert(pool_lock&&pool->kind==POOL_PHYS);
    return offset<=pool->size && size<=pool->size-offset;
}
static int optional_attempts,optional_installs,optional_ready;
static void mirror_attempt_locked(PoolRec *pool,int verify_import) {
    assert(verify_import&&!shared&&!held&&pool&&(pool->mstate==MS_NONE||pool->mstate==MS_PENDING));
    optional_attempts++;
    if (optional_ready) {
        pool->mstate=MS_ON;optional_installs++;
    }
}
static void rt_fatal(const char *msg){fprintf(stderr,"%s\n",msg);abort();}
static int mem_backing_read_hold_at(const MemBackingCapture *mapping,uint64_t address,uint64_t size) {
    assert(!read_held);
    if(!readable||!mem_backing_current_range(mapping,address,size))return 0;
    read_held=1;return 1;
}
static void mem_read_release(void){
    assert(read_held);
    read_held=0;
    if(replace_on_release) {
        assert(rc_backing_map_replace(mock_host_store,mock_host_map,replace_on_release,4096,replacement_backing,4096)==RC_BACKING_OK);
        replace_on_release=0;
    }
}
"""
lookup = r"""
static int lookups;
static int vk_mem_lookup(uint64_t address, uint64_t size, VkMemSpan *span) {
    lookups++;
    *span=(VkMemSpan){0};
    plat_mutex_lock_shared(&pool_lock);
    const pidx_ent *entry=pidx_find(&pool_index,address,size);
    PoolRec *pool=entry?entry->buf:NULL;
    int valid=0;
    if(pool && pool->kind==POOL_IMPORT) {
        *span=(VkMemSpan){.buffer=pool->buf,.offset=address-pool->import_lo,.avail=pool->size-(address-pool->lo),
                          .cls=VK_MCLS_HOST_CACHED,.pool=pool};
        valid=1;
    } else if(pool) valid=pool_device_span(pool,address,size,span,NULL);
    plat_mutex_unlock_shared(&pool_lock);
    return valid;
}
"""
draw_prefix = r"""
#define VK_PIPE_ZERO_BINDING 16
#define ZERO_BYTES 16
#define RT_TRACE_PERF 1
#define VK_MUSE_VB 0
typedef struct {uint64_t address,size;} NVNbufferRange;
typedef struct {uint32_t fmt:8,offset:19,stream:5;} va_st;
typedef struct {uint32_t stride,divisor;} TestStream;
typedef struct {uint32_t vattr[16];TestStream vstream[16];} VkPipeState;
typedef struct {uint32_t vs_inputs;} VkPipeInfo;
typedef struct {
    VkBuffer vb[17];
    VkDeviceSize vo[17],vsz[17],vstr[17];
    int vcls[17];
    void *vpool[17];
    NVNbufferRange vin[16];
    uint64_t vgen;
    uint32_t vlegacy;
    int vb_valid;
} CbState;
typedef struct {CbState B;uint32_t c0_binst;} DrawCtx;
static DrawCtx global_draw;
static DrawCtx *dc=&global_draw;
static uint64_t vk_exec_recording_fence(void){return recording_scene;}
typedef struct {
    VkMemInputProvider *input_providers;
    int ninput_providers,cinput_providers;
} CbRes;
typedef struct {
    VkCommandBuffer cb;
    uint64_t fence;
    DrawCtx *dc;
    VkPipeState pipe;
    NVNbufferRange vbuf[16];
    int in_pass,recorded;
    uint64_t resource_op,resource_next;
    struct { unsigned width; } tonemap_extent;
    int tonemap_drawn;
} VkExec;
static VkExec *x_cur;
static struct {uint64_t drained,out;} Q;
static int part_start;
typedef enum {
    VK_EXEC_GRAPHICS_READY, VK_EXEC_GRAPHICS_NOOP, VK_EXEC_GRAPHICS_UNAVAILABLE, VK_EXEC_GRAPHICS_INVALIDATED, VK_EXEC_GRAPHICS_COMPAT,
    VK_EXEC_GRAPHICS_NEEDS_ORDERED,
} VkExecGraphicsResult;
static struct {VkBuffer zbuf;} Z;
static CbRes batch;
static uint64_t gen=12;
static int rt_trace,fail,raw_resolves,raw_uses;
static void *completed_argument;
static void (*complete_argument)(void *);
static void (*cancel_argument)(void *,int);
static struct { void *owner; void (*done)(void *); void (*cancel)(void *,int); } second_argument;
static void vk_completion_add_owned(uint64_t fence,void (*done)(void *),void (*cancel)(void *,int),void *owner) {
    assert(fence==recording_scene&&owner);
    if(completed_argument) {
        assert(!second_argument.owner);
        second_argument.owner=owner;
        second_argument.done=done;
        second_argument.cancel=cancel;
        return;
    }
    completed_argument=owner;
    complete_argument=done;
    cancel_argument=cancel;
}
static VkBuffer last_buffers[17];
static VkDeviceSize last_sizes[17];
static CbRes *cb_res(VkExec *ex){assert(ex);return &batch;}
static void *grow(void *base,int *capacity,size_t bytes) {
    *capacity=*capacity?2**capacity:4;
    void *p=realloc(base,(size_t)*capacity*bytes);
    assert(p);return p;
}

static void vk_exec_required_graphics_unavailable(VkExec *ex,uint32_t opcode,const char *reason) {
    assert(ex&&opcode&&reason);fail++;
}
static VkFormat vk_vertex_format(int fmt,int *type){
    assert(!type);return fmt==1?VK_FORMAT_R32_SFLOAT:VK_FORMAT_UNDEFINED;
}
static uint64_t vk_mem_gen(void){return gen;}
static int vk_mem_resolve(uint64_t address,uint64_t size,VkMemSpan *span){
    raw_resolves++;return vk_mem_lookup(address,size,span);
}
static void mem_use(VkMemSpan *span) {
    raw_uses++;
    PoolRec *pool=span->pool;
    if(vk_mirror_on&&pool&&pool->mstate==MS_ON)span->buffer=pool->mbuf;
}
#define VK_MEM_USE(span) mem_use(&(span))
static void vk_mem_count(int cls,int use,uint64_t size){assert(cls&&use>=VK_MUSE_VB&&use<=2&&size);}
static void vk_mem_bound_note(void *pool){assert(pool);}
static int vk_pipeline_vertex_input_dynamic(void){return 1;}
typedef struct {
    union { VkMemSpan selected, span; };
    VkMemSpanCapture *native;
    uint64_t address, bytes;
    int freeze;
} RawStorageInput;
typedef struct { void *mapped; } RawStorageInputs;
typedef struct { RawStorageInput *argument; uint64_t address, bytes; RawStorageInputs *raw_storage; } DrawDescriptorGrant;
static uint64_t image_authority_vertex;
static int indexed_draw_projection_possible(uint64_t address,RcRegistryPosition read) {
    return address==image_authority_vertex && read.scene==position.scene && read.ordinal==position.ordinal;
}
typedef struct { const VkExec *ex; } RawStorageRead;
static int raw_storage_input_current(const RawStorageRead *read,const RawStorageInput *source) {
    assert(read && read->ex && source->native);
    return !indexed_draw_projection_possible(source->address,position);
}
static int prepared_buffer_requires_image(const VkMemSpan *span,uint64_t addr,uint64_t bytes,RcRegistryPosition read) {
    (void)span;(void)bytes;
    return addr==image_authority_vertex && read.scene==position.scene && read.ordinal==position.ordinal;
}
static const RawStorageInput *prepared_buffer_argument(const DrawDescriptorGrant *grant,uint64_t address,uint64_t bytes) {
    return grant && grant->address==address && grant->bytes==bytes ? grant->argument : NULL;
}
static int prepared_buffer_argument_record(VkExec *ex,const DrawDescriptorGrant *grant,const RawStorageInput *input,
                                            VkMemSpan *span) {
    assert(ex && grant && input==grant->argument);
    *span=input->selected;return 1;
}
#define vkCmdBindVertexBuffers2 record_vertex_binding
static void record_vertex_binding(VkCommandBuffer cb,uint32_t first,uint32_t count,const VkBuffer *buffers,
                                  const VkDeviceSize *offsets,const VkDeviceSize *sizes,const VkDeviceSize *strides){
    assert(cb==(VkCommandBuffer)(uintptr_t)5&&!first&&count==17&&offsets&&!strides);
    memcpy(last_buffers,buffers,sizeof last_buffers);
    memcpy(last_sizes,sizes,sizeof last_sizes);
}
"""
draw_types = r"""
typedef struct {
    int owner;
    struct { uint8_t present, attachment_seed; } sampled;
} VkExecPreparationFailure;
typedef struct {int dummy;} GraphicsSample;
typedef struct {
    unsigned opcode;
    int mesh,activated,compatibility;
    VkPipeState state;
    RcRegistryPosition read_position;
    struct {const VkPipeInfo *info;} use;
    GraphicsVertices vertices;
    DrawDescriptorGrant descriptors, indirect_snapshot;
    RawStorageInput indirect_index_range;
    VkDrawIndexedIndirectCommand indirect_command;
    VkMemSpan index, indirect;
    VkMemSpanCapture *index_hold, *indirect_hold;
    int exact_index, exact_indirect;
    int pool_index_retry;
    const void *payload;
    void *attachment_seeds[9];
    VkExecPreparationFailure failure;
    GraphicsSample *samples;
    uint32_t sample_count;
    VkDeviceSize mesh_ring_offset;
    void *widened_index;
    int restart;
} VkExecGraphicsDraw;
static VkExecGraphicsDraw *graphics_current;
typedef struct {VkBuffer b[17];VkDeviceSize o[17],sz[17],str[17];} MoVb;
static struct {
    MoVb *vb;
    int nvb,cvb,vlegacy;
    uint64_t vgen,fence;
    uint32_t vused;
    NVNbufferRange vin[16];
} MQ;
static void *mo_grow(void *p,int *capacity,int count,size_t size) {
    return count<=*capacity?p:grow(p,capacity,size);
}
"""
compatibility_draw = r"""
enum { OP_DRAW_IDX=26, OP_DRAW_IDX_IND=39, OP_DRAW_MESH_IND,
       NVN_INDEX_TYPE_UNSIGNED_BYTE=0, NVN_INDEX_TYPE_UNSIGNED_SHORT, NVN_INDEX_TYPE_UNSIGNED_INT,
       W_INDEX, W_INDIRECT, VK_MUSE_IND=1, VK_MUSE_IB=2, VK_MCLS_SCRATCH=9, NVN_STAGE_VS=0, RT_V111=0 };
typedef struct { int itype, prim; uint64_t ind, ibuf; } p_draw_idx_ind;
typedef struct { uint64_t ind; } p_draw_mesh_ind;
typedef struct { int itype, prim, count, instances, base_vertex, base_instance; uint64_t ibuf; } p_draw_idx;
typedef struct { int narrowed; } FixtureEffectInputs;
typedef struct { int index_captured; VkDescriptorBufferInfo indices; FixtureEffectInputs *capture[1]; } EffectDeclaration;
static EffectDeclaration *effect_current;
static int pipeline_ready, bound_index, issued_draw, issued_direct, mesh_copies;
static PoolRec *sibling_pool;
static unsigned sibling_installs;
static VkBuffer wanted_buffer, wanted_indirect_buffer;
static VkDeviceSize wanted_index, wanted_indirect;
static VkIndexType wanted_index_type=VK_INDEX_TYPE_UINT16;
static void warn_once(int kind,const char *message,...) { (void)kind;(void)message;abort(); }
static int restart_for(const VkExec *ex,int prim,int itype) {
    assert(ex&&prim==7&&(itype==NVN_INDEX_TYPE_UNSIGNED_SHORT||itype==NVN_INDEX_TYPE_UNSIGNED_INT||
        (itype==NVN_INDEX_TYPE_UNSIGNED_BYTE&&byte_index_device.has_index_type_uint8)));return 1;
}
static int prepare_draw(VkExec *ex,int prim,int32_t count,int restart) {
    assert(ex&&prim==7&&(count==INT32_MAX||count==30)&&restart);
    assert(!completed_argument&&!second_argument.owner);
    if (pipeline_ready && sibling_pool && sibling_pool->mstate==MS_NONE && !mirror_import_scene(sibling_pool)) {
        // Mock only the successful Vulkan placement made by a sibling's first legacy bind.
        sibling_pool->mstate=MS_ON;
        sibling_pool->mirror_incarnation++;
        sibling_installs++;
    }
    return pipeline_ready;
}
static int graphics_widen_bind(VkExec *ex,const p_draw_idx *draw,int restart) {(void)ex;(void)draw;(void)restart;abort();}
static void shield_capture(VkExec *ex,const p_draw_idx *draw) {(void)ex;(void)draw;abort();}
static void mvobj_draw(VkExec *ex,const p_draw_idx *draw,VkBuffer buffer,VkDeviceSize offset) {
    assert(ex&&draw&&buffer==wanted_buffer&&offset==wanted_index);
}
static int graphics_storage_begin(VkExec *ex,VkExecGraphicsDraw *draw) {assert(ex&&draw==graphics_current);return 1;}
static void graphics_storage_recorded(VkExec *ex,VkExecGraphicsDraw *draw) {assert(ex&&draw==graphics_current);}
static void storage_images_written(VkExec *ex) {assert(ex);}
static struct { unsigned draws; } vk_exec_frame;
#define VK_MEM_BOUND(s) (void)(s)
#define vkCmdBindIndexBuffer record_compat_index
static void record_compat_index(VkCommandBuffer cb,VkBuffer buffer,VkDeviceSize offset,VkIndexType type) {
    assert(cb&&buffer==wanted_buffer&&offset==wanted_index&&type==wanted_index_type);
    bound_index++;
}
#define vkCmdDrawIndexedIndirect record_compat_indexed_draw
static void record_compat_indexed_draw(VkCommandBuffer cb,VkBuffer buffer,VkDeviceSize offset,uint32_t count,uint32_t stride) {
    assert(cb&&buffer==(wanted_indirect_buffer?wanted_indirect_buffer:wanted_buffer)&&offset==wanted_indirect&&count==1&&stride==sizeof(VkDrawIndexedIndirectCommand));
    issued_draw++;
}
#define vkCmdDrawIndexed record_compat_direct_draw
static void record_compat_direct_draw(VkCommandBuffer cb,uint32_t count,uint32_t instances,uint32_t first,
                                      int32_t base,uint32_t instance) {
    assert(cb&&count==30&&instances==1&&!first&&!base&&!instance);
    issued_direct++;
}
#if HELD_IMPORT_SCOPE
static int behind;
static unsigned turns;
static int vk_exec_behind(void) { return behind; }
static void vk_exec_turn(void) { assert(behind&&shared);behind=shared=0;turns++; }
static int vk_exec_recording_failed(void) { return 0; }
#endif
#if HELD_COMPATIBILITY
static int graphics_widen_current(const VkExec *ex,const void *payload,int restart,const void *grant) {
    (void)ex;(void)payload;(void)restart;(void)grant;abort();
}
static int mesh_ring_entry(VkExec *ex,const p_draw_mesh_ind *indirect,const VkMemSpan *source,VkDeviceSize *offset) {
    assert(ex&&indirect&&source->buffer==wanted_buffer&&source->offset==wanted_indirect&&completed_argument);
    *offset=48;
    return ++mesh_copies;
}
#endif
"""
snapshot_stub = r'''
static VkDrawIndexedIndirectCommand captured_command, local_command;
static RawStorageInput owned_command;
static RawStorageInputs command_receipt;
static VkExecGraphicsResult prepared_buffer_inputs(VkExec *ex, const VkPipeInfo *info, DrawDescriptorGrant *grant,
    int ordered, const RawStorageInput *args, uint32_t count, VkBufferUsageFlags usage) {
    assert(ex && !info && grant && !grant->raw_storage && args && count == 1 &&
           usage == VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT && args->native && args->freeze);
    assert(args->bytes == sizeof captured_command && vk_mem_span_capture_current(args->native, &args->span));
    if (!ordered) return VK_EXEC_GRAPHICS_NEEDS_ORDERED;
    local_command = captured_command;
    owned_command.selected = (VkMemSpan){.buffer=(VkBuffer)(uintptr_t)0x914,.avail=sizeof captured_command,.cls=VK_MCLS_SCRATCH};
    command_receipt.mapped = &local_command;
    *grant = (DrawDescriptorGrant){.argument=&owned_command,.address=args->address,
        .bytes=sizeof captured_command,.raw_storage=&command_receipt};
    return VK_EXEC_GRAPHICS_READY;
}
'''
host_watch_stubs = r'''
enum { GEN_PAGE_SHIFT=12, REG_SHIFT=21 };
#define RT_ADDRESS_LIMIT (UINT64_C(1) << 63)
static uint32_t gen_now;
static uint64_t source_generation_lineage=1;
static uint64_t gen_pages[1024], os_dirty[8], page_address[8];
static uint32_t page_stamp[8];
static int watch_untracked, freeze_possible;
static uint64_t protection_clock=1;
static int mem_freeze(uint64_t address) { assert(!(address&((1ull<<21)-1)));return freeze_possible; }
static uint64_t mem_protection_serial(void) { return protection_clock; }
static int mem_frozen_stable(uint64_t address) { return mem_freeze(address); }
static int mem_protection_range_unchanged(uint64_t address,uint64_t size,uint64_t seen) {
    assert(address&&size);return protection_clock==seen;
}
static uint64_t ww_query(uint64_t address,uint64_t size) {
    assert(gen_lock && !(address&4095) && size && size<sizeof gen_pages/sizeof *gen_pages*4096);
    if (watch_untracked) return UINT64_MAX;
    unsigned count=0;
    for (unsigned i=0;i<8;i++)
        if (os_dirty[i]>=address && os_dirty[i]-address<size) {
            gen_pages[count++]=os_dirty[i];os_dirty[i]=0;
        }
    return count;
}
static uint32_t *gen_slot(uint64_t page,int create) {
    assert(gen_lock);
    for (unsigned i=0;i<8;i++) {
        if (page_address[i]==page) return &page_stamp[i];
        if (!page_address[i] && create) {
            page_address[i]=page;return &page_stamp[i];
        }
    }
    assert(!create);
    return NULL;
}
static void stamp_page(uint64_t page,uint32_t generation) {
    *gen_slot(page,1)=generation;
}
'''
cases = r"""
static void held_argument_checks(uint64_t address, uint64_t bytes, const VkMemSpan *selected, PoolRec *pool) {
    unsigned owners=pool->refs;
    VkMemSpanCapture *plain=vk_mem_span_capture(address,bytes,selected);
    assert(plain&&pool->refs==owners+1&&selected->buffer==pool->buf);
    int first=lookups;
    assert(vk_mem_span_held_argument_current(plain,address,bytes,selected));
    assert(!vk_mem_span_held_argument_current(plain,address+1,bytes,selected));
    assert(!vk_mem_span_held_argument_current(plain,address,bytes+1,selected));
    assert(!vk_mem_span_record_held_argument(plain,address,bytes+1,selected));
    void *old_shadow=pool->smap;
    pool->smap=pool;
    assert(!pool->stouched);
    assert(vk_mem_span_record_held_argument(plain,address,bytes,selected)&&pool->stouched==1);
    assert(vk_mem_span_record_held_argument(plain,address,bytes,selected)&&pool->stouched==1);
    pool->smap=old_shadow;
    int old_mirror=vk_mirror_on,old_state=pool->mstate;
    if (pool->kind==POOL_IMPORT) {
        vk_mirror_on=1;
        pool->mstate=MS_ON;
        assert(!vk_mem_span_held_argument_current(plain,address,bytes,selected));
        assert(!vk_mem_span_record_held_argument(plain,address,bytes,selected));
        pool->mstate=old_state;
        vk_mirror_on=old_mirror;
    }
    assert(lookups==first && pool->refs==owners+1);
    vk_mem_span_capture_release(&plain);
    assert(!plain&&pool->refs==owners);
}

static void protected_cpu_input(void) {
    static _Alignas(4096) uint8_t storage[8192];
    uint64_t address=(uint64_t)(uintptr_t)storage;
    const uint64_t page=4096, read=address+page-3, bytes=6;
    storage[page-3]=41;storage[page+2]=57;
    pool_backings=rc_backing_store_create((RcBackingConfig){16,16,32,16,32});
    mock_host_store=pool_backings;
    RcBackingId backing;
    assert(rc_backing_create(pool_backings,4*page,1,&backing)==RC_BACKING_OK);
    assert(rc_backing_create(pool_backings,4*page,2,&replacement_backing)==RC_BACKING_OK);
    assert(rc_backing_map_create(pool_backings,address,sizeof storage,&mock_host_map)==RC_BACKING_OK);
    // VA order is deliberately opposite backing offset order across two mapping incarnations.
    assert(rc_backing_map_replace(pool_backings,mock_host_map,address,page,backing,2*page)==RC_BACKING_OK);
    assert(rc_backing_map_replace(pool_backings,mock_host_map,address+page,page,backing,0)==RC_BACKING_OK);
    PoolRec imported={.kind=POOL_IMPORT,.refs=1,.lo=address,.size=sizeof storage,.import_lo=address,
                      .buf=(VkBuffer)(uintptr_t)75};
    assert(mem_backing_capture(address,sizeof storage,&imported.host_backing)==RC_BACKING_OK);
    pool_index.buf=&imported;
    VkMemSpan selected={.buffer=imported.buf,.offset=page-3,.avail=page+3,.cls=VK_MCLS_HOST_CACHED,.pool=&imported};
    VkMemSpanCapture *span=vk_mem_span_capture(read,bytes,&selected);
    assert(span&&imported.refs==2);
    held_argument_checks(read,bytes,&selected,&imported);
    VkMemSpanCapture *storage_inputs=NULL;
    VkMemSpan exact={0};
    assert(vk_mem_span_native_input(read,bytes,0,&exact,&storage_inputs)==RC_REGISTRY_OK);
    assert(vk_mem_span_input_cpu_visible(storage_inputs));
    RcInterval *input_intervals=NULL;
    assert(vk_mem_span_input_intervals(storage_inputs,read+2,bytes-3,&input_intervals)==2);
    assert(input_intervals[0].address==read+2&&input_intervals[0].range.offset==3*page-1&&
           input_intervals[1].range.offset==0&&input_intervals[1].range.size==2);
    free(input_intervals);
    vk_mem_span_capture_release(&storage_inputs);
    registry=rc_registry_create((RcRegistryConfig){32,16,2});
    assert(registry);
    RcSourceRange target=imported.host_backing->intervals[1].range;
    target.size=2;
    RcRegistryWrite earlier, later;
    assert(rc_registry_write_begin_at(registry,target,RC_REGISTRY_BUFFER,(RcRegistryPosition){3,3},&earlier)==RC_REGISTRY_OK);
    assert(rc_registry_write_begin_at(registry,target,RC_REGISTRY_BUFFER,(RcRegistryPosition){3,5},&later)==RC_REGISTRY_OK);
    position=(RcRegistryPosition){3,4};
    assert(rc_registry_declare_through(registry,position)==RC_REGISTRY_OK);
    VkExecHostInputRead input={0};
    assert(vk_exec_host_input_begin(span,&selected,read,bytes,position,&input)==VK_EXEC_HOST_INPUT_BLOCKED);
    assert(!input.opaque&&!read_held);
    assert(rc_registry_write_cancel(registry,earlier)==RC_REGISTRY_OK);
    assert(vk_exec_host_input_begin(span,&selected,read,bytes,position,&input)==VK_EXEC_HOST_INPUT_READY);
    assert(read_held&&input.bytes[0]==41&&input.bytes[5]==57);
    assert(vk_exec_host_input_finish(&input)==VK_EXEC_HOST_INPUT_READY);
    assert(!input.bytes&&!input.opaque&&!read_held);
    assert(rc_registry_write_cancel(registry,later)==RC_REGISTRY_OK);

    // An unrelated page in the same 2 MiB protection region cannot invalidate
    // the borrowed bytes. A write to a demanded page must survive every drain.
    const uint64_t same_region = read ^ 0x8000;
    const uint64_t first_page = read & ~UINT64_C(4095);
    assert((same_region >> 21) == (read >> 21));
    assert(vk_exec_host_input_begin(span,&selected,read,bytes,position,&input)==VK_EXEC_HOST_INPUT_READY);
    os_dirty[0] = same_region & ~UINT64_C(4095);
    assert(vk_exec_host_input_finish(&input)==VK_EXEC_HOST_INPUT_READY && os_dirty[0]);
    os_dirty[0]=0;
    assert(vk_exec_host_input_begin(span,&selected,read,bytes,position,&input)==VK_EXEC_HOST_INPUT_READY);
    storage[page-3] = 42;
    os_dirty[0] = first_page;
    assert(vk_exec_host_input_finish(&input)==VK_EXEC_HOST_INPUT_CHANGED && !os_dirty[0]);
    assert(!input.opaque && !input.bytes && !read_held && !registry->active_gpu_captures);
    assert(vk_exec_host_input_begin(span,&selected,read,bytes,position,&input)==VK_EXEC_HOST_INPUT_READY);
    storage[page+1] = 17;
    os_dirty[0] = first_page+page;
    assert(vk_exec_host_input_finish(&input)==VK_EXEC_HOST_INPUT_CHANGED && !os_dirty[0]);
    assert(vk_exec_host_input_begin(span,&selected,read,bytes,position,&input)==VK_EXEC_HOST_INPUT_READY);
    os_dirty[0] = first_page;
    plat_mutex_lock(&gen_lock);
    const uint32_t other_observer=generation_next();
    assert(query_stamp(first_page,first_page+page,other_observer)==STAMP_DIRTY && !os_dirty[0]);
    plat_mutex_unlock(&gen_lock);
    assert(vk_exec_host_input_finish(&input)==VK_EXEC_HOST_INPUT_CHANGED);
    plat_mutex_lock(&gen_lock);
    assert(*gen_slot(first_page,0)==other_observer);
    plat_mutex_unlock(&gen_lock);
    watch_untracked=1;
    assert(vk_exec_host_input_begin(span,&selected,read,bytes,position,&input)==VK_EXEC_HOST_INPUT_UNTRACKED);
    assert(!input.opaque && !input.bytes && !read_held && !registry->active_gpu_captures);
    freeze_possible=1;
    assert(vk_exec_host_input_begin(span,&selected,read,bytes,position,&input)==VK_EXEC_HOST_INPUT_READY);
    assert(vk_exec_host_input_finish(&input)==VK_EXEC_HOST_INPUT_READY);
    assert(vk_exec_host_input_begin(span,&selected,read,bytes,position,&input)==VK_EXEC_HOST_INPUT_READY);
    protection_clock++;
    assert(vk_exec_host_input_finish(&input)==VK_EXEC_HOST_INPUT_CHANGED && !read_held);
    freeze_possible=watch_untracked=0;

    // The translated storage pointer needs logical bytes. A coherent current import can supply them even
    // when the native GPU span uses a shadow or mirror, provided actual earlier GPU writes settled.
    const uint64_t pointer_address=read-1, pointer=UINT64_C(0x64321abcde50);
    memcpy((void *)(uintptr_t)pointer_address,&pointer,sizeof pointer);
    uint64_t observed=0;
    imported.smap=storage;
    assert(vk_exec_host_input_begin(span,&selected,read,bytes,position,&input)==VK_EXEC_HOST_INPUT_UNSUPPORTED);
    assert(vk_exec_host_pointer_read(pointer_address,position,&observed)==VK_EXEC_HOST_INPUT_READY&&observed==pointer);
    imported.smap=NULL;
    RcRegistryWrite authored;
    assert(rc_registry_write_begin(registry,target,RC_REGISTRY_BUFFER,&authored)==RC_REGISTRY_OK);
    assert(rc_registry_write_submit(registry,authored,3)==RC_REGISTRY_OK);
    observed=1;
    assert(vk_exec_host_pointer_read(pointer_address,position,&observed)==VK_EXEC_HOST_INPUT_BLOCKED&&observed==1);
    assert(rc_registry_write_complete(registry,authored,0)==RC_REGISTRY_OK);
    assert(vk_exec_host_pointer_read(pointer_address,position,&observed)==VK_EXEC_HOST_INPUT_BLOCKED&&observed==1);
    uint64_t changed=pointer^UINT64_C(0x0000010000000000);
    memcpy((void *)(uintptr_t)pointer_address,&changed,sizeof changed);
    assert(rc_registry_reconcile_cpu(registry,target)==RC_REGISTRY_OK);
    assert(vk_exec_host_pointer_read(pointer_address,position,&observed)==VK_EXEC_HOST_INPUT_READY&&observed==changed);
    assert(!read_held&&!registry->active_gpu_captures);

    assert(vk_exec_host_input_begin(span,&selected,read,bytes,position,&input)==VK_EXEC_HOST_INPUT_READY);
    RcRegistryWrite arriving;
    assert(rc_registry_write_begin(registry,target,RC_REGISTRY_BUFFER,&arriving)==RC_REGISTRY_OK);
    assert(rc_registry_write_submit(registry,arriving,3)==RC_REGISTRY_OK);
    assert(rc_registry_write_complete(registry,arriving,1)==RC_REGISTRY_OK);
    assert(vk_exec_host_input_finish(&input)==VK_EXEC_HOST_INPUT_STALE);
    assert(!input.bytes&&!input.opaque&&!read_held);
    readable=0;
    assert(vk_exec_host_input_begin(span,&selected,read,bytes,position,&input)==VK_EXEC_HOST_INPUT_BLOCKED);
    readable=1;
    assert(vk_exec_host_input_begin(span,&selected,read,bytes,position,&input)==VK_EXEC_HOST_INPUT_READY);
    vk_exec_host_input_cancel(&input);
    vk_exec_host_input_cancel(&input);
    assert(!input.bytes&&!input.opaque&&!read_held&&!registry->active_gpu_captures);
    stop=1;
    assert(vk_exec_host_input_begin(span,&selected,read,bytes,position,&input)==VK_EXEC_HOST_INPUT_STOPPED);
    stop=0;
    assert(vk_exec_host_input_begin(span,&selected,read,bytes,position,&input)==VK_EXEC_HOST_INPUT_READY);
    replace_on_release=address+page;
    assert(vk_exec_host_input_finish(&input)==VK_EXEC_HOST_INPUT_STALE);
    assert(input.refusal && !strcmp(input.refusal,"host-current-span") && input.mapping.range.size>0);
    assert(!input.opaque && !input.bytes);
    assert(vk_exec_host_input_begin(span,&selected,read,bytes,position,&input)==VK_EXEC_HOST_INPUT_STALE);
    assert(!input.bytes&&!input.opaque&&!read_held&&!registry->active_gpu_captures);
    assert(vk_exec_host_pointer_read(pointer_address,position,&observed)==VK_EXEC_HOST_INPUT_STALE&&observed==changed);
    assert(vk_exec_host_pointer_read(address-8,position,&observed)==VK_EXEC_HOST_INPUT_ABSENT&&observed==changed);
    vk_mem_span_capture_release(&span);
    mem_backing_release(&imported.host_backing);
    assert(!host_captures&&imported.refs==1);
    assert(rc_backing_map_close(pool_backings,mock_host_map)==RC_BACKING_OK);
    assert(rc_backing_finalize(pool_backings,backing)==RC_BACKING_OK);
    assert(rc_backing_finalize(pool_backings,replacement_backing)==RC_BACKING_OK);
    uintptr_t provider;
    assert(rc_backing_reclaim(pool_backings,backing,&provider)==RC_BACKING_OK);
    assert(rc_backing_reclaim(pool_backings,replacement_backing,&provider)==RC_BACKING_OK);
    assert(rc_backing_store_destroy(pool_backings)==RC_BACKING_OK);
    rc_registry_destroy(registry);
}
static void optional_order(void) {
    registry=rc_registry_create((RcRegistryConfig){8,4,4});assert(registry);
    VkExec ex={.fence=8,.resource_next=1,.resource_op=2};
    x_cur=&ex;recording_scene=cur_fence=ex.fence;issued=done=7;shared=0;
    position=(RcRegistryPosition){8,4};
    assert(rc_registry_declare_through(registry,position)==RC_REGISTRY_OK);
    assert(vk_exec_mirror_input_policy(&ex,0)==VK_MEM_MIRROR_EXISTING);
    Q.out=1;
    assert(vk_exec_mirror_input_policy(&ex,1)==VK_MEM_MIRROR_EXISTING);
    Q.drained=1;ex.recorded=1;
    assert(vk_exec_mirror_input_policy(&ex,1)==VK_MEM_MIRROR_IMPORT);
    ex.recorded=part_start=0;
    held_n=1;assert(vk_exec_mirror_input_policy(&ex,1)==VK_MEM_MIRROR_IMPORT);
    held_n=0;npart_cbs=1;assert(vk_exec_mirror_input_policy(&ex,1)==VK_MEM_MIRROR_IMPORT);
    npart_cbs=0;ordinary_parts=1;assert(vk_exec_mirror_input_policy(&ex,1)==VK_MEM_MIRROR_IMPORT);
    covered_ordinary_parts=1;done=6;
    assert(vk_exec_mirror_input_policy(&ex,1)==VK_MEM_MIRROR_IMPORT);
    done=7;exec_sig=8;assert(vk_exec_mirror_input_policy(&ex,1)==VK_MEM_MIRROR_IMPORT);
    exec_sig=7;assert(vk_exec_mirror_input_policy(&ex,1)==VK_MEM_MIRROR_PREPARE);
    position.ordinal=2;assert(vk_exec_mirror_input_policy(&ex,1)==VK_MEM_MIRROR_EXISTING);
    rc_registry_destroy(registry);
    x_cur=NULL;shared=1;
}

static void vertex_bindings(uint64_t address, uint64_t page, PoolRec *window, PoolRec *physical) {
    VkPipeInfo info={.vs_inputs=7}; // 2 used usable attrs share slot 2, 1 fixed input has no usable format.
    DrawCtx context={0};
    VkExec ex={.dc=&context,.fence=11,.cb=(VkCommandBuffer)(uintptr_t)5};
    recording_scene=ex.fence;
    va_st a={.fmt=1,.stream=2};
    memcpy(&ex.pipe.vattr[0],&a,sizeof a);
    memcpy(&ex.pipe.vattr[1],&a,sizeof a);
    a=(va_st){.fmt=2,.stream=3};
    memcpy(&ex.pipe.vattr[2],&a,sizeof a);
    ex.vbuf[2]=(NVNbufferRange){address+page-4,12};
    ex.vbuf[3]=(NVNbufferRange){address+page,64};
    Z.zbuf=(VkBuffer)(uintptr_t)9;
    position=(RcRegistryPosition){11,4};
    VkExecGraphicsDraw draw={.opcode=1,.state=ex.pipe,.read_position=position,.use={&info}};
    int before=lookups;
    assert(graphics_vertices_select(&ex,&draw,VK_MEM_MIRROR_EXISTING)==VK_EXEC_GRAPHICS_READY&&lookups==before+1);
    assert(draw.vertices.used==(1u<<2)&&draw.vertices.exact==(1u<<2));
    assert(graphics_vertices_current(&ex,&draw));
    assert(physical->refs==2&&window->refs==2);
    position.ordinal+=2;
    assert(!graphics_vertices_record(&ex,&draw)&&fail==1&&!batch.ninput_providers);
    position=draw.read_position;
    VkMemSpan argument;
    VkMemSpanCapture *issued_arg=NULL,*unissued_arg=NULL;
    assert(vk_mem_span_native_input(address+page-4,12,0,&argument,&issued_arg)==RC_REGISTRY_OK);
    unissued_arg=vk_mem_span_capture(address+page-4,12,&argument);
    assert(unissued_arg);
    int argument_lookups=lookups;
    assert(graphics_argument_use(&ex,1,address+page-4,12,&argument,&issued_arg));
    assert(!issued_arg&&completed_argument&&complete_argument&&cancel_argument);
    // A remap with identical native numbers must invalidate the capture before output.
    assert(rc_backing_map_replace(pool_backings,window->map,address+page,page,physical->backing,page)==RC_BACKING_OK);
    assert(!graphics_vertices_current(&ex,&draw)&&!graphics_vertices_record(&ex,&draw)&&fail==2);
    assert(!graphics_argument_use(&ex,1,address+page-4,12,&argument,&unissued_arg));
    assert(unissued_arg&&fail==3&&!vk_mem_span_held_argument_current(completed_argument,address+page-4,12,&argument));
    assert(argument.buffer==physical->buf&&lookups==argument_lookups&&physical->refs==4);
    vk_mem_span_capture_release(&unissued_arg);
    assert(physical->refs==3);
    complete_argument(completed_argument);
    completed_argument=NULL;
    assert(physical->refs==2);
    fail--; // Count the established vertex assertions independently of this argument case.
    assert(!batch.ninput_providers&&!ex.recorded);
    vk_exec_graphics_vertices_release(&draw.vertices);
    assert(physical->refs==1&&window->refs==1);
    assert(graphics_vertices_select(&ex,&draw,VK_MEM_MIRROR_EXISTING)==VK_EXEC_GRAPHICS_READY&&graphics_vertices_current(&ex,&draw));
    assert(graphics_vertices_record(&ex,&draw)&&batch.ninput_providers==1);
    assert(physical->refs==2&&window->refs==1);
    ++gen;pool_index.buf=NULL; // Legacy re-resolution would now yield no provider and bind zeros.
    assert(bind_vertex_buffers(&ex,&draw.vertices,NULL,draw.vertices.used));
    assert(!raw_resolves&&!raw_uses&&ex.recorded==1);
    assert(last_buffers[2]==physical->buf&&last_sizes[2]==12&&last_buffers[3]==Z.zbuf);
    pool_index.buf=window;
    vk_exec_graphics_vertices_release(&draw.vertices);
    assert(graphics_vertices_select(&ex,&draw,VK_MEM_MIRROR_EXISTING)==VK_EXEC_GRAPHICS_READY&&graphics_vertices_record(&ex,&draw));
    assert(bind_vertex_buffers(&ex,&draw.vertices,NULL,draw.vertices.used));
    assert(!raw_resolves&&!raw_uses&&ex.recorded==1);
    assert(batch.ninput_providers==1&&physical->refs==2); // Actual batch holds one native owner.
    VkPipeState motion_state=ex.pipe;
    a=(va_st){.fmt=1,.stream=3};
    memcpy(&motion_state.vattr[2],&a,sizeof a);
    draw.activated=1;
    graphics_current=&draw;
    MQ.fence=ex.fence;
    PoolRec successor={.kind=POOL_IMPORT,.buf=(VkBuffer)(uintptr_t)94,.refs=1,.lo=address,
                       .import_lo=address,.size=page*4};
    pool_index.buf=&successor;
    uint32_t motion=mo_vertex_buffers(&ex,&motion_state,&info);
    assert(MQ.vb[motion].b[2]==physical->buf&&MQ.vb[motion].b[3]==successor.buf&&raw_resolves==1);
    MQ.fence++;
    motion=mo_vertex_buffers(&ex,&motion_state,&info);
    assert(MQ.vb[motion].b[2]==successor.buf&&MQ.vb[motion].b[3]==successor.buf&&raw_resolves==3);
    free(MQ.vb);
    graphics_current=NULL;
    pool_index.buf=window;
    raw_resolves=raw_uses=0;
    vk_exec_graphics_vertices_release(&draw.vertices);
    // The following compatibility draw cannot read exact or unused slots from its old raw memo.
    assert(bind_vertex_buffers(&ex,NULL,NULL,0));
    assert(raw_resolves==2&&raw_uses==2&&last_buffers[3]==physical->buf&&ex.recorded==2);
    assert(bind_vertex_buffers(&ex,NULL,NULL,0));
    assert(raw_resolves==2&&raw_uses==2&&ex.recorded==2);
    image_authority_vertex=ex.vbuf[2].address;
    assert(!bind_vertex_buffers(&ex,NULL,NULL,1u<<2)); // A still-valid legacy Vulkan binding cannot hide image authority.
    assert(ex.recorded==2);
    assert(graphics_vertices_select(&ex,&draw,VK_MEM_MIRROR_EXISTING)==VK_EXEC_GRAPHICS_READY);
    assert(graphics_vertices_record(&ex,&draw));
    assert(!bind_vertex_buffers(&ex,&draw.vertices,NULL,draw.vertices.used));
    RawStorageInput staged={.selected=draw.vertices.streams[0].span};
    staged.selected.buffer=(VkBuffer)(uintptr_t)0x712;
    DrawDescriptorGrant buffer={.argument=&staged,.address=image_authority_vertex,.bytes=12};
    assert(bind_vertex_buffers(&ex,&draw.vertices,&buffer,draw.vertices.used));
    assert(last_buffers[2]==staged.selected.buffer && last_sizes[2]==12);
    image_authority_vertex=0;
    vk_exec_graphics_vertices_release(&draw.vertices);
    vk_mem_input_provider_release(batch.input_providers[0]);
    free(batch.input_providers);
    batch=(CbRes){0};
    assert(physical->refs==1&&window->refs==1);
}

static void compatibility_arguments(uint64_t address,uint64_t page,PoolRec *window,PoolRec *physical) {
    DrawCtx context={0};
    VkExec ex={.dc=&context,.fence=12,.cb=(VkCommandBuffer)(uintptr_t)6};
    recording_scene=ex.fence;
    position=(RcRegistryPosition){12,8};
    const p_draw_idx_ind cmd={.itype=NVN_INDEX_TYPE_UNSIGNED_SHORT,.prim=7,.ibuf=address+32,.ind=address+page+16};
    VkExecGraphicsDraw draw={.opcode=OP_DRAW_IDX_IND,.payload=&cmd,.read_position=position};
    assert(graphics_argument_select(cmd.ibuf,1,2,&draw.index,&draw.index_hold,&draw.exact_index)==VK_EXEC_GRAPHICS_READY);
    assert(graphics_argument_select(cmd.ind,sizeof(VkDrawIndexedIndirectCommand),4,&draw.indirect,&draw.indirect_hold,
                                    &draw.exact_indirect)==VK_EXEC_GRAPHICS_READY);
    VkMemSpanCapture *const original_index=draw.index_hold, *const original_indirect=draw.indirect_hold;
    const int original_lookups=lookups, original_resolves=raw_resolves, original_uses=raw_uses, original_failure=fail;
    draw.attachment_seeds[0]=&ex;
    assert(!graphics_compatibility_begin(&draw)&&draw.index_hold==original_index&&draw.indirect_hold==original_indirect);
    draw.attachment_seeds[0]=NULL;
    assert(graphics_compatibility_begin(&draw)&&draw.compatibility);
    assert(draw.index_hold==original_index&&draw.indirect_hold==original_indirect&&physical->refs==3&&window->refs==3);
#if HELD_COMPATIBILITY
    vk_exec_graphics_draw_release_gpa_inputs_retry(&draw);
    assert(draw.index_hold==original_index&&draw.indirect_hold==original_indirect&&physical->refs==3);
    assert(graphics_compat_arguments_current(&ex,&draw));
    position.ordinal+=2;
    assert(!graphics_compat_arguments_current(&ex,&draw));
    position=draw.read_position;
    assert(graphics_compat_argument_prepare(cmd.ibuf,1,2,&draw.index,&draw.index_hold)==VK_EXEC_GRAPHICS_READY);
    assert(graphics_compat_argument_prepare(cmd.ind,sizeof(VkDrawIndexedIndirectCommand),4,
        &draw.indirect,&draw.indirect_hold)==VK_EXEC_GRAPHICS_READY);
    assert(graphics_compat_argument_prepare(cmd.ind,12,4,&draw.indirect,&draw.indirect_hold)==VK_EXEC_GRAPHICS_INVALIDATED);
    draw.index.offset++;
    assert(graphics_compat_argument_prepare(cmd.ibuf,1,2,&draw.index,&draw.index_hold)==VK_EXEC_GRAPHICS_INVALIDATED);
    draw.index.offset--;
    assert(lookups==original_lookups&&raw_resolves==original_resolves);
    graphics_current=&draw;
    wanted_buffer=physical->buf;
    wanted_index=draw.index.offset;
    wanted_indirect=draw.indirect.offset;
    op_draw_idx_ind(&ex,&cmd);
    assert(!bound_index&&!issued_draw&&!completed_argument&&draw.index_hold==original_index&&draw.indirect_hold==original_indirect);
    pipeline_ready=1;
    op_draw_idx_ind(&ex,&cmd);
    assert(bound_index==1&&issued_draw==1&&!draw.index_hold&&!draw.indirect_hold&&raw_uses==original_uses);
    assert(completed_argument==original_index&&second_argument.owner==original_indirect&&physical->refs==3);
    assert(rc_backing_map_replace(pool_backings,window->map,address+page,page,physical->backing,page)==RC_BACKING_OK);
    assert(!vk_mem_span_held_argument_current(original_indirect,cmd.ind,sizeof(VkDrawIndexedIndirectCommand),&draw.indirect));
    vk_exec_graphics_draw_release_gpa_inputs(&draw);
    assert(physical->refs==3&&window->refs==3); // Both actual command uses still own the original allocation.
    complete_argument(completed_argument);completed_argument=NULL;
    assert(physical->refs==2);
    second_argument.cancel(second_argument.owner,0);second_argument.owner=NULL;
    assert(physical->refs==1&&window->refs==1);
    assert(lookups==original_lookups&&raw_resolves==original_resolves&&fail==original_failure);

    draw=(VkExecGraphicsDraw){.opcode=OP_DRAW_IDX_IND,.payload=&cmd,.read_position=position,.compatibility=1};
    assert(graphics_compat_argument_prepare(cmd.ibuf,1,2,&draw.index,&draw.index_hold)==VK_EXEC_GRAPHICS_READY&&draw.index_hold);
    assert(lookups==original_lookups+1&&raw_resolves==original_resolves); // Truly absent selection uses the first lookup.
    assert(graphics_compat_argument_prepare(cmd.ind,sizeof(VkDrawIndexedIndirectCommand),4,
        &draw.indirect,&draw.indirect_hold)==VK_EXEC_GRAPHICS_READY&&draw.indirect_hold);
    assert(rc_backing_map_replace(pool_backings,window->map,address+page,page,physical->backing,page)==RC_BACKING_OK);
    assert(graphics_compat_argument_prepare(cmd.ind,sizeof(VkDrawIndexedIndirectCommand),4,
        &draw.indirect,&draw.indirect_hold)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(!graphics_compat_arguments_current(&ex,&draw));
    op_draw_idx_ind(&ex,&cmd);
    assert(issued_draw==1&&bound_index==1&&fail==original_failure+1&&lookups==original_lookups+2);
    assert(draw.indirect_hold); // Stale unissued sibling stays with the operation until cancel.
    vk_exec_graphics_draw_release_gpa_inputs(&draw);
    cancel_argument(completed_argument,0);completed_argument=NULL;
    assert(physical->refs==1&&window->refs==1);

    draw=(VkExecGraphicsDraw){.opcode=OP_DRAW_IDX_IND,.payload=&cmd,.read_position=position,.compatibility=1};
    assert(graphics_compat_argument_prepare(cmd.ibuf,1,2,&draw.index,&draw.index_hold)==VK_EXEC_GRAPHICS_READY);
    const int uncovered_lookups=lookups;
    op_draw_idx_ind(&ex,&cmd); // Only the uncovered indirect argument has a legal legacy first resolution.
    assert(issued_draw==2&&bound_index==2&&lookups==uncovered_lookups+1&&raw_resolves==original_resolves+1);
    assert(raw_uses==original_uses+1&&completed_argument&&!second_argument.owner);
    complete_argument(completed_argument);completed_argument=NULL;
    assert(physical->refs==1&&window->refs==1);

    const p_draw_mesh_ind mesh={cmd.ind};
    draw=(VkExecGraphicsDraw){.opcode=OP_DRAW_MESH_IND,.payload=&mesh,.read_position=position};
    assert(graphics_argument_select(mesh.ind,sizeof(VkDrawMeshTasksIndirectCommandEXT),4,
        &draw.indirect,&draw.indirect_hold,&draw.exact_indirect)==VK_EXEC_GRAPHICS_READY);
    const int mesh_lookups=lookups;
    assert(graphics_compatibility_begin(&draw)&&draw.indirect_hold);
    assert(graphics_compat_mesh_ring_entry(&ex,&draw)&&mesh_copies==1&&draw.mesh_ring_offset==48&&!draw.indirect_hold);
    assert(raw_uses==original_uses+1&&lookups==mesh_lookups&&completed_argument);
    complete_argument(completed_argument);completed_argument=NULL;
    graphics_current=NULL;
    raw_resolves=original_resolves;
    raw_uses=original_uses;
    assert(physical->refs==1&&window->refs==1);
#else
    (void)original_lookups;(void)original_resolves;(void)original_uses;(void)original_failure;
    (void)bound_index;(void)issued_draw;
#endif
}

static void byte_indexed_draw_issue(uint64_t address) {
    DrawCtx context={0};
    VkExec ex={.dc=&context,.fence=19,.cb=(VkCommandBuffer)(uintptr_t)7};
    p_draw_idx_ind op={.itype=NVN_INDEX_TYPE_UNSIGNED_BYTE,.prim=7,.ibuf=address+16,.ind=address+128};
    VkExecGraphicsDraw draw={.opcode=OP_DRAW_IDX_IND,.payload=&op,.read_position={19,1614},.compatibility=1};
    recording_scene=ex.fence;position=draw.read_position;
    int old_shared=shared;shared=0;
    const int previous=issued_draw;
    graphics_current=&draw;
    // Feature disabled is a safe legacy refusal, with no native bind or command issue.
    byte_index_device.has_index_type_uint8=0;
    assert(issued_draw==previous);
    byte_index_device.has_index_type_uint8=1;
    captured_command=(VkDrawIndexedIndirectCommand){.indexCount=3,.instanceCount=2,.firstIndex=3,.vertexOffset=-7,.firstInstance=11};
    assert(graphics_argument_select(op.ind,sizeof captured_command,4,&draw.indirect,&draw.indirect_hold,&draw.exact_indirect)==VK_EXEC_GRAPHICS_READY);
    assert(indexed_draw_snapshot_prepare(&ex,&draw,0)==VK_EXEC_GRAPHICS_NEEDS_ORDERED);
    assert(!draw.indirect_index_range.native && !draw.index_hold);
    assert(indexed_draw_snapshot_prepare(&ex,&draw,1)==VK_EXEC_GRAPHICS_READY);
    RawStorageInput *range=&draw.indirect_index_range;
    assert(range->native && range->address==op.ibuf+3 && range->bytes==3 && range->span.offset==19);
    assert(!draw.index_hold && draw.indirect_command.firstIndex==3);
    local_command=indexed_draw_local_command(draw.indirect_command,1);
    assert(!local_command.firstIndex && local_command.indexCount==3 && local_command.instanceCount==2 &&
           local_command.vertexOffset==-7 && local_command.firstInstance==11);
    assert(captured_command.firstIndex==3);
    wanted_buffer=range->span.buffer;wanted_index=range->span.offset;wanted_index_type=VK_INDEX_TYPE_UINT8_KHR;
    wanted_indirect_buffer=owned_command.selected.buffer;wanted_indirect=0;
    pipeline_ready=1;
    op_draw_idx_ind(&ex,&op);
    assert(issued_draw==previous+1 && !draw.indirect_hold && !range->native && completed_argument && second_argument.owner);
    complete_argument(completed_argument);completed_argument=NULL;
    second_argument.cancel(second_argument.owner,0);second_argument.owner=NULL;

    // The original zero-count and zero-instance commands read no indices,
    // and no base-index registration is needed to discover that outcome.
    op.ibuf=0;
    for (unsigned zero_instances=0; zero_instances<2; zero_instances++) {
        draw.indirect_snapshot=(DrawDescriptorGrant){0};
        captured_command.indexCount=zero_instances ? 3 : 0;
        captured_command.instanceCount=zero_instances ? 0 : 1;
        assert(graphics_argument_select(op.ind,sizeof captured_command,4,&draw.indirect,&draw.indirect_hold,&draw.exact_indirect)==VK_EXEC_GRAPHICS_READY);
        assert(indexed_draw_snapshot_prepare(&ex,&draw,1)==VK_EXEC_GRAPHICS_NOOP && !range->native);
        vk_mem_span_capture_release(&draw.indirect_hold);
    }
    assert(issued_draw==previous+1);

    // Address wrap cannot reinterpret unrelated storage as indices.
    op.ibuf=UINT64_MAX-1;
    draw.indirect_snapshot=(DrawDescriptorGrant){0};
    captured_command.indexCount=captured_command.instanceCount=1;
    assert(graphics_argument_select(op.ind,sizeof captured_command,4,&draw.indirect,&draw.indirect_hold,&draw.exact_indirect)==VK_EXEC_GRAPHICS_READY);
    assert(indexed_draw_snapshot_prepare(&ex,&draw,1)==VK_EXEC_GRAPHICS_UNAVAILABLE && !range->native);
    vk_mem_span_capture_release(&draw.indirect_hold);

    graphics_current=NULL;
    byte_index_device.has_index_type_uint8=0;
    wanted_index_type=VK_INDEX_TYPE_UINT16;
    wanted_indirect_buffer=VK_NULL_HANDLE;
    shared=old_shared;
}

static void projected_indexed_draw_issue(uint64_t address) {
    DrawCtx context={0};
    VkExec ex={.dc=&context,.fence=19,.cb=(VkCommandBuffer)(uintptr_t)7};
    p_draw_idx_ind op={.itype=NVN_INDEX_TYPE_UNSIGNED_SHORT,.prim=7,.ibuf=address+16,.ind=address+128};
    VkExecGraphicsDraw draw={.opcode=OP_DRAW_IDX_IND,.payload=&op,.read_position={19,1614},.compatibility=1};
    recording_scene=ex.fence;position=draw.read_position;
    int old_shared=shared;shared=0;
    assert(vk_mem_lookup(op.ibuf,1,&draw.index) && vk_mem_lookup(op.ind,sizeof(VkDrawIndexedIndirectCommand),&draw.indirect));
    draw.index_hold=vk_mem_span_capture(op.ibuf,1,&draw.index);
    draw.indirect_hold=vk_mem_span_capture(op.ind,sizeof(VkDrawIndexedIndirectCommand),&draw.indirect);
    assert(draw.index_hold && draw.indirect_hold);
    RawStorageInput command={.selected=draw.indirect}, indices={.selected=draw.index};
    command.selected.buffer=(VkBuffer)(uintptr_t)0x714;
    command.selected.offset=0;
    indices.selected.buffer=(VkBuffer)(uintptr_t)0x715;
    indices.selected.offset=16;
    // Modeled physical snapshots. The separate mixed-source fixture exercises selecting and copying the original bytes.
    draw.indirect_index_range=(RawStorageInput){.native=draw.index_hold,.address=op.ibuf+8,.bytes=12};
    draw.descriptors=(DrawDescriptorGrant){.argument=&indices,.address=op.ibuf+8,.bytes=12};
    RawStorageInputs snapshot_receipt={0};
    draw.indirect_snapshot=(DrawDescriptorGrant){.argument=&command,.address=op.ind,
        .bytes=sizeof(VkDrawIndexedIndirectCommand),.raw_storage=&snapshot_receipt};
    wanted_buffer=indices.selected.buffer;wanted_index=indices.selected.offset;
    wanted_indirect_buffer=command.selected.buffer;wanted_indirect=command.selected.offset;
    graphics_current=&draw;
    pipeline_ready=1;
    const int previous=issued_draw;
    op_draw_idx_ind(&ex,&op);
    assert(issued_draw==previous+1 && !draw.index_hold && !draw.indirect_hold && completed_argument && second_argument.owner);
    complete_argument(completed_argument);completed_argument=NULL;
    second_argument.cancel(second_argument.owner,0);second_argument.owner=NULL;
    graphics_current=NULL;
    wanted_indirect_buffer=VK_NULL_HANDLE;
    shared=old_shared;
}

static void held_import_draw(uint64_t address,PoolRec *pool,RcBackingMapId map,RcBackingId backing) {
    DrawCtx context={0};
    VkExec ex={.dc=&context,.fence=19,.cb=(VkCommandBuffer)(uintptr_t)7};
    const p_draw_idx cmd={.itype=NVN_INDEX_TYPE_UNSIGNED_SHORT,.prim=7,.count=30,.ibuf=address+0x200};
    VkExecGraphicsDraw draw={.opcode=OP_DRAW_IDX,.payload=&cmd,.read_position={19,1614}};
    int old_shared=shared;
    recording_scene=ex.fence;position=draw.read_position;shared=0;vk_mirror_on=1;
    pool->mstate=MS_NONE;pool->mirror_import_only_scene=0;
    assert(graphics_argument_select(cmd.ibuf,60,2,&draw.index,&draw.index_hold,&draw.exact_index)==VK_EXEC_GRAPHICS_READY);
    assert(draw.index_hold && !draw.exact_index && draw.index.buffer==pool->buf);
    assert(graphics_compatibility_begin(&draw));
    const int original_lookups=lookups, original_resolves=raw_resolves, original_uses=raw_uses;
#if HELD_IMPORT_SCOPE
    assert(vk_mem_span_prepare_held_argument(draw.index_hold,&draw.index,0)==RC_REGISTRY_BUSY);
    assert(!mirror_import_scene(pool));
#endif
    assert(graphics_compat_argument_prepare(cmd.ibuf,60,2,&draw.index,&draw.index_hold)==VK_EXEC_GRAPHICS_READY);
    wanted_buffer=pool->buf;wanted_index=0x200;
    sibling_pool=pool;graphics_current=&draw;
    op_draw_idx(&ex,&cmd);
    assert(issued_direct==1 && pool->mstate==MS_NONE && !sibling_installs);
    assert(lookups==original_lookups && raw_resolves==original_resolves && raw_uses==original_uses);
    assert(!draw.index_hold && completed_argument && pool->refs==2);
    complete_argument(completed_argument);completed_argument=NULL;
    assert(pool->refs==1);
    sibling_pool=NULL;
    image_authority_vertex=cmd.ibuf;
    const int previous_fail=fail;
    assert(graphics_compat_argument_prepare(cmd.ibuf,60,2,&draw.index,&draw.index_hold)==VK_EXEC_GRAPHICS_READY);
    op_draw_idx(&ex,&cmd);
    assert(issued_direct==1 && fail==previous_fail+1 && completed_argument);
    complete_argument(completed_argument);completed_argument=NULL;
    for (int type=NVN_INDEX_TYPE_UNSIGNED_SHORT;type<=NVN_INDEX_TYPE_UNSIGNED_INT;type++) {
        p_draw_idx projected=cmd;
        projected.itype=type;
        const uint64_t bytes=(uint64_t)projected.count<<type;
        const uint32_t alignment=1u<<type;
        draw.payload=&projected;
        assert(graphics_compat_argument_prepare(projected.ibuf,bytes,alignment,&draw.index,&draw.index_hold)==VK_EXEC_GRAPHICS_READY);
        RawStorageInput staged={.selected=draw.index};
        staged.selected.buffer=(VkBuffer)(uintptr_t)0x713;
        staged.selected.offset=16;
        draw.descriptors=(DrawDescriptorGrant){.argument=&staged,.address=projected.ibuf,.bytes=bytes};
        wanted_buffer=staged.selected.buffer;wanted_index=staged.selected.offset;
        wanted_index_type=type==NVN_INDEX_TYPE_UNSIGNED_SHORT?VK_INDEX_TYPE_UINT16:VK_INDEX_TYPE_UINT32;
        const int before=issued_direct;
        op_draw_idx(&ex,&projected);
        assert(issued_direct==before+1 && !draw.index_hold && completed_argument);
        complete_argument(completed_argument);completed_argument=NULL;
    }
    image_authority_vertex=0;
    wanted_index_type=VK_INDEX_TYPE_UINT16;
    draw.descriptors=(DrawDescriptorGrant){0};
    draw.payload=&cmd;
    graphics_current=NULL;
#if HELD_IMPORT_SCOPE
    // Scoped import authority from an earlier pending writer is already an admitted provider.
    ++msub;
    pool->import_writer_refs=1;pool->import_writer_scene=mirror_scene();
    assert(vk_mem_lookup(cmd.ibuf,sizeof(VkDispatchIndirectCommand),&draw.indirect));
    draw.indirect_hold=vk_mem_span_capture(cmd.ibuf,sizeof(VkDispatchIndirectCommand),&draw.indirect);
    assert(draw.indirect_hold);
    assert(dispatch_admit_held_argument(draw.indirect_hold,&draw.indirect)==RC_REGISTRY_OK && !turns);
    assert(vk_mem_span_record_held_argument(draw.indirect_hold,cmd.ibuf,sizeof(VkDispatchIndirectCommand),&draw.indirect));
    assert(draw.indirect.buffer==pool->buf && pool->mstate==MS_NONE);
    vk_mem_span_capture_release(&draw.indirect_hold);
    pool->import_writer_refs=0;

    // An unscoped import behind earlier worker binds waits only for its recorder turn, then keeps its span.
    ++msub;
    draw.indirect_hold=vk_mem_span_capture(cmd.ibuf,sizeof(VkDispatchIndirectCommand),&draw.indirect);
    assert(draw.indirect_hold);
    behind=shared=1;
    const int before_dispatch=lookups;
    assert(dispatch_admit_held_argument(draw.indirect_hold,&draw.indirect)==RC_REGISTRY_OK && turns==1);
    assert(mirror_import_scene(pool)==mirror_scene() && lookups==before_dispatch);
    assert(vk_mem_span_record_held_argument(draw.indirect_hold,cmd.ibuf,sizeof(VkDispatchIndirectCommand),&draw.indirect));
    vk_mem_span_capture_release(&draw.indirect_hold);

    // An already retained native mirror is valid while shared, without an import reservation or another wait.
    ++msub;pool->mstate=MS_ON;pool->acc_sub=msub;shared=1;
    assert(vk_mem_span_native_input(cmd.ibuf,sizeof(VkDispatchIndirectCommand),0,&draw.indirect,&draw.indirect_hold)==RC_REGISTRY_OK);
    assert(draw.indirect.buffer==pool->mbuf && draw.indirect_hold && pool->mirror_input_refs==1);
    assert(dispatch_admit_held_argument(draw.indirect_hold,&draw.indirect)==RC_REGISTRY_OK && turns==1);
    assert(vk_mem_span_record_held_argument(draw.indirect_hold,cmd.ibuf,sizeof(VkDispatchIndirectCommand),&draw.indirect));
    vk_mem_span_capture_release(&draw.indirect_hold);
    assert(pool->refs==1 && !pool->mirror_input_refs);

    // A replacement mapping cannot inherit the original import scope, even with identical native handles.
    ++msub;pool->mstate=MS_NONE;shared=0;
    assert(vk_mem_lookup(cmd.ibuf,60,&draw.index));
    draw.index_hold=vk_mem_span_capture(cmd.ibuf,60,&draw.index);
    assert(draw.index_hold && !mirror_import_scene(pool));
    assert(rc_backing_map_replace(pool_backings,map,address,4096,backing,0)==RC_BACKING_OK);
    assert(vk_mem_span_prepare_held_argument(draw.index_hold,&draw.index,1)==RC_REGISTRY_STALE);
    assert(!mirror_import_scene(pool) && !vk_mem_span_record_held_argument(draw.index_hold,cmd.ibuf,60,&draw.index));
    vk_mem_span_capture_release(&draw.index_hold);
    assert(pool->refs==1);
    projected_indexed_draw_issue(address);
    byte_indexed_draw_issue(address);
    assert(pool->refs==1);
#else
    (void)map;(void)backing;
#endif
    shared=old_shared;
}

int main(void) {
    protected_cpu_input();
    optional_order();
    const uint64_t page=4096, address=0x100000, args=address+page-4, size=sizeof(VkDispatchIndirectCommand);
    pool_backings=rc_backing_store_create((RcBackingConfig){16,16,32,16,32});
    assert(pool_backings);
    PoolRec physical={.kind=POOL_PHYS,.refs=1,.size=page*4,.buf=(VkBuffer)(uintptr_t)43};
    physical.alloc=&physical;
    assert(rc_backing_create(pool_backings,physical.size,(uintptr_t)&physical,&physical.backing)==RC_BACKING_OK);
    PoolRec window={.kind=POOL_VIRT,.refs=1,.lo=address,.size=physical.size};
    assert(rc_backing_map_create(pool_backings,address,physical.size,&window.map)==RC_BACKING_OK);
    for(uint64_t off=0;off<3*page;off+=page)
        assert(rc_backing_map_replace(pool_backings,window.map,address+off,page,physical.backing,off)==RC_BACKING_OK);
    pool_index.buf=&window;
    VkMemSpan selected={physical.buf,page-4,2*page+4,physical.alloc,page-4,VK_MCLS_DEVICE,NULL};
    VkMemSpanCapture *receipt=vk_mem_span_capture(args,size,&selected);
    assert(receipt&&physical.refs==2&&window.refs==2&&vk_mem_span_capture_current(receipt,&selected));
    assert(rc_backing_map_replace(pool_backings,window.map,address+2*page,page,physical.backing,page*3)==RC_BACKING_OK);
    assert(vk_mem_span_capture_current(receipt,NULL)); // another mapping in this very window changed
    assert(rc_backing_map_replace(pool_backings,window.map,address+page,page,physical.backing,page)==RC_BACKING_OK);
    assert(!vk_mem_span_capture_current(receipt,&selected)); // same native handles/offset, new overlapping mapping
    vk_mem_span_capture_release(&receipt);
    assert(!receipt&&physical.refs==1&&window.refs==1);
    // Claim more vertex/index selections than the backing capture table could own at once.
    VkMemSpanCapture *inputs[24]={0};
    VkMemSpan exact={0};
    for(int i=0;i<24;i++)
        assert(vk_mem_span_native_input(args,size,0,&exact,&inputs[i])==RC_REGISTRY_OK);
    assert(physical.refs==25&&window.refs==25&&exact.avail==size);
    RcInterval *device_intervals=NULL;
    assert(!vk_mem_span_input_cpu_visible(inputs[0]));
    assert(vk_mem_span_input_intervals(inputs[0],args+2,size-2,&device_intervals)==2);
    assert(device_intervals[0].address==args+2&&device_intervals[0].range.offset==page-2&&
           device_intervals[1].range.offset==page&&device_intervals[1].range.size==size-4);
    free(device_intervals);
    assert(!vk_mem_span_take_input_provider(&inputs[0]).pool&&inputs[0]); // Cannot adopt before record validation.
    for(int i=1;i<24;i++)vk_mem_span_capture_release(&inputs[i]);
    assert(rc_backing_map_replace(pool_backings,window.map,address+2*page,page,physical.backing,page*3)==RC_BACKING_OK);
    int before_record=lookups;
    assert(vk_mem_span_record_input(inputs[0],&exact));
    assert(lookups==before_record); // Exact record path never resolves guest VA again.
    uint64_t device_value=42;
    assert(vk_exec_host_pointer_read(args,position,&device_value)==VK_EXEC_HOST_INPUT_UNSUPPORTED&&device_value==42);
    VkMemInputProvider native=vk_mem_span_take_input_provider(&inputs[0]);
    assert(!inputs[0]&&native.pool==&physical&&!native.mirror_incarnation&&physical.refs==2&&window.refs==1);
    vk_mem_input_provider_release(native);
    vertex_bindings(address,page,&window,&physical);
    compatibility_arguments(address,page,&window,&physical);
    assert(rc_backing_map_close(pool_backings,window.map)==RC_BACKING_OK);
    assert(physical.refs==1);

    PoolRec imported={.kind=POOL_IMPORT,.mstate=MS_ON,.refs=1,.lo=address,.size=page*4,.import_lo=address,
                     .buf=(VkBuffer)(uintptr_t)55,.mbuf=(VkBuffer)(uintptr_t)56,.mirror_incarnation=11};
    RcBackingMapId host;
    assert(rc_backing_map_create(pool_backings,address,physical.size,&host)==RC_BACKING_OK);
    assert(rc_backing_map_replace(pool_backings,host,address,physical.size,physical.backing,0)==RC_BACKING_OK);
    MemBackingCapture owner={.store=pool_backings};
    assert(rc_backing_capture(pool_backings,host,address,physical.size,RC_BACKING_REQUIRE_MAPPED,&owner.capture)==RC_BACKING_OK);
    imported.host_backing=&owner;pool_index.buf=&imported;
    VkMemSpanCapture *direct=NULL;
    vk_mirror_on=1;
    imported.mstate=MS_PENDING;
    assert(vk_mem_span_native_input(args,size,1,&exact,&direct)==RC_REGISTRY_UNSUPPORTED);
    assert(!direct&&imported.refs==1&&!imported.mirror_input_refs);
    imported.mirror_checked_sub=msub;
    assert(vk_mem_span_native_input(args,size,1,&exact,&direct)==RC_REGISTRY_OK);
    assert(exact.buffer==imported.buf&&exact.avail==size&&imported.refs==2&&!imported.mirror_input_refs);
    assert(vk_mem_span_record_input(direct,&exact)&&!mirror_syncs);
    msub++;
    imported.mirror_checked_sub=msub; // Even the same result in a new scene is a new cutoff.
    assert(!vk_mem_span_capture_current(direct,&exact)&&!vk_mem_span_record_input(direct,&exact));
    assert(!vk_mem_span_take_input_provider(&direct).pool);
    vk_mem_span_capture_release(&direct);
    assert(vk_mem_span_native_input(args,size,1,&exact,&direct)==RC_REGISTRY_OK);
    imported.mstate=MS_ON;
    assert(!vk_mem_span_record_input(direct,&exact)&&!mirror_syncs); // Installed mirror supersedes the import.
    vk_mem_span_capture_release(&direct);
    assert(vk_mem_span_native_input(args,size,1,&exact,&direct)==RC_REGISTRY_OK);
    assert(exact.buffer==imported.mbuf&&exact.avail==size&&imported.mirror_input_refs==1&&imported.refs==2);
    assert(vk_mem_span_record_input(direct,&exact)&&mirror_syncs==1&&imported.vd_use_sub==msub);
    assert(vk_mem_span_record_input(direct,&exact)&&mirror_syncs==1&&!use_lock);
    VkMemInputProvider mirror_provider=vk_mem_span_take_input_provider(&direct);
    assert(!direct&&mirror_provider.pool==&imported&&mirror_provider.mirror_incarnation==imported.mirror_incarnation);
    assert(imported.mirror_input_refs==1&&imported.refs==2);
    assert(vk_mem_span_native_input(args,size,1,&exact,&direct)==RC_REGISTRY_OK&&imported.mirror_input_refs==2);
    vk_mem_span_capture_release(&direct);
    assert(imported.mirror_input_refs==1&&imported.refs==2);
    vk_mem_input_provider_release(mirror_provider);
    assert(!imported.mirror_input_refs&&imported.refs==1);
    imported.mstate=MS_NONE;shared=0;
    assert(vk_mem_span_prepare_native_input(args,size,1,VK_MEM_MIRROR_IMPORT,&exact,&direct)==RC_REGISTRY_OK);
    assert(exact.buffer==imported.buf&&!optional_attempts&&!optional_installs);
    assert(mirror_import_scene(&imported)&&vk_mem_span_record_input(direct,&exact));
    vk_mem_span_capture_release(&direct);
    assert(vk_mem_span_native_input(args,size,1,&exact,&direct)==RC_REGISTRY_OK&&exact.buffer==imported.buf);
    vk_mem_span_capture_release(&direct);
    ++msub;
    assert(vk_mem_span_prepare_native_input(args,size,1,VK_MEM_MIRROR_PREPARE,&exact,&direct)==RC_REGISTRY_OK);
    assert(exact.buffer==imported.buf&&optional_attempts==1&&!optional_installs&&mirror_import_scene(&imported));
    vk_mem_span_capture_release(&direct);
    ++msub;optional_ready=1;
    assert(vk_mem_span_prepare_native_input(args,size,1,VK_MEM_MIRROR_PREPARE,&exact,&direct)==RC_REGISTRY_OK);
    assert(exact.buffer==imported.mbuf&&optional_attempts==2&&optional_installs==1&&imported.mirror_input_refs==1);
    vk_mem_span_capture_release(&direct);
    VkMemSpan raw_import=exact;raw_import.buffer=imported.buf;
    assert(vk_mem_span_capture_bound_input(args,size,&raw_import,&direct)==RC_REGISTRY_STALE && !direct);
    assert(!imported.mirror_input_refs && imported.refs==1);
    assert(vk_mem_span_capture_bound_input(args,size,&exact,&direct)==RC_REGISTRY_OK);
    VkMemSpanCapture *parked=vk_mem_span_capture_retain(direct);
    assert(parked==direct && imported.mirror_input_refs==1 && imported.refs==2);
    shared=1;
    assert(vk_mem_span_record_input(direct,&exact));
    VkMemInputProvider recorded=vk_mem_span_take_input_provider(&direct);
    assert(recorded.pool==&imported && !direct && imported.mirror_input_refs==2 && imported.refs==3);
    assert(vk_mem_span_capture_current(parked,&exact));
    vk_mem_span_capture_release(&parked);
    assert(imported.mirror_input_refs==1 && imported.refs==2);
    vk_mem_input_provider_release(recorded);
    assert(!imported.mirror_input_refs && imported.refs==1);
    shared=1;vk_mirror_on=0;
    assert(vk_mem_span_native_input(args,size,1,&exact,&direct)==RC_REGISTRY_OK);
    assert(exact.buffer==imported.buf&&exact.avail==size);
    imported.smap=&imported;
    assert(vk_mem_span_record_input(direct,&exact)&&imported.stouched==1);
    vk_mirror_on=1;
    assert(!vk_mem_span_record_input(direct,&exact)); // Current importer is no longer direct eligible.
    vk_mirror_on=0;
    vk_mem_span_capture_release(&direct);
    selected=(VkMemSpan){.buffer=imported.mbuf,.offset=page-4,.avail=3*page+4,.pool=&imported};
    receipt=vk_mem_span_capture(args,size,&selected);
    assert(receipt&&imported.refs==2);
    assert(rc_backing_map_replace(pool_backings,host,address+2*page,page,physical.backing,3*page)==RC_BACKING_OK);
    assert(vk_mem_span_capture_current(receipt,&selected)); // host subrange survives the wider capture's churn
    PoolRec replacement=imported;
    pool_index.buf=&replacement;
    assert(!vk_mem_span_capture_current(receipt,&selected)); // same VA and native values, a different owner
    pool_index.buf=&imported;
    imported.mirror_incarnation++;
    assert(!vk_mem_span_capture_current(receipt,&selected)); // reused same native values after mirror replacement
    vk_mem_span_capture_release(&receipt);
    assert(!receipt&&imported.refs==1);
    rc_backing_capture_release(&owner.capture);

    // Outward import alignment runs into a hole; the actual argument range is wholly mapped at the import cutoff.
    assert(rc_backing_map_replace(pool_backings,host,address+page*3,page,(RcBackingId){0},0)==RC_BACKING_OK);
    imported.host_backing=NULL;
    imported.host_mapping_cutoff=1;
    mock_host_store=pool_backings;mock_host_map=host;
    assert(vk_mem_span_capture(address+3*page,size,&selected)==NULL);
    MemBackingCapture *observed=NULL;
    assert(mem_backing_capture(args,size,&observed)==RC_BACKING_OK);
    receipt=vk_mem_span_capture(args,size,&selected);
    assert(receipt&&host_captures==2&&vk_mem_canonical_source_qualify_expected(observed,receipt)==RC_REGISTRY_OK);
    newer_mapping_address=address+page*2;
    assert(vk_mem_canonical_source_qualify_expected(observed,receipt)==RC_REGISTRY_OK);
    newer_mapping_address=args;
    assert(vk_mem_canonical_source_qualify_expected(observed,receipt)==RC_REGISTRY_STALE);
    vk_mem_span_capture_release(&receipt);
    mem_backing_release(&observed);
    assert(!host_captures&&imported.refs==1);
    held_import_draw(address,&imported,host,physical.backing);
    assert(!host_captures&&imported.refs==1);
    assert(rc_backing_map_close(pool_backings,host)==RC_BACKING_OK);
    assert(rc_backing_finalize(pool_backings,physical.backing)==RC_BACKING_OK);
    uintptr_t provider=0;
    assert(rc_backing_reclaim(pool_backings,physical.backing,&provider)==RC_BACKING_OK&&provider==(uintptr_t)&physical);
    assert(rc_backing_store_destroy(pool_backings)==RC_BACKING_OK);
    puts("Indirect arguments: retained compatibility binding through queue and cancellation, exact mapping, CPU cutoff and vertex ownership passed");
    return 0;
}
"""
if "graphics_drop_geometry_samples(draw)" in draw[compatibility:compatibility_end]:
    compatibility_draw += """
static void graphics_drop_geometry_samples(VkExecGraphicsDraw *draw) { assert(!draw->sample_count); }
"""

cc = Path(os.environ["CC"])
env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="indirect-argument-") as directory:
    c = Path(directory) / "check.c"
    c.write_text(f"#define HELD_COMPATIBILITY {int(held_compatibility)}\n#define HELD_IMPORT_SCOPE {int(import_scope)}\n" + prefix +
                 qualification_types + "\n" + source[device:device_end] + "\n" + capture_helpers + source[bound_input:end] + "\n" +
                 source[import_release:import_release_end] + "\n" +
                 source[vectors:vectors_end] + "\n" + source[expected:expected_end] + "\n" +
                 source[pending_scene:pending_scene_end] + "\n" + source[mirror:mirror_end] + "\n" +
                 source[prepare_mirror:prepare_mirror_end] + "\n" +
                 sync[publish_safe:publish_safe_end] + "\n" +
                 host_write[slice_start:slice_end] + "\n" +
                 host_watch_stubs + source[generation:generation_end] + source[stamp:stamp_end] +
                 source[protected:protected_end] + source[physical_origin:physical_origin_end] + "\n" +
                 source[input_mapping:input_mapping_end] + lookup +
                 draw_prefix + exec_source[mirror_policy:mirror_policy_end] + "\n" +
                 draw[vertices:vertex_bind_end] + draw_types + draw[provider:provider_end] + "\n" +
                 draw[argument_release:argument_record_end] + draw[argument_use:argument_use_end] +
                 draw[vertex_helpers:vertex_helpers_end] + draw[motion:motion_end] + "\n" +
                 compatibility_draw + draw[argument_select:argument_select_end] + release_compat +
                 draw[compatibility:compatibility_end] + mesh_slot + validate_arguments +
                 draw[index_types:index_types_end] + draw[bounded:bounded_end] + snapshot_stub + draw[snapshot:snapshot_end] +
                 draw[indexed:indexed_end] + draw[direct_indexed:direct_indexed_end] + dispatch_ready + "\n" + cases)
    exe = c.with_suffix(".exe")
    subprocess.run([str(cc), "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                    "-fsanitize-trap=all", *([] if held_compatibility else ["-Wno-unused-function", "-Wno-unused-variable"]),
                    "-I" + str(ROOT / "runtime"), str(c), "-o", str(exe)],
                   env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

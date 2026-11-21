"""Exercise production writer capture/cancel hooks and GPA mapping scopes, without Vulkan or game data."""
from pathlib import Path
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ["CC"])

PREFIX = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "resource_core_registry.h"
#include "resource_core_shader_writes.h"
static RcRegistry *test_registry;
#define rc_registry_runtime() test_registry
typedef int PlatMutex;
static PlatMutex gpa_lock;
static uint64_t plat_now_ns(void){static uint64_t n;return ++n;}
static void plat_mutex_lock(PlatMutex *m) {(void)m;}
static void plat_mutex_unlock(PlatMutex *m) {(void)m;}
static void rt_fatal(const char *why) {fputs(why,stderr); abort();}
static void rt_log(const char *format,...) {(void)format;}
typedef struct VkGpaWriteScope VkGpaWriteScope;
typedef struct VkMemSpanCapture VkMemSpanCapture;
static void vk_mem_span_capture_release(VkMemSpanCapture **hold);
typedef struct {uint64_t addr,size;} VkGpaEffectRange;
typedef struct {uint64_t program;uint32_t opcode,stage,set,binding;uint8_t shader_key[32];} VkGpaWriteOrigin;
typedef struct {RcRegistryWrite guard;RcRegistryPosition producer,read;RcInterval source;} VkGpaPrefixRequest;
static struct {
    struct {
        uint64_t lo, hi;
        RcInterval *host;
        uint32_t host_count, next, prev;
        int cpu_visible;
    } authority[64];
    uint32_t authority_head;
} G;
'''

TEST = r'''
static RcRegistryHandle source(uint64_t addr) {
    RcRegistryHandle h = {0};
    assert(rc_registry_source(rc_registry_runtime(),addr,(RcSourceRange){1,addr,32},&h)==RC_REGISTRY_OK);
    return h;
}
static int visible(RcRegistryHandle h) {
    RcRegistryObservation o;
    return rc_registry_observe(rc_registry_runtime(),h,1,&o)==RC_REGISTRY_OK;
}
static void draws(void) {
    RcRegistryHandle h=source(0x11000), h2=source(0x12000);
    CbRes resources={0}; DrawCtx ctx={.res=&resources,.res_fence=30}; VkExec ex={.dc=&ctx,.fence=30};
    VkMemSpan direct={.visible=1};
    note_written(&ex,0x11000,32,&direct);
    assert(!visible(h) && resources.nwritten==1 && !resources.committed_written);
    assert(resources.written[0].cpu_visible);
    note_written(&ex,0x11000,32,NULL); assert(resources.nwritten==1 && !resources.written[0].cpu_visible);
    cancel_unrecorded_writes(&ex); assert(visible(h) && !resources.nwritten);
    note_written(&ex,0x11000,32,NULL); commit_recorded_writes(&ex);
    note_written(&ex,0x12000,32,NULL); cancel_unrecorded_writes(&ex);
    assert(!visible(h) && visible(h2) && resources.nwritten==1 && resources.committed_written==1);
    RcRegistryWrite committed=resources.written[0].authority;
    assert(rc_registry_write_submit(rc_registry_runtime(),committed,30)==RC_REGISTRY_OK);
    assert(rc_registry_write_complete(rc_registry_runtime(),committed,1)==RC_REGISTRY_OK);
    assert(visible(h)); free(resources.written);
    // A skipped first GPA command releases its pending scope; a subsequent skipped command cannot cancel an
    // already-emitted earlier command in the same completion owner.
    memset(&resources,0,sizeof resources);
    resources.gpa_writes=vk_gpa_write_begin(); cancel_unrecorded_writes(&ex);
    assert(!resources.gpa_writes);
    resources.gpa_writes=vk_gpa_write_begin(); commit_recorded_writes(&ex); cancel_unrecorded_writes(&ex);
    assert(resources.gpa_writes && resources.committed_gpa);
    vk_gpa_write_cancel(resources.gpa_writes); // fixture did not actually emit GPU work

    resources=(CbRes){0};
    VkMemSpan device={.alloc=(void *)(uintptr_t)1,.buffer=17};
    const RcInterval mapping={0x12000,2,{3,40,32}};
    assert(!note_buffer_written(&ex,mapping.address,32,&device,0,NULL,0));
    assert(!resources.nwritten&&!bound_captures&&!bound_releases);
    assert(note_buffer_written(&ex,mapping.address,32,&device,0,&mapping,1));
    assert(!bound_captures&&!bound_releases&&!preserved); // Declaring no mapped writer grants no preservation.
    cancel_unrecorded_writes(&ex);

    canonical_prepare=1;
    assert(note_buffer_written(&ex,mapping.address,32,&device,0,&mapping,1));
    assert(bound_captures==1&&bound_releases==1&&!preserved); // Failed publication immediately drops exact hold.
    cancel_unrecorded_writes(&ex);

    preserve_accept=1;
    assert(note_buffer_written(&ex,mapping.address,32,&device,0,&mapping,1));
    assert(bound_captures==2&&bound_releases==1&&preserved&&preserved_release);
    cancel_unrecorded_writes(&ex);
    preserved_release(preserved); // The fake command effects owner returns the accepted hold at settlement.
    preserved=NULL;
    assert(bound_releases==2&&!resources.nwritten);
    VkMemSpanCapture selected={.buffer=device.buffer};
    assert(note_buffer_written_provider(&ex,mapping.address,32,&device,0,&mapping,1,&selected));
    assert(native_accesses==1&&bound_captures==2&&preserved&&preserved!=&selected);
    preserved_release(preserved);
    preserved=NULL;
    cancel_unrecorded_writes(&ex);
    assert(bound_releases==3&&!resources.nwritten);
    free(resources.written);
}
static void fixture_map(uint32_t record,uint64_t lo,uint64_t hi,int visible) {
    RcInterval interval={lo,1,{2,lo,hi-lo}};
    authority_map(record,lo,hi,visible,&interval,1);
}
static void scopes(void) {
    RcRegistryHandle a=source(0x1000),b=source(0x5000),c=source(0x9000);
    fixture_map(1,0x1000,0x2000,1);
    VkGpaWriteScope *first=vk_gpa_write_begin();
    assert(!visible(a)&&!visible(b)&&!visible(c)); // Unknown is one symbolic domain, including future mappings.
    assert(rc_registry_stats(test_registry).writers==2);
    fixture_map(2,0x5000,0x6000,1);
    fixture_map(3,0x9000,0xa000,0);
    for(int i=0;i<1000;i++)write_scope_mapping(first,0x9000,0xa000,1);
    assert(first->count==1 && first->ranges[0].count==1);
    vk_gpa_write_complete(first,10);
    assert(visible(a)&&visible(b)&&!visible(c));
    assert(rc_registry_reconcile_cpu(test_registry,(RcSourceRange){1,0x9000,0x1000})==RC_REGISTRY_OK);
    first=vk_gpa_write_begin();vk_gpa_write_cancel(first);assert(visible(a)&&visible(c));
    first=vk_gpa_write_begin();vk_gpa_write_abandon(first,1);
    assert(!visible(a)&&!visible(c));
    assert(rc_registry_reconcile_cpu(test_registry,(RcSourceRange){1,0,UINT64_MAX})==RC_REGISTRY_OK);
    RcRegistryWriteCompletion discarded_guard={.ticket=rc_registry_unresolved_guard(test_registry)};
    assert(discarded_guard.ticket.id && rc_registry_write_complete_extents(test_registry,&discarded_guard,1)==RC_REGISTRY_OK);
    assert(rc_registry_reconcile_cpu(test_registry,(RcSourceRange){2,0,UINT64_MAX})==RC_REGISTRY_OK);
    RcRegistryHandle effect=source(0x71000),gap=source(0x72000),canonical=source(0x81000);
    // Physical-pool holds publish before aliases and remain through all command users; this separate owner avoids
    // a giant pending bounding hull between virtual address and its canonical CPU backing.
    RcRegistryWrite physical={0};
    assert(rc_registry_hold_begin(test_registry,(RcSourceRange){1,0x80000,0x4000},RC_REGISTRY_UNKNOWN,&physical)==RC_REGISTRY_OK);
    fixture_map(4,0x70000,0x74000,0);
    VkGpaEffectRange ranges[]={{0x71000,64}};
    first=vk_gpa_write_begin_ranges(ranges,1);
    assert(!visible(effect)&&visible(gap)&&!visible(canonical));
    assert(first->count==1 && first->ranges[0].hi-first->ranges[0].lo==64);
    fixture_map(5,0x70000,0x74000,1);
    vk_gpa_write_complete(first,11);
    assert(!visible(effect)&&visible(gap)&&!visible(canonical)); // Later visible remap cannot upgrade prior uncertainty.
    assert(rc_registry_hold_release(test_registry,physical,0)==RC_REGISTRY_OK);
    assert(!visible(canonical)); // Real physical destruction leaves foreign, never silently current CPU bytes.
    assert(rc_registry_reconcile_cpu(test_registry,(RcSourceRange){1,0,UINT64_MAX})==RC_REGISTRY_OK);
    first=vk_gpa_write_begin_ranges(NULL,0);assert(!first->count);vk_gpa_write_cancel(first);
    // Exact effects reserve their address even before any mapping exists.
    ranges[0]=(VkGpaEffectRange){0xb1000,64};RcRegistryHandle later=source(0xb1000);
    first=vk_gpa_write_begin_ranges(ranges,1);assert(!visible(later));
    fixture_map(6,0xb0000,0xb4000,1);vk_gpa_write_complete(first,12);assert(visible(later));
    assert(rc_registry_reconcile_cpu(test_registry,(RcSourceRange){2,0,UINT64_MAX})==RC_REGISTRY_OK);
    assert(!rc_registry_stats(test_registry).writers);
}

static void shared_manifests(void) {
    RcRegistryPosition p={100,3};
    assert(rc_registry_declare_through(test_registry,(RcRegistryPosition){100,2})==RC_REGISTRY_OK);
    uint64_t visited=writer_cost.mappings;
    VkGpaWriteScope *a=vk_gpa_write_begin_at(NULL,0,0,p);
    assert(a->manifest && writer_cost.mappings>visited);
    visited=writer_cost.mappings;
    assert(rc_registry_declare_through(test_registry,(RcRegistryPosition){100,4})==RC_REGISTRY_OK);
    VkGpaWriteScope *b=vk_gpa_write_begin_at(NULL,0,0,(RcRegistryPosition){100,5});
    assert(b->manifest==a->manifest && writer_cost.mappings==visited && a->manifest->refs==2);
    uint64_t joins=writer_cost.map_joins;
    fixture_map(7,0xd0000,0xd1000,0);
    assert(writer_cost.map_joins==joins+1); // Mapping publication joins the scene, not both independent writers.
    fixture_map(8,0xd0000,0xd1000,1); // Later visible mapping cannot erase uncertainty of an earlier target.
    assert(rc_registry_declare_through(test_registry,(RcRegistryPosition){100,5})==RC_REGISTRY_OK);
    assert(rc_registry_handoff_through(test_registry,(RcRegistryPosition){100,5})==RC_REGISTRY_OK);
    RcRegistryHandle h=source(0xd0010);
    vk_gpa_write_complete(a,100);assert(!visible(h));
    vk_gpa_write_cancel(b);assert(!scene_manifests && !visible(h));
    assert(rc_registry_reconcile_cpu(test_registry,(RcSourceRange){1,0,UINT64_MAX})==RC_REGISTRY_OK);
    assert(visible(h));
    assert(rc_registry_declare_through(test_registry,(RcRegistryPosition){101,2})==RC_REGISTRY_OK);
    visited=writer_cost.mappings;
    a=vk_gpa_write_begin_at(NULL,0,0,(RcRegistryPosition){101,3});
    assert(a->manifest && writer_cost.mappings>visited); // A later scene resnapshots; no immortal history.
    vk_gpa_write_cancel(a);assert(!scene_manifests && visible(h));
    assert(rc_registry_reconcile_cpu(test_registry,(RcSourceRange){2,0,UINT64_MAX})==RC_REGISTRY_OK);
    for (uint32_t i=1;i<=8;i++) free(G.authority[i].host);
}

static uint32_t module[256];
static uint64_t words;
static void base(void) {
    memset(module,0,sizeof module);
    module[0]=0x07230203; module[1]=0x00010500; module[3]=128; words=5;
    uint32_t types[]={4u<<16|32,1,5349,9, 4u<<16|32,2,3,9,
                      4u<<16|120,1,10,30, 4u<<16|59,2,11,3};
    memcpy(module+words,types,sizeof types); words+=sizeof types/4;
}
static void emit(const uint32_t *w,size_t n) {memcpy(module+words,w,n*4); words+=n;}
#define ADD(...) do {const uint32_t w[]={__VA_ARGS__}; emit(w,sizeof w/4);} while(0)
static void shaders(void) {
    base(); ADD(4u<<16|61,9,12,10); ADD(3u<<16|62,11,12);
    assert(!rc_shader_physical_writes(module,words)); // Physical read plus ordinary output write.
    ADD(3u<<16|62,10,12); assert(rc_shader_physical_writes(module,words));
    const uint32_t producers[]={55,57,61,65,66,67,70,83,120,124,169,245};
    for(unsigned i=0;i<sizeof producers/4;i++) {
        base(); ADD(7u<<16|producers[i],1,20,10,11,10,11); ADD(3u<<16|62,20,12);
        assert(rc_shader_physical_writes(module,words));
        base(); ADD(7u<<16|producers[i],2,20,11,11,11,11); ADD(3u<<16|62,20,12);
        assert(!rc_shader_physical_writes(module,words));
    }
    base(); ADD(6u<<16|227,9,12,10,30,30); assert(!rc_shader_physical_writes(module,words));
    base(); ADD(7u<<16|234,9,12,10,30,30,30); assert(rc_shader_physical_writes(module,words));
    base(); ADD(4u<<16|63,11,10,0); assert(!rc_shader_physical_writes(module,words));
    base(); ADD(4u<<16|64,10,11,30); assert(rc_shader_physical_writes(module,words));
    base(); ADD(3u<<16|62,99,12); assert(rc_shader_physical_writes(module,words)); // Unknown pointer producer.
    const uint32_t benign[][6]={{3u<<16|5295,12,13},{4u<<16|5294,12,13,14},
        {4u<<16|5296,9,12,13},{3u<<16|5299,12,13},{1u<<16|5380},
        {3u<<16|5381,9,12},{3u<<16|5632,10,0},{4u<<16|5633,10,0,0}};
    for(unsigned j=0;j<sizeof benign/sizeof benign[0];j++) {
        base();emit(benign[j],benign[j][0]>>16);assert(!rc_shader_physical_writes(module,words));
        ADD(3u<<16|62,10,12);assert(rc_shader_physical_writes(module,words));
    }
    const uint32_t float_atomics[]={5614,5615,6035};
    for(unsigned j=0;j<sizeof float_atomics/4;j++) {
        base();ADD(7u<<16|float_atomics[j],9,12,11,30,30,30);assert(!rc_shader_physical_writes(module,words));
        base();ADD(7u<<16|float_atomics[j],9,12,10,30,30,30);assert(rc_shader_physical_writes(module,words));
    }
    base(); ADD(1u<<16|6000); assert(rc_shader_physical_writes(module,words));
    base(); ADD(8u<<16|62,10,12); assert(rc_shader_physical_writes(module,words)); // Truncation.
}

int main(void) {
    test_registry=rc_registry_create((RcRegistryConfig){128,128,16}); assert(test_registry);
    draws(); scopes(); shared_manifests(); shaders(); images();
    assert(!definite_scopes); // This extracted writer suite has no native projected donor inputs.
    rc_registry_destroy(test_registry);
    puts("resource writers: mapping arrival, image and buffer cancellation, visibility, handoff, merge and shader classification passed");
}
'''

IMAGE_PREFIX = r"""
#include "resource_image_write.h"
static int resource_effects_image_prepare(uint64_t i,uint64_t a,uint64_t n){(void)i;(void)a;(void)n;return 0;}
static void resource_effects_image_written(uint64_t i,uint64_t a){(void)i;(void)a;}
static void resource_effects_image_owned(uint64_t i,uint64_t a,RcRegistryWrite hold){(void)i;(void)a;assert(hold.id);}
static struct {void (*fn)(void*);void (*cancel)(void*,int);void *arg;uint64_t fence;} completions[16];
static int completion_count;
static int resource_effects_image_prepare_intervals(uint64_t image,const RcInterval *i,uint32_t n)
    {(void)image;(void)i;(void)n;return 0;}
static int resource_effects_image_prepare_pinned(uint64_t image,const RcInterval *i,uint32_t n)
    {(void)image;(void)i;(void)n;return 0;}
static void resource_effects_image_written_interval(uint64_t image,RcInterval i) {(void)image;(void)i;}
static void resource_effects_image_owned_interval(uint64_t image,RcInterval i,RcRegistryWrite hold)
    {(void)image;(void)i;(void)hold;}
static void vk_completion_add_owned(uint64_t fence,void (*fn)(void*),void (*cancel)(void*,int),void *arg) {
    assert(completion_count<16); completions[completion_count++]=(typeof(completions[0])){fn,cancel,arg,fence};
}
"""
IMAGE_TEST = r"""
static void images(void) {
    RcRegistryHandle h=source(0x21000),h2=source(0x22000);
    {RESOURCE_IMAGE_SCOPE(outer,50);resource_image_prepare(1,0x21000,32);
     resource_image_prepare(1,0x21000,32); assert(!visible(h));
     {RESOURCE_IMAGE_SCOPE(inner,50);resource_image_written(1,0x21000);}
    } // Nested helper cannot commit its caller's abandoned reservation.
    assert(visible(h) && !completion_count);
    {RESOURCE_IMAGE_SCOPE(scope,51);resource_image_prepare(1,0x21000,32);resource_image_written(1,0x21000);}
    assert(!visible(h) && completion_count==1);
    completions[0].fn(completions[0].arg); completion_count=0;
    assert(!visible(h)); // Fence does not prove image-to-guest visibility.
    assert(rc_registry_reconcile_cpu(test_registry,(RcSourceRange){1,0x21000,32})==RC_REGISTRY_OK);
    RcRegistryWrite hold={0};
    {RESOURCE_IMAGE_SCOPE(scope,52);resource_image_prepare(2,0x22000,32);resource_image_written(2,0x22000);
     assert(rc_registry_hold_begin(test_registry,(RcSourceRange){1,0x22000,32},RC_REGISTRY_IMAGE,&hold)==RC_REGISTRY_OK);
     resource_image_owned(2,0x22000,hold);
    }
    assert(!visible(h2) && !completion_count);
    assert(rc_registry_hold_release(test_registry,hold,1)==RC_REGISTRY_OK); assert(visible(h2));
    {RESOURCE_IMAGE_SCOPE(scope,53);resource_image_prepare(2,0x22000,32);resource_image_written(2,0x22000);
     assert(rc_registry_hold_begin(test_registry,(RcSourceRange){1,0x22000,32},RC_REGISTRY_IMAGE,&hold)==RC_REGISTRY_OK);
     resource_image_owned(2,0x22000,hold);resource_image_written(2,0x22000);}
    assert(completion_count==1); // New emission requires new persistent ownership acknowledgment.
    completions[0].fn(completions[0].arg); completion_count=0;
    assert(rc_registry_hold_release(test_registry,hold,0)==RC_REGISTRY_OK);
    assert(rc_registry_reconcile_cpu(test_registry,(RcSourceRange){1,0x22000,32})==RC_REGISTRY_OK);
    {RESOURCE_IMAGE_SCOPE(scope,54);resource_image_prepare(1,0x21000,32);resource_image_written(1,0x21000);}
    completions[0].cancel(completions[0].arg,0);completion_count=0;
    assert(visible(h)); // Never submitted: cancel does not invent a GPU write.
    {RESOURCE_IMAGE_SCOPE(scope,55);resource_image_prepare(1,0x21000,32);resource_image_written(1,0x21000);}
    completions[0].cancel(completions[0].arg,1);completion_count=0;
    assert(!visible(h)); // A split part may have executed: retain unknown contents without a fake fence.
    assert(rc_registry_reconcile_cpu(test_registry,(RcSourceRange){1,0x21000,32})==RC_REGISTRY_OK);
    assert(visible(h) && visible(h2) && !rc_registry_stats(test_registry).writers);
}
"""
image_source=(ROOT / "runtime/resource_image_write.c").read_text()
image_source=image_source[image_source.index("struct ResourceImageWrite {"):]

source = (ROOT / "runtime/vk_gpa.c").read_text()
start = source.index("typedef struct {\n    uint64_t lo, hi;\n    RcRegistryWrite ticket;")
end = source.index("\nstatic void put(", start)
gpa = source[start:end] + '\nstatic void write_scope_alias_pages(VkGpaWriteScope *scope) {(void)scope;}\n'
draw = (ROOT / "runtime/vk_exec_draw.c").read_text()
start = draw.index("typedef struct {\n    uint64_t addr, size;\n    int back, cpu_visible;")
end = draw.index("\n", draw.index("} Range;", start))
range_type = draw[start:end]
start = draw.index("static void buffer_preserve_release(")
end = draw.index("\n// ---- stand-ins", start)
draw_helpers = draw[start:end]
adapters = r'''
typedef struct {Range *written; int nwritten,cwritten,committed_written; VkGpaWriteScope *gpa_writes; int committed_gpa;} CbRes;
typedef struct {CbRes *res; uint64_t res_fence;} DrawCtx;
typedef struct {DrawCtx *dc; uint64_t fence;} VkExec;
typedef struct {int visible; void *pool, *alloc; VkBuffer buffer;} VkMemSpan;
struct VkMemSpanCapture { VkBuffer buffer; };
static struct { unsigned storage_write_alignment; } vk_caps = {4};
static unsigned bound_captures, bound_releases, native_accesses;
static int canonical_prepare, preserve_accept;
static VkMemSpanCapture *preserved;
static void (*preserved_release)(void *);

static void vk_mem_span_capture_release(VkMemSpanCapture **hold) {
    if (!*hold) return;
    free(*hold);
    *hold = NULL;
    bound_releases++;
}
static VkMemSpanCapture *vk_mem_span_capture_retain(VkMemSpanCapture *hold) {
    VkMemSpanCapture *retained=malloc(sizeof *retained);
    assert(retained);
    *retained=*hold;
    return retained;
}
static RcRegistryResult vk_mem_span_capture_bound_input(uint64_t a,uint64_t n,const VkMemSpan *s,VkMemSpanCapture **hold) {
    assert(a==0x12000 && n==32 && s->alloc && s->buffer==17 && !*hold);
    *hold=malloc(sizeof **hold);
    assert(*hold);
    **hold=(VkMemSpanCapture){.buffer=s->buffer};
    bound_captures++;
    return RC_REGISTRY_OK;
}
static int vk_mem_span_buffer_location(const VkMemSpanCapture *hold,const VkMemSpan *s,RcRegistryBufferLocation *location) {
    assert(hold->buffer==s->buffer);
    *location=(RcRegistryBufferLocation){17,19,64};
    return 1;
}
static int vk_mem_span_buffer_preserve_current(void *opaque,RcRegistryBufferSpan result,RcRegistryPosition read,int *initialized) {
    VkMemSpanCapture *hold=opaque;
    assert(hold && hold->buffer==result.native.allocation && read.scene==30 && read.ordinal==4);
    *initialized=1;
    return 1;
}
static int resource_effects_buffer_preserve_native(const RcInterval *mapping,uint32_t count,uint64_t address,
    RcRegistryBufferLocation location,void *hold,int (*current)(void *,RcRegistryBufferSpan,RcRegistryPosition,int *),
    void (*release)(void *)) {
    assert(!preserved && count==1 && mapping->address==address && mapping->mapping==2 &&
           mapping->range.backing==3 && mapping->range.offset==40 && mapping->range.size==32);
    assert(location.allocation==17 && location.incarnation==19 && location.offset==64);
    int initialized=0;
    assert(current(hold,(RcRegistryBufferSpan){*mapping,location},(RcRegistryPosition){30,4},&initialized)&&initialized);
    if (!preserve_accept) return 0;
    preserved=hold;
    preserved_release=release;
    return 1;
}

static int nvn_gpu_only(uint64_t addr) {return addr==UINT64_MAX;}
static uint32_t vk_mem_writer_intervals(const VkMemSpan *s,uint64_t a,uint64_t n,RcInterval **out)
    {(void)s;(void)a;(void)n;*out=NULL;return 0;}
static int vk_mem_span_mirror_access_location(const VkMemSpan *s,RcRegistryBufferLocation *location)
    {(void)s;(void)location;return 0;}
static void resource_effects_buffer_access_native(const RcInterval *i,uint32_t n,uint64_t a,RcRegistryBufferLocation location)
    {assert(i && n==1 && i->mapping==2 && i->address==a && location.allocation==17 &&
            location.incarnation==19 && location.offset==64);native_accesses++;}
static int resource_effects_buffer_prepare_intervals(const RcInterval *i,uint32_t n,int v)
    {assert(i && n==1 && !v);return canonical_prepare;}
static int vk_mem_cpu_visible_span(const VkMemSpan *sp,uint64_t addr,uint64_t size) {(void)addr;(void)size;return sp&&sp->visible;}
static int resource_effects_buffer_prepare(uint64_t a,uint64_t n,int v){(void)a;(void)n;(void)v;return canonical_prepare;}
static void resource_effects_buffers_commit_payload(RcRegistryBufferPayload *payload){assert(!payload);}
static void resource_effects_opaque_commit(void){}
static void effect_gpa_commit(void){}
static void effect_capture_dump(void){}
static CbRes *cb_res(VkExec *ex) {return ex->dc->res;}
static int vk_mem_recorded_write_mirrored(const VkMemSpan *span) {(void)span;return 0;}
static void *vk_mem_recorded_provider_acquire(const VkMemSpan *span) {return span ? span->pool : NULL;}
static void vk_mem_recorded_provider_release(void *pool) {(void)pool;}
static void vk_mem_gpu_pending(uint64_t addr,uint64_t size) {(void)addr;(void)size;}
static void vk_mem_gpu_pending_span(const VkMemSpan *span,uint64_t addr,uint64_t size)
    {(void)span;vk_mem_gpu_pending(addr,size);}
static void *grow(void *data,int *capacity,size_t size) {*capacity=*capacity?2**capacity:4;return realloc(data,*capacity*size);}
'''
host_adapter = r"""
typedef uintptr_t VkBuffer;
#define VK_NULL_HANDLE ((VkBuffer)0)
typedef struct {
    RcInterval *intervals;
    RcRegistryWrite *tickets;
    RcRegistryWrite guard;
    uint32_t count;
} ResourceHostWrite;
static ResourceHostWrite resource_host_write_prepare(RcInterval *i,uint32_t n,RcRegistryWriterKind k,int known) {
    (void)known;
    return (ResourceHostWrite){.intervals=i,.count=n,.guard={.id=k==RC_REGISTRY_BUFFER}};
}
static void resource_host_write_dispose(ResourceHostWrite *w,uint64_t f,int a,int v) {
    (void)f;(void)a;(void)v;free(w->intervals);*w=(ResourceHostWrite){0};
}
"""
fixture = PREFIX + gpa + host_adapter + range_type + adapters + draw_helpers + IMAGE_PREFIX + image_source + "\nstatic void images(void);\n" + TEST + IMAGE_TEST
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="resource-writers-") as temporary:
    path = Path(temporary) / "writers.c"
    path.write_text(fixture)
    exe = path.with_suffix(".exe")
    subprocess.run([str(CC), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fsanitize-trap=all", "-DRC_REGISTRY_SINGLE_THREADED",
                    "-I", str(ROOT / "runtime"), str(path), str(ROOT / "runtime/resource_core_registry.c"),
                    "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

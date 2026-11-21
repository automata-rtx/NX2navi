"""Exercise the production segmented buffer-transfer plan and its recording handoff."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, default=ROOT / "runtime/vk_exec_xfer.c")
parser.add_argument("--runtime", type=Path, default=ROOT / "runtime")
args = parser.parse_args()
xfer = args.source.read_text(encoding="utf-8")
begin = xfer.index("typedef struct {\n    VkMemSpan source, destination;")
end = xfer.index("// ---- texture clears", begin)

STUBS = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "resource_command_effects.h"

typedef uint64_t VkBuffer, VkDeviceSize;
typedef struct { VkBuffer buffer; uint64_t offset,avail; int cls; void *pool; } VkMemSpan;
typedef struct { uint64_t srcOffset,dstOffset,size; } VkBufferCopy;
typedef struct { uint64_t src,dst,size; } p_copy_b2b;
typedef struct { uint64_t addr,size; uint32_t value; } p_clear_buf;
typedef struct { uint64_t fence,cb,recorded,resource_op; } VkExec;
typedef struct {
    uint64_t addr,size;
    int back,gpu_only;
    void *provider;
    VkBuffer buffer;
} Range;
typedef struct VkExecBufferTransfer VkExecBufferTransfer;
typedef struct {
    uint64_t copies, clears, legs, bytes, partial, mirror_twin;
} VkBufferTransferProbeStats;
enum { OP_COPY_B2B=1, OP_CLEAR_BUF, W_OVERLAP,W_BUFFER, VK_GAP_COPY_B2B,VK_GAP_CLEAR_BUF,
       VK_MUSE_COPY_SRC,VK_MUSE_GPU_WRITE };

typedef struct {
    uint64_t addr,size,backing,mapping;
    VkBuffer buffer;
    int supported,refs,mirror;
} Window;
static Window windows[]={{16,8,10,21,100,1,0,0},{24,8,11,22,101,1,0,0},
                         {64,4,30,31,200,1,0,0},{68,8,31,32,201,1,0,1},{76,0,32,33,202,1,0,0}};
typedef struct { Window *window; uint64_t mapping,address; int recorded; } VkMemSpanCapture;
struct ResourceCommandEffects { unsigned count, raw, prepared, commits, native; RcInterval parts[8]; };
static ResourceCommandEffects effects;
static int pending, done, barrier_calls, pass_ends, failures, warnings, emitted, batches, native_lookups;
static unsigned completion_count;
static struct { Range *range; void (*done)(void *); void (*cancel)(void *,int); } callbacks[8];
static VkMemSpanCapture *batch[8];
typedef struct VkTextureAliasRead {
    RcInterval interval;
    unsigned refs;
} VkTextureAliasRead;
typedef struct { RcRegistryNativeImageFailure coverage; } VkTextureAliasFailure;
typedef enum { VK_EXEC_RESTORE_READY, VK_EXEC_RESTORE_STOPPED, VK_EXEC_RESTORE_STALE, VK_EXEC_RESTORE_FAILED } VkExecRestoreResult;
enum { VK_EXEC_ALIAS_READ_LIMIT = 8 };
static VkTextureAliasRead *alias_batch[VK_EXEC_ALIAS_READ_LIMIT];
static unsigned alias_owners, alias_copies, aliases_issued, alias_prefixes, gather_density;
static int alias_stale, alias_stop;
static uint64_t alias_overlap;
static void rt_log(const char *format,...) { assert(format); }
static uint64_t emitted_size;
static void warn_once(int key,const char *fmt,...) { assert((key==W_BUFFER||key==W_OVERLAP)&&fmt); warnings++; }
static Window *lookup(uint64_t address) {
    for(size_t i=0;i<sizeof windows/sizeof windows[0];i++)
        if(address>=windows[i].addr&&address-windows[i].addr<windows[i].size) return &windows[i];
    return NULL;
}
static int vk_mem_lookup(uint64_t address,uint64_t size,VkMemSpan *out) {
    assert(size==1&&out);native_lookups++;return lookup(address)!=NULL;
}
enum { VK_MEM_MIRROR_IMPORT = 1 };
static RcRegistryResult vk_mem_span_prepare_native_input(uint64_t address,uint64_t size,int clamp,int mirror_policy,
                                                         VkMemSpan *out,VkMemSpanCapture **capture) {
    assert(clamp&&size&&!*capture&&mirror_policy==VK_MEM_MIRROR_IMPORT);
    Window *w=lookup(address);
    if(!w||!w->supported)return RC_REGISTRY_UNSUPPORTED;
    uint64_t offset=address-w->addr,available=w->size-offset;
    *out=(VkMemSpan){.buffer=w->buffer,.offset=offset,.avail=size<available?size:available,.pool=w};
    *capture=malloc(sizeof **capture);assert(*capture);
    **capture=(VkMemSpanCapture){w,w->mapping,address,0};w->refs++;
    return RC_REGISTRY_OK;
}
static RcRegistryResult vk_texture_alias_buffer_capture(const VkMemSpanCapture *capture, const VkMemSpan *span,
    uint64_t address, uint64_t size, VkTextureAliasRead **out, VkTextureAliasFailure *failure) {
    assert(capture->recorded==0 && capture->window->buffer==span->buffer && capture->address==address);
    assert(span->buffer && !*out && size == 4 && failure);
    if (!(gather_density & (1u << ((address - 16) / 4)))) return RC_REGISTRY_OK;
    *out=malloc(sizeof **out);assert(*out);
    **out=(VkTextureAliasRead){{address, 21, {alias_overlap ? alias_overlap : 10, 0, size}}, 1};
    alias_owners++;
    return RC_REGISTRY_OK;
}
static void vk_texture_alias_read_cancel(VkTextureAliasRead *alias) {
    if (alias && !--alias->refs) { assert(alias_owners); alias_owners--;free(alias); }
}
static void vk_texture_alias_read_use(VkTextureAliasRead *alias, uint64_t fence) {
    assert(fence==7&&alias->refs&&aliases_issued<VK_EXEC_ALIAS_READ_LIMIT);
    alias->refs++;
    alias_batch[aliases_issued++]=alias;
}
static int vk_texture_alias_read_current(VkTextureAliasRead *alias) {
    assert(alias->refs);return !alias_stale;
}
static const RcInterval *vk_texture_alias_read_intervals(const VkTextureAliasRead *alias, uint32_t *count) {
    *count=1;return &alias->interval;
}
static VkExecRestoreResult vk_exec_prepare_alias_readback(VkExec *ex, VkTextureAliasRead *const *aliases, uint32_t count,
    int (*current)(void *), void *scope) {
    assert(ex->fence&&count&&aliases[0]->refs&&current(scope)&&!effects.count&&!emitted);
    alias_prefixes++;
    if (alias_stop) return VK_EXEC_RESTORE_STOPPED;
    return VK_EXEC_RESTORE_READY;
}
static int vk_texture_alias_buffer_source(VkTextureAliasRead *alias, VkBuffer *buffer, VkDeviceSize *offset) {
    assert(alias->refs&&alias_prefixes&&!emitted);
    *buffer=900+alias->interval.address;
    *offset=32;
    return !alias_stale;
}
static void vk_mem_span_capture_release(VkMemSpanCapture **capture) {
    if(!*capture)return;
    assert((*capture)->window->refs>0);
    (*capture)->window->refs--;free(*capture);*capture=NULL;
}
static int vk_mem_span_capture_current(const VkMemSpanCapture *capture,const VkMemSpan *span) {
    return capture&&capture->window==span->pool&&capture->window->buffer==span->buffer&&
           lookup(capture->address)==capture->window&&capture->mapping==capture->window->mapping;
}
static int vk_mem_span_record_input(VkMemSpanCapture *capture,const VkMemSpan *span) {
    if(!vk_mem_span_capture_current(capture,span))return 0;
    capture->recorded=1;return 1;
}
static void vk_exec_record_native_input_provider(VkExec *ex,VkMemSpanCapture **capture) {
    assert(ex->cb&&(*capture)->recorded&&batches<8);
    batch[batches++]=*capture;*capture=NULL;
}
static int vk_mem_span_buffer_location(const VkMemSpanCapture *capture, const VkMemSpan *span,
                                         RcRegistryBufferLocation *value) {
    assert(capture->recorded && vk_mem_span_capture_current(capture, span));
    *value=(RcRegistryBufferLocation){span->buffer, capture->window->mirror, span->offset};
    return 1;
}
void resource_effects_buffer_prepare_native(const RcInterval *intervals, uint32_t count, uint64_t address,
                                           RcRegistryBufferLocation value) {
    Window *w=lookup(address);
    assert(count==1&&intervals->address==address&&intervals->mapping==w->mapping);
    assert(value.allocation==w->buffer&&value.incarnation==(uint64_t)w->mirror&&value.offset==address-w->addr);
    assert(!emitted&&!batches);
    effects.native++;
}
static int vk_mem_span_input_cpu_visible(const VkMemSpanCapture *capture) { return !capture->window->mirror; }
static uint32_t vk_mem_span_input_intervals(const VkMemSpanCapture *capture,uint64_t address,uint64_t size,RcInterval **out) {
    assert(address==capture->address&&size);
    *out=malloc(sizeof **out);assert(*out);
    **out=(RcInterval){address,capture->mapping,{capture->window->backing,address-capture->window->addr,size}};
    return 1;
}
RcRegistryResult resource_effects_add_intervals(ResourceCommandEffects *scope,const RcInterval *actual,
                                                uint32_t n,const RcInterval *bounds,uint32_t nd,RcRegistryWriterKind kind) {
    assert(scope==&effects&&actual==bounds&&kind==RC_REGISTRY_BUFFER&&n==nd&&n<=8);
    scope->count=n;memcpy(scope->parts,actual,n*sizeof *actual);
    return RC_REGISTRY_OK;
}
void resource_effects_add(ResourceCommandEffects *scope,RcSourceRange actual,RcSourceRange bounds,RcRegistryWriterKind kind) {
    assert(scope==&effects&&actual.backing==RC_GUEST_COMPATIBILITY_BACKING&&rc_range_equal(actual,bounds)&&
           kind==RC_REGISTRY_BUFFER&&actual.size);
    scope->raw++;
}
int resource_effects_buffer_prepare_intervals(const RcInterval *parts,uint32_t count,int visible) {
    assert(count==1 && visible==!lookup(parts->address)->mirror);
    for(unsigned i=0;i<effects.count;i++)
        if(parts->mapping==effects.parts[i].mapping&&rc_range_equal(parts->range,effects.parts[i].range)) {
            effects.prepared++;return 1;
        }
    return 0;
}
int resource_effects_buffer_prepare(uint64_t address,uint64_t count,int visible) {
    assert(effects.raw&&address>=64&&count&&visible==!lookup(address)->mirror);return 1;
}
void resource_effects_buffers_commit(void) { effects.commits++; }
static void vk_exec_required_graphics_unavailable(VkExec *ex,uint32_t opcode,const char *reason) {
    assert(ex->fence&&(opcode==OP_COPY_B2B||opcode==OP_CLEAR_BUF)&&reason);failures++;
}
static void vk_gpu_gap_note(int reason,uint64_t size) { assert(reason==VK_GAP_COPY_B2B||reason==VK_GAP_CLEAR_BUF);(void)size; }
static void vk_exec_pass_end(VkExec *ex) { assert(ex->cb);pass_ends++; }
static void vk_exec_barrier(VkExec *ex) { assert(ex->cb);barrier_calls++; }
static int nvn_gpu_only(uint64_t address) { (void)address;return 0; }
static int vk_mem_recorded_write_mirrored(const VkMemSpan *span) { return ((Window *)span->pool)->mirror; }
static void *vk_mem_recorded_provider_acquire(const VkMemSpan *span) { return span->pool; }
static VkBuffer vk_mem_guest_twin(const VkMemSpan *span) { return ((Window *)span->pool)->mirror?301:0; }
static void vkCmdCopyBuffer(uint64_t cb,VkBuffer src,VkBuffer dst,uint32_t n,const VkBufferCopy *copy) {
    assert(cb&&(dst==200||dst==201)&&n==1&&copy->size==4&&batch[0]);
    if (src==916||src==920) {
        assert(copy->srcOffset==32 && aliases_issued>alias_copies);
        alias_copies++;
    } else assert(src==100||src==101);
    assert(effects.prepared==effects.count);
    emitted++;emitted_size+=copy->size;
}
static void vkCmdFillBuffer(uint64_t cb,VkBuffer dst,uint64_t offset,uint64_t size,uint32_t value) {
    assert(cb&&(dst==200||dst==201||dst==301)&&!offset&&value==0x12345678);
    assert(size==(dst==200?4:8)&&effects.prepared==effects.count);
    emitted++;if(dst!=301)emitted_size+=size;
}
static void vk_mem_count(int cls,int kind,uint64_t size){(void)cls;assert(kind&&size);}
static void vk_usnap_drop(uint64_t address,uint64_t size) { assert(address>=64&&size); }
static void vk_mem_gpu_pending_span(const VkMemSpan *span,uint64_t address,uint64_t size) {
    assert(span->pool==lookup(address)&&size);pending++;
}
static void vk_mem_gpu_wrote(const VkMemSpan *span,uint64_t size) { assert(span->pool&&size); }
static void guest_written_cb(void *p) {
    Range *r=p;assert(r->provider&&r->size&&!r->gpu_only);done++;free(r);
}
static void guest_written_cancel(void *p,int may_have_executed) { assert(!may_have_executed);free(p); }
static void vk_completion_add_owned(uint64_t fence,void (*done_fn)(void *),void (*cancel)(void *,int),void *p) {
    assert(fence==7&&completion_count<8&&p);
    callbacks[completion_count].range=p;
    callbacks[completion_count].done=done_fn;
    callbacks[completion_count++].cancel=cancel;
}
'''

CASES = r'''
static void finish(int submit) {
    for(unsigned i=0;i<completion_count;i++)
        if(submit)callbacks[i].done(callbacks[i].range);
        else callbacks[i].cancel(callbacks[i].range,0);
    completion_count=0;
    for(int i=0;i<batches;i++)vk_mem_span_capture_release(&batch[i]);
    batches=0;
    for (unsigned i=0;i<aliases_issued;i++) vk_texture_alias_read_cancel(alias_batch[i]);
    aliases_issued=0;
    assert(!alias_owners);
    for(size_t i=0;i<sizeof windows/sizeof windows[0];i++)assert(!windows[i].refs);
}
static void reset(void) {
    assert(!completion_count&&!batches);
    effects=(ResourceCommandEffects){0};
    pending=done=barrier_calls=pass_ends=failures=warnings=emitted=native_lookups=0;
    emitted_size=0;
}
int main(void) {
    const p_copy_b2b copy={16,64,16};
    const p_clear_buf clear={64,16,0x12345678};
    VkExecBufferTransfer *plan=NULL;
    VkExec ex={.fence=7,.cb=71,.resource_op=4};
    const RcRegistryPosition read={7,8}, output={7,9};
    vk_exec_buffer_transfer_probe_arm(1);
    assert(vk_exec_buffer_transfer_prepare(OP_COPY_B2B,&copy,read,&plan)==RC_REGISTRY_OK);
    assert(plan->count==3&&plan->missing&&plan->missing_address==76&&native_lookups==1);
    assert(vk_exec_buffer_transfer_declare(plan,&effects,output)==RC_REGISTRY_OK&&effects.count==3&&effects.raw==3);
    buffer_transfer_record(&ex,plan);vk_exec_buffer_transfer_release(plan);plan=NULL;
    assert(emitted_size==12&&emitted==3&&completion_count==3&&pending==3&&batches==6);
    assert(barrier_calls==1&&pass_ends==1&&native_lookups==2&&warnings==1&&!failures&&effects.commits==1&&effects.native==3);
    for(unsigned i=0;i<completion_count;i++)
        assert(callbacks[i].range->back==(i!=0));
    finish(1);assert(done==3);
    VkBufferTransferProbeStats stats=vk_exec_buffer_transfer_probe_stats();
    assert(stats.copies==1 && !stats.clears && stats.legs==3 && stats.bytes==12 && stats.partial==1 && !stats.mirror_twin);

    reset();
    assert(vk_exec_buffer_transfer_prepare(OP_CLEAR_BUF,&clear,read,&plan)==RC_REGISTRY_OK&&plan->count==2);
    assert(vk_exec_buffer_transfer_declare(plan,&effects,output)==RC_REGISTRY_OK&&effects.count==2);
    buffer_transfer_record(&ex,plan);vk_exec_buffer_transfer_release(plan);plan=NULL;
    assert(emitted==3&&emitted_size==12&&pending==2&&batches==2&&completion_count==2&&!failures&&effects.native==2);
    finish(0);assert(!done);
    stats=vk_exec_buffer_transfer_probe_stats();
    assert(stats.copies==1 && stats.clears==1 && stats.legs==5 && stats.bytes==24 && stats.partial==2 && stats.mirror_twin==1);

    reset();windows[3].supported=0;
    assert(vk_exec_buffer_transfer_prepare(OP_COPY_B2B,&copy,read,&plan)==RC_REGISTRY_UNSUPPORTED&&!plan);
    assert(!emitted&&!pending&&!effects.count);finish(0);
    windows[3].supported=1;

    reset();
    assert(vk_exec_buffer_transfer_prepare(OP_COPY_B2B,&copy,read,&plan)==RC_REGISTRY_OK);
    assert(vk_exec_buffer_transfer_declare(plan,&effects,output)==RC_REGISTRY_OK);
    windows[4].size=4;
    buffer_transfer_record(&ex,plan);vk_exec_buffer_transfer_release(plan);plan=NULL;
    assert(failures==1&&!emitted&&!pending&&!effects.commits&&!batches);
    finish(0);windows[4].size=0;

    reset();
    assert(vk_exec_buffer_transfer_prepare(OP_CLEAR_BUF,&clear,read,&plan)==RC_REGISTRY_OK);
    assert(vk_exec_buffer_transfer_declare(plan,&effects,output)==RC_REGISTRY_OK);
    windows[3].mapping++;
    buffer_transfer_record(&ex,plan);vk_exec_buffer_transfer_release(plan);plan=NULL;
    assert(failures==1&&!emitted&&!pending&&!effects.commits&&!batches);
    finish(0);windows[3].mapping--;
    stats=vk_exec_buffer_transfer_probe_stats();
    assert(stats.legs==5 && stats.bytes==24); // Rejected pre-emission plans are absent from the witness.
    vk_exec_buffer_transfer_probe_arm(0);

    reset();
    gather_density=3; // First two segments use a single prefix, the third still uses the original clean VkBuffer.
    assert(vk_exec_buffer_transfer_prepare(OP_COPY_B2B,&copy,read,&plan)==RC_REGISTRY_OK);
    assert(vk_exec_buffer_transfer_sources_prepare(&ex,plan)==RC_REGISTRY_OK);
    assert(alias_prefixes==1&&alias_owners==2&&!effects.count);
    assert(vk_exec_buffer_transfer_declare(plan,&effects,output)==RC_REGISTRY_OK&&effects.count==3);
    buffer_transfer_record(&ex,plan);vk_exec_buffer_transfer_release(plan);plan=NULL;
    assert(alias_copies==2&&aliases_issued==2&&alias_owners==2&&emitted==3&&!failures);
    finish(0);
    alias_copies=0;

    reset();
    gather_density=2;alias_overlap=30; // The second source would have to consume the first leg's own write.
    assert(vk_exec_buffer_transfer_prepare(OP_COPY_B2B,&copy,read,&plan)==RC_REGISTRY_OK);
    assert(vk_exec_buffer_transfer_sources_prepare(&ex,plan)==RC_REGISTRY_BLOCKED);
    vk_exec_buffer_transfer_release(plan);plan=NULL;
    assert(alias_prefixes==1&&!alias_owners&&!effects.count&&!emitted);
    finish(0);
    alias_overlap=0;

    reset();
    gather_density=1;alias_stop=1;
    assert(vk_exec_buffer_transfer_prepare(OP_COPY_B2B,&copy,read,&plan)==RC_REGISTRY_OK);
    assert(vk_exec_buffer_transfer_sources_prepare(&ex,plan)==RC_REGISTRY_BUSY);
    vk_exec_buffer_transfer_release(plan);plan=NULL;
    assert(alias_prefixes==2&&!alias_owners&&!effects.count&&!emitted);
    alias_stop=0;
    finish(0);

    reset();
    assert(vk_exec_buffer_transfer_prepare(OP_COPY_B2B,&copy,read,&plan)==RC_REGISTRY_OK);
    assert(vk_exec_buffer_transfer_sources_prepare(&ex,plan)==RC_REGISTRY_OK);
    assert(vk_exec_buffer_transfer_declare(plan,&effects,output)==RC_REGISTRY_OK);
    alias_stale=1;
    buffer_transfer_record(&ex,plan);vk_exec_buffer_transfer_release(plan);plan=NULL;
    assert(alias_prefixes==3&&!alias_owners&&!aliases_issued&&!emitted&&failures==1);
    finish(0);
    gather_density=alias_stale=0;

    const p_copy_b2b overlap={16,18,4};
    reset();
    assert(vk_exec_buffer_transfer_prepare(OP_COPY_B2B,&overlap,read,&plan)==RC_REGISTRY_OK&&plan->stop==BUFFER_TRANSFER_OVERLAP);
    assert(vk_exec_buffer_transfer_declare(plan,&effects,output)==RC_REGISTRY_OK&&!effects.count);
    buffer_transfer_record(&ex,plan);vk_exec_buffer_transfer_release(plan);plan=NULL;
    assert(warnings==1&&!emitted&&!batches&&!failures);finish(0);
    reset();
    const p_copy_b2b empty={16,64,0};
    assert(vk_exec_buffer_transfer_prepare(OP_COPY_B2B,&empty,read,&plan)==RC_REGISTRY_OK&&!plan->count);
    assert(vk_exec_buffer_transfer_declare(plan,&effects,output)==RC_REGISTRY_OK);
    buffer_transfer_record(&ex,plan);vk_exec_buffer_transfer_release(plan);plan=NULL;
    assert(!emitted&&!warnings&&!batches&&!native_lookups);finish(0);
    reset();
    const p_clear_buf unaligned={65,4,0x12345678};
    assert(vk_exec_buffer_transfer_prepare(OP_CLEAR_BUF,&unaligned,read,&plan)==RC_REGISTRY_OK);
    assert(plan->stop==BUFFER_TRANSFER_UNALIGNED&&!plan->count&&!native_lookups);
    assert(vk_exec_buffer_transfer_declare(plan,&effects,(RcRegistryPosition){7,11})==RC_REGISTRY_STALE);
    assert(!effects.count&&!effects.raw&&!emitted);
    assert(vk_exec_buffer_transfer_declare(plan,&effects,output)==RC_REGISTRY_OK);
    buffer_transfer_record(&ex,plan);vk_exec_buffer_transfer_release(plan);plan=NULL;
    assert(!effects.count&&!effects.raw&&!effects.commits&&!pass_ends&&!pending&&!emitted&&!batches&&warnings==1);
    finish(0);

    reset();
    assert(vk_exec_buffer_transfer_prepare(OP_COPY_B2B,&copy,read,&plan)==RC_REGISTRY_OK);
    ex.resource_op++;
    assert(vk_exec_buffer_transfer_sources_prepare(&ex,plan)==RC_REGISTRY_STALE);
    assert(!effects.count&&!emitted&&!alias_owners);
    vk_exec_buffer_transfer_release(plan);plan=NULL;
    finish(0);
    puts("Ordered buffer transfer: exact position, typed no-op, issued prefix, retained mappings, mirror twin, handoff and cancellation passed");
}
'''

cc = Path(os.environ["CC"])
env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="ordered-buffer-transfer-") as temp:
    c = Path(temp) / "check.c"
    c.write_text(STUBS + xfer[begin:end] + CASES, encoding="utf-8")
    exe = c.with_suffix(".exe")
    subprocess.run([str(cc), "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fsanitize-trap=all", "-I" + str(args.runtime),
                    str(c), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

"""Exercise actual inline-update payload selection after native writes, including aliased and remapped reads."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runtime', type=Path, default=ROOT / 'runtime')
args = parser.parse_args()
source = (args.runtime / 'vk_exec_draw.c').read_text()
start = source.index('// Each pointer-read payload names a native inline producer')
end = source.index('static int uniform_pointer_value(', start)

PREFIX = r'''
#include <assert.h>
#include <stdio.h>
#define RC_REGISTRY_SINGLE_THREADED
#define rc_registry_runtime registry_unused_runtime
#include "resource_core_registry.c"
#undef rc_registry_runtime
#define OK(x) assert((x)==RC_REGISTRY_OK)
typedef struct { uint64_t buffer,offset,avail; } VkMemSpan;
typedef struct { uint64_t fence,resource_op; } VkExec;
typedef struct { uint64_t dst,size; uint8_t data[64]; } p_update_ub;
typedef struct { VkMemSpan span; uint64_t address,incarnation; } VkMemSpanCapture;
typedef struct { VkMemSpan span; VkMemSpanCapture *hold; RcInterval *writes; uint32_t count; int backed; } VkExecUpdateDestination;
typedef enum { VK_EXEC_UNIFORM_UPDATE_NONE,VK_EXEC_UNIFORM_UPDATE_READY,VK_EXEC_UNIFORM_UPDATE_STALE } VkExecUniformUpdateRead;
typedef struct {
    uint64_t offset, bytes;
    const void *data;
    RcRegistryPosition output;
} VkExecUniformUpdateFragment;
static RcRegistry *r;
static RcRegistry *rc_registry_runtime(void){return r;}
static RcRegistryPosition position;
static RcRegistryPosition vk_exec_resource_position(void){return position;}
static _Noreturn void rt_fatal(const char *fmt){fputs(fmt,stderr);abort();}
static uint64_t incarnation=1, alias_buffer=7, backing=3;
static int unavailable, mapping_race, reads;
static uint32_t vk_exec_upd_top(uint32_t n){return n;}
static void vk_uniform_payload_reap(void) {} // This fixture owns only the local borrowed journal.
static RcRegistryResult vk_mem_span_native_input(uint64_t a,uint64_t n,int clamp,VkMemSpan *span,VkMemSpanCapture **out){
    *span=(VkMemSpan){0}; assert(!clamp&&!*out&&n&&n<=65536);
    if(unavailable)return RC_REGISTRY_UNSUPPORTED;
    *span=(VkMemSpan){(a>=0x2000&&a<0x3000)?alias_buffer:7,0x200+(a&0xfff),n};
    *out=malloc(sizeof **out);assert(*out);**out=(VkMemSpanCapture){*span,a,incarnation};
    reads=0;return RC_REGISTRY_OK;
}
static int vk_mem_span_capture_current(const VkMemSpanCapture *capture,const VkMemSpan *span){
    assert(span->buffer==capture->span.buffer);
    if(++reads==2&&mapping_race) incarnation++;
    return incarnation==capture->incarnation;
}
static void vk_mem_span_capture_release(VkMemSpanCapture **hold){free(*hold);*hold=NULL;}
static uint32_t vk_mem_span_input_intervals(const VkMemSpanCapture *capture,uint64_t a,uint64_t n,RcInterval **out){
    assert(capture->address==a&&!*out&&n==capture->span.avail);
    *out=calloc(2,sizeof **out);assert(*out);
    uint64_t offset=a&0xfff;
    uint64_t first=offset<8?8-offset:0;
    if(first>n)first=n;
    unsigned count=0;
    if(first)(*out)[count++]=(RcInterval){a,incarnation,{backing,100+offset,first}};
    if(n>first)(*out)[count++]=(RcInterval){a+first,incarnation,{backing+1,200+offset+first,n-first}};
    return count;
}
'''

CASES = r'''
static RcRegistryWrite declare(RcSourceRange coverage,uint64_t op){
    RcRegistryWrite ticket;OK(rc_registry_write_begin_at(r,coverage,RC_REGISTRY_BUFFER,(RcRegistryPosition){1,op},&ticket));
    return ticket;
}
static RcRegistryWrite record_update(VkExec *ex,const p_update_ub *command){
    VkExecUpdateDestination target={0};
    OK(vk_mem_span_native_input(command->dst,command->size,0,&target.span,&target.hold));
    target.count=vk_mem_span_input_intervals(target.hold,command->dst,command->size,&target.writes);
    assert(target.count==1&&target.writes);
    RcRegistryWrite actual=declare(target.writes[0].range,ex->resource_op*2+1);
    upd_note(ex,command,target.span,&target);
    vk_mem_span_capture_release(&target.hold);
    return actual;
}
static void partial_updates(void){
    const uint64_t base=0x1000,bytes=65536;
    VkExec ex={1,376};
    r=rc_registry_create((RcRegistryConfig){16,16,2});assert(r);
    upd_begin(&ex);OK(rc_registry_produced_scene_begin(r,1));UPD.watched=1;
    const p_update_ub earlier={.dst=base+0x20,.size=0x40};
    RcRegistryWrite first=record_update(&ex,&earlier);
    ex.resource_op=380;position=(RcRegistryPosition){1,760};
    OK(rc_registry_declare_through(r,(RcRegistryPosition){1,760}));
    VkMemSpan span={0};
    VkMemSpanCapture *hold=NULL;
    OK(vk_mem_span_native_input(base,bytes,0,&span,&hold));
    VkExecUniformUpdateFragment fragments[4];
    uint32_t count=UINT32_MAX;
    assert(vk_exec_uniform_update_native(&ex,base,bytes,&span,hold)==VK_EXEC_UNIFORM_UPDATE_STALE);
    OK(vk_exec_uniform_update_fragments(&ex,base,bytes,&span,hold,position,fragments,4,&count));
    assert(count==1&&fragments[0].offset==0x20&&fragments[0].bytes==0x40&&fragments[0].data==earlier.data&&
        fragments[0].output.scene==1&&fragments[0].output.ordinal==753);
    assert(vk_exec_uniform_update_fragments(&ex,base,bytes,&span,hold,position,NULL,0,&count)==RC_REGISTRY_CAPACITY&&!count);
    assert(vk_exec_uniform_update_fragments(&ex,base,bytes,&span,hold,(RcRegistryPosition){1,759},fragments,4,&count)==
        RC_REGISTRY_INVALID&&!count);

    ex.resource_op=381;
    const p_update_ub later={.dst=base+0x30,.size=0x10};
    RcRegistryWrite second=record_update(&ex,&later);
    RcRegistryWrite blocked=declare((RcSourceRange){4,200+0x24,2},765);
    RcRegistryWrite disjoint=declare((RcSourceRange){8,0,8},767);
    OK(rc_registry_declare_through(r,(RcRegistryPosition){1,770}));
    position.ordinal=764;
    OK(vk_exec_uniform_update_fragments(&ex,base,bytes,&span,hold,position,fragments,4,&count));
    assert(count==3&&fragments[0].offset==0x20&&fragments[0].bytes==0x10&&fragments[0].data==earlier.data&&
        fragments[1].offset==0x30&&fragments[1].bytes==0x10&&fragments[1].data==later.data&&
        fragments[2].offset==0x40&&fragments[2].bytes==0x20&&fragments[2].data==earlier.data+0x20);
    assert(vk_exec_uniform_update_fragments(&ex,base,bytes,&span,hold,position,fragments,2,&count)==RC_REGISTRY_CAPACITY&&!count);
    position.ordinal=768;
    assert(vk_exec_uniform_update_fragments(&ex,base,bytes,&span,hold,position,fragments,4,&count)==RC_REGISTRY_BLOCKED&&!count);
    OK(rc_registry_write_cancel(r,blocked));
    OK(rc_registry_handoff_through(r,(RcRegistryPosition){1,767}));
    OK(rc_registry_write_submit(r,first,1));OK(rc_registry_write_complete(r,first,1));
    OK(rc_registry_write_submit(r,second,1));OK(rc_registry_write_complete(r,second,1));
    OK(rc_registry_write_submit(r,disjoint,1));OK(rc_registry_write_complete(r,disjoint,1));
    OK(vk_exec_uniform_update_fragments(&ex,base,bytes,&span,hold,position,fragments,4,&count));
    assert(count==3&&fragments[1].data==later.data);
    incarnation++;
    assert(vk_exec_uniform_update_fragments(&ex,base,bytes,&span,hold,position,fragments,4,&count)==RC_REGISTRY_STALE&&!count);
    vk_mem_span_capture_release(&hold);
    const uint64_t old=backing;
    backing=91;
    OK(vk_mem_span_native_input(base,bytes,0,&span,&hold));
    OK(vk_exec_uniform_update_fragments(&ex,base,bytes,&span,hold,position,fragments,4,&count));
    assert(!count); // New actual backing at the same VA has no inline content to transplant.
    vk_mem_span_capture_release(&hold);
    backing=old;
    vk_exec_draw_update_history_end(1);
    rc_registry_destroy(r);
}
int main(void){
    r=rc_registry_create((RcRegistryConfig){16,16,2});assert(r);
    VkExec ex={1,2};p_update_ub command={.dst=0x1000,.size=16};
    for(unsigned i=0;i<16;i++)command.data[i]=(uint8_t)(0x61+i);
    VkExecUpdateDestination target={0};
    OK(vk_mem_span_native_input(command.dst,command.size,0,&target.span,&target.hold));
    target.count=vk_mem_span_input_intervals(target.hold,command.dst,command.size,&target.writes);
    assert(target.count==2);
    RcRegistryWrite emitted[2]={declare(target.writes[0].range,5),declare(target.writes[1].range,5)};
    RcRegistryWrite interferes=declare((RcSourceRange){4,210,2},9);
    RcRegistryWrite disjoint=declare((RcSourceRange){3,500,4},7);
    RcRegistryWrite future=declare((RcSourceRange){3,100,4},11);
    OK(rc_registry_declare_through(r,(RcRegistryPosition){1,12}));
    upd_begin(&ex);OK(rc_registry_produced_scene_begin(r,1));UPD.watched=1;
    upd_note(&ex,&command,target.span,&target);
    assert(!target.writes);
    vk_mem_span_capture_release(&target.hold);
    uint8_t bytes[16]={0};position=(RcRegistryPosition){1,6};
    assert(upd_read_uniform(&ex,0x2004,8,bytes)==VK_EXEC_UNIFORM_UPDATE_READY && !memcmp(bytes,command.data+4,8));
    VkMemSpan bound;
    VkMemSpanCapture *held=NULL;
    OK(vk_mem_span_native_input(0x2000,16,0,&bound,&held));
    assert(vk_exec_uniform_update_native(&ex,0x2000,16,&bound,held)==VK_EXEC_UNIFORM_UPDATE_READY);
    VkMemSpan wrong=bound;wrong.buffer++;
    assert(vk_exec_uniform_update_native(&ex,0x2000,16,&wrong,held)==VK_EXEC_UNIFORM_UPDATE_STALE);
    vk_mem_span_capture_release(&held);
    OK(vk_mem_span_native_input(0x2000,20,0,&bound,&held));
    assert(vk_exec_uniform_update_native(&ex,0x2000,20,&bound,held)==VK_EXEC_UNIFORM_UPDATE_STALE);
    vk_mem_span_capture_release(&held);
    assert(upd_read_uniform(&ex,0x2004,16,bytes)==VK_EXEC_UNIFORM_UPDATE_STALE); // Only a prefix was updated.
    assert(upd_read_uniform(&ex,0x3020,8,bytes)==VK_EXEC_UNIFORM_UPDATE_NONE);
    alias_buffer=13;assert(upd_read_uniform(&ex,0x2004,8,bytes)==VK_EXEC_UNIFORM_UPDATE_STALE); // Separate mirror, same guest backing.
    alias_buffer=7;
    uint64_t old=backing;backing=30;incarnation++;
    assert(upd_read_uniform(&ex,0x1004,8,bytes)==VK_EXEC_UNIFORM_UPDATE_NONE); // Proven new backing at the old address.
    backing=old;
    unavailable=1;assert(upd_read_uniform(&ex,0x1004,8,bytes)==VK_EXEC_UNIFORM_UPDATE_STALE);unavailable=0;
    mapping_race=1;assert(upd_read_uniform(&ex,0x1004,8,bytes)==VK_EXEC_UNIFORM_UPDATE_STALE);mapping_race=0;
    OK(rc_registry_handoff_through(r,(RcRegistryPosition){1,7}));
    for(unsigned i=0;i<2;i++){OK(rc_registry_write_submit(r,emitted[i],1));OK(rc_registry_write_complete(r,emitted[i],1));}
    OK(rc_registry_write_submit(r,disjoint,1));OK(rc_registry_write_complete(r,disjoint,1));
    position.ordinal=8;assert(upd_read_uniform(&ex,0x2004,8,bytes)==VK_EXEC_UNIFORM_UPDATE_READY);
    OK(vk_mem_span_native_input(0x2000,16,0,&bound,&held));
    assert(vk_exec_uniform_update_native(&ex,0x2000,16,&bound,held)==VK_EXEC_UNIFORM_UPDATE_READY);
    position.ordinal=10;assert(upd_read_uniform(&ex,0x2004,8,bytes)==VK_EXEC_UNIFORM_UPDATE_STALE);
    assert(vk_exec_uniform_update_native(&ex,0x2000,16,&bound,held)==VK_EXEC_UNIFORM_UPDATE_STALE);
    OK(rc_registry_write_cancel(r,interferes));
    assert(upd_read_uniform(&ex,0x2004,8,bytes)==VK_EXEC_UNIFORM_UPDATE_READY);
    assert(vk_exec_uniform_update_native(&ex,0x2000,16,&bound,held)==VK_EXEC_UNIFORM_UPDATE_READY);
    vk_mem_span_capture_release(&held);
    OK(rc_registry_write_cancel(r,future));
    vk_exec_draw_update_history_end(1);
    assert(!UPD.n&&!UPD.fence&&!UPD.u[0].writes&&!r->produced_scene);
    assert(upd_read_uniform(&ex,0x2004,8,bytes)==VK_EXEC_UNIFORM_UPDATE_NONE);
    rc_registry_destroy(r);
    partial_updates();
    puts("inline uniform producer: exact payload, normalized aliases, remap, interference and scene retirement passed");
}
'''

cc = Path(os.environ['CC'])
env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get('PATH', ''))
with tempfile.TemporaryDirectory(prefix='inline-uniform-backing-') as temp:
    c = Path(temp) / 'check.c'
    c.write_text(PREFIX + source[start:end] + CASES)
    exe = c.with_suffix('.exe')
    subprocess.run([str(cc), '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror', '-fsanitize=undefined',
                    '-fsanitize-trap=all', '-I' + str(args.runtime), '-I' + str(ROOT / 'runtime'), str(c), '-o', str(exe)],
                   env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

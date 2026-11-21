"""Check actual effect-storage capture against mirror-selection and emitted-write ownership."""
from pathlib import Path
import os, re, subprocess, tempfile
from bind_recipe_check import function

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ["CC"])
source = (ROOT / "runtime/vk_exec_draw.c").read_text()
host_write = (ROOT / "runtime/resource_host_write.c").read_text()
slice_at = host_write.index("uint32_t resource_host_interval_slice(")
slice_end = host_write.index("\n}", slice_at) + 2
def extract(name):
    match = re.search(r"^static [^\n]*\b" + name + r"\([^;{]*\)\s*\{", source, re.M)
    assert match, name
    return function(source[match.start():], name)

prefix = r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "resource_core_registry.h"
typedef struct { uint64_t buffer,offset,range; } VkDescriptorBufferInfo;
typedef struct { uint64_t buffer,offset,avail; void *pool; } VkMemSpan;
typedef struct { uint64_t addr,size; } p_bind_buf;
typedef struct { uint32_t set,binding,cbuf_slot,count,flags,ptr; } GfxShaderRes;
typedef struct {VkMemSpan span;unsigned mapping;} VkMemSpanCapture;
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
    int prepared;
} DeviceStoragePointer;
typedef struct {
    uint32_t set,binding;
    uint64_t addr,size,authority_size;
    VkDescriptorBufferInfo descriptor;
    VkMemSpan span;
    const GfxShaderRes *resource;
    VkMemSpanCapture *native;
    RcInterval *mapped;
    uint32_t mapped_count;
    int immutable,written,visible,recorded,borrowed_writer,writer_noted;
} EffectStorageCapture;
typedef struct { EffectStorageCapture storage; } SelectedStorageWriter;
typedef struct { RcInterval interval; uint64_t offset; } SelectedStorageStore;
typedef struct { SelectedStorageStore *stores; uint32_t store_count; } SelectedStorageWriters;
#define selected_storage_writer(...) ((SelectedStorageWriter *)0)
static void rt_fatal(const char *message){fputs(message, stderr);abort();}
typedef enum {RC_EFFECT_GPA,RC_EFFECT_SSBO,RC_EFFECT_SSBO_WHOLE} RcEffectKind;
typedef struct {RcEffectKind kind; uint32_t set,binding; uint64_t begin,size;} RcEffectRange;
typedef struct VkExec VkExec;
typedef struct {
    VkExec *ex;
    int stage,native_bindings,narrowed;
    EffectStorageCapture *storage;
    uint32_t storage_count,storage_capacity,native_storage;
    RcRegistryPosition read_position;
    const DeviceStoragePointer *pointer_sources;
    uint32_t pointer_count;
    SelectedStorageWriters *selected_writers;
} EffectInputCapture;
struct VkExec { EffectInputCapture *capture; p_bind_buf ssbo[8][16],ubo[8][16]; };
typedef struct { unsigned opcode; void *commands; } EffectDeclaration;
static EffectDeclaration effect={2,&effect}, *effect_current=&effect;
static struct { uint64_t ubo_max,sbo_align,sbo_max; } LIM={65536,16,65536};
static struct { uint64_t sbuf_align,storage_write_alignment; } vk_caps={16,16};
#define VK_GPA_CB_SLOT 64
#define MESHFB_CB_TASK 70
#define MESHFB_CB_LIST 73
typedef enum { VK_MEM_MIRROR_EXISTING, VK_MEM_MIRROR_IMPORT } VkMemMirrorInputPolicy;
static unsigned mirror_policy;
static unsigned selections,notes,writes,drops,resolves,native_captures,native_adoptions,native_checks,failed;
static unsigned normalized_receipts,raw_receipts,legacy_receipts,opaque_receipts;
static RcRegistryPosition position={3,8};
static uint64_t expected_buffer=222;
static unsigned live_mapping=17;
static int native_supported=1;
static uint64_t selected=222;
static uint64_t expected_write=43;
static void limits_init(void){abort();}
static int effect_binding_address(EffectInputCapture*c,const GfxShaderRes*r,p_bind_buf*b,int *uniform){
    (void)c;(void)r;*b=(p_bind_buf){0x1003,40};*uniform=0;return 1;
}
static int vk_mem_resolve(uint64_t addr,uint64_t bytes,VkMemSpan*sp){
    assert(addr==0x1000&&bytes==1);resolves++;*sp=(VkMemSpan){111,32,4096,(void*)1};return 1;
}
static int ssbo_slot(const GfxShaderRes *r,int st,int *uniform){
    assert(st==2);*uniform=0;return r->cbuf_slot==0?0:-1;
}
static RcRegistryResult vk_mem_span_native_input(uint64_t a,uint64_t size,int clamp,VkMemSpan *span,VkMemSpanCapture **p){
    assert(a==0x1000&&clamp&&!*p&&(size==43||size==48||size==65536));
    if (!native_supported)return RC_REGISTRY_UNSUPPORTED;
    *span=(VkMemSpan){selected,32,size>48?48:size,(void*)1};
    *p=malloc(sizeof **p);assert(*p);
    **p=(VkMemSpanCapture){*span,live_mapping};
    native_captures++;
    return RC_REGISTRY_OK;
}
static RcRegistryResult vk_mem_span_prepare_native_input(uint64_t a,uint64_t size,int clamp,
                                                  VkMemMirrorInputPolicy policy,VkMemSpan *span,VkMemSpanCapture **p){
    assert(policy == VK_MEM_MIRROR_IMPORT);
    mirror_policy++;
    return vk_mem_span_native_input(a,size,clamp,span,p);
}
static uint32_t vk_mem_span_input_intervals(const VkMemSpanCapture *p,uint64_t a,uint64_t size,RcInterval **out){
    assert(p&&p->span.avail==size&&a==0x1000);
    *out=calloc(2,sizeof **out);assert(*out);
    (*out)[0]=(RcInterval){a,p->mapping,{14,256,20}};
    (*out)[1]=(RcInterval){a+20,p->mapping+1,{15,128,size-20}};
    return 2;
}
enum { VK_MEM_FIXED_WRITER_PARTS=68 };
static uint32_t vk_mem_span_buffer_fixed_partitions(const VkMemSpanCapture *p,const VkMemSpan *span,
    const RcInterval *in,uint32_t count,RcInterval *out,uint32_t capacity) {
    assert(p&&span&&in&&count&&out&&capacity==VK_MEM_FIXED_WRITER_PARTS);
    return 0; // Host imports use the unchanged whole or already patterned declarations.
}
static int vk_mem_span_input_cpu_visible(const VkMemSpanCapture *p){return p->span.buffer==111;}
static int vk_mem_span_record_input(const VkMemSpanCapture *p,const VkMemSpan *span){
    assert(p&&p->span.buffer==span->buffer);native_checks++;return p->mapping==live_mapping;
}
static void vk_mem_span_capture_release(VkMemSpanCapture **p){free(*p);*p=NULL;}
static VkMemSpanCapture *vk_mem_span_capture_retain(VkMemSpanCapture *p){(void)p;abort();}
static void vk_exec_record_native_input_provider(VkExec *ex,VkMemSpanCapture **p){
    assert(ex&&*p);native_adoptions++;vk_mem_span_capture_release(p);
}
static RcRegistryPosition vk_exec_resource_position(void){return position;}
static void vk_exec_required_graphics_unavailable(VkExec *ex,uint32_t opcode,const char *reason){
    assert(ex&&opcode==2&&reason);failed++;
}
static void choose(VkMemSpan*sp){assert(sp->buffer==111);selections++;sp->buffer=selected;}
#define VK_MEM_USE(sp) choose(&(sp))
static EffectInputCapture *effect_scope_capture(VkExec*ex,int stage){assert(stage==2);return ex->capture;}
static int note_written(VkExec*ex,uint64_t addr,uint64_t size,const VkMemSpan*sp){(void)ex;assert(addr==0x1000&&size==expected_write&&sp->buffer==expected_buffer);notes++;return 1;}
static int prepared_buffer_requires_image(const VkMemSpan *span,uint64_t address,uint64_t size,RcRegistryPosition read) {
    assert(span&&address&&size&&!rc_registry_position_compare(read,position));
    return 0;
}
static void buffer_preserve_import(const EffectStorageCapture *v) {(void)v;abort();}
static int note_buffer_written(VkExec*ex,uint64_t addr,uint64_t size,const VkMemSpan*sp,int visible,
                                const RcInterval *mapped,uint32_t count){
    assert(mapped&&count==2&&mapped[0].range.backing==14&&mapped[0].mapping==17&&mapped[1].mapping==18);
    assert(!visible);
    return note_written(ex,addr,size,sp);
}
static void vk_mem_gpu_wrote_shader(const VkMemSpan*sp,uint64_t size){assert(sp->buffer==expected_buffer&&size==expected_write);writes++;}
static void vk_usnap_drop(uint64_t addr,uint64_t size){assert(addr==0x1000&&size==expected_write);drops++;}
static void effect_declare_host(EffectDeclaration *d,const VkMemSpan *span,uint64_t a,uint64_t n,
                                uint64_t write,uint64_t size,RcRegistryWriterKind kind){
    (void)d;(void)span;(void)a;(void)n;(void)write;(void)size;(void)kind;legacy_receipts++;
}
static int resource_effects_add(void *commands,RcSourceRange written,RcSourceRange descriptor,RcRegistryWriterKind kind){
    assert(commands==&effect&&kind==RC_REGISTRY_BUFFER&&written.backing==1&&descriptor.backing==1);
    assert(descriptor.offset==0x1000&&descriptor.size==48&&written.size<=48);
    raw_receipts++;return 1;
}
static void resource_effects_empty(void *commands,RcSourceRange descriptor){
    assert(commands==&effect&&descriptor.backing==1&&descriptor.offset==0x1000&&descriptor.size==48);raw_receipts++;
}
static RcRegistryResult resource_effects_add_intervals(void *commands,const RcInterval *written,uint32_t n,
                              const RcInterval *descriptor,uint32_t count,RcRegistryWriterKind kind){
    assert(commands==&effect&&kind==RC_REGISTRY_BUFFER&&descriptor&&count==2);
    assert(descriptor[0].mapping==17&&descriptor[0].range.backing==14&&descriptor[1].range.backing==15);
    if(n){
        assert(n==2&&written[0].mapping==17&&written[1].mapping==18);
        assert(written[0].range.size+written[1].range.size==expected_write);
    }
    normalized_receipts++;return RC_REGISTRY_OK;
}
static RcRegistryResult resource_effects_opaque_intervals(void *commands,RcRegistryWrite *out){
    assert(commands==&effect&&!out);opaque_receipts++;return RC_REGISTRY_OK;
}
RcRegistry *rc_registry_runtime(void){return NULL;}
static int vk_mem_span_capture_current(const VkMemSpanCapture *c, const VkMemSpan *span){
    return c->mapping==live_mapping&&c->span.buffer==span->buffer;
}
static int pointer_proof=1;
RcRegistryResult rc_registry_read_device_buffer_at(RcRegistry *r,const RcInterval *a,uint32_t count,RcRegistryPosition read){
    assert(!r&&a&&count==1&&a->range.size==8&&!rc_registry_position_compare(position,read));
    return pointer_proof?RC_REGISTRY_OK:RC_REGISTRY_BLOCKED;
}
void rc_registry_fail_closed(RcRegistry *registry){(void)registry;abort();}
'''
body = host_write[slice_at:slice_end] + "\n" + "\n".join(extract(name) for name in (
    "device_storage_pointer", "device_storage_pointer_current", "device_storage_pointer_value_current",
    "readonly_storage_pointer",
    "effect_storage_find", "effect_storage_fixed", "effect_storage_new",
    "effect_storage_capture_span", "effect_storage_move", "effect_storage_native",
    "effect_storage_bind", "effect_storage_descriptor", "effect_storage_whole", "effect_output_ranges_valid",
    "effect_declare_storage_host", "effect_declare_storage_output"))
tests = r'''
int main(void){
 EffectInputCapture c={.narrowed=1,.stage=2};VkExec ex={.capture=&c};
 GfxShaderRes r={.binding=3,.count=1,.flags=1};c.ex=&ex;
 EffectStorageCapture*v=effect_storage_bind(&c,&r);
 assert(v&&v->descriptor.buffer==222&&v->span.buffer==222&&v->descriptor.offset==32&&v->descriptor.range==43);
 assert(effect_storage_bind(&c,&r)==v&&selections==1);
 VkDescriptorBufferInfo actual={0};assert(effect_storage_descriptor(&ex,2,&r,1,&actual));
 assert(actual.buffer==222&&notes==1&&writes==1&&drops==1);
 RcEffectRange range={RC_EFFECT_SSBO_WHOLE,0,3,0,0};
 assert(effect_output_ranges_valid(&c,&range,1)&&range.kind==RC_EFFECT_SSBO&&range.size==48);
 assert(v->authority_size==48&&v->descriptor.range==43);
 expected_write=48;assert(effect_storage_descriptor(&ex,2,&r,1,&actual));assert(actual.range==43);
 VkMemSpanCapture table={.span={.buffer=666},.mapping=17};
 RcInterval table_range={0x7000,17,{83,0,8}};
 DeviceStoragePointer pointer={.resource=&r,.span=table.span,.capture=&table,.ranges=&table_range,
                              .read=position,.value=0x1003,.count=1,.stage=2,
                              .prepared=1,.binding={0x7000,8}};
 ex.ssbo[2][0]=pointer.binding;
 c.pointer_sources=&pointer;c.pointer_count=1;
 assert(effect_storage_descriptor(&ex,2,&r,1,&actual)&&actual.buffer==222);
 unsigned pointer_notes=notes;
 pointer_proof=0;
 assert(effect_storage_descriptor(&ex,2,&r,1,&actual)&&!actual.buffer&&failed==1&&notes==pointer_notes);
 pointer_proof=1;live_mapping++;
 assert(effect_storage_descriptor(&ex,2,&r,1,&actual)&&!actual.buffer&&failed==2&&notes==pointer_notes);
 live_mapping--;ex.ssbo[2][0].addr+=8;
 assert(effect_storage_descriptor(&ex,2,&r,1,&actual)&&!actual.buffer&&failed==3&&notes==pointer_notes);
 ex.ssbo[2][0]=pointer.binding;failed=0;c.pointer_sources=NULL;c.pointer_count=0;
 uint64_t extent=0;vk_caps.storage_write_alignment=0;assert(!effect_storage_whole(v,&extent));
 vk_caps.storage_write_alignment=16;v->span.avail=47;assert(!effect_storage_whole(v,&extent));v->span.avail=4096;
 v->descriptor.offset++;assert(!effect_storage_whole(v,&extent));v->descriptor.offset--;
 v->descriptor.buffer++;assert(!effect_storage_whole(v,&extent));v->descriptor.buffer--;
 v->written=0;assert(!effect_storage_whole(v,&extent));v->written=1;
 range=(RcEffectRange){RC_EFFECT_SSBO_WHOLE,0,4,0,0};assert(!effect_output_ranges_valid(&c,&range,1));
 GfxShaderRes gpa={.binding=4,.cbuf_slot=VK_GPA_CB_SLOT,.count=1,.flags=1};assert(!effect_storage_bind(&c,&gpa));
 range=(RcEffectRange){99,0,3,0,4};assert(!effect_output_ranges_valid(&c,&range,1));
 c.storage[1]=(EffectStorageCapture){.addr=0x102e,.descriptor={.range=4},.immutable=1};c.storage_count=2;
 range=(RcEffectRange){RC_EFFECT_SSBO_WHOLE,0,3,0,0};assert(!effect_output_ranges_valid(&c,&range,1));
 free(c.storage);c=(EffectInputCapture){.stage=2,.ex=&ex};selected=111;
 v=effect_storage_bind(&c,&r);assert(v&&v->descriptor.buffer==111&&v->span.buffer==111);
 free(c.storage);
 c=(EffectInputCapture){.stage=2,.native_bindings=1,.ex=&ex,.read_position=position};
 ex.ssbo[2][0]=(p_bind_buf){0x1003,40};
 selected=222;expected_buffer=222;expected_write=48;
 unsigned old_resolves=resolves;
 v=effect_storage_bind(&c,&r);
 assert(v&&v->native&&c.native_storage==1&&v->descriptor.buffer==222&&v->descriptor.range==43);
 assert(v->authority_size==48&&v->mapped_count==2&&native_captures==1&&resolves==old_resolves);
 effect_declare_storage_output(&effect,v,0,0);
 effect_declare_storage_output(&effect,v,0x1000,48);
 assert(raw_receipts==2&&normalized_receipts==2&&!legacy_receipts&&!opaque_receipts);
 selected=333; // A second VA lookup and ordinary mirror selection would now give the wrong provider.
 assert(effect_storage_descriptor(&ex,2,&r,1,&actual)&&actual.buffer==222&&actual.range==43);
 assert(native_adoptions==1&&native_checks==1&&!v->native&&resolves==old_resolves);
 assert(effect_storage_descriptor(&ex,2,&r,1,&actual)&&native_checks==1&&native_adoptions==1);
 unsigned old_notes=notes;
 RcInterval *mapping=v->mapped;
 ++position.ordinal;
 assert(effect_storage_descriptor(&ex,2,&r,1,&actual)&&!actual.buffer&&failed==1&&notes==old_notes);
 position=c.read_position;
 free(mapping);free(c.storage);

 c=(EffectInputCapture){.stage=2,.native_bindings=1,.ex=&ex,.read_position=position};
 GfxShaderRes read={.binding=5,.count=1};
 v=effect_storage_native(&c,&read);
 assert(v&&v->authority_size==43&&v->descriptor.range==43&&v->mapped_count==2);
 assert(effect_storage_descriptor(&ex,2,&read,0,&actual)&&actual.buffer==333&&notes==old_notes);
 assert(native_adoptions==2&&resolves==old_resolves);
 free(v->mapped);free(c.storage);
 c=(EffectInputCapture){.stage=2,.native_bindings=1,.ex=&ex,.read_position=position};
 v=effect_storage_native(&c,&r);assert(v);
 live_mapping++;
 assert(effect_storage_descriptor(&ex,2,&r,1,&actual)&&!actual.buffer&&failed==2&&notes==old_notes);
 assert(native_adoptions==2&&resolves==old_resolves&&v->native);
 vk_mem_span_capture_release(&v->native);
 free(v->mapped);free(c.storage);
 c=(EffectInputCapture){.stage=2,.native_bindings=1,.ex=&ex,.read_position=position};
 r.ptr=1;assert(!effect_storage_native(&c,&r));
 r.ptr=0;ex.ssbo[2][0].size=0;assert(!effect_storage_native(&c,&r));
 ex.ssbo[2][0].size=40;native_supported=0;assert(!effect_storage_native(&c,&r));
 assert(!c.storage&&resolves==old_resolves);
 native_supported=1;live_mapping=17;selected=333;expected_buffer=333;
 r.ptr=1;
 EffectStorageCapture pointed={0};
 vk_caps.storage_write_alignment=0;
 assert(!effect_storage_capture_span(&r,pointer.value,0,VK_MEM_MIRROR_IMPORT,&pointed));
 assert(!mirror_policy && !pointed.native);
 vk_caps.storage_write_alignment=16;
 assert(effect_storage_capture_span(&r,pointer.value,0,VK_MEM_MIRROR_IMPORT,&pointed));
 assert(mirror_policy==1&&pointed.mapped_count==2&&pointed.descriptor.buffer==333);
 assert(pointed.descriptor.range==48&&pointed.authority_size==48);
 ex.ssbo[2][0]=pointer.binding;
 c.native_bindings=0;c.pointer_sources=&pointer;c.pointer_count=1;
 assert(!effect_storage_bind(&c,&r)&&resolves==old_resolves);
 v=effect_storage_move(&c,&pointed);
 assert(v&&c.native_storage==1&&!pointed.native&&!pointed.mapped);
 assert(effect_storage_bind(&c,&r)==v);
 effect_declare_storage_output(&effect,v,v->addr,v->authority_size);
 assert(raw_receipts==3&&normalized_receipts==3&&!opaque_receipts);
 selected=999; // Changed legacy resolver would find another provider; the actual binding cannot consult it.
 assert(effect_storage_descriptor(&ex,2,&r,1,&actual)&&actual.buffer==333&&actual.range==48);
 assert(native_adoptions==3&&resolves==old_resolves);
 free(v->mapped);free(c.storage);
 puts("effect storage: exact native descriptor, finite robust output, retained device-pointer target, read-only and legacy paths passed");
}
'''
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="effect-storage-", dir=ROOT / "build/test") as tmp:
    path = Path(tmp) / "check.c"
    path.write_text(prefix + body + tests)
    exe = path.with_suffix(".exe")
    subprocess.run([str(CC), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fsanitize-trap=all", "-I" + str(ROOT / "runtime"), str(path), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

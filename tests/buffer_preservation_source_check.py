"""Check native possible-write predecessor custody against actual backing adapter logic."""
from pathlib import Path
import os
import subprocess
import tempfile

from bind_recipe_check import function

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ["CC"])
SOURCE = (ROOT / 'runtime/vk_memory.c').read_text(encoding='utf-8')

PREFIX = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "resource_core_registry.h"

enum { POOL_PHYS=1,POOL_IMPORT=2 };
typedef struct {uint64_t lo,hi;} FlushRange;
typedef struct {uint64_t offset,buffer,alloc;} VkMemSpan;
typedef struct {uint64_t address,size; RcRegistryPosition output;} VkMemBufferInlineProof;
typedef struct {uint64_t srcOffset,dstOffset,size;} VkBufferCopy;
typedef struct {
    int kind,smap,gpa_host_coherent,acc_sub,physical_origin_unknown;
    uint64_t publication,physical_origin_upload,physical_origin_bytes;
    uint64_t import_lo,import_hi,buf,mbuf;
    uint64_t mirror_origin_checked_scene,mirror_origin_full_queued,mirror_origin_full_copy,mirror_origin_last_copy;
    uint64_t mirror_origin_queued;
    uint64_t import_writer_scene,import_writer_fence,mirror_import_only_scene;
    unsigned import_writer_refs;
    unsigned physical_origin_poison_count;
    FlushRange physical_origin_poison[2];
    const RcInterval *host_backing;
} PoolRec;
typedef struct {
    int native_input,cold_writer,writer_published,input_recorded;
    unsigned device_count;
    uint64_t address,size,mirror_incarnation,import_mirror_scene,writer_issued_scene;
    VkMemSpan selected;
    PoolRec *provider,*window;
    RcInterval *device_intervals,*host;
} VkMemSpanCapture;
typedef uint64_t VkCommandBuffer;
typedef struct {PoolRec *r; uint64_t off,size;} MRange;
typedef void CpuVisibility;
static VkBufferCopy transfer_regions[8],*mregions=transfer_regions;
static int cregions=8;
static MRange queued[8],*pend=queued;
static int npend,cpend=8, issued;
static void grow(void *a,int *b,size_t n){(void)a;(void)b;(void)n;abort();}
static void cpu_visibility_add(CpuVisibility *v,PoolRec *r,uint64_t offset,uint64_t size){
    (void)v;(void)r;(void)offset;(void)size;abort();
}
static void vkCmdCopyBuffer(VkCommandBuffer cb,uint64_t src,uint64_t dst,uint32_t count,const VkBufferCopy *regions) {
    assert(cb==21&&src==33&&dst==44&&count==1);
    assert(regions[0].srcOffset==regions[0].dstOffset&&regions[0].size);
    issued++;
}
static RcRegistry *fixture_registry;
static int pool_lock, mapping_current=1, bind_pending;
static int msub=9;
static uint64_t submitted=7,completed=7,mirror_completed;
static uint64_t vk_sync_fence_done(void){return mirror_completed;}
static RcRegistryPosition current={1,6};
static RcRegistryPosition vk_exec_resource_position(void){return current;}
static void vk_upload_progress(uint64_t *s,uint64_t *c){*s=submitted;*c=completed;}
static void rt_fatal(const char *reason){puts(reason);abort();}
static void plat_mutex_lock(int *lock){assert(!*lock);*lock=1;}
static void plat_mutex_unlock(int *lock){assert(*lock);*lock=0;}
static int nvn_gpu_only(uint64_t address){return address>=0x8000000000ull;}
static void plat_mutex_lock_shared(int *lock){assert(!*lock);*lock=1;}
static void plat_mutex_unlock_shared(int *lock){assert(*lock);*lock=0;}
static int span_capture_match(const VkMemSpanCapture *capture,const VkMemSpan *selected){
    assert(pool_lock && selected==&capture->selected);return mapping_current;
}
static RcRegistry *fixture_registry_get(void) {return fixture_registry;}
static int vk_mem_span_buffer_location(const VkMemSpanCapture *capture,const VkMemSpan *selected,
                                       RcRegistryBufferLocation *location) {
    if (selected!=&capture->selected || !mapping_current) return 0;
    *location=(RcRegistryBufferLocation){capture->provider->publication,capture->mirror_incarnation,selected->offset};
    return 1;
}
static uint32_t vk_mem_span_input_intervals(const VkMemSpanCapture *capture,uint64_t address,uint64_t bytes,RcInterval **out) {
    const RcInterval *mapping=capture->host;
    assert(mapping&&address>=mapping->address&&address-mapping->address<=mapping->range.size-bytes);
    *out=malloc(sizeof **out);assert(*out);
    **out=*mapping;
    (*out)->address=address;
    (*out)->range.offset+=address-mapping->address;
    (*out)->range.size=bytes;
    return 1;
}
static int vk_mem_span_capture_current(const VkMemSpanCapture *capture,const VkMemSpan *selected) {
    return selected==&capture->selected&&mapping_current;
}
static const RcInterval *mem_backing_intervals(const RcInterval *mapping,uint32_t *count){*count=mapping?1:0;return mapping;}
static int physical_origin_pending_overlap(const PoolRec *provider,const RcRegistryBufferSpan *spans,uint32_t count){
    assert(pool_lock&&provider&&spans&&count==1);return bind_pending;
}
'''

TEST = r'''
static void fixed_partial_storage_continuation(void) {
    const uint64_t guest=0x8000002000ull;
    const RcInterval whole={guest,31,{42,120,64}};
    PoolRec physical={.kind=POOL_PHYS,.publication=17,.physical_origin_upload=7,.physical_origin_bytes=112,
                      .physical_origin_poison_count=1,.physical_origin_poison={{80,96}}};
    VkMemSpanCapture storage={.native_input=1,.address=guest,.size=64,.selected={.offset=64,.buffer=61,.alloc=1},
        .provider=&physical,.window=&physical,.device_intervals=(RcInterval *)&whole,.device_count=1};
    RcInterval parts[6], rejected[6];
    const uint32_t count=vk_mem_span_buffer_fixed_partitions(&storage,&storage.selected,&whole,1,parts,6);
    assert(count==4);
    uint64_t total=0;
    for(uint32_t i=0;i<count;i++){
        assert(parts[i].address==guest+total && parts[i].range.offset==120+total &&
               parts[i].mapping==whole.mapping && parts[i].range.backing==whole.range.backing && parts[i].range.size==16);
        total+=parts[i].range.size;
    }
    assert(total==whole.range.size);
    assert(!vk_mem_span_buffer_fixed_partitions(&storage,&storage.selected,&whole,1,rejected,3));
    physical.physical_origin_unknown=1;
    assert(!vk_mem_span_buffer_fixed_partitions(&storage,&storage.selected,&whole,1,rejected,6));
    physical.physical_origin_unknown=0;
    mapping_current=0;
    assert(!vk_mem_span_buffer_fixed_partitions(&storage,&storage.selected,&whole,1,rejected,6));
    mapping_current=1;

    for (int disposition=0;disposition<3;disposition++) {
        RcRegistry *r=rc_registry_create((RcRegistryConfig){16,32,8});
        assert(r);
        const RcInterval input={guest-8,31,{42,112,8}};
        RcRegistryWrite held, image_write;
        assert(rc_registry_image_hold_begin_extents(r,202,&input,1,&held)==RC_REGISTRY_OK);
        const RcRegistryPosition image={1,3}, write={1,7}, read={1,8};
        assert(rc_registry_write_begin_at(r,input.range,RC_REGISTRY_IMAGE,image,&image_write)==RC_REGISTRY_OK);
        assert(rc_registry_image_producer_add(r,image_write,image)==RC_REGISTRY_OK);
        RcRegistryWriteExtent extents[6];
        RcRegistryWrite tickets[6];
        for(uint32_t i=0;i<count;i++) extents[i]=(RcRegistryWriteExtent){parts[i].range,RC_REGISTRY_BUFFER};
        assert(rc_registry_write_begin_extents(r,extents,count,write,tickets)==RC_REGISTRY_OK);
        assert(rc_registry_declare_through(r,read)==RC_REGISTRY_OK);
        const RcRegistryNativeImageWrite projected={{input,held},&input.range,1};
        assert(rc_registry_native_image_write_fragments(r,202,&projected,1,image)==RC_REGISTRY_OK);
        const RcRegistryImageBinding binding={202,held};
        assert(rc_registry_image_producer_resolve(r,image_write,image,&binding,0)==RC_REGISTRY_OK);
        assert(rc_registry_handoff_through(r,image)==RC_REGISTRY_OK);
        rc_registry_image_producer_native_completed(r,image_write,image,binding);
        assert(rc_registry_write_cancel(r,image_write)==RC_REGISTRY_OK);

        current=(RcRegistryPosition){1,6};
        for(uint32_t i=0;i<count;i++) {
            const RcRegistryBufferSpan destination={parts[i],{17,0,64+16*i}};
            int initialized;
            if (vk_mem_span_buffer_preserve_current(&storage,destination,current,&initialized)) {
                const RcRegistryResult status=rc_registry_buffer_preserved_record(r,tickets[i],destination,current,initialized);
                assert(status==(i==0||i==2?RC_REGISTRY_OK:RC_REGISTRY_BLOCKED));
            } else assert(i==1);
        }
        if(disposition==1) assert(rc_registry_write_cancel_extents(r,tickets,count)==RC_REGISTRY_OK);
        else if(disposition==2) {
            assert(rc_registry_handoff_through(r,write)==RC_REGISTRY_OK);
            assert(rc_registry_write_submit_extents(r,tickets,count,1)==RC_REGISTRY_OK);
            assert(rc_registry_write_abandon_extents(r,tickets,count)==RC_REGISTRY_OK);
        }
        else {
            assert(rc_registry_handoff_through(r,write)==RC_REGISTRY_OK);
            assert(rc_registry_write_submit_extents(r,tickets,count,1)==RC_REGISTRY_OK);
            RcRegistryWriteCompletion completed[6];
            for(uint32_t i=0;i<count;i++) completed[i]=(RcRegistryWriteCompletion){.ticket=tickets[i],
                .written=&parts[i].range,.written_count=1,.uncertain=&parts[i].range,.uncertain_count=1};
            assert(rc_registry_write_complete_extents(r,completed,count)==RC_REGISTRY_OK);
        }
        RcRegistryNativeImageSnapshotFragment mixed[5];
        RcRegistryNativeImageFailure refusal;
        uint32_t used;
        const RcInterval demand={guest-8,31,{42,112,24}};
        const RcRegistryBufferSpan base={parts[0],{17,0,64}};
        const RcRegistryResult selected=rc_registry_native_image_snapshot_mixed(r,&demand,1,&base,1,read,
            mixed,5,&used,&refusal);
        if(disposition==2) assert(selected==RC_REGISTRY_BLOCKED);
        else {
            assert(selected==RC_REGISTRY_OK);
            assert(used==2&&mixed[0].fragment.binding.image==202&&mixed[1].native_buffer.offset==64&&
                mixed[1].fragment.output.ordinal==(disposition==1?0:7));
        }
        RcRegistryBufferSpan wrong=base;
        wrong.native.incarnation++;
        if(!disposition) {
            assert(rc_registry_native_image_snapshot_mixed(r,&demand,1,&wrong,1,read,mixed,5,&used,&refusal)==RC_REGISTRY_BLOCKED);
            wrong=base; wrong.native.allocation++;
            assert(rc_registry_native_image_snapshot_mixed(r,&demand,1,&wrong,1,read,mixed,5,&used,&refusal)==RC_REGISTRY_BLOCKED);
            RcRegistryBufferSpan gap={parts[1],{17,0,80}};
            assert(rc_registry_native_image_snapshot_mixed(r,&parts[1],1,&gap,1,read,mixed,5,&used,&refusal)==RC_REGISTRY_BLOCKED);
        }
        rc_registry_destroy(r);
    }
}

static void import_result_only_before_handoff(VkMemSpanCapture *capture,RcRegistryBufferSpan output,
                                               RcRegistryPosition read,int initialized) {
    RcRegistry *r=fixture_registry=rc_registry_create((RcRegistryConfig){8,16,4});
    assert(r);
    RcRegistryGpuCapture snapshot={0};
    RcRegistryWrite command;
    RcRegistryPosition issue={read.scene,read.ordinal+1},consumer={read.scene,read.ordinal+2};
    assert(rc_registry_gpu_capture_begin(r,&output.interval.range,1,&snapshot)==RC_REGISTRY_OK);
    assert(rc_registry_write_begin_at(r,output.interval.range,RC_REGISTRY_BUFFER,issue,&command)==RC_REGISTRY_OK);
    assert(rc_registry_declare_through(r,consumer)==RC_REGISTRY_OK);
    assert(rc_registry_buffer_preserved_record(r,command,output,read,initialized)==RC_REGISTRY_OK);
    current=consumer;
    RcRegistryBufferFragment fragment;
    uint32_t count=0;
    assert(vk_mem_span_buffer_fragments(capture,&capture->selected,output.interval.address,output.interval.range.size,
        snapshot,consumer,NULL,0,&fragment,1,&count)==RC_REGISTRY_OK);
    assert(count==1&&fragment.output.scene==issue.scene&&fragment.output.ordinal==issue.ordinal);
    assert(rc_registry_handoff_through(r,issue)==RC_REGISTRY_OK);
    assert(vk_mem_span_buffer_fragments(capture,&capture->selected,output.interval.address,output.interval.range.size,
        snapshot,consumer,NULL,0,&fragment,1,&count)==RC_REGISTRY_UNSUPPORTED);
    // The result is physically coherent but its host-visible contents are now mutable.
    assert(count==1&&fragment.output.ordinal==issue.ordinal);
    assert(rc_registry_gpu_capture_end(r,&snapshot)==RC_REGISTRY_OK);
    rc_registry_destroy(r);
}

static void mirror_registry_result(RcRegistryBufferSpan output,RcRegistryPosition read,int initialized,int canceled){
    assert(initialized);
    RcRegistry *registry=rc_registry_create((RcRegistryConfig){8,16,4});
    assert(registry);
    RcRegistryGpuCapture snapshot={0};
    RcRegistryWrite command;
    RcRegistryPosition issue={read.scene,read.ordinal+1},consumer={read.scene,read.ordinal+2};
    assert(rc_registry_gpu_capture_begin(registry,&output.interval.range,1,&snapshot)==RC_REGISTRY_OK);
    assert(rc_registry_write_begin_at(registry,output.interval.range,RC_REGISTRY_BUFFER,issue,&command)==RC_REGISTRY_OK);
    assert(rc_registry_declare_through(registry,consumer)==RC_REGISTRY_OK);
    assert(rc_registry_buffer_preserved_record(registry,command,output,read,initialized)==RC_REGISTRY_OK);
    RcRegistryBufferFragment fragment;
    uint32_t count;
    assert(rc_registry_gpu_capture_buffer_fragments(registry,snapshot,&output,1,consumer,&fragment,1,&count)==RC_REGISTRY_OK);
    assert(count==1&&fragment.output.scene==issue.scene&&fragment.output.ordinal==issue.ordinal);
    if (canceled){
        assert(rc_registry_write_cancel(registry,command)==RC_REGISTRY_OK);
        assert(rc_registry_gpu_capture_buffer_fragments(registry,snapshot,&output,1,consumer,&fragment,1,&count)==RC_REGISTRY_OK);
        assert(!count);
    }
    assert(rc_registry_gpu_capture_end(registry,&snapshot)==RC_REGISTRY_OK);
    rc_registry_destroy(registry);
}

int main(void){
    fixed_partial_storage_continuation();
    PoolRec fixed={.kind=POOL_PHYS,.publication=11,.physical_origin_upload=7,.physical_origin_bytes=256};
    RcInterval raw={0x8000000020ull,21,{31,32,128}};
    VkMemSpanCapture selected={.native_input=1,.device_count=1,.address=raw.address,.size=128,
        .selected={.offset=32},.provider=&fixed,.device_intervals=&raw};
    RcRegistryBufferSpan requested={{raw.address+8,raw.mapping,{31,40,16}},{11,0,40}};
    int initialized=-1;
    assert(vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&initialized&&!pool_lock);
    completed=6;
    assert(vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&!initialized);
    completed=7;bind_pending=1;
    assert(!vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&!initialized);
    bind_pending=0;fixed.physical_origin_poison_count=1;fixed.physical_origin_poison[0]=(FlushRange){45,60};
    assert(!vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&!initialized);
    fixed.physical_origin_poison_count=0;fixed.physical_origin_unknown=1;
    assert(!vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&!initialized);
    fixed.physical_origin_unknown=0;mapping_current=0;
    assert(!vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&!initialized);
    mapping_current=1;requested.interval.mapping++;
    assert(!vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized));
    requested.interval.mapping--;requested.native.offset++;
    assert(!vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized));
    requested.native.offset--;
    assert(!vk_mem_span_buffer_preserve_current(&selected,requested,(RcRegistryPosition){1,4},&initialized));

    PoolRec mirror={.kind=POOL_IMPORT,.publication=12,.gpa_host_coherent=1,.acc_sub=9,
        .import_lo=0x1fe0,.import_hi=0x20e0,.buf=33,.mbuf=44};
    RcInterval import={0x2000,22,{32,32,128}};
    selected=(VkMemSpanCapture){.native_input=1,.address=import.address,.size=128,.mirror_incarnation=13,
        .selected={.offset=32},.provider=&mirror,.host=&import};
    requested=(RcRegistryBufferSpan){{import.address+4,import.mapping,{32,36,16}},{12,13,36}};
    current.scene=3;
    mirror.mirror_origin_checked_scene=current.scene;
    assert(vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&!initialized);
    mirror_queue_upload(&mirror,0,256,current.scene);
    assert(npend==1&&pend[0].r==&mirror&&pend[0].size==256);
    mirror.mirror_origin_full_queued=current.scene;
    assert(vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&initialized);
    assert(!issued&&!mirror_completed&&!mirror.mirror_origin_full_copy);
    mirror_registry_result(requested,current,initialized,1); // recorded body never handed with its prologue

    // Ordinary retained mirror captures can borrow the whole import's mapping instead of owning a private capture.
    RcInterval whole_import={mirror.import_lo,import.mapping,{import.range.backing,0,256}};
    selected.host=NULL;
    assert(!vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&!initialized);
    mirror.host_backing=&whole_import;
    assert(vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&initialized);
    mirror_registry_result(requested,current,initialized,1);
    whole_import.mapping++;
    assert(!vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&!initialized);
    selected.host=&import; // a capture's own mapping is authoritative when present
    assert(vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&initialized);
    whole_import.mapping--;
    import.mapping++;
    assert(!vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&!initialized);
    import.mapping--;
    selected.host=NULL;
    uint64_t bytes=0,copies=0;
    assert(copy_ranges(21,pend,npend,1,&bytes,&copies,NULL,current.scene)==1);
    npend=0;
    assert(bytes==256&&copies==1&&issued==1&&mirror.mirror_origin_full_copy==3&&!mirror.mirror_origin_full_queued);
    assert(vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&initialized);
    mirror_registry_result(requested,current,initialized,0); // same scene, prefix already precedes later body
    mirror.host_backing=NULL;
    selected.host=&import;
    current.scene++;
    mirror.mirror_origin_checked_scene=current.scene;
    assert(vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&!initialized);
    mirror_completed=3;
    assert(vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&initialized);
    assert(!mirror_origin_initialized(&mirror,current.scene,mirror_completed,250,16));
    mirror_queue_upload(&mirror,32,16,current.scene+1); // other-scene or uncertain upload placement
    assert(vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&!initialized);
    npend=0;
    mirror_queue_upload(&mirror,32,16,current.scene); // ordered partial refresh of initialized storage
    assert(vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&initialized);
    assert(copy_ranges(21,pend,npend,1,&bytes,&copies,NULL,current.scene)==1);
    npend=0;
    assert(vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&initialized);
    ++current.scene;
    mirror.mirror_origin_checked_scene=current.scene;
    assert(vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&!initialized);
    mirror_completed=4;
    assert(vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&initialized);
    mirror.mirror_origin_checked_scene=0;
    assert(vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&!initialized);
    mirror.acc_sub--;
    assert(!vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized));
    mirror.acc_sub++;requested.native.incarnation--;
    assert(!vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized));
    requested.native.incarnation++;mirror.smap=1;
    assert(!vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized));
    mirror.smap=0;
    selected.mirror_incarnation=0;
    selected.selected.buffer=mirror.buf;
    requested.native.incarnation=0;
    assert(!vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized));
    selected.cold_writer=selected.writer_published=selected.input_recorded=1;
    selected.import_mirror_scene=mirror.import_writer_scene=19;
    selected.window=&mirror;
    mirror.import_writer_refs=1;
    assert(!vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized));
    import_writer_release(&selected,0); // abandoned binding has no actual dispatch
    assert(!selected.writer_published&&!selected.writer_issued_scene&&!mirror.import_writer_refs);
    assert(!vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized));
    selected.writer_published=1;
    mirror.import_writer_refs=1;
    vk_mem_span_import_writer_issued(&selected,current.scene);
    assert(!selected.writer_published&&selected.writer_issued_scene==current.scene&&!mirror.import_writer_refs);
    assert(mirror.import_writer_fence==current.scene&&mirror.mirror_import_only_scene==19);
    assert(vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized)&&initialized);
    RcRegistryPosition original=current;
    current.scene++;
    assert(!vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized));
    current=original;
    selected.input_recorded=0;
    assert(!vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized));
    selected.input_recorded=1;selected.selected.buffer=mirror.mbuf;
    assert(!vk_mem_span_buffer_preserve_current(&selected,requested,current,&initialized));
    selected.selected.buffer=mirror.buf;
    import_result_only_before_handoff(&selected,requested,current,1);
    assert(!pool_lock);
    puts("buffer preservation source: exact mutable import before handoff, post-handoff refusal, remap, fixed and mirror covers passed");
    puts("buffer preservation source: fixed origin, paired mirror prologue, completed mirror, ordered refresh, known-unhanded and stale selection passed");
}
'''

with tempfile.TemporaryDirectory(prefix='buffer-preservation-source-') as directory:
    c = Path(directory) / 'check.c'
    routines = ('range_add', 'range_cmp', 'mirror_queue_upload', 'mirror_record_upload',
                'mirror_origin_initialized', 'copy_ranges', 'physical_initial_fill_covers',
                'import_writer_release', 'vk_mem_span_import_writer_issued',
                'fixed_buffer_part_end', 'vk_mem_span_buffer_fixed_partitions',
                'vk_mem_span_buffer_preserve_current', 'vk_mem_span_buffer_fragments_with_payloads',
                'vk_mem_span_buffer_fragments')
    c.write_text(PREFIX + '\n'.join(function(SOURCE, routine).replace('rc_registry_runtime()', 'fixture_registry_get()') for routine in routines) + TEST, encoding='utf-8')
    exe = c.with_suffix('.exe')
    env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get('PATH', ''))
    subprocess.run([str(CC), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror', '-fsanitize=undefined',
                    '-fsanitize-trap=all', '-I', str(ROOT / 'runtime'), str(c),
                    str(ROOT / 'runtime/resource_core_registry.c'), '-DRC_REGISTRY_SINGLE_THREADED=1',
                    '-o', str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

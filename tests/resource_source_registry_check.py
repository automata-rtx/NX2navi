"""Exercise the production texture/source authority bridge, including GPU stores invisible to write-watch."""
from pathlib import Path
import os
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ["CC"])
source = (ROOT / "runtime/vk_texture.c").read_text()
source_functions = ("source_access_anchor", "source_mapping_free", "source_registration_release", "source_compatibility_acquire",
                    "source_footprint_append", "source_sample_footprint", "source_mapping_access",
                    "source_mapping_qualify", "source_mapping_open", "source_mapping_domain_current_report",
                    "source_mapping_domain_current", "source_mapping_due",
                    "source_registration_acquire", "source_private_sample", "source_effects_read_allowed",
                    "source_stamp_equal", "source_note_refusal")

refusal_at = source.index("    SOURCE_REFUSAL_NONE,")
refusal_enum = source[source.rindex("enum {", 0, refusal_at):source.index("};", refusal_at) + 2]
footprint_end = source.index("} TextureSampleFootprint;") + len("} TextureSampleFootprint;")
footprint_type = source[source.rindex("typedef struct {", 0, footprint_end):footprint_end]
mapping_report_end = source.index("} SampleMappingRefusal;")
mapping_report_type = source[source.rindex("typedef struct {", 0, mapping_report_end):
                             source.index("static RcRegistryResult source_mapping_qualify", mapping_report_end)]
host_write = (ROOT / "runtime/resource_host_write.h").read_text()
qualification_type = host_write[host_write.index("typedef enum VkMemCanonicalSourceQualificationStage {"):
                                host_write.index("// Qualifies an ordinary tracked import")]


def function(name):
    match = re.search(r"^\w[^\n]*\b" + name + r"\([^;]*?\)\s*\{", source, re.M)
    if not match:
        raise RuntimeError(name)
    start = source.index("{", match.start())
    depth, end = 1, start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


prefix = r'''
#include <assert.h>
#include <stdio.h>
#include "texture_source.h"
#include "nvn_tex.h"
#define RC_REGISTRY_SINGLE_THREADED
#define rc_registry_runtime fixture_unused_runtime
#include "resource_core_registry.c"
#undef rc_registry_runtime
#define NVN_MEMORY_POOL_FLAGS_PHYSICAL 1u
#define NVN_MEMORY_POOL_FLAGS_VIRTUAL 2u
#include "resource_texture_source.h"
#include "resource_core_backing.h"
static RcRegistry *registry;
static RcRegistry *rc_registry_runtime(void) { return registry; }
typedef struct {uint64_t unused;} TextureBindingProof;
typedef unsigned VkFormat;
enum { VK_FORMAT_ASTC_4x4_UNORM_BLOCK=1000, VK_FORMAT_ASTC_12x12_SRGB_BLOCK=1100,
       VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK=2000, VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK=2100 };
int nvn_texels_slice_ranges(const nvn_texdesc *,unsigned,unsigned,uint32_t,
                           void (*)(void *,uint64_t,uint64_t),void *);
int nvn_deswizzle(const nvn_texdesc *,unsigned,unsigned,const void *,void *,size_t);
const int rt_nvn_api_version[2]={200,0};
const char *cfg_get(const char *key){(void)key;return NULL;}
typedef struct { RcInterval intervals[2]; uint64_t incarnation; } MemBackingCapture;
typedef struct {
    MemBackingCapture *lease;
    const RcInterval *intervals;
    RcInterval *sample_intervals;
    TextureSourceSpan *sample_spans;
    RcRegistrySourceAccess *access;
    TextureSourceAuthority *observed, *next, *copied;
    uint64_t native_generation;
    uint32_t count,sample_count;
} TextureSourceMapping;
typedef struct {
    int released;
    int source_blocked;
    uint8_t source_refusal, source_refusal_result, source_snapshot_mismatch, source_tracking_failure;
    RcRegistryPosition source_refusal_position;
    struct { uint64_t addr; int mirrored, gpu_owned, gone; uint32_t levels,layers; } pub;
    int gpu_dirty, bound_to, sfill, material_writable;
    struct { uint64_t addr; } key;
    struct { uint64_t serial; struct { uint64_t size; } L; unsigned pool_flags; nvn_texdesc d; uint8_t *mem; } t;
    struct {VkFormat ideal;unsigned astc_bw;} f;
    TextureSourceStamp source_stamp, source_authority_stamp;
    void *source_copy;
    int source_borrowed;
    TextureSourceStamp source_copy_stamp;
    TextureSourceAuthority source_copy_authority;
    RcRegistrySourceAccess source_access;
    TextureSourceMapping *source_mapping;
    RcRegistryReadObservation source_vector_read;
    uint8_t source_canonical,source_mapping_pins;
    uint64_t source_compatibility_generation;
    RcRegistryReadVersion material_version;
    RcRegistryBindingReceipt compatibility_receipt;
    RcRegistryPosition compatibility_position;
    TextureSourceAuthority source_authority;
    TextureBindingProof binding_proof,pool_binding_proof;
    TextureSourceSpan *source_spans;
    uint32_t source_count, wgen, checked;
    uint64_t checked_authority;
    uint32_t source_retry_domains, source_retry_submit;
    uint8_t source_retry_source;
} Tex;
static int content_image_eligible(const Tex *);
static int canonical_ready, unrelated_tick, qualifiers;
static int rt_knob(const char *key,const char *fallback){assert(!strcmp(key,"GRAPHICS_PREP_TRACE")&&!strcmp(fallback,"0"));return 0;}
static uint64_t native_generation=1;
static uint64_t canonical_incarnation=10;
static uint64_t vk_mem_gen(void){return native_generation;}
static RcBackingResult mem_backing_capture(uint64_t address,uint64_t size,MemBackingCapture **out){
    *out=calloc(1,sizeof **out);assert(*out && (size==64 || size==0x2000));
    (*out)->incarnation=canonical_incarnation;
    (*out)->intervals[0]=(RcInterval){address,canonical_incarnation,{100+canonical_incarnation,0,size/2}};
    (*out)->intervals[1]=(RcInterval){address+size/2,canonical_incarnation+1,{200+canonical_incarnation,0,size/2}};
    return RC_BACKING_OK;
}
static const RcInterval *mem_backing_intervals(const MemBackingCapture *capture,uint32_t *count){
    *count=capture?2:0;return capture?capture->intervals:NULL;
}
static int mem_backing_current(const MemBackingCapture *capture){return capture->incarnation==canonical_incarnation;}
static RcRegistryResult vk_mem_canonical_source_qualify(const MemBackingCapture *capture,uint64_t address,uint64_t size){
    assert(address==capture->intervals[0].address&&size==2*capture->intervals[0].range.size);
    if(!mem_backing_current(capture)) return RC_REGISTRY_STALE;
    qualifiers++;
    if (unrelated_tick) { unrelated_tick = 0; ++native_generation; }
    return canonical_ready?RC_REGISTRY_OK:RC_REGISTRY_UNSUPPORTED;
}
static void mem_backing_release(MemBackingCapture **capture){free(*capture);*capture=NULL;}
static int same_operation_write, require_private_sample;
static int nvn_gpu_only(uint64_t address) { return address >> 63; }
static int vk_exec_effects_read_allowed(uint64_t addr,uint64_t size) { (void)addr;(void)size;return !same_operation_write; }
static int vk_exec_effects_read_allowed_range(RcSourceRange range,int private_sample){
    (void)range;
    return !same_operation_write && (!require_private_sample || private_sample);
}
static RcRegistryPosition read_position;
static RcRegistryPosition vk_exec_resource_position(void) { return read_position; }
static uint32_t submit_epoch = 1;
static struct {
    unsigned observed,blocked,source_capacity,source_stale,source_other,observe_authority,observe_stale,observe_other,raced;
    unsigned canonical_eligible,canonical_excluded,canonical_blocked;
} resource_bridge;
static int race, passed_prior, mapped = 1;
static uint32_t watermark = 1;
static uint64_t host_mapping = 1, host_protection = 1, overlapping_protection;
static int host_dirty;
static int source_spans_init(Tex *e) { return e->source_count != 0; }
static int mem_mapping_range_unchanged(uint64_t a, uint64_t n, uint64_t g) {
    (void)a; (void)n; (void)g; return mapped;
}
static int mem_protection_range_unchanged(uint64_t a, uint64_t n, uint64_t g) {
    return mem_mapping_range_unchanged(a,n,g) && g >= overlapping_protection;
}
static int vk_mem_observe_subresources(uint64_t a, uint64_t n, const TextureSourceStamp *prior,
    uint32_t epoch, const TextureSourceSpan *spans, uint32_t count, TextureSourceObservation *out) {
    (void)epoch; (void)spans;
    passed_prior = prior != NULL;
    out->stamp = (TextureSourceStamp){host_mapping, host_protection, 1, watermark}; // CPU cutoff can survive a GPU store.
    if (race) rc_registry_cpu_changed(registry, (RcSourceRange){1,a,n});
    if (!prior) texture_source_all_dirty(out, count);
    return !prior ? TEX_SOURCE_UNKNOWN : host_dirty ? TEX_SOURCE_DIRTY : TEX_SOURCE_CLEAN;
}
typedef struct { unsigned requests; int accept; RcRegistryWrite expected; } VkExec;
typedef struct { unsigned used; } TextureRegistration;
enum { VK_TEXTURE_REGISTRATION_CURRENT=0, VK_TEXTURE_REGISTRATION_GONE=-2 };
static TextureRegistration registered={1};
static struct { const TextureRegistration *registration; } tslots[1]={{&registered}};
static const Tex *required_source;
static int registration_gone, tex_deferred;
static void texture_registration_source_clear(Tex *e){(void)e;}
static int texture_registration_resolve(int logical,const TextureRegistration **record,int *native){
    assert(logical==9);*record=registration_gone?&registered:NULL;*native=0;
    return registration_gone?VK_TEXTURE_REGISTRATION_GONE:VK_TEXTURE_REGISTRATION_CURRENT;
}
static const Tex *texture_source_entry_locked(int native){assert(native==0);return required_source;}
static int vk_exec_request_gpa_prefix(VkExec *ex,RcRegistryWrite writer,RcRegistryPosition producer,
                                     RcRegistryPosition read,RcInterval target){
    assert(writer.id==ex->expected.id&&writer.index==ex->expected.index);
    assert(producer.scene==4&&producer.ordinal==41&&read.scene==4&&read.ordinal==134);
    assert(target.range.backing==100+canonical_incarnation&&target.mapping==canonical_incarnation);
    ex->requests++;
    return ex->accept;
}
'''

main = r'''
int main(void) {
    registry = rc_registry_create((RcRegistryConfig){32,16,4}); assert(registry);
    TextureSourceSpan span = {0,64};
    Tex e = {.pub={.addr=4096,.mirrored=1}, .t={7,{64},0}, .source_spans=&span, .source_count=1};
    TextureSourceObservation out;
    assert(source_observe(&e,NULL,&out)==TEX_SOURCE_UNKNOWN && !passed_prior);
    source_commit(&e,&out.stamp,0); assert(texture_source_stamp_valid(&e.source_stamp));
    assert(source_observe(&e,&e.source_stamp,&out)==TEX_SOURCE_CLEAN && passed_prior);
    source_commit(&e,&out.stamp,0);

    RcRegistryWrite write;
    assert(rc_registry_write_begin(registry,(RcSourceRange){1,4096,64},RC_REGISTRY_BUFFER,&write)==RC_REGISTRY_OK);
    assert(rc_registry_write_submit(registry,write,1)==RC_REGISTRY_OK);
    assert(source_observe(&e,&e.source_stamp,&out)==TEX_SOURCE_UNKNOWN);
    assert(!texture_source_stamp_valid(&out.stamp));
    assert(rc_registry_write_complete(registry,write,1)==RC_REGISTRY_OK);
    assert(source_observe(&e,&e.source_stamp,&out)==TEX_SOURCE_UNKNOWN && !passed_prior);
    source_commit(&e,&out.stamp,0);
    assert(source_observe(&e,&e.source_stamp,&out)==TEX_SOURCE_CLEAN && passed_prior);
    source_commit(&e,&out.stamp,0);

    assert(rc_registry_cpu_changed(registry,(RcSourceRange){1,8192,64})==RC_REGISTRY_OK);
    assert(source_observe(&e,&e.source_stamp,&out)==TEX_SOURCE_CLEAN && passed_prior);
    source_commit(&e,&out.stamp,0);
    submit_epoch++;
    assert(source_observe(&e,&e.source_stamp,&out)==TEX_SOURCE_CLEAN && passed_prior);
    source_commit(&e,&out.stamp,0);

    // Retained snapshot keeps the pre-copy proof even when a later memory observation advances its watermark.
    e.source_copy = &e;
    e.source_copy_stamp = out.stamp;
    e.source_copy_authority = e.source_authority;
    e.source_authority_stamp.generation++;
    source_commit(&e,&e.source_copy_stamp,0);
    assert(texture_source_stamp_valid(&e.source_stamp) && e.source_stamp.generation==1);
    rc_registry_cpu_changed(registry,(RcSourceRange){1,4096,64});
    source_commit(&e,&e.source_copy_stamp,0); assert(!texture_source_stamp_valid(&e.source_stamp));
    e.source_copy = NULL;
    source_observe(&e,NULL,&out); source_commit(&e,&out.stamp,0);

    // Optional admission failed after its empty observation advanced scratch state. The subsequent ordinary
    // upload takes a new, paired cutoff before copying. A raced refresh never grants a publication certificate.
    TextureSourceObservation earlier = out, scratch;
    watermark++;
    source_observe(&e,NULL,&scratch);
    TextureSourceStamp upload_cutoff = prepare_upload_observation(&e,&earlier);
    assert(upload_cutoff.generation==watermark);
    source_commit(&e,&upload_cutoff,0);assert(e.source_stamp.generation==watermark);
    race=1;upload_cutoff=prepare_upload_observation(&e,&earlier);race=0;
    source_commit(&e,&upload_cutoff,0);assert(!texture_source_stamp_valid(&e.source_stamp));
    // Owned snapshots keep their pre-copy cutoff instead of entering the noncaptured refresh path.
    source_observe(&e,NULL,&scratch);e.source_copy=&e;e.source_copy_stamp=scratch.stamp;
    e.source_copy_authority=e.source_authority;watermark++;source_observe(&e,NULL,&scratch);
    upload_cutoff=prepare_upload_observation(&e,&scratch);
    assert(upload_cutoff.generation==e.source_copy_stamp.generation);
    source_commit(&e,&upload_cutoff,0);assert(e.source_stamp.generation==watermark-1);
    e.source_copy=NULL;

    race=1;
    assert(source_observe(&e,&e.source_stamp,&out)==TEX_SOURCE_UNKNOWN);
    assert(e.source_refusal==SOURCE_REFUSAL_RACE && e.source_refusal_result==RC_REGISTRY_STALE);
    assert(!texture_source_stamp_valid(&out.stamp)); race=0;
    assert(source_observe(&e,NULL,&out)==TEX_SOURCE_UNKNOWN);
    rc_registry_cpu_changed(registry,(RcSourceRange){1,4096,64});
    source_commit(&e,&out.stamp,0); assert(!texture_source_stamp_valid(&e.source_stamp));
    assert(source_observe(&e,NULL,&out)==TEX_SOURCE_UNKNOWN);
    mapped=0; source_commit(&e,&out.stamp,0); assert(!texture_source_stamp_valid(&e.source_stamp)); mapped=1;
    e.released=1;
    assert(source_observe(&e,NULL,&out)==TEX_SOURCE_UNKNOWN && !texture_source_stamp_valid(&out.stamp));
    // Two logical representations can observe the same guest source. Releasing one does not retire the other.
    e.released=0;
    Tex other = {.pub={.addr=4096,.mirrored=1}, .t={7,{64},0}, .source_spans=&span, .source_count=1};
    assert(source_observe(&other,NULL,&out)==TEX_SOURCE_UNKNOWN);
    source_commit(&other,&out.stamp,0);
    RcRegistryHandle shared = other.source_access.owner.handle;
    assert(shared.id==e.source_access.owner.handle.id);
    source_registration_release(&e);
    assert(!e.source_access.owner.handle.id && !e.source_access.owner.id && !e.checked_authority);
    assert(source_observe(&other,&other.source_stamp,&out)==TEX_SOURCE_CLEAN);
    source_registration_release(&other);
    RcRegistryRecord record;
    assert(rc_registry_get(registry,shared,&record)!=RC_REGISTRY_OK || !record.current);
    // Guest identity replacement must not retain its predecessor's CPU certificate.
    e.t.serial=8;
    assert(source_observe(&e,&e.source_stamp,&out)==TEX_SOURCE_UNKNOWN && !passed_prior);
    assert(e.source_access.owner.handle.id!=shared.id);
    source_registration_release(&e);
    // An earlier draw may capture ordinary input before a declared future GPU writer; a later draw may not.
    e.t.serial=9;
    read_position=(RcRegistryPosition){2,2};
    RcRegistryWrite future;
    assert(rc_registry_write_begin_at(registry,(RcSourceRange){1,4096,64},RC_REGISTRY_BUFFER,
                                     (RcRegistryPosition){2,5},&future)==RC_REGISTRY_OK);
    assert(rc_registry_declare_through(registry,(RcRegistryPosition){2,7})==RC_REGISTRY_OK);
    assert(source_observe(&e,NULL,&out)==TEX_SOURCE_UNKNOWN && texture_source_stamp_valid(&out.stamp));
    source_commit(&e,&out.stamp,0);
    assert(source_observe(&e,&e.source_stamp,&out)==TEX_SOURCE_CLEAN);
    read_position.ordinal=6;
    assert(!texture_authority_current(registry,&e.source_access._content,read_position));
    assert(source_observe(&e,&e.source_stamp,&out)==TEX_SOURCE_UNKNOWN && e.source_blocked);
    assert(e.source_refusal==SOURCE_REFUSAL_AUTHORITY && e.source_refusal_result==RC_REGISTRY_BLOCKED);
    assert(rc_registry_write_cancel(registry,future)==RC_REGISTRY_OK);
    assert(source_observe(&e,&e.source_stamp,&out)==TEX_SOURCE_CLEAN && !e.source_blocked);
    assert(e.source_refusal==SOURCE_REFUSAL_NONE);
    source_commit(&e,&out.stamp,0);
    assert(rc_registry_handoff_through(registry,(RcRegistryPosition){2,7})==RC_REGISTRY_OK);
    read_position=(RcRegistryPosition){3,2};
    assert(rc_registry_declare_through(registry,(RcRegistryPosition){3,3})==RC_REGISTRY_OK);
    // The old position is closed, but a new observation of identical contents can reuse its memory stamp.
    assert(source_observe(&e,&e.source_stamp,&out)==TEX_SOURCE_CLEAN && passed_prior);
    same_operation_write=1;
    assert(source_observe(&e,&e.source_stamp,&out)==TEX_SOURCE_UNKNOWN && e.source_blocked);
    assert(e.source_refusal==SOURCE_REFUSAL_EFFECTS && e.source_refusal_result==RC_REGISTRY_BLOCKED);
    same_operation_write=0;
    source_registration_release(&e);
    canonical_ready=1;native_generation++;
    const uint64_t sample_address = UINT64_C(0x5804c1a00);
    Tex sample={.source_canonical=1, .pub={.addr=sample_address,.mirrored=1,.levels=1,.layers=1},
                .key.addr=sample_address, .t={.serial=357,.L.size=0x2000,
                  .d={.format=0x25,.target=1,.width=64,.height=32,.depth=1,.levels=1}}};
    unrelated_tick=1;
    uint64_t before_generation=native_generation;
    assert(source_registration_acquire(&sample)==RC_REGISTRY_OK && sample.source_mapping &&
           sample.source_mapping->native_generation==before_generation && native_generation==before_generation+1);
    const TextureSourceMapping *selected_mapping=sample.source_mapping;
    const RcRegistryHandle selected_source=selected_mapping->access[0].owner.handle;
    const int before_qualification=qualifiers;
    unrelated_tick=1;
    before_generation=native_generation;
    assert(source_registration_acquire(&sample)==RC_REGISTRY_OK && sample.source_mapping==selected_mapping &&
           selected_mapping->access[0].owner.handle.id==selected_source.id &&
           selected_mapping->native_generation==before_generation && native_generation==before_generation+1);
    assert(source_registration_acquire(&sample)==RC_REGISTRY_OK && qualifiers==before_qualification+2);
    canonical_ready=0;native_generation++;
    assert(source_registration_acquire(&sample)==RC_REGISTRY_STALE && sample.source_mapping==selected_mapping);
    canonical_ready=1;
    source_registration_release(&sample);
    e.t.serial=10;
    assert(source_observe(&e,NULL,&out)==TEX_SOURCE_UNKNOWN && texture_source_stamp_valid(&out.stamp));
    assert(e.source_canonical && e.source_mapping->count==2 && !e.source_access.owner.id);
    require_private_sample=1;
    assert(source_effects_read_allowed(&e));
    e.material_writable=1;
    assert(!source_effects_read_allowed(&e));
    e.material_writable=0;
    e.pub.gpu_owned=1;
    assert(!source_effects_read_allowed(&e));
    e.pub.gpu_owned=0;
    require_private_sample=0;
    canonical_ready=0;native_generation++;
    assert(source_observe(&e,&e.source_stamp,&out)==TEX_SOURCE_UNKNOWN && e.source_blocked);
    assert(e.source_refusal==SOURCE_REFUSAL_ACQUIRE && e.source_refusal_result==RC_REGISTRY_STALE);
    assert(e.source_canonical && !e.source_access.owner.id);
    canonical_ready=1;native_generation++;
    assert(source_observe(&e,NULL,&out)==TEX_SOURCE_UNKNOWN && !e.source_blocked);
    RcRegistryHandle previous=e.source_mapping->access[0].owner.handle;
    RcSourceRange second=e.source_mapping->intervals[1].range;
    source_commit(&e,&out.stamp,0);
    assert(source_observe(&e,&e.source_stamp,&out)==TEX_SOURCE_CLEAN && passed_prior);
    uint64_t cursor=rc_registry_change_position(registry);
    assert(rc_registry_cpu_changed(registry,second)==RC_REGISTRY_OK);
    RcRegistryChange change;
    int overflow;
    assert(rc_registry_changes(registry,&cursor,&change,1,&overflow)==1 && !overflow);
    assert(change.association==e.pub.addr && change.range.backing==second.backing);
    assert(source_observe(&e,&e.source_stamp,&out)==TEX_SOURCE_UNKNOWN && !passed_prior);
    source_commit(&e,&out.stamp,0);
    assert(source_snapshot_hold(&e) && e.source_mapping_pins==1);
    TextureSourceAuthority copy=e.source_authority;
    TextureSourceStamp copied_stamp=out.stamp;
    canonical_incarnation++;
    assert(source_observe(&e,&copied_stamp,&out)==TEX_SOURCE_UNKNOWN && e.source_blocked);
    assert(!source_snapshot_current(&e,&copy) && !e.source_mapping_pins);
    assert(source_observe(&e,&e.source_stamp,&out)==TEX_SOURCE_UNKNOWN && !passed_prior);
    assert(e.source_mapping->access[0].owner.handle.id!=previous.id && !e.source_access.owner.id);
    source_registration_release(&e);

    uint8_t guest[64], snapshot[64], pixels[32];
    memset(guest,0xde,sizeof guest);
    for (unsigned row=0;row<4;row++) memset(guest+row*16, (int)(row+1), 8);
    Tex padded={.pub={.addr=4096,.mirrored=1,.levels=1,.layers=1}, .key.addr=4096,
                .t={.serial=11,.L.size=64,.d={.format=0x25,.target=1,.flags=0x10,.stride=16,.width=2,.height=4},
                    .mem=guest},.source_spans=&span,.source_count=1};
    RcRegistryWrite padding;
    RcSourceRange padding_range={100+canonical_incarnation,8,8};
    assert(rc_registry_hold_begin(registry,padding_range,RC_REGISTRY_UNKNOWN,&padding)==RC_REGISTRY_OK);
    assert(source_observe(&padded,NULL,&out)==TEX_SOURCE_UNKNOWN && texture_source_stamp_valid(&out.stamp));
    assert(padded.source_mapping->sample_count==4 && padded.source_mapping->count==2);
    assert(rc_registry_stats(registry).live_sources==2); // One lease/shape per backing, even with 4 texel spans.
    RcRegistrySourceAccess full={0};
    assert(rc_registry_source_access_acquire_associated(registry,12,padded.source_mapping->intervals[0].range,
        padded.pub.addr,&full)==RC_REGISTRY_OK);
    RcRegistrySourceProof all;
    assert(rc_registry_source_access_observe(registry,&full,submit_epoch,read_position,&all)==RC_REGISTRY_BLOCKED);
    assert(rc_registry_source_access_release(registry,&full)==RC_REGISTRY_OK);
    assert(rc_registry_source_access_acquire_associated(registry,11,padded.source_mapping->intervals[0].range,
        padded.pub.addr,&full)==RC_REGISTRY_STALE); // Same shaped identity cannot be reinterpreted as complete bytes.
    TextureSourceAuthority exact=padded.source_authority;
    assert(source_snapshot_hold(&padded));
    source_snapshot_copy(&padded,snapshot,sizeof snapshot);
    assert(source_snapshot_current(&padded,&exact));
    for (unsigned i=0;i<64;i++) assert(snapshot[i]==(i%16<8 ? guest[i] : 0));
    assert(!nvn_deswizzle(&padded.t.d,0,0,snapshot,pixels,8));
    for (unsigned row=0;row<4;row++) for (unsigned i=0;i<8;i++) assert(pixels[row*8+i]==row+1);
    RcPreparedInput material={61,61,1,61,RC_PROVIDER_CAPTURE};
    RcRegistryReadVersion retained;
    assert(rc_registry_read_commit(registry,&exact.ordered,&material,&retained)==RC_REGISTRY_OK);
    padded.source_copy=snapshot;padded.source_copy_stamp=out.stamp;padded.source_copy_authority=exact;
    TextureSourceObservation selected;
    allow_upload_capture=1;
    assert(source_upload_prepare(&padded,NULL,&selected) && padded.source_copy==snapshot && !replaced_snapshots);
    // The old private capture is available for exact providers, but the next current read needs newer bytes.
    read_position.ordinal+=2;
    assert(rc_registry_declare_through(registry,read_position)==RC_REGISTRY_OK);
    guest[0]=0x53;
    assert(rc_registry_cpu_changed(registry,(RcSourceRange){100+canonical_incarnation,0,1})==RC_REGISTRY_OK);
    assert(source_upload_prepare(&padded,NULL,&selected) && padded.source_copy==upload_capture);
    assert(upload_captures==1 && replaced_snapshots==1 && snapshot[0]==1 && upload_capture[0]==0x53);
    source_commit(&padded,&padded.source_copy_stamp,0);
    assert(texture_source_stamp_valid(&padded.source_stamp) &&
           rc_registry_source_access_read_extents(registry,padded.source_mapping->access,padded.source_mapping->count,
                                                  submit_epoch,read_position,NULL)==RC_REGISTRY_OK);
    // A same-ordinal byte change or mapping replacement invalidates the source proof independently of age.
    guest[0]=0x54;
    assert(rc_registry_cpu_changed(registry,(RcSourceRange){100+canonical_incarnation,0,1})==RC_REGISTRY_OK);
    assert(source_upload_prepare(&padded,NULL,&selected) && upload_captures==2 && upload_capture[0]==0x54);
    canonical_incarnation++;guest[0]=0x55;
    assert(source_upload_prepare(&padded,NULL,&selected) && upload_captures==3 && upload_capture[0]==0x55);
    assert(replaced_snapshots==3 && snapshot[0]==1 && retained.id && retained.input.payload==61);
    source_commit(&padded,&padded.source_copy_stamp,0);
    assert(texture_source_stamp_valid(&padded.source_stamp) &&
           rc_registry_source_access_read_extents(registry,padded.source_mapping->access,padded.source_mapping->count,
                                                  submit_epoch,read_position,NULL)==RC_REGISTRY_OK);
    padded.source_copy=NULL;allow_upload_capture=0;

    // Actual GPU completion changes the canonical content revision without a CPU write-watch generation.
    // The existing uploaded material's host stamp equals the new snapshot stamp, but its authority is older.
    RcRegistryWrite settled;
    RcSourceRange meaningful = {200+canonical_incarnation,3,1};
    assert(rc_registry_write_begin(registry,meaningful,RC_REGISTRY_BUFFER,&settled)==RC_REGISTRY_OK);
    assert(rc_registry_write_submit(registry,settled,81)==RC_REGISTRY_OK);
    assert(rc_registry_write_complete(registry,settled,81)==RC_REGISTRY_OK);
    assert(source_observe(&padded,NULL,&out)==TEX_SOURCE_UNKNOWN && texture_source_stamp_valid(&out.stamp));
    assert(source_stamp_equal(&padded.source_stamp,&out.stamp));
    exact=padded.source_authority;
    assert(source_snapshot_hold(&padded));
    assert(source_snapshot_verify(&padded,&out.stamp,&exact) && !padded.source_mapping_pins);
    // Ordinary reuse of the earlier material still requires its committed proof, and sees the new write.
    assert(source_observe(&padded,&padded.source_stamp,&selected)==TEX_SOURCE_UNKNOWN && !passed_prior);

    // Global mapping/protection clocks advance for other addresses while the exact watched range is unchanged.
    assert(source_observe(&padded,NULL,&out)==TEX_SOURCE_UNKNOWN);
    exact=padded.source_authority;
    assert(source_snapshot_hold(&padded));
    host_mapping++; host_protection++;
    assert(source_snapshot_verify(&padded,&out.stamp,&exact) && !padded.source_mapping_pins);

    // A GPU/CPU write to an actual copied byte of the second extent invalidates the original cutoff.
    assert(source_observe(&padded,NULL,&out)==TEX_SOURCE_UNKNOWN);
    exact=padded.source_authority;
    assert(source_snapshot_hold(&padded));
    assert(rc_registry_cpu_changed(registry,meaningful)==RC_REGISTRY_OK);
    assert(!source_snapshot_verify(&padded,&out.stamp,&exact) && !padded.source_mapping_pins);
    assert((padded.source_snapshot_mismatch & 3)==3);

    // Local thaw and unreported CPU write evidence remain blockers even with unchanged registry proofs.
    assert(source_observe(&padded,NULL,&out)==TEX_SOURCE_UNKNOWN);
    exact=padded.source_authority;
    assert(source_snapshot_hold(&padded));
    overlapping_protection=++host_protection;
    assert(!source_snapshot_verify(&padded,&out.stamp,&exact) && !padded.source_mapping_pins);
    assert(padded.source_snapshot_mismatch==8);
    assert(source_observe(&padded,NULL,&out)==TEX_SOURCE_UNKNOWN);
    exact=padded.source_authority;
    assert(source_snapshot_hold(&padded));
    host_dirty=1;
    assert(!source_snapshot_verify(&padded,&out.stamp,&exact) && !padded.source_mapping_pins);
    assert(padded.source_snapshot_mismatch==1);
    host_dirty=0;
    RcRegistryWrite missing;
    RcSourceRange missing_range={200+canonical_incarnation,3,1};
    assert(rc_registry_hold_begin(registry,missing_range,RC_REGISTRY_UNKNOWN,&missing)==RC_REGISTRY_OK);
    assert(source_observe(&padded,NULL,&out)==TEX_SOURCE_UNKNOWN && !texture_source_stamp_valid(&out.stamp));
    assert(padded.source_blocked && padded.source_refusal==SOURCE_REFUSAL_AUTHORITY);
    assert(rc_registry_hold_release(registry,missing,1)==RC_REGISTRY_OK);
    assert(source_observe(&padded,NULL,&out)==TEX_SOURCE_UNKNOWN && texture_source_stamp_valid(&out.stamp));
    RcRegistryHandle old_object=padded.source_mapping->access[0].owner.handle;
    exact=padded.source_authority;
    assert(source_snapshot_hold(&padded));
    canonical_incarnation++;
    assert(!source_snapshot_verify(&padded,&out.stamp,&exact) && !padded.source_mapping_pins);
    source_registration_release(&padded);
    padded.t.serial=13;
    assert(source_observe(&padded,NULL,&out)==TEX_SOURCE_UNKNOWN && texture_source_stamp_valid(&out.stamp));
    assert(padded.source_mapping->access[0].owner.handle.id!=old_object.id);
    assert(retained.id && retained.origin.handle.id!=old_object.id && retained.input.payload==61 && snapshot[0]==1);

    VkExec ex={.accept=1};
    assert(rc_registry_write_guard_begin_reason_at(registry,RC_REGISTRY_BUFFER,(RcRegistryPosition){4,41},
        RC_REGISTRY_GUARD_GPA_PROJECTION,&ex.expected)==RC_REGISTRY_OK);
    assert(rc_registry_declare_through(registry,(RcRegistryPosition){4,135})==RC_REGISTRY_OK);
    read_position=(RcRegistryPosition){4,134};required_source=&padded;
    assert(source_observe(&padded,NULL,&out)==TEX_SOURCE_UNKNOWN && padded.source_blocked);
    registration_gone=1;
    assert(!vk_texture_sample_wait_gpa_locked(&ex,9) && !ex.requests);
    registration_gone=0;read_position.ordinal=132;
    assert(!vk_texture_sample_wait_gpa_locked(&ex,9) && !ex.requests);
    read_position.ordinal=134;
    ex.accept=0;assert(!vk_texture_sample_wait_gpa_locked(&ex,9) && !tex_deferred && ex.requests==1);
    ex.accept=1;assert(vk_texture_sample_wait_gpa_locked(&ex,9) && tex_deferred && ex.requests==2);
    assert(!vk_texture_sample_wait_gpa_locked(&ex,9) && ex.requests==2);
    tex_deferred=0;
    assert(source_observe(&padded,NULL,&out)==TEX_SOURCE_UNKNOWN && padded.source_blocked); // Request grants no authority.
    assert(rc_registry_write_cancel(registry,ex.expected)==RC_REGISTRY_OK);
    assert(source_observe(&padded,NULL,&out)==TEX_SOURCE_UNKNOWN && texture_source_stamp_valid(&out.stamp));
    source_registration_release(&padded);
    assert(rc_registry_hold_release(registry,padding,1)==RC_REGISTRY_OK);

    RcRegistry *bounded=rc_registry_create((RcRegistryConfig){1,2,2});
    RcSourceRange hull={99,0,128}, spans[33];
    for (unsigned i=0;i<33;i++) spans[i]=(RcSourceRange){99,i*2,1};
    assert(rc_registry_source_access_acquire_shaped(bounded,1,hull,4096,spans,33,&full)==RC_REGISTRY_CAPACITY);
    assert(!full.owner.id && !rc_registry_stats(bounded).records);
    assert(rc_registry_source_access_acquire_shaped(bounded,1,hull,4096,spans,32,&full)==RC_REGISTRY_OK);
    assert(rc_registry_source_access_release(bounded,&full)==RC_REGISTRY_OK);
    assert(rc_registry_source_access_acquire_shaped(bounded,2,hull,4096,spans,32,&full)==RC_REGISTRY_OK);
    rc_registry_destroy(bounded);
    rc_registry_destroy(registry);
    puts("source registry bridge: canonical vector, exact copied texels, padding, remap and capacity passed");
}
'''

work = ROOT / "build/test"
work.mkdir(parents=True, exist_ok=True)
work.resolve().relative_to(ROOT.resolve())
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="resource-source-", dir=work) as temporary:
    c = Path(temporary) / "check.c"
    prepare = """
static int vk_exec_turn_due(void){return 0;}
static int allow_upload_capture;
static unsigned replaced_snapshots,upload_captures;
static uint8_t upload_capture[64];
static void content_source_discard(Tex *e){
    assert(e->source_copy&&!e->source_borrowed);
    e->source_copy=NULL;replaced_snapshots++;
}
static int content_early_prepare(Tex *e){
    assert(allow_upload_capture&&e->t.L.size==64&&!e->source_copy);
    TextureSourceObservation observation;
    source_observe(e,NULL,&observation);
    if(!texture_source_stamp_valid(&observation.stamp))return 0;
    assert(source_snapshot_hold(e));
    source_snapshot_copy(e,upload_capture,sizeof upload_capture);
    assert(source_snapshot_current(e,&e->source_authority));
    e->source_copy=upload_capture;
    e->source_copy_stamp=observation.stamp;
    e->source_copy_authority=e->source_authority;
    upload_captures++;
    return 0;
}
""" + function("source_upload_retain_current") + function("source_upload_prepare") + """
static TextureSourceStamp prepare_upload_observation(Tex *e,const TextureSourceObservation *observed){
    TextureSourceObservation observation={0};
    if(!source_upload_prepare(e,observed,&observation))return (TextureSourceStamp){0};
    return e->source_copy?e->source_copy_stamp:observation.stamp;
}
"""
    host_copy = """
static int content_image_eligible(const Tex *e){return !e->source_blocked;}
static int mem_read_hold(uint64_t addr,uint64_t size){return addr&&size;}
static int mem_backing_read_hold(const MemBackingCapture *capture){return mem_backing_current(capture);}
"""
    qualification = """
static RcRegistryResult vk_mem_canonical_source_qualify_report(const MemBackingCapture *capture,
    uint64_t address,uint64_t size,VkMemCanonicalSourceQualification *out){
    assert(out);
    *out=(VkMemCanonicalSourceQualification){0};
    return vk_mem_canonical_source_qualify(capture,address,size);
}
"""
    funcs = source_functions + ("source_snapshot_hold", "source_snapshot_copy", "source_snapshot_current",
                                "source_observe_prior", "source_observe", "source_snapshot_verify", "source_commit",
                                "vk_texture_sample_wait_gpa_locked")
    c.write_text(prefix + host_copy + refusal_enum + footprint_type + qualification_type + mapping_report_type +
                 qualification + "\n".join(function(n) for n in funcs) + prepare + main)
    exe = c.with_suffix(".exe")
    subprocess.run([str(CC), "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                    "-fsanitize-trap=all", "-Wno-missing-field-initializers", "-I" + str(ROOT / "runtime"),
                    "-I" + str(CC.parent.parent / "include/vma"), str(c), str(ROOT / "runtime/tex_deswizzle.c"),
                    str(ROOT / "runtime/nvn_tex.c"), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

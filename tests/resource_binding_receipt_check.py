"""Check the production texture binding receipt with controlled source-validation boundaries."""
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from check_content import GENERATED_MARKER

PREFIX = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include "resource_core_registry.h"
#define RC_REGISTRY_SINGLE_THREADED
#include "resource_core_registry.c"
#include "texture_source.h"
#include "resource_texture_source.h"
typedef uint64_t VkImage;
/* BINDING_PROOF_TYPE */
#define VK_NULL_HANDLE 0
#define NVN_MEMORY_POOL_FLAGS_PHYSICAL 1u
#define NVN_MEMORY_POOL_FLAGS_VIRTUAL 2u
enum {VK_TEXTURE_BINDING_DIRECT=1,VK_TEXTURE_BINDING_SOURCE=2,VK_TEXTURE_BINDING_LEGACY=4};
typedef struct { struct { uint64_t size; } L; int mirrored; unsigned pool_flags; uint64_t serial; } tex_t;
typedef struct {uint64_t registration;int native_slot;uint8_t canonical,refused,projected;} VkTextureReadKey;
typedef struct {int set_layers;uint32_t min_layer;} view_t;
typedef struct {RcRegistryHandle handle;uint64_t publication;tex_t descriptor;struct Tex *source;view_t view;} TextureRegistration;
typedef struct { struct { uint64_t id; } value; } FakeInstance;
typedef struct { struct { FakeInstance *instance; } resident; } TextureMaterial;
typedef struct TextureFlat3D {
    struct { VkImage image; int gone; } image;
    TextureMaterial *material;
    uint64_t source_instance;
    VkImage source;
    uint64_t revision;
    uint32_t upload,layer;
    int ordered;
    RcRegistryPosition position;
    struct TextureFlat3D *next;
} TextureFlat3D;
typedef struct { RcRegistrySourceAccess *access; uint32_t count; } TextureSourceMapping;
typedef struct Tex {
    struct { VkImage image; uint64_t addr; int gone,gpu_owned; } pub;
    tex_t t;
    int uploaded,released,retiring,up_retry,source_blocked,storage,superseded;
    uint32_t checked,wgen,wt_n;
    uint64_t checked_authority;
    uint64_t content_revision;
    uint32_t n_up;
    TextureMaterial *material;
    int material_writable;
    TextureFlat3D *flat3d;
    RcRegistrySourceAccess source_access;
    TextureSourceMapping *source_mapping;
    RcRegistryReadObservation source_vector_read;
    uint8_t source_canonical;
    RcRegistryWrite image_authority;
    void *source_spans;
    unsigned source_count;
    RcRegistryPosition compatibility_position;
    RcRegistryBindingReceipt compatibility_receipt;
    VkImage compatibility_image;
    uint64_t compatibility_mapping,compatibility_protection;
    uint32_t compatibility_wgen,compatibility_writes;
    TextureBindingProof binding_proof;
    TextureBindingProof pool_binding_proof;
    RcRegistryPosition pool_schedule_edge;
    TextureSourceStamp source_stamp;
} Tex;
typedef struct { int used,finalized; Tex *ce; uint32_t cgen; tex_t t;
    int chas_view; view_t cview;
    TextureRegistration *registration;uint64_t pool_owner;
} TexSlot;
static TexSlot tslots[2];static int ntslots=2;
static uint32_t agen[2]={1,1};static uint64_t sinfo[2]={0,1};
static RcRegistryPosition position={1,2};static int current=1;static unsigned checks;
static RcRegistry *registry;
#define rc_registry_runtime() registry
static uint32_t submit_epoch=1;
static int tex_deferred,effects_allowed=1;
static uint64_t mapping=1,protection=1;
static uint64_t selected_publication=1,selected_pool=9;
static uint64_t vk_texture_registration_publication(void){return selected_publication;}
static uint64_t vk_exec_texture_pool(void){return selected_pool;}
static int from_storage(const Tex *e){return e->storage;}
static int preserve_pool_contents_current(const Tex *e){(void)e;return 1;}
static const Tex *texture_projected_sample_locked(const TexSlot *slot,int native,VkImage expected,
    void *sample,RcRegistryPosition *next){
    (void)slot;(void)native;(void)expected;(void)sample;(void)next;
    assert(0); // Projected image providers have a separate fixture.
    return NULL;
}
static int source_private_sample(const Tex *e){return e->storage && !e->pub.gone && !e->pub.gpu_owned && !e->material_writable;}
static Tex *texture_registration_source(const TexSlot *s){return s->registration?s->registration->source:NULL;}
static void slot_result_view(const TexSlot *s,const Tex *e,view_t *view,int *has_view) {
    assert(s->registration->source==e);*view=s->registration->view;*has_view=1;
}
static int tex_mirrored(const tex_t *t){return t->mirrored;}
static RcRegistryPosition vk_exec_resource_position(void){return position;}
static uint64_t mem_mapping_serial(void){return mapping;}
static uint64_t mem_protection_serial(void){return protection;}
static int nvn_gpu_only(uint64_t address){return address >> 63;}
static int mem_mapping_range_unchanged(uint64_t a,uint64_t n,uint64_t stamp){(void)a;(void)n;return stamp && stamp==mapping;}
static int mem_protection_range_unchanged(uint64_t a,uint64_t n,uint64_t stamp){(void)a;(void)n;return stamp && stamp==protection;}
static int vk_exec_effects_read_allowed(uint64_t a,uint64_t n){(void)a;(void)n;return effects_allowed;}
static int source_binding_current(Tex *e){
    checks++;RcRegistryPosition next=e->source_access._read.next_write;
    // The source helper rejects the current command's writable aliases before the producer write phase.
    if(next.scene==position.scene && next.ordinal==position.ordinal+1)return 0;
    return current && (!rc_registry_position_valid(next) || rc_registry_position_compare(position,next)<0);
}
'''
MAIN = r'''
int main(void){
    registry=rc_registry_create((RcRegistryConfig){4,12,1});assert(registry);
    Tex e={.pub.image=7,.uploaded=1,.storage=1};
    tslots[1]=(TexSlot){.used=1,.ce=&e,.cgen=1,.t.mirrored=1};
    e.source_access._content.ordered=(RcRegistryReadObservation){.handle.id=1,.position=position,
        .sealed_through={1,100},.next_write={1,99}};
    e.source_access._read.position=position;
    e.source_access._read.sealed_through=(RcRegistryPosition){1,100};
    e.source_access._read.next_write=(RcRegistryPosition){1,5};
    RcRegistryPosition next;
    assert(vk_texture_binding_receipt_locked(-1,0,&next));
    assert(!vk_texture_binding_receipt_locked(-1,7,&next));
    assert(vk_texture_binding_receipt_locked(0,0,&next));
    assert(!vk_texture_binding_receipt_locked(0,7,&next));
    assert(binding_path.absent==2 && binding_path.invalid_slot==2);
    assert(vk_texture_binding_receipt_locked(1,7,&next)==VK_TEXTURE_BINDING_SOURCE && next.ordinal==4 && checks==1);
    assert(binding_path.ordinary==1);
    TextureRegistration registration={.handle={77,1},.publication=2,.descriptor.serial=87};
    tslots[1].registration=&registration;tslots[1].pool_owner=9;e.t.serial=87;
    VkTextureReadKey selected={.registration=texture_slot_registration(1),.native_slot=1};
    assert(!vk_texture_resolved_receipt_locked(selected,7,&next)); // Too new for this horizon.
    selected_publication=2;
    selected_pool=10;
    assert(!vk_texture_resolved_receipt_locked(selected,7,&next)); // Recycled native slot in another pool.
    selected_pool=9;e.t.serial++;
    assert(!vk_texture_resolved_receipt_locked(selected,7,&next));
    e.t.serial--;
    assert(vk_texture_resolved_receipt_locked(selected,7,&next)==VK_TEXTURE_BINDING_SOURCE&&next.ordinal==4);
    selected.canonical=1;
    assert(!vk_texture_resolved_receipt_locked(selected,7,&next)); // The exact provider withdrew qualified mode.
    registration.source=&e;e.source_canonical=1;
    // Qualified consumers validate the actual registration even with a dead old cache pointer at this bucket.
    tslots[1].ce=(Tex *)(uintptr_t)1;agen[1]++;
    assert(vk_texture_resolved_receipt_locked(selected,7,&next)==VK_TEXTURE_BINDING_SOURCE);
    assert(vk_texture_binding_receipt_locked(1,7,&next)==VK_TEXTURE_BINDING_SOURCE);
    assert(!vk_texture_resolved_receipt_locked(selected,17,&next));
    VkTextureReadKey earlier=selected;earlier.registration--;
    assert(!vk_texture_resolved_receipt_locked(earlier,7,&next));
    current=0;
    assert(!vk_texture_resolved_receipt_locked(selected,7,&next));
    current=1;
    e.material_writable=1;
    assert(!vk_texture_resolved_receipt_locked(selected,7,&next));
    e.material_writable=0;e.pub.gpu_owned=1;
    assert(!vk_texture_resolved_receipt_locked(selected,7,&next));
    e.pub.gpu_owned=0;
    e.source_canonical=0;registration.source=NULL;
    tslots[1].ce=&e;tslots[1].cgen=agen[1];
    assert(e.source_access._read.next_write.ordinal==5&&e.source_access._content.ordered.next_write.ordinal==99);
    // A sealed READ phase is inclusive. Its exclusive boundary is the following ordinal, not this input itself.
    e.source_access._read.sealed_through=position;
    assert(vk_texture_binding_receipt_locked(1,7,&next)==VK_TEXTURE_BINDING_SOURCE && next.ordinal==3);
    e.source_access._read.sealed_through=(RcRegistryPosition){0};
    assert(!vk_texture_binding_receipt_locked(1,7,&next));
    assert(e.source_access._content.ordered.sealed_through.ordinal==100); // Retained byte provenance is not read permission.
    e.source_access._read.sealed_through=(RcRegistryPosition){1,100};
    assert(vk_texture_binding_receipt_locked(1,7,&next)==VK_TEXTURE_BINDING_SOURCE && next.ordinal==4);
    e.source_access._content.ordered=(RcRegistryReadObservation){0};
    e.source_access._content.legacy.handle.id=1;
    assert(vk_texture_binding_receipt_locked(1,7,&next)==VK_TEXTURE_BINDING_SOURCE&&next.ordinal==4);
    position.ordinal=4;assert(rc_registry_position_compare(position,next)==0);
    assert(!vk_texture_binding_current_locked(1,7));
    position.ordinal=8;assert(!vk_texture_binding_current_locked(1,7));
    position.ordinal=2;current=0;assert(!vk_texture_binding_current_locked(1,7));current=1;
    assert(!vk_texture_binding_current_locked(1,8));
    e.pub.gone=1;assert(!vk_texture_binding_current_locked(1,7));e.pub.gone=0;
    agen[1]++;tslots[1].ce=(Tex *)(uintptr_t)1;
    assert(!vk_texture_binding_current_locked(1,7)); // stale pointer rejected before dereference
    tslots[1].ce=&e;tslots[1].cgen=agen[1];
    tslots[1].finalized=1;e.released=1;current=0;
    assert(vk_texture_binding_receipt_locked(1,7,&next) && !next.scene);
    tslots[1].finalized=0;assert(!vk_texture_binding_current_locked(1,7));e.released=0;
    e.pub.gpu_owned=1;assert(vk_texture_binding_receipt_locked(1,7,&next)==VK_TEXTURE_BINDING_DIRECT && !next.scene);e.pub.gpu_owned=0;
    // A directly bound image has no guest upload. The same flags on a mirrored source are not ready.
    e.uploaded=0;e.storage=0;unsigned before_checks=checks;
    assert(vk_texture_binding_receipt_locked(1,7,&next) && !next.scene && checks==before_checks);
    e.storage=1;assert(!vk_texture_binding_current_locked(1,7));
    e.pub.gpu_owned=1;assert(vk_texture_binding_current_locked(1,7));e.pub.gpu_owned=0;
    // slot_resolve retains a released GPU-only slot's own nonsuperseded image. It cannot authorize a released
    // CPU slot, stale cached pointer, retiring image or failed upload merely because the physical handle matches.
    e.released=1;e.storage=0;tslots[1].t.mirrored=0;
    assert(vk_texture_binding_current_locked(1,7));
    tslots[1].t.mirrored=1;assert(!vk_texture_binding_current_locked(1,7));tslots[1].t.mirrored=0;
    e.superseded=1;assert(!vk_texture_binding_current_locked(1,7));e.superseded=0;
    e.retiring=1;assert(!vk_texture_binding_current_locked(1,7));e.retiring=0;
    e.up_retry=1;assert(!vk_texture_binding_current_locked(1,7));e.up_retry=0;
    tslots[1].cgen--;assert(!vk_texture_binding_current_locked(1,7));tslots[1].cgen++;
    e.released=0;e.storage=1;e.uploaded=1;tslots[1].t.mirrored=1;
    e.source_access._content=(TextureSourceAuthority){0};e.source_access._read=(RcRegistryReadObservation){0};
    assert(!vk_texture_binding_current_locked(1,7));
    e.compatibility_position=position;
    e.checked=submit_epoch;e.checked_authority=rc_registry_authority_epoch(registry);
    assert(vk_texture_binding_receipt_locked(1,7,&next)==(VK_TEXTURE_BINDING_SOURCE|VK_TEXTURE_BINDING_LEGACY) && next.ordinal==3);
    e.checked_authority--;
    assert(!vk_texture_binding_current_locked(1,7));
    assert(!source_binding_compat_current(&e));e.checked_authority++;
    e.checked++;assert(!vk_texture_binding_current_locked(1,7));e.checked--;
    position.ordinal=4;assert(!vk_texture_binding_current_locked(1,7));
    position=(RcRegistryPosition){0};e.compatibility_position=position;
    assert(!vk_texture_binding_current_locked(1,7));
    assert(source_binding_compat_current(&e)); // Outside replay preserves legacy lookup semantics, not an interval.

    // Actual compatibility helpers bracket a successful legacy preparation while an earlier producer blocks
    // source certification. The retained physical result is reusable only until the next producer's read phase.
    RcSourceRange range={1,100,20};RcRegistryWrite prior,future,unknown;
    e.t.serial=1;
    assert(rc_registry_write_begin_at(registry,(RcSourceRange){1,0,1000},RC_REGISTRY_UNKNOWN,
                                     (RcRegistryPosition){1,3},&prior)==RC_REGISTRY_OK);
    assert(rc_registry_write_begin_at(registry,range,RC_REGISTRY_BUFFER,(RcRegistryPosition){1,9},&future)==RC_REGISTRY_OK);
    assert(rc_registry_declare_through(registry,(RcRegistryPosition){1,10})==RC_REGISTRY_OK);
    position=(RcRegistryPosition){1,4};e.pub.addr=100;e.t.L.size=20;e.wgen=12;e.checked=submit_epoch;
    e.source_blocked=1;e.source_access._content.ordered.position=(RcRegistryPosition){1,2};
    assert(!source_binding_ordered(&e)); // A stale ordinary proof cannot eclipse the compatibility route.
    // Material subresource preparation has no recipe for this source. The actual compatibility helper acquires
    // one persistent raw-range lease without publishing CPU contents or creating material spans.
    assert(!e.source_access.owner.handle.id && !e.source_spans && !e.source_count);
    CompatibilityPreparation p=compatibility_begin(&e);assert(p.valid);
    assert(e.source_access.owner.id && !e.source_spans && !e.source_count);
    uint64_t lease=e.source_access.owner.id;
    assert(compatibility_begin(&e).valid && e.source_access.owner.id==lease && rc_registry_stats(registry).source_leases==1);
    compatibility_commit(&e,&p);assert(e.compatibility_receipt.observation.handle.id);
    RcRegistryReadObservation clean;
    assert(rc_registry_observe_at(registry,e.source_access.owner.handle,submit_epoch,position,&clean)==RC_REGISTRY_BLOCKED);
    position.ordinal=6;
    assert(vk_texture_binding_receipt_locked(1,7,&next)==VK_TEXTURE_BINDING_SOURCE && next.ordinal==8);
    assert(vk_texture_binding_current_locked(1,7)==1); // The untyped API returns a normalized boolean.
    assert(source_binding_compat_current(&e));
    // Unrelated authority transitions cannot invalidate this source-specific interval through the legacy gate.
    RcRegistryWrite unrelated;
    e.checked_authority=rc_registry_authority_epoch(registry);
    assert(rc_registry_write_begin(registry,(RcSourceRange){1,2000,20},RC_REGISTRY_BUFFER,&unrelated)==RC_REGISTRY_OK);
    assert(e.checked_authority!=rc_registry_authority_epoch(registry));
    assert(source_binding_compat_current(&e));
    assert(vk_texture_binding_receipt_locked(1,7,&next) && next.ordinal==8);
    assert(rc_registry_write_cancel(registry,unrelated)==RC_REGISTRY_OK);
    position.ordinal=8;assert(!vk_texture_binding_current_locked(1,7));position.ordinal=6;
    e.pub.image=8;assert(!source_binding_compat_receipt(&e,NULL));e.pub.image=7;
    e.wgen++;assert(!source_binding_compat_receipt(&e,NULL));e.wgen--;
    e.wt_n++;assert(!source_binding_compat_receipt(&e,NULL));e.wt_n--;
    mapping++;assert(!source_binding_compat_receipt(&e,NULL));mapping--;
    protection++;assert(!source_binding_compat_receipt(&e,NULL));protection--;
    submit_epoch++;assert(!source_binding_compat_receipt(&e,NULL));submit_epoch--;
    effects_allowed=0;assert(!source_binding_compat_receipt(&e,NULL));effects_allowed=1;

    // A transition during the real preparation bracket never publishes an interval, even when uploaded remains
    // true. Same-operation fallback remains available, but the next command cannot reuse the failed bracket.
    position.ordinal=4;e.compatibility_receipt=(RcRegistryBindingReceipt){0};
    p=compatibility_begin(&e);assert(p.valid);
    assert(rc_registry_cpu_changed(registry,range)==RC_REGISTRY_OK);
    compatibility_commit(&e,&p);assert(!e.compatibility_receipt.observation.handle.id);
    position.ordinal=6;assert(!vk_texture_binding_current_locked(1,7));position.ordinal=4;
    p=compatibility_begin(&e);assert(p.valid);mapping++;
    compatibility_commit(&e,&p);assert(!e.compatibility_receipt.observation.handle.id);mapping--;
    p=compatibility_begin(&e);assert(p.valid);protection++;
    compatibility_commit(&e,&p);assert(!e.compatibility_receipt.observation.handle.id);protection--;
    p=compatibility_begin(&e);assert(p.valid);tex_deferred=1;
    compatibility_commit(&e,&p);assert(!e.compatibility_receipt.observation.handle.id);tex_deferred=0;
    e.up_retry=1;compatibility_commit(&e,&p);assert(!e.compatibility_receipt.observation.handle.id);e.up_retry=0;
    assert(rc_registry_handoff_through(registry,(RcRegistryPosition){1,3})==RC_REGISTRY_OK);
    assert(rc_registry_write_submit(registry,prior,1)==RC_REGISTRY_OK);
    p=compatibility_begin(&e);assert(p.valid);
    assert(rc_registry_write_complete(registry,prior,0)==RC_REGISTRY_OK);
    compatibility_commit(&e,&p);assert(!e.compatibility_receipt.observation.handle.id);
    assert(rc_registry_write_begin(registry,range,RC_REGISTRY_UNKNOWN,&unknown)==RC_REGISTRY_OK);
    p=compatibility_begin(&e);assert(!p.valid);
    assert(rc_registry_write_cancel(registry,unknown)==RC_REGISTRY_OK);
    p=compatibility_begin(&e);assert(p.valid);compatibility_commit(&e,&p);
    position.ordinal=6;assert(source_binding_compat_receipt(&e,NULL));
    assert(rc_registry_range_gone(registry,range)==RC_REGISTRY_OK);
    assert(!vk_texture_binding_current_locked(1,7));
    // Shared image authority cannot grant a reusable receipt to a CPU alias. Direct use of the actual GPU image
    // remains valid under the renderer's command/barrier order, without pretending its guest bytes are current.
    RcRegistryWrite image_hold;
    assert(rc_registry_source_release(registry,&e.source_access.owner)==RC_REGISTRY_OK);
    e.source_access.owner.handle=(RcRegistryHandle){0};e.t.serial=2;
    assert(rc_registry_hold_begin(registry,range,RC_REGISTRY_IMAGE,&image_hold)==RC_REGISTRY_OK);
    p=compatibility_begin(&e);assert(!p.valid);
    e.compatibility_position=(RcRegistryPosition){0};
    assert(!vk_texture_binding_current_locked(1,7));
    e.pub.gpu_owned=1;e.uploaded=0;
    assert(vk_texture_binding_receipt_locked(1,7,&next) && !next.scene);
    e.pub.gpu_owned=0;e.uploaded=1;
    assert(rc_registry_hold_release(registry,image_hold,1)==RC_REGISTRY_OK);
    // Preparation counters include exits before the registry sees a call. Revalidation keeps its refusal class
    // separate from the successful begin/commit bracket, including out-of-order worker consumption.
    e.t.pool_flags=NVN_MEMORY_POOL_FLAGS_PHYSICAL;
    p=compatibility_begin(&e);assert(!p.valid && compatibility_path.begin_alias_pool==1);
    e.t.pool_flags=NVN_MEMORY_POOL_FLAGS_VIRTUAL;
    p=compatibility_begin(&e);assert(!p.valid && compatibility_path.begin_alias_pool==2);
    e.t.pool_flags=0;e.t.serial=0;p=compatibility_begin(&e);assert(!p.valid && compatibility_path.begin_no_source==1);
    compatibility_commit(&e,&p);assert(compatibility_path.commit_unbracketed==1);
    e.t.serial=2;e.pub.addr|=UINT64_C(1)<<63;
    p=compatibility_begin(&e);assert(!p.valid && compatibility_path.begin_no_source==2);e.pub.addr=100;
    position=(RcRegistryPosition){0};p=compatibility_begin(&e);assert(!p.valid && compatibility_path.begin_position==1);
    position=(RcRegistryPosition){1,6};effects_allowed=0;
    p=compatibility_begin(&e);assert(!p.valid && compatibility_path.begin_feedback==1);effects_allowed=1;
    mapping=0;p=compatibility_begin(&e);assert(!p.valid && compatibility_path.begin_serial==1);mapping=1;
    p=compatibility_begin(&e);assert(p.valid);compatibility_commit(&e,&p);
    assert(source_binding_compat_receipt(&e,NULL));
    position=(RcRegistryPosition){0};assert(!source_binding_compat_receipt(&e,NULL));
    position=(RcRegistryPosition){2,6};assert(!source_binding_compat_receipt(&e,NULL));
    position=(RcRegistryPosition){1,4};assert(!source_binding_compat_receipt(&e,NULL));
    position=(RcRegistryPosition){1,6};e.checked=0;
    assert(source_binding_compat_receipt(&e,NULL)); // Notification overflow alone cannot invalidate a scoped receipt.
    e.checked=submit_epoch;
    // The production release/takeback hooks invalidate receipts, including reuse within the same submission.
    // A new raw-range acquisition cannot recreate readiness without an actual successful preparation bracket.
    compatibility_invalidate(&e);e.released=1;
    assert(!source_binding_compat_receipt(&e,NULL) && !e.compatibility_position.scene);
    assert(!compatibility_begin(&e).valid);
    e.released=0;compatibility_invalidate(&e);
    assert(!source_binding_compat_receipt(&e,NULL));
    p=compatibility_begin(&e);assert(p.valid);compatibility_commit(&e,&p);
    assert(source_binding_compat_receipt(&e,NULL));
    assert(compatibility_path.use_position && compatibility_path.use_scope && compatibility_path.use_backwards);
    assert(compatibility_path.begin_ready && compatibility_path.begin_registry && compatibility_path.commit_ready &&
           compatibility_path.commit_registry && compatibility_path.commit_mapping && compatibility_path.commit_protection);
    assert(compatibility_path.begins==compatibility_path.begin_position+compatibility_path.begin_alias_pool+
           compatibility_path.begin_no_source+compatibility_path.begin_registration+compatibility_path.begin_feedback+compatibility_path.begin_serial+
           compatibility_path.begin_registry+compatibility_path.begin_ready);
    assert(compatibility_path.commits==compatibility_path.commit_unready+compatibility_path.commit_unbracketed+
           compatibility_path.commit_mapping+compatibility_path.commit_protection+compatibility_path.commit_registry+
           compatibility_path.commit_ready);
    assert(compatibility_path.uses==compatibility_path.use_empty+compatibility_path.use_position+compatibility_path.use_epoch+
           compatibility_path.use_identity+compatibility_path.use_feedback+compatibility_path.use_mapping+
           compatibility_path.use_protection+compatibility_path.use_scope+compatibility_path.use_backwards+
           compatibility_path.use_edge+compatibility_path.use_registry+compatibility_path.use_ready);
    assert(!rc_registry_stats(registry).read_versions);
    assert(binding_path.absent && binding_path.direct_gpu && binding_path.finalized && binding_path.ordinary &&
           binding_path.compatibility && binding_path.same_operation && binding_path.invalid_slot &&
           binding_path.stale_slot && binding_path.image_mismatch && binding_path.not_ready &&
           binding_path.ordinary_refused && binding_path.compatibility_refused);
    assert(binding_path.calls==binding_path.absent+binding_path.direct_gpu+binding_path.finalized+
           binding_path.ordinary+binding_path.compatibility+binding_path.same_operation+
           binding_path.invalid_slot+binding_path.stale_slot+binding_path.image_mismatch+binding_path.not_ready+
           binding_path.ordinary_refused+binding_path.compatibility_refused);
    // A derived descriptor proves both its exact copy/layer and the underlying source association.
    position=(RcRegistryPosition){9,2};current=1;
    TextureFlat3D copy={.image.image=17,.source=7,.upload=3,.revision=4,.ordered=1,.position=position,.layer=2};
    Tex derived={.pub.image=7,.pub.gpu_owned=1,.uploaded=1,.storage=1,.n_up=3,.content_revision=4,.flat3d=&copy};
    tslots[1]=(TexSlot){.used=1,.ce=&derived,.cgen=agen[1],.chas_view=1,.cview={1,2},.t.mirrored=1};
    assert(vk_texture_binding_receipt_locked(1,17,&next)==VK_TEXTURE_BINDING_SOURCE&&next.scene==9&&next.ordinal==3);
    assert(vk_texture_binding_schedule_locked(1,17,&next)==VK_TEXTURE_BINDING_SOURCE&&next.scene==9&&next.ordinal==3);
    derived.content_revision++; // final binding preserves the snapshot preceding this operation's target writes
    assert(vk_texture_binding_current_locked(1,17));
    position.ordinal=4;assert(!vk_texture_binding_current_locked(1,17));position.ordinal=2;
    tslots[1].cview.min_layer=1;assert(!vk_texture_binding_current_locked(1,17));tslots[1].cview.min_layer=2;
    derived.pub.gpu_owned=0;derived.content_revision=0;copy.ordered=0;copy.revision=0;
    derived.source_access._content.ordered.handle.id=1;
    derived.source_access._read.position=position;
    derived.source_access._read.sealed_through=(RcRegistryPosition){9,20};
    derived.source_access._read.next_write=(RcRegistryPosition){9,12};
    assert(vk_texture_binding_receipt_locked(1,17,&next)==VK_TEXTURE_BINDING_SOURCE&&next.ordinal==11);
    current=0;assert(!vk_texture_binding_current_locked(1,17));current=1;
    derived.n_up++;assert(!vk_texture_binding_current_locked(1,17));derived.n_up--;
    copy.image.gone=1;assert(!vk_texture_binding_current_locked(1,17));
    assert(binding_path.calls==binding_path.absent+binding_path.direct_gpu+binding_path.finalized+
           binding_path.ordinary+binding_path.compatibility+binding_path.same_operation+
           binding_path.invalid_slot+binding_path.stale_slot+binding_path.image_mismatch+binding_path.not_ready+
           binding_path.ordinary_refused+binding_path.compatibility_refused);
    rc_registry_destroy(registry);
    // The production typed preparation bracket belongs to its exact physical owner, even if freshness work
    // replaces that owner before commit. Reject image or hold changes without publishing an old proof.
    registry=rc_registry_create((RcRegistryConfig){8,8,2});assert(registry);
    position=(RcRegistryPosition){1,4};assert(rc_registry_declare_through(registry,(RcRegistryPosition){1,10})==RC_REGISTRY_OK);
    Tex mixed={.pub={.image=44,.addr=500},.t={.L.size=32,.serial=17},.uploaded=1,.storage=1};
    assert(rc_registry_hold_begin(registry,(RcSourceRange){1,500,32},RC_REGISTRY_IMAGE,&mixed.image_authority)==RC_REGISTRY_OK);
    assert(rc_registry_image_hold_bind(registry,mixed.image_authority,44)==RC_REGISTRY_OK);
    p=compatibility_begin(&mixed);assert(p.valid&&p.observation.image_binding.image==44);
    compatibility_commit(&mixed,&p);assert(mixed.compatibility_receipt.observation.image_binding.image==44);
    compatibility_invalidate(&mixed);p=compatibility_begin(&mixed);assert(p.valid);
    mixed.pub.image=45;compatibility_commit(&mixed,&p);assert(!mixed.compatibility_receipt.observation.handle.id);mixed.pub.image=44;
    p=compatibility_begin(&mixed);assert(p.valid);RcRegistryWrite saved_hold=mixed.image_authority;
    mixed.image_authority.id++;compatibility_commit(&mixed,&p);assert(!mixed.compatibility_receipt.observation.handle.id);mixed.image_authority=saved_hold;
    p=compatibility_begin(&mixed);assert(p.valid);mixed.image_authority.index++;
    compatibility_commit(&mixed,&p);assert(!mixed.compatibility_receipt.observation.handle.id);mixed.image_authority=saved_hold;
    p=compatibility_begin(&mixed);assert(p.valid);
    rc_registry_image_hold_forget(registry,mixed.image_authority);
    compatibility_commit(&mixed,&p);assert(!mixed.compatibility_receipt.observation.handle.id);
    assert(!compatibility_begin(&mixed).valid&&compatibility_path.image_commit_changed==3);
    rc_registry_destroy(registry);
    // Pool schedule certificates are distinct from finite source proofs and must discover future additions even
    // when the read remains inside the old seal. They cannot turn a changed source into the old uploaded image.
    registry=rc_registry_create((RcRegistryConfig){8,8,2});assert(registry);
    position=(RcRegistryPosition){1,2};submit_epoch=1;
    assert(rc_registry_declare_through(registry,(RcRegistryPosition){1,4})==RC_REGISTRY_OK);
    e=(Tex){.pub={.image=7,.addr=100},.t={.L.size=64},.storage=1,.uploaded=1};
    tslots[1]=(TexSlot){.used=1,.ce=&e,.cgen=agen[1],.t.mirrored=1};
    assert(rc_registry_source_access_acquire(registry,1,(RcSourceRange){1,100,64},&e.source_access)==RC_REGISTRY_OK);
    RcRegistrySourceProof observed;
    assert(rc_registry_source_access_observe(registry,&e.source_access,1,position,&observed)==RC_REGISTRY_OK);
    int changed;
    assert(rc_registry_source_access_commit(registry,&e.source_access,&observed,0,&changed)==RC_REGISTRY_OK);
    TextureSourceAuthority retained=e.source_access._content;
    assert(vk_texture_binding_schedule_locked(1,7,&next)==VK_TEXTURE_BINDING_SOURCE&&!next.scene);
    assert(!memcmp(&retained,&e.source_access._content,sizeof retained));
    assert(e.source_access._content.ordered.sealed_through.ordinal==4);
    assert(vk_texture_binding_receipt_locked(1,7,&next)==VK_TEXTURE_BINDING_SOURCE&&next.ordinal==5);
    RcRegistryWrite upcoming;
    assert(rc_registry_write_begin_at(registry,(RcSourceRange){1,100,64},RC_REGISTRY_BUFFER,(RcRegistryPosition){1,7},&upcoming)==RC_REGISTRY_OK);
    assert(vk_texture_binding_schedule_locked(1,7,&next)==VK_TEXTURE_BINDING_SOURCE&&next.ordinal==6);
    assert(rc_registry_declare_through(registry,(RcRegistryPosition){1,10})==RC_REGISTRY_OK);
    // The controlled source-current helper receives the renewed finite permission. Schedule discovery itself
    // must not update that input or the content anchor; the production source helper owns permission renewal.
    position.ordinal=6;RcRegistryReadObservation renewed;
    assert(rc_registry_read_renew(registry,&e.source_access._read,submit_epoch,position,&renewed)==RC_REGISTRY_OK);
    e.source_access._read=renewed;
    assert(!vk_texture_binding_schedule_locked(1,7,&next));
    position.ordinal=4;e.source_access._read=retained.ordered;
    assert(rc_registry_write_cancel(registry,upcoming)==RC_REGISTRY_OK);
    assert(vk_texture_binding_schedule_locked(1,7,&next)==VK_TEXTURE_BINDING_SOURCE&&!next.scene);
    assert(!memcmp(&retained,&e.source_access._content,sizeof retained));
    assert(rc_registry_cpu_changed(registry,(RcSourceRange){1,100,1})==RC_REGISTRY_OK);
    assert(!vk_texture_binding_schedule_locked(1,7,&next));

    RcRegistrySourceAccess access[2]={0};
    RcSourceRange intervals[2]={{3,100,32},{4,100,32}};
    RcRegistrySourceProof source_proof[2];
    for(unsigned i=0;i<2;i++) {
        assert(rc_registry_source_access_acquire(registry,17,intervals[i],&access[i])==RC_REGISTRY_OK);
    }
    TextureSourceMapping backing={.access=access,.count=2};
    assert(rc_registry_source_access_observe_extents(registry,access,2,1,position,source_proof)==RC_REGISTRY_OK);
    assert(rc_registry_source_access_commit_extents(registry,access,source_proof,2,0,&changed)==RC_REGISTRY_OK);
    e=(Tex){.pub={.image=18,.addr=100},.t={.L.size=64,.serial=17},.source_canonical=1,
        .source_mapping=&backing,.storage=1,.uploaded=1};
    assert(rc_registry_source_access_read_extents(registry,access,2,1,position,&e.source_vector_read)==RC_REGISTRY_OK);
    registration=(TextureRegistration){.handle={78,1},.publication=2,.descriptor.serial=17,.source=&e};
    tslots[1]=(TexSlot){.used=1,.ce=(Tex *)(uintptr_t)1,.cgen=0,.pool_owner=9,.registration=&registration};
    selected=(VkTextureReadKey){.registration=78,.native_slot=1,.canonical=1};
    assert(vk_texture_resolved_schedule_locked(selected,18,&next)==VK_TEXTURE_BINDING_SOURCE&&!next.scene);
    assert(vk_texture_binding_schedule_locked(1,18,&next)==VK_TEXTURE_BINDING_SOURCE&&!next.scene);
    FakeInstance source_instance={.value.id=401};
    TextureMaterial source_material={.resident.instance=&source_instance},copy_owner={0};
    TextureFlat3D selected_child={.image.image=44,.material=&copy_owner,.source_instance=401,
        .source=18,.position=position,.layer=1};
    e.material=&source_material;e.flat3d=&selected_child;
    registration.view=(view_t){.set_layers=1,.min_layer=1};
    assert(vk_texture_flat3d_selected_current_locked(selected,44));
    assert(vk_texture_resolved_receipt_locked(selected,44,&next)==VK_TEXTURE_BINDING_SOURCE);
    assert(vk_texture_resolved_schedule_locked(selected,44,&next)==VK_TEXTURE_BINDING_SOURCE&&!next.scene);
    source_instance.value.id++;
    assert(!vk_texture_resolved_receipt_locked(selected,44,&next));
    source_instance.value.id--;registration.view.min_layer=2;
    assert(!vk_texture_resolved_receipt_locked(selected,44,&next));
    registration.view.min_layer=1;tslots[1].finalized=1;
    assert(!vk_texture_resolved_receipt_locked(selected,44,&next));
    tslots[1].finalized=0;
    assert(rc_registry_write_begin_at(registry,intervals[1],RC_REGISTRY_BUFFER,(RcRegistryPosition){1,11},&upcoming)==RC_REGISTRY_OK);
    assert(vk_texture_resolved_schedule_locked(selected,18,&next)==VK_TEXTURE_BINDING_SOURCE&&next.ordinal==10);
    assert(rc_registry_write_cancel(registry,upcoming)==RC_REGISTRY_OK);
    assert(rc_registry_cpu_changed(registry,intervals[1])==RC_REGISTRY_OK);
    assert(!vk_texture_resolved_schedule_locked(selected,18,&next));
    assert(!vk_texture_resolved_schedule_locked(selected,44,&next));
    rc_registry_destroy(registry);
    puts("texture binding receipts: ordered boundaries, schedule certificates, exact finalized images, GPU contents and compatibility positions passed");
}
'''

def function(source, name):
    signature = re.search(r"^(?:static )?[A-Za-z_][\w \t*]*\b" + re.escape(name) + r"\([^;{}]*\)\s*\{", source, re.M)
    if not signature:
        raise ValueError(name)
    return source[signature.start():source.index("\n}\n", signature.end()) + 3]


def main():
    source = (ROOT / "runtime/vk_texture.c").read_text()
    start = source.index("static int source_binding_receipt(")
    mapping = "static void pool_mapping_source(const Tex *entry) { (void)entry; }\n"
    bindings = mapping + "\n".join(function(source, name) for name in (
        "source_binding_receipt", "texture_binding_entry", "vk_texture_binding_receipt_locked",
        "vk_texture_binding_current_locked", "texture_read_key_slot", "texture_read_key_source",
        "texture_read_derived_current", "vk_texture_flat3d_selected_current_locked", "texture_read_key_image",
        "texture_read_key_compatibility", "vk_texture_resolved_receipt_locked", "texture_slot_registration",
        "texture_binding_schedule", "vk_texture_binding_schedule_locked", "vk_texture_resolved_schedule_locked",
        "source_binding_compat_current",
    ))
    compatibility_start = source.index("typedef struct {\n    RcRegistryBindingObservation observation;")
    compatibility_end = source.index("static void make_fresh(Tex *e)", compatibility_start)
    acquire_start = source.index("static RcRegistryResult source_compatibility_acquire(Tex *e)")
    acquire_end = source.index("\n}\n", acquire_start) + 3
    anchor_start = source.index("static const RcRegistrySourceAccess *source_access_anchor(")
    anchor_end = source.index("static uint64_t texture_owned_bytes(", anchor_start)
    entry_start = source.index("static Tex *entry_of_mode(")
    entry_end = source.index("\n}\n", entry_start)
    takeback_start = source.index("    if (e->released) {", entry_start, entry_end)
    takeback_end = source.index("\n    }", takeback_start, entry_end)
    assert "texture_resume_use(e);" in source[takeback_start:takeback_end]
    resume_start = source.index("static void texture_resume_use(Tex *e) {")
    resume_end = source.index("\n}\n", resume_start)
    resume_event = source.index("flight_ev(e, FLIGHT_TEX_TAKEBACK", resume_start, resume_end)
    assert "compatibility_invalidate(e);" in source[resume_start:resume_event]
    release_start = source.index("static void mark_released(Tex *e)")
    assert "compatibility_invalidate(e);" in source[release_start:source.index("e->released = 1;", release_start)]
    ordered_start = source.index("static int source_binding_ordered(const Tex *e)")
    ordered_end = source.index("static TextureBindingProof source_binding_proof(", ordered_start)
    derived_start = source.index("static int flat3d_current(")
    derived_end = source.index("\n}\n", derived_start) + 3
    receipt_start = source.index("static int source_binding_compat_receipt(Tex *e,")
    proof_end = source.index("} TextureBindingProof;") + len("} TextureBindingProof;")
    proof_start = source.rfind("typedef struct {", 0, proof_end)
    helpers_start = source.index("static TextureBindingProof source_binding_proof(")
    helpers_end = source.index("static RcRegistryResult source_binding_mapped_permission(", helpers_start)
    stamp_start = source.index("static int source_stamp_equal(")
    stamp_end = source.index("\n}\n", stamp_start) + 3
    counters_end = source.index("} binding_path;") + len("} binding_path;")
    counters_start = source.rfind("static struct {", 0, counters_end)
    counters = source[counters_start:counters_end]
    counters_end = source.index("} compatibility_path;") + len("} compatibility_path;")
    counters_start = source.rfind("static struct {", 0, counters_end)
    counters += source[counters_start:counters_end]
    compiler = Path(os.environ.get("CC", "C:/Users/Andrew/AppData/Local/Programs/2NX2Navi/toolchain/msys64/ucrt64/bin/clang.exe"))
    env = dict(os.environ, PATH=str(compiler.parent) + os.pathsep + os.environ.get("PATH", ""))
    work = ROOT / "build/test"
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="binding-receipt-", dir=work) as directory:
        code = Path(directory) / "check.c"
        marker = GENERATED_MARKER.decode() if isinstance(GENERATED_MARKER, bytes) else GENERATED_MARKER
        prefix = PREFIX.replace("/* BINDING_PROOF_TYPE */", source[proof_start:proof_end])
        code.write_text("// " + marker + "\n" + prefix + counters + source[anchor_start:anchor_end] + source[acquire_start:acquire_end] +
                        source[compatibility_start:compatibility_end] +
                        source[ordered_start:ordered_end] + source[derived_start:derived_end] +
                        source[stamp_start:stamp_end] + source[helpers_start:helpers_end] +
                        source[receipt_start:start] + bindings + MAIN)
        exe = code.with_suffix(".exe")
        subprocess.run([str(compiler), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", "-I", str(ROOT / "runtime"),
                        str(code), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)

if __name__ == "__main__":
    main()

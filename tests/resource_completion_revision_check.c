// Completion changes source bytes only where execution actually wrote, independently of reservation topology.
#include <assert.h>
#include <stdio.h>
#define RC_REGISTRY_SINGLE_THREADED 1
#include "../runtime/resource_core_registry.c"
#include "../runtime/resource_texture_source.h"

#define OK(x) assert((x) == RC_REGISTRY_OK)
static RcSourceRange range(uint64_t a, uint64_t n) { return (RcSourceRange){1,a,n}; }
static RcRegistryPosition pos(uint64_t scene, uint64_t ordinal) { return (RcRegistryPosition){scene,ordinal}; }
static RcRegistry *make(unsigned writers) {
    RcRegistry *r=rc_registry_create((RcRegistryConfig){64,writers,4}); assert(r); return r;
}
static RcRegistryHandle source(RcRegistry *r, uint64_t a) {
    RcRegistryHandle h; OK(rc_registry_source(r,a,range(a,64),&h)); return h;
}
static RcRegistryWrite declare(RcRegistry *r, uint64_t scene) {
    RcRegistryWrite w;
    OK(rc_registry_write_begin_at(r,range(0,UINT64_MAX),RC_REGISTRY_IMAGE,pos(scene,3),&w));
    OK(rc_registry_image_producer_add(r,w,pos(scene,3)));
    OK(rc_registry_declare_through(r,pos(scene,4)));
    return w;
}
static void submit(RcRegistry *r, RcRegistryWrite w, uint64_t scene) {
    OK(rc_registry_image_producer_resolve(r,w,pos(scene,3),NULL,0));
    OK(rc_registry_handoff_through(r,pos(scene,3)));
    OK(rc_registry_write_submit(r,w,scene));
}
static RcRegistryObservation observe(RcRegistry *r,RcRegistryHandle h,uint64_t scope) {
    RcRegistryObservation o; OK(rc_registry_observe(r,h,scope,&o)); return o;
}
static void repeated_precise_gpu_completion(void) {
    enum { SOURCES=32, FRAMES=120 };
    RcRegistry *r=make(8); RcRegistryHandle h[SOURCES]; uint64_t versions[SOURCES]={0};
    RcSourceRange gpu=range(UINT64_C(0x8000001000),128);
    for(unsigned i=0;i<SOURCES;i++)h[i]=source(r,0x10000+256*i);
    for(unsigned frame=1;frame<=FRAMES;frame++) {
        RcRegistryWrite w=declare(r,frame);
        for(unsigned i=0;i<SOURCES;i++) {
            RcRegistryReadObservation o; RcRegistryReadVersion version;
            OK(rc_registry_observe_at(r,h[i],frame,pos(frame,2),&o));
            assert(o.content_revision==1 && o.next_write.ordinal==3);
            RcPreparedInput input={.content=i+1,.representation=2,.policy=3,.payload=i+100,.kind=RC_PROVIDER_CAPTURE};
            OK(rc_registry_read_commit(r,&o,&input,&version));
            if(!versions[i])versions[i]=version.id;
            assert(version.id==versions[i]);
        }
        submit(r,w,frame);
        OK(rc_registry_write_complete_effects(r,w,&gpu,1,&gpu,1));
        for(unsigned i=0;i<SOURCES;i++) {
            TextureSourceAuthority a={.legacy=observe(r,h[i],frame)},b={0};
            assert(a.legacy.content_revision==1);
            OK(texture_authority_observe(r,h[i],frame,pos(frame,4),&b));
            assert(texture_authority_same(&a,&b));
            RcPreparedInput input={.content=i+1,.representation=2,.policy=3,.payload=i+100,.kind=RC_PROVIDER_CAPTURE};
            RcRegistryReadVersion version; OK(rc_registry_read_commit(r,&b.ordered,&input,&version));
            assert(version.id==versions[i]);
        }
        OK(rc_registry_handoff_through(r,pos(frame,4)));
    }
    assert(rc_registry_stats(r).read_versions==SOURCES && rc_registry_stats(r).writer_foreign==1);
    rc_registry_destroy(r);
}
static void actual_content_and_cancellation(void) {
    RcRegistry *r=make(8); RcRegistryHandle hit=source(r,1000),other=source(r,2000);
    RcRegistryObservation prior=observe(r,other,1); RcRegistryWrite w=declare(r,1);
    OK(rc_registry_write_cancel(r,w));
    RcRegistryObservation now=observe(r,other,1);
    assert(now.revision!=prior.revision && now.content_revision==prior.content_revision);
    w=declare(r,2);submit(r,w,2);RcSourceRange written=range(1008,8);
    OK(rc_registry_write_complete_effects(r,w,&written,1,NULL,0));
    assert(observe(r,hit,2).content_revision>1 && observe(r,other,2).content_revision==1);
    assert(!rc_registry_stats(r).writers);
    w=declare(r,3);submit(r,w,3);
    OK(rc_registry_write_complete_effects(r,w,&written,1,&written,1));
    assert(rc_registry_observe(r,hit,3,&now)==RC_REGISTRY_BLOCKED);
    assert(observe(r,other,3).content_revision==1);
    OK(rc_registry_reconcile_cpu(r,written));
    assert(observe(r,hit,3).content_revision>1 && observe(r,other,3).content_revision==1);
    // The old API intentionally retains whole-declaration content semantics.
    w=declare(r,4);submit(r,w,4);
    OK(rc_registry_write_complete_ranges(r,w,NULL,0));
    assert(observe(r,other,4).content_revision>1);
    rc_registry_destroy(r);
}
static void refusals_keep_reservation(void) {
    RcRegistry *r=make(1);RcRegistryHandle h=source(r,1000);RcRegistryWrite w=declare(r,1);submit(r,w,1);
    RcSourceRange written[]={range(0x8000001000,8),range(0x8000002000,8)};
    RcSourceRange invalid=range(1000,8);uint64_t rev=r->records[h.index].content_revision;
    assert(rc_registry_write_complete_effects(r,w,written,1,&invalid,1)==RC_REGISTRY_INVALID);
    assert(rc_registry_write_complete_effects(r,w,written,2,written,2)==RC_REGISTRY_CAPACITY);
    assert(writer_of(r,w) && writer_of(r,w)->state==WRITE_SUBMITTED);
    assert(r->records[h.index].content_revision==rev);
    assert(rc_registry_observe(r,h,1,&(RcRegistryObservation){0})==RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_complete_effects(r,w,written,2,NULL,0));
    assert(observe(r,h,1).content_revision==rev);
    assert(rc_registry_write_complete_effects(r,w,written,2,NULL,0)==RC_REGISTRY_STALE);
    rc_registry_destroy(r);
}
int main(void) {
    repeated_precise_gpu_completion();actual_content_and_cancellation();refusals_keep_reservation();
    puts("completion revisions: 120 frames preserve 32 unrelated exact versions; actual writes, uncertainty, cancellation and atomic refusal passed");
}

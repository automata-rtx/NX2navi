// Installed-binding schedule permission stays separate from finite source-byte observations.
#include <assert.h>
#include <stdio.h>
#define RC_REGISTRY_SINGLE_THREADED 1
#include "../runtime/resource_core_registry.c"
#define OK(x) assert((x) == RC_REGISTRY_OK)
#define NO(x) assert((x) != RC_REGISTRY_OK)
static RcRegistryPosition at(uint64_t n) { return (RcRegistryPosition){1,n}; }
static RcSourceRange bytes(uint64_t a,uint64_t n) { return (RcSourceRange){1,a,n}; }
static RcRegistry *make(uint32_t records) {
    RcRegistry *r=rc_registry_create((RcRegistryConfig){records,16,4});assert(r);return r;
}
static RcRegistryHandle source(RcRegistry *r,uint64_t serial,uint64_t address) {
    RcRegistryHandle h;OK(rc_registry_source(r,serial,bytes(address,16),&h));return h;
}
static RcRegistryBindingReceipt prepare(RcRegistry *r,RcRegistryHandle h,uint64_t position) {
    RcRegistryBindingObservation o;RcRegistryBindingReceipt b;
    OK(rc_registry_binding_begin(r,h,1,at(position),&o));OK(rc_registry_binding_commit(r,&o,&b));return b;
}
static void soft_events_identify_sources_without_content_invalidation(void) {
    RcRegistry *r=make(32);RcRegistryHandle a=source(r,77,100),b=source(r,88,108),c=source(r,99,200);
    RcRegistryHandle descriptor;OK(rc_registry_register(r,1,33,bytes(100,16),1,&descriptor));
    uint64_t soft=rc_registry_schedule_change_position(r),hard=rc_registry_change_position(r);
    uint64_t epoch=rc_registry_ordered_epoch(r),av=record_of(r,a)->content_revision,bv=record_of(r,b)->content_revision;
    uint64_t cv=record_of(r,c)->revision,invalidations=rc_registry_stats(r).invalidations;
    RcRegistryWrite w;OK(rc_registry_write_begin_at(r,bytes(100,16),RC_REGISTRY_IMAGE,at(5),&w));
    // One existing registration traversal touches the two overlapping sources and descriptor exactly once.
    assert(rc_registry_stats(r).invalidations-invalidations==3);
    RcRegistryScheduleChange out[8];int lost=0;uint32_t n=rc_registry_schedule_changes(r,&soft,out,8,&lost);
    assert(n==2&&!lost&&rc_registry_change_position(r)==hard&&rc_registry_ordered_epoch(r)==epoch);
    for(uint32_t i=0;i<n;i++) {
        RcRegistryHandle h=out[i].handle;Record *e=record_of(r,h);
        assert((h.id==a.id||h.id==b.id)&&out[i].object_serial==e->object);
        assert(rc_range_equal(out[i].range,e->interval.range)&&out[i].topology_revision==e->revision);
        assert(out[i].writer.id==w.id&&out[i].writer.index==w.index&&out[i].position.ordinal==5);
        assert(out[i].kind==RC_REGISTRY_SCHEDULE_ADD);
    }
    assert(record_of(r,a)->content_revision==av&&record_of(r,b)->content_revision==bv&&record_of(r,c)->revision==cv);
    OK(rc_registry_image_producer_add(r,w,at(5)));
    assert(rc_registry_schedule_change_position(r)==soft); // First position already emitted by writer creation.
    OK(rc_registry_image_producer_add(r,w,at(9)));
    n=rc_registry_schedule_changes(r,&soft,out,8,&lost);assert(n==2&&!lost);
    assert(out[0].position.ordinal==9&&out[1].position.ordinal==9);
    OK(rc_registry_image_producer_resolve(r,w,at(9),NULL,1));
    OK(rc_registry_write_cancel(r,w));
    assert(rc_registry_schedule_change_position(r)==soft&&rc_registry_change_position(r)==hard);
    // Expansion retains its hard mapping/content semantics and also informs schedule consumers in one walk.
    OK(rc_registry_write_begin_at(r,bytes(100,16),RC_REGISTRY_BUFFER,at(11),&w));
    soft=rc_registry_schedule_change_position(r);hard=rc_registry_change_position(r);
    OK(rc_registry_write_expand(r,w,bytes(200,16)));
    n=rc_registry_schedule_changes(r,&soft,out,8,&lost);assert(n==3&&!lost);
    assert(rc_registry_change_position(r)==hard+3);
    for(uint32_t i=0;i<n;i++)assert(out[i].kind==RC_REGISTRY_SCHEDULE_EXPAND&&out[i].position.ordinal==11);
    assert(record_of(r,a)->content_revision>av&&record_of(r,b)->content_revision>bv);
    rc_registry_destroy(r);
}
static void certificates_refresh_inside_old_prefix_without_widening_bytes(void) {
    RcRegistry *r=make(16);RcRegistryHandle h=source(r,1,100);
    OK(rc_registry_declare_through(r,at(4)));
    RcRegistryReadObservation old;OK(rc_registry_observe_at(r,h,1,at(2),&old));
    RcRegistryBindingReceipt prepared=prepare(r,h,2);
    RcRegistryReadSchedule read;RcRegistryBindingSchedule binding;
    OK(rc_registry_read_schedule(r,&old,1,at(2),&read));
    OK(rc_registry_binding_schedule(r,&prepared,1,at(2),&binding));
    assert(!read.next_read.scene&&!binding.next_read.scene&&read.epoch==binding.epoch);
    assert(read.observation.sealed_through.ordinal==4&&binding.receipt.observation.next_read.ordinal==5);
    RcRegistryWrite w;OK(rc_registry_write_begin_at(r,bytes(100,16),RC_REGISTRY_BUFFER,at(9),&w));
    assert(read.epoch!=rc_registry_authority_epoch(r)); // Old aggregate cannot acknowledge this event silently.
    // Query is still inside the captured seal: it must refresh topology rather than use finite revalidate alone.
    OK(rc_registry_read_schedule(r,&read.observation,1,at(4),&read));
    OK(rc_registry_binding_schedule(r,&binding.receipt,1,at(4),&binding));
    assert(read.next_read.ordinal==8&&binding.next_read.ordinal==8);
    assert(read.observation.next_write.ordinal==9&&read.observation.sealed_through.ordinal==4);
    assert(binding.receipt.observation.position.ordinal==2&&binding.receipt.observation.next_read.ordinal==5);
    OK(rc_registry_declare_through(r,at(6)));
    NO(rc_registry_read_revalidate(r,&read.observation,1,at(6),&(RcRegistryReadObservation){0}));
    NO(rc_registry_binding_revalidate(r,&binding.receipt,1,at(6),&(RcRegistryBindingReceipt){0}));
    assert(rc_registry_position_open(r,at(6))&&!rc_registry_position_open(r,at(8)));
    // Installed-binding certificate remains usable only under the external event/epoch/open-position contract.
    assert(rc_registry_authority_epoch(r)==read.epoch&&rc_registry_position_compare(at(6),read.next_read)<0);
    OK(rc_registry_handoff_through(r,at(6)));assert(!rc_registry_position_open(r,at(6)));
    NO(rc_registry_read_schedule(r,&read.observation,1,at(6),&read));
    NO(rc_registry_binding_schedule(r,&binding.receipt,1,at(6),&binding));
    rc_registry_destroy(r);
}
static void compatibility_keeps_preparation_origin_and_real_edges(void) {
    RcRegistry *r=make(16);RcRegistryHandle h=source(r,1,100);RcRegistryWrite prior,future;
    OK(rc_registry_write_begin_at(r,bytes(100,16),RC_REGISTRY_BUFFER,at(1),&prior));
    OK(rc_registry_write_begin_at(r,bytes(100,16),RC_REGISTRY_BUFFER,at(9),&future));
    OK(rc_registry_declare_through(r,at(10)));
    RcRegistryBindingReceipt old=prepare(r,h,2);RcRegistryBindingSchedule schedule;
    assert(old.observation.next_read.ordinal==8&&old.observation.sealed_through.ordinal==10);
    OK(rc_registry_binding_schedule(r,&old,1,at(6),&schedule));
    assert(schedule.next_read.ordinal==8&&schedule.receipt.observation.position.ordinal==2);
    NO(rc_registry_binding_schedule(r,&old,1,at(8),&schedule));
    NO(rc_registry_binding_schedule(r,&old,1,at(10),&schedule));
    // A crossed real edge remains conservative even when canceled; callers can explicitly prepare again.
    OK(rc_registry_write_cancel(r,future));NO(rc_registry_binding_schedule(r,&old,1,at(10),&schedule));
    OK(rc_registry_binding_schedule(r,&old,1,at(6),&schedule));assert(!schedule.next_read.scene);
    assert(schedule.receipt.observation.position.ordinal==2);
    NO(rc_registry_binding_schedule(r,&old,2,at(6),&schedule));
    NO(rc_registry_binding_schedule(r,&old,1,(RcRegistryPosition){2,6},&schedule));
    NO(rc_registry_binding_schedule(r,&old,1,at(1),&schedule));
    rc_registry_destroy(r);
}
static void newly_exposed_producers_require_permission_or_preparation(void) {
    RcRegistry *r=make(16);RcRegistryHandle h=source(r,1,100);RcRegistryWrite w;
    OK(rc_registry_declare_through(r,at(2)));
    RcRegistryReadObservation old;OK(rc_registry_observe_at(r,h,1,at(2),&old));
    RcRegistryBindingReceipt prepared=prepare(r,h,2);RcRegistryReadSchedule read;RcRegistryBindingSchedule binding;
    OK(rc_registry_write_begin_at(r,bytes(100,16),RC_REGISTRY_BUFFER,at(5),&w));
    OK(rc_registry_declare_through(r,at(6)));
    NO(rc_registry_read_schedule(r,&old,1,at(6),&read));
    NO(rc_registry_binding_schedule(r,&prepared,1,at(6),&binding));
    OK(rc_registry_write_cancel(r,w));
    OK(rc_registry_read_schedule(r,&old,1,at(6),&read));
    OK(rc_registry_binding_schedule(r,&prepared,1,at(6),&binding));
    assert(read.observation.content_revision==old.content_revision&&binding.receipt.observation.position.ordinal==2);
    OK(rc_registry_write_begin(r,bytes(100,16),RC_REGISTRY_UNKNOWN,&w));
    NO(rc_registry_read_schedule(r,&read.observation,1,at(6),&read));
    NO(rc_registry_binding_schedule(r,&binding.receipt,1,at(6),&binding));
    assert(rc_registry_change_position(r)>0); // Unpositioned uncertainty is still a hard change.
    rc_registry_destroy(r);
}
static void repeated_image_positions_remain_exact_and_do_not_certify_cpu_bytes(void) {
    RcRegistry *r=make(16);RcRegistryHandle h=source(r,1,100);RcRegistryWrite w;
    RcRegistryImageBinding image={.image=99};
    OK(rc_registry_hold_begin(r,bytes(100,16),RC_REGISTRY_IMAGE,&image.hold));
    OK(rc_registry_image_hold_bind(r,image.hold,image.image));
    OK(rc_registry_write_begin_at(r,bytes(100,16),RC_REGISTRY_IMAGE,at(1),&w));
    OK(rc_registry_image_producer_add(r,w,at(1)));OK(rc_registry_image_producer_resolve(r,w,at(1),&image,0));
    OK(rc_registry_declare_through(r,at(2)));
    RcRegistryBindingObservation o;RcRegistryBindingReceipt old;RcRegistryBindingSchedule schedule;
    OK(rc_registry_binding_begin_image(r,h,1,at(2),image,&o));OK(rc_registry_binding_commit(r,&o,&old));
    OK(rc_registry_binding_schedule(r,&old,1,at(2),&schedule));assert(!schedule.next_read.scene);
    uint64_t cursor=rc_registry_schedule_change_position(r);
    OK(rc_registry_image_producer_add(r,w,at(7)));OK(rc_registry_declare_through(r,at(8)));
    RcRegistryScheduleChange change;int lost=0;
    assert(rc_registry_schedule_changes(r,&cursor,&change,1,&lost)==1&&!lost&&change.position.ordinal==7);
    OK(rc_registry_binding_schedule(r,&old,1,at(4),&schedule));assert(schedule.next_read.ordinal==6);
    NO(rc_registry_binding_schedule(r,&old,1,at(6),&schedule));
    OK(rc_registry_image_producer_resolve(r,w,at(7),&image,0));
    NO(rc_registry_binding_schedule(r,&old,1,at(8),&schedule)); // Same image still requires a new preparation.
    NO(rc_registry_observe_at(r,h,1,at(4),&(RcRegistryReadObservation){0}));
    OK(rc_registry_binding_begin_image(r,h,1,at(8),image,&o));OK(rc_registry_binding_commit(r,&o,&old));
    OK(rc_registry_binding_schedule(r,&old,1,at(8),&schedule));
    rc_registry_image_hold_forget(r,image.hold);NO(rc_registry_binding_schedule(r,&old,1,at(8),&schedule));
    rc_registry_destroy(r);
}
static void exact_completion_preserves_unwritten_source_schedule(void) {
    RcRegistry *r=make(16);RcRegistryHandle a=source(r,1,100),b=source(r,2,200);RcRegistryWrite w;
    OK(rc_registry_write_begin_at(r,bytes(0,UINT64_MAX),RC_REGISTRY_IMAGE,at(3),&w));
    uint64_t soft=rc_registry_schedule_change_position(r),hard=rc_registry_change_position(r);
    uint64_t av=record_of(r,a)->content_revision;
    OK(rc_registry_declare_through(r,at(4)));OK(rc_registry_handoff_through(r,at(4)));
    OK(rc_registry_write_submit(r,w,1));RcSourceRange written=bytes(200,16);
    OK(rc_registry_write_complete_effects(r,w,&written,1,NULL,0));
    assert(rc_registry_schedule_change_position(r)==soft&&record_of(r,a)->content_revision==av);
    RcRegistryChange change;int lost=0;
    assert(rc_registry_changes(r,&hard,&change,1,&lost)==1&&!lost&&change.handle.id==b.id);
    rc_registry_destroy(r);
}
static void loss_identity_and_poison_cannot_acknowledge_old_permission(void) {
    RcRegistry *r=make(4);RcRegistryHandle h=source(r,1,100);uint64_t cursor=0;
    for(unsigned i=0;i<7;i++) {
        RcRegistryWrite w;OK(rc_registry_write_begin_at(r,bytes(100,16),RC_REGISTRY_BUFFER,at(3+2*i),&w));
        OK(rc_registry_write_cancel(r,w));
    }
    RcRegistryScheduleChange out[8];int lost=0;
    assert(rc_registry_schedule_changes(r,&cursor,out,8,&lost)==4&&lost);
    assert(out[0].position.ordinal==9&&out[3].position.ordinal==15&&cursor==7);
    cursor=99;assert(rc_registry_schedule_changes(r,&cursor,out,8,&lost)==4&&lost&&cursor==7);
    OK(rc_registry_declare_through(r,at(16)));RcRegistryReadObservation old;
    OK(rc_registry_observe_at(r,h,1,at(16),&old));RcRegistryReadSchedule schedule;
    RcRegistryBindingReceipt binding=prepare(r,h,16);RcRegistryBindingSchedule bs;
    OK(rc_registry_range_gone(r,bytes(100,16)));
    NO(rc_registry_read_schedule(r,&old,1,at(16),&schedule));
    NO(rc_registry_binding_schedule(r,&binding,1,at(16),&bs));
    RcRegistryHandle fresh=source(r,2,100);assert(fresh.id!=h.id);
    OK(rc_registry_observe_at(r,fresh,1,at(16),&old));
    OK(rc_registry_finalize_object(r,2,bytes(100,16)));
    NO(rc_registry_read_schedule(r,&old,1,at(16),&schedule));
    rc_registry_fail_closed(r);assert(!rc_registry_position_open(r,at(16)));
    assert(!rc_registry_schedule_changes(r,&cursor,out,8,&lost)&&lost);
    rc_registry_destroy(r);
}
int main(void) {
    soft_events_identify_sources_without_content_invalidation();
    certificates_refresh_inside_old_prefix_without_widening_bytes();
    compatibility_keeps_preparation_origin_and_real_edges();
    newly_exposed_producers_require_permission_or_preparation();
    repeated_image_positions_remain_exact_and_do_not_certify_cpu_bytes();
    exact_completion_preserves_unwritten_source_schedule();
    loss_identity_and_poison_cannot_acknowledge_old_permission();
    puts("schedule certificates: exact source events, finite byte proofs, producer edges, identity and loss passed");
}

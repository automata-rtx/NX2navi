// Finite ordered permission survives later declarations without certifying any undeclared read position.
#include <assert.h>
#include <stdio.h>
#define RC_REGISTRY_SINGLE_THREADED 1
#include "../runtime/resource_core_registry.c"
#define OK(x) assert((x) == RC_REGISTRY_OK)
#define NO(x) assert((x) != RC_REGISTRY_OK)
static RcRegistryPosition at(uint64_t n) { return (RcRegistryPosition){1,n}; }
static RcSourceRange bytes(uint64_t a,uint64_t n) { return (RcSourceRange){1,a,n}; }
static RcRegistry *make(void) { RcRegistry *r=rc_registry_create((RcRegistryConfig){128,16,4});assert(r);return r; }
static RcRegistryHandle source(RcRegistry *r,uint64_t addr) {
    RcRegistryHandle h;OK(rc_registry_source(r,addr,bytes(addr,16),&h));return h;
}
static void later_declarations_do_not_fan_out_or_extend_old_proofs(void) {
    RcRegistry *r=make();RcRegistryHandle h[64];
    for(unsigned i=0;i<64;i++)h[i]=source(r,100+i*32);
    OK(rc_registry_declare_through(r,at(2))); // The read phase itself is a valid inclusive prefix.
    RcRegistryReadObservation read,fresh;
    RcRegistryBindingObservation binding;
    RcRegistryBindingReceipt receipt,next;
    RcRegistryObservation legacy;
    OK(rc_registry_observe_at(r,h[0],1,at(2),&read));
    OK(rc_registry_observe(r,h[0],1,&legacy));
    OK(rc_registry_binding_begin(r,h[0],1,at(2),&binding));
    OK(rc_registry_binding_commit(r,&binding,&receipt));
    assert(read.sealed_through.ordinal==2 && !read.next_write.scene);
    assert(binding.sealed_through.ordinal==2 && binding.next_read.ordinal==3);
    uint64_t epoch=rc_registry_ordered_epoch(r),old_epoch=rc_registry_authority_epoch(r);
    uint64_t changes=rc_registry_change_position(r),version=record_of(r,h[0])->content_revision;
    for(unsigned i=0;i<300;i++) {
        RcRegistryWrite w;uint64_t p=3+2*i;
        OK(rc_registry_write_begin_at(r,bytes(0,UINT64_MAX),RC_REGISTRY_IMAGE,at(p),&w));
        OK(rc_registry_declare_through(r,at(p+1)));
        assert(!rc_registry_observation_current(r,&legacy));
        assert(rc_registry_ordered_epoch(r)==epoch && rc_registry_change_position(r)==changes);
        OK(rc_registry_read_revalidate(r,&read,1,at(2),&fresh));
        OK(rc_registry_binding_revalidate(r,&receipt,1,at(2),&next));
        NO(rc_registry_read_revalidate(r,&read,1,at(4),&fresh));
        NO(rc_registry_binding_revalidate(r,&receipt,1,at(4),&next));
        OK(rc_registry_write_cancel(r,w));
    }
    assert(rc_registry_authority_epoch(r)>old_epoch && record_of(r,h[0])->content_revision==version);
    assert(rc_registry_ordered_epoch(r)==epoch && rc_registry_change_position(r)==changes);
    OK(rc_registry_observe_at(r,h[0],1,at(600),&fresh));
    assert(fresh.sealed_through.ordinal==602);
    OK(rc_registry_read_revalidate(r,&fresh,1,at(602),&read)); // Last sealed read is inclusive.
    NO(rc_registry_read_revalidate(r,&fresh,1,at(604),&read));
    rc_registry_destroy(r);
}
static void cancellations_keep_real_producer_deadlines(void) {
    RcRegistry *r=make();RcRegistryHandle h=source(r,100);RcRegistryWrite w;
    OK(rc_registry_write_begin_at(r,bytes(100,16),RC_REGISTRY_BUFFER,at(7),&w));
    OK(rc_registry_declare_through(r,at(10)));
    RcRegistryReadObservation read,fresh;RcRegistryBindingObservation binding;
    RcRegistryBindingReceipt receipt,next;
    OK(rc_registry_observe_at(r,h,1,at(2),&read));
    OK(rc_registry_binding_begin(r,h,1,at(2),&binding));OK(rc_registry_binding_commit(r,&binding,&receipt));
    assert(read.next_write.ordinal==7 && binding.next_read.ordinal==6);
    OK(rc_registry_write_cancel(r,w));
    OK(rc_registry_read_revalidate(r,&read,1,at(4),&fresh));
    OK(rc_registry_binding_revalidate(r,&receipt,1,at(4),&next));
    NO(rc_registry_binding_revalidate(r,&receipt,1,at(6),&next));
    NO(rc_registry_read_revalidate(r,&read,1,at(8),&fresh));
    OK(rc_registry_observe_at(r,h,1,at(8),&fresh)); // Only a fresh query can extend permission.
    OK(rc_registry_binding_begin(r,h,1,at(8),&binding));assert(binding.next_read.ordinal==11);
    rc_registry_destroy(r);
}
static void hard_events_invalidate_even_without_content_changes(void) {
    RcRegistry *r=make();RcRegistryHandle h=source(r,100);RcRegistryWrite w;
    OK(rc_registry_write_begin_at(r,bytes(100,16),RC_REGISTRY_IMAGE,at(7),&w));
    OK(rc_registry_image_producer_add(r,w,at(7)));OK(rc_registry_declare_through(r,at(10)));
    RcRegistryReadObservation read;OK(rc_registry_observe_at(r,h,1,at(2),&read));
    uint64_t content=read.content_revision,epoch=rc_registry_ordered_epoch(r),changes=rc_registry_change_position(r);
    rc_registry_image_producer_invalidate(r,w);
    assert(record_of(r,h)->content_revision==content && rc_registry_ordered_epoch(r)>epoch);
    assert(rc_registry_change_position(r)>changes && !rc_registry_read_observation_current(r,&read));
    OK(rc_registry_write_cancel(r,w));OK(rc_registry_observe_at(r,h,1,at(2),&read));
    OK(rc_registry_write_begin(r,bytes(100,16),RC_REGISTRY_UNKNOWN,&w));
    assert(!rc_registry_read_observation_current(r,&read));OK(rc_registry_write_cancel(r,w));
    OK(rc_registry_observe_at(r,h,1,at(2),&read));
    OK(rc_registry_finalize_object(r,100,bytes(100,16)));assert(!rc_registry_read_observation_current(r,&read));
    rc_registry_fail_closed(r);assert(!rc_registry_authority_epoch(r)&&!rc_registry_ordered_epoch(r));
    rc_registry_destroy(r);
}
static void precise_retirement_notifies_only_written_sources(void) {
    RcRegistry *r=make();RcRegistryHandle a=source(r,100),b=source(r,200);RcRegistryWrite w;
    OK(rc_registry_write_begin_at(r,bytes(0,UINT64_MAX),RC_REGISTRY_IMAGE,at(3),&w));
    OK(rc_registry_declare_through(r,at(4)));OK(rc_registry_handoff_through(r,at(4)));
    OK(rc_registry_write_submit(r,w,1));uint64_t cursor=rc_registry_change_position(r);
    uint64_t ar=record_of(r,a)->ordered_revision,br=record_of(r,b)->ordered_revision;
    RcSourceRange written=bytes(200,16);OK(rc_registry_write_complete_effects(r,w,&written,1,NULL,0));
    assert(record_of(r,a)->ordered_revision==ar && record_of(r,b)->ordered_revision>br);
    RcRegistryChange out[4];int overflow=0;assert(rc_registry_changes(r,&cursor,out,4,&overflow)==1&&!overflow);
    assert(out[0].handle.id==b.id);
    rc_registry_destroy(r);
}
static void deadline_overflow_is_explicit(void) {
    assert(!rc_registry_sealed_deadline((RcRegistryPosition){0}).scene);
    RcRegistryPosition edge=rc_registry_sealed_deadline((RcRegistryPosition){7,UINT64_MAX});
    assert(edge.scene==8&&edge.ordinal==1);
    assert(!rc_registry_sealed_deadline((RcRegistryPosition){UINT64_MAX,UINT64_MAX}).scene);
    RcRegistry *r=make();RcRegistryHandle h=source(r,100);
    OK(rc_registry_declare_through(r,(RcRegistryPosition){UINT64_MAX,UINT64_MAX}));
    NO(rc_registry_observe_at(r,h,1,at(2),&(RcRegistryReadObservation){0}));
    NO(rc_registry_binding_begin(r,h,1,at(2),&(RcRegistryBindingObservation){0}));
    rc_registry_destroy(r);
}
static void renewal_queries_current_authority_without_recapturing_bytes(void) {
    RcRegistry *r=make();RcRegistryHandle h=source(r,100);
    OK(rc_registry_declare_through(r,at(2)));
    RcRegistryReadObservation old,fresh,bad;
    OK(rc_registry_observe_at(r,h,1,at(2),&old));
    OK(rc_registry_declare_through(r,at(4)));
    NO(rc_registry_read_revalidate(r,&old,1,at(4),&fresh));
    OK(rc_registry_read_renew(r,&old,1,at(4),&fresh));
    assert(fresh.content_revision==old.content_revision && fresh.position.ordinal==4 && fresh.sealed_through.ordinal==4);
    assert(old.position.ordinal==2 && old.sealed_through.ordinal==2);
    NO(rc_registry_read_renew(r,&old,2,at(4),&fresh));
    NO(rc_registry_read_renew(r,&old,1,(RcRegistryPosition){2,4},&fresh));
    NO(rc_registry_read_renew(r,&old,1,at(1),&fresh));
    bad=old;bad.sealed_through=at(1);NO(rc_registry_read_renew(r,&bad,1,at(4),&fresh));
    RcRegistryWrite w;OK(rc_registry_write_begin_at(r,bytes(100,16),RC_REGISTRY_BUFFER,at(5),&w));
    OK(rc_registry_declare_through(r,at(6)));
    NO(rc_registry_read_renew(r,&old,1,at(6),&fresh));
    OK(rc_registry_write_cancel(r,w));OK(rc_registry_read_renew(r,&old,1,at(6),&fresh));
    assert(fresh.content_revision==old.content_revision);
    OK(rc_registry_cpu_changed(r,bytes(100,1)));NO(rc_registry_read_renew(r,&old,1,at(6),&fresh));
    rc_registry_destroy(r);
}
static void compatibility_renewal_cannot_accept_newly_crossed_producers(void) {
    RcRegistry *r=make();RcRegistryHandle h=source(r,100);RcRegistryWrite prior,w;
    OK(rc_registry_write_begin_at(r,bytes(100,16),RC_REGISTRY_BUFFER,at(1),&prior));
    OK(rc_registry_declare_through(r,at(2)));
    RcRegistryBindingObservation observation;RcRegistryBindingReceipt old,fresh,bad;
    OK(rc_registry_binding_begin(r,h,1,at(2),&observation));OK(rc_registry_binding_commit(r,&observation,&old));
    OK(rc_registry_write_begin_at(r,bytes(100,16),RC_REGISTRY_BUFFER,at(7),&w));
    OK(rc_registry_declare_through(r,at(10)));
    OK(rc_registry_binding_renew(r,&old,1,at(4),&fresh));
    assert(fresh.observation.position.ordinal==2 && fresh.observation.next_read.ordinal==6);
    NO(rc_registry_binding_renew(r,&old,1,at(8),&bad));
    NO(rc_registry_binding_renew(r,&fresh,1,at(8),&bad)); // Real-producer expiry needs actual preparation.
    OK(rc_registry_write_cancel(r,w));OK(rc_registry_binding_renew(r,&old,1,at(8),&fresh));
    assert(fresh.observation.position.ordinal==2 && fresh.observation.next_read.ordinal==11);
    NO(rc_registry_binding_renew(r,&old,2,at(8),&bad));
    NO(rc_registry_binding_renew(r,&old,1,(RcRegistryPosition){2,8},&bad));
    bad=old;bad.observation.sealed_through=at(1);NO(rc_registry_binding_renew(r,&bad,1,at(8),&fresh));
    OK(rc_registry_cpu_changed(r,bytes(100,1)));NO(rc_registry_binding_renew(r,&old,1,at(8),&fresh));
    rc_registry_destroy(r);
}
static void repeated_image_positions_need_new_preparation_when_crossed(void) {
    RcRegistry *r=make();RcRegistryHandle h=source(r,100);RcRegistryWrite w;
    RcRegistryImageBinding image={.image=99};
    OK(rc_registry_hold_begin(r,bytes(100,16),RC_REGISTRY_IMAGE,&image.hold));
    OK(rc_registry_image_hold_bind(r,image.hold,image.image));
    OK(rc_registry_write_begin_at(r,bytes(100,16),RC_REGISTRY_IMAGE,at(1),&w));
    OK(rc_registry_image_producer_add(r,w,at(1)));OK(rc_registry_image_producer_resolve(r,w,at(1),&image,0));
    OK(rc_registry_declare_through(r,at(2)));
    RcRegistryBindingObservation observation;RcRegistryBindingReceipt old,fresh;
    OK(rc_registry_binding_begin_image(r,h,1,at(2),image,&observation));OK(rc_registry_binding_commit(r,&observation,&old));
    OK(rc_registry_image_producer_add(r,w,at(5)));OK(rc_registry_declare_through(r,at(6)));
    NO(rc_registry_binding_renew(r,&old,1,at(4),&fresh)); // Newly exposed unresolved output of this read phase.
    OK(rc_registry_image_producer_resolve(r,w,at(5),NULL,1));
    OK(rc_registry_binding_renew(r,&old,1,at(4),&fresh));
    assert(fresh.observation.position.ordinal==2 && fresh.observation.sealed_through.ordinal==6);
    OK(rc_registry_image_producer_add(r,w,at(9)));OK(rc_registry_declare_through(r,at(10)));
    OK(rc_registry_image_producer_resolve(r,w,at(9),&image,0));
    NO(rc_registry_binding_renew(r,&fresh,1,at(10),&old)); // Even exact same-image writes do not silently reprepare.
    rc_registry_destroy(r);
}
static void unchanged_topology_renews_without_interval_work(void) {
    RcRegistry *r=make();RcRegistryHandle h=source(r,100);RcRegistryWrite unrelated;
    OK(rc_registry_write_begin_at(r,bytes(10000,16),RC_REGISTRY_BUFFER,at(3001),&unrelated));
    OK(rc_registry_declare_through(r,at(2)));
    RcRegistryReadObservation read;
    RcRegistryBindingObservation observation;RcRegistryBindingReceipt binding;
    OK(rc_registry_observe_at(r,h,1,at(2),&read));
    OK(rc_registry_binding_begin(r,h,1,at(2),&observation));OK(rc_registry_binding_commit(r,&observation,&binding));
    assert(read.topology_revision && binding.observation.topology_revision && !binding.observation.next_producer.scene);
    uint64_t visits=rc_registry_stats(r).overlap_visits;
    for(unsigned i=2;i<=1000;i++) {
        OK(rc_registry_declare_through(r,at(i*2)));
        OK(rc_registry_read_renew(r,&read,1,at(i*2),&read));
        OK(rc_registry_binding_renew(r,&binding,1,at(i*2),&binding));
        assert(read.sealed_through.ordinal==i*2 && binding.observation.next_read.ordinal==i*2+1);
        assert(binding.observation.position.ordinal==2);
    }
    RcRegistryStats stats=rc_registry_stats(r);
    assert(stats.overlap_visits==visits && stats.read_renew_fast==999 && stats.binding_renew_fast==999);
    assert(!stats.read_renew_queries && !stats.binding_renew_queries);
    OK(rc_registry_cpu_changed(r,bytes(100,1)));
    NO(rc_registry_read_renew(r,&read,1,at(2000),&(RcRegistryReadObservation){0}));
    NO(rc_registry_binding_renew(r,&binding,1,at(2000),&(RcRegistryBindingReceipt){0}));
    assert(rc_registry_stats(r).read_renew_fast==999 && rc_registry_stats(r).binding_renew_fast==999);
    rc_registry_destroy(r);
}
static void hidden_future_deadline_survives_fast_prefix_extensions(void) {
    RcRegistry *r=make();RcRegistryHandle h=source(r,100);RcRegistryWrite w;
    OK(rc_registry_write_begin_at(r,bytes(100,16),RC_REGISTRY_BUFFER,at(11),&w));
    OK(rc_registry_declare_through(r,at(2)));
    RcRegistryReadObservation read,fresh;
    RcRegistryBindingObservation observation;RcRegistryBindingReceipt binding,next;
    OK(rc_registry_observe_at(r,h,1,at(2),&read));
    OK(rc_registry_binding_begin(r,h,1,at(2),&observation));OK(rc_registry_binding_commit(r,&observation,&binding));
    assert(read.next_write.ordinal==11 && observation.next_producer.ordinal==10 && observation.next_read.ordinal==3);
    uint64_t visits=rc_registry_stats(r).overlap_visits;
    OK(rc_registry_declare_through(r,at(8)));
    OK(rc_registry_read_renew(r,&read,1,at(8),&read));
    OK(rc_registry_binding_renew(r,&binding,1,at(8),&binding));
    assert(rc_registry_stats(r).overlap_visits==visits && binding.observation.next_read.ordinal==9);
    assert(binding.observation.next_producer.ordinal==10 && read.next_write.ordinal==11);
    OK(rc_registry_declare_through(r,at(12)));
    NO(rc_registry_binding_renew(r,&binding,1,at(10),&next));
    NO(rc_registry_read_renew(r,&read,1,at(12),&fresh));
    OK(rc_registry_write_cancel(r,w));
    OK(rc_registry_read_renew(r,&read,1,at(12),&fresh));
    OK(rc_registry_binding_renew(r,&binding,1,at(12),&next));
    assert(!fresh.next_write.scene && !next.observation.next_producer.scene && next.observation.next_read.ordinal==13);
    assert(rc_registry_stats(r).read_renew_queries==2 && rc_registry_stats(r).binding_renew_queries==2);
    rc_registry_destroy(r);
}
static void changed_topology_queries_new_producers(void) {
    RcRegistry *r=make();RcRegistryHandle h=source(r,100);RcRegistryWrite w;
    OK(rc_registry_declare_through(r,at(2)));
    RcRegistryReadObservation old,fresh;OK(rc_registry_observe_at(r,h,1,at(2),&old));
    RcRegistryBindingObservation observation;RcRegistryBindingReceipt binding,next;
    OK(rc_registry_binding_begin(r,h,1,at(2),&observation));OK(rc_registry_binding_commit(r,&observation,&binding));
    OK(rc_registry_write_begin_at(r,bytes(100,16),RC_REGISTRY_BUFFER,at(9),&w));
    OK(rc_registry_declare_through(r,at(6)));
    OK(rc_registry_read_renew(r,&old,1,at(6),&fresh));
    OK(rc_registry_binding_renew(r,&binding,1,at(6),&next));
    assert(fresh.next_write.ordinal==9 && fresh.topology_revision!=old.topology_revision);
    assert(next.observation.next_producer.ordinal==8 && next.observation.next_read.ordinal==7);
    assert(rc_registry_stats(r).read_renew_queries==1 && rc_registry_stats(r).binding_renew_queries==1);
    assert(!rc_registry_stats(r).read_renew_fast && !rc_registry_stats(r).binding_renew_fast);
    rc_registry_destroy(r);
}
int main(void) {
    later_declarations_do_not_fan_out_or_extend_old_proofs();
    cancellations_keep_real_producer_deadlines();hard_events_invalidate_even_without_content_changes();
    precise_retirement_notifies_only_written_sources();deadline_overflow_is_explicit();
    renewal_queries_current_authority_without_recapturing_bytes();
    compatibility_renewal_cannot_accept_newly_crossed_producers();
    repeated_image_positions_need_new_preparation_when_crossed();
    unchanged_topology_renews_without_interval_work();
    hidden_future_deadline_survives_fast_prefix_extensions();changed_topology_queries_new_producers();
    puts("ordered prefix: finite permissions, soft topology, hard changes and exact completion passed");
}

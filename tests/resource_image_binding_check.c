// Exercise exact-image compatibility through the production registry, without Vulkan or game data.
#include <assert.h>
#include <stdio.h>
#define RC_REGISTRY_SINGLE_THREADED 1
#include "../runtime/resource_core_registry.c"

#define OK(x) assert((x) == RC_REGISTRY_OK)
#define REFUSED(x) assert((x) != RC_REGISTRY_OK)
static RcSourceRange bytes(uint64_t lo, uint64_t size) { return (RcSourceRange){1, lo, size}; }
static RcRegistryPosition at(uint64_t ordinal) { return (RcRegistryPosition){1, ordinal}; }
static RcRegistry *make(uint32_t records) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){records, 32, 4}); assert(r); return r;
}
static RcRegistryHandle source(RcRegistry *r) {
    RcRegistryHandle h; OK(rc_registry_source(r, 7, bytes(100, 64), &h)); return h;
}
static RcRegistryImageBinding owner(RcRegistry *r, uint64_t image, RcSourceRange range) {
    RcRegistryImageBinding binding = {.image = image};
    OK(rc_registry_hold_begin(r, range, RC_REGISTRY_IMAGE, &binding.hold));
    OK(rc_registry_image_hold_bind(r, binding.hold, image));
    return binding;
}
static RcRegistryWrite pending(RcRegistry *r, uint64_t ordinal) {
    RcRegistryWrite w; OK(rc_registry_write_begin_at(r, bytes(100, 64), RC_REGISTRY_IMAGE, at(ordinal), &w));
    return w;
}
static void separate_typed_compatibility_and_future_edges(void) {
    RcRegistry *r = make(32); RcRegistryHandle h = source(r);
    RcRegistryImageBinding image = owner(r, 11, bytes(100, 64));
    RcRegistryWrite w = pending(r, 3);
    OK(rc_registry_image_producer_add(r, w, at(3)));
    OK(rc_registry_image_producer_add(r, w, at(9)));
    OK(rc_registry_declare_through(r, at(20)));
    RcRegistryBindingObservation observation;
    REFUSED(rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation)); // Earlier command unresolved.
    OK(rc_registry_image_producer_resolve(r, w, at(3), &image, 0));
    OK(rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation));
    assert(observation.image_binding.image == image.image && observation.image_binding.hold.id == image.hold.id);
    assert(observation.next_read.scene == 1 && observation.next_read.ordinal == 8);
    RcRegistryBindingReceipt receipt, fresh;
    OK(rc_registry_binding_commit(r, &observation, &receipt));
    OK(rc_registry_binding_revalidate(r, &receipt, 1, at(6), &fresh));
    REFUSED(rc_registry_binding_revalidate(r, &receipt, 1, at(8), &fresh));
    REFUSED(rc_registry_binding_begin(r, h, 1, at(4), &observation));
    RcRegistryReadObservation cpu;
    REFUSED(rc_registry_observe_at(r, h, 1, at(4), &cpu));
    REFUSED(rc_registry_observe(r, h, 1, &(RcRegistryObservation){0}));
    // Resolution of a later command does not widen this earlier receipt: its already-recorded edge remains.
    OK(rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation));
    OK(rc_registry_image_producer_resolve(r, w, at(9), &image, 0));
    OK(rc_registry_binding_commit(r, &observation, &fresh));
    assert(fresh.observation.next_read.ordinal == 8);
    OK(rc_registry_binding_begin_image(r, h, 1, at(10), image, &observation));
    assert(observation.next_read.scene == 1 && observation.next_read.ordinal == 21);
    OK(rc_registry_binding_commit(r, &observation, &fresh));
    REFUSED(rc_registry_binding_revalidate(r, &fresh, 1, at(6), &receipt));
    rc_registry_destroy(r);
}
static void first_reference_must_be_declared(void) {
    RcRegistry *r = make(16); RcRegistryHandle h = source(r);
    RcRegistryImageBinding image = owner(r, 11, bytes(100, 64));
    RcRegistryWrite w = pending(r, 3);
    OK(rc_registry_declare_through(r, at(10)));
    RcRegistryBindingObservation observation;
    REFUSED(rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation));
    // A schedule cannot be repaired retroactively after earlier readers have been declared.
    REFUSED(rc_registry_image_producer_add(r, w, at(3)));
    REFUSED(rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation));
    rc_registry_destroy(r);
}
static void future_append_preserves_a_sealed_receipt(void) {
    RcRegistry *r = make(16); RcRegistryHandle h = source(r);
    RcRegistryImageBinding image = owner(r, 11, bytes(100, 64)); RcRegistryWrite w = pending(r, 3);
    OK(rc_registry_image_producer_add(r, w, at(3)));
    OK(rc_registry_image_producer_resolve(r, w, at(3), &image, 0));
    OK(rc_registry_declare_through(r, at(4)));
    RcRegistryBindingObservation observation; RcRegistryBindingReceipt receipt;
    OK(rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation));
    assert(observation.next_read.scene == 1 && observation.next_read.ordinal == 5);
    OK(rc_registry_binding_commit(r, &observation, &receipt));
    OK(rc_registry_image_producer_add(r, w, at(9)));
    OK(rc_registry_binding_commit(r, &observation, &(RcRegistryBindingReceipt){0}));
    OK(rc_registry_binding_revalidate(r, &receipt, 1, at(4), &(RcRegistryBindingReceipt){0}));
    OK(rc_registry_declare_through(r, at(10)));
    REFUSED(rc_registry_binding_revalidate(r, &receipt, 1, at(6), &(RcRegistryBindingReceipt){0}));
    OK(rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation));
    assert(observation.next_read.scene == 1 && observation.next_read.ordinal == 8);
    rc_registry_destroy(r);
}
static void all_shared_references_and_cancellation(void) {
    for (int mode = 0; mode < 4; ++mode) {
        RcRegistry *r = make(16); RcRegistryHandle h = source(r);
        RcRegistryImageBinding image = owner(r, 11, bytes(100, 64));
        RcRegistryWrite w = pending(r, 3);
        OK(rc_registry_image_producer_add(r, w, at(3)));
        OK(rc_registry_image_producer_add(r, w, at(3)));
        OK(rc_registry_declare_through(r, at(10)));
        RcRegistryBindingObservation observation;
        OK(rc_registry_image_producer_resolve(r, w, at(3), &image, 0));
        REFUSED(rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation));
        if (mode == 0) OK(rc_registry_image_producer_resolve(r, w, at(3), &image, 0));
        if (mode == 1) OK(rc_registry_image_producer_resolve(r, w, at(3), NULL, 1));
        if (mode == 2) OK(rc_registry_image_producer_resolve(r, w, at(3), NULL, 0));
        if (mode == 3) {
            RcRegistryImageBinding different = owner(r, 22, bytes(100, 64));
            OK(rc_registry_image_producer_resolve(r, w, at(3), &different, 0));
        }
        RcRegistryResult result = rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation);
        if (mode < 2) assert(result == RC_REGISTRY_OK); else assert(result != RC_REGISTRY_OK);
        REFUSED(rc_registry_image_producer_resolve(r, w, at(3), &image, 0));
        rc_registry_destroy(r);
    }
    RcRegistry *r = make(16); RcRegistryHandle h = source(r);
    RcRegistryImageBinding image = owner(r, 11, bytes(100, 64)); RcRegistryWrite w = pending(r, 3);
    OK(rc_registry_image_producer_add(r, w, at(3))); OK(rc_registry_image_producer_add(r, w, at(9)));
    OK(rc_registry_image_producer_resolve(r, w, at(3), NULL, 1));
    OK(rc_registry_image_producer_resolve(r, w, at(9), NULL, 1));
    OK(rc_registry_declare_through(r, at(10)));
    RcRegistryBindingObservation observation;
    OK(rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation));
    assert(observation.next_read.scene == 1 && observation.next_read.ordinal == 11);
    rc_registry_destroy(r);
}
static void exact_live_owner_and_other_aliases(void) {
    RcRegistry *r = make(16); RcRegistryHandle h = source(r);
    RcRegistryImageBinding image = owner(r, 11, bytes(100, 64));
    OK(rc_registry_declare_through(r, at(10)));
    RcRegistryBindingObservation observation;
    OK(rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation));
    RcRegistryImageBinding wrong = image; wrong.image++;
    REFUSED(rc_registry_binding_begin_image(r, h, 1, at(4), wrong, &observation));
    REFUSED(rc_registry_image_hold_bind(r, image.hold, wrong.image));
    wrong = image; wrong.hold.id++;
    REFUSED(rc_registry_binding_begin_image(r, h, 1, at(4), wrong, &observation));
    RcRegistryBindingReceipt receipt;
    OK(rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation));
    OK(rc_registry_binding_commit(r, &observation, &receipt));
    RcRegistryImageBinding different = owner(r, 22, bytes(120, 4));
    REFUSED(rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation));
    REFUSED(rc_registry_binding_revalidate(r, &receipt, 1, at(4), &(RcRegistryBindingReceipt){0}));
    OK(rc_registry_hold_release(r, different.hold, 1));
    OK(rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation));
    OK(rc_registry_binding_commit(r, &observation, &receipt));
    OK(rc_registry_hold_release(r, image.hold, 1));
    REFUSED(rc_registry_binding_revalidate(r, &receipt, 1, at(4), &(RcRegistryBindingReceipt){0}));
    REFUSED(rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation));
    rc_registry_destroy(r);
    // An IMAGE hold without actual physical-image binding cannot be upgraded by the consumer's assertion.
    r = make(16); h = source(r); image = (RcRegistryImageBinding){.image = 11};
    OK(rc_registry_hold_begin(r, bytes(100, 64), RC_REGISTRY_IMAGE, &image.hold));
    OK(rc_registry_declare_through(r, at(10)));
    REFUSED(rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation));
    rc_registry_destroy(r);
}
static void capacity_late_declaration_and_reused_writer(void) {
    RcRegistry *r = make(4); RcRegistryHandle h = source(r);
    RcRegistryImageBinding image = owner(r, 11, bytes(100, 64)); RcRegistryWrite w = pending(r, 3);
    for (uint64_t p = 3; p <= 9; p += 2) OK(rc_registry_image_producer_add(r, w, at(p)));
    REFUSED(rc_registry_image_producer_add(r, w, at(11)));
    for (uint64_t p = 3; p <= 9; p += 2) OK(rc_registry_image_producer_resolve(r, w, at(p), &image, 0));
    OK(rc_registry_declare_through(r, at(20)));
    RcRegistryBindingObservation observation;
    REFUSED(rc_registry_binding_begin_image(r, h, 1, at(12), image, &observation));
    REFUSED(rc_registry_observe_at(r, h, 1, at(12), &(RcRegistryReadObservation){0}));
    rc_registry_destroy(r);
    r = make(16); h = source(r); image = owner(r, 11, bytes(100, 64)); w = pending(r, 3);
    OK(rc_registry_image_producer_add(r, w, at(3)));
    OK(rc_registry_image_producer_resolve(r, w, at(3), &image, 0));
    OK(rc_registry_declare_through(r, at(4)));
    OK(rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation));
    REFUSED(rc_registry_image_producer_add(r, w, at(3)));
    REFUSED(rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation));
    rc_registry_destroy(r);
    r = make(16); h = source(r); image = owner(r, 11, bytes(100, 64)); w = pending(r, 3);
    OK(rc_registry_image_producer_add(r, w, at(3))); OK(rc_registry_write_cancel(r, w));
    RcRegistryWrite replacement = pending(r, 5); assert(replacement.id != w.id);
    REFUSED(rc_registry_image_producer_resolve(r, w, at(3), &image, 0));
    OK(rc_registry_declare_through(r, at(10)));
    REFUSED(rc_registry_binding_begin_image(r, h, 1, at(6), image, &observation));
    rc_registry_destroy(r);
}
static void sequential_queries_build_only_the_completed_prefix(void) {
    enum { N = 1000 };
    RcRegistry *r = make(N + 8); RcRegistryHandle h = source(r);
    RcRegistryImageBinding image = owner(r, 11, bytes(100, 64)); RcRegistryWrite ticket = pending(r, 3);
    for (uint64_t i = 0; i < N; ++i) OK(rc_registry_image_producer_add(r, ticket, at(3 + 4 * i)));
    OK(rc_registry_declare_through(r, at(4 * N + 2)));
    Writer *w = writer_of(r, ticket); assert(w && w->image_count == N && !w->image_dirty);
    uint64_t prefix_steps = 0;
    for (uint32_t i = 0; i < N; ++i) {
        OK(rc_registry_image_producer_resolve(r, ticket, at(3 + 4 * i), &image, 0));
        assert(w->image_dirty == i); // Resolving this entry cannot rebuild still-future declarations.
        RcRegistryBindingObservation observation;
        OK(rc_registry_binding_begin_image(r, h, 1, at(4 + 4 * i), image, &observation));
        assert(w->image_dirty == i + 1);
        prefix_steps += w->image_dirty - i; // Exact bounds of image_position_prefix's production loop.
        if (i + 1 < N) {
            assert(!w->image_positions[N - 1].prefix_bad && !w->image_positions[N - 1].prefix_active);
            assert(observation.next_read.ordinal == 6 + 4 * i);
        } else assert(observation.next_read.ordinal == 4 * N + 3);
        OK(rc_registry_binding_begin_image(r, h, 1, at(4 + 4 * i), image, &observation));
        assert(w->image_dirty == i + 1); // Another alias at the same point performs no prefix work.
    }
    assert(prefix_steps == N);
    rc_registry_destroy(r);
}
static void physical_replacement_forgets_typed_identity(void) {
    RcRegistry *r = make(16); RcRegistryHandle h = source(r);
    RcRegistryImageBinding image = owner(r, 11, bytes(100, 64));
    OK(rc_registry_declare_through(r, at(20)));
    RcRegistryBindingObservation observation;
    RcRegistryBindingReceipt receipt, fresh;
    OK(rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation));
    OK(rc_registry_binding_commit(r, &observation, &receipt));
    uint64_t epoch = rc_registry_authority_epoch(r);
    rc_registry_image_hold_forget(r, image.hold);
    assert(rc_registry_authority_epoch(r) != epoch);
    epoch = rc_registry_authority_epoch(r);
    rc_registry_image_hold_forget(r, image.hold);
    assert(rc_registry_authority_epoch(r) == epoch); // Idempotent, without an invalidation storm.
    assert(!rc_registry_image_binding_current(r, image));
    REFUSED(rc_registry_binding_revalidate(r, &receipt, 1, at(6), &fresh));
    REFUSED(rc_registry_image_hold_bind(r, image.hold, 22));
    REFUSED(rc_registry_image_hold_bind(r, image.hold, 11)); // A recycled numeric image cannot revive this hold.
    REFUSED(rc_registry_binding_begin_image(r, h, 1, at(6), image, &observation));
    assert(rc_registry_stats(r).writer_holds == 1);
    REFUSED(rc_registry_observe_at(r, h, 1, at(6), &(RcRegistryReadObservation){0}));
    OK(rc_registry_hold_release(r, image.hold, 1)); // Actual readback/replacement, independent of typed invalidation.
    RcRegistryImageBinding replacement = owner(r, 11, bytes(100, 64));
    assert(replacement.hold.id != image.hold.id);
    OK(rc_registry_binding_begin_image(r, h, 1, at(6), replacement, &observation));
    REFUSED(rc_registry_binding_revalidate(r, &receipt, 1, at(6), &fresh));
    rc_registry_destroy(r);
}
static void diagnostic_refusals_are_bounded_and_keep_provenance(void) {
    RcRegistry *r = make(32); RcRegistryHandle h = source(r);
    RcRegistryImageBinding image = owner(r, 11, bytes(100, 64));
    RcRegistryWrite ticket = pending(r, 3);
    OK(rc_registry_image_producer_add(r, ticket, at(3)));
    OK(rc_registry_declare_through(r, at(20)));
    RcRegistryBindingObservation observation;
    REFUSED(rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation));
    RcRegistryDiagnostics d;
    rc_registry_diagnostics(r, &d);
    assert(d.counts[RC_REGISTRY_DIAG_IMAGE_PREFIX] == 1);
    const RcRegistryDiagnosticSample *s = &d.samples[RC_REGISTRY_DIAG_IMAGE_PREFIX][0];
    assert(s->writer.id == ticket.id && s->producer_unresolved == 1 && s->prefix_bad);
    assert(s->requested.offset == 100 && s->read_position.ordinal == 4 && s->selected.image == 11);
    OK(rc_registry_image_producer_resolve(r, ticket, at(3), &image, 0));
    OK(rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation));
    OK(rc_registry_write_cancel(r, ticket));
    OK(rc_registry_hold_release(r, image.hold, 1));
    for (unsigned i = 0; i < 8; ++i) {
        image = owner(r, 20 + i, bytes(100, 64));
        rc_registry_image_hold_forget(r, image.hold);
        for (unsigned repeat = 0; repeat < 3; ++repeat)
            REFUSED(rc_registry_binding_begin_image(r, h, 1, at(4), image, &observation));
        OK(rc_registry_hold_release(r, image.hold, 1));
    }
    uint64_t epoch = rc_registry_authority_epoch(r);
    rc_registry_diagnostics(r, &d);
    assert(rc_registry_authority_epoch(r) == epoch);
    assert(d.counts[RC_REGISTRY_DIAG_TYPED_BEGIN] == 26 && d.counts[RC_REGISTRY_DIAG_TYPED_READY] == 1);
    assert(d.counts[RC_REGISTRY_DIAG_HOLD_FORGOTTEN] == 24);
    assert(d.sample_count[RC_REGISTRY_DIAG_HOLD_FORGOTTEN] == RC_REGISTRY_DIAGNOSTIC_SAMPLES);
    for (unsigned i = 0; i < RC_REGISTRY_DIAGNOSTIC_SAMPLES; ++i) {
        s = &d.samples[RC_REGISTRY_DIAG_HOLD_FORGOTTEN][i];
        assert(s->writer_image == 20 + i && (s->flags & RC_REGISTRY_DIAG_FORGOTTEN));
    }
    RcRegistryHandle affected;
    OK(rc_registry_source(r, 88, bytes(1000, 8), &affected));
    OK(rc_registry_write_begin(r, bytes(1000, 128), RC_REGISTRY_IMAGE, &ticket));
    rc_registry_diagnose_missing_image(r, ticket, at(9), 2);
    OK(rc_registry_write_submit(r, ticket, 1));
    OK(rc_registry_write_complete(r, ticket, 0));
    REFUSED(rc_registry_observe(r, affected, 1, &(RcRegistryObservation){0}));
    rc_registry_diagnostics(r, &d);
    assert(d.counts[RC_REGISTRY_DIAG_MISSING_IMAGE] == 1 && d.counts[RC_REGISTRY_DIAG_FOREIGN_IMAGE] == 1);
    assert(d.samples[RC_REGISTRY_DIAG_MISSING_IMAGE][0].prepared_actuals == 2);
    s = &d.samples[RC_REGISTRY_DIAG_SOURCE_FOREIGN_IMAGE][0];
    assert(s->range.offset == 1000 && s->range.size == 128 && s->requested.size == 8);
    assert((s->flags & (RC_REGISTRY_DIAG_SYNTHETIC | RC_REGISTRY_DIAG_COMPLETED)) ==
           (RC_REGISTRY_DIAG_SYNTHETIC | RC_REGISTRY_DIAG_COMPLETED));
    assert(!strcmp(rc_registry_diagnostic_name(RC_REGISTRY_DIAG_HOLD_FORGOTTEN), "hold_forgotten"));
    rc_registry_destroy(r);
}
int main(void) {
    separate_typed_compatibility_and_future_edges();
    first_reference_must_be_declared();
    future_append_preserves_a_sealed_receipt();
    all_shared_references_and_cancellation();
    exact_live_owner_and_other_aliases();
    capacity_late_declaration_and_reused_writer();
    sequential_queries_build_only_the_completed_prefix();
    physical_replacement_forgets_typed_identity();
    diagnostic_refusals_are_bounded_and_keep_provenance();
    puts("image compatibility: exact owner, complete producer positions, cancellation and fail-closed boundaries passed");
}

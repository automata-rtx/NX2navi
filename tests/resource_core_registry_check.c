#include <assert.h>
#include <stdio.h>
#define RC_REGISTRY_SINGLE_THREADED 1
#include "../runtime/resource_core_registry.c"
#include "../runtime/resource_core_lifetime.h"

#define OK(call) assert((call) == RC_REGISTRY_OK)
#define IS(call, result) assert((call) == (result))
static RcSourceRange range(uint64_t address, uint64_t bytes) { return (RcSourceRange){1, address, bytes}; }
static RcRegistry *make(uint32_t records, uint32_t writers, uint32_t scopes) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){records, writers, scopes}); assert(r); return r;
}

static void registration_history_and_captured_old_contents(void) {
    RcRegistry *r = make(16, 8, 4);
    RcRegistryHandle first, replacement;
    OK(rc_registry_register(r, 9, 100, range(1024, 128), 7, &first));
    RcRegistryObservation observation;
    OK(rc_registry_observe(r, first, 1, &observation));
    OK(rc_registry_publish(r, &observation, 101, 201, 301));
    RcRegistryScope old_scope;
    OK(rc_registry_scope_begin(r, &old_scope));
    RcRegistryCapture capture = {0};
    OK(rc_registry_capture(r, first, &capture));
    OK(rc_registry_finalize(r, 9, 100));
    OK(rc_registry_register(r, 9, 101, range(1024, 64), 8, &replacement));
    assert(first.id != replacement.id);
    IS(rc_registry_finalize(r, 9, 100), RC_REGISTRY_STALE);
    RcRegistryRecord old, current;
    OK(rc_registry_resolve(r, 9, old_scope, &old));
    OK(rc_registry_resolve(r, 9, (RcRegistryScope){0}, &current));
    assert(old.handle.id == first.id && old.content == 101 && old.finalized);
    assert(current.handle.id == replacement.id && !current.finalized);
    assert(capture.record.content == 101 && capture.record.range.size == 128);
    assert(!rc_registry_observation_current(r, &observation));
    assert(rc_registry_collect(r, 16) == 0);
    OK(rc_registry_scope_end(r, old_scope));
    assert(rc_registry_collect(r, 16) == 0);
    OK(rc_registry_capture_release(r, capture));
    IS(rc_registry_capture_release(r, capture), RC_REGISTRY_STALE);
    assert(rc_registry_collect(r, 16) == 1);
    IS(rc_registry_get(r, first, &old), RC_REGISTRY_STALE);
    IS(rc_registry_resolve(r, 9, old_scope, &old), RC_REGISTRY_STALE);
    rc_registry_destroy(r);
}

static void interval_authority_covers_new_aliases(void) {
    RcRegistry *r = make(16, 8, 4);
    RcRegistryHandle a, b, unrelated, alias;
    OK(rc_registry_source(r, 1, range(100, 100), &a));
    OK(rc_registry_register(r, 2, 2, range(180, 40), 1, &b));
    OK(rc_registry_source(r, 3, range(300, 20), &unrelated));
    RcRegistryObservation oa, ob, ou;
    OK(rc_registry_observe(r, a, 1, &oa));
    OK(rc_registry_observe(r, b, 1, &ob));
    OK(rc_registry_observe(r, unrelated, 1, &ou));
    RcRegistryWrite writer;
    OK(rc_registry_write_begin(r, range(190, 10), RC_REGISTRY_BUFFER, &writer));
    assert(!rc_registry_observation_current(r, &oa) && !rc_registry_observation_current(r, &ob));
    assert(rc_registry_observation_current(r, &ou));
    OK(rc_registry_source(r, 4, range(195, 2), &alias));
    IS(rc_registry_observe(r, alias, 1, &oa), RC_REGISTRY_BLOCKED);
    IS(rc_registry_reconcile_cpu(r, range(0, 1000)), RC_REGISTRY_BUSY);
    OK(rc_registry_write_submit(r, writer, 22));
    IS(rc_registry_write_cancel(r, writer), RC_REGISTRY_STALE);
    OK(rc_registry_write_complete(r, writer, 0));
    IS(rc_registry_write_complete(r, writer, 1), RC_REGISTRY_STALE);
    IS(rc_registry_observe(r, alias, 2, &oa), RC_REGISTRY_BLOCKED);
    OK(rc_registry_cpu_changed(r, range(195, 1)));
    IS(rc_registry_observe(r, alias, 2, &oa), RC_REGISTRY_BLOCKED);
    OK(rc_registry_reconcile_cpu(r, range(190, 10)));
    OK(rc_registry_observe(r, alias, 2, &oa));
    rc_registry_destroy(r);
}

static void coalescence_and_partial_visibility(void) {
    RcRegistry *r = make(16, 8, 2);
    RcRegistryWrite a, b;
    OK(rc_registry_write_begin(r, range(100, 100), RC_REGISTRY_IMAGE, &a));
    OK(rc_registry_write_submit(r, a, 1)); OK(rc_registry_write_complete(r, a, 0));
    OK(rc_registry_write_begin(r, range(180, 100), RC_REGISTRY_IMAGE, &b));
    OK(rc_registry_write_submit(r, b, 2)); OK(rc_registry_write_complete(r, b, 0));
    assert(rc_registry_stats(r).writers == 1);
    OK(rc_registry_reconcile_cpu(r, range(150, 70)));
    assert(rc_registry_stats(r).writers == 2);
    RcRegistryHandle middle, left, right; RcRegistryObservation o;
    OK(rc_registry_source(r, 1, range(160, 20), &middle));
    OK(rc_registry_source(r, 2, range(120, 20), &left));
    OK(rc_registry_source(r, 3, range(240, 20), &right));
    OK(rc_registry_observe(r, middle, 1, &o));
    IS(rc_registry_observe(r, left, 1, &o), RC_REGISTRY_BLOCKED);
    IS(rc_registry_observe(r, right, 1, &o), RC_REGISTRY_BLOCKED);
    OK(rc_registry_reconcile_cpu(r, range(100, 50)));
    OK(rc_registry_observe(r, left, 2, &o));
    IS(rc_registry_observe(r, right, 2, &o), RC_REGISTRY_BLOCKED);
    rc_registry_destroy(r);

    r = make(4, 1, 1);
    OK(rc_registry_write_begin(r, range(100, 100), RC_REGISTRY_IMAGE, &a));
    OK(rc_registry_write_submit(r, a, 1)); OK(rc_registry_write_complete(r, a, 0));
    IS(rc_registry_reconcile_cpu(r, range(120, 20)), RC_REGISTRY_CAPACITY);
    assert(rc_registry_stats(r).writers == 1 && !rc_registry_stats(r).fail_closed);
    OK(rc_registry_source(r, 1, range(120, 20), &middle));
    IS(rc_registry_observe(r, middle, 1, &o), RC_REGISTRY_BLOCKED);
    rc_registry_destroy(r);
}

static void canceled_writers_and_held_authority(void) {
    RcRegistry *r = make(4, 4, 2); RcRegistryHandle h; RcRegistryObservation old, fresh;
    OK(rc_registry_source(r, 1, range(100, 100), &h));
    OK(rc_registry_observe(r, h, 1, &old));
    RcRegistryWrite canceled, hold;
    OK(rc_registry_write_begin(r, range(100, 20), RC_REGISTRY_BUFFER, &canceled));
    OK(rc_registry_hold_begin(r, range(150, 20), RC_REGISTRY_IMAGE, &hold));
    OK(rc_registry_write_cancel(r, canceled));
    IS(rc_registry_write_cancel(r, canceled), RC_REGISTRY_STALE);
    IS(rc_registry_observe(r, h, 1, &fresh), RC_REGISTRY_BLOCKED);
    IS(rc_registry_write_submit(r, hold, 1), RC_REGISTRY_STALE);
    OK(rc_registry_hold_release(r, hold, 1));
    OK(rc_registry_observe(r, h, 1, &fresh));
    assert(!rc_registry_observation_current(r, &old));
    OK(rc_registry_cpu_changed(r, range(120, 1)));
    IS(rc_registry_publish(r, &fresh, 1, 1, 1), RC_REGISTRY_STALE);
    rc_registry_destroy(r);
}

static void image_hold_extents_follow_native_identity(void) {
    RcInterval intervals[] = {{100, 71, {2, 16, 32}}, {132, 72, {3, 0, 16}}};
    RcRegistryWrite holds[2];
    RcRegistry *r = make(4, 1, 1);
    IS(rc_registry_image_hold_begin_extents(r, 900, intervals, 2, holds), RC_REGISTRY_CAPACITY);
    assert(!holds[0].id && !holds[1].id && !rc_registry_stats(r).writers);
    rc_registry_destroy(r);

    r = make(4, 4, 1);
    RcRegistryHandle first, second, successor;
    RcRegistryObservation probe;
    OK(rc_registry_source(r, 1, intervals[0].range, &first));
    OK(rc_registry_source(r, 2, intervals[1].range, &second));
    OK(rc_registry_source(r, 3, (RcSourceRange){4, 16, 32}, &successor));
    OK(rc_registry_image_hold_begin_extents(r, 900, intervals, 2, holds));
    assert(rc_registry_image_binding_current(r, (RcRegistryImageBinding){900, holds[0]}));
    assert(rc_registry_image_binding_current(r, (RcRegistryImageBinding){900, holds[1]}));
    assert(!rc_registry_image_binding_current(r, (RcRegistryImageBinding){901, holds[0]}));
    assert(!rc_registry_image_hold_covers(r, holds[0], intervals[1].range));
    IS(rc_registry_observe(r, first, 1, &probe), RC_REGISTRY_BLOCKED);
    IS(rc_registry_observe(r, second, 1, &probe), RC_REGISTRY_BLOCKED);
    OK(rc_registry_observe(r, successor, 1, &probe));
    RcRegistryWrite duplicate[] = {holds[0], holds[0]};
    IS(rc_registry_image_hold_release_extents(r, duplicate, 2, 1), RC_REGISTRY_INVALID);
    assert(rc_registry_image_binding_current(r, (RcRegistryImageBinding){900, holds[0]}));
    assert(rc_registry_image_binding_current(r, (RcRegistryImageBinding){900, holds[1]}));
    OK(rc_registry_image_hold_release_extents(r, holds, 2, 0));
    assert(!rc_registry_image_binding_current(r, (RcRegistryImageBinding){900, holds[0]}));
    assert(!rc_registry_image_binding_current(r, (RcRegistryImageBinding){900, holds[1]}));
    IS(rc_registry_observe(r, first, 1, &probe), RC_REGISTRY_BLOCKED);
    IS(rc_registry_observe(r, second, 1, &probe), RC_REGISTRY_BLOCKED);
    OK(rc_registry_observe(r, successor, 1, &probe));
    rc_registry_destroy(r);
}

static void abandoned_writers_preserve_possible_effects(void) {
    RcRegistry *r = make(4, 1, 1); RcRegistryHandle h; RcRegistryObservation observation; RcRegistryWrite writer;
    OK(rc_registry_source(r, 1, range(100, 20), &h));
    OK(rc_registry_write_begin(r, range(100, 20), RC_REGISTRY_BUFFER, &writer));
    OK(rc_registry_write_abandon(r, writer, 0));
    OK(rc_registry_observe(r, h, 1, &observation));
    IS(rc_registry_write_abandon(r, writer, 0), RC_REGISTRY_STALE);
    OK(rc_registry_write_begin(r, range(100, 20), RC_REGISTRY_BUFFER, &writer));
    // The only writer slot is occupied. Disposal must retain foreign coverage without another allocation.
    OK(rc_registry_write_abandon(r, writer, 1));
    assert(rc_registry_stats(r).writers == 1 && !rc_registry_stats(r).fail_closed);
    IS(rc_registry_observe(r, h, 2, &observation), RC_REGISTRY_BLOCKED);
    IS(rc_registry_write_abandon(r, writer, 1), RC_REGISTRY_STALE);
    OK(rc_registry_reconcile_cpu(r, range(100, 20)));
    OK(rc_registry_write_begin(r, range(100, 20), RC_REGISTRY_BUFFER, &writer));
    OK(rc_registry_write_submit(r, writer, 9));
    IS(rc_registry_write_abandon(r, writer, 0), RC_REGISTRY_STALE);
    OK(rc_registry_write_abandon(r, writer, 1));
    IS(rc_registry_observe(r, h, 3, &observation), RC_REGISTRY_BLOCKED);
    OK(rc_registry_reconcile_cpu(r, range(100, 20)));
    OK(rc_registry_observe(r, h, 4, &observation));
    rc_registry_destroy(r);
}

static void capacity_is_backpressure_and_writer_failure_is_closed(void) {
    RcRegistry *r = make(2, 1, 1); RcRegistryHandle a, b, c; RcRegistryScope scope, excess;
    OK(rc_registry_register(r, 1, 1, range(0, 10), 0, &a));
    OK(rc_registry_scope_begin(r, &scope));
    IS(rc_registry_scope_begin(r, &excess), RC_REGISTRY_CAPACITY);
    OK(rc_registry_register(r, 1, 2, range(20, 10), 0, &b));
    RcRegistryCapture kept = {0};
    OK(rc_registry_capture(r, b, &kept));
    IS(rc_registry_register(r, 1, 3, range(40, 10), 0, &c), RC_REGISTRY_CAPACITY);
    RcRegistryRecord old;
    OK(rc_registry_resolve(r, 1, scope, &old)); assert(old.handle.id == a.id);
    OK(rc_registry_capture_release(r, kept));
    OK(rc_registry_register(r, 1, 3, range(40, 10), 0, &c));
    assert(c.id != a.id);
    OK(rc_registry_resolve(r, 1, scope, &old)); assert(old.handle.id == a.id);
    OK(rc_registry_scope_end(r, scope));
    RcRegistryObservation observation;
    OK(rc_registry_observe(r, c, 1, &observation));
    RcRegistryWrite first, refused;
    OK(rc_registry_write_begin(r, range(100, 10), RC_REGISTRY_BUFFER, &first));
    IS(rc_registry_write_begin(r, range(200, 10), RC_REGISTRY_BUFFER, &refused), RC_REGISTRY_CAPACITY);
    assert(!refused.id && !rc_registry_observation_current(r, &observation));
    OK(rc_registry_write_cancel(r, first));
    IS(rc_registry_observe(r, c, 2, &observation), RC_REGISTRY_BLOCKED);
    rc_registry_destroy(r);
}

static void backing_identity_and_mapping_replacement(void) {
    RcRegistry *r = make(8, 4, 2); RcRegistryHandle original, physical_alias, independent;
    RcSourceRange physical = {77, 16, 32};
    OK(rc_registry_source(r, 1, physical, &original));
    OK(rc_registry_register(r, 2, 2, physical, 3, &physical_alias));
    OK(rc_registry_source(r, 3, (RcSourceRange){78, 16, 32}, &independent));
    RcRegistryObservation a, b, c;
    OK(rc_registry_observe(r, original, 1, &a));
    OK(rc_registry_observe(r, physical_alias, 1, &b));
    OK(rc_registry_observe(r, independent, 1, &c));
    OK(rc_registry_cpu_changed(r, (RcSourceRange){77, 20, 1}));
    assert(!rc_registry_observation_current(r, &a) && !rc_registry_observation_current(r, &b));
    assert(rc_registry_observation_current(r, &c));
    OK(rc_registry_range_gone(r, physical));
    IS(rc_registry_observe(r, original, 2, &a), RC_REGISTRY_STALE);
    RcRegistryScope gap;
    OK(rc_registry_scope_begin(r, &gap));
    RcRegistryHandle replacement;
    OK(rc_registry_source(r, 4, physical, &replacement));
    assert(replacement.id != original.id);
    OK(rc_registry_observe(r, replacement, 2, &a));
    OK(rc_registry_register(r, 2, 4, physical, 3, &replacement));
    RcRegistryRecord absent;
    IS(rc_registry_resolve(r, 2, gap, &absent), RC_REGISTRY_STALE);
    OK(rc_registry_scope_end(r, gap));
    rc_registry_destroy(r);
}

static void ready_validation_and_churn_are_not_history_scans(void) {
    RcRegistry *r = make(2048, 8, 2); RcRegistryHandle target = {0};
    for (uint64_t i = 1; i <= 1000; i++) {
        RcRegistryHandle h;
        OK(rc_registry_source(r, i, range(i * 256, 32), &h));
        if (i == 500) target = h;
    }
    RcRegistryObservation o;
    OK(rc_registry_observe(r, target, 1, &o));
    uint64_t visits = rc_registry_stats(r).overlap_visits;
    for (int i = 0; i < 10000; i++) assert(rc_registry_observation_current(r, &o));
    assert(rc_registry_stats(r).overlap_visits == visits);
    OK(rc_registry_cpu_changed(r, range(500 * 256, 1)));
    assert(rc_registry_stats(r).overlap_visits - visits < 100);
    rc_registry_destroy(r);

    r = make(16, 2, 2);
    for (uint64_t i = 1; i <= 10000; i++) {
        RcRegistryHandle h;
        OK(rc_registry_source(r, i, range(i * 256, 16), &h));
        OK(rc_registry_finalize_object(r, i, range(i * 256, 16)));
        rc_registry_collect(r, 16);
        assert(!rc_registry_stats(r).records);
    }
    for (uint32_t i = 0; i < r->key_capacity; i++) assert(!r->keys[i]);
    rc_registry_destroy(r);
}

static void bounded_source_notifications(void) {
    RcRegistry *r = make(8, 8, 2);
    RcRegistryHandle a, b;
    OK(rc_registry_source(r, 11, range(100, 16), &a));
    OK(rc_registry_source(r, 12, range(200, 16), &b));
    uint64_t cursor = rc_registry_change_position(r), epoch = rc_registry_authority_epoch(r);
    OK(rc_registry_cpu_changed(r, range(104, 1)));
    assert(rc_registry_authority_epoch(r) > epoch);
    RcRegistryChange changes[4]; int overflow;
    assert(rc_registry_changes(r, &cursor, changes, 4, &overflow) == 1 && !overflow);
    assert(changes[0].object_serial == 11 && changes[0].range.offset == 100);
    for (int i = 0; i < 20; ++i) OK(rc_registry_cpu_changed(r, range(204, 1)));
    assert(rc_registry_changes(r, &cursor, changes, 4, &overflow) == 4 && overflow);
    for (int i = 0; i < 4; ++i) assert(changes[i].object_serial == 12);
    rc_registry_fail_closed(r);
    assert(!rc_registry_authority_epoch(r));
    rc_registry_destroy(r);
}

static void source_lease_ownership_and_replacement_churn(void) {
    RcRegistry *r = make(4, 2, 2);
    RcRegistrySource a = {0}, b = {0};
    OK(rc_registry_source_acquire(r, 11, range(100, 16), &a));
    RcRegistryHandle first = a.handle;
    for (unsigned i = 0; i < 1000; ++i) OK(rc_registry_source_acquire(r, 11, range(100, 16), &a));
    OK(rc_registry_source_acquire(r, 11, range(100, 16), &b));
    assert(a.id != b.id && a.handle.id == b.handle.id);
    assert(rc_registry_stats(r).source_leases == 2 && rc_registry_stats(r).live_sources == 1);
    RcRegistrySource copied = a;
    OK(rc_registry_source_release(r, &a));
    IS(rc_registry_source_release(r, &copied), RC_REGISTRY_STALE);
    assert(!copied.id && rc_registry_stats(r).source_leases == 1);
    RcRegistryObservation observation;
    OK(rc_registry_observe(r, b.handle, 1, &observation));
    RcRegistryCapture captured = {0};
    OK(rc_registry_capture(r, b.handle, &captured));
    OK(rc_registry_source_release(r, &b));
    assert(rc_registry_stats(r).retired_records == 1 && !rc_registry_stats(r).live_sources);
    RcRegistryRecord retained;
    OK(rc_registry_get(r, first, &retained));
    assert(retained.finalized && !retained.current);
    OK(rc_registry_capture_release(r, captured));
    assert(rc_registry_collect(r, 1) == 1 && !rc_registry_stats(r).records);
    // A Tex identity replacement/discard releases the last source owner without guest Finalize or a periodic scan.
    for (uint64_t serial = 12; serial < 10012; ++serial) {
        OK(rc_registry_source_acquire(r, serial, range(100, 16), &a));
        assert(a.handle.id != first.id);
        OK(rc_registry_source_release(r, &a));
        assert(!rc_registry_stats(r).records && !rc_registry_stats(r).source_leases);
    }
    rc_registry_destroy(r);

    // A descriptor can repeatedly change in a completely full pool when no reader owns its previous incarnation.
    r = make(1, 1, 1);
    RcRegistryHandle descriptor;
    for (uint64_t serial = 1; serial <= 10000; ++serial) {
        OK(rc_registry_register(r, 1, serial, range(100, 16), serial, &descriptor));
        assert(rc_registry_stats(r).records == 1 && rc_registry_stats(r).live_descriptors == 1);
        assert(!rc_registry_stats(r).retired_records && !rc_registry_stats(r).record_refusals);
    }
    rc_registry_destroy(r);
}

static void retired_queue_visits_only_obsolete_records(void) {
    RcRegistry *r = make(1024, 1, 1);
    RcRegistryHandle h;
    for (uint64_t i = 1; i <= 1000; ++i) OK(rc_registry_source(r, i, range(i * 32, 16), &h));
    OK(rc_registry_finalize_object(r, 999, range(999 * 32, 16)));
    assert(rc_registry_stats(r).retired_records == 1);
    assert(rc_registry_collect(r, 1) == 1); // A visit cannot be wasted on one of the 999 live records.
    assert(rc_registry_stats(r).records == 999 && !rc_registry_stats(r).retired_records);
    rc_registry_destroy(r);
}

static void descriptor_churn_preserves_only_reachable_history(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){4, 1, 4});
    assert(r);
    RcSourceRange range = {1, 0, 16};
    RcRegistryHandle original, current, invisible;
    RcRegistryRecord read;
    RcRegistryScope older, newer;
    assert(rc_registry_register(r, 1, 1, range, 101, &original) == RC_REGISTRY_OK);
    assert(rc_registry_scope_begin(r, &older) == RC_REGISTRY_OK);
    for (uint64_t serial = 2; serial <= 4096; serial++) {
        assert(rc_registry_register(r, 1, serial, range, serial + 100, &current) == RC_REGISTRY_OK);
        assert(rc_registry_resolve(r, 1, older, &read) == RC_REGISTRY_OK && read.handle.id == original.id);
        assert(rc_registry_stats(r).records == 2);
    }
    assert(rc_registry_register(r, 2, 1, range, 201, &invisible) == RC_REGISTRY_OK);
    RcRegistryCapture pinned = {0};
    assert(rc_registry_capture(r, invisible, &pinned) == RC_REGISTRY_OK);
    assert(rc_registry_retire_registration(r, 2, 1) == RC_REGISTRY_OK);
    assert(rc_registry_scope_begin(r, &newer) == RC_REGISTRY_OK);
    assert(!rc_registry_registration_reachable(r, invisible));
    assert(rc_registry_collect(r, 4) == 0); // The independent native/payload capture still pins the record.
    assert(rc_registry_get(r, invisible, &read) == RC_REGISTRY_OK);
    assert(rc_registry_capture_release(r, pinned) == RC_REGISTRY_OK);
    assert(rc_registry_collect(r, 4) == 1);
    assert(rc_registry_get(r, invisible, &read) == RC_REGISTRY_STALE);
    assert(rc_registry_resolve(r, 1, newer, &read) == RC_REGISTRY_OK && read.handle.id == current.id);
    RcRegistryHandle extra;
    assert(rc_registry_register(r, 3, 1, range, 301, &extra) == RC_REGISTRY_OK);
    assert(rc_registry_register(r, 4, 1, range, 401, &extra) == RC_REGISTRY_OK);
    assert(rc_registry_register(r, 1, 4097, range, 4197, &extra) == RC_REGISTRY_CAPACITY);
    assert(rc_registry_resolve(r, 1, older, &read) == RC_REGISTRY_OK && read.handle.id == original.id);
    assert(rc_registry_scope_end(r, older) == RC_REGISTRY_OK);
    assert(rc_registry_register(r, 1, 4097, range, 4197, &extra) == RC_REGISTRY_OK);
    assert(rc_registry_resolve(r, 1, newer, &read) == RC_REGISTRY_OK && read.handle.id == current.id);
    assert(rc_registry_scope_end(r, newer) == RC_REGISTRY_OK);
    rc_registry_destroy(r);
}

static void adjacent_foreign_authority_has_one_interval(void) {
    RcRegistry *r = make(2, 2, 1); RcRegistryWrite ticket;
    for (uint64_t i = 0; i < 10000; ++i) {
        OK(rc_registry_write_begin(r, range(i * 16, 16), RC_REGISTRY_BUFFER, &ticket));
        OK(rc_registry_write_submit(r, ticket, i + 1));
        OK(rc_registry_write_complete(r, ticket, 0));
        RcRegistryStats stats = rc_registry_stats(r);
        assert(stats.writers == 1 && stats.writer_foreign == 1 && stats.writer_buffer == 1);
        assert(!stats.writer_pending && !stats.writer_submitted && !stats.writer_holds && !stats.fail_closed);
    }
    // A real one-byte gap remains clean; adjacency merging must not round foreign coverage out to pages.
    OK(rc_registry_write_begin(r, range(160001, 16), RC_REGISTRY_IMAGE, &ticket));
    OK(rc_registry_write_submit(r, ticket, 10001));
    OK(rc_registry_write_complete(r, ticket, 0));
    assert(rc_registry_stats(r).writer_foreign == 2);
    RcRegistryHandle gap; RcRegistryObservation observation;
    OK(rc_registry_source(r, 1, range(160000, 1), &gap));
    OK(rc_registry_observe(r, gap, 1, &observation));
    OK(rc_registry_reconcile_cpu(r, range(0, 160017)));
    assert(!rc_registry_stats(r).writers && !rc_registry_stats(r).writer_foreign);
    rc_registry_destroy(r);
}

static void visibility_cutoffs_preserve_newer_authority(void) {
    RcRegistry *r = make(16, 16, 2);
    RcRegistryWrite old, future, pending, hold;
    RcRegistryHandle middle, left, later; RcRegistryObservation o;
    OK(rc_registry_source(r, 1, range(120, 20), &middle));
    OK(rc_registry_source(r, 2, range(100, 20), &left));
    OK(rc_registry_source(r, 3, range(300, 20), &later));
    OK(rc_registry_write_begin(r, range(100, 100), RC_REGISTRY_BUFFER, &old));
    OK(rc_registry_write_submit(r, old, 10));
    OK(rc_registry_write_complete(r, old, 0));
    // Reservation chronology is not queue order: this old reservation belongs to a later scene.
    OK(rc_registry_write_begin(r, range(300, 20), RC_REGISTRY_BUFFER, &future));
    RcRegistryVisibility proof;
    OK(rc_registry_visibility_begin(r, RC_REGISTRY_VISIBILITY_ORDERED, &proof));
    IS(rc_registry_visibility_apply(r, &proof, range(100, 100)), RC_REGISTRY_INVALID);
    proof.submission = 10;
    OK(rc_registry_write_submit(r, future, 11));
    OK(rc_registry_write_complete(r, future, 0));
    OK(rc_registry_write_begin(r, range(120, 10), RC_REGISTRY_UNKNOWN, &pending));
    OK(rc_registry_hold_begin(r, range(140, 10), RC_REGISTRY_IMAGE, &hold));
    // Exact middle visibility is retained even though a newer recording overlaps it; it only removes old foreign.
    OK(rc_registry_visibility_apply(r, &proof, range(120, 20)));
    IS(rc_registry_observe(r, middle, 1, &o), RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_cancel(r, pending));
    OK(rc_registry_observe(r, middle, 1, &o));
    IS(rc_registry_observe(r, left, 1, &o), RC_REGISTRY_BLOCKED);
    OK(rc_registry_visibility_apply(r, &proof, range(100, 220)));
    OK(rc_registry_observe(r, left, 2, &o));
    IS(rc_registry_observe(r, later, 2, &o), RC_REGISTRY_BLOCKED);
    assert(rc_registry_stats(r).writer_holds == 1 && rc_registry_stats(r).writer_foreign == 1);
    OK(rc_registry_hold_release(r, hold, 1));
    // A later proof can retire the later scene, without requiring its reservation to be newer.
    OK(rc_registry_visibility_begin(r, RC_REGISTRY_VISIBILITY_ORDERED, &proof));
    proof.submission = 11;
    OK(rc_registry_visibility_apply(r, &proof, range(300, 20)));
    OK(rc_registry_observe(r, later, 3, &o));
    assert(!rc_registry_stats(r).writers && !rc_registry_stats(r).reconcile_busy);
    OK(rc_registry_write_begin(r, range(100, 100), RC_REGISTRY_BUFFER, &old));
    OK(rc_registry_visibility_begin(r, RC_REGISTRY_VISIBILITY_ORDERED, &proof)); proof.submission = 12;
    // The current scene's producer completes before its final copy callback, after the proof was recorded.
    OK(rc_registry_write_submit(r, old, 12)); OK(rc_registry_write_complete(r, old, 0));
    OK(rc_registry_visibility_apply(r, &proof, range(100, 100)));
    OK(rc_registry_observe(r, middle, 4, &o));
    assert(!rc_registry_stats(r).writers);
    rc_registry_destroy(r);
}

static void visibility_completion_and_mapping_cutoffs(void) {
    RcRegistry *r = make(8, 8, 1); RcRegistryWrite a, b; RcRegistryVisibility proof;
    RcRegistryHandle source; RcRegistryObservation o;
    OK(rc_registry_source(r, 1, range(100, 100), &source));
    OK(rc_registry_write_begin(r, range(100, 100), RC_REGISTRY_BUFFER, &a));
    // A lazy copy cannot include this unsubmitted recording merely because it was reserved before the copy.
    OK(rc_registry_visibility_begin(r, RC_REGISTRY_VISIBILITY_COMPLETED, &proof));
    OK(rc_registry_write_submit(r, a, 1));
    OK(rc_registry_write_complete(r, a, 0));
    OK(rc_registry_visibility_apply(r, &proof, range(100, 100)));
    IS(rc_registry_observe(r, source, 1, &o), RC_REGISTRY_BLOCKED);
    OK(rc_registry_visibility_begin(r, RC_REGISTRY_VISIBILITY_COMPLETED, &proof));
    // A newer completed overlap must survive an older copy, including when coalescence retains the union.
    OK(rc_registry_write_begin(r, range(150, 100), RC_REGISTRY_BUFFER, &b));
    OK(rc_registry_write_submit(r, b, 2));
    OK(rc_registry_write_complete(r, b, 0));
    assert(rc_registry_stats(r).writers == 1 && rc_registry_stats(r).writer_buffer == 1);
    OK(rc_registry_visibility_apply(r, &proof, range(100, 150)));
    assert(rc_registry_stats(r).writer_foreign == 1);
    OK(rc_registry_visibility_begin(r, RC_REGISTRY_VISIBILITY_COMPLETED, &proof));
    OK(rc_registry_range_gone(r, range(100, 150)));
    OK(rc_registry_source(r, 2, range(100, 100), &source));
    OK(rc_registry_visibility_apply(r, &proof, range(100, 150)));
    IS(rc_registry_observe(r, source, 2, &o), RC_REGISTRY_BLOCKED);
    OK(rc_registry_visibility_begin(r, RC_REGISTRY_VISIBILITY_COMPLETED, &proof));
    OK(rc_registry_visibility_apply(r, &proof, range(120, 20)));
    assert(rc_registry_stats(r).writers == 2);
    // Split fragments retain provenance, rather than adopting the new allocation identity as a producer cutoff.
    OK(rc_registry_visibility_apply(r, &proof, range(100, 20)));
    OK(rc_registry_visibility_apply(r, &proof, range(140, 110)));
    OK(rc_registry_observe(r, source, 3, &o));
    assert(!rc_registry_stats(r).writers);
    // Holds and abandoned producers do not acquire a successful queue order when they leave foreign coverage.
    OK(rc_registry_hold_begin(r, range(100, 100), RC_REGISTRY_IMAGE, &a));
    OK(rc_registry_visibility_begin(r, RC_REGISTRY_VISIBILITY_ORDERED, &proof));
    proof.submission = 9;
    OK(rc_registry_hold_release(r, a, 0));
    OK(rc_registry_visibility_apply(r, &proof, range(100, 100)));
    assert(rc_registry_stats(r).writer_foreign == 1);
    OK(rc_registry_visibility_begin(r, RC_REGISTRY_VISIBILITY_COMPLETED, &proof));
    proof.writer_kinds = 1u << RC_REGISTRY_IMAGE;
    OK(rc_registry_visibility_apply(r, &proof, range(100, 100)));
    OK(rc_registry_write_begin(r, range(100, 100), RC_REGISTRY_BUFFER, &a));
    OK(rc_registry_write_submit(r, a, 3));
    OK(rc_registry_visibility_begin(r, RC_REGISTRY_VISIBILITY_ORDERED, &proof));
    proof.submission = 9;
    OK(rc_registry_write_abandon(r, a, 1));
    OK(rc_registry_visibility_apply(r, &proof, range(100, 100)));
    assert(rc_registry_stats(r).writer_foreign == 1);
    OK(rc_registry_reconcile_cpu(r, range(100, 100)));
    OK(rc_registry_write_begin(r, range(100, 100), RC_REGISTRY_BUFFER, &a));
    OK(rc_registry_visibility_begin(r, RC_REGISTRY_VISIBILITY_ORDERED, &proof)); proof.submission = 10;
    OK(rc_registry_range_gone(r, range(100, 100)));
    OK(rc_registry_write_submit(r, a, 10)); OK(rc_registry_write_complete(r, a, 0));
    OK(rc_registry_visibility_apply(r, &proof, range(100, 100)));
    assert(rc_registry_stats(r).writer_foreign == 1); // A delayed completion cannot erase the remap barrier.
    rc_registry_destroy(r);
}

static void visibility_capacity_is_atomic_and_blockers_are_classified(void) {
    RcRegistry *r = make(4, 1, 1); RcRegistryWrite a; RcRegistryVisibility proof;
    RcRegistryHandle h; RcRegistryObservation o;
    OK(rc_registry_source(r, 1, range(100, 100), &h));
    OK(rc_registry_write_begin(r, range(100, 100), RC_REGISTRY_BUFFER, &a));
    IS(rc_registry_observe(r, h, 1, &o), RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_submit(r, a, 1));
    IS(rc_registry_observe(r, h, 1, &o), RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_complete(r, a, 0));
    IS(rc_registry_observe(r, h, 1, &o), RC_REGISTRY_BLOCKED);
    OK(rc_registry_visibility_begin(r, RC_REGISTRY_VISIBILITY_COMPLETED, &proof));
    IS(rc_registry_visibility_apply(r, &proof, range(120, 20)), RC_REGISTRY_CAPACITY);
    assert(rc_registry_stats(r).writers == 1 && !rc_registry_stats(r).fail_closed);
    OK(rc_registry_visibility_apply(r, &proof, range(100, 100)));
    OK(rc_registry_hold_begin(r, range(100, 100), RC_REGISTRY_IMAGE, &a));
    IS(rc_registry_observe(r, h, 1, &o), RC_REGISTRY_BLOCKED);
    RcRegistryStats s = rc_registry_stats(r);
    assert(s.blocked_pending == 1 && s.blocked_submitted == 1 && s.blocked_foreign == 1 && s.blocked_held == 1);
    assert(s.blocked_buffer == 3 && s.blocked_image == 1 && !s.blocked_unknown);
    assert(s.visibility_applied == 1 && s.visibility_removed == 1);
    rc_registry_destroy(r);
}

static void visibility_keeps_provider_classes_separate(void) {
    RcRegistry *r = make(8, 8, 1); RcRegistryWrite buffer, image, unknown, abandoned;
    RcRegistryHandle outside, inside, adjacent; RcRegistryObservation o; RcRegistryVisibility proof;
    OK(rc_registry_source(r, 1, range(100, 20), &outside));
    OK(rc_registry_source(r, 2, range(140, 20), &inside));
    OK(rc_registry_source(r, 3, range(200, 20), &adjacent));
    OK(rc_registry_write_begin(r, range(100, 100), RC_REGISTRY_BUFFER, &buffer));
    OK(rc_registry_write_submit(r, buffer, 1)); OK(rc_registry_write_complete(r, buffer, 0));
    OK(rc_registry_write_begin(r, range(140, 20), RC_REGISTRY_IMAGE, &image));
    OK(rc_registry_write_submit(r, image, 1)); OK(rc_registry_write_complete(r, image, 0));
    OK(rc_registry_hold_begin(r, range(200, 20), RC_REGISTRY_UNKNOWN, &unknown));
    OK(rc_registry_hold_release(r, unknown, 0));
    assert(rc_registry_stats(r).writers == 3);
    OK(rc_registry_visibility_begin(r, RC_REGISTRY_VISIBILITY_COMPLETED, &proof));
    OK(rc_registry_visibility_apply(r, &proof, range(100, 120)));
    OK(rc_registry_observe(r, outside, 1, &o));
    IS(rc_registry_observe(r, inside, 1, &o), RC_REGISTRY_BLOCKED);
    IS(rc_registry_observe(r, adjacent, 1, &o), RC_REGISTRY_BLOCKED);
    assert(rc_registry_stats(r).writers == 2 && !rc_registry_stats(r).writer_buffer);
    OK(rc_registry_write_begin(r, range(100, 20), RC_REGISTRY_BUFFER, &buffer));
    OK(rc_registry_write_submit(r, buffer, 2)); OK(rc_registry_write_complete(r, buffer, 0));
    OK(rc_registry_write_begin(r, range(120, 20), RC_REGISTRY_BUFFER, &abandoned));
    OK(rc_registry_write_abandon(r, abandoned, 1));
    assert(rc_registry_stats(r).writers == 4); // Unordered cancellation cannot contaminate an adjacent queue proof.
    OK(rc_registry_visibility_begin(r, RC_REGISTRY_VISIBILITY_ORDERED, &proof)); proof.submission = 2;
    OK(rc_registry_visibility_apply(r, &proof, range(100, 120)));
    assert(rc_registry_stats(r).writers == 3 && rc_registry_stats(r).writer_buffer == 1);
    OK(rc_registry_observe(r, outside, 2, &o));
    rc_registry_destroy(r);
}

static RcRegistryPosition pos(uint64_t scene, uint64_t ordinal) { return (RcRegistryPosition){scene, ordinal}; }
static RcPreparedInput prepared(uint64_t id) {
    return (RcPreparedInput){.content = id, .representation = 2, .policy = 3, .payload = id + 100,
                             .kind = RC_PROVIDER_CAPTURE};
}

static void ordered_capture_owns_old_version_after_future_writes(void) {
    RcRegistry *r = make(8, 8, 1); RcRegistryHandle source; RcRegistryWrite future, later;
    RcRegistryReadObservation observed, newer; RcRegistryReadVersion first, reused, replacement;
    OK(rc_registry_source(r, 1, range(100, 100), &source));
    OK(rc_registry_write_begin_at(r, range(100, 100), RC_REGISTRY_BUFFER, pos(1, 5), &future));
    IS(rc_registry_observe_at(r, source, 1, pos(1, 4), &observed), RC_REGISTRY_BLOCKED);
    OK(rc_registry_declare_through(r, pos(1, 6)));
    OK(rc_registry_observe_at(r, source, 1, pos(1, 4), &observed));
    RcPreparedInput input = prepared(10);
    OK(rc_registry_read_commit(r, &observed, &input, &first));
    OK(rc_registry_read_commit(r, &observed, &input, &reused));
    assert(first.id == reused.id && first.input.payload == 110);
    RcPreparedInput conditional = input; conditional.kind = RC_PROVIDER_CONDITIONAL_GUEST;
    IS(rc_registry_read_commit(r, &observed, &conditional, &reused), RC_REGISTRY_INVALID);
    RcLifetimeIds ids = {0}; RcLifetimeVersion version = {0}; RcLifetimeResidency image = {0};
    RcLifetimeCapture use = {0};
    assert(rc_lifetime_version_init(&ids, &version, RC_LIFETIME_RECOVERY_IMMUTABLE, 0) == RC_LIFETIME_OK);
    assert(rc_lifetime_residency_install(&ids, &version, version.id, &image) == RC_LIFETIME_OK);
    assert(rc_lifetime_capture_begin(&version, &image, image.binding, &use) == RC_LIFETIME_OK);
    // Later declaration/cancellation does not turn an immutable version into a conditional guest-memory promise.
    OK(rc_registry_write_begin_at(r, range(100, 100), RC_REGISTRY_BUFFER, pos(1, 9), &later));
    assert(rc_registry_read_observation_current(r, &observed));
    OK(rc_registry_write_cancel(r, later));
    assert(rc_registry_read_observation_current(r, &observed));
    uint64_t visits = rc_registry_stats(r).overlap_visits;
    for (unsigned i = 0; i < 1000; i++) assert(rc_registry_read_observation_current(r, &observed));
    assert(rc_registry_stats(r).overlap_visits == visits);
    OK(rc_registry_cpu_changed(r, range(120, 1)));
    assert(!rc_registry_read_observation_current(r, &observed));
    IS(rc_registry_read_commit(r, &observed, &input, &reused), RC_REGISTRY_STALE);
    OK(rc_registry_observe_at(r, source, 1, pos(1, 4), &newer));
    input = prepared(11);
    OK(rc_registry_read_commit(r, &newer, &input, &replacement));
    assert(replacement.id != first.id && first.input.payload == 110);
    OK(rc_registry_handoff_through(r, pos(1, 6)));
    assert(!rc_registry_read_observation_current(r, &newer));
    OK(rc_registry_write_submit(r, future, 1)); OK(rc_registry_write_complete(r, future, 0));
    OK(rc_registry_declare_through(r, pos(2, 2)));
    IS(rc_registry_observe_at(r, source, 2, pos(2, 2), &newer), RC_REGISTRY_BLOCKED);
    assert(use.binding.version == version.id && first.input.payload == 110);
    assert(rc_lifetime_residency_detach(&version, &image, image.binding) == RC_LIFETIME_OK);
    assert(rc_lifetime_residency_retire(&version, &image, image.binding) == RC_LIFETIME_BUSY);
    RcLifetimeCompletion none = {0}, done = {0};
    assert(rc_lifetime_completion_add(&done, 1, 1) == RC_LIFETIME_OK);
    assert(rc_lifetime_capture_submit(&version, &image, &use, &none, &done) == RC_LIFETIME_OK);
    assert(rc_lifetime_capture_complete(&version, &image, &use, &none) == RC_LIFETIME_BUSY);
    assert(rc_lifetime_capture_complete(&version, &image, &use, &done) == RC_LIFETIME_OK);
    assert(rc_lifetime_residency_retire(&version, &image, image.binding) == RC_LIFETIME_OK);
    RcRegistryVisibility visible;
    OK(rc_registry_visibility_begin(r, RC_REGISTRY_VISIBILITY_COMPLETED, &visible));
    OK(rc_registry_visibility_apply(r, &visible, range(100, 100)));
    OK(rc_registry_observe_at(r, source, 2, pos(2, 2), &newer));
    OK(rc_registry_range_gone(r, range(100, 100)));
    assert(!rc_registry_read_observation_current(r, &newer));
    assert(first.input.payload == 110); // Old exact ownership is independent of source association retirement.
    rc_registry_destroy(r);
}

static void ordered_declarations_cannot_hide_predecessors(void) {
    RcRegistry *r = make(4, 8, 1); RcRegistryHandle source; RcRegistryWrite earlier, future, unknown;
    RcRegistryReadObservation observation;
    OK(rc_registry_source(r, 1, range(100, 100), &source));
    // Reverse worker arrival is harmless only after the complete effect prefix has been declared.
    OK(rc_registry_write_begin_at(r, range(100, 100), RC_REGISTRY_BUFFER, pos(1, 7), &future));
    OK(rc_registry_write_begin_at(r, range(100, 100), RC_REGISTRY_BUFFER, pos(1, 3), &earlier));
    OK(rc_registry_declare_through(r, pos(1, 8)));
    IS(rc_registry_observe_at(r, source, 1, pos(1, 4), &observation), RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_cancel(r, earlier));
    OK(rc_registry_observe_at(r, source, 1, pos(1, 4), &observation));
    IS(rc_registry_observe_at(r, source, 1, pos(1, 7), &observation), RC_REGISTRY_BLOCKED);
    OK(rc_registry_observe_at(r, source, 1, pos(1, 6), &observation));
    OK(rc_registry_write_begin(r, range(120, 1), RC_REGISTRY_UNKNOWN, &unknown));
    assert(!rc_registry_read_observation_current(r, &observation));
    IS(rc_registry_observe_at(r, source, 1, pos(1, 6), &observation), RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_cancel(r, unknown));
    OK(rc_registry_observe_at(r, source, 1, pos(1, 6), &observation));
    // The API rejects discovering a formerly absent predecessor after accepting this prefix.
    IS(rc_registry_write_begin_at(r, range(100, 100), RC_REGISTRY_BUFFER, pos(1, 5), &earlier), RC_REGISTRY_STALE);
    assert(rc_registry_stats(r).fail_closed && rc_registry_stats(r).late_declarations == 1);
    assert(!rc_registry_read_observation_current(r, &observation));
    rc_registry_destroy(r);

    r = make(2, 2, 1);
    OK(rc_registry_write_begin_at(r, range(100, 100), RC_REGISTRY_BUFFER, pos(2, 3), &future));
    OK(rc_registry_declare_through(r, pos(2, 4)));
    IS(rc_registry_write_submit(r, future, 2), RC_REGISTRY_STALE); // Missing pre-handoff closure fails closed.
    assert(rc_registry_stats(r).fail_closed);
    rc_registry_destroy(r);
}

static void immutable_effect_inputs_need_only_a_structural_range_check(void) {
    RcRegistry *r = make(1, 4, 1); RcRegistryWrite earlier, current;
    OK(rc_registry_write_begin_at(r, range(100, 20), RC_REGISTRY_BUFFER, pos(1, 3), &earlier));
    OK(rc_registry_declare_through(r, pos(1, 4))); // Current input is sealed; its output remains undeclared.
    IS(rc_registry_read_range_at(r, range(100, 20), pos(1, 4)), RC_REGISTRY_BLOCKED);
    OK(rc_registry_read_range_at(r, range(200, 20), pos(1, 4)));
    OK(rc_registry_write_begin_at(r, range(200, 20), RC_REGISTRY_BUFFER, pos(1, 5), &current));
    OK(rc_registry_declare_through(r, pos(1, 5)));
    OK(rc_registry_read_range_at(r, range(200, 20), pos(1, 4)));
    IS(rc_registry_read_range_at(r, range(200, 20), pos(1, 6)), RC_REGISTRY_BLOCKED);
    assert(!rc_registry_stats(r).records && !rc_registry_stats(r).source_leases);
    rc_registry_destroy(r);
}

static void existing_declarations_expand_and_complete_without_clean_gaps(void) {
    RcRegistry *r = make(8, 8, 1); RcRegistryHandle a, b; RcRegistryWrite domain;
    RcRegistryReadObservation observation;
    OK(rc_registry_source(r, 1, range(100, 20), &a));
    OK(rc_registry_source(r, 2, range(500, 20), &b));
    OK(rc_registry_write_begin_at(r, range(400, 200), RC_REGISTRY_BUFFER, pos(3, 5), &domain));
    OK(rc_registry_declare_through(r, pos(3, 6)));
    OK(rc_registry_observe_at(r, a, 1, pos(3, 4), &observation));
    OK(rc_registry_write_expand(r, domain, range(0, 1000)));
    assert(!rc_registry_read_observation_current(r, &observation));
    assert(rc_registry_stats(r).writers == 1 && !rc_registry_stats(r).late_declarations);
    RcRegistryVisibility proof;
    OK(rc_registry_visibility_begin(r, RC_REGISTRY_VISIBILITY_ORDERED, &proof)); proof.submission = 3;
    OK(rc_registry_handoff_through(r, pos(3, 6)));
    OK(rc_registry_write_submit(r, domain, 3));
    RcSourceRange uncertain[] = {range(500, 20), range(700, 20)};
    OK(rc_registry_write_complete_ranges(r, domain, uncertain, 2));
    assert(rc_registry_stats(r).writer_foreign == 2 && !rc_registry_stats(r).writer_pending);
    IS(rc_registry_write_complete_ranges(r, domain, NULL, 0), RC_REGISTRY_STALE);
    OK(rc_registry_declare_through(r, pos(4, 2)));
    OK(rc_registry_observe_at(r, a, 2, pos(4, 2), &observation));
    IS(rc_registry_observe_at(r, b, 2, pos(4, 2), &observation), RC_REGISTRY_BLOCKED);
    // Child metadata inherits the declared producer cutoff despite being allocated after this copy was recorded.
    OK(rc_registry_visibility_apply(r, &proof, range(500, 20)));
    OK(rc_registry_observe_at(r, b, 2, pos(4, 2), &observation));
    assert(rc_registry_stats(r).writers == 1);
    rc_registry_destroy(r);

    r = make(2, 1, 1);
    OK(rc_registry_source(r, 1, range(100, 20), &a));
    OK(rc_registry_write_begin(r, range(0, 1000), RC_REGISTRY_BUFFER, &domain));
    OK(rc_registry_write_submit(r, domain, 1));
    IS(rc_registry_write_complete_ranges(r, domain, uncertain, 2), RC_REGISTRY_CAPACITY);
    RcRegistryObservation blocked;
    IS(rc_registry_observe(r, a, 1, &blocked), RC_REGISTRY_BLOCKED);
    assert(rc_registry_stats(r).writers == 1 && rc_registry_stats(r).writer_submitted == 1 &&
           rc_registry_stats(r).completion_range_refusals == 1);
    RcSourceRange invalid = range(900, 200);
    IS(rc_registry_write_complete_ranges(r, domain, &invalid, 1), RC_REGISTRY_INVALID);
    OK(rc_registry_write_complete_ranges(r, domain, NULL, 0));
    OK(rc_registry_observe(r, a, 2, &blocked));
    assert(!rc_registry_stats(r).writers);
    rc_registry_destroy(r);
}

static void source_revalidation_skips_unrelated_authority_and_stops_at_writes(void) {
    RcRegistry *r=make(4,8,1); RcRegistryHandle source; RcRegistryWrite future,unrelated,newer;
    RcRegistryReadObservation old,next;
    OK(rc_registry_source(r,1,range(100,20),&source));
    OK(rc_registry_write_begin_at(r,range(100,20),RC_REGISTRY_BUFFER,pos(1,9),&future));
    OK(rc_registry_declare_through(r,pos(1,10)));
    OK(rc_registry_observe_at(r,source,1,pos(1,2),&old));
    uint64_t visits=rc_registry_stats(r).overlap_visits;
    OK(rc_registry_read_revalidate(r,&old,1,pos(1,8),&next));
    assert(rc_registry_stats(r).overlap_visits==visits);
    IS(rc_registry_read_revalidate(r,&old,1,pos(1,9),&next),RC_REGISTRY_STALE);
    OK(rc_registry_write_begin(r,range(900,20),RC_REGISTRY_BUFFER,&unrelated));
    OK(rc_registry_read_revalidate(r,&old,1,pos(1,8),&next));
    OK(rc_registry_write_cancel(r,unrelated));
    OK(rc_registry_write_cancel(r,future));
    OK(rc_registry_read_revalidate(r,&old,1,pos(1,8),&next));
    IS(rc_registry_read_revalidate(r,&old,1,pos(1,9),&next),RC_REGISTRY_STALE); // Retain conservative old edge.
    OK(rc_registry_observe_at(r,source,1,pos(1,8),&old));
    OK(rc_registry_handoff_through(r,pos(1,10)));
    OK(rc_registry_declare_through(r,pos(2,2)));
    IS(rc_registry_read_revalidate(r,&old,2,pos(2,2),&next),RC_REGISTRY_STALE);
    OK(rc_registry_observe_at(r,source,2,pos(2,2),&next));
    assert(next.scope==2 && next.position.scene==2);
    OK(rc_registry_write_begin_at(r,range(100,20),RC_REGISTRY_BUFFER,pos(2,3),&newer));
    OK(rc_registry_declare_through(r,pos(2,6)));
    IS(rc_registry_read_revalidate(r,&old,2,pos(2,2),&next),RC_REGISTRY_STALE);
    OK(rc_registry_observe_at(r,source,2,pos(2,2),&old));
    IS(rc_registry_read_revalidate(r,&old,2,pos(2,4),&next),RC_REGISTRY_STALE);
    OK(rc_registry_write_cancel(r,newer));
    rc_registry_destroy(r);
}

static void compatibility_intervals_do_not_certify_source_bytes(void) {
    RcRegistry *r = make(4, 12, 1);
    RcRegistryHandle source; RcRegistryWrite prior, future, unrelated, unknown;
    RcRegistryBindingObservation bracket;
    RcRegistryBindingReceipt receipt, fresh;
    RcRegistryReadObservation clean;
    OK(rc_registry_source(r, 1, range(100, 20), &source));
    // The broad unknown producer precedes this preparation. It must not hide a later, narrower producer.
    OK(rc_registry_write_begin_at(r, range(0, 1000), RC_REGISTRY_UNKNOWN, pos(1, 3), &prior));
    OK(rc_registry_write_begin_at(r, range(100, 20), RC_REGISTRY_BUFFER, pos(1, 9), &future));
    OK(rc_registry_declare_through(r, pos(1, 10)));
    IS(rc_registry_observe_at(r, source, 1, pos(1, 4), &clean), RC_REGISTRY_BLOCKED);
    OK(rc_registry_binding_begin(r, source, 1, pos(1, 4), &bracket));
    assert(bracket.next_read.scene == 1 && bracket.next_read.ordinal == 8);
    assert(bracket.provenance == RC_REGISTRY_BINDING_PRIOR_PENDING);
    assert(bracket.writer_kinds == ((1u << RC_REGISTRY_UNKNOWN) | (1u << RC_REGISTRY_BUFFER)));
    OK(rc_registry_binding_commit(r, &bracket, &receipt));
    uint64_t visits = rc_registry_stats(r).overlap_visits;
    for (unsigned i = 0; i < 1000; i++) OK(rc_registry_binding_revalidate(r, &receipt, 1, pos(1, 6), &fresh));
    assert(rc_registry_stats(r).overlap_visits == visits);
    assert(fresh.observation.position.ordinal == 4); // Retain the actual preparation's origin.
    IS(rc_registry_binding_revalidate(r, &receipt, 1, pos(1, 8), &fresh), RC_REGISTRY_STALE);
    IS(rc_registry_binding_begin(r, source, 1, pos(1, 8), &bracket), RC_REGISTRY_BLOCKED); // Own output phase 9.
    IS(rc_registry_binding_revalidate(r, &receipt, 2, pos(1, 6), &fresh), RC_REGISTRY_STALE);
    IS(rc_registry_binding_revalidate(r, &receipt, 1, pos(1, 2), &fresh), RC_REGISTRY_STALE);
    OK(rc_registry_write_begin(r, range(900, 20), RC_REGISTRY_BUFFER, &unrelated));
    OK(rc_registry_binding_revalidate(r, &receipt, 1, pos(1, 6), &fresh));
    OK(rc_registry_write_cancel(r, unrelated));
    OK(rc_registry_write_begin(r, range(105, 1), RC_REGISTRY_UNKNOWN, &unknown));
    IS(rc_registry_binding_revalidate(r, &receipt, 1, pos(1, 6), &fresh), RC_REGISTRY_STALE);
    IS(rc_registry_binding_begin(r, source, 1, pos(1, 6), &bracket), RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_submit(r, unknown, 1));
    IS(rc_registry_binding_begin(r, source, 1, pos(1, 6), &bracket), RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_complete(r, unknown, 0));
    OK(rc_registry_binding_begin(r, source, 1, pos(1, 6), &bracket));
    assert(bracket.provenance & RC_REGISTRY_BINDING_FOREIGN);
    IS(rc_registry_observe_at(r, source, 1, pos(1, 6), &clean), RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_cancel(r, future));
    OK(rc_registry_binding_commit(r, &bracket, &receipt)); // Cancellation cannot make prior preparation unsafe.
    assert(receipt.observation.next_read.ordinal == 8); // It also cannot silently extend the old deadline.
    OK(rc_registry_write_cancel(r, prior));
    RcRegistryStats stats = rc_registry_stats(r);
    assert(stats.binding_begins == 5 && stats.binding_ready == 2 && stats.binding_edge == 1 &&
           stats.binding_unpositioned == 2 && !stats.binding_unknown_held && !stats.binding_unordered_foreign);
    rc_registry_destroy(r);
}

static void compatibility_brackets_reject_races_and_replaced_sources(void) {
    RcRegistry *r = make(2, 6, 1); RcRegistryHandle source, replacement;
    RcRegistryWrite hold, future, alias;
    RcRegistryBindingObservation before;
    RcRegistryBindingReceipt ready, next;
    OK(rc_registry_source(r, 1, range(100, 20), &source));
    OK(rc_registry_hold_begin(r, range(100, 20), RC_REGISTRY_BUFFER, &hold));
    OK(rc_registry_declare_through(r, pos(2, 6)));
    OK(rc_registry_binding_begin(r, source, 3, pos(2, 2), &before));
    assert(before.provenance == RC_REGISTRY_BINDING_HELD);
    OK(rc_registry_cpu_changed(r, range(105, 1)));
    IS(rc_registry_binding_commit(r, &before, &ready), RC_REGISTRY_STALE);
    OK(rc_registry_binding_begin(r, source, 3, pos(2, 2), &before));
    OK(rc_registry_binding_commit(r, &before, &ready));
    OK(rc_registry_hold_release(r, hold, 0));
    IS(rc_registry_binding_revalidate(r, &ready, 3, pos(2, 4), &next), RC_REGISTRY_STALE);
    OK(rc_registry_binding_begin(r, source, 3, pos(2, 2), &before));
    OK(rc_registry_write_begin_at(r, range(100, 20), RC_REGISTRY_BUFFER, pos(2, 9), &future));
    OK(rc_registry_declare_through(r, pos(2, 10)));
    OK(rc_registry_binding_commit(r, &before, &ready));
    assert(ready.observation.sealed_through.ordinal == 6 && ready.observation.next_read.ordinal == 7);
    IS(rc_registry_binding_revalidate(r, &ready, 3, pos(2, 8), &next), RC_REGISTRY_STALE);
    OK(rc_registry_binding_begin(r, source, 3, pos(2, 2), &before));
    OK(rc_registry_binding_commit(r, &before, &ready));
    OK(rc_registry_write_begin_at(r, range(500, 20), RC_REGISTRY_BUFFER, pos(2, 11), &alias));
    OK(rc_registry_write_expand(r, alias, range(100, 20)));
    IS(rc_registry_binding_revalidate(r, &ready, 3, pos(2, 4), &next), RC_REGISTRY_STALE);
    OK(rc_registry_binding_begin(r, source, 3, pos(2, 2), &before));
    OK(rc_registry_binding_commit(r, &before, &ready));
    OK(rc_registry_handoff_through(r, pos(2, 6)));
    IS(rc_registry_binding_revalidate(r, &ready, 3, pos(2, 4), &next), RC_REGISTRY_STALE);
    OK(rc_registry_declare_through(r, pos(3, 2)));
    IS(rc_registry_binding_revalidate(r, &ready, 3, pos(3, 2), &next), RC_REGISTRY_STALE);
    OK(rc_registry_binding_begin(r, source, 4, pos(3, 2), &before));
    OK(rc_registry_binding_commit(r, &before, &ready));
    OK(rc_registry_range_gone(r, range(100, 20)));
    IS(rc_registry_binding_revalidate(r, &ready, 4, pos(3, 2), &next), RC_REGISTRY_STALE);
    OK(rc_registry_source(r, 2, range(100, 20), &replacement));
    assert(replacement.id != source.id);
    IS(rc_registry_binding_commit(r, &before, &ready), RC_REGISTRY_STALE);
    rc_registry_destroy(r);
}

static void compatibility_intervals_refuse_unpositioned_alias_coverage(void) {
    RcRegistry *r = make(2, 6, 1); RcRegistryHandle source;
    RcRegistryWrite physical, virtual_write, completed;
    RcRegistryBindingObservation before;
    OK(rc_registry_source(r, 1, range(100, 20), &source));
    OK(rc_registry_hold_begin(r, range(100, 20), RC_REGISTRY_UNKNOWN, &physical));
    // The virtual producer does not overlap the canonical source. The persistent pool hold is the only witness.
    OK(rc_registry_write_begin_at(r, range(500, 20), RC_REGISTRY_BUFFER, pos(1, 9), &virtual_write));
    OK(rc_registry_declare_through(r, pos(1, 10)));
    IS(rc_registry_binding_begin(r, source, 1, pos(1, 2), &before), RC_REGISTRY_BLOCKED);
    OK(rc_registry_hold_release(r, physical, 0));
    IS(rc_registry_binding_begin(r, source, 1, pos(1, 2), &before), RC_REGISTRY_BLOCKED);
    // A strong replacement of canonical contents removes the remnant; a completed exact guest-address effect
    // may remain compatibility provenance without becoming a clean-source certificate.
    OK(rc_registry_reconcile_cpu(r, range(100, 20)));
    OK(rc_registry_write_begin(r, range(100, 20), RC_REGISTRY_UNKNOWN, &completed));
    OK(rc_registry_write_submit(r, completed, 1));
    OK(rc_registry_write_complete(r, completed, 0));
    OK(rc_registry_binding_begin(r, source, 1, pos(1, 2), &before));
    assert(before.provenance == RC_REGISTRY_BINDING_FOREIGN);
    RcRegistryReadObservation clean;
    IS(rc_registry_observe_at(r, source, 1, pos(1, 2), &clean), RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_cancel(r, virtual_write));
    RcRegistryStats stats = rc_registry_stats(r);
    assert(stats.binding_begins == 3 && stats.binding_ready == 1 && stats.binding_unknown_held == 1 &&
           stats.binding_unordered_foreign == 1 && !stats.binding_unpositioned && !stats.binding_edge);
    rc_registry_destroy(r);
}

static void compatibility_intervals_refuse_shared_image_aliases(void) {
    RcRegistry *r = make(2, 6, 1); RcRegistryHandle source;
    RcRegistryWrite image, hold;
    RcRegistryBindingObservation before;
    RcRegistryBindingReceipt ready, next;
    OK(rc_registry_source(r, 1, range(100, 20), &source));
    // The image owner shares this first-position claim across writes at phases 5 and 9. Adding the second
    // use does not create another registry ticket or revision; a CPU alias cannot infer an edge at phase 9.
    OK(rc_registry_write_begin_at(r, range(100, 20), RC_REGISTRY_IMAGE, pos(1, 5), &image));
    OK(rc_registry_declare_through(r, pos(1, 10)));
    OK(rc_registry_binding_begin(r, source, 1, pos(1, 2), &before));
    assert(before.next_read.scene == 1 && before.next_read.ordinal == 4);
    OK(rc_registry_binding_commit(r, &before, &ready));
    IS(rc_registry_binding_revalidate(r, &ready, 1, pos(1, 6), &next), RC_REGISTRY_STALE);
    IS(rc_registry_binding_begin(r, source, 1, pos(1, 6), &before), RC_REGISTRY_BLOCKED);
    OK(rc_registry_handoff_through(r, pos(1, 5)));
    OK(rc_registry_write_submit(r, image, 1));
    IS(rc_registry_binding_begin(r, source, 1, pos(1, 6), &before), RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_complete(r, image, 1));
    OK(rc_registry_hold_begin(r, range(100, 20), RC_REGISTRY_IMAGE, &hold));
    IS(rc_registry_binding_begin(r, source, 1, pos(1, 6), &before), RC_REGISTRY_BLOCKED);
    OK(rc_registry_hold_release(r, hold, 0));
    // A retired image's foreign bytes do not imply another ongoing image write. They remain legacy provenance,
    // never a clean source certificate, while an actual preparation can receive a compatibility interval again.
    OK(rc_registry_binding_begin(r, source, 1, pos(1, 6), &before));
    assert(before.provenance == RC_REGISTRY_BINDING_FOREIGN);
    RcRegistryReadObservation clean;
    IS(rc_registry_observe_at(r, source, 1, pos(1, 6), &clean), RC_REGISTRY_BLOCKED);
    RcRegistryStats stats = rc_registry_stats(r);
    assert(stats.binding_begins == 5 && stats.binding_ready == 2 && stats.binding_image_alias == 3 &&
           !stats.binding_unknown_held && !stats.binding_unordered_foreign);
    rc_registry_destroy(r);
}

static void compatibility_interval_entry_refusals_are_counted(void) {
    RcRegistry *r = make(2, 1, 1); RcRegistryHandle source;
    RcRegistryBindingObservation before;
    OK(rc_registry_source(r, 1, range(100, 20), &source));
    IS(rc_registry_binding_begin(r, source, 1, pos(1, 2), &before), RC_REGISTRY_BLOCKED);
    OK(rc_registry_declare_through(r, pos(1, 2)));
    OK(rc_registry_binding_begin(r, source, 1, pos(1, 2), &before));
    OK(rc_registry_range_gone(r, range(100, 20)));
    IS(rc_registry_binding_begin(r, source, 1, pos(1, 2), &before), RC_REGISTRY_STALE);
    OK(rc_registry_source(r, 2, range(100, 20), &source));
    rc_registry_fail_closed(r);
    IS(rc_registry_binding_begin(r, source, 1, pos(1, 2), &before), RC_REGISTRY_BLOCKED);
    RcRegistryStats stats = rc_registry_stats(r);
    assert(stats.binding_begins == 4 && stats.binding_ready == 1 && stats.binding_stale == 1 &&
           stats.binding_closed == 1 && stats.binding_failed == 1);
    rc_registry_destroy(r);
}

static void retained_byte_provenance_never_grants_read_permission(void) {
    RcRegistry *r = make(16, 8, 4);
    RcRegistryHandle h, replacement;
    RcRegistryObservation old, fresh, malformed;
    OK(rc_registry_source(r, 77, range(100, 20), &h));
    OK(rc_registry_observe(r, h, 1, &old));
    assert(rc_registry_content_current(r, &old));
    RcRegistryWrite write;
    OK(rc_registry_write_begin_at(r, range(0, UINT64_MAX), RC_REGISTRY_IMAGE, pos(1, 3), &write));
    OK(rc_registry_declare_through(r, pos(1, 4)));
    assert(!rc_registry_observation_current(r, &old));
    assert(rc_registry_content_current(r, &old));
    IS(rc_registry_observe(r, h, 1, &fresh), RC_REGISTRY_BLOCKED);
    IS(rc_registry_publish(r, &old, 1, 2, 3), RC_REGISTRY_STALE);
    OK(rc_registry_write_cancel(r, write));
    assert(rc_registry_content_current(r, &old) && !rc_registry_observation_current(r, &old));
    OK(rc_registry_observe(r, h, 1, &fresh));
    assert(fresh.content_revision == old.content_revision);
    malformed = fresh; malformed.content_revision = 0; assert(!rc_registry_content_current(r, &malformed));
    malformed = fresh; malformed.guard = 0; assert(!rc_registry_content_current(r, &malformed));
    malformed = fresh; malformed.range.offset++; assert(!rc_registry_content_current(r, &malformed));
    OK(rc_registry_cpu_changed(r, range(100, 1)));
    assert(!rc_registry_content_current(r, &old));
    OK(rc_registry_observe(r, h, 1, &fresh));
    OK(rc_registry_finalize_object(r, 77, range(100, 20)));
    assert(!rc_registry_content_current(r, &fresh));
    OK(rc_registry_source(r, 78, range(100, 20), &replacement));
    assert(replacement.id != h.id && !rc_registry_content_current(r, &fresh));
    OK(rc_registry_observe(r, replacement, 1, &fresh));
    OK(rc_registry_range_gone(r, range(100, 20)));
    assert(!rc_registry_content_current(r, &fresh));
    OK(rc_registry_source(r, 79, range(100, 20), &replacement));
    OK(rc_registry_observe(r, replacement, 1, &fresh));
    rc_registry_fail_closed(r);
    assert(!rc_registry_content_current(r, &fresh));
    rc_registry_destroy(r);
}
int main(void) {
    retained_byte_provenance_never_grants_read_permission();
    compatibility_intervals_do_not_certify_source_bytes();
    compatibility_brackets_reject_races_and_replaced_sources();
    compatibility_intervals_refuse_unpositioned_alias_coverage();
    compatibility_intervals_refuse_shared_image_aliases();
    compatibility_interval_entry_refusals_are_counted();
    source_revalidation_skips_unrelated_authority_and_stops_at_writes();
    ordered_capture_owns_old_version_after_future_writes();
    ordered_declarations_cannot_hide_predecessors();
    immutable_effect_inputs_need_only_a_structural_range_check();
    existing_declarations_expand_and_complete_without_clean_gaps();
    visibility_cutoffs_preserve_newer_authority();
    visibility_completion_and_mapping_cutoffs();
    visibility_capacity_is_atomic_and_blockers_are_classified();
    visibility_keeps_provider_classes_separate();
    bounded_source_notifications();
    registration_history_and_captured_old_contents();
    interval_authority_covers_new_aliases();
    coalescence_and_partial_visibility();
    canceled_writers_and_held_authority();
    image_hold_extents_follow_native_identity();
    abandoned_writers_preserve_possible_effects();
    capacity_is_backpressure_and_writer_failure_is_closed();
    backing_identity_and_mapping_replacement();
    ready_validation_and_churn_are_not_history_scans();
    source_lease_ownership_and_replacement_churn();
    retired_queue_visits_only_obsolete_records();
    descriptor_churn_preserves_only_reachable_history();
    adjacent_foreign_authority_has_one_interval();
    puts("resource core registry checks passed");
    return 0;
}

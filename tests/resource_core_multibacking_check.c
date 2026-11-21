#include <assert.h>
#include <stdio.h>

#define RC_REGISTRY_SINGLE_THREADED 1
#include "../runtime/resource_core_registry.c"

#define OK(call) assert((call) == RC_REGISTRY_OK)

static void mixed_write_and_split(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){8, 3, 4});
    assert(r);
    RcSourceRange a = {2, 100, 80}, b = {3, 100, 80}, c = {4, 100, 80};
    RcRegistryHandle ah, bh, ch;
    OK(rc_registry_source(r, 1, a, &ah));
    OK(rc_registry_source(r, 2, b, &bh));
    OK(rc_registry_source(r, 3, c, &ch));
    RcRegistryObservation old_a, old_b, old_c;
    OK(rc_registry_observe(r, ah, 1, &old_a));
    OK(rc_registry_observe(r, bh, 1, &old_b));
    OK(rc_registry_observe(r, ch, 1, &old_c));

    RcRegistryWriteExtent extents[] = {{a, RC_REGISTRY_BUFFER}, {b, RC_REGISTRY_BUFFER}, {c, RC_REGISTRY_IMAGE}};
    RcRegistryWrite same[3], late;
    RcRegistryPosition write = {1, 5}, later = {1, 9};
    OK(rc_registry_write_begin_extents(r, extents + 2, 1, later, &late));
    assert(rc_registry_write_begin_extents(r, extents, 3, write, same) == RC_REGISTRY_CAPACITY);
    assert(!same[0].id && !same[1].id && !same[2].id && rc_registry_stats(r).writers == 1);
    assert(rc_registry_observation_current(r, &old_a));
    OK(rc_registry_write_begin_extents(r, extents, 2, write, same));
    OK(rc_registry_declare_through(r, (RcRegistryPosition){1, 12}));
    assert(rc_registry_write_begin_extents(r, extents, 1, write, &same[2]) == RC_REGISTRY_STALE && !same[2].id);
    OK(rc_registry_read_range_at(r, a, (RcRegistryPosition){1, 3}));
    assert(rc_registry_read_range_at(r, a, write) == RC_REGISTRY_BLOCKED);
    assert(rc_registry_read_range_at(r, b, write) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_read_range_at(r, c, write));

    OK(rc_registry_handoff_through(r, write));
    RcRegistryWrite wrong_split[] = {same[0], late};
    assert(rc_registry_write_submit_extents(r, wrong_split, 2, 1) == RC_REGISTRY_STALE);
    assert(rc_registry_stats(r).writer_pending == 3);
    OK(rc_registry_write_submit_extents(r, same, 2, 1));
    RcSourceRange parts[] = {{2, 100, 10}, {2, 120, 10}, {2, 140, 10}, {3, 110, 15}};
    RcRegistryWriteCompletion complete[] = {{same[0], parts, parts, 3, 3}, {same[1], parts + 3, parts + 3, 1, 1}};
    assert(rc_registry_write_complete_extents(r, complete, 2) == RC_REGISTRY_CAPACITY);
    assert(rc_registry_stats(r).writer_submitted == 2 && rc_registry_content_current(r, &old_a));
    complete[0].written_count = 1;
    complete[0].uncertain_count = 0;
    complete[1].written = parts;
    assert(rc_registry_write_complete_extents(r, complete, 2) == RC_REGISTRY_INVALID);
    assert(rc_registry_stats(r).writer_submitted == 2 && rc_registry_content_current(r, &old_b));
    complete[1].written = parts + 3;
    OK(rc_registry_write_complete_extents(r, complete, 2));
    OK(rc_registry_write_cancel_extents(r, &late, 1));
    assert(!rc_registry_content_current(r, &old_a) && !rc_registry_content_current(r, &old_b));
    assert(rc_registry_content_current(r, &old_c));
    RcRegistryObservation ready;
    OK(rc_registry_observe(r, ah, 2, &ready));
    assert(rc_registry_observe(r, bh, 2, &ready) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_read_range_at(r, (RcSourceRange){3, 140, 20}, (RcRegistryPosition){1, 11}));
    OK(rc_registry_observe(r, ch, 2, &ready));
    rc_registry_destroy(r);
}

static void guard_after_seal_and_loss(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){4, 4, 4});
    assert(r);
    RcSourceRange a = {2, 20, 40}, new_mapping = {57, 0, 32}, clean = {58, 0, 32};
    RcSourceRange raw = {RC_GUEST_COMPATIBILITY_BACKING, 0, 32};
    RcRegistryHandle ah, nh;
    OK(rc_registry_source(r, 1, a, &ah));
    RcRegistryWrite guard, second;
    RcRegistryPosition at = {1, 5};
    OK(rc_registry_write_guard_begin_at(r, RC_REGISTRY_BUFFER, at, &guard));
    OK(rc_registry_declare_through(r, (RcRegistryPosition){1, 8}));
    assert(rc_registry_write_guard_begin_at(r, RC_REGISTRY_BUFFER, at, &second) == RC_REGISTRY_STALE);
    assert(!second.id);
    OK(rc_registry_source(r, 2, new_mapping, &nh));
    OK(rc_registry_read_range_at(r, new_mapping, (RcRegistryPosition){1, 2}));
    OK(rc_registry_read_range_at(r, raw, at));
    assert(rc_registry_read_range_at(r, new_mapping, at) == RC_REGISTRY_BLOCKED);
    RcRegistryBindingObservation binding;
    assert(rc_registry_binding_begin(r, nh, 1, (RcRegistryPosition){1, 7}, &binding) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_handoff_through(r, at));
    OK(rc_registry_write_submit(r, guard, 1));
    OK(rc_registry_write_abandon(r, guard, 1));
    assert(rc_registry_stats(r).writer_foreign == 1);
    for (int i = 0; i < 8; ++i) OK(rc_registry_cpu_changed(r, a));
    RcRegistryChange changes[4];
    uint64_t cursor = 0;
    int overflow = 0;
    assert(rc_registry_changes(r, &cursor, changes, 4, &overflow) && overflow);
    assert(rc_registry_reconcile_cpu(r, new_mapping) == RC_REGISTRY_BUSY);
    RcRegistryVisibility vis;
    OK(rc_registry_visibility_begin(r, RC_REGISTRY_VISIBILITY_COMPLETED, &vis));
    OK(rc_registry_visibility_apply(r, &vis, new_mapping));
    assert(rc_registry_read_range_at(r, new_mapping, (RcRegistryPosition){1, 7}) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_read_range_at(r, raw, (RcRegistryPosition){1, 7}));

    RcSourceRange actual[] = {a, new_mapping};
    RcRegistryWriteCompletion recovery = {guard, actual, actual + 1, 2, 1};
    OK(rc_registry_write_complete_extents(r, &recovery, 1));
    OK(rc_registry_read_range_at(r, a, (RcRegistryPosition){1, 7}));
    OK(rc_registry_read_range_at(r, clean, (RcRegistryPosition){1, 7}));
    assert(rc_registry_read_range_at(r, new_mapping, (RcRegistryPosition){1, 7}) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_visibility_begin(r, RC_REGISTRY_VISIBILITY_COMPLETED, &vis));
    OK(rc_registry_visibility_apply(r, &vis, new_mapping));
    OK(rc_registry_read_range_at(r, new_mapping, (RcRegistryPosition){1, 7}));

    OK(rc_registry_write_guard_begin_at(r, RC_REGISTRY_UNKNOWN, (RcRegistryPosition){2, 3}, &second));
    OK(rc_registry_declare_through(r, (RcRegistryPosition){2, 5}));
    assert(rc_registry_read_range_at(r, clean, (RcRegistryPosition){2, 4}) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_cancel(r, second));
    OK(rc_registry_read_range_at(r, clean, (RcRegistryPosition){2, 4}));
    OK(rc_registry_write_guard_begin_at(r, RC_REGISTRY_BUFFER, (RcRegistryPosition){0}, &second));
    assert(rc_registry_read_range_at(r, clean, (RcRegistryPosition){2, 4}) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_cancel(r, second));
    RcRegistryWrite earliest={0};
    for (int i=0;i<8;i++) {
        OK(rc_registry_write_guard_begin_at(r,RC_REGISTRY_BUFFER,(RcRegistryPosition){0},&second));
        OK(rc_registry_write_guard_accumulate(r,second));
        assert(rc_registry_stats(r).writer_foreign==1);
        if (!i) earliest=rc_registry_unresolved_guard(r);
    }
    assert(earliest.id && rc_registry_unresolved_guard(r).id!=earliest.id);
    recovery=(RcRegistryWriteCompletion){.ticket=earliest};
    assert(rc_registry_write_complete_extents(r,&recovery,1)==RC_REGISTRY_STALE);
    recovery=(RcRegistryWriteCompletion){.ticket=rc_registry_unresolved_guard(r),.written=&a,.written_count=1};
    OK(rc_registry_write_complete_extents(r,&recovery,1));
    assert(!rc_registry_unresolved_guard(r).id);
    OK(rc_registry_read_range_at(r,clean,(RcRegistryPosition){2,4}));
    rc_registry_destroy(r);
}

static void source_vector_commit_and_deadline(void) {
    RcRegistry *r=rc_registry_create((RcRegistryConfig){8,8,4});
    assert(r);
    RcSourceRange a={4,100,32}, b={5,100,48};
    RcRegistrySourceAccess access[2]={0};
    RcRegistrySourceProof first[2], newer[2];
    OK(rc_registry_source_access_acquire(r,12,a,&access[0]));
    OK(rc_registry_source_access_acquire(r,12,b,&access[1]));
    RcRegistryWrite producer;
    OK(rc_registry_write_begin_at(r,b,RC_REGISTRY_BUFFER,(RcRegistryPosition){1,9},&producer));
    OK(rc_registry_declare_through(r,(RcRegistryPosition){1,12}));
    OK(rc_registry_source_access_observe_extents(r,access,2,1,(RcRegistryPosition){1,3},first));
    assert(rc_registry_source_proofs_current(r,first,2,(RcRegistryPosition){1,3}));
    int changed;
    OK(rc_registry_source_access_commit_extents(r,access,first,2,0,&changed));
    assert(changed);
    RcRegistryReadObservation anchor;
    OK(rc_registry_source_access_read_extents(r,access,2,1,(RcRegistryPosition){1,5},&anchor));
    assert(anchor.range.backing==a.backing && anchor.next_write.ordinal==9);
    RcRegistryReadSchedule schedule;
    OK(rc_registry_source_access_schedule_extents(r,access,2,1,(RcRegistryPosition){1,5},&schedule));
    assert(schedule.next_read.ordinal==8);
    assert(rc_registry_source_access_read_extents(r,access,2,1,(RcRegistryPosition){1,9},&anchor)==RC_REGISTRY_BLOCKED);
    assert(!anchor.handle.id);
    OK(rc_registry_source_access_observe_extents(r,access,2,1,(RcRegistryPosition){1,4},newer));
    OK(rc_registry_cpu_changed(r,b));
    assert(!rc_registry_source_proofs_current(r,newer,2,(RcRegistryPosition){1,4}));
    assert(rc_registry_source_access_commit_extents(r,access,newer,2,0,&changed)==RC_REGISTRY_STALE);
    assert(!changed && rc_registry_source_access_content(&access[0]).ordered.position.ordinal==3);
    OK(rc_registry_write_cancel(r,producer));
    OK(rc_registry_source_access_release(r,&access[0]));
    OK(rc_registry_source_access_release(r,&access[1]));
    rc_registry_destroy(r);
}

static void exact_image_readback(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 8, 2});
    const RcInterval mapped[] = {{100, 8, {2, 30, 12}}, {112, 9, {3, 40, 8}}};
    RcRegistryWrite holds[2], raw;
    RcSourceRange guest = {RC_GUEST_COMPATIBILITY_BACKING, 100, 20};
    OK(rc_registry_image_hold_begin_extents(r, 49, mapped, 2, holds));
    OK(rc_registry_hold_begin(r, guest, RC_REGISTRY_IMAGE, &raw));
    OK(rc_registry_image_hold_bind(r, raw, 49));
    RcRegistryImageWriteExtent targets[] = {{{100, 0, guest}, raw}, {mapped[0], holds[0]}, {mapped[1], holds[1]}};
    const RcSourceRange copied[] = {{1, 103, 5}, {1, 110, 7}};
    RcRegistryHandle transferred, padding, old_tail, new_alias;
    OK(rc_registry_source(r, 1, (RcSourceRange){3, 40, 5}, &transferred));
    OK(rc_registry_source(r, 2, (RcSourceRange){2, 38, 2}, &padding));
    OK(rc_registry_source(r, 3, (RcSourceRange){3, 45, 3}, &old_tail));
    OK(rc_registry_source(r, 4, (RcSourceRange){4, 40, 8}, &new_alias));
    RcRegistryObservation proof, successor;
    OK(rc_registry_observe(r, new_alias, 1, &successor));
    targets[2].interval.mapping++;
    assert(rc_registry_image_hold_readback(r, 49, targets, 3, copied, 2) == RC_REGISTRY_STALE);
    assert(rc_registry_stats(r).writer_holds == 3);
    targets[2].interval = mapped[1];
    assert(rc_registry_image_hold_readback(r, 50, targets, 3, copied, 2) == RC_REGISTRY_STALE);
    OK(rc_registry_image_hold_readback(r, 49, targets, 3, copied, 2));
    assert(!rc_registry_stats(r).writer_holds && rc_registry_stats(r).writer_foreign == 3);
    OK(rc_registry_observe(r, transferred, 1, &proof));
    assert(rc_registry_observe(r, padding, 1, &proof) == RC_REGISTRY_BLOCKED);
    assert(rc_registry_observe(r, old_tail, 1, &proof) == RC_REGISTRY_BLOCKED);
    assert(rc_registry_observation_current(r, &successor));

    RcSourceRange padding_store = {2, 38, 2}, younger_range = {3, 42, 1}, tail_part = {3, 45, 2};
    OK(rc_registry_cpu_changed(r, padding_store));
    assert(rc_registry_observe(r, padding, 1, &proof) == RC_REGISTRY_BLOCKED);
    RcRegistryWrite pending;
    OK(rc_registry_write_begin(r, younger_range, RC_REGISTRY_IMAGE, &pending));
    RcRegistryVisibility overwrite;
    OK(rc_registry_visibility_begin(r, RC_REGISTRY_VISIBILITY_COMPLETED, &overwrite));
    overwrite.writer_kinds = 1u << RC_REGISTRY_IMAGE;
    // A real CPU store can settle only the covered earlier bytes, even if a producer was reserved before it.
    OK(rc_registry_write_submit(r, pending, 1));
    OK(rc_registry_write_complete(r, pending, 0));
    OK(rc_registry_visibility_apply(r, &overwrite, padding_store));
    OK(rc_registry_visibility_apply(r, &overwrite, tail_part));
    OK(rc_registry_visibility_apply(r, &overwrite, younger_range));
    OK(rc_registry_observe(r, padding, 1, &proof));
    assert(rc_registry_observe(r, old_tail, 1, &proof) == RC_REGISTRY_BLOCKED);
    assert(rc_registry_observe(r, transferred, 1, &proof) == RC_REGISTRY_BLOCKED);
    assert(rc_registry_observation_current(r, &successor));
    rc_registry_destroy(r);

    r = rc_registry_create((RcRegistryConfig){8, 1, 2});
    OK(rc_registry_image_hold_begin_extents(r, 49, mapped, 1, holds));
    targets[1] = (RcRegistryImageWriteExtent){mapped[0], holds[0]};
    OK(rc_registry_image_hold_readback(r, 49, targets + 1, 1, copied, 2));
    assert(!rc_registry_stats(r).writer_holds && rc_registry_stats(r).writer_foreign == 1);
    rc_registry_destroy(r);
}

static void compact_image_readback(void) {
    enum { ROWS = 24000 };
    RcRegistry *r = rc_registry_create((RcRegistryConfig){2048, 4, 2});
    assert(r);
    RcSourceRange guest = {1, 4096, ROWS * 4}, mapped = {7, 120000, ROWS * 4};
    RcInterval interval = {guest.offset, 13, mapped};
    RcRegistryWrite can = {0}, raw = {0};
    OK(rc_registry_image_hold_begin_extents(r, 349, &interval, 1, &can));
    OK(rc_registry_hold_begin(r, guest, RC_REGISTRY_IMAGE, &raw));
    OK(rc_registry_image_hold_bind(r, raw, 349));
    RcRegistryImageWriteExtent targets[] = {{interval,can},{{guest.offset,0,guest},raw}};
    RcSourceRange *visible = malloc(ROWS * sizeof *visible);
    assert(visible);
    for (uint32_t i = 0; i < ROWS; ++i) visible[i] = (RcSourceRange){1, guest.offset + 4 * i + 1, 2};
    OK(rc_registry_image_hold_readback(r, 349, targets, 2, visible, ROWS));
    assert(rc_registry_stats(r).writers == 2 && rc_registry_stats(r).writer_foreign == 2);
    assert(!rc_registry_stats(r).writer_holds && r->foreign_span_count == ROWS + 1);
    assert(r->writers[can.index].foreign_coverage == r->writers[raw.index].foreign_coverage);
    assert(rc_registry_image_hold_release_extents(r,&can,1,1) == RC_REGISTRY_STALE);
    assert(rc_registry_hold_release(r,raw,1) == RC_REGISTRY_STALE);
    free(visible);

    RcRegistryPosition read = {2,2};
    OK(rc_registry_declare_through(r,read));
    for (uint32_t i = 0; i < ROWS; i += 19) {
        for (unsigned domain = 0; domain < 2; ++domain) {
            RcSourceRange extent = domain ? mapped : guest;
            extent.offset += 4 * i;
            assert(rc_registry_read_range_at(r,(RcSourceRange){extent.backing,extent.offset,1},read) == RC_REGISTRY_BLOCKED);
            OK(rc_registry_read_range_at(r,(RcSourceRange){extent.backing,extent.offset+1,2},read));
            assert(rc_registry_read_range_at(r,(RcSourceRange){extent.backing,extent.offset+2,2},read) == RC_REGISTRY_BLOCKED);
        }
    }
    RcSourceRange actual[] = {{7,mapped.offset+1,2},{7,mapped.offset+5,2}};
    RcRegistrySourceAccess meaningful = {0}, still_opaque = {0};
    RcRegistrySourceProof proof;
    OK(rc_registry_source_access_acquire_shaped(r,19,(RcSourceRange){7,mapped.offset,8},13,actual,2,&meaningful));
    OK(rc_registry_source_access_observe(r,&meaningful,1,read,&proof));
    actual[1].offset--;
    OK(rc_registry_source_access_acquire_shaped(r,20,(RcSourceRange){7,mapped.offset,8},13,actual,2,&still_opaque));
    assert(rc_registry_source_access_observe(r,&still_opaque,1,read,&proof) == RC_REGISTRY_BLOCKED);

    RcRegistryWrite guard;
    OK(rc_registry_write_guard_begin_at(r,RC_REGISTRY_UNKNOWN,(RcRegistryPosition){0},&guard));
    assert(rc_registry_read_range_at(r,actual[0],read) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_cancel(r,guard));
    OK(rc_registry_read_range_at(r,actual[0],read));

    RcSourceRange exact_store = {7,mapped.offset+3,1};
    OK(rc_registry_reconcile_cpu(r,exact_store));
    OK(rc_registry_read_range_at(r,exact_store,read));
    assert(rc_registry_read_range_at(r,(RcSourceRange){7,mapped.offset+4,1},read) == RC_REGISTRY_BLOCKED);
    assert(rc_registry_read_range_at(r,(RcSourceRange){1,guest.offset+3,1},read) == RC_REGISTRY_BLOCKED);
    assert(rc_registry_stats(r).writer_foreign == 2);

    RcRegistryVisibility stale;
    OK(rc_registry_visibility_begin(r,RC_REGISTRY_VISIBILITY_COMPLETED,&stale));
    stale.writer_kinds = 1u << RC_REGISTRY_IMAGE;
    OK(rc_registry_range_gone(r,(RcSourceRange){7,mapped.offset+8,1}));
    OK(rc_registry_visibility_apply(r,&stale,mapped));
    assert(rc_registry_read_range_at(r,(RcSourceRange){7,mapped.offset+8,1},read) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_read_range_at(r,(RcSourceRange){7,mapped.offset+12,1},read));
    OK(rc_registry_read_range_at(r,(RcSourceRange){8,mapped.offset+8,1},read)); // The successor has independent backing.
    assert(rc_registry_read_range_at(r,(RcSourceRange){1,guest.offset+12,1},read) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_reconcile_cpu(r,mapped));
    OK(rc_registry_reconcile_cpu(r,guest));
    assert(!rc_registry_stats(r).writers && !r->foreign_span_count);
    OK(rc_registry_read_range_at(r,mapped,read));
    OK(rc_registry_read_range_at(r,guest,read));
    OK(rc_registry_source_access_release(r,&meaningful));
    OK(rc_registry_source_access_release(r,&still_opaque));
    rc_registry_destroy(r);
}

static void image_readback_shape_capacity(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){1, 2, 1});
    RcInterval map = {4096, 13, {7, 100, 128}};
    RcSourceRange guest = {1,4096,128};
    RcRegistryWrite can, raw;
    OK(rc_registry_image_hold_begin_extents(r,20,&map,1,&can));
    OK(rc_registry_hold_begin(r,guest,RC_REGISTRY_IMAGE,&raw));
    OK(rc_registry_image_hold_bind(r,raw,20));
    RcRegistryImageWriteExtent targets[] = {{map,can},{{guest.offset,0,guest},raw}};
    RcSourceRange visible[32];
    for (uint32_t i = 0; i < 32; ++i) visible[i] = (RcSourceRange){1,guest.offset+4*i+1,2};
    assert(rc_registry_image_hold_readback(r,20,targets,2,visible,32) == RC_REGISTRY_CAPACITY);
    assert(rc_registry_stats(r).writer_holds == 2 && !r->foreign_span_count);
    OK(rc_registry_image_hold_readback(r,20,targets,2,visible,24));
    assert(rc_registry_stats(r).writer_foreign == 2 && r->foreign_span_count == 25);
    RcRegistryPosition read = {1,2};
    OK(rc_registry_declare_through(r,read));
    RcSourceRange first = {7,100,1};
    assert(rc_registry_reconcile_cpu(r,first) == RC_REGISTRY_CAPACITY);
    assert(rc_registry_read_range_at(r,first,read) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_read_range_at(r,(RcSourceRange){7,101,2},read));
    RcRegistryVisibility old;
    OK(rc_registry_visibility_begin(r,RC_REGISTRY_VISIBILITY_COMPLETED,&old));
    old.writer_kinds=1u<<RC_REGISTRY_IMAGE;
    OK(rc_registry_range_gone(r,(RcSourceRange){7,104,1}));
    OK(rc_registry_visibility_apply(r,&old,map.range));
    assert(rc_registry_read_range_at(r,first,read)==RC_REGISTRY_BLOCKED); // Safe whole-owner fallback at the shape limit.
    OK(rc_registry_visibility_apply(r,&old,guest));
    assert(r->foreign_span_count==25 && rc_registry_stats(r).writer_foreign==1);
    OK(rc_registry_reconcile_cpu(r,map.range));
    assert(!r->foreign_span_count && !rc_registry_stats(r).writers && !rc_registry_stats(r).fail_closed);
    rc_registry_destroy(r);
}

int main(void) {
    mixed_write_and_split();
    guard_after_seal_and_loss();
    source_vector_commit_and_deadline();
    exact_image_readback();
    compact_image_readback();
    image_readback_shape_capacity();
    puts("resource registry: multi-backing writes, split handoff, exact completion, opaque recovery and source vectors passed");
}

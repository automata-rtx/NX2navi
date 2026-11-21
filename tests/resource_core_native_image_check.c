#include <assert.h>
#include <stdio.h>

#define RC_REGISTRY_SINGLE_THREADED 1
#include "../runtime/resource_core_registry.c"

#define OK(call) assert((call) == RC_REGISTRY_OK)

static RcRegistryPosition at(uint64_t ordinal) { return (RcRegistryPosition){1, ordinal}; }

static RcInterval interval(uint64_t backing, uint64_t offset, uint64_t size, uint64_t address, uint64_t mapping) {
    return (RcInterval){.address = address, .mapping = mapping, .range = {backing, offset, size}};
}

static RcRegistryImageWriteExtent hold(RcRegistry *r, uint64_t image, RcInterval captured) {
    RcRegistryImageWriteExtent owner = {.interval = captured};
    OK(rc_registry_image_hold_begin_extents(r, image, &captured, 1, &owner.hold));
    return owner;
}

static RcRegistryWrite declare_image(RcRegistry *r, RcInterval captured, RcRegistryPosition output) {
    RcRegistryWrite writer;
    OK(rc_registry_write_begin_at(r, captured.range, RC_REGISTRY_IMAGE, output, &writer));
    OK(rc_registry_image_producer_add(r, writer, output));
    return writer;
}

static void publish(RcRegistry *r, uint64_t image, RcRegistryImageWriteExtent owner,
                    const RcSourceRange *written, uint32_t count, RcRegistryPosition output) {
    const RcRegistryNativeImageWrite write = {owner, written, count};
    OK(rc_registry_native_image_write_fragments(r, image, &write, 1, output));
}

static void record(RcRegistry *r, RcRegistryWrite writer, uint64_t image, RcRegistryImageWriteExtent owner,
                   RcRegistryPosition output) {
    const RcRegistryImageBinding binding = {image, owner.hold};
    OK(rc_registry_image_producer_resolve(r, writer, output, &binding, 0));
}

static void expect_fragment(RcRegistryNativeImageFragment f, uint32_t demand, uint64_t image,
                            RcRegistryWrite owner, RcInterval source, RcRegistryPosition output) {
    assert(f.demand == demand && f.binding.image == image);
    assert(f.binding.hold.id == owner.id && f.binding.hold.index == owner.index);
    assert(f.interval.address == source.address && f.interval.mapping == source.mapping);
    assert(rc_range_equal(f.interval.range, source.range) && !rc_registry_position_compare(f.output, output));
}

static void latest_exact_sparse_alias(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 16, 4});
    assert(r);
    RcInterval a = interval(2, 100, 60, 0x1000, 10), b = interval(2, 100, 60, 0x2000, 20);
    RcInterval demands[] = {interval(2, 100, 60, 0x3000, 30), interval(2, 125, 5, 0x9000, 40)};
    RcRegistryImageWriteExtent old = hold(r, 100, a), newer = hold(r, 200, b);
    RcRegistryWrite first = declare_image(r, a, at(3)), second = declare_image(r, b, at(9));
    const RcSourceRange center = {2, 120, 20};
    OK(rc_registry_declare_through(r, at(12)));
    publish(r, 100, old, &a.range, 1, at(3));
    assert(!rc_registry_native_image_has_prior_projection(r, demands, 2, at(2)));
    assert(rc_registry_native_image_has_prior_projection(r, demands, 2, at(4)));
    assert(!rc_registry_native_image_covers(r, 100, demands, 1, at(4)));
    record(r, first, 100, old, at(3));
    publish(r, 200, newer, &center, 1, at(9));
    RcRegistryNativeImageFragment output[4];
    uint32_t count;
    assert(rc_registry_native_image_fragments(r, demands, 1, at(10), output, 4, &count) == RC_REGISTRY_BLOCKED && !count);
    record(r, second, 200, newer, at(9));
    OK(rc_registry_native_image_fragments(r, demands, 1, at(8), output, 4, &count));
    assert(count == 1);
    assert(rc_registry_native_image_covers(r, 100, demands, 1, at(8)));
    assert(!rc_registry_native_image_covers(r, 200, demands, 1, at(8)));
    expect_fragment(output[0], 0, 100, old.hold, a, at(3));
    assert(rc_registry_native_image_fragments(r, demands, 2, at(10), output, 3, &count) == RC_REGISTRY_CAPACITY && !count);
    OK(rc_registry_native_image_fragments(r, demands, 2, at(10), output, 4, &count));
    assert(count == 4);
    assert(!rc_registry_native_image_covers(r, 100, demands, 2, at(10)));
    assert(!rc_registry_native_image_covers(r, 200, demands, 2, at(10)));
    assert(rc_registry_native_image_covers(r, 200, demands + 1, 1, at(10)));
    expect_fragment(output[0], 0, 100, old.hold, interval(2, 100, 20, 0x1000, 10), at(3));
    expect_fragment(output[1], 0, 200, newer.hold, interval(2, 120, 20, 0x2014, 20), at(9));
    expect_fragment(output[2], 0, 100, old.hold, interval(2, 140, 20, 0x1028, 10), at(3));
    expect_fragment(output[3], 1, 200, newer.hold, interval(2, 125, 5, 0x2019, 20), at(9));

    OK(rc_registry_handoff_through(r, at(9)));
    // The command claim is consumed by its confirmed persistent image hold after native handoff.
    OK(rc_registry_write_cancel(r, first));
    OK(rc_registry_write_cancel(r, second));
    rc_registry_image_hold_forget(r, newer.hold);
    assert(!rc_registry_native_image_covers(r, 200, demands + 1, 1, at(10)));
    assert(rc_registry_native_image_fragments(r, demands, 1, at(10), output, 4, &count) == RC_REGISTRY_BLOCKED);
    RcInterval untouched = interval(2, 100, 20, 0x9000, 44);
    OK(rc_registry_native_image_fragments(r, &untouched, 1, at(10), output, 4, &count));
    // A replacement mapping with the same guest address cannot consume the old backing's provider.
    RcInterval replacement = interval(22, 100, 20, untouched.address, 45);
    OK(rc_registry_range_gone(r, untouched.range));
    assert(rc_registry_native_image_fragments(r, &replacement, 1, at(10), output, 4, &count) == RC_REGISTRY_BLOCKED);
    assert(!rc_registry_native_image_covers(r, 100, &replacement, 1, at(10)));
    assert(!rc_registry_native_image_has_prior_projection(r, &replacement, 1, at(10)));
    OK(rc_registry_native_image_fragments(r, &untouched, 1, at(10), output, 4, &count));
    OK(rc_registry_image_hold_release_extents(r, &newer.hold, 1, 0));
    assert(rc_registry_native_image_fragments(r, demands, 1, at(10), output, 4, &count) == RC_REGISTRY_BLOCKED);
    assert(!rc_registry_native_image_covers(r, 100, demands, 1, at(10)));
    rc_registry_destroy(r);
}

static void canceled_and_settled_writes(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 16, 4});
    assert(r);
    RcInterval a = interval(2, 100, 60, 0x1000, 10);
    RcRegistryImageWriteExtent owner = hold(r, 100, a);
    RcRegistryWrite first = declare_image(r, a, at(3)), discarded = declare_image(r, a, at(7));
    RcRegistryWrite cpu_visible, later_guard;
    const RcSourceRange middle = {2, 115, 4};
    OK(rc_registry_write_begin_at(r, middle, RC_REGISTRY_BUFFER, at(11), &cpu_visible));
    OK(rc_registry_write_guard_begin_at(r, RC_REGISTRY_UNKNOWN, at(15), &later_guard));
    OK(rc_registry_declare_through(r, at(18)));
    publish(r, 100, owner, &a.range, 1, at(3));
    record(r, first, 100, owner, at(3));
    publish(r, 100, owner, &middle, 1, at(7));
    record(r, discarded, 100, owner, at(7));
    RcRegistryNativeImageFragment output[3];
    uint32_t count;
    OK(rc_registry_native_image_fragments(r, &a, 1, at(8), output, 3, &count));
    assert(count == 3 && output[1].output.ordinal == 7);
    OK(rc_registry_handoff_through(r, at(3)));
    OK(rc_registry_write_cancel(r, first));
    OK(rc_registry_write_abandon(r, discarded, 0));
    OK(rc_registry_handoff_through(r, at(9)));
    OK(rc_registry_native_image_fragments(r, &a, 1, at(10), output, 3, &count));
    assert(count == 1 && output[0].output.ordinal == 3);
    assert(rc_registry_native_image_covers(r, 100, &a, 1, at(10)));
    assert(rc_registry_native_image_fragments(r, &a, 1, at(12), output, 3, &count) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_handoff_through(r, at(11)));
    OK(rc_registry_write_submit(r, cpu_visible, 1));
    OK(rc_registry_write_complete(r, cpu_visible, 1));
    RcInterval left = interval(2, 100, 15, 0x4000, 40), right = interval(2, 119, 41, 0x4040, 40);
    OK(rc_registry_native_image_fragments(r, &left, 1, at(14), output, 3, &count));
    OK(rc_registry_native_image_fragments(r, &right, 1, at(14), output, 3, &count));
    assert(rc_registry_native_image_fragments(r, &a, 1, at(14), output, 3, &count) == RC_REGISTRY_BLOCKED);
    assert(rc_registry_native_image_fragments(r, &right, 1, at(16), output, 3, &count) == RC_REGISTRY_BLOCKED);
    assert(!rc_registry_native_image_covers(r, 100, &right, 1, at(16)));
    OK(rc_registry_write_cancel(r, later_guard));
    OK(rc_registry_native_image_fragments(r, &right, 1, at(16), output, 3, &count));
    OK(rc_registry_cpu_changed(r, (RcSourceRange){2, 121, 4}));
    assert(rc_registry_native_image_fragments(r, &right, 1, at(16), output, 3, &count) == RC_REGISTRY_BLOCKED);
    right.range.offset = 125;
    right.range.size = 35;
    OK(rc_registry_native_image_fragments(r, &right, 1, at(16), output, 3, &count));
    rc_registry_destroy(r);
}

static void contradictory_output_cannot_be_proved_after_handoff(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 12, 4});
    assert(r);
    RcInterval a = interval(2, 100, 60, 0x1000, 10), b = interval(2, 120, 20, 0x2000, 20);
    RcRegistryImageWriteExtent owner = hold(r, 100, a), other = hold(r, 200, b);
    RcRegistryWrite first = declare_image(r, a, at(3)), overlap = declare_image(r, b, at(3));
    OK(rc_registry_declare_through(r, at(8)));
    publish(r, 100, owner, &a.range, 1, at(3));
    publish(r, 200, other, &b.range, 1, at(3));
    record(r, first, 100, owner, at(3));
    record(r, overlap, 200, other, at(3));
    RcRegistryNativeImageFragment output[3];
    uint32_t count;
    assert(rc_registry_native_image_fragments(r, &a, 1, at(4), output, 3, &count) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_cancel(r, overlap));
    OK(rc_registry_native_image_fragments(r, &a, 1, at(4), output, 3, &count));
    rc_registry_image_producer_invalidate(r, first);
    OK(rc_registry_handoff_through(r, at(3)));
    OK(rc_registry_write_cancel(r, first));
    assert(rc_registry_native_image_fragments(r, &a, 1, at(4), output, 3, &count) == RC_REGISTRY_BLOCKED && !count);
    rc_registry_destroy(r);
}

static void unprojected_actual_output_survives_claim_retirement(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 12, 4});
    assert(r);
    RcInterval a = interval(2, 100, 60, 0x1000, 10);
    RcInterval partial = interval(2, 120, 20, 0x1014, 10), left = interval(2, 100, 20, 0x1000, 10);
    RcRegistryImageWriteExtent owner = hold(r, 100, a);
    RcRegistryWrite first = declare_image(r, a, at(3));
    RcRegistryWrite canceled = declare_image(r, partial, at(7));
    RcRegistryWrite actual = declare_image(r, partial, at(11));
    RcRegistryWrite unknown = declare_image(r, partial, at(15));
    OK(rc_registry_declare_through(r, at(16)));
    publish(r, 100, owner, &a.range, 1, at(3));
    record(r, first, 100, owner, at(3));
    record(r, canceled, 100, owner, at(7));
    RcRegistryNativeImageFragment output[3];
    uint32_t count;
    assert(rc_registry_native_image_fragments(r, &a, 1, at(8), output, 3, &count) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_abandon(r, canceled, 0));
    OK(rc_registry_native_image_fragments(r, &a, 1, at(8), output, 3, &count));
    record(r, actual, 100, owner, at(11));
    assert(rc_registry_native_image_fragments(r, &a, 1, at(12), output, 3, &count) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_handoff_through(r, at(11)));
    // A shared effect with an authoritative image owner can drop its command claim after the real handoff.
    OK(rc_registry_write_cancel(r, first));
    OK(rc_registry_write_cancel(r, actual));
    assert(rc_registry_native_image_fragments(r, &a, 1, at(12), output, 3, &count) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_native_image_fragments(r, &left, 1, at(12), output, 3, &count));
    // A tentative fragment report without positive destination binding must not survive known cancellation.
    publish(r, 100, owner, &left.range, 1, at(15));
    OK(rc_registry_image_producer_resolve(r, unknown, at(15), NULL, 0));
    assert(rc_registry_native_image_fragments(r, &left, 1, at(16), output, 3, &count) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_cancel(r, unknown));
    OK(rc_registry_native_image_fragments(r, &left, 1, at(16), output, 3, &count));
    rc_registry_destroy(r);
}

static void exact_fragment_capacity(void) {
    enum { SPANS = 4096 };
    RcRegistry *r = rc_registry_create((RcRegistryConfig){512, 8, 4});
    assert(r);
    RcInterval a = interval(2, 0, SPANS * 4, 0x10000, 10), b = interval(2, 0, SPANS * 4, 0x20000, 20);
    RcRegistryImageWriteExtent older = hold(r, 100, a), newer = hold(r, 200, b);
    RcRegistryWrite first = declare_image(r, a, at(3)), second = declare_image(r, b, at(5));
    RcSourceRange *meaningful = malloc(SPANS * sizeof *meaningful);
    RcRegistryNativeImageFragment *output = malloc(SPANS * 2 * sizeof *output);
    assert(meaningful && output);
    for (uint32_t i = 0; i < SPANS; ++i) meaningful[i] = (RcSourceRange){2, (uint64_t)i * 4, 2};
    OK(rc_registry_declare_through(r, at(6)));
    publish(r, 100, older, &a.range, 1, at(3));
    record(r, first, 100, older, at(3));
    publish(r, 200, newer, meaningful, SPANS, at(5));
    record(r, second, 200, newer, at(5));
    uint32_t count;
    assert(rc_registry_native_image_fragments(r, &a, 1, at(6), output, SPANS, &count) == RC_REGISTRY_CAPACITY && !count);
    OK(rc_registry_native_image_fragments(r, &a, 1, at(6), output, SPANS * 2, &count));
    assert(count == SPANS * 2 && rc_registry_stats(r).writers == 4);
    for (uint32_t i = 0; i < count; ++i) {
        assert(output[i].interval.range.offset == (uint64_t)i * 2 && output[i].interval.range.size == 2);
        assert(output[i].binding.image == (i & 1 ? 100 : 200));
    }
    free(output);
    rc_registry_destroy(r);

    r = rc_registry_create((RcRegistryConfig){1, 4, 4});
    assert(r);
    RcInterval other = interval(3, 0, 64, 0x30000, 30);
    older = hold(r, 100, a);
    newer = hold(r, 100, other);
    OK(rc_registry_declare_through(r, at(6)));
    const RcRegistryNativeImageWrite oversized[] = {{newer, &other.range, 1}, {older, meaningful, SPANS}};
    assert(rc_registry_native_image_write_fragments(r, 100, oversized, 2, at(5)) == RC_REGISTRY_CAPACITY);
    assert(!r->native_projection_spans && !rc_registry_stats(r).fail_closed);
    rc_registry_destroy(r);

    r = rc_registry_create((RcRegistryConfig){2, 4, 4});
    assert(r);
    older = hold(r, 100, a);
    first = declare_image(r, a, at(3));
    second = declare_image(r, a, at(5));
    OK(rc_registry_declare_through(r, at(8)));
    publish(r, 100, older, &a.range, 1, at(3));
    record(r, first, 100, older, at(3));
    const RcRegistryNativeImageWrite blocked = {older, meaningful, SPANS};
    assert(rc_registry_native_image_write_fragments(r, 100, &blocked, 1, at(5)) == RC_REGISTRY_CAPACITY);
    record(r, second, 100, older, at(5));
    RcRegistryNativeImageFragment single;
    assert(rc_registry_native_image_fragments(r, &a, 1, at(6), &single, 1, &count) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_cancel(r, second));
    OK(rc_registry_native_image_fragments(r, &a, 1, at(8), &single, 1, &count));
    assert(count == 1 && single.output.ordinal == 3);
    free(meaningful);
    rc_registry_destroy(r);
}

static void relocate_whole_bound_image(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 16, 4});
    assert(r);
    const RcInterval old_parts[] = {
        interval(2, 100, 40, 0x1000, 10),
        interval(3, 200, 60, 0x1028, 11),
    };
    RcRegistryWrite owners[2];
    OK(rc_registry_image_hold_begin_extents(r, 100, old_parts, 2, owners));
    const RcRegistryImageWriteExtent holds[] = {{old_parts[0], owners[0]}, {old_parts[1], owners[1]}};
    RcRegistryWrite old_writes[] = {declare_image(r, old_parts[0], at(3)), declare_image(r, old_parts[1], at(3))};
    RcRegistryWrite prior_unknown;
    OK(rc_registry_write_guard_begin_at(r, RC_REGISTRY_UNKNOWN, at(5), &prior_unknown));
    OK(rc_registry_declare_through(r, at(14)));
    assert(rc_registry_native_image_written(r, 100, old_parts, owners, 2, at(3)));
    const uint32_t initial_spans = r->native_projection_spans;
    r->native_projection_spans += image_span_room(r) - 1;
    assert(!rc_registry_native_image_whole_written(r, 100, holds, 2, at(3)));
    assert(rc_registry_image_binding_current(r, (RcRegistryImageBinding){100, owners[0]}));
    r->native_projection_spans = initial_spans;
    assert(rc_registry_native_image_whole_written(r, 100, holds, 2, at(3)));
    for (unsigned i = 0; i < 2; i++) {
        record(r, old_writes[i], 100, holds[i], at(3));
        assert(rc_registry_image_producer_projection_pending(r, old_writes[i], at(3),
            (RcRegistryImageBinding){100, owners[i]}));
    }
    assert(!rc_registry_native_image_has_prior_projection(r, old_parts, 2, at(4)));
    assert(rc_registry_native_image_relocate(r, 100, 101, holds, 2, at(3), at(4)) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_handoff_through(r, at(3)));
    assert(rc_registry_native_image_relocate(r, 100, 101, holds, 2, at(3), at(6)) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_cancel(r, prior_unknown));
    RcRegistryImageWriteExtent remap[] = {holds[0], holds[1]};
    remap[1].interval.mapping++;
    assert(rc_registry_native_image_relocate(r, 100, 101, remap, 2, at(3), at(6)) == RC_REGISTRY_STALE);
    assert(rc_registry_image_binding_current(r, (RcRegistryImageBinding){100, owners[0]}));
    assert(rc_registry_image_binding_current(r, (RcRegistryImageBinding){100, owners[1]}));
    assert(r->native_projection_spans == 2);
    assert(!rc_registry_native_image_has_prior_projection(r, old_parts, 2, at(6)));
    for (unsigned i = 0; i < 2; i++) {
        Writer *w = writer_of(r, old_writes[i]);
        image_position_prefix(w, 1);
        assert(w->image_positions[0].prefix_image.image == 100 && w->image_dirty == 1);
    }
    const uint32_t room = image_span_room(r);
    r->native_projection_spans += room;
    OK(rc_registry_native_image_relocate(r, 100, 101, holds, 2, at(3), at(6)));
    r->native_projection_spans -= room;
    assert(r->native_projection_spans == 2 && !r->native_image_outputs);
    for (unsigned i = 0; i < 2; i++) {
        assert(!rc_registry_image_binding_current(r, (RcRegistryImageBinding){100, owners[i]}));
        assert(rc_registry_image_binding_current(r, (RcRegistryImageBinding){101, owners[i]}));
        Writer *w = writer_of(r, old_writes[i]);
        assert(!w->image_dirty && w->image_positions[0].projected);
        image_position_prefix(w, 1);
        assert(w->image_positions[0].prefix_image.image == 101);
        // An unissued remainder cannot cancel this already completed old physical clone copy.
        assert(!rc_registry_image_producer_projection_abandon(r, old_writes[i], at(3), (RcRegistryImageBinding){100, owners[i]}, 0));
        OK(rc_registry_write_cancel(r, old_writes[i]));
    }
    RcRegistryNativeImageFragment output[4];
    uint32_t fragments;
    RcInterval demands[] = {interval(2, 100, 40, 0x4000, 90), interval(3, 200, 60, 0x4050, 91)};
    OK(rc_registry_native_image_fragments(r, demands, 2, at(6), output, 4, &fragments));
    assert(fragments == 2);
    expect_fragment(output[0], 0, 101, owners[0], old_parts[0], at(3));
    expect_fragment(output[1], 1, 101, owners[1], old_parts[1], at(3));

    RcInterval later = interval(2, 116, 8, 0x5000, 101);
    RcRegistryImageWriteExtent newer = hold(r, 200, later);
    RcRegistryWrite new_write = declare_image(r, later, at(15));
    OK(rc_registry_declare_through(r, at(16)));
    publish(r, 200, newer, &later.range, 1, at(15));
    record(r, new_write, 200, newer, at(15));
    OK(rc_registry_native_image_fragments(r, demands, 2, at(16), output, 4, &fragments));
    assert(fragments == 4);
    expect_fragment(output[0], 0, 101, owners[0], interval(2, 100, 16, 0x1000, 10), at(3));
    expect_fragment(output[1], 0, 200, newer.hold, later, at(15));
    expect_fragment(output[2], 0, 101, owners[0], interval(2, 124, 16, 0x1018, 10), at(3));
    expect_fragment(output[3], 1, 101, owners[1], old_parts[1], at(3));
    OK(rc_registry_write_abandon(r, new_write, 0));
    OK(rc_registry_native_image_fragments(r, demands, 2, at(16), output, 4, &fragments));
    assert(fragments == 2);
    rc_registry_image_hold_forget(r, owners[0]);
    assert(rc_registry_native_image_fragments(r, demands, 2, at(16), output, 4, &fragments) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_image_hold_release_extents(r, owners, 2, 0));
    assert(!r->native_projection_spans);
    rc_registry_destroy(r);
}


static void relocate_bound_recording_fate(void) {
    enum { PARTIAL, DROPPED, UNSENT, UNKNOWN, COMPLETED };
    for (int state = PARTIAL; state <= COMPLETED; ++state) {
        RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 16, 4});
        assert(r);
        RcInterval span = interval(2, 100, 64, 0x1000, 10);
        RcRegistryImageWriteExtent owner = hold(r, 100, span);
        RcRegistryWrite writer = declare_image(r, span, at(3));
        OK(rc_registry_declare_through(r, at(8)));
        assert(rc_registry_native_image_written(r, 100, &span, &owner.hold, 1, at(3)));
        if (state != PARTIAL) assert(rc_registry_native_image_whole_written(r, 100, &owner, 1, at(3)));
        record(r, writer, 100, owner, at(3));
        RcRegistryImageBinding source = {100, owner.hold};
        if (state == UNSENT || state == UNKNOWN)
            assert(rc_registry_image_producer_projection_abandon(r, writer, at(3), source, state == UNKNOWN));
        if (state == COMPLETED) {
            OK(rc_registry_handoff_through(r, at(3)));
            rc_registry_image_producer_native_completed(r, writer, at(3), source);
        }
        OK(rc_registry_write_cancel(r, writer));
        // The next recording can advance past a previously canceled ordinal. It does not prove the old image.
        const RcRegistryPosition successor = {2, 2};
        OK(rc_registry_declare_through(r, successor));
        OK(rc_registry_handoff_through(r, (RcRegistryPosition){2, 1}));
        RcRegistryResult result = rc_registry_native_image_relocate(r, 100, 101, &owner, 1, at(3), successor);
        assert((result == RC_REGISTRY_OK) == (state == COMPLETED));
        assert(rc_registry_image_binding_current(r, (RcRegistryImageBinding){state == COMPLETED ? 101 : 100, owner.hold}));
        assert(!rc_registry_native_image_covers(r, 100, &span, 1, successor));
        assert(rc_registry_native_image_covers(r, 101, &span, 1, successor) == (state == COMPLETED));
        rc_registry_destroy(r);
    }
}

static void relocate_padded_bound_image(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 16, 4});
    assert(r);
    const RcInterval extent = interval(2, 100, 64, 0x1000, 10);
    const RcInterval useful[] = {interval(2, 100, 12, 0x1000, 10), interval(2, 116, 20, 0x1010, 10)};
    const RcSourceRange spans[] = {useful[0].range, useful[1].range};
    const RcInterval unknown_padding = interval(2, 112, 4, 0x5000, 30);
    const RcRegistryImageWriteExtent owner = hold(r, 100, extent);
    RcRegistryWrite original = declare_image(r, extent, at(3));
    OK(rc_registry_declare_through(r, at(6)));
    assert(rc_registry_native_image_written(r, 100, &extent, &owner.hold, 1, at(3)));
    const RcRegistryNativeImageWrite exact = {owner, spans, 2};
    OK(rc_registry_native_image_bound_written(r, 100, &exact, 1, at(3)));
    record(r, original, 100, owner, at(3));
    RcRegistryNativeImageFragment fragments[5];
    uint32_t n = 99;
    assert(rc_registry_native_image_fragments(r, useful, 2, at(4), fragments, 5, &n) == RC_REGISTRY_BLOCKED && !n);
    assert(!rc_registry_native_image_has_prior_projection(r, useful, 2, at(4)));
    assert(rc_registry_native_image_relocate(r, 100, 101, &owner, 1, at(3), at(4)) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_handoff_through(r, at(3)));
    rc_registry_image_producer_native_completed(r, original, at(3), (RcRegistryImageBinding){100, owner.hold});
    OK(rc_registry_write_cancel(r, original));
    OK(rc_registry_native_image_relocate(r, 100, 101, &owner, 1, at(3), at(6)));
    assert(!rc_registry_image_binding_current(r, (RcRegistryImageBinding){100, owner.hold}));
    assert(rc_registry_image_binding_current(r, (RcRegistryImageBinding){101, owner.hold}));
    assert(rc_registry_native_image_covers(r, 101, useful, 2, at(6)));
    assert(!rc_registry_native_image_has_prior_projection(r, &unknown_padding, 1, at(6)));
    assert(rc_registry_native_image_fragments(r, &unknown_padding, 1, at(6), fragments, 5, &n) == RC_REGISTRY_BLOCKED && !n);
    assert(rc_registry_native_image_fragments(r, &extent, 1, at(6), fragments, 5, &n) == RC_REGISTRY_BLOCKED && !n);

    const RcInterval later = interval(2, 106, 10, 0x3000, 20);
    const RcRegistryImageWriteExtent new_owner = hold(r, 200, later);
    RcRegistryWrite new_write = declare_image(r, later, at(7));
    OK(rc_registry_declare_through(r, at(16)));
    publish(r, 200, new_owner, &later.range, 1, at(7));
    record(r, new_write, 200, new_owner, at(7));
    const RcInterval now_known = interval(2, 100, 36, 0x8000, 40);
    OK(rc_registry_native_image_fragments(r, &now_known, 1, at(8), fragments, 5, &n));
    assert(n == 3);
    expect_fragment(fragments[0], 0, 101, owner.hold, interval(2, 100, 6, 0x1000, 10), at(3));
    expect_fragment(fragments[1], 0, 200, new_owner.hold, later, at(7));
    expect_fragment(fragments[2], 0, 101, owner.hold, useful[1], at(3));
    assert(rc_registry_native_image_fragments(r, &extent, 1, at(8), fragments, 5, &n) == RC_REGISTRY_BLOCKED && !n);
    OK(rc_registry_write_cancel(r, new_write));
    assert(!rc_registry_native_image_covers(r, 101, &now_known, 1, at(10)));
    assert(rc_registry_native_image_covers(r, 101, useful, 2, at(10)));
    rc_registry_destroy(r);
}

static void later_projection_can_replace_a_handed_retired_provider(void) {
    for (int handed = 0; handed < 2; ++handed) {
        RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 16, 4});
        assert(r);
        const RcInterval extent = interval(6, 128, 64, 0x4000, 10);
        const RcRegistryImageWriteExtent older = hold(r, 100, extent);
        RcRegistryWrite old = declare_image(r, extent, at(3));
        OK(rc_registry_declare_through(r, at(4)));
        publish(r, 100, older, &extent.range, 1, at(3));
        record(r, old, 100, older, at(3));
        assert(rc_registry_native_image_covers(r, 100, &extent, 1, at(4)));
        if (handed) {
            OK(rc_registry_handoff_through(r, at(3)));
            OK(rc_registry_write_cancel(r, old));
        }

        // A true later whole image initialized from the prior contents has acquired its own storage.
        // Before reporting its recorded result the previous placement is no longer address-selectable.
        RcRegistryImageWriteExtent newer = hold(r, 101, extent);
        RcRegistryWrite successor = declare_image(r, extent, at(5));
        OK(rc_registry_declare_through(r, at(8)));
        OK(rc_registry_image_hold_release_extents(r, &older.hold, 1, 0));
        assert(!rc_registry_native_image_covers(r, 100, &extent, 1, at(4)));
        publish(r, 101, newer, &extent.range, 1, at(5));
        record(r, successor, 101, newer, at(5));
        assert(rc_registry_native_image_covers(r, 101, &extent, 1, at(6)) == handed);
        assert(rc_registry_stats(r).writer_foreign == 1);
        rc_registry_destroy(r);
    }
}

static void positioned_legacy_image_writes_do_not_hide_a_later_exact_provider(void) {
    for (int ordered = 0; ordered <= 1; ordered++) {
        RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 16, 4});
        assert(r);
        const RcInterval whole = interval(6, 0, 128, 0x2000, 17);
        const RcInterval older_part = interval(6, 0, 64, 0x2000, 17);
        const RcSourceRange meaningful = {6, 0, 120};
        const RcInterval covered = interval(6, 0, 120, 0x2000, 17);
        RcRegistryImageWriteExtent old = hold(r, 100, older_part), new_image = hold(r, 200, whole);
        const RcRegistryImageBinding old_binding = {100, old.hold}, binding = {200, new_image.hold};
        RcRegistryWrite first = declare_image(r, older_part, at(3)), exact = declare_image(r, whole, at(7));
        RcRegistryWrite later = declare_image(r, older_part, at(11));
        RcRegistryWrite uncertain = declare_image(r, older_part, at(15));
        OK(rc_registry_declare_through(r, at(18)));
        assert(!rc_registry_image_hold_recorded(r, first, binding, at(3), 1));
        assert(!rc_registry_image_hold_recorded(r, first, old_binding, at(5), 1));
        if (ordered) assert(rc_registry_image_hold_recorded(r, first, old_binding, at(3), 1));
        record(r, first, 100, old, at(3)); // actual writer, no claim about which pixels it touched
        if (ordered) assert(rc_registry_image_producer_projection_pending(r, first, at(3), old_binding));
        assert(rc_registry_image_hold_recorded(r, exact, binding, at(7), 1));
        publish(r, 200, new_image, &meaningful, 1, at(7));
        record(r, exact, 200, new_image, at(7));
        assert(rc_registry_native_image_covers(r, 200, &covered, 1, at(8)) == ordered);
        assert(!rc_registry_native_image_covers(r, 200, &whole, 1, at(8))); // physically placed padding carries no proof
        OK(rc_registry_handoff_through(r, at(7)));
        rc_registry_image_producer_native_completed(r, first, at(3), old_binding);
        rc_registry_image_producer_native_completed(r, exact, at(7), binding);
        OK(rc_registry_write_cancel(r, first));
        OK(rc_registry_write_cancel(r, exact));
        assert(rc_registry_native_image_covers(r, 200, &covered, 1, at(10)) == ordered);
        assert(!rc_registry_native_image_covers(r, 200, &covered, 1, at(12))); // unresolved possible later writer
        record(r, later, 100, old, at(11));
        assert(!rc_registry_native_image_covers(r, 200, &covered, 1, at(12))); // actual later legacy write
        if (ordered) {
            assert(rc_registry_image_producer_projection_abandon(r, later, at(11), old_binding, 0));
            OK(rc_registry_write_cancel(r, later));
            assert(rc_registry_native_image_covers(r, 200, &covered, 1, at(12)));
            // Retirement keeps the real handed order even if this older physical image never projected pixels.
            OK(rc_registry_image_hold_release_extents(r, &old.hold, 1, 0));
            assert(rc_registry_stats(r).writer_foreign == 1);
            assert(rc_registry_native_image_covers(r, 200, &covered, 1, at(12)));
            RcRegistryImageWriteExtent unknown = hold(r, 300, older_part);
            const RcRegistryImageBinding unknown_binding = {300, unknown.hold};
            assert(rc_registry_image_hold_recorded(r, uncertain, unknown_binding, at(15), 1));
            record(r, uncertain, 300, unknown, at(15));
            assert(!rc_registry_native_image_covers(r, 200, &covered, 1, at(16)));
            rc_registry_image_hold_unordered(r, 300, &unknown.hold, 1);
            assert(!rc_registry_native_image_covers(r, 200, &covered, 1, at(16)));
        }
        rc_registry_destroy(r);
    }
}

static void exact_native_fragment_refusal(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 16, 4});
    assert(r);
    const RcInterval whole = interval(6, 0, 128, 0x2000, 17);
    const RcInterval left = interval(6, 0, 96, 0x2000, 17);
    const RcInterval blocked = interval(6, 32, 32, 0x2020, 17);
    const RcInterval demands[] = {left, interval(6, 88, 24, 0x6010, 19)};
    RcRegistryImageWriteExtent owner = hold(r, 100, whole);
    RcRegistryWrite old = declare_image(r, whole, at(3));
    RcRegistryWrite later = declare_image(r, blocked, at(7));
    OK(rc_registry_declare_through(r, at(8)));
    RcRegistryNativeImageFragment fragments[4];
    RcRegistryNativeImageFailure failure;
    uint32_t count;
    publish(r, 100, owner, &left.range, 1, at(3));
    assert(rc_registry_native_image_fragments_checked(r, &left, 1, at(6), fragments, 4, &count, &failure) ==
           RC_REGISTRY_BLOCKED && !count && failure.gate == RC_NATIVE_IMAGE_READ_TENTATIVE);
    assert(failure.selected.image == 100 && failure.selected.hold.id == owner.hold.id && failure.selected_output.ordinal == 3);
    record(r, old, 100, owner, at(3));
    uint64_t authority = rc_registry_authority_epoch(r);
    assert(rc_registry_native_image_fragments_checked(r, demands, 2, at(6), fragments, 4, &count, &failure) ==
           RC_REGISTRY_BLOCKED && !count && failure.gate == RC_NATIVE_IMAGE_READ_NO_PRODUCER);
    assert(failure.demand == 1 && failure.interval.address == 0x6018 && failure.interval.mapping == 19 &&
           failure.interval.range.offset == 96 && failure.interval.range.size == 16 && !failure.selected.image &&
           !failure.blocker.id && rc_registry_authority_epoch(r) == authority);

    assert(rc_registry_native_image_fragments_checked(r, &left, 1, at(8), fragments, 4, &count, &failure) ==
           RC_REGISTRY_BLOCKED && !count && failure.gate == RC_NATIVE_IMAGE_READ_BLOCKED);
    assert(failure.cause == RC_NATIVE_IMAGE_BLOCK_PENDING_OUTPUT && failure.blocker.id == later.id &&
           failure.blocking_output.ordinal == 7 && failure.blocking_interval.range.offset == 32 &&
           failure.blocking_interval.range.size == 32 && failure.selected.hold.id == owner.hold.id &&
           failure.selected_output.ordinal == 3);
    OK(rc_registry_write_cancel(r, later));
    OK(rc_registry_native_image_fragments_checked(r, &left, 1, at(8), fragments, 4, &count, &failure));
    assert(count == 1 && failure.gate == RC_NATIVE_IMAGE_READ_OK && !failure.blocker.id);
    rc_registry_destroy(r);
}

static void ordered_bound_snapshot_keeps_registered_footprint(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 16, 4});
    assert(r);
    const RcInterval range = interval(9, 128, 64, 0x4000, 17);
    const RcInterval covered = interval(9, 128, 48, 0x5010, 18);
    const RcInterval padding = interval(9, 176, 16, 0x5040, 18);
    RcRegistryImageWriteExtent exact = hold(r, 100, range);
    RcRegistryWrite written = declare_image(r, range, at(3));
    RcRegistryWrite pending = declare_image(r, covered, at(7));
    OK(rc_registry_declare_through(r, at(12)));
    assert(rc_registry_native_image_written(r, 100, &range, &exact.hold, 1, at(3)));
    const RcRegistryNativeImageWrite write = {exact, &covered.range, 1};
    OK(rc_registry_native_image_bound_written(r, 100, &write, 1, at(3)));
    RcRegistryNativeImageFragment fragment;
    RcRegistryNativeImageFailure failure;
    uint32_t count;
    assert(rc_registry_native_image_snapshot_fragments(r, &covered, 1, at(4), &fragment, 1, &count, &failure) ==
           RC_REGISTRY_BLOCKED && failure.gate == RC_NATIVE_IMAGE_READ_TENTATIVE);
    record(r, written, 100, exact, at(3));
    OK(rc_registry_native_image_snapshot_fragments(r, &covered, 1, at(6), &fragment, 1, &count, &failure));
    assert(count == 1 && fragment.binding.image == 100 && fragment.binding.hold.id == exact.hold.id &&
           fragment.interval.address == 0x4000 && fragment.interval.mapping == 17 && !failure.gate);
    assert(rc_registry_native_image_fragments_checked(r, &covered, 1, at(6), &fragment, 1, &count, &failure) ==
           RC_REGISTRY_BLOCKED && failure.gate == RC_NATIVE_IMAGE_READ_BOUND);
    assert(rc_registry_native_image_snapshot_fragments(r, &padding, 1, at(6), &fragment, 1, &count, &failure) ==
           RC_REGISTRY_BLOCKED && failure.gate == RC_NATIVE_IMAGE_READ_NO_PRODUCER && !count);
    assert(rc_registry_native_image_snapshot_fragments(r, &covered, 1, at(8), &fragment, 1, &count, &failure) ==
           RC_REGISTRY_BLOCKED && failure.blocker.id == pending.id);
    OK(rc_registry_write_cancel(r, pending));
    OK(rc_registry_native_image_snapshot_fragments(r, &covered, 1, at(8), &fragment, 1, &count, &failure));
    assert(rc_registry_image_producer_projection_abandon(r, written, at(3), (RcRegistryImageBinding){100, exact.hold}, 0));
    assert(rc_registry_native_image_snapshot_fragments(r, &covered, 1, at(8), &fragment, 1, &count, &failure) ==
           RC_REGISTRY_BLOCKED && !count);
    rc_registry_destroy(r);
}

static void padded_bound_writer_leaves_older_independent_bytes_current(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 16, 4});
    assert(r);
    const RcInterval older = interval(5, 0, 128, 0x1000, 10);
    const RcInterval newer = interval(5, 0, 64, 0x3000, 20);
    const RcSourceRange meaningful = {5, 0, 48};
    const RcRegistryImageWriteExtent a = hold(r, 100, older), b = hold(r, 200, newer);
    RcRegistryWrite a_claim = declare_image(r, older, at(3)), b_claim = declare_image(r, newer, at(5));
    RcRegistryWrite later = declare_image(r, newer, at(7));
    OK(rc_registry_declare_through(r, at(12)));
    publish(r, 100, a, &older.range, 1, at(3));
    record(r, a_claim, 100, a, at(3));
    assert(rc_registry_native_image_written(r, 200, &newer, &b.hold, 1, at(5)));
    const RcRegistryNativeImageWrite write = {b, &meaningful, 1};
    assert(!rc_registry_native_image_has_prior_bound_projection(r, &older, 1, at(6)));
    OK(rc_registry_native_image_bound_written(r, 200, &write, 1, at(5)));
    record(r, b_claim, 200, b, at(5));
    const RcInterval padding = interval(5,48,16,0x3030,20);
    assert(rc_registry_native_image_has_prior_bound_projection(r, &older, 1, at(6)));
    assert(!rc_registry_native_image_has_prior_bound_projection(r, &older, 1, at(4)));
    assert(!rc_registry_native_image_has_prior_bound_projection(r, &padding, 1, at(6)));
    assert(rc_registry_native_image_has_prior_projection(r, &padding, 1, at(6)));
    RcRegistryNativeImageFragment fragments[3];
    RcRegistryNativeImageFailure failure;
    uint32_t count;
    OK(rc_registry_native_image_snapshot_fragments(r, &older, 1, at(6), fragments, 3, &count, &failure));
    assert(count == 2 && !failure.gate);
    expect_fragment(fragments[0], 0, 200, b.hold, interval(5,0,48,0x3000,20), at(5));
    expect_fragment(fragments[1], 0, 100, a.hold, interval(5,48,80,0x1030,10), at(3));
    assert(rc_registry_native_image_written(r, 200, &newer, &b.hold, 1, at(7)));
    record(r, later, 200, b, at(7));
    assert(rc_registry_native_image_snapshot_fragments(r, &older, 1, at(8), fragments, 3, &count, &failure) ==
           RC_REGISTRY_BLOCKED && !count);
    rc_registry_destroy(r);
}

static void clear_and_copy_keep_their_exact_same_image_planes(void) {
    for (int clear_proved = 0; clear_proved <= 1; ++clear_proved) {
        RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 16, 4});
        assert(r);
        const RcInterval place = interval(9, 0, 512, 0x1000, 17);
        const RcInterval read = interval(9, 64, 96, 0x5000, 18);
        const RcInterval hole = interval(9, 160, 32, 0x5060, 18);
        const RcInterval defined_prior = interval(9, 192, 32, 0x5080, 18);
        const RcRegistryImageWriteExtent older = hold(r, 100, defined_prior), selected = hold(r, 200, place);
        const RcRegistryWrite older_effect = declare_image(r, defined_prior, at(3)),
            clear = declare_image(r, place, at(5)), copy = declare_image(r, place, at(7));
        const RcSourceRange clear_plane = {9, 0, 160}, copied = {9, 64, 32};
        OK(rc_registry_declare_through(r, at(16)));
        publish(r, 100, older, &defined_prior.range, 1, at(3));
        record(r, older_effect, 100, older, at(3));
        const RcRegistryImageBinding binding = {200, selected.hold};
        assert(rc_registry_image_hold_recorded(r, clear, binding, at(5), 1));
        if (clear_proved) publish(r, 200, selected, &clear_plane, 1, at(5));
        record(r, clear, 200, selected, at(5));
        publish(r, 200, selected, &copied, 1, at(7));
        record(r, copy, 200, selected, at(7));
        OK(rc_registry_handoff_through(r, at(8)));
        rc_registry_image_producer_native_completed(r, clear, at(5), binding);
        OK(rc_registry_write_cancel(r, older_effect));
        OK(rc_registry_write_cancel(r, clear));
        OK(rc_registry_write_cancel(r, copy));
        RcRegistryNativeImageFailure failure;
        RcRegistryNativeImageFragment fragments[4];
        uint32_t count;
        RcRegistryResult got = rc_registry_native_image_snapshot_fragments(r, &read, 1, at(10), fragments, 4, &count, &failure);
        if (clear_proved) {
            assert(got == RC_REGISTRY_OK && count == 2);
            expect_fragment(fragments[0], 0, 200, selected.hold, interval(9, 64, 32, 0x1040, 17), at(7));
            expect_fragment(fragments[1], 0, 200, selected.hold, interval(9, 96, 64, 0x1060, 17), at(5));
            OK(rc_registry_native_image_snapshot_fragments(r, &defined_prior, 1, at(10), fragments, 4, &count, &failure));
            expect_fragment(fragments[0], 0, 100, older.hold, defined_prior, at(3));
        } else {
            assert(got == RC_REGISTRY_BLOCKED && !count && failure.gate == RC_NATIVE_IMAGE_READ_NO_PRODUCER &&
                failure.cause == RC_NATIVE_IMAGE_BLOCK_UNPROJECTED && failure.blocking_output.ordinal == 5);
        }
        assert(rc_registry_native_image_snapshot_fragments(r, &hole, 1, at(10), fragments, 4, &count, &failure) ==
            RC_REGISTRY_BLOCKED && !count && failure.gate == RC_NATIVE_IMAGE_READ_NO_PRODUCER);
        rc_registry_destroy(r);
    }
}

static void mixed_snapshot_uses_only_exact_current_providers(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 24, 4});
    assert(r);
    const RcInterval place = interval(9, 0, 64, 0x1000, 17);
    const RcInterval demand = interval(9, 0, 64, 0x5000, 18);
    const RcRegistryBufferSpan baseline = {demand, {20, 2, 512}};
    const RcSourceRange drawn = {9, 16, 16};
    const RcRegistryImageWriteExtent image = hold(r, 100, place);
    RcRegistryWrite draw = declare_image(r, place, at(3));
    RcRegistryBufferSpan buffer = {interval(9, 24, 24, 0x5018, 18), {20, 2, 536}};
    RcRegistryWrite typed, opaque;
    OK(rc_registry_write_begin_at(r, buffer.interval.range, RC_REGISTRY_BUFFER, at(5), &typed));
    OK(rc_registry_write_begin_at(r, buffer.interval.range, RC_REGISTRY_BUFFER, at(7), &opaque));
    OK(rc_registry_declare_through(r, at(18)));
    publish(r, 100, image, &drawn, 1, at(3));
    record(r, draw, 100, image, at(3));
    OK(rc_registry_buffer_write_record(r, typed, buffer, at(5)));
    // This read-modify-write receipt may touch bytes, but cannot substitute an exact written plane.
    OK(rc_registry_buffer_access_record(r, opaque, buffer, at(7)));
    RcRegistryNativeImageFailure failure;
    RcRegistryNativeImageSnapshotFragment fragments[5];
    uint32_t count = 0;
    assert(rc_registry_native_image_snapshot_mixed(r, &demand, 1, NULL, 0, at(4), fragments, 5, &count, &failure) ==
           RC_REGISTRY_BLOCKED && !count && failure.gate == RC_NATIVE_IMAGE_READ_NO_PRODUCER);
    OK(rc_registry_native_image_snapshot_mixed(r, &demand, 1, &baseline, 1, at(4), fragments, 5, &count, &failure));
    assert(count == 3 && !failure.gate);
    assert(fragments[0].baseline == 0 && !fragments[0].fragment.binding.image &&
           fragments[0].native_buffer.allocation == 20 && fragments[0].native_buffer.offset == 512 &&
           fragments[0].fragment.interval.mapping == 18 && fragments[0].fragment.interval.range.size == 16 &&
           !rc_registry_position_valid(fragments[0].fragment.output));
    assert(fragments[1].baseline == UINT32_MAX && !fragments[1].native_buffer.allocation);
    expect_fragment(fragments[1].fragment, 0, 100, image.hold, interval(9,16,16,0x1010,17), at(3));
    assert(fragments[2].fragment.interval.address == 0x5020 && fragments[2].native_buffer.offset == 544);

    OK(rc_registry_native_image_snapshot_mixed(r, &demand, 1, &baseline, 1, at(6), fragments, 5, &count, &failure));
    assert(count == 4 && fragments[1].fragment.interval.range.size == 8);
    assert(fragments[2].baseline == 0 && fragments[2].native_buffer.offset == 536 &&
           fragments[2].fragment.output.ordinal == 5 && fragments[2].fragment.interval.range.size == 24);
    assert(rc_registry_native_image_snapshot_mixed(r, &demand, 1, &baseline, 1, at(6), fragments, 3, &count, &failure) ==
           RC_REGISTRY_CAPACITY && !count);
    assert(rc_registry_native_image_snapshot_mixed(r, &demand, 1, &baseline, 1, at(8), fragments, 5, &count, &failure) ==
           RC_REGISTRY_BLOCKED && !count && failure.blocker.id == opaque.id);

    RcRegistryBufferSpan replaced = baseline;
    replaced.interval.mapping++;
    assert(rc_registry_native_image_snapshot_mixed(r, &demand, 1, &replaced, 1, at(6), fragments, 5, &count, &failure) ==
           RC_REGISTRY_BLOCKED && !count);
    replaced = baseline;
    replaced.native.incarnation++;
    assert(rc_registry_native_image_snapshot_mixed(r, &demand, 1, &replaced, 1, at(6), fragments, 5, &count, &failure) ==
           RC_REGISTRY_BLOCKED && !count && failure.blocker.id == typed.id);

    OK(rc_registry_write_cancel(r, opaque));
    OK(rc_registry_write_cancel(r, typed));
    OK(rc_registry_native_image_snapshot_mixed(r, &demand, 1, &baseline, 1, at(8), fragments, 5, &count, &failure));
    assert(count == 3);
    OK(rc_registry_handoff_through(r, at(8)));
    OK(rc_registry_write_cancel(r, draw));
    OK(rc_registry_image_hold_release_extents(r, &image.hold, 1, 0));
    assert(rc_registry_native_image_snapshot_mixed(r, &demand, 1, &baseline, 1, at(10), fragments, 5, &count, &failure) ==
           RC_REGISTRY_BLOCKED && !count && failure.cause == RC_NATIVE_IMAGE_BLOCK_FOREIGN);
    rc_registry_destroy(r);
}


enum {
    MIXED_DEMANDS = 197, MIXED_FRAGMENTS = 3498, MIXED_RAW = 954,
    MIXED_IMAGES = 9, MIXED_NINTH_DEMAND = 32, MIXED_BYTES = 0x80,
    MIXED_STRIDE = 19 * MIXED_BYTES, MIXED_BACKING = 1346, MIXED_MAPPING = 1349,
    MIXED_NINTH_OFFSET = 0x18ec080,
    MIXED_START = MIXED_NINTH_OFFSET - MIXED_NINTH_DEMAND * MIXED_STRIDE - MIXED_BYTES,
};

static uint32_t mixed_slot_count(uint32_t demand) {
    return 17 + (demand < MIXED_FRAGMENTS - 17 * MIXED_DEMANDS);
}

static int mixed_slot_source(uint32_t demand, uint32_t slot) {
    const uint32_t raw = 4 + (demand < MIXED_RAW - 4 * MIXED_DEMANDS);
    if (slot % 2 == 0 && slot / 2 < raw) return -1;
    if (demand == MIXED_NINTH_DEMAND && slot == 1) return MIXED_IMAGES - 1;
    const uint32_t raw_before = slot / 2 + 1 < raw ? slot / 2 + 1 : raw;
    const uint32_t available = demand < MIXED_NINTH_DEMAND ? MIXED_IMAGES - 1 : MIXED_IMAGES;
    return (demand + slot - raw_before) % available;
}

static RcRegistryPosition mixed_output(uint32_t source) {
    return (RcRegistryPosition){13750, 36103 + 2 * source};
}

static void expect_large_mixed_partition(const RcRegistryNativeImageSnapshotFragment *fragments,
    const RcRegistryImageWriteExtent *owners, const RcInterval *demands, const RcRegistryBufferSpan *baseline) {
    uint32_t at = 0, raw_count = 0, seen = 0;
    for (uint32_t i = 0; i < MIXED_DEMANDS; ++i) {
        for (uint32_t j = 0; j < mixed_slot_count(i); ++j) {
            const RcRegistryNativeImageSnapshotFragment f = fragments[at++];
            const int source = mixed_slot_source(i, j);
            const uint64_t within = (uint64_t)j * MIXED_BYTES;
            const RcInterval part = interval(MIXED_BACKING, demands[i].range.offset + within, MIXED_BYTES,
                                             demands[i].address + within, MIXED_MAPPING);
            if (source < 0) {
                ++raw_count;
                expect_fragment(f.fragment, i, 0, (RcRegistryWrite){0}, part, (RcRegistryPosition){0});
                assert(f.baseline == 0 && f.native_buffer.allocation == baseline->native.allocation &&
                       f.native_buffer.incarnation == baseline->native.incarnation &&
                       f.native_buffer.offset == baseline->native.offset + part.range.offset - MIXED_START);
            } else {
                if (!(seen & (1u << source)) && source == MIXED_IMAGES - 1)
                    assert(seen == (1u << (MIXED_IMAGES - 1)) - 1 && i == MIXED_NINTH_DEMAND &&
                           part.range.offset == MIXED_NINTH_OFFSET);
                seen |= 1u << source;
                expect_fragment(f.fragment, i, 100 + source, owners[source].hold, part, mixed_output(source));
                assert(f.baseline == UINT32_MAX && !f.native_buffer.allocation);
            }
        }
    }
    assert(at == MIXED_FRAGMENTS && raw_count == MIXED_RAW && seen == (1u << MIXED_IMAGES) - 1);
}

static void mixed_snapshot_keeps_many_image_versions_of_one_receiver(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){512, 32, 8});
    assert(r);
    const uint64_t address = UINT64_C(0x800000000) + MIXED_START;
    const RcInterval receiver = interval(MIXED_BACKING, MIXED_START, MIXED_DEMANDS * MIXED_STRIDE,
                                         address, MIXED_MAPPING);
    const RcRegistryBufferSpan baseline = {receiver, {20, 2, 4096}};
    const RcRegistryPosition read = {13750, 45294}, later = {13750, 45296}, overwrite_at = {13750, 45295};
    RcInterval demands[MIXED_DEMANDS];
    RcRegistryImageWriteExtent owners[MIXED_IMAGES];
    RcRegistryWrite writers[MIXED_IMAGES];
    uint32_t spans_count[MIXED_IMAGES] = {0};
    RcSourceRange (*spans)[MIXED_FRAGMENTS] = malloc(sizeof *spans * MIXED_IMAGES);
    RcRegistryNativeImageSnapshotFragment *fragments = malloc(sizeof *fragments * MIXED_FRAGMENTS);
    assert(spans && fragments);

    // Drive the true registry at the captured receiver's demand/fragment cardinalities. Geometry and
    // earlier source identities are synthetic. Each demand has meaningful raw slices and distinct
    // adjoining image winners. The gap between demands represents bytes outside the receiver view.
    for (uint32_t i = 0; i < MIXED_DEMANDS; ++i) {
        const uint64_t base = (uint64_t)i * MIXED_STRIDE;
        demands[i] = interval(MIXED_BACKING, MIXED_START + base, mixed_slot_count(i) * MIXED_BYTES,
                              address + base, MIXED_MAPPING);
        for (uint32_t j = 0; j < mixed_slot_count(i); ++j) {
            const int source = mixed_slot_source(i, j);
            if (source < 0) continue;
            const uint64_t offset = demands[i].range.offset + (uint64_t)j * MIXED_BYTES;
            spans[source][spans_count[source]++] = (RcSourceRange){MIXED_BACKING, offset, MIXED_BYTES};
            if (i == 0 && j == 3) {
                assert(source == 1);
                // The older interior span divides the sweep but must not divide the newer winner.
                spans[0][spans_count[0]++] = (RcSourceRange){MIXED_BACKING, offset + 32, 32};
            }
        }
    }
    for (uint32_t i = 0; i < MIXED_IMAGES; ++i) {
        owners[i] = hold(r, 100 + i, receiver);
        writers[i] = declare_image(r, receiver, mixed_output(i));
    }
    OK(rc_registry_declare_through(r, read));
    for (uint32_t i = 0; i < MIXED_IMAGES; ++i) {
        publish(r, 100 + i, owners[i], spans[i], spans_count[i], mixed_output(i));
        record(r, writers[i], 100 + i, owners[i], mixed_output(i));
    }
    free(spans);

    RcRegistryNativeImageFailure failure;
    uint32_t count;
    assert(rc_registry_native_image_snapshot_mixed(r, demands, MIXED_DEMANDS, NULL, 0, read,
        fragments, MIXED_FRAGMENTS, &count, &failure) == RC_REGISTRY_BLOCKED && !count &&
        failure.gate == RC_NATIVE_IMAGE_READ_NO_PRODUCER && failure.demand == 0);
    assert(rc_registry_native_image_snapshot_mixed(r, demands, MIXED_DEMANDS, &baseline, 1, read,
        fragments, MIXED_FRAGMENTS - 1, &count, &failure) == RC_REGISTRY_CAPACITY && !count &&
        failure.gate == RC_NATIVE_IMAGE_READ_CAPACITY && failure.demand == MIXED_DEMANDS - 1);
    OK(rc_registry_native_image_snapshot_mixed(r, demands, MIXED_DEMANDS, &baseline, 1, read,
        fragments, MIXED_FRAGMENTS, &count, &failure));
    assert(count == MIXED_FRAGMENTS && !failure.gate);
    expect_large_mixed_partition(fragments, owners, demands, &baseline);

    // A later image replaces only the middle of the ninth donor's original slice.
    const uint64_t patch_offset = MIXED_NINTH_OFFSET + 32;
    const RcInterval patch = interval(MIXED_BACKING, patch_offset, 64, 0x9000, MIXED_MAPPING + 1);
    const RcRegistryImageWriteExtent successor = hold(r, 200, patch);
    const RcRegistryWrite replacement = declare_image(r, patch, overwrite_at);
    OK(rc_registry_declare_through(r, later));
    publish(r, 200, successor, &patch.range, 1, overwrite_at);
    record(r, replacement, 200, successor, overwrite_at);
    const RcInterval *ninth = &demands[MIXED_NINTH_DEMAND];
    OK(rc_registry_native_image_snapshot_mixed(r, ninth, 1, &baseline, 1, later,
        fragments, MIXED_FRAGMENTS, &count, &failure));
    assert(count == mixed_slot_count(MIXED_NINTH_DEMAND) + 2);
    assert(fragments[0].baseline == 0 && fragments[4].baseline == 0);
    const uint64_t ninth_address = ninth->address + MIXED_BYTES;
    expect_fragment(fragments[1].fragment, 0, 108, owners[8].hold,
                    interval(MIXED_BACKING, MIXED_NINTH_OFFSET, 32, ninth_address, MIXED_MAPPING), mixed_output(8));
    expect_fragment(fragments[2].fragment, 0, 200, successor.hold, patch, overwrite_at);
    expect_fragment(fragments[3].fragment, 0, 108, owners[8].hold,
                    interval(MIXED_BACKING, MIXED_NINTH_OFFSET + 96, 32, ninth_address + 96, MIXED_MAPPING),
                    mixed_output(8));

    OK(rc_registry_native_image_snapshot_mixed(r, demands, MIXED_DEMANDS, &baseline, 1, read,
        fragments, MIXED_FRAGMENTS, &count, &failure));
    assert(count == MIXED_FRAGMENTS && !failure.gate);
    expect_large_mixed_partition(fragments, owners, demands, &baseline);
    free(fragments);
    rc_registry_destroy(r);
}

static void mixed_snapshot_does_not_use_raw_padding_under_bound_image(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 16, 4});
    assert(r);
    const RcInterval place = interval(9, 0, 64, 0x1000, 17);
    const RcInterval demand = interval(9, 32, 16, 0x2040, 18);
    const RcSourceRange meaningful = {9, 0, 32};
    // A fixed physical pool publishes its allocation once and has no mirror generation.
    const RcRegistryBufferSpan baseline = {demand, {20, 0, 512}};
    const RcRegistryImageWriteExtent image = hold(r, 100, place);
    RcRegistryWrite clear = declare_image(r, place, at(3)), typed;
    OK(rc_registry_write_begin_at(r, demand.range, RC_REGISTRY_BUFFER, at(5), &typed));
    OK(rc_registry_declare_through(r, at(12)));
    const RcRegistryNativeImageWrite write = {image, &meaningful, 1};
    assert(rc_registry_native_image_written(r, 100, &place, &image.hold, 1, at(3)));
    OK(rc_registry_native_image_bound_written(r, 100, &write, 1, at(3)));
    record(r, clear, 100, image, at(3));
    OK(rc_registry_buffer_write_record(r, typed, baseline, at(5)));
    RcRegistryNativeImageFailure failure;
    RcRegistryNativeImageSnapshotFragment fragment;
    uint32_t count = 0;
    assert(rc_registry_native_image_snapshot_mixed(r, &demand, 1, &baseline, 1, at(4), &fragment, 1, &count, &failure) ==
        RC_REGISTRY_BLOCKED && !count && failure.blocker.id == image.hold.id);
    OK(rc_registry_native_image_snapshot_mixed(r, &demand, 1, &baseline, 1, at(6), &fragment, 1, &count, &failure));
    assert(count == 1 && !fragment.fragment.binding.image && fragment.baseline == 0 &&
           fragment.fragment.output.ordinal == 5 && fragment.native_buffer.allocation == 20 &&
           fragment.native_buffer.incarnation == 0 && fragment.native_buffer.offset == 512);
    OK(rc_registry_handoff_through(r, at(7)));
    OK(rc_registry_write_submit(r, typed, 1));
    OK(rc_registry_write_complete(r, typed, 0));
    OK(rc_registry_write_cancel(r, clear));
    OK(rc_registry_native_image_snapshot_mixed(r, &demand, 1, &baseline, 1, at(8), &fragment, 1, &count, &failure));
    assert(count == 1 && fragment.fragment.output.ordinal == 5 && fragment.native_buffer.offset == 512);
    // New native placement or mapping cannot see the old fixed-pool writer.
    RcRegistryBufferSpan replaced = baseline;
    replaced.native.allocation++;
    assert(rc_registry_native_image_snapshot_mixed(r, &demand, 1, &replaced, 1, at(8), &fragment, 1, &count, &failure) ==
        RC_REGISTRY_BLOCKED && !count && failure.blocker.id == typed.id);
    replaced = baseline;
    replaced.native.incarnation++;
    assert(rc_registry_native_image_snapshot_mixed(r, &demand, 1, &replaced, 1, at(8), &fragment, 1, &count, &failure) ==
        RC_REGISTRY_BLOCKED && !count && failure.blocker.id == typed.id);
    replaced = baseline;
    replaced.interval.mapping++;
    assert(rc_registry_native_image_snapshot_mixed(r, &demand, 1, &replaced, 1, at(8), &fragment, 1, &count, &failure) ==
        RC_REGISTRY_BLOCKED && !count);
    replaced = baseline;
    replaced.native.allocation = 0;
    assert(rc_registry_native_image_snapshot_mixed(r, &demand, 1, &replaced, 1, at(8), &fragment, 1, &count, &failure) ==
        RC_REGISTRY_INVALID && !count);
    rc_registry_destroy(r);
}


static void mixed_snapshot_partitions_affine_baselines_per_demand(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 16, 4});
    assert(r);
    const RcInterval prior = interval(9, 0, 64, 0x1000, 17);
    const RcInterval demands[] = {interval(9, 48, 8, 0x5030, 18), interval(9, 0, 64, 0x5000, 18)};
    const RcSourceRange image_bytes = {9, 16, 16};
    const RcRegistryImageWriteExtent image = hold(r, 100, prior);
    RcRegistryWrite draw = declare_image(r, prior, at(3));
    OK(rc_registry_declare_through(r, at(12)));
    publish(r, 100, image, &image_bytes, 1, at(3));
    record(r, draw, 100, image, at(3));

    RcRegistryBufferSpan baselines[] = {
        {interval(9, 32, 32, 0x5020, 18), {20, 2, 544}},
        {interval(9, 0, 16, 0x5000, 18), {20, 2, 512}},
        {interval(9, 12, 12, 0x500c, 18), {20, 2, 524}},
    };
    RcRegistryNativeImageFailure failure;
    RcRegistryNativeImageSnapshotFragment fragments[5];
    uint32_t count;
    OK(rc_registry_native_image_snapshot_mixed(r, demands, 2, baselines, 2, at(4), fragments, 5, &count, &failure));
    assert(count == 4 && fragments[0].fragment.demand == 0 && fragments[0].native_buffer.offset == 560);
    assert(fragments[1].fragment.demand == 1 && fragments[1].baseline == 1 &&
        fragments[1].native_buffer.offset == 512 && fragments[1].fragment.interval.range.size == 16);
    expect_fragment(fragments[2].fragment, 1, 100, image.hold, interval(9, 16, 16, 0x1010, 17), at(3));
    assert(fragments[3].fragment.demand == 1 && fragments[3].baseline == 0 &&
        fragments[3].native_buffer.offset == 544 && fragments[3].fragment.interval.range.size == 32);

    // Equivalent overlapping certificates are safe and a real image wins over the older baseline.
    OK(rc_registry_native_image_snapshot_mixed(r, demands, 2, baselines, 3, at(4), fragments, 5, &count, &failure));
    assert(count == 4 && fragments[1].fragment.interval.range.size == 16);
    baselines[2].native.offset += 4;
    assert(rc_registry_native_image_snapshot_mixed(r, demands, 2, baselines, 3, at(4), fragments, 5, &count, &failure) ==
        RC_REGISTRY_BLOCKED && !count && failure.demand == 1 &&
        failure.interval.range.offset == 12 && failure.interval.range.size == 4);

    // A base shifted in VA is a different affine mapping, even over the same backing/mapping ids.
    baselines[1].interval.address += 4;
    assert(rc_registry_native_image_snapshot_mixed(r, demands, 2, baselines, 2, at(4), fragments, 5, &count, &failure) ==
        RC_REGISTRY_BLOCKED && !count && failure.demand == 1 && failure.interval.range.offset == 0);
    rc_registry_destroy(r);
}

static void raw_buffer_observes_independent_image_authority(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){12, 16, 4});
    assert(r);
    const RcInterval place = interval(9, 0, 64, 0x1000, 17);
    const RcInterval raw = interval(9, 0, 16, 0x5000, 20);
    const RcInterval texture = interval(9, 16, 16, 0x5010, 20);
    const RcRegistryImageWriteExtent image = hold(r, 100, place);
    const RcRegistryWrite write = declare_image(r, place, at(3));
    OK(rc_registry_declare_through(r, at(8)));
    assert(!rc_registry_native_image_buffer_requires_coverage(r, &texture, 1, at(2)));
    publish(r, 100, image, &texture.range, 1, at(3));
    record(r, write, 100, image, at(3));
    assert(!rc_registry_native_image_buffer_requires_coverage(r, &raw, 1, at(4)));
    assert(rc_registry_native_image_buffer_requires_coverage(r, &texture, 1, at(4)));
    OK(rc_registry_handoff_through(r, at(3)));
    OK(rc_registry_write_submit(r, write, 1));
    rc_registry_image_producer_native_completed(r, write, at(3), (RcRegistryImageBinding){100,image.hold});
    OK(rc_registry_write_complete_effects(r, write, NULL, 0, NULL, 0));
    OK(rc_registry_image_hold_release_extents(r, &image.hold, 1, 0));
    assert(!rc_registry_native_image_has_prior_projection(r, &texture, 1, at(6)));
    assert(rc_registry_native_image_buffer_requires_coverage(r, &texture, 1, at(6)));
    const RcRegistryBufferSpan old = {texture,{33,0,16}};
    RcRegistryNativeImageSnapshotFragment fragment;
    uint32_t count;
    assert(rc_registry_native_image_snapshot_mixed(r, &texture, 1, &old, 1, at(6), &fragment, 1, &count, NULL) ==
        RC_REGISTRY_BLOCKED && !count);
    rc_registry_destroy(r);
}

static void image_coverage_at_inclusive_handed_frontier(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){12, 16, 4});
    assert(r);
    const RcInterval place = interval(9, 0, 64, 0x1000, 17);
    const RcInterval second = interval(9, 0, 64, 0x3000, 19);
    const RcInterval center = interval(9, 16, 16, 0x5010, 20);
    const RcInterval whole = interval(9, 0, 64, 0x5000, 20);
    const RcRegistryImageWriteExtent old = hold(r, 100, place), newer = hold(r, 200, second);
    const RcRegistryWrite first = declare_image(r, place, at(3)), update = declare_image(r, second, at(9));
    OK(rc_registry_declare_through(r, at(12)));
    publish(r, 100, old, &place.range, 1, at(3));
    publish(r, 200, newer, &center.range, 1, at(9));
    record(r, first, 100, old, at(3));
    record(r, update, 200, newer, at(9));
    assert(!rc_registry_native_image_covers_handed(r, 100, &whole, 1));
    OK(rc_registry_handoff_through(r, at(3)));
    assert(rc_registry_native_image_covers_handed(r, 100, &whole, 1));
    assert(!rc_registry_native_image_covers_handed(r, 200, &center, 1));
    OK(rc_registry_write_submit(r, first, 1));
    assert(rc_registry_native_image_covers_handed(r, 100, &whole, 1));
    rc_registry_image_producer_native_completed(r, first, at(3),
        (RcRegistryImageBinding){100, old.hold});
    OK(rc_registry_write_complete_effects(r, first, NULL, 0, NULL, 0));
    assert(rc_registry_native_image_covers_handed(r, 100, &whole, 1));
    OK(rc_registry_handoff_through(r, at(9)));
    assert(rc_registry_native_image_covers_handed(r, 200, &center, 1));
    assert(!rc_registry_native_image_covers_handed(r, 100, &whole, 1));
    assert(!rc_registry_native_image_covers_handed(r, 200, &whole, 1));
    OK(rc_registry_write_submit(r, update, 1));
    rc_registry_image_producer_native_completed(r, update, at(9),
        (RcRegistryImageBinding){200, newer.hold});
    OK(rc_registry_write_complete_effects(r, update, NULL, 0, NULL, 0));
    assert(rc_registry_native_image_covers_handed(r, 200, &center, 1));
    rc_registry_destroy(r);

    r = rc_registry_create((RcRegistryConfig){12, 16, 4});
    assert(r);
    const RcRegistryImageWriteExtent physical = hold(r, 300, place);
    const RcRegistryWrite write = declare_image(r, place, at(3));
    const RcRegistryNativeImageWrite exact = {physical, &center.range, 1};
    OK(rc_registry_declare_through(r, at(6)));
    assert(rc_registry_native_image_written(r, 300, &place, &physical.hold, 1, at(3)));
    OK(rc_registry_native_image_bound_written(r, 300, &exact, 1, at(3)));
    record(r, write, 300, physical, at(3));
    OK(rc_registry_handoff_through(r, at(3)));
    assert(rc_registry_native_image_covers_handed(r, 300, &center, 1));
    assert(!rc_registry_native_image_covers_handed(r, 300, &whole, 1));
    rc_registry_destroy(r);
}

static void retired_image_stays_a_provider_until_completed_supersession(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){32, 24, 4});
    assert(r);
    const RcInterval placement = interval(9, 0, 64, 0x1000, 17);
    const RcInterval received = interval(9, 0, 64, 0x2000, 23);
    const RcInterval right = interval(9, 24, 16, 0x5018, 90);
    const RcRegistryBufferSpan buffer = {right, {100, 0, 536}};
    const RcSourceRange possible[] = {{9, 0, 16}, {9, 24, 16}, {9, 48, 8}};
    const RcRegistryImageWriteExtent old = hold(r, 100, placement), newer = hold(r, 200, received);
    const RcRegistryNativeImageWrite shape = {old, possible, 3};
    const RcRegistryImageBinding a = {100, old.hold}, b = {200, newer.hold};
    const RcRegistryWrite first = declare_image(r, placement, at(3)), left = declare_image(r, received, at(5));
    RcRegistryWrite neither, copy;
    OK(rc_registry_write_begin_at(r, right.range, RC_REGISTRY_BUFFER, at(7), &neither));
    OK(rc_registry_write_begin_at(r, right.range, RC_REGISTRY_BUFFER, at(9), &copy));
    OK(rc_registry_declare_through(r, at(14)));
    OK(rc_registry_native_image_possible(r, 100, &shape, 1));
    publish(r, 100, old, possible, 2, at(3));
    record(r, first, 100, old, at(3));
    publish(r, 200, newer, possible, 1, at(5));
    record(r, left, 200, newer, at(5));
    OK(rc_registry_buffer_access_record(r, neither, buffer, at(7)));
    OK(rc_registry_buffer_write_record(r, copy, buffer, at(9)));
    // Releasing descriptor residency does not relinquish the positively authored physical version.
    OK(rc_registry_handoff_through(r, at(3)));
    rc_registry_image_producer_native_completed(r, first, at(3), a);
    OK(rc_registry_write_cancel(r, first));
    assert(rc_registry_native_image_discard_superseded(r, 100, &old.hold, 1) == RC_REGISTRY_BLOCKED);
    RcRegistryNativeImageFragment selected;
    uint32_t count;
    OK(rc_registry_native_image_fragments(r, &right, 1, at(4), &selected, 1, &count));
    expect_fragment(selected, 0, 100, old.hold, interval(9, 24, 16, 0x1018, 17), at(3));

    OK(rc_registry_handoff_through(r, at(7)));
    assert(rc_registry_native_image_discard_superseded(r, 100, &old.hold, 1) == RC_REGISTRY_BLOCKED);
    rc_registry_image_producer_native_completed(r, left, at(5), b);
    OK(rc_registry_write_cancel(r, left));
    OK(rc_registry_write_abandon(r, neither, 1));
    assert(rc_registry_native_image_discard_superseded(r, 100, &old.hold, 1) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_handoff_through(r, at(9)));
    OK(rc_registry_write_submit(r, copy, 1));
    assert(rc_registry_native_image_discard_superseded(r, 100, &old.hold, 1) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_complete(r, copy, 0));
    OK(rc_registry_native_image_discard_superseded(r, 100, &old.hold, 1));
    assert(!rc_registry_image_binding_current(r, a));
    assert(rc_registry_native_image_discard_superseded(r, 100, &old.hold, 1) == RC_REGISTRY_STALE);
    // No fictitious new writes were required in the third region, which this image never authored.
    const RcInterval never_written = interval(9, 48, 8, 0x5030, 90);
    const RcRegistryBufferSpan baseline = {never_written, {100, 0, 560}};
    RcRegistryNativeImageSnapshotFragment untouched;
    OK(rc_registry_native_image_snapshot_mixed(r, &never_written, 1, &baseline, 1, at(10), &untouched, 1, &count, NULL));
    assert(count == 1 && untouched.baseline == 0 && !untouched.fragment.binding.image);
    rc_registry_destroy(r);
}

static void independent_footprint_admission_is_atomic(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){1, 12, 2});
    assert(r);
    RcRegistryImageWriteExtent first = hold(r, 100, interval(9, 0, 64, 0x1000, 17));
    RcRegistryImageWriteExtent second = hold(r, 100, interval(9, 64, 64, 0x1040, 17));
    RcSourceRange spans[32];
    for (unsigned i = 0; i < 32; ++i) spans[i] = (RcSourceRange){9, 64 + i * 2, 1};
    const RcSourceRange written = {9, 0, 16};
    RcRegistryNativeImageWrite shapes[] = {{first, &written, 1}, {second, spans, 32}};
    assert(rc_registry_native_image_possible(r, 100, shapes, 2) == RC_REGISTRY_CAPACITY);
    assert(!r->foreign_span_count && !writer_of(r, first.hold)->image_possible_known &&
           !writer_of(r, second.hold)->image_possible_known);
    shapes[1].owner.interval.mapping++;
    assert(rc_registry_native_image_possible(r, 100, shapes, 2) == RC_REGISTRY_STALE);
    assert(!r->foreign_span_count);
    const RcRegistryWrite claim = declare_image(r, first.interval, at(3));
    OK(rc_registry_declare_through(r, at(8)));
    publish(r, 100, first, &written, 1, at(3));
    record(r, claim, 100, first, at(3));
    OK(rc_registry_handoff_through(r, at(3)));
    rc_registry_image_producer_native_completed(r, claim, at(3), (RcRegistryImageBinding){100, first.hold});
    OK(rc_registry_write_cancel(r, claim));
    OK(rc_registry_image_hold_release_extents(r, &first.hold, 1, 0));
    RcInterval padding = interval(9, 16, 8, 0x3010, 74);
    RcRegistryBufferSpan baseline = {padding, {7, 0, 16}};
    RcRegistryNativeImageSnapshotFragment fragment;
    uint32_t count;
    assert(rc_registry_native_image_snapshot_mixed(r, &padding, 1, &baseline, 1, at(6),
        &fragment, 1, &count, NULL) == RC_REGISTRY_BLOCKED && !count);
    rc_registry_destroy(r);
}

static void retired_independent_image_preserves_exact_padding(void) {
    for (unsigned disposition = 0; disposition < 4; ++disposition) {
        RcRegistry *r = rc_registry_create((RcRegistryConfig){32, 16, 4});
        assert(r);
        const RcInterval placement = interval(9, 0, 64, 0x1000, 17);
        const RcInterval padding = interval(9, 16, 8, 0x3010, 74);
        const RcInterval uncertain = interval(9, 28, 8, 0x301c, 74);
        const RcRegistryBufferSpan baseline = {interval(9, 0, 64, 0x3000, 74), {100, 0, 512}};
        const RcSourceRange possible[] = {{9, 0, 16}, {9, 24, 16}};
        const RcRegistryImageWriteExtent image = hold(r, 100, placement);
        const RcRegistryNativeImageWrite shape = {image, possible, 2};
        const RcRegistryWrite claim = declare_image(r, placement, at(3));
        OK(rc_registry_declare_through(r, at(12)));
        if (disposition != 2) {
            OK(rc_registry_native_image_possible(r, 100, &shape, 1));
            const unsigned spans = r->foreign_span_count;
            OK(rc_registry_native_image_possible(r, 100, &shape, 1));
            assert(r->foreign_span_count == spans);
        }
        const RcRegistryImageBinding binding = {100, image.hold};
        if (disposition == 1 || disposition == 3)
            assert(rc_registry_image_hold_recorded(r, claim, binding, at(3), 1));
        if (disposition != 1)
            publish(r, 100, image, possible, 1, at(3));
        record(r, claim, 100, image, at(3));
        OK(rc_registry_handoff_through(r, at(3)));
        rc_registry_image_producer_native_completed(r, claim, at(3), binding);
        OK(rc_registry_write_cancel(r, claim));
        if (disposition == 3) {
            RcSourceRange visible = {RC_GUEST_COMPATIBILITY_BACKING, placement.address, 16};
            OK(rc_registry_image_hold_readback(r, 100, &image, 1, &visible, 1));
        } else OK(rc_registry_image_hold_release_extents(r, &image.hold, 1, 0));
        RcRegistryNativeImageSnapshotFragment fragment;
        RcRegistryNativeImageFailure failure;
        uint32_t count = 0;
        RcRegistryResult result = rc_registry_native_image_snapshot_mixed(r, &padding, 1, &baseline, 1, at(8),
            &fragment, 1, &count, &failure);
        if (disposition == 2) {
            assert(result == RC_REGISTRY_BLOCKED && !count && failure.cause == RC_NATIVE_IMAGE_BLOCK_FOREIGN);
        } else {
            assert(result == RC_REGISTRY_OK && count == 1 && fragment.baseline == 0 && !fragment.fragment.binding.image);
            assert(fragment.native_buffer.allocation == 100 && fragment.native_buffer.offset == 528);
        }
        assert(rc_registry_native_image_snapshot_mixed(r, &uncertain, 1, &baseline, 1, at(8), &fragment, 1,
            &count, &failure) == RC_REGISTRY_BLOCKED && !count && failure.cause == RC_NATIVE_IMAGE_BLOCK_FOREIGN);
        rc_registry_destroy(r);
    }
}

static RcRegistryWrite record_unprojected_image(RcRegistry *r, RcRegistryImageWriteExtent owner,
    uint64_t image, RcRegistryPosition output, int identified, int install_shape) {
    const RcSourceRange possible[] = {{9, 16, 8}, {9, 40, 8}};
    const RcRegistryNativeImageWrite shape = {owner, possible, 2};
    const RcRegistryImageBinding binding = {image, owner.hold};
    const RcRegistryWrite writer = declare_image(r, owner.interval, output);
    OK(rc_registry_declare_through(r, at(12)));
    assert(rc_registry_image_hold_recorded(r, writer, binding, output, 1));
    if (install_shape) OK(rc_registry_native_image_possible(r, image, &shape, 1));
    OK(rc_registry_image_producer_resolve(r, writer, output, identified ? &binding : NULL, 0));
    return writer;
}

static void storage_image_padding_uses_only_the_exact_older_source(void) {
    const RcInterval place = interval(9, 0, 64, 0x1000, 7);
    const RcInterval prefix = interval(9, 8, 8, 0x1008, 7);
    const RcRegistryBufferSpan raw = {place, {12, 0, 512}};
    RcRegistryNativeImageSnapshotFragment fragments[8];
    RcRegistryNativeImageFailure failure;
    uint32_t count;

    for (int completed = 0; completed <= 1; ++completed) {
        RcRegistry *r = rc_registry_create((RcRegistryConfig){48, 64, 8});
        assert(r);
        const RcRegistryImageWriteExtent owner = hold(r, 100, place);
        const RcRegistryImageBinding binding = {100, owner.hold};
        const RcRegistryWrite writer = record_unprojected_image(r, owner, 100, at(3), 1, 1);
        if (completed) {
            OK(rc_registry_handoff_through(r, at(3)));
            rc_registry_image_producer_native_completed(r, writer, at(3), binding);
            OK(rc_registry_write_cancel(r, writer));
        }
        OK(rc_registry_native_image_snapshot_mixed(r, &prefix, 1, &raw, 1, at(8), fragments, 8, &count, &failure));
        assert(count == 1 && !failure.gate && fragments[0].baseline == 0 &&
               !fragments[0].fragment.binding.image && fragments[0].native_buffer.offset == 520 &&
               rc_range_equal(fragments[0].fragment.interval.range, prefix.range));
        assert(rc_registry_native_image_snapshot_mixed(r, &prefix, 1, NULL, 0, at(8), fragments, 8, &count, &failure) ==
               RC_REGISTRY_BLOCKED && !count && failure.gate == RC_NATIVE_IMAGE_READ_NO_PRODUCER);
        assert(rc_registry_native_image_snapshot_mixed(r, &place, 1, &raw, 1, at(8), fragments, 8, &count, &failure) ==
               RC_REGISTRY_BLOCKED && !count && failure.cause ==
                   (completed ? RC_NATIVE_IMAGE_BLOCK_UNPROJECTED : RC_NATIVE_IMAGE_BLOCK_UNKNOWN_OUTPUT) &&
               failure.interval.range.offset == 16 && failure.interval.range.size == 8);
        RcRegistryBufferSpan another = raw;
        ++another.interval.mapping;
        RcInterval alias = prefix;
        ++alias.mapping;
        assert(rc_registry_native_image_snapshot_mixed(r, &alias, 1, &another, 1, at(8), fragments, 8, &count, &failure) ==
               RC_REGISTRY_BLOCKED && !count);
        another = raw;
        another.interval.address += 0x2000;
        alias = prefix;
        alias.address += 0x2000;
        assert(rc_registry_native_image_snapshot_mixed(r, &alias, 1, &another, 1, at(8), fragments, 8, &count, &failure) ==
               RC_REGISTRY_BLOCKED && !count);
        rc_registry_destroy(r);
    }

    RcRegistry *r = rc_registry_create((RcRegistryConfig){48, 64, 8});
    assert(r);
    const RcRegistryImageWriteExtent owner = hold(r, 100, place);
    const RcRegistryWrite qualified = declare_image(r, place, at(5));
    const RcSourceRange written[] = {{9, 16, 8}, {9, 40, 8}};
    RcRegistryWrite guard;
    OK(rc_registry_write_guard_begin_at(r, RC_REGISTRY_UNKNOWN, at(9), &guard));
    (void)record_unprojected_image(r, owner, 100, at(3), 1, 1);
    publish(r, 100, owner, written, 2, at(5));
    record(r, qualified, 100, owner, at(5));
    OK(rc_registry_native_image_snapshot_mixed(r, &place, 1, &raw, 1, at(8), fragments, 8, &count, &failure));
    assert(count == 5);
    const uint64_t begin[] = {0, 16, 24, 40, 48}, sizes[] = {16, 8, 16, 8, 16};
    for (uint32_t i = 0; i < count; ++i) {
        const RcInterval expected = interval(9, begin[i], sizes[i], 0x1000 + begin[i], 7);
        if (i & 1) expect_fragment(fragments[i].fragment, 0, 100, owner.hold, expected, at(5));
        else {
            assert(fragments[i].baseline == 0 && !fragments[i].fragment.binding.image &&
                   fragments[i].native_buffer.offset == 512 + begin[i] &&
                   rc_range_equal(fragments[i].fragment.interval.range, expected.range));
        }
    }
    assert(rc_registry_native_image_snapshot_mixed(r, &prefix, 1, &raw, 1, at(10), fragments, 8, &count, &failure) ==
           RC_REGISTRY_BLOCKED && !count && failure.cause == RC_NATIVE_IMAGE_BLOCK_GUARD);
    rc_registry_destroy(r);

    for (int disposition = 0; disposition < 4; ++disposition) {
        r = rc_registry_create((RcRegistryConfig){48, 64, 8});
        assert(r);
        const RcRegistryImageWriteExtent blocked = hold(r, 100, place);
        const RcRegistryImageBinding binding = {100, blocked.hold};
        const RcRegistryWrite writer = record_unprojected_image(r, blocked, 100, at(3), disposition != 0, disposition != 1);
        if (disposition == 2)
            assert(rc_registry_native_image_written(r, 100, &place, &blocked.hold, 1, at(5)));
        if (disposition == 3)
            assert(rc_registry_image_producer_projection_abandon(r, writer, at(3), binding, 1));
        assert(rc_registry_native_image_snapshot_mixed(r, &prefix, 1, &raw, 1, at(8), fragments, 8, &count, &failure) ==
               RC_REGISTRY_BLOCKED && !count);
        if (disposition == 3) {
            OK(rc_registry_handoff_through(r, at(3)));
            OK(rc_registry_write_cancel(r, writer));
            assert(rc_registry_native_image_snapshot_mixed(r, &prefix, 1, &raw, 1, at(8), fragments, 8, &count,
                &failure) == RC_REGISTRY_BLOCKED && !count && failure.cause == RC_NATIVE_IMAGE_BLOCK_UNPROJECTED);
        }
        rc_registry_destroy(r);
    }
}

static void resident_image_preserves_initialized_texels(void) {
    enum { PRESERVED, UNMARKED, WRONG_HOLD, UNCERTAIN, RAW_OVERLAP, RAW_SIBLING };
    for (int mode = PRESERVED; mode <= RAW_SIBLING; ++mode) {
        RcRegistry *r = rc_registry_create((RcRegistryConfig){20, 24, 4});
        assert(r);
        const RcInterval place = interval(1346, 0x3f48e00, 0xb64000, 0x8003f48e00, 1349);
        const RcInterval demand = interval(1346, 0x3f48e00, 0x142040, 0x8003f48e00, 1349);
        const RcInterval missing = interval(1346, 0x408ae3f, 2, 0x800408ae3f, 1349);
        const RcInterval sibling = interval(1346, 0x4400000, 0x2000, 0x8004400000, 1349);
        const RcRegistryPosition seed_at = {13303, 46893}, atomic_at = {13303, 46915};
        const RcRegistryPosition atomic_read = {13303, 46914}, sample_at = {13303, 46920};
        const RcRegistryImageWriteExtent owner = hold(r, 104032, place);
        const RcRegistryImageBinding binding = {104032, owner.hold};
        const RcRegistryWrite seed = declare_image(r, place, seed_at);
        const RcRegistryWrite atomic = declare_image(r, place, atomic_at);
        RcRegistryWrite raw = {0};
        if (mode == RAW_OVERLAP || mode == RAW_SIBLING) {
            const uint64_t offset = mode == RAW_OVERLAP ? demand.range.offset + 0x200 : sibling.range.offset + 1;
            const uint64_t ordinal = mode == RAW_OVERLAP ? 46917 : 46909;
            OK(rc_registry_write_begin_at(r, (RcSourceRange){1346, offset, 1}, RC_REGISTRY_BUFFER,
                                          (RcRegistryPosition){13303, ordinal}, &raw));
        }
        OK(rc_registry_declare_through(r, sample_at));
        assert(rc_registry_image_hold_recorded(r, seed, binding, seed_at, 1));
        const RcSourceRange seeded[] = {demand.range, sibling.range};
        publish(r, 104032, owner, seeded, 2, seed_at);
        record(r, seed, 104032, owner, seed_at);
        if (mode == RAW_SIBLING) {
            OK(rc_registry_handoff_through(r, (RcRegistryPosition){13303, 46909}));
            OK(rc_registry_write_abandon(r, raw, 1));
        }
        assert(rc_registry_native_image_covers(r, 104032, &demand, 1, atomic_read));
        assert(rc_registry_native_image_resident_covers(r, 104032, &owner.hold, 1, &demand, 1, atomic_read));
        assert(!rc_registry_native_image_resident_covers(r, 104032, &owner.hold, 1, &missing, 1, atomic_read));
        assert(rc_registry_native_image_resident_covers(r, 104032, &owner.hold, 1, &sibling, 1, atomic_read) ==
               (mode != RAW_SIBLING));
        assert(!rc_registry_native_image_resident_covers(r, 104032, &owner.hold, 1, &demand, 1, sample_at));
        assert(rc_registry_image_hold_recorded(r, atomic, binding, atomic_at, 0));
        if (mode != UNMARKED) assert(rc_registry_image_hold_preserved(r, atomic, binding, atomic_at, 0));
        RcRegistryImageWriteExtent resolved = mode == WRONG_HOLD ? hold(r, 104032, place) : owner;
        record(r, atomic, 104032, resolved, atomic_at);
        if (mode == UNCERTAIN)
            assert(rc_registry_image_producer_projection_abandon(r, atomic, atomic_at, binding, 1));
        assert(rc_registry_native_image_resident_covers(r, 104032, &owner.hold, 1, &demand, 1, sample_at) ==
               (mode == PRESERVED || mode == RAW_SIBLING));
        assert(rc_registry_native_image_resident_covers(r, 104032, &owner.hold, 1, &sibling, 1, sample_at) ==
               (mode == PRESERVED || mode == RAW_OVERLAP));
        assert(!rc_registry_native_image_covers(r, 104032, &demand, 1, sample_at));
        RcRegistryNativeImageFragment fragment;
        uint32_t count;
        assert(rc_registry_native_image_snapshot_fragments(r, &demand, 1, sample_at, &fragment, 1, &count,
                                                           NULL) == RC_REGISTRY_BLOCKED && !count);
        if (mode == PRESERVED) {
            RcInterval remapped = demand;
            remapped.mapping++;
            assert(!rc_registry_native_image_resident_covers(r, 104032, &owner.hold, 1, &remapped, 1, sample_at));
            remapped = demand;
            remapped.address++;
            assert(!rc_registry_native_image_resident_covers(r, 104032, &owner.hold, 1, &remapped, 1, sample_at));
            assert(!rc_registry_native_image_resident_covers(r, 104032, &owner.hold, 1, &missing, 1, sample_at));
        }
        OK(rc_registry_handoff_through(r, atomic_at));
        rc_registry_image_producer_native_completed(r, seed, seed_at, binding);
        rc_registry_image_producer_native_completed(r, atomic, atomic_at, binding);
        OK(rc_registry_write_cancel(r, seed));
        OK(rc_registry_write_cancel(r, atomic));
        assert(rc_registry_native_image_resident_covers(r, 104032, &owner.hold, 1, &demand, 1, sample_at) ==
               (mode == PRESERVED || mode == RAW_SIBLING));
        assert(rc_registry_native_image_resident_covers(r, 104032, &owner.hold, 1, &sibling, 1, sample_at) ==
               (mode == PRESERVED || mode == RAW_OVERLAP));
        assert(!rc_registry_native_image_covers(r, 104032, &demand, 1, sample_at));
        rc_registry_destroy(r);
    }
}

static void resident_image_requires_each_original_mapping(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){20, 24, 4});
    assert(r);
    const RcInterval first = interval(21, 0x1000, 0x100, 0x5f00, 1349);
    const RcInterval second = interval(22, 0x7000, 0x100, 0x6000, 1350);
    const RcInterval demand[] = {first, second};
    const RcRegistryImageWriteExtent a = hold(r, 104032, first), b = hold(r, 104032, second);
    const RcRegistryWrite holds[] = {a.hold, b.hold};
    const RcRegistryImageBinding a_bind = {104032, a.hold}, b_bind = {104032, b.hold};
    const RcRegistryWrite seed_a = declare_image(r, first, at(3)), seed_b = declare_image(r, second, at(3));
    const RcRegistryWrite atomic_a = declare_image(r, first, at(7)), unknown_b = declare_image(r, second, at(7));
    OK(rc_registry_declare_through(r, at(10)));
    assert(rc_registry_image_hold_recorded(r, seed_a, a_bind, at(3), 1));
    assert(rc_registry_image_hold_recorded(r, seed_b, b_bind, at(3), 1));
    publish(r, 104032, a, &first.range, 1, at(3));
    publish(r, 104032, b, &second.range, 1, at(3));
    record(r, seed_a, 104032, a, at(3));
    record(r, seed_b, 104032, b, at(3));
    assert(rc_registry_native_image_resident_covers(r, 104032, holds, 2, demand, 2, at(6)));
    assert(!rc_registry_native_image_resident_covers(r, 104032, holds, 1, demand, 2, at(6)));
    assert(rc_registry_image_hold_recorded(r, atomic_a, a_bind, at(7), 0));
    assert(!rc_registry_image_hold_preserved(r, unknown_b, a_bind, at(7), 0));
    assert(rc_registry_image_hold_preserved(r, atomic_a, a_bind, at(7), 0));
    record(r, atomic_a, 104032, a, at(7));
    record(r, unknown_b, 104032, b, at(7));
    assert(rc_registry_native_image_resident_covers(r, 104032, holds, 2, demand, 1, at(8)));
    assert(!rc_registry_native_image_resident_covers(r, 104032, holds, 2, demand, 2, at(8)));
    rc_registry_destroy(r);
}

static void handed_conditional_image_exports_initial_projection(void) {
    enum { EXPORT, RAW, NEW_IMAGE, OTHER_IMAGE_AT_OUTPUT, CANCELED, UNCERTAIN, RESIDENT_ONLY, UNKNOWN_LAYOUT };
    for (int mode = EXPORT; mode <= UNKNOWN_LAYOUT; ++mode) {
        RcRegistry *r = rc_registry_create((RcRegistryConfig){32, 32, 8});
        assert(r);
        const RcInterval place = interval(9, 0, 64, 0x1000, 7);
        const RcInterval demand = interval(9, 16, 16, 0x1010, 7);
        const RcInterval outside = interval(9, 32, 8, 0x1020, 7);
        const RcInterval sibling = interval(9, 40, 8, 0x1028, 7);
        const RcSourceRange meaningful[] = {demand.range, sibling.range};
        const RcRegistryImageWriteExtent owner = hold(r, 100, place);
        const RcRegistryImageBinding binding = {100, owner.hold};
        const RcRegistryNativeImageWrite shape = {owner, meaningful, 2};
        const RcRegistryBufferSpan baseline = {place, {20, 1, 256}};
        const RcRegistryBufferSpan raw = {demand, {20, 1, 272}};
        const RcRegistryWrite seed = declare_image(r, place, at(3));
        const RcRegistryWrite atomic = declare_image(r, place, at(7));
        RcRegistryWrite raw_writer = {0};
        if (mode == RAW)
            OK(rc_registry_write_begin_at(r, demand.range, RC_REGISTRY_BUFFER, at(9), &raw_writer));
        const uint64_t other_output = mode == NEW_IMAGE ? 11 : 7;
        RcRegistryWrite other = {0};
        if (mode == NEW_IMAGE || mode == OTHER_IMAGE_AT_OUTPUT)
            other = declare_image(r, demand, at(other_output));
        OK(rc_registry_declare_through(r, at(14)));
        if (mode != UNKNOWN_LAYOUT) OK(rc_registry_native_image_possible(r, 100, &shape, 1));
        assert(rc_registry_image_hold_recorded(r, seed, binding, at(3), 1));
        publish(r, 100, owner, meaningful, 2, at(3));
        record(r, seed, 100, owner, at(3));
        OK(rc_registry_handoff_through(r, at(3)));
        rc_registry_image_producer_native_completed(r, seed, at(3), binding);
        OK(rc_registry_write_cancel(r, seed));
        RcRegistryNativeImageSnapshotFragment fragments[5];
        RcRegistryNativeImageFailure failure;
        uint32_t count;
        OK(rc_registry_native_image_snapshot_mixed(r, &demand, 1, &baseline, 1, at(6), fragments, 5, &count, &failure));
        assert(count == 1 && fragments[0].baseline == UINT32_MAX);
        expect_fragment(fragments[0].fragment, 0, 100, owner.hold, demand, at(3));
        assert(rc_registry_image_hold_preserved(r, atomic, binding, at(7), mode != RESIDENT_ONLY));
        record(r, atomic, 100, owner, at(7));
        RcRegistryImageWriteExtent replacement = {0};
        if (other.id) {
            replacement = hold(r, 200, demand);
            publish(r, 200, replacement, &demand.range, 1, at(other_output));
            record(r, other, 200, replacement, at(other_output));
        }
        if (raw_writer.id) OK(rc_registry_buffer_write_record(r, raw_writer, raw, at(9)));
        // A resolved recording alone cannot authorize alias bytes.
        assert(rc_registry_native_image_snapshot_mixed(r, &sibling, 1, &baseline, 1, at(8),
            fragments, 5, &count, &failure) == RC_REGISTRY_BLOCKED && !count);
        if (mode == CANCELED || mode == UNCERTAIN)
            assert(rc_registry_image_producer_projection_abandon(r, atomic, at(7), binding, mode == UNCERTAIN));
        if (mode != CANCELED) {
            OK(rc_registry_handoff_through(r, at(7)));
            if (mode != UNCERTAIN) rc_registry_image_producer_native_completed(r, atomic, at(7), binding);
            // The exact completed position can remain in a shared live claim after handoff.
            if (mode == EXPORT) {
                OK(rc_registry_native_image_snapshot_mixed(r, &demand, 1, &baseline, 1, at(8),
                    fragments, 5, &count, &failure));
                expect_fragment(fragments[0].fragment, 0, 100, owner.hold, demand, at(7));
            }
        }
        OK(rc_registry_write_cancel(r, atomic));
        if (mode == OTHER_IMAGE_AT_OUTPUT) {
            // Give an equal-position competing projection a real completed claim before selection.
            rc_registry_image_producer_native_completed(r, other, at(7), (RcRegistryImageBinding){200, replacement.hold});
            OK(rc_registry_write_cancel(r, other));
        }
        const RcRegistryResult selected = rc_registry_native_image_snapshot_mixed(r, &demand, 1, &baseline, 1,
            at(14), fragments, 5, &count, &failure);
        if (mode == RAW) {
            assert(selected == RC_REGISTRY_OK && count == 1 && fragments[0].baseline == 0 &&
                   fragments[0].fragment.output.ordinal == 9 && fragments[0].native_buffer.offset == 272);
        } else if (mode == NEW_IMAGE) {
            assert(selected == RC_REGISTRY_OK && count == 1);
            expect_fragment(fragments[0].fragment, 0, 200, replacement.hold, demand, at(11));
        } else if (mode == EXPORT || mode == CANCELED) {
            assert(selected == RC_REGISTRY_OK && count == 1);
            expect_fragment(fragments[0].fragment, 0, 100, owner.hold, demand, at(mode == EXPORT ? 7 : 3));
        } else {
            assert(selected == RC_REGISTRY_BLOCKED && !count);
        }
        if (mode == EXPORT) {
            assert(rc_registry_native_image_snapshot_mixed(r, &outside, 1, NULL, 0, at(14),
                fragments, 5, &count, &failure) == RC_REGISTRY_BLOCKED && !count);
            OK(rc_registry_native_image_snapshot_mixed(r, &outside, 1, &baseline, 1, at(14),
                fragments, 5, &count, &failure));
            assert(count == 1 && fragments[0].baseline == 0 && !fragments[0].fragment.binding.image);
            RcInterval remap = demand;
            remap.mapping++;
            assert(rc_registry_native_image_snapshot_mixed(r, &remap, 1, NULL, 0, at(14),
                fragments, 5, &count, &failure) == RC_REGISTRY_BLOCKED && !count);
            remap = demand;
            remap.address++;
            assert(rc_registry_native_image_snapshot_mixed(r, &remap, 1, NULL, 0, at(14),
                fragments, 5, &count, &failure) == RC_REGISTRY_BLOCKED && !count);
        }
        rc_registry_destroy(r);
    }
}


static void repeated_completed_buffer_keeps_typed_storage(void) {
    for (int mode = 0; mode < 5; ++mode) {
        RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 24, 4});
        assert(r);
        const RcInterval region = interval(9, 16, 32, 0x5010, 18);
        const RcInterval exact = interval(9, 24, 24, 0x5018, 18);
        const RcRegistryBufferSpan raw = {exact, {20, 2, 536}};
        const RcInterval image_interval = interval(9, 16, 8, 0x5010, 18);
        const RcRegistryImageWriteExtent image = hold(r, 100, image_interval);
        RcRegistryWrite image_write = declare_image(r, image_interval, at(3));
        RcRegistryWrite first, second;
        OK(rc_registry_write_begin_at(r, exact.range, RC_REGISTRY_BUFFER, at(5), &first));
        RcRegistryBufferSpan next = raw;
        if (mode == 2) next.native.incarnation++;
        if (mode == 3) next.interval.range.size /= 2;
        OK(rc_registry_write_begin_at(r, next.interval.range, RC_REGISTRY_BUFFER, at(7), &second));
        OK(rc_registry_declare_through(r, at(10)));
        OK(rc_registry_buffer_write_record(r, first, raw, at(5)));
        if (mode != 1) OK(rc_registry_buffer_write_record(r, second, next, at(7)));
        publish(r, 100, image, &image_interval.range, 1, at(3));
        record(r, image_write, 100, image, at(3));
        OK(rc_registry_handoff_through(r, at(3)));
        rc_registry_image_producer_native_completed(r, image_write, at(3), (RcRegistryImageBinding){100, image.hold});
        OK(rc_registry_write_cancel(r, image_write));

        RcRegistryNativeImageSnapshotFragment result[3];
        RcRegistryNativeImageFailure failure;
        uint32_t count;
        OK(rc_registry_native_image_snapshot_mixed(r, &region, 1, &raw, 1, at(4), result, 3, &count, &failure));
        assert(count == 2 && result[0].fragment.binding.image == 100 &&
               !rc_registry_position_valid(result[1].fragment.output));
        OK(rc_registry_handoff_through(r, at(5)));
        OK(rc_registry_write_submit(r, first, 1));
        RcRegistryWriteCompletion complete = {.ticket=first,.written=&exact.range,.written_count=1,
            .uncertain=&exact.range,.uncertain_count=1};
        OK(rc_registry_write_complete_extents(r, &complete, 1));
        OK(rc_registry_native_image_snapshot_mixed(r, &region, 1, &raw, 1, at(6), result, 3, &count, &failure));
        assert(count == 2 && result[1].fragment.output.ordinal == 5);
        if (mode == 4) {
            OK(rc_registry_write_cancel(r, second));
        } else {
            OK(rc_registry_handoff_through(r, at(7)));
            OK(rc_registry_write_submit(r, second, 1));
            complete = (RcRegistryWriteCompletion){.ticket=second,.written=&next.interval.range,.written_count=1,
                .uncertain=&next.interval.range,.uncertain_count=1};
            OK(rc_registry_write_complete_extents(r, &complete, 1));
        }
        const RcRegistryResult status = rc_registry_native_image_snapshot_mixed(r, &region, 1,
            &raw, 1, at(8), result, 3, &count, &failure);
        if (mode == 1 || mode == 2) {
            assert(status == RC_REGISTRY_BLOCKED && !count);
        } else {
            const uint64_t expected_output = mode == 4 ? 5 : 7;
            assert(status == RC_REGISTRY_OK && count == (mode == 3 ? 3 : 2));
            assert(result[0].fragment.binding.image == 100 && result[1].native_buffer.allocation == raw.native.allocation &&
                   result[1].native_buffer.offset == raw.native.offset &&
                   result[1].fragment.output.ordinal == expected_output);
            if (mode == 3) assert(result[2].fragment.output.ordinal == 5 && result[2].native_buffer.offset == 548);
            if (!mode) {
                assert(rc_registry_stats(r).writer_foreign == 1);
                assert(rc_registry_native_image_snapshot_handed(r, &region, 1, &raw, 1, at(5), result, 3,
                    &count, &failure) == RC_REGISTRY_STALE && !count && failure.gate == RC_NATIVE_IMAGE_READ_CLOSED);
                OK(rc_registry_native_image_snapshot_handed(r, &region, 1, &raw, 1, at(7), result, 3, &count, &failure));
                assert(count == 2 && result[1].fragment.output.ordinal == 7);
            }
        }
        rc_registry_destroy(r);
    }
}

static void writable_buffer_requires_its_actual_original_contents(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 24, 4});
    assert(r);
    const RcInterval demand = interval(9, 0, 64, 0x5000, 20);
    const RcInterval place = interval(9, 0, 64, 0x1000, 17);
    const RcSourceRange authored = {9, 16, 16};
    const RcRegistryBufferSpan selected = {demand, {33, 0, 256}};
    OK(rc_registry_declare_through(r, at(2)));
    // A concrete VkBuffer and valid mapping do not provide the predecessor by themselves.
    assert(rc_registry_buffer_preserve_input_at(r, selected, at(2), 0) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_buffer_preserve_input_at(r, selected, at(2), 1));
    const RcRegistryImageWriteExtent image = hold(r, 100, place);
    RcRegistryWrite prior = declare_image(r, place, at(3)), storage;
    OK(rc_registry_write_begin_at(r, demand.range, RC_REGISTRY_BUFFER, at(5), &storage));
    OK(rc_registry_declare_through(r, at(10)));

    publish(r, 100, image, &authored, 1, at(3));
    record(r, prior, 100, image, at(3));
    assert(rc_registry_buffer_preserve_input_at(r, selected, at(4), 1) == RC_REGISTRY_BLOCKED);

    // The prior bytes are nevertheless accessible via the mixed input path. It must retain the
    // actual independent image, with the initialized raw carrier for the unaffected parts.
    RcRegistryNativeImageSnapshotFragment fragments[3];
    uint32_t count;
    OK(rc_registry_native_image_snapshot_mixed(r, &demand, 1, &selected, 1, at(4), fragments, 3, &count, NULL));
    assert(count == 3 && fragments[0].native_buffer.offset == 256 &&
        fragments[1].fragment.binding.image == 100 && fragments[1].fragment.interval.range.size == 16 &&
        fragments[2].native_buffer.offset == 288);

    // Merely binding the raw carrier or recording its possible access cannot authorize that result.
    OK(rc_registry_buffer_access_record(r, storage, selected, at(5)));
    assert(rc_registry_buffer_preserved_record(r, storage, selected, at(4), 1) == RC_REGISTRY_BLOCKED);
    assert(rc_registry_buffer_preserve_input_at(r, selected, at(6), 1) == RC_REGISTRY_BLOCKED);

    // Only the issued shader after an actual complete seed can publish an initialized native
    // result for the whole preserved extent. No new guest writer is needed for the copy.
    OK(rc_registry_buffer_write_record(r, storage, selected, at(5)));
    assert(!rc_registry_position_handed(r, at(5)));
    OK(rc_registry_buffer_preserve_input_at(r, selected, at(6), 0));
    RcRegistryBufferSpan different = selected;
    different.native.allocation++;
    assert(rc_registry_buffer_preserve_input_at(r, different, at(6), 1) == RC_REGISTRY_BLOCKED);
    different = selected;
    different.native.offset += 4;
    assert(rc_registry_buffer_preserve_input_at(r, different, at(6), 1) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_cancel(r, storage));
    assert(rc_registry_buffer_preserve_input_at(r, selected, at(6), 1) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_handoff_through(r, at(3)));
    OK(rc_registry_write_submit(r, prior, 1));
    rc_registry_image_producer_native_completed(r, prior, at(3), (RcRegistryImageBinding){100, image.hold});
    OK(rc_registry_write_complete_effects(r, prior, NULL, 0, NULL, 0));
    OK(rc_registry_image_hold_release_extents(r, &image.hold, 1, 0));
    assert(rc_registry_buffer_preserve_input_at(r, selected, at(8), 1) == RC_REGISTRY_BLOCKED);
    rc_registry_destroy(r);
}

int main(void) {
    writable_buffer_requires_its_actual_original_contents();
    repeated_completed_buffer_keeps_typed_storage();
    handed_conditional_image_exports_initial_projection();
    resident_image_preserves_initialized_texels();
    resident_image_requires_each_original_mapping();
    storage_image_padding_uses_only_the_exact_older_source();
    retired_independent_image_preserves_exact_padding();
    retired_image_stays_a_provider_until_completed_supersession();
    independent_footprint_admission_is_atomic();
    raw_buffer_observes_independent_image_authority();
    image_coverage_at_inclusive_handed_frontier();
    mixed_snapshot_uses_only_exact_current_providers();
    mixed_snapshot_keeps_many_image_versions_of_one_receiver();
    mixed_snapshot_does_not_use_raw_padding_under_bound_image();
    mixed_snapshot_partitions_affine_baselines_per_demand();
    clear_and_copy_keep_their_exact_same_image_planes();
    padded_bound_writer_leaves_older_independent_bytes_current();
    ordered_bound_snapshot_keeps_registered_footprint();
    exact_native_fragment_refusal();
    positioned_legacy_image_writes_do_not_hide_a_later_exact_provider();
    later_projection_can_replace_a_handed_retired_provider();
    relocate_whole_bound_image();
    relocate_padded_bound_image();
    relocate_bound_recording_fate();
    latest_exact_sparse_alias();
    canceled_and_settled_writes();
    contradictory_output_cannot_be_proved_after_handoff();
    unprojected_actual_output_survives_claim_retirement();
    exact_fragment_capacity();
    puts("native image fragments: ordered sparse alias winners, physical lifetime, cancellation, overwrite and bounded coverage passed");
    return 0;
}

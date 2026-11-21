#include <assert.h>
#include <stdio.h>

#define RC_REGISTRY_SINGLE_THREADED 1
#include "../runtime/resource_core_registry.c"

#define OK(call) assert((call) == RC_REGISTRY_OK)

static const RcRegistryPosition read = {5, 10};
static const RcRegistryPosition task_output = {5, 11};
static const RcRegistryPosition prior_output = {5, 9};
static const RcRegistryPosition later_read = {6, 2};

static RcRegistryBufferPhaseExtent declare_table(RcRegistry *r, uint64_t address, uint64_t mapping,
                                                 uint64_t backing, uint64_t offset) {
    RcRegistryBufferPhaseExtent table = {.interval = {.address = address, .mapping = mapping, .range = {backing, offset, 64}}};
    OK(rc_registry_write_begin_at(r, table.interval.range, RC_REGISTRY_BUFFER, task_output, &table.writer));
    return table;
}

static void positively_completed_task(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){8, 16, 4});
    assert(r);
    RcRegistryWrite guard;
    OK(rc_registry_write_guard_begin_at(r, RC_REGISTRY_UNKNOWN, (RcRegistryPosition){5, 7}, &guard));
    OK(rc_registry_declare_through(r, read));
    RcRegistryBufferPhaseExtent extents[] = {
        declare_table(r, 0x1000, 10, 2, 128), declare_table(r, 0x2000, 21, 3, 256),
    };
    RcRegistryWrite unissued;
    OK(rc_registry_write_begin_at(r, (RcSourceRange){4, 0, 64}, RC_REGISTRY_BUFFER, task_output, &unissued));
    OK(rc_registry_buffer_phase_bind(r, extents, 2, read));
    OK(rc_registry_declare_through(r, task_output));
    RcInterval requested = {.address = 0x8008, .mapping = 33, .range = {2, 136, 8}};
    assert(rc_registry_read_device_buffer_phase(r, &requested, 1, extents, 2, read) == RC_REGISTRY_STALE);
    OK(rc_registry_handoff_through(r, prior_output));
    assert(!rc_registry_position_handed(r, task_output));
    RcRegistryBufferPhaseExtent changed[] = {extents[0], extents[1]};
    changed[1].interval.mapping++;
    assert(rc_registry_buffer_phase_note(r, changed, 2, read, RC_REGISTRY_BUFFER_PHASE_COMPLETED) == RC_REGISTRY_STALE);
    assert(rc_registry_read_device_buffer_phase(r, &requested, 1, extents, 1, read) == RC_REGISTRY_STALE);
    OK(rc_registry_buffer_phase_note(r, extents, 2, read, RC_REGISTRY_BUFFER_PHASE_COMPLETED));
    assert(rc_registry_buffer_phase_note(r, extents, 2, read, RC_REGISTRY_BUFFER_PHASE_COMPLETED) == RC_REGISTRY_STALE);
    assert(rc_registry_read_device_buffer_phase(r, &requested, 1, extents, 2, read) == RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_cancel(r, guard));
    OK(rc_registry_read_device_buffer_phase(r, &requested, 1, extents, 2, read));
    RcInterval partial = requested;
    partial.range.offset = 188;
    assert(rc_registry_read_device_buffer_phase(r, &partial, 1, extents, 2, read) == RC_REGISTRY_BLOCKED);
    partial.range.backing = 4;
    assert(rc_registry_read_device_buffer_phase(r, &partial, 1, extents, 2, read) == RC_REGISTRY_BLOCKED);
    // A final-tail failure after the successful TASK marker cancels only genuinely unissued writers.
    RcRegistryWrite pending[] = {extents[0].writer, unissued, extents[1].writer};
    OK(rc_registry_write_cancel_extents(r, pending, 3));
    assert(rc_registry_stats(r).writers == 2 && rc_registry_stats(r).writer_foreign == 2);
    assert(rc_registry_read_device_buffer_phase(r, &requested, 1, extents, 2, read) == RC_REGISTRY_STALE);
    assert(!rc_registry_position_handed(r, task_output));
    OK(rc_registry_declare_through(r, later_read));
    OK(rc_registry_read_device_buffer_at(r, &requested, 1, later_read));
    assert(rc_registry_read_range_at(r, requested.range, later_read) == RC_REGISTRY_BLOCKED);
    rc_registry_destroy(r);
}

static void no_submission_and_uncertain_submission(void) {
    for (int possible = 0; possible < 2; ++possible) {
        RcRegistry *r = rc_registry_create((RcRegistryConfig){8, 16, 4});
        assert(r);
        OK(rc_registry_declare_through(r, read));
        RcRegistryBufferPhaseExtent extent = declare_table(r, 0x1000, 10, 2, 128);
        OK(rc_registry_buffer_phase_bind(r, &extent, 1, read));
        OK(rc_registry_declare_through(r, task_output));
        if (possible) OK(rc_registry_buffer_phase_note(r, &extent, 1, read, RC_REGISTRY_BUFFER_PHASE_POSSIBLY_ISSUED));
        assert(rc_registry_read_device_buffer_phase(r, &extent.interval, 1, &extent, 1, read) == RC_REGISTRY_STALE);
        OK(rc_registry_write_abandon(r, extent.writer, 0));
        assert(rc_registry_stats(r).writer_foreign == (unsigned)possible);
        OK(rc_registry_declare_through(r, later_read));
        assert(rc_registry_read_device_buffer_at(r, &extent.interval, 1, later_read) ==
               (possible ? RC_REGISTRY_BLOCKED : RC_REGISTRY_OK));
        rc_registry_destroy(r);
    }
}

int main(void) {
    positively_completed_task();
    no_submission_and_uncertain_submission();
    puts("native writer phases: actual part receipt, scoped aliases, incomplete range and tail disposition passed");
    return 0;
}

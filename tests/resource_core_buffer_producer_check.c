#include <assert.h>
#include <stdio.h>

#define RC_REGISTRY_SINGLE_THREADED 1
#include "../runtime/resource_core_registry.c"

#define OK(call) assert((call) == RC_REGISTRY_OK)
static RcRegistryPosition at(uint64_t ordinal) { return (RcRegistryPosition){1, ordinal}; }

static RcRegistryBufferSpan span(uint64_t map, uint64_t address, uint64_t offset, uint64_t size, uint64_t native) {
    return (RcRegistryBufferSpan){{address, map, {4, offset, size}}, {90, 2, native}};
}

static RcRegistryWrite reserve(RcRegistry *r, RcRegistryBufferSpan write, uint64_t ordinal) {
    RcRegistryWrite ticket;
    OK(rc_registry_write_begin_at(r, write.interval.range, RC_REGISTRY_BUFFER, at(ordinal), &ticket));
    return ticket;
}

static void query(RcRegistry *r, RcRegistryGpuCapture capture, RcRegistryBufferSpan read, uint64_t ordinal,
                  RcRegistryBufferFragment *fragments, uint32_t capacity, uint32_t count) {
    uint32_t n;
    OK(rc_registry_gpu_capture_buffer_fragments(r, capture, &read, 1, at(ordinal), fragments, capacity, &n));
    assert(n == count);
}

static void expect(RcRegistryBufferFragment f, uint64_t address, uint64_t bytes, uint64_t position) {
    assert(f.interval.address == address && f.interval.mapping == 9 && f.interval.range.size == bytes);
    assert(f.output.scene == 1 && f.output.ordinal == position && !f.demand);
}

static void prior_native_fragments(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 16, 4});
    assert(r);
    const RcRegistryBufferSpan demanded = span(9, 0x2000, 100, 64, 1000);
    const RcRegistryBufferSpan first = span(7, 0x1004, 104, 32, 1004);
    const RcRegistryBufferSpan later = span(8, 0x1010, 116, 16, 1016);
    RcRegistryGpuCapture snapshot = {0};
    OK(rc_registry_gpu_capture_begin(r, &demanded.interval.range, 1, &snapshot));
    RcRegistryWrite a = reserve(r, first, 3), b = reserve(r, later, 7);
    RcRegistryWrite uncertain = reserve(r, span(7, 0x1008, 108, 4, 1008), 11);
    RcRegistryWrite guard;
    OK(rc_registry_write_guard_begin_at(r, RC_REGISTRY_UNKNOWN, at(15), &guard));
    OK(rc_registry_declare_through(r, at(16)));
    RcRegistryBufferFragment f[4];
    uint32_t count = 22;
    assert(rc_registry_gpu_capture_buffer_fragments(r, snapshot, &demanded, 1, at(4), f, 4, &count) ==
        RC_REGISTRY_BLOCKED && !count);
    RcRegistryBufferSpan wrong = first;
    wrong.interval.range.size--;
    assert(rc_registry_buffer_write_record(r, a, wrong, at(3)) == RC_REGISTRY_STALE);
    OK(rc_registry_buffer_write_record(r, a, first, at(3)));
    OK(rc_registry_buffer_write_record(r, b, later, at(7)));
    query(r, snapshot, demanded, 4, f, 4, 1);
    expect(f[0], 0x2004, 32, 3);
    query(r, snapshot, demanded, 8, f, 4, 3);
    expect(f[0], 0x2004, 12, 3);
    expect(f[1], 0x2010, 16, 7);
    expect(f[2], 0x2020, 4, 3);
    assert(rc_registry_gpu_capture_buffer_fragments(r, snapshot, &demanded, 1, at(8), f, 2, &count) ==
        RC_REGISTRY_CAPACITY && !count);
    wrong = demanded;
    ++wrong.native.incarnation;
    assert(rc_registry_gpu_capture_buffer_fragments(r, snapshot, &wrong, 1, at(8), f, 4, &count) ==
        RC_REGISTRY_BLOCKED && !count);
    ++wrong.interval.range.backing;
    assert(rc_registry_gpu_capture_buffer_fragments(r, snapshot, &wrong, 1, at(8), f, 4, &count) ==
        RC_REGISTRY_INVALID && !count);
    assert(rc_registry_gpu_capture_buffer_fragments(r, snapshot, &demanded, 1, at(12), f, 4, &count) ==
        RC_REGISTRY_BLOCKED && !count);
    OK(rc_registry_write_cancel(r, uncertain));
    OK(rc_registry_handoff_through(r, at(7)));
    OK(rc_registry_write_submit(r, a, 1));
    OK(rc_registry_write_complete_effects(r, a, &first.interval.range, 1, NULL, 0));
    OK(rc_registry_write_submit(r, b, 1));
    OK(rc_registry_write_complete_effects(r, b, &later.interval.range, 1, &later.interval.range, 1));
    query(r, snapshot, demanded, 12, f, 4, 3);
    expect(f[0], 0x2004, 12, 3);
    expect(f[1], 0x2010, 16, 7);
    expect(f[2], 0x2020, 4, 3);
    assert(rc_registry_gpu_capture_buffer_fragments(r, snapshot, &demanded, 1, at(16), f, 4, &count) ==
        RC_REGISTRY_BLOCKED && !count);
    OK(rc_registry_write_cancel(r, guard));
    query(r, snapshot, demanded, 16, f, 4, 3);
    OK(rc_registry_gpu_capture_end(r, &snapshot));
    rc_registry_destroy(r);
}

static void cancellation_and_retention(void) {
    for (int uncertain = 0; uncertain != 2; ++uncertain) {
        RcRegistry *r = rc_registry_create((RcRegistryConfig){4, 8, 4});
        assert(r);
        const RcRegistryBufferSpan demanded = span(9, 0x2000, 100, 64, 1000);
        RcRegistryGpuCapture snapshot = {0};
        OK(rc_registry_gpu_capture_begin(r, &demanded.interval.range, 1, &snapshot));
        RcRegistryWrite writer = reserve(r, demanded, 3);
        OK(rc_registry_declare_through(r, at(8)));
        OK(rc_registry_buffer_write_record(r, writer, demanded, at(3)));
        RcRegistryBufferFragment f;
        uint32_t count = 1;
        if (uncertain) {
            OK(rc_registry_write_abandon(r, writer, 1));
            assert(rc_registry_gpu_capture_buffer_fragments(r, snapshot, &demanded, 1, at(6), &f, 1, &count) ==
                RC_REGISTRY_BLOCKED && !count);
        } else {
            OK(rc_registry_write_cancel(r, writer));
            query(r, snapshot, demanded, 6, &f, 1, 0);
        }
        OK(rc_registry_gpu_capture_end(r, &snapshot));
        rc_registry_destroy(r);
    }
    RcRegistry *r = rc_registry_create((RcRegistryConfig){4, 4, 4});
    assert(r);
    const RcRegistryBufferSpan demanded = span(9, 0x2000, 100, 64, 1000);
    RcRegistryGpuCapture snapshot = {0};
    OK(rc_registry_gpu_capture_begin(r, &demanded.interval.range, 1, &snapshot));
    for (unsigned i = 0; i < 5; ++i) {
        uint64_t ordinal = i * 2u + 3u;
        RcRegistryWrite writer = reserve(r, demanded, ordinal);
        OK(rc_registry_declare_through(r, at(ordinal + 1)));
        OK(rc_registry_buffer_write_record(r, writer, demanded, at(ordinal)));
        OK(rc_registry_handoff_through(r, at(ordinal)));
        OK(rc_registry_write_submit(r, writer, 1));
        OK(rc_registry_write_complete(r, writer, 1));
    }
    uint32_t count = 1;
    RcRegistryBufferFragment f;
    assert(rc_registry_gpu_capture_buffer_fragments(r, snapshot, &demanded, 1, at(12), &f, 1, &count) ==
        RC_REGISTRY_CAPACITY && !count);
    OK(rc_registry_gpu_capture_end(r, &snapshot));
    rc_registry_destroy(r);
}

static void projected_native_fragments(void) {
    for (int unknown_gap = 0; unknown_gap < 2; ++unknown_gap) {
        RcRegistry *r = rc_registry_create((RcRegistryConfig){16, 16, 4});
        assert(r);
        const RcRegistryBufferSpan demand = span(9, 0x2000, 100, 64, 1000);
        const RcRegistryBufferSpan actual[] = {span(7, 0x1004, 104, 4, 1004), span(7, 0x1020, 132, 4, 1032)};
        RcSourceRange affected[] = {actual[0].interval.range, actual[1].interval.range};
        if (unknown_gap) affected[0].size = 32;
        RcRegistryGpuCapture cap = {0};
        OK(rc_registry_gpu_capture_begin(r, &demand.interval.range, 1, &cap));
        RcRegistryWrite guard;
        OK(rc_registry_write_guard_begin_reason_at(r, RC_REGISTRY_BUFFER, at(3), RC_REGISTRY_GUARD_GPA_PROJECTION, &guard));
        OK(rc_registry_declare_through(r, at(6)));
        RcRegistryWriteCompletion completion = {guard, affected, NULL, unknown_gap ? 1 : 2, 0};
        assert(rc_registry_write_complete_gpa_projection(r, &completion, actual, 2) == RC_REGISTRY_STALE);
        OK(rc_registry_handoff_through(r, at(3)));
        OK(rc_registry_write_submit(r, guard, 1));
        const RcRegistryBufferSpan invalid[] = {actual[0], actual[0]};
        assert(rc_registry_write_complete_gpa_projection(r, &completion, invalid, 2) == RC_REGISTRY_INVALID);
        OK(rc_registry_write_complete_gpa_projection(r, &completion, actual, 2));
        RcRegistryBufferFragment f[2];
        if (!unknown_gap) {
            query(r, cap, demand, 6, f, 2, 2);
            expect(f[0], 0x2004, 4, 3);
            expect(f[1], 0x2020, 4, 3);
        } else {
            uint32_t count = 91;
            assert(rc_registry_gpu_capture_buffer_fragments(r, cap, &demand, 1, at(6), f, 2, &count) ==
                RC_REGISTRY_BLOCKED && !count);
            query(r, cap, span(9, 0x2020, 132, 4, 1032), 6, f, 2, 1);
            expect(f[0], 0x2020, 4, 3);
        }
        RcRegistryBufferSpan stale = span(9, 0x2020, 132, 4, 1032);
        ++stale.native.incarnation;
        uint32_t count = 91;
        assert(rc_registry_gpu_capture_buffer_fragments(r, cap, &stale, 1, at(6), f, 2, &count) ==
            RC_REGISTRY_BLOCKED && !count);
        OK(rc_registry_gpu_capture_end(r, &cap));
        rc_registry_destroy(r);
    }
}


static void preserved_result_sources(void) {
    for (unsigned mode = 0; mode < 7; mode++) {
        RcRegistry *r = rc_registry_create((RcRegistryConfig){16,32,4});
        assert(r);
        const RcRegistryBufferSpan output = span(9,0x3000,100,64,2000);
        RcRegistryGpuCapture capture = {0};
        OK(rc_registry_gpu_capture_begin(r,&output.interval.range,1,&capture));
        RcRegistryBufferSpan earlier = output;
        if (mode == 4) earlier.native.incarnation++;
        if (mode == 5) earlier.interval.range.size /= 2;
        RcRegistryWrite first = {0};
        if (mode >= 2 && mode <= 5) first = reserve(r,earlier,3);
        RcRegistryWrite actual = reserve(r,output,5);
        RcRegistryWrite later = reserve(r,output,9);
        OK(rc_registry_declare_through(r,at(10)));
        if (mode == 2) OK(rc_registry_buffer_access_record(r,first,earlier,at(3)));
        else if (mode >= 3 && mode <= 5) OK(rc_registry_buffer_write_record(r,first,earlier,at(3)));
        const int success = mode == 1 || mode == 3 || mode == 6;
        RcRegistryBufferFragment got;
        uint32_t count = 0;
        assert(rc_registry_gpu_capture_buffer_fragments(r,capture,&output,1,at(6),&got,1,&count)==RC_REGISTRY_BLOCKED);
        assert(!count);
        assert(rc_registry_buffer_preserved_record(r,actual,output,at(4),mode == 1 || mode == 2 || mode == 6) ==
               (success ? RC_REGISTRY_OK : RC_REGISTRY_BLOCKED));
        if (success) {
            query(r,capture,output,6,&got,1,1);
            expect(got,output.interval.address,output.interval.range.size,5);
            RcRegistryBufferSpan alias = output;
            alias.interval.address += 0x400;
            query(r,capture,alias,6,&got,1,1);
            expect(got,alias.interval.address,alias.interval.range.size,5);
            ++alias.native.allocation;
            assert(rc_registry_gpu_capture_buffer_fragments(r,capture,&alias,1,at(6),&got,1,&count)==RC_REGISTRY_BLOCKED);
            assert(!count);
        }
        if (mode == 6) {
            OK(rc_registry_write_cancel(r,actual));
            query(r,capture,output,6,&got,1,0);
        } else if (mode == 1) {
            OK(rc_registry_handoff_through(r,at(5)));
            OK(rc_registry_write_abandon(r,actual,1));
            assert(rc_registry_gpu_capture_buffer_fragments(r,capture,&output,1,at(6),&got,1,&count)==RC_REGISTRY_BLOCKED);
        }
        OK(rc_registry_write_cancel(r,later));
        OK(rc_registry_gpu_capture_end(r,&capture));
        rc_registry_destroy(r);
    }
}

int main(void) {
    preserved_result_sources();
    projected_native_fragments();
    prior_native_fragments();
    cancellation_and_retention();
    puts("native buffer producers: selected alias, affine offsets, actual latest slices, settled history, unknown and cancel passed");
}

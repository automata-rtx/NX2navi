#include <assert.h>
#include <stdio.h>
#define RC_REGISTRY_SINGLE_THREADED
#include "../runtime/resource_core_registry.c"

#define OK(x) assert((x) == RC_REGISTRY_OK)
#define IS(x, expected) assert((x) == (expected))

static RcRegistryResult read_capture(RcRegistry *r, RcRegistryGpuCapture capture, RcSourceRange range,
                                    uint64_t scene, uint64_t ordinal) {
    return rc_registry_gpu_capture_read_at(r, capture, &range, 1, (RcRegistryPosition){scene, ordinal});
}

static void hand(RcRegistry *r, RcRegistryWrite write, uint64_t scene, uint64_t ordinal) {
    OK(rc_registry_handoff_through(r, (RcRegistryPosition){scene, ordinal}));
    OK(rc_registry_write_submit(r, write, scene));
}

static void ordered_and_exact(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){8, 8, 3});
    assert(r);
    const RcSourceRange coverage[] = {{3,100,32}, {4,200,16}};
    const RcSourceRange write = {3,106,4}, neighbor = {3,120,4}, other = {4,204,4};
    RcRegistryGpuCapture older = {0}, later = {0}, spare = {0}, refused = {0};
    RcRegistryWrite first, canceled;
    OK(rc_registry_write_begin_at(r, write, RC_REGISTRY_BUFFER, (RcRegistryPosition){10,3}, &first));
    OK(rc_registry_write_begin_at(r, neighbor, RC_REGISTRY_BUFFER, (RcRegistryPosition){10,5}, &canceled));
    OK(rc_registry_gpu_capture_begin(r, coverage, 2, &older));
    OK(rc_registry_declare_through(r, (RcRegistryPosition){10,6}));
    IS(read_capture(r, older, write, 10, 4), RC_REGISTRY_BLOCKED);
    OK(read_capture(r, older, neighbor, 10, 4)); // Its output is later than the selected read.
    OK(read_capture(r, older, neighbor, 10, 5)); // The query includes only strictly earlier producers.
    OK(rc_registry_write_cancel(r, canceled));
    hand(r, first, 10, 3);
    IS(read_capture(r, older, write, 10, 4), RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_complete(r, first, 1));
    OK(rc_registry_read_range_at(r, write, (RcRegistryPosition){10,4})); // Current bytes are readable.
    IS(read_capture(r, older, write, 10, 4), RC_REGISTRY_STALE); // Earlier copied bytes are superseded.
    OK(read_capture(r, older, neighbor, 10, 4));
    OK(rc_registry_gpu_capture_begin(r, coverage, 2, &later));
    OK(read_capture(r, later, write, 10, 4));
    OK(rc_registry_cpu_changed(r, write));
    OK(read_capture(r, later, write, 10, 4)); // Later intentional CPU writes do not rewrite copied contents.
    IS(read_capture(r, later, (RcSourceRange){10,106,4}, 10, 4), RC_REGISTRY_INVALID); // A replacement backing.
    OK(rc_registry_gpu_capture_begin(r, coverage, 2, &spare));
    IS(rc_registry_gpu_capture_begin(r, coverage, 2, &refused), RC_REGISTRY_CAPACITY);
    assert(!refused.id && !refused.index);
    RcRegistryGpuCapture obsolete = later;
    OK(rc_registry_gpu_capture_end(r, &later));
    assert(!later.id && !later.index);
    OK(rc_registry_gpu_capture_begin(r, coverage, 2, &later));
    IS(read_capture(r, obsolete, write, 10, 4), RC_REGISTRY_STALE);
    OK(rc_registry_gpu_capture_end(r, &spare));
    OK(rc_registry_gpu_capture_end(r, &older));
    OK(rc_registry_gpu_capture_end(r, &later));

    // An earlier execution can settle before the copy while guest visibility still waits for readback.
    OK(rc_registry_write_begin_at(r, coverage[1], RC_REGISTRY_BUFFER, (RcRegistryPosition){11,3}, &first));
    OK(rc_registry_declare_through(r, (RcRegistryPosition){11,6}));
    hand(r, first, 11, 3);
    OK(rc_registry_write_complete(r, first, 0));
    OK(rc_registry_gpu_capture_begin(r, coverage, 2, &older));
    IS(read_capture(r, older, other, 11, 4), RC_REGISTRY_BLOCKED);
    OK(rc_registry_reconcile_cpu(r, other));
    IS(read_capture(r, older, other, 11, 4), RC_REGISTRY_STALE);
    IS(read_capture(r, older, (RcSourceRange){4,208,4}, 11, 4), RC_REGISTRY_BLOCKED);
    OK(rc_registry_gpu_capture_begin(r, coverage, 2, &later));
    OK(read_capture(r, later, other, 11, 4));
    OK(rc_registry_gpu_capture_end(r, &older));
    OK(rc_registry_gpu_capture_end(r, &later));

    // Precise actual coverage survives guard retirement. Unwritten bytes do not inherit the declaration.
    OK(rc_registry_write_guard_begin_reason_at(r, RC_REGISTRY_BUFFER, (RcRegistryPosition){12,3},
                                               RC_REGISTRY_GUARD_GPA_PROJECTION, &first));
    OK(rc_registry_declare_through(r, (RcRegistryPosition){12,6}));
    OK(rc_registry_gpu_capture_begin(r, coverage, 2, &older));
    IS(read_capture(r, older, neighbor, 12, 4), RC_REGISTRY_BLOCKED);
    hand(r, first, 12, 3);
    OK(rc_registry_write_guard_accumulate(r, first));
    IS(read_capture(r, older, neighbor, 12, 4), RC_REGISTRY_BLOCKED);
    RcRegistryWriteCompletion actual = {rc_registry_unresolved_guard(r), &write, NULL, 1, 0};
    OK(rc_registry_write_complete_extents(r, &actual, 1));
    IS(read_capture(r, older, write, 12, 4), RC_REGISTRY_STALE);
    OK(read_capture(r, older, neighbor, 12, 4));
    OK(rc_registry_gpu_capture_end(r, &older));
    rc_registry_destroy(r);
}

static void readback_and_overflow(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){2, 6, 2});
    assert(r);
    const RcSourceRange a = {3,100,16}, b = {4,200,16};
    RcRegistryGpuCapture untouched = {0}, affected = {0};
    OK(rc_registry_gpu_capture_begin(r, &a, 1, &untouched));
    OK(rc_registry_gpu_capture_begin(r, &b, 1, &affected));
    for (uint64_t scene = 1; scene <= 5; scene++) {
        RcRegistryWrite write;
        OK(rc_registry_write_begin_at(r, b, RC_REGISTRY_BUFFER, (RcRegistryPosition){scene,3}, &write));
        OK(rc_registry_declare_through(r, (RcRegistryPosition){scene,4}));
        hand(r, write, scene, 3);
        OK(rc_registry_write_complete(r, write, 1));
    }
    OK(read_capture(r, untouched, a, 5, 4)); // Unrelated cutoff unaffected by ring turnover.
    IS(read_capture(r, affected, b, 5, 4), RC_REGISTRY_CAPACITY);
    OK(rc_registry_gpu_capture_end(r, &affected));
    OK(rc_registry_gpu_capture_begin(r, &b, 1, &affected));
    OK(read_capture(r, affected, b, 5, 4)); // A new ordered observation can use the now-visible bytes.

    RcRegistryImageWriteExtent image = {.interval = {1000,77,a}};
    OK(rc_registry_image_hold_begin_extents(r, 51, &image.interval, 1, &image.hold));
    IS(read_capture(r, untouched, a, 5, 4), RC_REGISTRY_BLOCKED);
    const RcSourceRange visible = {RC_GUEST_COMPATIBILITY_BACKING,1002,4};
    OK(rc_registry_image_hold_readback(r, 51, &image, 1, &visible, 1));
    IS(read_capture(r, untouched, (RcSourceRange){3,102,4}, 5, 4), RC_REGISTRY_STALE);
    IS(read_capture(r, untouched, (RcSourceRange){3,107,4}, 5, 4), RC_REGISTRY_BLOCKED);
    OK(rc_registry_gpu_capture_end(r, &untouched));
    OK(rc_registry_gpu_capture_end(r, &affected));
    assert(!r->active_gpu_captures && !r->gpu_visible_sequence);
    rc_registry_destroy(r);
}

static void reserve_then_observe(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){8, 8, 3});
    assert(r);
    const RcSourceRange coverage = {3,100,32};
    const RcRegistryPosition read = {1,4};
    RcRegistryGpuCapture queued[2] = {0}, foreground = {0}, refused = {0};
    uint64_t epoch, later_epoch;
    for (unsigned i = 0; i < 2; ++i)
        OK(rc_registry_gpu_capture_reserve(r, &coverage, 1, 1, &queued[i], NULL));
    assert(!r->active_gpu_captures);
    IS(rc_registry_gpu_capture_reserve(r, &coverage, 1, 1, &refused, &epoch), RC_REGISTRY_BUSY);
    assert(!refused.id);
    OK(rc_registry_gpu_capture_begin(r, &coverage, 1, &foreground));
    IS(rc_registry_gpu_capture_reserve(r, &coverage, 1, 0, &refused, &later_epoch), RC_REGISTRY_BUSY);
    assert(later_epoch == epoch);
    RcRegistryWrite write;
    OK(rc_registry_write_begin_at(r, coverage, RC_REGISTRY_BUFFER, (RcRegistryPosition){1,3}, &write));
    OK(rc_registry_declare_through(r, read));
    RcRegistryGpuReadEvidence evidence = RC_GPU_CAPTURE_NO_EVIDENCE;
    IS(rc_registry_gpu_capture_read_evidence(r, foreground, &coverage, 1, read, &evidence), RC_REGISTRY_BLOCKED);
    assert(evidence == RC_GPU_CAPTURE_EARLIER_BUFFER);
    IS(rc_registry_gpu_capture_read_at(r, queued[0], &coverage, 1, read), RC_REGISTRY_INVALID);
    hand(r, write, 1, 3);
    OK(rc_registry_write_complete(r, write, 1));
    IS(rc_registry_gpu_capture_read_evidence(r, foreground, &coverage, 1, read, &evidence), RC_REGISTRY_STALE);
    assert(evidence == RC_GPU_CAPTURE_EARLIER_VISIBLE);
    OK(rc_registry_gpu_capture_start(r, queued[0]));
    OK(rc_registry_gpu_capture_read_at(r, queued[0], &coverage, 1, read));
    IS(rc_registry_gpu_capture_start(r, queued[0]), RC_REGISTRY_INVALID);
    RcRegistryGpuCapture canceled = queued[1];
    OK(rc_registry_gpu_capture_end(r, &queued[1]));
    IS(rc_registry_gpu_capture_start(r, canceled), RC_REGISTRY_STALE);
    assert(r->gpu_capture_release_epoch != epoch);
    OK(rc_registry_gpu_capture_reserve(r, &coverage, 1, 1, &queued[1], NULL));
    assert(queued[1].id != canceled.id && queued[1].index == canceled.index);
    rc_registry_fail_closed(r);
    IS(rc_registry_gpu_capture_start(r, queued[1]), RC_REGISTRY_BLOCKED);
    OK(rc_registry_gpu_capture_end(r, &queued[1]));
    OK(rc_registry_gpu_capture_end(r, &queued[0]));
    OK(rc_registry_gpu_capture_end(r, &foreground));
    assert(!r->active_gpu_captures && !r->gpu_visible_sequence);
    rc_registry_destroy(r);
}

static void queued_capacity_leaves_recorder_room(void) {
    enum { foreground_slots = 18, queued_slots = 2 };
    RcRegistry *r = rc_registry_create((RcRegistryConfig){8, 8, foreground_slots + queued_slots});
    assert(r);
    const RcSourceRange coverage = {3,100,32};
    RcRegistryGpuCapture queued[queued_slots] = {0}, foreground[foreground_slots] = {0}, refused = {0};
    for (unsigned i = 0; i < queued_slots; ++i)
        OK(rc_registry_gpu_capture_reserve(r, &coverage, 1, foreground_slots, &queued[i], NULL));
    uint64_t epoch;
    IS(rc_registry_gpu_capture_reserve(r, &coverage, 1, foreground_slots, &refused, &epoch), RC_REGISTRY_BUSY);
    for (unsigned i = 0; i < foreground_slots; ++i)
        OK(rc_registry_gpu_capture_begin(r, &coverage, 1, &foreground[i]));
    IS(rc_registry_gpu_capture_reserve(r, &coverage, 1, 0, &refused, NULL), RC_REGISTRY_BUSY);
    for (unsigned i = 0; i < foreground_slots; ++i)
        OK(rc_registry_gpu_capture_end(r, &foreground[i]));
    uint64_t later;
    IS(rc_registry_gpu_capture_reserve(r, &coverage, 1, foreground_slots, &refused, &later), RC_REGISTRY_BUSY);
    assert(later != epoch);
    for (unsigned i = 0; i < queued_slots; ++i)
        OK(rc_registry_gpu_capture_end(r, &queued[i]));
    rc_registry_destroy(r);
}

static RcRegistryResult read_inline(RcRegistry *r, RcInterval span, uint64_t output, uint64_t read) {
    return rc_registry_read_produced_at(r, &span, 1, (RcRegistryPosition){1,output}, (RcRegistryPosition){1,read});
}

static void produced_history(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){12,12,1});
    assert(r);
    const RcSourceRange a = {3,100,16}, disjoint = {3,300,16}, alias = {3,104,8};
    const RcInterval read = {0x3000,9,alias};
    RcRegistryGpuCapture captured = {0};
    OK(rc_registry_gpu_capture_begin(r, &a, 1, &captured));
    OK(rc_registry_produced_scene_begin(r,1)); // One real recorder shares the journal, even if all snapshot slots are occupied.
    RcRegistryWrite earlier, output, other, canceled, later, future, untimed;
    OK(rc_registry_write_begin_at(r,a,RC_REGISTRY_BUFFER,(RcRegistryPosition){1,1},&earlier));
    OK(rc_registry_write_begin_at(r,a,RC_REGISTRY_BUFFER,(RcRegistryPosition){1,3},&output));
    OK(rc_registry_write_begin_at(r,alias,RC_REGISTRY_BUFFER,(RcRegistryPosition){1,5},&canceled));
    OK(rc_registry_write_begin_at(r,disjoint,RC_REGISTRY_BUFFER,(RcRegistryPosition){1,7},&other));
    OK(rc_registry_write_begin_at(r,alias,RC_REGISTRY_BUFFER,(RcRegistryPosition){1,9},&later));
    OK(rc_registry_write_begin_at(r,alias,RC_REGISTRY_BUFFER,(RcRegistryPosition){1,13},&future));
    OK(rc_registry_declare_through(r,(RcRegistryPosition){1,14}));
    IS(read_inline(r,read,3,6), RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_cancel(r,canceled));
    OK(read_inline(r,read,3,6)); // Includes actual previous overwrite and excludes future positioned writers.
    hand(r,earlier,1,1); OK(rc_registry_write_complete(r,earlier,1));
    hand(r,output,1,3); OK(rc_registry_write_complete(r,output,1));
    OK(read_inline(r,read,3,6));
    hand(r,other,1,7); OK(rc_registry_write_complete(r,other,1));
    OK(read_inline(r,read,3,8)); // A disjoint settled write leaves the original selected bytes valid.
    IS(read_inline(r,read,3,10), RC_REGISTRY_BLOCKED);
    hand(r,later,1,9); OK(rc_registry_write_complete(r,later,1));
    IS(read_inline(r,read,3,10), RC_REGISTRY_STALE); // Actual completed bytes remain known after ticket retirement.
    OK(read_inline(r,read,9,10));
    OK(rc_registry_hold_begin(r,alias,RC_REGISTRY_UNKNOWN,&untimed));
    IS(read_inline(r,read,9,10), RC_REGISTRY_BLOCKED);
    OK(rc_registry_hold_release(r,untimed,1));
    IS(read_inline(r,read,9,10), RC_REGISTRY_STALE); // An unpositioned settlement cannot claim to precede the write.
    OK(rc_registry_write_cancel(r,future));
    OK(rc_registry_produced_scene_end(r,1));
    IS(read_inline(r,read,9,10), RC_REGISTRY_STALE);
    assert(r->gpu_visible_sequence); // The unrelated copied snapshot still owns its context.
    OK(rc_registry_gpu_capture_end(r,&captured));
    assert(!r->gpu_visible_sequence);
    rc_registry_destroy(r);

    r = rc_registry_create((RcRegistryConfig){2,8,1});
    assert(r);
    OK(rc_registry_produced_scene_begin(r,1));
    // Merged foreign ranges preserve both producer bounds before later readback removes the authority tree entries.
    OK(rc_registry_write_begin_at(r,a,RC_REGISTRY_BUFFER,(RcRegistryPosition){1,1},&earlier));
    OK(rc_registry_write_begin_at(r,alias,RC_REGISTRY_BUFFER,(RcRegistryPosition){1,5},&later));
    OK(rc_registry_declare_through(r,(RcRegistryPosition){1,6}));
    hand(r,earlier,1,1); OK(rc_registry_write_complete(r,earlier,0));
    hand(r,later,1,5); OK(rc_registry_write_complete(r,later,0));
    assert(rc_registry_stats(r).writer_foreign==1);
    OK(rc_registry_reconcile_cpu(r,alias));
    IS(read_inline(r,read,3,6), RC_REGISTRY_STALE);
    OK(read_inline(r,read,5,6));
    for (uint64_t p=7;p<=13;p+=2) {
        OK(rc_registry_write_begin_at(r,disjoint,RC_REGISTRY_BUFFER,(RcRegistryPosition){1,p},&other));
        OK(rc_registry_declare_through(r,(RcRegistryPosition){1,p+1}));
        hand(r,other,1,p); OK(rc_registry_write_complete(r,other,1));
    }
    IS(read_inline(r,read,3,14), RC_REGISTRY_CAPACITY);
    OK(read_inline(r,read,13,14)); // An actual later overwrite supplies a new usable anchor.
    OK(rc_registry_produced_scene_end(r,1));
    rc_registry_destroy(r);
}

static void ordered_device_buffer(void) {
    RcRegistry *r = rc_registry_create((RcRegistryConfig){12,12,1});
    assert(r);
    RcInterval native = {0x8000002000,51,{81,128,8}}, alias = {0x8000101000,66,{81,128,8}};
    RcInterval replaced = {0x8000002000,52,{82,128,8}};
    RcRegistryWrite output, future, opaque, image;
    OK(rc_registry_write_begin_at(r,alias.range,RC_REGISTRY_BUFFER,(RcRegistryPosition){8,3},&output));
    OK(rc_registry_write_begin_at(r,native.range,RC_REGISTRY_BUFFER,(RcRegistryPosition){8,13},&future));
    OK(rc_registry_write_guard_begin_reason_at(r,RC_REGISTRY_BUFFER,(RcRegistryPosition){8,7},
                                                RC_REGISTRY_GUARD_GPA_PROJECTION,&opaque));
    OK(rc_registry_write_begin_at(r,native.range,RC_REGISTRY_IMAGE,(RcRegistryPosition){8,9},&image));
    OK(rc_registry_declare_through(r,(RcRegistryPosition){8,14}));
    RcRegistryPosition read = {8,6};
    IS(rc_registry_read_range_at(r,native.range,read),RC_REGISTRY_BLOCKED);
    OK(rc_registry_read_device_buffer_at(r,&native,1,read));
    hand(r,output,8,3); OK(rc_registry_write_complete(r,output,0));
    IS(rc_registry_read_range_at(r,native.range,read),RC_REGISTRY_BLOCKED);
    OK(rc_registry_read_device_buffer_at(r,&native,1,read));
    read.ordinal=8;
    IS(rc_registry_read_device_buffer_at(r,&native,1,read),RC_REGISTRY_BLOCKED);
    OK(rc_registry_write_cancel(r,opaque));
    OK(rc_registry_read_device_buffer_at(r,&native,1,read));
    read.ordinal=10;
    IS(rc_registry_read_device_buffer_at(r,&native,1,read),RC_REGISTRY_BLOCKED);
    OK(rc_registry_read_device_buffer_at(r,&replaced,1,read)); // A different native backing, same guest address.
    OK(rc_registry_write_cancel(r,image));
    OK(rc_registry_read_device_buffer_at(r,&native,1,read));
    OK(rc_registry_write_cancel(r,future));
    rc_registry_destroy(r);
}

int main(void) {
    queued_capacity_leaves_recorder_room();
    reserve_then_observe();
    ordered_and_exact();
    readback_and_overflow();
    produced_history();
    ordered_device_buffer();
    puts("GPU capture cutoff: ordered native writers, visibility after completion, exact alias ranges and bounded retirement passed");
    return 0;
}

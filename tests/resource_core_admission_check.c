#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "../runtime/resource_core_admission.h"

#define OK(call) assert((call) == RC_ADMISSION_OK)
#define IS(call, expected) assert((call) == (expected))

enum { RAM, COMMIT, GPU0, GPU1, STAGING, DOMAINS };

static RcAdmissionVector amounts(uint64_t ram, uint64_t commit, uint64_t gpu0, uint64_t gpu1, uint64_t staging) {
    return (RcAdmissionVector){{ram, commit, gpu0, gpu1, staging}};
}

static void init(RcAdmission *b, RcAdmissionRecord *records, uint32_t count, uint64_t limit) {
    RcAdmissionDomain domains[] = {
        {RC_ADMISSION_RAM, 0, limit}, {RC_ADMISSION_COMMIT, 0, limit},
        {RC_ADMISSION_GPU_HEAP, 0, limit}, {RC_ADMISSION_GPU_HEAP, 1, limit},
        {RC_ADMISSION_STAGING, 0, limit}
    };
    OK(rc_admission_init(b, 1, domains, DOMAINS, records, count));
}

static uint64_t available(const RcAdmission *b, uint32_t domain) {
    uint64_t value = UINT64_MAX;
    OK(rc_admission_available(b, domain, &value));
    return value;
}

static void unchanged(const RcAdmission *b, const RcAdmission *before, const RcAdmissionRecord *records,
                      const RcAdmissionRecord *old_records) {
    assert(!memcmp(b, before, sizeof *b));
    assert(!memcmp(records, old_records, (size_t)b->record_count * sizeof *records));
}

static void all_or_none_and_capacity_domains(void) {
    RcAdmission b;
    RcAdmissionRecord records[8], old_records[8];
    init(&b, records, 8, 100);
    OK(rc_admission_set_limit(&b, GPU1, 1000));
    OK(rc_admission_set_limit(&b, STAGING, 64));
    for (unsigned limited = 0; limited < DOMAINS; limited++) {
        RcAdmissionVector need = amounts(10, 10, 10, 10, 10);
        need.amount[limited] = b.domain[limited].limit + 1;
        RcAdmission before = b;
        memcpy(old_records, records, sizeof records);
        RcAdmissionToken out = {0}, original = out;
        IS(rc_admission_reserve(&b, 1, &need, &out), RC_ADMISSION_REFUSED);
        unchanged(&b, &before, records, old_records);
        assert(!memcmp(&out, &original, sizeof out));
    }
    // Another heap's free VRAM does not satisfy the selected eligible heap.
    RcAdmissionVector wrong_heap = amounts(0, 0, 101, 0, 0);
    RcAdmissionToken out;
    IS(rc_admission_reserve(&b, 1, &wrong_heap, &out), RC_ADMISSION_REFUSED);
    assert(available(&b, GPU1) == 1000);

    // One staging allocation consumes its RAM/commit once and constrains arena capacity independently.
    RcAdmissionVector chunk = amounts(16, 16, 0, 0, 16);
    RcAdmissionToken chunks[4];
    for (unsigned i = 0; i < 4; i++) OK(rc_admission_reserve(&b, i + 1, &chunk, &chunks[i]));
    assert(b.reserved.amount[RAM] == 64 && b.reserved.amount[COMMIT] == 64);
    assert(available(&b, RAM) == 36 && available(&b, STAGING) == 0);
    IS(rc_admission_reserve(&b, 5, &chunk, &out), RC_ADMISSION_REFUSED);
    for (unsigned i = 0; i < 4; i++) OK(rc_admission_cancel(&b, chunks[i], i + 1));
    assert(available(&b, RAM) == 100 && available(&b, STAGING) == 64);
}

static void interleaved_producers_spend_capacity_once(void) {
    // Every ordering of these producers sees the same shared allowance. A finished compiler remains charged
    // while a texture is preparing, so rapid completion cannot repeatedly spend unchanged sampled headroom.
    static const unsigned order[6][3] = {{0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0}};
    for (unsigned run = 0; run < 6; run++) {
        RcAdmission b;
        RcAdmissionRecord records[8];
        init(&b, records, 8, 100);
        RcAdmissionVector peak = amounts(40, 40, 0, 0, 0);
        RcAdmissionVector actual = amounts(35, 35, 0, 0, 0);
        RcAdmissionToken ticket[3] = {{0}}, charge[3] = {{0}}, pending[3] = {{0}};
        unsigned a = order[run][0], c = order[run][1], refused = order[run][2];
        OK(rc_admission_reserve(&b, a + 1, &peak, &ticket[a]));
        OK(rc_admission_reserve(&b, c + 1, &peak, &ticket[c]));
        IS(rc_admission_reserve(&b, refused + 1, &peak, &ticket[refused]), RC_ADMISSION_REFUSED);
        OK(rc_admission_materialize(&b, ticket[a], a + 1, &actual, &charge[a]));
        assert(b.reserved.amount[RAM] == 40 && b.retained.amount[RAM] == 35);
        IS(rc_admission_reserve(&b, refused + 1, &peak, &ticket[refused]), RC_ADMISSION_REFUSED);
        OK(rc_admission_retire(&b, charge[a], a + 1, &pending[a]));
        assert(b.pending.amount[RAM] == 35 && available(&b, RAM) == 25);
        IS(rc_admission_reserve(&b, refused + 1, &peak, &ticket[refused]), RC_ADMISSION_REFUSED);
        OK(rc_admission_destroyed(&b, pending[a], a + 1));
        OK(rc_admission_reserve(&b, refused + 1, &peak, &ticket[refused]));
        IS(rc_admission_destroyed(&b, pending[a], a + 1), RC_ADMISSION_STALE);
        assert(b.reserved.amount[RAM] == 80);
        OK(rc_admission_cancel(&b, ticket[c], c + 1));
        OK(rc_admission_cancel(&b, ticket[refused], refused + 1));
        assert(available(&b, RAM) == 100);
    }
}

static void drain_handoff_survives_pressure(void) {
    RcAdmission b;
    RcAdmissionRecord records[8];
    init(&b, records, 8, 100);
    RcAdmissionVector peak = amounts(100, 100, 80, 0, 40);
    RcAdmissionVector source = amounts(40, 40, 0, 0, 0);
    RcAdmissionVector drain = amounts(60, 60, 80, 0, 40);
    RcAdmissionVector upload_actual = amounts(30, 30, 80, 0, 40);
    RcAdmissionToken whole, source_ticket, source_charge, drain_ticket, drain_owner, upload, pending;
    OK(rc_admission_reserve(&b, 10, &peak, &whole));
    OK(rc_admission_split(&b, whole, 10, &source, 11, &source_ticket));
    OK(rc_admission_materialize(&b, source_ticket, 11, &source, &source_charge));
    assert(b.retained.amount[RAM] == 40 && b.reserved.amount[RAM] == 60);
    OK(rc_admission_set_limit(&b, RAM, 70));
    OK(rc_admission_set_limit(&b, GPU0, 50));
    RcAdmissionVector competitor = amounts(1, 1, 1, 0, 0);
    RcAdmissionToken refused;
    IS(rc_admission_reserve(&b, 99, &competitor, &refused), RC_ADMISSION_REFUSED);

    // Work already admitted can hand its guarantee to the draining stage without reacquiring shrunken room.
    OK(rc_admission_split(&b, whole, 10, &drain, 12, &drain_ticket));
    OK(rc_admission_handoff(&b, drain_ticket, 12, 13, &drain_owner));
    IS(rc_admission_cancel(&b, drain_ticket, 12), RC_ADMISSION_STALE);
    IS(rc_admission_materialize(&b, drain_owner, 13, &upload_actual, &upload), RC_ADMISSION_ACCOUNTED_PRESSURE);
    OK(rc_admission_cancel(&b, whole, 10));
    assert(b.reserved.amount[RAM] == 0 && b.retained.amount[RAM] == 70);
    OK(rc_admission_retire(&b, source_charge, 11, &pending));
    assert(available(&b, RAM) == 0);
    OK(rc_admission_destroyed(&b, pending, 11));
    assert(available(&b, RAM) == 40 && available(&b, GPU0) == 0);
    IS(rc_admission_reserve(&b, 99, &competitor, &refused), RC_ADMISSION_REFUSED);

    // Pressure in one heap does not prevent an independent RAM-only drain operation from progressing.
    RcAdmissionVector host_only = amounts(20, 20, 0, 0, 0);
    OK(rc_admission_reserve(&b, 99, &host_only, &refused));
    OK(rc_admission_cancel(&b, refused, 99));
    OK(rc_admission_retire(&b, upload, 13, &pending));
    assert(b.pending.amount[GPU0] == 80 && available(&b, GPU0) == 0);
    OK(rc_admission_destroyed(&b, pending, 13));
    assert(available(&b, GPU0) == 50);
}

static void baseline_and_actual_measurement(void) {
    RcAdmission b;
    RcAdmissionRecord records[8];
    init(&b, records, 8, 100);
    RcAdmissionVector baseline = amounts(120, 120, 120, 0, 0);
    RcAdmissionToken charge, pending, ticket;
    IS(rc_admission_adopt(&b, 1, &baseline, &charge), RC_ADMISSION_ACCOUNTED_PRESSURE);
    assert(b.retained.amount[RAM] == 120 && available(&b, RAM) == 0);
    RcAdmissionVector small = amounts(10, 10, 10, 0, 0);
    IS(rc_admission_reserve(&b, 2, &small, &ticket), RC_ADMISSION_REFUSED);
    OK(rc_admission_retire(&b, charge, 1, &pending));
    assert(b.pending.amount[RAM] == 120);
    IS(rc_admission_remeasure(&b, pending, 1, &small), RC_ADMISSION_WRONG_STATE);
    IS(rc_admission_reserve(&b, 2, &small, &ticket), RC_ADMISSION_REFUSED);
    // External pressure can keep a domain unavailable even after actual suballocation destruction.
    OK(rc_admission_set_limit(&b, GPU0, 0));
    OK(rc_admission_destroyed(&b, pending, 1));
    assert(available(&b, RAM) == 100 && available(&b, GPU0) == 0);
    IS(rc_admission_reserve(&b, 2, &small, &ticket), RC_ADMISSION_REFUSED);
    OK(rc_admission_set_limit(&b, GPU0, 100));
    OK(rc_admission_reserve(&b, 2, &small, &ticket));
    // Underestimation is accounted explicitly instead of falsely refusing to acknowledge existing memory.
    IS(rc_admission_materialize(&b, ticket, 2, &baseline, &charge), RC_ADMISSION_ACCOUNTED_PRESSURE);
    assert(b.reserved.amount[RAM] == 0 && b.retained.amount[RAM] == 120);
    RcAdmissionVector measured = amounts(80, 80, 80, 0, 0);
    OK(rc_admission_remeasure(&b, charge, 2, &measured));
    assert(available(&b, RAM) == 20);
    OK(rc_admission_retire(&b, charge, 2, &pending));
    OK(rc_admission_destroyed(&b, pending, 2));
}

static void ownership_stale_handles_and_recycling(void) {
    RcAdmission b, other;
    RcAdmissionRecord records[2], other_records[2];
    init(&b, records, 2, 100);
    init(&other, other_records, 2, 100);
    RcAdmissionVector amount = amounts(10, 10, 10, 0, 0);
    RcAdmissionToken original, moved, charge, pending, replacement;
    OK(rc_admission_reserve(&b, 1, &amount, &original));
    IS(rc_admission_cancel(&b, original, 2), RC_ADMISSION_NOT_OWNER);
    IS(rc_admission_cancel(&other, original, 1), RC_ADMISSION_STALE);
    OK(rc_admission_handoff(&b, original, 1, 2, &moved));
    IS(rc_admission_cancel(&b, original, 1), RC_ADMISSION_STALE);
    IS(rc_admission_materialize(&b, moved, 1, &amount, &charge), RC_ADMISSION_NOT_OWNER);
    OK(rc_admission_materialize(&b, moved, 2, &amount, &charge));
    IS(rc_admission_materialize(&b, moved, 2, &amount, &replacement), RC_ADMISSION_STALE);
    IS(rc_admission_cancel(&b, moved, 2), RC_ADMISSION_STALE);
    IS(rc_admission_cancel(&b, charge, 2), RC_ADMISSION_WRONG_STATE);
    IS(rc_admission_destroyed(&b, charge, 2), RC_ADMISSION_WRONG_STATE);
    OK(rc_admission_retire(&b, charge, 2, &pending));
    IS(rc_admission_retire(&b, charge, 2, &replacement), RC_ADMISSION_STALE);
    IS(rc_admission_destroyed(&b, pending, 1), RC_ADMISSION_NOT_OWNER);
    OK(rc_admission_destroyed(&b, pending, 2));
    OK(rc_admission_reserve(&b, 2, &amount, &replacement));
    assert(replacement.slot == pending.slot && replacement.generation != pending.generation);
    IS(rc_admission_destroyed(&b, pending, 2), RC_ADMISSION_STALE);
    assert(b.reserved.amount[RAM] == 10);
    OK(rc_admission_cancel(&b, replacement, 2));
    IS(rc_admission_cancel(&b, replacement, 2), RC_ADMISSION_STALE);

    // A ledger rebuilt at the same address also invalidates handles, even if its first slot serial is reused.
    RcAdmissionDomain specs[DOMAINS];
    memcpy(specs, b.domain, sizeof specs);
    OK(rc_admission_init(&b, 2, specs, DOMAINS, records, 2));
    OK(rc_admission_reserve(&b, 1, &amount, &replacement));
    IS(rc_admission_cancel(&b, original, 1), RC_ADMISSION_STALE);
    OK(rc_admission_cancel(&b, replacement, 1));
}

static void record_exhaustion_preserves_live_obligations(void) {
    RcAdmission b, before;
    RcAdmissionRecord records[2], old_records[2];
    init(&b, records, 2, 100);
    RcAdmissionVector amount = amounts(20, 20, 0, 0, 0), half = amounts(10, 10, 0, 0, 0);
    RcAdmissionToken a, c, charge, pending, out = {0};
    OK(rc_admission_reserve(&b, 1, &amount, &a));
    OK(rc_admission_reserve(&b, 2, &amount, &c));
    before = b;
    memcpy(old_records, records, sizeof records);
    IS(rc_admission_reserve(&b, 3, &amount, &out), RC_ADMISSION_NO_RECORDS);
    unchanged(&b, &before, records, old_records);
    IS(rc_admission_split(&b, a, 1, &half, 3, &out), RC_ADMISSION_NO_RECORDS);
    unchanged(&b, &before, records, old_records);
    // Settlement and destruction need no additional bookkeeping slot.
    OK(rc_admission_materialize(&b, a, 1, &amount, &charge));
    OK(rc_admission_retire(&b, charge, 1, &pending));
    OK(rc_admission_destroyed(&b, pending, 1));
    OK(rc_admission_reserve(&b, 3, &amount, &out));
    IS(rc_admission_cancel(&b, a, 1), RC_ADMISSION_STALE);
    OK(rc_admission_cancel(&b, c, 2));
    OK(rc_admission_cancel(&b, out, 3));
}

static void arithmetic_and_domain_overflow_are_atomic(void) {
    RcAdmission b, before;
    RcAdmissionRecord records[4], old_records[4];
    init(&b, records, 4, UINT64_MAX);
    RcAdmissionVector large = amounts(UINT64_MAX - 1, 0, 0, 0, 0);
    RcAdmissionVector one = amounts(1, 0, 0, 0, 0), two = amounts(2, 0, 0, 0, 0);
    RcAdmissionToken base, ticket, charge, out = {0};
    OK(rc_admission_adopt(&b, 1, &large, &base));
    OK(rc_admission_reserve(&b, 2, &one, &ticket));
    assert(available(&b, RAM) == 0);
    before = b;
    memcpy(old_records, records, sizeof records);
    IS(rc_admission_adopt(&b, 3, &one, &out), RC_ADMISSION_OVERFLOW);
    unchanged(&b, &before, records, old_records);
    IS(rc_admission_materialize(&b, ticket, 2, &two, &out), RC_ADMISSION_OVERFLOW);
    unchanged(&b, &before, records, old_records);
    large.amount[RAM] = UINT64_MAX;
    IS(rc_admission_remeasure(&b, base, 1, &large), RC_ADMISSION_OVERFLOW);
    unchanged(&b, &before, records, old_records);
    OK(rc_admission_materialize(&b, ticket, 2, &one, &charge));
    RcAdmissionVector unknown = one;
    unknown.amount[DOMAINS] = 1;
    before = b;
    memcpy(old_records, records, sizeof records);
    IS(rc_admission_reserve(&b, 3, &unknown, &out), RC_ADMISSION_UNSUPPORTED);
    unchanged(&b, &before, records, old_records);
    uint64_t untouched = 123;
    IS(rc_admission_available(&b, DOMAINS, &untouched), RC_ADMISSION_UNSUPPORTED);
    assert(untouched == 123);
    IS(rc_admission_set_limit(&b, DOMAINS, 5), RC_ADMISSION_UNSUPPORTED);
    unchanged(&b, &before, records, old_records);
    RcAdmissionToken pending;
    OK(rc_admission_retire(&b, charge, 2, &pending));
    OK(rc_admission_destroyed(&b, pending, 2));
    OK(rc_admission_retire(&b, base, 1, &pending));
    OK(rc_admission_destroyed(&b, pending, 1));

    RcAdmissionDomain specs[RC_ADMISSION_MAX_DOMAINS + 1] = {{RC_ADMISSION_RAM, 0, 100}};
    before = b;
    memcpy(old_records, records, sizeof records);
    IS(rc_admission_init(&b, 2, specs, RC_ADMISSION_MAX_DOMAINS + 1, records, 4), RC_ADMISSION_UNSUPPORTED);
    unchanged(&b, &before, records, old_records);
    specs[1] = specs[0];
    IS(rc_admission_init(&b, 2, specs, 2, records, 4), RC_ADMISSION_INVALID);
    unchanged(&b, &before, records, old_records);
}

static void generation_exhaustion_keeps_cleanup_possible(void) {
    RcAdmission b, before;
    RcAdmissionRecord records[3], old_records[3];
    init(&b, records, 3, 100);
    RcAdmissionVector amount = amounts(20, 20, 0, 0, 0), half = amounts(10, 10, 0, 0, 0);
    RcAdmissionToken ticket, charge, pending, out = {0};
    b.next_generation = UINT64_MAX - 1;
    OK(rc_admission_reserve(&b, 1, &amount, &ticket));
    assert(ticket.generation == UINT64_MAX);
    before = b;
    memcpy(old_records, records, sizeof records);
    IS(rc_admission_reserve(&b, 2, &amount, &out), RC_ADMISSION_GENERATION_EXHAUSTED);
    unchanged(&b, &before, records, old_records);
    IS(rc_admission_split(&b, ticket, 1, &half, 2, &out), RC_ADMISSION_GENERATION_EXHAUSTED);
    unchanged(&b, &before, records, old_records);
    IS(rc_admission_handoff(&b, ticket, 1, 2, &out), RC_ADMISSION_GENERATION_EXHAUSTED);
    unchanged(&b, &before, records, old_records);
    OK(rc_admission_materialize(&b, ticket, 1, &amount, &charge));
    IS(rc_admission_cancel(&b, ticket, 1), RC_ADMISSION_STALE);
    OK(rc_admission_retire(&b, charge, 1, &pending));
    OK(rc_admission_destroyed(&b, pending, 1));
    IS(rc_admission_reserve(&b, 2, &amount, &out), RC_ADMISSION_GENERATION_EXHAUSTED);
    assert(available(&b, RAM) == 100);
}

int main(void) {
    all_or_none_and_capacity_domains();
    interleaved_producers_spend_capacity_once();
    drain_handoff_survives_pressure();
    baseline_and_actual_measurement();
    ownership_stale_handles_and_recycling();
    record_exhaustion_preserves_live_obligations();
    arithmetic_and_domain_overflow_are_atomic();
    generation_exhaustion_keeps_cleanup_possible();
    puts("resource_core_admission_check: passed");
    return 0;
}

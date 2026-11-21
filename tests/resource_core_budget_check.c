#include <assert.h>
#include <stdio.h>
#include <string.h>
#define RC_BUDGET_TEST
#include "../runtime/resource_core_budget.h"

#define MB (1024ull * 1024ull)
#define OK(call) assert((call) == RC_ADMISSION_OK)
#define IS(call, result) assert((call) == (result))

static RcBudgetSample observed;
static uint64_t clock_ns;
static unsigned lock_depth, sample_count;
uint64_t rc_budget_test_now_ns(void) { return clock_ns; }
void rc_budget_test_lock(void) { assert(!lock_depth); lock_depth++; }
void rc_budget_test_unlock(void) { assert(lock_depth == 1); lock_depth--; }
void rc_budget_test_sample(RcBudgetSample *out) { assert(lock_depth == 1); *out = observed; sample_count++; }
void rc_budget_test_wait(uint32_t milliseconds) { assert(!lock_depth); clock_ns += (uint64_t)milliseconds * 1000000; }
#include "../runtime/resource_core_budget.c"

static RcAdmissionVector cost(uint64_t host_mb, uint64_t gpu_mb, uint64_t staging_mb) {
    RcAdmissionVector v = {{0}};
    v.amount[RC_BUDGET_RAM] = v.amount[RC_BUDGET_COMMIT] = host_mb * MB;
    v.amount[RC_BUDGET_STAGING] = staging_mb * MB;
    OK(resource_core_budget_gpu_cost(&v, 0, gpu_mb * MB));
    return v;
}

static RcBudgetSnapshot snapshot(void) {
    RcBudgetSnapshot s;
    resource_core_budget_snapshot(&s);
    return s;
}

static void fresh(void) {
    clock_ns += 200000000;
    resource_core_budget_refresh();
}

static void start(uint64_t free_mb) {
    resource_core_budget_test_reset();
    clock_ns = 1000000000;
    sample_count = 0;
    observed = (RcBudgetSample){0};
    observed.known_mask = (1u << RC_BUDGET_RAM) | (1u << RC_BUDGET_COMMIT) |
        (1u << RC_BUDGET_STAGING) | (1u << RC_BUDGET_GPU_FIRST);
    observed.total.amount[RC_BUDGET_RAM] = 16384 * MB;
    observed.total.amount[RC_BUDGET_COMMIT] = 32768 * MB;
    observed.total.amount[RC_BUDGET_GPU_FIRST] = 8192 * MB;
    observed.free.amount[RC_BUDGET_RAM] = observed.free.amount[RC_BUDGET_COMMIT] = (free_mb + 256) * MB;
    observed.free.amount[RC_BUDGET_GPU_FIRST] = (free_mb + 64) * MB;
    observed.total.amount[RC_BUDGET_STAGING] = observed.free.amount[RC_BUDGET_STAGING] = 64 * MB;
    resource_core_budget_refresh();
}

static void shared_growth_and_actual_destruction(void) {
    start(100);
    RcAdmissionVector twenty = cost(20, 20, 8), ninety = cost(90, 90, 0);
    RcBudgetTicket texture = {0}, refused = {0};
    OK(resource_core_budget_reserve(RC_BUDGET_TEXTURE, 1, &twenty, &texture));
    RcBudgetSnapshot s = snapshot();
    assert(s.available.amount[RC_BUDGET_RAM] == 80 * MB);
    IS(resource_core_budget_reserve(RC_BUDGET_PIPELINE, 1, &ninety, &refused), RC_ADMISSION_REFUSED);
    assert(!refused.token.ledger);
    OK(resource_core_budget_materialize(&texture, &twenty));
    s = snapshot();
    assert(s.reserved.amount[RC_BUDGET_RAM] == 0 && s.retained.amount[RC_BUDGET_RAM] == 20 * MB);
    assert(s.available.amount[RC_BUDGET_RAM] == 80 * MB);
    assert(s.optional_available.amount[RC_BUDGET_STAGING] == 56 * MB);
    observed.free.amount[RC_BUDGET_RAM] -= 20 * MB;
    observed.free.amount[RC_BUDGET_COMMIT] -= 20 * MB;
    observed.free.amount[RC_BUDGET_GPU_FIRST] -= 20 * MB;
    fresh();
    s = snapshot();
    assert(s.unobserved.amount[RC_BUDGET_RAM] == 0);
    assert(s.available.amount[RC_BUDGET_RAM] == 80 * MB); // existing allocation is not charged twice
    OK(resource_core_budget_retire(&texture));
    s = snapshot();
    assert(s.pending.amount[RC_BUDGET_RAM] == 20 * MB && s.available.amount[RC_BUDGET_RAM] == 80 * MB);
    OK(resource_core_budget_destroyed(&texture));
    s = snapshot();
    assert(s.pending.amount[RC_BUDGET_RAM] == 0 && s.available.amount[RC_BUDGET_RAM] == 80 * MB);
    assert(s.optional_available.amount[RC_BUDGET_STAGING] == 64 * MB); // occupied-span target, not physical memory
    fresh(); // a driver still retaining its block provides no newly measured room
    assert(snapshot().available.amount[RC_BUDGET_GPU_FIRST] == 80 * MB);
    observed.free.amount[RC_BUDGET_RAM] += 20 * MB;
    observed.free.amount[RC_BUDGET_COMMIT] += 20 * MB;
    observed.free.amount[RC_BUDGET_GPU_FIRST] += 20 * MB;
    fresh();
    OK(resource_core_budget_reserve(RC_BUDGET_PIPELINE, 1, &ninety, &refused));
    OK(resource_core_budget_cancel(&refused));
}

static void opaque_estimates_settle_at_measurement(void) {
    start(100);
    RcAdmissionVector sixty = cost(60, 0, 0), forty = cost(40, 0, 0);
    RcBudgetTicket pipeline, concurrent, old;
    OK(resource_core_budget_reserve(RC_BUDGET_PIPELINE, 1, &sixty, &pipeline));
    old = pipeline;
    OK(resource_core_budget_opaque_done(&pipeline));
    assert(!pipeline.token.ledger);
    RcBudgetSnapshot s = snapshot();
    assert(s.producer[RC_BUDGET_PIPELINE].opaque_debt.amount[RC_BUDGET_COMMIT] == 60 * MB);
    assert(s.available.amount[RC_BUDGET_COMMIT] == 40 * MB);
    IS(resource_core_budget_opaque_done(&old), RC_ADMISSION_STALE);
    OK(resource_core_budget_reserve(RC_BUDGET_BUFFER, 1, &forty, &concurrent));
    IS(resource_core_budget_reserve(RC_BUDGET_PIPELINE, 1, &forty, &pipeline), RC_ADMISSION_REFUSED);
    // The actual retained driver result was 10 MiB; fresh observations replace the opaque 60 MiB estimate.
    observed.free.amount[RC_BUDGET_RAM] -= 10 * MB;
    observed.free.amount[RC_BUDGET_COMMIT] -= 10 * MB;
    fresh();
    s = snapshot();
    assert(s.producer[RC_BUDGET_PIPELINE].opaque_debt.amount[RC_BUDGET_COMMIT] == 0);
    assert(s.reserved.amount[RC_BUDGET_RAM] == 40 * MB && s.available.amount[RC_BUDGET_RAM] == 50 * MB);
    OK(resource_core_budget_reserve(RC_BUDGET_PIPELINE, 1, &forty, &pipeline));
    OK(resource_core_budget_cancel(&pipeline));
    OK(resource_core_budget_cancel(&concurrent));
}

static void split_drain_and_shrinking_headroom(void) {
    start(100);
    RcAdmissionVector peak = cost(80, 50, 32), payload = cost(30, 0, 0), drain = cost(50, 50, 32);
    RcBudgetTicket parent, source, upload, copy;
    OK(resource_core_budget_reserve(RC_BUDGET_TEXTURE, 1, &peak, &parent));
    OK(resource_core_budget_split(&parent, &payload, RC_BUDGET_TEXTURE, &source));
    OK(resource_core_budget_split(&parent, &drain, RC_BUDGET_STAGE, &upload));
    OK(resource_core_budget_cancel(&parent));
    observed.free.amount[RC_BUDGET_RAM] = 266 * MB;
    observed.free.amount[RC_BUDGET_COMMIT] = 266 * MB;
    fresh();
    copy = upload;
    IS(resource_core_budget_materialize(&source, &payload), RC_ADMISSION_ACCOUNTED_PRESSURE);
    IS(resource_core_budget_materialize(&upload, &drain), RC_ADMISSION_ACCOUNTED_PRESSURE);
    IS(resource_core_budget_cancel(&copy), RC_ADMISSION_STALE);
    RcBudgetSnapshot s = snapshot();
    assert(s.producer[RC_BUDGET_TEXTURE].retained.amount[RC_BUDGET_RAM] == 30 * MB);
    assert(s.producer[RC_BUDGET_STAGE].retained.amount[RC_BUDGET_RAM] == 50 * MB);
    assert(!s.reserved.amount[RC_BUDGET_RAM]);
    OK(resource_core_budget_retire(&source));
    OK(resource_core_budget_retire(&upload));
    OK(resource_core_budget_destroyed(&source));
    OK(resource_core_budget_destroyed(&upload));
    assert(snapshot().available.amount[RC_BUDGET_RAM] == 0); // no false credit from deferred or stale frees
}

static void available_memory_and_unknown_samples(void) {
    start(100);
    RcAdmissionVector one = cost(1, 0, 0), eighty = cost(80, 0, 0), gpu = cost(0, 1, 0);
    RcBudgetTicket t;
    IS(resource_core_budget_reserve(RC_BUDGET_PIPELINE, 0, &one, &t), RC_ADMISSION_REFUSED);
    OK(resource_core_budget_reserve(RC_BUDGET_TEXTURE, 1, &one, &t));
    OK(resource_core_budget_cancel(&t));
    // More installed capacity does not create free capacity beside another process's usage.
    observed.total.amount[RC_BUDGET_RAM] = 65536 * MB;
    fresh();
    IS(resource_core_budget_reserve(RC_BUDGET_TEXTURE, 1, &one, &t), RC_ADMISSION_REFUSED);

    resource_core_budget_test_reset();
    observed = (RcBudgetSample){0};
    fresh();
    IS(resource_core_budget_reserve(RC_BUDGET_PIPELINE, 0, &one, &t), RC_ADMISSION_UNSUPPORTED);
    IS(resource_core_budget_reserve(RC_BUDGET_TEXTURE, 1, &gpu, &t), RC_ADMISSION_UNSUPPORTED);
    OK(resource_core_budget_reserve(RC_BUDGET_PIPELINE, 1, &eighty, &t));
    OK(resource_core_budget_opaque_done(&t));
    fresh(); // missing telemetry cannot erase debt and supply another unbounded allowance
    IS(resource_core_budget_reserve(RC_BUDGET_PIPELINE, 1, &eighty, &t), RC_ADMISSION_REFUSED);
    assert(snapshot().unobserved.amount[RC_BUDGET_COMMIT] == 80 * MB);
    IS(resource_core_budget_gpu_cost(&gpu, RC_BUDGET_GPU_HEAPS, 1), RC_ADMISSION_UNSUPPORTED);
    resource_core_budget_wait(3);
    assert(!lock_depth && sample_count);
}

static void oversized_materialization_is_real_growth(void) {
    start(100);
    RcAdmissionVector promise = cost(64, 64, 0), actual = cost(120, 120, 0), one = cost(1, 1, 0);
    RcBudgetTicket allocation, refused;
    OK(resource_core_budget_reserve(RC_BUDGET_BUFFER, 1, &promise, &allocation));
    IS(resource_core_budget_materialize_growth(&allocation, &actual, &promise), RC_ADMISSION_ACCOUNTED_PRESSURE);
    RcBudgetSnapshot s = snapshot();
    assert(s.retained.amount[RC_BUDGET_GPU_FIRST] == 120 * MB);
    assert(s.unobserved.amount[RC_BUDGET_GPU_FIRST] == 120 * MB);
    IS(resource_core_budget_reserve(RC_BUDGET_TEXTURE, 1, &one, &refused), RC_ADMISSION_REFUSED);
    OK(resource_core_budget_retire(&allocation));
    OK(resource_core_budget_destroyed(&allocation));
    assert(snapshot().available.amount[RC_BUDGET_GPU_FIRST] == 0);
}

static void required_staging_obeys_physical_capacity(void) {
    start(4096);
    RcAdmissionVector upload = cost(80, 80, 80), one = cost(1, 1, 1);
    RcBudgetTicket ticket, refused;
    IS(resource_core_budget_reserve(RC_BUDGET_STAGE, 0, &upload, &refused), RC_ADMISSION_REFUSED);
    RcBudgetSnapshot s = snapshot();
    RcBudgetRefusal why = s.producer[RC_BUDGET_STAGE].last_refusal;
    assert(why.capacity_mask == (1u << RC_BUDGET_STAGING));
    assert(!why.required && why.request.amount[RC_BUDGET_STAGING] == 80 * MB);
    assert(why.available.amount[RC_BUDGET_STAGING] == 64 * MB);
    assert(s.producer[RC_BUDGET_STAGE].refused_domain[RC_BUDGET_STAGING] == 1);
    OK(resource_core_budget_reserve(RC_BUDGET_STAGE, 1, &upload, &ticket));
    OK(resource_core_budget_materialize(&ticket, &upload));
    s = snapshot();
    assert(s.retained.amount[RC_BUDGET_STAGING] == 80 * MB);
    assert(!s.optional_available.amount[RC_BUDGET_STAGING]);
    // Required progress ignores only the scheduling target. Each actual memory domain still refuses alone.
    const uint32_t hard[] = {RC_BUDGET_RAM, RC_BUDGET_COMMIT, RC_BUDGET_GPU_FIRST};
    const uint64_t margins[] = {256 * MB, 256 * MB, 64 * MB};
    for (unsigned i = 0; i < 3; i++) {
        uint64_t previous = observed.free.amount[hard[i]];
        observed.free.amount[hard[i]] = margins[i];
        fresh();
        IS(resource_core_budget_reserve(RC_BUDGET_STAGE, 1, &one, &refused), RC_ADMISSION_REFUSED);
        why = snapshot().producer[RC_BUDGET_STAGE].last_refusal;
        assert(why.required && why.capacity_mask == (1u << hard[i]));
        assert(why.request.amount[hard[i]] == MB && !why.available.amount[hard[i]]);
        observed.free.amount[hard[i]] = previous;
    }
    fresh();
    OK(resource_core_budget_retire(&ticket));
    assert(snapshot().pending.amount[RC_BUDGET_STAGING] == 80 * MB);
    OK(resource_core_budget_destroyed(&ticket));
    assert(!snapshot().pending.amount[RC_BUDGET_STAGING]);
    assert(snapshot().optional_available.amount[RC_BUDGET_STAGING] == 64 * MB);
    // Synthetic span metadata is optional; unknown real placement remains a hard unsupported domain.
    resource_core_budget_test_reset();
    observed = (RcBudgetSample){.known_mask = 1u << RC_BUDGET_GPU_FIRST};
    observed.total.amount[RC_BUDGET_GPU_FIRST] = 8192 * MB;
    observed.free.amount[RC_BUDGET_GPU_FIRST] = 128 * MB;
    fresh();
    RcAdmissionVector bounded = cost(16, 16, 80);
    OK(resource_core_budget_reserve(RC_BUDGET_STAGE, 1, &bounded, &ticket));
    OK(resource_core_budget_cancel(&ticket));
    resource_core_budget_test_reset();
    observed.known_mask = 0;
    fresh();
    IS(resource_core_budget_reserve(RC_BUDGET_STAGE, 1, &bounded, &refused), RC_ADMISSION_UNSUPPORTED);
    why = snapshot().producer[RC_BUDGET_STAGE].last_refusal;
    assert(why.unknown_mask == (1u << RC_BUDGET_GPU_FIRST));
}

static void optional_retry_progress_tracks_real_room(void) {
    start(2400);
    RcBudgetAvailability initial, charged, later;
    resource_core_budget_availability(&initial);
    assert(initial.progress && initial.available.amount[RC_BUDGET_RAM]);
    RcAdmissionVector amount = cost(32, 0, 0);
    RcBudgetTicket ticket;
    OK(resource_core_budget_reserve(RC_BUDGET_SNAPSHOT, 0, &amount, &ticket));
    resource_core_budget_availability(&charged);
    assert(charged.progress == initial.progress &&
           charged.available.amount[RC_BUDGET_RAM] + 32 * MB == initial.available.amount[RC_BUDGET_RAM]);
    OK(resource_core_budget_cancel(&ticket));
    resource_core_budget_availability(&later);
    assert(later.progress > charged.progress &&
           later.available.amount[RC_BUDGET_RAM] == initial.available.amount[RC_BUDGET_RAM]);
    initial = later;
    OK(resource_core_budget_reserve(RC_BUDGET_SNAPSHOT, 0, &amount, &ticket));
    OK(resource_core_budget_materialize(&ticket, &amount));
    resource_core_budget_availability(&charged);
    OK(resource_core_budget_retire(&ticket));
    OK(resource_core_budget_destroyed(&ticket));
    resource_core_budget_availability(&later);
    assert(later.progress == charged.progress &&
           later.available.amount[RC_BUDGET_RAM] == charged.available.amount[RC_BUDGET_RAM]);
    fresh(); // This synthetic sample confirms that the actual memory came back; destruction alone did not.
    resource_core_budget_availability(&later);
    assert(later.progress > charged.progress &&
           later.available.amount[RC_BUDGET_RAM] == initial.available.amount[RC_BUDGET_RAM]);
    initial = later;
    fresh();resource_core_budget_availability(&later);
    assert(later.progress == initial.progress); // Sampling unchanged room is not a retry event.
    observed.free.amount[RC_BUDGET_GPU_FIRST] += MB;
    fresh();resource_core_budget_availability(&later);
    assert(later.progress > initial.progress &&
           later.available.amount[RC_BUDGET_RAM] == initial.available.amount[RC_BUDGET_RAM]);
    // Consumers also compare their refused domains; unrelated heap relief cannot rearm a host-only capture.
}

static void request_progress_requires_all_constrained_domains(void) {
    start(0);
    RcAdmissionVector peak = cost(16, 16, 0);
    RcBudgetReadiness before, now;
    resource_core_budget_readiness(0, &before);
    assert(!resource_core_budget_peak_ready(&peak, &before));

    observed.free.amount[RC_BUDGET_GPU_FIRST + 1] += 4096 * MB;
    fresh();
    resource_core_budget_readiness(0, &now);
    assert(!resource_core_budget_peak_progress(&peak, &before, &now));
    observed.free.amount[RC_BUDGET_RAM] += 4096 * MB;
    observed.free.amount[RC_BUDGET_COMMIT] += 4096 * MB;
    fresh();
    resource_core_budget_readiness(0, &now);
    assert(!resource_core_budget_peak_progress(&peak, &before, &now));
    observed.free.amount[RC_BUDGET_GPU_FIRST] += 4096 * MB;
    fresh();
    resource_core_budget_readiness(0, &now);
    assert(resource_core_budget_peak_progress(&peak, &before, &now));
    assert(!resource_core_budget_peak_progress(&peak, &now, &now));

    // A capacity record is a separate prerequisite even when every byte allowance already fits.
    before = now;
    before.records_available = 0;
    assert(resource_core_budget_peak_progress(&peak, &before, &now));
}

static unsigned reclaim_calls;
static unsigned reclaim_callback(void *context, const RcAdmissionVector *deficit) {
    assert(!lock_depth && deficit->amount[RC_BUDGET_RAM]);
    reclaim_calls++;
    // Native destruction and budget updates must be callable without recursive budget-lock acquisition.
    resource_core_budget_progress();
    if (context) {
        observed.free.amount[RC_BUDGET_RAM] += 32 * MB;
        observed.free.amount[RC_BUDGET_COMMIT] += 32 * MB;
        return RC_BUDGET_RECLAIM_DESTROYED;
    }
    return RC_BUDGET_RECLAIM_PENDING;
}

static void shared_reclamation(void) {
    start(0);
    reclaim_calls = 0;
    uint32_t domains = (1u << RC_BUDGET_RAM) | (1u << RC_BUDGET_COMMIT);
    OK(resource_core_budget_register_reclaimer(RC_BUDGET_SNAPSHOT, domains, reclaim_callback, NULL));
    OK(resource_core_budget_register_reclaimer(RC_BUDGET_SNAPSHOT, domains, reclaim_callback, NULL));
    RcAdmissionVector demand = cost(16, 0, 0);
    RcBudgetProgress progress;
    resource_core_budget_service(1, &demand, &progress);
    assert(reclaim_calls == 1 && progress.blocked_mask && progress.reclaim == RC_BUDGET_RECLAIM_PENDING);
    // A scheduled free has no admission value until actual destruction and a confirming measurement.
    RcBudgetTicket ticket = {0};
    IS(resource_core_budget_reserve(RC_BUDGET_SNAPSHOT, 1, &demand, &ticket), RC_ADMISSION_REFUSED);
    OK(resource_core_budget_register_reclaimer(RC_BUDGET_SNAPSHOT, domains, reclaim_callback, &reclaim_calls));
    resource_core_budget_service(1, &demand, &progress);
    assert(reclaim_calls == 3 && !progress.blocked_mask);
    OK(resource_core_budget_reserve(RC_BUDGET_SNAPSHOT, 1, &demand, &ticket));
    uint64_t before = clock_ns;
    resource_core_budget_progress();
    resource_core_budget_wait_since(progress.event, 100);
    assert(clock_ns == before); // Progress between service and wait must not be lost.
    OK(resource_core_budget_cancel(&ticket));
}

int main(void) {
    shared_reclamation();
    optional_retry_progress_tracks_real_room();
    request_progress_requires_all_constrained_domains();
    shared_growth_and_actual_destruction();
    opaque_estimates_settle_at_measurement();
    split_drain_and_shrinking_headroom();
    available_memory_and_unknown_samples();
    oversized_materialization_is_real_growth();
    required_staging_obeys_physical_capacity();
    puts("resource_core_budget_check: passed");
    return 0;
}

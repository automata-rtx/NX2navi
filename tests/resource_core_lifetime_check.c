#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "../runtime/resource_core_lifetime.h"

#define OK(call) assert((call) == RC_LIFETIME_OK)
#define IS(call, result) assert((call) == (result))

static RcLifetimeCompletion point(uint64_t domain, uint64_t value) {
    RcLifetimeCompletion set = {0};
    OK(rc_lifetime_completion_add(&set, domain, value));
    return set;
}

static void old_recording_survives_replacement(void) {
    RcLifetimeIds ids = {0};
    RcLifetimeVersion old = {0}, next = {0};
    RcLifetimeResidency first = {0}, second = {0};
    RcLifetimeCapture use = {0}, late = {0};
    OK(rc_lifetime_version_init(&ids, &old, RC_LIFETIME_RECOVERY_CURRENT_SOURCE, 0));
    OK(rc_lifetime_residency_install(&ids, &old, old.id, &first));
    RcLifetimeBinding captured = first.binding;
    OK(rc_lifetime_capture_begin(&old, &first, captured, &use));
    OK(rc_lifetime_residency_detach(&old, &first, captured));
    IS(rc_lifetime_residency_retire(&old, &first, captured), RC_LIFETIME_BUSY);
    IS(rc_lifetime_capture_begin(&old, &first, captured, &late), RC_LIFETIME_INVALID);

    OK(rc_lifetime_version_init(&ids, &next, RC_LIFETIME_RECOVERY_IMMUTABLE, 0));
    OK(rc_lifetime_residency_install(&ids, &next, next.id, &second));
    assert(use.binding.version == old.id && use.binding.residency == captured.residency);
    assert(second.binding.residency != captured.residency);
    RcLifetimeCompletion done = point(11, 7), completed = point(11, 6), none = {0};
    OK(rc_lifetime_capture_submit(&old, &first, &use, &none, &done));
    IS(rc_lifetime_capture_cancel(&old, &first, &use), RC_LIFETIME_INVALID);
    IS(rc_lifetime_capture_complete(&old, &first, &use, &completed), RC_LIFETIME_BUSY);
    OK(rc_lifetime_capture_complete(&old, &first, &use, &done));
    IS(rc_lifetime_capture_complete(&old, &first, &use, &done), RC_LIFETIME_INVALID);
    OK(rc_lifetime_residency_retire(&old, &first, captured));
    OK(rc_lifetime_version_retire(&old, old.id));
}

static void exact_history_requires_real_recovery(void) {
    RcLifetimeIds ids = {0};
    RcLifetimeVersion v = {0};
    RcLifetimeResidency resident = {0};
    OK(rc_lifetime_version_init(&ids, &v, RC_LIFETIME_RECOVERY_CURRENT_SOURCE, 0));
    assert(rc_lifetime_reconstructible(v.recovery, RC_LIFETIME_CURRENT));
    assert(!rc_lifetime_reconstructible(v.recovery, RC_LIFETIME_EXACT));
    IS(rc_lifetime_exact_hold(&v, v.id), RC_LIFETIME_NO_RECOVERY);
    OK(rc_lifetime_residency_install(&ids, &v, v.id, &resident));
    OK(rc_lifetime_exact_hold(&v, v.id));
    IS(rc_lifetime_residency_detach(&v, &resident, resident.binding), RC_LIFETIME_NO_RECOVERY);
    OK(rc_lifetime_recovery_set(&v, v.id, RC_LIFETIME_RECOVERY_IMMUTABLE));
    RcLifetimeBinding previous = resident.binding;
    OK(rc_lifetime_residency_detach(&v, &resident, previous));
    OK(rc_lifetime_residency_retire(&v, &resident, previous));
    IS(rc_lifetime_recovery_set(&v, v.id, RC_LIFETIME_RECOVERY_CURRENT_SOURCE), RC_LIFETIME_NO_RECOVERY);
    IS(rc_lifetime_version_retire(&v, v.id), RC_LIFETIME_BUSY);
    OK(rc_lifetime_residency_install(&ids, &v, v.id, &resident));
    assert(resident.binding.version == previous.version && resident.binding.residency != previous.residency);
    RcLifetimeCapture stale = {0};
    IS(rc_lifetime_capture_begin(&v, &resident, previous, &stale), RC_LIFETIME_INVALID);
    OK(rc_lifetime_exact_release(&v, v.id));
    OK(rc_lifetime_recovery_set(&v, v.id, RC_LIFETIME_RECOVERY_CURRENT_SOURCE));
    OK(rc_lifetime_residency_detach(&v, &resident, resident.binding));
    OK(rc_lifetime_residency_retire(&v, &resident, resident.binding));
    OK(rc_lifetime_version_retire(&v, v.id));
}

static void all_completion_domains_must_retire(void) {
    RcLifetimeIds ids = {0};
    RcLifetimeVersion v = {0};
    RcLifetimeResidency r = {0};
    RcLifetimeCapture use = {0};
    OK(rc_lifetime_version_init(&ids, &v, RC_LIFETIME_RECOVERY_NONE, 0));
    OK(rc_lifetime_residency_install(&ids, &v, v.id, &r));
    OK(rc_lifetime_capture_begin(&v, &r, r.binding, &use));
    RcLifetimeCompletion need = point(1, 30), have = point(1, UINT64_MAX), none = {0};
    OK(rc_lifetime_completion_add(&need, 2, 4));
    OK(rc_lifetime_completion_add(&need, 3, 9));
    OK(rc_lifetime_capture_submit(&v, &r, &use, &none, &need));
    OK(rc_lifetime_residency_detach(&v, &r, r.binding));
    IS(rc_lifetime_capture_complete(&v, &r, &use, &have), RC_LIFETIME_BUSY);
    OK(rc_lifetime_completion_add(&have, 2, 4));
    IS(rc_lifetime_capture_complete(&v, &r, &use, &have), RC_LIFETIME_BUSY);
    OK(rc_lifetime_completion_add(&have, 3, 8));
    IS(rc_lifetime_capture_complete(&v, &r, &use, &have), RC_LIFETIME_BUSY);
    OK(rc_lifetime_completion_add(&have, 3, 9));
    OK(rc_lifetime_capture_complete(&v, &r, &use, &have));
    OK(rc_lifetime_residency_retire(&v, &r, r.binding));
}

static void pending_producer_cancellation_and_submission(void) {
    RcLifetimeIds ids = {0};
    RcLifetimeVersion v = {0};
    RcLifetimeResidency r = {0};
    RcLifetimeCapture a = {0}, b = {0};
    RcLifetimeCompletion producer = point(4, 12), consumer = point(5, 2), empty = {0};
    OK(rc_lifetime_version_init(&ids, &v, RC_LIFETIME_RECOVERY_NONE, 1));
    OK(rc_lifetime_residency_install(&ids, &v, v.id, &r));
    OK(rc_lifetime_capture_begin(&v, &r, r.binding, &a));
    OK(rc_lifetime_capture_begin(&v, &r, r.binding, &b));
    IS(rc_lifetime_capture_submit(&v, &r, &a, &producer, &consumer), RC_LIFETIME_DEPENDENCY);
    OK(rc_lifetime_capture_cancel(&v, &r, &a));
    assert(r.captures == 1);
    OK(rc_lifetime_producer_cancel(&v, v.id));
    IS(rc_lifetime_capture_submit(&v, &r, &b, &producer, &consumer), RC_LIFETIME_DEPENDENCY);
    OK(rc_lifetime_residency_detach(&v, &r, r.binding));
    IS(rc_lifetime_residency_retire(&v, &r, r.binding), RC_LIFETIME_BUSY);
    OK(rc_lifetime_capture_cancel(&v, &r, &b));
    OK(rc_lifetime_residency_retire(&v, &r, r.binding));
    uint64_t obsolete = v.id;
    OK(rc_lifetime_version_retire(&v, v.id));

    OK(rc_lifetime_version_init(&ids, &v, RC_LIFETIME_RECOVERY_NONE, 1));
    assert(v.id != obsolete);
    IS(rc_lifetime_producer_cancel(&v, obsolete), RC_LIFETIME_INVALID);
    OK(rc_lifetime_residency_install(&ids, &v, v.id, &r));
    OK(rc_lifetime_capture_begin(&v, &r, r.binding, &a));
    OK(rc_lifetime_producer_submit(&v, v.id, &producer));
    IS(rc_lifetime_producer_cancel(&v, v.id), RC_LIFETIME_INVALID);
    IS(rc_lifetime_capture_submit(&v, &r, &a, &empty, &consumer), RC_LIFETIME_DEPENDENCY);
    OK(rc_lifetime_capture_submit(&v, &r, &a, &producer, &consumer));
    OK(rc_lifetime_residency_detach(&v, &r, r.binding));
    OK(rc_lifetime_capture_complete(&v, &r, &a, &consumer));
    IS(rc_lifetime_residency_retire(&v, &r, r.binding), RC_LIFETIME_BUSY);
    IS(rc_lifetime_producer_complete(&v, v.id, &empty), RC_LIFETIME_BUSY);
    OK(rc_lifetime_producer_complete(&v, v.id, &producer));
    OK(rc_lifetime_residency_retire(&v, &r, r.binding));
}

static void bounds_fail_without_partial_changes(void) {
    RcLifetimeCompletion full = {0};
    for (uint64_t i = 1; i <= RC_LIFETIME_DOMAINS; i++) OK(rc_lifetime_completion_add(&full, i, 1));
    RcLifetimeCompletion before = full;
    IS(rc_lifetime_completion_add(&full, 99, 1), RC_LIFETIME_CAPACITY);
    assert(!memcmp(&full, &before, sizeof full));
    OK(rc_lifetime_completion_add(&full, 1, 7));
    OK(rc_lifetime_completion_add(&full, 1, 3));
    assert(full.points[0].value == 7);
    RcLifetimeIds ids = {UINT64_MAX};
    RcLifetimeVersion v = {0};
    IS(rc_lifetime_version_init(&ids, &v, RC_LIFETIME_RECOVERY_NONE, 0), RC_LIFETIME_EXHAUSTED);
    assert(!v.id && v.state == RC_LIFETIME_VERSION_EMPTY && ids.last == UINT64_MAX);
}

static void copied_tickets_cannot_release_other_holds(void) {
    RcLifetimeIds ids = {0};
    RcLifetimeVersion v = {0};
    RcLifetimeResidency r = {0};
    RcLifetimeCapture a = {0}, b = {0}, moved = {0};
    RcLifetimeCompletion done = point(1, 1), empty = {0};
    OK(rc_lifetime_version_init(&ids, &v, RC_LIFETIME_RECOVERY_NONE, 0));
    OK(rc_lifetime_residency_install(&ids, &v, v.id, &r));
    OK(rc_lifetime_capture_begin(&v, &r, r.binding, &a));
    OK(rc_lifetime_capture_begin(&v, &r, r.binding, &b));
    RcLifetimeCapture copied = a;
    IS(rc_lifetime_capture_cancel(&v, &r, &copied), RC_LIFETIME_INVALID);
    IS(rc_lifetime_capture_submit(&v, &r, &copied, &empty, &done), RC_LIFETIME_INVALID);
    assert(r.captures == 2);
    OK(rc_lifetime_capture_move(&moved, &a));
    IS(rc_lifetime_capture_cancel(&v, &r, &a), RC_LIFETIME_INVALID);
    OK(rc_lifetime_capture_cancel(&v, &r, &moved));
    assert(r.captures == 1);
    OK(rc_lifetime_capture_submit(&v, &r, &b, &empty, &done));
    copied = b;
    IS(rc_lifetime_capture_complete(&v, &r, &copied, &done), RC_LIFETIME_INVALID);
    OK(rc_lifetime_capture_move(&moved, &b));
    IS(rc_lifetime_capture_complete(&v, &r, &b, &done), RC_LIFETIME_INVALID);
    OK(rc_lifetime_capture_complete(&v, &r, &moved, &done));
    assert(!r.captures);
}

static void new_residency_has_its_own_upload_obligation(void) {
    RcLifetimeIds ids = {0};
    RcLifetimeVersion v = {0};
    RcLifetimeResidency r = {0};
    RcLifetimeCapture use = {0};
    RcLifetimeCompletion upload = point(7, 5), draw = point(1, 6), empty = {0};
    OK(rc_lifetime_version_init(&ids, &v, RC_LIFETIME_RECOVERY_IMMUTABLE, 0));
    OK(rc_lifetime_residency_install_pending(&ids, &v, v.id, &r));
    OK(rc_lifetime_capture_begin(&v, &r, r.binding, &use));
    IS(rc_lifetime_capture_submit(&v, &r, &use, &upload, &draw), RC_LIFETIME_DEPENDENCY);
    OK(rc_lifetime_upload_submit(&v, &r, r.binding, &upload));
    IS(rc_lifetime_capture_submit(&v, &r, &use, &empty, &draw), RC_LIFETIME_DEPENDENCY);
    OK(rc_lifetime_capture_submit(&v, &r, &use, &upload, &draw));
    OK(rc_lifetime_residency_detach(&v, &r, r.binding));
    OK(rc_lifetime_capture_complete(&v, &r, &use, &draw));
    IS(rc_lifetime_residency_retire(&v, &r, r.binding), RC_LIFETIME_BUSY);
    IS(rc_lifetime_upload_cancel(&v, &r, r.binding), RC_LIFETIME_INVALID);
    OK(rc_lifetime_upload_complete(&v, &r, r.binding, &upload));
    OK(rc_lifetime_residency_retire(&v, &r, r.binding));

    OK(rc_lifetime_residency_install_pending(&ids, &v, v.id, &r));
    OK(rc_lifetime_capture_begin(&v, &r, r.binding, &use));
    OK(rc_lifetime_upload_cancel(&v, &r, r.binding));
    IS(rc_lifetime_capture_submit(&v, &r, &use, &upload, &draw), RC_LIFETIME_DEPENDENCY);
    OK(rc_lifetime_capture_cancel(&v, &r, &use));
    OK(rc_lifetime_residency_retire(&v, &r, r.binding));
    assert(v.state == RC_LIFETIME_VERSION_READY && v.recovery == RC_LIFETIME_RECOVERY_IMMUTABLE);
}

int main(void) {
    old_recording_survives_replacement();
    exact_history_requires_real_recovery();
    all_completion_domains_must_retire();
    pending_producer_cancellation_and_submission();
    bounds_fail_without_partial_changes();
    copied_tickets_cannot_release_other_holds();
    new_residency_has_its_own_upload_obligation();
    puts("resource core lifetime checks passed");
    return 0;
}

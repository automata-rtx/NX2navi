// Synthetic adapter contract check, not a test of installed guest-memory or Vulkan hooks.
#include "../runtime/resource_core_observation.h"
#include "../runtime/resource_core_lifetime.h"
#include "../runtime/resource_core_admission.h"
#include <assert.h>
#include <stdio.h>

enum { PREP = 1, SNAPSHOT, STAGING, IMAGE, UPLOAD_DOMAIN = 11, DRAW_DOMAIN, PRESENT_DOMAIN };

static RcAdmissionVector amount(uint64_t ram, uint64_t commit, uint64_t gpu, uint64_t staging) {
    return (RcAdmissionVector){{ram, commit, gpu, staging}};
}

static RcAdmissionToken allocate_part(RcAdmission *b, RcAdmissionToken parent, uint64_t owner,
                                     RcAdmissionVector footprint) {
    RcAdmissionToken ticket, charge;
    assert(rc_admission_split(b, parent, PREP, &footprint, owner, &ticket) == RC_ADMISSION_OK);
    // A real adapter allocates only after the split succeeds, then reports the actual footprint.
    assert(rc_admission_materialize(b, ticket, owner, &footprint, &charge) == RC_ADMISSION_OK);
    return charge;
}

static void destroy_charge(RcAdmission *b, RcAdmissionToken charge, uint64_t owner) {
    RcAdmissionToken pending;
    assert(rc_admission_retire(b, charge, owner, &pending) == RC_ADMISSION_OK);
    // The fixture has no backend object. Live adapters must actually destroy before this notification.
    assert(rc_admission_destroyed(b, pending, owner) == RC_ADMISSION_OK);
}

static int publish_current(const RcSourceDomain *source, const RcObservation *observed,
                           uint64_t version, uint64_t *association) {
    // Synthetic serialized association update. Packet B must supply this atomic validation/publication boundary.
    if (!rc_observation_current(source, observed)) return 0;
    *association = version;
    return 1;
}

static void captured_old_version(void) {
    RcAdmissionDomain domains[] = {
        {RC_ADMISSION_RAM, 1, 32}, {RC_ADMISSION_COMMIT, 1, 32},
        {RC_ADMISSION_GPU_HEAP, 1, 96}, {RC_ADMISSION_STAGING, 1, 32}
    };
    RcAdmissionRecord records[8];
    RcAdmission budget;
    assert(rc_admission_init(&budget, 1, domains, 4, records, 8) == RC_ADMISSION_OK);
    RcAdmissionVector peak = amount(32, 32, 96, 32);
    RcAdmissionToken prep;
    assert(rc_admission_reserve(&budget, PREP, &peak, &prep) == RC_ADMISSION_OK);
    RcAdmissionToken snapshot_charge = allocate_part(&budget, prep, SNAPSHOT, amount(4, 4, 0, 0));
    RcAdmissionToken staging_charge = allocate_part(&budget, prep, STAGING, amount(16, 16, 0, 16));
    RcAdmissionToken image_charge = allocate_part(&budget, prep, IMAGE, amount(0, 0, 64, 0));
    assert(rc_admission_cancel(&budget, prep, PREP) == RC_ADMISSION_OK);
    assert(budget.retained.amount[0] == 20 && budget.retained.amount[2] == 64);

    RcSourceDomain source;
    RcSourceRange range = {1, 64, 4};
    RcObservation observed;
    assert(rc_source_init(&source, 1, range));
    assert(rc_source_observe(&source, &observed));
    unsigned char guest[4] = {1, 2, 3, 4}, immutable[4];
    memcpy(immutable, guest, sizeof immutable);
    // No writer races in this fixture. The live capture mechanism must establish that separately.
    assert(rc_observation_current(&source, &observed));
    RcLifetimeIds ids = {0};
    RcLifetimeVersion version = {0};
    RcLifetimeResidency residency = {0};
    RcLifetimeCapture draw = {0};
    assert(rc_lifetime_version_init(&ids, &version, RC_LIFETIME_RECOVERY_IMMUTABLE, 0) == RC_LIFETIME_OK);
    RcPreparedInput provider = {version.id, 1, 1, 1, RC_PROVIDER_CAPTURE};
    assert(rc_input_exact(&provider));
    assert(rc_lifetime_exact_hold(&version, version.id) == RC_LIFETIME_OK);
    assert(rc_lifetime_residency_install_pending(&ids, &version, version.id, &residency) == RC_LIFETIME_OK);
    RcLifetimeBinding old_binding = residency.binding;
    assert(rc_lifetime_capture_begin(&version, &residency, old_binding, &draw) == RC_LIFETIME_OK);
    uint64_t association = 0;
    assert(publish_current(&source, &observed, version.id, &association));

    guest[0] = 9;
    assert(rc_source_changed(&source, range, 1));
    assert(!publish_current(&source, &observed, version.id, &association));
    rc_source_finalize(&source);
    assert(rc_source_init(&source, 2, range));
    assert(!publish_current(&source, &observed, version.id, &association));
    RcObservation newer;
    RcLifetimeVersion next = {0};
    assert(rc_source_observe(&source, &newer));
    assert(rc_lifetime_version_init(&ids, &next, RC_LIFETIME_RECOVERY_CURRENT_SOURCE, 0) == RC_LIFETIME_OK);
    assert(publish_current(&source, &newer, next.id, &association));
    assert(association != old_binding.version && immutable[0] == 1 && guest[0] == 9);
    assert(!rc_lifetime_reconstructible(next.recovery, RC_LIFETIME_EXACT));

    RcLifetimeCompletion none = {0}, upload = {0}, consumer = {0}, completed = {0};
    assert(rc_lifetime_completion_add(&upload, UPLOAD_DOMAIN, 7) == RC_LIFETIME_OK);
    assert(rc_lifetime_completion_add(&consumer, DRAW_DOMAIN, 20) == RC_LIFETIME_OK);
    assert(rc_lifetime_completion_add(&consumer, PRESENT_DOMAIN, 3) == RC_LIFETIME_OK);
    assert(rc_lifetime_capture_submit(&version, &residency, &draw, &none, &consumer) == RC_LIFETIME_DEPENDENCY);
    assert(rc_lifetime_upload_submit(&version, &residency, old_binding, &upload) == RC_LIFETIME_OK);
    assert(rc_lifetime_capture_submit(&version, &residency, &draw, &none, &consumer) == RC_LIFETIME_DEPENDENCY);
    assert(rc_lifetime_capture_submit(&version, &residency, &draw, &upload, &consumer) == RC_LIFETIME_OK);
    assert(draw.binding.version == provider.content);
    assert(rc_lifetime_residency_detach(&version, &residency, old_binding) == RC_LIFETIME_OK);
    assert(rc_lifetime_residency_retire(&version, &residency, old_binding) == RC_LIFETIME_BUSY);
    assert(rc_lifetime_completion_add(&completed, DRAW_DOMAIN, 1000) == RC_LIFETIME_OK);
    assert(rc_lifetime_capture_complete(&version, &residency, &draw, &completed) == RC_LIFETIME_BUSY);
    assert(rc_lifetime_upload_complete(&version, &residency, old_binding, &completed) == RC_LIFETIME_BUSY);
    assert(rc_lifetime_completion_add(&completed, PRESENT_DOMAIN, 3) == RC_LIFETIME_OK);
    assert(rc_lifetime_capture_complete(&version, &residency, &draw, &completed) == RC_LIFETIME_OK);
    // A large graphics counter does not satisfy a different upload domain.
    assert(rc_lifetime_residency_retire(&version, &residency, old_binding) == RC_LIFETIME_BUSY);
    assert(rc_lifetime_completion_add(&completed, UPLOAD_DOMAIN, 7) == RC_LIFETIME_OK);
    assert(rc_lifetime_upload_complete(&version, &residency, old_binding, &completed) == RC_LIFETIME_OK);
    destroy_charge(&budget, staging_charge, STAGING);
    assert(rc_lifetime_residency_retire(&version, &residency, old_binding) == RC_LIFETIME_OK);
    RcAdmissionToken pending;
    assert(rc_admission_retire(&budget, image_charge, IMAGE, &pending) == RC_ADMISSION_OK);
    RcAdmissionVector replacement = amount(0, 0, 64, 0);
    RcAdmissionToken replacement_ticket = {0};
    assert(rc_admission_reserve(&budget, IMAGE, &replacement, &replacement_ticket) == RC_ADMISSION_REFUSED);
    assert(rc_admission_destroyed(&budget, pending, IMAGE) == RC_ADMISSION_OK);
    assert(rc_admission_reserve(&budget, IMAGE, &replacement, &replacement_ticket) == RC_ADMISSION_OK);
    assert(rc_admission_cancel(&budget, replacement_ticket, IMAGE) == RC_ADMISSION_OK);
    assert(rc_lifetime_version_retire(&version, version.id) == RC_LIFETIME_BUSY);
    assert(rc_lifetime_exact_release(&version, version.id) == RC_LIFETIME_OK);
    assert(rc_lifetime_version_retire(&version, version.id) == RC_LIFETIME_OK);
    destroy_charge(&budget, snapshot_charge, SNAPSHOT);
    assert(rc_lifetime_version_retire(&next, next.id) == RC_LIFETIME_OK);
    for (uint32_t i = 0; i < 4; i++) assert(rc_admission_used_at(&budget, i) == 0);
}

static void canceled_writer(void) {
    RcSourceDomain source;
    RcSourceRange range = {2, 0, 16};
    RcAuthorityTransfer writer = {0};
    RcObservation before, after;
    assert(rc_source_init(&source, 3, range));
    assert(rc_source_observe(&source, &before));
    assert(rc_authority_begin(&source, &writer));
    assert(!rc_source_observe(&source, &after));
    RcLifetimeIds ids = {0};
    RcLifetimeVersion produced = {0};
    RcLifetimeResidency image = {0};
    RcLifetimeCapture dependent = {0};
    assert(rc_lifetime_version_init(&ids, &produced, RC_LIFETIME_RECOVERY_NONE, 1) == RC_LIFETIME_OK);
    assert(rc_lifetime_residency_install(&ids, &produced, produced.id, &image) == RC_LIFETIME_OK);
    assert(rc_lifetime_capture_begin(&produced, &image, image.binding, &dependent) == RC_LIFETIME_OK);
    RcLifetimeCompletion none = {0}, done = {0};
    assert(rc_lifetime_completion_add(&done, DRAW_DOMAIN, 1) == RC_LIFETIME_OK);
    assert(rc_lifetime_capture_submit(&produced, &image, &dependent, &none, &done) == RC_LIFETIME_DEPENDENCY);
    assert(rc_source_changed(&source, range, 1));
    assert(!rc_authority_end(&source, &writer, RC_AUTH_GPU, 1));
    assert(rc_lifetime_producer_cancel(&produced, produced.id) == RC_LIFETIME_OK);
    assert(rc_authority_end(&source, &writer, RC_AUTH_CPU, 0));
    assert(source.authority == RC_AUTH_COMPATIBILITY && !rc_observation_current(&source, &before));
    assert(rc_lifetime_capture_submit(&produced, &image, &dependent, &none, &done) == RC_LIFETIME_DEPENDENCY);
    assert(rc_lifetime_capture_cancel(&produced, &image, &dependent) == RC_LIFETIME_OK);
    assert(rc_lifetime_residency_detach(&produced, &image, image.binding) == RC_LIFETIME_OK);
    assert(rc_lifetime_residency_retire(&produced, &image, image.binding) == RC_LIFETIME_OK);
    assert(rc_lifetime_version_retire(&produced, produced.id) == RC_LIFETIME_OK);
}

int main(void) {
    captured_old_version();
    canceled_writer();
    puts("resource core integration checks passed");
    return 0;
}

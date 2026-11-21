// Explicit content-version and physical-use lifetimes. This module allocates nothing and does not synchronize.
// The caller serializes every operation, including lookup plus acquisition, and keeps these records alive through
// their holds. Zero initialize records before first use and change active records only through these operations.
// IDs come from one issuer for its entire lifetime; neither the issuer nor completion-domain IDs may
// be reset/recycled while old identities exist. Completion values advance monotonically within their own domain.
#ifndef RESOURCE_CORE_LIFETIME_H
#define RESOURCE_CORE_LIFETIME_H

#include <stdint.h>

enum { RC_LIFETIME_DOMAINS = 8 };

typedef enum {
    RC_LIFETIME_OK, RC_LIFETIME_INVALID, RC_LIFETIME_BUSY, RC_LIFETIME_NO_RECOVERY,
    RC_LIFETIME_CAPACITY, RC_LIFETIME_EXHAUSTED, RC_LIFETIME_DEPENDENCY
} RcLifetimeResult;
typedef enum { RC_LIFETIME_CURRENT, RC_LIFETIME_EXACT } RcLifetimeRequest;
typedef enum {
    RC_LIFETIME_RECOVERY_NONE, RC_LIFETIME_RECOVERY_CURRENT_SOURCE, RC_LIFETIME_RECOVERY_IMMUTABLE
} RcLifetimeRecovery;
typedef enum {
    RC_LIFETIME_VERSION_EMPTY, RC_LIFETIME_VERSION_READY, RC_LIFETIME_VERSION_PENDING,
    RC_LIFETIME_VERSION_SUBMITTED, RC_LIFETIME_VERSION_CANCELED, RC_LIFETIME_VERSION_RETIRED
} RcLifetimeVersionState;
typedef enum {
    RC_LIFETIME_RESIDENCY_EMPTY, RC_LIFETIME_RESIDENCY_AVAILABLE,
    RC_LIFETIME_RESIDENCY_DETACHED, RC_LIFETIME_RESIDENCY_RETIRED
} RcLifetimeResidencyState;
typedef enum {
    RC_LIFETIME_CAPTURE_EMPTY, RC_LIFETIME_CAPTURE_RECORDING,
    RC_LIFETIME_CAPTURE_SUBMITTED, RC_LIFETIME_CAPTURE_RELEASED
} RcLifetimeCaptureState;
typedef enum {
    RC_LIFETIME_UPLOAD_READY, RC_LIFETIME_UPLOAD_PENDING,
    RC_LIFETIME_UPLOAD_SUBMITTED, RC_LIFETIME_UPLOAD_CANCELED
} RcLifetimeUploadState;

typedef struct { uint64_t last; } RcLifetimeIds;
typedef struct { uint64_t domain, value; } RcLifetimePoint;
typedef struct { RcLifetimePoint points[RC_LIFETIME_DOMAINS]; uint32_t count; } RcLifetimeCompletion;
typedef struct { uint64_t version, residency; } RcLifetimeBinding;

typedef struct {
    uint64_t id;
    RcLifetimeVersionState state;
    RcLifetimeRecovery recovery;
    uint32_t exact_holds, residencies, available_residencies;
    RcLifetimeCompletion producer_done;
} RcLifetimeVersion;

typedef struct {
    RcLifetimeBinding binding;
    RcLifetimeResidencyState state;
    uint32_t captures;
    RcLifetimeUploadState upload;
    RcLifetimeCompletion upload_done;
} RcLifetimeResidency;

// The binding is immutable from acquisition through release; lifecycle state belongs to this one owning ticket.
// Copying binding values is allowed. Active tickets have stable addresses; only capture_move transfers ownership.
// The owner address rejects accidental byte copies instead of allowing a copied ticket to release another hold.
typedef struct {
    RcLifetimeBinding binding;
    RcLifetimeCaptureState state;
    RcLifetimeCompletion done;
    uintptr_t owner;
} RcLifetimeCapture;

static inline RcLifetimeResult rc_lifetime_id_next(RcLifetimeIds *ids, uint64_t *out) {
    if (!ids || !out) return RC_LIFETIME_INVALID;
    if (ids->last == UINT64_MAX) return RC_LIFETIME_EXHAUSTED;
    *out = ++ids->last;
    return RC_LIFETIME_OK;
}

static inline int rc_lifetime_completion_valid(const RcLifetimeCompletion *set) {
    if (!set || set->count > RC_LIFETIME_DOMAINS) return 0;
    for (uint32_t i = 0; i < set->count; i++) {
        if (!set->points[i].domain || !set->points[i].value) return 0;
        for (uint32_t j = 0; j < i; j++)
            if (set->points[j].domain == set->points[i].domain) return 0;
    }
    return 1;
}

// Only values in the same completion domain may be reduced to their maximum.
static inline RcLifetimeResult rc_lifetime_completion_add(RcLifetimeCompletion *set, uint64_t domain,
                                                           uint64_t value) {
    if (!rc_lifetime_completion_valid(set) || !domain || !value) return RC_LIFETIME_INVALID;
    for (uint32_t i = 0; i < set->count; i++) {
        if (set->points[i].domain != domain) continue;
        if (value > set->points[i].value) set->points[i].value = value;
        return RC_LIFETIME_OK;
    }
    if (set->count == RC_LIFETIME_DOMAINS) return RC_LIFETIME_CAPACITY;
    set->points[set->count++] = (RcLifetimePoint){domain, value};
    return RC_LIFETIME_OK;
}

static inline int rc_lifetime_completion_covers(const RcLifetimeCompletion *have,
                                               const RcLifetimeCompletion *need) {
    if (!rc_lifetime_completion_valid(have) || !rc_lifetime_completion_valid(need)) return 0;
    for (uint32_t i = 0; i < need->count; i++) {
        uint32_t j = 0;
        for (; j < have->count; j++)
            if (have->points[j].domain == need->points[i].domain &&
                have->points[j].value >= need->points[i].value) break;
        if (j == have->count) return 0;
    }
    return 1;
}

static inline int rc_lifetime_version_matches(const RcLifetimeVersion *v, uint64_t id) {
    return v && id && v->id == id && v->state != RC_LIFETIME_VERSION_EMPTY &&
           v->state != RC_LIFETIME_VERSION_RETIRED;
}

static inline int rc_lifetime_binding_matches(const RcLifetimeVersion *v, const RcLifetimeResidency *r,
                                              RcLifetimeBinding binding) {
    return rc_lifetime_version_matches(v, binding.version) && r && binding.residency &&
           r->binding.version == binding.version && r->binding.residency == binding.residency &&
           r->state != RC_LIFETIME_RESIDENCY_EMPTY && r->state != RC_LIFETIME_RESIDENCY_RETIRED;
}

// CURRENT_SOURCE means reobserve a current association, possibly obtaining a different version. It never
// promises to reproduce this exact version. IMMUTABLE means the caller has admitted a real protected provider.
static inline int rc_lifetime_reconstructible(RcLifetimeRecovery recovery, RcLifetimeRequest request) {
    return (request == RC_LIFETIME_CURRENT || request == RC_LIFETIME_EXACT) &&
           (recovery == RC_LIFETIME_RECOVERY_IMMUTABLE ||
            (request == RC_LIFETIME_CURRENT && recovery == RC_LIFETIME_RECOVERY_CURRENT_SOURCE));
}

static inline RcLifetimeResult rc_lifetime_version_init(RcLifetimeIds *ids, RcLifetimeVersion *v,
                                                       RcLifetimeRecovery recovery, int pending_producer) {
    if (!v || (v->state != RC_LIFETIME_VERSION_EMPTY && v->state != RC_LIFETIME_VERSION_RETIRED) ||
        recovery < RC_LIFETIME_RECOVERY_NONE || recovery > RC_LIFETIME_RECOVERY_IMMUTABLE)
        return RC_LIFETIME_INVALID;
    uint64_t id;
    RcLifetimeResult result = rc_lifetime_id_next(ids, &id);
    if (result != RC_LIFETIME_OK) return result;
    *v = (RcLifetimeVersion){.id = id, .recovery = recovery,
        .state = pending_producer ? RC_LIFETIME_VERSION_PENDING : RC_LIFETIME_VERSION_READY};
    return RC_LIFETIME_OK;
}

static inline RcLifetimeResult rc_lifetime_recovery_set(RcLifetimeVersion *v, uint64_t id,
                                                       RcLifetimeRecovery recovery) {
    if (!rc_lifetime_version_matches(v, id) || v->state == RC_LIFETIME_VERSION_CANCELED ||
        recovery < RC_LIFETIME_RECOVERY_NONE || recovery > RC_LIFETIME_RECOVERY_IMMUTABLE)
        return RC_LIFETIME_INVALID;
    if (v->exact_holds && !v->available_residencies && recovery != RC_LIFETIME_RECOVERY_IMMUTABLE)
        return RC_LIFETIME_NO_RECOVERY;
    v->recovery = recovery;
    return RC_LIFETIME_OK;
}

static inline RcLifetimeResult rc_lifetime_exact_hold(RcLifetimeVersion *v, uint64_t id) {
    if (!rc_lifetime_version_matches(v, id) || v->state == RC_LIFETIME_VERSION_CANCELED)
        return RC_LIFETIME_INVALID;
    if (!v->available_residencies && !rc_lifetime_reconstructible(v->recovery, RC_LIFETIME_EXACT))
        return RC_LIFETIME_NO_RECOVERY;
    if (v->exact_holds == UINT32_MAX) return RC_LIFETIME_CAPACITY;
    v->exact_holds++;
    return RC_LIFETIME_OK;
}

static inline RcLifetimeResult rc_lifetime_exact_release(RcLifetimeVersion *v, uint64_t id) {
    if (!rc_lifetime_version_matches(v, id) || !v->exact_holds) return RC_LIFETIME_INVALID;
    v->exact_holds--;
    return RC_LIFETIME_OK;
}

static inline RcLifetimeResult rc_lifetime_producer_submit(RcLifetimeVersion *v, uint64_t id,
                                                         const RcLifetimeCompletion *done) {
    if (!rc_lifetime_version_matches(v, id) || v->state != RC_LIFETIME_VERSION_PENDING ||
        !rc_lifetime_completion_valid(done) || !done->count) return RC_LIFETIME_INVALID;
    v->producer_done = *done;
    v->state = RC_LIFETIME_VERSION_SUBMITTED;
    return RC_LIFETIME_OK;
}

static inline RcLifetimeResult rc_lifetime_producer_complete(RcLifetimeVersion *v, uint64_t id,
                                                           const RcLifetimeCompletion *completed) {
    if (!rc_lifetime_version_matches(v, id) || v->state != RC_LIFETIME_VERSION_SUBMITTED)
        return RC_LIFETIME_INVALID;
    if (!rc_lifetime_completion_covers(completed, &v->producer_done)) return RC_LIFETIME_BUSY;
    v->state = RC_LIFETIME_VERSION_READY;
    return RC_LIFETIME_OK;
}

// Unsent dependents must cancel/replan after this succeeds. Already submitted producers cannot be canceled.
static inline RcLifetimeResult rc_lifetime_producer_cancel(RcLifetimeVersion *v, uint64_t id) {
    if (!rc_lifetime_version_matches(v, id) || v->state != RC_LIFETIME_VERSION_PENDING)
        return RC_LIFETIME_INVALID;
    v->state = RC_LIFETIME_VERSION_CANCELED;
    return RC_LIFETIME_OK;
}

static inline RcLifetimeResult rc_lifetime_residency_install_state(RcLifetimeIds *ids, RcLifetimeVersion *v,
                                                                 uint64_t version, RcLifetimeResidency *r,
                                                                 RcLifetimeUploadState upload) {
    if (!rc_lifetime_version_matches(v, version) || v->state == RC_LIFETIME_VERSION_CANCELED || !r ||
        (r->state != RC_LIFETIME_RESIDENCY_EMPTY && r->state != RC_LIFETIME_RESIDENCY_RETIRED) ||
        (upload != RC_LIFETIME_UPLOAD_READY && upload != RC_LIFETIME_UPLOAD_PENDING) ||
        (upload == RC_LIFETIME_UPLOAD_PENDING && v->state != RC_LIFETIME_VERSION_READY))
        return RC_LIFETIME_INVALID;
    if (v->residencies == UINT32_MAX) return RC_LIFETIME_CAPACITY;
    uint64_t id;
    RcLifetimeResult result = rc_lifetime_id_next(ids, &id);
    if (result != RC_LIFETIME_OK) return result;
    *r = (RcLifetimeResidency){.binding = {version, id}, .state = RC_LIFETIME_RESIDENCY_AVAILABLE,
                             .upload = upload};
    v->residencies++;
    v->available_residencies++;
    return RC_LIFETIME_OK;
}

// This entry point requires an already populated backend image, or one populated by the version's own producer.
static inline RcLifetimeResult rc_lifetime_residency_install(RcLifetimeIds *ids, RcLifetimeVersion *v,
                                                           uint64_t version, RcLifetimeResidency *r) {
    return rc_lifetime_residency_install_state(ids, v, version, r, RC_LIFETIME_UPLOAD_READY);
}

// Reinstallation of a ready content version can still need an independent upload into its new physical instance.
// This form requires established contents; a pending content producer uses residency_install and its own dependency.
static inline RcLifetimeResult rc_lifetime_residency_install_pending(RcLifetimeIds *ids, RcLifetimeVersion *v,
                                                                   uint64_t version, RcLifetimeResidency *r) {
    return rc_lifetime_residency_install_state(ids, v, version, r, RC_LIFETIME_UPLOAD_PENDING);
}

// Detachment ends future acquisition, not existing recorded use. A future exact-version promise needs another
// available instance or an immutable recovery provider before the final available instance can be detached.
static inline RcLifetimeResult rc_lifetime_residency_detach(RcLifetimeVersion *v, RcLifetimeResidency *r,
                                                          RcLifetimeBinding binding) {
    if (!rc_lifetime_binding_matches(v, r, binding) || r->state != RC_LIFETIME_RESIDENCY_AVAILABLE ||
        !v->available_residencies) return RC_LIFETIME_INVALID;
    if (v->state != RC_LIFETIME_VERSION_CANCELED && v->exact_holds && v->available_residencies == 1 &&
        !rc_lifetime_reconstructible(v->recovery, RC_LIFETIME_EXACT)) return RC_LIFETIME_NO_RECOVERY;
    r->state = RC_LIFETIME_RESIDENCY_DETACHED;
    v->available_residencies--;
    return RC_LIFETIME_OK;
}

static inline RcLifetimeResult rc_lifetime_upload_submit(const RcLifetimeVersion *v, RcLifetimeResidency *r,
                                                        RcLifetimeBinding binding,
                                                        const RcLifetimeCompletion *done) {
    if (!rc_lifetime_binding_matches(v, r, binding) || v->state == RC_LIFETIME_VERSION_CANCELED ||
        r->upload != RC_LIFETIME_UPLOAD_PENDING || !rc_lifetime_completion_valid(done) || !done->count)
        return RC_LIFETIME_INVALID;
    r->upload_done = *done;
    r->upload = RC_LIFETIME_UPLOAD_SUBMITTED;
    return RC_LIFETIME_OK;
}

// Attach an upload dependency before publishing an installed instance. No captured use may already name it.
static inline RcLifetimeResult rc_lifetime_upload_prepare(const RcLifetimeVersion *v, RcLifetimeResidency *r,
                                                         RcLifetimeBinding binding) {
    if (!rc_lifetime_binding_matches(v, r, binding) || v->state != RC_LIFETIME_VERSION_READY ||
        r->state != RC_LIFETIME_RESIDENCY_AVAILABLE || r->upload != RC_LIFETIME_UPLOAD_READY || r->captures)
        return RC_LIFETIME_INVALID;
    r->upload = RC_LIFETIME_UPLOAD_PENDING;
    return RC_LIFETIME_OK;
}

static inline RcLifetimeResult rc_lifetime_upload_complete(const RcLifetimeVersion *v, RcLifetimeResidency *r,
                                                          RcLifetimeBinding binding,
                                                          const RcLifetimeCompletion *completed) {
    if (!rc_lifetime_binding_matches(v, r, binding) || r->upload != RC_LIFETIME_UPLOAD_SUBMITTED)
        return RC_LIFETIME_INVALID;
    if (!rc_lifetime_completion_covers(completed, &r->upload_done)) return RC_LIFETIME_BUSY;
    r->upload = RC_LIFETIME_UPLOAD_READY;
    return RC_LIFETIME_OK;
}

static inline RcLifetimeResult rc_lifetime_upload_cancel(RcLifetimeVersion *v, RcLifetimeResidency *r,
                                                        RcLifetimeBinding binding) {
    if (!rc_lifetime_binding_matches(v, r, binding) || r->upload != RC_LIFETIME_UPLOAD_PENDING)
        return RC_LIFETIME_INVALID;
    if (r->state == RC_LIFETIME_RESIDENCY_AVAILABLE) {
        RcLifetimeResult result = rc_lifetime_residency_detach(v, r, binding);
        if (result != RC_LIFETIME_OK) return result;
    }
    r->upload = RC_LIFETIME_UPLOAD_CANCELED;
    return RC_LIFETIME_OK;
}

static inline int rc_lifetime_capture_owned(const RcLifetimeCapture *capture) {
    return capture && capture->owner == (uintptr_t)capture;
}

static inline RcLifetimeResult rc_lifetime_capture_move(RcLifetimeCapture *dst, RcLifetimeCapture *src) {
    if (!dst || !rc_lifetime_capture_owned(src) ||
        (src->state != RC_LIFETIME_CAPTURE_RECORDING && src->state != RC_LIFETIME_CAPTURE_SUBMITTED) ||
        (dst->state != RC_LIFETIME_CAPTURE_EMPTY && dst->state != RC_LIFETIME_CAPTURE_RELEASED))
        return RC_LIFETIME_INVALID;
    *dst = *src;
    dst->owner = (uintptr_t)dst;
    src->owner = 0;
    src->state = RC_LIFETIME_CAPTURE_RELEASED;
    return RC_LIFETIME_OK;
}

static inline RcLifetimeResult rc_lifetime_capture_begin(RcLifetimeVersion *v, RcLifetimeResidency *r,
                                                       RcLifetimeBinding binding, RcLifetimeCapture *capture) {
    if (!rc_lifetime_binding_matches(v, r, binding) || v->state == RC_LIFETIME_VERSION_CANCELED ||
        r->state != RC_LIFETIME_RESIDENCY_AVAILABLE || r->upload == RC_LIFETIME_UPLOAD_CANCELED || !capture ||
        (capture->state != RC_LIFETIME_CAPTURE_EMPTY && capture->state != RC_LIFETIME_CAPTURE_RELEASED))
        return RC_LIFETIME_INVALID;
    if (r->captures == UINT32_MAX) return RC_LIFETIME_CAPACITY;
    *capture = (RcLifetimeCapture){.binding = binding, .state = RC_LIFETIME_CAPTURE_RECORDING,
                                  .owner = (uintptr_t)capture};
    r->captures++;
    return RC_LIFETIME_OK;
}

// A reusable descriptor publication already owns this exact instance. A new recording can retain it even after
// the current residency was detached. The source ticket must still be owned and active under the caller's lock.
static inline RcLifetimeResult rc_lifetime_capture_fork(RcLifetimeVersion *v, RcLifetimeResidency *r,
                                                        const RcLifetimeCapture *source,
                                                        RcLifetimeCapture *capture) {
    if (!rc_lifetime_capture_owned(source) ||
        (source->state != RC_LIFETIME_CAPTURE_RECORDING && source->state != RC_LIFETIME_CAPTURE_SUBMITTED) ||
        !rc_lifetime_binding_matches(v, r, source->binding) || v->state == RC_LIFETIME_VERSION_CANCELED ||
        (r->state != RC_LIFETIME_RESIDENCY_AVAILABLE && r->state != RC_LIFETIME_RESIDENCY_DETACHED) ||
        r->upload == RC_LIFETIME_UPLOAD_CANCELED || !r->captures || !capture ||
        (capture->state != RC_LIFETIME_CAPTURE_EMPTY && capture->state != RC_LIFETIME_CAPTURE_RELEASED))
        return RC_LIFETIME_INVALID;
    if (r->captures == UINT32_MAX) return RC_LIFETIME_CAPACITY;
    *capture = (RcLifetimeCapture){.binding = source->binding, .state = RC_LIFETIME_CAPTURE_RECORDING,
                                  .owner = (uintptr_t)capture};
    r->captures++;
    return RC_LIFETIME_OK;
}

// The caller supplies actual submission waits/order and completion signals. These are declarations, not Vulkan
// operations. An unsent producer cannot be a dependency of submitted work; independent domains need explicit waits.
static inline RcLifetimeResult rc_lifetime_capture_submit(const RcLifetimeVersion *v,
                                                         const RcLifetimeResidency *r,
                                                         RcLifetimeCapture *capture,
                                                         const RcLifetimeCompletion *waits,
                                                         const RcLifetimeCompletion *done) {
    if (!rc_lifetime_capture_owned(capture) || !rc_lifetime_binding_matches(v, r, capture->binding) ||
        capture->state != RC_LIFETIME_CAPTURE_RECORDING || !r->captures ||
        !rc_lifetime_completion_valid(done) || !done->count) return RC_LIFETIME_INVALID;
    if (v->state == RC_LIFETIME_VERSION_CANCELED || v->state == RC_LIFETIME_VERSION_PENDING ||
        r->upload == RC_LIFETIME_UPLOAD_PENDING || r->upload == RC_LIFETIME_UPLOAD_CANCELED)
        return RC_LIFETIME_DEPENDENCY;
    if (v->state == RC_LIFETIME_VERSION_SUBMITTED &&
        !rc_lifetime_completion_covers(waits, &v->producer_done)) return RC_LIFETIME_DEPENDENCY;
    if (r->upload == RC_LIFETIME_UPLOAD_SUBMITTED &&
        !rc_lifetime_completion_covers(waits, &r->upload_done)) return RC_LIFETIME_DEPENDENCY;
    capture->done = *done;
    capture->state = RC_LIFETIME_CAPTURE_SUBMITTED;
    return RC_LIFETIME_OK;
}

static inline RcLifetimeResult rc_lifetime_capture_cancel(const RcLifetimeVersion *v, RcLifetimeResidency *r,
                                                         RcLifetimeCapture *capture) {
    if (!rc_lifetime_capture_owned(capture) || !rc_lifetime_binding_matches(v, r, capture->binding) ||
        capture->state != RC_LIFETIME_CAPTURE_RECORDING || !r->captures) return RC_LIFETIME_INVALID;
    r->captures--;
    capture->state = RC_LIFETIME_CAPTURE_RELEASED;
    return RC_LIFETIME_OK;
}

static inline RcLifetimeResult rc_lifetime_capture_complete(const RcLifetimeVersion *v, RcLifetimeResidency *r,
                                                           RcLifetimeCapture *capture,
                                                           const RcLifetimeCompletion *completed) {
    if (!rc_lifetime_capture_owned(capture) || !rc_lifetime_binding_matches(v, r, capture->binding) ||
        capture->state != RC_LIFETIME_CAPTURE_SUBMITTED || !r->captures) return RC_LIFETIME_INVALID;
    if (!rc_lifetime_completion_covers(completed, &capture->done)) return RC_LIFETIME_BUSY;
    r->captures--;
    capture->state = RC_LIFETIME_CAPTURE_RELEASED;
    return RC_LIFETIME_OK;
}

// Success transfers permission to destroy the backend object; actual memory accounting/destruction is external.
// Even a detached image without readers can still be the destination of its pending/submitted producer.
static inline RcLifetimeResult rc_lifetime_residency_retire(RcLifetimeVersion *v, RcLifetimeResidency *r,
                                                          RcLifetimeBinding binding) {
    if (!rc_lifetime_binding_matches(v, r, binding) || r->state != RC_LIFETIME_RESIDENCY_DETACHED ||
        !v->residencies) return RC_LIFETIME_INVALID;
    if (r->captures || v->state == RC_LIFETIME_VERSION_PENDING || v->state == RC_LIFETIME_VERSION_SUBMITTED ||
        r->upload == RC_LIFETIME_UPLOAD_PENDING || r->upload == RC_LIFETIME_UPLOAD_SUBMITTED)
        return RC_LIFETIME_BUSY;
    r->state = RC_LIFETIME_RESIDENCY_RETIRED;
    v->residencies--;
    return RC_LIFETIME_OK;
}

static inline RcLifetimeResult rc_lifetime_version_retire(RcLifetimeVersion *v, uint64_t id) {
    if (!rc_lifetime_version_matches(v, id)) return RC_LIFETIME_INVALID;
    if (v->exact_holds || v->residencies || v->state == RC_LIFETIME_VERSION_PENDING ||
        v->state == RC_LIFETIME_VERSION_SUBMITTED) return RC_LIFETIME_BUSY;
    v->state = RC_LIFETIME_VERSION_RETIRED;
    return RC_LIFETIME_OK;
}

#endif

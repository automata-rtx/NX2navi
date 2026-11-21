// Resource-core observation seam. The caller serializes each domain and pins it while using these values.
// No OS reads, allocations or callbacks occur here. An observation is not an atomic guest-memory snapshot.
#pragma once
#include <stdint.h>
#include <string.h>

typedef struct { uint64_t backing, offset, size; } RcSourceRange;
// A mapping observation carries its logical address separately from canonical byte coverage.
typedef struct { uint64_t address, mapping; RcSourceRange range; } RcInterval;
typedef enum { RC_AUTH_CPU, RC_AUTH_GPU, RC_AUTH_COMPATIBILITY } RcAuthority;
typedef struct {
    uint64_t incarnation, revision, scope, next_transition, transition;
    RcSourceRange range;
    RcAuthority authority;
    int finalized, exhausted;
} RcSourceDomain;
typedef struct {
    uint64_t incarnation, revision, scope;
    RcSourceRange range;
} RcObservation;
typedef struct { uint64_t incarnation, transition, revision; } RcAuthorityTransfer;

static inline int rc_range_valid(RcSourceRange r) {
    return r.backing && r.size && r.offset <= UINT64_MAX - r.size;
}
static inline int rc_range_equal(RcSourceRange a, RcSourceRange b) {
    return a.backing == b.backing && a.offset == b.offset && a.size == b.size;
}
static inline int rc_range_overlap(RcSourceRange a, RcSourceRange b) {
    return rc_range_valid(a) && rc_range_valid(b) && a.backing == b.backing &&
        a.offset < b.offset + b.size && b.offset < a.offset + a.size;
}
// The registry supplies a never-reused incarnation; this object cannot generate cross-registry identity.
static inline int rc_source_init(RcSourceDomain *d, uint64_t incarnation, RcSourceRange range) {
    if (!d || !incarnation || !rc_range_valid(range)) return 0;
    *d = (RcSourceDomain){.incarnation = incarnation, .revision = 1, .scope = 1,
        .range = range, .authority = RC_AUTH_CPU};
    return 1;
}
static inline int rc_source_advance(RcSourceDomain *d) {
    if (d->exhausted || d->revision == UINT64_MAX) { d->exhausted = 1; return 0; }
    ++d->revision;
    return 1;
}
// Unknown coverage invalidates the complete domain; known unrelated writes do not disturb it.
static inline int rc_source_changed(RcSourceDomain *d, RcSourceRange written, int known) {
    if (known && !rc_range_valid(written)) return 0;
    if (known && !rc_range_overlap(d->range, written)) return 1;
    if (d->authority == RC_AUTH_GPU) d->authority = RC_AUTH_COMPATIBILITY;
    return rc_source_advance(d);
}
static inline int rc_source_new_scope(RcSourceDomain *d) {
    if (d->exhausted || d->scope == UINT64_MAX) { d->exhausted = 1; return 0; }
    ++d->scope;
    return 1;
}
static inline int rc_source_observe(const RcSourceDomain *d, RcObservation *o) {
    *o = (RcObservation){0};
    if (!d->incarnation || d->exhausted || d->finalized || d->transition || d->authority != RC_AUTH_CPU) return 0;
    *o = (RcObservation){d->incarnation, d->revision, d->scope, d->range};
    return 1;
}
static inline int rc_observation_current(const RcSourceDomain *d, const RcObservation *o) {
    RcObservation now;
    return rc_source_observe(d, &now) && now.incarnation == o->incarnation &&
        now.revision == o->revision && now.scope == o->scope && rc_range_equal(now.range, o->range);
}
// The caller must perform actual fresh source observation before certifying a new scope. This function cannot
// infer whether uninstrumented guest writes occurred; supplying that evidence belongs to the memory adapter.
static inline int rc_observation_revalidate(const RcSourceDomain *d, const RcObservation *old,
                                            int source_unchanged, RcObservation *out) {
    *out = (RcObservation){0};
    if (!source_unchanged || d->incarnation != old->incarnation || d->revision != old->revision ||
        !rc_range_equal(d->range, old->range)) return 0;
    return rc_source_observe(d, out);
}
static inline void rc_source_finalize(RcSourceDomain *d) {
    // Finalize this domain only when its last live association ends, not when one alias is released.
    d->finalized = 1;
    rc_source_advance(d); // Exact old versions are owned by lifetime captures, not this current-source association.
}
// A writer or compatibility handoff reserves authority before recording/issuing its operation. No stable CPU
// observation is possible until completion/cancellation. Disjoint domains can progress independently.
static inline int rc_authority_begin(RcSourceDomain *d, RcAuthorityTransfer *t) {
    if (!d || !t || t->transition || d->exhausted || d->transition || d->finalized ||
        d->next_transition == UINT64_MAX) return 0;
    if (!rc_source_advance(d)) return 0;
    d->transition = ++d->next_transition;
    *t = (RcAuthorityTransfer){d->incarnation, d->transition, d->revision};
    return 1;
}
// Commit requires the adapter to have preserved/transferred contents AND completion dependencies. Cancel retains
// previous authority but never resurrects old observation tokens. Lifetime/producer cancellation is a separate step.
static inline int rc_authority_end(RcSourceDomain *d, RcAuthorityTransfer *t, RcAuthority owner, int commit) {
    if (!t || !t->transition || t->incarnation != d->incarnation || t->transition != d->transition ||
        (commit && owner != RC_AUTH_CPU && owner != RC_AUTH_GPU && owner != RC_AUTH_COMPATIBILITY)) return 0;
    // An intervening write invalidates exclusive ownership. The adapter must reconcile mixed contents through
    // compatibility authority; it cannot overwrite the newer evidence with the handoff's original assumption.
    if (commit && owner != RC_AUTH_COMPATIBILITY && d->revision != t->revision) return 0;
    if (commit) d->authority = owner;
    else if (d->revision != t->revision) d->authority = RC_AUTH_COMPATIBILITY;
    d->transition = 0;
    *t = (RcAuthorityTransfer){0};
    rc_source_advance(d);
    return 1;
}

typedef enum { RC_PROVIDER_CAPTURE, RC_PROVIDER_PREPARED_FILE, RC_PROVIDER_CONDITIONAL_GUEST } RcProviderKind;
typedef struct {
    uint64_t content, representation, policy, payload;
    RcProviderKind kind;
} RcPreparedInput;
// IDs are opaque provider-owned identities. Equality never infers content from an address or dimensions.
static inline int rc_input_exact(const RcPreparedInput *p) {
    return p->content && p->representation && p->policy && p->payload &&
        (p->kind == RC_PROVIDER_CAPTURE || p->kind == RC_PROVIDER_PREPARED_FILE);
}

static inline int rc_input_equal(const RcPreparedInput *a, const RcPreparedInput *b) {
    return a->content == b->content && a->representation == b->representation && a->policy == b->policy &&
        a->payload == b->payload && a->kind == b->kind;
}

// Capacity ownership for preparation and allocation. The caller serializes every operation on one ledger.
// This module makes no allocation, samples no telemetry and chooses no victims or scheduling policy.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define RC_ADMISSION_MAX_DOMAINS 8
#define RC_ADMISSION_NO_SLOT UINT32_MAX

typedef enum {
    RC_ADMISSION_OK,
    RC_ADMISSION_REFUSED,
    RC_ADMISSION_INVALID,
    RC_ADMISSION_UNSUPPORTED,
    RC_ADMISSION_NO_RECORDS,
    RC_ADMISSION_STALE,
    RC_ADMISSION_NOT_OWNER,
    RC_ADMISSION_WRONG_STATE,
    RC_ADMISSION_OVERFLOW,
    RC_ADMISSION_GENERATION_EXHAUSTED,
    // The actual allocation is accounted, even though the ledger is now above a domain's limit. This is a
    // successful accounting operation, not permission to make another allocation or ignore the pressure.
    RC_ADMISSION_ACCOUNTED_PRESSURE
} RcAdmissionStatus;

typedef enum {
    RC_ADMISSION_RAM,
    RC_ADMISSION_COMMIT,
    RC_ADMISSION_GPU_HEAP,
    RC_ADMISSION_STAGING
} RcAdmissionDomainKind;

typedef struct {
    RcAdmissionDomainKind kind;
    uint64_t id;
    // Total allowance for this ledger's outstanding, retained and pending charges, after external usage and
    // policy headroom. This is not raw free memory from a sample that already subtracts the tracked allocations.
    uint64_t limit;
} RcAdmissionDomain;

// Components are independent constraints, never summed. Host staging may constrain RAM, commit and its arena
// capacity simultaneously; the staging component is not another physical byte charge. A GPU component names
// one chosen eligible heap, not total VRAM. Memory types on the same heap share that heap's capacity domain.
// The adapter validates placement compatibility before reservation. Unused components must be zero.
typedef struct { uint64_t amount[RC_ADMISSION_MAX_DOMAINS]; } RcAdmissionVector;

typedef enum {
    RC_ADMISSION_EMPTY,
    RC_ADMISSION_RESERVED,
    RC_ADMISSION_RETAINED,
    RC_ADMISSION_PENDING
} RcAdmissionState;

typedef struct {
    RcAdmissionVector amount;
    uint64_t generation, owner;
    uint32_t next_free;
    RcAdmissionState state;
} RcAdmissionRecord;

typedef struct RcAdmission {
    RcAdmissionDomain domain[RC_ADMISSION_MAX_DOMAINS];
    RcAdmissionVector reserved, retained, pending;
    RcAdmissionRecord *records;
    uint64_t incarnation, next_generation;
    uint32_t domain_count, record_count, free_head;
} RcAdmission;

typedef struct {
    const RcAdmission *ledger;
    uint64_t incarnation, generation;
    uint32_t slot;
    // State changes invalidate old tokens without consuming generations. Admitted work can still settle and
    // retire after the generation sequence is exhausted; only new records and ownership handoffs are refused.
    RcAdmissionState state;
} RcAdmissionToken;

// The ledger and its records stay at fixed addresses. Reinitialization requires no surviving obligations and
// a new nonzero incarnation supplied by the owner. Slot generations never wrap within that incarnation.
static inline RcAdmissionStatus rc_admission_init(RcAdmission *b, uint64_t incarnation,
                                                 const RcAdmissionDomain *domains, uint32_t domain_count,
                                                 RcAdmissionRecord *records, uint32_t record_count) {
    if (!b || !incarnation || !domains || !records || !domain_count || !record_count)
        return RC_ADMISSION_INVALID;
    if (domain_count > RC_ADMISSION_MAX_DOMAINS) return RC_ADMISSION_UNSUPPORTED;
    if ((size_t)record_count > SIZE_MAX / sizeof *records) return RC_ADMISSION_OVERFLOW;
    for (uint32_t i = 0; i < domain_count; i++) {
        if (domains[i].kind < RC_ADMISSION_RAM || domains[i].kind > RC_ADMISSION_STAGING)
            return RC_ADMISSION_UNSUPPORTED;
        for (uint32_t j = 0; j < i; j++)
            if (domains[i].kind == domains[j].kind && domains[i].id == domains[j].id)
                return RC_ADMISSION_INVALID;
    }
    RcAdmissionDomain specs[RC_ADMISSION_MAX_DOMAINS];
    memcpy(specs, domains, (size_t)domain_count * sizeof *domains);
    memset(b, 0, sizeof *b);
    memcpy(b->domain, specs, (size_t)domain_count * sizeof *domains);
    b->incarnation = incarnation;
    b->domain_count = domain_count;
    b->record_count = record_count;
    b->records = records;
    b->free_head = 0;
    memset(records, 0, (size_t)record_count * sizeof *records);
    for (uint32_t i = 0; i < record_count; i++)
        records[i].next_free = i + 1 < record_count ? i + 1 : RC_ADMISSION_NO_SLOT;
    return RC_ADMISSION_OK;
}

static inline int rc_admission_accounted(RcAdmissionStatus status) {
    return status == RC_ADMISSION_OK || status == RC_ADMISSION_ACCOUNTED_PRESSURE;
}

static inline RcAdmissionStatus rc_admission_vector_valid(const RcAdmission *b, const RcAdmissionVector *v,
                                                         int allow_zero) {
    if (!b || !b->incarnation || !v) return RC_ADMISSION_INVALID;
    int any = 0;
    for (uint32_t i = 0; i < RC_ADMISSION_MAX_DOMAINS; i++) {
        if (i >= b->domain_count && v->amount[i]) return RC_ADMISSION_UNSUPPORTED;
        any |= v->amount[i] != 0;
    }
    return any || allow_zero ? RC_ADMISSION_OK : RC_ADMISSION_INVALID;
}

// The sum stays representable even if policy later lowers a limit below the existing charges.
static inline uint64_t rc_admission_used_at(const RcAdmission *b, uint32_t domain) {
    return b->reserved.amount[domain] + b->retained.amount[domain] + b->pending.amount[domain];
}

static inline RcAdmissionStatus rc_admission_available(const RcAdmission *b, uint32_t domain, uint64_t *out) {
    if (!b || !b->incarnation || !out) return RC_ADMISSION_INVALID;
    if (domain >= b->domain_count) return RC_ADMISSION_UNSUPPORTED;
    uint64_t used = rc_admission_used_at(b, domain), limit = b->domain[domain].limit;
    *out = used < limit ? limit - used : 0;
    return RC_ADMISSION_OK;
}

static inline RcAdmissionStatus rc_admission_set_limit(RcAdmission *b, uint32_t domain, uint64_t limit) {
    if (!b || !b->incarnation) return RC_ADMISSION_INVALID;
    if (domain >= b->domain_count) return RC_ADMISSION_UNSUPPORTED;
    b->domain[domain].limit = limit;
    return RC_ADMISSION_OK;
}

static inline RcAdmissionStatus rc_admission_lookup(RcAdmission *b, RcAdmissionToken token, uint64_t owner,
                                                   RcAdmissionRecord **out) {
    if (!b || !b->incarnation || !owner || !out) return RC_ADMISSION_INVALID;
    if (token.ledger != b || token.incarnation != b->incarnation || token.slot >= b->record_count)
        return RC_ADMISSION_STALE;
    RcAdmissionRecord *r = &b->records[token.slot];
    if (r->state == RC_ADMISSION_EMPTY || r->state != token.state || r->generation != token.generation)
        return RC_ADMISSION_STALE;
    if (r->owner != owner) return RC_ADMISSION_NOT_OWNER;
    *out = r;
    return RC_ADMISSION_OK;
}

static inline RcAdmissionToken rc_admission_token(const RcAdmission *b, uint32_t slot) {
    return (RcAdmissionToken){b, b->incarnation, b->records[slot].generation, slot, b->records[slot].state};
}

static inline RcAdmissionStatus rc_admission_new_check(const RcAdmission *b) {
    if (b->free_head == RC_ADMISSION_NO_SLOT) return RC_ADMISSION_NO_RECORDS;
    if (b->next_generation == UINT64_MAX) return RC_ADMISSION_GENERATION_EXHAUSTED;
    return RC_ADMISSION_OK;
}

// All capacity and handle checks precede this helper, so publication cannot fail halfway through a vector.
static inline RcAdmissionToken rc_admission_new(RcAdmission *b, uint64_t owner, RcAdmissionState state,
                                               const RcAdmissionVector *amount) {
    uint32_t slot = b->free_head;
    RcAdmissionRecord *r = &b->records[slot];
    b->free_head = r->next_free;
    *r = (RcAdmissionRecord){.amount = *amount, .generation = ++b->next_generation, .owner = owner,
                             .next_free = RC_ADMISSION_NO_SLOT, .state = state};
    return rc_admission_token(b, slot);
}

static inline void rc_admission_free(RcAdmission *b, uint32_t slot) {
    b->records[slot] = (RcAdmissionRecord){.next_free = b->free_head};
    b->free_head = slot;
}

static inline RcAdmissionStatus rc_admission_pressure(const RcAdmission *b) {
    for (uint32_t i = 0; i < b->domain_count; i++)
        if (rc_admission_used_at(b, i) > b->domain[i].limit) return RC_ADMISSION_ACCOUNTED_PRESSURE;
    return RC_ADMISSION_OK;
}

// Reserve the whole incremental peak before work begins. Refusal, unsupported components, full record storage
// and generation exhaustion leave both the ledger and *out untouched. Owners are nonzero caller-chosen IDs.
static inline RcAdmissionStatus rc_admission_reserve(RcAdmission *b, uint64_t owner,
                                                    const RcAdmissionVector *peak, RcAdmissionToken *out) {
    if (!owner || !out) return RC_ADMISSION_INVALID;
    RcAdmissionStatus s = rc_admission_vector_valid(b, peak, 0);
    if (s != RC_ADMISSION_OK) return s;
    for (uint32_t i = 0; i < b->domain_count; i++) {
        uint64_t used = rc_admission_used_at(b, i), limit = b->domain[i].limit;
        if (peak->amount[i] && (used > limit || peak->amount[i] > limit - used)) return RC_ADMISSION_REFUSED;
    }
    s = rc_admission_new_check(b);
    if (s != RC_ADMISSION_OK) return s;
    RcAdmissionVector amount = *peak;
    RcAdmissionToken token = rc_admission_new(b, owner, RC_ADMISSION_RESERVED, &amount);
    for (uint32_t i = 0; i < b->domain_count; i++) b->reserved.amount[i] += amount.amount[i];
    *out = token;
    return RC_ADMISSION_OK;
}

// Partition an already guaranteed peak before starting an allocation or handing off a stage. The remainder
// stays on the parent ticket, including any drain reservation. No capacity is created, even after limits shrink.
// Split before backend allocation so record exhaustion cannot leave an allocation without an accounting record.
static inline RcAdmissionStatus rc_admission_split(RcAdmission *b, RcAdmissionToken ticket, uint64_t owner,
                                                  const RcAdmissionVector *portion, uint64_t next_owner,
                                                  RcAdmissionToken *out) {
    if (!next_owner || !out) return RC_ADMISSION_INVALID;
    RcAdmissionStatus s = rc_admission_vector_valid(b, portion, 0);
    if (s != RC_ADMISSION_OK) return s;
    RcAdmissionRecord *r;
    s = rc_admission_lookup(b, ticket, owner, &r);
    if (s != RC_ADMISSION_OK) return s;
    if (r->state != RC_ADMISSION_RESERVED) return RC_ADMISSION_WRONG_STATE;
    for (uint32_t i = 0; i < b->domain_count; i++)
        if (portion->amount[i] > r->amount.amount[i]) return RC_ADMISSION_REFUSED;
    s = rc_admission_new_check(b);
    if (s != RC_ADMISSION_OK) return s;
    RcAdmissionVector amount = *portion;
    RcAdmissionToken token = rc_admission_new(b, next_owner, RC_ADMISSION_RESERVED, &amount);
    for (uint32_t i = 0; i < b->domain_count; i++) r->amount.amount[i] -= amount.amount[i];
    *out = token;
    return RC_ADMISSION_OK;
}

// Record actual existing allocations even when an estimate or externally changed limit was insufficient.
// These accounting operations cannot authorize growth. A caller must first reserve before allocating; adopt
// is for baseline/externally observed allocations. Overflow fails atomically rather than hiding an excess.
static inline RcAdmissionStatus rc_admission_replace_check(const RcAdmission *b,
                                                           const RcAdmissionVector *old,
                                                           const RcAdmissionVector *actual) {
    for (uint32_t i = 0; i < b->domain_count; i++) {
        uint64_t base = rc_admission_used_at(b, i) - old->amount[i];
        if (actual->amount[i] > UINT64_MAX - base) return RC_ADMISSION_OVERFLOW;
    }
    return RC_ADMISSION_OK;
}

// Successful backend allocation turns its reservation into retained capacity. A smaller actual footprint
// returns only unused reservation; actual allocated capacity stays charged. The old ticket becomes stale.
// On backend failure cancel its ticket instead. Use split when another stage still needs the parent guarantee.
static inline RcAdmissionStatus rc_admission_materialize(RcAdmission *b, RcAdmissionToken ticket, uint64_t owner,
                                                        const RcAdmissionVector *actual, RcAdmissionToken *out) {
    if (!out) return RC_ADMISSION_INVALID;
    RcAdmissionStatus s = rc_admission_vector_valid(b, actual, 0);
    if (s != RC_ADMISSION_OK) return s;
    RcAdmissionRecord *r;
    s = rc_admission_lookup(b, ticket, owner, &r);
    if (s != RC_ADMISSION_OK) return s;
    if (r->state != RC_ADMISSION_RESERVED) return RC_ADMISSION_WRONG_STATE;
    s = rc_admission_replace_check(b, &r->amount, actual);
    if (s != RC_ADMISSION_OK) return s;
    RcAdmissionVector amount = *actual;
    for (uint32_t i = 0; i < b->domain_count; i++) {
        b->reserved.amount[i] -= r->amount.amount[i];
        b->retained.amount[i] += amount.amount[i];
    }
    r->amount = amount;
    r->state = RC_ADMISSION_RETAINED;
    *out = rc_admission_token(b, ticket.slot);
    return rc_admission_pressure(b);
}

static inline RcAdmissionStatus rc_admission_adopt(RcAdmission *b, uint64_t owner,
                                                  const RcAdmissionVector *actual, RcAdmissionToken *out) {
    if (!owner || !out) return RC_ADMISSION_INVALID;
    RcAdmissionStatus s = rc_admission_vector_valid(b, actual, 0);
    if (s != RC_ADMISSION_OK) return s;
    s = rc_admission_new_check(b);
    if (s != RC_ADMISSION_OK) return s;
    RcAdmissionVector zero = {{0}};
    s = rc_admission_replace_check(b, &zero, actual);
    if (s != RC_ADMISSION_OK) return s;
    RcAdmissionVector amount = *actual;
    *out = rc_admission_new(b, owner, RC_ADMISSION_RETAINED, &amount);
    for (uint32_t i = 0; i < b->domain_count; i++) b->retained.amount[i] += amount.amount[i];
    return rc_admission_pressure(b);
}

// Correct an opaque allocation's measured footprint. Decreases must be real measurement corrections or actual
// completed partial destruction, never anticipated frees. Pending objects retain their last actual charge.
static inline RcAdmissionStatus rc_admission_remeasure(RcAdmission *b, RcAdmissionToken charge, uint64_t owner,
                                                       const RcAdmissionVector *actual) {
    RcAdmissionStatus s = rc_admission_vector_valid(b, actual, 0);
    if (s != RC_ADMISSION_OK) return s;
    RcAdmissionRecord *r;
    s = rc_admission_lookup(b, charge, owner, &r);
    if (s != RC_ADMISSION_OK) return s;
    if (r->state != RC_ADMISSION_RETAINED) return RC_ADMISSION_WRONG_STATE;
    s = rc_admission_replace_check(b, &r->amount, actual);
    if (s != RC_ADMISSION_OK) return s;
    RcAdmissionVector amount = *actual;
    for (uint32_t i = 0; i < b->domain_count; i++)
        b->retained.amount[i] = b->retained.amount[i] - r->amount.amount[i] + amount.amount[i];
    r->amount = amount;
    return rc_admission_pressure(b);
}

static inline RcAdmissionStatus rc_admission_handoff(RcAdmission *b, RcAdmissionToken token, uint64_t owner,
                                                    uint64_t next_owner, RcAdmissionToken *out) {
    if (!next_owner || !out) return RC_ADMISSION_INVALID;
    RcAdmissionRecord *r;
    RcAdmissionStatus s = rc_admission_lookup(b, token, owner, &r);
    if (s != RC_ADMISSION_OK) return s;
    if (b->next_generation == UINT64_MAX) return RC_ADMISSION_GENERATION_EXHAUSTED;
    r->owner = next_owner;
    r->generation = ++b->next_generation;
    *out = rc_admission_token(b, token.slot);
    return RC_ADMISSION_OK;
}

// Cancellation releases only the still-unspent reservation. Split/materialized child records are independent
// obligations and survive cancellation of their former parent's remainder.
static inline RcAdmissionStatus rc_admission_cancel(RcAdmission *b, RcAdmissionToken ticket, uint64_t owner) {
    RcAdmissionRecord *r;
    RcAdmissionStatus s = rc_admission_lookup(b, ticket, owner, &r);
    if (s != RC_ADMISSION_OK) return s;
    if (r->state != RC_ADMISSION_RESERVED) return RC_ADMISSION_WRONG_STATE;
    for (uint32_t i = 0; i < b->domain_count; i++) b->reserved.amount[i] -= r->amount.amount[i];
    rc_admission_free(b, ticket.slot);
    return RC_ADMISSION_OK;
}

// Retirement is bookkeeping, not memory relief. The same vector remains charged until actual destruction.
// The owner separately enforces CPU leases, upload and GPU completion before calling destroyed.
static inline RcAdmissionStatus rc_admission_retire(RcAdmission *b, RcAdmissionToken charge, uint64_t owner,
                                                   RcAdmissionToken *out) {
    if (!out) return RC_ADMISSION_INVALID;
    RcAdmissionRecord *r;
    RcAdmissionStatus s = rc_admission_lookup(b, charge, owner, &r);
    if (s != RC_ADMISSION_OK) return s;
    if (r->state != RC_ADMISSION_RETAINED) return RC_ADMISSION_WRONG_STATE;
    for (uint32_t i = 0; i < b->domain_count; i++) {
        b->retained.amount[i] -= r->amount.amount[i];
        b->pending.amount[i] += r->amount.amount[i];
    }
    r->state = RC_ADMISSION_PENDING;
    *out = rc_admission_token(b, charge.slot);
    return RC_ADMISSION_OK;
}

// Call only after the allocation/capacity was actually released. Driver-retained blocks can still constrain
// limits supplied by policy; releasing a suballocation does not claim external RAM or VRAM telemetry fell.
static inline RcAdmissionStatus rc_admission_destroyed(RcAdmission *b, RcAdmissionToken pending, uint64_t owner) {
    RcAdmissionRecord *r;
    RcAdmissionStatus s = rc_admission_lookup(b, pending, owner, &r);
    if (s != RC_ADMISSION_OK) return s;
    if (r->state != RC_ADMISSION_PENDING) return RC_ADMISSION_WRONG_STATE;
    for (uint32_t i = 0; i < b->domain_count; i++) b->pending.amount[i] -= r->amount.amount[i];
    rc_admission_free(b, pending.slot);
    return RC_ADMISSION_OK;
}

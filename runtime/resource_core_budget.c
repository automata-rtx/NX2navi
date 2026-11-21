#include "resource_core_budget.h"

#ifndef RC_BUDGET_TEST
#include "platform/platform.h"
#include "vk.h"
#else
uint64_t rc_budget_test_now_ns(void);
void rc_budget_test_lock(void);
void rc_budget_test_unlock(void);
void rc_budget_test_sample(RcBudgetSample *out);
void rc_budget_test_wait(uint32_t milliseconds);
#endif

#define BUDGET_RECORDS 32768u
#define MIB (1024ull * 1024ull)
#define SAMPLE_NS 100000000ull
#define REFRESH_RETRY_NS 1000000ull
#define STALE_NS 1000000000ull
#define RECLAIM_PROVIDERS 64u

static struct {
    RcAdmission ledger;
    RcAdmissionRecord records[BUDGET_RECORDS];
    RcBudgetSample sample;
    RcAdmissionVector unobserved;
    RcBudgetProducerStats producer[RC_BUDGET_PRODUCERS];
    uint64_t sampled_at, good_at[RC_ADMISSION_MAX_DOMAINS], refreshes;
    RcBudgetAvailability availability;
    struct {
        RcBudgetReclaimFn fn;
        void *context;
        uint32_t domains;
        RcBudgetProducer producer;
        int required_only;
    } reclaimer[RECLAIM_PROVIDERS];
    uint32_t reclaimer_count;
    uint64_t event;
    int servicing;
    int sampled;
#ifndef RC_BUDGET_TEST
    PlatMutex lock;
    PlatCond changed;
#endif
} B;

static uint64_t add_sat(uint64_t a, uint64_t b) { return b > UINT64_MAX - a ? UINT64_MAX : a + b; }
static uint64_t clamp_bytes(uint64_t n, uint64_t lo, uint64_t hi) { return n < lo ? lo : n > hi ? hi : n; }
static int producer_valid(RcBudgetProducer p) { return p > 0 && p < RC_BUDGET_PRODUCERS; }

#ifdef RC_BUDGET_TEST
static void budget_lock(void) { rc_budget_test_lock(); }
static void budget_unlock(void) { rc_budget_test_unlock(); }
static uint64_t budget_now(void) { return rc_budget_test_now_ns(); }
static void budget_changed(void) { if (B.event != UINT64_MAX) B.event++; }
static void platform_sample(RcBudgetSample *out) { rc_budget_test_sample(out); }
#else
static void budget_lock(void) { plat_mutex_lock(&B.lock); }
static void budget_unlock(void) { plat_mutex_unlock(&B.lock); }
static uint64_t budget_now(void) { return plat_now_ns(); }
static void budget_changed(void) {
    if (B.event != UINT64_MAX) B.event++;
    plat_cond_broadcast(&B.changed);
}
static uint64_t from_mb(double mb) {
    if (!(mb > 0)) return 0;
    return mb >= (double)UINT64_MAX / MIB ? UINT64_MAX : (uint64_t)(mb * MIB);
}

static void platform_sample(RcBudgetSample *out) {
    *out = (RcBudgetSample){0};
    float ram_used, ram_total, resident;
    double proc_commit, commit_used, commit_total;
    plat_mem_info(&ram_used, &ram_total, &resident);
    plat_commit_info(&proc_commit, &commit_used, &commit_total);
    out->process_commit = from_mb(proc_commit);
    out->process_resident = from_mb(resident);
    if (ram_used >= 0 && ram_total > 0) {
        out->known_mask |= 1u << RC_BUDGET_RAM;
        out->total.amount[RC_BUDGET_RAM] = from_mb(ram_total);
        out->free.amount[RC_BUDGET_RAM] = from_mb((double)ram_total - ram_used);
        out->known_mask |= 1u << RC_BUDGET_STAGING;
        // A soft target for optional occupied spans, not a physical capacity or a required-upload queue cap.
        out->total.amount[RC_BUDGET_STAGING] =
            clamp_bytes(out->free.amount[RC_BUDGET_RAM] / 4, 32 * MIB, 512 * MIB);
        out->free.amount[RC_BUDGET_STAGING] = out->total.amount[RC_BUDGET_STAGING];
    }
    if (commit_used >= 0 && commit_total > 0) {
        out->known_mask |= 1u << RC_BUDGET_COMMIT;
        out->total.amount[RC_BUDGET_COMMIT] = from_mb(commit_total);
        out->free.amount[RC_BUDGET_COMMIT] = from_mb(commit_total - commit_used);
    }
    VkCtx *c = vk_ctx();
    if (c && c->has_memory_budget) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budgets = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 memory = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &budgets};
        vkGetPhysicalDeviceMemoryProperties2(c->phys, &memory);
        for (uint32_t h = 0; h < memory.memoryProperties.memoryHeapCount && h < RC_BUDGET_GPU_HEAPS; h++) {
            if (!budgets.heapBudget[h]) continue;
            uint32_t d = RC_BUDGET_GPU_FIRST + h;
            out->known_mask |= 1u << d;
            out->total.amount[d] = budgets.heapBudget[h];
            out->free.amount[d] = budgets.heapUsage[h] < budgets.heapBudget[h]
                ? budgets.heapBudget[h] - budgets.heapUsage[h] : 0;
        }
    }
    // No VK_EXT_memory_budget is an unknown GPU allowance, not the installed heap size treated as free VRAM.
}
#endif

static void init_locked(void) {
    if (B.ledger.incarnation) return;
    RcAdmissionDomain domains[RC_ADMISSION_MAX_DOMAINS] = {
        {RC_ADMISSION_RAM, 0, 0}, {RC_ADMISSION_COMMIT, 0, 0}, {RC_ADMISSION_STAGING, 0, 0}
    };
    for (uint32_t h = 0; h < RC_BUDGET_GPU_HEAPS; h++)
        domains[RC_BUDGET_GPU_FIRST + h] = (RcAdmissionDomain){RC_ADMISSION_GPU_HEAP, h, 0};
    (void)rc_admission_init(&B.ledger, 1, domains, RC_ADMISSION_MAX_DOMAINS, B.records, BUDGET_RECORDS);
}

static uint64_t margin_at(uint32_t domain, int required) {
    uint64_t total = B.sample.total.amount[domain];
    if (domain == RC_BUDGET_STAGING) return 0;
    if (domain == RC_BUDGET_RAM)
        return required ? clamp_bytes(total / 64, 128 * MIB, 512 * MIB)
                        : clamp_bytes(total / 8, 512 * MIB, 2048 * MIB);
    if (domain == RC_BUDGET_COMMIT)
        return required ? clamp_bytes(total / 128, 128 * MIB, 512 * MIB)
                        : clamp_bytes(total / 16, 512 * MIB, 4096 * MIB);
    return required ? clamp_bytes(total / 128, 32 * MIB, 128 * MIB)
                    : clamp_bytes(total / 16, 128 * MIB, 1024 * MIB);
}

// Known allocated bytes already appear in the usage sample. Add them to the ledger's total allowance so they
// are not subtracted twice, then subtract every allocation/opaque completion since that sample. Destruction
// lowers both the charge and allowance equally until a new sample demonstrates physical relief. A new allocation
// never spends a scheduled free, a driver's retained block or another heap's slack.
static uint64_t limit_at(uint32_t domain, int required) {
    if (domain == RC_BUDGET_STAGING)
        return required ? UINT64_MAX : B.sample.total.amount[domain];
    uint64_t tracked = B.ledger.retained.amount[domain] + B.ledger.pending.amount[domain];
    uint64_t room;
    if (!(B.sample.known_mask & (1u << domain))) {
        // Temporary CPU telemetry failure has a bounded, accounted progress allowance. Debt survives failed
        // samples. Optional work and unknown GPU placement have no fabricated headroom.
        room = required && domain <= RC_BUDGET_COMMIT ? 128 * MIB : 0;
    } else {
        uint64_t margin = margin_at(domain, required), free = B.sample.free.amount[domain];
        room = free > margin ? free - margin : 0;
    }
    uint64_t limit = add_sat(room, tracked);
    return limit > B.unobserved.amount[domain] ? limit - B.unobserved.amount[domain] : 0;
}

static void limits_locked(int required) {
    for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++)
        (void)rc_admission_set_limit(&B.ledger, d, limit_at(d, required));
}

static void sample_locked(int force) {
    uint64_t now = budget_now();
    if (force != 2 && B.sampled && now - B.sampled_at < (force ? REFRESH_RETRY_NS : SAMPLE_NS)) return;
    RcBudgetSample next;
    platform_sample(&next);
    for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) {
        if (!(next.known_mask & (1u << d))) continue;
        B.sample.free.amount[d] = next.free.amount[d];
        B.sample.total.amount[d] = next.total.amount[d];
        B.sample.known_mask |= 1u << d;
        B.sample.estimated_mask = (B.sample.estimated_mask & ~(1u << d)) | (next.estimated_mask & (1u << d));
        B.unobserved.amount[d] = 0;
        for (int p = 1; p < RC_BUDGET_PRODUCERS; p++) B.producer[p].opaque_debt.amount[d] = 0;
        B.good_at[d] = now;
    }
    B.sample.process_commit = next.process_commit;
    B.sample.process_resident = next.process_resident;
    B.sampled_at = now;
    B.sampled = 1;
    B.refreshes++;
    limits_locked(1);
}

static RcAdmissionStatus request_known(const RcAdmissionVector *peak, int required) {
    if (!peak) return RC_ADMISSION_INVALID;
    uint64_t now = budget_now();
    for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) {
        if (!peak->amount[d]) continue;
        if (required && d == RC_BUDGET_STAGING) continue;
        if (!(B.sample.known_mask & (1u << d))) {
            if (!(required && d <= RC_BUDGET_COMMIT)) return RC_ADMISSION_UNSUPPORTED;
        } else if (!required && now - B.good_at[d] >= STALE_NS) return RC_ADMISSION_REFUSED;
    }
    return RC_ADMISSION_OK;
}

static void refused_locked(RcBudgetProducer producer, int required, const RcAdmissionVector *request,
                            RcAdmissionStatus status) {
    RcBudgetProducerStats *stats = &B.producer[producer];
    RcBudgetRefusal *why = &stats->last_refusal;
    *why = (RcBudgetRefusal){.request = *request, .status = status, .required = required};
    uint64_t now = budget_now();
    for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) {
        (void)rc_admission_available(&B.ledger, d, &why->available.amount[d]);
        if (!request->amount[d]) continue;
        if (request->amount[d] > why->available.amount[d]) why->capacity_mask |= 1u << d;
        if (!(B.sample.known_mask & (1u << d))) {
            if (!(required && (d <= RC_BUDGET_COMMIT || d == RC_BUDGET_STAGING))) why->unknown_mask |= 1u << d;
        } else if (!required && now - B.good_at[d] >= STALE_NS) why->stale_mask |= 1u << d;
        if ((why->capacity_mask | why->unknown_mask | why->stale_mask) & (1u << d)) stats->refused_domain[d]++;
    }
    stats->refused++;
}

static void vec_add(RcAdmissionVector *a, const RcAdmissionVector *b) {
    for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) a->amount[d] += b->amount[d];
}
static void vec_sub(RcAdmissionVector *a, const RcAdmissionVector *b) {
    for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) a->amount[d] -= b->amount[d];
}
static void growth_locked(const RcAdmissionVector *actual) {
    for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++)
        if (d != RC_BUDGET_STAGING) B.unobserved.amount[d] = add_sat(B.unobserved.amount[d], actual->amount[d]);
}

static RcAdmissionStatus lookup_locked(const RcBudgetTicket *ticket, RcAdmissionRecord **record) {
    if (!ticket || !producer_valid(ticket->producer)) return RC_ADMISSION_INVALID;
    return rc_admission_lookup(&B.ledger, ticket->token, ticket->producer, record);
}

RcAdmissionStatus resource_core_budget_reserve(RcBudgetProducer producer, int required,
                                              const RcAdmissionVector *peak, RcBudgetTicket *out) {
    if (!producer_valid(producer) || !out || !peak) return RC_ADMISSION_INVALID;
    budget_lock();
    init_locked();
    sample_locked(0);
    RcAdmissionStatus status;
    RcAdmissionToken token;
    for (int retry = 0;; retry++) {
        limits_locked(required);
        status = request_known(peak, required);
        if (status == RC_ADMISSION_OK) status = rc_admission_reserve(&B.ledger, producer, peak, &token);
        if (status == RC_ADMISSION_OK || retry || (status != RC_ADMISSION_REFUSED && status != RC_ADMISSION_UNSUPPORTED)) break;
        sample_locked(1);
    }
    if (status == RC_ADMISSION_OK) {
        *out = (RcBudgetTicket){token, producer};
        vec_add(&B.producer[producer].reserved, peak);
        B.producer[producer].admitted++;
    } else refused_locked(producer, required, peak, status);
    limits_locked(1);
    budget_unlock();
    return status;
}

RcAdmissionStatus resource_core_budget_split(RcBudgetTicket *ticket, const RcAdmissionVector *portion,
                                            RcBudgetProducer next_producer, RcBudgetTicket *out) {
    if (!ticket || !producer_valid(next_producer) || !portion || !out || ticket == out) return RC_ADMISSION_INVALID;
    budget_lock();
    RcAdmissionRecord *record;
    RcAdmissionStatus status = lookup_locked(ticket, &record);
    RcAdmissionToken token;
    if (status == RC_ADMISSION_OK)
        status = rc_admission_split(&B.ledger, ticket->token, ticket->producer, portion, next_producer, &token);
    if (status == RC_ADMISSION_OK) {
        vec_sub(&B.producer[ticket->producer].reserved, portion);
        vec_add(&B.producer[next_producer].reserved, portion);
        *out = (RcBudgetTicket){token, next_producer};
    }
    budget_unlock();
    return status;
}

static RcAdmissionStatus materialize(RcBudgetTicket *ticket, const RcAdmissionVector *actual,
                                     const RcAdmissionVector *growth, int compatible) {
    if (!ticket || !actual || !growth) return RC_ADMISSION_INVALID;
    budget_lock();
    RcAdmissionRecord *record;
    RcAdmissionStatus status = lookup_locked(ticket, &record);
    if (status == RC_ADMISSION_OK && compatible)
        for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++)
            if (growth->amount[d] > record->amount.amount[d]) {
                budget_unlock();
                return RC_ADMISSION_REFUSED;
            }
    if (status == RC_ADMISSION_OK) {
        RcAdmissionVector before = record->amount;
        RcAdmissionToken token;
        status = rc_admission_materialize(&B.ledger, ticket->token, ticket->producer, actual, &token);
        if (rc_admission_accounted(status)) {
            vec_sub(&B.producer[ticket->producer].reserved, &before);
            vec_add(&B.producer[ticket->producer].retained, actual);
            RcAdmissionVector measured = *growth;
            // A native result larger than its promised bound is still real growth. Account it completely and
            // stop subsequent admissions under pressure; an underestimated promise cannot erase actual bytes.
            for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++)
                if (!compatible && actual->amount[d] > measured.amount[d]) measured.amount[d] = actual->amount[d];
            growth_locked(&measured);
            ticket->token = token;
            limits_locked(1);
            status = rc_admission_pressure(&B.ledger);
            budget_changed();
        }
    }
    budget_unlock();
    return status;
}

RcAdmissionStatus resource_core_budget_materialize_growth(RcBudgetTicket *ticket, const RcAdmissionVector *actual,
                                                         const RcAdmissionVector *growth) {
    return materialize(ticket, actual, growth, 0);
}

RcAdmissionStatus resource_core_budget_materialize_compatible(RcBudgetTicket *ticket, const RcAdmissionVector *actual,
                                                             const RcAdmissionVector *growth) {
    return materialize(ticket, actual, growth, 1);
}

RcAdmissionStatus resource_core_budget_materialize(RcBudgetTicket *ticket, const RcAdmissionVector *actual) {
    return resource_core_budget_materialize_growth(ticket, actual, actual);
}

RcAdmissionStatus resource_core_budget_adopt_compatible(RcBudgetProducer producer, int required,
                                                       const RcAdmissionVector *actual,
                                                       const RcAdmissionVector *growth, RcBudgetTicket *out) {
    if (!producer_valid(producer) || !actual || !growth || !out) return RC_ADMISSION_INVALID;
    budget_lock();
    init_locked();
    sample_locked(0);
    limits_locked(required);
    RcAdmissionStatus status = request_known(growth, required);
    for (uint32_t d = 0; status == RC_ADMISSION_OK && d < RC_ADMISSION_MAX_DOMAINS; d++) {
        uint64_t available;
        (void)rc_admission_available(&B.ledger, d, &available);
        if (growth->amount[d] > available) status = RC_ADMISSION_REFUSED;
    }
    RcAdmissionToken token;
    if (status == RC_ADMISSION_OK) status = rc_admission_adopt(&B.ledger, producer, actual, &token);
    if (rc_admission_accounted(status)) {
        vec_add(&B.producer[producer].retained, actual);
        growth_locked(growth);
        B.producer[producer].admitted++;
        *out = (RcBudgetTicket){token, producer};
        limits_locked(1);
        status = rc_admission_pressure(&B.ledger);
    } else refused_locked(producer, required, growth, status);
    limits_locked(1);
    budget_unlock();
    return status;
}

RcAdmissionStatus resource_core_budget_cancel(RcBudgetTicket *ticket) {
    budget_lock();
    RcAdmissionRecord *record;
    RcAdmissionStatus status = lookup_locked(ticket, &record);
    if (status == RC_ADMISSION_OK) {
        RcAdmissionVector before = record->amount;
        status = rc_admission_cancel(&B.ledger, ticket->token, ticket->producer);
        if (status == RC_ADMISSION_OK) {
            vec_sub(&B.producer[ticket->producer].reserved, &before);
            *ticket = (RcBudgetTicket){0};
            budget_changed();
        }
    }
    budget_unlock();
    return status;
}

RcAdmissionStatus resource_core_budget_retire(RcBudgetTicket *ticket) {
    budget_lock();
    RcAdmissionRecord *record;
    RcAdmissionStatus status = lookup_locked(ticket, &record);
    if (status == RC_ADMISSION_OK) {
        RcAdmissionVector before = record->amount;
        RcAdmissionToken token;
        status = rc_admission_retire(&B.ledger, ticket->token, ticket->producer, &token);
        if (status == RC_ADMISSION_OK) {
            vec_sub(&B.producer[ticket->producer].retained, &before);
            vec_add(&B.producer[ticket->producer].pending, &before);
            ticket->token = token;
        }
    }
    budget_unlock();
    return status;
}

RcAdmissionStatus resource_core_budget_destroyed(RcBudgetTicket *ticket) {
    budget_lock();
    RcAdmissionRecord *record;
    RcAdmissionStatus status = lookup_locked(ticket, &record);
    if (status == RC_ADMISSION_OK) {
        RcAdmissionVector before = record->amount;
        status = rc_admission_destroyed(&B.ledger, ticket->token, ticket->producer);
        if (status == RC_ADMISSION_OK) {
            vec_sub(&B.producer[ticket->producer].pending, &before);
            *ticket = (RcBudgetTicket){0};
            limits_locked(1);
            budget_changed();
        }
    }
    budget_unlock();
    return status;
}

RcAdmissionStatus resource_core_budget_opaque_done(RcBudgetTicket *ticket) {
    budget_lock();
    RcAdmissionRecord *record;
    RcAdmissionStatus status = lookup_locked(ticket, &record);
    if (status == RC_ADMISSION_OK) {
        RcAdmissionVector before = record->amount;
        status = rc_admission_cancel(&B.ledger, ticket->token, ticket->producer);
        if (status == RC_ADMISSION_OK) {
            RcBudgetProducerStats *producer = &B.producer[ticket->producer];
            vec_sub(&producer->reserved, &before);
            for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++)
                producer->opaque_debt.amount[d] = add_sat(producer->opaque_debt.amount[d], before.amount[d]);
            producer->opaque_finished++;
            growth_locked(&before);
            *ticket = (RcBudgetTicket){0};
            limits_locked(1);
            budget_changed();
        }
    }
    budget_unlock();
    return status;
}

void resource_core_budget_snapshot(RcBudgetSnapshot *out) {
    if (!out) return;
    budget_lock();
    init_locked();
    *out = (RcBudgetSnapshot){.sample = B.sample, .reserved = B.ledger.reserved,
        .retained = B.ledger.retained, .pending = B.ledger.pending, .unobserved = B.unobserved,
        .sample_age_ns = B.sampled ? budget_now() - B.sampled_at : UINT64_MAX, .refreshes = B.refreshes};
    memcpy(out->producer, B.producer, sizeof out->producer);
    for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) {
        uint64_t used = rc_admission_used_at(&B.ledger, d), limit = limit_at(d, 1), optional = limit_at(d, 0);
        out->available.amount[d] = limit > used ? limit - used : 0;
        out->optional_available.amount[d] = optional > used ? optional - used : 0;
    }
    budget_unlock();
}

RcAdmissionStatus resource_core_budget_ticket_cost(const RcBudgetTicket *ticket, RcAdmissionVector *out) {
    if (!ticket || !out) return RC_ADMISSION_INVALID;
    budget_lock();
    RcAdmissionRecord *record;
    RcBudgetTicket value = *ticket;
    RcAdmissionStatus status = lookup_locked(&value, &record);
    if (status == RC_ADMISSION_OK) *out = record->amount;
    budget_unlock();
    return status;
}

void resource_core_budget_refresh(void) {
    budget_lock();
    init_locked();
    sample_locked(1);
    budget_unlock();
}

static void readiness_locked(int required, RcBudgetReadiness *out) {
    *out = (RcBudgetReadiness){.records_available = rc_admission_new_check(&B.ledger) == RC_ADMISSION_OK};
    uint64_t now = budget_now();
    for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) {
        int known = B.sample.known_mask & (1u << d);
        if (required && (d <= RC_BUDGET_COMMIT || d == RC_BUDGET_STAGING)) known = 1;
        if (!known || (!required && now - B.good_at[d] >= STALE_NS)) continue;
        out->usable_mask |= 1u << d;
        uint64_t limit = limit_at(d, required), used = rc_admission_used_at(&B.ledger, d);
        out->available.amount[d] = limit > used ? limit - used : 0;
    }
}

void resource_core_budget_readiness(int required, RcBudgetReadiness *out) {
    if (!out) return;
    budget_lock();
    init_locked();
    sample_locked(0);
    readiness_locked(required, out);
    budget_unlock();
}

void resource_core_budget_availability(RcBudgetAvailability *out) {
    if (!out) return;
    budget_lock();
    init_locked();
    sample_locked(0);
    RcBudgetAvailability next = {.progress = B.availability.progress};
    RcBudgetReadiness room;
    readiness_locked(0, &room);
    int increased = 0;
    if (room.records_available) {
        next.available = room.available;
        for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) {
            increased |= next.available.amount[d] > B.availability.available.amount[d];
        }
    }
    if (increased && next.progress != UINT64_MAX) next.progress++;
    B.availability = next;
    *out = next;
    budget_unlock();
}

void resource_core_budget_wait(uint32_t milliseconds) {
    if (!milliseconds) return;
#ifdef RC_BUDGET_TEST
    rc_budget_test_wait(milliseconds);
#else
    budget_lock();
    plat_cond_wait_ms(&B.changed, &B.lock, milliseconds);
    budget_unlock();
#endif
}

static RcAdmissionStatus register_reclaimer(RcBudgetProducer provider, uint32_t domains, int required_only,
                                            RcBudgetReclaimFn reclaim, void *context) {
    if (!producer_valid(provider) || !domains || domains >> RC_ADMISSION_MAX_DOMAINS || !reclaim)
        return RC_ADMISSION_INVALID;
    budget_lock();
    for (uint32_t i = 0; i < B.reclaimer_count; i++) {
        if (B.reclaimer[i].fn != reclaim || B.reclaimer[i].context != context) continue;
        RcAdmissionStatus status = B.reclaimer[i].producer == provider && B.reclaimer[i].domains == domains &&
            B.reclaimer[i].required_only == required_only ? RC_ADMISSION_OK : RC_ADMISSION_INVALID;
        budget_unlock();
        return status;
    }
    RcAdmissionStatus status = RC_ADMISSION_NO_RECORDS;
    if (B.reclaimer_count < RECLAIM_PROVIDERS) {
        uint32_t i = B.reclaimer_count++;
        B.reclaimer[i].fn = reclaim;
        B.reclaimer[i].context = context;
        B.reclaimer[i].domains = domains;
        B.reclaimer[i].producer = provider;
        B.reclaimer[i].required_only = required_only;
        status = RC_ADMISSION_OK;
    }
    budget_unlock();
    return status;
}

RcAdmissionStatus resource_core_budget_register_reclaimer(RcBudgetProducer provider, uint32_t domains,
                                                         RcBudgetReclaimFn reclaim, void *context) {
    return register_reclaimer(provider, domains, 0, reclaim, context);
}

RcAdmissionStatus resource_core_budget_register_required_reclaimer(RcBudgetProducer provider, uint32_t domains,
                                                                 RcBudgetReclaimFn reclaim, void *context) {
    return register_reclaimer(provider, domains, 1, reclaim, context);
}

static uint32_t deficit_locked(int required, const RcAdmissionVector *peak, RcAdmissionVector *deficit) {
    uint32_t mask = 0;
    *deficit = (RcAdmissionVector){{0}};
    for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) {
        uint64_t limit = limit_at(d, required), used = rc_admission_used_at(&B.ledger, d);
        uint64_t available = limit > used ? limit - used : 0;
        if (peak->amount[d] > available) {
            deficit->amount[d] = peak->amount[d] - available;
            mask |= 1u << d;
        }
    }
    return mask;
}

void resource_core_budget_service(int required, const RcAdmissionVector *peak, RcBudgetProgress *out) {
    if (!peak || !out) return;
    *out = (RcBudgetProgress){0};
    budget_lock();
    init_locked();
    sample_locked(1);
    RcAdmissionVector deficit;
    out->blocked_mask = deficit_locked(required, peak, &deficit);
    if (B.servicing) {
        out->event = B.event;
        out->reclaim = RC_BUDGET_RECLAIM_BUSY;
        budget_unlock();
        return;
    }
    B.servicing = 1;
    for (uint32_t p = 0; p < B.reclaimer_count && out->blocked_mask; p++) {
        RcBudgetReclaimFn fn = B.reclaimer[p].fn;
        if (!fn || !(B.reclaimer[p].domains & out->blocked_mask) ||
            (B.reclaimer[p].required_only && !required)) continue;
        void *context = B.reclaimer[p].context;
        budget_unlock();
        unsigned result = fn(context, &deficit);
        budget_lock();
        out->reclaim |= result;
        if (result & RC_BUDGET_RECLAIM_DESTROYED) sample_locked(2);
        out->blocked_mask = deficit_locked(required, peak, &deficit);
    }
    B.servicing = 0;
    budget_changed();
    out->event = B.event;
    budget_unlock();
}

void resource_core_budget_progress(void) {
    budget_lock();
    budget_changed();
    budget_unlock();
}

void resource_core_budget_wait_since(uint64_t event, uint32_t milliseconds) {
    if (!milliseconds) return;
    budget_lock();
    if (event == B.event) {
#ifdef RC_BUDGET_TEST
        budget_unlock();
        rc_budget_test_wait(milliseconds);
        return;
#else
        plat_cond_wait_ms(&B.changed, &B.lock, milliseconds);
#endif
    }
    budget_unlock();
}

#ifdef RC_BUDGET_TEST
void resource_core_budget_test_reset(void) { memset(&B, 0, sizeof B); }
#endif

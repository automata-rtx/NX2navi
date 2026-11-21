// Shared live capacity admission. Producers keep their content/lifetime authority; this service owns capacity
// reservations and measured headroom. No call waits for capacity except the explicitly named wait function.
#pragma once
#include "resource_core_admission.h"

enum {
    RC_BUDGET_RAM = 0, RC_BUDGET_COMMIT = 1, RC_BUDGET_STAGING = 2, RC_BUDGET_GPU_FIRST = 3,
    RC_BUDGET_GPU_HEAPS = RC_ADMISSION_MAX_DOMAINS - RC_BUDGET_GPU_FIRST
};
typedef enum {
    RC_BUDGET_TEXTURE = 1, RC_BUDGET_BUFFER, RC_BUDGET_STAGE, RC_BUDGET_PIPELINE,
    RC_BUDGET_SNAPSHOT, RC_BUDGET_OTHER, RC_BUDGET_PRODUCERS
} RcBudgetProducer;

typedef struct {
    RcAdmissionToken token;
    RcBudgetProducer producer;
} RcBudgetTicket;

typedef struct {
    RcAdmissionVector free, total;
    uint64_t process_commit, process_resident;
    uint32_t known_mask, estimated_mask;
} RcBudgetSample;

typedef struct {
    RcAdmissionVector request, available;
    uint32_t capacity_mask, unknown_mask, stale_mask;
    RcAdmissionStatus status;
    int required;
} RcBudgetRefusal;

typedef struct {
    RcAdmissionVector reserved, retained, pending, opaque_debt;
    uint64_t admitted, refused, opaque_finished;
    uint64_t refused_domain[RC_ADMISSION_MAX_DOMAINS];
    // Captured at the failed decision, before another producer or sample changes the allowance.
    RcBudgetRefusal last_refusal;
} RcBudgetProducerStats;

typedef struct {
    RcBudgetSample sample;
    RcAdmissionVector available, optional_available, reserved, retained, pending, unobserved;
    RcBudgetProducerStats producer[RC_BUDGET_PRODUCERS];
    uint64_t sample_age_ns, refreshes;
} RcBudgetSnapshot;

typedef struct {
    RcAdmissionVector available;
    uint64_t progress;
} RcBudgetAvailability;

typedef struct {
    RcAdmissionVector available;
    uint32_t usable_mask;
    int records_available;
} RcBudgetReadiness;

// A request can make progress when every required allowance is available and at least one of its allowances
// or record availability improved. An event wake is only a reason to observe this evidence again.
void resource_core_budget_readiness(int required, RcBudgetReadiness *out);
static inline int resource_core_budget_peak_ready(const RcAdmissionVector *peak, const RcBudgetReadiness *now) {
    if (!now->records_available) return 0;
    for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++)
        if (peak->amount[d] && (!(now->usable_mask & (1u << d)) || peak->amount[d] > now->available.amount[d]))
            return 0;
    return 1;
}
static inline int resource_core_budget_peak_progress(const RcAdmissionVector *peak, const RcBudgetReadiness *before,
                                                      const RcBudgetReadiness *now) {
    if (!resource_core_budget_peak_ready(peak, now)) return 0;
    if (!resource_core_budget_peak_ready(peak, before)) return 1;
    for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++)
        if (peak->amount[d] && now->available.amount[d] > before->available.amount[d]) return 1;
    return 0;
}

// Bytes in distinct constraints; never sum the vector. GPU heap indices are Vulkan memory-heap indices, not
// memory-type indices or an aggregate of dedicated VRAM. An unsupported heap/component fails explicitly.
static inline RcAdmissionStatus resource_core_budget_gpu_cost(RcAdmissionVector *cost, uint32_t heap, uint64_t bytes) {
    if (!cost) return RC_ADMISSION_INVALID;
    if (heap >= RC_BUDGET_GPU_HEAPS) return RC_ADMISSION_UNSUPPORTED;
    cost->amount[RC_BUDGET_GPU_FIRST + heap] = bytes;
    return RC_ADMISSION_OK;
}

// Required work can use protected progress headroom; it is still charged and can be refused. On refusal the
// caller preserves its job/readiness and retries outside renderer locks. Optional work remains queued.
// STAGING tracks occupied spans and throttles optional work. Required staging is constrained by actual RAM,
// commit and its selected heap, not that soft target: the upload recorder cannot suspend and drain mid-call.
// A hard span cap requires a segmented, resumable recorder before it can govern required work safely.
RcAdmissionStatus resource_core_budget_reserve(RcBudgetProducer producer, int required,
                                              const RcAdmissionVector *peak, RcBudgetTicket *out);
RcAdmissionStatus resource_core_budget_split(RcBudgetTicket *ticket, const RcAdmissionVector *portion,
                                            RcBudgetProducer next_producer, RcBudgetTicket *out);
RcAdmissionStatus resource_core_budget_materialize(RcBudgetTicket *ticket, const RcAdmissionVector *actual);
// Allocation adapters distinguish suballocation bytes from physical block growth. The growth bound can exceed
// the resulting allocation's bytes. Actual bytes exceeding the bound remain fully charged. Only verified reuse
// of already committed compatible capacity may use adopt;
// host working-set growth and staging capacity still belong in its incremental vector when applicable.
RcAdmissionStatus resource_core_budget_materialize_growth(RcBudgetTicket *ticket, const RcAdmissionVector *actual,
                                                         const RcAdmissionVector *growth);
// Consume a reserved incremental working-set allowance for a verified suballocation of already committed
// backing. The full occupied footprint becomes owned, but only incremental growth is charged against telemetry.
RcAdmissionStatus resource_core_budget_materialize_compatible(RcBudgetTicket *ticket,
                                                             const RcAdmissionVector *actual,
                                                             const RcAdmissionVector *growth);
RcAdmissionStatus resource_core_budget_adopt_compatible(RcBudgetProducer producer, int required,
                                                       const RcAdmissionVector *actual,
                                                       const RcAdmissionVector *growth, RcBudgetTicket *out);
RcAdmissionStatus resource_core_budget_cancel(RcBudgetTicket *ticket);
RcAdmissionStatus resource_core_budget_retire(RcBudgetTicket *ticket);
RcAdmissionStatus resource_core_budget_destroyed(RcBudgetTicket *ticket);
RcAdmissionStatus resource_core_budget_ticket_cost(const RcBudgetTicket *ticket, RcAdmissionVector *out);

// A driver compiler's exact retained allocation is opaque. Keep its entire completed peak estimate as debt
// against the current sample, then let fresh system observations replace that estimate. This does not claim
// the estimate is measured retained pipeline memory. Failed builds may also leave driver-cache allocations.
RcAdmissionStatus resource_core_budget_opaque_done(RcBudgetTicket *ticket);

// Snapshot is read-only. Refresh samples now; reserve also refreshes periodically and once before refusing
// sampled headroom. Actual destruction gives no speculative credit against stale physical/commit/GPU usage.
void resource_core_budget_snapshot(RcBudgetSnapshot *out);
// Small retry advisory for optional work, not a reservation. Samples use the ordinary bounded cadence. Progress
// advances only when some effective optional allowance increases between advisories; callers compare their own
// refused domains/bytes as well. Pending destruction never supplies credit against an unchanged physical sample.
void resource_core_budget_availability(RcBudgetAvailability *out);
void resource_core_budget_refresh(void);
void resource_core_budget_wait(uint32_t milliseconds);

enum { RC_BUDGET_RECLAIM_NONE = 0, RC_BUDGET_RECLAIM_PENDING = 1, RC_BUDGET_RECLAIM_DESTROYED = 2,
       RC_BUDGET_RECLAIM_BUSY = 4 };
typedef unsigned (*RcBudgetReclaimFn)(void *context, const RcAdmissionVector *deficit);
typedef struct {
    uint64_t event;
    uint32_t blocked_mask;
    unsigned reclaim;
} RcBudgetProgress;
// Register a process-lifetime adapter once. A provider detaches only unused, completed allocations under its
// own lock, then destroys outside that lock. It must not wait or call service recursively. Different domains
// remain distinct; returning DESTROYED requests fresh measurement, never grants the callback's byte estimate.
RcAdmissionStatus resource_core_budget_register_reclaimer(RcBudgetProducer provider, uint32_t domains,
                                                         RcBudgetReclaimFn reclaim, void *context);
// Reusable capacity needed by mandatory work is offered only to required requests.
RcAdmissionStatus resource_core_budget_register_required_reclaimer(RcBudgetProducer provider, uint32_t domains,
                                                                 RcBudgetReclaimFn reclaim, void *context);
// Called outside renderer/provider locks after a refused admission. The common service asks compatible
// providers for actual giveback. Its result is advisory: the caller still reserves before allocating.
void resource_core_budget_service(int required, const RcAdmissionVector *peak, RcBudgetProgress *out);
// Notify when completed work makes previously held capacity reclaimable, even before destruction.
void resource_core_budget_progress(void);
// Avoid a lost wake between inspecting pressure and sleeping. A bounded telemetry refresh is still needed
// for capacity changes made by other processes; this is not an allocation retry count or a failure timeout.
void resource_core_budget_wait_since(uint64_t event, uint32_t milliseconds);

#ifdef RC_BUDGET_TEST
void resource_core_budget_test_reset(void);
#endif

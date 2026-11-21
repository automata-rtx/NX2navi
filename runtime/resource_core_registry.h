// Shared registration history and overlapping authority. The runtime facade serializes operations internally.
// Returned records are value snapshots, never pointers into the registry. Completion calls report real adapter
// facts; this registry neither waits for the GPU nor proves visibility of uninstrumented guest CPU stores.
#ifndef RESOURCE_CORE_REGISTRY_H
#define RESOURCE_CORE_REGISTRY_H
#include <stdint.h>
#include "resource_core_observation.h"

typedef struct RcRegistry RcRegistry;
enum { RC_GUEST_COMPATIBILITY_BACKING = 1 };
typedef struct { uint64_t id; uint32_t index; } RcRegistryHandle;
typedef struct { uint64_t id; uint32_t index; } RcRegistryScope;
typedef struct { uint64_t id; uint32_t index; } RcRegistryWrite;
typedef struct { uint64_t id; uint32_t index; } RcRegistryGpuCapture;
// A persistent GPU owner identifies one physical image incarnation. The hold is bound only by an actual writer;
// neither a raw Vulkan handle nor an overlapping address range establishes this representation identity.
typedef struct { uint64_t image; RcRegistryWrite hold; } RcRegistryImageBinding;
typedef struct { RcInterval interval; RcRegistryWrite hold; } RcRegistryImageWriteExtent;
typedef struct { uint64_t id; uint32_t index; RcRegistryHandle handle; } RcRegistrySource;
// Assigned in replay order before parallel recording. An operation's input/output phases use distinct ordinals.
// A resumed continuation receives positions in the scene that actually records it, not its original command address.
typedef struct { uint64_t scene, ordinal; } RcRegistryPosition;
typedef struct {
    RcSourceRange range;
    RcRegistryPosition first, latest;
    // A drained possible execution still excludes readers, but does not publish a completed write.
    RcRegistryPosition completed_first, completed_latest;
} RcRegistryImageEffectSpan;
static inline int rc_registry_position_valid(RcRegistryPosition p) { return p.scene && p.ordinal; }
static inline int rc_registry_position_compare(RcRegistryPosition a, RcRegistryPosition b) {
    if (a.scene != b.scene) return a.scene < b.scene ? -1 : 1;
    return a.ordinal < b.ordinal ? -1 : a.ordinal > b.ordinal;
}
// Exclusive edge of a captured inclusive declaration prefix. Zero means no representable finite deadline.
static inline RcRegistryPosition rc_registry_sealed_deadline(RcRegistryPosition sealed) {
    if (!rc_registry_position_valid(sealed)) return (RcRegistryPosition){0};
    if (sealed.ordinal != UINT64_MAX) return (RcRegistryPosition){sealed.scene, sealed.ordinal + 1};
    if (sealed.scene != UINT64_MAX) return (RcRegistryPosition){sealed.scene + 1, 1};
    return (RcRegistryPosition){0};
}
typedef enum {
    RC_REGISTRY_OK, RC_REGISTRY_INVALID, RC_REGISTRY_STALE, RC_REGISTRY_CAPACITY,
    RC_REGISTRY_BLOCKED, RC_REGISTRY_EXHAUSTED, RC_REGISTRY_BUSY, RC_REGISTRY_UNSUPPORTED
} RcRegistryResult;
typedef enum { RC_REGISTRY_IMAGE, RC_REGISTRY_BUFFER, RC_REGISTRY_UNKNOWN } RcRegistryWriterKind;
typedef enum {
    RC_REGISTRY_GUARD_UNCLASSIFIED,
    RC_REGISTRY_GUARD_MAPPING_UNAVAILABLE,
    RC_REGISTRY_GUARD_RESERVATION_REFUSED,
    RC_REGISTRY_GUARD_COMMAND_OPAQUE,
    RC_REGISTRY_GUARD_GPA_PROJECTION,
} RcRegistryGuardReason;
typedef struct { RcSourceRange range; RcRegistryWriterKind kind; } RcRegistryWriteExtent;
typedef struct {
    RcRegistryWrite ticket;
    const RcSourceRange *written, *uncertain;
    uint32_t written_count, uncertain_count;
} RcRegistryWriteCompletion;
typedef enum { RC_REGISTRY_VISIBILITY_ORDERED, RC_REGISTRY_VISIBILITY_COMPLETED } RcRegistryVisibilityMode;
typedef struct RcRegistryVisibility {
    uint64_t horizon, submission;
    RcRegistryVisibilityMode mode;
    uint32_t writer_kinds;
} RcRegistryVisibility;
typedef struct {
    uint32_t records, writers, scopes;
} RcRegistryConfig;
typedef struct {
    RcRegistryHandle handle;
    uint64_t key, object_serial, interpretation;
    RcSourceRange range;
    uint64_t content, representation, payload;
    int finalized, current;
} RcRegistryRecord;
typedef struct {
    RcRegistryHandle handle;
    // Same content provenance as RcRegistryReadObservation. Generic revision also includes declarations/cancels;
    // content_revision changes only when old source bytes can no longer be treated as unchanged.
    uint64_t revision, scope, guard, content_revision;
    RcSourceRange range;
} RcRegistryObservation;
typedef struct {
    RcRegistryHandle handle;
    uint64_t content_revision, scope, guard, revision, topology_revision;
    RcSourceRange range;
    // revision names hard ordered-authority changes. Later declarations cannot enter this sealed prefix.
    RcRegistryPosition position, next_write, sealed_through;
} RcRegistryReadObservation;
// One serialized source owner per prepared representation. The caller supplies stable storage and external
// exclusivity (the texture adapter uses its lock); only source_access APIs mutate these fields. Value snapshots
// are permission/provenance, never owning providers. Copying this object does not fork its lease.
typedef struct {
    RcRegistryObservation legacy;
    RcRegistryReadObservation ordered;
} RcRegistrySourceProof;
typedef struct {
    RcRegistrySource owner;
    RcRegistrySourceProof _content;
    RcRegistryReadObservation _read;
    RcRegistryObservation _maintenance;
} RcRegistrySourceAccess;
static inline RcRegistryHandle rc_registry_source_access_handle(const RcRegistrySourceAccess *a) {
    return a->owner.handle;
}
static inline RcRegistrySourceProof rc_registry_source_access_content(const RcRegistrySourceAccess *a) {
    return a->_content;
}
static inline RcRegistryReadObservation rc_registry_source_access_permission(const RcRegistrySourceAccess *a) {
    return a->_read;
}
// Canonical byte identity excludes permission topology. A new observation must separately authorize use.
static inline RcRegistryObservation rc_registry_source_proof_content(const RcRegistrySourceProof *p) {
    if (!rc_registry_position_valid(p->ordered.position)) return p->legacy;
    return (RcRegistryObservation){.handle = p->ordered.handle, .range = p->ordered.range,
        .content_revision = p->ordered.content_revision, .guard = p->ordered.guard};
}
static inline int rc_registry_source_proof_same(const RcRegistrySourceProof *a, const RcRegistrySourceProof *b) {
    if (!a || !b) return 0;
    RcRegistryObservation x = rc_registry_source_proof_content(a), y = rc_registry_source_proof_content(b);
    return x.handle.id && x.handle.id == y.handle.id && x.handle.index == y.handle.index &&
        x.content_revision && x.content_revision == y.content_revision && x.guard && x.guard == y.guard &&
        rc_range_valid(x.range) && rc_range_equal(x.range, y.range);
}
enum {
    RC_REGISTRY_BINDING_PRIOR_PENDING = 1u, RC_REGISTRY_BINDING_PRIOR_SUBMITTED = 2u,
    RC_REGISTRY_BINDING_FOREIGN = 4u, RC_REGISTRY_BINDING_HELD = 8u
};
// A compatibility preparation bracket describes unchanged resolution inputs, not authoritative source bytes.
// Its distinct types cannot be supplied to source publication or immutable read-version commit.
typedef struct {
    RcRegistryHandle handle;
    uint64_t revision, scope, guard, topology_revision;
    RcSourceRange range;
    // next_producer retains the actual schedule edge even when next_read is clamped by the sealed prefix.
    RcRegistryPosition position, next_read, sealed_through, next_producer;
    uint32_t provenance, writer_kinds;
    RcRegistryImageBinding image_binding;
} RcRegistryBindingObservation;
typedef struct { RcRegistryBindingObservation observation; } RcRegistryBindingReceipt;
// Installed-binding schedule certificates are distinct from finite byte/read observations. The aggregate owner
// must consume both hard and schedule event histories, bracket reuse with epoch, and check the read position is
// still open. next_read is the actual producer edge (zero if none), never permission to publish new source bytes.
typedef struct { RcRegistryReadObservation observation; RcRegistryPosition next_read; uint64_t epoch; } RcRegistryReadSchedule;
typedef struct { RcRegistryBindingReceipt receipt; RcRegistryPosition next_read; uint64_t epoch; } RcRegistryBindingSchedule;
// The caller owns the exact provider through RcLifetimeVersion/captures. This value does not keep guest memory
// reconstructible, and acquiring it after execution starts must use that retained provider rather than observe_at.
typedef struct {
    uint64_t id;
    RcRegistryReadObservation origin;
    RcPreparedInput input;
} RcRegistryReadVersion;
// Move-free value identity plus an owning token stored authoritatively in the registry. Copies cannot release twice.
typedef struct { uint64_t id; uint32_t index; RcRegistryRecord record; } RcRegistryCapture;
typedef struct {
    uint64_t registrations, observations, invalidations, overlap_visits, history_visits, collected;
    uint32_t records, writers, scopes, captures;
    uint32_t writer_peak;
    uint32_t source_records, retired_records, source_leases;
    uint32_t live_sources, live_descriptors;
    uint32_t writer_pending, writer_submitted, writer_foreign, writer_holds;
    uint32_t writer_image, writer_buffer, writer_unknown;
    uint64_t record_refusals, writer_refusals, reconcile_busy, reconcile_capacity;
    // One first overlapping blocker per failed observation, rather than a scan of every overlapping writer.
    uint64_t blocked_pending, blocked_submitted, blocked_foreign, blocked_held;
    uint64_t blocked_image, blocked_buffer, blocked_unknown, blocked_fail_closed;
    uint64_t visibility_proofs, visibility_applied, visibility_removed, visibility_retained;
    uint64_t ordered_observations, ordered_blocked, future_writers_ignored, read_versions, late_declarations;
    uint64_t read_renew_fast, read_renew_queries, binding_renew_fast, binding_renew_queries;
    uint64_t completion_range_refusals;
    // Compatibility bracket outcomes count the first actual refusal, without another overlap traversal.
    uint64_t binding_begins, binding_ready, binding_unknown_held, binding_unordered_foreign, binding_image_alias;
    uint64_t binding_unpositioned, binding_edge, binding_stale, binding_closed, binding_failed;
    uint32_t image_positions, image_position_peak;
    uint64_t image_position_refusals;
    int fail_closed;
} RcRegistryStats;

// Bounded diagnostic evidence. Counts are cumulative; each reason retains its first four distinct offenders.
// Reading this snapshot never traverses authority intervals or changes publication/readiness decisions.
typedef enum {
    RC_REGISTRY_DIAG_SOURCE_PENDING_IMAGE, RC_REGISTRY_DIAG_SOURCE_PENDING_BUFFER, RC_REGISTRY_DIAG_SOURCE_PENDING_UNKNOWN,
    RC_REGISTRY_DIAG_SOURCE_SUBMITTED_IMAGE, RC_REGISTRY_DIAG_SOURCE_SUBMITTED_BUFFER, RC_REGISTRY_DIAG_SOURCE_SUBMITTED_UNKNOWN,
    RC_REGISTRY_DIAG_SOURCE_FOREIGN_IMAGE, RC_REGISTRY_DIAG_SOURCE_FOREIGN_BUFFER, RC_REGISTRY_DIAG_SOURCE_FOREIGN_UNKNOWN,
    RC_REGISTRY_DIAG_SOURCE_HELD_IMAGE, RC_REGISTRY_DIAG_SOURCE_HELD_BUFFER, RC_REGISTRY_DIAG_SOURCE_HELD_UNKNOWN,
    RC_REGISTRY_DIAG_TYPED_BEGIN, RC_REGISTRY_DIAG_TYPED_READY,
    RC_REGISTRY_DIAG_HOLD_MISSING, RC_REGISTRY_DIAG_HOLD_STATE, RC_REGISTRY_DIAG_HOLD_KIND,
    RC_REGISTRY_DIAG_HOLD_FORGOTTEN, RC_REGISTRY_DIAG_HOLD_UNBOUND, RC_REGISTRY_DIAG_HOLD_IMAGE,
    RC_REGISTRY_DIAG_HOLD_RANGE, RC_REGISTRY_DIAG_OTHER_HOLD, RC_REGISTRY_DIAG_IMAGE_FOREIGN,
    RC_REGISTRY_DIAG_IMAGE_INCOMPLETE, RC_REGISTRY_DIAG_IMAGE_SCHEDULE,
    RC_REGISTRY_DIAG_IMAGE_PREFIX, RC_REGISTRY_DIAG_IMAGE_IDENTITY, RC_REGISTRY_DIAG_IMAGE_EDGE,
    RC_REGISTRY_DIAG_GENERIC_IMAGE,
    RC_REGISTRY_DIAG_FOREIGN_IMAGE, RC_REGISTRY_DIAG_FOREIGN_BUFFER, RC_REGISTRY_DIAG_FOREIGN_UNKNOWN,
    RC_REGISTRY_DIAG_MISSING_IMAGE, RC_REGISTRY_DIAG_COUNT
} RcRegistryDiagnosticReason;
enum {
    RC_REGISTRY_DIAG_SYNTHETIC = 1u, RC_REGISTRY_DIAG_HOLD_RELEASE = 2u,
    RC_REGISTRY_DIAG_ABANDONED = 4u, RC_REGISTRY_DIAG_COMPLETED = 8u,
    RC_REGISTRY_DIAG_FORGOTTEN = 16u, RC_REGISTRY_DIAG_INCOMPLETE = 32u,
    RC_REGISTRY_DIAG_UNORDERED = 64u
};
enum { RC_REGISTRY_DIAGNOSTIC_SAMPLES = 4 };
typedef struct {
    uint64_t sequence, submission, writer_image;
    RcRegistryWrite writer;
    RcSourceRange range, requested;
    RcRegistryPosition position, read_position, producer_position;
    RcRegistryImageBinding selected, producer_image;
    uint32_t state, kind, flags, producer_count, producer_refs, producer_unresolved, producer_active,
             producer_unknown, prefix_bad, prepared_actuals;
} RcRegistryDiagnosticSample;
typedef struct {
    uint64_t counts[RC_REGISTRY_DIAG_COUNT];
    uint32_t sample_count[RC_REGISTRY_DIAG_COUNT];
    RcRegistryDiagnosticSample samples[RC_REGISTRY_DIAG_COUNT][RC_REGISTRY_DIAGNOSTIC_SAMPLES];
} RcRegistryDiagnostics;
void rc_registry_diagnostics(RcRegistry *, RcRegistryDiagnostics *);
const char *rc_registry_diagnostic_name(RcRegistryDiagnosticReason);
typedef struct {
    RcRegistryDiagnosticSample sample;
    RcRegistryDiagnosticReason reason;
    uint64_t writer_address, writer_mapping; // retained native-image mapping, zero when absent
    RcRegistryPosition origin_position;
    uintptr_t writer_origin;
    uint32_t guard_reason;
} RcRegistryReadBlocker;
// Terminal diagnostics only. Report a presently intersecting writer, honoring ordered read position when supplied.
int rc_registry_diagnose_read_blocker(RcRegistry *, RcSourceRange, RcRegistryPosition, RcRegistryReadBlocker *);
// Records the adapter's missing-output fallback only; this function grants no authority or cancellation proof.
void rc_registry_diagnose_missing_image(RcRegistry *, RcRegistryWrite, RcRegistryPosition, uint32_t prepared_actuals);

// Fixed-capacity storage is allocated once. Creation/destruction require external exclusivity; ordinary calls lock.
RcRegistry *rc_registry_create(RcRegistryConfig config);
void rc_registry_destroy(RcRegistry *registry);
RcRegistry *rc_registry_runtime(void);
RcRegistryResult rc_registry_register(RcRegistry *, uint64_t key, uint64_t object_serial, RcSourceRange,
                                      uint64_t interpretation, RcRegistryHandle *);
// Sources have a separate key namespace and deduplicate object serial + range. Authority remains shared by overlap.
RcRegistryResult rc_registry_source(RcRegistry *, uint64_t object_serial, RcSourceRange, RcRegistryHandle *);
// Adapter ownership of a source association. Repeating acquire with the same live in/out lease is idempotent;
// different owners get independent tokens sharing one source record. Release before changing object/range.
// The last lease retires the source; captured records and registration scopes still retain required history.
RcRegistryResult rc_registry_source_acquire(RcRegistry *, uint64_t object_serial, RcSourceRange, RcRegistrySource *);
RcRegistryResult rc_registry_source_release(RcRegistry *, RcRegistrySource *);
RcRegistryResult rc_registry_source_access_acquire(RcRegistry *, uint64_t object_serial, RcSourceRange,
                                                  RcRegistrySourceAccess *);
// association is adapter metadata for routing source edits to the current owner. It grants no permission.
RcRegistryResult rc_registry_source_access_acquire_associated(RcRegistry *, uint64_t object_serial, RcSourceRange,
                                                             uint64_t association, RcRegistrySourceAccess *);
// One source record with immutable, sorted meaningful ranges inside its backing hull. The hull remains the
// remap/notification domain. Only these ranges contribute to byte authority, and global guards still apply.
enum { RC_REGISTRY_SOURCE_SPAN_LIMIT = 4096 };
RcRegistryResult rc_registry_source_access_acquire_shaped(RcRegistry *, uint64_t object_serial, RcSourceRange hull,
    uint64_t association, const RcSourceRange *spans, uint32_t count, RcRegistrySourceAccess *);
RcRegistryResult rc_registry_source_access_release(RcRegistry *, RcRegistrySourceAccess *);
RcRegistryResult rc_registry_source_access_observe(RcRegistry *, const RcRegistrySourceAccess *, uint64_t scope,
                                                  RcRegistryPosition, RcRegistrySourceProof *);
// Memory adapters must validate their exact copy cutoff before commit. Commit retains canonical provenance
// even if topology revoked permission, but never grants new permission from that fact. changed reports whether
// the prior prepared version must be detached; the physical provider stays owned by its lifetime service.
RcRegistryResult rc_registry_source_access_commit(RcRegistry *, RcRegistrySourceAccess *,
    const RcRegistrySourceProof *, int same_contents, int *changed);
RcRegistryResult rc_registry_source_access_read(RcRegistry *, RcRegistrySourceAccess *, uint64_t scope,
                                               RcRegistryPosition, RcRegistryReadObservation *);
int rc_registry_source_access_current(RcRegistry *, RcRegistrySourceAccess *, uint64_t scope, RcRegistryPosition);
RcRegistryResult rc_registry_source_access_schedule(RcRegistry *, RcRegistrySourceAccess *, uint64_t scope,
                                                   RcRegistryPosition, RcRegistryReadSchedule *);
// Source extents belong to one preparation. Observe and validate every leg at the same registry position;
// commit preserves all old content if any leg lost its authority. The read/schedule observation is an anchor
// for the first extent, and the schedule deadline is the earliest across all extents.
RcRegistryResult rc_registry_source_access_observe_extents(RcRegistry *, const RcRegistrySourceAccess *, uint32_t count,
    uint64_t scope, RcRegistryPosition, RcRegistrySourceProof *);
int rc_registry_source_proofs_current(RcRegistry *, const RcRegistrySourceProof *, uint32_t count, RcRegistryPosition);
RcRegistryResult rc_registry_source_access_commit_extents(RcRegistry *, RcRegistrySourceAccess *,
    const RcRegistrySourceProof *, uint32_t count, int same_contents, int *changed);
RcRegistryResult rc_registry_source_access_read_extents(RcRegistry *, RcRegistrySourceAccess *, uint32_t count,
    uint64_t scope, RcRegistryPosition, RcRegistryReadObservation *first);
int rc_registry_source_access_current_extents(RcRegistry *, RcRegistrySourceAccess *, uint32_t count,
    uint64_t scope, RcRegistryPosition);
RcRegistryResult rc_registry_source_access_schedule_extents(RcRegistry *, RcRegistrySourceAccess *, uint32_t count,
    uint64_t scope, RcRegistryPosition, RcRegistryReadSchedule *);

RcRegistryResult rc_registry_finalize(RcRegistry *, uint64_t key, uint64_t object_serial);
// End a descriptor's current association. Open horizons before this retirement retain its incarnation.
RcRegistryResult rc_registry_retire_registration(RcRegistry *, uint64_t key, uint64_t object_serial);
RcRegistryResult rc_registry_finalize_object(RcRegistry *, uint64_t object_serial, RcSourceRange);
RcRegistryResult rc_registry_range_gone(RcRegistry *, RcSourceRange);
RcRegistryResult rc_registry_cpu_changed(RcRegistry *, RcSourceRange);

RcRegistryResult rc_registry_scope_begin(RcRegistry *, RcRegistryScope *);
RcRegistryResult rc_registry_scope_end(RcRegistry *, RcRegistryScope);
// Zero scope means the current registration. A nonzero scope preserves the registration horizon it captured.
RcRegistryResult rc_registry_resolve(RcRegistry *, uint64_t key, RcRegistryScope, RcRegistryRecord *);
// Whether the exact registration can still be reached from an open resolution horizon. A current unscoped
// lookup is excluded here; adapters use this before deciding whether replacement needs retained history.
int rc_registry_registration_reachable(RcRegistry *, RcRegistryHandle);
RcRegistryResult rc_registry_get(RcRegistry *, RcRegistryHandle, RcRegistryRecord *);
RcRegistryResult rc_registry_capture(RcRegistry *, RcRegistryHandle, RcRegistryCapture *);
RcRegistryResult rc_registry_capture_release(RcRegistry *, RcRegistryCapture);

// scope is a nonzero semantic observation identity supplied by the replay boundary, not a claim of CPU quiescence.
RcRegistryResult rc_registry_observe(RcRegistry *, RcRegistryHandle, uint64_t scope, RcRegistryObservation *);
int rc_registry_observation_current(RcRegistry *, const RcRegistryObservation *);
// Retained byte provenance only: content identity/range/incarnation still match, even when generic producer
// topology has invalidated the old read permission. Never authorizes a read or publication; callers must obtain
// a fresh mode-specific observation before using retained bytes as current source contents.
int rc_registry_content_current(RcRegistry *, const RcRegistryObservation *);
RcRegistryResult rc_registry_publish(RcRegistry *, const RcRegistryObservation *, uint64_t content,
                                     uint64_t representation, uint64_t payload);

// Seal a declaration prefix only after every possible producer in it has a range reservation. Unclassified effects
// need a conservative reservation; absence of a late worker's registration is not proof of absence of its write.
// New ordered writers at/before the prefix are rejected and certification fails closed. Retried operations reuse
// their prepared tickets. Close CPU recording through a prefix before any GPU handoff can execute it; this closure
// is not a claim that vkQueueSubmit succeeded or that execution completed. Both frontiers advance monotonically.
RcRegistryResult rc_registry_declare_through(RcRegistry *, RcRegistryPosition);
// The exclusive recorder canceled every unhanded output of its latest single operation before replaying its input.
RcRegistryResult rc_registry_reopen_unemitted(RcRegistry *, RcRegistryPosition read);
// A pure preparation owns the last sealed read before declaring any output at that operation.
int rc_registry_provisional_read_current(RcRegistry *, RcRegistryPosition read);
RcRegistryResult rc_registry_handoff_through(RcRegistry *, RcRegistryPosition);
int rc_registry_position_handed(RcRegistry *, RcRegistryPosition);
// The logical recording prefix still needs independent actual queue handover proof.
RcRegistryPosition rc_registry_handed_cutoff(RcRegistry *);
// Initial observation checks overlapping authority. Only strictly future, positioned, still-pending writers are
// ignored. Earlier/same-phase/unpositioned producers and held/foreign authority block. Input-before-output phases
// apply only to ordinary non-writable-alias inputs; the adapter must exclude feedback and mixed authority classes.
// Pair this with the memory adapter's ordered observation of uninstrumented stores. It is not an atomic CPU snapshot.
RcRegistryResult rc_registry_observe_at(RcRegistry *, RcRegistryHandle, uint64_t scope,
                                       RcRegistryPosition, RcRegistryReadObservation *);
// Structural dependency check for an already immutable input capture, with no source record allocation. Seal the
// current input phase first; its output phase can be declared afterward. This does not certify snapshot freshness
// or pin bytes: the adapter must supply these same captured bytes to the real worker descriptor.
RcRegistryResult rc_registry_read_range_at(RcRegistry *, RcSourceRange, RcRegistryPosition);
// The caller retains an actual single device-memory buffer whose canonical backing owns these byte ranges.
// Earlier ordered buffer producers can supply native bytes without establishing CPU visibility.
RcRegistryResult rc_registry_read_device_buffer_at(RcRegistry *, const RcInterval *, uint32_t count,
                                                   RcRegistryPosition);
typedef struct { RcRegistryWrite writer; RcInterval interval; } RcRegistryBufferPhaseExtent;
typedef enum {
    RC_REGISTRY_BUFFER_PHASE_POSSIBLY_ISSUED,
    RC_REGISTRY_BUFFER_PHASE_COMPLETED
} RcRegistryBufferPhaseOutcome;
// An ordered operation can submit its first native phase without handing its entire output ordinal.
// Bind to the actual writer's normalized descriptor before emission, and note the actual native result once.
RcRegistryResult rc_registry_buffer_phase_bind(RcRegistry *, const RcRegistryBufferPhaseExtent *, uint32_t count,
                                               RcRegistryPosition read);
RcRegistryResult rc_registry_buffer_phase_note(RcRegistry *, const RcRegistryBufferPhaseExtent *, uint32_t count,
                                               RcRegistryPosition read, RcRegistryBufferPhaseOutcome);
// The caller proves the actual retained buffers and offsets correspond. Every input byte must be within
// a declared phase extent that has positively completed. The original read stays open across native progress.
RcRegistryResult rc_registry_read_device_buffer_phase(RcRegistry *, const RcInterval *input, uint32_t input_count,
    const RcRegistryBufferPhaseExtent *, uint32_t count, RcRegistryPosition read);
// Reserve an inactive slot. BUSY reports a release epoch. Queued owners pass the recorder's foreground allowance.
// The caller retains canonical coverage and the original mapping until end, including during reservation.
RcRegistryResult rc_registry_gpu_capture_reserve(RcRegistry *, const RcSourceRange *, uint32_t count,
                                                 uint32_t foreground_slots, RcRegistryGpuCapture *, uint64_t *release_epoch);
void rc_registry_gpu_capture_wait(RcRegistry *, uint64_t release_epoch, uint32_t milliseconds);
// Start immediately before copying bytes. This does not allocate or synchronize concurrent CPU stores.
RcRegistryResult rc_registry_gpu_capture_start(RcRegistry *, RcRegistryGpuCapture);
// Combined reserve and start for immediate protected borrows.
RcRegistryResult rc_registry_gpu_capture_begin(RcRegistry *, const RcSourceRange *coverage, uint32_t count,
                                               RcRegistryGpuCapture *);
// Query actual backing-relative subranges at an open read position. Earlier live authority returns BLOCKED,
// later CPU visibility of an earlier producer returns STALE, and lost relevant history returns CAPACITY.
// CPU-only content notifications do not invalidate captured bytes. The mapper validates current incarnation.
RcRegistryResult rc_registry_gpu_capture_read_at(RcRegistry *, RcRegistryGpuCapture,
                                                 const RcSourceRange *, uint32_t count, RcRegistryPosition);
typedef enum {
    RC_GPU_CAPTURE_NO_EVIDENCE, RC_GPU_CAPTURE_EARLIER_BUFFER, RC_GPU_CAPTURE_EARLIER_VISIBLE
} RcRegistryGpuReadEvidence;
RcRegistryResult rc_registry_gpu_capture_read_evidence(RcRegistry *, RcRegistryGpuCapture, const RcSourceRange *,
                                                       uint32_t count, RcRegistryPosition, RcRegistryGpuReadEvidence *);
RcRegistryResult rc_registry_gpu_capture_end(RcRegistry *, RcRegistryGpuCapture *);

// Native buffer identity from the actual retained adapter allocation. A fixed physical buffer uses its
// unique nonzero allocation publication and zero incarnation. A replaceable mirror has a nonzero incarnation.
// Both identity fields must match. Offset is affine at the beginning of the interval.
typedef struct { uint64_t allocation, incarnation, offset; } RcRegistryBufferLocation;
typedef struct { RcInterval interval; RcRegistryBufferLocation native; } RcRegistryBufferSpan;
typedef struct {
    RcInterval interval;
    RcRegistryPosition output;
    uint32_t demand;
} RcRegistryBufferFragment;

// Optional immutable buffer-output bytes retained independently of execution and guest visibility. Both
// callbacks may run under the registry lock and may only update ownership or enqueue deferred destruction.
typedef struct RcRegistryBufferPayload RcRegistryBufferPayload;
struct RcRegistryBufferPayload {
    void (*retain)(RcRegistryBufferPayload *);
    void (*release)(RcRegistryBufferPayload *);
};

// An already recorded exact whole overwrite can authenticate its existing normalized writer ticket.
RcRegistryResult rc_registry_buffer_write_record(RcRegistry *, RcRegistryWrite, RcRegistryBufferSpan,
                                                 RcRegistryPosition output);
// Attach one registry-owned reference to that exact overwrite. The caller retains its original reference.
RcRegistryResult rc_registry_buffer_write_payload_record(RcRegistry *, RcRegistryWrite, RcRegistryBufferSpan,
    RcRegistryPosition output, RcRegistryBufferPayload *);
// After an issued bounded write, publish the resulting native range using the adapter's independently
// initialized complete span at the original read, or complete earlier native results in that placement.
// This authenticates the resulting bytes for native consumers. It does not describe which bytes changed.
RcRegistryResult rc_registry_buffer_preserved_record(RcRegistry *, RcRegistryWrite, RcRegistryBufferSpan,
                                                     RcRegistryPosition read, int initialized);
// Verify all earlier authoritative contents reside in this exact selected buffer at the original read.
// The adapter independently retains the provider and proves any initialized physical baseline.
RcRegistryResult rc_registry_buffer_preserve_input_at(RcRegistry *, RcRegistryBufferSpan,
    RcRegistryPosition read, int initialized);
// Record the exact retained buffer placement of a BUFFER may-write after actual command emission. This is
// separate from exact write coverage and grants no publication or visibility claim.
RcRegistryResult rc_registry_buffer_access_record(RcRegistry *, RcRegistryWrite, RcRegistryBufferSpan,
                                                  RcRegistryPosition output);
// Every earlier live claim overlapping this exact interval must have an actual access receipt with the same
// affine VkBuffer allocation/incarnation/offset. Used for narrow conditional read-modify-write preservation.
RcRegistryResult rc_registry_buffer_access_check(RcRegistry *, RcRegistryBufferSpan, RcRegistryPosition read);
// Select actual changed intervals since the capture, bound to the caller's exact retained native span.
// Gaps still need the original immutable snapshot lease. Result count is zero on failure.
RcRegistryResult rc_registry_gpu_capture_buffer_fragments(RcRegistry *, RcRegistryGpuCapture,
    const RcRegistryBufferSpan *, uint32_t count, RcRegistryPosition read,
    RcRegistryBufferFragment *, uint32_t capacity, uint32_t *fragment_count);
// The same winning fragments, split at payload boundaries. Each nonnull parallel slot has one caller-owned
// reference on success. Payloads match the authored canonical mapping and guest/native affine offsets. A
// fragment may still need a native copy when its exact winner has no payload. Failure transfers no references.
RcRegistryResult rc_registry_gpu_capture_buffer_fragments_with_payloads(RcRegistry *, RcRegistryGpuCapture,
    const RcRegistryBufferSpan *, uint32_t count, RcRegistryPosition read,
    RcRegistryBufferFragment *, RcRegistryBufferPayload **, uint32_t capacity, uint32_t *fragment_count);
// One actual recorder owns this interval of the same bounded visibility history. A known immutable inline
// output can serve a later ordered read only if no intervening producer changed that backing coverage.
RcRegistryResult rc_registry_produced_scene_begin(RcRegistry *, uint64_t scene);
RcRegistryResult rc_registry_produced_scene_end(RcRegistry *, uint64_t scene);
RcRegistryResult rc_registry_read_produced_at(RcRegistry *, const RcInterval *, uint32_t count,
                                              RcRegistryPosition output, RcRegistryPosition read);
int rc_registry_read_observation_current(RcRegistry *, const RcRegistryReadObservation *);
// Advance unchanged permission only within its captured sealed prefix and before its original next writer.
// Later declarations and cancellation cannot extend that prefix; extending it needs a new overlap query.
RcRegistryResult rc_registry_read_revalidate(RcRegistry *, const RcRegistryReadObservation *, uint64_t scope,
                                             RcRegistryPosition, RcRegistryReadObservation *);
// Fresh schedule query for the same retained bytes after the old prefix expires. Same scope/scene, nonbackwards,
// identical source incarnation/content/guard, and current read permission are required; no new byte capture.
RcRegistryResult rc_registry_read_renew(RcRegistry *, const RcRegistryReadObservation *, uint64_t scope,
                                       RcRegistryPosition, RcRegistryReadObservation *);
RcRegistryResult rc_registry_read_commit(RcRegistry *, const RcRegistryReadObservation *,
                                        const RcPreparedInput *, RcRegistryReadVersion *);

// Bracket actual successful compatibility preparation with begin/commit. Older positioned producers and stable
// held/foreign authority remain explicit legacy provenance, never a clean-source claim. Unknown holds and their
// unordered foreign remnants refuse: future alias producers may not overlap this address range. Unpositioned
// pending or submitted work, same-operation outputs, and closed/undeclared prefixes also refuse. Image holds and
// prior image producers refuse because shared image declarations currently retain only their first write position;
// they cannot bound reuse of a CPU-backed alias across later writes. Direct GPU image bindings use GPU ordering.
// Begin scans overlaps and
// retains the earliest future producer's READ phase, including producers beyond an earlier unknown-range writer.
// The caller separately brackets CPU mapping/protection/freshness and owns the exact physical binding. Commit
// refuses every intervening overlapping authority transition. Revalidation is O(1), within the same observation
// scope and scene; it cannot advance a successful preparation across its next producer or a changed registration.
RcRegistryResult rc_registry_binding_begin(RcRegistry *, RcRegistryHandle, uint64_t scope,
                                           RcRegistryPosition, RcRegistryBindingObservation *);
// Compatibility-only: prior IMAGE producers must be resolved to this exact live image owner. Other-image,
// partial, unknown and unresolved producers refuse; future declarations still bound the interval. CPU source
// observations and immutable source publication retain their normal GPU-authority exclusion.
RcRegistryResult rc_registry_binding_begin_image(RcRegistry *, RcRegistryHandle, uint64_t scope,
    RcRegistryPosition, RcRegistryImageBinding, RcRegistryBindingObservation *);
// Bind once, after an actual image write. Reusing the range hold for another physical image refuses.
RcRegistryResult rc_registry_image_hold_bind(RcRegistry *, RcRegistryWrite, uint64_t image);
// Physical image replacement permanently removes this hold's typed association, including numeric handle reuse.
// Broad GPU authority remains until normal hold release; a newly allocated hold may establish a new identity.
void rc_registry_image_hold_forget(RcRegistry *, RcRegistryWrite);
int rc_registry_image_binding_current(RcRegistry *, RcRegistryImageBinding);
// Broad range ownership is independent of eligibility for a typed physical-image receipt. A forgotten image
// identity still excludes CPU authority until the live hold is actually released.
int rc_registry_image_hold_covers(RcRegistry *, RcRegistryWrite, RcSourceRange);
// One compact position entry per shared IMAGE writer/operation, with reference counts for repeated claims.
// Every reference, including the initial writer, must be added before sealing its declaration prefix and must
// be resolved exactly once by the adapter. NULL plus canceled=0 records unknown/partial representation coverage.
// Capacity failure preserves the broad GPU obligation and disables typed receipts for that writer.
RcRegistryResult rc_registry_image_producer_add(RcRegistry *, RcRegistryWrite, RcRegistryPosition);
RcRegistryResult rc_registry_image_producer_resolve(RcRegistry *, RcRegistryWrite, RcRegistryPosition,
    const RcRegistryImageBinding *, int canceled);
// Read-only diagnostic snapshot of the exact writer ticket and producer position.
typedef struct {
    RcInterval interval;
    RcRegistryImageBinding image;
    uint32_t unresolved, active;
    int unknown, projected, incomplete, exact_effect;
} RcRegistryImageProducerState;
int rc_registry_image_producer_state(RcRegistry *, RcRegistryWrite, RcRegistryPosition,
    RcRegistryImageProducerState *);
// An image that carries projected coverage was recorded. Its command-local reservation carries actual native fate even
// when the persistent image hold already owns the bytes and ordinary effect completion has no work left.
int rc_registry_image_producer_projection_pending(RcRegistry *, RcRegistryWrite writer,
    RcRegistryPosition output, RcRegistryImageBinding);
void rc_registry_image_producer_native_completed(RcRegistry *, RcRegistryWrite writer,
    RcRegistryPosition output, RcRegistryImageBinding);
int rc_registry_image_producer_projection_abandon(RcRegistry *, RcRegistryWrite writer,
    RcRegistryPosition output, RcRegistryImageBinding, int may_have_executed);
// A recording adapter discovered that its previously resolved claim was incomplete. This never consumes another
// reference or fabricates cancellation; the shared writer remains an exclusion until real disposal.
void rc_registry_image_producer_invalidate(RcRegistry *, RcRegistryWrite);
// Attach actual unowned coverage to this handed output, or discard an explicitly unhanded output.
// Exact coverage is applied only on successful shared-image settlement. Failure retains broad authority.
RcRegistryResult rc_registry_image_producer_settle(RcRegistry *, RcRegistryWrite, RcRegistryPosition, int executed);
RcRegistryResult rc_registry_image_effects_settle(RcRegistry *, RcRegistryWrite,
                                                const RcRegistryImageEffectSpan *, uint32_t count);
RcRegistryResult rc_registry_binding_commit(RcRegistry *, const RcRegistryBindingObservation *,
                                            RcRegistryBindingReceipt *);
RcRegistryResult rc_registry_binding_revalidate(RcRegistry *, const RcRegistryBindingReceipt *, uint64_t scope,
                                                RcRegistryPosition, RcRegistryBindingReceipt *);
// Renew only a prefix-limited preparation, never one expired at an earlier real producer. Newly exposed active
// producer boundaries must not have been crossed; a fresh begin/commit preparation is required when they have.
RcRegistryResult rc_registry_binding_renew(RcRegistry *, const RcRegistryBindingReceipt *, uint64_t scope,
                                          RcRegistryPosition, RcRegistryBindingReceipt *);
RcRegistryResult rc_registry_read_schedule(RcRegistry *, const RcRegistryReadObservation *, uint64_t scope,
                                          RcRegistryPosition, RcRegistryReadSchedule *);
RcRegistryResult rc_registry_binding_schedule(RcRegistry *, const RcRegistryBindingReceipt *, uint64_t scope,
                                             RcRegistryPosition, RcRegistryBindingSchedule *);
int rc_registry_position_open(RcRegistry *, RcRegistryPosition);

// Reserve before issuing a writer. A zero output on failure is NOT evidence of clean source data: exhausted or
// unbounded/unidentified authority fails all certification closed until the registry is recreated after a drain.
RcRegistryResult rc_registry_write_begin(RcRegistry *, RcSourceRange, RcRegistryWriterKind, RcRegistryWrite *);
RcRegistryResult rc_registry_write_begin_at(RcRegistry *, RcSourceRange, RcRegistryWriterKind,
                                           RcRegistryPosition, RcRegistryWrite *);
// All extents enter the declared position together, or none do. Caller-owned output has one ticket per extent.
// A refused declaration cannot be emitted. An all-zero position is the unpositioned compatibility boundary.
RcRegistryResult rc_registry_write_begin_extents(RcRegistry *, const RcRegistryWriteExtent *, uint32_t count,
                                                RcRegistryPosition, RcRegistryWrite *);
// An opaque producer whose accessible mappings may expand must declare its guard before sealing. It excludes
// canonical backings at/after its position, including mappings created after seal. A zero position guards an
// unpositioned producer immediately. Backing 1 is the separate raw compatibility lane and needs its own writer.
// Use bounded extents whenever reachability is known. The temporary guard is separate from any backing ID.
RcRegistryResult rc_registry_write_guard_begin_at(RcRegistry *, RcRegistryWriterKind, RcRegistryPosition, RcRegistryWrite *);
RcRegistryResult rc_registry_write_guard_begin_reason_at(RcRegistry *, RcRegistryWriterKind, RcRegistryPosition,
                                                         RcRegistryGuardReason, RcRegistryWrite *);
// Failure diagnostics use these actual native handoff watermarks, independently of registry declaration/submit.
// A partial scene watermark establishes only that some part reached the queue.
void rc_registry_runtime_native_handoff(uint64_t final, uint64_t part, uint64_t uncertain);
// After completion or a drain of possible execution, preserve unknown canonical coverage in one registry-owned
// guard. A new accumulation invalidates an older recovery ticket. Query the latest ticket for exhaustive recovery.
RcRegistryResult rc_registry_write_guard_accumulate(RcRegistry *, RcRegistryWrite);
RcRegistryWrite rc_registry_unresolved_guard(RcRegistry *);
// Expand an existing pending declaration by a conservative bounding union. This is a mapping event that revokes
// outstanding content proofs. Newly discovered aliases must already be excluded from ordinary observation, or
// covered by a predeclared unknown-domain guard; expansion cannot repair a previously accepted earlier read.
RcRegistryResult rc_registry_write_expand(RcRegistry *, RcRegistryWrite, RcSourceRange);
RcRegistryResult rc_registry_write_submit(RcRegistry *, RcRegistryWrite, uint64_t submission);
RcRegistryResult rc_registry_write_cancel(RcRegistry *, RcRegistryWrite);
RcRegistryResult rc_registry_write_submit_extents(RcRegistry *, const RcRegistryWrite *, uint32_t count, uint64_t submission);
RcRegistryResult rc_registry_write_cancel_extents(RcRegistry *, const RcRegistryWrite *, uint32_t count);
// Dispose an abandoned recording after proving it never reached the GPU, or after draining every possible GPU
// user. A possibly executed write retains foreign authority in place; this does not report successful execution.
RcRegistryResult rc_registry_write_abandon(RcRegistry *, RcRegistryWrite, int may_have_executed);
RcRegistryResult rc_registry_write_abandon_extents(RcRegistry *, const RcRegistryWrite *, uint32_t count);
// Existing authoritative images may predate a particular recorded submission. A hold protects that ownership
// without inventing GPU completion. Verified CPU visibility removes this hold; otherwise its release leaves foreign
// authority. Dropping an image or seeing a dirty CPU page cannot make stale guest bytes current.
RcRegistryResult rc_registry_hold_begin(RcRegistry *, RcSourceRange, RcRegistryWriterKind, RcRegistryWrite *);
RcRegistryResult rc_registry_hold_release(RcRegistry *, RcRegistryWrite, int cpu_visible);
// One actual image incarnation owns all fully validated normalized extents, or no hold is created.
RcRegistryResult rc_registry_image_hold_begin_extents(RcRegistry *, uint64_t image, const RcInterval *,
                                                      uint32_t count, RcRegistryWrite *);
// The normalized claim names this actual held physical image. Its existing completion owns the output fate.
int rc_registry_image_hold_recorded(RcRegistry *, RcRegistryWrite claim, RcRegistryImageBinding,
                                    RcRegistryPosition output, int first_write);
// The adapter has issued the ordered in-place operation on an initialized independent image. The actual claim
// carries this continuation only if resolution returns its exact retained owner. The adapter can attest
// that the selected native write view preserves the existing complete byte projection layout.
int rc_registry_image_hold_preserved(RcRegistry *, RcRegistryWrite claim, RcRegistryImageBinding,
                                     RcRegistryPosition output, int exact_projection_layout);
void rc_registry_image_hold_unordered(RcRegistry *, uint64_t image, const RcRegistryWrite *holds, uint32_t count);
RcRegistryResult rc_registry_image_hold_release_extents(RcRegistry *, const RcRegistryWrite *, uint32_t count,
                                                        int cpu_visible);
// A native-bound image owns these precise device extents after an actual recorded write. The read checks
// this output and all overlapping live writers at its own position. A positionless write revokes eligibility.
int rc_registry_native_image_written(RcRegistry *, uint64_t image, const RcInterval *, const RcRegistryWrite *,
                                      uint32_t count, RcRegistryPosition output);
int rc_registry_native_image_read_current(RcRegistry *, uint64_t image, const RcInterval *, const RcRegistryWrite *,
                                           uint32_t count, RcRegistryPosition output, RcRegistryPosition read);
typedef struct {
    RcRegistryImageWriteExtent owner;
    const RcSourceRange *written;
    uint32_t written_count;
} RcRegistryNativeImageWrite;
typedef struct {
    RcInterval interval;
    RcRegistryImageBinding binding;
    RcRegistryPosition output;
    uint32_t demand;
} RcRegistryNativeImageFragment;
typedef enum {
    RC_NATIVE_IMAGE_READ_OK, RC_NATIVE_IMAGE_READ_CLOSED, RC_NATIVE_IMAGE_READ_INVALID_DEMAND,
    RC_NATIVE_IMAGE_READ_NO_PRODUCER, RC_NATIVE_IMAGE_READ_AMBIGUOUS, RC_NATIVE_IMAGE_READ_TENTATIVE,
    RC_NATIVE_IMAGE_READ_BOUND, RC_NATIVE_IMAGE_READ_LOST_OWNER, RC_NATIVE_IMAGE_READ_DIFFERENT_IMAGE,
    RC_NATIVE_IMAGE_READ_UNHANDED, RC_NATIVE_IMAGE_READ_BLOCKED, RC_NATIVE_IMAGE_READ_CAPACITY,
} RcRegistryNativeImageGate;
typedef enum {
    RC_NATIVE_IMAGE_BLOCK_NONE, RC_NATIVE_IMAGE_BLOCK_UNORDERED_HOLD, RC_NATIVE_IMAGE_BLOCK_UNPROJECTED,
    RC_NATIVE_IMAGE_BLOCK_UNQUALIFIED_HOLD, RC_NATIVE_IMAGE_BLOCK_GUARD, RC_NATIVE_IMAGE_BLOCK_FOREIGN,
    RC_NATIVE_IMAGE_BLOCK_UNPOSITIONED, RC_NATIVE_IMAGE_BLOCK_OTHER_WRITER, RC_NATIVE_IMAGE_BLOCK_MISSING_OUTPUT,
    RC_NATIVE_IMAGE_BLOCK_PENDING_OUTPUT, RC_NATIVE_IMAGE_BLOCK_UNKNOWN_OUTPUT,
} RcRegistryNativeImageBlock;
typedef struct {
    RcInterval interval, blocking_interval;
    RcRegistryImageBinding selected;
    RcRegistryWrite blocker;
    RcRegistryPosition selected_output, blocking_output;
    RcRegistryNativeImageGate gate;
    RcRegistryNativeImageBlock cause;
    uint32_t demand, writer_kind, writer_state, guard_reason;
} RcRegistryNativeImageFailure;
// Optional maximum authored geometry for an independently placed, representable image. The adapter
// supplies all meaningful texel spans for the immutable allocation and recipe (sorted, nonoverlapping,
// backing-relative, clipped to each exact normalized hold). An empty span list certifies an extent of
// padding only. This grants no present or future data, visibility, or ordering. On capacity refusal the
// old conservative whole-hold authority remains. Repeat with an identical shape is allocation-free.
RcRegistryResult rc_registry_native_image_possible(RcRegistry *, uint64_t image,
    const RcRegistryNativeImageWrite *layouts, uint32_t count);
// Atomically consume an exact independent hold vector after its meaningful bytes are superseded
// by positively completed owners. Unwritten gaps add no obligation. Refusal does not change custody.
RcRegistryResult rc_registry_native_image_discard_superseded(RcRegistry *, uint64_t image,
    const RcRegistryWrite *holds, uint32_t count);
// The adapter proves reclamation of the exact explicit backing. Consume only this independent image's
// matching hold vector, without publishing visibility or waiting for a successor on a dead backing.
RcRegistryResult rc_registry_native_image_discard_reclaimed(RcRegistry *, uint64_t image, uint64_t backing,
    const RcRegistryWrite *holds, uint32_t count);
// Report actual ordered writes to an independently placed image. Each owner is a live typed hold; its
// exact nonempty written spans are sorted, disjoint and backing-relative. The whole vector publishes together.
RcRegistryResult rc_registry_native_image_write_fragments(RcRegistry *, uint64_t image,
    const RcRegistryNativeImageWrite *, uint32_t count, RcRegistryPosition output);
// Results partition every demand in input order (including when a different guest alias repeats the same
// backing bytes). Each interval is in the producer's captured mapping. Returns zero fragments on failure.
// The caller retains actual native providers and repeats this query at the original read before using them.
RcRegistryResult rc_registry_native_image_fragments(RcRegistry *, const RcInterval *demands, uint32_t count,
    RcRegistryPosition read, RcRegistryNativeImageFragment *, uint32_t capacity, uint32_t *fragment_count);
// Reports the exact first refused meaningful slice and the actual failing authority predicate from that sweep.
RcRegistryResult rc_registry_native_image_fragments_checked(RcRegistry *, const RcInterval *demands, uint32_t count,
    RcRegistryPosition read, RcRegistryNativeImageFragment *, uint32_t capacity, uint32_t *fragment_count,
    RcRegistryNativeImageFailure *);
// An ordered copy may retain a still-current whole-written bound native image as well. The adapter
// must verify the actual binding and layout, retain its memory, and record the snapshot before later work.
RcRegistryResult rc_registry_native_image_snapshot_fragments(RcRegistry *, const RcInterval *, uint32_t count,
    RcRegistryPosition read, RcRegistryNativeImageFragment *, uint32_t capacity, uint32_t *fragment_count,
    RcRegistryNativeImageFailure *);
typedef struct {
    RcRegistryNativeImageFragment fragment;
    RcRegistryBufferLocation native_buffer;
    uint32_t baseline;
} RcRegistryNativeImageSnapshotFragment;
// Each baseline is a retained, positively initialized device-only physical buffer in the actual current mapping.
// Its caller proves backing lifetime and that writes in this domain preserve ordered registry authority. A virtual
// mapping alone supplies no origin. Baselines may cover subranges of a demand. The snapshot partitions at
// their affine boundaries as well as ordered writer coverage, and conflicting raw receipts grant no raw read.
// Image fragments have baseline UINT32_MAX and no buffer. Buffer fragments name the exact baseline and affine
// native offset, with a zero output only for its initialized origin. No fragment is returned on failure.
RcRegistryResult rc_registry_native_image_snapshot_mixed(RcRegistry *, const RcInterval *, uint32_t count,
    const RcRegistryBufferSpan *baselines, uint32_t baseline_count, RcRegistryPosition read,
    RcRegistryNativeImageSnapshotFragment *, uint32_t capacity, uint32_t *fragment_count,
    RcRegistryNativeImageFailure *);
// True only if every demand has the same validated native winner. The caller still retains that image and
// validates its actual mapping and LOAD operation before committing a new output.
// Inclusive selection at the CPU caller's exact recorded cutoff. A changed cutoff is stale.
RcRegistryResult rc_registry_native_image_snapshot_handed(RcRegistry *, const RcInterval *, uint32_t count,
    const RcRegistryBufferSpan *, uint32_t baseline_count, RcRegistryPosition cut,
    RcRegistryNativeImageSnapshotFragment *, uint32_t capacity, uint32_t *fragment_count,
    RcRegistryNativeImageFailure *);
int rc_registry_native_image_handed_requires_coverage(RcRegistry *, const RcInterval *, uint32_t, RcRegistryPosition cut);
int rc_registry_native_image_covers(RcRegistry *, uint64_t image, const RcInterval *demands, uint32_t count,
                                    RcRegistryPosition read);
// Every requested texel needs a confirmed initialized predecessor on its exact retained placement. A later
// in-place native continuation can carry that initialization forward. This certifies only a native bind.
int rc_registry_native_image_resident_covers(RcRegistry *, uint64_t image, const RcRegistryWrite *, uint32_t hold_count,
    const RcInterval *demands, uint32_t count, RcRegistryPosition read);
// Covers the actual inclusive sealed replay frontier for a GPU consumer in the same ordered queue domain.
// This validates bytes, not GPU completion. The adapter retains the selected image and synchronizes its use.
int rc_registry_native_image_covers_handed(RcRegistry *, uint64_t image, const RcInterval *demands, uint32_t count);
// Advisory discovery before native allocation. A prior tentative or confirmed projected span intersects
// one of the captured backing ranges. Ordinary source/visibility checks still apply to every selected read.
int rc_registry_native_image_has_prior_projection(RcRegistry *, const RcInterval *, uint32_t count,
                                                   RcRegistryPosition read);
// Advisory discovery for retained bound image readers. No raw buffer or native binding permission is implied.
int rc_registry_native_image_has_prior_bound_projection(RcRegistry *, const RcInterval *, uint32_t count,
                                                         RcRegistryPosition read);
// True if the raw buffer could be stale due to current or lost independently placed image authority.
// This is a demand for a mixed proof, never a grant to consume the image or the buffer.
int rc_registry_native_image_buffer_requires_coverage(RcRegistry *, const RcInterval *, uint32_t count,
    RcRegistryPosition read);
// Record an actually emitted whole set of meaningful pixels of a physically bound image. Every holder
// has an exact nonempty span list and may include uninitialized physical padding outside that list.
// These spans cannot serve ordinary independent reads until a completed native copy preserves them.
RcRegistryResult rc_registry_native_image_bound_written(RcRegistry *, uint64_t image,
    const RcRegistryNativeImageWrite *, uint32_t count, RcRegistryPosition output);
// Convenience for an actual output whose meaningful writes cover every holder byte.
// This must run before command-effect release, while the original normalized output can settle its true fate.
int rc_registry_native_image_whole_written(RcRegistry *, uint64_t image,
    const RcRegistryImageWriteExtent *holders, uint32_t count, RcRegistryPosition output);
// After copying the proven full contents of a physically bound image into an independent image and
// completing that native prefix, transfer the same logical holds to the new provider at the original output.
// Only the meaningful ranges recorded at that original output move. Other bytes do not become known.
// The adapter validates actual guest mapping, copy and provider lifetime, and transfers hold cleanup atomically.
RcRegistryResult rc_registry_native_image_relocate(RcRegistry *, uint64_t image, uint64_t clone,
    const RcRegistryImageWriteExtent *holders, uint32_t count, RcRegistryPosition output, RcRegistryPosition read);
// Retire only the matching image hold set after a readback. Visible spans are sorted nonoverlapping guest byte
// ranges actually transferred to the captured destinations. Uncovered parts retain foreign authority.
RcRegistryResult rc_registry_image_hold_readback(RcRegistry *, uint64_t image, const RcRegistryImageWriteExtent *,
                                                 uint32_t count, const RcSourceRange *visible, uint32_t visible_count);
// cpu_visible means complete authoritative bytes really reached guest memory, not simply that execution finished.
// Otherwise completion retains a bounded foreign interval. Repeated overlapping foreign intervals coalesce.
RcRegistryResult rc_registry_write_complete(RcRegistry *, RcRegistryWrite, int cpu_visible);
// After actual submission completion, replace one broad reservation with precisely the uncertain subsets. The
// adapter proves the complement CPU-visible or unwritten. Every range must be inside the reservation. Allocation
// refusal retains the broad authority unchanged; observers never see a clean gap between parent and subsets.
RcRegistryResult rc_registry_write_complete_ranges(RcRegistry *, RcRegistryWrite,
                                                  const RcSourceRange *, uint32_t count);
// Exact actual-write coverage, separate from bytes whose CPU visibility is still uncertain. Retiring the broad
// reservation invalidates read authorization over that reservation, but changes content provenance only within
// written. CPU-visible writes must remain in written. Every uncertain range must be contained in one written
// range; adapters coalesce coverage where necessary. On refusal the original obligation remains intact.
RcRegistryResult rc_registry_write_complete_effects(RcRegistry *, RcRegistryWrite,
    const RcSourceRange *written, uint32_t written_count, const RcSourceRange *uncertain, uint32_t uncertain_count);
// Validate and settle every supplied submitted member atomically. A separately handed/canceled split uses only
// its own tickets. Completing a guard with this API requires exhaustive backing-relative coverage from the
// adapter, including old/new mappings reachable during execution. This same owned proof can recover a guard
// left foreign by a drained abandonment. A byte-range visibility proof or event history alone cannot settle it.
RcRegistryResult rc_registry_write_complete_extents(RcRegistry *, const RcRegistryWriteCompletion *, uint32_t count);
// Settle an already handed GPU address-table writer. Only the actual definitely stored canonical
// pieces have a native donor. Other affected pieces remain untyped and cannot overwrite a snapshot.
RcRegistryResult rc_registry_write_complete_gpa_projection(RcRegistry *, const RcRegistryWriteCompletion *,
    const RcRegistryBufferSpan *definite, uint32_t definite_count);
// A real visibility/replacement proof removes its byte coverage from completed foreign intervals, splitting them
// when necessary. Split-capacity refusal leaves all authority unchanged. Pending overlapping writers refuse it.
RcRegistryResult rc_registry_reconcile_cpu(RcRegistry *, RcSourceRange);
// Capture before recording/performing an exact GPU-to-guest copy. ORDERED requires that the copy follows all
// reserved producers through this cutoff, and that their completion callbacks run before apply. The caller sets
// submission to the real scene fence whose final copy follows every producer in that scene; globally earlier
// reservations submitted in later scenes cannot be cleared. Unordered/abandoned authority is also retained.
// COMPLETED ignores submission and is
// suitable when the copy only covers authority already completed at capture, excluding outstanding recordings.
// Apply only after successful execution and actual CPU visibility of the entire supplied range. It removes only
// eligible foreign coverage: pending/submitted writers and image holds remain, including older ones. Newer foreign
// authority or remapping cannot be cleared by an old proof. Coalesced mixed-age coverage is conservatively retained.
// begin selects BUFFER provenance only. Other bits (1u << RcRegistryWriterKind) require a copy that actually covers
// those producers' authoritative backing; buffer mirrors alone cannot prove image or unknown backing contents.
// This value proof does not own storage; there is no release operation. The same proof may cover several copy spans.
RcRegistryResult rc_registry_visibility_begin(RcRegistry *, RcRegistryVisibilityMode, RcRegistryVisibility *);
RcRegistryResult rc_registry_visibility_apply(RcRegistry *, const RcRegistryVisibility *, RcSourceRange);
void rc_registry_fail_closed(RcRegistry *);
// Bounded visited records, not merely bounded successful destruction. No periodic all-history traversal is required.
uint32_t rc_registry_collect(RcRegistry *, uint32_t visit_budget);
RcRegistryStats rc_registry_stats(RcRegistry *);
// Compatibility memo guard while callers migrate to per-resource captures. Zero disables reuse after exhaustion.
uint64_t rc_registry_authority_epoch(RcRegistry *);
// Changes that can invalidate a finite ordered proof. Future declarations/cancellation beyond or within its
// captured sealed prefix do not advance this epoch; unpositioned authority, actual writes and identities do.
// The legacy authority epoch above keeps its original all-transition semantics.
uint64_t rc_registry_ordered_epoch(RcRegistry *);
typedef struct {
    RcRegistryHandle handle;
    uint64_t object_serial, revision;
    RcSourceRange range;
    uint64_t association;
} RcRegistryChange;
// Source-only hard ordered-authority invalidations. revision is the ordered revision, not the legacy topology
// revision. Positioned additions and cancellation produce no entries; finite proofs retain their old deadline.
// Overflow explicitly requires rebuilding that caller's view.
uint32_t rc_registry_changes(RcRegistry *, uint64_t *cursor, RcRegistryChange *out, uint32_t capacity, int *overflow);
uint64_t rc_registry_change_position(RcRegistry *);
enum { RC_REGISTRY_SCHEDULE_ADD, RC_REGISTRY_SCHEDULE_EXPAND };
typedef struct {
    RcRegistryHandle handle;
    uint64_t object_serial, topology_revision;
    RcSourceRange range;
    RcRegistryWrite writer;
    RcRegistryPosition position;
    uint32_t kind;
    uint64_t association;
} RcRegistryScheduleChange;
// Separate from hard/source-content changes. Source-only ADD and coverage expansion events; removals need no
// invalidation of existing permission. Overflow requires schedule reconciliation, never discarding source bytes.
uint32_t rc_registry_schedule_changes(RcRegistry *, uint64_t *cursor, RcRegistryScheduleChange *out,
                                      uint32_t capacity, int *overflow);
uint64_t rc_registry_schedule_change_position(RcRegistry *);

#endif

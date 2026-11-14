#ifndef REPLAY_PHASE_TRACE_H
#define REPLAY_PHASE_TRACE_H
#include <stdint.h>

#ifndef NX2NAVI_REPLAY_PHASES
#define NX2NAVI_REPLAY_PHASES 0
#endif

#define RP_PHASES(X) \
    X(FRONT_PREPARE, "front-submit-prepare") \
    X(REQUEST, "submit-request") \
    X(PRESENT_DRAIN, "present-drain") \
    X(GATE_WAIT, "submit-gate") \
    X(TEXTURE_SUBMIT, "texture-submit") \
    X(SUBMIT, "native-submit") \
    X(SUBMIT_LOCK, "submit-lock") \
    X(SLOT_BEGIN, "slot-and-begin") \
    X(REPLAY, "replay") \
    X(SETUP, "replay-setup") \
    X(PIN_SWEEP, "pin-sweep") \
    X(EVENT_SYNC, "event-sync") \
    X(MIRROR_BEGIN, "mirror-begin") \
    X(MIRROR_LOCK, "mirror-locks") \
    X(MIRROR_QUERY, "mirror-query-ranges") \
    X(MIRROR_FREEZE, "mirror-freeze") \
    X(HELD_HANDLES, "held-handle-assembly") \
    X(USNAP_BEGIN, "uniform-snapshot") \
    X(USNAP_ADOPT, "snapshot-adopt") \
    X(USNAP_GATHER_LOCK, "snapshot-gather-lock") \
    X(USNAP_GATHER, "snapshot-gather") \
    X(USNAP_SORT, "snapshot-sort-prune") \
    X(USNAP_VALIDATE, "snapshot-merge-validate") \
    X(USNAP_ARENA, "snapshot-arena") \
    X(USNAP_COPY, "snapshot-copy-flush") \
    X(CONT_RESUME, "continuation-resume") \
    X(CONT_WALK, "continuation-walk") \
    X(COMMAND_WALK, "command-walk") \
    X(GROUP_DRAIN, "group-drain") \
    X(TEARDOWN, "replay-teardown") \
    X(GROUP_FRONT, "group-front") X(OWNER_HELP, "owner-help") X(FINAL_SPIN, "group-final-spin") \
    X(CHUNK_HOOK, "chunk-hook-wait") X(CHUNK_SETUP, "chunk-setup") \
    X(CHUNK_BODY, "chunk-body") X(CHUNK_TEARDOWN, "chunk-teardown") \
    X(RING_SPACE, "recorder-ring-space") X(REC_DRAIN, "recorder-drain")

#define RP_KEYS(X) \
    X(FENCE, "fence") \
    X(REPLAY_SEQ, "replay-sequence") \
    X(HANDLES, "handles") \
    X(HELD_BEFORE, "held-before") \
    X(HELD_AFTER, "held-after") \
    X(HELD_HANDLES, "held-handles") \
    X(REGIONS_IN, "regions-collected") \
    X(REGIONS_KEPT, "regions-kept") \
    X(BYTES, "bytes") \
    X(REFUSED, "refused") \
    X(INFLIGHT, "inflight") \
    X(OUTCOME, "outcome") \
    X(QUERIES, "queries") \
    X(WW_BYTES, "write-watch-bytes") \
    X(PINS_FREED, "pins-freed") \
    X(RECORDED_LAST, "recorded-last-part") \
    X(CONT_SINCE, "continuation-last-stopped-sequence") \
    X(CONT_INDEX, "continuation-handle-index") \
    X(WORK_UNIT, "work-unit")

#define RP_ENUM(name, text) RP_##name,
enum { RP_NONE, RP_PHASES(RP_ENUM) RP_PHASE_COUNT };
#undef RP_ENUM
#define RP_ENUM(name, text) RPK_##name,
enum { RPK_NONE, RP_KEYS(RP_ENUM) RP_KEY_COUNT };
#undef RP_ENUM

// Outcomes describe existing branches; they never select a different resource or ordering path.
enum { RPO_UNSPECIFIED, RPO_DISABLED, RPO_NO_WORK, RPO_ADOPTED, RPO_NO_REGIONS,
       RPO_NO_ARENA, RPO_FRESH_COPY, RPO_IMPORTED, RPO_MIRRORED };

#if NX2NAVI_REPLAY_PHASES
typedef struct { uint64_t id, parent; } RpScope;
RpScope gfx_rp_begin(unsigned phase);
void gfx_rp_end(RpScope *scope);
void gfx_rp_value(unsigned key, uint64_t value);
void gfx_rp_replay_reset(void);
void gfx_rp_ready(uint64_t ready);
#define RP_SCOPE(name, phase) RpScope name __attribute__((cleanup(gfx_rp_end))) = gfx_rp_begin(phase)
#define RP_END(name) gfx_rp_end(&(name))
#define RP_CALL(phase, expression) ({ RP_SCOPE(_rp_call_scope, phase); (expression); })
#define RP_VALUE(key, value) gfx_rp_value(key, (uint64_t)(value))
#define RP_REPLAY_RESET() gfx_rp_replay_reset()
#define RP_READY(id) gfx_rp_ready(id)
#else
#define RP_SCOPE(name, phase) ((void)0)
#define RP_END(name) ((void)0)
#define RP_CALL(phase, expression) (expression)
#define RP_VALUE(key, value) ((void)0)
#define RP_REPLAY_RESET() ((void)0)
#define RP_READY(id) ((void)0)
#endif
#endif

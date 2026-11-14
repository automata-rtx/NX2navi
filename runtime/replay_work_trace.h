#ifndef REPLAY_WORK_TRACE_H
#define REPLAY_WORK_TRACE_H
#include <stdint.h>
#ifndef NX2NAVI_REPLAY_WORK
#define NX2NAVI_REPLAY_WORK 0
#endif
#ifndef NX2NAVI_REPLAY_WORK_DESCRIPTORS
#define NX2NAVI_REPLAY_WORK_DESCRIPTORS 0
#endif

#define RWT_ROLES(X) \
    X(OWNER, "owner-serial") X(HELP, "owner-help") X(WORKER, "worker")
#define RWT_KINDS(X) \
    X(DRAW, "draw") X(MESH, "mesh") X(COMPUTE, "compute") X(UPDATE, "update")
#if NX2NAVI_REPLAY_WORK_DESCRIPTORS
#define RWT_DESCRIPTOR_STAGES(X) \
    X(PACKET_COMPARE, "packet-byte-comparison") X(PACKET_COPY, "packet-arena-copy") \
    X(DESC_ALLOC, "descriptor-allocation-api") X(DESC_REFILL, "descriptor-allocation-overhead") \
    X(DESC_UPDATE, "descriptor-update") X(POOL_READY, "pool-readiness") \
    X(POOL_WAIT, "pool-publication-or-turn-wait") X(POOL_LOCK, "pool-mutex-wait") \
    X(POOL_REFRESH, "pool-refresh")
#else
#define RWT_DESCRIPTOR_STAGES(X)
#endif
#define RWT_STAGES(X) \
    X(OTHER, "other") X(TARGET, "pass-target") X(SHADER, "shader-selection") \
    X(PIPELINE, "pipeline-selection") X(STATE, "state-and-hooks") \
    X(RECIPE, "recipe-lookup") X(INSTANCE, "instance-lookup-rebuild") \
    X(RESOLVE, "resource-resolve") X(DESCRIPTORS, "descriptor-packet-bind") \
    X(VERTEX, "vertex-input") X(DIRECT, "native-direct-wrapper") \
    X(PACK, "native-ring-wrapper") X(RING_WAIT, "ring-space-wait") X(DRAIN, "recorder-drain") \
    X(HISTORY, "diagnostic-history") RWT_DESCRIPTOR_STAGES(X)
#define RWT_COUNTS(X) \
    X(LOOKUPS, "cache-lookups") X(HITS, "cache-hits") X(MISSES, "cache-misses") \
    X(FIRST, "first-observed-context") X(REVISIT, "evicted-revisit") \
    X(KNOWN, "known-without-eviction") X(UNKNOWN, "history-unknown") \
    X(EVICTIONS, "evictions") X(EVICT_DISTANCE, "revisit-lookup-distance-sum") \
    X(HISTORY_OVERFLOW, "history-overflow") X(FENCE_RESET, "fence-reset") \
    X(CAPTURE_RESET, "capture-reset") X(RESET_DESCRIPTORS, "reset-descriptors") \
    X(REBUILD_DESCRIPTORS, "rebuild-descriptors") X(REUSED_STORAGE, "reused-storage") \
    X(ALLOCATE_STORAGE, "allocate-storage") X(REFUSED, "admission-refused") \
    X(RING_FALLBACK, "ring-fallback") X(ISSUED, "issued-graphics")
#define RWT_ENUM(name, text) RWT_ROLE_##name,
enum { RWT_ROLES(RWT_ENUM) RWT_ROLE_COUNT };
#undef RWT_ENUM
#define RWT_ENUM(name, text) RWT_KIND_##name,
enum { RWT_KINDS(RWT_ENUM) RWT_KIND_COUNT };
#undef RWT_ENUM
#define RWT_ENUM(name, text) RWT_STAGE_##name,
enum { RWT_STAGES(RWT_ENUM) RWT_STAGE_COUNT };
#undef RWT_ENUM
#define RWT_ENUM(name, text) RWT_COUNT_##name,
enum { RWT_COUNTS(RWT_ENUM) RWT_COUNT_COUNT };
#undef RWT_ENUM

#define RWT_CAP 1024u
#define RWT_SAMPLE_LOG2 5u
#define RWT_HISTORY_SLOTS 4096u
#define RWT_HISTORY_PROBES 16u

#if NX2NAVI_REPLAY_WORK
typedef struct { uintptr_t recipe; uint64_t last_evicted; } RwtHistoryEntry;
typedef struct {
    RwtHistoryEntry entry[RWT_HISTORY_SLOTS];
    uint64_t ordinal;
    uint32_t lost;
} RwtHistory;
typedef struct RwtUnit RwtUnit;
typedef struct { RwtUnit *unit, *previous; unsigned entered; } RwtUnitScope;
typedef struct RwtDrawScope {
    struct RwtDrawScope *previous;
    RwtUnit *unit;
    uint64_t begin, last;
    unsigned kind, stage, sampled, nested, stage_depth, entered;
} RwtDrawScope;
typedef struct { RwtDrawScope *draw; unsigned previous; } RwtStageScope;
RwtUnitScope gfx_rwt_unit_begin(unsigned role, uint64_t fence, uint64_t replay, uint64_t group, unsigned chunk);
void gfx_rwt_unit_end(RwtUnitScope *scope);
RwtDrawScope gfx_rwt_draw_begin(unsigned kind);
void gfx_rwt_draw_activate(RwtDrawScope *scope);
void gfx_rwt_draw_end(RwtDrawScope *scope);
RwtStageScope gfx_rwt_stage_begin(unsigned stage);
void gfx_rwt_stage_end(RwtStageScope *scope);
void gfx_rwt_count(unsigned key, uint64_t value);
void gfx_rwt_history_lookup(RwtHistory *h, const void *recipe, int hit);
void gfx_rwt_history_evict(RwtHistory *h, const void *recipe);
uint64_t gfx_rp_current(void);
uint64_t gfx_rwt_current_id(void);
uint64_t gfx_rwt_replay(void);
#define RWT_HISTORY_FIELD RwtHistory work_history;
#define RWT_UNIT(name, role, fence, replay, group, chunk) \
    RwtUnitScope name __attribute__((cleanup(gfx_rwt_unit_end))) = gfx_rwt_unit_begin(role, fence, replay, group, chunk)
#define RWT_UNIT_END(name) gfx_rwt_unit_end(&(name))
#define RWT_DRAW(name, kind) \
    RwtDrawScope name __attribute__((cleanup(gfx_rwt_draw_end))) = gfx_rwt_draw_begin(kind); gfx_rwt_draw_activate(&(name))
#define RWT_MAYBE_DRAW(name, enabled, kind) \
    RwtDrawScope name __attribute__((cleanup(gfx_rwt_draw_end))) = {0}; \
    if (enabled) { name = gfx_rwt_draw_begin(kind); gfx_rwt_draw_activate(&(name)); }
#define RWT_STAGE(name, stage) \
    RwtStageScope name __attribute__((cleanup(gfx_rwt_stage_end))) = gfx_rwt_stage_begin(stage)
#define RWT_STAGE_END(name) gfx_rwt_stage_end(&(name))
#define RWT_CALL(stage, expression) ({ RWT_STAGE(_rwt_call, stage); (expression); })
#define RWT_COUNT(key, value) gfx_rwt_count(key, (uint64_t)(value))
#define RWT_HISTORY_LOOKUP(cx, recipe, hit) gfx_rwt_history_lookup(&(cx)->work_history, recipe, hit)
#define RWT_HISTORY_EVICT(cx, recipe) gfx_rwt_history_evict(&(cx)->work_history, recipe)
#else
#define gfx_rwt_current_id() 0ull
#define gfx_rwt_replay() 0ull
#define RWT_HISTORY_FIELD
#define RWT_UNIT(name, role, fence, replay, group, chunk) ((void)0)
#define RWT_UNIT_END(name) ((void)0)
#define RWT_DRAW(name, kind) ((void)0)
#define RWT_MAYBE_DRAW(name, enabled, kind) ((void)0)
#define RWT_STAGE(name, stage) ((void)0)
#define RWT_STAGE_END(name) ((void)0)
#define RWT_CALL(stage, expression) (expression)
#define RWT_COUNT(key, value) ((void)0)
#define RWT_HISTORY_LOOKUP(cx, recipe, hit) ((void)0)
#define RWT_HISTORY_EVICT(cx, recipe) ((void)0)
#endif

// Extra subdivisions use the existing bounded draw sample. Ordinary profiling keeps its original stage costs.
#if NX2NAVI_REPLAY_WORK && NX2NAVI_REPLAY_WORK_DESCRIPTORS
#define RWD_STAGE(name, stage) RWT_STAGE(name, stage)
#define RWD_CALL(stage, expression) RWT_CALL(stage, expression)
#else
#define RWD_STAGE(name, stage) ((void)0)
#define RWD_CALL(stage, expression) (expression)
#endif
#endif

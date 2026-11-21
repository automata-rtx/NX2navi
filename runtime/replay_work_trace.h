#ifndef REPLAY_WORK_TRACE_H
#define REPLAY_WORK_TRACE_H
#include <stdint.h>
#ifndef NX2NAVI_REPLAY_WORK
#define NX2NAVI_REPLAY_WORK 0
#endif
#ifndef NX2NAVI_REPLAY_WORK_DESCRIPTORS
#define NX2NAVI_REPLAY_WORK_DESCRIPTORS 0
#endif
#ifndef NX2NAVI_REPLAY_WORK_POOL
#define NX2NAVI_REPLAY_WORK_POOL 0
#endif
#ifndef NX2NAVI_REPLAY_WORK_CHAIN
#define NX2NAVI_REPLAY_WORK_CHAIN 0
#endif

// These whole-unit regions include front preparation, which has no sampled draw scope. Nested regions partition
// wall time; residual is the unit's time outside them. They never put a clock around individual pool members.
#define RWT_REGIONS(X) \
    X(RESIDUAL, "residual") X(POOL_RESOLVE, "pool-resolve") X(POOL_LOCK, "pool-lock-wait") \
    X(TEXTURE_LOCK, "texture-lock-wait") X(POOL_WAIT, "pool-publication-wait") \
    X(POOL_SYNC, "pool-sync") X(POOL_TAKE, "pool-take-set") X(POOL_CURRENT, "pool-current") \
    X(POOL_ORDER, "pool-order-check") X(POOL_ENTRIES, "pool-sync-entries") \
    X(POOL_FRONTIER, "pool-frontier") X(POOL_REBUILD, "pool-frontier-rebuild") \
    X(POOL_INVALIDATE, "pool-authority-invalidation") X(POOL_DUE, "pool-due-members") \
    X(POOL_CATCH, "pool-catch-up") X(POOL_PUBLISH, "pool-publish") X(POOL_USE, "pool-publication-use") \
    X(POOL_ORDER_PUBLISH, "pool-order-publish") X(POOL_MANIFEST, "pool-publication-create-release") \
    X(PUBLICATION_LOCK, "publication-texture-lock-wait") X(PUBLICATION_CREATE, "publication-create-image-walk") \
    X(PUBLICATION_CANCEL, "publication-release-cancel") X(PUBLICATION_COLLECT, "publication-release-collect") \
    X(PUBLICATION_USE, "publication-use-nodes") \
    X(REPLAY_SERVICE, "replay-service") X(COMMAND_WALK, "command-walk") \
    X(GROUP_FRONT, "group-orchestration") X(CHUNK_BODY, "chunk-command-processing") \
    X(OP_DRAW, "operation-draw") X(OP_MESH, "operation-mesh") X(OP_COMPUTE, "operation-compute") \
    X(OP_UPDATE, "operation-update") X(OP_TRANSFER, "operation-transfer") X(OP_STATE, "operation-state") \
    X(DRAW_PREPARE, "draw-prepare") X(DRAW_BIND, "draw-bind-resources") \
    X(EFFECT_DECLARE, "effect-declare-and-seal") X(EFFECT_RECORDED, "effect-recorded") \
    X(EFFECT_RELEASE, "effect-release") X(COORDINATION, "recording-coordination") \
    X(COMPLETION, "completion-cleanup")
#define RWT_ENUM(name, text) RWT_REGION_##name,
enum { RWT_REGIONS(RWT_ENUM) RWT_REGION_COUNT };
#undef RWT_ENUM

#define RWT_ROLES(X) \
    X(OWNER, "owner-serial") X(HELP, "owner-help") X(WORKER, "worker") X(COMPLETION, "completion")
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
typedef struct { RwtUnit *unit, *previous; uint64_t begin; unsigned entered; } RwtUnitScope;
typedef struct {
    RwtUnit *unit;
    uint64_t begin, children;
    unsigned region, previous, depth;
} RwtRegionScope;
typedef struct RwtDrawScope {
    struct RwtDrawScope *previous;
    RwtUnit *unit;
    uint64_t begin, last;
    unsigned kind, stage, sampled, nested, stage_depth, entered;
} RwtDrawScope;
typedef struct { RwtDrawScope *draw; unsigned previous; } RwtStageScope;
RwtUnitScope gfx_rwt_unit_begin(unsigned role, uint64_t fence, uint64_t replay, uint64_t group, unsigned chunk);
RwtUnitScope gfx_rwt_completion_begin(uint64_t fence);
void gfx_rwt_unit_end(RwtUnitScope *scope);
RwtRegionScope gfx_rwt_region_begin(unsigned region);
void gfx_rwt_region_end(RwtRegionScope *scope);
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

#if NX2NAVI_REPLAY_WORK && NX2NAVI_REPLAY_WORK_POOL
#define RWT_REGION(name, region) \
    RwtRegionScope name __attribute__((cleanup(gfx_rwt_region_end))) = gfx_rwt_region_begin(region)
#define RWT_REGION_END(name) gfx_rwt_region_end(&(name))
#define RWT_REGION_CALL(region, expression) ({ RWT_REGION(_rwt_region_call, region); (expression); })
#else
#define RWT_REGION(name, region) ((void)0)
#define RWT_REGION_END(name) ((void)0)
#define RWT_REGION_CALL(region, expression) (expression)
#endif

#if NX2NAVI_REPLAY_WORK && NX2NAVI_REPLAY_WORK_CHAIN
#define RWT_CHAIN(name, region) \
    RwtRegionScope name __attribute__((cleanup(gfx_rwt_region_end))) = gfx_rwt_region_begin(region)
#define RWT_COMPLETION(name, fence) \
    RwtUnitScope name __attribute__((cleanup(gfx_rwt_unit_end))) = gfx_rwt_completion_begin(fence)
#else
#define RWT_CHAIN(name, region) ((void)0)
#define RWT_COMPLETION(name, fence) ((void)0)
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

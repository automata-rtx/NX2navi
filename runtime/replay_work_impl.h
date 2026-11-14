// Included once by vk_sync.c after the bounded timeline helpers. No rendering decisions depend on these records.
#if NX2NAVI_REPLAY_WORK
typedef struct {
    uint64_t eligible, sampled, nested, sampled_ns;
    uint64_t ns[RWT_STAGE_COUNT], calls[RWT_STAGE_COUNT];
} RwtDrawStats;
struct RwtUnit {
    uint64_t id, parent, replay, fence, group, begin, end;
    uint32_t tid, role, chunk, invalid;
    uint32_t rng[RWT_KIND_COUNT];
    RwtDrawStats draws[RWT_KIND_COUNT];
    uint64_t count[RWT_COUNT_COUNT];
};
static RwtUnit rwt_units[RWT_CAP];
static volatile uint32_t rwt_reserved;
static volatile uint8_t rwt_committed[RWT_CAP];
static _Thread_local RwtUnit *rwt_current;
static _Thread_local RwtDrawScope *rwt_draw;
uint64_t gfx_rwt_current_id(void) { return rwt_current ? rwt_current->id : 0; }
uint64_t gfx_rwt_replay(void) { return rwt_current ? rwt_current->replay : 0; }

static int32_t rwt_reserve(void) {
    uint32_t n = __atomic_load_n(&rwt_reserved, __ATOMIC_RELAXED);
    for (;;) {
        if ((n & TL_CLOSED) || n > RWT_CAP) return -1;
        if (__atomic_compare_exchange_n(&rwt_reserved, &n, n + 1, 1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
            return n < RWT_CAP ? (int32_t)n : -1;
    }
}

RwtUnitScope gfx_rwt_unit_begin(unsigned role, uint64_t fence, uint64_t replay, uint64_t group, unsigned chunk) {
    RwtUnitScope scope = {.previous = rwt_current, .entered = 1};
    rwt_current = NULL;
    if (!tl_recording()) return scope;
    int32_t i = rwt_reserve();
    if (i < 0) return scope;
    RwtUnit *u = &rwt_units[i];
    *u = (RwtUnit){.id = (uint64_t)i + 1, .parent = scope.previous ? scope.previous->id : 0,
        .replay = replay, .fence = fence, .group = group, .begin = plat_now_ns(),
        .tid = plat_thread_id(), .role = role, .chunk = chunk};
    // Units are scheduling boundaries, never nested inside a draw attempt. An unsupported overlap remains
    // unpublished so its outer clock cannot silently charge a child's work to the parent.
    for (RwtDrawScope *d = rwt_draw; d; d = d->previous) {
        if (d->unit) d->unit->invalid = 1, u->invalid = 1;
    }
    for (unsigned k = 0; k < RWT_KIND_COUNT; k++)
        u->rng[k] = 0x9e3779b9u ^ (uint32_t)fence * 1664525u ^ (uint32_t)group * 1013904223u ^
                    chunk * 69069u ^ role * 22695477u ^ k * 747796405u;
    scope.unit = rwt_current = u;
    return scope;
}

void gfx_rwt_unit_end(RwtUnitScope *scope) {
    if (!scope->entered) return;
    RwtUnit *u = scope->unit;
    if (u) {
        if (rwt_current != u) u->invalid = 1;
        for (RwtDrawScope *d = rwt_draw; d; d = d->previous)
            if (d->unit == u) u->invalid = 1;
        u->end = plat_now_ns();
        if (!u->invalid) __atomic_store_n(&rwt_committed[u->id - 1], 1, __ATOMIC_RELEASE);
    }
    rwt_current = scope->previous;
    scope->entered = 0;
}

RwtDrawScope gfx_rwt_draw_begin(unsigned kind) {
    RwtDrawScope s = {.previous = rwt_draw, .unit = rwt_current, .kind = kind, .stage = RWT_STAGE_OTHER, .entered = 1};
    if (!s.unit || kind >= RWT_KIND_COUNT) return s;
    RwtDrawStats *d = &s.unit->draws[kind];
    if (rwt_draw) { d->nested++; s.nested = 1; return s; }
    d->eligible++;
    uint32_t x = s.unit->rng[kind] * 1664525u + 1013904223u;
    s.unit->rng[kind] = x;
    s.sampled = (x >> (32 - RWT_SAMPLE_LOG2)) == 0;
    if (s.sampled) {
        d->sampled++;
        s.begin = s.last = plat_now_ns();
        d->calls[RWT_STAGE_OTHER]++;
    }
    return s;
}
void gfx_rwt_draw_activate(RwtDrawScope *scope) { rwt_draw = scope; }
void gfx_rwt_draw_end(RwtDrawScope *s) {
    if (!s->entered) return;
    if (s->stage_depth && s->unit) s->unit->invalid = 1;
    if (s->sampled) {
        uint64_t now = plat_now_ns();
        RwtDrawStats *d = &s->unit->draws[s->kind];
        d->ns[s->stage] += now - s->last;
        d->sampled_ns += now - s->begin;
    }
    if (rwt_draw != s && s->unit) s->unit->invalid = 1;
    rwt_draw = s->previous;
    s->entered = 0;
}

RwtStageScope gfx_rwt_stage_begin(unsigned stage) {
    RwtStageScope s = {0};
    RwtDrawScope *d = rwt_draw;
    if (!d || !d->sampled || stage >= RWT_STAGE_COUNT) return s;
    uint64_t now = plat_now_ns();
    d->unit->draws[d->kind].ns[d->stage] += now - d->last;
    s.draw = d; s.previous = d->stage;
    d->stage_depth++;
    d->stage = stage; d->last = now;
    d->unit->draws[d->kind].calls[stage]++;
    return s;
}
void gfx_rwt_stage_end(RwtStageScope *s) {
    if (!s->draw) return;
    RwtDrawScope *d = s->draw;
    if (d != rwt_draw) d->unit->invalid = 1;
    if (!d->stage_depth) d->unit->invalid = 1;
    else d->stage_depth--;
    uint64_t now = plat_now_ns();
    d->unit->draws[d->kind].ns[d->stage] += now - d->last;
    d->stage = s->previous; d->last = now;
    s->draw = NULL;
}
void gfx_rwt_count(unsigned key, uint64_t value) {
    if (rwt_current && key < RWT_COUNT_COUNT) rwt_current->count[key] += value;
}

static RwtHistoryEntry *rwt_history_find(RwtHistory *h, uintptr_t key, int insert) {
    uint32_t at = (uint32_t)((key >> 4) ^ (key >> 13) ^ (key >> 23)) & (RWT_HISTORY_SLOTS - 1);
    for (unsigned p = 0; p < RWT_HISTORY_PROBES; p++) {
        RwtHistoryEntry *e = &h->entry[(at + p) & (RWT_HISTORY_SLOTS - 1)];
        if (e->recipe == key) return e;
        if (!e->recipe) {
            if (insert) { e->recipe = key; return e; }
            return NULL;
        }
    }
    if (insert) { h->lost = 1; gfx_rwt_count(RWT_COUNT_HISTORY_OVERFLOW, 1); }
    return NULL;
}
void gfx_rwt_history_lookup(RwtHistory *h, const void *recipe, int hit) {
    RWT_STAGE(history_cost, RWT_STAGE_HISTORY);
    if (!recipe) return;
    h->ordinal++;
    gfx_rwt_count(RWT_COUNT_LOOKUPS, 1);
    gfx_rwt_count(hit ? RWT_COUNT_HITS : RWT_COUNT_MISSES, 1);
    RwtHistoryEntry *e = rwt_history_find(h, (uintptr_t)recipe, 0);
    if (e) {
        if (e->last_evicted) {
            gfx_rwt_count(RWT_COUNT_REVISIT, 1);
            gfx_rwt_count(RWT_COUNT_EVICT_DISTANCE, h->ordinal - e->last_evicted);
            e->last_evicted = 0;
        } else gfx_rwt_count(RWT_COUNT_KNOWN, 1);
    } else {
        int known_new = !h->lost;
        e = rwt_history_find(h, (uintptr_t)recipe, 1);
        gfx_rwt_count(known_new && e ? RWT_COUNT_FIRST : RWT_COUNT_UNKNOWN, 1);
    }
}
void gfx_rwt_history_evict(RwtHistory *h, const void *recipe) {
    RWT_STAGE(history_cost, RWT_STAGE_HISTORY);
    if (!recipe) return;
    gfx_rwt_count(RWT_COUNT_EVICTIONS, 1);
    RwtHistoryEntry *e = rwt_history_find(h, (uintptr_t)recipe, 0);
    if (e) e->last_evicted = h->ordinal ? h->ordinal : 1;
}

#define RWT_NAME(name, text) text,
static const char *const rwt_roles[] = {RWT_ROLES(RWT_NAME)};
static const char *const rwt_kinds[] = {RWT_KINDS(RWT_NAME)};
static const char *const rwt_stages[] = {RWT_STAGES(RWT_NAME)};
static const char *const rwt_counts[] = {RWT_COUNTS(RWT_NAME)};
#undef RWT_NAME
static void rwt_dump(uint64_t t0, uint64_t t1) {
    uint32_t reserved = __atomic_fetch_or(&rwt_reserved, TL_CLOSED, __ATOMIC_ACQ_REL) & ~TL_CLOSED;
    uint32_t limit = reserved < RWT_CAP ? reserved : RWT_CAP, committed = 0, emitted = 0;
    for (uint32_t i = 0; i < limit; i++) {
        if (!__atomic_load_n(&rwt_committed[i], __ATOMIC_ACQUIRE)) continue;
        const RwtUnit *u = &rwt_units[i];
        committed++;
        if (u->end < t0 || u->begin > t1) continue;
        emitted++;
        rt_log("vk: rw unit id %llu role %s tid %u replay %llu parent %llu fence %llu group %llu chunk %u begin %.6f end %.6f invalid %u",
            (unsigned long long)u->id, u->role < RWT_ROLE_COUNT ? rwt_roles[u->role] : "unknown", u->tid,
            (unsigned long long)u->replay, (unsigned long long)u->parent, (unsigned long long)u->fence,
            (unsigned long long)u->group, u->chunk, (double)(int64_t)(u->begin - t0) / 1e6,
            (double)(int64_t)(u->end - t0) / 1e6, u->invalid);
        for (unsigned k = 0; k < RWT_KIND_COUNT; k++) {
            const RwtDrawStats *d = &u->draws[k];
            rt_log("vk: rw draw unit %llu kind %s eligible %llu sampled %llu nested %llu sampled-ns %llu",
                (unsigned long long)u->id, rwt_kinds[k], (unsigned long long)d->eligible,
                (unsigned long long)d->sampled, (unsigned long long)d->nested, (unsigned long long)d->sampled_ns);
            for (unsigned s = 0; s < RWT_STAGE_COUNT; s++) if (d->calls[s] || d->ns[s])
                rt_log("vk: rw stage unit %llu kind %s stage %s sampled-ns %llu calls %llu",
                    (unsigned long long)u->id, rwt_kinds[k], rwt_stages[s],
                    (unsigned long long)d->ns[s], (unsigned long long)d->calls[s]);
        }
        for (unsigned c = 0; c < RWT_COUNT_COUNT; c++) if (u->count[c])
            rt_log("vk: rw count unit %llu key %s value %llu", (unsigned long long)u->id, rwt_counts[c],
                (unsigned long long)u->count[c]);
    }
    rt_log("vk: rw meta schema 1 capacity %u reserved %u committed %u emitted %u sample-log2 %u history-slots %u history-probes %u",
        RWT_CAP, reserved, committed, emitted, RWT_SAMPLE_LOG2, RWT_HISTORY_SLOTS, RWT_HISTORY_PROBES);
    if (reserved > RWT_CAP || committed != limit)
        rt_log("vk: rw incomplete: overflow or unpublished work units; attribution rejected");
}
#endif

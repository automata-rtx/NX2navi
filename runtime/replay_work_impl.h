// Included once by vk_sync.c after the bounded timeline helpers. No rendering decisions depend on these records.
#if NX2NAVI_REPLAY_WORK
typedef struct {
    uint64_t eligible, sampled, nested, sampled_ns;
    uint64_t ns[RWT_STAGE_COUNT], calls[RWT_STAGE_COUNT];
} RwtDrawStats;
struct RwtUnit {
    uint64_t id, parent, replay, fence, group, begin, end;
    uint32_t tid, role, chunk, invalid;
    uint64_t cpu_begin, cpu_ns, children_cpu_ns, children_ns, last;
    uint64_t region_ns[RWT_REGION_COUNT], region_inclusive_ns[RWT_REGION_COUNT], region_calls[RWT_REGION_COUNT];
    unsigned cpu_valid, region, region_depth;
    uint32_t rng[RWT_KIND_COUNT];
    RwtDrawStats draws[RWT_KIND_COUNT];
    uint64_t count[RWT_COUNT_COUNT];
};
static RwtUnit rwt_units[RWT_CAP];
static volatile uint32_t rwt_reserved;
static volatile uint8_t rwt_committed[RWT_CAP];
static _Thread_local RwtUnit *rwt_current;
static _Thread_local RwtDrawScope *rwt_draw;
static uint64_t rwt_unscoped_regions[RWT_REGION_COUNT];
#define RWT_UNSCOPED_CAP 256u
static struct {uint64_t at;unsigned region;} rwt_unscoped[RWT_UNSCOPED_CAP];
static volatile uint8_t rwt_unscoped_committed[RWT_UNSCOPED_CAP];
static uint32_t rwt_unscoped_count;
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

static void rwt_region_charge(RwtUnit *u, uint64_t now) {
    if (u->last) {
        if (now < u->last) u->invalid = 1;
        else u->region_ns[u->region] += now - u->last;
    }
    u->last = now;
}

RwtRegionScope gfx_rwt_region_begin(unsigned region) {
    RwtRegionScope s = {0};
    if (region >= RWT_REGION_COUNT) return s;
    RwtUnit *u = rwt_current;
    if (!u) {
        if (tl_recording()) {
            __atomic_fetch_add(&rwt_unscoped_regions[region], 1, __ATOMIC_RELAXED);
            uint32_t at=__atomic_fetch_add(&rwt_unscoped_count,1,__ATOMIC_RELAXED);
            if(at<RWT_UNSCOPED_CAP) {
                rwt_unscoped[at].at=plat_now_ns();rwt_unscoped[at].region=region;
                __atomic_store_n(&rwt_unscoped_committed[at],1,__ATOMIC_RELEASE);
            }
        }
        return s;
    }
    uint64_t now = plat_now_ns();
    rwt_region_charge(u, now);
    s = (RwtRegionScope){u, now, u->children_ns, region, u->region, ++u->region_depth};
    u->region = region;
    u->region_calls[region]++;
    return s;
}

void gfx_rwt_region_end(RwtRegionScope *s) {
    if (!s->unit) return;
    RwtUnit *u = s->unit;
    uint64_t now = plat_now_ns();
    if (rwt_current != u || u->region_depth != s->depth || u->region != s->region) u->invalid = 1;
    rwt_region_charge(u, now);
    uint64_t children = u->children_ns - s->children;
    if (now < s->begin || now - s->begin < children) u->invalid = 1;
    else u->region_inclusive_ns[s->region] += now - s->begin - children;
    u->region = s->previous;
    if (u->region_depth) u->region_depth--; else u->invalid = 1;
    s->unit = NULL;
}

RwtUnitScope gfx_rwt_unit_begin(unsigned role, uint64_t fence, uint64_t replay, uint64_t group, unsigned chunk) {
    RwtUnitScope scope = {.previous = rwt_current, .entered = 1};
    if (scope.previous) {
        scope.begin = plat_now_ns();
        rwt_region_charge(scope.previous, scope.begin);
        scope.previous->last = 0;
    }
    rwt_current = NULL;
    if (!tl_recording()) { if (scope.previous) scope.previous->invalid = 1; return scope; }
    int32_t i = rwt_reserve();
    if (i < 0) { if (scope.previous) scope.previous->invalid = 1; return scope; }
    RwtUnit *u = &rwt_units[i];
    *u = (RwtUnit){.id = (uint64_t)i + 1, .parent = scope.previous ? scope.previous->id : 0,
        .replay = replay, .fence = fence, .group = group, .begin = scope.previous ? scope.begin : plat_now_ns(),
        .tid = plat_thread_id(), .role = role, .chunk = chunk};
    u->last = u->begin;
    // GetThreadTimes reports kernel+user time in 100 ns units on Windows. Its accounting granularity is coarser
    // than that representation: retain raw deltas/validity and aggregate units, rather than claiming sub-ms precision.
    u->cpu_begin = plat_thread_cpu_ns(u->tid);
    u->cpu_valid = u->cpu_begin != 0;
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

// Completion callbacks can run on a different thread, outside recording. Reuse an existing unit when present;
// creating a nested unit inside a draw would disturb its sampled clock and falsely move time between roles.
RwtUnitScope gfx_rwt_completion_begin(uint64_t fence) {
    if(rwt_current)return (RwtUnitScope){0};
    return gfx_rwt_unit_begin(RWT_ROLE_COMPLETION,fence,0,0,0);
}

void gfx_rwt_unit_end(RwtUnitScope *scope) {
    if (!scope->entered) return;
    RwtUnit *u = scope->unit;
    uint64_t now = (u || scope->previous) ? plat_now_ns() : 0;
    if (u) {
        if (rwt_current != u) u->invalid = 1;
        for (RwtDrawScope *d = rwt_draw; d; d = d->previous)
            if (d->unit == u) u->invalid = 1;
        if (u->region_depth) u->invalid = 1;
        rwt_region_charge(u, now);
        u->end = now;
        uint64_t accounted = 0;
        for (unsigned r = 0; r < RWT_REGION_COUNT; r++) accounted += u->region_ns[r];
        if (now < u->begin || u->children_ns > now - u->begin ||
            accounted != now - u->begin - u->children_ns) u->invalid = 1;
        uint64_t cpu = plat_thread_cpu_ns(u->tid);
        if (!cpu || cpu < u->cpu_begin) u->cpu_valid = 0;
        else u->cpu_ns = cpu - u->cpu_begin;
        if (u->cpu_ns < u->children_cpu_ns) u->cpu_valid = 0;
        if (!u->invalid) __atomic_store_n(&rwt_committed[u->id - 1], 1, __ATOMIC_RELEASE);
    }
    if (scope->previous) {
        RwtUnit *parent = scope->previous;
        parent->children_ns += now - scope->begin;
        if (!u || !u->cpu_valid) parent->cpu_valid = 0;
        else parent->children_cpu_ns += u->cpu_ns;
        parent->last = now;
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
static const char *const rwt_regions[] = {RWT_REGIONS(RWT_NAME)};
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
        rt_log("vk: rw budget unit %llu inclusive-wall-ns %llu children-wall-ns %llu exclusive-wall-ns %llu "
               "inclusive-cpu-ns %llu children-cpu-ns %llu exclusive-cpu-ns %llu cpu-valid %u",
            (unsigned long long)u->id, (unsigned long long)(u->end - u->begin),
            (unsigned long long)u->children_ns, (unsigned long long)(u->end - u->begin - u->children_ns),
            (unsigned long long)u->cpu_ns, (unsigned long long)u->children_cpu_ns,
            (unsigned long long)(u->cpu_valid ? u->cpu_ns - u->children_cpu_ns : 0), u->cpu_valid);
        for (unsigned r = 0; r < RWT_REGION_COUNT; r++) if (u->region_calls[r] || u->region_ns[r])
            rt_log("vk: rw region unit %llu phase %s exclusive-ns %llu inclusive-ns %llu calls %llu",
                (unsigned long long)u->id, rwt_regions[r], (unsigned long long)u->region_ns[r],
                (unsigned long long)u->region_inclusive_ns[r], (unsigned long long)u->region_calls[r]);
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
    uint64_t inside[RWT_REGION_COUNT]={0},outside[RWT_REGION_COUNT]={0};
    uint32_t missing=0,unscoped=__atomic_load_n(&rwt_unscoped_count,__ATOMIC_ACQUIRE);
    for(uint32_t i=0;i<unscoped && i<RWT_UNSCOPED_CAP;i++) {
        if(!__atomic_load_n(&rwt_unscoped_committed[i],__ATOMIC_ACQUIRE)){missing++;continue;}
        uint64_t at=rwt_unscoped[i].at;
        (at>=t0 && at<=t1?inside:outside)[rwt_unscoped[i].region]++;
    }
    if(unscoped>RWT_UNSCOPED_CAP || missing)
        rt_log("vk: rw incomplete: unscoped event history overflow or unpublished events; attribution rejected");
    for(unsigned r=0;r<RWT_REGION_COUNT;r++) {
        if(inside[r])rt_log("vk: rw unscoped phase %s calls %llu; whole-unit coverage incomplete",rwt_regions[r],(unsigned long long)inside[r]);
        if(outside[r])rt_log("vk: rw outside-window unscoped phase %s calls %llu",rwt_regions[r],(unsigned long long)outside[r]);
    }
}
#endif

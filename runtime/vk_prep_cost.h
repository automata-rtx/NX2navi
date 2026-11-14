// Optional cost sampling for prepared descriptors. Recording contexts own counters; only coarse flushes are shared.
#pragma once
#ifndef NX2NAVI_PREP_COST_DIAGNOSTIC
#define NX2NAVI_PREP_COST_DIAGNOSTIC 0
#endif

#if NX2NAVI_PREP_COST_DIAGNOSTIC
enum {
    PC_BINDS, PC_NESTED, PC_SAMPLED, PC_CACHE_HIT, PC_CACHE_MISS, PC_FENCE_RESET, PC_CAPTURE_RESET, PC_RESET_DESCRIPTORS,
    PC_GROUP_FAST, PC_GROUP_DIRTY, PC_GROUP_STALE, PC_GROUP_BOTH, PC_GROUP_RECHECK,
    PC_GUARD_OK, PC_GUARD_UNSEALED, PC_GUARD_OVERRIDE, PC_GUARD_TEXTURE_MODE,
    PC_GUARD_MEMORY, PC_GUARD_SNAPSHOT, PC_GUARD_PROGRAM, PC_GUARD_TEXTURES, PC_GUARD_EPOCH,
    PC_GUARD_SAMPLERS, PC_GUARD_POOLS, PC_GUARD_FEEDBACK, PC_GUARD_PROOF,
    PC_OPS, PC_STABLE_SKIP, PC_STABLE_RESOLVE, PC_ORDERED, PC_ALL_STABLE_SKIP, PC_MIXED_RESOLVE,
    PC_RENEW_TRY, PC_RENEW_OK, PC_RENEW_FAIL, PC_FULL_SEAL, PC_SEAL_DESCRIPTORS, PC_SEAL_OK, PC_SEAL_FAIL,
    PC_POOL, PC_WRITE_STRUCTS, PC_NO_STABLE, PC_PARTIAL_SKIP, PC_PARTIAL_DESCRIPTORS, PC_PARTIAL_CHECKED,
    PC_EMPTY_RENEW_TRY, PC_EMPTY_RENEW_OK, PC_COUNT_N
};
enum {
    PCT_BIND, PCT_LOOKUP, PCT_INSTANCE, PCT_RESOLVE, PCT_RESOLVE_ALL_STABLE_SKIP, PCT_RESOLVE_MIXED,
    PCT_INITIAL_GUARDS, PCT_OP_LOOP, PCT_FINISH, PCT_GUARD, PCT_RENEW, PCT_FULL_SEAL, PCT_PACKET_PUSH, PCT_POOL,
    PCT_RESOLVE_NO_STABLE, PCT_N
};
typedef struct { uint64_t count[PC_COUNT_N], ns[PCT_N], samples[PCT_N]; } PrepCostStats;
typedef struct { PrepCostStats sum; uint32_t sequence, depth, active; } PrepCost;
static PrepCostStats prep_cost_totals;

static void prep_cost_count(PrepCost *p, unsigned item, uint64_t n) {
    if (p->depth == 1) p->sum.count[item] += n;
}
static uint64_t prep_cost_now(PrepCost *p) {
    return p->active && p->depth == 1 ? plat_now_ns() : 0;
}
static void prep_cost_time(PrepCost *p, unsigned item, uint64_t start) {
    if (start && p->depth == 1) {
        p->sum.ns[item] += plat_now_ns() - start;
        p->sum.samples[item]++;
    }
}
static void prep_cost_time_pair(PrepCost *p, unsigned a, unsigned b, uint64_t start) {
    if (start && p->depth == 1) {
        uint64_t ns = plat_now_ns() - start;
        p->sum.ns[a] += ns, p->sum.ns[b] += ns;
        p->sum.samples[a]++, p->sum.samples[b]++;
    }
}
static uint64_t prep_cost_begin(PrepCost *p) {
    if (p->depth++) { p->sum.count[PC_NESTED]++; return 0; }
    p->sum.count[PC_BINDS]++;
    // High bits avoid a fixed every-Nth-command phase. Sampling never changes resource decisions or input order.
    if (!p->sequence) p->sequence = (uint32_t)((uintptr_t)p ^ ((uintptr_t)p >> 32)) ^ 0x9e3779b9u;
    p->sequence = p->sequence * 1664525u + 1013904223u;
    p->active = (p->sequence >> 26) == 0;
    p->sum.count[PC_SAMPLED] += p->active;
    return prep_cost_now(p);
}
static void prep_cost_end(PrepCost *p, uint64_t start) {
    prep_cost_time(p, PCT_BIND, start);
    if (--p->depth == 0) p->active = 0;
}
static void prep_cost_flush(PrepCost *p) {
    for (unsigned i = 0; i < PC_COUNT_N; i++)
        if (p->sum.count[i]) __atomic_fetch_add(&prep_cost_totals.count[i], p->sum.count[i], __ATOMIC_RELAXED);
    for (unsigned i = 0; i < PCT_N; i++) {
        if (p->sum.ns[i]) __atomic_fetch_add(&prep_cost_totals.ns[i], p->sum.ns[i], __ATOMIC_RELAXED);
        if (p->sum.samples[i]) __atomic_fetch_add(&prep_cost_totals.samples[i], p->sum.samples[i], __ATOMIC_RELAXED);
    }
    memset(&p->sum, 0, sizeof p->sum);
}
static void prep_cost_collect(uint64_t frame) {
    static PrepCostStats sum;
    static uint64_t last;
    static const char *names[PC_COUNT_N] = {
        "binds", "nested", "sampled", "cache_hit", "cache_miss", "fence_reset", "capture_reset", "reset_desc",
        "group_fast", "group_dirty", "group_stale", "group_both", "group_recheck",
        "guard_ok", "unsealed", "override", "texture_mode", "memory", "snapshot", "program", "textures", "epoch",
        "samplers", "pools", "feedback", "proof", "ops", "stable_skip", "stable_resolve", "ordered",
        "all_stable_skip", "mixed_resolve", "renew_try", "renew_ok", "renew_fail", "full_seal", "seal_desc",
        "seal_ok", "seal_fail", "pool", "write_structs", "no_stable", "partial_skip", "partial_descriptors", "partial_checked",
        "empty_renew_try", "empty_renew_ok"
    };
    static const char *timings[PCT_N] = {
        "bind", "lookup", "instance", "resolve", "resolve_all_stable_skip", "resolve_mixed",
        "initial_guards", "op_loop", "finish", "guard", "renew", "full_seal", "packet_push", "pool", "resolve_no_stable"
    };
    for (unsigned i = 0; i < PC_COUNT_N; i++)
        sum.count[i] += __atomic_exchange_n(&prep_cost_totals.count[i], 0, __ATOMIC_RELAXED);
    for (unsigned i = 0; i < PCT_N; i++) {
        sum.ns[i] += __atomic_exchange_n(&prep_cost_totals.ns[i], 0, __ATOMIC_RELAXED);
        sum.samples[i] += __atomic_exchange_n(&prep_cost_totals.samples[i], 0, __ATOMIC_RELAXED);
    }
    if (frame / 60 == last) return;
    last = frame / 60;
    if (rt_trace & RT_TRACE_PERF) {
        char counts[4096], spans[2048];
        _Static_assert(PC_COUNT_N * 48 < sizeof counts, "counter text must hold every name and uint64 value");
        _Static_assert(PCT_N * 80 < sizeof spans, "span text must hold every name and two uint64 values");
        size_t a = 0, b = 0;
        for (unsigned i = 0; i < PC_COUNT_N; i++)
            a += (size_t)snprintf(counts + a, sizeof counts - a, "%s%s=%llu", i ? " " : "", names[i],
                                  (unsigned long long)sum.count[i]);
        for (unsigned i = 0; i < PCT_N; i++)
            b += (size_t)snprintf(spans + b, sizeof spans - b, "%s%s=%llu/%llu", i ? " " : "", timings[i],
                                  (unsigned long long)sum.ns[i], (unsigned long long)sum.samples[i]);
        rt_log("prep cost: frame %llu counts: %s", (unsigned long long)frame, counts);
        rt_log("prep cost: frame %llu sampled-ns/count: %s", (unsigned long long)frame, spans);
    }
    memset(&sum, 0, sizeof sum);
}
#define PC_FIELD PrepCost prep_cost;
#define PC_BEGIN(cx) prep_cost_begin(&(cx)->prep_cost)
#define PC_END(cx, start) prep_cost_end(&(cx)->prep_cost, start)
#define PC_NOW(cx) prep_cost_now(&(cx)->prep_cost)
#define PC_COUNT(cx, item) prep_cost_count(&(cx)->prep_cost, item, 1)
#define PC_ADD(cx, item, n) prep_cost_count(&(cx)->prep_cost, item, n)
#define PC_TIME(cx, item, start) prep_cost_time(&(cx)->prep_cost, item, start)
#define PC_TIME_PAIR(cx, a, b, start) prep_cost_time_pair(&(cx)->prep_cost, a, b, start)
#define PC_FLUSH(cx) prep_cost_flush(&(cx)->prep_cost)
#define PC_COLLECT() prep_cost_collect(__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED))
#define PC_GUARD_RETURN(ex, item, result) do { PC_COUNT((ex)->dc, item); \
    PC_TIME((ex)->dc, PCT_GUARD, prep_guard_start); return (result); } while (0)
#else
#define PC_FIELD
#define PC_BEGIN(cx) 0
#define PC_END(cx, start) ((void)0)
#define PC_NOW(cx) 0
#define PC_COUNT(cx, item) ((void)0)
#define PC_ADD(cx, item, n) ((void)0)
#define PC_TIME(cx, item, start) ((void)0)
#define PC_TIME_PAIR(cx, a, b, start) ((void)0)
#define PC_FLUSH(cx) ((void)0)
#define PC_COLLECT() ((void)0)
#define PC_GUARD_RETURN(ex, item, result) return (result)
#endif

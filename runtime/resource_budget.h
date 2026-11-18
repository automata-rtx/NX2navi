// Shared commit headroom and accounting for speculative preparation. The caller serializes updates.
#pragma once
#include "vk_build_gate.h"

// Negative configuration selects the automatic texture-pressure floor; zero explicitly disables that floor.
static inline double resource_commit_headroom_mb(double limit_mb, double configured_mb) {
    if (configured_mb >= 0) return configured_mb;
    return limit_mb > 40960.0 ? limit_mb / 10.0 : 4096.0;
}

// A small recovery band prevents admission and reclamation alternating at the same threshold. It is a target for
// measured free commit, not permission to count deferred frees or allocator slack as available system memory.
static inline double resource_commit_recovery_mb(double floor_mb) {
    double margin = floor_mb / 32.0;
    return floor_mb > 0 ? floor_mb + (margin < 256.0 ? margin : 256.0) : 0;
}

static inline double resource_commit_deficit_mb(int *pressured, double used_mb, double limit_mb, double floor_mb) {
    if (used_mb < 0 || limit_mb <= 0 || floor_mb <= 0) { *pressured = 0; return 0; }
    double free_mb = limit_mb - used_mb, recover = resource_commit_recovery_mb(floor_mb);
    *pressured = *pressured ? free_mb < recover : free_mb < floor_mb;
    return *pressured ? recover - free_mb : 0;
}

// Background preparation must stop before textures give way. An explicitly disabled build gate stays disabled.
static inline double resource_build_headroom_mb(double limit_mb, double build_mb, double texture_mb) {
    if (build_mb <= 0) return 0;
    double texture = resource_commit_recovery_mb(resource_commit_headroom_mb(limit_mb, texture_mb));
    return build_mb > texture ? build_mb : texture;
}

typedef struct {
    BuildGate active;
    double completed_mb; // Estimates completed since the latest usage sample, not yet reflected in that sample.
} ResourceBudget;

// A fresh system reading now includes completed builds' retained allocations. In-flight estimates still protect
// their possible future growth; keeping those is intentionally conservative, even if some growth is already seen.
static inline void resource_budget_sample(ResourceBudget *b) {
    b->completed_mb = 0;
}

static inline int resource_budget_enter(ResourceBudget *b, double used_mb, double limit_mb, double estimate_mb) {
    return build_gate_enter(&b->active, used_mb + b->completed_mb, limit_mb, estimate_mb);
}

// Finishing does not return the reservation to a stale sample: the compiled result may retain the memory. The
// next fresh sample retires this debt, so a rapid sequence of finished builds cannot repeatedly spend the same room.
static inline void resource_budget_leave(ResourceBudget *b, double estimate_mb) {
    if (estimate_mb < 0) return;
    double settled = estimate_mb < b->active.reserved_mb ? estimate_mb : b->active.reserved_mb;
    build_gate_leave(&b->active, settled);
    b->completed_mb += settled;
}

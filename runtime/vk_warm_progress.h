// Successful library requests survive interruption. The complete file is replaced under
// one cache-directory writer lock. A digest covers the identity, cursor and completion bits.
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>

typedef struct {
    uint8_t digest[32], key[32];
    uint64_t cursor, repair_bytes;
    uint32_t pending, version, full_pass, loss_checked, sample_round, missed_last_time;
    uint8_t bits[];
} WarmProgress;

static inline size_t warm_progress_size(size_t n) {
    return sizeof(WarmProgress) + 2 * (n / 8 + (n % 8 != 0));
}

static inline int warm_progress_valid(const WarmProgress *p, size_t bytes, size_t n, const uint8_t key[32]) {
    if (!p || bytes != warm_progress_size(n) || p->version != 3 || p->cursor >= n ||
        p->pending > 1 || p->missed_last_time > 1 || p->full_pass > 1 || p->loss_checked > 1 || memcmp(p->key, key, 32)) return 0;
    uint8_t digest[32];
    rt_sha256(p->key, bytes - offsetof(WarmProgress, key), digest);
    return !memcmp(p->digest, digest, sizeof digest);
}

static inline int warm_completed(const WarmProgress *p, size_t index) {
    return (p->bits[index / 8] >> (index % 8)) & 1;
}
static inline void warm_complete(WarmProgress *p, size_t index) {
    p->bits[index / 8] |= (uint8_t)(1u << (index % 8));
}

// Older prepared lists can label a guest vertex shader whose translation is a
// mesh or task stage as vertex. Keep these distinct from verified library hits.
static inline int warm_other_stage(const WarmProgress *p, size_t n, size_t i) {
    return (p->bits[(n + 7) / 8 + i / 8] >> (i % 8)) & 1;
}
static inline void warm_note_other_stage(WarmProgress *p, size_t n, size_t i) {
    p->bits[(n + 7) / 8 + i / 8] |= (uint8_t)(1u << (i % 8));
}

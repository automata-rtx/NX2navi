// Physical image ownership for ordinary CPU-backed textures. The texture lock serializes every operation.
// ImgKey, VkImage and VmaAllocation are declared by the including translation unit. Logical texture records keep
// their own address, freshness and view interpretation; this owner holds one allocation independently of them.
#pragma once
#include <assert.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    uint64_t resident_bytes, live_bytes, retained_bytes;
    uint64_t owners, next_id;
} TextureMaterialLedger;

typedef struct TextureMaterial {
    VkImage image;
    VmaAllocation alloc;
    ImgKey key;
    uint64_t bytes, id;
    uint64_t last_use_fence;
    uint32_t last_use_epoch;
    uint32_t refs, live_refs; // Logical associations; a cache pin is separate from these counts.
    uint32_t index_slot;
    uint8_t cached, ready, reclaim, retired;
    TextureMaterialLedger *ledger;
    struct TextureMaterial *prev, *next;
} TextureMaterial;

static inline void texture_material_reclassify(TextureMaterial *m, int was_live) {
    int live = m->live_refs != 0;
    if (live == was_live) return;
    if (live) {
        assert(m->ledger->retained_bytes >= m->bytes);
        m->ledger->retained_bytes -= m->bytes;
        m->ledger->live_bytes += m->bytes;
    } else {
        assert(m->ledger->live_bytes >= m->bytes);
        m->ledger->live_bytes -= m->bytes;
        m->ledger->retained_bytes += m->bytes;
    }
}

// Adopt one newly allocated image and its first logical association. Residency is charged once, irrespective of
// later association count. It ends when retirement is queued; the allocator separately accounts pending frees.
static inline void texture_material_init(TextureMaterial *m, TextureMaterialLedger *ledger, VkImage image,
                                         VmaAllocation alloc, const ImgKey *key, uint64_t bytes, int live) {
    assert(m && ledger && image && alloc && key);
    memset(m, 0, sizeof *m);
    m->image = image;
    m->alloc = alloc;
    m->key = *key;
    m->bytes = bytes;
    m->ledger = ledger;
    if (!++ledger->next_id) ++ledger->next_id;
    m->id = ledger->next_id;
    m->refs = 1;
    m->live_refs = live != 0;
    ledger->owners++;
    ledger->resident_bytes += bytes;
    if (live) ledger->live_bytes += bytes;
    else ledger->retained_bytes += bytes;
}

static inline void texture_material_acquire(TextureMaterial *m, int live) {
    assert(m && !m->retired && m->refs != UINT32_MAX);
    int was_live = m->live_refs != 0;
    m->refs++;
    m->live_refs += live != 0;
    texture_material_reclassify(m, was_live);
}

// Called exactly once when a logical association changes between live and released. Released associations still
// own a reference; their image cannot disappear until they detach, even if the content index evicts its cache pin.
static inline void texture_material_set_live(TextureMaterial *m, int live) {
    assert(m && !m->retired && m->refs);
    int was_live = m->live_refs != 0;
    if (live) {
        assert(m->live_refs < m->refs);
        m->live_refs++;
    } else {
        assert(m->live_refs);
        m->live_refs--;
    }
    texture_material_reclassify(m, was_live);
}

static inline int texture_material_unreferenced(const TextureMaterial *m) {
    return !m->refs && !m->cached;
}

// A true result permits final retirement. It is not memory reclaimed: only texture_material_retire transfers the
// unique allocation out of resident accounting, and the existing upload/timeline deferral releases it later.
static inline int texture_material_release(TextureMaterial *m, int live) {
    assert(m && !m->retired && m->refs && m->live_refs <= m->refs);
    int was_live = m->live_refs != 0;
    if (live) {
        assert(m->live_refs);
        m->live_refs--;
    } else {
        assert(m->refs > m->live_refs);
    }
    m->refs--;
    texture_material_reclassify(m, was_live);
    return texture_material_unreferenced(m);
}

// One bounded content-index entry may pin a ready image after its last logical association disappears. The pin
// owns no second allocation and must be removable by the same pressure policy that owns retained texture bytes.
static inline int texture_material_cache(TextureMaterial *m, int keep) {
    assert(m && !m->retired && (!keep || m->ready));
    m->cached = keep != 0;
    return texture_material_unreferenced(m);
}

static inline int texture_material_exclusive(const TextureMaterial *m) {
    return m && !m->retired && m->refs == 1 && !m->cached;
}

static inline void texture_material_note_use(TextureMaterial *m, uint64_t fence, uint32_t epoch) {
    assert(m && !m->retired);
    if (fence > m->last_use_fence) m->last_use_fence = fence;
    m->last_use_epoch = epoch;
}

// The caller purges cached views and defers the image exactly once after this succeeds, preserving both upload
// completion and recording/submission lifetime. Never purge global image views when merely detaching one Tex.
static inline uint64_t texture_material_retire(TextureMaterial *m) {
    assert(m && !m->retired && texture_material_unreferenced(m) && !m->live_refs);
    TextureMaterialLedger *ledger = m->ledger;
    assert(ledger->owners && ledger->resident_bytes >= m->bytes && ledger->retained_bytes >= m->bytes);
    ledger->owners--;
    ledger->resident_bytes -= m->bytes;
    ledger->retained_bytes -= m->bytes;
    m->retired = 1;
    m->ready = 0;
    return m->bytes;
}

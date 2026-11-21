// Prepared representations and physical residency for ordinary CPU-backed textures. The texture lock serializes
// every operation. Logical references and prepared identity can survive eviction of the current physical image.
// ImgKey, VkImage and VmaAllocation are declared by the including translation unit.
#pragma once
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "vk_texture_lifetime.h"
#include "resource_core_observation.h"

struct TextureMaterial;
struct TextureMaterialInstance;
struct Tex;
typedef struct {
    uint64_t resident_bytes, live_bytes, retained_bytes, provider_bytes;
    uint64_t owners, next_id, next_residency_id, retained_owners;
    struct TextureMaterial *retained_head, *retained_scan_next;
    struct TextureMaterial *resident_head, *resident_scan_next;
    uint64_t resident_owners;
    RcLifetimeIds lifetime_ids;
    struct TextureMaterialInstance *pending_head, *pending_scan_next;
    uint64_t pending_instances, pending_bytes, deferred_bytes, backend_accounted_bytes;
} TextureMaterialLedger;

// Capture this value, not a pointer into a material's current slot. Removing or replacing residency cannot mutate
// an already captured image/instance identity. The caller preserves its allocation through GPU completion.
typedef struct {
    VkImage image;
    VmaAllocation alloc;
    uint64_t bytes, id;
    uint64_t last_use_fence;
    uint32_t last_use_epoch;
    struct TextureMaterialInstance *instance;
} TextureMaterialResidency;

typedef struct TextureMaterialInstance {
    RcLifetimeResidency lifetime;
    TextureMaterialResidency value;
    struct TextureMaterial *material;
    uint64_t upload_sequence;
    struct TextureMaterialInstance *prev, *next;
    struct TextureMaterialInstance *use_map_next;
    struct TextureUseNode *use_head; // stable adapter-owned captures available to fork exact descriptor uses
    uint32_t destroyed;
    uint8_t queued, deferred, use_mapped, backend_accounted;
} TextureMaterialInstance;

typedef struct TextureMaterial {
    TextureMaterialResidency resident;
    ImgKey key;
    RcPreparedInput input;              // exact prepared contents owned by this material and its captured instances
    uint64_t id;
    uint32_t refs, live_refs; // Logical associations; a cache pin is separate from these counts.
    uint32_t index_slot;
    uint8_t cached, ready, reclaim, retired, retained_queued; // ready identifies prepared contents, not residency.
    TextureMaterialLedger *ledger;
    struct TextureMaterial *prev, *next;
    struct TextureMaterial *resident_prev, *resident_next;
    struct Tex *associations;
    void *provider; // Adapter-owned immutable bytes and decode recipe, with their own admission charge.
    RcLifetimeVersion lifetime;
    uint32_t instances;
} TextureMaterial;

static inline void texture_material_require(RcLifetimeResult result) {
    if (result != RC_LIFETIME_OK) abort();
}
static inline void texture_material_resident_insert(TextureMaterial *m) {
    TextureMaterialLedger *l = m->ledger;
    assert(m->resident.image && !m->resident_prev && !m->resident_next);
    m->resident_next = l->resident_head;
    if (m->resident_next) m->resident_next->resident_prev = m;
    l->resident_head = m;
    if (!l->resident_scan_next) l->resident_scan_next = m;
    l->resident_owners++;
}
static inline void texture_material_resident_remove(TextureMaterial *m) {
    TextureMaterialLedger *l = m->ledger;
    assert(l->resident_owners);
    if (l->resident_scan_next == m) l->resident_scan_next = m->resident_next;
    if (m->resident_prev) m->resident_prev->resident_next = m->resident_next;
    else { assert(l->resident_head == m); l->resident_head = m->resident_next; }
    if (m->resident_next) m->resident_next->resident_prev = m->resident_prev;
    m->resident_prev = m->resident_next = NULL;
    l->resident_owners--;
}
static inline TextureMaterial *texture_material_resident_next(TextureMaterialLedger *l) {
    TextureMaterial *m = l->resident_scan_next ? l->resident_scan_next : l->resident_head;
    if (m) l->resident_scan_next = m->resident_next;
    return m;
}

static inline void texture_material_pending_insert(TextureMaterialInstance *instance) {
    TextureMaterialLedger *ledger = instance->material->ledger;
    assert(!instance->queued);
    instance->next = ledger->pending_head;
    if (instance->next) instance->next->prev = instance;
    ledger->pending_head = instance;
    if (!ledger->pending_scan_next) ledger->pending_scan_next = instance;
    instance->queued = 1;
    ledger->pending_instances++;
}

static inline void texture_material_retained_insert(TextureMaterial *m) {
    TextureMaterialLedger *ledger = m->ledger;
    assert(!m->retained_queued && !m->live_refs && !m->retired && m->resident.image);
    m->next = ledger->retained_head;
    m->prev = NULL;
    if (m->next) m->next->prev = m;
    ledger->retained_head = m;
    if (!ledger->retained_scan_next) ledger->retained_scan_next = m;
    m->retained_queued = 1;
    ledger->retained_owners++;
}

static inline void texture_material_retained_remove(TextureMaterial *m) {
    TextureMaterialLedger *ledger = m->ledger;
    assert(m->retained_queued && ledger->retained_owners);
    if (ledger->retained_scan_next == m) ledger->retained_scan_next = m->next;
    if (m->prev) m->prev->next = m->next;
    else ledger->retained_head = m->next;
    if (m->next) m->next->prev = m->prev;
    m->prev = m->next = NULL;
    m->retained_queued = 0;
    ledger->retained_owners--;
}

static inline void texture_material_reclassify(TextureMaterial *m, int was_live) {
    int live = m->live_refs != 0;
    if (live == was_live || !m->resident.image) return;
    if (live) {
        assert(m->ledger->retained_bytes >= m->resident.bytes);
        m->ledger->retained_bytes -= m->resident.bytes;
        m->ledger->live_bytes += m->resident.bytes;
        texture_material_retained_remove(m);
    } else {
        assert(m->ledger->live_bytes >= m->resident.bytes);
        m->ledger->live_bytes -= m->resident.bytes;
        m->ledger->retained_bytes += m->resident.bytes;
        texture_material_retained_insert(m);
    }
}

// Install a physical instance of this same prepared representation. Changed contents require another representation;
// the including code establishes that identity before calling. An absent image has no byte charge or victim entry.
static inline TextureMaterialResidency texture_material_install_residency(TextureMaterial *m, VkImage image,
                                                                          VmaAllocation alloc, uint64_t bytes) {
    assert(m && !m->retired && m->ledger && image && alloc && !m->resident.image && !m->retained_queued);
    TextureMaterialLedger *ledger = m->ledger;
    TextureMaterialInstance *instance = calloc(1, sizeof *instance);
    if (!instance) return (TextureMaterialResidency){0};
    if (m->instances == UINT32_MAX) { free(instance); return (TextureMaterialResidency){0}; }
    if (rc_lifetime_residency_install(&ledger->lifetime_ids, &m->lifetime, m->lifetime.id,
                                     &instance->lifetime) != RC_LIFETIME_OK) {
        free(instance); return (TextureMaterialResidency){0};
    }
    ledger->next_residency_id = instance->lifetime.binding.residency;
    m->resident = (TextureMaterialResidency){.image = image, .alloc = alloc, .bytes = bytes,
                                           .id = ledger->next_residency_id, .instance = instance};
    instance->value = m->resident;
    instance->material = m;
    m->instances++;
    texture_material_resident_insert(m);
    ledger->resident_bytes += bytes;
    if (m->live_refs) ledger->live_bytes += bytes;
    else {
        ledger->retained_bytes += bytes;
        texture_material_retained_insert(m);
    }
    return m->resident;
}

static inline TextureMaterialResidency texture_material_capture(const TextureMaterial *m) {
    assert(m && !m->retired);
    return m->resident;
}

// Transfer physical ownership exactly once to the caller's deferred-destruction path. The prepared identity, source
// associations and cache pin survive. Accounting ends at scheduling; allocator pending-free accounting remains separate.
static inline TextureMaterialResidency texture_material_take_residency(TextureMaterial *m) {
    assert(m && !m->retired);
    TextureMaterialResidency old = m->resident;
    if (!old.image) return old;
    RcLifetimeResult detached = rc_lifetime_residency_detach(&m->lifetime, &old.instance->lifetime,
                                                           old.instance->lifetime.binding);
    if (detached == RC_LIFETIME_NO_RECOVERY) return (TextureMaterialResidency){0};
    assert(detached == RC_LIFETIME_OK);
    TextureMaterialLedger *ledger = m->ledger;
    assert(ledger->resident_bytes >= old.bytes);
    if (m->live_refs) {
        assert(!m->retained_queued && ledger->live_bytes >= old.bytes);
        ledger->live_bytes -= old.bytes;
    } else {
        assert(ledger->retained_bytes >= old.bytes);
        ledger->retained_bytes -= old.bytes;
        texture_material_retained_remove(m);
    }
    ledger->resident_bytes -= old.bytes;
    texture_material_resident_remove(m);
    memset(&m->resident, 0, sizeof m->resident);
    texture_material_pending_insert(old.instance);
    ledger->pending_bytes += old.bytes;
    return old;
}

// Adopt one newly allocated image and its first logical association. Residency is charged once, irrespective of
// later association count. It ends when retirement is queued; the allocator separately accounts pending frees.
static inline int texture_material_init(TextureMaterial *m, TextureMaterialLedger *ledger, VkImage image,
                                        VmaAllocation alloc, const ImgKey *key, uint64_t bytes, int live) {
    assert(m && ledger && image && alloc && key);
    memset(m, 0, sizeof *m);
    m->key = *key;
    m->ledger = ledger;
    if (rc_lifetime_version_init(&ledger->lifetime_ids, &m->lifetime,
                                RC_LIFETIME_RECOVERY_CURRENT_SOURCE, 0) != RC_LIFETIME_OK) return 0;
    m->id = ledger->next_id = m->lifetime.id;
    m->refs = 1;
    m->live_refs = live != 0;
    if (!texture_material_install_residency(m, image, alloc, bytes).image) {
        memset(m, 0, sizeof *m);
        return 0;
    }
    ledger->owners++;
    if (!live && rc_lifetime_exact_hold(&m->lifetime, m->lifetime.id) != RC_LIFETIME_OK) abort();
    return 1;
}

static inline void texture_material_acquire(TextureMaterial *m, int live) {
    assert(m && !m->retired && m->refs != UINT32_MAX);
    int was_live = m->live_refs != 0;
    if (!live && rc_lifetime_exact_hold(&m->lifetime, m->lifetime.id) != RC_LIFETIME_OK) abort();
    m->refs++;
    m->live_refs += live != 0;
    texture_material_reclassify(m, was_live);
}

// Called exactly once when a logical association changes between live and released. This tracks current demand;
// references preserve representation metadata even when its resident image has been evicted independently.
static inline void texture_material_set_live(TextureMaterial *m, int live) {
    assert(m && !m->retired && m->refs);
    int was_live = m->live_refs != 0;
    if (live) {
        assert(m->live_refs < m->refs);
        texture_material_require(rc_lifetime_exact_release(&m->lifetime, m->lifetime.id));
        m->live_refs++;
    } else {
        assert(m->live_refs);
        texture_material_require(rc_lifetime_exact_hold(&m->lifetime, m->lifetime.id));
        m->live_refs--;
    }
    texture_material_reclassify(m, was_live);
}

static inline int texture_material_unreferenced(const TextureMaterial *m) {
    return !m->refs && !m->cached;
}

// A true result permits final metadata retirement. It is not physical memory reclaimed: residency removal returns
// the unique allocation for the existing upload/timeline deferral to release later.
static inline int texture_material_release(TextureMaterial *m, int live) {
    assert(m && !m->retired && m->refs && m->live_refs <= m->refs);
    int was_live = m->live_refs != 0;
    if (live) {
        assert(m->live_refs);
        m->live_refs--;
    } else {
        assert(m->refs > m->live_refs);
        texture_material_require(rc_lifetime_exact_release(&m->lifetime, m->lifetime.id));
    }
    m->refs--;
    texture_material_reclassify(m, was_live);
    return texture_material_unreferenced(m);
}

// One bounded content-index entry may retain prepared identity after its last logical association disappears.
// The pin does not imply residency and owns no second physical allocation.
static inline int texture_material_cache(TextureMaterial *m, int keep) {
    assert(m && !m->retired && (!keep || m->ready));
    m->cached = keep != 0;
    return texture_material_unreferenced(m);
}

static inline int texture_material_exclusive(const TextureMaterial *m) {
    return m && !m->retired && m->refs == 1 && !m->cached;
}

static inline void texture_material_note_use(TextureMaterial *m, uint64_t fence, uint32_t epoch) {
    assert(m && !m->retired && m->resident.image);
    if (fence > m->resident.last_use_fence) m->resident.last_use_fence = fence;
    m->resident.last_use_epoch = epoch;
    m->resident.instance->value.last_use_fence = m->resident.last_use_fence;
    m->resident.instance->value.last_use_epoch = epoch;
}

// Each step visits one retained physical owner, without inspecting active owners or individual associations.
// The caller caps steps at the retained-owner count and its work budget. Removal repairs the cursor before a
// candidate becomes live or can be freed; an owner enters this queue only at the last live association's release.
static inline TextureMaterial *texture_material_scan_next(TextureMaterialLedger *ledger) {
    TextureMaterial *m = ledger->retained_scan_next ? ledger->retained_scan_next : ledger->retained_head;
    if (m) ledger->retained_scan_next = m->next;
    return m;
}

// The caller purges cached views and defers the image exactly once after this succeeds, preserving both upload
// completion and recording/submission lifetime. Never purge global image views when merely detaching one Tex.
static inline TextureMaterialResidency texture_material_retire(TextureMaterial *m) {
    assert(m && !m->retired && texture_material_unreferenced(m) && !m->live_refs);
    TextureMaterialLedger *ledger = m->ledger;
    assert(ledger->owners);
    TextureMaterialResidency old = texture_material_take_residency(m);
    ledger->owners--;
    m->retired = 1;
    m->ready = 0;
    if (!m->instances) texture_material_require(rc_lifetime_version_retire(&m->lifetime, m->lifetime.id));
    return old;
}

// Attach the actual upload-batch dependency while publication is still protected by the texture lock.
static inline void texture_material_set_upload(TextureMaterial *m, uint64_t sequence) {
    assert(m && !m->retired && m->resident.instance);
    if (!sequence) return;
    TextureMaterialInstance *instance = m->resident.instance;
    assert(!instance->upload_sequence);
    texture_material_require(rc_lifetime_upload_prepare(&m->lifetime, &instance->lifetime,
                                                        instance->lifetime.binding));
    instance->upload_sequence = sequence;
}

static inline RcLifetimeResult texture_material_upload_progress(TextureMaterialInstance *instance,
                                                                uint64_t submitted, uint64_t completed) {
    uint64_t required = instance->upload_sequence;
    if (!required) return RC_LIFETIME_OK;
    RcLifetimeCompletion point = {0};
    rc_lifetime_completion_add(&point, TEXTURE_MATERIAL_UPLOAD_DOMAIN, required);
    if (instance->lifetime.upload == RC_LIFETIME_UPLOAD_PENDING && (submitted >= required || completed >= required)) {
        RcLifetimeResult result = rc_lifetime_upload_submit(&instance->material->lifetime, &instance->lifetime,
                                                            instance->lifetime.binding, &point);
        if (result != RC_LIFETIME_OK) return result;
    }
    if (instance->lifetime.upload == RC_LIFETIME_UPLOAD_SUBMITTED && completed >= required)
        return rc_lifetime_upload_complete(&instance->material->lifetime, &instance->lifetime,
                                           instance->lifetime.binding, &point);
    return RC_LIFETIME_OK;
}

static inline RcLifetimeResult texture_material_use_begin(TextureMaterial *m, TextureMaterialUse *use) {
    if (!m || m->retired || !m->ready || !m->resident.instance || !use || use->instance) return RC_LIFETIME_INVALID;
    TextureMaterialInstance *instance = m->resident.instance;
    RcLifetimeResult result = rc_lifetime_capture_begin(&m->lifetime, &instance->lifetime,
                                                       instance->lifetime.binding, &use->capture);
    if (result == RC_LIFETIME_OK) use->instance = instance;
    return result;
}

static inline RcLifetimeResult texture_material_use_fork(const TextureMaterialUse *source, TextureMaterialUse *use) {
    if (!source || !source->instance || !use || use->instance) return RC_LIFETIME_INVALID;
    TextureMaterialInstance *instance = source->instance;
    RcLifetimeResult result = rc_lifetime_capture_fork(&instance->material->lifetime, &instance->lifetime,
                                                      &source->capture, &use->capture);
    if (result == RC_LIFETIME_OK) use->instance = instance;
    return result;
}

// submitted_upload is an actual submission frontier. done names this consumer's real completion domains.
static inline RcLifetimeResult texture_material_use_submit(TextureMaterialUse *use, uint64_t submitted_upload,
                                                           const RcLifetimeCompletion *done) {
    if (!use || !use->instance) return RC_LIFETIME_INVALID;
    TextureMaterialInstance *instance = use->instance;
    RcLifetimeResult result = texture_material_upload_progress(instance, submitted_upload, 0);
    if (result != RC_LIFETIME_OK) return result;
    RcLifetimeCompletion waits = {0};
    if (instance->upload_sequence)
        rc_lifetime_completion_add(&waits, TEXTURE_MATERIAL_UPLOAD_DOMAIN, instance->upload_sequence);
    return rc_lifetime_capture_submit(&instance->material->lifetime, &instance->lifetime,
                                      &use->capture, &waits, done);
}

static inline RcLifetimeResult texture_material_use_cancel(TextureMaterialUse *use) {
    if (!use || !use->instance) return RC_LIFETIME_INVALID;
    TextureMaterialInstance *instance = use->instance;
    RcLifetimeResult result = rc_lifetime_capture_cancel(&instance->material->lifetime, &instance->lifetime,
                                                        &use->capture);
    if (result == RC_LIFETIME_OK) use->instance = NULL;
    return result;
}

static inline RcLifetimeResult texture_material_use_complete(TextureMaterialUse *use, uint64_t upload_completed,
                                                             const RcLifetimeCompletion *completed) {
    if (!use || !use->instance) return RC_LIFETIME_INVALID;
    TextureMaterialInstance *instance = use->instance;
    RcLifetimeResult result = texture_material_upload_progress(instance, 0, upload_completed);
    if (result != RC_LIFETIME_OK) return result;
    result = rc_lifetime_capture_complete(&instance->material->lifetime, &instance->lifetime,
                                          &use->capture, completed);
    if (result == RC_LIFETIME_OK) use->instance = NULL;
    return result;
}

static inline TextureMaterialInstance *texture_material_pending_next(TextureMaterialLedger *ledger) {
    TextureMaterialInstance *instance = ledger->pending_scan_next ? ledger->pending_scan_next : ledger->pending_head;
    if (instance) ledger->pending_scan_next = instance->next;
    return instance;
}
static inline int texture_material_instance_defer_ready(const TextureMaterialInstance *instance) {
    return instance && instance->queued && !instance->deferred && !instance->lifetime.captures;
}
static inline TextureMaterialResidency texture_material_instance_defer_begin(TextureMaterialInstance *instance) {
    assert(texture_material_instance_defer_ready(instance));
    instance->deferred = 1;
    instance->material->ledger->deferred_bytes += instance->value.bytes;
    instance->backend_accounted = instance->material->reclaim;
    if (instance->backend_accounted) instance->material->ledger->backend_accounted_bytes += instance->value.bytes;
    return instance->value;
}
// Backend destruction callbacks can run under the upload lock at shutdown. This is deliberately a leaf store.
static inline void texture_material_destroyed(void *arg) {
    TextureMaterialInstance *instance = arg;
    __atomic_store_n(&instance->destroyed, 1, __ATOMIC_RELEASE);
}
static inline int texture_material_can_free(const TextureMaterial *m) {
    return m && m->retired && !m->instances;
}
// Run under the texture lock. Return the final metadata owner to free, if this was its last physical instance.
static inline TextureMaterial *texture_material_instance_reap(TextureMaterialInstance *instance) {
    if (!instance || !instance->deferred || !__atomic_load_n(&instance->destroyed, __ATOMIC_ACQUIRE)) return NULL;
    TextureMaterial *m = instance->material;
    TextureMaterialLedger *ledger = m->ledger;
    assert(!instance->lifetime.captures && instance->queued && m->instances);
    // Actual backend destruction proves that the queued upload and graphics obligations both completed.
    texture_material_require(texture_material_upload_progress(instance, instance->upload_sequence, instance->upload_sequence));
    texture_material_require(rc_lifetime_residency_retire(&m->lifetime, &instance->lifetime,
                                                         instance->lifetime.binding));
    if (ledger->pending_scan_next == instance) ledger->pending_scan_next = instance->next;
    if (instance->prev) instance->prev->next = instance->next;
    else ledger->pending_head = instance->next;
    if (instance->next) instance->next->prev = instance->prev;
    assert(ledger->pending_bytes >= instance->value.bytes);
    assert(ledger->deferred_bytes >= instance->value.bytes);
    ledger->deferred_bytes -= instance->value.bytes;
    if (instance->backend_accounted) {
        assert(ledger->backend_accounted_bytes >= instance->value.bytes);
        ledger->backend_accounted_bytes -= instance->value.bytes;
    }
    ledger->pending_instances--; ledger->pending_bytes -= instance->value.bytes; m->instances--;
    free(instance);
    if (!texture_material_can_free(m)) return NULL;
    texture_material_require(rc_lifetime_version_retire(&m->lifetime, m->lifetime.id));
    return m;
}

// Included by vk_texture.c after its material/content helpers. The texture lock owns these tables and all core
// transitions. Queue hand-over only publishes a leaf atomic state; it never acquires the texture or upload lock.
#pragma once

enum { TEXTURE_USE_BUCKETS = 256, TEXTURE_IMAGE_BUCKETS = 4096,
       TEXTURE_USE_LIMIT = 262144, TEXTURE_BATCH_LIMIT = 8192, TEXTURE_PUBLICATION_LIMIT = 4096 };
enum { TEXTURE_BATCH_RECORDING, TEXTURE_BATCH_SEALED, TEXTURE_BATCH_HANDED, TEXTURE_BATCH_ABANDONED };

typedef struct TextureUseNode {
    TextureMaterialUse use;
    struct TextureUseNode *next, *prev, *hash_next;
    struct TextureUseNode *instance_prev, *instance_next;
    struct TextureUseNode *publication_pending_next;
    uint32_t publication_refs;
    uint8_t publication_pending;
} TextureUseNode;

typedef struct TexturePinnedUse {
    struct TexturePinnedUse *next;
    const tex_t *pin;
} TexturePinnedUse;

typedef struct TextureNativeUse {
    struct TextureNativeUse *next;
    VkMemoryImageHold *hold;
    VkImage image;
    uint32_t descriptor;
    VkTextureAliasRead *alias;
} TextureNativeUse;

typedef struct VkRecordingPartDisposition {
    uint32_t refs, handover;
} TextureUsePartDisposition;

struct TextureUseBatch {
    uint64_t id, fence, upload_submitted;
    TextureUsePartDisposition *part;
    uint32_t handover;
    TextureUseNode *head, *buckets[TEXTURE_USE_BUCKETS];
    TexturePinnedUse *pinned;
    TextureNativeUse *native;
    struct TextureUseBatch *next, *active_next;
};
struct TextureImagePublication {
    TextureUseNode *head, *buckets[TEXTURE_USE_BUCKETS];
    uint64_t last_batch;
    struct TexturePublicationEntry *entries;
    TextureUseNode *pending;
    TextureNativeUse *native, *pending_native;
    uint32_t entry_limit, entry_count, entry_capacity;
};
typedef struct TexturePublicationEntry { uint32_t index; TextureUseNode *use; } TexturePublicationEntry;
static struct {
    TextureMaterialInstance *images[TEXTURE_IMAGE_BUCKETS];
    TextureUseBatch *active[TEXTURE_USE_BUCKETS], *sealed_head, *sealed_tail;
    uint64_t next_batch, completed, upload_completed;
    uint32_t nodes, batches, publications;
} texture_uses;

static unsigned texture_use_hash(uint64_t value, unsigned mask) {
    value ^= value >> 33; value *= UINT64_C(0xff51afd7ed558ccd); value ^= value >> 33;
    return (unsigned)value & mask;
}
static unsigned texture_use_image_hash(VkImage image) {
    return texture_use_hash((uint64_t)(uintptr_t)image, TEXTURE_IMAGE_BUCKETS - 1);
}
static TextureMaterialInstance *content_use_image_find(VkImage image) {
    for (TextureMaterialInstance *p = texture_uses.images[texture_use_image_hash(image)]; p; p = p->use_map_next)
        if (p->value.image == image) return p;
    return NULL;
}
static void content_use_image_add(TextureMaterialInstance *instance) {
    if (!instance || instance->use_mapped) return;
    if (content_use_image_find(instance->value.image)) rt_fatal("vk: material image identity already published");
    unsigned at = texture_use_image_hash(instance->value.image);
    instance->use_map_next = texture_uses.images[at];
    texture_uses.images[at] = instance;
    instance->use_mapped = 1;
}
static void content_use_image_remove(TextureMaterialInstance *instance) {
    if (!instance || !instance->use_mapped) return;
    TextureMaterialInstance **link = &texture_uses.images[texture_use_image_hash(instance->value.image)];
    while (*link && *link != instance) link = &(*link)->use_map_next;
    if (!*link || instance->lifetime.captures || instance->use_head)
        rt_fatal("vk: material image retired while published or recorded");
    *link = instance->use_map_next;
    instance->use_map_next = NULL; instance->use_mapped = 0;
}
static TextureUseNode *texture_use_find(TextureUseNode *const *buckets, const TextureMaterialInstance *instance) {
    unsigned at = texture_use_hash(instance->value.id, TEXTURE_USE_BUCKETS - 1);
    for (TextureUseNode *p = buckets[at]; p; p = p->hash_next)
        if (p->use.instance == instance) return p;
    return NULL;
}

static RcLifetimeResult texture_use_retain_instance(TextureUseNode *node, TextureMaterialInstance *instance,
                                                     const TextureMaterialUse *previous) {
    if (texture_uses.nodes == TEXTURE_USE_LIMIT) rt_fatal("vk: recorded material-use capacity exhausted");
    // A wanted image copied between descriptor sets may already be detached from its material. Its lookup entry
    // remains valid while an existing recording/publication owns it, and that owning ticket is the exact provider.
    if (!previous && instance->use_head) previous = &instance->use_head->use;
    if (previous ? previous->instance != instance : instance->material->resident.instance != instance)
        return RC_LIFETIME_INVALID;
    RcLifetimeResult result = previous ? texture_material_use_fork(previous, &node->use) :
                                        texture_material_use_begin(instance->material, &node->use);
    if (result != RC_LIFETIME_OK) return result;
    node->instance_next = instance->use_head;
    if (node->instance_next) node->instance_next->instance_prev = node;
    instance->use_head = node;
    texture_uses.nodes++;
    return RC_LIFETIME_OK;
}

static TextureUseNode *texture_use_insert(TextureUseNode **head, TextureUseNode **buckets, TextureMaterialInstance *instance,
                                         const TextureMaterialUse *previous) {
    TextureUseNode *existing = texture_use_find(buckets, instance);
    if (existing) return existing;
    TextureUseNode *node = calloc(1, sizeof *node);
    if (!node) rt_fatal("vk: allocating recorded material use");
    RcLifetimeResult result = texture_use_retain_instance(node, instance, previous);
    if (result != RC_LIFETIME_OK) rt_fatal("vk: stale material image publication (%d)", (int)result);
    unsigned at = texture_use_hash(instance->value.id, TEXTURE_USE_BUCKETS - 1);
    node->next = *head;
    if (node->next) node->next->prev = node;
    *head = node;
    node->hash_next = buckets[at]; buckets[at] = node;
    return node;
}
static void texture_use_unlink(TextureUseNode *node) {
    TextureMaterialInstance *instance = node->use.instance;
    if (node->instance_prev) node->instance_prev->instance_next = node->instance_next;
    else instance->use_head = node->instance_next;
    if (node->instance_next) node->instance_next->instance_prev = node->instance_prev;
}
static TextureUseBatch *texture_use_active(uint64_t fence, int create) {
    if (!fence) return NULL; // A guest query outside an actual replay recording is not a scene consumer.
    unsigned at = texture_use_hash(fence, TEXTURE_USE_BUCKETS - 1);
    for (TextureUseBatch *p = texture_uses.active[at]; p; p = p->active_next)
        if (p->fence == fence) return p;
    if (!create) return NULL;
    if (texture_uses.batches == TEXTURE_BATCH_LIMIT || texture_uses.next_batch == UINT64_MAX)
        rt_fatal("vk: recorded material-batch capacity exhausted");
    TextureUseBatch *batch = calloc(1, sizeof *batch);
    if (!batch) rt_fatal("vk: allocating recorded material batch");
    batch->id = ++texture_uses.next_batch; batch->fence = fence;
    batch->active_next = texture_uses.active[at]; texture_uses.active[at] = batch;
    texture_uses.batches++;
    return batch;
}
static void texture_use_scene_material_locked(TextureMaterialInstance *instance, const TextureMaterialUse *owner) {
    TextureUseBatch *batch = texture_use_active(vk_exec_recording_fence(), 1);
    if (batch) texture_use_insert(&batch->head, batch->buckets, instance, owner);
}

static void texture_use_active_remove(TextureUseBatch *batch) {
    TextureUseBatch **link = &texture_uses.active[texture_use_hash(batch->fence, TEXTURE_USE_BUCKETS - 1)];
    while (*link && *link != batch) link = &(*link)->active_next;
    if (!*link) rt_fatal("vk: missing recording material batch");
    *link = batch->active_next; batch->active_next = NULL;
}

static void texture_use_pin(const Tex *e, const tex_t *pin) {
    TextureUseBatch *batch = texture_use_active(vk_exec_recording_fence(), 1);
    if (!batch) rt_fatal("vk: pinned image use outside the native recorder");
    for (const TexturePinnedUse *use = batch->pinned; use; use = use->next)
        if (use->pin == pin) return;
    if (texture_uses.nodes >= TEXTURE_USE_LIMIT) rt_fatal("vk: scene texture use capacity exhausted");
    TexturePinnedUse *use = malloc(sizeof *use);
    if (!use) rt_fatal("vk: scene pinned texture use allocation failed");
    const tex_t *retained = nvn_texture_pin_retain_attachment(pin->serial, e);
    if (retained != pin) rt_fatal("vk: scene pinned texture owner was lost before recording");
    *use = (TexturePinnedUse){.next = batch->pinned, .pin = retained};
    batch->pinned = use;
    texture_uses.nodes++;
}

static void texture_native_uses_free(TextureNativeUse **head) {
    while (*head) {
        TextureNativeUse *use = *head;
        *head = use->next;
        if (use->alias) vk_texture_alias_read_cancel(use->alias);
        else vk_mem_image_release(use->hold);
        free(use);
        texture_uses.nodes--;
    }
}

static void texture_native_use_batch(TextureUseBatch *batch, const TextureNativeUse *choice) {
    for (const TextureNativeUse *use = batch->native; use; use = use->next)
        if (use->image == choice->image) return;
    if (texture_uses.nodes >= TEXTURE_USE_LIMIT) rt_fatal("vk: recorded native image-use capacity exhausted");
    TextureNativeUse *use = malloc(sizeof *use);
    if (!use) rt_fatal("vk: allocating native recording image use");
    *use = (TextureNativeUse){.next = batch->native, .image = choice->image,
        .hold = choice->hold ? vk_mem_image_hold_retain(choice->hold) : NULL,
        .alias = vk_texture_alias_read_retain(choice->alias)};
    batch->native = use;
    texture_uses.nodes++;
}

static TextureUsePartDisposition *texture_use_part_retain(TextureUseBatch *batch) {
    if (!batch->part) {
        batch->part = malloc(sizeof *batch->part);
        if (!batch->part) rt_fatal("vk: allocating native texture part disposition");
        *batch->part = (TextureUsePartDisposition){.refs = 1, .handover = batch->handover};
    }
    if (__atomic_add_fetch(&batch->part->refs, 1, __ATOMIC_RELAXED) == 0)
        rt_fatal("vk: native texture part disposition reference overflow");
    return batch->part;
}

static int texture_use_part_known_unhanded(const void *opaque) {
    const TextureUsePartDisposition *part = opaque;
    const uint32_t state = __atomic_load_n(&part->handover, __ATOMIC_ACQUIRE);
    // Effects settle after the recorder stops or a body retires. An open body never entered the queue.
    return state == TEXTURE_BATCH_RECORDING || state == TEXTURE_BATCH_ABANDONED;
}

static void texture_use_part_release(TextureUsePartDisposition *part) {
    if (part && !__atomic_sub_fetch(&part->refs, 1, __ATOMIC_ACQ_REL)) free(part);
}

static void texture_use_batch_free(TextureUseBatch *batch) {
    texture_native_uses_free(&batch->native);
    texture_use_part_release(batch->part);
    while (batch->pinned) {
        TexturePinnedUse *use = batch->pinned;
        batch->pinned = use->next;
        nvn_texture_pin_release(use->pin);
        free(use);
        texture_uses.nodes--;
    }
    free(batch);
    texture_uses.batches--;
}

static void content_use(Tex *e) {
    if (!e || !e->material || !e->material->ready || !e->material->resident.instance) return;
    TextureUseBatch *batch = texture_use_active(vk_exec_recording_fence(), 1);
    if (batch) texture_use_insert(&batch->head, batch->buckets, e->material->resident.instance, NULL);
}

void vk_texture_use_images(const VkImage *images, size_t count) {
    lock();
    TextureUseBatch *batch = texture_use_active(vk_exec_recording_fence(), count != 0);
    for (size_t i = 0; batch && i < count; ++i) {
        TextureMaterialInstance *instance = content_use_image_find(images[i]);
        if (instance) texture_use_insert(&batch->head, batch->buckets, instance, NULL);
    }
    unlock();
}
TextureImagePublication *vk_texture_publication_create(const VkImage *images, size_t count,
                                                       const TextureImagePublication *previous) {
    RWT_REGION_CALL(RWT_REGION_PUBLICATION_LOCK, lock());
    if (texture_uses.publications == TEXTURE_PUBLICATION_LIMIT)
        rt_fatal("vk: material descriptor-publication capacity exhausted");
    TextureImagePublication *publication = calloc(1, sizeof *publication);
    if (!publication) rt_fatal("vk: allocating material descriptor publication");
    texture_uses.publications++;
    {
        RWT_REGION(work_create, RWT_REGION_PUBLICATION_CREATE);
        for (size_t i = 0; i < count; ++i) {
            TextureMaterialInstance *instance = content_use_image_find(images[i]);
            if (!instance) continue; // Nonordinary images retain their existing backend ownership.
            TextureUseNode *old = previous ? texture_use_find(previous->buckets, instance) : NULL;
            texture_use_insert(&publication->head, publication->buckets, instance, old ? &old->use : NULL);
        }
    }
    unlock();
    return publication;
}
TextureImagePublication *vk_texture_publication_create_indexed(uint32_t entry_limit) {
    if (!entry_limit || entry_limit > UINT32_MAX / 2)
        rt_fatal("vk: invalid indexed publication extent");
    TextureImagePublication *publication = vk_texture_publication_create(NULL, 0, NULL);
    publication->entry_limit = entry_limit;
    return publication;
}
static uint32_t texture_publication_entry_at(const TextureImagePublication *publication, uint32_t index) {
    uint32_t mask = publication->entry_capacity - 1;
    uint32_t at = texture_use_hash(index, mask);
    while (publication->entries[at].use && publication->entries[at].index != index) at = (at + 1) & mask;
    return at;
}
// The texture lock protects the indexed owner until the caller forks its use.
static const TextureUseNode *texture_publication_owner_locked(const TextureImagePublication *set,
                                                               uint32_t descriptor, VkImage expected) {
    if (!set || !expected || descriptor >= set->entry_limit || !set->entry_capacity) return NULL;
    uint32_t slot = texture_publication_entry_at(set, descriptor);
    const TextureUseNode *owner = set->entries[slot].use;
    if (!owner || !owner->publication_refs || !owner->use.instance || owner->use.instance->value.image != expected)
        return NULL;
    return owner;
}

static const TextureMaterialInstance *texture_publication_image_locked(const TextureImagePublication *set,
                                                                         uint32_t descriptor, VkImage expected) {
    const TextureUseNode *owner = texture_publication_owner_locked(set, descriptor, expected);
    return owner ? owner->use.instance : NULL;
}

int vk_texture_publication_contains_native_locked(const TextureImagePublication *set, uint32_t descriptor,
                                                   VkImage expected) {
    for (const TextureNativeUse *use = set ? set->native : NULL; use; use = use->next)
        if (use->descriptor == descriptor && use->image == expected) return 1;
    return 0;
}

int vk_texture_publication_contains_image_locked(const TextureImagePublication *set, uint32_t descriptor,
                                                  VkImage expected) {
    return texture_publication_image_locked(set, descriptor, expected) ||
        vk_texture_publication_contains_native_locked(set, descriptor, expected);
}

VkTextureAliasRead *vk_texture_publication_alias_locked(const TextureImagePublication *set,
                                                        uint32_t descriptor, VkImage expected) {
    for (const TextureNativeUse *use = set ? set->native : NULL; use; use = use->next)
        if (use->descriptor == descriptor && use->image == expected) return use->alias;
    return NULL;
}

// The producer adapter supplies a live GPU-only image and real synchronized descriptor selection.
// Deduplicate across views only on actual recording use, while every indexed entry owns its own replacement edge.
static int texture_publication_native_retain(TextureImagePublication *set, uint32_t descriptor,
    VkImage image, VkTextureAliasRead *alias, VkMemoryImageHold *selected) {
    if (!set || descriptor >= set->entry_limit || !image || texture_uses.nodes >= TEXTURE_USE_LIMIT) return 0;
    for (const TextureNativeUse *p = set->native; p; p = p->next)
        if (p->descriptor == descriptor)
            return p->image == image && p->alias == alias &&
                (alias || (selected ? p->hold == selected : p->hold && !vk_mem_image_hold_retired(p->hold)));
    TextureNativeUse *use = malloc(sizeof *use);
    if (!use) return 0;
    VkMemoryImageHold *hold = NULL;
    if (!alias) {
        hold = selected ? vk_mem_image_hold_retain_exact(selected, image) : vk_mem_image_hold(image);
        if (!hold) { free(use); return 0; }
    }
    *use = (TextureNativeUse){.next = set->native, .image = image, .descriptor = descriptor,
        .hold = hold, .alias = vk_texture_alias_read_retain(alias)};
    set->native = use;
    texture_uses.nodes++;
    TextureUseBatch *batch = texture_use_active(vk_exec_recording_fence(), 0);
    if (batch && set->last_batch == batch->id) texture_native_use_batch(batch, use);
    return 1;
}

static int texture_publication_native_retain_locked(TextureImagePublication *set, uint32_t descriptor, VkImage image) {
    return texture_publication_native_retain(set, descriptor, image, NULL, NULL);
}

int vk_texture_publication_retain_alias_locked(TextureImagePublication *set, uint32_t descriptor,
                                               VkImage image, VkTextureAliasRead *alias) {
    return alias && texture_publication_native_retain(set, descriptor, image, alias, NULL);
}

int vk_texture_publication_copy_native_locked(TextureImagePublication *destination, const TextureImagePublication *source,
    uint32_t descriptor, VkImage expected) {
    if (!expected || destination == source) return 0;
    for (const TextureNativeUse *use = source ? source->native : NULL; use; use = use->next) {
        if (use->descriptor != descriptor || use->image != expected) continue;
        if (!use->hold && !use->alias) return -1;
        return texture_publication_native_retain(destination, descriptor, expected, use->alias, use->hold) ? 1 : -1;
    }
    return 0;
}

static void texture_publication_native_replaced(TextureImagePublication *set, uint32_t index, VkImage image) {
    TextureNativeUse **link = &set->native;
    while (*link && (*link)->descriptor != index) link = &(*link)->next;
    TextureNativeUse *use = *link;
    if (!use || use->image == image) return;
    *link = use->next;
    use->next = set->pending_native;
    set->pending_native = use;
}

static void texture_publication_entry_reserve(TextureImagePublication *publication) {
    if (publication->entry_count < publication->entry_capacity / 2) return;
    uint32_t capacity = publication->entry_capacity ? publication->entry_capacity * 2 : 16;
    if (capacity < publication->entry_capacity || (size_t)capacity > SIZE_MAX / sizeof(TexturePublicationEntry))
        rt_fatal("vk: indexed publication capacity overflow");
    TexturePublicationEntry *entries = calloc(capacity, sizeof *entries);
    if (!entries) rt_fatal("vk: allocating indexed material publication");
    TexturePublicationEntry *old = publication->entries;
    uint32_t old_capacity = publication->entry_capacity;
    publication->entries = entries; publication->entry_capacity = capacity;
    for (uint32_t i = 0; i < old_capacity; ++i) if (old[i].use)
        entries[texture_publication_entry_at(publication, old[i].index)] = old[i];
    free(old);
}
static void texture_publication_entry_remove(TextureImagePublication *publication, uint32_t at) {
    uint32_t mask = publication->entry_capacity - 1, hole = at;
    publication->entry_count--;
    // Close the probe chain without tombstones; repeated replacement cannot force a whole-table cleanup.
    for (uint32_t next = (at + 1) & mask; publication->entries[next].use; next = (next + 1) & mask) {
        uint32_t home = texture_use_hash(publication->entries[next].index, mask);
        if (((next - home) & mask) >= ((hole - home) & mask)) {
            publication->entries[hole] = publication->entries[next]; hole = next;
        }
    }
    publication->entries[hole] = (TexturePublicationEntry){0};
}
static void texture_publication_use_remove(TextureImagePublication *publication, TextureUseNode *node) {
    if (node->publication_refs) rt_fatal("vk: removing a referenced material publication");
    unsigned at = texture_use_hash(node->use.instance->value.id, TEXTURE_USE_BUCKETS - 1);
    TextureUseNode **link = &publication->buckets[at];
    while (*link && *link != node) link = &(*link)->hash_next;
    if (!*link) rt_fatal("vk: indexed publication lost its material hold");
    *link = node->hash_next;
    if (node->prev) node->prev->next = node->next; else publication->head = node->next;
    if (node->next) node->next->prev = node->prev;
    texture_use_unlink(node);
    texture_material_require(texture_material_use_cancel(&node->use));
    free(node); texture_uses.nodes--;
}
void vk_texture_publication_update(TextureImagePublication *publication, uint32_t entry, VkImage image) {
    if (!publication || !publication->entry_limit || entry >= publication->entry_limit)
        rt_fatal("vk: invalid indexed publication update");
    lock();
    texture_publication_native_replaced(publication, entry, image);
    TextureMaterialInstance *instance = image ? content_use_image_find(image) : NULL;
    uint32_t at = publication->entry_capacity ? texture_publication_entry_at(publication, entry) : 0;
    TextureUseNode *old = publication->entry_capacity ? publication->entries[at].use : NULL;
    if ((old ? old->use.instance : NULL) == instance) { unlock(); return; }
    TextureUseNode *replacement = NULL;
    if (instance) {
        if (!old) {
            if (publication->entry_count == publication->entry_limit)
                rt_fatal("vk: indexed publication exceeds its descriptor extent");
            texture_publication_entry_reserve(publication);
            at = texture_publication_entry_at(publication, entry);
        }
        replacement = texture_use_insert(&publication->head, publication->buckets, instance, NULL);
        if (replacement->publication_refs == UINT32_MAX) rt_fatal("vk: material publication reference overflow");
        replacement->publication_refs++;
        // Only additions need a scene capture after this publication was already used in the current batch.
        // A sealed part has no active batch: the next real use captures the current manifest independently.
        TextureUseBatch *batch = texture_use_active(vk_exec_recording_fence(), 0);
        if (batch && publication->last_batch == batch->id)
            texture_use_insert(&batch->head, batch->buckets, instance, &replacement->use);
        publication->entries[at] = (TexturePublicationEntry){entry, replacement};
        if (!old) publication->entry_count++;
    } else if (old) texture_publication_entry_remove(publication, at);
    // Acquire the incoming exact hold and required scene capture before releasing the outgoing association.
    if (old && !--old->publication_refs && !old->publication_pending) {
        old->publication_pending = 1;
        old->publication_pending_next = publication->pending;
        publication->pending = old;
    }
    // A native descriptor batch can still contain earlier writes naming a now-removed instance. Keep its hold
    // until the caller flushes that batch, even if a later lookup reclaims materials on this same thread.
    unlock();
}
void vk_texture_publication_commit_updates(TextureImagePublication *publication) {
    if (!publication || !publication->entry_limit) rt_fatal("vk: invalid indexed publication commit");
    lock();
    while (publication->pending) {
        TextureUseNode *node = publication->pending;
        publication->pending = node->publication_pending_next;
        node->publication_pending_next = NULL; node->publication_pending = 0;
        if (!node->publication_refs) texture_publication_use_remove(publication, node);
    }
    texture_native_uses_free(&publication->pending_native);
    unlock();
}
void vk_texture_publication_clear(TextureImagePublication *publication) {
    if (!publication || !publication->entry_limit) rt_fatal("vk: invalid material manifest clear");
    lock();
    for (TextureUseNode *p = publication->head; p; p = p->next) {
        p->publication_refs = 0;
        if (!p->publication_pending) {
            p->publication_pending = 1;
            p->publication_pending_next = publication->pending;
            publication->pending = p;
        }
    }
    while (publication->native) texture_publication_native_replaced(publication, publication->native->descriptor, VK_NULL_HANDLE);
    if (publication->entry_capacity)
        memset(publication->entries, 0, publication->entry_capacity * sizeof *publication->entries);
    publication->entry_count = 0;
    unlock();
}
static void texture_publication_use_locked(TextureImagePublication *publication) {
    RWT_REGION(work_use, RWT_REGION_PUBLICATION_USE);
    // An indexed set can acquire its first material after an earlier draw used the empty set. Remember that real
    // batch too, so the later descriptor update captures its incoming instance before the set can reach the GPU.
    TextureUseBatch *batch = texture_use_active(vk_exec_recording_fence(), publication->head != NULL || publication->entry_limit);
    if (batch && publication->last_batch != batch->id) {
        for (TextureUseNode *p = publication->head; p; p = p->next)
            texture_use_insert(&batch->head, batch->buckets, p->use.instance, &p->use);
        for (const TextureNativeUse *use = publication->native; use; use = use->next)
            texture_native_use_batch(batch, use);
        publication->last_batch = batch->id;
    }
}
void vk_texture_publication_use(TextureImagePublication *publication) {
    if (!publication) return;
    RWT_REGION_CALL(RWT_REGION_PUBLICATION_LOCK, lock());
    texture_publication_use_locked(publication);
    unlock();
}
void vk_texture_publication_capture(TextureImagePublication *destination, const TextureImagePublication *source) {
    if (!destination || destination->entry_limit || destination == source)
        rt_fatal("vk: invalid exact material publication capture");
    lock();
    // A native owner can survive several recording parts with the same final fence. Each actual part needs its
    // own capture even if every instance already belongs to that owner's union.
    texture_publication_use_locked(destination);
    TextureUseBatch *batch = texture_use_active(vk_exec_recording_fence(), source && source->head);
    for (TextureUseNode *p = source ? source->head : NULL; p; p = p->next) {
        if (source->entry_limit && !p->publication_refs) continue;
        TextureUseNode *owned = texture_use_insert(&destination->head, destination->buckets, p->use.instance, &p->use);
        if (batch) texture_use_insert(&batch->head, batch->buckets, owned->use.instance, &owned->use);
    }
    if (batch) destination->last_batch = batch->id;
    unlock();
}
// Descriptor pools retain the union of their published instances until their completion owner releases them.
// Extending that union also captures new entries for this recording segment; unchanged entries are hash lookups.
void vk_texture_publication_add_and_use(TextureImagePublication *publication, const VkImage *images, size_t count) {
    if (!publication) return;
    if (publication->entry_limit) rt_fatal("vk: indexed publication used as an unindexed descriptor union");
    lock();
    texture_publication_use_locked(publication);
    TextureUseBatch *batch = texture_use_active(vk_exec_recording_fence(), count != 0);
    for (size_t i = 0; i < count; ++i) {
        TextureMaterialInstance *instance = content_use_image_find(images[i]);
        if (!instance) continue;
        texture_use_insert(&publication->head, publication->buckets, instance, NULL);
        if (batch) {
            TextureUseNode *published = texture_use_find(publication->buckets, instance);
            texture_use_insert(&batch->head, batch->buckets, instance, &published->use);
        }
    }
    if (batch) publication->last_batch = batch->id;
    unlock();
}
static void texture_use_nodes_cancel(TextureUseNode *head) {
    while (head) {
        TextureUseNode *next = head->next;
        texture_use_unlink(head);
        texture_material_require(texture_material_use_cancel(&head->use));
        free(head); texture_uses.nodes--; head = next;
    }
}
// Pressure recovery asks about dormant descriptor manifests, not active scene captures. A queued physical
// instance has already lost its material residency and is waiting only for actual lifetime obligations.
int vk_texture_publication_has_retired(TextureImagePublication *publication) {
    if (!publication) return 0;
    lock();
    int retired = 0;
    for (TextureUseNode *p = publication->head; p; p = p->next)
        if (p->use.instance->queued) { retired = 1; break; }
    for (const TextureNativeUse *use = publication->native; !retired && use; use = use->next)
        retired = use->alias || vk_mem_image_hold_retired(use->hold);
    unlock();
    return retired;
}
void vk_texture_publication_release(TextureImagePublication *publication) {
    if (!publication) return;
    RWT_REGION_CALL(RWT_REGION_PUBLICATION_LOCK, lock());
    RWT_REGION_CALL(RWT_REGION_PUBLICATION_CANCEL, texture_use_nodes_cancel(publication->head));
    texture_native_uses_free(&publication->native);
    texture_native_uses_free(&publication->pending_native);
    free(publication->entries);
    free(publication); texture_uses.publications--;
    RWT_REGION_CALL(RWT_REGION_PUBLICATION_COLLECT, content_collect(64));
    unlock();
}

TextureUseBatch *vk_texture_uses_seal(uint64_t fence, uint64_t upload_submitted) {
    lock();
    TextureUseBatch *batch = texture_use_active(fence, 0);
    if (batch) {
        texture_use_active_remove(batch);
        // Every upload needed by this closed recording must actually precede its scene queue hand-over.
        for (TextureUseNode *p = batch->head; p; p = p->next)
            if (p->use.instance->upload_sequence > upload_submitted)
                rt_fatal("vk: scene material upload was not submitted before its consumer");
        batch->upload_submitted = upload_submitted;
        if (batch->part) __atomic_store_n(&batch->part->handover, TEXTURE_BATCH_SEALED, __ATOMIC_RELEASE);
        __atomic_store_n(&batch->handover, TEXTURE_BATCH_SEALED, __ATOMIC_RELEASE);
        if (texture_uses.sealed_tail) texture_uses.sealed_tail->next = batch;
        else texture_uses.sealed_head = batch;
        texture_uses.sealed_tail = batch;
    }
    unlock();
    return batch;
}
// These two functions are the only calls allowed while the queue lock is held. No allocation, texture lock or
// Vulkan call occurs here. The queue drops its pointer before completion can reclaim the batch.
void vk_texture_uses_handed(TextureUseBatch *batch) {
    if (!batch) return;
    uint32_t expected = TEXTURE_BATCH_SEALED;
    if (batch->part) __atomic_store_n(&batch->part->handover, TEXTURE_BATCH_HANDED, __ATOMIC_RELEASE);
    if (!__atomic_compare_exchange_n(&batch->handover, &expected, TEXTURE_BATCH_HANDED, 0,
                                     __ATOMIC_RELEASE, __ATOMIC_RELAXED)) abort();
}
void vk_texture_uses_abandon(TextureUseBatch *batch) {
    if (!batch) return;
    uint32_t expected = TEXTURE_BATCH_SEALED;
    if (batch->part) __atomic_store_n(&batch->part->handover, TEXTURE_BATCH_ABANDONED, __ATOMIC_RELEASE);
    if (!__atomic_compare_exchange_n(&batch->handover, &expected, TEXTURE_BATCH_ABANDONED, 0,
                                     __ATOMIC_RELEASE, __ATOMIC_RELAXED)) abort();
}
void vk_texture_uses_cancel(uint64_t fence) {
    lock();
    TextureUseBatch *batch = texture_use_active(fence, 0);
    if (batch) {
        texture_use_active_remove(batch);
        texture_use_nodes_cancel(batch->head);
        texture_use_batch_free(batch);
    }
    unlock();
}
static void texture_use_completed_locked(void) {
    TextureUseBatch **link = &texture_uses.sealed_head;
    while (*link) {
        TextureUseBatch *batch = *link;
        uint32_t state = __atomic_load_n(&batch->handover, __ATOMIC_ACQUIRE);
        if (state != TEXTURE_BATCH_ABANDONED &&
            (state != TEXTURE_BATCH_HANDED || batch->fence > texture_uses.completed)) break;
        if (state == TEXTURE_BATCH_ABANDONED) texture_use_nodes_cancel(batch->head);
        else {
            RcLifetimeCompletion done = {0};
            texture_material_require(rc_lifetime_completion_add(&done, TEXTURE_MATERIAL_GRAPHICS_DOMAIN, batch->fence));
            for (TextureUseNode *p = batch->head; p;) {
                TextureUseNode *next = p->next;
                texture_use_unlink(p);
                texture_material_require(texture_material_use_submit(&p->use, batch->upload_submitted, &done));
                texture_material_require(texture_material_use_complete(&p->use, texture_uses.upload_completed, &done));
                free(p); texture_uses.nodes--; p = next;
            }
        }
        *link = batch->next;
        texture_use_batch_free(batch);
    }
    if (!texture_uses.sealed_head) texture_uses.sealed_tail = NULL;
}
void vk_texture_uses_complete(uint64_t fence, uint64_t upload_completed) {
    lock();
    if (fence > texture_uses.completed) texture_uses.completed = fence;
    if (upload_completed > texture_uses.upload_completed) texture_uses.upload_completed = upload_completed;
    texture_use_completed_locked();
    content_collect(64);
    unlock();
}
void vk_texture_uses_idle(uint64_t last_handed, uint64_t upload_completed) {
    lock();
    uint64_t completed = texture_uses.completed;
    if (last_handed > texture_uses.completed) texture_uses.completed = last_handed;
    if (upload_completed > texture_uses.upload_completed) texture_uses.upload_completed = upload_completed;
    texture_use_completed_locked();
    // The proof covers these handed batches, not a future scene reusing an unsignaled timeline value after restart.
    texture_uses.completed = completed;
    content_collect(64);
    unlock();
}
void vk_texture_uses_cancel_unsubmitted(uint64_t last_handed) {
    lock();
    for (unsigned i = 0; i < TEXTURE_USE_BUCKETS; ++i) {
        TextureUseBatch **link = &texture_uses.active[i];
        while (*link) {
            TextureUseBatch *batch = *link;
            if (batch->fence <= last_handed) { link = &batch->active_next; continue; }
            *link = batch->active_next;
            texture_use_nodes_cancel(batch->head);
            texture_use_batch_free(batch);
        }
    }
    // The queue has already detached its abandoned pointers. Earlier handed work may still be in flight, so
    // remove only canceled batches rather than advancing the normal completion frontier across that work.
    TextureUseBatch **link = &texture_uses.sealed_head, *tail = NULL;
    while (*link) {
        TextureUseBatch *batch = *link;
        if (__atomic_load_n(&batch->handover, __ATOMIC_ACQUIRE) != TEXTURE_BATCH_ABANDONED) {
            tail = batch; link = &batch->next; continue;
        }
        *link = batch->next;
        texture_use_nodes_cancel(batch->head);
        texture_use_batch_free(batch);
    }
    texture_uses.sealed_tail = tail;
    content_collect(64);
    unlock();
}
// Healthy shutdown calls this after all issued scene work and uploads completed and replay stopped. Unsealed CPU
// recording is canceled explicitly. Publications belong to their descriptor sets and are released by that owner.
void vk_texture_uses_shutdown(void) {
    lock();
    texture_use_completed_locked();
    if (texture_uses.sealed_head) rt_fatal("vk: material scene uses remain pending at healthy shutdown");
    for (unsigned i = 0; i < TEXTURE_USE_BUCKETS; ++i) {
        TextureUseBatch *batch = texture_uses.active[i];
        while (batch) {
            TextureUseBatch *next = batch->active_next;
            texture_use_nodes_cancel(batch->head);
            texture_use_batch_free(batch);
            batch = next;
        }
        texture_uses.active[i] = NULL;
    }
    content_collect(UINT64_MAX);
    unlock();
}

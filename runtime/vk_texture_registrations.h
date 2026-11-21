// Included by vk_texture.c. The texture lock owns payloads and their exact native publication.
#pragma once

enum { TEXTURE_REGISTRATION_RECORDS = VK_POOL_TEX * 2, TEXTURE_REGISTRATION_SCOPES = 4096 };
enum { TEXTURE_REGISTRATION_CHANGES = 65536 };

struct TextureRegistration {
    TextureRegistration *retired_next;
    Tex *source;
    TextureRegistration *source_next, **source_previous;
    RcRegistryHandle handle;
    RcRegistryCapture retained_record;
    uint64_t publication;
    tex_t descriptor;
    view_t view, exact_view, selected_view;
    VkTexImage exact, selected;
    TextureFlat3D *flat3d;
    uint64_t selected_instance, exact_instance;
    int native_slot;
    uint8_t has_view, has_exact_view, has_selected_view;
};

struct VkTextureRegistrationScope {
    RcRegistryScope horizon;
    uint64_t publication;
    uint32_t references;
};

static RcRegistry *texture_registrations;
static TextureImagePublication *texture_registration_images;
static TextureRegistration *texture_retired_registration, *texture_retired_registration_tail;
static uint32_t texture_retired_count;
static uint64_t texture_registration_publication = 1;
static VkTextureRegistrationScope *texture_recording_scope[2];
static uint64_t texture_recording_revision, texture_recording_next;
typedef struct { uint64_t owner; int logical; } TextureRegistrationChange;
static TextureRegistrationChange texture_registration_slots[TEXTURE_REGISTRATION_CHANGES];

static Tex *texture_registration_source(const TexSlot *slot) {
    return slot->registration ? slot->registration->source : NULL;
}

static void texture_registration_source_unlink(TextureRegistration *record) {
    if (!record->source) return;
    *record->source_previous = record->source_next;
    if (record->source_next) record->source_next->source_previous = record->source_previous;
    record->source = NULL;
    record->source_next = NULL;
    record->source_previous = NULL;
}

// The registration owns logical selection. Contents, current mapping and the native instance
// have separate proof and lifetime checks at the actual consumer.
static int texture_registration_source_attach(const TexSlot *slot, Tex *e) {
    TextureRegistration *record = slot->registration;
    if (!record || !e->source_canonical || !e->source_mapping || !source_private_sample(e) ||
        e->released || record->descriptor.serial != e->t.serial || record->descriptor.L.size != e->t.L.size)
        return 0;
    if (record->source == e) return 1;
    TexKey key;
    make_key(&record->descriptor, &key);
    if (memcmp(&e->key, &key, sizeof key)) return 0;
    texture_registration_source_unlink(record);
    record->source = e;
    record->source_previous = &e->source_registrations;
    record->source_next = e->source_registrations;
    if (record->source_next) record->source_next->source_previous = &record->source_next;
    e->source_registrations = record;
    return 1;
}

static void texture_registration_source_clear(Tex *e) {
    if (!e->source_registrations) return;
    do {
        TextureRegistration *record = e->source_registrations;
        dirty_push((uint32_t)record->native_slot);
        texture_registration_source_unlink(record);
    } while (e->source_registrations);
    slots_gen++;
}

static void texture_registration_source_dirty(Tex *e, void (*visit)(int, void *), void *arg) {
    for (TextureRegistration *record = e->source_registrations; record; record = record->source_next)
        visit(record->native_slot, arg);
}

static uint64_t texture_registration_key(uint64_t pool, uint32_t slot) {
    return (pool << 21) | ((uint64_t)slot + 1);
}

static RcRegistry *texture_registration_store(void) {
    if (!texture_registrations) {
        texture_registrations = rc_registry_create((RcRegistryConfig){TEXTURE_REGISTRATION_RECORDS, 1,
                                                                      TEXTURE_REGISTRATION_SCOPES});
        if (!texture_registrations) rt_fatal("vk: texture registration store allocation failed");
    }
    return texture_registrations;
}

static void texture_registration_collect(uint32_t budget) {
    if (!texture_retired_registration) return;
    rc_registry_collect(texture_registrations, budget);
    uint32_t visits = texture_retired_count < budget ? texture_retired_count : budget;
    for (uint32_t i = 0; i < visits; ++i) {
        TextureRegistration *record = texture_retired_registration;
        texture_retired_registration = record->retired_next;
        if (!texture_retired_registration) texture_retired_registration_tail = NULL;
        record->retired_next = NULL;
        if (rc_registry_registration_reachable(texture_registrations, record->handle)) {
            if (texture_retired_registration_tail) texture_retired_registration_tail->retired_next = record;
            else texture_retired_registration = record;
            texture_retired_registration_tail = record;
            continue;
        }
        texture_retired_count--;
        if (record->exact.image) {
            vk_texture_publication_update(texture_registration_images, record->handle.index, VK_NULL_HANDLE);
            vk_texture_publication_commit_updates(texture_registration_images);
        }
        if (record->flat3d) flat3d_retire(record->flat3d);
        if (record->retained_record.id)
            rc_registry_capture_release(texture_registrations, record->retained_record);
        free(record);
    }
}

static VkTextureRegistrationScope *texture_registration_scope_new(void) {
    VkTextureRegistrationScope *scope = calloc(1, sizeof *scope);
    if (!scope) rt_fatal("vk: texture registration scope allocation failed");
    if (rc_registry_scope_begin(texture_registration_store(), &scope->horizon) != RC_REGISTRY_OK)
        rt_fatal("vk: texture registration horizon capacity exhausted");
    scope->publication = __atomic_load_n(&texture_registration_publication, __ATOMIC_RELAXED);
    scope->references = 1;
    return scope;
}

VkTextureRegistrationScope *vk_texture_registration_scope_begin(void) {
    lock();
    VkTextureRegistrationScope *scope = texture_registration_scope_new();
    unlock();
    return scope;
}

static void texture_recording_revision_update(void) {
    uint64_t revision = 0;
    if (texture_recording_scope[0] || texture_recording_scope[1]) {
        if (texture_recording_next == UINT64_MAX) rt_fatal("vk: texture recording identity exhausted");
        revision = ++texture_recording_next;
    }
    __atomic_store_n(&texture_recording_revision, revision, __ATOMIC_RELEASE);
}

VkTextureRegistrationScope *vk_texture_registration_record_begin(VkTextureRegistrationScope *carried, int resumed) {
    lock();
    if (texture_recording_scope[resumed]) rt_fatal("vk: overlapping texture recording scope");
    VkTextureRegistrationScope *scope = carried ? carried : texture_registration_scope_new();
    texture_recording_scope[resumed] = scope;
    texture_recording_revision_update();
    unlock();
    return scope;
}

void vk_texture_registration_record_end(VkTextureRegistrationScope *scope, int resumed) {
    lock();
    if (!scope || texture_recording_scope[resumed] != scope) rt_fatal("vk: ending an unrelated texture recording scope");
    texture_recording_scope[resumed] = NULL;
    texture_recording_revision_update();
    unlock();
    vk_backend_registration_record_progress();
}

uint64_t vk_texture_registration_record_revision(void) {
    return __atomic_load_n(&texture_recording_revision, __ATOMIC_ACQUIRE);
}

static uint64_t texture_registration_cutoff_locked(uint64_t queued) {
    for (int i = 0; i < 2; ++i)
        if (texture_recording_scope[i] && texture_recording_scope[i]->publication > queued)
            queued = texture_recording_scope[i]->publication;
    return queued;
}

VkTextureRegistrationScope *vk_texture_registration_scope_retain(VkTextureRegistrationScope *scope) {
    if (!scope) return NULL;
    lock();
    if (!scope->references || scope->references == UINT32_MAX) rt_fatal("vk: invalid texture history retain");
    scope->references++;
    unlock();
    return scope;
}

int vk_texture_registration_scope_current(const VkTextureRegistrationScope *scope) {
    return !scope || scope->publication == __atomic_load_n(&texture_registration_publication, __ATOMIC_ACQUIRE);
}

uint64_t vk_texture_registration_scope_publication(const VkTextureRegistrationScope *scope) {
    return scope ? scope->publication : 0;
}

void vk_texture_registration_scope_end(VkTextureRegistrationScope *scope) {
    if (!scope) return;
    lock();
    if (!scope->references) rt_fatal("vk: invalid texture history release");
    if (!--scope->references) {
        if (rc_registry_scope_end(texture_registrations, scope->horizon) != RC_REGISTRY_OK)
            rt_fatal("vk: texture registration horizon lost");
        free(scope);
        texture_registration_collect(64);
    }
    unlock();
}

static const VkTexImage *texture_registration_selected(const TexSlot *slot, int id, view_t *view,
                                                       uint8_t *has_view, uint64_t *instance_id) {
    const TextureRegistration *record = slot->registration;
    if (record->selected_instance) {
        const TextureMaterialInstance *instance = content_use_image_find(record->selected.image);
        if (instance && instance->value.id == record->selected_instance) {
            *view = record->selected_view;
            *has_view = record->has_selected_view;
            *instance_id = instance->value.id;
            return &record->selected;
        }
    }
    const Tex *entry = record->source;
    if (!entry && slot->sample_owned && slot->cgen == agen[sinfo[id] & 0xffff]) entry = slot->ce;
    if (entry) {
        const TextureMaterialInstance *instance = content_use_image_find(entry->pub.image);
        if (entry->t.serial == record->descriptor.serial && entry->material && entry->material_version.id &&
            entry->uploaded && !entry->up_retry && entry->pub.image && !entry->pub.gpu_owned && !entry->pub.skip &&
            !entry->pub.gone && entry->pub.type == VK_IMAGE_TYPE_2D && instance &&
            entry->material->resident.instance == instance) {
            if (record->source) {
                int has_current_view;
                slot_result_view(slot, entry, view, &has_current_view);
                *has_view = has_current_view != 0;
            } else {
                *view = slot->cview;
                *has_view = slot->chas_view;
            }
            *instance_id = instance->value.id;
            return &entry->pub;
        }
    }
    return NULL;
}

static int texture_registration_pending(const TexSlot *slot, uint64_t queued_publication) {
    const TextureRegistration *record = slot->registration;
    // This is a live record. The latest admitted unread horizon can reach it iff it was already published.
    // A native instance selected by an earlier read says nothing about that unread scope's producer order.
    return record && record->descriptor.mem && record->publication <= queued_publication;
}

static uint32_t texture_registration_exact_layer(const TextureRegistration *record) {
    return record->has_exact_view && record->exact_view.set_layers ? record->exact_view.min_layer : 0;
}

static void texture_registration_hold_flat3d(TextureRegistration *record) {
    const Tex *source = record->source;
    if (!source || !source->material || source->pub.image != record->exact.image ||
        !source->material->resident.instance) return;
    uint64_t instance = source->material->resident.instance->value.id;
    uint32_t layer = texture_registration_exact_layer(record);
    for (TextureFlat3D *copy = source->flat3d; copy; copy = copy->next) {
        if (copy->layer != layer || !copy->material || copy->source_instance != instance || copy->image.gone ||
            copy->source != record->exact.image || copy->upload != source->n_up ||
            copy->revision != source->content_revision || copy->ordered) continue;
        TextureFlat3D *held = malloc(sizeof *held);
        if (!held) return; // The exact archived parent can still supply the child at its actual read.
        *held = *copy;
        held->next = NULL;
        texture_material_acquire(held->material, 1);
        record->flat3d = held;
        return;
    }
}

// A reached read can keep its selected immutable provider across retirement. Cold queued reads must
// resolve before destructive frontend publication, or report missing exact material at consumption.
static void texture_registration_hold(TexSlot *slot, int id) {
    TextureRegistration *record = slot->registration;
    if (!record || record->exact.image ||
        !rc_registry_registration_reachable(texture_registrations, record->handle)) return;
    view_t view;
    uint8_t has_view;
    uint64_t instance;
    const VkTexImage *image = texture_registration_selected(slot, id, &view, &has_view, &instance);
    if (!image) return;
    if (rc_registry_capture(texture_registrations, record->handle, &record->retained_record) != RC_REGISTRY_OK)
        rt_fatal("vk: texture history lost a reachable registration");
    if (!texture_registration_images)
        texture_registration_images = vk_texture_publication_create_indexed(TEXTURE_REGISTRATION_RECORDS + 1);
    vk_texture_publication_update(texture_registration_images, record->handle.index, image->image);
    record->exact = *image;
    record->exact_instance = instance;
    record->exact_view = view;
    record->has_exact_view = has_view;
    texture_registration_hold_flat3d(record);
}

static int texture_registration_same(const TexSlot *slot, const tex_t *t, const view_t *view) {
    const TextureRegistration *record = slot->registration;
    return record && !memcmp(&record->descriptor, t, sizeof *t) &&
        record->has_view == (view != NULL) && (!view || !memcmp(&record->view, view, sizeof *view));
}

static void texture_registration_queue_retirement(TextureRegistration *old) {
    if (!old) return;
    texture_registration_source_unlink(old);
    if (texture_retired_registration_tail) texture_retired_registration_tail->retired_next = old;
    else texture_retired_registration = old;
    texture_retired_registration_tail = old;
    texture_retired_count++;
    texture_registration_collect(64);
}

static uint64_t texture_registration_changed(uint64_t owner, int logical) {
    uint64_t current = __atomic_load_n(&texture_registration_publication, __ATOMIC_RELAXED);
    if (current == UINT64_MAX) rt_fatal("vk: texture registration publication exhausted");
    texture_registration_slots[(current + 1) % TEXTURE_REGISTRATION_CHANGES] =
        (TextureRegistrationChange){owner, logical};
    __atomic_store_n(&texture_registration_publication, current + 1, __ATOMIC_RELEASE);
    return current + 1;
}

static void texture_registration_publish(TexSlot *slot, int id, const tex_t *t, const view_t *view) {
    TextureRegistration *old = slot->registration;
    if (old) texture_registration_hold(slot, id);
    TextureRegistration *record = calloc(1, sizeof *record);
    if (!record) rt_fatal("vk: texture registration payload allocation failed");
    record->descriptor = *t;
    record->native_slot = id;
    if (view) { record->view = *view; record->has_view = 1; }
    RcSourceRange range = {RC_GUEST_COMPATIBILITY_BACKING, (uint64_t)(uintptr_t)t->mem, t->L.size};
    if (!rc_range_valid(range)) range = (RcSourceRange){RC_GUEST_COMPATIBILITY_BACKING, 0, 1};
    uint64_t object = t->serial ? t->serial : UINT64_MAX; // uninitialized descriptor metadata has no texel lease
    RcRegistry *store = texture_registration_store();
    uint64_t key = texture_registration_key(slot->pool_owner, (uint32_t)slot->logical_id);
    uint64_t payload = (uint64_t)(uintptr_t)record;
    RcRegistryResult registered = rc_registry_register(store, key, object, range, payload, &record->handle);
    if (registered == RC_REGISTRY_CAPACITY) {
        // Some unreachable history still owns native payload captures until the adapter releases them.
        texture_registration_collect(texture_retired_count);
        registered = rc_registry_register(store, key, object, range, payload, &record->handle);
    }
    if (registered != RC_REGISTRY_OK) {
        RcRegistryStats stats = rc_registry_stats(store);
        rt_fatal("vk: texture registration publish failed (%d), records %u/%u live %u retired %u payloads %u "
                 "captures %u scopes %u pool %llu slot %d",
                 (int)registered, stats.records, TEXTURE_REGISTRATION_RECORDS, stats.live_descriptors, stats.retired_records,
                 texture_retired_count, stats.captures, stats.scopes, (unsigned long long)slot->pool_owner, slot->logical_id);
    }
    slot->registration = record;
    texture_registration_queue_retirement(old);
    record->publication = texture_registration_changed(slot->pool_owner, slot->logical_id);
}

static void texture_registration_finalize(TexSlot *slot, int id) {
    if (!slot->registration) return;
    texture_registration_hold(slot, id);
    uint64_t object = slot->t.serial ? slot->t.serial : UINT64_MAX;
    if (rc_registry_retire_registration(texture_registrations,
                                        texture_registration_key(slot->pool_owner, (uint32_t)slot->logical_id),
                                        object) != RC_REGISTRY_OK)
        rt_fatal("vk: texture registration changed before finalization");
    texture_registration_queue_retirement(slot->registration);
    slot->registration = NULL;
    texture_registration_changed(slot->pool_owner, slot->logical_id);
}

uint64_t vk_texture_registration_publication(void) {
    VkTextureRegistrationScope *scope = vk_exec_texture_registration_scope();
    return scope ? scope->publication : __atomic_load_n(&texture_registration_publication, __ATOMIC_ACQUIRE);
}

uint64_t vk_texture_registration_scope_token(void) {
    VkTextureRegistrationScope *scope = vk_exec_texture_registration_scope();
    return scope && scope->publication != __atomic_load_n(&texture_registration_publication, __ATOMIC_ACQUIRE) ?
        scope->publication : 0;
}

int vk_texture_registration_changes_locked(uint64_t owner, uint64_t from, uint64_t to,
                                           void (*visit)(int, void *), void *arg) {
    uint64_t older = from < to ? from : to, newer = from < to ? to : from;
    uint64_t published = __atomic_load_n(&texture_registration_publication, __ATOMIC_RELAXED);
    if (!older || newer > published || published - older > TEXTURE_REGISTRATION_CHANGES) return 0;
    while (older < newer) {
        TextureRegistrationChange change = texture_registration_slots[++older % TEXTURE_REGISTRATION_CHANGES];
        if (change.owner == owner) visit(change.logical, arg);
    }
    return 1;
}

static int texture_registration_ordinary(const tex_t *descriptor) {
    return descriptor->mem && !nvn_gpu_only((uintptr_t)descriptor->mem) && tex_mirrored(descriptor) &&
        !(descriptor->pool_flags & (NVN_MEMORY_POOL_FLAGS_VIRTUAL | NVN_MEMORY_POOL_FLAGS_PHYSICAL)) &&
        descriptor->d.samples <= 1 && descriptor->d.target != NVN_TEXTURE_TARGET_1D &&
        descriptor->d.target != NVN_TEXTURE_TARGET_1D_ARRAY && descriptor->d.target != NVN_TEXTURE_TARGET_3D &&
        descriptor->d.target != NVN_TEXTURE_TARGET_BUFFER;
}

static int texture_registration_resolve_owner(int logical, const TextureRegistration **record, int *native,
                                              const TextureUseNode **retained) {
    *record = NULL;
    if (retained) *retained = NULL;
    VkTextureRegistrationScope *scope = vk_exec_texture_registration_scope();
    if (!scope && !native) return 0;
    uint64_t owner = vk_exec_texture_pool();
    int id = vk_texture_pool_slot_locked(owner, logical);
    if (native) *native = id;
    if (!scope || logical < 0 || logical >= 1 << 20) return 0;
    int current = id >= 0 && tslots[id].registration;
    if (!vk_texture_registration_scope_token() && current) return 0;
    RcRegistryRecord resolved;
    if (rc_registry_resolve(texture_registrations, texture_registration_key(owner, (uint32_t)logical),
                            scope->horizon, &resolved) != RC_REGISTRY_OK) return VK_TEXTURE_REGISTRATION_ABSENT;
    const TextureRegistration *p = (const TextureRegistration *)(uintptr_t)resolved.interpretation;
    if (current && tslots[id].registration == p && !resolved.finalized) return 0;
    if (!p->descriptor.mem) return VK_TEXTURE_REGISTRATION_ABSENT;
    *record = p;
    if (!texture_registration_ordinary(&p->descriptor) || !p->exact_instance || p->exact.gone)
        return VK_TEXTURE_REGISTRATION_GONE;
    const TextureUseNode *material = texture_publication_owner_locked(texture_registration_images, p->handle.index,
                                                                   p->exact.image);
    if (!material || material->use.instance->value.id != p->exact_instance)
        return VK_TEXTURE_REGISTRATION_GONE;
    if (retained) *retained = material;
    return VK_TEXTURE_REGISTRATION_EXACT;
}

static int texture_registration_resolve(int logical, const TextureRegistration **record, int *native) {
    return texture_registration_resolve_owner(logical, record, native, NULL);
}

int vk_texture_registration_expected_locked(int id, VkImage *expected) {
    const TextureRegistration *record;
    int status = texture_registration_resolve(id, &record, NULL);
    if (expected) *expected = status == VK_TEXTURE_REGISTRATION_EXACT ? record->exact.image : VK_NULL_HANDLE;
    return status;
}

int vk_texture_registration_view_format_locked(int logical, uint32_t *format) {
    const TextureRegistration *record;
    int native;
    if (texture_registration_resolve(logical, &record, &native) != VK_TEXTURE_REGISTRATION_CURRENT || native < 0)
        return 0;
    record = tslots[native].registration;
    if (!record || tslots[native].finalized || !record->has_view || !record->view.set_format) return 0;
    *format = record->view.format;
    return 1;
}

void vk_texture_registration_report_unavailable(int logical) {
    lock();
    const TextureRegistration *record;
    int status = texture_registration_resolve(logical, &record, NULL);
    uint64_t horizon = vk_texture_registration_publication();
    uint64_t current = __atomic_load_n(&texture_registration_publication, __ATOMIC_RELAXED);
    if (record) {
        rt_log("vk: required texture registration slot %d status %d horizon %llu current %llu "
               "registration %llu publication %llu object %llu target %d format %d pool flags %#x "
               "selected %d retained %d",
               logical, status, (unsigned long long)horizon, (unsigned long long)current,
               (unsigned long long)record->handle.id, (unsigned long long)record->publication,
               (unsigned long long)record->descriptor.serial, record->descriptor.d.target, record->descriptor.d.format,
               (unsigned)record->descriptor.pool_flags, record->selected_instance != 0, record->exact.image != VK_NULL_HANDLE);
    } else {
        rt_log("vk: required texture registration slot %d status %d horizon %llu current %llu",
               logical, status, (unsigned long long)horizon, (unsigned long long)current);
    }
    if (status == VK_TEXTURE_REGISTRATION_CURRENT) texture_source_report_locked(logical);
    unlock();
}

void vk_texture_registration_note_locked(int id, const VkTexImage *image, const view_t *view, int has_view) {
    if (!image || !image->image || image->gone || image->gpu_owned || image->skip || image->type != VK_IMAGE_TYPE_2D) return;
    if (id < 0 || id >= ntslots || !tslots[id].registration) return;
    TexSlot *slot = &tslots[id];
    TextureRegistration *record = slot->registration;
    const Tex *entry = record->source;
    if (!entry && slot->cgen == agen[sinfo[id] & 0xffff]) entry = slot->ce;
    const TextureMaterialInstance *instance = content_use_image_find(image->image);
    if (!instance || !entry || &entry->pub != image || entry->t.serial != record->descriptor.serial ||
        !entry->material || entry->material->resident.instance != instance ||
        !entry->material_version.id || !entry->uploaded || entry->up_retry) return;
    record->selected = *image;
    record->selected_view = *view;
    record->has_selected_view = has_view != 0;
    record->selected_instance = instance->value.id;
}

static int texture_registration_image(int id, const VkTexImage **image, view_t *view, int *has_view,
                                      uint64_t *source_size, int *native) {
    const TextureRegistration *record;
    const TextureUseNode *owner;
    int status = texture_registration_resolve_owner(id, &record, native, &owner);
    if (status <= 0) return status;
    texture_use_scene_material_locked(owner->use.instance, &owner->use);
    *image = &record->exact;
    if (view) *view = record->exact_view;
    if (has_view) *has_view = record->has_exact_view;
    if (source_size) *source_size = record->descriptor.L.size;
    return 1;
}

int vk_texture_registration_image_locked(int id, const VkTexImage **image, view_t *view, int *has_view,
                                         uint64_t *source_size) {
    return texture_registration_image(id, image, view, has_view, source_size, NULL);
}

static void texture_registration_shutdown(void) {
    if (!texture_registrations) return;
    assert(!rc_registry_stats(texture_registrations).scopes);
    assert(!texture_recording_scope[0] && !texture_recording_scope[1]);
    vk_texture_publication_release(texture_registration_images);
    texture_registration_images = NULL;
    for (int i = 0; i < ntslots; ++i) {
        TextureRegistration *record = tslots[i].registration;
        if (record) {
            texture_registration_source_unlink(record);
            if (record->flat3d) flat3d_retire(record->flat3d);
        }
        free(record);
        tslots[i].registration = NULL;
    }
    while (texture_retired_registration) {
        TextureRegistration *retired = texture_retired_registration;
        texture_retired_registration = retired->retired_next;
        if (retired->flat3d) flat3d_retire(retired->flat3d);
        free(retired);
    }
    texture_retired_registration_tail = NULL;
    texture_retired_count = 0;
    rc_registry_destroy(texture_registrations);
    texture_registrations = NULL;
}

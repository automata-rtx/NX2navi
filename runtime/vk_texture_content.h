// Included by vk_texture.c with the texture lock held. Index entries pin immutable GPU contents independently of
// logical textures. Only final physical retirement purges image views and defers the allocation.
enum { MATERIAL_SLOTS = 8192, MATERIAL_LEVELS = 32, MATERIAL_RECLAIM_SCAN = 512 };
#define MATERIAL_SNAPSHOT_LIMIT (UINT64_C(32) << 20)
typedef struct {
    TextureMaterial *material;
    TexKey source;
    ImgKey intent;
    VkTexImage representation;
    TexVkFormat format;
    uint64_t source_bytes;
    uint8_t digest[32];
} MaterialSlot;
typedef struct {
    TexKey source;
    ImgKey intent;
    TextureSourceStamp stamp;
    uint64_t material_id;
    uint32_t slot; // Slot + 1 and owner generation; never a potentially dangling owner pointer.
} MaterialCertificate;
static MaterialSlot material_slots[MATERIAL_SLOTS];
static MaterialCertificate material_certificates[MATERIAL_SLOTS];
static uint64_t material_snapshot_bytes, source_capture_next;
typedef struct {
    uint8_t *bytes;
    uint64_t size;
    RcBudgetTicket budget;
    __typeof__(((Tex *)0)->t) recipe;
    VkTexImage representation;
    TexVkFormat format;
    ImgKey intent;
    TextureSourceStamp stamp;
    RcRegistryReadVersion version;
    uint8_t digest[32];
    uint32_t restore_refs;
} MaterialSource;
static int content_restore(Tex *, TextureMaterial *, const TextureSourceStamp *);
static void content_upload_reconstructed(Tex *);
static uint32_t content_recovery_domains(const Tex *);
static int content_residency_reclaimable(const TextureMaterial *, int);
static struct {
    uint64_t hashes, hash_bytes, hash_ns, lookups, hits, copy_ns, copied_bytes, avoided_texels, collisions;
    uint64_t early_hits, certificate_hits, snapshot_bytes, snapshot_refused, cow, cow_bytes, retired_bytes, same_owner_hits;
    uint64_t evicted, evicted_bytes, restored, restored_bytes, provider_dropped;
} material_stats;
enum { MATERIAL_RECOVERY_STARTS = 4 };
enum { MATERIAL_PREPARE_REQUIRED, MATERIAL_PREPARE_OPTIONAL };
static struct {
    RcBudgetAvailability available;
    uint64_t bytes;
    uint32_t submit, starts;
} material_recovery;
static void content_recovery_wait(Tex *e, uint32_t domains) {
    resource_core_budget_availability(&e->source_retry);
    e->source_retry_domains = domains;
    e->source_retry_peak = (RcAdmissionVector){{0}};
    e->source_retry_source = 0;
    e->source_retry_submit = submit_epoch;
}
static void content_recovery_wait_peak(Tex *e, const RcAdmissionVector *peak, const RcBudgetAvailability *before) {
    e->source_retry = *before;
    e->source_retry_peak = *peak;
    e->source_retry_domains = 0;
    for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++)
        if (peak->amount[d]) e->source_retry_domains |= 1u << d;
    e->source_retry_source = 0;
    e->source_retry_submit = submit_epoch;
}
// Preserve the evidence from before a failed capture. A later clean observation can have advanced its cutoff
// without changing bytes, so generation alone is not a retry event. Only this source's authority or memory lease
// can rearm the attempt; unrelated registry writes and mapping events do not make another capture useful.
static void content_recovery_wait_source(Tex *e, const TextureSourceStamp *stamp,
                                          const TextureSourceAuthority *authority) {
    int ordered = rc_registry_position_valid(authority->ordered.position);
    e->source_retry_handle = ordered ? authority->ordered.handle.id : authority->legacy.handle.id;
    e->source_retry_revision = ordered ? authority->ordered.revision : authority->legacy.revision;
    e->source_retry_guard = ordered ? authority->ordered.guard : authority->legacy.guard;
    e->source_retry_stamp = *stamp;
    e->source_retry_source = 1;
    e->source_retry_domains = 0;
    e->source_retry_submit = submit_epoch;
}
static int content_recovery_source_changed(const Tex *e) {
    const TextureSourceAuthority *authority = &e->source_authority;
    int ordered = rc_registry_position_valid(authority->ordered.position);
    uint64_t handle = ordered ? authority->ordered.handle.id : authority->legacy.handle.id;
    uint64_t revision = ordered ? authority->ordered.revision : authority->legacy.revision;
    uint64_t guard = ordered ? authority->ordered.guard : authority->legacy.guard;
    return handle != e->source_retry_handle || revision != e->source_retry_revision ||
        guard != e->source_retry_guard || e->source_stamp.lineage != e->source_retry_stamp.lineage ||
        !mem_mapping_range_unchanged(e->pub.addr, e->t.L.size, e->source_retry_stamp.mapping) ||
        !mem_protection_range_unchanged(e->pub.addr, e->t.L.size, e->source_retry_stamp.protection);
}
static int content_recovery_due(Tex *e) {
    if ((!e->source_retry_domains && !e->source_retry_source) || e->source_retry_submit == submit_epoch || e->material || !e->uploaded ||
        e->source_copy || e->released || !e->pub.image || !e->alloc) return 0;
    e->source_retry_submit = submit_epoch;
    if (material_recovery.submit != submit_epoch) {
        material_recovery.submit = submit_epoch;
        material_recovery.starts = 0; material_recovery.bytes = 0;
        resource_core_budget_availability(&material_recovery.available);
    }
    const RcBudgetAvailability *now = &material_recovery.available;
    int source_changed = e->source_retry_source && content_recovery_source_changed(e);
    if ((!source_changed && now->progress == e->source_retry.progress) || now->available.amount[RC_BUDGET_RAM] < e->t.L.size ||
        now->available.amount[RC_BUDGET_COMMIT] < e->t.L.size) return 0;
    int improved = source_changed;
    for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) {
        if (e->source_retry_peak.amount[d] > now->available.amount[d] && !source_changed) return 0;
        if ((e->source_retry_domains & (1u << d)) && now->available.amount[d] > e->source_retry.available.amount[d])
            improved = 1;
    }
    return improved && material_recovery.starts < MATERIAL_RECOVERY_STARTS &&
        e->t.L.size <= MATERIAL_SNAPSHOT_LIMIT - material_recovery.bytes;
}

static void content_associate(Tex *e, TextureMaterial *m) {
    assert(!e->material && !e->material_prev && !e->material_next);
    e->material = m;
    e->material_next = m->associations;
    if (m->associations) m->associations->material_prev = e;
    m->associations = e;
}
static void content_disassociate(Tex *e) {
    TextureMaterial *m = e->material;
    assert(m);
    if (e->material_prev) e->material_prev->material_next = e->material_next;
    else { assert(m->associations == e); m->associations = e->material_next; }
    if (e->material_next) e->material_next->material_prev = e->material_prev;
    e->material_prev = e->material_next = NULL;
    e->material = NULL;
}
static void content_provider_free(TextureMaterial *m) {
    MaterialSource *p = m->provider;
    if (!p) return;
    assert(!p->restore_refs);
    assert(material_snapshot_bytes >= p->size && material_ledger.provider_bytes >= p->size);
    material_snapshot_bytes -= p->size;
    material_ledger.provider_bytes -= p->size;
    resource_core_budget_retire(&p->budget);
    free(p->bytes);
    resource_core_budget_destroyed(&p->budget);
    free(p); m->provider = NULL;
}
static void content_material_free(TextureMaterial *m) {
    assert(!m->associations);
    content_provider_free(m);
    free(m);
}
// Host snapshots are optional while their exact image is available. Under simultaneous GPU pressure, keep the
// reconstruction path for cold images that can give back residency; prefer snapshots whose images must stay.
static uint64_t content_drop_providers_pressure(uint64_t wanted, int preserve_reclaimable) {
    uint64_t freed = 0, visits = material_ledger.resident_owners;
    if (visits > MATERIAL_RECLAIM_SCAN) visits = MATERIAL_RECLAIM_SCAN;
    for (uint64_t n = 0; n < visits && freed < wanted; n++) {
        TextureMaterial *m = texture_material_resident_next(&material_ledger);
        if (!m || !m->provider || !m->resident.image) continue;
        if (((const MaterialSource *)m->provider)->restore_refs) continue;
        if (preserve_reclaimable && content_residency_reclaimable(m, 1)) continue;
        if (rc_lifetime_recovery_set(&m->lifetime, m->lifetime.id, RC_LIFETIME_RECOVERY_CURRENT_SOURCE) != RC_LIFETIME_OK)
            continue;
        freed += ((MaterialSource *)m->provider)->size;
        content_provider_free(m);
    }
    material_stats.provider_dropped += freed;
    return freed;
}
static uint64_t content_drop_providers(uint64_t wanted) {
    return content_drop_providers_pressure(wanted, 0);
}

static unsigned content_image_bucket(const uint8_t digest[32]) {
    uint32_t h;
    memcpy(&h, digest, sizeof h);
    return h & (MATERIAL_SLOTS - 1);
}
static unsigned content_source_bucket(uint64_t addr) {
    return (unsigned)((addr >> 8) ^ (addr >> 21)) & (MATERIAL_SLOTS - 1);
}
static TexKey content_image_key(const Tex *e) {
    TexKey k = e->key;
    k.addr = 0;
    return k;
}
static void content_source_discard(Tex *e) {
    if (e->source_copy && !e->source_borrowed) {
        assert(material_snapshot_bytes >= e->source_copy_size);
        material_snapshot_bytes -= e->source_copy_size;
        free(e->source_copy);
        if (e->source_budget.token.ledger) {
            resource_core_budget_retire(&e->source_budget);
            resource_core_budget_destroyed(&e->source_budget);
        }
    }
    e->source_copy = NULL;
    e->source_copy_size = 0;
    e->source_recipe = NULL;
    e->source_borrowed = 0;
    e->source_copy_stamp = (TextureSourceStamp){0};
    e->source_copy_authority = (TextureSourceAuthority){0};
    e->source_copy_version = (RcRegistryReadVersion){0};
}
// Logical invalidation preserves immutable bytes that other associations or the content index still own.
static void content_image_forget(Tex *e) {
    MaterialCertificate *c = &material_certificates[content_source_bucket(e->key.addr)];
    if (c->slot && !memcmp(&c->source, &e->key, sizeof e->key)) c->slot = 0;
    e->source_digest_valid = 0;
    memset(&e->source_stamp, 0, sizeof e->source_stamp);
}
static int content_image_eligible(const Tex *e) {
    if (e->source_blocked || !e->pub.mirrored || !e->t.mem || nvn_gpu_only(e->key.addr) || e->material_writable ||
        (e->t.pool_flags & (NVN_MEMORY_POOL_FLAGS_PHYSICAL | NVN_MEMORY_POOL_FLAGS_VIRTUAL)) || e->pub.gpu_owned || e->gpu_dirty || e->bound_to ||
        e->sfill || e->flat3d || e->superseded || e->overwritten || e->retiring || e->pub.gone || e->pub.skip ||
        e->f.depth || e->f.stencil || e->pub.samples != 1 || e->pub.type != VK_IMAGE_TYPE_2D ||
        !e->pub.levels || e->pub.levels > MATERIAL_LEVELS || !e->pub.layers ||
        !e->t.L.size || e->t.L.size > MATERIAL_SNAPSHOT_LIMIT ||
        !(e->pub.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) || !(e->pub.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) return 0;
    const Tex *head = nvn_hm_get(&by_addr, &e->key.addr);
    if (head && (head != e || e->next_at)) return 0;
    return 1;
}
static int content_defer_instance_mode(TextureMaterialInstance *instance, int completed) {
    if (!texture_material_instance_defer_ready(instance)) return 0;
    content_use_image_remove(instance);
    TextureMaterialResidency old = texture_material_instance_defer_begin(instance);
    // Associate the retired image's owned views with that physical image before reclaiming it.
    VkMemoryImageHold *views = completed ? vk_mem_image_hold(old.image) : NULL;
    vk_exec_image_gone(old.image);
    if (views) vk_mem_image_release(views);
    if (completed && vk_mem_image_destroy_completed(old.image, old.alloc, old.last_use_fence,
            instance->upload_sequence, texture_material_destroyed, instance)) return 1;
    vk_defer_image_notify(old.image, old.alloc, instance->material->reclaim ? old.bytes : 0,
                           texture_material_destroyed, instance);
    return 0;
}

static void content_defer_instance(TextureMaterialInstance *instance) {
    content_defer_instance_mode(instance, 0);
}
// Bound visited nodes, including those still held by recording/submitted consumers. Actual destruction callbacks
// only set a leaf flag; metadata retirement and list updates happen here with the texture lock held.
static void content_collect(uint64_t limit) {
    uint64_t visits = material_ledger.pending_instances;
    if (visits > limit) visits = limit;
    for (uint64_t i = 0; i < visits; ++i) {
        TextureMaterialInstance *instance = texture_material_pending_next(&material_ledger);
        if (!instance) break;
        content_defer_instance(instance);
        TextureMaterial *owner = texture_material_instance_reap(instance);
        if (owner) content_material_free(owner);
    }
}
static uint64_t content_retire(TextureMaterial *m) {
    TextureMaterialResidency old = texture_material_retire(m);
    if (old.instance) content_defer_instance(old.instance);
    material_stats.retired_bytes += old.bytes;
    if (texture_material_can_free(m)) content_material_free(m);
    return old.bytes;
}
static uint64_t content_unpin(unsigned at, int reclaim) {
    MaterialSlot *s = &material_slots[at];
    TextureMaterial *m = s->material;
    if (!m) return 0;
    s->material = NULL;
    assert(m->index_slot == at + 1 && m->cached);
    m->index_slot = 0;
    m->reclaim |= reclaim != 0;
    return texture_material_cache(m, 0) ? content_retire(m) : 0;
}
static uint64_t content_release(Tex *e, int reclaim) {
    TextureMaterial *m = e->material;
    if (!m) return 0;
    if (reclaim && m->refs == 1 && m->cached) content_unpin(m->index_slot - 1, 1);
    m->reclaim |= reclaim != 0;
    content_disassociate(e);
    image_owner_replaced(e, VK_NULL_HANDLE);
    e->pub.image = VK_NULL_HANDLE;
    e->alloc = NULL;
    e->bytes = 0;
    int retire = texture_material_release(m, !e->released);
    // A nonresident cache-only provider has no future exact obligation. Do not retain raw captures outside the
    // bounded resident victim cursor after their final logical association disappears.
    if (!m->refs && !m->resident.image && m->cached) return content_unpin(m->index_slot - 1, reclaim);
    return retire ? content_retire(m) : 0;
}
// Visit physical owners once, including live logical associations when measured physical pressure requires it.
// The immutable provider preserves future finalization and exact reacquisition after residency is detached.
static int content_residency_reclaimable(const TextureMaterial *m, int live) {
    if (!m->refs && m->cached) return 1;
    if (!m->provider || !m->ready || (!live && m->live_refs) || m->resident.last_use_epoch == submit_epoch) return 0;
    for (const Tex *e = m->associations; e; e = e->material_next) {
        // This is the currently selected pool association, including a released texture selected by a finalized
        // slot. Old descriptor publications have independent captures and do not create PRef demand.
        if (e->ph > 0 && e->ph <= npref && pref[e->ph - 1].e == e && pref[e->ph - 1].refs) return 0;
        if (!e->released && !content_image_eligible(e)) return 0;
    }
    return 1;
}
static uint64_t content_detach_residency_mode(TextureMaterial *m, int *destroyed) {
    TextureMaterialResidency old = texture_material_take_residency(m);
    if (!old.image) return 0;
    m->reclaim = 1;
    for (Tex *e = m->associations; e; e = e->material_next) {
        image_owner_replaced(e, VK_NULL_HANDLE);
        e->pub.image = VK_NULL_HANDLE;
        e->uploaded = 0; e->checked = 0; e->checked_authority = 0;
        e->material_version = (RcRegistryReadVersion){0};
        e->material_prepared = 0; e->chash_ok = 0;
        e->upload_mapping = e->upload_protection = 0;
        tex_gen_bump(e->key.addr);
    }
    const int complete = content_defer_instance_mode(old.instance, destroyed != NULL);
    if (destroyed) *destroyed = complete;
    material_stats.evicted++;
    material_stats.evicted_bytes += old.bytes;
    return old.bytes;
}

static uint64_t content_detach_residency(TextureMaterial *m) {
    return content_detach_residency_mode(m, NULL);
}

static unsigned content_reclaim_completed(uint64_t wanted, int live, uint32_t domain,
                                          uint64_t scene_done, uint64_t upload_done) {
    unsigned outcome = RC_BUDGET_RECLAIM_NONE;
    uint64_t selected = 0;
    uint64_t visits = material_ledger.resident_owners;
    if (visits > MATERIAL_RECLAIM_SCAN) visits = MATERIAL_RECLAIM_SCAN;
    for (uint64_t n = 0; n < visits && selected < wanted; ++n) {
        TextureMaterial *m = texture_material_resident_next(&material_ledger);
        if (!m) break;
        const TextureMaterialResidency *resident = &m->resident;
        const TextureMaterialInstance *instance = resident->instance;
        if (!instance || instance->lifetime.captures || vk_mem_image_held(resident->image) ||
            (!resident->last_use_fence && !instance->upload_sequence) ||
            resident->last_use_fence > scene_done || instance->upload_sequence > upload_done ||
            !content_residency_reclaimable(m, live) || !texture_reclaim_on_heap(resident->alloc, domain)) continue;
        int destroyed = 0;
        selected += content_detach_residency_mode(m, &destroyed);
        if (!m->refs && m->cached) content_unpin(m->index_slot - 1, 1);
        outcome |= destroyed ? RC_BUDGET_RECLAIM_DESTROYED : RC_BUDGET_RECLAIM_PENDING;
    }
    return outcome;
}

static uint64_t content_reclaim_in(uint64_t wanted, int live, uint32_t domain) {
    uint64_t freed = 0;
    uint64_t visits = material_ledger.resident_owners;
    if (visits > MATERIAL_RECLAIM_SCAN) visits = MATERIAL_RECLAIM_SCAN;
    for (uint64_t n = 0; n < visits && freed < wanted; n++) {
        TextureMaterial *m = texture_material_resident_next(&material_ledger);
        if (!m) break;
        if (!content_residency_reclaimable(m, live) || !texture_reclaim_on_heap(m->resident.alloc, domain)) continue;
        if (!m->refs && m->cached) { freed += content_unpin(m->index_slot - 1, 1); continue; }
        freed += content_detach_residency(m);
    }
    return freed;
}
static uint64_t content_reclaim(uint64_t wanted, int live) { return content_reclaim_in(wanted, live, 0); }
static void content_shutdown(void) {
    for (unsigned i = 0; i < MATERIAL_SLOTS; i++) content_unpin(i, 0);
    memset(material_certificates, 0, sizeof material_certificates);
    content_collect(UINT64_MAX);
}
static int content_slot_matches(const MaterialSlot *s, const Tex *e, const uint8_t digest[32]) {
    TexKey key = content_image_key(e);
    return s->material && s->material->ready && !s->material->retired && s->source_bytes == e->t.L.size &&
        !memcmp(&s->source, &key, sizeof key) && !memcmp(&s->intent, &e->material_intent, sizeof s->intent) &&
        (!digest || !memcmp(s->digest, digest, 32));
}
static void content_certificate(Tex *e, unsigned slot) {
    MaterialSlot *s = &material_slots[slot];
    MaterialCertificate *c = &material_certificates[content_source_bucket(e->key.addr)];
    *c = (MaterialCertificate){.source = e->key, .intent = e->material_intent,
        .stamp = e->source_stamp, .material_id = s->material->id, .slot = slot + 1};
}
static void content_version(Tex *e, TextureMaterial *m) {
    TextureSourceAuthority origin = {.ordered = e->material_version.origin};
    TextureSourceAuthority content = rc_registry_source_access_content(source_access_anchor(e));
    if (rc_input_exact(&m->input) && texture_source_stamp_valid(&e->source_stamp) &&
        e->material_version.id && rc_input_equal(&e->material_version.input, &m->input) &&
        texture_authority_same(&origin, &content)) return;
    e->material_version = (RcRegistryReadVersion){0};
    RcRegistryReadObservation permission;
    if (rc_input_exact(&m->input) && texture_source_stamp_valid(&e->source_stamp) &&
        source_permission_at(e, vk_exec_resource_position(), &permission))
        rc_registry_read_commit(rc_registry_runtime(), &permission, &m->input, &e->material_version);
}
static void content_adopt(Tex *e, unsigned at, const TextureSourceStamp *stamp) {
    const MaterialSlot *s = &material_slots[at];
    TextureMaterial *m = s->material;
    TextureMaterialResidency resident = texture_material_capture(m);
    assert(resident.image);
    if (e->material == m && e->pub.image == resident.image && e->uploaded && !e->up_retry) {
        // A newer source proof for identical contents does not replace the image or change any binding. Preserve
        // descriptor and derived-view generations; only the evidence about the guest source advances.
        source_commit(e, stamp, 1);
        content_version(e, m);
        memcpy(e->source_digest, s->digest, 32);
        e->source_digest_valid = 1;
        texture_material_note_use(m, vk_exec_recording_fence(), submit_epoch);
        content_certificate(e, at);
        material_stats.hits++;
        material_stats.early_hits++;
        material_stats.same_owner_hits++;
        return;
    }
    texture_material_acquire(m, !e->released);
    if (e->material) content_release(e, 0);
    else if (e->pub.image) {
        if (e->released) rel_bytes -= e->bytes;
        vk_exec_image_gone(e->pub.image);
        vk_defer_image(e->pub.image, e->alloc);
    }
    uint8_t learned = e->pub.fb_learned;
    image_owner_replaced(e, resident.image);
    e->pub = s->representation;
    e->pub.image = resident.image;
    e->pub.addr = e->key.addr;
    e->pub.fb_learned = learned;
    e->pub.gone = e->pub.gpu_owned = e->pub.bound = 0;
    content_associate(e, m);
    e->alloc = NULL;
    e->bytes = 0;
    e->ik = m->key;
    e->f = s->format;
    e->bc1_try = 0;
    e->uploaded = 1;
    e->up_retry = 0;
    e->n_up++;
    e->chash_ok = 0;
    e->upload_mapping = e->upload_protection = 0;
    e->material_prepared = 1;
    e->reusable = e->reused = 0;
    source_commit(e, stamp, 0);
    content_version(e, m);
    memcpy(e->source_digest, s->digest, 32);
    e->source_digest_valid = 1;
    texture_material_note_use(m, vk_exec_recording_fence(), submit_epoch);
    content_certificate(e, at);
    tex_gen_bump(e->key.addr);
    material_stats.hits++;
    material_stats.early_hits++;
    for (uint32_t lv = 0; lv < e->pub.levels; lv++)
        material_stats.avoided_texels += (uint64_t)mip(e->pub.width, lv) * mip(e->pub.height, lv) * e->pub.layers;
}
// A hit needs no allocation, staging, conversion or copy. On a miss, upload must consume source_copy exactly.
static int content_early_prepare_policy(Tex *e, int policy) {
    content_source_discard(e);
    if (!content_image_eligible(e) || !source_spans_init(e)) return 0;
    e->source_retry_domains = 0;
    e->source_retry_source = 0;
    material_stats.lookups++;
    TextureSourceObservation obs;
    // An association retains its own provider even after a bounded index collision displaced its lookup entry.
    if (e->material && !e->pub.image && e->material->provider &&
        texture_source_stamp_valid(&e->source_stamp) &&
        source_observe(e, &e->source_stamp, &obs) == TEX_SOURCE_CLEAN) {
        if (policy == MATERIAL_PREPARE_OPTIONAL) {
            content_recovery_wait(e, content_recovery_domains(e));
            return 0;
        }
        if (!content_restore(e, e->material, &obs.stamp)) return 0;
        material_stats.certificate_hits++;
        return 1;
    }
    MaterialCertificate *c = &material_certificates[content_source_bucket(e->key.addr)];
    if (c->slot && !memcmp(&c->source, &e->key, sizeof e->key) &&
        !memcmp(&c->intent, &e->material_intent, sizeof c->intent)) {
        unsigned at = c->slot - 1;
        MaterialSlot *s = &material_slots[at];
        if (content_slot_matches(s, e, NULL) && s->material->id == c->material_id &&
            (policy != MATERIAL_PREPARE_OPTIONAL || s->material->provider) &&
            source_observe(e, &c->stamp, &obs) == TEX_SOURCE_CLEAN) {
            if (!s->material->resident.image) {
                if (policy == MATERIAL_PREPARE_OPTIONAL) {
                    content_recovery_wait(e, content_recovery_domains(e));
                    return 0;
                }
                if (!s->material->provider || !content_restore(e, s->material, &obs.stamp)) return 0;
            } else content_adopt(e, at, &obs.stamp);
            material_stats.certificate_hits++;
            return 1;
        }
    }
    TextureSourceStamp empty = {0};
    source_observe(e, &empty, &obs); // No prior stamp is expected to be unknown; its new observation can still be valid.
    if (!texture_source_stamp_valid(&obs.stamp)) {
        content_recovery_wait_source(e, &obs.stamp, &e->source_authority);
        return 0;
    }
    TextureSourceAuthority captured_authority = e->source_authority;
    uint64_t size = e->t.L.size;
    RcAdmissionVector cost = {{0}};
    cost.amount[RC_BUDGET_RAM] = cost.amount[RC_BUDGET_COMMIT] = size;
    // Qualified uploads must retain the exact captured bytes. Background pool refresh and optional cache
    // promotion still use discretionary headroom, as does compatibility capture with a valid native fallback.
    int required = policy == MATERIAL_PREPARE_REQUIRED && e->source_canonical && !pool_ctx;
    RcBudgetTicket capacity = {0};
    if (resource_core_budget_reserve(RC_BUDGET_SNAPSHOT, required, &cost, &capacity) != RC_ADMISSION_OK &&
        (!content_drop_providers(size) ||
         resource_core_budget_reserve(RC_BUDGET_SNAPSHOT, required, &cost, &capacity) != RC_ADMISSION_OK)) {
        source_note_refusal(e, SOURCE_REFUSAL_SNAPSHOT_CAPACITY, RC_REGISTRY_CAPACITY);
        material_stats.snapshot_refused++;
        content_recovery_wait(e, (1u << RC_BUDGET_RAM) | (1u << RC_BUDGET_COMMIT));
        return 0;
    }
    uint8_t *copy = malloc((size_t)size);
    if (!copy) {
        source_note_refusal(e, SOURCE_REFUSAL_SNAPSHOT_CAPACITY, RC_REGISTRY_CAPACITY);
        resource_core_budget_cancel(&capacity);
        content_recovery_wait(e, (1u << RC_BUDGET_RAM) | (1u << RC_BUDGET_COMMIT));
        return 0;
    }
    if (!source_snapshot_hold(e)) {
        source_note_refusal(e, SOURCE_REFUSAL_SNAPSHOT_MAPPING, RC_REGISTRY_STALE);
        free(copy);
        resource_core_budget_cancel(&capacity);
        content_recovery_wait_source(e, &obs.stamp, &captured_authority);
        return 0;
    }
    // Keep both host allowances reserved through the copy so a concurrent sample cannot count an
    // untouched malloc as materialized and forget the working-set growth.
    source_snapshot_copy(e, copy, (size_t)size);
    mem_read_release();
    if (!rc_admission_accounted(resource_core_budget_materialize(&capacity, &cost)))
        rt_fatal("vk: snapshot capacity settlement failed");
    // Later writes remain dirty. Both the digest and upload consume this snapshot, never a second guest read.
    if (!source_snapshot_verify(e, &obs.stamp, &captured_authority)) {
        source_note_refusal(e, SOURCE_REFUSAL_SNAPSHOT_CHANGED, RC_REGISTRY_STALE);
        free(copy);
        resource_core_budget_retire(&capacity);
        resource_core_budget_destroyed(&capacity);
        content_recovery_wait_source(e, &obs.stamp, &captured_authority);
        return 0;
    }
    RcRegistryReadVersion captured_version = {0};
    if (rc_registry_position_valid(captured_authority.ordered.position)) {
        if (source_capture_next == UINT64_MAX) rt_fatal("vk: source capture identity exhausted");
        uint64_t id = ++source_capture_next;
        RcPreparedInput input = {id, id, 1, id, RC_PROVIDER_CAPTURE};
        if (rc_registry_read_commit(rc_registry_runtime(), &captured_authority.ordered, &input,
                                    &captured_version) != RC_REGISTRY_OK) {
            source_note_refusal(e, SOURCE_REFUSAL_SNAPSHOT_COMMIT, RC_REGISTRY_STALE);
            free(copy);
            resource_core_budget_retire(&capacity);
            resource_core_budget_destroyed(&capacity);
            content_recovery_wait_source(e, &obs.stamp, &captured_authority);
            return 0;
        }
    }
    e->source_copy_version = captured_version;
    e->source_copy = copy;
    e->source_copy_size = size;
    // Later memory observations may advance their generation without changing these captured bytes. Preserve
    // the paired pre-copy authority/cutoff; never certify this payload with a later observation's generation.
    e->source_copy_stamp = obs.stamp;
    e->source_copy_authority = captured_authority;
    e->source_budget = capacity;
    material_snapshot_bytes += size;
    material_stats.snapshot_bytes += size;
    uint64_t start = kern_now_ns();
    blake3_hasher hash;
    blake3_hasher_init(&hash);
    blake3_hasher_update(&hash, copy, (size_t)size);
    blake3_hasher_finalize(&hash, e->source_digest, 32);
    material_stats.hashes++;
    material_stats.hash_bytes += size;
    material_stats.hash_ns += kern_now_ns() - start;
    e->source_digest_valid = 1;
    e->source_stamp = obs.stamp;
    if (e->material && !e->pub.image && e->material->provider) {
        MaterialSource *p = e->material->provider;
        if (p->size == size && !memcmp(p->digest, e->source_digest, 32) &&
            !memcmp(&p->intent, &e->material_intent, sizeof p->intent)) {
            if (policy == MATERIAL_PREPARE_OPTIONAL) {
                content_source_discard(e);
                content_recovery_wait(e, content_recovery_domains(e));
                return 0;
            }
            return content_restore(e, e->material, &obs.stamp);
        }
    }
    unsigned at = content_image_bucket(e->source_digest);
    if (!content_slot_matches(&material_slots[at], e, e->source_digest)) return 0;
    if (policy == MATERIAL_PREPARE_OPTIONAL && !material_slots[at].material->provider) return 0;
    if (!material_slots[at].material->resident.image && policy == MATERIAL_PREPARE_OPTIONAL) {
        // Optional admission cannot enter the required exact-restoration path and retire a drawable fallback.
        // Keep only a retry advisory; this miss does not retain a second bulk copy of the existing provider.
        content_source_discard(e);
        content_recovery_wait(e, content_recovery_domains(e));
        return 0;
    }
    if (!material_slots[at].material->resident.image)
        return material_slots[at].material->provider && content_restore(e, material_slots[at].material, &obs.stamp);
    content_adopt(e, at, &obs.stamp);
    content_source_discard(e);
    return 1;
}
static int content_early_prepare(Tex *e) { return content_early_prepare_policy(e, MATERIAL_PREPARE_REQUIRED); }
static void content_image_publish_prepared(Tex *e, const uint8_t digest[32], uint64_t bytes,
                                           TextureMaterial **prepared, MaterialSource **provided) {
    if (e->material_restoring) {
        TextureMaterial *m = e->material;
        if (!m || m->resident.image || !e->uploaded || e->up_retry || !e->alloc ||
            memcmp(&m->key, &e->ik, sizeof m->key)) return;
        if (!texture_material_install_residency(m, e->pub.image, e->alloc, e->bytes).image) return;
        texture_material_set_upload(m, vk_upload_dependency());
        content_use_image_add(m->resident.instance);
        e->alloc = NULL; e->bytes = 0; e->material_prepared = 1;
        memcpy(e->source_digest, digest, 32); e->source_digest_valid = 1;
        content_version(e, m);
        texture_material_note_use(m, vk_exec_recording_fence(), submit_epoch);
        tex_gen_bump(e->key.addr);
        material_stats.restored++; material_stats.restored_bytes += bytes;
        return;
    }
    // The snapshot's committed version remains exact even if later commands changed the guest-source proof.
    int exact = e->source_copy && e->source_copy_version.id && rc_input_exact(&e->source_copy_version.input);
    if (!e->uploaded || e->up_retry || bytes != e->t.L.size ||
        (!exact && (!texture_source_stamp_valid(&e->source_stamp) ||
        !source_permission_current(e))) ||
        !content_image_eligible(e) || (!e->material && (!e->alloc || !e->pub.image))) return;
    TextureMaterial *m = e->material;
    if (!m) {
        if (prepared) {
            m = *prepared;
            *prepared = NULL;
            assert(m && m->resident.image == e->pub.image && m->resident.alloc == e->alloc);
        } else {
            m = malloc(sizeof *m);
            if (!m) return;
            if (!texture_material_init(m, &material_ledger, e->pub.image, e->alloc, &e->ik, e->bytes, !e->released)) {
                free(m);
                return;
            }
        }
        texture_material_set_upload(m, vk_upload_dependency());
        if (e->released) rel_bytes -= e->bytes;
        content_associate(e, m);
        e->alloc = NULL;
        e->bytes = 0;
    }
    assert(!m->ready || m->refs == 1);
    if (m->cached) content_unpin(m->index_slot - 1, 0);
    unsigned at = content_image_bucket(digest);
    if (material_slots[at].material) {
        if (memcmp(material_slots[at].digest, digest, 32)) material_stats.collisions++;
        content_unpin(at, 0);
    }
    MaterialSlot *s = &material_slots[at];
    *s = (MaterialSlot){.material = m, .source = content_image_key(e), .intent = e->material_intent,
        .representation = e->pub, .format = e->f, .source_bytes = bytes};
    s->representation.image = VK_NULL_HANDLE; // Representation metadata does not own a resident instance.
    memcpy(s->digest, digest, 32);
    m->ready = 1;
    if (exact) {
        m->input = e->source_copy_version.input;
        e->material_version = e->source_copy_version;
    }
    if (!m->provider && e->source_copy && !e->source_borrowed) {
        MaterialSource *p = provided ? *provided : malloc(sizeof *p);
        if (p) {
            if (provided) *provided = NULL;
            *p = (MaterialSource){.bytes = e->source_copy, .size = e->source_copy_size,
                .budget = e->source_budget, .recipe = e->t, .representation = e->pub, .format = e->f,
                .intent = e->material_intent, .stamp = e->source_copy_stamp, .version = e->source_copy_version};
            p->recipe.mem = p->bytes; p->representation.image = VK_NULL_HANDLE;
            memcpy(p->digest, digest, 32);
            m->provider = p;
            e->source_copy = NULL; e->source_copy_size = 0; e->source_budget = (RcBudgetTicket){0};
            material_ledger.provider_bytes += p->size;
            texture_material_require(rc_lifetime_recovery_set(&m->lifetime, m->lifetime.id,
                                                               RC_LIFETIME_RECOVERY_IMMUTABLE));
        }
    }
    if (!rc_input_exact(&m->input)) {
        if (source_capture_next == UINT64_MAX) rt_fatal("vk: material provider identity exhausted");
        uint64_t id = ++source_capture_next;
        m->input = (RcPreparedInput){id, id, 1, id, RC_PROVIDER_CAPTURE};
    }
    content_use_image_add(m->resident.instance);
    m->index_slot = at + 1;
    texture_material_cache(m, 1);
    texture_material_note_use(m, vk_exec_recording_fence(), submit_epoch);
    memcpy(e->source_digest, digest, 32);
    e->source_digest_valid = 1;
    content_certificate(e, at);
}

static void content_image_publish(Tex *e, const uint8_t digest[32], uint64_t bytes) {
    content_image_publish_prepared(e, digest, bytes, NULL, NULL);
}
// Reinstall this exact representation from its owned bytes. Source revalidation belongs to the current-read caller;
// a finalized association passes no stamp and never consults reused or unmapped guest storage.
static int content_adopt_residency(Tex *e, TextureMaterial *m, const TextureSourceStamp *cutoff) {
    if (e->material != m || !m->resident.image || !m->provider) return 0;
    const MaterialSource *p = m->provider;
    image_owner_replaced(e, m->resident.image);
    e->pub.image = m->resident.image; e->uploaded = 1; e->up_retry = 0; e->material_prepared = 1;
    source_commit(e, cutoff, 0); content_version(e, m);
    content_source_discard(e);
    memcpy(e->source_digest, p->digest, 32); e->source_digest_valid = 1;
    texture_material_note_use(m, vk_exec_recording_fence(), submit_epoch);
    if (m->cached) content_certificate(e, m->index_slot - 1);
    tex_gen_bump(e->key.addr);
    return 1;
}

static int content_restore(Tex *e, TextureMaterial *m, const TextureSourceStamp *stamp) {
    MaterialSource *p = m->provider;
    if (!p || m->retired || !m->ready) return 0;
    TextureSourceStamp cutoff = stamp ? *stamp : (TextureSourceStamp){0};
    TextureSourceAuthority authority = stamp ? (e->source_copy ? e->source_copy_authority : e->source_authority)
                                             : (TextureSourceAuthority){0};
    // Another association may already have reconstructed this material after our association was invalidated.
    if (m->resident.image) return content_adopt_residency(e, m, &cutoff);
    const ImgKey *k = &m->key;
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = k->flags,
        .imageType = (VkImageType)k->type, .format = (VkFormat)k->format, .extent = {k->w, k->h, k->d},
        .mipLevels = k->levels, .arrayLayers = k->layers, .samples = (VkSampleCountFlagBits)k->samples,
        .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = k->usage, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocation alloc = NULL;
    VkImage image = suballoc_image(vk_ctx(), &ci, &alloc);
    if (!image && vma_image(vk_ctx(), &ci, k, &image, &alloc) != VK_SUCCESS) {
        tex_deferred = 1;
        return 0;
    }
    VmaAllocationInfo own;
    vmaGetAllocationInfo(vk_ctx()->vma, alloc, &own);
    if (e->material != m) {
        texture_material_acquire(m, !e->released);
        if (e->material) content_release(e, 0);
        else if (e->pub.image) {
            if (e->released) rel_bytes -= e->bytes;
            vk_exec_image_gone(e->pub.image);
            vk_defer_image(e->pub.image, e->alloc);
        }
        content_associate(e, m);
    }
    content_source_discard(e); // A same-digest candidate contributes no second retained raw allocation.
    uint8_t learned = e->pub.fb_learned;
    image_owner_replaced(e, image);
    e->pub = p->representation;
    e->pub.image = image; e->pub.addr = e->key.addr; e->pub.fb_learned = learned;
    e->pub.gone = e->pub.gpu_owned = e->pub.bound = 0;
    e->f = p->format; e->ik = m->key; e->alloc = alloc; e->bytes = own.size;
    e->uploaded = 0; e->up_retry = 0; e->reused = e->reusable = 0; e->bc1_try = 0;
    init_image(e, vk_upload_begin());
    vk_upload_end();
    // Image initialization invalidates the old physical contents. Establish the provider's reconstruction input
    // afterward, so upload can publish the completed instance instead of discarding it as an uncertified copy.
    e->source_copy = p->bytes; e->source_copy_size = p->size; e->source_recipe = &p->recipe;
    e->source_borrowed = 1; e->material_restoring = 1;
    e->source_copy_stamp = e->source_stamp = cutoff;
    e->source_copy_authority = authority; e->source_copy_version = p->version;
    memcpy(e->source_digest, p->digest, 32); e->source_digest_valid = 1;
    content_upload_reconstructed(e);
    e->material_restoring = 0;
    content_source_discard(e);
    if (!m->resident.image) {
        // A failed decode/upload never exposes a partially reconstructed physical instance. The exact provider and
        // logical association remain available for retry; backend deferral protects any commands already recorded.
        if (e->pub.image) {
            vk_exec_image_gone(e->pub.image);
            vk_defer_image(e->pub.image, e->alloc);
        }
        image_owner_replaced(e, VK_NULL_HANDLE);
        e->pub.image = VK_NULL_HANDLE; e->alloc = NULL; e->bytes = 0; e->uploaded = 0;
        tex_deferred = 1;
        return 0;
    }
    if (m->cached) content_certificate(e, m->index_slot - 1);
    return 1;
}
// Allocate and preserve before changing the logical association. Ready contents remain immutable even with one
// logical user: recorded draws and the index can still need their previous version.
// CPU uploads can be submitted before unhanded graphics parts. A second upload to an actively
// held private version would otherwise change their bytes even though VkImage lifetime survived.
static int retained_texture_upload_split(Tex *entry, int preserve) {
    const ImgKey *key = &entry->ik;
    const VkImageCreateInfo create = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = key->flags,
        .imageType = (VkImageType)key->type, .format = (VkFormat)key->format, .extent = {key->w, key->h, key->d},
        .mipLevels = key->levels, .arrayLayers = key->layers, .samples = (VkSampleCountFlagBits)key->samples,
        .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = key->usage, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VkCtx *ctx = vk_ctx();
    const VkImage previous = entry->pub.image;
    const VmaAllocation previous_allocation = entry->alloc;
    const VkDeviceSize previous_size = entry->bytes;
    RetainedTextureUpload *owner = entry->retained_sample_upload;
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = NULL;
    if (vma_image(ctx, &create, key, &image, &allocation) != VK_SUCCESS) {
        tex_deferred = 1;
        return 0;
    }
    VmaAllocationInfo info;
    vmaGetAllocationInfo(ctx->vma, allocation, &info);
    if (entry->pub.gone || entry->pub.image != previous || entry->alloc != previous_allocation ||
        entry->retained_sample_upload != owner) {
        vk_defer_image(image, allocation);
        tex_deferred = 1;
        return 0;
    }
    Tex fresh = {.pub = entry->pub, .f = entry->f, .alloc = allocation};
    fresh.pub.image = image;
    vk_exec_image_gone(image);
    VkCommandBuffer command = vk_upload_begin();
    const uint64_t sequence = vk_upload_recording_sequence(command);
    init_image(&fresh, command);
    if (preserve) {
        VkImageCopy copies[MATERIAL_LEVELS];
        for (uint32_t i = 0; i < key->levels; i++) {
            VkImageSubresourceLayers layer = {entry->pub.aspect, i, 0, key->layers};
            copies[i] = (VkImageCopy){.srcSubresource = layer, .dstSubresource = layer,
                .extent = {mip(key->w, i), mip(key->h, i), mip(key->d, i)}};
        }
        barrier(command, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
        vkCmdCopyImage(command, previous, VK_IMAGE_LAYOUT_GENERAL, image, VK_IMAGE_LAYOUT_GENERAL, key->levels, copies);
        barrier(command, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    }
    vk_upload_end();
    const int retained = image_owner_replaced(entry, image);
    entry->pub.image = image;
    entry->alloc = allocation;
    entry->bytes = info.size;
    entry->reused = 0;
    entry->sample_upload_sequence = preserve ? sequence : 0;
    if (!preserve) entry->uploaded = 0;
    compatibility_invalidate(entry);
    sample_general_record(entry, (RcRegistryPosition){0}, sequence);
    tex_gen_bump(entry->pub.addr);
    if (!retained) {
        vk_exec_image_gone(previous);
        vk_defer_image_reclaim(previous, previous_allocation, previous_size);
    }
    return 1;
}

static int content_write_begin(Tex *e, int preserve) {
    if (e->material_restoring) return 1;
    if (!e->material && e->retained_sample_upload) return retained_texture_upload_split(e, preserve);
    TextureMaterial *old = e->material;
    if (!old || (!old->ready && texture_material_exclusive(old))) return 1;
    if (preserve && !old->resident.image && !content_restore(e, old, NULL)) {
        tex_deferred = 1;
        return 0;
    }
    TextureMaterialUse source_use = {0};
    TextureMaterialResidency source = {0};
    if (preserve) {
        if (texture_material_use_begin(old, &source_use) != RC_LIFETIME_OK) {
            tex_deferred = 1;
            return 0;
        }
        // Admission may detach this residency and clear the association's current image. Retain the exact
        // instance, rather than relying on a mutable resident slot or on the last-use epoch's eviction heuristic.
        source = source_use.instance->value;
    }
    const ImgKey *k = &old->key;
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = k->flags,
        .imageType = (VkImageType)k->type, .format = (VkFormat)k->format, .extent = {k->w, k->h, k->d},
        .mipLevels = k->levels, .arrayLayers = k->layers, .samples = (VkSampleCountFlagBits)k->samples,
        .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = k->usage, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VkCtx *c = vk_ctx();
    VmaAllocation alloc = NULL;
    VkImage image = suballoc_image(c, &ci, &alloc);
    if (!image && vma_image(c, &ci, k, &image, &alloc) != VK_SUCCESS) {
        content_reclaim(preserve ? source.bytes : old->resident.bytes, 1);
        if (vma_image(c, &ci, k, &image, &alloc) != VK_SUCCESS) {
            if (source_use.instance) {
                texture_material_require(texture_material_use_cancel(&source_use));
                content_defer_instance(source.instance);
            }
            rt_fatal("vk: cannot allocate private texture contents for a required write (%ux%u)", k->w, k->h);
            return 0;
        }
    }
    VmaAllocationInfo own;
    vmaGetAllocationInfo(c->vma, alloc, &own);
    Tex fresh = *e;
    fresh.material = NULL;
    fresh.pub.image = image;
    fresh.alloc = alloc;
    fresh.bytes = own.size;
    fresh.reused = 0;
    vk_exec_image_gone(image);
    VkCommandBuffer cb = vk_upload_begin();
    init_image(&fresh, cb);
    if (preserve) {
        VkImageCopy copies[MATERIAL_LEVELS];
        for (uint32_t lv = 0; lv < k->levels; lv++) {
            VkImageSubresourceLayers sub = {e->pub.aspect, lv, 0, k->layers};
            copies[lv] = (VkImageCopy){.srcSubresource = sub, .dstSubresource = sub,
                .extent = {mip(k->w, lv), mip(k->h, lv), mip(k->d, lv)}};
        }
        barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
        vkCmdCopyImage(cb, source.image, VK_IMAGE_LAYOUT_GENERAL, image, VK_IMAGE_LAYOUT_GENERAL, k->levels, copies);
        barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
        material_stats.cow_bytes += source.bytes;
    }
    vk_upload_end();
    if (source_use.instance) {
        // The recording hold ends only after the copy is complete on the CPU. Backend deferral now captures its
        // recorded upload sequence, so a detached source stays allocated until that upload actually completes.
        texture_material_require(texture_material_use_cancel(&source_use));
        content_defer_instance(source.instance);
    }
    content_image_forget(e);
    content_release(e, 0);
    image_owner_replaced(e, image);
    e->pub.image = image;
    e->alloc = alloc;
    e->bytes = own.size;
    e->reused = 0;
    e->material_prepared = 0;
    if (preserve) { e->uploaded = 1; e->up_retry = 0; }
    if (e->released) rel_bytes += own.size;
    tex_gen_bump(e->key.addr);
    material_stats.cow++;
    return 1;
}

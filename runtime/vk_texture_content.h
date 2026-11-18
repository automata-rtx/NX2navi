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
static uint64_t material_snapshot_bytes;
static uint32_t material_reclaim_cursor;
static struct {
    uint64_t hashes, hash_bytes, hash_ns, lookups, hits, copy_ns, copied_bytes, avoided_texels, collisions;
    uint64_t early_hits, certificate_hits, snapshot_bytes, snapshot_refused, cow, cow_bytes, retired_bytes;
} material_stats;

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
    if (e->source_copy) {
        assert(material_snapshot_bytes >= e->source_copy_size);
        material_snapshot_bytes -= e->source_copy_size;
        free(e->source_copy);
    }
    e->source_copy = NULL;
    e->source_copy_size = 0;
}
// Logical invalidation preserves immutable bytes that other associations or the content index still own.
static void content_image_forget(Tex *e) {
    MaterialCertificate *c = &material_certificates[content_source_bucket(e->key.addr)];
    if (c->slot && !memcmp(&c->source, &e->key, sizeof e->key)) c->slot = 0;
    e->source_digest_valid = 0;
    memset(&e->source_stamp, 0, sizeof e->source_stamp);
}
static int content_image_eligible(const Tex *e) {
    if (!e->pub.mirrored || !e->t.mem || nvn_gpu_only(e->key.addr) || e->material_writable ||
        (e->t.pool_flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL) || e->pub.gpu_owned || e->gpu_dirty || e->bound_to ||
        e->sfill || e->flat3d || e->superseded || e->overwritten || e->retiring || e->pub.gone || e->pub.skip ||
        e->f.depth || e->f.stencil || e->pub.samples != 1 || e->pub.type != VK_IMAGE_TYPE_2D ||
        !e->pub.levels || e->pub.levels > MATERIAL_LEVELS || !e->pub.layers ||
        !e->t.L.size || e->t.L.size > MATERIAL_SNAPSHOT_LIMIT ||
        !(e->pub.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) || !(e->pub.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) return 0;
    const Tex *head = nvn_hm_get(&by_addr, &e->key.addr);
    if (head && (head != e || e->next_at)) return 0;
    return !content_source_gpu_overlap(e);
}
static uint64_t content_retire(TextureMaterial *m) {
    uint64_t bytes = texture_material_retire(m);
    vk_exec_image_gone(m->image);
    vk_defer_image_reclaim(m->image, m->alloc, m->reclaim ? bytes : 0);
    material_stats.retired_bytes += bytes;
    free(m);
    return bytes;
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
    e->material = NULL;
    e->pub.image = VK_NULL_HANDLE;
    e->alloc = NULL;
    e->bytes = 0;
    return texture_material_release(m, !e->released) ? content_retire(m) : 0;
}
// Only an orphan yields physical credit here. Released logical associations use the normal released list.
static uint64_t content_reclaim(uint64_t wanted) {
    uint64_t freed = 0;
    for (unsigned n = 0; n < MATERIAL_RECLAIM_SCAN && freed < wanted; n++) {
        unsigned at = material_reclaim_cursor++ & (MATERIAL_SLOTS - 1);
        TextureMaterial *m = material_slots[at].material;
        if (m && !m->refs) freed += content_unpin(at, 1);
    }
    return freed;
}
static void content_shutdown(void) {
    for (unsigned i = 0; i < MATERIAL_SLOTS; i++) content_unpin(i, 0);
    memset(material_certificates, 0, sizeof material_certificates);
    material_reclaim_cursor = 0;
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
static void content_adopt(Tex *e, unsigned at, const TextureSourceStamp *stamp) {
    const MaterialSlot *s = &material_slots[at];
    TextureMaterial *m = s->material;
    texture_material_acquire(m, !e->released);
    if (e->material) content_release(e, 0);
    else if (e->pub.image) {
        if (e->released) rel_bytes -= e->bytes;
        vk_exec_image_gone(e->pub.image);
        vk_defer_image(e->pub.image, e->alloc);
    }
    uint8_t learned = e->pub.fb_learned;
    e->pub = s->representation;
    e->pub.addr = e->key.addr;
    e->pub.fb_learned = learned;
    e->pub.gone = e->pub.gpu_owned = e->pub.bound = 0;
    e->material = m;
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
    source_commit(e, stamp);
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
static int content_early_prepare(Tex *e) {
    content_source_discard(e);
    if (!content_image_eligible(e) || !source_spans_init(e)) return 0;
    material_stats.lookups++;
    MaterialCertificate *c = &material_certificates[content_source_bucket(e->key.addr)];
    TextureSourceObservation obs;
    if (c->slot && !memcmp(&c->source, &e->key, sizeof e->key) &&
        !memcmp(&c->intent, &e->material_intent, sizeof c->intent)) {
        unsigned at = c->slot - 1;
        MaterialSlot *s = &material_slots[at];
        if (content_slot_matches(s, e, NULL) && s->material->id == c->material_id &&
            source_observe(e, &c->stamp, &obs) == TEX_SOURCE_CLEAN) {
            content_adopt(e, at, &obs.stamp);
            material_stats.certificate_hits++;
            return 1;
        }
    }
    TextureSourceStamp empty = {0};
    source_observe(e, &empty, &obs); // No prior stamp is expected to be unknown; its new observation can still be valid.
    if (!texture_source_stamp_valid(&obs.stamp)) return 0;
    uint64_t size = e->t.L.size;
    if (size > MATERIAL_SNAPSHOT_LIMIT - material_snapshot_bytes) {
        material_stats.snapshot_refused++;
        return 0;
    }
    uint8_t *copy = malloc((size_t)size);
    if (!copy) return 0;
    if (!mem_read_hold(e->key.addr, size)) { free(copy); return 0; }
    memcpy(copy, e->t.mem, (size_t)size);
    mem_read_release();
    // Later writes remain dirty. Both the digest and upload consume this snapshot, never a second guest read.
    TextureSourceObservation after;
    int state = source_observe(e, &obs.stamp, &after);
    if (state == TEX_SOURCE_UNKNOWN || after.stamp.mapping != obs.stamp.mapping ||
        after.stamp.lineage != obs.stamp.lineage || after.stamp.protection != obs.stamp.protection ||
        !content_image_eligible(e)) { free(copy); return 0; }
    e->source_copy = copy;
    e->source_copy_size = size;
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
    unsigned at = content_image_bucket(e->source_digest);
    if (!content_slot_matches(&material_slots[at], e, e->source_digest)) return 0;
    content_adopt(e, at, &obs.stamp);
    content_source_discard(e);
    return 1;
}
static void content_image_publish(Tex *e, const uint8_t digest[32], uint64_t bytes) {
    if (!e->uploaded || e->up_retry || bytes != e->t.L.size || !texture_source_stamp_valid(&e->source_stamp) ||
        !content_image_eligible(e) || (!e->material && (!e->alloc || !e->pub.image))) return;
    TextureMaterial *m = e->material;
    if (!m) {
        m = malloc(sizeof *m);
        if (!m) return;
        texture_material_init(m, &material_ledger, e->pub.image, e->alloc, &e->ik, e->bytes, !e->released);
        if (e->released) rel_bytes -= e->bytes;
        e->material = m;
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
    memcpy(s->digest, digest, 32);
    m->ready = 1;
    m->index_slot = at + 1;
    texture_material_cache(m, 1);
    texture_material_note_use(m, vk_exec_recording_fence(), submit_epoch);
    memcpy(e->source_digest, digest, 32);
    e->source_digest_valid = 1;
    content_certificate(e, at);
}
// Allocate and preserve before changing the logical association. Ready contents remain immutable even with one
// logical user: recorded draws and the index can still need their previous version.
static void content_write_begin(Tex *e, int preserve) {
    TextureMaterial *old = e->material;
    if (!old || (!old->ready && texture_material_exclusive(old))) return;
    const ImgKey *k = &old->key;
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = k->flags,
        .imageType = (VkImageType)k->type, .format = (VkFormat)k->format, .extent = {k->w, k->h, k->d},
        .mipLevels = k->levels, .arrayLayers = k->layers, .samples = (VkSampleCountFlagBits)k->samples,
        .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = k->usage, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VkCtx *c = vk_ctx();
    VmaAllocation alloc = NULL;
    VkImage image = suballoc_image(c, &ci, &alloc);
    if (!image && vma_image(c, &ci, k, &image, &alloc) != VK_SUCCESS) {
        content_reclaim(old->bytes);
        if (vma_image(c, &ci, k, &image, &alloc) != VK_SUCCESS)
            rt_fatal("vk: cannot allocate private texture contents for a required write (%ux%u)", k->w, k->h);
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
        vkCmdCopyImage(cb, old->image, VK_IMAGE_LAYOUT_GENERAL, image, VK_IMAGE_LAYOUT_GENERAL, k->levels, copies);
        barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
        material_stats.cow_bytes += old->bytes;
    }
    vk_upload_end();
    texture_material_note_use(old, vk_exec_recording_fence(), submit_epoch);
    content_image_forget(e);
    content_release(e, 0);
    e->pub.image = image;
    e->alloc = alloc;
    e->bytes = own.size;
    e->reused = 0;
    e->material_prepared = 0;
    if (e->released) rel_bytes += own.size;
    tex_gen_bump(e->key.addr);
    material_stats.cow++;
}

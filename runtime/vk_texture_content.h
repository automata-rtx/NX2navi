// Included by vk_texture.c. Non-owning index of already materialized, ordinary CPU texture contents.
// Texture locking serializes lookup, publication and every invalidation. The index retains no image allocation;
// eviction and destruction remain Tex's responsibility. A hit copies into a separately owned destination image.
enum { MATERIAL_SLOTS = 8192, MATERIAL_LEVELS = 32 };
typedef struct {
    Tex *owner;
    TexKey source;
    ImgKey image;
    uint64_t bytes;
    uint8_t digest[32];
} MaterialSlot;
static MaterialSlot material_slots[MATERIAL_SLOTS];
static struct {
    uint64_t hashes, hash_bytes, hash_ns, lookups, hits, copy_ns, copied_bytes, avoided_texels, collisions;
} material_stats;

static unsigned content_image_bucket(const uint8_t digest[32]) {
    uint32_t h;
    memcpy(&h, digest, sizeof h);
    return h & (MATERIAL_SLOTS - 1);
}

static void content_image_forget(Tex *e) {
    if (!e->content_slot) return;
    MaterialSlot *s = &material_slots[e->content_slot - 1];
    if (s->owner == e) s->owner = NULL;
    e->content_slot = 0;
}

static int content_image_eligible(const Tex *e) {
    if (!e->pub.image || !e->alloc || !e->pub.mirrored || e->pub.gpu_owned || e->gpu_dirty || e->bound_to ||
        e->sfill || e->flat3d || e->superseded || e->overwritten || e->retiring || e->pub.gone || e->pub.skip ||
        !e->f.astc_bw || e->f.depth || e->f.stencil || e->pub.samples != 1 || e->pub.type != VK_IMAGE_TYPE_2D ||
        !e->pub.levels || e->pub.levels > MATERIAL_LEVELS || !e->pub.layers ||
        !(e->pub.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) || !(e->pub.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) return 0;
    // An overlapping descriptor can select GPU content through the alias path. Keep that case on normal resolution.
    return nvn_hm_get(&by_addr, &e->key.addr) == e && !e->next_at;
}

static TexKey content_image_key(const Tex *e) {
    TexKey k = e->key;
    k.addr = 0;
    return k;
}

static Tex *content_image_find(const Tex *e, const uint8_t digest[32], uint64_t bytes) {
    material_stats.lookups++;
    MaterialSlot *s = &material_slots[content_image_bucket(digest)];
    if (!s->owner || s->owner == e || s->bytes != bytes || memcmp(s->digest, digest, 32) ||
        memcmp(&s->image, &e->ik, sizeof e->ik)) return NULL;
    TexKey key = content_image_key(e);
    if (memcmp(&s->source, &key, sizeof key)) return NULL;
    Tex *donor = s->owner;
    if (!donor->uploaded || donor->up_retry || !content_image_eligible(donor) ||
        donor->pub.image == e->pub.image || memcmp(&donor->ik, &s->image, sizeof s->image)) return NULL;
    return donor;
}

static void content_image_publish(Tex *e, const uint8_t digest[32], uint64_t bytes) {
    if (!e->uploaded || e->up_retry || !content_image_eligible(e)) return;
    content_image_forget(e);
    unsigned at = content_image_bucket(digest);
    MaterialSlot *s = &material_slots[at];
    if (s->owner) {
        s->owner->content_slot = 0;
        if (memcmp(s->digest, digest, 32)) material_stats.collisions++;
    }
    *s = (MaterialSlot){.owner = e, .source = content_image_key(e), .image = e->ik, .bytes = bytes};
    memcpy(s->digest, digest, 32);
    e->content_slot = at + 1;
}

static int content_image_copy(Tex *e, Tex *donor, VkCommandBuffer cb, uint64_t bytes) {
    if (!cb || !content_image_eligible(e) || !content_image_eligible(donor) ||
        e->pub.image == donor->pub.image || memcmp(&e->ik, &donor->ik, sizeof e->ik)) return 0;
    uint64_t start = kern_now_ns(), texels = 0;
    VkImageCopy regions[MATERIAL_LEVELS];
    for (uint32_t level = 0; level < e->pub.levels; level++) {
        VkImageSubresourceLayers sub = {.aspectMask = e->pub.aspect, .mipLevel = level,
                                        .baseArrayLayer = 0, .layerCount = e->pub.layers};
        VkExtent3D extent = {mip(e->pub.width, level), mip(e->pub.height, level), 1};
        regions[level] = (VkImageCopy){.srcSubresource = sub, .dstSubresource = sub, .extent = extent};
        texels += (uint64_t)extent.width * extent.height * e->pub.layers;
    }
    // The source upload was recorded before publication. Both images stay GENERAL, and both operations use the
    // graphics queue's ordered upload stream; no CPU wait or completion assumption is necessary.
    barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
    vkCmdCopyImage(cb, donor->pub.image, VK_IMAGE_LAYOUT_GENERAL, e->pub.image, VK_IMAGE_LAYOUT_GENERAL,
                   e->pub.levels, regions);
    barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    touch(donor);
    material_stats.hits++;
    material_stats.copy_ns += kern_now_ns() - start;
    material_stats.copied_bytes += bytes;
    material_stats.avoided_texels += texels;
    return 1;
}

// Included by vk_texture.c. A registration predicts only the shape of a future allocation.
// The replay still selects the alias, uploads current bytes and publishes the descriptor.
// Access to the shape table and the renderer-side calls is serialized by the texture lock.
enum { STREAM_SHAPES = 256, STREAM_SLOTS = 8, STREAM_IMAGE_LIMIT = 2 << 20 };
typedef struct {
    nvn_texdesc desc;
    ImgKey key;
    VkImageCreateInfo ci;
} StreamShape;
static struct {
    StreamShape shapes[STREAM_SHAPES];
    PlatMutex lock;
    PlatCond wake;
    PlatThread *thread;
    int stop, disabled;
    uint64_t requests, hits, failures;
    struct {
        StreamShape shape;
        VkImage image;
        VmaAllocation allocation;
        uint64_t requested;
        enum { STREAM_FREE, STREAM_REQUEST, STREAM_WORK, STREAM_READY } state;
    } slots[STREAM_SLOTS];
} stream_images;

static int stream_images_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *v = cfg_get("NX2NAVI_STREAM_IMAGES");
        enabled = v && !strcmp(v, "1");
    }
    return enabled;
}

static StreamShape *stream_shape(const nvn_texdesc *d) {
    uint32_t h = d->width * 31u + d->height * 17u + d->depth * 7u + d->levels;
    h ^= (uint32_t)d->format * 2654435761u;
    return &stream_images.shapes[(h ^ (h >> 16)) % STREAM_SHAPES];
}

static void stream_image_worker(void *arg) {
    VkCtx *c = arg;
    plat_thread_priority(PLAT_PRIO_BELOW_NORMAL);
    plat_mutex_lock(&stream_images.lock);
    while (!stream_images.stop) {
        unsigned i = 0;
        while (i < STREAM_SLOTS && stream_images.slots[i].state != STREAM_REQUEST) i++;
        if (i == STREAM_SLOTS) {
            plat_cond_wait(&stream_images.wake, &stream_images.lock);
            continue;
        }
        typeof(stream_images.slots[0]) *slot = &stream_images.slots[i];
        slot->state = STREAM_WORK;
        VkImageCreateInfo ci = slot->shape.ci;
        VkImage old = slot->image;
        VmaAllocation allocation = slot->allocation;
        slot->image = VK_NULL_HANDLE;
        slot->allocation = NULL;
        plat_mutex_unlock(&stream_images.lock);
        if (old) resource_core_vma_destroy_image(c->vma, old, allocation);
        allocation = NULL;
        // Speculation can only occupy free space in existing device blocks.
        VkImage image = suballoc_image_limit(c, &ci, &allocation, STREAM_IMAGE_LIMIT,
                                              VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT);
        plat_mutex_lock(&stream_images.lock);
        slot->image = image;
        slot->allocation = allocation;
        slot->state = image ? STREAM_READY : STREAM_FREE;
        if (!image) stream_images.failures++;
    }
    plat_mutex_unlock(&stream_images.lock);
}

static void stream_image_request(const tex_t *t) {
    StreamShape *shape = stream_shape(&t->d);
    if (!shape->ci.sType || memcmp(&shape->desc, &t->d, sizeof t->d) || stream_images.disabled) return;
    TexKey key;
    make_key(t, &key);
    if (nvn_hm_get(&by_key, &key)) return;
    if (!stream_images.thread) {
        VkCtx *c = vk_ctx();
        if (!c) return;
        stream_images.thread = plat_thread_new(stream_image_worker, c, "Texture allocation");
        if (!stream_images.thread) { stream_images.disabled = 1; return; }
    }
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    plat_mutex_lock(&stream_images.lock);
    int victim = -1;
    for (int i = 0; i < STREAM_SLOTS; i++) {
        typeof(stream_images.slots[0]) *slot = &stream_images.slots[i];
        if (slot->state != STREAM_FREE && !memcmp(&slot->shape.key, &shape->key, sizeof shape->key)) {
            slot->requested = frame;
            plat_mutex_unlock(&stream_images.lock);
            return;
        }
        if (slot->state == STREAM_FREE ||
            (victim < 0 && slot->state == STREAM_READY && frame - slot->requested > 2)) victim = i;
    }
    if (victim >= 0) {
        stream_images.slots[victim].shape = *shape;
        stream_images.slots[victim].requested = frame;
        stream_images.slots[victim].state = STREAM_REQUEST;
        stream_images.requests++;
        plat_cond_signal(&stream_images.wake);
    }
    plat_mutex_unlock(&stream_images.lock);
}

static VkImage stream_image_take(const nvn_texdesc *d, const VkImageCreateInfo *ci,
                                 const ImgKey *key, VmaAllocation *allocation) {
    // Only shapes made by this exact image path are learned. The key covers every
    // varying creation parameter. Never keep pointers to a caller's extension chain.
    if (ci->pNext || ci->pQueueFamilyIndices || ci->sharingMode != VK_SHARING_MODE_EXCLUSIVE ||
        ci->tiling != VK_IMAGE_TILING_OPTIMAL || ci->initialLayout != VK_IMAGE_LAYOUT_UNDEFINED ||
        ci->extent.width > 2048 || ci->extent.height > 2048 || ci->arrayLayers != 1 ||
        ci->imageType != VK_IMAGE_TYPE_2D) return VK_NULL_HANDLE;
    StreamShape *shape = stream_shape(d);
    *shape = (StreamShape){.desc = *d, .key = *key, .ci = *ci};
    if (!stream_images.thread) return VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    plat_mutex_lock(&stream_images.lock);
    for (int i = 0; i < STREAM_SLOTS; i++) {
        typeof(stream_images.slots[0]) *slot = &stream_images.slots[i];
        if (slot->state != STREAM_READY || memcmp(&slot->shape.key, key, sizeof *key)) continue;
        image = slot->image;
        *allocation = slot->allocation;
        slot->image = VK_NULL_HANDLE;
        slot->allocation = NULL;
        slot->state = STREAM_FREE;
        stream_images.hits++;
        break;
    }
    plat_mutex_unlock(&stream_images.lock);
    return image;
}

static void stream_image_report(uint64_t frame) {
    plat_mutex_lock(&stream_images.lock);
    uint64_t requests = stream_images.requests, hits = stream_images.hits, failures = stream_images.failures;
    plat_mutex_unlock(&stream_images.lock);
    rt_log("stream images: frame %llu: %llu requests, %llu allocations used, %llu unavailable (cumulative)",
           (unsigned long long)frame, (unsigned long long)requests, (unsigned long long)hits, (unsigned long long)failures);
}

static void stream_images_stop(VkCtx *c) {
    if (stream_images.thread) {
        plat_mutex_lock(&stream_images.lock);
        stream_images.stop = 1;
        plat_cond_signal(&stream_images.wake);
        plat_mutex_unlock(&stream_images.lock);
        plat_thread_join(stream_images.thread);
        for (unsigned i = 0; i < STREAM_SLOTS; i++)
            if (stream_images.slots[i].image)
                resource_core_vma_destroy_image(c->vma, stream_images.slots[i].image, stream_images.slots[i].allocation);
        rt_log("stream images: %llu requests, %llu allocations used, %llu unavailable",
               (unsigned long long)stream_images.requests, (unsigned long long)stream_images.hits,
               (unsigned long long)stream_images.failures);
    }
    stream_images.thread = NULL;
    stream_images.stop = stream_images.disabled = 0;
    stream_images.requests = stream_images.hits = stream_images.failures = 0;
    memset(stream_images.slots, 0, sizeof stream_images.slots);
    memset(stream_images.shapes, 0, sizeof stream_images.shapes);
}

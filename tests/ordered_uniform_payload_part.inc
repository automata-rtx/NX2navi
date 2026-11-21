typedef struct TextureUseBatch {
    uint32_t handover;
    TextureUsePartDisposition *part;
} TextureUseBatch;
static TextureUseBatch active_part;
static PlatMutex part_lock;
static unsigned payload_parts, payload_part_drop;
static uint64_t vk_exec_recording_fence(void) { return recorder.fence; }
static void lock(void) { plat_mutex_lock(&part_lock); }
static void unlock(void) { plat_mutex_unlock(&part_lock); }
static TextureUseBatch *texture_use_active(uint64_t fence, int create) {
    assert(fence == recorder.fence && create && cross_scene_fixture && part_lock);
    if (!active_part.part) payload_parts++;
    return &active_part;
}
static void texture_use_part_release(TextureUsePartDisposition *part) {
    if (part && !__atomic_sub_fetch(&part->refs, 1, __ATOMIC_ACQ_REL)) {
        payload_part_drop++;
        free(part);
    }
}
static void finish_fixture_part(int handoff) {
    assert(active_part.part && active_part.handover == TEXTURE_BATCH_RECORDING);
    active_part.handover = handoff ? TEXTURE_BATCH_HANDED : TEXTURE_BATCH_ABANDONED;
    __atomic_store_n(&active_part.part->handover, active_part.handover, __ATOMIC_RELEASE);
    texture_use_part_release(active_part.part);
    active_part = (TextureUseBatch){0};
}

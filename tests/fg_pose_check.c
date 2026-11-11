// Compile with the runtime's Vulkan headers. This exercises the draw-time history without a GPU or the game.
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk_exec.h"
#include "host.h"
#include "platform/platform.h"

#define NVN_STAGE_VS 0
static struct { VkPipeState st; } B;
static struct { uint32_t ubo_max, ubo_align; } LIM = {65536, 16};
static unsigned char uniforms[1024], indices[6] = {0, 0, 1, 0, 2, 0};
static int readable = 1;
static int tracked = 1, live = 1;
static uint32_t generation = 1;
static uint64_t mapping = 1;
static const char *test_exe;

int plat_exe_path(char *path, size_t cap) {
    int n = snprintf(path, cap, "%s", test_exe);
    return n > 0 && (size_t)n < cap ? n : 0;
}

// The test drives one producer and one replay in sequence.
void plat_once(PlatOnce *once, void (*fn)(void *), void *arg) {
    if (!once->state) { fn(arg); once->state = 2; }
}
void plat_mutex_lock(PlatMutex *m) { (void)m; }
void plat_mutex_unlock(PlatMutex *m) { (void)m; }
void plat_mutex_lock_shared(PlatMutex *m) { (void)m; }
void plat_mutex_unlock_shared(PlatMutex *m) { (void)m; }
int mem_read_hold(uint64_t addr, uint64_t n) { return addr == (uintptr_t)indices && n == sizeof indices && readable; }
void mem_read_release(void) {}
uint64_t vk_mem_import_identity(uint64_t addr, uint64_t size) {
    return addr == (uintptr_t)indices && size == sizeof indices ? mapping : 0;
}
int vk_mem_changed(uint64_t addr, uint64_t n, uint32_t since, uint32_t *now) {
    assert(addr == (uintptr_t)indices && n == sizeof indices);
    *now = generation;
    return tracked ? generation != since : -1;
}

void rt_log(const char *msg, ...) { (void)msg; }
uint32_t gfx_shader_inputs(const GfxShader *s, GfxShaderIo io[32]) { (void)s; (void)io; return 1; }
const uint8_t *vk_usnap_map(void) { return uniforms; }
int vk_usnap_find(uint64_t addr, uint64_t n, uint64_t align, VkDescriptorBufferInfo *b) {
    if (!addr || addr >= sizeof uniforms || n > sizeof uniforms - addr || addr % align) return 0;
    b->offset = addr;
    return 1;
}
const uint8_t *vk_usnap_read_source(uint64_t addr, uint64_t n, int *is_live) {
    *is_live = live;
    return mem_read_hold(addr, n) ? indices : NULL;
}
#include "vk_link_pose.inc"

int main(int argc, char **argv) {
    test_exe = argv[0];
    if (argc == 2 && !strcmp(argv[1], "--catalog")) {
        assert(pose_enabled());
        unsigned counts[POSE_ADULT + 1] = {0};
        for (int i = 0; i < pose_count; ++i) ++counts[pose_ranges[i].actor];
        assert(counts[POSE_CHILD] && counts[POSE_ADULT]);
        printf("Loaded %u child and %u adult body index ranges\n", counts[POSE_CHILD], counts[POSE_ADULT]);
        return 0;
    }
    float basis[12] = {1,0,0,10, 0,1,0,20, 0,0,1,30};
    float turned[12] = {0,0,1,0, 0,1,0,0, -1,0,0,0};
    ActorPose before = {.bytes = (uint8_t *)basis, .size = {0,0,0,sizeof basis}, .frame = 10};
    ActorPose after = {.bytes = (uint8_t *)turned, .size = {0,0,0,sizeof turned}, .frame = 11};
    assert(pose_rotation_vote(&after, &before) == 2);
    memcpy(turned, basis, sizeof basis);
    turned[3] += 100.0f;
    assert(pose_rotation_vote(&after, &before) == 0);
    turned[0] = 0.0f;
    assert(pose_rotation_vote(&after, &before) == -1);
    VkImage test_depth = (VkImage)(uintptr_t)1;
    assert(!pose_turn_active(1, test_depth, 6, 1, 2)); // one moving part
    assert(pose_turn_active(2, test_depth, 6, 4, 4));
    assert(pose_turn_active(3, test_depth, 6, 0, 4));
    for (uint64_t frame = 4; frame <= 8; ++frame) assert(pose_turn_active(frame, test_depth, 6, 0, 0));
    assert(!pose_turn_active(9, test_depth, 6, 0, 0));
    assert(pose_turn_active(10, test_depth, 6, 4, 4));
    assert(!pose_turn_active(12, test_depth, 6, 0, 0)); // history gap
    PoseRange *link = &pose_ranges[0], *adult = &pose_ranges[1];
    pose_count = 2;
    pose_catalog_once.state = 2;
    pose_lengths[0] = 1u << sizeof indices;
    link->count = 3;
    link->width = 2;
    link->actor = 1;
    link->hash = 0x37f11cf1d36d7afeull; // FNV-1a of the six synthetic index bytes above
    adult->count = 6;
    adult->actor = POSE_ADULT;
    p_draw_idx d = {.ibuf = (uintptr_t)indices, .itype = NVN_INDEX_TYPE_UNSIGNED_SHORT, .count = 3};
    vk_link_pose_register_buffer(d.ibuf, sizeof indices);
    assert(pose_buffer_slot(d.ibuf, sizeof indices)->hash == link->hash);
    assert(pose_match(&d) == link);
    indices[0] = 1;
    ++generation; // same address, CPU rewrote its contents
    assert(!pose_match(&d));
    indices[0] = 0;
    ++mapping; // pool replacement with a clean write watch
    assert(pose_match(&d) == link);
    mapping = 0; // an unregistered source can still be read, but its identity cannot be reused
    indices[0] = 1;
    assert(!pose_match(&d));
    indices[0] = 0;
    assert(pose_match(&d) == link);
    mapping = 3;
    assert(pose_match(&d) == link);
    indices[0] = 1;
    live = 0; // submission snapshot differs from the registered live range
    assert(!pose_match(&d));
    live = 1;
    tracked = 0;
    assert(!pose_match(&d));
    indices[0] = 0;
    assert(pose_match(&d) == link);
    tracked = 1;
    readable = 0;
    assert(!pose_match(&d));
    VkExec ex = {0};
    B.st.vattr[0] = 1; // location zero reads stream zero
    ex.depth.image = (VkImage)(uintptr_t)64;
    ex.vbuf[0] = (NVNbufferRange){64, 512};
    ex.ubo[0][0] = (p_bind_buf){.addr = 16, .size = 16};
    ex.ubo[0][1] = (p_bind_buf){.addr = 32, .size = 16};
    ex.ubo[0][3] = (p_bind_buf){.addr = 48, .size = 16};
    float dir = 1.0f;
    memcpy(uniforms + 48, &dir, 4);
    const uint32_t banks = (1u << 3) | (1u << 4) | (1u << 6);
    ActorPose *candidate;
    assert(!pose_previous(link, &ex, &d, 10, banks, &candidate));
    uniforms[48] = 127;
    assert(!pose_previous(adult, &ex, &d, 10, banks, &candidate));
    pose_select(10, 16, ex.depth.image);

    ex.ubo[0][3].addr = 80;
    dir = -1.0f;
    memcpy(uniforms + 80, &dir, 4);
    ActorPose *old = pose_previous(link, &ex, &d, 11, banks, &candidate);
    assert(old && old->bytes[32] != 127);
    float old_dir;
    memcpy(&old_dir, old->bytes + 32, 4);
    assert(old_dir == 1.0f);
    old = pose_previous(adult, &ex, &d, 11, banks, &candidate);
    assert(old && old->bytes[32] == 127);
    assert(!pose_previous(adult, &ex, &d, 11, banks, &candidate)); // identical repeated draw
    assert(!candidate->ambiguous);
    uniforms[80] ^= 1;
    assert(!pose_previous(adult, &ex, &d, 11, banks, &candidate)); // different pose at the same address
    pose_select(11, 16, ex.depth.image);
    assert(!adult->history[1].scene);
    assert(!pose_previous(adult, &ex, &d, 12, banks, &candidate));
    old = pose_previous(link, &ex, &d, 12, banks, &candidate);
    assert(old);
    memcpy(&old_dir, old->bytes + 32, 4);
    assert(old_dir == -1.0f);
    pose_select(12, 96, ex.depth.image); // this view was not selected by the scene
    assert(!pose_previous(link, &ex, &d, 13, banks, &candidate));
    pose_select(13, 16, ex.depth.image);
    assert(!pose_previous(link, &ex, &d, 15, banks, &candidate)); // skipped a frame
    pose_select(15, 16, ex.depth.image);
    ex.vbuf[0].address++;
    assert(!pose_previous(link, &ex, &d, 16, banks, &candidate));
    pose_select(16, 16, ex.depth.image);
    ex.ubo[0][3].addr = 4096;
    assert(!pose_previous(link, &ex, &d, 17, banks, &candidate)); // unavailable skeleton
    pose_select(17, 16, ex.depth.image);
    assert(!link->history[1].scene);
    // The main draw, an offscreen camera and a full-size copy of the scene depth all draw Link.
    // The scene may arrive first or last. Only its own queued draw survives the resolve.
    VkImage scene = ex.depth.image, copy = (VkImage)(uintptr_t)128;
    ex.ubo[0][3].addr = 80;
    for (uint64_t frame = 20; frame <= 22; ++frame) {
        ActorPose *main_draw = NULL, *offscreen = NULL, *copy_draw = NULL;
        for (int i = 0; i < 3; ++i) {
            int pass = (i + frame) % 3;
            ex.ubo[0][0].addr = pass == 1 ? 96 : 16;
            ex.depth.image = pass == 2 ? copy : scene;
            ActorPose **slot = pass == 0 ? &main_draw : pass == 1 ? &offscreen : &copy_draw;
            old = pose_previous(link, &ex, &d, frame, banks, slot);
            if (frame > 20 && pass == 0) assert(old && old == link->history[(frame & 1) ^ 1].scene);
        }
        pose_select(frame, 16, scene);
        assert(main_draw->selected && !offscreen->selected && !copy_draw->selected);
        assert(link->history[frame & 1].scene == main_draw);
    }
    pose_shutdown();
    puts("Pose history and index matching passed");
    return 0;
}

// The hooks and their target classification are production functions. Only Vulkan, view creation and replay
// ownership services are mocked; semantic draw IDs make an extra pass boundary observable without requiring it.
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "vk_exec.h"

enum { MEMORY = 1, PIPELINE, DESCRIPTORS, CONSTANTS, DISPATCH, INVALIDATE, DRAW_BASE = 100 };
typedef struct { unsigned event[128], n, dispatches, invalidations; } Ledger;
static Ledger ledger;
static int missing_view, behind, waits, turns, hooks;
static float uniforms[0x940 / sizeof(float)];
static VkTexImage scene, linear;

static void event(unsigned value) { assert(ledger.n < 128); ledger.event[ledger.n++] = value; }
void rt_log(const char *fmt, ...) { (void)fmt; }
int vk_exec_behind(void) { return behind; }
void vk_exec_hooks_wait(void) { waits++; }
void vk_exec_turn(void) { turns++; }
void vk_exec_hook_turn(void) { hooks++; }
void vk_exec_draw_new_cb(void) { ledger.invalidations++; event(INVALIDATE); }
VkCtx *vk_ctx(void) { assert(!"Unexpected device initialization"); return NULL; }
static int init(VkCtx *ctx) { (void)ctx; assert(!"Fixture initializes the device-independent context"); return 0; }

VkImageView vk_exec_target_view(const VkTexImage *img, VkFormat fmt, VkImageViewType type, uint32_t aspect,
                               uint32_t level, uint32_t base, uint32_t layers, int blocks) {
    assert(type == VK_IMAGE_VIEW_TYPE_2D && aspect == VK_IMAGE_ASPECT_COLOR_BIT);
    assert(!level && !base && layers == 1 && !blocks);
    assert((img == &linear && fmt == VK_FORMAT_R32_SFLOAT) ||
           (img == &scene && fmt == VK_FORMAT_B10G11R11_UFLOAT_PACK32));
    return missing_view ? VK_NULL_HANDLE : (VkImageView)(uintptr_t)(img == &linear ? 1 : 2);
}
static void capture_barrier(VkCommandBuffer cb, const VkDependencyInfo *info) {
    (void)cb;
    assert(info->memoryBarrierCount == 1);
    const VkMemoryBarrier2 *m = info->pMemoryBarriers;
    assert(m->srcStageMask == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT &&
           m->dstStageMask == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    assert(m->srcAccessMask == VK_ACCESS_2_MEMORY_WRITE_BIT &&
           m->dstAccessMask == (VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT));
    event(MEMORY);
}
static void capture_pipeline(VkCommandBuffer cb, VkPipelineBindPoint point, VkPipeline pipe) {
    (void)cb; (void)pipe;
    assert(point == VK_PIPELINE_BIND_POINT_COMPUTE); event(PIPELINE);
}
void rec_push(PFN_vkCmdPushDescriptorSetKHR fn, VkCommandBuffer cb, VkPipelineBindPoint point, VkPipelineLayout layout,
              uint32_t set, uint32_t n, const VkWriteDescriptorSet *writes) {
    (void)fn; (void)cb; (void)layout;
    assert(point == VK_PIPELINE_BIND_POINT_COMPUTE && set == 0 && n == 2);
    assert(writes[0].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
           writes[1].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    assert(writes[0].pImageInfo->imageView == (VkImageView)(uintptr_t)1 &&
           writes[1].pImageInfo->imageView == (VkImageView)(uintptr_t)2);
    event(DESCRIPTORS);
}
static void capture_constants(VkCommandBuffer cb, VkPipelineLayout layout, VkShaderStageFlags stages,
                              uint32_t offset, uint32_t size, const void *data) {
    (void)cb; (void)layout;
    const float *v = data;
    assert(stages == VK_SHADER_STAGE_COMPUTE_BIT && offset == 0 && size == 64);
    assert(v[0] == 2 && v[1] == 3 && v[2] == 1000 && v[3] == 0.5f);
    assert(fabsf(v[4] * v[4] + v[5] * v[5] + v[6] * v[6] - 1) < 0.00001f);
    assert(v[8] == 1280 && v[9] == 720); event(CONSTANTS);
}
static void capture_dispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    (void)cb; assert(x == 160 && y == 90 && z == 1); ledger.dispatches++; event(DISPATCH);
}
#undef vkCmdPipelineBarrier2
#undef vkCmdBindPipeline
#undef vkCmdPushConstants
#undef vkCmdDispatch
#define vkCmdPipelineBarrier2 capture_barrier
#define vkCmdBindPipeline capture_pipeline
#define vkCmdPushConstants capture_constants
#define vkCmdDispatch capture_dispatch
#include "contact_under_test.h"

static VkExec reset(void) {
    memset(&C, 0, sizeof C);
    C.tried = C.ok = 1, C.strength = 50;
    ledger = (Ledger){0}; missing_view = behind = waits = turns = hooks = 0;
    memset(uniforms, 0, sizeof uniforms);
    uniforms[7 * 4] = 2; uniforms[8 * 4 + 1] = 3; uniforms[14 * 4 + 1] = 1000;
    uniforms[44 * 4] = -1; uniforms[44 * 4 + 1] = -2; uniforms[44 * 4 + 2] = -2;
    scene = (VkTexImage){.image = (VkImage)(uintptr_t)1, .usage = VK_IMAGE_USAGE_STORAGE_BIT};
    linear = (VkTexImage){.image = (VkImage)(uintptr_t)2};
    VkExec ex = {.pass_w = 1280, .pass_h = 720, .pass_ncolor = 2};
    ex.ubo[1][1] = (p_bind_buf){.addr = (uint64_t)(uintptr_t)uniforms, .size = sizeof uniforms};
    ex.depth.format = VK_FORMAT_D32_SFLOAT;
    ex.color[0] = (VkExecTarget){.image = scene.image, .img = &scene,
        .format = VK_FORMAT_B10G11R11_UFLOAT_PACK32, .width = 1280, .height = 720};
    ex.color[1] = (VkExecTarget){.image = linear.image, .img = &linear,
        .format = VK_FORMAT_R32_SFLOAT, .width = 1280, .height = 720};
    return ex;
}
static void start_pass(VkExec *ex) {
    assert(!ex->in_pass); vk_contact_pass_begin(ex); ex->in_pass = 1;
}
static void draw(unsigned id) { event(DRAW_BASE + id); }
static void end_pass(VkExec *ex) { ex->in_pass = 0; }
static void composite(VkExec *ex) { ex->pass_ncolor = 1; ex->depth.format = VK_FORMAT_D16_UNORM; }
static void later(VkExec *ex) { ex->pass_ncolor = 0; ex->depth.format = VK_FORMAT_UNDEFINED; }

static Ledger sequence(int split, int composite_leader, int invalid, int absent) {
    VkExec ex = reset();
    start_pass(&ex); draw(1); end_pass(&ex);
    assert(C.scene == &scene && C.depth == &linear && !vk_contact_pending());
    composite(&ex);
    start_pass(&ex); draw(2); end_pass(&ex);
    assert(vk_contact_pending());
    if (!composite_leader) later(&ex);
    if (invalid) uniforms[14 * 4 + 1] = 0;
    missing_view = absent;
    start_pass(&ex); draw(3);
    if (split) { end_pass(&ex); start_pass(&ex); }
    draw(4); end_pass(&ex);
    later(&ex); start_pass(&ex); draw(5); end_pass(&ex);
    vk_contact_flush(&ex); // a later transfer must not repeat the completed dispatch
    assert(!vk_contact_pending());
    assert(ledger.dispatches == (unsigned)(!invalid && !absent));
    assert(ledger.invalidations == ledger.dispatches);
    assert(ex.recorded == (int)(6 * ledger.dispatches));
    return ledger;
}

int main(void) {
    for (int composite_leader = 0; composite_leader < 2; composite_leader++)
        for (int invalid = 0; invalid < 2; invalid++) for (int absent = 0; absent < 2; absent++) {
            Ledger control = sequence(0, composite_leader, invalid, absent);
            Ledger leader = sequence(1, composite_leader, invalid, absent);
            assert(!memcmp(&control, &leader, sizeof control));
        }
    VkExec ex = reset();
    start_pass(&ex); end_pass(&ex); composite(&ex); start_pass(&ex); end_pass(&ex);
    unsigned before = ledger.n;
    ex.in_pass = 1;
    vk_contact_flush(&ex);
    assert(!vk_contact_pending() && ledger.n == before); // production clears pending on this early return
    ex.in_pass = 0;
    vk_contact_flush(&ex);
    assert(ledger.n == before); // a missing original pass boundary would permanently lose this dispatch
    start_pass(&ex); end_pass(&ex);
    behind = 1;
    vk_contact_flush(&ex);
    assert(waits == 1 && turns == 1 && ledger.dispatches == 1);
    vk_contact_flush(&ex);
    assert(waits == 2 && turns == 1 && ledger.dispatches == 1);
    ex = reset();
    scene.usage = 0;
    start_pass(&ex); end_pass(&ex);
    assert(!C.scene && !vk_contact_pending());
    ex = reset();
    start_pass(&ex); end_pass(&ex); composite(&ex); start_pass(&ex); end_pass(&ex);
    ex.pass_ncolor = 2; ex.depth.format = VK_FORMAT_D32_SFLOAT;
    start_pass(&ex); end_pass(&ex);
    assert(!vk_contact_pending() && !ledger.dispatches); // a new G-buffer replaces the old pending scene
    puts("contact order: production hooks preserve dispatch/draw order across leader split; pending, invalid inputs, views and binding invalidation passed");
    return 0;
}

// Production clear logic chooses image versus attachment paths and exact masks/rectangles. Vulkan commands,
// target resolution and pass lifecycle are test boundaries; shader selection and target lifetimes are not modeled.
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "vk_exec.h"

enum { RESOLVE = 1, IMAGE_CLEAR, ATTACH_CLEAR, FEEDBACK, WRITE, DRAW = 20, W_STENCIL_MASK = 1 };
typedef struct {
    unsigned event[32], n, image_clears, attachment_clears, writes, warnings;
    VkImageAspectFlags aspect;
    VkClearDepthStencilValue value;
    VkRect2D rect;
    uint32_t layers, fb_dirty;
} Ledger;
static Ledger ledger;
static VkExec *current;
static int deferred;
unsigned vk_cond_clears_run, vk_cond_clears_skip, vk_cond_draws_skip, vk_cond_depth_run;
static void event(unsigned e) { assert(ledger.n < 32); ledger.event[ledger.n++] = e; }
static void clear_cond_log(const VkExec *ex, const char *name, int index) { (void)ex; (void)name; (void)index; }
static void warn_once(int kind, const char *fmt, ...) { (void)fmt; assert(kind == W_STENCIL_MASK); ledger.warnings++; }
static void resolve_rt(VkExec *ex) { assert(ex == current); event(RESOLVE); }
int vk_texture_deferred(void) { return deferred; }
void vk_exec_barrier(VkExec *ex) { ex->in_pass = 0; }
void vk_exec_pass_begin(VkExec *ex) { ex->in_pass = 1; }
void vk_exec_feedback_war(VkExec *ex, uint32_t written) {
    assert(ex->in_pass && written == 0x100); event(FEEDBACK);
}
void vk_gpu_gap_note(int kind, uint64_t units) { assert(kind == VK_GAP_CLEAR_TARGET && units == 640u * 480u); }
void vk_texture_gpu_written(const tex_t *tex) { (void)tex; ledger.writes++; event(WRITE); }
static void capture_image(VkCommandBuffer cb, VkImage image, VkImageLayout layout,
                          const VkClearDepthStencilValue *value, uint32_t n, const VkImageSubresourceRange *range) {
    (void)cb; assert(!current->in_pass && image == current->depth.image && layout == VK_IMAGE_LAYOUT_GENERAL && n == 1);
    assert(range->baseMipLevel == current->depth.level && range->baseArrayLayer == current->depth.base_layer);
    ledger.aspect = range->aspectMask, ledger.value = *value, ledger.layers = range->layerCount;
    ledger.image_clears++; event(IMAGE_CLEAR);
}
static void capture_attachment(VkCommandBuffer cb, uint32_t n, const VkClearAttachment *attachment,
                               uint32_t nr, const VkClearRect *rect) {
    (void)cb; assert(current->in_pass && n == 1 && nr == 1 && rect->baseArrayLayer == 0);
    ledger.aspect = attachment->aspectMask, ledger.value = attachment->clearValue.depthStencil;
    ledger.rect = rect->rect, ledger.layers = rect->layerCount;
    ledger.attachment_clears++; event(ATTACH_CLEAR);
}
#undef vkCmdClearDepthStencilImage
#undef vkCmdClearAttachments
#define vkCmdClearDepthStencilImage capture_image
#define vkCmdClearAttachments capture_attachment
#include "clear_under_test.h"

static Ledger sequence(int split, unsigned kind) {
    ledger = (Ledger){0}; deferred = 0;
    vk_cond_clears_run = vk_cond_clears_skip = vk_cond_depth_run = 0;
    VkExec ex = {.pass_w = 640, .pass_h = 480, .pass_layers = 2, .has_render_cond = 1};
    ex.depth = (VkExecTarget){.image = (VkImage)(uintptr_t)1, .width = 640, .height = 480,
        .aspect = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, .layers = 2, .base_layer = 3, .level = 1};
    current = &ex;
    p_clear_ds clear = {.depth = 2, .depth_mask = 1, .stencil = 0x123, .stencil_mask = 0xff};
    switch (kind) {
    case 0: break; // whole target, clamped depth and masked stencil
    case 1: ex.has_scissor = 1; ex.scissor = (p_scissor){.x = 7, .y = 9, .w = 20, .h = 30}; break;
    case 2: ex.has_scissor = 1; ex.scissor = (p_scissor){.x = 900, .y = 900, .w = 20, .h = 30}; break;
    case 3: ex.depth.is3d = 1; break;
    case 4: ex.depth.blocks = 1; break;
    case 5: ex.render_disabled = 1; break;
    case 6: deferred = 1; break;
    case 7: ex.depth.image = VK_NULL_HANDLE; break;
    case 8: clear.depth_mask = 0; clear.stencil_mask = 0x7f; break;
    case 9: clear.depth_mask = 0; break;
    case 10: clear.stencil_mask = 0; clear.depth = -1; break;
    case 11: ex.has_scissor = 1; ex.pass_scale = 2; ex.scissor = (p_scissor){.x = -2, .y = 4, .w = 12, .h = 15}; break;
    default: assert(0);
    }
    ex.in_pass = 1;
    vk_exec_barrier(&ex); // original group-entry boundary, required on either schedule
    clear_ds(&ex, &clear);
    if (!ex.in_pass) vk_exec_pass_begin(&ex);
    event(DRAW);
    if (split) { vk_exec_barrier(&ex); vk_exec_pass_begin(&ex); }
    event(DRAW + 1);
    ledger.fb_dirty = ex.fb_dirty;
    if (kind == 5) assert(vk_cond_clears_skip == 1 && !vk_cond_clears_run);
    else assert(vk_cond_clears_run == 1 && vk_cond_depth_run == 1);
    return ledger;
}

int main(void) {
    for (unsigned kind = 0; kind < 12; kind++) {
        Ledger control = sequence(0, kind), candidate = sequence(1, kind);
        assert(!memcmp(&control, &candidate, sizeof control));
        if (kind == 0 || kind == 9 || kind == 10) {
            assert(candidate.image_clears == 1 && candidate.writes == 1 && !candidate.attachment_clears);
            assert(candidate.value.depth == (kind == 10 ? 0 : 1) && candidate.value.stencil == 0x23);
            assert(candidate.layers == 2);
        } else if (kind == 1 || kind == 3 || kind == 4 || kind == 11) {
            assert(candidate.attachment_clears == 1 && !candidate.image_clears && candidate.fb_dirty == 0x100);
        } else assert(!candidate.image_clears && !candidate.attachment_clears && !candidate.writes);
        if (kind == 1) assert(candidate.rect.offset.x == 7 && candidate.rect.offset.y == 9 &&
                              candidate.rect.extent.width == 20 && candidate.rect.extent.height == 30);
        if (kind == 11) assert(candidate.rect.offset.x == 0 && candidate.rect.offset.y == 8 &&
                               candidate.rect.extent.width == 20 && candidate.rect.extent.height == 30);
        if (kind == 8) assert(candidate.warnings == 1);
        if (kind == 9) assert(candidate.aspect == VK_IMAGE_ASPECT_STENCIL_BIT);
        if (kind == 10) assert(candidate.aspect == VK_IMAGE_ASPECT_DEPTH_BIT);
    }
    puts("clear order: production depth/stencil clear paths, masks, clamping, scissor, feedback and exactly-once writes passed across leader split");
    return 0;
}

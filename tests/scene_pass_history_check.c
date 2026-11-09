// Link with runtime/vk_target_history.c. Uses synthetic attachments and a simulated present counter.
#include <assert.h>
#include <stdio.h>
#include "vk_exec.h"

GfxStats gfx_counters;

void wide_picture(unsigned *w, unsigned *h) { *w = 3416, *h = 1440; }

static VkExecTarget target(uintptr_t id, VkFormat format, unsigned w, unsigned h) {
    return (VkExecTarget){.image = (VkImage)id, .format = format, .width = w, .height = h,
                          .samples = 1, .layers = 1};
}

static void finish_pass(const VkExec *ex, uint64_t stamp) {
    gfx_counters.frames = stamp - 1;
    vk_exec_targets_note(ex);
}

int main(void) {
    for (unsigned n = 1; n <= 2; n++) {
        uint64_t first = n == 1 ? 1 : 10;
        VkExec ex = {.pass_ncolor = n};
        ex.depth = target(1, VK_FORMAT_D32_SFLOAT, 3416 / n, 1440 / n);
        VkFormat format = n == 1 ? VK_FORMAT_B10G11R11_UFLOAT_PACK32 : VK_FORMAT_R16G16B16A16_SFLOAT;
        for (unsigned i = 0; i < n; i++) ex.color[i] = target(10 + i, format, ex.depth.width, ex.depth.height);
        finish_pass(&ex, first);
        assert(!vk_exec_targets_recent(&ex, 1, 1));
        for (unsigned i = 0; i < n; i++) ex.color[i].image = (VkImage)(uintptr_t)(20 + i);
        gfx_counters.frames = first;
        assert(!vk_exec_targets_recent(&ex, 1, 0));
        assert(vk_exec_targets_recent(&ex, 1, 1));
        finish_pass(&ex, first + 1);
        assert(vk_exec_targets_recent(&ex, 1, 1));
        ex.color[0].level = 1;
        assert(!vk_exec_targets_recent(&ex, 1, 1));
        ex.color[0].level = 0;
        ex.color[0].format = VK_FORMAT_R8G8B8A8_UNORM;
        assert(!vk_exec_targets_recent(&ex, 1, 1));
        ex.color[0].format = format;
        ex.depth.image = (VkImage)(uintptr_t)3;
        assert(!vk_exec_targets_recent(&ex, 1, 1));
        ex.depth.image = (VkImage)(uintptr_t)1;
        gfx_counters.frames = first + 2;
        assert(!vk_exec_targets_recent(&ex, 1, 1));
    }
    puts("Scene pass history checks passed");
    return 0;
}

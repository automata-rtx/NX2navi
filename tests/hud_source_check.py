
"""Extract and exercise the native HUD producer's receipt and admission code."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def section(s, start, end):
    i = s.index(start)
    return s[i:s.index(end, i)]

def fixture(source):
    producer = section(source, 'static void hudless(', 'static int scene_color(')
    producer = producer[producer.rindex('static void hudless('):]
    retained = producer.index('vk_texture_present_retain_image(texture, target)')
    parked = producer.index('vk_exec_temporal_prepare(ex, hud_group_prepare, &want)')
    emitted = producer.index('vkCmdBlitImage(cb, source')
    assert retained < producer.index('if (!target_owner) return;') < parked < emitted
    assert producer.index('size_scene_use(ex, reconstruction)') < parked
    assert producer.count('vk_texture_present_release(target_owner)') == 2
    owner = section(source, 'struct VkDlaaHudSource {', 'struct VkDlaaFgSizeUse {')
    allocation = section(source, 'static VkImageCreateInfo dlaa_image_2d_info', 'enum { DLAA_SIZE_NATIVE_IMAGES = 7 };')
    hud = section(source, 'static void hud_group_drop_target', 'VkNgxImage vk_dlaa_present_alpha')
    allocation += hud.split('void vk_dlaa_present_alpha_unrecorded', 1)[0]
    return r'''
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk_exec.h"
#include "vk_dlaa.h"
#include "resource_core_vma.h"

#define H(type, value) ((type)(uintptr_t)(value))
static struct { uint64_t presents, fg_at; } D;
static VkCtx ctx;
struct VkTexturePresentSource { unsigned drops; };
static struct VkTexturePresentSource guest;

static uint64_t handed, expected_peak;
static unsigned placed, reserve_count, cancels, creates, allocations, made_views, destroyed_images,
                destroyed_views, fail_create, fail_view, live_images, live_views, records;
volatile int32_t vk_lost;
static int pressure, stopping;
static VkDeviceSize pixels = 320 * 180;
static struct Native { VkImage img; VkImageView view; VmaAllocation memory; } native[32];
static struct Callback { void (*done)(void *); void (*cancel)(void *, int); void *context; uint64_t scene; } record[8];

VkCtx *vk_ctx(void) { return &ctx; }
uint64_t vk_sync_handed_fence(void) { return handed; }
int vk_sync_stopping(void) { return stopping; }
void resource_core_budget_progress(void) {}
void rt_fatal(const char *why, ...) { fprintf(stderr, "%s\n", why); abort(); }
void vk_texture_present_release(VkTexturePresentSource *source) { if (source) source->drops++; }
void vk_completion_add_owned(uint64_t scene, void (*done)(void *), void (*cancel)(void *, int), void *context) {
    assert(records < sizeof record / sizeof *record && scene && done && cancel && context);
    record[records++] = (struct Callback){done, cancel, context, scene};
}

static uint64_t cost(VkFormat format) {
    uint64_t bpp = format == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : format == VK_FORMAT_R8_UNORM ? 1 : 4;
    return (pixels * bpp + 4095) & ~(uint64_t)4095;
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties(VkPhysicalDevice d, VkFormat f, VkImageType t,
        VkImageTiling tiling, VkImageUsageFlags usage, VkImageCreateFlags flags, VkImageFormatProperties *p) {
    (void)d; (void)f; (void)t; (void)tiling; (void)usage; (void)flags;
    p->maxExtent = (VkExtent3D){4096, 4096, 1};
    return VK_SUCCESS;
}

VkResult resource_core_vma_image_peak(const RcVmaRequest *req, const VkImageCreateInfo *image,
    const VmaAllocationCreateInfo *alloc,VmaAllocationCreateInfo *placement,RcAdmissionVector *peak) {
    assert(req->producer == RC_BUDGET_TEXTURE && !req->required && !req->peak_ticket);
    assert(alloc->flags == VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT && image->extent.width * image->extent.height == pixels);
    assert((image->usage & (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) ==
        (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT));
    if (image->format == VK_FORMAT_R8_UNORM) assert(image->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    else if (image->format == VK_FORMAT_R8G8B8A8_SRGB) assert(!(image->usage & VK_IMAGE_USAGE_STORAGE_BIT));
    else assert(image->usage & VK_IMAGE_USAGE_STORAGE_BIT);
    *placement = *alloc;
    placement->memoryTypeBits = 1;
    *peak = (RcAdmissionVector){0};
    peak->amount[RC_BUDGET_GPU_FIRST] = cost(image->format);
    placed++;
    return VK_SUCCESS;
}

RcAdmissionStatus resource_core_budget_reserve(RcBudgetProducer producer, int required,
        const RcAdmissionVector *peak, RcBudgetTicket *ticket) {
    (void)ticket;
    assert(producer == RC_BUDGET_TEXTURE && !required);
    reserve_count++;
    expected_peak = peak->amount[RC_BUDGET_GPU_FIRST];
    if (pressure && expected_peak > 2 * cost(VK_FORMAT_R8G8B8A8_UNORM)) return RC_ADMISSION_REFUSED;
    return RC_ADMISSION_OK;
}
RcAdmissionStatus resource_core_budget_cancel(RcBudgetTicket *ticket) { assert(ticket); cancels++; return RC_ADMISSION_OK; }

VkResult resource_core_vma_create_image(const RcVmaRequest *req,VmaAllocator alloc,const VkImageCreateInfo *image,
    const VmaAllocationCreateInfo *placement,VkImage *img,VmaAllocation *mem,VmaAllocationInfo *info,RcAdmissionStatus *admission) {
    (void)alloc; (void)info;
    assert(req->peak_ticket && placement->memoryTypeBits==1 && reserve_count>cancels);
    creates++;
    if (creates == fail_create) {
        *admission = RC_ADMISSION_REFUSED;
        return VK_NOT_READY;
    }
    assert(expected_peak >= cost(image->format) && allocations + 1 < sizeof native / sizeof *native);
    expected_peak -= cost(image->format);
    unsigned i = ++allocations;
    *img = native[i].img = H(VkImage,i);
    *mem = native[i].memory = H(VmaAllocation, i+100);
    live_images++;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(VkDevice d,const VkImageViewCreateInfo *ci,
        const VkAllocationCallbacks *cb,VkImageView *view) {
    (void)d; (void)cb;
    assert(ci->format != VK_FORMAT_R8G8B8A8_SRGB);
    if (++made_views == fail_view) return VK_ERROR_OUT_OF_HOST_MEMORY;
    unsigned i = (unsigned)(uintptr_t)ci->image;
    *view = native[i].view = H(VkImageView, i+200);
    live_views++;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(VkDevice d,VkImageView v,const VkAllocationCallbacks *cb) {
    (void)d; (void)cb;
    unsigned i = (unsigned)(uintptr_t)v-200;
    assert(native[i].view == v && native[i].img);
    native[i].view = VK_NULL_HANDLE;
    live_views--; destroyed_views++;
}

void resource_core_vma_destroy_image(VmaAllocator allocator,VkImage image,VmaAllocation alloc) {
    (void)allocator;
    unsigned i = (unsigned)(uintptr_t)image;
    assert(native[i].img == image && native[i].memory == alloc && !native[i].view);
    native[i] = (struct Native){0};
    live_images--; destroyed_images++;
}
''' + owner + allocation + r'''
static void budget_and_rollback(void) {
    HudSourceRequest want = {.w=320,.h=180,.format=VK_FORMAT_R8G8B8A8_UNORM,.coverage=1};
    VkDlaaHudSource out = {0};
    pressure = 1;
    assert(hud_group_create(&want, &out) == VK_NOT_READY && !out.images.color.img && !creates && !cancels);
    assert(placed == 3 && expected_peak == 2*cost(want.format)+cost(VK_FORMAT_R8_UNORM));
    pressure = 0;
    fail_view = 2;
    assert(hud_group_create(&want, &out) == VK_ERROR_OUT_OF_HOST_MEMORY && !out.images.color.img);
    assert(!live_images && !live_views && destroyed_images == 2 && destroyed_views == 1 && cancels == 1);
    fail_create = creates + 3;
    assert(hud_group_create(&want, &out) == VK_NOT_READY && !out.images.color.img);
    assert(!live_images && !live_views && destroyed_images == 4 && destroyed_views == 3);
    assert(hud_group_create(&want, &out) == VK_SUCCESS && !expected_peak);
    assert(live_images == 3 && live_views == 2 && out.srgb && out.coverage.fmt == VK_FORMAT_R8_UNORM);
    hud_group_destroy(&out);
    assert(!live_images && !live_views && cancels == 3);
}

static void exact_source_lifetime(void) {
    HudSourceRequest want = {.w=320,.h=180,.format=VK_FORMAT_R16G16B16A16_SFLOAT,.coverage=1};
    VkDlaaHudSource *group = &hud_sources.groups[0];
    assert(hud_group_create(&want, group) == VK_SUCCESS && !group->srgb && live_images == 2);
    group->target_owner = &guest;
    group->target = H(VkImage,900);
    group->target_format = want.format;
    group->images.generation = 19;
    group->turn = D.presents = D.fg_at = 7;
    hud_sources.open = group;
    VkExec ex = {.fence=40, .pass_ncolor=1, .pass_layers=1, .in_pass=1};
    ex.color[0]=(VkExecTarget){.image=group->target,.width=320,.height=180,.samples=1};
    VkRenderingAttachmentInfo alpha;
    assert(vk_dlaa_hud_attachment(&ex, &alpha) && alpha.imageView == group->coverage.view);
    ex.hud_alpha_attachment=1;
    vk_dlaa_hud_target_report(&ex);
    vk_dlaa_hud_write(group->target);
    vk_dlaa_hud_target_report(NULL);
    assert(!group->alpha_bad);
    vk_dlaa_hud_draw_emitted(&ex);
    hud_group_record(&ex,group);
    hud_group_record(&ex,group);
    ex.fence++;
    hud_group_record(&ex,group);
    assert(records == 2 && group->scenes == 2 && !hud_group_find(&want));
    vk_dlaa_hud_frame(100);
    VkTexImage target = {.image=group->target,.width=320,.height=180,.format=want.format};
    assert(!vk_dlaa_hud_take(99,&target));
    VkDlaaHudSource *receipt = vk_dlaa_hud_take(100,&target);
    assert(receipt == group && !vk_dlaa_hud_take(100,&target) && !vk_dlaa_hud_source_images(receipt,100));
    handed = 40;
    record[0].done(record[0].context);
    assert(!hud_group_find(&want) && !vk_dlaa_hud_source_images(receipt,100));
    handed = 41;
    const VkDlaaHudImages *images = vk_dlaa_hud_source_images(receipt,100);
    assert(images && images->generation==19 && images->exact && images->alpha.view==group->coverage.view);
    record[1].done(record[1].context);
    assert(!hud_group_find(&want));
    vk_dlaa_hud_source_release(receipt);
    assert(hud_group_find(&want)==group);
    hud_group_collect();
    assert(guest.drops==1);

    D.presents++;
    D.fg_at++;
    group->turn=D.presents;
    hud_sources.open=group;
    group->target=ex.color[0].image;
    ex.hud_alpha_attachment=0;
    vk_dlaa_hud_draw_emitted(&ex);
    assert(group->alpha_bad);
    vk_dlaa_hud_frame(102);
    target.image=H(VkImage,901);
    assert((receipt=vk_dlaa_hud_take(102,&target)) == group);
    assert((images=vk_dlaa_hud_source_images(receipt,102)) && !images->exact && !images->alpha.img);
    vk_dlaa_hud_source_release(receipt);
    vk_dlaa_hud_shutdown();
    assert(!live_images && !live_views);
}

static void cancel_and_optional(void) {
    HudSourceRequest want = {.w=320,.h=180,.format=VK_FORMAT_R8G8B8A8_UNORM,.coverage=1};
    pressure=1;
    assert(hud_group_prepare(&want));
    VkDlaaHudSource *group=want.ready;
    assert(group && !group->coverage.img && group->srgb && reserve_count==2 && cancels==1);
    group->turn = D.presents;
    hud_sources.open=group;
    VkExec ex={.fence=4};
    hud_group_record(&ex,group);
    vk_dlaa_hud_frame(33);
    record[0].cancel(record[0].context,1);
    VkTexImage target={.image=H(VkImage,3),.format=want.format,.width=320,.height=180};
    assert(!vk_dlaa_hud_take(33,&target) && !hud_group_find(&want));
    vk_dlaa_hud_shutdown();
    assert(!live_images && !live_views);
}

int main(int argc,char **argv) {
    assert(argc==2);
    if (!strcmp(argv[1],"budget")) budget_and_rollback();
    else if (!strcmp(argv[1],"source")) exact_source_lifetime();
    else if (!strcmp(argv[1],"cancel")) cancel_and_optional();
    else abort();
    return 0;
}
'''

def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runtime/vk_dlaa.c"
    cc = os.environ.get("CC") or shutil.which("clang")
    if not cc:
        raise SystemExit("Clang is required")
    clang = Path(shutil.which(cc) or cc).resolve()
    vma = os.environ.get("VMA_INCLUDE", str(clang.parent.parent / "include" / "vma"))
    overlay = os.environ.get("HUD_VFS_OVERLAY")
    overlay_args = ["-ivfsoverlay", overlay] if overlay else []
    env = dict(os.environ, PATH=str(clang.parent) + os.pathsep + os.environ["PATH"])
    with tempfile.TemporaryDirectory(prefix="hud-source-check-") as t:
        c, exe = Path(t) / "check.c", Path(t) / ("check.exe" if os.name == "nt" else "check")
        c.write_text(fixture(path.read_text()), encoding="utf-8")
        subprocess.run([str(clang), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=undefined", *overlay_args,
                        "-I"+str(ROOT / "runtime"), "-I"+vma, str(c), "-o", str(exe)],
                       env=env, check=True)
        for case in ("budget", "source", "cancel"):
            subprocess.run([str(exe), case], env=env, check=True)
    print("HUD source ownership passed")

if __name__ == "__main__":
    main()

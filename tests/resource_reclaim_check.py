"""Exercise production deferred-image reclamation accounting without a game or Vulkan device."""
from pathlib import Path
import os
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ.get("CC", r"C:\msys64\ucrt64\bin\clang.exe"))
if not CC.is_file():
    CC = Path(r"C:\Users\Andrew\AppData\Local\Programs\2NX2Navi\toolchain\msys64\ucrt64\bin\clang.exe")
sys.path.insert(0, str(ROOT / "tools"))
from check_content import GENERATED_MARKER

source = (ROOT / "runtime/vk_memory.c").read_text()


def function(name):
    match = re.search(r"^(?:static )?[^\n]*\b" + name + r"\([^;]*?\)\s*\{", source, re.M)
    if not match:
        raise RuntimeError("Missing production function " + name)
    start = source.index("{", match.start())
    depth, end = 1, start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end] + "\n"


types = source[source.index("enum { DEAD_IMAGE ="):source.index("static PlatRMutex cmd_lock;", source.index("enum { DEAD_IMAGE ="))]
fixture = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef uintptr_t VkImage;
typedef uintptr_t VkImageView;
typedef uintptr_t VkBuffer;
typedef void *VmaAllocation;
typedef struct { int device, vma, timeline; } VkCtx;
enum { REC_VIEW, REC_BUFFER, REC_IMAGE };
static VkCtx ctx = {1, 1, 1};
static int locked, cmd_once, cmd_lock;
static uint64_t fake_timeline, fake_recording, fake_issued;
static uint64_t destroyed, before_destroy, last_expected;
static VkCtx *vk_ctx(void) { return &ctx; }
static void cmd_init(void *p) { (void)p; }
static void plat_once(int *o, void (*fn)(void *), void *p) { (void)o; (void)fn; (void)p; }
static void plat_rmutex_lock(int *p) { (void)p; assert(!locked); locked = 1; }
static void plat_rmutex_unlock(int *p) { (void)p; assert(locked); locked = 0; }
static int rt_knob(const char *n, const char *v) { (void)n; (void)v; return 1; }
static uint64_t vk_exec_recording_fence(void) { return fake_recording; }
static uint64_t vk_sync_issued(void) { return fake_issued; }
static uint64_t vk_sync_fence_done(void) { return fake_timeline; }
static void rec_check_handle(int k, uint64_t h, int d, uint64_t t, uint64_t f) {
    (void)k; (void)h; (void)d; (void)t; (void)f;
}
static void vkGetSemaphoreCounterValue(int d, int t, uint64_t *out) { (void)d; (void)t; *out = fake_timeline; }
static void rt_fatal(const char *s) { (void)s; abort(); }
''' + types + r'''
static Dead *dead;
static int ndead, cap_dead, npwait;
static uint64_t next_seq, done_seq, tl_noted;
static int open_active;
static struct { uint64_t seq; } open_b;
static void retire(VkCtx *c, int wait) { (void)c; (void)wait; }
static void pool_wait_collect(uint64_t t) { (void)t; assert(0); }
static void vmaDestroyImage(int v, VkImage i, VmaAllocation a) {
    (void)v; (void)i; (void)a;
    before_destroy = vk_mem_reclaim_pending();
    assert(before_destroy >= last_expected);
    destroyed++;
}
static void vkDestroyImage(int d, VkImage i, void *a) { (void)d; (void)i; (void)a; destroyed++; }
static void vkDestroyImageView(int d, VkImageView i, void *a) { (void)d; (void)i; (void)a; }
static void vmaDestroyBuffer(int v, VkBuffer b, VmaAllocation a) { (void)v; (void)b; (void)a; }
'''
for name in ["grow", "destroy_dead", "collect_locked_out", "collect_locked", "defer",
             "vk_defer_image", "vk_defer_image_reclaim", "vk_defer_view", "vk_defer_free",
             "vk_defer_buffer", "vk_mem_collect"]:
    fixture += function(name)
fixture += r'''
int main(void) {
    VmaAllocation own = (void *)1;
    fake_issued = 4; fake_recording = 8; tl_noted = 6; next_seq = 3;
    vk_defer_image_reclaim(0, own, 999);
    vk_defer_image_reclaim(1, NULL, 999);
    vk_defer_image(2, own);
    vk_defer_image_reclaim(3, own, 100);
    assert(ndead == 3 && vk_mem_reclaim_pending() == 100);
    assert(dead[2].tl == 8 && dead[2].seq == 3);
    // Neither one completed fence nor allocator-eligible objects removed into an output array release the charge.
    fake_timeline = 8; done_seq = 2; vk_mem_collect();
    assert(ndead == 3 && !destroyed && vk_mem_reclaim_pending() == 100);
    done_seq = 3; fake_timeline = 7; vk_mem_collect(); assert(!destroyed);
    fake_timeline = 8;
    Dead *out = NULL; int nout = 0, cout = 0;
    collect_locked_out(&ctx, &out, &nout, &cout);
    assert(ndead == 0 && nout == 3 && vk_mem_reclaim_pending() == 100);
    last_expected = 100;
    for (int i = 0; i < nout; i++) destroy_dead(&ctx, &out[i]);
    assert(destroyed == 3 && before_destroy == 100 && !vk_mem_reclaim_pending());
    free(out);
    // Public collector retains the charge until destruction, including an open upload batch's newer sequence.
    open_active = 1; open_b.seq = 10; fake_recording = 0; fake_issued = 9;
    vk_defer_image_reclaim(4, own, 300); last_expected = 300;
    assert(dead[0].tl == 10 && dead[0].seq == 10);
    fake_timeline = 10; done_seq = 9; vk_mem_collect(); assert(vk_mem_reclaim_pending() == 300);
    done_seq = 10; vk_mem_collect();
    assert(before_destroy == 300 && !vk_mem_reclaim_pending() && !locked);
    // Shutdown zeroes both stamps after device idle, then uses this same inline destruction path.
    vk_defer_image_reclaim(5, own, 400); last_expected = 400;
    for (int i = 0; i < ndead; i++) dead[i].seq = dead[i].tl = 0;
    collect_locked(&ctx); assert(before_destroy == 400 && !vk_mem_reclaim_pending());
    vk_defer_view(6); vk_defer_free(malloc(1)); vk_defer_buffer(7, own);
    collect_locked(&ctx); assert(!ndead && !vk_mem_reclaim_pending());
    free(dead);
    puts("deferred resource reclamation: passed");
    return 0;
}
'''
out = ROOT / "build/test/resource-reclaim"
out.mkdir(parents=True, exist_ok=True)
marker = GENERATED_MARKER.decode() if isinstance(GENERATED_MARKER, bytes) else GENERATED_MARKER
src = out / "reclaim.c"
src.write_text("// " + marker + "\n" + fixture)
exe = out / "reclaim.exe"
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
subprocess.run([str(CC), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                "-fsanitize=undefined", "-fsanitize-trap=all", str(src), "-o", str(exe)], env=env, check=True)
subprocess.run([str(exe)], env=env, check=True)

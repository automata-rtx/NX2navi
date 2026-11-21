"""Check recorder-owned and completion-owned native mirror snapshot chunks."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


RUNTIME = Path(__file__).resolve().parents[1] / "runtime"


PREFIX = r"""
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uintptr_t VkBuffer;
typedef uintptr_t VmaAllocation;
typedef struct { int vma; } VkCtx;
typedef struct { int held; } PlatMutex;
typedef struct {
    uint64_t fence;
    void (*fn)(void *);
    void *arg;
    void (*cancel)(void *, int);
} Pending;
#define VK_NULL_HANDLE 0

static VkCtx context = {1};
static VkCtx *vk_ctx(void) { return &context; }
static void plat_mutex_lock(PlatMutex *lock) { assert(!lock->held++); }
static void plat_mutex_unlock(PlatMutex *lock) { assert(lock->held-- == 1); }
#define rt_fatal(...) abort()

static Pending jobs[4];
static int n_jobs;
static unsigned destroys[8];
static void vk_completion_add_owned(uint64_t fence, void (*fn)(void *), void (*cancel)(void *, int), void *arg) {
    assert(fence == 7 && fn && cancel && arg && n_jobs < 4);
    jobs[n_jobs++] = (Pending){fence, fn, arg, cancel};
}
static void resource_core_vma_destroy_buffer(int vma, VkBuffer buf, VmaAllocation alloc) {
    assert(vma == 1 && buf && buf < 8 && alloc == buf + 10 && !destroys[buf]++);
}
"""


STATE = r"""
enum { SNAP_POOL_N = 4 };
static SnapChunk *chunks, snap_pool[SNAP_POOL_N];
static int nchunks, cchunks, nchunks_transferred, nsnap_pool;
static uint64_t snap_completion_fence, snap_pool_bytes;
static PlatMutex snap_pool_lock;
static uint64_t snap_pool_cap(void) { return 64; }
"""


CASES = r"""
static void open_scene(void) {
    assert(!chunks && !nchunks && !nchunks_transferred && !nsnap_pool && !snap_completion_fence);
    n_jobs = 0;
    memset(destroys, 0, sizeof destroys);
    chunks = calloc(3, sizeof *chunks);
    assert(chunks);
    cchunks = 3;
}
static void add(VkBuffer buf) {
    assert(nchunks < cchunks);
    chunks[nchunks++] = (SnapChunk){.buf = buf, .alloc = buf + 10, .size = 8};
}
static void stop_before_transfer(void) {
    open_scene();
    add(1);
    add(2);
    snap_transfer(0);
    assert(!n_jobs && !nchunks_transferred && !destroys[1] && !destroys[2]);
    snap_chunks_shutdown(&context);
    assert(destroys[1] == 1 && destroys[2] == 1);
}
static void finish_with_appended_capture(void) {
    open_scene();
    add(1);
    snap_completion_fence = 7;
    snap_transfer(snap_completion_fence);
    assert(n_jobs == 1 && nchunks_transferred == 1 && chunks[0].buf == 1);
    add(2);
    snap_transfer(snap_completion_fence);
    snap_transfer(snap_completion_fence);
    assert(n_jobs == 2 && nchunks_transferred == 2 && chunks[1].buf == 2);
    assert(pending_disposable(jobs[0], 7, 7, 0) && pending_disposable(jobs[1], 7, 7, 0));
    pending_dispose(jobs[0], 7, 7);
    pending_dispose(jobs[1], 7, 7);
    assert(nsnap_pool == 2 && !destroys[1] && !destroys[2]);
    snap_chunks_shutdown(&context);
    assert(destroys[1] == 1 && destroys[2] == 1);
}
static void cancel_after_first_transfer(int handed) {
    open_scene();
    add(1);
    snap_transfer(7);
    add(2);
    assert(n_jobs == 1 && nchunks_transferred == 1);
    assert(pending_disposable(jobs[0], 6, handed, 0) == (handed == 6));
    assert(!nsnap_pool && !destroys[1] && !destroys[2]);
    assert(pending_disposable(jobs[0], 6, handed, 1));
    pending_dispose(jobs[0], 6, handed);
    assert(!nsnap_pool && destroys[1] == 1 && !destroys[2]);
    snap_chunks_shutdown(&context);
    assert(destroys[1] == 1 && destroys[2] == 1);
}
static void cancel_after_final_transfer(int handed) {
    open_scene();
    add(1);
    snap_transfer(7);
    add(2);
    snap_transfer(7);
    assert(n_jobs == 2 && nchunks_transferred == 2);
    for (int i = 0; i < n_jobs; ++i) {
        assert(pending_disposable(jobs[i], 6, handed, 0) == (handed == 6));
        assert(pending_disposable(jobs[i], 6, handed, 1));
        pending_dispose(jobs[i], 6, handed);
    }
    assert(!nsnap_pool && destroys[1] == 1 && destroys[2] == 1);
    snap_chunks_shutdown(&context);
    assert(destroys[1] == 1 && destroys[2] == 1);
}
int main(void) {
    stop_before_transfer();
    finish_with_appended_capture();
    cancel_after_first_transfer(6);
    cancel_after_first_transfer(7);
    cancel_after_final_transfer(6);
    cancel_after_final_transfer(7);
    puts("mirror snapshots: six stopped, completed, and transferred-cancellation cases passed");
    return 0;
}
"""


def main():
    compiler = os.environ.get("CC") or shutil.which("clang")
    if not compiler:
        raise SystemExit("Clang is required (set CC)")
    cc = Path(shutil.which(compiler) or compiler).resolve()
    memory = (RUNTIME / "vk_memory.c").read_text(encoding="utf-8")
    sync = (RUNTIME / "vk_sync.c").read_text(encoding="utf-8")
    type_start = memory.index("typedef struct { VkBuffer buf; VmaAllocation alloc;", memory.index("} SnapRange;"))
    type_end = memory.index("static SnapRange *snl;", type_start)
    owners_start = memory.index("static void snap_chunks_destroy(")
    owners_end = memory.index("// ---- lazy write-back", owners_start)
    dispose_start = sync.index("static int pending_disposable(")
    dispose_end = sync.index("static void pending_dispose_queue(", dispose_start)
    prologue = memory[memory.index("void vk_mem_mirror_prologue("):memory.index("// ---- verify mode")]
    assert "snap_completion_fence = 0;" in prologue and "snap_transfer(completion_fence);" in prologue
    assert "snap_completion_fence = vk_mem_mirror_pending() ? fence : 0;" in memory
    body = (PREFIX + memory[type_start:type_end] + STATE + memory[owners_start:owners_end] +
            sync[dispose_start:dispose_end] + CASES)
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="mirror-snapshot-lifetime-") as temporary:
        source = Path(temporary) / "check.c"
        exe = source.with_suffix(".exe" if os.name == "nt" else "")
        source.write_text(body, encoding="utf-8")
        subprocess.run([str(cc), "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=undefined", str(source), "-o", str(exe)],
                       env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()

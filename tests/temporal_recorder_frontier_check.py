"""Check that required reconstruction preparation parks only its ordered recorder."""

import os
from pathlib import Path
import subprocess
import tempfile

from ordered_dispatch_preparation_check import REPLAY, between, function


STUBS = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>

typedef struct { int cb; } VkExec;
typedef struct { unsigned index; int cb; } Chunk;
typedef struct { Chunk c[4]; int at; uint64_t out_ns; } Group;
typedef enum {
    VK_SYNC_PROGRESS_COMPLETE, VK_SYNC_PROGRESS_STOPPED, VK_SYNC_PROGRESS_UNAVAILABLE,
} VkSyncProgressResult;
enum { PW_HOOKS, QN = 8 };
static struct { Chunk *q[QN]; uint64_t out, drained, put; } Q;
static struct { int n; } RW = {3};
static VkExec *x_cur;
static Chunk *rw_chunk, *temporal_hook_chunk;
static int stopping, lost, failed, sizing_open, close_on_drain;
static unsigned drains, parks, first_chunks, next_calls;
static int vk_sync_stopping(void) { return stopping; }
static int vk_device_is_lost(void) { return lost; }
static int vk_exec_recording_failed(void) { return failed; }
static int vk_dlaa_temporal_hook_needs_recorder(void) { return sizing_open; }
static void group_drain(VkExec *ex, int why) {
    assert(ex == x_cur && why == PW_HOOKS && !rw_chunk);
    ++drains;
    Q.drained = Q.out;
    if (close_on_drain) sizing_open = 0;
}
static VkSyncProgressResult vk_sync_progress_prepare_pipeline(int (*prepare)(void *), void *context) {
    ++parks;
    return prepare(context) ? VK_SYNC_PROGRESS_COMPLETE : VK_SYNC_PROGRESS_STOPPED;
}
static int prepare(void *context) {
    assert(context == &next_calls);
    ++next_calls;
    return !stopping;
}
static uint64_t kern_now_ns(void) { return 42; }
static int vk_sync_append_later(int count, int *at) {
    assert(count == 3);
    *at = 17;
    return 51;
}
'''


CHECKS = r'''
static void chunk_record(Chunk *c, int pool) {
    assert(c == temporal_hook_chunk && c->index == 0 && pool == RW.n);
    assert(Q.put == QN - 1 && Q.drained + 1 == Q.out);
    rw_chunk = c;
    assert(vk_exec_temporal_prepare(x_cur, prepare, &next_calls) == VK_SYNC_PROGRESS_COMPLETE);
    rw_chunk = NULL;
    c->cb = 17;
    ++first_chunks;
    sizing_open = 0;
}

static void check_group(int open, int closed_during_drain) {
    Group group = {.c = {{0, 0}, {1, 0}, {2, 0}}};
    VkExec ex = {0};
    x_cur = &ex;
    Q.out = 4;
    Q.drained = 3;
    Q.put = QN - 1;
    sizing_open = open;
    close_on_drain = closed_during_drain;
    unsigned earlier_drains = drains, earlier_parks = parks, earlier_chunks = first_chunks;
    assert(publish_test_group(&ex, &group, 3));
    assert(ex.cb == 51 && group.at == 17 && group.out_ns == 42);
    assert(!temporal_hook_chunk && !rw_chunk);
    int owner = open && !closed_during_drain;
    assert(drains - earlier_drains == (unsigned)open);
    assert(parks - earlier_parks == (unsigned)owner);
    assert(first_chunks - earlier_chunks == (unsigned)owner);
    assert(group.c[0].cb == (owner ? 17 : 0));
    assert(Q.put == QN + 2u - (unsigned)owner);
    assert(Q.q[QN - 1] == &group.c[owner] && Q.q[0] == &group.c[owner + 1]);
    if (!owner) assert(Q.q[1] == &group.c[2]);
}

int main(void) {
    check_group(1, 0);
    check_group(0, 0);
    check_group(1, 1);
    VkExec ex = {0}, other = {0};
    Chunk worker = {1, 0};
    x_cur = &ex;
    rw_chunk = &worker;
    assert(vk_exec_temporal_prepare(&ex, prepare, &next_calls) == VK_SYNC_PROGRESS_UNAVAILABLE);
    rw_chunk = NULL;
    assert(vk_exec_temporal_prepare(&other, prepare, &next_calls) == VK_SYNC_PROGRESS_UNAVAILABLE);
    assert(vk_exec_temporal_prepare(&ex, NULL, &next_calls) == VK_SYNC_PROGRESS_UNAVAILABLE);
    stopping = 1;
    assert(vk_exec_temporal_prepare(&ex, prepare, &next_calls) == VK_SYNC_PROGRESS_STOPPED);
    stopping = 0;
    assert(vk_exec_temporal_prepare(&ex, prepare, &next_calls) == VK_SYNC_PROGRESS_COMPLETE);
    assert(parks == 2 && next_calls == 2);
}
'''


def main():
    source = REPLAY.read_text(encoding="utf-8")
    group_start = function(source, "group_start")
    drain = between(group_start, "    if (vk_dlaa_temporal_hook_needs_recorder())", "    if (RW.pipe)")
    publish = between(group_start, "    ex->cb = vk_sync_append_later", "    for (int k = 0;")
    group = """
static int publish_test_group(VkExec *ex, Group *g, int nc) {
""" + drain.replace("return h;", "return 0;") + "    ++Q.out;\n" + publish + "    return 1;\n}\n"
    cc = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="temporal-recorder-frontier-") as tmp:
        c = Path(tmp) / "check.c"
        exe = c.with_suffix(".exe")
        c.write_text("\n".join((STUBS, function(source, "vk_exec_temporal_prepare"),
                               "static void chunk_record(Chunk *, int);", group, CHECKS)), encoding="utf-8")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", str(c), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True, timeout=20)
    print("Temporal recording: ordered first chunk, warm parallel publication and worker refusal passed")


if __name__ == "__main__":
    main()

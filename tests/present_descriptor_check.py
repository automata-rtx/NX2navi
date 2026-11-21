"""Host checks for capture of the guest's queued and held present descriptors."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def section(source, start, end):
    offset = source.index(start)
    return source[offset:source.index(end, offset)]


def function(source, name):
    match = re.search(r'^[A-Za-z_][^;\n]*\b' + name + r'\([^;{}]*\)\s*\{', source, re.M)
    if not match:
        raise RuntimeError('Missing production definition: ' + name)
    braces = 1
    end = match.end()
    while braces:
        braces += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[match.start():end] + '\n'


def fixture(source):
    queue = section(source, 'enum { RI_SUBMIT, RI_PRESENT, RI_RELEASE };', 'static const char *const dr_name')
    held = section(source, '#define PH_MAX ', 'static int present_held_on')
    body = ''.join(function(source, name) for name in
                   ('replay_frontend_begin', 'replay_frontend_end', 'present_held_on', 'held_resume',
                    'ph_flush', 'replay_main', 'replay_enqueue', 'replay_drain', 'vk_be_present'))
    return r'''
#include <assert.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nvn_api.h"
#include "resource_core_budget.h"

// Submit handle storage never exists in either of these present-only cases.
static void vk_replay_metadata_free(void *handle, RcBudgetTicket *budget) {
    assert(!handle && !budget->token.ledger);
}

typedef struct { int unused; } VkReplayCaptureBundle, VkTextureRegistrationScope;
static void captured_present(const tex_t *, const int [4], uint64_t, uint64_t, uint64_t, uint64_t);
#define present_image captured_present
static void *capture_alloc(size_t);
static void capture_free(void *);
#define malloc capture_alloc
#define free capture_free
''' + queue + held + r'''
#undef malloc
#undef free
static ReplayItem rq[RQ_CAP];
static int rl, rcv, rq_head, rq_n, r_busy, r_started, r_frontend_calls;
static uint64_t r_presents_queued, r_presents_done, r_submit_done, game_frames;
static uint64_t r_registration_issued, r_registration_done, r_registration_publication;
static uint64_t dr_calls[DR_N], dr_tcalls[DR_N], next_fence;
GfxStats gfx_counters;
static int vk_sub_in_frame;
uint64_t vk_exec_submit_seq;
static int async_mode, stopping, held_work, held_ready, worker_active;
static unsigned copy_count, release_count, worker_waits, shown, held_submits, threads;
static void *live_copy;
static jmp_buf queue_empty;
static void replay_main(void *);

static void *capture_alloc(size_t size) {
    assert(!rl && !live_copy && r_frontend_calls == 1 && size == sizeof(tex_t));
    live_copy = malloc(size);
    assert(live_copy);
    copy_count++;
    return live_copy;
}
static void capture_free(void *object) {
    if (!object) return;
    assert(!rl && worker_active && r_busy && object == live_copy);
    free(object);
    live_copy = NULL;
    release_count++;
}
static void run_replay_until_empty(void) {
    assert(!rl && !worker_active && rq_n);
    worker_active = 1;
    if (setjmp(queue_empty) == 0) {
        replay_main(NULL);
        abort();
    }
    assert(rl && !rq_n && !r_busy);
    rl = worker_active = 0;
}
static void plat_mutex_lock(int *lock) { assert(lock == &rl && !rl); rl = 1; }
static void plat_mutex_unlock(int *lock) { assert(lock == &rl && rl); rl = 0; }
static void plat_cond_broadcast(int *cond) { assert(rl && cond == &rcv); }
static void plat_cond_wait(int *cond, int *lock) {
    assert(rl && cond == &rcv && lock == &rl);
    if (worker_active) {
        assert(!rq_n && !r_busy);
        longjmp(queue_empty, 1);
    }
    worker_waits++;
    rl = 0;
    run_replay_until_empty();
    rl = 1;
}
static int plat_thread_spawn(void (*thread)(void *), void *arg, const char *name, int flags) {
    assert(rl && thread == replay_main && !arg && name && !flags);
    assert(!threads++);
    return 1;
}
int gfx_replay_async(void) { return async_mode; }
static int vk_sync_stopping(void) { return stopping; }
static int vk_device_is_lost(void) { return 0; }
void rt_log(const char *fmt, ...) { (void)fmt; }
void rt_fatal(const char *fmt, ...) { fputs(fmt, stderr); abort(); }
int rt_knob(const char *name, const char *value) { assert(name && value); return 1; }
uint64_t kern_now_ns(void) { static uint64_t now; return ++now; }
uint64_t perf_begin(void) { return 1; }
void perf_replay_thread(void) { assert(worker_active); }
void perf_present(uint64_t frame) { assert(r_frontend_calls && frame); }
static void motion_owner_present(void) { assert(r_frontend_calls); }
static void vk_sync_start(void) { assert(r_frontend_calls); }
void gfx_tl_present(void) { assert(r_frontend_calls); }
static void vk_dlssg_game_present(void) { assert(r_frontend_calls); }
static void vk_dlssg_render_start(uint64_t frame) { assert(!rl && frame); }
static void vk_exec_evt_frame(uint64_t frame) { assert(!rl && frame); }
static int vk_exec_held(void) { return held_work; }
static int vk_exec_held_ready(void) { return held_ready; }
static void vk_present_drain(void) { assert(!rl); }
static void replay_log(uint64_t frame, int final) { (void)frame; (void)final; abort(); }
static void wait_counted(int why, uint64_t t) { assert(rl && (why == DR_FRAME || why == DR_FINISH) && t); }
static int core_away_unlocked(void) { assert(rl); return 0; }
static void core_back_unlocked(int *away) { assert(rl && !*away); }
static uint64_t vk_sync_reserve(void) { assert(rl); return ++next_fence; }
static uint64_t submit_sync(const uint64_t *h, int n) {
    assert(!h && !n && held_ready && held_work);
    held_work = 0;
    vk_exec_submit_seq++;
    held_submits++;
    return 1;
}
static uint64_t submit_sync_capture(const uint64_t *h, int n, VkReplayCaptureBundle *cap,
                                    VkTextureRegistrationScope *reg) {
    (void)h; (void)n; (void)cap; (void)reg; abort();
}
static void vk_usnap_queue_bundle_cancel(VkReplayCaptureBundle *cap) { (void)cap; abort(); }
static void vk_texture_registration_scope_end(VkTextureRegistrationScope *scope) { (void)scope; abort(); }
static int vk_texture_registration_scope_current(VkTextureRegistrationScope *scope) { (void)scope; abort(); }
static uint64_t vk_texture_registration_scope_publication(VkTextureRegistrationScope *scope) { (void)scope; abort(); }
static void vk_texture_release_replayed(const tex_t *tex) { (void)tex; abort(); }

static struct {
    tex_t tex;
    int crop[4], has_crop, has_texture;
    uint64_t at, queued, frame, value;
} seen[8];
static void captured_present(const tex_t *t, const int crop[4], uint64_t at, uint64_t queued,
                             uint64_t frame, uint64_t value) {
    assert(!rl && !stopping && shown < sizeof seen / sizeof *seen && seen[shown].frame == 0);
    assert((async_mode && worker_active && r_busy) || (!async_mode && !worker_active));
    if (t) seen[shown].tex = *t;
    seen[shown].has_texture = t != NULL;
    if (crop) memcpy(seen[shown].crop, crop, sizeof seen[shown].crop);
    seen[shown].has_crop = crop != NULL;
    seen[shown].at = at, seen[shown].queued = queued, seen[shown].frame = frame, seen[shown].value = value;
    shown++;
}

#define malloc capture_alloc
#define free capture_free
''' + body + r'''
#undef malloc
#undef free

static pool_t first_pool, second_pool;
static tex_t texture(uint64_t serial) {
    tex_t t = {.serial = serial, .mem = (uint8_t *)(uintptr_t)(0x4000 + serial * 4096), .off = serial * 1024};
    t.pool = serial % 2 ? &first_pool : &second_pool;
    t.d.width = (unsigned)(640 + serial * 8);
    t.d.height = (unsigned)(360 + serial * 2);
    t.d.format = serial % 2 ? NVN_FORMAT_RGBA8 : NVN_FORMAT_R32UI;
    t.L.size = 4096 * serial;
    return t;
}
static void check_seen(unsigned index, const tex_t *expected, const int crop[4], uint64_t at, uint64_t value) {
    assert(shown > index && seen[index].frame == index + 1 && seen[index].queued);
    assert(seen[index].has_texture == (expected != NULL));
    if (expected) {
        const tex_t *actual = &seen[index].tex;
        if (actual->serial != expected->serial) {
            fprintf(stderr, "descriptor identity changed after capture, expected %llu, observed %llu\n",
                    (unsigned long long)expected->serial, (unsigned long long)actual->serial);
            exit(1);
        }
        assert(actual->mem == expected->mem && actual->pool == expected->pool && actual->off == expected->off);
        assert(actual->d.width == expected->d.width && actual->d.height == expected->d.height);
        assert(actual->d.format == expected->d.format && actual->L.size == expected->L.size);
    }
    assert(seen[index].has_crop == (crop != NULL));
    if (crop) assert(!memcmp(seen[index].crop, crop, sizeof seen[index].crop));
    assert(seen[index].at == at && seen[index].value == value);
}

static void queued(void) {
    async_mode = 1;
    const tex_t a = texture(1), b = texture(2), c = texture(3);
    const int crop_a[4] = {2, 3, 400, 225}, crop_b[4] = {7, 8, 300, 169};
    int crop[4];
    memcpy(crop, crop_a, sizeof crop);
    tex_t guest = a;
    vk_be_present(&guest, crop, 101);
    assert(rq_n == 1 && r_presents_queued == 1 && !r_presents_done && !shown && !r_frontend_calls);
    guest = b;
    memcpy(crop, crop_b, sizeof crop);
    // The real frontend frame bound drives the first queued worker before accepting the second.
    vk_be_present(&guest, crop, 102);
    assert(rq_n == 1 && r_presents_queued == 2 && r_presents_done == 1 && worker_waits == 1);
    check_seen(0, &a, crop_a, 101, 1);
    guest = c;
    memset(crop, 0, sizeof crop);
    replay_drain(DR_FINISH);
    check_seen(1, &b, crop_b, 102, 2);
    assert(!rq_n && !live_copy && copy_count == 2 && release_count == 2);
    vk_be_present(NULL, NULL, 103);
    replay_drain(DR_FINISH);
    check_seen(2, NULL, NULL, 103, 3);
    assert(copy_count == 2 && release_count == 2);
    vk_be_present(&guest, NULL, 104);
    assert(rq_n == 1 && live_copy && copy_count == 3 && release_count == 2);
    stopping = 1;
    replay_drain(DR_FINISH);
    assert(shown == 3 && !rq_n && r_presents_queued == 4 && r_presents_done == 4);
    assert(!live_copy && copy_count == release_count);
    vk_be_present(&guest, NULL, 105);
    assert(shown == 3 && !rq_n && copy_count == 3 && release_count == 3);
}

static void held_present(void) {
    const tex_t a = texture(10), b = texture(11), c = texture(12);
    const int crop_a[4] = {2, 3, 400, 225}, crop_b[4] = {7, 8, 300, 169};
    int crop[4];
    memcpy(crop, crop_a, sizeof crop);
    tex_t guest = a;
    held_work = 1;
    vk_be_present(&guest, crop, 201);
    assert(PH.on && !shown && !r_frontend_calls);
    guest = b;
    memcpy(crop, crop_b, sizeof crop);
    // A new still-held present flushes its predecessor, then captures its own descriptor.
    vk_be_present(&guest, crop, 202);
    assert(PH.on && PH.next == 1 && shown == 1);
    check_seen(0, &a, crop_a, 201, 0);
    guest = c;
    memset(crop, 0, sizeof crop);
    held_ready = 1;
    assert(held_resume());
    ph_flush(&PH.after);
    check_seen(1, &b, crop_b, 202, 0);
    ph_flush(&PH.after);
    assert(!PH.on && PH.after == 1 && shown == 2 && held_submits == 1);
    held_work = 1;
    held_ready = 0;
    vk_be_present(NULL, NULL, 203);
    assert(PH.on && shown == 2);
    ph_flush(&PH.capped);
    check_seen(2, NULL, NULL, 203, 0);
    assert(!PH.on && PH.capped == 1 && !rq_n && !copy_count && !release_count);
}

int main(int argc, char **argv) {
    assert(argc == 2);
    if (!strcmp(argv[1], "queued")) queued();
    else if (!strcmp(argv[1], "held")) held_present();
    else abort();
    assert(!r_submit_done && !r_registration_issued && !r_registration_done && !r_registration_publication);
    printf("Present descriptors: %s capture and lifetime passed\n", argv[1]);
}
'''


def main():
    source = (ROOT / 'runtime/vk_backend.c').read_text()
    cc = os.environ.get('CC') or shutil.which('clang')
    if not cc:
        raise SystemExit('Clang is required (set CC)')
    clang = Path(shutil.which(cc) or cc).resolve()
    env = dict(os.environ, PATH=str(clang.parent) + os.pathsep + os.environ.get('PATH', ''))
    with tempfile.TemporaryDirectory(prefix='present-descriptor-') as directory:
        c = Path(directory) / 'check.c'
        exe = c.with_suffix('.exe' if os.name == 'nt' else '.out')
        c.write_text(fixture(source))
        subprocess.run([str(clang), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function',
                        '-fsanitize=undefined', '-fsanitize-trap=undefined',
                        '-I' + str(ROOT / 'runtime'), str(c), '-o', str(exe)], env=env, check=True)
        for case in ('queued', 'held'):
            subprocess.run([str(exe), case], env=env, check=True)


if __name__ == '__main__':
    main()

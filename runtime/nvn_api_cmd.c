// NVN front end, command-buffer slice: command buffers, their memory, recording
// as op records (nvn_obj.h), recording copies behind command handles, and the submit-time resolution of deferred binds
// (nvn_submit_prepare, called by nvnQueueSubmitCommands in nvn_api_core.c), and the *Deferred and memory-callback entry
// points. Texture pins are counted (section "texture pins").
// Contains code Copyright (c) 2026 jhaizhou (MIT; see THIRD_PARTY_NOTICES.md).
//
// Records are written into a growable host buffer per command buffer, never into the game's command memory: games hand
// the same command memory to several command buffers recording at once (budgeted for real NVN encodings), and our
// records are larger. The game's command and control memory is accounted only. EndRecording copies the finished
// recording into host memory; the handle is that copy (a flat OP_BEGIN ... OP_END sequence) and lives until the game
// adds the same control memory again. CallCommands and CopyCommands inline the called recording, so copies hold no
// OP_CALL or OP_JUMP.
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include "replay_phase_trace.h"
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_hmap.h"
#include "nvn_exec_state_blob.h"
#include "resource_prep.h"
#include "platform/platform.h"

#define CMD_MAGIC MAGIC('C', 'M', 'D', 'B')
#define COPY_MAGIC 0x59504f43444d43ull     // "CMDCOPY": prefix of a live recording copy
#define OP_RESERVE 24                       // room always kept at the end of the recording buffer
#define MIN_COMMAND_SIZE 512                // NVN_DEVICE_INFO_COMMAND_BUFFER_MIN_COMMAND_SIZE (nvn_api_core.c devinfo)
#define MIN_CONTROL_SIZE 256                // NVN_DEVICE_INFO_COMMAND_BUFFER_MIN_CONTROL_SIZE
#define CTL_PER_RECORDING 64                // control memory one recording is reported to use
#define COPY_PAGE_SHIFT 16                  // recording copy index granularity (release_copies)

static cmd_t *cmd_of(const Cpu *c) { return arg_p(c, 0); }

// ---------- memory callbacks
// NVN calls the game's memory callback when a recording needs command or control memory and none is left. Records
// live in host memory here, so the callback is called only where the hardware certainly would: a recording starts
// without any command memory, or ends without any control memory. It runs on the calling guest thread through
// rt_call; whatever memory it adds is accounted. Set{Command,Control}MemoryCallbackEnabled(false) suppress it.
enum { CB_NO_COMMAND = 1, CB_NO_CONTROL = 2 };
static PlatMutex cb_lock = PLAT_MUTEX_INIT;
static nvn_hmap cb_off = {.klen = 8};       // command buffer -> disabled callback kinds | 0x100

static unsigned cb_disabled(const cmd_t *o) {
    plat_mutex_lock_shared(&cb_lock);
    uintptr_t v = (uintptr_t)nvn_hm_get(&cb_off, &o);
    plat_mutex_unlock_shared(&cb_lock);
    return (unsigned)v & 0xff;
}

static void cb_set_disabled(const cmd_t *o, unsigned kind, int disabled) {
    plat_mutex_lock(&cb_lock);
    unsigned v = (unsigned)(uintptr_t)nvn_hm_get(&cb_off, &o) & 0xff;
    v = disabled ? v | kind : v & ~kind;
    if (v) nvn_hm_put(&cb_off, &o, (void *)(uintptr_t)(v | 0x100));
    else nvn_hm_del(&cb_off, &o);
    plat_mutex_unlock(&cb_lock);
}

static void memory_event(Cpu *c, cmd_t *o, int event, uint64_t min) {
    if (!o->cb || (cb_disabled(o) & (event == NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY ? CB_NO_COMMAND
                                                                                                     : CB_NO_CONTROL)))
        return;
    NVN_LOG("command buffer %p: memory callback %p (%s memory, %llu bytes)", (void *)o, (void *)o->cb,
            event == NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY ? "command" : "control",
            (unsigned long long)min);
    rt_call(c, (uint64_t)(uintptr_t)o->cb, 4, (uint64_t)(uintptr_t)o, (uint64_t)event, min,
            (uint64_t)(uintptr_t)o->cb_data);
}

// ---------- recording buffer
static void put_hdr(uint8_t *p, uint32_t op, uint32_t size) {
    op_hdr h = {op, size};
    memcpy(p, &h, sizeof h);
}

// 1.4.3 initializes and finalizes command buffers about 2000 times a second while a scene loads and several times a
// frame in play, and its two largest per-frame recordings take 1-2 MiB. Freeing the recording buffer at Finalize made
// every new recording grow again from 64 KiB through up to five reallocations and copies, the last ones fresh 1-2 MiB
// blocks the heap takes from the system: new commit and zero-filled pages twice a frame. Those growths were most of the
// process's new commit requests, so when the system ran short for a moment (a Depths load at render scale 200 on 16 GB)
// a 1 MiB growth was the allocation that failed. Finalized buffers therefore wait in a small pool for the next
// recording, and a failed allocation waits for memory instead of ending the game (realloc_wait).
#define POOL_SLOTS 16
#define POOL_MAX_BYTES ((size_t)4 << 20)   // a larger buffer (a recording that inlined a big one) goes back to the heap
static PlatMutex pool_lock = PLAT_MUTEX_INIT;
static struct { uint8_t *p; size_t size; } pool[POOL_SLOTS];
static uint32_t npool;

// The largest pooled buffer and its size, or NULL: the largest, so the big per-frame recordings stop growing once the
// pool holds enough buffers of their size.
static uint8_t *pool_take(size_t *size) {
    uint8_t *p = NULL;
    plat_mutex_lock(&pool_lock);
    if (npool) {
        uint32_t best = 0;
        for (uint32_t i = 1; i < npool; i++)
            if (pool[i].size > pool[best].size) best = i;
        p = pool[best].p;
        *size = pool[best].size;
        pool[best] = pool[--npool];
    }
    plat_mutex_unlock(&pool_lock);
    return p;
}

// Keeps a finalized buffer when the pool has room or holds a smaller one (which is freed instead).
static void pool_put(uint8_t *p, size_t size) {
    if (!p) return;
    if (size <= POOL_MAX_BYTES) {
        plat_mutex_lock(&pool_lock);
        if (npool < POOL_SLOTS) {
            pool[npool].p = p;
            pool[npool++].size = size;
            p = NULL;
        } else {
            uint32_t small = 0;
            for (uint32_t i = 1; i < npool; i++)
                if (pool[i].size < pool[small].size) small = i;
            if (pool[small].size < size) {
                uint8_t *drop = pool[small].p;
                pool[small].p = p;
                pool[small].size = size;
                p = drop;
            }
        }
        plat_mutex_unlock(&pool_lock);
    }
    free(p);
}

static void pool_drain(void) {
    plat_mutex_lock(&pool_lock);
    while (npool) free(pool[--npool].p);
    plat_mutex_unlock(&pool_lock);
}

// realloc that waits out a short commit shortage: Windows refuses commit beyond the current limit while it grows a
// system-managed page file, so the same request succeeds a moment later, and a stall of some frames is better than
// ending the game. Frees the pooled buffers first. NULL only after about 30 s without memory; the caller ends the game.
// Never called with a lock held.
static void *realloc_wait(void *old, size_t n, const char *what) {
    void *p = realloc(old, n);
    if (p) return p;
    pool_drain();
    static uint32_t reports;
    int report = __atomic_fetch_add(&reports, 1, __ATOMIC_RELAXED) < 8;
    if (report) {
        double proc, used, limit;
        plat_commit_info(&proc, &used, &limit);
        rt_log("nvn: %s of %zu bytes failed (memory commit: this process %.0f MB, the system %.0f of %.0f MB); waiting "
               "for memory", what, n, proc, used, limit);
    }
    uint64_t t0 = plat_now_ms();
    while (plat_now_ms() - t0 < 30000) {
        plat_sleep_ms(10);
        if ((p = realloc(old, n))) {
            if (report)
                rt_log("nvn: %s of %zu bytes succeeded after %llu ms", what, n,
                       (unsigned long long)(plat_now_ms() - t0));
            return p;
        }
    }
    return NULL;
}

static void cmd_reserve(cmd_t *o, size_t need) {
    if (o->seg && o->cur + need + OP_RESERVE <= o->end) return;
    if (!o->seg) {   // the first record since Initialize: a pooled buffer when one waits
        size_t size = 0;
        uint8_t *p = pool_take(&size);
        if (p) {
            o->seg = o->cur = p;
            o->end = p + size;
            if (need + OP_RESERVE <= size) return;
        }
    }
    size_t used = o->seg ? (size_t)(o->cur - o->seg) : 0;
    size_t cap = o->seg ? (size_t)(o->end - o->seg) : 0;
    if (!cap) cap = 64 << 10;
    while (used + need + OP_RESERVE > cap) cap *= 2;
    uint8_t *b = realloc_wait(o->seg, cap, "command recording buffer");
    // Still failing after the wait: the system has no commit left (the fatal report states it), or seg is no block of
    // ours: a command buffer object the game overwrote or copied shows a wrong magic or more used than its old size.
    if (!b)
        rt_fatal("nvn: command recording buffer of %zu bytes (command buffer %p, magic %#x, buffer %p of %zu bytes, %zu "
                 "used, %zu asked)", cap, (void *)o, o->magic, (void *)o->seg, o->seg ? (size_t)(o->end - o->seg) : 0,
                 used, need);
    o->seg = b;
    o->cur = b + used;
    o->end = b + cap;
}

// Space for one record (header included); the header's op is filled by the caller.
static uint8_t *cmd_alloc(cmd_t *o, uint32_t size) {
    size = (size + 7) & ~7u;
    cmd_reserve(o, size);
    uint8_t *p = o->cur;
    o->cur += size;
    o->cmd_used += size;
    o->hw_used += size;
    o->last_rec = size;
    put_hdr(p, 0, size);
    return p;
}

// The game sums nvnCommandBufferGetCommandMemoryUsed over recordings to size resource heaps, so usage must be what the
// real NVN push-buffer encoding takes, not our record size. Known encodings (Maxwell methods, from a real recording):
// DrawElements 12 words, DrawElementsBaseVertex 11, BindVertexAttribState 1+16, BindVertexStreamState 16+n,
// BindVertexBuffer 6 words each.
static void hw_fix(cmd_t *o, size_t bytes) {
    o->hw_used += bytes;
    o->hw_used -= o->last_rec;
    o->last_rec = bytes;
}

// A zeroed payload of one record.
static void *rec(cmd_t *o, uint32_t op, size_t payload) {
    uint32_t sz = (uint32_t)((8 + payload + 7) & ~(size_t)7);
    uint8_t *p = cmd_alloc(o, sz);
    put_hdr(p, op, sz);
    memset(p + 8, 0, sz - 8);
    return p + 8;
}
#define REC(o, OP, T) ((T *)rec((o), (OP), sizeof(T)))
// Array-carrying binds take the prefix fields plus n elements; the replay reads only the first n.
#define RECN(o, OP, T, ARR, N) ((T *)rec((o), (OP), offsetof(T, ARR) + sizeof(((T *)0)->ARR[0]) * (size_t)(N)))

static void copy_view(view_t *dst, uint8_t *has, const NVNtextureView *v) {
    *has = v != NULL;
    if (v) *dst = *(const view_t *)v;
}

// ---------- texture pins
// Recordings refer to textures by pointer, but the game may finalize or reuse an NVNtexture after recording and before
// the submit (NVN itself captured only GPU state). Records therefore point at an immutable host copy per texture
// initialization (tex_t.serial). The game initializes its transient targets again every frame, about 29 new serials a
// frame on 1.4.3, so copies kept for the run grew the heap by about 44 MiB per minute of play, standing still too, until
// the commit limit ended the game after a few hours. A copy is freed once nothing can read it any more:
// - refs counts the recording copies whose ops name it: live ones, released ones a queued submission still reads, and
//   per-submission clones (pin_refs, at flatten_recording and wherever such a copy is freed or cloned);
// - a recording in progress holds pins no copy counts yet, through its own tex_pin calls and through ops inlined from a
//   copy that was released since: a pin goes only NVN_PIN_AGE submits after its last tex_pin and after refs fell to 0;
// - the replay's current render targets carry over to later submissions and are compared by pointer (vk_exec.c set_rt),
//   so a freed address reused by a new pin would read as the same target: nvn_pins_sweep runs between submissions on
//   the thread that replays them and keeps the pins it is given.
// Instead of marking pins by walking every live recording each window, counting at copy and release walks each
// copy's ops once more, when it is freed. NX2NAVI_PIN_SWEEP=0 keeps every copy.
typedef struct {
    tex_t t;              // first: ops hold pins as tex_t pointers
    uint64_t last_use;    // nsubmits at the last tex_pin or the last drop of refs
    uint32_t refs;
    uint8_t queued;       // in pin_q
} pin_t;
_Static_assert(offsetof(pin_t, t) == 0, "ops hold pins as tex_t pointers");
typedef struct { pin_t *p; uint64_t at; } pin_due;
static PlatMutex pin_lock = PLAT_MUTEX_INIT;   // lock order: copy_lock, then pin_lock
static nvn_hmap pins = {.klen = 8};         // serial -> pin_t
static pin_due *pin_q;                      // pins with no counted copy, oldest first: [pin_qh, pin_qn)
static size_t pin_qh, pin_qn, pin_qcap;
static uint64_t pins_live, pins_freed;
static uint64_t nsubmits;                   // QueueSubmitCommands calls so far

static void pin_queue(pin_t *p, uint64_t at) {   // pin_lock held
    if (p->queued) return;
    if (pin_qh && pin_qh * 2 >= pin_qn) {
        memmove(pin_q, pin_q + pin_qh, sizeof *pin_q * (pin_qn - pin_qh));
        pin_qn -= pin_qh, pin_qh = 0;
    }
    if (pin_qn == pin_qcap) {
        pin_qcap = pin_qcap ? pin_qcap * 2 : 4096;
        if (!(pin_q = realloc(pin_q, sizeof *pin_q * pin_qcap))) rt_fatal("nvn: out of memory (texture pin queue)");
    }
    pin_q[pin_qn++] = (pin_due){p, at};
    p->queued = 1;
}

static const tex_t *tex_pin(const NVNtexture *nt) {
    const tex_t *t = (const tex_t *)nt;
    if (!t) return NULL;
    uint64_t serial = t->serial;
    if (!serial) rt_fatal("nvn: a recording references the uninitialized texture object %p", (const void *)t);
    uint64_t now = __atomic_load_n(&nsubmits, __ATOMIC_RELAXED);
    plat_mutex_lock_shared(&pin_lock);   // nvn_pins_sweep frees under the exclusive lock
    pin_t *r = nvn_hm_get(&pins, &serial);
    if (r) __atomic_store_n(&r->last_use, now, __ATOMIC_RELAXED);
    plat_mutex_unlock_shared(&pin_lock);
    if (r) return &r->t;
    plat_mutex_lock(&pin_lock);
    r = nvn_hm_get(&pins, &serial);
    if (!r) {
        if (!(r = malloc(sizeof *r))) rt_fatal("nvn: texture pin copy");
        *r = (pin_t){.t = *t};
        nvn_hm_put(&pins, &serial, r);
        pins_live++;
        pin_queue(r, now);   // freed like any other if no copy ever names it
    }
    __atomic_store_n(&r->last_use, now, __ATOMIC_RELAXED);
    plat_mutex_unlock(&pin_lock);
    return &r->t;
}

// The ops whose payload names pins (op_pins): five of the op kinds. The walks over every record of a recording
// (flatten_recording, pin_refs) test this mask before they call op_pins.
#define PIN_OPS (1ull << OP_SET_RT | 1ull << OP_COPY_B2T | 1ull << OP_COPY_T2B | 1ull << OP_CLEAR_TEX | 1ull << OP_COPY_T2T)
_Static_assert(OP_COUNT_ <= 64, "PIN_OPS holds one bit per op kind");

// The pins an op's payload names, into out (at most 9); returns how many.
static int op_pins(uint32_t op, const void *a, const tex_t *out[9]) {
    int k = 0;
    switch (op) {
    case OP_SET_RT: {
        const p_set_rt *r = a;
        int n = r->n < 0 ? 0 : r->n > 8 ? 8 : r->n;
        for (int i = 0; i < n; i++)
            if (r->color[i]) out[k++] = r->color[i];
        if (r->depth) out[k++] = r->depth;
        break;
    }
    case OP_COPY_B2T: case OP_COPY_T2B:
        if (((const p_copy_b2t *)a)->tex) out[k++] = ((const p_copy_b2t *)a)->tex;
        break;
    case OP_CLEAR_TEX:
        if (((const p_clear_tex *)a)->tex) out[k++] = ((const p_clear_tex *)a)->tex;
        break;
    case OP_COPY_T2T: {
        const p_copy_t2t *c = a;
        if (c->src) out[k++] = c->src;
        if (c->dst) out[k++] = c->dst;
        break;
    }
    default: break;
    }
    return k;
}

static void pin_ref(const tex_t *t, int delta, uint64_t now) {
    pin_t *p = (pin_t *)(uintptr_t)t;
    if (delta > 0) {
        __atomic_add_fetch(&p->refs, 1, __ATOMIC_RELAXED);
        return;
    }
    __atomic_store_n(&p->last_use, now, __ATOMIC_RELAXED);
    if (__atomic_sub_fetch(&p->refs, 1, __ATOMIC_ACQ_REL)) return;
    plat_mutex_lock(&pin_lock);
    pin_queue(p, now);
    plat_mutex_unlock(&pin_lock);
}

// Counts (delta 1) or uncounts (-1) the pins named by the copy at handle, OP_BEGIN to OP_END; returns how many.
static uint32_t pin_refs(const uint8_t *copy, int delta) {
    uint64_t now = __atomic_load_n(&nsubmits, __ATOMIC_RELAXED);
    uint32_t n = 0;
    for (const op_hdr *h = nvn_op_first((uintptr_t)copy); h->op != OP_END; h = nvn_op_next(h)) {
        if (h->op >= 64 || !(PIN_OPS >> h->op & 1)) continue;
        const tex_t *p[9];
        int k = op_pins(h->op, nvn_op_payload(h), p);
        for (int i = 0; i < k; i++) pin_ref(p[i], delta, now);
        n += (uint32_t)k;
    }
    return n;
}

void nvn_pins_sweep(uint64_t now, const tex_t *const *keep, int nkeep) {
    RP_SCOPE(rp_pins, RP_PIN_SWEEP);
    if (now < NVN_PIN_AGE) { RP_VALUE(RPK_OUTCOME, RPO_NO_WORK); return; }
    static int on = -1;
    if (on < 0) {
        const char *s = cfg_get("NX2NAVI_PIN_SWEEP");
        on = !(s && !strcmp(s, "0"));
        if (!on) rt_log("nvn: NX2NAVI_PIN_SWEEP=0: texture pin copies are kept for the run (the heap grows in long sessions)");
    }
    if (!on) { RP_VALUE(RPK_OUTCOME, RPO_DISABLED); return; }
    static uint64_t logged;
    plat_mutex_lock(&pin_lock);
#if NX2NAVI_REPLAY_PHASES
    uint64_t rp_freed = pins_freed;
#endif
    for (int budget = 1024; pin_qh < pin_qn && budget; budget--) {
        pin_due d = pin_q[pin_qh];
        if (d.at > now - NVN_PIN_AGE) break;
        pin_qh++;
        pin_t *p = d.p;
        p->queued = 0;
        if (__atomic_load_n(&p->refs, __ATOMIC_ACQUIRE)) continue;   // queued again when the count falls to 0
        uint64_t last = __atomic_load_n(&p->last_use, __ATOMIC_RELAXED);
        int kept = 0;
        for (int i = 0; i < nkeep; i++) kept |= keep[i] == &p->t;
        if (kept || last > now - NVN_PIN_AGE) {
            pin_queue(p, kept ? now : last);
            continue;
        }
        nvn_hm_del(&pins, &p->t.serial);
        free(p);
        pins_live--, pins_freed++;
    }
    if (now / 7200 != logged) {
        logged = now / 7200;
        rt_log("nvn: texture pins at submit %llu: %llu live, %llu waiting, %llu freed so far", (unsigned long long)now,
               (unsigned long long)pins_live, (unsigned long long)(pin_qn - pin_qh), (unsigned long long)pins_freed);
    }
    RP_VALUE(RPK_PINS_FREED, pins_freed - rp_freed);
    plat_mutex_unlock(&pin_lock);
}

void nvn_pins_stats(uint64_t *live, uint64_t *freed) {
    plat_mutex_lock_shared(&pin_lock);
    *live = pins_live, *freed = pins_freed;
    plat_mutex_unlock_shared(&pin_lock);
}

// ---------- recording copies
// A handle points just past a copy_prefix. Copies are indexed by the 64 KiB block of their control-memory position, so
// adding control memory again releases the copies recorded there without walking every live copy. 4 KiB pages cost
// about 500 lookups per AddControlMemory under the exclusive lock (1.4.3 field: 53 calls a frame, 2 MiB ranges, 3.7
// copies per page), and the recording threads waited on it.
// deferred: 0 unknown, 1 none, 2 some; pins: texture pins its ops name, counted in their refs (pin_refs)
typedef struct {
    uint64_t magic, bytes, made_at, used_at, deferred, pins;
    // Immutable offsets into the command bytes, stored after the OP_END record. Clones copy this tail as well, so
    // their deferred payloads can change without rebasing any metadata or sharing mutable state with another submit.
    uint32_t indexed, plan_complete;
    size_t nuniform, ndeferred, npinops, ncapture, nspans, nexec;
    uint32_t nfront, plan_readers;  // front plan is limited to 16 MiB; readers are guarded by copy_lock
    size_t state_bytes;
} copy_prefix;
_Static_assert(sizeof(copy_prefix) % 8 == 0, "records stay 8-byte aligned");
typedef struct { uintptr_t lo, hi; uint8_t *copy; } cmd_copy;
typedef struct { cmd_copy *v; uint32_t n, cap; } copy_bucket;
static PlatMutex copy_lock = PLAT_MUTEX_INIT;    // guards the two maps, the counters and every prefix
static nvn_hmap copy_pages = {.klen = 8};   // page -> copy_bucket
static nvn_hmap copy_live = {.klen = 8};    // handle -> copy_prefix of a live copy
// Includes retired copies and submission clones until replay releases them. Looking up here also makes raw command
// sequences used by backend probes safe: no access before an unrecognized handle to guess whether it has a prefix.
static nvn_hmap copy_indexed = {.klen = 8};
static uint64_t ncopies, copy_bytes, copies_made;
// Asynchronous replay: copies released by the game, and per-submission clones, that a queued submission still reads.
// Freed once gfx_replayed_submits() reaches their used_at. Empty with synchronous replay.
static copy_prefix **zombies;
static uint32_t nzombies, cap_zombies;

static const size_t *copy_offsets(const copy_prefix *pf) {
    return (const size_t *)((const uint8_t *)(pf + 1) + pf->bytes);
}

static size_t copy_size(const copy_prefix *pf) {
    return sizeof *pf + (size_t)pf->bytes +
           (pf->nuniform + pf->ndeferred + pf->npinops + pf->ncapture) * sizeof(size_t) +
           pf->nspans * sizeof(NvnRecordSpan) + pf->nexec * sizeof(NvnExecStep) +
           pf->nfront * sizeof(NvnFrontStep) + pf->state_bytes;
}

static uint32_t copy_pin_refs(const copy_prefix *pf, int delta) {
    const uint8_t *cmd = (const uint8_t *)(pf + 1);
    if (!pf->indexed) return pin_refs(cmd, delta);
    const size_t *off = copy_offsets(pf) + pf->nuniform + pf->ndeferred;
    uint64_t now = __atomic_load_n(&nsubmits, __ATOMIC_RELAXED);
    uint32_t n = 0;
    for (size_t j = 0; j < pf->npinops; j++) {
        const op_hdr *h = (const op_hdr *)(cmd + off[j]);
        const tex_t *p[9];
        int k = op_pins(h->op, nvn_op_payload(h), p);
        for (int i = 0; i < k; i++) pin_ref(p[i], delta, now);
        n += (uint32_t)k;
    }
    return n;
}

int nvn_record_uniform_offsets(uint64_t handle, const size_t **offsets, size_t *count) {
    plat_mutex_lock_shared(&copy_lock);
    const copy_prefix *pf = nvn_hm_get(&copy_indexed, &handle);
    if (pf) *offsets = copy_offsets(pf), *count = pf->nuniform;
    plat_mutex_unlock_shared(&copy_lock);
    // The caller holds the same submission lifetime that protects reading the command bytes themselves.
    return pf != NULL;
}

int nvn_record_plan(uint64_t handle, NvnRecordPlan *out) {
    *out = (NvnRecordPlan){0};
    plat_mutex_lock_shared(&copy_lock);
    const copy_prefix *pf = nvn_hm_get(&copy_indexed, &handle);
    if (pf) {
        const size_t *capture = copy_offsets(pf) + pf->nuniform + pf->ndeferred + pf->npinops;
        const NvnRecordSpan *spans = (const NvnRecordSpan *)(capture + pf->ncapture);
        const NvnExecStep *exec = (const NvnExecStep *)(spans + pf->nspans);
        const NvnFrontStep *front = (const NvnFrontStep *)(exec + pf->nexec);
        *out = (NvnRecordPlan){.commands = (const uint8_t *)(pf + 1), .bytes = (size_t)pf->bytes,
                              .capture_offsets = capture, .ncapture = pf->ncapture, .spans = spans,
                              .nspans = pf->nspans, .exec = exec, .nexec = pf->nexec,
                              .front = front, .nfront = pf->nfront,
                              .state_data = (const uint8_t *)(front + pf->nfront), .state_bytes = pf->state_bytes};
    }
    plat_mutex_unlock_shared(&copy_lock);
    return pf != NULL;
}

static void zombie_add(copy_prefix *pf) {   // copy_lock held
    if (nzombies == cap_zombies) {
        cap_zombies = cap_zombies ? cap_zombies * 2 : 64;
        zombies = realloc(zombies, sizeof *zombies * cap_zombies);
        if (!zombies) rt_fatal("nvn: out of memory (released recording copies)");
    }
    zombies[nzombies++] = pf;
}

static void copy_free(copy_prefix *pf) {   // copy_lock not held
    if (pf->indexed) {
        uint64_t handle = (uintptr_t)(pf + 1);
        plat_mutex_lock(&copy_lock);
        nvn_hm_del(&copy_indexed, &handle);
        plat_mutex_unlock(&copy_lock);
    }
    if (pf->pins) copy_pin_refs(pf, -1);
    free(pf);
}

// Released copies nothing reads any more are freed once copy_lock is dropped: uncounting a copy's pins walks its ops,
// and that walk and free() under the exclusive lock held up every thread that was recording or submitting (the
// main-loop thread waited there for 2.8% of its time). doom chains a copy for that, through made_at, which nothing
// reads after the release; doom_free frees the chain.
static copy_prefix *doom(copy_prefix *list, copy_prefix *pf) {   // copy_lock held
    pf->made_at = (uint64_t)(uintptr_t)list;
    return pf;
}

static void doom_free(copy_prefix *list) {   // copy_lock not held
    while (list) {
        copy_prefix *next = (copy_prefix *)(uintptr_t)list->made_at;
        copy_free(list);
        list = next;
    }
}

static copy_prefix *zombie_sweep(uint64_t replayed, copy_prefix *doomed) {   // copy_lock held; returns the doom chain
    for (uint32_t i = 0; i < nzombies;) {
        if (!zombies[i]->plan_readers && zombies[i]->used_at <= replayed) {
            doomed = doom(doomed, zombies[i]);
            zombies[i] = zombies[--nzombies];
        } else {
            i++;
        }
    }
    return doomed;
}

static copy_prefix *copy_find(uint64_t handle) {   // copy_lock held
    return nvn_hm_get(&copy_live, &handle);
}

int nvn_handle_live(uint64_t handle) {
    plat_mutex_lock_shared(&copy_lock);
    int live = copy_find(handle) != NULL;
    plat_mutex_unlock_shared(&copy_lock);
    return live;
}

uint64_t nvn_handle_bytes(uint64_t handle) {
    plat_mutex_lock_shared(&copy_lock);
    const copy_prefix *pf = copy_find(handle);
    uint64_t n = pf ? pf->bytes : 0;
    plat_mutex_unlock_shared(&copy_lock);
    return n;
}

// Marks a live copy used by submission now; its size from OP_BEGIN to OP_END inclusive.
#define PLAN_CHILD_MIN (16u << 10)
static inline uint64_t copy_touch_id(uint64_t handle, uint64_t now, const char *what, copy_prefix **reader) {
    plat_mutex_lock_shared(&copy_lock);
    copy_prefix *pf = copy_find(handle);
    uint64_t bytes = 0;
    if (pf) __atomic_store_n(&pf->used_at, now, __ATOMIC_RELAXED), bytes = pf->bytes;
    if (reader) {
        *reader = pf && pf->nexec && bytes >= PLAN_CHILD_MIN + 2 * sizeof(op_hdr) ? pf : NULL;
        if (*reader) __atomic_add_fetch(&pf->plan_readers, 1, __ATOMIC_RELAXED);
    }
    plat_mutex_unlock_shared(&copy_lock);
    if (!pf) rt_fatal("nvn: %s: command handle %#llx is not a live recording (released or never recorded)", what,
                      (unsigned long long)handle);
    return bytes;
}

static uint64_t copy_touch(uint64_t handle, uint64_t now, const char *what) {
    return copy_touch_id(handle, now, what, NULL);
}

typedef struct { size_t *v, n, cap; } offset_list;
typedef struct { NvnRecordSpan *v; size_t n, cap; } span_list;
typedef struct {
    NvnExecStep *steps;
    NvnFrontStep *front;
    uint8_t *state;
    size_t n, cap, bytes, state_cap;
    size_t nfront, front_cap;
    int failed;
    unsigned emits;
} exec_builder;
typedef struct { p_bind_buf binding; uint64_t stamp; } capture_slot;
typedef struct { capture_slot *slots; size_t cap, used; uint64_t stamp; } capture_set;
typedef struct PlanScratch {
    struct PlanScratch *next;
    offset_list uniform, deferred, pinops, capture;
    span_list spans;
    exec_builder exec;
    capture_set captured;
} PlanScratch;
static PlatMutex plan_scratch_lock = PLAT_MUTEX_INIT;
static PlanScratch *plan_scratch_free;
static uint32_t plan_scratch_count;
static uint64_t plan_scratch_bytes;

static size_t plan_scratch_size(const PlanScratch *s) {
    return sizeof *s + (s->uniform.cap + s->deferred.cap + s->pinops.cap + s->capture.cap) * sizeof(size_t) +
           s->spans.cap * sizeof(NvnRecordSpan) + s->captured.cap * sizeof(capture_slot) +
           s->exec.cap * sizeof(NvnExecStep) + s->exec.front_cap * sizeof(NvnFrontStep) + s->exec.state_cap;
}

static PlanScratch *plan_scratch_get(void) {
    plat_mutex_lock(&plan_scratch_lock);
    PlanScratch *s = plan_scratch_free;
    if (s) {
        plan_scratch_free = s->next;
        plan_scratch_count--;
        plan_scratch_bytes -= plan_scratch_size(s);
    }
    plat_mutex_unlock(&plan_scratch_lock);
    if (!s && !(s = calloc(1, sizeof *s))) rt_fatal("nvn: command plan scratch allocation");
    return s;
}

// Scratch belongs to one builder only while EndRecording runs. A bounded shared pool survives guest worker churn
// without abandoning thread-local allocations on every exit; unusually large recordings do not permanently grow it.
static void plan_scratch_put(PlanScratch *s) {
    size_t bytes = plan_scratch_size(s);
    plat_mutex_lock(&plan_scratch_lock);
    if (plan_scratch_count < 8 && bytes <= (4u << 20)) {
        s->next = plan_scratch_free, plan_scratch_free = s;
        plan_scratch_count++, plan_scratch_bytes += bytes;
        s = NULL;
    }
    plat_mutex_unlock(&plan_scratch_lock);
    if (!s) return;
    free(s->uniform.v);
    free(s->deferred.v);
    free(s->pinops.v);
    free(s->capture.v);
    free(s->spans.v);
    free(s->exec.steps);
    free(s->exec.front);
    free(s->exec.state);
    free(s->captured.slots);
    free(s);
}

static uint64_t capture_hash(const p_bind_buf *b) {
    uint64_t x = b->addr ^ (b->size * UINT64_C(0x9e3779b97f4a7c15));
    x ^= ((uint64_t)(uint32_t)b->stage << 32) | (uint32_t)b->index;
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static void capture_reset(capture_set *s) {
    s->used = 0;
    if (!++s->stamp) {
        if (s->cap) memset(s->slots, 0, s->cap * sizeof *s->slots);
        s->stamp = 1;
    }
}

// Deduplication concerns capture requests, never command execution: every original bind is still replayed. Keep
// stage/index as well as address/size because motion reconstruction can request neighbouring blocks for some banks.
static int capture_add(capture_set *s, const p_bind_buf *b) {
    if (!s->cap || s->used >= s->cap / 2) {
        if (s->cap > SIZE_MAX / 2 / sizeof *s->slots) rt_fatal("nvn: uniform capture plan too large");
        size_t cap = s->cap ? s->cap * 2 : 128;
        capture_slot *slots = calloc(cap, sizeof *slots);
        if (!slots) rt_fatal("nvn: uniform capture plan allocation");
        for (size_t i = 0; i < s->cap; i++) {
            if (s->slots[i].stamp != s->stamp) continue;
            size_t at = (size_t)capture_hash(&s->slots[i].binding) & (cap - 1);
            while (slots[at].stamp == s->stamp) at = (at + 1) & (cap - 1);
            slots[at] = s->slots[i];
        }
        free(s->slots);
        s->slots = slots, s->cap = cap;
    }
    size_t at = (size_t)capture_hash(b) & (s->cap - 1);
    while (s->slots[at].stamp == s->stamp) {
        const p_bind_buf *k = &s->slots[at].binding;
        if (k->addr == b->addr && k->size == b->size && k->stage == b->stage && k->index == b->index) return 0;
        at = (at + 1) & (s->cap - 1);
    }
    s->slots[at] = (capture_slot){*b, s->stamp};
    s->used++;
    return 1;
}

// Isolated draws stay in ordinary spans. Only a run long enough to amortize two extra boundaries retains stride
// metadata. The last span can be a short candidate until the next operation reveals whether that run continues.
static void span_finish_draw(span_list *s) {
    if (!s->n) return;
    NvnRecordSpan *last = &s->v[s->n - 1];
    if (!last->draw_stride || last->ops >= 4) return;
    last->draw_stride = 0;
    if (s->n > 1) {
        NvnRecordSpan *prev = last - 1;
        if (prev->kind == NVN_PLAN_RUN && !prev->draw_stride && prev->ops <= 1024 - last->ops) {
            prev->end = last->end, prev->ops += last->ops, prev->draws += last->draws;
            s->n--;
        }
    }
}

static void span_add(span_list *s, const op_hdr *h, size_t at) {
    uint32_t kind = nvn_record_op_class(h->op);
    uint32_t draw = h->op == OP_DRAW || h->op == OP_DRAW_IDX || h->op == OP_DRAW_IDX_IND || kind == NVN_PLAN_MESH;
    uint32_t stride = draw && kind == NVN_PLAN_RUN ? h->size : 0;
    if (s->n && s->v[s->n - 1].draw_stride != stride) span_finish_draw(s);
    NvnRecordSpan *last = s->n ? &s->v[s->n - 1] : NULL;
    // Bounded ordinary spans keep the replay's finite read-ahead limit exact. State-dependent boundaries are
    // single-operation spans, so replay still inspects the target, mesh mode and update order at their original place.
    if (kind == NVN_PLAN_RUN && last && last->kind == kind && last->draw_stride == stride && last->ops < 1024) {
        last->end = at + h->size;
        last->ops++;
        last->draws += draw;
        return;
    }
    if (s->n == s->cap) {
        if (s->cap > SIZE_MAX / 2 / sizeof *s->v) rt_fatal("nvn: replay plan too large");
        size_t cap = s->cap ? 2 * s->cap : 64;
        NvnRecordSpan *v = realloc_wait(s->v, cap * sizeof *v, "command replay plan");
        if (!v) rt_fatal("nvn: command replay plan allocation");
        s->v = v, s->cap = cap;
    }
    s->v[s->n++] = (NvnRecordSpan){at, at + h->size, 1, draw, kind, stride};
}

static void offset_add(offset_list *a, size_t off) {
    if (a->n == a->cap) {
        size_t cap = a->cap ? 2 * a->cap : 64;
        size_t *v = realloc_wait(a->v, cap * sizeof *v, "command resource index");
        if (!v) rt_fatal("nvn: command resource index allocation");
        a->v = v, a->cap = cap;
    }
    a->v[a->n++] = off;
}

static int op_deferred(const op_hdr *h) {
    if (h->op != OP_BIND_TEX && h->op != OP_BIND_IMG && h->op != OP_BIND_SAMP) return 0;
    const p_bind_handles *b = nvn_op_payload(h);
    return b->src && b->n > 0 && b->n <= 32;
}

// Optional execution metadata has a bounded per-recording footprint. Failure discards the complete executable
// index, leaving the validated command stream and its existing resource indexes usable by the reference walker.
#define EXEC_PLAN_BYTES (16u << 20)
static void exec_emit(void *user, const void *bytes, size_t n) {
    exec_builder *b = user;
    b->emits++;
    if (b->failed) return;
    if (!n || n > EXEC_PLAN_BYTES - 7 || b->bytes > EXEC_PLAN_BYTES - ((n + 7) & ~(size_t)7)) {
        b->failed = 1;
        return;
    }
    size_t padded = (n + 7) & ~(size_t)7, need = b->bytes + padded;
    if (need > b->state_cap) {
        size_t cap = b->state_cap ? b->state_cap : 4096;
        while (cap < need) cap *= 2;
        uint8_t *p = realloc(b->state, cap);
        if (!p) { b->failed = 1; return; }
        b->state = p, b->state_cap = cap;
    }
    memcpy(b->state + b->bytes, bytes, n);
    memset(b->state + b->bytes + n, 0, padded - n);
    b->bytes = need;
}

static void exec_build(exec_builder *b, const uint8_t *commands, size_t bytes) {
    b->n = b->nfront = b->bytes = 0, b->failed = 0;
    for (size_t at = 0; at < bytes && !b->failed;) {
        const op_hdr *h = (const op_hdr *)(commands + at);
        NvnExecStep step = {.begin = at, .end = at + h->size, .ops = 1, .opcode = h->op};
        NvnExecStateSpan span = {0};
        size_t state_at = b->bytes;
        b->emits = 0;
        if ((nvn_record_exec_flags(h->op) & NVN_EXEC_STATE_INPUT) &&
            nvn_exec_state_compile(commands, at, bytes, exec_emit, b, &span)) {
            if (b->failed) break;
            if (b->emits != 1 || span.end <= at || span.end > bytes || (span.end & 7) || !span.ops ||
                span.ops > (span.end - at) / sizeof(op_hdr))
                rt_fatal("nvn: invalid compiled state interval");
            step.end = span.end, step.ops = span.ops, step.opcode = NVN_EXEC_STATE;
            step.state_offset = (uint32_t)state_at, step.state_bytes = (uint32_t)(b->bytes - state_at);
        } else {
            if (b->emits) rt_fatal("nvn: state compiler emitted an unsupported interval");
            step.run = gfx_exec_prepared_handler(h->op);
        }
        step.flags = nvn_record_exec_flags(step.opcode);
        if (b->bytes > EXEC_PLAN_BYTES || b->n >= (EXEC_PLAN_BYTES - b->bytes) / sizeof step) break;
        if (b->n == b->cap) {
            size_t cap = b->cap ? b->cap * 2 : 128;
            if (cap > EXEC_PLAN_BYTES / sizeof step) cap = EXEC_PLAN_BYTES / sizeof step;
            NvnExecStep *p = realloc(b->steps, cap * sizeof *p);
            if (!p) break;
            b->steps = p, b->cap = cap;
        }
        b->steps[b->n++] = step;
        at = step.end;
    }
    if (b->failed || !b->n || b->steps[b->n - 1].end != bytes) b->n = b->bytes = 0, b->failed = 1;
}

static void front_build(exec_builder *b, const uint8_t *commands, size_t bytes) {
    if (!b->n) return;
    size_t primary_state_bytes = b->bytes;
    for (size_t primary = 0; primary < b->n;) {
        const NvnExecStep *first = &b->steps[primary];
        if (first->opcode != NVN_EXEC_STATE && !(first->flags & NVN_EXEC_STATE_INPUT)) {
            // Pure draw sequences already have structural runs. Summaries begin at state and consume complete
            // primary steps, reusing their already-folded deltas instead of decoding the source state again.
            primary++;
            continue;
        }
        size_t at = first->begin;
        size_t state_at = b->bytes;
        b->emits = 0, b->failed = 0;
        NvnExecFrontComposeSpan span = {0};
        // Appending the summary can reallocate b->state. The composer finishes its reads before invoking exec_emit;
        // every invocation receives the current pointer, with a bound that excludes previously appended summaries.
        if (nvn_exec_front_compose(commands, bytes, b->steps, b->n, b->state, primary_state_bytes,
                                    primary, exec_emit, b, &span)) {
            if (!b->failed && (b->emits != 1 || span.end <= at || span.end > bytes || (span.end & 7) ||
                span.end_step <= primary || span.end_step > b->n ||
                span.end != b->steps[span.end_step - 1].end ||
                span.draws < 2 || span.draws > 16 || span.ops < span.draws + 2 || span.ops > 1024 ||
                span.ops > (span.end - at) / sizeof(op_hdr)))
                rt_fatal("nvn: invalid front-pass summary interval");
            size_t used = b->n * sizeof(NvnExecStep) + b->bytes;
            if (b->failed || used > EXEC_PLAN_BYTES || b->nfront >= (EXEC_PLAN_BYTES - used) / sizeof(NvnFrontStep)) {
                b->bytes = state_at;
                b->failed = 1;
                break;
            }
            if (b->nfront == b->front_cap) {
                size_t cap = b->front_cap ? b->front_cap * 2 : 64;
                NvnFrontStep *p = realloc(b->front, cap * sizeof *p);
                if (!p) { b->bytes = state_at; b->failed = 1; break; }
                b->front = p, b->front_cap = cap;
            }
            b->front[b->nfront++] = (NvnFrontStep){at, span.end, span.ops, span.draws,
                                                  (uint32_t)state_at, (uint32_t)(b->bytes - state_at)};
            primary = span.end_step;
        } else {
            if (b->emits) rt_fatal("nvn: front composer emitted an unsupported interval");
            primary++;
        }
    }
}

static void copy_exec_check(const copy_prefix *pf) {
    if (!pf->nexec) {
        if (pf->state_bytes) rt_fatal("nvn: execution state has no index");
        return;
    }
    const uint8_t *commands = (const uint8_t *)(pf + 1);
    const NvnRecordSpan *spans = (const NvnRecordSpan *)(copy_offsets(pf) + pf->nuniform + pf->ndeferred +
                                                       pf->npinops + pf->ncapture);
    const NvnExecStep *steps = (const NvnExecStep *)(spans + pf->nspans);
    size_t at = 0, state_end = 0;
    for (size_t i = 0; i < pf->nexec; i++) {
        const NvnExecStep *s = &steps[i];
        if (s->begin != at || s->end <= at || s->end > pf->bytes || !s->ops)
            rt_fatal("nvn: execution plan coverage mismatch");
        if (s->flags != nvn_record_exec_flags(s->opcode) || s->reserved)
            rt_fatal("nvn: execution classification mismatch");
        if (s->opcode == NVN_EXEC_STATE) {
            if (s->run || s->state_offset != state_end || !s->state_bytes || (s->state_bytes & 7) ||
                s->state_bytes > pf->state_bytes - state_end)
                rt_fatal("nvn: execution state storage mismatch");
            state_end += s->state_bytes;
        } else if (s->ops != 1 || s->state_bytes || s->state_offset ||
                   ((const op_hdr *)(commands + at))->op != s->opcode)
            rt_fatal("nvn: execution operation mismatch");
        uint32_t ops = 0;
        while (at < s->end) {
            const op_hdr *h = (const op_hdr *)(commands + at);
            // Compiled state cannot cross a draw or any action that can observe intermediate state or guest bytes.
            if (s->opcode == NVN_EXEC_STATE && (nvn_record_op_class(h->op) != NVN_PLAN_RUN ||
                h->op == OP_DRAW || h->op == OP_DRAW_IDX || h->op == OP_DRAW_IDX_IND ||
                h->op == OP_REPORT_COUNTER || h->op == OP_RESET_COUNTER || h->op == OP_RENDER_ENABLE ||
                h->op == OP_RENDER_COND || h->op == OP_SET_TEX_POOL || h->op == OP_SET_SAMP_POOL))
                rt_fatal("nvn: compiled state crosses an ordered operation");
            at += h->size, ops++;
        }
        if (at != s->end || ops != s->ops) rt_fatal("nvn: execution operation count mismatch");
    }
    if (at != pf->bytes) rt_fatal("nvn: execution plan incomplete");
    const NvnFrontStep *front = (const NvnFrontStep *)(steps + pf->nexec);
    size_t front_end = 0;
    for (size_t i = 0; i < pf->nfront; i++) {
        const NvnFrontStep *s = &front[i];
        if (s->begin < front_end || s->end <= s->begin || s->end > pf->bytes ||
            s->state_offset != state_end || !s->state_bytes || (s->state_bytes & 7) ||
            s->state_bytes > pf->state_bytes - state_end)
            rt_fatal("nvn: front summary storage mismatch");
        uint32_t ops = 0, draws = 0;
        at = s->begin;
        while (at < s->end) {
            const op_hdr *h = (const op_hdr *)(commands + at);
            if (nvn_record_op_class(h->op) != NVN_PLAN_RUN || h->op == OP_REPORT_COUNTER ||
                h->op == OP_RESET_COUNTER || h->op == OP_RENDER_ENABLE || h->op == OP_RENDER_COND ||
                h->op == OP_SET_TEX_POOL || h->op == OP_SET_SAMP_POOL)
                rt_fatal("nvn: front summary crosses an ordered operation");
            draws += (nvn_record_exec_flags(h->op) & NVN_EXEC_DRAW) != 0;
            at += h->size, ops++;
        }
        if (at != s->end || ops != s->ops || draws != s->draws || draws < 2 || draws > 16)
            rt_fatal("nvn: front summary operation count mismatch");
        state_end += s->state_bytes, front_end = s->end;
    }
    if (state_end != pf->state_bytes) rt_fatal("nvn: execution state storage incomplete");
}

static int capture_compare(const void *aa, const void *bb) {
    const p_bind_buf *a = aa, *b = bb;
    if (a->addr != b->addr) return a->addr < b->addr ? -1 : 1;
    if (a->size != b->size) return a->size < b->size ? -1 : 1;
    if (a->stage != b->stage) return a->stage < b->stage ? -1 : 1;
    return a->index < b->index ? -1 : a->index > b->index;
}

static void copy_plan_check(const copy_prefix *pf) {
    copy_exec_check(pf);
    const uint8_t *cmd = (const uint8_t *)(pf + 1);
    const size_t *off = copy_offsets(pf), *capture = off + pf->nuniform + pf->ndeferred + pf->npinops;
    const NvnRecordSpan *spans = (const NvnRecordSpan *)(capture + pf->ncapture);
    size_t at = 0;
    for (size_t i = 0; i < pf->nspans; i++) {
        const NvnRecordSpan *s = &spans[i];
        if (s->begin != at || s->end <= at || s->end > pf->bytes || s->kind > NVN_PLAN_UPDATE)
            rt_fatal("nvn: replay plan coverage mismatch");
        uint32_t ops = 0, draws = 0;
        while (at < s->end) {
            const op_hdr *h = (const op_hdr *)(cmd + at);
            if (nvn_record_op_class(h->op) != s->kind) rt_fatal("nvn: replay plan operation mismatch");
            if (s->draw_stride && (h->size != s->draw_stride ||
                (h->op != OP_DRAW && h->op != OP_DRAW_IDX && h->op != OP_DRAW_IDX_IND)))
                rt_fatal("nvn: replay plan draw stride mismatch");
            ops++;
            draws += h->op == OP_DRAW || h->op == OP_DRAW_IDX || h->op == OP_DRAW_IDX_IND ||
                     h->op == OP_DRAW_MESH || h->op == OP_DRAW_MESH_IND;
            at += h->size;
        }
        if (at != s->end || ops != s->ops || draws != s->draws || (s->kind != NVN_PLAN_RUN && ops != 1))
            rt_fatal("nvn: replay plan count mismatch");
        if (s->draw_stride && (ops != draws || ops < 4)) rt_fatal("nvn: replay plan draw run mismatch");
    }
    if (at != pf->bytes) rt_fatal("nvn: replay plan missing command bytes");
    // The producer deduplicates with a hash table. The oracle instead sorts complete binding values, then compares
    // the unique reference set with the plan's set. Both read the finished copy, including a submission clone.
    if (!pf->nuniform) {
        if (pf->ncapture) rt_fatal("nvn: capture plan has unexpected bindings");
        return;
    }
    if (!pf->ncapture || pf->ncapture > pf->nuniform) rt_fatal("nvn: capture plan count mismatch");
    p_bind_buf *ref = malloc(pf->nuniform * sizeof *ref), *got = malloc(pf->ncapture * sizeof *got);
    if (!ref || !got) rt_fatal("nvn: capture plan check allocation");
    for (size_t i = 0; i < pf->nuniform; i++) ref[i] = *(const p_bind_buf *)(cmd + off[i] + sizeof(op_hdr));
    for (size_t i = 0; i < pf->ncapture; i++) {
        if (capture[i] > pf->bytes - sizeof(op_hdr) - sizeof(p_bind_buf) ||
            ((const op_hdr *)(cmd + capture[i]))->op != OP_BIND_UBO)
            rt_fatal("nvn: capture plan binding offset mismatch");
        got[i] = *(const p_bind_buf *)(cmd + capture[i] + sizeof(op_hdr));
    }
    qsort(ref, pf->nuniform, sizeof *ref, capture_compare);
    qsort(got, pf->ncapture, sizeof *got, capture_compare);
    size_t nref = 0;
    for (size_t i = 0; i < pf->nuniform; i++)
        if (!nref || capture_compare(&ref[nref - 1], &ref[i])) ref[nref++] = ref[i];
    if (nref != pf->ncapture) rt_fatal("nvn: capture plan unique count mismatch");
    for (size_t i = 0; i < nref; i++)
        if (capture_compare(&ref[i], &got[i])) rt_fatal("nvn: capture plan binding mismatch");
    free(got);
    free(ref);
}

// Check the finished copy, including its copied tail. This also checks clones independently of the builder's scratch
// storage. Only check mode walks all commands again; normal preparation consumes the compact indexes and plans.
static void copy_index_check(const copy_prefix *pf) {
    if (!pf->indexed || resource_prep_mode() != 2) return;
    const uint8_t *cmd = (const uint8_t *)(pf + 1);
    const size_t *u = copy_offsets(pf), *d = u + pf->nuniform, *p = d + pf->ndeferred;
    size_t nu = 0, nd = 0, np = 0;
    for (const op_hdr *h = (const op_hdr *)cmd; h->op != OP_END; h = nvn_op_next(h)) {
        size_t off = (const uint8_t *)h - cmd;
        if (h->op == OP_BIND_UBO && (nu >= pf->nuniform || u[nu++] != off))
            rt_fatal("nvn: uniform command index mismatch");
        if (op_deferred(h) && (nd >= pf->ndeferred || d[nd++] != off))
            rt_fatal("nvn: deferred command index mismatch");
        if (h->op < 64 && (PIN_OPS >> h->op & 1) && (np >= pf->npinops || p[np++] != off))
            rt_fatal("nvn: texture pin command index mismatch");
    }
    if (nu != pf->nuniform || nd != pf->ndeferred || np != pf->npinops)
        rt_fatal("nvn: command resource index count mismatch");
    copy_plan_check(pf);
}

#include "nvn_parent_plan.h"

// Validate newly recorded bytes and build their indexes. Child interiors have already passed this exact walk.
static inline uint64_t index_range(const uint8_t *p, size_t n, size_t at, size_t end, int indexed,
                                   offset_list *uniform, offset_list *deferred, offset_list *pinops,
                                   offset_list *capture, capture_set *captured, span_list *spans) {
    uint64_t pins_named = 0;
    while (at < end) {
        if (end - at < sizeof(op_hdr)) rt_fatal("nvn: recording has no complete end record");
        op_hdr h;
        memcpy(&h, p + at, sizeof h);
        if (h.size < 8 || (h.size & 7) || h.op >= OP_COUNT_ || h.op == OP_JUMP || h.op == OP_CALL || h.size > end - at ||
            (!at && h.op != OP_BEGIN))
            rt_fatal("nvn: corrupt recording at %p (op %u size %u)", (const void *)(p + at), h.op, h.size);
        const tex_t *pinned[9];
        int pinop = h.op < 64 && (PIN_OPS >> h.op & 1);
        int k = pinop ? op_pins(h.op, p + at + 8, pinned) : 0;
        for (int i = 0; i < k; i++) pin_ref(pinned[i], 1, 0);
        pins_named += (uint64_t)k;
        if (indexed) {
            if (h.op == OP_BIND_UBO) {
                if (h.size < sizeof(op_hdr) + sizeof(p_bind_buf)) rt_fatal("nvn: truncated uniform binding");
                offset_add(uniform, at);
                if (capture_add(captured, (const p_bind_buf *)(p + at + sizeof(op_hdr)))) offset_add(capture, at);
            }
            if (op_deferred((const op_hdr *)(p + at))) offset_add(deferred, at);
            if (pinop) offset_add(pinops, at);
            span_add(spans, &h, at);
        }
        at += h.size;
        if (h.op == OP_END) {
            if (at != n) rt_fatal("nvn: recording at %p ends at %zu of %zu bytes", (const void *)p, at, n);
            return pins_named;
        }
    }
    if (end == n) rt_fatal("nvn: recording has no complete end record");
    return pins_named;
}

// The recording and its resource metadata share one allocation. The validation walk already required at EndRecording
// produces offsets into the copied bytes; submit, snapshot gathering and pin retirement need not scan unrelated ops.
// Scratch lists retain their capacity in a bounded builder pool. Command bytes are still copied only once.
static uint8_t *flatten_planned(const uint8_t *p, size_t n, parent_plan *parent) {
    int indexed = resource_prep_mode() != 0;
    PlanScratch *scratch = indexed ? plan_scratch_get() : NULL;
    offset_list uniform = scratch ? scratch->uniform : (offset_list){0};
    offset_list deferred = scratch ? scratch->deferred : (offset_list){0};
    offset_list pinops = scratch ? scratch->pinops : (offset_list){0};
    offset_list capture = scratch ? scratch->capture : (offset_list){0};
    span_list spans = scratch ? scratch->spans : (span_list){0};
    capture_set captured = scratch ? scratch->captured : (capture_set){0};
    exec_builder executable = scratch ? scratch->exec : (exec_builder){0};
    uniform.n = deferred.n = pinops.n = capture.n = spans.n = 0;
    if (indexed) capture_reset(&captured);
    uint64_t pins_named = 0;
    int selected = indexed && parent;
    size_t cursor = 0;
    if (selected) {
        for (size_t i = 0; i < parent->eligible; i++) {
            const plan_piece *part = &parent->interior[i];
            pins_named += index_range(p, n, cursor, part->first + part->delta, indexed, &uniform, &deferred,
                                      &pinops, &capture, &captured, &spans);
            pins_named += import_resources(part, p, &uniform, &deferred, &pinops, &capture, &captured, &spans);
            cursor = part->end + part->delta;
        }
    }
    pins_named += index_range(p, n, cursor, n, indexed, &uniform, &deferred, &pinops, &capture, &captured, &spans);
    int imported = selected && parent_plan_build(parent, p, n);
    // The rare invalid relocation retries only the executable metadata; pin references were already acquired.
build_metadata:
    if (indexed && !imported) exec_build(&executable, p, n), front_build(&executable, p, n);
    size_t ne = imported ? parent->n : executable.n;
    size_t nf = imported ? parent->nf : executable.nfront;
    size_t ns = imported ? parent->bytes : executable.bytes;
    size_t ni = uniform.n + deferred.n + pinops.n + capture.n;
    if (n > SIZE_MAX - sizeof(copy_prefix)) rt_fatal("nvn: command copy too large");
    if (ni > (SIZE_MAX - sizeof(copy_prefix) - n) / sizeof(size_t)) rt_fatal("nvn: command resource index too large");
    size_t bytes = sizeof(copy_prefix) + n + ni * sizeof(size_t);
    if (spans.n > (SIZE_MAX - bytes) / sizeof(NvnRecordSpan)) rt_fatal("nvn: command replay plan too large");
    bytes += spans.n * sizeof(NvnRecordSpan);
    if (ne > (SIZE_MAX - bytes) / sizeof(NvnExecStep)) rt_fatal("nvn: execution plan too large");
    bytes += ne * sizeof(NvnExecStep);
    if (nf > (SIZE_MAX - bytes) / sizeof(NvnFrontStep)) rt_fatal("nvn: front summary too large");
    bytes += nf * sizeof(NvnFrontStep);
    if (ns > SIZE_MAX - bytes) rt_fatal("nvn: execution state too large");
    bytes += ns;
    copy_prefix *pf = realloc_wait(NULL, bytes, "command copy");
    if (!pf) rt_fatal("nvn: command copy allocation");
    *pf = (copy_prefix){.magic = COPY_MAGIC, .bytes = n, .pins = pins_named, .indexed = (uint32_t)indexed,
                       .plan_complete = imported || (indexed && !executable.failed && ne),
                       .deferred = indexed ? (deferred.n ? 2 : 1) : 0,
                       .nuniform = uniform.n, .ndeferred = deferred.n, .npinops = pinops.n,
                       .ncapture = capture.n, .nspans = spans.n, .nexec = ne,
                       .nfront = nf, .state_bytes = ns};
    memcpy(pf + 1, p, n);
    size_t *off = (size_t *)((uint8_t *)(pf + 1) + n);
    if (uniform.n) memcpy(off, uniform.v, uniform.n * sizeof *off);
    if (deferred.n) memcpy(off + uniform.n, deferred.v, deferred.n * sizeof *off);
    if (pinops.n) memcpy(off + uniform.n + deferred.n, pinops.v, pinops.n * sizeof *off);
    if (capture.n) memcpy(off + uniform.n + deferred.n + pinops.n, capture.v, capture.n * sizeof *off);
    if (spans.n) memcpy(off + ni, spans.v, spans.n * sizeof *spans.v);
    NvnExecStep *steps = (NvnExecStep *)((NvnRecordSpan *)(off + ni) + spans.n);
    if (imported) {
        if (!parent_plan_write(parent, steps)) {
            free(pf);
            imported = 0;
            goto build_metadata;
        }
    } else {
        if (ne) memcpy(steps, executable.steps, ne * sizeof *steps);
        NvnFrontStep *front = (NvnFrontStep *)(steps + ne);
        if (nf) memcpy(front, executable.front, nf * sizeof *front);
        if (ns) memcpy(front + nf, executable.state, ns);
    }
    copy_index_check(pf);
    if (scratch) {
        scratch->uniform = uniform, scratch->deferred = deferred, scratch->pinops = pinops, scratch->capture = capture;
        scratch->spans = spans, scratch->captured = captured;
        scratch->exec = executable;
        plan_scratch_put(scratch);
    }
    return (uint8_t *)(pf + 1);
}

static uint8_t *flatten_recording(const uint8_t *p, size_t n) { return flatten_planned(p, n, NULL); }

static __attribute__((noinline)) uint8_t *flatten_parent(cmd_t *o, size_t n) {
    parent_plan plan = {0};
    int eligible = parent_plan_select(&plan, n);
    uint8_t *copy = eligible ? flatten_planned(o->seg, n, &plan) : flatten_recording(o->seg, n);
    parent_plan_release(&plan);
    return copy;
}

static uint8_t *finish_copy(cmd_t *o) {
    size_t n = (size_t)(o->cur - o->seg);
    if (n >= PLAN_PARENT_MIN && o->plan_cookie && o->plan_cookie == recording_hints.cookie &&
        recording_hints.owner == o) return flatten_parent(o, n);
    return flatten_recording(o->seg, n);
}

static void copy_add(uintptr_t lo, uintptr_t hi, uint8_t *copy) {
    copy_prefix *pf = (copy_prefix *)copy - 1;
    uint64_t handle = (uintptr_t)copy;
    uintptr_t page = lo >> COPY_PAGE_SHIFT;
    plat_mutex_lock(&copy_lock);
    pf->made_at = copies_made + 1;  // distinguish recordings even when the allocator recycles a handle
    copy_bucket *b = nvn_hm_get(&copy_pages, &page);
    if (!b) {
        b = calloc(1, sizeof *b);
        if (!b) rt_fatal("nvn: command copy index");
        nvn_hm_put(&copy_pages, &page, b);
    }
    if (b->n == b->cap) {
        b->cap = b->cap ? b->cap * 2 : 8;
        b->v = realloc(b->v, b->cap * sizeof *b->v);
        if (!b->v) rt_fatal("nvn: command copy index");
    }
    b->v[b->n++] = (cmd_copy){lo, hi, copy};
    nvn_hm_put(&copy_live, &handle, pf);
    if (pf->indexed) nvn_hm_put(&copy_indexed, &handle, pf);
    ncopies++;
    copy_bytes += copy_size(pf);
    if (++copies_made % 20000 == 0)
        NVN_LOG("recording copies: %llu made, %llu live, %.1f MiB, submit %llu", (unsigned long long)copies_made,
                (unsigned long long)ncopies, copy_bytes / 1048576.0, (unsigned long long)nsubmits);
    plat_mutex_unlock(&copy_lock);
}

// Adding control memory again retires every recording copy made in it (the driver's handle lived there).
static void release_copies(uintptr_t lo, uintptr_t hi) {
    if (hi <= lo) return;
    uint64_t replayed = gfx_replayed_submits();
    copy_prefix *doomed = NULL;
    plat_mutex_lock(&copy_lock);
    if (nzombies) doomed = zombie_sweep(replayed, doomed);
    for (uintptr_t pg = lo >> COPY_PAGE_SHIFT; pg <= (hi - 1) >> COPY_PAGE_SHIFT; pg++) {
        copy_bucket *b = nvn_hm_get(&copy_pages, &pg);
        if (!b) continue;
        for (uint32_t i = 0; i < b->n;) {
            cmd_copy *k = &b->v[i];
            if (!(k->lo < hi && lo < k->hi)) { i++; continue; }
            uint64_t handle = (uintptr_t)k->copy;
            copy_prefix *pf = (copy_prefix *)k->copy - 1;
            nvn_hm_del(&copy_live, &handle);
            ncopies--;
            copy_bytes -= copy_size(pf);
            pf->magic = 0;
            if (pf->plan_readers || pf->used_at > replayed) zombie_add(pf);
            else doomed = doom(doomed, pf);
            b->v[i] = b->v[--b->n];
        }
        if (!b->n) {
            nvn_hm_del(&copy_pages, &pg);
            free(b->v);
            free(b);
        }
    }
    plat_mutex_unlock(&copy_lock);
    doom_free(doomed);
}

// ---------- CommandBuffer: object and memory
// NVNboolean nvnCommandBufferInitialize(NVNcommandBuffer *cmd, NVNdevice *device): the inline fast-path fields are kept
// at ~0 / 0 so NVN 55 inline code never finds room and calls the API (nvn_obj.h cmd_t)
NVN_API(nvnCommandBufferInitialize) {
    cmd_t *o = cmd_of(c);
    // Initialize over an object that was never finalized drops its recording buffer (seg lives in host memory). Logged
    // only: the old bytes may be a copy whose buffer another object owns, so freeing it here is not safe.
    static uint32_t reinits;
    if (o->magic == CMD_MAGIC && o->seg && __atomic_fetch_add(&reinits, 1, __ATOMIC_RELAXED) < 8)
        rt_log("nvn: Initialize over the command buffer %p, which was not finalized; its %zu-byte recording buffer %p "
               "is dropped", (void *)o, (size_t)(o->end - o->seg), (void *)o->seg);
    memset(o, 0, sizeof *o);
    o->magic = CMD_MAGIC;
    o->dev = arg_p(c, 1);
    o->inline_cur = ~0ull;
    o->inline_end = 0;
    cb_set_disabled(o, CB_NO_COMMAND | CB_NO_CONTROL, 0);
    ret_x(c, 1);
}
// void nvnCommandBufferFinalize(NVNcommandBuffer *cmd): handles outlive their recorder
NVN_API(nvnCommandBufferFinalize) {
    cmd_t *o = cmd_of(c);
    pool_put(o->seg, (size_t)(o->end - o->seg));
    o->seg = o->cur = o->end = NULL;
    o->magic = 0;
    cb_set_disabled(o, CB_NO_COMMAND | CB_NO_CONTROL, 0);
}
// void nvnCommandBufferSetMemoryCallback(NVNcommandBuffer *cmd, PFNNVNCOMMANDBUFFERMEMORYCALLBACKPROC callback)
NVN_API(nvnCommandBufferSetMemoryCallback) { cmd_of(c)->cb = arg_p(c, 1); }
// void nvnCommandBufferSetMemoryCallbackData(NVNcommandBuffer *cmd, void *data)
NVN_API(nvnCommandBufferSetMemoryCallbackData) { cmd_of(c)->cb_data = arg_p(c, 1); }
// PFNNVNCOMMANDBUFFERMEMORYCALLBACKPROC nvnCommandBufferGetMemoryCallback(const NVNcommandBuffer *cmd)
NVN_API(nvnCommandBufferGetMemoryCallback) { ret_x(c, (uintptr_t)cmd_of(c)->cb); }
// void *nvnCommandBufferGetMemoryCallbackData(const NVNcommandBuffer *cmd) (nvn.h declares it void)
NVN_API(nvnCommandBufferGetMemoryCallbackData) { ret_x(c, (uintptr_t)cmd_of(c)->cb_data); }
// void nvnCommandBufferSetCommandMemoryCallbackEnabled / SetControlMemoryCallbackEnabled(NVNcommandBuffer *cmd,
// NVNboolean enabled) (NVN 55, no prototype in nvn.h)
NVN_API(nvnCommandBufferSetCommandMemoryCallbackEnabled) { cb_set_disabled(cmd_of(c), CB_NO_COMMAND, !nvn_arg_bool(c, 1)); }
NVN_API(nvnCommandBufferSetControlMemoryCallbackEnabled) { cb_set_disabled(cmd_of(c), CB_NO_CONTROL, !nvn_arg_bool(c, 1)); }
// void nvnCommandBufferAddCommandMemory(NVNcommandBuffer *cmd, const NVNmemoryPool *pool, ptrdiff_t offset, size_t size):
// accounting only; command memory can be shared by independent recordings with distinct control memory
NVN_API(nvnCommandBufferAddCommandMemory) {
    cmd_t *o = cmd_of(c);
    const pool_t *p = arg_p(c, 1);
    o->cmd_base = (uintptr_t)(p ? p->mem : NULL) + (uintptr_t)arg_x(c, 2);
    o->game_size = arg_x(c, 3);
    o->cmd_used = 0;
    o->hw_used = 0;
}
// void nvnCommandBufferAddControlMemory(NVNcommandBuffer *cmd, void *memory, size_t size): the SDK encodes handles in
// control memory (sdk+0x444530), so reusing that storage retires the copies recorded there
NVN_API(nvnCommandBufferAddControlMemory) {
    cmd_t *o = cmd_of(c);
    void *m = arg_p(c, 1);
    size_t n = arg_x(c, 2);
    release_copies((uintptr_t)m, (uintptr_t)m + n);
    o->ctl = m;
    o->ctl_size = n;
    o->ctl_used = 0;
}
// size_t nvnCommandBufferGet{Control,Command}Memory{Size,Used,Free}(const NVNcommandBuffer *cmd): usage is bounded by
// the memory the game gave, so the game never sees more used than it provided
static size_t min_size(size_t a, size_t b) { return a < b ? a : b; }
NVN_API(nvnCommandBufferGetControlMemorySize) { ret_x(c, cmd_of(c)->ctl_size); }
NVN_API(nvnCommandBufferGetControlMemoryUsed) { const cmd_t *o = cmd_of(c); ret_x(c, min_size(o->ctl_used, o->ctl_size)); }
NVN_API(nvnCommandBufferGetControlMemoryFree) { const cmd_t *o = cmd_of(c); ret_x(c, o->ctl_size - min_size(o->ctl_used, o->ctl_size)); }
NVN_API(nvnCommandBufferGetCommandMemorySize) { ret_x(c, cmd_of(c)->game_size); }
NVN_API(nvnCommandBufferGetCommandMemoryUsed) { const cmd_t *o = cmd_of(c); ret_x(c, min_size(o->hw_used, o->game_size)); }
NVN_API(nvnCommandBufferGetCommandMemoryFree) { const cmd_t *o = cmd_of(c); ret_x(c, o->game_size - min_size(o->hw_used, o->game_size)); }

// ---------- CommandBuffer: recording
// void nvnCommandBufferBeginRecording(NVNcommandBuffer *cmd)
NVN_API(nvnCommandBufferBeginRecording) {
    cmd_t *o = cmd_of(c);
    if (o->recording) rt_log("nvn: BeginRecording on command buffer %p while it records", (void *)o);
    o->recording = 1;
    o->cur = o->seg;   // the previous recording was copied at EndRecording
    o->plan_cookie = 0;
    put_hdr(cmd_alloc(o, 8), OP_BEGIN, 8);
    o->hw_used -= 8;   // BEGIN and END have no hardware equivalent
    o->hw_start = o->hw_used;
    if (!o->game_size) memory_event(c, o, NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY, MIN_COMMAND_SIZE);
}
// NVNcommandHandle nvnCommandBufferEndRecording(NVNcommandBuffer *cmd)
NVN_API(nvnCommandBufferEndRecording) {
    cmd_t *o = cmd_of(c);
    if (!o->recording || !o->seg) rt_fatal("nvn: EndRecording on command buffer %p, which does not record", (void *)o);
    if (!o->ctl_size) memory_event(c, o, NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_CONTROL_MEMORY, MIN_CONTROL_SIZE);
    put_hdr(cmd_alloc(o, 8), OP_END, 8);
    o->hw_used -= 8;
    o->recording = 0;
    uint8_t *copy = finish_copy(o);
    // Finalizing the recorder does not retire its handle; only reusing the control memory it was recorded in does.
    uintptr_t lo = (uintptr_t)o->ctl + (o->ctl_used < o->ctl_size ? o->ctl_used : 0);
    uintptr_t hi = o->ctl ? lo + 1 : 0;
    o->ctl_used += CTL_PER_RECORDING;
    copy_add(lo, hi, copy);
    ret_x(c, (uintptr_t)copy);
}
// NVNboolean nvnCommandBufferIsRecording(const NVNcommandBuffer *cmd)
NVN_API(nvnCommandBufferIsRecording) { ret_x(c, cmd_of(c)->recording != 0); }

// The called recording is copied into the caller at the call position: the game recycles command memory in small steps,
// so a handle may be gone by the time the caller is submitted. Deferred binds stay deferred. A copy is OP_BEGIN, its
// records and OP_END (flatten_recording checked them), so the records go over as one block; usage grows by the same
// bytes as record by record.
static void inline_recording(cmd_t *o, uint64_t handle, const char *what) {
    if (!handle) rt_fatal("nvn: %s with a null handle", what);
    copy_prefix *reader;
    uint64_t bytes = copy_touch_id(handle, __atomic_load_n(&nsubmits, __ATOMIC_RELAXED), what, &reader);
    if (bytes <= 16) return;
    size_t body = (size_t)bytes - 16;
    cmd_reserve(o, body);
    if (reader) plan_note_child(o, handle, reader->made_at, body);
    memcpy(o->cur, (const uint8_t *)(uintptr_t)handle + 8, body);
    if (reader) plan_drop_readers(&reader, 1);
    o->cur += body;
    o->cmd_used += body;
    o->hw_used += body;
    o->last_rec = 0;
}
// void nvnCommandBuffer{Call,Copy}Commands(NVNcommandBuffer *cmd, int n, const NVNcommandHandle *handles)
NVN_API(nvnCommandBufferCallCommands) {
    cmd_t *o = cmd_of(c);
    int n = (int)arg_x(c, 1);
    const uint64_t *h = arg_p(c, 2);
    for (int i = 0; i < n; i++) inline_recording(o, h[i], "CallCommands");
}
NVN_API(nvnCommandBufferCopyCommands) {
    cmd_t *o = cmd_of(c);
    int n = (int)arg_x(c, 1);
    const uint64_t *h = arg_p(c, 2);
    for (int i = 0; i < n; i++) inline_recording(o, h[i], "CopyCommands");
}

// ---------- CommandBuffer: commands with CPU-visible results (performed by the back end in GPU order)
// void nvnCommandBufferSignalEvent(NVNcommandBuffer *cmd, const NVNevent *event, NVNeventSignalMode mode,
// NVNeventSignalLocation location, int flags, uint32_t value)
NVN_API(nvnCommandBufferSignalEvent) {
    const ev_t *e = arg_p(c, 1);
    *REC(cmd_of(c), OP_SIGNAL_EVENT, p_signal_event) = (p_signal_event){e->val, (uint32_t)arg_x(c, 2), (uint32_t)arg_x(c, 5)};
}
// void nvnCommandBufferResetCounter(NVNcommandBuffer *cmd, NVNcounterType type)
NVN_API(nvnCommandBufferResetCounter) { REC(cmd_of(c), OP_RESET_COUNTER, p_reset_counter)->type = (uint32_t)arg_x(c, 1); }
// void nvnCommandBufferReportCounter(NVNcommandBuffer *cmd, NVNcounterType type, NVNbufferAddress address)
NVN_API(nvnCommandBufferReportCounter) {
    *REC(cmd_of(c), OP_REPORT_COUNTER, p_report_counter) = (p_report_counter){(uint32_t)arg_x(c, 1), arg_x(c, 2)};
}
// void nvnCommandBufferCopyBufferToBuffer(NVNcommandBuffer *cmd, NVNbufferAddress src, NVNbufferAddress dst,
// size_t size, int flags)
NVN_API(nvnCommandBufferCopyBufferToBuffer) {
    *REC(cmd_of(c), OP_COPY_B2B, p_copy_b2b) = (p_copy_b2b){arg_x(c, 1), arg_x(c, 2), arg_x(c, 3)};
}
// NVN2: void nvnCommandBufferPushComputeConstants(NVNcommandBuffer *cmd, offset, ?, size_t size, const void *data):
// constants the compute stage reads from its driver bank (c0, from byte 0x10), copied at record time. Arguments as the
// game calls it (Ocarina main+0x20f811c: x1 0, x2 0, x3 0x40, x4 the data); x1 is taken as the byte offset.
#define PUSH_CC_MAX 1024
NVN_API(nvnCommandBufferPushComputeConstants) {
    uint64_t off = arg_x(c, 1), n = arg_x(c, 3);
    const void *data = arg_p(c, 4);
    static int logged;
    if (arg_x(c, 2) && !logged++) NVN_LOG("PushComputeConstants: second argument %#llx not understood", (unsigned long long)arg_x(c, 2));
    static struct { uint64_t a1, a2, a3; } seen[24];   // diagnosis: the argument shapes the game uses
    static int nseen;
    int known = 0;
    for (int i = 0; i < nseen; i++) known |= seen[i].a1 == off && seen[i].a2 == arg_x(c, 2) && seen[i].a3 == n;
    if (!known && nseen < 24) {
        seen[nseen++] = (typeof(seen[0])){off, arg_x(c, 2), n};
        rt_log("nvn: PushComputeConstants: x1 %#llx x2 %#llx x3 %#llx x4 %p x5 %#llx x6 %#llx", (unsigned long long)off,
                (unsigned long long)arg_x(c, 2), (unsigned long long)n, data, (unsigned long long)arg_x(c, 5),
                (unsigned long long)arg_x(c, 6));
    }
    if (!data || off + n > PUSH_CC_MAX) { NVN_LOG("PushComputeConstants: %#llx bytes at %#llx dropped", (unsigned long long)n, (unsigned long long)off); return; }
    p_push_cc *r = rec(cmd_of(c), OP_PUSH_CC, offsetof(p_push_cc, data) + n);
    r->offset = (uint32_t)off;
    r->size = (uint32_t)n;
    memcpy(r->data, data, n);
}
// void nvnCommandBufferUpdateUniformBuffer(NVNcommandBuffer *cmd, NVNbufferAddress buffer, size_t buffer_size,
// ptrdiff_t offset, size_t size, const void *data): the data is copied at record time (the caller may change it
// afterwards) and written into the buffer at replay
NVN_API(nvnCommandBufferUpdateUniformBuffer) {
    uint64_t dst = arg_x(c, 1) + arg_x(c, 3), n = arg_x(c, 4);
    const void *data = arg_p(c, 5);
    if (n > (16u << 20)) rt_fatal("nvn: UpdateUniformBuffer of %#llx bytes", (unsigned long long)n);
    p_update_ub *r = rec(cmd_of(c), OP_UPDATE_UB, offsetof(p_update_ub, data) + n);
    r->dst = dst;
    r->size = n;
    memcpy(r->data, data, n);
}

// ---------- CommandBuffer: render state, payloads of nvn_obj.h
// void nvnCommandBufferSetRenderTargets(NVNcommandBuffer *cmd, int n, const NVNtexture *const *colors,
// const NVNtextureView *const *color_views, const NVNtexture *depth, const NVNtextureView *depth_view)
NVN_API(nvnCommandBufferSetRenderTargets) {
    int n = (int)arg_x(c, 1);
    const NVNtexture *const *ct = arg_p(c, 2);
    const NVNtextureView *const *cv = arg_p(c, 3);
    const NVNtexture *dt = arg_p(c, 4);
    const NVNtextureView *dv = arg_p(c, 5);
    if (n < 0 || n > 8) {
        rt_log("nvn: SetRenderTargets with %d colour targets", n);
        n = n < 0 ? 0 : 8;
    }
    p_set_rt *r = REC(cmd_of(c), OP_SET_RT, p_set_rt);
    r->n = n;
    for (int i = 0; i < n; i++) {
        r->color[i] = ct ? tex_pin(ct[i]) : NULL;
        copy_view(&r->cview[i], &r->has_cview[i], cv ? cv[i] : NULL);
    }
    r->depth = tex_pin(dt);
    copy_view(&r->dview, &r->has_dview, dv);
}
// void nvnCommandBufferClearColor(NVNcommandBuffer *cmd, int index, const float *rgba, int mask)
NVN_API(nvnCommandBufferClearColor) {
    p_clear_color *r = REC(cmd_of(c), OP_CLEAR_COLOR, p_clear_color);
    r->index = (int)arg_x(c, 1);
    r->mask = (int)arg_x(c, 3);
    memcpy(r->rgba, arg_p(c, 2), 16);
}
// void nvnCommandBufferClearDepthStencil(NVNcommandBuffer *cmd, float depth, NVNboolean depth_mask, int stencil,
// int stencil_mask)
NVN_API(nvnCommandBufferClearDepthStencil) {
    *REC(cmd_of(c), OP_CLEAR_DS, p_clear_ds) = (p_clear_ds){arg_s(c, 0), nvn_arg_bool(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3)};
}
static void set_viewports(cmd_t *o, int first, int n, const float *v) {
    if (first < 0 || n < 0 || first + n > 16) {
        rt_log("nvn: SetViewports first %d n %d", first, n);
        return;
    }
    p_viewports *r = REC(o, OP_VIEWPORTS, p_viewports);
    r->first = first;
    r->n = n;
    for (int i = 0; i < n; i++) memcpy(r->v[i], v + 4 * i, 16);
}
// void nvnCommandBufferSetViewports(NVNcommandBuffer *cmd, int first, int n, const float *xywh)
NVN_API(nvnCommandBufferSetViewports) { set_viewports(cmd_of(c), (int)arg_x(c, 1), (int)arg_x(c, 2), arg_p(c, 3)); }
// void nvnCommandBufferSetViewport(NVNcommandBuffer *cmd, int x, int y, int w, int h)
NVN_API(nvnCommandBufferSetViewport) {
    float v[4] = {(float)(int)arg_x(c, 1), (float)(int)arg_x(c, 2), (float)(int)arg_x(c, 3), (float)(int)arg_x(c, 4)};
    set_viewports(cmd_of(c), 0, 1, v);
}
static void set_scissor(cmd_t *o, int x, int y, int w, int h) { *REC(o, OP_SCISSOR, p_scissor) = (p_scissor){x, y, w, h}; }
// void nvnCommandBufferSetScissor(NVNcommandBuffer *cmd, int x, int y, int w, int h)
NVN_API(nvnCommandBufferSetScissor) {
    set_scissor(cmd_of(c), (int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), (int)arg_x(c, 4));
}
// void nvnCommandBufferSetScissors(NVNcommandBuffer *cmd, int first, int n, const int *xywh): the replay keeps one
// scissor (index 0); other indices matter only to shaders that select a viewport index, so differing ones are reported
NVN_API(nvnCommandBufferSetScissors) {
    int first = (int)arg_x(c, 1), n = (int)arg_x(c, 2);
    const int *r = arg_p(c, 3);
    if (first < 0 || n < 0 || first + n > 16) rt_fatal("nvn: SetScissors first %d n %d", first, n);
    for (int i = 1; i < n; i++)
        if (memcmp(r + 4 * i, r, 16)) {
            static int once;
            if (!once++) rt_log("nvn: SetScissors: differing scissor %d ignored (single-scissor replay)", first + i);
        }
    if (first == 0 && n > 0) set_scissor(cmd_of(c), r[0], r[1], r[2], r[3]);
}
static void set_depth_range(cmd_t *o, float n, float f) { *REC(o, OP_DEPTH_RANGE, p_depth_range) = (p_depth_range){n, f}; }
// void nvnCommandBufferSetDepthRange(NVNcommandBuffer *cmd, float near, float far)
NVN_API(nvnCommandBufferSetDepthRange) { set_depth_range(cmd_of(c), arg_s(c, 0), arg_s(c, 1)); }
// void nvnCommandBufferSetDepthRanges(NVNcommandBuffer *cmd, int first, int n, const float *ranges)
NVN_API(nvnCommandBufferSetDepthRanges) {
    int first = (int)arg_x(c, 1), n = (int)arg_x(c, 2);
    const float *r = arg_p(c, 3);
    if (first < 0 || n < 0 || first + n > 16) rt_fatal("nvn: SetDepthRanges first %d n %d", first, n);
    for (int i = 1; i < n; i++)
        if (memcmp(r + 2 * i, r, 8)) {
            static int once;
            if (!once++) rt_log("nvn: SetDepthRanges: differing range %d ignored (single-range replay)", first + i);
        }
    if (first == 0 && n > 0) set_depth_range(cmd_of(c), r[0], r[1]);
}
// void nvnCommandBufferSetDepthClamp / SetRenderEnable / SetRasterizerDiscard(NVNcommandBuffer *cmd, NVNboolean enable)
NVN_API(nvnCommandBufferSetDepthClamp) { REC(cmd_of(c), OP_DEPTH_CLAMP, p_flag)->value = nvn_arg_bool(c, 1) != 0; }
NVN_API(nvnCommandBufferSetRenderEnable) { REC(cmd_of(c), OP_RENDER_ENABLE, p_flag)->value = nvn_arg_bool(c, 1) != 0; }
// Rasterizer discard runs vertex work without rasterising; with no transform feedback that equals not drawing.
NVN_API(nvnCommandBufferSetRasterizerDiscard) { REC(cmd_of(c), OP_RASTER_DISCARD, p_flag)->value = nvn_arg_bool(c, 1) != 0; }
// void nvnCommandBufferSetPrimitiveRestart(NVNcommandBuffer *cmd, NVNboolean enable, int index): NX2NAVI draws its
// terrain and cliff strips with 16-bit indices and restart at 0xffff; the replay applies it per indexed draw (op_draw_idx)
NVN_API(nvnCommandBufferSetPrimitiveRestart) {
    *REC(cmd_of(c), OP_PRIM_RESTART, p_prim_restart) = (p_prim_restart){nvn_arg_bool(c, 1) != 0, (uint32_t)arg_x(c, 2)};
}
// void nvnCommandBufferSetRenderEnableConditional(NVNcommandBuffer *cmd, NVNconditionalRenderMode mode,
// NVNbufferAddress counters): rendering is enabled when the two NVNcounterData.counter values at addr / addr+16
// compare equal (mode 0) or not equal (mode 1). NX2NAVI gates most of its ClearColor commands with it; the replay
// evaluates it in GPU order.
NVN_API(nvnCommandBufferSetRenderEnableConditional) {
    int m = (int)arg_x(c, 1);
    if (m != 0 && m != 1) rt_fatal("nvn: SetRenderEnableConditional: unknown mode %d", m);
    p_render_cond *r = REC(cmd_of(c), OP_RENDER_COND, p_render_cond);
    r->mode = m;
    r->addr = arg_x(c, 2);
}
// void nvnCommandBufferBindProgram(NVNcommandBuffer *cmd, const NVNprogram *program, int stages)
NVN_API(nvnCommandBufferBindProgram) {
    *REC(cmd_of(c), OP_BIND_PROGRAM, p_bind_program) = (p_bind_program){arg_p(c, 1), (int)arg_x(c, 2)};
}
// void nvnCommandBufferBind{Blend,Color,ChannelMask,Multisample,Polygon,DepthStencil}State(NVNcommandBuffer *cmd,
// const NVN*State *state): the state object's bytes are copied (p_bind_state.kind order)
static void bind_state(const Cpu *c, int kind, size_t n) {
    p_bind_state *r = REC(cmd_of(c), OP_BIND_STATE, p_bind_state);
    r->kind = kind;
    memcpy(r->bytes, arg_p(c, 1), n);
}
NVN_API(nvnCommandBufferBindBlendState) { bind_state(c, 0, sizeof(blend_st)); }
NVN_API(nvnCommandBufferBindColorState) { bind_state(c, 1, sizeof(color_st)); }
NVN_API(nvnCommandBufferBindChannelMaskState) { bind_state(c, 2, sizeof(chmask_st)); }
NVN_API(nvnCommandBufferBindMultisampleState) { bind_state(c, 3, sizeof(ms_st)); }
NVN_API(nvnCommandBufferBindPolygonState) { bind_state(c, 4, sizeof(poly_st)); }
NVN_API(nvnCommandBufferBindDepthStencilState) { bind_state(c, 5, sizeof(ds_st)); }
// void nvnCommandBufferSetPointSize(NVNcommandBuffer *cmd, float size): recorded as bind-state kind 6 with the float as payload
NVN_API(nvnCommandBufferSetPointSize) {
    p_bind_state *r = REC(cmd_of(c), OP_BIND_STATE, p_bind_state);
    float s = arg_s(c, 0);
    r->kind = 6;
    memcpy(r->bytes, &s, sizeof s);
}
// void nvnCommandBufferBindVertexAttribState(NVNcommandBuffer *cmd, int n, const NVNvertexAttribState *states)
NVN_API(nvnCommandBufferBindVertexAttribState) {
    cmd_t *o = cmd_of(c);
    int n = (int)arg_x(c, 1);
    if (n < 0 || n > 16) { rt_log("nvn: BindVertexAttribState n %d", n); return; }
    p_bind_vattr *r = RECN(o, OP_BIND_VATTR, p_bind_vattr, st, n);
    r->first = 0;
    r->n = n;
    memcpy(r->st, arg_p(c, 2), 4 * (size_t)n);
    hw_fix(o, 68);   // the driver always writes all 16 attribute formats: 17 words
}
// void nvnCommandBufferBindVertexStreamState(NVNcommandBuffer *cmd, int n, const NVNvertexStreamState *states)
NVN_API(nvnCommandBufferBindVertexStreamState) {
    cmd_t *o = cmd_of(c);
    int n = (int)arg_x(c, 1);
    if (n < 0 || n > 16) { rt_log("nvn: BindVertexStreamState n %d", n); return; }
    p_bind_vstream *r = RECN(o, OP_BIND_VSTREAM, p_bind_vstream, st, n);
    r->first = 0;
    r->n = n;
    memcpy(r->st, arg_p(c, 2), 8 * (size_t)n);
    hw_fix(o, 4 * (16 + (size_t)n));
}
static void bind_vertex_buffers(cmd_t *o, int first, int n, const NVNbufferRange *rg) {
    if (first < 0 || n < 0 || first + n > 16) { rt_log("nvn: BindVertexBuffers first %d n %d", first, n); return; }
    p_bind_vbuf *r = RECN(o, OP_BIND_VBUF, p_bind_vbuf, r, n);
    r->first = first;
    r->n = n;
    memcpy(r->r, rg, sizeof *rg * (size_t)n);
    hw_fix(o, 24 * (size_t)n);
}
// void nvnCommandBufferBindVertexBuffers(NVNcommandBuffer *cmd, int first, int n, const NVNbufferRange *ranges)
NVN_API(nvnCommandBufferBindVertexBuffers) { bind_vertex_buffers(cmd_of(c), (int)arg_x(c, 1), (int)arg_x(c, 2), arg_p(c, 3)); }
// void nvnCommandBufferBindVertexBuffer(NVNcommandBuffer *cmd, int index, NVNbufferAddress address, size_t size)
NVN_API(nvnCommandBufferBindVertexBuffer) {
    NVNbufferRange r = {arg_x(c, 2), arg_x(c, 3)};
    bind_vertex_buffers(cmd_of(c), (int)arg_x(c, 1), 1, &r);
}
// Uniform and storage buffers: one record per binding (CB_SIZE/ADDRESS 4 words + CB_BIND 1 on hardware).
static void bind_buffers(const Cpu *c, uint32_t op) {   // (cmd, stage, first, n, const NVNbufferRange *ranges)
    cmd_t *o = cmd_of(c);
    int st = (int)arg_x(c, 1), first = (int)arg_x(c, 2), n = (int)arg_x(c, 3);
    const NVNbufferRange *rg = arg_p(c, 4);
    for (int i = 0; i < n; i++) {
        *REC(o, op, p_bind_buf) = (p_bind_buf){st, first + i, rg[i].address, rg[i].size};
        hw_fix(o, 20);
    }
}
static void bind_buffer(const Cpu *c, uint32_t op) {    // (cmd, stage, index, NVNbufferAddress address, size_t size)
    cmd_t *o = cmd_of(c);
    *REC(o, op, p_bind_buf) = (p_bind_buf){(int)arg_x(c, 1), (int)arg_x(c, 2), arg_x(c, 3), arg_x(c, 4)};
    hw_fix(o, 20);
}
NVN_API(nvnCommandBufferBindUniformBuffer) { bind_buffer(c, OP_BIND_UBO); }
NVN_API(nvnCommandBufferBindUniformBuffers) { bind_buffers(c, OP_BIND_UBO); }
NVN_API(nvnCommandBufferBindStorageBuffer) { bind_buffer(c, OP_BIND_SSBO); }
NVN_API(nvnCommandBufferBindStorageBuffers) { bind_buffers(c, OP_BIND_SSBO); }
// Texture, image and separate handle binds. A deferred bind (NVN 55 *Deferred entry points, the fallback of the inline
// fast paths) keeps the pointer to the game's handle array: the original driver (sdk+0x46cc00 BindTexturesDeferred)
// reads the handles when the command buffer is submitted, and the game fills its material handle tables after
// recording and before the submit (nvn_submit_prepare). hw_bytes: the hardware size of the plain variant.
static void bind_handles(cmd_t *o, uint32_t op, int st, int first, int n, const uint64_t *h, int deferred, size_t hw_bytes) {
    if (n < 0 || n > 32) { rt_log("nvn: bind of %d handles", n); return; }
    p_bind_handles *r = RECN(o, op, p_bind_handles, h, n);
    r->stage = st;
    r->first = first;
    r->n = n;
    r->src = deferred ? (uint64_t)(uintptr_t)h : 0;
    memcpy(r->h, h, 8 * (size_t)n);
    hw_fix(o, hw_bytes ? hw_bytes : 16 + 4 * (size_t)n);   // macro 7 + method 0x1a2c + CB_POS + one word per binding
}
static void bind_handle_array(const Cpu *c, uint32_t op, int deferred, int wide) {   // (cmd, stage, first, n, handles)
    int n = (int)arg_x(c, 3);
    bind_handles(cmd_of(c), op, (int)arg_x(c, 1), (int)arg_x(c, 2), n, arg_p(c, 4), deferred,
                 wide && n >= 0 && n <= 32 ? 16 + 8 * (size_t)n : 0);
}
static void bind_one_handle(const Cpu *c, uint32_t op, size_t hw_bytes) {   // (cmd, stage, index, handle)
    uint64_t v = arg_x(c, 3);
    bind_handles(cmd_of(c), op, (int)arg_x(c, 1), (int)arg_x(c, 2), 1, &v, 0, hw_bytes);
}
// Combined texture handles share the texture slots with separate textures (sampler id in bits 20 and up). Images
// (writable textures): handle = texture pool index; the game does not call RegisterImage, images and textures share
// their registration.
NVN_API(nvnCommandBufferBindTexture) { bind_one_handle(c, OP_BIND_TEX, 0); }
NVN_API(nvnCommandBufferBindTextures) { bind_handle_array(c, OP_BIND_TEX, 0, 0); }
NVN_API(nvnCommandBufferBindImage) { bind_one_handle(c, OP_BIND_IMG, 24); }
NVN_API(nvnCommandBufferBindImages) { bind_handle_array(c, OP_BIND_IMG, 0, 1); }
NVN_API(nvnCommandBufferBindSeparateTexture) { bind_one_handle(c, OP_BIND_TEX, 24); }
NVN_API(nvnCommandBufferBindSeparateTextures) { bind_handle_array(c, OP_BIND_TEX, 0, 1); }
NVN_API(nvnCommandBufferBindSeparateSampler) { bind_one_handle(c, OP_BIND_SAMP, 24); }
NVN_API(nvnCommandBufferBindSeparateSamplers) { bind_handle_array(c, OP_BIND_SAMP, 0, 1); }
// void nvnCommandBufferBind{Textures,Images,SeparateTextures,SeparateSamplers}Deferred(NVNcommandBuffer *cmd,
// NVNshaderStage stage, int first, int n, const T *handles) (NVN 55, no prototype in nvn.h)
NVN_API(nvnCommandBufferBindTexturesDeferred) { bind_handle_array(c, OP_BIND_TEX, 1, 0); }
NVN_API(nvnCommandBufferBindImagesDeferred) { bind_handle_array(c, OP_BIND_IMG, 1, 1); }
NVN_API(nvnCommandBufferBindSeparateTexturesDeferred) { bind_handle_array(c, OP_BIND_TEX, 1, 1); }
NVN_API(nvnCommandBufferBindSeparateSamplersDeferred) { bind_handle_array(c, OP_BIND_SAMP, 1, 1); }
// void nvnCommandBufferBind{Uniform,Storage}BuffersDeferred(NVNcommandBuffer *cmd, NVNshaderStage stage, int first,
// int n, const NVNbufferRange *ranges): buffer ranges still bind at record time
NVN_API(nvnCommandBufferBindUniformBuffersDeferred) {
    static int once;
    if (!once++) NVN_LOG("BindUniformBuffersDeferred: buffer ranges bind at record time");
    bind_buffers(c, OP_BIND_UBO);
}
NVN_API(nvnCommandBufferBindStorageBuffersDeferred) {
    static int once;
    if (!once++) NVN_LOG("BindStorageBuffersDeferred: buffer ranges bind at record time");
    bind_buffers(c, OP_BIND_SSBO);
}
// void nvnCommandBufferSetStencil{ValueMask,Mask,Ref}(NVNcommandBuffer *cmd, NVNface face, int value)
NVN_API(nvnCommandBufferSetStencilValueMask) {
    *REC(cmd_of(c), OP_STENCIL, p_stencil) = (p_stencil){(int)arg_x(c, 1), (int)arg_x(c, 2), 0, 0, 0};
}
NVN_API(nvnCommandBufferSetStencilMask) {
    *REC(cmd_of(c), OP_STENCIL, p_stencil) = (p_stencil){(int)arg_x(c, 1), 0, (int)arg_x(c, 2), 0, 1};
}
NVN_API(nvnCommandBufferSetStencilRef) {
    *REC(cmd_of(c), OP_STENCIL, p_stencil) = (p_stencil){(int)arg_x(c, 1), 0, 0, (int)arg_x(c, 2), 2};
}
// void nvnCommandBufferSetPolygonOffsetClamp(NVNcommandBuffer *cmd, float factor, float units, float clamp)
NVN_API(nvnCommandBufferSetPolygonOffsetClamp) {
    *REC(cmd_of(c), OP_POLY_OFFSET, p_poly_offset) = (p_poly_offset){arg_s(c, 0), arg_s(c, 1), arg_s(c, 2)};
    // NX2NAVI_POLY_OFFSET_LOG: what the game asks for, to tell a decal that relies on the offset from one that does not
    static int logged, on = -1;   // the environment read once: this runs for every offset the game sets
    if (on < 0) on = getenv("NX2NAVI_POLY_OFFSET_LOG") != NULL;
    float f = (float)arg_s(c, 0), u = (float)arg_s(c, 1), cl = (float)arg_s(c, 2);
    if (on && logged < 24 && (f != 0.0f || u != 0.0f)) {
        logged++;
        rt_log("nvn: polygon offset factor %.3f units %.3f clamp %.3f", f, u, cl);
    }
}
// void nvnCommandBufferDrawArrays(NVNcommandBuffer *cmd, NVNdrawPrimitive prim, int first, int count)
NVN_API(nvnCommandBufferDrawArrays) {
    *REC(cmd_of(c), OP_DRAW, p_draw) = (p_draw){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), 0, 1};
}
// void nvnCommandBufferDrawArraysInstanced(NVNcommandBuffer *cmd, NVNdrawPrimitive prim, int first, int count,
// int base_instance, int instances): zero instances draw nothing
NVN_API(nvnCommandBufferDrawArraysInstanced) {
    int inst = (int)arg_x(c, 5);
    if (inst <= 0) return;
    *REC(cmd_of(c), OP_DRAW, p_draw) = (p_draw){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), (int)arg_x(c, 4), inst};
}
// void nvnCommandBufferDrawElements(NVNcommandBuffer *cmd, NVNdrawPrimitive prim, NVNindexType type, int count,
// NVNbufferAddress indices)
NVN_API(nvnCommandBufferDrawElements) {
    cmd_t *o = cmd_of(c);
    *REC(o, OP_DRAW_IDX, p_draw_idx) = (p_draw_idx){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), 0, 0, 1, arg_x(c, 4)};
    hw_fix(o, 48);
}
// void nvnCommandBufferDrawElementsBaseVertex(..., NVNbufferAddress indices, int base_vertex)
NVN_API(nvnCommandBufferDrawElementsBaseVertex) {
    cmd_t *o = cmd_of(c);
    *REC(o, OP_DRAW_IDX, p_draw_idx) =
        (p_draw_idx){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), (int)arg_x(c, 5), 0, 1, arg_x(c, 4)};
    hw_fix(o, 44);
}
// void nvnCommandBufferDrawElementsInstanced(..., NVNbufferAddress indices, int base_vertex, int base_instance,
// int instances): assumed to use the 6-parameter draw macro of the base-vertex form (unverified)
NVN_API(nvnCommandBufferDrawElementsInstanced) {
    cmd_t *o = cmd_of(c);
    *REC(o, OP_DRAW_IDX, p_draw_idx) = (p_draw_idx){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3),
                                                    (int)arg_x(c, 5), (int)arg_x(c, 6), (int)arg_x(c, 7), arg_x(c, 4)};
    hw_fix(o, 44);
}
// void nvnCommandBufferDrawElementsIndirect(NVNcommandBuffer *cmd, NVNdrawPrimitive prim, NVNindexType type,
// NVNbufferAddress indices, NVNbufferAddress indirect): the arguments at indirect are often GPU-written, so the replay
// draws with vkCmdDrawIndexedIndirect from the pool buffer
NVN_API(nvnCommandBufferDrawElementsIndirect) {
    *REC(cmd_of(c), OP_DRAW_IDX_IND, p_draw_idx_ind) = (p_draw_idx_ind){(int)arg_x(c, 1), (int)arg_x(c, 2), arg_x(c, 3), arg_x(c, 4)};
}
// void nvnCommandBufferDispatchCompute(NVNcommandBuffer *cmd, int x, int y, int z): caller = the game's return address
NVN_API(nvnCommandBufferDispatchCompute) {
    *REC(cmd_of(c), OP_DISPATCH, p_dispatch) = (p_dispatch){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), c->x[30]};
}
// void nvnCommandBufferDispatchComputeIndirect(NVNcommandBuffer *cmd, NVNbufferAddress indirect): vkCmdDispatchIndirect
NVN_API(nvnCommandBufferDispatchComputeIndirect) {
    *REC(cmd_of(c), OP_DISPATCH_IND, p_dispatch_ind) = (p_dispatch_ind){arg_x(c, 1), c->x[30]};
}
// NVN2: void nvnCommandBufferDrawMeshTasks(NVNcommandBuffer *cmd, int first, int count): count task (or, without a
// task stage, mesh) workgroups of the bound program from index first (as the game calls it: x1 0, x2 12).
NVN_API(nvnCommandBufferDrawMeshTasks) {
    *REC(cmd_of(c), OP_DRAW_MESH, p_draw_mesh) = (p_draw_mesh){(uint32_t)arg_x(c, 1), (uint32_t)arg_x(c, 2), c->x[30]};
}
// NVN2: nvnCommandBufferDrawMeshTasksIndirect(cmd, NVNbufferAddress indirect, x2, x3, x4): the workgroup count is read
// by the GPU at indirect. The game passes x2 1, x3 the same address, x4 0x4040; taken as one draw.
NVN_API(nvnCommandBufferDrawMeshTasksIndirect) {
    *REC(cmd_of(c), OP_DRAW_MESH_IND, p_draw_mesh_ind) =
        (p_draw_mesh_ind){arg_x(c, 1), arg_x(c, 3), (uint32_t)arg_x(c, 2), (uint32_t)arg_x(c, 4), c->x[30]};
}
// NVN2: nvnCommandBufferSetGeometryProcessingMode(cmd, mode): mode 1 before mesh draws. The replay picks the stages
// from the bound program, so nothing is recorded.
NVN_API(nvnCommandBufferSetGeometryProcessingMode) {
    static int logged;
    if (!logged++) NVN_LOG("SetGeometryProcessingMode(%llu): taken from the bound program", (unsigned long long)arg_x(c, 1));
}
// void nvnCommandBufferClearBuffer(NVNcommandBuffer *cmd, NVNbufferAddress addr, size_t size, uint32_t value)
NVN_API(nvnCommandBufferClearBuffer) {
    *REC(cmd_of(c), OP_CLEAR_BUF, p_clear_buf) = (p_clear_buf){arg_x(c, 1), arg_x(c, 2), (uint32_t)arg_x(c, 3), 0};
}
// void nvnCommandBufferReportValue(NVNcommandBuffer *cmd, uint32_t value, NVNbufferAddress addr): the value lands when
// the GPU reaches this point. Recorded as a 4-byte ClearBuffer, so it stays in GPU order with the ClearBuffer resets of
// the same counter (main+0x7b9274 modes 1-3) and works in CPU-invisible pools; the CPU sees it after the fence, as a
// counter report.
NVN_API(nvnCommandBufferReportValue) {
    *REC(cmd_of(c), OP_CLEAR_BUF, p_clear_buf) = (p_clear_buf){arg_x(c, 2), 4, (uint32_t)arg_x(c, 1), 0};
}
// Colour reduction is a Maxwell bandwidth hint for blending into compressed targets: no effect on the result.
// void nvnCommandBufferSetColorReductionEnable(cmd, NVNboolean enable), SetColorReductionThresholds(cmd, class, const int *)
NVN_API(nvnCommandBufferSetColorReductionEnable) { (void)c; }
NVN_API(nvnCommandBufferSetColorReductionThresholds) { (void)c; }
// void nvnCommandBufferBarrier(NVNcommandBuffer *cmd, int bits)
NVN_API(nvnCommandBufferBarrier) { REC(cmd_of(c), OP_BARRIER, p_barrier)->bits = (int)arg_x(c, 1); }
// void nvnCommandBufferSetTexturePool / SetSamplerPool(NVNcommandBuffer *cmd, const NVNtexturePool / NVNsamplerPool *)
NVN_API(nvnCommandBufferSetTexturePool) { REC(cmd_of(c), OP_SET_TEX_POOL, p_set_pool)->pool = arg_p(c, 1); }
NVN_API(nvnCommandBufferSetSamplerPool) { REC(cmd_of(c), OP_SET_SAMP_POOL, p_set_pool)->pool = arg_p(c, 1); }
// void nvnCommandBufferCopyBufferToTexture(NVNcommandBuffer *cmd, NVNbufferAddress src, const NVNtexture *dst,
// const NVNtextureView *view, const NVNcopyRegion *region, int flags)
NVN_API(nvnCommandBufferCopyBufferToTexture) {
    p_copy_b2t *o = REC(cmd_of(c), OP_COPY_B2T, p_copy_b2t);
    o->src = arg_x(c, 1);
    o->tex = tex_pin(arg_p(c, 2));
    copy_view(&o->view, &o->has_view, arg_p(c, 3));
    o->r = *(const NVNcopyRegion *)arg_p(c, 4);
    o->flags = (int)arg_x(c, 5);
}
// void nvnCommandBufferCopyTextureToBuffer(NVNcommandBuffer *cmd, const NVNtexture *src, const NVNtextureView *view,
// const NVNcopyRegion *region, NVNbufferAddress dst, int flags): p_copy_b2t with src = the destination address
NVN_API(nvnCommandBufferCopyTextureToBuffer) {
    p_copy_b2t *o = REC(cmd_of(c), OP_COPY_T2B, p_copy_b2t);
    o->src = arg_x(c, 4);
    o->tex = tex_pin(arg_p(c, 1));
    copy_view(&o->view, &o->has_view, arg_p(c, 2));
    o->r = *(const NVNcopyRegion *)arg_p(c, 3);
    o->flags = (int)arg_x(c, 5);
}
// void nvnCommandBufferClearTexture{,ui}(NVNcommandBuffer *cmd, const NVNtexture *tex, const NVNtextureView *view,
// const NVNcopyRegion *region, const float / uint32_t *rgba, int mask)
static void clear_texture(const Cpu *c, int is_uint) {
    p_clear_tex *o = REC(cmd_of(c), OP_CLEAR_TEX, p_clear_tex);
    o->tex = tex_pin(arg_p(c, 1));
    copy_view(&o->view, &o->has_view, arg_p(c, 2));
    o->r = *(const NVNcopyRegion *)arg_p(c, 3);
    o->mask = (int)arg_x(c, 5);
    o->is_uint = (uint8_t)is_uint;
    memcpy(o->c.u, arg_p(c, 4), 16);
}
NVN_API(nvnCommandBufferClearTexture) { clear_texture(c, 0); }
NVN_API(nvnCommandBufferClearTextureui) { clear_texture(c, 1); }
// void nvnCommandBufferCopyTextureToTexture(NVNcommandBuffer *cmd, const NVNtexture *src, const NVNtextureView *src_view,
// const NVNcopyRegion *src_region, const NVNtexture *dst, const NVNtextureView *dst_view,
// const NVNcopyRegion *dst_region, int flags)
NVN_API(nvnCommandBufferCopyTextureToTexture) {
    p_copy_t2t *o = REC(cmd_of(c), OP_COPY_T2T, p_copy_t2t);
    o->src = tex_pin(arg_p(c, 1));
    o->dst = tex_pin(arg_p(c, 4));
    copy_view(&o->sv, &o->has_sv, arg_p(c, 2));
    copy_view(&o->dv, &o->has_dv, arg_p(c, 5));
    o->sr = *(const NVNcopyRegion *)arg_p(c, 3);
    o->dr = *(const NVNcopyRegion *)arg_p(c, 6);
    o->flags = (int)arg_x(c, 7);
}
// void nvnCommandBufferDownsample(NVNcommandBuffer *cmd, const NVNtexture *src, const NVNtexture *dst): level 0 of src
// into level 0 of dst, whole extents, filtered. Recorded as a linear-filter texture copy: copy_t2t resolves a
// multisample source into an equal-size target and blits (2x2 average at half size) between sizes
// (docs/maxwell.md 1.19).
NVN_API(nvnCommandBufferDownsample) {
    p_copy_t2t *o = REC(cmd_of(c), OP_COPY_T2T, p_copy_t2t);
    o->src = tex_pin(arg_p(c, 1));
    o->dst = tex_pin(arg_p(c, 2));
    if (o->src) o->sr = (NVNcopyRegion){0, 0, 0, (int)o->src->d.width, (int)o->src->d.height, 1};
    if (o->dst) o->dr = (NVNcopyRegion){0, 0, 0, (int)o->dst->d.width, (int)o->dst->d.height, 1};
    o->flags = NVN_COPY_FLAGS_LINEAR_FILTER_BIT;
}
// void nvnCommandBufferSetCopyRowStride(NVNcommandBuffer *cmd, ptrdiff_t stride)
NVN_API(nvnCommandBufferSetCopyRowStride) { REC(cmd_of(c), OP_COPY_STRIDE, p_copy_stride)->stride = (int64_t)arg_x(c, 1); }
// void nvnCommandBufferSetCopyImageStride(NVNcommandBuffer *cmd, ptrdiff_t stride): the byte distance between the
// 2D slices of a buffer side in 3D / array buffer <-> texture copies. Not replayed yet: the Vulkan copies keep
// tightly packed slices (bufferImageHeight 0). TODO: carry it as bufferImageHeight = stride / row stride.
NVN_API(nvnCommandBufferSetCopyImageStride) {
    static volatile int32_t logged;
    int64_t stride = (int64_t)arg_x(c, 1);
    if (stride && !plat_xchg32(&logged, 1))
        rt_log("nvn: copy image stride %lld not replayed (slices assumed tightly packed)", (long long)stride);
}
// void nvnCommandBufferSetBlendColor(NVNcommandBuffer *cmd, const float *rgba)
NVN_API(nvnCommandBufferSetBlendColor) { memcpy(REC(cmd_of(c), OP_BLEND_COLOR, p_blend_color)->rgba, arg_p(c, 1), 16); }

// Commands without an effect on this replay (hle_nvn.c g_nops): WaitSync and WaitEvent order work against other queues
// or the CPU, and every submit replays in order on one queue; shader scratch memory and the tessellation patch size
// have no host counterpart the replay needs.
// void nvnCommandBufferWaitSync(NVNcommandBuffer *cmd, const NVNsync *sync), WaitEvent(cmd, const NVNevent *event,
// NVNeventWaitMode mode, uint32_t value), SetShaderScratchMemory(cmd, const NVNmemoryPool *pool, ptrdiff_t offset,
// size_t size), SetPatchSize(cmd, int size)
// WaitSync logs its first call: a game that orders two queues with it replays here in submission order only.
NVN_API(nvnCommandBufferWaitSync) {
    static volatile int32_t logged;
    if (!plat_xchg32(&logged, 1)) rt_log("nvn: first CommandBufferWaitSync: recorded as nothing (one replay queue)");
    (void)c;
}
// WaitEvent is recorded (OP_WAIT_EVENT) so that the replay can count the waits its submission order would have
// passed before the signal they wait for (vk_exec_xfer.c wait_event); it waits for nothing.
NVN_API(nvnCommandBufferWaitEvent) {
    const ev_t *e = arg_p(c, 1);
    *REC(cmd_of(c), OP_WAIT_EVENT, p_wait_event) = (p_wait_event){e->val, (uint32_t)arg_x(c, 2), (uint32_t)arg_x(c, 3)};
}
NVN_API(nvnCommandBufferSetShaderScratchMemory) { (void)c; }
NVN_API(nvnCommandBufferSetPatchSize) { (void)c; }

// ---------- submit
// What the console reads at the submit call is read here, on the game thread: every handle must be a live recording,
// and deferred binds take the handles the game's arrays hold now (the game fills its tables for the next frame as soon
// as the submit returns). The result is written into the recording copy.
static uint64_t def_reads, def_changed, n_clones;

static void resolve_handles(p_bind_handles *b) {
    uint64_t cur[32];
    memcpy(cur, (const void *)(uintptr_t)b->src, 8 * (size_t)b->n);
    __atomic_fetch_add(&def_reads, 1, __ATOMIC_RELAXED);
    if (memcmp(cur, b->h, 8 * (size_t)b->n)) {
        if (__atomic_fetch_add(&def_changed, 1, __ATOMIC_RELAXED) < 6)
            NVN_LOG("deferred bind read new handles at submit: stage %d slot %d, %#llx at record -> %#llx",
                    b->stage, b->first, (unsigned long long)b->h[0], (unsigned long long)cur[0]);
        memcpy(b->h, cur, 8 * (size_t)b->n);
    }
}

static int resolve_deferred(const uint8_t *p, int depth) {   // returns the number of deferred binds
    int nd = 0;
    for (;;) {
        op_hdr h;
        memcpy(&h, p, sizeof h);
        if (h.size < 8 || (h.size & 7) || h.op >= OP_COUNT_)
            rt_fatal("nvn: corrupt recording at %p while resolving deferred binds (op %u size %u)", (const void *)p,
                     h.op, h.size);
        if (h.op == OP_END) return nd;
        if (h.op == OP_JUMP) {
            memcpy(&p, p + 8, 8);
            continue;
        }
        if (h.op == OP_CALL) {
            uint64_t t;
            memcpy(&t, p + 8, 8);
            if (depth >= 16) rt_fatal("nvn: deferred bind resolution: call depth over 16");
            nd += resolve_deferred((const uint8_t *)(uintptr_t)t, depth + 1);
        } else if (h.op == OP_BIND_TEX || h.op == OP_BIND_IMG || h.op == OP_BIND_SAMP) {
            p_bind_handles *b = (p_bind_handles *)(uintptr_t)(p + 8);
            if (b->src && b->n > 0 && b->n <= 32) {
                nd++;
                resolve_handles(b);
            }
        }
        p += h.size;
    }
}

static void copy_resolve_deferred(copy_prefix *pf) {
    if (!pf->indexed) {
        int nd = resolve_deferred((const uint8_t *)(pf + 1), 0);
        pf->deferred = nd ? 2 : 1;
        return;
    }
    copy_index_check(pf);
    const size_t *off = copy_offsets(pf) + pf->nuniform;
    for (size_t i = 0; i < pf->ndeferred; i++) {
        p_bind_handles *b = (p_bind_handles *)((uint8_t *)(pf + 1) + off[i] + sizeof(op_hdr));
        resolve_handles(b);
    }
}

void nvn_submit_prepare(const uint64_t *handles, int n) {
    RP_SCOPE(rp_front_prepare, RP_FRONT_PREPARE);
    RP_VALUE(RPK_HANDLES, n);
    uint64_t now = __atomic_add_fetch(&nsubmits, 1, __ATOMIC_RELAXED);
    for (int i = 0; i < n; i++) copy_touch(handles[i], now, "QueueSubmitCommands");
    for (int i = 0; i < n; i++) copy_resolve_deferred((copy_prefix *)(uintptr_t)handles[i] - 1);
}

// Asynchronous replay. A copy a queued submission still reads keeps that submission's deferred binds: when it holds
// deferred binds (or is not known to hold none), this submission gets a clone, resolved here and freed after its replay.
void nvn_submit_prepare_async(const uint64_t *handles, uint64_t *out, int n) {
    uint64_t now = __atomic_add_fetch(&nsubmits, 1, __ATOMIC_RELAXED);
    uint64_t replayed = gfx_replayed_submits();
    copy_prefix *doomed = NULL;
    plat_mutex_lock(&copy_lock);
    if (nzombies) doomed = zombie_sweep(replayed, doomed);
    for (int i = 0; i < n; i++) {
        copy_prefix *pf = copy_find(handles[i]);
        if (!pf) {
            plat_mutex_unlock(&copy_lock);
            rt_fatal("nvn: QueueSubmitCommands: command handle %#llx is not a live recording (released or never recorded)",
                     (unsigned long long)handles[i]);
        }
        out[i] = handles[i];
        if (pf->used_at > replayed && pf->used_at != now && pf->deferred != 1) {
            size_t sz = copy_size(pf);
            copy_prefix *cl = malloc(sz);
            if (!cl) rt_fatal("nvn: out of memory (recording clone, %zu bytes)", sz);
            memcpy(cl, pf, sz);
            cl->plan_readers = 0;
            if (cl->pins) copy_pin_refs(cl, 1);
            cl->magic = 0;   // not a live handle: the game never sees it
            cl->used_at = now;
            zombie_add(cl);
            out[i] = (uintptr_t)(cl + 1);
            if (cl->indexed) nvn_hm_put(&copy_indexed, &out[i], cl);
            __atomic_fetch_add(&n_clones, 1, __ATOMIC_RELAXED);
        } else {
            pf->used_at = now;
        }
    }
    plat_mutex_unlock(&copy_lock);
    doom_free(doomed);
    for (int i = 0; i < n; i++) copy_resolve_deferred((copy_prefix *)(uintptr_t)out[i] - 1);
    if (now % 7200 == 0)
        rt_log("nvn: asynchronous replay at submit %llu: %llu recording clones so far, %u copies waiting for the replay",
               (unsigned long long)now, (unsigned long long)__atomic_load_n(&n_clones, __ATOMIC_RELAXED), nzombies);
}

uint64_t nvn_submit_count(void) { return __atomic_load_n(&nsubmits, __ATOMIC_RELAXED); }

const char *nvn_op_name(uint32_t op) {
    static const char *const names[] = {
        "END", "JUMP", "BEGIN", "CALL", "SIGNAL_EVENT", "REPORT_COUNTER", "COPY_B2B", "UPDATE_UB", "SET_RT",
        "CLEAR_COLOR", "CLEAR_DS", "VIEWPORTS", "SCISSOR", "DEPTH_RANGE", "BIND_PROGRAM", "BIND_STATE", "BIND_VATTR",
        "BIND_VSTREAM", "BIND_VBUF", "BIND_UBO", "BIND_SSBO", "BIND_TEX", "BIND_SAMP", "STENCIL", "POLY_OFFSET", "DRAW",
        "DRAW_IDX", "DISPATCH", "BARRIER", "SET_TEX_POOL", "SET_SAMP_POOL", "COPY_B2T", "COPY_STRIDE", "BLEND_COLOR",
        "COPY_T2B", "BIND_IMG", "COPY_T2T", "RESET_COUNTER", "CLEAR_TEX", "DEPTH_CLAMP", "RENDER_ENABLE",
        "RASTER_DISCARD", "RENDER_COND", "PRIM_RESTART", "DRAW_IDX_IND", "DISPATCH_IND", "CLEAR_BUF", "PUSH_CC", "DRAW_MESH", "DRAW_MESH_IND",
        "WAIT_EVENT",
    };
    _Static_assert(sizeof names / sizeof *names == OP_COUNT_, "one name per op");
    return op < OP_COUNT_ ? names[op] : "?";
}

// NX2NAVI_TRACE=perf "mem: sizes" (perf.c): live recording copies and the zombies waiting for their submissions, texture
// pins with their sweep queue, and the recording-buffer pool.
void nvn_cmd_sizes(PerfSizes *s) {
    plat_mutex_lock_shared(&copy_lock);
    uint64_t nc = ncopies, cb = copy_bytes, nz = nzombies, zb = 0;
    for (uint32_t i = 0; i < nzombies; i++) zb += copy_size(zombies[i]);
    plat_mutex_unlock_shared(&copy_lock);
    plat_mutex_lock(&pin_lock);
    uint64_t pins = pins_live, pq = pin_qn - pin_qh;
    plat_mutex_unlock(&pin_lock);
    plat_mutex_lock(&pool_lock);
    uint64_t np = npool, pb = 0;
    for (uint32_t i = 0; i < npool; i++) pb += pool[i].size;
    plat_mutex_unlock(&pool_lock);
    perf_size(s, "recording copies", nc, 0, cb);
    perf_size(s, "recording copy zombies", nz, 0, zb);
    perf_size(s, "texture pins", pins, 0, PERF_SIZE_NONE);
    perf_size(s, "pin sweep queue", pq, 0, PERF_SIZE_NONE);
    perf_size(s, "recording-buffer pool", np, POOL_SLOTS, pb);
    plat_mutex_lock(&plan_scratch_lock);
    uint64_t plan_count = plan_scratch_count, plan_bytes = plan_scratch_bytes;
    plat_mutex_unlock(&plan_scratch_lock);
    perf_size(s, "command plan scratch pool", plan_count, 8, plan_bytes);
}
